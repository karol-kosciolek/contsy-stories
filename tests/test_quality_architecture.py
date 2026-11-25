# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The quality architecture of series episodes (docs/QUALITY_ARCHITECTURE.md): the series bible as data, the shot
contract and its prompt compiler, the independent shot check on the regression set (tests/fixtures/quality: copies of
pieces of the Lantern Woods episodes, each labelled with its defects A-Q), the episode mix and the sound check, and
the delivery gate. CPU only: the vision model and the video model are stand-ins; nothing reads or writes config/ or
the state folders."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import bible, items, qc, qc_report, scene_parts, scene_sound, shot_qc, shot_spec, timeline, video_gen
from app.shot_spec import ShotSpec
from engine.media import audio, episode_mix, loudness

FIXTURES = Path(__file__).parent / "fixtures" / "quality"
RATE = audio.RATE


def fixture_bible() -> bible.Bible:
    data = json.loads((FIXTURES / "series.json").read_text(encoding="utf-8"))
    refs = FIXTURES / "refs"
    return bible.from_series(data, photos={"lumi": [str(refs / "lumi.png")], "barley": [str(refs / "barley.png")]},
                             plates={"the_clearing": str(refs / "the_clearing.png"),
                                     "lumi_s_burrow": str(refs / "lumi_s_burrow.png")})


def labelled(clip_id: str) -> dict:
    rows = json.loads((FIXTURES / "labels.json").read_text(encoding="utf-8"))["clips"]
    return next(row for row in rows if row["id"] == clip_id)


def spec(**fields) -> ShotSpec:
    base = {"location": "the_clearing", "time_of_day": "morning", "camera": "slow tracking",
            "action": "Lumi walks from the left to the right along the mossy path", "key_visual": ["a small puddle"]}
    return ShotSpec.from_json({**base, **fields})


class BibleTests(unittest.TestCase):
    def setUp(self):
        self.bible = fixture_bible()

    def test_cast_as_data(self):
        lumi, barley = self.bible.character("lumi"), self.bible.character("barley")
        self.assertEqual([(one.name, one.color) for one in lumi.must_have], [("bobble hat", "yellow")])
        self.assertEqual([(one.name, one.color) for one in barley.must_have], [("bow tie", "red")])
        self.assertIn("owl", barley.aliases)  # "the owl" names Barley (defect D)
        self.assertIn("yellow bobble hat", lumi.canonical)
        self.assertIn("front only", lumi.canonical)  # the belly patch never goes on the back (defect C)
        self.assertTrue(self.bible.narrator_only and self.bible.only_named_cast)

    def test_canonical_texts_hold_no_negations(self):
        clearing = self.bible.place("the_clearing")
        self.assertNotIn("No people", clearing.canonical)  # the series.json description ends "No people, no text."
        self.assertEqual(bible.positive("A path, tall trees. No people, no text."), "A path, tall trees.")
        self.assertEqual(bible.positive("A TORTOISE (an animal, not a human): a shell"), "A TORTOISE: a shell")

    def test_body_colour_is_the_first_word(self):
        # "teal-blue": blue is also the sky and the water around Lumi, so only teal identifies him
        self.assertEqual(self.bible.character("lumi").body_ranges, tuple(bible.palette.COLORS["teal"]))


class ShotSpecTests(unittest.TestCase):
    def setUp(self):
        self.bible = fixture_bible()

    def test_negations_are_refused(self):
        problems = shot_spec.validate(spec(action="Lumi walks along the path, no people around"), self.bible)
        self.assertTrue(any("never write what must not be there" in one for one in problems), problems)

    def test_cast_comes_from_names_and_aliases(self):
        one = spec(action="The owl flutters down and lands beside Lumi", cast=["lumi"])
        self.assertEqual(shot_spec.shot_cast(one, self.bible), ("lumi", "barley"))  # defect D
        prompt = shot_spec.compile_prompt(one, self.bible)
        self.assertIn("Barley is a big, round, soft lavender-grey owl", prompt)
        self.assertIn("red bow tie", prompt)

    def test_compiled_prompt_is_deterministic_and_positive(self):
        one = spec(must_not_show=["people", "a boy"])
        prompt = shot_spec.compile_prompt(one, self.bible)
        self.assertEqual(prompt, shot_spec.compile_prompt(spec(must_not_show=["people", "a boy"]), self.bible))
        self.assertEqual(shot_spec.negations(prompt), [])
        self.assertNotIn("boy", prompt)  # must_not_show goes to the check only (defect A)
        self.assertNotRegex(prompt, r"(?i)lip|talks")  # a narrated shot never moves mouths (defect I)
        self.assertTrue(prompt.startswith("The Clearing: A grassy clearing in the Lantern Woods"))
        pattern = items.name_pattern({"name": "The Clearing", "slug": "the_clearing"})
        self.assertTrue(pattern.search(prompt))  # the place's plate becomes a reference of the keyframe

    def test_series_rules_in_code(self):
        problems = shot_spec.validate(spec(action="A boy and a fox walk to Lumi"), self.bible)
        self.assertTrue(any("only its named characters" in one for one in problems), problems)
        self.assertTrue(any("narrated" in one for one in shot_spec.validate(spec(), self.bible, spoken_by=["lumi"])))
        self.assertTrue(shot_spec.validate(spec(location="the moon"), self.bible))
        self.assertTrue(shot_spec.validate(spec(camera="whip pan"), self.bible))
        still = shot_spec.validate(spec(action="Lumi stands and looks at the camera", camera="static medium"),
                                   self.bible)
        self.assertTrue(any("still picture" in one for one in still), still)

    def test_one_picture_pass_and_short_chains(self):
        self.assertTrue(any("one picture pass" in one for one in
                            shot_spec.validate(spec(), self.bible, seconds=7.5, max_seconds=6)))
        first, second = spec(), spec(continuous=True)
        self.assertEqual(shot_spec.validate(second, self.bible, previous=first, chain=1), [])
        self.assertTrue(shot_spec.validate(second, self.bible, previous=first, chain=2))
        moved = spec(continuous=True, location="lumi_s_burrow")
        self.assertTrue(any("stays in its location" in one for one in
                            shot_spec.validate(moved, self.bible, previous=first)))
        self.assertIn("Lumi walks", shot_spec.compile_prompt(second, self.bible))  # compiled on its own fields

    def test_round_trip(self):
        one = spec(props=[], must_show=["Lumi on the path"])
        self.assertEqual(ShotSpec.from_json(one.dumps()), one)


class SceneTests(unittest.TestCase):
    def test_a_scene_without_a_shot_keeps_its_identity(self):
        scene = timeline.Scene("A cat sleeps.", 5)
        self.assertEqual(scene.footage(), ("A cat sleeps.", 5, None, (), (), "", (), ()))
        self.assertNotIn("shot", scene.clip_settings())

    def test_a_shot_scene_round_trips(self):
        one = spec()
        scene = timeline.Scene("compiled", 5, shot=one.dumps())
        again = timeline.Scene.from_json(json.loads(json.dumps(scene.settings())))
        self.assertEqual(again, scene)
        self.assertEqual(again.spec, one)
        self.assertEqual(again.footage()[-1], one.dumps())

    def test_a_narrator_line_never_asks_for_lip_movement(self):
        cast = [{"slug": "lumi", "name": "Lumi", "kind": "creature"}]
        narrated = timeline.Scene("Lumi walks.", 5, cast=("lumi",), lines=(scene_parts.Cue("Once.", "@julia"),))
        project = timeline.Project("p" * 32, 1, "t", (narrated,), style="3d")
        self.assertEqual(timeline.effective(project, narrated, cast)[0], "Lumi walks.")  # defect I
        talking = timeline.Scene("Lumi walks.", 5, cast=("lumi",), lines=(scene_parts.Cue("Hi.", "lumi"),))
        self.assertTrue(timeline.effective(project, talking, cast)[0].endswith("natural lip movement"))
        realistic = timeline.Project("p" * 32, 1, "t", (narrated,), style=timeline.styles.REALISTIC)
        self.assertTrue(timeline.effective(realistic, narrated, cast)[0].endswith("natural lip movement"))

    def test_a_creature_keeps_its_colour_words(self):
        creature = {"slug": "glim", "name": "Glim", "kind": "creature", "look": "a firefly"}
        prompt = video_gen._wan_prompt("Glim hovers with a golden glow", [creature], {}, False, "")
        self.assertIn("golden glow", prompt)  # defect Q: it used to become "skin"
        person = {"slug": "ada", "name": "Ada", "kind": "person", "look": ""}
        self.assertNotIn("golden glow", video_gen._wan_prompt("Ada with a golden glow", [person], {}, False, ""))


class VerdictTests(unittest.TestCase):
    def test_nothing_checked_is_never_a_pass(self):
        record = qc.Verdict(ok=False, status="unchecked").record()
        self.assertEqual((record["ok"], record["status"]), (False, "unchecked"))  # defect O
        self.assertEqual(shot_qc.status_of({"ok": True, "reasons": [], "face": None}), shot_qc.UNCHECKED)
        self.assertEqual(shot_qc.status_of(None), shot_qc.UNCHECKED)
        self.assertEqual(shot_qc.status_of({"ok": False, "reasons": ["face drift"]}), shot_qc.FAIL)
        verdict = shot_qc.QcVerdict(shot_qc.PASS, [shot_qc.Check("backdrop", shot_qc.PASS)], [0.0, 0.5])
        self.assertEqual(shot_qc.QcVerdict.from_record(verdict.record()).status, shot_qc.PASS)

    def test_dense_sampling_with_the_tail(self):
        picked = shot_qc.sample_indices(180, 30)  # 6 s
        self.assertEqual(picked[0], 0)
        self.assertEqual(picked[-1], 179)
        self.assertTrue({0, 15, 30, 165}.issubset(picked))  # every 0.5 s
        tail = [index for index in picked if index >= 179 - 45]
        self.assertTrue({179, 175, 171, 167, 163, 159, 155, 151, 147, 143, 139, 135}.issubset(tail))

    def test_combine(self):
        self.assertEqual(shot_qc.combine([shot_qc.Check("a", shot_qc.PASS), shot_qc.Check("b", shot_qc.FAIL)]),
                         shot_qc.FAIL)
        self.assertEqual(shot_qc.combine([shot_qc.Check("vision", shot_qc.UNCHECKED)]), shot_qc.UNCHECKED)
        self.assertEqual(shot_qc.combine([]), shot_qc.UNCHECKED)


class DetectorTests(unittest.TestCase):
    """The CPU detectors on the regression set: what they must catch and the good shots they must pass."""

    @classmethod
    def setUpClass(cls):
        cls.bible = fixture_bible()
        cls.verdicts = {}
        for clip_id in ("B_02_s08_flat", "E_03_s06_blue", "J_07_s08_crop", "J_07_s10_crop", "good_05_s07",
                        "good_06_s04", "good_06_s05", "good_08_s03", "A_02_s10_boy"):
            row = labelled(clip_id)
            context = shot_qc.Context(cast=tuple(cls.bible.character(slug) for slug in row["cast"]), vision=False)
            cls.verdicts[clip_id] = shot_qc.check_clip(FIXTURES / row["file"], context)

    def failing(self, clip_id):
        return {check.name for check in self.verdicts[clip_id].checks if check.status == shot_qc.FAIL}

    def test_studio_backdrop_and_a_still_shot(self):
        self.assertTrue({"backdrop", "motion"} <= self.failing("B_02_s08_flat"))  # defects B, I

    def test_colour_drift_and_a_lost_hat(self):
        self.assertTrue({"color_drift", "lumi_bobble_hat"} <= self.failing("E_03_s06_blue"))  # defect E

    def test_the_camera_runs_off_the_characters(self):
        self.assertIn("barley_framing", self.failing("J_07_s08_crop"))  # defect J
        self.assertIn("lumi_framing", self.failing("J_07_s10_crop"))

    def test_good_shots_pass(self):
        for clip_id in ("good_05_s07", "good_06_s04", "good_06_s05", "good_08_s03"):
            self.assertEqual(self.failing(clip_id), set(), clip_id)

    def test_without_the_vision_model_a_shot_is_not_passed_by_default(self):
        self.assertEqual(self.verdicts["good_08_s03"].status, shot_qc.PASS)  # vision off by the caller's choice
        row = labelled("good_08_s03")
        context = shot_qc.Context(cast=(self.bible.character("lumi"),), vision=True,
                                  work=Path(tempfile.mkdtemp()))
        with patch.object(shot_qc, "ask_vision", side_effect=RuntimeError("no model")):
            verdict = shot_qc.check_clip(FIXTURES / row["file"], context)
        self.assertEqual(verdict.status, shot_qc.UNCHECKED)

    def test_the_tail_is_zoomed_for_the_vision_model(self):
        row = labelled("A_02_s10_boy")
        frames, _ = shot_qc.read_frames(FIXTURES / row["file"])
        zooms = shot_qc.hotspots(frames, ())
        self.assertTrue(zooms)
        self.assertTrue(all(label.startswith("ZOOM") for label, _ in zooms))

    def test_vision_answers_become_binding_checks(self):
        row = labelled("good_08_s03")
        lumi = self.bible.character("lumi")
        context = shot_qc.Context(cast=(lumi,), spec=spec(key_visual=["soap bubbles"]), bible=self.bible,
                                  work=Path(tempfile.mkdtemp()))
        asked = shot_qc.questions(context.spec, context.cast, self.bible)
        answers = {one.id: ("A", "") for one in asked}
        answers["only_cast"] = ("B", "a boy at 7.4 s")
        with patch.object(shot_qc, "ask_vision", return_value=answers) as ask:
            verdict = shot_qc.check_clip(FIXTURES / row["file"], context)
        self.assertEqual(verdict.status, shot_qc.FAIL)
        self.assertIn("vision_only_cast", {check.name for check in verdict.checks if check.status == shot_qc.FAIL})
        self.assertTrue(ask.call_args.kwargs["extra"])  # the character crops beside the reference
        ids = {one.id for one in asked}
        self.assertTrue({"only_cast", "lumi_identity", "lumi_bobble_hat", "shows_1", "stays_1", "physics", "stable",
                         "action"} <= ids)

    def test_an_unsure_answer_goes_to_the_arbiter(self):
        row = labelled("good_08_s03")
        calls = []

        def arbiter(sheet, asked, local, extra=()):
            calls.append([one.id for one in asked])
            return {one.id: ("A", "(Claude) clear") for one in asked}

        context = shot_qc.Context(cast=(self.bible.character("lumi"),), bible=self.bible, arbiter=arbiter,
                                  work=Path(tempfile.mkdtemp()))
        asked = shot_qc.questions(None, context.cast, self.bible)
        answers = {one.id: ("A", "") for one in asked}
        answers["physics"] = ("unsure", "?")
        with patch.object(shot_qc, "ask_vision", return_value=answers):
            verdict = shot_qc.check_clip(FIXTURES / row["file"], context)
        self.assertEqual(calls, [["physics"]])
        self.assertEqual(verdict.status, shot_qc.PASS)

    def test_a_title_over_a_face_is_measured(self):
        row = labelled("K_01_s01_title")
        frames, _ = shot_qc.read_frames(FIXTURES / row["file"])
        lumi = (self.bible.character("lumi"),)
        early = [frame for frame in frames if frame.seconds < 3.0]
        center = shot_qc.text_box_overlap(early, lumi, qc_report.TITLE_BANDS["center"])
        top = shot_qc.text_box_overlap(early, lumi, qc_report.TITLE_BANDS["top"])
        self.assertGreater(center, qc_report.TITLE_COVER)  # defect K: the title sat on Lumi's face
        self.assertLess(top, center)


class SoundTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())

    def stem(self, name, samples):
        path = self.folder / f"{name}.wav"
        audio.write_wav(path, samples)
        return path

    def tone(self, seconds, frequency, level, start=0.0, length=None):
        out = np.zeros((2, int(round(seconds * RATE))), dtype=np.float32)
        a, b = int(start * RATE), int((start + (length if length is not None else seconds - start)) * RATE)
        t = np.arange(b - a) / RATE
        out[:, a:b] = level * np.sin(2 * np.pi * frequency * t)
        return out

    def test_meter(self):
        sine = self.tone(5, 997, 0.1)
        self.assertAlmostEqual(loudness.integrated(sine, RATE), -20.0, delta=0.05)
        self.assertAlmostEqual(loudness.integrated(sine[:1], RATE), -23.01, delta=0.05)
        self.assertLess(loudness.true_peak(sine, RATE), -19.9)
        self.assertEqual(loudness.silences(np.zeros((2, RATE)), RATE), [(0.0, 1.0)])

    def test_the_old_films_were_too_quiet(self):
        samples, _ = audio.read(FIXTURES / "clips" / "audio_03_0_40.m4a", rate=RATE)
        self.assertLess(loudness.integrated(samples, RATE), -16.0)  # defect N: about -19 LUFS

    def test_episode_mix(self):
        handle = audio.TIMELINE_AUDIO_HANDLE
        scenes = []
        for index, (place, voice_from) in enumerate((("woods", 0.3), ("woods", 0.3), ("home", 0.3))):
            noise = np.random.default_rng(index).standard_normal((2, int((5 + 2 * handle) * RATE))).astype(np.float32)
            bed = self.stem(f"bed{index}", noise * (0.02 if index != 2 else 0.0))  # the last place's bed is silent
            voice = self.stem(f"voice{index}", self.tone(5, 220 + 40 * index, 0.3, voice_from, 3.5))
            scenes.append(episode_mix.SceneSound(bed, voice, 5.0, location=place))
        report = episode_mix.MixReport()
        pieces = episode_mix.episode_soundtrack(scenes, report=report)
        self.assertEqual([piece.shape[1] for piece in pieces], [5 * RATE] * 3)
        mix = np.concatenate(pieces, axis=1)
        self.assertAlmostEqual(loudness.integrated(mix, RATE), episode_mix.TARGET_LUFS, delta=0.5)  # defect N
        self.assertLessEqual(loudness.true_peak(mix, RATE), episode_mix.TARGET_TP + 0.05)
        self.assertEqual(loudness.silences(mix, RATE), [])  # defect M: never dead air, even over a silent bed
        self.assertGreaterEqual(report.bed_under_voice, qc_report.MIN_BED_UNDER_VOICE)
        self.assertEqual([where for where, _, _ in report.locations], ["woods", "home"])  # one bed per place
        self.assertEqual(report.room_tone, ["home"])
        self.assertTrue(report.j_cuts)  # a line starts a little before its picture where there is room (defect L)
        self.assertIn("music library is empty", " ".join(report.notes))

    def test_narration_pace(self):
        self.assertEqual(scene_sound.tempo_for("one two three", 3.0, 135), 1.0)
        line = "After a night of rain, the Lantern Woods sparkled, and little Lumi went out for a morning walk."
        self.assertAlmostEqual(scene_sound.tempo_for(line, 7.5, 135), 135 / (18 / 7.5 * 60), places=3)
        self.assertEqual(scene_sound.tempo_for(line, 5.0, 135), scene_parts.SLOWEST_TEMPO)
        cue = scene_parts.Cue("Hello.", "@julia", None, scene_parts.NARRATOR_PACE)
        self.assertEqual(scene_parts.Cue.from_json(cue.to_json()), cue)
        self.assertNotIn("pace", scene_parts.Cue("Hello.").to_json())
        with self.assertRaises(ValueError):
            scene_parts.Cue("Hello.", pace=10)

    def test_sound_checks_of_a_film(self):
        film = self.folder / "film.wav"
        loud = self.tone(10, 300, 0.5)
        audio.write_wav(film, loud)
        checks, measured = qc_report.audio_checks(film, {"bed_under_voice": 4.0}, [], [])
        failing = {check.name for check in checks if check.status == shot_qc.FAIL}
        self.assertIn("ambience_under_voice", failing)  # defect M: the ambience only 4 dB under the narrator
        self.assertIn("loudness", failing)
        quiet = self.tone(10, 300, 0.2, 0.0, 4.0)
        audio.write_wav(film, quiet)
        checks, _ = qc_report.audio_checks(film, None, [], [])
        self.assertIn("dead_air", {check.name for check in checks if check.status == shot_qc.FAIL})


class GateTests(unittest.TestCase):
    def test_a_shot_that_did_not_pass_blocks_delivery(self):
        from app import agent_delivery

        project = timeline.Project("a" * 32, 3, "Episode", (timeline.Scene("x", 5), timeline.Scene("y", 5)))
        records = {"k1": {"qc": shot_qc.QcVerdict(shot_qc.PASS).record()},
                   "k2": {"qc": {"ok": True, "reasons": [], "face": None}}}  # an old clip: unchecked
        with patch.object(timeline, "prepared", return_value={"assets": ["k1", "k2"]}), \
                patch.object(timeline, "asset", side_effect=lambda project, key, verify=False: records.get(key)), \
                patch.object(timeline, "exported", return_value=None):
            failing = agent_delivery.quality_failures(project)
        self.assertEqual([code for code, _ in failing], ["G8"])
        self.assertIn("scene 2 unchecked", failing[0][1])


class ClipRetryTests(unittest.TestCase):
    """_build_clip checks a checked scene's clip and makes it again with a new seed when the check fails."""

    def test_retry_then_needs_review(self):
        from tests.test_timeline_stage4 import Stage4Case

        case = Stage4Case()
        case.setUp()
        self.addCleanup(case.doCleanups)
        project = case.save([timeline.Scene("Ada waits.", 3, cut=True)])
        failed = shot_qc.QcVerdict(shot_qc.FAIL, [shot_qc.Check("backdrop", shot_qc.FAIL, "flat", "B")])
        passed = shot_qc.QcVerdict(shot_qc.PASS, [shot_qc.Check("backdrop", shot_qc.PASS)])
        with patch.object(timeline.shot_pipeline, "enabled", return_value=True), \
                patch.object(timeline.shot_pipeline, "keyframe_checker", return_value=lambda path: ""), \
                patch.object(timeline.shot_pipeline, "check_clip", side_effect=[failed, passed]) as check:
            case.prepare(project)
        self.assertEqual(check.call_count, 2)
        seeds = [call.get("seed") for call in case.video.calls]
        self.assertEqual(len(seeds), 2)
        self.assertNotEqual(seeds[0], seeds[1])
        key = timeline.prepared(project)["assets"][0]
        self.assertEqual(timeline.asset(project, key)["qc"]["status"], shot_qc.PASS)

        project = case.save([timeline.Scene("Ada waits again.", 3, cut=True)], project)
        with patch.object(timeline.shot_pipeline, "enabled", return_value=True), \
                patch.object(timeline.shot_pipeline, "keyframe_checker", return_value=lambda path: ""), \
                patch.object(timeline.shot_pipeline, "check_clip", side_effect=[failed] * 3):
            case.prepare(project)
        key = timeline.prepared(project)["assets"][0]
        self.assertEqual(timeline.asset(project, key)["qc"]["status"], shot_qc.NEEDS_REVIEW)


if __name__ == "__main__":
    unittest.main()


class SaveProjectShotTests(unittest.TestCase):
    """save_project in a series episode: shots are required, validated and compiled by the studio."""

    def compile(self, rows):
        from app import agent_tools_film, agent_tools_series

        with patch.object(agent_tools_series, "bound", return_value={"series": "s1", "episode": "e1"}), \
                patch.object(agent_tools_film.bible_mod, "for_series", return_value=fixture_bible()), \
                patch.object(video_gen, "part_seconds_limit", return_value=6):
            return agent_tools_film.compile_shots(None, rows)

    def test_a_series_scene_needs_a_shot(self):
        from app.agent_host import ToolError

        with self.assertRaises(ToolError) as caught:
            self.compile([{"prompt": "Lumi walks, no people.", "seconds": 5}])
        self.assertIn("needs a shot", str(caught.exception))

    def test_problems_are_listed_by_scene(self):
        from app.agent_host import ToolError

        rows = [{"seconds": 7.5, "shot": json.loads(spec().dumps())},
                {"seconds": 5, "shot": json.loads(spec(action="Lumi walks, no boy").dumps())},
                {"seconds": 5, "shot": json.loads(spec().dumps()), "lines": [{"text": "Hi.", "speaker": "lumi"}]}]
        with self.assertRaises(ToolError) as caught:
            self.compile(rows)
        text = str(caught.exception)
        self.assertIn("scene 1: The shot lasts 7.5 s", text)
        self.assertIn("scene 2: shot.action says", text)
        self.assertIn("scene 3: The series is narrated", text)

    def test_compiled_scene(self):
        from app import agent_tools_film

        row = {"seconds": 5, "shot": json.loads(spec(action="The owl lands beside Lumi").dumps()),
               "lines": [{"text": "Barley came to help.", "speaker": "@julia"}]}
        shots = self.compile([row])
        scene = agent_tools_film.make_scene(None, row, set(), shots[0])
        self.assertEqual(scene.cast, ("lumi", "barley"))
        self.assertTrue(scene.cut)
        self.assertEqual(scene.spec.action, "The owl lands beside Lumi")
        self.assertIn("Barley is", scene.prompt)
        self.assertEqual(scene.lines[0].pace, scene_parts.NARRATOR_PACE)


class FollowUpRepairTests(unittest.TestCase):
    def test_a_hard_stopped_run_gets_its_open_calls_answered_before_a_follow_up(self):
        from app import agent_runner, agent_store

        with tempfile.TemporaryDirectory() as folder, patch.object(agent_store, "RUNS_DIR", Path(folder)):
            run_id = agent_store.new_run_id()
            agent_store.run_dir(run_id).mkdir(parents=True)
            store = agent_store.RunStore(run_id)
            store.append_message("user", [{"type": "text", "text": "Make the episode."}])
            store.append_message("assistant", [{"type": "tool_use", "id": "toolu_1", "name": "wait_for_jobs",
                                                "input": {}}])
            agent_runner._answer_open_calls(store)
            last = store.records()[-1]
            self.assertEqual(last["role"], "user")
            self.assertEqual(last["content"][0]["tool_use_id"], "toolu_1")
            agent_runner._answer_open_calls(store)  # nothing open any more
            self.assertEqual(len(store.records()), 3)


class CompositionWordingTests(unittest.TestCase):
    def test_a_shot_without_characters_names_no_person(self):
        from app import refs

        empty = refs.edit_prompt("The Clearing: a leaf drifts down", [1], False, False, places_only=True)
        self.assertNotRegex(empty, r"(?i)\bperson|people")  # defect A: "the exact person" drew a girl
        self.assertIn("the exact place and things shown in image 1", empty)
        self.assertIn("person", refs.edit_prompt("A walk", [1], False, False))  # other shots keep their wording
        from app import characters

        self.assertNotRegex(characters._CREATURE_LEAD, r"(?i)scarf|glasses|bow tie")  # examples are drawn


class EmptyShotTests(unittest.TestCase):
    def test_a_shot_without_characters_never_mentions_them(self):
        empty = spec(action="A big green leaf drifts down from the tall trees", camera="slow pan right")
        self.assertEqual(shot_spec.shot_cast(empty, fixture_bible()), ())
        prompt = shot_spec.compile_prompt(empty, fixture_bible())
        self.assertNotRegex(prompt, r"(?i)character")  # "following the characters" drew two children
        self.assertIn("following the subject", prompt)
