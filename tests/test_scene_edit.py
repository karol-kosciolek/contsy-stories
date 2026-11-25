# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""AI edits of made Timeline scenes (timeline.edit, edit_job, the takes of a scene's other versions, retry_edit), the
film's places and items (Project.items, the bin's Items tab) and the inspector's Edit tab. Temporary project folders
only: the GPU queue, the picture generator, the image editor and the language model are stood in for."""
from contextlib import ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
import numpy as np
from PIL import Image

from app import config, dialogue, history, items, llm, queue_manager, timeline, ui_timeline, video_gen
from app.scene_parts import Cue
from app.ui_timeline import Draft
from engine.media import audio, video
from tests.test_ui_timeline import _Isolated

SKIP = gr.skip()


def _film(path, rgb, seconds, *, last_rgb=None, tone=None):
    """A 48x32 film at 30 FPS of `seconds` (one colour, the last frame `last_rgb`), with a tone when `tone`."""
    track = None
    if tone:
        t = np.arange(int(seconds * audio.RATE)) / audio.RATE
        mono = (0.2 * np.sin(2 * np.pi * tone * t)).astype(np.float32)
        track = (np.stack([mono, mono]), audio.RATE)
    frame = np.full((32, 48, 3), rgb, dtype=np.uint8)
    last = frame if last_rgb is None else np.full_like(frame, last_rgb)
    with video.Writer(path, 30, 48, 32, preset="ultrafast", comment=video.AI_COMMENT, audio=track) as out:
        for _ in range(int(30 * seconds) - 1):
            out.write(frame)
        out.write(last)


def _wav(path, seconds, tone):
    t = np.arange(int(seconds * audio.RATE)) / audio.RATE
    wave = (0.2 * np.sin(2 * np.pi * tone * t)).astype(np.float32)
    audio.write_wav(path, np.stack([wave, wave]), audio.RATE)
    return Path(path)


def _colour(path) -> tuple:
    with Image.open(path) as picture:
        return tuple(int(value) for value in np.asarray(picture.convert("RGB")).reshape(-1, 3).mean(axis=0))


class _Project(unittest.TestCase):
    """A temporary project folder; queued jobs are only collected (self.added) and every clip comes from _generate,
    a 48x32 film whose colour tells the attempt."""

    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.root = Path(self.context.enter_context(tempfile.TemporaryDirectory())).resolve()
        enter = self.context.enter_context
        enter(patch.object(timeline, "DIR", self.root / "projects"))
        enter(patch("app.items.DIR", self.root / "items"))
        enter(patch.object(config, "OUTPUTS_DIR", self.root / "outputs"))
        enter(patch.object(config, "HISTORY_DIR", self.root / "history"))
        enter(patch.object(history, "_HISTORY_FILE", self.root / "history/history.jsonl"))
        enter(patch.object(timeline.model_manager, "ensure", return_value=None))
        enter(patch.object(timeline.sound, "available", return_value=True))
        enter(patch.object(timeline.tts, "available", return_value=True))
        enter(patch.object(timeline.model_manager, "VERIFIED_FILE", self.root / "verified_models.json"))
        enter(patch.object(timeline.model_manager, "_STORED", None))
        self.running = []  # jobs the queue shows as waiting or running
        self.added = []
        enter(patch.object(queue_manager, "snapshot", side_effect=lambda: list(self.running)))
        enter(patch.object(queue_manager, "add", side_effect=self._add))
        self.calls = []
        enter(patch.object(timeline.video_gen, "generate", side_effect=self._generate))
        enter(patch.object(timeline.video_gen, "redub", side_effect=AssertionError("No redub expected")))
        enter(patch.object(timeline.llm, "inspect_review", side_effect=AssertionError("No visual review expected")))
        enter(patch.object(timeline.refs, "compose", side_effect=AssertionError("No picture change expected")))
        enter(patch.object(timeline.engine_runtime, "engine", return_value=object()))

    def _add(self, kind, label, params, files=None):
        job = queue_manager.Job(len(self.added) + 1, kind, label, json.loads(json.dumps(params)))
        self.added.append(job)
        return job

    def _generate(self, prompt, seconds, aspect, *, scene_dir, image_path=None, ref_images=None, sound_on=False,
                  **options):
        scene_dir.mkdir(parents=True, exist_ok=True)
        number = len(self.calls) + 1
        self.calls.append({"prompt": prompt, "start": _colour(image_path) if image_path else None,
                           "refs": list(ref_images or ()), "seconds": seconds})
        path = scene_dir / f"scene{number}.mp4"
        _film(path, (20 * number, 100, 60), math.ceil(seconds), last_rgb=(10, 30 * number, 200))
        return video_gen.VideoOutput(True, [str(path)], info={"qc": {"ok": True}, "width": 48, "height": 32})

    def save(self, scenes, previous=None, *, audio_on=False, style=None):
        return timeline.save(previous.id if previous else None, previous.revision if previous else None,
                             "Coffee film", tuple(scenes), "Landscape 16:9", audio_on, [], style=style)

    def made(self, project, generate_scene=None):
        """Every scene's clip made (prepare_job, with `generate_scene` as its renderer when given), the saved project
        back."""
        ok, _, notes, error = timeline.prepare_job({"project": project.id, "key": project.key}, lambda *_: None,
                                                   generate_scene=generate_scene)
        self.assertTrue(ok, error)
        return timeline.load(project.id)

    def keys(self, project):
        return list(timeline.prepared(project)["assets"])

    def run_edit(self, job):
        ok, _, notes, error = timeline.edit_job(job.params, lambda *_: None)
        self.assertTrue(ok, error)
        return notes


class EditTests(_Project):
    def test_an_edit_makes_a_new_take_from_the_first_frame_and_the_earlier_version_comes_back(self):
        film = self.made(self.save([timeline.Scene("A red cup on a table", 1), timeline.Scene("The cup slides", 1)]))
        first, second = self.keys(film)
        source_start = Path(self.root / "start.png")
        timeline.ffmpeg_utils.frame_at(timeline.asset(film, first)["path"], 0.0, source_start)
        blue = replace(film.scenes[0], prompt="A blue cup on a table")
        edited, job = timeline.edit(film, 0, blue)
        self.assertEqual((job.kind, job.params["scope"], job.params["sources"]), (timeline.EDIT, "scene", {"0": first}))
        self.assertEqual(edited.scenes[0].prompt, "A blue cup on a table")
        self.assertEqual(edited.revision, film.revision + 1)
        self.assertEqual(timeline.edit_source(edited, 0), first)  # its clip is not made yet: the take it starts from
        self.assertEqual(self.run_edit(job)[0], "Scene 1 edited: 1 scene made.")
        made = self.calls[-1]
        self.assertTrue(made["prompt"].startswith(timeline._CONTINUE))
        self.assertEqual(made["refs"], [])  # nothing is composed into the take's own picture again
        self.assertEqual(made["start"], _colour(source_start))  # it starts from the take's first frame
        new = self.keys(edited)[0]
        self.assertNotIn(new, (first, None))
        record = timeline.asset(edited, new)
        self.assertEqual((record["edit"]["kind"], record["edit"]["from"]), ("animate", first))
        self.assertEqual(timeline.outdated(edited), [1])  # scene 2 continues from the old last frame

        listed = timeline.takes(edited, 0)
        self.assertEqual([(take["key"], take["current"]) for take in listed], [(new, True), (first, False)])
        self.assertTrue(listed[1]["label"].startswith("Other version"))
        self.assertEqual(listed[1]["settings"]["prompt"], "A red cup on a table")
        back = timeline.use_take(edited, 0, first)
        self.assertEqual(back.scenes[0].prompt, "A red cup on a table")
        self.assertEqual(self.keys(back)[:2], [first, second])  # the film as it was, scene 2 included
        self.assertEqual(timeline.outdated(back), [])
        again = timeline.takes(back, 0)
        self.assertEqual([(take["key"], take.get("settings", {}).get("prompt")) for take in again],
                         [(first, None), (new, "A blue cup on a table")])
        forward = timeline.use_take(back, 0, new)
        self.assertEqual((forward.scenes[0].prompt, self.keys(forward)[0]), ("A blue cup on a table", new))

    def test_an_edit_carried_on_remakes_the_scenes_that_continue_from_it(self):
        film = self.made(self.save([timeline.Scene("A red cup", 1), timeline.Scene("The cup slides", 1),
                                    timeline.Scene("A street", 1, cut=True), timeline.Scene("A car passes", 1)]))
        old = self.keys(film)
        edited, job = timeline.edit(film, 0, replace(film.scenes[0], prompt="A blue cup"), scope="after")
        self.assertEqual(job.params["seconds"], 2)  # the cut ends the run: scenes 3 and 4 stay
        before = len(self.calls)
        self.run_edit(job)
        new = self.keys(edited)
        self.assertEqual(len(self.calls), before + 2)
        self.assertEqual(new[2:], old[2:])
        self.assertNotIn(new[1], (old[1], None))
        last = Path(self.root / "last.png")
        timeline.ffmpeg_utils.last_frame(timeline.asset(edited, new[0])["path"], last)
        self.assertEqual(self.calls[-1]["start"], _colour(last))  # scene 2 from the new last frame of scene 1
        self.assertEqual(timeline.outdated(edited), [])
        self.assertEqual(timeline.takes(edited, 1)[1]["key"], old[1])  # scene 2's earlier take stays a take

    def test_a_picture_change_through_the_film_changes_every_new_shot_and_its_shown_text(self):
        film = self.made(self.save([
            timeline.Scene('A large screen displays the text "CRYPTO NEWS" behind a dog', 1),
            timeline.Scene("The dog turns its head", 1),
            timeline.Scene('A wide shot of the studio; the screen displays the text "CRYPTO NEWS"', 1, cut=True)]))
        old = self.keys(film)
        changes = []

        def compose(client, images, prompt, width, height, out_path, seed=None, progress_cb=None, style=None):
            changes.append((_colour(images[0]), prompt, style))
            Image.new("RGB", (64, 36), (250, 0, 250)).save(out_path)
            return True, str(out_path)

        scene = replace(film.scenes[0], prompt=film.scenes[0].prompt.replace("CRYPTO NEWS", "BREAKING NEWS"))
        with patch.object(timeline.refs, "compose", side_effect=compose):
            edited, job = timeline.edit(film, 0, scene, change='Change the screen text to "BREAKING NEWS"',
                                        scope="film")
            self.assertIn('"BREAKING NEWS"', edited.scenes[2].prompt)  # the shown text, in the other shot too
            self.assertEqual(job.params["changed"], [0, 2])
            before = len(self.calls)
            self.run_edit(job)
        self.assertEqual(len(changes), 2)  # the first frames of both shots
        self.assertTrue(changes[0][1].startswith('Change the screen text to "BREAKING NEWS". Keep everything else'))
        made = self.calls[before:]
        self.assertEqual([call["start"] for call in made][0], (250, 0, 250))
        self.assertEqual(made[2]["start"], (250, 0, 250))
        new = self.keys(edited)
        self.assertTrue(all(key and key not in old for key in new))
        self.assertEqual(timeline.outdated(edited), [])
        record = timeline.asset(edited, new[2])
        self.assertEqual((record["edit"]["from"], record["edit"]["change"]),
                         (old[2], 'Change the screen text to "BREAKING NEWS"'))
        self.assertEqual(timeline.takes(edited, 2)[1]["settings"]["prompt"], film.scenes[2].prompt)

    def test_steps_of_each_scope(self):
        scenes = (timeline.Scene("A", 1), timeline.Scene("B", 1), timeline.Scene("C", 1, cut=True),
                  timeline.Scene("D", 1), timeline.Scene("E", 1))
        project = timeline.Project("0" * 32, 1, "Steps", scenes)
        every = {number: "f" * 64 for number in range(5)}
        steps = timeline._edit_steps
        self.assertEqual(steps(project, 1, "scene", "", [], every), [(1, "edit")])
        self.assertEqual(steps(project, 1, "after", "", [], every), [(1, "edit")])
        self.assertEqual(steps(project, 0, "after", "", [], every), [(0, "edit"), (1, "follow")])
        self.assertEqual(steps(project, 1, "after", "Make it night", [], every),
                         [(1, "edit"), (2, "edit"), (3, "follow"), (4, "follow")])
        self.assertEqual(steps(project, 3, "film", "Make it night", [], every),
                         [(0, "edit"), (1, "follow"), (2, "edit"), (3, "follow"), (4, "follow")])
        self.assertEqual(steps(project, 0, "after", "", [], {0: "f" * 64, 2: "f" * 64}), [(0, "edit")])
        self.assertEqual(steps(project, 0, "film", "", [3], every),
                         [(0, "edit"), (1, "follow"), (3, "edit"), (4, "follow")])

    def test_refusals_keep_the_project_as_it_was(self):
        film = self.save([timeline.Scene("A red cup", 1), timeline.Scene("Steam rises", 1)])
        with self.assertRaisesRegex(ValueError, "Generate scene 1 first"):
            timeline.edit(film, 0, replace(film.scenes[0], prompt="A blue cup"))
        film = self.made(film)
        with self.assertRaisesRegex(ValueError, "nothing to change"):
            timeline.edit(film, 0, film.scenes[0])
        with self.assertRaisesRegex(ValueError, "which scenes"):
            timeline.edit(film, 0, film.scenes[0], change="Make it night", scope="everything")
        with self.assertRaisesRegex(ValueError, "under 500"):
            timeline.edit(film, 0, film.scenes[0], change="x" * 501)
        with self.assertRaisesRegex(ValueError, "Blocked"):
            timeline.edit(film, 0, film.scenes[0], change="make her a naked child")
        self.running.append(queue_manager.Job(9, timeline.PREPARE, "Generate", {"project": film.id}))
        with self.assertRaisesRegex(ValueError, "already has a job"):
            timeline.edit(film, 0, film.scenes[0], change="Make it night")
        self.running.clear()
        timeline.save(film.id, film.revision, film.title, (replace(film.scenes[0], seconds=2), film.scenes[1]),
                      film.aspect, film.audio, [])
        with self.assertRaisesRegex(ValueError, "timeline changed"):
            timeline.edit(film, 0, film.scenes[0], change="Make it night")
        self.assertEqual(self.added, [])

    def test_a_failed_edit_is_retried_with_a_new_take_and_keeps_its_earlier_version(self):
        film = self.made(self.save([timeline.Scene("A red cup", 1)]))
        first = self.keys(film)[0]
        edited, job = timeline.edit(film, 0, replace(film.scenes[0], prompt="A blue cup"))
        with patch.object(timeline.video_gen, "generate",
                          return_value=video_gen.VideoOutput(False, error="The picture broke.")):
            ok, _, _, error = timeline.edit_job(job.params, lambda *_: None)
        self.assertFalse(ok)
        self.assertEqual(error, "Scene 1: The picture broke.")
        failed = timeline.failure(edited)
        self.assertEqual((failed["kind"], failed["scene"]), (timeline.EDIT, 0))
        self.assertEqual(timeline.edit_source(edited, 0), first)
        self.assertEqual([take["key"] for take in timeline.takes(edited, 0)], [first])  # the way back
        again = timeline.retry_edit(edited)
        self.assertEqual(again.params["take"], job.params["take"])
        self.assertNotEqual(again.params["retakes"]["0"], job.params["take"])
        self.run_edit(again)
        self.assertIsNone(timeline.failure(edited))
        with self.assertRaisesRegex(ValueError, "no failed edit"):
            timeline.retry_edit(edited)

    def test_a_paused_edit_finds_the_takes_it_made(self):
        film = self.made(self.save([timeline.Scene("A red cup", 1), timeline.Scene("It slides", 1)]))
        edited, job = timeline.edit(film, 0, replace(film.scenes[0], prompt="A blue cup"), scope="after")
        self.run_edit(job)
        made = self.keys(edited)
        before = len(self.calls)
        self.run_edit(job)  # the same job again (a pause sends it back to the queue)
        self.assertEqual((len(self.calls), self.keys(edited)), (before, made))

    @staticmethod
    def _talking(text, *, skipped=""):
        """A renderer (prepare_job's generate_scene) of a 1 s talking take with its background and voice stems."""
        def render(project, scene, members, seed, style, folder, cb, anchor=None):
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "talk.mp4"
            _film(path, (90, 90, 90), 1, tone=300)
            stems = _wav(folder / "bg.wav", 1.5, 300), _wav(folder / "voice.wav", 1, 500)
            info = {"qc": {"ok": True}, "width": 48, "height": 32, "voices": ["Hanna"],
                    "dialogue_timings": [{"text": text, "character": "", "voice": "hanna", "start": 0.3, "end": 0.8}]}
            if skipped:
                info["lipsync_skipped"] = skipped
            return video_gen.VideoOutput(True, [str(path)], audio_stems=stems, info=info)
        return render

    def _speaking(self):
        """Its lines said by a library voice, with no voice library read."""
        enter = self.context.enter_context
        speaker = timeline.scene_sound.Speaker("female", "hanna", "", "female#hanna", "Hanna")
        enter(patch.object(timeline.scene_sound, "resolve_speakers", side_effect=lambda cues, *_: [speaker] * len(cues)))
        enter(patch.object(timeline, "_audition", return_value=None))
        enter(patch.object(timeline, "_check_issues"))
        enter(patch.object(timeline.voices, "get", return_value=None))
        return speaker

    def test_a_dialogue_edit_of_a_picture_without_mouth_movement_only_makes_the_sound_again(self):
        speaker = self._speaking()
        sounds = []

        def scene_sound(project, index, scene, members, cb, speech=None):
            folder = self.root / f"sound{len(sounds)}"
            folder.mkdir()
            sounds.append(scene.lines[0].text)
            return timeline.scene_sound.SceneSound(
                folder, "a" * 64, scene.seconds, tuple(cue.text for cue in scene.lines), (speaker,),
                (_wav(folder / "line.wav", 0.5, 500),), ((0.3, 0.8),), _wav(folder / "voice.wav", 1, 500),
                _wav(folder / "background.wav", 1.5, 300), "a quiet room", (), ())

        film = self.save([timeline.Scene("A radio on a table", 1, lines=(Cue("Good morning", "@hanna"),))],
                         audio_on=True)
        film = self.made(film, self._talking("Good morning", skipped="Lip sync skipped (no face)."))
        source = timeline.asset(film, self.keys(film)[0])
        edited, job = timeline.edit(film, 0, replace(film.scenes[0], lines=(Cue("Good evening", "@hanna"),)))
        with patch.object(timeline, "_scene_sound", side_effect=scene_sound):
            self.run_edit(job)
        self.assertEqual(self.calls, [])  # no new picture
        record = timeline.asset(edited, self.keys(edited)[0])
        self.assertEqual(record["edit"]["kind"], "voice")
        self.assertEqual(record["picture_sha256"], source["picture_sha256"])  # the next scene keeps its anchor
        self.assertEqual(record["lipsync_skipped"], "Lip sync skipped (no face).")
        self.assertEqual(record["dialogue_timings"][0]["text"], "Good evening")
        self.assertEqual(sounds, ["Good evening"])
        self.assertTrue(record["audio_sources"][1].is_file())

    def test_a_realistic_talking_picture_keeps_its_frames_and_only_moves_the_mouth_again(self):
        self._speaking()
        film = self.save([timeline.Scene("An adult man at a desk", 1, lines=(Cue("Hello", "@hanna"),))],
                         audio_on=True)
        film = self.made(film, self._talking("Hello"))
        source = timeline.asset(film, self.keys(film)[0])
        redubs = []

        def redub(picture, package, cast, seconds, seed, size, scene_dir, progress_cb=None):
            redubs.append((Path(picture), package, size))
            return self._talking("Goodbye")(None, None, None, seed, "", scene_dir, None)

        package = SimpleNamespace(key="b" * 64)
        edited, job = timeline.edit(film, 0, replace(film.scenes[0], lines=(Cue("Goodbye", "@hanna"),)))
        with (patch.object(timeline, "_scene_sound", return_value=package),
              patch.object(timeline.video_gen, "redub", side_effect=redub)):
            self.run_edit(job)
        self.assertEqual(self.calls, [])  # no new picture
        self.assertEqual(redubs, [(source["path"], package, (48, 32))])
        self.assertEqual(timeline.asset(edited, self.keys(edited)[0])["edit"]["kind"], "redub")

    def test_only_a_picture_change_redraws_a_dialogue_scene_from_its_frame(self):
        project = timeline.Project("0" * 32, 1, "Modes", (timeline.Scene("A man", 1),), audio=False)
        source = {"seconds": 1, "voices": [], "dialogue_timings": []}
        with patch.object(timeline, "_versions", return_value={}):
            self.assertEqual(timeline._edit_mode(project, project.scenes[0], [], "f" * 64, source, ""), "animate")
            self.assertEqual(timeline._edit_mode(project, project.scenes[0], [], "f" * 64, source, "Night"), "animate")
        signature = timeline._scene_signature(project, project.scenes[0], [])
        same = {signature: {"settings": {}, "bases": {"f" * 64: {"settings": project.scenes[0].clip_settings()}}}}
        with patch.object(timeline, "_versions", return_value=same):
            self.assertEqual(timeline._edit_mode(project, project.scenes[0], [], "f" * 64, source, ""), "voice")
            longer = {**source, "seconds": 2}
            self.assertEqual(timeline._edit_mode(project, project.scenes[0], [], "f" * 64, longer, ""), "animate")


class FilmItemsTests(_Project):
    def test_the_film_keeps_its_places_and_items_outside_its_key(self):
        film = self.save([timeline.Scene("A studio", 1)])
        listed = timeline.save(film.id, film.revision, film.title, film.scenes, film.aspect, film.audio, [],
                               film_items=("tv_studio", "red_mug", "tv_studio"))
        self.assertEqual(listed.items, ("tv_studio", "red_mug"))
        self.assertEqual(listed.key, film.key)  # no clip depends on it
        self.assertEqual(listed.revision, film.revision + 1)
        self.assertEqual(timeline.load(film.id).items, ("tv_studio", "red_mug"))
        kept = timeline.save(film.id, listed.revision, "New name", film.scenes, film.aspect, film.audio, [])
        self.assertEqual(kept.items, ("tv_studio", "red_mug"))
        cleared = timeline.save(film.id, kept.revision, kept.title, film.scenes, film.aspect, film.audio, [],
                                film_items=())
        stored = json.loads((timeline.DIR / film.id / "project.json").read_text(encoding="utf-8"))
        self.assertNotIn("items", stored)  # a film without the list keeps the file it always had
        self.assertEqual(cleared.items, ())
        for broken in (("",), (1,), tuple(f"item{n}" for n in range(timeline.MAX_FILM_ITEMS + 1))):
            with self.subTest(items=broken), self.assertRaisesRegex(ValueError, "Invalid project settings"):
                timeline._validated(replace(cleared, items=broken))

    def test_a_changed_shown_text_is_replaced_and_a_spoken_line_is_not(self):
        prompt = 'A sign reads "OPEN". She says "OPEN the door". A screen displays the text "OPEN".'
        self.assertEqual(dialogue.replace_shown(prompt, "OPEN", "CLOSED"),
                         'A sign reads "CLOSED". She says "OPEN the door". A screen displays the text "CLOSED".')
        self.assertEqual(timeline._shown_changes('a sign reads "OPEN" and the text "SALE"',
                                                 'a sign reads "OPEN" and the text "GONE"'), [("SALE", "GONE")])
        self.assertEqual(timeline._shown_changes('the text "SALE"', 'no text'), [])

    def test_new_line_texts_keep_who_says_them(self):
        scene = timeline.Scene("Two people", 6, lines=(Cue("Hi", "ada", 0.5), Cue("Hello", "kit", 2.0)))
        changed = timeline.with_lines(scene, ["Good evening", "Good night", "Bye"])
        self.assertEqual([(cue.text, cue.speaker, cue.start) for cue in changed.lines],
                         [("Good evening", "ada", 0.5), ("Good night", "kit", 2.0), ("Bye", "kit", None)])
        self.assertEqual([cue.text for cue in timeline.with_lines(scene, ["Only one"]).lines], ["Only one"])


class PlanEditTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(llm, "ensure", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_plan_is_read_from_the_model_s_json(self):
        answer = json.dumps({"picture_text": '@chain_dog sits; a screen displays the text "BREAKING NEWS", 3D '
                                             'animated film', "lines": [],
                             "frame_change": 'Change the screen text to "BREAKING NEWS".'})
        with patch.object(llm, "ask", return_value=(True, answer)) as ask:
            ok, plan = llm.plan_edit("zmień napis na BREAKING NEWS", '@chain_dog sits; a screen displays the text '
                                     '"CRYPTO NEWS"', names=["chain_dog"], style="3d")
        self.assertTrue(ok, plan)
        self.assertEqual(plan["frame_change"], 'Change the screen text to "BREAKING NEWS".')
        self.assertIn('"BREAKING NEWS"', plan["picture_text"])
        system, user = ask.call_args.args[:2]
        self.assertIn("FIRST FRAME", system)
        self.assertIn("@chain_dog", user)
        self.assertIs(ask.call_args.kwargs["schema"], llm._EDIT_SCHEMA)
        check = ask.call_args.kwargs["check"]
        self.assertFalse(check(json.dumps({"picture_text": "chain dog sits", "lines": [], "frame_change": ""})))
        self.assertFalse(check("not json"))
        self.assertTrue(check(answer))

    def test_quotes_go_to_the_model_typographic_and_a_text_cut_at_a_quote_is_refused(self):
        """2026-11-17: a straight quote ended the JSON string, and the plan stopped at 'displays the text'."""
        scene = '@chain_dog sits. A screen displays the text "CRYPTO NEWS". He says "Good evening!"'
        answer = json.dumps({"picture_text": '@chain_dog sits. A screen displays the text “CRYPTO NEWS”. He says '
                                             '“Good evening! Bitcoin is down today.”', "lines": [], "frame_change": ""})
        with patch.object(llm, "ask", return_value=(True, answer)) as ask:
            ok, plan = llm.plan_edit("powie: Bitcoin dziś spada", scene, names=["chain_dog"])
        self.assertTrue(ok, plan)
        self.assertIn("“CRYPTO NEWS”", ask.call_args.args[1])
        self.assertNotIn('"CRYPTO NEWS"', ask.call_args.args[1])
        self.assertEqual(plan["picture_text"], '@chain_dog sits. A screen displays the text "CRYPTO NEWS". He says '
                                               '"Good evening! Bitcoin is down today."')  # the scene's own marks
        cut = json.dumps({"picture_text": "@chain_dog sits. A screen displays the text", "lines": [],
                          "frame_change": ""})
        self.assertFalse(ask.call_args.kwargs["check"](cut))
        with patch.object(llm, "ask", return_value=(True, cut)):
            self.assertEqual(llm.plan_edit("x", scene, names=["chain_dog"]),
                             (False, "The assistant's plan came back cut short. Try again."))
        dropped = json.dumps({"picture_text": "@chain_dog sits silently.", "lines": [], "frame_change": ""})
        with patch.object(llm, "ask", return_value=(True, dropped)):
            self.assertTrue(llm.plan_edit("bez napisu i bez słów", scene, names=["chain_dog"])[0])

    def test_lines_come_back_only_for_a_scene_that_has_them_and_bad_answers_are_refused(self):
        answer = json.dumps({"picture_text": "A man waves", "lines": ["Goodbye"], "frame_change": ""})
        with patch.object(llm, "ask", return_value=(True, answer)):
            self.assertEqual(llm.plan_edit("say goodbye", "A man waves", ["Hello"])[1]["lines"], ["Goodbye"])
            self.assertEqual(llm.plan_edit("say goodbye", "A man waves")[1]["lines"], [])
        with patch.object(llm, "ask", return_value=(True, "{broken")):
            self.assertEqual(llm.plan_edit("x", "A man waves"),
                             (False, "The assistant's plan could not be read. Try again."))
        self.assertEqual(llm.plan_edit("", "A man waves"), (False, "Describe the change in any language."))
        self.assertEqual(llm.plan_edit("make her a naked child", "A woman"), (False, llm.BLOCK_MSG))


class EditTabTests(_Isolated):
    def setUp(self):
        super().setUp()
        enter = self.context.enter_context
        enter(patch.object(timeline, "scene_issues", return_value=[]))
        enter(patch.object(timeline, "sound_preview", return_value=None))
        enter(patch.object(timeline, "preview_mix", return_value=None))
        enter(patch.object(timeline, "leveled_clip", return_value=None))
        enter(patch.object(timeline, "takes", return_value=[]))
        enter(patch.object(timeline, "failure", return_value=None))
        enter(patch.object(timeline, "reference_path", return_value=None))
        self.film = timeline.save(None, None, "Rain", (timeline.Scene("Ada waits", 4), timeline.Scene("Kit runs", 2)),
                                  "Landscape 16:9", True, ["ada"])

    def values(self, draft, *extra):
        scene = draft.scenes[draft.selected]
        form = (scene.prompt, scene.seconds, scene.cut, scene.cast is not None, list(scene.cast or ()), "",
                None, None, scene.background, None, scene.voice_db, scene.background_db)
        return (draft, *form, "Rain", "Landscape 16:9", True, "", ["ada"], False, "realistic", *extra)

    def test_the_tab_has_the_edit_and_items_parts(self):
        demo, _ = self.build()
        config_text = str(demo.get_config_file())
        for label in ("Edit", "Plan with AI", "Make the edit", "Change to the picture", "Make the change in",
                      "Items", "Places and items in this film"):
            self.assertIn(label, config_text)
        self.assertIn("timeline_edit", {fn.api_name for fn in demo.fns.values()})
        use = next(fn for fn in demo.fns.values() if fn.api_name == "timeline_use_take")
        self.assertIs(use.fn, ui_timeline._use_take)

    def test_make_the_edit_saves_the_scene_from_the_inspector_and_refuses_other_unsaved_changes(self):
        draft = Draft(self.film, self.film.scenes, 0, "form")
        typed = replace(draft, scenes=(timeline.Scene("Ada waits in the rain", 4), self.film.scenes[1]))
        job = queue_manager.Job(4, timeline.EDIT, "Edit", {})
        with patch.object(timeline, "edit", return_value=(self.film, job)) as edit:
            result = ui_timeline._make_scene_edit(*self.values(typed, "Make it night", "after"))
        edit.assert_called_once()
        self.assertEqual(edit.call_args.args[:2], (self.film, 0))
        self.assertEqual(edit.call_args.args[2].prompt, "Ada waits in the rain")
        self.assertEqual(edit.call_args.kwargs, {"change": "Make it night", "scope": "after"})
        self.assertEqual(len(result), ui_timeline._FORM_FIELDS + 1 + 7)
        self.assertIsInstance(result[0], Draft)
        self.assertEqual(result[-1], result[0].form)  # the Edit fields belong to the scene loaded again
        self.assertEqual(result[-4:-2], ("", ""))
        self.assertIn("Job #4", result[ui_timeline._FORM_FIELDS + 2])

        other = replace(draft, scenes=(self.film.scenes[0], timeline.Scene("Kit walks", 2)))
        with patch.object(timeline, "edit") as edit:
            result = ui_timeline._make_scene_edit(*self.values(other, "", "scene"))
        edit.assert_not_called()
        self.assertIn("Save the project first", result[-2])
        self.assertTrue(all(value == SKIP for value in result[:-2]))

    def test_plan_with_ai_loads_the_planned_scene_unsaved(self):
        draft = Draft(self.film, self.film.scenes, 0, "form")
        plan = {"picture_text": "Ada waits at night", "lines": [], "frame_change": "Make it night."}
        with patch.object(llm, "plan_edit", return_value=(True, plan)) as planner:
            result = ui_timeline._plan_scene_edit("zrób noc", *self.values(draft))
        self.assertEqual(planner.call_args.args[:2], ("zrób noc", "Ada waits"))
        loaded = result[0]
        self.assertEqual(loaded.scenes[0].prompt, "Ada waits at night")
        self.assertEqual(loaded.baseline, self.film)  # not saved
        self.assertEqual(result[1], "Ada waits at night")
        self.assertEqual(result[-3], "Make it night.")
        self.assertIn("a new picture text", result[-2])
        self.assertIn("a change to the first frame", result[-2])
        self.assertEqual(result[-1], loaded.form)
        with patch.object(llm, "plan_edit", return_value=(False, "The model is busy.")):
            refused = ui_timeline._plan_scene_edit("zrób noc", *self.values(draft))
        self.assertIn("The model is busy.", refused[-2])
        self.assertTrue(all(value == SKIP for value in refused[:-2]))

    def test_another_scene_empties_the_edit_fields(self):
        self.assertEqual(ui_timeline._edit_fields_for(Draft(form="a"), "a"), (SKIP,) * 4)
        self.assertEqual(ui_timeline._edit_fields_for(Draft(form="b"), "a"), ("", "", "", "b"))

    def test_use_this_take_loads_a_scene_brought_back_with_its_take(self):
        earlier = replace(self.film, revision=self.film.revision + 1,
                          scenes=(timeline.Scene("Ada waited", 4), self.film.scenes[1]))
        draft = Draft(self.film, self.film.scenes, 0, "form")
        with patch.object(timeline, "use_take", return_value=earlier):
            result = ui_timeline._use_take(*self.values(draft, "k1"))
        self.assertEqual(result[0].baseline, earlier)
        self.assertEqual(result[1], "Ada waited")
        self.assertIn("back to the chosen version", result[ui_timeline._FORM_FIELDS + 2])
        with patch.object(timeline, "use_take", return_value=self.film):
            result = ui_timeline._use_take(*self.values(draft, "k1"))
        self.assertIn("now uses the chosen take", result[ui_timeline._FORM_FIELDS + 2])
        result = ui_timeline._use_take(*self.values(draft, None))
        self.assertIn("Choose a take first", result[ui_timeline._FORM_FIELDS + 2])

    def test_the_bin_lists_the_film_s_places_and_items_and_the_scene_picker_shows_them_first(self):
        pool = ({"slug": "red_mug", "name": "Red mug", "kind": items.OBJECT},
                {"slug": "tv_studio", "name": "TV studio", "kind": items.PLACE})
        film = replace(self.film, items=("tv_studio", "gone"),
                       scenes=(timeline.Scene("Ada in the tv studio", 4), timeline.Scene("Kit runs", 2)))
        with (patch.object(ui_timeline, "_item_pool", return_value=pool),
              patch.object(items, "main_photo", return_value=None),
              patch.object(timeline, "item_reference", return_value="")):
            cards = ui_timeline._film_item_cards(film)
            choices = ui_timeline._film_item_choices(film.items)
            picker = ui_timeline._scene_items(film.scenes[1], [], film.items)[0]
            context = ui_timeline._scene_context(Draft(film, film.scenes, 0))
        self.assertIn("TV studio", cards)
        self.assertIn("Place · In scene 1", cards)
        self.assertIn("gone (unavailable)", cards)
        self.assertEqual(choices[-1], ("gone (unavailable)", "gone"))
        self.assertEqual(picker[0], ("TV studio", "tv_studio"))
        self.assertIn("TV studio (a place)", context)
        self.assertNotIn("Red mug", context)
        picked = ui_timeline._pick_film_items(Draft(film, film.scenes), ["red_mug", "red_mug", ""])
        self.assertEqual(picked.items, ("red_mug",))
        self.assertEqual(ui_timeline._candidate(picked, "Rain", "Landscape 16:9", True, "", ["ada"]).items,
                         ("red_mug",))
        empty = ui_timeline._film_item_cards(replace(film, items=()))
        self.assertIn("No places or items in this film yet", empty)


if __name__ == "__main__":
    unittest.main()
