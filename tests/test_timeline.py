# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Timeline persistence, continuation frames, and exact picture/audio export clocks."""
from contextlib import ExitStack
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
from PIL import Image

from app import config, dialogue, history, studio, timeline, video_gen, voices
from app.ui_timeline import Draft, _candidate, _edited, _movie_view, _scene_order, _speaker_key, track_html
from engine.media import audio, video


def _film(path, rgb, seconds, frequency=None, *, last_rgb=None):
    track = None
    if frequency:
        t = np.arange(seconds * audio.RATE) / audio.RATE
        mono = (0.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
        track = (np.stack([mono, mono]), audio.RATE)
    frame = np.full((32, 48, 3), rgb, dtype=np.uint8) if isinstance(rgb, tuple) else rgb
    last = frame if last_rgb is None else np.full_like(frame, last_rgb)
    with video.Writer(path, 30, 48, 32, preset="ultrafast", comment=video.AI_COMMENT, audio=track) as out:
        for _ in range(30 * seconds - 1):
            out.write(frame)
        out.write(last)


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.root = Path(self.context.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.context.enter_context(patch.object(timeline, "DIR", self.root / "projects"))
        self.context.enter_context(patch("app.items.DIR", self.root / "items"))
        self.context.enter_context(patch.object(config, "OUTPUTS_DIR", self.root / "outputs"))
        self.context.enter_context(patch.object(config, "HISTORY_DIR", self.root / "history"))
        self.context.enter_context(patch.object(history, "_HISTORY_FILE", self.root / "history/history.jsonl"))
        # A scene with sound gets its sound before the picture (timeline._scene_sound): no model, no GPU here.
        self.context.enter_context(patch.object(timeline.model_manager, "ensure", return_value=None))
        self.context.enter_context(patch.object(timeline.scene_sound, "prepare", side_effect=self._sound))
        # Never read the real model files (13 GB of sound models were hashed here) or record them in the live
        # config/verified_models.json.
        self.context.enter_context(patch.object(timeline.sound, "available", return_value=True))
        self.context.enter_context(patch.object(timeline.tts, "available", return_value=True))
        self.context.enter_context(patch.object(timeline.model_manager, "VERIFIED_FILE",
                                                self.root / "verified_models.json"))
        self.context.enter_context(patch.object(timeline.model_manager, "_STORED", None))

    @staticmethod
    def _sound(cues, speakers, prompt, effects, seconds, seed, folder, progress_cb=None, **_options):
        return timeline.scene_sound.SceneSound(Path(folder), "0" * 64, seconds, (), (), (), (), None, None, prompt,
                                               (), ())

    def save(self, scenes, previous=None, with_audio=True):
        return timeline.save(previous.id if previous else None, previous.revision if previous else None,
                             "Coffee project", tuple(scenes), "Landscape 16:9", with_audio, [])

    def test_drafts_reopen_independently_and_a_stale_tab_cannot_replace_newer_edits(self):
        original = self.save([timeline.Scene("A cup", 4), timeline.Scene("", 7)])
        other = self.save([timeline.Scene("A tree", 2)])
        edited = self.save([timeline.Scene("A glass cup", 8)], original)
        self.assertEqual(len(timeline.projects()), 2)
        self.assertEqual(timeline.load(original.id), edited)
        self.assertEqual(timeline.load(other.id), other)
        with self.assertRaisesRegex(ValueError, "another tab"):
            self.save([timeline.Scene("Outdated", 1)], original)
        with self.assertRaisesRegex(ValueError, "Invalid timeline project"):
            timeline.load("../elsewhere")
        self.assertIn("00:11", track_html(Draft(scenes=original.scenes)))
        self.assertIn("&lt;script&gt;", track_html(Draft(scenes=(timeline.Scene("<script>", 5),))))

    def test_scene_moves_keep_duration_and_reuse_the_source_clips(self):
        first = timeline.Scene("The cup", 12)
        second = timeline.Scene("The same cup", 18)
        third = timeline.Scene("A new location", 30, cut=True)
        saved = self.save([first, second, third])
        draft = Draft(saved, saved.scenes, 0)
        moved = _scene_order(draft, "drop", (0, 2))
        self.assertEqual(moved.scenes, (second, first, third))
        self.assertEqual(moved.selected, 1)
        self.assertIn("01:00", track_html(moved))
        duplicated = _scene_order(moved, "duplicate")
        self.assertEqual(_scene_order(duplicated, "remove"), Draft(saved, moved.scenes, 2))

        edited = self.save(moved.scenes, saved)
        self.assertEqual(timeline.carry_clips(edited, saved, ["one", "two", "three"]), ["two", "one", "three"])
        self.assertEqual(timeline.load(saved.id), edited)

    def test_continuation_uses_the_last_frame_and_exports_even_with_a_visual_jump(self):
        project = self.save([timeline.Scene("The cup sits on a red table", 1),
                             timeline.Scene("Keep the red table in view", 2)])
        request = {"project": project.id, "key": project.key}
        attempts = []

        def generate(prompt, seconds, _aspect, *, scene_dir, image_path, **options):
            scene_dir.mkdir(parents=True)
            attempts.append((seconds, image_path is not None))
            path = scene_dir / "scene.mp4"
            tone = 440 if seconds == 1 else 660
            if image_path:
                with Image.open(image_path) as anchor:
                    rgb = np.array(anchor.convert("RGB"))
                self.assertGreater(rgb[:, :, 0].mean(), 180)
                _film(path, (0, 0, 210), seconds, tone)
            else:
                _film(path, (0, 220, 20), seconds, tone, last_rgb=(220, 40, 20))
            background = scene_dir / "background.wav"
            t = np.arange(int((seconds + 0.5) * audio.RATE)) / audio.RATE
            wave = (0.2 * np.sin(2 * np.pi * tone * t)).astype(np.float32)
            audio.write_wav(background, np.stack([wave, wave]), audio.RATE)
            return video_gen.VideoOutput(True, [str(path)], info={"qc": {"ok": True}, "width": 48, "height": 32},
                                         audio_stems=(background, None))

        with (patch.object(timeline.video_gen, "generate", side_effect=generate),
              patch.object(timeline.llm, "inspect_review", side_effect=AssertionError("Visual comparison called"))):
            self.assertTrue(timeline.prepare_job(request, lambda *_: None)[0])
            self.assertTrue(timeline.prepare_job(request, lambda *_: None)[0])
        self.assertEqual(attempts, [(1, False), (2, True)])
        self.assertFalse(history.recent())
        saved = timeline.load(project.id)
        record = timeline.asset(saved, timeline.prepared(saved)["assets"][1])
        self.assertTrue(record["path"].is_file())
        self.assertIn("preceding scene", record["transition_note"])
        ok, files, _, error = timeline.render_job(request, lambda *_: None)
        self.assertTrue(ok, error)
        self.assertTrue(studio.is_recorded(files[0]))
        self.assertEqual(timeline.exported(project), Path(files[0]))
        export = timeline.movie_preview(project, timeline.prepared(project))
        self.assertEqual(export.revision, project.revision)
        edited = self.save([timeline.Scene("A different cup", 2), project.scenes[1]], project)
        self.assertIsNone(timeline.exported(edited))
        earlier = timeline.movie_preview(edited, timeline.prepared(edited))
        self.assertEqual(earlier.path, export.path)
        self.assertFalse(earlier.current)
        self.assertEqual(_movie_view(edited, edited, export, False).label, "Earlier export")
        self.assertEqual(_movie_view(edited, project, export, False).label, "Earlier export")
        self.assertEqual(_movie_view(project, project, export, False).label, "Current export")
        with av.open(files[0]) as out:
            frames = list(out.decode(video=0))
        self.assertEqual([frame.pts * frame.time_base for frame in frames], [Fraction(i, 30) for i in range(90)])
        self.assertGreater(frames[29].to_ndarray(format="rgb24")[:, :, 0].mean(), 180)
        self.assertGreater(frames[30].to_ndarray(format="rgb24")[:, :, 2].mean(), 180)
        pcm, rate = audio.read(files[0])
        self.assertEqual(pcm.shape, (2, 3 * rate))
        join = pcm[:, int(0.95 * rate):int(1.05 * rate)]
        self.assertGreater(float(np.sqrt(np.mean(join * join))), 0.016)
        for second, tone in ((0.25, 440), (1.5, 660)):
            window = pcm[0, int(second * rate):int((second + .1) * rate)]
            frequency = np.argmax(abs(np.fft.rfft(window))) * 10
            self.assertEqual(frequency, tone)

        project = self.save(project.scenes, edited)
        request = {"project": project.id, "key": project.key}
        with (patch.object(timeline, "_CONTINUITY_VERSION", timeline._CONTINUITY_VERSION + 1),
              patch.object(timeline.video_gen, "generate", side_effect=AssertionError("Already generated"))):
            self.assertIsNone(timeline.exported(project))
            self.assertTrue(timeline.prepare_job(request, lambda *_: None)[0])

    def test_a_continuation_frame_is_reused_without_the_assistant_and_a_cut_starts_afresh(self):
        project = self.save([timeline.Scene("A cup", 1), timeline.Scene("A second view", 1)], with_audio=False)
        request = {"project": project.id, "key": project.key}
        attempts = []

        def generate(*_args, scene_dir, image_path, **_options):
            attempts.append(image_path is not None)
            scene_dir.mkdir(parents=True)
            path = scene_dir / "scene.mp4"
            _film(path, (190, 0, 0), 1)
            return video_gen.VideoOutput(True, [str(path)], info={"qc": {"ok": True}, "width": 48, "height": 32})

        # The previous scene's last frame starts the next one as it is: no review loads the vision model.
        unused = AssertionError("The assistant was asked")
        with (patch.object(timeline.video_gen, "generate", side_effect=generate),
              patch.object(timeline.llm, "ensure", side_effect=unused),
              patch.object(timeline.llm, "inspect_frames", side_effect=unused)):
            ok, _, _, error = timeline.prepare_job(request, lambda *_: None)
            self.assertTrue(ok, error)
            self.assertEqual(len(timeline.prepared(project)["assets"]), 2)
            edited = self.save([timeline.Scene("A cup", 1), timeline.Scene("A second view", 1, cut=True)],
                               project, with_audio=False)
            self.assertTrue(timeline.prepare_job({"project": edited.id, "key": edited.key}, lambda *_: None)[0])
        self.assertEqual(attempts, [False, True, False])  # the first scene's clip is kept through the edit
        self.assertFalse(history.recent())
        with self.assertRaisesRegex(ValueError, "edited"):
            timeline.render_job(request, lambda *_: None)

    def test_generate_one_scene_leaves_later_draft_and_reuses_previous_last_frame(self):
        project = self.save([timeline.Scene("A cup", 1), timeline.Scene("The cup turns", 1),
                             timeline.Scene("", 5)], with_audio=False)
        params = {"project": project.id, "key": project.key}
        starts = []

        def generate(_prompt, seconds, _aspect, *, scene_dir, image_path, **_kwargs):
            starts.append(image_path is not None)
            scene_dir.mkdir(parents=True, exist_ok=True)
            path = scene_dir / "clip.mp4"
            _film(path, (170, 30, 30), seconds)
            return video_gen.VideoOutput(True, [str(path)], info={"qc": {"ok": True}, "width": 48, "height": 32})

        with patch.object(timeline.video_gen, "generate", side_effect=generate):
            ok, _, _, error = timeline.prepare_job({**params, "scene_index": 1}, lambda *_: None)
            self.assertFalse(ok)
            self.assertIn("Generate scene 1 first", error)
            for index in (0, 1):
                self.assertTrue(timeline.prepare_job({**params, "scene_index": index}, lambda *_: None)[0])
        self.assertEqual(starts, [False, True])
        keys = timeline.prepared(project)["assets"]
        self.assertEqual(len(keys), 2)
        self.assertFalse(timeline.all_clips(project, keys))

    def test_join_repairs_only_the_boundary_and_export_uses_the_joined_picture_and_sound(self):
        project = self.save([timeline.Scene("A cup", 1), timeline.Scene("Same cup", 1),
                             timeline.Scene("Different room", 1, cut=True)])
        params = {"project": project.id, "key": project.key, "require_merge": True}
        keys = ["1" * 64, "2" * 64, "3" * 64]
        colors = [(200, 0, 0), (0, 0, 200), (0, 180, 0)]
        for key, color in zip(keys, colors):
            clip, metadata = timeline._asset_paths(project, key)
            clip.parent.mkdir(parents=True, exist_ok=True)
            _film(clip, color, 1, 440)
            timeline._write(metadata, {"qc": {"ok": True, "reasons": [], "face": None}, "sha256": timeline._sha(clip), "seconds": 1,
                                       "width": 48, "height": 32, "characters": [], "voices": [],
                                       "transition_note": ""})
        timeline._write_plan(project, {"assets": keys})

        def bridge(_first, _last, output, frames, fps):
            with video.Writer(output, fps, 48, 32) as writer:
                for _ in range(frames):
                    writer.write(np.full((32, 48, 3), (0, 180, 0), dtype=np.uint8))

        self.assertFalse(timeline.render_job(params, lambda *_: None)[0])
        with (patch.object(timeline.llm, "inspect_review", return_value={"verdict": "bridge", "reason": "Small jump."}) as check,
              patch.object(timeline.engine_runtime, "transition_frames", side_effect=bridge)):
            ok, files, notes, error = timeline.merge_job(params, lambda *_: None)
        self.assertTrue(ok, error)
        self.assertEqual(files, [])
        self.assertIn("opening frames smoothed", notes[0])
        self.assertIn("intentional cut", notes[1])
        check.assert_called_once()
        self.assertEqual(len(check.call_args.args[0]), 3)
        self.assertEqual(timeline.prepared(project)["assets"], keys)
        self.assertFalse(history.recent())

        ok, files, _, error = timeline.render_job(params, lambda *_: None)
        self.assertTrue(ok, error)
        exported = Path(files[0])
        self.assertTrue(studio.is_recorded(str(exported)))
        self.assertEqual(video.probe(exported).duration, 3)
        with av.open(str(exported)) as source:
            frames = [frame.to_ndarray(format="rgb24").mean(axis=(0, 1)) for frame in source.decode(video=0)]
        self.assertEqual(len(frames), 90)
        self.assertGreater(frames[29][0], 150)
        self.assertGreater(frames[30][1], 140)
        self.assertGreater(frames[30 + timeline._BRIDGE_FRAMES][2], 140)
        pcm, rate = audio.read(exported)
        self.assertEqual(pcm.shape[-1], 3 * rate)
        self.assertGreater(float(np.abs(pcm[:, rate:2 * rate]).mean()), 0.05)

    def test_project_cast_uses_saved_universe_members_and_optional_character(self):
        profiles = [{"slug": slug, "name": slug.capitalize()} for slug in ("ada", "vera", "kit", "rose")]
        with patch.object(timeline.universes, "get", return_value={"members": ["ada", "vera"]}):
            draft = Draft()
            chosen = _candidate(draft, "Cast test", "Landscape 16:9", True, "friends", ["rose"])
            saved = timeline.save(None, None, chosen.title, chosen.scenes, chosen.aspect, chosen.audio,
                                  list(chosen.characters), universe=chosen.universe)
        with patch.object(timeline.universes, "get", return_value={"members": ["ada", "kit"]}):
            reopened = timeline.load(saved.id)
            current = _candidate(Draft(reopened, reopened.scenes), reopened.title, reopened.aspect,
                                 reopened.audio, reopened.universe, reopened.characters)
        self.assertEqual(current.cast_slugs, ("ada", "vera", "rose"))
        with (patch.object(timeline.characters, "all_characters", return_value=profiles),
              patch.object(timeline.characters, "timeline_profile",
                           side_effect=lambda slug: next((p for p in profiles if p["slug"] == slug), None))):
            people = timeline._scene_cast(current, timeline.Scene('Vera greets Rose. Rose says “Hello.”', 2))
            self.assertEqual([person["slug"] for person in people], ["vera", "rose"])
            with self.assertRaisesRegex(ValueError, "Add Kit"):
                timeline._scene_cast(current, timeline.Scene("Kit closes the door.", 1))

    def test_scene_roles_survive_reopening_and_a_voice_change_invalidates_footage(self):
        prompt = 'One person opens the door and says “Wait.” Another answers “I’m coming.”'
        before = timeline.Scene(prompt, 5)
        saved = timeline.save(None, None, "Scene roles", (before,), "Portrait 9:16",
                              True, [], universe="crew", universe_members=("ada", "kit"))
        # "preview": the one video mode under the name projects saved before 2026-11-14 hashed
        legacy = [saved.aspect, "preview", saved.audio, saved.characters,
                  [{"prompt": prompt, "seconds": 5, "cut": False}], {"universe_cast": ("ada", "kit")}]
        self.assertEqual(saved.key, timeline._hash(legacy))
        self.assertNotIn("speakers", json.loads((timeline.DIR / saved.id / "project.json").read_text(
            encoding="utf-8"))["scenes"][0])
        after = timeline.Scene(prompt, 5, cast=("ada", "kit"), speakers=("kit", "ada"))
        updated = timeline.save(saved.id, saved.revision, saved.title, (after,), saved.aspect,
                                 True, [], universe="crew")
        self.assertEqual(timeline.load(saved.id), updated)
        self.assertEqual(timeline.carry_clips(updated, saved, ["old"]), [None])
        draft = Draft(updated, updated.scenes)
        assignments = json.dumps({"key": _speaker_key(draft), "assignments": [["Wait.", "ada"], ["I’m coming.", "kit"]]})
        switched = _edited(draft, prompt, 5, False, True, ["ada", "kit"], assignments)
        self.assertEqual(switched.scenes[0].speakers, ("ada", "kit"))
        rewritten = _edited(draft, prompt.replace("Wait.", "Now."), 5, False, True, ["ada", "kit"])
        self.assertEqual(rewritten.scenes[0].speakers, ("", "ada"))
        with self.assertRaisesRegex(ValueError, "character appearing"):
            timeline.save(saved.id, updated.revision, saved.title,
                          (timeline.Scene(prompt, 5, cast=("ada",), speakers=("kit",)),),
                          saved.aspect, True, [])

    def test_explicit_scene_uses_the_cast_even_when_the_prompt_does_not_name_them(self):
        profiles = {"ada": {"slug": "ada", "name": "Ada", "gender": "woman", "voice": "hanna"},
                    "kit": {"slug": "kit", "name": "Kit", "gender": "man", "voice": "marek"}}
        scene = timeline.Scene('They turn to the door. She says “Wait.” He says “Ready.”', 5,
                               cast=("ada", "kit"), speakers=("kit", "ada"))
        saved = timeline.save(None, None, "Scene roles", (scene,), "Portrait 9:16",
                              True, [], universe="crew", universe_members=("ada", "kit"))
        pool = [voices.Voice("Hanna", "hanna", "female", str(self.root / "hanna.wav")),
                voices.Voice("Marek", "marek", "male", str(self.root / "marek.wav"))]
        with (patch.object(timeline.characters, "timeline_profile", side_effect=profiles.get),
              patch.object(timeline.voices, "all_voices", return_value=pool),
              patch.object(timeline, "_audition", return_value=None),
              patch.object(timeline, "_scene_sound", return_value=None),
              patch.object(video_gen, "generate") as generate):
            members = timeline._scene_cast(saved, scene)
            timeline._generate(saved, scene, members, 1, "", self.root, lambda *_: None)
        self.assertEqual(generate.call_args.kwargs["cast"], list(profiles.values()))
        self.assertTrue(generate.call_args.kwargs["exact_cast"])
        # The quoted lines reach generation as cues that name their speakers; the picture text has no quotes.
        cues = generate.call_args.kwargs["cues"]
        self.assertEqual([(cue.text, cue.speaker) for cue in cues], [("Wait.", "kit"), ("Ready.", "ada")])
        self.assertFalse(dialogue.quotes(generate.call_args.args[0]))
        self.assertTrue(generate.call_args.args[0].endswith("natural lip movement"))
        with patch.object(timeline.voices, "catalog", return_value={"female": ["hanna"], "male": ["marek"]}):
            speakers = timeline.scene_sound.resolve_speakers(cues, members, pool)
        self.assertEqual([speaker.who for speaker in speakers], ["male#char:kit", "female#char:ada"])
        self.assertEqual([speaker.voice for speaker in speakers], ["marek", "hanna"])


class TimelineAudioTests(unittest.TestCase):
    def test_replaces_inaudible_background_and_keeps_dialogue_at_its_original_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, joined = root / "scene.mp4", root / "joined.mp4"
            quiet, room, voice = root / "quiet.wav", root / "room.wav", root / "voice.wav"
            _film(source, (120, 90, 50), 2)
            rate = audio.RATE
            t = np.arange(int(2.5 * rate)) / rate
            audio.write_wav(quiet, np.zeros((1, t.size), dtype=np.float32), rate)
            audio.write_wav(room, (0.05 * np.sin(2 * np.pi * 290 * t))[None, :], rate)
            speech = np.zeros(rate, dtype=np.float32)
            speech[rate // 3:] = 0.25 * np.sin(2 * np.pi * 810 * t[:rate - rate // 3])
            audio.write_wav(voice, speech[None, :], rate)
            clips = [(source, 2)] * 3
            stems = [(room, None), (quiet, voice), (room, None)]
            with self.assertRaisesRegex(audio.SilentBackground, "Scene 2: .*audio is silent"):
                video.join_timeline(clips, joined, 30, sound_stems=stems)
            with self.assertRaisesRegex(audio.SilentBackground, "Scene 2: replacement background failed"):
                video.join_timeline(clips, joined, 30, sound_stems=stems, background_fallback=lambda *_: quiet)
            replaced = []

            def fallback(number, seconds):
                replaced.append((number, seconds))
                return room

            original = quiet.read_bytes()
            self.assertEqual(video.join_timeline(clips, joined, 30, sound_stems=stems,
                                                 background_fallback=fallback), 180)
            self.assertEqual(replaced, [(2, 2)])
            self.assertEqual(quiet.read_bytes(), original)
            pcm, _ = audio.read(joined)
            self.assertEqual(pcm.shape, (2, 6 * rate))
            self.assertGreater(float(np.sqrt(np.mean(pcm[:, 3 * rate:4 * rate] ** 2))), 0.01)
            self.assertGreater(float(np.corrcoef(pcm[0, 2 * rate:3 * rate], speech)[0, 1]), 0.95)

    def test_five_ten_second_scenes_keep_audio_and_dialogue_on_the_timeline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, joined, voice = root / "scene.mp4", root / "joined.mp4", root / "voice.wav"
            _film(source, (120, 90, 50), 10)
            rate = audio.RATE
            t = np.arange(int(10.5 * rate)) / rate
            stems = []
            for index, frequency in enumerate((210, 250, 290, 330, 370)):
                background = root / f"background-{index}.wav"
                bed = (0.1 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
                bed[:rate // 3] = 0
                bed[-rate // 3:] = 0
                audio.write_wav(background, np.stack([bed, bed]), rate)
                stems.append((background, voice if index == 2 else None))
            line = np.zeros(rate, dtype=np.float32)
            line[rate // 3:] = 0.25 * np.sin(2 * np.pi * 810 * t[:rate - rate // 3])
            audio.write_wav(voice, line[None, :], rate)
            self.assertEqual(video.join_timeline([(source, 10)] * 5, joined, 30, sound_stems=stems), 1500)
            with av.open(str(joined)) as output:
                stream = output.streams.video[0]
                self.assertEqual(stream.duration * stream.time_base, 50)
            pcm, _ = audio.read(joined)
            self.assertEqual(pcm.shape, (2, 50 * rate))
            for seconds in (10, 20, 30, 40):
                middle = seconds * rate
                near = pcm[:, middle - rate // 20:middle + rate // 20]
                self.assertGreater(float(np.sqrt(np.mean(near * near))), 0.016)
            self.assertGreater(float(np.corrcoef(pcm[0, 20 * rate:21 * rate], line)[0, 1]), 0.95)

    def test_silent_scene_keeps_its_place_between_audible_scenes(self):
        with tempfile.TemporaryDirectory() as folder:
            files = [Path(folder) / f"scene{i}.mp4" for i in range(3)]
            for index, frequency in enumerate([440, None, 660]):
                _film(files[index], (90 + 30 * index, 60, 40), 1, frequency)
            joined = Path(folder) / "joined.mp4"
            self.assertEqual(video.join_timeline([(f, 1) for f in files], joined, 30), 90)
            pcm, rate = audio.read(joined)
            self.assertEqual(pcm.shape, (2, 3 * rate))
            self.assertLess(abs(pcm[:, int(1.2 * rate):int(1.8 * rate)]).max(), .001)
            self.assertGreater(abs(pcm[:, int(2.2 * rate):int(2.8 * rate)]).max(), .1)
