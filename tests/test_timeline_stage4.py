# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Timeline stage 4: structured scenes (lines, background, effects, references, levels), sound
previews, takes, failure records, the export job, join checks that survive a level change and the
reuse of clips whose picture did not change. CPU only: picture, speech and sound generation are stand-ins."""
from contextlib import ExitStack
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import av
import numpy as np
from PIL import Image

from app import characters, config, history, llm, queue_manager, scene_parts, scene_sound, studio, timeline
from app import video_gen, voices
from app.scene_parts import Cue, Effect
from engine.media import audio, video

RATE = audio.RATE


def _tone(seconds, frequency, level=0.1):
    t = np.arange(int(round(seconds * RATE))) / RATE
    mono = (level * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    return np.stack([mono, mono])


def _film(path, rgb, seconds, frequency=None, size=(48, 32)):
    track = (_tone(seconds, frequency, 0.2), RATE) if frequency else None
    width, height = size
    frame = np.full((height, width, 3), rgb, dtype=np.uint8)
    with video.Writer(path, 30, width, height, preset="ultrafast", comment=video.AI_COMMENT, audio=track) as out:
        for _ in range(30 * seconds):
            out.write(frame)


def _video_packets(path):
    with av.open(str(path)) as container:
        return [bytes(packet) for packet in container.demux(container.streams.video[0]) if packet.size]


class FakeVideo:
    """video_gen.generate without a GPU: a tiny clip with its background (and dialogue) stems."""

    def __init__(self):
        self.calls = []
        self.size = (48, 32)  # the clip's frame; a test can ask for the real 480p one

    def __call__(self, prompt, seconds, aspect, *, scene_dir, image_path=None, sound_on=True, cues=None,
                 prepared_sound=None, **options):
        self.calls.append({"prompt": prompt, "seconds": seconds, "image_path": image_path, "cues": cues,
                           "prepared_sound": prepared_sound, **options})
        scene_dir.mkdir(parents=True, exist_ok=True)
        clip = scene_dir / "scene.mp4"
        _film(clip, (30 + 25 * len(self.calls), 90, 160), seconds, 440 if sound_on else None, self.size)
        info = {"qc": {"ok": True, "reasons": [], "face": None}, "width": self.size[0], "height": self.size[1]}
        if prepared_sound is not None:
            info["sound_key"] = prepared_sound.key  # as video_gen reports the scene sound it used
        stems = None
        if sound_on:
            background = scene_dir / "background.wav"
            audio.write_wav(background, _tone(seconds + 0.5, 220))
            voice = None
            if cues:
                voice = scene_dir / "voice.wav"
                audio.write_wav(voice, _tone(seconds, 660, 0.2))
                windows = scene_parts.timings(cues, [0.6] * len(cues))
                info["voices"] = ["Hanna"] * len(cues)
                info["dialogue_timings"] = [{"text": cue.text, "character": cue.character, "voice": "hanna",
                                             "start": start, "end": end} for cue, (start, end) in zip(cues, windows)]
            stems = background, voice
        return video_gen.VideoOutput(True, [str(clip)], info=info, audio_stems=stems)


class FakeSound:
    """scene_sound.prepare / load without the speech and sound models: the package is a few tones."""

    def __init__(self):
        self.prepared = []
        self.batches = []  # synthesize_lines calls: the texts of each scene in the shared batch
        self.fail = None

    def synthesize_lines(self, scenes, folder, progress_cb=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.batches.append([tuple(cue.text for cue in cues) for cues, _ in scenes])
        made = []
        for number, (cues, _speakers) in enumerate(scenes, 1):
            paths = tuple(folder / f"scene{number}_line{line}.wav" for line in range(1, len(cues) + 1))
            for path in paths:
                audio.write_wav(path, _tone(1.0, 660, 0.2))
            made.append(paths)
        return made

    def prepare(self, cues, speakers, background_prompt, effects, seconds, seed, folder, progress_cb=None, *,
                client=None, voice=True, background=True, lines=None):
        key = scene_sound.sound_key(cues, speakers, background_prompt, effects, seconds, seed, voice=voice,
                                    background=background)
        if (found := self.load(folder, key)) is not None:
            return found
        if self.fail:
            raise ValueError(self.fail)
        folder = Path(folder)
        self.prepared.append({"cues": tuple(cues), "voice": voice, "background": background, "folder": folder,
                              "prompt": background_prompt, "effects": tuple(effects), "seed": seed,
                              "lines": None if lines is None else tuple(Path(path).name for path in lines)})
        folder.mkdir(parents=True, exist_ok=True)
        talk = voice and bool(cues)
        windows = scene_parts.timings(cues, [1.0] * len(cues)) if talk else []
        if talk:
            for number in range(1, len(cues) + 1):
                audio.write_wav(folder / f"line{number}.wav", _tone(1.0, 660, 0.2))
            audio.write_wav(folder / "voice.wav", _tone(seconds, 660, 0.2))
        if background:
            audio.write_wav(folder / "background.wav", _tone(seconds + 0.5, 330))
        manifest = {"key": key, "seconds": seconds, "texts": [cue.text for cue in cues] if talk else [],
                    "speakers": [asdict(speaker) for speaker in speakers] if talk else [], "timings": windows,
                    "voice": talk, "background": background, "prompt": background_prompt if background else "",
                    "effects": [effect.text for effect in effects] if background else []}
        (folder / "fake.json").write_text(json.dumps(manifest), encoding="utf-8")
        if progress_cb:
            progress_cb(1.0, "Sound ready.")
        return self.load(folder, key)

    def load(self, folder, key=None, *, verify=False):
        folder = Path(folder)
        try:
            data = json.loads((folder / "fake.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if key is not None and data["key"] != key:
            return None
        count = len(data["texts"])
        return scene_sound.SceneSound(
            folder, data["key"], data["seconds"], tuple(data["texts"]),
            tuple(scene_sound.Speaker(**person) for person in data["speakers"]),
            tuple(folder / f"line{number}.wav" for number in range(1, count + 1)),
            tuple(tuple(window) for window in data["timings"]),
            folder / "voice.wav" if data["voice"] else None, folder / "background.wav" if data["background"] else None,
            data["prompt"], (), tuple(data["effects"]))


class Stage4Case(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        enter = self.context.enter_context
        self.root = Path(enter(tempfile.TemporaryDirectory())).resolve()
        enter(patch.object(timeline, "DIR", self.root / "projects"))
        enter(patch("app.items.DIR", self.root / "items"))
        enter(patch.object(config, "OUTPUTS_DIR", self.root / "outputs"))
        enter(patch.object(config, "HISTORY_DIR", self.root / "history"))
        enter(patch.object(history, "_HISTORY_FILE", self.root / "history/history.jsonl"))
        enter(patch.object(queue_manager, "_STATE", self.root / "queue.json"))
        enter(patch.object(queue_manager, "FILES_DIR", self.root / "queue_files"))
        enter(patch.object(queue_manager, "_JOBS", []))
        enter(patch.object(queue_manager, "_STATE_INFO", {"next_id": 1, "paused": False}))
        for memo in (timeline._MEMO, timeline._CLIP_INDEX):
            memo.clear()
            self.addCleanup(memo.clear)
        self.profiles = {
            "ada": {"slug": "ada", "name": "Ada", "gender": "woman", "voice": "hanna", "status": "ready"},
            "kit": {"slug": "kit", "name": "Kit", "gender": "man", "voice": "", "status": "ready"},
        }
        samples = self.root / "voices"
        samples.mkdir()
        self.pool = []
        for name, gender in (("Hanna", "female"), ("Zoe", "female"), ("Marek", "male")):
            sample = samples / f"{name.lower()}.wav"
            audio.write_wav(sample, _tone(1.0, 200))
            self.pool.append(voices.Voice(name, name.lower(), gender, str(sample)))
        catalog = {"female": ["hanna", "zoe"], "male": ["marek"]}
        enter(patch.object(characters, "timeline_profile", side_effect=self.profiles.get))
        enter(patch.object(characters, "all_characters", side_effect=lambda: list(self.profiles.values())))
        enter(patch.object(characters, "get", side_effect=self.profiles.get))
        enter(patch.object(characters, "usable", side_effect=self.profiles.get))
        enter(patch.object(voices, "all_voices", side_effect=lambda: list(self.pool)))
        enter(patch.object(voices, "catalog", side_effect=lambda pool=None: {k: list(v) for k, v in catalog.items()}))
        self.speech = enter(patch.object(timeline.tts, "available", return_value=True))
        self.ambience = enter(patch.object(timeline.sound, "available", return_value=True))
        enter(patch.object(timeline.model_manager, "tier", return_value="full"))
        self.sound = FakeSound()
        enter(patch.object(scene_sound, "prepare", side_effect=self.sound.prepare))
        enter(patch.object(scene_sound, "load", side_effect=self.sound.load))
        enter(patch.object(scene_sound, "synthesize_lines", side_effect=self.sound.synthesize_lines))
        # the job's sound pass trims the card and opens one audio session: no real engine in a test
        self.runtime = enter(patch.object(timeline.engine_runtime, "engine", return_value=MagicMock()))
        self.video = FakeVideo()
        enter(patch.object(video_gen, "generate", side_effect=self.video))
        self.ensure = enter(patch.object(timeline.model_manager, "ensure", return_value=None))

    def save(self, scenes, previous=None, *, sound=True, title="Stage four"):
        return timeline.save(previous.id if previous else None, previous.revision if previous else None,
                             title, tuple(scenes), "Landscape 16:9", sound, [],
                             universe="crew", universe_members=("ada", "kit"))

    def run_job(self, job, runner=None):
        runner = runner or {timeline.PREPARE: timeline.prepare_job, timeline.SOUND: timeline.sound_job,
                            timeline.MERGE: timeline.merge_job, timeline.RENDER: timeline.render_job,
                            timeline.PRODUCE: timeline.produce_job}[job.kind]
        try:
            return runner(job.params, lambda *_: None)
        finally:
            job.status = queue_manager.DONE

    def prepare(self, project, **params):
        ok, _, notes, error = timeline.prepare_job({"project": project.id, "key": project.key, **params},
                                                   lambda *_: None)
        self.assertTrue(ok, error)
        return timeline.prepared(project)["assets"]

    def image(self, name, size=(64, 48), mode="RGB", color=(200, 30, 30)):
        path = self.root / name
        Image.new(mode, size, color).save(path)
        return path


class SceneModelTests(Stage4Case):
    def test_new_fields_round_trip_and_stay_out_of_older_scenes(self):
        reference = timeline.store_reference(self.image("cup.png"))
        scene = timeline.Scene("Ada holds the cup [1] at the door.", 4,
                               lines=(Cue("Hello there.", "ada", 0.5), Cue("Hi.", "@marek")),
                               background="rain on the window", effects=(Effect("door knock", 1, 1.5),),
                               references=(reference,), voice_db=-3.26, background_db=2)
        saved = self.save([scene, timeline.Scene("Kit waits.", 2)])
        stored = json.loads((timeline.DIR / saved.id / "project.json").read_text(encoding="utf-8"))["scenes"]
        self.assertEqual(stored[0], {
            "prompt": "Ada holds the cup [1] at the door.", "seconds": 4, "cut": False,
            "lines": [{"text": "Hello there.", "speaker": "ada", "start": 0.5}, {"text": "Hi.", "speaker": "@marek"}],
            "background": "rain on the window", "effects": [{"text": "door knock", "at": 1.0, "seconds": 1.5}],
            "references": [reference], "voice_db": -3.5, "background_db": 2.0})
        self.assertEqual(stored[1], {"prompt": "Kit waits.", "seconds": 2, "cut": False})
        self.assertIs(type(saved.scenes[0].background_db), float)
        reopened = timeline.load(saved.id)
        self.assertEqual(reopened, saved)
        self.assertEqual(reopened.key, saved.key)
        self.assertEqual(timeline.Scene.from_json(json.loads(json.dumps(saved.scenes[0].settings()))),
                         saved.scenes[0])
        self.assertNotIn("voice_db", saved.scenes[0].clip_settings())
        self.assertEqual(saved.scenes[1].clip_settings(), saved.scenes[1].settings())
        quieter = replace(saved.scenes[0], voice_db=-12.0)
        self.assertEqual(quieter.footage(), saved.scenes[0].footage())
        self.assertNotEqual(quieter.settings(), saved.scenes[0].settings())
        louder_rain = replace(saved.scenes[0], background="storm", effects=())
        self.assertEqual(louder_rain.picture_footage(), saved.scenes[0].picture_footage())
        self.assertNotEqual(louder_rain.footage(), saved.scenes[0].footage())
        with self.assertRaisesRegex(ValueError, "Invalid scene"):
            timeline.Scene.from_json({"prompt": "A cup", "seconds": 1, "cut": False, "mood": "calm"})
        with self.assertRaisesRegex(ValueError, "spoken line"):
            timeline.Scene.from_json({"prompt": "A cup", "seconds": 1, "cut": False, "lines": [{"words": "Hi"}]})

    _FIRST = {"prompt": "Ada and Kit stand at the door.", "seconds": 1, "cut": False}
    _SECOND = {"prompt": "She says “Wait.” He says “Ready.”", "seconds": 1, "cut": False,
               "cast": ["ada", "kit"], "speakers": ["ada", "kit"]}

    def _old_project(self, project_id, mode):
        """A project saved before stage four and before the one video mode, in `mode` ("preview", "fast"
        or "quality"), with the clips that mode made, keyed as they were then: (project, folder, clip
        keys, clip records)."""
        first, second = self._FIRST, self._SECOND
        folder = timeline.DIR / project_id
        folder.mkdir(parents=True)
        fields = {"id": project_id, "revision": 3, "title": "Before stage four", "scenes": [first, second],
                  "aspect": "Landscape 16:9", "speed": mode, "audio": True, "characters": [],
                  "updated": 1.0, "reference_characters": [], "universe": "crew", "universe_members": ["ada", "kit"]}
        (folder / "project.json").write_text(json.dumps(fields), encoding="utf-8")
        project = timeline.load(project_id)

        # The clip keys of the stage before: scene.settings(), the members, the previous clip's sha256.
        style, members = timeline.prefs.STYLE["video"], [self.profiles["ada"], self.profiles["kit"]]
        keys, records, previous = [], [], None
        for number, settings in enumerate((first, second)):
            continuity = [previous["sha256"], first["prompt"]] if previous else None
            key = timeline._hash([project.aspect, mode, True, style, "full", settings, members, number == 0,
                                  continuity, "separate-sound-v1", timeline.sound.MODEL_ID])
            clip, metadata = timeline._asset_paths(project, key)
            clip.parent.mkdir(parents=True, exist_ok=True)
            _film(clip, (90, 40 + 80 * number, 20), 1, 440)
            background = timeline._sound_path(clip, "background")
            audio.write_wav(background, _tone(1.5, 220))
            # written before 2026-11-14: the clip record still carries the removed review flag
            previous = {"qc": {"safe": True}, "sha256": timeline._sha(clip), "seconds": 1, "width": 48,
                        "height": 32, "characters": ["ada", "kit"], "voices": [], "transition_note": "",
                        "continuity_version": 1, "sound_hashes": {"background": timeline._sha(background)}}
            timeline._write(metadata, previous)
            keys.append(key)
            records.append(metadata.read_text(encoding="utf-8"))
        return project, folder, keys, records

    def _legacy_key(self, mode):
        return timeline._hash(["Landscape 16:9", mode, True, [], [self._FIRST, self._SECOND],
                               {"universe_cast": ["ada", "kit"]}])

    def test_an_old_project_file_keeps_its_key_and_reuses_its_clips(self):
        """A project saved in Preview, the mode that remains: same key, same clips, nothing generated."""
        project_id = "0123456789abcdef0123456789abcdef"
        project, folder, keys, records = self._old_project(project_id, "preview")
        self.assertEqual(project.key, self._legacy_key("preview"))
        self.assertEqual(self.prepare(project), keys)
        self.assertEqual(self.video.calls, [])
        self.assertEqual([timeline._asset_paths(project, key)[1].read_text(encoding="utf-8") for key in keys], records)
        self.assertEqual(json.loads((folder / "project.json").read_text(encoding="utf-8"))["scenes"],
                         [self._FIRST, self._SECOND])
        self.assertEqual(timeline.prepared(timeline.load(project_id))["assets"], keys)

    def test_a_fast_or_quality_project_opens_in_the_one_mode_and_makes_its_clips_again(self):
        """Its 1080p clips no longer match its keys: they count as outdated, are made again in the one
        mode, and the old files stay on disk untouched."""
        for number, mode in enumerate(("fast", "quality")):
            with self.subTest(mode=mode):
                self.video.calls.clear()
                project, folder, old_keys, records = self._old_project(f"{number:032x}", mode)
                self.assertFalse(hasattr(project, "speed"))
                self.assertNotEqual(project.key, self._legacy_key(mode))
                self.assertEqual(project.key, self._legacy_key("preview"))
                self.assertEqual(timeline.outdated(project), [0, 1])
                made = self.prepare(project)
                self.assertEqual(len(self.video.calls), 2)
                self.assertNotIn("speed", self.video.calls[0])
                self.assertTrue(made and not set(made) & set(old_keys))
                self.assertEqual([timeline._asset_paths(project, key)[1].read_text(encoding="utf-8")
                                  for key in old_keys], records)

    def test_a_scene_that_shows_quoted_text_is_made_again_and_the_others_keep_their_keys(self):
        """Until 2026-11-16 the text a scene shows in its picture ("a sign reads "OPEN"") was spoken and replaced by
        "talking": only such scenes get new keys, so their clips show Outdated."""
        project = self.save([timeline.Scene("Ada waits.", 3), timeline.Scene("A sign reads “OPEN”. Ada waits.", 3)])

        def keys():
            return [(timeline._picture_key(project, scene, [], project.video_style, "full", number, None),
                     timeline._clip_key(project, scene.clip_settings(), [], project.video_style, "full", number, None))
                    for number, scene in enumerate(project.scenes)]

        now = keys()
        with patch.object(timeline.dialogue, "shown", return_value=()):
            before = keys()  # as every key was before
        self.assertEqual(now[0], before[0])
        self.assertNotEqual(now[1][0], before[1][0])
        self.assertNotEqual(now[1][1], before[1][1])

    def test_a_scene_lasts_tenths_of_a_second_and_whole_seconds_keep_their_keys(self):
        project = self.save([timeline.Scene("Ada waits.", 2.5), timeline.Scene("Kit runs.", 3)])
        opened = timeline.load(project.id)
        self.assertEqual([scene.seconds for scene in opened.scenes], [2.5, 3])
        self.assertEqual((opened.seconds, timeline.Scene("Kit runs.", 3.0) == timeline.Scene("Kit runs.", 3)),
                         (5.5, True))
        # 3.0 is stored as 3, so a scene of whole seconds keeps the key it always had
        same = self.save([timeline.Scene("Ada waits.", 2.5), timeline.Scene("Kit runs.", 3.0)], project)
        self.assertEqual(type(same.scenes[1].seconds), int)
        self.assertEqual(same.key, project.key)
        for wrong in (2.55, 0.5, 31):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, "tenths of a second"):
                self.save([timeline.Scene("Ada waits.", wrong)], same)

    def test_validation_names_the_problem(self):
        ok = self.save([timeline.Scene("Ada waits.", 3)])
        cases = [
            (timeline.Scene("Ada says “Hi.”", 3, lines=(Cue("Hello."),)),
             "Scene 1: move the quoted lines into Dialogue or remove the quotes."),
            (timeline.Scene("Ada says “Hi.”", 3, cast=("ada",), speakers=("ada",), lines=(Cue("Hello.", "ada"),)),
             "choose who speaks in the Dialogue lines"),
            (timeline.Scene("Ada waits.", 3, cast=("ada",), lines=(Cue("Hello.", "kit"),)), "not in this scene"),
            (timeline.Scene("Ada waits.", 3, lines=(Cue("Hello.", "rose"),)), "not in the project’s cast"),
            (timeline.Scene("Ada waits.", 3, voice_db=20), "Choose a level"),
            (timeline.Scene("Ada waits.", 3, background_db="loud"), "Enter a level in dB"),
            (timeline.Scene("Ada waits.", 3, background="rain " * 61), "background sound under 300"),
            (timeline.Scene("Ada waits.", 3, references=("cup",)), "reference images"),
            (timeline.Scene("Ada waits.", 3, references=("a" * 64, "a" * 64)), "different reference images"),
            (timeline.Scene("Ada waits.", 3, references=tuple(f"{n:064x}" for n in range(7))), "at most 6"),
            (timeline.Scene("Ada holds [2].", 3, references=("b" * 64,)), r"\[2\] does not match"),
            (timeline.Scene("Ada waits.", 3, lines=[Cue("Hello.")]), "invalid"),
        ]
        for scene, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.save([scene], ok)
        # the scene's length is never refused: a line or an effect after its end is only not heard (a warning)
        short = self.save([timeline.Scene("Ada waits.", 1, lines=(Cue("Hello.", "", 5),), effects=(Effect("bang", 4),))],
                          ok)
        self.assertEqual((short.scenes[0].seconds, short.scenes[0].lines[0].start), (1, 5.0))
        ok = short
        # An opened project may still refer to an image that is not attached; saving asks for it.
        stored = timeline.DIR / ok.id / "project.json"
        fields = json.loads(stored.read_text(encoding="utf-8"))
        fields["scenes"][0]["prompt"] = "Ada holds the cup [1]."
        stored.write_text(json.dumps(fields), encoding="utf-8")
        opened = timeline.load(ok.id)
        with self.assertRaisesRegex(ValueError, r"\[1\] does not match"):
            self.save(opened.scenes, opened)
        fields["scenes"][0]["mood"] = "calm"
        stored.write_text(json.dumps(fields), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Could not open"):
            timeline.load(ok.id)

    def test_a_minor_and_sexual_content_in_different_fields_stop_the_scene_before_any_model(self):
        """The queued Timeline jobs carry no text: the minors guard runs on the scene's description, lines,
        background and effects together, so an age in a line and clothing in an effect are refused too."""
        blocked = timeline.Scene("Ada waits.", 3, lines=(Cue("a 15-year-old", "ada"),),
                                 effects=(Effect("in a bikini", 1),))
        project = self.save([blocked])
        with self.assertRaisesRegex(ValueError, re.escape(llm.BLOCK_MSG)):
            timeline._scene_cast(project, project.scenes[0])
        with self.assertRaisesRegex(ValueError, re.escape(llm.BLOCK_MSG)):
            timeline.enqueue(project, timeline.PREPARE)
        ok, _, _, error = timeline.prepare_job({"project": project.id, "key": project.key}, lambda *_: None)
        self.assertFalse(ok)
        self.assertIn(llm.BLOCK_MSG, error)
        self.assertEqual((self.video.calls, self.sound.prepared, self.sound.batches), ([], [], []))
        allowed = self.save([timeline.Scene("Ada waits.", 3, lines=(Cue("Hello.", "ada"),),
                                            effects=(Effect("in a bikini", 1),))], project)
        self.assertEqual([person["slug"] for person in timeline._scene_cast(allowed, allowed.scenes[0])], ["ada"])

    def test_effective_text_and_cues(self):
        project = self.save([timeline.Scene("Ada waits.", 5)])
        talk = timeline.Scene("Ada and Kit sit at a table.", 5,
                              lines=(Cue("Hello.", "kit"), Cue("Hi.", "ada"), Cue("Bye.", "kit"), Cue("Psst.", "@zoe")))
        picture, cues = timeline.effective(project, talk)
        self.assertEqual(picture, "Ada and Kit sit at a table. Kit talks. Ada talks, natural lip movement")
        self.assertEqual(cues, talk.lines)
        alone = timeline.Scene("Ada waves", 5, cast=("ada",), lines=(Cue("Hello."),))
        self.assertEqual(timeline.effective(project, alone)[0], "Ada waves. Ada talks, natural lip movement")
        quiet = timeline.Scene("  Ada waits by the @hanna door.  ", 5)
        self.assertEqual(timeline.effective(project, quiet), (quiet.prompt, ()))
        legacy = timeline.Scene("Ada opens the door and says “Come in.” A man answers “Thanks.”", 5)
        picture, cues = timeline.effective(project, legacy)
        self.assertFalse(timeline.dialogue.quotes(picture))
        self.assertTrue(picture.endswith(", natural lip movement"))
        self.assertEqual([(cue.text, cue.speaker, cue.start) for cue in cues],
                         [("Come in.", "ada", None), ("Thanks.", "@marek", None)])
        assigned = timeline.Scene("They turn. She says “Wait.” He says “Ready.”", 5, cast=("ada", "kit"),
                                  speakers=("kit", ""))
        # Legacy speakers win; a line nobody was assigned keeps the voice the old pipeline picked for it.
        self.assertEqual([cue.speaker for cue in timeline.effective(project, assigned)[1]], ["kit", "@marek"])

    def test_an_unnamed_line_in_the_leads_own_voice_goes_to_the_lead(self):
        """A character's own voice has no "@" name. The old pipeline gives an unnamed line of the lead's gender the
        lead's voice; with an own voice that line now goes to the lead (it once failed: "Choose who says the line")."""
        sample = self.root / "voices" / "ada.wav"
        audio.write_wav(sample, _tone(1.0, 300))
        own = voices.Voice("Ada", "character:ada", "female", str(sample))
        self.profiles["ada"]["voice"] = "character:ada"
        project = self.save([timeline.Scene("Ada waits.", 5)])
        legacy = timeline.Scene("Ada waits by the door. A woman says “Come in.” A man answers “Thanks.”", 5)
        with patch.object(characters, "own_voice", side_effect=lambda key: own if key == own.slug else None):
            cues = timeline.effective(project, legacy)[1]
            converted = timeline.quotes_to_cues(project, legacy)
            speakers = scene_sound.resolve_speakers(cues, [self.profiles["ada"]], self.pool)
        self.assertEqual([(cue.text, cue.speaker) for cue in cues], [("Come in.", "ada"), ("Thanks.", "@marek")])
        self.assertEqual(converted.lines, cues)
        self.assertEqual([speaker.voice for speaker in speakers], ["character:ada", "marek"])

    def test_quotes_become_dialogue_lines_with_their_speakers(self):
        project = self.save([timeline.Scene("Ada waits.", 5)])
        scene = timeline.Scene("They turn to the door. She says “Wait.” He says “Ready.”", 5,
                               cast=("ada", "kit"), speakers=("kit", "ada"))
        converted = timeline.quotes_to_cues(project, scene)
        self.assertEqual(converted.prompt, "They turn to the door. She talks. He talks.")
        self.assertEqual(converted.lines, (Cue("Wait.", "kit"), Cue("Ready.", "ada")))
        self.assertEqual((converted.speakers, converted.cast, converted.seconds), ((), ("ada", "kit"), 5))
        self.assertEqual(timeline.quotes_to_cues(project, converted), converted)
        saved = self.save([converted], project)
        self.assertEqual(saved.scenes[0].lines, converted.lines)
        quoted = timeline.quotes_to_cues(project, timeline.Scene("A woman smiles: “Open”. A man waves.", 3))
        self.assertEqual((quoted.prompt, quoted.lines), ("A woman smiles. A man waves.", (Cue("Open", "@hanna"),)))
        # a sign's text is shown in the picture, never spoken: it stays where it is (2026-11-16)
        sign = timeline.Scene("A sign: “Open”. Ada smiles.", 3)
        self.assertEqual(timeline.quotes_to_cues(project, sign), sign)
        mixed = timeline.quotes_to_cues(project, timeline.Scene(
            "A screen displays the exact text “CRYPTO NEWS”. Ada says “Crypto news!”", 3, cast=("ada",)))
        self.assertEqual(mixed.prompt, "A screen displays the exact text “CRYPTO NEWS”. Ada talks.")
        self.assertEqual(mixed.lines, (Cue("Crypto news!", "ada"),))


class ReferenceTests(Stage4Case):
    def test_store_reference_validates_and_names_the_file_by_its_content(self):
        cutout = self.image("cutout.png", mode="RGBA", color=(255, 0, 0, 0))
        name = timeline.store_reference(cutout)
        self.assertTrue(scene_parts.is_reference(name))
        stored = timeline.reference_path(name)
        self.assertEqual(stored, (timeline.DIR / "refs" / f"{name}.png").resolve())
        self.assertEqual(timeline._sha(stored), name)
        with Image.open(stored) as image:
            self.assertEqual((image.format, image.mode, image.size), ("PNG", "RGB", (64, 48)))
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))  # transparency becomes white
        self.assertEqual(timeline.store_reference(cutout), name)
        wide = timeline.store_reference(self.image("wide.png", size=(4096, 16)))
        with Image.open(timeline.reference_path(wide)) as image:
            self.assertEqual(image.size, (timeline.REFERENCE_SIDE, 8))

        text = self.root / "notes.png"
        text.write_text("not a picture", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not an image"):
            timeline.store_reference(text)
        truncated = self.root / "truncated.png"
        truncated.write_bytes(cutout.read_bytes()[:60])
        with self.assertRaisesRegex(ValueError, "not an image"):
            timeline.store_reference(truncated)
        with self.assertRaisesRegex(ValueError, "Could not read"):
            timeline.store_reference(self.root / "missing.png")
        with self.assertRaisesRegex(ValueError, "Could not read"):
            timeline.store_reference(self.root)
        with patch.object(timeline, "MAX_REFERENCE_BYTES", 10), self.assertRaisesRegex(ValueError, "smaller than"):
            timeline.store_reference(cutout)
        with patch.object(timeline, "MAX_REFERENCE_PIXELS", 1000), self.assertRaisesRegex(ValueError, "megapixels"):
            timeline.store_reference(cutout)

        for wrong in ("../" * 4 + "x", name.upper(), None, 7, "0" * 64):
            self.assertIsNone(timeline.reference_path(wrong))
        self.assertEqual(sorted(path.name for path in (timeline.DIR / "refs").iterdir()),
                         sorted([f"{name}.png", f"{wide}.png"]))

    def test_references_reach_generation_and_a_missing_one_blocks_it(self):
        reference = timeline.store_reference(self.image("cup.png"))
        project = self.save([timeline.Scene("Ada holds the cup [1].", 1, references=(reference,))])
        self.prepare(project)
        self.assertEqual(self.video.calls[0]["ref_images"], [str(timeline.reference_path(reference))])
        self.assertEqual(self.video.calls[0]["prompt"], "Ada holds the cup [1].")
        timeline.reference_path(reference).unlink()
        timeline._MEMO.clear()
        issues = timeline.scene_issues(project, 0)
        self.assertIn("Reference image 1 is missing. Add it again or remove it.",
                      [issue.text for issue in issues if issue.level == scene_parts.ERROR])
        with self.assertRaisesRegex(ValueError, "Scene 1: Reference image 1 is missing"):
            timeline.enqueue(project, timeline.PREPARE, scene_index=0)


class SoundTests(Stage4Case):
    def test_scene_issues_report_sources_timing_and_stay_cheap(self):
        project = self.save([
            timeline.Scene("Ada and Kit talk in the cafe.", 5,
                           lines=(Cue("Hello, how was the trip?"), Cue("Long.", "kit"), Cue("Hm.", "@nobody"),
                                  Cue("Sure.", "kit", 4.6))),
            timeline.Scene("Ada waves.", 5, cast=("ada",), lines=(Cue("Bye!"),), background="soft rain"),
        ])
        with patch.object(timeline, "_scene_cast", wraps=timeline._scene_cast) as cast_calls:
            issues = timeline.scene_issues(project, 0)
            self.assertEqual(timeline.scene_issues(project, 0), issues)
        self.assertEqual(cast_calls.call_count, 1)
        texts = [(issue.level, issue.text) for issue in issues]
        self.assertIn((scene_parts.ERROR, "Line 1: choose who says it."), texts)
        self.assertIn((scene_parts.ERROR, "Line 3: no voice named @nobody."), texts)
        self.assertIn((scene_parts.WARNING, "Line 4 should end around 5.4 s, after the scene ends at 5 s, so its end "
                                            "is cut off. Make the scene longer to hear all of it."), texts)
        self.assertEqual(issues[0].level, scene_parts.ERROR)
        self.assertTrue(any(level == scene_parts.INFO and text.startswith("Background sound from the picture: ")
                            and "cafe" in text for level, text in texts))

        second = timeline.scene_issues(project, 1)
        self.assertEqual([issue.level for issue in second], [])
        talk = timeline.Scene("Ada and Kit talk in the cafe.", 5, lines=(Cue("Long.", "kit"),))
        project = self.save([talk, project.scenes[1]], project)
        texts = [(issue.level, issue.text) for issue in timeline.scene_issues(project, 0)]
        self.assertIn((scene_parts.INFO, "Kit has no voice of their own, so their lines use Marek."), texts)
        self.assertFalse([text for level, text in texts if level == scene_parts.ERROR])

        # Models that are not installed yet are downloaded on first use: news, never a reason to refuse.
        self.speech.return_value = self.ambience.return_value = False
        timeline._MEMO.clear()
        issues = timeline.scene_issues(project, 0)
        self.assertFalse([issue for issue in issues if issue.level == scene_parts.ERROR])
        self.assertIn((scene_parts.INFO, "VoxCPM2 (speech) will be downloaded on first use."),
                      [(issue.level, issue.text) for issue in issues])
        self.assertIn((scene_parts.INFO, "EzAudio-XL (background sound and effects) will be downloaded on "
                                         "first use."), [(issue.level, issue.text) for issue in issues])
        job = timeline.enqueue(project, timeline.SOUND, scene_index=0)
        self.assertTrue(self.run_job(job)[0])
        self.ensure.assert_called_once()  # downloaded before any sound was made
        self.assertEqual(self.ensure.call_args.args[0], [timeline.tts.MODEL_ID, timeline.sound.MODEL_ID])
        silent = self.save(project.scenes, project, sound=False)
        self.assertEqual([(issue.level, issue.text) for issue in timeline.scene_issues(silent, 0)],
                         [(scene_parts.WARNING, "Sound is off for this project, so the lines, background and "
                                                "effects are not heard.")])

    def test_sound_preview_measures_lines_and_mixes_with_the_scene_levels(self):
        scene = timeline.Scene("Ada waves at the window.", 3, cast=("ada",),
                               lines=(Cue("Hello."), Cue("Look at the rain over there!", "", 1.95)),
                               effects=(Effect("door knock", 1),))
        project = self.save([scene])
        self.assertIsNone(timeline.sound_preview(project, 0))
        self.assertIsNone(timeline.preview_mix(project, 0))
        warnings = [issue.text for issue in timeline.scene_issues(project, 0) if issue.level != scene_parts.INFO]
        self.assertEqual(warnings, ["Line 2 should end around 4.5 s, after the scene ends at 3 s, so its end is cut "
                                    "off. Make the scene longer to hear all of it."])
        job = timeline.enqueue(project, timeline.SOUND, scene_index=0)
        self.assertEqual((job.params["scene_index"], job.params["seconds"], job.units()), (0, 3, 3.0))
        self.assertEqual(job.estimate(), 3 * queue_manager.rate(timeline.SOUND))
        with self.assertRaisesRegex(ValueError, "already has a job"):
            timeline.enqueue(project, timeline.SOUND, scene_index=0)
        ok, _, notes, error = self.run_job(job)
        self.assertTrue(ok, error)
        self.assertEqual(notes, ["Scene 1 sound preview ready."])
        [made] = self.sound.prepared
        self.assertEqual((made["cues"], made["voice"], made["background"]), (scene.lines, True, True))
        self.assertEqual(made["folder"].parent, timeline.DIR / project.id / "sound")
        preview = timeline.sound_preview(project, 0)
        self.assertEqual((preview.texts, preview.timings),
                         (("Hello.", "Look at the rain over there!"), ((0.3, 1.3), (1.95, 2.95))))
        # Measured lines replace the estimates: the second line really ends inside the scene.
        self.assertFalse([issue for issue in timeline.scene_issues(project, 0) if issue.level != scene_parts.INFO])

        with patch.object(scene_sound, "preview_mix", wraps=scene_sound.preview_mix) as mixing:
            first = timeline.preview_mix(project, 0)
            self.assertEqual(timeline.preview_mix(project, 0), first)
        self.assertEqual(mixing.call_count, 1)
        self.assertEqual(first.parent, timeline.DIR / project.id / "previews")
        pcm, rate = audio.read(first)
        self.assertEqual((pcm.shape[1], rate), (3 * RATE, RATE))
        with self.assertRaisesRegex(ValueError, "sound preview of scene 1 is ready"):
            timeline.enqueue(project, timeline.SOUND, scene_index=0)  # nothing to make: no job
        self.assertEqual(self.run_job(job)[0], True)  # a job that finds it ready reuses the folder
        self.assertEqual(len(self.sound.prepared), 1)

        quieter = self.save([replace(scene, voice_db=-12.0, background_db=3.0)], project)
        self.assertEqual(timeline.sound_preview(quieter, 0).key, preview.key)  # levels do not change the sound
        second = timeline.preview_mix(quieter, 0)
        self.assertNotEqual(second, first)
        self.assertNotEqual(timeline._sha(second), timeline._sha(first))
        changed = self.save([replace(scene, effects=(Effect("door knock", 2),))], quieter)
        self.assertIsNone(timeline.sound_preview(changed, 0))
        with self.assertRaisesRegex(ValueError, "Choose a scene"):
            timeline.enqueue(changed, timeline.SOUND)
        with self.assertRaisesRegex(ValueError, "Turn on sound"):
            timeline.enqueue(self.save(changed.scenes, changed, sound=False), timeline.SOUND, scene_index=0)

    def test_leveled_clip_keeps_the_picture_and_is_made_once(self):
        scene = timeline.Scene("Ada waves.", 1, cast=("ada",), lines=(Cue("Hi."),))
        project = self.save([scene])
        [key] = self.prepare(project)
        clip = timeline.asset(project, key)["path"]
        # At 0 dB / 0 dB too the scene plays the export's mix of its stems, not the clip's own mix.
        with patch.object(audio, "scene_mix", wraps=audio.scene_mix) as mixing:
            flat = timeline.leveled_clip(project, 0)
            self.assertEqual(timeline.leveled_clip(project, 0), flat)
        self.assertEqual(mixing.call_count, 1)
        self.assertEqual(mixing.call_args.args[2:], (1, 0.0, 0.0))
        self.assertNotEqual(flat, clip)
        self.assertEqual(flat.parent, timeline.DIR / project.id / "previews")
        self.assertEqual(_video_packets(flat), _video_packets(clip))
        self.assertEqual(timeline.preview_mix(project, 0).suffix, ".wav")
        draft = replace(project, scenes=(replace(scene, voice_db=-6.0),))
        with patch.object(audio, "scene_mix", wraps=audio.scene_mix) as mixing:
            leveled = timeline.leveled_clip(draft, 0)  # an unsaved level change plays without saving first
            self.assertEqual(timeline.leveled_clip(draft, 0), leveled)
        self.assertEqual(mixing.call_count, 1)
        self.assertEqual(mixing.call_args.args[2:], (1, -6.0, 0.0))
        self.assertEqual(leveled.parent, timeline.DIR / project.id / "previews")
        self.assertEqual(_video_packets(leveled), _video_packets(clip))
        with av.open(str(leveled)) as container:
            self.assertEqual(len(container.streams.audio), 1)
        with av.open(str(leveled)) as container:
            self.assertEqual(container.metadata.get("comment"), video.AI_COMMENT)
        self.assertIsNone(timeline.leveled_clip(replace(draft, id=""), 0))
        # Without separate stems (or with sound off) the clip itself plays.
        silent = self.save([timeline.Scene("Kit waits.", 1)], sound=False, title="Silent")
        [silent_key] = self.prepare(silent)
        self.assertEqual(timeline.leveled_clip(silent, 0), timeline.asset(silent, silent_key)["path"])


class TakeAndFailureTests(Stage4Case):
    def test_takes_are_listed_and_swapped(self):
        project = self.save([timeline.Scene("A cup on a table", 1), timeline.Scene("The cup turns", 1)], sound=False)
        first, second = self.prepare(project)
        retake = self.prepare(project, scene_index=0, take="b" * 32)[0]
        self.assertNotIn(retake, (first, second))
        listed = timeline.takes(project, 0)
        self.assertEqual([(take["key"], take["current"], take["anchor_ok"]) for take in listed],
                         [(retake, True, True), (first, False, True)])
        self.assertTrue(listed[0]["label"].startswith("Take 2 · "))
        self.assertTrue(listed[1]["label"].startswith("Take 1 · "))
        self.assertGreaterEqual(listed[0]["created"], listed[1]["created"])
        # Scene 2 was made from the first take's last frame.
        self.assertEqual([(take["key"], take["anchor_ok"]) for take in timeline.takes(project, 1)], [(second, False)])
        record = timeline.asset(project, second)
        self.assertEqual(record["anchor_sha256"], timeline.asset(project, first)["sha256"])
        self.assertEqual(record["picture_sha256"], record["sha256"])
        self.assertEqual(timeline.asset(project, retake)["take"], "b" * 32)

        timeline.use_take(project, 0, first)
        self.assertEqual(timeline.prepared(project)["assets"], [first, second])
        self.assertTrue(timeline.takes(project, 1)[0]["anchor_ok"])
        self.assertEqual([take["current"] for take in timeline.takes(project, 0)], [False, True])
        for wrong in (second, "c" * 64, None):
            with self.assertRaisesRegex(ValueError, "Choose a take"):
                timeline.use_take(project, 0, wrong)
        job = timeline.enqueue(project, timeline.MERGE)
        with self.assertRaisesRegex(ValueError, "Wait for this project"):
            timeline.use_take(project, 0, retake)
        job.status = queue_manager.DONE
        edited = self.save([timeline.Scene("A cup on a red table", 1), project.scenes[1]], project, sound=False)
        with self.assertRaisesRegex(ValueError, "timeline changed"):
            timeline.use_take(project, 0, retake)
        self.assertEqual(timeline.takes(edited, 0), [])

    def test_a_failure_is_recorded_until_its_scene_succeeds(self):
        project = self.save([timeline.Scene("A cup", 1), timeline.Scene("A plate", 1, cut=True)], sound=False)
        params = {"project": project.id, "key": project.key}
        broken = video_gen.VideoOutput(False, error="The GPU ran out of memory.")
        with patch.object(video_gen, "generate", side_effect=[self.video(
                "A cup", 1, "", scene_dir=self.root / "work", sound_on=False), broken]):
            ok, _, _, error = timeline.prepare_job(params, lambda *_: None)
        self.assertFalse(ok)
        self.assertEqual(error, "Scene 2: The GPU ran out of memory.")
        self.assertEqual(timeline.failure(project), {"scene": 1, "error": error, "kind": timeline.PREPARE})
        self.assertIsNone(timeline.failure(replace(project, scenes=project.scenes[:1])))
        self.assertTrue(timeline.prepare_job({**params, "scene_index": 0}, lambda *_: None)[0])
        self.assertEqual(timeline.failure(project)["scene"], 1)  # another scene's success keeps it
        self.assertTrue(timeline.prepare_job({**params, "scene_index": 1}, lambda *_: None)[0])
        self.assertIsNone(timeline.failure(project))

        sounding = self.save([timeline.Scene("A cup", 1), timeline.Scene("A plate", 1, cut=True)], project)
        self.sound.fail = "Line 1 ends at 6.0 s, after the scene ends at 5 s."
        ok, _, _, error = self.run_job(timeline.enqueue(sounding, timeline.SOUND, scene_index=1))
        self.assertFalse(ok)
        self.assertEqual(timeline.failure(sounding), {"scene": 1, "error": "Scene 2: " + self.sound.fail,
                                                      "kind": timeline.SOUND})
        self.sound.fail = None
        self.assertTrue(self.run_job(timeline.enqueue(sounding, timeline.SOUND, scene_index=0))[0])
        self.assertIsNotNone(timeline.failure(sounding))
        self.assertTrue(self.run_job(timeline.enqueue(sounding, timeline.SOUND, scene_index=1))[0])
        self.assertIsNone(timeline.failure(sounding))


class ProductionTests(Stage4Case):
    def test_export_job_prepares_joins_and_records_the_movie(self):
        project = self.save([timeline.Scene("Ada sets a cup down", 1),
                             timeline.Scene("Kit picks it up", 2, lines=(Cue("Thanks.", "kit"),))])
        progress = []
        with self.assertRaisesRegex(ValueError, "^Scene 1: write a description"):
            timeline.enqueue(self.save([timeline.Scene("", 1)], title="Empty"), timeline.PRODUCE)
        job = timeline.enqueue(project, timeline.PRODUCE)
        self.assertEqual(job.label, "Export movie · Stage four")
        self.assertGreater(job.params["work_seconds"], 3)
        self.assertEqual(job.units(), job.params["work_seconds"])
        self.assertEqual(queue_manager.Job(99, timeline.PRODUCE, "Older export", {"seconds": 60}).units(), 60.0)
        self.assertIn(timeline.PRODUCE, timeline.JOB_KINDS)
        self.assertIs(queue_manager._RUNNERS[timeline.PRODUCE], timeline.produce_job)
        self.assertIs(queue_manager._RUNNERS[timeline.SOUND], timeline.sound_job)
        with patch.object(timeline.llm, "inspect_review", return_value={"verdict": "match", "reason": "Same table."}):
            ok, files, notes, error = timeline.produce_job(job.params, lambda fraction, _: progress.append(fraction))
        job.status = queue_manager.DONE
        self.assertTrue(ok, error)
        self.assertEqual(len(self.video.calls), 2)
        self.assertEqual(progress, sorted(progress))
        self.assertTrue(0 <= progress[0] and progress[-1] == 1)
        self.assertIn("Scenes 1 → 2 · Same table.", notes)
        movie = Path(files[0])
        self.assertEqual(timeline.exported(project), movie)
        self.assertTrue(studio.is_recorded(str(movie)))
        [entry] = history.recent()
        self.assertEqual((entry["mode"], entry["files"], entry["project"]), ("video_timeline", [str(movie)], project.id))
        self.assertEqual(video.probe(movie).duration, 3)
        self.assertIsNone(timeline.failure(project))
        # A current export is not queued again; a job that finds it ready only hands it back.
        with self.assertRaisesRegex(ValueError, "already exported"):
            timeline.enqueue(project, timeline.PRODUCE)
        with patch.object(video_gen, "generate", side_effect=AssertionError("Already made")):
            self.assertEqual(self.run_job(job)[1], [str(movie)])
        self.assertEqual(len(history.recent()), 1)

    def test_480p_clips_of_every_aspect_ratio_join_and_export(self):
        """The one video mode's clips (video_gen.sizes) go through the joined edit and the export at
        their own frame size, for every aspect ratio the projects offer."""
        review = {"verdict": "match", "reason": "Same room."}
        for aspect, ratio in video_gen.ASPECTS.items():
            with self.subTest(aspect=aspect):
                self.video.size = frame = video_gen.sizes(ratio)[0]
                project = timeline.save(None, None, f"Aspect {aspect}", (timeline.Scene("Ada sets a cup down", 1),
                                        timeline.Scene("Ada lifts the cup", 1)), aspect, True, [],
                                        universe="crew", universe_members=("ada", "kit"))
                with patch.object(timeline.llm, "inspect_review", return_value=review):
                    ok, files, _, error = self.run_job(timeline.enqueue(project, timeline.PRODUCE))
                self.assertTrue(ok, error)
                joined = timeline.merged_clip(project, timeline.prepared(project), verify=True)
                for path in (joined, Path(files[0])):
                    info = video.probe(path)
                    self.assertEqual((info.width, info.height, info.fps), (*frame, video_gen.FPS), path.name)
                self.assertEqual(video.probe(Path(files[0])).duration, 2)
                self.assertEqual(history.recent()[0]["width"], frame[0])

    def test_an_export_that_fails_on_a_scene_names_it(self):
        project = self.save([timeline.Scene("A cup", 1), timeline.Scene("A plate", 1, cut=True)])
        with patch.object(video_gen, "generate", side_effect=[self.video("A cup", 1, "", scene_dir=self.root / "w"),
                                                              video_gen.VideoOutput(False, error="Wan failed.")]):
            ok, files, _, error = self.run_job(timeline.enqueue(project, timeline.PRODUCE))
        self.assertEqual((ok, files, error), (False, [], "Scene 2: Wan failed."))
        self.assertEqual(timeline.failure(project), {"scene": 1, "error": error, "kind": timeline.PRODUCE})
        self.assertEqual(len(timeline.prepared(project)["assets"]), 1)  # the finished scene is kept
        with patch.object(timeline.llm, "inspect_review", side_effect=AssertionError("A cut needs no check")):
            self.assertTrue(self.run_job(timeline.enqueue(project, timeline.PRODUCE))[0])
        self.assertIsNone(timeline.failure(project))
        self.assertEqual(len(self.video.calls), 2)

    def test_a_level_change_rejoins_without_another_review_and_reaches_the_export(self):
        project = self.save([timeline.Scene("A cup", 1), timeline.Scene("Same cup", 1),
                             timeline.Scene("Different room", 1, cut=True)])
        keys = self.prepare(project)

        def bridge(_first, _last, output, frames, fps):
            with video.Writer(output, fps, 48, 32) as writer:
                for _ in range(frames):
                    writer.write(np.full((32, 48, 3), (0, 180, 0), dtype=np.uint8))

        with (patch.object(timeline.llm, "inspect_review", return_value={"verdict": "bridge", "reason": "Small jump."}) as review,
              patch.object(timeline.engine_runtime, "transition_frames", side_effect=bridge) as smoothing,
              patch.object(timeline.media, "join_timeline", wraps=video.join_timeline) as join):
            ok, _, notes, error = timeline.merge_job({"project": project.id, "key": project.key}, lambda *_: None)
        self.assertTrue(ok, error)
        self.assertEqual((review.call_count, smoothing.call_count), (1, 1))
        self.assertIsNone(join.call_args.kwargs["sound_levels"])
        self.assertIn("opening frames smoothed", notes[0])
        stored = json.loads((timeline.DIR / project.id / "edits" / "joins.json").read_text(encoding="utf-8"))
        self.assertEqual([entry["verdict"] for entry in stored["joins"].values()], ["bridge"])

        scenes = list(project.scenes)
        scenes[1] = replace(scenes[1], voice_db=-6.0, background_db=3.0)
        leveled = self.save(scenes, project)
        self.assertEqual(timeline.prepared(leveled)["assets"], keys)  # every clip carries over
        self.assertIsNone(timeline.merged_clip(leveled, timeline.prepared(leveled)))
        with (patch.object(timeline.llm, "inspect_review", side_effect=AssertionError("Reviewed again")),
              patch.object(timeline.engine_runtime, "transition_frames", side_effect=AssertionError("Smoothed again")),
              patch.object(timeline.media, "join_timeline", wraps=video.join_timeline) as join):
            ok, _, again, error = timeline.merge_job({"project": leveled.id, "key": leveled.key}, lambda *_: None)
            self.assertTrue(ok, error)
            self.assertEqual(again, notes)
            self.assertEqual(join.call_args.kwargs["sound_levels"], [(0.0, 0.0), (-6.0, 3.0), (0.0, 0.0)])
            ok, files, _, error = timeline.render_job({"project": leveled.id, "key": leveled.key,
                                                       "require_merge": True}, lambda *_: None)
        self.assertTrue(ok, error)
        self.assertEqual(video.probe(files[0]).duration, 3)
        self.assertEqual(len(self.video.calls), 3)

    def test_sound_only_change_reuses_the_picture_and_the_next_scene(self):
        project = self.save([timeline.Scene("A cup on a table", 1), timeline.Scene("The cup turns", 1)])
        first, second = self.prepare(project)
        original = timeline.asset(project, first)
        rainy = replace(project.scenes[0], background="heavy rain", effects=(Effect("thunder", 0.2, 0.5),))
        edited = self.save([rainy, project.scenes[1]], project)
        self.assertEqual(timeline.prepared(edited)["assets"], [None, second])
        before = len(self.sound.prepared)  # each generated scene got its sound first (one sound per scene)
        self.assertEqual(before, 2)
        with (patch.object(video_gen, "generate", side_effect=AssertionError("The picture did not change")),
              patch.object(scene_sound, "clip_sound", wraps=scene_sound.clip_sound) as mixing):
            refreshed_key, kept = self.prepare(edited)
        self.assertEqual(kept, second)
        # The refreshed clip carries its own mix made as video_gen mixes a new clip (not the export mix).
        self.assertEqual(mixing.call_args.args[0], original["path"])
        [made] = self.sound.prepared[before:]
        self.assertEqual((made["cues"], made["voice"], made["effects"]), ((), False, rainy.effects))
        self.assertIn("heavy rain", made["prompt"])
        refreshed = timeline.asset(edited, refreshed_key)
        self.assertEqual(refreshed["picture_sha256"], original["sha256"])
        self.assertNotEqual(refreshed["sha256"], original["sha256"])
        self.assertEqual(_video_packets(refreshed["path"]), _video_packets(original["path"]))
        self.assertEqual(timeline._sha(refreshed["audio_sources"][0]), timeline._sha(made["folder"] / "background.wav"))
        self.assertEqual(refreshed["picture_key"], timeline.asset(project, first)["picture_key"])
        # Continuity follows the picture: scene 2 computes the same key and keeps its clip.
        with patch.object(video_gen, "generate", side_effect=AssertionError("Scene 2 did not change")):
            self.assertEqual(self.prepare(edited, scene_index=1), [refreshed_key, second])
        # Another sound change finds the picture through picture_key even without the first clip.
        for path in (timeline.DIR / project.id / "clips").glob(first + ".*"):
            path.unlink()
        windy = self.save([replace(rainy, background="wind"), project.scenes[1]], edited)
        with patch.object(video_gen, "generate", side_effect=AssertionError("The picture did not change")):
            again = self.prepare(windy)
        self.assertEqual(timeline.asset(windy, again[0])["picture_sha256"], original["sha256"])
        self.assertEqual(len(self.video.calls), 2)


class AcceptanceTests(Stage4Case):
    def test_change_a_line_and_an_effect_listen_and_rejoin_with_the_other_shots(self):
        scenes = [timeline.Scene("Ada sets two cups on the table.", 1),
                  timeline.Scene("Ada and Kit sit down.", 4, lines=(Cue("Coffee?", "ada"), Cue("Please.", "kit")),
                                 effects=(Effect("cups clink", 0.5, 0.5),)),
                  timeline.Scene("Rain runs down the window.", 1, cut=True)]
        project = self.save(scenes)
        first, middle, last = self.prepare(project)
        self.assertEqual(len(self.video.calls), 3)

        changed = replace(scenes[1], lines=(Cue("Coffee?", "ada"), Cue("Yes, please.", "kit")),
                                   effects=(Effect("cups clink", 1.0, 0.5),))
        edited = self.save([scenes[0], changed, scenes[2]], project)
        state = timeline.prepared(edited)
        self.assertEqual(state["assets"], [first, None, last])  # scenes 1 and 3 keep their clips
        self.assertEqual(timeline.outdated(edited), [1])
        self.assertIsNone(timeline.exported(edited))
        self.assertFalse([issue for issue in timeline.scene_issues(edited, 1) if issue.level == scene_parts.ERROR])

        ok, _, _, error = self.run_job(timeline.enqueue(edited, timeline.SOUND, scene_index=1))
        self.assertTrue(ok, error)
        preview = timeline.sound_preview(edited, 1)
        self.assertEqual(preview.texts, ("Coffee?", "Yes, please."))
        self.assertTrue(timeline.preview_mix(edited, 1).is_file())
        self.assertEqual([speaker.voice for speaker in preview.speakers], ["hanna", "marek"])

        ok, _, _, error = self.run_job(timeline.enqueue(edited, timeline.PREPARE, scene_index=1))
        self.assertTrue(ok, error)
        self.assertEqual(len(self.video.calls), 4)
        self.assertEqual(self.video.calls[-1]["prepared_sound"].key, preview.key)  # the audition is what plays
        keys = timeline.prepared(edited)["assets"]
        self.assertEqual((keys[0], keys[2]), (first, last))
        self.assertNotIn(keys[1], (None, middle))
        with patch.object(timeline.llm, "inspect_review", return_value={"verdict": "match", "reason": "Same room."}):
            ok, _, notes, error = self.run_job(timeline.enqueue(edited, timeline.MERGE))
        self.assertTrue(ok, error)
        state = timeline.prepared(edited)
        self.assertIsNotNone(timeline.merged_clip(edited, state, verify=True))
        self.assertEqual(state["merged"]["assets"], [first, keys[1], last])
        self.assertEqual(len(self.video.calls), 4)


class OneSoundPerSceneTests(Stage4Case):
    def test_generation_makes_the_scene_sound_first_and_the_preview_and_takes_share_it(self):
        scene = timeline.Scene("Ada waves at the window.", 2, cast=("ada",), lines=(Cue("Hello."),),
                               effects=(Effect("door knock", 1.0, 0.5),))
        project = self.save([scene])
        order = []
        self.ensure.side_effect = lambda ids, *_: order.append(("ensure", tuple(ids)))
        self.speech.return_value = False  # VoxCPM2 is not installed yet: it is downloaded first, not refused
        made = self.sound.prepare

        def sound_first(*args, **kwargs):
            order.append("sound")
            return made(*args, **kwargs)

        def video(*args, **kwargs):
            order.append("video")
            return self.video(*args, **kwargs)

        with patch.object(scene_sound, "prepare", side_effect=sound_first), \
                patch.object(video_gen, "generate", side_effect=video):
            [key] = self.prepare(project)
        self.assertEqual(order, [("ensure", (timeline.tts.MODEL_ID,)), "sound", "video"])
        [sound] = self.sound.prepared
        inputs = timeline._sound_inputs(project, scene, [self.profiles["ada"]], index=0)
        self.assertEqual((sound["folder"], sound["seed"]), (timeline.DIR / project.id / "sound" / inputs.key,
                                                            inputs.seed))
        self.assertEqual(self.video.calls[0]["prepared_sound"].key, inputs.key)
        record = timeline.asset(project, key)
        self.assertEqual(record["sound_key"], inputs.key)
        # the clip's sound is the scene's sound preview: there is nothing left to prepare
        self.assertEqual(timeline.sound_preview(project, 0).key, inputs.key)
        self.assertEqual(timeline._scene_clip(project, 0)["sound_key"], timeline.sound_preview(project, 0).key)
        with self.assertRaisesRegex(ValueError, "sound preview of scene 1 is ready"):
            timeline.enqueue(project, timeline.SOUND, scene_index=0)
        # a new take speaks with the same sound
        self.prepare(project, scene_index=0, take="b" * 32)
        self.assertEqual(len(self.sound.prepared), 1)
        self.assertEqual(self.video.calls[-1]["prepared_sound"].key, inputs.key)

    def test_a_film_job_makes_every_scene_sound_first_in_one_speech_batch(self):
        """A job that makes several scenes makes their sounds before their pictures: every line in one VoxCPM2 batch
        (a silent scene only gets its background) and the backgrounds in one EzAudio session; each scene then finds
        its own sound. A scene made alone synthesizes its own lines."""
        scenes = [timeline.Scene("Ada waves.", 2, cast=("ada",), lines=(Cue("Hi."), Cue("Over here."))),
                  timeline.Scene("Ada walks closer.", 2, cast=("ada",)),
                  timeline.Scene("Ada smiles.", 2, cast=("ada",), lines=(Cue("Good to see you."),)),
                  timeline.Scene("Kit opens a door.", 2, cut=True, cast=("kit",), lines=(Cue("Who is it?"),)),
                  timeline.Scene("Kit steps out.", 2, cast=("kit",), lines=(Cue("Oh, it is you."),))]
        project = self.save(scenes)
        self.prepare(project)
        self.assertEqual(self.sound.batches, [[("Hi.", "Over here."), ("Good to see you.",), ("Who is it?",),
                                               ("Oh, it is you.",)]])
        made = {entry["cues"]: entry["lines"] for entry in self.sound.prepared}
        self.assertEqual(made[scenes[0].lines], ("scene1_line1.wav", "scene1_line2.wav"))
        self.assertEqual(made[scenes[2].lines], ("scene2_line1.wav",))  # handed over, not synthesized again
        self.assertEqual(made[scenes[4].lines], ("scene4_line1.wav",))
        self.assertEqual(len(self.sound.prepared), 5)  # the silent scene's background too
        self.assertEqual(len(self.video.calls), 5)
        for call, scene in zip(self.video.calls, scenes):  # each clip speaks its own scene's sound
            self.assertEqual(call["prepared_sound"].texts, tuple(cue.text for cue in scene.lines))

        edited = self.save([*scenes[:4], replace(scenes[4], lines=(Cue("Oh. It is you."),))], project)
        self.prepare(edited, scene_index=4)
        self.assertEqual(len(self.sound.batches), 1)  # one scene alone: its own lines, no shared batch
        self.assertIsNone(self.sound.prepared[-1]["lines"])

    def test_a_film_job_makes_the_sounds_a_window_of_scenes_at_a_time(self):
        """The sounds made ahead are kept in the scenes' sound folders, so a failed or paused job loses none; the
        window of SOUNDS_AHEAD_SCENES scenes only bounds how long the first picture waits."""
        texts = [f"Line {number}." for number in range(1, 7)]
        scenes = [timeline.Scene(f"Ada takes step {number}.", 2, cast=("ada",), lines=(Cue(text),))
                  for number, text in enumerate(texts, 1)]
        with patch.object(timeline, "SOUNDS_AHEAD_SCENES", 3):
            self.prepare(self.save(scenes))
        self.assertEqual(self.sound.batches, [[(text,) for text in texts[:3]], [(text,) for text in texts[3:]]])
        made = {entry["cues"]: entry["lines"] for entry in self.sound.prepared}
        self.assertEqual([made[scene.lines] is not None for scene in scenes], [True] * 6)  # every scene got its lines

    def test_a_scene_is_made_without_the_assistant(self):
        # Sound and picture need no vision model: a missing assistant runtime does not hold a scene back.
        project = self.save([timeline.Scene("Ada waves.", 1, cast=("ada",), lines=(Cue("Hi."),))])
        with patch.object(timeline.llm, "ensure", return_value="Install the assistant runtime.") as assistant,                 patch.object(timeline.llm, "inspect_frames", side_effect=AssertionError("A picture was reviewed")):
            ok, _, notes, error = timeline.prepare_job({"project": project.id, "key": project.key}, lambda *_: None)
        self.assertTrue(ok, error)
        self.assertNotIn(timeline.REVIEW_UNAVAILABLE, notes)
        assistant.assert_not_called()
        self.assertEqual((len(self.sound.prepared), len(self.video.calls)), (1, 1))

    def test_each_scene_gets_a_bed_of_its_own(self):
        street = timeline.Scene("Ada walks down a busy street.", 3, cast=("ada",), lines=(Cue("Hi."),))
        answer = replace(street, lines=(Cue("Bye."),))
        project = self.save([street, answer, street])  # the third scene is a duplicate of the first
        members = [self.profiles["ada"]]
        inputs = [timeline._sound_inputs(project, scene, members, index=index)
                  for index, scene in enumerate(project.scenes)]
        self.assertEqual(len({item.prompt for item in inputs}), 1)
        self.assertEqual(len({item.seed for item in inputs}), 3)
        self.assertEqual(len({item.key for item in inputs}), 3)
        beds = {timeline._sound_inputs(project, scene, members, voice=False, index=index).key
                for index, scene in enumerate(project.scenes)}
        self.assertEqual(len(beds), 3)  # the sound-only refresh of each scene gets its own bed too
        # a line edit keeps the scene's seed, so its other parts stay as they were heard
        edited = replace(project, scenes=(replace(street, lines=(Cue("Hello there."),)), *project.scenes[1:]))
        self.assertEqual(timeline._sound_inputs(edited, edited.scenes[0], members, index=0).seed, inputs[0].seed)
        for index in (0, 1):
            self.assertTrue(self.run_job(timeline.enqueue(project, timeline.SOUND, scene_index=index))[0])
        self.assertEqual(len({made["seed"] for made in self.sound.prepared}), 2)
        self.assertEqual(len({made["folder"] for made in self.sound.prepared}), 2)

    def test_the_scene_mix_plays_the_clips_sound_when_the_clip_was_made_from_another(self):
        scene = timeline.Scene("Ada waves.", 1, cast=("ada",), lines=(Cue("Hi."),))
        project = self.save([scene])
        with patch.object(timeline, "_scene_sound", return_value=None):  # a clip made before one sound per scene
            self.prepare(project)
        self.assertIsNone(timeline.sound_preview(project, 0))
        self.assertTrue(self.run_job(timeline.enqueue(project, timeline.SOUND, scene_index=0))[0])
        clip = timeline._scene_clip(project, 0)  # the clip has sound, but not the new preview's
        self.assertTrue(clip["audio_sources"])
        self.assertNotEqual(clip.get("sound_key"), timeline.sound_preview(project, 0).key)
        with patch.object(scene_sound, "mix_stems", wraps=scene_sound.mix_stems) as stems, \
                patch.object(scene_sound, "preview_mix", wraps=scene_sound.preview_mix) as audition:
            self.assertIsNotNone(timeline.preview_mix(project, 0))
        self.assertEqual((stems.call_count, audition.call_count), (1, 0))  # the export's sound, not the preview's

    def test_a_nearly_silent_background_is_reported(self):
        project = self.save([timeline.Scene("Rain on a window.", 2)])
        self.assertTrue(self.run_job(timeline.enqueue(project, timeline.SOUND, scene_index=0))[0])
        texts = [issue.text for issue in timeline.scene_issues(project, 0)]
        self.assertNotIn(timeline.NEARLY_SILENT, texts)
        preview = timeline.sound_preview(project, 0)
        audio.write_wav(preview.background, _tone(2.5, 220, 0.00012))  # what a sound generator sometimes returns
        timeline._MEMO.clear()
        issues = timeline.scene_issues(project, 0)
        self.assertIn((scene_parts.WARNING, timeline.NEARLY_SILENT), [(issue.level, issue.text) for issue in issues])
        # the stem of a clip is checked the same way (measured once per file version)
        clip_only = self.save([timeline.Scene("A quiet hall.", 1)], title="Hall")
        with patch.object(timeline, "_scene_sound", return_value=None):
            [key] = self.prepare(clip_only)
        self.assertNotIn(timeline.NEARLY_SILENT, [issue.text for issue in timeline.scene_issues(clip_only, 0)])
        audio.write_wav(timeline.asset(clip_only, key)["audio_sources"][0], _tone(1.5, 220, 0.00012))
        timeline._MEMO.clear()
        self.assertIn(timeline.NEARLY_SILENT, [issue.text for issue in timeline.scene_issues(clip_only, 0)])


class ProductionScriptTests(Stage4Case):
    """The hooks a production script uses: prepare_job with its own scene renderer (generate_scene) and a cache
    namespace, and a script that queues PREPARE until every scene has a clip and reads the 'Scene N:' errors."""

    def hook(self, calls, fail=None):
        def render(project, scene, members, seed, style, folder, cb, *args, **kwargs):
            calls.append({"args": (project, scene, members, seed, style, folder, cb) + args, "kwargs": kwargs})
            if fail and scene.prompt == fail:
                return video_gen.VideoOutput(False, error="Wan failed.")
            anchor = kwargs.get("anchor")
            return self.video(scene.prompt, scene.seconds, project.aspect, scene_dir=folder, sound_on=project.audio,
                              image_path=str(anchor) if anchor is not None else None)
        return render

    def test_a_cache_namespace_evaluates_every_scene_and_the_hook_keeps_its_call(self):
        project = self.save([timeline.Scene("Ada opens the door", 1), timeline.Scene("Ada looks back", 1)])
        params, calls = {"project": project.id, "key": project.key}, []
        for namespace, made in (("recut-v1", 2), ("recut-v1", 0), ("recut-v2", 2)):
            with self.subTest(namespace=namespace):
                before = len(calls)
                ok, _, _, error = timeline.prepare_job(params, lambda *_: None, generate_scene=self.hook(calls),
                                                       cache_namespace=namespace)
                self.assertTrue(ok, error)
                self.assertEqual(len(calls) - before, made)
        for call in calls:
            self.assertEqual(len(call["args"]), 7)
            self.assertEqual(set(call["kwargs"]), {"anchor"})  # no other keyword reaches the hook
        self.assertEqual(self.sound.prepared, [])  # the hook makes its own sound
        keys = timeline.prepared(project)["assets"]
        self.assertTrue(timeline.all_clips(project, keys))
        self.assertTrue(all(timeline.asset(project, key)["cache_namespace"] == "recut-v2" for key in keys))
        self.assertEqual(timeline.outdated(project), [])  # the editor does not flag a namespace's clips
        before = len(calls)
        self.assertTrue(timeline.prepare_job(params, lambda *_: None)[0])  # without a namespace: only what is missing
        self.assertEqual(len(calls), before)

    def test_every_scene_failure_starts_with_its_scene_number(self):
        project = self.save([timeline.Scene("Ada opens the door", 1), timeline.Scene("Ada looks back", 1)])
        params = {"project": project.id, "key": project.key}
        ok, _, _, error = timeline.prepare_job(params, lambda *_: None, generate_scene=self.hook([], "Ada looks back"),
                                               cache_namespace="recut")
        self.assertFalse(ok)
        self.assertEqual(error, "Scene 2: Wan failed.")
        self.assertEqual(re.search(r"Scene (\d+):", error, re.I)[1], "2")
        empty = self.save([timeline.Scene("Ada waits", 1), timeline.Scene("", 1)], title="Empty")
        ok, _, _, error = timeline.prepare_job({"project": empty.id, "key": empty.key}, lambda *_: None,
                                               cache_namespace="recut")
        self.assertFalse(ok)
        self.assertEqual(re.search(r"Scene (\d+):", error, re.I)[1], "2")
        with self.assertRaises(ValueError) as refused:
            timeline.enqueue(empty, timeline.PREPARE)
        self.assertEqual(re.search(r"Scene (\d+):", str(refused.exception), re.I)[1], "2")

    def test_the_production_loop_queues_until_every_scene_has_a_clip(self):
        project = self.save([timeline.Scene("Ada opens the door", 1), timeline.Scene("Ada looks back", 1)])
        rounds = 0
        while not timeline.all_clips(project, timeline.prepared(project)["assets"]):
            rounds += 1
            self.assertTrue(self.run_job(timeline.enqueue(project, timeline.PREPARE))[0])
        self.assertEqual(rounds, 1)


class FirstFilesTests(Stage4Case):
    """What a job reads ahead for a scene while something else computes (timeline._first_files): the starting
    frame's models for a shot with people, Z-Image's for a 2D shot it draws, else the video experts."""

    def test_each_kind_of_scene_names_the_files_it_reads_first(self):
        from app import styles
        from engine.api import WanFiles

        frame, drawn = [Path("text"), Path("transformer")], [Path("zimage")]
        experts = WanFiles(Path("umt5"), Path("high"), Path("low"), Path("vae"), "i2v")
        project = self.save([timeline.Scene("Ada opens the door", 2, cast=("ada",)),
                             timeline.Scene("Ada looks back", 2, cast=("ada",)),
                             timeline.Scene("A quiet street", 2, cut=True)])
        with patch.object(timeline.engine_runtime, "start_frame_files", return_value=frame) as start, \
                patch.object(timeline.engine_runtime, "drawn_frame_files", return_value=drawn), \
                patch.object(timeline.engine_runtime, "zimage_files", return_value=object()), \
                patch.object(timeline.engine_runtime, "video_experts",
                             return_value=(experts, {"high": [(Path("lora_high"), 1.0)]})) as video:
            ada = [self.profiles["ada"]]
            self.assertEqual(timeline._first_files(project, 0, ada), frame)  # a shot with a character
            self.assertEqual(timeline._first_files(project, 1, ada),  # continues from scene 1: image to video
                             [Path("high"), Path("lora_high"), Path("low")])
            self.assertEqual(video.call_args.args[0], "i2v")
            self.assertEqual(timeline._first_files(project, 2, []), [Path("high"), Path("lora_high"), Path("low")])
            self.assertEqual(video.call_args.args[0], "t2v")  # a Realistic shot with nothing to start from
            start.assert_called_with(project.look.id)
            self.assertEqual(timeline._first_files(project, None, ada), [])
            self.assertEqual(timeline._first_files(project, 7, ada), [])
            self.assertEqual(timeline._first_files(project, 0, None), [])
            with patch.object(type(project), "look", new=property(lambda _self: styles.get(styles.CARTOON_2D))):
                self.assertEqual(timeline._first_files(project, 2, []), drawn)  # the 2D style draws it
            # the job's first scene also reads the text encoder, after its frame or before its experts
            with patch.object(timeline.video_gen, "text_encoder_files", return_value=[Path("umt5")]):
                self.assertEqual(timeline._first_files(project, 0, ada, text=True), [*frame, Path("umt5")])
                self.assertEqual(timeline._first_files(project, 2, [], text=True),
                                 [Path("umt5"), Path("high"), Path("lora_high"), Path("low")])
        with patch.object(timeline.engine_runtime, "start_frame_files", side_effect=FileNotFoundError("gone")):
            self.assertEqual(timeline._first_files(project, 0, [self.profiles["ada"]]), [])


class PromptsAheadTests(Stage4Case):
    """A prepare job announces the Wan prompts of the scenes it will make (engine_runtime.expect_video_prompts), so
    the text encoder loads once for all of them; each announced prompt is the one the scene's Wan part gets."""

    def announced(self, project, **params):
        calls = []
        with patch.object(timeline.engine_runtime, "expect_video_prompts", side_effect=lambda p: calls.append(tuple(p))):
            self.prepare(project, **params)
        return calls

    def test_the_announced_prompts_are_the_ones_the_scenes_give_wan(self):
        project = self.save([timeline.Scene("Ada opens the door", 2), timeline.Scene("Ada looks back", 2),
                             timeline.Scene("Kit waits outside", 2, cut=True, cast=("kit",))])
        calls = self.announced(project)
        expected = tuple(video_gen.scene_wan_prompt(call["prompt"], call["style"], call["cast"],
                                                    exact_cast=call["exact_cast"], ref_images=call["ref_images"])
                         for call in self.video.calls)
        self.assertEqual(len(self.video.calls), 3)
        self.assertTrue(self.video.calls[1]["prompt"].startswith(timeline._CONTINUE))  # anchored to scene 1
        self.assertEqual(calls, [expected, ()])  # announced before the first scene, withdrawn at the end
        self.assertEqual(len(set(expected)), 3)

    def wan_prompts(self):
        return [video_gen.scene_wan_prompt(call["prompt"], call["style"], call["cast"], exact_cast=call["exact_cast"],
                                           ref_images=call["ref_images"]) for call in self.video.calls]

    def test_a_single_scene_job_also_announces_the_next_scenes_still_to_make(self):
        """Generate on one scene encodes, in the same text-encoder load, the prompts of the film's other scenes that
        still need a clip, the next ones first and a few at most: the engine keeps the encodings between jobs, so
        making them one by one skips the encoder."""
        project = self.save([timeline.Scene("Ada opens the door", 2), timeline.Scene("Ada looks back", 2, cut=True),
                             timeline.Scene("Kit waits outside", 2, cut=True, cast=("kit",))])
        calls = self.announced(project, scene_index=1)
        own = self.wan_prompts()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1], ())
        self.assertEqual(calls[0][0], own[0])
        self.video.calls.clear()
        self.prepare(project)  # the film's other scenes get exactly the prompts that were announced
        first, third = self.wan_prompts()
        self.assertEqual(calls[0][1:], (third, first))  # the scene after the one made first, then the earlier one
        with patch.object(timeline, "PROMPTS_AHEAD_OTHER_SCENES", 1):
            other = self.save([timeline.Scene("Ada opens the door", 2), timeline.Scene("Ada looks back", 2, cut=True),
                               timeline.Scene("Kit waits outside", 2, cut=True, cast=("kit",))], title="Other")
            self.assertEqual([len(call) for call in self.announced(other, scene_index=0)], [2, 0])

    def test_nothing_is_announced_for_one_scene_or_a_custom_renderer(self):
        project = self.save([timeline.Scene("Ada opens the door", 2)])
        self.assertEqual(self.announced(project, scene_index=0), [])
        project = self.save([timeline.Scene("Ada opens the door", 2), timeline.Scene("Ada looks back", 2)],
                            title="Two")
        calls = []
        with patch.object(timeline.engine_runtime, "expect_video_prompts", side_effect=lambda p: calls.append(p)):
            hook = ProductionScriptTests.hook(self, [])
            ok, _, _, error = timeline.prepare_job({"project": project.id, "key": project.key}, lambda *_: None,
                                                   generate_scene=hook, cache_namespace="ns")
        self.assertTrue(ok, error)
        self.assertEqual(calls, [])

    def test_a_failure_to_work_out_the_prompts_only_skips_the_announcement(self):
        project = self.save([timeline.Scene("Ada opens the door", 2), timeline.Scene("Ada looks back", 2)])
        with patch.object(timeline.video_gen, "scene_wan_prompt", side_effect=RuntimeError("no voices")):
            self.assertEqual(self.announced(project), [])
        self.assertEqual(len(self.video.calls), 2)


class PlanTests(Stage4Case):
    def test_a_sound_only_change_keeps_the_chosen_take(self):
        project = self.save([timeline.Scene("A cup on a table", 1), timeline.Scene("A plate", 1, cut=True)])
        first, plate = self.prepare(project)
        take = self.prepare(project, scene_index=0, take="b" * 32)[0]
        chosen = timeline.asset(project, take)
        rainy = replace(project.scenes[0], background="heavy rain")
        edited = self.save([rainy, project.scenes[1]], project)
        state = timeline.prepared(edited)
        self.assertEqual((state["assets"], state["sound_sources"]), ([None, plate], [take, None]))
        with patch.object(video_gen, "generate", side_effect=AssertionError("The picture did not change")):
            refreshed_key, kept = self.prepare(edited)
        refreshed = timeline.asset(edited, refreshed_key)
        self.assertEqual(kept, plate)
        self.assertEqual(refreshed["picture_sha256"], chosen["sha256"])  # take 2's picture, not take 1's
        self.assertEqual(refreshed["take"], "b" * 32)
        self.assertNotEqual(refreshed["picture_sha256"], timeline.asset(project, first)["sha256"])
        self.assertEqual(timeline.outdated(edited), [])
        # sound edits one after another, without generating in between, still lead back to the take's picture
        windy = self.save([replace(rainy, background="wind"), project.scenes[1]], edited)
        self.assertEqual(timeline.prepared(windy)["sound_sources"], [refreshed_key, None])
        stormy = self.save([replace(rainy, background="storm"), project.scenes[1]], windy)
        self.assertEqual(timeline.prepared(stormy)["sound_sources"], [refreshed_key, None])
        with patch.object(video_gen, "generate", side_effect=AssertionError("The picture did not change")):
            again = self.prepare(stormy)[0]
        self.assertEqual(timeline.asset(stormy, again)["picture_sha256"], chosen["sha256"])

    def test_clips_that_no_longer_fit_are_made_again(self):
        project = self.save([timeline.Scene("Ada holds a red cup", 1), timeline.Scene("Ada turns the cup", 1)])
        first, second = self.prepare(project)
        self.assertEqual(timeline.outdated(project), [])
        with self.assertRaisesRegex(ValueError, "Every scene already has a clip"):
            timeline.enqueue(project, timeline.PREPARE)
        # A new picture for scene 1: scene 2 continued from the old one and is made again too.
        blue = self.save([timeline.Scene("Ada holds a blue cup", 1), project.scenes[1]], project)
        self.assertEqual(timeline.prepared(blue)["assets"], [None, second])
        self.assertEqual(timeline.outdated(blue), [0, 1])
        job = timeline.enqueue(blue, timeline.PREPARE)
        self.assertEqual(job.params["seconds"], 2)
        self.assertTrue(self.run_job(job)[0])
        new_first, new_second = timeline.prepared(blue)["assets"]
        self.assertNotEqual(new_second, second)
        self.assertEqual(timeline.asset(blue, new_second)["anchor_sha256"], timeline.asset(blue, new_first)["sha256"])
        self.assertTrue(timeline.takes(blue, 1)[0]["anchor_ok"])
        self.assertEqual(len(self.video.calls), 4)
        # An intentional cut toggled on: the clip that continued the previous one does not fit.
        cut = self.save([blue.scenes[0], replace(blue.scenes[1], cut=True)], blue)
        self.assertEqual(timeline.prepared(cut)["assets"], [new_first, new_second])
        self.assertEqual(timeline.outdated(cut), [1])
        self.assertTrue(self.run_job(timeline.enqueue(cut, timeline.PREPARE))[0])
        self.assertEqual(timeline.asset(cut, timeline.prepared(cut)["assets"][1])["transition"], "cut")
        self.assertEqual(len(self.video.calls), 5)
        # A character's profile changed (here the voice): every clip of the character is made again.
        self.profiles["ada"] = {**self.profiles["ada"], "voice": "zoe"}
        timeline._MEMO.clear()
        self.assertEqual(timeline.outdated(cut), [0, 1])
        # The export makes them before joining; a chosen take that still fits is kept.
        self.assertTrue(self.run_job(timeline.enqueue(cut, timeline.PREPARE))[0])
        take = self.prepare(cut, scene_index=1, take="c" * 32)[1]
        self.assertEqual(timeline.outdated(cut), [])
        with patch.object(timeline.llm, "inspect_review", side_effect=AssertionError("A cut needs no review")):
            ok, files, _, error = self.run_job(timeline.enqueue(cut, timeline.PRODUCE))
        self.assertTrue(ok, error)
        self.assertEqual(timeline.prepared(cut)["assets"][1], take)

    def test_join_and_check_again_keeps_the_edit_when_the_answer_is_remembered(self):
        project = self.save([timeline.Scene("A cup", 1), timeline.Scene("A different room", 1)])
        self.prepare(project)
        params = {"project": project.id, "key": project.key}
        with patch.object(timeline.llm, "inspect_review", return_value={"verdict": "mismatch", "reason": "Other room."}):
            self.assertTrue(timeline.merge_job(params, lambda *_: None)[0])
        merged = timeline.prepared(project)["merged"]
        self.assertEqual((merged["attention"], merged["retry"]), ([1], []))
        ok, files, _, _ = timeline.render_job(params, lambda *_: None)
        self.assertTrue(ok)
        with patch.object(timeline.llm, "inspect_review", side_effect=AssertionError("Reviewed again")), \
                patch.object(timeline.media, "join_timeline", side_effect=AssertionError("Joined again")), \
                patch.object(timeline.queue_manager, "reused") as reused:
            ok, _, notes, _ = timeline.merge_job(params, lambda *_: None)
        self.assertTrue(ok)
        reused.assert_called_once_with()
        self.assertIn("Review the join or regenerate scene 2.", " ".join(notes))
        self.assertEqual(timeline.prepared(project)["merged"]["file"], merged["file"])
        self.assertEqual(timeline.exported(project), Path(files[0]))  # the export stays current
        self.assertEqual(len(history.recent()), 1)
        # A join the model could not check is checked again on the next try.
        other = self.save([timeline.Scene("A plate", 1), timeline.Scene("A garden", 1)], title="Other")
        self.prepare(other)
        params = {"project": other.id, "key": other.key}
        with patch.object(timeline.llm, "inspect_review", side_effect=timeline.llm.VisualReviewUnavailable("busy")):
            self.assertTrue(timeline.merge_job(params, lambda *_: None)[0])
        self.assertEqual(timeline.prepared(other)["merged"]["retry"], [1])
        with patch.object(timeline.llm, "inspect_review", return_value={"verdict": "match", "reason": "Fine."}) as review:
            self.assertTrue(timeline.merge_job(params, lambda *_: None)[0])
        review.assert_called_once()

    def test_jobs_that_find_their_result_ready_do_not_teach_the_queue_a_speed(self):
        def run(kind, params, seconds, *, reuse):
            def runner(_params, _cb):
                if reuse:
                    queue_manager.reused()
                return True, [], [], ""
            job = queue_manager.Job(1, kind, "Job", params)
            job.started = time.time() - seconds
            with patch.dict(queue_manager._RUNNERS, {kind: runner}), patch.object(queue_manager, "_save"), \
                    patch.dict(queue_manager._RATE, {kind: 77.0}):
                queue_manager._run(job)
                return queue_manager._RATE[kind]

        export = {"work_seconds": 0.5}
        self.assertEqual(run(timeline.PRODUCE, export, 30.0, reuse=True), 77.0)
        self.assertEqual(run(timeline.PRODUCE, export, 0.2, reuse=False), 77.0)  # ended at once: nothing learned
        # The job really runs for a few milliseconds on top of the 30 s it pretends to have taken.
        self.assertAlmostEqual(run(timeline.PRODUCE, export, 30.0, reuse=False), 0.7 * 77 + 0.3 * 30 / 0.5, places=2)
        self.assertAlmostEqual(run(timeline.SOUND, {"seconds": 0.5}, 30.0, reuse=False), 0.7 * 77 + 0.3 * 30,
                               places=2)
        project = self.save([timeline.Scene("A cup", 1)])
        self.prepare(project)
        with patch.object(timeline.llm, "inspect_review", side_effect=AssertionError("No join")):
            self.assertTrue(self.run_job(timeline.enqueue(project, timeline.PRODUCE))[0])
        with patch.object(timeline.queue_manager, "reused") as reused:
            ok, files, _, _ = timeline.produce_job({"project": project.id, "key": project.key}, lambda *_: None)
        self.assertTrue(ok)
        reused.assert_called_once_with()

    def test_a_postscript_file_named_like_a_picture_never_reaches_ghostscript(self):
        from PIL import EpsImagePlugin

        fake = self.root / "photo.png"
        fake.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nnewpath 0 0 moveto showpage\n%%EOF\n")
        with patch.object(EpsImagePlugin, "has_ghostscript", return_value=True), \
                patch.object(EpsImagePlugin.subprocess, "check_call", side_effect=AssertionError("Ghostscript ran")), \
                self.assertRaisesRegex(ValueError, "not an image the studio can read"):
            timeline.store_reference(fake)
        for name, image in (("photo.jpg", "JPEG"), ("photo.webp", "WEBP"), ("photo.gif", "GIF"), ("photo.bmp", "BMP"),
                            ("photo.tif", "TIFF")):
            with self.subTest(format=image):
                path = self.root / name
                Image.new("RGB", (16, 12), (40, 80, 120)).save(path, format=image)
                self.assertTrue(scene_parts.is_reference(timeline.store_reference(path)))

    def test_a_locked_plan_file_is_written_again_and_a_lasting_lock_is_reported_plainly(self):
        target = self.root / "plan.json"
        with patch.object(timeline, "atomic_text", side_effect=[PermissionError(32, "In use"), None]) as write:
            timeline._write(target, {"assets": []})
        self.assertEqual(write.call_count, 2)
        project = self.save([timeline.Scene("A cup", 1)])
        params = {"project": project.id, "key": project.key}
        with patch.object(timeline, "_write_plan", side_effect=PermissionError(32, "In use")):
            ok, _, _, error = timeline.prepare_job(params, lambda *_: None)
        self.assertFalse(ok)
        self.assertEqual(error, timeline.PLAN_NOT_SAVED)
        self.assertEqual(timeline.failure(project), {"scene": 0, "error": timeline.PLAN_NOT_SAVED,
                                                     "kind": timeline.PREPARE})
        self.assertTrue(timeline.prepare_job(params, lambda *_: None)[0])  # the clip was kept: no new generation
        self.assertEqual(len(self.video.calls), 1)
        self.assertIsNone(timeline.failure(project))
        with patch.object(timeline, "_write_plan", side_effect=PermissionError(32, "In use")), \
                self.assertRaisesRegex(OSError, "^Could not save the timeline plan"):
            timeline.merge_job(params, lambda *_: None)


class RetakeTests(Stage4Case):
    """A scene whose picture failed (for example no face for lip sync) is retried with a new take, because
    the same seed would make the same picture; an unavailable review keeps the take and its saved picture."""

    FACELESS = "Lip sync failed: No face could be lip-synced"

    def flaky(self, failures):
        """video_gen.generate that fails for prompts in `failures` (each failure is used once)."""
        def generate(prompt, seconds, aspect, **options):
            for text in list(failures):
                if text in prompt:
                    failure = failures.pop(text)
                    if isinstance(failure, Exception):
                        raise failure
                    self.video.calls.append({"prompt": prompt, "failed": True, **options})
                    return video_gen.VideoOutput(False, error=failure)
            return self.video(prompt, seconds, aspect, **options)
        return generate

    def seeds(self, text):
        return [call["seed"] for call in self.video.calls if text in call["prompt"]]

    def test_a_faceless_scene_explains_itself_and_its_retry_uses_a_new_seed(self):
        project = self.save([timeline.Scene("Ada opens the door", 1, cast=("ada",)),
                             timeline.Scene("Kit looks up", 1, True, ("kit",))])
        params = {"project": project.id, "key": project.key}
        video_gen.generate.side_effect = self.flaky({"Kit looks up": self.FACELESS})
        ok, _, _, error = timeline.prepare_job(params, lambda *_: None)
        self.assertFalse(ok)
        self.assertTrue(error.startswith("Scene 2: Lip sync failed"), error)
        self.assertIn(timeline.NO_FACE_HINT, error)
        self.assertEqual(timeline.failure(project)["scene"], 1)
        retake = timeline._retake(project, 1)
        self.assertRegex(retake or "", r"^[0-9a-f]{32}$")
        ok, _, _, error = timeline.prepare_job(params, lambda *_: None)  # "Generate missing clips" again
        self.assertTrue(ok, error)
        first, second = self.seeds("Kit looks up")
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.seeds("Ada opens the door")), 1)  # the finished scene is not made again
        keys = timeline.prepared(project)["assets"]
        self.assertEqual(timeline.asset(project, keys[1])["take"], retake)
        self.assertIsNone(timeline.failure(project))
        self.assertIsNone(timeline._retake(project, 1))

    def test_retrying_a_scene_with_a_clip_after_a_failed_take_tries_another_take(self):
        project = self.save([timeline.Scene("Kit looks up", 1, cast=("kit",))])
        params = {"project": project.id, "key": project.key}
        self.assertTrue(timeline.prepare_job(params, lambda *_: None)[0])
        job = timeline.enqueue(project, timeline.PREPARE, scene_index=0)  # Generate new take
        video_gen.generate.side_effect = self.flaky({"Kit looks up": self.FACELESS})
        ok, _, _, _ = self.run_job(job)
        self.assertFalse(ok)
        retake = timeline._retake(project, 0)
        self.assertIsNotNone(retake)
        self.assertNotEqual(retake, job.params["take"])
        again = timeline.enqueue(project, timeline.PREPARE, scene_index=0)  # Retry scene 1
        self.assertEqual(again.params["take"], retake)
        ok, _, _, error = self.run_job(again)
        self.assertTrue(ok, error)
        self.assertEqual(len(set(self.seeds("Kit looks up"))), 3)

    def test_an_unavailable_review_keeps_the_take_for_its_saved_picture(self):
        project = self.save([timeline.Scene("Kit looks up", 1, cast=("kit",))])
        params = {"project": project.id, "key": project.key}
        video_gen.generate.side_effect = self.flaky(
            {"Kit looks up": timeline.llm.VisualReviewUnavailable("The visual review could not run.")})
        ok, _, notes, error = timeline.prepare_job(params, lambda *_: None)  # (ok, files, notes, error)
        self.assertFalse(ok)
        self.assertIn(timeline.REVIEW_UNAVAILABLE, notes)
        self.assertEqual(timeline.failure(project)["scene"], 0)
        self.assertIsNone(timeline._retake(project, 0))
        self.assertTrue(timeline.prepare_job(params, lambda *_: None)[0])

    def test_other_errors_keep_their_text(self):
        self.assertEqual(timeline._explain("Wan failed."), "Wan failed.")
        explained = timeline._explain(self.FACELESS)
        self.assertTrue(explained.startswith(self.FACELESS))
        self.assertEqual(timeline._explain(explained), explained)
        self.assertIn(timeline.NO_FACE_HINT, timeline._explain(
            self.FACELESS + " (Lip sync covered 0 of 80 speech frames of the speaker (no face found in 80); at least"
                            " 50% are needed.)"))
        technical = self.FACELESS + " (CUDA out of memory. Tried to allocate 2.00 GiB)"
        self.assertEqual(timeline._explain(technical), technical)  # not a framing problem: no hint
        self.assertEqual(timeline._explain("A piano face lid opens."), "A piano face lid opens.")


class RetakeEdgeTests(Stage4Case):
    """Retakes survive interruptions, never replace a chosen take after a sound-only failure, and a failed
    take is not reused once its failure record is gone."""

    FACELESS = RetakeTests.FACELESS

    def scripted(self, plan):
        """video_gen.generate following `plan` (None succeeds, str fails, an exception is raised); every call's
        seed is recorded."""
        self.attempts = []

        def generate(prompt, seconds, aspect, **options):
            outcome = plan.pop(0) if plan else None
            self.attempts.append(options.get("seed"))
            if isinstance(outcome, BaseException):
                raise outcome
            if isinstance(outcome, str):
                return video_gen.VideoOutput(False, error=outcome)
            return self.video(prompt, seconds, aspect, **options)
        video_gen.generate.side_effect = generate

    def test_a_review_outage_keeps_the_retake_and_its_seed(self):
        project = self.save([timeline.Scene("Kit looks up", 1, cast=("kit",))])
        params = {"project": project.id, "key": project.key}
        self.scripted([self.FACELESS, timeline.llm.VisualReviewUnavailable("busy"), None])
        self.assertFalse(timeline.prepare_job(params, lambda *_: None)[0])
        retake = timeline._retake(project, 0)
        self.assertFalse(timeline.prepare_job({**params, "scene_index": 0}, lambda *_: None)[0])
        self.assertEqual(timeline._retake(project, 0), retake)
        self.assertTrue(timeline.prepare_job({**params, "scene_index": 0}, lambda *_: None)[0])
        self.assertNotEqual(self.attempts[1], self.attempts[0])
        self.assertEqual(self.attempts[2], self.attempts[1])  # the retake runs again with the same seed

    def test_a_failed_plan_write_after_a_retake_only_writes_the_plan_again(self):
        project = self.save([timeline.Scene("Kit looks up", 1, cast=("kit",))])
        params = {"project": project.id, "key": project.key}
        self.scripted([self.FACELESS, None, None])
        self.assertFalse(timeline.prepare_job(params, lambda *_: None)[0])
        with patch.object(timeline, "_write_plan", side_effect=PermissionError(32, "In use")):
            self.assertEqual(timeline.prepare_job(params, lambda *_: None)[3], timeline.PLAN_NOT_SAVED)
        self.assertTrue(timeline.prepare_job(params, lambda *_: None)[0])
        self.assertEqual(len(self.attempts), 2)  # the stored clip was found again

    def test_a_failed_sound_refresh_keeps_the_chosen_take_picture(self):
        project = self.save([timeline.Scene("A cup on a table", 1), timeline.Scene("A plate", 1, cut=True)])
        self.prepare(project)
        self.prepare(project, scene_index=0, take="b" * 32)
        edited = self.save([replace(project.scenes[0], background="heavy rain"), project.scenes[1]], project)
        params = {"project": edited.id, "key": edited.key}
        self.sound.fail = "EzAudio ran out of GPU memory."
        with patch.object(video_gen, "generate", side_effect=AssertionError("The picture did not change")):
            self.assertFalse(timeline.prepare_job(params, lambda *_: None)[0])
            self.assertIsNone(timeline._retake(edited, 0))
            self.sound.fail = None
            ok, _, _, error = timeline.prepare_job(params, lambda *_: None)
        self.assertTrue(ok, error)
        self.assertEqual(timeline.asset(edited, timeline.prepared(edited)["assets"][0])["take"], "b" * 32)

    def test_a_failed_take_is_not_resumed_after_its_record_is_cleared(self):
        project = self.save([timeline.Scene("A cup", 1), timeline.Scene("Kit looks up", 1, cut=True, cast=("kit",))])
        self.prepare(project)
        job = timeline.enqueue(project, timeline.PREPARE, scene_index=1)  # Generate new take
        self.scripted([self.FACELESS])
        self.assertFalse(timeline.prepare_job(job.params, lambda *_: None)[0])
        job.status, job.error = queue_manager.ERROR, "Scene 2: " + self.FACELESS
        with patch.object(timeline.llm, "inspect_review", side_effect=AssertionError("A cut needs no check")):
            merge = timeline.enqueue(project, timeline.MERGE)
            self.assertTrue(timeline.merge_job(merge.params, lambda *_: None)[0])  # clears the failure record
            merge.status = queue_manager.DONE
        again = timeline.enqueue(project, timeline.PREPARE, scene_index=1)
        self.assertNotEqual(again.params["take"], job.params["take"])
        again.status = queue_manager.STOPPED  # paused before it finished: the next attempt resumes its take
        self.assertEqual(timeline.enqueue(project, timeline.PREPARE, scene_index=1).params["take"], again.params["take"])

    def test_a_cache_namespace_keeps_a_scene_finished_through_a_retake(self):
        project = self.save([timeline.Scene("Ada opens the door", 1), timeline.Scene("Kit looks up", 1, cut=True)])
        params = {"project": project.id, "key": project.key}
        seeds = []

        def hook(project, scene, members, seed, style, folder, cb, *, anchor=None):
            if scene.prompt == "Kit looks up":
                seeds.append(seed)
                if len(seeds) in (1, 3):  # the original seed never shows a face
                    return video_gen.VideoOutput(False, error=self.FACELESS)
            return self.video(scene.prompt, scene.seconds, project.aspect, scene_dir=folder, sound_on=project.audio)

        self.assertFalse(timeline.prepare_job(params, lambda *_: None, generate_scene=hook, cache_namespace="ns")[0])
        self.assertTrue(timeline.prepare_job(params, lambda *_: None, generate_scene=hook, cache_namespace="ns")[0])
        edited = self.save([timeline.Scene("Ada opens the red door", 1), project.scenes[1]], project)
        ok, _, _, error = timeline.prepare_job({"project": edited.id, "key": edited.key}, lambda *_: None,
                                               generate_scene=hook, cache_namespace="ns")
        self.assertTrue(ok, error)
        self.assertEqual(len(seeds), 2)  # the unchanged cut scene keeps its retaken clip


if __name__ == "__main__":
    unittest.main()
