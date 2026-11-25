# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Scene sound before the picture (app/scene_sound.py): speakers, the sound key, lines on the scene
clock, the background with timed effects, the manifest and the level previews. CPU only: speech and
EzAudio are replaced by fakes that write plain tones."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from app import characters, dialogue, prefs, scene_sound, sound, tts, voices
from app.scene_parts import Cue, Effect
from engine.media import audio as media_audio

RATE = media_audio.RATE
HANDLE = media_audio.TIMELINE_AUDIO_HANDLE
PROMPT = "steady rain on a window, high quality realistic sound effects recording"
ROSE = {"name": "Rose", "slug": "rose", "gender": "woman"}
IVY = {"name": "Ivy", "slug": "ivy", "gender": "woman"}
OLAF = {"name": "Olaf", "slug": "olaf", "gender": "man", "voice": "adam"}


def _tone(path, seconds, frequency, level=0.2, channels=1):
    t = np.arange(int(round(seconds * RATE))) / RATE
    wave = (level * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    media_audio.write_wav(path, np.stack([wave] * channels), RATE)
    return Path(path)


def _rms(samples, start, end, rate=RATE):
    window = samples[..., int(round(start * rate)):int(round(end * rate))]
    return float(np.sqrt(np.mean(window * window)))


def _mono(path):
    return media_audio.read(path, rate=RATE, channels=1)[0][0]


def _library(root):
    pool = []
    for name, slug, gender in (("Hanna", "hanna", "female"), ("Zoe", "zoe", "female"), ("Adam", "adam", "male")):
        pool.append(voices.Voice(name, slug, gender, str(_tone(root / f"{slug}.wav", 0.5, 180)), builtin=True))
    return pool


class FakeSpeech:
    """tts.synthesize_many: one tone per line, as long as `lengths` says (1 s otherwise)."""

    def __init__(self, events, lengths=None):
        self.events, self.lengths, self.calls = events, dict(lengths or {}), []

    def __call__(self, items, progress_cb=None):
        self.events.append("speech")
        self.calls.append(list(items))
        for index, (text, _voice, out) in enumerate(items):
            _tone(out, self.lengths.get(text, 1.0), 300 + 110 * index)
        return True, str(items[-1][2])


class FakeSound:
    """sound.generate: `seconds` + both handles of a steady tone, 220 Hz for the bed and 1500 Hz for
    an effect, as quiet as each other."""

    def __init__(self, events):
        self.events, self.calls = events, []

    def __call__(self, client, scene, custom, seconds, seed, out_path, progress_cb=None):
        self.events.append("ambience")
        self.calls.append((custom, seconds, seed))
        _tone(out_path, seconds + 2 * HANDLE, 220 if custom == PROMPT else 1500, level=0.05)
        return True, str(out_path)


class SceneSoundCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.pool = _library(self.root)
        self.events = []
        self.speech = FakeSpeech(self.events)
        self.ambience = FakeSound(self.events)
        self.client = MagicMock()
        self.client.free.side_effect = lambda: self.events.append("free")
        self.client.trim.side_effect = lambda: self.events.append("trim")
        self.stack.enter_context(patch.object(voices, "all_voices", return_value=self.pool))
        self.stack.enter_context(patch.object(prefs, "_FILE", self.root / "prefs.json"))
        self.stack.enter_context(patch.object(tts, "synthesize_many", side_effect=self.speech))
        self.stack.enter_context(patch.object(sound, "generate", side_effect=self.ambience))

    def prepare(self, cues=(), effects=(), seconds=4, folder="sound", members=(ROSE,), **options):
        speakers = scene_sound.resolve_speakers(cues, list(members))
        return scene_sound.prepare(cues, speakers, PROMPT, effects, seconds, 17, self.root / folder,
                                   client=self.client, **options)


class SpeakerTests(SceneSoundCase):
    def test_characters_get_their_own_or_a_free_voice_and_library_voices_keep_their_gender(self):
        cues = (Cue("Hi.", "rose"), Cue("Hey.", "ivy"), Cue("Hello.", "@adam"), Cue("Bye.", "olaf"))
        speakers = scene_sound.resolve_speakers(cues, [ROSE, IVY, OLAF], self.pool)
        self.assertEqual([(s.voice, s.who, s.name, s.character) for s in speakers],
                         [("hanna", "female#char:rose", "Rose", "rose"), ("zoe", "female#char:ivy", "Ivy", "ivy"),
                          ("adam", "male#adam", "Adam", ""), ("adam", "male#char:olaf", "Olaf", "olaf")])
        own = {**IVY, "voice": "hanna"}  # Ivy's own voice wins; Rose takes the next free one
        speakers = scene_sound.resolve_speakers((Cue("Hi.", "rose"), Cue("Hey.", "ivy")), [ROSE, own], self.pool)
        self.assertEqual([s.voice for s in speakers], ["zoe", "hanna"])
        self.assertEqual([s.gender for s in speakers], ["female", "female"])

    def test_an_empty_speaker_is_the_only_character(self):
        self.assertEqual(scene_sound.resolve_speakers((Cue("Hi."),), [OLAF], self.pool)[0].who, "male#char:olaf")
        for members in ([], [ROSE, IVY]):
            with self.assertRaisesRegex(ValueError, r"^Line 2: choose who says it\.$"):
                scene_sound.resolve_speakers((Cue("Hi.", "@zoe"), Cue("Hey.")), members, self.pool)

    def test_unknown_speakers_name_the_line(self):
        with patch.object(characters, "get", return_value={"name": "Kit", "slug": "kit"}):
            with self.assertRaisesRegex(ValueError, r"^Line 2: Kit is not in this scene\.$"):
                scene_sound.resolve_speakers((Cue("Hi.", "rose"), Cue("Hey.", "kit")), [ROSE], self.pool)
        with self.assertRaisesRegex(ValueError, r"^Line 1: no voice named @nobody\.$"):
            scene_sound.resolve_speakers((Cue("Hi.", "@nobody"),), [ROSE], self.pool)
        with self.assertRaisesRegex(ValueError, "no man's voice"):
            scene_sound.resolve_speakers((Cue("Hi.", "olaf"),), [{**OLAF, "voice": ""}], self.pool[:2])


class SoundKeyTests(SceneSoundCase):
    def key(self, cues=(Cue("Hi there.", "rose", 0.5),), effects=(Effect("door", 1.0),), prompt=PROMPT,
            seconds=5, seed=7, members=(ROSE,), speakers=None, **flags):
        speakers = speakers if speakers is not None else scene_sound.resolve_speakers(cues, list(members))
        return scene_sound.sound_key(cues, speakers, prompt, effects, seconds, seed, **flags)

    def test_the_key_follows_every_input_of_the_sound_and_nothing_else(self):
        base = self.key()
        self.assertRegex(base, r"^[0-9a-f]{64}$")
        self.assertEqual(self.key(), base)
        changed = [
            self.key(cues=(Cue("Hi again.", "rose", 0.5),)),
            self.key(cues=(Cue("Hi there.", "rose", 0.6),)),
            self.key(cues=(Cue("Hi there.", "rose"),)),
            self.key(cues=(Cue("Hi there.", "@zoe", 0.5),)),
            self.key(effects=(Effect("door", 1.5),)),
            self.key(effects=(Effect("door slam", 1.0),)),
            self.key(effects=(Effect("door", 1.0, 3.0),)),
            self.key(effects=()),
            self.key(prompt="wind"),
            self.key(seconds=6),
            self.key(seed=8),
        ]
        self.assertEqual(len(set(changed + [base])), len(changed) + 1)
        renamed = [scene_sound.Speaker("female", "hanna", "rose", "female#char:rose", "Rosie")]
        self.assertEqual(self.key(speakers=renamed), base)  # a display name is not part of the sound
        # the voice sample itself
        sample = Path(self.pool[0].path)
        stat = sample.stat()
        _tone(sample, 0.5, 260)
        os.utime(sample, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
        self.assertNotEqual(self.key(), base)

    def test_parts_that_are_not_prepared_do_not_count(self):
        voice_only, background_only = {"background": False}, {"voice": False}
        self.assertEqual(self.key(effects=(), **voice_only), self.key(**voice_only))
        self.assertEqual(self.key(prompt="wind", **voice_only), self.key(**voice_only))
        self.assertEqual(self.key(cues=(Cue("Other.", "rose"),), **background_only), self.key(**background_only))
        self.assertNotEqual(self.key(**voice_only), self.key(**background_only))

    def test_voice_samples_are_hashed_once_while_they_do_not_change(self):
        scene_sound._DIGESTS.clear()
        with patch.object(scene_sound.hashlib, "file_digest", wraps=hashlib.file_digest) as digest:
            first = self.key()
            self.assertEqual(digest.call_count, 1)
            self.assertEqual(self.key(), first)
            self.assertEqual(digest.call_count, 1)


class PrepareTests(SceneSoundCase):
    def test_automatic_starts_follow_the_measured_lines(self):
        self.speech.lengths.update({"First line.": 1.0, "Second line.": 0.8})
        result = self.prepare((Cue("First line.", "rose"), Cue("Second line.", "rose")))
        self.assertEqual(result.timings, ((0.3, 1.3), (1.8, 2.6)))
        self.assertEqual(result.texts, ("First line.", "Second line."))
        self.assertEqual(len(self.speech.calls), 1)  # one batch for every line
        self.assertEqual([voice for _, voice, _ in self.speech.calls[0]], ["hanna", "hanna"])
        track = _mono(result.voice)
        self.assertEqual(track.size, 4 * RATE)
        self.assertGreater(_rms(track, 0.35, 1.25), 0.1)
        self.assertLess(_rms(track, 1.35, 1.75), 1e-3)
        self.assertGreater(_rms(track, 1.85, 2.55), 0.1)
        self.assertLess(_rms(track, 2.65, 4.0), 1e-3)
        self.assertEqual([path.name for path in result.lines], ["line1.wav", "line2.wav"])
        # unused GPU memory is given back before each model; models kept warm in RAM stay
        self.assertEqual(self.events[:4], ["trim", "speech", "trim", "ambience"])
        self.client.free.assert_not_called()

    def test_lines_of_several_scenes_share_one_batch_and_build_the_same_package(self):
        """synthesize_lines makes the lines of two scenes in one speech call (one model load); a scene given
        its files builds exactly the package it would build synthesizing them itself."""
        first = (Cue("First line.", "rose"), Cue("Second line.", "rose"))
        second = (Cue("Hello.", "@adam"),)
        speakers = [scene_sound.resolve_speakers(cues, [ROSE]) for cues in (first, second)]
        made = scene_sound.synthesize_lines(list(zip((first, second), speakers)), self.root / "batch")
        self.assertEqual(len(self.speech.calls), 1)
        self.assertEqual([(text, voice) for text, voice, _ in self.speech.calls[0]],
                         [("First line.", "hanna"), ("Second line.", "hanna"), ("Hello.", "adam")])
        self.assertEqual([[path.name for path in paths] for paths in made],
                         [["scene1_line1.wav", "scene1_line2.wav"], ["scene2_line1.wav"]])
        given = self.prepare(first, folder="given", lines=made[0])
        self.assertEqual(len(self.speech.calls), 1)  # nothing synthesized again
        alone = self.prepare(first, folder="alone")
        self.assertEqual(len(self.speech.calls), 2)
        self.assertEqual((given.key, given.timings), (alone.key, alone.timings))
        for kept, own in zip((*given.lines, given.voice), (*alone.lines, alone.voice)):
            self.assertEqual(Path(kept).read_bytes(), Path(own).read_bytes())
        self.assertEqual([Path(path).read_bytes() for path in made[0]],
                         [Path(path).read_bytes() for path in given.lines])  # copied, the job's files stay
        with self.assertRaisesRegex(ValueError, "synthesized file"):
            self.prepare(first, folder="short", lines=made[0][:1])

    def test_explicit_starts_overlap_and_the_lines_are_summed(self):
        result = self.prepare((Cue("One.", "rose", start=0.5), Cue("Two.", "@adam", start=1.0)), seconds=3)
        self.assertEqual(result.timings, ((0.5, 1.5), (1.0, 2.0)))
        first, second = (sf.read(str(path), dtype="float32")[0] for path in result.lines)
        track = sf.read(str(result.voice), dtype="float32")[0]
        expected = np.zeros(3 * RATE, dtype=np.float32)
        expected[RATE // 2:RATE // 2 + first.size] += first
        expected[RATE:RATE + second.size] += second
        self.assertLess(float(np.max(np.abs(track - expected))), 1e-3)
        overlap = slice(int(1.05 * RATE), int(1.45 * RATE))
        self.assertGreater(float(np.max(np.abs(track[overlap]))), 0.3)  # both lines at once

    def test_a_line_past_the_scene_end_is_cut_and_one_after_it_is_not_heard(self):
        """The scene keeps the length chosen for it (the owner's decision of 2026-11-16): a measured line that runs
        past the scene end is cut there and one that starts after it is not heard, never an error."""
        self.speech.lengths["Too long."] = 1.0
        cut = self.prepare((Cue("Too long.", "rose", start=1.5),), seconds=2)
        self.assertEqual(cut.timings, ((1.5, 2.5),))  # what was measured stays known
        voice = _mono(cut.voice)
        self.assertEqual(voice.size, 2 * RATE)  # the scene's own length
        self.assertGreater(_rms(voice, 1.6, 2.0), 0.01)  # the line plays up to the scene end
        late = self.prepare((Cue("A.", "rose"), Cue("B.", "rose"), Cue("C.", "rose")), seconds=3, folder="late")
        self.assertEqual(len(late.lines), 3)
        self.assertEqual([round(start, 1) for start, _ in late.timings], [0.3, 1.8, 3.3])  # line 3: after the end
        self.assertEqual(_mono(late.voice).size, 3 * RATE)
        self.assertEqual(len(self.ambience.calls), 2)  # both scenes got their background

    def test_the_bed_and_its_effects_share_one_audio_session(self):
        """EzAudio is read from disk once for the bed and every effect of a scene (Engine.audio_session), not once
        per call."""
        from contextlib import contextmanager

        @contextmanager
        def session():
            self.events.append("session")
            yield
            self.events.append("end")

        self.client.audio_session.side_effect = session
        self.prepare((Cue("Hi.", "rose"),), effects=(Effect("door slam", at=1.0, seconds=1.0),
                                                     Effect("glass", at=2.0, seconds=0.5)), seconds=3)
        self.assertEqual(self.events[self.events.index("session"):], ["session", "ambience", "ambience", "ambience", "end"])
        self.assertLess(self.events.index("speech"), self.events.index("session"))  # VoxCPM2 is not in it

    def test_effects_play_at_their_time_over_a_handle_padded_background(self):
        result = self.prepare(effects=(Effect("door slam", at=1.0, seconds=1.0),), seconds=3)
        background = _mono(result.background)
        self.assertEqual(background.size, int(3.5 * RATE))  # the scene plus both handles
        effect = _rms(background, HANDLE + 1.05, HANDLE + 1.95)
        for before, after in ((HANDLE + 0.1, HANDLE + 0.9), (HANDLE + 2.1, HANDLE + 2.9)):
            self.assertGreater(effect, 2 * _rms(background, before, after))
        self.assertLessEqual(float(np.max(np.abs(background))), 1.0)
        self.assertEqual(self.ambience.calls, [(PROMPT, 3, 17), ("door slam", 1.0, 17 + 1009)])
        self.assertEqual(_mono(result.effects[0]).size, int(1.5 * RATE))  # as generated, with its handles
        self.assertEqual(result.effect_texts, ("door slam",))
        self.assertEqual(sorted(path.name for path in result.folder.iterdir()),
                         ["background.wav", "effect1.wav", "overlay.wav", "sound.json"])
        self.assertIsNone(result.voice)
        # The overlay is the effects alone, as laid into the background: background - overlay is the bed.
        overlay = _mono(result.overlay)
        self.assertEqual(overlay.size, background.size)
        bed = background - overlay
        self.assertLess(_rms(overlay, HANDLE + 0.1, HANDLE + 0.9), 1e-4)
        self.assertGreater(_rms(overlay, HANDLE + 1.05, HANDLE + 1.95), 0.05)
        self.assertAlmostEqual(_rms(bed, HANDLE + 1.05, HANDLE + 1.95), _rms(bed, HANDLE + 0.1, HANDLE + 0.9),
                               delta=0.003)
        self.assertGreater(result.bed_level, scene_sound.QUIET_BED)
        self.assertEqual(result.bed_audible, 1.0)

    def test_an_effect_starts_where_its_sound_starts(self):
        """The generator put a camera click in the first 50 ms of the effect file; a fixed trim of the
        background's 0.25 s handle used to cut it out, leaving a silent overlay."""
        def ambience(client, scene, custom, seconds, seed, out_path, progress_cb=None):
            self.events.append("ambience")
            if custom == PROMPT:
                _tone(out_path, seconds + 2 * HANDLE, 220, level=0.05)
            else:
                samples = np.zeros(int(round((seconds + 2 * HANDLE) * RATE)), dtype=np.float32)
                start, stop = int(0.03 * RATE), int(0.06 * RATE)
                samples[start:stop] = 0.8 * np.sin(2 * np.pi * 2500 * np.arange(stop - start) / RATE)
                sf.write(str(out_path), samples, RATE)
            return True, str(out_path)

        sound.generate.side_effect = ambience
        result = self.prepare(effects=(Effect("camera shutter click", at=1.5, seconds=0.8),), seconds=3)
        overlay = _mono(result.overlay)
        click = _rms(overlay, HANDLE + 1.49, HANDLE + 1.56)
        self.assertGreater(click, 0.05)  # it is there, at its time
        self.assertLess(_rms(overlay, HANDLE + 1.0, HANDLE + 1.45), 1e-4)  # and not before it
        loudest = int(np.argmax(np.abs(overlay)))
        self.assertAlmostEqual(loudest / RATE, HANDLE + 1.5 + 0.01, delta=0.03)

    def test_an_effect_is_cut_at_the_scene_end_and_a_plain_bed_is_kept_as_generated(self):
        result = self.prepare(effects=(Effect("thunder", at=2.5, seconds=2.0),), seconds=3)
        background = _mono(result.background)
        bed = _rms(background, HANDLE + 0.2, HANDLE + 2.2)
        self.assertGreater(_rms(background, HANDLE + 2.55, HANDLE + 2.95), 2 * bed)
        self.assertAlmostEqual(_rms(background, HANDLE + 3.06, HANDLE + 3.24), bed, delta=0.2 * bed)
        plain = self.prepare(seconds=3, folder="plain")
        self.assertEqual(_mono(plain.background).size, int(3.5 * RATE))
        self.assertEqual(plain.effects, ())

    def test_the_manifest_round_trip_and_what_load_rejects(self):
        result = self.prepare((Cue("Hello.", "rose", 0.5),), (Effect("door", 1.0),))
        self.assertEqual(scene_sound.load(result.folder, result.key), result)
        self.assertEqual(scene_sound.load(result.folder), result)
        self.assertEqual(scene_sound.load(result.folder, result.key, verify=True), result)
        self.assertIsNone(scene_sound.load(result.folder, "0" * 64))
        self.assertIsNone(scene_sound.load(self.root / "nowhere"))
        self.assertEqual(result.speakers[0], scene_sound.Speaker("female", "hanna", "rose", "female#char:rose", "Rose"))
        self.assertEqual(result.background_prompt, PROMPT)

        background = result.background
        stat = background.stat()
        _tone(background, 4.5, 330, level=0.05)  # same size, other sound
        os.utime(background, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(scene_sound.load(result.folder, result.key), result)  # the cheap check sees no change
        self.assertIsNone(scene_sound.load(result.folder, result.key, verify=True))
        os.utime(background, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
        self.assertIsNone(scene_sound.load(result.folder, result.key))
        result.lines[0].unlink()
        self.assertIsNone(scene_sound.load(result.folder, result.key))

    def test_a_ready_folder_is_reused_and_a_broken_one_is_replaced(self):
        cues = (Cue("Hello.", "rose", 0.5),)
        first = self.prepare(cues)
        again = self.prepare(cues)
        self.assertEqual(again, first)
        self.assertEqual((len(self.speech.calls), len(self.ambience.calls)), (1, 1))
        first.voice.unlink()
        rebuilt = self.prepare(cues)
        self.assertEqual(rebuilt.key, first.key)
        self.assertTrue(rebuilt.voice.is_file())
        self.assertEqual((len(self.speech.calls), len(self.ambience.calls)), (2, 2))
        self.assertEqual([path.name for path in self.root.iterdir() if path.name.startswith(".")], [])

    def test_a_folder_of_other_files_is_never_replaced(self):
        folder = self.root / "taken"
        folder.mkdir()
        (folder / "keep.txt").write_text("mine", encoding="utf-8")
        with self.assertRaises(OSError):
            self.prepare((Cue("Hello.", "rose"),), folder="taken")
        self.assertEqual(self.events, [])  # refused before any speech or background
        self.assertEqual((folder / "keep.txt").read_text(encoding="utf-8"), "mine")
        self.assertEqual([path.name for path in self.root.iterdir() if path.name.startswith(".")], [])

    def test_inputs_are_checked_before_any_work(self):
        with self.assertRaisesRegex(ValueError, "Effect 1: describe"):
            self.prepare(effects=(Effect("", at=1.0),), seconds=5)
        with self.assertRaisesRegex(ValueError, "speaker"):
            scene_sound.prepare((Cue("Hi."),), (), PROMPT, (), 4, 1, self.root / "x", client=self.client)
        self.assertEqual(self.events, [])

    def test_a_scene_of_tenths_of_a_second_gets_its_sound_at_that_length(self):
        result = self.prepare((Cue("Hi.", "rose"),), effects=(Effect("door slam", at=1.0, seconds=1.0),), seconds=2.5)
        self.assertEqual(result.seconds, 2.5)
        self.assertEqual(_mono(result.voice).size, int(2.5 * RATE))
        self.assertEqual(_mono(result.background).size, int(round((2.5 + 2 * HANDLE) * RATE)))
        self.assertEqual(scene_sound.load(result.folder, result.key), result)
        whole = scene_sound.sound_key((), (), PROMPT, (), 3, 17)
        self.assertEqual(scene_sound.sound_key((), (), PROMPT, (), 3.0, 17), whole)  # 3.0 is 3: the same key
        with self.assertRaisesRegex(ValueError, "in tenths of a second"):
            self.prepare(seconds=2.55, folder="odd")

    def test_an_effect_after_the_scene_end_is_not_generated(self):
        result = self.prepare(effects=(Effect("door slam", at=1.0, seconds=1.0), Effect("late bell", at=5.0)),
                              seconds=5)
        self.assertEqual(len(result.effects), 1)
        self.assertEqual([call[0] for call in self.ambience.calls if call[0] != PROMPT], ["door slam"])

    def test_only_the_dialogue_or_only_the_background(self):
        voice = self.prepare((Cue("Hello.", "rose"),), (Effect("door", 1.0),), background=False, folder="voice")
        self.assertIsNone(voice.background)
        self.assertEqual((voice.effects, voice.background_prompt), ((), ""))
        self.assertTrue(voice.voice.is_file())
        self.assertNotIn("ambience", self.events)
        background = self.prepare((Cue("Hello.", "rose"),), voice=False, folder="background")
        self.assertEqual((background.texts, background.voice), ((), None))
        self.assertEqual(self.events.count("speech"), 1)
        self.assertEqual(scene_sound.load(background.folder, background.key), background)


class QuietBedTests(SceneSoundCase):
    def test_a_nearly_silent_bed_is_made_again_and_the_attempts_are_recorded(self):
        levels = {17: 0.0002, 18: 0.0003}  # two nearly silent beds, then an audible one

        def ambience(client, scene, custom, seconds, seed, out_path, progress_cb=None):
            self.ambience.calls.append((custom, seconds, seed))
            _tone(out_path, seconds + 2 * HANDLE, 220, level=levels.get(seed, 0.05))
            return True, str(out_path)

        with patch.object(sound, "generate", side_effect=ambience):
            result = self.prepare(seconds=2)
        self.assertEqual(self.ambience.calls, [(PROMPT, 2, 17), (PROMPT, 2, 18), (PROMPT, 2, 19)])
        self.assertGreater(_rms(_mono(result.background), 0.3, 2.0), 0.02)  # the audible one was kept
        manifest = json.loads((result.folder / scene_sound.MANIFEST).read_text(encoding="utf-8"))
        attempts = manifest["bed"]["attempts"]
        self.assertEqual([attempt["seed"] for attempt in attempts], [17, 18, 19])
        self.assertTrue(attempts[-1]["kept"])
        self.assertTrue(scene_sound.quiet(attempts[0]["level"], attempts[0]["audible"]))
        self.assertFalse(scene_sound.quiet(result.bed_level, result.bed_audible))
        self.assertEqual(sorted(path.name for path in result.folder.iterdir()), ["background.wav", "sound.json"])

    def test_after_two_more_seeds_the_audible_fallback_prompt_is_tried_and_the_loudest_bed_kept(self):
        fallback = sound.prompt_from_scene("", "")
        self.assertIn("air conditioner", fallback)  # an audible description, not "subtle room tone"

        def ambience(client, scene, custom, seconds, seed, out_path, progress_cb=None):
            self.ambience.calls.append((custom, seconds, seed))
            _tone(out_path, seconds + 2 * HANDLE, 220, level=0.0002 * len(self.ambience.calls))
            return True, str(out_path)

        with patch.object(sound, "generate", side_effect=ambience):
            result = self.prepare(seconds=2, effects=(Effect("door", 0.5, 0.5),))
        prompts = [call[0] for call in self.ambience.calls]
        self.assertEqual(prompts, [PROMPT, PROMPT, PROMPT, fallback, "door"])
        self.assertEqual([call[2] for call in self.ambience.calls[:4]], [17, 18, 19, 17])
        manifest = json.loads((result.folder / scene_sound.MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["bed"]["attempts"]), 4)
        self.assertEqual([attempt.get("kept", False) for attempt in manifest["bed"]["attempts"]],
                         [False, False, False, True])  # all quiet: the loudest (the last) stays
        self.assertTrue(scene_sound.quiet(result.bed_level, result.bed_audible))
        self.assertEqual(sorted(path.name for path in result.folder.iterdir()),
                         ["background.wav", "effect1.wav", "overlay.wav", "sound.json"])

    def test_bed_loudness_measures_like_the_export_and_is_remembered(self):
        quiet, loud = self.root / "quiet.wav", self.root / "loud.wav"
        _tone(quiet, 3 + 2 * HANDLE, 220, level=0.00008)
        _tone(loud, 3 + 2 * HANDLE, 220, level=0.01)
        self.assertEqual(scene_sound.bed_loudness(quiet, 3), (0.0, 0.0))  # nothing above the audible threshold
        level, audible = scene_sound.bed_loudness(loud, 3)
        self.assertAlmostEqual(level, media_audio.BED_TARGET, delta=0.002)  # the export lifts it to the target
        self.assertEqual(audible, 1.0)
        with patch.object(media_audio, "background_loudness", side_effect=AssertionError("measured again")):
            self.assertEqual(scene_sound.bed_loudness(loud, 3), (level, audible))


class PreviewTests(SceneSoundCase):
    def test_the_preview_mix_follows_the_levels_like_the_export(self):
        result = self.prepare((Cue("Hello.", "rose", 2.0),))
        plain = media_audio.read(scene_sound.preview_mix(result, 0, 0, self.root / "plain.wav"))[0]
        louder = media_audio.read(scene_sound.preview_mix(result, 0, 6, self.root / "louder.wav"))[0]
        self.assertEqual(plain.shape, (2, 4 * RATE))
        self.assertAlmostEqual(_rms(louder, 0.6, 1.6) / _rms(plain, 0.6, 1.6), 2.0, delta=0.08)
        self.assertLessEqual(float(np.max(np.abs(louder))), 1.0)
        self.assertEqual(sorted(path.name for path in self.root.glob("*.wav") if path.name.startswith(".")), [])
        with self.assertRaisesRegex(ValueError, "level"):
            scene_sound.preview_mix(result, 0, 30, self.root / "loud.wav")
        with self.assertRaisesRegex(ValueError, "no sound"):
            scene_sound.mix_stems(None, None, 4, 0, 0, self.root / "none.wav")

    def test_a_leveled_clip_keeps_the_picture_and_takes_its_sound_from_the_stems(self):
        import av

        from engine.media import video as media_video
        result = self.prepare((Cue("Hello.", "rose", 1.0),), seconds=2)
        t = np.arange(2 * RATE) / RATE
        own = np.stack([0.3 * np.sin(2 * np.pi * 1000 * t)] * 2).astype(np.float32)  # the clip's own mix
        clip = self.root / "clip.mp4"
        with media_video.Writer(clip, 30, 48, 32, preset="ultrafast", audio=(own, RATE)) as writer:
            for index in range(60):
                writer.write(np.full((32, 48, 3), 4 * index, dtype=np.uint8))
        out = scene_sound.leveled_clip(clip, result.background, result.voice, 2, 0, 6, self.root / "leveled.mp4")
        with av.open(str(out)) as container:
            self.assertEqual(len(container.streams.audio), 1)
            self.assertEqual(container.metadata.get("comment"), media_video.AI_COMMENT)
            frames = [frame.to_ndarray(format="rgb24").mean() for frame in container.decode(video=0)]
        self.assertEqual(len(frames), 60)
        self.assertAlmostEqual(frames[-1], 4 * 59, delta=2)
        mixed = media_audio.read(out)[0]
        expected = media_audio.scene_mix(result.background, result.voice, 2, 0, 6)
        self.assertAlmostEqual(_rms(mixed, 0.2, 0.9) / _rms(expected, 0.2, 0.9), 1.0, delta=0.1)
        spectrum = np.abs(np.fft.rfft(mixed[0, :RATE]))
        self.assertLess(spectrum[1000], 0.05 * spectrum.max())  # the clip's own 1 kHz mix is gone
        self.assertEqual(sorted(path.name for path in self.root.iterdir() if path.name.startswith(".")), [])

        # a sound-only change: the clip's own mix made again from new stems, lined up with the picture
        refreshed = self.prepare(effects=(Effect("door slam", at=1.0, seconds=0.5),), seconds=2, folder="refresh")
        out = scene_sound.clip_sound(clip, refreshed.background, None, self.root / "refreshed.mp4")
        with av.open(str(out)) as container:
            self.assertEqual(len(container.streams.audio), 1)
            self.assertEqual(sum(1 for _ in container.decode(video=0)), 60)
        sound_track = media_audio.read(out)[0]
        effect = _rms(sound_track, 1.03, 1.3)  # the effect plays from 1.0 s, not a handle (0.25 s) late or early
        self.assertGreater(effect, 2 * _rms(sound_track, 0.55, 0.95))
        self.assertGreater(effect, 2 * _rms(sound_track, 0.8, 0.97))
        spectrum = np.abs(np.fft.rfft(sound_track[0, :RATE]))
        self.assertLess(spectrum[1000], 0.05 * spectrum.max())
        self.assertEqual(sorted(path.name for path in self.root.iterdir() if path.name.startswith(".")), [])
        with self.assertRaisesRegex(ValueError, "no sound"):
            scene_sound.clip_sound(clip, None, None, self.root / "none.mp4")


class SoundPromptTests(unittest.TestCase):
    def test_ezaudio_gets_the_bed_prompt_once_styled_and_each_effect_with_the_style(self):
        style = "high quality realistic sound effects recording"
        prompt = scene_sound.background_prompt("A man walks through a forest.", "")
        self.assertTrue(prompt.endswith(style))
        self.assertEqual(scene_sound.background_prompt("", "Steady rain."), f"Steady rain, {style}")
        self.assertEqual(sound.prompt_from_scene("", prompt), prompt)  # a ready prompt passes unchanged
        requests = []

        def render(text, seconds, seed, out_path, progress_cb=None, *, runtime):
            requests.append((text, seconds, seed))
            _tone(out_path, seconds + 2 * HANDLE, 220, level=0.05)

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(sound, "available", return_value=True), \
                patch.object(sound.engine_runtime, "background_audio", side_effect=render):
            result = scene_sound.prepare((), (), prompt, (Effect("glass breaks", 0.5, 1.0),), 2, 5,
                                         Path(folder) / "sound", client=MagicMock())
            self.assertEqual(_mono(result.background).size, int(2.5 * RATE))
        self.assertEqual(requests, [(prompt, 2, 5), (f"glass breaks, {style}", 1.0, 5 + 1009)])


class PlacementTests(unittest.TestCase):
    def test_lines_are_placed_summed_and_cut_only_at_the_scene_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, second = _tone(root / "a.wav", 1.0, 400, 0.6), _tone(root / "b.wav", 1.0, 400, 0.6)
            self.assertEqual(dialogue.durations([first, second]), [1.0, 1.0])
            out = root / "voice.wav"
            dialogue.place([first, second], [0.0, 0.0], out, 1.5)
            track, rate = sf.read(str(out), dtype="float32")
            self.assertEqual((track.size, rate), (int(1.5 * RATE), RATE))
            self.assertAlmostEqual(float(np.max(np.abs(track))), dialogue.PLACED_PEAK, delta=1e-3)  # 1.2 scaled
            dialogue.place([first], [1.0], out, 1.5)
            track = sf.read(str(out), dtype="float32")[0]
            self.assertEqual(track.size, int(1.5 * RATE))
            self.assertLess(float(np.max(np.abs(track[:RATE - 1]))), 1e-4)
            self.assertGreater(float(np.max(np.abs(track[RATE:]))), 0.5)
            with self.assertRaises(ValueError):
                dialogue.place([first], [], out, 1.5)

    def test_a_speakers_lip_sync_track_has_only_their_own_lines_where_lines_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rose, olaf = _tone(root / "rose.wav", 0.5, 300, 0.3), _tone(root / "olaf.wav", 1.0, 900, 0.3)
            voice = root / "voice.wav"
            timings = [(1.0, 1.5), (1.2, 2.2)]  # Olaf starts while Rose still talks
            dialogue.place([rose, olaf], [1.0, 1.2], voice, 3)
            for keep, own, other in (([0], 300, 900), ([1], 900, 300)):
                with self.subTest(keep=keep):
                    sliced, clean = root / "sliced.wav", root / "clean.wav"
                    dialogue.speaker_track(voice, timings, keep, sliced, 3)  # the summed track, as before
                    dialogue.speaker_track(voice, timings, keep, clean, 3, lines=[rose, olaf])
                    overlap = slice(int(1.25 * RATE), int(1.45 * RATE))
                    for path, heard in ((sliced, True), (clean, False)):
                        track = sf.read(str(path), dtype="float32")[0][overlap]
                        spectrum = np.abs(np.fft.rfft(track * np.hanning(track.size)))
                        bins = {hz: spectrum[round(hz * track.size / RATE)] for hz in (own, other)}
                        self.assertGreater(bins[own], 1.0)
                        self.assertEqual(bins[other] > 0.1 * bins[own], heard, path.name)
                    clean_track = sf.read(str(clean), dtype="float32")[0]
                    start, end = timings[keep[0]]
                    self.assertLess(float(np.max(np.abs(clean_track[:int((start - 0.01) * RATE)]))), 1e-4)
                    self.assertLess(float(np.max(np.abs(clean_track[int((end + 0.01) * RATE):]))), 1e-4)
            with self.assertRaisesRegex(ValueError, "own file"):
                dialogue.speaker_track(voice, timings, [0], root / "x.wav", 3, lines=[rose])


if __name__ == "__main__":
    unittest.main()
