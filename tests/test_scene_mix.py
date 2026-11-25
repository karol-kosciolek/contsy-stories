# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Scene voice and background levels in the timeline soundtrack (engine/media/audio.py), the scene
previews mixed the same way, and the background of a clip's own mix lined up with its picture."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app import ffmpeg_utils
from engine.media import audio, video
from engine.media.errors import MediaError

RATE = audio.RATE
HANDLE = audio.TIMELINE_AUDIO_HANDLE


def _legacy_soundtrack(stems, rate=RATE):
    """timeline_soundtrack as it was before scene levels existed: the reference the soundtrack without
    levels (or at 0/0) must equal sample for sample."""
    handle = round(HANDLE * rate)
    angle = np.linspace(0.0, np.pi / 2, 2 * handle, dtype=np.float32)
    fade_out, fade_in = np.cos(angle), np.sin(angle)
    pending = None
    for background, voice, seconds in stems:
        raw = audio._timeline_background(background, seconds, rate)
        core = raw[:, handle:-handle].copy()
        if pending is None:
            fade = min(core.shape[1], round(0.5 * rate))
            core[:, :fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        else:
            previous, previous_voice, previous_seconds, tail = pending
            joined = tail * fade_out + raw[:, :2 * handle] * fade_in
            previous[:, -handle:] = joined[:, :handle]
            core[:, :handle] = joined[:, handle:]
            yield audio._timeline_mix(previous, previous_voice, previous_seconds, rate)
        pending = core, voice, seconds, raw[:, -2 * handle:].copy()
    core, voice, seconds, _ = pending
    fade = min(core.shape[1], round(0.6 * rate))
    core[:, -fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    yield audio._timeline_mix(core, voice, seconds, rate)


def _rms(samples, start, end):
    window = samples[..., int(round(start * RATE)):int(round(end * RATE))]
    return float(np.sqrt(np.mean(window * window)))


def _tone_level(samples, start, end, frequency):
    """Amplitude of one frequency in a window of the left channel."""
    window = samples[0, int(round(start * RATE)):int(round(end * RATE))]
    spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size)))
    return float(spectrum[int(round(frequency * window.size / RATE))])


def _onset(samples, threshold):
    """Time of the first 10 ms block whose level reaches `threshold`."""
    block = RATE // 100
    count = samples.shape[-1] // block
    levels = np.sqrt(np.mean(samples[..., :count * block].reshape(-1, count, block) ** 2, axis=(0, 2)))
    return int(np.argmax(levels >= threshold)) * block / RATE


class Scenes:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def background(self, name, seconds, frequency=200, level=0.05, burst=None, noise=0.0):
        """A handle-padded scene background (it starts HANDLE before the scene); burst = (scene time,
        length) adds a loud 1500 Hz event at that time."""
        t = np.arange(int(round((seconds + 2 * HANDLE) * RATE))) / RATE
        wave = level * np.sin(2 * np.pi * frequency * t)
        if noise:
            wave += noise * np.random.default_rng(len(name)).standard_normal(t.size)
        if burst is not None:
            start = int(round((HANDLE + burst[0]) * RATE))
            stop = start + int(round(burst[1] * RATE))
            wave[start:stop] += 0.3 * np.sin(2 * np.pi * 1500 * t[start:stop])
        audio.write_wav(self.dir / name, wave[None, :].astype(np.float32), RATE)
        return self.dir / name

    def voice(self, name, seconds, start, end, frequency=700, level=0.3, bursts=False):
        """Dialogue on the scene clock: a tone from `start` to `end` (short loud bursts with `bursts`)."""
        t = np.arange(int(round(seconds * RATE))) / RATE
        wave = np.zeros(t.size)
        speech = (t >= start) & (t < end)
        wave[speech] = level * np.sin(2 * np.pi * frequency * t[speech])
        if bursts:
            wave[speech & ((t * 4) % 1 >= 0.2)] = 0.0
        audio.write_wav(self.dir / name, wave[None, :].astype(np.float32), RATE)
        return self.dir / name


class TimelineLevelTests(Scenes, unittest.TestCase):
    def test_no_levels_and_zero_levels_are_the_soundtrack_as_it_was(self):
        stems = [(self.background("a.wav", 2, 210, noise=0.01), None, 2),
                 (self.background("b.wav", 3, 260, noise=0.01), self.voice("v1.wav", 3, 0.4, 1.0), 3),
                 (self.background("c.wav", 1, 330, noise=0.01), self.voice("v2.wav", 1, 0.2, 0.6), 1)]
        reference = list(_legacy_soundtrack(stems))
        for levels in (None, [(0.0, 0.0)] * 3, [None] * 3, [(0, 0), (-0.0, 0.0), None]):
            with self.subTest(levels=levels):
                mixed = list(audio.timeline_soundtrack(stems, levels=levels))
                self.assertEqual(len(mixed), 3)
                for scene, expected in zip(mixed, reference):
                    self.assertEqual(scene.dtype, expected.dtype)
                    self.assertTrue(np.array_equal(scene, expected))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(audio.timeline_soundtrack(stems), reference)))

    def test_background_and_voice_levels_apply_after_the_leveling(self):
        stems = [(self.background("bed.wav", 4), self.voice("voice.wav", 4, 2.0, 2.6), 4)]
        base = next(audio.timeline_soundtrack(stems))
        louder = next(audio.timeline_soundtrack(stems, levels=[(0.0, 6.0)]))
        quieter = next(audio.timeline_soundtrack(stems, levels=[(0.0, -6.0)]))
        voice = next(audio.timeline_soundtrack(stems, levels=[(6.0, 0.0)]))
        self.assertAlmostEqual(_rms(louder, 0.6, 1.8) / _rms(base, 0.6, 1.8), 10 ** (6 / 20), delta=0.02)
        self.assertAlmostEqual(_rms(quieter, 0.6, 1.8) / _rms(base, 0.6, 1.8), 10 ** (-6 / 20), delta=0.01)
        self.assertAlmostEqual(_tone_level(voice, 2.1, 2.5, 700) / _tone_level(base, 2.1, 2.5, 700),
                               10 ** (6 / 20), delta=0.05)
        before = int(1.8 * RATE)  # nothing but the background before the line
        self.assertTrue(np.allclose(voice[:, :before], base[:, :before], atol=1e-6))
        for mixed in (base, louder, quieter, voice):
            self.assertEqual(mixed.shape, (2, 4 * RATE))
            self.assertLessEqual(float(np.max(np.abs(mixed))), 1.0)

    def test_a_join_fades_from_one_scene_level_to_the_next(self):
        stems = [(self.background("a.wav", 2), None, 2), (self.background("b.wav", 2), None, 2)]
        first, second = audio.timeline_soundtrack(stems, levels=[(0.0, 0.0), (0.0, 6.0)])
        plain_first, plain_second = audio.timeline_soundtrack(stems)
        self.assertAlmostEqual(_rms(first, 0.8, 1.6) / _rms(plain_first, 0.8, 1.6), 1.0, delta=0.01)
        self.assertAlmostEqual(_rms(second, 0.4, 1.2) / _rms(plain_second, 0.4, 1.2), 10 ** (6 / 20), delta=0.02)
        # the cross-fade around the join is between the two levels, without a jump
        edge = np.concatenate([first[:, -RATE // 10:], second[:, :RATE // 10]], axis=1)
        self.assertLess(float(np.max(np.abs(np.diff(edge, axis=1)))), 0.05)

    def test_loud_levels_are_limited_below_the_ceiling(self):
        stems = [(self.background("bed.wav", 3), self.voice("voice.wav", 3, 0.5, 2.5, 300, 0.9, bursts=True), 3)]
        limited = next(audio.timeline_soundtrack(stems, levels=[(12.0, 12.0)]))
        with patch.object(audio, "limit_peaks", side_effect=lambda samples, *args, **kwargs: samples):
            unlimited = next(audio.timeline_soundtrack(stems, levels=[(12.0, 12.0)]))
        self.assertGreater(float(np.max(np.abs(unlimited))), audio.LEVEL_CEILING)
        self.assertLessEqual(float(np.max(np.abs(limited))), audio.LEVEL_CEILING + 1e-6)
        self.assertEqual(limited.shape, unlimited.shape)
        self.assertAlmostEqual(_rms(limited, 0.05, 0.45) / _rms(unlimited, 0.05, 0.45), 1.0, delta=0.02)

    def test_limit_peaks_leaves_quiet_audio_alone(self):
        quiet = np.full((2, 4800), 0.5, dtype=np.float32)
        self.assertIs(audio.limit_peaks(quiet), quiet)
        loud = np.zeros((2, RATE), dtype=np.float32)
        loud[:, :RATE // 2] = 0.2
        loud[:, RATE // 2:RATE // 2 + 48] = 1.7
        out = audio.limit_peaks(loud)
        self.assertEqual(out.shape, loud.shape)
        self.assertLessEqual(float(np.max(np.abs(out))), audio.LEVEL_CEILING)
        self.assertTrue(np.allclose(out[:, :RATE // 4], loud[:, :RATE // 4], atol=1e-4))

    def test_levels_must_be_numbers_for_every_scene(self):
        stems = [(self.background("a.wav", 1), None, 1), (self.background("b.wav", 1), None, 1)]
        with self.assertRaisesRegex(MediaError, "level is required for each scene"):
            list(audio.timeline_soundtrack(stems, levels=[(0.0, 0.0)]))
        with self.assertRaisesRegex(MediaError, "two numbers"):
            list(audio.timeline_soundtrack(stems, levels=[(0.0, float("nan")), None]))

    def test_scenes_of_tenths_of_a_second_join_exactly(self):
        """A scene may last 2.5 s (the owner's request of 2026-11-16): its frames and its sound join on the clock."""
        clips = []
        for name, frames in (("a.mp4", 45), ("b.mp4", 75)):
            clip = self.dir / name
            with video.Writer(clip, 30, 48, 32, preset="ultrafast") as writer:
                for _ in range(frames):
                    writer.write(np.full((32, 48, 3), 90, dtype=np.uint8))
            clips.append(clip)
        stems = [(self.background("a.wav", 1.5), None), (self.background("b.wav", 2.5), None)]
        joined = self.dir / "joined.mp4"
        self.assertEqual(video.join_timeline(list(zip(clips, (1.5, 2.5))), joined, 30, sound_stems=stems), 120)
        self.assertAlmostEqual(audio.read(joined)[0].shape[1] / audio.RATE, 4.0, delta=0.03)
        silent = self.dir / "silent.mp4"
        self.assertEqual(video.join_timeline(list(zip(clips, (1.5, 2.5))), silent, 30), 120)

    def test_join_timeline_mixes_the_soundtrack_with_the_scene_levels(self):
        clip = self.dir / "scene.mp4"
        with video.Writer(clip, 30, 48, 32, preset="ultrafast") as writer:
            for _ in range(60):
                writer.write(np.full((32, 48, 3), 90, dtype=np.uint8))
        stems = [(self.background("bed.wav", 2), None)]
        outputs = {}
        for name, levels in (("plain", None), ("louder", [(0.0, 6.0)])):
            outputs[name] = self.dir / f"{name}.mp4"
            with patch.object(video._audio, "timeline_soundtrack", wraps=audio.timeline_soundtrack) as soundtrack:
                self.assertEqual(video.join_timeline([(clip, 2)], outputs[name], 30, sound_stems=stems,
                                                     sound_levels=levels), 60)
            self.assertEqual(soundtrack.call_args.kwargs["levels"], levels)
        plain, louder = (audio.read(outputs[name])[0] for name in ("plain", "louder"))
        self.assertAlmostEqual(_rms(louder, 0.6, 1.3) / _rms(plain, 0.6, 1.3), 10 ** (6 / 20), delta=0.1)
        with self.assertRaisesRegex(MediaError, "level is required for each scene"):
            video.join_timeline([(clip, 2)], self.dir / "bad.mp4", 30, sound_stems=stems, sound_levels=[])


class SceneMixTests(Scenes, unittest.TestCase):
    def test_a_preview_is_the_export_mix_of_the_scene(self):
        background = self.background("bed.wav", 3, burst=(1.0, 0.3))
        voice = self.voice("voice.wav", 3, 1.8, 2.2)
        preview = audio.scene_mix(background, voice, 3)
        export = next(audio.timeline_soundtrack([(background, voice, 3)]))
        self.assertEqual(preview.shape, (2, 3 * RATE))
        middle = slice(int(0.6 * RATE), int(2.3 * RATE))  # away from the export's longer fades
        self.assertTrue(np.allclose(preview[:, middle], export[:, middle], atol=1e-6))
        for edge in (preview[:, :1], preview[:, -1:]):  # short fades: no click
            self.assertEqual(float(np.max(np.abs(edge))), 0.0)
        self.assertLess(float(np.max(np.abs(preview[:, :24]))), 1e-3)
        self.assertGreater(_rms(preview, 0.06, 0.4), 0.5 * _rms(preview, 0.6, 0.9))  # but only short ones
        # the handle is trimmed: the event at 1.0 s of the scene is heard at 1.0 s
        self.assertAlmostEqual(_onset(preview, 0.08), 1.0, delta=0.02)

    def test_preview_levels_and_missing_parts(self):
        background = self.background("bed.wav", 3)
        voice = self.voice("voice.wav", 3, 1.8, 2.2)
        base = audio.scene_mix(background, voice, 3)
        louder = audio.scene_mix(background, voice, 3, background_db=6.0)
        self.assertAlmostEqual(_rms(louder, 0.6, 1.6) / _rms(base, 0.6, 1.6), 10 ** (6 / 20), delta=0.02)
        boosted = audio.scene_mix(background, voice, 3, voice_db=6.0)
        self.assertAlmostEqual(_tone_level(boosted, 1.85, 2.15, 700) / _tone_level(base, 1.85, 2.15, 700),
                               10 ** (6 / 20), delta=0.05)
        silent = self.dir / "silent.wav"
        audio.write_wav(silent, np.zeros((1, int(3.5 * RATE)), dtype=np.float32), RATE)
        only_voice = audio.scene_mix(silent, voice, 3)
        self.assertLess(_rms(only_voice, 0.2, 1.6), 1e-6)
        self.assertGreater(_rms(only_voice, 1.85, 2.15), 0.01)
        self.assertEqual(audio.scene_mix(None, voice, 3).shape, (2, 3 * RATE))
        self.assertEqual(audio.scene_mix(background, None, 3).shape, (2, 3 * RATE))
        with self.assertRaises(MediaError):
            audio.scene_mix(None, None, 3)


class EffectsOverlayTests(Scenes, unittest.TestCase):
    """Effects laid into the background are kept apart (the overlay): only the room tone is leveled,
    edge-filled, joined and faded; the effects play at their own time and level."""

    def layered(self, name, seconds, bed_level, effect_start, effect_seconds, noise=True):
        """(background, overlay): a noise or tone bed with a 1500 Hz effect from `effect_start` (scene time)."""
        rng = np.random.default_rng(len(name))
        length = int(round((seconds + 2 * HANDLE) * RATE))
        t = np.arange(length) / RATE
        bed = bed_level * (rng.standard_normal(length) if noise else np.sin(2 * np.pi * 210 * t))
        overlay = np.zeros(length)
        start = int(round((HANDLE + effect_start) * RATE))
        stop = min(start + int(round(effect_seconds * RATE)), int(round((HANDLE + seconds) * RATE)))
        overlay[start:stop] = 0.1 * np.sin(2 * np.pi * 1500 * t[start:stop])
        background, effects = self.dir / f"{name}.wav", self.dir / f"{name}.fx.wav"
        audio.write_wav(background, (bed + overlay)[None, :].astype(np.float32), RATE)
        audio.write_wav(effects, overlay[None, :].astype(np.float32), RATE)
        return background, effects

    def test_an_effect_never_plays_before_its_time_or_sets_the_room_tone_level(self):
        background, effects = self.layered("engine", 3, 1e-3, 0.8, 2.0)  # the effect fills most of the stem
        baked = audio.scene_mix(background, None, 3)  # the effects treated as room tone (no overlay known)
        apart = audio.scene_mix(background, None, 3, effects=effects)
        self.assertGreater(_tone_level(baked, 0.1, 0.5, 1500), 1.0)  # it leaks into the quiet start
        self.assertLess(_tone_level(apart, 0.1, 0.5, 1500), 0.05 * _tone_level(apart, 1.2, 1.6, 1500))
        self.assertGreater(_tone_level(apart, 1.2, 1.6, 1500), 2 * _tone_level(baked, 1.2, 1.6, 1500))
        # the room tone is leveled as if the effect were not there
        bed_only = self.dir / "bed-only.wav"
        samples = audio.read(background, channels=1)[0] - audio.read(effects, channels=1)[0]
        audio.write_wav(bed_only, samples, RATE)
        alone = audio.scene_mix(bed_only, None, 3)
        self.assertAlmostEqual(_rms(apart, 0.1, 0.7), _rms(alone, 0.1, 0.7), delta=0.1 * _rms(alone, 0.1, 0.7))
        export = next(audio.timeline_soundtrack([(background, None, 3, effects)]))
        self.assertLess(_tone_level(export, 0.1, 0.5, 1500), 0.05 * _tone_level(export, 1.2, 1.6, 1500))
        self.assertAlmostEqual(_tone_level(export, 1.2, 1.6, 1500) / _tone_level(apart, 1.2, 1.6, 1500), 1.0,
                               delta=0.05)
        # the room tone keeps its own ceiling; the effect may rise above it, up to EFFECT_PEAK
        self.assertLessEqual(float(np.max(np.abs(export[:, :int(0.7 * RATE)]))), audio.BACKGROUND_PEAK + 1e-3)
        self.assertLessEqual(float(np.max(np.abs(export))), audio.EFFECT_PEAK + 1e-3)

    def test_a_short_click_stays_audible_over_the_rain(self):
        """A shutter or switch click is a few loud milliseconds: the room-tone limiter must not flatten it."""
        rng = np.random.default_rng(7)
        seconds, length = 4, int(round((4 + 2 * HANDLE) * RATE))
        bed = 0.003 * rng.standard_normal(length)  # quiet rain: the export lifts it about 8x
        overlay = np.zeros(length)
        start = int(round((HANDLE + 2.0) * RATE))
        count = int(round(0.06 * RATE))
        decay = np.exp(-np.arange(count) / RATE / 0.012)  # a sharp attack that dies away within ~30 ms
        overlay[start:start + count] = 0.12 * decay * rng.standard_normal(count)
        background, effects = self.dir / "rain.wav", self.dir / "rain.fx.wav"
        audio.write_wav(background, (bed + overlay)[None, :].astype(np.float32), RATE)
        audio.write_wav(effects, overlay[None, :].astype(np.float32), RATE)
        export = next(audio.timeline_soundtrack([(background, None, seconds, effects)]))
        loudest = max(_rms(export, 2.0 + step / 200, 2.05 + step / 200) for step in range(0, 8))
        rain = _rms(export, 0.8, 1.8)
        # measured: +5.9 dB with the room tone's 0.2 ceiling; the balanced click with EFFECT_PEAK keeps more
        self.assertGreater(20 * np.log10(loudest / rain), 8.0)
        self.assertLessEqual(float(np.max(np.abs(export))), audio.LEVEL_CEILING + 1e-3)

    def test_a_click_right_after_a_line_is_not_swallowed_by_the_ducking(self):
        """The room tone ducks under dialogue and recovers over ~0.4 s; effects are not part of that ducking, so a
        click 0.2 s after a line keeps its level, and an effect under a spoken line is only softened by 6 dB."""
        rng = np.random.default_rng(5)
        seconds, length = 5, int(round((5 + 2 * HANDLE) * RATE))
        bed = 0.02 * rng.standard_normal(length)
        overlay = np.zeros(length)
        count = int(round(0.06 * RATE))
        decay = np.exp(-np.arange(count) / RATE / 0.012)
        for at in (1.0, 2.2):  # under the line, and 0.2 s after it ends
            start = int(round((HANDLE + at) * RATE))
            overlay[start:start + count] = 0.08 * decay * rng.standard_normal(count)
        background, effects = self.dir / "talk.wav", self.dir / "talk.fx.wav"
        audio.write_wav(background, (bed + overlay)[None, :].astype(np.float32), RATE)
        audio.write_wav(effects, overlay[None, :].astype(np.float32), RATE)
        voice = self.voice("line.wav", seconds, 0.5, 2.0)
        silent = self.voice("none.wav", seconds, 4.9, 4.9)
        talking = next(audio.timeline_soundtrack([(background, voice, seconds, effects)]))
        alone = next(audio.timeline_soundtrack([(background, silent, seconds, effects)]))

        def loudest(samples, at):
            return max(_rms(samples, at + step / 200, at + 0.05 + step / 200) for step in range(0, 8))

        def treble(samples, start, end):
            """Level above 2 kHz in a window: the click without the line's 700 Hz tone."""
            window = samples[0, int(round(start * RATE)):int(round(end * RATE))]
            spectrum = np.fft.rfft(window)
            spectrum[:int(2000 * window.size / RATE)] = 0
            return float(np.sqrt((np.abs(spectrum) ** 2).sum()) / window.size)

        self.assertGreater(loudest(talking, 2.2) / loudest(alone, 2.2), audio.gain(-1.0))  # not ducked after the line
        self.assertGreater(20 * np.log10(loudest(talking, 2.2) / _rms(talking, 3.0, 4.0)), 8.0)
        softened = 20 * np.log10(treble(talking, 1.0, 1.06) / treble(alone, 1.0, 1.06))
        self.assertAlmostEqual(softened, -6.0, delta=1.0)  # the click under the line

    def test_each_effect_is_balanced_without_the_creator_touching_a_level(self):
        """A generator gives effects any level: a deafening door and a faint click in one scene both land in
        EFFECT_RANGE_DB over the export's room tone."""
        rng = np.random.default_rng(11)
        seconds, length = 6, int(round((6 + 2 * HANDLE) * RATE))
        bed = 0.02 * rng.standard_normal(length)
        overlay = np.zeros(length)
        door, click = int(round((HANDLE + 1.0) * RATE)), int(round((HANDLE + 4.0) * RATE))
        overlay[door:door + int(0.8 * RATE)] = 0.6 * rng.standard_normal(int(0.8 * RATE))  # far too loud
        count = int(round(0.06 * RATE))
        overlay[click:click + count] = 0.01 * rng.standard_normal(count)  # far too quiet (about -40 dBFS)
        background, effects = self.dir / "room.wav", self.dir / "room.fx.wav"
        audio.write_wav(background, (bed + overlay)[None, :].astype(np.float32), RATE)
        audio.write_wav(effects, overlay[None, :].astype(np.float32), RATE)
        layer = audio._scene_effects(effects, 1.0, seconds, RATE)
        low, high = (audio.BED_TARGET * audio.gain(db) for db in audio.EFFECT_RANGE_DB)
        regions = audio._effect_regions(layer, RATE)
        self.assertEqual(len(regions), 2)
        for start, stop in regions:
            loud = audio._loudest(layer[:, start:stop], RATE)
            self.assertGreaterEqual(loud, low * 0.99)
            self.assertLessEqual(loud, high * 1.01)
        export = next(audio.timeline_soundtrack([(background, None, seconds, effects)]))
        self.assertLessEqual(float(np.max(np.abs(export))), audio.LEVEL_CEILING + 1e-3)

    def test_the_movie_fades_leave_effects_at_the_start_and_end_alone(self):
        first = self.layered("first", 2, 0.02, 0.05, 0.4, noise=False)
        last = self.layered("last", 2, 0.02, 1.5, 0.45, noise=False)
        start, end = audio.timeline_soundtrack([(*first[:1], None, 2, first[1]), (*last[:1], None, 2, last[1])])
        previews = [audio.scene_mix(first[0], None, 2, effects=first[1]),
                    audio.scene_mix(last[0], None, 2, effects=last[1])]
        for exported, preview, window in ((start, previews[0], (0.1, 0.4)), (end, previews[1], (1.55, 1.85))):
            self.assertAlmostEqual(_tone_level(exported, *window, 1500) / _tone_level(preview, *window, 1500), 1.0,
                                   delta=0.08)
        # the room tone itself still fades in and out with the movie
        self.assertLess(_tone_level(start, 0.0, 0.1, 210), 0.5 * _tone_level(start, 0.7, 0.8, 210))

    def test_a_silent_room_tone_keeps_the_effects_at_their_time(self):
        background, effects = self.layered("silent", 3, 0.0, 2.0, 0.5)
        baked = next(audio.timeline_soundtrack([(background, None, 3)]))  # the effect taken for room tone
        self.assertGreater(_tone_level(baked, 0.1, 0.5, 1500), 1.0)
        export = next(audio.timeline_soundtrack([(background, None, 3, effects)]))
        self.assertEqual(export.shape, (2, 3 * RATE))
        self.assertLess(_rms(export, 0.1, 1.9), 1e-4)
        self.assertGreater(_tone_level(export, 2.05, 2.45, 1500), 1.0)
        preview = audio.scene_mix(background, None, 3, effects=effects)
        self.assertLess(_rms(preview, 0.1, 1.9), 1e-4)
        self.assertGreater(_tone_level(preview, 2.05, 2.45, 1500), 1.0)

    def test_scenes_without_an_overlay_mix_exactly_as_before(self):
        stems = [(self.background("a.wav", 2, 210, noise=0.01), None, 2),
                 (self.background("b.wav", 2, 260, noise=0.01), self.voice("v.wav", 2, 0.4, 1.0), 2)]
        reference = list(_legacy_soundtrack(stems))
        for mixed, expected in zip(audio.timeline_soundtrack([(*stem, None) for stem in stems]), reference):
            self.assertTrue(np.array_equal(mixed, expected))
        clip = self.dir / "scene.mp4"
        with video.Writer(clip, 30, 48, 32, preset="ultrafast") as writer:
            for _ in range(60):
                writer.write(np.full((32, 48, 3), 90, dtype=np.uint8))
        background, effects = self.layered("joined", 2, 0.02, 0.5, 0.5)
        seen, soundtrack = [], audio.timeline_soundtrack

        def capture(stems, *args, **kwargs):
            seen.extend(stems)
            return soundtrack(seen, *args, **kwargs)

        with patch.object(video._audio, "timeline_soundtrack", side_effect=capture):
            video.join_timeline([(clip, 2)], self.dir / "out.mp4", 30, sound_stems=[(background, None, effects)])
        self.assertEqual(seen, [(background, None, 2, effects)])

    def test_background_loudness_is_the_level_the_export_reaches(self):
        quiet, quiet_effects = self.layered("quiet", 3, 2e-5, 1.0, 0.5)
        loud, loud_effects = self.layered("loud", 3, 0.004, 1.0, 1.5)
        level, audible = audio.background_loudness(loud, 3, effects=loud_effects)
        self.assertAlmostEqual(level, audio.BED_TARGET, delta=0.003)
        self.assertEqual(audible, 1.0)
        self.assertGreater(audio.background_loudness(quiet, 3)[1], 0.0)  # the effect looks like sound...
        self.assertEqual(audio.background_loudness(quiet, 3, effects=quiet_effects), (0.0, 0.0))  # ...the bed is not


class ClipMixTests(Scenes, unittest.TestCase):
    def test_the_clip_mix_skips_the_background_handle(self):
        background = self.background("bed.wav", 3, burst=(1.5, 0.3))
        aligned, early = self.dir / "aligned.wav", self.dir / "early.wav"
        self.assertTrue(ffmpeg_utils.mix_voice_and_background(None, str(background), aligned, 3.0,
                                                              background_offset=HANDLE)[0])
        self.assertTrue(ffmpeg_utils.mix_voice_and_background(None, str(background), early, 3.0)[0])
        for path, onset in ((aligned, 1.5), (early, 1.5 + HANDLE)):
            mixed, _ = audio.read(path)
            self.assertEqual(mixed.shape, (2, 3 * RATE))
            peak = float(np.max(np.sqrt(np.mean(mixed[:, :mixed.shape[1] // 480 * 480].reshape(2, -1, 480) ** 2,
                                                axis=(0, 2)))))
            self.assertAlmostEqual(_onset(mixed, 0.5 * peak), onset, delta=0.02)


if __name__ == "__main__":
    unittest.main()
