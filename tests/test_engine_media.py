# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Media layer of the engine (PyAV): video files, joins, soundtrack. Small synthetic clips, no ffmpeg program."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAVE = importlib.util.find_spec("av") is not None and importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAVE, "PyAV is not installed")
class MediaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def clip(self, name, frames, fps=16, first=0, size=(64, 48), crf=10, preset="ultrafast", tagged=False):
        """Frame i is a flat grey of level 8 * (first + i): the level tells which frame a picture is.
        `tagged` marks the stream BT.709, limited range, like the videos of the product. The default preset has
        no B-frames; the product's presets do."""
        import numpy as np

        from engine.media import video
        path = self.dir / name
        tags = {"colorspace": 1, "color_range": 1} if tagged else {}
        with video.Writer(path, fps, size[0], size[1], crf=crf, preset=preset, **tags) as writer:
            for i in range(frames):
                writer.write(np.full((size[1], size[0], 3), min(8 * (first + i), 255), np.uint8))
        return path

    def levels(self, path):
        from engine.media import video
        return [int(video.to_rgb(f).mean() / 8 + 0.5) for f in video.read_frames(path)]

    def tone(self, name, seconds, hz=440, level=0.3, rate=48000):
        import numpy as np

        from engine.media import audio
        t = np.arange(int(seconds * rate)) / rate
        samples = np.stack([level * np.sin(2 * np.pi * hz * t)] * 2).astype(np.float32)
        audio.write_wav(self.dir / name, samples, rate)
        return self.dir / name

    def test_filter_text_is_split_into_filters_and_arguments(self):
        from engine.media.graph import split_chain
        self.assertEqual(split_chain("highpass=f=40:poles=2,apad"), [("highpass", "f=40:poles=2"), ("apad", None)])
        self.assertEqual(split_chain("select='eq(n,1)+eq(n,3)',null"), [("select", "'eq(n,1)+eq(n,3)'"), ("null", None)])
        with self.assertRaises(ValueError):
            split_chain("fps=30,,null")

    def test_writer_makes_a_constant_rate_h264_file_with_the_comment(self):
        import av

        from engine.media import video
        path = self.clip("a.mp4", 17)
        info = video.probe(path)
        self.assertEqual((info.width, info.height, info.fps, info.has_audio), (64, 48, 16.0, False))
        self.assertAlmostEqual(info.duration, 17 / 16, places=2)
        self.assertEqual(self.levels(path), list(range(17)))
        video.tag_metadata(path)
        with av.open(str(path)) as container:
            self.assertEqual(container.streams.video[0].codec_context.name, "h264")
            self.assertEqual(container.metadata.get("comment"), video.AI_COMMENT)
        self.assertEqual(self.levels(path), list(range(17)))  # the remux copied the pictures

    def test_convert_fps_picks_frames_by_time_and_keeps_the_length(self):
        from engine.media import video
        source = self.clip("a.mp4", 17)
        count = video.convert_fps(source, self.dir / "b.mp4", 30, 17 / 16)
        out = video.probe(self.dir / "b.mp4")
        self.assertEqual((count, out.fps), (32, 30.0))
        self.assertAlmostEqual(out.duration, 32 / 30, places=2)
        levels = self.levels(self.dir / "b.mp4")
        self.assertEqual(levels[0], 0)
        self.assertEqual(levels, sorted(levels))

    def test_join_drops_the_repeated_frame_of_every_following_shot(self):
        from engine.media import video
        first, second, third = self.clip("1.mp4", 9), self.clip("2.mp4", 9, first=8), self.clip("3.mp4", 9, first=16)
        count = video.join_segments([first, second, third], self.dir / "j.mp4", 16)
        self.assertEqual(count, 25)  # 9 + 8 + 8
        self.assertEqual(self.levels(self.dir / "j.mp4"), list(range(25)))
        cut = video.join_segments([first, second, third], self.dir / "k.mp4", 16, seconds=1.0)
        self.assertEqual(cut, 16)

    def test_overlapped_parts_sit_on_the_output_clock(self):
        """Three parts as the video page makes them (2 s each, the later ones 4 Wan frames longer and starting from
        the frame 0.25 s before the end of the one before), retimed like RIFE does: every frame carries its true
        moment in its pixels. Outside the cross-fades every output frame must show its own moment, within half a
        frame, at 25 and at 30 fps (a probed duration counts the last frame's interval and put each seam late)."""
        import numpy as np

        from engine.media import video
        from engine.media.retime import frame_rate_plan

        width, height = 64, 32

        def picture(units):  # a moment in 1/48 s: coarse level on the left, fine on the right
            array = np.zeros((height, width, 3), np.uint8)
            array[:, :width // 2] = (units // 16) * 4
            array[:, width // 2:] = (units % 16) * 16
            return array

        def shown(path):
            moments = []
            for frame in video.read_frames(path):
                array = video.to_rgb(frame).astype(float)
                coarse = round(array[4:-4, 4:width // 2 - 4].mean() / 4)
                fine = round(array[4:-4, width // 2 + 4:-4].mean() / 16)
                moments.append((coarse * 16 + fine) / 48)
            return moments

        for fps in (25, 30):
            parts, overlaps, handoffs, starts, start = [], [], [], [], 0.0
            for k in range(3):
                frames = 16 * 2 + 1 + (4 if k else 0)
                last = frames - 1 - (4 if k < 2 else 0)  # the frame the next part starts from
                _tasks, layout = frame_rate_plan(frames, 16, fps)
                path = self.dir / f"part{fps}_{k}.mp4"
                with video.Writer(path, fps, width, height, crf=0, preset="ultrafast") as writer:
                    for index, (pair, slot) in enumerate(layout):
                        local = pair / 16 if slot < 0 else index / fps
                        writer.write(picture(round((start + local) * 48)))
                parts.append(path)
                overlaps.append(0.25 if k else 0.0)
                handoffs.append(last / 16)
                starts.append(start)
                start += last / 16
            out = self.dir / f"joined{fps}.mp4"
            self.assertEqual(video.join_segments(parts, out, fps, overlaps, 6.0, handoffs=handoffs), 6 * fps)
            fades = [(begin - 1 / fps, begin + 0.125 + 1 / fps) for begin in starts[1:]]
            for index, moment in enumerate(shown(out)):
                now = index / fps
                if not any(low <= now <= high for low, high in fades):
                    # half a frame (0.25 s is not a whole number of frames) plus the 1/96 s rounding of the moments
                    self.assertLessEqual(abs(moment - now), 0.5 / fps + 1 / 96 + 1e-6, (fps, index))

    def test_join_with_an_overlap_cross_fades_the_shots(self):
        from engine.media import video
        first, second = self.clip("1.mp4", 17), self.clip("2.mp4", 17, first=200 // 8)
        video.join_segments([first, second], self.dir / "j.mp4", 16, overlaps=[0, 0.5], handoffs=[16 / 16 - 0.5])
        info = video.probe(self.dir / "j.mp4")
        self.assertAlmostEqual(info.duration, 2 * 17 / 16 - 0.5, delta=0.1)
        levels = self.levels(self.dir / "j.mp4")
        self.assertTrue(any(17 <= level <= 24 for level in levels), levels)  # a hard cut would jump from 16 to 25

    def test_the_last_frame(self):
        import numpy as np
        from PIL import Image

        from engine.media import video
        path = self.clip("a.mp4", 33)  # 2.0625 s
        video.last_frame(path, self.dir / "last.png")
        grey = lambda name: int(np.asarray(Image.open(self.dir / name).convert("RGB")).mean() / 8 + 0.5)  # noqa: E731
        self.assertEqual(grey("last.png"), 32)

    def test_narration_longer_than_the_video_holds_the_last_frame(self):
        from engine.media import audio, video
        source, narration = self.clip("a.mp4", 17), self.tone("voice.wav", 3.0)
        video.combine_video_audio(source, narration, self.dir / "out.mp4")
        info = video.probe(self.dir / "out.mp4")
        self.assertTrue(info.has_audio)
        self.assertAlmostEqual(info.duration, 3.3, delta=0.1)  # the whole line and a short breath
        self.assertEqual(set(self.levels(self.dir / "out.mp4")[17:]), {16})  # the last picture is held
        samples, _ = audio.read(self.dir / "out.mp4")
        self.assertAlmostEqual(samples.shape[1] / 48000, 3.3, delta=0.1)

    def test_shorter_audio_is_padded_with_silence_and_the_picture_is_copied(self):
        from engine.media import audio, video
        source, short = self.clip("a.mp4", 33), self.tone("short.wav", 1.0)
        video.combine_video_audio(source, short, self.dir / "out.mp4")
        self.assertAlmostEqual(video.probe(self.dir / "out.mp4").duration, 33 / 16, delta=0.05)
        self.assertEqual(self.levels(self.dir / "out.mp4"), list(range(33)))
        samples, _ = audio.read(self.dir / "out.mp4")
        self.assertLess(abs(samples[:, 60000:]).max(), 0.01)  # silence after the tone
        video.combine_video_audio(source, None, self.dir / "plain.mp4", compress=True)
        self.assertFalse(video.probe(self.dir / "plain.mp4").has_audio)

    def measure(self, samples):
        """(integrated loudness in LUFS, true peak in dBTP) as loudnorm measures them."""
        from engine.media import audio, graph
        with graph.libav_messages() as lines:
            audio.run_chain(samples, 48000, "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json")
            graph.release()
        found = next(m for _, _, m in lines if "input_i" in m)
        data = json.loads(found[found.index("{"):found.rindex("}") + 1])
        return float(data["input_i"]), float(data["input_tp"])

    def noise(self, seconds, level, seed):
        import numpy as np
        return (level * np.random.default_rng(seed).standard_normal((2, int(seconds * 48000)))).astype(np.float32)

    def test_soundtrack_has_the_exact_length_and_the_web_loudness(self):
        import numpy as np

        from engine.media import audio
        t = np.arange(48000) / 48000
        voice = np.stack([0.4 * np.sin(2 * np.pi * 200 * t)] * 2).astype(np.float32)
        for tracks in ((voice, self.noise(3.0, 0.1, 1)), (voice, None), (None, self.noise(3.0, 0.1, 2))):
            mixed = audio.voice_over_background(*tracks, seconds=2.0)
            self.assertEqual(mixed.shape, (2, 96000))
            loudness, peak = self.measure(mixed)
            self.assertAlmostEqual(loudness, -16.0, delta=0.3)
            self.assertLess(peak, -1.0)
        with self.assertRaises(Exception):
            audio.voice_over_background(None, None, 2.0)

    def test_quiet_background_with_unmeasurable_loudness_still_mixes(self):
        import numpy as np

        from engine.media import audio

        for level in (0, 1e-5):
            with self.subTest(level=level):
                mixed = audio.voice_over_background(None, self.noise(3.0, level, 17), 3.0)
                self.assertEqual(mixed.shape, (2, 144000))
                self.assertTrue(np.isfinite(mixed).all())
                self.assertEqual(bool(np.any(mixed)), level > 0)

    def test_background_is_ducked_while_the_narrator_speaks(self):
        import numpy as np

        from engine.media import audio
        rate = 48000
        t = np.arange(4 * rate) / rate
        voice = np.stack([0.5 * np.sin(2 * np.pi * 300 * t)] * 2).astype(np.float32)
        voice[:, :int(1.0 * rate)] = 0
        voice[:, int(1.6 * rate):] = 0  # speaks from 1.0 s to 1.6 s
        background = self.noise(4.0, 0.2, 3)

        def hiss(samples, begin, end):
            """Level of the part above 2 kHz (the narrator's 300 Hz tone is left out) in a window."""
            window = samples[0, int(begin * rate):int(end * rate)]
            spectrum = np.fft.rfft(window)
            spectrum[:int(2000 * len(window) / rate)] = 0
            return float(np.sqrt((np.abs(spectrum) ** 2).sum()) / len(window))

        with_voice = audio.voice_over_background(voice, background, 4.0)
        alone = audio.voice_over_background(None, background, 4.0)
        ducked = hiss(with_voice, 1.1, 1.5) / hiss(with_voice, 2.4, 2.8)
        plain = hiss(alone, 1.1, 1.5) / hiss(alone, 2.4, 2.8)
        self.assertAlmostEqual(plain, 1.0, delta=0.2)
        self.assertLess(ducked, 0.8 * plain)

    def test_patches_are_pasted_with_soft_edges_and_the_last_patch_frame_is_held(self):
        import numpy as np

        from engine.media import video
        base = self.clip("base.mp4", 17, size=(128, 96))
        patch = self.dir / "patch.mp4"
        with video.Writer(patch, 16, 32, 32, crf=10, preset="ultrafast") as writer:
            for _ in range(6):  # shorter than the base
                writer.write(np.full((32, 32, 3), 250, np.uint8))
        count = video.overlay_layers(base, [(str(patch), 40, 20, 32, 32)], self.dir / "out.mp4")
        self.assertEqual(count, 17)
        for index in (0, 16):  # the patch is still there after its own video ended
            frame = video.to_rgb(list(video.read_frames(self.dir / "out.mp4"))[index]).astype(int)
            self.assertGreater(frame[36, 56].mean(), 200)  # centre of the patch
            self.assertLess(abs(frame[80, 10].mean() - 8 * index), 4)  # far away: still the base
            self.assertLess(frame[20, 40].mean(), frame[36, 56].mean())  # the edge is blended

    def test_a_filter_chain_re_encodes_a_video_and_crops_it(self):
        from engine.media import video
        source = self.clip("a.mp4", 16, size=(64, 48))
        count = video.reencode(source, self.dir / "half.mp4", chain="fps=8")
        self.assertEqual((count, video.probe(self.dir / "half.mp4").fps), (8, 8.0))
        video.reencode(source, self.dir / "crop.mp4", chain="crop=32:24:0:0")
        small = video.probe(self.dir / "crop.mp4")
        self.assertEqual((small.width, small.height, small.fps), (32, 24, 16.0))  # size and rate of the result
        video.reencode(source, self.dir / "big.mp4", chain="crop=32:24:0:0,scale=96:72:flags=lanczos")
        self.assertEqual(video.probe(self.dir / "big.mp4").width, 96)

    def test_splice_puts_the_patch_in_place_of_a_stretch_of_frames(self):
        from engine.media import video
        full = self.clip("full.mp4", 12)
        patch = self.clip("patch.mp4", 6, first=20)  # 6 frames; only 4 fit in the gap
        count = video.splice(full, patch, 4, 8, self.dir / "out.mp4")
        self.assertEqual(count, 12)
        self.assertEqual(self.levels(self.dir / "out.mp4"), [0, 1, 2, 3, 20, 21, 22, 23, 8, 9, 10, 11])

    def test_a_voice_sample_is_trimmed_normalised_and_made_mono(self):
        import numpy as np

        from engine.media import audio
        rate = 22050
        t = np.arange(int(2.0 * rate)) / rate
        tone = 0.2 * np.sin(2 * np.pi * 220 * t)
        padded = np.concatenate([np.zeros(rate), tone, np.zeros(rate)])  # 1 s of silence on each side
        audio.write_wav(self.dir / "in.wav", np.stack([padded, padded]).astype(np.float32), rate)
        audio.prepare_voice_sample(self.dir / "in.wav", self.dir / "out.wav", 60, 12)
        samples, out_rate = audio.read(self.dir / "out.wav", rate=None, channels=None)
        self.assertEqual((samples.shape[0], out_rate), (1, 48000))
        self.assertAlmostEqual(samples.shape[1] / out_rate, 2.45, delta=0.15)  # the tone, 0.15 s before it, 0.3 s after
        audio.prepare_voice_sample(self.dir / "in.wav", self.dir / "short.wav", 60, 1.0)
        self.assertLessEqual(audio.read(self.dir / "short.wav", rate=None, channels=None)[0].shape[1], 48000)

    def test_a_stereo_voice_sample_is_downmixed_without_gaining_3_db(self):
        """The old ffmpeg run downmixed in 16-bit integers, with the matrix scaled to add up to 1: L = R keeps its level.
        In float the plain downmix adds 3 dB for such channels (measured against FFmpeg 9.0.2: 0.00 dB difference)."""
        import numpy as np

        from engine.media import audio
        rate = 48000
        wave = (0.2 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)).astype(np.float32)
        both = np.stack([wave, wave])
        mono, _ = audio.run_chain(both, rate, audio.VOICE_DOWNMIX)
        self.assertEqual(mono.shape[0], 1)
        np.testing.assert_allclose(mono[0, :rate], wave, atol=1e-4)  # the same level as one channel
        plain, _ = audio.run_chain(both, rate, "aresample=48000,aformat=channel_layouts=mono")
        self.assertGreater(float(np.abs(plain).max()), 1.3 * float(np.abs(wave).max()))  # what it used to do: +3 dB
        unequal, _ = audio.run_chain(np.stack([wave, 0 * wave]), rate, audio.VOICE_DOWNMIX)  # one silent channel: half
        np.testing.assert_allclose(unequal[0, :rate], wave / 2, atol=1e-4)

    def test_prepare_voice_sample_uses_that_downmix_and_a_stereo_file_comes_out_3_db_below_its_mono_twin(self):
        """FFmpeg measured the stereo recording 3 LU louder (two channels), turned it down, then downmixed without gain;
        with the float downmix of before the two files ended at the same level."""
        import numpy as np
        from unittest.mock import patch

        from engine.media import audio
        rate = 22050
        t = np.arange(int(2.0 * rate)) / rate
        padded = np.concatenate([np.zeros(rate), 0.2 * np.sin(2 * np.pi * 220 * t), np.zeros(rate)]).astype(np.float32)
        audio.write_wav(self.dir / "stereo.wav", np.stack([padded, padded]), rate)
        audio.write_wav(self.dir / "mono.wav", padded[None, :], rate)
        levels, chains, real = {}, [], audio.run_chain

        def spy(samples, sample_rate, chain, *args, **kwargs):
            chains.append(chain)
            return real(samples, sample_rate, chain, *args, **kwargs)

        with patch.object(audio, "run_chain", spy):
            for name in ("stereo", "mono"):
                audio.prepare_voice_sample(self.dir / f"{name}.wav", self.dir / f"{name}_out.wav", 60, 12)
                samples, _ = audio.read(self.dir / f"{name}_out.wav", rate=None, channels=None)
                levels[name] = 20 * np.log10(float(np.sqrt(np.mean(samples ** 2))))
        self.assertTrue(all(chain.split(",")[-1] == audio.VOICE_DOWNMIX for chain in chains), chains)
        self.assertAlmostEqual(levels["mono"] - levels["stereo"], 3.0, delta=0.4)  # measured: 3.01 dB, as with the ffmpeg program

    def test_reading_audio_can_stop_early_and_keep_the_files_own_format(self):
        from engine.media import audio
        path = self.tone("t.wav", 3.0)
        samples, rate = audio.read(path, seconds=1.0)
        self.assertEqual((samples.shape, rate), ((2, 48000), 48000))
        native, native_rate = audio.read(path, rate=None, channels=None)
        self.assertEqual((native.shape[0], native_rate), (2, 48000))

    def test_frame_at_gives_the_first_frame_from_that_moment(self):
        import numpy as np
        from PIL import Image

        from engine.media import video
        path = self.clip("a.mp4", 33)
        video.frame_at(path, 1.0, self.dir / "f.png")  # frame 16
        self.assertEqual(int(np.asarray(Image.open(self.dir / "f.png").convert("RGB")).mean() / 8 + 0.5), 16)
        video.frame_at(path, 99.0, self.dir / "end.png")  # past the end: the last frame
        self.assertEqual(int(np.asarray(Image.open(self.dir / "end.png").convert("RGB")).mean() / 8 + 0.5), 32)


    def tags(self, path):
        import av

        with av.open(str(path)) as container:
            context = container.streams.video[0].codec_context
            return context.colorspace, context.color_range

    def test_the_colour_description_survives_every_re_encode(self):
        """The ffmpeg program copied the tags of the source to the encoder; without them a tag-aware reader
        decodes a BT.709 picture with the BT.601 matrix and the colours drift."""
        from engine.media import video

        source, other = self.clip("a.mp4", 17, tagged=True), self.clip("b.mp4", 17, first=17, tagged=True)
        patch = self.clip("patch.mp4", 6, size=(32, 32), tagged=True)
        self.assertEqual(self.tags(source), (1, 1))
        video.convert_fps(source, self.dir / "fps.mp4", 30, 17 / 16)
        video.join_segments([source, other], self.dir / "join.mp4", 16)
        video.join_segments([source, other], self.dir / "fade.mp4", 16, overlaps=[0, 0.25], handoffs=[16 / 16 - 0.25])
        video.reencode(source, self.dir / "crop.mp4", chain="crop=32:24:0:0")
        video.splice(source, patch, 2, 6, self.dir / "splice.mp4")
        video.overlay_layers(source, [(str(patch), 8, 8, 32, 32)], self.dir / "overlay.mp4")
        video.combine_video_audio(source, self.tone("t.wav", 1.0), self.dir / "mix.mp4", compress=True)
        for name in ("fps", "join", "fade", "crop", "splice", "overlay", "mix"):
            self.assertEqual(self.tags(self.dir / f"{name}.mp4"), (1, 1), name)
        untagged = self.clip("plain.mp4", 9)  # an untagged source stays untagged: nothing is invented
        video.reencode(untagged, self.dir / "plain2.mp4", chain="fps=8")
        self.assertEqual(self.tags(self.dir / "plain2.mp4"), (2, 0))

    def test_patches_of_any_even_size_can_be_laid_over_a_video(self):
        import numpy as np

        from engine.media import video

        base = self.clip("base.mp4", 6, size=(128, 98))  # the product's face crops are even but not multiples of 4
        for width, height in ((66, 66), (64, 66), (66, 64), (30, 30)):
            patch = self.dir / f"patch{width}x{height}.mp4"
            with video.Writer(patch, 16, width, height, crf=10, preset="ultrafast") as writer:
                for _ in range(6):
                    writer.write(np.full((height, width, 3), 250, np.uint8))
            out = self.dir / f"out{width}x{height}.mp4"
            self.assertEqual(video.overlay_layers(base, [(str(patch), 20, 12, width, height)], out), 6)
            frame = video.to_rgb(next(iter(video.read_frames(out)))).astype(int)
            self.assertGreater(frame[12 + height // 2, 20 + width // 2].mean(), 200, (width, height))

    def test_sources_with_b_frames_join_convert_and_combine(self):
        """The product's encoder presets use B-frames; the fast test clips do not."""
        from engine.media import video

        first = self.clip("1.mp4", 33, preset="medium", crf=18)
        second = self.clip("2.mp4", 33, first=32, preset="medium", crf=18)
        self.assertEqual(video.join_segments([first, second], self.dir / "j.mp4", 16), 65)
        video.convert_fps(self.dir / "j.mp4", self.dir / "c.mp4", 30, 65 / 16)
        video.combine_video_audio(self.dir / "c.mp4", self.tone("t.wav", 2.0), self.dir / "out.mp4")
        info = video.probe(self.dir / "out.mp4")
        self.assertEqual((info.fps, info.has_audio), (30.0, True))
        self.assertAlmostEqual(info.duration, round(65 / 16 * 30) / 30, delta=0.1)


if __name__ == "__main__":
    unittest.main()
