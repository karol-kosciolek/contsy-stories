# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import av
import numpy as np

from app import ffmpeg_utils, video_gen
from engine.media.audio import write_wav
from engine.media.video import Writer


def _profile(path):
    with av.open(str(path)) as source:
        return source.streams.video[0].codec_context.profile


def _video(path, fps, frames):
    with Writer(path, fps, 128, 72, crf=0) as out:
        for i in range(frames):
            frame = np.full((72, 128, 3), i * 5 % 256, dtype=np.uint8)
            out.write(frame)


def _tone(path):
    rate = 44100
    samples = np.sin(np.arange(rate) * (2 * np.pi * 440 / rate)).astype(np.float32) * 0.1
    write_wav(path, samples[None, :], rate)


class LipSyncInputTests(unittest.TestCase):
    def test_lip_sync_takes_the_25_fps_picture_as_it_is_and_refuses_another_rate(self):
        """The one lip-sync entry point receives 25 FPS frames: it no longer resamples a 30 FPS clip
        (which dropped frames RIFE had made), and a picture at another rate is refused."""
        from app import lipsync

        with tempfile.TemporaryDirectory() as folder:
            v25, v30, wav, out = (Path(folder) / name for name in ("v25.mp4", "v30.mp4", "voice.wav", "out.mp4"))
            _video(v25, 25, 50)
            _video(v30, 30, 60)
            _tone(wav)
            with patch.object(lipsync, "_dialogue_clip", return_value=(True, str(out))) as clip, \
                    patch.object(ffmpeg_utils, "transform") as transform:
                ok, message, _notes = lipsync.run_dialogue(str(v25), str(wav), [(0.0, 2.0)], ["female#0"], str(out))
                self.assertTrue(ok, message)
                self.assertEqual(clip.call_args.args[0], v25)  # the picture itself
                self.assertEqual(clip.call_args.args[5], 2.0)  # 50 frames at 25 FPS
                transform.assert_not_called()
                ok, message, _notes = lipsync.run_dialogue(str(v30), str(wav), [(0.0, 2.0)], ["female#0"], str(out))
                self.assertFalse(ok)
                self.assertIn("25 FPS", message)
                clip.assert_called_once()

    def test_lipsync_conversion_uses_original_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'source.mp4', Path(folder) / 'out.mp4'
            client = Mock()
            with patch.object(ffmpeg_utils, 'probe_duration', return_value=2.0), \
                 patch.object(video_gen.engine_runtime, 'interpolate_video',
                              side_effect=lambda engine, src, dst, *args: dst.write_bytes(b'test')) as retime, \
                 patch.object(ffmpeg_utils, 'convert_fps', return_value=(True, str(output))) as convert:
                self.assertTrue(video_gen._lipsync_to_output_fps(client, source, output, 2.0)[0])
                self.assertEqual(retime.call_args.args[:5], (client, source, output.with_name("out_rife.mp4"), 25, 30))
                self.assertEqual(convert.call_args.args[2], 30)
                self.assertEqual(convert.call_args.kwargs['duration'], 2.0)
            self.assertFalse(output.with_name('out_rife.mp4').exists())

    def test_the_25_fps_parts_join_to_exactly_their_seconds(self):
        """A talking clip's parts (RIFE 16 -> 25: 25 s + 1 frames, or 25 s + 8 with the 4-frame overlap) join to
        exactly 25 frames a second, the count lip sync and the final 25 -> 30 step rely on."""
        with tempfile.TemporaryDirectory() as folder:
            first, second, joined, faded = (Path(folder) / name for name in ("1.mp4", "2.mp4", "j.mp4", "f.mp4"))
            _video(first, 25, 4 * 25 + 1)
            _video(second, 25, 3 * 25 + 8)
            ok, error = ffmpeg_utils.join_segments([str(first), str(first)], joined, 25, [0, 0], duration=8.0)
            self.assertTrue(ok, error)
            ok, error = ffmpeg_utils.join_segments([str(first), str(second)], faded, 25, [0, 0.25], duration=7.0,
                                                   handoffs=[4.0 - 0.25])
            self.assertTrue(ok, error)
            for path, seconds in ((joined, 8), (faded, 7)):
                self.assertEqual(ffmpeg_utils.video_fps(path), 25)
                with av.open(str(path)) as result:
                    self.assertEqual(sum(1 for _ in result.decode(video=0)), seconds * 25)


class VideoTimingTests(unittest.TestCase):
    def test_join_removes_shared_endpoint_and_keeps_requested_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            source, joined = Path(folder) / 'part.mp4', Path(folder) / 'joined.mp4'
            _video(source, 30, 31)
            ok, error = ffmpeg_utils.join_segments([str(source), str(source)], joined, 30,
                                                   [0, 0], duration=2.0)
            self.assertTrue(ok, error)
            self.assertEqual(ffmpeg_utils.video_fps(joined), 30)
            self.assertAlmostEqual(ffmpeg_utils.probe_duration(joined), 2.0, places=2)

    def test_conversion_and_audio_mux_keep_30_frames_and_one_second(self):
        with tempfile.TemporaryDirectory() as folder:
            source, converted, audio, final = [Path(folder) / name for name in
                                             ('source.mp4', '30fps.mp4', 'audio.wav', 'final.mp4')]
            _video(source, 50, 50)
            ok, error = ffmpeg_utils.convert_fps(source, converted, 30, duration=1.0)
            self.assertTrue(ok, error)
            _tone(audio)
            ok, error = ffmpeg_utils.combine_video_audio(converted, audio, final)
            self.assertTrue(ok, error)
            self.assertEqual(ffmpeg_utils.video_fps(final), 30)
            self.assertAlmostEqual(ffmpeg_utils.probe_duration(final), 1.0, places=2)
            with av.open(str(final)) as result:
                self.assertEqual(result.streams.video[0].frames, 30)
                self.assertEqual(len(result.streams.audio), 1)

    def test_compress_turns_the_lossless_intermediate_into_plain_h264(self):
        with tempfile.TemporaryDirectory() as folder:
            source, audio, final, silent = [Path(folder) / name for name in
                                            ('lossless.mp4', 'audio.wav', 'final.mp4', 'silent.mp4')]
            _video(source, 30, 30)
            _tone(audio)
            self.assertEqual(_profile(source), 'High 4:4:4 Predictive')  # jak plik z convert_fps
            for out, track in ((final, audio), (silent, None)):
                ok, error = ffmpeg_utils.combine_video_audio(source, track, out, compress=True)
                self.assertTrue(ok, error)
                self.assertEqual(_profile(out), 'High')
                self.assertEqual(ffmpeg_utils.video_fps(out), 30)
                self.assertAlmostEqual(ffmpeg_utils.probe_duration(out), 1.0, places=2)


if __name__ == '__main__':
    unittest.main()
