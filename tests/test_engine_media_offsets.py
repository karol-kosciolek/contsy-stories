# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Clips whose first timestamp is not 0 (an edit list in mp4, a start time in mkv) behave like clips that start at 0,
as with the ffmpeg program, which takes the start time off every input; and a re-encode keeps the stream's own frame
rate unless the filter chain changes it (the first two timestamps of a 24 fps clip in a 1/1000 time base are 42 ms apart)."""
import importlib.util
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

HAVE = importlib.util.find_spec("av") is not None and importlib.util.find_spec("numpy") is not None


def make_clip(path, frames, fps, start_seconds=0.0, size=(64, 48), base_level=0):
    """Frame i is a flat grey of level base_level + 8 * i; the first timestamp is `start_seconds` (not 0 for a late clip)."""
    import av
    import numpy as np

    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "10", "bf": "0"}
        stream.codec_context.time_base = Fraction(1, fps)
        offset = int(start_seconds * fps)
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((size[1], size[0], 3), min(base_level + 8 * i, 255), np.uint8), format="rgb24")
            frame.pts, frame.time_base = offset + i, Fraction(1, fps)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


@unittest.skipUnless(HAVE, "PyAV is not installed")
class StartOffsetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def late(self, ext, frames=25, fps=25, start=3.0):
        import av

        path = self.dir / f"clip_{start}.{ext}"
        make_clip(path, frames, fps, start)
        with av.open(str(path)) as container:  # the premise of every test here
            self.assertAlmostEqual((container.start_time or 0) / av.time_base, start, delta=0.05)
        return path

    def count(self, path):
        from engine.media import video

        return sum(1 for _ in video.read_frames(path))

    def test_late_shots_are_joined_like_shots_that_start_at_zero(self):
        from engine.media import video

        for ext in ("mp4", "mkv"):
            with self.subTest(ext=ext):
                late, early = self.late(ext), self.dir / f"early.{ext}"
                make_clip(early, 25, 25, 0.0)
                out_late, out_early = self.dir / f"j_late.{ext}", self.dir / f"j_early.{ext}"
                video.join_segments([late, late], out_late, 25)
                video.join_segments([early, early], out_early, 25)
                self.assertEqual(self.count(out_early), 49)  # 25 + 24: the repeated frame of the second shot is dropped
                self.assertEqual(self.count(out_late), 49)

    def test_frames_are_picked_by_time_from_the_start_of_the_file(self):
        import numpy as np
        from engine.media import video
        from PIL import Image

        late = self.late("mp4")
        for seconds, index in ((0.5, 13), (0.0, 0), (99.0, 24)):  # the last frame when the clip ends sooner
            video.frame_at(late, seconds, self.dir / "f.png")
            level = float(np.asarray(Image.open(self.dir / "f.png")).mean())
            with self.subTest(seconds=seconds):
                self.assertAlmostEqual(level, min(8 * index, 255), delta=4)
        video.last_frame(late, self.dir / "last.png")
        self.assertAlmostEqual(float(np.asarray(Image.open(self.dir / "last.png")).mean()), 8 * 24, delta=4)

    def test_reencode_and_convert_fps_see_a_late_clip_from_its_first_frame(self):
        """With the raw timestamps a trim, fps or tpad in the chain works on 3 s, 3.04 s ... and drops the whole clip."""
        import numpy as np
        from engine.media import video

        late = self.late("mp4")
        out = self.dir / "re.mp4"
        self.assertEqual(video.reencode(late, out, chain="crop=32:32:0:0"), 25)
        self.assertEqual(self.count(out), 25)
        # a trim on the clock of the clip: 0.4 s are 10 frames at 25 fps; with raw timestamps (3.0 s on) it would keep none
        self.assertEqual(video.reencode(late, out, chain="trim=end=0.4,setpts=PTS-STARTPTS"), 10)
        fps_out = self.dir / "fps.mp4"
        self.assertEqual(video.convert_fps(late, fps_out, 30), 30)  # one second at 30 fps
        levels = [float(np.asarray(video.to_rgb(f)).mean()) for f in video.read_frames(fps_out)]
        self.assertEqual(len(levels), 30)
        self.assertAlmostEqual(levels[0], 0, delta=4)  # the first frame of the clip
        self.assertAlmostEqual(levels[-1], 8 * 24, delta=6)  # and its last, held to fill the second

    def test_a_patch_is_laid_over_a_late_base_from_the_patchs_first_frame(self):
        """The base starts at 3 s, the patch at 0: both are clips that start at 0, so patch frame 0 goes over base frame 0."""
        import numpy as np
        from engine.media import video

        base, patch = self.late("mp4"), self.dir / "patch.mp4"
        make_clip(patch, 9, 25, 0.0, size=(32, 32), base_level=120)  # levels 120, 128 ... 184
        out = self.dir / "over.mp4"
        self.assertEqual(video.overlay_layers(base, [(str(patch), 16, 8, 32, 32)], out), 25)
        first = next(iter(video.read_frames(out)))
        center = float(np.asarray(video.to_rgb(first))[22:26, 30:34].mean())  # the middle of the 32 x 32 patch box
        self.assertAlmostEqual(center, 110, delta=15)  # patch frame 0 (level 120, softened by the mask), not about 170:
        # that is the last patch frame (level 184), which a base whose clock starts at 3 s would show from its first frame

    def test_a_late_patch_is_laid_over_an_early_base_from_the_first_frame(self):
        """The other way round: with the raw timestamps the patch (3 s on) would never be due while the base plays."""
        import numpy as np
        from engine.media import video

        base, patch = self.dir / "early_base.mp4", self.dir / "late_patch.mp4"
        make_clip(base, 25, 25, 0.0)
        make_clip(patch, 9, 25, 3.0, size=(32, 32), base_level=120)
        out = self.dir / "over2.mp4"
        self.assertEqual(video.overlay_layers(base, [(str(patch), 16, 8, 32, 32)], out), 25)
        first = next(iter(video.read_frames(out)))
        center = float(np.asarray(video.to_rgb(first))[22:26, 30:34].mean())
        self.assertAlmostEqual(center, 110, delta=15)  # the patch is there; without it the centre would be the base (level 0)



@unittest.skipUnless(HAVE, "PyAV is not installed")
class FrameRateTests(unittest.TestCase):
    def test_a_re_encode_keeps_the_rate_of_the_stream_unless_the_chain_changes_it(self):
        import av
        from engine.media import video

        with tempfile.TemporaryDirectory() as folder:
            clip, out = Path(folder) / "c.mkv", Path(folder) / "o.mp4"
            make_clip(clip, 48, 24)
            with av.open(str(clip)) as container:
                first = [f.pts for f in container.decode(container.streams.video[0])][:3]
            self.assertEqual(first[1] - first[0], 42)  # 1/1000 time base: 41.67 ms became 42 ms
            video.reencode(clip, out, chain="crop=32:32:0:0")
            self.assertEqual(video.probe(out).fps, 24.0)  # not 1000 / 42 = 23.8095
            video.reencode(clip, out, chain="fps=12")
            self.assertEqual(video.probe(out).fps, 12.0)  # a chain that changes the rate: measured from its output

    def test_which_filters_change_the_rate(self):
        from engine.media.video import _changes_rate

        self.assertFalse(_changes_rate(None))
        self.assertFalse(_changes_rate("crop=300:160:0:0,scale='min(1280,iw)':-2,format=yuv420p"))
        for chain in ("fps=25", "scale=64:48,fps=30", "framestep=2", "select='not(mod(n,2))'", "setpts=0.5*PTS"):
            self.assertTrue(_changes_rate(chain), chain)


if __name__ == "__main__":
    unittest.main()
