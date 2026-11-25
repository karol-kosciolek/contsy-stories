# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Burned-in subtitles (engine/media/subtitles.burn, app/ffmpeg_utils.burn_subtitles): the text is drawn only
while its cue is due, the rest of the picture and its colours stay, the sound is copied, long lines are paged,
overlapping lines are stacked, too wide words are cut, missing glyphs never show as boxes, and the layout fits
every frame shape the Timeline exports at a reading length."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAVE = all(importlib.util.find_spec(name) is not None for name in ("av", "numpy", "PIL"))
LONG = ("I told you we would be late again, and now the whole street is closed because of the parade that "
        "nobody remembered, so we walk from here and we do not complain about it")


@unittest.skipUnless(HAVE, "PyAV, NumPy or Pillow is not installed")
class BurnTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def clip(self, name, frames, size=(320, 180), fps=16, tagged=True):
        import numpy as np

        from engine.media import video

        path = self.dir / name
        tags = {"colorspace": 1, "color_range": 1} if tagged else {}
        with video.Writer(path, fps, size[0], size[1], crf=10, preset="ultrafast", **tags) as writer:
            for i in range(frames):
                writer.write(np.full((size[1], size[0], 3), min(8 * i, 255), np.uint8))
        return path

    def frames(self, path):
        from engine.media import video

        return [video.to_rgb(frame).astype(int) for frame in video.read_frames(path)]

    def test_text_is_drawn_only_while_due_and_the_sound_and_tag_stay(self):
        import av
        import numpy as np

        from app import ffmpeg_utils
        from engine.media import audio, video

        base = self.clip("base.mp4", 32)
        rate = 48000
        tone = np.stack([0.3 * np.sin(np.arange(2 * rate) * 0.05)] * 2).astype(np.float32)
        audio.write_wav(self.dir / "tone.wav", tone, rate)
        video.combine_video_audio(base, self.dir / "tone.wav", self.dir / "sound.mp4", tag_ai=False)
        ok, message = ffmpeg_utils.burn_subtitles(self.dir / "sound.mp4", [(0.5, 1.0, "Hello there")],
                                                  self.dir / "out.mp4")
        self.assertTrue(ok, message)
        frames = self.frames(self.dir / "out.mp4")
        self.assertEqual(len(frames), 32)
        band = lambda frame: frame[100:150].max()  # noqa: E731 - the subtitle sits in this band
        self.assertGreater(band(frames[12]), 200)  # 0.75 s: white text
        self.assertLess(band(frames[2]), 30)  # 0.125 s: only the flat base
        self.assertLess(band(frames[24]), 220)  # 1.5 s: gone again
        self.assertTrue(video.probe(self.dir / "out.mp4").has_audio)
        with av.open(str(self.dir / "out.mp4")) as container:
            self.assertEqual(container.metadata.get("comment"), video.AI_COMMENT)
            context = container.streams.video[0].codec_context
            self.assertEqual((context.colorspace, context.color_range), (1, 1))

    def test_the_rest_of_the_picture_keeps_its_colours(self):
        from engine.media import subtitles

        source = self.clip("a.mp4", 16)
        subtitles.burn(source, [(0.5, 2.0, "Hello there")], self.dir / "burn.mp4")
        before, after = self.frames(source), self.frames(self.dir / "burn.mp4")
        for index in (2, 12):  # before the cue and while it is shown: the upper half is as it was
            self.assertLess(abs(after[index][:90] - before[index][:90]).mean(), 1.0, index)
        self.assertGreater(abs(after[12][100:150] - before[12][100:150]).max(), 100)  # the text itself

    def test_cues_follow_the_clock_of_a_clip_that_does_not_start_at_zero(self):
        import numpy as np

        from engine.media import subtitles, video
        from tests.test_engine_media_offsets import make_clip

        late = self.dir / "late.mp4"
        make_clip(late, 25, 25, 3.0)
        out = self.dir / "burned.mp4"
        subtitles.burn(late, [(0.0, 0.2, "Hello")], out)  # five frames at 25 fps
        changed = [i for i, (a, b) in enumerate(zip(video.read_frames(late), video.read_frames(out)))
                   if np.abs(video.to_rgb(a).astype(int) - video.to_rgb(b).astype(int)).max() > 60]
        self.assertEqual(changed, [0, 1, 2, 3, 4])

    def test_a_missing_film_is_reported_not_raised(self):
        from app import ffmpeg_utils

        ok, message = ffmpeg_utils.burn_subtitles(self.dir / "none.mp4", [(0.0, 1.0, "Hi")], self.dir / "out.mp4")
        self.assertFalse(ok)
        self.assertTrue(message)


@unittest.skipUnless(HAVE, "PyAV, NumPy or Pillow is not installed")
class LayoutTests(unittest.TestCase):
    def sizes(self):
        """The frame of every Timeline aspect (the export keeps Wan's 480p frame), plus a 4:3 one."""
        from app import video_gen

        frames = {name: video_gen.sizes(ratio)[0] for name, ratio in video_gen.ASPECTS.items()}
        frames["Landscape 4:3"] = video_gen.frame_size(4 / 3, video_gen.VIDEO_PIXELS)
        return frames

    def test_every_export_shape_fits_the_text_in_the_bottom_safe_area(self):
        from engine.media import subtitles

        style = subtitles.Style()
        for name, (width, height) in self.sizes().items():
            for text in ("Yes.", LONG):
                with self.subTest(frame=name, text=text[:10]):
                    for _, _, page in subtitles.paginate([(0.0, 6.0, text)], (width, height)):
                        patch, (left, top) = subtitles.render(page, (width, height))
                        self.assertGreaterEqual(left, 0)
                        self.assertLessEqual(left + patch.width, width)
                        bottom = top + patch.height
                        safe = height - round(style.margin_bottom * height / subtitles.SCRIPT_HEIGHT)
                        self.assertLessEqual(bottom, safe + round(style.outline * height / subtitles.SCRIPT_HEIGHT) + 1)
                        self.assertGreater(top, height // 2)  # never over the middle of the picture
                        lines = subtitles._lines(page, (width, height), style)
                        self.assertLessEqual(len(lines), subtitles.MAX_LINES)

    def test_the_text_size_is_a_share_of_the_frame_height(self):
        from engine.media import subtitles

        heights = {}
        for name, size in self.sizes().items():
            patch, _ = subtitles.render("Hello", size)
            heights[name] = patch.height / size[1]
        self.assertLess(max(heights.values()) - min(heights.values()), 0.01, heights)

    def test_long_lines_are_paged_over_the_time_of_their_cue(self):
        from engine.media import subtitles

        portrait = self.sizes()["Portrait 9:16"]
        pages = subtitles.paginate([(2.0, 8.0, LONG), (0.5, 1.5, "  "), (9.0, 9.0, "no time")], portrait)
        self.assertGreater(len(pages), 1)
        self.assertEqual((pages[0][0], pages[-1][1]), (2.0, 8.0))
        for (_, end, _), (start, _, _) in zip(pages, pages[1:]):
            self.assertAlmostEqual(end, start)  # one after another, no gap, no overlap
        self.assertEqual(" ".join(page for _, _, page in pages), " ".join(LONG.split()))

    def test_of_overlapping_lines_the_newest_is_shown(self):
        from engine.media import subtitles

        cues = [(0.0, 3.0, "first"), (2.0, 4.0, "second")]
        self.assertEqual([subtitles._due(cues, t) for t in (1.0, 2.5, 3.5, 4.5)], [0, 1, 1, None])

    def test_lines_that_overlap_are_stacked_with_the_newest_at_the_bottom(self):
        from engine.media import subtitles

        wide = self.sizes()["Landscape 16:9"]
        shown = subtitles.screens([(0.0, 3.0, "first"), (2.0, 4.0, "second"), (4.0, 5.0, "third")], wide)
        self.assertEqual(shown, [(0.0, 2.0, "first"), (2.0, 3.0, "first\nsecond"), (3.0, 4.0, "second"),
                                 (4.0, 5.0, "third")])
        self.assertEqual(subtitles._lines("first\nsecond", wide, subtitles.Style()), ["first", "second"])
        # a later page of an earlier long line never hides a newer speaker, and one text is shown at a time
        portrait = self.sizes()["Portrait 9:16"]
        shown = subtitles.screens([(0.0, 6.0, LONG), (1.0, 2.0, "Wait!")], portrait)
        self.assertTrue(all(text.endswith("\nWait!") for start, end, text in shown if start < 2.0 and end > 1.0),
                        shown)
        self.assertTrue(all(later[0] >= earlier[1] for earlier, later in zip(shown, shown[1:])))

    def test_the_timeline_keeps_lines_to_a_reading_length_in_wide_frames(self):
        from engine.media import subtitles

        text = "I told you we would be late again, and now the whole street is closed because of the parade."
        sizes = self.sizes()
        wide = sizes["Landscape 16:9"]
        self.assertEqual(len(subtitles._lines(text, wide, subtitles.Style())), 1)  # libass: across the whole frame
        lines = subtitles._lines(text, wide, subtitles.READABLE)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(line) <= 45 for line in lines), lines)
        patch, (left, _) = subtitles.render(text, wide, subtitles.READABLE)
        self.assertLessEqual(abs(left + patch.width / 2 - wide[0] / 2), 1)  # still centred
        portrait = sizes["Portrait 9:16"]  # narrower than the reading length: wrapped as before
        self.assertEqual(subtitles._lines(text, portrait, subtitles.READABLE),
                         subtitles._lines(text, portrait, subtitles.Style()))

    def test_a_word_wider_than_the_frame_is_cut_to_fit(self):
        from PIL import Image, ImageDraw

        from engine.media import subtitles

        word = "Supercalifragilisticexpialidociousandmoreandmoreandmorewordsrunningonandon"
        draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        for name, size in self.sizes().items():
            with self.subTest(frame=name):
                lines = subtitles._lines(word, size, subtitles.READABLE)
                self.assertEqual("".join(lines), word)
                patch, (left, _) = subtitles.render(word, size, subtitles.READABLE)
                self.assertGreaterEqual(left, 0)
                self.assertLessEqual(left + patch.width, size[0])
                font = subtitles._font("arial.ttf", 13 * size[1] / subtitles.SCRIPT_HEIGHT)
                self.assertTrue(all(draw.textlength(line, font=font) <= patch.width for line in lines))

    def test_characters_the_font_lacks_use_a_font_that_has_them_or_are_left_out(self):
        from engine.media import subtitles

        style = subtitles.Style()
        self.assertEqual(subtitles._drawable("Hello there", style), (style.font, "Hello there"))
        self.assertEqual(subtitles._drawable("\U0010FFFF ok", style), (style.font, "ok"))  # no font has it
        self.assertEqual(subtitles.paginate([(0.0, 1.0, "\U0010FFFF"), (1.0, 2.0, "Hi")], (848, 480)),
                         [(1.0, 2.0, "Hi")])  # nothing drawable: no empty subtitle
        name, text = subtitles._drawable("你好，朋友", style)
        if name == style.font:
            self.skipTest("no font of this system draws Chinese")
        self.assertEqual(text, "你好，朋友")
        self.assertTrue(all(subtitles._has_glyph(name, char) for char in text))
        patch, _ = subtitles.render(text, (848, 480))
        self.assertGreater(patch.width, 0)


if __name__ == "__main__":
    unittest.main()
