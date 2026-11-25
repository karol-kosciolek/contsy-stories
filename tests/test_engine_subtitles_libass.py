# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The PIL subtitle renderer against what libass drew for the product's old command (FFmpeg 9.0.2 full build,
subtitles filter, FontName=Arial,FontSize=13,Outline=2,Alignment=2,MarginV=70) on gray frames: size, position and
line breaks. A size given as an em instead of a line height made the text 12 to 16 percent too big."""
import importlib.util
import unittest

HAVE = all(importlib.util.find_spec(name) is not None for name in ("av", "numpy", "PIL"))

SHORT = "Short"
LONG = "Hello there, this is a rather long subtitle line that has to be wrapped over two lines nicely"
SENTENCE = "Go to the store and buy two loaves of bread, a bottle of milk, six eggs and some cheese today"
# (frame size, text) -> the bounding box (left, top, right, bottom) of the white fill that libass drew
LIBASS = {
    ((640, 360), SHORT): (304, 259, 335, 268), ((640, 360), LONG): (46, 259, 600, 271),
    ((640, 360), SENTENCE): (19, 259, 620, 271),  # one line in libass; the PIL text is a little wider (1 percent)
    ((832, 480), SHORT): (394, 346, 437, 358), ((832, 480), LONG): (42, 346, 790, 362),
    ((832, 480), SENTENCE): (216, 324, 615, 362),  # libass broke this one into two even lines
    ((1280, 720), SHORT): (607, 518, 673, 538), ((1280, 720), LONG): (79, 518, 1202, 543),
    ((1280, 720), SENTENCE): (38, 518, 1242, 543),
}


STORY = ("When the children came home from school they found their mother in the garden with a basket of apples "
         "and she told them a story about the old house near the river")
POLISH = "Zażółć gęślą jaźń, to jest zdanie testowe, które musi zostać zawinięte w kilku wierszach, bo jest wystarczająco długie."
# (frame size, text) -> widths of the lines of white fill libass drew (long texts, portrait and landscape frames)
WRAPPED = {
    ((720, 1280), STORY): [553, 658, 586, 621, 621, 627], ((720, 1280), POLISH): [564, 485, 549, 524, 486],
    ((480, 854), STORY): [369, 440, 391, 414, 413, 419], ((480, 854), POLISH): [376, 323, 366, 349, 323],
    ((1080, 1920), STORY): [830, 988, 880, 933, 931, 942], ((1080, 1920), POLISH): [847, 727, 826, 787, 729],
    ((1280, 720), STORY): [1034, 1072], ((1280, 720), POLISH): [762, 736],
    ((720, 1280), SENTENCE): [629, 545, 462, 445],  # libass fills the early lines more: only the number of lines is compared
}


def arial_present():
    from PIL import ImageFont

    try:
        ImageFont.truetype("arial.ttf", 10)
        return True
    except OSError:
        return False


def white_box(text, size):
    import numpy as np
    from engine.media import subtitles

    patch, (left, top) = subtitles.render(text, size)
    pixels = np.asarray(patch)
    ys, xs = np.nonzero((pixels[..., 0] > 235) & (pixels[..., 3] > 200))
    return left + int(xs.min()), top + int(ys.min()), left + int(xs.max()), top + int(ys.max())


@unittest.skipUnless(HAVE and arial_present(), "needs PyAV, numpy, PIL and the Arial font")
class LibassFidelityTests(unittest.TestCase):
    def test_size_and_position_follow_libass(self):
        for (size, text), reference in LIBASS.items():
            with self.subTest(size=size, text=text[:12]):
                ours = white_box(text, size)
                slack_x = max(7, round(0.008 * size[0]))  # glyph advances are hinted a little differently
                self.assertLessEqual(abs(ours[0] - reference[0]), slack_x, f"left {ours} vs libass {reference}")
                self.assertLessEqual(abs(ours[2] - reference[2]), slack_x, f"right {ours} vs libass {reference}")
                self.assertLessEqual(abs(ours[3] - reference[3]), 2, f"bottom {ours} vs libass {reference}")
                self.assertLessEqual(abs(ours[1] - reference[1]), 3, f"top {ours} vs libass {reference}")

    def test_long_texts_break_into_as_many_lines_as_libass_with_nearly_the_same_widths(self):
        """Line breaks are decided on unhinted widths like libass does; hinted ones made lines 0.5 to 1 percent wider and moved
        the breaks (a line that libass just fits overflowed and cost an extra line)."""
        import numpy as np
        from engine.media import subtitles

        for (size, text), reference in WRAPPED.items():
            patch, _ = subtitles.render(text, size)
            pixels = np.asarray(patch)
            mask = (pixels[..., 0] > 235) & (pixels[..., 3] > 200)
            rows = np.nonzero(mask.any(axis=1))[0]
            runs, start, last = [], rows[0], rows[0]
            for row in rows[1:]:
                if row > last + 1:
                    runs.append((start, last))
                    start = row
                last = row
            runs.append((start, last))
            ours = []
            for first, final in runs:
                columns = np.nonzero(mask[first:final + 1].any(axis=0))[0]
                if columns.max() - columns.min() + 1 >= 12:  # a descender tip under a line is not a line
                    ours.append(int(columns.max() - columns.min() + 1))
            with self.subTest(size=size, text=text[:14]):
                self.assertEqual(len(ours), len(reference), f"{ours} vs libass {reference}")
                if text != SENTENCE:  # its break points differ on purpose (libass packs the early lines)
                    for mine, theirs in zip(ours, reference):
                        self.assertLessEqual(abs(mine - theirs), max(10, 0.03 * theirs), f"{ours} vs libass {reference}")

    def test_a_text_that_needs_two_lines_is_cut_in_the_middle(self):
        from engine.media import subtitles
        from PIL import Image, ImageDraw

        font = subtitles._font("arial.ttf", 13 * 480 / 288)
        draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        lines = subtitles.wrap(SENTENCE, font, draw, 832 - 2 * 22)
        self.assertEqual(len(lines), 2)
        first, second = (draw.textlength(line, font=font) for line in lines)
        self.assertLess(abs(first - second), 0.1 * max(first, second))  # even lines (libass drew 400 and 400), no stub
        self.assertEqual(subtitles.wrap("one two", font, draw, 10_000), ["one two"])
        self.assertEqual(subtitles.wrap("", font, draw, 100), [])

    def test_a_text_that_exactly_fits_stays_on_one_line(self):
        from engine.media import subtitles
        from PIL import Image, ImageDraw

        font = subtitles._font("arial.ttf", 20)
        draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        width = int(draw.textlength(LONG, font=font)) + 1  # exactly fits on one line
        self.assertEqual(len(subtitles.wrap(LONG, font, draw, width)), 1)  # the outline is drawn outside the box


if __name__ == "__main__":
    unittest.main()
