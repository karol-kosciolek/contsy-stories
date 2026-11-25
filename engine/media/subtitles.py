# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Burned-in subtitles. The FFmpeg inside PyAV has no libass or freetype, so the text is drawn with PIL (`render`)
and laid into the frames in the video's own colour planes (`burn`: `to_yuv` and `_chroma`), so the rest of the
picture keeps its exact colours.

The look is the one the product always had: white Arial, black outline, centred near the bottom, above
the button bar of the player. Sizes are given in the units libass uses for subtitles without a script:
a frame is 384 x 288 units, so a size scales with the real frame. As in libass, a size is the height of a
line of text (ascent plus descent of the font), not the em that PIL calls its size: Arial's line is about
1.12 em, so taking the number as an em would draw the text about 12 percent too big. Because every size is a
share of the frame height, the text has the same proportions in 16:9, 9:16, 1:1 and 3:4 frames; a narrow frame
only wraps it into more lines. A style may also cap a line at a reading length (`Style.max_line_chars`,
`READABLE`), so a wide frame does not stretch one line across the whole picture.

A cue too long for MAX_LINES lines is shown as consecutive pages (`paginate`), each for its share of the cue's
time by length, so no line is cut off at the bottom of the frame. Lines that overlap in time are stacked
(`screens`): the two newest at once, the newer one at the bottom. A word wider than a line is cut where it
must be. Characters the font lacks (emoji, Chinese, Japanese, Korean in Arial) are drawn with a Windows font
that has them (FALLBACK_FONTS), or left out, never drawn as empty boxes.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import video
from .errors import MediaError

SCRIPT_WIDTH, SCRIPT_HEIGHT = 384, 288
LAYOUT_SCALE = 8  # line breaks are decided on text measured at this many times the real size
MAX_LINES = 2  # lines of one page; more would cover the faces of a close-up
MAX_STACKED = 2  # cues on screen at once when lines overlap
Cue = Tuple[float, float, str]  # start s, end s, text
# Fonts of Windows for the characters the style's font lacks: symbols and emoji in one colour, Chinese, Japanese,
# Korean. A text is drawn in one font: the style's when it has every character, else the one that has the most.
FALLBACK_FONTS = ("seguisym.ttf", "msyh.ttc", "YuGothM.ttc", "malgun.ttf")
_NOT_A_CHARACTER = "\U0010FFFF"  # every font draws its "missing glyph" box for it
# Text of average English letter widths: a reading length in characters becomes pixels with it
_SAMPLE = "the quick brown fox jumps over the lazy dog"


@dataclass(frozen=True)
class Style:
    font: str = "arial.ttf"
    size: float = 13.0
    outline: float = 2.0
    margin_bottom: float = 70.0
    margin_side: float = 10.0
    color: Tuple[int, int, int] = (255, 255, 255)
    outline_color: Tuple[int, int, int] = (0, 0, 0)
    max_line_chars: int = 0  # the longest line in average characters (0: the frame between its margins, as libass)


# The Timeline export: about 42 characters a line, the usual reading length of subtitles, so a 16:9 line is not a
# long thin strip across the frame (in 9:16 the frame is narrower than that anyway).
READABLE = Style(max_line_chars=42)


def _make_font(name: str, size: float):
    try:
        return ImageFont.truetype(name, size)
    except OSError:  # no such font on this system: Pillow's own scalable one
        return ImageFont.load_default(size=size)


def _font(name: str, line_pixels: float):
    """The font whose line (ascent + descent) is `line_pixels` high, the way libass sizes subtitle fonts."""
    ascent, descent = _make_font(name, 1000).getmetrics()  # at a big size the metrics are exact to a tenth of a percent
    return _make_font(name, max(line_pixels * 1000 / (ascent + descent), 1.0))


@functools.lru_cache(maxsize=None)
def _probe_font(name: str):
    """The font at a size where glyphs are told apart, or None when this system does not have it (a fallback)."""
    try:
        return ImageFont.truetype(name, 48)
    except OSError:
        return None if name in FALLBACK_FONTS else ImageFont.load_default(size=48)


@functools.lru_cache(maxsize=8192)
def _has_glyph(name: str, char: str) -> bool:
    """Whether the font draws the character itself rather than its missing-glyph box."""
    font = _probe_font(name)
    if font is None:
        return False
    if char.isspace():
        return True
    drawn, missing = font.getmask(char), font.getmask(_NOT_A_CHARACTER)
    return (drawn.size, bytes(drawn)) != (missing.size, bytes(missing))


def _drawable(text: str, style: Style) -> Tuple[str, str]:
    """(font, text) to draw `text` with: the style's font when it has every character, else the fallback font that
    has the most of them; the characters it lacks are left out (and the spaces they leave are closed)."""
    chars = {char for char in text if not char.isspace()}
    if all(_has_glyph(style.font, char) for char in chars):
        return style.font, text
    name = max((style.font, *FALLBACK_FONTS), key=lambda font: sum(_has_glyph(font, char) for char in chars))
    kept = "".join(char for char in text if _has_glyph(name, char))
    return name, "\n".join(" ".join(part.split()) for part in kept.split("\n"))


def _greedy(words: List[str], font, draw: ImageDraw.ImageDraw, width: float) -> List[str]:
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if current and draw.textlength(trial, font=font) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    return lines + [current] if current else lines


def _pieces(word: str, font, draw: ImageDraw.ImageDraw, width: float) -> List[str]:
    """A word wider than a line cut into pieces that fit (a web address, a long run of letters); others whole.
    libass would let such a word run off both sides of the frame."""
    if draw.textlength(word, font=font) <= width:
        return [word]
    pieces, current = [], ""
    for char in word:
        if current and draw.textlength(current + char, font=font) > width:
            pieces.append(current)
            current = char
        else:
            current += char
    return pieces + [current]


def wrap(text: str, font, draw: ImageDraw.ImageDraw, width: int) -> List[str]:
    """Word wrap to `width` pixels, then evened out like libass's default wrapping (WrapStyle 0): a text that
    needs two lines is cut in the middle instead of leaving a stub on the second. The outline is drawn outside
    the line box and does not count, as in libass. A word wider than `width` is cut into pieces that fit."""
    words = [piece for word in text.split() for piece in _pieces(word, font, draw, width)]
    lines = _greedy(words, font, draw, width)
    if len(lines) < 2:
        return lines
    low = int(max(draw.textlength(word, font=font) for word in words))  # no line can be narrower than a word
    high = int(width)
    while low < high:  # the narrowest limit that still fits in the same number of lines
        middle = (low + high) // 2
        if len(_greedy(words, font, draw, middle)) <= len(lines):
            high = middle
        else:
            low = middle + 1
    return _greedy(words, font, draw, high)


def _width(size: Tuple[int, int], style: Style, font_name: str) -> int:
    """The widest line in pixels: the frame between its side margins, and with `max_line_chars` at most that many
    average characters of the font at the style's size."""
    width, height = size
    available = width - 2 * round(style.margin_side * width / SCRIPT_WIDTH)
    if style.max_line_chars > 0:
        font = _font(font_name, style.size * height / SCRIPT_HEIGHT * LAYOUT_SCALE)
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        advance = probe.textlength(_SAMPLE, font=font) / len(_SAMPLE) / LAYOUT_SCALE
        available = min(available, round(style.max_line_chars * advance))
    return available


def _lines(text: str, size: Tuple[int, int], style: Style) -> List[str]:
    """The lines `render` breaks `text` into in a frame of `size` (a line break in `text` starts a new line)."""
    name, text = _drawable(text, style)
    unit = size[1] / SCRIPT_HEIGHT
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    # libass lays text out with unhinted advances; at small sizes the hinted advances of PIL come out 0.5 to 1 percent
    # wider, which is enough to move a break. The breaks are decided with a big font and scaled back, the text is drawn
    # with the real one. Its lines can then be a little wider than the space (hinted), so the patch has some slack.
    font = _font(name, style.size * unit * LAYOUT_SCALE)
    available = _width(size, style, name) * LAYOUT_SCALE
    return [line for part in text.split("\n") for line in wrap(part, font, probe, available)]


def render(text: str, size: Tuple[int, int], style: Style = Style()) -> Tuple[Image.Image, Tuple[int, int]]:
    """The subtitle as an RGBA patch and where its top-left corner goes in a frame of `size`."""
    width, height = size
    unit = height / SCRIPT_HEIGHT
    name = _drawable(text, style)[0]
    font = _font(name, style.size * unit)
    outline = max(round(style.outline * unit), 1)
    available = _width(size, style, name)
    margin = (width - available) // 2  # the side margin, or more when a reading length makes the lines shorter
    lines = _lines(text, size, style)
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    slack = min(margin, max(4, round(0.02 * available)))
    margin -= slack
    patch = Image.new("RGBA", (available + 2 * slack, line_height * len(lines) + 2 * outline), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    for index, line in enumerate(lines):
        left = (patch.width - draw.textlength(line, font=font)) / 2
        draw.text((left, outline + index * line_height), line, font=font, fill=style.color + (255,),
                  stroke_width=outline, stroke_fill=style.outline_color + (255,))
    # libass measures the bottom margin to the line box; the outline is drawn outside of it, below the last line
    top = max(height - round(style.margin_bottom * unit) - (patch.height - outline), 0)
    if top + patch.height > height:  # more lines than the frame has room for: the last ones are cut
        patch = patch.crop((0, 0, patch.width, height - top))
    return patch, (margin, top)


_LUMA = {1: (0.2126, 0.0722), 5: (0.299, 0.114), 6: (0.299, 0.114), 2: (0.299, 0.114)}  # FFmpeg matrix id -> (Kr, Kb)


def to_yuv(rgb: np.ndarray, matrix: int, full_range: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y, U, V planes (float, in the 0..255 numbers a video stores) of an RGB picture [h, w, 3] in 0..255.
    The matrix is the video's own (BT.709 or BT.601; unspecified means BT.601 like in swscale)."""
    kr, kb = _LUMA.get(matrix, _LUMA[2])
    r, g, b = (rgb[..., i].astype(np.float32) / 255.0 for i in range(3))
    y = kr * r + (1.0 - kr - kb) * g + kb * b
    pb, pr = (b - y) / (2.0 * (1.0 - kb)), (r - y) / (2.0 * (1.0 - kr))
    if full_range:
        return y * 255.0, 128.0 + pb * 255.0, 128.0 + pr * 255.0
    return 16.0 + y * 219.0, 128.0 + pb * 224.0, 128.0 + pr * 224.0


def _chroma(plane: np.ndarray) -> np.ndarray:
    """Average 2 x 2 blocks (the edge row and column are repeated when the size is odd)."""
    padded = np.pad(plane, ((0, plane.shape[0] % 2), (0, plane.shape[1] % 2)), mode="edge")
    return padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2).mean(axis=(1, 3))


def paginate(cues: Sequence[Cue], size: Tuple[int, int], style: Style = Style(),
             max_lines: int = MAX_LINES) -> List[Cue]:
    """The cues in start order, each cut into pages of at most `max_lines` lines as `render` wraps them in a
    frame of `size`. A page is shown for its share of the cue's time by its number of characters. Empty cues
    (also those with only characters no font has) and cues without time are left out."""
    pages: List[Cue] = []
    for start, end, text in sorted(cues, key=lambda cue: (cue[0], cue[1])):
        text = _drawable(" ".join(str(text).split()), style)[1]
        if not text or end <= start:
            continue
        lines = _lines(text, size, style)
        chunks = [" ".join(lines[i:i + max_lines]) for i in range(0, len(lines), max_lines)] or [text]
        total = sum(len(chunk) for chunk in chunks)
        at = start
        for index, chunk in enumerate(chunks):
            until = end if index == len(chunks) - 1 else at + (end - start) * len(chunk) / total
            pages.append((at, until, chunk))
            at = until
    return pages


def screens(cues: Sequence[Cue], size: Tuple[int, int], style: Style = Style(),
            max_lines: int = MAX_LINES) -> List[Cue]:
    """What is on screen over time, one text at a time in time order: the cues paged (`paginate`), and while
    lines overlap the MAX_STACKED newest of them stacked, the one that started last at the bottom (each page of a
    cue keeps its cue's place, so a later page of an earlier line never hides a newer speaker)."""
    pages = []  # (start, end, text, order of its cue)
    for order, cue in enumerate(sorted(cues, key=lambda cue: (cue[0], cue[1]))):
        pages += [(*page, order) for page in paginate([cue], size, style, max_lines)]
    bounds = sorted({time for start, end, _, _ in pages for time in (start, end)})
    out: List[Cue] = []
    for start, end in zip(bounds, bounds[1:]):
        live = sorted((page for page in pages if page[0] <= start < page[1]), key=lambda page: page[3])
        if not live:
            continue
        text = "\n".join(page[2] for page in live[-MAX_STACKED:])
        if out and out[-1][2] == text and out[-1][1] == start:
            out[-1] = (out[-1][0], end, text)
        else:
            out.append((start, end, text))
    return out


def _due(cues: Sequence[Cue], now: float) -> Optional[int]:
    """The cue on screen at `now`: of overlapping cues the one that started last (`screens` never overlap)."""
    due = None
    for index, (start, end, _) in enumerate(cues):
        if start <= now < end:
            due = index
    return due


def burn(src: str | Path, cues: Sequence[Cue], dst: str | Path, comment: Optional[str] = None,
         style: Style = Style(), crf: int = 17) -> int:
    """Copy a video with the cues burned in; the sound is copied as it is. Returns the frame count.

    The text is laid into the Y, U and V planes of the video itself, with the video's own matrix, so the
    rest of the picture is not touched and does not shift colour while a subtitle is shown. Every frame is
    encoded again (H.264 at `crf`, 17 like the Timeline's export)."""
    shown: Optional[int] = None
    prepared = None
    try:
        with av.open(str(src)) as source:
            stream = source.streams.video[0]
            size = (stream.width, stream.height)
            cues = screens(cues, size, style)
            with video.Writer(dst, stream.average_rate, size[0], size[1], crf=crf, comment=comment,
                              audio_from=src) as writer:
                for frame in video._decode(source, stream):
                    index = _due(cues, float(frame.pts * frame.time_base))
                    if index is not None:
                        if index != shown:
                            patch, corner = render(cues[index][2], size, style)
                            alpha = np.asarray(patch.getchannel("A"), dtype=np.float32) / 255.0
                            y, u, v = to_yuv(np.asarray(patch.convert("RGB")), frame.colorspace, frame.color_range == 2)
                            prepared = (corner, alpha, y, _chroma(alpha), _chroma(u), _chroma(v))
                            shown = index
                        corner, alpha, y, alpha_c, u, v = prepared
                        array = frame.to_ndarray(format="yuv420p")
                        planes = video._planes(array, size[0], size[1])
                        for plane, patch_plane, patch_alpha, scale in ((planes[0], y, alpha, 1), (planes[1], u, alpha_c, 2),
                                                                       (planes[2], v, alpha_c, 2)):
                            video._blend(plane, patch_plane, patch_alpha, corner[0] // scale, corner[1] // scale)
                        composed = av.VideoFrame.from_ndarray(array, format="yuv420p")
                        video.copy_color(frame, composed)
                        frame = composed
                    writer.write(frame)
                return writer.count
    except av.error.FFmpegError as error:
        raise MediaError(f"cannot burn subtitles into {src}: {error}") from error

