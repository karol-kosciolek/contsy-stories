# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Exact text burned into a finished film (app/overlays.py describes it): an end card, a tagline, a web address.

It is drawn the way the subtitles are (engine/media/subtitles.py: PIL letters with an outline, laid into the video's own
Y, U and V planes, so the rest of the picture keeps its exact colours), with a place on the frame of its own: the top, the
middle or the bottom. A text too big for the frame is made smaller until it fits (and `report` says so). A dark panel or a
full card can sit behind the letters, and every overlay fades in and out over a quarter of a second at most. The sound is
copied as it is."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import av
import numpy as np
from PIL import Image, ImageDraw

from . import subtitles, video
from .errors import MediaError

SIZE_UNITS = {"small": 14.0, "medium": 20.0, "large": 28.0}      # a line's height in the units of subtitles.Style
OUTLINE_UNITS = {"small": 1.6, "medium": 2.2, "large": 3.0}
MIN_UNITS = 7.0
MAX_SHARE = 0.8        # of the frame's height one text may take before it is made smaller
SHRINK = 0.88
SIDE_MARGIN = 18.0     # units
BOTTOM_MARGIN = 36.0   # units, from the frame's bottom edge to the text's last line
TOP_SHARE = 0.07       # of the frame's height, from the top edge to the text
PANEL = (0, 0, 0, 184)
CARD = (0, 0, 0, 255)
FADE = 0.25            # seconds, at most; a short overlay fades over a third of its time


@dataclass(frozen=True)
class Layer:
    start: float
    end: float
    text: str
    position: str = "center"
    size: str = "medium"
    backdrop: str = "none"


@dataclass(frozen=True)
class Composed:
    image: Image.Image          # RGBA: the text, with its panel or card
    corner: Tuple[int, int]     # where its top-left corner goes in the frame
    shrunk: bool                # the text had to be made smaller to fit
    lost: bool = False          # some characters no font of this PC can draw were left out


def _letters(text: str) -> int:
    """How many characters of `text` are drawn: not the spaces, and not the variation selectors (U+FE00 to U+FE0F) that
    choose a text or emoji look of the character before them and have no picture of their own."""
    return sum(not char.isspace() and not "\ufe00" <= char <= "\ufe0f" for char in text)


def _words_fit(text: str, size: Tuple[int, int], style: subtitles.Style) -> bool:
    """Whether every word (a web address too) fits on a line of the frame, so that none is cut in the middle."""
    name, drawn = subtitles._drawable(text, style)
    font = subtitles._font(name, style.size * size[1] / subtitles.SCRIPT_HEIGHT)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    available = subtitles._width(size, style, name)
    return all(probe.textlength(word, font=font) <= available for word in drawn.split())


def compose(layer: Layer, size: Tuple[int, int]) -> Composed:
    """The overlay as a picture for a frame of `size` (width, height). The text is made smaller while it takes more than
    MAX_SHARE of the frame's height or a word would be cut."""
    width, height = size
    base = SIZE_UNITS[layer.size]
    units, shrunk = base, False
    wanted = _letters(layer.text)
    kept = _letters(subtitles._drawable(layer.text, subtitles.Style())[1])
    if not kept:
        raise MediaError(f"no font of this PC can draw the text “{layer.text.splitlines()[0][:40]}”")
    lost = kept < wanted
    while True:
        style = subtitles.Style(size=units, outline=OUTLINE_UNITS[layer.size] * units / base,
                                margin_bottom=BOTTOM_MARGIN, margin_side=SIDE_MARGIN)
        patch, corner = subtitles.render(layer.text, size, style)
        if (patch.height <= MAX_SHARE * height and _words_fit(layer.text, size, style)) or units <= MIN_UNITS:
            break
        units, shrunk = max(units * SHRINK, MIN_UNITS), True
    x, y = corner
    if layer.position == "top":
        y = round(TOP_SHARE * height)
    elif layer.position == "center":
        y = (height - patch.height) // 2
    if layer.backdrop == "card":
        image = Image.new("RGBA", size, CARD)
        image.alpha_composite(patch, (x, y))
        return Composed(image, (0, 0), shrunk, lost)
    if layer.backdrop == "panel":
        pad = round(0.03 * height)
        side = max(min(pad, x), 0)
        image = Image.new("RGBA", (patch.width + 2 * side, patch.height + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(image).rounded_rectangle((0, 0, image.width - 1, image.height - 1), radius=pad, fill=PANEL)
        image.alpha_composite(patch, (side, pad))
        return Composed(image, (x - side, y - pad), shrunk, lost)
    return Composed(patch, (x, y), shrunk, lost)


def fade(now: float, start: float, end: float, film_end: Optional[float] = None) -> float:
    """How visible an overlay is at `now`: 0 to 1 over its first moments, 1, then 1 to 0 over its last. An overlay that
    runs to the end of the film (`film_end`) does not fade out: the film ends on it, as an end card should."""
    length = min(FADE, (end - start) / 3.0)
    if length <= 0:
        return 1.0
    out = 1.0 if film_end is not None and end >= film_end - 0.1 else max(0.0, (end - now) / length)
    return float(min(1.0, max(0.0, (now - start) / length), out))


def _length(source, stream) -> Optional[float]:
    """The film's length in seconds, when the file says."""
    if stream.duration and stream.time_base:
        return float(stream.duration * stream.time_base)
    return source.duration / av.time_base if source.duration else None


def _prepare(composed: Composed, frame) -> tuple:
    image = composed.image
    alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
    y, u, v = subtitles.to_yuv(np.asarray(image.convert("RGB")), frame.colorspace, frame.color_range == 2)
    return composed.corner, alpha, y, subtitles._chroma(alpha), subtitles._chroma(u), subtitles._chroma(v)


def burn(src: str | Path, layers: Sequence[Layer], dst: str | Path, comment: Optional[str] = None, crf: int = 17,
         report: Optional[list] = None) -> int:
    """Copy a video with the overlays burned in; the sound is copied as it is. Returns the frame count. `report` (a list)
    gets a sentence for every overlay whose text had to be made smaller to fit the frame."""
    prepared: dict = {}
    try:
        with av.open(str(src)) as source:
            stream = source.streams.video[0]
            size = (stream.width, stream.height)
            shown = [layer for layer in layers if layer.end > layer.start]
            composed = [compose(layer, size) for layer in shown]
            film_end = _length(source, stream)
            for layer, one in zip(shown, composed):
                if report is not None and one.shrunk:
                    report.append(f"The text “{layer.text.splitlines()[0][:40]}” was too big for the frame and was made "
                                  "smaller to fit.")
                if report is not None and one.lost:
                    report.append(f"Some characters of the text “{layer.text.splitlines()[0][:40]}” cannot be drawn with the "
                                  "fonts of this PC and were left out.")
            with video.Writer(dst, stream.average_rate, size[0], size[1], crf=crf, comment=comment,
                              audio_from=src) as writer:
                for frame in video._decode(source, stream):
                    now = float(frame.pts * frame.time_base)
                    due = sorted((number for number, layer in enumerate(shown) if layer.start <= now < layer.end),
                                 key=lambda number: (shown[number].backdrop != "card", number))  # cards go underneath
                    if due:
                        array = frame.to_ndarray(format="yuv420p")
                        planes = video._planes(array, size[0], size[1])
                        for number in due:
                            if number not in prepared:
                                prepared[number] = _prepare(composed[number], frame)
                            corner, alpha, y, alpha_c, u, v = prepared[number]
                            level = fade(now, shown[number].start, shown[number].end, film_end)
                            if level < 1.0:
                                alpha, alpha_c = alpha * level, alpha_c * level
                            for plane, patch_plane, patch_alpha, scale in ((planes[0], y, alpha, 1),
                                                                           (planes[1], u, alpha_c, 2),
                                                                           (planes[2], v, alpha_c, 2)):
                                video._blend(plane, patch_plane, patch_alpha, corner[0] // scale, corner[1] // scale)
                        mixed = av.VideoFrame.from_ndarray(array, format="yuv420p")
                        video.copy_color(frame, mixed)
                        frame = mixed
                    writer.write(frame)
                return writer.count
    except av.error.FFmpegError as error:
        raise MediaError(f"cannot burn text into {src}: {error}") from error
