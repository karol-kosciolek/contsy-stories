# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Colour words as HSV ranges, and pixel masks for them.

The series bible names its characters' colours in words ("teal-blue body, yellow bobble hat", app/bible.py). The shot
check (app/shot_qc.py) turns those words into pixel masks to see whether a character, its accessory or its body colour is
in a frame. Everything here is plain numpy on the CPU.

Hue is in degrees (0-360), saturation and value 0-1. A range may wrap around 0 (red). The ranges were measured on the
reference photos of the Lantern Woods cast (tests/fixtures/quality/refs) and on frames of the series' episodes: the
teal of Lumi's body stays inside "teal", his hat inside "yellow", Barley's lavender feathers inside "lavender" and his
bow tie inside "red", while the forest greens, the sky and the water fall outside all four.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class HsvRange:
    """Pixels with hue in [h_lo, h_hi] (wrapping when h_lo > h_hi), saturation in [s_lo, s_hi] and value in
    [v_lo, v_hi]."""
    h_lo: float
    h_hi: float
    s_lo: float = 0.0
    s_hi: float = 1.0
    v_lo: float = 0.0
    v_hi: float = 1.0

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        hue = (h >= self.h_lo) & (h <= self.h_hi) if self.h_lo <= self.h_hi else (h >= self.h_lo) | (h <= self.h_hi)
        return hue & (s >= self.s_lo) & (s <= self.s_hi) & (v >= self.v_lo) & (v <= self.v_hi)


# The colour words the bible understands. Achromatic words (white, grey, black) ignore the hue.
COLORS: Dict[str, Tuple[HsvRange, ...]] = {
    "red": (HsvRange(345, 12, 0.50, 1.0, 0.35, 1.0),),
    "orange": (HsvRange(14, 36, 0.55, 1.0, 0.45, 1.0),),
    "yellow": (HsvRange(38, 66, 0.42, 1.0, 0.45, 1.0),),
    "gold": (HsvRange(36, 56, 0.45, 1.0, 0.45, 1.0),),
    "green": (HsvRange(70, 158, 0.25, 1.0, 0.18, 1.0),),
    "teal": (HsvRange(160, 196, 0.22, 1.0, 0.30, 1.0),),
    "turquoise": (HsvRange(160, 196, 0.22, 1.0, 0.30, 1.0),),
    "blue": (HsvRange(196, 248, 0.28, 1.0, 0.22, 1.0),),
    "navy": (HsvRange(210, 250, 0.35, 1.0, 0.10, 0.5),),
    "lavender": (HsvRange(232, 300, 0.10, 0.60, 0.38, 1.0),),
    "purple": (HsvRange(250, 300, 0.18, 1.0, 0.25, 1.0),),
    "violet": (HsvRange(250, 300, 0.18, 1.0, 0.25, 1.0),),
    "lilac": (HsvRange(232, 300, 0.10, 0.60, 0.38, 1.0),),
    "pink": (HsvRange(300, 345, 0.18, 0.80, 0.50, 1.0),),
    "cream": (HsvRange(20, 65, 0.04, 0.32, 0.72, 1.0),),
    "beige": (HsvRange(20, 55, 0.08, 0.40, 0.55, 1.0),),
    "brown": (HsvRange(8, 40, 0.30, 0.85, 0.15, 0.62),),
    "white": (HsvRange(0, 360, 0.0, 0.10, 0.86, 1.0),),
    "grey": (HsvRange(0, 360, 0.0, 0.10, 0.28, 0.86),),
    "gray": (HsvRange(0, 360, 0.0, 0.10, 0.28, 0.86),),
    "silver": (HsvRange(0, 360, 0.0, 0.10, 0.55, 0.95),),
    "black": (HsvRange(0, 360, 0.0, 1.0, 0.0, 0.14),),
}
_WORD = re.compile(r"[a-z]+")


def words(text: str) -> List[str]:
    """The colour words of a phrase, in order ("teal-blue body" -> ["teal", "blue"])."""
    return [word for word in _WORD.findall((text or "").lower()) if word in COLORS]


def ranges(text: str) -> Tuple[HsvRange, ...]:
    """The HSV ranges of a colour phrase: a compound ("teal-blue", "lavender-grey") is the union of its words, so a
    character that is described between two colours is found in either; () when it names no known colour."""
    found: List[HsvRange] = []
    for word in words(text):
        found.extend(COLORS[word])
    return tuple(dict.fromkeys(found))


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB (uint8 or 0-1 floats, ...x3) to HSV with hue in degrees and saturation/value 0-1."""
    rgb = rgb.astype(np.float32)
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    high = rgb.max(axis=-1)
    low = rgb.min(axis=-1)
    delta = high - low
    safe = np.where(delta > 1e-6, delta, 1.0)
    hue = np.where(high == r, ((g - b) / safe) % 6.0,
                   np.where(high == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0)) * 60.0
    hue = np.where(delta > 1e-6, hue, 0.0)
    saturation = np.where(high > 1e-6, delta / np.where(high > 1e-6, high, 1.0), 0.0)
    return np.stack([hue % 360.0, saturation, high], axis=-1)


def mask(hsv: np.ndarray, found: Iterable[HsvRange]) -> np.ndarray:
    """Pixels inside any of the ranges."""
    out = np.zeros(hsv.shape[:2], dtype=bool)
    for one in found:
        out |= one.mask(hsv)
    return out


def share(hsv: np.ndarray, found: Iterable[HsvRange], region: Optional[np.ndarray] = None) -> float:
    """The share of the frame (or of `region`, a boolean mask) whose pixels are inside the ranges."""
    hit = mask(hsv, found)
    if region is not None:
        total = int(region.sum())
        return float((hit & region).sum()) / total if total else 0.0
    return float(hit.mean()) if hit.size else 0.0


def largest_blob_box(hit: np.ndarray, min_share: float = 0.0005) -> Optional[Tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) of the pixels of a mask, after dropping rows and columns with almost none of them (stray
    pixels of the same colour elsewhere); None when the mask holds under `min_share` of the frame."""
    if hit.size == 0 or hit.mean() < min_share:
        return None
    rows = hit.sum(axis=1)
    cols = hit.sum(axis=0)
    keep_rows = np.nonzero(rows >= max(1, 0.08 * rows.max()))[0]
    keep_cols = np.nonzero(cols >= max(1, 0.08 * cols.max()))[0]
    if not keep_rows.size or not keep_cols.size:
        return None
    return int(keep_cols[0]), int(keep_rows[0]), int(keep_cols[-1]) + 1, int(keep_rows[-1]) + 1
