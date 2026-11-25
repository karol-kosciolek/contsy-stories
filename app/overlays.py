# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Exact text over the finished film: an end card, a tagline, a web address, a product name. A video model cannot spell
(it draws letters that look like text and are not), so text that must be right is never asked of it: the project keeps
it as overlays (Project.overlays) and the export burns it into the picture itself (engine/media/overlays.py), with
the real letters, in the same place on every frame.

An overlay is shown from `start` to `end` seconds of the film, or during the last `last` seconds of it (an end card
stays at the end when scenes change length). It has a position (top, center, bottom), a size, and a backdrop: none (the
letters have an outline), a dark panel behind them, or a full card that hides the picture. Times are in tenths of a
second, like a scene's length. This module only describes and checks them; it does not touch a file."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

MAX_OVERLAYS = 9      # the agent looks at a frame of each on one sheet (agent_eyes.CELLS_PER_SHEET)
MAX_TEXT = 200       # characters of one overlay
MAX_LINES = 4        # lines written by hand (a long line is wrapped by the frame's width as well)
MIN_SECONDS = 0.5
MAX_TIME = 3600.0
POSITIONS = ("center", "top", "bottom")
SIZES = ("small", "medium", "large")
BACKDROPS = ("none", "panel", "card")
# Scripts whose letters change shape with their neighbours or run right to left: the burner lays letters out one after the
# other, left to right (no raqm), so these would be drawn wrongly. Text that must be exact is refused instead.
_NOT_DRAWN_RIGHT = ("ARABIC", "HEBREW", "SYRIAC", "THAANA", "NKO", "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA",
                    "TAMIL", "TELUGU", "KANNADA", "MALAYALAM", "SINHALA", "THAI", "LAO", "TIBETAN", "MYANMAR", "KHMER")
_DEFAULTS = {"start": 0.0, "end": 0.0, "last": 0.0, "position": "center", "size": "medium", "backdrop": "none"}


@dataclass(frozen=True)
class Overlay:
    text: str
    start: float = 0.0       # seconds from the start of the film ...
    end: float = 0.0         # ... to this second (both 0 when `last` is used)
    last: float = 0.0        # above 0: shown during the last this many seconds of the film
    position: str = "center"
    size: str = "medium"
    backdrop: str = "none"

    def to_json(self) -> dict:
        """The stored form: the text and only what differs from the defaults."""
        row = {"text": self.text}
        for name, default in _DEFAULTS.items():
            value = getattr(self, name)
            value = float(value) if isinstance(default, float) else value  # 1 and 1.0 are the same stored overlay
            if value != default:
                row[name] = value
        return row


def _seconds(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number of seconds.")
    try:
        number = float(value)  # a huge whole number overflows here
    except OverflowError:
        number = math.inf
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a number of seconds.")
    return round(number, 1)


def _text(value) -> str:
    if not isinstance(value, str):
        raise ValueError("The text of an overlay must be text.")
    lines = []
    for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if any(unicodedata.category(char).startswith("C") and not char.isspace() for char in line):
            raise ValueError("The text of an overlay holds a control or invisible character.")
        if any(unicodedata.bidirectional(char) in ("R", "AL") or unicodedata.name(char, "").startswith(_NOT_DRAWN_RIGHT)
               for char in line):
            raise ValueError("Right-to-left scripts (Arabic, Hebrew) and scripts that join or reorder letters (Thai, "
                             "Devanagari and the like) cannot be drawn correctly yet: use Latin, Cyrillic, Greek, "
                             "Chinese, Japanese or Korean text.")
        line = " ".join(line.split())
        if line:
            lines.append(line)
    if not lines:
        raise ValueError("An overlay needs some text.")
    if len(lines) > MAX_LINES:
        raise ValueError(f"An overlay has at most {MAX_LINES} lines.")
    if sum(map(len, lines)) + len(lines) - 1 > MAX_TEXT:
        raise ValueError(f"The text of an overlay is at most {MAX_TEXT} characters.")
    return "\n".join(lines)


def clean(overlay) -> Overlay:
    """The overlay with its text tidied (spaces, empty lines) and its times on the grid of tenths; ValueError says what is
    wrong. A row (a dict) is accepted too."""
    if isinstance(overlay, dict):
        unknown = set(overlay) - {"text", *_DEFAULTS}
        if unknown:
            raise ValueError(f"An overlay has no {sorted(unknown)[0]!r}.")
        overlay = Overlay(**{name: overlay[name] for name in overlay})
    if not isinstance(overlay, Overlay):
        raise ValueError("Invalid overlay.")
    start, end, last = (_seconds(getattr(overlay, name), name) for name in ("start", "end", "last"))
    if last:
        if start or end:
            raise ValueError("An overlay is shown either from start to end or during the last seconds, not both.")
        if not MIN_SECONDS <= last <= MAX_TIME:
            raise ValueError(f"last must be between {MIN_SECONDS} and {MAX_TIME:g} seconds.")
    else:
        if start < 0 or end > MAX_TIME:
            raise ValueError(f"An overlay is shown between 0 and {MAX_TIME:g} seconds of the film.")
        if end - start < MIN_SECONDS - 1e-9:
            raise ValueError(f"An overlay is shown for at least {MIN_SECONDS} seconds (end after start).")
    for name, allowed in (("position", POSITIONS), ("size", SIZES), ("backdrop", BACKDROPS)):
        if getattr(overlay, name) not in allowed:
            raise ValueError(f"{name} is one of: {', '.join(allowed)}.")
    return Overlay(_text(overlay.text), start, end, last, overlay.position, overlay.size, overlay.backdrop)


def cleaned(rows: Iterable) -> tuple[Overlay, ...]:
    """Every overlay checked (see clean); at most MAX_OVERLAYS. ValueError names the overlay that is wrong."""
    rows = tuple(rows)
    if len(rows) > MAX_OVERLAYS:
        raise ValueError(f"A film has at most {MAX_OVERLAYS} overlays.")
    found = []
    for number, row in enumerate(rows, 1):
        try:
            found.append(clean(row))
        except (ValueError, TypeError) as error:
            raise ValueError(f"Overlay {number}: {error}") from error
    return tuple(found)


def tolerant(rows) -> tuple[Overlay, ...]:
    """The overlays of a stored project: a row that cannot be read is left out, so a damaged entry never makes a project
    unopenable (the next save writes the rest)."""
    if not isinstance(rows, list):
        return ()
    found = []
    for row in rows[:MAX_OVERLAYS]:
        try:
            found.append(clean(row))
        except (ValueError, TypeError):
            continue
    return tuple(found)


def key(overlays: Sequence[Overlay]) -> str:
    """What an export records about its overlays: '' without any, else a short hash of all of them."""
    if not overlays:
        return ""
    data = json.dumps([overlay.to_json() for overlay in overlays], sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def window(overlay: Overlay, length: float) -> tuple[float, float] | None:
    """(from, to) seconds on the clock of a film that is `length` seconds long, or None when it is not shown at all (it
    starts after the film's end). A time past the end is cut at the end."""
    if overlay.last:
        start, end = max(length - overlay.last, 0.0), length
    else:
        start, end = overlay.start, min(overlay.end, length)
    return (round(start, 3), round(end, 3)) if end - start > 0.05 else None


def layers(overlays: Sequence[Overlay], length: float) -> tuple[list[tuple], list[str]]:
    """([(start, end, text, position, size, backdrop)] for the burner, notes about the overlays that are not shown)."""
    found, notes = [], []
    for number, overlay in enumerate(overlays, 1):
        shown = window(overlay, length)
        if shown is None:
            notes.append(f"Overlay {number} (“{overlay.text.splitlines()[0][:40]}”) starts after the end of the film "
                         f"({length:.1f} s) and was left out.")
            continue
        found.append((*shown, overlay.text, overlay.position, overlay.size, overlay.backdrop))
    return found, notes


def rows(overlays: Sequence[Overlay]) -> list[dict]:
    """The overlays as the tools take them (set_overlays, save_project): what get_project shows can be sent back as it is.
    `last` is called last_seconds there; what is the default is left out."""
    out = []
    for overlay in overlays:
        row = overlay.to_json()
        if "last" in row:
            row["last_seconds"] = row.pop("last")
        out.append(row)
    return out


def view(overlays: Sequence[Overlay], length: float | None = None) -> list[dict]:
    """The overlays as a report shows them: number (1 is the first of the list), the row `rows` gives, and when it is shown
    (`shown_from_to`, resolved against the film's length `length`; null when it is not shown at all)."""
    out = []
    for number, (overlay, row) in enumerate(zip(overlays, rows(overlays)), 1):
        shown = window(overlay, length) if length is not None else None
        out.append({"number": number, **row, **({"shown_from_to": list(shown) if shown else None}
                                               if length is not None else {})})
    return out


def clashes(overlays: Sequence[Overlay], length: float) -> list[str]:
    """Pairs of overlays that are shown at the same time in the same place (one would hide the other). A card covers the
    whole frame, so two cards clash wherever they are; text is always drawn over a card (the burner draws cards first)."""
    out = []
    for first in range(len(overlays)):
        for second in range(first + 1, len(overlays)):
            one, two = overlays[first], overlays[second]
            a, b = window(one, length), window(two, length)
            if not (a and b and a[0] < b[1] and b[0] < a[1]):
                continue
            if one.backdrop == "card" and two.backdrop == "card":
                out.append(f"Overlays {first + 1} and {second + 1} are both cards shown at the same time: one covers the "
                           "other. Give them different times.")
            elif one.position == two.position and "card" not in (one.backdrop, two.backdrop):
                out.append(f"Overlays {first + 1} and {second + 1} are shown at the same time at the {one.position}: one "
                           "covers the other. Give them different positions or times.")
    return out
