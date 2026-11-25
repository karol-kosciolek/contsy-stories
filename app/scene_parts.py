# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Structured parts of a Timeline scene: spoken lines with entry times, timed sound effects,
mix levels and the sound checks shown before a render.

These small value types are shared by the project file (app/timeline.py), sound preparation
(app/scene_sound.py, app/video_gen.py) and the editor (app/ui_timeline.py). Every time is in
seconds on the scene clock: 0.0 is the first frame of the scene.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

MAX_LINES = 12
MAX_EFFECTS = 8
MAX_REFERENCES = 6
MAX_TEXT = 300              # characters in a spoken line or a sound description
FIRST_LINE = 0.3            # automatic placement: the first line starts here (as dialogue.assemble)
LINE_GAP = 0.5              # automatic placement: silence after the previous line
WORDS_PER_SECOND = 2.6      # speech length estimate before synthesis
MIN_LINE_SECONDS = 0.8
EFFECT_SECONDS = 2.0        # default length of a sound effect
EFFECT_RANGE = (0.5, 8.0)
LEVEL_RANGE = (-24.0, 12.0)  # dB applied to dialogue or background in previews and the export
VOICE_PREFIX = "@"          # speaker "@hanna": a library voice for a person without a character
ERROR, WARNING, INFO = "error", "warning", "info"
PACE_RANGE = (90.0, 200.0)  # words a minute a line's pace may cap it at
NARRATOR_PACE = 135.0       # a children's narrator (ages 3-7: 120-140 words a minute)
SLOWEST_TEMPO = 0.8         # a line is never slowed down more than this (VoxCPM reads ~170 words a minute)

_SPEAKER = re.compile(r"@?[a-z0-9_]{1,24}")
_SHA = re.compile(r"[0-9a-f]{64}")
_WORD = re.compile(r"\w+")


def _text(value, what: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid {what}.")
    text = " ".join(value.split())
    if len(text) > MAX_TEXT:
        raise ValueError(f"Keep each {what} under {MAX_TEXT} characters.")
    return text


def _time(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Enter the {what} in seconds from the scene start.")
    return round(float(value), 2)


@dataclass(frozen=True)
class Cue:
    """One spoken line.

    speaker: a character slug, "@<voice slug>" for a library voice without a character, or ""
    for the scene's only character. start: entry time on the scene clock; None places the line
    LINE_GAP after the previous one (the first at FIRST_LINE). pace: the most words a minute the line is spoken at
    (0 = as synthesized); a faster line is slowed down (scene_sound.prepare), so a children's narrator reads calmly."""
    text: str
    speaker: str = ""
    start: Optional[float] = None
    pace: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "text", _text(self.text, "spoken line"))
        if not isinstance(self.speaker, str) or (self.speaker and not _SPEAKER.fullmatch(self.speaker)):
            raise ValueError("Choose who says the line.")
        if self.start is not None:
            object.__setattr__(self, "start", _time(self.start, "line start"))
        if isinstance(self.pace, bool) or not isinstance(self.pace, (int, float)) or not (
                self.pace == 0 or PACE_RANGE[0] <= self.pace <= PACE_RANGE[1]):
            raise ValueError(f"A line's pace is 0 or {PACE_RANGE[0]:g}-{PACE_RANGE[1]:g} words a minute.")
        object.__setattr__(self, "pace", float(self.pace))

    @property
    def voice(self) -> str:
        """The library voice slug of an "@voice" speaker, else ""."""
        return self.speaker[1:] if self.speaker.startswith(VOICE_PREFIX) else ""

    @property
    def character(self) -> str:
        """The character slug of a character speaker, else ""."""
        return "" if self.speaker.startswith(VOICE_PREFIX) else self.speaker

    def to_json(self) -> dict:
        data = {"text": self.text}
        if self.speaker:
            data["speaker"] = self.speaker
        if self.start is not None:
            data["start"] = self.start
        if self.pace:
            data["pace"] = self.pace
        return data

    @classmethod
    def from_json(cls, data) -> "Cue":
        if not isinstance(data, dict) or not set(data) <= {"text", "speaker", "start", "pace"}:
            raise ValueError("Invalid spoken line.")
        return cls(data.get("text"), data.get("speaker", ""), data.get("start"), data.get("pace", 0.0))


@dataclass(frozen=True)
class Effect:
    """A sound effect generated from its description and placed at `at` for `seconds`."""
    text: str
    at: float = 0.0
    seconds: float = EFFECT_SECONDS

    def __post_init__(self):
        object.__setattr__(self, "text", _text(self.text, "sound effect"))
        object.__setattr__(self, "at", _time(self.at, "effect time"))
        length = _time(self.seconds, "effect length")
        if not EFFECT_RANGE[0] <= length <= EFFECT_RANGE[1]:
            raise ValueError(f"An effect lasts {EFFECT_RANGE[0]:g}–{EFFECT_RANGE[1]:g} s.")
        object.__setattr__(self, "seconds", length)

    def to_json(self) -> dict:
        data = {"text": self.text, "at": self.at}
        if self.seconds != EFFECT_SECONDS:
            data["seconds"] = self.seconds
        return data

    @classmethod
    def from_json(cls, data) -> "Effect":
        if not isinstance(data, dict) or not set(data) <= {"text", "at", "seconds"}:
            raise ValueError("Invalid sound effect.")
        return cls(data.get("text"), data.get("at", 0.0), data.get("seconds", EFFECT_SECONDS))


@dataclass(frozen=True)
class Issue:
    """A sound or dialogue problem. "error" blocks generation, "warning" needs a look, "info"
    explains a default the scene will use."""
    level: str
    text: str
    line: Optional[int] = None    # index into the scene's cues
    effect: Optional[int] = None  # index into the scene's effects


def level(value) -> float:
    """A mix level in dB, rounded to 0.5 dB."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Enter a level in dB.")
    if not LEVEL_RANGE[0] <= value <= LEVEL_RANGE[1]:
        raise ValueError(f"Choose a level from {LEVEL_RANGE[0]:g} to +{LEVEL_RANGE[1]:g} dB.")
    return round(float(value) * 2) / 2 + 0.0  # + 0.0 turns -0.0 into 0.0


def length(value) -> int | float:
    """A scene length on the 0.1 s grid (the owner's request of 2026-11-16): a whole number of seconds stays an int,
    as every scene before stored it (so its keys stay the same), any other length a float with one decimal.
    ValueError for anything else; the 1 s minimum and the maximum are the caller's."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Enter the scene length in seconds.")
    rounded = round(float(value), 1)
    if abs(rounded - value) > 1e-6:
        raise ValueError("Use tenths of a second for the scene length, for example 2.5.")
    return int(rounded) if rounded == int(rounded) else rounded


def is_length(value) -> bool:
    """A number `length` takes as it is (0.1 s grid)."""
    try:
        length(value)
    except ValueError:
        return False
    return True


def is_reference(name) -> bool:
    """A stored reference image name: the sha256 of its PNG file."""
    return isinstance(name, str) and bool(_SHA.fullmatch(name))


def estimate_seconds(text: str) -> float:
    """Spoken length of a line before it is synthesized."""
    words = len(_WORD.findall(text or ""))
    return round(max(MIN_LINE_SECONDS, words / WORDS_PER_SECOND + 0.25), 2)


def timings(cues: Sequence[Cue], durations: Optional[Sequence[float]] = None) -> list[tuple[float, float]]:
    """(start, end) of every cue on the scene clock.

    durations: measured lengths after synthesis, one per cue; estimate_seconds otherwise. A cue
    without a start follows the end of the earlier cues (the latest one, so it never lands on a
    line that is still being spoken) after LINE_GAP; the first one starts at FIRST_LINE."""
    if durations is not None and len(durations) != len(cues):
        raise ValueError("Each spoken line needs its measured length.")
    result: list[tuple[float, float]] = []
    latest_end = None
    for index, cue in enumerate(cues):
        length = float(durations[index]) if durations is not None else estimate_seconds(cue.text)
        if cue.start is not None:
            start = cue.start
        else:
            start = FIRST_LINE if latest_end is None else latest_end + LINE_GAP
        end = start + max(length, 0.0)
        result.append((round(start, 3), round(end, 3)))
        latest_end = end if latest_end is None else max(latest_end, end)
    return result


def check(cues: Sequence[Cue], effects: Sequence[Effect], seconds: int) -> None:
    """Raise ValueError for parts that cannot be saved. The scene's length is never refused: a line or an effect
    that starts after the scene end is only not heard (timing_issues says so)."""
    if len(cues) > MAX_LINES:
        raise ValueError(f"Use at most {MAX_LINES} spoken lines in a scene.")
    if len(effects) > MAX_EFFECTS:
        raise ValueError(f"Use at most {MAX_EFFECTS} sound effects in a scene.")
    for number, cue in enumerate(cues, 1):
        if not cue.text:
            raise ValueError(f"Line {number}: write the spoken text or remove the line.")
    for number, effect in enumerate(effects, 1):
        if not effect.text:
            raise ValueError(f"Effect {number}: describe the sound or remove the effect.")


def heard(effects: Sequence[Effect], seconds: float) -> tuple[Effect, ...]:
    """The effects that start before the scene end: the others are not heard, so they are never generated."""
    return tuple(effect for effect in effects if effect.at < seconds)


def timing_issues(cues: Sequence[Cue], effects: Sequence[Effect], seconds: int,
                  durations: Optional[Sequence[float]] = None) -> list[Issue]:
    """Lines that run past the scene or overlap, and effects outside it. None of them stops a scene: its length is
    the one chosen for it (the owner's decision of 2026-11-16, even a 1-second scene), so a line or an effect is cut
    at the scene end, and one that starts after it is not heard. These are warnings, with estimated or measured
    durations alike."""
    issues: list[Issue] = []
    windows = timings(cues, durations)
    speaking = None  # (end, index) of the earlier line that ends last
    for index, (start, end) in enumerate(windows):
        number = index + 1
        if start >= seconds:
            issues.append(Issue(WARNING, f"Line {number} starts at {start:.1f} s, after the scene ends at {seconds} s, "
                                         "so it is not heard. Make the scene longer to hear it.", line=index))
        elif end > seconds + 1e-6:
            when = f"should end around {end:.1f} s" if durations is None else f"ends at {end:.1f} s"
            issues.append(Issue(WARNING, f"Line {number} {when}, after the scene ends at {seconds} s, so its end is "
                                         "cut off. Make the scene longer to hear all of it.", line=index))
        if speaking is not None and start < speaking[0] - 0.05:
            issues.append(Issue(WARNING, f"Line {number} starts before line {speaking[1] + 1} ends.", line=index))
        if speaking is None or end > speaking[0]:
            speaking = end, index
    for index, effect in enumerate(effects):
        number = index + 1
        if effect.at >= seconds:
            issues.append(Issue(WARNING, f"Effect {number} is at {effect.at:.1f} s, after the scene ends at "
                                         f"{seconds} s, so it is not heard.", effect=index))
        elif effect.at + effect.seconds > seconds + 1e-6:
            issues.append(Issue(INFO, f"Effect {number} is cut at the scene end.", effect=index))
    return issues


def cues_from_json(text: str) -> tuple[Cue, ...]:
    """Cues from the editor's JSON list. Rows without text are editing placeholders and are
    left out."""
    rows = _rows(text, "spoken lines")
    return tuple(Cue.from_json(row) for row in rows if not _blank(row))


def effects_from_json(text: str) -> tuple[Effect, ...]:
    """Effects from the editor's JSON list; rows without a description are left out."""
    rows = _rows(text, "sound effects")
    return tuple(Effect.from_json(row) for row in rows if not _blank(row))


def _rows(text: str, what: str) -> list:
    if not text or not str(text).strip():
        return []
    try:
        rows = json.loads(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Could not read the {what}.") from error
    if not isinstance(rows, list):
        raise ValueError(f"Could not read the {what}.")
    return rows


def _blank(row) -> bool:
    return isinstance(row, dict) and isinstance(row.get("text"), str) and not row["text"].strip()
