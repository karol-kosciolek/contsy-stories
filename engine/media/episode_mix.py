# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The soundtrack of a series episode, mixed as one programme (docs/QUALITY_ARCHITECTURE.md, layer 5).

engine/media/audio.timeline_soundtrack mixes every scene on its own: its own room tone, its own speech level, and
nothing that joins them but a short cross-fade. A narrated episode needs what a real production does:

- one narration track: every scene's speech placed on the episode clock at one loudness (VOICE_LUFS), and a line may
  start a little before its picture (a J-cut, J_CUT seconds at most, only where the previous line left room);
- ambience per place, not per clip: the scenes of one location in a row share one continuous bed (the first audible
  background of the group, looped with long cross-fades); it changes, with a cross-fade, only when the place does;
  never digital silence (a room tone is put under a silent bed);
- an optional music bed from the owner's library (no model makes it), ducked under the narrator;
- the beds stay at least BED_UNDER_VOICE dB under the narrator, and duck further while he speaks;
- a master: the whole programme to TARGET_LUFS integrated with its true peak at or under TARGET_TP.

episode_soundtrack() returns the per-scene pieces (for join_timeline's writer) and a report of what it measured.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from . import audio as _audio
from . import loudness
from .errors import MediaError

TARGET_LUFS = -14.0      # YouTube and most platforms normalize to -14 LUFS
TARGET_TP = -1.5         # dBTP, measured on the delivered file
MASTER_TP = -2.0         # ...so the mix aims lower: AAC encoding raised -1.50 to -1.34 dBTP in the test episode
VOICE_LUFS = -18.0       # the narration before the master (the master then lifts the programme as a whole)
BED_UNDER_VOICE = 14.0   # the ambience this many LU under the narration (requirement: at least 12)
MUSIC_UNDER_VOICE = 18.0
DUCK_BED_DB = 5.0        # extra cut of the ambience while the narrator speaks
DUCK_MUSIC_DB = 8.0
J_CUT = 0.35             # the most a line starts before its picture
J_GAP = 0.45             # ...only where the line before ended at least this long before the cut, plus the shift
CROSSFADE = 1.0          # between two places' beds, and inside a looped bed
ROOM_TONE_LUFS = -52.0   # what plays under a bed that is silent: never digital silence
FADE_IN, FADE_OUT = 0.5, 0.8


@dataclass
class SceneSound:
    background: Optional[Path]     # the handle-padded background stem (TIMELINE_AUDIO_HANDLE before and after)
    voice: Optional[Path]          # the scene's speech on its own clock
    seconds: float
    effects: Optional[Path] = None  # the effects overlay inside the background
    location: str = ""             # scenes in a row with the same location share one ambience
    voice_db: float = 0.0
    background_db: float = 0.0


@dataclass
class MixReport:
    loudness: float = 0.0
    true_peak: float = 0.0
    voice_loudness: Optional[float] = None
    bed_loudness: Optional[float] = None
    music_loudness: Optional[float] = None
    bed_under_voice: Optional[float] = None
    music: str = ""
    locations: List[Tuple[str, float, float]] = field(default_factory=list)  # (location, start s, end s)
    j_cuts: List[Tuple[int, float]] = field(default_factory=list)              # (scene number from 1, seconds early)
    room_tone: List[str] = field(default_factory=list)                         # places whose bed was silent
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {key: (round(value, 2) if isinstance(value, float) else value) for key, value in self.__dict__.items()}


def _gain_db(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _room_tone(count: int, rate: int, seed: int = 7) -> np.ndarray:
    """Soft brown-ish noise (stereo) at ROOM_TONE_LUFS: the air of a room, for a place whose bed is silent."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((2, count)).astype(np.float32)
    b, a = signal.butter(1, 400 / (rate / 2), "low")
    noise = signal.lfilter(b, a, noise, axis=1).astype(np.float32)
    level = loudness.integrated(noise, rate)
    return noise * np.float32(_gain_db(ROOM_TONE_LUFS - level)) if level > loudness.SILENT + 1 else noise


def _loop(source: np.ndarray, count: int, rate: int) -> np.ndarray:
    """`source` repeated to `count` samples with CROSSFADE-long equal-power joins."""
    if source.shape[1] >= count:
        return source[:, :count].copy()
    fade = min(int(CROSSFADE * rate), source.shape[1] // 3)
    if fade < 1:
        return np.pad(source, ((0, 0), (0, count - source.shape[1])), mode="wrap")
    angle = np.linspace(0.0, np.pi / 2, fade, dtype=np.float32)
    out = source.copy()
    while out.shape[1] < count:
        joined = out[:, -fade:] * np.cos(angle) + source[:, :fade] * np.sin(angle)
        out = np.concatenate([out[:, :-fade], joined, source[:, fade:]], axis=1)
    return out[:, :count]


def _envelope(voice: np.ndarray, rate: int, attack: float = 0.05, release: float = 0.4) -> np.ndarray:
    """0..1 per sample: 1 while the narrator speaks (within 30 dB of his loudest 10 ms), smoothed with the attack
    and release of a broadcast ducker."""
    count = voice.shape[1]
    block = max(1, int(0.01 * rate))
    blocks = count // block + (1 if count % block else 0)
    padded = np.pad(np.abs(voice).max(axis=0), (0, blocks * block - count))
    rms = np.sqrt((padded.reshape(blocks, block) ** 2).mean(axis=1))
    if not rms.size or rms.max() <= 1e-6:
        return np.zeros(count, dtype=np.float32)
    active = (rms > rms.max() * 10 ** (-30 / 20)).astype(np.float32)
    up, down = 1 - math.exp(-0.01 / attack), 1 - math.exp(-0.01 / release)
    smooth = np.zeros_like(active)
    level = 0.0
    for index, value in enumerate(active):
        level += (up if value > level else down) * (value - level)
        smooth[index] = level
    return np.repeat(smooth, block)[:count]


def _speech_span(voice: np.ndarray, rate: int) -> Optional[Tuple[float, float]]:
    """(first, last) second where a scene's speech is audible (-45 dBFS), None for silence."""
    if voice.size == 0:
        return None
    block = max(1, int(0.01 * rate))
    count = voice.shape[1] // block
    if not count:
        return None
    rms = np.sqrt((voice[:, :count * block].reshape(voice.shape[0], count, block) ** 2).mean(axis=(0, 2)))
    loud = np.nonzero(rms > 10 ** (-45 / 20))[0]
    return (loud[0] * 0.01, (loud[-1] + 1) * 0.01) if loud.size else None


def _true_peak_limit(samples: np.ndarray, rate: int, ceiling_db: float) -> np.ndarray:
    """A look-ahead limiter run at 4x the rate, so inter-sample peaks are caught (ffmpeg alimiter)."""
    over = signal.resample_poly(samples, 4, 1, axis=1).astype(np.float32)
    limited, _ = _audio.run_chain(over, rate * 4,
                                  f"alimiter=limit={_gain_db(ceiling_db):.5f}:attack=1:release=60:level=false:"
                                  "latency=true")
    limited = _audio._samples(limited, over.shape[1])
    return signal.resample_poly(limited, 1, 4, axis=1).astype(np.float32)[:, :samples.shape[1]]


def master(samples: np.ndarray, rate: int, report: Optional[MixReport] = None) -> np.ndarray:
    """The programme at TARGET_LUFS with its true peak at or under MASTER_TP (headroom for the AAC encode)."""
    level = loudness.integrated(samples, rate)
    if level <= loudness.SILENT + 1:
        return samples
    out = samples * np.float32(_gain_db(TARGET_LUFS - level))
    ceiling = MASTER_TP - 0.4
    for _ in range(4):
        peak = loudness.true_peak(out, rate)
        if peak <= MASTER_TP:
            break
        out = _true_peak_limit(out, rate, ceiling)
        ceiling -= 0.3
    # the limiter takes a little loudness: give back what the peaks allow
    level = loudness.integrated(out, rate)
    room = MASTER_TP - loudness.true_peak(out, rate)
    lift = min(TARGET_LUFS - level, room)
    if lift > 0.05:
        out = out * np.float32(_gain_db(lift))
    if report is not None:
        report.loudness = loudness.integrated(out, rate)
        report.true_peak = loudness.true_peak(out, rate)
    return out


def _bed_source(scenes: Sequence[SceneSound], rate: int, fallback) -> Tuple[Optional[np.ndarray], int]:
    """The first audible leveled background of a group of scenes (its core, without the handles) and its index."""
    handle = int(round(_audio.TIMELINE_AUDIO_HANDLE * rate))
    for index, scene in enumerate(scenes):
        if scene.background is None:
            continue
        try:
            raw, _ = _audio._leveled_background(Path(scene.background), scene.seconds, rate,
                                                Path(scene.effects) if scene.effects else None)
        except _audio.SilentBackground:
            continue
        except MediaError:
            continue
        core = raw[:, handle:raw.shape[1] - handle]
        if core.shape[1] and float(np.abs(core).max()) > 1e-4:
            return core, index
    if fallback is not None:
        try:
            raw = _audio._timeline_background(fallback(1, int(math.ceil(scenes[0].seconds))),
                                              int(math.ceil(scenes[0].seconds)), rate)
            return raw[:, handle:raw.shape[1] - handle], 0
        except MediaError:
            pass
    return None, -1


def episode_soundtrack(scenes: Sequence[SceneSound], rate: int = _audio.RATE, *, music: Optional[Path] = None,
                       background_fallback: Optional[Callable[[int, int], Path]] = None,
                       report: Optional[MixReport] = None) -> List[np.ndarray]:
    """The episode's soundtrack as one piece per scene (stereo, `rate`), see the module docstring."""
    report = report if report is not None else MixReport()
    counts = [int(round(scene.seconds * rate)) for scene in scenes]
    total = sum(counts)
    if not total:
        raise MediaError("The timeline has no scenes.")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)

    # --- narration: one track on the episode clock --------------------------------------------------------------
    voice = np.zeros((2, total), dtype=np.float32)
    previous_end: Optional[float] = None  # seconds on the episode clock where the previous scene's speech ended
    for number, (scene, start, count) in enumerate(zip(scenes, starts, counts)):
        if scene.voice is None:
            continue
        speech, _ = _audio.read(scene.voice, rate=rate)
        speech = _audio._samples(speech, count)
        if scene.voice_db:
            speech = speech * np.float32(_gain_db(scene.voice_db))
        span = _speech_span(speech, rate)
        if span is None:
            continue
        shift = 0.0
        if number and previous_end is not None:
            room = start / rate - previous_end - J_GAP
            shift = max(0.0, min(J_CUT, room, span[0] + J_CUT))
            shift = round(shift, 2) if shift >= 0.05 else 0.0
        offset = int(start - round(shift * rate))
        voice[:, offset:offset + count] += speech[:, :total - offset]
        if shift:
            report.j_cuts.append((number + 1, shift))
        previous_end = (offset / rate) + span[1]
    has_voice = bool(np.abs(voice).max() > 1e-5)
    if has_voice:
        level = loudness.integrated(voice, rate)
        voice *= np.float32(_gain_db(VOICE_LUFS - level))
        report.voice_loudness = loudness.integrated(voice, rate)

    # --- ambience: one bed per place ---------------------------------------------------------------------------
    groups: List[Tuple[str, int, int]] = []  # (location, first scene, last scene)
    for index, scene in enumerate(scenes):
        key = scene.location or f"scene{index}"
        if groups and groups[-1][0] == key:
            groups[-1] = (key, groups[-1][1], index)
        else:
            groups.append((key, index, index))
    bed = np.zeros((2, total), dtype=np.float32)
    fade = int(CROSSFADE * rate)
    for number, (key, first, last) in enumerate(groups):
        lo, hi = int(starts[first]), int(starts[last] + counts[last])
        lo_padded, hi_padded = max(0, lo - fade // 2), min(total, hi + fade // 2)
        source, _ = _bed_source(scenes[first:last + 1], rate, background_fallback)
        if source is None:
            piece = _room_tone(hi_padded - lo_padded, rate, seed=number + 7)
            report.room_tone.append(key)
        else:
            piece = _loop(source, hi_padded - lo_padded, rate)
            level = loudness.integrated(piece, rate)
            target = (report.voice_loudness or VOICE_LUFS) - BED_UNDER_VOICE
            if level > loudness.SILENT + 1:
                piece = piece * np.float32(_gain_db(min(30.0, target - level)))
        ramp = np.ones(piece.shape[1], dtype=np.float32)
        if lo_padded < lo:  # cross-fade with the place before
            width = lo - lo_padded + fade // 2
            ramp[:width] = np.sin(np.linspace(0, np.pi / 2, width, dtype=np.float32))
        if hi_padded > hi:
            width = hi_padded - hi + fade // 2
            ramp[-width:] = np.cos(np.linspace(0, np.pi / 2, width, dtype=np.float32))
        bed[:, lo_padded:hi_padded] += piece * ramp
        report.locations.append((key, round(lo / rate, 2), round(hi / rate, 2)))
    # the scenes' timed effects, on their own clocks, at their background levels
    for scene, start, count in zip(scenes, starts, counts):
        if scene.effects is None:
            continue
        try:
            layer = _audio._scene_effects(Path(scene.effects), None, int(math.ceil(scene.seconds)), rate)
        except MediaError:
            continue
        if layer is not None:
            layer = _audio._samples(layer, count)
            bed[:, start:start + count] += layer * np.float32(_gain_db(scene.background_db))
    # a soft room tone under every bed: a generated ambience may hold dead stretches, and dead air is never allowed
    bed += _room_tone(total, rate, seed=3)
    if bed.shape[1]:
        bed[:, :int(FADE_IN * rate)] *= np.linspace(0, 1, min(bed.shape[1], int(FADE_IN * rate)), dtype=np.float32)

    # --- music ---------------------------------------------------------------------------------------------------
    music_track = None
    if music is not None:
        try:
            samples, _ = _audio.read(music, rate=rate)
            if samples.shape[1] and float(np.abs(samples).max()) > 1e-4:
                music_track = _loop(samples, total, rate)
                level = loudness.integrated(music_track, rate)
                target = (report.voice_loudness or VOICE_LUFS) - MUSIC_UNDER_VOICE
                music_track = music_track * np.float32(_gain_db(target - level))
                ramp_in, ramp_out = min(total, 2 * rate), min(total, 3 * rate)
                music_track[:, :ramp_in] *= np.linspace(0, 1, ramp_in, dtype=np.float32)
                music_track[:, -ramp_out:] *= np.linspace(1, 0, ramp_out, dtype=np.float32)
                report.music = Path(music).name
        except MediaError as error:
            report.notes.append(f"The music could not be read ({error}).")
    if music is None:
        report.notes.append("No music: the owner's music library is empty.")

    # --- ducking and the mix --------------------------------------------------------------------------------------
    if has_voice:
        envelope = _envelope(voice, rate)
        bed_gain = 1.0 - (1.0 - _gain_db(-DUCK_BED_DB)) * envelope
        bed = bed * bed_gain.astype(np.float32)
        if music_track is not None:
            music_track = music_track * (1.0 - (1.0 - _gain_db(-DUCK_MUSIC_DB)) * envelope).astype(np.float32)
        speaking = envelope > 0.5
        if speaking.any():
            report.bed_loudness = loudness.integrated(bed[:, speaking], rate)
            report.bed_under_voice = (report.voice_loudness or 0.0) - report.bed_loudness
            if music_track is not None:
                report.music_loudness = loudness.integrated(music_track[:, speaking], rate)
    mix = bed + voice + (music_track if music_track is not None else 0.0)
    out_ramp = min(mix.shape[1], int(FADE_OUT * rate))
    mix[:, -out_ramp:] *= np.linspace(1, 0, out_ramp, dtype=np.float32)
    mix = master(mix.astype(np.float32), rate, report)
    return [mix[:, start:start + count] for start, count in zip(starts, counts)]
