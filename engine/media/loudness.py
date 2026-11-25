# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Loudness and level measurements (ITU-R BS.1770-4 / EBU R 128) on float arrays, numpy and scipy only.

- integrated(samples, rate): gated integrated loudness in LUFS (400 ms blocks, 75 % overlap, -70 LUFS absolute and
  -10 LU relative gates), with the K-weighting filter of BS.1770 (its coefficients worked out for any sample rate),
- true_peak(samples, rate): the highest inter-sample peak in dBTP (4x oversampling, as BS.1770-4 Annex 2),
- short_term(samples, rate): the 3 s loudness every 100 ms,
- silences(samples, rate): stretches quieter than a level (digital or near-digital silence),
- clipped(samples): the share of samples at full scale.

Samples are float32 arrays shaped (channels, n), as everywhere in engine/media/audio.py.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy import signal

ABSOLUTE_GATE = -70.0
RELATIVE_GATE = -10.0
SILENT = -120.0


def _k_weighting(rate: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """The two biquads of the K-weighting (a high shelf, then a high pass) for any sample rate (the BS.1770
    coefficients are given for 48 kHz; these are the analog prototypes they come from, as in libebur128)."""
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = 10 ** (gain_db / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf_b = np.array([(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0])
    shelf_a = np.array([1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1 + k / q + k * k
    pass_b = np.array([1.0, -2.0, 1.0])
    pass_a = np.array([1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])
    return (shelf_b, shelf_a), (pass_b, pass_a)


def _weighted_power(samples: np.ndarray, rate: int) -> np.ndarray:
    """Mean square of the K-weighted signal per sample, summed over channels (front channels, weight 1)."""
    (b1, a1), (b2, a2) = _k_weighting(rate)
    data = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    weighted = signal.lfilter(b2, a2, signal.lfilter(b1, a1, data, axis=1), axis=1)
    return (weighted ** 2).sum(axis=0)


def _blocks(power: np.ndarray, rate: int, block: float, step: float) -> np.ndarray:
    size, hop = int(round(block * rate)), int(round(step * rate))
    if power.size < size:
        return np.array([power.mean()]) if power.size else np.zeros(0)
    cumulative = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, power.size - size + 1, hop)
    return (cumulative[starts + size] - cumulative[starts]) / size


def _lufs(power) -> float:
    return -0.691 + 10 * math.log10(power) if power > 0 else SILENT


def integrated(samples: np.ndarray, rate: int) -> float:
    """Gated integrated loudness, LUFS (SILENT for silence)."""
    blocks = _blocks(_weighted_power(samples, rate), rate, 0.4, 0.1)
    if not blocks.size:
        return SILENT
    loud = np.array([_lufs(p) for p in blocks])
    kept = blocks[loud > ABSOLUTE_GATE]
    if not kept.size:
        return SILENT
    relative = _lufs(kept.mean()) + RELATIVE_GATE
    kept = blocks[(loud > ABSOLUTE_GATE) & (loud > relative)]
    return _lufs(kept.mean()) if kept.size else SILENT


def short_term(samples: np.ndarray, rate: int) -> np.ndarray:
    """The 3 s loudness every 100 ms (LUFS)."""
    blocks = _blocks(_weighted_power(samples, rate), rate, 3.0, 0.1)
    return np.array([_lufs(p) for p in blocks])


def momentary(samples: np.ndarray, rate: int) -> np.ndarray:
    """The 400 ms loudness every 100 ms (LUFS)."""
    blocks = _blocks(_weighted_power(samples, rate), rate, 0.4, 0.1)
    return np.array([_lufs(p) for p in blocks])


def true_peak(samples: np.ndarray, rate: int) -> float:
    """The highest inter-sample peak in dBTP (4x oversampled below 96 kHz)."""
    data = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    if not data.size:
        return SILENT
    factor = 4 if rate < 96000 else 2
    peak = 0.0
    for channel in data:
        over = signal.resample_poly(channel, factor, 1)
        peak = max(peak, float(np.max(np.abs(over))) if over.size else 0.0, float(np.max(np.abs(channel))))
    return 20 * math.log10(peak) if peak > 0 else SILENT


def sample_peak(samples: np.ndarray) -> float:
    data = np.asarray(samples)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    return 20 * math.log10(peak) if peak > 0 else SILENT


def clipped(samples: np.ndarray, level: float = 0.999) -> float:
    """The share of samples at (or over) full scale."""
    data = np.asarray(samples)
    return float((np.abs(data) >= level).mean()) if data.size else 0.0


def silences(samples: np.ndarray, rate: int, *, below_db: float = -60.0, longest: float = 0.4,
             window: float = 0.05) -> List[Tuple[float, float]]:
    """(start, end) seconds of every stretch longer than `longest` whose 50 ms RMS (all channels) stays under
    `below_db` dBFS: dead air in a film."""
    data = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    size = max(1, int(round(window * rate)))
    count = data.shape[1] // size
    if not count:
        return []
    pieces = data[:, :count * size].reshape(data.shape[0], count, size)
    rms = np.sqrt((pieces ** 2).mean(axis=(0, 2)))
    quiet = 20 * np.log10(np.maximum(rms, 1e-12)) < below_db
    found, start = [], None
    for index, flag in enumerate(np.append(quiet, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if (index - start) * window > longest:
                found.append((round(start * window, 2), round(index * window, 2)))
            start = None
    return found


def gain_to(samples: np.ndarray, rate: int, target: float) -> float:
    """The gain in dB that brings the samples' integrated loudness to `target` LUFS (0 for silence)."""
    level = integrated(samples, rate)
    return 0.0 if level <= SILENT + 1 else target - level
