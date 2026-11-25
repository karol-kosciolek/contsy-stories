# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Exact frame-rate retiming: where the interpolated frames of a clip must sit on the output clock.

Wan renders 16 fps and the product delivers exactly 30 fps; lip sync works at 25 fps and is
brought back to 30 fps. Interpolating by a whole multiplier and then dropping frames makes the
clip drift or stutter, so every output frame gets its own timestamp i / target_fps instead.
"""
from __future__ import annotations

from typing import List, Tuple


def frame_rate_plan(frame_count: int, source_fps: int,
                    target_fps: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, int]]]:
    """RIFE timesteps on the output clock, retaining the final source endpoint.

    Returns `(tasks, layout)`. `tasks` lists the frames to interpolate as
    `(pair, fraction)`: between source frames `pair` and `pair + 1`, `fraction` of the way.
    `layout` has one entry per output frame: `(pair, -1)` is source frame `pair` itself,
    `(pair, k)` is the k-th interpolated frame of that pair (in task order)."""
    if frame_count < 1 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("Frame count and frame rates must be positive")
    count = ((frame_count - 1) * target_fps + source_fps - 1) // source_fps + 1
    tasks: List[Tuple[int, float]] = []
    layout: List[Tuple[int, int]] = []
    per_pair: dict = {}
    for index in range(count):
        pair, remainder = divmod(index * source_fps, target_fps)
        if remainder == 0 or pair >= frame_count - 1:
            layout.append((min(pair, frame_count - 1), -1))
        else:
            layout.append((pair, per_pair.get(pair, 0)))
            per_pair[pair] = per_pair.get(pair, 0) + 1
            tasks.append((pair, remainder / target_fps))
    return tasks, layout
