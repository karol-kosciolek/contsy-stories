# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""libavfilter graphs inside the process.

The FFmpeg libraries ship inside the PyAV wheel, so the engine runs the filters the ffmpeg program
used to run (loudnorm, sidechaincompress, xfade, ...) without starting a program. PyAV builds a graph
node by node and cannot parse ffmpeg's filter text, so `add_chain` takes the familiar
"name=args,name=args" form for a linear stretch; callers link the nodes with several inputs themselves.
"""
from __future__ import annotations

import contextlib
import gc
from typing import Iterator, List, Optional, Tuple

import av
import av.logging
import numpy as np


def split_chain(text: str) -> List[Tuple[str, Optional[str]]]:
    """`"highpass=f=40:poles=2,apad"` -> `[("highpass", "f=40:poles=2"), ("apad", None)]`.

    A comma inside single quotes belongs to the filter's arguments (expressions use them)."""
    items, current, quoted = [], [], False
    for char in text:
        if char == "'":
            quoted = not quoted
        if char == "," and not quoted:
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    items.append("".join(current))
    chain = []
    for item in items:
        name, _, args = item.strip().partition("=")
        if not name:
            raise ValueError(f"empty filter in {text!r}")
        chain.append((name, args or None))
    return chain


def add_chain(graph: av.filter.Graph, node, text: str):
    """Append the filters of `text` after `node`; returns the last node."""
    for name, args in split_chain(text):
        following = graph.add(name, args)
        node.link_to(following)
        node = following
    return node


def pull_available(sink) -> Iterator[av.frame.Frame]:
    """Every frame the sink can deliver right now (it stops when the graph wants more input or ended)."""
    while True:
        try:
            yield sink.pull()
        except (av.error.BlockingIOError, av.error.EOFError):
            return


def push(source, frame) -> bool:
    """Feed a frame (None ends the input). False once the graph wants no more from this source:
    a trim downstream has reached its end, so further frames would only be refused."""
    try:
        source.push(frame)
        return True
    except av.error.EOFError:
        return False


def audio_frame(samples: np.ndarray, rate: int, position: int = 0) -> av.AudioFrame:
    """Planar float frame from a (channels, samples) array; `position` is its first sample's index."""
    from fractions import Fraction

    layout = {1: "mono", 2: "stereo"}.get(samples.shape[0])
    if layout is None:
        raise ValueError(f"only mono and stereo audio is supported, got {samples.shape[0]} channels")
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples, dtype=np.float32), format="fltp", layout=layout)
    frame.sample_rate, frame.pts, frame.time_base = rate, position, Fraction(1, rate)
    return frame


@contextlib.contextmanager
def libav_messages() -> Iterator[list]:
    """Collect libav log messages as (level, name, text). Some filters report their measurements this way,
    and only when the graph is released, so call `release` before leaving the block. Identical
    consecutive messages are normally folded into one, which would hide a second equal measurement."""
    previous_level, previous_skip = av.logging.get_level(), av.logging.get_skip_repeated()
    av.logging.set_level(av.logging.INFO)
    av.logging.set_skip_repeated(False)
    try:
        with av.logging.Capture() as lines:
            yield lines
    finally:
        av.logging.set_level(previous_level)
        av.logging.set_skip_repeated(previous_skip)


def release() -> None:
    """Collect dropped filter graphs now. A graph holds reference cycles, and filters such as loudnorm
    report their measurements only when they are freed, so drop every reference first, then call this."""
    gc.collect()
