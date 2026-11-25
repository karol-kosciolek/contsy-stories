# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Stage timing and memory release shared by the pipelines."""
from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import torch

StatsCb = Callable[[str, float, int], None]  # (stage, seconds, peak CUDA bytes during the stage; 0 on CPU)


@contextmanager
def stage(name: str, device: torch.device, stats: Optional[StatsCb]) -> Iterator[None]:
    """Time a stage and report the CUDA memory peak reached inside it to `stats`. Nothing is measured, and the GPU
    is not synchronised, when nobody listens."""
    measure = stats is not None and device.type == "cuda"
    if measure:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        if stats is not None:
            if measure:
                torch.cuda.synchronize(device)
            stats(name, time.perf_counter() - start, torch.cuda.max_memory_allocated(device) if measure else 0)


def release(device: torch.device) -> None:
    """Free what dropped models and activations held: collect reference cycles, then empty the CUDA cache."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
