# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""GPU kernels written in Triton: Contsy's FP8 helpers (`fp8.py`), rotary embedding (`rope.py`) and fused residual
adds, layer norms and VAE norms (`norms.py`), and the SageAttention kernels (`sage/`, vendored unchanged from the
authors' Apache-2.0 repository, see THIRD_PARTY_NOTICES.md).

Triton is optional. Without it (no `triton` package, no CUDA, a CPU run) `available()` is False and every caller
takes its PyTorch path, which computes the same thing more slowly. Triton compiles a kernel the first time it runs
for a new specialization and keeps the binary in its cache folder; `use_cache_dir` points that folder at one that
lasts across runs, so only the very first run of an installation pays for the compilation.
"""
from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

CACHE_ENV = "TRITON_CACHE_DIR"


def use_cache_dir(folder: str | Path) -> Path:
    """Keep Triton's compiled kernels in `folder` unless the user chose a folder with TRITON_CACHE_DIR already.

    Must run before the first kernel compiles (Triton reads the variable then, and makes the folder when it first
    writes to it). Returns the folder in use."""
    chosen = os.environ.get(CACHE_ENV)
    if chosen:
        return Path(chosen)
    os.environ[CACHE_ENV] = str(folder)
    return Path(folder)


@lru_cache(maxsize=None)
def _importable() -> bool:
    if importlib.util.find_spec("triton") is None:
        return False
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception:  # noqa: BLE001 - a broken install (missing compiler, DLL) counts as no Triton
        return False
    return True


def available(device=None) -> bool:
    """Whether Triton kernels can run on `device` (default: the current CUDA device)."""
    import torch

    if not torch.cuda.is_available() or not _importable():
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    return True
