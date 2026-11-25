# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Build a network without drawing its random starting weights, when every weight comes from a checkpoint next.

A library that builds its model in Python and then loads the checkpoint into it (VoxCPM2) first fills every layer
with random numbers (`torch.nn.init`), only to overwrite them all: 5.7 s of CPU for VoxCPM2's 2.4 billion weights.
Inside `no_weight_init()` those fill functions leave the tensors as they are (uninitialised memory), so the caller
must load a checkpoint that covers every parameter, and check that it did (`load_state_dict` reports what is
missing). Buffers a module computes itself (rotary tables, masks) are made as always.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

# The random fills of torch.nn.init (the costly ones). Constant fills (zeros, ones) stay: they are cheap, and a
# buffer a module fills with them is not in any checkpoint.
_FILLS = ("uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_",
          "xavier_normal_", "orthogonal_", "sparse_")


@contextmanager
def no_weight_init() -> Iterator[None]:
    """Within the block, torch.nn.init's fills do nothing (see the module docstring). Not thread-safe against
    another thread building a model that relies on its random initialisation at the same moment: the engine builds
    one model at a time, under its lock."""
    init = torch.nn.init
    saved = {name: getattr(init, name) for name in _FILLS if hasattr(init, name)}

    def keep(tensor, *_args, **_kwargs):
        return tensor

    try:
        for name in saved:
            setattr(init, name, keep)
        yield
    finally:
        for name, function in saved.items():
            setattr(init, name, function)
