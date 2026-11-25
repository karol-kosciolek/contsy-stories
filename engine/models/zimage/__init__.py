# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Z-Image-Turbo adapter: tokenizer + Qwen3 text encoder (text.py), the S3-DiT transformer
(transformer.py) and the Flux autoencoder (vae.py).

Every network class comes from `transformers` / `diffusers`; this package owns the file loading,
the conventions of the model (each module's docstring names them and their source) and the glue between them.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn


def materialize(model: nn.Module, tensors: Iterable[Tuple[str, Tensor]], device: torch.device | str,
                dtype: Optional[torch.dtype], what: str, optional: Iterable[str] = (),
                dtype_for: Optional[Callable[[str, Tensor], Optional[torch.dtype]]] = None, copy: bool = True) -> None:
    """Fill a model that was built on the meta device, one tensor at a time.

    `tensors` yields (name, tensor) pairs already renamed to the model's state-dict keys. Each tensor
    is copied to `device` (and cast to `dtype` when it is a floating-point tensor) and put in place of
    its meta parameter, so the weights are never held twice in memory. The key set must match the
    model exactly, in both directions; otherwise a KeyError lists what is missing or unexpected.
    `optional` names keys that the file may leave out because the model shares them with another
    (the tied embedding of T5: `encoder.embed_tokens.weight` is `shared.weight`).
    `dtype_for(name, tensor)` may return another dtype for one tensor (None: use `dtype`), so that small tensors
    can be kept at the precision of the file instead of being rounded to `dtype` first.
    With `copy=False` a tensor already on `device` in its dtype is used as it is: a tensor that safetensors maps from
    its file then stays that mapping (read from the disk when first used, file cache instead of process memory).
    """
    expected = set(model.state_dict())
    seen = set()
    unexpected = []
    for name, tensor in tensors:
        if name not in expected:
            unexpected.append(name)
            continue
        owner_path, _, leaf = name.rpartition(".")
        owner = model.get_submodule(owner_path)
        if leaf in owner._parameters:
            current = owner._parameters[leaf]
            kind = "parameter"
        else:
            current = owner._buffers[leaf]
            kind = "buffer"
        if tuple(tensor.shape) != tuple(current.shape):
            raise ValueError(f"{what}: {name} has shape {tuple(tensor.shape)}, "
                             f"the model expects {tuple(current.shape)}")
        target = (dtype_for(name, tensor) if dtype_for is not None else None) or dtype
        value = tensor.to(device=device, dtype=target if tensor.is_floating_point() else None, copy=copy)
        if kind == "parameter":
            owner._parameters[leaf] = nn.Parameter(value, requires_grad=False)
        else:
            owner._buffers[leaf] = value
        seen.add(name)
    missing = expected - seen - set(optional)
    if unexpected or missing:
        raise KeyError(f"{what}: the file does not match the model; unexpected keys {sorted(unexpected)[:8]}, "
                       f"missing keys {sorted(missing)[:8]}")


def assert_loaded(model, what: str) -> None:
    """Fail now if any parameter or buffer is still on the meta device.

    Tensors that are not in the state dict (non-persistent buffers such as a rotary table) are not
    filled by `materialize`; a loader rebuilds the ones it knows, and calls this at the end so that a
    library upgrade which adds another one fails here instead of inside a forward pass."""
    from ...core.quant import QLinear

    placeholders = {f"{n}.weight" for n, m in model.named_modules() if isinstance(m, QLinear)}  # no memory on purpose
    on_meta = [n for n, t in [*model.named_parameters(), *model.named_buffers()]
               if t.is_meta and n not in placeholders]
    if on_meta:
        raise RuntimeError(f"{what}: {on_meta[:8]} are still on the meta device after loading")
