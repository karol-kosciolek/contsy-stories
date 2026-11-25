# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Linear layers that keep their weight block-quantized (GGUF) and expand it for every call.

A 14B video model in Q6_K is 12 GB; expanded to bfloat16 it would be 28 GB. The layer keeps the
quantized bytes (on the CPU, pinned, or on the GPU) and dequantizes just the matrix it is about to
use, in the dtype of the input. `QLinear` is an `nn.Linear` (so LoRA hooks and shape checks see an
ordinary layer) whose `weight` is a placeholder on the meta device and takes no memory.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..formats.ggml import QTensor


class QLinear(nn.Linear):
    """`nn.Linear` with a GGUF-quantized weight: `qweight` holds the bytes, `bias` stays a dense parameter."""

    def __init__(self, qweight: QTensor, bias: Optional[Tensor]):
        out_features, in_features = qweight.shape
        super().__init__(in_features, out_features, bias=False, device="meta")
        self.qweight = qweight
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.qweight.to(dtype=x.dtype, device=x.device)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)

    def _apply(self, fn, recurse: bool = True):
        """Move the bytes and the bias; the placeholder weight stays on the meta device. The target device is what
        `fn` does to a tensor on the bytes' own device, so a dtype-only `.to(torch.bfloat16)` leaves them in place."""
        probe = fn(torch.empty(0, device=self.qweight.data.device))
        moved = self.qweight.data.to(probe.device)
        self.qweight = QTensor(self.qweight.name, self.qweight.qtype, self.qweight.shape, moved)
        if self.bias is not None:
            self.bias = nn.Parameter(fn(self.bias.data), requires_grad=False)
        return self

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, qtype={self.qweight.qtype.name}"


def materialize_quantized(model: nn.Module, tensors: Iterable[Tuple[str, Union[Tensor, QTensor]]],
                          device: torch.device | str, dtype: torch.dtype, what: str, *, optional=(), dtype_for=None) -> int:
    """Fill a meta-device model from (name, tensor-or-QTensor-or-ScaledFP8) pairs in the model's own key names.

    Plain tensors are copied to `device` and cast to `dtype` (floats) like `models.zimage.materialize`.
    A block-quantized 2-D weight of an `nn.Linear` replaces that layer by a `QLinear` that keeps the
    bytes on `device`, a per-channel scaled FP8 weight (`formats.fp8.ScaledFP8`) by an `FP8Linear`;
    any other quantized tensor is expanded once to `dtype`. The key set must match the model exactly.
    Returns the number of layers that stay quantized."""
    from ..formats.fp8 import ScaledFP8
    from .fp8_linear import FP8Linear

    quantized_layers = (QLinear, FP8Linear)
    expected = set(model.state_dict())
    seen, unexpected, quantized = set(), [], 0
    pending_bias = {}
    for name, tensor in tensors:
        if name not in expected:
            unexpected.append(name)
            continue
        owner_path, _, leaf = name.rpartition(".")
        owner = model.get_submodule(owner_path)
        current = owner._parameters.get(leaf, owner._buffers.get(leaf))
        want = (owner.out_features,) if current is None and isinstance(owner, quantized_layers) else tuple(current.shape)
        if tuple(tensor.shape) != want:
            raise ValueError(f"{what}: {name} has shape {tuple(tensor.shape)}, the model expects {want}")
        seen.add(name)
        if isinstance(tensor, QTensor) and tensor.quantized and leaf == "weight" and type(owner) is nn.Linear:
            parent_path, _, child = owner_path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            bias = pending_bias.pop(owner_path, None)
            layer = QLinear(QTensor(tensor.name, tensor.qtype, tensor.shape, tensor.data.to(device)), bias)
            setattr(parent, child, layer)
            quantized += 1
            continue
        if isinstance(tensor, ScaledFP8) and leaf == "weight" and type(owner) is nn.Linear:
            parent_path, _, child = owner_path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            layer = FP8Linear(tensor.weight.to(device), tensor.scale.to(device), pending_bias.pop(owner_path, None))
            setattr(parent, child, layer)
            quantized += 1
            continue
        storage_dtype = dtype_for(name, tensor) if dtype_for is not None else dtype
        value = tensor.to(dtype=storage_dtype, device=device) if isinstance(tensor, (QTensor, ScaledFP8)) else tensor.to(
            device=device, dtype=storage_dtype if tensor.is_floating_point() else None, copy=True)
        if isinstance(owner, quantized_layers) and leaf == "bias":
            owner.bias = nn.Parameter(value, requires_grad=False)
        elif type(owner) is nn.Linear and leaf == "bias" and f"{owner_path}.weight" not in seen:
            pending_bias[owner_path] = value  # the weight has not come yet: it may turn out to be quantized
            owner._parameters[leaf] = nn.Parameter(value, requires_grad=False)
        elif leaf in owner._parameters:
            owner._parameters[leaf] = nn.Parameter(value, requires_grad=False)
        else:
            owner._buffers[leaf] = value
    missing = expected - seen - set(optional)
    if unexpected or missing:
        raise KeyError(f"{what}: the file does not match the model; unexpected keys {sorted(unexpected)[:8]}, "
                       f"missing keys {sorted(missing)[:8]}")
    return quantized
