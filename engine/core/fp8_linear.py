# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Linear layers whose weight stays in FP8 (float8_e4m3fn) with one scale per output channel.

This is the layout of LightX2V's scaled FP8 checkpoints (`X.weight` float8_e4m3fn [out, in], `X.weight_scale`
[out, 1]; the real weight is `weight * weight_scale`, row by row). On a GPU with FP8 tensor cores (compute
capability 8.9 and later) the layer multiplies in FP8 the way LightX2V runs these files: the activations are
quantized per token (one scale per row, `kernels.fp8.quantize_rows`), `torch._scaled_mm` multiplies the FP8
matrices, and the row and channel scales and the bias are applied to the product (`kernels.fp8.scale_output_`).
Anywhere else (an older GPU, the CPU, float32 or float16 inputs) the weight is expanded to the input's dtype for the
call and an ordinary `F.linear` runs: the same result up to the rounding of the activations, only slower. float16
stays out of the FP8 product on purpose: `_scaled_mm` runs with unit scales, and its unscaled sums of e4m3 products
(each up to 448 x 448, summed over thousands of inputs) overflow float16's range before the scales bring them back.

`FP8Linear` is an `nn.Linear` (shape checks and module walks see an ordinary layer). Its weight is a real
parameter in float8, so block streaming moves it like any other tensor, half the bytes of bfloat16; moving the
layer never changes the dtype of the weight or its scale.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..kernels import fp8 as kernels

FP8 = kernels.FP8
COMPUTE_DTYPES = (torch.bfloat16,)  # the FP8 product's output dtype: wide enough for its unscaled sums


@lru_cache(maxsize=None)
def _fp8_capable(index: int) -> bool:
    return torch.cuda.get_device_capability(index) >= (8, 9) and hasattr(torch, "_scaled_mm")


def fp8_matmul_available(device: torch.device | str) -> bool:
    """Whether `torch._scaled_mm` can multiply FP8 matrices on `device` (CUDA, compute capability 8.9 or later)."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return _fp8_capable(device.index if device.index is not None else torch.cuda.current_device())


_ONES = {}


def _one(device: torch.device) -> Tensor:
    one = _ONES.get(device)
    if one is None:
        one = _ONES[device] = torch.ones((), dtype=torch.float32, device=device)
    return one


class FP8Linear(nn.Linear):
    """`nn.Linear` with an FP8 weight [out, in] and a float32 scale per output channel [out]."""

    def __init__(self, weight: Tensor, scale: Tensor, bias: Optional[Tensor] = None):
        if weight.dtype != FP8 or weight.dim() != 2:
            raise ValueError(f"FP8Linear needs a 2-D float8_e4m3fn weight, got {weight.dtype} {tuple(weight.shape)}")
        out_features, in_features = weight.shape
        if scale.numel() != out_features:
            raise ValueError(f"the weight has {out_features} output channels but {scale.numel()} scales")
        super().__init__(in_features, out_features, bias=False, device="meta")
        self.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
        self.register_buffer("weight_scale", scale.detach().reshape(out_features).to(torch.float32))
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def dequantized(self, dtype: torch.dtype) -> Tensor:
        """The real weight `weight * scale` in `dtype` (float32 arithmetic, one rounding)."""
        return (self.weight.to(torch.float32) * self.weight_scale[:, None]).to(dtype)

    def uses_fp8(self, x: Tensor) -> bool:
        return (x.dtype in COMPUTE_DTYPES and x.device == self.weight.device and fp8_matmul_available(x.device)
                and self.in_features % 16 == 0 and self.out_features % 16 == 0)

    def forward(self, x: Tensor) -> Tensor:
        if not self.uses_fp8(x):
            bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
            return F.linear(x, self.dequantized(x.dtype).to(x.device), bias)
        quantized, row_scale = kernels.quantize_rows(x)
        return self.multiply(quantized, row_scale, x.dtype).reshape(*x.shape[:-1], self.out_features)

    def multiply(self, quantized: Tensor, row_scale: Tensor, dtype: torch.dtype,
                 activation: Optional[str] = None) -> Tensor:
        """The layer's output [M, out] for activations already quantized by `kernels.fp8.quantize_rows`, with the
        activation of `kernels.fp8.scale_output_` (None, or "gelu_tanh") applied in the same pass."""
        one = _one(quantized.device)
        out = torch._scaled_mm(quantized, self.weight.t(), scale_a=one, scale_b=one, out_dtype=dtype)
        return kernels.scale_output_(out, row_scale, self.weight_scale, self.bias, activation=activation)

    def _apply(self, fn, recurse: bool = True):
        """Move the weight, its scale and the bias to another device, never to another dtype. The target device is
        what `fn` does to a tensor on the weight's own device: a dtype-only `.to(torch.bfloat16)` keeps it there."""
        device = fn(torch.empty(0, device=self.weight.device)).device
        self.weight = nn.Parameter(self.weight.data.to(device), requires_grad=False)
        self.weight_scale = self.weight_scale.to(device)
        if self.bias is not None:
            self.bias = nn.Parameter(fn(self.bias.data), requires_grad=False)
        return self

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, weight=float8_e4m3fn"


def unhooked_fp8(layer: nn.Module, x: Tensor) -> bool:
    """Whether `layer` is an FP8Linear that multiplies `x` in FP8 and has no hooks (no LoRA attached)."""
    return (type(layer) is FP8Linear and layer.uses_fp8(x) and not layer._forward_hooks
            and not layer._forward_pre_hooks)


def linear_gelu(layer: nn.Module, x: Tensor) -> Tensor:
    """`F.gelu(layer(x), approximate="tanh")`, the GELU in the FP8 product's output pass when `layer` is an unhooked
    FP8Linear running in FP8 (`kernels.fp8.scale_output_`)."""
    if not unhooked_fp8(layer, x):
        return F.gelu(layer(x), approximate="tanh")
    quantized, row_scale = kernels.quantize_rows(x)
    return layer.multiply(quantized, row_scale, x.dtype, "gelu_tanh").reshape(*x.shape[:-1], layer.out_features)


def project(x: Tensor, layers: Sequence[nn.Module]) -> list:
    """`[layer(x) for layer in layers]`, quantizing `x` once when every layer is an FP8Linear running in FP8 with no
    hooks (the query, key and value projections of one input). Hooked layers (a LoRA) are called normally."""
    fast = all(unhooked_fp8(layer, x) for layer in layers)
    if not fast or len(layers) < 2:
        return [layer(x) for layer in layers]
    quantized, row_scale = kernels.quantize_rows(x)
    return [layer.multiply(quantized, row_scale, x.dtype).reshape(*x.shape[:-1], layer.out_features)
            for layer in layers]
