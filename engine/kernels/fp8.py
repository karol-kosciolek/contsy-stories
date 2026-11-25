# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The two passes around an FP8 matrix product with per-row and per-column scales.

`torch._scaled_mm` runs the FP8 tensor cores, but on Ada (sm89) PyTorch only offers it with one scale for a whole
matrix, while accuracy needs one scale per row of the activations (per token) and one per output channel of the
weight (what LightX2V's scaled FP8 files store). So the product runs with unit scales and the scales are applied
around it:

* `quantize_rows(x)`: every row of `x` scaled so that its largest magnitude is 448 (the largest float8_e4m3fn
  value), rounded to the nearest FP8 value, and the scale `amax / 448` of every row, in one pass over `x`;
* `scale_output_(out, row_scale, col_scale, bias)`: `out * row_scale[:, None] * col_scale[None, :] + bias`, in
  place, in one pass (float32 arithmetic, one rounding to the output dtype); with `activation="gelu_tanh"` the same
  pass also applies the tanh-approximated GELU that follows Wan's first feed-forward layer, to the rounded value as
  `F.gelu(..., approximate="tanh")` would (float32, libdevice's tanh, one more rounding), which saves a pass over
  the widest activation of the block (tokens x 13824).

Each has a Triton kernel and a PyTorch fallback that computes the same values (used without Triton, e.g. on the CPU
in tests); the fallback of `scale_output_` rounds after each operation, so it can differ in the last bfloat16 bit.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from . import available

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
TINY = 1e-12  # the scale of an all-zero row: the row quantizes to zeros instead of dividing by zero
_KERNELS = None


def _build():
    """Compile-on-first-use kernels, created only when Triton is imported (it is optional)."""
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def quantize_rows_kernel(X, Q, S, K, stride_x, stride_q, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, BLOCK)
        inside = offsets < K
        x = tl.load(X + row * stride_x + offsets, mask=inside, other=0.0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)
        scale = amax / 448.0
        q = tl.clamp(tl.math.div_rn(x, scale), -448.0, 448.0)  # IEEE division, as PyTorch divides
        tl.store(Q + row * stride_q + offsets, q.to(tl.float8e4nv, fp_downcast_rounding="rtne"), mask=inside)
        tl.store(S + row, scale)

    from triton.language.extra import libdevice

    @triton.jit
    def scale_output_kernel(O, RS, CS, B, N, stride_o, BLOCK: tl.constexpr, HAS_BIAS: tl.constexpr,
                            GELU_TANH: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        inside = offsets < N
        out = tl.load(O + row * stride_o + offsets, mask=inside, other=0.0).to(tl.float32)
        out = out * tl.load(RS + row) * tl.load(CS + offsets, mask=inside, other=0.0)
        if HAS_BIAS:
            out += tl.load(B + offsets, mask=inside, other=0.0).to(tl.float32)
        if GELU_TANH:  # PyTorch's formula on the rounded value: 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))
            x = out.to(O.dtype.element_ty).to(tl.float32)
            inner = 0.7978845608028654 * (x + 0.044715 * (x * x * x))
            out = 0.5 * x * (1.0 + libdevice.tanh(inner))
        tl.store(O + row * stride_o + offsets, out.to(O.dtype.element_ty), mask=inside)

    _KERNELS = quantize_rows_kernel, scale_output_kernel
    return _KERNELS


def _rows(x: Tensor) -> Tensor:
    x = x.reshape(-1, x.shape[-1])
    return x if x.stride(-1) == 1 else x.contiguous()


def quantize_rows(x: Tensor, use_triton: Optional[bool] = None) -> Tuple[Tensor, Tensor]:
    """`x` [..., K] as FP8 rows [M, K] and their float32 scales [M] (M = all leading dims), `x ~ q * scale`."""
    rows = _rows(x)
    if use_triton is None:
        use_triton = rows.is_cuda and available(rows.device)
    if not use_triton:
        amax = rows.abs().amax(dim=1).float().clamp_min(TINY)
        scale = amax / FP8_MAX
        quantized = (rows.float() / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        return quantized, scale
    import triton

    count, width = rows.shape
    quantized = torch.empty((count, width), dtype=FP8, device=rows.device)
    scale = torch.empty((count,), dtype=torch.float32, device=rows.device)
    if count:
        block = triton.next_power_of_2(width)
        kernel = _build()[0]
        kernel[(count,)](rows, quantized, scale, width, rows.stride(0), quantized.stride(0), BLOCK=block,
                         num_warps=16 if block > 8192 else 8 if block > 2048 else 4)
    return quantized, scale


ACTIVATIONS = (None, "gelu_tanh")


def scale_output_(out: Tensor, row_scale: Tensor, col_scale: Tensor, bias: Optional[Tensor] = None,
                  use_triton: Optional[bool] = None, activation: Optional[str] = None) -> Tensor:
    """In place: `out[m, n] = out[m, n] * row_scale[m] * col_scale[n] + bias[n]` for a row-major `out` [M, N], then
    `activation` (None, or "gelu_tanh": `F.gelu(out, approximate="tanh")`)."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}, not {activation!r}")
    if use_triton is None:
        use_triton = out.is_cuda and available(out.device)
    if not use_triton:
        out.mul_(row_scale[:, None].to(out.dtype)).mul_(col_scale[None, :].to(out.dtype))
        if bias is not None:
            out.add_(bias.to(out.dtype))
        if activation == "gelu_tanh":
            out.copy_(F.gelu(out, approximate="tanh"))
        return out
    import triton

    count, width = out.shape
    if count:
        block = 1024
        kernel = _build()[1]
        kernel[(count, triton.cdiv(width, block))](out, row_scale, col_scale, row_scale if bias is None else bias,
                                                   width, out.stride(0), BLOCK=block, HAS_BIAS=bias is not None,
                                                   GELU_TANH=activation == "gelu_tanh", num_warps=4)
    return out
