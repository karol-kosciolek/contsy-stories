# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Wan's rotary position embedding in one pass (Triton).

diffusers rotates every interleaved pair (x[2i], x[2i+1]) of a query or key by the angle of its token and channel:
`out[2i] = x[2i] cos[2i] - x[2i+1] sin[2i+1]`, `out[2i+1] = x[2i] sin[2i+1] + x[2i+1] cos[2i]`, in float32, rounded
once to the input's dtype. In PyTorch that is about eight passes over a 400 MB tensor (13 ms for each of the query
and the key of a 848x480, 97-frame clip on an RTX 4070 Ti Super); this kernel reads the tensor once and writes the
result once. The arithmetic is the same; the compiler may fuse a multiply and an add, so a value can differ from
PyTorch's in the last float32 bit before the rounding to bfloat16.
"""
from __future__ import annotations

from torch import Tensor

_KERNEL = None


def _build():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def rotate_kernel(X, COS, SIN, OUT, heads, tokens, stride_xb, stride_xl, stride_xh, stride_cl, stride_sl,
                      stride_ob, stride_ol, stride_oh, BLOCK_H: tl.constexpr, HALF: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)  # one token of one batch entry
        batch, token = row // tokens, row % tokens
        h = tl.arange(0, BLOCK_H)[:, None]
        pair = tl.arange(0, HALF)[None, :]
        inside = h < heads
        source = X + batch * stride_xb + token * stride_xl + h * stride_xh + 2 * pair
        even = tl.load(source, mask=inside, other=0.0).to(tl.float32)
        odd = tl.load(source + 1, mask=inside, other=0.0).to(tl.float32)
        cos = tl.load(COS + token * stride_cl + 2 * pair).to(tl.float32)
        sin = tl.load(SIN + token * stride_sl + 2 * pair + 1).to(tl.float32)
        target = OUT + batch * stride_ob + token * stride_ol + h * stride_oh + 2 * pair
        tl.store(target, (even * cos - odd * sin).to(OUT.dtype.element_ty), mask=inside)
        tl.store(target + 1, (even * sin + odd * cos).to(OUT.dtype.element_ty), mask=inside)

    _KERNEL = rotate_kernel
    return _KERNEL


def fits(hidden: Tensor, cos: Tensor, sin: Tensor) -> bool:
    """Inputs the kernel takes: [B, L, H, D] on CUDA with unit stride over D (D a power of two), and tables
    [1, L, 1, D] (any float dtype) on the same device."""
    if hidden.dim() != 4 or cos.dim() != 4 or not hidden.is_cuda:
        return False
    batch, tokens, heads, width = hidden.shape
    table = (1, tokens, 1, width)
    return (width >= 2 and width & (width - 1) == 0 and hidden.stride(-1) == 1 and tuple(cos.shape) == table
            and tuple(sin.shape) == table and cos.stride(-1) == 1 and sin.stride(-1) == 1
            and cos.device == hidden.device == sin.device)


def rotate(hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """`hidden` [B, L, H, D] rotated as diffusers' Wan attention does (see the module docstring), same dtype."""
    import torch
    import triton

    batch, tokens, heads, width = hidden.shape
    out = torch.empty_like(hidden, memory_format=torch.contiguous_format)
    if hidden.numel():
        with torch.cuda.device(hidden.device):
            _build()[(batch * tokens,)](hidden, cos, sin, out, heads, tokens, hidden.stride(0), hidden.stride(1),
                                        hidden.stride(2), cos.stride(1), sin.stride(1), out.stride(0),
                                        out.stride(1), out.stride(2), BLOCK_H=triton.next_power_of_2(heads),
                                        HALF=width // 2, num_warps=4)
    return out
