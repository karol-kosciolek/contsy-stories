# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The residual adds and layer norms between the attention and feed-forward parts of a Wan block, in one pass.

diffusers' `WanTransformerBlock` (0.42, Apache-2.0) works on the hidden state `x` (bfloat16 [tokens, 5120] for Wan 2.2
A14B) with a few chains of elementwise PyTorch operations, each a separate pass over hundreds of megabytes:

* the modulated layer norm `(layer_norm(x.float()) * (1 + scale) + shift).type_as(x)` before the self-attention and
  before the feed-forward part (`scale`, `shift`: float32 [dim] from the time embedding);
* the gated residual add `(x.float() + y * gate).type_as(x)` after the self-attention and the feed-forward part;
* the plain residual add `x + y` after the cross-attention, and the affine layer norm
  `layer_norm(x.float(), weight, bias).type_as(x)` before it.

`add_norm` computes one residual add and the layer norm that follows it in a single Triton pass: it reads `x` and `y`
once, writes the new hidden state `h` (rounded to its dtype, as the PyTorch chain rounds it) and the normalised `n`
(float32 arithmetic from the rounded `h`, one rounding at the end). That is the same arithmetic as the PyTorch chain;
the kernel sums the row in another order and the compiler may fuse a multiply and an add, so a value can differ from
PyTorch's in the last float32 bits before the rounding (measured: about one bfloat16 value in a thousand moves by one
step, see tests/test_engine_kernels_gpu.py). The PyTorch fallback (`use_triton=False`, the CPU, other dtypes) is the
chain of diffusers itself.

`rms_channels` is the same idea for the Wan VAE: its `WanRMS_norm` (`F.normalize` over the channels of a [B, C, ...]
tensor in float32, rounded, times sqrt(C), rounded, times a per-channel gamma, rounded) and the SiLU that follows it in
every residual block and at the head of the encoder and the decoder, in one Triton pass (two reads of a tile, the second
from the L2 cache) instead of about eight passes over full-resolution video features. It reproduces every rounding of
the PyTorch chain; only the sum of squares is added up in another order.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from . import available

NONE, AFFINE, MODULATED = 0, 1, 2  # the layer norm after the add: none, weight and bias, (1 + scale) and shift
_KERNEL = None


def _build():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def add_norm_kernel(X, Y, GATE, A, B, H, N, width, eps, stride_x, stride_y, stride_h, stride_n,
                        BLOCK: tl.constexpr, HAS_Y: tl.constexpr, HAS_GATE: tl.constexpr, NORM: tl.constexpr,
                        HAS_N: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        inside = cols < width
        x = tl.load(X + row * stride_x + cols, mask=inside, other=0.0).to(tl.float32)
        if HAS_Y:
            y = tl.load(Y + row * stride_y + cols, mask=inside, other=0.0).to(tl.float32)
            if HAS_GATE:
                x = x + y * tl.load(GATE + cols, mask=inside, other=0.0).to(tl.float32)
            else:
                x = x + y
            h = x.to(H.dtype.element_ty)  # the PyTorch chain rounds the new hidden state before its norm
            tl.store(H + row * stride_h + cols, h, mask=inside)
            x = h.to(tl.float32)
        if HAS_N:
            mean = tl.sum(x, axis=0) / width
            centred = tl.where(inside, x - mean, 0.0)
            variance = tl.sum(centred * centred, axis=0) / width
            normed = centred * tl.math.rsqrt(variance + eps)
            if NORM == 1:
                normed = normed * tl.load(A + cols, mask=inside, other=0.0).to(tl.float32) \
                    + tl.load(B + cols, mask=inside, other=0.0).to(tl.float32)
            elif NORM == 2:
                normed = normed * (1.0 + tl.load(A + cols, mask=inside, other=0.0).to(tl.float32)) \
                    + tl.load(B + cols, mask=inside, other=0.0).to(tl.float32)
            tl.store(N + row * stride_n + cols, normed.to(N.dtype.element_ty), mask=inside)

    _KERNEL = add_norm_kernel
    return _KERNEL


_RMS_KERNEL = None


def _build_rms():
    global _RMS_KERNEL
    if _RMS_KERNEL is not None:
        return _RMS_KERNEL
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def rms_channels_kernel(X, G, Y, channels, positions, scale, stride_xb, stride_xc, stride_xp, stride_yb,
                            stride_yc, stride_yp, BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr, SILU: tl.constexpr):
        batch = tl.program_id(0).to(tl.int64)
        p = tl.program_id(1).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
        p_in = p < positions
        source = X + batch * stride_xb + p[None, :] * stride_xp
        squares = tl.zeros([BLOCK_P], dtype=tl.float32)
        for start in range(0, channels, BLOCK_C):
            c = (start + tl.arange(0, BLOCK_C)).to(tl.int64)
            inside = (c[:, None] < channels) & p_in[None, :]
            x = tl.load(source + c[:, None] * stride_xc, mask=inside, other=0.0).to(tl.float32)
            squares += tl.sum(x * x, axis=0)
        norm = tl.maximum(tl.sqrt_rn(squares), 1e-12)  # F.normalize: x / max(||x||, eps)
        target = Y + batch * stride_yb + p[None, :] * stride_yp
        for start in range(0, channels, BLOCK_C):
            c = (start + tl.arange(0, BLOCK_C)).to(tl.int64)
            inside = (c[:, None] < channels) & p_in[None, :]
            x = tl.load(source + c[:, None] * stride_xc, mask=inside, other=0.0).to(tl.float32)
            n = tl.math.div_rn(x, norm[None, :]).to(Y.dtype.element_ty).to(tl.float32)
            n = (n * scale).to(Y.dtype.element_ty).to(tl.float32)
            gamma = tl.load(G + c, mask=c < channels, other=0.0).to(tl.float32)
            n = (n * gamma[:, None]).to(Y.dtype.element_ty).to(tl.float32)
            if SILU:  # PyTorch: x / (1 + exp(-x)) in float32
                n = tl.math.div_rn(n, 1.0 + libdevice.exp(-n))
            tl.store(target + c[:, None] * stride_yc, n.to(Y.dtype.element_ty), mask=inside)

    _RMS_KERNEL = rms_channels_kernel
    return _RMS_KERNEL


def _rms_reference(x: Tensor, gamma: Tensor, scale: float, silu: bool) -> Tensor:
    """diffusers' WanRMS_norm (channel first, no bias) on a 16-bit tensor, then the SiLU."""
    out = F.normalize(x.float(), dim=1).to(x.dtype) * scale * gamma + 0.0
    return F.silu(out) if silu else out


def _flat_positions(x: Tensor) -> Optional[int]:
    """The one stride of the positions of a [B, C, *positions] tensor when they can be walked with it (contiguous or
    channels-last memory), else None."""
    stride = x.stride(-1)
    expected = stride
    for size, step in zip(reversed(x.shape[2:]), reversed(x.stride()[2:])):
        if size != 1 and step != expected:
            return None
        expected *= size
    return stride


def rms_channels_fits(x: Tensor, gamma: Tensor) -> bool:
    return (x.is_cuda and x.dim() >= 3 and x.dtype in (torch.bfloat16, torch.float16) and gamma.device == x.device
            and gamma.numel() == x.shape[1] and _flat_positions(x) is not None and available(x.device))


def rms_channels(x: Tensor, gamma: Tensor, scale: float, silu: bool = False, use_triton: Optional[bool] = None) -> Tensor:
    """diffusers' `WanRMS_norm` (channel first, no bias) of `x` [B, C, ...] with `gamma` [C] (any shape of C values)
    and `scale` (sqrt(C)), then `F.silu` when `silu`, in the dtype of `x` and in its memory format."""
    if use_triton is None:
        use_triton = rms_channels_fits(x, gamma)
    if not use_triton:
        return _rms_reference(x, gamma, scale, silu)
    import triton

    out = torch.empty_like(x)
    batch, channels = x.shape[0], x.shape[1]
    positions = x[0, 0].numel()
    if x.numel():
        gamma = gamma.reshape(-1)
        block_p = 128
        with torch.cuda.device(x.device):
            _build_rms()[(batch, triton.cdiv(positions, block_p))](
                x, gamma if gamma.stride(0) == 1 else gamma.contiguous(), out, channels, positions, float(scale),
                x.stride(0), x.stride(1), _flat_positions(x), out.stride(0), out.stride(1), _flat_positions(out),
                BLOCK_C=32, BLOCK_P=block_p, SILU=silu, num_warps=4)
    return out


def _vector(value: Optional[Tensor], width: int) -> Optional[Tensor]:
    """A per-channel vector ([dim], [1, dim] or [1, 1, dim]) as a contiguous [dim] view."""
    if value is None:
        return None
    flat = value.reshape(-1)
    if flat.numel() != width:
        raise ValueError(f"a per-channel vector of {width} values was expected, got shape {tuple(value.shape)}")
    return flat if flat.stride(0) == 1 else flat.contiguous()


def _reference(x: Tensor, y: Optional[Tensor], gate: Optional[Tensor], norm: int, a: Optional[Tensor],
               b: Optional[Tensor], eps: float, normalise: bool) -> Tuple[Tensor, Optional[Tensor]]:
    """diffusers' PyTorch chain."""
    if y is None:
        h = x
    elif gate is None:
        h = x + y
    else:
        h = (x.float() + y.float() * gate).type_as(x)
    if not normalise:
        return h, None
    width = h.shape[-1]
    if norm == AFFINE:
        n = F.layer_norm(h.float(), (width,), a.float(), b.float(), eps).type_as(h)
    elif norm == MODULATED:
        n = (F.layer_norm(h.float(), (width,), None, None, eps) * (1 + a) + b).type_as(h)
    else:
        n = F.layer_norm(h.float(), (width,), None, None, eps).type_as(h)
    return h, n


def fits(x: Tensor, *others: Optional[Tensor]) -> bool:
    """Inputs the kernel takes: CUDA tensors with Triton available, one row per token with unit stride, at most 16384
    channels, and every other tensor on the same device."""
    return (x.is_cuda and x.dim() >= 1 and 0 < x.shape[-1] <= 16384 and x.stride(-1) == 1
            and all(other is None or other.device == x.device for other in others) and available(x.device))


def add_norm(x: Tensor, y: Optional[Tensor] = None, gate: Optional[Tensor] = None, *, norm: int = NONE,
             a: Optional[Tensor] = None, b: Optional[Tensor] = None, eps: float = 1e-6, normalise: bool = True,
             use_triton: Optional[bool] = None) -> Tuple[Tensor, Optional[Tensor]]:
    """`(h, n)` for a hidden state `x` [..., dim]:

    * `h` = `x` (no `y`), `x + y` (no `gate`) or `(x.float() + y * gate).type_as(x)`; `y` has the shape of `x`;
    * `n` = the layer norm of `h` (float32), then `* a + b` (`norm` AFFINE: weight and bias), `* (1 + a) + b`
      (MODULATED: scale and shift) or nothing (NONE), rounded to the dtype of `x`; None when `normalise` is False.

    `gate`, `a` and `b` are per-channel vectors of `dim` values ([dim] or [1, 1, dim]); `h` is `x` itself when there
    is no `y`."""
    width = x.shape[-1]
    gate, a, b = _vector(gate, width), _vector(a, width), _vector(b, width)
    if norm != NONE and (a is None or b is None):
        raise ValueError("an affine or modulated layer norm needs both of its vectors")
    if use_triton is None:
        use_triton = fits(x, y, gate, a, b)
    if not use_triton:
        return _reference(x, y, gate, norm, a, b, eps, normalise)
    import triton

    rows = x.reshape(-1, width)
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    other = None
    if y is not None:
        other = y.reshape(-1, width)
        if other.stride(-1) != 1:
            other = other.contiguous()
    h = x if y is None else torch.empty_like(rows)
    n = torch.empty_like(rows) if normalise else None
    count = rows.shape[0]
    if count and (y is not None or normalise):
        block = triton.next_power_of_2(width)
        with torch.cuda.device(x.device):
            _build()[(count,)](
                rows, rows if other is None else other, rows if gate is None else gate, rows if a is None else a,
                rows if b is None else b, rows if y is None else h, rows if n is None else n, width, eps,
                rows.stride(0), 0 if other is None else other.stride(0), 0 if y is None else h.stride(0),
                0 if n is None else n.stride(0), BLOCK=block, HAS_Y=y is not None, HAS_GATE=gate is not None,
                NORM=norm, HAS_N=normalise, num_warps=16 if block > 8192 else 8 if block > 2048 else 4)
    if y is not None:
        h = h.reshape(x.shape)
    return h, None if n is None else n.reshape(x.shape)
