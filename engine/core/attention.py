# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Attention for the diffusion transformers: PyTorch's scaled dot product attention, or SageAttention.

SageAttention (Zhang et al., "SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration",
ICLR 2025) quantizes the queries and keys to INT8 per block of 128 / 64 tokens, after subtracting the mean key
(which leaves the softmax unchanged and removes the keys' common offset, the main source of quantization error),
and multiplies the probabilities with the values in FP16 with an FP16 accumulator per block. On GeForce cards
both run at twice the rate of the BF16 kernels with float32 accumulation. The kernels are the authors' Triton
ones (`engine/kernels/sage`, Apache-2.0), so they need no compiled CUDA extension.

The backend is chosen per call site with `use(...)` (a context variable, so threads do not share it); outside of
it, and whenever the inputs do not fit the kernels (no Triton, the CPU, a mask, a head size other than 64 or 128,
float32), attention is PyTorch's `scaled_dot_product_attention`, the same as without this module.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .. import kernels

SDPA, SAGE, AUTO = "sdpa", "sage", "auto"
SETTINGS = (AUTO, SDPA)  # what the product's engine.attention setting may say
HEAD_DIMS = (64, 128)
_BACKEND: contextvars.ContextVar = contextvars.ContextVar("contsy_attention", default=SDPA)


def sage_available(device: torch.device | str | None = None) -> bool:
    """Whether the SageAttention kernels can run: Triton is importable and the GPU has INT8 tensor cores of
    compute capability 8.0 or later (Ampere, Ada, Hopper, Blackwell)."""
    if not kernels.available(device):
        return False
    index = None if device is None else torch.device(device).index
    return torch.cuda.get_device_capability(index) >= (8, 0)


def resolve(setting: Optional[str], device: torch.device | str | None = None) -> str:
    """The backend for an `engine.attention` setting: "auto" (or "sage") is SageAttention where it can run, else SDPA."""
    if setting in (None, SDPA):
        return SDPA
    if setting not in (AUTO, SAGE):
        raise ValueError(f"attention must be one of {SETTINGS + (SAGE,)}, not {setting!r}")
    return SAGE if sage_available(device) else SDPA


@contextmanager
def use(backend: str) -> Iterator[None]:
    """Run the block with `backend` ("sdpa" or "sage", see `resolve`) for every `attend` call of this thread."""
    if backend not in (SDPA, SAGE):
        raise ValueError(f"unknown attention backend {backend!r}")
    token = _BACKEND.set(backend)
    try:
        yield
    finally:
        _BACKEND.reset(token)


def current() -> str:
    return _BACKEND.get()


def fits_sage(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> bool:
    """Whether SageAttention's kernels take these [batch, tokens, heads, head_dim] inputs as they are."""
    return (mask is None and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and q.dtype == k.dtype == v.dtype
            and q.dim() == 4 and q.shape[-1] in HEAD_DIMS and k.shape == v.shape and k.shape[-1] == q.shape[-1]
            and q.shape[2] % k.shape[2] == 0 and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
            and q.device == k.device == v.device)


def sage(q: Tensor, k: Tensor, v: Tensor, scale: Optional[float] = None) -> Tensor:
    """SageAttention on [batch, tokens, heads, head_dim] inputs (no mask), output in the dtype of `q`."""
    from ..kernels.sage.attn_qk_int8_per_block import forward
    from ..kernels.sage.quant_per_block import per_block_int8

    with torch.cuda.device(q.device):  # Triton launches on the current device
        mean_key = k.mean(dim=1, keepdim=True)
        q8, q_scale, k8, k_scale = per_block_int8(q, k, km=mean_key, sm_scale=scale or q.shape[-1] ** -0.5,
                                                  tensor_layout="NHD")
        out, _ = forward(q8, k8, v if v.dtype == torch.float16 else v.to(torch.float16), q_scale, k_scale,
                         tensor_layout="NHD", output_dtype=q.dtype)
    return out


def sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, scale: Optional[float] = None) -> Tensor:
    """PyTorch's attention on [batch, tokens, heads, head_dim] inputs, the same layout back."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
                                         scale=scale)
    return out.transpose(1, 2)


def attend(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, scale: Optional[float] = None) -> Tensor:
    """Softmax attention of queries [B, Lq, H, D] over keys and values [B, Lk, Hkv, D]: [B, Lq, H, D].

    SageAttention when `use("sage")` is in force and the inputs fit (`fits_sage`), otherwise SDPA."""
    if _BACKEND.get() == SAGE and fits_sage(q, k, v, mask):
        return sage(q, k, v, scale)
    return sdpa(q, k, v, mask, scale)
