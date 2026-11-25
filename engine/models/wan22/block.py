# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Wan transformer block with its elementwise work fused into a few Triton passes.

diffusers' `WanTransformerBlock.forward` (0.42, Apache-2.0) runs, around its two attentions and its feed-forward part,
chains of PyTorch operations on the whole hidden state (tokens x 5120: 400 MB in bfloat16 for an 848x480, 97-frame
clip): a float32 copy, the layer norm, the modulation `* (1 + scale) + shift` and the rounding back each make a pass;
measured, those chains took about 100 ms of the 458 ms of one FP8 block on an RTX 4070 Ti Super. Here every residual
add and the layer norm after it are one pass (`kernels.norms.add_norm`), and the GELU after the first feed-forward
layer is applied in the FP8 product's output pass (`core.fp8_linear.linear_gelu`) when that layer is an FP8Linear.

The fused forward runs where the Triton kernels are in use, i.e. under `core.attention.use("sage")` (as the rotary
kernel, `kernels.rope`), on CUDA, for bfloat16 hidden states and the per-sample time embedding of Wan 2.2 A14B. Anywhere
else it is diffusers' own forward. The arithmetic is the same (see `kernels.norms`); `install` swaps the class of
every block of a loaded model (a subclass, not a bound method, so no reference cycle keeps the weights alive).
"""
from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor, nn

from ...core import attention
from ...core.fp8_linear import linear_gelu
from ...kernels import norms


@lru_cache(maxsize=None)
def fused_class(base: type) -> type:
    """`base` (diffusers' WanTransformerBlock) with the fused forward."""

    class FusedWanBlock(base):
        def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor, temb: Tensor,
                    rotary_emb) -> Tensor:
            if not _fusable(self, hidden_states, temb):
                return super().forward(hidden_states, encoder_hidden_states, temb, rotary_emb)
            shift, scale, gate, c_shift, c_scale, c_gate = (self.scale_shift_table + temb.float()).chunk(6, dim=1)
            eps = self.norm1.eps
            # 1. self-attention on the modulated norm
            _, normed = norms.add_norm(hidden_states, norm=norms.MODULATED, a=scale, b=shift, eps=eps)
            out = self.attn1(normed, None, None, rotary_emb)
            # 2. gated residual, the cross-attention's norm, the cross-attention
            if isinstance(self.norm2, nn.Identity):  # a block without the cross-attention norm
                hidden_states, _ = norms.add_norm(hidden_states, out, gate, normalise=False)
                normed = hidden_states
            else:
                affine = self.norm2.weight is not None
                hidden_states, normed = norms.add_norm(
                    hidden_states, out, gate, norm=norms.AFFINE if affine else norms.NONE,
                    a=self.norm2.weight, b=self.norm2.bias, eps=self.norm2.eps)
            out = self.attn2(normed, encoder_hidden_states, None, None)
            # 3. residual, the feed-forward part on the modulated norm, gated residual
            hidden_states, normed = norms.add_norm(hidden_states, out, norm=norms.MODULATED, a=c_scale, b=c_shift,
                                                   eps=self.norm3.eps)
            out = _feed_forward(self.ffn, normed)
            hidden_states, _ = norms.add_norm(hidden_states, out, c_gate, normalise=False)
            return hidden_states

    FusedWanBlock.__name__ = FusedWanBlock.__qualname__ = f"Fused{base.__name__}"
    return FusedWanBlock


def _fusable(block, hidden_states: Tensor, temb: Tensor) -> bool:
    norm2 = block.norm2
    return (attention.current() == attention.SAGE and hidden_states.dtype == torch.bfloat16 and temb.ndim == 3
            and temb.shape[0] == 1 and (isinstance(norm2, nn.Identity) or norm2.weight is None
                                        or norm2.bias is not None) and norms.fits(hidden_states, temb))


def _feed_forward(ffn, x: Tensor) -> Tensor:
    """diffusers' FeedForward with the tanh GELU (`GELU(approximate="tanh")`, then dropout and the second layer), the
    GELU applied in the first layer's FP8 output pass where it can be (`core.fp8_linear.linear_gelu`)."""
    first = ffn.net[0]
    if getattr(first, "approximate", None) != "tanh" or first._forward_hooks or first._forward_pre_hooks:
        return ffn(x)
    hidden = linear_gelu(first.proj, x)
    for module in ffn.net[1:]:
        hidden = module(hidden)
    return hidden


def install(model) -> int:
    """Give every Wan transformer block of `model` the fused forward; returns how many there were."""
    from diffusers.models.transformers.transformer_wan import WanTransformerBlock

    count = 0
    for module in model.modules():
        if isinstance(module, WanTransformerBlock) and not type(module).__name__.startswith("Fused"):
            module.__class__ = fused_class(type(module))
            count += 1
    return count
