# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The attention of the Wan transformer blocks, computed through `core.attention`.

It is the computation of diffusers' `WanAttnProcessor` (diffusers 0.42, Apache-2.0): for Wan 2.2 A14B, which has no
image key and value branch, and for Wan 2.1 image-to-video (InfiniteTalk), whose cross-attention also attends to the
CLIP image tokens before the text (`add_k_proj` / `add_v_proj`, `norm_added_k`; the two attentions are added). The
query, key and value projections, the RMS norms of the query and key over all heads,
the rotary embedding (self-attention only), softmax attention and the output projection. Only the attention call
differs: `core.attention.attend`, which is SageAttention under `core.attention.use("sage")` and PyTorch's SDPA
otherwise (then the result is the one diffusers computes); under "sage" the rotary embedding is one Triton pass too
(`kernels.rope`, the same arithmetic). When every projection of one input is an `FP8Linear`, that
input is quantized once for all of them (`core.fp8_linear.project`).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ...core import attention
from ...core.fp8_linear import project
from ...kernels import rope

TEXT_TOKENS = 512  # the text context of every Wan model (diffusers pads the prompt to it)


def _rotate(hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """diffusers' rotary embedding of Wan: interleaved pairs (x1, x2) rotated by the angles of `cos` / `sin`.

    With the Triton kernels in use (`core.attention` on "sage") one fused pass (`kernels.rope`) computes it."""
    if attention.current() == attention.SAGE and rope.fits(hidden, cos, sin):
        return rope.rotate(hidden, cos, sin)
    x1, x2 = hidden.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = cos[..., 0::2], sin[..., 1::2]
    out = torch.empty_like(hidden)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden)


class WanAttnProcessor:
    """Replaces diffusers' processor on every `WanAttention` of the model (`install`)."""

    def __call__(self, attn, hidden_states: Tensor, encoder_hidden_states: Optional[Tensor] = None,
                 attention_mask: Optional[Tensor] = None, rotary_emb: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        if attn.fused_projections:
            raise NotImplementedError("fused projections are not used by the Wan models of this engine")
        image = None
        if attn.add_k_proj is not None and encoder_hidden_states is not None:
            # Wan 2.1 I2V: the image tokens come first, the 512 text tokens last (diffusers' split)
            split = encoder_hidden_states.shape[1] - TEXT_TOKENS
            image, encoder_hidden_states = encoder_hidden_states[:, :split], encoder_hidden_states[:, split:]
        if encoder_hidden_states is None:
            query, key, value = project(hidden_states, (attn.to_q, attn.to_k, attn.to_v))
        else:
            query = attn.to_q(hidden_states)
            key, value = project(encoder_hidden_states, (attn.to_k, attn.to_v))
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query, key = _rotate(query, *rotary_emb), _rotate(key, *rotary_emb)
        out = attention.attend(query, key, value, attention_mask).flatten(2, 3).type_as(query)
        if image is not None:
            image_key = attn.norm_added_k(attn.add_k_proj(image)).unflatten(2, (attn.heads, -1))
            image_value = attn.add_v_proj(image).unflatten(2, (attn.heads, -1))
            out = out + attention.attend(query, image_key, image_value, None).flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](out))


def install(model) -> int:
    """Give every Wan attention module of `model` this processor; returns how many there were."""
    from diffusers.models.transformers.transformer_wan import WanAttention

    count = 0
    for module in model.modules():
        if isinstance(module, WanAttention):
            module.set_processor(WanAttnProcessor())
            count += 1
    return count
