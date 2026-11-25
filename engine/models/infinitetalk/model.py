# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The InfiniteTalk network: Wan 2.1 image-to-video 14B (480p) with MeiGen-AI's single-speaker audio adapter.

Reference: MeiGen-AI/InfiniteTalk (Apache-2.0), `wan/modules/multitalk_model.py` (`WanModel`, `WanAttentionBlock`,
`AudioProjModel`) and `wan/modules/attention.py` (`SingleStreamAttention`).

* The Wan part is diffusers' `WanTransformer3DModel` with the Wan 2.1 I2V config (36 input channels, the CLIP image
  embedder and the image key/value branch of every cross-attention, see `models.wan22.attention`), filled from
  LightX2V's step- and guidance-distilled FP8 release of Wan 2.1 I2V 480p (`block_<n>.safetensors` and
  `non_block.safetensors`, per-channel scaled FP8: the linear layers stay FP8, `core.fp8_linear.FP8Linear`). The 4-step
  distillation is in those weights, so no LoRA is attached.
* Every block gets, after its cross-attention, an audio cross-attention on its own input norm (`norm_x`, affine
  LayerNorm): the queries of each latent frame attend to the 32 audio tokens of that frame only (single speaker:
  `human_num` 1, no query or key norms, biases on every projection), and the result is added to the hidden state
  before the feed-forward part.
* `audio_proj` turns the speech features of a chunk ([1, frames, 5, 12, 768], `audio.windows`) into 32 tokens of 768
  per latent frame: the first frame's window (5 x 12 x 768) through `proj1`; each later group of four frames as one
  vector of 8 x 12 x 768 (the first frame's window up to its centre, the centres of the two middle frames, the last
  frame's window from its centre) through `proj1_vf`; then ReLU, `proj2`, ReLU, `proj3` to 32 x 768 and LayerNorm.

The adapter's keys (`audio_proj.*`, `blocks.<n>.audio_cross_attn.*`, `blocks.<n>.norm_x.*`) are the module paths of
this network, so they are loaded as they are; the Wan keys go through diffusers' converter. The text context is
padded to 512 tokens with zeros, as Wan's own text embedding does: the image branch takes the tokens before the last
512 (diffusers' split).
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...core.quant import materialize_quantized
from ...formats import fp8 as fp8_format
from ...sampling.flow import denoised_from_velocity
from ..zimage import assert_loaded
from ..wan22 import attention as wan_attention
from ..wan22 import block as fused_block
from ..wan22.transformer import TIMESTEP_SCALE, _rebuild_rope, diffusers_tensors

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "wan21_i2v_480p_transformer.json"
TEXT_TOKENS = wan_attention.TEXT_TOKENS
AUDIO_DIM, AUDIO_TOKENS, AUDIO_LAYERS, AUDIO_WINDOW, VAE_SCALE = 768, 32, 12, 5, 4
_ADAPTER_PREFIXES = ("audio_proj.",)
_ADAPTER_PARTS = (".audio_cross_attn.", ".norm_x.")


class AudioProjection(nn.Module):
    """InfiniteTalk's `AudioProjModel` (window 5, later frames 8, 12 layers, 768 channels, 512 inside, 32 tokens)."""

    def __init__(self, channels: int = AUDIO_DIM, inner: int = 512, tokens: int = AUDIO_TOKENS,
                 layers: int = AUDIO_LAYERS, window: int = AUDIO_WINDOW, scale: int = VAE_SCALE):
        super().__init__()
        self.tokens, self.channels = tokens, channels
        self.proj1 = nn.Linear(window * layers * channels, inner)
        self.proj1_vf = nn.Linear((window + scale - 1) * layers * channels, inner)
        self.proj2 = nn.Linear(inner, inner)
        self.proj3 = nn.Linear(inner, tokens * channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, first: Tensor, later: Tensor) -> Tensor:
        """first [1, 1, 5, 12, C], later [1, n, 8, 12, C] -> [1, 1 + n, 32, C]."""
        batch = first.shape[0]
        a = torch.relu(self.proj1(first.flatten(2)))
        b = torch.relu(self.proj1_vf(later.flatten(2)))
        hidden = torch.relu(self.proj2(torch.cat([a, b], dim=1)))
        out = self.proj3(hidden).unflatten(-1, (self.tokens, self.channels))
        return F.layer_norm(out.float(), (self.channels,), self.norm.weight.float(), self.norm.bias.float(),
                            self.norm.eps).to(out.dtype).reshape(batch, -1, self.tokens, self.channels)


def group_windows(windows: Tensor, middle: int = AUDIO_WINDOW // 2, scale: int = VAE_SCALE) -> Tuple[Tensor, Tensor]:
    """A chunk's windows [1, F, 5, 12, C] (F = 4k + 1) -> (first frame [1, 1, 5, 12, C], later groups [1, k, 8, 12, C])
    exactly as the reference's forward builds them."""
    first = windows[:, :1]
    later = windows[:, 1:].unflatten(1, (-1, scale))  # [1, k, 4, 5, 12, C]
    head = later[:, :, :1, :middle + 1].flatten(2, 3)
    middle_frames = later[:, :, 1:-1, middle:middle + 1].flatten(2, 3)
    tail = later[:, :, -1:, middle:].flatten(2, 3)
    return first, torch.cat([head, middle_frames, tail], dim=2)


class AudioCrossAttention(nn.Module):
    """`SingleStreamAttention` for one speaker: per latent frame, its tokens attend to that frame's audio tokens."""

    def __init__(self, dim: int = 5120, context_dim: int = AUDIO_DIM, heads: int = 40):
        super().__init__()
        self.heads = heads
        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(context_dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, audio: Tensor) -> Tensor:
        """x [1, frames * S, dim], audio [frames, 32, C] -> [1, frames * S, dim]."""
        frames = audio.shape[0]
        dim = x.shape[-1]
        q = self.q_linear(x).view(frames, -1, self.heads, dim // self.heads).transpose(1, 2)
        k, v = self.kv_linear(audio.to(x.dtype)).view(frames, -1, 2, self.heads, dim // self.heads).unbind(2)
        out = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(out.transpose(1, 2).reshape(1, -1, dim))


@lru_cache(maxsize=None)
def talk_class(base: type) -> type:
    """`base` (diffusers' WanTransformerBlock) with the audio cross-attention between its cross-attention and its
    feed-forward part. The audio tokens of the call are set on the block (`audio_tokens`) around the model call.
    Where the engine's Triton kernels run (`block._fusable`), the residual adds and norms are fused passes as in
    `models.wan22.block`; elsewhere it is diffusers' arithmetic plus the audio stage."""
    from ...kernels import norms

    class TalkBlock(base):
        audio = None  # [frames, 32, C] during a model call

        def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor, temb: Tensor, rotary_emb) -> Tensor:
            if self.audio is None:
                raise RuntimeError("the InfiniteTalk block was called without its audio tokens")
            shift, scale, gate, c_shift, c_scale, c_gate = (self.scale_shift_table + temb.float()).chunk(6, dim=1)
            if fused_block._fusable(self, hidden_states, temb):
                _, normed = norms.add_norm(hidden_states, norm=norms.MODULATED, a=scale, b=shift, eps=self.norm1.eps)
                out = self.attn1(normed, None, None, rotary_emb)
                hidden_states, normed = norms.add_norm(hidden_states, out, gate, norm=norms.AFFINE,
                                                       a=self.norm2.weight, b=self.norm2.bias, eps=self.norm2.eps)
                out = self.attn2(normed, encoder_hidden_states, None, None)
                hidden_states, normed = norms.add_norm(hidden_states, out, norm=norms.AFFINE, a=self.norm_x.weight,
                                                       b=self.norm_x.bias, eps=self.norm_x.eps)
                out = self.audio_cross_attn(normed, self.audio)
                hidden_states, normed = norms.add_norm(hidden_states, out, norm=norms.MODULATED, a=c_scale, b=c_shift,
                                                       eps=self.norm3.eps)
                out = fused_block._feed_forward(self.ffn, normed)
                hidden_states, _ = norms.add_norm(hidden_states, out, c_gate, normalise=False)
                return hidden_states
            normed = (self.norm1(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
            out = self.attn1(normed, None, None, rotary_emb)
            hidden_states = (hidden_states.float() + out * gate).type_as(hidden_states)
            normed = self.norm2(hidden_states.float()).type_as(hidden_states)
            hidden_states = hidden_states + self.attn2(normed, encoder_hidden_states, None, None)
            normed = F.layer_norm(hidden_states.float(), (hidden_states.shape[-1],), self.norm_x.weight.float(),
                                  self.norm_x.bias.float(), self.norm_x.eps).type_as(hidden_states)
            hidden_states = hidden_states + self.audio_cross_attn(normed, self.audio)
            normed = (self.norm3(hidden_states.float()) * (1 + c_scale) + c_shift).type_as(hidden_states)
            out = self.ffn(normed)
            return (hidden_states.float() + out.float() * c_gate).type_as(hidden_states)

    TalkBlock.__name__ = TalkBlock.__qualname__ = f"Talk{base.__name__}"
    return TalkBlock


def build(config: Optional[Dict] = None):
    """The network on the meta device: Wan 2.1 I2V (default: the shipped 480p config) with the audio adapter."""
    from diffusers import WanTransformer3DModel

    if config is None:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = WanTransformer3DModel.from_config({k: v for k, v in config.items() if not k.startswith("_")})
        dim = model.config.num_attention_heads * model.config.attention_head_dim
        model.audio_proj = AudioProjection()
        for block in model.blocks:
            block.audio_cross_attn = AudioCrossAttention(dim, AUDIO_DIM, model.config.num_attention_heads)
            block.norm_x = nn.LayerNorm(dim, eps=model.config.eps)
            block.__class__ = talk_class(type(block))
    return model


def _adapter_key(name: str) -> bool:
    return name.startswith(_ADAPTER_PREFIXES) or any(part in name for part in _ADAPTER_PARTS)


def checkpoint_tensors(base_files: Sequence[str | Path], adapter: str | Path) -> Iterator[Tuple[str, object]]:
    """(name, tensor or ScaledFP8) of the Wan files (original names, renamed for diffusers) and of the adapter (its
    own names), one tensor at a time."""
    for path in base_files:
        yield from diffusers_tensors(fp8_format.iter_kept_fp8(path))
    for name, tensor in fp8_format.iter_kept_fp8(adapter):
        if not _adapter_key(name):
            raise ValueError(f"{Path(adapter).name}: {name} is not part of the InfiniteTalk adapter")
        yield name, tensor


def load(base_files: Sequence[str | Path], adapter: str | Path, device: torch.device | str = "cpu",
         dtype: torch.dtype = torch.bfloat16, config: Optional[Dict] = None):
    """The network filled from the Wan files and the adapter; FP8 linear layers stay FP8 (`FP8Linear`)."""
    model = build(config)
    materialize_quantized(model, checkpoint_tensors(base_files, adapter), device, dtype, "InfiniteTalk")
    _rebuild_rope(model, device)
    wan_attention.install(model)
    assert_loaded(model, "InfiniteTalk")
    return model.eval()


def pad_context(context: Tensor, tokens: int = TEXT_TOKENS) -> Tensor:
    """[1, L, 4096] -> [1, 512, 4096], zeros after the prompt (Wan's own padding)."""
    if context.shape[1] > tokens:
        return context[:, :tokens]
    return F.pad(context, (0, 0, 0, tokens - context.shape[1]))


@contextmanager
def audio_tokens(model, tokens: Tensor):
    """Give every block the audio tokens [frames, 32, C] for the calls inside."""
    blocks = list(model.blocks)
    for block in blocks:
        block.audio = tokens
    try:
        yield
    finally:
        for block in blocks:
            block.audio = None


def predict_velocity(model, latent: Tensor, sigma: Tensor, context: Tensor, concat: Tensor, image: Tensor,
                     windows: Tensor) -> Tensor:
    """One forward pass: the velocity (float32, shape of `latent` [1, 16, T, h, w]). `context` the padded text
    [1, 512, 4096], `concat` the I2V conditioning [1, 20, T, h, w], `image` the CLIP tokens [1, 257, 1280], `windows`
    the chunk's speech windows [1, 4(T - 1) + 1, 5, 12, 768] (zeros: the speech left out, for audio guidance)."""
    dtype = next(model.audio_proj.parameters()).dtype
    device = latent.device
    first, later = group_windows(windows.to(device=device, dtype=dtype))
    tokens = model.audio_proj(first, later)[0]  # [T, 32, C]
    x = torch.cat([latent, concat.to(latent.dtype)], dim=1).to(dtype)
    t = (sigma.to(device=device, dtype=torch.float32) * TIMESTEP_SCALE).reshape(1)
    with audio_tokens(model, tokens):
        out = model(x, t, context.to(device=device, dtype=dtype), encoder_hidden_states_image=image.to(device, dtype),
                    return_dict=False)[0]
    return out.float()


def predict_denoised(model, latent: Tensor, sigma: Tensor, *args) -> Tensor:
    return denoised_from_velocity(latent, sigma.to(latent.device), predict_velocity(model, latent, sigma, *args))
