# SPDX-FileCopyrightText: 2022 The HuggingFace Team
# SPDX-FileCopyrightText: 2024 Bytedance Ltd. and/or its affiliates
# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only AND Apache-2.0
"""A Stable-Diffusion-1.x UNet (the network of diffusers' `UNet2DConditionModel` for SD-1.x configurations).

Used by MuseTalk 1.5 (`engine.models.musetalk.unet`). Own implementation on torch alone, modelled on diffusers
(Apache-2.0, https://github.com/huggingface/diffusers, `models/unets/unet_2d_condition.py`, `unet_2d_blocks.py`,
`resnet.py`, `attention.py`, `transformer_2d.py`, `embeddings.py`); it was first written for ByteDance LatentSync's
video version of the same network (Apache-2.0), from which only the 2D layers are left. Taken from upstream: the
module tree and parameter names (a checkpoint is a plain `state_dict`, so names and shapes must match exactly) and
the sequence of operations. Only the options of the SD-1.x family exist; the config reader of the model that uses
this network (`musetalk.unet.unet_config`) refuses any other.

Conventions read from the reference, each of which changes the output if it is "fixed":

* "attention_head_dim: 8" in an SD-1.x config is the NUMBER of heads of every attention; the size of a head is
  the channel count divided by 8.
* Every spatial transformer has self-attention, then cross-attention to `encoder_hidden_states`
  [batch, tokens, cross_attention_dim], then a GEGLU feed-forward, each after a LayerNorm and with a residual.
* Time: sinusoidal embedding of block_out_channels[0] channels (cos first with flip_sin_to_cos), computed in
  float32, cast to the model dtype, then Linear - SiLU - Linear to 4x; every ResNet adds `Linear(SiLU(emb))` as a
  bias per channel.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_CROSS_ATTENTION_BLOCKS = {"CrossAttnDownBlock2D": True, "DownBlock2D": False,
                           "CrossAttnUpBlock2D": True, "UpBlock2D": False}
MID_BLOCK = "UNetMidBlock2DCrossAttn"


# ----------------------------------------------------------------------------------------------
# small layers
# ----------------------------------------------------------------------------------------------

def timestep_embedding(timesteps: Tensor, dim: int, flip_sin_to_cos: bool = True, freq_shift: float = 0.0,
                       max_period: int = 10000) -> Tensor:
    """Sinusoidal embedding [N, dim] of 1D timesteps (diffusers `get_timestep_embedding`), in float32.

    With flip_sin_to_cos the cosine half comes first. `freq_shift` 0 divides the exponent by dim/2, not
    dim/2 - 1."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
    frequencies = torch.exp(exponent / (half - freq_shift))
    angles = timesteps[:, None].float() * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat([embedding[:, half:], embedding[:, :half]], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class Upsample2D(nn.Module):
    """Nearest-neighbour x2 (or to the size of the skip connection), then a 3x3 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden_states: Tensor, output_size: Optional[Sequence[int]] = None) -> Tensor:
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:  # upsample_nearest is not implemented for bfloat16 on every backend
            hidden_states = hidden_states.to(torch.float32)
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=tuple(output_size), mode="nearest")
        return self.conv(hidden_states.to(dtype))


class Downsample2D(nn.Module):
    """A strided 3x3 convolution (the 'op' downsampler of SD 1.x): height and width halve."""

    def __init__(self, channels: int, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=padding)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.conv(hidden_states)


class ResnetBlock2D(nn.Module):
    """GroupNorm - SiLU - conv, + a per-channel time bias, GroupNorm - SiLU - conv, + the (projected) input."""

    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, groups: int = 32,
                 eps: float = 1e-6, output_scale_factor: float = 1.0):
        super().__init__()
        self.output_scale_factor = output_scale_factor
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = (nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
                              if in_channels != out_channels else None)

    def forward(self, input_tensor: Tensor, temb: Tensor) -> Tensor:
        hidden_states = self.conv1(F.silu(self.norm1(input_tensor)))
        hidden_states = hidden_states + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        hidden_states = self.conv2(F.silu(self.norm2(hidden_states)))
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        return (input_tensor + hidden_states) / self.output_scale_factor


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class FeedForward(nn.Module):
    """GEGLU feed-forward with a 4x inner size. Names `net.0.proj` and `net.2` are the checkpoint's."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        self.net = nn.ModuleList([GEGLU(dim, inner), nn.Identity(), nn.Linear(inner, dim)])

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class Attention(nn.Module):
    """Multi-head attention with plain projections: q, k, v without bias, output with bias. Keys and
    values come from `encoder_hidden_states` (cross-attention) or from the input itself."""

    def __init__(self, query_dim: int, cross_attention_dim: Optional[int], heads: int, dim_head: int):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(cross_attention_dim if cross_attention_dim is not None else query_dim, inner, bias=False)
        self.to_v = nn.Linear(cross_attention_dim if cross_attention_dim is not None else query_dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, query_dim)])  # upstream has a Dropout(0.0) at index 1

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Optional[Tensor] = None) -> Tensor:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch = hidden_states.shape[0]
        query, key, value = (
            projection(source).reshape(batch, -1, self.heads, projection.out_features // self.heads).transpose(1, 2)
            for projection, source in ((self.to_q, hidden_states), (self.to_k, context), (self.to_v, context)))
        attended = F.scaled_dot_product_attention(query, key, value)
        return self.to_out[0](attended.transpose(1, 2).reshape(batch, -1, self.heads * query.shape[-1]))


class BasicTransformerBlock(nn.Module):
    """Self-attention, cross-attention, feed-forward; each with a pre-LayerNorm and a residual."""

    def __init__(self, dim: int, heads: int, head_dim: int, cross_attention_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, None, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(dim, cross_attention_dim, heads, head_dim)
        self.ff = FeedForward(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), encoder_hidden_states) + hidden_states
        return self.ff(self.norm3(hidden_states)) + hidden_states


class Transformer2DModel(nn.Module):
    """Spatial transformer of one resolution: GroupNorm, 1x1 conv in, blocks over the pixels, 1x1 conv out, residual."""

    def __init__(self, heads: int, head_dim: int, in_channels: int, cross_attention_dim: int, norm_num_groups: int,
                 num_layers: int = 1):
        super().__init__()
        inner = heads * head_dim
        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(in_channels, inner, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner, heads, head_dim, cross_attention_dim) for _ in range(num_layers)])
        self.proj_out = nn.Conv2d(inner, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        batch, _, height, width = residual.shape
        hidden_states = self.proj_in(self.norm(residual))
        inner = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, inner).permute(0, 3, 1, 2).contiguous()
        return self.proj_out(hidden_states) + residual


# ----------------------------------------------------------------------------------------------
# blocks
# ----------------------------------------------------------------------------------------------

class UNetMidBlock2DCrossAttn(nn.Module):
    """resnet, then (spatial transformer, resnet) once."""

    def __init__(self, in_channels: int, temb_channels: int, eps: float, groups: int, heads: int,
                 cross_attention_dim: int, output_scale_factor: float):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, in_channels, temb_channels, groups, eps, output_scale_factor)
            for _ in range(2)])
        self.attentions = nn.ModuleList([
            Transformer2DModel(heads, in_channels // heads, in_channels, cross_attention_dim, groups)])

    def forward(self, hidden_states: Tensor, temb: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = resnet(attention(hidden_states, encoder_hidden_states), temb)
        return hidden_states


class DownBlock2D(nn.Module):
    """`num_layers` x (resnet, [spatial transformer]), then an optional downsampler.

    Each layer's output is kept as a skip connection for the up path, and so is the downsampler's."""

    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, num_layers: int, add_downsample: bool,
                 eps: float, groups: int, downsample_padding: int, heads: int, cross_attention_dim: int,
                 with_attention: bool):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, temb_channels, groups, eps)
            for i in range(num_layers)])
        self.attentions = nn.ModuleList([
            Transformer2DModel(heads, out_channels // heads, out_channels, cross_attention_dim, groups)
            for _ in range(num_layers)]) if with_attention else None
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, downsample_padding)]) if add_downsample else None

    def forward(self, hidden_states: Tensor, temb: Tensor,
                encoder_hidden_states: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        output_states: Tuple[Tensor, ...] = ()
        attentions = self.attentions if self.attentions is not None else [None] * len(self.resnets)
        for resnet, attention in zip(self.resnets, attentions):
            hidden_states = resnet(hidden_states, temb)
            if attention is not None:
                hidden_states = attention(hidden_states, encoder_hidden_states)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class UpBlock2D(nn.Module):
    """`num_layers` x (concat a skip connection, resnet, [spatial transformer]), then an optional upsampler."""

    def __init__(self, in_channels: int, out_channels: int, prev_output_channel: int, temb_channels: int,
                 num_layers: int, add_upsample: bool, eps: float, groups: int, heads: int, cross_attention_dim: int,
                 with_attention: bool):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnets.append(ResnetBlock2D((prev_output_channel if i == 0 else out_channels) + skip_channels,
                                         out_channels, temb_channels, groups, eps))
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList([
            Transformer2DModel(heads, out_channels // heads, out_channels, cross_attention_dim, groups)
            for _ in range(num_layers)]) if with_attention else None
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None

    def forward(self, hidden_states: Tensor, res_hidden_states_tuple: Tuple[Tensor, ...], temb: Tensor,
                encoder_hidden_states: Tensor, upsample_size: Optional[Sequence[int]] = None) -> Tensor:
        attentions = self.attentions if self.attentions is not None else [None] * len(self.resnets)
        for resnet, attention in zip(self.resnets, attentions):
            hidden_states = torch.cat([hidden_states, res_hidden_states_tuple[-1]], dim=1)
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = resnet(hidden_states, temb)
            if attention is not None:
                hidden_states = attention(hidden_states, encoder_hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


# ----------------------------------------------------------------------------------------------
# the network
# ----------------------------------------------------------------------------------------------

class SDUNet(nn.Module):
    """The SD-1.x UNet. The constructor takes the keys of a diffusers `UNet2DConditionModel` config of that
    family (without the `_class_name` / `_diffusers_version` entries)."""

    def __init__(self, sample_size: Optional[int] = None, in_channels: int = 4, out_channels: int = 4,
                 center_input_sample: bool = False, flip_sin_to_cos: bool = True, freq_shift: int = 0,
                 down_block_types: Sequence[str] = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                                                    "CrossAttnDownBlock2D", "DownBlock2D"),
                 mid_block_type: str = MID_BLOCK,
                 up_block_types: Sequence[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                                                  "CrossAttnUpBlock2D"),
                 block_out_channels: Sequence[int] = (320, 640, 1280, 1280), layers_per_block: int = 2,
                 downsample_padding: int = 1, mid_block_scale_factor: float = 1, act_fn: str = "silu",
                 norm_num_groups: int = 32, norm_eps: float = 1e-5, cross_attention_dim: int = 1280,
                 attention_head_dim: Union[int, Sequence[int]] = 8):
        super().__init__()
        if act_fn not in ("silu", "swish"):
            raise ValueError(f"act_fn={act_fn!r} is not supported")
        if mid_block_type != MID_BLOCK:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")
        for kind in (*down_block_types, *up_block_types):
            if kind not in _CROSS_ATTENTION_BLOCKS:
                raise ValueError(f"{kind} does not exist.")
        levels = len(block_out_channels)
        if len(down_block_types) != levels or len(up_block_types) != levels:
            raise ValueError("block_out_channels, down_block_types and up_block_types must have the same length")
        heads_per_level = ([attention_head_dim] * levels if isinstance(attention_head_dim, int)
                           else list(attention_head_dim))

        del sample_size  # the latent size it was trained at; the network takes any size
        self.center_input_sample = center_input_sample
        self.flip_sin_to_cos, self.freq_shift = flip_sin_to_cos, freq_shift
        time_embed_dim = block_out_channels[0] * 4
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, kind in enumerate(down_block_types):
            input_channel, output_channel = output_channel, block_out_channels[i]
            self.down_blocks.append(DownBlock2D(
                input_channel, output_channel, time_embed_dim, layers_per_block, i != levels - 1, norm_eps,
                norm_num_groups, downsample_padding, heads_per_level[i], cross_attention_dim,
                _CROSS_ATTENTION_BLOCKS[kind]))

        self.mid_block = UNetMidBlock2DCrossAttn(
            block_out_channels[-1], time_embed_dim, norm_eps, norm_num_groups, heads_per_level[-1],
            cross_attention_dim, mid_block_scale_factor)

        self.num_upsamplers = 0
        self.up_blocks = nn.ModuleList([])
        reversed_channels = list(reversed(block_out_channels))
        reversed_heads = list(reversed(heads_per_level))
        output_channel = reversed_channels[0]
        for i, kind in enumerate(up_block_types):
            is_final = i == levels - 1
            prev_output_channel, output_channel = output_channel, reversed_channels[i]
            input_channel = reversed_channels[min(i + 1, levels - 1)]
            if not is_final:
                self.num_upsamplers += 1
            self.up_blocks.append(UpBlock2D(
                input_channel, output_channel, prev_output_channel, time_embed_dim, layers_per_block + 1,
                not is_final, norm_eps, norm_num_groups, reversed_heads[i], cross_attention_dim,
                _CROSS_ATTENTION_BLOCKS[kind]))

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, sample: Tensor, timestep: Union[Tensor, float, int], encoder_hidden_states: Tensor) -> Tensor:
        """The output [batch, out_channels, height, width] for `sample` [batch, in_channels, height, width] at
        `timestep` (a number or a tensor of one or `batch` numbers), attending to `encoder_hidden_states`
        [batch, tokens, cross_attention_dim]."""
        factor = 2 ** self.num_upsamplers
        forward_upsample_size = any(s % factor != 0 for s in sample.shape[-2:])
        upsample_size = None
        if self.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float64 if isinstance(timestep, float) else torch.int64,
                                     device=sample.device)
        elif timesteps.dim() == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        emb = timestep_embedding(timesteps, self.conv_in.out_channels, self.flip_sin_to_cos, self.freq_shift)
        emb = self.time_embedding(emb.to(dtype=self.dtype))

        sample = self.conv_in(sample)
        down_block_res_samples: Tuple[Tensor, ...] = (sample,)
        for block in self.down_blocks:
            sample, res_samples = block(sample, emb, encoder_hidden_states)
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, emb, encoder_hidden_states)

        for i, block in enumerate(self.up_blocks):
            count = len(block.resnets)
            res_samples = down_block_res_samples[-count:]
            down_block_res_samples = down_block_res_samples[:-count]
            if i != len(self.up_blocks) - 1 and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            sample = block(sample, res_samples, emb, encoder_hidden_states, upsample_size)

        return self.conv_out(F.silu(self.conv_norm_out(sample)))

