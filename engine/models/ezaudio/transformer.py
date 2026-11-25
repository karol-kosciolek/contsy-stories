# SPDX-FileCopyrightText: 2025 Jiarui Hai
# SPDX-FileCopyrightText: 2024 Junsong Chen, Jincheng Yu, Enze Xie
# SPDX-FileCopyrightText: 2022 Fan Bao
# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only AND MIT AND Apache-2.0
# The block (`Block`) is EzAudio's "modified PixArt block with adaptive layer norm (adaLN-single)" (src/models/
# blocks.py), so it also derives from PixArt-alpha (https://github.com/PixArt-alpha/PixArt-alpha, Apache-2.0; text
# in LICENSES/Apache-2.0.txt). The long skip connections (`Block.merge_skip`, the skip stack in
# `Transformer.forward`) are written from U-ViT (https://github.com/baofff/U-ViT, MIT, Copyright (c) 2022 Fan Bao;
# text in LICENSES/MIT.txt). The LayerNorm in front of the skip projection is there because EzAudio's checkpoint
# holds one; other diffusion transformers that normalize their long skips the same way contributed no code.
# Changes for Contsy: rewritten for the one published configuration, float32 residual stream and normalizations
# around 16-bit matrices, checkpoint loading by key renaming. See THIRD_PARTY_NOTICES.md.
"""The EzAudio diffusion transformer: MaskDiT around a U-ViT with long skip connections.

Ported from github.com/haidog-yaqub/EzAudio at 86f18268 (src/models/conditioners.py, udit.py, blocks.py and
utils/{attention,modules,rotary}.py, MIT), for the published configuration only: one-dimensional input with
patch size 1, cross-attention to the text, rotary positions in the self-attention, LayerNorm on the queries and
keys, GEGLU feed-forward layers and the "ada_sola_bias" time conditioning. Rotary positions and the sinusoidal
time embedding are the standard formulas written out here; no code of the upstream rotary module (which comes from
xformers) is carried over.

The network predicts the velocity of a 128-channel latent at 50 frames per second. Its input stacks three things
per frame: the noisy latent, a clean latent to continue or to fill in (`known`), and a mask channel that is 1
where the network has to generate. Without known audio the second part is a learned token and the mask is all
ones; this is plain text-to-audio.

In float32 the forward pass repeats the reference operation for operation. With 16-bit weights the residual
stream, the normalizations, the time conditioning and the four layers at the edges of the network (input, text and
output projections, final convolution) stay float32; only the matrices inside the blocks run in 16 bits.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..zimage import assert_loaded, materialize

_TIME_FREQUENCIES = 256
_MODULATIONS = 6  # shift, scale and gate for the self-attention and for the feed-forward layer
Rotary = Tuple[Tensor, Tensor]


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotary positions on [..., length, head_dim] with the head split in two halves (not interleaved)."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.LayerNorm(dim // heads)
        self.norm_k = nn.LayerNorm(dim // heads)
        self.proj = nn.Linear(dim, dim)

    def _split(self, x: Tensor) -> Tensor:
        return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def forward(self, x: Tensor, context: Tensor, rotary: Optional[Rotary] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        value = self._split(self.to_v(context))
        # Normalized and rotated in float32, as the reference does under mixed precision.
        query = self.norm_q(self._split(self.to_q(x)).float())
        key = self.norm_k(self._split(self.to_k(context)).float())
        if rotary is not None:
            query, key = _rotate(query, *rotary), _rotate(key, *rotary)
        out = F.scaled_dot_product_attention(query.to(value.dtype), key.to(value.dtype), value, attn_mask=mask)
        return self.proj(out.transpose(1, 2).flatten(2))


class Block(nn.Module):
    """Self-attention, cross-attention to the text and a GEGLU feed-forward layer. A block of the second half
    first merges the output of its mirror block in the first half (`skip`, see `merge_skip`)."""

    def __init__(self, dim: int, heads: int, hidden: int, rank: int, skip: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.cross_attn = Attention(dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp_in = nn.Linear(dim, 2 * hidden)  # GEGLU: the values and their gates
        self.mlp_out = nn.Linear(hidden, dim)
        # AdaLN-SOLA: the modulation shared by all blocks plus a low-rank correction and a bias per block.
        self.scale_shift_table = nn.Parameter(torch.zeros(_MODULATIONS, dim))
        self.sola_a = nn.Linear(dim, _MODULATIONS * rank, bias=False)
        self.sola_b = nn.Linear(_MODULATIONS * rank, _MODULATIONS * dim, bias=False)
        if skip:  # the two layers of the long skip, under the names of EzAudio's checkpoint
            self.skip_norm = nn.LayerNorm(2 * dim)
            self.skip_linear = nn.Linear(2 * dim, dim)

    def merge_skip(self, x: Tensor, skip: Tensor, dtype: torch.dtype) -> Tensor:
        """U-ViT's long skip connection (Bao et al., "All are Worth Words: A ViT Backbone for Diffusion Models",
        CVPR 2023): the tokens entering this block and the tokens its mirror block in the first half produced are
        joined feature by feature, [x, skip] with 2 * dim features, and a linear layer maps them back to dim.
        EzAudio normalizes the joined features (zero mean and unit variance per token over all 2 * dim features,
        then its learned scale and bias) before that layer. The normalization runs in float32 like the residual
        stream; the projection runs in the blocks' compute dtype."""
        joined = torch.cat((x, skip), dim=-1)
        joined = F.layer_norm(joined, self.skip_norm.normalized_shape, self.skip_norm.weight, self.skip_norm.bias,
                              self.skip_norm.eps)
        return F.linear(joined.to(dtype), self.skip_linear.weight, self.skip_linear.bias).float()

    def forward(self, x: Tensor, context: Tensor, context_mask: Optional[Tensor], time: Tensor, time_ada: Tensor,
                rotary: Rotary, skip: Optional[Tensor] = None) -> Tensor:
        dtype = self.mlp_out.weight.dtype
        if skip is not None:
            x = self.merge_skip(x, skip, dtype)
        # The published configurations set alpha equal to the rank, so the low-rank term is not scaled.
        modulation = (time_ada + self.sola_b(self.sola_a(time))).unflatten(1, (_MODULATIONS, -1))
        modulation = self.scale_shift_table + modulation
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(_MODULATIONS, dim=1)

        hidden = (self.norm1(x) * (1 + scale_attn) + shift_attn).to(dtype)
        x = x + (1 - gate_attn) * self.attn(hidden, hidden, rotary)
        x = x + self.cross_attn(self.norm2(x).to(dtype), self.norm_context(context.float()).to(dtype),
                                mask=context_mask)
        hidden = (self.norm3(x) * (1 + scale_mlp) + shift_mlp).to(dtype)
        values, gates = self.mlp_in(hidden).chunk(2, dim=-1)
        return x + (1 - gate_mlp) * self.mlp_out(values * F.gelu(gates))


class Transformer(nn.Module):
    """MaskDiT + UDiT. `blocks` holds the first half, the middle block and the second half in one list."""

    def __init__(self, *, dim: int, latent_channels: int, context_dim: int, heads: int, hidden: int, rank: int,
                 depth: int):
        super().__init__()
        self.sizes = dict(dim=dim, latent_channels=latent_channels, context_dim=context_dim, heads=heads,
                          hidden=hidden, rank=rank, depth=depth)  # compared with the YAML by config.check_sizes
        self.latent_channels = latent_channels
        self.context_dim = context_dim
        self.mask_embed = nn.Parameter(torch.zeros(latent_channels))  # stands for "no known audio in this frame"
        self.patch_embed = nn.Conv1d(2 * latent_channels + 1, dim, 1)
        self.time_embed = nn.Sequential(nn.Linear(_TIME_FREQUENCIES, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_ada = nn.Linear(dim, _MODULATIONS * dim)
        self.time_ada_final = nn.Linear(dim, 2 * dim)
        self.context_embed = nn.Sequential(nn.Linear(context_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        # One list (first half, middle block, second half) so that core.streaming can lend the GPU block by block.
        self.blocks = nn.ModuleList(Block(dim, heads, hidden, rank, skip=index > depth // 2)
                                    for index in range(depth + 1))
        self.final_norm = nn.LayerNorm(dim)
        self.final_linear = nn.Linear(dim, latent_channels)
        self.final_conv = nn.Conv1d(latent_channels, latent_channels, 3, padding=1)
        self.register_buffer("rotary_inv_freq", torch.zeros(dim // heads // 2))

    def prepare_context(self, embedding: Tensor) -> Tensor:
        """The text tokens as the blocks read them; they do not depend on the step, so compute them once."""
        return self.context_embed(embedding.to(self.context_embed[0].weight.dtype))

    def prepare_rotary(self, length: int) -> Rotary:
        positions = torch.arange(length, device=self.rotary_inv_freq.device, dtype=torch.float32)
        angles = torch.outer(positions, self.rotary_inv_freq.float())
        return angles.cos(), angles.sin()

    def _time(self, timestep: Tensor, batch: int) -> Tensor:
        frequencies = torch.exp(-math.log(10000) * torch.arange(_TIME_FREQUENCIES // 2, dtype=torch.float32)
                                / (_TIME_FREQUENCIES // 2)).to(timestep.device)
        phase = timestep.reshape(-1).expand(batch)[:, None].float() * frequencies[None]
        return F.silu(self.time_embed(torch.cat((phase.cos(), phase.sin()), dim=-1)))

    def _input(self, latent: Tensor, known: Optional[Tensor], mask: Optional[Tensor]) -> Tensor:
        token = self.mask_embed.view(1, -1, 1)
        if known is None:
            return torch.cat((latent, token.expand_as(latent), torch.ones_like(latent[:, :1])), dim=1)
        if mask is None:
            raise ValueError("known audio needs the mask of the frames to generate")
        generate = mask.to(torch.bool).expand(latent.shape[0], 1, latent.shape[2])
        return torch.cat((latent, torch.where(generate, token, known.to(latent.dtype)),
                          generate.to(latent.dtype)), dim=1)

    def forward(self, latent: Tensor, timestep: Tensor, context: Tensor, context_mask: Optional[Tensor],
                rotary: Rotary, known: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        """The velocity for `latent` [B, C, L] at `timestep` (0..999, one value or one per sample).

        `context` comes from `prepare_context` and `context_mask` [B, T] is True on real text tokens.
        `known` [B, C, L] with `mask` [B, 1, L] (True where the network generates) continues or fills in audio."""
        time = self._time(timestep, latent.shape[0])
        time_ada = self.time_ada(time)
        keys = None if context_mask is None else context_mask[:, None, None, :]
        x = self.patch_embed(self._input(latent.float(), known, mask).to(self.patch_embed.weight.dtype))
        x = x.transpose(1, 2).float()
        # U-ViT's long skips: each block of the first half keeps its output, and the second half takes them back in
        # reverse order, so block half + k merges the output of block half - k.
        half = len(self.blocks) // 2
        skips = []
        for index, block in enumerate(self.blocks):
            x = block(x, context, keys, time, time_ada, rotary, skips.pop() if index > half else None)
            if index < half:
                skips.append(x)
        shift, scale = self.time_ada_final(time).unflatten(1, (2, -1)).chunk(2, dim=1)
        x = self.final_linear((self.final_norm(x) * (1 + scale) + shift).to(self.final_linear.weight.dtype))
        return self.final_conv(x.transpose(1, 2).to(self.final_conv.weight.dtype)).float()


_ROTARY_KEY = ".attn.rotary.inv_freq"
_RENAMED = (
    ("model.patch_embed.proj.", "patch_embed."),
    ("model.time_embed.mlp.", "time_embed."),
    ("model.final_block.norm.", "final_norm."),
    ("model.final_block.linear.", "final_linear."),
    ("model.final_block.final_layer.", "final_conv."),
    ("model.time_ada", "time_ada"),
    ("model.context_embed.", "context_embed."),
)
_RENAMED_IN_BLOCK = (
    (".mlp.net.0.proj.", ".mlp_in."),
    (".mlp.net.2.", ".mlp_out."),
    (".adaln.scale_shift_table", ".scale_shift_table"),
    (".adaln.lora_a.", ".sola_a."),
    (".adaln.lora_b.", ".sola_b."),
)


def _block_name(name: str, half: int) -> Optional[str]:
    """`blocks.N.<rest>` for a tensor of the first half, the middle block or the second half; None for others."""
    if name.startswith("model.mid_block."):
        index, rest = half, name.removeprefix("model.mid_block.")
    elif name.startswith(("model.in_blocks.", "model.out_blocks.")):
        group, number, rest = name.removeprefix("model.").split(".", 2)
        index = int(number) + (half + 1 if group == "out_blocks" else 0)
    else:
        return None
    for old, new in _RENAMED_IN_BLOCK:
        rest = f".{rest}".replace(old, new, 1).removeprefix(".")
    return f"blocks.{index}.{rest}"


def _weight_name(name: str, half: int) -> str:
    block = _block_name(name, half)
    if block is not None:
        return block
    for old, new in _RENAMED:
        if name.startswith(old):
            return new + name.removeprefix(old)
    return name


def _tensors(state: dict, half: int) -> Iterator[Tuple[str, Tensor]]:
    """The checkpoint under the names of `Transformer`. Every self-attention stores the same rotary frequencies;
    they are checked to be equal and kept once."""
    frequencies = None
    for name, tensor in state.items():
        if name.endswith(_ROTARY_KEY):
            if frequencies is None:
                frequencies = tensor
                yield "rotary_inv_freq", tensor
            elif not torch.equal(tensor, frequencies):
                raise ValueError(f"EzAudio diffusion transformer: {name} differs from the other rotary frequencies")
            continue
        yield _weight_name(name, half), tensor


# The layers at the edges of the network: the input and text projections feed every block, the output projection and
# convolution write the velocity itself. About 4.2 M parameters (17 MB in float32). In bfloat16 they alone raised
# the error of one forward on the real weights from at most 1.9 % to 2.35 % (CPU, against float32).
_EDGE_LAYERS = ("patch_embed.", "context_embed.", "final_linear.", "final_conv.")


def _weight_dtype(name: str, tensor: Tensor) -> Optional[torch.dtype]:
    """Normalizations, the time conditioning, the tokens and the edge layers stay float32; the matrices inside the
    blocks take the compute dtype."""
    small = ("norm" in name or name.startswith(("time_", "mask_embed", "rotary_", *_EDGE_LAYERS)) or ".sola_" in name
             or name.endswith("scale_shift_table"))
    return torch.float32 if small else None


def _config(state: dict) -> dict:
    """The size of the network, read from the shapes of its tensors."""
    try:
        dim = state["model.patch_embed.proj.weight"].shape[0]
        half = 1 + max(int(name.split(".")[2]) for name in state if name.startswith("model.in_blocks."))
        return dict(dim=dim, latent_channels=state["mask_embed"].shape[0],
                    context_dim=state["model.context_embed.0.weight"].shape[1],
                    heads=dim // state["model.mid_block.attn.norm_q.weight"].shape[0],
                    hidden=state["model.mid_block.mlp.net.2.weight"].shape[1],
                    rank=state["model.mid_block.adaln.lora_a.weight"].shape[0] // _MODULATIONS, depth=2 * half)
    except (KeyError, ValueError, IndexError) as error:
        raise ValueError("The file is not an EzAudio diffusion transformer checkpoint.") from error


def load(path: Path, device: torch.device | str, dtype: torch.dtype) -> Transformer:
    """The transformer from an official `ezaudio_*.pt` checkpoint; every tensor of the file must find its place."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model"}:
        raise ValueError("An EzAudio checkpoint holds exactly one entry, 'model'.")
    state = checkpoint["model"]
    config = _config(state)
    with torch.device("meta"):
        model = Transformer(**config)
    materialize(model, _tensors(state, config["depth"] // 2), device, dtype, "EzAudio diffusion transformer",
                dtype_for=_weight_dtype)
    assert_loaded(model, "EzAudio diffusion transformer")
    return model.eval()
