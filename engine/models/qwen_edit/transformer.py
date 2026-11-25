# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Qwen-Image-Edit-2511 diffusion transformer (a 60-block double-stream MMDiT) with reference pictures.

The network classes are diffusers' `QwenImageTransformer2DModel` (Apache-2.0, Qwen-Image Team and The HuggingFace
Team); they are imported, not copied. `_forward`, `pack` and `unpack` follow the order of operations of diffusers'
`QwenImageTransformer2DModel.forward` and `QwenImagePipeline._pack_latents` / `_unpack_latents` (Apache-2.0),
re-written here for the reasons below. The check that a file carrying the `__index_timestep_zero__` marker needs
`zero_cond_t` follows musubi-tuner's loader (Apache-2.0).

Conventions (diffusers 0.42 `QwenImageEditPlusPipeline` describes the model; where this module differs, the
difference is Contsy's own choice and says so):

* Sampling state is a velocity flow like Z-Image's: `x = sigma * noise + (1 - sigma) * latent`. The network is
  fed `t = sigma` as a float32 number and multiplies it by 1000 itself (`Timesteps(scale=1000)`). The raw output is
  the velocity as it is (no sign flip, unlike Z-Image), so the denoised estimate is `x - sigma * output`.
* diffusers' own `forward` rounds the timestep to the dtype of the weights first (`timestep.to(hidden_states.dtype)`):
  in bfloat16 a sigma of 0.9029 becomes 0.9023, which shifts the highest frequency of the embedding by half a
  radian. `_forward` below hands the float32 number to the sinusoid and rounds only the 256-wide embedding (own
  choice); the blocks themselves are diffusers'.
* Token layout: the 2 x 2 patches of the target latent come first, then those of every reference latent in the
  order given (`[1, 16, h, w]` becomes `[1, ceil(h/2) * ceil(w/2), 64]`, the 64 values ordered (channel, row,
  column)). A latent with an odd side is padded by one element (circular; own choice), the padded row or column is
  cut from the output; reference pictures are rounded to multiples of 8 pixels, not 16, so odd latent sides are
  normal.
* Reference timestep zero (the files carry a marker tensor `__index_timestep_zero__`; diffusers calls it
  `zero_cond_t`): the reference tokens are modulated with the embedding of timestep 0, the target tokens with the
  real one. With no reference the network sees plain timesteps, which is what diffusers computes too (all tokens
  select the first half). The reference latents arrive normalised like the sampler's latent.
* Position numbers (RoPE), own rule: a picture of `h_len x w_len` tokens (`h_len = ceil(latent height / 2)`) gets
  rows `0 .. h_len - 1` minus `h_len // 2` and columns likewise, the first axis holds the picture's index (target 0,
  references 1, 2, ...) and no offset is added to a reference. diffusers' `QwenEmbedRope` centres with
  `-(h - h // 2) .. h // 2 - 1`, which is the same for even `h_len` and one position off for odd (1280 x 720
  pictures have `h_len = 45`). Text tokens start at `max(h_len // 2, w_len // 2)` of the TARGET only (diffusers uses
  the largest picture, references included) and carry the same number on all three axes. The angles are computed
  in float64 and rounded to complex64.
* The Lightning LoRA needs no guidance, so one forward pass per step with the prompt embedding is the whole model
  call. The prompt embedding has no padding, so no attention mask is passed.
* Weights: a single `.safetensors` file in diffusers' tensor names (bf16/fp16, or scaled fp8 through
  `engine.formats.fp8`) or a `.gguf` (block-quantized linear layers stay quantized as `QLinear`); the empty
  `__index_timestep_zero__` marker is dropped. The input latent is rounded to the weight dtype, the output is
  widened to float32.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import Tensor

from ...core.quant import materialize_quantized
from ...formats import fp8
from ...formats.ggml import GGUFFile
from ...sampling.flow import denoised_from_velocity
from ..zimage import assert_loaded, materialize

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "qwen_edit_transformer.json"
MARKER = "__index_timestep_zero__"  # the empty tensor that marks a file of the 2511 variant (reference timestep zero)
PREFIXES = ("model.diffusion_model.", "diffusion_model.")  # checkpoints that were merged into a bigger file
LATENT_CHANNELS = 16
PATCH = 2
ROPE_THETA = 10000
ROPE_AXES = (16, 56, 56)  # rotary width of the picture index, row and column axes (the config's `axes_dims_rope`)


def _build(config: Optional[Dict] = None):
    """A QwenImageTransformer2DModel on the meta device (default: the shipped Qwen-Image-Edit-2511 config)."""
    from diffusers import QwenImageTransformer2DModel

    if config is None:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    with torch.device("meta"):
        return QwenImageTransformer2DModel.from_config({k: v for k, v in config.items() if not k.startswith("_")})


def diffusers_name(name: str) -> Optional[str]:
    """The model's key for a file's key: a merged-checkpoint prefix is stripped, the empty marker is dropped (None)."""
    for prefix in PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return None if name == MARKER else name


def diffusers_tensors(checkpoint: Iterator[Tuple[str, Tensor]]) -> Iterator[Tuple[str, Tensor]]:
    """(name, tensor) pairs of a file with the names diffusers uses; a tensor may be a QTensor. Both kinds of
    file already carry diffusers' names, so this only strips a prefix and drops the marker."""
    for name, tensor in checkpoint:
        renamed = diffusers_name(name)
        if renamed is not None:
            yield renamed, tensor


def has_marker(path: str | Path) -> bool:
    """Whether the file carries the `__index_timestep_zero__` marker (header only; no weights are read)."""
    path = Path(path)
    if path.suffix.lower() == ".gguf":
        with GGUFFile(path) as gguf_file:
            return any(diffusers_name(name) is None for name in gguf_file.tensors)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return any(diffusers_name(key) is None for key in handle.keys())


def load_transformer(path: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16,
                     config: Optional[Dict] = None):
    """Build the transformer on the meta device and fill it from a `.safetensors` or `.gguf` file.

    A safetensors file (bf16/fp16, or fp8 with per-layer scales) is read and copied to `device` one tensor at a
    time. A GGUF keeps its block-quantized linear layers quantized (`QLinear`, bytes on `device`, expanded for
    every call) and expands the rest. `config` replaces the shipped config (tiny models in tests). A file with
    the `__index_timestep_zero__` marker needs a config with `zero_cond_t`; otherwise the references would be
    modulated like the target and the pictures would silently come out wrong."""
    model = _build(config)
    if has_marker(path) and not model.config.zero_cond_t:
        raise ValueError(f"{Path(path).name} is the reference-timestep-zero variant (marker {MARKER}), "
                         "but the config has zero_cond_t = false")
    if Path(path).suffix.lower() == ".gguf":
        with GGUFFile(path) as gguf_file:
            materialize_quantized(model, diffusers_tensors(iter(gguf_file.tensors.items())), device, dtype,
                                  "qwen transformer")
    else:
        materialize(model, diffusers_tensors(fp8.iter_tensors(path, dtype, "cpu")), device, dtype, "qwen transformer")
    assert_loaded(model, "qwen transformer")
    return model.eval()


# ----------------------------- tokens and position numbers -----------------------------

def token_grid(height: int, width: int) -> Tuple[int, int]:
    """Rows and columns of 2 x 2 patches of a latent: odd sides are padded to even first."""
    return (height + PATCH - 1) // PATCH, (width + PATCH - 1) // PATCH


def pack(latent: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
    """A latent [1, C, H, W] as tokens [1, rows * columns, C * 4] (values ordered channel, row, column) and the
    token grid (rows, columns). An odd side is first padded by one circular element."""
    _, channels, height, width = latent.shape
    if height % PATCH or width % PATCH:
        latent = F.pad(latent, (0, width % PATCH, 0, height % PATCH), mode="circular")
    rows, columns = latent.shape[-2] // PATCH, latent.shape[-1] // PATCH
    tokens = latent.reshape(1, channels, rows, PATCH, columns, PATCH).permute(0, 2, 4, 1, 3, 5)
    return tokens.reshape(1, rows * columns, channels * PATCH * PATCH), (rows, columns)


def unpack(tokens: Tensor, height: int, width: int) -> Tensor:
    """The inverse of `pack` for the picture of `height` x `width` latent elements; the padding is cut off."""
    rows, columns = token_grid(height, width)
    channels = tokens.shape[-1] // (PATCH * PATCH)
    latent = tokens.reshape(1, rows, columns, channels, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5)
    return latent.reshape(1, channels, rows * PATCH, columns * PATCH)[:, :, :height, :width]


def _picture_ids(index: int, rows: int, columns: int) -> Tensor:
    """Position numbers [rows * columns, 3] of one picture: its index, then centred rows and columns."""
    ids = torch.zeros(rows, columns, 3, dtype=torch.int64)
    ids[..., 0] = index
    ids[..., 1] = (torch.arange(rows) - rows // 2).reshape(-1, 1)
    ids[..., 2] = (torch.arange(columns) - columns // 2).reshape(1, -1)
    return ids.reshape(-1, 3)


def rope_ids(target: Tuple[int, int], references: Sequence[Tuple[int, int]],
             text_length: int) -> Tuple[Tensor, Tensor]:
    """The position numbers as int64 tensors: ([N, 3] for the picture tokens, target first and then each
    reference in order, [text_length, 3] for the text tokens). `target` and `references` are the latent sizes
    (height, width), not the token grids."""
    grids = [token_grid(*target)] + [token_grid(*size) for size in references]
    image_ids = torch.cat([_picture_ids(index, rows, columns) for index, (rows, columns) in enumerate(grids)])
    start = max(grids[0][0] // 2, grids[0][1] // 2)
    text_ids = torch.arange(start, start + text_length, dtype=torch.int64).reshape(-1, 1).repeat(1, 3)
    return image_ids, text_ids


def rope_frequencies(ids: Tensor, axes: Sequence[int] = ROPE_AXES, theta: float = ROPE_THETA) -> Tensor:
    """Rotary factors e^(i * position * omega) as complex64 [N, sum(axes) / 2] for int position numbers [N, 3].

    Angles are formed in float64 (an angle of 1000 radians keeps its low digits), and the
    factors of the three axes sit side by side in the order the attention reads them."""
    parts = []
    for axis, width in enumerate(axes):
        omega = 1.0 / (theta ** torch.linspace(0, (width - 2) / width, steps=width // 2, dtype=torch.float64))
        angle = ids[:, axis].to(torch.float64).reshape(-1, 1) * omega.reshape(1, -1)
        parts.append(torch.polar(torch.ones_like(angle), angle))
    return torch.cat(parts, dim=-1).to(torch.complex64)


# ----------------------------- the forward pass -----------------------------

def _compute_dtype(model) -> torch.dtype:
    """The dtype the weights compute in. `next(model.parameters())` would be the placeholder of a quantized layer
    (a meta float32), so the first real parameter is used."""
    return next(p.dtype for p in model.parameters() if not p.is_meta)


def _forward(model, tokens: Tensor, context: Tensor, timestep: Tensor, frequencies: Tuple[Tensor, Tensor],
             modulate_index: Optional[Tensor]) -> Tensor:
    """diffusers' `QwenImageTransformer2DModel.forward` with a float32 timestep and given rotary factors.

    `tokens` [1, N, 64] (target first), `context` [1, L, 3584], `timestep` [1] or [2] (the real one and zeros),
    `modulate_index` [1, N] with 0 for tokens that use the first timestep and 1 for those that use the second."""
    hidden = model.img_in(tokens)
    encoder = model.txt_in(model.txt_norm(context))
    temb = model.time_text_embed(timestep, hidden)
    for block in model.transformer_blocks:
        encoder, hidden = block(hidden_states=hidden, encoder_hidden_states=encoder, encoder_hidden_states_mask=None,
                                temb=temb, image_rotary_emb=frequencies, joint_attention_kwargs=None,
                                modulate_index=modulate_index)
    if modulate_index is not None:
        temb = temb.chunk(2, dim=0)[0]  # the output layer uses the real timestep for every token
    return model.proj_out(model.norm_out(hidden, temb))


def predict_velocity(model, latent: Tensor, sigma: Tensor, context: Tensor,
                     ref_latents: Sequence[Tensor] = ()) -> Tensor:
    """One forward pass: the velocity of the flow at noise level `sigma` (float32, shape of `latent`).

    `latent` is [1, 16, h, w] float32 in the sampler's space, `sigma` a scalar, `context` the prompt embedding
    [1, L, 3584], `ref_latents` the reference pictures as [1, 16, h_i, w_i] latents in the same normalised space
    (what `vae.encode` returns). Only the target's tokens come back, unpacked."""
    if latent.ndim != 4 or latent.shape[0] != 1 or latent.shape[1] != LATENT_CHANNELS:
        raise ValueError(f"the latent must be [1, {LATENT_CHANNELS}, h, w], got {tuple(latent.shape)}")
    if context.ndim != 3 or context.shape[0] != 1:
        raise ValueError(f"the prompt embedding must be [1, L, D], got {tuple(context.shape)}")
    for ref in ref_latents:
        if ref.ndim != 4 or ref.shape[0] != 1 or ref.shape[1] != LATENT_CHANNELS:
            raise ValueError(f"a reference latent must be [1, {LATENT_CHANNELS}, h, w], got {tuple(ref.shape)}")
    dtype, device = _compute_dtype(model), latent.device
    height, width = latent.shape[-2:]
    pieces = [pack(latent)[0]] + [pack(ref.to(latent.dtype))[0] for ref in ref_latents]
    tokens = torch.cat(pieces, dim=1).to(dtype)
    image_ids, text_ids = rope_ids((height, width), [tuple(ref.shape[-2:]) for ref in ref_latents], context.shape[1])
    frequencies = (rope_frequencies(image_ids, model.config.axes_dims_rope).to(device),
                   rope_frequencies(text_ids, model.config.axes_dims_rope).to(device))
    t = torch.as_tensor(sigma, dtype=torch.float32, device=device).reshape(1)
    modulate_index = None
    if model.config.zero_cond_t and ref_latents:
        t = torch.cat([t, t * 0])
        target_tokens = pieces[0].shape[1]
        modulate_index = torch.tensor([[0] * target_tokens + [1] * (tokens.shape[1] - target_tokens)],
                                      dtype=torch.int32, device=device)
    out = _forward(model, tokens, context.to(device=device, dtype=dtype), t, frequencies, modulate_index)
    return unpack(out[:, :pieces[0].shape[1]], height, width).float()


def predict_denoised(model, latent: Tensor, sigma: Tensor, context: Tensor,
                     ref_latents: Sequence[Tensor] = ()) -> Tensor:
    """The sampler's view of the model: the estimate of the clean latent, `x - sigma * velocity`."""
    sigma = torch.as_tensor(sigma, dtype=torch.float32)
    return denoised_from_velocity(latent, sigma.to(latent.device), predict_velocity(model, latent, sigma, context,
                                                                                   ref_latents))
