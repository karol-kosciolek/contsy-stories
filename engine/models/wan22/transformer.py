# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Wan 2.2 A14B diffusion transformers (high-noise and low-noise expert), T2V and I2V.

Conventions (Wan-Video/Wan2.2, Apache-2.0, and diffusers 0.42 `WanPipeline` / `WanImageToVideoPipeline`):

* The state is a velocity flow like Z-Image's: `x = sigma * noise + (1 - sigma) * latent`. The network
  is given `t = sigma * 1000` and returns the velocity as it is, so the denoised estimate is
  `x - sigma * output` (no sign flip, unlike Z-Image).
* Wan 2.2 I2V A14B has no image-embedding branch (the official config: `model_type` i2v, `in_dim` 36, no
  CLIP image encoder): the 20 extra input channels are the image conditioning (see `conditioning.py`),
  concatenated to the 16 noise channels.
* The file keeps the original Wan names; diffusers' converter renames them one key at a time.
* diffusers builds its rotary tables as non-persistent buffers, which a meta-device load cannot fill,
  so they are rebuilt after the weights are in.
* The attention modules run `attention.WanAttnProcessor` (diffusers' computation, attention through
  `core.attention`, so SageAttention can be switched on around a call), and the blocks `block.fused_class`
  (diffusers' forward, with the residual adds, layer norms and GELU in fused Triton passes where the Triton kernels
  are in use).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

import torch
from safetensors import safe_open
from torch import Tensor

from ...core.quant import materialize_quantized
from ...formats import fp8 as fp8_format
from ...formats.ggml import GGUFFile
from ...sampling.flow import denoised_from_velocity
from ..zimage import assert_loaded, materialize
from . import attention
from . import block as fused_block

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONFIG_FILES = {"t2v": "wan22_t2v_transformer.json", "i2v": "wan22_i2v_transformer.json"}
TIMESTEP_SCALE = 1000.0  # the SD3 flow's multiplier


def _build(kind: str = "t2v", config: Optional[Dict] = None):
    """A WanTransformer3DModel on the meta device (default: the shipped Wan 2.2 A14B config of `kind`)."""
    from diffusers import WanTransformer3DModel

    if config is None:
        config = json.loads((CONFIGS / CONFIG_FILES[kind]).read_text(encoding="utf-8"))
    with torch.device("meta"):
        return WanTransformer3DModel.from_config({k: v for k, v in config.items() if not k.startswith("_")})


def diffusers_tensors(checkpoint: Iterator[Tuple[str, Union[Tensor, object]]]) -> Iterator[Tuple[str, object]]:
    """Original-format (name, tensor) pairs to diffusers names, one entry at a time (a tensor may be a QTensor)."""
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

    for name, tensor in checkpoint:
        yield from convert_wan_transformer_to_diffusers({name: tensor}).items()


def _file_tensors(path: Path) -> Iterator[Tuple[str, Tensor]]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            yield key, f.get_tensor(key)


CHECK_BLOCKS = (0, 13, 26, 39)  # the blocks lora_merged_share samples (A14B has 40)


def lora_merged_share(merged: str | Path, base: str | Path, lora: str | Path, blocks=CHECK_BLOCKS) -> float:
    """The share of a sample of `merged`'s values that equal `base` with the full parameter changes of `lora` added:
    the biases and norm weights of `blocks` that the LoRA changes (`.diff_b`, `.diff`), rounded to bfloat16 as merged
    files store them. All three files use the original Wan key names; only these small tensors are read.

    It tells whether an expert with a distillation merged in (the FP8 experts) holds the distillation of a LoRA. With
    the default blocks, LightX2V's low-noise FP8 expert scores 0.966 against the Q6_K GGUF and the `_1022` LoRA (0.43
    for the base alone; the misses are the GGUF's rounding of the base), while LightX2V's high-noise expert scores
    0.48, below the 0.51 of the base alone: it holds another LightX2V distillation. The high-noise expert that
    `convert.merge_lora_to_fp8` makes from the Q6_K GGUF and the `_1022` LoRA scores 1.0 (in all 40 blocks too).
    `base` is a GGUF or safetensors file. A LoRA without such changes (only low-rank matrices: the text-to-video
    `_1217` LoRAs) is checked on sampled rows of its matrices instead (`convert.merged_rows_share`)."""
    prefix = "diffusion_model."
    wanted = tuple(f"blocks.{index}." for index in blocks)
    with safe_open(str(lora), framework="pt", device="cpu") as changes:
        keys = {}
        full_changes = False
        for key in changes.keys():
            stem = key.removeprefix(prefix)
            for suffix, leaf in ((".diff_b", ".bias"), (".diff", ".weight")):
                if stem.endswith(suffix):
                    full_changes = True
                    if stem.startswith(wanted):
                        keys[stem[:-len(suffix)] + leaf] = key
        deltas = {name: changes.get_tensor(key).float() for name, key in keys.items()}
    if not full_changes:  # a LoRA of low-rank matrices only (the text-to-video _1217 LoRAs): sampled rows merged again
        from .convert import merged_rows_share

        return merged_rows_share(merged, base, lora, blocks)
    with safe_open(str(merged), framework="pt", device="cpu") as merged_file:
        present = set(merged_file.keys())
        values = {name: merged_file.get_tensor(name).float() for name in deltas if name in present}
    if not values:
        raise ValueError(f"{Path(lora).name} changes no bias or norm of {Path(merged).name} in blocks {blocks}")
    if Path(base).suffix.lower() == ".gguf":
        with GGUFFile(base) as gguf_file:
            originals = {name: gguf_file.tensors[name].to(torch.float32, "cpu") for name in values}
    else:
        with safe_open(str(base), framework="pt", device="cpu") as base_file:
            originals = {name: base_file.get_tensor(name).float() for name in values}
    equal = total = 0
    for name, value in values.items():
        expected = (originals[name].reshape(value.shape) + deltas[name].reshape(value.shape)).to(torch.bfloat16)
        equal += int((value == expected.float()).sum())
        total += value.numel()
    return equal / total


def _rebuild_rope(model, device: torch.device | str) -> None:
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

    old = model.rope
    model.rope = WanRotaryPosEmbed(old.attention_head_dim, old.patch_size, old.max_seq_len).to(device)


def load_transformer(path: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16,
                     kind: str = "t2v", config: Optional[Dict] = None):
    """Build the transformer on the meta device and fill it from a file in the original Wan key layout.

    A `.safetensors` file (bf16/fp16, or fp8 without per-layer scales) is read, renamed and copied to
    `device` one tensor at a time. A `.gguf` file keeps its block-quantized linear layers quantized
    (`QLinear`, bytes on `device`, expanded per call) and expands the rest. A safetensors file with FP8
    weights and per-channel `weight_scale` tensors (LightX2V's scaled FP8 experts) keeps those layers in FP8
    (`FP8Linear`, FP8 tensor-core products where the GPU has them) and casts the rest to `dtype`. `kind` picks
    the shipped config ("t2v" 16 input channels, "i2v" 36); `config` replaces it (tiny models in tests)."""
    model = _build(kind, config)
    if Path(path).suffix.lower() == ".gguf":
        with GGUFFile(path) as gguf_file:
            materialize_quantized(model, diffusers_tensors(iter(gguf_file.tensors.items())), device, dtype,
                                  "wan transformer")
    elif fp8_format.channel_scaled(path):
        materialize_quantized(model, diffusers_tensors(fp8_format.iter_channel_scaled(path)), device, dtype,
                              "wan transformer")
    else:
        materialize(model, diffusers_tensors(_file_tensors(Path(path))), device, dtype, "wan transformer")
    _rebuild_rope(model, device)
    attention.install(model)
    fused_block.install(model)
    assert_loaded(model, "wan transformer")
    return model.eval()


def predict_velocity(model, latent: Tensor, sigma: Tensor, context: Tensor, concat: Optional[Tensor] = None) -> Tensor:
    """One forward pass: the velocity at noise level `sigma` (float32, shape of `latent`).

    `latent` is [1, 16, T, H, W] float32, `sigma` a scalar tensor, `context` the text embedding [1, L, D];
    `concat` is the I2V conditioning [1, 20, T, H, W] appended to the channels (None for T2V; an I2V
    network without it gets zeros)."""
    dtype = next(model.parameters()).dtype
    device = latent.device
    missing = model.config.in_channels - latent.shape[1] - (0 if concat is None else concat.shape[1])
    if missing > 0:  # a text-to-video call on the image-to-video network gets zero conditioning channels
        zeros = torch.zeros(latent.shape[0], missing, *latent.shape[2:], dtype=latent.dtype, device=device)
        concat = zeros if concat is None else torch.cat([concat.to(latent.dtype), zeros], dim=1)
    x = latent if concat is None else torch.cat([latent, concat.to(latent.dtype)], dim=1)
    t = (sigma.to(device=device, dtype=torch.float32) * TIMESTEP_SCALE).reshape(1)
    out = model(x.to(dtype), t, context.to(device=device, dtype=dtype), return_dict=False)[0]
    return out.float()


def predict_denoised(model, latent: Tensor, sigma: Tensor, context: Tensor, concat: Optional[Tensor] = None) -> Tensor:
    """The sampler's view of the model: the estimate of the clean latent, `x - sigma * velocity`."""
    return denoised_from_velocity(latent, sigma.to(latent.device), predict_velocity(model, latent, sigma, context, concat))
