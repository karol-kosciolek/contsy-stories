# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Flux autoencoder used by Z-Image-Turbo: decoding.

File: Tongyi-MAI/Z-Image-Turbo vae/diffusion_pytorch_model.safetensors (bf16, diffusers names), loaded as it is.
Its tensors are Black Forest Labs' float32 Flux autoencoder (ae.safetensors, original latent-diffusion names)
rounded to bf16 to nearest, ties to even (checked tensor by tensor on 2026-11-15: all 244 equal). A file in the
original layout is still read and converted with diffusers' `convert_ldm_vae_checkpoint`.

Conventions (diffusers 0.42 `ZImagePipeline` and `AutoencoderKL`, with the numbers of the VAE config):

* The sampler works in "model space": the decoder gets `z / scaling_factor + shift_factor`
  (`z / 0.3611 + 0.1159`).
* The decoder output lives in [-1, 1]; it is mapped with `(x + 1) / 2`, clamped to [0, 1],
  multiplied by 255 and converted to uint8 by truncation (Contsy's own choice; diffusers rounds).
* The VAE runs in bf16 for pictures (the weights exactly as stored); a float32 load widens the bf16 weights
  exactly (the tests compare with it).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor

from . import assert_loaded, materialize

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "zimage_vae.json"

# The out-of-memory fallback decodes tiles of 512 pixels (64 latent pixels) overlapping by a quarter, in one pass.
# diffusers would tile at the config's sample_size (1024 pixels): nothing at all for pictures up to that size.
TILE_LATENT = 64
# A tensor that only the diffusers naming has (the original layout calls it `decoder.mid.attn_1.q.weight`).
DIFFUSERS_NAME = "decoder.mid_block.attentions.0.to_q.weight"


def _config(config: Optional[Dict]) -> Dict:
    if config is None:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {k: v for k, v in config.items() if not k.startswith("_")}


def load_vae(path: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16,
             config: Optional[Dict] = None):
    """AutoencoderKL from the official file (168 MB in bf16, so read at once), or from a file in the original
    latent-diffusion layout."""
    from diffusers import AutoencoderKL

    config = _config(config)
    with torch.device("meta"):
        model = AutoencoderKL.from_config(config)
    state = load_file(str(path))
    if DIFFUSERS_NAME not in state:
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

        state = convert_ldm_vae_checkpoint(state, config)
    materialize(model, state.items(), device, dtype, "vae")
    assert_loaded(model, "vae")
    return model.eval()


@torch.inference_mode()
def decode(vae, latent: Tensor, tiled: bool = False) -> Image.Image:
    """Decode a sampler latent [1, 16, H/8, W/8] into an RGB PIL image of size (W, H)."""
    z = latent.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
    z = z / vae.config.scaling_factor + vae.config.shift_factor
    if tiled:
        saved = (vae.tile_latent_min_size, vae.tile_sample_min_size)
        vae.tile_latent_min_size = TILE_LATENT
        vae.tile_sample_min_size = TILE_LATENT * 2 ** (len(vae.config.block_out_channels) - 1)
        vae.enable_tiling()
    try:
        pixels = vae.decode(z, return_dict=False)[0]
    finally:
        if tiled:
            vae.disable_tiling()
            vae.tile_latent_min_size, vae.tile_sample_min_size = saved
    pixels = ((pixels.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    array = (pixels[0] * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, "RGB")
