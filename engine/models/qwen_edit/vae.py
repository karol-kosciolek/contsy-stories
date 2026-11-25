# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Qwen-Image autoencoder: reference pictures in, finished pictures out.

The network is diffusers' `AutoencoderKLQwenImage` (Apache-2.0, The HuggingFace Team), imported and not copied. The
file is the official `vae/diffusion_pytorch_model.safetensors` of Qwen/Qwen-Image-Edit-2511 (Apache-2.0): 194 bfloat16
tensors that already carry the names of `AutoencoderKLQwenImage`, so they are loaded as they are.

Conventions (diffusers 0.42 `QwenImageEditPlusPipeline` and `AutoencoderKLQwenImage`):

* 16 latent channels, base width 96, factor 8: the architecture is Wan 2.1's, the decoder was fine-tuned for
  text-rich pictures.
* Transitional: the previous catalog's `qwen_image_vae.safetensors` holds the same 194 bfloat16 tensors in the
  original Wan layout (`conv1`, `encoder.downsamples.N...`). Its names go through diffusers'
  `convert_wan_vae_to_diffusers` (the Wan 2.2 adapter's `diffusers_tensors`), which gives exactly the official file's
  tensors (194 of 194 bit-identical, checked 2026-11-15). A running Studio that started on the previous catalog
  still hands that file to this module until it restarts; remove this path with the other previous-catalog paths
  after that restart.
* Pixels are [0, 1] floats; the network sees `x * 2 - 1`. A picture is first cropped centrally to multiples of 8
  (`crop_to_multiple`, a no-op for the reference pictures, which are already multiples of 8) and treated as a
  one-frame video `[1, 3, 1, H, W]`.
* Encoding gives the mean of the posterior (`latent_dist.mode()`, no sampling). The sampler works in a normalised
  space, `(z - latents_mean) / latents_std` per channel, as the pipeline normalises its image latents, so `encode`
  returns the normalised latent: it is what the transformer takes as a reference.
* `decode` takes the sampler's normalised latent, undoes the normalisation (`z * std + mean`), decodes, maps
  `(x + 1) / 2` and clamps to [0, 1]. The statistics are those of the shipped config.
* Out-of-memory fallback: diffusers' tiling (256-pixel tiles every 192 pixels, blended over the 64 pixels they share).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from torch import Tensor

from ..wan22.vae import diffusers_tensors
from ..zimage import assert_loaded, materialize

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "qwen_vae.json"
SCALE = 8  # pixels per latent element


def _config(config: Optional[Dict]) -> Dict:
    if config is None:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {k: v for k, v in config.items() if not k.startswith("_")}


def _file_tensors(path: Path) -> Iterator[Tuple[str, Tensor]]:
    """(name, tensor) pairs with the names of `AutoencoderKLQwenImage`, one tensor at a time."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        tensors = ((key, handle.get_tensor(key)) for key in keys)
        if "conv1.weight" in keys:  # the previous catalog's file in the original Wan layout (transitional)
            tensors = diffusers_tensors(tensors)
        yield from tensors


def load_vae(path: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16,
             config: Optional[Dict] = None):
    """AutoencoderKLQwenImage on the meta device, filled from the file one tensor at a time (either layout)."""
    from diffusers import AutoencoderKLQwenImage

    with torch.device("meta"):
        model = AutoencoderKLQwenImage.from_config(_config(config))
    materialize(model, _file_tensors(Path(path)), device, dtype, "qwen vae")
    assert_loaded(model, "qwen vae")
    return model.eval()


def _stats(vae, like: Tensor) -> Tuple[Tensor, Tensor]:
    mean = torch.tensor(vae.config.latents_mean, dtype=like.dtype, device=like.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=like.dtype, device=like.device).view(1, -1, 1, 1, 1)
    return mean, std


def _pixels(picture: Union[Image.Image, Tensor]) -> Tensor:
    """A picture as float32 [1, 3, H, W] in 0..1."""
    if isinstance(picture, Image.Image):
        array = np.array(picture.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)[None]
    if picture.ndim != 4 or picture.shape[0] != 1 or picture.shape[1] != 3:
        raise ValueError(f"a picture tensor must be [1, 3, H, W], got {tuple(picture.shape)}")
    return picture.float()


def crop_to_multiple(pixels: Tensor, multiple: int = SCALE) -> Tensor:
    """Each side is cut centrally down to a multiple (the odd pixel goes to the end)."""
    height, width = pixels.shape[-2:]
    if height < multiple or width < multiple:
        raise ValueError(f"the picture must be at least {multiple} pixels on each side, got {width}x{height}")
    rows, columns = height // multiple * multiple, width // multiple * multiple
    top, left = (height % multiple) // 2, (width % multiple) // 2
    return pixels[..., top:top + rows, left:left + columns]


@torch.inference_mode()
def encode(vae, picture: Union[Image.Image, Tensor], tiled: bool = False) -> Tensor:
    """A picture (PIL image or float tensor [1, 3, H, W] in 0..1) to a normalised sampler latent [1, 16, H/8, W/8]
    in float32 on the device of the VAE: the mean of the posterior, then `(z - mean) / std`.
    `tiled` encodes in overlapping tiles for pictures that do not fit the memory."""
    parameter = next(vae.parameters())
    pixels = crop_to_multiple(_pixels(picture))
    video = (pixels.unsqueeze(2) * 2.0 - 1.0).to(device=parameter.device, dtype=parameter.dtype)
    if tiled:
        vae.enable_tiling()
    try:
        raw = vae.encode(video).latent_dist.mode().float()
    finally:
        if tiled:
            vae.disable_tiling()
    mean, std = _stats(vae, raw)
    return ((raw - mean) / std).squeeze(2)


@torch.inference_mode()
def decode(vae, latent: Tensor, tiled: bool = False) -> Tensor:
    """The sampler's latent [1, 16, h, w] to a picture [1, 3, 8h, 8w] in 0..1 (float32, on the CPU).
    `tiled` decodes in overlapping tiles for pictures that do not fit the memory."""
    parameter = next(vae.parameters())
    z = latent.to(device=parameter.device, dtype=torch.float32).unsqueeze(2)
    mean, std = _stats(vae, z)
    z = (z * std + mean).to(parameter.dtype)
    if tiled:
        vae.enable_tiling()
    try:
        video = vae.decode(z, return_dict=False)[0]
    finally:
        if tiled:
            vae.disable_tiling()
    return ((video[:, :, 0].float() + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
