# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Wan 2.1 video autoencoder that Wan 2.2 A14B uses.

The weights are Wan-AI's own: `vae/diffusion_pytorch_model.safetensors` of Wan-AI/Wan2.2-T2V-A14B-Diffusers (float32,
diffusers names; the I2V repository holds the same file), filled into `AutoencoderKLWan` as they are and cast to the
run dtype one tensor at a time. A file in the original Wan layout (the names of Wan-AI's `Wan2.1_VAE.pth`) is still
accepted: its names go through diffusers' `convert_wan_vae_to_diffusers`. Rounded to bfloat16 (the product runs the VAE
in bfloat16), the official float32 tensors are bit-identical to the bfloat16 file the catalog used before (194 of 194
tensors, checked 2026-11-15), so the decoded pictures are the same.

Conventions (Wan-Video/Wan2.2 `wan/modules/vae2_1.py`, Apache-2.0, and diffusers 0.42 `AutoencoderKLWan`):

* Pixels are [0, 1] floats; the network sees `x * 2 - 1`. Video is [N, 3, H, W] here and [1, 3, N, H, W]
  inside the network; N should be 4k + 1 (one latent frame for the first picture, one per four after).
* Encoding gives the mean of the posterior (no sampling). The sampler works in a normalised space,
  `(z - mean) / std` per channel (`latents_mean` / `latents_std` of the config); `encode` returns the raw
  latent, because the image-to-video conditioning normalises it itself.
* `decode` takes the sampler's normalised latent, undoes the normalisation (`z * std + mean`), decodes,
  maps `(x + 1) / 2` and clamps to [0, 1].
* The network runs causal 3D convolutions with feature caches, frame group by frame group (diffusers'
  `AutoencoderKLWan`). Where the Triton kernels are in use (`core.attention.use("sage")`, which the video pipeline
  also sets around the VAE) its RMS norms and the SiLUs after them are one pass each (`fuse_norms`) and the
  convolutions pad inside cuDNN (`fuse_convs`); on a GPU the convolution weights are channels-last (`channels_last`).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import Tensor

from ...core import attention
from ...kernels import norms
from ..zimage import assert_loaded, materialize

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "wan_vae.json"


def _config(config: Optional[Dict]) -> Dict:
    if config is None:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {k: v for k, v in config.items() if not k.startswith("_")}


def diffusers_tensors(checkpoint: Iterator[Tuple[str, Tensor]]) -> Iterator[Tuple[str, Tensor]]:
    """Original-format (name, tensor) pairs to diffusers names, one entry at a time."""
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

    for name, tensor in checkpoint:
        yield from convert_wan_vae_to_diffusers({name: tensor}).items()


def _file_tensors(path: Path) -> Iterator[Tuple[str, Tensor]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            yield key, handle.get_tensor(key)


def diffusers_layout(path: str | Path) -> bool:
    """True for a file with diffusers' names (the official Wan-AI file), False for the original Wan layout.

    The 1x1 convolutions around the latent tell them apart: `quant_conv` / `post_quant_conv` in diffusers,
    `conv1` / `conv2` in the original layout."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    if "quant_conv.weight" in keys:
        return True
    if "conv1.weight" in keys:
        return False
    raise ValueError(f"{Path(path).name}: not a Wan 2.1 VAE (neither diffusers nor original names)")


def load_vae(path: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16,
             config: Optional[Dict] = None):
    """AutoencoderKLWan on the meta device, filled from the file one tensor at a time (either layout), with the fused
    norm-and-SiLU passes (`fuse_norms`) and the convolutions' padding inside cuDNN (`fuse_convs`), and on a GPU with
    channels-last convolution weights (`channels_last`). Measured on the start picture of a 848x480, 97-frame scene
    (RTX 4070 Ti Super): encoding 6.5 -> 4.1 s, decoding 10.9 -> 7.0 s; the decoded frames are 60 dB (PSNR) from
    diffusers' own forward, about as far as cuDNN's other algorithms (cudnn.benchmark) put them."""
    from diffusers import AutoencoderKLWan

    cfg = _config(config)
    with torch.device("meta"):
        model = AutoencoderKLWan.from_config(cfg)
    tensors = _file_tensors(Path(path))
    if not diffusers_layout(path):
        tensors = diffusers_tensors(tensors)
    materialize(model, tensors, device, dtype, "wan vae")
    assert_loaded(model, "wan vae")
    fuse_norms(model)
    fuse_convs(model)
    if torch.device(device).type == "cuda":
        channels_last(model)
    return model.eval()


def channels_last(model) -> None:
    """Store the 3-D convolutions' weights channels-last: cuDNN then keeps the video features channels-last from one
    convolution to the next instead of converting every input and output (about 1.2 s of an 848x480, 97-frame decode).
    Moving the model between devices keeps the layout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d):
            module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last_3d)


@lru_cache(maxsize=None)
def _fused_norm_class(base: type) -> type:
    """`base` (diffusers' WanRMS_norm) that computes its norm, and the SiLU after it where `silu` is set, in one
    Triton pass (`kernels.norms.rms_channels`) where the Triton kernels are in use (`core.attention.use("sage")`, as
    for the transformer's fused kernels) and the input fits; anywhere else diffusers' forward, then `F.silu`."""

    class FusedRMSNorm(base):
        silu = False

        def forward(self, x: Tensor) -> Tensor:
            if (attention.current() == attention.SAGE and self.channel_first and isinstance(self.bias, float)
                    and self.bias == 0.0 and norms.rms_channels_fits(x, self.gamma)):
                return norms.rms_channels(x, self.gamma, self.scale, self.silu, use_triton=True)
            out = super().forward(x)
            return F.silu(out) if self.silu else out

    FusedRMSNorm.__name__ = FusedRMSNorm.__qualname__ = f"Fused{base.__name__}"
    return FusedRMSNorm


@lru_cache(maxsize=None)
def _padded_conv_class(base: type) -> type:
    """`base` (diffusers' WanCausalConv3d) that lets cuDNN add the zero border of the frames instead of `F.pad`
    copying every input into a larger zero tensor first (a fill and a copy per convolution, about 0.6 s of an 848x480
    decode). The causal time padding (the cached frames, zeros before the first one) is unchanged. Under
    `core.attention.use("sage")` on CUDA, like the other fused paths; anywhere else diffusers' forward."""

    class PaddedCausalConv3d(base):
        def forward(self, x: Tensor, cache_x: Optional[Tensor] = None) -> Tensor:
            if attention.current() != attention.SAGE or not x.is_cuda:
                return super().forward(x, cache_x)
            time = self._padding[4]
            if cache_x is not None and time > 0:
                x = torch.cat([cache_x.to(x.device), x], dim=2)
                time -= cache_x.shape[2]
            if time > 0:
                x = F.pad(x, (0, 0, 0, 0, time, 0))
            elif time < 0:
                x = x[:, :, -time:]
            return F.conv3d(x, self.weight, self.bias, self.stride, (0, self._padding[2], self._padding[0]),
                            self.dilation, self.groups)

    PaddedCausalConv3d.__name__ = PaddedCausalConv3d.__qualname__ = f"Padded{base.__name__}"
    return PaddedCausalConv3d


def fuse_convs(model) -> int:
    """Give every causal convolution of the autoencoder the padding inside cuDNN (`_padded_conv_class`); returns how
    many there were."""
    from diffusers.models.autoencoders import autoencoder_kl_wan as wan

    count = 0
    for module in model.modules():
        if isinstance(module, wan.WanCausalConv3d) and not type(module).__name__.startswith("Padded"):
            if module._padding[0] != module._padding[1] or module._padding[2] != module._padding[3] \
                    or module._padding[5] != 0:
                continue  # an asymmetric border: keep F.pad
            module.__class__ = _padded_conv_class(type(module))
            count += 1
    return count


def fuse_norms(model) -> int:
    """Give every WanRMS_norm of the autoencoder the fused forward; where a norm is always followed by the SiLU of its
    module (the residual blocks, the heads of the encoder and the decoder), the norm applies it and the module's
    `nonlinearity` becomes an identity. Returns how many norms take the SiLU."""
    from diffusers.models.autoencoders import autoencoder_kl_wan as wan

    for module in model.modules():
        if isinstance(module, wan.WanRMS_norm) and not type(module).__name__.startswith("Fused"):
            module.__class__ = _fused_norm_class(type(module))
    owners = ((wan.WanResidualBlock, ("norm1", "norm2")), (wan.WanEncoder3d, ("norm_out",)),
              (wan.WanDecoder3d, ("norm_out",)))
    fused = 0
    for module in model.modules():
        for kind, names in owners:
            if isinstance(module, kind) and isinstance(getattr(module, "nonlinearity", None), torch.nn.SiLU):
                for name in names:
                    getattr(module, name).silu = True
                    fused += 1
                module.nonlinearity = torch.nn.Identity()
    return fused


def _stats(vae, like: Tensor) -> Tuple[Tensor, Tensor]:
    mean = torch.tensor(vae.config.latents_mean, dtype=like.dtype, device=like.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=like.dtype, device=like.device).view(1, -1, 1, 1, 1)
    return mean, std


def _encode_settled(vae, video: Tensor, computed: int) -> Tensor:
    """diffusers' `AutoencoderKLWan._encode` (one latent frame per chunk of four frames after the first, through the
    causal feature caches) for the first `computed` latent frames only; the last of them stands for the remaining
    ones. Returns the posterior mean."""
    total = 1 + (video.shape[2] - 1) // 4
    vae.clear_cache()
    try:
        outputs = []
        for index in range(computed):
            vae._enc_conv_idx = [0]
            chunk = video[:, :, :1] if index == 0 else video[:, :, 1 + 4 * (index - 1):1 + 4 * index]
            outputs.append(vae.encoder(chunk, feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx))
        encoded = vae.quant_conv(torch.cat(outputs + [outputs[-1]] * (total - computed), dim=2))
    finally:
        vae.clear_cache()
    return torch.chunk(encoded, 2, dim=1)[0]


@torch.inference_mode()
def encode(vae, frames: Tensor, tiled: bool = False, computed: Optional[int] = None) -> Tensor:
    """Pixel frames [N, 3, H, W] in 0..1 to the raw latent [1, 16, T, H/8, W/8] (float32): the posterior mean.

    `computed` encodes only the first `computed` latent frames and repeats the last of them for the rest: for a
    sequence whose frames after the first few are all the same (the gray padding of the image-to-video conditioning)
    the causal encoder's output settles to within bfloat16 rounding after a few frames."""
    parameter = next(vae.parameters())
    video = (frames.permute(1, 0, 2, 3)[None].float() * 2.0 - 1.0).to(device=parameter.device, dtype=parameter.dtype)
    if (computed is not None and not tiled and 0 < computed < 1 + (video.shape[2] - 1) // 4
            and getattr(vae.config, "patch_size", None) is None and hasattr(vae, "clear_cache")):
        return _encode_settled(vae, video, computed).float()
    if tiled:
        vae.enable_tiling()
    try:
        return vae.encode(video).latent_dist.mode().float()
    finally:
        if tiled:
            vae.disable_tiling()


@torch.inference_mode()
def decode(vae, latent: Tensor, tiled: bool = False) -> Tensor:
    """The sampler's latent [1, 16, T, h, w] to pixel frames [N, 3, H, W] in 0..1 (float32, on the CPU)."""
    parameter = next(vae.parameters())
    z = latent.to(device=parameter.device, dtype=torch.float32)
    mean, std = _stats(vae, z)
    z = (z * std + mean).to(parameter.dtype)
    if tiled:
        vae.enable_tiling()
    try:
        video = vae.decode(z, return_dict=False)[0]
    finally:
        if tiled:
            vae.disable_tiling()
    return ((video[0].float() + 1.0) / 2.0).clamp(0.0, 1.0).permute(1, 0, 2, 3).contiguous().cpu()
