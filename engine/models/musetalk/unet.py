# SPDX-FileCopyrightText: 2024 Tencent Music Entertainment Group
# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only AND MIT
"""The MuseTalk 1.5 network: a Stable-Diffusion-1.x UNet with 8 input channels and Whisper cross-attention.

Upstream (TMElyralab/MuseTalk, MIT) builds a diffusers `UNet2DConditionModel` from `musetalkV15/musetalk.json`
and loads `musetalkV15/unet.pth` into it. Our own SD-1.x UNet (`engine.models.sd_unet`) builds the same parameter
tree: the 686 names and shapes of the checkpoint match it exactly.

Conventions of the reference that change the output if they are "fixed":

* One forward at timestep 0 (`torch.tensor([0])`); MuseTalk is not sampled, it predicts the clean latent.
* Input channels: 4 latent channels of the face with its lower half blacked out (masked), then 4 of the
  unmasked reference face. Output: the 4 latent channels of the face.
* `encoder_hidden_states` is the Whisper window of the frame, [N, 50, 384], with the sinusoidal position
  table of `audio.positions` already added (the table is code, not a checkpoint tensor).
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import torch
from torch import Tensor

from ..sd_unet import MID_BLOCK, SDUNet
from ..zimage import assert_loaded, materialize

#: musetalkV15/musetalk.json at TMElyralab/MuseTalk 2bcb936e (748 bytes), a diffusers UNet2DConditionModel config.
MUSETALK_JSON: Dict = {
    "_class_name": "UNet2DConditionModel",
    "_diffusers_version": "0.6.0.dev0",
    "act_fn": "silu",
    "attention_head_dim": 8,
    "block_out_channels": [320, 640, 1280, 1280],
    "center_input_sample": False,
    "cross_attention_dim": 384,
    "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
    "downsample_padding": 1,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 8,
    "layers_per_block": 2,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "sample_size": 64,
    "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
}

_BLOCKS = {"CrossAttnDownBlock2D", "DownBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"}
_PASSED = {"act_fn", "attention_head_dim", "block_out_channels", "center_input_sample", "cross_attention_dim",
           "downsample_padding", "flip_sin_to_cos", "freq_shift", "in_channels", "layers_per_block",
           "mid_block_scale_factor", "norm_eps", "norm_num_groups", "out_channels", "sample_size"}


def unet_config(diffusers_config: Mapping) -> Dict:
    """Our UNet's keyword arguments for a diffusers UNet2DConditionModel config of the SD-1.x family.

    Every key must be understood: an unknown option would otherwise build a different network silently."""
    config: Dict = {}
    for key, value in diffusers_config.items():
        if key.startswith("_"):
            continue
        if key in ("down_block_types", "up_block_types"):
            unknown = [kind for kind in value if kind not in _BLOCKS]
            if unknown:
                raise ValueError(f"MuseTalk UNet: unsupported block types {unknown}")
            config[key] = list(value)
        elif key == "mid_block_type":
            if value != MID_BLOCK:
                raise ValueError(f"MuseTalk UNet: unsupported mid block {value!r}")
            config[key] = value
        elif key in _PASSED:
            config[key] = list(value) if isinstance(value, (list, tuple)) else value
        else:
            raise ValueError(f"MuseTalk UNet: config option {key!r} is not supported")
    return config


UNET_CONFIG: Dict = unet_config(MUSETALK_JSON)


def build_unet(config: Optional[Mapping] = None) -> SDUNet:
    """The network with random weights (on the current default device); `config` is our class's kwargs."""
    return SDUNet(**dict(UNET_CONFIG if config is None else config))


def load_unet(path: Union[str, Path], device: Union[torch.device, str] = "cpu", dtype: torch.dtype = torch.float16,
              config: Optional[Mapping] = None) -> SDUNet:
    """The UNet from `unet.pth`, a plain float32 state_dict pickle (3.4 GB).

    The pickle is read with `weights_only=True` (only tensors and an OrderedDict are allowed) and
    memory-mapped; each tensor is copied to `device` and cast to `dtype` one at a time, so the float32
    file is never held in memory as a whole. Upstream calls `load_state_dict` (strict); here every name
    and shape must match as well, and a missing, extra or reshaped tensor raises instead of leaving a
    layer random. `config` is for tests (tiny networks); the default is MuseTalk 1.5."""
    with torch.device("meta"):
        model = build_unet(config)
    checkpoint = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
    try:
        if not isinstance(checkpoint, Mapping) or not all(isinstance(t, Tensor) for t in checkpoint.values()):
            raise ValueError(f"{path} is not a MuseTalk UNet state_dict")
        materialize(model, checkpoint.items(), device, dtype, "MuseTalk UNet")
    finally:
        checkpoint = None
        gc.collect()  # the memory map of the file is released with its last tensor
    assert_loaded(model, "MuseTalk UNet")
    return model.eval()


def predict(network: SDUNet, latents: Tensor, audio: Tensor) -> Tensor:
    """The face latent [N, 4, h, w] for the masked and reference latents [N, 8, h, w] and audio [N, 50, 384]."""
    if latents.ndim != 4 or audio.ndim != 3 or len(latents) != len(audio):
        raise ValueError("MuseTalk needs latents [N, 8, h, w] and one audio window [N, tokens, 384] per frame")
    return network(latents, 0, audio)
