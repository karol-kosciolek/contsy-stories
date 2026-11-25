# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Image-to-video conditioning of Wan 2.2 (Wan-Video/Wan2.2 `wan/image2video.py`, Apache-2.0).

The start picture is scaled to the video size, put in front of gray padding frames (0.5 in [0, 1], the zeros of
the official [-1, 1] range), and the whole sequence goes through the VAE encoder. The network then gets, next to
its 16 noise channels, 4 mask channels (1 where a frame is known, the first frame's value repeated four times) and
the 16 latent channels of that sequence, normalised with the VAE's per-channel mean and std. Order: mask first.
The scaling is Contsy's own choice: a centre crop to the target aspect ratio, then a bilinear resize (the official
code resizes bicubically to a size that keeps the picture's aspect ratio).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

TEMPORAL_SCALE = 4  # one latent frame per four pixel frames (after the first)


def center_scale(frames: Tensor, width: int, height: int) -> Tensor:
    """[N, 3, H, W] to [N, 3, height, width]: crop to the target aspect ratio from the centre, then
    bilinear resize."""
    old_height, old_width = frames.shape[-2], frames.shape[-1]
    old_aspect, new_aspect = old_width / old_height, width / height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    cropped = frames.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    return F.interpolate(cropped, size=(height, width), mode="bilinear")


def start_sequence(start: Tensor, length: int, width: int, height: int) -> Tensor:
    """Pixel frames [length, 3, height, width] in 0..1: the scaled start picture(s) first, then gray."""
    start = center_scale(start.reshape(-1, *start.shape[-3:])[:length].float(), width, height)
    frames = torch.full((length, 3, height, width), 0.5)
    frames[:start.shape[0]] = start
    return frames


def known_latent_frames(start_count: int = 1) -> int:
    """Latent frames covered by `start_count` known pixel frames."""
    return (start_count - 1) // TEMPORAL_SCALE + 1


def i2v_concat(image_latent: Tensor, known: int, mean: Sequence[float], std: Sequence[float]) -> Tensor:
    """The 20 extra channels [1, 20, T, h, w] from the VAE latent [1, 16, T, h, w] of the start sequence.

    `known` latent frames are given (mask 1), the rest are generated (mask 0)."""
    mean = torch.tensor(mean, dtype=image_latent.dtype, device=image_latent.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(std, dtype=image_latent.dtype, device=image_latent.device).view(1, -1, 1, 1, 1)
    normalised = (image_latent - mean) / std
    mask = torch.zeros(image_latent.shape[0], 1, *image_latent.shape[2:], dtype=image_latent.dtype,
                       device=image_latent.device)
    mask[:, :, :known] = 1.0
    return torch.cat((mask.repeat(1, 4, 1, 1, 1), normalised), dim=1)
