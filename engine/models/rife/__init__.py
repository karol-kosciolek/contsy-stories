# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-FileCopyrightText: 2021 hzwer
# SPDX-License-Identifier: GPL-3.0-only AND MIT
# The network is RIFE 4.7, the architecture of the rife49.pth weights, as published in Practical-RIFE
# (https://github.com/hzwer/Practical-RIFE, MIT; IFNet_HDv3 of the 4.7 release and model/warplayer.py).
# Written for Contsy from that architecture: inference only, padding to multiples of 64, a cached sampling
# grid per device and size, and a timestamp per batch item. Text in LICENSES/MIT.txt. See NOTICE and
# THIRD_PARTY_NOTICES.md.
"""RIFE frame interpolation: the in-between frame of two pictures at any moment in [0, 1].

Wan renders 16 fps and the product delivers exactly 30: every output frame gets its own timestamp
on the output clock (`engine.media.retime.frame_rate_plan`), so frames are interpolated at arbitrary
fractions instead of doubling and dropping. The network runs in float32 at scale 1.0, one pass per
frame (Practical-RIFE's optional ensemble pass, which averages with the reversed pair, is not used).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...media.retime import frame_rate_plan

SCALES = (8, 4, 2, 1)  # the coarse-to-fine scale of the four blocks at scale_factor 1.0
PAD = 64  # the pictures are padded to multiples of this inside the network
_GRIDS: Dict[Tuple[str, int, int, int], Tensor] = {}


def release_grids() -> None:
    """Drop the grids held on the compute device after interpolation."""
    _GRIDS.clear()


def _resize(x: Tensor, factor: float) -> Tensor:
    return F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)


def _halving(inputs: int, outputs: int) -> nn.Sequential:
    """A stride-2 3x3 convolution followed by LeakyReLU(0.2)."""
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel_size=3, stride=2, padding=1),
                         nn.LeakyReLU(negative_slope=0.2, inplace=True))


class ResConv(nn.Module):
    """A 3x3 convolution added to its input with a learned gain per channel (`beta`), then LeakyReLU(0.2)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """One refinement level: at 1/scale of the size, predicts four flow channels (picture 0 and picture 1, x and y,
    in pixels) and one blend logit."""

    def __init__(self, inputs: int, channels: int):
        super().__init__()
        self.conv0 = nn.Sequential(_halving(inputs, channels // 2), _halving(channels // 2, channels))
        self.convblock = nn.Sequential(*(ResConv(channels) for _ in range(8)))
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(channels, 4 * 6, kernel_size=4, stride=2, padding=1),
                                      nn.PixelShuffle(2))

    def forward(self, x: Tensor, flow: Optional[Tensor], scale: float) -> Tuple[Tensor, Tensor]:
        x = _resize(x, 1.0 / scale)
        if flow is not None:  # the flow so far, in the pixels of the smaller size
            x = torch.cat((x, _resize(flow, 1.0 / scale) / scale), dim=1)
        predicted = _resize(self.lastconv(self.convblock(self.conv0(x))), scale)
        return predicted[:, :4] * scale, predicted[:, 4:5]


def warp(picture: Tensor, flow: Tensor) -> Tensor:
    """Backward warp: sample `picture` at the positions the flow (in pixels) points to, bilinearly, with the border
    repeated outside the picture."""
    batch, _, height, width = flow.shape
    key = (str(flow.device), batch, height, width)
    base = _GRIDS.get(key)
    if base is None:
        xs = torch.linspace(-1.0, 1.0, width, device=flow.device).view(1, 1, 1, width).expand(batch, 1, height, width)
        ys = torch.linspace(-1.0, 1.0, height, device=flow.device).view(1, 1, height, 1).expand(batch, 1, height, width)
        base = _GRIDS[key] = torch.cat((xs, ys), dim=1)
    # pixels -> the [-1, 1] coordinates of grid_sample (align_corners: -1 and 1 are the centres of the edge pixels)
    offset = torch.cat((flow[:, 0:1] / ((picture.shape[3] - 1.0) / 2.0),
                        flow[:, 1:2] / ((picture.shape[2] - 1.0) / 2.0)), dim=1)
    grid = (base + offset).permute(0, 2, 3, 1).to(picture.dtype)
    return F.grid_sample(picture, grid, mode="bilinear", padding_mode="border", align_corners=True)


class IFNet(nn.Module):
    """RIFE 4.7: four refinement blocks over the two pictures, the moment and a 4-channel feature of each picture."""

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(3 + 3 + 4 + 4 + 1, 192)  # both pictures, both features, the moment
        self.block1 = IFBlock(3 + 3 + 4 + 4 + 1 + 1 + 4, 128)  # ... warped, the blend logit and the flow
        self.block2 = IFBlock(3 + 3 + 4 + 4 + 1 + 1 + 4, 96)
        self.block3 = IFBlock(3 + 3 + 4 + 4 + 1 + 1 + 4, 64)
        self.encode = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                                    nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1))

    def forward(self, img0: Tensor, img1: Tensor, timestep, scales: Sequence[float] = SCALES) -> Tensor:
        """The picture at `timestep` (0 = img0, 1 = img1): float tensors [N, 3, H, W] in [0, 1];
        `timestep` is a float or a tensor [N, 1, 1, 1]. H and W are padded to multiples of 64 inside."""
        height, width = img0.shape[2:]
        padding = (0, -width % PAD, 0, -height % PAD)
        first, second = F.pad(img0.clamp(0, 1), padding), F.pad(img1.clamp(0, 1), padding)
        if torch.is_tensor(timestep):
            moment = timestep.repeat(1, 1, first.shape[2], first.shape[3])
        else:
            moment = torch.full_like(first[:, :1], 1.0) * timestep
        feature0, feature1 = self.encode(first), self.encode(second)
        flow = logit = None
        warped0, warped1 = first, second
        for block, scale in zip((self.block0, self.block1, self.block2, self.block3), scales):
            if flow is None:
                flow, logit = block(torch.cat((first, second, feature0, feature1, moment), dim=1), None, scale)
            else:
                inputs = torch.cat((warped0, warped1, warp(feature0, flow[:, :2]), warp(feature1, flow[:, 2:4]),
                                    moment, logit), dim=1)
                change, logit = block(inputs, flow, scale)
                flow = flow + change
            warped0, warped1 = warp(first, flow[:, :2]), warp(second, flow[:, 2:4])
        weight = torch.sigmoid(logit)
        return (warped0 * weight + warped1 * (1 - weight))[:, :, :height, :width]


def load_rife(path: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> IFNet:
    """The RIFE 4.9 weights on `device`: either rife49.pth (a plain state dict) or the author's own file,
    train_log/flownet.pkl of the Practical-RIFE v4.9 archive, whose names all carry a "module." prefix (the
    checkpoint was saved from a data-parallel wrapper). The tensors are the same; only the prefix is dropped."""
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    prefix = "module."
    if state and all(name.startswith(prefix) for name in state):
        state = {name[len(prefix):]: tensor for name, tensor in state.items()}
    model = IFNet()
    model.load_state_dict(state)
    return model.to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def interpolate(model: IFNet, frames: Tensor, source_fps: int, target_fps: int, batch_size: int = 1,
                device: Optional[torch.device | str] = None, *, cancel=None, progress_cb=None) -> Tensor:
    """Frames [N, 3, H, W] (float, 0..1) at `source_fps` to frames at `target_fps`, float32 on the CPU.

    Every output frame sits at i / target_fps seconds: source frames that fall exactly on the clock are
    copied, the others are interpolated between their neighbours at the exact fraction, and the final
    source frame is kept (no extrapolation past it)."""
    parameter = next(model.parameters())
    device = torch.device(device) if device is not None else parameter.device
    tasks, layout = frame_rate_plan(len(frames), source_fps, target_fps)
    middle: Dict[int, List[Tensor]] = {}
    for start in range(0, len(tasks), batch_size):
        if cancel is not None:
            cancel.check()
        batch = tasks[start:start + batch_size]
        first = torch.cat([frames[pair:pair + 1] for pair, _ in batch]).to(device, parameter.dtype)
        second = torch.cat([frames[pair + 1:pair + 2] for pair, _ in batch]).to(device, parameter.dtype)
        moments = torch.tensor([moment for _, moment in batch], dtype=parameter.dtype, device=device).view(-1, 1, 1, 1)
        made = model(first, second, moments).clamp(0, 1).detach().cpu()
        for index, (pair, _) in enumerate(batch):
            middle.setdefault(pair, []).append(made[index:index + 1])
        if progress_cb is not None:
            progress_cb((start + len(batch)) / len(tasks), "Frame interpolation (RIFE)")
    if cancel is not None:
        cancel.check()
    out = [frames[pair:pair + 1] if slot < 0 else middle[pair][slot] for pair, slot in layout]
    return torch.cat(out).to(torch.float32)
