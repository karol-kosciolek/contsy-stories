# SPDX-FileCopyrightText: 2023 Stability AI
# SPDX-FileCopyrightText: 2022 NVIDIA CORPORATION
# SPDX-FileCopyrightText: 2025 Jiarui Hai
# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only AND MIT
"""Decode EzAudio latents to a 24 kHz mono waveform.

The decoder half of the "Oobleck" autoencoder of stable-audio-tools (MIT) as EzAudio ships it
(github.com/haidog-yaqub/EzAudio at 86f18268, src/modules/stable_vae/models/autoencoders.py); its SnakeBeta
activation comes from BigVGAN (MIT). 128 latent channels at 50 frames per second become samples directly: there
is no spectrogram and no vocoder. The model generates latents, so the encoder half is not needed.

The decoder has no final tanh: its output can pass full scale slightly, which the pipeline handles.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterator, Tuple

import torch
from torch import Tensor, nn

from ..zimage import assert_loaded, materialize

SAMPLE_RATE = 24000
HOP_LENGTH = 480
LATENT_CHANNELS = 128
_CHUNK_FRAMES = 200
# The convolution stack reaches 9 latent frames to each side (8 are measurably too few, see
# tests/test_engine_ezaudio.py); 16 leave a margin and cost under 10 percent more work per 200-frame chunk.
_CONTEXT_FRAMES = 16
_PREFIX = "autoencoder.decoder."
# The file is a training checkpoint; these parts of it are not the decoder.
_OTHER_PARTS = ("autoencoder.encoder.", "discriminator.", "losses_gen.", "sdstft.")


class SnakeBeta(nn.Module):
    """x + sin(alpha x)^2 / beta per channel, with alpha and beta stored as logarithms."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        alpha = self.alpha.exp()[None, :, None]
        beta = self.beta.exp()[None, :, None]
        return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(
            SnakeBeta(channels),
            nn.Conv1d(channels, channels, 7, dilation=dilation, padding=3 * dilation),
            SnakeBeta(channels),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x) + x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            SnakeBeta(in_channels),
            nn.ConvTranspose1d(in_channels, out_channels, 2 * stride, stride=stride, padding=math.ceil(stride / 2)),
            ResidualUnit(out_channels, 1),
            ResidualUnit(out_channels, 3),
            ResidualUnit(out_channels, 9),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, channels: int, multipliers: list[int], strides: list[int]):
        super().__init__()
        widths = [channels * multiplier for multiplier in (1, *multipliers)]
        layers = [nn.Conv1d(latent_dim, widths[-1], 7, padding=3)]
        for index in range(len(widths) - 1, 0, -1):
            layers.append(DecoderBlock(widths[index], widths[index - 1], strides[index - 1]))
        layers.extend((SnakeBeta(widths[0]), nn.Conv1d(widths[0], 1, 7, padding=3, bias=False)))
        self.layers = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        return self.layers(latent)


def _decoder_weights(state: dict) -> Iterator[Tuple[str, Tensor]]:
    """The decoder tensors with weight normalization folded into plain weights (g * v / |v|)."""
    for name, tensor in state.items():
        if name.startswith(_OTHER_PARTS) or name.endswith(".weight_g"):
            continue
        target = name.removeprefix(_PREFIX)  # a name without the prefix is unknown and fails in materialize
        if name.endswith(".weight_v"):
            scale = state[f"{name.removesuffix('.weight_v')}.weight_g"]
            yield f"{target.removesuffix('.weight_v')}.weight", torch._weight_norm(tensor, scale, 0)
        else:
            yield target, tensor


def _check_folded(state: dict) -> None:
    """A weight_g without its weight_v would be dropped by `_decoder_weights`; refuse such a file."""
    orphans = [name for name in state if name.startswith(_PREFIX) and name.endswith(".weight_g")
               and f"{name.removesuffix('.weight_g')}.weight_v" not in state]
    if orphans:
        raise KeyError(f"EzAudio decoder: {orphans[:8]} have no weight_v")


def _decoder_config(path: Path) -> dict:
    """The decoder settings of the bundle's config.json, refused when the engine does not implement them."""
    config = json.loads(path.read_text(encoding="utf-8"))
    model = config["model"]
    decoder = dict(model["decoder"]["config"])
    hop = math.prod(decoder["strides"])
    supported = (config["sample_rate"] == SAMPLE_RATE and model["decoder"]["type"] == "oobleck"
                 and model["io_channels"] == decoder["out_channels"] == 1 and decoder["use_snake"]
                 and not decoder.get("final_tanh", True) and not decoder.get("antialias_activation", False)
                 and not decoder.get("use_nearest_upsample", False)
                 and all(stride % 2 == 0 for stride in decoder["strides"])
                 and hop == model["downsampling_ratio"] == HOP_LENGTH and decoder["latent_dim"] == LATENT_CHANNELS)
    if not supported:
        raise ValueError("The EzAudio bundle requires the 24 kHz mono Oobleck decoder with 128 latent channels.")
    return decoder


def load(path: Path, device: torch.device | str, dtype: torch.dtype = torch.float32) -> Decoder:
    """The decoder from `vae/1m.pt` and the `config.json` next to it."""
    config = _decoder_config(Path(path).with_name("config.json"))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"state_dict"}:
        raise ValueError("An EzAudio autoencoder checkpoint holds exactly one entry, 'state_dict'.")
    state = checkpoint["state_dict"]
    _check_folded(state)
    with torch.device("meta"):
        decoder = Decoder(config["latent_dim"], config["channels"], config["c_mults"], config["strides"])
    materialize(decoder, _decoder_weights(state), device, dtype, "EzAudio decoder")
    assert_loaded(decoder, "EzAudio decoder")
    return decoder.eval()


def decode(decoder: Decoder, latent: Tensor, check: Callable[[], None]) -> Tensor:
    """The waveform [1, frames * 480] of a latent [1, 128, frames] on the CPU.

    Decoded in chunks with their neighboring frames, whose samples are discarded, so the memory does not grow
    with the length and the result equals decoding everything at once."""
    weight = decoder.layers[0].weight
    frames = latent.shape[-1]
    pieces = []
    for start in range(0, frames, _CHUNK_FRAMES):
        check()
        end = min(start + _CHUNK_FRAMES, frames)
        left = max(0, start - _CONTEXT_FRAMES)
        decoded = decoder(latent[..., left:end + _CONTEXT_FRAMES].to(weight.device, weight.dtype))
        first = (start - left) * HOP_LENGTH
        pieces.append(decoded[0, :, first:first + (end - start) * HOP_LENGTH].float().cpu())
    return torch.cat(pieces, dim=-1)
