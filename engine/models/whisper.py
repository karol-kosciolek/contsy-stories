# SPDX-FileCopyrightText: 2022 OpenAI
# SPDX-FileCopyrightText: 2024 Bytedance Ltd. and/or its affiliates
# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only AND MIT AND Apache-2.0
"""The Whisper tiny audio encoder and its log-mel input, for lip sync (MuseTalk, `engine.models.musetalk.audio`).

The network and the mel conventions follow OpenAI Whisper (MIT). Returning the output of every encoder layer
instead of the last one follows ByteDance LatentSync (Apache-2.0), where this module was first written. Audio
arrives as 16 kHz mono samples; neither a speech decoder nor the ffmpeg command line is needed.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..core.cancel import CancelToken
from .zimage import assert_loaded, materialize

SAMPLE_RATE = 16000
FFT_SIZE = 400
HOP = 160
N_MELS = 80
MEL_WINDOW = 3000
_DIMS = {"n_mels": N_MELS, "n_audio_ctx": MEL_WINDOW // 2, "n_audio_state": 384, "n_audio_head": 6, "n_audio_layer": 4}


@lru_cache(maxsize=1)
def _mel_filters() -> Tensor:
    """Slaney-area normalized triangles, matching the 80-band filterbank in the Whisper assets."""
    step = 200.0 / 3
    transition = 1000.0 / step
    log_step = np.log(6.4) / 27
    end = transition + np.log((SAMPLE_RATE / 2) / 1000) / log_step
    mel = np.linspace(0, end, N_MELS + 2)
    hz = np.where(mel >= transition, 1000 * np.exp(log_step * (mel - transition)), step * mel)
    bins = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
    edges = np.diff(hz)
    ramps = hz[:, None] - bins
    weights = np.maximum(0, np.minimum(-ramps[:-2] / edges[:-1, None], ramps[2:] / edges[1:, None]))
    # librosa writes the triangle values to float32 before applying the area factors in place.
    weights = weights.astype(np.float32)
    weights *= (2.0 / (hz[2:] - hz[:-2]))[:, None]
    return torch.from_numpy(weights)


def log_mel(audio: np.ndarray | Tensor) -> Tensor:
    waveform = torch.as_tensor(audio, dtype=torch.float32)
    if waveform.ndim != 1 or not waveform.numel():
        raise ValueError("Lip sync needs nonempty mono audio at 16 kHz.")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("Lip sync audio contains nonfinite samples.")
    if waveform.numel() < 2 * HOP:
        waveform = F.pad(waveform, (0, 2 * HOP - waveform.numel()))
    window = torch.hann_window(FFT_SIZE, device=waveform.device)
    spectrum = torch.stft(waveform, FFT_SIZE, HOP, window=window, return_complex=True)
    mel = _mel_filters().to(waveform.device) @ spectrum[:, :-1].abs().square()
    logs = mel.clamp_min(1e-10).log10()
    return (torch.maximum(logs, logs.max() - 8.0) + 4.0) / 4.0


class _LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).to(x.dtype)


class _Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class _Conv(nn.Conv1d):
    def forward(self, x: Tensor) -> Tensor:
        return self._conv_forward(x, self.weight.to(x.dtype), self.bias.to(x.dtype))


class _Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query = _Linear(width, width)
        self.key = _Linear(width, width, bias=False)
        self.value = _Linear(width, width)
        self.out = _Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        scale = (x.shape[-1] // self.heads) ** -0.25
        q = self.query(x).unflatten(-1, (self.heads, -1)).transpose(1, 2) * scale
        k = self.key(x).unflatten(-1, (self.heads, -1)).permute(0, 2, 3, 1) * scale
        v = self.value(x).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        weights = (q @ k).float().softmax(dim=-1).to(q.dtype)
        return self.out((weights @ v).transpose(1, 2).flatten(2))


class _Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attn = _Attention(width, heads)
        self.attn_ln = _LayerNorm(width)
        self.mlp = nn.Sequential(_Linear(width, width * 4), nn.GELU(), _Linear(width * 4, width))
        self.mlp_ln = _LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_ln(x))
        return x + self.mlp(self.mlp_ln(x))


class AudioEncoder(nn.Module):
    """Whisper encoder returning its convolution output and every block's output, before the final norm."""

    def __init__(self, bands: int, context: int, width: int, heads: int, layers: int):
        super().__init__()
        self.conv1 = _Conv(bands, width, kernel_size=3, padding=1)
        self.conv2 = _Conv(width, width, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", torch.zeros(context, width))
        self.blocks = nn.ModuleList(_Block(width, heads) for _ in range(layers))
        self.ln_post = _LayerNorm(width)  # in the checkpoint; the caller applies it to the states it needs

    def forward(self, mel: Tensor, cancel: CancelToken | None = None) -> Tensor:
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        x = (x + self.positional_embedding).to(x.dtype)
        features = [x]
        for block in self.blocks:
            if cancel is not None:
                cancel.check()
            x = block(x)
            features.append(x)
        return torch.stack(features, dim=2)


def load_audio_encoder(path: str | Path, device: str | torch.device) -> AudioEncoder:
    """The encoder of OpenAI's `tiny.pt` in float32 (read with weights_only=True; its dims are checked)."""
    data = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    dims = data["dims"]
    if any(dims.get(name) != expected for name, expected in _DIMS.items()):
        raise ValueError("The lip-sync audio checkpoint must be Whisper tiny with 80 mel bands.")
    with torch.device("meta"):
        encoder = AudioEncoder(*_DIMS.values())
    weights = ((name.removeprefix("encoder."), tensor) for name, tensor in data["model_state_dict"].items()
               if name.startswith("encoder."))
    materialize(encoder, weights, device, torch.float32, "Whisper tiny")
    assert_loaded(encoder, "Whisper tiny")
    return encoder.eval()
