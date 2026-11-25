# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The speech features InfiniteTalk listens to (MeiGen-AI/InfiniteTalk, Apache-2.0: `generate_infinitetalk.py`
`audio_prepare_single` / `get_embedding`, `src/audio_analysis/wav2vec2.py`).

The speech is read at 16 kHz mono and brought to an integrated loudness of -23 LUFS (ITU-R BS.1770, as the reference
does with pyloudnorm: K-weighting, 400 ms blocks with 75 % overlap, the -70 LUFS absolute and -10 LU relative gates;
`integrated_loudness` follows pyloudnorm's `Meter`, MIT). The raw samples (the feature extractor of
chinese-wav2vec2-base does not normalise) go through wav2vec 2.0's convolutional feature encoder; its features are
linearly interpolated to the video's frame count at 25 FPS (`align_corners`), then projected and run through the 12
transformer layers. The hidden states of every layer (not the input embedding) are the features: [frames, 12, 768].

For a chunk of the video, every frame gets the features of a window of five frames centred on it (`windows`; indices
before the start or past the end repeat the first or the last frame).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RATE = 16000
FPS = 25
TARGET_LUFS = -23.0
WINDOW = 5  # frames of speech features per video frame (-2..+2)


def _biquad(kind: str, gain_db: float, q: float, fc: float, rate: int):
    """pyloudnorm's IIRfilter coefficients (RBJ cookbook), normalised by a0."""
    a = 10 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / rate
    alpha = math.sin(w0) / (2.0 * q)
    cos = math.cos(w0)
    if kind == "high_shelf":
        b = [a * ((a + 1) + (a - 1) * cos + 2 * math.sqrt(a) * alpha), -2 * a * ((a - 1) + (a + 1) * cos),
             a * ((a + 1) + (a - 1) * cos - 2 * math.sqrt(a) * alpha)]
        den = [(a + 1) - (a - 1) * cos + 2 * math.sqrt(a) * alpha, 2 * ((a - 1) - (a + 1) * cos),
               (a + 1) - (a - 1) * cos - 2 * math.sqrt(a) * alpha]
    else:  # high_pass
        b = [(1 + cos) / 2, -(1 + cos), (1 + cos) / 2]
        den = [1 + alpha, -2 * cos, 1 - alpha]
    return np.array(b) / den[0], np.array(den) / den[0]


def integrated_loudness(samples: np.ndarray, rate: int = RATE, block: float = 0.4) -> float:
    """BS.1770 integrated loudness (LUFS) of mono samples; -inf for silence or less than one block."""
    from scipy.signal import lfilter

    data = np.asarray(samples, dtype=np.float64)
    if data.size < block * rate:
        return float("-inf")
    for kind, gain, q, fc in (("high_shelf", 4.0, 1 / math.sqrt(2), 1500.0), ("high_pass", 0.0, 0.5, 38.0)):
        b, a = _biquad(kind, gain, q, fc, rate)
        data = lfilter(b, a, data)
    step = 0.25  # 75 % overlap
    seconds = data.size / rate
    count = int(round((seconds - block) / (block * step))) + 1
    energy = np.zeros(count)
    for j in range(count):
        lower, upper = int(block * (j * step) * rate), int(block * (j * step + 1) * rate)
        energy[j] = np.sum(data[lower:upper] ** 2) / (block * rate)
    with np.errstate(divide="ignore"):
        loudness = -0.691 + 10.0 * np.log10(energy)
    gated = energy[loudness >= -70.0]
    if gated.size == 0:
        return float("-inf")
    relative = -0.691 + 10.0 * np.log10(np.mean(gated)) - 10.0
    gated = energy[(loudness > relative) & (loudness > -70.0)]
    if gated.size == 0:
        return float("-inf")
    return float(-0.691 + 10.0 * np.log10(np.mean(gated)))


def normalize_loudness(samples: np.ndarray, rate: int = RATE, target: float = TARGET_LUFS) -> np.ndarray:
    """The samples brought to `target` LUFS; unchanged when the loudness cannot be measured (silence, too short), as
    the reference leaves a loudness outside +-100 alone."""
    loudness = integrated_loudness(samples, rate)
    if not math.isfinite(loudness) or abs(loudness) > 100:
        return np.asarray(samples, dtype=np.float32)
    return (np.asarray(samples, dtype=np.float64) * 10 ** ((target - loudness) / 20.0)).astype(np.float32)


def load_encoder(folder: str | Path, device: torch.device | str = "cpu"):
    """chinese-wav2vec2-base from its folder (config.json, pytorch_model.bin; read with transformers' safe loading).
    The pretraining heads of the checkpoint are not part of the encoder and are left out."""
    from transformers import Wav2Vec2Model
    from transformers.utils import logging as hf_logging

    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()  # the unused pretraining weights are expected, not news
    try:
        model = Wav2Vec2Model.from_pretrained(str(folder), local_files_only=True)
    finally:
        hf_logging.set_verbosity(level)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def features(encoder, speech: np.ndarray, frames: int) -> torch.Tensor:
    """[frames, 12, 768] float32 on the CPU for 16 kHz mono `speech` (already loudness-normalised), the video being
    `frames` long at 25 FPS."""
    device = next(encoder.parameters()).device
    values = torch.from_numpy(np.asarray(speech, dtype=np.float32))[None].to(device)
    extracted = encoder.feature_extractor(values).transpose(1, 2)  # [1, T, 512]
    extracted = F.interpolate(extracted.transpose(1, 2), size=frames, mode="linear",
                              align_corners=True).transpose(1, 2)
    hidden, _ = encoder.feature_projection(extracted)
    # The output of every transformer layer, i.e. the reference's `hidden_states[1:]` (the encoder's own collection of
    # them is not filled when the encoder is called on its own in transformers 5)
    layers = []
    hooks = [layer.register_forward_hook(lambda _module, _args, output: layers.append(
        output[0] if isinstance(output, tuple) else output)) for layer in encoder.encoder.layers]
    try:
        encoder.encoder(hidden)
    finally:
        for hook in hooks:
            hook.remove()
    if len(layers) != len(encoder.encoder.layers):
        raise RuntimeError("the speech encoder did not run all of its layers")
    return torch.stack(layers, dim=2)[0].float().cpu()  # [frames, 12, 768]


def speech_features(folder: str | Path, speech: np.ndarray, frames: int) -> torch.Tensor:
    """The whole path for one speaker: loudness, the encoder (on the CPU, float32, as the reference runs it)."""
    encoder = load_encoder(folder, "cpu")
    try:
        return features(encoder, normalize_loudness(speech), frames)
    finally:
        del encoder


def extend(feats: torch.Tensor, frames: int) -> torch.Tensor:
    """Features for at least `frames` frames: a shorter sequence goes on mirrored from its end (the reference's
    `torch.flip` of the missing tail), so the last chunk has speech features for every frame it makes."""
    while feats.shape[0] < frames:
        missing = min(frames - feats.shape[0] + 3, feats.shape[0])
        feats = torch.cat([feats, torch.flip(feats[-missing:], dims=[0])], dim=0)
    return feats


def windows(feats: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """[1, length, 5, 12, 768]: for each frame of [start, start + length) the features of frames -2..+2 around it,
    clamped to the sequence."""
    offsets = torch.arange(WINDOW) - WINDOW // 2
    index = (torch.arange(start, start + length)[:, None] + offsets[None]).clamp(0, feats.shape[0] - 1)
    return feats[index][None]
