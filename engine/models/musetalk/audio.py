# SPDX-FileCopyrightText: 2024 Tencent Music Entertainment Group
# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only AND MIT
"""Whisper-tiny features and the per-frame audio windows of MuseTalk 1.5.

Own implementation of the conventions of upstream `musetalk/utils/audio_processor.py` (TMElyralab/MuseTalk,
MIT) and `musetalk/models/unet.py::PositionalEncoding`, on our Whisper encoder (`engine.models.whisper`,
which follows OpenAI Whisper, MIT):

1. The waveform is cut into 30 s segments and each segment is zero-padded to 30 s BEFORE the log-mel
   (Hugging Face's WhisperFeatureExtractor pads the samples, so the clamp at `max - 8` and the last STFT
   frames see the padding; padding the finished mel instead gives other features).
2. Five states per 20 ms step: the embedding plus positions, the outputs of blocks 1-3, and block 4 AFTER
   the final layer norm (`hidden_states` of transformers' WhisperEncoder).
3. The features of all segments are concatenated and cut to floor(seconds * 50) steps. Frame i at `fps`
   reads steps [floor(i * 50 / fps) - 2c, ... + 10) with c = ceil(50 / fps), zeros outside the audio
   (not the edge steps repeated). The 10 x 5 states are flattened step-major to 50 tokens of 384.
4. A sinusoidal table over the 50 tokens (d = 384, sin on even channels) is added before the UNet.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ...core.cancel import CancelToken
from .. import whisper
from ..zimage import assert_loaded, materialize

SAMPLE_RATE = whisper.SAMPLE_RATE
FEATURE_RATE = 50                    # Whisper encoder steps per second
SEGMENT = 30 * SAMPLE_RATE           # samples per Whisper window (30 s)
STATES = 5                           # embedding + 4 blocks of Whisper tiny
WIDTH = 384                          # Whisper tiny's hidden size
CONTEXT = 2                          # audio_padding_length_left/right of the reference
STEPS = 2 * (2 * CONTEXT + 1)        # 10 steps (200 ms) per video frame
TOKENS = STEPS * STATES              # 50 tokens of WIDTH channels
TINY_DIMS = (80, 1500, WIDTH, 6, 4)  # mel bands, encoder steps, width, heads, blocks of Whisper tiny


#: Audio this much shorter than a whole number of frames still counts as reaching that frame (1 ms):
#: resampling a track of exactly N frames often lands a sample or two short.
FRAME_TOLERANCE = SAMPLE_RATE // 1000


def frame_count(samples: int, fps: int) -> int:
    """Video frames that `samples` of 16 kHz audio cover: floor(seconds * fps), the reference's count.

    In integers, with `FRAME_TOLERANCE`: the reference's float expression (samples / 16000) * fps gives
    N - 1 for about 6 % of exact lengths of N frames (29 * 640 samples -> 28.999...), and a sample short
    after resampling would do the same; the last frame would then lose its lip sync and fade out."""
    return (int(samples) + FRAME_TOLERANCE) * int(fps) // SAMPLE_RATE


def feature_count(samples: int) -> int:
    """Whisper steps kept: the reference's own float expression, so the windows stay bit-compatible (a
    step lost to float rounding is the last 20 ms, which frame_windows pads with zeros anyway)."""
    return math.floor((samples / SAMPLE_RATE) * FEATURE_RATE)


def waveform(audio: np.ndarray | Tensor) -> Tensor:
    """The audio as a float32 CPU vector, checked: nonempty, mono, finite."""
    wave = torch.as_tensor(np.asarray(audio) if not isinstance(audio, Tensor) else audio).detach().cpu()
    wave = wave.to(torch.float32)
    if wave.ndim != 1 or not wave.numel():
        raise ValueError("MuseTalk needs nonempty mono audio at 16 kHz.")
    if not bool(torch.isfinite(wave).all()):
        raise ValueError("MuseTalk audio contains nonfinite samples.")
    return wave


def segment_mels(audio: np.ndarray | Tensor) -> list[Tensor]:
    """80 x 3000 log-mels of the 30 s segments, each segment zero-padded to 30 s first."""
    wave = waveform(audio)
    return [whisper.log_mel(F.pad(part, (0, SEGMENT - part.numel()))) for part in wave.split(SEGMENT)]


def encoder_states(encoder: whisper.AudioEncoder, mel: Tensor, cancel: CancelToken | None = None) -> Tensor:
    """[1500, 5, width] for one padded 30 s mel: the four block inputs and the final-norm output."""
    device = encoder.positional_embedding.device
    states = encoder(mel[None].to(device=device, dtype=torch.float32), cancel)[0]
    return torch.cat((states[:, :-1], encoder.ln_post(states[:, -1:])), dim=1)


@torch.inference_mode()
def whisper_features(encoder: whisper.AudioEncoder, audio: np.ndarray | Tensor,
                     cancel: CancelToken | None = None) -> Tensor:
    """[floor(seconds * 50), 5, width] Whisper states on the CPU, in float32 (the encoder is tiny)."""
    wave = waveform(audio)
    parts = []
    for mel in segment_mels(wave):
        if cancel is not None:
            cancel.check()
        parts.append(encoder_states(encoder, mel, cancel).float().cpu())
    return torch.cat(parts)[:feature_count(wave.numel())]


def frame_windows(features: Tensor, frames: int, fps: int) -> Tensor:
    """[frames, 50, width]: the 10-step window of every video frame, zeros outside the audio."""
    if features.ndim != 3 or frames < 0 or int(fps) <= 0:
        raise ValueError("Invalid MuseTalk audio features or frame count.")
    if not frames:
        return features.new_zeros((0, STEPS * features.shape[1], features.shape[2]))
    multiplier = FEATURE_RATE / int(fps)
    left = math.ceil(multiplier) * CONTEXT
    starts = torch.tensor([math.floor(index * multiplier) for index in range(frames)])
    # The reference pads 3 * CONTEXT * ceil(multiplier) zeros on the right; more padding never changes a
    # window it can make, and keeps frames past the end of the audio defined (all zeros).
    right = max(math.ceil(multiplier) * 3 * CONTEXT, int(starts[-1]) + STEPS - left - len(features))
    padded = torch.cat((features.new_zeros((left, *features.shape[1:])), features,
                        features.new_zeros((right, *features.shape[1:]))))
    return padded[starts[:, None] + torch.arange(STEPS)].flatten(1, 2)


def windows(encoder: whisper.AudioEncoder, audio: np.ndarray | Tensor, fps: int, limit: int,
            cancel: CancelToken | None = None) -> Tensor:
    """The windows [count, 50, width] of the first count = min(limit, frames of the audio) video frames."""
    wave = waveform(audio)
    count = min(int(limit), frame_count(wave.numel(), fps))
    return frame_windows(whisper_features(encoder, wave, cancel), max(count, 0), fps)


SPEECH_FLOOR_DB = -50.0   # a frame quieter than this (RMS, dBFS) is silence, whatever the level of the track
SPEECH_RANGE_DB = 30.0    # ... and so is a frame this far below the track's loud frames (its 95th percentile)


def speech_frames(audio: np.ndarray | Tensor, fps: int, count: int) -> np.ndarray:
    """bool [count]: which of the first `count` video frames have speech in their own 1/fps of the audio.

    Only for reporting how much of the speech was lip-synced: a speaker's track is digital silence
    where the person does not speak (app.dialogue.speaker_track), and quiet pauses inside a line
    count as silence too. Frames past the end of the audio have none."""
    wave = waveform(audio).numpy().astype(np.float64)
    edges = [index * SAMPLE_RATE // int(fps) for index in range(int(count) + 1)]
    parts = (wave[a:b] for a, b in zip(edges[:-1], edges[1:], strict=True))
    rms = np.array([math.sqrt(float(np.mean(np.square(part)))) if part.size else 0.0 for part in parts])
    if not len(rms):
        return np.zeros(0, dtype=bool)
    loud = float(np.percentile(rms, 95))
    threshold = max(10 ** (SPEECH_FLOOR_DB / 20), loud * 10 ** (-SPEECH_RANGE_DB / 20))
    return rms >= threshold


def positions(tokens: int = TOKENS, width: int = WIDTH) -> Tensor:
    """The sinusoidal table [tokens, width] that the reference adds to the audio windows (float32)."""
    position = torch.arange(0, tokens, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
    table = torch.zeros(tokens, width)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)
    return table


def conditioning(windows: Tensor) -> Tensor:
    """The UNet's encoder_hidden_states for windows [N, 50, width] (float32 here; the caller casts)."""
    return windows.float() + positions(windows.shape[1], windows.shape[2])


# ----------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------

_HF_LAYER = re.compile(r"^layers\.(\d+)\.(.+)$")
_HF_LAYER_PARTS = {"self_attn.q_proj": "attn.query", "self_attn.k_proj": "attn.key", "self_attn.v_proj": "attn.value",
                   "self_attn.out_proj": "attn.out", "self_attn_layer_norm": "attn_ln", "fc1": "mlp.0",
                   "fc2": "mlp.2", "final_layer_norm": "mlp_ln"}
_HF_TOP = {"conv1.weight": "conv1.weight", "conv1.bias": "conv1.bias", "conv2.weight": "conv2.weight",
           "conv2.bias": "conv2.bias", "embed_positions.weight": "positional_embedding",
           "layer_norm.weight": "ln_post.weight", "layer_norm.bias": "ln_post.bias"}


def hf_encoder_name(name: str) -> str | None:
    """Our encoder's state-dict key for a Hugging Face Whisper key, None for a decoder or unrelated key."""
    for prefix in ("model.encoder.", "encoder."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    else:
        return None
    if name in _HF_TOP:
        return _HF_TOP[name]
    match = _HF_LAYER.match(name)
    if match:
        index, rest = match.groups()
        part, _, leaf = rest.rpartition(".")
        if part in _HF_LAYER_PARTS:
            return f"blocks.{index}.{_HF_LAYER_PARTS[part]}.{leaf}"
    raise KeyError(f"Whisper encoder: unknown Hugging Face tensor {name!r}")


def _hf_encoder_tensors(path: str | Path) -> Iterable[Tuple[str, Tensor]]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            ours = hf_encoder_name(name)
            if ours is not None:
                yield ours, handle.get_tensor(name)


def load_audio_encoder(path: str | Path, device: str | torch.device,
                       dims: Tuple[int, int, int, int, int] = TINY_DIMS) -> whisper.AudioEncoder:
    """Whisper tiny's encoder in float32, from OpenAI's tiny.pt or Hugging Face's model.safetensors.

    tiny.pt is the file the model catalog installs (read with weights_only=True, its dims are checked);
    the Hugging Face file (openai/whisper-tiny) needs the key map above, and `dims` (mel bands, steps,
    width, heads, blocks) for other sizes in tests. Either way every encoder tensor must be present."""
    if Path(path).suffix.lower() != ".safetensors":
        return whisper.load_audio_encoder(path, device)
    with torch.device("meta"):
        encoder = whisper.AudioEncoder(*dims)
    materialize(encoder, _hf_encoder_tensors(path), device, torch.float32, "MuseTalk Whisper")
    assert_loaded(encoder, "MuseTalk Whisper")
    return encoder.eval()
