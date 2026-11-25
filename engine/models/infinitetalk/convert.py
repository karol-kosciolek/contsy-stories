# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The install step of the InfiniteTalk audio adapter: the official float32 release in the engine's FP8 layout.

MeiGen-AI publishes the single-speaker adapter (`single/infinitetalk.safetensors`, Apache-2.0) in float32, 10 GB: per
Wan block an audio cross-attention (`q_linear` 5120 x 5120, `kv_linear` 10240 x 768, `proj` 5120 x 5120) and its
input norm `norm_x`, plus the audio projection `audio_proj`. `adapter_fp8` keeps the 120 large linear weights as
float8_e4m3fn with one float32 scale per output row (`X.weight_scale` [out, 1], the layout of LightX2V's FP8 files that
`formats.fp8` reads and `core.fp8_linear.FP8Linear` multiplies in FP8), the scale being the row's largest magnitude over
448, and every other tensor as bfloat16, the dtype the network runs in: 2.5 GB. The output is the same bytes on every
PC (sorted names, no metadata, deterministic rounding on the CPU).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from safetensors import safe_open

FP8_MAX = 448.0  # the largest finite float8_e4m3fn value
QUANTIZED = (".audio_cross_attn.q_linear.weight", ".audio_cross_attn.kv_linear.weight",
             ".audio_cross_attn.proj.weight")


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(float8_e4m3fn weight, float32 scale [out, 1]) with `weight ~= fp8 * scale` row by row."""
    weight = weight.to(torch.float32)
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    return (weight / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), scale


def adapter_tensors(source: str | Path, cancel_flag: Optional[Callable[[], bool]] = None) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if cancel_flag is not None and cancel_flag():
                raise ValueError("conversion cancelled; run it again to start over")
            tensor = handle.get_tensor(name)
            if name.endswith(QUANTIZED):
                out[name], out[name + "_scale"] = quantize_rows(tensor)
            else:
                out[name] = tensor.to(torch.bfloat16).contiguous()
    if sum(name.endswith(QUANTIZED) for name in out) != 3 * 40:
        raise ValueError(f"{Path(source).name} is not the 40-block InfiniteTalk adapter")
    return dict(sorted(out.items()))


def adapter_fp8(source: str | Path, output: str | Path, *, expected_size: Optional[int] = None,
                expected_sha256: Optional[str] = None, progress_cb: Optional[Callable[[int, int], None]] = None,
                cancel_flag: Optional[Callable[[], bool]] = None) -> str:
    """Write the adapter `source` as described above to `output` (through `<output>.part`, renamed once its size and
    SHA-256 equal the expected ones when given). Returns the SHA-256."""
    from safetensors.torch import save_file

    output = Path(output)
    part = output.with_name(output.name + ".part")
    if part.is_symlink() or output.is_symlink():
        raise ValueError("refusing to write through a link")
    tensors = adapter_tensors(source, cancel_flag)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(part))
    del tensors
    size = part.stat().st_size
    digest = hashlib.sha256()
    with part.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
            if progress_cb is not None:
                progress_cb(stream.tell(), size)
    sha256 = digest.hexdigest()
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{part.name} has {size} bytes, the catalog expects {expected_size}")
    if expected_sha256 is not None and sha256 != expected_sha256.lower():
        raise ValueError(f"{part.name} has SHA-256 {sha256}, the catalog expects {expected_sha256}")
    os.replace(part, output)
    return sha256
