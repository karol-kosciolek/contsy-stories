# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Scaled-fp8 safetensors: one scale per tensor (the catalog's UMT5 and Qwen2.5-VL text encoders) or one per
output channel (LightX2V's FP8 Wan 2.2 experts).

Per tensor: every large linear weight is stored as float8 (`X.weight`) next to a scalar `X.scale_weight`; the real
weight is `weight * scale_weight`. A marker tensor `scaled_fp8` (and, in some files, `X.scale_input`) is bookkeeping
only. `iter_tensors` expands them to regular tensors.

Per channel (LightX2V, `lightx2v/Wan2.2-Distill-Models`, and the high-noise expert `models.wan22.convert` writes in
the same layout): `X.weight` float8_e4m3fn [out, in] and `X.weight_scale` [out, 1]; the real weight is
`weight * weight_scale` row by row. `iter_channel_scaled` keeps such a weight in FP8 as a
`ScaledFP8` (for `core.fp8_linear.FP8Linear`), so a 14B expert stays 15 GB instead of 28.

Reading is lazy: both load one tensor at a time, so a large file can be expanded or placed layer by layer.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple, Union

import torch
from safetensors import safe_open

FP8_TYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
BOOKKEEPING_SUFFIXES = (".scale_weight", ".scale_input")
CHANNEL_SCALE_SUFFIX = ".weight_scale"
FP8_HEADER_DTYPES = ("F8_E4M3", "F8_E5M2")  # how a safetensors header names FP8 tensors


class UnsupportedQuantization(ValueError):
    """The file uses a quantization the engine does not implement."""


def dequantize_fp8(weight: torch.Tensor, scale: torch.Tensor,
                   dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """`weight * scale`, multiplied in float32 and cast once (never loses the scale's precision)."""
    return (weight.to(torch.float32) * scale.to(torch.float32)).to(dtype)


def iter_tensors(path: str | Path, dtype: torch.dtype = torch.bfloat16,
                 device: str | torch.device = "cpu") -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield `(name, tensor)` for every real tensor of the file.

    Scaled fp8 weights come out as `dtype`; fp8 weights without a scale are only cast; every
    other tensor keeps the dtype it was stored in. Scale and marker tensors are not yielded."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        present = set(keys)
        for key in keys:
            if key == "scaled_fp8" or key.endswith(BOOKKEEPING_SUFFIXES):
                continue
            tensor = handle.get_tensor(key)
            if tensor.dtype in FP8_TYPES:
                scale_key = key[:-len("weight")] + "scale_weight" if key.endswith(".weight") else None
                if scale_key in present:
                    tensor = dequantize_fp8(tensor, handle.get_tensor(scale_key), dtype)
                else:
                    tensor = tensor.to(dtype)
            yield key, tensor.to(device)


@dataclass(frozen=True)
class ScaledFP8:
    """An FP8 weight [out, in] with one scale per output channel: the real weight is `weight * scale[:, None]`."""
    weight: torch.Tensor
    scale: torch.Tensor  # [out] or [out, 1], any float dtype

    @property
    def shape(self) -> torch.Size:
        return self.weight.shape

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return dequantize_fp8(self.weight, self.scale.reshape(-1, *([1] * (self.weight.dim() - 1))), dtype)

    def to(self, dtype: torch.dtype = torch.bfloat16, device: str | torch.device = "cpu") -> torch.Tensor:
        """The expanded weight, for a tensor that no FP8 layer can hold (as `QTensor.to` expands GGUF blocks)."""
        return self.dequantize(dtype).to(device)


def _header(path: str | Path) -> dict:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 100 << 20:
            raise UnsupportedQuantization(f"{Path(path).name}: the safetensors header is implausibly large")
        return json.loads(handle.read(size))


def channel_scaled(path: str | Path) -> bool:
    """Whether a safetensors file holds FP8 weights with per-channel `.weight_scale` tensors (reads the header only)."""
    header = _header(path)
    return any(name.endswith(CHANNEL_SCALE_SUFFIX) and name[:-len("_scale")] in header
               and header[name[:-len("_scale")]].get("dtype") in FP8_HEADER_DTYPES for name in header)


def iter_channel_scaled(path: str | Path) -> Iterator[Tuple[str, Union[torch.Tensor, ScaledFP8]]]:
    """Yield `(name, value)` for every weight of a per-channel scaled FP8 file, on the CPU.

    An FP8 `X.weight` with an `X.weight_scale` comes out as one `ScaledFP8` under `X.weight` (the scale is not
    yielded on its own); every other tensor as it is stored. FP8 without a scale, or a scale whose shape does not
    match its weight, is refused: the result would be silently wrong."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        present = set(keys)
        for key in keys:
            if key.endswith(CHANNEL_SCALE_SUFFIX) and key[:-len("_scale")] in present:
                continue
            tensor = handle.get_tensor(key)
            if tensor.dtype not in FP8_TYPES:
                yield key, tensor
                continue
            scale_key = key + "_scale"
            if scale_key not in present:
                raise UnsupportedQuantization(f"{Path(path).name}: {key} is FP8 without a {scale_key}")
            scale = handle.get_tensor(scale_key)
            if tensor.dtype != torch.float8_e4m3fn or tensor.dim() != 2 or scale.numel() != tensor.shape[0]:
                raise UnsupportedQuantization(f"{Path(path).name}: {key} {tensor.dtype} {tuple(tensor.shape)} with a "
                                              f"scale of shape {tuple(scale.shape)} is not per-channel float8_e4m3fn")
            yield key, ScaledFP8(tensor, scale.reshape(-1).to(torch.float32))


def iter_kept_fp8(path: str | Path) -> Iterator[Tuple[str, Union[torch.Tensor, ScaledFP8]]]:
    """Yield `(name, value)` for every weight of a file whose 2-D float8_e4m3fn weights are to stay FP8, on the CPU:
    one with a per-tensor `X.scale_weight` (Kijai's scaled FP8 Wan 2.1 files) comes out as a `ScaledFP8` whose
    per-channel scale repeats that one value, one with a per-channel `X.weight_scale` as in `iter_channel_scaled`, and
    an FP8 weight stored without any scale (the InfiniteTalk adapter's FP8 release) with a scale of 1. Scale and
    marker tensors are not yielded; any other FP8 tensor is refused, as `iter_channel_scaled` does."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        present = set(keys)
        for key in keys:
            if key == "scaled_fp8" or key.endswith(BOOKKEEPING_SUFFIXES) or (
                    key.endswith(CHANNEL_SCALE_SUFFIX) and key[:-len("_scale")] in present):
                continue
            tensor = handle.get_tensor(key)
            if tensor.dtype not in FP8_TYPES:
                yield key, tensor
                continue
            if tensor.dtype != torch.float8_e4m3fn or tensor.dim() != 2:
                raise UnsupportedQuantization(f"{Path(path).name}: {key} {tensor.dtype} {tuple(tensor.shape)} is not "
                                              "a 2-D float8_e4m3fn weight")
            rows = tensor.shape[0]
            per_tensor = key[:-len("weight")] + "scale_weight" if key.endswith(".weight") else None
            if key + "_scale" in present:
                scale = handle.get_tensor(key + "_scale").reshape(-1).to(torch.float32)
            elif per_tensor in present:
                scale = handle.get_tensor(per_tensor).reshape(-1).to(torch.float32)
                if scale.numel() != 1:
                    raise UnsupportedQuantization(f"{Path(path).name}: {per_tensor} holds {scale.numel()} values")
                scale = scale.expand(rows).contiguous()
            else:
                scale = torch.ones(rows, dtype=torch.float32)
            if scale.numel() != rows:
                raise UnsupportedQuantization(f"{Path(path).name}: the scale of {key} has {scale.numel()} values for "
                                              f"{rows} rows")
            yield key, ScaledFP8(tensor, scale)
