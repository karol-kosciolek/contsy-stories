# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Install step: the official Qwen-Image-Edit-2511 text encoder (bfloat16 shards) to the engine's scaled-fp8 file.

Source: `text_encoder/model-0000{1..4}-of-00004.safetensors` of Qwen/Qwen-Image-Edit-2511 (Apache-2.0), the
Qwen2.5-VL-7B decoder and vision tower in bfloat16 under transformers-4 names (`model.layers.N...`,
`visual.blocks.N...`, `model.embed_tokens.weight`, `lm_head.weight`); 16.6 GB.

Output: one safetensors file of about 8.3 GB that `text.load_text_encoder` reads.

* Every matrix of a linear layer (`QUANTIZED`: the decoder's q/k/v/o and gate/up/down projections, the vision blocks'
  qkv/proj and gate/up/down projections and the two layers of the vision merger: 358 in the real model) becomes
  float8 e4m3fn `X.weight` next to a 0-dimensional float32 `X.scale_weight`, so that `weight * scale_weight` is the
  matrix. Per tensor: `scale = max|w| / 416` in float32 and `weight = w / scale` rounded to the nearest float8 value
  (ties to even). 416 is the e4m3fn step just below the format's maximum 448, so the largest weight sits one step
  under the top of the range.
* Everything else (the embedding table, the norms, the biases, the 5-D patch embedding) is copied unchanged.
* `lm_head.weight` (1.1 GB) is left out: the conditioning never uses the output head.

Measured on 2026-11-15 against the official shards of revision 6f3ccc0b: the result holds exactly the tensors of the
fp8 encoder the product used before this step (every float8 byte, every scale and every bfloat16 tensor equal), so the
conditioning does not change.

The file is written by this module, not by the safetensors library, so that its bytes depend on the input only and the
catalog can pin its SHA-256: the tensors are laid out by element size (float32, then bfloat16, then float8, each
group by name), the header is compact JSON padded with spaces to a multiple of 8 bytes, and the metadata names the
source. Memory: one tensor at a time (the largest is the 1.1 GB embedding table), plus a float32 copy of one matrix
while it is quantized (at most 272 MB).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import torch
from safetensors import safe_open

INDEX_SUFFIX = ".safetensors.index.json"
FP8 = torch.float8_e4m3fn
FP8_TOP = 416.0  # the largest |w| of a matrix maps here (see the docstring)
SKIPPED = ("lm_head.weight",)
# The matrices that become float8 (transformers-4 names of Qwen2.5-VL).
QUANTIZED = re.compile(r"^(model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
                       r"|visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(gate|up|down)_proj)"
                       r"|visual\.merger\.mlp\.\d+)\.weight$")
DTYPES = {"F32": (torch.float32, 4), "BF16": (torch.bfloat16, 2), "F8_E4M3": (FP8, 1)}
METADATA = {"format": "pt", "source": "Qwen/Qwen-Image-Edit-2511 text_encoder (bfloat16)",
            "quantization": "per-tensor float8 e4m3fn, scale_weight = max|w| / 416"}

ProgressCb = Callable[[int, int], None]


class ConversionError(Exception):
    """The source files are not what the step expects, or the result does not match the catalog."""


class _Entry(NamedTuple):
    name: str  # name in the output file
    dtype: str  # safetensors dtype of the output
    shape: List[int]
    source: Path
    key: str  # tensor of the source file
    kind: str  # "copy", "quantize" (the float8 matrix) or "scale" (its scale)


def _index_shards(index: Path) -> List[Tuple[Path, Set[str]]]:
    """[(shard, the names the index assigns to it)] of a sharded checkpoint's index. Shard names must be plain file
    names next to the index, so a damaged index cannot point outside its folder."""
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ConversionError(f"{index.name} is not a sharded-checkpoint index: {error}") from None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ConversionError(f"{index.name} has an empty weight_map")
    shards: Dict[Path, Set[str]] = {}
    for name, shard in weight_map.items():
        if (not isinstance(shard, str) or not shard.endswith(".safetensors") or Path(shard).name != shard
                or "\\" in shard):
            raise ConversionError(f"{index.name}: invalid shard name {shard!r} for {name}")
        shards.setdefault(index.parent / shard, set()).add(name)
    return sorted(shards.items())


def _sources(source: str | Path | Sequence[str | Path]) -> List[Tuple[Path, Optional[Set[str]]]]:
    """The shard files to read: a list of them, a `*.safetensors.index.json`, or the folder that holds one index.
    With an index, each shard must hold exactly the names the index assigns to it."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            found = sorted(path.glob("*" + INDEX_SUFFIX))
            if len(found) != 1:
                raise ConversionError(f"{path} must hold exactly one *{INDEX_SUFFIX}, found {len(found)}")
            path = found[0]
        if path.name.endswith(INDEX_SUFFIX):
            return _index_shards(path)
        return [(path, None)]
    return [(Path(item), None) for item in sorted(source, key=lambda item: Path(item).name)]


def _plan(sources: Sequence[Tuple[Path, Optional[Set[str]]]]) -> List[_Entry]:
    """The output tensors in file order, from the headers of the shards only."""
    entries: List[_Entry] = []
    seen: Dict[str, Path] = {}
    for source, assigned in sources:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if assigned is not None and set(keys) != assigned:
                missing, extra = sorted(assigned - set(keys))[:5], sorted(set(keys) - assigned)[:5]
                raise ConversionError(f"{source.name} does not match its index: missing {missing}, not listed {extra}")
            for key in keys:
                if key in seen:
                    raise ConversionError(f"{key} is in both {seen[key].name} and {source.name}")
                seen[key] = source
                piece = handle.get_slice(key)
                dtype, shape = piece.get_dtype(), list(piece.get_shape())
                if dtype != "BF16":
                    raise ConversionError(f"{source.name}: {key} is {dtype}; the official shards are bfloat16")
                if key in SKIPPED:
                    continue
                if QUANTIZED.match(key):
                    if len(shape) != 2:
                        raise ConversionError(f"{source.name}: {key} has shape {shape}, expected a matrix")
                    entries.append(_Entry(key, "F8_E4M3", shape, source, key, "quantize"))
                    entries.append(_Entry(key[:-len("weight")] + "scale_weight", "F32", [], source, key, "scale"))
                else:
                    entries.append(_Entry(key, "BF16", shape, source, key, "copy"))
    if not any(entry.kind == "quantize" for entry in entries):
        raise ConversionError("the source files hold no linear layer of Qwen2.5-VL")
    return sorted(entries, key=lambda entry: (-DTYPES[entry.dtype][1], entry.name))


def _nbytes(entry: _Entry) -> int:
    return math.prod(entry.shape) * DTYPES[entry.dtype][1]


def _header(entries: Sequence[_Entry]) -> bytes:
    header: Dict[str, object] = {"__metadata__": METADATA}
    offset = 0
    for entry in entries:
        size = _nbytes(entry)
        header[entry.name] = {"dtype": entry.dtype, "shape": entry.shape, "data_offsets": [offset, offset + size]}
        offset += size
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return text + b" " * (-len(text) % 8)


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(float8 matrix, 0-dim float32 scale) of one weight matrix; see the docstring of the module."""
    values = weight.to(dtype=torch.float32, copy=True)
    low, high = torch.aminmax(values)
    if not (math.isfinite(float(low)) and math.isfinite(float(high))):
        raise ConversionError("a weight matrix holds a value that is not finite")
    peak = torch.maximum(-low, high)
    scale = peak / FP8_TOP if float(peak) > 0 else torch.ones((), dtype=torch.float32)
    return values.div_(scale).to(FP8), scale.reshape(())


def _raw(tensor: torch.Tensor) -> memoryview:
    return memoryview(tensor.contiguous().reshape(-1).view(torch.uint8).numpy())


def quantize_text_encoder(source: str | Path | Sequence[str | Path], output: str | Path, *, expected_size: int = 0,
                          expected_sha256: Optional[str] = None, progress_cb: Optional[ProgressCb] = None,
                          cancel_flag: Optional[Callable[[], bool]] = None) -> str:
    """Write the scaled-fp8 encoder made from the official shards to `output`; returns its SHA-256.

    `source` is the shards' `model.safetensors.index.json`, the folder that holds it, or a list of the shard files
    (in any order). The file is written to `output` + ".part" and renamed into place only when its size and SHA-256
    match `expected_size` and `expected_sha256` (when given); the size is checked before anything is written.
    `progress_cb(done_bytes, total_bytes)` follows the writing; `cancel_flag()` returning True stops it. A failed or
    cancelled run leaves the `.part` file, which the next run overwrites."""
    if sys.byteorder != "little":
        raise ConversionError("safetensors files are little-endian; this machine is not")
    entries = _plan(_sources(source))
    header = _header(entries)
    total = 8 + len(header) + sum(_nbytes(entry) for entry in entries)
    if expected_size and total != expected_size:
        raise ConversionError(f"the converted encoder would have {total} bytes, the catalog expects {expected_size}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".part")
    scale_entries = [entry for entry in entries if entry.kind == "scale"]  # all float32, so the first data bytes
    scales: Dict[str, torch.Tensor] = {}
    with part.open("wb") as stream:
        stream.write(len(header).to_bytes(8, "little"))
        stream.write(header)
        scale_start = stream.tell()
        stream.write(bytes(4 * len(scale_entries)))  # filled in once every matrix has been read
        done = stream.tell()
        for entry in entries:
            if entry.kind == "scale":
                continue
            if cancel_flag is not None and cancel_flag():
                raise ConversionError("Conversion cancelled. Run it again to restart it.")
            with safe_open(str(entry.source), framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(entry.key)
            if entry.kind == "quantize":
                tensor, scales[entry.key] = quantize(tensor)
            data = _raw(tensor)
            if data.nbytes != _nbytes(entry):
                raise ConversionError(f"{entry.name}: {data.nbytes} bytes, the header says {_nbytes(entry)}")
            stream.write(data)
            done += data.nbytes
            del tensor, data
            if progress_cb is not None:
                progress_cb(done, total)
        stream.seek(scale_start)
        for entry in scale_entries:
            stream.write(_raw(scales[entry.key].reshape(1)))
        stream.flush()
        os.fsync(stream.fileno())
    if part.stat().st_size != total:
        raise ConversionError(f"{part.name} has {part.stat().st_size} bytes, expected {total}")
    with part.open("rb") as written:  # the hash of the bytes on disk, not of what was meant to be written
        result = hashlib.file_digest(written, "sha256").hexdigest()
    if expected_sha256 and result != expected_sha256.lower():
        raise ConversionError(f"Integrity check failed for the converted encoder {output.name}; "
                              f"{part.name} has been kept, do not use it.")
    part.replace(output)
    return result
