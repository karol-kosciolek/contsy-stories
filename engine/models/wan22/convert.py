# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Install step: a Wan 2.2 expert with a LightX2V distillation LoRA merged in, as scaled FP8 in the layout of
LightX2V's own FP8 experts.

Inputs (both Apache-2.0):

* the expert in the original Wan key names, either as GGUF (QuantStack's `Wan2.2-I2V-A14B-HighNoise-Q6_K.gguf`: the 400
  block linears Q6_K, every other tensor F16, the patch embedding F32) or as safetensors in float32, float16 or bfloat16
  (the official Index-AniSora V3.2 experts, `V3.2/{high,low}_noise_model/diffusion_pytorch_model.safetensors`, float32,
  fine-tuned from Wan 2.2 I2V A14B for anime: the 2D style's experts, app/styles.py);
* the LoRA in LightX2V's key style (`wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors`):
  `diffusion_model.<name>.lora_down.weight` / `.lora_up.weight` (rank 64, bfloat16, no alpha) and full changes
  `.diff` (a weight), `.diff_b` (a bias) and `.diff_m` (the modulation table of a block or of the head), merged at
  strength 1.0 as LightX2V's loader applies them (`parameter + change`, `weight + up @ down`).

Output: one safetensors file with the tensors, dtypes, shapes and byte layout of LightX2V's
`wan2.2_i2v_A14b_{high,low}_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors` (formats.fp8 and
transformer.load_transformer read it as they read those):

* each block linear (`blocks.N.{self,cross}_attn.{q,k,v,o}.weight`, `blocks.N.ffn.{0,2}.weight`) is
  float8_e4m3fn `X.weight` [out, in] with a bfloat16 `X.weight_scale` [out, 1], one scale per output channel, so
  that `weight * weight_scale` is the matrix; the largest value of every row is stored as +-448;
* every other tensor is float32 holding a bfloat16 value (as LightX2V stores them; the engine loads them as bfloat16);
* the tensors are laid out float32 first, then bfloat16, then float8, each group by name, from offset 0 without gaps;
  the header is compact JSON in that order padded with spaces to a multiple of 8 bytes, without metadata.

The bytes must be the same on every PC (the catalog pins the file's SHA-256), so the step runs on the CPU with
operations whose results are fixed by IEEE 754 and never depend on the CPU, the number of threads or the order in
which a library sums: elementwise additions, multiplications and divisions (each rounded once), the maximum of a row
(exact), and integer arithmetic on the bits for every rounding to a narrower format. No matrix product is used.

1. The GGUF tensor in float32 (formats.ggml: each Q6_K value is one rounded product of the block's scales and its
   quant). A full change is added in float32.
2. The LoRA product, `sum over r of up[:, r] * down[r, :]`, is summed in float64 with r in order, one addition per
   term. Every product of two float32-or-narrower values is exact in float64 (at most 48 significant bits), so each
   term rounds once whether or not the CPU fuses the multiply and the add. The merged weight is
   `float32(float64(base) + product)`.
3. A tensor outside the block linears is rounded to bfloat16 (nearest, ties to even).
4. A block linear, row by row: `scale = bfloat16(max|w| / 448)` (1 for a row of zeros), `q = clamp(w / scale, +-448)`
   rounded to float8 e4m3fn (nearest, ties to even). Like LightX2V's files, the stored scale is the one the row was
   divided by, so `q * scale` is the merged weight up to the FP8 rounding alone.

Memory: a few rows at a time (about 1M values), the GGUF is memory-mapped and the LoRA read one matrix at a time; the
bfloat16 scales (5 MB for A14B) are kept until the end, where they are written into the room left for them before
the FP8 weights. The SHA-256 is read back from the written file.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from safetensors import safe_open
from torch import Tensor

from ...formats.ggml import GGUFFile, QTensor, dequantize

_SAFETENSORS_PLAIN = {"F32": "F32", "F16": "F16", "BF16": "BF16"}  # safetensors dtype -> GGML plain type name

FP8_MAX = 448.0  # the largest float8 e4m3fn value
BLOCK_LINEAR = re.compile(r"^blocks\.\d+\.((self|cross)_attn\.[qkvo]|ffn\.[02])\.weight$")
LORA_PREFIX = "diffusion_model."
# LightX2V's low-noise `_1022` LoRA is its Wan 2.1 I2V LoRA: its changes of the image key/value branch (which Wan 2.2
# I2V does not have) are left out, as the engine's LoRA loader leaves them out when it applies the LoRA at run time
WAN21_IMAGE_BRANCH = re.compile(r"(^|\.)(k_img|v_img|norm_k_img|img_emb)(\.|$)")
# LoRA key suffix -> (the suffix of the tensor it changes, its role)
LORA_SUFFIXES = ((".lora_down.weight", ".weight", "down"), (".lora_up.weight", ".weight", "up"),
                 (".alpha", ".weight", "alpha"), (".diff_b", ".bias", "diff"), (".diff_m", ".modulation", "diff"),
                 (".diff", ".weight", "diff"))
LORA_DTYPES = (torch.bfloat16, torch.float16, torch.float32)  # products of these are exact in float64
DTYPES = {"F32": 4, "BF16": 2, "F8_E4M3": 1}  # bytes per value
CHUNK_VALUES = 1 << 20  # values merged at a time (rows of a matrix)

ProgressCb = Callable[[int, int], None]


class ConversionError(Exception):
    """The input files are not what the step expects, or the result does not match the catalog."""


@dataclass(frozen=True)
class _Entry:
    name: str  # name in the output file
    dtype: str  # safetensors dtype
    shape: Tuple[int, ...]
    source: str  # the GGUF tensor it is made from
    kind: str  # "plain", "fp8" (the float8 matrix) or "scale" (its scales)

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * DTYPES[self.dtype]


def round_bf16(values: Tensor) -> Tensor:
    """float32 `values` rounded to the nearest bfloat16 (ties to even), as float32: integer arithmetic on the bits,
    for finite values (no carry reaches the sign bit)."""
    bits = values.contiguous().view(torch.int32)
    return ((bits + ((bits >> 16) & 1) + 0x7FFF) & -65536).view(torch.float32)


def e4m3_bytes(values: Tensor) -> Tensor:
    """float32 `values` with |v| <= 448 rounded to the nearest float8 e4m3fn (ties to even), as its bytes (uint8):
    integer arithmetic on the bits.

    A float32 with biased exponent E and 24-bit significand m (with the hidden bit) keeps its top four bits when it is
    a normal e4m3 number (E >= 121, i.e. |v| >= 2**-6); below that the step is the subnormal 2**-9, so 121 - E more
    bits go. `q = round(m >> dropped)` counts steps; the byte is `(E - 121) * 8 + q` for normal values (q = 16, a
    carry, lands on the next exponent by itself) and `q` for subnormal ones (q = 8 is the smallest normal, 0x08)."""
    bits = values.contiguous().view(torch.int32).to(torch.int64)
    magnitude = bits & 0x7FFFFFFF
    exponent = magnitude >> 23
    significand = (magnitude & 0x7FFFFF) | torch.where(exponent > 0, 0x800000, 0)
    dropped = (20 + (121 - exponent).clamp(min=0)).clamp(max=30)
    steps = significand >> dropped
    rest = significand - (steps << dropped)
    half = torch.ones_like(dropped) << (dropped - 1)
    steps = steps + ((rest > half) | ((rest == half) & ((steps & 1) == 1))).to(torch.int64)
    code = ((exponent - 121).clamp(min=0) << 3) + steps
    return (code | ((bits < 0).to(torch.int64) << 7)).to(torch.uint8)


def low_rank_product(up: Tensor, down: Tensor) -> Tensor:
    """`up @ down` in float64, summed over the rank in order, one rounding per term (see the module docstring).
    `up` is [rows, rank] and `down` [rank, in], both float32 or narrower; `down` may already be float64."""
    up = up.to(torch.float64)
    down = down.to(torch.float64)
    total = up[:, 0:1] * down[0:1]
    for r in range(1, down.shape[0]):
        total.addcmul_(up[:, r:r + 1], down[r:r + 1])
    return total


def quantize_rows(weight: Tensor) -> Tuple[Tensor, Tensor]:
    """(float8 e4m3fn bytes [rows, in] as uint8, float32 scale [rows] holding a bfloat16 value) of float32 rows."""
    peak = weight.abs().amax(dim=1)
    scale = torch.where(peak > 0, peak / torch.full_like(peak, FP8_MAX), torch.ones_like(peak))
    scale = round_bf16(scale)
    quotient = (weight / scale[:, None]).clamp_(-FP8_MAX, FP8_MAX)
    return e4m3_bytes(quotient), scale


class _Changes:
    """The LoRA's changes, {GGUF tensor name: {role: key}}, checked against the base before anything is written."""

    def __init__(self, path: Path, base: Dict[str, QTensor]):
        self.path = path
        self.keys: Dict[str, Dict[str, str]] = {}
        self.skipped: List[str] = []  # keys of the Wan 2.1 image branch (formats.lora skips them the same way)
        unmatched: List[str] = []
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                stem = key.removeprefix(LORA_PREFIX)
                found = next(((stem[:-len(suffix)] + target, role) for suffix, target, role in LORA_SUFFIXES
                              if stem.endswith(suffix)), None)
                if found is None or found[0] not in base:
                    if WAN21_IMAGE_BRANCH.search(stem):  # the image branch of Wan 2.1 I2V, which Wan 2.2 lacks
                        self.skipped.append(key)
                        continue
                    unmatched.append(key)
                    continue
                name, role = found
                roles = self.keys.setdefault(name, {})
                if role in roles:
                    raise ConversionError(f"{path.name}: {roles[role]} and {key} both change {name}")
                roles[role] = key
                piece = handle.get_slice(key)
                dtype, shape = piece.get_dtype(), tuple(piece.get_shape())
                if role != "alpha" and dtype not in ("BF16", "F16", "F32"):
                    raise ConversionError(f"{path.name}: {key} is {dtype}, not bfloat16, float16 or float32")
                if role == "diff" and math.prod(shape) != math.prod(base[name].shape):
                    raise ConversionError(f"{path.name}: {key} has shape {shape}, {name} {base[name].shape}")
            if unmatched:
                raise ConversionError(f"{path.name}: {len(unmatched)} keys change no tensor of the expert, for "
                                      f"example {unmatched[:3]}")
            for name, roles in self.keys.items():
                low_rank = {"down", "up"} & set(roles)
                if low_rank and low_rank != {"down", "up"}:
                    raise ConversionError(f"{path.name}: only one of the two LoRA matrices of {name}")
                if "alpha" in roles and not low_rank:
                    raise ConversionError(f"{path.name}: an alpha without LoRA matrices for {name}")
                if low_rank:
                    down, up = (tuple(handle.get_slice(roles[role]).get_shape()) for role in ("down", "up"))
                    shape = base[name].shape
                    if len(shape) != 2 or len(down) != 2 or len(up) != 2 or down[0] != up[1] or down[0] < 1 \
                            or (up[0], down[1]) != tuple(shape):
                        raise ConversionError(f"{path.name}: LoRA matrices {down} and {up} do not fit {name} {shape}")

    def tensor(self, handle, name: str, role: str) -> Optional[Tensor]:
        key = self.keys.get(name, {}).get(role)
        return None if key is None else handle.get_tensor(key)


def _plan(base: Dict[str, QTensor]) -> List[_Entry]:
    """The output tensors in file order (float32, then bfloat16, then float8, each by name), from the GGUF header."""
    entries: List[_Entry] = []
    for name, tensor in base.items():
        shape = tuple(int(n) for n in tensor.shape)
        if BLOCK_LINEAR.match(name):
            if len(shape) != 2:
                raise ConversionError(f"{name} has shape {shape}, expected a matrix")
            entries.append(_Entry(name, "F8_E4M3", shape, name, "fp8"))
            entries.append(_Entry(name + "_scale", "BF16", (shape[0], 1), name, "scale"))
        else:
            entries.append(_Entry(name, "F32", shape, name, "plain"))
    if not any(entry.kind == "fp8" for entry in entries):
        raise ConversionError("the base holds no block linear of a Wan transformer")
    return sorted(entries, key=lambda entry: (-DTYPES[entry.dtype], entry.name))


def _header(entries: List[_Entry]) -> bytes:
    header: Dict[str, object] = {}
    offset = 0
    for entry in entries:
        header[entry.name] = {"dtype": entry.dtype, "shape": list(entry.shape),
                              "data_offsets": [offset, offset + entry.nbytes]}
        offset += entry.nbytes
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return text + b" " * (-len(text) % 8)


def _merged_rows(tensor: QTensor, changes: _Changes, handle, limit: Optional[int] = None) -> Iterator[Tensor]:
    """The merged tensor in float32, a block of rows at a time (all of it for a vector); `limit`: its first rows only
    (merged_rows_share)."""
    shape = tuple(int(n) for n in tensor.shape)
    rows = shape[0]
    per_row = math.prod(shape[1:])
    data = tensor.data.reshape(rows, -1)
    if limit is not None:
        rows = min(rows, limit)
    diff = changes.tensor(handle, tensor.name, "diff")
    if diff is not None:
        diff = diff.to(torch.float32).reshape(shape)
    down = changes.tensor(handle, tensor.name, "down")
    up = changes.tensor(handle, tensor.name, "up")
    alpha = changes.tensor(handle, tensor.name, "alpha")
    if down is not None:
        down = down.to(torch.float64)
    step = max(1, CHUNK_VALUES // max(per_row, 1))
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        value = dequantize(data[start:stop], tensor.qtype, (stop - start, *shape[1:]), torch.float32)
        if diff is not None:
            value = value + diff[start:stop]
        if down is not None:
            product = low_rank_product(up[start:stop], down)
            if alpha is not None:
                product.mul_(float(alpha) / down.shape[0])
            value = (value.to(torch.float64) + product).to(torch.float32)
        if not bool(torch.isfinite(value).all()):
            raise ConversionError(f"{tensor.name}: the merged tensor holds a value that is not finite")
        yield value


CHECK_ROWS = 8  # rows of each sampled matrix that merged_rows_share makes again


class _NoChanges:
    """The changes of no LoRA (`_merged_rows` of the base alone)."""

    @staticmethod
    def tensor(handle, name: str, role: str) -> None:
        return None


def merged_rows_share(merged: str | Path, base: str | Path, lora: str | Path, blocks=(0, 13, 26, 39),
                      rows: int = CHECK_ROWS) -> float:
    """Of the FP8 values and scales that `lora` changes, in the first `rows` rows of the block linears of `blocks` it
    has low-rank matrices for (the values whose FP8 byte or scale differs between `base` alone and `base` with
    `lora`), the share that the FP8 expert `merged` holds exactly as `merge_lora_to_fp8` makes them: 1.0 for an expert
    made by this converter from those files, about 0 for the base without the LoRA or with another one. For a LoRA
    without the full parameter changes (`.diff`, `.diff_b`) that `transformer.lora_merged_share` compares, such as
    LightX2V's text-to-video `_1217` LoRAs. Only the sampled rows are read and merged."""
    wanted = tuple(f"blocks.{index}." for index in blocks)
    with _base_tensors(Path(base)) as tensors:
        changes = _Changes(Path(lora), tensors)
        names = sorted(name for name, roles in changes.keys.items()
                       if "down" in roles and name.startswith(wanted) and BLOCK_LINEAR.match(name))
        if not names:
            raise ConversionError(f"{Path(lora).name} changes no block linear of blocks {blocks}")
        equal = total = 0
        with safe_open(str(lora), framework="pt", device="cpu") as handle, \
                safe_open(str(merged), framework="pt", device="cpu") as merged_file:
            present = set(merged_file.keys())
            for name in names:
                expected = next(_merged_rows(tensors[name], changes, handle, limit=rows))
                data, scale = quantize_rows(expected)
                plain, plain_scale = quantize_rows(next(_merged_rows(tensors[name], _NoChanges(), None, limit=rows)))
                data, plain = data.view(torch.uint8), plain.view(torch.uint8)
                scale, plain_scale = scale.to(torch.bfloat16).reshape(-1), plain_scale.to(torch.bfloat16).reshape(-1)
                changed, scale_changed = data != plain, scale != plain_scale
                count = int(changed.sum()) + int(scale_changed.sum())
                total += count
                if not count or name not in present or name + "_scale" not in present:
                    continue
                stored = merged_file.get_slice(name)[:len(expected)]
                stored_scale = merged_file.get_slice(name + "_scale")[:len(expected)]
                if stored.element_size() != 1 or tuple(stored.shape) != tuple(data.shape):  # not an FP8 matrix
                    continue
                equal += int(((stored.view(torch.uint8) == data) & changed).sum())
                equal += int(((stored_scale.to(torch.bfloat16).reshape(-1) == scale) & scale_changed).sum())
    if not total:
        raise ConversionError(f"{Path(lora).name} changes none of the sampled FP8 values")
    return equal / total


class _SafetensorsTensor:
    """A tensor of a safetensors base, as `_plan` and `_merged_rows` read a GGUF `QTensor`: its name, its plain GGML
    type, its shape and `data`, read from the file when it is asked for (one tensor at a time, so a 57 GB float32
    expert is never held whole)."""

    def __init__(self, handle, name: str):
        from gguf import GGMLQuantizationType

        piece = handle.get_slice(name)
        dtype = piece.get_dtype()
        if dtype not in _SAFETENSORS_PLAIN:
            raise ConversionError(f"{name} is {dtype}; the base must hold float32, float16 or bfloat16 tensors")
        self.name, self.shape = name, tuple(int(n) for n in piece.get_shape())
        self.qtype = getattr(GGMLQuantizationType, _SAFETENSORS_PLAIN[dtype])
        self._handle = handle

    @property
    def data(self) -> Tensor:
        return self._handle.get_tensor(self.name)


@contextmanager
def _base_tensors(path: Path):
    """{name: tensor} of the base expert, a GGUF or a safetensors file in the original Wan key names."""
    if path.suffix.lower() == ".gguf":
        with GGUFFile(path) as gguf_file:
            if gguf_file.arch != "wan":
                raise ConversionError(f"{path.name} is not a Wan GGUF file (architecture {gguf_file.arch!r})")
            yield gguf_file.tensors
        return
    # pread: each tensor is read on its own. The mmap backend maps the whole file copy-on-write, which Windows charges
    # to the commit limit (57 GB for a float32 Wan expert): with less free, reading a tensor crashed (2026-11-15).
    with safe_open(str(path), framework="pt", device="cpu", backend="pread") as handle:
        yield {name: _SafetensorsTensor(handle, name) for name in handle.keys()}


def _raw(tensor: Tensor) -> memoryview:
    return memoryview(tensor.contiguous().reshape(-1).view(torch.uint8).numpy())


def merge_lora_to_fp8(base: str | Path, lora: str | Path, output: str | Path, *, expected_size: int = 0,
                      expected_sha256: Optional[str] = None, progress_cb: Optional[ProgressCb] = None,
                      cancel_flag: Optional[Callable[[], bool]] = None) -> str:
    """Write the expert `base` (GGUF, or safetensors in the original Wan names) with the LoRA `lora` merged in as scaled
    FP8 to `output`; returns its SHA-256.

    The file is written to `output` + ".part" and renamed into place only when its size and SHA-256 equal
    `expected_size` and `expected_sha256` (when given); the size is checked before anything is written.
    `progress_cb(done_bytes, total_bytes)` follows the writing; `cancel_flag()` returning True stops it. A failed or
    cancelled run leaves the `.part` file, which the next run overwrites."""
    if sys.byteorder != "little":
        raise ConversionError("safetensors files are little-endian; this machine is not")
    base, lora, output = Path(base), Path(lora), Path(output)
    part = output.with_name(output.name + ".part")
    if part.is_symlink() or output.is_symlink():
        raise ConversionError("refusing to write through a link")
    with _base_tensors(base) as tensors:
        changes = _Changes(lora, tensors)
        entries = _plan(tensors)
        header = _header(entries)
        total = 8 + len(header) + sum(entry.nbytes for entry in entries)
        if expected_size and total != expected_size:
            raise ConversionError(f"the converted expert would have {total} bytes, the catalog expects {expected_size}")
        output.parent.mkdir(parents=True, exist_ok=True)
        scales: Dict[str, Tensor] = {}
        scale_offsets: List[Tuple[int, _Entry]] = []
        with part.open("wb") as stream, safe_open(str(lora), framework="pt", device="cpu") as handle:
            stream.write(len(header).to_bytes(8, "little"))
            stream.write(header)
            done = stream.tell()
            for entry in entries:
                if entry.kind == "scale":  # written once its matrix is done (the scales come before the matrices)
                    scale_offsets.append((stream.tell(), entry))
                    stream.write(bytes(entry.nbytes))
                    done += entry.nbytes
                    continue
                pieces = []
                for rows in _merged_rows(tensors[entry.source], changes, handle):
                    if cancel_flag is not None and cancel_flag():
                        raise ConversionError("Conversion cancelled. Run it again to restart it.")
                    if entry.kind == "fp8":
                        data, scale = quantize_rows(rows)
                        pieces.append(scale)
                    else:
                        data = round_bf16(rows)
                    raw = _raw(data)
                    stream.write(raw)
                    done += raw.nbytes
                    if progress_cb is not None:
                        progress_cb(done, total)
                if entry.kind == "fp8":
                    scales[entry.name + "_scale"] = torch.cat(pieces)
            for offset, entry in scale_offsets:
                stream.seek(offset)
                stream.write(_raw(scales[entry.name].to(torch.bfloat16)))  # exact: each is a bfloat16 value
            stream.seek(0, os.SEEK_END)
            stream.flush()
            os.fsync(stream.fileno())
        del tensors
    if part.stat().st_size != total:
        raise ConversionError(f"{part.name} has {part.stat().st_size} bytes, expected {total}")
    with part.open("rb") as written:  # the hash of the bytes on disk, not of what was meant to be written
        result = hashlib.file_digest(written, "sha256").hexdigest()
    if expected_sha256 and result != expected_sha256.lower():
        raise ConversionError(f"Integrity check failed for the converted expert {output.name}; {part.name} has been "
                              "kept, do not use it.")
    os.replace(part, output)
    return result


def _main(argv: List[str]) -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(prog="python -m engine.models.wan22.convert",
                                     description="Merge a LightX2V LoRA into a Wan 2.2 GGUF expert as scaled FP8.")
    parser.add_argument("base", help="the expert as GGUF or safetensors (original Wan names)")
    parser.add_argument("lora", help="the LoRA (LightX2V key style)")
    parser.add_argument("output")
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--sha256", default=None)
    args = parser.parse_args(argv)
    last = [-1]

    def progress(done: int, total: int) -> None:
        percent = done * 100 // total
        if percent // 5 != last[0] // 5:
            last[0] = percent
            print(f"{percent}% ({done / 2**30:.2f} of {total / 2**30:.2f} GiB)", flush=True)

    started = time.perf_counter()
    sha256 = merge_lora_to_fp8(args.base, args.lora, args.output, expected_size=args.size,
                               expected_sha256=args.sha256, progress_cb=progress)
    print(f"{args.output}\nsize {Path(args.output).stat().st_size}\nsha256 {sha256}\n"
          f"seconds {time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
