# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""GGUF files: memory-mapped reading and torch dequantization of GGML-quantized tensors.

A model stored as GGUF keeps its large matrices in block-quantized form. `GGUFFile` maps the file
and hands out `QTensor`s without copying them; `dequantize` expands a tensor into a regular torch
tensor on any device, so the engine can keep the quantized bytes in RAM or VRAM and expand one
layer at a time while it runs.

The block types are the four that the catalog's GGUF files use (Q8_0, Q4_K, Q5_K, Q6_K) plus the
plain F32/F16/BF16. Their byte layouts are the ones ggml defines (`block_q8_0`, `block_q4_K`,
`block_q5_K` and `block_q6_K` in ggml-common.h, the reconstruction in `dequantize_row_*` of
ggml-quants.c; https://github.com/ggml-org/ggml, MIT, Copyright (c) 2023-2024 The ggml authors).
The code below is written for torch from those layouts: every block row is decoded with whole-
tensor operations in float32 and cast once at the end. tests/test_engine_ggml.py checks it against
`gguf.quants.dequantize` of gguf-py (MIT, the same authors) on random and on real blocks. Any
other type raises NotImplementedError instead of being guessed.

Shapes are in torch order (rows first). GGUF stores them reversed.
"""
from __future__ import annotations

import math
import mmap
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple

import gguf
import torch

QT = gguf.GGMLQuantizationType
SUPER_BLOCK = 256  # values per K-quant block (QK_K)


def _half(blocks: torch.Tensor, offset: int) -> torch.Tensor:
    """The little-endian float16 at byte `offset` of every block, as float32 of shape [blocks, 1]."""
    return blocks[:, offset:offset + 2].view(torch.float16).float()


def _shifts(device, *amounts: int) -> torch.Tensor:
    return torch.tensor(amounts, dtype=torch.uint8, device=device)


def _q8_0(blocks: torch.Tensor) -> torch.Tensor:
    # 34 bytes: float16 scale, 32 signed bytes; value = scale * byte
    return _half(blocks, 0) * blocks[:, 2:34].view(torch.int8)


def _k_scales(packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The eight 6-bit scales and eight 6-bit minimums packed into the 12 bytes of a Q4_K/Q5_K block.

    Bytes 0-3 hold scales 0-3 and bytes 4-7 minimums 0-3 in their low six bits; bytes 8-11 hold
    the low four bits of scales 4-7 (low nibble) and of minimums 4-7 (high nibble), whose top two
    bits are the top two bits of bytes 0-3 and 4-7."""
    first, second, rest = packed[:, 0:4], packed[:, 4:8], packed[:, 8:12]
    scales = torch.cat((first & 0x3F, (rest & 0x0F) | ((first >> 6) << 4)), dim=1)
    minimums = torch.cat((second & 0x3F, (rest >> 4) | ((second >> 6) << 4)), dim=1)
    return scales, minimums


def _nibbles(packed: torch.Tensor, groups: int) -> torch.Tensor:
    """[blocks, groups * 32] bytes -> [blocks, 2 * groups, 32] four-bit values: for every group of
    32 bytes first the 32 low nibbles, then the 32 high ones."""
    packed = packed.reshape(packed.shape[0], groups, 1, 32)
    return ((packed >> _shifts(packed.device, 0, 4).reshape(1, 1, 2, 1)) & 0x0F).reshape(
        packed.shape[0], 2 * groups, 32)


def _q4_k(blocks: torch.Tensor) -> torch.Tensor:
    # 144 bytes: float16 d, float16 dmin, 12 scale bytes, 128 bytes of nibbles; eight sub-blocks of 32
    count = blocks.shape[0]
    scales, minimums = _k_scales(blocks[:, 4:16])
    step = (_half(blocks, 0) * scales).reshape(count, 8, 1)
    offset = (_half(blocks, 2) * minimums).reshape(count, 8, 1)
    return (step * _nibbles(blocks[:, 16:144], 4) - offset).reshape(count, SUPER_BLOCK)


def _q5_k(blocks: torch.Tensor) -> torch.Tensor:
    # 176 bytes: like Q4_K with 32 extra bytes before the nibbles; bit s of byte l is the fifth bit
    # of value l of sub-block s
    count = blocks.shape[0]
    scales, minimums = _k_scales(blocks[:, 4:16])
    step = (_half(blocks, 0) * scales).reshape(count, 8, 1)
    offset = (_half(blocks, 2) * minimums).reshape(count, 8, 1)
    fifth = (blocks[:, 16:48].reshape(count, 1, 32) >> _shifts(blocks.device, *range(8)).reshape(1, 8, 1)) & 1
    return (step * (_nibbles(blocks[:, 48:176], 4) | (fifth << 4)) - offset).reshape(count, SUPER_BLOCK)


def _q6_k(blocks: torch.Tensor) -> torch.Tensor:
    # 210 bytes: 128 bytes of low nibbles, 64 bytes of 2-bit high parts, 16 signed scales (one per 16
    # values), float16 d. Each half of 128 values takes 64 low-nibble bytes and 32 high-part bytes:
    # value 32k + l (k = 0..3) has the low nibble of byte l (k = 0), l + 32 (k = 1) or the high nibble
    # of those (k = 2, 3), and bits 2k..2k+1 of high-part byte l.
    count = blocks.shape[0]
    low = blocks[:, 0:128].reshape(count, 2, 1, 64)
    low = ((low >> _shifts(blocks.device, 0, 4).reshape(1, 1, 2, 1)) & 0x0F).reshape(count, 2, 128)
    high = blocks[:, 128:192].reshape(count, 2, 1, 32)
    high = ((high >> _shifts(blocks.device, 0, 2, 4, 6).reshape(1, 1, 4, 1)) & 0x03).reshape(count, 2, 128)
    values = (low | (high << 4)).to(torch.int8) - 32
    step = (_half(blocks, 208) * blocks[:, 192:208].view(torch.int8)).reshape(count, 16, 1)
    return (step * values.reshape(count, 16, 16)).reshape(count, SUPER_BLOCK)


_BLOCK_FUNCTIONS: Dict[Any, Callable[[torch.Tensor], torch.Tensor]] = {
    QT.Q8_0: _q8_0, QT.Q4_K: _q4_k, QT.Q5_K: _q5_k, QT.Q6_K: _q6_k,
}
PLAIN_TYPES: Dict[Any, torch.dtype] = {QT.F32: torch.float32, QT.F16: torch.float16,
                                       QT.BF16: torch.bfloat16}
BLOCK_QUANTS: FrozenSet = frozenset(_BLOCK_FUNCTIONS)


def dequantize(data: torch.Tensor, qtype, oshape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Expand a GGUF tensor into a regular one of shape `oshape` and dtype `dtype`.

    `data` is what `GGUFFile` hands out: typed for F32/F16, raw uint8 rows (last dimension in
    bytes) for everything else. It may live on any device; the result stays on it."""
    oshape = tuple(oshape)
    if qtype in PLAIN_TYPES:
        plain = PLAIN_TYPES[qtype]
        if data.dtype != plain:  # raw bytes (BF16 arrives as uint8)
            data = data.reshape((-1, data.shape[-1])).view(plain)
        return data.reshape(oshape).to(dtype)
    function = _BLOCK_FUNCTIONS.get(qtype)
    if function is None:
        name = getattr(qtype, "name", repr(qtype))
        raise NotImplementedError(f"GGUF quantization {name} is not supported by the engine.")
    type_size = gguf.GGML_QUANT_SIZES[qtype][1]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    blocks = rows.reshape((rows.numel() // type_size, type_size))
    return function(blocks).reshape(oshape).to(dtype)


@dataclass(frozen=True)
class QTensor:
    """One tensor of a GGUF file: its quantization, torch-order shape and memory-mapped bytes."""
    name: str
    qtype: Any
    shape: Tuple[int, ...]
    data: torch.Tensor

    @property
    def quantized(self) -> bool:
        return self.qtype in BLOCK_QUANTS

    def to(self, dtype: torch.dtype = torch.bfloat16, device=None) -> torch.Tensor:
        data = self.data if device is None else self.data.to(device)
        return dequantize(data, self.qtype, self.shape, dtype)


class GGUFError(ValueError):
    """The file is not a well-formed GGUF file, or it is cut short."""


# GGUF value types that have a fixed size in bytes (UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
# UINT64, INT64, FLOAT64); STRING (8) and ARRAY (9) are variable.
_FIXED_VALUE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY = 8, 9


def check_layout(path: str | Path) -> None:
    """Walk the header of a GGUF file with every length checked against the size of the file.

    The gguf package trusts the counts and lengths it reads: an array that claims a billion entries makes it
    loop (and allocate) until the memory is gone, and a truncated download only fails later, in numpy, without
    a name. This raises GGUFError instead, naming the file and, for cut-short data, the first tensor that does
    not fit. Only the header is read.
    """
    path = Path(path)
    size = path.stat().st_size
    if size < 24:
        raise GGUFError(f"{path.name}: {size} bytes are too few for a GGUF file")
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        position = 0

        def need(count: int, what: str) -> None:
            if count < 0 or position + count > size:
                raise GGUFError(f"{path.name}: {what} runs past the end of the file ({size} bytes)")

        def unpack(fmt: str, what: str):
            nonlocal position
            length = struct.calcsize(fmt)
            need(length, what)
            values = struct.unpack_from(fmt, view, position)
            position += length
            return values

        def string(what: str, keep: bool = False) -> str:
            nonlocal position
            (length,) = unpack("<Q", what)
            need(length, what)
            text = bytes(view[position:position + length]).decode("utf-8", "replace") if keep else ""
            position += length
            return text

        def skip(kind: int, depth: int = 0) -> None:
            nonlocal position
            if kind in _FIXED_VALUE_SIZES:
                need(_FIXED_VALUE_SIZES[kind], "a metadata value")
                position += _FIXED_VALUE_SIZES[kind]
            elif kind == _STRING:
                string("a metadata string")
            elif kind == _ARRAY and depth < 4:
                inner, count = unpack("<IQ", "an array header")
                if inner in _FIXED_VALUE_SIZES:
                    need(count * _FIXED_VALUE_SIZES[inner], f"an array of {count} entries")
                    position += count * _FIXED_VALUE_SIZES[inner]
                elif inner in (_STRING, _ARRAY):
                    need(count * 8, f"an array of {count} entries")  # every entry takes at least its 8-byte length
                    for _ in range(count):
                        skip(inner, depth + 1)
                else:
                    raise GGUFError(f"{path.name}: unknown array element type {inner}")
            else:
                raise GGUFError(f"{path.name}: unknown metadata value type {kind}")

        (magic,) = unpack("<4s", "the magic number")
        if magic != b"GGUF":
            raise GGUFError(f"{path.name} is not a GGUF file (it starts with {magic!r})")
        (version,) = unpack("<I", "the version")
        if version not in (2, 3):
            raise GGUFError(f"{path.name}: GGUF version {version} is not supported (little-endian version 2 or 3)")
        tensor_count, kv_count = unpack("<QQ", "the entry counts")
        if tensor_count * 24 > size or kv_count * 13 > size:  # each entry has a minimum size
            raise GGUFError(f"{path.name}: {tensor_count} tensors and {kv_count} metadata entries cannot fit "
                            f"in {size} bytes")
        alignment = 32
        for _ in range(kv_count):
            key = string("a metadata key", keep=True)
            (kind,) = unpack("<I", "a metadata type")
            if key == "general.alignment" and kind in (4, 10):
                (alignment,) = unpack("<I" if kind == 4 else "<Q", "the alignment")
                if alignment < 1 or alignment & (alignment - 1):
                    raise GGUFError(f"{path.name}: alignment {alignment} is not a power of two")
            else:
                skip(kind)
        infos = []
        for _ in range(tensor_count):
            name = string("a tensor name", keep=True)
            (n_dims,) = unpack("<I", "a tensor header")
            if n_dims > 8:
                raise GGUFError(f"{path.name}: tensor {name!r} claims {n_dims} dimensions")
            dims = unpack(f"<{n_dims}Q", "tensor dimensions") if n_dims else ()
            kind, offset = unpack("<IQ", "a tensor header")
            infos.append((name, dims, kind, offset))
        data_start = -(-position // alignment) * alignment
        for name, dims, kind, offset in infos:
            try:
                block_size, type_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(kind)]
            except (ValueError, KeyError):
                raise GGUFError(f"{path.name}: tensor {name!r} has the unknown type {kind}") from None
            count = math.prod(dims)
            if count % block_size:
                raise GGUFError(f"{path.name}: tensor {name!r} has {count} values, not a multiple of "
                                f"the block size {block_size}")
            if count and data_start + offset + count // block_size * type_size > size:  # an empty tensor needs no data
                raise GGUFError(f"{path.name}: the data of tensor {name!r} lies beyond the end of the file "
                                f"({size} bytes): the download is probably incomplete")


class GGUFFile:
    """A memory-mapped GGUF file. Nothing is read until a tensor is used."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        check_layout(self.path)  # before the reader trusts any count or length in the header
        self._reader = gguf.GGUFReader(str(self.path))
        self.tensors: Dict[str, QTensor] = {}
        with warnings.catch_warnings():  # read-only memmaps: torch warns, we never write
            warnings.simplefilter("ignore")
            for tensor in self._reader.tensors:
                shape = tuple(int(d) for d in reversed(tensor.shape.tolist()))
                self.tensors[tensor.name] = QTensor(tensor.name, tensor.tensor_type, shape,
                                                    torch.from_numpy(tensor.data))
        self.arch: Optional[str] = self.field("general.architecture")

    def field(self, key: str, default: Any = None) -> Any:
        field = self._reader.fields.get(key)
        if field is None:
            return default
        contents = getattr(field, "contents", None)
        if contents is None:  # older gguf versions have no ReaderField.contents()
            return default
        return contents()  # a corrupt field must raise, not turn into None

    def close(self) -> None:
        """Forget the tensors. The memory map itself lives as long as any QTensor or view of one
        does (on Windows the file cannot be deleted before that): drop those references first."""
        self.tensors.clear()
        self._reader = None  # type: ignore[assignment]

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
