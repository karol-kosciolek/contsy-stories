# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""How many layers of a GGUF model go to the GPU: a header reader, a memory estimate and the arithmetic.

llama.cpp's `-ngl auto` (the `--fit` feature) loads as many layers as the free video memory allows, keeping a
margin of 1024 MiB. llama-cpp-python has no such mode: `n_gpu_layers` is a number. This module computes it before
the model is loaded.

What is taken from llama.cpp (ggml-org, MIT; see THIRD_PARTY_NOTICES.md): the GGUF file layout (version 2 and
3), the rule that the last `n` repeating blocks are the ones offloaded (`n_gpu_layers`), that the output layer
joins them only when `n` exceeds the block count, that the token embedding stays in system memory, and the 1024
MiB default margin of `--fit-target`. The header reader is written here and needs neither torch nor the `gguf`
package, because the worker process must start fast; every length is checked against the file size (the same
discipline as engine/formats/ggml.py `check_layout`), so a damaged download raises `GGUFHeaderError`.

Numbers, and where they come from. File sizes of the tensors are read from the header (a tensor ends where the
next begins). The key-value cache and the recurrent state follow the architecture keys of the file; for the
Qwen3.5 hybrid (a full attention block every `full_attention_interval`-th, linear-attention blocks in between)
only the attention blocks hold a KV cache. `COMPUTE_MIB` and `VISION_RUNTIME_MIB` are measurements of
llama-cpp-python 0.3.36 with Qwen3.5-9B Q6_K on a Vulkan build (8 November 2026, RTX 4070 Ti SUPER 16 GB): about 7.0
GiB for the text model at a 4096-token context (6.3 GiB of weights on the card, 0.2 GiB of cache and state, the
rest compute buffers), and 5.3 GiB more after calls with 1024 px images. They were measured on one card and one
model; other GPUs and drivers will differ, so the margin is generous and a failed load steps down (`ladder`).
"""
from __future__ import annotations

import mmap
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

MIB = 1024 * 1024

RESERVE_MIB = 1024  # llama.cpp --fit-target default: room for the desktop and other programs
COMPUTE_MIB = 640  # compute buffers at 512 tokens per micro-batch (~500 MiB measured), a little rounded up
VISION_RUNTIME_MIB = 4400  # encoder scratch of 1024 px images and the larger decode batches (5295 MiB measured, minus the 875 MiB file)
UNKNOWN_STATE_MIB = 4.0  # recurrent state of one linear-attention block when the file does not say (measured: 2.1)


class GGUFHeaderError(ValueError):
    """The file is not a well-formed GGUF file, or it is cut short."""


_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_SIZES = {kind: struct.calcsize(fmt) for kind, fmt in _SCALARS.items()}
_STRING, _ARRAY = 8, 9
_MAX_KEPT_STRING = 1 << 16  # the chat template (8 KB) is kept; nothing longer is
_MAX_KEPT_ARRAY = 4096  # per-layer head counts are kept; the 250 000-entry token lists are skipped


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: Tuple[int, ...]
    ggml_type: int
    offset: int  # from the start of the data section
    nbytes: int  # up to the next tensor (alignment padding included, at most 31 bytes)


@dataclass(frozen=True)
class GGUFInfo:
    path: str
    file_size: int
    version: int
    data_start: int
    alignment: int
    meta: Mapping[str, Any]  # numbers, booleans, strings up to 64 KiB and arrays of up to 4096 numbers
    tensors: Tuple[TensorInfo, ...]

    @property
    def arch(self) -> str:
        return str(self.meta.get("general.architecture", ""))

    def key(self, name: str, default: Any = None) -> Any:
        """A key of the model's architecture, e.g. key("block_count") reads `qwen35.block_count`."""
        return self.meta.get(f"{self.arch}.{name}", default)


class _Cursor:
    def __init__(self, view: Any, size: int, label: str) -> None:
        self.view, self.size, self.label, self.pos = view, size, label, 0

    def need(self, count: int, what: str) -> None:
        if count < 0 or self.pos + count > self.size:
            raise GGUFHeaderError(f"{self.label}: {what} runs past the end of the file ({self.size} bytes)")

    def unpack(self, fmt: str, what: str) -> tuple:
        length = struct.calcsize(fmt)
        self.need(length, what)
        values = struct.unpack_from(fmt, self.view, self.pos)
        self.pos += length
        return values

    def string(self, what: str, keep: bool = True) -> str:
        (length,) = self.unpack("<Q", what)
        self.need(length, what)
        text = bytes(self.view[self.pos:self.pos + length]).decode("utf-8", "replace") if keep else ""
        self.pos += length
        return text

    def skip(self, kind: int, depth: int = 0) -> None:
        if kind in _SIZES:
            self.need(_SIZES[kind], "a value")
            self.pos += _SIZES[kind]
        elif kind == _STRING:
            self.string("a string", keep=False)
        elif kind == _ARRAY and depth < 3:
            inner, count = self.unpack("<IQ", "an array header")
            if inner in _SIZES:
                self.need(count * _SIZES[inner], f"an array of {count} entries")
                self.pos += count * _SIZES[inner]
            elif inner in (_STRING, _ARRAY):
                self.need(count * 8, f"an array of {count} entries")  # an entry takes at least its 8-byte length
                for _ in range(count):
                    self.skip(inner, depth + 1)
            else:
                raise GGUFHeaderError(f"{self.label}: unknown array element type {inner}")
        else:
            raise GGUFHeaderError(f"{self.label}: unknown value type {kind}")

    def value(self, kind: int) -> Any:
        if kind in _SCALARS:
            return self.unpack(_SCALARS[kind], "a value")[0]
        if kind == _STRING:
            start = self.pos
            (length,) = self.unpack("<Q", "a string")
            self.pos = start
            if length <= _MAX_KEPT_STRING:
                return self.string("a string")
            self.string("a string", keep=False)
            return None
        if kind == _ARRAY:
            start = self.pos
            inner, count = self.unpack("<IQ", "an array header")
            if inner in _SCALARS and count <= _MAX_KEPT_ARRAY:
                self.need(count * _SIZES[inner], f"an array of {count} entries")
                values = struct.unpack_from(f"<{count}{_SCALARS[inner][1]}", self.view, self.pos)
                self.pos += count * _SIZES[inner]
                return tuple(values)
            self.pos = start
            self.skip(kind)
            return None
        raise GGUFHeaderError(f"{self.label}: unknown value type {kind}")


def _parse(view: Any, size: int, label: str, path: str) -> GGUFInfo:
    cur = _Cursor(view, size, label)
    (magic,) = cur.unpack("<4s", "the magic number")
    if magic != b"GGUF":
        raise GGUFHeaderError(f"{label} is not a GGUF file (it starts with {magic!r})")
    (version,) = cur.unpack("<I", "the version")
    if version not in (2, 3):
        raise GGUFHeaderError(f"{label}: GGUF version {version} is not supported")
    tensor_count, kv_count = cur.unpack("<QQ", "the entry counts")
    if tensor_count * 24 > size or kv_count * 13 > size:
        raise GGUFHeaderError(f"{label}: {tensor_count} tensors and {kv_count} entries cannot fit in {size} bytes")
    meta: Dict[str, Any] = {}
    alignment = 32
    for _ in range(kv_count):
        key = cur.string("a key")
        (kind,) = cur.unpack("<I", "a value type")
        value = cur.value(kind)
        if value is not None:
            meta[key] = value
        if key == "general.alignment":
            if not isinstance(value, int) or value < 1 or value & (value - 1):
                raise GGUFHeaderError(f"{label}: alignment {value!r} is not a power of two")
            alignment = value
    raw: List[Tuple[str, Tuple[int, ...], int, int]] = []
    for _ in range(tensor_count):
        name = cur.string("a tensor name")
        (n_dims,) = cur.unpack("<I", "a tensor header")
        if n_dims > 8:
            raise GGUFHeaderError(f"{label}: tensor {name!r} claims {n_dims} dimensions")
        dims = cur.unpack(f"<{n_dims}Q", "tensor dimensions") if n_dims else ()
        kind, offset = cur.unpack("<IQ", "a tensor header")
        raw.append((name, tuple(dims), kind, offset))
    data_start = -(-cur.pos // alignment) * alignment
    order = sorted(range(len(raw)), key=lambda i: raw[i][3])
    sizes = [0] * len(raw)
    for rank, i in enumerate(order):
        end = raw[order[rank + 1]][3] if rank + 1 < len(order) else size - data_start
        sizes[i] = end - raw[i][3]
        if sizes[i] < 0 or data_start + raw[i][3] > size:
            raise GGUFHeaderError(f"{label}: the data of tensor {raw[i][0]!r} lies beyond the end of the file "
                                  f"({size} bytes): the download is probably incomplete")
    tensors = tuple(TensorInfo(name, dims, kind, offset, sizes[i]) for i, (name, dims, kind, offset) in enumerate(raw))
    return GGUFInfo(path, size, version, data_start, alignment, meta, tensors)


def read_gguf_info(path: "str | Path") -> GGUFInfo:
    """Read the metadata and the tensor table of a GGUF file (nothing else is read from the weights)."""
    path = Path(path)
    size = path.stat().st_size
    if size < 24:
        raise GGUFHeaderError(f"{path.name}: {size} bytes are too few for a GGUF file")
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        return _parse(view, size, path.name, str(path))


@lru_cache(maxsize=8)
def _cached(path: str, size: int, mtime_ns: int) -> GGUFInfo:
    return read_gguf_info(path)


def read_gguf_info_cached(path: "str | Path") -> GGUFInfo:
    """`read_gguf_info`, remembered per file identity (a model is read again on every reload)."""
    stat = Path(path).stat()
    return _cached(str(path), stat.st_size, stat.st_mtime_ns)


# --- memory estimate -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Footprint:
    """Bytes that go to the GPU, per repeating block and for the output layer, at a given context length."""
    n_layers: int
    block_bytes: Tuple[int, ...]  # weights of block i plus its cache or recurrent state
    output_bytes: int  # output norm and matrix (and the tied token embedding when there is no separate matrix)
    host_bytes: int  # token embedding and small tensors: stay in system memory

    @property
    def total_mib(self) -> float:
        return (sum(self.block_bytes) + self.output_bytes) / MIB


_BLOCK = re.compile(r"blk\.(\d+)\.")
_OUTPUT = ("output.weight", "output_norm.weight", "output_norm.bias", "output.bias")


def _state_bytes(info: GGUFInfo, layer: int, n_ctx: int, kv_bytes: int = 2) -> int:
    """Cache (attention block) or recurrent state (linear-attention block) of one block, in bytes."""
    n_embd = int(info.key("embedding_length", 0) or 0)
    n_head = info.key("attention.head_count", 0)
    n_head = int(n_head) if isinstance(n_head, int) else 0
    interval = info.key("full_attention_interval")
    if isinstance(interval, int) and interval > 0 and (layer + 1) % interval != 0:
        inner, state = info.key("ssm.inner_size"), info.key("ssm.state_size")
        conv, groups = info.key("ssm.conv_kernel"), info.key("ssm.group_count")
        if all(isinstance(v, int) for v in (inner, state, conv, groups)):
            return 4 * ((conv - 1) * (inner + 2 * groups * state) + state * inner)
        return int(UNKNOWN_STATE_MIB * MIB)
    heads_kv = info.key("attention.head_count_kv", n_head)
    if isinstance(heads_kv, tuple):  # per-block counts
        heads_kv = heads_kv[layer] if layer < len(heads_kv) else max(heads_kv or (0,))
    heads_kv = int(heads_kv or 0)
    key_len = info.key("attention.key_length") or (n_embd // n_head if n_head else 0)
    value_len = info.key("attention.value_length") or key_len
    return int(n_ctx) * heads_kv * (int(key_len) + int(value_len)) * kv_bytes


def footprint(info: GGUFInfo, n_ctx: int, kv_bytes: int = 2) -> Footprint:
    """Split the model into what an offload moves: blocks (with their cache) and the output layer."""
    count = info.key("block_count")
    if not isinstance(count, int) or count < 1:
        raise GGUFHeaderError(f"{Path(info.path).name}: the file does not say how many blocks the model has")
    blocks = [0] * count
    output = embedding = other = 0
    names = {t.name for t in info.tensors}
    for tensor in info.tensors:
        match = _BLOCK.match(tensor.name)
        if match and int(match.group(1)) < count:
            blocks[int(match.group(1))] += tensor.nbytes
        elif tensor.name in _OUTPUT:
            output += tensor.nbytes
        elif tensor.name == "token_embd.weight":
            embedding = tensor.nbytes
        else:
            other += tensor.nbytes
    if "output.weight" in names:
        other += embedding
    else:  # tied: the output projection is the token embedding, offloaded with the output layer
        output += embedding
    blocks = [blocks[i] + _state_bytes(info, i, n_ctx, kv_bytes) for i in range(count)]
    return Footprint(count, tuple(blocks), output, other)


# --- the choice ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuPlan:
    n_gpu_layers: int  # for llama.cpp: 0 = none, n = the last n blocks, n_layers + 1 or -1 = everything
    n_layers: int
    estimated_mib: float  # what the chosen offload is expected to take on the card
    free_mib: Optional[float]
    note: str

    @property
    def all_layers(self) -> bool:
        return self.n_gpu_layers < 0 or self.n_gpu_layers > self.n_layers


def plan_gpu_layers(fp: Footprint, free_mib: Optional[float], *, compute_mib: float = COMPUTE_MIB,
                    reserve_mib: float = RESERVE_MIB, vision_mib: float = 0.0, unknown: str = "all") -> GpuPlan:
    """The largest offload whose estimate fits in `free_mib` minus the margin; 0 when not even a block fits.

    `vision_mib` is the memory the vision encoder will need once the first picture arrives (its file plus the
    scratch space of a large picture), so that the first image does not run the card out of memory. When the
    free memory is unknown (no NVIDIA driver, or no pynvml) `unknown` decides: "all" offloads everything and
    lets llama.cpp use whatever device it finds (a load failure then steps down), "none" stays on the CPU."""
    if free_mib is None:
        if unknown == "none":
            return GpuPlan(0, fp.n_layers, 0.0, None, "free video memory unknown: CPU only")
        return GpuPlan(-1, fp.n_layers, fp.total_mib + compute_mib + vision_mib, None,
                       "free video memory unknown: all layers")
    budget = free_mib - reserve_mib - compute_mib - vision_mib
    used = 0.0
    n = 0
    for k in range(1, fp.n_layers + 1):  # the last k blocks
        step = fp.block_bytes[fp.n_layers - k] / MIB
        if used + step > budget:
            break
        used += step
        n = k
    else:
        if used + fp.output_bytes / MIB <= budget:
            return GpuPlan(-1, fp.n_layers, used + fp.output_bytes / MIB + compute_mib + vision_mib, free_mib,
                           "everything fits")
    if n == 0:
        return GpuPlan(0, fp.n_layers, 0.0, free_mib, "not even one block fits next to the margin: CPU only")
    return GpuPlan(n, fp.n_layers, used + compute_mib + vision_mib, free_mib,
                   f"{n} of {fp.n_layers} blocks fit")


def ladder(plan: GpuPlan) -> List[int]:
    """Offloads to try one after another when a load fails for lack of memory, from the plan down to the CPU."""
    top = plan.n_layers + 1 if plan.all_layers else plan.n_gpu_layers
    steps = [top] + [top * share // 4 for share in (3, 2, 1)] + [0]
    seen: List[int] = []
    for n in steps:
        if n not in seen and n >= 0:
            seen.append(n)
    return [(-1 if n > plan.n_layers else n) for n in seen]


def query_free_vram_mib() -> Optional[float]:
    """Free memory of the first NVIDIA card in MiB, or None (no driver, no pynvml, no card, any failure)."""
    try:
        import pynvml  # nvidia-ml-py
    except Exception:  # noqa: BLE001
        return None
    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001
        return None
    try:
        if pynvml.nvmlDeviceGetCount() < 1:
            return None
        return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / MIB
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
