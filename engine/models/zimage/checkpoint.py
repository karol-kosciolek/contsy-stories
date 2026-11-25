# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The official Z-Image-Turbo checkpoint files (Tongyi-MAI/Z-Image-Turbo, Apache-2.0): reading and merging them.

Tongyi-MAI publishes the transformer as three float32 safetensors shards and the Qwen3-4B text encoder as three
bf16 shards. Each set comes with a `*.safetensors.index.json` whose "weight_map" names the shard of every tensor
(the Hugging Face sharded-checkpoint convention). The engine keeps both networks in bf16, so the installer merges
each set into one bf16 file (`merge`), and the loaders read that file. `tensors` reads a single file and a sharded
set alike.

The safetensors layout (https://github.com/huggingface/safetensors, Apache-2.0): an unsigned little-endian 64-bit
header length N; N bytes of JSON that give every tensor's dtype, shape and [begin, end) byte offsets into the data
after the header, plus an optional "__metadata__" map of strings, padded at the end with spaces; then the tensors'
bytes, little-endian and row-major, back to back. `merge` writes the tensors sorted by name from offset 0 without
gaps, with no metadata and the header padded with spaces to a multiple of 8 bytes. That is the layout
`safetensors.torch.save_file` writes for tensors of one dtype, so the same tensors always give the same bytes and
SHA-256.

Floating-point tensors become bfloat16 with torch's cast: round to nearest, ties to even. A bf16 source is copied
unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from safetensors import safe_open
from torch import Tensor

INDEX_SUFFIX = ".safetensors.index.json"
_FLOATS = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
_DTYPE_NAMES = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16"}

ProgressCb = Callable[[int, int], None]


class CheckpointError(ValueError):
    pass


def resolve(path: str | Path) -> Path:
    """The file to read: `path` itself, or the one index inside a folder of shards."""
    path = Path(path)
    if path.is_dir():
        found = sorted(path.glob("*" + INDEX_SUFFIX))
        if len(found) != 1:
            raise CheckpointError(f"{path} must hold exactly one *{INDEX_SUFFIX}, found {len(found)}")
        return found[0]
    return path


def shard_map(index: str | Path) -> Dict[str, Path]:
    """{tensor name: shard file} from a sharded checkpoint's index. Shard names must be plain file names next to the
    index, so a damaged or hostile index cannot point outside its folder."""
    index = Path(index)
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CheckpointError(f"{index.name} is not a sharded-checkpoint index: {error}") from None
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointError(f"{index.name} has an empty weight_map")
    shards = {}
    for name, shard in weight_map.items():
        if (not isinstance(shard, str) or not shard.endswith(".safetensors") or Path(shard).name != shard
                or shard in (".", "..") or "\\" in shard):
            raise CheckpointError(f"{index.name}: invalid shard name {shard!r} for {name}")
        shards[name] = index.parent / shard
    return shards


def _files(path: Path) -> List[Tuple[Path, Optional[set]]]:
    """[(safetensors file, the names the index assigns to it or None for a single file)]."""
    if path.name.endswith(INDEX_SUFFIX):
        assigned: Dict[Path, set] = {}
        for name, shard in shard_map(path).items():
            assigned.setdefault(shard, set()).add(name)
        return list(assigned.items())
    return [(path, None)]


def _check_shard(file: Path, keys, assigned: Optional[set]) -> None:
    if assigned is not None and set(keys) != assigned:
        missing, extra = sorted(assigned - set(keys)), sorted(set(keys) - assigned)
        raise CheckpointError(f"{file.name} does not match its index: missing {missing[:5]}, not listed {extra[:5]}")


def tensors(path: str | Path) -> Iterator[Tuple[str, Tensor]]:
    """(name, tensor) of one safetensors file, of a sharded set given by its index, or of a folder holding one
    index; one tensor at a time, shard after shard. Every shard must hold exactly the names its index assigns
    to it."""
    for file, assigned in _files(resolve(path)):
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            _check_shard(file, keys, assigned)
            for key in keys:
                yield key, handle.get_tensor(key)


def names(path: str | Path) -> List[str]:
    """The tensor names of a file or sharded set, from the headers only."""
    found: List[str] = []
    for file, assigned in _files(resolve(path)):
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        _check_shard(file, keys, assigned)
        found.extend(keys)
    return found


def _plan(files, dtype: torch.dtype):
    """Every tensor's source and output layout, from the headers: [(name, file, shape, nbytes)], sorted by name."""
    size = torch.empty((), dtype=dtype).element_size()
    entries = {}
    for file, assigned in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            _check_shard(file, keys, assigned)
            for key in keys:
                view = handle.get_slice(key)
                if view.get_dtype() not in _FLOATS:
                    raise CheckpointError(f"{key} is {view.get_dtype()}, only float tensors are merged")
                if key in entries:
                    raise CheckpointError(f"{key} is stored twice")
                shape = [int(n) for n in view.get_shape()]
                count = 1
                for n in shape:
                    count *= n
                entries[key] = (file, shape, count * size)
    return [(key, *entries[key]) for key in sorted(entries)]


def _header(plan, dtype: torch.dtype) -> bytes:
    offset = 0
    header = {}
    for name, _, shape, nbytes in plan:
        header[name] = {"dtype": _DTYPE_NAMES[dtype], "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return raw + b" " * (-len(raw) % 8)


def merge(source: str | Path, output: str | Path, *, dtype: torch.dtype = torch.bfloat16,
          expected_size: Optional[int] = None, expected_sha256: Optional[str] = None,
          progress_cb: Optional[ProgressCb] = None, cancel_flag: Optional[Callable[[], bool]] = None) -> str:
    """Write the tensors of `source` (a file, a sharded set's index, or a folder with one index) as one safetensors
    file of `dtype`, one tensor at a time. Returns the SHA-256 of the written file.

    The file is written as `<output>.part` and renamed to `output` only when its size and SHA-256 equal
    `expected_size` / `expected_sha256` (when given). The size is checked before anything is written, so a changed
    source fails at once. A failed or cancelled run leaves the `.part` file, which the next run overwrites.
    """
    if sys.byteorder != "little":
        raise CheckpointError("safetensors data is little-endian; this machine is not")
    if dtype not in _DTYPE_NAMES:
        raise CheckpointError(f"unsupported output dtype {dtype}")
    output = Path(output)
    files = _files(resolve(source))
    plan = _plan(files, dtype)
    header = _header(plan, dtype)
    total = 8 + len(header) + sum(entry[3] for entry in plan)
    if expected_size is not None and total != expected_size:
        raise CheckpointError(f"the merged file would have {total} bytes, the catalog expects {expected_size}")
    part = output.with_name(output.name + ".part")
    if part.is_symlink() or output.is_symlink():
        raise CheckpointError("refusing to write through a link")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with ExitStack() as stack:
        handles = {file: stack.enter_context(safe_open(str(file), framework="pt", device="cpu")) for file, _ in files}
        stream = stack.enter_context(part.open("wb"))
        for chunk in (struct.pack("<Q", len(header)), header):
            stream.write(chunk)
            digest.update(chunk)
            written += len(chunk)
        for name, file, shape, nbytes in plan:
            if cancel_flag is not None and cancel_flag():
                raise CheckpointError("conversion cancelled; run it again to start over")
            tensor = handles[file].get_tensor(name).to(dtype).contiguous()
            if list(tensor.shape) != shape:
                raise CheckpointError(f"{name} has shape {list(tensor.shape)}, its header says {shape}")
            data = tensor.reshape(-1).view(torch.uint8).numpy()
            if data.nbytes != nbytes:
                raise CheckpointError(f"{name} has {data.nbytes} bytes, expected {nbytes}")
            stream.write(data)
            digest.update(data)
            written += nbytes
            if progress_cb is not None:
                progress_cb(written, total)
        stream.flush()
        os.fsync(stream.fileno())
    sha256 = digest.hexdigest()
    if part.stat().st_size != total:
        raise CheckpointError(f"{part.name} has {part.stat().st_size} bytes, expected {total}")
    if expected_sha256 is not None and sha256 != expected_sha256.lower():
        raise CheckpointError(f"the merged file has SHA-256 {sha256}, the catalog expects {expected_sha256}; "
                              f"{part.name} is kept for inspection")
    os.replace(part, output)
    return sha256


def _main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m engine.models.zimage.checkpoint",
                                     description="Merge an official sharded checkpoint into one bf16 safetensors file.")
    parser.add_argument("source", help="a *.safetensors.index.json, a folder holding one, or a safetensors file")
    parser.add_argument("output")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--sha256", default=None)
    args = parser.parse_args(argv)
    last = [-1]

    def progress(done: int, total: int) -> None:
        percent = done * 100 // total
        if percent // 5 != last[0] // 5:
            last[0] = percent
            print(f"{percent}% ({done / 2**30:.2f} of {total / 2**30:.2f} GiB)", flush=True)

    sha256 = merge(args.source, args.output, expected_size=args.size, expected_sha256=args.sha256,
                   progress_cb=progress)
    print(f"{args.output}\nsize {Path(args.output).stat().st_size}\nsha256 {sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
