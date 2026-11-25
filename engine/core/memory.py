# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Bounded reuse of decoded model weights in host memory between engine jobs, and of prompt encodings."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch

from .quant import QLinear


def limit_cuda(device: str, gib: int | None) -> None:
    """Apply an optional per-process ceiling without raising a stricter existing limit."""
    if gib is None or not device.startswith("cuda"):
        return
    index = torch.device(device).index
    _, total = torch.cuda.mem_get_info(index)
    current = torch.cuda.get_per_process_memory_fraction(index)
    fraction = min(current, (gib << 30) / total)
    if fraction < current:
        torch.cuda.set_per_process_memory_fraction(fraction, index)


def model_bytes(model: torch.nn.Module) -> int:
    """Bytes of unique parameter, buffer and quantized-weight storages."""
    tensors = [*model.parameters(), *model.buffers()]
    tensors.extend(module.qweight.data for module in model.modules() if isinstance(module, QLinear))
    storages = {}
    for tensor in tensors:
        if not tensor.is_meta and tensor.numel():
            storage = tensor.untyped_storage()
            storages[tensor.device, storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def file_identity(path) -> tuple:
    """What tells one weight file from another, or from a replaced one at the same path."""
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return str(resolved), stat.st_size, stat.st_mtime_ns


class EmbeddingCache:
    """The last few text-encoder outputs, on the CPU, by (encoder file identity, dtype, text).

    The parts of one clip share their prompt, so only the first part loads the text encoder (Wan's UMT5 in the
    Full set: 11.4 GB of bf16 on disk and in memory); an encoding takes a few megabytes. The encoder is
    deterministic, so a kept encoding is exactly what encoding the text again would give.

    A job that knows the texts it will ask for (the prompts of a Timeline job's scenes) announces them with
    `expect`: the pipeline that next loads the encoder encodes them too, in the same load, and room is kept for
    them until the job ends (`expect(())`)."""

    MAX_EXPECTED = 64  # about 8 MiB each (Wan: 512 x 4096 float32)

    def __init__(self, entries: int = 8):
        self.entries = entries
        self._items: OrderedDict = OrderedDict()
        self._expected: tuple = ()

    def get(self, key):
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def __contains__(self, key) -> bool:
        return key in self._items

    def put(self, key, value: torch.Tensor) -> None:
        self._items[key] = value.detach().to("cpu")
        self._items.move_to_end(key)
        self._trim()

    def expect(self, texts) -> None:
        """The texts about to be asked for, in order (an empty sequence: none). They are only a list of
        strings; whoever loads the encoder next encodes the ones not kept yet (`expected`)."""
        self._expected = tuple(dict.fromkeys(text for text in texts if isinstance(text, str) and text))
        self._expected = self._expected[:self.MAX_EXPECTED]
        self._trim()

    def expected(self) -> tuple:
        return self._expected

    def consume(self, text) -> None:
        """`text` was asked for: it is no longer expected (its encoding stays among the most recent entries)."""
        if text in self._expected:
            self._expected = tuple(item for item in self._expected if item != text)

    def _trim(self) -> None:
        while len(self._items) > self.entries + len(self._expected):
            self._items.popitem(last=False)

    def clear(self) -> None:
        """Drop the encodings (memory); the announced texts stay expected until `expect(())`."""
        self._items.clear()


class ModelCache:
    """Lease one model per stage under the Engine job lock. Idle entries are always on the CPU.

    Identity includes file metadata, compute dtype and architecture role. Replacing a checkpoint at the
    same path invalidates its old entry. Errors during a job discard its cache via Engine.free().
    A `spare` model (a text encoder, whose output EmbeddingCache keeps) never pushes other entries out:
    it is reused while idle here and kept afterwards only when it fits beside them. A model loaded with
    `keep=False` makes room like any other (the cache's bound covers it while it runs) but is dropped after
    use: models that run in turn and cannot all be kept (`fits`) would only push each other out unused.
    """

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self._idle = OrderedDict()
        self._active = {}
        self._kind = None
        self.bytes = 0

    def prepare(self, kind: str) -> None:
        if kind != self._kind:
            self.clear()
            self._kind = kind

    def fits(self, *paths) -> bool:
        """Whether the files at `paths` fit in the cache together (by file size, as `load` makes room)."""
        return bool(self.capacity_bytes) and sum(Path(path).stat().st_size for path in paths) <= self.capacity_bytes

    def _discard(self, key):
        model, size = self._idle.pop(key)
        self.bytes -= size
        return model

    def load(self, path, role: str, loader, device, dtype, *, spare: bool = False, keep: bool = True):
        if not self.capacity_bytes:
            return loader(path, device, dtype)
        resolved = Path(path).resolve(strict=True)
        stat = resolved.stat()
        key = (resolved, role, dtype, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        for old in tuple(self._idle):
            if old[:3] == key[:3] and old != key:
                self._discard(old)
        if key in self._idle:
            model = self._discard(key).to(device)
        else:
            while not spare and self._idle and self.bytes + stat.st_size > self.capacity_bytes:
                self._discard(next(iter(self._idle)))
            model = loader(path, device, dtype)
        if keep:
            self._active[id(model)] = key, spare
        return model

    def release(self, model: torch.nn.Module) -> None:
        key, spare = self._active.pop(id(model), (None, False))
        if key is None:
            return
        size = model_bytes(model)
        if size > self.capacity_bytes:
            return
        if key in self._idle:
            self._discard(key)
        if spare and self.bytes + size > self.capacity_bytes:
            return  # dropped: the denoisers it would push out cost more to load again
        while self.bytes + size > self.capacity_bytes:
            self._discard(next(iter(self._idle)))
        self._idle[key] = model.to("cpu"), size
        self.bytes += size

    def clear(self) -> None:
        self._idle.clear()
        self._active.clear()
        self.bytes = 0
