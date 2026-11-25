# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Run a model whose weights do not fit the GPU by lending the GPU to one block at a time.

The repeated blocks of a big transformer (40 of them in Wan 2.2 A14B) keep their weights in host memory
(pinned, so copies are fast and asynchronous). Just before a block runs, its weights are copied to the
device; while it computes, the copies of the next blocks are already under way on a second CUDA stream;
when it is done, the device copy is dropped (the host tensors never changed, so nothing is copied back).
Only a few blocks plus the activations are on the GPU at any moment. Quantized layers work as well: their
bytes travel like any other weight.

Two details keep the memory bounded, on the host and on the device:

* The CUDA host allocator rounds every pinned allocation up to a power of two and keeps freed blocks for
  reuse. Pinning each weight tensor on its own would waste about 40 percent (16 GB for an 11.5 GB expert),
  so the weights are packed into pinned chunks whose size is itself a power of two (`_Arena`).
* Memory on the device is taken when a copy is queued, not when the GPU performs it. A host thread that
  runs far ahead of the GPU would queue the copies of every block at once, so the host waits until the
  block that ran `prefetch + 2` blocks ago has finished (an event per block).

Everything outside the blocks (embeddings, the output head) is expected to be on the device already.
"""
from __future__ import annotations

import collections
import warnings
from typing import Callable, Deque, List, Optional, Tuple

import torch
from torch import Tensor, nn

from ..formats.ggml import QTensor
from .quant import QLinear

Transfer = Callable[[Tensor], Tensor]
ChunkAllocator = Callable[[int], Tensor]  # a size in bytes -> a uint8 buffer of exactly that size

CHUNK_BYTES = 256 << 20  # a power of two, so the host allocator does not round it up
ALIGNMENT = 256  # every packed tensor starts on a multiple of this, which any element size divides


def _pinned_chunk(size: int) -> Tensor:
    return torch.empty(size, dtype=torch.uint8, pin_memory=True)


def release_host_cache() -> None:
    """Give back the pinned memory that the CUDA host allocator keeps for reuse after tensors are freed."""
    empty = getattr(torch._C, "_host_emptyCache", None)
    if empty is not None:
        try:
            empty()
        except RuntimeError:  # a cache that cannot be emptied only costs memory until the process ends
            pass


class _Arena:
    """Packs tensors into a few big host buffers so that pinned memory is not wasted on rounding.

    `allocate(size)` returns a uint8 buffer; None means "do not copy, leave the tensors where they are"
    (nothing is pinned, e.g. without a GPU). When an allocation fails (the system would not lock that
    much memory) the tensors that no longer fit stay pageable: they still work, only slower. They are
    counted in `pageable_bytes`."""

    def __init__(self, allocate: Optional[ChunkAllocator]):
        self.allocate = allocate
        self.chunks: List[Tensor] = []
        self.used: List[int] = []
        self.packed_bytes = 0
        self.pageable_bytes = 0
        self._refused = False

    def place(self, tensor: Tensor) -> Tensor:
        """The values of `tensor` in arena memory (same dtype and shape), or the host tensor itself."""
        tensor = tensor.detach().cpu()
        if self.allocate is None:
            return tensor
        tensor = tensor.contiguous()
        size = tensor.numel() * tensor.element_size()
        if size == 0:
            return tensor
        spot = self._find(size)
        if spot is None:
            self.pageable_bytes += size
            return tensor
        chunk, offset = spot
        view = chunk[offset:offset + size].view(tensor.dtype).reshape(tensor.shape)
        view.copy_(tensor)
        self.packed_bytes += size
        return view

    def _find(self, size: int) -> Optional[Tuple[Tensor, int]]:
        for index, chunk in enumerate(self.chunks):
            start = -(-self.used[index] // ALIGNMENT) * ALIGNMENT
            if start + size <= chunk.numel():
                self.used[index] = start + size
                return chunk, start
        if self._refused:
            return None
        length = max(CHUNK_BYTES, 1 << (size - 1).bit_length())  # a power of two in every case
        try:
            chunk = self.allocate(length)
        except RuntimeError:
            self._refused = True  # do not ask again: the answer will not change
            return None
        self.chunks.append(chunk)
        self.used.append(size)
        return chunk, 0


class _Slots:
    """The tensors of one block that travel: parameters, buffers and the bytes of quantized layers."""

    def __init__(self, block: nn.Module, arena: _Arena):
        self.items: List[Tuple[Callable[[], Tensor], Callable[[Tensor], None]]] = []
        self.host: List[Tensor] = []
        for tensor_owner in block.modules():
            if isinstance(tensor_owner, QLinear):
                self._add(lambda m=tensor_owner: m.qweight.data, lambda t, m=tensor_owner: self._set_bytes(m, t), arena)
        for tensor in [*block.parameters(), *block.buffers()]:
            if not tensor.is_meta:
                self._add(lambda t=tensor: t.data, lambda value, t=tensor: setattr(t, "data", value), arena)

    @staticmethod
    def _set_bytes(layer: QLinear, data: Tensor) -> None:
        q = layer.qweight
        layer.qweight = QTensor(q.name, q.qtype, q.shape, data)

    def _add(self, get, put, arena: _Arena) -> None:
        host = arena.place(get())
        put(host)
        self.items.append((get, put))
        self.host.append(host)

    def bind(self, transfer: Transfer) -> List[Tensor]:
        moved = [transfer(host) for host in self.host]
        for (_, put), tensor in zip(self.items, moved):
            put(tensor)
        return moved

    def unbind(self) -> None:
        for (_, put), host in zip(self.items, self.host):
            put(host)


class BlockStreamer:
    """Hooks a list of blocks so that they run on `device` with their weights streamed in.

    `prefetch` is how many blocks ahead are copied while one computes, `resident` how many of the first
    blocks stay on the device for good. `transfer` replaces the host-to-device copy, `make_event` the
    CUDA event and `arena` the pinned host memory (tests)."""

    def __init__(self, blocks: nn.ModuleList, device: torch.device | str, prefetch: int = 1, resident: int = 0,
                 pin: bool = True, transfer: Optional[Transfer] = None,
                 make_event: Optional[Callable[[], object]] = None, arena: Optional[_Arena] = None):
        self.blocks, self.prefetch, self.resident = list(blocks), prefetch, resident
        self.device = torch.device(device)
        cuda = self.device.type == "cuda" and transfer is None
        self.stream = torch.cuda.Stream(self.device) if cuda else None
        self.transfer: Transfer = transfer or (lambda t: t.to(self.device, non_blocking=True))
        self._make_event = make_event or (torch.cuda.Event if cuda else None)
        self.arena = arena or _Arena(_pinned_chunk if pin and torch.cuda.is_available() else None)
        self.slots = [_Slots(block, self.arena) for block in self.blocks]
        if self.arena.pageable_bytes:
            warnings.warn(f"{self.arena.pageable_bytes / 2**30:.1f} GiB of block weights could not be pinned "
                          f"({self.arena.packed_bytes / 2**30:.1f} GiB were); those copies are slower and block "
                          "the host thread. The system may be short of lockable memory.", RuntimeWarning)
        self.ready: dict = {}  # block index -> event (cuda) or True
        self.handles: list = []
        self._pending: Deque = collections.deque()  # compute events of blocks whose kernels may still be queued

    def _start(self, index: int) -> None:
        if index >= len(self.blocks) or index in self.ready:
            return
        if self.stream is None:
            self.slots[index].bind(self.transfer)
            self.ready[index] = True
            return
        with torch.cuda.stream(self.stream):
            moved = self.slots[index].bind(self.transfer)
            event = torch.cuda.Event()
            event.record(self.stream)
        current = torch.cuda.current_stream(self.device)
        for tensor in moved:
            tensor.record_stream(current)  # freed by the allocator only after the compute stream is done with it
        self.ready[index] = event

    def _before(self, index: int) -> None:
        # Bound how far the host runs ahead: device memory is allocated when a copy is queued.
        while self._make_event is not None and len(self._pending) > self.prefetch + 1:
            self._pending.popleft().synchronize()
        self._start(index)
        for ahead in range(1, self.prefetch + 1):
            self._start(index + ahead)
        state = self.ready[index]
        if state is not True:
            torch.cuda.current_stream(self.device).wait_event(state)

    def _after(self, index: int) -> None:
        if self._make_event is not None:
            event = self._make_event()
            event.record(torch.cuda.current_stream(self.device) if self.stream is not None else None)
            self._pending.append(event)
        if index >= self.resident:
            self.slots[index].unbind()
            self.ready.pop(index, None)
        if index == len(self.blocks) - 1:  # the next forward pass (the next sampling step) starts at block 0
            for first in range(self.prefetch):
                self._start(first)

    def attach(self) -> Callable[[], None]:
        """Install the hooks; returns a function that removes them and puts everything back to the host."""
        for index, block in enumerate(self.blocks):
            self.handles.append(block.register_forward_pre_hook(lambda module, args, kwargs, i=index: self._before(i)
                                                                or None, with_kwargs=True))
            self.handles.append(block.register_forward_hook(lambda module, args, kwargs, output, i=index:
                                                            self._after(i), with_kwargs=True))
        return self.detach

    def detach(self) -> None:
        """Remove the hooks, put every block back on the host and drop all references to the weights.

        The streamer cannot be used afterwards. Whoever still holds it (the bound method that `attach`
        returned, for instance) then no longer keeps the pinned host copies of the blocks alive."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for slots in self.slots:
            slots.unbind()
        self.ready.clear()
        self._pending.clear()
        self.blocks, self.slots, self.arena = [], [], _Arena(None)


def stream_model(model: nn.Module, device: torch.device | str, blocks: str = "blocks", prefetch: int = 1,
                 resident: int = 0, pin: bool = True, transfer: Optional[Transfer] = None) -> Callable[[], None]:
    """Prepare a model loaded on the host for streaming: everything except the repeated blocks goes to
    `device` for good, the blocks (the attribute named `blocks`) are streamed. Returns the function that
    undoes the hooks and puts the blocks back on the host."""
    device = torch.device(device)
    for name, child in model.named_children():
        if name != blocks:
            child.to(device)
    for parameter in model.parameters(recurse=False):
        parameter.data = parameter.data.to(device)
    for name, buffer in list(model.named_buffers(recurse=False)):
        model._buffers[name] = buffer.to(device)
    return BlockStreamer(getattr(model, blocks), device, prefetch, resident, pin, transfer).attach()
