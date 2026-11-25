# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Block streaming: the mechanics (which blocks are on the device when, restoration after errors, quantized
layers) with a stand-in for the host-to-device copy, since the real copy needs a GPU."""
import importlib.util
import unittest
import unittest.mock

HAVE = all(importlib.util.find_spec(name) is not None for name in ("torch", "gguf", "numpy"))


class Block:  # built lazily so that torch is only imported when the tests run
    pass


def make_blocks(count=4, quantized_first=False):
    import gguf
    import numpy as np
    import torch
    from engine.core.quant import QLinear
    from engine.formats import ggml
    from torch import nn

    class Plain(nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.linear = nn.Linear(32, 32)
            self.register_buffer("scale", torch.rand(32) + 0.5)
            self.seen = []

        def forward(self, x):
            return torch.tanh(self.linear(x)) * self.scale

    blocks = nn.ModuleList([Plain(i) for i in range(count)])
    if quantized_first:
        weight = np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32)
        raw = gguf.quants.quantize(weight, gguf.GGMLQuantizationType.Q8_0)
        blocks[0].linear = QLinear(ggml.QTensor("w", ggml.QT.Q8_0, (32, 32), torch.from_numpy(raw)), torch.zeros(32))
    return blocks


def run(blocks, x):
    for block in blocks:
        x = block(x)
    return x


@unittest.skipUnless(HAVE, "torch or gguf is not installed")
class StreamerTests(unittest.TestCase):
    def test_the_streamed_model_gives_the_same_result_and_weights_return_to_the_host(self):
        import torch
        from engine.core.streaming import BlockStreamer

        blocks = make_blocks(quantized_first=True)
        x = torch.randn(3, 32)
        expected = run(blocks, x)
        hosts = [b.linear.bias.data for b in blocks]
        streamer = BlockStreamer(blocks, "cpu", prefetch=1, pin=False, transfer=lambda t: t.clone())
        remove = streamer.attach()
        got = run(blocks, x)
        torch.testing.assert_close(got, expected)
        for block, host in zip(blocks[1:], hosts[1:]):  # evicted: the parameters point at the host tensors again
            self.assertEqual(block.linear.bias.data.data_ptr(), host.data_ptr())
        self.assertIn(0, streamer.ready)  # the first block is already on its way for the next pass
        remove()
        for block, host in zip(blocks, hosts):  # detaching puts every block back on the host
            self.assertEqual(block.linear.bias.data.data_ptr(), host.data_ptr())
        torch.testing.assert_close(run(blocks, x), expected)
        self.assertEqual([len(b._forward_hooks) + len(b._forward_pre_hooks) for b in blocks], [0, 0, 0, 0])

    def test_only_a_few_blocks_are_on_the_device_at_any_moment(self):
        import torch
        from engine.core.streaming import BlockStreamer

        blocks = make_blocks(6)
        streamer = BlockStreamer(blocks, "cpu", prefetch=2, pin=False, transfer=lambda t: t.clone())
        peaks = []
        for block in blocks:
            block.register_forward_hook(lambda m, a, o: peaks.append(len(streamer.ready)))
        streamer.attach()
        for _ in range(3):  # three sampling steps
            run(blocks, torch.randn(2, 32))
        self.assertLessEqual(max(peaks), 3)  # the running block and two ahead
        self.assertGreaterEqual(max(peaks), 2)

    def test_resident_blocks_stay_on_the_device(self):
        import torch
        from engine.core.streaming import BlockStreamer

        blocks = make_blocks(4)
        streamer = BlockStreamer(blocks, "cpu", prefetch=1, resident=2, pin=False, transfer=lambda t: t.clone())
        streamer.attach()
        run(blocks, torch.randn(2, 32))
        self.assertTrue(0 in streamer.ready and 1 in streamer.ready)
        self.assertFalse(2 in streamer.ready or 3 in streamer.ready)

    def test_an_error_inside_a_block_leaves_nothing_behind_after_detach(self):
        import torch
        from engine.core.streaming import BlockStreamer

        blocks = make_blocks(3)
        expected = run(blocks, torch.ones(2, 32))
        streamer = BlockStreamer(blocks, "cpu", prefetch=1, pin=False, transfer=lambda t: t.clone())
        remove = streamer.attach()

        def explode(module, args):
            raise RuntimeError("out of memory")

        blocks[1].register_forward_pre_hook(explode)
        with self.assertRaises(RuntimeError):
            run(blocks, torch.ones(2, 32))
        remove()
        blocks[1]._forward_pre_hooks.clear()
        torch.testing.assert_close(run(blocks, torch.ones(2, 32)), expected)
        self.assertEqual(streamer.ready, {})


@unittest.skipUnless(HAVE, "torch or gguf is not installed")
class ArenaTests(unittest.TestCase):
    """Pinned host memory is packed into power-of-two chunks; here the chunks are ordinary memory."""

    @staticmethod
    def plain_chunk(size):
        import torch

        return torch.empty(size, dtype=torch.uint8)

    def test_tensors_are_packed_aligned_and_unchanged(self):
        import torch
        from engine.core import streaming

        original = [torch.randn(3, 5), torch.randn(7).to(torch.bfloat16), torch.randint(0, 255, (100,), dtype=torch.uint8),
                    torch.randn(4, 6).to(torch.float16), torch.randn(5, 4).t(), torch.arange(9, dtype=torch.int64)]
        arena = streaming._Arena(self.plain_chunk)
        with unittest.mock.patch.object(streaming, "CHUNK_BYTES", 1024):
            placed = [arena.place(t) for t in original]
        for source, view in zip(original, placed):
            self.assertEqual(view.dtype, source.dtype)
            self.assertEqual(view.shape, source.shape)
            self.assertTrue(torch.equal(view, source))
            self.assertNotEqual(view.data_ptr(), source.data_ptr())  # a copy in arena memory
        self.assertGreater(len(arena.chunks), 1)  # 1 KiB chunks: the tensors did not all fit in one
        for chunk in arena.chunks:
            self.assertEqual(chunk.numel() & (chunk.numel() - 1), 0)  # a power of two, so the host allocator keeps it
        for view in placed:
            owner = next(c for c in arena.chunks if c.data_ptr() <= view.data_ptr() < c.data_ptr() + c.numel())
            self.assertEqual((view.data_ptr() - owner.data_ptr()) % streaming.ALIGNMENT, 0)
        self.assertEqual(arena.packed_bytes, sum(t.numel() * t.element_size() for t in original))
        self.assertEqual(arena.pageable_bytes, 0)

    def test_a_tensor_bigger_than_a_chunk_gets_a_chunk_of_its_own(self):
        import torch
        from engine.core import streaming

        arena = streaming._Arena(self.plain_chunk)
        with unittest.mock.patch.object(streaming, "CHUNK_BYTES", 256):
            big = torch.randn(300)  # 1200 bytes
            view = arena.place(big)
        self.assertTrue(torch.equal(view, big))
        self.assertEqual(arena.chunks[0].numel(), 2048)  # the next power of two

    def test_when_pinning_is_refused_the_rest_stays_pageable_and_is_counted(self):
        import torch
        from engine.core import streaming

        asked = []

        def stingy(size):
            asked.append(size)
            if len(asked) > 1:
                raise RuntimeError("cudaHostAlloc failed: out of memory")
            return torch.empty(size, dtype=torch.uint8)

        arena = streaming._Arena(stingy)
        with unittest.mock.patch.object(streaming, "CHUNK_BYTES", 512):
            tensors = [torch.randn(100) for _ in range(5)]  # 400 bytes each: one per chunk
            placed = [arena.place(t) for t in tensors]
        self.assertTrue(all(torch.equal(p, t) for p, t in zip(placed, tensors)))  # still correct
        self.assertEqual(len(asked), 2)  # after the refusal nobody asks again
        self.assertEqual(arena.pageable_bytes, 400 * 4)
        self.assertEqual(arena.packed_bytes, 400)

    def test_the_streamer_warns_when_part_of_the_weights_could_not_be_pinned(self):
        import torch
        from engine.core import streaming

        def refuse(size):
            raise RuntimeError("no lockable memory")

        blocks = make_blocks(2)
        with self.assertWarns(RuntimeWarning):
            streamer = streaming.BlockStreamer(blocks, "cpu", transfer=lambda t: t.clone(),
                                               arena=streaming._Arena(refuse))
        self.assertGreater(streamer.arena.pageable_bytes, 0)
        streamer.attach()()
        torch.testing.assert_close(run(blocks, torch.ones(2, 32)), run(make_blocks(2), torch.ones(2, 32)))


class FakeEvent:
    """Stands in for a CUDA event: remembers when it was made and whether the host waited for it."""
    made = 0
    synced = []

    def __init__(self):
        self.number = FakeEvent.made
        FakeEvent.made += 1

    def record(self, stream=None):
        pass

    def synchronize(self):
        FakeEvent.synced.append(self.number)


@unittest.skipUnless(HAVE, "torch or gguf is not installed")
class RunAheadTests(unittest.TestCase):
    def test_the_host_never_runs_more_than_prefetch_plus_one_blocks_ahead(self):
        import torch
        from engine.core.streaming import BlockStreamer

        for prefetch in (1, 2):
            FakeEvent.made, FakeEvent.synced = 0, []
            blocks = make_blocks(7)
            streamer = BlockStreamer(blocks, "cpu", prefetch=prefetch, pin=False, transfer=lambda t: t.clone(),
                                     make_event=FakeEvent)
            started, seen = [0], []

            def at_block_start(module, args):  # registered after the streamer's own hook, so it runs after it
                n = started[0]
                started[0] += 1
                seen.append((n, sorted(FakeEvent.synced)))

            streamer.attach()
            for block in blocks:
                block.register_forward_pre_hook(at_block_start)
            for _ in range(3):  # three sampling steps: the passes follow each other without a pause
                run(blocks, torch.randn(2, 32))
            for n, synced in seen:
                with self.subTest(prefetch=prefetch, block=n):
                    # every block that ran more than prefetch + 1 blocks ago is finished, none later is waited for
                    self.assertEqual(synced, list(range(max(0, n - prefetch - 1))))
            self.assertLessEqual(len(streamer._pending), prefetch + 2)

    def test_without_events_nothing_is_waited_for(self):
        import torch
        from engine.core.streaming import BlockStreamer

        blocks = make_blocks(4)
        streamer = BlockStreamer(blocks, "cpu", pin=False, transfer=lambda t: t.clone())  # no GPU: no events
        streamer.attach()
        run(blocks, torch.randn(2, 32))
        self.assertEqual(len(streamer._pending), 0)

    def test_detaching_leaves_the_streamer_holding_no_weights(self):
        import gc
        import weakref

        import torch
        from engine.core import streaming

        blocks = make_blocks(3)
        first_block = weakref.ref(blocks[0])
        streamer = streaming.BlockStreamer(blocks, "cpu", transfer=lambda t: t.clone(),
                                           arena=streaming._Arena(lambda n: torch.empty(n, dtype=torch.uint8)))
        remove = streamer.attach()  # the bound method keeps the streamer alive, as in the video pipeline
        run(blocks, torch.ones(2, 32))
        remove()
        self.assertEqual((streamer.blocks, streamer.slots, streamer.arena.chunks, streamer.ready), ([], [], [], {}))
        del blocks
        gc.collect()
        self.assertIsNone(first_block())  # the streamer is still referenced (by `remove`), but holds no block
        self.assertIsNotNone(remove)


@unittest.skipUnless(HAVE, "torch or gguf is not installed")
class HostCacheTests(unittest.TestCase):
    def test_the_pinned_host_cache_is_emptied_through_torch_and_a_failure_is_ignored(self):
        import torch
        from engine.core import streaming

        calls = []
        with unittest.mock.patch.object(torch._C, "_host_emptyCache", lambda: calls.append(1), create=True):
            streaming.release_host_cache()
        self.assertEqual(calls, [1])

        def refuses():
            raise RuntimeError("the cache cannot be emptied")

        with unittest.mock.patch.object(torch._C, "_host_emptyCache", refuses, create=True):
            streaming.release_host_cache()  # costs memory until the process ends, nothing more
        with unittest.mock.patch.object(torch._C, "_host_emptyCache", None, create=True):
            streaming.release_host_cache()  # a torch without that function: nothing to do


if __name__ == "__main__":
    unittest.main()
