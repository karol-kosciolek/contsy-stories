# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Reading model files ahead (engine/core/readahead.py) and the Wan experts' read-ahead (engine/pipelines/video.py).
The machine's own free memory never decides a test: readahead._room_for is patched wherever files are read."""
import importlib.util
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None


class AheadTests(unittest.TestCase):
    def setUp(self):
        from engine.core import readahead

        self.real_room_for = readahead._room_for  # the gate itself is tested on its own
        room = patch.object(readahead, "_room_for", return_value=True)
        room.start()
        self.addCleanup(room.stop)

    def test_a_stage_reads_a_listed_file_itself_instead_of_waiting_for_its_turn(self):
        """settle() waits for a file an ahead() thread is reading this moment, but not for one that is only listed
        (waiting for memory room or behind other files): it reads that one itself and the ahead() thread leaves it
        out, so a composition never waits out another stage's patience."""
        import time

        from engine.core import readahead

        with tempfile.TemporaryDirectory() as work:
            first, second = Path(work) / "first", Path(work) / "second"
            first.write_bytes(b"1" * 10)
            second.write_bytes(b"2" * 10)
            read = []
            with patch.object(readahead, "_room_for", return_value=False), \
                    patch.object(readahead, "read_file", side_effect=lambda path: read.append(Path(path).name) or 1), \
                    patch.dict(readahead._FINISHED, clear=True):
                thread = readahead.ahead([first, second], patience=30)  # waits for room that never comes
                started = time.monotonic()
                readahead.settle(second)
                self.assertLess(time.monotonic() - started, 5)
                self.assertEqual(read, ["second"])  # read by the stage itself
            thread.join(40)
            self.assertEqual(read, ["second"])  # the ahead() thread did not read it again
            self.assertEqual((readahead._READING, readahead._ACTIVE, readahead._CLAIMED), (set(), set(), set()))
    def test_files_are_read_in_order_once_the_event_is_set_and_never_twice_at_once(self):
        from engine.core import readahead

        with tempfile.TemporaryDirectory() as work:
            files = []
            for name in ("high", "lora", "low"):
                path = Path(work) / name
                path.write_bytes(name.encode() * 10)
                files.append(path)
            (Path(work) / "empty").write_bytes(b"")
            read, go, release = [], threading.Event(), threading.Event()

            def fake_read(path):
                read.append(Path(path).name)
                if Path(path).name == "high":
                    release.wait(5)  # still reading "high" while the second call is made
                return 1

            with patch.object(readahead, "read_file", side_effect=fake_read), \
                    patch.dict(readahead._FINISHED, clear=True):
                thread = readahead.ahead([*files, files[0], Path(work) / "missing", Path(work) / "empty"], wait=go)
                self.assertEqual(read, [])  # waits for the event
                go.set()
                for _ in range(100):
                    if read:
                        break
                    threading.Event().wait(0.01)
                self.assertIsNone(readahead.ahead([files[0]]))  # being read by the first thread: left out
                release.set()
                thread.join(5)
                self.assertEqual(read, ["high", "lora", "low"])  # in order, once each; missing and empty left out
                self.assertEqual(readahead._READING, set())
                self.assertIsNone(readahead.ahead(files))  # read a moment ago: not read again
                with patch.object(readahead, "read_file", side_effect=OSError("gone")):
                    readahead.ahead(files, skip_recent=False).join(5)  # an unreadable file is only not read ahead
                self.assertEqual(readahead._READING, set())

    def test_a_file_is_read_ahead_only_with_room_beside_the_reserve(self):
        """The other studio's and the job's memory is never pushed out: a file is read ahead only when the available
        memory (free pages and cached files, no working set) holds it and AHEAD_RESERVE; a read with patience waits for
        the room a stage gives back when it ends."""
        from engine.core import readahead

        gib = 1 << 30
        with patch.object(readahead, "_memory_status", return_value=(64 * gib, 30 * gib)), \
                patch.object(readahead.sys, "platform", "win32"):
            self.assertEqual(readahead.AHEAD_RESERVE, 8 * gib)
            self.assertTrue(self.real_room_for(22 * gib))
            self.assertFalse(self.real_room_for(23 * gib))
        with tempfile.TemporaryDirectory() as work:
            big = Path(work) / "big"
            big.write_bytes(b"x")
            rooms = iter([False, False, True, True])
            with patch.object(readahead, "_room_for", side_effect=lambda size: next(rooms)), \
                    patch.object(readahead, "read_file", return_value=1) as read_file, \
                    patch.dict(readahead._FINISHED, clear=True):
                readahead.ahead([big], patience=5).join(10)
            read_file.assert_called_once()  # read once the room was there
            with tempfile.TemporaryDirectory() as work:
                big = Path(work) / "big"
                big.write_bytes(b"x")
                with patch.object(readahead, "_room_for", return_value=False), \
                        patch.object(readahead, "read_file") as read_file:
                    readahead.ahead([big]).join(5)
                read_file.assert_not_called()  # left to the stage's page faults
                self.assertEqual(readahead._READING, set())

    @unittest.skipUnless(sys.platform == "win32", "Windows memory priorities")
    def test_background_reads_keep_the_normal_memory_priority(self):
        """Background mode lowers the I/O and the memory priority (to 1, very low); the pages read ahead must not be
        the first ones Windows hands out again."""
        import ctypes
        from ctypes import wintypes

        from engine.core import readahead

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentThread.restype = wintypes.HANDLE
        kernel32.GetThreadInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]

        class Priority(ctypes.Structure):
            _fields_ = [("MemoryPriority", wintypes.ULONG)]

        def priority():
            value = Priority()
            kernel32.GetThreadInformation(kernel32.GetCurrentThread(), 0, ctypes.byref(value), ctypes.sizeof(value))
            return value.MemoryPriority

        with readahead.background_io():
            self.assertEqual(priority(), 5)
        self.assertEqual(priority(), 5)
        self.assertGreater(readahead.physical_memory(), 1 << 30)


@unittest.skipUnless(HAVE_TORCH, "torch or safetensors missing")
class NetworkOrderTests(unittest.TestCase):
    def test_a_networks_file_is_read_block_by_block_in_the_order_the_network_uses_it(self):
        """The experts store their tensors by name (blocks 0, 1, 10, ... 2): the read goes header and the tensors
        outside the blocks first, then block 0, 1, 2, ... 10, and covers every byte once."""
        import torch
        from safetensors.torch import save_file

        from engine.core import readahead

        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "expert.safetensors"
            tensors = {f"blocks.{index}.ffn.0.weight": torch.full((4, 4), float(index)) for index in (0, 1, 2, 10)}
            tensors.update({"blocks.2.ffn.0.weight_scale": torch.ones(4, 1), "head.weight": torch.full((2, 2), 9.0)})
            save_file(tensors, str(path))
            ranges = readahead.network_order(path)
            self.assertEqual(sum(length for _, length in ranges), path.stat().st_size)
            data = path.read_bytes()
            size = int.from_bytes(data[:8], "little")
            header = __import__("json").loads(data[8:8 + size])

            def owner(offset):
                return next(name for name, entry in header.items() if name != "__metadata__"
                            and 8 + size + entry["data_offsets"][0] <= offset < 8 + size + entry["data_offsets"][1])

            names = []
            for offset, length in ranges[1:]:
                position = offset
                while position < offset + length:
                    name = owner(position)
                    names.append(name)
                    position = 8 + size + header[name]["data_offsets"][1]
            blocks = [name.split(".")[1] for name in names if name.startswith("blocks.")]
            self.assertEqual(names[0], "head.weight")
            self.assertEqual(list(dict.fromkeys(blocks)), ["0", "1", "2", "10"])
            with patch.object(readahead, "READ_BYTES", 7):  # read in small pieces: every byte, once
                self.assertEqual(readahead.read_file(path), path.stat().st_size)
            other = Path(work) / "notes.bin"
            other.write_bytes(b"x" * 20)
            self.assertIsNone(readahead.network_order(other))
            self.assertEqual(readahead.read_file(other), 20)

    def test_the_talking_networks_files_go_embeddings_adapter_then_blocks_by_number(self):
        from engine.pipelines import talk

        base = [Path(f"block_{index}.safetensors") for index in (0, 1, 10, 2)] + [Path("non_block.safetensors")]
        self.assertEqual([path.name for path in talk._use_order(base, Path("adapter.safetensors"))],
                         ["non_block.safetensors", "adapter.safetensors", "block_0.safetensors", "block_1.safetensors",
                          "block_2.safetensors", "block_10.safetensors"])


@unittest.skipUnless(HAVE_TORCH, "torch or safetensors missing")
class ExpertsAheadTests(unittest.TestCase):
    def test_each_expert_is_read_after_the_text_encoder_with_its_loras_when_memory_allows(self):
        from engine.core import readahead
        from engine.pipelines import video

        with tempfile.TemporaryDirectory() as work:
            paths = {}
            for name in ("high", "low", "high_lora", "low_lora"):
                paths[name] = Path(work) / name
                paths[name].write_bytes(b"x" * 1000)
            models = video.WanModels(load_text=None, load_transformer=None, load_vae=None,
                                     expert_files=(paths["high"], paths["low"]))
            loras = {"high": [(paths["high_lora"], 1.0)], "low": [(paths["low_lora"], 1.0)]}
            event = threading.Event()
            with patch.object(readahead, "ahead") as ahead, \
                    patch.object(readahead, "physical_memory", return_value=video.EXPERTS_AHEAD_ROOM + 4000):
                video._read_experts_ahead(models, loras, event)
                ahead.assert_called_once()
                self.assertEqual(ahead.call_args.args[0],
                                 [paths["high"], paths["high_lora"], paths["low"], paths["low_lora"]])
                self.assertIs(ahead.call_args.kwargs["wait"], event)
            with patch.object(readahead, "ahead") as ahead, \
                    patch.object(readahead, "physical_memory", return_value=video.EXPERTS_AHEAD_ROOM + 3999):
                self.assertIsNone(video._read_experts_ahead(models, loras, event))  # 32 GB machines: not read ahead
                ahead.assert_not_called()
            tiny = video.WanModels(load_text=None, load_transformer=None, load_vae=None)  # test models: no files
            with patch.object(readahead, "ahead") as ahead:
                self.assertIsNone(video._read_experts_ahead(tiny, {}, event))
                ahead.assert_not_called()


if __name__ == "__main__":
    unittest.main()
