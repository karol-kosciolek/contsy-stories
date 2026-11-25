# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""GGUF files that lie about their size: a corrupt, truncated or hostile file must raise GGUFError quickly and
without allocating memory, naming the problem, while every well-formed file still opens."""
import importlib.util
import struct
import tempfile
import time
import unittest
from pathlib import Path

from app.config import destination_dir

HAVE = all(importlib.util.find_spec(name) is not None for name in ("torch", "gguf", "numpy"))


def header(kv_count=0, tensor_count=0, version=3, magic=b"GGUF"):
    return magic + struct.pack("<IQQ", version, tensor_count, kv_count)


def key(name):
    return struct.pack("<Q", len(name)) + name.encode()


def write_valid(path):
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(str(path), "test")
    writer.add_uint32("test.answer", 42)
    writer.add_tensor("first.weight", np.arange(64, dtype=np.float32).reshape(8, 8))
    writer.add_tensor("second.weight", np.ones((4, 32), dtype=np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@unittest.skipUnless(HAVE, "torch or gguf is not installed")
class HostileFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def refuse(self, content, text):
        from engine.formats.ggml import GGUFError, GGUFFile

        path = self.dir / "bad.gguf"
        path.write_bytes(content)
        began = time.time()
        with self.assertRaises(GGUFError) as raised:
            GGUFFile(path)
        self.assertLess(time.time() - began, 2.0)  # the gguf package would loop over the claimed entries
        self.assertIn(text, str(raised.exception))
        self.assertIn("bad.gguf", str(raised.exception))

    def test_a_well_formed_file_opens_and_gives_its_tensors(self):
        from engine.formats.ggml import GGUFFile

        write_valid(self.dir / "ok.gguf")
        with GGUFFile(self.dir / "ok.gguf") as file:
            self.assertEqual(sorted(file.tensors), ["first.weight", "second.weight"])
            self.assertEqual(file.tensors["first.weight"].shape, (8, 8))

    def test_a_truncated_download_names_the_first_tensor_that_does_not_fit(self):
        write_valid(self.dir / "ok.gguf")
        whole = (self.dir / "ok.gguf").read_bytes()
        self.refuse(whole[:-40], "second.weight")  # the last tensor is cut short
        self.refuse(whole[:-40], "incomplete")

    def test_an_array_that_claims_a_trillion_entries_is_refused_at_once(self):
        content = header(kv_count=1) + key("tokens") + struct.pack("<IIQ", 9, 0, 10**12)
        self.refuse(content, "an array of 1000000000000 entries")

    def test_an_array_of_strings_cannot_claim_more_entries_than_the_file_has_room_for(self):
        content = header(kv_count=1) + key("tokens") + struct.pack("<IIQ", 9, 8, 10**9) + b"\0" * 64
        self.refuse(content, "an array of 1000000000 entries")

    def test_a_string_longer_than_the_file_is_refused(self):
        content = header(kv_count=1) + key("name") + struct.pack("<IQ", 8, 10**12) + b"abc"
        self.refuse(content, "a metadata string")

    def test_absurd_entry_counts_are_refused(self):
        self.refuse(header(kv_count=2**40) + b"\0" * 64, "cannot fit")
        self.refuse(header(tensor_count=2**40) + b"\0" * 64, "cannot fit")

    def test_not_gguf_and_unsupported_versions(self):
        self.refuse(b"NOPE" + b"\0" * 60, "is not a GGUF file")
        self.refuse(header(version=1) + b"\0" * 32, "version 1")
        self.refuse(b"GGUF", "too few")

    def test_nested_arrays_cannot_recurse_without_limit(self):
        nested = struct.pack("<IQ", 9, 1) * 40 + b"\0" * 64  # an array of one array of one array ...
        self.refuse(header(kv_count=1) + key("deep") + struct.pack("<I", 9) + nested, "unknown metadata value type")

    # --- every guard of the header walk, by a file that breaks exactly that rule ---

    def crafted(self, kv=(), tensors=(), version=3, data=b""):
        """A GGUF file made by hand: `kv` are encoded entries, `tensors` are (name, dims, type, offset)."""
        body = b"".join(kv)
        for name, dims, kind, offset in tensors:
            body += key(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
            body += struct.pack("<IQ", kind, offset)
        content = header(kv_count=len(kv), tensor_count=len(tensors), version=version) + body
        content += b"\0" * (-len(content) % 32) + data  # the data section starts on the alignment (32 unless said otherwise)
        path = self.dir / "crafted.gguf"
        path.write_bytes(content)
        return path

    def check(self, path):
        from engine.formats.ggml import check_layout

        check_layout(path)

    def refused(self, path, text):
        from engine.formats.ggml import GGUFError

        with self.assertRaises(GGUFError) as raised:
            self.check(path)
        self.assertIn(text, str(raised.exception))

    def test_an_alignment_that_is_not_a_power_of_two_is_refused(self):
        self.refused(self.crafted(kv=[key("general.alignment") + struct.pack("<II", 4, 24)]), "not a power of two")

    def test_a_tensor_with_too_many_dimensions_is_refused(self):
        self.refused(self.crafted(tensors=[("t", (1,) * 9, 0, 0)]), "claims 9 dimensions")

    def test_an_unknown_tensor_type_is_refused_by_name(self):
        self.refused(self.crafted(tensors=[("weird.weight", (4,), 9999, 0)]), "tensor 'weird.weight' has the unknown type 9999")

    def test_a_block_quantised_tensor_must_hold_whole_blocks(self):
        import gguf

        q8 = int(gguf.GGMLQuantizationType.Q8_0)  # blocks of 32 values
        self.refused(self.crafted(tensors=[("q.weight", (33,), q8, 0)], data=b"\0" * 64), "not a multiple of the block size 32")
        self.check(self.crafted(tensors=[("q.weight", (64,), q8, 0)], data=b"\0" * 68))  # two blocks of 34 bytes fit

    def test_a_tensor_that_starts_beyond_the_end_of_the_file_is_refused(self):
        self.refused(self.crafted(tensors=[("late.weight", (4,), 0, 10**9)], data=b"\0" * 16), "late.weight")

    def test_a_custom_alignment_is_honoured_and_version_2_is_accepted(self):
        import gguf

        from engine.formats.ggml import GGUFFile

        path = self.dir / "aligned.gguf"
        import numpy as np

        writer = gguf.GGUFWriter(str(path), "test")
        writer.add_custom_alignment(64)
        writer.add_tensor("a.weight", np.arange(32, dtype=np.float32).reshape(4, 8))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        with GGUFFile(path) as file:
            self.assertEqual(file.tensors["a.weight"].shape, (4, 8))
        self.check(self.crafted(version=2))  # no entries at all

    def test_an_empty_tensor_needs_no_data_even_when_the_file_ends_at_the_header(self):
        content = header(tensor_count=1) + key("empty.weight") + struct.pack("<I", 1) + struct.pack("<Q", 0)
        content += struct.pack("<IQ", 0, 0)  # F32, offset 0: the file ends right here, without padding or data
        path = self.dir / "empty.gguf"
        path.write_bytes(content)
        self.check(path)

    def test_the_real_gguf_files_on_this_machine_pass(self):
        from engine.formats.ggml import check_layout

        files = sorted(path for dest in ("diffusion_models", "text_encoders")
                       for path in destination_dir(dest).glob("*.gguf"))
        if not files:
            self.skipTest("no GGUF file is installed here")
        for file in files:
            with self.subTest(file=file.name):
                check_layout(file)  # reads the header only, however big the file is


if __name__ == "__main__":
    unittest.main()
