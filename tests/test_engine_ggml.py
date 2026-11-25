# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Torch dequantization of GGUF tensors must equal the reference implementation in gguf-py."""
import importlib.util
import unittest
from app.config import destination_dir

DIFFUSION = destination_dir("diffusion_models")
ENCODERS = destination_dir("text_encoders")
HAVE = all(importlib.util.find_spec(name) for name in ("torch", "gguf", "numpy"))


@unittest.skipUnless(HAVE, "torch and gguf are not installed")
class RandomBlockTests(unittest.TestCase):
    def test_the_supported_block_types_are_the_ones_the_catalog_uses(self):
        from engine.formats import ggml

        self.assertEqual({t.name for t in ggml.BLOCK_QUANTS}, {"Q8_0", "Q4_K", "Q5_K", "Q6_K"})

    def test_every_supported_block_type_matches_gguf_py(self):
        import gguf
        import numpy as np
        import torch
        from engine.formats import ggml

        rng = np.random.default_rng(7)
        for qtype in sorted(ggml.BLOCK_QUANTS, key=int):
            block, size = gguf.GGML_QUANT_SIZES[qtype]
            rows, per_row = 3, 4
            raw = rng.integers(0, 256, size=(rows, per_row * size), dtype=np.uint8)
            with np.errstate(all="ignore"):  # random scale bytes give inf/nan in both implementations
                expected = torch.from_numpy(gguf.quants.dequantize(raw, qtype))
            got = ggml.dequantize(torch.from_numpy(raw), qtype, (rows, per_row * block), torch.float32)
            with self.subTest(qtype=qtype.name):
                torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6, equal_nan=True)

    def test_plain_types_and_requested_dtype(self):
        import numpy as np
        import torch
        from engine.formats import ggml

        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        got = ggml.dequantize(torch.from_numpy(values), ggml.QT.F32, (3, 4), torch.bfloat16)
        self.assertEqual(got.dtype, torch.bfloat16)
        torch.testing.assert_close(got.float(), torch.from_numpy(values))
        raw = torch.from_numpy(values).to(torch.bfloat16).view(torch.uint8).reshape(3, 8)
        torch.testing.assert_close(ggml.dequantize(raw, ggml.QT.BF16, (3, 4), torch.float32),
                                   torch.from_numpy(values))

    def test_unsupported_type_is_an_error_not_a_guess(self):
        import torch
        from engine.formats import ggml

        with self.assertRaises(NotImplementedError):
            ggml.dequantize(torch.zeros(1, 66, dtype=torch.uint8), ggml.QT.IQ2_XXS, (1, 256))
        with self.assertRaises(NotImplementedError):  # a legacy type no catalog file uses
            ggml.dequantize(torch.zeros(1, 18, dtype=torch.uint8), ggml.QT.Q4_0, (1, 32))


@unittest.skipUnless(HAVE, "torch and gguf are not installed")
class RealFileTests(unittest.TestCase):
    """The GGUF files Contsy actually ships: every tensor type must be supported, and a real
    tensor of each type must dequantize exactly like gguf-py does."""

    FILES = [*sorted(DIFFUSION.glob("Wan2.2-T2V-A14B-HighNoise-Q*.gguf")),
             *sorted(DIFFUSION.glob("qwen-image-edit-2511-Q4_K_M.gguf")),
             *sorted(ENCODERS.glob("umt5-xxl-encoder-Q5_K_M.gguf"))]

    def test_every_tensor_type_in_the_shipped_files_is_supported(self):
        if not self.FILES:
            self.skipTest("no GGUF model files on this machine")
        from engine.formats import ggml

        for path in self.FILES:
            with self.subTest(file=path.name), ggml.GGUFFile(path) as file:
                self.assertTrue(file.tensors)
                types = {tensor.qtype for tensor in file.tensors.values()}
                self.assertEqual({t.name for t in types - ggml.BLOCK_QUANTS - set(ggml.PLAIN_TYPES)}, set())

    def test_real_tensors_match_gguf_py(self):
        if not self.FILES:
            self.skipTest("no GGUF model files on this machine")
        import math

        import gguf
        import torch
        from engine.formats import ggml

        for path in self.FILES:
            reference = {t.name: t for t in gguf.GGUFReader(str(path)).tensors}
            with ggml.GGUFFile(path) as file:
                smallest = {}
                for tensor in file.tensors.values():
                    size = math.prod(tensor.shape)
                    if tensor.quantized and size >= 4096:
                        best = smallest.get(tensor.qtype)
                        if best is None or size < math.prod(best.shape):
                            smallest[tensor.qtype] = tensor
                for qtype, tensor in smallest.items():
                    expected = gguf.quants.dequantize(reference[tensor.name].data, qtype)
                    got = tensor.to(torch.float32)
                    with self.subTest(file=path.name, qtype=qtype.name, tensor=tensor.name):
                        self.assertEqual(tuple(got.shape), tuple(expected.shape))
                        torch.testing.assert_close(got, torch.from_numpy(expected),
                                                   rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
