# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Scaled-fp8 safetensors are expanded to regular tensors."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAVE = all(importlib.util.find_spec(name) for name in ("torch", "safetensors"))


@unittest.skipUnless(HAVE, "torch and safetensors are not installed")
class ScaledFp8Tests(unittest.TestCase):
    def test_weight_is_multiplied_by_its_scale_and_bookkeeping_is_hidden(self):
        import torch
        from safetensors.torch import save_file
        from engine.formats import fp8

        torch.manual_seed(0)
        weight = torch.randn(64, 32)
        scale = (weight.abs().max() / 448).reshape(())
        stored = {"layer.weight": (weight / scale).to(torch.float8_e4m3fn),
                  "layer.scale_weight": scale.float(),
                  "norm.weight": torch.ones(8, dtype=torch.bfloat16),
                  "scaled_fp8": torch.zeros(2, dtype=torch.float8_e4m3fn)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoder.safetensors"
            save_file(stored, str(path))
            out = dict(fp8.iter_tensors(path, torch.bfloat16))
        self.assertEqual(sorted(out), ["layer.weight", "norm.weight"])
        self.assertEqual(out["layer.weight"].dtype, torch.bfloat16)
        self.assertEqual(out["norm.weight"].dtype, torch.bfloat16)  # untouched
        torch.testing.assert_close(out["layer.weight"].float(), weight, rtol=0.07,
                                   atol=float(scale) * 0.01)

    def test_fp8_without_a_scale_is_only_cast(self):
        import torch
        from safetensors.torch import save_file
        from engine.formats import fp8

        raw = torch.tensor([[1.0, -2.0], [0.5, 4.0]]).to(torch.float8_e4m3fn)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.safetensors"
            save_file({"a.weight": raw}, str(path))
            out = dict(fp8.iter_tensors(path, torch.float32))
        torch.testing.assert_close(out["a.weight"], raw.float())


if __name__ == "__main__":
    unittest.main()
