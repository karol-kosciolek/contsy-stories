# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The attention switch of the video experts: SDPA unless SageAttention is asked for and can run, a fallback to SDPA
for every input the kernels do not take, the setting per thread, and Wan's own attention processor computing exactly
what diffusers' processor computes. Everything here runs on the CPU (SageAttention itself needs a CUDA GPU)."""
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import seeded

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
HAVE = importlib.util.find_spec("torch") is not None
HAVE_WAN = HAVE and all(importlib.util.find_spec(name) for name in ("diffusers", "safetensors")) \
    and (FIXTURES / "engine_wan_transformer.json").is_file()


@unittest.skipUnless(HAVE, "torch is not installed")
class BackendTests(unittest.TestCase):
    def test_the_setting_picks_sage_only_where_triton_and_the_gpu_allow_it(self):
        from engine.core import attention

        self.assertEqual(attention.resolve("sdpa"), "sdpa")
        self.assertEqual(attention.resolve(None), "sdpa")
        with patch.object(attention, "sage_available", return_value=False):
            self.assertEqual(attention.resolve("auto"), "sdpa")
        with patch.object(attention, "sage_available", return_value=True):
            self.assertEqual(attention.resolve("auto"), "sage")
            self.assertEqual(attention.resolve("sage"), "sage")
        with self.assertRaisesRegex(ValueError, "attention must be one of"):
            attention.resolve("flash")

    def test_sage_needs_triton_and_a_gpu_of_compute_capability_8_or_newer(self):
        from engine.core import attention

        with patch.object(attention.kernels, "available", return_value=False):
            self.assertFalse(attention.sage_available("cuda"))
        with patch.object(attention.kernels, "available", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(7, 5)):
            self.assertFalse(attention.sage_available("cuda"))
        with patch.object(attention.kernels, "available", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(8, 9)):
            self.assertTrue(attention.sage_available("cuda"))

    def test_without_triton_or_cuda_the_kernels_are_unavailable(self):
        from engine import kernels

        with patch("torch.cuda.is_available", return_value=False):
            self.assertFalse(kernels.available())
        with patch("torch.cuda.is_available", return_value=True):
            self.assertFalse(kernels.available("cpu"))

    def test_use_sets_the_backend_for_this_thread_only_and_restores_it(self):
        from engine.core import attention

        seen = []
        with attention.use("sage"):
            self.assertEqual(attention.current(), "sage")
            other = threading.Thread(target=lambda: seen.append(attention.current()))
            other.start()
            other.join()
            with attention.use("sdpa"):
                self.assertEqual(attention.current(), "sdpa")
            self.assertEqual(attention.current(), "sage")
        self.assertEqual(attention.current(), "sdpa")
        self.assertEqual(seen, ["sdpa"])
        with self.assertRaisesRegex(ValueError, "unknown attention backend"):
            with attention.use("auto"):  # a setting, not a backend: resolve it first
                pass

    def test_inputs_the_kernels_do_not_take_fall_back_to_sdpa_with_the_same_result(self):
        import torch
        import torch.nn.functional as F
        from engine.core import attention

        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 10, 2, 64) for _ in range(3))
        expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        self.assertFalse(attention.fits_sage(q, k, v))  # the CPU, float32
        with attention.use("sage"), patch.object(attention, "sage", side_effect=AssertionError("no kernels here")):
            torch.testing.assert_close(attention.attend(q, k, v), expected)
        mask = torch.zeros(1, 1, 10, 10, dtype=torch.bool)
        mask[..., :5] = True
        masked = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                attn_mask=mask).transpose(1, 2)
        torch.testing.assert_close(attention.attend(q, k, v, mask), masked)

    def test_only_cuda_half_precision_inputs_with_64_or_128_channels_per_head_and_no_mask_fit(self):
        import torch
        from engine.core import attention

        def fits(shape=(1, 8, 2, 128), dtype=torch.bfloat16, mask=None, kv_heads=2):
            q = torch.empty(shape, dtype=dtype)
            k = torch.empty(shape[0], 5, kv_heads, shape[3], dtype=dtype)
            with patch.object(torch.Tensor, "is_cuda", True):
                return attention.fits_sage(q, k, k, mask)

        self.assertTrue(fits())
        self.assertTrue(fits(shape=(1, 8, 4, 64), kv_heads=2))  # grouped keys and values
        self.assertTrue(fits(dtype=torch.float16))
        self.assertFalse(fits(dtype=torch.float32))
        self.assertFalse(fits(shape=(1, 8, 2, 96)))
        self.assertFalse(fits(mask=torch.ones(1, 1, 8, 5, dtype=torch.bool)))
        self.assertFalse(fits(shape=(1, 8, 3, 128), kv_heads=2))


@unittest.skipUnless(HAVE_WAN, "torch, diffusers or the Wan fixture is missing")
class WanProcessorTests(unittest.TestCase):
    def test_the_engines_processor_computes_what_diffusers_computes(self):
        import torch
        from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor as Original
        from safetensors.torch import save_file
        from engine.models.wan22 import attention as wan_attention
        from engine.models.wan22 import transformer
        from tests.test_engine_wan import diffusers_config

        data = json.loads((FIXTURES / "engine_wan_transformer.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wan.safetensors"
            save_file({k: v.contiguous() for k, v in seeded.seeded_state_dict(data["shapes"]).items()}, str(path))
            model = transformer.load_transformer(path, "cpu", torch.float32, config=diffusers_config(data["config"]))
        modules = [m for m in model.modules() if isinstance(m, WanAttention)]
        self.assertTrue(modules)
        self.assertTrue(all(isinstance(m.processor, wan_attention.WanAttnProcessor) for m in modules))
        latent = seeded.seeded_inputs(data["latent_shape"], data["latent_seed"])
        context = seeded.seeded_inputs(data["context_shape"], data["context_seed"])

        def velocity():
            return transformer.predict_velocity(model, latent[:, :16], torch.tensor(0.4), context, latent[:, 16:])

        ours = velocity()
        for module in modules:
            module.set_processor(Original())
        torch.testing.assert_close(ours, velocity(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
