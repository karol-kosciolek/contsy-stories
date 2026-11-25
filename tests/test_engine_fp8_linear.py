# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""LightX2V's scaled FP8 experts: the file layout (FP8 weight and a scale per output channel) is read as FP8 layers,
the layers compute `x @ (weight * scale).T + bias` (in FP8 on a capable GPU, by expanding the weight elsewhere),
keep their dtype when moved or streamed, and a tiny Wan transformer stored this way gives the result of the same
network with the expanded weights. Everything here runs on the CPU."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import seeded

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
HAVE = all(importlib.util.find_spec(name) is not None for name in ("torch", "safetensors"))
HAVE_WAN = HAVE and importlib.util.find_spec("diffusers") is not None \
    and (FIXTURES / "engine_wan_transformer.json").is_file()


def channel_quantize(weight):
    """A float weight [out, in] as LightX2V stores it: float8_e4m3fn and a bf16 scale per output channel [out, 1]."""
    import torch

    scale = (weight.abs().amax(dim=1, keepdim=True) / 448.0).to(torch.bfloat16)
    return (weight / scale.float()).clamp(-448, 448).to(torch.float8_e4m3fn), scale


@unittest.skipUnless(HAVE, "torch and safetensors are not installed")
class ChannelScaledFileTests(unittest.TestCase):
    def write(self, tensors) -> Path:
        from safetensors.torch import save_file

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "expert.safetensors"
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
        return path

    def test_a_weight_and_its_scale_come_out_as_one_fp8_weight_and_the_rest_unchanged(self):
        import torch
        from engine.formats import fp8

        torch.manual_seed(0)
        weight, scale = channel_quantize(torch.randn(32, 16))
        path = self.write({"blocks.0.q.weight": weight, "blocks.0.q.weight_scale": scale,
                           "blocks.0.q.bias": torch.ones(32), "norm.weight": torch.ones(16, dtype=torch.bfloat16)})
        self.assertTrue(fp8.channel_scaled(path))
        out = dict(fp8.iter_channel_scaled(path))
        self.assertEqual(sorted(out), ["blocks.0.q.bias", "blocks.0.q.weight", "norm.weight"])
        packed = out["blocks.0.q.weight"]
        self.assertIsInstance(packed, fp8.ScaledFP8)
        self.assertEqual((packed.weight.dtype, tuple(packed.shape), packed.scale.dtype, tuple(packed.scale.shape)),
                         (torch.float8_e4m3fn, (32, 16), torch.float32, (32,)))
        torch.testing.assert_close(packed.dequantize(torch.float32), weight.float() * scale.float())
        self.assertEqual(out["norm.weight"].dtype, torch.bfloat16)

    def test_files_without_per_channel_scales_are_not_taken_for_lightx2v_files(self):
        import torch
        from engine.formats import fp8

        plain = self.write({"a.weight": torch.ones(4, 4)})
        per_tensor = self.write({"a.weight": torch.ones(4, 4).to(torch.float8_e4m3fn), "a.scale_weight": torch.ones(())})
        self.assertFalse(fp8.channel_scaled(plain))
        self.assertFalse(fp8.channel_scaled(per_tensor))

    def test_fp8_without_its_scale_or_with_a_scale_of_the_wrong_size_is_refused(self):
        import torch
        from engine.formats import fp8

        weight, scale = channel_quantize(torch.randn(8, 16))
        lonely = self.write({"a.weight": weight, "b.weight": weight.clone(), "b.weight_scale": scale})
        with self.assertRaisesRegex(fp8.UnsupportedQuantization, "a.weight is FP8 without"):
            list(fp8.iter_channel_scaled(lonely))
        wrong = self.write({"a.weight": weight, "a.weight_scale": scale[:4]})
        with self.assertRaisesRegex(fp8.UnsupportedQuantization, "not per-channel"):
            list(fp8.iter_channel_scaled(wrong))


@unittest.skipUnless(HAVE, "torch is not installed")
class KernelFallbackTests(unittest.TestCase):
    """The PyTorch versions of the two passes around the FP8 product (what runs without Triton)."""

    def test_rows_are_scaled_to_the_fp8_range_and_round_trip_within_its_precision(self):
        import torch
        from engine.kernels import fp8

        torch.manual_seed(1)
        x = (torch.randn(5, 3, 64) * torch.tensor([1.0, 30.0, 0.01]).reshape(1, 3, 1)).to(torch.bfloat16)
        x[0, 0] = 0  # an all-zero row quantizes to zeros
        quantized, scale = fp8.quantize_rows(x, use_triton=False)
        rows = x.reshape(-1, 64).float()
        self.assertEqual((quantized.dtype, tuple(quantized.shape), tuple(scale.shape)),
                         (torch.float8_e4m3fn, (15, 64), (15,)))
        torch.testing.assert_close(scale[1:], rows[1:].abs().amax(dim=1) / 448.0)
        self.assertEqual(float(quantized[1:].float().abs().amax(dim=1).min()), 448.0)
        self.assertTrue(torch.equal(quantized[0].float(), torch.zeros(64)))
        back = quantized.float() * scale[:, None]
        self.assertLess(float(((back - rows).abs() / rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)).max()),
                        1 / 16)  # e4m3 keeps 3 mantissa bits: at most half a step of 1/8 relative to the row's max

    def test_the_output_gets_the_row_and_column_scales_and_the_bias(self):
        import torch
        from engine.kernels import fp8

        out = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        fp8.scale_output_(out, torch.tensor([2.0, 0.5]), torch.tensor([1.0, 10.0, 0.25]), torch.tensor([1.0, 0.0, -1.0]),
                          use_triton=False)
        torch.testing.assert_close(out, torch.tensor([[3.0, 40.0, 0.5], [3.0, 25.0, -0.25]]))


@unittest.skipUnless(HAVE, "torch is not installed")
class FP8LinearTests(unittest.TestCase):
    def layer(self, bias=True):
        import torch
        from engine.core.fp8_linear import FP8Linear

        torch.manual_seed(2)
        weight, scale = channel_quantize(torch.randn(32, 64))
        return FP8Linear(weight, scale.reshape(-1).float(), torch.randn(32) if bias else None)

    def test_without_fp8_kernels_the_weight_is_expanded_and_an_ordinary_linear_runs(self):
        import torch
        import torch.nn.functional as F

        layer = self.layer()
        x = torch.randn(2, 7, 64)
        expected = F.linear(x, layer.weight.float() * layer.weight_scale[:, None], layer.bias)
        torch.testing.assert_close(layer(x), expected)
        self.assertFalse(layer.uses_fp8(x))  # the CPU has no FP8 tensor cores
        self.assertEqual(layer(x.to(torch.bfloat16)).dtype, torch.bfloat16)

    def test_moving_or_casting_the_layer_keeps_the_weight_in_fp8_and_the_scale_in_float32(self):
        import torch

        layer = self.layer()
        layer.to(torch.bfloat16)
        layer.to("cpu")
        self.assertEqual((layer.weight.dtype, layer.weight_scale.dtype, layer.bias.dtype),
                         (torch.float8_e4m3fn, torch.float32, torch.bfloat16))
        self.assertEqual(set(dict(layer.named_buffers())), {"weight_scale"})
        self.assertIsInstance(layer, torch.nn.Linear)

    def test_a_dtype_only_cast_leaves_the_weight_on_its_device(self):
        """`.to(torch.bfloat16)` names no device: a layer on the GPU must stay there (the meta device stands in for
        it here, where a CPU-ward move fails loudly instead of silently running every call on the CPU)."""
        import torch
        from engine.core.fp8_linear import FP8Linear

        layer = FP8Linear(torch.empty(32, 64, dtype=torch.float8_e4m3fn, device="meta"),
                          torch.empty(32, device="meta"), torch.empty(32, device="meta"))
        layer.to(torch.bfloat16)
        self.assertEqual((layer.weight.device.type, layer.weight_scale.device.type, layer.bias.device.type),
                         ("meta", "meta", "meta"))
        self.assertEqual((layer.weight.dtype, layer.weight_scale.dtype, layer.bias.dtype),
                         (torch.float8_e4m3fn, torch.float32, torch.bfloat16))

    def test_float16_inputs_never_take_the_fp8_product(self):
        """With unit scales the product's unscaled sums overflow float16 (inf, then NaN video); bfloat16 has room."""
        import torch
        from engine.core import fp8_linear

        layer = self.layer()
        with patch.object(fp8_linear, "fp8_matmul_available", return_value=True):
            self.assertTrue(layer.uses_fp8(torch.randn(2, 64).to(torch.bfloat16)))
            for dtype in (torch.float16, torch.float32):
                with self.subTest(dtype=dtype):
                    self.assertFalse(layer.uses_fp8(torch.randn(2, 64).to(dtype)))

    def test_a_weight_that_is_not_fp8_or_a_scale_of_another_size_is_refused(self):
        import torch
        from engine.core.fp8_linear import FP8Linear

        with self.assertRaisesRegex(ValueError, "float8_e4m3fn"):
            FP8Linear(torch.ones(4, 4), torch.ones(4))
        with self.assertRaisesRegex(ValueError, "4 output channels but 3 scales"):
            FP8Linear(torch.ones(4, 4).to(torch.float8_e4m3fn), torch.ones(3))

    def test_the_fp8_path_quantizes_per_row_and_scales_the_product(self):
        """What a GPU with FP8 tensor cores computes, with torch._scaled_mm replaced by a float product."""
        import torch
        import torch.nn.functional as F
        from engine.core import fp8_linear

        layer = self.layer()
        x = torch.randn(5, 64)

        def scaled_mm(a, b, scale_a, scale_b, out_dtype):
            self.assertEqual((a.dtype, b.dtype, float(scale_a), float(scale_b)),
                             (torch.float8_e4m3fn, torch.float8_e4m3fn, 1.0, 1.0))
            self.assertEqual(b.stride(), (1, 64))  # the weight goes in column-major, as cuBLAS wants it
            return (a.float() @ b.float()).to(out_dtype)

        with patch.object(fp8_linear.FP8Linear, "uses_fp8", return_value=True), \
                patch("torch._scaled_mm", side_effect=scaled_mm, create=True):
            got = layer(x)
        expected = F.linear(x, layer.dequantized(torch.float32), layer.bias)
        self.assertLess(float((got - expected).norm() / expected.norm()), 0.05)  # the activations' FP8 rounding

    def test_projections_of_one_input_share_its_quantization_unless_a_layer_is_hooked(self):
        import torch
        from engine.core import fp8_linear

        layers = [self.layer(), self.layer(bias=False)]
        x = torch.randn(3, 64)
        with patch.object(fp8_linear.FP8Linear, "uses_fp8", return_value=True), \
                patch.object(fp8_linear.kernels, "quantize_rows", wraps=fp8_linear.kernels.quantize_rows) as quantize, \
                patch("torch._scaled_mm", side_effect=lambda a, b, scale_a, scale_b, out_dtype: (a.float() @ b.float())
                      .to(out_dtype), create=True):
            outputs = fp8_linear.project(x, layers)
            self.assertEqual(quantize.call_count, 1)
            torch.testing.assert_close(outputs[0], layers[0](x))
            handle = layers[1].register_forward_hook(lambda module, args, output: output + 1)
            quantize.reset_mock()
            hooked = fp8_linear.project(x, layers)
            handle.remove()
            self.assertEqual(quantize.call_count, 2)  # one call per layer: the hook (a LoRA) must see its input
            torch.testing.assert_close(hooked[1], layers[1](x) + 1)

    def test_fp8_layers_stream_like_any_other_weight_and_come_back_unchanged(self):
        import torch
        from torch import nn
        from engine.core import streaming

        blocks = nn.ModuleList([nn.Sequential(self.layer(), nn.Linear(32, 64)) for _ in range(3)])
        before = [block[0].weight.clone() for block in blocks]
        x = torch.randn(2, 64)
        expected = x
        for block in blocks:
            expected = block(expected)
        moved = []

        def transfer(tensor):
            moved.append(tensor.dtype)
            return tensor.clone()

        remove = streaming.BlockStreamer(blocks, "cpu", prefetch=1, pin=False, transfer=transfer).attach()
        got = x
        for block in blocks:
            got = block(got)
        remove()
        torch.testing.assert_close(got, expected)
        self.assertIn(torch.float8_e4m3fn, moved)
        for block, weight in zip(blocks, before):
            self.assertTrue(torch.equal(block[0].weight.view(torch.uint8), weight.view(torch.uint8)))
            self.assertEqual(block[0].weight.dtype, torch.float8_e4m3fn)


def _diffusers_config(tiny: dict) -> dict:
    from tests.test_engine_wan import diffusers_config

    return diffusers_config(tiny)


@unittest.skipUnless(HAVE_WAN, "torch, diffusers or the Wan fixture is missing")
class ScaledWanLoadingTests(unittest.TestCase):
    """A tiny Wan transformer written as LightX2V writes its experts (every block linear in FP8 with a scale per output
    channel, the rest in float32) loads as FP8 layers and computes what the same network with the expanded weights
    computes."""

    @classmethod
    def setUpClass(cls):
        import torch
        from safetensors.torch import save_file

        cls.data = json.loads((FIXTURES / "engine_wan_transformer.json").read_text(encoding="utf-8"))
        cls.config = _diffusers_config(cls.data["config"])
        state = seeded.seeded_state_dict(cls.data["shapes"])
        stored, dense, cls.scaled = {}, {}, []
        for name, tensor in state.items():
            linear = name.startswith("blocks.") and name.endswith(".weight") and tensor.dim() == 2
            if linear:
                weight, scale = channel_quantize(tensor)
                stored[name], stored[name + "_scale"] = weight, scale
                dense[name] = weight.float() * scale.float()
                cls.scaled.append(name)
            else:
                stored[name] = dense[name] = tensor
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.fp8_path = Path(cls.tmp.name) / "expert_fp8.safetensors"
        cls.dense_path = Path(cls.tmp.name) / "expert_dense.safetensors"
        save_file({k: v.contiguous() for k, v in stored.items()}, str(cls.fp8_path))
        save_file({k: v.contiguous() for k, v in dense.items()}, str(cls.dense_path))

    def forward(self, model):
        import torch
        from engine.models.wan22 import transformer

        latent = seeded.seeded_inputs(self.data["latent_shape"], self.data["latent_seed"])
        context = seeded.seeded_inputs(self.data["context_shape"], self.data["context_seed"])
        return transformer.predict_velocity(model, latent[:, :16], torch.tensor(0.7), context, latent[:, 16:])

    def test_every_block_linear_is_an_fp8_layer_with_the_result_of_the_expanded_weights(self):
        import torch
        from engine.core.fp8_linear import FP8Linear
        from engine.models.wan22 import transformer

        packed = transformer.load_transformer(self.fp8_path, "cpu", torch.float32, kind="i2v", config=self.config)
        dense = transformer.load_transformer(self.dense_path, "cpu", torch.float32, kind="i2v", config=self.config)
        layers = [module for module in packed.modules() if isinstance(module, FP8Linear)]
        self.assertEqual(len(layers), len(self.scaled))
        self.assertTrue(all(layer.weight.dtype == torch.float8_e4m3fn and layer.bias is not None for layer in layers))
        self.assertFalse(any(isinstance(module, FP8Linear) for module in dense.modules()))
        torch.testing.assert_close(self.forward(packed), self.forward(dense), rtol=1e-5, atol=1e-5)

    def test_a_scaled_file_that_does_not_fit_the_model_is_refused(self):
        import torch
        from engine.models.wan22 import transformer

        with self.assertRaisesRegex(KeyError, "missing keys .*blocks.2."):
            transformer.load_transformer(self.fp8_path, "cpu", torch.float32, config=dict(self.config, num_layers=3))


@unittest.skipUnless(HAVE_WAN and importlib.util.find_spec("gguf") is not None, "torch, diffusers or gguf is missing")
class MergedDistillationTests(unittest.TestCase):
    """lora_merged_share tells an expert with a LoRA's distillation merged in from one with another distillation, by
    the biases and norms that the LoRA's full parameter changes (`.diff_b`, `.diff`) touch, against a GGUF or a
    safetensors base."""

    def setUp(self):
        import gguf
        import torch
        from safetensors.torch import save_file

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        generator = torch.Generator().manual_seed(5)
        names = {"blocks.0.self_attn.q.bias": 64, "blocks.0.self_attn.norm_q.weight": 64, "blocks.0.ffn.0.bias": 128,
                 "blocks.39.cross_attn.o.bias": 64}
        self.base = {name: torch.randn(size, generator=generator) for name, size in names.items()}
        self.base["blocks.1.ffn.2.bias"] = torch.randn(64, generator=generator)  # a block the check does not sample
        changes = {name: torch.randn(value.shape, generator=generator) * 0.01 for name, value in self.base.items()}
        lora = {"diffusion_model." + name.replace(".bias", ".diff_b").replace(".weight", ".diff"): change
                for name, change in changes.items()}
        lora["diffusion_model.blocks.0.self_attn.q.lora_down.weight"] = torch.randn(4, 64, generator=generator)
        lora["diffusion_model.blocks.0.self_attn.q.lora_up.weight"] = torch.zeros(64, 4)
        self.lora = self.root / "lora.safetensors"
        save_file(lora, str(self.lora))
        weight, scale = channel_quantize(torch.randn(64, 64, generator=generator))
        fp8 = {"blocks.0.self_attn.q.weight": weight, "blocks.0.self_attn.q.weight_scale": scale}
        self.merged = self.root / "merged.safetensors"  # stored like LightX2V's files: bf16-exact float32
        save_file({**fp8, **{name: (value + changes[name]).to(torch.bfloat16).float()
                             for name, value in self.base.items()}}, str(self.merged))
        other = {name: (value + torch.randn(value.shape, generator=generator) * 0.01).to(torch.bfloat16).float()
                 for name, value in self.base.items()}
        self.other = self.root / "other.safetensors"
        save_file({**fp8, **other}, str(self.other))
        self.dense_base = self.root / "base.safetensors"
        save_file(self.base, str(self.dense_base))
        self.gguf_base = self.root / "base.gguf"
        writer = gguf.GGUFWriter(str(self.gguf_base), "wan")
        for name, value in self.base.items():
            writer.add_tensor(name, value.numpy())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_a_merged_lora_matches_and_another_distillation_does_not(self):
        from engine.models.wan22.transformer import lora_merged_share

        for base in (self.dense_base, self.gguf_base):
            with self.subTest(base=base.suffix):
                self.assertEqual(lora_merged_share(self.merged, base, self.lora), 1.0)
                self.assertLess(lora_merged_share(self.other, base, self.lora), 0.5)

    def test_a_lora_without_changes_in_the_sampled_blocks_is_refused(self):
        from engine.models.wan22.transformer import lora_merged_share

        with self.assertRaisesRegex(ValueError, "changes no bias or norm"):
            lora_merged_share(self.merged, self.dense_base, self.lora, blocks=(7,))


if __name__ == "__main__":
    unittest.main()
