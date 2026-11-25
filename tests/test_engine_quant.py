# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Quantized linear layers: the result equals a dense layer with the dequantized weight, in every
loading order, and LoRA hooks work on top."""
import importlib.util
import unittest
from types import SimpleNamespace

HAVE = all(importlib.util.find_spec(name) is not None for name in ("torch", "gguf", "numpy"))


def quantized(name, rows, columns, seed):
    """(dense reference weight, QTensor) of a Q8_0 matrix."""
    import gguf
    import numpy as np
    import torch
    from engine.formats import ggml

    weight = np.random.default_rng(seed).standard_normal((rows, columns)).astype(np.float32)
    raw = gguf.quants.quantize(weight, gguf.GGMLQuantizationType.Q8_0)
    tensor = ggml.QTensor(name, ggml.QT.Q8_0, (rows, columns), torch.from_numpy(raw))
    return tensor.to(torch.float32), tensor


@unittest.skipUnless(HAVE, "torch or gguf is not installed")
class QuantizedLinearTests(unittest.TestCase):
    def test_zimage_fused_qkv_splits_quantized_rows_without_expanding(self):
        import torch
        from engine.models.zimage.transformer import _gguf_tensors

        key = "model.diffusion_model.layers.0.attention.qkv.weight"
        dense, packed = quantized(key, 12, 64, 15)
        result = dict(_gguf_tensors(SimpleNamespace(tensors={key: packed})))
        expected_names = [f"layers.0.attention.to_{name}.weight" for name in "qkv"]
        self.assertEqual(list(result), expected_names)
        for key, expected in zip(expected_names, dense.chunk(3)):
            self.assertTrue(result[key].quantized)
            torch.testing.assert_close(result[key].to(torch.float32), expected)

    def test_the_result_equals_a_dense_layer_with_the_dequantized_weight(self):
        import torch
        from engine.core.quant import QLinear
        from torch import nn

        dense, packed = quantized("w", 32, 64, 1)
        bias = torch.randn(32)
        layer = QLinear(packed, bias)
        x = torch.randn(5, 64)
        self.assertIsInstance(layer, nn.Linear)  # LoRA hooks and shape checks see an ordinary layer
        self.assertEqual((tuple(layer.weight.shape), layer.weight.is_meta), ((32, 64), True))
        torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, dense, bias))
        small = layer(x.to(torch.bfloat16))
        self.assertEqual(small.dtype, torch.bfloat16)

    def test_moving_a_model_moves_the_bytes_and_keeps_the_placeholder(self):
        import torch
        from engine.core.quant import QLinear

        _, packed = quantized("w", 8, 64, 2)
        model = torch.nn.Sequential(QLinear(packed, torch.zeros(8)))
        model.to("cpu")
        model.to(torch.float64)
        layer = model[0]
        self.assertTrue(layer.weight.is_meta)
        self.assertEqual((layer.qweight.data.dtype, layer.bias.dtype), (torch.uint8, torch.float64))

    def test_a_dtype_only_cast_leaves_the_bytes_on_their_device(self):
        """`.to(torch.bfloat16)` names no device: bytes on the GPU stay there (the meta device stands in for it)."""
        import torch
        from engine.core.quant import QLinear
        from engine.formats import ggml

        _, packed = quantized("w", 8, 64, 3)
        on_meta = ggml.QTensor(packed.name, packed.qtype, packed.shape, torch.empty_like(packed.data, device="meta"))
        layer = QLinear(on_meta, torch.zeros(8, device="meta"))
        layer.to(torch.bfloat16)
        self.assertEqual((layer.qweight.data.device.type, layer.bias.device.type, layer.bias.dtype),
                         ("meta", "meta", torch.bfloat16))

    def build(self):
        import torch
        from torch import nn

        with torch.device("meta"):
            return nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 16), nn.LayerNorm(16))

    def test_loading_order_does_not_matter_and_plain_tensors_stay_dense(self):
        import torch
        from engine.core.quant import QLinear, materialize_quantized

        d0, q0 = quantized("0.weight", 32, 64, 3)
        d1, q1 = quantized("1.weight", 16, 32, 4)
        biases = {"0.bias": torch.randn(32), "1.bias": torch.randn(16)}
        norm = {"2.weight": torch.rand(16) + 0.5, "2.bias": torch.randn(16)}
        orders = [[("0.bias", biases["0.bias"]), ("0.weight", q0), ("1.weight", q1), ("1.bias", biases["1.bias"])],
                  [("0.weight", q0), ("0.bias", biases["0.bias"]), ("1.bias", biases["1.bias"]), ("1.weight", q1)]]
        x = torch.randn(3, 64)
        for order in orders:
            model = self.build()
            count = materialize_quantized(model, [*order, *norm.items()], "cpu", torch.float32, "test")
            self.assertEqual(count, 2)
            self.assertIsInstance(model[0], QLinear)
            self.assertNotIsInstance(model[2], QLinear)
            reference = torch.nn.functional.layer_norm(
                torch.nn.functional.linear(torch.nn.functional.linear(x, d0, biases["0.bias"]), d1, biases["1.bias"]),
                (16,), norm["2.weight"], norm["2.bias"])
            torch.testing.assert_close(model(x), reference)

    def test_a_dense_gguf_tensor_and_wrong_files_are_handled(self):
        import torch
        from engine.core.quant import materialize_quantized
        from engine.formats import ggml

        model = self.build()
        _, q0 = quantized("0.weight", 32, 64, 5)
        dense_one = ggml.QTensor("1.weight", ggml.QT.F32, (16, 32), torch.randn(16, 32))  # not block-quantized
        items = [("0.weight", q0), ("0.bias", torch.zeros(32)), ("1.weight", dense_one), ("1.bias", torch.zeros(16)),
                 ("2.weight", torch.ones(16)), ("2.bias", torch.zeros(16))]
        self.assertEqual(materialize_quantized(model, items, "cpu", torch.float32, "test"), 1)
        self.assertEqual(type(model[1]).__name__, "Linear")
        self.assertFalse(model[1].weight.is_meta)
        with self.assertRaises(KeyError):  # a missing key
            materialize_quantized(self.build(), items[:-1], "cpu", torch.float32, "test")
        with self.assertRaises(ValueError):  # a wrong shape
            materialize_quantized(self.build(), [("0.bias", torch.zeros(31))], "cpu", torch.float32, "test")

    def test_lora_hooks_work_on_quantized_layers(self):
        import torch
        from engine.core.quant import materialize_quantized
        from engine.formats.lora import LoRA, LoRAWeights, attach

        dense, packed = quantized("0.weight", 32, 64, 6)
        model = torch.nn.Sequential(torch.nn.Linear(64, 32, device="meta"))
        materialize_quantized(model, [("0.weight", packed), ("0.bias", torch.zeros(32))], "cpu", torch.float32, "t")
        down, up = torch.randn(4, 64) * 0.1, torch.randn(32, 4) * 0.1
        x = torch.randn(6, 64)
        before = model(x)
        remove = attach(model, [(LoRA({"0": LoRAWeights(down, up, None)}), 1.0)])
        after = model(x)
        remove()
        torch.testing.assert_close(after - before, (x @ down.T) @ up.T, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(model(x), before)


if __name__ == "__main__":
    unittest.main()
