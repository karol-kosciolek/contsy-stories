# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The UMT5 encoder computes in float32 even when its weights are stored in bfloat16, small tensors
keep the precision of the file, and an unknown text encoder format is refused before anything loads."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAVE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors", "gguf"))

CONFIG = dict(vocab_size=300, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2, num_heads=4,
              relative_attention_num_buckets=32, relative_attention_max_distance=128, dense_act_fn="gelu_new",
              feed_forward_proj="gated-gelu", is_gated_act=True, tie_word_embeddings=False)


def write_encoder(path, norm_value=None):
    """A tiny UMT5 whose weights are exactly representable in bfloat16 (so storing them as bf16 loses nothing)."""
    import torch
    from safetensors.torch import save_file
    from transformers import UMT5Config, UMT5EncoderModel

    torch.manual_seed(7)
    model = UMT5EncoderModel(UMT5Config(**CONFIG))
    state = {}
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().clone().to(torch.bfloat16).to(torch.float32).contiguous()
        if norm_value is not None and tensor.ndim == 1:
            tensor = torch.full_like(tensor, norm_value)
        state[name] = tensor
    save_file(state, str(path))
    return state


@unittest.skipUnless(HAVE, "torch, transformers or gguf is not installed")
class TextPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.path = Path(cls.tmp.name) / "umt5.safetensors"
        write_encoder(cls.path)

    def encode(self, dtype):
        import torch
        from engine.models.wan22 import text

        encoder = text.load_text_encoder(self.path, "cpu", dtype, config=dict(CONFIG))
        tokens = [5, 17, 99, 42, 8, 1, 0, 0]
        return encoder, text._encode_tokens(encoder, tokens, torch.device("cpu"))

    def test_bfloat16_storage_gives_the_float32_result_of_the_same_numbers(self):
        import torch

        _, from_bf16 = self.encode(torch.bfloat16)
        _, from_f32 = self.encode(torch.float32)
        # Before the fix the activation was rounded to bfloat16 before `wo` in every layer: 0.7 percent off.
        self.assertLess(float((from_bf16 - from_f32).abs().max()), 1e-5 * float(from_f32.abs().max()) + 1e-6)

    def test_the_activation_enters_wo_in_float32(self):
        import torch

        encoder, _ = self.encode(torch.bfloat16)
        seen = []
        for block in encoder.encoder.block:
            block.layer[1].DenseReluDense.wo.register_forward_pre_hook(lambda m, args: seen.append(args[0].dtype))
        from engine.models.wan22 import text

        text._encode_tokens(encoder, [5, 1], torch.device("cpu"))
        self.assertEqual(seen, [torch.float32] * CONFIG["num_layers"])
        self.assertEqual(encoder.encoder.block[0].layer[1].DenseReluDense.wo.weight.element_size(), 2)  # still stored in 16 bits

    def test_small_tensors_and_the_embedding_keep_the_precision_of_the_file(self):
        import torch
        from engine.models.wan22 import text
        from safetensors.torch import load_file, save_file

        path = Path(self.tmp.name) / "fp16.safetensors"
        state = load_file(str(self.path))
        value = 1.0 + 2.0 ** -10  # exact in float16, not representable in bfloat16
        state = {k: (torch.full_like(v, value) if v.ndim == 1 else v) for k, v in state.items()}
        state["shared.weight"] = state["shared.weight"].to(torch.float16)
        state["encoder.embed_tokens.weight"] = state["shared.weight"].clone()
        state["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = \
            state["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] + 2.0 ** -10
        save_file({k: v.contiguous() for k, v in state.items()}, str(path))
        encoder = text.load_text_encoder(path, "cpu", torch.bfloat16, config=dict(CONFIG))
        norm = encoder.encoder.final_layer_norm.weight
        self.assertEqual((norm.dtype, float(norm[0])), (torch.float32, value))  # not 1.0078125
        bias = encoder.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight
        self.assertEqual(bias.dtype, torch.float32)
        self.assertTrue(torch.equal(bias, state["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]))
        self.assertEqual(encoder.shared.weight.dtype, torch.float16)  # as stored, no rounding to bfloat16

    def test_weights_expanded_from_scaled_fp8_keep_eleven_significant_bits_not_eight(self):
        """weight * scale rounded to bfloat16 lost 0.4 percent per weight (0.6 percent on the conditioning, measured against
        the product computed in float32); the engine stores it as float16 instead, at the same memory."""
        import torch
        from engine.models.wan22 import text
        from safetensors.torch import load_file, save_file

        torch.manual_seed(5)
        state = load_file(str(self.path))
        scaled, expected = {}, {}
        for name, tensor in state.items():
            if tensor.ndim == 2 and ("SelfAttention." in name or "DenseReluDense." in name) and name.endswith(".weight"):
                scale = tensor.abs().max() / 448.0
                packed = (tensor / scale).to(torch.float8_e4m3fn)
                scaled[name], scaled[name[:-len("weight")] + "scale_weight"] = packed, scale.reshape(())
                expected[name] = packed.float() * scale  # the product in float32
            else:
                scaled[name] = tensor
        scaled["scaled_fp8"] = torch.zeros(0, dtype=torch.float8_e4m3fn)
        path = Path(self.tmp.name) / "scaled.safetensors"
        save_file({k: v.contiguous() for k, v in scaled.items()}, str(path))
        encoder = text.load_text_encoder(path, "cpu", torch.bfloat16, config=dict(CONFIG))
        name = "encoder.block.0.layer.0.SelfAttention.q.weight"
        loaded = encoder.encoder.block[0].layer[0].SelfAttention.q.weight
        self.assertEqual(loaded.dtype, torch.float16)
        self.assertTrue(torch.equal(loaded, expected[name].to(torch.float16)))
        narrow_error = (loaded.float() - expected[name]).abs().max()
        bfloat_error = (expected[name].to(torch.bfloat16).float() - expected[name]).abs().max()
        self.assertLess(float(narrow_error), float(bfloat_error) / 4)  # float16 is about 8 times finer than bfloat16

    def test_an_unknown_text_encoder_format_is_refused_before_anything_loads(self):
        from engine.formats.fp8 import UnsupportedQuantization
        from engine.models.wan22 import text
        from engine.pipelines.video import WanModels

        for call in (lambda: text.check_format("umt5.pt"), lambda: text.read_spiece("umt5.pt"),
                     lambda: text.load_text_encoder("umt5.pt"),
                     lambda: WanModels.from_files("umt5.pt", "h.gguf", "l.gguf", "v.safetensors")):
            with self.assertRaises(UnsupportedQuantization) as raised:
                call()
            self.assertIn("GGUF", str(raised.exception))
            self.assertIn("safetensors", str(raised.exception))
        text.check_format("umt5-xxl-encoder-Q5_K_M.gguf")
        text.check_format("umt5_xxl_fp8_e4m3fn_scaled.safetensors")  # a safetensors file passes


if __name__ == "__main__":
    unittest.main()
