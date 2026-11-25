# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Wan 2.2 in the engine: the networks keep their pinned outputs on tiny networks with deterministic weights
(tests/fixtures/engine_wan_*.json, reference values of 2026-11-09 for the conventions documented in
engine/models/wan22), the tokenizer gives the official ids, and GGUF files load. Nothing here loads a real model."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from app.config import destination_dir
from tests import seeded

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
HAVE = all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers", "safetensors"))
HAVE_FIXTURE = (FIXTURES / "engine_wan_transformer.json").is_file()


def diffusers_config(tiny: dict) -> dict:
    """The WanTransformer3DModel config that corresponds to the tiny network of the fixture (original Wan names)."""
    return {"patch_size": tiny["patch_size"], "num_attention_heads": tiny["num_heads"],
            "attention_head_dim": tiny["dim"] // tiny["num_heads"], "in_channels": tiny["in_dim"],
            "out_channels": tiny["out_dim"], "text_dim": tiny["text_dim"], "freq_dim": tiny["freq_dim"],
            "ffn_dim": tiny["ffn_dim"], "num_layers": tiny["num_layers"], "cross_attn_norm": True,
            "qk_norm": "rms_norm_across_heads", "eps": tiny["eps"], "image_dim": None, "added_kv_proj_dim": None,
            "rope_max_seq_len": 1024, "pos_embed_seq_len": None}


@unittest.skipUnless(HAVE and HAVE_FIXTURE, "torch, diffusers or the fixture is missing")
class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from engine.models.wan22 import transformer
        from safetensors.torch import save_file

        cls.data = json.loads((FIXTURES / "engine_wan_transformer.json").read_text(encoding="utf-8"))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.path = Path(cls.tmp.name) / "wan.safetensors"
        save_file({k: v.contiguous() for k, v in seeded.seeded_state_dict(cls.data["shapes"]).items()}, str(cls.path))
        cls.model = transformer.load_transformer(cls.path, "cpu", torch.float32, config=diffusers_config(cls.data["config"]))
        cls.latent = seeded.seeded_inputs(cls.data["latent_shape"], cls.data["latent_seed"])
        cls.context = seeded.seeded_inputs(cls.data["context_shape"], cls.data["context_seed"])

    def test_the_output_keeps_its_pinned_values_at_every_noise_level(self):
        import torch
        from engine.models.wan22 import transformer

        noise, concat = self.latent[:, :16], self.latent[:, 16:]
        for case in self.data["outputs"]:
            sigma = torch.tensor(case["timestep"] / 1000.0)
            got = transformer.predict_velocity(self.model, noise, sigma, self.context, concat)
            expected = torch.tensor(case["output"], dtype=torch.float32).reshape(self.data["output_shape"])
            self.assertEqual(tuple(got.shape), tuple(expected.shape))
            self.assertLess(float((got - expected).abs().max()), 2e-4, case["timestep"])

    def test_the_denoised_estimate_is_x_minus_sigma_times_the_output(self):
        import torch
        from engine.models.wan22 import transformer

        noise, concat = self.latent[:, :16], self.latent[:, 16:]
        sigma = torch.tensor(0.7)
        velocity = transformer.predict_velocity(self.model, noise, sigma, self.context, concat)
        denoised = transformer.predict_denoised(self.model, noise, sigma, self.context, concat)
        torch.testing.assert_close(denoised, noise - 0.7 * velocity)

    def test_every_key_of_the_file_is_used_and_nothing_stays_on_the_meta_device(self):
        names = {name for name, _ in self.model.named_parameters()} | {name for name, _ in self.model.named_buffers()}
        self.assertFalse(any(t.is_meta for t in [*self.model.parameters(), *self.model.buffers()]))
        self.assertIn("rope.freqs_cos", names)  # rebuilt after the load


@unittest.skipUnless(HAVE and (FIXTURES / "engine_wan_conditioning.json").is_file(), "torch or the fixture is missing")
class ConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((FIXTURES / "engine_wan_conditioning.json").read_text(encoding="utf-8"))

    def test_the_start_picture_is_cropped_to_the_centre_and_scaled_bilinearly(self):
        import torch
        from engine.models.wan22 import conditioning

        picture = seeded.seeded_inputs(self.data["picture_shape"], self.data["picture_seed"]).sigmoid()
        width, height = self.data["target"]
        got = conditioning.center_scale(picture, width, height)
        expected = torch.tensor(self.data["scaled"]).reshape(self.data["scaled_shape"])
        self.assertLess(float((got - expected).abs().max()), 1e-6)

    def test_the_conditioning_channels_keep_their_pinned_values(self):
        import torch
        from engine.models.wan22 import conditioning

        latent = seeded.seeded_inputs(self.data["latent_shape"], self.data["latent_seed"])
        got = conditioning.i2v_concat(latent, self.data["known_latent_frames"], self.data["latents_mean"],
                                      self.data["latents_std"])
        expected = torch.tensor(self.data["concat"]).reshape(self.data["concat_shape"])
        self.assertEqual(tuple(got.shape), (1, 20, 3, 6, 8))
        self.assertLess(float((got - expected).abs().max()), 1e-5)
        self.assertEqual(float(got[0, :4, 0].min()), 1.0)  # the known frame, four mask channels
        self.assertEqual(float(got[0, :4, 1:].max()), 0.0)

    def test_the_vae_statistics_of_the_shipped_config_are_the_pinned_ones(self):
        config = json.loads((ROOT / "engine" / "models" / "configs" / "wan_vae.json").read_text(encoding="utf-8"))
        for ours, theirs in ((config["latents_mean"], self.data["latents_mean"]),
                             (config["latents_std"], self.data["latents_std"])):
            self.assertEqual(len(ours), 16)
            for a, b in zip(ours, theirs):
                self.assertAlmostEqual(a, b, places=4)

    def test_the_padding_frames_are_gray_and_the_first_one_is_the_picture(self):
        import torch
        from engine.models.wan22 import conditioning

        start = torch.rand(3, 12, 20)
        frames = conditioning.start_sequence(start, 9, 8, 6)
        self.assertEqual(tuple(frames.shape), (9, 3, 6, 8))
        self.assertTrue(torch.equal(frames[1:], torch.full((8, 3, 6, 8), 0.5)))
        self.assertTrue(torch.equal(frames[0], conditioning.center_scale(start[None], 8, 6)[0]))
        self.assertEqual([conditioning.known_latent_frames(n) for n in (1, 2, 5, 9)], [1, 1, 2, 3])


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"engine_wan_{name}.json").read_text(encoding="utf-8"))


def _umt5_source() -> Path:
    """The Full set's encoder: Wan-AI's sharded set (its index) when installed, else the older single file."""
    from tests.test_engine_wan_official import OFFICIAL_INDEX

    return OFFICIAL_INDEX if OFFICIAL_INDEX.is_file() else (
        destination_dir("text_encoders") / "umt5_xxl_fp8_e4m3fn_scaled.safetensors")


UMT5_FILE = _umt5_source()


@unittest.skipUnless(HAVE and UMT5_FILE.is_file() and (FIXTURES / "engine_wan_tokens.json").is_file(),
                     "the UMT5 encoder file or the fixture is missing")
class TokenizerTests(unittest.TestCase):
    def test_token_ids_are_the_official_pieces_then_the_end_token_then_padding_to_512(self):
        import hashlib

        from engine.models.wan22 import text

        data = fixture("tokens")
        model = text.read_spiece(UMT5_FILE)
        self.assertEqual(hashlib.sha256(model).hexdigest(), data["spiece_sha256"])
        sp = text.load_tokenizer(model)
        for case in data["cases"]:
            expected = case["ids"] + [1]
            expected += [0] * max(0, 512 - len(expected))
            self.assertEqual(text.tokenize(sp, case["prompt"]), expected, case["prompt"][:40])

    def test_parentheses_are_plain_text(self):
        from engine.models.wan22 import text

        sp = text.load_tokenizer(text.read_spiece(UMT5_FILE))
        prompt = "A (very) tall man (in a red coat:1.4)"
        self.assertEqual(text.tokenize(sp, prompt)[:len(sp.encode(prompt)) + 1], list(sp.encode(prompt)) + [1])


@unittest.skipUnless(HAVE and (FIXTURES / "engine_wan_text.json").is_file() and importlib.util.find_spec("transformers"),
                     "torch, transformers or the fixture is missing")
class TextEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from engine.models.wan22 import text
        from safetensors.torch import save_file

        cls.data = fixture("text")
        cfg = cls.data["config"]
        shipped = json.loads((ROOT / "engine" / "models" / "configs" / "wan_text_encoder.json").read_text(encoding="utf-8"))
        config = {k: v for k, v in shipped.items() if not k.startswith("_")}
        config.update(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], d_kv=cfg["d_kv"], d_ff=cfg["d_ff"],
                      num_layers=cfg["num_layers"], num_decoder_layers=cfg["num_layers"], num_heads=cfg["num_heads"])
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        path = Path(cls.tmp.name) / "umt5.safetensors"
        save_file({k: v.contiguous() for k, v in seeded.seeded_state_dict(cls.data["shapes"]).items()}, str(path))
        cls.config = config
        cls.encoder = text.load_text_encoder(path, "cpu", torch.float32, config=config)

    def test_the_hidden_states_equal_the_pinned_values_with_zeros_on_the_padding(self):
        import torch
        from engine.models.wan22 import text

        ids = self.data["ids"]
        got = text._encode_tokens(self.encoder, ids, torch.device("cpu"))
        expected = torch.tensor(self.data["hidden"]).reshape(self.data["hidden_shape"])
        self.assertLess(float((got - expected).abs().max()), 1e-4)
        end = ids.index(1)
        self.assertTrue(bool((got[0, end + 1:] == 0).all()))  # exact zeros after the end token
        self.assertTrue(bool((got[0, :end + 1].abs().sum(dim=-1) > 0).all()))

    def test_a_prompt_is_encoded_once_and_linear_layers_compute_in_float32(self):
        import torch
        from engine.models.wan22 import text

        class Words:
            def encode(self, prompt):
                return [5, 17, 99, 42] if prompt.strip() else []

        got = text.encode_prompt(Words(), self.encoder, "(aaa:1.5)", min_length=12)
        expected = text._encode_tokens(self.encoder, [5, 17, 99, 42, 1] + [0] * 7, torch.device("cpu"))
        torch.testing.assert_close(got, expected)
        self.assertEqual(type(self.encoder.encoder.block[0].layer[0].SelfAttention.q).__name__, "CastLinear")

    def test_the_encoder_keeps_the_files_mapped_tensors_instead_of_copying_them(self):
        """Loaded to the CPU, a bfloat16 matrix stays the tensor safetensors maps from the file (no 10.6 GiB copy);
        `materialize` copies by default and only uses a tensor as it is when asked to and it needs no conversion."""
        from unittest.mock import patch

        import torch
        from torch import nn
        from engine.models import zimage
        from engine.models.wan22 import text
        from safetensors.torch import save_file

        path = Path(self.tmp.name) / "umt5_mapped.safetensors"
        save_file({k: v.contiguous().to(torch.bfloat16) for k, v in seeded.seeded_state_dict(self.data["shapes"]).items()},
                  str(path))
        with patch.object(text, "materialize", wraps=text.materialize) as spy:
            encoder = text.load_text_encoder(path, "cpu", torch.bfloat16, config=self.config)
        self.assertIs(spy.call_args.kwargs["copy"], False)
        self.assertEqual(encoder.encoder.block[0].layer[0].SelfAttention.q.weight.dtype, torch.bfloat16)
        layer = nn.Linear(2, 3, bias=False, device="meta")
        weight = torch.ones(3, 2)
        zimage.materialize(layer, [("weight", weight)], "cpu", torch.float32, "test", copy=False)
        self.assertEqual(layer.weight.data_ptr(), weight.data_ptr())
        zimage.materialize(layer, [("weight", weight)], "cpu", torch.float32, "test")
        self.assertNotEqual(layer.weight.data_ptr(), weight.data_ptr())
        zimage.materialize(layer, [("weight", weight)], "cpu", torch.bfloat16, "test", copy=False)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)  # a conversion still makes its own tensor

    def test_a_dropped_encoder_is_freed_without_the_garbage_collector(self):
        """No reference cycle in the loaded encoder: a gated layer that held a bound method of itself kept its
        weights allocated until the garbage collector ran."""
        import gc
        import weakref

        import torch
        from engine.models.wan22 import text
        from safetensors.torch import save_file

        path = Path(self.tmp.name) / "umt5_gc.safetensors"
        save_file({k: v.contiguous() for k, v in seeded.seeded_state_dict(self.data["shapes"]).items()}, str(path))
        encoder = text.load_text_encoder(path, "cpu", torch.bfloat16, config=self.config)
        self.assertTrue(any(type(m).__name__ == "GatedDense" for m in encoder.modules()))
        layers = [weakref.ref(module) for module in encoder.modules()]
        gc.collect()
        gc.disable()
        try:
            del encoder
            self.assertEqual(sum(layer() is not None for layer in layers), 0)
        finally:
            gc.enable()


@unittest.skipUnless(HAVE and (FIXTURES / "engine_wan_vae.json").is_file(), "torch, diffusers or the fixture is missing")
class VaeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from engine.models.wan22 import vae
        from safetensors.torch import save_file

        cls.data = fixture("vae")
        tiny = cls.data["config"]
        cls.config = {"base_dim": tiny["dim"], "z_dim": tiny["z_dim"], "dim_mult": tiny["dim_mult"],
                      "num_res_blocks": tiny["num_res_blocks"], "attn_scales": tiny["attn_scales"],
                      "temperal_downsample": tiny["temperal_downsample"], "dropout": 0.0,
                      "latents_mean": [0.0] * 16, "latents_std": [1.0] * 16, "is_residual": False,
                      "in_channels": 3, "out_channels": 3, "patch_size": None, "scale_factor_temporal": 4,
                      "scale_factor_spatial": 8}
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        path = Path(cls.tmp.name) / "vae.safetensors"
        save_file({k: v.contiguous() for k, v in seeded.seeded_state_dict(cls.data["shapes"]).items()}, str(path))
        cls.vae = vae.load_vae(path, "cpu", torch.float32, config=cls.config)

    def test_encoding_gives_the_pinned_posterior_mean(self):
        import torch
        from engine.models.wan22 import vae

        video = seeded.seeded_inputs(self.data["video_shape"], self.data["video_seed"]).tanh()
        frames = ((video[0] + 1.0) / 2.0).permute(1, 0, 2, 3)  # [N, 3, H, W] in 0..1
        got = vae.encode(self.vae, frames)
        expected = torch.tensor(self.data["encoded"]).reshape(self.data["encoded_shape"])
        self.assertEqual(tuple(got.shape), tuple(expected.shape))
        self.assertLess(float((got - expected).abs().max()), 1e-4)

    def test_decoding_gives_the_pinned_pixels_after_the_pixel_mapping(self):
        import torch
        from engine.models.wan22 import vae

        latent = seeded.seeded_inputs(self.data["latent_shape"], self.data["latent_seed"])
        got = vae.decode(self.vae, latent)
        raw = torch.tensor(self.data["decoded"]).reshape(self.data["decoded_shape"])
        expected = ((raw[0].clamp(-1, 1) + 1.0) / 2.0).permute(1, 0, 2, 3)
        self.assertEqual(tuple(got.shape), (5, 3, 32, 32))
        self.assertLess(float((got - expected).abs().max()), 1e-4)
        self.assertTrue(0.0 <= float(got.min()) and float(got.max()) <= 1.0)

    def test_the_normalisation_of_the_latent_is_undone_before_decoding(self):
        import torch
        from engine.models.wan22 import vae

        class Stub(torch.nn.Module):
            config = type("C", (), {"latents_mean": [1.0] * 16, "latents_std": [2.0] * 16})()

            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def decode(self, z, return_dict=False):
                self.seen = z
                return (torch.zeros(1, 3, 1, 8, 8),)

        stub = Stub()
        vae.decode(stub, torch.full((1, 16, 1, 1, 1), 3.0))
        self.assertTrue(bool((stub.seen == 7.0).all()))  # 3 * 2 + 1


@unittest.skipUnless(HAVE and HAVE_FIXTURE and importlib.util.find_spec("gguf"), "torch, diffusers, gguf or the fixture is missing")
class GgufLoadingTests(unittest.TestCase):
    """A tiny Wan transformer written as a GGUF file (Q8_0 where the shape allows): the quantized layers must
    give the same result as dense layers made from the dequantized weights."""

    @classmethod
    def setUpClass(cls):
        import gguf
        import numpy as np
        import torch
        from engine.models.wan22 import transformer
        from safetensors.torch import save_file

        cls.data = json.loads((FIXTURES / "engine_wan_transformer.json").read_text(encoding="utf-8"))
        cls.config = diffusers_config(cls.data["config"])
        state = seeded.seeded_state_dict(cls.data["shapes"])
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.gguf_path = Path(cls.tmp.name) / "wan.gguf"
        writer = gguf.GGUFWriter(str(cls.gguf_path), "wan")
        dense = {}
        cls.quantized_names = []
        for name, tensor in state.items():
            array = tensor.numpy().astype(np.float32)
            if array.ndim == 2 and array.shape[1] % 32 == 0 and name.endswith(".weight"):
                packed = gguf.quants.quantize(array, gguf.GGMLQuantizationType.Q8_0)
                writer.add_tensor(name, packed, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                dense[name] = torch.from_numpy(gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.Q8_0))
                cls.quantized_names.append(name)
            else:
                writer.add_tensor(name, array)
                dense[name] = tensor
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        cls.dense_path = Path(cls.tmp.name) / "dense.safetensors"
        save_file({k: v.contiguous() for k, v in dense.items()}, str(cls.dense_path))
        cls.transformer = transformer

    def forward(self, model):
        import torch

        latent = seeded.seeded_inputs(self.data["latent_shape"], self.data["latent_seed"])
        context = seeded.seeded_inputs(self.data["context_shape"], self.data["context_seed"])
        return self.transformer.predict_velocity(model, latent[:, :16], torch.tensor(0.7), context, latent[:, 16:])

    def test_quantized_layers_give_the_dense_result_of_the_dequantized_weights(self):
        import torch
        from engine.core.quant import QLinear

        packed = self.transformer.load_transformer(self.gguf_path, "cpu", torch.float32, config=self.config)
        dense = self.transformer.load_transformer(self.dense_path, "cpu", torch.float32, config=self.config)
        count = sum(isinstance(m, QLinear) for m in packed.modules())
        self.assertGreater(count, 10)
        self.assertEqual(count, len(self.quantized_names))
        torch.testing.assert_close(self.forward(packed), self.forward(dense), rtol=1e-4, atol=1e-4)

    def test_a_file_that_does_not_fit_the_model_is_refused(self):
        import torch

        for path in (self.gguf_path, self.dense_path):
            with self.subTest(file=path.name), self.assertRaisesRegex(KeyError, "missing keys .*blocks.2."):
                self.transformer.load_transformer(path, "cpu", torch.float32,
                                                  config=dict(self.config, num_layers=3))  # deeper than the file


if __name__ == "__main__":
    unittest.main()
