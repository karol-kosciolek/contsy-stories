# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Z-Image-Turbo in the engine: conventions and pinned values, and the pipeline on tiny networks.

Fixtures (tests/fixtures/engine_zimage_*.json): token ids of the official chat template, and reference values of
2026-11-08 for the transformer, the Qwen3 text-encoder path and the Flux autoencoder on tiny networks with
deterministic weights (tests/seeded.py), for the conventions documented in engine/models/zimage. Nothing
here loads a real model; the tests that need torch, diffusers, transformers, the tokenizer files or
a model file skip when they are missing. The headers of the installed official files are checked in
test_engine_zimage_checkpoint.
"""
import gc
import importlib.util
import json
import tempfile
import unittest
import weakref
from pathlib import Path

from app.config import MODELS_DIR
from tests import seeded
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
TOKENIZER_DIR = MODELS_DIR / "tokenizer_z_image"


def _have(*modules: str) -> bool:
    return all(importlib.util.find_spec(name) for name in modules)


HAVE_NETS = _have("torch", "diffusers", "transformers", "safetensors", "PIL")
HAVE_FIXTURES = all((FIXTURES / f"engine_zimage_{name}.json").is_file()
                    for name in ("tokens", "transformer", "text", "vae"))


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"engine_zimage_{name}.json").read_text(encoding="utf-8"))


class ConfigConsistencyTests(unittest.TestCase):
    """The three shipped configs must fit together (no torch needed)."""

    def config(self, name: str) -> dict:
        return json.loads((ROOT / "engine" / "models" / "configs" / f"{name}.json").read_text(encoding="utf-8"))

    def test_text_encoder_feeds_the_transformer(self):
        self.assertEqual(self.config("zimage_text_encoder")["hidden_size"],
                         self.config("zimage_transformer")["cap_feat_dim"])

    def test_vae_latent_matches_the_transformer(self):
        vae, transformer = self.config("zimage_vae"), self.config("zimage_transformer")
        self.assertEqual(vae["latent_channels"], transformer["in_channels"])
        # the VAE config: the decoder gets z / scaling_factor + shift_factor = z / 0.3611 + 0.1159
        self.assertEqual((vae["scaling_factor"], vae["shift_factor"]), (0.3611, 0.1159))

    def test_transformer_time_scale(self):
        # ZImageTransformer2DModel: t_scale = 1000.0
        self.assertEqual(self.config("zimage_transformer")["t_scale"], 1000.0)


@unittest.skipUnless(HAVE_NETS and HAVE_FIXTURES and (TOKENIZER_DIR / "vocab.json").is_file(),
                     "torch/transformers, the fixtures or the tokenizer files are not available")
class TokenizerTests(unittest.TestCase):
    def test_ids_equal_the_official_chat_template(self):
        """diffusers' ZImagePipeline: apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)."""
        from engine.models.zimage import text

        tokenizer = text.load_tokenizer(TOKENIZER_DIR)
        data = fixture("tokens")
        self.assertGreaterEqual(len(data["prompts"]), 5)
        for prompt, expected in zip(data["prompts"], data["ids"]):
            official = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                     add_generation_prompt=True, enable_thinking=True)
            official = list(official["input_ids"] if hasattr(official, "keys") else official)
            with self.subTest(prompt=prompt[:40]):
                self.assertEqual(text.tokenize(tokenizer, prompt), official)
                self.assertEqual(official, expected)

    def test_template_has_no_padding_and_no_think_block(self):
        from engine.models.zimage import text

        tokenizer = text.load_tokenizer(TOKENIZER_DIR)
        ids = text.tokenize(tokenizer, "")
        # <|im_start|>user \n <|im_end|> \n <|im_start|>assistant \n  - nothing else
        self.assertEqual(ids, [151644, 872, 198, 151645, 198, 151644, 77091, 198])

    def test_corpus_covers_unicode_and_long_prompts(self):
        data = fixture("tokens")
        self.assertTrue(any("ż" in p for p in data["prompts"]))
        self.assertTrue(any(len(ids) > 500 for ids in data["ids"]))


@unittest.skipUnless(HAVE_NETS, "torch, diffusers or transformers are not installed")
class PromptLimitTests(unittest.TestCase):
    def test_too_long_prompt_is_refused_not_truncated(self):
        from engine.models.zimage import text

        with self.assertRaises(ValueError):
            text.tokenize(lambda s: {"input_ids": list(range(text.MAX_PROMPT_TOKENS + 1))}, "x")


@unittest.skipUnless(HAVE_NETS and HAVE_FIXTURES, "torch, diffusers, transformers or the fixtures are missing")
class PinnedNetworkTests(unittest.TestCase):
    """Each network, loaded from a file of tiny deterministic weights, must keep its pinned output."""

    @classmethod
    def setUpClass(cls):
        cls.cache = {}
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)

    def save(self, name: str, tensors: dict) -> Path:
        from safetensors.torch import save_file

        path = Path(self.tmp.name) / name
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
        return path

    # --- transformer: t = 1 - sigma in, the negated output is the velocity (diffusers ZImagePipeline) ---

    def transformer_case(self):
        import torch
        from engine.models.zimage import transformer

        if "transformer" not in self.cache:
            data = fixture("transformer")
            path = self.save("transformer.safetensors", seeded.seeded_state_dict(data["shapes"]))
            model = transformer.load_transformer(path, "cpu", torch.float32, config=data["config"])
            latent = seeded.seeded_inputs((1, 16, 8, 8), data["latent_seed"])
            context = seeded.seeded_inputs((5, data["config"]["cap_feat_dim"]), data["context_seed"])
            self.cache["transformer"] = (data, model, latent, context)
        return self.cache["transformer"]

    def test_transformer_velocity_and_denoised_keep_their_pinned_values(self):
        import torch
        from engine.models.zimage import transformer

        data, model, latent, context = self.transformer_case()
        for case in data["outputs"]:
            sigma = torch.tensor(case["sigma"])
            with self.subTest(sigma=case["sigma"]):
                for got, key in ((transformer.predict_velocity(model, latent, sigma, context), "velocity"),
                                 (transformer.predict_denoised(model, latent, sigma, context), "denoised")):
                    expected = torch.tensor(case[key]).reshape(got.shape)
                    torch.testing.assert_close(got, expected, rtol=0, atol=1e-4)

    def test_timestep_and_sign_conventions(self):
        """diffusers' ZImagePipeline: the timestep is `(1000 - t) / 1000` and `noise_pred = -noise_pred`."""
        import torch

        data, model, latent, context = self.transformer_case()
        case = data["outputs"][0]
        sigma = case["sigma"]
        expected = torch.tensor(case["velocity"]).reshape(latent.shape)

        def raw(t: float):
            with torch.no_grad():
                out = model([latent[0].unsqueeze(1)], torch.tensor([t]), [context], return_dict=False)[0]
                return out[0].squeeze(1)[None]

        torch.testing.assert_close(-raw(1.0 - sigma), expected, rtol=0, atol=1e-4)  # the convention
        self.assertGreater((-raw(sigma) - expected).abs().max().item(), 0.05)  # t = sigma would be wrong
        self.assertGreater((raw(1.0 - sigma) - expected).abs().max().item(), 0.5)  # the raw output is the negative

    def test_transformer_key_conversion_splits_fused_qkv(self):
        import torch
        from engine.models.zimage import transformer

        dim = 8
        fused = torch.arange(3 * dim * dim, dtype=torch.float32).reshape(3 * dim, dim)
        out = dict(transformer.diffusers_tensors(iter([("layers.0.attention.qkv.weight", fused),
                                                       ("layers.0.attention.out.weight", torch.zeros(dim, dim)),
                                                       ("layers.0.attention.q_norm.weight", torch.ones(2)),
                                                       ("x_embedder.weight", torch.zeros(dim, 4)),
                                                       ("final_layer.linear.weight", torch.zeros(4, dim))])))
        self.assertEqual(sorted(out), ["all_final_layer.2-1.linear.weight", "all_x_embedder.2-1.weight",
                                       "layers.0.attention.norm_q.weight", "layers.0.attention.to_k.weight",
                                       "layers.0.attention.to_out.0.weight", "layers.0.attention.to_q.weight",
                                       "layers.0.attention.to_v.weight"])
        torch.testing.assert_close(out["layers.0.attention.to_q.weight"], fused[:dim])
        torch.testing.assert_close(out["layers.0.attention.to_v.weight"], fused[2 * dim:])

    # --- text encoder: Qwen3, hidden_states[-2], float32 compute on bf16 weights ---

    def text_case(self):
        import torch
        from engine.models.zimage import text

        if "text" not in self.cache:
            data = fixture("text")
            bf16 = {k: v.to(torch.bfloat16) for k, v in seeded.seeded_state_dict(data["shapes"]).items()}
            path = self.save("text.safetensors", bf16)
            model = text.load_text_encoder(path, "cpu", torch.bfloat16, config=data["config"])
            self.cache["text"] = (data, model)
        return self.cache["text"]

    def test_text_encoder_keeps_its_pinned_values(self):
        import torch
        from engine.models.zimage import text

        data, model = self.text_case()
        self.assertEqual(data["attention_mask"], [1] * len(data["ids"]))  # no padding, so no masking
        got = text.encode(model, data["ids"])
        expected = torch.tensor(data["conditioning"]).reshape(got.shape)
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-4)
        # the weights are still bf16 after the layer-by-layer widening
        self.assertEqual(model.layers[0].self_attn.q_proj.weight.dtype, torch.bfloat16)

    def test_bf16_compute_would_not_match(self):
        """The encoder computes in float32 although its weights are stored in bf16."""
        import torch

        data, model = self.text_case()
        expected = torch.tensor(data["conditioning"])
        with torch.no_grad():
            narrow = model(inputs_embeds=model.embed_tokens(torch.tensor([data["ids"]])),
                           output_hidden_states=True).hidden_states[-2][0].float()
        self.assertGreater((narrow.reshape(-1) - expected).abs().max().item(), 5e-3)

    def test_hidden_state_is_the_output_of_the_second_to_last_layer(self):
        """`hidden_states[-2]`: the output of the second-to-last layer, before the final norm."""
        import torch

        data, model = self.text_case()
        captured = {}
        handle = model.layers[-2].register_forward_hook(lambda module, args, kwargs, out: captured.update(out=out),
                                                        with_kwargs=True)
        try:
            from engine.models.zimage import text

            got = text.encode(model, data["ids"])
        finally:
            handle.remove()
        out = captured["out"]
        out = out if isinstance(out, torch.Tensor) else out[0]
        torch.testing.assert_close(got, out[0].float())

    # --- VAE: Flux autoencoder, the [-1, 1] -> [0, 1] mapping and the 8-bit conversion ---

    def test_vae_decode_keeps_its_pinned_values(self):
        import numpy as np
        import torch
        from engine.models.zimage import vae

        data = fixture("vae")
        path = self.save("vae.safetensors", seeded.seeded_state_dict(data["shapes"]))
        model = vae.load_vae(path, "cpu", torch.float32, config=data["config"])
        latent = seeded.seeded_inputs((1, 16, 6, 8), data["latent_seed"])
        got = np.asarray(vae.decode(model, latent)).astype(int)
        expected = np.array(data["image"]).reshape(data["image_shape"]).astype(int)
        self.assertEqual(got.shape, (48, 64, 3))
        self.assertLessEqual(np.abs(got - expected).max(), 1)  # truncation at a rounding boundary may differ by one level
        self.assertLess((got != expected).mean(), 0.01)
        tiled = np.asarray(vae.decode(model, latent, tiled=True)).astype(int)  # the out-of-memory fallback path
        self.assertLessEqual(np.abs(tiled - expected).max(), 1)
        self.assertFalse(model.use_tiling)


@unittest.skipUnless(HAVE_NETS, "torch, diffusers or transformers are not installed")
class MaterializeTests(unittest.TestCase):
    def tiny(self):
        import torch

        with torch.device("meta"):
            return torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))

    def values(self):
        import torch

        return {"0.weight": torch.ones(2, 3), "0.bias": torch.zeros(2), "1.weight": torch.ones(2), "1.bias": torch.zeros(2)}

    def test_fills_and_casts(self):
        import torch
        from engine.models.zimage import materialize

        model = self.tiny()
        materialize(model, self.values().items(), "cpu", torch.bfloat16, "tiny")
        self.assertEqual(model[0].weight.dtype, torch.bfloat16)
        self.assertFalse(model[0].weight.is_meta)
        self.assertFalse(model[0].weight.requires_grad)

    def test_missing_unexpected_and_misshapen_keys_are_errors(self):
        import torch
        from engine.models.zimage import materialize

        values = self.values()
        with self.assertRaisesRegex(KeyError, "missing"):
            materialize(self.tiny(), [(k, v) for k, v in values.items() if k != "1.bias"], "cpu", None, "tiny")
        with self.assertRaisesRegex(KeyError, "unexpected"):
            materialize(self.tiny(), [*values.items(), ("extra", torch.zeros(1))], "cpu", None, "tiny")
        with self.assertRaisesRegex(ValueError, "shape"):
            materialize(self.tiny(), [("0.weight", torch.ones(3, 3)), *list(values.items())[1:]], "cpu", None, "tiny")


def tiny_models(shared: Optional[dict] = None):
    """Random tiny networks behind the pipeline's loader seams.

    Pass an empty dict as `shared` to get the same transformer and VAE instances on every load (they are
    stored in the dict), so that state left behind by a run, such as a forward hook, stays visible."""
    import torch
    from diffusers import AutoencoderKL, ZImageTransformer2DModel
    from engine.pipelines.image import ZImageModels
    from transformers import Qwen3Config, Qwen3Model

    class Tokenizer:
        def __call__(self, text):
            return {"input_ids": [ord(c) % 250 + 1 for c in text]}

    def text(device, dtype):
        torch.manual_seed(1)
        config = Qwen3Config(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8)
        return Tokenizer(), Qwen3Model(config).to(device=device, dtype=dtype).eval()

    def transformer(device, dtype):
        torch.manual_seed(2)
        model = ZImageTransformer2DModel(
            all_patch_size=(2,), all_f_patch_size=(1,), in_channels=16, dim=64, n_layers=2, n_refiner_layers=1,
            n_heads=4, n_kv_heads=4, cap_feat_dim=32, axes_dims=[4, 6, 6], axes_lens=[256, 64, 64])
        for parameter in model.parameters():
            if parameter.ndim > 1:
                torch.nn.init.normal_(parameter, std=0.05)
        return model.to(device=device, dtype=dtype).eval()

    def autoencoder(device, dtype):
        torch.manual_seed(3)
        return AutoencoderKL(
            in_channels=3, out_channels=3, down_block_types=["DownEncoderBlock2D"] * 4,
            up_block_types=["UpDecoderBlock2D"] * 4, block_out_channels=[8, 8, 16, 16], layers_per_block=1,
            latent_channels=16, norm_num_groups=4, scaling_factor=0.3611, shift_factor=0.1159,
            use_quant_conv=False, use_post_quant_conv=False).to(device=device, dtype=dtype).eval()

    if shared is None:
        return ZImageModels(text, transformer, autoencoder)

    def once(name, build):
        def load(device, dtype):
            if name not in shared:
                shared[name] = build(device, dtype)
            return shared[name]
        return load

    return ZImageModels(text, once("transformer", transformer), once("vae", autoencoder))


def hook_count(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


@unittest.skipUnless(HAVE_NETS, "torch, diffusers or transformers are not installed")
class PipelineTests(unittest.TestCase):

    def generate(self, models=None, seed=5, steps=4, **kwargs):
        import torch
        from engine.pipelines import image

        return image.txt2img(models or tiny_models(), "a photo of a cat", 64, 48, seed, steps=steps, device="cpu",
                             dtype=torch.float32, **kwargs)

    def test_size_and_determinism_per_seed(self):
        first, again, other = self.generate(), self.generate(), self.generate(seed=6)
        self.assertEqual(first.size, (64, 48))
        self.assertEqual(first.mode, "RGB")
        self.assertEqual(first.tobytes(), again.tobytes())
        self.assertNotEqual(first.tobytes(), other.tobytes())

    def test_the_same_seed_gives_the_same_picture_and_leaves_no_hooks(self):
        shared = {}
        models = tiny_models(shared)  # the same transformer object every time: leaked hooks would show
        first, again = self.generate(models), self.generate(models)
        self.assertEqual(first.tobytes(), again.tobytes())
        self.assertNotEqual(first.tobytes(), self.generate(models, seed=6).tobytes())
        self.assertEqual(hook_count(shared["transformer"]), 0)

    def test_progress_is_monotonic_and_reports_every_step(self):
        log = []
        self.generate(steps=4, progress_cb=lambda fraction, message: log.append((fraction, message)))
        fractions = [f for f, _ in log]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[0], 0.0)
        self.assertEqual(fractions[-1], 1.0)
        self.assertEqual([m for _, m in log if m.startswith("Step")], [f"Step {i}/4" for i in range(1, 5)])

    def test_cancelling_mid_run_raises_and_cleans_up(self):
        from engine.core.cancel import CancelToken, Cancelled

        token = CancelToken()
        shared = {}
        models = tiny_models(shared)
        forwards = []
        original = models.load_transformer

        def counting(device, dtype):
            network = original(device, dtype)
            network.register_forward_pre_hook(lambda module, args, kwargs: forwards.append(1), with_kwargs=True)
            return network

        counted = type(models)(models.load_text, counting, models.load_vae)

        def on_progress(fraction, message):
            if message == "Step 2/6":
                token.cancel()

        with self.assertRaises(Cancelled):
            self.generate(counted, steps=6, progress_cb=on_progress, cancel=token)
        self.assertEqual(len(forwards), 2)  # the third step never started
        self.assertEqual(hook_count(shared["transformer"]), 0)  # no hooks left behind
        self.assertFalse(issubclass(Cancelled, Exception))

    def test_cancelled_before_start_loads_nothing(self):
        from engine.core.cancel import CancelToken, Cancelled

        token = CancelToken()
        token.cancel()
        loaded = []
        models = tiny_models()
        spy = type(models)(lambda d, t: loaded.append("text") or models.load_text(d, t),
                           models.load_transformer, models.load_vae)
        with self.assertRaises(Cancelled):
            self.generate(spy, cancel=token)
        self.assertEqual(loaded, [])

    def test_networks_are_released_one_after_another(self):
        """The text encoder is gone before the transformer is built, the transformer before the VAE."""
        base = tiny_models()
        refs = {}
        seen = {}

        def load_text(device, dtype):
            tokenizer, encoder = base.load_text(device, dtype)
            refs["text"] = weakref.ref(encoder)
            return tokenizer, encoder

        def load_transformer(device, dtype):
            gc.collect()
            seen["text alive when the transformer loads"] = refs["text"]() is not None
            network = base.load_transformer(device, dtype)
            refs["transformer"] = weakref.ref(network)
            return network

        def load_vae(device, dtype):
            gc.collect()
            seen["transformer alive when the vae loads"] = refs["transformer"]() is not None
            return base.load_vae(device, dtype)

        self.generate(type(base)(load_text, load_transformer, load_vae))
        self.assertEqual(seen, {"text alive when the transformer loads": False,
                                "transformer alive when the vae loads": False})

    def test_stage_statistics(self):
        stages = []
        self.generate(stats=lambda name, seconds, peak: stages.append((name, seconds, peak)))
        self.assertEqual([name for name, _, _ in stages], ["text encoder", "transformer", "vae"])
        self.assertTrue(all(seconds >= 0 and peak == 0 for _, seconds, peak in stages))  # peak memory is CUDA only

    def test_invalid_arguments(self):
        for kwargs in ({"steps": 0},):
            with self.assertRaises(ValueError):
                self.generate(**kwargs)
        import torch
        from engine.pipelines import image

        for size in ((60, 48), (64, 40), (0, 48)):
            with self.assertRaises(ValueError):
                image.txt2img(tiny_models(), "x", *size, 1, device="cpu", dtype=torch.float32)


class CancelTokenTests(unittest.TestCase):
    def test_token(self):
        from engine.core.cancel import CancelToken, Cancelled

        token = CancelToken()
        token.check()
        self.assertFalse(token.cancelled)
        token.cancel()
        self.assertTrue(token.cancelled)
        with self.assertRaises(Cancelled):
            token.check()
        with self.assertRaises(BaseException):
            token.check()
        self.assertFalse(issubclass(Cancelled, Exception))


if __name__ == "__main__":
    unittest.main()
