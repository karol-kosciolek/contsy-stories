# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""EzAudio-XL in the engine: loaders, the DDIM sampler, the networks and the pipeline contract, on the CPU.

Tiny random networks stand in for the real ones. The math of the ported networks and of the sampling chain is pinned
against the official code by tests/fixtures/engine_ezaudio.json: the outputs of the official MaskDiT, Oobleck decoder
and inference() loop on tiny networks with deterministic weights, made by tools/parity_ezaudio.py --make-fixture.
Tests with the real files skip when the bundle is missing (set CONTSY_EZAUDIO_BUNDLE to a folder laid out like
models/ezaudio). Tests against the official modules themselves run when CONTSY_EZAUDIO_REFERENCE names a checkout of
github.com/haidog-yaqub/EzAudio (tools/parity_ezaudio.py does the same with the real weights).
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import MODELS_DIR

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "engine_ezaudio.json"
HAVE = all(importlib.util.find_spec(name)
           for name in ("torch", "safetensors", "transformers", "sentencepiece", "scipy"))
HAVE_DIFFUSERS = HAVE and importlib.util.find_spec("diffusers") is not None
BUNDLE = Path(os.environ.get("CONTSY_EZAUDIO_BUNDLE", MODELS_DIR / "ezaudio"))
REFERENCE = os.environ.get("CONTSY_EZAUDIO_REFERENCE", "")
HAVE_BUNDLE = HAVE and (BUNDLE / "ezaudio_s3_xl.pt").is_file()
HAVE_REFERENCE = HAVE and bool(REFERENCE) and (Path(REFERENCE) / "src" / "models" / "conditioners.py").is_file()

TINY = dict(dim=32, latent_channels=4, context_dim=12, heads=2, hidden=48, rank=3, depth=4)
TINY_T5 = dict(model_type="t5", vocab_size=40, d_model=12, d_kv=6, d_ff=20, num_layers=2, num_heads=2,
               relative_attention_num_buckets=8, relative_attention_max_distance=16, feed_forward_proj="gated-gelu",
               layer_norm_epsilon=1e-6, dropout_rate=0.0, tie_word_embeddings=False, is_encoder_decoder=True,
               pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
XL_YAML = """model_name: EzAudio-XL

model:
  mae: True
  mae_prob: 0.25
  mask_ratio: [0.25, 1.0]
  mask_span: 10
  img_size: 500
  patch_size: 1
  in_chans: 257
  out_chans: 128
  input_type: '1d'
  embed_dim: 1152
  depth: 28
  num_heads: 16
  mlp_ratio: 4.0
  qkv_bias: false
  qk_scale: null
  qk_norm: layernorm
  norm_layer: layernorm
  act_layer: geglu
  context_norm: true
  use_checkpoint: true
  time_fusion: 'ada_sola_bias'
  ada_sola_rank: 36
  ada_sola_alpha: 36
  cls_dim: null
  context_dim: 2048
  context_fusion: 'cross'
  context_max_length: null
  context_pe_method: 'none'
  pe_method: 'none'
  rope_mode: 'shared'
  use_conv: true
  skip: true
  skip_norm: true

autoencoder:
  name: stable_vae
  dim: 128
  sr: 24000
  latent_sr: 50
  q_first: true
  scale: 1.0
  shift: 0.0

text_encoder:
  model: google/flan-t5-xl
  max_length: 100
  cfg: 0.1

diff:
  num_train_timesteps: 1000
  beta_schedule: 'scaled_linear'
  beta_start: 0.00085
  beta_end: 0.012
  prediction_type: 'v_prediction'
  rescale_betas_zero_snr: true
  timestep_spacing: 'trailing'
  clip_sample: false
"""
VAE_CONFIG = {"sample_rate": 24000, "model": {
    "decoder": {"type": "oobleck", "config": {"out_channels": 1, "channels": 4, "c_mults": [1, 2, 4, 8],
                                              "strides": [2, 4, 6, 10], "latent_dim": 128, "use_snake": True,
                                              "final_tanh": False}},
    "latent_dim": 128, "downsampling_ratio": 480, "io_channels": 1}}


def randomize(module, seed: int):
    """Every parameter random (norm scales near 1), so that no zero initialization hides a wrong wiring."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            scale = name.rsplit(".", 1)[-1] == "weight" and ("norm" in name or name.endswith("layer_norm.weight"))
            parameter.copy_(1 + 0.1 * noise if scale else 0.3 * noise)
    return module


def tiny_transformer(seed: int = 0):
    import torch

    from engine.models.ezaudio import transformer

    model = randomize(transformer.Transformer(**TINY), seed)
    head = TINY["dim"] // TINY["heads"]
    model.rotary_inv_freq.copy_(1.0 / (10000 ** (torch.arange(0, head, 2).float() / head)))
    return model.eval()


_TOP = (("patch_embed.", "model.patch_embed.proj."), ("time_embed.", "model.time_embed.mlp."),
        ("final_norm.", "model.final_block.norm."), ("final_linear.", "model.final_block.linear."),
        ("final_conv.", "model.final_block.final_layer."), ("time_ada", "model.time_ada"),
        ("context_embed.", "model.context_embed."))
_IN_BLOCK = (("mlp_in.", "mlp.net.0.proj."), ("mlp_out.", "mlp.net.2."),
             ("scale_shift_table", "adaln.scale_shift_table"), ("sola_a.", "adaln.lora_a."),
             ("sola_b.", "adaln.lora_b."))


def _rename(name: str, pairs) -> str:
    for old, new in pairs:
        if name.startswith(old):
            return new + name[len(old):]
    return name


def official_state(model) -> dict:
    """The tiny transformer's tensors under the names of the official checkpoint (MaskDiT + UDiT)."""
    half = len(model.blocks) // 2

    def prefix(index: int) -> str:
        if index < half:
            return f"model.in_blocks.{index}"
        return "model.mid_block" if index == half else f"model.out_blocks.{index - half - 1}"

    state = {}
    for name, tensor in model.state_dict().items():
        if name == "rotary_inv_freq":
            continue
        if name.startswith("blocks."):
            _, index, rest = name.split(".", 2)
            state[f"{prefix(int(index))}.{_rename(rest, _IN_BLOCK)}"] = tensor.clone()
        else:
            state[_rename(name, _TOP)] = tensor.clone()
    for index in range(len(model.blocks)):
        state[f"{prefix(index)}.attn.rotary.inv_freq"] = model.rotary_inv_freq.clone()
    return state


def tiny_decoder(seed: int = 0):
    """Random decoder whose output stays near unit scale (weights ~ 1/sqrt(fan-in)), so float32 rounding stays small."""
    import torch

    from engine.models.ezaudio import vae

    decoder = vae.Decoder(4, 4, [1, 2, 4, 8], [2, 4, 6, 10])
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in decoder.named_parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            if parameter.ndim == 3:
                fan_in = parameter.shape[1] * parameter.shape[2]
                if isinstance(decoder.get_submodule(name.rsplit(".", 1)[0]), torch.nn.ConvTranspose1d):
                    fan_in = parameter.shape[0] * parameter.shape[2] // 2
                parameter.copy_(noise / fan_in ** 0.5)
            else:
                parameter.copy_(0.1 * noise)
    return decoder.eval()


def tiny_t5(seed: int = 0):
    import torch
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(seed)
    return T5EncoderModel(T5Config(**TINY_T5)).eval()


class CharTokenizer:
    """A stand-in for SentencePiece: one id per character."""

    @staticmethod
    def encode(prompt):
        return [2 + ord(char) % 37 for char in prompt]


@contextlib.contextmanager
def reference_on_path():
    """Import the official modules (pure PyTorch) from the reference checkout; its prints are silenced."""
    sys.path.insert(0, REFERENCE)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        sys.path.remove(REFERENCE)
        for name in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
            del sys.modules[name]


@unittest.skipUnless(HAVE, "torch, safetensors, transformers, sentencepiece or scipy is missing")
class ConfigTests(unittest.TestCase):
    def write(self, folder, text):
        path = Path(folder) / "ezaudio-xl.yml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_the_published_file_gives_the_sizes_and_the_schedule(self):
        from engine.models.ezaudio import config

        with tempfile.TemporaryDirectory() as folder:
            settings = config.load(self.write(folder, XL_YAML))
        self.assertEqual(settings.sizes, dict(dim=1152, latent_channels=128, context_dim=2048, heads=16, hidden=4608,
                                              rank=36, depth=28))
        self.assertEqual((settings.max_tokens, settings.sample_rate, settings.latent_rate, settings.window_frames),
                         (100, 24000, 50, 500))
        self.assertEqual((settings.train_timesteps, settings.beta_start, settings.beta_end), (1000, 0.00085, 0.012))
        self.assertEqual((settings.latent_scale, settings.latent_shift), (1.0, 0.0))

    def test_reader_handles_the_scalars_of_the_files(self):
        from engine.models.ezaudio import config

        with tempfile.TemporaryDirectory() as folder:
            data = config.read_yaml(self.write(folder, XL_YAML + "# a comment\nextra: 'x'  # trailing\n"))
        self.assertEqual(data["model"]["mask_ratio"], [0.25, 1.0])
        self.assertIs(data["model"]["mae"], True)
        self.assertIsNone(data["model"]["qk_scale"])
        self.assertEqual(data["model"]["input_type"], "1d")
        self.assertEqual(data["text_encoder"]["model"], "google/flan-t5-xl")
        self.assertEqual(data["extra"], "x")

    def test_settings_the_engine_does_not_implement_are_refused(self):
        from engine.models.ezaudio import config

        changes = [("time_fusion: 'ada_sola_bias'", "time_fusion: 'ada'"), ("ada_sola_alpha: 36", "ada_sola_alpha: 18"),
                   ("act_layer: geglu", "act_layer: gelu"), ("rope_mode: 'shared'", "rope_mode: 'dual'"),
                   ("timestep_spacing: 'trailing'", "timestep_spacing: 'leading'"),
                   ("prediction_type: 'v_prediction'", "prediction_type: 'epsilon'")]
        for old, new in changes:
            with self.subTest(new=new), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    config.load(self.write(folder, XL_YAML.replace(old, new)))

    def test_checkpoint_sizes_must_match_the_configuration(self):
        from engine.models.ezaudio import config

        with tempfile.TemporaryDirectory() as folder:
            settings = config.load(self.write(folder, XL_YAML))
        config.check_sizes(settings, dict(settings.sizes))
        with self.assertRaises(ValueError):
            config.check_sizes(settings, {**settings.sizes, "depth": 24})


@unittest.skipUnless(HAVE_DIFFUSERS, "diffusers is missing")
class DdimTests(unittest.TestCase):
    """engine.sampling.ddim against diffusers' DDIMScheduler built from the EzAudio YAML values."""

    def reference(self):
        from diffusers import DDIMScheduler

        return DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", beta_start=0.00085,
                             beta_end=0.012, prediction_type="v_prediction", rescale_betas_zero_snr=True,
                             timestep_spacing="trailing", clip_sample=False)

    def test_schedule_and_timesteps_are_identical(self):
        import torch

        from engine.sampling import ddim

        reference = self.reference()
        ours = ddim.scaled_linear(1000, 0.00085, 0.012, True)
        self.assertTrue(torch.equal(ours.alphas_cumprod, reference.alphas_cumprod))
        self.assertEqual(float(ours.alphas_cumprod[-1]), 0.0)  # zero terminal SNR
        for steps in (1, 7, 25, 30, 50, 100, 1000):
            reference.set_timesteps(steps)
            self.assertEqual(ddim.trailing_timesteps(1000, steps), reference.timesteps.tolist())
        with self.assertRaises(ValueError):
            ddim.trailing_timesteps(1000, 1001)

    def test_every_step_equals_diffusers(self):
        import torch

        from engine.sampling import ddim

        reference, ours = self.reference(), ddim.scaled_linear(1000, 0.00085, 0.012, True)
        for steps in (7, 30, 50, 100):
            for eta in (0.0, 1.0):
                reference.set_timesteps(steps)
                generator = torch.Generator().manual_seed(steps)
                sample = torch.randn(1, 6, 21, generator=generator)
                for timestep in ddim.trailing_timesteps(1000, steps):
                    velocity = 0.5 * torch.randn(sample.shape, generator=generator) + 0.1 * sample
                    noise = torch.randn(sample.shape, generator=generator)
                    expected = reference.step(velocity, torch.tensor(timestep), sample, eta=eta,
                                              variance_noise=noise if eta else None).prev_sample
                    got = ddim.step(ours, steps, velocity, timestep, sample, eta, noise)
                    with self.subTest(steps=steps, eta=eta, timestep=timestep):
                        self.assertLessEqual(float((got - expected).abs().max()), 1e-6)
                    sample = expected

    def test_stochastic_steps_never_turn_into_nan(self):
        """diffusers returns NaN at 25 steps with eta 1 (a rounding error under a square root); ours does not."""
        import torch

        from engine.sampling import ddim

        schedule = ddim.scaled_linear(1000, 0.00085, 0.012, True)
        generator = torch.Generator().manual_seed(5)
        sample = torch.randn(1, 4, 16, generator=generator)
        for timestep in ddim.trailing_timesteps(1000, 25):
            sample = ddim.step(schedule, 25, 0.1 * sample, timestep, sample, 1.0,
                               torch.randn(sample.shape, generator=generator))
        self.assertTrue(bool(torch.isfinite(sample).all()))

    def test_guidance_rescale_matches_the_reference_formula(self):
        import torch

        from engine.sampling import ddim

        def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):  # EzAudio src/inference.py
            std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
            std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
            noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
            return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg

        generator = torch.Generator().manual_seed(1)
        guided, conditional = torch.randn(2, 1, 8, 30, generator=generator)
        for factor in (0.0, 0.75, 1.0):
            torch.testing.assert_close(ddim.rescale_guidance(guided, conditional, factor),
                                       rescale_noise_cfg(guided, conditional, factor), rtol=0, atol=0)


@unittest.skipUnless(HAVE, "torch, safetensors, transformers, sentencepiece or scipy is missing")
class TransformerTests(unittest.TestCase):
    def save(self, folder, state):
        import torch

        path = Path(folder) / "ezaudio.pt"
        torch.save(state, path)
        return path

    def test_official_checkpoint_layout_loads_every_tensor(self):
        import torch

        from engine.models.ezaudio import transformer

        model = tiny_transformer()
        with tempfile.TemporaryDirectory() as folder:
            loaded = transformer.load(self.save(folder, {"model": official_state(model)}), "cpu", torch.float32)
        self.assertEqual(loaded.sizes, TINY)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0, msg=name)

    def test_missing_unused_or_inconsistent_tensors_are_refused(self):
        import torch

        from engine.models.ezaudio import transformer

        state = official_state(tiny_transformer())
        broken = {
            "extra": {**state, "model.unused.weight": torch.zeros(1)},
            "missing": {key: value for key, value in state.items() if key != "model.out_blocks.0.skip_linear.weight"},
            "rotary": {**state, "model.out_blocks.1.attn.rotary.inv_freq": torch.ones(8)},
        }
        for case, bad in broken.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises((KeyError, ValueError)):
                    transformer.load(self.save(folder, {"model": bad}), "cpu", torch.float32)
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            transformer.load(self.save(folder, {"model": state, "ema": state}), "cpu", torch.float32)

    def test_bfloat16_weights_keep_norms_and_time_conditioning_in_float32(self):
        import torch

        from engine.models.ezaudio import transformer

        model = tiny_transformer()
        with tempfile.TemporaryDirectory() as folder:
            half = transformer.load(self.save(folder, {"model": official_state(model)}), "cpu", torch.bfloat16)
        dtypes = {name: tensor.dtype for name, tensor in half.state_dict().items()}
        # The edge layers (input, text and output projections, final convolution) stay float32: in bfloat16 they
        # pushed the per-forward error of the real network above 2 percent.
        for name in ("blocks.0.norm1.weight", "blocks.0.attn.norm_q.weight", "time_ada.weight",
                     "blocks.3.sola_b.weight", "mask_embed", "rotary_inv_freq", "blocks.0.scale_shift_table",
                     "blocks.4.skip_norm.weight", "patch_embed.weight", "context_embed.0.weight",
                     "context_embed.2.weight", "final_linear.weight", "final_conv.weight"):
            self.assertEqual(dtypes[name], torch.float32, name)
        for name in ("blocks.0.attn.to_q.weight", "blocks.2.mlp_in.weight", "blocks.1.attn.proj.weight",
                     "blocks.4.skip_linear.weight", "blocks.2.cross_attn.to_k.weight"):
            self.assertEqual(dtypes[name], torch.bfloat16, name)
        generator = torch.Generator().manual_seed(3)
        latent = torch.randn(2, 4, 15, generator=generator)
        embedding = torch.randn(2, 6, 12, generator=generator)
        keys = torch.tensor([[True] * 6, [True] * 2 + [False] * 4])
        with torch.inference_mode():
            full = model(latent, torch.tensor(500), model.prepare_context(embedding), keys, model.prepare_rotary(15))
            low = half(latent, torch.tensor(500), half.prepare_context(embedding), keys, half.prepare_rotary(15))
        self.assertLess(float((low - full).norm() / full.norm()), 0.03)

    def test_masked_text_tokens_do_not_change_the_output(self):
        import torch

        model = tiny_transformer()
        generator = torch.Generator().manual_seed(4)
        latent = torch.randn(1, 4, 9, generator=generator)
        embedding = torch.randn(1, 6, 12, generator=generator)
        other = embedding.clone()
        other[:, 3:] = 100.0  # padding as the encoder leaves it must not matter
        keys = torch.tensor([[True] * 3 + [False] * 3])
        with torch.inference_mode():
            a = model(latent, torch.tensor(10), model.prepare_context(embedding), keys, model.prepare_rotary(9))
            b = model(latent, torch.tensor(10), model.prepare_context(other), keys, model.prepare_rotary(9))
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(HAVE_REFERENCE, "CONTSY_EZAUDIO_REFERENCE does not name an EzAudio checkout")
    def test_forward_equals_the_official_masked_dit(self):
        import torch

        from engine.models.ezaudio import transformer

        with reference_on_path():
            from src.models.conditioners import MaskDiT

            reference = MaskDiT(
                mae=True, mae_prob=0.25, mask_ratio=[0.25, 1.0], mask_span=10, img_size=500, patch_size=1, in_chans=9,
                out_chans=4, input_type="1d", embed_dim=32, depth=4, num_heads=2, mlp_ratio=1.5, qkv_bias=False,
                qk_scale=None, qk_norm="layernorm", norm_layer="layernorm", act_layer="geglu", context_norm=True,
                use_checkpoint=False, time_fusion="ada_sola_bias", ada_sola_rank=3, ada_sola_alpha=3, cls_dim=None,
                context_dim=12, context_fusion="cross", context_max_length=None, context_pe_method="none",
                pe_method="none", rope_mode="shared", use_conv=True, skip=True, skip_norm=True)
            randomize(reference, 9).eval()
            with tempfile.TemporaryDirectory() as folder:
                ours = transformer.load(self.save(folder, {"model": reference.state_dict()}), "cpu", torch.float32)
            generator = torch.Generator().manual_seed(6)
            latent = torch.randn(2, 4, 23, generator=generator)
            embedding = torch.randn(2, 7, 12, generator=generator)
            keys = torch.tensor([[True] * 7, [True] * 3 + [False] * 4])
            known = torch.randn(2, 4, 23, generator=generator)
            generate = torch.zeros(2, 4, 23, dtype=torch.bool)
            generate[..., 8:17] = True
            for timestep in (999, 500, 20):
                with torch.inference_mode():
                    plain, _ = reference(latent, torch.tensor(timestep), embedding, context_mask=keys)
                    filled, _ = reference(latent, torch.tensor(timestep), embedding, context_mask=keys,
                                          gt=known.clone(), mae_mask_infer=generate)
                    context, rotary = ours.prepare_context(embedding), ours.prepare_rotary(23)
                    got_plain = ours(latent, torch.tensor(timestep), context, keys, rotary)
                    got_filled = ours(latent, torch.tensor(timestep), context, keys, rotary, known, generate[:, :1])
                with self.subTest(timestep=timestep):
                    torch.testing.assert_close(got_plain, plain, rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(got_filled, filled, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(HAVE, "torch, safetensors, transformers, sentencepiece or scipy is missing")
class DecoderTests(unittest.TestCase):
    def write(self, folder, state, config=VAE_CONFIG):
        import torch

        folder = Path(folder)
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
        torch.save({"state_dict": state}, folder / "1m.pt")
        return folder / "1m.pt"

    def official_state(self, decoder, generator):
        """The decoder's tensors as the training checkpoint stores them: weight-normalized convolutions (g, v),
        prefixed, next to encoder, discriminator and loss tensors that the loader must skip."""
        import torch

        state = {"autoencoder.encoder.layers.0.weight_v": torch.zeros(2),
                 "discriminator.discriminators.0.w": torch.zeros(1), "losses_gen.losses.0.x": torch.zeros(1),
                 "sdstft.stft_losses.0.window": torch.zeros(1)}
        expected = {}
        for name, tensor in decoder.state_dict().items():
            if name.endswith(".weight") and tensor.ndim == 3:
                v = torch.randn(tensor.shape, generator=generator)
                g = torch.rand((tensor.shape[0], 1, 1), generator=generator) + 0.5
                stem = f"autoencoder.decoder.{name.removesuffix('.weight')}"
                state[f"{stem}.weight_g"], state[f"{stem}.weight_v"] = g, v
                expected[name] = g * v / v.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()  # weight_norm with dim 0
            else:
                state[f"autoencoder.decoder.{name}"] = tensor.clone()
                expected[name] = tensor
        return state, expected

    def test_weight_norm_is_folded_and_other_parts_are_skipped(self):
        import torch

        from engine.models.ezaudio import vae

        state, expected = self.official_state(vae.Decoder(128, 4, [1, 2, 4, 8], [2, 4, 6, 10]),
                                              torch.Generator().manual_seed(2))
        with tempfile.TemporaryDirectory() as folder:
            decoder = vae.load(self.write(folder, state), "cpu")
        for name, tensor in decoder.state_dict().items():
            torch.testing.assert_close(tensor, expected[name], rtol=1e-6, atol=1e-6, msg=name)

    def test_unknown_or_orphan_tensors_and_other_decoders_are_refused(self):
        import torch

        from engine.models.ezaudio import vae

        state, _ = self.official_state(vae.Decoder(128, 4, [1, 2, 4, 8], [2, 4, 6, 10]),
                                       torch.Generator().manual_seed(2))
        orphan = {key: value for key, value in state.items() if key != "autoencoder.decoder.layers.0.weight_v"}
        for bad in ({**state, "autoencoder.bottleneck.scale": torch.ones(1)}, orphan):
            with tempfile.TemporaryDirectory() as folder, self.assertRaises((KeyError, ValueError)):
                vae.load(self.write(folder, bad), "cpu")
        decoder = VAE_CONFIG["model"]["decoder"]["config"]
        for change in ({"final_tanh": True}, {"antialias_activation": True}, {"strides": [2, 4, 6, 5]}):
            config = json.loads(json.dumps(VAE_CONFIG))
            config["model"]["decoder"]["config"] = {**decoder, **change}
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
                vae.load(self.write(folder, state, config), "cpu")

    def test_chunked_decoding_equals_decoding_at_once(self):
        import torch

        from engine.models.ezaudio import vae

        decoder = tiny_decoder()
        latent = torch.randn(1, 4, 47, generator=torch.Generator().manual_seed(8))
        with torch.inference_mode():
            expected = decoder(latent)[0]
            scale = float(expected.abs().max())
            with patch.object(vae, "_CHUNK_FRAMES", 10):
                got = vae.decode(decoder, latent, lambda: None)
                # The receptive field is 9 frames per side: 9 context frames suffice, 8 do not.
                with patch.object(vae, "_CONTEXT_FRAMES", 9):
                    nine = vae.decode(decoder, latent, lambda: None)
                with patch.object(vae, "_CONTEXT_FRAMES", 8):
                    eight = vae.decode(decoder, latent, lambda: None)
        self.assertEqual(tuple(got.shape), (1, 47 * vae.HOP_LENGTH))
        self.assertLess(float((got - expected).abs().max()), 2e-5 * scale)
        self.assertLess(float((nine - expected).abs().max()), 2e-5 * scale)
        self.assertGreater(float((eight - expected).abs().max()), 1e-3 * scale)


@unittest.skipUnless(HAVE, "torch, safetensors, transformers, sentencepiece or scipy is missing")
class TextTests(unittest.TestCase):
    def test_tokens_end_with_the_end_token_and_are_padded_or_cut(self):
        from engine.models.ezaudio import text

        ids, mask = text.tokenize(CharTokenizer(), ["abc", "", "x" * 20], 8)
        self.assertEqual(ids[0].tolist(), [2 + ord("a") % 37, 2 + ord("b") % 37, 2 + ord("c") % 37, 1, 0, 0, 0, 0])
        self.assertEqual(mask[0].tolist(), [True] * 4 + [False] * 4)
        self.assertEqual(ids[1].tolist(), [1] + [0] * 7)  # the reference's unconditional text
        self.assertEqual(mask[1].tolist(), [True] + [False] * 7)
        self.assertEqual(ids[2].tolist(), [2 + ord("x") % 37] * 7 + [1])
        self.assertTrue(bool(mask[2].all()))

    def test_prompts_are_cleaned_to_one_line_of_plain_text(self):
        from engine.models.ezaudio import text

        self.assertEqual(text.clean_prompt("  rain &amp;amp; wind\n on   glass "), "rain & wind on glass")

    def write_folder(self, folder, model, extra=None):
        from safetensors.torch import save_file
        import torch

        folder = Path(folder)
        (folder / "config.json").write_text(json.dumps(TINY_T5), encoding="utf-8")
        tensors = {name: tensor.contiguous() for name, tensor in model.state_dict().items()
                   if name != "encoder.embed_tokens.weight"}
        tensors["decoder.block.0.layer.0.SelfAttention.q.weight"] = torch.zeros(12, 12)
        tensors["lm_head.weight"] = torch.zeros(40, 12)
        tensors.update(extra or {})
        save_file(tensors, str(folder / "model-00001-of-00002.safetensors"))
        names = list(tensors) + ["encoder.embed_tokens.weight"]  # the official index lists the tied table too
        weight_map = {name: "model-00001-of-00002.safetensors" for name in names}
        weight_map["decoder.final_layer_norm.weight"] = "model-00002-of-00002.safetensors"  # never read
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
        return folder

    def test_encoder_reads_only_the_encoder_and_matches_transformers(self):
        import torch

        from engine.models.ezaudio import text

        reference = tiny_t5()
        ids, mask = text.tokenize(CharTokenizer(), ["a dog barks", ""], 8)
        with torch.inference_mode():
            expected = reference(input_ids=ids, attention_mask=mask).last_hidden_state
        with tempfile.TemporaryDirectory() as folder:
            self.write_folder(folder, reference)
            exact = text.load_encoder(folder, "cpu", torch.float32)
            half = text.load_encoder(folder, "cpu", torch.bfloat16)
        torch.testing.assert_close(text.encode(exact, ids, mask), expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(half.encoder.block[0].layer[1].DenseReluDense.wo.weight.dtype, torch.float16)
        self.assertEqual(half.encoder.final_layer_norm.weight.dtype, torch.float32)
        self.assertIs(half.encoder.embed_tokens.weight, half.shared.weight)
        got = text.encode(half, ids, mask)
        self.assertEqual(got.dtype, torch.float32)
        torch.testing.assert_close(got, expected, rtol=2e-3, atol=2e-3)

    def test_a_dropped_encoder_is_freed_without_the_garbage_collector(self):
        """No reference cycle in the loaded encoder: its weights go when the pipeline drops it. A gated layer that
        held a bound method of itself kept 0.6-1.4 GiB allocated on the GPU after calls until gc ran."""
        import gc
        import weakref

        import torch

        from engine.models.ezaudio import text

        with tempfile.TemporaryDirectory() as folder:
            self.write_folder(folder, tiny_t5())
            encoder = text.load_encoder(folder, "cpu", torch.bfloat16)
        self.assertTrue(any(type(m).__name__ == "GatedDense" for m in encoder.modules()))
        layers = [weakref.ref(module) for module in encoder.modules()]
        gc.collect()
        gc.disable()
        try:
            del encoder
            self.assertEqual(sum(layer() is not None for layer in layers), 0)
        finally:
            gc.enable()

    def test_unknown_tensors_in_the_checkpoint_are_refused(self):
        import torch

        from engine.models.ezaudio import text

        with tempfile.TemporaryDirectory() as folder:
            self.write_folder(folder, tiny_t5(), {"pooler.weight": torch.zeros(2)})
            with self.assertRaises(KeyError):
                text.load_encoder(folder, "cpu", torch.float32)

    @unittest.skipUnless(HAVE and (BUNDLE / "text_encoder" / "spiece.model").is_file(),
                         "the FLAN-T5 tokenizer is missing")
    def test_token_ids_equal_the_transformers_t5_tokenizer(self):
        from transformers import T5Tokenizer

        from app import sound
        from engine.models.ezaudio import text

        folder = BUNDLE / "text_encoder"
        reference = T5Tokenizer.from_pretrained(str(folder), local_files_only=True)
        sp = text.load_tokenizer(folder)
        scenes = ["A dog barks at the door while rain falls on the roof", "Kroki na korytarzu, ktoś puka do drzwi",
                  "A busy cafe with clinking cups and quiet chatter", "Waves crash on the beach; seagulls call",
                  "", "Thunder rolls over the forest at night, wind in the trees, an owl hoots"]
        prompts = [sound.prompt_from_scene(scene, "") for scene in scenes]
        prompts += [tag for _, tag, _ in sound._LEXICON] + ["", "x " * 150, "Rain (heavy) - 3 cars pass, whoosh!"]
        prompts = [text.clean_prompt(prompt) for prompt in prompts]
        ids, mask = text.tokenize(sp, prompts, 100)
        expected = reference(prompts, max_length=100, padding="max_length", truncation=True, return_tensors="pt")
        self.assertEqual(ids.tolist(), expected.input_ids.tolist())
        self.assertEqual(mask.tolist(), expected.attention_mask.bool().tolist())


def tiny_settings():
    from engine.models.ezaudio import config

    return config.Settings(text_encoder="google/flan-t5-xl", max_tokens=8, sample_rate=24000, latent_rate=50,
                           window_frames=40, latent_scale=1.0, latent_shift=0.0, train_timesteps=1000,
                           beta_start=0.00085, beta_end=0.012, sizes=dict(TINY))


@unittest.skipUnless(HAVE, "torch, safetensors, transformers, sentencepiece or scipy is missing")
class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.calls = {"text": 0, "transformer": 0, "vae": 0}
        self.recycled = []
        context = patch("engine.pipelines.ezaudio.CONTEXT_SECONDS", 0.2)  # 10 frames of a 40-frame window
        context.start()
        self.addCleanup(context.stop)

    def models(self, transformer_model=None):
        from engine.pipelines import ezaudio

        def load_text(device, dtype):
            self.calls["text"] += 1
            return CharTokenizer(), tiny_t5()

        def load_transformer(device, dtype):
            self.calls["transformer"] += 1
            return transformer_model or tiny_transformer()

        def load_vae(device):
            self.calls["vae"] += 1
            return tiny_decoder()

        return ezaudio.EzAudioModels(tiny_settings(), load_text, load_transformer, load_vae,
                                     recycle=self.recycled.append)

    def run_many(self, sounds, **kwargs):
        from engine.pipelines import ezaudio

        kwargs = {"device": "cpu", "steps": 2, **kwargs}
        return ezaudio.generate_many(self.models(kwargs.pop("transformer_model", None)), sounds, "", **kwargs)

    def test_each_sound_is_its_length_plus_both_handles_at_48_khz(self):
        from engine.pipelines import ezaudio

        sounds = [ezaudio.Sound("rain", 0.3, 1), ezaudio.Sound("door", 1.37, 2), ezaudio.Sound("wind", 1.0, 3)]
        waveforms = self.run_many(sounds)
        for sound, waveform in zip(sounds, waveforms):
            self.assertEqual(tuple(waveform.shape), (1, round((sound.seconds + 0.5) * 48000)))
            self.assertEqual(str(waveform.dtype), "torch.float32")
            self.assertEqual(waveform.device.type, "cpu")
            self.assertLessEqual(float(waveform.abs().max()), 1.0)
        self.assertEqual(self.calls, {"text": 1, "transformer": 1, "vae": 1})  # one load for all sounds
        self.assertEqual(len(self.recycled), 3)

    def test_same_seed_same_sound_other_seed_other_sound(self):
        import torch

        from engine.pipelines import ezaudio

        first, again, other = self.run_many([ezaudio.Sound("rain", 0.5, 7), ezaudio.Sound("rain", 0.5, 7),
                                             ezaudio.Sound("rain", 0.5, 8)])
        torch.testing.assert_close(first, again, rtol=0, atol=0)
        self.assertGreater(float((first - other).abs().max()), 1e-4)

    def test_sampling_loop_equals_the_reference_inference_loop(self):
        """The EzAudio src/inference.py loop with diffusers' DDIMScheduler, a CPU generator and the same network:
        guidance as a batch of two, guidance rescale, stochastic DDIM, the known frames copied back."""
        import torch
        from diffusers import DDIMScheduler

        from engine.pipelines import ezaudio
        from engine.sampling import ddim

        model = tiny_transformer(3)
        embedding = torch.randn(2, 8, 12, generator=torch.Generator().manual_seed(1))
        keys = torch.tensor([[True] * 5 + [False] * 3, [True] + [False] * 7])
        context = model.prepare_context(embedding)
        known = torch.randn(1, 4, 6, generator=torch.Generator().manual_seed(2))
        for steps, cfg, rescale, eta, continued in ((4, 5.0, 0.75, 1.0, False), (3, 3.5, 0.0, 0.5, True),
                                                    (5, 1.0, 0.0, 0.0, False)):
            scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", beta_start=0.00085,
                                      beta_end=0.012, prediction_type="v_prediction", rescale_betas_zero_snr=True,
                                      timestep_spacing="trailing", clip_sample=False)
            frames = 20
            gt = mask = None
            if continued:
                gt = torch.cat((known, torch.zeros(1, 4, 14)), dim=-1)
                mask = torch.zeros(1, 1, frames, dtype=torch.bool)
                mask[..., 6:] = True
            generator = torch.Generator().manual_seed(11)
            scheduler.set_timesteps(steps)
            latents = torch.randn((1, 4, frames), generator=generator)
            with torch.inference_mode():
                for t in scheduler.timesteps:
                    if cfg != 1.0:
                        both = model(torch.cat([latents, latents]), t, context, keys, model.prepare_rotary(frames),
                                     gt, mask)
                        text_out, uncond_out = torch.chunk(both, 2, dim=0)
                        pred = uncond_out + cfg * (text_out - uncond_out)
                        if rescale > 0:
                            std_text = text_out.std(dim=[1, 2], keepdim=True)
                            std_cfg = pred.std(dim=[1, 2], keepdim=True)
                            pred = rescale * (pred * (std_text / std_cfg)) + (1 - rescale) * pred
                    else:
                        pred = model(latents, t, context[:1], keys[:1], model.prepare_rotary(frames), gt, mask)
                    latents = scheduler.step(model_output=pred, timestep=t, sample=latents, eta=eta,
                                             generator=generator).prev_sample
                if continued:
                    latents = torch.where(mask, latents, gt)
                got = ezaudio._sample_window(
                    model, ddim.scaled_linear(1000, 0.00085, 0.012, True), ddim.trailing_timesteps(1000, steps), steps,
                    context, keys, known if continued else None, 14 if continued else frames, cfg, rescale, eta,
                    torch.Generator().manual_seed(11), torch.device("cpu"), lambda: None)
            with self.subTest(steps=steps, cfg=cfg, eta=eta, continued=continued):
                torch.testing.assert_close(got, latents, rtol=1e-6, atol=1e-6)

    def test_long_sounds_continue_with_the_last_frames_as_known_audio(self):
        import torch

        from engine.pipelines import ezaudio

        model = tiny_transformer()
        seen = []
        original = model.forward

        def spy(latent, timestep, context, context_mask, rotary, known=None, mask=None):
            seen.append((latent.shape[-1], None if known is None else known.clone(),
                         None if mask is None else mask.clone()))
            return original(latent, timestep, context, context_mask, rotary, known, mask)

        windows = []
        sample_window = ezaudio._sample_window

        def record(*args, **kwargs):
            result = sample_window(*args, **kwargs)
            windows.append(result.clone())
            return result

        with patch.object(model, "forward", side_effect=spy), patch.object(ezaudio, "_sample_window", record):
            waveform, = self.run_many([ezaudio.Sound("rain", 1.5, 4)], transformer_model=model, steps=1)
        # 2 s with handles = 100 frames: 40 new, then 30 + 30 with 10 known each.
        self.assertEqual(ezaudio.window_plan(100, 40, 10), [ezaudio._Window(0, 40), ezaudio._Window(10, 30),
                                                             ezaudio._Window(10, 30)])
        self.assertEqual([frames for frames, _, _ in seen], [40, 40, 40])
        self.assertIsNone(seen[0][1])
        for previous, (_, known, mask) in zip(windows, seen[1:]):
            torch.testing.assert_close(known[..., :10], previous[..., -10:])
            self.assertEqual(mask[0, 0].tolist(), [False] * 10 + [True] * 30)
        self.assertEqual(tuple(waveform.shape), (1, 2 * 48000))

    def test_min_seconds_generates_a_longer_window_and_returns_its_beginning(self):
        from engine.pipelines import ezaudio

        model = tiny_transformer()
        frames = []
        original = model.forward

        def spy(latent, *args):
            frames.append(latent.shape[-1])
            return original(latent, *args)

        with patch.object(model, "forward", side_effect=spy):
            waveform, = self.run_many([ezaudio.Sound("click", 0.1, 4)], transformer_model=model, steps=1,
                                      min_seconds=0.2)
        self.assertEqual(frames, [35])  # (0.2 + 0.5) s at 50 frames per second, not (0.1 + 0.5) s
        self.assertEqual(tuple(waveform.shape), (1, round(0.6 * 48000)))
        for bad in (-1.0, 31.0, float("inf")):
            with self.subTest(min_seconds=bad), self.assertRaises(ValueError):
                self.run_many([ezaudio.Sound("click", 0.1, 4)], min_seconds=bad)

    def test_window_plan_for_the_longest_bed(self):
        from engine.pipelines import ezaudio

        plan = ezaudio.window_plan(1525, 500, 125)
        self.assertEqual([(part.known, part.new) for part in plan], [(0, 500), (125, 375), (125, 375), (125, 275)])
        self.assertEqual([(part.known, part.new) for part in ezaudio.window_plan(275, 500, 125)], [(0, 275)])

    def test_cancel_stops_before_any_weight_is_loaded(self):
        from engine.core.cancel import Cancelled, CancelToken
        from engine.pipelines import ezaudio

        token = CancelToken()
        token.cancel()
        with self.assertRaises(Cancelled):
            self.run_many([ezaudio.Sound("rain", 1.0, 1)], cancel=token)
        self.assertEqual(self.calls, {"text": 0, "transformer": 0, "vae": 0})
        with self.assertRaises(Cancelled):
            ezaudio.generate(Path("no such bundle"), "rain", "", 5, 1, device="cpu", cancel=token)

    def test_cancel_during_sampling_releases_the_transformer(self):
        from engine.core.cancel import Cancelled, CancelToken
        from engine.pipelines import ezaudio

        token = CancelToken()
        steps = []

        def progress(fraction, message):
            if message.startswith("Generating"):
                steps.append(fraction)
                if len(steps) == 2:
                    token.cancel()

        with self.assertRaises(Cancelled):
            self.run_many([ezaudio.Sound("rain", 1.0, 1)], cancel=token, progress_cb=progress, steps=4)
        self.assertEqual(self.calls["vae"], 0)
        self.assertEqual(len(self.recycled), 2)  # the encoder and the transformer went back

    def test_invalid_requests_fail_before_loading(self):
        from engine.pipelines import ezaudio

        bad = [([ezaudio.Sound("rain", 0, 1)], {}), ([ezaudio.Sound("rain", 30.5, 1)], {}),
               ([ezaudio.Sound("rain", float("nan"), 1)], {}), ([ezaudio.Sound("rain", 1, -1)], {}),
               ([ezaudio.Sound("rain", 1, 1)], {"steps": 0}), ([ezaudio.Sound("rain", 1, 1)], {"cfg": -1.0}),
               # The reference's "guidance off" (0) would be the unconditional prediction here: refused.
               ([ezaudio.Sound("rain", 1, 1)], {"cfg": 0.0}), ([ezaudio.Sound("rain", 1, 1)], {"cfg": 0.5}),
               ([ezaudio.Sound("rain", 1, 1)], {"cfg": float("nan")}),
               ([ezaudio.Sound("rain", 1, 1)], {"rescale": 2.0}), ([], {})]
        for sounds, kwargs in bad:
            with self.subTest(sounds=sounds, kwargs=kwargs), self.assertRaises(ValueError):
                self.run_many(sounds, **kwargs)
        self.assertEqual(self.calls["text"], 0)

    def test_a_transformer_of_other_sizes_is_refused(self):
        from engine.models.ezaudio import transformer
        from engine.pipelines import ezaudio

        other = transformer.Transformer(**{**TINY, "depth": 2})
        with self.assertRaises(ValueError):
            self.run_many([ezaudio.Sound("rain", 0.5, 1)], transformer_model=other)

    def test_loud_decoder_output_is_scaled_not_clipped(self):
        import torch

        from engine.pipelines import ezaudio

        def loud(decoder, latent, check):
            ramp = torch.linspace(-3.0, 3.0, latent.shape[-1] * 480).reshape(1, -1)
            return ramp

        with patch.object(ezaudio.vae, "decode", side_effect=loud):
            waveform, = self.run_many([ezaudio.Sound("thunder", 0.5, 1)])
        self.assertAlmostEqual(float(waveform.abs().max()), 1.0, places=5)
        middle = waveform[0, 2000:-2000]  # away from the resampling filter's ringing at the ends
        self.assertGreater(float((middle[1:] - middle[:-1]).min()), 0.0)  # still a rising ramp, no flat top
        self.assertGreater(float(middle.max()), 0.7)

    def test_resampling_keeps_the_band_and_the_level(self):
        import numpy as np
        import torch

        from engine.pipelines import ezaudio

        time = np.arange(24000) / 24000
        for frequency in (1000.0, 10000.0):
            tone = torch.from_numpy(np.sin(2 * np.pi * frequency * time)).float()[None]
            out = ezaudio.to_output_rate(tone, 24000, 48000)[0].double().numpy()
            middle = out[4800:-4800]
            expected = np.sin(2 * np.pi * frequency * np.arange(48000) / 48000)[4800:-4800]
            with self.subTest(frequency=frequency):
                self.assertLess(np.abs(middle - expected).max(), 2e-3)
        image = torch.from_numpy(np.sin(2 * np.pi * 11000.0 * time)).float()[None]
        resampled = ezaudio.to_output_rate(image, 24000, 48000)[0].double().numpy()
        spectrum = np.abs(np.fft.rfft(resampled * np.hanning(resampled.size)))
        self.assertLess(spectrum[13000] / spectrum[11000], 1e-3)  # the image at 13 kHz is removed
        self.assertEqual(tuple(ezaudio.to_output_rate(image, 24000, 50000).shape), (1, 50000))  # padded

    def test_bundle_files_are_listed_by_role(self):
        from engine.pipelines import ezaudio

        with tempfile.TemporaryDirectory() as folder:
            files = ezaudio.EzAudioFiles.from_bundle(folder)
            self.assertIn("transformer", files.missing())
            self.assertIn("text encoder weights", files.missing())
            with self.assertRaises(FileNotFoundError):
                ezaudio.EzAudioModels.from_files(files)


def load_tool(name: str):
    """A script of tools/ as a module (the parity tool's seeded weights and stand-ins are shared with its fixture
    generator)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAVE and FIXTURE.is_file(), "torch, transformers, scipy or the fixture is missing")
class OfficialFixtureTests(unittest.TestCase):
    """The port against numbers the official code computed on tiny networks (tests/fixtures/engine_ezaudio.json),
    so wrong block, decoder or sampling math fails on every machine, without the reference checkout."""

    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool("parity_ezaudio")
        cls.data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)

    def transformer(self):
        """The tiny network from a checkpoint in the official layout, through the engine's loader."""
        import torch

        from engine.models.ezaudio import transformer

        data = self.data["transformer"]
        state = self.tool.seeded_tensors(data["shapes"])
        state.update({name: torch.tensor(values) for name, values in data["buffers"].items()})
        path = Path(self.folder.name) / "ezaudio.pt"
        torch.save({"model": state}, path)
        return transformer.load(path, "cpu", torch.float32)

    def test_transformer_equals_the_official_masked_dit(self):
        import torch

        data = self.data["transformer"]
        model = self.transformer()
        self.assertEqual(model.sizes, TINY)
        channels, frames = TINY["latent_channels"], data["frames"]
        latent = self.tool.seeded_inputs((2, channels, frames), data["latent_seed"])
        embedding = self.tool.seeded_inputs((2, data["tokens"], TINY["context_dim"]), data["context_seed"])
        known = self.tool.seeded_inputs((2, channels, frames), data["known_seed"])
        keys = torch.tensor(data["context_mask"])
        generate = torch.zeros((1, 1, frames), dtype=torch.bool)
        generate[..., data["generate"][0]:data["generate"][1]] = True
        with torch.inference_mode():
            context, rotary = model.prepare_context(embedding), model.prepare_rotary(frames)
            for case in data["outputs"]:
                timestep = torch.tensor(case["timestep"])
                got = {"plain": model(latent, timestep, context, keys, rotary),
                       "inpainting": model(latent, timestep, context, keys, rotary, known, generate)}
                for kind, output in got.items():
                    with self.subTest(timestep=case["timestep"], kind=kind):
                        expected = torch.tensor(case[kind]).reshape(output.shape)
                        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_decoder_equals_the_official_oobleck_decoder(self):
        import torch

        from engine.models.ezaudio import vae

        data = self.data["decoder"]
        folder = Path(self.folder.name) / "vae"
        folder.mkdir(exist_ok=True)
        (folder / "config.json").write_text(json.dumps(data["config_json"]), encoding="utf-8")
        state = {f"autoencoder.decoder.{name}": tensor
                 for name, tensor in self.tool.seeded_tensors(data["shapes"]).items()}  # weight_g / weight_v pairs
        torch.save({"state_dict": state}, folder / "1m.pt")
        decoder = vae.load(folder / "1m.pt", "cpu")
        latent = self.tool.seeded_inputs((1, 128, data["frames"]), data["latent_seed"])
        expected = torch.tensor(data["waveform"]).reshape(1, -1)
        with torch.inference_mode():
            whole = decoder(latent)[0]
            with patch.object(vae, "_CHUNK_FRAMES", 2):
                chunked = vae.decode(decoder, latent, lambda: None)
        for name, got in (("at once", whole), ("in chunks", chunked)):
            with self.subTest(name):
                torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6 * data["peak"])

    def test_generate_many_hands_the_decoder_the_latent_of_the_official_loop(self):
        """generate_many (prompts and negative in one text batch, guidance pairing, rescale, DDIM with the CPU
        generator, latent scale and shift) against src/inference.py with the same tiny networks; at cfg 1 against
        the reference's guidance 0."""
        import dataclasses

        import torch

        from engine.models.ezaudio import vae
        from engine.pipelines import ezaudio

        chain = self.data["chain"]
        settings = dataclasses.replace(tiny_settings(), max_tokens=chain["max_tokens"],
                                       latent_rate=chain["latent_rate"], latent_scale=chain["latent_scale"],
                                       latent_shift=chain["latent_shift"])
        model = self.transformer()
        encoder = self.tool.table_encoder(chain["vocab"], TINY["context_dim"], chain["table_seed"])
        models = ezaudio.EzAudioModels(settings, lambda device, dtype: (self.tool.CharTokens(), encoder),
                                       lambda device, dtype: model, lambda device: None)
        sounds = [ezaudio.Sound(prompt, seconds, seed) for prompt, seconds, seed in chain["sounds"]]
        for run in chain["runs"]:
            latents = []

            def capture(decoder, latent, check):
                latents.append(latent.clone())
                return torch.zeros((1, latent.shape[-1] * vae.HOP_LENGTH))

            # Every sound fits one window of the tiny settings (40 frames); the context of continuations must too.
            with patch.object(ezaudio.vae, "decode", side_effect=capture), \
                    patch.object(ezaudio, "CONTEXT_SECONDS", 0.2):
                ezaudio.generate_many(models, sounds, chain["negative"], device="cpu", dtype=torch.float32,
                                      steps=chain["steps"], cfg=run["cfg"], rescale=run["rescale"], eta=run["eta"])
            self.assertEqual(len(latents), len(sounds))
            for number, (got, values, frames) in enumerate(zip(latents, run["latents"], run["frames"])):
                with self.subTest(cfg=run["cfg"], sound=number):
                    self.assertEqual(tuple(got.shape), (1, TINY["latent_channels"], frames))
                    torch.testing.assert_close(got, torch.tensor(values).reshape(got.shape), rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(HAVE_REFERENCE and HAVE_DIFFUSERS,
                         "CONTSY_EZAUDIO_REFERENCE does not name an EzAudio checkout")
    def test_fixture_is_what_the_official_code_computes(self):
        """Regenerates the fixture from the checkout in a separate process (the generator replaces unused imports of
        the reference with empty modules, which must not leak into this one) and compares the numbers."""
        import torch

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "engine_ezaudio.json"
            subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "parity_ezaudio.py"), "--make-fixture",
                            "--reference", REFERENCE, "--fixture-path", str(target)],
                           check=True, capture_output=True, timeout=600)
            fresh = json.loads(target.read_text(encoding="utf-8"))
        stored = self.data
        self.assertEqual(fresh["transformer"]["shapes"], stored["transformer"]["shapes"])
        self.assertEqual(fresh["decoder"]["shapes"], stored["decoder"]["shapes"])
        pairs = [(a[kind], b[kind]) for a, b in zip(fresh["transformer"]["outputs"], stored["transformer"]["outputs"])
                 for kind in ("plain", "inpainting")]
        pairs.append((fresh["decoder"]["waveform"], stored["decoder"]["waveform"]))
        pairs += [(a, b) for run_a, run_b in zip(fresh["chain"]["runs"], stored["chain"]["runs"])
                  for a, b in zip(run_a["latents"], run_b["latents"])]
        for number, (new, old) in enumerate(pairs):
            with self.subTest(part=number):
                torch.testing.assert_close(torch.tensor(new), torch.tensor(old), rtol=1e-5,
                                           atol=1e-6 * max(1.0, max(map(abs, old))))


@unittest.skipUnless(HAVE, "torch, safetensors, transformers, sentencepiece or scipy is missing")
class BenchToolTests(unittest.TestCase):
    """tools/bench_ezaudio.py: the figures the lead records on the GPU."""

    def setUp(self):
        self.bench = load_tool("bench_ezaudio")

    def test_compute_is_divided_by_the_clip_not_by_the_layered_sounds(self):
        selected = self.bench.cases(["window", "scene", "long", "warm"], 1)
        self.assertEqual({name: clip for name, (clip, _) in selected.items()},
                         {"window": 9.5, "scene": 5.0, "long": 30.0, "warm": 5.0})
        clip, sounds = selected["scene"]
        self.assertEqual([sound.seconds for sound in sounds], [5.0, 2.0, 2.0])  # the bed and two effects over it
        stages = {name: {"seconds": 1.0} for name in self.bench.LOADS + self.bench.COMPUTE}
        result = self.bench.figures(stages, 6.5, clip, sounds, 0.25)
        self.assertAlmostEqual(result["compute_per_clip_second"], 3.0 / 5.0)
        self.assertAlmostEqual(result["total_per_clip_second"], 6.5 / 5.0)
        self.assertAlmostEqual(result["generated_seconds"], 10.5)
        self.assertAlmostEqual(result["compute_per_generated_second"], 3.0 / 10.5)

    def test_memory_left_on_the_gpu_is_flagged(self):
        before = {"allocated": 100, "reserved": 2 ** 20}
        self.assertFalse(self.bench.memory_change(before, dict(before))["left_behind"])
        self.assertTrue(self.bench.memory_change(before, {"allocated": 612, "reserved": 2 ** 20})["left_behind"])
        self.assertTrue(self.bench.memory_change(before, {"allocated": 100, "reserved": 2 ** 21})["left_behind"])
        self.assertEqual(self.bench.memory_change(before, {"allocated": 612, "reserved": 2 ** 20})
                         ["left_allocated_bytes"], 512)
        self.assertEqual(self.bench.memory_change(None, None), {})
        self.assertIsNone(self.bench.memory_now("cpu"))

    def test_one_cpu_run_reports_per_clip_figures_and_memory_fields(self):
        import argparse

        from engine.pipelines import ezaudio

        models = ezaudio.EzAudioModels(tiny_settings(), lambda device, dtype: (CharTokenizer(), tiny_t5()),
                                       lambda device, dtype: tiny_transformer(), lambda device: tiny_decoder())
        args = argparse.Namespace(device="cpu", dtype="float32", negative="", cfg=5.0, rescale=0.75, eta=1.0,
                                  min_seconds=0.0)
        sounds = [ezaudio.Sound("rain", 0.2, 1), ezaudio.Sound("door", 0.1, 2)]
        with tempfile.TemporaryDirectory() as folder, patch.object(ezaudio, "CONTEXT_SECONDS", 0.2):
            result = self.bench._run(models, 0.2, sounds, args, 1, Path(folder), "case")
            self.assertEqual(sorted(path.name for path in Path(folder).iterdir()), ["case_0.wav", "case_1.wav"])
        self.assertEqual(result["clip_seconds"], 0.2)
        self.assertAlmostEqual(result["compute_per_clip_second"], result["compute_seconds"] / 0.2)
        self.assertEqual(set(result["stages"]), set(self.bench.LOADS + self.bench.COMPUTE))
        self.assertEqual(result["stages"]["sample"]["peak_reserved_gib"], 0.0)
        self.assertEqual(result["memory"], {})


@unittest.skipUnless(HAVE_BUNDLE, "the EzAudio bundle is not installed")
class RealFileTests(unittest.TestCase):
    """The official files: every tensor finds its place, sizes agree with the YAML, the bundle is complete."""

    def test_bundle_is_complete_and_its_configuration_supported(self):
        from engine.models.ezaudio import config
        from engine.pipelines import ezaudio

        files = ezaudio.EzAudioFiles.from_bundle(BUNDLE)
        self.assertEqual(files.missing(), [])
        settings = config.load(files.config)
        self.assertEqual((settings.window_frames, settings.latent_rate, settings.sample_rate), (500, 50, 24000))

    def test_transformer_and_decoder_load_every_tensor(self):
        import torch

        from engine.models.ezaudio import config, transformer, vae

        settings = config.load(BUNDLE / "ezaudio-xl.yml")
        model = transformer.load(BUNDLE / "ezaudio_s3_xl.pt", "cpu", torch.bfloat16)
        config.check_sizes(settings, model.sizes)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 874_760_256)  # + 36 rotary frequencies
        del model
        decoder = vae.load(BUNDLE / "vae" / "1m.pt", "cpu")
        self.assertEqual(sum(p.numel() for p in decoder.parameters()), 21_980_032)

    def test_text_encoder_loads_from_the_first_shard_alone(self):
        import torch

        from engine.models.ezaudio import text

        folder = BUNDLE / "text_encoder"
        self.assertEqual([path.name for path in text.shard_files(folder)], ["model-00001-of-00002.safetensors"])
        encoder = text.load_encoder(folder, "cpu", torch.bfloat16)
        ids, mask = text.tokenize(text.load_tokenizer(folder), ["a dog barking in the distance", ""], 100)
        hidden = text.encode(encoder, ids, mask)
        self.assertEqual(tuple(hidden.shape), (2, 100, 2048))
        self.assertTrue(math.isfinite(float(hidden.abs().sum())))


if __name__ == "__main__":
    unittest.main()
