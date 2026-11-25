# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Wan's VAE and UMT5 encoder from Wan-AI's own files (Wan-AI/Wan2.2-T2V-A14B-Diffusers: the float32 VAE with
diffusers names, the bfloat16 encoder shards with their index, tokenizer/spiece.model).

On tiny networks: a diffusers-named VAE file and an original-layout file load to the same weights and pictures; a
sharded encoder loads exactly like one file holding the same tensors, and finds its tokenizer. On this machine's
real files (skipped when they are missing): the shapes fit the shipped configs, and the official VAE rounded to
bfloat16 is bit-identical to the single-file VAE used before. No real network is loaded whole."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from app.config import CATALOG, MODELS_DIR, destination_dir
from tests import seeded

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
HAVE = all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers", "safetensors", "transformers"))
OFFICIAL = MODELS_DIR / "official" / "Wan2.2-T2V-A14B-Diffusers"


def official(dest: str, folder: str) -> Path:
    """The catalog's folder for an official Wan file, or where it is downloaded before the catalog names it."""
    return destination_dir(dest) if dest in CATALOG["destinations"] else OFFICIAL / folder


OFFICIAL_VAE = official("wan_vae", "vae") / "diffusion_pytorch_model.safetensors"
OFFICIAL_INDEX = official("wan_text_encoder", "text_encoder") / "model.safetensors.index.json"
OLD_VAE = MODELS_DIR / "vae" / "wan_2.1_vae.safetensors"
TINY_UMT5 = dict(vocab_size=300, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2, num_heads=4,
                 relative_attention_num_buckets=32, relative_attention_max_distance=128, dense_act_fn="gelu_new",
                 feed_forward_proj="gated-gelu", is_gated_act=True, tie_word_embeddings=False)


@unittest.skipUnless(HAVE and (FIXTURES / "engine_wan_vae.json").is_file(), "torch, diffusers or the fixture is missing")
class VaeLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from safetensors.torch import save_file

        from engine.models.wan22 import vae

        data = json.loads((FIXTURES / "engine_wan_vae.json").read_text(encoding="utf-8"))
        tiny = data["config"]
        cls.config = {"base_dim": tiny["dim"], "z_dim": tiny["z_dim"], "dim_mult": tiny["dim_mult"],
                      "num_res_blocks": tiny["num_res_blocks"], "attn_scales": tiny["attn_scales"],
                      "temperal_downsample": tiny["temperal_downsample"], "dropout": 0.0,
                      "latents_mean": [0.0] * 16, "latents_std": [1.0] * 16, "is_residual": False,
                      "in_channels": 3, "out_channels": 3, "patch_size": None, "scale_factor_temporal": 4,
                      "scale_factor_spatial": 8}
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        original = {k: v.contiguous() for k, v in seeded.seeded_state_dict(data["shapes"]).items()}
        cls.original = Path(cls.tmp.name) / "original.safetensors"
        cls.diffusers = Path(cls.tmp.name) / "diffusion_pytorch_model.safetensors"
        save_file(original, str(cls.original))
        save_file({k: v.contiguous() for k, v in vae.diffusers_tensors(original.items())}, str(cls.diffusers))

    def test_the_layout_is_recognised_from_the_names(self):
        from engine.models.wan22 import vae

        self.assertTrue(vae.diffusers_layout(self.diffusers))
        self.assertFalse(vae.diffusers_layout(self.original))

    def test_both_layouts_give_the_same_weights_and_pictures(self):
        import torch

        from engine.models.wan22 import vae

        a = vae.load_vae(self.original, "cpu", torch.bfloat16, config=self.config)
        b = vae.load_vae(self.diffusers, "cpu", torch.bfloat16, config=self.config)
        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(set(sa), set(sb))
        for name in sa:
            self.assertTrue(torch.equal(sa[name], sb[name]), name)
        latent = seeded.seeded_inputs([1, 16, 2, 4, 4], 11)
        self.assertTrue(torch.equal(vae.decode(a, latent), vae.decode(b, latent)))

    def test_a_file_with_other_names_is_refused(self):
        import torch
        from safetensors.torch import save_file

        from engine.models.wan22 import vae

        path = Path(self.tmp.name) / "other.safetensors"
        save_file({"something.weight": torch.zeros(2)}, str(path))
        with self.assertRaises(ValueError):
            vae.load_vae(path, "cpu", torch.float32, config=self.config)


def write_tiny_umt5(folder: Path, shards: int = 2):
    """A tiny UMT5 as one file and as a sharded set (index + shards in `folder`/text_encoder, tokenizer beside)."""
    import torch
    from safetensors.torch import save_file
    from transformers import UMT5Config, UMT5EncoderModel

    torch.manual_seed(5)
    model = UMT5EncoderModel(UMT5Config(**TINY_UMT5))
    state = {name: tensor.detach().clone().to(torch.bfloat16).contiguous() for name, tensor in model.state_dict().items()
             if name != "encoder.embed_tokens.weight"}
    single = folder / "umt5.safetensors"
    save_file(state, str(single))
    encoder_dir = folder / "text_encoder"
    encoder_dir.mkdir()
    names = sorted(state)
    weight_map = {}
    for part in range(shards):
        shard = f"model-{part + 1:05d}-of-{shards:05d}.safetensors"
        chunk = names[part::shards]
        save_file({name: state[name] for name in chunk}, str(encoder_dir / shard))
        weight_map.update({name: shard for name in chunk})
    index = encoder_dir / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    return single, index


@unittest.skipUnless(HAVE, "torch, transformers or safetensors is missing")
class ShardedEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.folder = Path(cls.tmp.name)
        cls.single, cls.index = write_tiny_umt5(cls.folder)
        (cls.folder / "tokenizer").mkdir()
        cls.spiece = cls.folder / "tokenizer" / "spiece.model"
        cls.spiece.write_bytes(b"sentencepiece model bytes")

    def test_the_sharded_set_loads_exactly_like_one_file(self):
        import torch

        from engine.models.wan22 import text

        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                one = text.load_text_encoder(self.single, "cpu", dtype, config=dict(TINY_UMT5))
                sharded = text.load_text_encoder(self.index, "cpu", dtype, config=dict(TINY_UMT5))
                a, b = one.state_dict(), sharded.state_dict()
                self.assertEqual(set(a), set(b))
                for name in a:
                    self.assertEqual(a[name].dtype, b[name].dtype, name)
                    self.assertTrue(torch.equal(a[name], b[name]), name)
                tokens = [5, 17, 99, 42, 8, 1, 0, 0]
                self.assertTrue(torch.equal(text._encode_tokens(one, tokens, torch.device("cpu")),
                                            text._encode_tokens(sharded, tokens, torch.device("cpu"))))
                self.assertEqual(type(sharded.encoder.block[0].layer[0].SelfAttention.q).__name__, "CastLinear")

    def test_the_bf16_shards_stay_bf16_and_small_tensors_float32(self):
        import torch

        from engine.models.wan22 import text

        encoder = text.load_text_encoder(self.index, "cpu", torch.bfloat16, config=dict(TINY_UMT5))
        self.assertEqual(encoder.encoder.block[0].layer[1].DenseReluDense.wo.weight.dtype, torch.bfloat16)
        self.assertEqual(encoder.shared.weight.dtype, torch.bfloat16)
        self.assertEqual(encoder.encoder.block[0].layer[0].layer_norm.weight.dtype, torch.float32)
        self.assertEqual(encoder.encoder.block[1].layer[0].SelfAttention.relative_attention_bias.weight.dtype,
                         torch.float32)

    def test_the_tokenizer_is_found_beside_the_shards_or_in_the_repository_layout(self):
        from engine.models.wan22 import text

        self.assertEqual(text.read_spiece(self.index), self.spiece.read_bytes())
        beside = self.index.parent / "spiece.model"
        beside.write_bytes(b"beside")
        try:
            self.assertEqual(text.read_spiece(self.index), b"beside")  # the file beside the shards wins
        finally:
            beside.unlink()
        self.assertEqual(text.files(self.index), [self.index, *sorted(self.index.parent.glob("model-*.safetensors")),
                                                  self.spiece])
        self.assertEqual(text.files(self.single), [self.single])

    def test_a_missing_tokenizer_is_reported(self):
        from engine.models.wan22 import text

        with tempfile.TemporaryDirectory() as tmp:
            _, index = write_tiny_umt5(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                text.read_spiece(index)

    def test_an_index_naming_a_file_outside_its_folder_is_refused(self):
        from engine.models.wan22 import text

        bad = self.index.parent / "bad.safetensors.index.json"
        for shard in ("../umt5.safetensors", "sub/model.safetensors", "model.bin"):
            with self.subTest(shard=shard):
                bad.write_text(json.dumps({"weight_map": {"shared.weight": shard}}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    text.load_text_encoder(bad, "cpu", config=dict(TINY_UMT5))

    def test_the_index_is_an_accepted_format_and_the_video_pipeline_keys_encodings_by_every_file(self):
        from engine.core.memory import file_identity
        from engine.models.wan22 import text
        from engine.pipelines.video import WanModels

        text.check_format("model.safetensors.index.json")
        models = WanModels.from_files(self.index, "h.gguf", "l.gguf", "v.safetensors")
        self.assertEqual(models.text_id, tuple(file_identity(p) for p in text.files(self.index)))
        self.assertEqual(len(models.text_id), 4)  # the index, two shards, the tokenizer


@unittest.skipUnless(HAVE and OFFICIAL_VAE.is_file(), "the official Wan VAE is not on this machine")
class OfficialVaeFileTests(unittest.TestCase):
    def test_the_official_file_fits_the_shipped_config_without_renaming(self):
        from safetensors import safe_open

        from tests.test_engine_configs import meta_model

        model = meta_model("wan_vae")
        expected = {name: tuple(t.shape) for name, t in model.state_dict().items()}
        with safe_open(str(OFFICIAL_VAE), "pt") as file:
            found = {key: tuple(file.get_slice(key).get_shape()) for key in file.keys()}
            self.assertEqual({file.get_slice(key).get_dtype() for key in file.keys()}, {"F32"})
        self.assertEqual(found, expected)

    @unittest.skipUnless(OLD_VAE.is_file(), "the single-file Wan VAE is not on this machine")
    def test_rounded_to_bfloat16_the_official_tensors_equal_the_single_file_vae(self):
        """The product runs the VAE in bfloat16: the loaded weights, and so the pictures, are the same."""
        import torch

        from engine.models.wan22 import vae

        old = dict(vae.diffusers_tensors(vae._file_tensors(OLD_VAE)))
        new = dict(vae._file_tensors(OFFICIAL_VAE))
        self.assertEqual(set(old), set(new))
        for name, tensor in new.items():
            self.assertEqual(old[name].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(tensor.to(torch.bfloat16).view(torch.int16), old[name].view(torch.int16)), name)


@unittest.skipUnless(HAVE and OFFICIAL_INDEX.is_file(), "the official Wan UMT5 shards are not on this machine")
class OfficialEncoderFileTests(unittest.TestCase):
    def test_the_shards_fit_the_shipped_config(self):
        from safetensors import safe_open

        from engine.models.wan22 import text

        model = text._build()
        expected = {name: tuple(t.shape) for name, t in model.state_dict().items()
                    if name != "encoder.embed_tokens.weight"}  # tied to shared.weight
        found, dtypes = {}, set()
        for shard in text.files(OFFICIAL_INDEX)[1:-1]:
            with safe_open(str(shard), "pt") as file:
                for key in file.keys():
                    found[key] = tuple(file.get_slice(key).get_shape())
                    dtypes.add(file.get_slice(key).get_dtype())
        self.assertEqual(found, expected)
        self.assertEqual(dtypes, {"BF16"})

    def test_the_tokenizer_is_the_official_umt5_model_and_gives_the_pinned_ids(self):
        from engine.models.wan22 import text

        data = json.loads((FIXTURES / "engine_wan_tokens.json").read_text(encoding="utf-8"))
        model = text.read_spiece(OFFICIAL_INDEX)
        self.assertEqual(hashlib.sha256(model).hexdigest(), data["spiece_sha256"])
        sp = text.load_tokenizer(model)
        for case in data["cases"]:
            expected = case["ids"] + [1]
            expected += [0] * max(0, 512 - len(expected))
            self.assertEqual(text.tokenize(sp, case["prompt"]), expected, case["prompt"][:40])


if __name__ == "__main__":
    unittest.main()
