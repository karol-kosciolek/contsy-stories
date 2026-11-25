# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The model configs in engine/models/configs must build the networks the shipped weight files fit."""
import importlib.util
import json
import unittest
from pathlib import Path

from app.config import MODELS_DIR

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "engine" / "models" / "configs"
MODELS = MODELS_DIR
HAVE = all(importlib.util.find_spec(name) for name in ("torch", "diffusers", "safetensors", "gguf"))

NAMES = ("zimage_transformer", "zimage_vae", "zimage_text_encoder",
         "wan22_t2v_transformer", "wan22_i2v_transformer", "wan_vae", "wan_text_encoder",
         "qwen_edit_transformer", "qwen_vae", "qwen_vl_text_encoder")

# (config, expected parameter count range in billions)
NETWORKS = (("zimage_transformer", 6.0, 6.3), ("zimage_vae", 0.08, 0.09),
            ("wan22_t2v_transformer", 14.0, 14.6), ("wan22_i2v_transformer", 14.0, 14.6),
            ("wan_vae", 0.12, 0.13), ("qwen_edit_transformer", 20.0, 20.8), ("qwen_vae", 0.12, 0.13))


def load_config(name: str) -> dict:
    """The config without our provenance field, as the network classes want it."""
    config = json.loads((CONFIGS / f"{name}.json").read_text(encoding="utf-8"))
    config.pop("_source")
    return config


def meta_model(name: str):
    """The network of a config on the meta device: shapes and names, no memory."""
    import diffusers
    import torch

    config = load_config(name)
    with torch.device("meta"):
        return getattr(diffusers, config["_class_name"]).from_config(config)


def safetensors_shapes(path: Path) -> dict:
    """Tensor name -> meta tensor of the right shape, read from the header only."""
    import torch
    from safetensors import safe_open

    with safe_open(str(path), "pt") as file:
        return {key: torch.empty(file.get_slice(key).get_shape(), device="meta") for key in file.keys()}


def gguf_shapes(path: Path) -> dict:
    """Same for a GGUF file; GGUF lists dimensions in reverse order."""
    import gguf
    import torch

    return {t.name: torch.empty([int(n) for n in reversed(t.shape)], device="meta")
            for t in gguf.GGUFReader(str(path)).tensors}


def shapes(state: dict) -> dict:
    return {key: tuple(value.shape) for key, value in state.items()}


class ConfigFileTests(unittest.TestCase):
    def test_every_config_records_where_it_came_from(self):
        for name in NAMES:
            with self.subTest(config=name):
                config = json.loads((CONFIGS / f"{name}.json").read_text(encoding="utf-8"))
                source = config["_source"]
                self.assertEqual(set(source), {"repo", "path", "revision", "fetched"})
                self.assertRegex(source["revision"], r"^[0-9a-f]{40}$")
                self.assertTrue(source["path"].endswith("config.json"))

    def test_wan_experts_share_one_config(self):
        # transformer_2 (the low-noise expert) has the same config as transformer in both repos,
        # so one file per task serves both experts; I2V differs from T2V in the input channels.
        t2v = load_config("wan22_t2v_transformer")
        i2v = load_config("wan22_i2v_transformer")
        self.assertEqual((t2v["in_channels"], i2v["in_channels"]), (16, 36))
        self.assertEqual({k for k in t2v if t2v[k] != i2v[k]}, {"in_channels"})


@unittest.skipUnless(HAVE, "torch, diffusers, safetensors and gguf are not installed")
class NetworkTests(unittest.TestCase):
    def test_networks_build_on_the_meta_device_with_plausible_size(self):
        for name, low, high in NETWORKS:
            with self.subTest(config=name):
                count = sum(p.numel() for p in meta_model(name).parameters()) / 1e9
                self.assertGreater(count, low)
                self.assertLess(count, high)

    def test_text_encoder_configs_build(self):
        import torch
        import transformers

        expected = {"zimage_text_encoder": (3.9, 4.1), "wan_text_encoder": (5.6, 5.8),
                    "qwen_vl_text_encoder": (8.2, 8.4)}
        for name, (low, high) in expected.items():
            with self.subTest(config=name):
                config = load_config(name)
                architecture = getattr(transformers, config["architectures"][0])
                with torch.device("meta"):
                    model = architecture(transformers.AutoConfig.for_model(**config))
                count = sum(p.numel() for p in model.parameters()) / 1e9
                self.assertGreater(count, low)
                self.assertLess(count, high)


@unittest.skipUnless(HAVE, "torch, diffusers, safetensors and gguf are not installed")
class CheckpointTests(unittest.TestCase):
    """Names and shapes of the weight files on disk against the diffusers networks, after the
    key conversion that diffusers' own single-file loader applies."""

    def assert_same_layout(self, converted: dict, config: str, ignore: frozenset = frozenset()):
        expected = shapes(meta_model(config).state_dict())
        got = {k: v for k, v in shapes(converted).items() if k not in ignore}
        self.assertEqual(sorted(set(expected) - set(got)), [], "missing from the file")
        self.assertEqual(sorted(set(got) - set(expected)), [], "not in the network")
        self.assertEqual({k for k in expected if expected[k] != got[k]}, set(), "different shapes")

    def need(self, path: Path) -> Path:
        if not path.exists():
            self.skipTest(f"{path.name} is not on this machine")
        return path

    def test_zimage_transformer(self):
        # The bf16 file merged from Tongyi-MAI's shards keeps the official diffusers names.
        disk = safetensors_shapes(self.need(MODELS / "official" / "Z-Image-Turbo" / "transformer_bf16"
                                            / "z-image-turbo-transformer-bf16.safetensors"))
        self.assert_same_layout(disk, "zimage_transformer")

    def test_zimage_transformer_lite(self):
        from diffusers.loaders.single_file_utils import convert_z_image_transformer_checkpoint_to_diffusers

        disk = gguf_shapes(self.need(MODELS / "diffusion_models" / "z-image-turbo-Q8_0.gguf"))
        # The GGUF has a fused attention.qkv and single-file names (x_embedder, final_layer, q_norm).
        self.assertIn("layers.0.attention.qkv.weight", disk)
        self.assert_same_layout(convert_z_image_transformer_checkpoint_to_diffusers(disk),
                                "zimage_transformer")

    def test_zimage_vae(self):
        # Tongyi-MAI's VAE file has the diffusers names.
        disk = safetensors_shapes(self.need(MODELS / "official" / "Z-Image-Turbo" / "vae"
                                            / "diffusion_pytorch_model.safetensors"))
        self.assert_same_layout(disk, "zimage_vae")

    def test_wan_transformers(self):
        from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

        for task, config in (("T2V", "wan22_t2v_transformer"), ("I2V", "wan22_i2v_transformer")):
            for noise in ("HighNoise", "LowNoise"):
                path = MODELS / "diffusion_models" / f"Wan2.2-{task}-A14B-{noise}-Q6_K.gguf"
                if not path.exists():
                    continue
                with self.subTest(file=path.name):
                    disk = gguf_shapes(path)
                    # Original Wan names: blocks.N.self_attn.q, cross_attn.k, ffn.0, modulation, head.head.
                    self.assertIn("blocks.0.self_attn.q.weight", disk)
                    self.assert_same_layout(convert_wan_transformer_to_diffusers(disk), config)
        if not any(MODELS.glob("diffusion_models/Wan2.2-*-A14B-*-Q6_K.gguf")):
            self.skipTest("no Wan GGUF files on this machine")

    def test_qwen_edit_transformer(self):
        disk = gguf_shapes(self.need(MODELS / "diffusion_models" / "qwen-image-edit-2511-Q4_K_M.gguf"))
        # The names are already the diffusers names; the only extra tensor is an empty marker.
        self.assertEqual(shapes({"m": disk["__index_timestep_zero__"]}), {"m": (0,)})
        self.assert_same_layout(disk, "qwen_edit_transformer", ignore=frozenset({"__index_timestep_zero__"}))

    def test_vaes(self):
        """The official Wan and Qwen-Image VAEs the catalog installs have the diffusers names: no renaming."""
        from app import config as app_config

        found = 0
        for model_id, dest, config in (("wan22-a14b-i2v", "wan_vae", "wan_vae"),
                                       ("qwen-image-edit-2511", "qwen_edit_vae", "qwen_vae")):
            spec = next(s for s in app_config.get_model(model_id)["files"] if s["dest"] == dest)
            path = app_config.destination_dir(dest) / (spec.get("save_as") or Path(spec["filename"]).name)
            if not path.is_file():
                continue
            found += 1
            with self.subTest(file=path.name, folder=path.parent.name):
                disk = safetensors_shapes(path)
                self.assertNotIn("conv1.weight", disk)  # not the original Wan names
                self.assert_same_layout(disk, config)
        if not found:
            self.skipTest("no official VAE file on this machine")


if __name__ == "__main__":
    unittest.main()
