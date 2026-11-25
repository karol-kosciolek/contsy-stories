# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""LoRA key mapping against the real files (the LightX2V and Lightning accelerators) and the hook-based
application against merged weights."""
import copy
import gc
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

from app.config import destination_dir

LORAS = destination_dir("loras")
HAVE = all(importlib.util.find_spec(name) for name in ("torch", "safetensors"))
HAVE_MODELS = HAVE and importlib.util.find_spec("diffusers") is not None

# LoRA file -> config of the network it is for.
REAL_FILES = (
    [(f"wan2.2_{task}_A14b_{noise}_noise_lora_rank64_lightx2v_4step_{day}.safetensors", f"wan22_{task}_transformer")
     for task, day in (("t2v", "1217"), ("i2v", "1022")) for noise in ("high", "low")]
    + [("Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "qwen_edit_transformer")])


def header_keys(path: Path) -> list:
    with open(path, "rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        return [k for k in json.loads(file.read(size)) if k != "__metadata__"]


@unittest.skipUnless(HAVE_MODELS, "torch, diffusers and safetensors are not installed")
class RealFileTests(unittest.TestCase):
    def test_every_key_maps_to_exactly_one_matching_module(self):
        from engine.formats.lora import attach, load_lora
        from tests.test_engine_configs import meta_model

        models = {}
        checked = 0
        for name, config in REAL_FILES:
            path = LORAS / name
            if not path.exists():
                continue
            with self.subTest(file=name):
                model = models.setdefault(config, meta_model(config))
                lora = load_lora(path, [n for n, _ in model.named_modules()])
                keys = header_keys(path)
                alphas = sum(k.endswith(".alpha") for k in keys)
                changes = sum(len(d) for d in lora.deltas.values()) + len(lora.modulations)
                # Nothing is left over: every key is a matrix, an alpha, a change or a known skip.
                self.assertEqual(len(keys), 2 * len(lora.modules) + alphas + changes + len(lora.skipped))
                for key in lora.skipped:
                    self.assertRegex(key, r"k_img|v_img|img_emb")
                self.assertTrue(lora.modules)
                # Features and parameter shapes fit the network; attach checks that without hooks left over.
                attach(model, [(lora, 1.0)])()
                self.assertFalse(any(m._forward_hooks for m in model.modules()))
                checked += 1
            del lora
            gc.collect()
        if not checked:
            self.skipTest("no LoRA files on this machine")

    def test_known_layouts(self):
        from engine.formats.lora import load_lora
        from tests.test_engine_configs import meta_model

        def load(name, config):
            if not (LORAS / name).exists():
                self.skipTest(f"{name} is not on this machine")
            return load_lora(LORAS / name, [n for n, _ in meta_model(config).named_modules()])

        qwen = load("Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "qwen_edit_transformer")
        weights = qwen.modules["transformer_blocks.5.img_mlp.net.2"]
        self.assertEqual((weights.alpha, weights.scale), (8.0, 8.0 / 64))
        self.assertEqual(qwen.skipped, ())
        del qwen
        gc.collect()

        wan = load("wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors", "wan22_t2v_transformer")
        self.assertEqual(len(wan.modules), 400)
        self.assertIn("blocks.7.attn2.to_out.0", wan.modules)  # cross_attn.o
        self.assertIn("blocks.7.ffn.net.0.proj", wan.modules)  # ffn.0
        self.assertEqual((wan.deltas, wan.modulations, wan.skipped), ({}, {}, ()))
        del wan
        gc.collect()

        i2v = load("wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors", "wan22_i2v_transformer")
        self.assertIn("condition_embedder.time_proj", i2v.modules)  # time_projection.1
        self.assertEqual(set(i2v.deltas["blocks.0.attn1.to_q"]), {"bias"})
        self.assertEqual(set(i2v.deltas["blocks.0.norm2"]), {"weight", "bias"})  # norm3
        self.assertEqual(set(i2v.deltas["proj_out"]), {"weight", "bias"})  # head.head
        self.assertEqual(set(i2v.deltas["patch_embedding"]), {"weight", "bias"})
        self.assertTrue(i2v.skipped)  # the Wan 2.1 image branch
        del i2v
        gc.collect()

        high = load("wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors", "wan22_i2v_transformer")
        self.assertEqual(len(high.modulations), 41)  # the 40 blocks and the output head
        self.assertEqual(tuple(high.modulations["blocks.0"].shape), (1, 6, 5120))
        self.assertEqual(tuple(high.modulations[""].shape), (1, 2, 5120))


@unittest.skipUnless(HAVE, "torch and safetensors are not installed")
class SyntheticTests(unittest.TestCase):
    def setUp(self):
        import torch
        from torch import nn

        torch.manual_seed(0)
        self.torch = torch
        self.model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).double()
        self.x = torch.randn(5, 8, dtype=torch.double)

    def random_lora(self, rank=3, alpha=None, names=("0", "2")):
        from engine.formats.lora import LoRA, LoRAWeights

        modules = dict(self.model.named_modules())
        torch = self.torch
        return LoRA({n: LoRAWeights(torch.randn(rank, modules[n].in_features, dtype=torch.double),
                                    torch.randn(modules[n].out_features, rank, dtype=torch.double), alpha)
                     for n in names})

    def merged(self, loras):
        """The reference: a copy (taken before any hooks exist) with strength * scale * up @ down merged in."""
        model = copy.deepcopy(self.model)
        modules = dict(model.named_modules())
        for lora, strength in loras:
            for name, w in lora.modules.items():
                modules[name].weight.data += strength * w.scale * (w.up @ w.down)
        return model

    def test_matches_merged_weights_and_leaves_weights_alone(self):
        from engine.formats.lora import attach

        lora = self.random_lora()
        before = copy.deepcopy(self.model.state_dict())
        base = self.model(self.x)
        expected = self.merged([(lora, 0.7)])(self.x)
        remove = attach(self.model, [(lora, 0.7)])
        self.torch.testing.assert_close(self.model(self.x), expected)
        for key, value in self.model.state_dict().items():
            self.assertTrue(self.torch.equal(value, before[key]))
        remove()
        self.assertTrue(self.torch.equal(self.model(self.x), base))

    def test_strength_scales_the_effect(self):
        from engine.formats.lora import attach

        lora = self.random_lora(names=("2",))  # last layer: the effect is exactly linear in strength
        base = self.model(self.x)
        outputs = {}
        for strength in (0.0, 0.5, 1.0, 2.0):
            remove = attach(self.model, [(lora, strength)])
            outputs[strength] = self.model(self.x) - base
            remove()
        self.assertTrue(self.torch.equal(outputs[0.0], self.torch.zeros_like(base)))
        self.torch.testing.assert_close(outputs[0.5] * 2, outputs[1.0])
        self.torch.testing.assert_close(outputs[2.0], outputs[1.0] * 2)

    def test_two_loras_stack(self):
        from engine.formats.lora import attach

        first, second = self.random_lora(rank=2), self.random_lora(rank=5)
        expected = self.merged([(first, 1.0), (second, 0.4)])(self.x)
        remove = attach(self.model, [(first, 1.0), (second, 0.4)])
        self.torch.testing.assert_close(self.model(self.x), expected)
        remove()

    def test_alpha_over_rank_scaling(self):
        from engine.formats.lora import attach

        lora = self.random_lora(rank=4, alpha=2.0)
        self.assertEqual(lora.modules["0"].scale, 0.5)
        expected = self.merged([(lora, 1.0)])(self.x)
        remove = attach(self.model, [(lora, 1.0)])
        self.torch.testing.assert_close(self.model(self.x), expected)
        remove()

    def test_mismatched_lora_attaches_nothing(self):
        from engine.formats.lora import LoRA, LoRAKeyError, LoRAWeights, attach

        torch = self.torch
        good = self.random_lora(names=("0",))
        wrong_size = LoRA({"2": LoRAWeights(torch.zeros(2, 15, dtype=torch.double), torch.zeros(4, 2, dtype=torch.double))})
        with self.assertRaisesRegex(ValueError, "fits a 15 -> 4"):
            attach(self.model, [(good, 1.0), (wrong_size, 1.0)])
        with self.assertRaises(LoRAKeyError):
            attach(self.model, [(LoRA({"nope": good.modules["0"]}), 1.0)])
        with self.assertRaises(TypeError):
            attach(self.model, [(LoRA({"1": good.modules["0"]}), 1.0)])  # the ReLU
        self.assertFalse(any(m._forward_hooks for m in self.model.modules()))

    def test_parameter_changes_match_modified_parameters(self):
        """The `.diff` / `.diff_b` keys: bias, weight, norm and convolution changes, exactly."""
        from torch import nn

        from engine.formats.lora import LoRA, attach

        torch = self.torch
        double = {"dtype": torch.double}
        net = nn.ModuleDict({"linear": nn.Linear(6, 5), "layer_norm": nn.LayerNorm(6), "rms": nn.RMSNorm(6, eps=1e-6),
                             "conv": nn.Conv3d(2, 4, (1, 2, 2), stride=(1, 2, 2))}).double()
        x = torch.randn(3, 6, **double)
        video = torch.randn(1, 2, 2, 4, 4, **double)
        deltas = {"linear": {"weight": torch.randn(5, 6, **double), "bias": torch.randn(5, **double)},
                  "layer_norm": {"weight": torch.randn(6, **double), "bias": torch.randn(6, **double)},
                  "rms": {"weight": torch.randn(6, **double)},
                  "conv": {"weight": torch.randn(4, 2, 1, 2, 2, **double), "bias": torch.randn(4, **double)}}
        strength = 0.6
        expected = copy.deepcopy(net)
        for name, change in deltas.items():
            for param, value in change.items():
                getattr(expected[name], param).data += strength * value
        calls = {name: (lambda m, data=data: m(data)) for name, data in
                 (("linear", x), ("layer_norm", x), ("rms", x), ("conv", video))}
        before = {name: calls[name](net[name]) for name in net}
        remove = attach(net, [(LoRA(deltas=deltas), strength)])
        for name in net:
            with self.subTest(module=name):
                torch.testing.assert_close(calls[name](net[name]), calls[name](expected[name]))
        remove()
        for name in net:
            self.assertTrue(torch.equal(calls[name](net[name]), before[name]))


@unittest.skipUnless(HAVE, "torch and safetensors are not installed")
class SyntheticFileTests(unittest.TestCase):
    NAMES = ["blocks.0", "blocks.0.attn1.to_q", "blocks.0.attn1.to_out.0", "blocks.0.ffn.net.0.proj", "blocks.0.norm2",
             "condition_embedder.time_proj", "layers.1.attention.to_q"]

    def write(self, tensors: dict):
        import torch
        from safetensors.torch import save_file

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "lora.safetensors"
        save_file({k: torch.ones(*shape) if shape else torch.tensor(1.0) for k, shape in tensors.items()}, str(path))
        return path

    def test_every_key_style_reaches_its_module(self):
        from engine.formats.lora import load_lora

        path = self.write({
            "diffusion_model.layers.1.attention.to_q.lora_down.weight": (4, 8),
            "diffusion_model.layers.1.attention.to_q.lora_up.weight": (8, 4),
            "diffusion_model.blocks.0.self_attn.q.lora_down.weight": (4, 8),  # original Wan names
            "diffusion_model.blocks.0.self_attn.q.lora_up.weight": (8, 4),
            "diffusion_model.blocks.0.self_attn.o.lora_down.weight": (4, 8),
            "diffusion_model.blocks.0.self_attn.o.lora_up.weight": (8, 4),
            "blocks.0.ffn.0.lora_down.weight": (4, 8),  # no prefix, with alpha
            "blocks.0.ffn.0.lora_up.weight": (16, 4),
            "blocks.0.ffn.0.alpha": (),
            "diffusion_model.time_projection.1.lora_down.weight": (4, 8),
            "diffusion_model.time_projection.1.lora_up.weight": (48, 4),
            "diffusion_model.blocks.0.norm3.diff_b": (8,),
            "diffusion_model.blocks.0.diff_m": (1, 6, 8),
            "diffusion_model.blocks.0.cross_attn.k_img.diff_b": (8,),
        })
        lora = load_lora(path, self.NAMES)
        self.assertEqual(set(lora.modules), {"layers.1.attention.to_q", "blocks.0.attn1.to_q", "blocks.0.attn1.to_out.0",
                                             "blocks.0.ffn.net.0.proj", "condition_embedder.time_proj"})
        self.assertEqual(lora.modules["blocks.0.ffn.net.0.proj"].scale, 1.0 / 4)
        self.assertIsNone(lora.modules["blocks.0.attn1.to_q"].alpha)
        self.assertEqual(set(lora.deltas), {"blocks.0.norm2"})
        self.assertEqual(set(lora.modulations), {"blocks.0"})
        self.assertEqual(lora.skipped, ("diffusion_model.blocks.0.cross_attn.k_img.diff_b",))

    def test_a_block_modulation_change_is_added_in_place_and_restored(self):
        """`.diff_m` changes the block's scale_shift_table by strength * change (LightX2V's loader does the same)
        while the LoRA is attached; removing the LoRA gives back the exact original table."""
        import torch
        from engine.formats.lora import attach, load_lora
        from torch import nn

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale_shift_table = nn.Parameter(torch.randn(1, 6, 8))
                self.proj = nn.Linear(8, 8)

            def forward(self, x):
                return self.proj(x) * (1 + self.scale_shift_table[0, 1]) + self.scale_shift_table[0, 0]

        torch.manual_seed(3)
        model = nn.ModuleDict({"blocks": nn.ModuleList([Block()])})
        path = self.write({"diffusion_model.blocks.0.diff_m": (1, 6, 8)})
        lora = load_lora(path, [n for n, _ in model.named_modules()])
        table = model["blocks"][0].scale_shift_table
        before = table.detach().clone()
        remove = attach(model, [(lora, 0.5), (lora, 0.25)])
        torch.testing.assert_close(table.detach(), before + 0.75)
        remove()
        self.assertTrue(torch.equal(table.detach(), before))
        bad = self.write({"diffusion_model.blocks.0.diff_m": (1, 6, 4)})
        with self.assertRaises(ValueError):
            attach(model, [(load_lora(bad, [n for n, _ in model.named_modules()]), 1.0)])
        self.assertTrue(torch.equal(table.detach(), before))

    def test_unmappable_keys_raise_a_clear_error(self):
        from engine.formats.lora import LoRAKeyError, load_lora

        path = self.write({
            "diffusion_model.layers.1.attention.to_q.lora_down.weight": (4, 8),
            "diffusion_model.layers.1.attention.to_q.lora_up.weight": (8, 4),
            "diffusion_model.layers.1.attention.qkv.lora_down.weight": (4, 8),  # fused qkv: not supported
            "diffusion_model.layers.1.attention.qkv.lora_up.weight": (24, 4),
            "lora_unet_something.lora_down.weight": (4, 8),
            "diffusion_model.layers.1.attention.to_q.dora_scale": (8,),
        })
        with self.assertRaises(LoRAKeyError) as caught:
            load_lora(path, self.NAMES)
        message = str(caught.exception)
        self.assertIn("lora.safetensors: 4 keys fit no module", message)
        self.assertIn("attention.qkv", message)

    def test_the_style_loras_peft_keys_are_down_and_up_with_a_scale_of_one(self):
        """The style LoRAs of the animated films (app/styles.py) use PEFT's `lora_A` / `lora_B`, behind
        `diffusion_model.` (Z-Image) or with `.default` (Qwen-Image-Edit), without an alpha."""
        from engine.formats.lora import LoRAKeyError, load_lora

        for a, b in (("diffusion_model.layers.1.attention.to_q.lora_A.weight",
                      "diffusion_model.layers.1.attention.to_q.lora_B.weight"),
                     ("layers.1.attention.to_q.lora_A.default.weight", "layers.1.attention.to_q.lora_B.default.weight")):
            with self.subTest(key=a):
                lora = load_lora(self.write({a: (4, 8), b: (8, 4)}), self.NAMES)
                weights = lora.modules["layers.1.attention.to_q"]
                self.assertEqual((tuple(weights.down.shape), tuple(weights.up.shape), weights.scale), ((4, 8), (8, 4), 1.0))
        with self.assertRaisesRegex(LoRAKeyError, "2 keys fit no module"):
            load_lora(self.write({"layers.7.attention.to_q.lora_A.weight": (4, 8),
                                  "layers.7.attention.to_q.lora_B.weight": (8, 4)}), self.NAMES)

    def test_incomplete_pair_is_an_error(self):
        from engine.formats.lora import LoRAKeyError, load_lora

        path = self.write({"layers.1.attention.to_q.lora_down.weight": (4, 8)})
        with self.assertRaisesRegex(LoRAKeyError, "only one of the two"):
            load_lora(path, self.NAMES)


if __name__ == "__main__":
    unittest.main()
