# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The product picks the FP8 image-to-video experts per expert: by default ("matching") each installed one whose file
holds the distillation of the LoRA it replaces (LightX2V's low-noise file and the high-noise file made at
installation; LightX2V's own high-noise FP8 file, another distillation, is refused), with engine.fp8_experts "off"
none. The Full set's GGUF experts and LoRAs run otherwise (the Lite set, text to video, a LoRA strength other than the
merged one), and the engine gets its attention setting and a lasting kernel cache."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import config, engine_runtime


class FastExpertTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.files = {}
        for name in ("high_fp8.safetensors", "low_fp8.safetensors", "lightx2v_high.safetensors", "high.gguf",
                     "low.gguf", "lora_high.safetensors", "lora_low.safetensors", "vae.safetensors", "index.json",
                     "rife.pkl"):
            (self.root / name).write_bytes(b"x")
            self.files[name] = self.root / name
        self.installed = {(engine_runtime.FAST_I2V_ID, "wan_i2v_fp8_local", "high_noise"):
                          self.files["high_fp8.safetensors"],
                          (engine_runtime.FAST_I2V_ID, "wan_i2v_fp8", "low_noise"): self.files["low_fp8.safetensors"]}
        self.tier = "full"
        # what engine.models.wan22.transformer.lora_merged_share finds in the converted high-noise file, LightX2V's
        # low-noise file and LightX2V's high-noise file (another distillation)
        self.shares = {"high_fp8.safetensors": 1.0, "low_fp8.safetensors": 0.966, "lightx2v_high.safetensors": 0.48}
        self.share = Mock(side_effect=lambda merged, base, lora: self.shares[Path(merged).name])
        patches = (patch.object(engine_runtime.model_manager, "model_path", side_effect=self.model_path),
                   patch.object(engine_runtime.model_manager, "tier", side_effect=lambda: self.tier),
                   patch("engine.models.wan22.transformer.lora_merged_share", self.share),
                   patch.dict(engine_runtime._MERGE_CHECKS, clear=True),
                   patch.dict(config.SETTINGS["engine"], {"fp8_experts": "matching"}),
                   patch("sys.stderr"))
        for target in patches:
            target.start()
            self.addCleanup(target.stop)

    def model_path(self, model_id, dest, *fragments, tier_name=None):
        if model_id == engine_runtime.FAST_I2V_ID:
            return self.installed.get((model_id, dest, *fragments))
        wan = {("diffusion_models", "highnoise"): "high.gguf", ("diffusion_models", "lownoise"): "low.gguf",
               ("loras", "high_noise"): "lora_high.safetensors", ("loras", "low_noise"): "lora_low.safetensors",
               ("wan_vae",): "vae.safetensors", ("wan_text_encoder", "index"): "index.json"}
        name = wan.get((dest, *fragments))
        return self.files[name] if name else None

    def test_by_default_each_fp8_expert_that_holds_its_loras_distillation_replaces_its_gguf_expert(self):
        self.assertEqual(engine_runtime.fast_i2v_experts(), {"high": self.files["high_fp8.safetensors"],
                                                             "low": self.files["low_fp8.safetensors"]})
        self.assertEqual({Path(call.args[0]).name: (Path(call.args[1]).name, Path(call.args[2]).name)
                          for call in self.share.call_args_list},
                         {"high_fp8.safetensors": ("high.gguf", "lora_high.safetensors"),
                          "low_fp8.safetensors": ("low.gguf", "lora_low.safetensors")})
        engine_runtime.fast_i2v_experts()
        self.assertEqual(self.share.call_count, 2)  # checked once per file version
        (self.root / "low_fp8.safetensors").write_bytes(b"replaced")
        engine_runtime.fast_i2v_experts()
        self.assertEqual(self.share.call_count, 3)

    def test_a_file_with_another_distillation_is_refused_and_off_takes_none(self):
        self.installed[(engine_runtime.FAST_I2V_ID, "wan_i2v_fp8_local", "high_noise")] = \
            self.files["lightx2v_high.safetensors"]
        self.assertEqual(engine_runtime.fast_i2v_experts(), {"low": self.files["low_fp8.safetensors"]})
        with patch.dict(config.SETTINGS["engine"], {"fp8_experts": "off"}):
            self.assertEqual(engine_runtime.fast_i2v_experts(), {})

    def test_the_lite_set_another_strength_or_missing_files_keep_the_gguf_experts(self):
        self.assertEqual(engine_runtime.fast_i2v_experts(0.7), {})  # a merged distillation is strength 1.0
        self.tier = "lite"
        self.assertEqual(engine_runtime.fast_i2v_experts(), {})
        self.tier = "full"
        self.installed.clear()
        self.assertEqual(engine_runtime.fast_i2v_experts(), {})

    def test_a_file_that_cannot_be_checked_is_not_used(self):
        self.share.side_effect = ValueError("not a Wan expert")
        self.assertEqual(engine_runtime.fast_i2v_experts(), {})

    def part(self, kind: str):
        import torch

        frames = torch.zeros(5, 3, 16, 16)
        runtime = SimpleNamespace(video=Mock(return_value=frames), interpolate=Mock(return_value=frames))
        with patch.object(engine_runtime, "_rife_weights", return_value=self.files["rife.pkl"]), \
                patch.object(engine_runtime, "_encode_frames"):
            engine_runtime.video_part(runtime, kind, None, self.root / "part.mp4", 16, 30, prompt="p", width=16,
                                      height=16, frame_count=5, last_index=4, seed=1, steps=4, shift=5.0, split=2,
                                      lora_strength=1.0)
        files = runtime.video.call_args.args[0]
        return files, runtime.video.call_args.kwargs["loras"]

    def test_image_to_video_runs_both_fp8_experts_without_loras(self):
        files, loras = self.part("i2v")
        self.assertEqual((files.high, files.low), (self.files["high_fp8.safetensors"], self.files["low_fp8.safetensors"]))
        self.assertEqual((files.vae, files.text_encoder, files.kind),
                         (self.files["vae.safetensors"], self.files["index.json"], "i2v"))
        self.assertIsNone(loras)

    def test_without_the_high_noise_fp8_file_its_gguf_expert_runs_with_its_lora(self):
        del self.installed[(engine_runtime.FAST_I2V_ID, "wan_i2v_fp8_local", "high_noise")]
        files, loras = self.part("i2v")
        self.assertEqual((files.high, files.low), (self.files["high.gguf"], self.files["low_fp8.safetensors"]))
        self.assertEqual(loras, {"high": [(self.files["lora_high.safetensors"], 1.0)]})

    def test_without_the_fp8_files_and_for_text_to_video_the_gguf_experts_and_loras_run(self):
        for kind in ("t2v", "i2v"):
            if kind == "i2v":
                self.installed.clear()
            with self.subTest(kind=kind):
                files, loras = self.part(kind)
                self.assertEqual((files.high.name, files.low.name), ("high.gguf", "low.gguf"))
                self.assertEqual(loras, {"high": [(self.files["lora_high.safetensors"], 1.0)],
                                         "low": [(self.files["lora_low.safetensors"], 1.0)]})


class EngineSettingTests(unittest.TestCase):
    def test_the_engine_gets_the_attention_setting_and_a_kernel_cache_that_lasts(self):
        with patch.object(engine_runtime, "_ENGINE", None), \
                patch.dict(config.SETTINGS["engine"], {"attention": "sdpa"}), \
                patch.dict(os.environ, {}, clear=False) as environment:
            environment.pop("TRITON_CACHE_DIR", None)
            runtime = engine_runtime.engine()
            self.assertEqual(runtime.attention, "sdpa")
            self.assertEqual(Path(environment["TRITON_CACHE_DIR"]), engine_runtime.TRITON_CACHE_DIR)
        self.assertEqual(engine_runtime.TRITON_CACHE_DIR.parent.parent, config.MODELS_DIR)

    def test_a_cache_folder_the_user_chose_is_kept(self):
        from engine import kernels

        with patch.dict(os.environ, {"TRITON_CACHE_DIR": "D:/my-cache"}):
            self.assertEqual(kernels.use_cache_dir("elsewhere"), Path("D:/my-cache"))
            self.assertEqual(os.environ["TRITON_CACHE_DIR"], "D:/my-cache")


if __name__ == "__main__":
    unittest.main()
