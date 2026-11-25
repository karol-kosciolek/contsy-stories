# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Wan 2.2 pipeline on tiny random networks behind its loader seams: shapes, determinism, the split
between the two experts, guidance passes, LoRA per expert, cancellation, and the order in which networks
are loaded and released. Nothing here needs a GPU or a real model."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAVE = all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers", "transformers", "safetensors"))


def tiny_models(events=None, calls=None):
    """Random tiny networks: UMT5, two Wan transformers (36 input channels), the Wan VAE."""
    import torch
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from engine.pipelines.video import WanModels
    from transformers import UMT5Config, UMT5EncoderModel

    events = events if events is not None else []
    calls = calls if calls is not None else {}

    class Words:
        def encode(self, word):
            return [ord(c) % 250 + 2 for c in word.strip()] if word.strip() else []

    def text(device, dtype):
        events.append("load text")
        torch.manual_seed(1)
        config = UMT5Config(vocab_size=300, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=4,
                            relative_attention_num_buckets=32, dense_act_fn="gelu_new", feed_forward_proj="gated-gelu")
        return Words(), UMT5EncoderModel(config).to(device=device, dtype=dtype).eval()

    def transformer(expert, device, dtype):
        events.append(f"load {expert}")
        torch.manual_seed(2 if expert == "high" else 3)
        model = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16, in_channels=36,
                                      out_channels=16, text_dim=32, freq_dim=32, ffn_dim=64, num_layers=1,
                                      cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=None,
                                      added_kv_proj_dim=None, rope_max_seq_len=64, pos_embed_seq_len=None)
        for parameter in model.parameters():
            if parameter.ndim > 1:
                torch.nn.init.normal_(parameter, std=0.05)
        model.register_forward_hook(lambda *args, e=expert: calls.__setitem__(e, calls.get(e, 0) + 1))
        return model.to(device=device, dtype=dtype).eval()

    def autoencoder(device, dtype):
        events.append("load vae")
        torch.manual_seed(4)
        return AutoencoderKLWan(base_dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1, attn_scales=[],
                                temperal_downsample=[False, True, True], latents_mean=[0.0] * 16,
                                latents_std=[1.0] * 16).to(device=device, dtype=dtype).eval()

    return WanModels(text, transformer, autoencoder)


def torch_ones(value):
    import torch

    return torch.ones(2) * value


def picture(value=128):
    from PIL import Image

    return Image.new("RGB", (48, 40), (value, 40, 200 - value // 2))


@unittest.skipUnless(HAVE, "torch, diffusers or transformers are not installed")
class VideoPipelineTests(unittest.TestCase):
    def run_clip(self, models=None, seed=5, steps=4, start=None, prompt="a cat walks", **kwargs):
        import torch
        from engine.pipelines import video

        return video.generate(models or tiny_models(), prompt, 32, 32, 5, seed, steps, 5.0,
                              start_image=start, device="cpu", dtype=torch.float32, **kwargs)

    def test_shape_range_and_determinism_per_seed(self):
        first, again, other = self.run_clip(), self.run_clip(), self.run_clip(seed=6)
        self.assertEqual(tuple(first.shape), (5, 3, 32, 32))
        self.assertTrue(0.0 <= float(first.min()) and float(first.max()) <= 1.0)
        self.assertTrue((first == again).all())
        self.assertFalse((first == other).all())

    def test_the_start_picture_conditions_the_clip(self):
        plain, with_a, with_b = self.run_clip(), self.run_clip(start=picture(30)), self.run_clip(start=picture(220))
        self.assertFalse((plain == with_a).all())
        self.assertFalse((with_a == with_b).all())

    def test_each_expert_runs_its_share_of_the_steps_with_one_pass_per_step(self):
        events, calls = [], {}
        self.run_clip(tiny_models(events, calls), steps=6)  # no guidance: one pass per step, split at 3
        self.assertEqual(calls, {"high": 3, "low": 3})
        self.assertEqual(events, ["load text", "load high", "load low", "load vae"])  # one after the other
        calls.clear()
        self.run_clip(tiny_models([], calls), steps=6, split=2)
        self.assertEqual(calls, {"high": 2, "low": 4})

    def test_the_vae_is_loaded_for_the_start_picture_and_again_for_decoding(self):
        events = []
        self.run_clip(tiny_models(events), start=picture())
        self.assertEqual(events, ["load text", "load vae", "load high", "load low", "load vae"])

    def test_progress_is_monotonic_and_reports_every_step(self):
        seen = []
        self.run_clip(progress_cb=lambda fraction, message: seen.append((fraction, message)), steps=4)
        self.assertEqual([f for f, _ in seen], sorted(f for f, _ in seen))
        self.assertEqual(seen[-1], (1.0, "Done"))
        self.assertEqual([m for _, m in seen if m.startswith("Step")], ["Step 1/4", "Step 2/4", "Step 3/4"])

    def test_invalid_arguments(self):
        import torch
        from engine.pipelines import video

        for size, length, steps in ((((30, 32)), 5, 4), ((32, 32), 6, 4), ((32, 32), 5, 1)):
            with self.assertRaises(ValueError):
                video.generate(tiny_models(), "x", size[0], size[1], length, 1, steps, 5.0, device="cpu",
                               dtype=torch.float32)

    def write_lora(self, path, strength_seed=7):
        import torch
        from safetensors.torch import save_file

        generator = torch.Generator().manual_seed(strength_seed)
        save_file({"diffusion_model.blocks.0.self_attn.q.lora_down.weight": torch.randn(4, 32, generator=generator) * 0.3,
                   "diffusion_model.blocks.0.self_attn.q.lora_up.weight": torch.randn(32, 4, generator=generator) * 0.3,
                   "diffusion_model.blocks.0.ffn.0.lora_down.weight": torch.randn(4, 32, generator=generator) * 0.3,
                   "diffusion_model.blocks.0.ffn.0.lora_up.weight": torch.randn(64, 4, generator=generator) * 0.3},
                  str(path))

    def test_a_lora_for_one_expert_changes_the_clip_and_leaves_no_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = Path(tmp) / "high.safetensors"
            self.write_lora(lora)
            plain = self.run_clip()
            high_only = self.run_clip(loras={"high": [(lora, 1.0)]})
            low_only = self.run_clip(loras={"low": [(lora, 1.0)]})
            both = self.run_clip(loras={"high": [(lora, 1.0)], "low": [(lora, 1.0)]})
        for clip in (high_only, low_only, both):
            self.assertFalse((plain == clip).all())
        self.assertFalse((high_only == low_only).all())
        self.assertTrue((plain == self.run_clip()).all())  # the networks are rebuilt per call: no state leaks

    def test_streamed_experts_give_the_same_clip_and_every_block_travels(self):
        copies = []
        plain = self.run_clip(steps=4)
        streamed = self.run_clip(steps=4, stream={"prefetch": 1, "pin": False,
                                                  "transfer": lambda t: copies.append(t.numel()) or t.clone()})
        self.assertTrue((plain == streamed).all())
        self.assertGreater(len(copies), 0)  # blocks were copied (a one-block tiny model: once per step and expert at least)


    def test_a_bad_request_is_refused_before_any_network_is_loaded(self):
        import dataclasses

        events = []
        for steps, split in ((6, 0), (6, 6), (6, 9)):  # the low-noise expert needs at least one step, the high one too
            with self.assertRaises(ValueError, msg=f"split {split} of {steps}"):
                self.run_clip(tiny_models(events), steps=steps, split=split)
        t2v = dataclasses.replace(tiny_models(events), kind="t2v")
        with self.assertRaises(ValueError):  # a start picture needs the image-to-video networks
            self.run_clip(t2v, start=picture())
        self.assertEqual(events, [])  # nothing was loaded for any of them
        self.run_clip(t2v)  # a text-to-video request on those networks is fine

    def test_the_first_experts_streamer_and_blocks_are_gone_when_the_second_expert_loads(self):
        import dataclasses
        import gc
        import weakref

        models = tiny_models()
        first, alive_at_second_load = [], []

        def load(expert, device, dtype):
            if expert == "low":
                gc.collect()
                alive_at_second_load.append(first[0]() is not None)
            network = models.load_transformer(expert, device, dtype)
            if expert == "high":
                first.append(weakref.ref(network.blocks[0]))
            return network

        self.run_clip(dataclasses.replace(models, load_transformer=load), steps=4, stream={"prefetch": 1, "pin": False})
        self.assertEqual(alive_at_second_load, [False])  # the pinned host copies of the high-noise expert are freed first

    def test_the_pinned_host_cache_is_released_after_the_experts_only_when_streaming(self):
        from unittest.mock import patch

        from engine.pipelines import video

        with patch.object(video.streaming, "release_host_cache") as release:
            self.run_clip(steps=4, stream={"prefetch": 1, "pin": False})
            self.assertEqual(release.call_count, 1)
            self.run_clip(steps=4)
            self.assertEqual(release.call_count, 1)  # without streaming nothing was pinned

    def test_no_expert_and_no_streamer_is_alive_when_the_clip_is_decoded(self):
        import dataclasses
        import gc
        import weakref

        models, experts, alive = tiny_models(), [], []

        def load_transformer(expert, device, dtype):
            network = models.load_transformer(expert, device, dtype)
            experts.append(weakref.ref(network.blocks[0]))
            return network

        def load_vae(device, dtype):  # called for the decoding: both experts must be gone by now
            gc.collect()
            alive.append([ref() is not None for ref in experts])
            return models.load_vae(device, dtype)

        wrapped = dataclasses.replace(models, load_transformer=load_transformer, load_vae=load_vae)
        self.run_clip(wrapped, steps=4, stream={"prefetch": 1, "pin": False})
        self.assertEqual(alive, [[False, False]])  # the last expert's bound `detach` used to keep it alive through the decoding

    def test_cancelling_between_steps_stops_the_run_and_releases_the_networks(self):
        from engine.core.cancel import CancelToken, Cancelled

        token, events = CancelToken(), []

        def on_progress(fraction, message):
            if message == "Step 2/6":
                token.cancel()

        with self.assertRaises(Cancelled):
            self.run_clip(tiny_models(events), steps=6, cancel=token, progress_cb=on_progress)
        self.assertEqual(events, ["load text", "load high"])  # the low-noise expert and the VAE were never loaded

    def test_stage_statistics_name_every_network(self):
        stages = []
        self.run_clip(stats=lambda name, seconds, peak: stages.append(name), start=picture())
        self.assertEqual(stages, ["text encoder", "vae encode", "high noise expert", "low noise expert", "vae"])

    def test_a_prompt_encoded_before_skips_the_text_encoder_and_gives_the_same_clip(self):
        """The next part of a clip continues with the same prompt: its encoding is kept (EmbeddingCache), so the
        text encoder is not loaded again, and the clip is exactly the one a fresh encoding gives."""
        import dataclasses

        from engine.core.memory import EmbeddingCache

        events, kept = [], EmbeddingCache()
        models = dataclasses.replace(tiny_models(events), text_id=("umt5.safetensors", 1, 2))
        fresh = self.run_clip(models, start=picture())
        first = self.run_clip(models, start=picture(), embeddings=kept)
        again = self.run_clip(models, start=picture(30), embeddings=kept)  # the next part: another start frame
        self.assertEqual(events.count("load text"), 2)  # the plain run and the first cached one
        self.assertTrue((first == fresh).all())
        self.assertFalse((again == first).all())
        self.assertTrue((again == self.run_clip(models, start=picture(30))).all())
        events.clear()
        self.run_clip(models, embeddings=kept, prompt="a dog runs")  # another prompt is encoded
        self.assertEqual(events[0], "load text")
        events.clear()
        self.run_clip(models, embeddings=kept, prompt="a bird sings")  # encoded once, then reused
        self.run_clip(models, embeddings=kept, prompt="a bird sings")
        self.assertEqual(events.count("load text"), 1)
        other_file = dataclasses.replace(models, text_id=("umt5.safetensors", 1, 3))  # a replaced encoder file
        events.clear()
        self.run_clip(other_file, embeddings=kept)
        self.assertEqual(events[0], "load text")
        events.clear()
        self.run_clip(tiny_models(events), embeddings=kept)  # networks without a file identity: never kept
        self.run_clip(tiny_models(events), embeddings=kept)
        self.assertEqual(events.count("load text"), 2)

    def test_expected_prompts_are_encoded_in_one_load_exactly_as_alone(self):
        """A Timeline job announces its scenes' prompts (EmbeddingCache.expect): the first scene's encoder load
        encodes them all, each exactly as it would be encoded alone, and the later scenes skip the encoder."""
        import dataclasses
        from unittest.mock import patch

        from engine.core.memory import EmbeddingCache
        from engine.pipelines import video

        events = []
        models = dataclasses.replace(tiny_models(events), text_id=("umt5.safetensors", 1, 2))
        prompts = ("a cat walks", "a dog runs", "a bird sings")
        fresh = {prompt: self.run_clip(models, prompt=prompt) for prompt in prompts}
        for streamed in (False, True):
            with self.subTest(streamed=streamed), patch.object(video, "_blocks_fit", return_value=True):
                kept = EmbeddingCache(entries=1)  # room for the announced prompts is kept besides the entries
                kept.expect(prompts[1:] + ("a dog runs", ""))
                self.assertEqual(kept.expected(), prompts[1:])
                events.clear()
                clips = {prompt: self.run_clip(models, prompt=prompt, embeddings=kept, stream_text=streamed)
                         for prompt in prompts}
                self.assertEqual(events.count("load text"), 1)
                for prompt in prompts:
                    self.assertTrue((clips[prompt] == fresh[prompt]).all(), prompt)
                self.assertEqual(kept.expected(), ())  # each was served: none is waited for any more
                kept.expect(prompts)
                kept.clear()  # memory given back: the announcement stays until the job ends it
                self.assertEqual(kept.expected(), prompts)
                kept.expect(())
        kept = EmbeddingCache(entries=1)
        kept.expect(prompts)
        for index, prompt in enumerate(prompts):
            kept.put(("id", prompt), torch_ones(index))
        self.assertTrue(all(("id", prompt) in kept for prompt in prompts))
        kept.expect(())  # the job ended: back to one entry
        self.assertEqual([("id", prompt) in kept for prompt in prompts], [False, False, True])

    def test_the_embedding_cache_keeps_the_most_recent_prompts(self):
        import torch

        from engine.core.memory import EmbeddingCache

        kept = EmbeddingCache(entries=2)
        for name in "abc":
            kept.put(name, torch.ones(2, device="cpu") * ord(name))
        self.assertIsNone(kept.get("a"))
        self.assertEqual(float(kept.get("b")[0]), ord("b"))
        kept.put("d", torch.zeros(1))  # "b" was used last: "c" goes
        self.assertIsNone(kept.get("c"))
        self.assertIsNotNone(kept.get("b"))
        kept.clear()
        self.assertIsNone(kept.get("b"))


if __name__ == "__main__":
    unittest.main()
