# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""InfiniteTalk in the engine (engine/models/infinitetalk, engine/pipelines/talk.py): the speech features and their
windows, the loudness, the audio projection's grouping, the FP8 layouts, the CLIP extraction, and the whole pipeline
on tiny random networks on the CPU (no model file, no GPU)."""
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

HAVE = all(importlib.util.find_spec(name) is not None
           for name in ("torch", "diffusers", "transformers", "safetensors", "scipy"))


@unittest.skipUnless(HAVE, "needs torch, diffusers, transformers, safetensors and scipy")
class SpeechTests(unittest.TestCase):
    def test_a_full_scale_1khz_sine_measures_minus_3_lufs_and_is_brought_to_minus_23(self):
        import numpy as np
        from engine.models.infinitetalk import audio

        t = np.arange(5 * audio.RATE) / audio.RATE
        sine = np.sin(2 * np.pi * 1000 * t)
        self.assertAlmostEqual(audio.integrated_loudness(sine), -3.01, delta=0.1)  # BS.1770: -3.01 (the 16 kHz filters)
        self.assertAlmostEqual(audio.integrated_loudness(audio.normalize_loudness(sine)), -23.0, delta=0.05)
        silence = np.zeros(audio.RATE, dtype=np.float32)
        self.assertIs(audio.normalize_loudness(silence).dtype, np.dtype(np.float32))
        self.assertTrue(np.array_equal(audio.normalize_loudness(silence), silence))  # nothing to measure: unchanged
        self.assertEqual(audio.integrated_loudness(sine[:100]), float("-inf"))

    def test_the_features_are_every_layers_output_per_frame_as_the_full_model_gives_them(self):
        """[frames, layers, hidden]: at the encoder's own frame count the same numbers as transformers' hidden_states[1:]
        of the whole model, and any other count a resampling of the convolutional features (a tiny random wav2vec 2.0)."""
        import numpy as np
        import torch
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        from engine.models.infinitetalk import audio

        torch.manual_seed(0)
        config = Wav2Vec2Config(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64,
                                conv_dim=(16, 16), conv_stride=(5, 4), conv_kernel=(10, 8), num_conv_pos_embeddings=8,
                                num_conv_pos_embedding_groups=4, feat_extract_norm="group", do_stable_layer_norm=False)
        encoder = Wav2Vec2Model(config).eval()
        speech = np.random.default_rng(0).standard_normal(4000).astype(np.float32) * 0.1
        with torch.inference_mode():
            natural = encoder.feature_extractor(torch.from_numpy(speech)[None]).shape[-1]
            reference = torch.stack(encoder(torch.from_numpy(speech)[None], output_hidden_states=True).hidden_states[1:],
                                    dim=2)[0]
        found = audio.features(encoder, speech, natural)
        self.assertEqual(tuple(found.shape), (natural, 3, 32))
        self.assertTrue(torch.allclose(found, reference, atol=1e-5))
        self.assertEqual(tuple(audio.features(encoder, speech, 25).shape), (25, 3, 32))

    def test_every_frame_gets_the_window_around_it_and_a_short_sequence_is_mirrored_on(self):
        import torch
        from engine.models.infinitetalk import audio

        feats = torch.arange(6, dtype=torch.float32)[:, None, None].expand(6, 12, 768)
        window = audio.windows(feats, 0, 3)
        self.assertEqual(tuple(window.shape), (1, 3, 5, 12, 768))
        self.assertEqual(window[0, :, :, 0, 0].tolist(), [[0, 0, 0, 1, 2], [0, 0, 1, 2, 3], [0, 1, 2, 3, 4]])
        self.assertEqual(audio.windows(feats, 4, 2)[0, :, :, 0, 0].tolist(), [[2, 3, 4, 5, 5], [3, 4, 5, 5, 5]])
        longer = audio.extend(feats, 10)
        self.assertGreaterEqual(longer.shape[0], 10)
        self.assertEqual(longer[6:9, 0, 0].tolist(), [5, 4, 3])  # the reference's flip of the tail


@unittest.skipUnless(HAVE, "needs torch, diffusers, transformers, safetensors and scipy")
class NetworkPartsTests(unittest.TestCase):
    def test_the_audio_windows_are_grouped_as_the_reference_rearranges_them(self):
        import torch
        from engine.models.infinitetalk.model import group_windows

        windows = torch.arange(9 * 5, dtype=torch.float32).view(1, 9, 5, 1, 1).expand(1, 9, 5, 12, 2)
        first, later = group_windows(windows)
        self.assertEqual((tuple(first.shape), tuple(later.shape)), ((1, 1, 5, 12, 2), (1, 2, 8, 12, 2)))
        value = later[0, :, :, 0, 0].tolist()
        # frames 1-4: frame 1's window up to its centre, the centres of frames 2 and 3, frame 4 from its centre
        self.assertEqual(value[0], [5, 6, 7, 12, 17, 22, 23, 24])
        self.assertEqual(value[1], [25, 26, 27, 32, 37, 42, 43, 44])

    def test_rows_keep_their_values_within_fp8_precision(self):
        import torch
        from engine.models.infinitetalk.convert import quantize_rows

        torch.manual_seed(0)
        weight = torch.randn(64, 32) * torch.logspace(-3, 0, 64)[:, None]
        fp8, scale = quantize_rows(weight)
        self.assertEqual((fp8.dtype, tuple(scale.shape)), (torch.float8_e4m3fn, (64, 1)))
        error = ((fp8.float() * scale - weight).abs() / weight.abs().amax(dim=1, keepdim=True)).max()
        self.assertLess(float(error), 1 / 16)  # 3 mantissa bits, per row of its own scale

    def test_fp8_weights_stay_fp8_with_a_scale_per_row_whatever_the_file_stores(self):
        import torch
        from safetensors.torch import save_file
        from engine.formats.fp8 import ScaledFP8, iter_kept_fp8

        weight = torch.ones(4, 3).to(torch.float8_e4m3fn)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mixed.safetensors"
            save_file({"a.weight": weight.clone(), "a.scale_weight": torch.tensor([0.5]), "b.weight": weight.clone(),
                       "b.weight_scale": torch.tensor([[1.0], [2.0], [3.0], [4.0]]), "c.weight": weight.clone(),
                       "c.bias": torch.zeros(4), "scaled_fp8": torch.zeros(2).to(torch.float8_e4m3fn)}, str(path))
            found = dict(iter_kept_fp8(path))
        self.assertEqual(sorted(found), ["a.weight", "b.weight", "c.bias", "c.weight"])
        self.assertTrue(all(isinstance(found[name], ScaledFP8) for name in ("a.weight", "b.weight", "c.weight")))
        self.assertEqual(found["a.weight"].scale.tolist(), [0.5] * 4)
        self.assertEqual(found["b.weight"].dequantize(torch.float32)[:, 0].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(found["c.weight"].scale.tolist(), [1.0] * 4)

    def test_the_clip_extraction_keeps_the_31_blocks_the_encoder_reads(self):
        import torch
        from engine.models.infinitetalk import clip

        state = {"visual.head": torch.zeros(2), "visual.post_norm.weight": torch.zeros(2),
                 "visual.transformer.31.mlp.0.weight": torch.zeros(2), "visual.transformer.30.mlp.0.weight": torch.zeros(2),
                 "visual.pos_embedding": torch.zeros(2), "textual.token_embedding.weight": torch.zeros(2)}
        kept = clip.visual_tensors(state)
        self.assertEqual(sorted(kept), ["pos_embedding", "transformer.30.mlp.0.weight"])
        self.assertTrue(all(tensor.dtype == torch.float16 for tensor in kept.values()))
        prepared = clip.prepare(torch.zeros(1, 3, 48, 64))  # mid gray in the VAE's range
        self.assertEqual(tuple(prepared.shape), (1, 3, 224, 224))
        self.assertAlmostEqual(float(prepared[0, 0, 0, 0]), (0.5 - clip.MEAN[0]) / clip.STD[0], places=5)

    def test_chunks_cover_the_clip_with_their_overlap(self):
        from engine.pipelines.talk import plan

        self.assertEqual(plan(50), [53])  # a short clip: one chunk of 4k + 1 frames
        self.assertEqual(plan(81), [81])
        self.assertEqual(plan(100), [57, 53])  # a 4 s clip: 57 + 48 new frames, the last chunk no longer than needed
        self.assertEqual(plan(125), [69, 65])  # 5 s
        self.assertEqual(plan(150), [81, 81])
        self.assertEqual(plan(225), [81, 81, 81])
        self.assertEqual(plan(1), [5])
        for frames in range(82, 400, 7):  # every plan covers its clip with chunks of at most 81 frames
            lengths = plan(frames)
            self.assertTrue(all(length <= 81 and (length - 1) % 4 == 0 for length in lengths))
            made = lengths[0] + sum(length - 9 for length in lengths[1:])
            self.assertGreaterEqual(made, frames)
            self.assertLess(made - frames, 5)  # no more than the 4k + 1 rounding of the last chunk

def tiny_talk_models(calls):
    """Random tiny networks for the pipeline: UMT5, the Wan 2.1 I2V transformer with the audio adapter, the Wan VAE,
    a CLIP stand-in and speech features."""
    import torch
    from diffusers import AutoencoderKLWan
    from transformers import UMT5Config, UMT5EncoderModel

    from engine.models.infinitetalk import model as talk_model
    from engine.models.wan22 import attention as wan_attention
    from engine.pipelines.talk import TalkModels

    class Words:
        def encode(self, word):
            return [ord(c) % 250 + 2 for c in word.strip()] if word.strip() else []

    def text(device, dtype):
        torch.manual_seed(1)
        config = UMT5Config(vocab_size=300, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=4,
                            relative_attention_num_buckets=32, dense_act_fn="gelu_new", feed_forward_proj="gated-gelu")
        return Words(), UMT5EncoderModel(config).to(device=device, dtype=dtype).eval()

    def transformer(device, dtype):
        torch.manual_seed(2)
        model = talk_model.build({"patch_size": [1, 2, 2], "num_attention_heads": 2, "attention_head_dim": 16,
                                  "in_channels": 36, "out_channels": 16, "text_dim": 32, "freq_dim": 32, "ffn_dim": 64,
                                  "num_layers": 1, "cross_attn_norm": True, "qk_norm": "rms_norm_across_heads",
                                  "eps": 1e-6, "image_dim": 1280, "added_kv_proj_dim": 32, "rope_max_seq_len": 64})
        model.to_empty(device="cpu")
        for name, parameter in model.named_parameters():
            torch.nn.init.normal_(parameter, std=0.02) if parameter.ndim > 1 else torch.nn.init.ones_(parameter)
        talk_model._rebuild_rope(model, "cpu")
        wan_attention.install(model)
        model.register_forward_hook(lambda *args: calls.__setitem__("forward", calls.get("forward", 0) + 1))
        return model.to(device=device, dtype=dtype).eval()

    def autoencoder(device, dtype):
        torch.manual_seed(4)
        return AutoencoderKLWan(base_dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1, attn_scales=[],
                                temperal_downsample=[False, True, True], latents_mean=[0.0] * 16,
                                latents_std=[1.0] * 16).to(device=device, dtype=dtype).eval()

    class Clip(torch.nn.Module):
        def forward(self, pictures):
            calls["clip"] = tuple(pictures.shape)
            return torch.ones(pictures.shape[0], 257, 1280)

    def speech_features(speech, frames):
        calls["spoken"] = frames
        torch.manual_seed(5)
        return torch.randn(frames, 12, 768)

    return TalkModels(text, transformer, autoencoder, lambda device, dtype: Clip(), speech_features)


@unittest.skipUnless(HAVE, "needs torch, diffusers, transformers, safetensors and scipy")
class TalkPipelineTests(unittest.TestCase):
    def test_a_clip_longer_than_a_chunk_is_made_in_chunks_at_25_fps(self):
        import numpy as np
        import torch
        from PIL import Image
        from engine.pipelines import talk

        calls, messages = {}, []
        speech = np.zeros(int(3.6 * 16000), dtype=np.float32)
        frames = talk.generate(tiny_talk_models(calls), "A cartoon fox talks", Image.new("RGB", (80, 60), "orange"),
                               speech, 90, 64, 48, seed=3, steps=2, device="cpu", dtype=torch.float32,
                               progress_cb=lambda fraction, message: messages.append(message))
        self.assertEqual(tuple(frames.shape), (90, 3, 48, 64))
        self.assertTrue(0.0 <= float(frames.min()) and float(frames.max()) <= 1.0)
        self.assertEqual(calls["forward"], 2 * 2)  # two chunks of two steps, one pass each (no audio guidance)
        self.assertEqual(calls["spoken"], 90)       # 3.6 s at 25 FPS
        self.assertEqual(calls["clip"], (1, 3, 224, 224))
        self.assertIn("Chunk 2/2, step 2/2", messages)
        calls.clear()
        guided = talk.generate(tiny_talk_models(calls), "A cartoon fox talks", Image.new("RGB", (80, 60), "orange"),
                               speech, 20, 64, 48, seed=3, steps=2, audio_scale=2.0, device="cpu", dtype=torch.float32)
        self.assertEqual(tuple(guided.shape), (20, 3, 48, 64))
        self.assertEqual(calls["forward"], 2 * 2)  # one chunk, the speech and the silence per step
        with self.assertRaisesRegex(ValueError, "multiples of 16"):
            talk.generate(tiny_talk_models({}), "x", Image.new("RGB", (8, 8)), speech, 10, 60, 48, seed=1,
                          device="cpu")

    def test_the_block_hears_only_its_own_frames_audio(self):
        """Changing the audio tokens of the second latent frame leaves the first frame's output unchanged."""
        import torch
        from engine.models.infinitetalk.model import AudioCrossAttention

        torch.manual_seed(0)
        layer = AudioCrossAttention(dim=32, context_dim=768, heads=2)
        x = torch.randn(1, 2 * 6, 32)
        audio = torch.randn(2, 32, 768)
        changed = audio.clone()
        changed[1] += 1.0
        a, b = layer(x, audio), layer(x, changed)
        self.assertTrue(torch.allclose(a[:, :6], b[:, :6]))
        self.assertFalse(torch.allclose(a[:, 6:], b[:, 6:]))


if __name__ == "__main__":
    unittest.main()
