# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Triton kernels on a real GPU against their PyTorch definitions: FP8 row quantization and output scaling, the
FP8 linear layer, Wan's rotary embedding and SageAttention. They need CUDA, Triton and a few hundred MB of GPU memory,
so they run only when asked for: set CONTSY_GPU_TESTS=1 (with the live Studio's queue paused). On the CPU, the rotary
embedding's dispatch is checked: it never reaches for Triton there."""
import importlib.util
import os
import unittest

HAVE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("diffusers") is not None


def _gpu() -> bool:
    if os.environ.get("CONTSY_GPU_TESTS") != "1" or not HAVE:
        return False
    import torch
    from engine import kernels

    return torch.cuda.is_available() and kernels.available("cuda")


@unittest.skipUnless(HAVE, "torch and diffusers are not installed")
class RotaryDispatchTests(unittest.TestCase):
    def test_on_the_cpu_the_rotary_embedding_is_pytorchs_even_with_sage_asked_for(self):
        import torch
        from engine.core import attention
        from engine.kernels import rope
        from engine.models.wan22 import attention as wan_attention

        hidden = torch.randn(1, 6, 2, 8)
        cos, sin = torch.randn(1, 6, 1, 8), torch.randn(1, 6, 1, 8)
        self.assertFalse(rope.fits(hidden, cos, sin))
        expected = wan_attention._rotate(hidden, cos, sin)
        x1, x2 = hidden[..., 0::2], hidden[..., 1::2]
        torch.testing.assert_close(expected[..., 0::2], x1 * cos[..., 0::2] - x2 * sin[..., 1::2])
        torch.testing.assert_close(expected[..., 1::2], x1 * sin[..., 1::2] + x2 * cos[..., 0::2])
        with attention.use("sage"):
            torch.testing.assert_close(wan_attention._rotate(hidden, cos, sin), expected, rtol=0, atol=0)


@unittest.skipUnless(HAVE, "torch and diffusers are not installed")
class FusedBlockDispatchTests(unittest.TestCase):
    def test_on_the_cpu_the_fused_block_is_diffusers_forward_and_the_fallbacks_are_its_arithmetic(self):
        import torch
        import torch.nn.functional as F
        from diffusers import WanTransformer3DModel
        from engine.core import attention
        from engine.kernels import fp8, norms
        from engine.models.wan22 import block as fused_block

        torch.manual_seed(0)
        model = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16,
                                      in_channels=16, out_channels=16, text_dim=32, freq_dim=32, ffn_dim=64,
                                      num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6,
                                      rope_max_seq_len=64).to(torch.bfloat16).eval()
        latent = torch.randn(1, 16, 2, 8, 8, dtype=torch.bfloat16)
        context = torch.randn(1, 7, 32, dtype=torch.bfloat16)
        t = torch.tensor([500.0])
        with torch.inference_mode(), attention.use("sage"):
            expected = model(latent, t, context, return_dict=False)[0]
            self.assertEqual(fused_block.install(model), 2)
            self.assertEqual(fused_block.install(model), 0)  # already installed
            got = model(latent, t, context, return_dict=False)[0]
        self.assertTrue(torch.equal(got, expected))
        x = torch.randn(5, 32, dtype=torch.bfloat16)
        y = torch.randn(5, 32, dtype=torch.bfloat16)
        gate, scale, shift = torch.randn(3, 1, 1, 32).unbind(0)
        h, n = norms.add_norm(x, y, gate, norm=norms.MODULATED, a=scale, b=shift)
        self.assertTrue(torch.equal(h, (x.float() + y * gate).type_as(x)[0]))
        self.assertTrue(torch.equal(n, (F.layer_norm(h.float(), (32,), eps=1e-6) * (1 + scale) + shift).type_as(x)[0]))
        out = torch.randn(4, 32, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            fp8.scale_output_(out, torch.ones(4), torch.ones(32), activation="relu")


def tiny_vae(dtype, device="cpu"):
    import torch
    from diffusers import AutoencoderKLWan

    torch.manual_seed(4)
    model = AutoencoderKLWan(base_dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1, attn_scales=[],
                             temperal_downsample=[False, True, True], latents_mean=[0.0] * 16, latents_std=[1.0] * 16)
    for name, parameter in model.named_parameters():
        if name.endswith("gamma"):
            torch.nn.init.uniform_(parameter, 0.5, 1.5)
    return model.to(device=device, dtype=dtype).eval()


@unittest.skipUnless(HAVE, "torch and diffusers are not installed")
class FusedVaeDispatchTests(unittest.TestCase):
    def test_on_the_cpu_the_fused_vae_norms_give_diffusers_pictures(self):
        import torch
        from engine.core import attention
        from engine.models.wan22 import vae

        model = tiny_vae(torch.bfloat16)
        frames = torch.rand(5, 3, 32, 48)
        latent = torch.randn(1, 16, 2, 4, 6)
        with attention.use("sage"):
            expected = vae.encode(model, frames), vae.decode(model, latent)
            from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock

            residual = sum(isinstance(module, WanResidualBlock) for module in model.modules())
            self.assertEqual(vae.fuse_norms(model), 2 * residual + 2)  # two norms per residual block, two heads
            self.assertEqual(vae.fuse_norms(model), 0)  # the SiLUs are already taken over
            self.assertGreater(vae.fuse_convs(model), 10)
            self.assertEqual(vae.fuse_convs(model), 0)
            got = vae.encode(model, frames), vae.decode(model, latent)
        self.assertTrue(torch.equal(got[0], expected[0]))
        self.assertTrue(torch.equal(got[1], expected[1]))

    def test_encoding_part_of_a_sequence_repeats_its_last_latent_frame(self):
        import torch
        from engine.models.wan22 import vae

        model = tiny_vae(torch.float32)
        frames = torch.full((13, 3, 32, 48), 0.5)
        frames[0] = torch.rand(3, 32, 48)
        full = vae.encode(model, frames)  # 4 latent frames
        part = vae.encode(model, frames, computed=2)
        self.assertEqual(part.shape, full.shape)
        torch.testing.assert_close(part[:, :, :2], full[:, :, :2], rtol=1e-5, atol=1e-6)  # the same causal chunks
        self.assertTrue(torch.equal(part[:, :, 2], part[:, :, 1]) and torch.equal(part[:, :, 3], part[:, :, 1]))
        for whole in (None, 4, 9):
            self.assertTrue(torch.equal(vae.encode(model, frames, computed=whole), full))


@unittest.skipUnless(_gpu(), "set CONTSY_GPU_TESTS=1 on a machine with CUDA and Triton")
class KernelTests(unittest.TestCase):
    def test_row_quantization_matches_pytorch(self):
        import torch
        from engine.kernels import fp8

        x = torch.randn(300, 5120, device="cuda", dtype=torch.bfloat16) * torch.rand(300, 1, device="cuda") * 10
        x[7] = 0
        quantized, scale = fp8.quantize_rows(x)
        expected_q, expected_scale = fp8.quantize_rows(x, use_triton=False)
        torch.testing.assert_close(scale, expected_scale)
        same = (quantized.view(torch.uint8) == expected_q.view(torch.uint8)).float().mean().item()
        self.assertGreater(same, 0.99999)

    def test_output_scaling_matches_float32_arithmetic(self):
        import torch
        from engine.kernels import fp8

        out = torch.randn(257, 3000, device="cuda", dtype=torch.bfloat16)
        rows, cols = torch.rand(257, device="cuda"), torch.rand(3000, device="cuda")
        bias = torch.randn(3000, device="cuda", dtype=torch.bfloat16)
        expected = out.float() * rows[:, None] * cols[None, :] + bias.float()
        got = fp8.scale_output_(out.clone(), rows, cols, bias)
        torch.testing.assert_close(got.float(), expected, rtol=1e-2, atol=1e-2)

    def test_the_fp8_layer_is_close_to_the_expanded_weight(self):
        import torch
        import torch.nn.functional as F
        from engine.core.fp8_linear import FP8Linear, fp8_matmul_available

        if not fp8_matmul_available("cuda"):
            self.skipTest("this GPU has no FP8 tensor cores")
        weight = torch.randn(1024, 512, device="cuda") / 20
        scale = weight.abs().amax(dim=1) / 448
        layer = FP8Linear((weight / scale[:, None]).to(torch.float8_e4m3fn), scale, torch.randn(1024, device="cuda"))
        x = torch.randn(3, 77, 512, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(layer.uses_fp8(x))
        expected = F.linear(x.float(), layer.dequantized(torch.float32), layer.bias.float())
        got = layer(x)
        self.assertEqual((got.shape, got.dtype), (expected.shape, torch.bfloat16))
        self.assertLess(float((got.float() - expected).norm() / expected.norm()), 0.04)

    def test_float16_inputs_stay_finite_and_a_dtype_only_cast_keeps_the_layer_on_the_gpu(self):
        """A layer the size of Wan's feed-forward input: float16 takes the expanded weight (the FP8 product's
        unscaled sums would overflow it), and `.to(torch.bfloat16)` moves nothing to the CPU."""
        import torch
        import torch.nn.functional as F
        from engine.core.fp8_linear import FP8Linear

        torch.manual_seed(0)
        weight = torch.randn(2048, 5120, device="cuda") / 40
        scale = weight.abs().amax(dim=1) / 448
        layer = FP8Linear((weight / scale[:, None]).to(torch.float8_e4m3fn), scale,
                          torch.randn(2048, device="cuda", dtype=torch.bfloat16))
        x = torch.randn(512, 5120, device="cuda")
        expected = F.linear(x, layer.dequantized(torch.float32), layer.bias.float())
        self.assertFalse(layer.uses_fp8(x.half()))
        got = layer(x.half())
        self.assertEqual(got.dtype, torch.float16)
        self.assertTrue(bool(torch.isfinite(got).all()))
        self.assertLess(float((got.float() - expected).norm() / expected.norm()), 0.01)
        layer.to(torch.bfloat16)
        self.assertEqual({layer.weight.device.type, layer.weight_scale.device.type, layer.bias.device.type}, {"cuda"})
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)

    def test_the_rotary_kernel_matches_diffusers_arithmetic(self):
        import torch
        from engine.kernels import rope
        from engine.models.wan22 import attention as wan_attention

        hidden = torch.randn(2, 300, 40, 128, device="cuda", dtype=torch.bfloat16) * 3
        angles = torch.rand(1, 300, 1, 128, device="cuda") * 6.28
        cos, sin = angles.cos(), angles.sin()
        self.assertTrue(rope.fits(hidden, cos, sin))
        expected = wan_attention._rotate(hidden, cos, sin)  # PyTorch: no backend asked for
        got = rope.rotate(hidden, cos, sin)
        torch.testing.assert_close(got.float(), expected.float(), rtol=0.01, atol=1e-3)  # near-zero sums cancel
        self.assertGreater((got == expected).float().mean().item(), 0.999)

    def test_the_fused_residual_adds_and_layer_norms_match_diffusers_arithmetic(self):
        import torch
        from engine.kernels import norms

        torch.manual_seed(0)
        x = (torch.randn(3001, 5120, device="cuda") * torch.rand(3001, 1, device="cuda") * 8).to(torch.bfloat16)
        y = (torch.randn(3001, 5120, device="cuda") * 3).to(torch.bfloat16)
        gate, scale, shift = (torch.randn(1, 1, 5120, device="cuda") * 0.5 for _ in range(3))
        weight = torch.randn(5120, device="cuda").to(torch.bfloat16)
        bias = torch.randn(5120, device="cuda").to(torch.bfloat16)
        cases = {"modulated": dict(norm=norms.MODULATED, a=scale, b=shift),
                 "gated + affine": dict(y=y, gate=gate, norm=norms.AFFINE, a=weight, b=bias),
                 "add + modulated": dict(y=y, norm=norms.MODULATED, a=scale, b=shift),
                 "gated, no norm": dict(y=y, gate=gate, normalise=False),
                 "plain norm": dict(norm=norms.NONE)}
        for name, options in cases.items():
            with self.subTest(name):
                options = dict(options)
                other = options.pop("y", None)
                h, n = norms.add_norm(x, other, options.pop("gate", None), **options)
                expected_h, expected_n = norms.add_norm(x, other, cases[name].get("gate"), use_triton=False, **options)
                self.assertEqual(h.dtype, torch.bfloat16)
                self.assertGreater((h == expected_h).float().mean().item(), 0.999)
                torch.testing.assert_close(h.float(), expected_h.float(), rtol=0.01, atol=0.02)
                if expected_n is None:
                    self.assertIsNone(n)
                    continue
                self.assertGreater((n == expected_n).float().mean().item(), 0.995)
                torch.testing.assert_close(n.float(), expected_n.float(), rtol=0.02, atol=0.05)

    def test_the_gelu_in_the_fp8_output_pass_matches_pytorch(self):
        import torch
        import torch.nn.functional as F
        from engine.kernels import fp8

        out = torch.randn(513, 13824, device="cuda", dtype=torch.bfloat16) * 4
        rows, cols = torch.rand(513, device="cuda"), torch.rand(13824, device="cuda")
        bias = torch.randn(13824, device="cuda", dtype=torch.bfloat16)
        expected = F.gelu(fp8.scale_output_(out.clone(), rows, cols, bias), approximate="tanh")
        got = fp8.scale_output_(out.clone(), rows, cols, bias, activation="gelu_tanh")
        self.assertGreater((got == expected).float().mean().item(), 0.999)
        torch.testing.assert_close(got.float(), expected.float(), rtol=0.01, atol=0.01)
        fallback = fp8.scale_output_(out.clone(), rows, cols, bias, use_triton=False, activation="gelu_tanh")
        # the fallback rounds the scales and each product to bfloat16 (see kernels.fp8): its GELU input can be a
        # bfloat16 step away
        torch.testing.assert_close(fallback.float(), expected.float(), rtol=0.07, atol=0.07)

    def test_the_fused_wan_block_matches_diffusers_forward(self):
        """A Wan 2.2 A14B block at its real width (5120, 40 heads of 128, feed-forward 13824) with an FP8 first
        feed-forward layer: the fused forward against diffusers' forward, both with SageAttention."""
        import torch
        import torch.nn.functional as F
        from diffusers.models.transformers.transformer_wan import WanTransformerBlock
        from engine.core import attention
        from engine.core.fp8_linear import FP8Linear, fp8_matmul_available
        from engine.models.wan22 import attention as wan_attention
        from engine.models.wan22 import block as fused_block

        torch.manual_seed(0)
        plain = WanTransformerBlock(5120, 13824, 40, "rms_norm_across_heads", True, 1e-6).to("cuda", torch.bfloat16)
        for parameter in plain.parameters():
            if parameter.ndim > 1:
                torch.nn.init.normal_(parameter, std=0.02)
        if fp8_matmul_available("cuda"):
            proj = plain.ffn.net[0].proj
            weight = proj.weight.float()
            scale = weight.abs().amax(dim=1) / 448
            plain.ffn.net[0].proj = FP8Linear((weight / scale[:, None]).to(torch.float8_e4m3fn), scale, proj.bias)
        wan_attention.install(plain)
        tokens = 2 * 30 * 53
        x = torch.randn(1, tokens, 5120, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(1, 512, 5120, device="cuda", dtype=torch.bfloat16)
        temb = torch.randn(1, 6, 5120, device="cuda") * 0.1
        angles = torch.rand(1, tokens, 1, 128, device="cuda") * 6.28
        rotary = (angles.cos(), angles.sin())
        with torch.inference_mode(), attention.use("sage"):
            expected = plain(x, context, temb, rotary)
            self.assertEqual(fused_block.install(plain), 1)
            got = plain(x, context, temb, rotary)
            with attention.use("sdpa"):
                reference = plain(x, context, temb, rotary)  # the fused class falls back to diffusers' forward
        self.assertEqual((got.shape, got.dtype), (expected.shape, torch.bfloat16))
        cosine = F.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item()
        self.assertGreater(cosine, 0.99999)
        # a last-bit difference before an FP8 quantization can move a value by one FP8 step, so fewer outputs are
        # bit-identical than in the passes alone (measured 0.64 here, 0.85 at the clip's 39750 tokens)
        self.assertGreater((got == expected).float().mean().item(), 0.5)
        self.assertGreater(F.cosine_similarity(reference.float().flatten(), expected.float().flatten(), dim=0).item(),
                           0.999)

    def test_the_fused_vae_norm_matches_diffusers_rounding_in_both_memory_formats(self):
        import torch
        from engine.kernels import norms

        torch.manual_seed(0)
        for channels in (96, 384, 20):
            x = (torch.randn(1, channels, 4, 60, 106, device="cuda") * 3).to(torch.bfloat16)
            gamma = (torch.rand(channels, 1, 1, 1, device="cuda") + 0.5).to(torch.bfloat16)
            for silu in (False, True):
                expected = norms.rms_channels(x, gamma, channels ** 0.5, silu, use_triton=False)
                for layout in (torch.contiguous_format, torch.channels_last_3d):
                    with self.subTest(channels=channels, silu=silu, layout=layout):
                        source = x.contiguous(memory_format=layout)
                        self.assertTrue(norms.rms_channels_fits(source, gamma))
                        got = norms.rms_channels(source, gamma, channels ** 0.5, silu)
                        self.assertTrue(got.is_contiguous(memory_format=layout))
                        self.assertGreater((got == expected).float().mean().item(), 0.999)
                        torch.testing.assert_close(got.float(), expected.float(), rtol=0.02, atol=0.02)

    def test_the_fused_vae_gives_nearly_the_same_pictures(self):
        import torch
        from engine.core import attention
        from engine.models.wan22 import vae

        model = tiny_vae(torch.bfloat16, "cuda")
        frames = torch.rand(9, 3, 64, 96)
        latent = torch.randn(1, 16, 3, 8, 12)
        with attention.use("sage"):
            expected = vae.encode(model, frames), vae.decode(model, latent)
            vae.fuse_norms(model)
            vae.fuse_convs(model)
            vae.channels_last(model)
            got = vae.encode(model, frames), vae.decode(model, latent)
        for a, b in zip(got, expected):
            mse = float(((a.float() - b.float()) ** 2).mean())
            self.assertLess(mse, 1e-5)

    def test_sage_attention_is_close_to_sdpa(self):
        import torch
        import torch.nn.functional as F
        from engine.core import attention

        torch.manual_seed(0)
        for kv_tokens in (2000, 512):
            q = torch.randn(1, 2000, 8, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, kv_tokens, 8, 128, device="cuda", dtype=torch.bfloat16) + 0.5
            v = torch.randn(1, kv_tokens, 8, 128, device="cuda", dtype=torch.bfloat16)
            self.assertTrue(attention.fits_sage(q, k, v))
            got = attention.sage(q, k, v)
            expected = attention.sdpa(q, k, v)
            cosine = F.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item()
            self.assertGreater(cosine, 0.999, kv_tokens)
            self.assertEqual((got.shape, got.dtype), (q.shape, torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
