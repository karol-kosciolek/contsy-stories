# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The out-of-memory fallback of the Z-Image VAE must really tile, also for pictures up to 1024 pixels
(diffusers alone tiles only above its config's sample_size, which is 1024)."""
import importlib.util
import unittest
import unittest.mock

HAVE = all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers", "safetensors", "PIL"))


def tiny_vae():
    """The shipped VAE config (sample_size 1024, four blocks) with channel widths small enough for a test."""
    import torch
    from diffusers import AutoencoderKL
    from engine.models.zimage import vae

    config = vae._config(None)
    config.update(block_out_channels=[8, 8, 8, 8], norm_num_groups=4, layers_per_block=1)
    torch.manual_seed(0)
    return AutoencoderKL.from_config(config).eval()


@unittest.skipUnless(HAVE, "torch, diffusers or PIL is not installed")
class TiledDecodeTests(unittest.TestCase):
    def test_the_fallback_tiles_a_1024_pixel_picture_in_512_pixel_tiles(self):
        import torch
        from engine.models.zimage import vae

        model = tiny_vae()
        tile_sizes = []
        model.decoder.register_forward_pre_hook(lambda module, args: tile_sizes.append(tuple(args[0].shape[-2:])))
        with unittest.mock.patch.object(model, "tiled_decode", wraps=model.tiled_decode) as tiled:
            picture = vae.decode(model, torch.randn(1, 16, 128, 128), tiled=True)  # a 1024 x 1024 picture
        self.assertEqual(picture.size, (1024, 1024))
        self.assertEqual(tiled.call_count, 1)  # before the fix: 0, the 'fallback' was the same full decode
        self.assertGreater(len(tile_sizes), 1)
        self.assertLessEqual(max(max(size) for size in tile_sizes), 64)  # latent tiles of 64 = 512 pixels
        self.assertEqual(max(max(size) for size in tile_sizes), 64)

    def test_a_plain_decode_does_not_tile_and_the_settings_come_back(self):
        import torch
        from engine.models.zimage import vae

        model = tiny_vae()
        before = (model.tile_latent_min_size, model.tile_sample_min_size, model.use_tiling)
        with unittest.mock.patch.object(model, "tiled_decode", wraps=model.tiled_decode) as tiled:
            vae.decode(model, torch.randn(1, 16, 16, 16))
            self.assertEqual(tiled.call_count, 0)
            vae.decode(model, torch.randn(1, 16, 96, 96), tiled=True)  # 768 pixels: above one 512-pixel tile
            self.assertEqual(tiled.call_count, 1)
        self.assertEqual((model.tile_latent_min_size, model.tile_sample_min_size, model.use_tiling), before)

    def test_the_settings_come_back_also_when_the_decode_fails(self):
        import torch
        from engine.models.zimage import vae

        model = tiny_vae()
        before = (model.tile_latent_min_size, model.tile_sample_min_size, model.use_tiling)
        with unittest.mock.patch.object(model, "tiled_decode", side_effect=torch.OutOfMemoryError("boom")):
            with self.assertRaises(torch.OutOfMemoryError):
                vae.decode(model, torch.randn(1, 16, 96, 96), tiled=True)
        self.assertEqual((model.tile_latent_min_size, model.tile_sample_min_size, model.use_tiling), before)


if __name__ == "__main__":
    unittest.main()
