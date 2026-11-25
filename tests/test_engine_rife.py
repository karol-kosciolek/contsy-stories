# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""RIFE in the engine: the network keeps its pinned outputs on seeded weights (tests/fixtures/engine_rife.json,
reference values of 2026-11-09 for the RIFE 4.7 architecture), the real weights fit it key for key, and the retimed
interpolation puts every output frame on the output clock."""
import importlib.util
import json
import unittest
from pathlib import Path

from app.config import destination_dir
from tests import seeded

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "engine_rife.json"
# The author's file as the installer extracts it, or the same tensors kept by an earlier installation.
WEIGHTS = next((destination_dir("rife") / name for name in ("flownet_v4.9.pkl", "rife49.pth")
                if (destination_dir("rife") / name).is_file()), destination_dir("rife") / "flownet_v4.9.pkl")
HAVE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAVE, "torch is not installed")
class NetworkTests(unittest.TestCase):
    @unittest.skipUnless(FIXTURE.is_file(), "the fixture is missing")
    def test_the_network_keeps_its_pinned_outputs_on_seeded_weights(self):
        import torch

        from engine.models import rife

        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        net = rife.IFNet()
        net.load_state_dict(seeded.seeded_state_dict(data["shapes"]))
        net.eval()
        size = tuple(data["size"])
        first = seeded.seeded_inputs((1, 3, *size), data["seeds"][0]).sigmoid()
        second = seeded.seeded_inputs((1, 3, *size), data["seeds"][1]).sigmoid()
        for case in data["outputs"]:
            stamp = torch.tensor([case["moment"]], dtype=torch.float32).view(-1, 1, 1, 1)
            with torch.inference_mode():
                got = net(first, second, stamp).clamp(0, 1)
            expected = torch.tensor(case["picture"], dtype=torch.float32).reshape(1, 3, *size)
            self.assertEqual(tuple(got.shape), (1, 3, *size))  # padded to 64 inside, cropped back
            self.assertLess(float((got - expected).abs().max()), 2e-5, case["moment"])

    @unittest.skipUnless(WEIGHTS.is_file(), "the RIFE 4.9 weights are not installed")
    def test_the_real_weights_fit_without_a_single_key_left_over(self):
        import torch

        from engine.models import rife

        state = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        own = rife.IFNet().state_dict()
        self.assertEqual({k: tuple(v.shape) for k, v in state.items()}, {k: tuple(v.shape) for k, v in own.items()})

    def test_the_authors_checkpoint_layout_loads_like_the_plain_one(self):
        """Practical-RIFE's flownet.pkl names every tensor "module.<name>"; it loads to the same network as a plain
        state dict, and a file with only some names prefixed is refused instead of half loaded."""
        import tempfile
        from collections import OrderedDict

        import torch

        from engine.models import rife

        torch.manual_seed(3)
        plain = rife.IFNet().state_dict()
        with tempfile.TemporaryDirectory() as folder:
            torch.save(plain, Path(folder) / "rife49.pth")
            torch.save(OrderedDict(("module." + k, v) for k, v in plain.items()), Path(folder) / "flownet.pkl")
            mixed = OrderedDict(plain)
            mixed["module.encode.0.bias"] = mixed.pop("encode.0.bias")
            torch.save(mixed, Path(folder) / "mixed.pkl")
            for name in ("rife49.pth", "flownet.pkl"):
                loaded = rife.load_rife(Path(folder) / name).state_dict()
                self.assertEqual(list(loaded), list(plain), name)
                self.assertTrue(all(torch.equal(loaded[k], plain[k]) for k in plain), name)
            with self.assertRaises(RuntimeError):
                rife.load_rife(Path(folder) / "mixed.pkl")

    def test_any_picture_size_comes_back_at_its_own_size(self):
        import torch

        from engine.models import rife

        torch.manual_seed(0)
        net = rife.IFNet().eval()
        for height, width in ((33, 70), (64, 64), (65, 129)):
            a, b = torch.rand(1, 3, height, width), torch.rand(1, 3, height, width)
            with torch.inference_mode():
                self.assertEqual(tuple(net(a, b, 0.5).shape), (1, 3, height, width))


class Blend:
    """Stands in for the network: the moment itself is the answer, so the layout can be read off."""

    def __init__(self):
        import torch

        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.moments = []

    def parameters(self):
        yield self.weight

    def __call__(self, first, second, moments):
        self.moments.extend(moments.flatten().tolist())
        return first * (1 - moments) + second * moments


@unittest.skipUnless(HAVE, "torch is not installed")
class InterpolateTests(unittest.TestCase):
    def ramp(self, count):
        import torch

        return torch.stack([torch.full((3, 4, 4), i / (count - 1)) for i in range(count)])

    def test_every_output_frame_sits_on_the_output_clock(self):
        from engine.models import rife

        frames = self.ramp(7)  # 7 frames at 16 fps last 0.375 s
        out = rife.interpolate(Blend(), frames, 16, 30)
        self.assertEqual(out.shape[0], 13)  # 0 .. 12 / 30 s, the last frame is the last source frame
        for index in range(13):
            expected = min(index * 16 / 30, 6.0) / 6.0  # position in source frames, never past the end
            self.assertAlmostEqual(float(out[index, 0, 0, 0]), expected, places=5, msg=index)

    def test_frames_that_fall_on_the_clock_are_copied_not_interpolated(self):
        import torch

        from engine.models import rife

        frames = self.ramp(5)
        model = Blend()
        out = rife.interpolate(model, frames, 15, 30)  # every other frame is an original
        self.assertEqual(out.shape[0], 9)
        for index in range(0, 9, 2):
            self.assertTrue(torch.equal(out[index], frames[index // 2]))
        self.assertEqual(model.moments, [0.5] * 4)

    def test_batches_do_not_change_the_result(self):
        import torch

        from engine.models import rife

        torch.manual_seed(1)
        net = rife.IFNet().eval()
        frames = torch.rand(5, 3, 40, 56)
        one = rife.interpolate(net, frames, 16, 30, batch_size=1)
        many = rife.interpolate(net, frames, 16, 30, batch_size=4)
        self.assertEqual(one.dtype, torch.float32)
        self.assertLess(float((one - many).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
