# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Building a network without its random starting weights (engine.core.skip_init), for checkpoints that cover them."""
import importlib.util
import unittest

HAVE_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAVE_TORCH, "torch is missing")
class NoWeightInitTest(unittest.TestCase):
    def test_random_fills_are_skipped_inside_and_restored_after(self):
        import torch
        from engine.core.skip_init import no_weight_init

        original = torch.nn.init.kaiming_uniform_
        with no_weight_init():
            self.assertIsNot(torch.nn.init.kaiming_uniform_, original)
            tensor = torch.full((3,), 7.0)
            self.assertIs(torch.nn.init.normal_(tensor), tensor)
            self.assertEqual(tensor.tolist(), [7.0, 7.0, 7.0])
            self.assertEqual(torch.nn.init.zeros_(tensor).tolist(), [0.0, 0.0, 0.0])  # constant fills stay
        self.assertIs(torch.nn.init.kaiming_uniform_, original)

    def test_the_fills_come_back_after_an_error(self):
        import torch
        from engine.core.skip_init import no_weight_init

        original = torch.nn.init.uniform_
        with self.assertRaises(RuntimeError):
            with no_weight_init():
                raise RuntimeError("load failed")
        self.assertIs(torch.nn.init.uniform_, original)

    def test_a_network_built_inside_equals_its_checkpoint(self):
        import torch
        from engine.core.skip_init import no_weight_init

        torch.manual_seed(0)
        source = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.LayerNorm(16), torch.nn.Embedding(5, 8))
        with no_weight_init():
            built = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.LayerNorm(16), torch.nn.Embedding(5, 8))
        result = built.load_state_dict(source.state_dict())
        self.assertEqual((result.missing_keys, result.unexpected_keys), ([], []))
        for (name, mine), (_, theirs) in zip(built.state_dict().items(), source.state_dict().items()):
            self.assertTrue(torch.equal(mine, theirs), name)


if __name__ == "__main__":
    unittest.main()
