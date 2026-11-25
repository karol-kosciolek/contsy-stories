# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""LoRA hooks on bfloat16 layers: small terms are kept on average with stochastic rounding, and the
hook also works when a layer is called with its input as a keyword argument."""
import importlib.util
import unittest

HAVE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAVE, "torch is not installed")
class RoundingTests(unittest.TestCase):
    def layer_and_lora(self):
        import torch
        from torch import nn

        from engine.formats.lora import LoRA, LoRAWeights

        torch.manual_seed(0)
        layer = nn.Linear(64, 64, bias=False)
        layer.weight.data *= 8  # outputs of about 5, where a bfloat16 step (0.03) dwarfs the LoRA term (0.002)
        layer = layer.to(torch.bfloat16)
        down = (0.01 * torch.randn(4, 64)).to(torch.bfloat16)
        up = (0.01 * torch.randn(64, 4)).to(torch.bfloat16)
        return nn.Sequential(layer), LoRA({"0": LoRAWeights(down, up, None)}), down, up

    def survival(self, rounding, seed=0):
        """Share of the exact LoRA effect that reaches a bfloat16 output (1.0 = all of it)."""
        import torch
        from torch.nn import functional as F

        from engine.formats.lora import attach

        model, lora, down, up = self.layer_and_lora()
        x = torch.randn(2048, 64).to(torch.bfloat16)
        plain = model(x).detach().float()
        remove = attach(model, [(lora, 1.0)], rounding=rounding, seed=seed)
        with_lora = model(x).detach().float()
        remove()
        exact = F.linear(F.linear(x.float(), down.float()), up.float())
        return float(((with_lora - plain) * exact).sum() / (exact * exact).sum())

    def test_stochastic_rounding_keeps_the_average_of_a_term_below_half_a_step(self):
        import torch

        from engine.formats.lora import _stochastic_bf16

        generator = torch.Generator().manual_seed(0)
        for value in (1.001, -1.001):  # a bfloat16 step at 1.0 is 0.0078; the term is 0.001
            numbers = torch.full((1_000_000,), value, dtype=torch.float32)
            self.assertEqual(float(numbers.to(torch.bfloat16).float().mean()), round(value))
            self.assertAlmostEqual(float(_stochastic_bf16(numbers, generator).float().mean()), value, delta=3e-4)

    def test_nearest_loses_most_of_a_small_term_and_stochastic_keeps_it(self):
        self.assertLess(self.survival("nearest"), 0.6)
        self.assertGreater(self.survival("stochastic"), 0.9)

    def test_stochastic_result_depends_only_on_the_seed(self):
        import torch

        from engine.formats.lora import attach

        outputs = []
        for seed in (1, 1, 2):
            model, lora, _, _ = self.layer_and_lora()
            x = torch.randn(256, 64).to(torch.bfloat16)
            remove = attach(model, [(lora, 1.0)], rounding="stochastic", seed=seed)
            outputs.append(model(x))
            remove()
        self.assertTrue(torch.equal(outputs[0], outputs[1]))
        self.assertFalse(torch.equal(outputs[0], outputs[2]))

    def test_an_unknown_rounding_mode_is_refused(self):
        from engine.formats.lora import attach

        model, lora, _, _ = self.layer_and_lora()
        with self.assertRaises(ValueError):
            attach(model, [(lora, 1.0)], rounding="round-half-even")

    def test_hook_works_when_the_input_is_given_by_keyword(self):
        import torch

        from engine.formats.lora import attach

        model, lora, _, _ = self.layer_and_lora()
        layer = model[0]
        x = torch.randn(8, 64).to(torch.bfloat16)
        remove = attach(model, [(lora, 1.0)])
        positional, keyword = layer(x), layer(input=x)
        remove()
        self.assertTrue(torch.equal(positional, keyword))
        self.assertFalse(torch.equal(layer(x), positional))  # the hook is gone and the term mattered


if __name__ == "__main__":
    unittest.main()
