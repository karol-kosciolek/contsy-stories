# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The LoRA hook on activations too big to handle in one piece (Wan video): slices of tokens give the same
numbers as one piece, the heavy operations only ever see a slice, and the output is updated in place only when
that is safe. Small activations (every Z-Image picture) take the unchanged one-piece path."""
import importlib.util
import unittest
import unittest.mock

HAVE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None


def build(kind, dtype):
    """(model, LoRA) with one hooked module of the given kind."""
    import torch
    from engine.formats.lora import LoRA, LoRAWeights
    from torch import nn

    torch.manual_seed(3)
    if kind == "linear":
        layer = nn.Linear(64, 96).to(dtype)
        lora = LoRA(modules={"0": LoRAWeights(0.1 * torch.randn(4, 64), 0.1 * torch.randn(96, 4), None)},
                    deltas={"0": {"bias": 0.05 * torch.randn(96)}})
    elif kind == "layernorm":
        layer = nn.LayerNorm(64)
        lora = LoRA(deltas={"0": {"weight": 0.1 * torch.randn(64), "bias": 0.1 * torch.randn(64)}})
    elif kind == "layernorm-bias-only":
        layer = nn.LayerNorm(64)
        lora = LoRA(deltas={"0": {"bias": 0.1 * torch.randn(64)}})
    else:
        layer = nn.RMSNorm(64).to(dtype)
        lora = LoRA(deltas={"0": {"weight": 0.1 * torch.randn(64)}})
    return nn.Sequential(layer).eval(), lora


def run(kind, dtype, x, rounding="nearest", slices=None):
    """The hooked module's output; `slices` = (threshold, slice size) forces the sliced path."""
    import torch
    from engine.formats import lora as module
    from engine.formats.lora import attach

    model, lora = build(kind, dtype)
    patches = [] if slices is None else [unittest.mock.patch.object(module, "_SLICE_THRESHOLD", slices[0]),
                                         unittest.mock.patch.object(module, "_SLICE_ELEMENTS", slices[1])]
    for patch in patches:
        patch.start()
    remove = attach(model, [(lora, 0.7)], rounding=rounding, seed=5)
    try:
        with torch.inference_mode():
            return model(x).clone()
    finally:
        remove()
        for patch in patches:
            patch.stop()


@unittest.skipUnless(HAVE, "torch is not installed")
class SlicedHookTests(unittest.TestCase):
    def test_a_small_activation_gets_exactly_the_documented_sum(self):
        import torch
        import torch.nn.functional as F

        x = torch.randn(8, 64).to(torch.bfloat16)
        model, lora = build("linear", torch.bfloat16)
        plain = model(x)
        weights = lora.modules["0"]
        extra = 0.7 * F.linear(F.linear(x, weights.down.to(torch.bfloat16)), weights.up.to(torch.bfloat16))
        extra = extra + 0.7 * lora.deltas["0"]["bias"].to(torch.bfloat16)
        expected = plain + extra.to(plain.dtype)
        self.assertTrue(torch.equal(run("linear", torch.bfloat16, x), expected))

    def test_slices_give_the_numbers_of_one_piece(self):
        import torch

        x = torch.randn(2, 50, 64)
        for kind in ("linear", "layernorm", "layernorm-bias-only", "rmsnorm"):
            for dtype in (torch.float32, torch.bfloat16):
                for rounding in ("nearest", "stochastic"):
                    with self.subTest(kind=kind, dtype=str(dtype), rounding=rounding):
                        xs = x.to(dtype)
                        whole = run(kind, dtype, xs, rounding)
                        sliced = run(kind, dtype, xs, rounding, slices=(1000, 700))
                        self.assertEqual(whole.dtype, sliced.dtype)
                        tolerance = 5e-2 if dtype is torch.bfloat16 else 1e-5
                        torch.testing.assert_close(sliced.float(), whole.float(), rtol=tolerance, atol=tolerance)

    def test_the_heavy_operations_only_see_slices(self):
        import torch
        import torch.nn.functional as F

        x = torch.randn(40, 64)
        seen = []
        real_layer_norm = F.layer_norm

        def spy(*args, **kwargs):
            if args[2] is None:  # the hook's own normalisation (the module itself passes its weight)
                seen.append(args[0].numel())
            return real_layer_norm(*args, **kwargs)

        with unittest.mock.patch.object(F, "layer_norm", spy):
            run("layernorm", torch.float32, x, slices=(1000, 300))
        self.assertGreaterEqual(len(seen), 10)  # 2560 elements in slices of 4 tokens
        self.assertLessEqual(max(seen), 300)
        seen.clear()
        with unittest.mock.patch.object(F, "layer_norm", spy):
            run("layernorm", torch.float32, x)  # below the threshold: one piece
        self.assertEqual(seen, [x.numel()])

    def test_the_output_is_updated_in_place_only_when_that_is_safe(self):
        import torch
        from engine.formats import lora as module
        from engine.formats.lora import attach

        x = torch.randn(30, 64)
        for grad, in_place in ((False, True), (True, False)):
            model, lora = build("linear", torch.float32)
            raw, final = [], []
            model[0].register_forward_hook(lambda m, a, o: raw.append(o))  # sees the module's own output
            with unittest.mock.patch.object(module, "_SLICE_THRESHOLD", 100), \
                    unittest.mock.patch.object(module, "_SLICE_ELEMENTS", 400):
                remove = attach(model, [(lora, 1.0)])
                model[0].register_forward_hook(lambda m, a, o: final.append(o))  # sees what the LoRA hook returned
                with torch.set_grad_enabled(grad):
                    model(x)
                remove()
            with self.subTest(grad=grad):
                self.assertEqual(final[0] is raw[0], in_place)
                if not in_place:  # the module's own result must not have been touched
                    plain, _ = build("linear", torch.float32)
                    torch.testing.assert_close(raw[0], plain(x), rtol=0, atol=0)

    def test_hooks_are_removed_after_a_sliced_run(self):
        import torch
        from engine.formats.lora import attach

        model, lora = build("rmsnorm", torch.float32)
        remove = attach(model, [(lora, 1.0)])
        remove()
        self.assertEqual(len(model[0]._forward_hooks), 0)
        self.assertEqual(model(torch.randn(2, 64)).shape, (2, 64))

    def test_the_slice_size_counts_the_batch_dimension_too(self):
        import torch
        import torch.nn.functional as F

        x = torch.randn(3, 40, 64)  # a batch of 3: every slice of tokens holds 3 x 64 elements per token
        seen = []
        real_layer_norm = F.layer_norm

        def spy(*args, **kwargs):
            if args[2] is None:
                seen.append(args[0].numel())
            return real_layer_norm(*args, **kwargs)

        with unittest.mock.patch.object(F, "layer_norm", spy):
            whole = run("layernorm", torch.float32, x)
            seen.clear()
            sliced = run("layernorm", torch.float32, x, slices=(1000, 300))
        self.assertLessEqual(max(seen), 300)  # a step that ignored the batch would hand 768 elements to every call
        torch.testing.assert_close(sliced, whole, rtol=1e-5, atol=1e-5)

    def test_a_norm_over_several_dimensions_is_not_cut_into_slices(self):
        import torch
        from engine.formats.lora import LoRA, attach
        from torch import nn

        torch.manual_seed(1)
        norm = nn.Sequential(nn.LayerNorm((4, 16)))
        lora = LoRA(deltas={"0": {"weight": 0.1 * torch.randn(4, 16)}})
        x = torch.randn(2, 30, 4, 16)  # tokens x (4, 16): dim -2 is one of the normalised axes
        remove = attach(norm, [(lora, 1.0)])
        with torch.inference_mode():
            expected = norm(x).clone()
        remove()
        from engine.formats import lora as module

        with unittest.mock.patch.object(module, "_SLICE_THRESHOLD", 100), unittest.mock.patch.object(module, "_SLICE_ELEMENTS", 400):
            remove = attach(norm, [(lora, 1.0)])
            with torch.inference_mode():
                got = norm(x).clone()  # must not raise inside the hook
            remove()
        torch.testing.assert_close(got, expected)

    def test_an_output_made_in_inference_mode_is_copied_not_written_to_outside_it(self):
        import torch
        from engine.formats import lora as module
        from engine.formats.lora import LoRA, LoRAWeights, attach
        from torch import nn

        class InferenceLinear(nn.Linear):  # its output is an inference tensor although it is called in normal mode
            def forward(self, x):
                with torch.inference_mode():
                    return super().forward(x.detach())

        torch.manual_seed(2)
        model = nn.Sequential(InferenceLinear(64, 96))
        lora = LoRA(modules={"0": LoRAWeights(0.1 * torch.randn(4, 64), 0.1 * torch.randn(96, 4), None)})
        x = torch.randn(30, 64)
        plain = model(x)
        with unittest.mock.patch.object(module, "_SLICE_THRESHOLD", 100), unittest.mock.patch.object(module, "_SLICE_ELEMENTS", 400):
            remove = attach(model, [(lora, 1.0)])
            got = model(x)  # writing into an inference tensor outside inference mode would raise
            remove()
        self.assertFalse(torch.equal(got, plain))  # the LoRA term was added
        inside = []
        with unittest.mock.patch.object(module, "_SLICE_THRESHOLD", 100), unittest.mock.patch.object(module, "_SLICE_ELEMENTS", 400):
            remove = attach(model, [(lora, 1.0)])
            with torch.inference_mode():
                inside.append(model(x))  # inside inference mode the in-place path is fine
            remove()
        torch.testing.assert_close(inside[0], got)

    def test_every_slice_and_every_call_draws_its_own_rounding_noise(self):
        import torch
        from engine.formats import lora as module
        from engine.formats.lora import LoRA, LoRAWeights, attach
        from torch import nn

        torch.manual_seed(3)
        model = nn.Sequential(nn.Linear(64, 96).to(torch.bfloat16))
        model[0].weight.data *= 8
        lora = LoRA(modules={"0": LoRAWeights(0.01 * torch.randn(4, 64), 0.01 * torch.randn(96, 4), None)})
        row = torch.randn(1, 64).to(torch.bfloat16)
        x = row.repeat(40, 1)  # identical rows: only the rounding noise can make two output rows differ
        with unittest.mock.patch.object(module, "_SLICE_THRESHOLD", 100), unittest.mock.patch.object(module, "_SLICE_ELEMENTS", 400):
            remove = attach(model, [(lora, 1.0)], rounding="stochastic", seed=9)
            with torch.inference_mode():
                first, second = model(x).clone(), model(x).clone()
            remove()
            remove = attach(model, [(lora, 1.0)], rounding="stochastic", seed=9)
            with torch.inference_mode():
                again = model(x).clone()
            remove()
        step = 400 // 96  # rows per slice
        self.assertFalse(torch.equal(first[:step], first[step:2 * step]))  # a slice with the noise of the one before would match
        self.assertFalse(torch.equal(first, second))  # the second call continues the noise stream
        self.assertTrue(torch.equal(first, again))  # and the whole sequence depends only on the seed


if __name__ == "__main__":
    unittest.main()
