# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""engine.sampling: schedules and samplers checked against closed forms (no fixture)."""
import importlib.util
import math
import unittest

HAVE = importlib.util.find_spec("torch") is not None


def toy(x, sigma):
    """A nonlinear denoiser for the bookkeeping tests."""
    import torch

    return 0.9 * x + 0.1 * torch.sin(x)


@unittest.skipUnless(HAVE, "torch is not installed")
class ScheduleTests(unittest.TestCase):
    def test_the_table_is_the_shifted_time(self):
        import torch
        from engine.sampling import flow

        for shift in (1.0, 3.0, 3.1, 5.0):
            t = torch.arange(1, 1001) / 1000
            expected = shift * t / (1 + (shift - 1) * t)
            with self.subTest(shift=shift):
                torch.testing.assert_close(flow.sigma_table(shift), expected, rtol=0, atol=1e-7)
                self.assertEqual(float(flow.sigma_table(shift)[-1]), 1.0)

    def test_wan_lightx2v_sigmas_are_the_shifted_timesteps_1000_750_500_250(self):
        """LightX2V's distilled Wan 2.2 configurations: denoising_step_list [1000, 750, 500, 250], shift 5."""
        import torch
        from engine.sampling import flow

        t = torch.tensor([1.0, 0.75, 0.5, 0.25])
        expected = torch.cat([5 * t / (1 + 4 * t), torch.zeros(1)])
        torch.testing.assert_close(flow.spaced_sigmas(5.0, 4), expected, rtol=0, atol=1e-7)

    def test_steps_that_divide_1000_give_the_shifted_linspace(self):
        """Wan 2.2's get_sampling_sigmas: shifted linspace(1, 0, steps + 1) without the final 0, then 0."""
        import torch
        from engine.sampling import flow

        for steps in (1, 2, 4, 5, 8, 10, 20, 25, 40, 50):
            t = torch.linspace(1, 0, steps + 1)[:-1]
            expected = torch.cat([3 * t / (1 + 2 * t), torch.zeros(1)])
            with self.subTest(steps=steps):
                torch.testing.assert_close(flow.spaced_sigmas(3.0, steps), expected, rtol=0, atol=1e-6)

    def test_schedules_strictly_decrease_and_end_at_zero(self):
        from engine.sampling import flow

        for shift in (3.0, 3.1, 5.0, 8.0):
            for steps in (1, 2, 3, 4, 7, 8, 20, 50):
                for denoise in (1.0, 0.6, 0.5, 0.25):
                    sigmas = flow.denoise_sigmas(shift, steps, denoise)
                    with self.subTest(shift=shift, steps=steps, denoise=denoise):
                        self.assertEqual(len(sigmas), steps + 1)
                        self.assertEqual(float(sigmas[-1]), 0.0)
                        self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))

    def test_full_schedule_starts_at_one(self):
        from engine.sampling import flow

        self.assertAlmostEqual(float(flow.spaced_sigmas(3.0, 8)[0]), 1.0, places=6)

    def test_denoise_half_starts_in_the_middle(self):
        from engine.sampling import flow

        full = flow.spaced_sigmas(3.0, 16)
        half = flow.denoise_sigmas(3.0, 8, 0.5)
        self.assertTrue(bool((half == full[8:]).all()))
        self.assertLess(float(half[0]), 1.0)
        self.assertGreater(float(half[0]), float(half[1]))

    def test_denoise_one_is_the_full_schedule_and_bad_values_fail(self):
        from engine.sampling import flow

        self.assertTrue(bool((flow.denoise_sigmas(3.0, 8) == flow.spaced_sigmas(3.0, 8)).all()))
        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                flow.denoise_sigmas(3.0, 8, bad)
        with self.assertRaises(ValueError):
            flow.spaced_sigmas(3.0, 0)

    def test_shifted_time_is_the_identity_for_shift_one(self):
        from engine.sampling import flow

        self.assertEqual(flow.shifted_time(1.0, 0.3), 0.3)
        self.assertAlmostEqual(flow.shifted_time(3.0, 0.5), 0.75)


@unittest.skipUnless(HAVE, "torch is not installed")
class NoiseTests(unittest.TestCase):
    def test_noise_is_torch_randn_of_the_seed_on_the_cpu_and_casts_after_drawing(self):
        import torch
        from engine.sampling import flow

        noise = flow.make_noise(1234, (1, 4, 8, 8))
        torch.manual_seed(1234)
        self.assertTrue(torch.equal(noise, torch.randn(1, 4, 8, 8)))
        self.assertTrue(torch.equal(noise, flow.make_noise(1234, (1, 4, 8, 8))))
        self.assertFalse(torch.equal(noise, flow.make_noise(1235, (1, 4, 8, 8))))
        half = flow.make_noise(1234, (1, 4, 8, 8), torch.bfloat16)
        self.assertEqual(half.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(half, noise.to(torch.bfloat16)))

    def test_noise_scaling_and_the_starting_latent(self):
        import torch
        from engine.sampling import flow

        noise, latent = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        torch.testing.assert_close(flow.noise_scaling(0.6, noise, latent), 0.6 * noise + 0.4 * latent)
        self.assertTrue(torch.equal(flow.noise_scaling(1.0, noise, latent), noise))
        sigmas = flow.denoise_sigmas(3.0, 4, 0.5)
        torch.testing.assert_close(flow.initial_latent(sigmas, noise, latent),
                                   float(sigmas[0]) * noise + (1 - float(sigmas[0])) * latent)
        torch.testing.assert_close(flow.initial_latent(flow.spaced_sigmas(3.0, 4), noise), noise)

    def test_velocity_to_denoised(self):
        import torch
        from engine.sampling import flow

        x, v = torch.randn(2, 3), torch.randn(2, 3)
        torch.testing.assert_close(flow.denoised_from_velocity(x, 0.25, v), x - 0.25 * v)


@unittest.skipUnless(HAVE, "torch is not installed")
class ExpertTests(unittest.TestCase):
    def test_expert_split_covers_every_step_exactly_once(self):
        from engine.sampling import flow

        for steps in range(2, 31):
            for split in range(1, steps):
                high, low = flow.split_experts(steps, split)
                covered = list(range(high.start, high.stop)) + list(range(low.start, low.stop))
                with self.subTest(steps=steps, split=split):
                    self.assertEqual(covered, list(range(steps)))
        for bad in (0, 4, -1):
            with self.assertRaises(ValueError):
                flow.split_experts(4, bad)

    def test_expert_stage_sigmas_share_the_boundary(self):
        from engine.sampling import flow

        sigmas = flow.spaced_sigmas(5.0, 6)
        high, low = flow.split_experts(6, 3)
        self.assertEqual(len(high.sigmas(sigmas)), 4)
        self.assertEqual(len(low.sigmas(sigmas)), 4)
        self.assertEqual(float(high.sigmas(sigmas)[-1]), float(low.sigmas(sigmas)[0]))
        self.assertEqual(float(low.sigmas(sigmas)[-1]), 0.0)

    def test_two_euler_stages_equal_one_run_over_all_steps(self):
        """The sample is handed from the high-noise to the low-noise expert unchanged."""
        import torch
        from engine.sampling import flow, samplers

        sigmas = flow.spaced_sigmas(5.0, 4)
        start = flow.initial_latent(sigmas, flow.make_noise(7, (1, 4, 8, 8)))
        high, low = flow.split_experts(4, 2)
        staged = samplers.euler(toy, samplers.euler(toy, start, high.sigmas(sigmas)), low.sigmas(sigmas))
        self.assertTrue(torch.equal(staged, samplers.euler(toy, start, sigmas)))


@unittest.skipUnless(HAVE, "torch is not installed")
class SamplerTests(unittest.TestCase):
    SAMPLERS = ("euler", "exponential_multistep")

    def samplers(self):
        from engine.sampling import samplers

        return [getattr(samplers, name) for name in self.SAMPLERS]

    def test_a_constant_target_is_reached_exactly(self):
        """For a denoiser that does not change, Euler in sigma and the exponential step are both exact."""
        import torch
        from engine.sampling import flow

        target = torch.randn(1, 4, 8, 8)
        noise = flow.make_noise(3, (1, 4, 8, 8))
        sigmas = flow.spaced_sigmas(3.0, 6)
        for sampler in self.samplers():
            out = sampler(lambda x, sigma: target, flow.initial_latent(sigmas, noise), sigmas)
            torch.testing.assert_close(out, target, rtol=0, atol=1e-5)

    @staticmethod
    def linear_run(sampler, steps):
        """Integrate dx/dsigma = (x - a x) / sigma from sigma 1 to 0.05 (no final step to 0); the exact solution is
        x(sigma) = x(1) * sigma^(1 - a). Returns the relative error of the end point."""
        import torch

        a = 0.3
        sigmas = torch.exp(torch.linspace(0.0, math.log(0.05), steps + 1, dtype=torch.float64))
        x = torch.ones(1, dtype=torch.float64)
        out = sampler(lambda x, sigma: a * x, x, sigmas)
        exact = 0.05 ** (1 - a)
        return abs(float(out) - exact) / exact

    def test_euler_converges_at_first_order(self):
        from engine.sampling import samplers

        errors = [self.linear_run(samplers.euler, n) for n in (16, 32, 64)]
        for coarse, fine in zip(errors, errors[1:]):
            self.assertAlmostEqual(coarse / fine, 2.0, delta=0.3)

    def test_the_exponential_multistep_converges_at_second_order(self):
        """The method of arXiv 2308.02157 is second order: halving the step divides the error by about four."""
        from engine.sampling import samplers

        errors = [self.linear_run(samplers.exponential_multistep, n) for n in (16, 32, 64)]
        for coarse, fine in zip(errors, errors[1:]):
            self.assertAlmostEqual(coarse / fine, 4.0, delta=0.6)
        self.assertLess(errors[-1], self.linear_run(samplers.euler, 64) / 10)

    def test_a_denoiser_linear_in_log_sigma_is_integrated_exactly_after_the_first_step(self):
        """With D = c0 + c1 * lam (lam = -log sigma) the linear extrapolation through the last two estimates is exact,
        so every multistep step follows the exact solution x_next = e^-h x + int e^-(h - s) D(lam + s) ds."""
        import torch
        from engine.sampling import samplers

        c0, c1 = 0.4, -0.7

        def denoise(x, sigma):
            return torch.full_like(x, c0 + c1 * -math.log(float(sigma)))

        sigmas = torch.tensor([1.0, 0.8, 0.55, 0.3, 0.12], dtype=torch.float64)
        x0 = torch.tensor([0.25], dtype=torch.float64)
        after_first = samplers.euler(denoise, x0, sigmas[:2])  # the first step has no history: an Euler step
        got = samplers.exponential_multistep(denoise, x0, sigmas)
        x = float(after_first)
        for i in range(1, len(sigmas) - 1):
            lam, lam_next = -math.log(float(sigmas[i])), -math.log(float(sigmas[i + 1]))
            h = lam_next - lam
            # integral of e^-(h - s) (c0 + c1 (lam + s)) over [0, h]
            x = math.exp(-h) * x + (c0 + c1 * lam) * (1 - math.exp(-h)) + c1 * (h - 1 + math.exp(-h))
        self.assertAlmostEqual(float(got), x, places=12)

    def test_callback_sees_every_step_and_can_cancel(self):
        import torch
        from engine.sampling import flow

        sigmas = flow.spaced_sigmas(3.0, 5)
        x = torch.zeros(1, 4, 2, 2)
        for sampler in self.samplers():
            seen = []
            sampler(toy, x, sigmas, callback=lambda step, total: seen.append((step, total)))
            self.assertEqual(seen, [(i, 5) for i in range(5)])

            class Stop(Exception):
                pass

            calls = []

            def denoise(x, sigma):
                calls.append(1)
                return x

            def cancel(step, total):
                if step == 2:
                    raise Stop

            with self.assertRaises(Stop):
                sampler(denoise, x, sigmas, callback=cancel)
            self.assertEqual(len(calls), 2)

    def test_cfg_helpers(self):
        import torch
        from engine.sampling import samplers

        cond, uncond = torch.randn(1, 4, 3, 3), torch.randn(1, 4, 3, 3)
        torch.testing.assert_close(samplers.cfg_combine(cond, uncond, 1.0), cond)
        torch.testing.assert_close(samplers.cfg_combine(cond, uncond, 0.0), uncond)
        torch.testing.assert_close(samplers.cfg_combine(cond, uncond, 3.5), uncond + 3.5 * (cond - uncond))
        self.assertFalse(samplers.uses_uncond(1.0))
        self.assertTrue(samplers.uses_uncond(3.5))


if __name__ == "__main__":
    unittest.main()
