# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Sigma schedules, noise and step bookkeeping for the flow-matching models of the product.

* A noisy sample is `x = sigma * noise + (1 - sigma) * latent`; sigma runs from 1 (pure noise) to 0
  (rectified flow, as in diffusers' `FlowMatchEulerDiscreteScheduler`).
* The network predicts a velocity `v`; the denoised estimate is `x - sigma * v` (`denoised_from_velocity`).
* Time is shifted towards the noisy end: `sigma(t) = shift * t / (1 + (shift - 1) * t)`, the shift of the SD3
  paper (Esser et al. 2024, arXiv 2403.03206) that diffusers' scheduler applies as `shift` and the Wan 2.2 code
  (Wan-Video/Wan2.2, Apache-2.0) as `sample_shift`. The sampling sigmas are picked from the 1000 shifted values
  of t in {1/1000, ..., 1} (`sigma_table`).
* How a model turns sigma into its timestep input is the model adapter's job (Z-Image and Qwen-Image take
  sigma-based values on a 0-1 scale, Wan takes sigma * 1000).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

TRAIN_TIMESTEPS = 1000


def shifted_time(shift: float, t):
    """Shifted time: `shift * t / (1 + (shift - 1) * t)`; identity for shift 1. Works on floats and tensors."""
    if shift == 1.0:
        return t
    return shift * t / (1 + (shift - 1) * t)


def sigma_table(shift: float) -> torch.Tensor:
    """Ascending float32 table of the 1000 shifted sigmas; the last one is 1."""
    t = torch.arange(1, TRAIN_TIMESTEPS + 1, 1) / TRAIN_TIMESTEPS
    return shifted_time(float(shift), t)


def spaced_sigmas(shift: float, steps: int) -> torch.Tensor:
    """`steps` sigmas at evenly spaced positions of the table, descending, plus a final 0.

    Position i is table entry 999 - floor(i * 1000 / steps), i.e. t = 1 - floor(i * 1000 / steps) / 1000. When
    `steps` divides 1000 these are the shifted values of linspace(1, 0, steps + 1) without the 0, as the Wan 2.2
    code's `get_sampling_sigmas` computes them; for 4 steps they are the timesteps 1000, 750, 500 and 250 of
    LightX2V's distilled Wan 2.2 configurations (`denoising_step_list`)."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    table = sigma_table(shift)
    stride = len(table) / steps
    picked = [float(table[-(1 + int(i * stride))]) for i in range(steps)]
    return torch.tensor(picked + [0.0], dtype=torch.float32)


def denoise_sigmas(shift: float, steps: int, denoise: float = 1.0) -> torch.Tensor:
    """Sigmas for `steps` steps when only the last `denoise` fraction of the noise is removed.

    For denoise < 1 a longer schedule of int(steps / denoise) steps is built and only its last
    `steps + 1` values are used, so the sampler starts in the middle of the schedule.
    """
    if not 0.0 < denoise <= 1.0:
        raise ValueError("denoise must be in (0, 1]")
    if denoise > 0.9999:
        return spaced_sigmas(shift, steps)
    return spaced_sigmas(shift, int(steps / denoise))[-(steps + 1):]


def make_noise(seed: int, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Gaussian noise drawn in float32 on the CPU from `torch.manual_seed(seed)`, then cast to `dtype`: a seed
    gives the same noise on any GPU."""
    generator = torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, generator=generator, device="cpu").to(dtype)


def noise_scaling(sigma: float, noise: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mix noise into a latent at level sigma. Without a latent (txt2img) this is `sigma * noise`."""
    if latent is None:
        return sigma * noise
    return sigma * noise + (1.0 - sigma) * latent


def initial_latent(sigmas: torch.Tensor, noise: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The sampler's starting point: pure scaled noise for txt2img, the noised image for img2img."""
    return noise_scaling(float(sigmas[0]), noise, latent)


def denoised_from_velocity(x: torch.Tensor, sigma, velocity: torch.Tensor) -> torch.Tensor:
    """x0 estimate of a flow model whose output is the velocity: `x - sigma * v`."""
    return x - sigma * velocity


@dataclass(frozen=True)
class ExpertStage:
    """One Wan expert's share of the schedule: steps [start, stop) of the full list of sigmas."""
    start: int
    stop: int

    def sigmas(self, all_sigmas: torch.Tensor) -> torch.Tensor:
        """The sigmas this stage's sampler call receives (`stop - start + 1` values)."""
        return all_sigmas[self.start:self.stop + 1]


def split_experts(steps: int, split: int) -> Tuple[ExpertStage, ExpertStage]:
    """Wan 2.2 A14B: the high-noise expert runs steps [0, split), the low-noise expert [split, steps).

    The Wan 2.2 code switches from the high-noise to the low-noise network at a boundary timestep; LightX2V's
    distilled 4-step configurations put it after the second step (`boundary_step_index` 2). The sample is handed
    over unchanged. The multistep history of a sampler would not carry over, which is why Wan uses Euler.
    """
    if not 0 < split < steps:
        raise ValueError(f"split must lie strictly inside the {steps} steps, got {split}")
    return ExpertStage(0, split), ExpertStage(split, steps)
