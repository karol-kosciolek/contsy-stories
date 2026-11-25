# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Deterministic samplers and guidance for flow-matching models.

A sampler walks a descending list of sigmas (ending in 0) with a callable
`denoise(x, sigma) -> denoised`, where `sigma` is a 0-dim float32 tensor and `denoised` is the
model's estimate of the clean sample. How that estimate is produced (velocity, timestep scale,
guidance, LoRA) is the model adapter's business and stays out of this module.

`callback(step, total)` is called before each step starts, so it can report progress and may
raise to cancel the run; nothing is left half-updated when it does.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch

Denoiser = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
StepCallback = Callable[[int, int], None]


def _euler_step(x: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Explicit Euler along dx/dsigma = (x - denoised) / sigma."""
    return x + (x - denoised) / sigma * (sigma_next - sigma)


def euler(denoise: Denoiser, x: torch.Tensor, sigmas: torch.Tensor,
          callback: Optional[StepCallback] = None) -> torch.Tensor:
    """First-order Euler sampler."""
    total = len(sigmas) - 1
    for i in range(total):
        if callback is not None:
            callback(i, total)
        denoised = denoise(x, sigmas[i])
        x = _euler_step(x, sigmas[i], sigmas[i + 1], denoised)
    return x


def exponential_multistep(denoise: Denoiser, x: torch.Tensor, sigmas: torch.Tensor,
                          callback: Optional[StepCallback] = None) -> torch.Tensor:
    """Second-order exponential multistep sampler, deterministic.

    The method of Zhang, Tao and Chen, "Improved order analysis and design of exponential integrator for
    diffusion models sampling" (arXiv 2308.02157), for a flow model. In log-noise time lam = -log(sigma) the
    sampling ODE dx/dsigma = (x - D) / sigma becomes dx/dlam = D - x, and one step of length
    h = lam_next - lam is solved exactly for any D:

        x_next = e^-h x + integral over s in [0, h] of e^-(h - s) D(lam + s) ds.

    D is not known inside the step, so it is taken to change linearly in lam, along the line through the
    estimate of the previous step (at lam_prev) and the current one:  D(lam + s) = D + s * (D - D_prev) / r
    with r = lam - lam_prev. The two integrals are 1 - e^-h and h - 1 + e^-h, so

        x_next = e^-h x + (1 - e^-h) D + (h - 1 + e^-h) / r * (D - D_prev).

    The first step has no previous estimate and the step to sigma 0 has an infinite h: both are Euler steps
    (which are exact when D does not change). The scalar weights are computed in float64.
    """
    total = len(sigmas) - 1
    previous: Optional[torch.Tensor] = None
    for i in range(total):
        if callback is not None:
            callback(i, total)
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = denoise(x, sigma)
        if previous is None or float(sigma_next) == 0.0:
            x = _euler_step(x, sigma, sigma_next, denoised)
        else:
            lam_prev, lam, lam_next = (-math.log(float(sigmas[j])) for j in (i - 1, i, i + 1))
            h, r = lam_next - lam, lam - lam_prev
            decay = math.exp(-h)
            slope_weight = (h + math.expm1(-h)) / r
            x = decay * x + (-math.expm1(-h)) * denoised + slope_weight * (denoised - previous)
        previous = denoised
    return x


def uses_uncond(cfg_scale: float) -> bool:
    """At cfg 1 guidance reduces to the conditional prediction, so the unconditional pass is skipped."""
    return cfg_scale != 1.0


def cfg_combine(cond: torch.Tensor, uncond: torch.Tensor, scale: float) -> torch.Tensor:
    """Classifier-free guidance: `uncond + scale * (cond - uncond)`. Linear, so it gives the same
    result on denoised estimates as on velocities."""
    return uncond + (cond - uncond) * scale
