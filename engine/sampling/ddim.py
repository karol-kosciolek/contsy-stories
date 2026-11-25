# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""DDIM for discrete-time v-prediction models (EzAudio), step for step as diffusers' DDIMScheduler computes it.

EzAudio samples with `DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", beta_start=0.00085,
beta_end=0.012, prediction_type="v_prediction", rescale_betas_zero_snr=True, timestep_spacing="trailing",
clip_sample=False)` and its defaults (`set_alpha_to_one=True`, `steps_offset=0`). The operations below are the
same float32 tensor operations in the same order, so the results agree to the last bit or nearly
(tests/test_engine_ezaudio.py compares them with diffusers 0.42.0).

Notation: alpha_bar(t) is the cumulative product of (1 - beta); a noisy sample is
x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise, and the network predicts v = sqrt(alpha_bar) * noise -
sqrt(1 - alpha_bar) * x_0. With the zero-terminal-SNR rescale (arXiv 2305.08891) alpha_bar(999) is exactly 0: the
first step starts from pure noise and its estimate of x_0 is -v.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Schedule:
    alphas_cumprod: Tensor  # [train_timesteps] float32

    @property
    def train_timesteps(self) -> int:
        return int(self.alphas_cumprod.shape[0])


def _zero_terminal_snr(betas: Tensor) -> Tensor:
    """Shift and scale sqrt(alpha_bar) so that the last one is 0 and the first one is unchanged (Algorithm 1)."""
    alphas_bar_sqrt = torch.cumprod(1.0 - betas, dim=0).sqrt()
    first, last = alphas_bar_sqrt[0].clone(), alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= last
    alphas_bar_sqrt *= first / (first - last)
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = torch.cat([alphas_bar[0:1], alphas_bar[1:] / alphas_bar[:-1]])
    return 1 - alphas


def scaled_linear(train_timesteps: int, beta_start: float, beta_end: float, zero_terminal_snr: bool) -> Schedule:
    """The "scaled_linear" betas (linear in sqrt(beta)), optionally rescaled to zero terminal SNR."""
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, train_timesteps, dtype=torch.float32) ** 2
    if zero_terminal_snr:
        betas = _zero_terminal_snr(betas)
    return Schedule(torch.cumprod(1.0 - betas, dim=0))


def trailing_timesteps(train_timesteps: int, steps: int) -> List[int]:
    """The "trailing" spacing: steps evenly spaced timesteps ending at the last one (999, ...)."""
    if not 1 <= steps <= train_timesteps:
        raise ValueError(f"DDIM needs between 1 and {train_timesteps} steps, got {steps}")
    ratio = train_timesteps / steps
    return (np.round(np.arange(train_timesteps, 0, -ratio)).astype(np.int64) - 1).tolist()


def step(schedule: Schedule, steps: int, velocity: Tensor, timestep: int, sample: Tensor, eta: float,
         noise: Optional[Tensor] = None) -> Tensor:
    """x_{t-1} from x_t and the predicted velocity (DDIM paper eq. 12, with eta = 0 deterministic, 1 = DDPM).

    The previous timestep is `timestep - train_timesteps // steps` as in diffusers, also when that is not exactly
    the next timestep of the trailing schedule (steps that do not divide 1000). `noise` (standard normal, the shape
    of `sample`) is needed when eta > 0."""
    alphas_cumprod = schedule.alphas_cumprod
    previous = timestep - schedule.train_timesteps // steps
    alpha_prod_t = alphas_cumprod[timestep]
    alpha_prod_prev = alphas_cumprod[previous] if previous >= 0 else torch.tensor(1.0)
    beta_prod_t = 1 - alpha_prod_t

    original = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * velocity
    epsilon = (alpha_prod_t ** 0.5) * velocity + (beta_prod_t ** 0.5) * sample

    variance = ((1 - alpha_prod_prev) / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_prev)
    std_dev = eta * variance ** 0.5
    # At eta 1 the two terms can cancel exactly; a rounding error below zero would turn the sample into NaN.
    direction = (1 - alpha_prod_prev - std_dev ** 2).clamp(min=0) ** 0.5 * epsilon
    previous_sample = alpha_prod_prev ** 0.5 * original + direction
    if eta > 0:
        if noise is None:
            raise ValueError("DDIM with eta > 0 needs noise")
        previous_sample = previous_sample + std_dev * noise
    return previous_sample


def rescale_guidance(guided: Tensor, conditional: Tensor, factor: float) -> Tensor:
    """Guidance rescale (arXiv 2305.08891, section 3.4): the guided prediction scaled to the standard deviation of
    the conditional one, mixed with the unscaled prediction by `factor` (0 leaves it as it is). Per sample."""
    dims = list(range(1, conditional.ndim))
    std_conditional = conditional.std(dim=dims, keepdim=True)
    std_guided = guided.std(dim=dims, keepdim=True)
    rescaled = guided * (std_conditional / std_guided)
    return factor * rescaled + (1 - factor) * guided
