# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Text-to-image with Z-Image-Turbo.

What runs: the Qwen3-4B text encoder turns the prompt into the conditioning (`models.zimage.text`); the
transformer denoises a latent of the picture's size in 8 steps of the
second-order exponential multistep sampler (`sampling.samplers.exponential_multistep`) over evenly spaced
sigmas with flow shift 3 (`sampling.flow.denoise_sigmas`); the VAE decodes the result. Z-Image-Turbo is
distilled for guidance-free sampling (Tongyi-MAI/Z-Image, Apache-2.0; diffusers `ZImagePipeline` with
guidance 0), so every step is a single forward pass with the prompt. The sampler and the 'simple'-style table
picks are Contsy's own choice; the official pipeline samples with Euler (`FlowMatchEulerDiscreteScheduler`,
shift 3).

Peak memory is kept low by running the networks one after another: the text encoder encodes the
prompt and is released, the transformer runs all steps and is released, the VAE decodes last. Each
network is loaded straight onto the compute device, tensor by tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image

from ..core import streaming
from ..core.cancel import CancelToken
from ..core.stages import StatsCb, release, stage
from ..formats import lora as lora_format
from ..models.zimage import text, transformer, vae
from ..sampling import flow, samplers

SHIFT = 3.0  # flow shift (sampling.flow), the value of Z-Image-Turbo's scheduler config
LATENT_CHANNELS = 16
PIXELS_PER_LATENT = 8
SIZE_MULTIPLE = 16  # the transformer patches the latent 2x2, so pictures are multiples of 8 * 2

ProgressCb = Callable[[float, str], None]

# Share of the progress bar: loading and encoding, the denoising steps, decoding.
_ENCODED, _DENOISED = 0.08, 0.92


@dataclass(frozen=True)
class ZImageModels:
    """How to obtain the three networks. The product uses `from_files`; tests pass tiny random models."""
    load_text: Callable[[torch.device, torch.dtype], Tuple[object, torch.nn.Module]]  # (tokenizer, encoder)
    load_transformer: Callable[[torch.device, torch.dtype], torch.nn.Module]
    load_vae: Callable[[torch.device, torch.dtype], torch.nn.Module]

    @classmethod
    def from_files(cls, transformer_file: str | Path, text_encoder_file: str | Path, vae_file: str | Path,
                   tokenizer_dir: str | Path, cache=None) -> "ZImageModels":
        def load(loader, path, role, device, dtype):
            return (loader(path, device, dtype) if cache is None else cache.load(path, role, loader, device, dtype))

        return cls(
            load_text=lambda device, dtype: (text.load_tokenizer(tokenizer_dir),
                                             load(text.load_text_encoder, text_encoder_file, "text", device, dtype)),
            load_transformer=lambda device, dtype: load(transformer.load_transformer, transformer_file, "dit", device, dtype),
            load_vae=lambda device, dtype: load(vae.load_vae, vae_file, "vae", device, dtype),
            recycle=None if cache is None else cache.release)

    recycle: Optional[Callable[[torch.nn.Module], None]] = None


def _check_size(width: int, height: int) -> None:
    if width % SIZE_MULTIPLE or height % SIZE_MULTIPLE or width < SIZE_MULTIPLE or height < SIZE_MULTIPLE:
        raise ValueError(f"width and height must be multiples of {SIZE_MULTIPLE}, got {width}x{height}")


def txt2img(models: ZImageModels, prompt: str, width: int, height: int, seed: int, steps: int = 8,
            device: torch.device | str = "cuda",
            dtype: torch.dtype = torch.bfloat16, vae_dtype: Optional[torch.dtype] = None,
            progress_cb: Optional[ProgressCb] = None, cancel: Optional[CancelToken] = None,
            stats: Optional[StatsCb] = None, stream: bool = False, loras=(), then=()) -> Image.Image:
    """Generate one picture of exactly `width` x `height` pixels (both multiples of 16). `then`: files the stage after
    the picture reads first, read ahead while the steps run (video.read_next_stage_ahead).

    `loras` is a list of (file, strength) attached to the transformer for this picture (a style LoRA of an
    animated film, app/styles.py); the weights themselves are never changed.

    `dtype` is the weight/compute dtype of the text encoder and the
    transformer; the VAE uses `vae_dtype` (default: `dtype`). `cancel.check()` runs before every
    step and raises `Cancelled`. `progress_cb(fraction, text)` is called after loading/encoding, after
    every step and when the picture is done.
    """
    _check_size(width, height)
    return _generate(models, prompt, (width, height), seed, steps, device, dtype, vae_dtype, progress_cb, cancel,
                     stats, stream, tuple(loras or ()), then)


def _generate(models: ZImageModels, prompt: str, size: Tuple[int, int], seed: int, steps: int, device, dtype,
              vae_dtype: Optional[torch.dtype], progress_cb: Optional[ProgressCb], cancel: Optional[CancelToken],
              stats: Optional[StatsCb], stream: bool, loras=(), then=()) -> Image.Image:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    width, height = size
    device = torch.device(device)
    cancel = cancel or CancelToken()
    autoencoder = None

    def report(fraction: float, message: str) -> None:
        if progress_cb is not None:
            progress_cb(fraction, message)

    try:
        # 1. Prompt -> embedding, then the text encoder is released.
        cancel.check()
        report(0.0, "Loading the text encoder")
        with stage("text encoder", device, stats):
            tokenizer, encoder = models.load_text(torch.device("cpu") if stream else device, dtype)
            remove_stream = None
            try:
                if stream:
                    remove_stream = streaming.stream_model(encoder, device, blocks="layers", pin=False)
                context = text.encode_prompt(tokenizer, encoder, prompt)
            finally:
                if remove_stream is not None:
                    remove_stream()
                if models.recycle is not None:
                    models.recycle(encoder)
                del encoder
                release(device)
        cancel.check()

        # 2. Denoise from float32 noise drawn on the CPU from the seed (the same on any GPU).
        sigmas = flow.denoise_sigmas(SHIFT, steps)
        shape = (1, LATENT_CHANNELS, height // PIXELS_PER_LATENT, width // PIXELS_PER_LATENT)
        noise = flow.make_noise(seed, shape)
        latent = flow.initial_latent(sigmas, noise).to(device)
        report(_ENCODED / 2, "Loading the image model")
        with stage("transformer", device, stats):
            network = models.load_transformer(torch.device("cpu") if stream else device, dtype)
            remove_stream = remove_loras = None
            try:
                if stream:
                    remove_stream = streaming.stream_model(network, device, blocks="layers", pin=False)
                if loras:
                    names = [name for name, _ in network.named_modules()]
                    remove_loras = lora_format.attach(
                        network, [(lora_format.load_lora(path, names).to(device, dtype), strength)
                                  for path, strength in loras], seed=seed)
                report(_ENCODED, "Generating")
                from .video import read_next_stage_ahead

                read_next_stage_ahead(then)

                def denoise_step(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                    return transformer.predict_denoised(network, x, sigma, context)

                def before_step(step: int, total: int) -> None:
                    if step > 0:  # steps 0..step-1 are done
                        report(_ENCODED + (_DENOISED - _ENCODED) * step / total, f"Step {step}/{total}")
                    cancel.check()  # after the report, so a cancel requested from progress_cb stops the very next step

                with torch.inference_mode():
                    latent = samplers.exponential_multistep(denoise_step, latent, sigmas, before_step)
                report(_DENOISED, f"Step {steps}/{steps}")
            finally:
                if remove_loras is not None:
                    remove_loras()
                if remove_stream is not None:
                    remove_stream()
                if models.recycle is not None:
                    models.recycle(network)
                del network
                release(device)
        cancel.check()
        report(_DENOISED, "Decoding")

        # 3. Latent -> picture.
        with stage("vae", device, stats):
            autoencoder = models.load_vae(device, vae_dtype or dtype)
            try:
                result = vae.decode(autoencoder, latent)
            except torch.OutOfMemoryError:
                result = None  # retried below, once the failed attempt's memory can be released
            if result is None:
                release(device)
                result = vae.decode(autoencoder, latent, tiled=True)
        report(1.0, "Done")
        return result
    finally:
        if models.recycle is not None and autoencoder is not None:
            models.recycle(autoencoder)
        del autoencoder
        release(device)
