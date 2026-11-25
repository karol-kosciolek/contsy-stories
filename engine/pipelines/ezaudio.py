# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Text-to-audio for scene ambience and timed effects with EzAudio-XL.

What a call does, as the reference (EzAudio src/inference.py, api/ezaudio.py) does it:

1. Every prompt of the call and the negative text are encoded together by FLAN-T5-XL (100 tokens each). The
   reference's unconditional text is the empty string, which is also how it was trained (10 percent of the
   captions replaced by ""); another negative text is allowed but is outside the training distribution.
2. The diffusion transformer denoises a 128-channel latent at 50 frames per second with DDIM (v-prediction,
   zero terminal SNR, trailing timesteps), classifier-free guidance as one batch of two (conditional,
   unconditional) and guidance rescale. Reference defaults: 100 steps, guidance 5, rescale 0.75, eta 1.
3. The waveform decoder turns the latent into 24 kHz mono samples; the engine resamples them to 48 kHz, the rate
   of the product's mix.

Length: a call makes exactly the requested length plus the timeline's two handles (`seconds + 0.5 s`), rounded up to
whole latent frames, so no time is spent on audio nobody uses. Up to 10 s (the training length) is one window.
Longer audio is continued with the model's own inpainting: each next window sees the last 2.5 s of latent made so
far as known frames (mask 0) and generates up to 7.5 s more (mask 1), with the reference's editing settings
(guidance 3.5, no rescale). The windows are joined in latent space and decoded once, so there is no seam in the
waveform. 30.5 s takes four windows.

Randomness: the initial noise and the noise of every DDIM step (eta > 0) come from one CPU generator per sound,
seeded with its seed, in the reference's order; the result does not depend on the GPU, and a CPU run of the
reference with the same seed draws the same numbers.

Memory (16 GB plan): the networks run one after another, each loaded straight to the GPU and released (or handed to
a ModelCache, which keeps it in host memory) before the next: FLAN-T5-XL encoder about 2.5 GB (float16 storage,
float32 compute), the transformer about 1.9 GB (16-bit matrices in the blocks; float32 norms, time conditioning and
edge layers, 68 M parameters), the decoder under 0.1 GB (float32, decoded in chunks of 4 s). Activations for one
10 s window with guidance are a few hundred MB. Nothing needs block streaming. Measured on an RTX 4070 Ti SUPER: a
5 s scene with a bed and two effects at 50 steps peaks at about 3.2 GiB allocated (3.4 GiB reserved).

Precision: on CUDA the transformer's block matrices are float16 by default. Against float32 they are off by
0.03-0.29 % per forward (bfloat16: 0.22-2.09 %), the largest activation stays 20 times below the float16 limit, and
on the GPU float16 ran as fast as bfloat16 with no non-finite output. A non-finite sample still fails the call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..core.cancel import CancelToken
from ..core.stages import release, stage
from ..media.audio import RATE as OUTPUT_RATE
from ..media.audio import TIMELINE_AUDIO_HANDLE as HANDLE  # extra audio the timeline wants at both ends
from ..models.ezaudio import config as ezconfig
from ..models.ezaudio import text, transformer, vae
from ..sampling import ddim, samplers

MAX_SECONDS = 30.0  # the longest sound a call makes (without the handles)
# Steps: the reference defaults to 100; 50 (what its paper's ablations use) made a 5 s scene with a bed and two
# effects in about 8.5 s on the GPU, against about 14 s at 100.
STEPS = 50
CFG, RESCALE, ETA = 5.0, 0.75, 1.0
NEGATIVE = ""  # the reference's unconditional text
# Continuation windows: the reference's editing settings, and how much of the previous window they see.
CONTINUE_CFG, CONTINUE_RESCALE = 3.5, 0.0
CONTEXT_SECONDS = 2.5

Progress = Callable[[float, str], None]
Stats = Callable[[str, float, int], None]  # (stage, seconds, peak CUDA bytes during the stage; 0 on the CPU)
_ENCODED, _SAMPLED = 0.08, 0.92  # share of the progress bar: loading and encoding, then sampling, then decoding
_RESAMPLE_TAPS, _RESAMPLE_BETA = 257, 8.6  # about 1 kHz of transition at 12 kHz and 86 dB of image rejection


@dataclass(frozen=True)
class EzAudioFiles:
    """The files of an EzAudio bundle as the official repositories ship them."""
    transformer: Path    # ckpts/s3/ezaudio_s3_xl.pt
    config: Path         # ckpts/ezaudio-xl.yml
    vae: Path            # ckpts/vae/1m.pt, with config.json next to it
    text_encoder: Path   # google/flan-t5-xl: config.json, spiece.model, the shard index and shard 1

    @classmethod
    def from_bundle(cls, bundle: str | Path) -> "EzAudioFiles":
        bundle = Path(bundle)
        return cls(bundle / "ezaudio_s3_xl.pt", bundle / "ezaudio-xl.yml", bundle / "vae" / "1m.pt",
                   bundle / "text_encoder")

    def missing(self) -> List[str]:
        """What is not on disk, by role."""
        needed = {"transformer": self.transformer, "configuration": self.config, "waveform decoder": self.vae,
                  "decoder configuration": self.vae.with_name("config.json"),
                  "text encoder configuration": self.text_encoder / "config.json",
                  "tokenizer": self.text_encoder / "spiece.model"}
        absent = [role for role, path in needed.items() if not Path(path).is_file()]
        index = self.text_encoder / text.INDEX_FILE
        if index.is_file():
            try:
                shards = text.shard_files(self.text_encoder)
            except (ValueError, KeyError):
                return absent + ["text encoder index"]
            absent += [f"text encoder {path.name}" for path in shards if not path.is_file()]
        elif not (self.text_encoder / text.SINGLE_FILE).is_file():
            absent.append("text encoder weights")
        return absent


@dataclass(frozen=True)
class EzAudioModels:
    """How to obtain the three networks. The product uses `from_files`; tests pass tiny random models."""
    settings: ezconfig.Settings
    load_text: Callable[[torch.device, torch.dtype], Tuple[object, torch.nn.Module]]  # (tokenizer, encoder)
    load_transformer: Callable[[torch.device, torch.dtype], torch.nn.Module]
    load_vae: Callable[[torch.device], torch.nn.Module]
    recycle: Optional[Callable[[torch.nn.Module], None]] = None

    @classmethod
    def from_files(cls, files: EzAudioFiles, cache=None) -> "EzAudioModels":
        absent = files.missing()
        if absent:
            raise FileNotFoundError("EzAudio files are missing: " + ", ".join(absent))
        settings = ezconfig.load(files.config)

        def load(loader, path, role, device, dtype):
            return loader(path, device, dtype) if cache is None else cache.load(path, role, loader, device, dtype)

        return cls(
            settings=settings,
            load_text=lambda device, dtype: (text.load_tokenizer(files.text_encoder),
                                             load(text.load_encoder, files.text_encoder, "text", device, dtype)),
            load_transformer=lambda device, dtype: load(transformer.load, files.transformer, "dit", device, dtype),
            load_vae=lambda device: load(vae.load, files.vae, "vae", device, torch.float32),
            recycle=None if cache is None else cache.release)


@dataclass(frozen=True)
class Sound:
    """One sound of a call: the prompt, its length on the timeline (the handles are added) and its seed."""
    prompt: str
    seconds: float
    seed: int


@dataclass(frozen=True)
class _Window:
    known: int  # frames taken over from the latent made so far (0 for the first window)
    new: int    # frames this window generates


def window_plan(frames: int, window: int, context: int) -> List[_Window]:
    """The windows that make `frames` latent frames: one window up to the training length, then continuations that
    each see `context` known frames and add at most `window - context` new ones."""
    if frames < 1 or not 0 < context < window:
        raise ValueError("window plan needs frames >= 1 and 0 < context < window")
    plan = [_Window(0, min(frames, window))]
    made = plan[0].new
    while made < frames:
        plan.append(_Window(context, min(window - context, frames - made)))
        made += plan[-1].new
    return plan


def _check(sounds: Sequence[Sound], steps: int, cfg: float, rescale: float, eta: float, train_timesteps: int) -> None:
    if not sounds:
        raise ValueError("no sound requested")
    for sound in sounds:
        if not isinstance(sound.seconds, (int, float)) or not math.isfinite(sound.seconds) \
                or not 0 < sound.seconds <= MAX_SECONDS:
            raise ValueError(f"audio duration must be between 0 and {MAX_SECONDS:g} seconds")
        if not 0 <= int(sound.seed) < 2 ** 64:
            raise ValueError("audio seed must be an unsigned 64-bit integer")
    if not 1 <= steps <= train_timesteps:
        raise ValueError(f"audio sampling needs between 1 and {train_timesteps} steps")
    # The reference spells "no guidance" as 0, which here would mean the unconditional prediction alone (a sound that
    # ignores the prompt); below 1 is refused so that a setting carried over from it fails loudly instead.
    if not math.isfinite(cfg) or cfg < 1:
        raise ValueError("audio guidance must be at least 1 (1 = the conditional pass only, the reference's 0)")
    if not 0 <= rescale <= 1 or not math.isfinite(eta) or eta < 0:
        raise ValueError("audio guidance rescale must be in [0, 1] and eta nonnegative")


def _frames(seconds: float, latent_rate: int) -> int:
    """Latent frames that cover the sound and both handles."""
    return math.ceil(round((seconds + 2 * HANDLE) * latent_rate, 6))


def _sample_window(model, schedule: ddim.Schedule, timesteps: List[int], steps: int, context: Tensor,
                   context_mask: Tensor, known: Optional[Tensor], new: int, cfg: float, rescale: float, eta: float,
                   generator: torch.Generator, device: torch.device, before_step: Callable[[], None]) -> Tensor:
    """One window: `known` [1, C, K] (or None) followed by `new` generated frames. Returns all K + new frames.

    `context` is [2, T, D] (conditional, unconditional) from `prepare_context`, `context_mask` [2, T]."""
    channels = model.latent_channels
    frames = new + (0 if known is None else known.shape[-1])
    latent = torch.randn((1, channels, frames), generator=generator, dtype=torch.float32).to(device)
    mask = full = None
    if known is not None:
        # The known frames come first; the network sees the mask token where it generates (mask 1).
        mask = torch.zeros((1, 1, frames), dtype=torch.bool, device=device)
        mask[..., known.shape[-1]:] = True
        full = torch.cat((known.to(device), torch.zeros((1, channels, new), device=device)), dim=-1)
    guided = samplers.uses_uncond(cfg)
    batch = 2 if guided else 1
    context, context_mask = context[:batch], context_mask[:batch]
    rotary = model.prepare_rotary(frames)
    for timestep in timesteps:
        before_step()
        output = model(latent.expand(batch, -1, -1), torch.tensor(timestep, device=device), context, context_mask,
                       rotary, full, mask).float()
        velocity = output
        if guided:
            conditional, unconditional = output.chunk(2)
            velocity = samplers.cfg_combine(conditional, unconditional, cfg)
            if rescale > 0:
                velocity = ddim.rescale_guidance(velocity, conditional, rescale)
        noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32).to(device) if eta > 0 else None
        latent = ddim.step(schedule, steps, velocity, timestep, latent, eta, noise)
    if known is not None:
        latent = torch.where(mask, latent, full)  # the reference copies the known frames back
    return latent


@lru_cache(maxsize=4)
def _resample_filter(up: int):
    from scipy.signal import firwin

    return firwin(_RESAMPLE_TAPS, 1.0 / up, window=("kaiser", _RESAMPLE_BETA))


def to_output_rate(waveform: Tensor, rate: int, samples: int) -> Tensor:
    """[1, n] float32 at `rate` -> [1, samples] float32 at 48 kHz (a windowed-sinc polyphase filter that keeps the
    band up to the source's Nyquist frequency), shorter audio padded with silence."""
    import numpy as np
    from scipy.signal import resample_poly

    if OUTPUT_RATE % rate:
        raise ValueError(f"cannot resample {rate} Hz to {OUTPUT_RATE} Hz by an integer factor")
    up = OUTPUT_RATE // rate
    signal = waveform.double().numpy()
    if up > 1:
        signal = resample_poly(signal, up, 1, axis=-1, window=_resample_filter(up).copy())  # the cached one stays
    out = torch.from_numpy(np.ascontiguousarray(signal[..., :samples])).float()
    if out.shape[-1] < samples:
        out = torch.nn.functional.pad(out, (0, samples - out.shape[-1]))
    return out


def _limit_peak(waveform: Tensor) -> Tensor:
    """The decoder ends without tanh, so a loud sound can pass full scale; scale it down instead of clipping."""
    peak = float(waveform.abs().max()) if waveform.numel() else 0.0
    return waveform / peak if peak > 1.0 else waveform


@torch.inference_mode()
def generate_many(models: EzAudioModels, sounds: Sequence[Sound], negative: str = NEGATIVE, *,
                  device: torch.device | str = "cuda", dtype: Optional[torch.dtype] = None, steps: int = STEPS,
                  cfg: float = CFG, rescale: float = RESCALE, eta: float = ETA,
                  min_seconds: float = 0.0, progress_cb: Optional[Progress] = None,
                  cancel: Optional[CancelToken] = None, stats: Optional[Stats] = None) -> List[Tensor]:
    """Mono CPU float32 waveforms at 48 kHz, one per sound, each `seconds + 0.5 s` long (the timeline's handles).

    The three networks are loaded once for all sounds (a scene's bed and its effects in one call). `cancel.check()`
    runs before anything is loaded, before every step and between decoded chunks.

    `cfg` is the usual guidance scale (uncond + cfg * (cond - uncond)), at least 1; at 1 only the conditional pass
    runs, which is what the reference does for a guidance of 0 or None (0 is refused here, see `_check`). `rescale`
    is guidance rescale, `eta` the DDIM noise (0 is deterministic). Continuation windows (sounds over 10 s) always
    use CONTINUE_CFG and CONTINUE_RESCALE.

    `min_seconds` > 0 generates at least that much (handles excluded) and returns the beginning: the model was trained
    on 10 s clips, and in a CPU check short windows came out louder and heavy in the bass; 0 makes exactly the
    requested length."""
    settings = models.settings
    _check(sounds, steps, cfg, rescale, eta, settings.train_timesteps)
    if not math.isfinite(min_seconds) or not 0 <= min_seconds <= MAX_SECONDS:
        raise ValueError(f"min_seconds must be between 0 and {MAX_SECONDS:g}")
    device = torch.device(device)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    cancel = cancel or CancelToken()

    def report(fraction: float, message: str) -> None:
        if progress_cb is not None:
            progress_cb(fraction, message)

    cancel.check()
    window = settings.window_frames
    context_frames = round(CONTEXT_SECONDS * settings.latent_rate)
    plans = [window_plan(_frames(max(sound.seconds, min_seconds), settings.latent_rate), window, context_frames)
             for sound in sounds]

    # 1. All prompts and the negative text in one batch; the encoder is released before the transformer loads.
    report(0.0, "Encoding the sound prompts")
    with stage("load text encoder", device, stats):
        tokenizer, encoder = models.load_text(device, dtype)
    try:
        cancel.check()
        with stage("encode", device, stats):
            prompts = [text.clean_prompt(sound.prompt) for sound in sounds] + [text.clean_prompt(negative)]
            ids, mask = text.tokenize(tokenizer, prompts, settings.max_tokens)
            hidden = text.encode(encoder, ids, mask)
    finally:
        if models.recycle is not None:
            models.recycle(encoder)
        del encoder
        release(device)

    # 2. Sampling: every window of every sound, with progress proportional to the frames it computes.
    cancel.check()
    report(_ENCODED / 2, "Loading EzAudio")
    schedule = ddim.scaled_linear(settings.train_timesteps, settings.beta_start, settings.beta_end, True)
    timesteps = ddim.trailing_timesteps(settings.train_timesteps, steps)
    total = sum(part.known + part.new for plan in plans for part in plan) * steps
    done = 0
    latents: List[Tensor] = []
    with stage("load transformer", device, stats):
        network = models.load_transformer(device, dtype)
    try:
        ezconfig.check_sizes(settings, network.sizes)
        with stage("sample", device, stats):
            for number, (sound, plan) in enumerate(zip(sounds, plans)):
                generator = torch.Generator(device="cpu").manual_seed(int(sound.seed))
                pair = torch.stack((hidden[number], hidden[-1])).to(device)
                context = network.prepare_context(pair)
                context_mask = torch.stack((mask[number], mask[-1])).to(device)
                made: Optional[Tensor] = None
                for part in plan:
                    frames = part.known + part.new

                    def before_step(frames=frames) -> None:
                        nonlocal done
                        report(_ENCODED + (_SAMPLED - _ENCODED) * done / total,
                               f"Generating sound {number + 1}/{len(sounds)}")
                        cancel.check()
                        done += frames

                    known = None if made is None else made[..., made.shape[-1] - part.known:]
                    first = made is None
                    piece = _sample_window(network, schedule, timesteps, steps, context, context_mask, known,
                                           part.new, cfg if first else CONTINUE_CFG,
                                           rescale if first else CONTINUE_RESCALE, eta, generator, device,
                                           before_step)
                    piece = piece[..., part.known:].float().cpu()
                    made = piece if made is None else torch.cat((made, piece), dim=-1)
                latents.append(made / settings.latent_scale - settings.latent_shift)
    finally:
        if models.recycle is not None:
            models.recycle(network)
        del network
        release(device)

    # 3. Decoding, 24 kHz -> 48 kHz, exactly seconds + 2 handles.
    cancel.check()
    report(_SAMPLED, "Decoding the sounds")
    results: List[Tensor] = []
    with stage("load vae", device, stats):
        decoder = models.load_vae(device)
    try:
        with stage("decode", device, stats):
            for sound, latent in zip(sounds, latents):
                waveform = vae.decode(decoder, latent, cancel.check)
                samples = round((sound.seconds + 2 * HANDLE) * OUTPUT_RATE)
                waveform = _limit_peak(to_output_rate(waveform, settings.sample_rate, samples))
                if not torch.isfinite(waveform).all():
                    raise RuntimeError("EzAudio produced non-finite audio samples.")
                results.append(waveform)
    finally:
        if models.recycle is not None:
            models.recycle(decoder)
        del decoder
        release(device)
    report(1.0, "Sound ready")
    return results


def generate(bundle: str | Path, prompt: str, negative: str, seconds: float, seed: int, *,
             device: torch.device | str = "cuda", dtype: Optional[torch.dtype] = None, steps: int = STEPS,
             cfg: float = CFG, rescale: float = RESCALE, eta: float = ETA, min_seconds: float = 0.0,
             progress_cb: Optional[Progress] = None, cancel: Optional[CancelToken] = None, cache=None,
             stats: Optional[Stats] = None) -> Tuple[Tensor, int]:
    """What `Engine.audio_bg` returns: mono CPU float32 [1, n] at 48 kHz including the half second of handles, and
    the rate. `bundle` is the folder of `EzAudioFiles.from_bundle`; `cache` (a ModelCache) keeps the networks in
    host memory between calls."""
    cancel = cancel or CancelToken()
    cancel.check()  # before the files are even looked at
    models = EzAudioModels.from_files(EzAudioFiles.from_bundle(bundle), cache)
    waveform, = generate_many(models, [Sound(prompt, seconds, seed)], negative, device=device, dtype=dtype,
                              steps=steps, cfg=cfg, rescale=rescale, eta=eta, min_seconds=min_seconds,
                              progress_cb=progress_cb, cancel=cancel, stats=stats)
    return waveform, OUTPUT_RATE
