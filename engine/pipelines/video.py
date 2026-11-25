# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Wan 2.2 A14B text-to-video and image-to-video in the one mode of the product: the LightX2V 4-step distillation.

What runs: the UMT5 encoder turns the prompt into the conditioning; (image-to-video) the VAE encodes the start picture
followed by gray frames into the extra input channels (`models.wan22.conditioning`); the high-noise expert runs the
first `split` Euler steps and the low-noise expert the rest, each with its LightX2V LoRA; the VAE decodes. The
settings are those of LightX2V's distilled Wan 2.2 configurations (ModelTC/LightX2V, Apache-2.0,
`configs/distill/wan22/wan_moe_t2v_distill_lora.json` and `wan_moe_i2v_distill_with_lora.json`): 4 steps at the
timesteps 1000, 750, 500 and 250 (`sampling.flow.spaced_sigmas`), shift 5, the expert switch after step 2, and no
classifier-free guidance, so every step is one forward pass with the prompt.

The networks are loaded one after another to keep the peak low: the text encoder encodes the prompt and is
released, the VAE encodes the start picture and is released, each expert runs its steps and is released, and the
VAE is loaded again to decode. Prompt encodings can be kept between calls (`embeddings`): the next part of a clip,
which continues from the last frame with the same prompt, skips the text encoder, and a Timeline job that announced
the prompts of its scenes (`core.memory.EmbeddingCache.expect`) has them all encoded in its first scene's encoder load.
"""
from __future__ import annotations

import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..core import attention as attention_backend
from ..core import streaming
from ..core.cancel import CancelToken
from ..core.stages import StatsCb, release, stage
from ..formats import lora as lora_format
from ..models.wan22 import conditioning, text, transformer, vae
from ..sampling import flow, samplers

LATENT_CHANNELS = 16
PIXELS_PER_LATENT = 8
SIZE_MULTIPLE = 16  # the transformer patches the latent 2 x 2
HIGH, LOW = "high", "low"

# The start sequence's latent frames the VAE computes; the gray padding after them settles, and the last computed
# frame stands for the rest (`vae.encode` `computed`). Measured on three start pictures (a scene's frame, another
# frame, noise): from the 16th latent frame on, the computed frames differ from it by at most 0.021 and 0.0045 RMS (in
# the sampler's normalised units, values of about +-3), as much as the start picture's latent changes when cuDNN picks
# another algorithm; encoding 61 of 97 frames instead of all of them saves about 1.5 s.
START_LATENT_FRAMES = 16
ProgressCb = Callable[[float, str], None]
_PREPARED, _SAMPLED = 0.08, 0.92  # share of the progress bar: loading and encoding, the steps, decoding


@dataclass(frozen=True)
class WanModels:
    """How to obtain the networks. The product uses `from_files`; tests pass tiny random models."""
    load_text: Callable[[torch.device, torch.dtype], Tuple[object, torch.nn.Module]]  # (tokenizer, encoder)
    load_transformer: Callable[[str, torch.device, torch.dtype], torch.nn.Module]  # expert "high" or "low"
    load_vae: Callable[[torch.device, torch.dtype], torch.nn.Module]
    kind: Optional[str] = None  # "i2v" or "t2v" when known: lets generate() refuse a bad request before loading
    recycle: Optional[Callable[[torch.nn.Module], None]] = None
    text_id: Optional[tuple] = None  # identity of the text encoder's files: prompt encodings are kept under it
    text_files: Tuple[Path, ...] = ()  # the text encoder's files, read ahead when a prompt has to be encoded
    expert_files: Tuple[Path, ...] = ()  # the high- and low-noise experts' files, read ahead in the order they run

    @classmethod
    def from_files(cls, text_file: str | Path, high_file: str | Path, low_file: str | Path, vae_file: str | Path,
                   kind: str = "i2v", cache=None) -> "WanModels":
        from ..core.memory import file_identity

        text.check_format(text_file)
        experts = {HIGH: Path(high_file), LOW: Path(low_file)}
        # The experts run in turn (high, low, then high again for the next part or scene). When the cache cannot
        # hold both, each would push the other out before it runs again: neither is kept (each loads from its file).
        keep_experts = cache is not None and cache.fits(*experts.values())

        def load(loader, path, role, device, dtype, spare=False, keep=True):
            if cache is None:
                return loader(path, device, dtype)
            return cache.load(path, role, loader, device, dtype, spare=spare, keep=keep)

        def load_expert(path, device, dtype):
            return transformer.load_transformer(path, device, dtype, kind=kind)

        return cls(
            load_text=lambda device, dtype: (text.load_tokenizer(text.read_spiece(text_file)),
                                             load(text.load_text_encoder, text_file, "text", device, dtype, True)),
            load_transformer=lambda expert, device, dtype: load(load_expert, experts[expert], kind, device, dtype,
                                                                 keep=keep_experts),
            load_vae=lambda device, dtype: load(vae.load_vae, vae_file, "vae", device, dtype), kind=kind,
            recycle=None if cache is None else cache.release,
            text_id=tuple(file_identity(part) for part in text.files(text_file)),
            text_files=tuple(text.files(text_file)), expert_files=(experts[HIGH], experts[LOW]))


def latent_shape(width: int, height: int, length: int) -> Tuple[int, int, int, int, int]:
    return (1, LATENT_CHANNELS, (length - 1) // conditioning.TEMPORAL_SCALE + 1, height // PIXELS_PER_LATENT,
            width // PIXELS_PER_LATENT)


def _denoising_attention(device: torch.device):
    if device.type != "cuda" or not torch.backends.cuda.cudnn_sdp_enabled():
        return nullcontext()
    # For Wan's long sequences cuDNN is faster than the memory-efficient kernel.
    # PyTorch can still dispatch to a different kernel when cuDNN is unavailable for a shape.
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True)


def _picture(image: Image.Image) -> Tensor:
    import numpy as np

    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]


def _read_text_files(files) -> None:
    """Read the text encoder's files into the file cache at disk speed, one thread per file, while the encoder loads:
    its memory-mapped bf16 shards otherwise come in by page faults at about 0.9 GB/s (12.6-13.4 s for UMT5's 11.4 GB,
    2026-11-15). Pages already cached cost little; a file that cannot be read is only not read ahead."""
    from ..core import readahead
    import threading

    def read(path) -> None:
        try:
            readahead.read_file(path)
        except OSError:
            pass

    for path in files:
        if Path(path).is_file() and Path(path).stat().st_size > (64 << 20):
            threading.Thread(target=read, args=(path,), name="text-read-ahead", daemon=True).start()


# Physical memory beside the two experts' files that reading them ahead leaves for the text encoder, the job and the
# system (both FP8 experts are 28 GiB: read ahead on this 64 GB test PC, not on a 32 GB one).
EXPERTS_AHEAD_ROOM = 24 << 30


def _read_experts_ahead(models: WanModels, loras: dict, after: threading.Event) -> Optional[threading.Thread]:
    """Read the experts' files, each followed by its LoRAs, into the file cache at disk speed in the order they run,
    starting once `after` is set (the text encoder has read its own files): the high-noise expert while the prompt
    and the start picture are encoded, the low-noise one while the high-noise expert computes. Without it each
    expert's first step pulls its 12-15 GB in by page faults (about 7 s for the high- and 10 s for the low-noise FP8
    expert of a Timeline scene, 2026-11-15). Only when the machine's memory holds them beside EXPERTS_AHEAD_ROOM."""
    from ..core import readahead

    files = []
    for expert, path in zip((HIGH, LOW), getattr(models, "expert_files", ())):
        files += [Path(path), *(Path(lora) for lora, _ in loras.get(expert, ()))]
    try:
        size = sum(path.stat().st_size for path in files if path.is_file())
    except OSError:
        return None
    if not size or readahead.physical_memory() < size + EXPERTS_AHEAD_ROOM:
        return None
    return readahead.ahead(files, wait=after, name="wan-experts-ahead")


# Physical memory beside the next stage's files that reading them ahead leaves for the network computing meanwhile
# (a 15 GB expert or the 20 GB talking network), the job and the system.
NEXT_STAGE_ROOM = 24 << 30
NEXT_STAGE_PATIENCE = 120.0  # s a next stage's file waits for room before the stage reads it by page faults


def read_next_stage_ahead(files) -> Optional[threading.Thread]:
    """Read the next stage's files (`then` of `generate` and of `pipelines.talk.generate`) into the file cache at
    background disk priority while the GPU computes the last steps of this one: the next scene's starting frame
    otherwise waited about 12-14 s for Qwen-Image-Edit's 31 GB (2026-11-15). Only when the machine's memory holds them
    beside NEXT_STAGE_ROOM."""
    from ..core import readahead

    files = [Path(path) for path in files or () if path]
    try:
        size = sum(path.stat().st_size for path in files if path.is_file())
    except OSError:
        return None
    if not size or readahead.physical_memory() < size + NEXT_STAGE_ROOM:
        return None
    # it waits for room the stage computing now gives back when it ends (its network, 15-20 GB)
    return readahead.ahead(files, background=True, name="next-stage-ahead", patience=NEXT_STAGE_PATIENCE)


TEXT_ROOM_MARGIN = 1 << 30  # GPU memory left over when the text encoder's blocks stay on the GPU between prompts


def _blocks_fit(blocks: torch.nn.Module, device: torch.device) -> bool:
    """Whether every block of a streamed encoder can stay on `device` at once, with a margin for the activations
    (the per-process limit of `core.memory.limit_cuda` included)."""
    if device.type != "cuda":
        return False
    from ..core.memory import model_bytes

    index = device.index if device.index is not None else torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    allowed = total * torch.cuda.get_per_process_memory_fraction(index) - torch.cuda.memory_reserved(index)
    return model_bytes(blocks) + TEXT_ROOM_MARGIN <= min(free, allowed)


def _encode_text(models: WanModels, prompt: str, device: torch.device, dtype: torch.dtype, stream_text: bool,
                 stats: Optional[StatsCb], embeddings, report: ProgressCb) -> Tensor:
    """The conditioning of the prompt, on `device`. A prompt found in `embeddings` (an EmbeddingCache) is not
    encoded again; otherwise the encoder is loaded for it, and for the prompts the cache expects
    (`EmbeddingCache.expect`: the other scenes of a Timeline job) that it does not hold yet, and released right
    after. Each prompt is encoded on its own, exactly as it would be alone; when there are several and the GPU
    has room, the streamed blocks stay on it after the first prompt, so the others do not stream them again."""
    def key_of(item: str):
        return (models.text_id, dtype, item) if embeddings is not None and models.text_id else None

    key = key_of(prompt)
    found = embeddings.get(key) if key is not None else None
    if key is not None:
        embeddings.consume(prompt)  # served now: a later encoder load need not encode it again
    if found is not None:
        report(0.0, "Reusing the prompt's text encoding")
        return found.to(device)
    prompts = [prompt]
    if key is not None:
        prompts += [item for item in embeddings.expected() if item != prompt and key_of(item) not in embeddings]
    report(0.0, "Loading the text encoder" if len(prompts) == 1 else
           f"Loading the text encoder for {len(prompts)} prompts")
    _read_text_files(getattr(models, "text_files", ()))
    with stage("text encoder", device, stats):
        tokenizer, encoder = models.load_text(torch.device("cpu") if stream_text else device, dtype)
        remove_text_stream = None
        try:
            if stream_text:
                encoder.shared.to(device)
                resident = len(encoder.encoder.block) if len(prompts) > 1 and _blocks_fit(
                    encoder.encoder.block, device) else 0
                remove_text_stream = streaming.stream_model(encoder.encoder, device, blocks="block", pin=False,
                                                            resident=resident)
            for item in prompts:
                encoded = text.encode_prompt(tokenizer, encoder, item).to(device)
                if item == prompt:
                    found = encoded
                if key is not None:
                    embeddings.put(key_of(item), encoded)
        finally:
            if remove_text_stream is not None:
                remove_text_stream()
            if models.recycle is not None:
                models.recycle(encoder)
            del encoder
            release(device)
    return found


def _check(width: int, height: int, length: int, steps: int) -> None:
    if width % SIZE_MULTIPLE or height % SIZE_MULTIPLE or width < SIZE_MULTIPLE or height < SIZE_MULTIPLE:
        raise ValueError(f"width and height must be multiples of {SIZE_MULTIPLE}, got {width}x{height}")
    if length < 1 or (length - 1) % conditioning.TEMPORAL_SCALE:
        raise ValueError(f"length must be 4k + 1 frames, got {length}")
    if steps < 2:
        raise ValueError("steps must be at least 2: each expert needs a step")


def generate(models: WanModels, prompt: str, width: int, height: int, length: int, seed: int,
             steps: int, shift: float, start_image: Optional[Image.Image] = None,
             split: Optional[int] = None, loras: Optional[dict] = None, device: torch.device | str = "cuda",
             dtype: torch.dtype = torch.bfloat16, vae_dtype: Optional[torch.dtype] = None,
             progress_cb: Optional[ProgressCb] = None, cancel: Optional[CancelToken] = None,
             stats: Optional[StatsCb] = None, lora_rounding: str = "nearest",
             stream: Optional[dict] = None, stream_text: bool = False, embeddings=None,
             attention: str = attention_backend.SDPA, then=()) -> Tensor:
    """A clip of `length` frames (4k + 1) as float32 pictures [N, 3, H, W] in 0..1 on the CPU.

    `start_image` makes it image-to-video (the models must be the I2V pair), otherwise text-to-video.
    `split` is the first step of the low-noise expert (default: half of the steps); `loras` maps "high" and
    "low" to lists of (file, strength). `shift` is the flow shift (5 for the LightX2V LoRAs).
    `stream` (a dict of `core.streaming.stream_model` options, e.g. {"prefetch": 1}) runs the experts with
    their blocks streamed from host memory instead of resident on the GPU, for cards that cannot hold one.
    `stream_text` likewise stages the UMT5 layers on the compute device. `embeddings` (a
    core.memory.EmbeddingCache) keeps prompt encodings, so a prompt encoded before skips the text encoder.
    `attention` is the experts' attention ("sdpa", or "auto" / "sage" for SageAttention where it can run, see
    `core.attention.resolve`). `then`: files the next stage reads first (the next scene's starting frame), read ahead
    while the low-noise expert computes and the disk is idle (`read_next_stage_ahead`)."""
    _check(width, height, length, steps)
    device = torch.device(device)
    backend = attention_backend.resolve(attention, device)
    cancel = cancel or CancelToken()
    split = steps // 2 if split is None else split
    stages = flow.split_experts(steps, split)  # refuses a split outside the steps now, not after the encoders ran
    if start_image is not None and models.kind == "t2v":
        raise ValueError("a start picture needs the image-to-video networks; these are text-to-video networks")
    loras = loras or {}

    def report(fraction: float, message: str) -> None:
        if progress_cb is not None:
            progress_cb(fraction, message)

    # 1. Prompt -> embedding (a prompt encoded recently is reused), then the text encoder is released. The experts'
    # files are read ahead from then on.
    cancel.check()
    text_done = threading.Event()
    _read_experts_ahead(models, loras, text_done)
    try:
        context = _encode_text(models, prompt, device, dtype, stream_text, stats, embeddings, report)
    finally:
        text_done.set()
    cancel.check()

    # 2. Image-to-video: the start picture and gray frames go through the VAE for the extra channels.
    concat = None
    if start_image is not None:
        report(_PREPARED / 2, "Encoding the start picture")
        with stage("vae encode", device, stats):
            autoencoder = models.load_vae(device, vae_dtype or dtype)
            try:
                frames = conditioning.start_sequence(_picture(start_image), length, width, height)
                with attention_backend.use(backend):  # the VAE's fused norms run with the other Triton kernels
                    try:
                        raw = vae.encode(autoencoder, frames, computed=START_LATENT_FRAMES).to(device)
                    except torch.OutOfMemoryError:
                        raw = None
                    if raw is None:
                        release(device)
                        raw = vae.encode(autoencoder, frames, tiled=True).to(device)
                concat = conditioning.i2v_concat(raw, conditioning.known_latent_frames(1),
                                                 autoencoder.config.latents_mean, autoencoder.config.latents_std)
            finally:
                if models.recycle is not None:
                    models.recycle(autoencoder)
                del autoencoder
                release(device)
        cancel.check()

    # 3. Denoise from float32 noise drawn on the CPU from the seed (the same on any GPU).
    sigmas = flow.spaced_sigmas(shift, steps)
    x = flow.initial_latent(sigmas, flow.make_noise(seed, latent_shape(width, height, length))).to(device)
    report(_PREPARED, "Generating")
    for part, expert in zip(stages, (HIGH, LOW)):
        remove_loras = remove_stream = None
        if expert == LOW:
            read_next_stage_ahead(then)
        with stage(f"{expert} noise expert", device, stats):
            network = models.load_transformer(expert, torch.device("cpu") if stream is not None else device, dtype)
            try:
                if stream is not None:
                    remove_stream = streaming.stream_model(network, device, **stream)
                if loras.get(expert):
                    names = [name for name, _ in network.named_modules()]
                    remove_loras = lora_format.attach(
                        network, [(lora_format.load_lora(path, names).to(device, dtype), strength)
                                  for path, strength in loras[expert]], rounding=lora_rounding, seed=seed)

                def denoise(state: Tensor, sigma: Tensor) -> Tensor:
                    return transformer.predict_denoised(network, state, sigma, context, concat)

                def before_step(step: int, total: int, first: int = part.start) -> None:
                    done = first + step
                    if done > 0:
                        report(_PREPARED + (_SAMPLED - _PREPARED) * done / steps, f"Step {done}/{steps}")
                    cancel.check()  # after the report, so a cancel requested from progress_cb stops the very next step

                with torch.inference_mode(), _denoising_attention(device), attention_backend.use(backend):
                    x = samplers.euler(denoise, x, part.sigmas(sigmas), before_step)
            finally:
                if remove_loras is not None:
                    remove_loras()
                if remove_stream is not None:
                    remove_stream()
                remove_loras = remove_stream = None  # the bound methods keep the streamer and its pinned host copies alive
                if models.recycle is not None:
                    models.recycle(network)
                del network
                release(device)
        cancel.check()
    if stream is not None:
        streaming.release_host_cache()  # the pinned copies of both experts are not needed any more
    report(_SAMPLED, "Decoding")

    # 4. Latent -> pictures.
    with stage("vae", device, stats):
        autoencoder = models.load_vae(device, vae_dtype or dtype)
        try:
            with attention_backend.use(backend):
                try:
                    clip = vae.decode(autoencoder, x)
                except torch.OutOfMemoryError:
                    clip = None  # retried below, once the failed attempt's memory can be released
                if clip is None:
                    release(device)
                    clip = vae.decode(autoencoder, x, tiled=True)
        finally:
            if models.recycle is not None:
                models.recycle(autoencoder)
            del autoencoder
            release(device)
    report(1.0, "Done")
    return clip
