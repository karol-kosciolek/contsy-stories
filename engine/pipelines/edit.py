# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Picture composition with Qwen-Image-Edit-2511: one to three reference pictures and a prompt make a new picture.

What runs (diffusers 0.42 `QwenImageEditPlusPipeline`, Apache-2.0, describes the same model): the Qwen2.5-VL
encoder reads the prompt together with a copy of about 384 x 384 pixels of every picture
(`models.qwen_edit.text.encode_prompt`, which gets the original PIL pictures and scales them itself); the VAE
encodes a copy of about 1024 x 1024 pixels of every picture into a reference latent (`prepare_reference`); the
transformer denoises a latent of the result's size (noise only) with the reference latents beside it
(`models.qwen_edit.transformer`) in 4 Euler steps over evenly spaced sigmas with flow shift 3.1 and the
Qwen-Image-Edit-2511 Lightning LoRA at strength 1 (ModelTC/Qwen-Image-Lightning, Apache-2.0, a 4-step
distillation without guidance, so every step is one forward pass); the VAE decodes the result.

Contsy's own choices where the diffusers pipeline differs: shift 3.1 where the Lightning authors'
`generate_with_diffusers.py` uses shift 3 (exponential time shift log 3); the VAE copy with sides rounded to
multiples of 8 and scaled with torch's "area" interpolation where diffusers rounds to 32 and uses its image
processor.

The networks run one after another to keep the peak low: the text encoder encodes the prompt and is released, the
VAE encodes the references (and stays for the decoding, it is small), the transformer runs the steps (optionally
with its blocks streamed from host memory for cards that cannot hold it), the VAE decodes last.
"""
from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import Tensor

from ..core import readahead, streaming
from ..core.cancel import CancelToken
from ..core.quant import QLinear
from ..core.stages import StatsCb, release, stage
from ..formats import lora as lora_format
from ..models.qwen_edit import text, transformer, vae
from ..sampling import flow, samplers

MAX_IMAGES = text.MAX_IMAGES  # reference pictures per pass; more are composed in several passes by the caller
STEPS = 4  # the Lightning LoRA is a 4-step distillation
SHIFT = 3.1  # flow shift (sampling.flow)
LATENT_CHANNELS = 16
PIXELS_PER_LATENT = 8
SIZE_MULTIPLE = 8  # one latent cell is 8 x 8 pixels; an odd latent side is handled by the transformer
REFERENCE_AREA = 1024 * 1024  # pixels of a picture for the VAE (the encoder copy has 384 * 384)
REFERENCE_MULTIPLE = 8

ProgressCb = Callable[[float, str], None]
Lora = Tuple[str | Path, float]
EncodeHook = Callable[[object, torch.nn.Module, str, Sequence[Image.Image]], Tensor]

# Share of the progress bar: loading and encoding, the denoising steps, decoding.
_ENCODED, _DENOISED = 0.08, 0.92


@dataclass(frozen=True)
class EditModels:
    """How to obtain the networks. The product uses `from_files`; tests pass tiny random models.

    `encode(tokenizer, encoder, prompt, images) -> [1, L, 3584]` turns the prompt and the reference pictures into
    the prompt embedding; None means `models.qwen_edit.text.encode_prompt`. `transformer_file` is the file
    `load_transformer` reads, when there is one: a streamed transformer's file is read ahead (see _Preload)."""
    load_text: Callable[[torch.device, torch.dtype], Tuple[object, torch.nn.Module]]  # (tokenizer, encoder)
    load_transformer: Callable[[torch.device, torch.dtype], torch.nn.Module]
    load_vae: Callable[[torch.device, torch.dtype], torch.nn.Module]
    encode: Optional[EncodeHook] = None
    transformer_file: Optional[Path] = None

    @classmethod
    def from_files(cls, transformer_file: str | Path, text_encoder_file: str | Path, vae_file: str | Path,
                   tokenizer_dir: str | Path) -> "EditModels":
        def load_text(device: torch.device, dtype: torch.dtype):
            return text.load_tokenizer(tokenizer_dir), text.load_text_encoder(text_encoder_file, device, dtype)

        return cls(
            load_text=load_text,
            load_transformer=lambda device, dtype: transformer.load_transformer(transformer_file, device, dtype),
            load_vae=lambda device, dtype: vae.load_vae(vae_file, device, dtype),
            transformer_file=Path(transformer_file))


def reference_size(width: int, height: int, area: Optional[int] = None) -> Tuple[int, int]:
    """(width, height) of a reference picture after scaling to `area` pixels (default `REFERENCE_AREA`), the sides
    rounded to multiples of 8 with Python's `round`."""
    area = REFERENCE_AREA if area is None else area
    scale = math.sqrt(area / (width * height))
    return (round(width * scale / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE,
            round(height * scale / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)


def prepare_reference(image: Image.Image, area: Optional[int] = None) -> Tensor:
    """A reference picture as the float32 tensor [1, 3, H, W] in 0..1 that goes to the VAE.

    The EXIF rotation is applied and the picture converted to RGB without touching the colours (an alpha channel
    is dropped), then scaled to about `area` pixels with torch's "area" interpolation, both sides rounded to
    multiples of 8."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = reference_size(image.width, image.height, area)
    if width < REFERENCE_MULTIPLE or height < REFERENCE_MULTIPLE:
        raise ValueError(f"a {image.width}x{image.height} picture is too thin to scale to {area or REFERENCE_AREA} pixels")
    pixels = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    return F.interpolate(pixels, size=(height, width), mode="area")


def to_image(pixels: Tensor) -> Image.Image:
    """[1, 3, H, W] floats in 0..1 to an RGB image; values are truncated to 8 bits (not rounded)."""
    array = (pixels[0] * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, "RGB")


def _check(images: Sequence[Image.Image], width: int, height: int, steps: int, shift: float) -> None:
    if not 1 <= len(images) <= MAX_IMAGES:
        raise ValueError(f"one composition pass takes 1 to {MAX_IMAGES} reference pictures, got {len(images)}; "
                         "compose more pictures in several passes")
    for index, image in enumerate(images):
        if not isinstance(image, Image.Image):
            raise TypeError(f"reference picture {index + 1} is a {type(image).__name__}, not a PIL image")
    if width % SIZE_MULTIPLE or height % SIZE_MULTIPLE or width < SIZE_MULTIPLE or height < SIZE_MULTIPLE:
        raise ValueError(f"width and height must be multiples of {SIZE_MULTIPLE}, got {width}x{height}")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if shift <= 0:
        raise ValueError("shift must be positive")


PAGE_BYTES = 4096  # the memory page: touching one byte brings in the whole page
TOUCH_THREADS = 4  # page faults are taken in parallel


def touch_pages(network: torch.nn.Module) -> int:
    """Read one byte of every page of the quantized weights a GGUF transformer keeps in its memory-mapped file, so
    the operating system maps them into the process now rather than one page fault at a time during the first
    sampling step (about 4 s for the 21 GB of Qwen-Image-Edit's Q8_0 file, 2026-11-15). Returns the bytes covered."""
    # numpy on a few threads of its own, without the GIL: a torch reduction would take torch's CPU threads from the
    # main thread's work (the reference pictures took 6 s instead of 1 s beside it)
    weights = [module.qweight.data for module in network.modules() if isinstance(module, QLinear)]
    arrays = [data.reshape(-1).view(torch.uint8).numpy() for data in weights
              if data.device.type == "cpu" and data.numel()]

    def touch(array) -> int:
        int(array[::PAGE_BYTES].sum(dtype=np.int64))
        return array.size

    with ThreadPoolExecutor(TOUCH_THREADS, thread_name_prefix="edit-touch") as pool:
        return sum(pool.map(touch, arrays))


def import_networks() -> None:
    """Import the diffusers classes of the transformer and the VAE on the calling (main) thread before _Preload's
    worker builds the transformer: two threads that import diffusers' lazily loaded modules at the same time can see
    one of them half initialised."""
    from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel  # noqa: F401


class _Preload:
    """The streamed transformer, its file read ahead (core.readahead, when the file is known: a page fault reads the
    disk in small pieces, about 0.8 GB/s for the 21 GB file after Wan had pushed it out of the cache, 2026-11-15),
    the network loaded and its pages brought in (touch_pages), on a worker thread while the text encoder and the VAE
    run on the main one. `take()` waits for it; `drop()` lets it go unused (an error, a cancel) without waiting."""

    def __init__(self, load: Callable[[], torch.nn.Module], file: Optional[str | Path] = None):
        self._lock = threading.Lock()
        self._network: Optional[torch.nn.Module] = None
        self._error: Optional[BaseException] = None
        self._dropped = False
        self._thread = threading.Thread(target=self._run, args=(load, file), name="edit-preload", daemon=True)
        self._thread.start()

    def _run(self, load, file) -> None:
        try:
            if file is not None:
                # at background disk priority (the text encoder, loading meanwhile, reads first); a file the scene
                # already read ahead is waited for or left alone instead of being read again (about 3 s of copying)
                readahead.settle(file)
            network = load()
            touch_pages(network)
        except BaseException as error:  # noqa: BLE001 - handed to take()
            with self._lock:
                self._error = None if self._dropped else error
            return
        with self._lock:
            if not self._dropped:
                self._network = network
        del network

    def take(self) -> torch.nn.Module:
        self._thread.join()
        with self._lock:
            network, error = self._network, self._error
            self._network = self._error = None
        if error is not None:
            raise error
        return network

    def drop(self) -> None:
        with self._lock:
            self._dropped = True
            self._network = self._error = None


def _encode_reference(autoencoder: torch.nn.Module, picture: Tensor, device: torch.device) -> Tensor:
    """One reference latent; an out-of-memory error is retried once with tiles, after the failed attempt is released."""
    try:
        return vae.encode(autoencoder, picture)
    except torch.OutOfMemoryError:
        pass
    release(device)
    return vae.encode(autoencoder, picture, tiled=True)


def edit(models: EditModels, images: Sequence[Image.Image], prompt: str, width: int, height: int, seed: int,
         steps: int = STEPS, shift: float = SHIFT, loras: Sequence[Lora] = (), device: torch.device | str = "cuda", dtype: torch.dtype = torch.bfloat16,
         vae_dtype: Optional[torch.dtype] = None, progress_cb: Optional[ProgressCb] = None,
         cancel: Optional[CancelToken] = None, stats: Optional[StatsCb] = None, lora_rounding: str = "nearest",
         stream: Optional[dict] = None, stream_text: bool = False, then=()) -> Image.Image:
    """Compose one picture of exactly `width` x `height` pixels (multiples of 8) from 1 to 3 reference pictures.
    `then`: files the stage after the picture reads first, read ahead while the steps run
    (video.read_next_stage_ahead).

    `prompt` refers to the pictures as the text encoder expects ("... from image 2"). `loras` is a list of
    (file, strength) applied to the transformer (the product passes the Lightning LoRA at 1.0 with the defaults
    steps = 4, shift = 3.1). `stream` (a dict of
    `core.streaming.stream_model` options, e.g. {"prefetch": 1}) runs the transformer with its blocks streamed
    from host memory instead of resident on the GPU. `lora_rounding` is the mode of `engine.formats.lora.attach`.
    `stream_text` also stages the text and vision blocks (from pageable host memory) instead of keeping the entire
    encoder on CUDA.
    `cancel.check()` runs before every step and raises `Cancelled`; `progress_cb(fraction, text)` is called after
    loading/encoding, after every step and when the picture is done."""
    _check(images, width, height, steps, shift)
    encode_prompt = models.encode or text.encode_prompt
    images = [ImageOps.exif_transpose(image).convert("RGB") for image in images]
    device = torch.device(device)
    cancel = cancel or CancelToken()
    autoencoder = None
    preload = None

    def report(fraction: float, message: str) -> None:
        if progress_cb is not None:
            progress_cb(fraction, message)

    try:
        if stream is not None and device.type == "cuda":  # read the streamed transformer while the encoders run
            import_networks()
            preload = _Preload(lambda: models.load_transformer(torch.device("cpu"), dtype), models.transformer_file)
        # 1. Prompt and pictures -> embedding, then the text encoder is released.
        cancel.check()
        report(0.0, "Loading the text encoder")
        with stage("text encoder", device, stats):
            tokenizer, encoder = models.load_text(torch.device("cpu") if stream_text else device, dtype)
            try:
                with ExitStack() as hooks:
                    if stream_text:  # unpinned: pinning the encoder's 7 GB cost more than it saved
                        hooks.callback(streaming.stream_model(encoder.model.visual, device, blocks="blocks", pin=False))
                        hooks.callback(streaming.stream_model(encoder.model.language_model, device, blocks="layers",
                                                              pin=False))
                    context = encode_prompt(tokenizer, encoder, prompt, images).to(device)
            finally:
                del encoder
                release(device)
                if stream_text:
                    streaming.release_host_cache()
        cancel.check()

        # 2. The reference pictures become latents; the VAE stays loaded for the decoding at the end.
        report(_ENCODED / 4, "Encoding the reference pictures")
        with stage("vae encode", device, stats):
            autoencoder = models.load_vae(device, vae_dtype or dtype)
            references = []
            for image in images:
                cancel.check()
                references.append(_encode_reference(autoencoder, prepare_reference(image), device).to(device))
            release(device)  # the activations of the encoding must not sit in the cache while the transformer loads
        cancel.check()

        # 3. Denoise from float32 noise drawn on the CPU from the seed (the same on any GPU).
        sigmas = flow.spaced_sigmas(shift, steps)
        shape = (1, LATENT_CHANNELS, height // PIXELS_PER_LATENT, width // PIXELS_PER_LATENT)
        latent = flow.initial_latent(sigmas, flow.make_noise(seed, shape)).to(device)
        report(_ENCODED / 2, "Loading the image model")
        with stage("transformer", device, stats):
            if preload is not None:
                network, preload = preload.take(), None
            else:
                network = models.load_transformer(torch.device("cpu") if stream is not None else device, dtype)
            remove_loras = remove_stream = None
            try:
                if stream is not None:
                    remove_stream = streaming.stream_model(network, device, **{"blocks": "transformer_blocks", **stream})
                if loras:
                    names = [name for name, _ in network.named_modules()]
                    remove_loras = lora_format.attach(
                        network, [(lora_format.load_lora(path, names).to(device, dtype), strength)
                                  for path, strength in loras], rounding=lora_rounding, seed=seed)
                report(_ENCODED, "Generating")
                from .video import read_next_stage_ahead

                read_next_stage_ahead(then)

                def denoise_step(x: Tensor, sigma: Tensor) -> Tensor:
                    return transformer.predict_denoised(network, x, sigma, context, references)

                def before_step(step: int, total: int) -> None:
                    if step > 0:  # steps 0..step-1 are done
                        report(_ENCODED + (_DENOISED - _ENCODED) * step / total, f"Step {step}/{total}")
                    cancel.check()  # after the report, so a cancel requested from progress_cb stops the very next step

                with torch.inference_mode():
                    latent = samplers.euler(denoise_step, latent, sigmas, before_step)
                report(_DENOISED, f"Step {steps}/{steps}")
            finally:
                if remove_loras is not None:
                    remove_loras()
                if remove_stream is not None:
                    remove_stream()
                remove_loras = remove_stream = None  # the bound methods keep the streamer and its pinned host copies alive
                del network
                release(device)
        cancel.check()
        report(_DENOISED, "Decoding")

        # 4. Latent -> picture.
        with stage("vae", device, stats):
            try:
                pixels = vae.decode(autoencoder, latent)
            except torch.OutOfMemoryError:
                pixels = None  # retried below, once the failed attempt's memory can be released
            if pixels is None:
                release(device)
                pixels = vae.decode(autoencoder, latent, tiled=True)
        report(1.0, "Done")
        return to_image(pixels)
    finally:
        if preload is not None:
            preload.drop()
        del autoencoder
        release(device)
        if stream is not None:
            streaming.release_host_cache()
