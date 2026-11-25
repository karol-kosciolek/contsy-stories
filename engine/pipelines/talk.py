# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""A talking clip of an animated character with InfiniteTalk (MeiGen-AI/InfiniteTalk, Apache-2.0, `wan/multitalk.py`
`generate_infinitetalk`, image input): the picture of the character and its speech in, frames at 25 FPS out.

What runs, one network at a time where it can:
1. the speech features (`models.infinitetalk.audio`, wav2vec 2.0 on the CPU) for every frame at 25 FPS;
2. the prompt through the UMT5 encoder (`video._encode_text`, so a prompt encoded before is reused), padded to 512;
3. the picture through CLIP ViT-H (`models.infinitetalk.clip`) for the image tokens, and with gray frames after it
   through the VAE for the image-to-video channels (`models.wan22.conditioning`, mask first);
4. the clip in chunks of 81 frames (a shorter clip in one chunk of the 4k + 1 frames that cover it). Every chunk
   starts from noise drawn from `seed + chunk`, and before and after every step its first latent frames are the
   clean latent of the frames it continues from: the picture for the first chunk, the last 9 frames of the previous
   chunk (3 latent frames) after that, which the chunk then leaves out of the result. The conditioning channels and
   the image tokens stay those of the picture in every chunk. The steps are Euler steps of the velocity flow at the
   4-step distilled schedule (LightX2V: timesteps 1000, 750, 500, 250 at shift 5, no text guidance); with
   `audio_scale` other than 1 a second pass without the speech (zero features) gives audio guidance,
   `drop + audio_scale * (cond - drop)`, the reference's guidance when the text scale is 1.
5. every chunk is decoded by the VAE, which stays loaded for the job.
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from ..core import attention as attention_backend
from ..core import streaming
from ..core.cancel import CancelToken
from ..core.stages import StatsCb, release, stage
from ..models.infinitetalk import audio as speech_audio
from ..models.infinitetalk import clip as clip_model
from ..models.infinitetalk import model as talk_model
from ..models.wan22 import conditioning, text, vae
from ..sampling import flow
from .video import START_LATENT_FRAMES, _denoising_attention, _encode_text, _picture, read_next_stage_ahead

FPS = speech_audio.FPS
CHUNK, MOTION = 81, 9
LATENT_CHANNELS, PIXELS_PER_LATENT, SIZE_MULTIPLE = 16, 8, 16
STEPS, SHIFT = 4, 5.0
ProgressCb = Callable[[float, str], None]
_PREPARED, _SAMPLED = 0.1, 0.97


@dataclass(frozen=True)
class TalkModels:
    """How to obtain the networks. The product uses `from_files`; tests pass tiny random models."""
    load_text: Callable[[torch.device, torch.dtype], Tuple[object, torch.nn.Module]]
    load_transformer: Callable[[torch.device, torch.dtype], torch.nn.Module]
    load_vae: Callable[[torch.device, torch.dtype], torch.nn.Module]
    load_clip: Callable[[torch.device, torch.dtype], torch.nn.Module]
    speech_features: Callable[[np.ndarray, int], Tensor]  # 16 kHz speech, frames -> [frames, 12, 768]
    recycle: Optional[Callable[[torch.nn.Module], None]] = None
    text_id: Optional[tuple] = None
    text_files: Tuple[Path, ...] = ()     # read ahead when the prompt has to be encoded (video._encode_text)
    network_files: Tuple[Path, ...] = ()  # read ahead while the speech, the prompt and the picture are encoded

    @classmethod
    def from_files(cls, text_file: str | Path, base_files: Sequence[str | Path], adapter: str | Path,
                   clip_file: str | Path, vae_file: str | Path, wav2vec: str | Path, cache=None) -> "TalkModels":
        from ..core.memory import file_identity

        text.check_format(text_file)
        base_files = [Path(path) for path in base_files]

        def load(loader, path, role, device, dtype, spare=False, keep=True):
            if cache is None:
                return loader(path, device, dtype)
            return cache.load(path, role, loader, device, dtype, spare=spare, keep=keep)

        keep_network = cache is not None and cache.fits(*base_files, adapter)
        return cls(
            load_text=lambda device, dtype: (text.load_tokenizer(text.read_spiece(text_file)),
                                             load(text.load_text_encoder, text_file, "text", device, dtype, True)),
            load_transformer=lambda device, dtype: load(
                lambda _path, d, t: talk_model.load(base_files, adapter, d, t), adapter, "talk", device, dtype,
                keep=keep_network),
            load_vae=lambda device, dtype: load(vae.load_vae, vae_file, "vae", device, dtype),
            load_clip=lambda device, dtype: clip_model.load(clip_file, device, dtype),
            speech_features=lambda speech, frames: speech_audio.speech_features(wav2vec, speech, frames),
            recycle=None if cache is None else cache.release,
            text_id=tuple(file_identity(part) for part in text.files(text_file)),
            text_files=tuple(text.files(text_file)), network_files=_use_order(base_files, adapter))


def _use_order(base_files: Sequence[str | Path], adapter: str | Path) -> Tuple[Path, ...]:
    """The talking network's files in the order a step uses them, for reading them ahead: the base's parts outside its
    blocks (the embeddings), the audio adapter (every block uses it), then the blocks by number."""
    def place(path: Path):
        found = re.search(r"block_(\d+)", path.stem)
        return (1, int(found.group(1))) if found and "non_block" not in path.stem else (0, 0)

    return (*sorted((Path(path) for path in base_files if "non_block" in Path(path).stem), key=place), Path(adapter),
            *sorted((Path(path) for path in base_files if "non_block" not in Path(path).stem), key=place))


def _cover(frames: int) -> int:
    """The shortest 4k + 1 frames that hold `frames`."""
    return 4 * math.ceil((max(frames, 5) - 1) / 4) + 1


def plan(frames: int, chunk: int = CHUNK, motion: int = MOTION) -> List[int]:
    """The length of every chunk of a clip of `frames`: one chunk of the 4k + 1 frames that cover a short clip; else
    as few chunks of at most `chunk` frames (the length InfiniteTalk was trained on) as cover it, of the shortest
    4k + 1 length that does, each after the first adding `length - motion` new frames, and the last only as long as the
    frames it still has to add (a 4 s clip, 100 frames: 57 + 53, not 81 + 81 of which 53 frames were thrown away:
    43 s of compute per clip-second with full chunks on 2026-11-15)."""
    if frames <= chunk:
        return [_cover(frames)]
    count = 1 + math.ceil((frames - chunk) / (chunk - motion))
    length = min(_cover(motion + math.ceil((frames - motion) / count)), chunk)
    lengths = [length] * count
    lengths[-1] = _cover(frames - (length + (count - 2) * (length - motion)) + motion)
    return lengths


class _Background:
    """A computation on a thread of its own; `result()` waits for it and raises what it raised."""

    def __init__(self, work: Callable[[], object]):
        self._value = self._error = None
        self._thread = threading.Thread(target=self._run, args=(work,), name="talk-speech", daemon=True)
        self._thread.start()

    def _run(self, work) -> None:
        try:
            self._value = work()
        except BaseException as error:  # noqa: BLE001 - handed to result()
            self._error = error

    def result(self):
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._value


def _check(width: int, height: int, frames: int, steps: int) -> None:
    if width % SIZE_MULTIPLE or height % SIZE_MULTIPLE or width < SIZE_MULTIPLE or height < SIZE_MULTIPLE:
        raise ValueError(f"width and height must be multiples of {SIZE_MULTIPLE}, got {width}x{height}")
    if frames < 1:
        raise ValueError("a talking clip needs at least one frame")
    if steps < 1:
        raise ValueError("steps must be at least 1")


def _normalised(autoencoder, raw: Tensor) -> Tensor:
    config = autoencoder.config
    mean = torch.tensor(config.latents_mean, dtype=raw.dtype, device=raw.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(config.latents_std, dtype=raw.dtype, device=raw.device).view(1, -1, 1, 1, 1)
    return (raw - mean) / std


def _encode(autoencoder, frames: Tensor, device: torch.device, computed: Optional[int] = None) -> Tensor:
    try:
        return vae.encode(autoencoder, frames, computed=computed).to(device)
    except torch.OutOfMemoryError:
        pass
    release(device)
    return vae.encode(autoencoder, frames, tiled=True).to(device)


def _decode(autoencoder, latent: Tensor, device: torch.device) -> Tensor:
    try:
        return vae.decode(autoencoder, latent)
    except torch.OutOfMemoryError:
        pass
    release(device)
    return vae.decode(autoencoder, latent, tiled=True)


def generate(models: TalkModels, prompt: str, picture: Image.Image, speech: np.ndarray, frames: int, width: int,
             height: int, seed: int, steps: int = STEPS, shift: float = SHIFT, audio_scale: float = 1.0,
             chunk: int = CHUNK, motion: int = MOTION, device: torch.device | str = "cuda",
             dtype: torch.dtype = torch.bfloat16, progress_cb: Optional[ProgressCb] = None,
             cancel: Optional[CancelToken] = None, stats: Optional[StatsCb] = None,
             stream: Optional[dict] = None, stream_text: bool = False, embeddings=None,
             attention: str = attention_backend.SDPA, then=()) -> Tensor:
    """`frames` pictures [frames, 3, height, width] in 0..1 (float32, CPU) at 25 FPS of the character in `picture`
    saying `speech` (16 kHz mono float samples, the clip's clock from 0). See the module docstring. `then`: files the
    next stage reads first, read ahead while the last chunk computes (video.read_next_stage_ahead)."""
    _check(width, height, frames, steps)
    if (motion - 1) % conditioning.TEMPORAL_SCALE or (chunk - 1) % conditioning.TEMPORAL_SCALE or motion >= chunk:
        raise ValueError("chunk and motion must be 4k + 1 frames, motion shorter than chunk")
    device = torch.device(device)
    backend = attention_backend.resolve(attention, device)
    cancel = cancel or CancelToken()
    lengths = plan(frames, chunk, motion)
    count = len(lengths)

    def report(fraction: float, message: str) -> None:
        if progress_cb is not None:
            progress_cb(fraction, message)

    # The talking network's files are read into the file cache at disk speed while the speech, the prompt and the
    # picture are encoded (else the first step pulls 19.7 GB in by page faults, about 13 s on 2026-11-15)
    if models.network_files:
        from ..core import readahead

        readahead.ahead(models.network_files, background=True, name="talk-read-ahead")  # behind the encoders' reads
    # 1. Speech features for every frame the chunks make (a shorter sequence is mirrored on, as the reference does),
    # on the CPU on a thread of its own while the GPU encodes the prompt and the picture
    report(0.0, "Reading the speech")
    spoken = max(1, int(len(speech) / speech_audio.RATE * FPS))
    total = lengths[0] + sum(length - motion for length in lengths[1:])
    speech_job = _Background(lambda: speech_audio.extend(models.speech_features(speech, spoken), total))

    # 2. The prompt, padded to Wan's 512 text tokens.
    context = talk_model.pad_context(_encode_text(models, prompt, device, dtype, stream_text, stats, embeddings,
                                                  report))
    cancel.check()

    # 3. The picture: CLIP image tokens, then the VAE conditioning of the chunk length and its first latent frame.
    still = conditioning.center_scale(_picture(picture), width, height)  # [1, 3, H, W] in 0..1
    report(_PREPARED / 2, "Reading the picture")
    with stage("image encoder", device, stats):
        encoder = models.load_clip(device, torch.float16)
        try:
            with torch.inference_mode():
                image_tokens = encoder(clip_model.prepare(still.to(device) * 2 - 1)).to(dtype)
        finally:
            del encoder
            release(device)
    cancel.check()
    autoencoder = network = remove_stream = None
    pictures = []
    try:
        with stage("vae encode", device, stats):
            autoencoder = models.load_vae(device, dtype)
            with attention_backend.use(backend):
                longest = max(lengths)
                raw = _encode(autoencoder, conditioning.start_sequence(still, longest, width, height), device,
                              computed=min(START_LATENT_FRAMES, (longest - 1) // 4 + 1))
            concat = conditioning.i2v_concat(raw, conditioning.known_latent_frames(1),
                                             autoencoder.config.latents_mean, autoencoder.config.latents_std)
            motion_latent = _normalised(autoencoder, raw[:, :, :1])
            del raw
        cancel.check()

        # 4. The chunks.
        report(_PREPARED, "Loading the talking model")
        with stage("talking model", device, stats):
            network = models.load_transformer(torch.device("cpu") if stream is not None else device, dtype)
            if stream is not None:
                remove_stream = streaming.stream_model(network, device, **stream)
            sigmas = flow.spaced_sigmas(shift, steps).to(device)
            feats = speech_job.result()
            silent = None
            start = 0
            for k, length in enumerate(lengths):
                if k == len(lengths) - 1:
                    read_next_stage_ahead(then)
                latent_frames = (length - 1) // 4 + 1
                shape = (1, LATENT_CHANNELS, latent_frames, height // PIXELS_PER_LATENT, width // PIXELS_PER_LATENT)
                chunk_concat = concat[:, :, :latent_frames]  # the causal encoder: a shorter chunk's own conditioning
                windows = speech_audio.windows(feats, start, length).to(device)
                if audio_scale != 1.0:
                    silent = torch.zeros_like(windows)
                x = flow.initial_latent(sigmas, flow.make_noise(seed + k, shape)).to(device)
                known = motion_latent.shape[2]
                for i in range(steps):
                    report(_PREPARED + (_SAMPLED - _PREPARED) * (k * steps + i) / (count * steps),
                           f"Chunk {k + 1}/{count}, step {i + 1}/{steps}")
                    cancel.check()
                    x[:, :, :known] = motion_latent
                    with torch.inference_mode(), _denoising_attention(device), attention_backend.use(backend):
                        velocity = talk_model.predict_velocity(network, x, sigmas[i], context, chunk_concat,
                                                               image_tokens, windows)
                        if silent is not None:
                            drop = talk_model.predict_velocity(network, x, sigmas[i], context, chunk_concat,
                                                               image_tokens, silent)
                            velocity = drop + audio_scale * (velocity - drop)
                    x = x + (sigmas[i + 1] - sigmas[i]) * velocity
                    x[:, :, :known] = motion_latent
                cancel.check()
                with attention_backend.use(backend):
                    clip = _decode(autoencoder, x, device)  # [length, 3, H, W] on the CPU
                pictures.append(clip if k == 0 else clip[motion:])
                start += length - motion
                if k + 1 < count:
                    with attention_backend.use(backend):
                        motion_latent = _normalised(autoencoder, _encode(autoencoder, clip[-motion:], device))
                del x
    finally:
        if remove_stream is not None:
            remove_stream()
        remove_stream = None
        for model in (network, autoencoder):
            if model is not None and models.recycle is not None:
                models.recycle(model)
        network = autoencoder = None
        release(device)
    if stream is not None:
        streaming.release_host_cache()
    report(1.0, "Done")
    return torch.cat(pictures)[:frames]
