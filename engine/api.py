# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The engine's front door for local generation.

`Engine` hides the plumbing of the pipelines (loaders, device, precision). The files come from the
caller: the product resolves them through its model catalogue, so swapping the model weights means
passing other paths. Every run releases GPU weights when done. An optional bounded host cache reuses
loaded weights between jobs, and Wan's prompt encodings are kept between calls. The lock serializes engine
jobs and explicit releases.
"""
from __future__ import annotations

import importlib.util
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .core.cancel import CancelToken, Cancelled

# Imported by the pipelines (network classes, tokenizers, file readers); when one is missing Engine.available() is
# False and the app reports the engine as unavailable.
REQUIRED_PACKAGES = ("diffusers", "transformers", "safetensors", "gguf", "PIL", "numpy")

ProgressCb = Callable[[float, str], None]
# Host memory for EzAudio's networks while an audio_session is open: the FLAN-T5-XL encoder (2.5 GB), the diffusion
# transformer (1.9 GB) and the waveform decoder (0.1 GB) fit, so a scene's sounds load them from disk once.
SOUND_SESSION_BYTES = 6 << 30


@dataclass(frozen=True)
class ZImageFiles:
    """The four files Z-Image-Turbo needs (the tokenizer is a folder)."""
    transformer: Path
    text_encoder: Path
    vae: Path
    tokenizer: Path

    def missing(self) -> List[str]:
        """What is not on disk, by role."""
        found = {"transformer": self.transformer, "text encoder": self.text_encoder, "vae": self.vae,
                 "tokenizer": self.tokenizer / "vocab.json"}
        return [role for role, path in found.items() if not Path(path).exists()]


@dataclass(frozen=True)
class QwenEditFiles:
    transformer: Path
    text_encoder: Path
    vae: Path
    tokenizer: Path
    lightning: Path

    def missing(self) -> List[str]:
        found = {"transformer": self.transformer, "text encoder": self.text_encoder, "vae": self.vae,
                 "tokenizer vocab": self.tokenizer / "vocab.json", "tokenizer merges": self.tokenizer / "merges.txt",
                 "tokenizer config": self.tokenizer / "tokenizer_config.json", "Lightning LoRA": self.lightning}
        return [role for role, path in found.items() if not path.is_file()]


@dataclass(frozen=True)
class TalkFiles:
    """InfiniteTalk (`pipelines.talk`): the Wan 2.1 I2V files (LightX2V's distilled FP8 blocks), the audio adapter,
    the CLIP image encoder, the Wan VAE and UMT5 encoder it shares with Wan 2.2, and the wav2vec 2.0 folder."""
    text_encoder: Path
    base: Tuple[Path, ...]
    adapter: Path
    clip: Path
    vae: Path
    wav2vec: Path

    def missing(self) -> List[str]:
        found = {"text encoder": self.text_encoder, "adapter": self.adapter, "image encoder": self.clip,
                 "vae": self.vae, "audio encoder": self.wav2vec / "pytorch_model.bin"}
        found.update({f"base {path.name}": path for path in self.base})
        return [role for role, path in found.items() if not Path(path).exists()] + ([] if self.base else ["base"])


@dataclass(frozen=True)
class WanFiles:
    text_encoder: Path
    high: Path
    low: Path
    vae: Path
    kind: str


class Engine:
    """The front door of every local generation: pictures (Z-Image-Turbo), edits (Qwen-Image-Edit-2511), video
    (Wan 2.2 A14B), frame interpolation (RIFE), speech and voice design (VoxCPM2), sound (EzAudio-XL) and lip sync
    (MuseTalk 1.5).

    `cancel` is polled between sampling steps (the product passes the token of `app.cancel`);
    `lora_rounding` is the mode of `engine.formats.lora.attach` for the accelerator LoRAs of the video and edit
    models (LightX2V, Lightning); `attention` the attention of the video experts ("sdpa", or "auto": SageAttention
    where Triton and the GPU allow it, see `engine.core.attention`)."""

    def __init__(self, device: str = "cuda", cancel: Optional[CancelToken] = None, lora_rounding: str = "nearest",
                 warm_cache_bytes: int = 0, vram_limit_gib: int | None = None, attention: str = "sdpa"):
        self.device = device
        self.attention = attention
        self.cancel = cancel or CancelToken()
        self.lora_rounding = lora_rounding
        self.vram_limit_gib = vram_limit_gib
        self._lock = threading.RLock()
        self._cache = None
        self._embeddings = None  # core.memory.EmbeddingCache, made on the first video (it needs torch)
        self._sound_cache = None  # core.memory.ModelCache of EzAudio's networks while an audio_session is open
        if warm_cache_bytes:
            from .core.memory import ModelCache

            self._cache = ModelCache(warm_cache_bytes)

    @staticmethod
    def available() -> bool:
        """True when a CUDA GPU can be used and every library the pipelines import is installed."""
        try:
            import torch
        except ImportError:
            return False
        if not torch.cuda.is_available():
            return False
        return all(importlib.util.find_spec(name) is not None for name in REQUIRED_PACKAGES)

    def _guarded(self, run, clean: Tuple[type, ...] = ()):
        """Run a pipeline; after a failure (an out-of-memory error, a cancel) give the GPU memory back.

        The traceback of an error holds the frames of the pipeline and with them the networks, so the cache
        cannot be emptied while it is attached; it is dropped first, and its text is kept as a note. Errors of a
        `clean` type pass straight through and keep the warm models: the pipeline raises them after releasing
        everything it loaded (lip sync's verdict that too few frames had a usable face, which the app answers by
        trying again on a crop of the face)."""
        try:
            if self.vram_limit_gib is not None and self.device.startswith("cuda"):
                from .core.memory import limit_cuda

                limit_cuda(self.device, self.vram_limit_gib)
            return run()
        except clean:
            raise
        except BaseException as error:  # noqa: BLE001 - Cancelled is a BaseException and passes through
            failure, where = error, "".join(traceback.format_tb(error.__traceback__))
        failure.__traceback__ = None
        if not isinstance(failure, Cancelled):
            failure.add_note("engine traceback:" + chr(10) + where.rstrip())
        self.free()
        raise failure

    def _models(self, files: ZImageFiles, keep: bool = True):
        from .pipelines.image import ZImageModels

        absent = files.missing()
        if absent:
            raise FileNotFoundError("Z-Image files are missing: " + ", ".join(absent))
        cache = self._cache
        if cache is not None:
            cache.prepare("zimage")
        return ZImageModels.from_files(files.transformer, files.text_encoder, files.vae, files.tokenizer,
                                       cache=cache if keep else None)

    def _stream_images(self) -> bool:
        if not self.device.startswith("cuda"):
            return False
        import torch

        index = torch.device(self.device).index
        free, total = torch.cuda.mem_get_info(index)
        cap = total * torch.cuda.get_per_process_memory_fraction(index)
        return min(free, cap) < (14 << 30)

    def txt2img(self, files: ZImageFiles, prompt: str, width: int, height: int, seed: int, steps: int = 8,
                progress_cb: Optional[ProgressCb] = None, loras=(), keep: bool = True, then=()):
        """A new picture of exactly width x height pixels (multiples of 16) as a PIL image. `loras`: (file, strength)
        pairs for this picture only (a style LoRA). `keep` False: the networks are not kept warm afterwards (a
        Timeline shot's drawn starting frame, after which the video stage would drop them unused: keeping them meant
        copying about 20 GB back into host memory first)."""
        import torch

        from .pipelines import image

        with self._lock:
            return self._guarded(lambda: image.txt2img(
                self._models(files, keep), prompt, width, height, seed, steps=steps, device=self.device,
                dtype=torch.bfloat16, vae_dtype=torch.bfloat16, progress_cb=progress_cb, cancel=self.cancel,
                stream=self._stream_images(), loras=tuple(loras or ()), then=then))

    def edit(self, files: QwenEditFiles, pictures, prompt: str, width: int, height: int, seed: int,
             progress_cb: Optional[ProgressCb] = None, loras=(), then=()):
        """Compose a picture with Qwen-Image-Edit and its four-step Lightning adapter; `loras` adds (file, strength)
        pairs after it (a style LoRA of an animated film). `then`: files the next stage reads first, read ahead while
        the steps run."""
        from .pipelines import edit

        absent = files.missing()
        if absent:
            raise FileNotFoundError("Qwen-Image-Edit files are missing: " + ", ".join(absent))
        models = edit.EditModels.from_files(files.transformer, files.text_encoder, files.vae, files.tokenizer)
        cuda = self.device.split(":", 1)[0] == "cuda"
        with self._lock:
            if self._cache is not None:
                self._cache.prepare("edit")
            # The transformer (21 GB) streams its blocks from the file's pages without pinning them first: pinning
            # took about 11 s per picture on a 16 GB card with 64 GB of RAM, longer than the slower copies cost
            # (2026-11-15, the same picture to the bit).
            return self._guarded(lambda: edit.edit(
                models, pictures, prompt, width, height, seed, loras=[(files.lightning, 1.0), *(loras or ())],
                device=self.device, progress_cb=progress_cb, cancel=self.cancel, lora_rounding=self.lora_rounding,
                stream={"pin": False} if cuda else None, stream_text=cuda, then=then))

    def video(self, files: WanFiles, prompt: str, width: int, height: int, length: int, seed: int,
              steps: int, shift: float, split: int, *, start_image=None, loras=None, progress_cb=None, then=()):
        """Generate native-rate frames. The caller retains the VAE frame needed to continue the shot.

        A prompt encoded by an earlier call (the previous part of the clip, or with `expect_video_prompts` an
        earlier scene of the job) is not encoded again. With the warm cache the experts stream from its pageable
        copies; without it they are pinned for each call. `then`: files the next stage reads first, read ahead
        while the low-noise expert computes."""
        from .core.memory import EmbeddingCache
        from .pipelines import video

        cuda = self.device.split(":", 1)[0] == "cuda"
        with self._lock:
            if self._cache is not None:
                self._cache.prepare("wan")
            if self._embeddings is None:
                self._embeddings = EmbeddingCache()
            models = video.WanModels.from_files(files.text_encoder, files.high, files.low, files.vae, files.kind,
                                               cache=self._cache)
            return self._guarded(lambda: video.generate(
                models, prompt, width, height, length, seed, steps, shift, start_image=start_image,
                split=split, loras=loras, device=self.device, progress_cb=progress_cb, cancel=self.cancel,
                lora_rounding=self.lora_rounding, stream={"pin": self._cache is None} if cuda else None,
                stream_text=cuda, embeddings=self._embeddings, attention=self.attention, then=then))

    def talk(self, files: TalkFiles, prompt: str, picture, speech, frames: int, width: int, height: int, seed: int, *,
             steps: int = 4, shift: float = 5.0, audio_scale: float = 1.0, progress_cb=None, then=()):
        """A talking clip of the character in `picture` saying `speech` (16 kHz mono samples): `frames` pictures
        [N, 3, H, W] in 0..1 at 25 FPS (InfiniteTalk, `pipelines.talk`). `then`: files the next stage reads first,
        read ahead while the last chunk computes."""
        from .core.memory import EmbeddingCache
        from .pipelines import talk

        absent = files.missing()
        if absent:
            raise FileNotFoundError("InfiniteTalk files are missing: " + ", ".join(absent))
        cuda = self.device.split(":", 1)[0] == "cuda"
        with self._lock:
            if self._cache is not None:
                self._cache.prepare("talk")
            if self._embeddings is None:
                self._embeddings = EmbeddingCache()
            models = talk.TalkModels.from_files(files.text_encoder, files.base, files.adapter, files.clip, files.vae,
                                                files.wav2vec, cache=self._cache)
            return self._guarded(lambda: talk.generate(
                models, prompt, picture, speech, frames, width, height, seed, steps=steps, shift=shift,
                audio_scale=audio_scale, device=self.device, progress_cb=progress_cb, cancel=self.cancel,
                stream={"pin": self._cache is None} if cuda else None, stream_text=cuda,
                embeddings=self._embeddings, attention=self.attention, then=then))

    def expect_video_prompts(self, prompts) -> None:
        """The prompts the next videos will ask for (the scenes of a Timeline job), in order; an empty sequence
        when the job ends. The first video that loads the text encoder encodes them all in that one load, and
        they are kept until used (`core.memory.EmbeddingCache.expect`)."""
        from .core.memory import EmbeddingCache

        with self._lock:
            if self._embeddings is None:
                if not prompts:
                    return
                self._embeddings = EmbeddingCache()
            self._embeddings.expect(prompts)

    def interpolate(self, weights: Path, frames, source_fps: int, target_fps: int, progress_cb=None):
        """Use the source and destination frame clocks to interpolate with RIFE."""
        from .models import rife

        def run():
            self.cancel.check()
            model = rife.load_rife(weights, self.device)
            return rife.interpolate(model, frames, source_fps, target_fps, cancel=self.cancel, progress_cb=progress_cb)

        with self._lock:
            try:
                return self._guarded(run)
            finally:
                rife.release_grids()
                self._empty_allocator_caches()

    # Speech (VoxCPM2, 4.6 GB), sound (EzAudio-XL, 4.4 GB) and lip sync (MuseTalk 1.5, 2 GB) load outside the
    # warm cache and fit in host memory beside it, so they leave its models in place: a Timeline job keeps Wan's
    # experts across its scenes' sound and lip sync. The cache is on only on machines with room for it
    # (app/engine_runtime.warm_cache_gib).

    def audio_bg(self, bundle: Path, prompt: str, negative: str, seconds: float, seed: int, progress_cb=None):
        """A mono 48 kHz EzAudio-XL sound (a background or an effect) with the timeline's handles, and its rate.
        Inside audio_session the networks wait in host memory between calls."""
        from .pipelines import ezaudio

        with self._lock:
            return self._guarded(lambda: ezaudio.generate(bundle, prompt, negative, seconds, seed, device=self.device,
                                                          progress_cb=progress_cb, cancel=self.cancel,
                                                          cache=self._sound_cache))

    @contextmanager
    def audio_session(self):
        """The sound calls inside (a scene's bed, its retries and its effects) read EzAudio's networks from disk
        once: between calls they wait in host memory, in a cache of the session's own (the warm cache keeps Wan's
        experts), and they are released when the block ends."""
        from .core.memory import ModelCache

        with self._lock:
            outer = self._sound_cache is not None
            if not outer:
                self._sound_cache = ModelCache(SOUND_SESSION_BYTES)
        try:
            yield
        finally:
            if not outer:
                with self._lock:
                    self._sound_cache.clear()
                    self._sound_cache = None
                    self._empty_allocator_caches()

    def tts(self, model_dir: Path, lines, *, isolated: bool, progress_cb=None):
        from .pipelines import speech

        with self._lock:
            return self._guarded(lambda: speech.synthesize_many(model_dir, lines, device=self.device,
                                                               isolated=isolated, progress_cb=progress_cb,
                                                               cancel=self.cancel, vram_limit_gib=self.vram_limit_gib))

    def design_voice(self, model_dir: Path, description: str, output: Path, *, seed: int, isolated: bool,
                     text: Optional[str] = None, progress_cb=None) -> Path:
        """A new VoxCPM2 voice sample for a text description, written to `output` (speech.design_voice)."""
        from .pipelines import speech

        design = speech.VoiceDesign(description, output, text or speech.DESIGN_TEXT, seed)
        with self._lock:
            return self._guarded(lambda: speech.design_voice(model_dir, design, device=self.device, isolated=isolated,
                                                             progress_cb=progress_cb, cancel=self.cancel,
                                                             vram_limit_gib=self.vram_limit_gib))

    def lipsync(self, unet: Path, whisper: Path, vae_dir: Path, frames, audio, detector, *, low_vram: bool,
                progress_cb=None):
        """A lip-synced copy of 25 fps RGB `frames` for one speaker (16 kHz mono `audio`): MuseTalk 1.5 on the
        face `detector` finds in every frame. Raises musetalk.CoverageError when too few of the speech frames had
        a usable face."""
        from .pipelines import musetalk

        with self._lock:
            models = musetalk.MuseTalkModels.from_files(unet, whisper, vae_dir)
            return self._guarded(lambda: musetalk.lipsync(models, frames, audio, detector, low_vram=low_vram,
                                                          progress_cb=progress_cb, cancel=self.cancel,
                                                          device=self.device), clean=(musetalk.CoverageError,))

    def lipsync_tracks(self, unet: Path, whisper: Path, vae_dir: Path, frames, tracks, *, low_vram: bool,
                       progress_cb=None, report_cb=None):
        """Every speaker of a shot in one pass: `tracks` are musetalk.FaceTrack (a person's audio and face in every
        frame). Frames a track cannot use keep their pixels; `report_cb` receives what each track covered."""
        from .pipelines import musetalk

        with self._lock:
            models = musetalk.MuseTalkModels.from_files(unet, whisper, vae_dir)
            return self._guarded(lambda: musetalk.musetalk(models, frames, tracks, low_vram=low_vram,
                                                           progress_cb=progress_cb, cancel=self.cancel,
                                                           device=self.device, report_cb=report_cb))

    def free(self) -> None:
        """Release cached host models and unused GPU and pinned host memory. The kept prompt encodings stay (a few
        MB each, at most `EmbeddingCache.entries` once no job announces prompts): the next job of an idle studio,
        such as Generate on the next scene, then skips the text encoder for a prompt encoded before (13 s with
        UMT5); `forget_prompts` drops them."""
        with self._lock:
            if self._cache is not None:
                self._cache.clear()
            if self._sound_cache is not None:
                self._sound_cache.clear()
            self._empty_allocator_caches()

    def forget_prompts(self) -> None:
        """Drop the kept prompt encodings."""
        with self._lock:
            if self._embeddings is not None:
                self._embeddings.clear()

    def trim(self) -> None:
        """Give back unused GPU and pinned host memory before another model takes the card, keeping the
        warm models (they wait in host memory) and the prompt encodings."""
        with self._lock:
            self._empty_allocator_caches()

    @staticmethod
    def _empty_allocator_caches() -> None:
        import gc

        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                from .core.streaming import release_host_cache
            except ImportError:  # the streaming code needs the gguf package, which this install may lack
                return
            release_host_cache()
