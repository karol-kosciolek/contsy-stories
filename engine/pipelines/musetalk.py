# SPDX-FileCopyrightText: 2024 Tencent Music Entertainment Group
# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only AND MIT
"""MuseTalk 1.5 lip sync on decoded RGB frames: one UNet step per face and frame, no sampling loop.

Own implementation of the inference of TMElyralab/MuseTalk (MIT, `scripts/inference.py` at 0a89dec): for
every frame with a usable face, the face box is cropped to 256 px; its copy with the lower half blacked out
and the unmasked copy are encoded by the SD VAE (x 0.18215) and concatenated; the UNet predicts the face's
latent at timestep 0 from them and the frame's Whisper window; the VAE decodes it and the lower face is
blended back into the frame.

Deliberate differences from the reference script (all in `Settings`, `REFERENCE` turns them off):

* Frames are never dropped or reordered. A frame without a usable face keeps its pixels (the reference
  removes it from the latent list, which shifts every later frame against the audio); frames after the
  end of the audio keep theirs too (the reference ping-pongs the video to the length of the audio).
* Latents use the mean of the VAE posterior (`mode`, as in MuseTalk's training code) instead of a fresh
  random sample per frame, which flickers.
* Our face boxes (MediaPipe / YuNet), smoothed over a few frames, and our geometric mask, see `faces`.
  The chin margin is sized for that mask's feather (about 10 % of the box height instead of 10 px).
* Several faces of one shot are separate tracks with their own audio, batched together.
* Every call reports how many speech frames each track covered (`Report`), and can fail below a minimum
  share (`CoverageError`, on by default in `lipsync`); a face the network returns as NaN or inf is never
  pasted (it would be a black mouth).

Memory on a 16 GB card: Whisper tiny runs first (float32, under 0.1 GB) and is released; then the UNet
(about 1.7 GB in float16) and the VAE (0.17 GB) stay on the card while batches of crops stream through.
Nothing is offloaded: the models are small; the batch size bounds the activations (the VAE decoder's at
256 px dominate). Frames, crops and blending stay on the CPU.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..core.cancel import CancelToken
from ..core.stages import release
from ..models.musetalk import audio as speech
from ..models.musetalk import faces as geometry
from ..models.musetalk import unet as network

FPS = 25
RESOLUTION = 256
SCALE = 0.18215  # sd-vae-ft-mse's scaling factor; the VAE config does not override it


@dataclass(frozen=True)
class Settings:
    """Knobs of the face handling. The defaults are ours; `REFERENCE` reproduces the reference script, except that
    the VAE latent is always the posterior mean (the script samples it), which keeps every run deterministic."""
    resolution: int = RESOLUTION   # side of the face crop the network sees
    smooth: int = 5                # frames averaged for box and mask (1: off)
    fill_gap: int = 3              # missing detections bridged by interpolation (0: off)
    fade: int = 3                  # frames over which the change fades in and out around unused frames
    min_face: float = 32.0         # narrower face boxes (pixels) keep their pixels
    max_turn: float = 0.8          # more turned faces (0 frontal, 1 profile) keep their pixels
    margin_px: int = 0             # chin margin in pixels (the reference: 10 px, trained 0-20)
    feather: float = 1.5           # blur width relative to the reference's kernel
    inset: bool = True             # weights reach zero before the box edge (and the margin grows to suit)
    pad_outside: bool = True       # boxes may leave the frame (edge pixels repeated) instead of being clamped
    shrink_area: bool = True       # area resampling when the face is pasted into a smaller box

    def margin(self, box: geometry.Box) -> int:
        """Chin margin of a face box. With `inset` at least `faces.blend_margin` (about 10 % of the box
        height): a smaller one lets the cleared border reach the chin, so the jaw could not change."""
        return max(self.margin_px, geometry.blend_margin(box, self.feather)) if self.inset else self.margin_px


DEFAULT = Settings()
REFERENCE = Settings(smooth=1, fill_gap=0, fade=0, min_face=1.0, max_turn=math.inf, margin_px=10,
                     feather=1.0, inset=False, pad_outside=False, shrink_area=False)
#: `lipsync` fails when less than this share of the speech frames got a generated mouth, so a caller that
#: retries (app/lipsync.py crops the face and tries again) or reports the failure still can.
MIN_SPEECH_COVERAGE = 0.5


@dataclass(frozen=True)
class TrackReport:
    """What one track's lip sync covered. Speech frames are the frames with speech in the track's own
    audio (`audio.speech_frames`); frames past the end of the audio have none."""
    frames: int = 0      # frames of the clip
    speech: int = 0      # speech frames
    generated: int = 0   # frames given a generated mouth (any blend strength, fades included)
    synced: int = 0      # speech frames among them
    no_face: int = 0     # speech frames left out because no face was found (and none could be bridged)
    small: int = 0       # ... because the face was narrower than Settings.min_face
    turned: int = 0      # ... because the face was turned further than Settings.max_turn
    invalid: int = 0     # generated frames left out because the network returned NaN or inf

    @property
    def coverage(self) -> float:
        """The share of the speech frames that were lip-synced (1.0 without speech: nothing to do)."""
        return self.synced / self.speech if self.speech else 1.0


@dataclass(frozen=True)
class Report:
    """The outcome of one `musetalk` call, passed to `report_cb` (also when it raises CoverageError)."""
    tracks: tuple = ()           # one TrackReport per FaceTrack, in order
    nonfinite_latents: int = 0   # UNet outputs with NaN or inf (fp16 overflow)
    nonfinite_faces: int = 0     # decoded faces with NaN or inf: left out, their frames keep their pixels


class CoverageError(RuntimeError):
    """Too few speech frames could be lip-synced (no face, a small or turned one, or invalid output).
    The message says why and is meant for the user; `report` has the counts."""

    def __init__(self, message: str, report: Report):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class MuseTalkModels:
    load_unet: Callable[[torch.device, torch.dtype], torch.nn.Module]
    load_audio: Callable[[torch.device], torch.nn.Module]
    load_vae: Callable[[torch.device, torch.dtype], torch.nn.Module]

    @classmethod
    def from_files(cls, unet_file: str | Path, whisper_file: str | Path, vae_dir: str | Path) -> "MuseTalkModels":
        def load_vae(device, dtype):
            from diffusers import AutoencoderKL

            vae, loading = AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=dtype, local_files_only=True,
                                                         use_safetensors=True, output_loading_info=True)
            if any(loading.values()):
                raise ValueError(f"The MuseTalk VAE checkpoint does not match its configuration: {loading}")
            return vae.eval().to(device)

        return cls(lambda device, dtype: network.load_unet(unet_file, device, dtype),
                   lambda device: speech.load_audio_encoder(whisper_file, device), load_vae)


@dataclass(frozen=True)
class FaceTrack:
    """One speaker: 16 kHz mono audio (silence where this person does not speak) and the person's face in
    every frame (None where it is not visible), as many entries as there are frames."""
    audio: np.ndarray | Tensor
    faces: Sequence[Optional[geometry.Face]]


@dataclass(frozen=True)
class _Job:
    frame: int
    track: int
    box: geometry.IntBox
    face: geometry.Face
    strength: float


class _Clock:
    """Seconds and peak allocated memory per stage. Interleaved stages (the batch loop) accumulate their
    time and keep their highest peak; the CUDA peak counter is reset at every stage start, so each stage
    reports its own peak (the overall peak is the largest of them). The GPU is synchronised at stage
    edges only when someone listens (`stats`)."""

    def __init__(self, device: torch.device, stats):
        self.device, self.stats = device, stats
        self.measure = stats is not None and device.type == "cuda"
        self.seconds: Dict[str, float] = {}
        self.peaks: Dict[str, int] = {}

    @contextmanager
    def __call__(self, name: str):
        if self.measure:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.measure:
                torch.cuda.synchronize(self.device)
                self.peaks[name] = max(self.peaks.get(name, 0), torch.cuda.max_memory_allocated(self.device))
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start

    def report(self, *names: str):
        for name in names:
            if self.stats is not None and name in self.seconds:
                self.stats(name, self.seconds[name], self.peaks.get(name, 0))


def _check(frames: np.ndarray, tracks: Sequence[FaceTrack], fps: int, settings: Settings):
    if not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 \
            or min(frames.shape) < 1:
        raise ValueError("MuseTalk needs nonempty uint8 RGB frames [N, H, W, 3].")
    if int(fps) != fps or fps <= 0:
        raise ValueError("MuseTalk needs a whole positive frame rate.")
    if settings.resolution <= 0 or settings.resolution % 8:
        raise ValueError("Invalid MuseTalk settings.")
    for track in tracks:
        if len(track.faces) != len(frames):
            raise ValueError("Every face track needs one entry per frame.")


def plan_track(faces: Sequence[Optional[geometry.Face]], frame_shape: Sequence[int], count: int,
               settings: Settings) -> List[Optional[tuple]]:
    """Per frame (box, face, strength) or None: which frames of one track are generated, and how strongly
    they are blended. Frames from `count` on (past the audio) are never generated."""
    faces = geometry.usable(list(faces[:count]) + [None] * max(0, len(faces) - count),
                            settings.min_face, settings.max_turn)
    faces = geometry.smooth(geometry.fill_gaps(faces, settings.fill_gap), settings.smooth)
    boxes = [None if face is None else
             geometry.crop_box(face, frame_shape, settings.margin(face.box), settings.pad_outside) for face in faces]
    strength = geometry.strengths([box is not None for box in boxes], settings.fade)
    return [None if box is None or strength[i] <= 0 else (box, faces[i], float(strength[i]))
            for i, box in enumerate(boxes)]


def track_report(faces: Sequence[Optional[geometry.Face]], plan: Sequence[Optional[tuple]], speech_mask: np.ndarray,
                 settings: Settings) -> TrackReport:
    """The planned coverage of one track (`plan` from `plan_track`; `speech_mask` for its first frames),
    with the reason each left-out speech frame was left out."""
    reasons = {"no_face": 0, "small": 0, "turned": 0}
    synced = 0
    for index in np.flatnonzero(speech_mask):
        if plan[index] is not None:
            synced += 1
            continue
        face = faces[index]
        if face is None:
            reasons["no_face"] += 1
        elif face.box[2] - face.box[0] < settings.min_face:
            reasons["small"] += 1
        elif face.turn > settings.max_turn:
            reasons["turned"] += 1
    return TrackReport(frames=len(plan), speech=int(np.count_nonzero(speech_mask)),
                       generated=sum(step is not None for step in plan), synced=synced, **reasons)


def coverage_problem(report: Report, min_coverage: float) -> Optional[str]:
    """The user-facing reason when a track's coverage is below `min_coverage`, else None."""
    for number, track in enumerate(report.tracks, start=1):
        if not track.speech or track.coverage >= min_coverage:
            continue
        who = "the speaker" if len(report.tracks) == 1 else f"speaker {number}"
        reasons = ", ".join(f"{label} in {count}" for label, count in (
            ("no face found", track.no_face), ("face too small", track.small), ("face turned away", track.turned),
            ("invalid network output", track.invalid)) if count)
        return (f"Lip sync covered {track.synced} of {track.speech} speech frames of {who}"
                + (f" ({reasons})" if reasons else "") + f"; at least {min_coverage:.0%} are needed.")
    return None


def _encode(vae, pixels: Tensor, dtype: torch.dtype) -> Tensor:
    """Masked and reference latents [N, 8, h, w] for float32 crops [N, 3, s, s] in [0, 1] (the reference's
    order: masked first). The reference normalises in float32 and casts for the VAE."""
    normalized = pixels * 2.0 - 1.0
    masked = normalized.clone()
    masked[:, :, normalized.shape[2] // 2:] = -1.0  # x * (top-half mask) before normalising -> -1 below
    latents = []
    for image in (masked, normalized):
        latent = vae.encode(image.to(dtype)).latent_dist.mode()
        latents.append(latent * getattr(vae.config, "scaling_factor", SCALE))
    return torch.cat(latents, dim=1)


def _finite(batch: Tensor) -> Tensor:
    """bool [N]: which items of the batch hold only finite values."""
    return torch.isfinite(batch).flatten(1).all(dim=1)


def _decode(vae, latents: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """uint8 faces [N, s, s, 3] and which of them are finite. A NaN would otherwise become 0 in the cast
    and paste a black mouth; the caller leaves such faces out."""
    images = vae.decode(latents / getattr(vae.config, "scaling_factor", SCALE)).sample.float()
    finite = _finite(images)
    images = ((images / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
    return images.permute(0, 2, 3, 1).cpu().numpy(), finite.cpu().numpy()


@torch.inference_mode()
def musetalk(models: MuseTalkModels, frames: np.ndarray, tracks: Sequence[FaceTrack], *, fps: int = FPS,
             settings: Settings = DEFAULT, batch_size: Optional[int] = None,
             mask_fn: Optional[geometry.MaskFn] = None, progress_cb=None, cancel: CancelToken | None = None,
             stats=None, low_vram: bool = False, device: str | torch.device = "cuda",
             dtype: torch.dtype | None = None, min_coverage: float = 0.0,
             report_cb: Optional[Callable[[Report], None]] = None) -> np.ndarray:
    """Return a lip-synced copy of `frames` (same count, size and order).

    Every track's face is generated from its own audio; tracks are processed in the given order, so a later
    track blends over an earlier one where the two overlap. `mask_fn(face, box)` replaces the geometric
    lower-face mask (a 0/255 uint8 array over the box). `stats(stage, seconds, peak_bytes)` receives
    "audio", "load", "crop", "vae_encode", "unet", "vae_decode" and "blend" (each with its own CUDA peak).

    `report_cb(Report)` receives what each track covered, once, also before a CoverageError. With
    `min_coverage` > 0, a track with speech of which a smaller share could be lip-synced raises
    CoverageError: before any model is loaded when the faces already decide it, or at the end when the
    network returned NaN or inf for too many faces (those faces are never pasted)."""
    _check(frames, tracks, fps, settings)
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage is a share between 0 and 1.")
    device = torch.device(device)
    dtype = dtype or (torch.float16 if device.type == "cuda" else torch.float32)
    cancel = cancel or CancelToken()
    batch_size = batch_size or ((4 if low_vram else 16) if device.type == "cuda" else 4)
    clock = _Clock(device, stats)

    def progress(fraction: float, message: str):
        cancel.check()
        if progress_cb is not None:
            progress_cb(fraction, message)

    def finish(report: Report) -> None:
        if report_cb is not None:
            report_cb(report)
        problem = coverage_problem(report, min_coverage) if min_coverage > 0 else None
        if problem is not None:
            raise CoverageError(problem, report)

    output = frames.copy()
    # Plan from the faces and the audio lengths first: when nothing can be lip-synced, no model is loaded.
    waves = [speech.waveform(track.audio) for track in tracks]
    counts = [min(len(frames), speech.frame_count(wave.numel(), fps)) for wave in waves]
    plans = [plan_track(track.faces, frames.shape[1:3], count, settings)
             for track, count in zip(tracks, counts, strict=True)]
    voiced = [speech.speech_frames(wave, fps, count) for wave, count in zip(waves, counts, strict=True)]
    planned = Report(tuple(track_report(track.faces, plan, mask, settings)
                           for track, plan, mask in zip(tracks, plans, voiced, strict=True)))
    jobs = [_Job(i, t, *plan[i]) for i in range(len(frames)) for t, plan in enumerate(plans) if plan[i] is not None]
    short = min_coverage > 0 and coverage_problem(planned, min_coverage) is not None
    if not jobs or short:
        if not jobs and tracks:
            progress(1.0, "Lip sync: no usable face")
        finish(planned)  # raises CoverageError when short
        return output

    progress(0.02, "Lip sync: encoding audio")
    with clock("audio"):
        encoder = models.load_audio(device)
        windows = [speech.windows(encoder, wave, fps, count, cancel) for wave, count in zip(waves, counts, strict=True)]
        del encoder
        release(device)
    clock.report("audio")

    progress(0.08, "Lip sync: loading MuseTalk")
    with clock("load"):
        unet = models.load_unet(device, dtype)
        vae = models.load_vae(device, dtype)
        if low_vram and hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
    clock.report("load")

    left_out: List[set] = [set() for _ in tracks]
    nonfinite_latents = 0
    for first in range(0, len(jobs), batch_size):
        batch = jobs[first:first + batch_size]
        progress(0.1 + 0.88 * first / len(jobs), f"Lip sync: frame {first + 1}/{len(jobs)}")
        with clock("crop"):
            crops = np.stack([geometry.crop(frames[job.frame], job.box, settings.resolution) for job in batch])
            pixels = torch.from_numpy(crops).to(device).permute(0, 3, 1, 2).float() / 255.0
            audio = speech.conditioning(torch.stack([windows[job.track][job.frame] for job in batch]))
        with clock("vae_encode"):
            latents = _encode(vae, pixels, dtype)
        cancel.check()
        with clock("unet"):
            predicted = network.predict(unet, latents.to(dtype), audio.to(device=device, dtype=dtype))
            nonfinite_latents += int((~_finite(predicted)).sum())
        cancel.check()
        with clock("vae_decode"):
            generated, finite = _decode(vae, predicted)
        with clock("blend"):
            for job, face, ok in zip(batch, generated, finite, strict=True):
                if not ok:  # a NaN face would paste a black mouth: keep the frame's own pixels
                    left_out[job.track].add(job.frame)
                    continue
                raw = mask_fn(job.face, job.box) if mask_fn is not None else geometry.lower_face_mask(job.face, job.box)
                weights = geometry.feather(raw, job.box, settings.feather, settings.inset) * job.strength
                geometry.paste(output[job.frame], face, job.box, weights, settings.shrink_area)
    clock.report("crop", "vae_encode", "unet", "vae_decode", "blend")
    del unet, vae, pixels, audio, latents, predicted  # the card is free when the call returns
    release(device)
    progress(1.0, "Lip sync: done")
    finish(_final_report(planned, voiced, left_out, nonfinite_latents))
    return output


def _final_report(planned: Report, voiced: Sequence[np.ndarray], left_out: Sequence[set],
                  nonfinite_latents: int) -> Report:
    """The planned report minus the faces left out for NaN or inf (`voiced`: each track's speech frames)."""
    tracks = []
    for track, mask, frames in zip(planned.tracks, voiced, left_out, strict=True):
        if frames:
            lost_speech = sum(1 for index in frames if index < len(mask) and mask[index])
            track = replace(track, generated=track.generated - len(frames), synced=track.synced - lost_speech,
                            invalid=len(frames))
        tracks.append(track)
    return Report(tuple(tracks), nonfinite_latents, sum(len(frames) for frames in left_out))


def detect(frames: np.ndarray, detector, cancel: CancelToken | None = None) -> List[Optional[geometry.Face]]:
    """The detector's face for every frame (a callable RGB frame -> Face or None)."""
    cancel = cancel or CancelToken()
    found = []
    for frame in frames:
        cancel.check()
        found.append(detector(frame))
    return found


def lipsync(models: MuseTalkModels, frames: np.ndarray, audio: np.ndarray | Tensor, detector, *,
            low_vram: bool = False, progress_cb=None, cancel: CancelToken | None = None, stats=None,
            device: str | torch.device = "cuda", dtype: torch.dtype | None = None, fps: int = FPS,
            settings: Settings = DEFAULT, batch_size: Optional[int] = None,
            min_coverage: float = MIN_SPEECH_COVERAGE,
            report_cb: Optional[Callable[[Report], None]] = None) -> np.ndarray:
    """One speaker (Engine.lipsync): `detector` maps an RGB frame to a Face or None
    (`faces.MuseTalkFaceDetector`).

    It raises CoverageError when the face could not be used for enough of the speech (`min_coverage`), so a
    caller can try again on a crop of the face; frames without a usable face otherwise keep their pixels."""
    cancel = cancel or CancelToken()
    if progress_cb is not None:
        progress_cb(0.0, "Lip sync: finding the face")
    start = time.perf_counter()
    found = detect(frames, detector, cancel)
    if stats is not None:
        stats("detect", time.perf_counter() - start, 0)
    return musetalk(models, frames, [FaceTrack(audio, found)], fps=fps, settings=settings, batch_size=batch_size,
                    progress_cb=progress_cb, cancel=cancel, stats=stats, low_vram=low_vram, device=device,
                    dtype=dtype, min_coverage=min_coverage, report_cb=report_cb)


__all__ = ["DEFAULT", "MIN_SPEECH_COVERAGE", "REFERENCE", "CoverageError", "FaceTrack", "MuseTalkModels", "Report",
           "Settings", "TrackReport", "coverage_problem", "detect", "lipsync", "musetalk", "plan_track",
           "track_report"]
