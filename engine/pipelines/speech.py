# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""VoxCPM2 speech: clone local reference voices for a batch of WAV lines, and design a new voice from a text
description (a reference sample made without any reference).

Voice design is VoxCPM2's own instruction syntax: a sentence that starts with "(<description>) " is spoken in
the voice the parentheses describe. The sample reads DESIGN_TEXT, a neutral sentence of about five seconds,
and is accepted only when it is 3 to 12 seconds of finite, audible sound; otherwise the next seed is tried.
"""
from __future__ import annotations

import gc
import multiprocessing
import os
import re
import tempfile
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..core.cancel import CancelToken

SEED = 42
GUIDANCE = 2.0
STEPS = 20
# Voice design (the recipe of designed character voices): fewer steps are enough for a sample
# that only has to carry a timbre, and three seeds cover the rare take that is too short or silent.
DESIGN_TEXT = "Take a second and look over here. We can leave this light on until you are ready to go."
DESIGN_STEPS = 12
DESIGN_TRIES = 3
DESIGN_SECONDS = (3.0, 12.0)
DESIGN_PEAK = 0.01  # a take whose loudest sample stays below this is silence
MAX_DESCRIPTION = 300
_RNG_LOCK = threading.Lock()
ProgressCb = Callable[[float, str], None]


@dataclass(frozen=True)
class SpeechLine:
    text: str
    reference: str | Path
    output: str | Path


@dataclass(frozen=True)
class VoiceDesign:
    """A voice to design: what it sounds like, where the sample goes, the sentence it reads and the first seed."""
    description: str
    output: str | Path
    text: str = DESIGN_TEXT
    seed: int = SEED


def clean_description(description: str) -> str:
    """The description as one line without parentheses: VoxCPM2 ends the instruction at the first ")"."""
    text = " ".join(re.sub(r"[()（）]", " ", description or "").split())
    return text[:MAX_DESCRIPTION].rstrip(" ,.")


def design_prompt(description: str, text: str = DESIGN_TEXT) -> str:
    """The text VoxCPM2 reads for a designed voice: the instruction in parentheses, then the sentence."""
    return f"({clean_description(description)}, natural conversational English) {text.strip()}"


def acceptable(wav, rate: int) -> bool:
    """Whether a designed take can be a voice sample: one channel of 3 to 12 seconds, finite and audible."""
    import numpy as np

    wav = np.asarray(wav)
    low, high = DESIGN_SECONDS
    return (wav.ndim == 1 and rate * low <= wav.size <= rate * high and bool(np.isfinite(wav).all())
            and float(np.max(np.abs(wav))) > DESIGN_PEAK)


def _load(model_dir: Path, device: str):
    from voxcpm import VoxCPM

    from ..core.skip_init import no_weight_init

    # Paths are resolved by the caller. With local_files_only the loader
    # cannot fetch a snapshot even if the directory disappears in the meantime.
    # VoxCPM builds the network, fills it with random weights and then loads the checkpoint over all of them: the
    # checkpoint covers every parameter (checked against the pinned files, 2026-11-15: no missing key, the same 898
    # tensors to the bit), so the random fill (5.7 s of CPU per load) is skipped.
    with no_weight_init():
        return VoxCPM.from_pretrained(str(model_dir), device=device, load_denoiser=False,
                                      local_files_only=True, optimize=False)


def _with_model(model_dir: Path, device: str, cancel: CancelToken, report: ProgressCb, work) -> None:
    """Load VoxCPM2 under the RNG lock and run `work(model, seed_all)`; the caller's RNG state is kept."""
    import torch

    target = torch.device(device)
    cuda = target.type == "cuda"
    devices = [target.index if target.index is not None else torch.cuda.current_device()] if cuda else []

    def seed_all(seed: int) -> None:
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(seed)

    # The pinned VoxCPM inference code draws randomness from torch only. Save
    # both states so loading and seeding do not affect the next pipeline.
    while not _RNG_LOCK.acquire(timeout=0.1):
        cancel.check()
    model = None
    try:
        with torch.random.fork_rng(devices=devices):
            report(0.10, "Voice: loading the model")
            model = _load(model_dir, device)
            work(model, seed_all)
    finally:
        del model
        _RNG_LOCK.release()
        if cuda:
            gc.collect()
            torch.cuda.empty_cache()


def _generate(model_dir: Path, lines: list[SpeechLine], device: str,
              report: ProgressCb, cancel: CancelToken) -> None:
    import numpy as np
    import soundfile as sf

    def work(model, seed_all) -> None:
        for index, line in enumerate(lines):
            report(0.30 + 0.70 * index / len(lines), f"Voice: line {index + 1}/{len(lines)}")
            seed_all(SEED)
            wav = np.asarray(model.generate(text=line.text, reference_wav_path=str(line.reference),
                                            cfg_value=GUIDANCE, inference_timesteps=STEPS))
            cancel.check()
            if wav.ndim != 1 or wav.size == 0 or not np.isfinite(wav).all():
                raise ValueError("VoxCPM2 returned an invalid waveform")
            sf.write(line.output, wav, model.tts_model.sample_rate, format="WAV")

    _with_model(model_dir, device, cancel, report, work)


def _design(model_dir: Path, design: VoiceDesign, device: str, report: ProgressCb, cancel: CancelToken) -> None:
    import numpy as np
    import soundfile as sf

    def work(model, seed_all) -> None:
        rate = model.tts_model.sample_rate
        prompt = design_prompt(design.description, design.text)
        for attempt in range(DESIGN_TRIES):
            report(0.30 + 0.60 * attempt / DESIGN_TRIES, f"Voice: designing, take {attempt + 1}/{DESIGN_TRIES}")
            seed_all(design.seed + attempt)
            wav = np.asarray(model.generate(text=prompt, cfg_value=GUIDANCE, inference_timesteps=DESIGN_STEPS),
                             dtype=np.float32).reshape(-1)
            cancel.check()
            if acceptable(wav, rate):
                sf.write(design.output, wav, rate, format="WAV")
                return
        raise ValueError(f"VoxCPM2 made no usable {DESIGN_SECONDS[0]:.0f}–{DESIGN_SECONDS[1]:.0f} second "
                         "sample for this description. Describe the voice differently and try again.")

    _with_model(model_dir, device, cancel, report, work)


_TASKS = {"lines": _generate, "design": _design}


def _worker(task: str, model_dir: Path, payload, device: str, writer, vram_limit_gib: int | None) -> None:
    os.environ["TQDM_DISABLE"] = "1"
    with writer:
        try:
            if vram_limit_gib is not None and device.startswith("cuda"):
                from ..core.memory import limit_cuda

                limit_cuda(device, vram_limit_gib)
            _TASKS[task](model_dir, payload, device, lambda fraction, desc: writer.send(("progress", fraction, desc)),
                         CancelToken())
        except Exception as exc:
            writer.send(("error", str(exc)[:1000] or type(exc).__name__))
            raise SystemExit(1) from None


def _isolated(task: str, model_dir: Path, payload, device: str, report: ProgressCb, cancel: CancelToken,
              vram_limit_gib: int | None) -> None:
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(task, model_dir, payload, device, writer, vram_limit_gib),
                              name="contsy-speech-worker", daemon=True)
    error = None
    try:
        cancel.check()
        process.start()
        writer.close()
        while True:
            cancel.check()
            if not reader.poll(0.1):
                continue
            try:
                event = reader.recv()
            except EOFError:
                break
            if event[0] == "progress":
                report(event[1], event[2])
            elif event[0] == "error":
                error = event[1]
        while process.is_alive():
            cancel.check()
            process.join(0.1)
        cancel.check()
        if error or process.exitcode != 0:
            raise RuntimeError(f"VoxCPM2 worker: {error or f'exit code {process.exitcode}'}")
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join()
        reader.close()
        writer.close()
        process.close()


def _model_dir(model_dir: str | Path) -> Path:
    model_dir = Path(model_dir).resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"VoxCPM2 model is missing: {model_dir}")
    return model_dir


def _staged(output: Path, cleanup: ExitStack) -> Path:
    """An empty file beside `output` that is removed unless it is moved into place."""
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.stem}-", suffix=".wav",
                                     delete=False) as file:
        path = Path(file.name)
    cleanup.callback(path.unlink, missing_ok=True)
    return path


def _reporter(progress_cb: ProgressCb | None, cancel: CancelToken) -> ProgressCb:
    def report(fraction: float, desc: str) -> None:
        if progress_cb is not None:
            progress_cb(fraction, desc)
        cancel.check()
    return report


def synthesize_many(model_dir: str | Path, lines: Iterable[SpeechLine], *, device: str = "cuda",
                    isolated: bool = False, progress_cb: ProgressCb | None = None,
                    cancel: CancelToken | None = None, vram_limit_gib: int | None = None) -> Path:
    """Generate each line with one model load and return the path of the last WAV.

    Each output is staged beside its destination. A failed or cancelled
    generation leaves previously published outputs intact. With `isolated`,
    the engine can cancel during a line by terminating its own Python worker.
    """
    cancel = cancel or CancelToken()
    cancel.check()
    model_dir = _model_dir(model_dir)
    requested = [SpeechLine(line.text, Path(line.reference).resolve(), Path(line.output).resolve()) for line in lines]
    if not requested or any(not line.text.strip() for line in requested):
        raise ValueError("Enter some text for each spoken line.")
    for line in requested:
        if not line.reference.is_file():
            raise FileNotFoundError(f"Voice sample is missing: {line.reference}")
    report = _reporter(progress_cb, cancel)

    with ExitStack() as cleanup:
        staged = [SpeechLine(line.text, line.reference, _staged(line.output, cleanup)) for line in requested]
        if isolated:
            _isolated("lines", model_dir, staged, device, report, cancel, vram_limit_gib)
        else:
            _generate(model_dir, staged, device, report, cancel)
        if any(line.output.stat().st_size == 0 for line in staged):
            raise RuntimeError("VoxCPM2 did not write every requested WAV")
        report(1.0, "Voice: done")
        for temp, requested_line in zip(staged, requested):
            temp.output.replace(requested_line.output)
    return requested[-1].output


def design_voice(model_dir: str | Path, design: VoiceDesign, *, device: str = "cuda", isolated: bool = False,
                 progress_cb: ProgressCb | None = None, cancel: CancelToken | None = None,
                 vram_limit_gib: int | None = None) -> Path:
    """Write a new voice sample (WAV at the model's rate) for `design.description` and return its path.

    The sample is staged beside its destination, so a failed or cancelled design leaves an existing file as it
    was. Raises ValueError for an empty description or when no seed gave a usable take."""
    cancel = cancel or CancelToken()
    cancel.check()
    model_dir = _model_dir(model_dir)
    if not clean_description(design.description):
        raise ValueError("Describe the voice first.")
    if not design.text.strip():
        raise ValueError("Enter the sentence the voice sample reads.")
    output = Path(design.output).resolve()
    report = _reporter(progress_cb, cancel)
    with ExitStack() as cleanup:
        staged = VoiceDesign(design.description, _staged(output, cleanup), design.text, design.seed)
        if isolated:
            _isolated("design", model_dir, staged, device, report, cancel, vram_limit_gib)
        else:
            _design(model_dir, staged, device, report, cancel)
        if Path(staged.output).stat().st_size == 0:
            raise RuntimeError("VoxCPM2 did not write the voice sample")
        report(1.0, "Voice: done")
        Path(staged.output).replace(output)
    return output
