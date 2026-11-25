# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Sound of one Timeline scene, made before its picture: the spoken lines on the scene clock, the
background bed with its timed effects, and previews mixed the way the export mixes them.

prepare() synthesizes every line in one VoxCPM2 batch and places the lines at their entry times
(a line that runs past the scene end is an error), then generates the EzAudio-XL bed and each effect and
lays the effects over the bed. A nearly silent bed (a generator can return one) is made again with
other seeds, then with an audible fallback prompt. The result is a folder with a manifest
(sound.json) that load() reads back without decoding any audio, so the editor can list and play an
audition and generation can use it instead of synthesizing again. Like the generated sound it comes
from, the background keeps the export's handles: it starts TIMELINE_AUDIO_HANDLE before the scene
and ends as long after it. The effects laid into it are also kept alone on the same clock (the
overlay), so the export levels only the room tone and lays the effects over it afterwards.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from engine.media import audio as media_audio
from engine.media import video as media_video
from engine.media.errors import MediaError

from . import cancel, characters, dialogue, engine_runtime, ffmpeg_utils, scene_parts, sound, tts, voices
from .scene_parts import Cue, Effect

ProgressCb = Callable[[float, str], None]

# v3: effects start at their own onset (a click at 0.03 s was cut); v4: the sound model is EzAudio-XL, so a
# background or effect made before would no longer match what generation makes now
SOUND_VERSION = "scene-sound-v4"
MANIFEST = "sound.json"
RATE = media_audio.RATE
HANDLE = media_audio.TIMELINE_AUDIO_HANDLE
EFFECT_OVER_BED_DB = 9.0     # the loud part of an effect is lifted at least this far above the bed
MAX_EFFECT_BOOST_DB = 24.0   # ...but never by more than this
EFFECT_FADE_IN, EFFECT_FADE_OUT = 0.02, 0.05  # s: an effect starts and stops without a click
EFFECT_PRE_ROLL = 0.01  # s kept before the measured start of an effect's sound (_onset)
PEAK = 0.98                  # a background whose effects would clip is scaled below this peak
QUIET_BED = 0.01             # -40 dBFS: a bed the export's leveling cannot lift above this is nearly silent,
AUDIBLE_SHARE = 0.5          # ...and so is one with fewer than half of its 50 ms blocks audible
BED_RETRIES = 2              # a nearly silent bed is made again with seed + 1, seed + 2, then the fallback prompt
OVERLAY = "overlay.wav"

_KEY = re.compile(r"[0-9a-f]{64}")
_FILE = re.compile(r"[a-z0-9_]{1,40}\.wav")
_SPEAKER_FIELDS = frozenset(("gender", "voice", "character", "who", "name"))
_LINE_FIELDS = frozenset(("text", "speaker", "file", "start", "end"))
_EFFECT_FIELDS = frozenset(("text", "at", "seconds", "file"))
_BED_FIELDS = frozenset(("level", "audible", "attempts"))
_BLOCK = round(0.05 * RATE)  # level measurements work on 50 ms blocks, as the export's leveling does

_DIGESTS: dict[tuple[str, int, int], str] = {}
_DIGESTS_LOCK = threading.Lock()
_LOUDNESS: dict[tuple, tuple[float, float]] = {}
_LOUDNESS_LOCK = threading.Lock()


@dataclass(frozen=True)
class Speaker:
    """Who says a line: gender ("female" | "male"), voice slug, character slug ("" for a library
    voice), the lip-sync identity app/lipsync.run_dialogue matches to a face ("female#char:rose",
    "male#adam") and the display name."""
    gender: str
    voice: str
    character: str
    who: str
    name: str


@dataclass(frozen=True)
class SceneSound:
    """A prepared scene sound (see prepare). Paths point into `folder`."""
    folder: Path
    key: str
    seconds: int | float  # whole seconds or tenths of one (scene_parts.length)
    texts: tuple[str, ...]
    speakers: tuple[Speaker, ...]
    lines: tuple[Path, ...]                    # one 48 kHz WAV per line, as synthesized
    timings: tuple[tuple[float, float], ...]   # (start, end) of every line on the scene clock
    voice: Optional[Path]                      # all lines on the scene clock, exactly `seconds` long
    background: Optional[Path]                 # bed + effects, seconds + 2 * HANDLE long (starts HANDLE early)
    background_prompt: str
    effects: tuple[Path, ...]                  # one WAV per effect as generated (for auditioning)
    effect_texts: tuple[str, ...]
    overlay: Optional[Path] = None             # the effects alone as laid into `background` (same clock)
    bed_level: Optional[float] = None          # background_loudness of the kept bed: the leveled median level
    bed_audible: Optional[float] = None        # ...and the share of its audible 50 ms blocks


# --- speakers -------------------------------------------------------------------------------------

def _character_name(slug: str) -> str:
    try:
        profile = characters.get(slug)
    except ValueError:
        profile = None
    return profile["name"] if profile else slug


def resolve_speakers(cues: Sequence[Cue], members: Sequence[dict], pool=None) -> list[Speaker]:
    """Who says each line and with which voice.

    members: the scene's character profiles (the cast video_gen gets); pool: the voice library
    (voices.all_voices() by default). A character speaker must be one of the members, "@voice" a
    voice of the library and "" stands for the scene's only character. Characters keep their own
    voice, else they get the first voice of their gender that no other character in the scene
    uses, as dialogue.voices_for picks them."""
    pool = voices.all_voices() if pool is None else list(pool)
    members = [member for member in (members or ()) if member]
    by_slug = {member.get("slug"): member for member in members}
    lines: list[dialogue.Line] = []
    names: list[str] = []
    for number, cue in enumerate(cues, 1):
        if cue.voice:
            found = next((item for item in pool if item.slug == cue.voice), None)
            if found is None:
                raise ValueError(f"Line {number}: no voice named @{cue.voice}.")
            lines.append(dialogue.Line(found.gender, cue.text, who=f"{found.gender}#{found.slug}",
                                       voice=found.slug))
            names.append(found.name)
            continue
        if cue.character:
            member = by_slug.get(cue.character)
            if member is None:
                raise ValueError(f"Line {number}: {_character_name(cue.character)} is not in this scene.")
        elif len(members) == 1:
            member = members[0]
        else:
            raise ValueError(f"Line {number}: choose who says it.")
        gender = "male" if member.get("gender") == "man" else "female"
        own = voices.get(member.get("voice") or "", pool)
        slug = member["slug"]
        lines.append(dialogue.Line(gender, cue.text, who=f"{gender}#char:{slug}",
                                   voice=own.slug if own else "", character=slug))
        names.append(member.get("name") or slug)
    chosen = dialogue.voices_for(lines, voices.catalog(pool))
    speakers = []
    for number, (line, voice, name) in enumerate(zip(lines, chosen, names), 1):
        if not voice:
            kind = "man's" if line.speaker == "male" else "woman's"
            raise ValueError(f"Line {number}: the voice library has no {kind} voice. Add one on the Voices page.")
        speakers.append(Speaker(line.speaker, voice, line.character, line.who, name))
    return speakers


# --- identity ---------------------------------------------------------------------------------------

def background_prompt(picture_text: str, background_text: str) -> str:
    """The sound prompt of the scene's background: the user's description, else the sounds found in
    the picture text."""
    return sound.prompt_from_scene(picture_text or "", background_text or "")


def _sample_digest(path) -> str:
    """sha256 of a voice sample, remembered by (path, size, modification time)."""
    if not path:
        return "missing"
    try:
        resolved = Path(path).resolve()
        stat = resolved.stat()
    except OSError:
        return "missing"
    memo = (str(resolved), stat.st_size, stat.st_mtime_ns)
    with _DIGESTS_LOCK:
        found = _DIGESTS.get(memo)
    if found is None:
        try:
            with resolved.open("rb") as stream:
                found = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError:
            return "missing"
        with _DIGESTS_LOCK:
            if len(_DIGESTS) >= 512:
                _DIGESTS.clear()
            _DIGESTS[memo] = found
    return found


def _sample_path(slug: str, library: list) -> Optional[str]:
    """The sample synthesis will use for a voice slug (tts.resolve's fallbacks for a missing one)."""
    found = voices.get(slug, library) or tts.resolve(slug)
    return found.path if found else None


def sound_key(cues: Sequence[Cue], speakers: Sequence[Speaker], background_prompt: str,
              effects: Sequence[Effect], seconds: int, seed: int, *, voice: bool = True,
              background: bool = True) -> str:
    """Identity of a prepared scene sound (sha256 hex). The dialogue part covers every line's text,
    start, speaker, voice and the digest of that voice's sample; the background part the prompt and
    the effects; both the scene length, the seed, the models and SOUND_VERSION. A part that is not
    prepared (voice=False / background=False) does not count. Sample digests are remembered by
    file stat, so the key is cheap to compute again. An effect that starts after the scene end is not heard and
    does not count (scene_parts.heard)."""
    cues, speakers, effects = tuple(cues or ()), tuple(speakers or ()), scene_parts.heard(effects or (), seconds)
    payload = {"version": SOUND_VERSION, "seconds": scene_parts.length(seconds), "seed": int(seed),
               "voice": bool(voice), "background": bool(background)}
    if voice:
        if len(speakers) != len(cues):
            raise ValueError("Each spoken line needs its speaker.")
        library = voices.all_voices() if cues else []
        payload["speech_model"] = tts.MODEL_ID
        payload["lines"] = [[cue.text, cue.start, speaker.voice, speaker.who,
                             _sample_digest(_sample_path(speaker.voice, library))] + ([cue.pace] if cue.pace else [])
                            for cue, speaker in zip(cues, speakers)]
    if background:
        payload["background_model"] = sound.MODEL_ID
        payload["prompt"] = background_prompt or ""
        payload["effects"] = [[effect.text, effect.at, effect.seconds] for effect in effects]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


# --- generation -------------------------------------------------------------------------------------

def _report(progress_cb: Optional[ProgressCb], fraction: float, message: str) -> None:
    if progress_cb:
        progress_cb(min(max(fraction, 0.0), 1.0), message)


def _span(progress_cb: Optional[ProgressCb], low: float, high: float, prefix: str = "") -> ProgressCb:
    def report(fraction: float, message: str) -> None:
        _report(progress_cb, low + (high - low) * min(max(fraction, 0.0), 1.0), prefix + message)
    return report


def _effect_seed(seed: int, number: int) -> int:
    return (int(seed) + 1009 * number) % (2 ** 31 - 1)


def _read_mono(path: Path) -> np.ndarray:
    try:
        samples, _ = media_audio.read(path, rate=RATE, channels=1)
    except MediaError as error:
        raise ValueError(f"Could not read the generated sound: {error}") from error
    return samples


def _level(samples: np.ndarray, quantile: float) -> float:
    """Level of the audible 50 ms blocks at a quantile (0.5: the steady level, 0.9: the loud part)."""
    count = samples.shape[1] // _BLOCK
    if count == 0:
        return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
    blocks = samples[:, :count * _BLOCK].reshape(samples.shape[0], count, _BLOCK)
    levels = np.sqrt(np.mean(blocks * blocks, axis=(0, 2)))
    audible = levels[levels > 1e-4]
    return float(np.quantile(audible, quantile)) if audible.size else 0.0


def _fade(piece: np.ndarray) -> None:
    """Short linear fades at both ends of an effect, in place."""
    length = piece.shape[1]
    for seconds, at_start in ((EFFECT_FADE_IN, True), (EFFECT_FADE_OUT, False)):
        size = min(int(round(seconds * RATE)), length // 2)
        if size <= 0:
            continue
        ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)
        if at_start:
            piece[:, :size] *= ramp
        else:
            piece[:, length - size:] *= ramp[::-1]


def _onset(samples: np.ndarray) -> int:
    """Index where a generated effect's sound begins: its first 5 ms block within 30 dB of its loudest one, minus
    EFFECT_PRE_ROLL. The generator returns an effect padded like a background (half a second more than asked) and may
    put a transient anywhere in it, also in its first milliseconds: a fixed trim of the background's handle cut a
    camera click out completely."""
    block = max(1, round(0.005 * RATE))
    count = samples.shape[1] // block
    if not count:
        return 0
    levels = np.sqrt(np.mean(samples[:, :count * block].reshape(samples.shape[0], count, block) ** 2, axis=(0, 2)))
    loudest = float(np.max(levels))
    if loudest <= 1e-6:
        return 0
    first = int(np.argmax(levels >= loudest * 10 ** (-30 / 20)))
    return max(0, first * block - int(round(EFFECT_PRE_ROLL * RATE)))


def _layer(bed_path: Path, effects: Sequence[Effect], effect_paths: Sequence[Path],
           seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """(background, overlay): the bed with every effect laid over it at HANDLE + effect.at (the bed
    starts HANDLE before the scene), and the effects alone on the same clock. Each effect starts where its
    sound starts (_onset), is cut at the scene end, fades in and out briefly and is lifted to at least
    EFFECT_OVER_BED_DB above the bed's steady level; the sum never clips (both are scaled alike)."""
    length = int(round((seconds + 2 * HANDLE) * RATE))
    bed = _read_mono(bed_path)
    base = np.zeros((1, length), dtype=np.float32)
    base[:, :min(length, bed.shape[1])] = bed[:, :length]
    bed_level = _level(base, 0.5)
    overlay = np.zeros((1, length), dtype=np.float32)
    handle = int(round(HANDLE * RATE))
    scene_end = handle + int(round(seconds * RATE))
    for effect, path in zip(effects, effect_paths):
        start = handle + int(round(effect.at * RATE))
        sound = _read_mono(path)
        onset = _onset(sound)  # the effect starts where its sound starts, not after a bed-style handle
        piece = sound[:, onset:onset + int(round(effect.seconds * RATE))]
        piece = piece[:, :max(scene_end - start, 0)].copy()
        if not piece.shape[1]:
            continue
        effect_level = _level(piece, 0.9)
        if effect_level > 0 and bed_level > 0:
            target = bed_level * media_audio.gain(EFFECT_OVER_BED_DB)
            boost = min(max(target / effect_level, 1.0), media_audio.gain(MAX_EFFECT_BOOST_DB))
            piece *= np.float32(boost)
        _fade(piece)
        overlay[:, start:start + piece.shape[1]] += piece
    layered = base + overlay
    peak = float(np.max(np.abs(layered))) if layered.size else 0.0
    if peak > PEAK:
        layered *= np.float32(PEAK / peak)
        overlay *= np.float32(PEAK / peak)
    return layered, overlay


def quiet(level: float, audible: float) -> bool:
    """A bed the export would hear as nearly silent (see background_loudness)."""
    return level < QUIET_BED or audible < AUDIBLE_SHARE


def _measure(path: Path, seconds: float) -> tuple[float, float]:
    try:
        return media_audio.background_loudness(path, seconds)
    except MediaError as error:
        raise ValueError(f"Could not read the generated sound: {error}") from error


def _make_bed(client, prompt: str, seconds: float, seed: int, out_path: Path,
              progress_cb: Optional[ProgressCb]) -> tuple[dict, ...]:
    """The generated bed for `prompt` in out_path. A nearly silent one (quiet()) is made again with seed + 1 and
    seed + 2, then with the audible fallback prompt; the loudest one is kept. Returns every attempt as
    {"prompt", "seed", "level", "audible"}, the kept one marked "kept": True."""
    fallback = sound.prompt_from_scene("", "")
    plans = [(prompt, (int(seed) + n) % (2 ** 31 - 1)) for n in range(BED_RETRIES + 1)]
    plans.append((fallback, int(seed) if fallback != prompt else (int(seed) + BED_RETRIES + 1) % (2 ** 31 - 1)))
    attempts: list[dict] = []
    files: list[Path] = []
    try:
        for number, (text, attempt_seed) in enumerate(plans):
            if number:
                cancel.check()
            target = out_path if number == 0 else out_path.with_name(f"{out_path.stem}_try{number}.wav")
            files.append(target)
            label = "" if number == 0 else "the bed was nearly silent, making it again: "
            ok, message = sound.generate(client, "", text, seconds, attempt_seed, target,
                                         _span(progress_cb, 0.0, 1.0, label))
            if not ok:
                raise ValueError(message)
            level, audible = _measure(target, seconds)
            attempts.append({"prompt": text, "seed": attempt_seed, "level": round(level, 6),
                             "audible": round(audible, 4)})
            if not quiet(level, audible):
                break
        best = max(range(len(attempts)), key=lambda index: (not quiet(attempts[index]["level"],
                                                                      attempts[index]["audible"]),
                                                            attempts[index]["level"]))
        if files[best] != out_path:
            os.replace(files[best], out_path)
        attempts[best]["kept"] = True
    finally:
        for path in files:
            if path != out_path:
                path.unlink(missing_ok=True)
    return tuple(attempts)


@dataclass(frozen=True)
class Background:
    """What generate_background made: each effect as generated, the overlay (the effects as laid into
    the background, or None without effects) and every bed it made (see _make_bed)."""
    effects: tuple[Path, ...]
    overlay: Optional[Path]
    attempts: tuple[dict, ...]

    @property
    def kept(self) -> dict:
        return next((attempt for attempt in self.attempts if attempt.get("kept")), self.attempts[-1])


def generate_background(client, prompt: str, effects: Sequence[Effect], seconds: float, seed: int,
                        out_path: str | Path, progress_cb: Optional[ProgressCb] = None, *,
                        effect_paths: Sequence[str | Path],
                        overlay_path: Optional[str | Path] = None) -> Background:
    """The scene background in `out_path`: the EzAudio-XL bed for `prompt` (`seconds` + 2 * HANDLE long, it
    starts HANDLE before the scene) with `effects` laid over it. A nearly silent bed is made again
    (_make_bed). Each effect is generated for its own length into effect_paths, kept as generated; with overlay_path the effects as laid
    into the background are also saved alone there. Without effects the bed is the background,
    unchanged. The bed, its retries and the effects read EzAudio's networks from disk once (the engine's audio
    session). Raises ValueError when the generation fails."""
    session = getattr(client, "audio_session", None)
    with session() if callable(session) else contextlib.nullcontext():
        return _background(client, prompt, effects, seconds, seed, out_path, progress_cb, effect_paths, overlay_path)


def _background(client, prompt: str, effects: Sequence[Effect], seconds: float, seed: int, out_path: str | Path,
                progress_cb: Optional[ProgressCb], effect_paths: Sequence[str | Path],
                overlay_path: Optional[str | Path]) -> Background:
    out_path = Path(out_path)
    effects = tuple(effects or ())
    effect_paths = tuple(Path(path) for path in effect_paths)
    if len(effect_paths) != len(effects):
        raise ValueError("Each sound effect needs its own file.")
    total = float(seconds) + sum(effect.seconds for effect in effects)
    if not effects:
        return Background((), None, _make_bed(client, prompt, seconds, seed, out_path, _span(progress_cb, 0.0, 1.0)))
    bed_path = out_path.with_name(f"{out_path.stem}_bed.wav")
    overlay = Path(overlay_path) if overlay_path is not None else None
    try:
        attempts = _make_bed(client, prompt, seconds, seed, bed_path, _span(progress_cb, 0.0, float(seconds) / total))
        done = float(seconds)
        for number, (effect, path) in enumerate(zip(effects, effect_paths), 1):
            cancel.check()
            label = f"Sound effect {number}/{len(effects)}: "
            ok, message = sound.generate(client, "", effect.text, effect.seconds, _effect_seed(seed, number), path,
                                         _span(progress_cb, done / total, (done + effect.seconds) / total, label))
            if not ok:
                raise ValueError(f"Effect {number}: {message}")
            done += effect.seconds
        _report(progress_cb, 0.99, "Laying the sound effects over the background…")
        try:
            layered, alone = _layer(bed_path, effects, effect_paths, seconds)
            media_audio.write_wav(out_path, layered, RATE)
            if overlay is not None:
                media_audio.write_wav(overlay, alone, RATE)
        except MediaError as error:
            raise ValueError(f"Could not save the scene background: {error}") from error
    finally:
        bed_path.unlink(missing_ok=True)
    return Background(effect_paths, overlay, attempts)


def _record(path: Path) -> dict:
    stat = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def _check_destination(folder: Path) -> None:
    """A package may replace an empty folder or an older package, never anything else."""
    if folder.exists() and (not folder.is_dir() or (any(folder.iterdir()) and not (folder / MANIFEST).is_file())):
        raise OSError(f"{folder} exists and is not a scene sound folder.")


def _install(staged: Path, folder: Path) -> None:
    """Put a finished package in place of `folder`. An older package there (one that did not load)
    is replaced; any other existing folder is left alone."""
    old = None
    if folder.exists():
        _check_destination(folder)
        old = folder.with_name(f".{folder.name}.{uuid.uuid4().hex[:12]}.old")
        os.replace(folder, old)
    try:
        os.replace(staged, folder)
    except OSError:
        if old is not None:
            os.replace(old, folder)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def synthesize_lines(scenes: Sequence[tuple[Sequence[Cue], Sequence[Speaker]]], folder: str | Path,
                     progress_cb: Optional[ProgressCb] = None) -> list[tuple[Path, ...]]:
    """The lines of several scenes ((cues, speakers) each) in one VoxCPM2 batch: one model load for all of
    them instead of one per scene. Every line is seeded on its own (engine/pipelines/speech.py), so it
    sounds as it would in its scene's own batch. Returns each scene's line files in `folder`, in the order
    prepare(lines=...) takes them. Raises ValueError when synthesis fails."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str, str]] = []
    result: list[tuple[Path, ...]] = []
    for number, (cues, speakers) in enumerate(scenes, 1):
        if len(cues) != len(speakers):
            raise ValueError("Each spoken line needs its speaker.")
        paths = tuple(folder / f"scene{number}_line{line}.wav" for line in range(1, len(cues) + 1))
        items += [(cue.text, speaker.voice, str(path)) for cue, speaker, path in zip(cues, speakers, paths)]
        result.append(paths)
    if items:
        ok, message = tts.synthesize_many(items, progress_cb)
        if not ok:
            raise ValueError(message)
        missing = next((path for paths in result for path in paths if not path.is_file()), None)
        if missing is not None:
            raise ValueError(f"Speech synthesis did not produce {missing.name}.")
    return result


def prepare(cues: Sequence[Cue], speakers: Sequence[Speaker], background_prompt: str,
            effects: Sequence[Effect], seconds: int, seed: int, folder: str | Path,
            progress_cb: Optional[ProgressCb] = None, *, client=None, voice: bool = True,
            background: bool = True, lines: Optional[Sequence[str | Path]] = None) -> SceneSound:
    """Make the scene's sound in `folder` and return it (GPU).

    voice: every line in one VoxCPM2 batch, placed on the scene clock at its measured length
    (scene_parts.timings); overlapping lines are summed. The scene keeps its length: a line that runs past the
    scene end is cut there and one that starts after it is not heard (dialogue.place), never an error. lines:
    the cues already synthesized, one WAV each (synthesize_lines, for several scenes in one batch): they are
    copied instead. background: the EzAudio-XL bed with each effect laid over it (generate_background); an
    effect that starts after the scene end is not generated (scene_parts.heard). Unused GPU memory is given back
    before the speech and between the speech and the background (models kept warm stay). The package
    is built in a temporary sibling folder and renamed into place; a folder that already holds this
    sound is returned as it is. Inputs are never deleted."""
    if not scene_parts.is_length(seconds) or seconds < 1:
        raise ValueError("A scene lasts at least one second, in tenths of a second.")
    seconds = scene_parts.length(seconds)  # whole seconds or tenths of one
    effects = scene_parts.heard(effects or (), seconds)
    cues, speakers, effects = tuple(cues or ()), tuple(speakers or ()), tuple(effects or ())
    if not (voice or background):
        raise ValueError("Choose the dialogue or the background to prepare.")
    if voice and len(speakers) != len(cues):
        raise ValueError("Each spoken line needs its speaker.")
    if lines is not None and (not voice or len(lines) != len(cues)):
        raise ValueError("Each spoken line needs its synthesized file.")
    scene_parts.check(cues if voice else (), effects if background else (), seconds)
    folder = Path(folder)
    key = sound_key(cues, speakers, background_prompt, effects, seconds, seed, voice=voice, background=background)
    existing = load(folder, key)
    if existing is not None:
        _report(progress_cb, 1.0, "Sound ready.")
        return existing
    _check_destination(folder)
    talk = voice and bool(cues)
    folder.parent.mkdir(parents=True, exist_ok=True)
    staged = folder.parent / f".{folder.name}.{uuid.uuid4().hex[:12]}.tmp"
    staged.mkdir()
    try:
        engine = client
        if engine is None and (talk or background):
            engine = engine_runtime.engine()
            # the models' files at the disk's speed while they load (only in a queued job)
            engine_runtime.read_ahead_sound(talk and lines is None, background)
        speech_end = (0.4 if background else 0.95) if talk else 0.0
        line_paths: list[Path] = []
        timings: list[tuple[float, float]] = []
        voice_path = None
        if talk:
            line_paths = [staged / f"line{number}.wav" for number in range(1, len(cues) + 1)]
            if lines is not None:
                _report(progress_cb, speech_end, "Voice: lines ready.")
                for source, path in zip(lines, line_paths):
                    shutil.copyfile(source, path)
            else:
                _report(progress_cb, 0.0, "Voice: preparing…")
                engine.trim()  # the card for VoxCPM2; models kept warm wait in RAM
                ok, message = tts.synthesize_many(
                    [(cue.text, speaker.voice, str(path)) for cue, speaker, path in zip(cues, speakers, line_paths)],
                    _span(progress_cb, 0.0, speech_end))
                if not ok:
                    raise ValueError(message)
            for number, path in enumerate(line_paths, 1):
                if not path.is_file():
                    raise ValueError(f"Speech synthesis did not produce line {number}.")
            _pace(cues, line_paths)
            measured = dialogue.durations(line_paths)
            timings = scene_parts.timings(cues, measured)  # past the scene end: cut by dialogue.place
            voice_path = staged / "voice.wav"
            dialogue.place(line_paths, [start for start, _ in timings], voice_path, seconds)
        background_path = None
        effect_paths: tuple[Path, ...] = ()
        made = None
        if background:
            cancel.check()
            if talk and lines is None:
                engine.trim()  # VoxCPM2's leftovers off the card before EzAudio comes in
            background_path = staged / "background.wav"
            made = generate_background(
                engine, background_prompt, effects, seconds, seed, background_path,
                _span(progress_cb, speech_end, 0.98, "Background audio: "),
                effect_paths=[staged / f"effect{n}.wav" for n in range(1, len(effects) + 1)],
                overlay_path=staged / OVERLAY if effects else None)
            effect_paths = made.effects
        overlay = made.overlay if made is not None else None
        files = [*line_paths, *([voice_path] if voice_path else []), *([background_path] if background_path else []),
                 *effect_paths, *([overlay] if overlay else [])]
        manifest = {
            "version": SOUND_VERSION, "key": key, "seconds": seconds,
            "lines": [{"text": cue.text, "speaker": asdict(speaker), "file": path.name, "start": start, "end": end}
                      for cue, speaker, path, (start, end) in zip(cues, speakers, line_paths, timings)],
            "voice": voice_path.name if voice_path else None,
            "background": background_path.name if background_path else None,
            "background_prompt": (background_prompt or "") if background else "",
            "effects": [{"text": effect.text, "at": effect.at, "seconds": effect.seconds, "file": path.name}
                        for effect, path in zip(effects, effect_paths)],
            "overlay": overlay.name if overlay else None,
            # the beds made for the background: a nearly silent one is made again (_make_bed)
            "bed": ({"level": made.kept["level"], "audible": made.kept["audible"], "attempts": list(made.attempts)}
                    if made is not None else None),
            "files": {path.name: _record(path) for path in files},
        }
        (staged / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        _install(staged, folder)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    result = load(folder, key)
    if result is None:
        raise OSError(f"The prepared sound in {folder} could not be read back.")
    _report(progress_cb, 1.0, "Sound ready.")
    return result


def tempo_for(text: str, seconds: float, pace: float) -> float:
    """The tempo (1 = as synthesized) that brings a line of `seconds` to at most `pace` words a minute, never under
    scene_parts.SLOWEST_TEMPO."""
    words = len(re.findall(r"[\w']+", text or ""))
    if not pace or seconds <= 0 or not words:
        return 1.0
    rate = words / seconds * 60.0
    return 1.0 if rate <= pace else max(scene_parts.SLOWEST_TEMPO, pace / rate)


def _pace(cues: Sequence[Cue], paths: Sequence[Path]) -> None:
    """Slow down, in place, every line spoken faster than its cue's pace (ffmpeg atempo: the pitch stays)."""
    from engine.media import audio as media_audio

    for cue, path in zip(cues, paths):
        if not cue.pace:
            continue
        samples, rate = media_audio.read(path, rate=None, channels=None)
        tempo = tempo_for(cue.text, samples.shape[1] / rate if rate else 0.0, cue.pace)
        if tempo < 0.999:
            slowed, _ = media_audio.run_chain(samples, rate, f"atempo={tempo:.4f}")
            media_audio.write_wav(path, slowed, rate)


# --- reading and previews ---------------------------------------------------------------------------

def _time(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError("Invalid time.")
    return float(value)


def load(folder: str | Path, key: Optional[str] = None, *, verify: bool = False) -> Optional[SceneSound]:
    """The prepared sound in `folder`, or None when there is none, it is incomplete, a file changed
    (size or modification time; with verify=True also its sha256) or it is not the sound `key` names.
    Cheap without verify: it reads the manifest and looks at the files, no audio is decoded."""
    try:
        folder = Path(folder)
        data = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != SOUND_VERSION:
            return None
        found = data["key"]
        if not isinstance(found, str) or not _KEY.fullmatch(found) or (key is not None and found != key):
            return None
        seconds, files = data["seconds"], data["files"]
        if not scene_parts.is_length(seconds) or seconds < 1 or not isinstance(files, dict):
            return None

        def media(name) -> Path:
            if not isinstance(name, str) or not _FILE.fullmatch(name) or not isinstance(files.get(name), dict):
                raise ValueError("Invalid sound file.")
            path, record = folder / name, files[name]
            stat = path.stat()
            if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
                raise ValueError("A sound file changed.")
            if verify:
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                        raise ValueError("A sound file changed.")
            return path

        texts, speakers, lines, timings = [], [], [], []
        for line in data["lines"]:
            if not isinstance(line, dict) or set(line) != _LINE_FIELDS or not isinstance(line["text"], str):
                return None
            person = line["speaker"]
            if (not isinstance(person, dict) or set(person) != _SPEAKER_FIELDS
                    or not all(isinstance(value, str) for value in person.values())
                    or person["gender"] not in ("female", "male")):
                return None
            start, end = _time(line["start"]), _time(line["end"])
            if end < start:
                return None
            texts.append(line["text"])
            speakers.append(Speaker(**person))
            lines.append(media(line["file"]))
            timings.append((start, end))
        voice = media(data["voice"]) if data["voice"] is not None else None
        background = media(data["background"]) if data["background"] is not None else None
        if bool(lines) != (voice is not None):
            return None
        effects, effect_texts = [], []
        for effect in data["effects"]:
            if (not isinstance(effect, dict) or set(effect) != _EFFECT_FIELDS or not isinstance(effect["text"], str)
                    or background is None or _time(effect["seconds"]) <= 0):
                return None
            _time(effect["at"])
            effects.append(media(effect["file"]))
            effect_texts.append(effect["text"])
        prompt = data["background_prompt"]
        if not isinstance(prompt, str):
            return None
        overlay = media(data["overlay"]) if data["overlay"] is not None else None
        if overlay is not None and (background is None or not effects):
            return None
        bed, level, audible = data["bed"], None, None
        if (bed is None) != (background is None):
            return None
        if bed is not None:
            if not isinstance(bed, dict) or set(bed) != _BED_FIELDS or not isinstance(bed["attempts"], list):
                return None
            level, audible = _time(bed["level"]), _time(bed["audible"])
        return SceneSound(folder, found, seconds, tuple(texts), tuple(speakers), tuple(lines), tuple(timings),
                          voice, background, prompt, tuple(effects), tuple(effect_texts), overlay, level, audible)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _stat(path) -> tuple:
    if path is None:
        return ()
    try:
        resolved = Path(path).resolve()
        info = resolved.stat()
    except OSError:
        return (str(path),)
    return str(resolved), info.st_size, info.st_mtime_ns


def bed_loudness(background: str | Path, seconds: float, effects: Optional[str | Path] = None) -> tuple[float, float]:
    """engine background_loudness of a stored scene background without its effects overlay: (leveled
    median level, audible share). Remembered by the files' size and modification time, so the editor
    can ask on every refresh. Raises ValueError when the file cannot be read."""
    key = (_stat(background), _stat(effects), float(seconds))
    with _LOUDNESS_LOCK:
        found = _LOUDNESS.get(key)
    if found is None:
        try:
            found = media_audio.background_loudness(background, seconds, effects=effects)
        except MediaError as error:
            raise ValueError(f"Could not read the scene background: {error}") from error
        with _LOUDNESS_LOCK:
            if len(_LOUDNESS) >= 256:
                _LOUDNESS.clear()
            _LOUDNESS[key] = found
    return found


def mix_stems(background: Optional[str | Path], voice: Optional[str | Path], seconds: int, voice_db: float,
              background_db: float, out: str | Path, *, effects: Optional[str | Path] = None) -> Path:
    """A scene's background (handle-padded, with its effects `overlay` when known) and dialogue (scene
    clock) mixed with the scene's levels exactly as the export mixes them (engine scene_mix), saved as a
    stereo 48 kHz WAV at `out` (written beside it and renamed into place). CPU only."""
    voice_db, background_db = scene_parts.level(voice_db), scene_parts.level(background_db)
    if background is None and voice is None:
        raise ValueError("This scene has no sound to play yet.")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_name(f".{out.stem}.{uuid.uuid4().hex[:12]}.tmp.wav")
    try:
        samples = media_audio.scene_mix(background, voice, seconds, voice_db, background_db, effects=effects)
        media_audio.write_wav(staged, samples, RATE)
        os.replace(staged, out)
    except MediaError as error:
        raise ValueError(f"Could not mix the scene sound: {error}") from error
    finally:
        staged.unlink(missing_ok=True)
    return out


def preview_mix(sound: SceneSound, voice_db: float, background_db: float, out: str | Path) -> Path:
    """The prepared scene sound mixed with the scene's levels as the export will mix it (CPU)."""
    return mix_stems(sound.background, sound.voice, sound.seconds, voice_db, background_db, out,
                     effects=sound.overlay)


def clip_sound(clip: str | Path, background: Optional[str | Path], voice: Optional[str | Path],
               out: str | Path) -> Path:
    """A copy of a scene clip whose own mix is made again from the scene's stems, exactly as
    video_gen mixes a new clip (the handle-padded background lined up with the picture, the dialogue
    over it); the picture is copied without re-encoding. For a sound-only change (a new background
    or new effects) of a clip whose picture is reused. CPU only."""
    if background is None and voice is None:
        raise ValueError("This scene has no sound to mix.")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:12]
    mixed, staged = out.with_name(f".{out.stem}.{tag}.mix.wav"), out.with_name(f".{out.stem}.{tag}.tmp{out.suffix}")
    try:
        length = ffmpeg_utils.probe_duration(clip)
        if not length:
            raise ValueError("Could not read the scene clip.")
        ok, message = ffmpeg_utils.mix_voice_and_background(
            str(voice) if voice is not None else None, str(background) if background is not None else None,
            mixed, length, background_offset=HANDLE if background is not None else 0.0)
        if ok:
            ok, message = ffmpeg_utils.replace_audio(clip, mixed, staged)
        if not ok:
            raise ValueError(f"Could not mix the scene sound: {message}")
        os.replace(staged, out)
    finally:
        mixed.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)
    return out


def leveled_clip(clip: str | Path, background: Optional[str | Path], voice: Optional[str | Path], seconds: int,
                 voice_db: float, background_db: float, out: str | Path, *,
                 effects: Optional[str | Path] = None) -> Path:
    """A copy of a scene clip whose sound is its stems mixed with the scene's levels as the export
    mixes them (mix_stems); the picture is copied without re-encoding and the clip's own sound is
    left out. Saved at `out` (written beside it and renamed into place). CPU only."""
    voice_db, background_db = scene_parts.level(voice_db), scene_parts.level(background_db)
    if background is None and voice is None:
        raise ValueError("This scene has no sound to play yet.")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_name(f".{out.stem}.{uuid.uuid4().hex[:12]}.tmp{out.suffix}")
    try:
        samples = media_audio.scene_mix(background, voice, seconds, voice_db, background_db, effects=effects)
        media_video.remux(clip, staged, comment=media_video.AI_COMMENT, audio=(samples, RATE), keep_audio=False)
        os.replace(staged, out)
    except MediaError as error:
        raise ValueError(f"Could not mix the scene sound: {error}") from error
    finally:
        staged.unlink(missing_ok=True)
    return out
