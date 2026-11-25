# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The video service: one mode, Wan 2.2 A14B with the LightX2V 4-step distillation at 480p.

Pipeline: dialogue (VoxCPM2), starting frames from references (Qwen-Image-Edit), Wan 2.2 with RIFE to
30 fps, background sound (EzAudio-XL), lip sync (MuseTalk 1.5) and assembly with metadata in PyAV.
The audio stages are optional. The finished film keeps the 480p frame Wan renders (owner decision,
2026-11-14: one fast mode).

Wan A14B renders 16 fps. One pass makes up to 6 seconds (97 frames); a longer video is made of parts as
even as whole seconds allow (7 s = 4 + 3, 12 s = 6 + 6), each starting from the last frame of the
previous one (image to video). A prompt can hold several shots ("Shot 2: ..."), each with its own
description and reference images (app/refs.py).

Frames are interpolated (RIFE) once to the rate the next step needs: a silent clip 16 -> 30 FPS; a
talking clip 16 -> 25 FPS, the lip-sync model's rate, then 25 -> 30 FPS as the last picture step.
"""
from __future__ import annotations

import math
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from engine.api import Engine

from . import cancel, characters, config, dialogue, engine_runtime, ffmpeg_utils, history, llm
from . import model_manager, prefs, qc, queue_manager, refs, scene_parts, scene_sound, sound, styles, tts, voices
from . import cast as cast_mod
from . import lipsync as lipsync_proc
from .vram import low_vram_hint

ProgressCb = Callable[[float, str], None]

NATIVE_FPS = 16          # Wan 2.2 A14B
# The longest part one Wan pass makes: 97 frames (Wan needs 4k + 1). At 480p that is 39,750 tokens (41,340
# with the overlap below), about half of the 720p pass (75,600) this engine ran on a 16 GB card.
MAX_PART_SECONDS = 6
# Cards under 15 GB keep the 81 frames measured within an 8 GiB limit (the longer pass needs about 1 GiB more).
SMALL_GPU_PART_SECONDS = 5
MAX_SECONDS = 30  # 5 czesci po 6 s; kolejna czesc startuje z bezstratnej klatki z modelu
FPS = 30                 # docelowy plik
LIPSYNC_FPS = lipsync_proc.FPS  # a talking picture is made at the lip-sync model's rate, then brought to FPS
ASPECTS = {"Landscape 16:9": 16 / 9, "Portrait 9:16": 9 / 16, "Square 1:1": 1.0,
           "Portrait 3:4": 3 / 4}
# The one video mode (owner decision 2026-11-14): Wan renders 480p in 4 steps with the LightX2V
# distillation LoRA - 2 steps on the high-noise expert, 2 on the low-noise one, no guidance (one model pass
# per step) and shift 5, the settings the LoRA was trained with.
VIDEO_PIXELS = 832 * 480
RESOLUTION = "480p"  # how the pages name the VIDEO_PIXELS frame
# Starting frames (a character's scene, a composed reference frame) are made at 1280x720 class: Wan
# downsizes them itself, and a face drawn larger stays clearer than one drawn at 480p.
START_FRAME_PIXELS = 1280 * 720
SAMPLING = {"steps": 4, "split": 2, "shift": 5.0, "lora": 1.0}
OVERLAP_FRAMES = 4  # zakladka czesci (klatki Wan, 0,25 s) - dlugosc 16s+1+4 = 4k+1 jak chce Wan

_OOM_HINTS = ("out of memory", "allocation on device", "oom")


@dataclass
class VideoOutput:
    ok: bool
    files: List[str] = field(default_factory=list)
    seed: int = -1
    error: str = ""
    notes: List[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    audio_stems: tuple[Path, Optional[Path]] | None = None
    # Timeline scenes with effects: the effects inside the background stem, alone on its clock (the export
    # levels the room tone without them and lays them over it afterwards)
    sound_overlay: Optional[Path] = None


def _resolve_seed(seed: int) -> int:
    return random.randint(0, 2 ** 31 - 1) if seed is None or seed < 0 else int(seed)


def _friendly(exc: Exception) -> str:
    msg = str(exc)
    if any(h in msg.lower() for h in _OOM_HINTS):
        return ("Not enough GPU memory. Close other apps using the GPU (such as games or OBS) "
                "and try again.\n" + low_vram_hint(15.0))
    return f"Video generation failed: {msg}"


def frame_size(aspect: float, pixels: int, step: int = 16) -> Tuple[int, int]:
    """Rozmiar klatki o danych proporcjach i liczbie pikseli (wielokrotnosci `step`)."""
    aspect = min(max(aspect, 0.5), 2.0)
    w = round(math.sqrt(pixels * aspect) / step) * step
    h = round(w / aspect / step) * step   # z proporcji, zeby 1:1 bylo kwadratem
    return int(w), int(h)


def sizes(ratio: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """(the frame Wan renders, the starting-frame size) for an aspect ratio. The finished film keeps
    the first; the second is where a character or a composed frame is drawn before Wan takes it."""
    return frame_size(ratio, VIDEO_PIXELS), frame_size(ratio, START_FRAME_PIXELS)


def _drawn_start_frame(client: Engine, text: str, width: int, height: int, path: Path, seed: int, style: str, lo: float,
                       label: str, progress_cb: Optional[ProgressCb], temp_files: List) -> Optional[str]:
    """The start frame of a shot with nothing to start from, drawn by Z-Image in the style (its image text and image
    LoRA, app/styles.py `draws_start`) as a PNG at `path`; None when it could not be drawn (the shot is then made by
    text to video). Cancelling the job is not caught."""
    if progress_cb:
        progress_cb(lo, f"{label}starting frame from the description...")

    def step(frac: float, desc: str) -> None:
        if progress_cb:
            progress_cb(lo, f"{label}starting frame: {desc}")

    temp_files.append(path)
    in_style = {} if style == styles.REALISTIC else {"style": style}
    try:
        # not kept warm: the video stage right after would drop Z-Image unused
        path.write_bytes(engine_runtime.render(prefs.styled(text, "image", style), width, height, seed,
                                               prefs.IMAGE_STEPS, step, runtime=client, keep=False, **in_style))
    except Exception as error:  # noqa: BLE001 - text to video still makes the shot
        print(f"[video] The starting frame could not be drawn ({error}); text to video makes the shot.", flush=True)
        return None
    return str(path)


def text_encoder_files(path) -> list:
    """The files Wan's text encoder at `path` reads (the sharded UMT5 with its index, or one file); [] when unknown."""
    try:
        from engine.models.wan22 import text

        return list(text.files(path))
    except (OSError, ValueError):
        return []


def _after_frame_files(talking: bool, style: str) -> list:
    """What a scene's picture stage reads first after its starting frame: InfiniteTalk's network (embeddings, adapter,
    blocks) and image encoder for a talking scene, else the image-to-video experts of the style. [] when unknown."""
    try:
        if talking:
            from engine.pipelines.talk import _use_order

            files = engine_runtime.talk_files()
            return [*_use_order(files.base, files.adapter), files.clip]
        experts, loras = engine_runtime.video_experts("i2v", SAMPLING["lora"], style)
        return [experts.high, *(path for path, _ in (loras or {}).get("high", ())),
                experts.low, *(path for path, _ in (loras or {}).get("low", ()))]
    except (FileNotFoundError, ValueError, OSError):
        return []


def _lipsync_to_output_fps(client: Engine, source: Path, output: Path, duration: float,
                          progress_cb: Optional[ProgressCb] = None) -> Tuple[bool, str]:
    """The talking picture from the lip-sync rate (25 FPS) to the output's 30 FPS, the last picture step:
    RIFE fills the frames between, and the clip keeps `duration` (the picture's length) exactly, so its
    sound stays in sync."""
    intermediate = output.with_name(output.stem + "_rife.mp4")
    try:
        engine_runtime.interpolate_video(client, source, intermediate, LIPSYNC_FPS, FPS, progress_cb)
        return ffmpeg_utils.convert_fps(intermediate, output, FPS, duration=duration)
    except Exception as exc:
        return False, f"Interpolation to {FPS} FPS failed: {exc}"
    finally:
        intermediate.unlink(missing_ok=True)


def _drift(frame: Path, members: Sequence[dict], style: str = styles.REALISTIC) -> str:
    """Why the frame the next part starts from no longer shows the shot's characters ("" = it does, or the
    check is off). Nothing is redrawn: the next part starts from this frame either way. style: the film's style
    (its face thresholds, qc.FACE_RULES). Only people are checked: an animal or creature has no face to compare."""
    members = [person for person in members if not characters.is_creature(person)]
    if not members or not qc.enabled():
        return ""
    in_style = {} if style == styles.REALISTIC else {"style": style}  # a Realistic check keeps the call it had
    verdict = qc.check(str(frame), list(members) if len(members) > 1 else members[0], require_all=False,
                       **in_style)
    return "" if verdict.ok else verdict.summary()


def part_seconds_limit() -> int:
    """The longest part one Wan pass makes on this card (MAX_PART_SECONDS, or SMALL_GPU_PART_SECONDS)."""
    return SMALL_GPU_PART_SECONDS if engine_runtime.small_gpu() else MAX_PART_SECONDS


def _segments(seconds: int, longest: int = MAX_PART_SECONDS) -> List[int]:
    """Parts of at most `longest` seconds, as even as whole seconds allow, longer ones first: 6 -> [6],
    7 -> [4, 3], 11 -> [6, 5], 12 -> [6, 6], 13 -> [5, 4, 4]. Never a short tail: a part pays its model
    loads and adds a seam whatever its length, so a 1-second tail (17 frames) cost nearly a whole part."""
    seconds = max(int(seconds), 1)
    count = -(-seconds // max(int(longest), 1))
    base, extra = divmod(seconds, count)
    return [base + 1] * extra + [base] * (count - extra)


def _shot_refs(shots: List[str], count: int) -> List[List[int]]:
    """Numery obrazow pomocniczych dla kazdego ujecia (do refs.MAX_REFS; wiecej niz miesci
    jeden przebieg modelu sklada refs.compose_refs). Bez numerow w prompcie wszystkie obrazy
    ida do pierwszego ujecia."""
    marked = [[n for n in refs.numbers(s) if 1 <= n <= count] for s in shots]
    if count and not any(marked):
        marked[0] = list(range(1, count + 1))
    return [used[:refs.MAX_REFS] for used in marked]


def _with_style(text: str, style: str) -> str:
    return f"{text.strip().rstrip('.')}, {style}" if style else text


def _wan_prompt(shot_visual: str, members: Sequence[dict], looks: Dict[int, str], any_refs: bool,
                style: str) -> str:
    """The prompt Wan gets for a part of a shot: the shot's picture text without picture numbers, with the
    characters by their labels and looks (several), or the look of the one character and of the people in
    reference pictures (`looks`), and the style. An animal or creature is never "the person": the one creature of a
    shot looks like this under its name when the shot names it, else as "the character"."""
    shot_char = members[0] if members else None
    person_looks = ([shot_char.get("look", "")] if shot_char else []) + list(looks.values())
    # neutral_look takes a person's skin, hair and age out of the text, so the photo decides them. An animal or
    # creature has no such photo-only features: its colours ("a golden glow", "teal fur") are kept (defect Q).
    neutral = refs.neutral_look if (looks or any(not characters.is_creature(one) for one in members)) else str
    if len(members) > 1:  # each character by its label ("the woman", "the man"...)
        positive = (neutral(refs.strip(cast_mod.label_names(shot_visual, members)))
                    + cast_mod.look_sentences(members)
                    + "".join(f". The person has {lk}" for lk in dict.fromkeys(looks.values()) if lk))
    elif characters.is_creature(shot_char):
        look = (shot_char.get("look") or "").strip().rstrip(".")
        name = shot_char["name"] if cast_mod.mentions(shot_visual, [shot_char]) else "The character"
        positive = (neutral(refs.strip(shot_visual)) + (f". {name} looks like this: {look}" if look else "")
                    + "".join(f". The person has {lk}" for lk in dict.fromkeys(looks.values()) if lk))
    elif any_refs or shot_char:
        positive = (refs.neutral_look(refs.strip(shot_visual))
                    + "".join(f". The person has {lk}" for lk in dict.fromkeys(person_looks) if lk))
    else:
        positive = refs.strip(shot_visual)
    return _with_style(positive, style)


def scene_wan_prompt(prompt: str, style: str = "", cast: Optional[List[dict]] = None, exact_cast: bool = False,
                     ref_images: Optional[List[str]] = None) -> Optional[str]:
    """The prompt Wan gets for a Timeline scene that generate(..., scene_dir=...) makes with these arguments
    (one shot; every part of it uses the same prompt), worked out without making anything, so a job can have
    the prompts of all its scenes encoded in one text-encoder load (engine_runtime.expect_video_prompts).
    None when the prompt depends on the looks of people in reference pictures, which are described by the
    language model only when the scene runs."""
    cast = [c for c in (cast or []) if c][:cast_mod.MAX_CAST]
    ref_images = [p for p in (ref_images or []) if p and Path(p).exists()]
    shots = [prompt]
    if _shot_refs(shots, len(ref_images))[0]:
        return None
    members = cast if exact_cast else cast_mod.shot_casts(shots, cast)[0]
    shot_visual = dialogue.parse(prompt, people=cast_mod.people(cast))[0]
    return _wan_prompt(shot_visual, members, {}, False, style)


def _prepared(audition: scene_sound.SceneSound, cues: Sequence[scene_parts.Cue],
              seconds: int) -> Optional[scene_sound.SceneSound]:
    """The audition read again from its folder, when its files are unchanged (checksums, it is about
    to become the clip's sound) and it is the sound of these lines for a scene of this length; else
    None."""
    current = scene_sound.load(audition.folder, audition.key, verify=True)
    if (current is None or current.seconds != seconds or current.texts != tuple(cue.text for cue in cues)
            or current.background is None or (cues and current.voice is None)):
        return None
    return current


def _sound_copies(package: scene_sound.SceneSound, output_dir: Path, stamp: str,
                  temp_files: List[Path]) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """Copies of a prepared scene sound's voice track, background, effects overlay and line files in
    the scene's work folder: (voice, background, overlay, lines). The work files are temporary (and the
    kept stems move to the clip), so the prepared files are only ever read."""
    copies: List[Optional[str]] = []
    for source, name in ((package.voice, "voice"), (package.background, "bg"), (package.overlay, "overlay")):
        if source is None:
            copies.append(None)
            continue
        target = output_dir / f"video_{stamp}_{name}.wav"
        temp_files.append(target)
        shutil.copyfile(source, target)
        copies.append(str(target))
    lines = []
    for number, source in enumerate(package.lines, 1):
        target = output_dir / f"video_{stamp}_speech{number}.wav"
        temp_files.append(target)
        shutil.copyfile(source, target)
        lines.append(str(target))
    return copies[0], copies[1], copies[2], lines


def generate(*args, **kwargs) -> VideoOutput:
    """Patrz _generate. Przy przerwaniu przez uzytkownika (cancel.Cancelled) sprzata pliki
    posrednie i przekazuje wyjatek dalej - do kolejki. A vision outage that a production script's
    starting_frame hook reports (llm.VisualReviewUnavailable) is passed on the same way.
    style: the Wan suffix of the picture's style (app/styles.py, a Style's `video`; "" = none). It also decides
    how start frames and compositions are worded, and a style other than Realistic lets a scene keep its picture
    when lip sync finds no usable face (the speech still plays; the note says so).
    An animal or creature of the cast (app/characters.py) has no face to check and no mouth MuseTalk can move: its
    lines play over the picture with a note and never fail a scene (InfiniteTalk still makes a scene of an animated
    film where it speaks alone), and the Video page (no scene_dir, a Realistic style) refuses it."""
    temp_files: List[Path] = []
    try:
        return _generate(*args, _temp=temp_files, **kwargs)
    except (cancel.Cancelled, llm.VisualReviewUnavailable):
        for tmp in temp_files:
            Path(tmp).unlink(missing_ok=True)
        raise
    except Exception as exc:
        for tmp in temp_files:
            Path(tmp).unlink(missing_ok=True)
        return VideoOutput(ok=False, error=_friendly(exc))
    finally:
        engine_runtime.after_start_frame(())  # a later picture outside this clip reads nothing ahead for it


def _generate(prompt: str, seconds: int, aspect: str, seed: int = -1,
              image_path: Optional[str] = None, ref_images: Optional[List[str]] = None,
              sound_on: bool = True, style: str = "",
              progress_cb: Optional[ProgressCb] = None,
              _temp: Optional[List[Path]] = None,
              character: Optional[dict] = None,
              cast: Optional[List[dict]] = None,
              scene_dir: Optional[Path] = None,
              starting_frame: Optional[Callable[[], str]] = None,
              exact_cast: bool = False,
              cues: Optional[Sequence[scene_parts.Cue]] = None, background_text: str = "",
              effects: Sequence[scene_parts.Effect] = (),
              prepared_sound: Optional[scene_sound.SceneSound] = None,
              wan_text: Optional[str] = None,
              keyframe_check: Optional[Callable[[str], str]] = None,
              keyframe_tries: int = 3) -> VideoOutput:
    """Kwestie postaci w cudzyslowie w prompcie (np. says „Cześć!”) staja sie dialogiem:
    glos dobrany do plci mowiacego i ruch ust kazdej mowiacej postaci. Obrazy pomocnicze
    wskazane w prompcie numerem ([1], [2]...) trafiaja do klatki startowej ujecia.
    character - a character (app/characters.py): every shot starts from its scene composed by
    Qwen-Image-Edit from its primary reference photo (reference images add things to it), then
    Wan turns that frame into video. A later part of the same shot starts from the previous part's
    last frame; the SFace check reports a face that drifted from the character's photos.
    cast - several characters (app/cast.py, the first is the main one): the names in each shot
    decide who is in it; a shot with two or more of them starts from a frame composed from all
    their photos.
    The video is made in the one mode (SAMPLING at VIDEO_PIXELS, see the top of this file).
    sound_on=False - film bez dzwieku. Brakujace modele pobieraja sie same.
    Timeline scenes (scene_dir) can bring their sound in parts (app/scene_sound.py): cues - the
    spoken lines with speakers and entry times (the prompt is then the picture text alone); the
    sound comes from prepared_sound (an audition, copied, never moved) or is prepared before the
    picture, so a line that runs past the scene fails before Wan starts. background_text - the
    background sound description (else derived from the picture); effects - timed sound effects
    laid over the background. Both come with the scene's cues.
    wan_text - the exact prompt the video model gets (a shot compiled from its ShotSpec, app/shot_spec.py), instead
    of the one made from the picture text and the characters' looks; the style words are still added.
    keyframe_check - called with the starting frame of a Timeline scene before the video model runs: "" when it
    passes, else why not; a frame that fails is composed again with a new seed, up to keyframe_tries times. The
    frame the clip started from is info["keyframe"] (kept in scene_dir for the clip's own check)."""
    structured = cues is not None
    cues = tuple(cues or ())
    effects = tuple(effects or ())
    background_text = (background_text or "").strip()
    if scene_dir is None and (structured or prepared_sound is not None or background_text or effects):
        raise ValueError("Dialogue lines, background sound and effects need a Timeline scene.")
    if not structured and (background_text or effects):
        raise ValueError("A scene's background description and effects come with its cues.")
    extra_texts = [text for text in (*(cue.text for cue in cues), background_text,
                                     *(effect.text for effect in effects)) if text]
    if llm.blocked(prompt, style, *extra_texts):
        return VideoOutput(ok=False, error=llm.BLOCK_MSG)
    if not str(prompt or "").strip():
        return VideoOutput(ok=False, error="Enter a description for the video.")
    seed = _resolve_seed(seed)
    plan = SAMPLING
    art = styles.for_video_text(style)  # the picture's style: compositions, face thresholds, lip sync
    # a Realistic picture's compositions keep the calls they always had
    in_style = {} if art.realistic else {"style": art.id}
    ref_images = [p for p in (ref_images or []) if p and Path(p).exists()]
    # A timeline scene has its own clock and stays private until the timeline is exported.
    shots = [prompt] if scene_dir is not None else refs.split_shots(prompt)
    # A Timeline scene may last tenths of a second (scene_parts.length): its picture is made in whole seconds and cut
    # to `exact`, and its sound is made for `exact`. Everything else lasts whole seconds, as it always did.
    exact = (scene_parts.length(min(max(float(seconds), 1.0), MAX_SECONDS))
             if scene_dir is not None and scene_parts.is_length(seconds) else None)
    seconds = min(max(math.ceil(float(seconds) - 1e-6), 1, 2 * len(shots) if len(shots) > 1 else 1), MAX_SECONDS)
    exact = seconds if exact is None or exact > seconds else exact
    notes: List[str] = []
    cast = [c for c in (cast or ([character] if character else [])) if c][:cast_mod.MAX_CAST]
    character = cast[0] if cast else None
    if scene_dir is None and not characters.creature_style_ok(art.id):  # the Video page is always Realistic
        refused = cast_mod.creature_error(cast)
        if refused:
            return VideoOutput(ok=False, error=refused)
    # animals and creatures of the cast (app/characters.py): no face to check, no mouth MuseTalk can move
    creatures = {person["slug"]: person for person in cast if characters.is_creature(person)}
    people = cast_mod.people(cast)
    shot_cast = [cast] if exact_cast and scene_dir is not None else cast_mod.shot_casts(shots, cast)
    _, sound_prompt, lines = dialogue.parse(refs.plain(prompt), voices.tokens(), people)
    if structured or not sound_on:
        lines = []  # structured: the dialogue is in cues; the picture text has no spoken lines
    background = sound_on
    talk = bool(sound_on and (cues if structured else lines))
    use_lipsync = talk
    spoken = []
    bed_prompt = scene_sound.background_prompt(sound_prompt, background_text)
    scene_speakers: List[scene_sound.Speaker] = []
    package: Optional[scene_sound.SceneSound] = None  # the scene's prepared sound (structured mode)
    if structured and sound_on:
        try:
            scene_speakers = scene_sound.resolve_speakers(cues, cast)
        except ValueError as error:
            return VideoOutput(ok=False, error=str(error))
        if prepared_sound is not None:
            package = _prepared(prepared_sound, cues, exact)
            if package is None:
                notes.append("The sound preview no longer matched this scene, so the sound was prepared again.")
            else:
                scene_speakers = list(package.speakers)
        spoken = [speaker.voice for speaker in scene_speakers]
    elif lines:
        # Named voice > character voice > default. Ignore picked characters absent from the shots.
        shown = {person["slug"] for members in shot_cast for person in members}
        lead = character if character and character["slug"] in shown else None
        spoken = dialogue.voices_for(lines, voices.catalog(), voices.for_character(lead))

    if starting_frame is not None:
        image_path = starting_frame()

    def dcb(frac: float, desc: str) -> None:
        if progress_cb:
            progress_cb(0.01 * frac, desc)

    needed_models = []
    if ref_images or cast:  # characters come into a frame through their reference photos
        needed_models.append(engine_runtime.QWEN_EDIT_ID)
    if art.has_loras:  # an animated style's LoRAs (app/styles.py)
        needed_models.append(styles.STYLE_MODEL_ID)
    if structured:
        if talk and any(speaker.character not in creatures for speaker in scene_speakers):
            needed_models.append(lipsync_proc.MODEL_ID)  # a person speaks: MuseTalk moves human mouths only
        if sound_on and package is None:  # an audition already has its speech and background
            needed_models.extend(((tts.MODEL_ID,) if talk else ()) + (sound.MODEL_ID,))
    else:
        if lines:
            needed_models.extend((tts.MODEL_ID, lipsync_proc.MODEL_ID))
        if sound_on:
            needed_models.append(sound.MODEL_ID)
    err = model_manager.ensure(needed_models, dcb)
    if err:
        return VideoOutput(ok=False, error=err)
    timings: List[Tuple[float, float]] = []
    # Shares of the progress bar (and the queue's ETA): voice, video, background, lip sync; the
    # finishing (frame interpolation, assembly) starts at p_end
    if structured:  # the scene's sound (speech, background and effects) is made before the picture
        p_start = 0.09 if sound_on and package is None else 0.02
    else:
        p_start = 0.05 if lines else 0.02
    p_video, p_bg, p_ls = (0.60, 0.63, 0.65) if use_lipsync else (0.88, 0.94, 0.0)
    p_end = 0.95

    client = engine_runtime.engine()
    if (ref_images or cast) and not image_path:
        engine_runtime.read_ahead_start_frame(art.id)  # the starting frame's models, read while the sound is made

    output_dir = scene_dir if scene_dir is not None else config.OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    temp_files: List[Path] = _temp if _temp is not None else []
    overlay_file: Optional[str] = None  # the effects inside the background, alone (see VideoOutput)

    def fail(error: str) -> VideoOutput:
        for tmp in temp_files:
            Path(tmp).unlink(missing_ok=True)
        return VideoOutput(ok=False, error=error)

    # --- 1. Dialog (VoxCPM2) - najpierw, bo moze wydluzyc wideo ---
    voice_file: Optional[str] = None
    bg_file: Optional[str] = None
    speech_files: List[str] = []  # every line's own file (Timeline scenes): lip sync per speaker
    who = [line.who for line in lines]
    if structured and sound_on:
        # Lines on the scene clock and the background with its effects, all before the picture: a
        # line that runs past the scene end is cut there (the scene keeps the length chosen for it).
        if package is None:
            def scb(frac: float, desc: str) -> None:
                if progress_cb:
                    progress_cb(0.01 + (p_start - 0.01) * frac, desc)

            if progress_cb:
                progress_cb(0.01, "Sound: preparing...")
            try:
                with tempfile.TemporaryDirectory(dir=output_dir, prefix=f"video_{stamp}_sound_",
                                                 ignore_cleanup_errors=True) as scratch:
                    package = scene_sound.prepare(cues, scene_speakers, bed_prompt, effects, exact, seed,
                                                  Path(scratch) / "sound", scb, client=client, voice=talk)
                    voice_file, bg_file, overlay_file, speech_files = _sound_copies(
                        package, output_dir, stamp, temp_files)
            except ValueError as error:
                return fail(str(error))
        else:
            voice_file, bg_file, overlay_file, speech_files = _sound_copies(package, output_dir, stamp, temp_files)
        if talk and voice_file is None:
            return fail("The scene's dialogue track is missing. Prepare the sound again.")
        timings = list(package.timings)
        who = [speaker.who for speaker in package.speakers]
        spoken = [speaker.voice for speaker in package.speakers]
    elif lines:
        if progress_cb:
            progress_cb(0.01, "Voice: preparing...")
        client.trim()  # the card for VoxCPM2; models kept warm wait in RAM
        line_files = [output_dir / f"video_{stamp}_line{i + 1}.wav"
                      for i in range(len(lines))]
        voice_path = output_dir / f"video_{stamp}_voice.wav"
        temp_files += [*line_files, voice_path]

        def tcb(frac: float, desc: str) -> None:
            if progress_cb:
                progress_cb(0.01 + (p_start - 0.01) * frac, desc)

        ok, msg = tts.synthesize_many(
            [(line.text, v, str(f)) for line, v, f in zip(lines, spoken, line_files)], tcb)
        if ok:
            length, timings = dialogue.assemble([str(f) for f in line_files], voice_path)
            voice_file = str(voice_path)
            # dluzszy dialog -> dluzsze wideo zamiast zatrzymanej ostatniej klatki
            needed = math.ceil(length + 0.3)
            if needed > seconds:
                if scene_dir is not None:
                    return fail(f"The dialogue needs {needed} s. Extend this scene or shorten its dialogue.")
                notes.append(f"Video extended from {seconds} to {min(needed, MAX_SECONDS)} s "
                             "to fit the dialogue.")
                seconds = exact = min(needed, MAX_SECONDS)
        else:
            if scene_dir is not None:
                return fail(msg)
            notes.append(msg)
            spoken = []  # no speech in the video, so no voices in the history record
            use_lipsync = False

    # --- Plan ujec: dlugosc, obrazy pomocnicze, czesci do 6 s ---
    shot_secs = refs.split_seconds(seconds, len(shots))
    if len(shot_secs) < len(shots):
        notes.append(f"The video is too short for {len(shots)} shots; using {len(shot_secs)}.")
        shots = shots[:len(shot_secs)]
        shot_cast = shot_cast[:len(shot_secs)]
    # characters seen in the video (QC, lip sync, history): every shot's own set
    appearing = list({c["slug"]: c for members in shot_cast for c in members}.values())
    shot_refs = _shot_refs(shots, len(ref_images))
    longest = part_seconds_limit()
    plan_parts = [(i, s) for i, secs in enumerate(shot_secs) for s in _segments(secs, longest)]
    picture_seconds = exact if exact != seconds else sum(s for _, s in plan_parts)  # cut to the scene's tenths
    # A talking picture is made at the lip-sync rate (RIFE 16 -> 25 per part) and synchronized there without
    # a resample; it reaches 30 FPS at the end (RIFE 25 -> 30). A silent one goes 16 -> 30 directly.
    talking = bool(use_lipsync and voice_file)
    said_by = ([speaker.character for speaker in package.speakers] if package is not None
               else [line.character for line in lines])
    # In an animated style a line of a library voice without a character is a narrator's voice-over (Contsy Stories):
    # no mouth is moved for it, so a narrated scene skips InfiniteTalk and MuseTalk (and the lip-sync frame rate).
    animated = art.family != styles.REALISTIC
    voice_over = [i for i, slug in enumerate(said_by) if animated and not slug]
    # An animated character who speaks alone in a one-shot scene talks through InfiniteTalk (the whole picture with
    # its mouth movement, at the lip-sync rate) instead of Wan 2.2 and MuseTalk, which was trained on real faces.
    use_talk = (talking and animated and len(shots) == 1 and len(set(who)) == 1 and not voice_over
                and engine_runtime.talk_installed())
    talked = False
    # The lines MuseTalk may lip-sync: the people's. An animal's or creature's lines play over the picture, so a
    # picture where only animals or creatures speak is made like a silent one (unless InfiniteTalk makes it).
    lip_lines = [i for i, slug in enumerate(said_by) if slug not in creatures and i not in voice_over]
    picture_fps = LIPSYNC_FPS if talking and (lip_lines or use_talk) else FPS
    first_kind = "i2v" if (image_path or shot_refs[0] or character) else "t2v"
    # An animated style that draws its start frames (the 2D style, whose AniSora experts are image to video only):
    # nothing to start from means a frame drawn by Z-Image in the style, then image to video
    text_start = first_kind == "t2v" and art.draws_start and engine_runtime.zimage_files() is not None
    if text_start:
        first_kind = "i2v"
    kinds = [first_kind] + ["i2v"] * (len(plan_parts) - 1)
    err = model_manager.ensure([f"wan22-a14b-{k}" for k in dict.fromkeys(kinds)], dcb)
    if err:
        return fail(err)
    installed_kinds = set(kinds)

    if image_path and scene_dir is None:  # proporcje jak na zdjeciu - bez obcinania kadru
        with Image.open(image_path) as im:
            ratio = im.width / im.height
    else:
        ratio = ASPECTS.get(aspect, 16 / 9)
    (width, height), (start_w, start_h) = sizes(ratio)

    if wan_text and not cast:
        # A compiled shot of a place or a thing: the style's words about characters ("expressive animated characters")
        # would draw strangers into the empty frame (defect A); the rest of the style stays.
        style = ", ".join(part for part in style.split(",") if "character" not in part.lower()).strip(", ")

    def with_style(text: str) -> str:
        return _with_style(text, style)

    parts: List[str] = []
    overlaps: List[float] = []  # zakladka (s) kazdej czesci z poprzednia - join_segments
    handoffs: List[float] = []  # chwila (s, zegar czesci) klatki, od ktorej startuje nastepna czesc
    overlap = 0.0
    start_image = image_path
    looks: Dict[int, str] = {}  # wyglad osob ze zdjec pomocniczych (llm.describe_person)
    done_shot = -1
    drifts: List[str] = []  # faces that moved away from the characters at a part's handoff frame

    # The files the picture stage reads right after a starting frame, read ahead while the frame's steps run: the
    # talking network for a scene InfiniteTalk makes, else the image-to-video experts (engine_runtime.after_start_frame)
    if (text_start or ((ref_images or cast) and not image_path)) and queue_manager.in_worker():
        engine_runtime.after_start_frame(_after_frame_files(use_talk, art.id))
    start_frame_note = None  # why a shot that should start from a drawn frame was made from its text
    keyframe: Optional[str] = None  # the starting frame the shot check looked at

    # --- 2. Wideo: kolejne ujecia i ich czesci ---
    for k, (shot_i, seg_seconds) in enumerate(plan_parts):
        lo = p_start + (p_video - p_start) * k / len(plan_parts)
        hi = p_start + (p_video - p_start) * (k + 1) / len(plan_parts)
        label = f"Part {k + 1}/{len(plan_parts)}: " if len(plan_parts) > 1 else "Video: "
        shot_visual = dialogue.parse(shots[shot_i], people=people)[0]
        members = shot_cast[shot_i] if shot_i < len(shot_cast) else []
        shot_char = members[0] if members else None  # the character of a one-character shot
        if shot_i != done_shot:  # poczatek ujecia: klatka startowa z obrazow pomocniczych
            done_shot = shot_i
            used = shot_refs[shot_i]
            char_base = False
            composed = False
            base_image = start_image  # what the shot starts from before any composition
            # A Timeline scene with a shot check (keyframe_check) gets its starting frame checked before the
            # video model sees it: a frame that fails is composed again with a new seed (layer 3 of
            # docs/QUALITY_ARCHITECTURE.md); after the last try the clip's own check decides.
            tries = keyframe_tries if keyframe_check is not None and shot_i == 0 else 1
            for frame_try in range(tries):
                start_image, char_base, composed = base_image, False, False
                frame_seed = seed + 7919 * frame_try

                def kcb(frac: float, desc: str, lo=lo, label=label) -> None:
                    if progress_cb:
                        progress_cb(lo, f"{label}reference frame: {desc}")

                for n in used:  # wyglad osoby ze zdjecia (raz na zdjecie, zapamietany)
                    if n not in looks:
                        looks[n] = llm.describe_person(ref_images[n - 1], kcb)
                if len(members) > 1 and not (shot_i == 0 and image_path):
                    key = output_dir / f"video_{stamp}_cast{shot_i + 1}.png"
                    temp_files.append(key)

                    def ccb(frac: float, desc: str, lo=lo, label=label) -> None:
                        if progress_cb:
                            progress_cb(lo, f"{label}starting frame - {desc}")

                    ok, msg = cast_mod.compose_frame(
                        client, members, with_style(shot_visual), start_w, start_h, key,
                        frame_seed + 1000 + shot_i, ccb,
                        extra_refs=[(n, ref_images[n - 1]) for n in used],
                        looks={n: looks[n] for n in used}, **in_style)
                    if not ok:
                        return fail(f"Shot {shot_i + 1}: {msg}")
                    start_image, char_base, composed = msg, True, True
                elif shot_char and not (shot_i == 0 and image_path):
                    ck = output_dir / f"video_{stamp}_char{shot_i + 1}.png"
                    temp_files.append(ck)
                    if progress_cb:
                        progress_cb(lo, f"{label}starting frame - character {shot_char['name']}...")

                    def fcb(frac: float, desc: str, lo=lo, label=label, name=shot_char["name"]) -> None:
                        if progress_cb:
                            progress_cb(lo, f"{label}starting frame - {name}: {desc}")

                    # composed once from the character's photo; the face is not measured here
                    ok, msg = characters.render(client, shot_char, with_style(shot_visual), start_w, start_h, ck,
                                                frame_seed + 2000 + shot_i, fcb, **in_style)
                    if ok:
                        start_image, char_base = msg, True
                    else:
                        return fail(f"Shot {shot_i + 1}: {msg}")
                if used and not composed:
                    key = output_dir / f"video_{stamp}_key{shot_i + 1}.png"
                    temp_files.append(key)
                    base_lead = refs.CREATURE_CHARACTER if characters.is_creature(shot_char) else refs.CHARACTER
                    ok, msg = refs.compose_refs(
                        client, start_image, [(n, ref_images[n - 1]) for n in used],
                        with_style(shot_visual), start_w, start_h, key, frame_seed + 1000 + shot_i,
                        continuing=shot_i > 0 and not char_base, progress_cb=kcb,
                        looks={n: looks[n] for n in used},
                        base_lead=base_lead if char_base else "",
                        **({"places_only": True} if keyframe_check is not None and not members
                           and not any(looks.get(n) for n in used) else {}), **in_style)
                    if ok:
                        start_image = msg
                    else:
                        return fail(f"Shot {shot_i + 1}: {msg}")
                if tries == 1 or not start_image:
                    break
                problem = keyframe_check(start_image)
                keyframe = start_image
                if not problem:
                    break
                notes.append(f"Starting frame {frame_try + 1} failed the shot check: {problem}")
                if start_image == base_image:  # nothing was composed (a given picture): a new seed changes nothing
                    break
            kept_frame = output_dir / f"video_{stamp}_keyframe.png"
            if keyframe and Path(keyframe) != kept_frame:  # kept beside the clip for its own check
                shutil.copyfile(keyframe, kept_frame)  # (the composed files are temporary)
                keyframe = str(kept_frame)
        if start_image is None and text_start:
            drawn = _drawn_start_frame(client, refs.strip(shot_visual), start_w, start_h,
                                       output_dir / f"video_{stamp}_start{shot_i + 1}.png", seed + 3000 + shot_i, art.id,
                                       lo, label, progress_cb, temp_files)
            if drawn:
                start_image = drawn
            else:
                start_frame_note = "The starting frame could not be drawn, so the clip was made from the text alone."
                notes.append(start_frame_note)
        kind = "i2v" if start_image else "t2v"
        if kind not in installed_kinds:
            err = model_manager.ensure([f"wan22-a14b-{kind}"], dcb)
            if err:
                return fail(err)
            installed_kinds.add(kind)
        wan_prompt = (with_style(refs.strip(wan_text)) if wan_text
                      else _wan_prompt(shot_visual, members, looks, any(shot_refs), style))
        if use_talk and k == 0 and start_image:
            talk_path = output_dir / f"video_{stamp}_talk.mp4"
            temp_files.append(talk_path)

            def talk_progress(fraction, message):
                if progress_cb:
                    progress_cb(p_start + (p_end - p_start) * fraction, f"Talking picture: {message}")

            try:
                engine_runtime.talk_part(client, start_image, voice_file, talk_path, prompt=wan_prompt, width=width,
                                         height=height, seconds=picture_seconds, seed=seed,
                                         progress_cb=talk_progress, then=engine_runtime.next_stage_files())
            except Exception as error:  # noqa: BLE001 - the scene is made the usual way instead
                notes.append(f"The talking picture could not be made with InfiniteTalk ({_friendly(error)[:200]}); "
                             "the scene was made with Wan 2.2" + (" and MuseTalk." if lip_lines else "."))
            else:
                talked = True
                parts, overlaps, handoffs = [str(talk_path)], [0.0], [picture_seconds]
                notes.append("The character's mouth movement was made with InfiniteTalk.")
                break
        # ta sama scena z postacia w nastepnej czesci - start z zakladki (patrz nizej)
        next_same = (bool(members) and k + 1 < len(plan_parts)
                     and plan_parts[k + 1][0] == shot_i)
        n_frames = NATIVE_FPS * seg_seconds + 1 + (OVERLAP_FRAMES if overlap else 0)
        last_index = n_frames - 1 - (OVERLAP_FRAMES if next_same else 0)

        def native_progress(fraction, message):
            if progress_cb:
                progress_cb(lo + (hi - lo) * fraction, label + message)

        if progress_cb:
            progress_cb(lo, label + "generating in the engine...")
        part = output_dir / f"video_{stamp}_part{k + 1}.mp4"
        temp_files.append(part)
        try:
            handoff = engine_runtime.video_part(
                client, kind, start_image, part, NATIVE_FPS, picture_fps, prompt=wan_prompt,
                width=width, height=height, frame_count=n_frames,
                last_index=last_index, seed=seed + k, steps=plan["steps"],
                shift=plan["shift"], split=plan["split"], lora_strength=plan["lora"], progress_cb=native_progress,
                then=engine_runtime.next_stage_files() if k + 1 == len(plan_parts) else (), **in_style)
        except Exception as e:  # noqa: BLE001
            return fail(_friendly(e))

        parts.append(str(part))
        overlaps.append(overlap)
        handoffs.append(last_index / NATIVE_FPS)
        overlap = 0.0
        if k + 1 < len(plan_parts):
            frame = output_dir / f"video_{stamp}_last{k + 1}.png"
            temp_files.append(frame)
            # a character and the same shot go on: the next part starts from a frame shortly
            # before the end (that stretch is cross-faded when the parts are joined); the frame
            # is used as it is, and a face that drifted from the character is reported
            frame.write_bytes(handoff)
            start_image = str(frame)
            if next_same:
                overlap = OVERLAP_FRAMES / NATIVE_FPS
                drift = _drift(frame, members, art.id)
                if drift:
                    drifts.append(f"part {k + 2}: {drift}")

    if drifts:
        notes.append("⚠ The face moved away from the character between parts (" + "; ".join(drifts)
                     + "). Generate the video again if it shows.")

    # The parts joined on the picture's clock (picture_fps), cut to exactly picture_seconds
    raw_video = output_dir / f"video_{stamp}_raw.mp4"
    temp_files.append(raw_video)
    if len(parts) == 1:
        ok, msg = ffmpeg_utils.convert_fps(parts[0], raw_video, picture_fps, duration=float(picture_seconds))
        if not ok:
            return fail(f"Could not save the {picture_fps} FPS video: {msg}")
    else:
        ok, msg = ffmpeg_utils.join_segments(parts, raw_video, picture_fps, overlaps, duration=picture_seconds,
                                             handoffs=handoffs)
        if not ok:
            return fail(f"Could not assemble the video: {msg}")

    # --- 3. Dzwiek tla z opisu sceny ---
    if background and not structured:  # structured scenes made theirs before the picture
        if progress_cb:
            progress_cb(p_video, "Background audio...")
        bg_path = output_dir / f"video_{stamp}_bg.wav"
        temp_files.append(bg_path)

        def bcb(frac: float, desc: str) -> None:
            if progress_cb:
                progress_cb(p_video + (p_bg - p_video) * frac, "Background audio: " + desc)

        length = ffmpeg_utils.probe_duration(raw_video) or seconds
        ok, msg = sound.generate(client, sound_prompt, "", length, seed, bg_path, bcb)
        if ok:
            bg_file = str(bg_path)
        else:
            if scene_dir is not None:
                return fail(msg)
            notes.append(msg)

    # --- 4. Ruch ust (MuseTalk) - kazda mowiaca postac w kadrze, na obrazie 25 FPS ---
    video_track = raw_video
    lipsync_note = ""  # why lip sync was skipped (a stylized scene's face it could not use, a creature's lines)
    lipsync_out = output_dir / f"video_{stamp}_lipsync.mp4"
    temp_files.append(lipsync_out)
    if talking and not talked and not art.musetalk:
        # MuseTalk would paint a real mouth on a drawn face (styles.Style.musetalk): the speech plays over the picture
        lipsync_note = ("Lip sync left out: MuseTalk cannot animate a drawn mouth. The dialogue plays over the "
                        "picture without mouth movement.")
        notes.append(lipsync_note)
    elif talking and not talked:
        if any(slug in creatures for slug in said_by):  # an animal or creature speaks: its lines never fail the scene
            lipsync_note = lipsync_proc.creature_note([creatures[slug]["name"] for slug in dict.fromkeys(said_by)
                                                       if slug in creatures])
            notes.append(lipsync_note)
        if lip_lines:  # the people's lines (all of them when no animal or creature speaks)
            if progress_cb:
                progress_cb(p_ls, "Lip sync: preparing...")
            client.trim()  # the card for the lip-sync model; Wan's experts wait in RAM when kept warm

            def lcb(frac: float, desc: str) -> None:
                if progress_cb:
                    progress_cb(p_ls + (p_end - p_ls) * frac, desc)

            own = {"lines": [speech_files[i] for i in lip_lines]} \
                if speech_files and len(speech_files) == len(timings) else {}
            ok, msg, ls_notes = lipsync_proc.run_dialogue(
                str(raw_video), voice_file, [timings[i] for i in lip_lines], [who[i] for i in lip_lines],
                str(lipsync_out), lcb, cast=appearing, **own)
            notes += ls_notes
            if ok:
                video_track = lipsync_out
            elif scene_dir is None:
                notes.append(f"Lip sync skipped ({msg}). Dialogue added as an audio track.")
            elif art.family == styles.REALISTIC:  # a real-looking face it could not use: a new take is needed
                return fail(f"Lip sync failed: {msg}")
            else:
                # A drawn or animated face that MuseTalk (trained on real faces) cannot use keeps its picture: the
                # speech still plays over it, and the job log says so.
                skipped = f"Lip sync skipped ({msg}). The dialogue plays over the picture without mouth movement."
                notes.append(skipped)
                lipsync_note = f"{lipsync_note} {skipped}".strip()
    if picture_fps != FPS:
        # The last picture step: RIFE 25 -> 30 FPS, the length kept to the frame so the sound stays in sync.
        fps_path = output_dir / f"video_{stamp}_{FPS}fps.mp4"
        temp_files.append(fps_path)
        if progress_cb:
            progress_cb(p_end, f"Interpolating to {FPS} FPS (RIFE)...")
        def rcb(frac: float, desc: str) -> None:
            if progress_cb:
                progress_cb(p_end + (0.96 - p_end) * frac, desc)

        ok, msg = _lipsync_to_output_fps(client, Path(video_track), fps_path, duration=float(picture_seconds),
                                         progress_cb=rcb)
        if not ok:
            return fail(msg)
        video_track = fps_path

    if structured:
        said = list(package.texts) if package is not None else []
        speaking = [speaker.character for speaker in package.speakers] if package is not None else []
    else:
        said, speaking = [line.text for line in lines], [line.character for line in lines]
    info = {"mode": "video_i2v" if (image_path or shot_refs[0]) else "video_t2v",
            "positive": prompt, "seconds": exact, "fps": FPS, "steps": plan["steps"],
            "seed": seed, "shots": len(shots), "ref_images": len(ref_images),
            "dialogue": said,
            "voices": [getattr(voices.get(s), "name", s) for s in spoken],
            "dialogue_timings": [
                {"text": text, "character": person, "voice": voice, "start": start, "end": end}
                for text, person, voice, (start, end) in zip(said, speaking, spoken, timings)
            ],
            "background": bool(bg_file), "background_model": sound.MODEL_ID if bg_file else None}
    if package is not None:
        info.update(sound_key=package.key, background_prompt=package.background_prompt)
    if drifts:
        info["drift"] = drifts
    if lipsync_note:
        info["lipsync_skipped"] = lipsync_note
    if start_frame_note:
        info["start_frame_note"] = start_frame_note  # a Timeline job reports it (the clip was made from text)
    if keyframe:
        info["keyframe"] = keyframe  # the checked starting frame: the reference of the clip's own check
        info["keyframe_notes"] = [note for note in notes if note.startswith("Starting frame ")]
    if talked:
        info["lipsync"] = "infinitetalk"
    if not art.realistic:
        info["style"] = art.id
    if character:
        info["character"] = character["name"]
    if [c["slug"] for c in appearing] != [c["slug"] for c in cast[:1]]:
        info["characters"] = [c["name"] for c in appearing]
    if len(cast) > 1:
        info["cast"] = [c["slug"] for c in appearing]
    return _finish_picture(Path(video_track), bg_file, voice_file, info, notes, (width, height), output_dir,
                           stamp, scene_dir is not None, progress_cb, temp_files, fail, overlay=overlay_file)


def redub(picture: Path, package: scene_sound.SceneSound, cast: Sequence[dict], seconds: int | float, seed: int,
          size: Tuple[int, int], scene_dir: Path, progress_cb: Optional[ProgressCb] = None) -> VideoOutput:
    """A Timeline scene's talking clip that keeps the picture of an earlier clip (`picture`, at FPS, of Wan's frame
    `size`) and speaks the scene's new sound `package` (an AI edit of its dialogue, app/timeline.py): the picture
    brought to the lip-sync rate, MuseTalk moving every speaking person's mouth to the new lines (an animal's or
    creature's lines play over the picture), RIFE back to FPS, then assembled as generate assembles a talking clip.
    VideoOutput like generate's; cancelling is passed on."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    temp_files: List[Path] = []

    def fail(error: str) -> VideoOutput:
        for tmp in temp_files:
            Path(tmp).unlink(missing_ok=True)
        return VideoOutput(ok=False, error=error)

    def report(low: float, high: float):
        return lambda fraction, message: progress_cb(low + (high - low) * min(max(fraction, 0.0), 1.0), message) \
            if progress_cb else None

    try:
        if package.voice is None:
            return fail("The scene's new dialogue has no voice track. Prepare the sound again.")
        err = model_manager.ensure([lipsync_proc.MODEL_ID], report(0.0, 0.02))
        if err:
            return fail(err)
        voice_file, bg_file, overlay_file, speech_files = _sound_copies(package, scene_dir, stamp, temp_files)
        exact = float(seconds)
        steady = scene_dir / f"video_{stamp}_lipsync_in.mp4"
        temp_files.append(steady)
        if progress_cb:
            progress_cb(0.03, "Lip sync: reading the picture...")
        ok, msg = ffmpeg_utils.convert_fps(picture, steady, LIPSYNC_FPS, duration=exact)
        if not ok:
            return fail(f"Could not read the picture at {LIPSYNC_FPS} FPS: {msg}")
        creatures = {person["slug"] for person in cast if characters.is_creature(person)}
        who = [speaker.who for speaker in package.speakers]
        lip_lines = [i for i, speaker in enumerate(package.speakers) if speaker.character not in creatures]
        notes: List[str] = []
        synced = scene_dir / f"video_{stamp}_lipsync.mp4"
        temp_files.append(synced)
        if lip_lines:
            engine_runtime.engine().trim()  # the card for the lip-sync model
            own = {"lines": [speech_files[i] for i in lip_lines]} \
                if speech_files and len(speech_files) == len(package.timings) else {}
            ok, msg, ls_notes = lipsync_proc.run_dialogue(
                str(steady), voice_file, [package.timings[i] for i in lip_lines], [who[i] for i in lip_lines],
                str(synced), report(0.05, 0.8), cast=list(cast), **own)
            notes += ls_notes
            if not ok:
                return fail(f"Lip sync failed: {msg}")
        else:  # only animals or creatures speak: their lines play over the picture
            shutil.copyfile(steady, synced)
            notes.append(lipsync_proc.creature_note([person["name"] for person in cast
                                                     if person["slug"] in creatures]))
        final = scene_dir / f"video_{stamp}_{FPS}fps.mp4"
        temp_files.append(final)
        if progress_cb:
            progress_cb(0.8, f"Interpolating to {FPS} FPS (RIFE)...")
        ok, msg = _lipsync_to_output_fps(engine_runtime.engine(), synced, final, duration=exact,
                                         progress_cb=report(0.8, 0.95))
        if not ok:
            return fail(msg)
        said = list(package.texts)
        spoken = [speaker.voice for speaker in package.speakers]
        info = {"mode": "video_lipsync", "positive": "", "seconds": seconds, "fps": FPS, "steps": 0, "seed": seed,
                "shots": 1, "ref_images": 0, "dialogue": said,
                "voices": [getattr(voices.get(s), "name", s) for s in spoken],
                "dialogue_timings": [{"text": text, "character": speaker.character, "voice": speaker.voice,
                                      "start": start, "end": end}
                                     for text, speaker, (start, end) in zip(said, package.speakers, package.timings)],
                "background": bool(bg_file), "background_model": sound.MODEL_ID if bg_file else None,
                "sound_key": package.key, "background_prompt": package.background_prompt}
        return _finish_picture(final, bg_file, voice_file, info, notes, size, scene_dir, stamp, True, progress_cb,
                               temp_files, fail, overlay=overlay_file)
    except cancel.Cancelled:
        for tmp in temp_files:
            Path(tmp).unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 - the scene's job reports it
        return fail(_friendly(exc))


def _finish_picture(video_track: Path, bg_file: Optional[str], voice_file: Optional[str], info: dict,
                    notes: List[str], size: Tuple[int, int], output_dir: Path, stamp: str, timeline_scene: bool,
                    progress_cb, temp_files: List[Path], fail, *, overlay: Optional[str] = None) -> VideoOutput:
    """The film from its finished picture and its background and dialogue tracks: the assembly, the quality
    record and the history entry. `size` is Wan's frame."""
    width, height = size

    # --- 5. Montaz (PyAV) ---
    # Pod obraz (lip sync oddaje same klatki) trafia narracja 48 kHz zmiksowana z tlem
    # (tlo przycisza sie pod glosem).
    if progress_cb:
        progress_cb(0.96, "Assembling video...")
    final_path = output_dir / f"video_{stamp}.mp4"
    temp_files.append(final_path)
    mixed: Optional[str] = None
    if voice_file or bg_file:
        mixed_path = output_dir / f"video_{stamp}_audio.wav"
        temp_files.append(mixed_path)
        # dlugosc dzwieku = dlugosc obrazu (narracja jest juz w niej zmieszczona - wyjatek:
        # narracja dluzsza niz MAX_SECONDS, wtedy montaz przedluzy ostatnia klatke)
        v_len = ffmpeg_utils.probe_duration(video_track) or float(info["seconds"])
        if timeline_scene:
            # A Timeline clip is exactly its scene, and its background (the stem kept for the
            # export) starts TIMELINE_AUDIO_HANDLE before the picture: skip that part so sound
            # effects play on their frames.
            okm, mix_error = ffmpeg_utils.mix_voice_and_background(
                voice_file, bg_file, mixed_path, v_len, background_offset=ffmpeg_utils.TIMELINE_AUDIO_HANDLE)
        else:
            a_len = (ffmpeg_utils.probe_duration(voice_file) or 0.0) + 0.3 if voice_file else 0.0
            okm, mix_error = ffmpeg_utils.mix_voice_and_background(voice_file, bg_file, mixed_path,
                                                                   max(v_len, a_len))
        mixed = str(mixed_path) if okm else None
        if not okm:
            if timeline_scene:
                return fail(f"Could not mix the scene audio: {mix_error}")
            notes.append("Could not mix the audio; saving a silent video.")
    # Encode the picture for ordinary players: an intermediate file may be lossless H.264 (the High 4:4:4
    # Predictive profile, which many players cannot open).
    ok, msg = ffmpeg_utils.combine_video_audio(video_track, mixed, final_path, compress=True)
    if not ok:
        if timeline_scene:
            return fail(f"Could not assemble the scene: {msg}")
        notes.append(f"Video assembly failed ({msg}). Saved the video without final assembly.")
        ffmpeg_utils.tag_video_metadata(video_track)
        final_path = Path(video_track)

    actual_fps = ffmpeg_utils.video_fps(final_path)
    if actual_fps is None or abs(actual_fps - FPS) > 0.001:
        return fail(f"The output does not have the required {FPS} FPS. Check the video details in the queue.")

    # The clip's quality record, kept with its history entry and its Timeline clip: the face drift found at
    # the handoff frames between its parts. The finished clip itself is not read here, so it is "unchecked", never a
    # pass: a Timeline clip gets its own check (app/shot_qc.py) right after it is made.
    drift = info.pop("drift", [])
    verdict = qc.Verdict(ok=False, reasons=[f"face drift at {where}" for where in drift],
                         status="fail" if drift else "unchecked")
    audio_stems = sound_overlay = None
    kept = {final_path}
    if timeline_scene and bg_file:
        audio_stems = Path(bg_file), Path(voice_file) if voice_file else None
        kept.update(path for path in audio_stems if path is not None)
        if overlay and Path(overlay).is_file():
            sound_overlay = Path(overlay)
            kept.add(sound_overlay)
    # Timeline also retains its unmixed sound for the final soundtrack.
    for tmp in temp_files:
        tmp = Path(tmp)
        if tmp not in kept and tmp.exists():
            tmp.unlink()

    if progress_cb:
        progress_cb(1.0, "Done")

    info.update(engine="Wan 2.2 A14B", width=width, height=height, files=[str(final_path)], qc=verdict.record())
    if not timeline_scene:
        history.record(info)
    return VideoOutput(ok=True, files=[str(final_path)], seed=info["seed"], notes=notes, info=info,
                       audio_stems=audio_stems, sound_overlay=sound_overlay)
