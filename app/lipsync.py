# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Lip sync of dialogue with MuseTalk 1.5, and the face tracking that tells it who speaks where."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import soundfile as sf

from . import cancel, characters, dialogue, engine_runtime, ffmpeg_utils, model_manager, qc

MODEL_ID = "musetalk-15"
FPS = 25           # engine.pipelines.musetalk.FPS (that module imports torch; a test keeps them equal)
MIN_FACE_PX = 48   # a smaller face shows no readable mouth movement
CROP_PX = 512      # a face crop smaller than this is enlarged, so the face detectors find a small face
# The face tracks (who is where) are found on every FACE_STRIDE-th frame, about 8 per second: a face's place, size
# and head crop barely change in 0.12 s, and MuseTalk finds the face again in every frame for the mouth. Every frame
# took about 5 s of CPU for a 6 s clip (2026-11-15).
FACE_STRIDE = 3

ProgressCb = Callable[[float, str], None]


def find_faces(video: str, workdir: str) -> dict:
    """Find face tracks with bounding boxes, sizes and a head crop each. When tracking itself fails (a model
    that could not be downloaded or read, a video that could not be decoded) there are no tracks and "error"
    says why, so that failure is not reported as a framing problem."""
    from engine.models.faces import find_faces as detect
    from .aux_models import ensure_landmarker

    try:
        return detect(video, workdir, landmarker_path=ensure_landmarker(), yunet_path=qc.yunet_path(),
                      cancel=cancel.TOKEN, stride=FACE_STRIDE)
    except Exception as exc:  # noqa: BLE001 - cancel.Cancelled is a BaseException and passes
        return {"tracks": [], "error": f"{type(exc).__name__}: {exc}"}


def run_dialogue(video: str, dialog_wav: str, timings: List[Tuple[float, float]],
                 who: List[str], output: str,
                 progress_cb: Optional[ProgressCb] = None,
                 cast: Optional[List[dict]] = None,
                 lines: Optional[List[str]] = None) -> Tuple[bool, str, List[str]]:
    """Mouth movement for every speaking person in the frame, with no limit on the number of people.

    who[i] is the person who speaks line i ("female#0", "male#0", "female#char:ola"...), timings[i] its time. A
    character from cast gets its own face by face recognition; other people get the faces nobody
    recognised, largest first (see assign_faces). One face: the whole frame. Several: every speaker in one pass
    over the whole frame, each face hearing only its own lines (silence elsewhere keeps the mouth closed); a
    face that pass could not use is tried again on a crop around it, enlarged, which is laid back in place.
    lines: each line's own file (Timeline scenes): a face then hears only its own lines, also where lines
    overlap (dialogue.speaker_track).
    video: the picture at FPS (25), as app/video_gen makes a talking clip. It is used as it is: the one entry
    point of lip sync never resamples (an fps=25 resample of a 30 FPS clip dropped frames RIFE had just made),
    and the output has the same frames at the same rate.
    The lines of an animal or creature character are never lip-synced (assign_faces): when only they speak, the
    output is the picture as it is, with a note.
    Returns (ok, path or message, notes).
    """
    notes: List[str] = []
    rate = ffmpeg_utils.video_fps(video)
    if rate is None or abs(rate - FPS) > ffmpeg_utils.FPS_TOLERANCE:
        return False, f"Lip sync needs a {FPS} FPS picture (got {rate or 'an unreadable one'}).", notes
    silent = _creature_speakers(who, cast)
    if who and len(silent) == len(set(who)):  # only animals or creatures speak: no mouth MuseTalk can move
        shutil.copyfile(video, output)
        return True, output, [creature_note(list(silent.values()))]
    work = tempfile.mkdtemp(prefix="aistudio_dialog_")
    try:
        if progress_cb:
            progress_cb(0.0, "Lip sync: preparing the speech part")
        v25 = Path(video)
        total = round((ffmpeg_utils.probe_duration(v25) or 0.0) * FPS)
        # Only the part with speech (+0.4 s): in another shot of the clip the person may turn away or run.
        f0 = max(0, int((min(a for a, _ in timings) - 0.4) * FPS))
        f1 = min(total, int((max(b for _, b in timings) + 0.4) * FPS) + 1)
        if f0 < 5 and f1 > total - 5:
            f0, f1 = 0, total
        clip, wav = v25, Path(dialog_wav)
        if (f0, f1) != (0, total):
            clip, wav = Path(work) / "clip.mp4", Path(work) / "clip.wav"
            ffmpeg_utils.transform(v25, clip, f"trim=start_frame={f0}:end_frame={f1},setpts=PTS-STARTPTS")
            audio, rate = sf.read(str(dialog_wav), dtype="float32", always_2d=False)
            sf.write(str(wav), audio[int(f0 / FPS * rate):int(f1 / FPS * rate)], rate)
            timings = [(a - f0 / FPS, b - f0 / FPS) for a, b in timings]
        out_clip = Path(work) / "clip_ls.mp4" if clip != v25 else Path(output)
        ok, msg = _dialogue_clip(clip, str(wav), timings, who, str(out_clip), (f1 - f0) / FPS,
                                 work, progress_cb, notes, cast or [], lines)
        if not ok or clip == v25:
            return ok, msg, notes
        # the processed frames [f0, f1) back into the whole 25 FPS video
        ok, msg = ffmpeg_utils.splice_frames(v25, out_clip, f0, f1, Path(output))
        return ok, (output if ok else msg), notes
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _gender(profile: dict) -> str:
    """A character profile's stored gender in the speakers' terms ("female" / "male")."""
    return "male" if profile.get("gender") == "man" else "female"


def _creature_speakers(who: Sequence[str], cast: Optional[Sequence[dict]]) -> Dict[str, str]:
    """Speaker -> name of each animal or creature character among the speakers ("female#char:<slug>"), in the order
    they first talk: MuseTalk moves human mouths only, so their lines are never lip-synced."""
    creatures = {person["slug"]: person["name"] for person in cast or [] if characters.is_creature(person)}
    found = {}
    for speaker in dict.fromkeys(who):
        slug = speaker.split("#char:", 1)[1] if "#char:" in speaker else ""
        if slug in creatures:
            found[speaker] = creatures[slug]
    return found


def creature_note(names: Sequence[str]) -> str:
    """The note for the lines of animal or creature characters, which MuseTalk never lip-syncs."""
    names = list(names)
    who = " and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names)
    return f"Lip sync left out for {who}: MuseTalk only moves human mouths, so the lines play over the picture."


def assign_faces(who: List[str], tracks: List[dict], cast: List[dict], work: str,
                 notes: List[str]) -> Dict[str, dict]:
    """Speaker -> face track. Every face is matched against the photos of every character in the video (face
    recognition on the track's head crop, so a silent look-alike cannot take a speaker's face); a recognised
    face is that character, with the gender stored in the character's profile. A character
    ("female#char:<slug>") gets its own recognised face. Everyone else, and a character that is not recognised
    with certainty, gets the first free face nobody was recognised in (largest first), else a free recognised
    face of the speaker's gender, else any free face. No gender is guessed from a face: with several unnamed
    speakers and no recognised face, faces go by size in the order the speakers first talk.
    An animal or creature character never takes a face (MuseTalk moves human mouths only): its lines are left
    out, with a note."""
    assign: Dict[str, dict] = {}
    left_out = _creature_speakers(who, cast)
    if left_out:
        notes.append(creature_note(list(left_out.values())))
    persons = [p for p in dict.fromkeys(who) if p not in left_out]
    cast = [person for person in cast or [] if not characters.is_creature(person)]
    named = {p.split("#char:", 1)[1]: p for p in persons if "#char:" in p}
    crops = [str(Path(work) / t["crop"]) if t.get("crop") else "" for t in tracks]
    known: List[Tuple[dict, str]] = []  # (face track, gender of the character recognised in it)
    if cast and all(crops):
        try:
            found = qc.match_crops(crops, list(cast))
        except Exception:  # noqa: BLE001 - no face models: size only
            found = {}
        genders = {person["slug"]: _gender(person) for person in cast}
        for slug, i in found.items():
            known.append((tracks[i], genders.get(slug, "")))
            if slug in named:
                assign[named[slug]] = tracks[i]
    for person in persons:
        if person in assign:
            continue
        free = [t for t in tracks if all(t is not taken for taken in assign.values())]
        unknown = [t for t in free if all(t is not face for face, _ in known)]
        same = [t for t in free if any(t is face and gender == person.split("#")[0] for face, gender in known)]
        pick = unknown or same or free
        if pick:
            assign[person] = pick[0]
        else:
            notes.append("Lip sync: there are more speakers than visible faces; "
                         "some lines will be heard off-screen.")
    return assign


def _span(progress_cb: Optional[ProgressCb], low: float, high: float) -> Optional[ProgressCb]:
    if progress_cb is None:
        return None
    return lambda fraction, message: progress_cb(low + (high - low) * fraction, message)


def _dialogue_clip(v25: Path, dialog_wav: str, timings: List[Tuple[float, float]],
                   who: List[str], output: str, length: float, work: str,
                   progress_cb: Optional[ProgressCb],
                   notes: List[str], cast: Optional[List[dict]] = None,
                   lines: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Mouth movement on a 25 fps fragment (see run_dialogue). Returns (ok, path or error)."""
    if progress_cb:
        progress_cb(0.0, "Lip sync: finding faces")
    faces = find_faces(str(v25), work)
    if faces.get("error"):
        return False, f"Face tracking failed: {faces['error'][:200]}"
    tracks = sorted((t for t in faces.get("tracks", []) if t["size"] >= MIN_FACE_PX),
                    key=lambda t: -t["size"] * t["presence"])
    if not tracks:
        return False, ("No face in the frame is large enough for lip sync; "
                       "use a close-up or medium shot")
    if progress_cb:
        progress_cb(0.0, "Lip sync: recognizing the speakers")
    assign = assign_faces(who, tracks, cast or [], work, notes)
    voices = {}
    for k, person in enumerate(assign):
        voices[person] = Path(work) / f"a{k}.wav"
        dialogue.speaker_track(dialog_wav, timings, [i for i, p in enumerate(who) if p == person],
                               voices[person], length, lines)
    base, retry, retry_cb, one_pass = v25, list(assign), progress_cb, ()
    if len(tracks) == 1 and len(assign) == 1:
        ok, msg = run(str(v25), str(voices[retry[0]]), output, progress_cb)
        if ok:
            return True, output
        # The face detectors can miss a small face in a wide shot of the whole picture; a crop around
        # it, enlarged, shows it to them at a size they find.
        notes.append("Lip sync on the whole frame failed; retried on the face only.")
    elif assign:
        together = Path(work) / "together.mp4"
        speakers = [(str(voices[person]), track["bbox"]) for person, track in assign.items()]
        ok, result = run_tracks(str(v25), speakers, str(together), _span(progress_cb, 0.0, 0.7),
                                num_faces=len(faces["tracks"]))
        if ok:
            from engine.pipelines.musetalk import MIN_SPEECH_COVERAGE

            base, retry_cb, one_pass = together, _span(progress_cb, 0.7, 1.0), tuple(result)
            retry = [person for person, report in zip(assign, result) if report.coverage < MIN_SPEECH_COVERAGE]
            if not retry:
                shutil.move(str(together), output)
                return True, output
            notes.append("Lip sync on the whole frame missed a face; retried on that face only.")
        else:
            notes.append(f"Lip sync of all speakers at once failed ({result[:120]}); retried face by face.")

    W, H = faces["width"], faces["height"]
    layers, reasons, persons = [], [], list(assign)
    for n, person in enumerate(retry):
        k = persons.index(person)
        x0, y0, x1, y1 = assign[person]["bbox"]
        side = max(x1 - x0, y1 - y0) * 1.8   # the face with room for head movement
        w, h = min(int(side) // 2 * 2, W // 2 * 2), min(int(side) // 2 * 2, H // 2 * 2)
        x = int(min(max((x0 + x1 - w) / 2, 0), W - w))
        y = int(min(max((y0 + y1 - h) / 2, 0), H - h))
        crop, out = Path(work) / f"crop{k}.mp4", Path(work) / f"ls{k}.mp4"
        # the result is scaled back to the crop's size when it is laid over the video
        up = max(1.0, CROP_PX / max(w, h))
        scale = f",scale={round(w * up) // 2 * 2}:{round(h * up) // 2 * 2}:flags=lanczos" if up > 1 else ""
        ffmpeg_utils.transform(v25, crop, f"crop={w}:{h}:{x}:{y}{scale}")

        def cb(f, d, n=n, k=k):
            if retry_cb:
                retry_cb((n + f) / len(retry), f"Lip sync: person {k + 1}/{len(persons)} - {d}")

        ok, msg = run(str(crop), str(voices[person]), str(out), cb)
        if ok:
            layers.append((str(out), x, y, w, h))
        else:
            reasons.append(msg)
            notes.append(f"Lip sync for person {k + 1} skipped ({msg[:120]}).")
    if not layers:
        # The one pass is the result only when it reached the threshold for at least one speaker.
        if base != v25 and len(retry) < len(assign):
            shutil.move(str(base), output)
            return True, output
        # Keep the first real cause: a face turned away reads differently from running out of memory.
        reason = next((" ".join(str(r).split()) for r in reasons if str(r).strip()), "")
        if not reason and one_pass:  # no crop said why: what the one pass covered
            from engine.pipelines.musetalk import MIN_SPEECH_COVERAGE, Report, coverage_problem

            reason = coverage_problem(Report(tracks=one_pass), MIN_SPEECH_COVERAGE) or ""
        return False, "No face could be lip-synced" + (f" ({reason[:160]})" if reason else "")
    ok, msg = ffmpeg_utils.overlay_layers(base, layers, output)
    return ok, (output if ok else msg)


def run(video: str, audio: str, output: str, progress_cb: Optional[ProgressCb] = None) -> Tuple[bool, str]:
    """Lip-sync the face of the picture to the narration. Returns (ok, message); the result has no sound
    (the clip's mix adds it). Fails when the face could be used for too few of the speech frames."""
    try:
        _run_native(video, audio, output, progress_cb)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, output


def run_tracks(video: str, speakers: Sequence[Tuple[str, Sequence[float]]], output: str,
               progress_cb: Optional[ProgressCb] = None, *, num_faces: int = 0) -> Tuple[bool, object]:
    """Every speaker of the picture in one pass: `speakers` are (that person's audio file, the box of their
    face track from find_faces); `num_faces` is how many faces the shot shows (at least one per speaker), so a
    silent face never stands in for a speaker's. Returns (True, one musetalk.TrackReport per speaker) or
    (False, message); the result has no sound (the clip's mix adds it)."""
    try:
        return True, _run_native_tracks(video, speakers, output, progress_cb, max(num_faces, len(speakers)))
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _files() -> Tuple[Path, Path, Path]:
    """MuseTalk's UNet, Whisper tiny and the folder of the SD VAE."""
    unet = model_manager.model_path(MODEL_ID, "musetalk")
    whisper = model_manager.model_path(MODEL_ID, "lipsync_whisper")
    vae = model_manager.model_path(MODEL_ID, "lipsync_vae", ".safetensors")
    if unet is None or whisper is None or vae is None:
        raise FileNotFoundError("The MuseTalk model bundle is incomplete.")
    return unet, whisper, vae.parent


def _frames(video: str):
    import numpy as np
    from engine.media import video as media_video

    pictures = []
    for frame in media_video.read_frames(video):
        cancel.check()
        pictures.append(frame.to_ndarray(format="rgb24"))
    if not pictures:
        raise ValueError("There are no video frames to synchronize.")
    return np.stack(pictures)


def _write(output: str, frames) -> None:
    from engine.media import video as media_video

    height, width = frames.shape[1:3]
    with media_video.Writer(output, FPS, width, height, crf=18) as writer:
        for frame in frames:
            cancel.check()
            writer.write(frame)


def _run_native(video, narration, output, progress_cb) -> None:
    from engine.media import audio
    from engine.models.musetalk.faces import MuseTalkFaceDetector
    from .aux_models import ensure_landmarker

    unet, whisper, vae_dir = _files()
    frames = _frames(video)
    samples, _ = audio.read(narration, rate=16000, channels=1)
    with MuseTalkFaceDetector(ensure_landmarker(), qc.yunet_path()) as detector:
        result = engine_runtime.engine().lipsync(unet, whisper, vae_dir, frames, samples[0], detector,
                                                  low_vram=engine_runtime.small_gpu(), progress_cb=progress_cb)
    _write(output, result)


def _run_native_tracks(video, speakers, output, progress_cb, num_faces: int) -> list:
    from engine.media import audio
    from engine.models.musetalk.faces import MuseTalkFaceDetector, follow_tracks
    from engine.pipelines.musetalk import FaceTrack
    from .aux_models import ensure_landmarker

    unet, whisper, vae_dir = _files()
    frames = _frames(video)
    voices = [audio.read(path, rate=16000, channels=1)[0][0] for path, _box in speakers]
    if progress_cb:
        progress_cb(0.0, "Lip sync: finding the faces")
    with MuseTalkFaceDetector(ensure_landmarker(), qc.yunet_path(), num_faces=num_faces) as detector:
        found = []
        for frame in frames:
            cancel.check()
            found.append(detector.faces(frame))
    faces = follow_tracks(found, [tuple(float(v) for v in box) for _path, box in speakers])
    reports = []
    result = engine_runtime.engine().lipsync_tracks(
        unet, whisper, vae_dir, frames, [FaceTrack(voice, track) for voice, track in zip(voices, faces)],
        low_vram=engine_runtime.small_gpu(), progress_cb=progress_cb, report_cb=reports.append)
    _write(output, result)
    return list(reports[0].tracks)
