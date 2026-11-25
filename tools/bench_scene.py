# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Measure whole Timeline scenes the way the queue makes them, and save what they made for an A/B.

Every scene runs through app.timeline.prepare_job, the runner of the queue's "Generate scene" job, as the queue's
worker thread runs it (queue_manager.in_worker() is true), with a fixed take (so the same seeds every time), on a
COPY of a project: a fresh state folder under --out holds the project (project.json only, unless --sound-preview),
its characters and their library voices, and the owner's user_prefs.json. The models are read from the
installation's models folder (--live); nothing is written into that installation (no downloads, no links, no
compiled kernels: Triton's cache is --triton-cache).

    venv\\Scripts\\python.exe tools\\bench_scene.py --out ..\\bench\\after --scenes 1,3 --cast rose=ada,elias=kit

--scenes are 0-based indexes, made one after another in one process as queued jobs are (the warm models of one job
wait for the next one; --release-between gives them back as an idle queue does). --code measures another source
tree with this script (for example the code before a change, from `git archive`); --verified carries a
verified_models.json between runs (an app that was started before) or, without it, every run starts as a fresh
app start would (with the persistent store of this version, an empty one).

For each scene --out gets: scene<N>.tsv (the queue's stage log: seconds since the job started, progress, step
text), scene<N>.json (the steps with their seconds, the stage totals and the identity scores), scene<N>.mp4 (the
clip) with its sound stems, and frames/scene<N>_edit<k>.png / _image<k>.png (every Qwen-Image-Edit and Z-Image
picture the scene made: the start frames). The identity scores are SFace similarities (app.qc.face_scores) of each
start frame and of every 10th frame of the clip to the cast's reference photos.

The GPU is shared: a run refuses to start while the live queue (config/queue.json, read only) has a running or
waiting job, or while more than --max-busy-gib of the GPU are in use (--skip-gpu-check skips both).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = "eba4d9fe361a4ee79834b935c9f7237d"  # "Night Studio - stage 5 test"
DEFAULT_TAKE = "b0" * 16
GIB = 1 << 30
_PREFIX = re.compile(r"^Scene \d+/\d+ · ")

# Stage of a step text (without its "Scene n/m · " prefix), first match wins.
STAGES = (
    ("model check", re.compile(r"^Checking the .* files|^First use")),
    ("scene check", re.compile(r"^Checking saved clip")),
    ("speech", re.compile(r"^Voice:")),
    ("sound", re.compile(r"^(Sound|Background audio|Sound effect|Laying the sound)")),
    ("start frame", re.compile(r"(starting frame|reference frame)")),
    ("wan rife", re.compile(r"^(Video|Part \d+/\d+): Frame interpolation")),
    ("wan", re.compile(r"^(Video|Part \d+/\d+): ")),
    ("lip sync", re.compile(r"^Lip sync")),
    ("rife", re.compile(r"(Interpolating to|Frame interpolation)")),
    ("assembly", re.compile(r"^(Assembling video|Done$)")),
    ("finish", re.compile(r"clips ready")),
)


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", type=Path, default=HERE,
                        help="the installation whose models, project, characters and voices are read (never written)")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--scenes", default="1", help="comma-separated 0-based scene indexes, made one after another")
    parser.add_argument("--cast", default="", help="characters to use instead, old=new slugs: rose=ada,elias=kit")
    parser.add_argument("--take", default=DEFAULT_TAKE, help="the take of every scene (it fixes the seeds)")
    parser.add_argument("--out", type=Path, required=True, help="folder for the results and the state copy")
    parser.add_argument("--code", type=Path, default=HERE, help="the source tree to measure (default: this one)")
    parser.add_argument("--verified", type=Path, help="a verified_models.json kept between runs")
    parser.add_argument("--triton-cache", type=Path, help="Triton's kernel cache (default: <out>/../triton-cache)")
    parser.add_argument("--sound-preview", action="store_true", help="copy the project's prepared sounds too")
    parser.add_argument("--release-between", action="store_true", help="give the warm models back between scenes")
    parser.add_argument("--app-started", action="store_true",
                        help="import the generation libraries first, as an app that started a while ago has "
                             "(engine_runtime.warm_imports, when the measured code has it); their time is printed")
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--max-busy-gib", type=float, default=3.0)
    return parser.parse_args(argv)


def _queue_busy(live: Path) -> int:
    """Running or waiting jobs of the live queue (its file is only read)."""
    try:
        data = json.loads((live / "config" / "queue.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return sum(job.get("status") in ("generowanie", "w kolejce") for job in data.get("jobs", []))


def _gpu_refusal(args) -> str | None:
    import torch

    busy_jobs = _queue_busy(args.live)
    if busy_jobs:
        return f"the live queue has {busy_jobs} running or waiting job(s)"
    free, total = torch.cuda.mem_get_info()
    if (total - free) / GIB > args.max_busy_gib:
        return f"{(total - free) / GIB:.1f} GiB of the GPU are in use (limit {args.max_busy_gib} GiB)"
    return None


def _cast_map(text: str) -> dict:
    pairs = [item.split("=", 1) for item in text.split(",") if item.strip()]
    return {old.strip(): new.strip() for old, new in pairs}


def _copy_state(args, state: Path) -> dict:
    """The project, its characters and their library voices, and the owner's preferences, copied into `state`.
    Returns the project as copied (with --cast applied)."""
    live = args.live
    source = live / "timeline_projects" / args.project
    project_dir = state / "timeline_projects" / args.project
    project_dir.mkdir(parents=True)
    project = json.loads((source / "project.json").read_text(encoding="utf-8"))
    if args.sound_preview and (source / "sound").is_dir():
        shutil.copytree(source / "sound", project_dir / "sound")
    mapping = _cast_map(args.cast)
    names = {}
    for old, new in mapping.items():
        record = json.loads((live / "characters" / new / "character.json").read_text(encoding="utf-8"))
        names[old] = (old[:1].upper() + old[1:], record["name"])
    if mapping:
        def swap(slug):
            return mapping.get(slug, slug)

        def rename(text):
            for old_name, new_name in names.values():
                text = re.sub(rf"(?<![\w@]){re.escape(old_name)}(?!\w)", new_name, text)
            return text

        project["characters"] = [swap(slug) for slug in project.get("characters", [])]
        for scene in project["scenes"]:
            scene["prompt"] = rename(scene["prompt"])
            if isinstance(scene.get("cast"), list):
                scene["cast"] = [swap(slug) for slug in scene["cast"]]
            for line in scene.get("lines", []):
                line["speaker"] = swap(line.get("speaker", ""))
                line["text"] = rename(line["text"])
    (project_dir / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=1), encoding="utf-8")
    slugs = {slug for scene in project["scenes"] for slug in scene.get("cast") or []}
    slugs |= set(project.get("characters", []))
    for slug in sorted(slugs):
        folder = live / "characters" / slug
        if not folder.is_dir():
            raise SystemExit(f"Character {slug} is not saved in {live / 'characters'} (use --cast {slug}=...).")
        shutil.copytree(folder, state / "characters" / slug)
        record = json.loads((folder / "character.json").read_text(encoding="utf-8"))
        voice = record.get("voice")
        if isinstance(voice, str) and voice and (live / "voices" / voice).is_dir() \
                and not (state / "voices" / voice).exists():
            shutil.copytree(live / "voices" / voice, state / "voices" / voice)
    (state / "config").mkdir(parents=True, exist_ok=True)
    prefs = live / "config" / "user_prefs.json"
    if prefs.is_file():
        shutil.copyfile(prefs, state / "config" / "user_prefs.json")
    if args.verified is not None and args.verified.is_file():
        shutil.copyfile(args.verified, state / "config" / "verified_models.json")
    return project


def _point_the_app(args, state: Path, aux_root: Path):
    """Import the app of --code with its state in `state` and its models read from the live installation."""
    sys.path.insert(0, str(args.code.resolve()))
    from app import config

    models = (args.live / "models").resolve()
    config.STATE_ROOT = state
    config.STATE_CONFIG_DIR = state / "config"
    config.OUTPUTS_DIR = state / "outputs"
    config.HISTORY_DIR = state / "history"
    config.MODELS_DIR = models
    config.CATALOG = config.load_model_catalog(str(models))
    from app import aux_models, downloader, engine_runtime, model_manager, timeline  # noqa: F401 - registers jobs

    if Path(timeline.DIR).resolve() != (state / "timeline_projects").resolve():
        raise SystemExit("The app did not take the copied state folder.")
    # Nothing may be written into the live installation: no downloads, no links, no compiled kernels.
    aux_models.ROOT = aux_root

    def refuse(*_args, **_kwargs):
        raise RuntimeError("bench_scene: a model is missing; install it in the Studio first.")

    model_manager.download_model = refuse
    downloader._preserve_invalid = refuse
    model_manager.os = _NoLinks(os)
    engine_runtime.TRITON_CACHE_DIR = args.triton_cache.resolve()
    return config


class _NoLinks:
    """os for model_manager without hard links (it would link an earlier file into the live models folder)."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    @staticmethod
    def link(*_args, **_kwargs):
        raise OSError("bench_scene: no links into the live models folder")


class _Pictures:
    """Every Qwen-Image-Edit and Z-Image picture the engine makes is also saved under `frames` (the start frames),
    named after the scene being made."""

    def __init__(self, frames: Path):
        from app import engine_runtime

        self.frames, self.scene, self.made = frames, 0, []
        engine_runtime.edit = self._keep("edit", engine_runtime.edit)
        engine_runtime.render = self._keep("image", engine_runtime.render)

    def start(self, scene: int) -> list:
        self.scene, self.made = scene, []
        return self.made

    def _keep(self, kind, function):
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            data = function(*args, **kwargs)
            number = sum(item["kind"] == kind for item in self.made) + 1
            path = self.frames / f"scene{self.scene}_{kind}{number}.png"
            path.write_bytes(data)
            self.made.append({"kind": kind, "path": str(path), "seconds": round(time.perf_counter() - started, 2)})
            return data
        return wrapper


def _stage_of(text: str) -> str:
    plain = _PREFIX.sub("", text)
    return next((name for name, pattern in STAGES if pattern.search(plain)), "other")


def _table(log: list, end: float) -> tuple[list, dict]:
    steps, totals = [], {}
    for index, (at, progress, text) in enumerate(log):
        until = log[index + 1][0] if index + 1 < len(log) else end
        stage = _stage_of(text)
        steps.append({"at": round(at, 1), "seconds": round(until - at, 1), "stage": stage, "text": text})
        totals[stage] = round(totals.get(stage, 0.0) + until - at, 1)
    return steps, totals


def _identity(project, index: int, clip: Path | None, pictures: list) -> dict:
    """SFace similarity of each start frame and of every 10th frame of the clip to the scene's characters."""
    import numpy as np
    from PIL import Image

    from app import characters, qc

    scene = project.scenes[index]
    members = [person for slug in (scene.cast or []) if (person := characters.get(slug))]
    if not members:
        return {}
    result = {"frames": {Path(item["path"]).name: qc.face_scores(item["path"], members) for item in pictures}}
    if clip is not None and clip.is_file():
        from engine.media.video import read_frames

        scores = {person["slug"]: [] for person in members}
        sample = clip.with_name(clip.stem + "_sample.png")
        try:
            for number, frame in enumerate(read_frames(clip)):
                if number % 10:
                    continue
                Image.fromarray(frame.to_ndarray(format="rgb24")).save(sample)
                for slug, score in qc.face_scores(str(sample), members).items():
                    scores[slug].append(score)
        finally:
            sample.unlink(missing_ok=True)
        result["clip"] = {slug: {"mean": round(float(np.mean([v for v in values if v is not None])), 4)
                                 if any(v is not None for v in values) else None,
                                 "frames": len(values), "without_face": values.count(None)}
                          for slug, values in scores.items()}
    return result


def _run_scene(args, out: Path, index: int, recorder: _Pictures) -> dict:
    from app import cancel, queue_manager, timeline

    project = timeline.load(args.project)
    params = {"project": project.id, "key": project.key, "seconds": project.scenes[index].seconds,
              "scene_index": index, "take": args.take}
    log, last = [], [None]
    started = time.time()
    tsv = out / f"scene{index + 1}.tsv"

    def cb(fraction: float, desc: str = "") -> None:  # the queue's job callback (queue_manager._run)
        cancel.check()
        if desc and desc != last[0]:
            at = time.time() - started
            log.append((at, min(max(fraction, 0.0), 1.0), desc))
            with tsv.open("a", encoding="utf-8") as stream:
                stream.write(f"{at:.1f}\t{fraction:.3f}\t{desc}\n")
            print(f"{at:7.1f}s  {desc}", flush=True)
        if desc:
            last[0] = desc

    pictures = recorder.start(index + 1)
    queue_manager._CONTEXT.active = True  # as in the queue's worker thread (queue_manager._process)
    try:
        ok, files, notes, error = timeline.prepare_job(params, cb)
    finally:
        queue_manager._CONTEXT.active = False
    seconds = time.time() - started
    steps, totals = _table(log, seconds)
    clip = None
    state = timeline.prepared(timeline.load(args.project))
    key = state["assets"][index] if index < len(state["assets"]) else None
    if ok and key:
        record = timeline.asset(timeline.load(args.project), key, verify=False)
        clip = out / f"scene{index + 1}.mp4"
        shutil.copyfile(record["path"], clip)
        for kind, stem in zip(("background", "voice"), record.get("audio_sources") or ()):
            if stem is not None:
                shutil.copyfile(stem, out / f"scene{index + 1}.{kind}.wav")
        if record.get("effects_source"):
            shutil.copyfile(record["effects_source"], out / f"scene{index + 1}.effects.wav")
    result = {"scene": index + 1, "ok": ok, "error": error, "notes": notes, "seconds": round(seconds, 1),
              "clip_seconds": project.scenes[index].seconds,
              "seconds_per_clip_second": round(seconds / project.scenes[index].seconds, 1),
              "totals": totals, "steps": steps, "pictures": pictures, "clip": str(clip) if clip else None}
    try:
        result["identity"] = _identity(project, index, clip, pictures)
    except Exception as error:  # noqa: BLE001 - the measurement stands without the face scores
        result["identity_error"] = f"{type(error).__name__}: {error}"
    (out / f"scene{index + 1}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def _print(result: dict) -> None:
    print(f"\nScene {result['scene']}: {'ok' if result['ok'] else 'FAILED: ' + result['error']}"
          f"  {result['seconds']} s = {result['seconds_per_clip_second']} s per clip second")
    for name, seconds in result["totals"].items():
        print(f"  {name:<12} {seconds:7.1f} s")
    if result.get("identity"):
        print("  identity:", json.dumps(result["identity"]))


def main(argv=None) -> int:
    args = _arguments(argv)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: every run starts from a fresh copy.")
    if args.triton_cache is None:
        args.triton_cache = out.parent / "triton-cache"
    live = args.live.resolve()
    for inside in (out, args.triton_cache.resolve()):
        if inside == live or live in inside.parents:
            raise SystemExit("--out and --triton-cache must be outside the live installation.")
    state, frames, aux_root = out / "state", out / "frames", out / "aux"
    frames.mkdir(parents=True)
    (aux_root / "models" / "mediapipe").mkdir(parents=True)
    landmarker = live / "models" / "mediapipe" / "face_landmarker.task"
    if landmarker.is_file():
        shutil.copyfile(landmarker, aux_root / "models" / "mediapipe" / landmarker.name)
    _copy_state(args, state)
    _point_the_app(args, state, aux_root)
    if not args.skip_gpu_check and (refusal := _gpu_refusal(args)):
        raise SystemExit(f"Not measuring: {refusal}.")
    if args.app_started:
        from app import engine_runtime

        if hasattr(engine_runtime, "warm_imports"):
            started = time.perf_counter()
            failed = engine_runtime.warm_imports()
            print(f"warm imports: {time.perf_counter() - started:.1f} s, failed: {failed}", flush=True)
    results, recorder = [], _Pictures(frames)
    for number, index in enumerate(int(item) for item in args.scenes.split(",")):
        if number and args.release_between:
            from app import engine_runtime

            engine_runtime.release()
        results.append(_run_scene(args, out, index, recorder))
        _print(results[-1])
    if args.verified is not None and (state / "config" / "verified_models.json").is_file():
        shutil.copyfile(state / "config" / "verified_models.json", args.verified)
    (out / "summary.json").write_text(json.dumps({"code": str(args.code.resolve()), "args": vars(args) | {
        key: str(value) for key, value in vars(args).items() if isinstance(value, Path)}, "scenes": results},
        indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
