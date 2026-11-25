# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""How the Timeline uses the shot contract and the shot check (docs/QUALITY_ARCHITECTURE.md, layers 3 and 4).

- Which scenes are checked: every scene of a series episode (its project belongs to a series, app/bible.py) and every
  scene with a ShotSpec; with the "qc" preference off nothing is checked and clips stay "unchecked" (never passed).
- The keyframe: the starting frame video_gen composes from the place's plate, the cast's photos and the props is
  checked before the video model runs (backdrop, palette, vision questions); a failing one is composed again.
- An anchored shot (`continuous`) starts from the previous clip's last frame only when that frame passes the same
  check; otherwise it starts from a fresh keyframe.
- The clip: checked right after it is made; a failing clip is made again with a new seed, up to
  shot_qc.MAX_RETRIES times, then it is "needs_review" and the episode cannot be delivered as complete.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from . import bible as bible_mod
from . import config, prefs, shot_qc

_CACHE: dict = {}
_TTL = 60.0


def bible_for(project) -> Optional[bible_mod.Bible]:
    """The bible of the series the project belongs to (remembered for a minute), or None."""
    now = time.monotonic()
    cached = _CACHE.get(project.id)
    if cached and now - cached[0] < _TTL:
        return cached[1]
    try:
        found = bible_mod.for_project(project.id)
    except Exception:  # noqa: BLE001 - a damaged series file must not stop a film outside it
        found = None
    _CACHE[project.id] = (now, found)
    return found


def settings() -> dict:
    """The shot check's settings (config "qc": vision, arbiter, retries)."""
    found = config.SETTINGS.get("qc") if isinstance(config.SETTINGS.get("qc"), dict) else {}
    return {"vision": bool(found.get("vision", True)), "arbiter": bool(found.get("arbiter", False)),
            "retries": int(found.get("retries", shot_qc.MAX_RETRIES)),
            "keyframe_tries": int(found.get("keyframe_tries", 3))}


def enabled(project, scene) -> bool:
    if not bool(prefs.load().get("qc", True)):
        return False
    return bool(scene.shot) or bible_for(project) is not None


def cast_specs(bible: Optional[bible_mod.Bible], members: Sequence[dict]) -> Tuple[bible_mod.CharacterSpec, ...]:
    """The bible's sheet of each scene member (a library character outside the bible gets one made from its own
    profile), with the member's reference photos."""
    from . import characters

    out = []
    for member in members:
        photos = []
        try:
            photos = [str(path) for path in characters.reference_photos(member)]
        except Exception:  # noqa: BLE001 - a character without readable photos is still checked by colour
            photos = []
        found = bible.character(member.get("slug", "")) if bible is not None else None
        if found is None:
            found = bible_mod.character_spec(member)
        out.append(bible_mod.CharacterSpec(**{**found.__dict__, "photo": photos[0] if photos else found.photo,
                                               "extra_photos": tuple(photos[1:3]) or found.extra_photos}))
    return tuple(out)


def _arbiter():
    if not settings()["arbiter"]:
        return None
    try:
        from . import shot_arbiter
    except ImportError:
        return None
    return shot_arbiter.ask if shot_arbiter.available() else None


def context(project, scene, members: Sequence[dict], *, reference=None, work: Optional[Path] = None,
            vision: Optional[bool] = None) -> shot_qc.Context:
    bible = bible_for(project)
    return shot_qc.Context(cast=cast_specs(bible, members), spec=scene.spec, bible=bible, reference=reference,
                           vision=settings()["vision"] if vision is None else vision, arbiter=_arbiter(), work=work)


def keyframe_checker(project, scene, members: Sequence[dict], work: Path) -> Callable[[str], str]:
    """video_gen's keyframe_check: "" when the starting frame passes, else the reasons."""
    def check(path: str) -> str:
        verdict = shot_qc.check_still(path, context(project, scene, members, work=work / "keyframe"))
        # only a found problem composes the frame again: a vision model that did not answer is the clip check's to
        # settle (the clip is then "unchecked" if it still cannot answer, never passed)
        return shot_qc.summary(verdict) if verdict.status == shot_qc.FAIL else ""
    return check


def anchor_problem(project, scene, members: Sequence[dict], frame: Path, work: Path) -> str:
    """Why the previous clip's last frame must not start this anchored shot ("" = it may)."""
    verdict = shot_qc.check_still(frame, context(project, scene, members, work=work / "anchor"))
    return "" if verdict.status == shot_qc.PASS else shot_qc.summary(verdict)


def check_clip(project, scene, members: Sequence[dict], clip: Path, *, keyframe=None, work: Path,
               attempt: int = 0) -> shot_qc.QcVerdict:
    verdict = shot_qc.check_clip(clip, context(project, scene, members, reference=keyframe, work=work / "qc"))
    verdict.attempt = attempt
    return verdict


def keep_sheets(project, number: int, attempt: int, work: Path) -> None:
    """Keep what the check looked at (the keyframe and the clip sheets of every try, a failed one too) in the
    project's qc/ folder, so the owner can see why a try was made again. Best effort: a copy that fails is skipped."""
    import shutil

    from . import timeline

    try:
        folder = timeline._inside(timeline._root(project.id), "qc")
        folder.mkdir(parents=True, exist_ok=True)
        for source in sorted(Path(work).rglob("qc_*.jpg")):
            target = folder / f"scene{number + 1:02d}_try{attempt + 1}_{source.parent.name}_{source.name}"
            shutil.copyfile(source, target)
    except (OSError, ValueError):
        pass


def record(verdict: shot_qc.QcVerdict, legacy: Optional[dict]) -> dict:
    """The clip's "qc" record: the verdict, with the face score and the reasons of the older face check kept."""
    out = verdict.record()
    if isinstance(legacy, dict):
        out["face"] = legacy.get("face")
        if legacy.get("reasons"):
            out["reasons"] = list(legacy["reasons"]) + out["reasons"]
    return out
