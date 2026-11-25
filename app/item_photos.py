# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Reference photos of one place or object for the Items creator (the GPU part of app/items.py), made like a
character's (app/char_photos.py).

The first photo is drawn by Z-Image-Turbo from the description in the draft's style: a place as an empty wide shot at
eye level (landscape, as the scenes it is composed into), an object alone on a plain background (square). Every other
photo is another view of the same place or object made by Qwen-Image-Edit from the first one ("the same place as in
image 1"), so it keeps its design, colors and layout; drawing each view from the text alone would give another place
every time. Every photo passes the item gate (items.check_photo: no one who looks under 18); nothing is compared by
face.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import engine_runtime, items, model_manager, prefs, refs, styles

ProgressCb = Callable[[float, str], None]
# A place is composed into 16:9 scenes: landscape, about 1 megapixel; an object square.
SIZES = {items.PLACE: (1216, 832), items.OBJECT: (1024, 1024)}
MAX_TRIES = 3
_ANCHOR = {
    items.PLACE: ("{look}. The place is empty: no people and no animals. Wide establishing shot at eye level, the "
                  "whole space clearly visible, balanced even light"),
    items.OBJECT: ("{look}. One single object alone in the picture, centered and fully in view, three-quarter view, "
                   "plain light gray background, soft even studio light, no people and no animals"),
}
# The views after the first photo, in order (a regenerated photo keeps its slot's view).
PLACE_VIEWS = (
    "a reverse angle of the same place, seen from the opposite side, eye-level camera",
    "a wide shot of the whole place from a higher angle",
    "a medium shot of the same place from the left side, eye-level camera",
    "a medium shot of the same place from the right side, eye-level camera",
    "a closer view of the main area of the same place, eye-level camera",
    "a low-angle wide shot of the same place",
    "the same place seen from its entrance, eye-level camera",
)
OBJECT_VIEWS = (
    "the same object seen from its left side, plain light gray background",
    "the same object seen from its right side, plain light gray background",
    "the same object seen from behind, plain light gray background",
    "the same object seen from slightly above, plain light gray background",
    "a close-up of the details of the same object, plain light gray background",
    "the same object standing on a wooden table in a bright empty room",
    "the same object in a three-quarter view from the other side, plain white background",
)
_SAME = {
    items.PLACE: ("The same place as in image 1: the identical location with the same architecture, layout, "
                  "furniture, objects, screens, signs, colors, materials and lighting. No people and no animals. "
                  "Show "),
    items.OBJECT: ("The same object as in image 1: the identical object with the same shape, colors, materials, "
                   "markings, text and proportions. Only this one object, no people and no animals. Show "),
}
# The style's Z-Image words that are about a character (the 3D style's "expressive animated character", the 2D
# cartoon's "expressive face"): a place or an object leaves them out.
_CHARACTER_WORDS = re.compile(r"\b(?:characters?|faces?|persons?|people)\b", re.I)


def style_words(style: str) -> str:
    """The style's words for a Z-Image picture of a place or an object: its image suffix without the parts about a
    character."""
    parts = [part.strip() for part in styles.get(style).image.split(",")]
    return ", ".join(part for part in parts if part and not _CHARACTER_WORDS.search(part))


def anchor_prompt(look: str, kind: str, style: str) -> str:
    """The Z-Image prompt of the first photo."""
    return f"{_ANCHOR[kind].format(look=look.strip().rstrip('.'))}, {style_words(style)}"


def view_prompt(view: str, kind: str, style: str) -> str:
    """The Qwen-Image-Edit prompt of a further view: the same place or object as image 1, in its art style."""
    look = styles.get(style)
    if look.family == styles.REALISTIC:
        tail = ". Photorealistic photo, natural light."
    else:
        medium = look.medium or styles.BASES[look.family].medium
        tail = f". In exactly the art style of image 1 ({medium})."
    return _SAME[kind] + view + tail


def _views(kind: str) -> Tuple[str, ...]:
    return OBJECT_VIEWS if kind == items.OBJECT else PLACE_VIEWS


def _models(cb: ProgressCb, need_anchor: bool, style: str) -> None:
    ids = ([engine_runtime.ZIMAGE_ID] if need_anchor else []) + [engine_runtime.QWEN_EDIT_ID]
    if styles.get(style).has_loras:
        ids.append(styles.STYLE_MODEL_ID)
    problem = model_manager.ensure(ids, lambda f, d: cb(0.0, d)) or items.gate_models(lambda f, d: cb(0.0, d))
    if problem:
        raise ValueError(problem)


def _gate(target: Path, style: str) -> str:
    """items.check_photo; a refusal (⛔) removes the file and stops the job."""
    why = items.check_photo(target, style)
    if why.startswith("⛔"):
        target.unlink(missing_ok=True)
        raise ValueError(why)
    return why


def _anchor(client, look: str, kind: str, target: Path, seed: int, notes: List[str], style: str) -> bool:
    """Draw the first photo into `target`; True when one take passed the gate."""
    prompt = anchor_prompt(look, kind, style)
    width, height = SIZES[kind]
    for attempt in range(MAX_TRIES):
        try:
            data = engine_runtime.render(prompt, width, height, seed + 7919 * attempt, prefs.IMAGE_STEPS,
                                         runtime=client, **({} if style == styles.REALISTIC else {"style": style}))
        except Exception as error:  # noqa: BLE001 - the message goes to the job
            raise ValueError(f"The first photo could not be generated ({error}).") from error
        target.write_bytes(data)
        why = _gate(target, style)
        if not why:
            return True
        notes.append(f"Photo 1, take {attempt + 1}: {why}")
    target.unlink(missing_ok=True)
    return False


def _view(client, anchor: Path, kind: str, view: str, target: Path, seed: int, style: str) -> Tuple[bool, str]:
    """One view of the first photo's place or object into `target`: (True, "") once a take passes the gate, else
    (False, why the last take was refused)."""
    prompt = view_prompt(view, kind, style)
    width, height = SIZES[kind]
    why = ""
    for attempt in range(MAX_TRIES):
        ok, message = refs.compose(client, [str(anchor)], prompt, width, height, target, seed + 7919 * attempt,
                                   **({} if style == styles.REALISTIC else {"style": style}))
        if not ok:
            raise ValueError(message)
        why = _gate(target, style)
        if not why:
            return True, ""
    target.unlink(missing_ok=True)
    return False, why


def generate(draft_id: str, count: int, keep_own: bool, cb: ProgressCb) -> List[str]:
    """Fill the draft with `count` photos of one place or object; returns the job's notes. keep_own: the uploaded or
    saved photos stay (the first is the main one) and views of it are added up to `count`; otherwise a new place or
    object replaces every photo."""
    view = items.draft(draft_id)
    look, kind, style = view["look"], view["kind"], view["style"]
    photos = items.draft_photos(draft_id)
    kept = [entry for _, entry in photos if entry.get("origin") in ("uploaded", "kept")] if keep_own else []
    anchor_path: Optional[Path] = next(path for path, entry in photos if entry is kept[0]) if kept else None
    if not kept and (problem := items.check_look(look, kind)):
        raise ValueError(problem)
    _models(cb, need_anchor=anchor_path is None, style=style)
    client = engine_runtime.engine()
    seed = random.randint(0, 2 ** 31 - 1)
    notes: List[str] = []
    new: List[Tuple[Path, str]] = []
    try:
        if anchor_path is None:
            cb(0.02, "Photo 1: the place..." if kind == items.PLACE else "Photo 1: the object...")
            anchor_path = items.new_photo_path(draft_id)
            if not _anchor(client, look, kind, anchor_path, seed, notes, style):
                why = f" Last try: {notes[-1].split(': ', 1)[-1]}" if notes else ""
                raise ValueError(f"No first photo passed the check. Change the description and try again.{why}")
            new.append((anchor_path, "generated"))
        wanted = max(0, min(count, items.MAX_PHOTOS) - len(kept) - len(new))
        views = _views(kind)
        for number in range(wanted):
            index = len(kept) + len(new)
            cb(0.05 + 0.9 * number / max(wanted, 1), f"Photo {index + 1}/{count}...")
            target = items.new_photo_path(draft_id)
            ok, why = _view(client, anchor_path, kind, views[number % len(views)], target,
                            seed + 101 * (index + 1), style)
            if ok:
                new.append((target, "generated"))
            else:
                notes.append(f"Photo {index + 1} was left out: {why}.")
    except BaseException:  # cancelled or failed: the draft keeps the photos it had
        for path, _ in new:
            path.unlink(missing_ok=True)
        raise
    items.replace_photos(draft_id, [*kept, *new])
    notes.insert(0, f"{len(kept) + len(new)} reference photos ready. Review them before saving.")
    return notes


def regenerate(draft_id: str, index: int, cb: ProgressCb) -> List[str]:
    """A new take of the generated photo at `index`: a view of the first photo, or for the first photo itself a new
    first photo composed from the second (a new one from the description when it is the only photo)."""
    photos = items.draft_photos(draft_id)
    if not 0 <= index < len(photos) or photos[index][1].get("origin") != "generated":
        raise ValueError("That photo can no longer be generated again.")
    view = items.draft(draft_id)
    kind, style = view["kind"], view["style"]
    source = photos[1][0] if index == 0 and len(photos) > 1 else photos[0][0] if index else None
    if source is None and (problem := items.check_look(view["look"], kind)):
        raise ValueError(problem)
    _models(cb, need_anchor=source is None, style=style)
    client = engine_runtime.engine()
    seed = random.randint(0, 2 ** 31 - 1)
    notes: List[str] = []
    target = items.new_photo_path(draft_id)
    cb(0.05, f"Photo {index + 1}...")
    if source is None:
        ok = _anchor(client, view["look"], kind, target, seed, notes, style)
        why = notes[-1].split(": ", 1)[-1] if notes else "it did not pass the check"
    else:
        views = _views(kind)
        shot = (("the same place again, wide establishing shot at eye level" if kind == items.PLACE else
                 "the same object again, centered, three-quarter view, plain light gray background")
                if index == 0 else views[(index - 1) % len(views)])
        ok, why = _view(client, source, kind, shot, target, seed, style)
    if not ok:
        raise ValueError(f"Photo {index + 1} was not replaced: {why}.")
    entries: list = [entry for _, entry in photos]
    entries[index] = (target, "generated")
    items.replace_photos(draft_id, entries)
    return [f"Photo {index + 1} generated again."]
