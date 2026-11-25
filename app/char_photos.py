# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Reference photos of one new character for the character creator (the GPU part of app/characters.py).

The first photo is drawn by Z-Image-Turbo from the look and gender: a frontal head-and-shoulders portrait
(the anchor). Every other photo is a different view of the same person made by Qwen-Image-Edit from the
anchor ("the same person as in image 1"): drawing each photo from the text alone gave a different person
every time, while an edit of the anchor keeps the face. That is checked, not assumed: a view whose face is
less similar to the anchor than qc.FACE_MIN (SFace) is drawn again with another seed, up to MAX_TRIES, and a
view that never matches is left out with a note. Every photo, the anchor too, passes the adult / one-face
gate (characters.check_photo) before it reaches the draft.

The draft's style (app/styles.py) decides how the photos are drawn: a real person's photos, or a 3D animated, 2D
cartoon, anime or custom-style character (the anchor's wording, its image style, and the end of every view's prompt). The face
check of a view uses that style's threshold (qc.FACE_RULES).

An animal or creature (the draft's kind characters.CREATURE, only in a style that allows_creatures) has words of its
own: the style's creature anchor drawn from the look alone (no gender, no default outfit), a creature lead and
CREATURE_VIEWS (its own outfit or none, nobody else in the picture), and the style's creature view tail. Its photos
pass the creature gate (no person under 18) and are never compared by face.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import characters, engine_runtime, model_manager, prefs, qc, refs, styles

ProgressCb = Callable[[float, str], None]
SIZE = (832, 1216)  # portrait, about 1 megapixel: the size both models draw well
MAX_TRIES = 3
# The anchor names a plain top (a look default, not a content rule): every view is composed from this portrait,
# and a man's portrait without any clothing in the prompt often came out shirtless.
ANCHOR = styles.get(styles.REALISTIC).anchor  # each style has its own (styles.Style.anchor)
# The views after the anchor: angles, framings and expressions a scene may need, all with the face visible.
VIEWS = (
    "head and shoulders, three-quarter view turned slightly to the left, soft smile, plain light gray "
    "background, soft daylight",
    "head and shoulders, three-quarter view turned slightly to the right, neutral expression, plain light "
    "background, window light",
    "half body, standing in a relaxed pose, looking at the camera, plain studio background, even light",
    "close-up portrait, laughing, looking slightly to the side, soft daylight",
    "full body, standing, casual everyday clothes, plain light gray studio background, face clearly visible",
    "half body, sitting at a table in a bright cafe, natural window light, looking at the camera",
    "medium shot outdoors on a quiet street, overcast daylight, looking at the camera",
)
_SAME = ("The same person as in image 1: identical face, facial features, hairstyle, hair color, skin tone and "
         "body. One person only. ")
_FRONT = "head and shoulders, looking straight at the camera, neutral relaxed expression, plain light gray background"
# The views of an animal or creature, in the order of VIEWS (a regenerated photo keeps its slot's view): the same
# angles and expressions, its own outfit (or none) instead of everyday clothes, and nobody else in the cafe or street.
CREATURE_VIEWS = (
    *VIEWS[:4],
    "full body, standing, wearing exactly the same outfit as in image 1, or nothing if image 1 shows none, plain "
    "light gray studio background, face clearly visible",
    "half body, sitting at a table in an empty bright cafe, nobody else in the picture, natural window light, looking "
    "at the camera",
    "medium shot outdoors on an empty quiet street, nobody else in the picture, overcast daylight, looking at the "
    "camera",
)
_SAME_CREATURE = ("The same character as in image 1: the identical non-human creature with the same species, face, "
                  "colors, markings, body shape, proportions and outfit. Only this one character in the picture: no "
                  "people and no other animals. ")


def _kind(view: dict) -> str:
    """The kind of character a draft (characters.draft) holds: characters.CREATURE or PERSON (an unknown kind too).
    ValueError when its style cannot draw it: a creature needs a style that allows_creatures."""
    if not characters.is_creature(view):
        return characters.PERSON
    if not styles.get(view.get("style", styles.REALISTIC)).allows_creatures:
        raise ValueError(characters.CREATURE_STYLE_MSG)
    return characters.CREATURE


def _views(kind: str) -> Tuple[str, ...]:
    return CREATURE_VIEWS if kind == characters.CREATURE else VIEWS


def _passes(kind: str) -> str:
    """What the gate lets through, for the message about a photo that did not pass it."""
    return "no person under 18" if kind == characters.CREATURE else "one adult, one clear face"


def _models(cb: ProgressCb, need_anchor: bool, style: str = styles.REALISTIC) -> None:
    ids = ([engine_runtime.ZIMAGE_ID] if need_anchor else []) + [engine_runtime.QWEN_EDIT_ID]
    if styles.get(style).has_loras:
        ids.append(styles.STYLE_MODEL_ID)
    problem = model_manager.ensure(ids, lambda f, d: cb(0.0, d)) or characters.gate_models(lambda f, d: cb(0.0, d))
    if problem:
        raise ValueError(problem)


def _anchor(client, look: str, gender: str, target: Path, seed: int, notes: List[str],
            style: str = styles.REALISTIC, kind: str = characters.PERSON) -> bool:
    """Draw the anchor portrait in `style` into `target`; True when one take passed the gate. A creature is drawn
    from the style's creature anchor: its look alone, without a gender or a default outfit."""
    if kind == characters.CREATURE:
        text = styles.get(style).creature_anchor.format(look=look.rstrip("."))
    else:
        text = styles.get(style).anchor.format(gender=gender, look=look.rstrip("."))
    prompt = prefs.styled(text, "image", style)
    for attempt in range(MAX_TRIES):
        try:
            data = engine_runtime.render(prompt, *SIZE, seed + 7919 * attempt, prefs.IMAGE_STEPS, runtime=client,
                                         **({} if style == styles.REALISTIC else {"style": style}))
        except Exception as error:  # noqa: BLE001 - the message goes to the job
            raise ValueError(f"The first photo could not be generated ({error}).") from error
        target.write_bytes(data)
        problem = characters.check_photo(target, style, kind=kind)
        if not problem:
            return True
        if problem.startswith("⛔"):
            target.unlink(missing_ok=True)
            raise ValueError(problem)
        notes.append(f"Photo 1, take {attempt + 1}: {problem}")
    target.unlink(missing_ok=True)
    return False


def _view(client, anchor: Path, ch: dict, view: str, target: Path, seed: int,
          style: str = styles.REALISTIC, kind: str = characters.PERSON) -> Tuple[bool, str]:
    """One view of the anchor's character in `style` into `target`: (True, "") once a take passes the gate and keeps
    the face (the style's FaceRules.minimum; a creature's face is never compared), else (False, why the last take
    was refused)."""
    creature = kind == characters.CREATURE
    if creature:
        prompt = _SAME_CREATURE + view + styles.get(style).creature_view
    else:
        prompt = _SAME + view + styles.get(style).view
    minimum = qc.rules(style).minimum
    why = ""
    for attempt in range(MAX_TRIES):
        ok, message = refs.compose(client, [str(anchor)], prompt, *SIZE, target, seed + 7919 * attempt,
                                   **({} if style == styles.REALISTIC else {"style": style}))
        if not ok:
            raise ValueError(message)
        why = characters.check_photo(target, style, kind=kind)
        if why.startswith("⛔"):
            target.unlink(missing_ok=True)
            raise ValueError(why)
        if not why and (creature or not qc.rules(style).detects):
            return True, ""  # no face to measure (a creature's, or this style's): the edit of the anchor keeps it
        if not why:
            score = qc.face_score(str(target), ch)
            if score is not None and score >= minimum:
                return True, ""
            why = "the face did not match the first photo" if score is not None else "no clear face"
    target.unlink(missing_ok=True)
    return False, why


def _anchor_profile(anchor: Path, style: str = styles.REALISTIC) -> dict:
    """A stand-in character whose only reference is the anchor, for the SFace comparison of a view."""
    return {"slug": f"draft-{anchor.stem}", "photos": [str(anchor)], "style": style}


def generate(draft_id: str, count: int, keep_own: bool, cb: ProgressCb) -> List[str]:
    """Fill the draft with `count` photos of one person (an animal or creature for a creature draft); returns the
    job's notes. keep_own: the uploaded photos stay (the first is the anchor) and views are added up to `count`;
    otherwise a new person replaces every photo."""
    view = characters.draft(draft_id)
    look, gender, style = view["look"], view["gender"], view.get("style", styles.REALISTIC)
    kind = _kind(view)
    photos = characters.draft_photos(draft_id)
    kept = [entry for _, entry in photos if entry.get("origin") in ("uploaded", "kept")] if keep_own else []
    anchor_path: Optional[Path] = next(path for path, entry in photos if entry is kept[0]) if kept else None
    if not kept and (problem := characters.check_look(look, kind=kind)):
        raise ValueError(problem)
    _models(cb, need_anchor=anchor_path is None, style=style)
    client = engine_runtime.engine()
    seed = random.randint(0, 2 ** 31 - 1)
    notes: List[str] = []
    new: List[Tuple[Path, str]] = []
    try:
        if anchor_path is None:
            cb(0.02, "Photo 1: the character..." if kind == characters.CREATURE else "Photo 1: the person...")
            anchor_path = characters.new_photo_path(draft_id)
            if not _anchor(client, look, gender, anchor_path, seed, notes, style, kind=kind):
                why = (f" Last try: {notes[-1].split(': ', 1)[-1]}" if kind == characters.CREATURE and notes
                       else "")
                raise ValueError(f"No first photo passed the check ({_passes(kind)}). Change the look description "
                                 f"and try again.{why}")
            new.append((anchor_path, "generated"))
        wanted = max(0, min(count, characters.MAX_PHOTOS) - len(kept) - len(new))
        views = _views(kind)
        for number in range(wanted):
            index = len(kept) + len(new)
            cb(0.05 + 0.9 * number / max(wanted, 1), f"Photo {index + 1}/{count}...")
            target = characters.new_photo_path(draft_id)
            ok, why = _view(client, anchor_path, _anchor_profile(anchor_path, style), views[number % len(views)],
                            target, seed + 101 * (index + 1), style, kind=kind)
            if ok:
                new.append((target, "generated"))
            else:
                notes.append(f"Photo {index + 1} was left out: {why}.")
    except BaseException:  # cancelled or failed: the draft keeps the photos it had
        for path, _ in new:
            path.unlink(missing_ok=True)
        raise
    characters.replace_photos(draft_id, [*kept, *new])
    notes.insert(0, f"{len(kept) + len(new)} reference photos ready. Review them before saving.")
    return notes


def regenerate(draft_id: str, index: int, cb: ProgressCb) -> List[str]:
    """A new take of the generated photo at `index`: a view of the first photo, or for the first photo itself a
    new frontal portrait of the same person composed from the second (a new person when it is the only one)."""
    photos = characters.draft_photos(draft_id)
    if not 0 <= index < len(photos) or photos[index][1].get("origin") != "generated":
        raise ValueError("That photo can no longer be generated again.")
    view = characters.draft(draft_id)
    style = view.get("style", styles.REALISTIC)
    kind = _kind(view)
    source = photos[1][0] if index == 0 and len(photos) > 1 else photos[0][0] if index else None
    if source is None and (problem := characters.check_look(view["look"], kind=kind)):  # a new character from the
        raise ValueError(problem)                                                       # look, as generate() draws one
    _models(cb, need_anchor=source is None, style=style)
    client = engine_runtime.engine()
    seed = random.randint(0, 2 ** 31 - 1)
    notes: List[str] = []
    target = characters.new_photo_path(draft_id)
    cb(0.05, f"Photo {index + 1}...")
    if source is None:
        ok = _anchor(client, view["look"], view["gender"], target, seed, notes, style, kind=kind)
        why = f"it did not pass the check ({_passes(kind)})"
    else:
        views = _views(kind)
        shot = _FRONT if index == 0 else views[(index - 1) % len(views)]  # the view this slot was made with
        ok, why = _view(client, source, _anchor_profile(source, style), shot, target, seed, style, kind=kind)
    if not ok:
        raise ValueError(f"Photo {index + 1} was not replaced: {why}.")
    items: list = [entry for _, entry in photos]
    items[index] = (target, "generated")
    characters.replace_photos(draft_id, items)
    return [f"Photo {index + 1} generated again."]
