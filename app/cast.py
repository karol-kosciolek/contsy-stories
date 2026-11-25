# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The cast of a video or photo: which characters take part (each with its own reference photos).

The cast is the main character (the top bar's character picker: a character, or a universe -
its first ready member), the characters picked in "Characters in this video/photo" and every ready character
named in the prompt (up to MAX_CAST). A name works as written ("Ola", "Anna Maria"), with
@ ("@ola", "@anna_maria") and in common Polish case forms ("Kubie", "Oli", "Jana").

Inside a video, names decide who is in each shot: a shot that names characters shows exactly
them, a shot without names continues with the previous shot's characters, and a first shot
without names shows the main character - as before. A frame with two or more characters is
made by compose_frame: the primary reference photo of every character is composed into the
described scene (Qwen-Image-Edit, app/refs.py), each person from their own image. A character's
name also picks its voice.

An animal or creature character (characters.is_creature) is "the creature" in compositions and to the
video model, never a woman, a man or a person; it is drawn only in animated styles, so the always-Realistic
Video and Image pages refuse it (creature_error, check_prompt).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import characters, dialogue, prefs, universes

MAX_CAST = 5
UNIVERSE = "universe:"
# Polish case forms of a first name: Kuba -> Kuby, Kubie, Kubę, Kubą; Jan -> Jana, Janowi, Janem
_A_FORMS = ("y", "i", "ie", "ę", "ą", "o", "u")
_CONSONANT_FORMS = ("a", "owi", "em", "ie", "u", "y")


# ----------------------------- names in prompts -----------------------------

def name_pattern(ch: dict) -> "re.Pattern[str]":
    """The character's name in a prompt: @slug or @name (any letter case), the name as written
    (first letter capital, rest any case) and, for one-word names, Polish case forms."""
    name = " ".join(ch["name"].split())
    words = name.split()
    tail = re.escape(words[0][1:]) + "".join(r"\s+" + re.escape(w) for w in words[1:])
    exact = re.escape(words[0][:1].upper()) + (f"(?i:{tail})" if tail else "")
    if re.search(r"[_\d]", name):  # "chain_dog" is no ordinary word: any letter case names the character
        exact = f"(?i:{re.escape(words[0])}{tail[len(re.escape(words[0][1:])):] if tail else ''})"
    spaced = r"[\s_]+".join(re.escape(w) for w in words)
    tagged = rf"@(?i:{re.escape(ch['slug'])}|{spaced})"
    forms = ""
    if len(words) == 1 and len(name) >= 3 and name[0].isalpha():
        cap = name[0].upper() + name[1:]
        if name[-1].lower() == "a":
            forms = rf"|(?P<f>{re.escape(cap[:-1])}(?:{'|'.join(_A_FORMS)}))"
        elif name[-1].lower() not in "aeiouyąęó":
            forms = rf"|(?P<f>{re.escape(cap)}(?:{'|'.join(_CONSONANT_FORMS)}))"
    return re.compile(rf"(?<![\w@])(?:(?P<x>{tagged}|{exact}){forms})(?!\w)")


def _spans(text: str, chars: Sequence[dict]) -> List[Tuple[int, int, dict]]:
    """Non-overlapping mentions (start, end, character) in text order; a name written
    exactly wins over another character's case form at the same place ("Anna" vs "Ann")."""
    found = []
    for ch in chars:
        for m in name_pattern(ch).finditer(text):
            found.append((m.start(), m.end(), 0 if m.group("x") else 1, ch))
    found.sort(key=lambda f: (f[0], f[2], -(f[1] - f[0])))
    out, end = [], -1
    for start, stop, _, ch in found:
        if start >= end:
            out.append((start, stop, ch))
            end = stop
    return out


def mentions(text: str, chars: Sequence[dict]) -> List[dict]:
    """Characters named in text (outside quoted dialogue), in order of first mention."""
    plain = dialogue.QUOTE.sub(" ", text or "")
    seen: Dict[str, dict] = {}
    for _, _, ch in _spans(plain, chars):
        seen.setdefault(ch["slug"], ch)
    return list(seen.values())


def strip_mentions(text: str, chars: Sequence[dict]) -> str:
    """text without the characters' names (to find other @tags, e.g. voices)."""
    out, pos = [], 0
    for start, end, _ in _spans(text or "", chars):
        out += [text[pos:start], " "]
        pos = end
    return "".join(out) + (text or "")[pos:]


def _replace(text: str, members: Sequence[dict], label: Callable[[dict], str]) -> str:
    out, pos = [], 0
    for start, end, ch in _spans(text, members):
        word = label(ch)
        if not text[:start].strip() or re.search(r"[.!?]\s*$", text[:start]):
            word = word[:1].upper() + word[1:]
        out += [text[pos:start], word]
        pos = end
    return "".join(out) + text[pos:]


def mark_names(text: str, members: Sequence[dict]) -> str:
    """Names -> "the woman [1]" (the character's image number) for composing the scene; an animal or creature
    is "the creature [2]"."""
    number = {ch["slug"]: i + 1 for i, ch in enumerate(members)}
    return _replace(text, members, lambda ch: f"the {_gender_word(ch)} [{number[ch['slug']]}]")


def labels(members: Sequence[dict]) -> Dict[str, str]:
    """How the video model hears each character: "the woman", or "the first woman" and
    "the second woman" when the shot has two of the same gender. Animals and creatures are a group
    of their own: "the creature", "the first creature"..."""
    ordinal = ("first", "second", "third", "fourth", "fifth")
    by_gender: Dict[str, List[dict]] = {}
    for ch in members:
        by_gender.setdefault(_gender_word(ch), []).append(ch)
    out = {}
    for word, group in by_gender.items():
        for i, ch in enumerate(group):
            out[ch["slug"]] = f"the {word}" if len(group) == 1 else \
                f"the {ordinal[min(i, len(ordinal) - 1)]} {word}"
    return out


def label_names(text: str, members: Sequence[dict]) -> str:
    names = labels(members)
    return _replace(text, members, lambda ch: names[ch["slug"]])


def look_sentences(members: Sequence[dict]) -> str:
    """". The woman has ... . The man has ..." - each character's look for the video model; an animal or
    creature's reads ". The creature looks like this: ..." (its look as written)."""
    names = labels(members)
    out = []
    for ch in members:
        look = ch.get("look")
        if not look:
            continue
        label = names[ch["slug"]][:1].upper() + names[ch["slug"]][1:]
        out.append(f". {label} looks like this: {look.strip().rstrip('.')}" if characters.is_creature(ch) else
                   f". {label} has {look[:1].lower()}{look[1:].rstrip('.')}")
    return "".join(out)


def _gender_word(ch: dict) -> str:
    """How compositions and the video model name a character: "woman", "man", or "creature" for an animal or
    creature (whose gender is only its voice and its she / he)."""
    if characters.is_creature(ch):
        return "creature"
    return "man" if ch.get("gender") == "man" else "woman"


# ----------------------------- who is in the job -----------------------------

def ready_characters() -> List[dict]:
    return [ch for ch in characters.all_characters() if characters.usable(ch["slug"])]


def resolve(selected: Sequence[str], prompt: str, cap: int = MAX_CAST) -> Tuple[List[dict], str]:
    """(cast, error): the picked characters in order (the first is the main one), then every
    ready character named in the prompt. More than `cap` is an error, never a silent cut."""
    cast: List[dict] = []
    for slug in selected or []:
        ch = characters.usable(slug) if slug else None
        if ch and all(c["slug"] != ch["slug"] for c in cast):
            cast.append(ch)
    for ch in mentions(prompt, ready_characters()):
        if all(c["slug"] != ch["slug"] for c in cast):
            cast.append(ch)
    if len(cast) > cap:
        names = ", ".join(c["name"] for c in cast)
        return cast, (f"Up to {cap} characters can be in one video or photo (now: {names}). "
                      "Remove some from “Characters” or name fewer in the prompt.")
    return cast, ""


def not_ready(prompt: str) -> List[str]:
    """Names of characters mentioned in the prompt that cannot be used (no reference photos)."""
    waiting = [ch for ch in characters.all_characters() if not characters.usable(ch["slug"])]
    return [ch["name"] for ch in mentions(prompt, waiting)]


def shot_casts(shots: Sequence[str], cast: Sequence[dict]) -> List[List[dict]]:
    """Characters in each shot: the ones it names; none named - the previous shot's; the
    first shot without names - the main character only."""
    out, previous = [], list(cast[:1])
    for shot in shots:
        named = mentions(shot, cast)
        current = named or previous
        out.append(list(current))
        previous = current
    return out


def appearing(shots: Sequence[str], cast: Sequence[dict]) -> List[dict]:
    """Cast members who are actually in the video or photo (shot_casts), in cast order: a
    picked character the prompt never names stays out; the main one stays first."""
    seen = {c["slug"] for members in shot_casts(shots, cast) for c in members}
    return [c for c in cast if c["slug"] in seen]


def untag(text: str, cast: Sequence[dict]) -> str:
    """'@kuba smiles' -> 'Kuba smiles' outside quoted text, so image models never see the @
    syntax (as on the Video page); a quoted sign reading "@kuba" stays as written."""
    ppl, out, pos = people(cast), [], 0
    for q in dialogue.QUOTE.finditer(text or ""):
        out += [dialogue._untag(text[pos:q.start()], ppl), q.group(0)]
        pos = q.end()
    out.append(dialogue._untag((text or "")[pos:], ppl))
    return "".join(out)


def people(cast: Sequence[dict], pool=None) -> List[Tuple["re.Pattern[str]", str, str, str, str]]:
    """For app/dialogue.py: (name pattern, slug, 'female'/'male', voice slug or '', name)."""
    from . import voices

    out = []
    for ch in cast:
        voice = voices.get(ch.get("voice") or "", pool)
        out.append((name_pattern(ch), ch["slug"], "male" if ch.get("gender") == "man" else "female",
                    voice.slug if voice else "", ch["name"]))
    return out


def summary(cast: Sequence[dict]) -> str:
    """" · Ola + Kuba" for job labels ("" without characters)."""
    return f" · {' + '.join(ch['name'] for ch in cast)}" if cast else ""


def tag(params: dict, cast: Sequence[dict], shots: Sequence[str] = ()) -> str:
    """Put the cast into a job's params (main character first); returns the label suffix.
    shots - the video's shots (refs.split_shots) or the photo's prompt: the characters
    composed into shared frames beyond the first make the queue estimate longer."""
    if cast:
        params["character"] = cast[0]["slug"]
        params["characters"] = [ch["slug"] for ch in cast]
        params["extra_characters"] = sum(len(m) - 1 for m in shot_casts(list(shots), cast)
                                         if len(m) > 1)
    return summary(cast)


def picker_choices() -> List[Tuple[str, str]]:
    """(name, slug) of every ready character, for "Characters in this video/photo"."""
    return [(ch["name"], ch["slug"]) for ch in ready_characters()]


def creature_error(members: Sequence[Optional[dict]]) -> str:
    """Why these characters cannot be in a picture or video of a Realistic style ("" = they can): the first animal
    or creature among them, which is drawn only in animated styles (the Video and Image pages are always
    Realistic)."""
    found = next((ch for ch in members if characters.is_creature(ch)), None)
    return f"{found['name']} is an animal or creature: use it in a Timeline film in an animated style." \
        if found else ""


def check_prompt(prompt: str, picked_slugs: Sequence[str],
                 shots: Optional[Sequence[str]] = None, *,
                 style: str = "") -> Tuple[List[dict], str, List[str]]:
    """For the Video and Image pages, before queueing: (cast, error, notes). The cast is who
    will be in the result (shots - the video's shots; a photo is one): picked characters the
    prompt never names stay out of the job, its gallery entries and its time estimate. The
    error says when an animal or creature would appear in a picture of `style` (those pages: Realistic, "";
    a Timeline film passes its own), more than MAX_CAST characters would appear or an @tag is neither a
    character nor a voice."""
    from . import voices

    pool, _ = resolve(picked_slugs, prompt, cap=10 ** 6)
    found = appearing(list(shots) if shots else [prompt], pool)
    refused = "" if characters.creature_style_ok(style) else creature_error(found)
    if refused:
        return found, refused, []
    if len(found) > MAX_CAST:
        names = ", ".join(c["name"] for c in found)
        return found, (f"Up to {MAX_CAST} characters can be in one video or photo (now: {names}). "
                       "Name fewer characters in the prompt."), []
    known = voices.tokens()
    others = dialogue.tags(strip_mentions(prompt, characters.all_characters()))
    unknown = [t for t in others if t.casefold() not in known]
    if unknown:
        return found, (f"No voice or character is named @{unknown[0]}. Use a name from the "
                       "Voices page or a character, or add it first."), []
    notes = []
    waiting = not_ready(prompt)
    if waiting:
        notes.append(f"No reference photos yet, so left out: {', '.join(waiting)}.")
    unnamed = [c["name"] for c in pool if c not in found]
    if unnamed:
        notes.append(f"Not named in the prompt, so not in it: {', '.join(unnamed)}.")
    return found, "", notes


def for_job(p: dict) -> Tuple[List[dict], str]:
    """The cast of a queued job: params['characters'] when the page set them; otherwise the
    main character plus characters named in the prompt."""
    if isinstance(p.get("characters"), list):
        cast = [ch for ch in (characters.usable(s) for s in p["characters"]) if ch]
        missing = [s for s in p["characters"] if not characters.usable(s)]
        note = ("Some characters are unavailable and were left out: " + ", ".join(missing) + "."
                if missing else "")
        return cast[:MAX_CAST], note
    main, note = characters.for_job(p)
    cast, _ = resolve([main["slug"]] if main else [], str(p.get("prompt") or ""))
    return cast[:MAX_CAST], note


# ------------- the top bar's character picker (Tools panel): a character or a universe -------------

def top_picker_choices() -> List[Tuple[str, str]]:
    """The top picker's choices: every character, then every universe as "universe:<slug>"."""
    out = list(characters.choices())
    for u in universes.all_universes():
        out.append((f"Universe: {u['name']} ({len(universes.members(u))})", UNIVERSE + u["slug"]))
    return out


def current_universe() -> Optional[dict]:
    return universes.get(prefs.load().get("universe", ""))


def top_picker_value() -> str:
    """The top picker's value now; with a universe selected, its main character is kept in step
    (a member was added, deleted...)."""
    u = current_universe()
    if not u:
        return prefs.load().get("character", "")
    ready = universes.members(u)
    main = ready[0]["slug"] if ready else ""
    if prefs.load().get("character", "") != main:
        prefs.save(character=main)
    return UNIVERSE + u["slug"]


def select(value: str) -> str:
    """Save the top picker's choice. A universe makes its first ready member the main character.
    Returns the main character's slug ('' = none)."""
    value = value or ""
    if value.startswith(UNIVERSE):
        u = universes.get(value[len(UNIVERSE):])
        ready = universes.members(u)
        main = ready[0]["slug"] if ready else ""
        prefs.save(character=main, universe=u["slug"] if u else "")
        return main
    prefs.save(character=value, universe="")
    return value


def picked(value: Optional[str] = None) -> List[str]:
    """Pre-fill of "Characters in this video/photo" for a top picker value (default: the saved
    one): the character, or every ready member of the universe."""
    if value is None:
        u = current_universe()
        if not u:
            ch = characters.selected()
            return [ch["slug"]] if ch else []
    elif value.startswith(UNIVERSE):
        u = universes.get(value[len(UNIVERSE):])
    else:
        return [value] if value and characters.usable(value) else []
    return [ch["slug"] for ch in universes.members(u)][:MAX_CAST]


def profiles() -> List[str]:
    """Characters whose results the Dashboard and Gallery show now (a universe: all members)."""
    u = current_universe()
    if u:
        return [ch["slug"] for ch in universes.members(u)] or list(u["members"])
    ch = characters.selected()  # only a usable character filters, like studio.current()
    return [ch["slug"]] if ch else []


# ----------------------------- a frame with several characters -----------------------------

def _solo(ch: dict) -> Tuple[bool, str]:
    """The image a character brings into a composed frame: its primary reference photo."""
    source = characters.reference_portrait(ch)
    if source is None:
        return False, "this character has no reference photo"
    return True, str(source)


def compose_frame(client, members: Sequence[dict], text: str, width: int, height: int,
                  out_path: str | Path, seed: int,
                  progress_cb: Optional[Callable[[float, str], None]] = None,
                  extra_refs: Optional[List[Tuple[int, str]]] = None,
                  looks: Optional[Dict[int, str]] = None, style: str = "realistic") -> Tuple[bool, str]:
    """A frame (width x height) with every member of the shot, composed from their primary reference
    photos and the user's reference images. text - the shot description with names;
    extra_refs - the user's reference images [(number, path)] with their looks; style - the film's style
    (app/styles.py).
    Returns (ok, frame path or error). When it is stopped (cancel.Cancelled) or fails with an
    exception, it removes the files of compose_refs' passes first."""
    out_path = Path(out_path)
    try:
        return _compose_frame(client, list(members), text, width, height, out_path, seed, progress_cb,
                              extra_refs, looks, style)
    except BaseException:
        for path in out_path.parent.glob(f"{out_path.stem}_*.png"):
            path.unlink(missing_ok=True)
        raise


def _compose_frame(client, members: List[dict], text: str, width: int, height: int,
                   out_path: Path, seed: int,
                   progress_cb: Optional[Callable[[float, str], None]],
                   extra_refs: Optional[List[Tuple[int, str]]],
                   looks: Optional[Dict[int, str]], style: str = "realistic") -> Tuple[bool, str]:
    from . import refs

    def step(i: float, desc: str) -> None:
        if progress_cb:
            progress_cb(i, desc)

    solos = []
    for ch in members:
        ok, msg = _solo(ch)
        if not ok:
            return False, f"{ch['name']}: {msg}"
        solos.append(msg)
    k = len(members)
    scene = mark_names(refs.renumber(text, k), members)  # the user's [n] first, then names
    ref_list = [(i + 1, path) for i, path in enumerate(solos)]
    ref_list += [(n + k, path) for n, path in (extra_refs or [])]
    all_looks = {i + 1: ch.get("look", "") for i, ch in enumerate(members)}
    all_looks.update({n + k: look for n, look in (looks or {}).items()})
    by_number = {i + 1: ch for i, ch in enumerate(members)}
    creatures = [n for n, ch in by_number.items() if characters.is_creature(ch)]  # their images, in their words
    leads = []
    for p, chunk in enumerate(refs.split_passes(ref_list, False)):
        first = 1 if p == 0 else 2  # a later pass: image 1 is the picture made so far
        shown = [n for n, _ in chunk if n in by_number]
        leads.append(refs.cast_lead([by_number[n] for n in shown], first_image=first, adding=p > 0, style=style,
                                    creatures=[first + i for i, n in enumerate(shown) if n in creatures]))
    step(0.0, "composing the scene...")
    ok, msg = refs.compose_refs(
        client, None, ref_list, scene, width, height, out_path, seed,
        progress_cb=lambda f, d: step(f, f"composing the scene: {d}"),
        looks=all_looks, pass_leads=leads, style=style, creatures=creatures)
    if not ok:
        return False, msg
    return True, str(out_path)
