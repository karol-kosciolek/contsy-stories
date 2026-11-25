# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The owner's own styles: names for custom styles (app/styles.py), kept in characters/styles.json next to the
characters and universes.

A custom style's id holds its whole definition ("custom:1:<starting point>:<look words>"), so a film or a character
needs only its id: this library only gives those ids the owner's names and lists them in the pickers. Editing a style's
look or starting point makes a new id; the films and characters that use the old one keep it (shown as an earlier
version) until the owner picks the style again for them. Deleting a style never changes or locks a film either.
Reading never creates the file; a damaged file reads as no styles and is set aside before the next save.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from . import characters, llm, styles
from .persistence import atomic_text

MAX_NAME = 40
MAX_EARLIER = 10  # earlier versions of a style that keep its name in the pickers
NEW_STYLE = "New style"
_LOCK = threading.RLock()
_CACHE: dict = {}  # file stamp -> parsed entries


def _file() -> Path:
    return characters.DIR / "styles.json"


def _stamp(path: Path):
    try:
        stat = path.stat()
    except OSError:
        return None
    return str(path), stat.st_mtime_ns, stat.st_size


def _parse(data) -> List[dict]:
    """The entries of a library file's JSON data (an entry that cannot be read is left out); ValueError when the data
    is not a style library at all."""
    if not isinstance(data, dict) or not isinstance(data.get("styles", []), list):
        raise ValueError("This is not a style library.")
    out = []
    for item in data.get("styles", []):
        try:
            key, name = str(item["key"]), " ".join(str(item["name"]).split())
            style_id = styles.custom_id(item["base"], item["look"])
        except (KeyError, TypeError, ValueError):
            continue  # an entry that cannot be read is left out
        earlier = []
        for old in item.get("earlier", []) if isinstance(item.get("earlier"), list) else []:
            try:
                earlier.append(styles.custom_id(old["base"], old["look"]))
            except (KeyError, TypeError, ValueError):
                continue
        out.append({"key": key, "name": name, "base": item["base"], "look": styles.clean_look(item["look"]),
                    "id": style_id, "earlier": [old for old in dict.fromkeys(earlier) if old != style_id],
                    "created": item.get("created", 0), "updated": item.get("updated", 0)})
    return out


def _intact(text: str) -> bool:
    """Whether a library file's text reads in full: a style library with no entry left out."""
    try:
        data = json.loads(text)
        return len(_parse(data)) == len(data.get("styles", []))
    except ValueError:
        return False


def all_custom() -> List[dict]:
    """The owner's styles, each with its current id and its earlier ids (a missing or damaged file: none)."""
    path = _file()
    stamp = _stamp(path)
    if stamp is None:
        return []
    with _LOCK:
        if stamp in _CACHE:
            return [dict(entry, earlier=list(entry["earlier"])) for entry in _CACHE[stamp]]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []  # not remembered: the next call reads the file again
    try:
        entries = _parse(json.loads(text))
    except ValueError:
        entries = []
    with _LOCK:
        _CACHE.clear()
        _CACHE[stamp] = entries
    return [dict(entry, earlier=list(entry["earlier"])) for entry in entries]


def get(key: str) -> Optional[dict]:
    return next((entry for entry in all_custom() if entry["key"] == key), None) if key else None


def find(style_id: str) -> Optional[Tuple[dict, bool]]:
    """(entry, True) for a style's current version, (entry, False) for an earlier one, None otherwise."""
    entries = all_custom()
    for entry in entries:
        if entry["id"] == style_id:
            return entry, True
    for entry in entries:
        if style_id in entry["earlier"]:
            return entry, False
    return None


def label(style_id) -> str:
    """The name the pickers show: a preset's label, the owner's name of a custom style (an earlier version says
    so), or "Custom: <look>" for one not in the library."""
    look = styles.of(style_id)
    if not look.custom:
        return look.label
    found = find(look.id)
    if found is None:
        return look.label
    entry, current = found
    return entry["name"] if current else f"{entry['name']} (earlier version)"


def about(style_id) -> str:
    """One line on what the style looks like and where it comes from."""
    look = styles.of(style_id)
    if not look.custom:
        return look.about
    start = styles.base_label(look.id)
    found = find(look.id)
    if found is None:
        return f"Not in your styles any more; it keeps working here: {look.about} (starting from {start})."
    entry, current = found
    if current:
        return f"Your style “{entry['name']}”, starting from {start}: {look.about}."
    return (f"An earlier version of your style “{entry['name']}”: {look.about}. Choose “{entry['name']}” to use the "
            "new version; the scenes made in this one then show Outdated.")


def choices(selected: Optional[str] = None) -> List[Tuple[str, str]]:
    """(name, id) for a style picker: the presets, the owner's styles by name, and `selected` when it is a valid style
    the list does not hold (an earlier version, a deleted style), so a saved value is always among the choices."""
    found = styles.choices() + sorted(((entry["name"], entry["id"]) for entry in all_custom()),
                                      key=lambda item: item[0].casefold())
    # Contsy Stories makes animation only: the Realistic family is offered only where a stored record already uses it
    found = [(name, value) for name, value in found if styles.of(value).family != styles.REALISTIC]
    if selected and styles.valid(selected) and selected not in {value for _, value in found}:
        found.append((label(selected), selected))
    return found


def check(name: str, base: str, look: str, key: str = "", entries: Optional[List[dict]] = None) -> str:
    """Why these cannot be saved as a style ("" = fine); key - the style being edited; entries - the library as read
    for the change (default: as cached)."""
    name = " ".join((name or "").split())
    if not name:
        return "Enter a name for the style."
    if len(name) > MAX_NAME:
        return f"Use at most {MAX_NAME} characters for the style name."
    if llm.blocked(name):
        return llm.BLOCK_MSG
    reserved = {label.casefold() for label, _ in styles.base_choices()} | {
        style.label.casefold() for style in styles.BASES.values()} | {NEW_STYLE.casefold()}
    if name.casefold() in reserved:
        return "Choose a name that is not one of the built-in styles."
    entries = all_custom() if entries is None else entries
    if any(entry["name"].casefold() == name.casefold() and entry["key"] != key for entry in entries):
        return f"You already have a style named “{name}”."
    if base not in styles.BASES:
        return "Choose what the style starts from."
    try:
        style_id = styles.custom_id(base, look)
    except ValueError as error:
        return str(error)
    problem = llm.style_problem(style_id)
    if problem:
        return problem
    same = next((entry for entry in entries if entry["id"] == style_id and entry["key"] != key), None)
    if same is not None:
        return f"You already have this style: “{same['name']}”."
    return ""


def _current() -> List[dict]:
    """The library read again for a change, never from the cache. OSError when the file is there but cannot be read,
    so a change never overwrites styles it could not see; a damaged file reads as none (_write sets it aside)."""
    path = _file()
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    try:
        return _parse(json.loads(text))
    except ValueError:
        return []


def _set_aside_damaged(path: Path) -> None:
    """A file that is there but does not read in full (bad JSON, another shape, an entry left out) is renamed before
    it is overwritten, so nothing the owner made is lost (OSError when it cannot be read or renamed)."""
    if not path.exists() or _intact(path.read_text(encoding="utf-8")):
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target, number = path.with_name(f"styles.damaged-{stamp}.json"), 1
    while target.exists():
        number += 1
        target = path.with_name(f"styles.damaged-{stamp}-{number}.json")
    path.rename(target)


def _write(entries: List[dict]) -> None:
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    _set_aside_damaged(path)
    stored = [{"key": entry["key"], "name": entry["name"], "base": entry["base"], "look": entry["look"],
               "earlier": [{"base": styles.of(old).base, "look": styles.of(old).about} for old in entry["earlier"]],
               "created": entry.get("created", 0), "updated": entry.get("updated", 0)} for entry in entries]
    atomic_text(path, json.dumps({"format": 1, "styles": stored}, ensure_ascii=False, indent=1))
    with _LOCK:
        _CACHE.clear()


def save(name: str, base: str, look: str, key: str = "") -> Tuple[Optional[dict], str]:
    """Save a new style (key "") or the style `key`: (the entry, a message for the owner) or (None, why not)."""
    with _LOCK:
        try:
            entries = _current()
        except OSError as error:
            return None, f"Could not read your styles: {error}"
        problem = check(name, base, look, key, entries)
        if problem:
            return None, problem
        name = " ".join(name.split())
        style_id = styles.custom_id(base, look)
        now = time.time()
        if not key:
            entry = {"key": secrets.token_hex(6), "name": name, "base": base, "look": styles.clean_look(look),
                     "id": style_id, "earlier": [], "created": now, "updated": now}
            try:
                _write(entries + [entry])
            except OSError as error:
                return None, f"Could not save the style: {error}"
            return entry, (f"Style “{name}” saved. Choose it in Timeline → Project → Style, or for a character in "
                           "Characters → 1. Look.")
        entry = next((item for item in entries if item["key"] == key), None)
        if entry is None:
            return None, "This style is no longer in your styles."
        renamed_only = entry["id"] == style_id
        if not renamed_only:
            earlier = [entry["id"], *entry["earlier"]]
            entry["earlier"] = [old for old in dict.fromkeys(earlier) if old != style_id][:MAX_EARLIER]
        entry.update(name=name, base=base, look=styles.clean_look(look), id=style_id, updated=now)
        try:
            _write(entries)
        except OSError as error:
            return None, f"Could not save the style: {error}"
        if renamed_only:
            return entry, f"Renamed to “{name}”. Nothing needs to be made again."
        return entry, (f"“{name}” saved as a new version. Films and characters that use the earlier version keep it; "
                       f"choose “{name}” for them again to use the new one (the scenes already made then show "
                       "Outdated).")


def delete(key: str) -> Tuple[bool, str]:
    with _LOCK:
        try:
            entries = _current()
        except OSError as error:
            return False, f"Could not read your styles: {error}"
        entry = next((item for item in entries if item["key"] == key), None)
        if entry is None:
            return False, "This style is no longer in your styles."
        try:
            _write([item for item in entries if item["key"] != key])
        except OSError as error:
            return False, f"Could not delete the style: {error}"
        return True, f"“{entry['name']}” is no longer in your styles. Films and characters that use it keep their look."
