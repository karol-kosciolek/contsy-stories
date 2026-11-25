# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Universes: named groups of characters (for example "Family" = Ola, Kuba, Jan).

Selecting a universe in the top bar's Tools panel makes its first ready member the main character, shows
the whole group's results on the Dashboard and in the Gallery, and loads all its members
into "Characters in this video/photo" (app/cast.py). Stored in characters/universes.json
next to the characters; a deleted character leaves every universe (forget). A member whose folder
disappeared without a deletion (an archived character) stays listed until the universe is saved
again: it is left out of `members` and named by `notice`.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import characters, llm
from .persistence import atomic_text

MAX_NAME = 40
_LOCK = threading.RLock()


def _file() -> Path:
    return characters.DIR / "universes.json"


def all_universes() -> List[dict]:
    """Every universe, oldest first (an unreadable file counts as none)."""
    try:
        data = json.loads(_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = data.get("universes", []) if isinstance(data, dict) else []
    out = []
    for u in items:
        if (isinstance(u, dict) and isinstance(u.get("name"), str) and isinstance(u.get("slug"), str)
                and isinstance(u.get("members"), list)):
            out.append({"name": u["name"], "slug": u["slug"],
                        "members": [m for m in u["members"] if isinstance(m, str)],
                        "created": u.get("created", 0)})
    return out


def _write(items: List[dict]) -> None:
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path, json.dumps({"universes": items}, ensure_ascii=False, indent=1))


def get(slug: str) -> Optional[dict]:
    return next((u for u in all_universes() if u["slug"] == slug), None) if slug else None


def members(universe: Optional[dict]) -> List[dict]:
    """The universe's characters that are ready to generate, in the stored order."""
    if not universe:
        return []
    ready = [characters.usable(slug) for slug in universe.get("members", [])]
    return [ch for ch in ready if ch]


def notice(universe: Optional[dict]) -> str:
    """A message naming the universe's members that no longer exist ("" when every one exists)."""
    return characters.missing_message((universe or {}).get("members", []))


def check_name(name: str, slug: str = "") -> str:
    """Why the name cannot be used for a universe ('' = fine); slug - the universe being renamed."""
    name = " ".join((name or "").split())
    if not name:
        return "Enter a universe name."
    if len(name) > MAX_NAME:
        return f"Use at most {MAX_NAME} characters for the universe name."
    if llm.blocked(name):
        return llm.BLOCK_MSG
    if any(u["name"].casefold() == name.casefold() and u["slug"] != slug for u in all_universes()):
        return f"A universe named ‘{name}’ already exists. Choose another name."
    if any(c["name"].casefold() == name.casefold() for c in characters.all_characters()):
        return f"‘{name}’ is a character name. Choose another universe name."
    return ""


def save(name: str, member_slugs: Sequence[str], slug: str = "") -> Tuple[Optional[dict], str]:
    """Create a universe, or update the one with this slug. Returns (universe, message)."""
    name = " ".join((name or "").split())
    with _LOCK:
        error = check_name(name, slug)
        if error:
            return None, error
        known = {c["slug"]: c for c in characters.all_characters()}
        chosen = list(dict.fromkeys(s for s in (member_slugs or []) if s in known))
        if len(chosen) < 2:
            return None, "Pick at least two characters for the universe."
        from .cast import MAX_CAST  # lazy: app/cast.py imports this module

        if len(chosen) > MAX_CAST:
            return None, (f"A universe can have at most {MAX_CAST} characters, the most one "
                          "video or photo can show.")
        items = all_universes()
        current = next((u for u in items if u["slug"] == slug), None) if slug else None
        if current is None:
            base = characters.slugify(name)
            new_slug, i = base, 2
            while any(u["slug"] == new_slug for u in items):
                new_slug, i = f"{base}_{i}", i + 1
            current = {"name": name, "slug": new_slug, "members": chosen, "created": time.time()}
            items.append(current)
        else:
            current.update(name=name, members=chosen)
        _write(items)
    names = ", ".join(known[s]["name"] for s in chosen)
    return current, f"Universe ‘{name}’ saved ({len(chosen)} characters: {names})."


def delete(slug: str) -> Tuple[bool, str]:
    """Remove a universe (its characters stay). Returns (ok, message)."""
    with _LOCK:
        items = all_universes()
        current = next((u for u in items if u["slug"] == slug), None)
        if current is None:
            return False, "Universe not found."
        _write([u for u in items if u["slug"] != slug])
    return True, f"Universe ‘{current['name']}’ deleted. Its characters stay in the library."


def forget(char_slug: str) -> None:
    """A deleted character leaves every universe."""
    with _LOCK:
        items = all_universes()
        if any(char_slug in u["members"] for u in items):
            for u in items:
                u["members"] = [m for m in u["members"] if m != char_slug]
            _write(items)
