# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Contsy Stories: animated series, their bible and their story memory (docs/CONTSY_STORIES_ARCHITECTURE.md).

The structure is Series -> Seasons -> Episodes -> Scenes -> Takes. A series and its episodes are stored here, in the
state folder (config.STATE_ROOT):

    series/<series id>/series.json             the bible: world, rules, tone, style, format, cast, places and props,
                                               relationships, the current state of the story and its facts
    series/<series id>/episodes/<episode>.json one episode: its request, status, Timeline project, agent run, chat
                                               and the summary of what happened in it

Scenes and takes are not copied: an episode points at its Timeline project (app/timeline.py), whose scenes and takes
are the real ones the editor, the agent and the export use (episode_scenes reads them).

Story memory: every episode keeps a summary of what happened, the series keeps facts (who met whom, who lost what)
and a "story so far" state. story_context() gives the agent the bible plus the memory of every earlier episode when
it writes the next one. The agent saves the memory after an episode (agent_tools_series.save_episode_memory); when it
did not, refresh_episode takes the run's own final summary and marks it "auto". The owner can correct every summary,
fact and the story state by hand ("owner" source); a corrected text is never overwritten by the agent.

Characters of a series are design sheets (colors, silhouette, outfit, marks, personality, voice) linked by name to the
library character the creator or the agent made (app/characters.py). Their stable id is the sheet's id; the library slug
is filled in when a character of that name exists (sync_links).
"""
from __future__ import annotations

import json
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional

from . import config, styles
from .persistence import atomic_text

DIR = config.STATE_ROOT / "series"
_LOCK = threading.RLock()
_ID = re.compile(r"[0-9a-f]{12}")
MAX_NAME = 60
MAX_TEXT = 4000
MAX_SHORT = 400
MAX_CHARACTERS = 100
MAX_PLACES = 200
MAX_FACTS = 300
MAX_SUMMARY = 3000
CONTEXT_CHARS = 14000  # the most series context one agent request gets (older episodes are shortened first)

FORMATS = {"16:9": "Landscape 16:9", "9:16": "Portrait 9:16"}   # YouTube / YouTube Shorts -> Timeline aspect
NARRATIONS = ("dialogue", "narrator", "both")
AUDIENCES = ("kids", "family", "teens and adults")
LANGUAGES = ("English",)  # the speech engine speaks English only today (VoxCPM2 is driven in English by the studio)
EPISODE_STATES = ("draft", "writing", "storyboard", "approved", "producing", "done", "failed")
MIN_EPISODE_SECONDS, MAX_EPISODE_SECONDS = 15, 900
# The longest episode one production run is tested for. Longer ones are allowed but built in chapters (see
# chapters()): each chapter is its own Timeline project, so a failure costs one chapter, not the whole film.
CHAPTER_SECONDS = 180


# --- the style catalog of the product ---------------------------------------------------------------------------
# Only animated looks. "ready": an engine preset whose path is tested (app/styles.py: its own LoRAs, AniSora experts for
# drawn looks); "experimental": a custom style over a preset (styles.custom_id), which works but is not tuned.
_LOOKS = {
    "cartoon_2d": ("2D Cartoon", styles.CARTOON, "", "ready",
                   "Bold outlines, flat bright colors, squash and stretch."),
    "cartoon_3d": ("Stylized 3D Cartoon", styles.ANIMATION_3D, "", "ready",
                   "Soft stylized 3D characters with rounded shapes, like a family animated feature."),
    "anime": ("Anime", styles.ANIME, "", "ready", "Clean line art and cel shading."),
    "storybook": ("Storybook Illustration", styles.CARTOON,
                  "storybook illustration, soft watercolor textures, gentle pencil lines, warm muted palette",
                  "experimental", "Soft watercolor picture-book look."),
    "clay": ("Clay Animation", styles.ANIMATION_3D,
             "clay animation, handmade plasticine characters, visible fingerprints, stop motion look",
             "experimental", "Handmade plasticine, stop-motion feel."),
    "paper": ("Paper Cutout", styles.CARTOON,
              "paper cutout animation, layered colored paper shapes, flat cut edges, subtle paper texture",
              "experimental", "Layered colored paper shapes."),
    "comic": ("Comic Animation", styles.CARTOON,
              "comic book animation, bold ink lines, halftone shading, vivid flat colors",
              "experimental", "Bold ink lines and halftone shading."),
}
DEFAULT_LOOK = "cartoon_2d"


def looks() -> List[dict]:
    """The animation styles a series can choose: {"key", "label", "status", "about"}, the ready ones first."""
    rows = [{"key": key, "label": label, "status": status, "about": about}
            for key, (label, _, _, status, about) in _LOOKS.items()]
    return sorted(rows, key=lambda row: row["status"] != "ready")


def look_choices() -> List[tuple]:
    return [(row["label"] + ("" if row["status"] == "ready" else " (experimental)"), row["key"]) for row in looks()]


def engine_style(look_key: str) -> str:
    """The app/styles.py id a series look draws with (a preset, or a custom style over one). Never Realistic."""
    label, base, words, _, _ = _LOOKS.get(look_key) or _LOOKS[DEFAULT_LOOK]
    return styles.custom_id(base, words) if words else base


def look_label(look_key: str) -> str:
    return (_LOOKS.get(look_key) or _LOOKS[DEFAULT_LOOK])[0]


# --- storage ----------------------------------------------------------------------------------------------------

class SeriesError(ValueError):
    """A series or episode cannot be saved or found; the message is for the owner."""


def _new_id() -> str:
    return secrets.token_hex(6)


def valid_id(value) -> bool:
    return isinstance(value, str) and bool(_ID.fullmatch(value))


def _folder(series_id: str) -> Path:
    if not valid_id(series_id):
        raise SeriesError("This series does not exist.")
    return Path(DIR) / series_id


def _read(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write(path: Path, data: dict) -> None:
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=1))


def _clip(value, limit: int) -> str:
    text = " ".join(str(value or "").split()) if limit <= MAX_SHORT else str(value or "").strip()
    return text[:limit]


def _check_name(name: str) -> str:
    name = _clip(name, MAX_NAME)
    if len(name) < 2:
        raise SeriesError("Give the series a name of at least 2 characters.")
    return name


def _safe_text(*texts: str) -> None:
    """The content rules of the studio apply to the bible as to any prompt (app/llm.py)."""
    from . import llm
    if llm.blocked(*texts):
        raise SeriesError(llm.BLOCK_MSG)


# --- series --------------------------------------------------------------------------------------------------------

def all_series() -> List[dict]:
    """Every saved series, the latest changed first."""
    found = []
    try:
        folders = [p for p in Path(DIR).iterdir() if p.is_dir() and valid_id(p.name)]
    except OSError:
        return []
    for folder in folders:
        data = _read(folder / "series.json")
        if data and data.get("id") == folder.name:
            found.append(_normal(data))
    return sorted(found, key=lambda s: -float(s.get("updated") or 0))


def get(series_id: str) -> Optional[dict]:
    if not valid_id(series_id):
        return None
    data = _read(_folder(series_id) / "series.json")
    return _normal(data) if data and data.get("id") == series_id else None


def _normal(data: dict) -> dict:
    """A stored series with every field present (older or hand-edited files load too)."""
    out = {"id": data["id"], "name": str(data.get("name") or "Untitled series"),
           "logline": str(data.get("logline") or ""), "world": str(data.get("world") or ""),
           "rules": str(data.get("rules") or ""), "tone": str(data.get("tone") or ""),
           "narration_style": str(data.get("narration_style") or ""),
           "look": data.get("look") if data.get("look") in _LOOKS else DEFAULT_LOOK,
           "format": data.get("format") if data.get("format") in FORMATS else "16:9",
           "narration": data.get("narration") if data.get("narration") in NARRATIONS else "dialogue",
           "audience": data.get("audience") if data.get("audience") in AUDIENCES else "family",
           "language": data.get("language") if data.get("language") in LANGUAGES else LANGUAGES[0],
           "narrator_voice": str(data.get("narrator_voice") or ""),
           "characters": [c for c in data.get("characters") or [] if isinstance(c, dict) and c.get("id")],
           "places": [p for p in data.get("places") or [] if isinstance(p, dict) and p.get("id")],
           "relationships": [r for r in data.get("relationships") or [] if isinstance(r, dict)],
           "story_state": str(data.get("story_state") or ""),
           "story_state_source": str(data.get("story_state_source") or ""),
           "facts": [f for f in data.get("facts") or [] if isinstance(f, dict) and f.get("id")],
           "seasons": [s for s in data.get("seasons") or [] if isinstance(s, dict)] or [{"number": 1, "title": ""}],
           "created": float(data.get("created") or 0), "updated": float(data.get("updated") or 0)}
    return out


def _save(series: dict) -> dict:
    series = dict(series)
    series["updated"] = time.time()
    _write(_folder(series["id"]) / "series.json", series)
    return series


def create(name: str, *, look: str = DEFAULT_LOOK, logline: str = "", fmt: str = "16:9",
           narration: str = "dialogue", audience: str = "family") -> dict:
    """A new series from its name, animation style and first choices; the rest of the bible is filled in with
    update."""
    name = _check_name(name)
    _safe_text(name, logline)
    if look not in _LOOKS:
        raise SeriesError("Choose an animation style.")
    with _LOCK:
        now = time.time()
        series = _normal({"id": _new_id(), "name": name, "logline": _clip(logline, MAX_TEXT), "look": look,
                          "format": fmt, "narration": narration, "audience": audience, "created": now})
        return _save(series)


BIBLE_FIELDS = {"name": MAX_NAME, "logline": MAX_TEXT, "world": MAX_TEXT, "rules": MAX_TEXT, "tone": MAX_SHORT,
                "narration_style": MAX_SHORT, "narrator_voice": 80}
CHOICE_FIELDS = {"look": tuple(_LOOKS), "format": tuple(FORMATS), "narration": NARRATIONS, "audience": AUDIENCES,
                 "language": LANGUAGES}


def update(series_id: str, **fields) -> dict:
    """Change bible fields (texts and choices); unknown fields are refused."""
    with _LOCK:
        series = get(series_id)
        if series is None:
            raise SeriesError("This series does not exist.")
        for key, value in fields.items():
            if key in BIBLE_FIELDS:
                value = _check_name(value) if key == "name" else _clip(value, BIBLE_FIELDS[key])
            elif key in CHOICE_FIELDS:
                if value not in CHOICE_FIELDS[key]:
                    raise SeriesError(f"Choose a valid {key}.")
            elif key == "story_state":
                value = _clip(value, MAX_TEXT)
                series["story_state_source"] = "owner"
            else:
                raise SeriesError(f"Unknown series field {key}.")
            series[key] = value
        _safe_text(*(str(series.get(k) or "") for k in BIBLE_FIELDS), series["story_state"])
        return _save(series)


def delete(series_id: str) -> None:
    """Set the series (bible and episodes, not its Timeline projects, characters or films) aside in
    series/.deleted/, from where it can be restored by moving its folder back; portable, unlike the Windows
    Recycle Bin."""
    folder = _folder(series_id)
    with _LOCK:
        if not folder.exists():
            raise SeriesError("This series does not exist.")
        _set_aside(folder)


def _set_aside(path: Path) -> None:
    target = Path(DIR) / ".deleted" / f"{int(time.time())}-{path.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    path.replace(target)


# --- characters (design sheets) ----------------------------------------------------------------------------------

CHARACTER_FIELDS = {"name": MAX_NAME, "role": MAX_SHORT, "kind": 20, "description": MAX_TEXT,
                    "personality": MAX_TEXT, "colors": MAX_SHORT, "silhouette": MAX_SHORT, "outfit": MAX_SHORT,
                    "marks": MAX_SHORT, "voice": 120, "slug": 80}


def save_character(series_id: str, sheet: dict, char_id: str = "") -> dict:
    """Add a character sheet (char_id "") or change one; returns the sheet. A name is unique in its series."""
    with _LOCK:
        series = get(series_id)
        if series is None:
            raise SeriesError("This series does not exist.")
        sheet = {key: _clip(sheet.get(key), limit) for key, limit in CHARACTER_FIELDS.items() if key in sheet}
        cast = series["characters"]
        current = next((c for c in cast if c["id"] == char_id), None) if char_id else None
        if char_id and current is None:
            raise SeriesError("This character is not in the series.")
        merged = {**(current or {}), **sheet}
        name = _clip(merged.get("name"), MAX_NAME)
        if len(name) < 1:
            raise SeriesError("Give the character a name.")
        if any(c["name"].casefold() == name.casefold() and c["id"] != char_id for c in cast):
            raise SeriesError(f"{name} is already a character of this series. Edit that one instead.")
        if merged.get("kind") not in ("creature", "person"):
            merged["kind"] = "creature"
        _safe_text(*(str(merged.get(k) or "") for k in CHARACTER_FIELDS))
        from . import characters, llm
        if merged["kind"] == "person" and llm.refers_to_minor(merged.get("description", ""), merged.get("role", "")):
            raise SeriesError("Characters are adults, or fully grown animals and creatures (see the content rules "
                              "in docs/CONTSY_STORIES_PRODUCT.md).")
        if current is None:
            if len(cast) >= MAX_CHARACTERS:
                raise SeriesError(f"A series has at most {MAX_CHARACTERS} characters.")
            merged["id"] = "c" + _new_id()
            merged["created"] = time.time()
            cast.append(merged)
        else:
            cast[cast.index(current)] = merged
        merged["name"] = name
        if merged.get("slug") and characters.get(merged["slug"]) is None:
            merged["slug"] = ""
        _save(series)
        return merged


def set_outfits(series_id: str, char_id: str, outfits: Iterable[str]) -> dict:
    """The character's optional outfit variants (each a short description)."""
    with _LOCK:
        series = get(series_id)
        if series is None:
            raise SeriesError("This series does not exist.")
        sheet = next((c for c in series["characters"] if c["id"] == char_id), None)
        if sheet is None:
            raise SeriesError("This character is not in the series.")
        sheet["outfits"] = [_clip(o, MAX_SHORT) for o in outfits if str(o or "").strip()][:12]
        _safe_text(*sheet["outfits"])
        _save(series)
        return sheet


def remove_character(series_id: str, char_id: str) -> None:
    with _LOCK:
        series = get(series_id)
        if series is None:
            raise SeriesError("This series does not exist.")
        series["characters"] = [c for c in series["characters"] if c["id"] != char_id]
        series["relationships"] = [r for r in series["relationships"] if char_id not in (r.get("a"), r.get("b"))]
        _save(series)


def character_look(sheet: dict) -> str:
    """The look text a library character is created from: its description and the parts that keep it the same."""
    parts = [sheet.get("description", "")]
    for key, label in (("colors", "colors"), ("silhouette", "silhouette"), ("outfit", "wears"),
                       ("marks", "distinctive marks")):
        if sheet.get(key):
            parts.append(f"{label}: {sheet[key]}")
    return "; ".join(p for p in parts if p)[:600]


def sync_links(series: dict) -> dict:
    """Link each sheet and place to the library character or item of the same name (made since by the creator or the
    agent), and drop links whose library entry is gone. Saved only when something changed."""
    from . import characters, items
    library = {c["name"].casefold(): c["slug"] for c in characters.all_characters() if c.get("name")}
    things = {i["name"].casefold(): i["slug"] for i in items.all_items() if i.get("name")}
    changed = False
    for sheet, pool, getter in ([(c, library, characters.get) for c in series["characters"]]
                                + [(p, things, items.get) for p in series["places"]]):
        slug = sheet.get("slug") or ""
        if slug and getter(slug) is None:
            sheet["slug"], changed = "", True
            slug = ""
        if not slug and sheet.get("name", "").casefold() in pool:
            sheet["slug"], changed = pool[sheet["name"].casefold()], True
    if changed:
        with _LOCK:
            series = _save(series)
    return series


# --- places and props --------------------------------------------------------------------------------------------

def save_place(series_id: str, name: str, description: str, kind: str = "location", place_id: str = "") -> dict:
    with _LOCK:
        series = get(series_id)
        if series is None:
            raise SeriesError("This series does not exist.")
        name = _clip(name, MAX_NAME)
        if not name:
            raise SeriesError("Give the place or prop a name.")
        if kind not in ("location", "prop"):
            raise SeriesError("A place is a location or a prop.")
        _safe_text(name, description)
        pool = series["places"]
        if any(p["name"].casefold() == name.casefold() and p["id"] != place_id for p in pool):
            raise SeriesError(f"{name} is already in this series.")
        current = next((p for p in pool if p["id"] == place_id), None) if place_id else None
        if place_id and current is None:
            raise SeriesError("This place is not in the series.")
        entry = {**(current or {"id": "p" + _new_id(), "slug": ""}), "name": name, "kind": kind,
                 "description": _clip(description, MAX_TEXT)}
        if current is None:
            if len(pool) >= MAX_PLACES:
                raise SeriesError(f"A series has at most {MAX_PLACES} places and props.")
            pool.append(entry)
        else:
            pool[pool.index(current)] = entry
        _save(series)
        return entry


def remove_place(series_id: str, place_id: str) -> None:
    with _LOCK:
        series = get(series_id)
        if series is None:
            raise SeriesError("This series does not exist.")
        series["places"] = [p for p in series["places"] if p["id"] != place_id]
        _save(series)


# --- relationships, facts and the story state --------------------------------------------------------------------

def set_relationship(series_id: str, a: str, b: str, text: str) -> None:
    """How two characters (sheet ids) stand to each other; empty text removes it."""
    with _LOCK:
        series = get(series_id)
        if series is None:
            raise SeriesError("This series does not exist.")
        ids = {c["id"] for c in series["characters"]}
        if a not in ids or b not in ids or a == b:
            raise SeriesError("Choose two different characters of the series.")
        pair = {a, b}
        rows = [r for r in series["relationships"] if {r.get("a"), r.get("b")} != pair]
        text = _clip(text, MAX_SHORT)
        _safe_text(text)
        if text:
            rows.append({"a": a, "b": b, "text": text})
        series["relationships"] = rows
        _save(series)


def add_fact(series_id: str, text: str, *, episode: str = "", source: str = "owner") -> dict:
    """A fact of the story that later episodes must respect ("Pip lost the silver key in the river")."""
    with _LOCK:
        series = get(series_id)
        if series is None:
            raise SeriesError("This series does not exist.")
        text = _clip(text, MAX_SHORT)
        if not text:
            raise SeriesError("Write the fact.")
        _safe_text(text)
        if any(f["text"].casefold() == text.casefold() and f.get("active", True) for f in series["facts"]):
            return next(f for f in series["facts"] if f["text"].casefold() == text.casefold())
        fact = {"id": "f" + _new_id(), "text": text, "episode": episode if valid_id(episode) else "",
                "source": "owner" if source == "owner" else "auto", "active": True, "created": time.time()}
        series["facts"].append(fact)
        series["facts"] = series["facts"][-MAX_FACTS:]
        _save(series)
        return fact


def edit_fact(series_id: str, fact_id: str, text: Optional[str] = None, active: Optional[bool] = None) -> None:
    """The owner corrects a fact (it becomes the owner's) or retires it (it stays, inactive, out of the context)."""
    with _LOCK:
        series = get(series_id)
        if series is None:
            raise SeriesError("This series does not exist.")
        fact = next((f for f in series["facts"] if f["id"] == fact_id), None)
        if fact is None:
            raise SeriesError("This fact is not in the series.")
        if text is not None:
            text = _clip(text, MAX_SHORT)
            if not text:
                raise SeriesError("Write the fact, or retire it.")
            _safe_text(text)
            fact["text"], fact["source"] = text, "owner"
        if active is not None:
            fact["active"] = bool(active)
        _save(series)


def remove_fact(series_id: str, fact_id: str) -> None:
    with _LOCK:
        series = get(series_id)
        if series is None:
            raise SeriesError("This series does not exist.")
        series["facts"] = [f for f in series["facts"] if f["id"] != fact_id]
        _save(series)


# --- episodes -----------------------------------------------------------------------------------------------------

def _episode_path(series_id: str, episode_id: str) -> Path:
    if not valid_id(episode_id):
        raise SeriesError("This episode does not exist.")
    return _folder(series_id) / "episodes" / f"{episode_id}.json"


def episodes(series_id: str) -> List[dict]:
    """The series' episodes in story order (season, number)."""
    try:
        files = list((_folder(series_id) / "episodes").glob("*.json"))
    except (OSError, SeriesError):
        return []
    found = []
    for path in files:
        data = _read(path)
        if data and data.get("id") == path.stem and data.get("series") == series_id:
            found.append(_episode_normal(data))
    return sorted(found, key=lambda e: (e["season"], e["number"], e["created"]))


def get_episode(series_id: str, episode_id: str) -> Optional[dict]:
    try:
        data = _read(_episode_path(series_id, episode_id))
    except SeriesError:
        return None
    return _episode_normal(data) if data and data.get("id") == episode_id and data.get("series") == series_id else None


def _episode_normal(data: dict) -> dict:
    return {"id": data["id"], "series": data["series"], "season": int(data.get("season") or 1),
            "number": int(data.get("number") or 1), "title": str(data.get("title") or ""),
            "idea": str(data.get("idea") or ""),
            "seconds": int(data.get("seconds") or 60),
            "format": data.get("format") if data.get("format") in FORMATS else "16:9",
            "narration": data.get("narration") if data.get("narration") in NARRATIONS else "dialogue",
            "language": data.get("language") if data.get("language") in LANGUAGES else LANGUAGES[0],
            "status": data.get("status") if data.get("status") in EPISODE_STATES else "draft",
            "projects": [p for p in data.get("projects") or [] if isinstance(p, str)],
            "run": str(data.get("run") or ""),
            "chat": [m for m in data.get("chat") or [] if isinstance(m, dict)][-200:],
            "summary": str(data.get("summary") or ""), "summary_source": str(data.get("summary_source") or ""),
            "created": float(data.get("created") or 0), "updated": float(data.get("updated") or 0)}


def _save_episode(episode: dict) -> dict:
    episode = dict(episode)
    episode["updated"] = time.time()
    _write(_episode_path(episode["series"], episode["id"]), episode)
    # the series moves to the top of the list when an episode changes
    series = get(episode["series"])
    if series is not None:
        _save(series)
    return episode


def create_episode(series_id: str, idea: str, *, title: str = "", seconds: int = 60, fmt: str = "",
                   narration: str = "", season: int = 0) -> dict:
    """A new episode at the end of its season (the series' latest season by default)."""
    with _LOCK:
        series = get(series_id)
        if series is None:
            raise SeriesError("This series does not exist.")
        idea = _clip(idea, MAX_TEXT)
        if len(idea) < 5:
            raise SeriesError("Describe what happens in the episode (a sentence or two is enough).")
        _safe_text(idea, title)
        try:
            seconds = int(seconds)
        except (TypeError, ValueError):
            raise SeriesError("Give the episode length in seconds.") from None
        if not MIN_EPISODE_SECONDS <= seconds <= MAX_EPISODE_SECONDS:
            raise SeriesError(f"An episode lasts {MIN_EPISODE_SECONDS} s to {MAX_EPISODE_SECONDS // 60} min.")
        fmt = fmt or series["format"]
        narration = narration or series["narration"]
        if fmt not in FORMATS or narration not in NARRATIONS:
            raise SeriesError("Choose the format and the narration.")
        season = int(season or max(s.get("number", 1) for s in series["seasons"]))
        if season not in {s.get("number") for s in series["seasons"]}:
            series["seasons"].append({"number": season, "title": ""})
            _save(series)
        number = 1 + max((e["number"] for e in episodes(series_id) if e["season"] == season), default=0)
        now = time.time()
        episode = _episode_normal({"id": _new_id(), "series": series_id, "season": season, "number": number,
                                   "title": _clip(title, MAX_NAME) or f"Episode {number}", "idea": idea,
                                   "seconds": seconds, "format": fmt, "narration": narration, "status": "draft",
                                   "created": now})
        return _save_episode(episode)


EPISODE_FIELDS = {"title", "idea", "seconds", "format", "narration", "status", "projects", "run", "summary"}


def update_episode(series_id: str, episode_id: str, **fields) -> dict:
    with _LOCK:
        episode = get_episode(series_id, episode_id)
        if episode is None:
            raise SeriesError("This episode does not exist.")
        for key, value in fields.items():
            if key not in EPISODE_FIELDS:
                raise SeriesError(f"Unknown episode field {key}.")
            if key == "status" and value not in EPISODE_STATES:
                raise SeriesError("Unknown episode status.")
            if key in ("title",):
                value = _clip(value, MAX_NAME)
            if key in ("idea",):
                value = _clip(value, MAX_TEXT)
            episode[key] = value
        _safe_text(episode["title"], episode["idea"], episode["summary"])
        return _save_episode(episode)


def set_summary(series_id: str, episode_id: str, summary: str, *, source: str = "owner") -> dict:
    """The memory of what happened in the episode. An owner's summary is never replaced by an automatic one."""
    with _LOCK:
        episode = get_episode(series_id, episode_id)
        if episode is None:
            raise SeriesError("This episode does not exist.")
        if source != "owner" and episode["summary_source"] == "owner":
            return episode
        summary = _clip(summary, MAX_SUMMARY)
        _safe_text(summary)
        episode["summary"], episode["summary_source"] = summary, ("owner" if source == "owner" else source)
        return _save_episode(episode)


def add_chat(series_id: str, episode_id: str, role: str, text: str) -> dict:
    """One message of the episode's conversation with the agent (what the page shows; the run keeps the real one)."""
    with _LOCK:
        episode = get_episode(series_id, episode_id)
        if episode is None:
            raise SeriesError("This episode does not exist.")
        episode["chat"].append({"role": "owner" if role == "owner" else "agent", "text": _clip(text, MAX_TEXT),
                                "time": time.time()})
        return _save_episode(episode)


def delete_episode(series_id: str, episode_id: str) -> None:
    path = _episode_path(series_id, episode_id)
    with _LOCK:
        if not path.exists():
            raise SeriesError("This episode does not exist.")
        _set_aside(path)


def find_episode_of_run(run_id: str) -> Optional[dict]:
    """The episode an agent run works on (each series folder is small, so a scan is cheap)."""
    for series in all_series():
        for episode in episodes(series["id"]):
            if episode["run"] == run_id:
                return episode
    return None


def chapters(seconds: int) -> List[int]:
    """How a long episode is split into chapters of at most CHAPTER_SECONDS (each one Timeline project)."""
    seconds = max(MIN_EPISODE_SECONDS, int(seconds))
    count = -(-seconds // CHAPTER_SECONDS)
    base = seconds // count
    return [base + (1 if i < seconds % count else 0) for i in range(count)]


# --- scenes and takes (read from the Timeline) ---------------------------------------------------------------------

def episode_scenes(episode: dict) -> List[dict]:
    """Every scene of the episode across its chapters: {"chapter", "project", "index", "prompt", "seconds", "cast",
    "lines", "takes", "has_clip", "outdated"}. Missing projects are left out."""
    from . import timeline
    out = []
    for chapter, project_id in enumerate(episode["projects"], start=1):
        try:
            project = timeline.load(project_id)
        except (ValueError, OSError):
            continue
        try:
            stale = set(timeline.outdated(project))
        except (ValueError, OSError):
            stale = set()
        for index, scene in enumerate(project.scenes):
            try:
                takes = timeline.takes(project, index)
            except (ValueError, OSError):
                takes = []
            out.append({"chapter": chapter, "project": project_id, "index": index, "prompt": scene.prompt,
                        "seconds": float(scene.seconds), "cast": list(scene.cast or ()),
                        "lines": [{"speaker": cue.speaker, "text": cue.text} for cue in scene.lines],
                        "takes": takes, "has_clip": bool(takes), "outdated": index in stale})
    return out


# --- story memory for the agent ------------------------------------------------------------------------------------

def _sheet_text(sheet: dict) -> str:
    bits = [f"- {sheet['name']} [{sheet['id']}]"
            + (f" (library character slug: {sheet['slug']})" if sheet.get("slug") else " (NOT CREATED YET)")]
    for key in ("role", "kind", "description", "personality", "colors", "silhouette", "outfit", "marks", "voice"):
        if sheet.get(key):
            bits.append(f"  {key}: {sheet[key]}")
    if sheet.get("outfits"):
        bits.append("  outfit variants: " + " | ".join(sheet["outfits"]))
    return "\n".join(bits)


def story_context(series_id: str, episode_id: str = "") -> str:
    """The series bible and the story so far, as the agent reads it before (and while) it writes an episode. The
    episodes after `episode_id` are not part of it; the oldest summaries are shortened first to fit CONTEXT_CHARS."""
    series = get(series_id)
    if series is None:
        raise SeriesError("This series does not exist.")
    series = sync_links(series)
    names = {c["id"]: c["name"] for c in series["characters"]}
    head = [f"SERIES: {series['name']}",
            f"Animation style: {look_label(series['look'])} (style id {engine_style(series['look'])}; set it once on "
            "the project, do not repeat style words in every scene).",
            f"Audience: {series['audience']}. Default format: {series['format']}. Narration: {series['narration']}.",
            ]
    for key, label in (("logline", "Premise"), ("world", "World"), ("rules", "Rules of the world"),
                       ("tone", "Tone"), ("narration_style", "Storytelling")):
        if series[key]:
            head.append(f"{label}: {series[key]}")
    if series["narrator_voice"]:
        head.append(f"Narrator voice: {series['narrator_voice']}")
    head.append("CAST (fixed: reuse these characters, never invent a new design for them):")
    head += [_sheet_text(c) for c in series["characters"]] or ["- (no characters yet)"]
    if series["places"]:
        head.append("LOCATIONS AND PROPS:")
        head += [f"- {p['name']} ({p['kind']})" + (f" [library item {p['slug']}]" if p.get("slug") else "")
                 + (f": {p['description']}" if p.get("description") else "") for p in series["places"]]
    if series["relationships"]:
        head.append("RELATIONSHIPS:")
        head += [f"- {names.get(r['a'], '?')} & {names.get(r['b'], '?')}: {r['text']}" for r in series["relationships"]]
    facts = [f for f in series["facts"] if f.get("active", True)]
    if facts:
        head.append("ESTABLISHED FACTS (must stay true unless an episode changes them on purpose):")
        head += [f"- {f['text']}" for f in facts]
    if series["story_state"]:
        head.append("STORY SO FAR: " + series["story_state"])
    current = get_episode(series_id, episode_id) if episode_id else None
    earlier = [e for e in episodes(series_id)
               if e["summary"] and (current is None or (e["season"], e["number"]) < (current["season"], current["number"]))]
    recaps = [f"S{e['season']}E{e['number']} {e['title']}: {e['summary']}" for e in earlier]
    text = "\n".join(head)
    budget = CONTEXT_CHARS - len(text) - 200
    kept: List[str] = []
    for i, recap in enumerate(reversed(recaps)):  # the latest episodes in full, older ones shortened, then dropped
        if i >= 3:
            recap = recap[:300] + ("…" if len(recap) > 300 else "")
        if budget - len(recap) < 0:
            kept.append(f"({len(recaps) - i} older episodes left out for length)")
            break
        budget -= len(recap) + 1
        kept.append(recap)
    if kept:
        text += "\nEARLIER EPISODES (oldest first):\n" + "\n".join(reversed(kept))
    return text


def episode_brief(series_id: str, episode_id: str) -> str:
    """What the agent run of an episode is asked to do: the request, the series context and the rules of the
    product. The agent saves the memory of the episode with save_episode_memory before finish_run."""
    episode = get_episode(series_id, episode_id)
    if episode is None:
        raise SeriesError("This episode does not exist.")
    parts = chapters(episode["seconds"])
    narration = {"dialogue": "the characters speak to each other (dialogue lines with their own voices)",
                 "narrator": "a narrator tells the story; characters do not need visible lip movement",
                 "both": "a narrator and short dialogue lines of the characters"}[episode["narration"]]
    lines = [f"Make episode S{episode['season']}E{episode['number']} \"{episode['title']}\" of the animated series "
             "below.",
             f"Episode idea from the owner: {episode['idea']}",
             f"Length: about {episode['seconds']} seconds. Format: {episode['format']} "
             f"({FORMATS[episode['format']]}). Language of speech and on-screen text: {episode['language']}. "
             f"Narration: {narration}.",
             ]
    if len(parts) > 1:
        lines.append(f"This episode is long: build it as {len(parts)} chapters of about "
                     + ", ".join(f"{p} s" for p in parts) + ", each its own Timeline project titled "
                     f"\"{episode['title']} — Chapter N\", made and checked one chapter at a time.")
    lines += ["Use the series' animation style for every project, its cast for every character and its locations for "
              "every place. Characters marked NOT CREATED YET are created with create_character from their sheet "
              "(look = description, colors, silhouette, outfit and marks; kind creature unless the sheet says person) "
              "before they appear. Keep each character's colors, outfit and marks identical in every scene text.",
              f"Title card: set the episode title \"{episode['title']}\" as an exact text overlay (set_overlays) over "
              "the first 3 seconds; never ask the video model to draw it. Turn subtitles on when the episode has "
              "dialogue.",
              "Before finish_run, call save_episode_memory with a short summary of what happened, the new facts "
              "later episodes must respect, and the story state.",
              "", story_context(series_id, episode_id)]
    return "\n".join(lines)


# --- run state -> episode state -----------------------------------------------------------------------------------

def refresh_episode(series_id: str, episode_id: str) -> Optional[dict]:
    """Bring the episode's status up to date with its agent run (the page calls it while it shows the episode), link
    the projects the run made, and keep the run's final summary as the episode memory when the agent saved none."""
    from . import agent_store
    episode = get_episode(series_id, episode_id)
    if episode is None or not episode["run"] or not agent_store.valid_id(episode["run"]):
        return episode
    record = agent_store.RunStore(episode["run"]).read() or {}
    state = record.get("state")
    mode = (record.get("settings") or {}).get("mode")
    changes: dict = {}
    projects = list(episode["projects"])
    for key in (record.get("resources") or {}).get("created") or []:
        kind, _, project_id = str(key).partition(":")
        if kind == "project" and project_id not in projects:
            projects.append(project_id)
    if projects != episode["projects"]:
        changes["projects"] = projects
    status = episode["status"]
    if state in ("running", "waiting"):
        status = "producing" if mode == "produce" else "writing"
    elif state == "completed":
        status = "done" if mode == "produce" else "storyboard"
    elif state in ("failed", "stopped", "budget_stopped", "interrupted"):
        status = "failed" if status not in ("done",) else status
    if status != episode["status"]:
        changes["status"] = status
    if changes:
        episode = update_episode(series_id, episode_id, **changes)
    outcome = record.get("outcome") or {}
    if state == "completed" and mode == "produce" and not episode["summary"] and outcome.get("summary"):
        episode = set_summary(series_id, episode_id, str(outcome["summary"]), source="auto")
    return episode


# --- continuity warnings -------------------------------------------------------------------------------------------

def warnings(series_id: str, episode_id: str) -> List[str]:
    """What the owner should look at before publishing the episode. Deterministic checks of the plan and the files
    (who is in a scene, voices, missing or outdated shots, the export); a picture's look is judged by the agent's
    review (its findings are listed too), which is a few frames, not a guarantee."""
    from . import agent_store, characters, timeline
    series = get(series_id)
    episode = get_episode(series_id, episode_id)
    if series is None or episode is None:
        return ["This episode does not exist."]
    series = sync_links(series)
    out: List[str] = []
    for sheet in series["characters"]:
        if not sheet.get("slug"):
            out.append(f"{sheet['name']} is not created yet: the series has only a design sheet.")
    cast_slugs = {c["slug"]: c for c in series["characters"] if c.get("slug")}
    if not episode["projects"]:
        out.append("No storyboard yet.")
        return out
    total = 0.0
    for chapter, project_id in enumerate(episode["projects"], start=1):
        try:
            project = timeline.load(project_id)
        except (ValueError, OSError):
            out.append(f"Chapter {chapter}: its Timeline project is missing.")
            continue
        where = f"Chapter {chapter}, scene" if len(episode["projects"]) > 1 else "Scene"
        if styles.of(project.style).id != engine_style(series["look"]):
            out.append(f"Chapter {chapter} is drawn in {styles.label(project.style)}, not the series style "
                       f"{look_label(series['look'])}.")
        if project.aspect != FORMATS[episode["format"]]:
            out.append(f"Chapter {chapter} is {project.aspect}, the episode is {FORMATS[episode['format']]}.")
        try:
            stale = set(timeline.outdated(project))
        except (ValueError, OSError):
            stale = set()
        for index, scene in enumerate(project.scenes):
            total += float(scene.seconds)
            members = set(scene.cast or ()) | {cue.speaker for cue in scene.lines if cue.speaker
                                                and not cue.speaker.startswith("@")}
            for slug in sorted(members):
                if slug not in cast_slugs:
                    ch = characters.get(slug)
                    out.append(f"{where} {index + 1}: {(ch or {}).get('name', slug)} is not part of the series cast "
                               "(an extra character).")
            for cue in scene.lines:
                if cue.speaker.startswith("@") and series["narration"] == "dialogue":
                    out.append(f"{where} {index + 1}: a line is spoken by a library voice, not by a series "
                               "character.")
                    break
            for slug in sorted(members & set(cast_slugs)):
                ch = characters.get(slug) or {}
                sheet = cast_slugs[slug]
                if not ch.get("voice"):
                    out.append(f"{where} {index + 1}: {sheet['name']} has no saved voice.")
                elif sheet.get("voice") and sheet["voice"] not in (ch.get("voice"), "own") \
                        and not str(ch.get("voice")).endswith(sheet["voice"]):
                    out.append(f"{where} {index + 1}: {sheet['name']} speaks with {ch.get('voice')}, the sheet says "
                               f"{sheet['voice']}.")
            try:
                has_take = bool(timeline.takes(project, index))
            except (ValueError, OSError):
                has_take = False
            if not has_take:
                out.append(f"{where} {index + 1}: shot not made yet.")
            elif index in stale:
                out.append(f"{where} {index + 1}: the shot no longer matches its plan (outdated).")
        try:
            if project.scenes and timeline.exported(project) is None:
                out.append(f"Chapter {chapter}: no current export (the film is incomplete or changed since).")
        except (ValueError, OSError, KeyError):
            out.append(f"Chapter {chapter}: the export could not be checked.")
    if total and abs(total - episode["seconds"]) > max(10.0, 0.25 * episode["seconds"]):
        out.append(f"The storyboard lasts {total:.0f} s; the episode was planned for {episode['seconds']} s.")
    if episode["run"] and agent_store.valid_id(episode["run"]):
        outcome = (agent_store.RunStore(episode["run"]).read() or {}).get("outcome") or {}
        for item in (outcome.get("failing") or [])[:10]:
            out.append("Agent review: " + str(item)[:300])
        for item in (outcome.get("owner_should_check") or [])[:10]:
            out.append("Check by eye: " + str(item)[:300])
    if episode["status"] == "done" and not episode["summary"]:
        out.append("The episode has no story memory yet: write a summary so the next episode knows what happened.")
    return out
