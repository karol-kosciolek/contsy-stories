# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Places and items: reference pictures of a place (a TV studio, a kitchen) or an object (a news desk, a red bicycle)
that a film shows again and again, made and kept like characters (app/characters.py).

An item is a name, a kind (PLACE or OBJECT), a description of how it looks, a style (app/styles.py, the style its
photos are drawn in) and 1 to 8 reference photos; the first is the main one. In a Timeline scene the main photo is a
reference image of the scene (app/timeline.py): picked in the scene inspector ("Places and items in this scene"), or
named in the scene's picture text, the way a character's name brings the character in. Nothing is trained.

Storage, one folder per item under items/ (never written before Save):

    items/<slug>/item.json      {"format": 1, "name", "slug", "kind": "place" | "object", "look",
                                 "style" (app/styles.py; absent = "realistic"),
                                 "photos": [{"file", "sha256", "origin"}], "created", "updated"}
    items/<slug>/photos/01.png ... 08.png

A photo whose file no longer matches its sha256 is ignored. Items have no person rules: a description is refused only
when it combines minors with sexual content (llm.blocked, as everywhere), and a photo only when it shows someone who
looks under 18 (`check_photo`).

Reading: get(slug) -> profile {"name", "slug", "kind", "look", "style", "photos" (verified paths, main first),
"created", "updated", "record"} or None; all_items(); usable(slug) (at least one photo); usable_items(); main_photo
(item); identity(item) for clip keys; mentions(text, pool) for names in a picture text; state(item).

The creator screen (Items page) works on a draft under item_drafts/, discarded with Discard, New, Save or when its
browser tab is gone. Every call takes the draft id and returns plain data:

    new_draft() -> id                           an empty draft (a place)
    load_draft(slug) -> id                      a draft holding copies of an item (to edit it)
    draft(id) -> dict                           name, kind, look, style, photo paths and origins, slug, busy
    update_draft(id, name=, kind=, look=, style=)
    enqueue_photos(id, count, keep_own=)        GPU queue job: `count` photos of one place or object
    enqueue_regenerate(id, index)               GPU queue job: a new take of one generated photo
    remove_photo(id, index)
    add_upload(id, path, consent) -> (ok, message)
    save_draft(id, name=, kind=, look=, style=) -> (profile | None, message)
    discard_draft(id)
    delete(slug) -> message                     the item's folder to the Recycle Bin
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import characters, config, dialogue, llm, queue_manager, styles, trash
from .persistence import atomic_text

ProgressCb = Callable[[float, str], None]

DIR = config.STATE_ROOT / "items"
DRAFTS_DIR = config.STATE_ROOT / "item_drafts"
FORMAT = 1
MIN_PHOTOS, MAX_PHOTOS = 1, 8
PLACE, OBJECT = "place", "object"
KINDS = (PLACE, OBJECT)
MAX_NAME, MAX_LOOK = 40, 600
DRAFT_TTL = 24 * 3600  # a draft left behind (a closed browser tab) is removed after a day
PHOTO_KIND = "item_photos"
CONSENT = ("I confirm that I may use these photos, and that any person in them is an adult who agreed to it.")
_ID = re.compile(r"[0-9a-f]{32}")
_REPLACED = re.compile(r"\.replaced_([A-Za-z0-9_-]{1,100})_[0-9a-f]{8}")  # a folder a save set aside (_swap_in)
_PHOTO_FILE = re.compile(r"\d{2}\.png")
_DRAFT_PHOTO = re.compile(r"p_[0-9a-f]{16}\.png")
_SAVE_LOCK = threading.RLock()
_DRAFT_LOCK = threading.RLock()
_digest = characters._digest  # one sha256 memo for every library photo


# ----------------------------- names and folders -----------------------------

# Windows device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9) cannot be folder names: a slug
# equal to one gets an underscore, so an item named "Aux" still gets a usable folder
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def slugify(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.replace("ł", "l").replace("Ł", "L"))
    text = re.sub(r"[^a-z0-9]+", "_", text.encode("ascii", "ignore").decode().lower()).strip("_")
    text = text[:24] or "item"
    return f"{text}_" if text in _RESERVED else text


def _folder(slug: str) -> Path:
    if not isinstance(slug, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", slug):
        raise ValueError("Invalid item identifier.")
    folder = DIR / slug
    if folder.resolve().parent != DIR.resolve():
        raise ValueError("Item folder is outside the item library.")
    return folder


def _file(slug: str) -> Path:
    return _folder(slug) / "item.json"


def _kind(value: object) -> str:
    """A stored kind as one of KINDS: anything but OBJECT (absent, unknown) is a place."""
    return OBJECT if value == OBJECT else PLACE


# ----------------------------- reading items -----------------------------

def _verified(path: Path, sha: object) -> Optional[Path]:
    return path if isinstance(sha, str) and path.is_file() and _digest(path) == sha else None


def _photo_paths(folder: Path, record: dict) -> List[Path]:
    photos = record.get("photos")
    found = []
    for entry in photos[:MAX_PHOTOS] if isinstance(photos, list) else []:
        name = entry.get("file") if isinstance(entry, dict) else None
        if isinstance(name, str) and _PHOTO_FILE.fullmatch(name):
            path = _verified(folder / "photos" / name, entry.get("sha256"))
            if path is not None:
                found.append(path)
    return found


def _profile(slug: str, record: dict) -> dict:
    """The item as the rest of the app sees it; "record" is item.json as stored."""
    return {"name": record["name"], "slug": slug, "kind": _kind(record.get("kind")),
            "look": str(record.get("look") or ""), "style": styles.of(record.get("style")).id,
            "photos": [str(path) for path in _photo_paths(_folder(slug), record)],
            "created": record.get("created", 0), "updated": record.get("updated", record.get("created", 0)),
            "record": record}


def get(slug: str) -> Optional[dict]:
    if not slug:
        return None
    try:
        data = json.loads(_file(slug).read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("slug") != slug or not isinstance(data.get("name"), str)
                or data.get("format") != FORMAT):
            return None
        return _profile(slug, data)
    except (OSError, ValueError):
        return None


def all_items() -> List[dict]:
    if not DIR.exists():
        return []
    found = [get(path.name) for path in DIR.iterdir() if path.is_dir() and not path.name.startswith(".")]
    return sorted((item for item in found if item), key=lambda item: item.get("created", 0))


def usable(slug: str) -> Optional[dict]:
    """The item when a scene can use it (at least one verified photo), else None."""
    item = get(slug)
    return item if item and item.get("photos") else None


def usable_items() -> List[dict]:
    return [item for item in all_items() if item.get("photos")]


def main_photo(item: Optional[dict]) -> Optional[Path]:
    """The item's main (first) photo: what a scene composes the place or object from."""
    photos = (item or {}).get("photos") or []
    return Path(photos[0]) if photos else None


def identity(item: dict) -> dict:
    """What an item contributes to the key of a Timeline clip it is in through its name: its slug, its name and the
    sha256 of the main photo (the only photo a scene uses), never paths or save times."""
    photo = main_photo(item)
    return {"item": item["slug"], "name": item["name"], "photo": _digest(photo) if photo else ""}


def state(item: dict) -> str:
    return "Ready" if item.get("photos") else "Needs photos"


def startup() -> None:
    """At the start of the app: a save cut short by a crash is settled (as characters.startup does), and the drafts
    of the previous run, which no browser tab can reach any more, are removed."""
    if DIR.exists():
        with _SAVE_LOCK:
            for backup in DIR.iterdir():
                match = _REPLACED.fullmatch(backup.name)
                if match and backup.is_dir():
                    target = DIR / match[1]
                    try:
                        if not target.exists():
                            backup.replace(target)
                        else:
                            shutil.rmtree(backup, ignore_errors=True)
                    except OSError as error:
                        print(f"[items] Could not settle {backup.name}: {error}", flush=True)
    if DRAFTS_DIR.exists():
        for folder in DRAFTS_DIR.iterdir():
            shutil.rmtree(folder, ignore_errors=True)


# ----------------------------- names in a picture text -----------------------------

def name_pattern(item: dict) -> "re.Pattern[str]":
    """The item's name in a picture text, in any letter case, its words apart by spaces, "_" or "-": "TV studio",
    "tv studio", "TV-studio". Whole words only, so "desk" does not match "desktop"."""
    words = " ".join(item["name"].split()).split()
    joined = r"[\s_-]+".join(re.escape(word) for word in words)
    return re.compile(rf"(?<![\w@])(?i:{joined})(?!\w)")


def _spans(text: str, pool: Sequence[dict]) -> List[Tuple[int, int, dict]]:
    """Non-overlapping mentions (start, end, item) in text order; at one place the longer name wins ("TV studio
    desk" over "TV studio")."""
    found = []
    for item in pool:
        for match in name_pattern(item).finditer(text):
            found.append((match.start(), match.end(), item))
    found.sort(key=lambda entry: (entry[0], -(entry[1] - entry[0])))
    out, end = [], -1
    for start, stop, item in found:
        if start >= end:
            out.append((start, stop, item))
            end = stop
    return out


def mentions(text: str, pool: Sequence[dict]) -> List[dict]:
    """The items of `pool` the text names (outside quoted dialogue), in order of first mention."""
    plain = dialogue.QUOTE.sub(" ", text or "")
    seen: Dict[str, dict] = {}
    for _, _, item in _spans(plain, pool):
        seen.setdefault(item["slug"], item)
    return list(seen.values())


def mark(text: str, numbered: Sequence[Tuple[dict, int]]) -> str:
    """The text with the image number after the first mention of each item: "in the TV studio" -> "in the TV studio
    [3]" (app/refs.py: the composition takes "[3]" as "from image 3"). Quoted dialogue is left as it is."""
    numbers = {item["slug"]: number for item, number in numbered}
    quoted = [(m.start(), m.end()) for m in dialogue.QUOTE.finditer(text or "")]
    out, pos, done = [], 0, set()
    for start, stop, item in _spans(text or "", [item for item, _ in numbered]):
        if item["slug"] in done or any(a <= start < b for a, b in quoted):
            continue
        done.add(item["slug"])
        out += [text[pos:stop], f" [{numbers[item['slug']]}]"]
        pos = stop
    return "".join(out) + (text or "")[pos:]


# ----------------------------- checks -----------------------------

def check_name(name: str, slug: str = "") -> str:
    """Why `name` cannot name an item ("" = fine); `slug` is the item being edited. A character's name is refused:
    a picture text that names it would mean both."""
    name = " ".join((name or "").split())
    if not name:
        return "Enter a name for the place or item."
    if len(name) > MAX_NAME:
        return f"Use at most {MAX_NAME} characters for the name."
    if llm.blocked(name):
        return llm.BLOCK_MSG
    if not re.search(r"\w", name):
        return "Use letters or numbers in the name."
    if any(item["name"].casefold() == name.casefold() and item["slug"] != slug for item in all_items()):
        return f"A place or item named ‘{name}’ already exists. Choose another name."
    if any(person["name"].casefold() == name.casefold() for person in characters.all_characters()):
        return f"A character is named ‘{name}’. Choose another name for the place or item."
    return ""


def check_look(look: str, kind: str = PLACE) -> str:
    """Why a description cannot be used ("" = fine): only text that combines minors with sexual content is refused,
    as everywhere (llm.blocked)."""
    look = " ".join((look or "").split())
    if not look:
        return ("Describe the place: what it is, its colors, furniture, light and mood." if kind == PLACE else
                "Describe the object: what it is, its shape, colors and materials.")
    if len(look) > MAX_LOOK:
        return f"Use at most {MAX_LOOK} characters for the description."
    if llm.blocked(look):
        return llm.BLOCK_MSG
    return ""


def check_photo(path: str | Path, style: str = styles.REALISTIC) -> str:
    """The gate of an item's photo ("" = it may be used): the vision model sees no person who looks under 18. Nothing
    else about the picture is refused: a place or an object has no person rules."""
    info = llm.check_item_photo(str(path), style)
    if info is None:  # an answer that could not be read is asked again
        info = llm.check_item_photo(str(path), style)
    if info is None:
        return "The photo could not be checked. Try again."
    if info["humans"] and info["youngest_human_age"] < 18:
        return (f"⛔ The photo appears to show a minor (estimated age {info['youngest_human_age']}). Only adults are "
                "allowed.")
    return ""


def gate_models(progress_cb: Optional[ProgressCb] = None) -> Optional[str]:
    """Install what check_photo needs (the vision model); an error message or None."""
    from . import model_manager

    return model_manager.ensure([llm.MODEL_ID], progress_cb)


# ----------------------------- drafts -----------------------------

def _draft_folder(draft_id: str) -> Path:
    if not isinstance(draft_id, str) or not _ID.fullmatch(draft_id):
        raise ValueError("Invalid item draft.")
    return DRAFTS_DIR / draft_id


def _manifest(draft_id: str) -> dict:
    """The draft's manifest; ValueError when the draft is gone (discarded, or left behind and removed)."""
    try:
        data = json.loads((_draft_folder(draft_id) / "draft.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("This item draft was closed. Start again.") from error
    if not isinstance(data, dict) or data.get("id") != draft_id:
        raise ValueError("This item draft was closed. Start again.")
    return data


def _write_manifest(data: dict) -> None:
    data["updated"] = time.time()
    atomic_text(_draft_folder(data["id"]) / "draft.json", json.dumps(data, ensure_ascii=False, indent=1))


def _draft_photos(draft_id: str, data: dict) -> List[Tuple[Path, dict]]:
    """(path, entry) of every photo of the draft whose file is still the one recorded."""
    folder = _draft_folder(draft_id)
    out = []
    for entry in data.get("photos") or []:
        name = entry.get("file") if isinstance(entry, dict) else None
        if isinstance(name, str) and _DRAFT_PHOTO.fullmatch(name):
            path = _verified(folder / name, entry.get("sha256"))
            if path is not None:
                out.append((path, entry))
    return out


def _clean_old_drafts() -> None:
    if not DRAFTS_DIR.exists():
        return
    now = time.time()
    for folder in DRAFTS_DIR.iterdir():
        try:
            if folder.is_dir() and now - folder.stat().st_mtime > DRAFT_TTL:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            continue


def new_draft() -> str:
    """An empty draft (a place); returns its id."""
    _clean_old_drafts()
    draft_id = uuid.uuid4().hex
    _draft_folder(draft_id).mkdir(parents=True)
    _write_manifest({"id": draft_id, "slug": "", "name": "", "kind": PLACE, "look": "", "style": styles.DEFAULT,
                     "photos": [], "created": time.time()})
    return draft_id


def _new_photo_name() -> str:
    return f"p_{uuid.uuid4().hex[:16]}.png"


def _add_photo_entry(folder: Path, source: Path, origin: str) -> dict:
    name = _new_photo_name()
    shutil.copyfile(source, folder / name)
    return {"file": name, "sha256": _digest(folder / name), "origin": origin}


def load_draft(slug: str) -> str:
    """A draft holding copies of the item's photos, for editing it on the creator screen."""
    item = get(slug)
    if item is None:
        raise ValueError("Place or item not found.")
    draft_id = new_draft()
    folder = _draft_folder(draft_id)
    with _DRAFT_LOCK:
        data = _manifest(draft_id)
        data.update(slug=slug, name=item["name"], kind=item["kind"], look=item["look"], style=item["style"])
        data["photos"] = [_add_photo_entry(folder, Path(path), "kept") for path in item["photos"]]
        _write_manifest(data)
    return draft_id


def draft(draft_id: str) -> dict:
    """What the creator screen shows: the fields and the photo paths in order with their origins."""
    data = _manifest(draft_id)
    photos = _draft_photos(draft_id, data)
    return {"id": draft_id, "slug": data.get("slug") or "", "name": data.get("name") or "",
            "kind": _kind(data.get("kind")), "look": data.get("look") or "", "style": styles.of(data.get("style")).id,
            "photos": [str(path) for path, _ in photos], "origins": [entry.get("origin", "") for _, entry in photos],
            "busy": [job.id for job in _draft_jobs(draft_id)]}


def update_draft(draft_id: str, *, name: Optional[str] = None, kind: Optional[str] = None,
                 look: Optional[str] = None, style: Optional[str] = None) -> dict:
    """Remember the form's fields in the draft (a photo job reads the kind, look and style from it). A new style or
    kind applies to the photos made from then on. Nothing is remembered on a ValueError."""
    with _DRAFT_LOCK:
        data = _manifest(draft_id)
        if name is not None:
            data["name"] = " ".join(str(name).split())[:MAX_NAME]
        if kind is not None:
            if kind not in KINDS:
                raise ValueError("Choose a place or an object.")
            data["kind"] = kind
        if look is not None:
            data["look"] = " ".join(str(look).split())[:MAX_LOOK]
        if style is not None:
            chosen = styles.get(style).id
            if chosen != data.get("style"):  # a stored item keeps working; only a change is checked
                problem = llm.style_problem(chosen)
                if problem:
                    raise ValueError(problem)
            data["style"] = chosen
        _write_manifest(data)
    return draft(draft_id)


def discard_draft(draft_id: str) -> None:
    """The draft and its files are removed; its queued jobs are withdrawn and a running one is stopped."""
    folder = _draft_folder(draft_id)
    busy = [job.id for job in _draft_jobs(draft_id)]
    if busy:
        queue_manager.remove(busy)
    shutil.rmtree(folder, ignore_errors=True)


def remove_photo(draft_id: str, index: int) -> dict:
    with _DRAFT_LOCK:
        data = _manifest(draft_id)
        photos = _draft_photos(draft_id, data)
        if not 0 <= index < len(photos):
            raise ValueError("That photo is no longer in the draft.")
        path, entry = photos[index]
        data["photos"] = [item for _, item in photos if item is not entry]
        _write_manifest(data)
        path.unlink(missing_ok=True)
    return draft(draft_id)


def add_upload(draft_id: str, path: str | Path, consent: bool) -> Tuple[bool, str]:
    """Add your own photo: it needs the consent box and passes the photo gate (no one who looks under 18)."""
    if not consent:
        return False, "Confirm that you may use the photo and that any person in it is an adult who agreed."
    data = _manifest(draft_id)
    if len(_draft_photos(draft_id, data)) >= MAX_PHOTOS:
        return False, f"A place or item has at most {MAX_PHOTOS} photos. Remove one first."
    problem = gate_models()
    if problem:
        return False, problem
    folder = _draft_folder(draft_id)
    staged = folder / f".upload_{uuid.uuid4().hex[:8]}.png"
    try:
        problem = characters.normalize_photo(path, staged) or check_photo(staged, styles.of(data.get("style")).id)
        if problem:
            return False, problem
        with _DRAFT_LOCK:
            data = _manifest(draft_id)
            if len(_draft_photos(draft_id, data)) >= MAX_PHOTOS:
                return False, f"A place or item has at most {MAX_PHOTOS} photos. Remove one first."
            entry = _add_photo_entry(folder, staged, "uploaded")
            entry["consent"] = True
            data["photos"] = [item for _, item in _draft_photos(draft_id, data)] + [entry]
            _write_manifest(data)
    finally:
        staged.unlink(missing_ok=True)
    return True, "Photo added."


def replace_photos(draft_id: str, entries: Sequence[object]) -> None:
    """Called by the photo job: the draft's photos become `entries` in this order, each a manifest entry (a photo that
    stays) or (path in the draft folder, origin) for a new file; files no longer listed are removed."""
    folder = _draft_folder(draft_id)
    with _DRAFT_LOCK:
        data = _manifest(draft_id)
        rows = [dict(entry) if isinstance(entry, dict) else
                {"file": Path(entry[0]).name, "sha256": _digest(Path(entry[0])), "origin": entry[1]}
                for entry in entries][:MAX_PHOTOS]
        listed = {row["file"] for row in rows}
        data["photos"] = rows
        _write_manifest(data)
        for stray in folder.glob("p_*.png"):
            if stray.name not in listed:
                stray.unlink(missing_ok=True)


def draft_photos(draft_id: str) -> List[Tuple[Path, dict]]:
    """(path, manifest entry) of the draft's photos in order (for the photo job)."""
    return _draft_photos(draft_id, _manifest(draft_id))


def new_photo_path(draft_id: str) -> Path:
    """A fresh file name in the draft folder for a generated photo."""
    return _draft_folder(draft_id) / _new_photo_name()


# ----------------------------- jobs on a draft (GPU queue) -----------------------------

def _draft_jobs(draft_id: str) -> List[queue_manager.Job]:
    return [job for job in queue_manager.snapshot() if job.kind == PHOTO_KIND and job.params.get("draft") == draft_id
            and job.status in (queue_manager.PENDING, queue_manager.RUNNING)]


def enqueue_photos(draft_id: str, count: int, *, keep_own: bool = False) -> queue_manager.Job:
    """Queue `count` photos (1-8) of one place or object from the draft's kind, description and style; with keep_own
    the uploaded or saved photos stay and the new ones are views of the first of them."""
    data = _manifest(draft_id)
    kind = _kind(data.get("kind"))
    count = int(count)
    if not MIN_PHOTOS <= count <= MAX_PHOTOS:
        raise ValueError(f"Choose {MIN_PHOTOS} to {MAX_PHOTOS} photos.")
    if not keep_own and (problem := check_look(data.get("look", ""), kind)):
        raise ValueError(problem)
    own = sum(entry.get("origin") in ("uploaded", "kept") for _, entry in _draft_photos(draft_id, data))
    if keep_own and not own:
        raise ValueError("Upload a photo first, or generate a new place or object.")
    if keep_own and count <= own:
        raise ValueError(f"There {'are' if own != 1 else 'is'} already {own} photo{'s' if own != 1 else ''} of your "
                         "own. Raise the number of photos to add views of the first one, or untick “Keep my own "
                         "photos” to generate a new one.")
    label = f"Place or item photos: {data.get('name') or 'new'} ({count})"
    return queue_manager.add(PHOTO_KIND, label, {"draft": draft_id, "count": count, "keep_own": bool(keep_own),
                                                 "look": data.get("look", ""), "kind": kind,
                                                 "style": styles.of(data.get("style")).id})


def enqueue_regenerate(draft_id: str, index: int) -> queue_manager.Job:
    """Queue a new take of one generated photo (the others stay)."""
    data = _manifest(draft_id)
    photos = _draft_photos(draft_id, data)
    if not 0 <= index < len(photos):
        raise ValueError("That photo is no longer in the draft.")
    if photos[index][1].get("origin") != "generated":
        raise ValueError("Only generated photos can be generated again. Remove this one or upload another.")
    kind = _kind(data.get("kind"))
    if len(photos) == 1 and (problem := check_look(data.get("look", ""), kind)):  # the only photo: drawn from the text
        raise ValueError(problem)
    return queue_manager.add(PHOTO_KIND, f"Place or item photo: {data.get('name') or 'new'}",
                             {"draft": draft_id, "count": 1, "index": int(index), "look": data.get("look", ""),
                              "kind": kind})


def run_photo_job(p: dict, cb) -> tuple:
    """Queue runner: generate photos into the draft (app/item_photos.py)."""
    from . import item_photos

    draft_id = p.get("draft", "")
    try:
        _manifest(draft_id)
        if p.get("index") is not None:
            notes = item_photos.regenerate(draft_id, int(p["index"]), cb)
        else:
            notes = item_photos.generate(draft_id, int(p.get("count") or 1), bool(p.get("keep_own")), cb)
    except ValueError as error:
        return False, [], [], str(error)
    cb(1.0, "Photos ready")
    return True, [], notes, ""


# ----------------------------- saving and deleting -----------------------------

def _unique_slug(name: str) -> str:
    base = slugify(name)
    slug, number = base, 2
    while (DIR / slug).exists():
        slug, number = f"{base}_{number}", number + 1
    return slug


def _swap_in(staging: Path, slug: str) -> None:
    """Put the staged folder in place of items/<slug>; an existing folder is replaced, never half (a crash between
    the two renames is settled by `startup`)."""
    target = _folder(slug)
    DIR.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        staging.replace(target)
        return
    backup = DIR / f".replaced_{slug}_{uuid.uuid4().hex[:8]}"
    target.replace(backup)
    try:
        staging.replace(target)
    except OSError:
        backup.replace(target)
        raise
    shutil.rmtree(backup, ignore_errors=True)  # everything it held was copied into the draft and saved again


def save_draft(draft_id: str, *, name: Optional[str] = None, kind: Optional[str] = None,
               look: Optional[str] = None, style: Optional[str] = None) -> Tuple[Optional[dict], str]:
    """Save the draft as an item: photos, name, kind, description and style together. Returns (profile, message);
    nothing is written on an error. A draft made by load_draft replaces that item (same slug), or becomes a new item
    when that one was deleted while it was open. The style is stored only when it is not Realistic."""
    if any(value is not None for value in (name, kind, look, style)):
        update_draft(draft_id, name=name, kind=kind, look=look, style=style)
    if _draft_jobs(draft_id):
        return None, "Wait for the photos that are still being made."
    with _SAVE_LOCK, _DRAFT_LOCK:
        data = _manifest(draft_id)
        slug = data.get("slug") or ""
        if slug and get(slug) is None:
            slug = ""  # deleted while it was open: a new item
        kind = _kind(data.get("kind"))
        problem = check_name(data.get("name", ""), slug) or check_look(data.get("look", ""), kind)
        if problem:
            return None, problem
        photos = _draft_photos(draft_id, data)
        if not MIN_PHOTOS <= len(photos) <= MAX_PHOTOS:
            return None, f"Add {MIN_PHOTOS} to {MAX_PHOTOS} reference photos."
        previous = json.loads(_file(slug).read_text(encoding="utf-8")) if slug else {}
        slug = slug or _unique_slug(data["name"])
        now = time.time()
        record = {"format": FORMAT, "name": data["name"], "slug": slug, "kind": kind, "look": data["look"],
                  "created": previous.get("created", now), "updated": now}
        if styles.of(data.get("style")).id != styles.REALISTIC:
            record["style"] = styles.of(data.get("style")).id
        staging = DRAFTS_DIR / f".staging_{uuid.uuid4().hex[:12]}"
        try:
            (staging / "photos").mkdir(parents=True)
            entries = []
            for number, (path, entry) in enumerate(photos, 1):
                target = staging / "photos" / f"{number:02d}.png"
                shutil.copyfile(path, target)
                entries.append({"file": target.name, "sha256": _digest(target),
                                "origin": "uploaded" if entry.get("origin") == "uploaded" else
                                entry.get("origin", "kept")})
            record["photos"] = entries
            atomic_text(staging / "item.json", json.dumps(record, ensure_ascii=False, indent=1))
            _swap_in(staging, slug)
        except OSError as error:
            return None, f"Could not save the place or item ({error}). Close files that use it and try again."
        finally:
            shutil.rmtree(staging, ignore_errors=True)  # gone already once it is in place
    discard_draft(draft_id)
    count = f"{len(photos)} photo{'s' if len(photos) != 1 else ''}"
    noun = "Place" if kind == PLACE else "Object"
    return get(slug), f"{noun} ‘{record['name']}’ saved with {count}."


def delete(slug: str) -> str:
    item = get(slug)
    if not item:
        return "Place or item not found."
    if not trash.move(_folder(slug)):
        return "Could not move the folder to the Recycle Bin. Close files that use it and try again."
    return (f"‘{item['name']}’ moved to the Recycle Bin. Scenes that use its photo as a reference image keep it; "
            "scenes that only name it no longer get it.")


queue_manager.register(PHOTO_KIND, run_photo_job)
