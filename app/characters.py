# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Characters: a name, a gender, a kind (a person, or an animal or creature), a look description, 1 to 8 reference
photos of one adult and the character's own voice.

Identity in pictures and scenes comes from the reference photos: Qwen-Image-Edit composes a start frame
with the character from its first (primary) photo only (`render`, app/cast.py, app/refs.py), and the SFace
face check (app/qc.py) compares faces with every photo, so the other views (turned heads, half and full
body) make that check hold when a scene does not show the face from the front. Nothing is trained, so a
character is ready the moment it is saved.

Storage, one folder per character under characters/ (never written before Save):

    characters/<slug>/character.json   {"format": 2, "name", "slug", "gender": "woman" | "man", "look",
                                         "style" (app/styles.py; absent = "realistic"),
                                         "kind": "creature" (an animal or creature; absent = a person),
                                         "photos": [{"file", "sha256", "origin"}], "voice": {"file", "sha256",
                                         "origin", "description", "preset", "seconds", "f0_hz", "consent"},
                                         "created", "updated"}
    characters/<slug>/photos/01.png ... 08.png
    characters/<slug>/voice/sample.wav  48 kHz mono, 3-12 s

A photo or the voice whose file no longer matches its sha256 is ignored. The voice belongs to the
character, not to the voice library: it is addressed as "character:<slug>" (dialogue.CHARACTER_VOICE),
which voices.get resolves, and the character's dialogue lines use it by default.

A character's style is the style its photos were drawn in (Realistic, 3D animation, 2D cartoon, Anime or one of the
owner's own, app/styles.py): the creator draws the first photo and every view in it, and the photo gate and the face
thresholds (qc.FACE_RULES) follow it. A character can still appear in a film of another style: the composition then
redraws it in the film's style.

A character's kind is PERSON (every character before 2026-11-16, and the default) or CREATURE: an animal or creature,
such as a corgi that stands upright in a hoodie. A creature lives only in animated styles (`creature_style_ok`: never
Realistic nor the owner's styles based on it, so never on the Video and Image pages); its photos are refused only
when they show a person who looks under 18 (`check_photo`). It has no face the face checks know: it is never face-detected, compared or
lip-synced (app/qc.py, app/lipsync.py). Its gender is its voice and its pronoun (she / he). Every text of a person's
character stays as it was; a creature's texts never call it a man, a woman or a person.

Reading: get(slug) -> profile {"name", "slug", "gender", "kind", "look", "style", "photos" (verified paths, primary
first), "voice" (voice key or ""), "voice_info", "legacy", "created", "updated", "record"} or None;
all_characters(); usable(slug) (at least one photo); is_creature(profile); state(profile) (the status text);
missing(slugs) and missing_message(slugs) for casts that name deleted or archived characters; identity(profile) for
cache keys.

Older characters still load: a reference-portrait character (status "reference", preview.png and
portrait_sha256, a voice from the library) is a character with one photo; a character from the removed
LoRA training has no reference photos (legacy "lora") and cannot be used until it is edited and saved
with photos. Saving an older character in the current format sends its old folder (training photos,
captions, preview) to the Recycle Bin instead of deleting it. A slug whose folder is gone (a deleted or
archived character) is reported by `missing`. `startup()` runs once when the app starts.

The creator screen (Characters page) works on a draft: a private scratch folder under character_drafts/
that is discarded when the screen is left. Every call takes the draft id and returns plain data:

    new_draft() -> id                       an empty draft (a person)
    load_draft(slug) -> id                  a draft holding copies of a character (to edit it)
    draft(id) -> dict                       the view: name, gender, kind, look, style, photo paths and origins, voice
                                            path and voice_info, slug (when editing), busy (ids of running jobs)
    update_draft(id, name=, gender=, look=, style=, kind=)  remember the form's fields (the photo jobs read them;
                                            the kind changes only while the draft has no photos)
    enqueue_photos(id, count, keep_own=)    GPU queue job: generate `count` photos of one character (keep_own:
                                            keep the uploaded photos and add views of the first of them)
    enqueue_regenerate(id, index)           GPU queue job: a new take of one generated photo
    remove_photo(id, index)
    add_upload(id, path, consent) -> (ok, message)      an own photo, through the photo gate of the draft's kind
    enqueue_voice_design(id, description)   GPU queue job: VoxCPM2 voice design from the description
    clone_voice(id, sample, consent) -> (ok, message)   3-12 s of speech becomes the voice
    use_preset_voice(id, voice_slug) -> (ok, message)   start from a bundled voice
    set_designed_voice(id, sample, description) -> (ok, message)   a sample designed elsewhere (a script)
    save_draft(id, name=, gender=, look=, style=, kind=) -> (profile | None, message)
                                            photos, voice, name, look, gender, style and kind saved together
    discard_draft(id)
    delete(slug) -> message                 the character folder to the Recycle Bin

For the form: check_name(name, slug), check_look(look, kind) and check_voice_description(text) return why a value
is refused ("" = fine; a look or a voice that names a minor is refused, and a creature's look that names a young
animal); CONSENT is the checkbox text for uploaded photos, voices.CONSENT_MSG the one for a cloned voice.

The jobs report through the queue (queue_manager.Job: status, progress, error, notes); their results land in
the draft, so the screen reads `draft(id)` again when a job ends. Generated photos and uploads alike pass the
gate (`check_photo`): the vision model must see one real-looking adult, and YuNet exactly one face. Generated
views must also keep the face of the first photo (SFace similarity of at least qc.FACE_MIN, else the view is
drawn again), and an upload must show the person of the first photo (`same_person`). An animal or creature's photos
pass the creature gate instead: the vision model sees exactly one animal or creature and no human, and no face is
detected or compared. Each draft photo records the kind whose gate it passed ("gate"), and Save takes only photos
that passed the gate of the draft's kind.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import config, dialogue, llm, prefs, queue_manager, refs, styles, trash
from .persistence import atomic_text

ProgressCb = Callable[[float, str], None]

DIR = config.STATE_ROOT / "characters"
DRAFTS_DIR = config.STATE_ROOT / "character_drafts"
FORMAT = 2
MIN_PHOTOS, MAX_PHOTOS = 1, 8
GENDERS = ("woman", "man")
PERSON, CREATURE = "person", "creature"  # a character's kind; stored as "kind" for a creature only
KINDS = (PERSON, CREATURE)
CREATURE_STYLE_MSG = ("An animal or creature needs an animated style: 3D animation, 2D cartoon, Anime or one of your "
                      "own animated styles.")
CREATURE_ADULT_MSG = "⛔ Only adult characters are supported: describe a fully grown animal or creature."
CREATURE_SEXUAL_MSG = "⛔ Sexual or nude content is not allowed with an animal or creature character."
MAX_NAME, MAX_LOOK, MAX_VOICE_DESCRIPTION = 40, 600, 300
MAX_PHOTO_BYTES = 50 * 1024 * 1024
MAX_PHOTO_PIXELS = 40_000_000
PHOTO_SIDE, MIN_PHOTO_SIDE = 1536, 384
CLONE_SECONDS = (3.0, 12.0)  # a cloned voice: at least 3 s of speech; longer samples are cut to 12 s
DRAFT_TTL = 24 * 3600  # a draft left behind (a closed browser tab) is removed after a day
PHOTO_KIND, VOICE_KIND = "character_photos", "character_voice"
DEFAULT_LABEL = "No character"
REFERENCE = "reference"  # the status of an older reference-portrait character
CONSENT = ("I confirm that this photo shows a fictional AI character or an adult who has agreed to the use of "
           "their likeness.")
CREATURE_CONSENT = "I confirm that this photo shows a fictional character or an animal, and that I may use it."
_ID = re.compile(r"[0-9a-f]{32}")
_REPLACED = re.compile(r"\.replaced_([A-Za-z0-9_-]{1,100})_[0-9a-f]{8}")  # a folder a save set aside (_swap_in)
_PHOTO_FILE = re.compile(r"\d{2}\.png")
_DRAFT_PHOTO = re.compile(r"p_[0-9a-f]{16}\.png")
_SAVE_LOCK = threading.RLock()
_DRAFT_LOCK = threading.RLock()


# ----------------------------- names and folders -----------------------------

# Windows device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9) cannot be folder names: a slug
# equal to one gets an underscore, so a character named "Con" still gets a usable folder
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def slugify(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.replace("ł", "l").replace("Ł", "L"))
    text = re.sub(r"[^a-z0-9]+", "_", text.encode("ascii", "ignore").decode().lower()).strip("_")
    text = text[:24] or "postac"
    return f"{text}_" if text in _RESERVED else text


def _folder(slug: str) -> Path:
    if not isinstance(slug, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", slug):
        raise ValueError("Invalid character identifier.")
    folder = DIR / slug
    if folder.resolve().parent != DIR.resolve():
        raise ValueError("Character folder is outside the character library.")
    return folder


def _file(slug: str) -> Path:
    return _folder(slug) / "character.json"


_DIGESTS: Dict[tuple, str] = {}
_DIGESTS_LOCK = threading.Lock()


def _digest(path: Path) -> str:
    """sha256 of a file, remembered by (path, size, modification time): pages ask for characters every few
    seconds and a photo is only hashed again when it changes."""
    try:
        path = Path(path)
        if path.is_symlink():
            return ""
        stat = path.stat()
        memo = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return ""
    with _DIGESTS_LOCK:
        found = _DIGESTS.get(memo)
    if found is None:
        try:
            with path.open("rb") as stream:
                found = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError:
            return ""
        with _DIGESTS_LOCK:
            if len(_DIGESTS) >= 1024:
                _DIGESTS.clear()
            _DIGESTS[memo] = found
    return found


# ----------------------------- kinds -----------------------------

def _kind(value: object) -> str:
    """A stored kind as one of KINDS: anything but CREATURE (absent, unknown) is a person."""
    return CREATURE if value == CREATURE else PERSON


def is_creature(ch: object) -> bool:
    """True for an animal or creature character: a profile, a draft view or anything else with "kind"."""
    return isinstance(ch, dict) and ch.get("kind") == CREATURE


def creature_style_problem(style_id: object) -> str:
    """Why an animal or creature cannot be drawn in this style ("" = it can): the Realistic family (Realistic and the
    owner's styles based on it) draws real-looking people and places, and an own style whose look words are sexual
    or nude (llm.sexual) would put them into every picture of it."""
    look = styles.of(style_id)
    if look.family == styles.REALISTIC:
        return CREATURE_STYLE_MSG
    return CREATURE_SEXUAL_MSG if look.custom and llm.sexual(look.about) else ""


def creature_style_ok(style_id: object) -> bool:
    """Whether an animal or creature can be drawn in this style (creature_style_problem)."""
    return not creature_style_problem(style_id)


# ----------------------------- reading characters -----------------------------

def _verified(path: Path, sha: object) -> Optional[Path]:
    return path if isinstance(sha, str) and path.is_file() and _digest(path) == sha else None


def _photo_paths(folder: Path, record: dict) -> List[Path]:
    photos = record.get("photos")
    found = []
    for item in photos[:MAX_PHOTOS] if isinstance(photos, list) else []:
        name = item.get("file") if isinstance(item, dict) else None
        if isinstance(name, str) and _PHOTO_FILE.fullmatch(name):
            path = _verified(folder / "photos" / name, item.get("sha256"))
            if path is not None:
                found.append(path)
    return found


def _voice_info(folder: Path, record: dict) -> Optional[dict]:
    voice = record.get("voice")
    if not isinstance(voice, dict) or voice.get("file") != "sample.wav":
        return None
    path = _verified(folder / "voice" / "sample.wav", voice.get("sha256"))
    return {**voice, "path": str(path)} if path else None


def _profile(slug: str, record: dict) -> dict:
    """The character as the rest of the app sees it: name, slug, gender, kind (PERSON when the record names none),
    look, the verified photo paths (first = primary) and the voice key ("character:<slug>", a library voice of an
    older character, or ""). "record" is character.json as stored (see `identity`)."""
    folder = _folder(slug)
    gender = record.get("gender") if record.get("gender") in GENDERS else "woman"
    base = {"name": record["name"], "slug": slug, "gender": gender, "kind": _kind(record.get("kind")),
            "look": str(record.get("look") or ""), "style": styles.of(record.get("style")).id,
            "created": record.get("created", 0), "updated": record.get("updated", record.get("created", 0)),
            "record": record}
    if record.get("format") == FORMAT:
        own = _voice_info(folder, record)
        return {**base, "photos": [str(path) for path in _photo_paths(folder, record)],
                "voice": dialogue.CHARACTER_VOICE + slug if own else "", "voice_info": own, "legacy": ""}
    library_voice = record.get("voice") if isinstance(record.get("voice"), str) else ""
    if record.get("status") == REFERENCE:
        portrait = _verified(folder / "preview.png", record.get("portrait_sha256"))
        return {**base, "photos": [str(portrait)] if portrait else [], "voice": library_voice,
                "voice_info": None, "legacy": "reference"}
    # A character of the removed LoRA training: its photos were training material, not checked references.
    return {**base, "photos": [], "voice": library_voice, "voice_info": None, "legacy": "lora"}


def get(slug: str) -> Optional[dict]:
    if not slug:
        return None
    try:
        data = json.loads(_file(slug).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("slug") != slug or not isinstance(data.get("name"), str):
            return None
        return _profile(slug, data)
    except (OSError, ValueError):
        return None


def identity(ch: dict) -> dict:
    """What a character contributes to the key of a Timeline clip or picture: what changes a generated result
    (name, gender, look, the sha256 of each verified photo in order and of the voice), never local paths, the
    save time or where a photo came from, so saving a character again without a change keeps its clips. An
    older character's identity is its stored record: the keys of clips made before 15.11.2026 hashed that, so an
    unchanged older character keeps its clips. A profile without a record is its own identity. An animal or creature
    adds its kind (a person keeps the identity it had)."""
    if not isinstance(ch, dict) or not isinstance(ch.get("record"), dict):
        return ch
    if ch["record"].get("format") != FORMAT:
        return ch["record"]
    found = {"format": FORMAT, "slug": ch["slug"], "name": ch["name"], "gender": ch["gender"], "look": ch["look"],
             "photos": [_digest(Path(path)) for path in ch.get("photos") or []],
             "voice": (ch.get("voice_info") or {}).get("sha256")}
    if ch.get("style", styles.REALISTIC) != styles.REALISTIC:  # a realistic character keeps the identity it had
        found["style"] = ch["style"]
    if is_creature(ch):
        found["kind"] = CREATURE
    return found


def all_characters() -> List[dict]:
    if not DIR.exists():
        return []
    found = [get(p.name) for p in DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted((c for c in found if c), key=lambda c: c.get("created", 0))


def startup() -> None:
    """At the start of the app, before any page: finish what a save cut short and drop what no browser tab can
    reach any more. A save replaces characters/<slug> with two renames (`_swap_in`); after a crash between them
    the old folder waits as .replaced_<slug>_<id> and goes back in place, and one whose new folder is in place is
    cleaned up as the save would have. Drafts and staged saves belong to browser tabs of the previous run (Gradio
    keeps a tab's state in memory), so they are removed."""
    if DIR.exists():
        with _SAVE_LOCK:
            for backup in DIR.iterdir():
                match = _REPLACED.fullmatch(backup.name)
                if match and backup.is_dir():
                    _settle_replaced(backup, DIR / match[1])
    if DRAFTS_DIR.exists():
        for folder in DRAFTS_DIR.iterdir():
            shutil.rmtree(folder, ignore_errors=True)


def _settle_replaced(backup: Path, target: Path) -> None:
    """A folder _swap_in set aside: back in place when its character is missing, else disposed of."""
    try:
        if not target.exists():
            backup.replace(target)
        elif not _dispose(backup):
            print(f"[characters] {backup.name} could not go to the Recycle Bin; it is tried again at the next start.",
                  flush=True)
    except OSError as error:  # a file still open: tried again at the next start
        print(f"[characters] Could not settle {backup.name}: {error}", flush=True)


def reference_photos(ch: Optional[dict]) -> List[Path]:
    """The character's verified reference photos, the primary one first."""
    return [Path(path) for path in (ch or {}).get("photos") or []]


def reference_portrait(ch: Optional[dict]) -> Optional[Path]:
    """The primary reference photo: what a composed picture takes the identity from."""
    photos = reference_photos(ch)
    return photos[0] if photos else None


def preview_path(ch: Optional[dict]) -> Optional[Path]:
    """The picture that stands for the character on cards and lists (its primary photo), or None."""
    return reference_portrait(ch)


def usable(slug: str) -> Optional[dict]:
    """The character when it can be used in a picture or a scene (at least one verified photo), else None."""
    ch = get(slug)
    return ch if ch and ch.get("photos") else None


def timeline_profile(slug: str) -> Optional[dict]:
    return usable(slug)


def missing(slugs: Sequence[str]) -> List[str]:
    """The slugs that name no character any more (deleted or archived)."""
    return [slug for slug in dict.fromkeys(slugs or ()) if slug and get(slug) is None]


def missing_message(slugs: Sequence[str]) -> str:
    """A sentence for the slugs `missing` found ("" when none)."""
    gone = missing(slugs)
    if not gone:
        return ""
    names = ", ".join(gone)
    return (f"These characters no longer exist and were left out: {names}. Remove them from the cast or create "
            "them again on the Characters page." if len(gone) > 1 else
            f"The character {names} no longer exists and was left out. Remove it from the cast or create it "
            "again on the Characters page.")


def state(ch: dict) -> str:
    """The status a page shows for a character."""
    if ch.get("photos"):
        return "Ready"
    return "Needs photos (made with the removed LoRA training)" if ch.get("legacy") == "lora" \
        else "Needs reference photos"


def own_voice(key: str):
    """The voices.Voice of a "character:<slug>" key, or None."""
    from . import voices

    if not isinstance(key, str) or not key.startswith(dialogue.CHARACTER_VOICE):
        return None
    ch = get(key[len(dialogue.CHARACTER_VOICE):])
    info = (ch or {}).get("voice_info")
    if not info:
        return None
    return voices.Voice(name=ch["name"], slug=key, gender="male" if ch["gender"] == "man" else "female",
                        path=info["path"], description=str(info.get("description") or ""),
                        seconds=float(info.get("seconds") or 0.0), created=float(ch.get("updated") or 0.0),
                        f0_hz=float(info.get("f0_hz") or 0.0))


def forget_library_voice(voice_slug: str) -> None:
    """A library voice was deleted: older characters that spoke with it go back to the default voice."""
    with _SAVE_LOCK:
        for path in DIR.glob("*/character.json") if DIR.exists() else []:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("format") != FORMAT and data.get("voice") == voice_slug:
                data["voice"] = ""
                atomic_text(path, json.dumps(data, ensure_ascii=False, indent=1))


def choices() -> List[Tuple[str, str]]:
    """The top picker's characters: no character, then every usable one."""
    return [(DEFAULT_LABEL, "")] + [(c["name"], c["slug"]) for c in all_characters() if c.get("photos")]


def selected() -> Optional[dict]:
    """The character chosen in the top bar's Tools panel (None = no character)."""
    return usable(prefs.load().get("character", ""))


def for_job(p: dict) -> Tuple[Optional[dict], str]:
    """In a job: (the character or None, a note when the character is gone since the job was added)."""
    slug = p.get("character") or ""
    ch = usable(slug)
    return ch, ("" if ch or not slug else
                "The selected character is unavailable; the picture was made without it.")


# ----------------------------- pictures with a character -----------------------------

# What a prompt states about clothing: an outfit, or less of it or none ("nude", "topless",
# "shirtless"). Either is the user's choice and stays as written; only a prompt silent about
# clothing gets a default outfit. Only words about clothing count: "master suite", "on top of the
# bed" and "swimming" say nothing about what a person wears.
_CLOTHING_WORDS = (
    r"\w*wear\w*|dress(?:es|ed)?|cloth(?:es|ing|ed)?|outfits?|shirts?|t-shirts?|"
    r"(?<!on )tops?(?!\s+of\b)|tank|bikinis?|swim(?:suit|wear)s?|lingerie|bras?|bralette|"
    r"jeans|pants|trousers|skirts?|shorts|sweaters?|hoodies?|coats?|jackets?|\w*suits?|"
    r"uniforms?|robes?|towels?|leggings|blouses?|gowns?|costumes?|underwear|panties|"
    r"stockings|sweatshirts?|cardigans?|kimonos?|aprons?|armou?r|corsets?|pajamas|pyjamas")
_UNDRESS_WORDS = (
    r"nude|nudes|nudity|naked|topless|shirtless|bottomless|braless|bare[- ]?(?:chested|breasted)|"
    r"bare[- ]chest|undress\w*|unclothed|disrob\w*|in\s+the\s+buff|(?:no|without)\s+(?:clothes|clothing)|"
    r"nag(?:a|i|o|ie|ą)|nadzy|rozebran\w*|bez\s+ubra\w*")
_ATTIRE_STATED = re.compile(rf"\b(?:{_CLOTHING_WORDS}|{_UNDRESS_WORDS})\b", re.I)


def states_attire(text: str) -> bool:
    """True when the text says what a person wears - or that they wear little or nothing."""
    return bool(_ATTIRE_STATED.search(text))


# A concrete outfit that fits the scene, first match wins, for a prompt that says nothing about clothing:
# otherwise the composed picture repeats whatever the reference photo wears, so a character on a beach
# would keep the winter coat of its photo. A named outfit was followed far more often than a general
# wording ("an outfit that fits the scene").
_OUTFITS = [
    (re.compile(r"\b(?:beach\w*|pool\w*|ocean|sea|seaside|yacht\w*|boat\w*|swimming|lake|"
                r"tropical|sunbath\w*)\b", re.I), "a black one-piece swimsuit"),
    (re.compile(r"\b(?:gym|workout\w*|fitness|yoga|running|jogging|treadmill)\b", re.I),
     "a fitted long-sleeve sports top and leggings"),
    (re.compile(r"\b(?:snow\w*|ski|skiing|skier|winter|mountain\w*|cabin|chalet|fireplace)\b",
                re.I), "a cozy knit sweater and jeans"),
    (re.compile(r"\b(?:bed|beds|bedroom\w*|duvet|sheets|pillows?)\b", re.I),
     "long-sleeve satin pajamas"),
]
_DEFAULT_OUTFIT = "a fitted black knit dress with long sleeves"
# The same scenes for a man (the women's list put male characters in dresses and swimsuits)
_OUTFITS_MAN = ["black swim shorts and an open short-sleeve linen shirt",
                "a fitted athletic t-shirt and training shorts",
                "a warm knit sweater and dark jeans", "long-sleeve cotton pajamas"]
_DEFAULT_OUTFIT_MAN = "a navy crew-neck sweater and dark jeans"


def outfit_for(scene: str, gender: str = "woman") -> str:
    """A concrete outfit that fits the scene, for a woman (default) or a man."""
    if gender == "man":
        return next((outfit for (rx, _), outfit in zip(_OUTFITS, _OUTFITS_MAN) if rx.search(scene)),
                    _DEFAULT_OUTFIT_MAN)
    return next((outfit for rx, outfit in _OUTFITS if rx.search(scene)), _DEFAULT_OUTFIT)


# The lead of an animal or creature's scene: never "man", "woman" or "person", and no default outfit after the scene
# (its photo shows what it wears, or that it wears nothing, and the scene text may say otherwise).
_CREATURE_LEAD = ("Use the exact character in image 1, a fully grown animal or creature. Preserve the same species, "
                  "face, body shape, colors and markings, and keep its outfit exactly as image 1 shows it. ")


def _creature_render(style: str) -> str:
    """The style's lead of a scene (styles.Style.render) for an animal or creature: the creature and everything
    around it, where a person's scene says the person and everything around them."""
    return styles.get(style).render.replace("the person and everything around them",
                                            "the creature and everything around it")


def render(client, ch: dict, text: str, width: int, height: int, out_path: Path,
           seed: Optional[int] = None,
           progress_cb: Optional[ProgressCb] = None, style: str = styles.REALISTIC) -> Tuple[bool, str]:
    """The character in a scene (width x height), composed by Qwen-Image-Edit from its primary photo, without
    a history entry (a video's start frame, the base of a picture with reference images). Returns (ok, path or
    error). Only the primary photo goes into the composition; the other photos serve the face checks.
    style - the film's style (app/styles.py): the scene is drawn in it, whatever style the photo has. An animal or
    creature has its own lead and gets no default outfit, and is never drawn in the Realistic family of styles."""
    import random

    source = reference_portrait(ch)
    if source is None:
        return False, f"{ch.get('name', 'The character')} has no reference photo. Add photos on the Characters page."
    if is_creature(ch) and (problem := creature_style_problem(style)):
        return False, (f"{ch.get('name', 'The character')} cannot be in a realistic scene. {problem}"
                       if problem == CREATURE_STYLE_MSG else problem)
    if is_creature(ch) and llm.sexual(text):
        return False, CREATURE_SEXUAL_MSG
    seed = random.randint(0, 2 ** 31 - 1) if seed is None or seed < 0 else int(seed)
    if is_creature(ch):
        prompt = _CREATURE_LEAD + _creature_render(style) + text.strip()
    else:
        outfit = f" Wearing {outfit_for(text, ch['gender'])}." if not states_attire(text) else ""
        prompt = ("Use the exact adult identity in image 1. Preserve the same face, hair and skin tone. "
                  + styles.get(style).render + text.strip() + outfit)
    if style == styles.REALISTIC:  # the call it always had
        return refs.compose(client, [str(source)], prompt, width, height, out_path, seed, progress_cb)
    return refs.compose(client, [str(source)], prompt, width, height, out_path, seed, progress_cb, style=style)


# ----------------------------- checks -----------------------------

def check_look(look: str, kind: str = PERSON) -> str:
    """Why a look description cannot be used ("" = fine). An animal or creature (kind CREATURE) is fully grown: its
    look may not name a minor, a young animal (a puppy, a kitten, a cub...) or a childlike look (llm.refers_to_young;
    "baby blue" is a colour)."""
    look = " ".join((look or "").split())
    creature = kind == CREATURE
    if not look:
        return ("Describe how the character looks: what animal or creature it is, its colors, build and what it "
                "wears, if anything." if creature else
                "Describe how the character looks: hair, eyes, skin tone, build and style.")
    if len(look) > MAX_LOOK:
        return f"Use at most {MAX_LOOK} characters for the look."
    if creature:
        if llm.sexual(look):
            return CREATURE_SEXUAL_MSG
        return CREATURE_ADULT_MSG if (llm.blocked(look) or llm.refers_to_minor(look)) else ""
    if llm.blocked(look) or llm.refers_to_minor(look):
        return "⛔ Only adult characters are supported. This description appears to refer to a minor."
    return ""


def check_voice_description(description: str, kind: str = PERSON) -> str:
    """Why a voice description cannot be designed ("" = fine): a character's voice is an adult's voice, and an animal
    or creature's is a fully grown one's (no puppy or cub voice: llm.refers_to_young)."""
    description = " ".join((description or "").split())
    if not description:
        return "Describe the voice first, for example: a warm, low, calm voice."
    if len(description) > MAX_VOICE_DESCRIPTION:
        return f"Use at most {MAX_VOICE_DESCRIPTION} characters for the voice description."
    if llm.blocked(description) or llm.refers_to_minor(description):
        return "⛔ Only adult voices are supported. This description appears to refer to a minor."
    return ""


def check_name(name: str, slug: str = "") -> str:
    """Why `name` cannot name a character ("" = fine); `slug` is the character being edited."""
    from . import voices

    name = " ".join((name or "").split())
    if not name:
        return "Enter a character name."
    if len(name) > MAX_NAME:
        return f"Use at most {MAX_NAME} characters for the name."
    if llm.blocked(name):
        return llm.BLOCK_MSG
    if dialogue.is_gender_word(name):  # "Man", "Pan"... would confuse who speaks a line
        return "This word already means a woman or a man in prompts. Choose a different name."
    if any(c["name"].casefold() == name.casefold() and c["slug"] != slug for c in all_characters()):
        return f"A character named ‘{name}’ already exists. Choose another name."
    if voices.get(name, voices.all_voices()):  # "@Name" would mean two things in a prompt
        return f"‘{name}’ is the name of a voice in the voice library. Choose another name."
    from . import items

    if any(item["name"].casefold() == name.casefold() for item in items.all_items()):  # a scene text would mean both
        return f"A place or item is named ‘{name}’. Choose another name."
    return ""


def check_photo(path: str | Path, style: str = styles.REALISTIC, *, kind: str = PERSON) -> str:
    """The gate every reference photo passes ("" = it may be used): the vision model sees one adult (estimated age
    18 or more) - a real-looking one, or for a character of another style (app/styles.py) a person also as a 3D
    render or a drawing - and YuNet finds exactly one face. An animal or creature (kind CREATURE) passes
    `_creature_gate` instead."""
    from . import qc

    if kind == CREATURE:
        return _creature_gate(path, style)
    info = llm.check_photo(str(path)) if style == styles.REALISTIC else llm.check_photo(str(path), style=style)
    if info is None:
        return "The photo could not be checked. Try again."
    if info["person"] and info["age"] < 18:
        return f"⛔ The photo appears to show a minor (estimated age {info['age']}). Only adults are allowed."
    if not info["person"]:
        return "No clear person is visible in the photo."
    if not qc.rules(style).detects:
        return ""  # the face detector does not see faces of this style: the vision model's one adult decides
    try:
        faces = qc.count_faces(path)
    except Exception:  # noqa: BLE001 - no OpenCV / unreadable file: the photo is not verified
        faces = 0
    if faces != 1:
        return "The photo must show exactly one clear face."
    return ""


def _creature_gate(path: str | Path, style: str) -> str:
    """The gate of an animal or creature's photo ("" = it may be used): only in an animated style, and the vision
    model sees no person who looks under 18 (the owner's decision of 2026-11-16: nothing else about the picture is
    refused). No face is detected: the face detector knows human faces only."""
    if problem := creature_style_problem(style):
        return problem
    info = llm.check_creature_photo(str(path), style)
    if info is None:  # an answer that could not be read is asked again before the take is drawn again
        info = llm.check_creature_photo(str(path), style)
    if info is None:
        return "The photo could not be checked. Try again."
    if info["humans"] and info["youngest_human_age"] < 18:
        return (f"⛔ The photo appears to show a minor (estimated age {info['youngest_human_age']}). Only adults are "
                "allowed.")
    return ""


def same_person(path: str | Path, primary: str | Path, style: str = styles.REALISTIC, *, kind: str = PERSON) -> str:
    """Whether the photo shows the person of the character's primary photo ("" = it does). SFace similarity of at
    least qc.FACE_MATCH (the style's FaceRules.match), where faces count as the same character elsewhere (lip
    sync): real photos of one person in other light score lower than generated views, so the stricter FACE_MIN of
    those views would refuse them. An animal or creature (kind CREATURE) is never compared by face: ""."""
    from . import qc

    if kind == CREATURE:
        return ""
    if not qc.rules(style).detects:
        return ""  # no face of this style can be compared (qc.FaceRules.detects)
    try:
        score = qc.face_score(str(path), {"slug": f"draft-{Path(primary).stem}", "photos": [str(primary)]})
    except Exception:  # noqa: BLE001 - no OpenCV / unreadable file: the photo is not verified
        score = None
    if score is None:
        return "The face could not be compared with the first photo. Try another photo."
    if score < qc.rules(style).match:
        return ("This photo seems to show a different person than the first photo. Every photo of a character must "
                "show the same person.")
    return ""


def gate_models(progress_cb: Optional[ProgressCb] = None) -> Optional[str]:
    """Install what check_photo needs (the vision model and the face check); an error message or None."""
    from . import model_manager, qc

    return model_manager.ensure([llm.MODEL_ID, qc.MODEL_ID], progress_cb)


def normalize_photo(src: str | Path, dst: Path) -> str:
    """Copy a picture as an upright RGB PNG of at most PHOTO_SIDE pixels ("" = done, else why not)."""
    import warnings

    from PIL import Image, ImageOps

    try:
        if Path(src).stat().st_size > MAX_PHOTO_BYTES:
            return "The photo is larger than 50 MB."
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(src) as im:
                if im.width * im.height > MAX_PHOTO_PIXELS:
                    return "The photo has more than 40 megapixels."
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((PHOTO_SIDE, PHOTO_SIDE), Image.LANCZOS)
                if min(im.size) < MIN_PHOTO_SIDE:
                    return f"The photo is too small (at least {MIN_PHOTO_SIDE} pixels on each side)."
                dst.parent.mkdir(parents=True, exist_ok=True)
                im.save(dst, "PNG")
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return "The file is not a readable picture."
    return ""


# ----------------------------- drafts -----------------------------

def _draft_folder(draft_id: str) -> Path:
    if not isinstance(draft_id, str) or not _ID.fullmatch(draft_id):
        raise ValueError("Invalid character draft.")
    return DRAFTS_DIR / draft_id


def _manifest(draft_id: str) -> dict:
    """The draft's manifest; ValueError when the draft is gone (discarded, or left behind and removed)."""
    try:
        data = json.loads((_draft_folder(draft_id) / "draft.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("This character draft was closed. Start again.") from error
    if not isinstance(data, dict) or data.get("id") != draft_id:
        raise ValueError("This character draft was closed. Start again.")
    return data


def _write_manifest(data: dict) -> None:
    data["updated"] = time.time()
    atomic_text(_draft_folder(data["id"]) / "draft.json", json.dumps(data, ensure_ascii=False, indent=1))


def _draft_photos(draft_id: str, data: dict) -> List[Tuple[Path, dict]]:
    """(path, entry) of every photo of the draft whose file is still the one recorded."""
    folder = _draft_folder(draft_id)
    out = []
    for item in data.get("photos") or []:
        name = item.get("file") if isinstance(item, dict) else None
        if isinstance(name, str) and _DRAFT_PHOTO.fullmatch(name):
            path = _verified(folder / name, item.get("sha256"))
            if path is not None:
                out.append((path, item))
    return out


def _draft_voice(draft_id: str, data: dict) -> Optional[Path]:
    voice = data.get("voice")
    if not isinstance(voice, dict):
        return None
    return _verified(_draft_folder(draft_id) / "voice.wav", voice.get("sha256"))


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
    """An empty draft (a person); returns its id."""
    _clean_old_drafts()
    draft_id = uuid.uuid4().hex
    _draft_folder(draft_id).mkdir(parents=True)
    _write_manifest({"id": draft_id, "slug": "", "name": "", "gender": "woman", "kind": PERSON, "look": "",
                     "style": styles.DEFAULT, "photos": [], "voice": None, "created": time.time()})
    return draft_id


def _new_photo_name() -> str:
    return f"p_{uuid.uuid4().hex[:16]}.png"


def _add_photo_entry(folder: Path, source: Path, origin: str, gate: str) -> dict:
    """Copy `source` into the draft folder and return its manifest entry (not yet added to `data`); `gate` is the
    kind whose photo gate it passed."""
    name = _new_photo_name()
    shutil.copyfile(source, folder / name)
    return {"file": name, "sha256": _digest(folder / name), "origin": origin, "gate": gate}


def load_draft(slug: str) -> str:
    """A draft holding copies of the character's photos and voice, for editing it on the creator screen. Its photos
    count as passing the gate of the character's kind."""
    from . import voices

    ch = get(slug)
    if ch is None:
        raise ValueError("Character not found.")
    draft_id = new_draft()
    folder = _draft_folder(draft_id)
    with _DRAFT_LOCK:
        data = _manifest(draft_id)
        data.update(slug=slug, name=ch["name"], gender=ch["gender"], kind=ch["kind"], look=ch["look"],
                    style=ch["style"])
        data["photos"] = [_add_photo_entry(folder, Path(path), "kept", ch["kind"]) for path in ch["photos"]]
        info = ch.get("voice_info")
        library = None if info else voices.get(ch.get("voice") or "", voices.all_voices())
        source = Path(info["path"]) if info else Path(library.path) if library else None
        if source is not None and source.is_file():
            shutil.copyfile(source, folder / "voice.wav")
            data["voice"] = ({key: info[key] for key in info if key != "path"} if info else
                             {"origin": "preset" if library.builtin else "library", "preset": library.slug,
                              "description": library.description, "seconds": library.seconds,
                              "f0_hz": library.f0_hz, "consent": not library.builtin})
            data["voice"].update(file="voice.wav", sha256=_digest(folder / "voice.wav"))
        _write_manifest(data)
    return draft_id


def draft(draft_id: str) -> dict:
    """What the creator screen shows: the fields, the photo paths in order, the voice sample and its details."""
    data = _manifest(draft_id)
    voice = _draft_voice(draft_id, data)
    info = data.get("voice") if voice else None
    return {"id": draft_id, "slug": data.get("slug") or "", "name": data.get("name") or "",
            "gender": data.get("gender") or "woman", "kind": _kind(data.get("kind")), "look": data.get("look") or "",
            "style": styles.of(data.get("style")).id,
            "photos": [str(path) for path, _ in _draft_photos(draft_id, data)],
            "origins": [entry.get("origin", "") for _, entry in _draft_photos(draft_id, data)],
            "voice": str(voice) if voice else None,
            "voice_info": {k: v for k, v in (info or {}).items() if k not in ("file", "sha256")} or None,
            "busy": [job.id for job in _draft_jobs(draft_id)]}


def update_draft(draft_id: str, *, name: Optional[str] = None, gender: Optional[str] = None,
                 look: Optional[str] = None, style: Optional[str] = None, kind: Optional[str] = None) -> dict:
    """Remember the form's fields in the draft (a photo job reads the look, gender, style and kind from it). A new
    style applies to the photos made from then on; the draft's photos stay as they are. The kind changes only while
    the draft has no photos and makes none, since every photo passed the gate of the draft's kind, and an animal or
    creature needs a style outside the Realistic family (CREATURE_STYLE_MSG). Nothing is remembered on a ValueError."""
    with _DRAFT_LOCK:
        data = _manifest(draft_id)
        if name is not None:
            data["name"] = " ".join(str(name).split())[:MAX_NAME]
        if gender is not None:
            if gender not in GENDERS:
                raise ValueError("Choose woman or man.")
            data["gender"] = gender
        if look is not None:
            data["look"] = " ".join(str(look).split())[:MAX_LOOK]
        if style is not None:
            chosen = styles.get(style).id
            if chosen != data.get("style"):  # a stored character keeps working; only a change is checked
                problem = llm.style_problem(chosen)
                if problem:
                    raise ValueError(problem)
            data["style"] = chosen
        if kind is not None:
            if kind not in KINDS:
                raise ValueError("Choose a person or an animal or creature.")
            if kind != _kind(data.get("kind")):
                if _draft_photos(draft_id, data):
                    raise ValueError("Remove the photos before changing what kind of character this is.")
                if any(job.kind == PHOTO_KIND for job in _draft_jobs(draft_id)):
                    raise ValueError("Wait for the photos that are being made before changing what kind of character "
                                     "this is.")
            data["kind"] = kind
        if _kind(data.get("kind")) == CREATURE and (problem := creature_style_problem(data.get("style"))):
            raise ValueError(problem)
        _write_manifest(data)
    return draft(draft_id)


def discard_draft(draft_id: str) -> None:
    """Leave the creator screen: the draft and its files are removed. Its queued jobs are withdrawn and a running
    one is stopped: its result would land in a draft nobody can save any more."""
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
    """Add your own photo: it needs the consent box and passes the photo gate of the draft's kind (`check_photo`:
    one adult and one face, or one animal or creature without people); a person's photo must also show the person
    of the first photo."""
    if not consent:
        return False, "Confirm that the photo shows a fictional character or an adult who agreed to its use."
    data = _manifest(draft_id)
    if len(_draft_photos(draft_id, data)) >= MAX_PHOTOS:
        return False, f"A character has at most {MAX_PHOTOS} photos. Remove one first."
    problem = gate_models()
    if problem:
        return False, problem
    folder = _draft_folder(draft_id)
    staged = folder / f".upload_{uuid.uuid4().hex[:8]}.png"
    style = styles.of(data.get("style")).id
    kind = _kind(data.get("kind"))
    try:
        problem = normalize_photo(path, staged) or check_photo(staged, style, kind=kind)
        if problem:
            return False, problem
        photos = _draft_photos(draft_id, _manifest(draft_id))
        if kind == PERSON and photos and (problem := same_person(staged, photos[0][0], style)):
            return False, problem  # one character, one person (a creature is never compared by face)
        with _DRAFT_LOCK:
            data = _manifest(draft_id)
            if _kind(data.get("kind")) != kind:  # the form changed it while the photo was checked
                return False, "The kind of character changed while the photo was checked. Add the photo again."
            if len(_draft_photos(draft_id, data)) >= MAX_PHOTOS:
                return False, f"A character has at most {MAX_PHOTOS} photos. Remove one first."
            entry = _add_photo_entry(folder, staged, "uploaded", kind)
            entry["consent"] = True
            data["photos"] = [item for _, item in _draft_photos(draft_id, data)] + [entry]
            _write_manifest(data)
    finally:
        staged.unlink(missing_ok=True)
    return True, "Photo added."


def _set_voice(draft_id: str, source: Path, info: dict) -> None:
    """Make `source` (a prepared 48 kHz sample) the draft's voice."""
    from . import ffmpeg_utils, voices

    folder = _draft_folder(draft_id)
    with _DRAFT_LOCK:
        data = _manifest(draft_id)
        target = folder / "voice.wav"
        shutil.copyfile(source, target)
        seconds = ffmpeg_utils.probe_duration(target) or 0.0
        data["voice"] = {**info, "file": "voice.wav", "sha256": _digest(target), "seconds": round(seconds, 2),
                         "f0_hz": round(voices.measure_pitch(target))}
        _write_manifest(data)


def _prepared_sample(draft_id: str, source: str | Path) -> Tuple[Optional[Path], str]:
    """`source` as a library-style voice sample (48 kHz mono, silence trimmed, at most 12 s) in the draft folder."""
    from . import ffmpeg_utils, voices

    target = _draft_folder(draft_id) / f".sample_{uuid.uuid4().hex[:8]}.wav"
    ok, message = ffmpeg_utils.prepare_voice_sample(source, target, voices.SCAN_SECONDS, CLONE_SECONDS[1])
    if not ok:
        target.unlink(missing_ok=True)
        detail = message.strip().splitlines()[-1] if message.strip() else "unsupported audio"
        return None, (f"Could not read the sample ({detail[:120]}). Use a WAV, MP3, M4A, OGG or FLAC recording.")
    return target, ""


def clone_voice(draft_id: str, sample: Optional[str], consent: bool) -> Tuple[bool, str]:
    """The voice from a recording of 3-12 s of one person speaking (consent is needed for a real person's voice)."""
    from . import ffmpeg_utils, voices

    if not consent:
        return False, voices.CONSENT_MSG
    src = Path(sample) if sample else None
    if src is None or not src.is_file():
        return False, "Upload or record a voice sample first."
    if src.stat().st_size > voices.MAX_BYTES:
        return False, "The sample file is too large (over 200 MB). Use a shorter recording."
    _manifest(draft_id)
    prepared, message = _prepared_sample(draft_id, src)
    if prepared is None:
        return False, message
    try:
        seconds = ffmpeg_utils.probe_duration(prepared) or 0.0
        if seconds < CLONE_SECONDS[0]:
            return False, (f"Only {seconds:.1f} s of speech was found. Record 3–12 seconds of one person "
                           "speaking clearly, without music or long pauses.")
        _set_voice(draft_id, prepared, {"origin": "cloned", "description": "", "preset": "", "consent": True,
                                        "source": src.name[:120]})
    finally:
        prepared.unlink(missing_ok=True)
    return True, f"Voice cloned from {seconds:.1f} s of speech."


def set_designed_voice(draft_id: str, sample: str | Path, description: str) -> Tuple[bool, str]:
    """A voice sample designed from `description` (tts.design_voice) becomes the draft's voice."""
    _manifest(draft_id)
    prepared, message = _prepared_sample(draft_id, sample)
    if prepared is None:
        return False, message
    try:
        _set_voice(draft_id, prepared, {"origin": "designed", "preset": "", "consent": False,
                                        "description": " ".join(str(description).split())[:MAX_VOICE_DESCRIPTION]})
    finally:
        prepared.unlink(missing_ok=True)
    return True, "Voice ready."


def use_preset_voice(draft_id: str, voice_slug: str) -> Tuple[bool, str]:
    """Start from one of the bundled voices (a copy of its sample becomes the character's voice)."""
    from . import voices

    preset = next((v for v in voices.builtin() if v.slug == voice_slug), None)
    if preset is None:
        return False, "Choose one of the bundled voices."
    _set_voice(draft_id, Path(preset.path), {"origin": "preset", "preset": preset.slug,
                                             "description": preset.description, "consent": False})
    return True, f"Voice {preset.name} copied to this character."


# ----------------------------- jobs on a draft (GPU queue) -----------------------------

def _draft_jobs(draft_id: str) -> List[queue_manager.Job]:
    return [job for job in queue_manager.snapshot() if job.kind in (PHOTO_KIND, VOICE_KIND)
            and job.params.get("draft") == draft_id and job.status in (queue_manager.PENDING, queue_manager.RUNNING)]


def enqueue_photos(draft_id: str, count: int, *, keep_own: bool = False) -> queue_manager.Job:
    """Queue the generation of `count` photos (1-8) of one new character from the draft's look, gender, style and
    kind; with keep_own the uploaded photos stay and the new ones are views of the first of them. An animal or
    creature needs an animated style (CREATURE_STYLE_MSG)."""
    data = _manifest(draft_id)
    kind = _kind(data.get("kind"))
    count = int(count)
    if not MIN_PHOTOS <= count <= MAX_PHOTOS:
        raise ValueError(f"Choose {MIN_PHOTOS} to {MAX_PHOTOS} photos.")
    if kind == CREATURE and (problem := creature_style_problem(data.get("style"))):
        raise ValueError(problem)
    if not keep_own and (problem := check_look(data.get("look", ""), kind)):
        raise ValueError(problem)
    own = sum(entry.get("origin") in ("uploaded", "kept") for _, entry in _draft_photos(draft_id, data))
    if keep_own and not own:
        raise ValueError("Upload a photo first, or generate a new character.")
    if keep_own and count <= own:  # the views are added up to `count`: nothing would be made
        raise ValueError(f"The character already has {own} of its own photo{'s' if own != 1 else ''}. Raise the "
                         "number of photos to add views of the first one, or untick “Keep my own photos” to "
                         "generate a new character.")
    label = f"Character photos: {data.get('name') or 'new character'} ({count})"
    return queue_manager.add(PHOTO_KIND, label, {"draft": draft_id, "count": count, "keep_own": bool(keep_own),
                                                 "look": data.get("look", ""), "gender": data.get("gender"),
                                                 "style": styles.of(data.get("style")).id, "kind": kind})


def enqueue_regenerate(draft_id: str, index: int) -> queue_manager.Job:
    """Queue a new take of one generated photo (the others stay). An animal or creature needs an animated style
    (CREATURE_STYLE_MSG)."""
    data = _manifest(draft_id)
    kind = _kind(data.get("kind"))
    photos = _draft_photos(draft_id, data)
    if not 0 <= index < len(photos):
        raise ValueError("That photo is no longer in the draft.")
    if photos[index][1].get("origin") != "generated":
        raise ValueError("Only generated photos can be generated again. Remove this one or upload another.")
    if kind == CREATURE and (problem := creature_style_problem(data.get("style"))):
        raise ValueError(problem)
    if len(photos) == 1 and (problem := check_look(data.get("look", ""), kind)):  # the only photo: a new character
        raise ValueError(problem)                                                  # from the look (char_photos)
    return queue_manager.add(PHOTO_KIND, f"Character photo: {data.get('name') or 'new character'}",
                             {"draft": draft_id, "count": 1, "index": int(index), "look": data.get("look", ""),
                              "gender": data.get("gender"), "kind": kind})


def enqueue_voice_design(draft_id: str, description: str) -> queue_manager.Job:
    """Queue VoxCPM2 voice design from a description of the voice."""
    data = _manifest(draft_id)
    description = " ".join((description or "").split())
    if problem := check_voice_description(description, _kind(data.get("kind"))):
        raise ValueError(problem)
    return queue_manager.add(VOICE_KIND, f"Character voice: {data.get('name') or 'new character'}",
                             {"draft": draft_id, "description": description, "gender": data.get("gender")})


def _voice_description(description: str, gender: str) -> str:
    """The description VoxCPM2 gets: it always says that the speaker is an adult, and names the character's
    gender when the text does not ("girl" and "boy" do not count: they could make a child's voice)."""
    if re.search(r"\badults?\b", description, re.I):
        return description
    if re.search(r"\b(?:wom[ae]n|m[ae]n|female|male|lady|gentleman)\b", description, re.I):
        return f"An adult, {description}"
    return f"An adult {'man' if gender == 'man' else 'woman'}, {description}"


def run_voice_job(p: dict, cb) -> tuple:
    """Queue runner: design the voice into the draft."""
    from . import tts

    draft_id = p.get("draft", "")
    try:
        data = _manifest(draft_id)
    except ValueError as error:
        return False, [], [], str(error)
    description = " ".join(str(p.get("description") or "").split())
    # the runner trusts no queued params, as enqueue does not
    if problem := check_voice_description(description, _kind(data.get("kind"))):
        return False, [], [], problem
    folder = _draft_folder(draft_id)
    raw = folder / f".designed_{uuid.uuid4().hex[:8]}.wav"
    try:
        ok, message = tts.design_voice(_voice_description(description, p.get("gender") or "woman"), raw,
                                       progress_cb=lambda f, d: cb(0.9 * f, d))
        if not ok:
            return False, [], [], message
        ok, message = set_designed_voice(draft_id, raw, description)
        if not ok:
            return False, [], [], message
    except ValueError as error:  # the draft was closed while the voice was being made
        return False, [], [], str(error)
    finally:
        raw.unlink(missing_ok=True)
    cb(1.0, "Voice ready")
    return True, [], ["The voice is ready in the character draft."], ""


def run_photo_job(p: dict, cb) -> tuple:
    """Queue runner: generate photos into the draft (app/char_photos.py)."""
    from . import char_photos

    draft_id = p.get("draft", "")
    try:
        _manifest(draft_id)
        if p.get("index") is not None:
            notes = char_photos.regenerate(draft_id, int(p["index"]), cb)
        else:
            notes = char_photos.generate(draft_id, int(p.get("count") or 1), bool(p.get("keep_own")), cb)
    except ValueError as error:
        return False, [], [], str(error)
    cb(1.0, "Photos ready")
    return True, [], notes, ""


def replace_photos(draft_id: str, items: Sequence[object]) -> None:
    """Called by the photo job: the draft's photos become `items` in this order, each an entry of the manifest
    (a photo that stays) or (path in the draft folder, origin) for a new file; files no longer listed are
    removed. A new file passed the gate of the draft's kind (the job checks every photo with it, and the kind
    cannot change while a photo job of the draft waits or runs: update_draft)."""
    folder = _draft_folder(draft_id)
    with _DRAFT_LOCK:
        data = _manifest(draft_id)
        gate = _kind(data.get("kind"))
        entries = [dict(item) if isinstance(item, dict) else
                   {"file": Path(item[0]).name, "sha256": _digest(Path(item[0])), "origin": item[1], "gate": gate}
                   for item in items][:MAX_PHOTOS]
        listed = {entry["file"] for entry in entries}
        data["photos"] = entries
        _write_manifest(data)
        for stray in folder.glob("p_*.png"):
            if stray.name not in listed:
                stray.unlink(missing_ok=True)


def draft_photos(draft_id: str) -> List[Tuple[Path, dict]]:
    """(path, manifest entry) of the draft's photos in order (for the photo job)."""
    return _draft_photos(draft_id, _manifest(draft_id))


def draft_folder(draft_id: str) -> Path:
    return _draft_folder(draft_id)


def new_photo_path(draft_id: str) -> Path:
    """A fresh file name in the draft folder for a generated photo."""
    return _draft_folder(draft_id) / _new_photo_name()


# ----------------------------- saving and deleting -----------------------------

def _unique_slug(name: str) -> str:
    base = slugify(name)
    slug, i = base, 2
    while (DIR / slug).exists():
        slug, i = f"{base}_{i}", i + 1
    return slug


def _stage(staging: Path, data: dict, record: dict, photos: List[Path], voice: Path) -> None:
    """The character's new folder, complete, in `staging` beside the drafts (the same volume as characters/)."""
    (staging / "photos").mkdir(parents=True)
    (staging / "voice").mkdir()
    entries = []
    for number, (path, entry) in enumerate(zip(photos, data["photos"]), 1):
        target = staging / "photos" / f"{number:02d}.png"
        shutil.copyfile(path, target)
        entries.append({"file": target.name, "sha256": _digest(target),
                        "origin": "uploaded" if entry.get("origin") == "uploaded" else entry.get("origin", "kept")})
    shutil.copyfile(voice, staging / "voice" / "sample.wav")
    voice_entry = {key: value for key, value in data["voice"].items() if key not in ("file", "sha256")}
    voice_entry.update(file="sample.wav", sha256=_digest(staging / "voice" / "sample.wav"))
    record.update(photos=entries, voice=voice_entry)
    atomic_text(staging / "character.json", json.dumps(record, ensure_ascii=False, indent=1))


def _dispose(old: Path) -> bool:
    """Remove a character folder that a save replaced. A character in the current format is deleted: everything
    it held was copied into the draft and saved again, or removed there by the user. An older character's folder
    goes to the Recycle Bin, because it holds files the current format does not carry over (the photos and
    captions of the removed LoRA training, its preview) that the creator never showed. False when the Recycle
    Bin refused it: the folder then stays where it is."""
    try:
        current = json.loads((old / "character.json").read_text(encoding="utf-8")).get("format") == FORMAT
    except (OSError, ValueError, AttributeError):
        current = False  # unreadable: kept for the user like an older character
    if current:
        shutil.rmtree(old, ignore_errors=True)
        return True
    return trash.move(old)


def _swap_in(staging: Path, slug: str) -> str:
    """Put the staged folder in place of characters/<slug>; an existing folder is replaced, never half (a crash
    between the two renames is settled by `startup`). Returns a note for the user ("" = none)."""
    target = _folder(slug)
    DIR.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        staging.replace(target)
        return ""
    backup = DIR / f".replaced_{slug}_{uuid.uuid4().hex[:8]}"
    target.replace(backup)
    try:
        staging.replace(target)
    except OSError:
        backup.replace(target)
        raise
    if _dispose(backup):
        return ""
    return (f"Its earlier files could not be moved to the Recycle Bin. They are kept in the hidden folder "
            f"characters\\{backup.name}, and the next start of the app tries again.")


def save_draft(draft_id: str, *, name: Optional[str] = None, gender: Optional[str] = None,
               look: Optional[str] = None, style: Optional[str] = None,
               kind: Optional[str] = None) -> Tuple[Optional[dict], str]:
    """Save the draft as a character: photos, voice, name, look, gender, style and kind together. Returns (profile,
    message); nothing is written on an error. A draft made by load_draft replaces that character (same slug), or
    becomes a new character when that one was deleted while it was open. The style is stored only when it is not
    Realistic and the kind only for an animal or creature, so a realistic person's file keeps the fields it always
    had. Everything is checked again: an animal or creature only in an animated style and with a fully grown look,
    and every photo passed the gate of the draft's kind."""
    if any(value is not None for value in (name, gender, look, style, kind)):
        update_draft(draft_id, name=name, gender=gender, look=look, style=style, kind=kind)
    if _draft_jobs(draft_id):
        return None, "Wait for the photos or the voice that are still being made."
    with _SAVE_LOCK, _DRAFT_LOCK:
        data = _manifest(draft_id)
        slug = data.get("slug") or ""
        if slug and get(slug) is None:
            slug = ""  # deleted or archived while it was open (the creator then says "New character"): a new one
        kind = data.get("kind", PERSON)  # a draft without one was a person's
        problem = (check_name(data.get("name", ""), slug)
                   or ("" if kind in KINDS else "Choose a person or an animal or creature.")
                   or ("" if kind == PERSON else creature_style_problem(data.get("style")))
                   or check_look(data.get("look", ""), kind)
                   # a voice designed before the draft became an animal or creature is checked as one's too
                   or (check_voice_description(str((data.get("voice") or {}).get("description") or ""), kind)
                       if kind == CREATURE and (data.get("voice") or {}).get("origin") == "designed" else "")
                   or ("" if data.get("gender") in GENDERS else "Choose woman or man."))
        if problem:
            return None, problem
        photos = _draft_photos(draft_id, data)
        if not MIN_PHOTOS <= len(photos) <= MAX_PHOTOS:
            return None, f"Add {MIN_PHOTOS} to {MAX_PHOTOS} reference photos."
        if any(entry.get("gate", PERSON) != kind for _, entry in photos):
            return None, ("Some photos were checked for another kind of character. Remove them, then upload or "
                          "generate them again.")
        voice = _draft_voice(draft_id, data)
        if voice is None:
            return None, "Give the character a voice: design one, clone a recording or pick a bundled voice."
        previous = json.loads(_file(slug).read_text(encoding="utf-8")) if slug else {}
        slug = slug or _unique_slug(data["name"])
        now = time.time()
        record = {"format": FORMAT, "name": data["name"], "slug": slug, "gender": data["gender"],
                  "look": data["look"], "created": previous.get("created", now), "updated": now}
        if styles.of(data.get("style")).id != styles.REALISTIC:
            record["style"] = styles.of(data.get("style")).id
        if kind == CREATURE:
            record["kind"] = CREATURE
        data["photos"] = [entry for _, entry in photos]
        staging = DRAFTS_DIR / f".staging_{uuid.uuid4().hex[:12]}"
        try:
            _stage(staging, data, record, [path for path, _ in photos], voice)
            note = _swap_in(staging, slug)
        except OSError as error:
            return None, f"Could not save the character ({error}). Close files that use it and try again."
        finally:
            shutil.rmtree(staging, ignore_errors=True)  # gone already once it is in place
    discard_draft(draft_id)
    profile = get(slug)
    count = f"{len(photos)} photo{'s' if len(photos) != 1 else ''}"
    return profile, " ".join(part for part in (
        f"Character ‘{record['name']}’ saved with {count} and its own voice.", note) if part)


def delete(slug: str) -> str:
    ch = get(slug)
    if not ch:
        return "Character not found."
    if not trash.move(_folder(slug)):
        return "Could not move the character folder to the Recycle Bin. Close files that use it and try again."
    from . import universes

    universes.forget(slug)  # it leaves every universe it was in
    if prefs.load().get("character") == slug:
        prefs.save(character="")
    return f"Character ‘{ch['name']}’ moved to the Recycle Bin."


queue_manager.register(PHOTO_KIND, run_photo_job)
queue_manager.register(VOICE_KIND, run_voice_job)
