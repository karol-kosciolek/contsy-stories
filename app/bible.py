# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The series bible as data the code enforces (docs/QUALITY_ARCHITECTURE.md, layer 2).

app/series.py stores the bible as the owner and the agent wrote it: design sheets with free text. This module turns it
into what the shot contract (app/shot_spec.py) and the shot check (app/shot_qc.py) need:

- a character: its canonical description (the words every shot's prompt repeats), its aliases (Barley = "the owl"),
  the accessories it must always wear with their colours ("yellow bobble hat") and its body colours;
- a place (a location or a prop): its canonical description without negations and its reference picture (the plate,
  the library item's first photo).

A sheet may carry these fields explicitly (`canonical`, `aliases`, `must_have`, `palette`); otherwise they are derived
from the sheet's description, colours and outfit, the same way every time. Nothing here writes to the series.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import palette

# Words that name a kind of animal or creature: an alias of the one character of that kind in the cast.
SPECIES = ("owl", "fox", "rabbit", "bunny", "hare", "tortoise", "turtle", "bear", "cat", "kitten", "dog", "puppy",
           "mouse", "hedgehog", "squirrel", "badger", "otter", "duck", "frog", "pig", "piglet", "deer", "fawn", "lamb",
           "sheep", "goat", "cow", "horse", "pony", "donkey", "elephant", "lion", "tiger", "monkey", "panda", "koala",
           "penguin", "chick", "hen", "bird", "robin", "sparrow", "crow", "raven", "parrot", "bee", "snail", "ladybug",
           "butterfly", "caterpillar", "worm", "spider", "dragon", "dinosaur", "unicorn", "mole", "raccoon", "beaver",
           "wolf", "seal", "whale", "fish", "octopus", "crab", "creature", "monster", "robot")
# Words for people: a person in a picture of a series whose cast has none is a stranger (defect A).
PEOPLE = ("person", "people", "human", "humans", "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
          "child", "children", "kid", "kids", "baby", "babies", "toddler", "lady", "gentleman", "grandma", "grandpa",
          "farmer", "king", "queen", "prince", "princess", "crowd")
# Things a character wears or carries: a coloured one is an accessory the character must always show.
ACCESSORIES = ("hat", "cap", "beanie", "bow tie", "bowtie", "tie", "scarf", "glasses", "spectacles", "cardigan",
               "overalls", "boots", "bag", "backpack", "ribbon", "bow", "collar", "vest", "jacket", "dress", "shirt",
               "sweater", "jumper", "apron", "cape", "crown", "necklace", "mittens", "gloves", "socks", "shoes")
BODY = ("body", "fur", "feathers", "skin", "shell", "scales", "coat", "head")
_NEGATION = re.compile(r"\b(?:no|not|never|without|nobody|nothing|none|neither|nor)\b|n't\b", re.I)
_PAREN = re.compile(r"\s*\([^()]*\)")
_CLAUSE = re.compile(r"([^,.;:!?]*)([,.;:!?]|$)")
_PLURAL = ("glasses", "spectacles", "overalls", "boots", "mittens", "gloves", "socks", "shoes")
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z]+(?:[-'][a-z]+)*")


@dataclass(frozen=True)
class Accessory:
    """Something a character always wears, with the colour the check looks for."""
    name: str            # "bobble hat"
    color: str           # "yellow"

    @property
    def ranges(self) -> Tuple[palette.HsvRange, ...]:
        return palette.ranges(self.color)


@dataclass(frozen=True)
class CharacterSpec:
    slug: str
    name: str
    kind: str                         # "creature" or "person"
    canonical: str                    # the words every prompt repeats
    aliases: Tuple[str, ...]
    must_have: Tuple[Accessory, ...]
    body: str                         # body colour words ("teal blue"), "" when unknown
    species: str = ""
    photo: Optional[str] = None       # the reference photo (front view)
    extra_photos: Tuple[str, ...] = ()  # other views (3/4, profile) of the same character

    @property
    def body_ranges(self) -> Tuple[palette.HsvRange, ...]:
        """The body's main colour only: the first word of a compound ("teal-blue" -> teal, "lavender-grey" ->
        lavender), since the second one is also the sky, the water or the shadows around the character."""
        found = palette.words(self.body)
        return palette.ranges(found[0]) if found else ()

    def sentence(self) -> str:
        """The character for a prompt: "<Name> is <canonical>"."""
        text = self.canonical
        first = text.split(" ", 1)[0]
        if first in ("A", "An", "The"):
            text = first.lower() + text[len(first):]
        return f"{self.name} is {text}"


@dataclass(frozen=True)
class PlaceSpec:
    slug: str
    name: str
    kind: str                         # "location" or "prop"
    canonical: str
    plate: Optional[str] = None       # the reference picture
    aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Bible:
    """What a series (or a film's cast) allows: its characters, places and props, and its rules as data."""
    characters: Tuple[CharacterSpec, ...]
    places: Tuple[PlaceSpec, ...] = ()
    narrator_only: bool = False       # nobody on screen speaks: no lip sync, no "talks"
    only_named_cast: bool = True      # nobody but the named characters on screen (no people, no other animals)
    style: str = ""
    max_cast: int = 3
    series_id: str = ""

    def character(self, slug: str) -> Optional[CharacterSpec]:
        return next((one for one in self.characters if one.slug == slug), None)

    def place(self, key: str) -> Optional[PlaceSpec]:
        key = (key or "").strip().casefold()
        return next((one for one in self.places if key in (one.slug.casefold(), one.name.casefold())), None)

    @property
    def locations(self) -> Tuple[PlaceSpec, ...]:
        return tuple(one for one in self.places if one.kind == "location")

    @property
    def props(self) -> Tuple[PlaceSpec, ...]:
        return tuple(one for one in self.places if one.kind == "prop")


def positive(text: str) -> str:
    """The text without its sentences and clauses that say what is NOT there ("No people, no text."): the video model
    has no negative prompt, so "no boy" only puts a boy in its mind (defect A)."""
    source = (text or "").strip()
    text = _PAREN.sub(lambda match: "" if _NEGATION.search(match.group(0)) else match.group(0), source)
    out: List[str] = []
    for clause, stop in _CLAUSE.findall(text):
        if not clause.strip():
            if stop and out and stop in ".;:!?" and out[-1][-1:] in ",;:":
                out[-1] = out[-1][:-1] + stop
            continue
        if _NEGATION.search(clause):
            if stop and stop in ".;!?" and out and out[-1][-1:] in ",;:":
                out[-1] = out[-1][:-1] + stop  # "a, no b." -> "a."
            continue
        out.append(clause + stop)
    kept = re.sub(r"\s+([,.;:])", r"\1", _SPACE.sub(" ", "".join(out))).strip(" ,;:")
    if kept and source[:1].isupper():
        kept = kept[0].upper() + kept[1:]
    return kept


def _clip(text: str, words: int) -> str:
    """At most about `words` words, cut at a comma when the text is longer."""
    text = (text or "").strip()
    if len(text.split()) <= words:
        return text
    kept = ""
    for clause in text.split(","):
        candidate = f"{kept},{clause}" if kept else clause
        if len(candidate.split()) > words:
            break
        kept = candidate
    return (kept or " ".join(text.split()[:words])).strip().rstrip(",;:")


def _species(text: str) -> str:
    """The first word of the description that names a kind of animal or creature ("owl")."""
    found = [word for word in _WORD.findall((text or "").lower()) if word in SPECIES]
    return next((word for word in found if word not in ("creature", "monster")), found[0] if found else "")


def _parts(colors: str) -> List[str]:
    return [part.strip() for part in re.split(r"[,;]| and ", colors or "") if part.strip()]


def _accessories(sheet: dict) -> Tuple[Accessory, ...]:
    """Accessories with a colour, from the sheet's colours, outfit and marks ("yellow bobble hat", "red bow tie")."""
    explicit = sheet.get("must_have")
    if isinstance(explicit, list):
        return tuple(Accessory(str(row.get("name", "")).strip(), str(row.get("color", "")).strip())
                     for row in explicit if isinstance(row, dict) and row.get("name") and palette.words(row.get("color")))
    found: Dict[str, Accessory] = {}
    for source in (sheet.get("colors", ""), sheet.get("outfit", ""), sheet.get("marks", ""),
                   sheet.get("description", "") or sheet.get("look", "")):
        for part in _parts(source):
            low = part.lower()
            noun = next((item for item in sorted(ACCESSORIES, key=len, reverse=True)
                         if re.search(rf"\b{re.escape(item)}\b", low)), "")
            color = palette.words(low)
            if noun and color and noun not in found:
                words = [word for word in _WORD.findall(low) if word not in ("a", "an", "the", "small", "big", "little",
                                                                             "always", "worn", "round", "with")]
                start = next((i for i, word in enumerate(words) if word in color), 0)
                name = " ".join(word for word in words[start:] if word not in color)
                name = name[:name.find(noun) + len(noun)] if noun in name else noun
                found[noun] = Accessory(name.strip() or noun, " ".join(dict.fromkeys(color)))
    return tuple(found.values())


def _body(sheet: dict) -> str:
    explicit = sheet.get("palette")
    if isinstance(explicit, str) and palette.words(explicit):
        return " ".join(palette.words(explicit))
    for part in _parts(sheet.get("colors", "")):
        low = part.lower()
        if any(re.search(rf"\b{word}\b", low) for word in BODY) and palette.words(low):
            return " ".join(dict.fromkeys(palette.words(low)))
    first = (sheet.get("description") or sheet.get("look") or "").split(":")[0]
    found = [word for word in palette.words(first) if word not in ("cream", "white", "black")]
    return " ".join(dict.fromkeys(found))


def _canonical(sheet: dict, accessories: Sequence[Accessory]) -> str:
    """The character's canonical words: the sheet's own, or its description's lead (up to the colon) with its colours
    and the accessories it always wears. A belly patch is said to be on the front only (defect C: Lumi's patch on his
    back)."""
    if isinstance(sheet.get("canonical"), str) and sheet["canonical"].strip():
        return positive(sheet["canonical"].strip()).rstrip(".")
    text = positive(sheet.get("description") or sheet.get("look") or "")
    lead = text.split(":")[0].strip().rstrip(".")
    lead = _clip(lead, 22)
    if lead and lead.split()[0].lower() not in ("a", "an", "the"):
        lead = f"a {lead[0].lower()}{lead[1:]}"
    details = []
    colors = (sheet.get("colors") or "").strip()
    if colors:
        details.append(colors.rstrip("."))
    elif ":" in text:
        details.append(_clip(text.split(":", 1)[1].strip().rstrip("."), 18))
    worn = [f"{one.color} {one.name}" for one in accessories]
    sentence = lead
    if details:
        sentence += f" ({'; '.join(details)})"
    if worn:
        sentence += ", always wearing " + " and ".join(
            item if item.split()[-1] in _PLURAL else f"a {item}" for item in worn)
    if re.search(r"\bbelly\b", (colors + text).lower()):
        sentence += ", the belly patch on the front only and a plain back"
    return _SPACE.sub(" ", sentence).strip()


def character_spec(sheet: dict, *, photo: Optional[str] = None, extra_photos: Sequence[str] = ()) -> CharacterSpec:
    """A series cast sheet (or a library character: name, slug, look, kind) as a CharacterSpec."""
    name = str(sheet.get("name") or sheet.get("slug") or "").strip()
    slug = str(sheet.get("slug") or "").strip() or re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    accessories = _accessories(sheet)
    species = _species(sheet.get("description") or sheet.get("look") or "")
    aliases = [name]
    if isinstance(sheet.get("aliases"), list):
        aliases += [str(one).strip() for one in sheet["aliases"] if str(one).strip()]
    if species:
        aliases.append(species)
    kind = "creature" if sheet.get("kind") == "creature" else "person"
    return CharacterSpec(slug=slug, name=name, kind=kind, canonical=_canonical(sheet, accessories),
                         aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)), must_have=accessories,
                         body=_body(sheet), species=species, photo=photo, extra_photos=tuple(extra_photos))


def _unique_aliases(specs: Sequence[CharacterSpec]) -> Tuple[CharacterSpec, ...]:
    """Drop an alias two characters share ("creature" for two creatures): it names neither for sure."""
    counts: Dict[str, int] = {}
    for spec in specs:
        for alias in spec.aliases:
            counts[alias.casefold()] = counts.get(alias.casefold(), 0) + 1
    out = []
    for spec in specs:
        kept = tuple(alias for alias in spec.aliases if counts[alias.casefold()] == 1 or alias == spec.name)
        out.append(CharacterSpec(**{**spec.__dict__, "aliases": kept}))
    return tuple(out)


def place_spec(row: dict, *, plate: Optional[str] = None) -> PlaceSpec:
    name = str(row.get("name") or row.get("slug") or "").strip()
    kind = "prop" if row.get("kind") in ("prop", "item", "object") else "location"
    aliases = [name] + [str(one) for one in row.get("aliases") or [] if str(one).strip()]
    return PlaceSpec(slug=str(row.get("slug") or "").strip(), name=name, kind=kind,
                     canonical=positive(row.get("canonical") or row.get("description") or row.get("look") or ""),
                     plate=plate, aliases=tuple(dict.fromkeys(aliases)))


_NARRATOR_ONLY = re.compile(r"\bnarrator[- ]only\b|\bcharacters never speak\b|\bnever speak\b", re.I)
_ONLY_NAMED = re.compile(r"\bonly the named characters\b|\bexactly the named characters\b|\bno other animals or people\b"
                         r"|\bno humans\b|\bno extra (?:animals|characters)\b", re.I)


def from_series(data: dict, *, photos=None, plates=None) -> Bible:
    """The bible of a series (series.get): its cast with the photos of their library characters (`photos`: slug ->
    list of paths, the front view first; looked up in the library when None), its places and props with their plates
    (`plates`: slug -> path) and its rules as flags."""
    if photos is None or plates is None:
        found_photos, found_plates = _library_pictures(data)
        photos = found_photos if photos is None else photos
        plates = found_plates if plates is None else plates
    specs = []
    for sheet in data.get("characters") or []:
        files = list((photos or {}).get(sheet.get("slug") or "", []))
        specs.append(character_spec(sheet, photo=files[0] if files else None, extra_photos=files[1:]))
    places = tuple(place_spec(row, plate=(plates or {}).get(row.get("slug") or "")) for row in data.get("places") or [])
    rules = " ".join(str(data.get(key) or "") for key in ("rules", "narration_style"))
    return Bible(characters=_unique_aliases(specs), places=places,
                 narrator_only=data.get("narration") == "narrator" or bool(_NARRATOR_ONLY.search(rules)),
                 only_named_cast=True if _ONLY_NAMED.search(rules) else bool(data.get("audience") == "kids"),
                 style=str(data.get("look") or ""), series_id=str(data.get("id") or ""))


def from_profiles(profiles: Sequence[dict], items: Sequence[dict] = (), *, narrator_only: bool = False) -> Bible:
    """A bible for a film outside a series: the library characters (characters.get profiles) and items it uses."""
    specs = []
    for profile in profiles:
        files = [str(path) for path in profile.get("photos") or [] if path]
        specs.append(character_spec(profile, photo=files[0] if files else None, extra_photos=files[1:3]))
    places = []
    for item in items:
        photos = [str(path) for path in item.get("photos") or [] if path]
        places.append(place_spec({**item, "kind": "location" if item.get("kind") == "place" else "prop"},
                                 plate=photos[0] if photos else None))
    return Bible(characters=_unique_aliases(specs), places=tuple(places), narrator_only=narrator_only,
                 only_named_cast=False)


def _library_pictures(data: dict) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """The reference photos of the series' library characters and the first photo of its library places."""
    photos: Dict[str, List[str]] = {}
    plates: Dict[str, str] = {}
    try:
        from . import characters, items
    except ImportError:  # pragma: no cover - the app package is always there
        return photos, plates
    for sheet in data.get("characters") or []:
        slug = sheet.get("slug")
        try:
            profile = characters.get(slug) if slug else None
        except (OSError, ValueError):
            profile = None
        if profile:
            files = [str(path) for path in characters.reference_photos(profile) if Path(path).is_file()]
            if files:
                photos[slug] = files
    for row in data.get("places") or []:
        slug = row.get("slug")
        try:
            item = items.get(slug) if slug else None
            main = items.main_photo(item) if item else None
        except (OSError, ValueError):
            main = None
        if main is not None and Path(main).is_file():
            plates[slug] = str(main)
    return photos, plates


def for_series(series_id: str) -> Optional[Bible]:
    """The bible of a saved series, or None."""
    from . import series

    data = series.get(series_id)
    return from_series(data) if data else None


def series_of_project(project_id: str) -> Optional[str]:
    """The id of the series a Timeline project belongs to: an episode that lists it, else the episode run that
    created it (agent_store: run.json "series" and its created resources). None for a film outside any series."""
    from . import agent_store, series

    for data in series.all_series():
        for episode in series.episodes(data["id"]):
            if project_id in episode["projects"]:
                return data["id"]
    for record in agent_store.list_runs():
        bound = record.get("series")
        created = (record.get("resources") or {}).get("created") or []
        if isinstance(bound, dict) and f"project:{project_id}" in created and series.valid_id(bound.get("series")):
            return bound["series"]
    return None


def for_project(project_id: str) -> Optional[Bible]:
    """The bible of the series a project belongs to, or None."""
    found = series_of_project(project_id)
    return for_series(found) if found else None
