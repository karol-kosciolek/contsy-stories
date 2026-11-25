# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Obrazy pomocnicze (referencje) i ujecia w prompcie wideo.

Uzytkownik dolacza zdjecia (np. bluza, osoba, budynek) i wskazuje je w opisie numerem:
    A young man wearing the hoodie [1] walks on a rooftop.
    Ujęcie 2: he jumps onto the glass building [2].
Kazde ujecie (tekst po "Ujęcie N:" / "Shot N:") dostaje klatke startowa zlozona przez
Qwen-Image-Edit-2511 z obrazow, ktore wskazuje (do 10; model bierze 3 naraz, wiec wiecej
sklada w przebiegach - compose_refs) - z zachowaniem wzoru, kolorow i logo - a Wan 2.2
ozywia ja jako wideo. Kolejne ujecie sklada sie na bazie ostatniej
klatki poprzedniego, wiec osoba i stroj zostaja te same. Bez numerow wszystkie obrazy trafiaja
do pierwszego ujecia.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from engine.api import Engine

from . import styles

MAX_IMAGES = 3  # Qwen-Image-Edit-2511 takes up to 3 pictures per pass (engine.pipelines.edit.MAX_IMAGES)
MAX_REFS = 10   # obrazow pomocniczych na zadanie (wiecej niz 3 - skladanie w przebiegach)

# [1] albo @1 (numer obrazu pomocniczego)
_MARK = re.compile(r"\s*(?:\[(\d{1,2})\]|@(\d{1,2})\b)")
# "Ujęcie 2:", "Ujecie 2 -", "Shot 2:", "Scena 2:", "Scene 2." na poczatku linii
# Na poczatku linii (takze "Ujęcie 2." z kropka) albo w srodku tekstu z dwukropkiem/myslnikiem -
# wpisane w jednej linii "... okno. Ujęcie 2: wychodzi ..." sklejalo caly opis w jedno ujecie
_SHOT = re.compile(r"(?:^\s*(?:uj[ęe]cie|shot|scen[ae])\s*(\d+)\s*[:.\-–—]|"
                   r"(?<!\w)(?:uj[ęe]cie|shot|scen[ae])\s*(\d+)\s*[:\-–—])\s*", re.I | re.M)

# Wskazowki dla modelu edycji: wierne odwzorowanie referencji, jedna scena w stylu filmu, nie kolaz
_FAITHFUL = ("Every person from the reference images "
             "stays exactly the same person: same face, facial features, hairstyle, hair color, "
             "skin tone and body - only pose, expression, clothing and setting change as described. "
             "Keep the exact design, colors, patterns, logos, text and proportions of the items from "
             "the reference images.")


def keep(style: str = styles.REALISTIC) -> str:
    """The composition rule in a style: its opening (styles.Style.keep), then the faithfulness to the references."""
    return f"{styles.get(style).keep} {_FAITHFUL}"


_KEEP = keep()  # the realistic rule, as it always read
_CONTINUE = ("Image 1 is the previous moment of the same video: keep the same person, face, "
             "outfit, lighting and style.")
_STEP = ("Image 1 is the picture made so far: keep its scene, people, faces, poses, framing "
         "and everything already in it unchanged; only add or apply what comes from the other "
         "images.")
# The base is the character's own scene (characters.render, composed from its reference photo): it is the
# main person, and the reference images add things, places and other people
CHARACTER = ("Image 1 shows the main person: keep exactly her or his face, facial features, "
             "hairstyle, hair color, skin tone and body from image 1. Do not tan, darken, "
             "lighten or restyle her or him.")
# The same for an animal or creature character (app/characters.py, kind "creature"): it is no person
CREATURE_CHARACTER = ("Image 1 shows the main character, an animal or creature (not a human): keep exactly the same "
                      "kind of animal or creature, face, body shape, colors and markings from image 1, and its outfit "
                      "exactly as image 1 shows it. Do not change or restyle it.")
# How a composition names an animal or creature character (cast_lead) and what it keeps of one
_CREATURE = "an animal or creature character (not a human)"
_KEEP_CREATURE = ("Keep the creature from image {n} exactly as it is there: the same kind of animal or creature, "
                  "face, body shape, colors and markings, and its outfit exactly as image {n} shows it.")
# Na poczatku promptu - model edycji najmocniej slucha pierwszych zdan
_IDENTITY = ("The person must be exactly the same person as in the reference image: identical "
             "face, skin tone, hair color, hairstyle, eye color and body. Do not tan, darken, "
             "lighten or restyle her or him.")

# Opisy wygladu, ktore przy osobie ze zdjecia wygrywaly ze zdjeciem (test: blada modelka
# z platynowymi wlosami + "deep tan, sun-kissed skin" w prompcie = opalona blondynka)
_LOOK = [
    (r"\b(?:a |an )?\d{2}[- ]year[- ]old\b", ""),
    (r"\b(?:aged|age) \d{2}\b", ""),
    (r"\bin (?:her|his) (?:early |mid |late )?(?:twenties|thirties|forties|fifties)\b", ""),
    (r"\b(?:the )?(?:deep |rich |warm |golden |dark )?colou?r of (?:her|his) skin\b", "skin"),
    (r"\b(?:deep(?:ly)?|golden|bronzed?|sun[- ]?kissed|sun[- ]?bleached|sun[- ]?tanned|tanned|"
     r"tan|olive|pale|fair|porcelain|ivory|dark|ebony|caramel|honey|alabaster)[- ]"
     r"(?:skin(?:ned)?|complexion|tan|glow)\b", "skin"),
    (r"\b(?:deep|golden|bronze|dark|light|healthy|even) tan\b", ""),
    (r"\b(?:tanned|sun-kissed|sunkissed|bronzed|sun-bleached|suntanned)\b", ""),
    (r"\b(?:(?:long|short|shoulder-length|wavy|straight|curly|flowing|messy|silky|sleek|"
     r"windswept|wet|thick)[ ,]+)*(?:blonde?|brunette|platinum|silver|white|black|brown|red|"
     r"auburn|chestnut|golden|dark|light|ginger|copper|honey|ash|strawberry|jet-black|raven)"
     r"(?:[- ]blonde?)?[- ]hair(?:ed)?\b", "hair"),
    (r"\b(?:blue|green|brown|hazel|grey|gray|amber|dark|light|ice-blue|piercing blue) eyes\b",
     "eyes"),
]
_LOOK_RE = [(re.compile(p, re.I), r) for p, r in _LOOK]


def neutral_look(text: str) -> str:
    """Opis bez wieku, koloru skory, opalenizny, koloru wlosow i oczu - przy osobie ze zdjecia
    wyglad ma pochodzic ze zdjecia, nie z tekstu."""
    for rx, repl in _LOOK_RE:
        text = rx.sub(repl, text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def numbers(text: str) -> List[int]:
    """Numery obrazow wskazane w tekscie (1 = pierwszy), w kolejnosci, bez powtorzen."""
    out: List[int] = []
    for m in _MARK.finditer(text):
        n = int(m.group(1) or m.group(2))
        if n not in out:
            out.append(n)
    return out


def strip(text: str) -> str:
    """Tekst bez numerow obrazow (dla Wan i dla opisu dzwieku)."""
    return re.sub(r"\s{2,}", " ", _MARK.sub("", text)).strip()


def plain(prompt: str) -> str:
    """Caly prompt bez oznaczen ujec i numerow obrazow (do dialogow i dzwieku tla)."""
    return strip(_SHOT.sub(" ", prompt))


def split_shots(prompt: str) -> List[str]:
    """Ujecia oznaczone "Ujęcie N:" / "Shot N:"; bez oznaczen - jedno ujecie.
    Gdy pierwsze oznaczenie to "Ujęcie 1:", tekst przed nim (np. tytul, opis calosci)
    dolacza do kazdego ujecia jako wspolny kontekst; gdy prompt zaczyna sie od razu opisem,
    a dalej jest "Ujęcie 2:", ten opis jest ujeciem 1."""
    marks = list(_SHOT.finditer(prompt))
    if not marks:
        return [prompt.strip()]
    head = prompt[:marks[0].start()].strip()
    intro = head if int(marks[0].group(1) or marks[0].group(2)) <= 1 else ""
    shots = [head] if head and not intro else []
    for m, nxt in zip(marks, marks[1:] + [None]):
        body = prompt[m.end():nxt.start() if nxt else len(prompt)].strip()
        if body:
            shots.append(f"{intro}\n{body}".strip() if intro else body)
    return shots or [prompt.strip()]


def split_seconds(total: int, shots: int) -> List[int]:
    """Rowny podzial dlugosci filmu na ujecia (kazde min. 2 s)."""
    shots = max(1, min(shots, max(1, total // 2)))
    base, rest = divmod(total, shots)
    return [base + (1 if i < rest else 0) for i in range(shots)]


def edit_prompt(text: str, used: List[int], has_base: bool, continuing: bool,
                step: bool = False, looks: Optional[Dict[int, str]] = None,
                base_lead: str = "", style: str = styles.REALISTIC,
                creatures: Sequence[int] = (), places_only: bool = False) -> str:
    """Prompt dla Qwen-Image-Edit: [n] -> "from image k" (k = miejsce na liscie obrazow);
    numery spoza `used` (juz dodane albo dodawane w kolejnym przebiegu) znikaja.
    step - kolejny przebieg: obraz 1 to dotychczasowy wynik, ktory ma zostac bez zmian.
    looks - {numer: wyglad osoby} (llm.describe_person): tekst wspiera zdjecie zamiast z nim
    walczyc - bez tego slonce na plazy opalalo blada modelke ze zdjecia.
    base_lead - zdanie o bazie (CHARACTER: baza to postac, ktora ma zostac glowna osoba;
    CREATURE_CHARACTER: an animal or creature).
    style - styl filmu (app/styles.py): jak brzmi regula kompozycji (keep).
    creatures - the numbers (as in `used` and `looks`) of the images that show an animal or creature character:
    their look and what the composition takes from them are said in a creature's words, never a person's."""
    offset = 1 if has_base else 0

    def repl(m: re.Match) -> str:
        n = int(m.group(1) or m.group(2))
        return f" from image {used.index(n) + 1 + offset}" if n in used else ""

    body = re.sub(r"\s{2,}", " ", _MARK.sub(repl, text)).strip()
    if used:  # wyglad osoby ma pochodzic ze zdjecia, nie z tekstu
        # The owner's look words of a custom style at the end (the scene's Wan text) describe the pictures, not a
        # person: they are kept word for word. A preset's text is left as it always was (neutral_look keeps it).
        suffix = f", {styles.of(style).video}"
        if styles.of(style).custom and body.endswith(suffix):
            body = neutral_look(body[:-len(suffix)]).rstrip(" ,.") + suffix
        else:
            body = neutral_look(body)
    lead = _STEP if step else (_CONTINUE if continuing else "")
    marked = numbers(text)
    unmarked = [n for n in used if n not in marked]
    if unmarked:
        # obrazy bez numerow w tekscie: na poczatku (najwazniejsze dla modelu), ze osoby
        # i rzeczy z opisu to te ze zdjec - inaczej opis wyglądu wygrywal ze zdjeciem. Also
        # a cast: its characters are numbered [1]..[k] (app/cast.py mark_names), the user's
        # own images often are not
        listed = " and ".join(f"image {used.index(n) + 1 + offset}" for n in unmarked)
        verb = "Add" if step else "Use"
        # an animal or creature among them: "characters" (people's own words stay as they were)
        beings = "characters" if any(n in creatures for n in unmarked) else "people"
        if marked:
            lead = (f"{lead} {verb} the exact items, places and any other {beings} shown in "
                    f"{listed}.").strip()
        elif base_lead and has_base:
            main = "main character" if base_lead == CREATURE_CHARACTER else "main person"
            lead = (f"{lead} {verb} the exact items, places and any other {beings} shown in "
                    f"{listed}; the {main} is the one from image 1.").strip()
        elif places_only:
            # A shot without characters: the images are places and things. Naming a person here drew one into the
            # frame (a girl in an empty Clearing, test episode 2026-11-22), so only places and things are named.
            lead = (f"{lead} {verb} the exact place and things shown in {listed}: the place and things in the "
                    "description below are them.").strip()
        else:
            lead = (f"{lead} {verb} the exact {'person' if beings == 'people' else beings} and items shown in "
                    f"{listed}: the {beings} and things in the description below are them.").strip()
    looks = {n: look for n, look in (looks or {}).items() if look and n in used}
    who = " ".join(f"The creature from image {used.index(n) + 1 + offset} looks like this: "
                   f"{look.strip().rstrip('.')} - keep exactly this look in any light." if n in creatures else
                   f"The person from image {used.index(n) + 1 + offset} has {look} - keep "
                   "exactly this look in any light." for n, look in looks.items())
    # powtorzone na koncu (model slucha poczatku i konca); jasna karnacja - wprost "nie opalona",
    # bo samo slonce/plaza w opisie opalalo osobe w czesci prob
    tail = " ".join(f"The look of the creature from image {used.index(n) + 1 + offset} stays exactly: "
                    f"{look.strip().rstrip('.')}." if n in creatures else
                    f"Her or his look stays exactly: {look}"
                    + (", not tanned" if re.search(r"pale|porcelain|fair|light|ivory|white",
                                                    look, re.I) else "") + "."
                    for n, look in looks.items())
    ident = base_lead or (_IDENTITY if used and not places_only else "")
    rule = keep(style)
    if places_only:  # no sentence about people: every word about a person can put one into the picture
        rule = " ".join(part for part in re.split(r"(?<=[.!?])\s+", rule) if not re.search(r"(?i)person|people", part))
    return " ".join(x for x in (ident, who, lead, rule, body, tail) if x)


def cast_lead(members: List[dict], first_image: int = 1, adding: bool = False,
              style: str = styles.REALISTIC, creatures: Sequence[int] = ()) -> str:
    """Opening sentence for composing several characters, each from their own image: who is
    in which image and that their faces must stay apart. adding - a later pass that adds
    these people to the picture made so far (image 1). style - how the scene is named.
    creatures - the numbers of the images (as this lead names them: first_image, first_image + 1...) that
    show an animal or creature character: it is named as one, never as a person, and a sentence of its own
    says what to keep of it. Without them the lead reads as it always did."""
    if not members:
        return ""
    numbers = [first_image + i for i in range(len(members))]
    if not any(n in creatures for n in numbers):
        shown = "; ".join(f"image {n} shows {ch['name']}, a {ch.get('gender') or 'person'}"
                          for n, ch in zip(numbers, members))
        where = ("Add them to the picture from image 1 next to the people already there, "
                 if adding else f"Show them together in {styles.get(style).scene} as described, ")
        return (f"These are different people: {shown}. {where}keeping each person's exact face, "
                "facial features, hairstyle, hair color, skin tone and body from their own image. "
                "Do not merge, swap or blend their faces.")
    shown = "; ".join(f"image {n} shows {ch['name']}, "
                      + (_CREATURE if n in creatures else f"a {ch.get('gender') or 'person'}")
                      for n, ch in zip(numbers, members))
    where = ("Add them to the picture from image 1 next to the characters already there"
             if adding else f"Show them together in {styles.get(style).scene} as described")
    people = (", keeping each person's exact face, facial features, hairstyle, hair color, skin tone and body "
              "from their own image" if any(n not in creatures for n in numbers) else "")
    kept = " ".join(_KEEP_CREATURE.format(n=n) for n in numbers if n in creatures)
    return (f"These are different characters: {shown}. {where}{people}. {kept} "
            "Do not merge, swap or blend their faces.")


def renumber(text: str, shift: int) -> str:
    """Image markers [n] / @n moved by `shift` (character images come first in a cast)."""
    if not shift:
        return text
    return _MARK.sub(lambda m: f" [{int(m.group(1) or m.group(2)) + shift}]", text)


def split_passes(ref_list: list, has_base: bool) -> List[list]:
    """How compose_refs spreads the images over passes: the first pass holds up to
    MAX_IMAGES (one fewer with a base), every next one the result so far plus two more."""
    passes, rest = [], list(ref_list)
    first = MAX_IMAGES - (1 if has_base else 0)
    passes.append(rest[:first])
    rest = rest[first:]
    while rest:
        passes.append(rest[:MAX_IMAGES - 1])
        rest = rest[MAX_IMAGES - 1:]
    return passes


def compose_refs(client: Engine, base: Optional[str], ref_list: List[Tuple[int, str]],
                 text: str, width: int, height: int, out_path: str | Path,
                 seed: Optional[int] = None, continuing: bool = False,
                 progress_cb: Optional[Callable[[float, str], None]] = None,
                 looks: Optional[Dict[int, str]] = None,
                 base_lead: str = "",
                 pass_leads: Optional[List[str]] = None,
                 style: str = styles.REALISTIC,
                 creatures: Sequence[int] = (), places_only: bool = False) -> Tuple[bool, str]:
    """Klatka z dowolnej liczby obrazow pomocniczych (do MAX_REFS). Qwen-Image-Edit bierze 3
    obrazy naraz, wiec przy wiekszej liczbie sklada w przebiegach: pierwszy z baza (zdjecie
    startowe / ostatnia klatka) i pierwszymi obrazami, kazdy kolejny dokłada 2 nastepne do
    dotychczasowego wyniku, ktory ma zostac bez zmian. text - opis z numerami [n];
    ref_list - [(numer, sciezka)]. pass_leads - opening sentence of each pass (several
    characters: who is in which image), instead of base_lead. style - styl filmu (app/styles.py).
    creatures - the numbers in ref_list of the images that show an animal or creature character (edit_prompt).
    Zwraca (ok, sciezka/blad)."""
    if not ref_list:
        return False, "No reference images were provided."
    passes = split_passes(ref_list, bool(base))
    current = base
    for i, chunk in enumerate(passes):
        used = [n for n, _ in chunk]
        lead = (pass_leads[i] if pass_leads and i < len(pass_leads) else "") or \
            (base_lead if i == 0 else "")
        prompt = edit_prompt(text, used, bool(current), continuing and i == 0, step=i > 0,
                             looks=looks, base_lead=lead, style=style, creatures=creatures,
                             places_only=places_only)
        target = Path(out_path) if i == len(passes) - 1 else \
            Path(out_path).with_name(f"{Path(out_path).stem}_p{i + 1}.png")

        def cb(frac: float, desc: str, i=i) -> None:
            if progress_cb:
                label = f"pass {i + 1}/{len(passes)}, " if len(passes) > 1 else ""
                progress_cb((i + frac) / len(passes), label + desc)

        images = ([current] if current else []) + [p for _, p in chunk]
        ok, msg = compose(client, images, prompt, width, height, target,
                          None if seed is None else seed + i, cb,
                          **({} if style == styles.REALISTIC else {"style": style}))
        if i > 0 and current and current != base:
            Path(current).unlink(missing_ok=True)  # posredni wynik poprzedniego przebiegu
        if not ok:
            return False, msg
        current = str(target)
    return True, current


def compose(client: Engine, images: List[str], prompt: str, width: int, height: int,
            out_path: str | Path, seed: Optional[int] = None,
            progress_cb: Optional[Callable[[float, str], None]] = None,
            style: str = styles.REALISTIC) -> Tuple[bool, str]:
    """Compose one image from 1–3 references: (ok, path or error message). An animated `style` (app/styles.py)
    composes with its edit LoRA, and the sentence the LoRA was trained with opens the prompt."""
    from . import engine_runtime, llm

    if llm.blocked(prompt):
        return False, llm.BLOCK_MSG
    if not images:
        return False, "No reference images were provided."
    if len(images) > MAX_IMAGES:
        return False, f"A composition pass supports at most {MAX_IMAGES} images."
    seed = random.randint(0, 2 ** 31 - 1) if seed is None or seed < 0 else int(seed)
    look = styles.get(style)
    options = {}
    if look.edit_lora:
        prompt = f"{look.edit_trigger} {prompt}".strip()
        options["style"] = look.id
    try:
        picture = engine_runtime.edit(images, prompt, width, height, seed, progress_cb, runtime=client, **options)
    except Exception as e:  # noqa: BLE001
        return False, f"Reference composition failed: {e}"
    if not picture:
        return False, "The engine returned no image."
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(picture)
    return True, str(out_path)
