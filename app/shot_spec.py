# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The shot contract (docs/QUALITY_ARCHITECTURE.md, layer 1).

A series episode's agent no longer writes the video model's prompt. It writes a ShotSpec per shot: the place, the
time of day, who is in it, one clear action, the camera, what the narrator's words need to see (key_visual), the
props, and what the shot check must find or must not find. This module

- validates a spec against the series bible (app/bible.py) and the series' rules (only the named cast, narrator only),
- derives the shot's cast from every name and alias it mentions (Barley = "the owl"), so a character named in the
  text is always in the cast and gets its photo and description (defect D),
- refuses negations ("no people") in any text that reaches the prompt: the sampler has no negative prompt, so a
  negation only puts the thing in the model's mind (defect A); must_not_show goes to the check only,
- compiles the Wan prompt deterministically: the place's canonical words, the light, each cast member's canonical
  words with its accessories, the action, the key visuals and the camera. Never "talks" or "lip movement" in a
  narrated shot (defect I).

The compiled prompt is what the scene's `prompt` holds, so the Timeline, its keys and its clips work unchanged; the
spec itself is kept with the scene (timeline.Scene.shot) for the keyframe, the check and the editor.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Iterable, List, Optional, Sequence, Tuple

from . import bible as bible_mod

TIMES = {"morning": "soft morning light", "day": "bright daylight", "afternoon": "warm afternoon light",
         "golden hour": "warm golden evening light", "dusk": "soft dusk light with glowing fireflies",
         "night": "calm night with soft moonlight", "indoor warm": "warm cozy lamp light",
         "indoor night": "warm lamp light at night"}
CAMERAS = {
    "static medium": "Static medium shot at the characters' eye level.",
    "static close-up": "Static close-up at the character's eye level.",
    "static wide": "Static wide shot, the characters clearly visible in the middle of the frame.",
    "slow push-in": "The camera pushes in slowly and gently, keeping the characters centered.",
    "slow pull-back": "The camera pulls back slowly and gently, keeping the characters centered.",
    "slow pan left": "The camera pans slowly to the left, following the characters.",
    "slow pan right": "The camera pans slowly to the right, following the characters.",
    "slow tracking": "The camera tracks slowly alongside the characters at their eye level.",
    "low angle static": "Static low-angle medium shot, the characters fully in frame.",
}
# Verbs of a visible action: a shot must have one (defect I: characters standing still facing the camera).
MOTION = ("walk", "walks", "walking", "run", "runs", "hop", "hops", "jump", "jumps", "skip", "skips", "bend", "bends",
          "pick", "picks", "lift", "lifts", "place", "places", "put", "puts", "set", "sets", "carry", "carries",
          "push", "pushes", "pull", "pulls", "roll", "rolls", "turn", "turns", "spin", "spins", "wave", "waves",
          "point", "points", "reach", "reaches", "climb", "climbs", "sit", "sits", "lie", "lies", "kneel", "kneels",
          "fly", "flies", "flutter", "flutters", "flap", "flaps", "land", "lands", "float", "floats", "drift",
          "drifts", "fall", "falls", "blow", "blows", "hug", "hugs", "nod", "nods", "yawn", "yawns", "stretch",
          "stretches", "tiptoe", "tiptoes", "peek", "peeks", "open", "opens", "close", "closes", "dig", "digs",
          "plant", "plants", "water", "waters", "pour", "pours", "throw", "throws", "catch", "catches", "splash",
          "splashes", "shake", "shakes", "wiggle", "wiggles", "dance", "dances", "march", "marches", "crawl",
          "crawls", "lean", "leans", "look", "looks", "glance", "glances", "smile", "smiles", "laugh", "laughs",
          "giggle", "giggles", "sniff", "sniffs", "tap", "taps", "pat", "pats", "stroke", "strokes", "tuck",
          "tucks", "cover", "covers", "snuggle", "snuggles", "follow", "follows", "chase", "chases", "swing",
          "swings", "slide", "slides", "rise", "rises", "sink", "sinks", "sway", "sways", "rustle", "rustles",
          "twinkle", "twinkles", "glow", "glows", "fly", "build", "builds", "stack", "stacks", "hold", "holds",
          "hand", "hands", "give", "gives", "share", "shares", "eat", "eats", "drink", "drinks", "sleep", "sleeps",
          "wake", "wakes", "watch", "watches", "step", "steps", "stand", "stands", "come", "comes", "go", "goes",
          "enter", "enters", "leave", "leaves", "move", "moves", "grow", "grows", "melt", "melts", "sail", "sails")
STATIC = ("stand", "stands", "sit", "sits", "look", "looks", "watch", "watches", "smile", "smiles", "sleep", "sleeps",
          "glow", "glows", "twinkle", "twinkles", "hold", "holds")
_NEGATIONS = re.compile(r"\b(?:no|not|never|without|none|nobody|nothing|neither|nor|avoid|free of)\b"
                        r"|\b(?:don't|doesn't|isn't|aren't|won't|can't|do not|does not)\b", re.I)
_LIP = re.compile(r"\b(?:talks?|talking|speaks?|speaking|says?|saying|lip|lips|mouth movement|lip[- ]sync)\b", re.I)
_WORD = re.compile(r"[a-z]+(?:[-'][a-z]+)*")
MAX_KEY_VISUALS = 3
MAX_ASSERTIONS = 5
MAX_CHAIN = 2  # shots in a row that continue one another's last frame (`continuous`)


class ShotError(ValueError):
    """What is wrong with a shot, in words the agent can act on."""


@dataclass(frozen=True)
class ShotSpec:
    location: str                      # a series place (slug or name) of kind location
    time_of_day: str                   # one of TIMES
    action: str                        # one visible action with its direction, e.g. "Lumi walks from left to right"
    camera: str                        # one of CAMERAS
    cast: Tuple[str, ...] = ()         # slugs; names and aliases in the texts are added to it
    key_visual: Tuple[str, ...] = ()   # 1-3 things from the narration that MUST be seen
    props: Tuple[str, ...] = ()        # series props (slugs or names)
    must_show: Tuple[str, ...] = ()    # assertions for the check, not prompt text
    must_not_show: Tuple[str, ...] = ()
    continuous: bool = False           # continues the previous shot's last frame (same place and cast)

    def to_json(self) -> dict:
        data = asdict(self)
        return {key: list(value) if isinstance(value, tuple) else value for key, value in data.items()
                if value not in ((), "", False) or key in ("location", "time_of_day", "action", "camera")}

    def dumps(self) -> str:
        """The spec as the canonical JSON the scene keeps (sorted keys, so equal specs are equal strings)."""
        return json.dumps(self.to_json(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, data) -> "ShotSpec":
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise ShotError("A shot is an object with location, time_of_day, action and camera.")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ShotError(f"Unknown shot fields: {', '.join(sorted(unknown))}.")

        def strings(name: str) -> Tuple[str, ...]:
            value = data.get(name) or []
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list) or any(not isinstance(one, str) for one in value):
                raise ShotError(f"shot.{name} is a list of short texts.")
            return tuple(" ".join(one.split()) for one in value if one.strip())

        def one(name: str) -> str:
            value = data.get(name, "")
            if not isinstance(value, str):
                raise ShotError(f"shot.{name} is a text.")
            return " ".join(value.split())

        return cls(location=one("location"), time_of_day=one("time_of_day").lower(), action=one("action"),
                   camera=one("camera").lower(), cast=strings("cast"), key_visual=strings("key_visual"),
                   props=strings("props"), must_show=strings("must_show"), must_not_show=strings("must_not_show"),
                   continuous=bool(data.get("continuous", False)))


def negations(text: str) -> List[str]:
    """The negations in a text that would reach the prompt ("no people", "without a hat")."""
    found = []
    for match in _NEGATIONS.finditer(text or ""):
        tail = " ".join((text[match.end():].split())[:2])
        found.append(f"{match.group(0)} {tail}".strip())
    return found


def _mentions(text: str, alias: str) -> bool:
    return bool(re.search(rf"(?<![\w-]){re.escape(alias)}(?:'s|s)?(?![\w-])", text or "", re.I))


def mentioned(texts: Iterable[str], cast: Sequence[bible_mod.CharacterSpec]) -> List[str]:
    """The slugs of the characters a text names by name or alias, in the order of the cast."""
    joined = " ".join(texts)
    return [spec.slug for spec in cast if any(_mentions(joined, alias) for alias in spec.aliases)]


def strangers(texts: Iterable[str], allowed: bible_mod.Bible) -> List[str]:
    """Words for people or animals in a text that are not the bible's characters (a boy, a fox in a series of an owl
    and a creature): under the rule "only the named characters" they must not be asked for."""
    words = _WORD.findall(" ".join(texts).lower())
    known = {alias.casefold() for spec in allowed.characters for alias in spec.aliases}
    known |= {spec.species for spec in allowed.characters if spec.species}
    found = []
    for word in words:
        base = word[:-1] if word.endswith("s") and word[:-1] in bible_mod.SPECIES else word
        if (word in bible_mod.PEOPLE or base in bible_mod.SPECIES) and base not in known and word not in known:
            if base in ("creature", "monster") and any(spec.species in ("creature", "monster")
                                                       for spec in allowed.characters):
                continue
            found.append(word)
    return list(dict.fromkeys(found))


def shot_cast(spec: ShotSpec, allowed: bible_mod.Bible) -> Tuple[str, ...]:
    """The shot's cast: the slugs it gives and every character its action, key visuals or must_show name."""
    named = mentioned([spec.action, *spec.key_visual, *spec.must_show], allowed.characters)
    given = [slug.strip().lower() for slug in spec.cast]
    return tuple(dict.fromkeys([*given, *named]))


def validate(spec: ShotSpec, allowed: bible_mod.Bible, *, previous: Optional[ShotSpec] = None,
             chain: int = 0, seconds: Optional[float] = None, max_seconds: Optional[float] = None,
             spoken_by: Sequence[str] = ()) -> List[str]:
    """Every problem with a shot, as sentences ([] = fine). previous/chain: the shot before and how many shots in a
    row already continue one another (for `continuous`); seconds/max_seconds: one picture pass only; spoken_by: the
    slugs of characters with lines in the shot (a narrated series has none)."""
    problems: List[str] = []
    place = allowed.place(spec.location)
    if not spec.location:
        problems.append("shot.location is required: a location of the series bible.")
    elif place is None or place.kind != "location":
        names = ", ".join(one.slug for one in allowed.locations) or "none yet: add the place to the series first"
        problems.append(f"shot.location ‘{spec.location}’ is not a location of the series ({names}).")
    if spec.time_of_day not in TIMES:
        problems.append(f"shot.time_of_day must be one of: {', '.join(TIMES)}.")
    if spec.camera not in CAMERAS:
        problems.append(f"shot.camera must be one of: {', '.join(CAMERAS)}.")
    if len(spec.action.split()) < 3:
        problems.append("shot.action must be one clear visible action with its direction, e.g. ‘Lumi walks from the "
                        "left to the right along the path’.")
    verbs = [word for word in _WORD.findall(spec.action.lower()) if word in MOTION]
    if spec.action and not verbs:
        problems.append("shot.action has no action verb (walks, bends down, picks up, flies, turns...).")
    elif verbs and all(verb in STATIC for verb in verbs) and spec.camera.startswith("static"):
        problems.append("shot.action only stands, sits or looks and the camera is static: the shot would be a still "
                        "picture. Give the character a visible movement (walks, turns, reaches, waves...) or move "
                        "the camera slowly.")
    if not 1 <= len(spec.key_visual) <= MAX_KEY_VISUALS:
        problems.append(f"shot.key_visual needs 1-{MAX_KEY_VISUALS} things the narration speaks about that must be "
                        "seen.")
    if any(len(one) > 60 for one in spec.key_visual):
        problems.append("shot.key_visual: each item is a short noun phrase (at most 60 characters).")
    for name in ("must_show", "must_not_show"):
        rows = getattr(spec, name)
        if len(rows) > MAX_ASSERTIONS or any(len(one) > 100 for one in rows):
            problems.append(f"shot.{name}: at most {MAX_ASSERTIONS} short statements.")
    for text_name in ("action", "key_visual", "must_show"):
        value = getattr(spec, text_name)
        for found in negations(" ".join(value) if isinstance(value, tuple) else value):
            problems.append(f"shot.{text_name} says ‘{found}’: never write what must not be there in the picture "
                            "(the video model has no negative prompt, so it draws it). Put it in must_not_show.")
    if _LIP.search(spec.action) and allowed.narrator_only:
        problems.append("shot.action makes a character talk, but the series is narrated: characters never speak.")
    cast = shot_cast(spec, allowed)
    unknown = [slug for slug in cast if allowed.character(slug) is None]
    if unknown:
        problems.append(f"shot.cast: {', '.join(unknown)} is not a character of the series "
                        f"({', '.join(one.slug for one in allowed.characters)}).")
    if len(cast) > allowed.max_cast:
        problems.append(f"At most {allowed.max_cast} characters in a shot.")
    if allowed.only_named_cast:
        extra = strangers([spec.action, *spec.key_visual, *spec.must_show], allowed)
        if extra:
            problems.append(f"The series shows only its named characters, but the shot asks for "
                            f"{', '.join(extra)}. Leave them out.")
    for prop in spec.props:
        found = allowed.place(prop)
        if found is None or found.kind != "prop":
            problems.append(f"shot.props: ‘{prop}’ is not a prop of the series "
                            f"({', '.join(one.slug for one in allowed.props) or 'none yet'}).")
    if allowed.narrator_only and spoken_by:
        problems.append("The series is narrated: lines are the narrator's (speaker @<narrator voice>), never a "
                        "character's.")
    if spec.continuous:
        if previous is None:
            problems.append("shot.continuous: the first shot cannot continue a previous one.")
        else:
            if allowed.place(previous.location) != place:
                problems.append("shot.continuous: a shot that continues the previous one stays in its location.")
            if set(shot_cast(previous, allowed)) != set(cast):
                problems.append("shot.continuous: a shot that continues the previous one keeps its cast.")
            if chain + 1 > MAX_CHAIN:
                problems.append(f"shot.continuous: at most {MAX_CHAIN} shots in a row may continue one another. Make "
                                "this one a cut.")
    if seconds is not None and max_seconds is not None and seconds > max_seconds + 1e-6:
        problems.append(f"The shot lasts {seconds:g} s; one picture pass makes at most {max_seconds:g} s. Split the "
                        "narration over two shots (a line may run on over the cut).")
    return problems


def compile_prompt(spec: ShotSpec, allowed: bible_mod.Bible) -> str:
    """The Wan prompt of a shot (without the style words, which the video stage adds). Deterministic: the same spec
    and bible always give the same prompt. Raises ShotError for a spec that does not validate."""
    # the shot's own fields only: whether it may continue the shot before it is decided where the shots are in order
    problems = validate(replace(spec, continuous=False), allowed)
    if problems:
        raise ShotError(" ".join(problems))
    place = allowed.place(spec.location)
    cast = [allowed.character(slug) for slug in shot_cast(spec, allowed)]
    # The place by its library name first: the Timeline puts the named place's plate into the keyframe
    # (timeline.named_items), so every shot of the place starts from the same approved picture of it (defects B, H).
    parts = [f"{place.name}: {bible_mod.positive(place.canonical).rstrip('.')}, {TIMES[spec.time_of_day]}."]
    for member in cast:
        parts.append(member.sentence().rstrip(".") + ".")
    action = spec.action.rstrip(".") + "."
    parts.append(action[0].upper() + action[1:])
    visuals = [one for one in spec.key_visual if one.casefold() not in action.casefold()]
    for prop in spec.props:
        found = allowed.place(prop)
        if found is not None and found.canonical:
            visuals.append(f"{found.name}: {bible_mod.positive(found.canonical).rstrip('.')}")
    if visuals:
        parts.append("Clearly visible: " + "; ".join(visuals) + ".")
    if len(cast) > 1:
        parts.append(f"{len(cast)} characters only: " + " and ".join(member.name for member in cast) + ".")
    elif len(cast) == 1:
        parts.append(f"{cast[0].name} is the only character in the scene.")
    camera = CAMERAS[spec.camera]
    if not cast:  # a shot of a place or a thing never mentions characters: the model draws what it reads (defect A)
        camera = (camera.replace("the characters' eye level", "eye level").replace("the character's eye level",
                                                                                   "eye level")
                  .replace("keeping the characters centered", "keeping the subject centered")
                  .replace("following the characters", "following the subject")
                  .replace("alongside the characters", "alongside the subject")
                  .replace("the characters clearly visible in the middle of the frame",
                           "the subject clearly visible in the middle of the frame")
                  .replace("the characters fully in frame", "the subject fully in frame"))
    parts.append(camera)
    prompt = " ".join(parts)
    if negations(prompt):  # a canonical text of the bible kept a negation: never send it
        raise ShotError(f"The compiled prompt still says ‘{negations(prompt)[0]}’. Remove it from the series bible.")
    return prompt


def lint_prompt(text: str, *, narrated: bool) -> List[str]:
    """Problems of a free prompt (a scene without a spec): negations, and talking in a narrated film."""
    problems = [f"The description says ‘{found}’: the video model has no negative prompt, so it draws what you "
                "exclude. Describe only what is there." for found in negations(text)]
    if narrated and _LIP.search(text or ""):
        problems.append("The description makes a character talk, but the film is narrated.")
    return problems
