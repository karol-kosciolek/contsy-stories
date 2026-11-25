# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Local prompt writing and picture reviews using the cataloged Qwen model.

The assistant writes generation prompts. Its vision side describes a reference person, checks
a character's reference photos (one adult person, or one animal or creature without people) and answers
structured reviews such as the Timeline's join review. The model loads only when one of these is asked for.
Generation prompts use English, spoken lines too (the films are in English); a line the user quoted is kept word for word. The installed
llama-cpp-python binding runs the model in a Python worker.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from engine.models.llm.backend import runtime_problem
from engine.models.llm.client import LlmClient
from engine.models.llm.errors import LlmError

from . import config, model_manager, refs, styles

MODEL_ID = "prompt-llm"
ProgressCb = Callable[[float, str], None]

BLOCK_MSG = ("Blocked: sexualized depictions of minors are not allowed. "
             "Characters in intimate clothing must be clearly adult.")

# --- granica stala: nieletni + tresci seksualne (PL i EN) ---
_MINOR = re.compile(
    r"\b(child|children|kids?|minors?|underage|under-age|teens?|teenagers?|teenage|preteens?|"
    r"pre-teens?|loli|lolita|shota|schoolgirls?|schoolboys?|little (girl|boy)s?|young (girl|boy)s?|"
    r"toddlers?|infants?|newborns?|juveniles?|pubescent|prepubescent|"
    r"([1-9]|1[0-7])[ -]?(yo|y/o|years? old|year-old)|"
    r"dzieck\w*|dzieci\w*|nastolat\w*|nieletni\w*|niepełnolet\w*|małolat\w*|uczennic\w*|"
    r"dziewczyn(ka|ki|ce|kę|eczk\w*)|chłop(czyk\w*|iec|ca|cem)|niemowl\w*|gimnazjalist\w*|"
    r"podstawówk\w*|([1-9]|1[0-7])[ -]?(lat|latk\w*|letni\w*))\b", re.I)
_SEXUAL = re.compile(
    r"\b(nsfw|nude|naked|nudity|sex\w*|erotic\w*|porn\w*|explicit|topless|bottomless|lingerie|"
    r"underwear|bikini|breasts?|boobs?|nipples?|genital\w*|penis|vagina|pussy|aroused|orgasm\w*|"
    r"bdsm|fetish\w*|seductive|sensual|undress\w*|strip(ping|tease|per|pers)|"
    r"nago|nag(a|i|ie|ich|y)|nagość\w*|seks\w*|eroty\w*|porno\w*|bielizn\w*|bieliź\w*|piersi|"
    r"biust\w*|sutk\w*|podnieca\w*|rozebran\w*|rozbier\w*|zmysłow\w*|uwodzic\w*)\b", re.I)


def normalize_for_check(text: str) -> str:
    """Comparison copy of a text for the guard below: compatibility forms are folded (NFKC) and
    invisible format characters removed, so full-width letters or a zero-width joiner cannot hide
    a word. The prompt that is sent or saved is never changed."""
    normalized = unicodedata.normalize("NFKC", text or "")
    return "".join(char for char in normalized if unicodedata.category(char) != "Cf")


def blocked(*texts: str) -> bool:
    """True, gdy tekst laczy nieletnich z tresciami seksualnymi."""
    text = normalize_for_check(" ".join(t for t in texts if t))
    return bool(_MINOR.search(text) and _SEXUAL.search(text))


def refers_to_minor(*texts: str) -> bool:
    """True when a text names a minor at all (a child, a teen, an age under 18), with the same normalisation as
    `blocked`. For texts that describe the person itself (a character's look, its voice), where only adults are
    allowed, not for scene prompts (a family scene may name a child)."""
    return bool(_MINOR.search(normalize_for_check(" ".join(t for t in texts if t))))


# A style's look words describe how every picture of a film is drawn, its characters included, and every character is
# an adult: childlike drawing styles are refused along with words that name a minor.
_CHILDLIKE = re.compile(r"\b(chibi\w*|child[\s-]*like|baby[\s-]*fac(?:e|es|ed)|baby[\s-]*ish|kawaii[\s-]*kids?|"
                        r"super[\s-]*deformed)\b", re.I)
STYLE_MINOR_MSG = ("⛔ A style describes how the pictures look, and every character is an adult. Leave out words about "
                   "children or childlike looks: for example “picture-book illustration” instead of “children's book "
                   "illustration”.")


def style_problem(style_id: str) -> str:
    """Why the owner's custom style `style_id` (app/styles.py) cannot be used ("" = fine; "" for a preset or an
    unknown id): its look words are refused like a prompt (`blocked`), and so are words about minors or childlike
    looks, since every character drawn in it is an adult."""
    look = styles.of(style_id)
    if not look.custom:
        return ""
    words = look.about.replace("_", " ")  # the look may join words with "_", which \b does not split
    if blocked(words):
        return BLOCK_MSG
    if refers_to_minor(words) or _CHILDLIKE.search(normalize_for_check(words)):
        return STYLE_MINOR_MSG
    return ""


# An animal or creature character is fully grown, like every character: its look may not describe a young animal or
# a childlike look. "baby blue" and "baby pink" are colours, "fawn-colored" too (a fawn alone is a young deer), and
# "kitten heels" are shoes.
_YOUNG = re.compile(
    r"\b(pupp(?:y|ies)|pups?|kittens?(?![\s-]*heels?\b)|cubs?|foals?|fawns?(?![\s-]*colou?r)|chicks?|hatchlings?|"
    r"ducklings?|goslings?|piglets?|fledglings?|nestlings?|babies|baby(?![\s-]*(?:blue|pink)\b)|young\w*|"
    r"juveniles?)\b", re.I)


# Sexual or nude content (_SEXUAL without the clothing words: a bikini or underwear alone is no reason). An animal or
# creature character is never in such a picture (app/characters.py, app/timeline.py).
_SEX_OR_NUDE = re.compile(
    r"\b(nsfw|nude|naked|nudity|sex\w*|erotic\w*|porn\w*|explicit|topless|bottomless|breasts?|boobs?|nipples?|"
    r"genital\w*|penis|vagina|pussy|aroused|orgasm\w*|bdsm|fetish\w*|seductive|sensual|undress\w*|"
    r"strip(ping|tease|per|pers)|nago|nag(a|i|ie|ich|y)|nagość\w*|seks\w*|eroty\w*|porno\w*|piersi|biust\w*|"
    r"sutk\w*|podnieca\w*|rozebran\w*|rozbier\w*|zmysłow\w*|uwodzic\w*|"
    r"without\s+(?:any\s+)?cloth(?:es|ing)|unclothed|disrob\w*|bare[\s-]*breasted|in\s+the\s+buff|"
    r"nadzy|nag(?:ą|im|imi|iej|o)|bez\s+ubra\w*)\b", re.I)


def sexual(*texts: str) -> bool:
    """True when a text describes sexual or nude content, with the same normalisation as `blocked`."""
    return bool(_SEX_OR_NUDE.search(normalize_for_check(" ".join(t for t in texts if t)).replace("_", " ")))


def refers_to_young(*texts: str) -> bool:
    """True when a text describes a young animal or a childlike look (a puppy, a kitten, a cub, a baby, "young",
    chibi...), with the same normalisation as `blocked`. For the look of an animal or creature character
    (app/characters.py), which must be fully grown; words that name a minor are `refers_to_minor`."""
    text = normalize_for_check(" ".join(t for t in texts if t)).replace("_", " ")  # \b does not split at "_"
    return bool(_YOUNG.search(text) or _CHILDLIKE.search(text))


# An animal in the idea is no person (on 2026-11-16 "rudy pies corgi", a ginger corgi dog, came back as "Rudy, an adult
# man aged 35"): the age rule is for people, and an animal or creature character keeps its name and its kind.
_ANIMALS = ("Animals and creatures stay animals and creatures: describe them as such, never as people, and give them no "
            "human age; a name the user gives them stays their name. ")

# How Contsy reads a prompt (the owner's request of 2026-11-16: the assistants wrote generic prompts). A saved
# character comes in by its @name (app/cast.py), quoted text after a speaking verb is spoken, quoted text after "the
# text" / "a sign reads" is drawn (dialogue.shown), and the film's style words are added by Contsy itself.
_CONTSY_NAMES = ("Write a saved character as @ followed by its name exactly as given (for example @chain_dog), with "
                 "no age: that is "
                 "how Contsy puts that character, with its own reference photos, into the picture. Its photos and its "
                 "saved description define its face, body, colors, outfit and what it always wears, and Contsy adds them "
                 "to the prompt itself, so describe what it does - action, pose, expression, where it looks - instead of "
                 "how it looks. ")
_CONTSY_SHOWN = ("Text that is seen in the picture - on a screen, a sign, a banner, a shirt - goes in quotes right "
                 "after 'the text', 'a sign reads' or 'displays the text' (for example 'a large screen displays the "
                 "text \"CRYPTO NEWS\"'): Contsy draws it and nobody says it. Keep such text to 1-3 words. ")
_CONTSY_STYLE = "Contsy adds the film's style words itself: do not write style words such as 3D, Pixar or anime. "

_SYSTEM = {
    "image": (
        "You write prompts for a photorealistic image generator. The user's idea may be in any language, "
        "often colloquial: understand every word literally and keep its exact meaning - never "
        "replace or reinterpret it. The prompt MUST be written in English. "
        "Reply with ONE prompt: a single paragraph of 60-120 words, no preamble, no title, no "
        "quotes around it, no markdown, no notes. Describe the subject and their appearance, clothing, pose and "
        "expression, the setting, lighting, camera (lens, angle, framing), mood and fine details. "
        "Follow the user's idea exactly and describe every person as an adult with a concrete "
        "age of 21 or more (e.g. 'a 27-year-old woman'). " + _ANIMALS + "Markers like [1], [2] point to the "
        "user's reference photos: copy "
        "each one right after the thing it names and never replace that thing with something "
        "else; do not invent its color, print or details - the photo defines them. "
        "Example: idea 'man [1] in hoodie [2]' -> 'a man [1] wearing the hoodie [2], "
        "...'. " + _CONTSY_NAMES + _CONTSY_SHOWN + "A photo has no speech: never use quotes for anything but text "
        "seen in the picture. " + _CONTSY_STYLE),
    "video": (
        "You write prompts for a realistic text-to-video model that makes 5-second shots. The "
        "user's idea may be in any language, often colloquial: understand its meaning and keep "
        "its exact meaning - never replace or reinterpret it. The prompt MUST "
        "be written in English. Spoken lines are English too: put what the user wants said into a short natural English line; only a line the user wrote in quotes is kept word for word. Reply with the prompt only, no preamble, no markdown, no notes. Per shot: "
        "one main "
        "action, the subject large in frame, their appearance and clothing, setting, lighting, "
        "camera movement and framing, and end with what is heard (e.g. 'sound of rain, distant "
        "traffic'). If someone speaks, write the line inside „…” right after "
        "'says' - use „…” ONLY for human speech, never for animal or other sounds (write 'a dog "
        "barking', not „Woof!”). A word starting with @ before 'says' (e.g. '@Hanna says „…”') "
        "names that speaker's voice: keep it exactly as written, right before that person's "
        "'says', and never invent @ tags other than the user's and the characters' named below. Use a single shot (one paragraph, "
        "50-110 words) unless the idea "
        "clearly describes a sequence (e.g. 'then', 'next'); only then write each "
        "scene on its own line starting with 'Shot 1:', 'Shot 2:' and so on. Follow the "
        "user's idea exactly and describe every person as an adult with a concrete age of 21 "
        "or more. " + _ANIMALS + "Write spoken lines only when the idea has them: never invent speech. Markers like [1], "
        "[2] point to the user's reference photos: copy each one right after the thing it names "
        "and never replace that thing with something else; do not invent its color, print or "
        "details - the photo defines them. Example: idea 'man in hoodie [1] "
        "jumps onto building [2]' -> 'Shot 1: an adult man wearing the hoodie [1] ...\\nShot 2: "
        "he jumps onto the building [2] ...'. " + _CONTSY_NAMES + "Spoken lines go in „…” only right after says, asks, "
        "shouts, whispers or replies, with the speaker named before the verb. " + _CONTSY_SHOWN + "A character speaks "
        "about 2.5 words per second: keep the spoken words short enough for the shot. Start with the place and its "
        "light, then one main action, then the camera (static, slow push-in or slow pan). " + _CONTSY_STYLE),
}

def system_prompt(kind: str, style: str = styles.REALISTIC) -> str:
    """The prompt writer's instructions for `kind` ('image' / 'video') in a film's style (app/styles.py): the
    model it writes for is named in that style, so no realistic wording reaches a prompt of an animated film.
    Realistic is _SYSTEM as it always read."""
    kind = "video" if kind == "video" else "image"
    look = styles.get(style)
    if look.realistic:
        return _SYSTEM[kind]
    text = _SYSTEM[kind].replace(styles.get(styles.REALISTIC).writer_image, look.writer_image, 1) \
        .replace(styles.get(styles.REALISTIC).writer_video, look.writer_video, 1)
    # the style is named so the prompt fits it (no daylight photo realism in a cartoon); Contsy adds its words itself
    return text + (f" Every picture is in this style, which Contsy adds to the prompt: {look.image}." if kind == "image"
                   else f" Every shot is in this style, which Contsy adds to the prompt: {look.video}.")


_REMIND = ("\nIMPORTANT: answer in English only, spoken lines too (a line the user quoted is the only "
           "exception) and keep every [n] image marker and every @voice tag.")


_ORDINALS = {"pierwsz": 1, "drug": 2, "trzec": 3, "czwart": 4, "piąt": 5, "szóst": 6,
             "siódm": 7, "ósm": 8, "dziewiąt": 9, "dziesiąt": 10}
_PHOTO = r"(?:zdjęci\w*|załącznik\w*|obraz\w*|obrazk\w*|fotk\w*|fotografi\w*|foto)"


def mark_refs(text: str, n: int) -> str:
    """Polskie odwolania do zdjec -> numery, zanim tekst trafi do modelu (9B mylil "z
    pierwszego zdjecia" / "z drugiego zdjecia"): "z drugiego zdjęcia" -> [2], "ze zdjęcia 3"
    -> [3], przy jednym zdjeciu "ze zdjęcia", "na zdjęciu", "załączona" -> [1]."""
    if not n:
        return text

    ordinals = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
    text = re.sub(r"\b(?:from |in |on )?(?:the )?(first|second|third|fourth|fifth|sixth|"
                  r"seventh|eighth|ninth|tenth) (?:photo|image|picture|attachment)\b",
                  lambda m: f"[{ordinals[m.group(1).lower()]}]"
                  if ordinals[m.group(1).lower()] <= n else m.group(0), text, flags=re.I)
    text = re.sub(r"\b(?:photo|image|picture|attachment)\s*(?:#|no\.?\s*)?(\d{1,2})\b",
                  lambda m: f"[{m.group(1)}]" if 1 <= int(m.group(1)) <= n else m.group(0),
                  text, flags=re.I)

    def ordinal(m: re.Match) -> str:
        k = next((v for s, v in _ORDINALS.items() if m.group(1).lower().startswith(s)), 0)
        return f" [{k}]" if 1 <= k <= n else m.group(0)

    text = re.sub(rf"\s*\b(?:z|ze|na|w)?\s*(\w+)\s+{_PHOTO}\b", lambda m: ordinal(m)
                  if any(m.group(1).lower().startswith(s) for s in _ORDINALS) else m.group(0),
                  text, flags=re.I)
    text = re.sub(rf"\s*\b(?:z|ze|na|w)?\s*{_PHOTO}\s*(?:nr\.?|numer)?\s*(\d{{1,2}})\b",
                  lambda m: f" [{m.group(1)}]" if 1 <= int(m.group(1)) <= n else m.group(0),
                  text, flags=re.I)
    if n == 1:
        text = re.sub(rf"\s*\b(?:z|ze|na)\s+{_PHOTO}\b", " [1]", text, flags=re.I)
        text = re.sub(r"\b(?:załączon|dołączon)\w*\s*", "", text, flags=re.I)
        if "[1]" not in text:
            text = re.sub(r"\s*\b(?:z|ze)\s+załącznik\w*\b", " [1]", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip()


def refs_rule(n: int) -> str:
    """Wskazowka dla modelu, gdy uzytkownik dolaczyl n zdjec-referencji."""
    one = (" With one photo, the person (or item) the idea talks about is the one in [1], "
           "unless the idea clearly describes someone else." if n == 1 else "")
    return (f" The user attached {n} reference photo{'s' if n > 1 else ''} ([1]"
            f"{f' to [{n}]' if n > 1 else ''}). When the idea mentions the photo or what is in "
            "it ('from the photo', 'in the attachment', 'this woman', 'that outfit'), refer to it with its marker right after it, e.g. 'the woman [1]', 'the "
            f"dress [2]'.{one} Never describe the face, hair, skin, body shape or look of a "
            "marked person, or the color, print or details of a marked item - the photo defines "
            "them; describe only pose, action, expression, clothing changes the user asks for, "
            "setting, lighting and camera.")
_GPU_LOCK = threading.Lock()  # zajety, gdy model liczy na karcie - kolejka czeka (release_gpu)
_VISION_BATCH = ContextVar("queued_vision_batch", default=False)
_LLM_LIMIT = config.SETTINGS["engine"].get("vram_limit_gib")
_OWN_LLM = LlmClient(log_path=config.ROOT / "llm_worker.log",
                     options={"max_gpu_mib": _LLM_LIMIT * 1024} if _LLM_LIMIT is not None else None)


def shutdown() -> None:
    """Stop the assistant worker before the studio process exits."""
    _OWN_LLM.close()


def ensure(progress_cb: Optional[ProgressCb] = None) -> Optional[str]:
    """Check the installed runtime and install the selected model files."""
    return runtime_problem() or model_manager.ensure([MODEL_ID], progress_cb)


def _catalog_file(projector: bool = False) -> Optional[str]:
    """Select only the configured model bundle, preserving older custom files on disk."""
    model = config.get_model(model_manager.tiered(MODEL_ID)) or {}
    for spec in model.get("files", []):
        name = spec.get("save_as") or Path(spec["filename"]).name
        if ("mmproj" in name.lower()) == projector and spec.get("dest") == "llm":
            if (config.destination_dir("llm") / name).is_file():
                return name
    return None


def _model_file() -> Optional[str]:
    return _catalog_file()


_LOOK_PROMPT = (
    "Describe only this person's physical appearance for an image generator, in one line of at "
    "most 45 words: exact skin tone (e.g. very pale porcelain, fair, olive, tanned, dark), hair "
    "color, length and style, eye color, face shape and distinctive facial features, body type. "
    "No clothing, no background, no pose. If there is no person, answer NONE.")
_LOOKS_FILE = config.destination_dir("llm") / "looks.json"


@contextmanager
def vision_batch():
    """Keep the vision model loaded between the attempts of one review in the queue worker, then
    hand the GPU back to the queue."""
    from . import queue_manager

    if _VISION_BATCH.get() or not queue_manager.in_worker():
        yield
        return
    batch = _VISION_BATCH.set(True)
    try:
        yield
    finally:
        _VISION_BATCH.reset(batch)
        _OWN_LLM.unload_for_gpu_handover()


def _vision(image_path: str, prompt: str, max_tokens: int, schema: Optional[dict] = None,
            size: int = 512) -> Optional[str]:
    return inspect_frames([image_path], prompt, max_tokens, schema, size)


def inspect_frames(images: Sequence[str], prompt: str, max_tokens: int, schema: Optional[dict] = None,
                   size: int = 512, *, on_error: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Inspect ordered pictures. Queued vision can use the GPU while background vision yields to it."""
    from . import cancel, queue_manager

    if error := ensure():
        if on_error:
            on_error(error)
        return None
    mmproj = _catalog_file(projector=True)
    model = _model_file()
    if not mmproj or not model:
        if on_error:
            on_error("The configured vision weights or projector are missing.")
        return None
    queued = queue_manager.in_worker()
    destination = config.destination_dir("llm")

    def run(gpu):
        return _OWN_LLM.ask("", prompt, model_path=destination / model, mmproj_path=destination / mmproj,
                            gpu=gpu, n_ctx=4096, images=images, image_size=size, schema=schema,
                            max_tokens=max_tokens, temperature=0.2, timeout=600, keep_loaded=queued and _VISION_BATCH.get(),
                            cancel=cancel.TOKEN if queued else None)

    try:
        with _GPU_LOCK:
            if queued or (not queue_manager.paused() and not any(
                    job.status == queue_manager.RUNNING for job in queue_manager.snapshot())):
                return run(gpu=True)
        return run(gpu=False)
    except (LlmError, OSError, ValueError) as error:
        if queued:
            cancel.check()
        if on_error:
            on_error(str(error))
        return None


class VisualReviewUnavailable(ValueError):
    """No valid verdict was obtained. This is not a rejection of the media."""


def inspect_review(images: Sequence[str], prompt: str, max_tokens: int, schema: dict,
                   valid: Callable[[dict], bool], sizes: tuple[int, ...]) -> dict:
    """Retry the same ordered frames when the visual reviewer fails to return a verdict."""
    last_error = "Empty or invalid structured verdict."
    with vision_batch():
        for attempt, size in enumerate(sizes, 1):
            errors = []
            raw = inspect_frames(images, prompt, max_tokens, schema, size=size, on_error=errors.append)
            try:
                verdict = json.loads(raw or "")
            except (json.JSONDecodeError, TypeError):
                verdict = None
            if isinstance(verdict, dict) and valid(verdict):
                return verdict
            last_error = errors[-1] if errors else "Empty or invalid structured verdict."
            last_error = " ".join(last_error.split())[:320]
            print(f"[vision review] Attempt {attempt}/{len(sizes)}: {last_error}", flush=True)
    raise VisualReviewUnavailable(f"Visual review unavailable after {len(sizes)} attempts. {last_error}")


def describe_person(image_path: str, progress_cb: Optional[ProgressCb] = None) -> str:
    """Wyglad osoby ze zdjecia pomocniczego (skora, wlosy, oczy, rysy) - jedna linia po
    angielsku albo "" (brak osoby / blad). Wstawiany do promptow, zeby tekst wspieral zdjecie:
    sama "plaza w Dubaju" opalala blada modelke ze zdjecia. Wynik zapamietany wg zawartosci
    pliku (seria zdjec z ta sama referencja liczy go raz); liczenie ok. 25-45 s."""

    try:
        key = hashlib.sha1(Path(image_path).read_bytes()).hexdigest()
        cache = json.loads(_LOOKS_FILE.read_text(encoding="utf-8")) if _LOOKS_FILE.exists() else {}
    except (OSError, ValueError):
        key, cache = "", {}
    if key and key in cache:
        return cache[key]
    if progress_cb:
        progress_cb(0.0, "describing the person in the image...")
    text = _vision(image_path, _LOOK_PROMPT, 120)
    if text is None:
        return ""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    look = lines[-1].rstrip(".") if lines else ""
    if look.upper().startswith("NONE") or blocked(look):
        look = ""
    if key:
        cache[key] = look
        try:
            _LOOKS_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        except OSError:
            pass
    return look


_PHOTO_PROMPT = (
    "This photo is a reference photo of a character. Return JSON with: person - true if "
    "one real-looking human is clearly visible; age - your best estimate of that person's age in "
    "years; gender - woman or man; caption - one English sentence of at most 30 words "
    "describing ONLY the clothing, pose, facial expression, framing (close-up, half body or "
    "full body), background and lighting. Do not describe the face, hair, skin, eyes, body "
    "shape or age in the caption.")
_PHOTO_SCHEMA = {
    "type": "object",
    "properties": {"person": {"type": "boolean"}, "age": {"type": "integer"},
                   "gender": {"type": "string", "enum": ["woman", "man"]},
                   "caption": {"type": "string"}},
    "required": ["person", "age", "gender", "caption"],
}


def check_photo(image_path: str, style: str = styles.REALISTIC) -> Optional[dict]:
    """A character's reference photo: {"person", "age", "gender", "caption"}, or None (an error).
    The age is a hard gate: a photo of a person who looks under 18 is never used for a character, in every
    style. style - the character's style (app/styles.py): an animated or drawn character counts as a person."""
    look = styles.get(style)
    # a style that starts from Realistic draws real-looking people: the gate for a real person's photo
    prompt = _PHOTO_PROMPT if look.family == styles.REALISTIC else _PHOTO_PROMPT.replace(
        styles.get(styles.REALISTIC).subject, look.subject, 1) + (
        " The picture may be a 3D animated render or a drawing; estimate the age the character is drawn as.")
    text = _vision(image_path, prompt, 160, schema=_PHOTO_SCHEMA, size=448)
    if text is None:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return None
    age, person, gender, caption = (data.get(k) for k in ("age", "person", "gender", "caption"))
    if (type(age) is not int or not 0 <= age <= 130 or type(person) is not bool
            or gender not in ("woman", "man") or not isinstance(caption, str)
            or not caption.strip()):
        return None
    caption = caption.strip().rstrip(".")
    return {"person": person, "age": age, "gender": gender,
            "caption": "" if blocked(caption) else refs.neutral_look(caption)}


# The gate of an animal or creature character's photo counts what it sees: a human there is refused (and named as a
# minor when young), so nobody becomes a "creature" by a pair of cat ears, while a talking dog in a hoodie is no human.
_CREATURE_PROMPT = (
    "This picture is a reference picture of a film character. It may be a photo, a 3D animated render or a drawing "
    "(art style: {medium}). Return JSON with: humans - how many humans are visible, real, 3D or drawn, counting every "
    "human-looking person, also one with animal ears, a tail, horns or wings (an elf, a fairy or a person with cat "
    "ears or a tail IS a human); youngest_human_age - your best estimate of the age in years of the youngest visible "
    "human, 0 when there is none; creatures - how many animals or other non-human creatures are visible: an "
    "anthropomorphic animal that stands upright, wears clothes or talks is NOT a human, count it as a creature; "
    "baby - true only if the animal or creature is clearly drawn as a newborn or baby animal (a puppy, kitten, cub, "
    "chick or hatchling), false when it is grown, even if it looks cute, small, round or big-eyed in a cartoon way; "
    "caption - one English sentence of at most 30 words describing ONLY the clothing, pose, expression, framing "
    "(close-up, half body or full body), background and lighting.")
_CREATURE_SCHEMA = {
    "type": "object",
    "properties": {"humans": {"type": "integer"}, "youngest_human_age": {"type": "integer"},
                   "creatures": {"type": "integer"}, "baby": {"type": "boolean"}, "caption": {"type": "string"}},
    "required": ["humans", "youngest_human_age", "creatures", "baby", "caption"],
}
_MAX_COUNT = 20  # more humans or creatures than this in a reference picture is no answer to trust


def check_creature_photo(image_path: str, style: str) -> Optional[dict]:
    """An animal or creature character's reference photo (app/characters.py, kind "creature"): {"humans",
    "youngest_human_age", "creatures", "baby", "caption"}, or None (an error or an answer out of range). humans counts
    the humans the vision model sees (real, 3D or drawn; one with animal ears or a tail is a human, an anthropomorphic
    animal is not), youngest_human_age is the age of the youngest of them (0 when there is none), creatures the
    animals or creatures, and baby whether it is drawn as a baby animal (cute, small or big-eyed alone is not: on
    2026-11-16 the model called four cartoon pictures of a grown corgi grown and a drawn puppy and kitten babies).
    style - the character's style (app/styles.py): its medium is named to the model."""
    look = styles.of(style)
    medium = look.medium or styles.BASES[look.family].medium  # a custom style draws in its base's medium
    text = _vision(image_path, _CREATURE_PROMPT.format(medium=medium), 160, schema=_CREATURE_SCHEMA, size=448)
    if text is None:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return None
    humans, age, creatures, caption = (data.get(k) for k in ("humans", "youngest_human_age", "creatures", "caption"))
    if any(type(count) is not int or not 0 <= count <= _MAX_COUNT for count in (humans, creatures)):
        return None
    if type(data.get("baby")) is not bool:  # asked, but no reason to refuse: the owner's decision of 2026-11-16
        return None
    if humans and (type(age) is not int or not 0 <= age <= 130):  # the age matters only when a human is there
        return None
    caption = caption.strip().rstrip(".") if isinstance(caption, str) else ""
    return {"humans": humans, "youngest_human_age": age if humans else 0, "creatures": creatures,
            "baby": data["baby"], "caption": "" if blocked(caption) else caption}


# The gate of a place's or an object's reference picture (app/items.py) only asks for the people in it: a picture with
# someone who looks under 18 is refused, nothing else about a place or an object is.
_ITEM_PROMPT = (
    "This picture is a reference picture of a place or an object for a film. It may be a photo, a 3D animated render "
    "or a drawing (art style: {medium}). Return JSON with: humans - how many humans are visible, real, 3D or drawn, "
    "counting every human-looking person, also one on a screen, a poster or a photo in the picture, and one with "
    "animal ears, a tail, horns or wings; youngest_human_age - your best estimate of the age in years of the youngest "
    "visible human, 0 when there is none.")
_ITEM_SCHEMA = {
    "type": "object",
    "properties": {"humans": {"type": "integer"}, "youngest_human_age": {"type": "integer"}},
    "required": ["humans", "youngest_human_age"],
}


def check_item_photo(image_path: str, style: str = styles.REALISTIC) -> Optional[dict]:
    """A place's or an object's reference picture (app/items.py): {"humans", "youngest_human_age"}, or None (an error
    or an answer out of range). humans counts the humans the vision model sees (real, 3D or drawn, on a screen or a
    poster too), youngest_human_age is the age of the youngest of them (0 when there is none). style - the item's
    style (app/styles.py): its medium is named to the model."""
    look = styles.of(style)
    medium = look.medium or styles.BASES[look.family].medium  # a custom style draws in its base's medium
    text = _vision(image_path, _ITEM_PROMPT.format(medium=medium), 60, schema=_ITEM_SCHEMA, size=448)
    if text is None:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return None
    humans, age = data.get("humans"), data.get("youngest_human_age")
    if type(humans) is not int or not 0 <= humans <= _MAX_COUNT:
        return None
    if humans and (type(age) is not int or not 0 <= age <= 130):  # the age matters only when a human is there
        return None
    return {"humans": humans, "youngest_human_age": age if humans else 0}


def release_gpu(timeout: float = 120.0) -> None:
    """Wait for the assistant to hand the GPU back to the queue."""
    _OWN_LLM.unload_for_gpu_handover()
    if _GPU_LOCK.acquire(timeout=timeout):
        try:
            _OWN_LLM.unload_for_gpu_handover()
        finally:
            _GPU_LOCK.release()


def _clean(text: str) -> str:
    text = text.replace("[end of text]", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^(here('s| is)[^:\n]*:|prompt:)\s*", "", text, flags=re.I).strip()
    text = text.replace("\\n", "\n")
    # kazde "Ujęcie N:" od nowej linii - tylko tak aplikacja dzieli film na ujecia
    text = re.sub(r"\s*\b((?:Uj[ęe]cie|Shot) \d+\s*:)", r"\n\1", text).strip()
    return text.strip("`\"' \n")


def _generate(system: str, user: str, gpu: bool, schema: Optional[dict] = None,
              max_tokens: int = 700, temp: float = 0.8) -> Tuple[bool, str]:
    """Generate an answer using the installed binding and an optional JSON schema."""
    from . import cancel, queue_manager

    model = _model_file()
    if not model:
        return False, "The prompt assistant model is missing."
    try:
        answer = _OWN_LLM.ask(system, user, model_path=config.destination_dir("llm") / model,
                              gpu=gpu, schema=schema, max_tokens=max_tokens, temperature=temp,
                              cancel=cancel.TOKEN if queue_manager.in_worker() else None)
    except (LlmError, OSError, ValueError) as exc:
        return False, f"Prompt assistant engine error: {exc}"
    return True, answer


def ask(system: str, user: str, schema: Optional[dict] = None, max_tokens: int = 700,
        check: Optional[Callable[[str], bool]] = None, retry_suffix: str = "",
        progress_cb: Optional[ProgressCb] = None, desc: str = "Writing the prompt...",
        frac: float = 0.3, temp: float = 0.8) -> Tuple[bool, str]:
    """Odpowiedz modelu (surowy tekst). Na karcie, gdy kolejka nic nie generuje, inaczej na
    procesorze. check - gdy odpowiedz go nie spelnia, druga proba z retry_suffix."""
    from . import queue_manager  # karta tylko wtedy, gdy kolejka nic nie generuje

    def once(gpu: bool) -> Tuple[bool, str]:
        ok, text = _generate(system, user, gpu, schema, max_tokens, temp)
        if ok and check is not None and not check(text):
            ok, text = _generate(system, user + retry_suffix, gpu, schema, max_tokens, temp)
        return ok, text

    if queue_manager.paused():  # pauza karty (np. gra) - pomocnik nie zabiera jej VRAM
        if progress_cb:
            progress_cb(frac, desc + " (CPU — GPU paused)")
        return once(gpu=False)
    if any(j.status == queue_manager.RUNNING for j in queue_manager.snapshot()):
        if progress_cb:
            progress_cb(frac, desc + " (CPU — GPU busy)")
        return once(gpu=False)
    with _GPU_LOCK:
        # The queue can claim the card between the first status check and this lock.
        if not queue_manager.paused() and not any(j.status == queue_manager.RUNNING for j in queue_manager.snapshot()):
            if progress_cb:
                progress_cb(frac, desc)
            return once(gpu=True)
    if progress_cb:
        progress_cb(frac, desc + " (CPU — GPU unavailable)")
    return once(gpu=False)


def characters_context(people: Sequence[dict]) -> str:
    """The saved characters a prompt may use, for write_prompt's context: each as the @name Contsy recognises
    (app/cast.py) and what it is. Never its look: the writer copied it into the prompt (2026-11-16), and the
    character's photos define it."""
    rows = []
    for person in people:
        name = (person.get("slug") or person.get("name") or "").strip()
        if not name:
            continue
        kind = ("an animal or creature, not a human" if person.get("kind") == "creature" else
                "a man" if person.get("gender") == "man" else "a woman")
        rows.append(f"@{name} ({kind})")
    if not rows:
        return ""
    return ("Saved characters of the user's studio - when the idea mentions one, write it exactly as here and do not "
            "describe its looks: " + "; ".join(rows) + ".")


def items_context(pool: Sequence[dict]) -> str:
    """The saved places and items a prompt may use, for write_prompt's context: each by the name Contsy finds in a
    picture text (app/items.py: its photo then comes into the picture) and what it is. Never its look: its photos
    define it."""
    rows = []
    for item in pool:
        name = " ".join(str(item.get("name") or "").split())
        if name:
            rows.append(f"{name} ({'a place' if item.get('kind') == 'place' else 'an object'})")
    if not rows:
        return ""
    return ("Saved places and items of the user's studio - when the idea mentions one, write its name exactly as here "
            "(Contsy then puts its photo into the picture) and do not describe its looks: " + "; ".join(rows) + ".")


# An AI edit of a made Timeline scene (the owner's request of 2026-11-17, app/timeline.py edit): the planner turns what
# the user wants changed into the scene's new texts and, when the picture must show the change from its first frame,
# an instruction for the image editor that changes that frame.
_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "picture_text": {"type": "string"},
        "lines": {"type": "array", "items": {"type": "string"}},
        "frame_change": {"type": "string"},
    },
    "required": ["picture_text", "lines", "frame_change"],
    "additionalProperties": False,
}
_EDIT_PLAN = (
    "You plan an edit of one scene of a film made in Contsy. The scene already has a video. The user says what should "
    "change, in any language and often colloquially: understand it literally and change only that. Reply with JSON: "
    "picture_text, lines, frame_change. "
    "picture_text: the scene's whole picture text after the change, in English. Copy every part the change does not "
    "concern word for word; when the change does not concern it, copy it unchanged. " + _CONTSY_NAMES + _CONTSY_SHOWN
    + "When the scene's spoken lines are in its picture text (in quotes after says, asks, shouts...), change what is "
    "said there and keep who says it. " + _CONTSY_STYLE
    + "lines: when the scene has separate dialogue lines (Line 1, Line 2...), the text of every line after the change, "
    "in the same order and number unless the user adds or removes one; otherwise an empty list. Spoken lines are "
    "English; a line the user wrote in quotes stays word for word. A character speaks about 2.5 words per second. "
    "frame_change: one or two English sentences for an image editor that changes the scene's FIRST FRAME so the change "
    "is seen from the start: text on a screen or a sign, colors, clothing, objects, the place, light, time of day, "
    "weather. Name what to change and into what, for example: Change the text on the large screen from \"CRYPTO "
    "NEWS\" to \"BREAKING NEWS\". Leave it empty when the change is only about what happens during the scene (action, "
    "motion, an expression, the camera) or only about what is said. Never describe the style. Every person stays an "
    "adult. " + _ANIMALS + "Inside the JSON strings write every quotation mark as “ and ”, never as a straight "
    "double quote.")
# A straight double quote inside a JSON string ended the string under the schema's grammar: on 2026-11-17 the planner
# returned '... a large screen displays the text' and nothing after it. The scene's texts go to it with typographic
# quotes, which it copies, and come back with the quote marks they had.
_STRAIGHT = re.compile(r'"([^"\n]*)"')
_QUOTED = re.compile(r'[„“"«][^„“”"«»]{2,}?[”"»]')


def _curly(text: str) -> str:
    return _STRAIGHT.sub(lambda match: f"“{match.group(1)}”", text)


def _cut_short(text: str, before: str) -> bool:
    """A planned picture text that stops inside a quote: fewer quotes than the scene had, and no end mark."""
    text = text.rstrip()
    return len(_QUOTED.findall(text)) < len(_QUOTED.findall(before)) and not re.search(r'[.!?”"»]$', text)


def _plan_fields(raw: str) -> Optional[dict]:
    """The planner's JSON answer as {"picture_text", "lines", "frame_change"}, or None when it is not one."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip().strip("`")
    text = re.sub(r"^json\s*", "", text, flags=re.I)
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if (not isinstance(data, dict) or not isinstance(data.get("picture_text"), str)
            or not isinstance(data.get("frame_change"), str) or not isinstance(data.get("lines"), list)
            or not all(isinstance(line, str) for line in data["lines"])):
        return None
    return {"picture_text": data["picture_text"], "lines": data["lines"], "frame_change": data["frame_change"]}


def plan_edit(change: str, picture: str, lines: Sequence[str] = (), *, style: str = styles.REALISTIC,
              context: str = "", names: Sequence[str] = (),
              progress_cb: Optional[ProgressCb] = None) -> Tuple[bool, object]:
    """What an AI edit of a made Timeline scene changes: from the user's words `change` (any language) and the
    scene's picture text and separate dialogue `lines`, its new picture text and lines (English, written for Contsy)
    and the instruction that changes its first frame ("" when the picture needs none). Returns (True, {"picture_text",
    "lines", "frame_change"}) or (False, message). style: the film's (app/styles.py); context: facts for the planner
    (characters_context, items_context, the scene's length); names: the saved characters' @names the scene uses, kept
    as written. The minors guard runs on the request and again on the plan; the film's own style words are taken out
    of the picture text (Contsy adds them)."""
    change, picture = (change or "").strip(), (picture or "").strip()
    lines = [str(line).strip() for line in lines or () if str(line).strip()]
    if not change:
        return False, "Describe the change in any language."
    if blocked(change, picture, *lines):
        return False, BLOCK_MSG
    err = ensure(progress_cb)
    if err:
        return False, err
    names = [name for name in dict.fromkeys(names or ()) if name]
    straight = '"' in " ".join([picture, *lines]) and not re.search(r"[“”]", " ".join([picture, *lines]))
    user = [f"Change: {_curly(change)}", "", "The scene's picture text:", _curly(picture) or "(empty)", ""]
    if lines:
        user += ["Its dialogue lines:", *(f"Line {number}: {_curly(text)}" for number, text in enumerate(lines, 1))]
    else:
        user.append("It has no separate dialogue lines.")
    if names:
        user.append("Write these saved characters exactly like this: " + ", ".join(f"@{name}" for name in names))
    user.append("Write the texts in English.")

    def fine(raw: str) -> bool:
        data = _plan_fields(raw)
        if data is None:
            return False
        text = data["picture_text"]
        named = all(re.search(rf"(?<![\w@])@{re.escape(name)}(?!\w)", text, re.I) for name in names)
        return (bool(text.strip()) and named and not _cut_short(text, picture) and english(text)
                and english(data["frame_change"]) and all(english(line) for line in data["lines"]))

    look = styles.get(style) if styles.valid(style) else styles.get(styles.REALISTIC)
    system = _EDIT_PLAN + ("" if look.realistic else
                           f" The film is in this style, which Contsy adds to every picture itself: {look.video}.")
    if context:
        system += f"\n{context}"
    ok, raw = ask(system, "\n".join(user), schema=_EDIT_SCHEMA, max_tokens=900, check=fine, retry_suffix=_REMIND,
                  progress_cb=progress_cb, desc="Planning the edit...", temp=0.4)
    if not ok:
        return False, raw
    data = _plan_fields(raw)
    if data is None:
        return False, "The assistant's plan could not be read. Try again."
    if _cut_short(data["picture_text"], picture):
        return False, "The assistant's plan came back cut short. Try again."

    def marks(text: str) -> str:  # the quote marks the scene's texts had
        text = " ".join(text.split())
        return text.replace("“", '"').replace("”", '"') if straight else text

    # not _clean: it would cut the closing quote of a text the picture shows at the end
    text = _without_style(marks(data["picture_text"]), style if styles.valid(style) else styles.REALISTIC)
    planned = {"picture_text": text or picture,
               "lines": [marks(line) for line in data["lines"] if line.strip()] if lines else [],
               "frame_change": " ".join(data["frame_change"].split())}
    if blocked(planned["picture_text"], planned["frame_change"], *planned["lines"]):
        return False, BLOCK_MSG
    return True, planned


def _without_style(text: str, style: str) -> str:
    """The prompt without the film's own style words, which Contsy adds to every prompt itself (the writer repeated
    them although it was told not to)."""
    look = styles.get(style) if styles.valid(style) else styles.get(styles.REALISTIC)
    if look.realistic:
        return text
    for words in (look.video, look.image):
        text = text.replace(words, "")
    text = re.sub(r"\s*,\s*([.!?])", r"\1", text)
    text = re.sub(r"([.!?])\s*[.,]+", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip(" ,")


def write_prompt(idea: str, kind: str, current: str = "",
                 progress_cb: Optional[ProgressCb] = None,
                 n_refs: int = 0, style: str = styles.REALISTIC, context: str = "",
                 names: Sequence[str] = ()) -> Tuple[bool, str]:
    """Prompt z pomyslu (albo poprawiony obecny prompt). kind: 'image' / 'video'.
    n_refs - ile zdjec-referencji jest dolaczonych w zakladce (model wskaze je [1], [2]...).
    style - styl filmu (app/styles.py, system_prompt).
    context - facts about where the prompt goes (characters_context, a Timeline scene's length and the scene before
    it), added to the writer's instructions. names - the saved characters the idea names (their slugs): the prompt
    must write each as @slug, or the writer is asked again (on 2026-11-16 it made "chain_dog" a 21-year-old woman).
    The film's own style words are taken out of the answer: Contsy adds them to every prompt.
    The minors guard (blocked) runs on the idea and again on the finished prompt.
    Zwraca (ok, prompt_albo_komunikat)."""
    idea, current = mark_refs((idea or "").strip(), n_refs), (current or "").strip()
    if not idea and not current:
        return False, "Enter an idea in any language, or a prompt to improve."
    if blocked(idea, current):
        return False, BLOCK_MSG
    err = ensure(progress_cb)
    if err:
        return False, err
    if idea and current:
        user = (f"My idea: {idea}\n\nRewrite and improve this prompt according to "
                f"the idea, keep what still fits:\n{current}")
    elif idea:
        user = f"My idea: {idea}"
    else:
        user = f"Improve this prompt, make it more detailed and precise:\n{current}"
    marks = refs.numbers(f"{idea} {current}")
    if marks:
        user += "\nKeep these image markers in the prompt: " + ", ".join(f"[{n}]" for n in marks)
    names = [name for name in dict.fromkeys(names or ()) if name]
    quoted = bool(re.search(r"[„“\"«]", f"{idea} {current}"))  # a line the user wrote word for word stays as written
    if names:
        user += ("\nSaved characters in this idea - write each exactly like this, as the one doing what the idea "
                 "says, never as a person of some age: " + ", ".join(f"@{name}" for name in names))
    user += "\nWrite the prompt in English."

    def fine(raw: str) -> bool:
        """Po angielsku (poza kwestiami w cudzyslowie), ze wszystkimi numerami obrazow,
        a gdy sa dolaczone zdjecia - z co najmniej jednym numerem."""
        text = _clean(raw)
        got = refs.numbers(text)
        named = all(re.search(rf"(?<![\w@])@{re.escape(name)}(?!\w)", text, re.I) for name in names)
        # the films are in English: a line the user did not quote word for word is English too
        lines_english = quoted or len(re.findall(r"[ąćęłńóśźż]", text, re.I)) / max(len(text), 1) < 0.004
        return english(text) and set(marks) <= set(got) and (bool(got) or not n_refs) and named and lines_english

    system = system_prompt(kind, style) + (refs_rule(n_refs) if n_refs else "") + (f"\n{context}" if context else "")
    remind = _REMIND + (" Write " + ", ".join(f"@{name}" for name in names) + " exactly like that." if names else "")
    ok, text = ask(system, user, check=fine, retry_suffix=remind, progress_cb=progress_cb)
    if not ok:
        return False, text
    text = _without_style(_clean(text), style)
    # nadal bez numeru przy jednym zdjeciu - osoba/rzecz z opisu to ta ze zdjecia
    text = re.sub(r"\s*\[(\d{1,2})\]", lambda m: m.group(0) if int(m.group(1)) <= n_refs else "",
                  text)
    if n_refs == 1 and not refs.numbers(text):
        text = f"The subject is the one from the reference photo [1]. {text}".strip()
    if not text:
        return False, "The model returned no prompt. Try again."
    if blocked(text):
        return False, BLOCK_MSG
    return True, text


_STYLE_WORDS = (
    "You turn a short description of a film's visual style, written in any language, into English style words for "
    "image and video generators. Reply with ONE line: 5 to 25 comma-separated English words or short phrases about "
    "how the pictures look: medium, line work, shading, colors, texture, lighting and era. Describe visual traits "
    "instead of naming films, studios, brands, artists or real people. Never describe characters, ages, audiences or "
    "a story. No quotes, brackets, colons, @ or #, no preamble.")


def style_words(description: str, progress_cb: Optional[ProgressCb] = None) -> Tuple[bool, str]:
    """English look words for a custom style from the owner's description in any language (the Styles page's
    "Translate to English"): (True, words that styles.clean_look accepts) or (False, a message)."""
    text = " ".join((description or "").split())
    if not text:
        return False, "Describe the look first."
    if blocked(text):
        return False, BLOCK_MSG
    problem = ensure(progress_cb)
    if problem:
        return False, problem
    ok, raw = ask(_STYLE_WORDS, "Style description: " + text, max_tokens=160, temp=0.3,
                  check=lambda answer: english(_clean(answer)),
                  retry_suffix="\nIMPORTANT: answer in English only.", progress_cb=progress_cb,
                  desc="Translating the style...")
    if not ok:
        return False, raw
    line = (_clean(raw).splitlines() or [""])[0]
    line = re.sub(r"[^\w ,.;'()/&+!?-]", " ", line)
    if len(line) > styles.MAX_LOOK:
        line = line[:styles.MAX_LOOK].rsplit(",", 1)[0]
    try:
        words = styles.clean_look(line)
    except ValueError as error:
        return False, f"The assistant gave no usable style words ({error}). Write them in English."
    if not english(words):
        return False, "The assistant did not answer in English. Write the style words in English."
    problem = style_problem(styles.custom_id(styles.OTHER, words))
    if problem:
        return False, problem
    return True, words


def english(text: str) -> bool:
    """Tekst po angielsku - poza kwestiami w cudzyslowie prawie bez polskich liter."""
    outside = re.sub(r'„[^”]*”|“[^”]*”|«[^»]*»|"[^"]*"', "", text)
    outside = re.sub(r"(?<!\w)@\w+", "", outside)  # @voice tags may carry Polish letters
    return len(re.findall(r"[ąćęłńóśźż]", outside, re.I)) / max(len(outside), 1) < 0.004
