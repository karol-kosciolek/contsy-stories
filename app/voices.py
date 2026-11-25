# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Voice library: the bundled voices and the standalone voices the user imports from a sample (a narrator,
unnamed speakers). A character's voice is not here: it is stored with the character (app/characters.py) and
reached through the key "character:<slug>", which `get` resolves.

A voice is a short 48 kHz mono sample. VoxCPM2 clones its timbre for every spoken line
(app/tts.py), so a voice sounds the same in every video. Bundled voices live in
assets/voices with voices.json (synthetic voices made with tools/make_voices.py). Imported
voices are user data in voices/<slug>/ (sample.wav + voice.json, never committed): the
uploaded or recorded sample is converted with PyAV, trimmed and normalised.

A voice is named by one word, for example Hanna. In a video prompt "@Hanna says „…”" speaks
that line with it (app/dialogue.py). Lines that name no voice use the voice of the character who
says them, or the per-gender default chosen on the Voices page (app/ui_voices.py; stored by app/prefs.py).
"""
from __future__ import annotations

import json
import re
import shutil
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config, dialogue, ffmpeg_utils, llm, prefs, trash
from .persistence import atomic_text

BUILTIN_DIR = config.ROOT / "assets" / "voices"
MANIFEST = BUILTIN_DIR / "voices.json"
USER_DIR = config.STATE_ROOT / "voices"
GENDERS = ("female", "male")
GENDER_LABEL = {"female": "Woman", "male": "Man"}
AGES = ("20s", "30s", "40s", "50s", "60s", "70s")
# Pitch bands (median F0 in Hz) per gender: below = low voice, above = high voice
PITCH_BANDS = {"female": (175.0, 230.0), "male": (105.0, 140.0)}
MIN_SECONDS = 2.0        # speech that must remain after trimming silence
MAX_SECONDS = 12.0       # a longer reference only costs time and GPU memory
SCAN_SECONDS = 90        # how much of a long recording is read at all
MAX_BYTES = 200 * 1024 * 1024
NAME_RE = re.compile(r"^\w{2,24}$")
CONSENT_MSG = ("Confirm that the sample is your own voice, a synthetic voice, or the voice of an "
               "adult who agreed to its cloning.")


@dataclass
class Voice:
    name: str
    slug: str
    gender: str              # "female" / "male"
    path: str                # 48 kHz mono wav sample
    description: str = ""    # character of the voice, shown in lists
    builtin: bool = False
    seconds: float = 0.0
    created: float = 0.0
    f0_hz: float = 0.0       # median pitch of the sample (0 = not measured)
    age: str = ""            # one of AGES or "" (not set)

    @property
    def label(self) -> str:
        """Name with the type, age and description, for lists in the UI."""
        parts = [GENDER_LABEL.get(self.gender, "")]
        if self.age and not self.description.startswith(self.age):
            parts.append(self.age)
        parts.append(self.description)
        extra = ", ".join(x for x in parts if x)
        return f"{self.name} — {extra}" if extra else self.name


def pitch(voice: Voice) -> str:
    """"low" / "medium" / "high" for the voice's gender, "" when the pitch is unknown."""
    if not voice.f0_hz or voice.f0_hz <= 0:
        return ""
    low, high = PITCH_BANDS.get(voice.gender, (150.0, 200.0))
    return "low" if voice.f0_hz < low else "high" if voice.f0_hz > high else "medium"


def measure_pitch(path: str | Path) -> float:
    """Median F0 of the voiced parts of a sample in Hz (librosa.pyin); 0.0 when it fails."""
    try:
        import librosa
        import numpy as np

        y, _ = librosa.load(str(path), sr=16000, mono=True)
        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=420, sr=16000)
        values = f0[voiced & np.isfinite(f0)]
        return float(np.median(values)) if values.size else 0.0
    except Exception:  # noqa: BLE001 - the pitch only sorts the library
        return 0.0


# Windows device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9) cannot be folder names: a slug
# equal to one gets an underscore, so a voice named "Nul" still gets a usable folder
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def slugify(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.replace("ł", "l").replace("Ł", "L"))
    text = re.sub(r"[^a-z0-9]+", "_", text.encode("ascii", "ignore").decode().lower()).strip("_")[:24]
    return f"{text}_" if text in _RESERVED else text


def _user_folder(slug: str) -> Path:
    if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9_]{1,24}", slug):
        raise ValueError("Invalid voice identifier.")
    folder = USER_DIR / slug
    if folder.resolve().parent != USER_DIR.resolve():
        raise ValueError("Voice folder is outside the voice library.")
    return folder


def _voice(data: dict, path: Path, builtin: bool) -> Optional[Voice]:
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return None
    if data.get("gender") not in GENDERS or not path.is_file():
        return None
    slug = data.get("slug") if isinstance(data.get("slug"), str) else slugify(data["name"])
    try:
        seconds, created = float(data.get("seconds") or 0.0), float(data.get("created") or 0.0)
        f0 = float(data.get("f0_hz") or 0.0)
    except (TypeError, ValueError):
        seconds, created, f0 = 0.0, 0.0, 0.0
    age = data.get("age") if data.get("age") in AGES else ""
    return Voice(name=data["name"], slug=slug or slugify(data["name"]), gender=data["gender"],
                 path=str(path), description=str(data.get("description") or ""),
                 builtin=builtin, seconds=seconds, created=created, f0_hz=f0, age=age)


def builtin() -> List[Voice]:
    """Bundled voices in the order of assets/voices/voices.json (missing files are skipped)."""
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: List[Voice] = []
    for entry in data.get("voices", []) if isinstance(data, dict) else []:
        if isinstance(entry, dict) and isinstance(entry.get("file"), str):
            voice = _voice(entry, BUILTIN_DIR / Path(entry["file"]).name, True)
            if voice:
                out.append(voice)
    return out


def imported() -> List[Voice]:
    """Voices the user added, oldest first."""
    if not USER_DIR.exists():
        return []
    out: List[Voice] = []
    for folder in USER_DIR.iterdir():
        if not folder.is_dir():
            continue
        try:
            data = json.loads((folder / "voice.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        voice = _voice(data, folder / "sample.wav", False)
        if voice and voice.slug == folder.name:
            out.append(voice)
    return sorted(out, key=lambda v: v.created)


def all_voices() -> List[Voice]:
    return builtin() + imported()


def _pool(voices: Optional[List[Voice]]) -> List[Voice]:
    return voices if voices is not None else all_voices()


def get(key: Optional[str], voices: Optional[List[Voice]] = None) -> Optional[Voice]:
    """Voice by slug or by name (case-insensitive); a character's own voice by its key "character:<slug>"."""
    if not key or not isinstance(key, str):
        return None
    if key.startswith(dialogue.CHARACTER_VOICE):
        from . import characters

        return characters.own_voice(key)
    k = key.strip().lstrip("@").casefold()
    return next((v for v in _pool(voices) if v.slug == k or v.name.casefold() == k), None)


def tokens(voices: Optional[List[Voice]] = None) -> Dict[str, Tuple[str, str]]:
    """Casefolded names and slugs -> (slug, gender): what app/dialogue.py recognises in a prompt."""
    out: Dict[str, Tuple[str, str]] = {}
    for v in _pool(voices):
        out.setdefault(v.slug, (v.slug, v.gender))
        out.setdefault(v.name.casefold(), (v.slug, v.gender))
    return out


def defaults(voices: Optional[List[Voice]] = None) -> Dict[str, str]:
    """Default voice slug per gender: the choice made on the Voices page (prefs default_<gender>)
    when it still exists and has that gender, else the first voice of the gender."""
    pool = _pool(voices)
    saved = prefs.load()
    out: Dict[str, str] = {}
    for gender in GENDERS:
        chosen = get(saved.get(f"default_{gender}", ""), pool)
        if chosen is None or chosen.gender != gender:
            chosen = next((v for v in pool if v.gender == gender), None)
        out[gender] = chosen.slug if chosen else ""
    return out


def catalog(voices: Optional[List[Voice]] = None) -> Dict[str, List[str]]:
    """Per gender: all slugs with the default voice first (for dialogue.voices_for)."""
    pool = _pool(voices)
    chosen = defaults(pool)
    out: Dict[str, List[str]] = {}
    for gender in GENDERS:
        slugs = [v.slug for v in pool if v.gender == gender]
        if chosen[gender] in slugs:
            slugs.remove(chosen[gender])
            slugs.insert(0, chosen[gender])
        out[gender] = slugs
    return out


def for_character(ch: Optional[dict],
                  voices: Optional[List[Voice]] = None) -> Optional[Tuple[str, str]]:
    """(slug, gender) of the character's voice when one is set and still exists."""
    if not ch or not isinstance(ch, dict):
        return None
    voice = get(ch.get("voice") or "", voices)
    return (voice.slug, voice.gender) if voice else None


def reference_path(key: Optional[str]) -> Optional[str]:
    voice = get(key)
    return voice.path if voice else None


def choices(voices: Optional[List[Voice]] = None) -> List[Tuple[str, str]]:
    """(label, slug) pairs for dropdowns."""
    return [(v.label, v.slug) for v in _pool(voices)]


def check_name(name: str, voices: Optional[List[Voice]] = None) -> str:
    """Why the name cannot be used for a new voice, or ''."""
    name = (name or "").strip()
    if not name:
        return "Enter a voice name."
    if not NAME_RE.match(name) or name[0].isdigit() or not slugify(name):
        return ("Use one word of 2–24 letters or digits, not starting with a digit, as the name "
                "(for example Hanna): it is what you write after @ in a prompt.")
    if dialogue.is_gender_word(name):
        return "This word already means a woman or a man in prompts. Choose a different name."
    if llm.blocked(name):
        return llm.BLOCK_MSG
    from . import characters

    if any(c["name"].casefold() == name.casefold() for c in characters.all_characters()):
        return (f"‘{name}’ is the name of a character (in prompts it means that "
                "character). Choose another voice name.")
    pool = _pool(voices)
    if get(name, pool) or any(v.slug == slugify(name) for v in pool):
        return f"A voice named ‘{name}’ already exists. Choose another name."
    return ""


def _convert(src: Path, dst: Path) -> Tuple[bool, str]:
    """Any audio -> 48 kHz mono 16-bit wav: silence trimmed at both ends, at most MAX_SECONDS,
    loudness normalised. Only the first SCAN_SECONDS of a long recording are read."""
    return ffmpeg_utils.prepare_voice_sample(src, dst, SCAN_SECONDS, MAX_SECONDS)


def create(name: str, gender: str, sample: Optional[str], consent: bool,
           description: str = "", age: str = "") -> Tuple[Optional[Voice], str]:
    """Import a voice from an audio sample. Returns (voice, message)."""
    name = (name or "").strip()
    description = " ".join((description or "").split())[:80]
    age = age if age in AGES else ""
    error = check_name(name)
    if error:
        return None, error
    return _import_voice(name, slugify(name), gender, sample, consent, description, age)


def _import_voice(name: str, slug: str, gender: str, sample: Optional[str], consent: bool,
                  description: str, age: str) -> Tuple[Optional[Voice], str]:
    if gender not in GENDERS:
        return None, "Choose whether it is a woman's or a man's voice."
    if not consent:
        return None, CONSENT_MSG
    if description and llm.blocked(description):
        return None, llm.BLOCK_MSG
    src = Path(sample) if sample else None
    if src is None or not src.is_file():
        return None, "Upload or record a voice sample first."
    if src.stat().st_size > MAX_BYTES:
        return None, "The sample file is too large (over 200 MB). Use a shorter recording."
    folder = _user_folder(slug)
    try:
        folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return None, f"A voice named ‘{name}’ already exists. Choose another name."
    target = folder / "sample.wav"
    ok, msg = _convert(src, target)
    if not ok:
        shutil.rmtree(folder, ignore_errors=True)
        detail = msg.strip().splitlines()[-1] if msg.strip() else "unsupported audio"
        return None, (f"Could not read the sample ({detail[:120]}). Use a WAV, MP3, M4A, OGG "
                      "or FLAC recording.")
    seconds = (ffmpeg_utils.probe_duration(target) or 0.0) if target.exists() else 0.0
    if seconds < MIN_SECONDS:
        shutil.rmtree(folder, ignore_errors=True)
        if not seconds:
            return None, ("No speech was found in the sample: it is silent or too quiet. Record "
                          "3–12 seconds of one person speaking clearly, close to the microphone.")
        return None, (f"Only {seconds:.1f} s of speech was found. Record 3–12 seconds of one "
                      "person speaking clearly, without music or long pauses.")
    voice = Voice(name=name, slug=slug, gender=gender, path=str(target), description=description,
                  builtin=False, seconds=round(seconds, 2), created=time.time(),
                  f0_hz=round(measure_pitch(target)), age=age)
    try:
        atomic_text(folder / "voice.json", json.dumps(
            {"name": name, "slug": slug, "gender": gender, "age": age, "description": description,
             "seconds": voice.seconds, "f0_hz": voice.f0_hz, "created": voice.created,
             "consent": True, "source": src.name}, ensure_ascii=False, indent=1))
    except OSError:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    tag = name if slugify(name) == slug else slug
    return voice, (f"Voice ‘{name}’ added ({seconds:.1f} s sample). Write @{tag} before a "
                   f"spoken line, for example: @{tag} says “Hello!”")


def delete(slug: str) -> Tuple[bool, str]:
    """Move an imported voice to the Recycle Bin; built-in voices stay. Older characters and the
    defaults that used it go back to automatic. Returns (ok, message)."""
    from . import characters

    voice = get(slug)
    if not voice:
        return False, "Voice not found."
    if voice.builtin:
        return False, "Built-in voices cannot be removed."
    if not trash.move(_user_folder(voice.slug)):
        return False, ("Could not move the voice to the Recycle Bin. Close files that use it "
                       "and try again.")
    saved = prefs.load()
    reset = {f"default_{g}": "" for g in GENDERS if saved.get(f"default_{g}") == voice.slug}
    if reset:
        prefs.save(**reset)
    characters.forget_library_voice(voice.slug)
    return True, f"Voice ‘{voice.name}’ moved to the Recycle Bin."
