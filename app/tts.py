# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Speech synthesis (narration and character lines) and voice design with VoxCPM2 at 48 kHz.

A fixed sample supplies each voice: a voice of the library or a character's own voice. The engine loads
VoxCPM2 once per batch of lines and can isolate its CUDA context on smaller cards. Voice design makes a new
sample from a text description (engine.pipelines.speech.design_voice); the character creator and
production scripts use it.
"""
from __future__ import annotations

import re
from pathlib import Path
import random
from typing import Callable, List, Optional, Tuple

from . import config, engine_runtime, model_manager, voices

MODEL_ID = "voxcpm2"

# Voice names from older versions ("Woman 1 — higher voice", "Mężczyzna 2 — średni głos")
# resolve to the default voice of that gender.
_LEGACY_MALE = ("man", "mężczyzna", "mezczyzna")


def available() -> bool:
    return model_manager.is_installed(MODEL_ID)


def resolve(voice: Optional[str]) -> Optional[voices.Voice]:
    """Voice for a slug or name. Unknown and legacy names fall back to the default voice of the
    gender (legacy men's names) or the default woman's voice; None when there are no voices."""
    pool = voices.all_voices()
    found = voices.get(voice, pool)
    if found:
        return found
    key = (voice or "").strip().casefold()
    gender = "male" if key.startswith(_LEGACY_MALE) else "female"
    chosen = voices.defaults(pool)
    return voices.get(chosen.get(gender) or chosen.get("female"), pool) or (pool[0] if pool else None)


def _speakable(text: str) -> str:
    """VoxCPM2 reads a leading "(…)" as a style instruction, so a line that starts with a
    parenthesis loses its parentheses."""
    text = text.strip()
    if text.startswith(("(", "（")):
        text = re.sub(r"[()（）]", "", text).strip()
    return text


def synthesize(text: str, voice: Optional[str], out_path: str | Path,
               progress_cb: Optional[Callable[[float, str], None]] = None) -> Tuple[bool, str]:
    """Generuje plik WAV (48 kHz) z jednym tekstem. Zwraca (ok, sciezka_lub_komunikat)."""
    if not text or not text.strip():
        return False, "Enter some text."
    return synthesize_many([(text, voice, str(out_path))], progress_cb)


def synthesize_many(items: List[Tuple[str, Optional[str], str]],
                    progress_cb: Optional[Callable[[float, str], None]] = None
                    ) -> Tuple[bool, str]:
    """Kilka kwestii (tekst, glos, plik) w jednym procesie - model laduje sie raz."""
    if not items or any(not str(text or "").strip() for text, _, _ in items):
        return False, "Enter some text for each spoken line."
    if not available():
        return False, "Voice skipped: install the VoxCPM2 model (Video + Audio package)."
    chosen = [resolve(voice) for _, voice, _ in items]
    if any(v is None for v in chosen):
        return False, ("No voice samples were found. Reinstall the app or add a voice on the "
                       "Voices page.")
    from engine.pipelines.speech import SpeechLine

    lines = [SpeechLine(text=_speakable(text), reference=v.path, output=str(Path(out).resolve()))
             for (text, _, out), v in zip(items, chosen)]
    try:
        result = engine_runtime.engine().tts(config.destination_dir(MODEL_ID), lines,
                                            isolated=engine_runtime.small_gpu(), progress_cb=progress_cb)
    except Exception as exc:  # noqa: BLE001
        return False, f"Speech synthesis failed: {exc}"
    return True, str(result)


def design_voice(description: str, out_path: str | Path, seed: Optional[int] = None,
                 progress_cb: Optional[Callable[[float, str], None]] = None,
                 text: Optional[str] = None) -> Tuple[bool, str]:
    """A new voice sample (WAV) for a text description, e.g. "a warm, low, calm voice". Returns (ok, path or
    message). The same description and seed give the same sample. Only adult voices are made: a description that
    names a minor is refused here, for every caller (the character creator, the production scripts)."""
    from . import llm

    description = " ".join((description or "").split())
    if not description:
        return False, "Describe the voice first."
    if llm.blocked(description) or llm.refers_to_minor(description):
        return False, "⛔ Only adult voices are supported. This description appears to refer to a minor."
    if not available():
        return False, "Install the VoxCPM2 model (Video + Audio package) to design voices."
    seed = random.randint(0, 2 ** 31 - 1) if seed is None else int(seed)
    try:
        path = engine_runtime.engine().design_voice(config.destination_dir(MODEL_ID), description,
                                                    Path(out_path).resolve(), seed=seed, text=text,
                                                    isolated=engine_runtime.small_gpu(), progress_cb=progress_cb)
    except ValueError as exc:  # no usable take, an empty description
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001 - the message goes to the job
        return False, f"Voice design failed: {exc}"
    return True, str(path)
