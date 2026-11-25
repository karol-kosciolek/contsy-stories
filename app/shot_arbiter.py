# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""A second opinion from Claude for shots the local vision model was unsure about (docs/QUALITY_ARCHITECTURE.md,
layer 4). Off by default: the owner turns it on in config ("qc": {"arbiter": true}) and it uses the owner's own API key
(agent_config). Only the unsure questions and the same contact sheet are sent. Every call's tokens and price are
added to `spent` (and to the clip's QC record by the caller), so the cost is measured, not guessed.

Model: Claude Sonnet 5.5 (the owner's choice for this check: a few cents a shot); config "qc": {"arbiter_model"}
can name another. Structured output gives the answers as JSON; a refusal or an error leaves the questions unsure.
"""
from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Dict, Sequence, Tuple

from . import config

MODEL = "claude-sonnet-5-5"
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-5-5": (0.10, 0.50)}
_LOCK = threading.Lock()
spent = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}

_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answers"], "properties": {
    "answers": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                           "required": ["id", "answer", "reason"],
                                           "properties": {"id": {"type": "string"},
                                                          "answer": {"type": "string", "enum": ["A", "B"]},
                                                          "reason": {"type": "string"}}}}}}


def _model() -> str:
    found = config.SETTINGS.get("qc") if isinstance(config.SETTINGS.get("qc"), dict) else {}
    return str(found.get("arbiter_model") or MODEL)


def available() -> bool:
    try:
        from . import agent_config

        return bool(agent_config.get_key()[0])
    except Exception:  # noqa: BLE001 - no key, no arbiter
        return False


def ask(sheet: Path, questions: Sequence, local: Dict[str, Tuple[str, str]], *,
        extra: Sequence[Path] = ()) -> Dict[str, Tuple[str, str]]:
    """Claude's yes/no answers to the questions the local model was unsure about: {id: (answer, reason)}. `extra`:
    the character sheet (picture 2) the identity questions refer to."""
    from . import agent_claude, agent_config

    key = agent_config.get_key()[0]
    if not key or not questions:
        return {}
    client = agent_claude.make_client(key)
    model = _model()
    listing = "\n".join(f"- {one.id}: {one.text} A = {one.good}. B = {one.bad}." for one in questions)
    pictures = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")}}
                for path in [sheet, *extra]]
    response = client.messages.create(
        model=model, max_tokens=4000, output_config={"effort": "medium", "format": {"type": "json_schema",
                                                                                     "schema": _SCHEMA}},
        system="You inspect frames of a children's animated series for production defects. The first row of the "
               "first picture holds labelled reference pictures; the rest are frames of one shot in time order and "
               "enlarged zoom crops. A second picture, when there is one, shows each character's reference beside crops "
               "of it from the shot. The yellow captions are the inspector's, never text in the shot. Answer strictly from what is visible.",
        messages=[{"role": "user", "content": [
            *pictures,
            {"type": "text", "text": "Answer each question with A or B (the one that describes what you see) and a short reason naming "
                                     "the frame time:\n"
                                     + listing}]}])
    usage = response.usage
    price_in, price_out = PRICES.get(model, PRICES[MODEL])
    with _LOCK:
        spent["calls"] += 1
        spent["input_tokens"] += usage.input_tokens
        spent["output_tokens"] += usage.output_tokens
        spent["usd"] += usage.input_tokens / 1e6 * price_in + usage.output_tokens / 1e6 * price_out
    if response.stop_reason == "refusal":
        return {}
    text = next((block.text for block in response.content if block.type == "text"), "")
    try:
        rows = json.loads(text)["answers"]
    except (ValueError, KeyError, TypeError):
        return {}
    asked = {one.id for one in questions}
    return {row["id"]: (row["answer"], f"(Claude) {row.get('reason', '')}"[:200])
            for row in rows if isinstance(row, dict) and row.get("id") in asked}
