# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Stored characters for tests: the folder app/characters.py saves (character.json, photos/, voice/)."""
import hashlib
import io
import json
from pathlib import Path
from typing import Optional

from PIL import Image


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def store_character(library: Path, data: dict, photos: int = 1, voice: Optional[bytes] = None) -> dict:
    """Write a character as Save does: `data` gives name, slug, gender, look (and created, a style and a kind such as
    "creature", stored only when given); `photos` reference photos (small PNGs, each a different colour) and, with
    `voice`, the bytes of its own voice sample."""
    folder = Path(library) / data["slug"]
    (folder / "photos").mkdir(parents=True, exist_ok=True)
    entries = []
    for number in range(1, photos + 1):
        path = folder / "photos" / f"{number:02d}.png"
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (number * 20 % 256, len(data["slug"]) * 9 % 256, 70)).save(buffer, "PNG")
        path.write_bytes(buffer.getvalue())
        entries.append({"file": path.name, "sha256": _sha(path), "origin": "generated"})
    record = {"format": 2, "gender": "woman", "look": "", "created": 0, **{k: v for k, v in data.items()
                                                                           if k in ("name", "slug", "gender", "look",
                                                                                    "created", "style", "kind")},
              "photos": entries, "voice": None}
    if voice is not None:
        (folder / "voice").mkdir(exist_ok=True)
        sample = folder / "voice" / "sample.wav"
        sample.write_bytes(voice)
        record["voice"] = {"file": "sample.wav", "sha256": _sha(sample), "origin": "designed", "description": ""}
    (folder / "character.json").write_text(json.dumps(record), encoding="utf-8")
    return record
