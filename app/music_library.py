# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The owner's music library: licensed tracks the episode mix lays under the narration (engine/media/episode_mix.py).

The files live in the state folder's `music/` (config.STATE_ROOT), put there by the owner. Nothing is
downloaded or generated: an empty library means an episode without music, and the QC report says so. A project always
gets the same track (picked by its id), so exporting again does not change the music.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

from . import config

EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus")


def folder() -> Path:
    """The music folder, looked up on every call so a reassigned config.STATE_ROOT (tests) is followed."""
    return config.STATE_ROOT / "music"


def tracks() -> List[Path]:
    """The library's tracks, by name."""
    try:
        return sorted(path for path in folder().iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS)
    except OSError:
        return []


def pick(key: str) -> Optional[Path]:
    """The track of a project (or a series): the same one every time for the same key; None for an empty library."""
    found = tracks()
    if not found:
        return None
    index = int(hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:8], 16) % len(found)
    return found[index]
