# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Pinned, verified auxiliary model used by face tracking and lip sync: the MediaPipe Face Landmarker."""
from __future__ import annotations

import json
from pathlib import Path

from .downloader import download_url

ROOT = Path(__file__).resolve().parent.parent
CATALOG = json.loads((ROOT / "config/auxiliary-models.json").read_text(encoding="utf-8"))


def ensure_landmarker() -> Path:
    """The pinned Face Landmarker, downloaded on first use (resumed, size and SHA-256 checked, see
    `downloader.download_url`)."""
    spec = CATALOG["landmarker"]
    return download_url(spec["url"], ROOT / "models" / "mediapipe", "face_landmarker.task",
                        expected_size=spec["size_bytes"], sha256=spec["sha256"])
