# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""What the Agent page checks before a run can start: the Claude key, the GPU queue, the models the chosen mode needs
(judged by file name and size, so nothing is hashed), and the free disk space. No network, nothing is changed.

A model the run needs but that is missing is a stop for the agent (a run never downloads a model unless the owner
gave it the permission to), so the page says so before Start.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import agent_store, config, model_manager, queue_manager

# What each mode uses. Preparing builds characters and places (their photos and voices), reads the brief's pictures
# and describes frames with the local assistant; producing also makes the clips, their sound and the lip sync.
PREPARE_MODELS = ("z-image-turbo", "qwen-image-edit-2511", "animation-styles", "voxcpm2", "prompt-llm", "face-check")
PRODUCE_MODELS = (*PREPARE_MODELS, "wan22-a14b-i2v", "wan22-a14b-t2v", "infinitetalk", "ezaudio-xl", "musetalk-15")
LOW_DISK_GB, TIGHT_DISK_GB = 20.0, 5.0


@dataclass(frozen=True)
class Check:
    name: str
    state: str      # ok | warn | err | info
    detail: str


def models_for(mode: str) -> tuple[str, ...]:
    return PRODUCE_MODELS if mode == "produce" else PREPARE_MODELS


def model_rows(mode: str) -> list[tuple[str, str, int]]:
    """(catalog entry id, its name, bytes still to download) of the models `mode` uses, for the current model set."""
    rows = []
    for model_id in models_for(mode):
        entry = model_manager.tiered(model_id)
        model = config.get_model(entry)
        if model is None:
            continue
        download, _ = model_manager.missing_bytes([entry])
        rows.append((entry, str(model.get("name") or entry), int(download)))
    return rows


def missing_models(mode: str) -> list[tuple[str, str, int]]:
    return [row for row in model_rows(mode) if row[2] > 0]


def _gb(size: float) -> str:
    return f"{size / 1e9:.1f} GB"


def checks(mode: str, *, key: dict, downloads_allowed: bool = False) -> list[Check]:
    """The page's preflight rows for `mode` ("prepare" or "produce")."""
    found: list[Check] = []
    if key.get("source") == "none":
        found.append(Check("Claude key", "err", "No key saved. Paste it above and click Save key."))
    elif key.get("source") == "environment":
        found.append(Check("Claude key", "ok", f"From the ANTHROPIC_API_KEY variable (ends {key.get('hint', '')[-4:]})."))
    else:
        found.append(Check("Claude key", "ok", f"Saved (ends {key.get('hint', '')[-4:]})."))

    try:
        jobs = queue_manager.snapshot()
        waiting = sum(1 for job in jobs if job.status == queue_manager.PENDING)
        running = sum(1 for job in jobs if job.status == queue_manager.RUNNING)
        if queue_manager.paused():
            found.append(Check("GPU queue", "warn", "Paused. Planning works; generation waits until you resume it."))
        elif running or waiting:
            found.append(Check("GPU queue", "info", f"{running} running, {waiting} waiting. The agent's jobs queue "
                                                    "behind them."))
        else:
            found.append(Check("GPU queue", "ok", "Empty."))
    except Exception:  # noqa: BLE001 - a check must never break the page
        found.append(Check("GPU queue", "warn", "The queue could not be read."))

    try:
        rows = model_rows(mode)
    except (OSError, KeyError, ValueError):
        rows = []
        found.append(Check("Models", "warn", "The installed models could not be checked."))
    absent = [row for row in rows if row[2] > 0]
    if rows and not absent:
        found.append(Check("Models", "ok", f"All {len(rows)} models this mode uses are installed."))
    elif absent:
        names = ", ".join(name for _, name, _ in absent)
        total = sum(size for _, _, size in absent)
        if downloads_allowed:
            found.append(Check("Models", "warn", f"Missing: {names} ({_gb(total)} to download; the agent may "
                                                 "download them)."))
        else:
            found.append(Check("Models", "err", f"Missing: {names} ({_gb(total)}). Install them in Settings, or allow "
                                                "downloads below; otherwise the agent stops when it needs one."))

    try:
        free = agent_store.free_gb()
    except OSError:
        free = -1.0
    if free < 0:
        found.append(Check("Disk space", "warn", "The free space could not be read."))
    elif free < TIGHT_DISK_GB:
        found.append(Check("Disk space", "err", f"Only {free:.1f} GB free. A run writes clips and documents."))
    elif free < LOW_DISK_GB:
        found.append(Check("Disk space", "warn", f"{free:.1f} GB free."))
    else:
        found.append(Check("Disk space", "ok", f"{free:.0f} GB free."))
    return found


def blocking_models(mode: str, downloads_allowed: bool) -> str:
    """"" or the reason Start must wait because models are missing (and may not be downloaded). Only producing waits:
    preparing may still do its text work, and the preflight lists what it lacks."""
    if mode != "produce" or downloads_allowed:
        return ""
    try:
        absent = missing_models(mode)
    except (OSError, KeyError, ValueError):
        return ""
    if not absent:
        return ""
    return ("Some models this mode needs are not installed: " + ", ".join(name for _, name, _ in absent)
            + ". Install them in Settings, allow downloads, or use Prepare.")
