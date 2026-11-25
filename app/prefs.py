# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Ustawienia uzytkownika (autozapis) i styl obrazu.

Do promptu dopisujemy styl obrazu: na stronach Image i Video zawsze realistyczny (STYLE), w Timeline styl filmu
(app/styles.py: Realistic, 3D animation, 2D cartoon, Anime or a style of the owner's own). Autozapis zapisywany w config/user_prefs.json.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List

from . import config, styles
from .persistence import atomic_text

_FILE = config.STATE_CONFIG_DIR / "user_prefs.json"
_LOCK = threading.RLock()

# The realistic style's texts: the Image and Video pages, and every film saved before styles existed
STYLE = {"video": styles.get(styles.REALISTIC).video, "image": styles.get(styles.REALISTIC).image}
IMAGE_STEPS = 8   # Z-Image-Turbo uczony na 8 krokach

# character: slug postaci; qc: kontrola jakosci wynikow postaci (app/qc.py);
# tier: model set "full" or "lite" ("" = chosen from the GPU, see model_manager.tier);
# default_female / default_male: voice slugs for spoken lines that name no voice (app/voices.py);
# universe: the group of characters selected in the top bar's Tools panel (app/universes.py, app/cast.py) -
# then "character" is its main character
_DEFAULTS = {"autosave": False, "folder": "", "character": "", "qc": True,
             "tier": "", "default_female": "", "default_male": "", "universe": ""}


def load() -> Dict:
    data = {}
    for attempt in range(3):  # Windows may refuse the read while another save is replacing the file
        try:
            data = json.loads(_FILE.read_text(encoding="utf-8"))
            break
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
        except (OSError, ValueError):
            break
    return {**_DEFAULTS, **{k: v for k, v in (data if isinstance(data, dict) else {}).items() if k in _DEFAULTS}}


def save(**changes) -> Dict:
    with _LOCK:
        data = {**load(), **{k: v for k, v in changes.items() if k in _DEFAULTS}}
        atomic_text(_FILE, json.dumps(data, ensure_ascii=False, indent=2))
        return data


def styled(prompt: str, kind: str, style: str = styles.REALISTIC) -> str:
    """Prompt z dopisanym stylem (kind: 'video' / 'image'; style: id z app/styles.py, domyslnie realistyczny)."""
    suffix = getattr(styles.get(style), kind)
    return f"{prompt.strip().rstrip('.')}, {suffix}" if prompt and prompt.strip() else prompt


def autosave(files: List[str]) -> str:
    """Kopiuje gotowe pliki do folderu autozapisu (gdy wlaczony). Zwraca opis albo ''."""
    p = load()
    if not p["autosave"] or not p["folder"] or not files:
        return ""
    folder = Path(p["folder"])
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            target = folder / Path(f).name
            if target.resolve() == Path(f).resolve():
                continue
            number = 1
            while target.exists():
                number += 1
                target = folder / f"{Path(f).stem}_{number}{Path(f).suffix}"
            shutil.copy2(f, target)
    except OSError as e:
        return f"Autosave failed: {e}"
    return f"Saved to folder: {folder}"


def choose_folder(current: str = "") -> str:
    """Systemowe okno wyboru folderu (aplikacja dziala lokalnie, wiec okno pojawia sie
    na ekranie uzytkownika). Zwraca wybrany folder albo dotychczasowy."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(initialdir=current or str(Path.home()),
                                         title="Autosave folder")
        root.destroy()
    except Exception:  # noqa: BLE001 - brak okna = zostaje stary folder
        chosen = ""
    return chosen or current
