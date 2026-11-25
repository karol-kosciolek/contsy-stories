# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Konfiguracja i sciezki projektu.

Laduje config/settings.json (z opcjonalnym nadpisaniem przez settings.local.json)
oraz katalog modeli config/models.json. Wszystkie sciezki licza sie od katalogu
glownego projektu (rodzic folderu app/).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

# Katalog glowny = rodzic folderu app/
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    """Rekurencyjnie nadpisuje slownik base wartosciami z override."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _override_from_env():
    """The override file the app starts with. CONTSY_SETTINGS_LOCAL: another file (a second instance on its own port
    and state folder), or "none" for settings.json alone (the test suite, so the owner's settings.local.json does
    not change what it tests); unset: config/settings.local.json."""
    chosen = os.environ.get("CONTSY_SETTINGS_LOCAL", "").strip()
    return None if chosen.lower() == "none" else Path(chosen) if chosen else "default"


def load_settings(local="default") -> Dict[str, Any]:
    """Wczytuje settings.json i (jesli jest) settings.local.json jako nadpisanie (`local`: another override file, or
    None for none)."""
    with open(CONFIG_DIR / "settings.json", encoding="utf-8") as f:
        settings = json.load(f)
    if local == "default":
        local = CONFIG_DIR / "settings.local.json"
    if local is not None and Path(local).exists():
        with open(local, encoding="utf-8") as f:
            settings = _deep_merge(settings, json.load(f))
    state_dir = settings["paths"].setdefault("state_dir", ".")
    if not isinstance(state_dir, str) or not state_dir.strip():
        raise ValueError("paths.state_dir must be a nonempty local path.")
    own = settings.setdefault("engine", {})
    if not isinstance(own, dict):
        raise ValueError("engine must be an object.")
    cfg = settings["app"]
    if cfg.get("host") not in ("127.0.0.1", "localhost"):
        raise ValueError("app.host must be 127.0.0.1 or localhost. Contsy is a local-only application.")
    port = cfg.get("port")
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("app.port must be an integer between 1024 and 65535.")
    if own.setdefault("lora_rounding", "nearest") not in ("nearest", "stochastic"):
        raise ValueError("engine.lora_rounding must be 'nearest' or 'stochastic'.")
    # "auto": models stay warm in RAM only on machines with room for them (app/engine_runtime.warm_cache_gib)
    cache_gib = own.setdefault("warm_cache_gib", "auto")
    if cache_gib != "auto" and (type(cache_gib) is not int or cache_gib < 0):
        raise ValueError('engine.warm_cache_gib must be "auto" or a non-negative integer.')
    vram_limit = own.setdefault("vram_limit_gib", None)
    if vram_limit is not None and (type(vram_limit) is not int or vram_limit < 1):
        raise ValueError("engine.vram_limit_gib must be null or a positive integer.")
    # "auto": the video experts use SageAttention (and a fused rotary embedding) where Triton and the GPU allow it;
    # "sdpa": PyTorch's attention, exactly as before
    if own.setdefault("attention", "auto") not in ("auto", "sdpa"):
        raise ValueError("engine.attention must be 'auto' or 'sdpa'.")
    # Whether the installed FP8 image-to-video experts replace the GGUF experts and their LoRAs: "matching" each one
    # that a check finds to be its GGUF expert with that LoRA merged in, "off" none
    # (app/engine_runtime.fast_i2v_experts)
    if own.setdefault("fp8_experts", "matching") not in ("matching", "off"):
        raise ValueError("engine.fp8_experts must be 'matching' or 'off'.")
    # Where users get the source code of this exact version, shown in Settings (About).
    source_url = settings["app"].setdefault("source_url", "")
    if not isinstance(source_url, str) or (
            source_url and not re.fullmatch(r"https?://[^\s'\"<>]+", source_url)):
        raise ValueError("app.source_url must be empty or the http(s) address of this version's source code.")
    return settings


def load_model_catalog(model_dir: str = "models") -> Dict[str, Any]:
    """Wczytuje katalog modeli (config/models.json)."""
    with open(CONFIG_DIR / "models.json", encoding="utf-8") as f:
        catalog = json.load(f)
    catalog["destinations"] = {key: str(Path(model_dir) / relative)
                               for key, relative in catalog["destinations"].items()}
    return catalog


SETTINGS: Dict[str, Any] = load_settings(_override_from_env())
CATALOG: Dict[str, Any] = load_model_catalog(SETTINGS["paths"]["models_dir"])


def abs_path(rel: str) -> Path:
    """Zamienia sciezke wzgledna z konfiguracji na absolutna w katalogu projektu."""
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)


# Czesto uzywane sciezki
MODELS_DIR = abs_path(SETTINGS["paths"]["models_dir"])
STATE_ROOT = abs_path(SETTINGS["paths"]["state_dir"]).resolve()
STATE_CONFIG_DIR = STATE_ROOT / "config"
OUTPUTS_DIR = STATE_ROOT / SETTINGS["paths"]["outputs_dir"]
HISTORY_DIR = STATE_ROOT / SETTINGS["paths"]["history_dir"]


def destination_dir(dest_key: str) -> Path:
    """Zwraca absolutny katalog docelowy dla wpisu 'dest' z katalogu modeli."""
    rel = CATALOG["destinations"].get(dest_key, dest_key)
    return abs_path(rel)


def get_model(model_id: str) -> dict | None:
    for m in CATALOG["models"]:
        if m["id"] == model_id:
            return m
    return None


def ensure_dirs() -> None:
    """Tworzy podstawowe foldery robocze, jesli ich nie ma."""
    for d in (MODELS_DIR, STATE_CONFIG_DIR, OUTPUTS_DIR, HISTORY_DIR):
        d.mkdir(parents=True, exist_ok=True)
