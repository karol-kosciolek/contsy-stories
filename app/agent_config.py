# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Agent page's settings and its API key.

The key is the only secret Contsy keeps. It is saved encrypted with Windows DPAPI (`CryptProtectData`, bound to the
Windows account: a copy of the file is useless on another PC or account; other programs of the same account can still
ask Windows to decrypt it) in `agent_credentials.json` of the state's config folder, which the web server never
serves. The environment variable ANTHROPIC_API_KEY, when set, wins over a saved key (the page says so). The key lives
in memory only inside the provider's client and is masked everywhere else; `redact` removes it (and anything shaped
like a Claude key) from text before that text is written to a log, a run file or the page.

The settings (model, thoroughness, limits, permissions) live in `agent_settings.json` next to it. The permission to use
files attached to a run as photos and voices ("uploads", the owner's consent) is never saved: it is asked for each run.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import re
import sys
import threading
import time
from ctypes import wintypes

from . import agent_provider as provider
from . import config
from .persistence import atomic_text

CREDENTIALS_FILE = config.STATE_CONFIG_DIR / "agent_credentials.json"
SETTINGS_FILE = config.STATE_CONFIG_DIR / "agent_settings.json"
ENV_KEY = "ANTHROPIC_API_KEY"
KEY_SHAPE = re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}")
_LOOSE_KEY = re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}")
_ENTROPY = b"contsy-agent-key-v1"
_LOCK = threading.RLock()
_CACHE: dict = {"stamp": None, "key": ""}

MODES = ("prepare", "produce")
# What the agent may do beyond reading and creating (the page's Permissions card). "write": use and change the
# owner's library and projects (every change is backed up first); "delete": delete (Recycle Bin); "uploads": use
# the files attached to a run as photos and voice samples (the owner's consent, never saved); "queue_all": manage the
# whole GPU queue, not only its own jobs; "settings": change Contsy's settings; "downloads": download missing models.
PERMISSIONS = ("write", "delete", "uploads", "queue_all", "settings", "downloads")
DEFAULT_PERMISSIONS = {"write": True, "delete": False, "uploads": False, "queue_all": False, "settings": False,
                       "downloads": False}
NEVER_SAVED = ("uploads",)
BUDGET_LIMITS = {"cost_usd": (0.1, 1000.0)}  # the cost is the run's one limit (steps, GPU time and hours are only shown)
DEFAULT_BUDGETS = {"cost_usd": 20.0}
DEFAULTS = {"provider": provider.DEFAULT_PROVIDER, "model": provider.DEFAULT_MODEL, "effort": provider.DEFAULT_EFFORT,
            "mode": "prepare", "send_images": True, "keep_awake": True, "budgets": dict(DEFAULT_BUDGETS),
            "permissions": dict(DEFAULT_PERMISSIONS)}


# --- Windows DPAPI (ctypes, no package) -------------------------------------------------------------------------

class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def available() -> bool:
    """Whether a key can be saved here (DPAPI exists on Windows only; elsewhere use the environment variable)."""
    return sys.platform == "win32"


def _blob(data: bytes):
    buffer = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer


def _free(blob: _Blob) -> None:
    kernel32 = ctypes.windll.kernel32
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree(ctypes.cast(blob.pbData, ctypes.c_void_p))


def _protect(data: bytes) -> bytes:
    source, keep = _blob(data)
    entropy, keep_entropy = _blob(_ENTROPY)
    out = _Blob()
    if not ctypes.windll.crypt32.CryptProtectData(ctypes.byref(source), "Contsy Agent key", ctypes.byref(entropy),
                                                  None, None, 0, ctypes.byref(out)):
        raise OSError("Windows could not encrypt the key.")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        _free(out)


def _unprotect(data: bytes) -> bytes:
    source, keep = _blob(data)
    entropy, keep_entropy = _blob(_ENTROPY)
    out = _Blob()
    if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(source), None, ctypes.byref(entropy), None, None, 0,
                                                    ctypes.byref(out)):
        raise OSError("Windows could not decrypt the saved key (it was saved by another Windows account or PC).")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        _free(out)


# --- the key ----------------------------------------------------------------------------------------------------

def check_key(key: str) -> str:
    """"" when `key` looks like a Claude API key, else why not."""
    key = (key or "").strip()
    if not key:
        return "Paste your Claude API key first."
    if not KEY_SHAPE.fullmatch(key):
        return "That does not look like a Claude API key. It starts with sk-ant- and has no spaces."
    return ""


def key_hint(key: str) -> str:
    """The masked form shown on the page: the start and the last four characters."""
    key = (key or "").strip()
    return f"sk-ant-…{key[-4:]}" if len(key) >= 12 else ""


def save_key(key: str) -> str:
    """Save `key`, encrypted for this Windows account. Returns "" or the reason it was refused."""
    key = (key or "").strip()
    problem = check_key(key)
    if problem:
        return problem
    if not available():
        return "This PC cannot encrypt the key. Set the ANTHROPIC_API_KEY environment variable instead."
    try:
        blob = base64.b64encode(_protect(key.encode("utf-8"))).decode("ascii")
        record = {"v": 1, "provider": provider.DEFAULT_PROVIDER, "protect": "dpapi-user", "blob": blob,
                  "hint": key_hint(key), "saved": time.time()}
        CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(CREDENTIALS_FILE, json.dumps(record, indent=1))
    except OSError as error:
        return f"The key could not be saved: {error}"
    with _LOCK:
        _CACHE.update(stamp=None, key="")
    return ""


def clear_key() -> None:
    """Forget the saved key: its file is overwritten, then removed."""
    with _LOCK:
        _CACHE.update(stamp=None, key="")
        try:
            size = CREDENTIALS_FILE.stat().st_size
            with CREDENTIALS_FILE.open("r+b") as stream:
                stream.write(b"\0" * size)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            pass
        try:
            CREDENTIALS_FILE.unlink()
        except OSError:
            pass


def _saved_key() -> str:
    try:
        stat = CREDENTIALS_FILE.stat()
    except OSError:
        return ""
    stamp = (stat.st_mtime_ns, stat.st_size)
    with _LOCK:
        if _CACHE["stamp"] == stamp:
            return _CACHE["key"]
        try:
            record = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
            key = _unprotect(base64.b64decode(record["blob"])).decode("utf-8")
        except (OSError, ValueError, KeyError, TypeError):
            key = ""
        _CACHE.update(stamp=stamp, key=key)
        return key


def get_key() -> tuple[str, str]:
    """(key, source): source is "environment" (ANTHROPIC_API_KEY wins), "saved" or "none" with an empty key."""
    env = (os.environ.get(ENV_KEY) or "").strip()
    if env:
        return env, "environment"
    key = _saved_key()
    return (key, "saved") if key else ("", "none")


def key_status() -> dict:
    """What the page shows about the key, never the key itself: {"source", "hint", "saved", "environment",
    "encrypt"}. `saved` says a key file exists even while the environment variable is the one in use."""
    key, source = get_key()
    saved = _saved_key()
    return {"source": source, "hint": key_hint(key), "saved": bool(saved), "environment": source == "environment",
            "saved_hint": key_hint(saved), "encrypt": available()}


def redact(text, *extra: str) -> str:
    """`text` without the API key (the exact current one, any `extra` secret, and anything shaped like a Claude key)."""
    text = "" if text is None else str(text)
    secrets = [secret for secret in (get_key()[0], *extra) if secret and len(secret) >= 8]
    for secret in sorted(set(secrets), key=len, reverse=True):
        text = text.replace(secret, "[key removed]")
    return _LOOSE_KEY.sub("[key removed]", text)


# --- settings ---------------------------------------------------------------------------------------------------

def _number(value, low, high, default, kind=float):
    try:
        number = kind(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return kind(min(max(number, low), high))


def clean_budgets(values) -> dict:
    values = values if isinstance(values, dict) else {}
    return {name: _number(values.get(name, DEFAULT_BUDGETS[name]), *BUDGET_LIMITS[name], DEFAULT_BUDGETS[name],
                          type(DEFAULT_BUDGETS[name])) for name in DEFAULT_BUDGETS}


def clean_permissions(values, *, saving: bool = False) -> dict:
    """The permissions as booleans; unknown names are dropped. With `saving`, the consent for attached files is
    left out (it is asked for each run)."""
    values = values if isinstance(values, dict) else {}
    result = {name: bool(values.get(name, DEFAULT_PERMISSIONS[name])) for name in PERMISSIONS}
    if saving:
        for name in NEVER_SAVED:
            result[name] = DEFAULT_PERMISSIONS[name]
    return result


def clean_settings(values, *, saving: bool = False) -> dict:
    """`values` as a complete, valid settings dict: every field is checked and a bad one falls back to its default."""
    values = values if isinstance(values, dict) else {}
    model = values.get("model")
    effort = values.get("effort")
    mode = values.get("mode")
    return {
        "provider": provider.DEFAULT_PROVIDER,
        "model": model if model in provider.MODELS else DEFAULTS["model"],
        "effort": effort if effort in provider.EFFORTS else DEFAULTS["effort"],
        "mode": mode if mode in MODES else DEFAULTS["mode"],
        "send_images": bool(values.get("send_images", DEFAULTS["send_images"])),
        "keep_awake": bool(values.get("keep_awake", DEFAULTS["keep_awake"])),
        "budgets": clean_budgets(values.get("budgets")),
        "permissions": clean_permissions(values.get("permissions"), saving=saving),
    }


def load_settings() -> dict:
    try:
        stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    return clean_settings(stored)


def save_settings(values) -> dict:
    """Save `values` (checked, without the per-run consent) and return what was saved."""
    clean = clean_settings(values, saving=True)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(SETTINGS_FILE, json.dumps(clean, indent=1))
    return clean


def allowed(settings: dict, permission: str) -> bool:
    """Whether `settings` (a clean settings dict) grants `permission`."""
    return bool((settings.get("permissions") or {}).get(permission))
