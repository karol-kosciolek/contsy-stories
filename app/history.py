# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Historia generacji - zapis promptow, seedow, parametrow i plikow wynikowych.

Przechowujemy jako JSON Lines (history/history.jsonl) - dopisywanie jest tanie,
a odczyt ostatnich wpisow prosty.
"""
from __future__ import annotations

import functools
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .persistence import atomic_text

_HISTORY_FILE = config.HISTORY_DIR / "history.jsonl"
_CTX = threading.local()
_LOCK = threading.RLock()


def _synchronized(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _LOCK:
            return fn(*args, **kwargs)
    return wrapped


def job_context(params: Optional[Dict]) -> None:
    """Attach the queued job's cast to history entries written by generators."""
    _CTX.extra = {}
    if params and params.get("character"):
        _CTX.extra["profile"] = params["character"]
    if params and isinstance(params.get("characters"), list) and len(params["characters"]) > 1:
        _CTX.extra["cast"] = list(params["characters"])  # every character in the job (slugs)


@_synchronized
def record(entry: Dict) -> Dict:
    """Dopisuje wpis do historii. Uzupelnia znacznik czasu, jesli brak."""
    entry = {**getattr(_CTX, "extra", {}), **entry}
    entry.setdefault("ts", time.time())
    entry.setdefault("time_str", time.strftime("%Y-%m-%d %H:%M:%S",
                                               time.localtime(entry["ts"])))
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(_HISTORY_FILE, "a", encoding="utf-8") as f:
        # default=str: a value JSON cannot hold (a Path, a date) is written as text, so it never fails a finished job
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry


# history file -> (its stamp, entries newest first): the file is parsed once per change, not on every page refresh
_PARSED: Dict[str, tuple] = {}


def _entries() -> List[Dict]:
    try:
        stat = _HISTORY_FILE.stat()
    except OSError:
        return []
    key = str(_HISTORY_FILE)
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _PARSED.get(key)
    if cached and cached[0] == stamp:
        return cached[1]
    out: List[Dict] = []
    for line in reversed(_HISTORY_FILE.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            out.append(entry)
    if len(_PARSED) > 256:  # one entry per history file read (tests point _HISTORY_FILE elsewhere); kept bounded
        _PARSED.clear()
    _PARSED[key] = (stamp, out)
    return out


@_synchronized
def recent(limit: int = 50) -> List[Dict]:
    """Zwraca ostatnie wpisy (najnowsze pierwsze). The file is parsed again only when it changed."""
    return [dict(entry) for entry in _entries()[:max(limit, 0)]]


def _same_file(left: str, right: str) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if Path(right).is_absolute():
        return str(Path(left).resolve()).casefold() == str(Path(right).resolve()).casefold()
    return Path(left).name == Path(right).name


def entry_for(path: str, limit: int = 1000) -> Optional[Dict]:
    """Najnowszy wpis historii z tym plikiem (po nazwie) albo None."""
    return next((e for e in recent(limit)
                 if any(_same_file(f, path) for f in e.get("files") or [])), None)


@_synchronized
def forget(paths: List[str]) -> None:
    """Usuwa z historii wpisy usunietych plikow (Galeria -> Usun)."""
    if not _HISTORY_FILE.exists() or not paths:
        return
    out = []
    for line in _HISTORY_FILE.read_text(encoding="utf-8").splitlines(keepends=True):
        try:
            entry = json.loads(line)
        except (ValueError, TypeError):
            out.append(line)
            continue
        if not isinstance(entry, dict):
            out.append(line)
            continue
        files = entry.get("files") or []
        remaining = [f for f in files if not any(_same_file(f, path) for path in paths)]
        if len(remaining) == len(files):
            out.append(line)
        elif remaining:
            out.append(json.dumps({**entry, "files": remaining}, ensure_ascii=False) + "\n")
    atomic_text(_HISTORY_FILE, "".join(out))
