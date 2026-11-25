# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Characters as studio profiles: what belongs to which - queue jobs and results from the history -
and the numbers on the Dashboard.

Profil = postac wybrana w panelu Tools na gornym pasku ("Character or universe"). Wtedy GPU Queue, Gallery i Dashboard pokazuja tylko ja;
"Wszystkie (bez postaci)" pokazuje wszystko. Kolejka fizycznie jest jedna (jedna karta), wiec
"kolejka modelki" to widok na jej zadania. Wyniki zostaja w outputs/ - do kogo naleza, mowi
historia: nowe wpisy maja "profile" (slug), starsze "character" (nazwa postaci).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Union

from . import characters, config, history, queue_manager

MEDIA = (".png", ".jpg", ".jpeg", ".webp", ".mp4")


Profiles = Union[str, Sequence[str]]


def current() -> str:
    """Profil wybrany w panelu Tools na gornym pasku (slug) albo "" - wszystkie."""
    ch = characters.selected()
    return ch["slug"] if ch else ""


def current_set() -> List[str]:
    """Characters whose results and jobs the pages show now: the selected character, or all
    members of the selected universe ([] = everything)."""
    from . import cast

    return cast.profiles()


def _wanted(profile: Profiles) -> set:
    if not profile:
        return set()
    return {profile} if isinstance(profile, str) else {p for p in profile if p}


def name(profile: str) -> str:
    ch = characters.get(profile) if profile else None
    return ch["name"] if ch else ""


def jobs(profile: Profiles, snapshot: Optional[List[queue_manager.Job]] = None
         ) -> List[queue_manager.Job]:
    """Zadania profilu (wszystkie, gdy profile == ""); profile may be several slugs. A job
    with several characters belongs to each of them."""
    snap = queue_manager.snapshot() if snapshot is None else snapshot
    wanted = _wanted(profile)
    return [j for j in snap if not wanted or job_owner(j) in wanted
            or wanted & set((j.params or {}).get("characters") or [])]


def job_owner(job: queue_manager.Job) -> str:
    """The character a job belongs to: its main character ("" for a job without one)."""
    params = job.params if isinstance(job.params, dict) else {}
    return params.get("character") or ""


def _owner(entry: Dict, by_name: Dict[str, str]) -> str:
    """The profile of a history entry: "profile" (newer entries) or the character's name (videos). Entries
    of the removed character training, and older images that named only the LoRA of a character, belong to nobody."""
    if entry.get("profile"):
        return entry["profile"]
    return by_name.get(entry.get("character") or "", "")


def file_owners(entries: Optional[List[Dict]] = None) -> Dict[str, str]:
    """Nazwa pliku wyniku -> profil (slug) wg historii; wynik bez postaci -> ""."""
    by_name = {c["name"]: c["slug"] for c in characters.all_characters()}
    owners: Dict[str, str] = {}
    for e in reversed(entries if entries is not None else history.recent(5000)):
        slug = _owner(e, by_name)
        for f in e.get("files") or []:
            owners[Path(f).name] = slug
    return owners


def file_members(entries: Optional[List[Dict]] = None) -> Dict[str, Set[str]]:
    """Nazwa pliku wyniku -> every character in it (slugs): the owner plus the characters of a
    video or photo with several of them, so it shows in each of their galleries."""
    by_name = {c["name"]: c["slug"] for c in characters.all_characters()}
    members: Dict[str, Set[str]] = {}
    for e in reversed(entries if entries is not None else history.recent(5000)):
        slugs = {_owner(e, by_name)} | set(e.get("cast") or []) | \
            {by_name[n] for n in (e.get("characters") or []) if n in by_name}
        slugs.discard("")
        for f in e.get("files") or []:
            members[Path(f).name] = slugs
    return members


def _media_key(path: str) -> str:
    return str(Path(path).resolve()).casefold()


def recorded_files(entries: Optional[List[Dict]] = None) -> set[str]:
    """Every result file the history lists, as comparison keys. The pages show these and nothing
    else from outputs/: a job's work files lie there too while it runs."""
    recorded = set()
    for entry in history.recent(5000) if entries is None else entries:
        files = entry.get("files") if isinstance(entry, dict) else None
        if isinstance(files, list):
            recorded.update(_media_key(path) for path in files if isinstance(path, str))
    return recorded


def is_recorded(path: str, entries: Optional[List[Dict]] = None) -> bool:
    """Whether the history lists this file as a result."""
    return _media_key(path) in recorded_files(entries)


def results(profile: Profiles = "", exts=MEDIA, limit: Optional[int] = None,
            owners: Optional[Dict[str, str]] = None) -> List[str]:
    """Wyniki z outputs/ (najnowsze pierwsze), przy profilu - tylko jego (profile may be
    several slugs; a result with several characters shows for each of them)."""
    d = config.OUTPUTS_DIR
    if not d.exists():
        return []
    recorded = recorded_files()
    files = [p for p in d.iterdir() if p.is_file() and _media_key(str(p)) in recorded and p.suffix.lower() in exts
             and not p.stem.endswith("_raw") and "_part" not in p.stem]
    wanted = _wanted(profile)
    if wanted and owners is not None:
        files = [p for p in files if owners.get(p.name) in wanted]
    elif wanted:
        members = file_members()
        files = [p for p in files if members.get(p.name, set()) & wanted]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files[:limit]]


def overview(snapshot: Optional[List[queue_manager.Job]] = None,
             entries: Optional[List[Dict]] = None, now: Optional[float] = None) -> List[Dict]:
    """Wiersz na kazda postac (plus "bez postaci"): stan, zadania w kolejce, wyniki dzis
    i razem, ostatni wynik, uwagi. Liczy tylko istniejace pliki."""
    snap = queue_manager.snapshot() if snapshot is None else snapshot
    entries = history.recent(5000) if entries is None else entries
    owners = file_owners(entries)
    midnight = time.mktime(time.localtime(now or time.time())[:3] + (0, 0, 0, 0, 0, -1))
    seen: Dict[str, Dict] = {}
    for e in entries:
        for f in e.get("files") or []:
            name_ = Path(f).name
            if name_ in seen or not Path(f).exists():
                continue
            seen[name_] = {"owner": owners.get(name_, ""), "ts": float(e.get("ts") or 0)}

    def row(slug: str, ch: Optional[dict]) -> Dict:
        mine = [v for v in seen.values() if v["owner"] == slug]
        queued = [j for j in snap if j.status in (queue_manager.PENDING, queue_manager.RUNNING)
                  and job_owner(j) == slug]
        preview = characters.preview_path(ch)
        return {"slug": slug, "name": ch["name"] if ch else "No character",
                "status": characters.state(ch) if ch else "", "usable": bool(slug and
                                                                             characters.usable(slug)),
                "preview": str(preview) if preview else "",
                "queued": len(queued), "running": any(j.status == queue_manager.RUNNING
                                                      for j in queued),
                "eta": sum((j.eta() or 0.0) if j.status == queue_manager.RUNNING
                           else j.estimate() for j in queued),
                "today": sum(1 for v in mine if v["ts"] >= midnight), "total": len(mine),
                "last": max((v["ts"] for v in mine), default=0.0), "note": ""}

    rows = [row(c["slug"], c) for c in characters.all_characters()]
    return rows + [row("", None)]
