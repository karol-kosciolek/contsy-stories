# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Menedzer modeli: stan instalacji i pobieranie modeli z katalogu.

Katalog modeli pochodzi z config/models.json. Model moze byc paczka kilku plikow
(np. model + enkoder + VAE), a plik moze pochodzic z innego repo niz reszta.
Stan "zainstalowany" = wszystkie pliki modelu leza w katalogach docelowych.

A catalog file is downloaded as it is, unless it has a "convert" step: then the installer downloads the official
files that step lists ("sources", each into its own "dest" or else into the folder "source_dest"), runs the step on
"input" (the file name of one source, or a list of them for a step that reads several) and publishes the result only
when its size and SHA-256 equal the catalog's. A step runs only while its result is missing, and the sources that
are already on disk (verified by size and SHA-256) are not downloaded again. "equivalents" names files of
earlier installations that hold the same weights (in the file's own folder, or in the folder "dest" names); one of
them on disk counts as the file being installed. One with the file's own SHA-256 holds the very same bytes, so it is
hard-linked into place under the catalog name (and used where it is when the link is impossible).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import threading
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from . import config, prefs, vram
from .downloader import (BLOCKED_PREFIX, DownloadError, download_file, download_url, downloads_blocked, get_hf_token,
                         _safe_name)
from .persistence import atomic_text

ProgressCb = Callable[[float, str], None]
_VERIFIED = {}
_VERIFY_LOCK = threading.Lock()
# Big files (VERIFY_CACHE_BYTES and more) whose SHA-256 matched the catalog, kept across restarts: by resolved path,
# the file's size, modification and change (Windows: creation) times, file index and the expected SHA-256. A file
# whose size, times or index differ from the record is hashed again, so a replaced or edited file is never trusted.
# Hashing the models of one scene took about 60 s after every start of the app (2026-11-15).
VERIFIED_FILE = config.STATE_CONFIG_DIR / "verified_models.json"
VERIFY_CACHE_BYTES = 64 * 1024 * 1024
_STORED: Optional[Dict[str, dict]] = None  # VERIFIED_FILE as read once by this process (under _VERIFY_LOCK)
# The install steps a catalog "convert" block may name: (module, function). Each takes (*inputs, output, *,
# expected_size, expected_sha256, progress_cb, cancel_flag), its inputs in the order of the block's "input", and
# publishes `output` only when both match.
CONVERTERS: Dict[str, Tuple[str, str]] = {
    "engine.models.zimage.checkpoint.merge": ("engine.models.zimage.checkpoint", "merge"),
    "engine.models.qwen_edit.convert.quantize_text_encoder": ("engine.models.qwen_edit.convert",
                                                              "quantize_text_encoder"),
    "app.downloader.extract_member": ("app.downloader", "extract_member"),
    "engine.models.wan22.convert.merge_lora_to_fp8": ("engine.models.wan22.convert", "merge_lora_to_fp8"),
    "engine.models.infinitetalk.clip.extract_visual": ("engine.models.infinitetalk.clip", "extract_visual"),
    "engine.models.infinitetalk.convert.adapter_fp8": ("engine.models.infinitetalk.convert", "adapter_fp8"),
}
# Extra arguments a step takes from its "convert" block.
_CONVERTER_OPTIONS = ("member",)


def _stored() -> Dict[str, dict]:
    """VERIFIED_FILE's records (read once per process; an unreadable or foreign file counts as empty). Call under
    _VERIFY_LOCK."""
    global _STORED
    if _STORED is None:
        try:
            data = json.loads(VERIFIED_FILE.read_text(encoding="utf-8"))
            files = data.get("files") if isinstance(data, dict) and data.get("version") == 1 else None
            _STORED = {path: record for path, record in (files or {}).items()
                       if isinstance(path, str) and isinstance(record, dict)}
        except (OSError, ValueError):
            _STORED = {}
    return _STORED


def _identity(stat: os.stat_result, expected: str) -> dict:
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
            "ino": stat.st_ino, "sha256": expected.lower()}


def _remember(path: str, record: dict) -> None:
    """Keep a verified file's identity in VERIFIED_FILE. Call under _VERIFY_LOCK. A file that cannot be written only
    means the file is hashed again after the next start."""
    stored = _stored()
    if stored.get(path) == record:
        return
    stored[path] = record
    try:
        atomic_text(VERIFIED_FILE, json.dumps({"version": 1, "files": stored}, indent=1))
    except OSError as error:
        print(f"[models] Could not save {VERIFIED_FILE.name}: {error}", flush=True)


def _verified_file(target: Path, file_spec: dict) -> bool:
    """Verify catalog bytes once per file identity; never trust size alone. A big file verified once (also by an
    earlier run of the app, see VERIFIED_FILE) is not hashed again until its size, times or file index change."""
    try:
        if target.is_symlink() or not target.is_file():
            return False
        stat = target.stat()
        if not stat.st_size or (file_spec.get("size_bytes") and stat.st_size != file_spec["size_bytes"]):
            return False
        expected = file_spec.get("sha256")
        if not expected:
            return True
        path = str(target.resolve())
        key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, expected)
        identity = _identity(stat, expected)
        cacheable = stat.st_size >= VERIFY_CACHE_BYTES
        with _VERIFY_LOCK:
            if cacheable and key in _VERIFIED:
                return _VERIFIED[key]
            if cacheable and _stored().get(path) == identity:
                _VERIFIED[key] = True
                return True
        with target.open("rb") as stream:
            valid = hashlib.file_digest(stream, "sha256").hexdigest().lower() == expected.lower()
        if cacheable:
            with _VERIFY_LOCK:
                _VERIFIED[key] = valid
                if valid and _identity(target.stat(), expected) == identity:  # unchanged while it was read
                    _remember(path, identity)
        return valid
    except OSError:
        return False


def _known(target: Path, file_spec: dict) -> bool:
    """Whether _verified_file answers for this file without reading all of it (a small file, a file without a
    pinned SHA-256, or a big one whose identity was verified before)."""
    try:
        stat = target.stat()
        expected = file_spec.get("sha256")
        if not expected or stat.st_size < VERIFY_CACHE_BYTES:
            return True
        path = str(target.resolve())
        key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, expected)
        with _VERIFY_LOCK:
            return key in _VERIFIED or _stored().get(path) == _identity(stat, expected)
    except OSError:
        return True  # a missing file is not read


def announce_checks(model_ids: List[str], progress_cb: Optional[ProgressCb], exact: bool = False) -> None:
    """Tell progress_cb which model's files are about to be read in full to check their SHA-256 (the first use of a
    file, or a file that changed), so the time it takes has its own step in a job's log."""
    if progress_cb is None:
        return
    for model_id in dict.fromkeys(model_ids if exact else [tiered(m) for m in model_ids]):
        model = config.get_model(model_id)
        if model and not all(_known(_file_target(spec), spec) for spec in model["files"]):
            progress_cb(0.0, f"Checking the {model['name'].split(' — ')[0]} files…")


def _file_target(file_spec: dict) -> Path:
    name = file_spec.get("save_as") or Path(file_spec["filename"]).name
    return config.destination_dir(file_spec["dest"]) / _safe_name(name)


def sources(file_spec: dict, model: Optional[dict] = None) -> List[dict]:
    """What the installer downloads for a catalog file: the file itself, or for a file made by an install step the
    official files the step reads. Each item has filename, dest, size_bytes, sha256 and either url or repo_id and
    revision (a source's own dest, repo_id and revision win; dest falls back to the step's "source_dest", repo_id to
    the file's or the model's, revision to the file's)."""
    convert = file_spec.get("convert")
    if not convert:
        items = [file_spec]
    else:
        items = [dict(source, dest=source.get("dest") or convert["source_dest"],
                      **({} if source.get("url") else
                         {"revision": source.get("revision") or file_spec.get("revision")}))
                 for source in convert["sources"]]
    repo = file_spec.get("repo_id") or (model or {}).get("repo_id")
    return [item if item.get("url") or item.get("repo_id") or not repo else dict(item, repo_id=repo)
            for item in items]


def _saved_name(item: dict) -> str:
    return item.get("save_as") or Path(item["filename"]).name


def step_inputs(file_spec: dict, model: Optional[dict] = None) -> List[Path]:
    """The files an install step reads, in the order of its "input" (one file name or a list), each in the folder of
    the source of that name."""
    convert = file_spec["convert"]
    names = convert["input"] if isinstance(convert["input"], list) else [convert["input"]]
    folders = {_saved_name(source): source["dest"] for source in sources(file_spec, model)}
    missing = [name for name in names if name not in folders]
    if missing or not names:
        raise DownloadError(f"The model catalog's install step for {_saved_name(file_spec)} reads "
                            f"{missing or 'nothing'}, which is not among its sources.")
    return [config.destination_dir(folders[name]) / _safe_name(name) for name in names]


def _equivalent_path(file_spec: dict, other: dict) -> Path:
    return config.destination_dir(other.get("dest", file_spec["dest"])) / _safe_name(other["filename"])


def _equivalent(file_spec: dict, verify: bool = True) -> Optional[Path]:
    """An earlier installation's file that holds the same weights (catalog "equivalents"), when it is on disk with
    its pinned size (and, with verify, its SHA-256)."""
    for other in file_spec.get("equivalents", ()):
        path = _equivalent_path(file_spec, other)
        if verify and _verified_file(path, other):
            return path
        if not verify and path.is_file() and not path.is_symlink() and path.stat().st_size == other["size_bytes"]:
            return path
    return None


def _link_identical(file_spec: dict, target: Path) -> bool:
    """Hard-link an equivalent with the catalog file's own size and SHA-256 (the very same bytes) into place, so
    the file is installed under its catalog name without downloading or making it again. The earlier file stays."""
    if target.exists() or target.is_symlink() or not file_spec.get("sha256"):
        return False
    for other in file_spec.get("equivalents", ()):
        if (other.get("sha256", "").lower() != file_spec["sha256"].lower()
                or other.get("size_bytes") != file_spec.get("size_bytes")):
            continue
        source = _equivalent_path(file_spec, other)
        if _verified_file(source, file_spec):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
            except OSError:
                continue
            return True
    return False


def _file_installed(file_spec: dict) -> bool:
    target = _file_target(file_spec)
    return (_verified_file(target, file_spec) or _link_identical(file_spec, target)
            or _equivalent(file_spec) is not None)


def installed(model: dict) -> bool:
    """Every file of the model is on disk with the catalog's size and SHA-256 (or one of its "equivalents" is).
    A file that an earlier installation keeps under an earlier name with the same bytes ("equivalents") is linked
    into place first, so every check sees it, not only `ensure`."""
    return all(_file_installed(fs) for fs in model["files"])


# ----------------------------- Model sets -----------------------------
# "full" uses the catalog's base entries. "lite" swaps in each "<id>-lite" entry: smaller
# quantizations of the same models for GPUs with less VRAM, RAM and disk.
TIERS = ("full", "lite")
LITE_BELOW_GB = 15.0


@lru_cache(maxsize=1)
def _gpu_total_gb() -> float:
    return vram.get_gpu_info().total_gb


def small_gpu() -> bool:
    """A GPU with less than 15 GB of VRAM (unknown counts as large)."""
    return 0 < _gpu_total_gb() < LITE_BELOW_GB


def default_tier() -> str:
    """Lite on a GPU with less than 15 GB of VRAM; full otherwise or when it is unknown."""
    return "lite" if small_gpu() else "full"


def tier() -> str:
    """The model set chosen in Settings, or the one suited to this GPU."""
    chosen = prefs.load().get("tier")
    return chosen if chosen in TIERS else default_tier()


def tiered(model_id: str, tier_name: Optional[str] = None) -> str:
    """Catalog entry that serves `model_id` in the given (default: current) model set."""
    lite = model_id + "-lite"
    return lite if (tier_name or tier()) == "lite" and config.get_model(lite) else model_id


def model_path(model_id: str, dest: str, *subs: str, tier_name: Optional[str] = None) -> Optional[Path]:
    """Absolute path of the installed file of the model set's entry in `dest` whose name contains every
    fragment. Selection follows the catalog, so full and Lite files can sit side by side."""
    model = config.get_model(tiered(model_id, tier_name)) or {}
    for spec in model.get("files", []):
        target = _file_target(spec)
        if spec["dest"] != dest or not all(s in target.name.lower() for s in subs):
            continue
        if target.is_file():
            return target
        equivalent = _equivalent(spec, verify=False)
        if equivalent is not None:
            return equivalent
    return None


def is_installed(model_id: str, exact: bool = False) -> bool:
    """exact=False checks the entry of the current model set (see tiered)."""
    model = config.get_model(model_id if exact else tiered(model_id))
    return model is not None and installed(model)


def ensure(model_ids: List[str], progress_cb: Optional[ProgressCb] = None,
           exact: bool = False) -> Optional[str]:
    """Plug and play: pobiera brakujace modele przed pierwszym uzyciem.

    Ids are mapped to the current model set unless exact=True.
    Zwraca komunikat bledu albo None, gdy wszystko jest na dysku.
    """
    if not exact:
        model_ids = [tiered(m) for m in model_ids]
    missing = []
    for model_id in dict.fromkeys(model_ids):
        announce_checks([model_id], progress_cb, exact=True)
        if not is_installed(model_id, exact=True):
            missing.append(model_id)
    if missing and downloads_blocked():  # the Agent page's work, without the owner's permission to download
        names = [(config.get_model(m) or {"name": m})["name"].split(" — ")[0] for m in missing]
        return f"{BLOCKED_PREFIX}: {', '.join(names)}."
    for i, mid in enumerate(missing):
        model = config.get_model(mid)
        if model is None:
            return f"Unknown model: {mid}. Check the installed model catalog."
        name = model["name"].split(" — ")[0]

        def cb(frac: float, desc: str, _i=i, _name=name) -> None:
            if progress_cb:
                progress_cb((_i + frac) / len(missing),
                            f"First use — downloading {_name}: {desc}")

        try:
            download_model(mid, progress_cb=cb)
        except Exception as e:  # noqa: BLE001 - komunikat trafia do UI
            return f"Could not download {name}: {e}"
    return None


def download_model(model_id: str, progress_cb: Optional[ProgressCb] = None,
                   cancel_flag: Optional[Callable[[], bool]] = None) -> None:
    """Pobiera wszystkie pliki modelu (istniejace pomija). progress_cb(0-1, opis)."""
    model = config.get_model(model_id)
    if model is None:
        raise ValueError(f"Unknown model: {model_id}")

    token = get_hf_token()
    files = model["files"]
    n = len(files)
    for i, fs in enumerate(files):
        name = fs.get("save_as") or Path(fs["filename"]).name
        target = _file_target(fs)
        if _link_identical(fs, target):
            if progress_cb is not None:
                progress_cb((i + 1) / n, f"Reused {name} from this installation")
            continue

        def file_cb(done: int, total: int, _i=i, _name=name, _verb="") -> None:
            if progress_cb is None:
                return
            frac = (done / total) if total else 0.0
            progress_cb((_i + frac) / n,
                        f"{_verb}{_name}: {done / 2**20:.0f}/{total / 2**20:.0f} MB")

        if fs.get("convert") or fs.get("equivalents"):
            if _verified_file(target, fs) or _equivalent(fs) is not None:
                continue
        if fs.get("convert"):
            _install_converted(model, fs, token, file_cb, cancel_flag)
            continue
        _fetch(sources(fs, model)[0], token, file_cb, cancel_flag)
    if progress_cb:
        progress_cb(1.0, "Ready")


def _fetch(source: dict, token: Optional[str], progress_cb, cancel_flag) -> Path:
    """Download one source (see `sources`) into its folder, or keep it when it is there already."""
    folder = config.destination_dir(source["dest"])
    if source.get("url"):
        return download_url(source["url"], folder, source.get("save_as") or Path(source["filename"]).name,
                            expected_size=source["size_bytes"], sha256=source["sha256"],
                            progress_cb=progress_cb, cancel_flag=cancel_flag)
    return download_file(repo_id=source["repo_id"], filename=source["filename"], dest_dir=folder, token=token,
                         progress_cb=progress_cb, cancel_flag=cancel_flag, save_as=source.get("save_as"),
                         revision=source.get("revision", "main"), expected_size=source.get("size_bytes", 0),
                         sha256=source.get("sha256"))


def _catalog_targets() -> set:
    return {_file_target(spec).resolve() for model in config.CATALOG["models"] for spec in model["files"]}


def _install_converted(model: dict, file_spec: dict, token: Optional[str], file_cb, cancel_flag) -> None:
    """Download the official files of an install step (those already on disk are only verified), run the step,
    and delete its sources afterwards when the catalog says so: they are pinned and can be downloaded again."""
    convert = file_spec["convert"]
    if convert["tool"] not in CONVERTERS:
        raise DownloadError(f"The model catalog names an unknown install step: {convert['tool']}.")
    module, function = CONVERTERS[convert["tool"]]
    step = getattr(importlib.import_module(module), function)
    fetched = [_fetch(source, token, file_cb, cancel_flag) for source in sources(file_spec, model)]
    target = _file_target(file_spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Every step writes <target>.part from its first byte, so the .part a cancelled or failed run left is room too.
    part = target.with_name(target.name + ".part")
    leftover = part.stat().st_size if part.is_file() and not part.is_symlink() else 0
    if shutil.disk_usage(target.parent).free + leftover < file_spec["size_bytes"]:
        counted = f", counting the {leftover / 1e9:.1f} GB of {part.name} it overwrites" if leftover else ""
        raise DownloadError(f"{target.name} needs {file_spec['size_bytes'] / 1e9:.1f} GB of free disk space"
                            f"{counted}.")
    options = {key: convert[key] for key in _CONVERTER_OPTIONS if key in convert}
    try:
        step(*step_inputs(file_spec, model), target,
             expected_size=file_spec["size_bytes"], expected_sha256=file_spec["sha256"],
             progress_cb=lambda done, total: file_cb(done, total, _verb="Preparing "), cancel_flag=cancel_flag,
             **options)
    except DownloadError:
        raise
    except Exception as exc:  # noqa: BLE001 - the step's own message, for the UI
        raise DownloadError(f"Could not prepare {target.name}: {exc}") from exc
    if convert.get("delete_sources_after"):
        keep = _catalog_targets()
        for path in fetched:
            if path.resolve() not in keep:
                path.unlink(missing_ok=True)


# ----------------------------- Sizes -----------------------------

def _unique_specs(model_ids: List[str]) -> Iterator[dict]:
    """Each catalog file of the given entries once (entries share VAEs, LoRAs and encoders)."""
    seen = set()
    for model_id in model_ids:
        model = config.get_model(model_id) or {}
        for spec in model.get("files", []):
            key = (spec["dest"], _saved_name(spec))
            if key not in seen:
                seen.add(key)
                yield dict(spec, _model=model)


def download_bytes(model_ids: List[str]) -> int:
    """What a clean installation of these entries downloads: the files, or an install step's sources."""
    found = {}
    for spec in _unique_specs(model_ids):
        for source in sources(spec, spec["_model"]):
            found[(source["dest"], _saved_name(source))] = source["size_bytes"]
    return sum(found.values())


def disk_bytes(model_ids: List[str]) -> int:
    """The disk space these entries take once installed: their files, and the sources of an install step that keeps
    them ("delete_sources_after" false). A kept source that is a file of these entries itself counts once."""
    specs = list(_unique_specs(model_ids))
    counted = {(spec["dest"], _saved_name(spec)) for spec in specs}
    total = sum(spec["size_bytes"] for spec in specs)
    for spec in specs:
        convert = spec.get("convert")
        if convert and not convert.get("delete_sources_after"):
            for source in sources(spec, spec["_model"]):
                key = (source["dest"], _saved_name(source))
                if key not in counted:
                    counted.add(key)
                    total += source["size_bytes"]
    return total


def _present(path: Path, size: int) -> bool:
    return path.is_file() and (not size or path.stat().st_size == size)


def missing_bytes(model_ids: List[str]) -> Tuple[int, int]:
    """(bytes still to download, free disk space the installation needs at its peak) for these entries, judged by
    name and size (hashes are checked while installing). An install step needs its missing sources and its
    result on disk at the same time; its sources are deleted afterwards when the catalog says so. A source that is a
    file of these entries itself (or another step's source) counts once."""
    download = need = largest_sources = 0
    specs = list(_unique_specs(model_ids))
    counted = {(spec["dest"], _saved_name(spec)) for spec in specs}  # catalog files count as themselves
    for spec in specs:
        target = _file_target(spec)
        if _present(target, spec["size_bytes"]) or _equivalent(spec, verify=False) is not None:
            continue
        convert = spec.get("convert")
        if not convert:
            download += spec["size_bytes"]
            need += spec["size_bytes"]
            continue
        missing = 0
        for source in sources(spec, spec["_model"]):
            key = (source["dest"], _saved_name(source))
            if key in counted:
                continue
            counted.add(key)
            if not _present(config.destination_dir(source["dest"]) / _saved_name(source), source["size_bytes"]):
                missing += source["size_bytes"]
        download += missing
        need += spec["size_bytes"]
        if convert.get("delete_sources_after"):
            largest_sources = max(largest_sources, missing)
        else:
            need += missing
    return download, need + largest_sources
