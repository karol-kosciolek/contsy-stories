# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Where an agent run lives on disk: `agent_runs/<run id>/`. The disk is the source of truth (the page only reads it,
a closed browser tab or a restart loses nothing).

    run.json          the summary (state, usage, counters, owned things...), rewritten atomically
    events.jsonl      what the page's live view shows: one JSON line per step, appended (keys masked)
    transcript.jsonl  the conversation as the API must get it back, appended and never rewritten: a thinking block is
                      only valid in the conversation that made it, so a block is stored exactly as it came. Large
                      images and PDFs are kept as blobs and put back when a request is made
    system.txt, tools.json   the instructions and the tool list of the run, frozen when it starts
    brief/  results/  blobs/  backups/  work/  deliverables/

The conversation is checked for the API key before every append: a line that holds it is refused (LeakError) instead of
being rewritten, because rewriting would change what the model's own blocks say.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from . import agent_config, config, trash
from .persistence import atomic_text

RUNS_DIR = config.STATE_ROOT / "agent_runs"
_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
_BLOB = re.compile(r"[0-9a-f]{64}\.[a-z0-9]{2,5}")
FOLDERS = ("brief", "results", "blobs", "backups", "work", "deliverables")
STATES = ("created", "running", "waiting", "paused", "interrupted", "budget_stopped", "stopped", "failed",
          "completed")
LIVE_STATES = ("running", "waiting")
RESUMABLE = ("created", "paused", "interrupted", "budget_stopped", "stopped", "failed")
LINE_END = bytes([10])
WRITE_TRIES, WRITE_WAIT = 6, 0.05  # tries (and the first wait, s) of a read or replace of run.json
BLOB_MIN = 2048  # an image or a document of base64 text longer than this is kept as a blob
_SUFFIX = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif",
           "application/pdf": "pdf"}


class LeakError(RuntimeError):
    """A line of the conversation holds the API key; nothing was written."""


def new_run_id(now: Optional[float] = None) -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime(now)) + "-" + secrets.token_hex(3)


def valid_id(run_id) -> bool:
    return isinstance(run_id, str) and bool(_ID.fullmatch(run_id))


def run_dir(run_id: str) -> Path:
    if not valid_id(run_id):
        raise ValueError("Invalid agent run.")
    base = RUNS_DIR.resolve()
    path = base / run_id
    if path.resolve().parent != base:
        raise ValueError("The run folder has an unexpected location.")
    return path


def _append(path: Path, text: str) -> None:
    """One line appended and flushed to the disk (binary, so Windows does not change the line ends). A file that does not
    end with a line end (a write that was cut short) gets one first, so the cut line stays alone and the new one is whole."""
    with path.open("ab") as stream:
        if stream.tell():
            with path.open("rb") as check:
                check.seek(-1, os.SEEK_END)
                if check.read(1) != LINE_END:
                    stream.write(LINE_END)
        stream.write(text.encode("utf-8") + LINE_END)
        stream.flush()
        os.fsync(stream.fileno())


def _lines(path: Path) -> list[dict]:
    """The JSON lines of `path`; a line that does not parse (a write cut short) is left out. Lines end at a line feed
    only: text.splitlines() would also cut at U+2028, U+2029 or U+0085 inside a message and lose it."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    found = []
    for line in text.split(LINE_END.decode()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            found.append(value)
    return found


_MARK = re.compile(r"[$]+blob")


def _escape(value):
    """A key a model chose that looks like the blob marker ($blob, $$blob...) gets one more $, so it cannot be taken for a
    marker when the conversation is read back (_expand takes one off again)."""
    return {("$" + key if _MARK.fullmatch(key) else key): item for key, item in value.items()}


def _externalize(value, store: "RunStore"):
    """`value` with every large base64 payload moved into a blob (in place of its text: {"$blob": name})."""
    if isinstance(value, dict):
        source = value.get("source")
        if (value.get("type") in ("image", "document") and isinstance(source, dict)
                and source.get("type") == "base64" and isinstance(source.get("data"), str)
                and len(source["data"]) > BLOB_MIN):
            try:
                data = base64.b64decode(source["data"], validate=True)
            except (binascii.Error, ValueError):
                data = None
            if data is not None and base64.b64encode(data).decode("ascii") == source["data"]:
                name = store.put_blob(data, _SUFFIX.get(str(source.get("media_type")), "bin"))
                changed = {**source, "data": {"$blob": name}}
                return _escape({key: (changed if key == "source" else _externalize(item, store))
                                for key, item in value.items()})
        return _escape({key: _externalize(item, store) for key, item in value.items()})
    if isinstance(value, list):
        return [_externalize(item, store) for item in value]
    return value


def _expand(value, store: "RunStore"):
    """The inverse of _externalize: every {"$blob": name} becomes the base64 text it stands for, every escaped key loses
    the $ it was given."""
    if isinstance(value, dict):
        if set(value) == {"$blob"}:
            return base64.b64encode(store.read_blob(value["$blob"])).decode("ascii")
        return {(key[1:] if _MARK.fullmatch(key) and key.startswith("$$") else key): _expand(item, store)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, store) for item in value]
    return value


class RunStore:
    """One run's folder. Safe for one writer thread (the run) and any number of readers (the page)."""

    def __init__(self, run_id: str):
        self.id = run_id
        self.dir = run_dir(run_id)
        self._lock = threading.RLock()
        self._seq: Optional[int] = None
        self._turns: Optional[int] = None

    # --- files -----------------------------------------------------------------------------------------------

    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    def exists(self) -> bool:
        return self.path("run.json").is_file()

    def create(self, *, summary: dict, system: str, tools: list) -> dict:
        """Make the folder of a new run with its frozen instructions and tool list; `summary` is its first run.json."""
        with self._lock:
            if self.dir.exists():
                raise ValueError("This run already exists.")
            for name in FOLDERS:
                self.path(name).mkdir(parents=True)
            self.path("system.txt").write_bytes(system.encode("utf-8"))
            self.path("tools.json").write_bytes(json.dumps(tools, ensure_ascii=False, sort_keys=True)
                                                .encode("utf-8"))
            record = {"schema": 1, "id": self.id, "state": "created", "created": time.time(), **summary}
            self._write(record)
            return record

    def system_text(self) -> str:
        return self.path("system.txt").read_bytes().decode("utf-8")

    def tools(self) -> list:
        return json.loads(self.path("tools.json").read_bytes().decode("utf-8"))

    # --- run.json --------------------------------------------------------------------------------------------

    def _write(self, record: dict) -> None:
        """Replace run.json. Windows refuses the replace while the page (or a scanner) has the file open for a moment, so
        it is tried again a few times before it fails."""
        text = json.dumps(record, ensure_ascii=False, indent=1)
        for attempt in range(WRITE_TRIES):
            try:
                atomic_text(self.path("run.json"), text)
                return
            except PermissionError:
                if attempt == WRITE_TRIES - 1:
                    raise
                time.sleep(WRITE_WAIT * (attempt + 1))

    def read(self) -> dict:
        for attempt in range(WRITE_TRIES):
            try:
                record = json.loads(self.path("run.json").read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {}
            except PermissionError:  # the file is being replaced at this moment: look again
                if attempt == WRITE_TRIES - 1:
                    return {}
                time.sleep(WRITE_WAIT * (attempt + 1))
                continue
            except (OSError, ValueError):
                return {}
            return record if isinstance(record, dict) else {}
        return {}

    def update(self, **fields) -> dict:
        """Merge `fields` into run.json (a dict field is merged one level deep, any other replaced)."""
        with self._lock:
            record = self.read()
            if not record:
                raise ValueError("This run has no record.")
            for name, value in fields.items():
                if isinstance(value, dict) and isinstance(record.get(name), dict):
                    record[name] = {**record[name], **value}
                else:
                    record[name] = value
            record["updated"] = time.time()
            self._write(record)
            return record

    # --- events (the live view) ------------------------------------------------------------------------------

    def log(self, type_: str, **fields) -> int:
        """Append an event; the text of every field is masked of the key. Returns its number."""
        with self._lock:
            if self._seq is None:
                self._seq = len(_lines(self.path("events.jsonl")))
            self._seq += 1
            line = json.dumps({"seq": self._seq, "t": round(time.time(), 3), "type": type_, **fields},
                              ensure_ascii=False, default=str)
            _append(self.path("events.jsonl"), agent_config.redact(line))
            return self._seq

    def events(self, since: int = 0, limit: int = 200) -> list[dict]:
        found = [event for event in _lines(self.path("events.jsonl")) if event.get("seq", 0) > since]
        return found[-limit:] if limit else found

    # --- the conversation ------------------------------------------------------------------------------------

    def append_message(self, role: str, content, meta: Optional[dict] = None) -> int:
        """Append one message exactly as it must be sent back. Large images and documents go to blobs. Refused with
        LeakError when the line holds the API key."""
        with self._lock:
            if self._turns is None:
                self._turns = len(_lines(self.path("transcript.jsonl")))
            stored = _externalize(content, self)
            line = json.dumps({"i": self._turns + 1, "role": role, "content": stored, "meta": meta or {}},
                              ensure_ascii=False, separators=(",", ":"))
            if agent_config.redact(line) != line:
                raise LeakError("A message of the run holds the API key, so it was not saved. The run was stopped.")
            _append(self.path("transcript.jsonl"), line)
            self._turns += 1
            return self._turns

    def records(self) -> list[dict]:
        """The saved messages with their meta (request id, usage...), blobs still as references."""
        return _lines(self.path("transcript.jsonl"))

    def wire_messages(self) -> list[dict]:
        """The conversation as a request sends it: {"role", "content"} with the blobs put back."""
        return [{"role": record["role"], "content": _expand(record["content"], self)} for record in self.records()]

    # --- blobs and the brief ---------------------------------------------------------------------------------

    def put_blob(self, data: bytes, suffix: str) -> str:
        name = f"{hashlib.sha256(data).hexdigest()}.{suffix}"
        target = self.path("blobs", name)
        if not target.exists():
            temp = target.with_suffix(target.suffix + ".part")
            temp.write_bytes(data)
            temp.replace(target)
        return name

    def read_blob(self, name: str) -> bytes:
        if not isinstance(name, str) or not _BLOB.fullmatch(name):
            raise ValueError("Invalid blob name.")
        return self.path("blobs", name).read_bytes()

    def put_brief(self, data: bytes, suffix: str) -> str:
        """Keep a brief file in the run (named by its hash); returns the name."""
        name = f"{hashlib.sha256(data).hexdigest()}.{suffix}"
        target = self.path("brief", name)
        # A file of the wrong size was cut short (a crash mid-write) and is written again; the .part file and the
        # rename keep a half-written file from ever carrying the final name
        if not target.exists() or target.stat().st_size != len(data):
            part = target.with_name(target.name + ".part")
            part.write_bytes(data)
            part.replace(target)
        return name

    # --- the page's view -------------------------------------------------------------------------------------

    def view(self, events: int = 60) -> dict:
        record = self.read()
        return {**record, "events": self.events(limit=events)}


def list_runs() -> list[dict]:
    """The run.json of every run, newest first (a folder without a readable record is left out)."""
    try:
        names = sorted((path.name for path in RUNS_DIR.iterdir() if path.is_dir() and valid_id(path.name)),
                       reverse=True)
    except OSError:
        return []
    runs = []
    for name in names:
        record = RunStore(name).read()
        if record.get("id") == name:
            runs.append(record)
    return runs


def startup_sweep() -> list[str]:
    """After a start of the app no run can still be working: a run left `running` or `waiting` is marked
    `interrupted` (it can be resumed). Returns their ids."""
    swept = []
    for record in list_runs():
        if record.get("state") in LIVE_STATES:
            store = RunStore(record["id"])
            store.update(state="interrupted", wait_reason="", interrupt_reason="app_restart")
            store.log("state", state="interrupted", reason="app_restart")
            swept.append(record["id"])
    return swept


def move_to_trash(run_id: str) -> bool:
    """Move a finished run's folder to the Recycle Bin (an action of the owner only; refused while it works)."""
    store = RunStore(run_id)
    if store.read().get("state") in LIVE_STATES:
        raise ValueError("This run is still working. Stop it first.")
    return trash.move(store.dir)


def free_gb() -> float:
    """Free disk space where the runs are kept, in GB."""
    base = RUNS_DIR
    while not base.exists() and base.parent != base:
        base = base.parent
    return shutil.disk_usage(base).free / 1e9
