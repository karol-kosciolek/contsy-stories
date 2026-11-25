# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""What the agent's tools stand on: the owner's permissions, the backups made before a change, the ownership of queue
jobs, and every operation that deletes or changes something that is not the agent's own.

The tools (app/agent_tools*.py) never delete, stop, pause or change a setting themselves: they call the methods here,
and each of those asks `require` first. A test (tests/test_agent_tools.py) reads the tools' source and fails when one
reaches such an operation without going through this class."""
from __future__ import annotations

import json
import re
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from . import agent_config, characters, config, history, items, prefs, queue_manager, studio, style_library
from . import timeline, trash, universes, voices
from .agent_store import RunStore

CODES = ("INVALID", "NOT_FOUND", "NOT_OWNED", "STALE", "BUSY", "CONTENT_RULE", "MODE", "BUDGET", "MODELS_MISSING",
         "PERMISSION", "REVIEW_REQUIRED", "OWNER_STOPPED", "NEED_OWNER", "INTERNAL")
LABELS = {"write": "Use and change my library and projects", "delete": "Delete things (Recycle Bin)",
          "uploads": "Use the files I attach as photos", "queue_all": "Manage the whole GPU queue",
          "settings": "Change Contsy's settings", "downloads": "Download missing models"}
MAX_CHANGES = 500
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
DELETABLE = ("project", "character", "item", "universe", "style", "voice", "gallery_file")
QUEUE_ACTIONS = ("pause", "resume", "stop_current", "clear_finished")
MAX_PICTURES = 80  # pictures one conversation may hold: the API takes 100 in a request, the brief's come first
SLEEP_SLICE = 1.0  # s: how long a wait sleeps before it looks at the stop flag again


class ToolError(Exception):
    """A refusal a tool reports to the model: a code from CODES and what to do instead."""

    def __init__(self, code: str, message: str, **extra):
        super().__init__(message)
        if code not in CODES:
            raise ValueError(f"Unknown tool error code {code}.")
        self.code = code
        self.extra = extra


def _safe(text: str) -> str:
    return _SAFE.sub("_", str(text))[:60] or "x"


class Host:
    def __init__(self, store: RunStore, settings: dict, *, mode: Optional[str] = None, nonce: Optional[str] = None,
                 clock: Optional[Callable[[], float]] = None, sleep: Optional[Callable[[float], None]] = None):
        self.store = store
        self.settings = agent_config.clean_settings(settings)
        self.mode = mode if mode in agent_config.MODES else self.settings["mode"]
        self.nonce = nonce or secrets.token_hex(6)
        self.owner_tag = f"agent:{store.id}"
        self.clock = clock or time.time
        self._sleep = sleep or time.sleep
        self.stop_requested: Callable[[], bool] = lambda: False  # the runner sets this (Stop, shutdown)
        self.finished: Optional[dict] = None  # set by finish_run: how the run ended (the loop reads it)
        self._lock = threading.RLock()

    # --- permissions and mode ------------------------------------------------------------------------------

    def allowed(self, permission: str) -> bool:
        return agent_config.allowed(self.settings, permission)

    def require(self, permission: str, doing: str) -> None:
        if not self.allowed(permission):
            raise ToolError("PERMISSION", f"{doing} needs the owner's permission “{LABELS.get(permission, permission)}”, "
                            "and it is switched off for this run. Carry on without it, or ask_owner to switch it on for "
                            "the next run.")

    def require_produce(self, doing: str) -> None:
        if self.mode != "produce":
            raise ToolError("MODE", f"{doing} is final production, which this run does not do (it prepares everything "
                            "for the owner's approval). Prepare it without it and say in finish_run what is left to "
                            "produce.")

    def fence(self, text: str, label: str) -> str:
        """Text that came from outside the agent (the brief, a name, a caption) marked as data, not instructions."""
        label = _safe(label)
        return f"[[data {self.nonce} · {label}]]\n{text}\n[[end data {self.nonce}]]"

    # --- the run's record --------------------------------------------------------------------------------------

    def _resources(self) -> dict:
        found = self.store.read().get("resources")
        found = found if isinstance(found, dict) else {}
        return {"created": list(found.get("created") or []), "backups": dict(found.get("backups") or {})}

    def created(self, kind: str, ref: str) -> None:
        """Remember that this run made the thing: changing it later needs no backup."""
        with self._lock:
            resources = self._resources()
            key = f"{kind}:{ref}"
            if key not in resources["created"]:
                resources["created"].append(key)
                self.store.update(resources=resources)

    def is_mine(self, kind: str, ref: str) -> bool:
        return f"{kind}:{ref}" in self._resources()["created"]

    def note_change(self, text: str) -> None:
        """A line of the report's list of changes to the owner's things (what was changed, with the old value)."""
        with self._lock:
            changes = list(self.store.read().get("changes") or [])
            if len(changes) < MAX_CHANGES:
                changes.append(str(text)[:400])
                self.store.update(changes=changes)

    def ensure_backup(self, kind: str, ref: str, source=None, data=None) -> Optional[str]:
        """Before this run changes something of the owner's for the first time: a copy in the run's backups folder.
        `source` is a file or folder to copy, or `data` a JSON-able record. A thing the run made itself, or one already
        copied, is skipped. When the copy cannot be made nothing is changed."""
        with self._lock:
            resources = self._resources()
            key = f"{kind}:{ref}"
            if key in resources["created"] or key in resources["backups"]:
                return None
            name = f"{len(resources['backups']) + 1:03d}-{kind}-{_safe(ref)}"
            target = self.store.path("backups", name)
            try:
                if data is not None:
                    target.mkdir(parents=True)
                    (target / "record.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
                elif source is not None and Path(source).is_dir():
                    shutil.copytree(source, target)
                elif source is not None and Path(source).is_file():
                    target.mkdir(parents=True)
                    shutil.copy2(source, target / Path(source).name)
                else:
                    raise OSError("nothing to copy")
            except OSError as error:
                shutil.rmtree(target, ignore_errors=True)
                raise ToolError("INTERNAL", f"Could not back up {kind} {ref} first ({error}), so it was not changed.")
            resources["backups"][key] = name
            self.store.update(resources=resources)
            return name

    # --- files the owner attached to the run -----------------------------------------------------------------

    def brief_files(self) -> list[dict]:
        found = self.store.read().get("brief")
        return [item for item in found if isinstance(item, dict)] if isinstance(found, list) else []

    def reference_name(self, short: str) -> str:
        """The Timeline reference name of an attached picture, from the id the first message gave (its first 12
        characters); NOT_FOUND for anything else."""
        short = str(short or "").strip().lower()
        for item in self.brief_files():
            ref = item.get("ref") or ""
            if item.get("kind") == "image" and len(short) >= 8 and ref.lower().startswith(short):
                return ref
        raise ToolError("NOT_FOUND", "No attached picture has that id. Use an id from the owner's first message "
                        "(get_brief lists them).")

    # --- queue jobs ---------------------------------------------------------------------------------------------

    def queue_scope(self):
        """Jobs added inside this block are this run's: marked, labelled “Agent ·”, and unable to download models
        unless the owner allowed it."""
        return queue_manager.owner(self.owner_tag, downloads=self.allowed("downloads"), prefix="Agent: ")

    def own_jobs(self) -> list:
        return [job for job in queue_manager.snapshot() if job.owner == self.owner_tag]

    def job(self, job_id: int):
        """A job this run may look at: its own, or any with the whole-queue permission."""
        job = next((one for one in queue_manager.snapshot() if one.id == job_id), None)
        if job is None:
            raise ToolError("NOT_FOUND", f"Job #{job_id} is not in the queue any more (finished jobs are kept only for a "
                            "while).")
        if job.owner != self.owner_tag and not self.allowed("queue_all"):
            raise ToolError("NOT_OWNED", f"Job #{job_id} is not this run's. Only the owner's permission “"
                            f"{LABELS['queue_all']}” lets the agent see or stop it.")
        return job

    def account_jobs(self) -> float:
        """The GPU seconds this run's jobs have used so far; finished ones are remembered by id, because the queue keeps
        only the last 50."""
        with self._lock:
            record = self.store.read().get("gpu")
            record = record if isinstance(record, dict) else {}
            counted = set(record.get("counted") or [])
            seconds = float(record.get("seconds") or 0.0)
            live = 0.0
            for job in self.own_jobs():
                if job.status in (queue_manager.DONE, queue_manager.ERROR, queue_manager.STOPPED):
                    if job.id not in counted:
                        counted.add(job.id)
                        seconds += job.elapsed()
                elif job.status == queue_manager.RUNNING:
                    live += job.elapsed()
            if counted != set(record.get("counted") or []):
                self.store.update(gpu={"seconds": seconds, "counted": sorted(counted)[-500:]})
            return seconds + live

    # --- pictures the agent has been shown, and tries it has used ------------------------------------------------

    def pictures(self) -> int:
        """How many pictures the conversation now holds (the brief's and the ones the look_at tools showed)."""
        return int(self.store.read().get("pictures") or 0)

    def room_for_pictures(self, count: int) -> bool:
        return self.pictures() + count <= MAX_PICTURES

    def showed(self, count: int, scenes=()) -> None:
        """Note that `count` pictures were shown, covering these scenes (0 is the first): the report says which scenes the
        agent really looked at."""
        with self._lock:
            record = self.store.read()
            seen = sorted(set(record.get("saw") or []) | set(scenes))
            self.store.update(pictures=int(record.get("pictures") or 0) + count, saw=seen)

    def spend_try(self, key: str, limit: int, what: str) -> None:
        """Count one try at correcting something (a scene, a photo): the `limit`-th try is the last, then BUDGET. The count
        is in run.json, so it survives a resume and a fresh conversation."""
        with self._lock:
            tries = dict(self.store.read().get("tries") or {})
            if tries.get(key, 0) >= limit:
                raise ToolError("BUDGET", f"{what} has been corrected {limit} times already. Accept it as it is, and say in "
                                "finish_run that it is not right, so the owner can decide.")
            self.store.update(tries={key: tries.get(key, 0) + 1})

    def sleep(self, seconds: float) -> None:
        """Wait, in short slices; ToolError OWNER_STOPPED as soon as the run is told to stop."""
        end = self.clock() + seconds
        while (left := end - self.clock()) > 0:
            if self.stop_requested():
                raise ToolError("OWNER_STOPPED", "The owner stopped or paused this run.")
            self._sleep(min(SLEEP_SLICE, left))

    # --- operations the tools may not call themselves -------------------------------------------------------------

    def stop_jobs(self, job_ids: list[int]) -> list[int]:
        """Stop and withdraw jobs. A job that is not this run's needs the whole-queue permission."""
        stopped = []
        for job_id in job_ids:
            job = self.job(job_id)  # refuses a stranger's job without the permission
            if job.status == queue_manager.RUNNING:
                if queue_manager.stop_current(job.id):
                    stopped.append(job.id)
            elif job.status == queue_manager.PENDING:
                queue_manager.remove([job.id])
                stopped.append(job.id)
        return stopped

    def queue_control(self, action: str) -> str:
        self.require("queue_all", "Controlling the whole GPU queue")
        if action not in QUEUE_ACTIONS:
            raise ToolError("INVALID", f"Choose one of: {', '.join(QUEUE_ACTIONS)}.")
        if action == "pause":
            queue_manager.set_paused(True)
        elif action == "resume":
            queue_manager.set_paused(False)
        elif action == "stop_current":
            queue_manager.stop_current()
        else:
            queue_manager.clear_finished()
        self.note_change(f"GPU queue: {action}.")
        return f"GPU queue: {action} done."

    def set_preference(self, key: str, value) -> object:
        self.require("settings", "Changing a Contsy setting")
        if key not in prefs._DEFAULTS or key == "folder":  # the autosave folder is a path: only the owner picks one
            raise ToolError("INVALID", "Unknown setting. Choose one of: "
                            f"{', '.join(sorted(name for name in prefs._DEFAULTS if name != 'folder'))}.")
        old = prefs.load().get(key)
        if type(value) is not type(prefs._DEFAULTS[key]):
            raise ToolError("INVALID", f"{key} takes a {type(prefs._DEFAULTS[key]).__name__}.")
        prefs.save(**{key: value})
        self.note_change(f"Setting {key}: {old!r} -> {value!r}.")
        return old

    def delete(self, kind: str, ref: str) -> str:
        """Delete one thing the owner (not this run) made, with a copy of it first. Everything goes to the Recycle Bin
        or, for a style or universe, is kept in the run's backups."""
        self.require("delete", f"Deleting a {kind}")
        if kind == "project":
            try:
                project = timeline.load(ref)
            except (ValueError, OSError) as error:
                raise ToolError("NOT_FOUND", str(error))
            self.ensure_backup("project", ref, source=timeline._inside(timeline._root(ref), "project.json"))
            try:
                title = timeline.delete(ref)
            except timeline.Busy as error:
                raise ToolError("BUSY", str(error))
            except ValueError as error:
                busy = "GPU queue" in str(error) or "still in use" in str(error)
                raise ToolError("BUSY" if busy else "NOT_FOUND", str(error))
            message = f"Project ‘{title or project.title}’ moved to the Recycle Bin."
        elif kind == "character":
            found = characters.get(ref)
            if found is None:
                raise ToolError("NOT_FOUND", "No such character.")
            self.ensure_backup("character", ref, source=characters._folder(ref))
            leaving = [one for one in universes.all_universes() if ref in one.get("members", [])]
            if leaving:  # deleting the character also takes it out of its universes: copy them first
                self.ensure_backup("universes", f"with-{ref}", data=leaving)
            message = characters.delete(ref)
            if characters.get(ref) is not None:
                raise ToolError("BUSY", message)
            if leaving:
                self.note_change(f"Character {ref} left the universes: {', '.join(one['name'] for one in leaving)}.")
        elif kind == "item":
            found = items.get(ref)
            if found is None:
                raise ToolError("NOT_FOUND", "No such place or item.")
            self.ensure_backup("item", ref, source=items._folder(ref))
            message = items.delete(ref)
            if items.get(ref) is not None:
                raise ToolError("BUSY", message)
        elif kind == "universe":
            found = universes.get(ref)
            if found is None:
                raise ToolError("NOT_FOUND", "No such universe.")
            self.ensure_backup("universe", ref, data=found)
            ok, message = universes.delete(ref)
            if not ok:
                raise ToolError("NOT_FOUND", message)
        elif kind == "style":
            found = style_library.get(ref)
            if found is None:
                raise ToolError("NOT_FOUND", "No such style of the owner's.")
            self.ensure_backup("style", ref, data=found)
            ok, message = style_library.delete(ref)
            if not ok:
                raise ToolError("NOT_FOUND", message)
        elif kind == "voice":
            found = voices.get(ref)
            if found is None or found.builtin:
                raise ToolError("NOT_FOUND", "No such imported voice (the built-in voices cannot be deleted).")
            ok, message = voices.delete(ref)
            if not ok:
                raise ToolError("NOT_FOUND", message)
        elif kind == "gallery_file":
            message = self._delete_result(ref)
        else:
            raise ToolError("INVALID", f"Choose what to delete: {', '.join(DELETABLE)}.")
        self.note_change(f"Deleted {kind} {ref}.")
        return message

    def _delete_result(self, name: str) -> str:
        path = config.OUTPUTS_DIR / Path(str(name)).name
        if Path(str(name)).name != str(name) or not path.is_file() or not studio.is_recorded(str(path)):
            raise ToolError("NOT_FOUND", "No such result in the Gallery (use a file name from list_gallery).")
        if not trash.move(path):
            raise ToolError("INTERNAL", "Could not move the file to the Recycle Bin.")
        history.forget([str(path)])
        return f"{path.name} moved to the Recycle Bin."
