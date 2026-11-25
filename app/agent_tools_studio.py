# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The agent's tools for the rest of Contsy: the GPU queue, the Gallery, the status of the PC, and Contsy's settings."""
from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from . import agent_eyes as eyes, agent_preflight, agent_store, characters, config, ffmpeg_utils, history, prefs
from . import queue_manager, studio, timeline, vram
from .agent_host import Host, ToolError
from .agent_tools import Reply, choice, many, number, obj, text, tool, whole
from .agent_tools_eyes import FULL
from .agent_tools_film import job_view

WAIT_DEFAULT, WAIT_MAX = 240, 1500
POLL_SECONDS = 2.0
IMAGE_SIDE = 1024
_MEDIA = {"image": (".png", ".jpg", ".jpeg", ".webp"), "video": (".mp4",), "all": studio.MEDIA}


# --- the queue -----------------------------------------------------------------------------------------------------

@tool("get_queue", "This run's jobs in the GPU queue, and how many of the owner's own jobs wait (the queue is shared and "
                   "goes first come, first served).", obj({}))
def _get_queue(host: Host, args: dict):
    jobs = queue_manager.snapshot()
    mine = [job_view(job) for job in jobs if job.owner == host.owner_tag]
    others = [job for job in jobs if job.owner != host.owner_tag]
    found = {"paused": queue_manager.paused(), "mine": mine,
             "others_waiting": sum(job.status in (queue_manager.PENDING, queue_manager.RUNNING) for job in others),
             "gpu_seconds_used": int(host.account_jobs())}
    if host.allowed("queue_all"):
        found["others"] = [job_view(job) for job in others if job.status in (queue_manager.PENDING,
                                                                              queue_manager.RUNNING)][:30]
    return found


@tool("wait_for_jobs",
      "Wait until the given jobs have finished or the time is up, then report them (status, progress, error, result "
      "file names). A long wait is normal: clips take minutes. A finished job's files are named here; "
      "add_to_package takes them from here.",
      obj({"job_ids": many("Job ids.", whole("A job id.", 1, 10_000_000), max_items=20),
           "timeout_seconds": whole(f"How long to wait at most (default {WAIT_DEFAULT}, up to {WAIT_MAX}).", 5, WAIT_MAX)},
          ("job_ids",)))
def _wait_for_jobs(host: Host, args: dict):
    ids = list(dict.fromkeys(args["job_ids"]))
    limit = args.get("timeout_seconds", WAIT_DEFAULT)
    jobs = [host.job(job_id) for job_id in ids]  # the queue changes these very objects, even after it forgets them
    started = host.clock()
    note = ""
    while True:
        host.account_jobs()
        if all(job.status in (queue_manager.DONE, queue_manager.ERROR, queue_manager.STOPPED) for job in jobs):
            break
        if host.stop_requested():
            raise ToolError("OWNER_STOPPED", "The owner stopped or paused this run.")
        if queue_manager.paused() and not any(job.status == queue_manager.RUNNING for job in queue_manager.snapshot()):
            note = "The owner paused the GPU queue, so nothing is running. Waiting does not help until it is resumed."
            break
        left = limit - (host.clock() - started)
        if left <= 0:
            note = "Time is up; the jobs are still going. Wait again, or do something else meanwhile."
            break
        host.sleep(min(POLL_SECONDS, left))
    rows = []
    for job in jobs:
        row = job_view(job)
        if job.error.startswith("Not installed, and downloads are switched off"):
            row["code"] = "MODELS_MISSING"
        rows.append(row)
    return {"jobs": rows, "all_finished": all(row["status"] in ("done", "failed", "stopped") for row in rows),
            **({"note": note} if note else {})}


@tool("stop_jobs", "Stop and withdraw this run's jobs (a running one is interrupted, a waiting one removed).",
      obj({"job_ids": many("Job ids.", whole("A job id.", 1, 10_000_000), max_items=20)}, ("job_ids",)))
def _stop_jobs(host: Host, args: dict):
    stopped = host.stop_jobs(args["job_ids"])
    return {"stopped": stopped}


@tool("queue_control", "Pause or resume the whole GPU queue, interrupt the running job, or clear the finished jobs. "
                       "Only with the owner's permission to manage the whole queue.",
      obj({"action": choice("What to do.", "pause", "resume", "stop_current", "clear_finished")}, ("action",)))
def _queue_control(host: Host, args: dict):
    return host.queue_control(args["action"])


# --- the PC and the settings -------------------------------------------------------------------------------------

@tool("get_status", "The state of this PC for planning: the GPU, free disk space, the queue, the models the run still "
                    "lacks, and the run's own GPU time used.", obj({}))
def _get_status(host: Host, args: dict):
    gpu = vram.get_gpu_info()
    jobs = queue_manager.snapshot()
    return {"mode": host.mode, "gpu": {"name": gpu.name, "memory_gb": round(gpu.total_gb, 1),
                                       "free_gb": round(gpu.free_gb, 1)} if gpu.available else None,
            "disk_free_gb": round(agent_store.free_gb(), 1),
            "queue": {"paused": queue_manager.paused(),
                      "running": sum(job.status == queue_manager.RUNNING for job in jobs),
                      "waiting": sum(job.status == queue_manager.PENDING for job in jobs)},
            "gpu_minutes_used": round(host.account_jobs() / 60, 1),
            "downloads_allowed": host.allowed("downloads"),
            "missing_models": [{"name": name, "download_gb": round(size / 2 ** 30, 1)}
                               for _, name, size in agent_preflight.missing_models(host.mode)]}


@tool("get_settings", "Contsy's own settings (autosave, face check, model set, default voices...).", obj({}))
def _get_settings(host: Host, args: dict):
    return prefs.load()


@tool("set_preference", "Change one Contsy setting. Only with the owner's permission to change settings; the old value "
                        "goes into the report.",
      obj({"key": choice("The setting.", *sorted(key for key in prefs._DEFAULTS if key != "folder")),
           "value": {"type": "string", "description": "The new value as text; true or false for a switch.",
                     "maxLength": 300}}, ("key", "value")))
def _set_preference(host: Host, args: dict):
    kind = type(prefs._DEFAULTS[args["key"]])
    value = args["value"]
    if kind is bool:
        if value not in ("true", "false"):
            raise ToolError("INVALID", f"{args['key']} is a switch: true or false.")
        value = value == "true"
    old = host.set_preference(args["key"], value)
    return {"key": args["key"], "old": old, "new": value}


# --- the Gallery ---------------------------------------------------------------------------------------------------

def _gallery_row(path: str) -> dict:
    found = Path(path)
    entry = history.entry_for(path) or {}
    row = {"file": found.name, "kind": "video" if found.suffix.lower() == ".mp4" else "image",
           "mb": round(found.stat().st_size / 2 ** 20, 2), "made": int(found.stat().st_mtime)}
    if entry.get("prompt"):
        row["prompt"] = str(entry["prompt"])[:200]
    return row


@tool("list_gallery", "The newest results in the Gallery: pictures and videos (file name, kind, size, the prompt that made "
                      "it).", obj({"kind": choice("image, video or all (default).", "image", "video", "all"),
                                   "limit": whole("How many (default 20).", 1, 50)}))
def _list_gallery(host: Host, args: dict):
    rows = []
    for path in studio.results(exts=_MEDIA[args.get("kind", "all")], limit=args.get("limit", 20)):
        try:
            rows.append(_gallery_row(path))
        except OSError:
            continue
    return rows


def _result_path(name: str) -> Path:
    path = config.OUTPUTS_DIR / Path(str(name)).name
    if Path(str(name)).name != str(name) or not path.is_file() or not studio.is_recorded(str(path)):
        raise ToolError("NOT_FOUND", "No such result. list_gallery and wait_for_jobs name the files.")
    return path


def _shows_uploaded_photo(path: Path) -> bool:
    """Whether this result may show a character or a place whose pictures must stay on this PC (agent_eyes.only_generated).
    Fails closed: a name the history keeps that no character has any more, a project that cannot be opened, a record that
    cannot be read: all count as yes."""
    try:
        entry = history.entry_for(str(path)) or {}
        known = {one["name"] for one in characters.all_characters()}
        named = [*(entry.get("characters") or []), *([entry["character"]] if entry.get("character") else [])]
        if any(name not in known for name in named):
            return True
        if any(eyes.photos_were_uploaded("character", slug) for slug in studio.file_members().get(path.name, set())):
            return True
        if entry.get("project"):
            return eyes.film_shows_uploaded(timeline.load(entry["project"]))
    except Exception:  # noqa: BLE001 - what cannot be worked out is not shown
        return True
    return False


def _small_png(source: Path) -> bytes:
    with Image.open(source) as picture:
        picture = picture.convert("RGB")
        picture.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
        out = io.BytesIO()
        picture.save(out, "PNG")
        return out.getvalue()


@tool("inspect_media",
      "Look at a result of the Gallery: its size and length and, only if the owner allowed sending pictures to the cloud, "
      "the picture itself or a frame of a video so you can check it.",
      obj({"file": text("A file name from list_gallery or wait_for_jobs.", max_length=200, min_length=1),
           "at_seconds": number("For a video: which second to show (default the middle).", 0, 3600)}, ("file",)))
def _inspect_media(host: Host, args: dict):
    path = _result_path(args["file"])
    video = path.suffix.lower() == ".mp4"
    info = {"file": path.name, "kind": "video" if video else "image", "mb": round(path.stat().st_size / 2 ** 20, 2)}
    seconds = None
    if video:
        seconds = ffmpeg_utils.probe_duration(path)
        info["seconds"] = round(seconds, 2) if seconds else None
    else:
        with Image.open(path) as picture:
            info["width"], info["height"] = picture.size
    if not host.settings["send_images"]:
        info["picture"] = "not shown: the owner has not allowed sending pictures to the cloud"
        return info
    if _shows_uploaded_photo(path):
        info["picture"] = "not shown: a character or place in it is not proven to be generated by Contsy, so it stays on this PC"
        return info
    if not host.room_for_pictures(1):
        info["picture"] = FULL
        return info
    if not video:
        host.showed(1)
        return Reply(info, [_small_png(path)])
    frame = host.store.path("work", f"frame_{path.stem}_{int(args.get('at_seconds', (seconds or 2) / 2) * 1000)}.png")
    ok, _ = ffmpeg_utils.frame_at(path, args.get("at_seconds", (seconds or 2) / 2), frame)
    if not ok or not frame.is_file():
        info["picture"] = "could not read a frame"
        return info
    host.showed(1)
    return Reply(info, [_small_png(frame)])
