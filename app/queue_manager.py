# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Kolejka zadan bez limitu - zapisywana na dysku, wykonywana po kolei w tle.

Klikniecie "Generuj" tylko dodaje zadanie (add) i od razu wraca, wiec mozna dodac dziesiatki
filmow. Jeden watek roboczy bierze zadania po kolei (na karcie 16 GB zawsze jedno naraz).
Oczekujace zadania i ich zalaczniki (kopie w queue_files/) przetrwaja restart aplikacji -
przerwane w polowie zadanie wraca na poczatek kolejki. Zadania mozna usuwac pojedynczo,
wszystkie naraz, wstrzymac kolejke albo przerwac biezace (app/cancel.py).
"""
from __future__ import annotations

import json
import math
import re
import shutil
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from . import cancel, config, history
from .downloader import no_downloads
from .persistence import atomic_text

PENDING, RUNNING, DONE, ERROR, STOPPED = "w kolejce", "generowanie", "gotowe", "blad", "przerwane"
_STATE = config.STATE_CONFIG_DIR / "queue.json"
FILES_DIR = config.STATE_ROOT / "queue_files"
_KEEP_FINISHED = 50

# runner(params, progress_cb) -> (ok, pliki, uwagi, blad)
Runner = Callable[[dict, Callable[[float, str], None]], Tuple[bool, List[str], List[str], str]]
_RUNNERS: Dict[str, Runner] = {}
# szacunek czasu przed pierwszym pomiarem (s na jednostke: wideo na sekunde filmu, 480p w jednym trybie
# app/video_gen.py), obraz - potem srednia z ostatnich zadan.
_RATE = {"wideo": 35.0, "obraz": 30.0, "character_photos": 25.0, "character_voice": 40.0, "item_photos": 25.0,
         "timeline_prepare": 110.0, "timeline_merge": 3.0, "timeline_render": 0.6,
         # Timeline sound preview: speech and EzAudio background per second of the scene. Timeline export
         # (generate missing clips, join, render): per second of clip generation, see Job.units.
         "timeline_sound": 8.0, "timeline_produce": 110.0,
         # Timeline AI edit of made scenes: per second of the scenes it makes again (a scene's take is made from
         # its own picture, so no starting frame is composed)
         "timeline_edit": 30.0}
# Seconds of work per unit for the timeline jobs: scene or film seconds, or for an export the seconds
# of clip generation it needs (its params["work_seconds"]: missing clips plus joining and writing).
_SECOND_KINDS = ("wideo", "timeline_prepare", "timeline_merge", "timeline_render", "timeline_sound", "timeline_edit")
# Averages saved under retired names. Before the one video mode (2026-11-14) each mode learned its own:
# the one that remains was Preview ("wideo-podglad"), so a saved queue brings that average back as the
# video rate and drops those of the removed Quality ("wideo") and Fast ("wideo-szybkie") modes. Image
# upscaling ("upscale") and narration jobs ("narracja") are gone too, and so is character training
# (2026-11-15): the LoRA training ("trening") and its photo sets ("zestaw").
_RETIRED_RATES = {"wideo-podglad": "wideo", "wideo-szybkie": None, "upscale": None, "narracja": None,
                  "trening": None, "zestaw": None}
# Job kinds of removed features: saved jobs of these kinds still load and are listed, but never run.
REMOVED_KINDS = {"trening": "Character training (removed)", "zestaw": "Training photo set (removed)"}
REMOVED_MSG = "Character training was removed from Contsy. Characters now use reference photos."
# Jobs on a character draft of the creator: the draft belongs to a browser tab, whose state lives in the server's
# memory, so after a restart no tab can open or save it any more and such a job must not run.
SESSION_KINDS = ("character_photos", "character_voice", "item_photos")
SESSION_MSG = ("The app was restarted, so the character draft this job was for was closed. Start again on the "
               "Characters page.")
# The Items page's draft jobs (app/items.py) say where to start again.
_SESSION_MSGS = {"item_photos": ("The app was restarted, so the place or item draft this job was for was closed. Start "
                                 "again on the Items page.")}
# GPU seconds per character beyond the first in a shared frame (an estimate, not measured): its primary photo is
# one more picture in the Qwen-Image-Edit pass that composes the frame (about 25 s per pass), and every two beyond
# the first three images add a pass.
CAST_EXTRA = {"wideo": 15.0, "obraz": 15.0}


@dataclass
class Job:
    id: int
    kind: str          # a kind with a runner (see register): wideo, obraz, character_*, timeline_*
    label: str
    params: dict
    status: str = PENDING
    progress: float = 0.0
    desc: str = ""
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    error: str = ""
    files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    owner: str = ""          # who queued it: "" = a person at the pages, else the tag of the work that did (owner())
    downloads: bool = True   # False: the job may not download models (see downloader.no_downloads)

    def elapsed(self) -> float:
        return ((self.finished or time.time()) - self.started) if self.started else 0.0

    def units(self) -> float:
        """Jednostki pracy do szacowania czasu (wideo: sekundy filmu, character photos: photos)."""
        if self.kind in ("character_photos", "item_photos"):
            return float(self.params.get("count", 1))
        if self.kind in _SECOND_KINDS:
            return float(self.params.get("seconds", 1))
        if self.kind == "timeline_produce":
            return float(self.params.get("work_seconds", self.params.get("seconds", 1)))
        return 1.0

    def cast_extra(self) -> float:
        """Extra GPU seconds for the characters composed into shared frames beyond the first
        in each (app/cast.py tag: their primary photos go into the frame's Qwen-Image-Edit composition)."""
        n = self.params.get("extra_characters", 0) if isinstance(self.params, dict) else 0
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            n = 0
        return CAST_EXTRA.get(self.kind, 0.0) * n

    def estimate(self) -> float:
        return rate(self.kind) * self.units() + self.cast_extra()

    def eta(self) -> Optional[float]:
        if self.status != RUNNING:
            return None
        total = self.elapsed() / self.progress if self.progress > 0.05 else self.estimate()
        return max(0.0, total - self.elapsed())


_JOBS: List[Job] = []
_COND = threading.Condition()
_CONTEXT = threading.local()
_STATE_INFO = {"next_id": 1, "paused": False}
_REQUEUE = threading.Event()  # przerwanie przez pause_now: zadanie wraca do kolejki
_started = False


# ----------------------------- zapis na dysku -----------------------------

def _save() -> None:
    """Atomically save the queue without stopping its worker on a filesystem error."""
    data = {**_STATE_INFO, "rates": _RATE, "jobs": [asdict(j) for j in _JOBS]}
    try:
        atomic_text(_STATE, json.dumps(data, ensure_ascii=False, indent=1))
    except OSError as exc:
        print(f"[queue] Could not save {_STATE.name}: {exc}", flush=True)


def _load() -> None:
    try:
        data = json.loads(_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    try:
        _STATE_INFO["next_id"] = max(1, int(data.get("next_id", 1)))
    except (ValueError, TypeError, OverflowError):
        _STATE_INFO["next_id"] = 1
    _STATE_INFO["paused"] = bool(data.get("paused", False))
    rates = data.get("rates", {})
    if isinstance(rates, dict):
        if "wideo-podglad" in rates:  # saved before the one video mode: "wideo" was the Quality average
            rates = {key: value for key, value in rates.items() if key != "wideo"}
        for key, value in rates.items():
            key = _RETIRED_RATES.get(key, key)
            if key is None:
                continue
            try:
                value = float(value)
                if math.isfinite(value) and value > 0:
                    _RATE[key] = value
            except (ValueError, TypeError, OverflowError):
                continue
    seen = {j.id for j in _JOBS}
    for item in data.get("jobs", []) if isinstance(data.get("jobs"), list) else []:
        try:
            if not isinstance(item, dict) or not isinstance(item.get("params"), dict):
                continue
            job = Job(**{k: v for k, v in item.items() if k in Job.__dataclass_fields__})
            if not isinstance(job.id, int) or isinstance(job.id, bool) or job.id < 1 or job.id in seen:
                continue
            if not isinstance(job.kind, str) or not isinstance(job.label, str):
                continue
            if not isinstance(job.status, str) or job.status not in (PENDING, RUNNING, DONE, ERROR, STOPPED):
                continue
            if not all(isinstance(value, str) for value in (job.desc, job.error, job.owner)):
                continue
            if not isinstance(job.downloads, bool):
                continue
            if not all(isinstance(value, list) and all(isinstance(entry, str) for entry in value)
                       for value in (job.files, job.notes)):
                continue
            for key in ("progress", "created"):
                number = float(getattr(job, key))
                if not math.isfinite(number):
                    raise ValueError("Invalid saved job timestamp or progress.")
                setattr(job, key, number)
            job.progress = min(max(job.progress, 0.0), 1.0)
            for key in ("started", "finished"):
                value = getattr(job, key)
                if value is not None:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError("Invalid saved job timestamp.")
                    setattr(job, key, number)
            for key in ("steps", "seconds", "count", "work_seconds"):
                if key in job.params:
                    number = float(job.params[key])
                    if not math.isfinite(number) or number <= 0:
                        raise ValueError("Invalid job size.")
            seen.add(job.id)
            if job.status == RUNNING:
                job.status, job.progress, job.desc, job.started = PENDING, 0.0, "", None
            if job.status == PENDING and job.kind in REMOVED_KINDS:  # waiting since before the removal
                job.status, job.error, job.finished = STOPPED, REMOVED_MSG, time.time()
            if job.status == PENDING and job.kind in SESSION_KINDS:
                job.status, job.error, job.finished = STOPPED, _SESSION_MSGS.get(job.kind, SESSION_MSG), time.time()
            if job.status == PENDING and (refusal := _prompt_refusal(job.params)):
                job.status, job.error, job.finished = ERROR, refusal, time.time()
            _JOBS.append(job)
        except (TypeError, ValueError, OverflowError) as exc:  # said: no job vanishes without a word
            print(f"[queue] A saved job could not be read and was left out: {exc}", flush=True)
            continue
    _STATE_INFO["next_id"] = max([_STATE_INFO["next_id"], *[j.id + 1 for j in _JOBS]])


def _prompt_refusal(params: dict) -> str:
    """Why a job must not run ("" = it may): broken parameters, or texts that together combine
    minors with sexual content (llm.blocked). Asked when a job is added, restored and started."""
    if not isinstance(params, dict):
        return "Invalid job parameters."
    keys = {"idea", "positive", "prompt", "text", "story", "description", "theme",
            "scene", "caption", "script", "dialogue", "scenes", "lines", "appearance", "design", "look"}
    def texts(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for entry in value:
                yield from texts(entry)
        elif isinstance(value, dict):
            for key, entry in value.items():
                if key in keys:
                    yield from texts(entry)
    from . import llm  # Lazy import avoids the queue/LLM initialization cycle.
    return llm.BLOCK_MSG if llm.blocked(*texts(params)) else ""


# ----------------------------- API dla interfejsu -----------------------------

def rate(kind: str) -> float:
    """Sekundy pracy karty na jednostke (srednia z ostatnich zadan tego rodzaju)."""
    return _RATE.get(kind, 60.0)


def register(kind: str, runner: Runner) -> None:
    _RUNNERS[kind] = runner


@contextmanager
def owner(tag: str, *, downloads: bool = True, prefix: str = "") -> Iterator[None]:
    """Jobs added by this thread inside the block belong to `tag` (and carry `prefix` before their label), so the
    work that queued them can tell them from the person's and stop only its own. With downloads=False they may
    not fetch models. Nothing changes for a job added outside the block."""
    if not isinstance(tag, str) or not tag or not re.fullmatch(r"[A-Za-z0-9_:.-]{1,80}", tag):
        raise ValueError("An owner tag is 1-80 letters, digits or _ : . -")
    before = getattr(_CONTEXT, "owner", None)
    _CONTEXT.owner = (tag, bool(downloads), prefix)
    try:
        yield
    finally:
        _CONTEXT.owner = before


def add(kind: str, label: str, params: dict, files: Optional[Dict[str, object]] = None) -> Job:
    """Dodaje zadanie. `files` - parametry z plikami (sciezka albo lista sciezek): kopiujemy
    je do queue_files/<id>/, bo pliki tymczasowe przegladarki znikaja po restarcie."""
    refusal = _prompt_refusal(params)
    if refusal:
        raise ValueError(refusal)
    tag, downloads, prefix = getattr(_CONTEXT, "owner", None) or ("", True, "")
    with _COND:
        job = Job(id=_STATE_INFO["next_id"], kind=kind, label=f"{prefix}{label}", params=dict(params), owner=tag,
                  downloads=downloads)
        _STATE_INFO["next_id"] += 1
        folder = FILES_DIR / str(job.id)
        try:
            for key, value in (files or {}).items():
                if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_]+", key):
                    raise ValueError("Invalid attachment field name.")
                paths = value if isinstance(value, list) else [value]
                copied = []
                for i, src in enumerate(p for p in paths if p):
                    folder.mkdir(parents=True, exist_ok=True)
                    dst = folder / f"{key}_{i}{Path(src).suffix.lower() or '.png'}"
                    shutil.copy2(src, dst)
                    copied.append(str(dst))
                job.params[key] = copied if isinstance(value, list) else (copied[0] if copied else None)
        except BaseException:  # a job that is not queued leaves no half-copied attachments behind
            shutil.rmtree(folder, ignore_errors=True)
            raise
        _JOBS.append(job)
        _save()
        _COND.notify_all()
    return job


def position(job_id: int) -> int:
    """Ile zadan jest przed danym (lacznie z biezacym)."""
    with _COND:
        ahead = [j for j in _JOBS if j.status in (PENDING, RUNNING) and j.id < job_id]
        return len(ahead)


def remove(job_ids: List[int]) -> None:
    """Usuwa oczekujace zadania; zaznaczone biezace - przerywa."""
    ids = set(job_ids)
    with _COND:
        if any(j.id in ids and j.status == RUNNING for j in _JOBS):
            cancel.request()
        for j in [j for j in _JOBS if j.id in ids and j.status == PENDING]:
            # A caller may still hold the Job object while waiting for its result.
            j.status, j.error, j.finished = STOPPED, "Removed from queue.", time.time()
            _JOBS.remove(j)
            shutil.rmtree(FILES_DIR / str(j.id), ignore_errors=True)
        _save()


def stop_current(job_id: Optional[int] = None) -> bool:
    """Przerywa biezace zadanie - przy job_id tylko gdy to wlasnie ono trwa (spoznione
    klikniecie nie zatrzyma kolejnego zadania). Zwraca False, gdy nic nie przerwano."""
    with _COND:
        run = next((j for j in _JOBS if j.status == RUNNING), None)
        if run is None or (job_id is not None and run.id != job_id):
            return False
        # inside the lock: the worker resets the flag for the next job under the same lock, so a Stop that comes
        # as this job ends cannot slip past that reset and stop the next job instead
        cancel.request()
    return True


def set_paused(paused: bool) -> None:
    with _COND:
        _STATE_INFO["paused"] = paused
        _save()
        _COND.notify_all()


def pause_now() -> Optional[int]:
    """Pauza "od razu" (przycisk "Pause and release GPU" na gornym pasku, np. na czas gry): kolejka staje, biezace
    zadanie jest przerywane i wraca na swoje miejsce (po wznowieniu liczy sie od nowa),
    a silnik oddaje pamiec karty. Nic nie dzieje sie samo - tylko na klikniecie.
    Zwraca id przerwanego zadania albo None."""
    set_paused(True)
    with _COND:
        run = next((j for j in _JOBS if j.status == RUNNING), None)
    if run is not None:
        _REQUEUE.set()
        cancel.request()
    threading.Thread(target=_free_gpu, name="pauza-vram", daemon=True).start()
    return run.id if run else None


def _free_gpu(timeout: float = 60.0) -> None:
    """Po pauzie: oddaj pamiec karty, gdy biezace zadanie skonczy sprzatanie."""
    deadline = time.time() + timeout
    while time.time() < deadline and any(j.status == RUNNING for j in snapshot()):
        time.sleep(0.5)
    if not any(j.status == RUNNING for j in snapshot()):
        from . import engine_runtime, llm

        llm.release_gpu()
        engine_runtime.release()


def paused() -> bool:
    return bool(_STATE_INFO["paused"])


def in_worker() -> bool:
    """True only in the thread currently executing a job's product code."""
    return getattr(_CONTEXT, "active", False)


def clear_finished() -> None:
    with _COND:
        _JOBS[:] = [j for j in _JOBS if j.status in (PENDING, RUNNING)]
        _save()


def snapshot() -> List[Job]:
    with _COND:
        return list(_JOBS)


def latest_finished(kind: str) -> Optional[Job]:
    done = [j for j in snapshot() if j.kind == kind and j.status in (DONE, ERROR, STOPPED)]
    return max(done, key=lambda j: j.finished or 0) if done else None


def remaining_seconds() -> float:
    """Szacowany czas do konca calej kolejki."""
    total = 0.0
    for j in snapshot():
        if j.status == RUNNING:
            total += j.eta() or 0.0
        elif j.status == PENDING:
            total += j.estimate()
    return total


def fmt_time(secs: Optional[float]) -> str:
    if secs is None:
        return "—"
    secs = int(secs)
    if secs >= 3600:
        return f"{secs // 3600} h {secs % 3600 // 60:02d} min"
    return f"{round(secs / 60)} min" if secs >= 90 else f"{secs} s"


# ----------------------------- watek roboczy -----------------------------

MIN_LEARNED_SECONDS = 1.0  # a job that ends sooner did no GPU work worth learning a speed from


def reused() -> None:
    """Called by a running job that found its result ready (an export or a sound preview that already
    exists): the job did no work, so the queue does not learn its speed from it."""
    _CONTEXT.reused = True


def _next() -> Optional[Job]:
    if _STATE_INFO["paused"]:
        return None
    return min((job for job in _JOBS if job.status == PENDING), key=lambda job: job.id, default=None)


def _release_when_idle() -> None:
    """After a job with nothing waiting (or the queue paused), give back the models the engine keeps warm
    (engine_runtime.warm_cache_gib): they speed up the next job in line, and an idle studio holds no
    model memory."""
    with _COND:
        waiting = _next() is not None
    if not waiting:
        from . import engine_runtime

        engine_runtime.release()


STAGES_DIR = config.HISTORY_DIR / "stages"  # <job id>.tsv: seconds since start, progress, step text


def _stage_log(job: Job) -> Callable[[float, str], None]:
    """Appends every new step text of a running job with its time, so the time of each stage of a job can be
    read afterwards. A file that cannot be written only loses the log, never the job."""
    started = time.time()
    path = STAGES_DIR / f"{job.id}.tsv"
    try:  # a job requeued by a pause starts again: its new attempt is marked, not mixed into the first one
        if path.exists():
            with path.open("a", encoding="utf-8") as log:
                log.write("# new attempt\n")
    except OSError:
        pass

    def write(progress: float, text: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as log:
                log.write(f"{time.time() - started:.1f}\t{progress:.3f}\t{text}\n")
        except OSError:
            pass
    return write


def _run(job: Job) -> None:
    stage = _stage_log(job)

    def cb(frac: float, desc: str = "") -> None:
        cancel.check()
        job.progress = min(max(frac, 0.0), 1.0)
        if desc and desc != job.desc:
            stage(job.progress, desc)
        if desc:
            job.desc = desc

    runner = _RUNNERS.get(job.kind)
    history.job_context(job.params)
    _CONTEXT.reused = False
    try:
        refusal = _prompt_refusal(job.params)
        if refusal:
            raise ValueError(refusal)
        if runner is None:
            raise RuntimeError(REMOVED_MSG if job.kind in REMOVED_KINDS else f"Unknown job type: {job.kind}")
        with (nullcontext() if job.downloads else no_downloads()):
            ok, files, notes, error = runner(job.params, cb)
    except cancel.Cancelled:
        ok, files, notes, error = False, [], [], ""
    except Exception as e:  # noqa: BLE001 - blad jednego zadania nie zatrzymuje kolejki
        ok, files, notes, error = False, [], [], str(e)
    finally:
        history.job_context(None)
        reuse, _CONTEXT.reused = getattr(_CONTEXT, "reused", False), False
    with _COND:
        job.finished = time.time()
        job.files, job.notes = files, notes
        # pauza "od razu" (pause_now): niedokonczone zadanie wraca na swoje miejsce; gotowe
        # w chwili pauzy zostaje gotowe
        requeue = cancel.requested() and _REQUEUE.is_set() and not ok
        if requeue:
            job.status, job.progress, job.desc, job.error = PENDING, 0.0, "", ""
            job.started = job.finished = None
            job.files, job.notes = [], []
        elif ok:  # checked before Stop: a job that finished as Stop was clicked is done (its files exist), not stopped
            job.status, job.progress = DONE, 1.0
            # srednia ruchoma czasu na jednostke pracy - dokladniejsze ETA kolejki; a job with
            # composed frames runs another pipeline, so it does not change the base rate, and a job
            # that only found its result ready (reused) or ended at once did no measurable work
            if not job.cast_extra() and not reuse and job.elapsed() >= MIN_LEARNED_SECONDS:
                key = job.kind
                floor = 0.1 if job.kind == "timeline_produce" else 1.0  # an export's units are clip seconds
                _RATE[key] = (0.7 * _RATE.get(key, job.elapsed())
                              + 0.3 * job.elapsed() / max(job.units(), floor))
        elif cancel.requested() and not _REQUEUE.is_set():
            job.status, job.error = STOPPED, "Stopped."
        else:
            job.status, job.error = ERROR, error or "Unknown error."
        _REQUEUE.clear()
        if not requeue:  # zadanie wracajace do kolejki potrzebuje swoich zdjec
            shutil.rmtree(FILES_DIR / str(job.id), ignore_errors=True)
        finished = [j for j in _JOBS if j.status in (DONE, ERROR, STOPPED)]
        for old in finished[:-_KEEP_FINISHED]:
            _JOBS.remove(old)
        _save()


def _process(job: Job) -> None:
    """Run one job the worker has claimed, then give the warm models back when nothing waits."""
    from . import llm  # pomocnik promptu na karcie oddaje VRAM przed generowaniem

    llm.release_gpu()
    _CONTEXT.active = True
    try:
        _run(job)
    finally:
        _CONTEXT.active = False
    _release_when_idle()


def _worker() -> None:
    while True:
        job = None
        try:
            with _COND:
                job = _next()
                while job is None:
                    _COND.wait(timeout=2.0)
                    job = _next()
                job.status, job.started, job.progress, job.desc = RUNNING, time.time(), 0.0, ""
                cancel.reset()
                _REQUEUE.clear()
                _save()
            _process(job)
        except BaseException as e:  # noqa: BLE001 - watek kolejki nie moze umrzec
            print(f"[queue] Worker error: {e!r}", flush=True)
            with _COND:
                if job is not None and job.status == RUNNING:
                    job.status, job.error, job.finished = ERROR, f"Queue error: {e}", time.time()
                    _save()
            time.sleep(1.0)


def start() -> None:
    """Wczytuje zapisana kolejke i uruchamia watek roboczy (raz, przy starcie aplikacji)."""
    global _started
    if _started:
        return
    _started = True
    with _COND:
        _load()
    threading.Thread(target=_worker, name="kolejka", daemon=True).start()
