# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Starting, pausing, stopping and resuming the agent's run, and the one thread it works on.

The run belongs to the app, not to a browser tab: it works on a module-level daemon thread, so closing or reloading the
page changes nothing, and the page only reads the run's folder. One run works at a time. Everything else (the state of
a run, its record) is on disk in app/agent_store.py, so a run the app lost track of (a restart, a crash) is found there
as `interrupted` and can be resumed."""
from __future__ import annotations

import secrets
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from . import agent_brief, agent_config, agent_eyes, agent_preflight, agent_prompt, agent_provider, agent_store, agent_tools
from . import timeline
from .agent_host import Host
from .agent_loop import Loop, RealClock, count_pictures

MIN_DISK_GB = 5.0
JOIN_SECONDS = 20.0
APPROVAL = ("The owner has approved production. The rule in your instructions that this run only prepares no longer "
            "applies: from now on you may make the clips, edit scenes, export the film and use every tool. The "
            "preparation above stands. Continue from the Timeline project: read its state first (get_project), make "
            "what is missing, check the result, then package and finish as before. The limits of this run now "
            "include the GPU time for production.")


class RunnerError(ValueError):
    """The run cannot start or continue; the message says why, in plain words."""


@dataclass
class _Active:
    run_id: str
    thread: threading.Thread
    stop: threading.Event
    pause: threading.Event


_LOCK = threading.RLock()
_ACTIVE: Optional[_Active] = None


def make_provider(name: str):
    """The provider of a run (the tests replace this with a scripted one)."""
    return agent_provider.get_provider(name)


def make_clock():
    return RealClock()


def active() -> Optional[str]:
    """The id of the run that is working now, or None."""
    with _LOCK:
        return _ACTIVE.run_id if _ACTIVE and _ACTIVE.thread.is_alive() else None


def busy() -> bool:
    return active() is not None


def startup() -> list[str]:
    """At the start of the app: runs the previous process left working are marked interrupted."""
    return agent_store.startup_sweep()


def shutdown(timeout: float = JOIN_SECONDS) -> None:
    """Before the app exits: the working run is paused (it can be resumed), and its thread given a moment to finish."""
    with _LOCK:
        current = _ACTIVE
    if current is None or not current.thread.is_alive():
        return
    current.pause.set()
    current.thread.join(timeout)


def pause(run_id: str) -> bool:
    with _LOCK:
        if _ACTIVE and _ACTIVE.run_id == run_id and _ACTIVE.thread.is_alive():
            _ACTIVE.pause.set()
            return True
    return False


def stop(run_id: str) -> bool:
    with _LOCK:
        if _ACTIVE and _ACTIVE.run_id == run_id and _ACTIVE.thread.is_alive():
            _ACTIVE.stop.set()
            return True
    return False


def wait(run_id: str, timeout: float = 30.0) -> None:
    """Block until the run's thread ends (tests and shutdown)."""
    with _LOCK:
        current = _ACTIVE if _ACTIVE and _ACTIVE.run_id == run_id else None
    if current:
        current.thread.join(timeout)


# --- the thread ---------------------------------------------------------------------------------------------------

def _keep_awake(on: bool) -> None:
    """Windows may not put the computer to sleep while the run works (a flag of this thread)."""
    try:
        import ctypes

        flags = 0x80000000 | (0x00000001 if on else 0)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except (AttributeError, OSError):
        pass


def _work(store: agent_store.RunStore, settings: dict, nonce: str, stop_flag: threading.Event,
          pause_flag: threading.Event) -> None:
    global _ACTIVE
    awake = bool(settings.get("keep_awake"))
    try:
        if awake:
            _keep_awake(True)
        clock = make_clock()
        def interrupted() -> bool:  # a Stop or a Pause (or the app closing) ends a wait inside a tool too
            return stop_flag.is_set() or pause_flag.is_set()

        host = Host(store, settings, nonce=nonce, clock=clock.time, sleep=lambda seconds: clock.sleep(seconds, interrupted))
        host.stop_requested = interrupted
        Loop(store, host, make_provider(settings["provider"]), clock=clock, stop=stop_flag, pause=pause_flag).run()
    except BaseException as error:  # noqa: BLE001 - whatever happened, the run must not stay "running"
        store.log("crash", error=type(error).__name__, detail=str(error)[:300])
        store.update(state="failed", wait_reason="", failure={"kind": "internal", "message": "The agent could not "
                                                              f"start ({type(error).__name__}). Resume it to try again."})
    finally:
        if awake:
            _keep_awake(False)
        with _LOCK:
            if _ACTIVE and _ACTIVE.run_id == store.id:
                _ACTIVE = None


def _launch(store: agent_store.RunStore, settings: dict, nonce: str) -> None:
    global _ACTIVE
    stop_flag, pause_flag = threading.Event(), threading.Event()
    thread = threading.Thread(target=_work, args=(store, settings, nonce, stop_flag, pause_flag),
                              name=f"agent-{store.id}", daemon=True)
    _ACTIVE = _Active(store.id, thread, stop_flag, pause_flag)
    thread.start()


# --- starting a run -----------------------------------------------------------------------------------------------

def _title(attachments: Sequence, extra: str) -> str:
    for item in attachments:
        if item.kind != "image":
            return Path(item.name).stem[:80]
    return (extra.strip().splitlines() or ["Untitled job"])[0][:80]


def start(files: Sequence, extra: str, settings: dict, *, extra_limit: int = agent_brief.MAX_EXTRA,
          series_binding: Optional[dict] = None, title: str = "") -> str:
    """Create a run from the brief and begin it. `files`: paths or (path, name) pairs; `settings`: the page's settings
    for this run (permissions and consent included). Returns the run's id; RunnerError says why a run cannot start."""
    settings = agent_config.clean_settings(settings)
    extra = (extra or "").strip()
    with _LOCK:
        if busy():
            raise RunnerError("A run is already working. Stop it first, or wait until it ends.")
        key, _ = agent_config.get_key()
        if not key:
            raise RunnerError("Add your Claude API key first (Agent page → Connection).")
        attachments, rejected = agent_brief.inspect_many(files, settings["send_images"])
        if rejected:
            raise RunnerError("; ".join(f"{name}: {reason}" for name, reason in rejected))
        if not attachments and not extra:
            raise RunnerError("Add a brief file, or write what the agent should make.")
        if problem := agent_brief.extra_problem(extra, extra_limit):
            raise RunnerError(problem)
        if problem := agent_preflight.blocking_models(settings["mode"], agent_config.allowed(settings, "downloads")):
            raise RunnerError(problem)
        if agent_store.free_gb() < MIN_DISK_GB:
            raise RunnerError("There is less than 5 GB of free disk space. Free some before a run.")
        sources = [Path(str(item[0] if isinstance(item, tuple) else item)) for item in files or ()]
        nonce = secrets.token_hex(6)
        store = agent_store.RunStore(agent_store.new_run_id())
        try:
            _create(store, attachments, extra, settings, nonce, sources)
            if series_binding:  # before the thread starts: its very first tool call may need the episode
                store.update(series=dict(series_binding), **({"title": title[:80]} if title else {}))
        except BaseException:
            shutil.rmtree(store.dir, ignore_errors=True)  # a run that never began leaves no folder behind
            raise
        _launch(store, settings, nonce)
        return store.id


def _create(store: agent_store.RunStore, attachments: Sequence, extra: str, settings: dict, nonce: str,
            sources: Sequence[Path]) -> None:
    """The folder of a new run: its record, its frozen instructions and tools, the brief files and the first message."""
    store.create(summary={
        "title": _title(attachments, extra), "settings": settings, "brief": agent_brief.summary(attachments),
        "extra": extra, "nonce": nonce, "usage": {"cost_usd": 0.0, "turns": 0}, "steps": 0, "elapsed_seconds": 0.0,
        "status": {"phase": "intake", "text": "Reading the brief"}},
        system=agent_prompt.system_prompt(settings), tools=agent_tools.definitions())

    def read_bytes(item) -> bytes:
        if item.kind == "image":  # the model gets a JPEG of at most 1568 px, as for every picture it is shown
            with Image.open(Path(timeline.reference_path(item.ref))) as picture:
                return agent_eyes.encode(picture)
        return store.path("brief", f"{item.sha256}.{item.suffix}").read_bytes()

    for item, source in zip(attachments, sources):  # no file was refused, so they come in the order they were given
        if item.kind != "image":
            store.put_brief(source.read_bytes(), item.suffix)
    first = agent_brief.first_message(attachments, read_bytes, nonce=nonce, extra=extra,
                                      send_images=settings["send_images"])
    store.append_message("user", first)
    store.update(pictures=count_pictures(first))


# --- continuing a run ---------------------------------------------------------------------------------------------

def resume(run_id: str, *, approve_production: bool = False, budgets: Optional[dict] = None) -> str:
    """Continue a paused, stopped, interrupted, failed or budget-stopped run from its transcript. With
    approve_production a finished run that only prepared goes on to produce; `budgets` replaces the run's limits."""
    with _LOCK:
        if busy():
            raise RunnerError("A run is already working. Stop it first, or wait until it ends.")
        if not agent_store.valid_id(run_id):
            raise RunnerError("This run does not exist.")
        store = agent_store.RunStore(run_id)
        record = store.read()
        if not record:
            raise RunnerError("This run does not exist.")
        state = record.get("state")
        settings = agent_config.clean_settings(record.get("settings"))
        if approve_production:
            if state != "completed" or settings["mode"] != "prepare":
                raise RunnerError("Only a finished run that prepared the job can be approved for production.")
        elif state not in agent_store.RESUMABLE:
            raise RunnerError("This run cannot be resumed: it " + ("is working." if state in agent_store.LIVE_STATES
                                                                    else "has finished."))
        key, _ = agent_config.get_key()
        if not key:
            raise RunnerError("Add your Claude API key first (Agent page → Connection).")
        if budgets is not None:
            settings["budgets"] = agent_config.clean_budgets({**settings["budgets"], **budgets})
        if approve_production:
            settings["mode"] = "produce"
            if problem := agent_preflight.blocking_models("produce", agent_config.allowed(settings, "downloads")):
                raise RunnerError(problem)
        if agent_store.free_gb() < MIN_DISK_GB:
            raise RunnerError("There is less than 5 GB of free disk space. Free some before a run.")
        store.update(settings=settings, failure=None, interrupt_reason="", budget_reason="",
                     **({"outcome": None} if approve_production else {}))
        if state == "budget_stopped":
            budget = settings["budgets"]
            store.append_message("user", [{"type": "text", "text": (
                "The owner resumed this run after it reached its cost limit. The limit now is "
                f"${budget['cost_usd']:g} of API cost. The earlier instruction to call finish_run at once no longer "
                "applies: carry on with the work, and finish when it is done.")}], meta={"note": True})
        if approve_production:
            store.append_message("user", [{"type": "text", "text": APPROVAL}], meta={"approval": True})
        store.log("resume", approve=approve_production)
        _launch(store, settings, record.get("nonce") or secrets.token_hex(6))
        return run_id


def view(run_id: Optional[str] = None) -> dict:
    """What the page shows of a run (the working one by default): its record and its latest events."""
    run_id = run_id or active()
    if not run_id or not agent_store.valid_id(run_id):
        return {}
    store = agent_store.RunStore(run_id)
    view_ = store.view()
    view_["working"] = active() == run_id
    return view_


# --- Contsy Stories: episode runs and the conversation after them -------------------------------------------------

FOLLOW_UP_USD = 2.0  # the API cost one follow-up request of the owner may add on top of what the run has spent
FOLLOW_UP = ("The owner has a follow-up request about this episode (it may be in any language):\n{text}\n\n"
             "Work on the episode as it is now: read the project first, change only what the request needs (never "
             "remake the whole film), re-export only if a clip changed, update the episode memory with "
             "save_episode_memory if the story changed, then call finish_run saying what you changed.")


def _episode_settings(produce: bool) -> dict:
    settings = agent_config.load_settings()
    settings["mode"] = "produce" if produce else "prepare"
    return agent_config.clean_settings(settings)


def start_episode(series_id: str, episode_id: str, *, produce: bool = False) -> str:
    """Start the agent run that writes (and with produce, also makes) an episode of a series: its brief is the episode
    request with the series bible and the story so far (series.episode_brief). The run is bound to the episode, so
    its tools keep the series style and cast and it can save the episode's memory."""
    from . import series
    episode = series.get_episode(series_id, episode_id)
    if episode is None:
        raise RunnerError("This episode does not exist.")
    if episode["run"] and agent_store.valid_id(episode["run"]):
        state = (agent_store.RunStore(episode["run"]).read() or {}).get("state")
        if state in agent_store.LIVE_STATES:
            raise RunnerError("The agent is already working on this episode.")
    brief = series.episode_brief(series_id, episode_id)
    limit = agent_brief.MAX_SERIES_BRIEF
    if len(brief) > limit:  # the bible is cut at series.CONTEXT_CHARS already; get_series_bible returns it again
        brief = brief[:limit - 80] + "\n(continued: call get_series_bible for the whole bible)"
    run_id = start([], brief, _episode_settings(produce), extra_limit=limit,
                   series_binding={"series": series_id, "episode": episode_id},
                   title=f"{episode['title']} — {series.get(series_id)['name']}")
    series.update_episode(series_id, episode_id, run=run_id, status="producing" if produce else "writing")
    series.add_chat(series_id, episode_id, "owner", episode["idea"])
    return run_id


def _answer_open_calls(store: agent_store.RunStore) -> None:
    """A run stopped hard (the app closed while a tool ran) ends on tool calls nobody answered; the API refuses a
    conversation that goes on after them, so they get an INTERRUPTED answer first (as a resumed run's do)."""
    records = store.records()
    if not records or records[-1]["role"] != "assistant":
        return
    calls = [block for block in records[-1]["content"] if isinstance(block, dict) and block.get("type") == "tool_use"]
    if calls:
        store.append_message("user", [{"type": "tool_result", "tool_use_id": block["id"], "is_error": True, "content": [
            {"type": "text", "text": "INTERRUPTED: the studio stopped while this call ran; it may or may not have taken "
                                     "effect. Check the state before repeating it."}]} for block in calls],
                             meta={"repaired": True})


def follow_up(run_id: str, text: str) -> str:
    """Continue the conversation of a run that has ended (completed, failed, stopped...): the owner's new request is
    the next message of the same transcript, so the agent remembers the whole episode. Returns the run id."""
    text = (text or "").strip()
    if not text:
        raise RunnerError("Write what should change.")
    if problem := agent_brief.extra_problem(text):
        raise RunnerError(problem)
    with _LOCK:
        if busy():
            raise RunnerError("A run is already working. Stop it first, or wait until it ends.")
        if not agent_store.valid_id(run_id):
            raise RunnerError("This run does not exist.")
        store = agent_store.RunStore(run_id)
        record = store.read()
        if not record:
            raise RunnerError("This run does not exist.")
        if record.get("state") in agent_store.LIVE_STATES:
            raise RunnerError("The agent is still working on this. Wait until it ends, or stop it.")
        key, _ = agent_config.get_key()
        if not key:
            raise RunnerError("The agent is not connected (no Claude API key).")
        settings = agent_config.clean_settings(record.get("settings"))
        spent = float((record.get("usage") or {}).get("cost_usd") or 0.0)
        allowance = FOLLOW_UP_USD
        settings["budgets"] = agent_config.clean_budgets({**settings["budgets"], "cost_usd": round(spent + allowance,
                                                                                                    2)})
        if agent_store.free_gb() < MIN_DISK_GB:
            raise RunnerError("There is less than 5 GB of free disk space. Free some before a run.")
        store.update(settings=settings, failure=None, interrupt_reason="", budget_reason="", outcome=None)
        _answer_open_calls(store)
        store.append_message("user", [{"type": "text", "text": FOLLOW_UP.format(text=text)}], meta={"follow_up": True})
        store.log("follow_up")
        _launch(store, settings, record.get("nonce") or secrets.token_hex(6))
    bound = record.get("series") if isinstance(record.get("series"), dict) else None
    if bound:
        from . import series
        try:
            series.add_chat(bound["series"], bound["episode"], "owner", text)
        except series.SeriesError:
            pass
    return run_id
