# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Agent page's live view of a run: its state, what it has used of its limits, what it is doing, and the buttons that
steer it. A run lives on its own thread (app/agent_runner.py), so this only reads its folder: a reloaded page shows the
same run, and the timer runs only while a run works. Every word that came from a model, a file or the owner goes through
esc() before it reaches the page."""
from __future__ import annotations

import time

import gradio as gr

from . import agent_delivery, agent_runner, agent_store
from .ui_theme import badge, cache_media, esc, kpis, note, table

STATE_BADGE = {"created": ("Ready", "info"), "running": ("Working", "info"), "waiting": ("Waiting", "info"),
               "paused": ("Paused", "warn"), "interrupted": ("Interrupted", "warn"),
               "budget_stopped": ("Needs your approval", "warn"), "stopped": ("Stopped", "info"),
               "failed": ("Failed", "err"), "completed": ("Finished", "ok")}
WAIT_TEXT = {"gpu_job": "Waiting for the GPU to finish a job", "rate_limit": "Waiting: the API asked for a pause",
             "api_outage": "Waiting: the API is not answering, trying again", "queue_paused": "The GPU queue is paused"}
INTERRUPT_TEXT = {"app_restart": "Contsy was restarted while the run worked.", "api_outage": "The API did not answer for "
                  "a long time.", "aborted": "The connection was cut."}
BUDGET_TEXT = {"cost": "the cost limit"}
EVENT_TYPES = ("status", "tool_done", "question", "state", "api_error", "fresh_conversation", "finish", "resume")
SHOWN_EVENTS = 14
LIVE = ("running", "waiting")


def _when(stamp) -> str:
    try:
        return time.strftime("%d %b %H:%M", time.localtime(float(stamp)))
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def run_choices() -> list[tuple[str, str]]:
    return [(f"{_when(run.get('created'))} · {run.get('title') or 'Untitled'} · "
             f"{STATE_BADGE.get(run.get('state'), (str(run.get('state')), ''))[0]}", run["id"])
            for run in agent_store.list_runs()[:25]]


def _event_row(event: dict) -> list[str] | None:
    kind = event.get("type")
    if kind == "tool_done":
        return [esc(_when(event.get("t"))), badge("ok" if event.get("ok") else str(event.get("code") or "error"),
                                                   "ok" if event.get("ok") else "err"), esc(event.get("tool", ""))]
    if kind == "status":
        return [esc(_when(event.get("t"))), badge(str(event.get("phase", "")), "info"), esc(event.get("text", ""))]
    if kind == "question":
        return [esc(_when(event.get("t"))), badge("Question", "warn"), esc(event.get("text", ""))]
    if kind == "api_error":
        return [esc(_when(event.get("t"))), badge("API", "warn"), esc(event.get("message", ""))]
    if kind == "state":
        return [esc(_when(event.get("t"))), badge("State", "info"), esc(event.get("state", ""))]
    if kind in ("fresh_conversation", "resume", "finish"):
        return [esc(_when(event.get("t"))), badge(kind.replace("_", " ").capitalize(), "info"),
                esc(event.get("reason") or event.get("outcome") or "")]
    return None


def _num(value, default: float = 0.0) -> float:
    """A number from a record that may hold anything (the page must draw whatever is there)."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if number == number and abs(number) != float("inf") else default


def _hours(seconds) -> str:
    seconds = int(_num(seconds))
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60} s"
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d} h"


def run_html(view: dict) -> str:
    """The card of one run (empty page text when there is no run)."""
    if not view:
        return "<div class='empty'><b>No run selected</b>Start one above, or pick an earlier run.</div>"
    state = view.get("state", "")
    label, kind = STATE_BADGE.get(state, (state, ""))
    settings = view.get("settings") or {}
    budgets = settings.get("budgets") or {}
    usage = view.get("usage") or {}
    spent = _num(usage.get("cost_usd")) + _num(usage.get("reserved_usd"))
    gpu = _num((view.get("gpu") or {}).get("seconds"))
    limit = _num(budgets.get("cost_usd")) or 1.0
    head = (f"<p><b>{esc(view.get('title') or 'Untitled')}</b> {badge(label, kind)} "
            f"<span class='muted'>{esc(settings.get('mode', ''))} · {esc(settings.get('model', ''))}</span></p>")
    cards = kpis([
        {"label": "Cost (estimate)", "value": f"${spent:.2f}", "sub": f"of ${limit:.0f}", "progress": spent / limit},
        {"label": "Steps", "value": str(int(_num(view.get("steps"))))},
        {"label": "GPU time", "value": f"{gpu / 60:.0f} min"},
        {"label": "Working time", "value": _hours(view.get("elapsed_seconds"))}])
    status = view.get("status") or {}
    lines = []
    if status.get("text"):
        lines.append(f"<p>{badge(str(status.get('phase', '')), 'info')} {esc(status['text'])}</p>")
    if view.get("wait_reason"):
        lines.append(f"<p class='muted'>{esc(WAIT_TEXT.get(view['wait_reason'], view['wait_reason']))}.</p>")
    if state == "interrupted":
        lines.append(f"<p class='muted'>{esc(INTERRUPT_TEXT.get(view.get('interrupt_reason'), 'The run was interrupted.'))}"
                     " Resume it to continue where it stopped.</p>")
    if state == "budget_stopped":
        lines.append("<p class='muted'>The run reached "
                     f"{esc(BUDGET_TEXT.get(view.get('budget_reason'), 'a limit'))}. Raise the cost limit above and "
                     "press Resume to let it go on.</p>")
    if failure := view.get("failure"):
        lines.append(f"<p>{badge('Failed', 'err')} {esc(failure.get('message', ''))}</p>")
    for question in view.get("questions") or []:
        lines.append(f"<p>{badge('Question', 'warn')} {esc(question)}</p>")
    outcome = view.get("outcome") or {}
    if outcome.get("summary"):
        label = "Finished" if outcome.get("kind") == "complete" else "Finished with open items"
        lines.append(f"<p>{badge(label, 'ok' if outcome.get('kind') == 'complete' else 'warn')} "
                     f"{esc(outcome['summary'])}</p>")
    rows = [row for event in (view.get("events") or []) if event.get("type") in EVENT_TYPES
            if (row := _event_row(event))][-SHOWN_EVENTS:][::-1]
    log = table(["Time", "What", "Details"], rows, empty="<div class='muted'>Nothing yet.</div>")
    return head + cards + "".join(lines) + log


def buttons(view: dict) -> tuple:
    """Updates for Pause, Stop, Resume and Approve and produce."""
    state = (view or {}).get("state")
    settings = (view or {}).get("settings") or {}
    working = state in LIVE
    can_resume = state in agent_store.RESUMABLE and state != "created"
    can_approve = state == "completed" and settings.get("mode") == "prepare" and \
        ((view or {}).get("outcome") or {}).get("kind") in ("complete", "open_items")
    return (gr.update(interactive=working), gr.update(interactive=working), gr.update(interactive=can_resume),
            gr.update(interactive=can_approve))


def view_signature(view: dict) -> str:
    events = view.get("events") or [{}]
    return f"{view.get('state')}|{view.get('updated')}|{events[-1].get('seq')}"


def downloads(run_id: str):
    """An update for the downloads: the run's files (the ZIP first) in Gradio's file cache, hidden when there are none."""
    if not run_id or not agent_store.valid_id(run_id):
        return gr.update(value=None, visible=False)
    paths = [path for path in (cache_media(found) for found in agent_delivery.downloads(agent_store.RunStore(run_id)))
             if path]
    return gr.update(value=paths or None, visible=bool(paths))


def show(run_id: str, *, force: bool = False, signature: str = "") -> tuple:
    """(run card, Pause, Stop, Resume, Approve, downloads, timer, signature) for the page; everything but the timer is
    skipped when nothing changed since `signature`."""
    view = agent_runner.view(run_id) if run_id else {}
    new_signature = view_signature(view) if view else ""
    working = bool(view.get("working")) if view else False
    skip = gr.skip()
    if not force and new_signature == signature:
        return skip, skip, skip, skip, skip, skip, gr.Timer(active=working), skip
    return (run_html(view), *buttons(view), downloads(run_id), gr.Timer(active=working), new_signature)


# --- the buttons ---------------------------------------------------------------------------------------------------

def start_click(paths: list, extra: str, settings: dict) -> tuple:
    """Start a run; returns (message, the new run's id or a skip)."""
    try:
        run_id = agent_runner.start(paths, extra, settings)
    except agent_runner.RunnerError as error:
        return note(f"❌ {esc(error)}"), gr.skip()
    except Exception as error:  # noqa: BLE001 - the page must answer
        return note(f"❌ The run could not start ({esc(type(error).__name__)})."), gr.skip()
    return note("✅ The agent has started. Watch it below; you can close this page, the run goes on."), run_id


def pause_click(run_id: str) -> str:
    return note("✅ Pausing at the next step." if agent_runner.pause(run_id) else "⚠ This run is not working.")


def stop_click(run_id: str) -> str:
    return note("✅ Stopping. Its waiting GPU jobs are withdrawn." if agent_runner.stop(run_id)
                else "⚠ This run is not working.")


def resume_click(run_id: str, budgets: dict | None = None) -> str:
    try:
        agent_runner.resume(run_id, budgets=budgets)
    except agent_runner.RunnerError as error:
        return note(f"❌ {esc(error)}")
    return note("✅ Resumed from where it stopped.")


def approve_click(run_id: str) -> str:
    try:
        agent_runner.resume(run_id, approve_production=True)
    except agent_runner.RunnerError as error:
        return note(f"❌ {esc(error)}")
    return note("✅ Approved. The same run goes on to produce; this uses the GPU.")
