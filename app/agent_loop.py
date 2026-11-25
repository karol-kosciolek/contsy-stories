# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""One run of the agent: ask the model, run the tools it calls, keep the transcript, the cost and the clock, and end in
one of the run's states. No threads and no screen here (app/agent_runner.py starts it on a thread); everything the loop
needs from outside is handed in, so the tests drive it with a scripted provider and a clock that does not wait.

The transcript is the conversation exactly as the API must receive it again, so a run that is stopped, interrupted or
paused continues from the very same bytes (thinking blocks included)."""
from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional

from . import agent_config, agent_delivery, agent_provider as provider, agent_tools, queue_manager
from .agent_host import Host, ToolError
from .agent_provider import Aborted, ProviderError, Turn, TurnRequest
from .agent_store import LeakError, RunStore

BACKOFF = (5, 10, 20, 40, 80, 160, 300)   # seconds between tries of a failing request; the last repeats
MAX_OUTAGE = 90 * 60                      # s of failing requests before the run is left interrupted
MAX_UNKNOWN = 3                           # tries of an error nobody has classified
NUDGES = 2                                # reminders to call finish_run before the run ends "with open items"
FINAL_TURNS = 2                           # turns left to call finish_run once a limit is reached
STUCK_WARN, STUCK_STOP = 3, 6             # the same call in a row: a warning, then the end
WARN_AT = 0.9
RESERVE_SHARE = 0.1                       # of the cost limit: the most that cut-off requests can set aside in all
SUMMARY_CHARS = 6000
COMPACT_INSTRUCTIONS = (
    "Keep: the brief and the owner's instructions, every id, slug, project id, job id and file name this run made, the "
    "decisions taken and why, what is done and what is left, and every problem still open. Drop: tool answers that "
    "were only read once.")
_DONE = ("done", "failed", "stopped")


class Ended(Exception):
    """The run leaves the loop in `state` (a state of agent_store.STATES)."""

    def __init__(self, state: str, **fields):
        super().__init__(state)
        self.state, self.fields = state, fields


def count_pictures(value) -> int:
    """How many image blocks are in a message (or any nesting of lists and dicts)."""
    if isinstance(value, dict):
        return 1 if value.get("type") == "image" else sum(count_pictures(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_pictures(item) for item in value)
    return 0


def _estimate_tokens(messages) -> int:
    """A rough size of a conversation in tokens: its text, with a flat allowance for each picture or document (their
    bytes are not text)."""
    def chars(value) -> int:
        if isinstance(value, str):
            return len(value)
        if isinstance(value, dict):
            return 6000 if value.get("type") in ("image", "document") else sum(chars(item) for item in value.values())
        if isinstance(value, list):
            return sum(chars(item) for item in value)
        return 0
    return chars(messages) // 4


class _Fresh(Exception):
    """The conversation does not fit: the loop starts a new one (fresh_conversation); this never leaves the loop."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class RealClock:
    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float, wake: Callable[[], bool] = lambda: False) -> None:
        end = time.monotonic() + seconds
        while not wake() and (left := end - time.monotonic()) > 0:
            time.sleep(min(0.5, left))


class Loop:
    def __init__(self, store: RunStore, host: Host, model: provider.Provider, *, clock=None,
                 stop: Optional[threading.Event] = None, pause: Optional[threading.Event] = None):
        self.store, self.host, self.model = store, host, model
        self.clock = clock or RealClock()
        self.stop = stop or threading.Event()
        self.pause = pause or threading.Event()
        self.final_only = False
        self.last_signature, self.same_calls = "", 0
        self._active_since = 0.0
        self._outage = 0.0
        self._unknown = 0
        self._notes: list[str] = []

    # --- the run ---------------------------------------------------------------------------------------------

    def run(self) -> str:
        """Work until the run ends or is left; returns its state."""
        self._active_since = self.clock.time()
        try:
            self.store.update(final_turns=0, nudges=0)  # a run resumed after a limit gets its turns again
            self._set_state("running", wait_reason="")
            self._repair()
            self._work()
        except Ended as ended:
            self._finish(ended.state, **ended.fields)
        except LeakError as error:
            self._finish("failed", failure={"kind": "leak", "message": str(error)})
        except Exception as error:  # noqa: BLE001 - the run must end in a state, whatever went wrong
            self.store.log("crash", error=type(error).__name__, detail=str(error)[:300])
            self._finish("failed", failure={"kind": "internal", "message": "The agent stopped because of an internal "
                                            f"error ({type(error).__name__}). Resume it to try again."})
        return self.store.read().get("state", "failed")

    def _work(self) -> None:
        while True:
            self._flags()
            self._limits()
            turn = self._ask()  # a response that arrived is kept, then the flags are looked at
            if self._handle(turn):
                raise Ended("completed")

    # --- state and the clock -----------------------------------------------------------------------------------

    def _tick(self) -> None:
        """Add the time since the last look to the run's working time."""
        now = self.clock.time()
        record = self.store.read()
        self.store.update(elapsed_seconds=float(record.get("elapsed_seconds") or 0.0) + max(now - self._active_since, 0))
        self._active_since = now

    def _set_state(self, state: str, **fields) -> None:
        self.store.update(state=state, **fields)
        self.store.log("state", state=state, **{key: value for key, value in fields.items()
                                                if key in ("wait_reason", "interrupt_reason")})

    def _finish(self, state: str, **fields) -> None:
        self._tick()
        if state == "stopped":
            for job in self.host.own_jobs():
                if job.status in (queue_manager.PENDING, queue_manager.RUNNING):
                    try:
                        self.host.stop_jobs([job.id])
                    except (ToolError, OSError) as error:  # the owner removed it first, or the queue file would not save
                        self.store.log("withdraw_failed", job=job.id, error=type(error).__name__)
        if state == "completed":
            try:
                agent_delivery.finalize(self.store)  # the owner never sees a finished run without its package
            except Exception as error:  # noqa: BLE001 - the run is over either way; the report says nothing is packed
                self.store.log("finalize_failed", error=type(error).__name__, detail=str(error)[:300])
        self.host.account_jobs()
        self._set_state(state, wait_reason="", **fields)

    def _flags(self) -> None:
        if self.stop.is_set():
            raise Ended("stopped")
        if self.pause.is_set():
            raise Ended("paused")

    def _wait(self, seconds: float, reason: str) -> None:
        """Wait (state waiting) for a back-off or a limit of the API; a stop or pause ends it at once."""
        self._set_state("waiting", wait_reason=reason)
        self.clock.sleep(seconds, lambda: self.stop.is_set() or self.pause.is_set())
        self._flags()
        self._set_state("running", wait_reason="")

    # --- limits --------------------------------------------------------------------------------------------------

    def _spent(self) -> float:
        usage = self.store.read().get("usage") or {}
        return float(usage.get("cost_usd") or 0.0) + float(usage.get("reserved_usd") or 0.0)

    def _used(self) -> float:
        """The share of the cost limit that is spent."""
        return self._spent() / self.host.settings["budgets"]["cost_usd"]

    def _limits(self) -> None:
        self._tick()
        used = self._used()
        record = self.store.read()
        if used >= 1.0:
            turns = int(record.get("final_turns") or 0)
            if not self.final_only:
                self.final_only = True
                self._say("You have reached the run's cost limit. Do nothing more except call finish_run now: say "
                          "honestly what is done and what is open.")
            if turns >= FINAL_TURNS:
                raise Ended("budget_stopped", budget_reason="cost")
            self.store.update(final_turns=turns + 1)
        elif used >= WARN_AT and not record.get("warned"):
            self.store.update(warned=True)
            self._notes.append(f"Notice: {used * 100:.0f}% of the cost limit of this run is spent. Plan to finish soon.")

    def _say(self, text: str) -> None:
        self.store.append_message("user", [{"type": "text", "text": text}], meta={"note": True})

    # --- asking the model ------------------------------------------------------------------------------------------

    def _request(self) -> TurnRequest:
        settings = self.host.settings
        boundary = int(self.store.read().get("boundary") or 0)
        return TurnRequest(
            model=settings["model"], system=self.store.system_text(), tools=self.store.tools(),
            messages=self.store.wire_messages()[boundary:], effort=settings["effort"],
            features=provider.Features(compaction=True, compact_instructions=COMPACT_INSTRUCTIONS))

    def _ask(self) -> Turn:
        """One response; a conversation that does not fit starts a fresh one (twice at most in a row)."""
        for attempt in range(3):
            try:
                return self._ask_once()
            except _Fresh as fresh:
                if attempt == 2:
                    raise Ended("failed", failure={"kind": "too_large", "message": "The conversation does not fit even "
                                                   "after starting a new one. The brief may be too large."})
                self.fresh_conversation(fresh.reason)
        raise AssertionError("unreachable")

    def _ask_once(self) -> Turn:
        """One response, with the retries and the ladder of fixes for what the API refuses."""
        started = self.clock.time()
        failures = 0
        while True:
            request = self._request()
            try:
                turn = self.model.complete(request, should_abort=lambda: self.stop.is_set() or self.pause.is_set(),
                                           on_event=lambda name: None)
            except Aborted:
                self._reserve(request, ProviderError("network", "aborted"))  # billed for what it had made so far
                self._flags()
                raise Ended("interrupted", interrupt_reason="aborted")
            except ProviderError as error:
                self._reserve(request, error)
                self._unknown = self._unknown + 1 if error.kind == "unknown" else self._unknown
                self._on_error(error, failures, started)
                failures += 1
                continue
            self._outage = 0.0
            self._unknown = 0
            if turn.stop_reason == "model_context_window_exceeded":
                self._charge(turn)
                raise _Fresh("context")
            return turn

    def _reserve(self, request: TurnRequest, error: ProviderError) -> None:
        """A request that was cut off reports no usage, so a part of what it could have cost is set aside."""
        if error.kind in ("network", "timeout"):
            usage = self.store.read().get("usage") or {}
            reserve = provider.cost([provider.UsageEntry(request.model, input_tokens=_estimate_tokens(request.messages),
                                                         output_tokens=2000)])
            cap = RESERVE_SHARE * self.host.settings["budgets"]["cost_usd"]
            self.store.update(usage={**usage, "reserved_usd": round(min(float(usage.get("reserved_usd") or 0) + reserve,
                                                                         cap), 4)})

    def _on_error(self, error: ProviderError, failures: int, started: float) -> None:
        self.store.log("api_error", kind=error.kind, status=error.status, request_id=error.request_id,
                       message=error.user_message)
        if error.kind in ("context", "too_large"):
            raise _Fresh(error.kind)
        if error.kind in ("auth", "billing", "model", "bad_request", "binding", "feature"):
            raise Ended("failed", failure={"kind": error.kind, "message": error.user_message})
        if error.kind == "unknown" and self._unknown > MAX_UNKNOWN:
            raise Ended("failed", failure={"kind": "unknown", "message": error.user_message})
        pause = error.retry_after if error.kind == "rate_limit" and error.retry_after else BACKOFF[min(failures,
                                                                                                         len(BACKOFF) - 1)]
        pause = min(max(pause, 1.0), 600.0)
        self._outage += pause
        if self._outage > MAX_OUTAGE:
            raise Ended("interrupted", interrupt_reason="api_outage")
        self._wait(pause, "rate_limit" if error.kind == "rate_limit" else "api_outage")

    # --- what the model said -----------------------------------------------------------------------------------------

    def _charge(self, turn: Turn, *, kept: bool = False) -> None:
        """Add what a response cost to the run's meter (kept: it is a turn of the conversation)."""
        usage = self.store.read().get("usage") or {}
        self.store.update(usage={
            **usage, "cost_usd": round(float(usage.get("cost_usd") or 0.0) + turn.cost(), 6),
            "input_tokens": int(usage.get("input_tokens") or 0) + sum(one.input_tokens for one in turn.usage),
            "output_tokens": int(usage.get("output_tokens") or 0) + sum(one.output_tokens for one in turn.usage),
            "turns": int(usage.get("turns") or 0) + (1 if kept else 0)})

    def _record(self, turn: Turn) -> int:
        self._charge(turn, kept=True)
        entries = [{"model": one.model, "in": one.input_tokens, "out": one.output_tokens} for one in turn.usage]
        meta = {"request_id": turn.request_id, "model": turn.model, "stop": turn.stop_reason, "usage": entries}
        if turn.served_by_fallback:
            meta["fallback"] = True
        return self.store.append_message("assistant", turn.content, meta=meta)

    def _handle(self, turn: Turn) -> bool:
        """Act on one response; True when the run is over."""
        self.store.log("turn", model=turn.model, stop=turn.stop_reason, request_id=turn.request_id,
                       cost=round(turn.cost(), 4), context=turn.context_tokens, text=turn.text[:300])
        if turn.stop_reason == "refusal":
            category = (turn.stop_details or {}).get("category", "")
            self._charge(turn)
            raise Ended("failed", failure={"kind": "refusal", "message": "The model declined to continue"
                                           + (f" ({category})" if category else "") + ". The owner can change the brief "
                                           "and start a new run."})
        cut = turn.stop_reason == "max_tokens"
        if not turn.content:  # an empty answer is not a message the API accepts back: it is not kept
            self._charge(turn)
            empties = int(self.store.read().get("empties") or 0) + 1
            self.store.update(empties=empties)
            if empties > NUDGES:
                return self._open_items("The agent stopped without finishing.")
            self._say("Your last answer was empty. Carry on with your next tool call, or call finish_run if the job is "
                      "done.")
            return False
        if cut and any(block.get("type") == "tool_use" for block in turn.content):
            self._charge(turn)  # the cut answer is not kept: its tool call is incomplete
            self._say("Your last answer was cut off before its tool call was complete, so nothing was run. Make smaller "
                      "calls (fewer scenes or less text at a time).")
            return False
        self._record(turn)
        if turn.tool_calls:
            self.store.update(nudges=0, empties=0)  # the model is working again: its reminders start over
            return self._tools(turn)
        if cut:
            self._say("Your last answer was cut off. Continue.")
            return False
        return self._ended_turn(turn)

    def _ended_turn(self, turn: Turn) -> bool:
        record = self.store.read()
        nudges = int(record.get("nudges") or 0)
        if nudges < NUDGES and not self.final_only:
            self.store.update(nudges=nudges + 1)
            self._say("You ended your turn without calling finish_run. If the job is done, call finish_run now "
                      "(outcome complete or partial). If not, carry on with your next tool call.")
            return False
        return self._open_items(turn.text)

    def _open_items(self, text: str) -> bool:
        """End the run "with open items": the model stopped without calling finish_run."""
        self.store.update(outcome={
            "kind": "open_items", "summary": text[:3000] or "The agent stopped without finishing.", "failing": [],
            "ended_without_finish": True, "checklist": [], "assumptions": [], "deviations": [],
            "owner_should_check": []})
        return True

    # --- the tools -------------------------------------------------------------------------------------------------

    def _tools(self, turn: Turn) -> bool:
        results = []
        for call in turn.tool_calls:
            if self.host.finished is not None:
                results.append(agent_tools.Result("NOT RUN: the run already ended with finish_run in this answer.",
                                                  ok=False, code="INVALID").block(call.id))
            elif self.stop.is_set() or self.pause.is_set():
                results.append(self._interrupted(call, "The run was stopped before this call."))
            else:
                results.append(self._one(call))
        content = [result for result in results]
        for note in self._notes:
            content.append({"type": "text", "text": note})
        self._notes.clear()
        self.store.append_message("user", content, meta={"tools": len(results)})
        self.host.account_jobs()
        return self.host.finished is not None

    def _one(self, call) -> dict:
        record = self.store.read()
        self.store.update(steps=int(record.get("steps") or 0) + 1)
        self.store.log("tool_start", id=call.id, tool=call.name)
        if self.final_only and call.name != "finish_run":
            result = agent_tools.Result("BUDGET: this run's limit is reached. Call finish_run now.", ok=False,
                                        code="BUDGET")
        else:
            waits = call.name == "wait_for_jobs"
            if waits:
                self._set_state("waiting", wait_reason="gpu_job")
            result = agent_tools.call(self.host, call.name, call.input)
            if waits:
                self._set_state("running", wait_reason="")
        self.store.log("tool_done", id=call.id, tool=call.name, ok=result.ok, code=result.code)
        self._repeats(call, result)
        return result.block(call.id)

    def _repeats(self, call, result) -> None:
        if call.name == "wait_for_jobs" or not result.ok:
            self.last_signature, self.same_calls = "", 0
            return
        signature = call.name + json.dumps(call.input, sort_keys=True, default=str)
        self.same_calls = self.same_calls + 1 if signature == self.last_signature else 1
        self.last_signature = signature
        if self.same_calls == STUCK_WARN:
            self._notes.append(f"Notice: you have made the same call to {call.name} {STUCK_WARN} times in a row with the "
                               "same answer. Do something different.")
        elif self.same_calls >= STUCK_STOP:
            raise Ended("failed", failure={"kind": "stuck", "message": f"The agent called {call.name} with the same "
                                           f"arguments {STUCK_STOP} times in a row and was stopped."})

    def _interrupted(self, call, why: str) -> dict:
        return agent_tools.Result(f"OWNER_STOPPED: {why}", ok=False, code="OWNER_STOPPED").block(call.id)

    # --- resuming ----------------------------------------------------------------------------------------------------

    def _repair(self) -> None:
        """After a stop or a crash the last message may be a tool call nobody answered, or an answer without a next
        step. Make the transcript valid again before the next request."""
        records = self.store.records()
        if not records:
            raise Ended("failed", failure={"kind": "internal", "message": "The run has no conversation yet."})
        last = records[-1]
        if last["role"] != "assistant":
            return
        calls = [block for block in last["content"] if isinstance(block, dict) and block.get("type") == "tool_use"]
        if not calls:
            self._say("The run was interrupted. Continue with the next step.")
            return
        events = self.store.events(limit=0)
        last_turn = max((index for index, event in enumerate(events) if event.get("type") == "turn"), default=-1)
        events = events[last_turn + 1:]  # only what happened after the last answer of the model (ids may repeat)
        done = {event["id"] for event in events if event.get("type") == "tool_done"}
        started = {event["id"] for event in events if event.get("type") == "tool_start"}
        content = []
        for block in calls:
            if block["id"] in done:
                text = ("COMPLETED: this call finished before the interruption, but its answer was lost. Read the state "
                        "again (get_project, get_queue, ...) before relying on it.")
            elif block["id"] in started:
                text = ("INTERRUPTED: the studio stopped while this call ran; it may or may not have taken effect. Check "
                        "the state before repeating it.")
            else:
                text = "NOT RUN: the studio stopped before this call began. Make it again if it is still needed."
            content.append({"type": "tool_result", "tool_use_id": block["id"], "is_error": True,
                            "content": [{"type": "text", "text": text}]})
        self.store.append_message("user", content, meta={"repaired": True})

    # --- a fresh conversation ----------------------------------------------------------------------------------------

    def fresh_conversation(self, reason: str) -> None:
        """The conversation is too long for the API: continue in a new one that starts with the brief again and a card of
        where the run stands, built from the run's own record (nothing is asked of the model)."""
        blocks = list(self.store.wire_messages()[0]["content"])
        blocks.append({"type": "text", "text": self.state_card(reason)})
        index = self.store.append_message("user", blocks, meta={"fresh": reason})
        self.store.update(pictures=count_pictures(blocks))  # the pictures of the earlier messages are no longer sent
        self.store.update(boundary=index - 1,
                          fresh_conversations=int(self.store.read().get("fresh_conversations") or 0) + 1)
        self.store.log("fresh_conversation", reason=reason)

    def state_card(self, reason: str) -> str:
        record = self.store.read()
        lines = ["The earlier messages of this run were cleared because the conversation became too long to send "
                 f"({reason}). The brief is above. This is where the run stands; read the real state with the tools "
                 "(get_project, get_queue) before you act, and carry on."]
        if self.host.mode == "produce":
            lines.append("This run PRODUCES: the owner approved production, so the instruction that this run only prepares "
                         "no longer applies. You may make clips, edit scenes and export the film.")
        status = record.get("status") or {}
        if status:
            lines.append(f"Phase: {status.get('phase')} - {status.get('text')}")
        resources = (record.get("resources") or {}).get("created") or []
        if resources:
            lines.append("Made by this run: " + ", ".join(resources[:80]))
        jobs = [f"#{job.id} {job.label[:60]} [{job.status}]" for job in self.host.own_jobs()][-20:]
        if jobs:
            lines.append("This run's jobs: " + "; ".join(jobs))
        changes = record.get("changes") or []
        if changes:
            lines.append("Changes to the owner's things: " + " ".join(changes[-20:]))
        questions = record.get("questions") or []
        if questions:
            lines.append("Questions already asked of the owner: " + " | ".join(questions))
        tools = [f"{event['tool']}{'' if event.get('ok') else ' (' + str(event.get('code')) + ')'}"
                 for event in self.store.events(limit=0) if event.get("type") == "tool_done"][-25:]
        if tools:
            lines.append("Your last calls: " + ", ".join(tools))
        return agent_config.redact("\n".join(lines))[:SUMMARY_CHARS]


