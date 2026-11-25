# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Fakes for the Agent page's tests: a scripted provider (no SDK, no network), a clock that does not wait, and the
Server-Sent-Events text of a Claude response for tests that run the real SDK against a mock transport."""
from __future__ import annotations

import json
import threading
from typing import Callable, Optional

from app.agent_provider import Aborted, ProviderError, ToolCall, Turn, TurnRequest, UsageEntry

MODEL = "claude-opus-5-5"


def text_turn(text: str = "Done.", *, usage: Optional[list] = None, model: str = MODEL, stop: str = "end_turn",
              request_id: str = "req_test", context: int = 1000) -> Turn:
    block = {"type": "text", "text": text}
    return Turn(content=[block], stop_reason=stop, model=model, text=text,
                usage=usage if usage is not None else [UsageEntry(model, input_tokens=1000, output_tokens=100)],
                request_id=request_id, context_tokens=context)


def tool_turn(*calls: tuple, text: str = "", model: str = MODEL, usage: Optional[list] = None, thinking: bool = True,
              context: int = 1000) -> Turn:
    """A turn that calls tools: `calls` are (name, input) or (name, input, id) tuples."""
    content: list[dict] = []
    if thinking:
        content.append({"type": "thinking", "thinking": "", "signature": "sig-" + "x" * 16})
    if text:
        content.append({"type": "text", "text": text})
    made = []
    for number, call in enumerate(calls, 1):
        name, arguments = call[0], call[1]
        identifier = call[2] if len(call) > 2 else f"toolu_{number:02d}"
        content.append({"type": "tool_use", "id": identifier, "name": name, "input": arguments})
        made.append(ToolCall(identifier, name, arguments))
    return Turn(content=content, stop_reason="tool_use", model=model, text=text, tool_calls=made,
                usage=usage if usage is not None else [UsageEntry(model, input_tokens=2000, output_tokens=300)],
                request_id="req_tool", context_tokens=context)


def refusal_turn(category: str = "general_harms") -> Turn:
    return Turn(content=[], stop_reason="refusal", model=MODEL, usage=[UsageEntry(MODEL, input_tokens=500)],
                stop_details={"type": "refusal", "category": category, "explanation": "declined"})


def error(kind: str, message: str = "failed", **fields) -> ProviderError:
    return ProviderError(kind, message, **fields)


class ScriptedProvider:
    """A provider that returns the next scripted item: a Turn, a ProviderError (raised), or a callable
    `(request) -> Turn | ProviderError`. Records every TurnRequest, honours the stop flag, and tells the script's
    progress through `calls`."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests: list[TurnRequest] = []
        self.snapshots: list[list[dict]] = []  # a copy of each request's messages
        self.pings: list[str] = []
        self.lock = threading.Lock()
        self.ping_result: object = "Claude Opus 5.5"

    @property
    def calls(self) -> int:
        return len(self.requests)

    def complete(self, request: TurnRequest, *, should_abort: Callable[[], bool] = lambda: False,
                 on_event: Callable[[str], None] = lambda name: None) -> Turn:
        with self.lock:
            self.requests.append(request)
            self.snapshots.append(json.loads(json.dumps(request.messages)))
            if not self.script:
                raise AssertionError("The scripted provider ran out of turns.")
            item = self.script.pop(0)
        if should_abort():
            raise Aborted()
        on_event("message_start")
        if callable(item):
            item = item(request)
        if isinstance(item, BaseException):
            raise item
        return item

    def ping(self, model: str) -> str:
        self.pings.append(model)
        if isinstance(self.ping_result, BaseException):
            raise self.ping_result
        return str(self.ping_result)


class FakeClock:
    """Time that moves only when something sleeps; `sleep(seconds, wake)` returns early when `wake()` is true."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float, wake: Callable[[], bool] = lambda: False) -> None:
        self.slept.append(seconds)
        self.now += seconds


# --- Server-Sent Events of a response (for the real SDK on a mock transport) ----------------------------------------

def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def message_sse(blocks: list[dict], *, stop_reason: str = "end_turn", model: str = MODEL,
                usage: Optional[dict] = None, message_id: str = "msg_test") -> str:
    """The SSE text of a response made of `blocks` (text, thinking, tool_use), streamed the way the API streams it."""
    usage = usage or {"input_tokens": 120, "output_tokens": 40, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}
    out = [_event("message_start", {"type": "message_start", "message": {
        "id": message_id, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {**usage, "output_tokens": 1}}})]
    for index, block in enumerate(blocks):
        kind = block["type"]
        if kind == "text":
            out.append(_event("content_block_start", {"type": "content_block_start", "index": index,
                                                      "content_block": {"type": "text", "text": ""}}))
            out.append(_event("content_block_delta", {"type": "content_block_delta", "index": index,
                                                      "delta": {"type": "text_delta", "text": block["text"]}}))
        elif kind == "thinking":
            out.append(_event("content_block_start", {"type": "content_block_start", "index": index,
                                                      "content_block": {"type": "thinking", "thinking": ""}}))
            if block.get("thinking"):
                out.append(_event("content_block_delta", {"type": "content_block_delta", "index": index,
                                                          "delta": {"type": "thinking_delta",
                                                                    "thinking": block["thinking"]}}))
            out.append(_event("content_block_delta", {"type": "content_block_delta", "index": index,
                                                      "delta": {"type": "signature_delta",
                                                                "signature": block["signature"]}}))
        elif kind == "tool_use":
            out.append(_event("content_block_start", {"type": "content_block_start", "index": index,
                                                      "content_block": {"type": "tool_use", "id": block["id"],
                                                                        "name": block["name"], "input": {}}}))
            out.append(_event("content_block_delta", {"type": "content_block_delta", "index": index,
                                                      "delta": {"type": "input_json_delta",
                                                                "partial_json": json.dumps(block["input"])}}))
        else:
            raise ValueError(f"No SSE for a {kind} block.")
        out.append(_event("content_block_stop", {"type": "content_block_stop", "index": index}))
    out.append(_event("message_delta", {"type": "message_delta",
                                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                        "usage": {"output_tokens": usage["output_tokens"]}}))
    out.append(_event("message_stop", {"type": "message_stop"}))
    return "".join(out)


def error_body(kind: str, message: str) -> dict:
    return {"type": "error", "error": {"type": kind, "message": message}}
