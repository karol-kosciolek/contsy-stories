# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Claude (Anthropic) as the Agent page's provider. This is the only module that imports the `anthropic` package, and
it does so only when a request is made, so the app starts the same without a key.

How a request is made (anthropic 2.10.0, checked against the installed package, see tests/test_agent_claude.py):
- `client.beta.messages.stream(...)` and its stream is read event by event, so the stop flag can close it; the final
  message becomes a Turn whose blocks are the plain JSON the server sent (thinking blocks included) and are sent back
  unchanged later.
- thinking adaptive (never disabled: the 5.x models reject it), `output_config.effort` set explicitly, top-level
  `cache_control`, no sampling parameters, no prefill, no forced tool choice.
- optional features, each behind a beta header: automatic compaction of a long conversation (`compact-2026-01-12`), the
  server-side fallback on a refusal (`server-side-fallback-2026-07-01`, not offered for the smallest model) and, after
  the server declined a thinking block of a resumed conversation, `drop_block` (`thinking-binding-controls-2026-08-01`).
  When the server refuses a beta by name, that feature is turned off for this provider and the request is repeated.
- the client has no retries of its own (the run loop retries, so every wait is logged and can be stopped), the key is
  passed explicitly with an explicit base URL, and every ANTHROPIC_* variable of the environment is hidden while the
  client is built, so a stray token, profile or proxy header cannot change where the request goes or who it bills.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import threading
from typing import Callable, Optional

from . import agent_config
from .agent_provider import (Aborted, ProviderError, ToolCall, Turn, TurnRequest, UsageEntry, spec_of)

BASE_URL = "https://api.anthropic.com"
CONNECT_TIMEOUT, READ_TIMEOUT = 15.0, 300.0
BETAS = {"compaction": "compact-2026-01-12", "fallbacks": "server-side-fallback-2026-07-01",
         "binding": "thinking-binding-controls-2026-08-01"}
_FEATURE_OF_BETA = {beta: feature for feature, beta in BETAS.items()}
_ENV_LOCK = threading.Lock()
_UNKNOWN_BETA = re.compile(r"Unexpected value\(s\)\s*`?(?P<names>[^`]+?)`?\s+for the `anthropic-beta` header", re.I)
_PARAMETER_HINTS = {"fallbacks": "fallbacks", "context_management": "compaction", "compact": "compaction",
                    "block_binding": "binding"}


@contextlib.contextmanager
def _clean_environment():
    """Hide every ANTHROPIC_* variable while the client is built (it reads them when it is made)."""
    with _ENV_LOCK:
        hidden = {name: value for name, value in os.environ.items() if name.upper().startswith("ANTHROPIC_")}
        for name in hidden:
            os.environ.pop(name, None)
        try:
            yield
        finally:
            os.environ.update(hidden)


def make_client(key: str, http_client=None):
    import anthropic

    with _clean_environment():
        return anthropic.Anthropic(api_key=key, base_url=BASE_URL, max_retries=0,
                                   timeout=anthropic.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
                                   http_client=http_client)


def _exception_classes() -> dict:
    import anthropic

    return {name: getattr(anthropic, name, None) for name in (
        "AuthenticationError", "PermissionDeniedError", "PaymentRequiredError", "EnforcedSpendLimitReachedError",
        "RateLimitError", "RequestTooLargeError", "ContentTooLargeError", "NotFoundError", "OverloadedError",
        "InternalServerError", "ServiceUnavailableError", "GatewayTimeoutError", "RequestTimeoutError",
        "APIConnectionError", "APITimeoutError", "BadRequestError", "APIStatusError", "APIError")}


def _body_text(exc) -> tuple[str, str]:
    """(error type, message) of an API error, from its parsed body when there is one."""
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    kind = str((error or {}).get("type") or "") if isinstance(error, dict) else ""
    message = str((error or {}).get("message") or "") if isinstance(error, dict) else ""
    return kind, message or str(getattr(exc, "message", "") or exc)


_TRANSPORT_MODULES = ("httpx", "httpx2", "httpcore", "h11", "anyio", "ssl", "socket")


def _transport_kind(exc: BaseException) -> str:
    """"network" or "timeout" for an error of the connection itself (a read that broke in the middle of an answer
    arrives as an httpx or socket error, not as an API error), else ""."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "timeout" if isinstance(exc, TimeoutError) else "network"
    for cls in type(exc).__mro__:
        if cls.__module__.split(".")[0] in _TRANSPORT_MODULES:
            return "timeout" if "timeout" in type(exc).__name__.lower() else "network"
    return ""


def map_error(exc: BaseException) -> ProviderError:
    """An SDK exception as a ProviderError, classified by its class, its error type and its message (a mid-stream error
    frame carries no useful HTTP status)."""
    if isinstance(exc, ProviderError):
        return exc
    classes = _exception_classes()

    def is_a(*names) -> bool:
        return any(classes.get(name) is not None and isinstance(exc, classes[name]) for name in names)

    status = getattr(exc, "status_code", None)
    etype, message = _body_text(exc)
    lower = f"{etype} {message} {json.dumps(getattr(exc, 'body', None), default=str)}".lower()
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        retry_after = float(headers.get("retry-after", 0) or 0)
    except (TypeError, ValueError):
        retry_after = 0.0
    request_id = str(getattr(exc, "request_id", "") or "")
    detail = agent_config.redact(message)

    def error(kind: str, text: str) -> ProviderError:
        return ProviderError(kind, agent_config.redact(text), status=status, retry_after=retry_after,
                             request_id=request_id, error_type=etype, detail=detail)

    spend = ("enforced_spend_limit" in lower or "spend limit" in lower or "billing_error" in lower
             or "credit balance" in lower)
    if is_a("PaymentRequiredError", "EnforcedSpendLimitReachedError") or status == 402 or (spend and status in (400, 429)):
        return error("billing", "Your Anthropic account has no credit left or has reached its spend limit. Add credit "
                                "or raise the limit in the Claude Console, then resume the run.")
    if is_a("AuthenticationError", "PermissionDeniedError") or status in (401, 403):
        return error("auth", "Anthropic did not accept the API key. Check it in the Claude Console and save it again.")
    if is_a("RequestTooLargeError", "ContentTooLargeError") or status == 413:
        return error("too_large", "The request is too large for the API (a big brief or too many images).")
    if is_a("RateLimitError") or status == 429:
        return error("rate_limit", "Anthropic's rate limit was reached; waiting before trying again.")
    if is_a("NotFoundError") or status == 404:
        return error("model", "This model is not available to your account. Choose another model.")
    if is_a("BadRequestError") or status == 400:
        if "bound to a different conversation" in lower or "invalid signature" in lower:
            return error("binding", "The saved conversation no longer matches what the server remembers of it.")
        if "prompt is too long" in lower or "context window" in lower or "too many tokens" in lower:
            return error("context", "The conversation no longer fits the model's context window.")
        if "retention" in lower:
            return error("model", "This model needs data retention to be enabled for your Anthropic organization.")
        if _UNKNOWN_BETA.search(message):
            return error("feature", "A request feature is not available to this account.")
        return error("bad_request", f"Anthropic refused the request: {message[:300]}")
    if is_a("OverloadedError") or status == 529 or etype == "overloaded_error":
        return error("overloaded", "Anthropic is overloaded right now; trying again.")
    if is_a("APITimeoutError", "RequestTimeoutError", "GatewayTimeoutError") or status in (408, 504):
        return error("timeout", "Anthropic took too long to answer; trying again.")
    if is_a("APIConnectionError"):
        return error("network", "Could not reach Anthropic. Check the internet connection; trying again.")
    if is_a("InternalServerError", "ServiceUnavailableError") or (status or 0) >= 500 or etype in ("api_error",):
        return error("server", "Anthropic had a problem on its side; trying again.")
    if "rate_limit" in lower:
        return error("rate_limit", "Anthropic's rate limit was reached; waiting before trying again.")
    if transport := _transport_kind(exc):
        return error(transport, "The connection to Anthropic broke; trying again.")
    return error("unknown", f"Unexpected error: {agent_config.redact(str(exc))[:300]}")


def _features_named(message: str) -> list[str]:
    found = _UNKNOWN_BETA.search(message or "")
    if not found:
        return []
    names = [name.strip(" `'\"") for name in re.split(r"[,\s]+", found.group("names")) if name.strip()]
    return [_FEATURE_OF_BETA[name] for name in names if name in _FEATURE_OF_BETA]


def _usage_entry(item, model: str) -> UsageEntry:
    created = getattr(item, "cache_creation", None)
    if created is not None:
        write_5m = int(getattr(created, "ephemeral_5m_input_tokens", 0) or 0)
        write_1h = int(getattr(created, "ephemeral_1h_input_tokens", 0) or 0)
    else:
        write_5m, write_1h = int(getattr(item, "cache_creation_input_tokens", 0) or 0), 0
    return UsageEntry(model=str(getattr(item, "model", None) or model), input_tokens=int(item.input_tokens or 0),
                      output_tokens=int(item.output_tokens or 0), cache_read=int(item.cache_read_input_tokens or 0),
                      cache_write_5m=write_5m, cache_write_1h=write_1h, kind=str(getattr(item, "type", "message")))


def usage_entries(final, requested: str = "") -> list[UsageEntry]:
    """The sampling iterations of a response: its `usage.iterations` when the server itemized them (a fallback attempt,
    a compaction), else the one entry of its top-level usage. An iteration that names no model ran on the model that
    was asked for (`requested`); the response's own model may be a fallback's."""
    usage = final.usage
    iterations = getattr(usage, "iterations", None)
    if iterations:
        return [_usage_entry(item, requested or str(final.model)) for item in iterations]
    return [_usage_entry(usage, str(final.model))]


def to_turn(final, requested: str = "") -> Turn:
    content = [block.model_dump(mode="json", exclude_none=True) for block in final.content]
    entries = usage_entries(final, requested)
    top = final.usage
    details = getattr(final, "stop_details", None)
    transformations = getattr(final, "input_transformations", None) or []
    return Turn(
        content=content, stop_reason=str(final.stop_reason or ""), model=str(final.model or ""),
        text="".join(block.get("text", "") for block in content if block.get("type") == "text"),
        tool_calls=[ToolCall(block["id"], block["name"], block.get("input") if isinstance(block.get("input"), dict)
                             else {}) for block in content if block.get("type") == "tool_use"],
        usage=entries, request_id=str(getattr(final, "_request_id", "") or ""),
        stop_details=details.model_dump(mode="json", exclude_none=True) if details is not None else None,
        served_by_fallback=any(entry.kind == "fallback_message" for entry in entries),
        context_tokens=int(top.input_tokens or 0) + int(top.cache_read_input_tokens or 0)
        + int(top.cache_creation_input_tokens or 0),
        transformations=[item.model_dump(mode="json", exclude_none=True) if hasattr(item, "model_dump") else item
                         for item in transformations])


class ClaudeProvider:
    """Requests to the Claude API. `key`: the API key (default: the page's saved key or the environment variable,
    read when a request is made); `client`: an SDK client to use instead (tests)."""

    def __init__(self, key: Optional[str] = None, client=None):
        self._key, self._client, self._built_for = key, client, None
        self._lock = threading.Lock()
        self.off: set[str] = set()          # features the server refused: compaction, fallbacks, binding
        self.drop_blocks = False            # thinking blocks the server does not accept are dropped

    # --- request -----------------------------------------------------------------------------------------------

    def build_params(self, request: TurnRequest) -> dict:
        spec = spec_of(request.model)
        features = request.features
        thinking: dict = {"type": "adaptive"}
        betas: list[str] = []
        params: dict = {
            "model": request.model, "max_tokens": request.max_tokens, "system": request.system,
            "tools": request.tools, "messages": request.messages, "thinking": thinking,
            "output_config": {"effort": request.effort},
            "cache_control": ({"type": "ephemeral", "ttl": "1h"} if features.cache_ttl == "1h"
                              else {"type": "ephemeral"}),
        }
        if features.compaction and "compaction" not in self.off:
            edit: dict = {"type": "compact_20260112", "trigger": {"type": "input_tokens",
                                                                   "value": features.compact_trigger}}
            if features.compact_instructions:
                edit["instructions"] = features.compact_instructions
            params["context_management"] = {"edits": [edit]}
            betas.append(BETAS["compaction"])
        if features.fallbacks and spec.fallbacks and "fallbacks" not in self.off:
            params["fallbacks"] = "default"
            betas.append(BETAS["fallbacks"])
        if self.drop_blocks and "binding" not in self.off:
            thinking["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
            betas.append(BETAS["binding"])
        if betas:
            params["betas"] = betas
        return params

    def _get_client(self):
        with self._lock:
            if self._client is not None and self._built_for in (None, self._key_now()):
                return self._client
            key = self._key_now()
            if not key:
                raise ProviderError("auth", "No Claude API key is saved. Paste it on the Agent page and save it.")
            self._client, self._built_for = make_client(key), key
            return self._client

    def _key_now(self) -> str:
        return self._key or agent_config.get_key()[0]

    # --- turns -------------------------------------------------------------------------------------------------

    def complete(self, request: TurnRequest, *, should_abort: Callable[[], bool] = lambda: False,
                 on_event: Callable[[str], None] = lambda name: None) -> Turn:
        for _ in range(4):
            params = self.build_params(request)
            try:
                return self._stream(params, should_abort, on_event)
            except Aborted:
                raise
            except Exception as exc:  # noqa: BLE001 - classified; unknown ones keep their text
                error = map_error(exc)
                if self._adapt(error):
                    continue
                raise error from exc
        raise ProviderError("bad_request", "The server kept refusing optional request features.")

    def _stream(self, params: dict, should_abort, on_event) -> Turn:
        client = self._get_client()
        with client.beta.messages.stream(**params) as stream:
            for event in stream:
                if should_abort():
                    stream.close()
                    raise Aborted()
                on_event(str(getattr(event, "type", "")))
            final = stream.get_final_message()
        if not getattr(final, "stop_reason", None):  # the stream ended before the message did: a half answer
            raise ProviderError("network", "The answer was cut off before it was complete; trying again.")
        turn = to_turn(final, str(params.get("model") or ""))
        if not turn.request_id:
            turn.request_id = str(getattr(stream, "request_id", "") or "")
        return turn

    def _adapt(self, error: ProviderError) -> bool:
        """Turn off what the server refused, or start dropping thinking blocks it cannot match. True: repeat."""
        if error.kind == "feature":
            fresh = [name for name in _features_named(error.detail) if name not in self.off]
            self.off.update(fresh)
            return bool(fresh)
        if error.kind == "binding" and not self.drop_blocks:
            self.drop_blocks = True
            return True
        if error.kind == "bad_request":
            lower = error.detail.lower()
            fresh = [feature for hint, feature in _PARAMETER_HINTS.items()
                     if hint in lower and feature not in self.off]
            self.off.update(fresh)
            return bool(fresh)
        return False

    def ping(self, model: str) -> str:
        """The display name the server gives `model`: proves the key and the model without sampling anything."""
        try:
            info = self._get_client().models.retrieve(model)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise map_error(exc) from exc
        return str(getattr(info, "display_name", "") or model)
