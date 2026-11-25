# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The agent's tools: how one is described, checked, run and reported.

Every function of Contsy the agent may use is one typed tool (the groups are in app/agent_tools_*.py and register
themselves here). A tool takes ids, slugs, numbers and text, never a path or a command, and answers with data the
model can read. `call` is the one door: it checks the arguments against the tool's schema, the run's mode, then runs the
handler and turns any refusal into a short error with a code (agent_host.CODES) that says what to do instead."""
from __future__ import annotations

import base64
import json
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

from . import agent_config, llm, timeline
from .agent_host import Host, ToolError
from .downloader import BLOCKED_PREFIX, DownloadsBlocked, no_downloads

MAX_RESULT_CHARS = 24_000
MAX_TEXT = 8_000          # the longest text any tool takes unless its schema says less
MAX_LOG_CHARS = 300
_TOOLS: dict[str, "Tool"] = {}
_GROUPS = ("agent_tools_library", "agent_tools_film", "agent_tools_studio", "agent_tools_run", "agent_tools_delivery",
           "agent_tools_eyes", "agent_tools_series")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict
    handler: Callable[[Host, dict], Any]
    produce: bool = False  # final production: refused in a run that only prepares


@dataclass
class Reply:
    """What a handler returns when it has pictures to show the model as well as data."""
    data: Any
    images: list = field(default_factory=list)   # PNG or JPEG bytes


@dataclass
class Result:
    text: str
    ok: bool = True
    code: str = ""
    images: list = field(default_factory=list)

    def block(self, tool_use_id: str) -> dict:
        content: list[dict] = [{"type": "text", "text": self.text}]
        for data in self.images:
            media = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
            content.append({"type": "image", "source": {"type": "base64", "media_type": media,
                                                        "data": base64.b64encode(data).decode("ascii")}})
        block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
        if not self.ok:
            block["is_error"] = True
        return block


# --- schemas ---------------------------------------------------------------------------------------------------

def obj(properties: dict, required: tuple = ()) -> dict:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def text(description: str, *, max_length: int = MAX_TEXT, min_length: int = 0, pattern: str = "") -> dict:
    found = {"type": "string", "description": description, "maxLength": max_length}
    if min_length:
        found["minLength"] = min_length
    if pattern:
        found["pattern"] = pattern
    return found


def whole(description: str, low: int, high: int) -> dict:
    return {"type": "integer", "description": description, "minimum": low, "maximum": high}


def number(description: str, low: float, high: float) -> dict:
    return {"type": "number", "description": description, "minimum": low, "maximum": high}


def flag(description: str) -> dict:
    return {"type": "boolean", "description": description}


def choice(description: str, *values: str) -> dict:
    return {"type": "string", "description": description, "enum": list(values)}


def many(description: str, item: dict, *, max_items: int = 50) -> dict:
    return {"type": "array", "description": description, "items": item, "maxItems": max_items}


def _fail(path: str, why: str):
    raise ToolError("INVALID", f"{path}: {why}")


def validate(schema: dict, value, path: str = "input") -> None:
    """The subset of JSON Schema the tools use (type, enum, required, properties, additionalProperties, lengths,
    bounds, pattern, items). Anything not allowed by the schema is refused; nothing is coerced."""
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            _fail(path, "must be an object")
        props = schema.get("properties", {})
        for name in value:
            if name not in props:
                _fail(f"{path}.{name}", "is not an argument of this tool")
        for name in schema.get("required", ()):
            if name not in value:
                _fail(f"{path}.{name}", "is required")
        for name, sub in props.items():
            if name in value:
                validate(sub, value[name], f"{path}.{name}")
    elif kind == "array":
        if not isinstance(value, list):
            _fail(path, "must be a list")
        if len(value) > schema.get("maxItems", 10_000):
            _fail(path, f"has more than {schema['maxItems']} entries")
        if len(value) < schema.get("minItems", 0):
            _fail(path, f"needs at least {schema['minItems']} entries")
        for number_, entry in enumerate(value):
            validate(schema.get("items", {}), entry, f"{path}[{number_}]")
    elif kind == "string":
        if not isinstance(value, str):
            _fail(path, "must be text")
        if len(value) > schema.get("maxLength", MAX_TEXT):
            _fail(path, f"is longer than {schema.get('maxLength', MAX_TEXT)} characters")
        if len(value) < schema.get("minLength", 0):
            _fail(path, f"must have at least {schema['minLength']} characters")
        if "enum" in schema and value not in schema["enum"]:
            _fail(path, f"must be one of: {', '.join(schema['enum'])}")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            _fail(path, "has the wrong form")
    elif kind in ("integer", "number"):
        ok = isinstance(value, int) if kind == "integer" else isinstance(value, (int, float))
        if isinstance(value, bool) or not ok or value != value or value in (float("inf"), float("-inf")):
            _fail(path, f"must be {'a whole number' if kind == 'integer' else 'a number'}")
        if value < schema.get("minimum", float("-inf")) or value > schema.get("maximum", float("inf")):
            _fail(path, f"must be between {schema.get('minimum')} and {schema.get('maximum')}")
    elif kind == "boolean":
        if not isinstance(value, bool):
            _fail(path, "must be true or false")
    elif kind is not None:
        raise ValueError(f"Unsupported schema type {kind}.")


# --- registry --------------------------------------------------------------------------------------------------

def tool(name: str, description: str, schema: dict, *, produce: bool = False):
    """Register a handler `(host, args) -> data` as the tool `name`."""
    def register(handler: Callable[[Host, dict], Any]):
        if name in _TOOLS:
            raise ValueError(f"The tool {name} is registered twice.")
        _TOOLS[name] = Tool(name, " ".join(description.split()), schema, handler, produce)
        return handler
    return register


def registry() -> dict[str, Tool]:
    import importlib

    for module in _GROUPS:
        try:
            importlib.import_module(f"app.{module}")
        except ModuleNotFoundError as error:  # a group that is not part of this build yet
            if error.name != f"app.{module}":
                raise
    return dict(sorted(_TOOLS.items()))


def definitions() -> list[dict]:
    """The tool list as the API wants it: the same text every time, so the cached prefix of the run stays valid."""
    return [{"name": one.name, "description": one.description, "input_schema": one.schema}
            for one in registry().values()]


# --- running a tool ----------------------------------------------------------------------------------------------

def _refusal(error: BaseException) -> ToolError:
    """What a Contsy function's complaint means for the model."""
    if isinstance(error, ToolError):
        return error
    message = agent_config.redact(str(error)).strip() or type(error).__name__
    if isinstance(error, timeline.Busy):
        return ToolError("BUSY", message)
    if isinstance(error, timeline.Deleted):
        return ToolError("NOT_FOUND", message)
    if isinstance(error, DownloadsBlocked) or message.startswith(BLOCKED_PREFIX):
        return ToolError("MODELS_MISSING", message)
    if message.startswith(("Blocked:", "⛔")) or message == llm.BLOCK_MSG:
        return ToolError("CONTENT_RULE", message)
    if "another tab" in message or "timeline changed" in message or "Open the saved project again" in message:
        return ToolError("STALE", message + " Read the project again (get_project) and repeat the change.")
    if "already has a job in the queue" in message or "job in the GPU queue" in message:
        return ToolError("BUSY", message)
    if isinstance(error, ValueError):
        return ToolError("INVALID", message)
    return ToolError("INTERNAL", f"The tool failed unexpectedly ({type(error).__name__}). Try something else.")


def _clip(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)[:MAX_LOG_CHARS]
    except (TypeError, ValueError):
        return "?"


def _text(data) -> str:
    out = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(out) > MAX_RESULT_CHARS:
        out = out[:MAX_RESULT_CHARS] + "\n…(the answer was cut; ask for a narrower selection)"
    return out


def call(host: Host, name: str, arguments) -> Result:
    """Run one tool call of the model. Never raises: every refusal comes back as an error result."""
    tools = registry()
    started = time.monotonic()
    one = tools.get(name)
    try:
        if one is None:
            raise ToolError("INVALID", f"There is no tool called {name}.")
        validate(one.schema, arguments if arguments is not None else {})
        if one.produce:
            host.require_produce(f"The tool {name}")
        with (nullcontext() if host.allowed("downloads") else no_downloads()):  # a check that loads a model may not fetch it
            reply = one.handler(host, arguments or {})
        images = []
        if isinstance(reply, Reply):
            reply, images = reply.data, reply.images
        result = Result(_text(reply if reply is not None else "Done."), images=images)
    except ToolError as error:
        result = Result(f"{error.code}: {error}", ok=False, code=error.code)
    except Exception as error:  # noqa: BLE001 - a tool must never stop the run
        refusal = _refusal(error)
        if refusal.code == "INTERNAL":
            host.store.log("tool_crash", tool=name, error=type(error).__name__, detail=str(error)[:300])
        result = Result(f"{refusal.code}: {refusal}", ok=False, code=refusal.code)
    host.store.log("tool", tool=name, input=_clip(arguments), ok=result.ok, code=result.code,
                   ms=int((time.monotonic() - started) * 1000))
    return result

