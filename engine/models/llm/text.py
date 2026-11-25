# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The text side of the language model: the ChatML prompt, cleaning of an answer, strict JSON, a schema check.

* The text prompt is written in the Qwen ChatML layout, `<|im_start|>system\\n...<|im_end|>\\n` and
  `<|im_start|>user\\n...<|im_end|>\\n`, and ends with `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`. The empty
  think block turns thinking off: the model answers directly and every `max_tokens` goes to the answer.
* The answer loses `<think>...</think>` blocks (a model sometimes thinks anyway). A lone `</think>` keeps only what
  follows it (the prompt opened the block), and an unfinished `<think>` drops the rest (an unfinished thought is not
  an answer).
* Text that goes to the model has its special tokens removed, so an idea that contains `<|im_start|>system` cannot
  forge a turn.

JSON answers are fail-closed. `parse_json_output` returns a value only when the answer holds exactly one JSON
object (or array) that satisfies the schema; anything else raises `LlmBadOutput`. It never returns a default.
The validator covers the keywords the product's schemas use plus the common ones around them; a schema that uses
a keyword it does not know is refused up front rather than checked loosely.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Optional, Sequence

from .errors import LlmBadOutput, LlmError

EMPTY_THINK = "<think>\n\n</think>\n\n"
STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")  # end-of-turn markers; the EOG token normally ends the run first

_SPECIAL_TOKEN = re.compile(r"<\|[A-Za-z0-9_]{1,40}\|>|<__media__>")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
_END_MARKERS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


def neutralize_special_tokens(text: str) -> str:
    """Remove `<|im_start|>`-style control tokens (and the multimodal media marker) from text sent to the model."""
    return _SPECIAL_TOKEN.sub("", text)


def build_chatml(messages: Sequence[Mapping[str, Any]]) -> str:
    """The prompt string of a plain-text request. `messages` hold `role` (system, user, assistant) and `content`."""
    parts = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant"):
            raise LlmError(f"Unknown message role {role!r}.")
        if not isinstance(content, str):
            raise LlmError("A text request takes plain-text messages only.")
        parts.append(f"<|im_start|>{role}\n{neutralize_special_tokens(content)}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n" + EMPTY_THINK)
    return "".join(parts)


def strip_think(text: str) -> str:
    """Drop thinking: closed blocks, then everything up to a lone `</think>`, then an unfinished `<think>` tail."""
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    cut = text.find("<think>")
    return text if cut < 0 else text[:cut]


def clean_output(text: str) -> str:
    """The answer as the product wants it: no think text, no end markers, no surrounding blank space."""
    text = strip_think(text)
    for marker in _END_MARKERS:
        cut = text.find(marker)
        if cut >= 0:
            text = text[:cut]
    return text.strip()


# --- strict JSON ---------------------------------------------------------------------------------------------

def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _reject_duplicates(pairs: list) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _finite_float(number: str) -> float:
    value = float(number)
    if not math.isfinite(value):
        raise ValueError(f"{number} is outside the supported number range")
    return value


_DECODER = json.JSONDecoder(object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant,
                            parse_float=_finite_float)
_FENCE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\n(.*?)\n?```\s*$", re.S)


def extract_json(text: str) -> dict | list:
    """Decode one complete JSON object or array, optionally in a Markdown fence."""
    body = text.strip()
    while body.startswith("<think>"):
        closing = body.find("</think>")
        if closing < 0:
            raise LlmBadOutput("The model did not finish its answer.", body)
        body = body[closing + len("</think>"):].lstrip()
    while marker := next((m for m in STOP_STRINGS if body.endswith(m)), None):
        body = body[:-len(marker)].rstrip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        value = _DECODER.decode(body)
    except (ValueError, RecursionError) as exc:
        raise LlmBadOutput(f"The model's answer is not valid JSON or was cut off ({exc}).", body) from None
    if not isinstance(value, (dict, list)):
        raise LlmBadOutput("The model's answer must be a JSON object or array.", body)
    return value


# --- schema check --------------------------------------------------------------------------------------------

_ANNOTATIONS = frozenset({"$schema", "$id", "$comment", "title", "description", "default", "examples", "format",
                          "deprecated", "readOnly", "writeOnly"})
_CHECKED = frozenset({"type", "enum", "const", "required", "properties", "additionalProperties", "items",
                      "minItems", "maxItems", "uniqueItems", "minLength", "maxLength", "pattern", "minimum",
                      "maximum", "anyOf", "oneOf", "allOf"})
_TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})


class _Violation(Exception):
    pass


def check_schema(schema: Any, path: str = "$") -> None:
    """Refuse a schema that uses a keyword the validator does not implement (so nothing is checked loosely)."""
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise LlmError(f"The JSON schema at {path} must be an object.")
    for key, value in schema.items():
        if key in _ANNOTATIONS:
            continue
        if key not in _CHECKED:
            raise LlmError(f"The JSON schema keyword {key!r} at {path} is not supported.")
        if key == "type":
            names = [value] if isinstance(value, str) else value
            if not isinstance(names, list) or not names or any(not isinstance(n, str) or n not in _TYPES
                                                                 for n in names):
                raise LlmError(f"The JSON schema type at {path} is not valid.")
        elif key == "properties":
            if not isinstance(value, dict) or not all(isinstance(name, str) for name in value):
                raise LlmError(f"The JSON schema properties at {path} must be an object.")
            for name, sub in value.items():
                check_schema(sub, f"{path}.{name}")
        elif key == "items":
            if isinstance(value, list):
                raise LlmError(f"The JSON schema at {path} uses a list of item schemas, which is not supported.")
            check_schema(value, f"{path}[]")
        elif key == "additionalProperties":
            check_schema(value, f"{path}.*")
        elif key in ("anyOf", "oneOf", "allOf"):
            if not isinstance(value, list) or not value:
                raise LlmError(f"The JSON schema {key} at {path} must be a non-empty list.")
            for sub in value:
                check_schema(sub, path)
        elif key == "pattern":
            try:
                re.compile(value)
            except (re.error, TypeError):
                raise LlmError(f"The JSON schema pattern at {path} is not a valid expression.") from None
        elif key == "required":
            if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
                raise LlmError(f"The JSON schema required at {path} must be a list of property names.")
        elif key == "enum":
            if not isinstance(value, list) or not value:
                raise LlmError(f"The JSON schema enum at {path} must be a non-empty list.")
        elif key in ("minItems", "maxItems", "minLength", "maxLength"):
            if type(value) is not int or value < 0:
                raise LlmError(f"The JSON schema {key} at {path} must be a nonnegative integer.")
        elif key in ("minimum", "maximum"):
            if type(value) not in (int, float) or isinstance(value, float) and not math.isfinite(value):
                raise LlmError(f"The JSON schema {key} at {path} must be a finite number.")
        elif key == "uniqueItems" and type(value) is not bool:
            raise LlmError(f"The JSON schema uniqueItems at {path} must be a boolean.")


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return {str: "string", list: "array", dict: "object"}.get(type(value), type(value).__name__)


def _is_type(value: Any, name: str) -> bool:
    if name == "integer":  # a float such as 3.0 is not accepted: the grammar writes integers and the product tests `type(x) is int`
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return _kind(value) == name


def _equal(a: Any, b: Any) -> bool:
    """JSON equality: True is not 1, and containers compare element by element."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _validate(value: Any, schema: Any, path: str) -> None:
    if schema is True:
        return
    if schema is False:
        raise _Violation(f"{path}: no value is allowed here")
    names = schema.get("type")
    if names is not None:
        names = [names] if isinstance(names, str) else names
        if not any(_is_type(value, n) for n in names):
            raise _Violation(f"{path}: expected {' or '.join(names)}, got {_kind(value)}")
    if "const" in schema and not _equal(value, schema["const"]):
        raise _Violation(f"{path}: not the required constant")
    if "enum" in schema and not any(_equal(value, option) for option in schema["enum"]):
        raise _Violation(f"{path}: not one of the allowed values")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise _Violation(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise _Violation(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise _Violation(f"{path}: does not match the pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise _Violation(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise _Violation(f"{path}: above {schema['maximum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise _Violation(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise _Violation(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems") and any(_equal(value[i], value[j])
                                             for i in range(len(value)) for j in range(i)):
            raise _Violation(f"{path}: items repeat")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate(item, schema["items"], f"{path}[{index}]")
    if isinstance(value, dict):
        for name in schema.get("required", ()):
            if name not in value:
                raise _Violation(f"{path}: the field {name!r} is missing")
        properties = schema.get("properties", {})
        for name, item in value.items():
            if name in properties:
                _validate(item, properties[name], f"{path}.{name}")
            elif "additionalProperties" in schema:
                if schema["additionalProperties"] is False:
                    raise _Violation(f"{path}: the field {name!r} is not allowed")
                _validate(item, schema["additionalProperties"], f"{path}.{name}")
    for sub in schema.get("allOf", ()):
        _validate(value, sub, path)
    if "anyOf" in schema and not _count_matches(value, schema["anyOf"], path):
        raise _Violation(f"{path}: matches none of the allowed shapes")
    if "oneOf" in schema and _count_matches(value, schema["oneOf"], path) != 1:
        raise _Violation(f"{path}: does not match exactly one of the allowed shapes")


def _count_matches(value: Any, options: Sequence[Any], path: str) -> int:
    count = 0
    for option in options:
        try:
            _validate(value, option, path)
            count += 1
        except _Violation:
            pass
    return count


def validate(value: Any, schema: Any) -> None:
    """Raise LlmBadOutput unless `value` satisfies `schema` (see the module docstring for the keywords)."""
    check_schema(schema)
    try:
        _validate(value, schema, "$")
    except _Violation as exc:
        raise LlmBadOutput(f"The model's answer does not match the expected format ({exc}).") from None


def parse_json_output(text: str, schema: Optional[Mapping[str, Any]] = None) -> Any:
    """Parse the answer as the one JSON value it must be and check it against `schema` (when given)."""
    value = extract_json(text)
    if schema is not None:
        try:
            validate(value, schema)
        except LlmBadOutput as exc:
            raise LlmBadOutput(str(exc), text) from None
    return value
