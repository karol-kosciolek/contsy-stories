# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Errors of the in-process language model and their user-facing messages; `str(error)` of every `LlmError`
is fit for the UI.

Cancellation is not an `LlmError`: a cancelled run raises `engine.core.cancel.Cancelled` (a BaseException),
like every other long engine run, so a broad `except Exception` in the product cannot swallow it.
"""
from __future__ import annotations

MSG_MODEL_MISSING = "The prompt assistant model is missing."
MSG_VISION_MISSING = "The prompt assistant vision file is missing."
MSG_RUNTIME_MISSING = ("The language model runtime is not installed (the llama-cpp-python package). "
                       "Run install.bat to install it.")
MSG_COULD_NOT_START = "Prompt assistant could not start: {detail}"


class LlmError(Exception):
    """Base of every failure of the language model; the message is meant for the user."""


class LlmUnavailable(LlmError):
    """The model, its vision file or the llama-cpp-python runtime is not there, or cannot be used here."""


class LlmTimeout(LlmError):
    """A request ran longer than its time limit; the generation was stopped (the worker restarts if needed)."""


class LlmCrashed(LlmError):
    """The worker process died during a request. The next request starts a fresh worker."""


class LlmAborted(LlmError):
    """A request was stopped from outside because the GPU was handed over or the client was closed.

    A user cancellation is `engine.core.cancel.Cancelled`, not this."""


class LlmBadOutput(LlmError):
    """The answer is not valid JSON or does not satisfy the schema. Nothing permissive is ever returned instead.

    `answer` keeps the (cleaned, shortened) text for diagnostics; it is not part of `str(error)`."""

    def __init__(self, message: str, answer: str = "") -> None:
        super().__init__(message)
        self.answer = answer[:500]


class GenerationAborted(Exception):
    """Raised by a backend when `should_abort()` turned true during a generation.

    It stays between the backend and the worker; the application sees `Cancelled` or `LlmAborted`."""
