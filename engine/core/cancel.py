# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Cooperative cancellation of long engine runs.

A job owns a `CancelToken` and passes it to the pipeline; another thread (the UI, the queue) calls
`cancel()`. The pipeline calls `check()` between steps, so a run stops at the next safe point and
releases what it holds through the usual `finally` blocks.
"""
from __future__ import annotations

import threading


class Cancelled(BaseException):
    """Raised by `CancelToken.check()` when a run was cancelled.

    It derives from BaseException, not Exception, so a broad `except Exception` in a pipeline or a
    caller cannot swallow a cancellation by accident. `app.cancel` uses this very class, so the queue
    catches what the engine raises.
    """


class CancelToken:
    """A flag that one thread sets and the running pipeline polls."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        """Clear the flag, so that the token serves the next run."""
        self._event.clear()

    def check(self) -> None:
        """Raise `Cancelled` if the token was cancelled; otherwise return immediately."""
        if self._event.is_set():
            raise Cancelled()
