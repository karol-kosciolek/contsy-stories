# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""A deterministic backend for testing the process protocol without models or native libraries."""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

from engine.models.llm.errors import GenerationAborted


class FakeBackend:
    def __init__(self, *, answers=None, delay=0.0, ignore_abort=False, trace_path=None,
                 crash_on=None, delay_on=None):
        self.answers = dict(answers or {})
        self.delay, self.ignore_abort = delay, ignore_abort
        self.trace_path = Path(trace_path) if trace_path else None
        self.crash_on, self.delay_on = crash_on, delay_on
        self.spec = None
        self.loads = 0

    def _record(self, event):
        if self.trace_path is not None:
            with self.trace_path.open("a", encoding="utf-8") as trace:
                trace.write(json.dumps({"event": event, "pid": os.getpid()}) + chr(10))

    def load(self, model_path, mmproj_path=None, n_gpu_layers="auto", n_ctx=8192):
        self.loads += 1
        self.spec = dict(model_path=model_path, mmproj_path=mmproj_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
        self._record("load")
        return dict(self.spec)

    def complete(self, messages, images=(), schema=None, should_abort=None, **params):
        user = messages[-1]["content"]
        self._record("generate")
        if self.crash_on is not None and user == self.crash_on:
            os._exit(19)
        if self.delay_on is None or user == self.delay_on:
            deadline = time.monotonic() + self.delay
            while time.monotonic() < deadline:
                if not self.ignore_abort and should_abort is not None and should_abort():
                    raise GenerationAborted()
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        if user in self.answers:
            return self.answers[user]
        from PIL import Image

        pictures = []
        for data in images:
            with Image.open(io.BytesIO(data)) as image:
                pictures.append(dict(size=image.size, mode=image.mode, format=image.format))
        return json.dumps(dict(pid=os.getpid(), loads=self.loads, spec=self.spec, messages=messages,
                               images=pictures, params=params, schema=schema))

    def is_loaded(self):
        return self.spec is not None

    def memory_mb(self):
        return None

    def close(self):
        if self.spec is not None:
            self._record("close")
        self.spec = None
