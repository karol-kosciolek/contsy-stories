# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Serialize local language-model requests and supervise the worker's lifetime."""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...core.cancel import CancelToken
from . import text, worker
from .backend import DEFAULT_N_CTX, ModelSpec
from .errors import MSG_VISION_MISSING, LlmAborted, LlmCrashed, LlmError, LlmTimeout, LlmUnavailable
from .images import ImageSource, prepare_image

_FACTORY = "engine.models.llm.backend:LlamaCppBackend"
_POLL = 0.05


@dataclass
class _Request:
    gpu: bool
    deadline: float
    cancel: CancelToken | None
    interrupt: threading.Event = field(default_factory=threading.Event)

    def check(self) -> None:
        if self.cancel is not None:
            self.cancel.check()
        if self.interrupt.is_set():
            raise LlmAborted("The prompt assistant released the GPU or was closed.")
        if time.monotonic() >= self.deadline:
            raise LlmTimeout("The prompt assistant did not answer in time.")


class _IdleWorker(Exception):
    """The worker exited for idleness before accepting this request."""


class LlmClient:
    """One spawned worker, used from any application thread.

    Requests are serialized. A timeout, cancellation or GPU handover ends a worker even if its native library
    is stuck loading a model or processing an image. The next request can start a fresh worker.
    """

    def __init__(self, *, log_path: str | Path | None = None, idle_timeout: float = 20.0,
                 startup_timeout: float = 20.0, factory: str = _FACTORY,
                 options: Mapping[str, Any] | None = None) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in (idle_timeout, startup_timeout)):
            raise ValueError("Worker timeouts must be positive and finite.")
        self._log_path = str(log_path) if log_path is not None else None
        self._idle_timeout = idle_timeout
        self._startup_timeout = startup_timeout
        self._factory = factory
        self._options = dict(options or {})
        self._state = threading.Condition()
        self._active: _Request | None = None
        self._unloading = False
        self._process: worker.WorkerProcess | None = None
        self._process_gpu = False
        self._loaded = False

    def __enter__(self) -> LlmClient:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _stop(self) -> None:
        """Called by the request owner, or by an unloader after that request has ended."""
        with self._state:
            proc = self._process
            self._loaded = False
        if proc is None:
            return
        if proc.alive():
            try:
                proc.send({"op": "shutdown"})
            except (EOFError, OSError, ValueError):
                pass
            proc.process.join(0.25)
            if proc.alive():
                proc.kill()
        if proc.alive():
            raise LlmCrashed("The prompt assistant worker could not be stopped.")
        with self._state:
            proc.release()
            self._process = None
            self._process_gpu = False

    def _unload(self, gpu_only: bool) -> None:
        with self._state:
            while self._unloading:
                self._state.wait()
            active = self._active
            if gpu_only and not (self._process_gpu or active is not None and active.gpu):
                return
            self._unloading = True
            if active is not None:
                active.interrupt.set()
        try:
            with self._state:
                while self._active is not None:
                    self._state.wait()
            self._stop()
        finally:
            with self._state:
                self._unloading = False
                self._state.notify_all()

    def unload_for_gpu_handover(self) -> None:
        """Release a GPU worker before the queue gives the card to another model."""
        self._unload(gpu_only=True)

    def close(self) -> None:
        """Stop any worker. A subsequent request may open another."""
        self._unload(gpu_only=False)

    def is_loaded(self) -> bool:
        """Whether the worker that last reported a loaded model is still alive."""
        with self._state:
            return bool(self._loaded and self._process is not None and self._process.alive())

    @staticmethod
    def _receive(proc: worker.WorkerProcess, req: _Request, deadline: float) -> dict:
        while True:
            req.check()
            left = deadline - time.monotonic()
            if left <= 0:
                raise LlmTimeout("The prompt assistant worker did not respond in time.")
            if proc.poll(min(left, _POLL)):
                try:
                    message = proc.recv()
                except (EOFError, OSError, ValueError):
                    break
                req.check()
                if not isinstance(message, dict):
                    raise LlmCrashed("Invalid reply from the prompt assistant worker.")
                if message.get("op") == "bye" and message.get("reason") == "idle":
                    raise _IdleWorker()
                return message
            if not proc.alive():
                break
        raise LlmCrashed("The prompt assistant worker exited unexpectedly.")

    def _worker(self, req: _Request) -> worker.WorkerProcess:
        proc = self._process
        if proc is not None and proc.alive() and self._process_gpu == req.gpu:
            return proc
        self._stop()
        req.check()
        try:
            proc = worker.launch(self._factory, self._options, idle_timeout=self._idle_timeout,
                                 log_path=self._log_path)
        except OSError as exc:
            raise LlmUnavailable(f"Prompt assistant could not start: {exc}") from None
        with self._state:
            self._process, self._process_gpu = proc, req.gpu
        reply = self._receive(proc, req, time.monotonic() + self._startup_timeout)
        if reply.get("op") != "ready":
            raise LlmUnavailable(f"Prompt assistant could not start: {reply.get('error', 'invalid startup reply')}")
        return proc

    def _complete(self, message: dict, req: _Request) -> str:
        for attempt in range(2):
            try:
                proc = self._worker(req)
                try:
                    proc.send(message)
                except (EOFError, OSError, ValueError):
                    # The idle worker may have sent its final reply before closing the request pipe.
                    if not proc.poll(0):
                        raise LlmCrashed("The prompt assistant worker stopped accepting requests.") from None
                reply = self._receive(proc, req, req.deadline)
                break
            except _IdleWorker:
                self._stop()
                if attempt:
                    raise LlmCrashed("The prompt assistant worker exited before accepting the request.") from None
        if reply.get("id") != message["id"] or type(reply.get("ok")) is not bool:
            raise LlmCrashed("Invalid reply from the prompt assistant worker.")
        if not reply["ok"]:
            kind = reply.get("kind")
            error = reply.get("error", "The prompt assistant failed.")
            exception = {"unavailable": LlmUnavailable, "aborted": LlmAborted,
                         "engine": LlmCrashed}.get(kind, LlmError)
            raise exception(error)
        answer = reply.get("text")
        if not isinstance(answer, str):
            raise LlmCrashed("Invalid answer from the prompt assistant worker.")
        with self._state:
            self._loaded = True
        return answer

    def _answer(self, system: str, user: str, *, model_path: str | Path, mmproj_path: str | Path | None = None,
                gpu: bool = True, n_ctx: int = DEFAULT_N_CTX, n_gpu_layers: int | str = "auto",
                images: Sequence[ImageSource] = (), image_size: int = 512, schema: Mapping[str, Any] | None = None,
                max_tokens: int = 700, temperature: float = 0.8, seed: int | None = None, timeout: float = 900,
                cancel: CancelToken | None = None, keep_loaded: bool = True) -> str:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("The prompt assistant timeout must be positive and finite.")
        if schema is not None:
            text.check_schema(schema)
        req = _Request(gpu and n_gpu_layers != 0, time.monotonic() + timeout, cancel)
        with self._state:
            while self._active is not None or self._unloading:
                req.check()
                self._state.wait(_POLL)
            req.check()
            self._active = req
        completed = False
        try:
            if images and mmproj_path is None:
                raise LlmUnavailable(MSG_VISION_MISSING)
            spec = ModelSpec(str(Path(model_path).resolve()),
                             str(Path(mmproj_path).resolve()) if mmproj_path is not None else None,
                             n_ctx, req.gpu, n_gpu_layers)
            prepared = []
            for image in images:
                req.check()
                prepared.append(prepare_image(image, image_size))
            message = {"op": "complete", "id": worker.next_request_id(), "spec": asdict(spec),
                       "messages": ([{"role": "system", "content": system}] if system else [])
                       + [{"role": "user", "content": user}],
                       "images": prepared, "schema": schema,
                       "params": {"max_tokens": max_tokens, "temperature": temperature, "seed": seed}}
            req.check()
            answer = self._complete(message, req)
            req.check()
            completed = True
            return answer
        finally:
            try:
                if not completed or not keep_loaded:
                    self._stop()
            finally:
                with self._state:
                    self._active = None
                    self._state.notify_all()

    def ask(self, system: str, user: str, *, schema: Mapping[str, Any] | None = None, **options: Any) -> str:
        """Return cleaned text, or validated JSON as text when a schema is supplied."""
        answer = self._answer(system, user, schema=schema, **options)
        if schema is not None:
            return json.dumps(text.parse_json_output(answer, schema), ensure_ascii=False)
        return text.clean_output(answer)

    def ask_json(self, system: str, user: str, *, schema: Mapping[str, Any] | None = None, **options: Any) -> Any:
        """Return the parsed answer after checking its structure. Invalid output always raises LlmBadOutput."""
        answer = self._answer(system, user, schema=schema, **options)
        return text.parse_json_output(answer, schema)
