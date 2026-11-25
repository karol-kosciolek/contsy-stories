# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The worker process that hosts a language model Backend, and the parent-side handle to start and talk to it.

Why a process. Measured on 8 November 2026 (llama-cpp-python 0.3.36 on Vulkan, RTX 4070 Ti SUPER 16 GB), Qwen3.5-9B
held 8.0 GB of video memory after loading and 13.3 GB after calls with 1024 px pictures, and the diffusion pipelines
need the card. A process that ends
gives every byte back at a known moment; an in-process `close()` does not promise that. The process is started
with the `spawn` method (a fresh interpreter: no forked CUDA/Vulkan state, no inherited locks), so the parent
module that calls `launch` must not run application code at import time (`if __name__ == "__main__":` as in
app/main.py), and the worker imports nothing heavy of its own: no torch, no llama_cpp until a model loads.

Protocol (two one-way pipes, plain dicts; every request has an integer id that the reply repeats):

  parent -> worker   {"op": "complete", "id", "spec": ModelSpec fields, "messages", "images", "schema",
                      "params": max_tokens, temperature, top_p, top_k, min_p, repeat_penalty, seed, stop}
                     {"op": "shutdown"}              stop the current request, close the backend and exit
  worker -> parent   {"op": "ready", "pid"}          after start-up, before any model is loaded
                     {"op": "fatal", "error"}        the backend factory could not be built
                     {"id", "ok": True, "text", "info"}
                     {"id", "ok": False, "kind", "error"}   kind: aborted, unavailable, bad_request, engine
                     {"op": "bye", "reason"}         idle, shutdown or parent-gone; nothing is processed after it

Behaviour. A reader thread takes requests off the pipe while the main thread generates, so a `shutdown` is seen
within a token; the backend polls `should_abort`. The model loads on the first request that needs it and is reused while the
`ModelSpec` matches (a text request reuses a model loaded for pictures; a picture request on a text-only load
reloads with the vision projector). After `idle_timeout` seconds without a request the worker says `bye`,
frees the backend and exits, so an idle application does not hold the card. When the parent disappears (the
pipe closes) the worker aborts what it is doing and exits as well, so a crashed application leaves no orphan
holding video memory. The process ends with `os._exit` after the backend is closed: no interpreter shutdown
(the library's finalizers print errors at that point), no hang in a native destructor.
"""
from __future__ import annotations

import importlib
import itertools
import multiprocessing
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .backend import Backend, ModelSpec
from .errors import GenerationAborted, LlmError, LlmUnavailable

_SHUTDOWN = object()
_PARENT_GONE = object()


def _import_factory(path: str) -> Any:
    module, _, attribute = path.partition(":")
    if not module or not attribute:
        raise ValueError(f"The backend factory {path!r} must look like 'package.module:Name'.")
    return getattr(importlib.import_module(module), attribute)


def _file_key(path: str) -> tuple:
    try:
        stat = Path(path).stat()
    except OSError:
        return path, None
    return path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class Worker:
    """The serving loop. `requests` and `replies` are multiprocessing connections (or anything with the same
    recv/send); the loop runs in the calling thread and returns the reason it ended."""

    def __init__(self, backend: Backend, requests: Any, replies: Any, idle_timeout: Optional[float] = None) -> None:
        self._backend, self._requests, self._replies = backend, requests, replies
        self._idle_timeout = idle_timeout
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._lock = threading.Lock()
        self._current: Optional[Tuple[int, threading.Event]] = None
        self._loaded: Optional[Tuple[Tuple, Optional[Tuple], Dict[str, Any]]] = None  # (key, projector, info)
        self._parent_gone = False
        self._stopping = threading.Event()

    # --- reader thread ---------------------------------------------------------------------------------------

    def _read(self) -> None:
        while True:
            try:
                message = self._requests.recv()
            except (EOFError, OSError):
                self._abort_current()
                self._queue.put(_PARENT_GONE)
                return
            operation = message.get("op") if isinstance(message, dict) else None
            if operation == "shutdown":
                self._abort_current()
                self._queue.put(_SHUTDOWN)
                return
            else:
                self._queue.put(message)

    def _abort_current(self) -> None:
        self._stopping.set()
        with self._lock:
            if self._current is not None:
                self._current[1].set()

    # --- sending ---------------------------------------------------------------------------------------------

    def _send(self, message: Dict[str, Any]) -> None:
        try:
            self._replies.send(message)
        except (BrokenPipeError, EOFError, OSError, ValueError):
            self._parent_gone = True

    def _error(self, request_id: Any, kind: str, error: str) -> None:
        self._send({"id": request_id, "ok": False, "kind": kind, "error": error})

    # --- the model -------------------------------------------------------------------------------------------

    def _close_backend(self) -> None:
        self._loaded = None
        try:
            self._backend.close()
        except Exception:  # noqa: BLE001 - the process ends next; there is nothing better to do
            traceback.print_exc()

    def _ensure_loaded(self, spec: ModelSpec, want_vision: bool) -> Tuple[float, Dict[str, Any]]:
        """Load the model unless a matching one is in memory; returns (seconds spent loading, load info)."""
        key = (_file_key(spec.model_path), spec.n_ctx, spec.gpu, spec.n_gpu_layers)
        projector = spec.mmproj_path if want_vision else None
        projector_key = _file_key(projector) if projector is not None else None
        if self._loaded is not None:
            loaded_key, loaded_projector, info = self._loaded
            if loaded_key == key and (not want_vision or loaded_projector == projector_key):
                return 0.0, info
            self._close_backend()
        started = time.monotonic()
        try:
            info = self._backend.load(model_path=spec.model_path, mmproj_path=projector,
                                      n_gpu_layers=spec.n_gpu_layers if spec.gpu else 0, n_ctx=spec.n_ctx)
        except BaseException:
            self._close_backend()
            raise
        self._loaded = (key, projector_key, dict(info or {}))
        return time.monotonic() - started, self._loaded[2]

    def _memory(self) -> Optional[float]:
        try:
            return self._backend.memory_mb()
        except Exception:  # noqa: BLE001
            return None

    # --- requests --------------------------------------------------------------------------------------------

    def _complete(self, message: Dict[str, Any]) -> None:
        request_id = message.get("id")
        with self._lock:
            cancelled = self._stopping.is_set()
            if not cancelled:
                abort = threading.Event()
                self._current = (request_id, abort)
        if cancelled:
            self._error(request_id, "aborted", "The request was cancelled before it started.")
            return
        started = time.monotonic()
        try:
            spec = ModelSpec(**message["spec"])
            images = list(message.get("images") or ())
            load_s, load_info = self._ensure_loaded(spec, bool(images))
            if abort.is_set():
                raise GenerationAborted()
            generating = time.monotonic()
            text = self._backend.complete(message["messages"], images=images, schema=message.get("schema"),
                                          should_abort=abort.is_set, **(message.get("params") or {}))
            self._send({"id": request_id, "ok": True, "text": text,
                        "info": {"load_s": round(load_s, 3), "generate_s": round(time.monotonic() - generating, 3),
                                 "total_s": round(time.monotonic() - started, 3), "memory_mb": self._memory(),
                                 "loaded": True, "load": load_info}})
        except GenerationAborted:
            self._error(request_id, "aborted", "The generation was stopped.")
        except LlmUnavailable as exc:
            self._error(request_id, "unavailable", str(exc))
        except LlmError as exc:  # a refusal of this request; the model in memory is fine
            self._error(request_id, "bad_request", str(exc))
        except BaseException as exc:  # noqa: BLE001 - unknown state of the runtime: start from scratch next time
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self._close_backend()
            self._error(request_id, "engine", f"{type(exc).__name__}: {exc}"[:400])
        finally:
            with self._lock:
                self._current = None

    def _dispatch(self, message: Any) -> None:
        operation = message.get("op") if isinstance(message, dict) else None
        if operation == "complete":
            self._complete(message)
        else:
            self._error(message.get("id") if isinstance(message, dict) else None, "bad_request",
                        f"Unknown request {operation!r}.")

    # --- the loop --------------------------------------------------------------------------------------------

    def serve(self) -> str:
        self._send({"op": "ready", "pid": os.getpid()})
        threading.Thread(target=self._read, name="llm-worker-reader", daemon=True).start()
        reason = "idle"
        try:
            while not self._parent_gone:
                try:
                    item = self._queue.get(timeout=self._idle_timeout)
                except queue.Empty:
                    break
                if item is _SHUTDOWN:
                    reason = "shutdown"
                    break
                if item is _PARENT_GONE:
                    reason = "parent-gone"
                    break
                self._dispatch(item)
            else:
                reason = "parent-gone"
        finally:
            self._send({"op": "bye", "reason": reason})  # before the slow part: the parent learns at once
            self._close_backend()
        return reason


def _redirect_output(log_path: Optional[str]) -> None:
    """Send stdout and stderr (the C-level ones too) to a log file, so a crash can be explained afterwards."""
    if not log_path:
        return
    try:
        descriptor = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
        os.close(descriptor)
        sys.stdout = sys.stderr = open(2, "w", encoding="utf-8", errors="replace", buffering=1, closefd=False)
    except OSError:
        pass


def _hard_exit(code: int) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    os._exit(code)


def worker_main(requests: Any, replies: Any, spec: Dict[str, Any]) -> None:
    """Entry point of the worker process (the target given to multiprocessing)."""
    os.environ.update(spec.get("env") or {})
    _redirect_output(spec.get("log_path"))
    try:
        backend = _import_factory(spec["factory"])(**(spec.get("options") or {}))
    except BaseException as exc:  # noqa: BLE001
        try:
            replies.send({"op": "fatal", "error": f"{type(exc).__name__}: {exc}"[:400]})
        except Exception:  # noqa: BLE001
            pass
        traceback.print_exc()
        _hard_exit(2)
        return
    code = 0
    try:
        Worker(backend, requests, replies, idle_timeout=spec.get("idle_timeout")).serve()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    _hard_exit(code)


# --- the parent's side ---------------------------------------------------------------------------------------

_IDS = itertools.count(1)


def next_request_id() -> int:
    return next(_IDS)


@dataclass
class WorkerProcess:
    """A started worker: the process and the two pipe ends the parent keeps."""
    process: Any
    requests: Any  # parent writes
    replies: Any  # parent reads

    def alive(self) -> bool:
        return self.process.is_alive()

    def send(self, message: Dict[str, Any]) -> None:
        self.requests.send(message)

    def poll(self, timeout: float) -> bool:
        """True when a message (or the end of the pipe) can be read; False on timeout or a broken pipe."""
        try:
            return bool(self.replies.poll(timeout))
        except (OSError, EOFError, ValueError):
            return False

    def recv(self) -> Dict[str, Any]:
        return self.replies.recv()

    def kill(self, wait: float = 5.0) -> None:
        if self.process.is_alive():
            self.process.kill()
        self.process.join(wait)

    def release(self) -> None:
        """Close the pipes and the process handle (after the process has ended)."""
        for connection in (self.requests, self.replies):
            try:
                connection.close()
            except Exception:  # noqa: BLE001
                pass
        if not self.process.is_alive():
            try:
                self.process.close()
            except ValueError:
                pass


def launch(factory: str, options: Optional[Dict[str, Any]] = None, *, idle_timeout: Optional[float] = None,
           log_path: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> WorkerProcess:
    """Start a worker process hosting `factory(**options)` (an import path like 'package.module:Name')."""
    context = multiprocessing.get_context("spawn")
    request_reader, request_writer = context.Pipe(duplex=False)
    reply_reader, reply_writer = context.Pipe(duplex=False)
    spec = {"factory": factory, "options": dict(options or {}), "idle_timeout": idle_timeout,
            "log_path": log_path, "env": dict(env or {})}
    process = context.Process(target=worker_main, args=(request_reader, reply_writer, spec),
                              name="contsy-llm-worker", daemon=True)
    try:
        process.start()
    except BaseException:
        for connection in (request_reader, request_writer, reply_reader, reply_writer):
            connection.close()
        raise
    # the child holds its own duplicates; closing ours is what lets either side see the other end go away
    request_reader.close()
    reply_writer.close()
    return WorkerProcess(process, request_writer, reply_reader)
