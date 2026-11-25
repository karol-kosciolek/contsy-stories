# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The language model backend: the `Backend` protocol and `LlamaCppBackend`, which runs llama.cpp in process.

`LlamaCppBackend` drives the llama-cpp-python library (MIT, abetlen; Python bindings of llama.cpp, MIT, ggml-org)
through its public API (read from the 0.3.36 sources; nothing of the library is copied).

The backend is meant to live in a worker process (engine/models/llm/worker.py). Measured on 8 November 2026 with
llama-cpp-python 0.3.36 (the official Vulkan wheel), Qwen3.5-9B Q6_K and its F16 projector on an RTX 4070 Ti SUPER
16 GB: 8.0 GB of video memory in use after the load and 13.3 GB after calls with 1024 px pictures, so the only
reliable way to give the card back is to end the process. Behaviour:

* Text requests use the ChatML prompt with the empty think block (text.build_chatml) through
  `Llama.create_completion`, with a context of 8192 tokens, top-p 0.95, top-k 40, min-p 0.05 and no repeat
  penalty. Each request starts from an empty context (`Llama.reset`).
* Pictures go through `MTMDChatHandler` (the multimodal handler of llama-cpp-python) as data URLs placed before
  the text of the last user message. The handler renders the model's own chat template; `enable_thinking=False`
  makes the Qwen3.5 template emit the same empty think block.
* JSON answers are forced by a grammar built from the schema (`LlamaGrammar.from_json_schema`).
* The seed is random per request. llama-cpp-python would otherwise derive the seed of every request from the
  previous one, so the first request after each start would always be the same.
* Generation is streamed so that the caller can stop it between tokens (`should_abort`).
* `n_gpu_layers` comes from fitting.plan_gpu_layers, which computes the offload from the GGUF header and the free
  video memory; a load that fails for lack of memory steps down (fitting.ladder). CPU mode (`n_gpu_layers=0`)
  also switches off `op_offload`, so that llama.cpp does not borrow the card for the large matrix products of a
  prompt (the product pauses the GPU for games and for running jobs).

The library is imported lazily, inside `load`, because importing it loads native libraries into the process:
the application itself never imports it. When it is missing, `LlmUnavailable` says so.
"""
from __future__ import annotations

import base64
import gc
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Union

from . import fitting, text
from .errors import (MSG_COULD_NOT_START, MSG_MODEL_MISSING, MSG_RUNTIME_MISSING, MSG_VISION_MISSING,
                     GenerationAborted, LlmError, LlmUnavailable)

DEFAULT_N_CTX = 8192  # tokens of context for text; picture requests fit in it too
MSG_TOO_LONG = "The request is too long for the prompt assistant's context window."


@dataclass(frozen=True)
class ModelSpec:
    """Which files to run and how. Compared field by field to decide whether a loaded model can be reused."""
    model_path: str
    mmproj_path: Optional[str] = None  # the vision projector; loaded only when a request carries pictures
    n_ctx: int = DEFAULT_N_CTX
    gpu: bool = True  # False: CPU only (the product's "GPU paused or busy" mode)
    n_gpu_layers: Union[int, str] = "auto"  # "auto" fits to the free video memory; an integer forces that count


class Backend(Protocol):
    """What the worker needs from a language model runtime. LlamaCppBackend implements it (and the tests' FakeBackend)."""

    def load(self, model_path: str, mmproj_path: Optional[str] = None, n_gpu_layers: Union[int, str] = "auto",
             n_ctx: int = DEFAULT_N_CTX) -> Dict[str, Any]:
        """Load the model (and prepare the vision projector when a path is given). Returns facts about the load."""

    def complete(self, messages: Sequence[Mapping[str, str]], images: Sequence[bytes] = (),
                 schema: Optional[Mapping[str, Any]] = None, max_tokens: int = 700, temperature: float = 0.8,
                 top_p: float = 0.95, top_k: int = 40, min_p: float = 0.05, repeat_penalty: float = 1.0,
                 seed: Optional[int] = None, stop: Sequence[str] = (),
                 should_abort: Optional[Callable[[], bool]] = None) -> str:
        """Answer the conversation. `images` are encoded picture files that belong to the last user message and
        come before its text. With a `schema` a grammar constrains generation. Raises GenerationAborted as soon
        as `should_abort()` is true. The client cleans and validates the answer."""

    def close(self) -> None:
        """Free the model. Safe to call twice."""

    def is_loaded(self) -> bool: ...

    def memory_mb(self) -> Optional[float]:
        """Memory held by the process (None when unknown), reported to the client for its recycling rule."""


def runtime_problem() -> Optional[str]:
    """None when llama-cpp-python can be imported by the worker, otherwise why not. Does not import it."""
    if "llama_cpp" in sys.modules:
        return None
    try:
        found = importlib.util.find_spec("llama_cpp")
    except (ImportError, ValueError):
        found = None
    return None if found is not None else MSG_RUNTIME_MISSING


def _data_url(data: bytes) -> str:
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _completion_piece(chunk: Mapping[str, Any]) -> str:
    return chunk["choices"][0].get("text") or ""


def _chat_piece(chunk: Mapping[str, Any]) -> str:
    return (chunk["choices"][0].get("delta") or {}).get("content") or ""


class LlamaCppBackend:
    """llama.cpp through llama-cpp-python. One model at a time; used from a single thread."""

    def __init__(self, *, flash_attn: bool = False, n_batch: int = 512, n_ubatch: int = 512,
                 n_threads: Optional[int] = None, reserve_mib: float = fitting.RESERVE_MIB,
                 unknown_vram: str = "all", free_vram: Optional[Callable[[], Optional[float]]] = None,
                 verbose: bool = False, max_gpu_mib: Optional[float] = None) -> None:
        # flash_attn stays off as in the 8 November 2026 measurement; turning it on is a possible saving of video
        # memory for large pictures that has not been tried on a free card.
        self.flash_attn, self.n_batch, self.n_ubatch, self.n_threads = flash_attn, n_batch, n_ubatch, n_threads
        self.reserve_mib, self.unknown_vram, self.verbose = reserve_mib, unknown_vram, verbose
        self._free_vram = free_vram or fitting.query_free_vram_mib
        self.max_gpu_mib = max_gpu_mib
        self._lib: Optional[SimpleNamespace] = None
        self._llm: Any = None
        self._handler: Any = None
        self._info: Dict[str, Any] = {}

    # --- loading ---------------------------------------------------------------------------------------------

    def _import_runtime(self, vision: bool) -> SimpleNamespace:
        try:
            import llama_cpp
            from llama_cpp.llama_grammar import LlamaGrammar
        except ImportError as exc:
            raise LlmUnavailable(f"{MSG_RUNTIME_MISSING} ({exc})") from None
        handler = None
        if vision:
            try:
                from llama_cpp.llama_chat_format import MTMDChatHandler as handler
            except ImportError as exc:
                raise LlmUnavailable("The installed llama-cpp-python has no multimodal chat handler "
                                     f"(MTMDChatHandler); version 0.3.36 or newer is needed. ({exc})") from None
        return SimpleNamespace(module=llama_cpp, Llama=llama_cpp.Llama, LlamaGrammar=LlamaGrammar,
                               MTMDChatHandler=handler, version=getattr(llama_cpp, "__version__", "?"))

    def _plan(self, lib: SimpleNamespace, model: Path, mmproj: Optional[Path], n_gpu_layers: Union[int, str],
              n_ctx: int) -> fitting.GpuPlan:
        """The offload to try first. An integer is taken as given; "auto" is computed."""
        try:
            info = fitting.read_gguf_info_cached(model)
            fp = fitting.footprint(info, n_ctx)
        except fitting.GGUFHeaderError as exc:
            raise LlmUnavailable(f"The prompt assistant model file is damaged or incomplete ({exc}).") from None
        if n_gpu_layers != "auto":
            n = int(n_gpu_layers)
            return fitting.GpuPlan(n, fp.n_layers, 0.0, None, "forced by the caller")
        supports = getattr(lib.module, "llama_supports_gpu_offload", None)
        if supports is not None and not supports():
            return fitting.GpuPlan(0, fp.n_layers, 0.0, None, "this llama.cpp build has no GPU support")
        vision_mib = 0.0
        if mmproj is not None:
            vision_mib = mmproj.stat().st_size / fitting.MIB + fitting.VISION_RUNTIME_MIB
        free = self._free_vram()
        if self.max_gpu_mib is not None:
            free = self.max_gpu_mib if free is None else min(self.max_gpu_mib, free)
        return fitting.plan_gpu_layers(fp, free, reserve_mib=self.reserve_mib, vision_mib=vision_mib,
                                       unknown=self.unknown_vram)

    def load(self, model_path: str, mmproj_path: Optional[str] = None, n_gpu_layers: Union[int, str] = "auto",
             n_ctx: int = DEFAULT_N_CTX) -> Dict[str, Any]:
        self.close()
        model = Path(model_path)
        if not model.is_file():
            raise LlmUnavailable(MSG_MODEL_MISSING)
        mmproj = Path(mmproj_path) if mmproj_path else None
        if mmproj is not None and not mmproj.is_file():
            raise LlmUnavailable(MSG_VISION_MISSING)
        started = time.monotonic()
        lib = self._import_runtime(vision=mmproj is not None)
        plan = self._plan(lib, model, mmproj, n_gpu_layers, int(n_ctx))
        tries = [plan.n_gpu_layers] if n_gpu_layers != "auto" else fitting.ladder(plan)
        failures: List[str] = []
        used = None
        for n in tries:
            options: Dict[str, Any] = dict(model_path=str(model), n_gpu_layers=n, n_ctx=int(n_ctx),
                                           n_batch=self.n_batch, n_ubatch=self.n_ubatch,
                                           flash_attn=self.flash_attn, verbose=self.verbose)
            if n == 0:
                options.update(op_offload=False, offload_kqv=False)
            if self.n_threads:
                options["n_threads"] = self.n_threads
            try:
                self._llm = lib.Llama(**options)
                used = n
                break
            except (ValueError, RuntimeError, OSError, MemoryError) as exc:
                failures.append(f"{n}: {exc}")
                self._llm = None
                gc.collect()
        if self._llm is None:
            raise LlmError(MSG_COULD_NOT_START.format(detail="; ".join(failures)[:300]))
        if mmproj is not None:
            try:
                self._handler = lib.MTMDChatHandler(clip_model_path=str(mmproj), verbose=self.verbose,
                                                    use_gpu=used != 0)
            except Exception:
                self.close()
                raise
        self._lib = lib
        self._info = {"n_gpu_layers": used, "n_layers": plan.n_layers, "plan": plan.note,
                      "free_mib": plan.free_mib, "estimated_mib": round(plan.estimated_mib, 1),
                      "vision": self._handler is not None, "n_ctx": int(n_ctx), "runtime": lib.version,
                      "attempts": failures, "load_s": round(time.monotonic() - started, 2)}
        return dict(self._info)

    # --- generating ------------------------------------------------------------------------------------------

    def _grammar(self, schema: Mapping[str, Any]) -> Any:
        try:
            return self._lib.LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
        except (TypeError, ValueError, AssertionError, KeyError) as exc:
            raise LlmError(f"The JSON schema cannot be turned into a grammar: {exc}") from None

    @staticmethod
    def _vision_messages(messages: Sequence[Mapping[str, str]], images: Sequence[bytes]) -> List[Dict[str, Any]]:
        users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not users:
            raise LlmError("A request with pictures needs a user message.")
        result: List[Dict[str, Any]] = []
        for index, message in enumerate(messages):
            content = text.neutralize_special_tokens(message["content"])
            if index == users[-1]:
                parts: List[Dict[str, Any]] = [{"type": "image_url", "image_url": {"url": _data_url(data)}}
                                               for data in images]
                parts.append({"type": "text", "text": content})
                result.append({"role": "user", "content": parts})
            else:
                result.append({"role": message["role"], "content": content})
        return result

    @staticmethod
    def _drain(stream: Iterator[Mapping[str, Any]], pick: Callable[[Mapping[str, Any]], str],
               should_abort: Optional[Callable[[], bool]]) -> str:
        """Collect a streamed answer, checking for an abort before the first token and after every chunk."""
        parts: List[str] = []
        try:
            if should_abort is not None and should_abort():
                raise GenerationAborted()
            for chunk in stream:
                if should_abort is not None and should_abort():
                    raise GenerationAborted()
                piece = pick(chunk)
                if piece:
                    parts.append(piece)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        return "".join(parts)

    def complete(self, messages: Sequence[Mapping[str, str]], images: Sequence[bytes] = (),
                 schema: Optional[Mapping[str, Any]] = None, max_tokens: int = 700, temperature: float = 0.8,
                 top_p: float = 0.95, top_k: int = 40, min_p: float = 0.05, repeat_penalty: float = 1.0,
                 seed: Optional[int] = None, stop: Sequence[str] = (),
                 should_abort: Optional[Callable[[], bool]] = None) -> str:
        if self._llm is None:
            raise LlmError("The prompt assistant is not loaded.")
        if images and self._handler is None:
            raise LlmUnavailable(MSG_VISION_MISSING)
        if should_abort is not None and should_abort():
            raise GenerationAborted()
        grammar = self._grammar(schema) if schema is not None else None
        # llama-cpp-python seeds a request without a seed from the previous one (a fixed sequence per start)
        seed = random.SystemRandom().randrange(0, 2 ** 31 - 1) if seed is None else int(seed) & 0x7FFFFFFF
        options: Dict[str, Any] = dict(max_tokens=int(max_tokens), temperature=float(temperature),
                                       top_p=float(top_p), top_k=int(top_k), min_p=float(min_p),
                                       repeat_penalty=float(repeat_penalty), seed=seed,
                                       stop=list(dict.fromkeys([*stop, *text.STOP_STRINGS])),
                                       grammar=grammar, stream=True)
        try:
            if images:
                stream = self._handler(llama=self._llm, messages=self._vision_messages(messages, images),
                                       enable_thinking=False, **options)
                raw = self._drain(stream, _chat_piece, should_abort)
            else:
                self._llm.reset()
                stream = self._llm.create_completion(prompt=text.build_chatml(messages), **options)
                raw = self._drain(stream, _completion_piece, should_abort)
        except ValueError as exc:
            lowered = str(exc).lower()
            if "n_ctx" in lowered or "context window" in lowered:
                raise LlmError(MSG_TOO_LONG) from None
            raise
        return raw

    # --- state -----------------------------------------------------------------------------------------------

    def is_loaded(self) -> bool:
        return self._llm is not None

    def memory_mb(self) -> Optional[float]:
        """Working set of this process. Per-process video memory is not available on Windows (WDDM) from NVML."""
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:  # noqa: BLE001
            return None

    def close(self) -> None:
        llm, self._llm, self._handler, self._lib = self._llm, None, None, None
        if llm is not None:
            try:
                llm.close()  # frees the context and the model, and the vision projector registered on it
            except Exception:  # noqa: BLE001 - nothing more can be done; the process ends right after
                pass
        gc.collect()
