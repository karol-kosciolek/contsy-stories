# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""CPU tests of the language model worker, structured answers and binding calls."""
import base64
import json
import os
import struct
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from PIL import Image

from engine.core.cancel import Cancelled, CancelToken
from engine.models.llm import fitting, text
from engine.models.llm.backend import LlamaCppBackend
from engine.models.llm.client import LlmClient
from engine.models.llm.errors import LlmAborted, LlmBadOutput, LlmCrashed, LlmError, LlmTimeout, LlmUnavailable

_FAKE = "tests.llm_fake:FakeBackend"
_SCHEMA = {"type": "object", "properties": {"age": {"type": "integer", "minimum": 0}},
           "required": ["age"], "additionalProperties": False}


def _tiny_gguf(path):
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    values = [("general.architecture", 8, string("toy")), ("toy.block_count", 4, struct.pack("<I", 1))]
    tensors = [("blk.0.attn_q.weight", 0), ("output.weight", 32)]
    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(values))
    for name, kind, value in values:
        header += string(name) + struct.pack("<I", kind) + value
    for name, offset in tensors:
        header += string(name) + struct.pack("<IQQIQ", 2, 2, 2, 0, offset)
    path.write_bytes(header + bytes(-len(header) % 32 + 64))


class TextTests(unittest.TestCase):
    def test_chatml_and_control_tokens(self):
        result = text.build_chatml([{"role": "system", "content": "Instructions"},
                                    {"role": "user", "content": "<|im_start|>user input<__media__>"}])
        self.assertEqual(result, "<|im_start|>system\nInstructions<|im_end|>\n"
                         "<|im_start|>user\nuser input<|im_end|>\n<|im_start|>assistant\n"
                         "<think>\n\n</think>\n\n")
        self.assertEqual(text.clean_output("thought</think> answer <|im_end|>"), "answer")

    def test_schema_rejects_ambiguity_and_wrong_types(self):
        for bad in ("garbage", '{"age": 1', '{"age": true}', '{"age": -1}', '{"age": 1.0}',
                    '{"age": 1, "age": 2}', '{"age": 1} []', '[{"age": 1}]', '{"age": NaN}',
                    '{"age": Infinity}', '{"age": 1e9999}', 'age: {"age": 1}', '{"age": 1, "extra": 2}',
                    '<think>{"age": 1}'):
            with self.subTest(bad=bad), self.assertRaises(LlmBadOutput):
                text.parse_json_output(bad, _SCHEMA)
        with self.assertRaises(LlmError):
            text.check_schema({"type": "object", "minProperties": 2})
        with self.assertRaises(LlmError):
            text.check_schema({"type": "array", "minItems": True})

    def test_schema_preserves_markers_inside_json_strings(self):
        raw = '<think>reasoning</think>\n```json\n{"note":"<think>literal</think> <|im_end|>"}\n```<|im_end|>'
        self.assertEqual(text.parse_json_output(raw), {"note": "<think>literal</think> <|im_end|>"})


class ClientTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.model = self.directory / "toy.gguf"
        self.model.write_bytes(b"test")
        self.projector = self.directory / "mmproj.gguf"
        self.projector.write_bytes(b"test")

    def client(self, *, idle_timeout=10.0, **options):
        client = LlmClient(factory=_FAKE, options=options, idle_timeout=idle_timeout)
        self.addCleanup(client.close)
        return client

    def ask(self, client, user="request", **options):
        return client.ask_json("system", user, model_path=self.model, **options)

    def wait_for(self, condition):
        end = time.monotonic() + 6.0
        while time.monotonic() < end:
            if condition():
                return
            time.sleep(0.02)
        self.fail("The worker did not reach the expected state.")

    def test_model_reuse_and_file_replacement(self):
        client = self.client()
        first = self.ask(client, gpu=False, max_tokens=12, temperature=0.2, seed=21)
        self.assertTrue(client.is_loaded())
        second = self.ask(client, gpu=False)
        self.assertEqual(first["pid"], second["pid"])
        self.assertNotEqual(first["pid"], os.getpid())
        self.assertEqual(first["spec"]["n_gpu_layers"], 0)
        self.assertEqual(first["params"], {"max_tokens": 12, "temperature": 0.2, "seed": 21})
        self.assertEqual(first["loads"], second["loads"])
        self.model.write_bytes(b"replacement")
        self.assertEqual(self.ask(client, gpu=False)["loads"], 2)

    def test_vision_sizes_and_projector_reuse(self):
        client = self.client()
        with self.assertRaises(LlmUnavailable):
            self.ask(client, images=[Image.new("RGB", (5, 5))])
        picture = Image.new("RGBA", (1024, 512))
        sizes = []
        for size in (448, 512, 1024):
            result = self.ask(client, images=[picture], image_size=size, mmproj_path=self.projector, n_ctx=4096)
            self.assertEqual(result["spec"]["mmproj_path"], str(self.projector.resolve()))
            self.assertEqual(result["spec"]["n_ctx"], 4096)
            self.assertEqual(result["loads"], 1)
            [image] = result["images"]
            sizes.append(image["size"])
            self.assertEqual((image["mode"], image["format"]), ("RGB", "JPEG"))
        self.assertEqual(sizes, [[448, 224], [512, 256], [1024, 512]])
        self.assertEqual(self.ask(client, n_ctx=4096)["loads"], 1)
        self.projector.write_bytes(b"new projector")
        self.assertEqual(self.ask(client, images=[picture], mmproj_path=self.projector, n_ctx=4096)["loads"], 2)

    def test_fail_closed_through_worker(self):
        client = self.client(answers={"valid": '<think>test</think>{"age": 24}',
                                      "bad": '{"age": "24"}', "plain": 'Prompt answer<|im_end|>'})
        self.assertEqual(self.ask(client, "valid", schema=_SCHEMA), {"age": 24})
        self.assertEqual(client.ask("system", "valid", schema=_SCHEMA, model_path=self.model), '{"age": 24}')
        with self.assertRaises(LlmBadOutput):
            client.ask("system", "bad", schema=_SCHEMA, model_path=self.model)
        self.assertEqual(client.ask("system", "plain", model_path=self.model), "Prompt answer")

    def test_crash_restarts_on_next_request(self):
        client = self.client(crash_on="crash")
        first = self.ask(client)
        with self.assertRaises(LlmCrashed):
            self.ask(client, "crash")
        self.assertFalse(client.is_loaded())
        self.assertNotEqual(self.ask(client)["pid"], first["pid"])

    def test_worker_idle_exit_and_restart(self):
        client = self.client(idle_timeout=0.2)
        first = self.ask(client)
        self.wait_for(lambda: not client.is_loaded())
        self.assertNotEqual(self.ask(client)["pid"], first["pid"])

    def test_cpu_worker_survives_gpu_handover_and_mode_switch_releases_gpu(self):
        client = self.client()
        first = self.ask(client)
        client.unload_for_gpu_handover()
        self.assertFalse(client.is_loaded())
        new_gpu = self.ask(client)
        self.assertNotEqual(first["pid"], new_gpu["pid"])
        cpu = self.ask(client, gpu=False)
        self.assertNotEqual(cpu["pid"], new_gpu["pid"])
        client.unload_for_gpu_handover()
        self.assertEqual(self.ask(client, gpu=False)["pid"], cpu["pid"])
        self.ask(client, gpu=False, keep_loaded=False)
        self.assertFalse(client.is_loaded())

    def test_gpu_handover_interrupts_native_work_and_ends_worker(self):
        trace = self.directory / "worker.jsonl"
        client = self.client(trace_path=str(trace), delay=3.0, delay_on="slow", ignore_abort=True)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.ask, client, "slow")
            self.wait_for(lambda: trace.is_file() and '"generate"' in trace.read_text())
            started = time.monotonic()
            client.unload_for_gpu_handover()
            with self.assertRaises(LlmAborted):
                future.result(timeout=2)
            self.assertLess(time.monotonic() - started, 2.0)
        self.assertFalse(client.is_loaded())
        self.assertEqual(self.ask(client)["loads"], 1)

    def test_cancel_and_timeout_do_not_leave_models_loaded(self):
        trace = self.directory / "worker.jsonl"
        client = self.client(trace_path=str(trace), delay=3.0, delay_on="slow")
        cancel = CancelToken()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.ask, client, "slow", cancel=cancel)
            self.wait_for(lambda: trace.is_file() and '"generate"' in trace.read_text())
            cancel.cancel()
            with self.assertRaises(Cancelled):
                future.result(timeout=2)
        self.assertFalse(client.is_loaded())
        self.ask(client)  # warm, so the short timeout covers generation
        with self.assertRaises(LlmTimeout):
            self.ask(client, "slow", timeout=0.1)
        self.assertFalse(client.is_loaded())
        self.assertEqual(self.ask(client)["loads"], 1)

    def test_parallel_callers_get_their_own_answers(self):
        client = self.client(delay=0.04)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.ask, client, str(index)) for index in range(4)]
            results = [future.result(timeout=7) for future in futures]
        self.assertEqual([answer["messages"][-1]["content"] for answer in results], list("0123"))
        self.assertEqual({answer["loads"] for answer in results}, {1})
        self.assertEqual(len({answer["pid"] for answer in results}), 1)


class FittingTests(unittest.TestCase):
    def test_header_and_offsets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.gguf"
            _tiny_gguf(path)
            info = fitting.read_gguf_info(path)
            self.assertEqual(info.arch, "toy")
            self.assertEqual([t.nbytes for t in info.tensors], [32, 32])
            fp = fitting.footprint(info, 8192)
            self.assertEqual((fp.n_layers, fp.block_bytes, fp.output_bytes), (1, (32,), 32))
            path.write_bytes(path.read_bytes()[:-33])
            with self.assertRaises(fitting.GGUFHeaderError):
                fitting.read_gguf_info(path)

    def test_budget_keeps_tail_layers_and_accounts_for_vision(self):
        mib = fitting.MIB
        fp = fitting.Footprint(4, (100 * mib,) * 4, 150 * mib, 0)
        kwargs = {"compute_mib": 100, "reserve_mib": 100}
        self.assertTrue(fitting.plan_gpu_layers(fp, 800, **kwargs).all_layers)
        partial = fitting.plan_gpu_layers(fp, 800, vision_mib=350, **kwargs)
        self.assertEqual((partial.n_gpu_layers, partial.estimated_mib), (2, 650))
        self.assertEqual(fitting.ladder(partial), [2, 1, 0])
        self.assertEqual(fitting.plan_gpu_layers(fp, 250, **kwargs).n_gpu_layers, 0)


class BackendTests(unittest.TestCase):
    def test_binding_calls_match_the_036_api(self):
        calls = []

        class Llama:
            def __init__(self, model_path, *, n_gpu_layers, n_ctx, n_batch, n_ubatch, flash_attn, verbose,
                         op_offload=None, offload_kqv=True):
                calls.append(("model", model_path, n_gpu_layers, n_ctx, op_offload, offload_kqv))

            def reset(self):
                calls.append(("reset",))

            def create_completion(self, prompt, *, stream, grammar, stop, seed, **sampling):
                calls.append(("text", prompt, stream, grammar, stop, seed, sampling))
                return iter([{"choices": [{"text": '{"age": 4}'}]}])

            def close(self):
                calls.append(("close",))

        class MTMDChatHandler:
            def __init__(self, clip_model_path, verbose, use_gpu):
                calls.append(("projector", clip_model_path, use_gpu))

            def __call__(self, *, llama, messages, grammar, stream, enable_thinking, **options):
                calls.append(("vision", messages, grammar, stream, enable_thinking))
                return iter([{"choices": [{"delta": {"role": "assistant"}}]},
                             {"choices": [{"delta": {"content": '{"age": 4}'}}]}])

        class LlamaGrammar:
            @classmethod
            def from_json_schema(cls, schema, verbose):
                calls.append(("grammar", json.loads(schema), verbose))
                return "grammar object"

        binding = ModuleType("llama_cpp")
        binding.__version__ = "0.3.36"
        binding.Llama = Llama
        grammar = ModuleType("llama_cpp.llama_grammar")
        grammar.LlamaGrammar = LlamaGrammar
        chat = ModuleType("llama_cpp.llama_chat_format")
        chat.MTMDChatHandler = MTMDChatHandler
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "look<|im_end|>"}]
        with tempfile.TemporaryDirectory() as folder, patch.dict(
                sys.modules, {"llama_cpp": binding, grammar.__name__: grammar, chat.__name__: chat}):
            model, projector = Path(folder) / "toy.gguf", Path(folder) / "mmproj.gguf"
            _tiny_gguf(model)
            projector.touch()
            backend = LlamaCppBackend()
            backend.load(str(model), str(projector), n_gpu_layers=0, n_ctx=4096)
            self.assertEqual(backend.complete(messages, schema=_SCHEMA, seed=42), '{"age": 4}')
            self.assertEqual(backend.complete(messages, images=[b"jpeg"], schema=_SCHEMA), '{"age": 4}')
            backend.close()
        self.assertEqual(calls[0], ("model", str(model), 0, 4096, False, False))
        self.assertEqual(calls[1], ("projector", str(projector), False))
        self.assertEqual(calls[2:4], [("grammar", _SCHEMA, False), ("reset",)])
        _, prompt, streaming, grammar_object, stop, seed, sampling = calls[4]
        self.assertNotIn("look<|im_end|><|im_end|>", prompt)
        self.assertEqual((streaming, grammar_object, stop, seed), (True, "grammar object", list(text.STOP_STRINGS), 42))
        self.assertEqual(sampling["top_p"], 0.95)
        self.assertEqual(calls[5], ("grammar", _SCHEMA, False))
        _, messages, grammar_object, streaming, thinking = calls[6]
        self.assertEqual((grammar_object, streaming, thinking), ("grammar object", True, False))
        url = "data:image/jpeg;base64," + base64.b64encode(b"jpeg").decode()
        self.assertEqual(messages[-1]["content"], [{"type": "image_url", "image_url": {"url": url}},
                                                  {"type": "text", "text": "look"}])
        self.assertEqual(calls[7], ("close",))
