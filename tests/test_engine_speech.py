# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The VoxCPM2 boundary with deterministic waveforms and an actual spawned worker: cloned lines and designed
voices."""
import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "soundfile"))

# A Python module for the spawned worker. It records calls and makes the
# determinism of the user's seed visible in the waveform.
FAKE = """
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import torch

class VoxCPM:
    def record(self, payload):
        with (self.path / 'calls.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(payload) + '\\n')

    @classmethod
    def from_pretrained(cls, path, **options):
        model = cls()
        model.path = Path(path)
        model.tts_model = SimpleNamespace(sample_rate=48000)
        model.record(options)
        torch.rand(4)  # model constructors may consume RNG state
        return model

    def generate(self, **options):
        self.record(options)
        text = options['text']
        if text.startswith('('):  # a designed voice: the description decides the take
            if 'silent' in text:
                return torch.zeros(48000 * 4).numpy()
            if 'short' in text or ('flaky' in text and not getattr(self, 'flaked', False)):
                self.flaked = True
                return (torch.rand(100) - 0.5).numpy()
            return (torch.rand(48000 * 4) - 0.5).numpy()
        if text == 'fail':
            raise ValueError('simulated generation failure')
        if text == 'crash':
            os._exit(17)
        if text == 'wait':
            (self.path / 'started').touch()
            threading.Event().wait(30)
        return (torch.rand(32) - 0.5).numpy()
"""


@unittest.skipUnless(HAVE, "speech dependencies are missing")
class SpeechTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.model = self.root / "local_model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        self.ref = self.root / "ref.wav"
        self.ref.touch()
        self.a = self.root / "one.wav"
        self.b = self.root / "two.wav"
        (self.root / "voxcpm.py").write_text(FAKE, encoding="utf-8", newline="\n")
        sys.path.insert(0, str(self.root))
        self.addCleanup(sys.path.remove, str(self.root))
        # Swap only the external library, keeping every engine code path real.
        spec = importlib.util.spec_from_file_location("voxcpm", self.root / "voxcpm.py")
        fake = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fake)
        context = patch.dict(sys.modules, {"voxcpm": fake})
        context.start()
        self.addCleanup(context.stop)

    def _speak(self, second="Żółw", **options):
        from engine.pipelines.speech import SpeechLine, synthesize_many

        lines = [SpeechLine("Raz", self.ref, self.a), SpeechLine(second, self.ref, self.b)]
        return synthesize_many(self.model, lines, device="cpu", **options)

    def _calls(self):
        return [json.loads(line) for line in (self.model / "calls.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_in_process_preserves_voice_recipe_and_rng_state(self):
        import soundfile as sf
        import torch

        progress = []
        state = torch.random.get_rng_state()
        self.assertEqual(self._speak(progress_cb=lambda frac, _msg: progress.append(frac)), self.b)
        self.assertEqual(progress, [0.1, 0.3, 0.6499999999999999, 1.0])
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(self.a.read_bytes(), self.b.read_bytes())
        self.assertEqual(sf.info(self.a).samplerate, 48000)
        self.assertEqual(self._calls(), [
            {"device": "cpu", "load_denoiser": False, "local_files_only": True, "optimize": False},
            {"text": "Raz", "reference_wav_path": str(self.ref), "cfg_value": 2.0, "inference_timesteps": 20},
            {"text": "Żółw", "reference_wav_path": str(self.ref), "cfg_value": 2.0, "inference_timesteps": 20},
        ])

    def test_spawned_worker_produces_the_same_batch_and_progress(self):
        self._speak()
        expected = self.a.read_bytes()
        progress = []
        self.assertEqual(self._speak(isolated=True, progress_cb=lambda frac, _msg: progress.append(frac)), self.b)
        self.assertEqual(self.a.read_bytes(), expected)
        self.assertEqual(self.b.read_bytes(), expected)
        self.assertEqual(progress[0], 0.10)
        self.assertEqual(progress[-1], 1.0)

    def test_failure_and_worker_crash_preserve_existing_outputs(self):
        self.a.write_bytes(b"previous")
        with self.assertRaisesRegex(ValueError, "simulated generation failure"):
            self._speak("fail")
        with self.assertRaisesRegex(RuntimeError, "exit code 17"):
            self._speak("crash", isolated=True)
        self.assertEqual(self.a.read_bytes(), b"previous")
        self.assertFalse(self.b.exists())
        self.assertFalse(list(self.root.glob(".*.wav")))

    def test_cancel_during_isolated_line_removes_worker_and_staged_files(self):
        from engine.core.cancel import CancelToken, Cancelled

        token = CancelToken()
        start = time.monotonic()
        marker = self.model / "started"
        done = threading.Event()

        def cancel_while_running():
            deadline = time.monotonic() + 18
            while not marker.exists() and not done.wait(0.02) and time.monotonic() < deadline:
                pass
            token.cancel()

        canceller = threading.Thread(target=cancel_while_running)
        canceller.start()
        try:
            with self.assertRaises(Cancelled):
                self._speak("wait", isolated=True, cancel=token)
        finally:
            done.set()
            canceller.join()
        self.assertTrue(marker.exists())
        self.assertLess(time.monotonic() - start, 18)
        self.assertFalse(self.a.exists())
        self.assertFalse(self.b.exists())
        self.assertFalse(list(self.root.glob(".*.wav")))

    def test_missing_model_refuses_before_loading_the_library(self):
        from engine.pipelines.speech import SpeechLine, synthesize_many

        with self.assertRaisesRegex(FileNotFoundError, "model is missing"):
            synthesize_many(self.root / "nonexistent", [SpeechLine("Raz", self.ref, self.a)], device="cpu")
        self.assertFalse((self.model / "calls.jsonl").exists())

    # --- voice design ---

    def _design(self, description="warm low alto (calm)", **options):
        from engine.pipelines.speech import VoiceDesign, design_voice

        return design_voice(self.model, VoiceDesign(description, self.a, seed=7), device="cpu", **options)

    def test_a_designed_voice_reads_the_sentence_in_the_described_voice(self):
        import soundfile as sf
        import torch

        from engine.pipelines import speech

        state = torch.random.get_rng_state()
        self.assertEqual(self._design(), self.a)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        info = sf.info(self.a)
        self.assertEqual((info.samplerate, info.frames, info.channels), (48000, 48000 * 4, 1))
        load, take = self._calls()
        self.assertEqual(load, {"device": "cpu", "load_denoiser": False, "local_files_only": True, "optimize": False})
        self.assertEqual(take, {"text": "(warm low alto calm, natural conversational English) " + speech.DESIGN_TEXT,
                                "cfg_value": 2.0, "inference_timesteps": 12})
        first = self.a.read_bytes()
        self._design()
        self.assertEqual(self.a.read_bytes(), first)  # the seed makes the take repeatable

    def test_a_short_take_is_followed_by_the_next_seed_and_a_silent_voice_is_refused(self):
        self._design("flaky voice")
        self.assertEqual(len([call for call in self._calls() if "text" in call]), 2)
        self.a.write_bytes(b"previous")
        with self.assertRaisesRegex(ValueError, "no usable 3–12 second sample"):
            self._design("silent voice")
        self.assertEqual(self.a.read_bytes(), b"previous")
        self.assertFalse(list(self.root.glob(".*.wav")))

    def test_the_spawned_worker_designs_the_same_sample(self):
        self._design()
        expected = self.a.read_bytes()
        progress = []
        self._design(isolated=True, progress_cb=lambda fraction, _msg: progress.append(fraction))
        self.assertEqual(self.a.read_bytes(), expected)
        self.assertEqual((progress[0], progress[-1]), (0.10, 1.0))

    def test_an_empty_description_is_refused_before_the_model_loads(self):
        from engine.pipelines import speech

        with self.assertRaisesRegex(ValueError, "Describe the voice"):
            self._design(" ( ) ")
        self.assertFalse((self.model / "calls.jsonl").exists())
        self.assertEqual(speech.clean_description("(a) deep, (warm) voice."), "a deep, warm voice")
        self.assertEqual(len(speech.clean_description("x" * 500)), speech.MAX_DESCRIPTION)

    def test_only_a_finite_audible_three_to_twelve_second_take_is_a_sample(self):
        import numpy as np

        from engine.pipelines.speech import acceptable

        self.assertTrue(acceptable(np.full(300, 0.5), 100))
        self.assertFalse(acceptable(np.full(299, 0.5), 100))
        self.assertFalse(acceptable(np.full(1201, 0.5), 100))
        self.assertFalse(acceptable(np.full(300, 0.001), 100))
        self.assertFalse(acceptable(np.full((2, 300), 0.5), 100))
        self.assertFalse(acceptable(np.array([np.nan] + [0.5] * 299), 100))
