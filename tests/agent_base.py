# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""A base for the tests of the Agent's host and tools: every store of Contsy in a temporary folder, the Recycle Bin
replaced by a folder, the vision gates and the face checks answered by fixed values, and an empty GPU queue that is
never saved and never run. Nothing live is read or written."""
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import agent_config, agent_host, agent_store, agent_tools, characters, config, history, items, llm
from app import model_manager, prefs, qc, queue_manager, timeline, trash, voices
from engine.media import audio
from tests.agent_fake import FakeClock

ADULT = {"person": True, "age": 30, "gender": "woman", "caption": "a portrait"}
NO_PEOPLE = {"humans": 0, "youngest_human_age": 0}


def picture(path: Path, size=(512, 640), color=(120, 100, 90)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def speech(path: Path, seconds: float, rate: int = 48000) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    wave = (0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    audio.write_wav(path, wave[None, :], rate)
    return path


class AgentCase(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        enter = self.context.enter_context
        self.root = Path(enter(tempfile.TemporaryDirectory())).resolve()
        self.bin = self.root / "recycle_bin"
        self.bin.mkdir()
        builtin = self.root / "assets"
        builtin.mkdir()
        speech(builtin / "hanna.wav", 5.0)
        (builtin / "voices.json").write_text(json.dumps({"voices": [
            {"name": "Hanna", "file": "hanna.wav", "gender": "female", "description": "20s, bright"}]}),
            encoding="utf-8")
        for target in (
                patch.object(agent_store, "RUNS_DIR", self.root / "agent_runs"),
                patch.object(agent_config, "CREDENTIALS_FILE", self.root / "agent_credentials.json"),
                patch.object(agent_config, "SETTINGS_FILE", self.root / "agent_settings.json"),
                patch.object(characters, "DIR", self.root / "characters"),
                patch.object(characters, "DRAFTS_DIR", self.root / "character_drafts"),
                patch.object(items, "DIR", self.root / "items"),
                patch.object(items, "DRAFTS_DIR", self.root / "item_drafts"),
                patch.object(voices, "BUILTIN_DIR", builtin),
                patch.object(voices, "MANIFEST", builtin / "voices.json"),
                patch.object(voices, "USER_DIR", self.root / "voices"),
                patch.object(voices, "measure_pitch", return_value=180.0),
                patch.object(prefs, "_FILE", self.root / "prefs.json"),
                patch.object(timeline, "DIR", self.root / "projects"),
                patch.object(config, "OUTPUTS_DIR", self.root / "outputs"),
                patch.object(config, "HISTORY_DIR", self.root / "history"),
                patch.object(history, "_HISTORY_FILE", self.root / "history" / "history.jsonl"),
                patch.object(characters, "gate_models", return_value=None),
                patch.object(items, "gate_models", return_value=None),
                patch.object(llm, "check_photo", return_value=ADULT),
                patch.object(llm, "check_item_photo", return_value=dict(NO_PEOPLE)),
                patch.object(qc, "_faces", return_value=[object()]),
                patch.object(qc, "face_score", return_value=0.9),
                patch.object(model_manager, "VERIFIED_FILE", self.root / "verified_models.json"),
                patch.object(model_manager, "_STORED", None),
                patch.object(timeline.sound, "available", return_value=True),
                patch.object(timeline.tts, "available", return_value=True),
                patch.object(timeline.model_manager, "ensure", return_value=None),
                patch.object(queue_manager, "_JOBS", []),
                patch.object(queue_manager, "_STATE", self.root / "queue.json"),
                patch.object(queue_manager, "_STATE_INFO", {"next_id": 1, "paused": False}),
                patch.object(queue_manager, "FILES_DIR", self.root / "queue_files"),
                patch.object(queue_manager, "STAGES_DIR", self.root / "stages"),
                patch.object(queue_manager, "_save"),
                patch.object(trash, "move", side_effect=self.to_bin)):
            enter(target)
        (self.root / "outputs").mkdir()
        self.clock = FakeClock()
        self.runs = 0

    def to_bin(self, path) -> bool:
        path = Path(path)
        shutil.move(str(path), str(self.bin / f"{len(list(self.bin.iterdir())):03d}-{path.name}"))
        return True

    def binned(self) -> list:
        return sorted(item.name.split("-", 1)[1] for item in self.bin.iterdir())

    # --- a run and its host -------------------------------------------------------------------------------------

    def host(self, brief=(), **settings) -> agent_host.Host:
        """A host for a new run; `settings` are the run's settings (permissions as a dict, mode, budgets...)."""
        store = agent_store.RunStore(agent_store.new_run_id(1_700_000_000 + self.runs))
        self.runs += 1
        store.create(summary={"brief": list(brief)}, system="system", tools=[])
        host = agent_host.Host(store, settings, clock=self.clock.time, sleep=self.clock.sleep)
        return host

    def allow(self, **permissions) -> dict:
        return {"permissions": permissions}

    def call(self, host, tool, /, **arguments) -> agent_tools.Result:
        return agent_tools.call(host, tool, arguments)

    def ok(self, host, tool, /, **arguments):
        """The answer of a tool that must succeed, as data."""
        result = self.call(host, tool, **arguments)
        self.assertTrue(result.ok, result.text)
        try:
            return json.loads(result.text)
        except ValueError:
            return result.text

    def refused(self, host, tool, code, /, **arguments) -> str:
        result = self.call(host, tool, **arguments)
        self.assertFalse(result.ok, result.text)
        self.assertEqual(result.code, code, result.text)
        return result.text

    # --- library fixtures ---------------------------------------------------------------------------------------

    def saved_character(self, name="Mia", look="Short red hair, freckles, green eyes", photos=2) -> dict:
        draft = characters.new_draft()
        characters.update_draft(draft, name=name, gender="woman", look=look)
        for index in range(photos):
            ok, message = characters.add_upload(draft, picture(self.root / f"{name}{index}.png",
                                                               color=(100 + index, 90, 80 + len(name))), True)
            self.assertTrue(ok, message)
        ok, message = characters.use_preset_voice(draft, "hanna")
        self.assertTrue(ok, message)
        profile, message = characters.save_draft(draft)
        self.assertIsNotNone(profile, message)
        return profile

    def saved_item(self, name="TV studio", kind=items.PLACE, look="A modern TV news studio with a blue anchor desk",
                   photos=2) -> dict:
        draft = items.new_draft()
        items.update_draft(draft, name=name, kind=kind, look=look)
        for index in range(photos):
            ok, message = items.add_upload(draft, picture(self.root / f"{name}{index}.png",
                                                          color=(60 + index, 70 + len(name), 140)), True)
            self.assertTrue(ok, message)
        profile, message = items.save_draft(draft)
        self.assertIsNotNone(profile, message)
        return profile

    # --- jobs ---------------------------------------------------------------------------------------------------

    def finish_jobs(self, *job_ids, status=queue_manager.DONE, files=(), error="") -> None:
        """What the worker would do: end the jobs (the queue is never run in these tests)."""
        for job in queue_manager.snapshot():
            if job.id in job_ids:
                job.status, job.started, job.finished = status, self.clock.time() - 30, self.clock.time()
                job.progress, job.files, job.error = 1.0, list(files), error

    def clear_queue(self) -> None:
        queue_manager._JOBS.clear()
