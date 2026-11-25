# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Regression checks for the queue's content guard and the preservation of local data."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app import characters, history, prefs, queue_manager as qm


class CharacterPhotoTests(unittest.TestCase):
    def test_a_photo_that_looks_like_a_minor_is_refused_and_the_users_file_is_preserved(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo = root / "mine.png"
            Image.new("RGB", (512, 512), "gray").save(photo)
            original = photo.read_bytes()
            with patch.object(characters, "DRAFTS_DIR", root / "drafts"), \
                    patch.object(characters, "DIR", root / "characters"), \
                    patch.object(characters, "gate_models", return_value=None), \
                    patch.object(characters.llm, "check_photo", return_value={"person": True, "age": 16}):
                draft = characters.new_draft()
                ok, message = characters.add_upload(draft, photo, True)
                self.assertFalse(ok)
                self.assertIn("minor", message)
                self.assertEqual(characters.draft(draft)["photos"], [])
            self.assertEqual(photo.read_bytes(), original)
            self.assertFalse((root / "characters").exists())


class RemovedFeatureTests(unittest.TestCase):
    def test_saved_jobs_of_the_removed_character_training_load_and_never_run(self):
        saved = [{"id": 1, "kind": "trening", "label": "Character training: Ola", "params": {"slug": "ola", "steps": 2400},
                  "status": qm.PENDING, "mark_t": 5.0, "mark_p": 0.2},
                 {"id": 2, "kind": "zestaw", "label": "Character image set: Ola", "params": {"count": 20},
                  "status": qm.DONE, "files": ["set.zip"]},
                 {"id": 3, "kind": "trening", "label": "Character training: Kuba", "params": {"slug": "kuba"},
                  "status": qm.RUNNING}]
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "queue.json"
            state.write_text(json.dumps({"jobs": saved, "rates": {"trening": 4.5, "zestaw": 12.0, "obraz": 31.0}}),
                             encoding="utf-8")
            rates = dict(qm._RATE)
            with patch.object(qm, "_STATE", state), patch.object(qm, "_JOBS", []), \
                    patch.object(qm, "_STATE_INFO", {"next_id": 1, "paused": False}), patch.dict(qm._RATE, rates):
                qm._load()
                jobs = {job.id: job for job in qm.snapshot()}
                self.assertNotIn("trening", qm._RATE)
                self.assertNotIn("zestaw", qm._RATE)
                self.assertIsNone(qm._next())  # nothing of a removed feature waits to run
        self.assertEqual((jobs[1].status, jobs[1].error), (qm.STOPPED, qm.REMOVED_MSG))
        self.assertEqual((jobs[2].status, jobs[2].files), (qm.DONE, ["set.zip"]))
        self.assertEqual((jobs[3].status, jobs[3].error), (qm.STOPPED, qm.REMOVED_MSG))
        from app import ui_queue

        self.assertEqual(ui_queue.KINDS["trening"], "Character training (removed)")
        job = qm.Job(9, "trening", "old", {})
        with patch.object(qm, "_save"), patch.object(qm, "FILES_DIR", Path(tmp)):
            qm._run(job)
        self.assertEqual((job.status, job.error), (qm.ERROR, qm.REMOVED_MSG))

    def test_old_history_entries_of_training_and_character_loras_still_load(self):
        from app import studio

        with tempfile.TemporaryDirectory() as tmp, patch.object(history, "_HISTORY_FILE", Path(tmp) / "h.jsonl"), \
                patch.object(characters, "DIR", Path(tmp) / "characters"):
            history.record({"mode": "character_set", "files": [str(Path(tmp) / "set.zip")]})
            history.record({"mode": "txt2img", "loras": [{"name": "aistudio_char_ola.safetensors", "strength": 0.7}],
                            "files": [str(Path(tmp) / "img.png")]})
            entries = history.recent()
            self.assertEqual([entry["mode"] for entry in entries], ["txt2img", "character_set"])
            self.assertEqual(studio.file_owners(entries), {"set.zip": "", "img.png": ""})


class QueueTests(unittest.TestCase):
    def test_jobs_saved_by_earlier_versions_are_restored_as_they_are(self):
        """Jobs written while prompts were still filtered: an old content "level" field and prompts that
        filter refused do not stop a job any more, and loading never rewrites the saved queue."""
        params = ({"prompt": "kobieta opalająca się na plaży bez stroju kąpielowego"},
                  {"prompt": "an adult on a beach", "level": "nsfw"}, {"prompt": "a landscape", "level": "sfw"})
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "queue.json"
            saved = []
            for entry in params:
                for status in (qm.PENDING, qm.RUNNING, qm.DONE):
                    saved.append({"id": len(saved) + 1, "kind": "test", "label": "legacy",
                                  "params": entry, "status": status,
                                  "files": ["preserved-output.png"] if status == qm.DONE else []})
            state.write_text(json.dumps({"jobs": saved}, ensure_ascii=False), encoding="utf-8")
            original = state.read_bytes()
            with patch.object(qm, "_STATE", state), patch.object(qm, "_JOBS", []), patch.object(qm, "_STATE_INFO", {"next_id": 1, "paused": False}):
                qm._load()
                jobs = qm.snapshot()
                self.assertEqual(len(jobs), len(saved))
                for job, entry in zip(jobs, saved):
                    with self.subTest(params=entry["params"], status=entry["status"]):
                        self.assertEqual(job.params, entry["params"])
                        self.assertEqual(job.error, "")
                        if entry["status"] == qm.DONE:
                            self.assertEqual(job.status, qm.DONE)
                            self.assertEqual(job.files, entry["files"])
                        else:  # an interrupted job waits again
                            self.assertEqual(job.status, qm.PENDING)
                self.assertEqual(state.read_bytes(), original)

    def test_restored_queue_skips_malformed_entries_and_blocks_minor_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "queue.json"
            state.write_text(json.dumps({"next_id": "broken", "rates": {"obraz": "bad"}, "jobs": [None, {"id": 1}, {"id": 2, "kind": "obraz", "label": "blocked", "params": {"positive": "a 15-year-old in a bikini"}}, {"id": 3, "kind": "obraz", "label": "fine", "params": {"positive": "a landscape"}, "status": qm.RUNNING}]}), encoding="utf-8")
            with patch.object(qm, "_STATE", state), patch.object(qm, "_JOBS", []), patch.object(qm, "_STATE_INFO", {"next_id": 1, "paused": False}):
                qm._load()
                jobs = {j.id: j for j in qm.snapshot()}
                self.assertEqual(set(jobs), {2, 3})
                self.assertEqual(jobs[2].status, qm.ERROR)
                self.assertIn("minors", jobs[2].error)
                self.assertEqual(jobs[3].status, qm.PENDING)
                self.assertEqual(qm._STATE_INFO["next_id"], 4)

    def test_restored_malformed_status_and_timestamps_do_not_break_other_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "queue.json"
            base = {"kind": "obraz", "label": "image", "params": {}}
            jobs = [{**base, "id": 1, "status": []},
                    {**base, "id": 2, "created": "invalid"},
                    {**base, "id": 3, "files": "bad"},
                    {**base, "id": 4}]
            state.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
            with patch.object(qm, "_STATE", state), patch.object(qm, "_JOBS", []), patch.object(qm, "_STATE_INFO", {"next_id": 1, "paused": False}):
                qm._load()
                self.assertEqual([j.id for j in qm.snapshot()], [4])

    def test_removed_pending_job_is_stopped_for_callers_and_absent_from_saved_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "queue.json"
            removed = qm.Job(1, "test", "Pending", {})
            finished = qm.Job(2, "test", "Earlier result", {}, status=qm.DONE)
            with (patch.object(qm, "_STATE", state), patch.object(qm, "FILES_DIR", root / "attachments"),
                  patch.object(qm, "_JOBS", [removed, finished])):
                qm.remove([removed.id, finished.id])
                self.assertEqual(qm.snapshot(), [finished])
                self.assertEqual(removed.status, qm.STOPPED)
                self.assertEqual(removed.error, "Removed from queue.")
                self.assertIsNotNone(removed.finished)
                self.assertEqual([job["id"] for job in json.loads(state.read_text(encoding="utf-8"))["jobs"]],
                                 [finished.id])

    def test_adult_prompts_reach_the_runner(self):
        """The queue has no adult-content filter: only the minors guard below stops a job."""
        for prompt in ("a topless adult woman on a beach", "an adult without any clothes",
                       "kobieta opalająca się na plaży bez stroju kąpielowego"):
            job = qm.Job(id=1, kind="test", label="adult", params={"positive": prompt})
            runner = unittest.mock.Mock(return_value=(True, [], [], ""))
            with self.subTest(prompt=prompt), patch.dict(qm._RUNNERS, {"test": runner}), patch.object(qm, "_save"), patch.object(qm, "FILES_DIR", Path(tempfile.gettempdir()) / "nonexistent-qm-security-test"):
                self.assertEqual(qm._prompt_refusal(job.params), "")
                qm._run(job)
                runner.assert_called_once()
                self.assertEqual((job.status, job.error), (qm.DONE, ""))

    def test_minor_in_intimate_clothing_is_blocked_for_every_queue_kind(self):
        for kind in ("obraz", "wideo", "test"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "minors"):
                qm.add(kind, "blocked", {"text": "a 15-year-old in a bikini"})

    def test_cast_portrait_and_voice_design_texts_are_guarded(self):
        """Production cast jobs carry a person's look and voice design: both are checked before any render."""
        for key in ("appearance", "design", "look"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "minors"):
                qm.add("test", "blocked", {"name": "Ann", key: "a 15-year-old in a bikini"})

    def test_restored_minor_job_never_reaches_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "queue.json"
            state.write_text(json.dumps({"jobs": [{"id": 1, "kind": "test", "label": "legacy",
                                                  "params": {"text": "a １５ｙｏ in a bi\u200bkini"}}]}), encoding="utf-8")
            with patch.object(qm, "_STATE", state), patch.object(qm, "_JOBS", []), patch.object(qm, "_STATE_INFO", {"next_id": 1, "paused": False}):
                qm._load()
                self.assertEqual(qm.snapshot()[0].status, qm.ERROR)
                self.assertIn("minors", qm.snapshot()[0].error)

    def test_runner_preflight_combines_age_and_clothing_from_separate_fields(self):
        job = qm.Job(id=1, kind="test", label="blocked", params={"idea": "a 15-year-old", "text": "in a bikini"})
        with patch.dict(qm._RUNNERS, {"test": unittest.mock.Mock()}), patch.object(qm, "_save"), patch.object(qm, "FILES_DIR", Path(tempfile.gettempdir()) / "nonexistent-qm-security-test"):
            qm._run(job)
            qm._RUNNERS["test"].assert_not_called()
        self.assertEqual(job.status, qm.ERROR)
        self.assertIn("minors", job.error)

    def test_attachment_field_cannot_escape_queue_folder(self):
        with self.assertRaisesRegex(ValueError, "attachment"):
            qm.add("test", "test", {}, files={"../outside": "anything"})


class PersistenceTests(unittest.TestCase):
    def test_concurrent_preference_updates_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(prefs, "_FILE", Path(tmp) / "prefs.json"):
            barrier = threading.Barrier(2)
            def save(key, value):
                barrier.wait()
                prefs.save(**{key: value})
            threads = [threading.Thread(target=save, args=("folder", "my-folder")), threading.Thread(target=save, args=("character", "test"))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            result = prefs.load()
            self.assertEqual(result["folder"], "my-folder")
            self.assertEqual(result["character"], "test")

    def test_autosave_preserves_an_existing_file_with_the_same_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "image.png"
            source.write_bytes(b"new")
            destination = root / "saved"
            destination.mkdir()
            (destination / "image.png").write_bytes(b"old")
            with patch.object(prefs, "load", return_value={"autosave": True, "folder": str(destination)}):
                prefs.autosave([str(source)])
            self.assertEqual((destination / "image.png").read_bytes(), b"old")
            self.assertEqual((destination / "image_2.png").read_bytes(), b"new")

    def test_history_forget_does_not_match_prompt_or_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(history, "_HISTORY_FILE", Path(tmp) / "history.jsonl"), patch.object(history.config, "HISTORY_DIR", Path(tmp)):
            history.record({"idea": "a.png", "files": ["C:/out/b.png"]})
            history.record({"files": ["C:/out/a.png", "C:/out/c.png"]})
            history.forget(["C:/out/a.png"])
            records = history.recent()
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["files"], ["C:/out/c.png"])
            self.assertEqual(records[1]["idea"], "a.png")

    def test_history_keeps_same_named_files_in_different_folders_apart(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(history, "_HISTORY_FILE", Path(tmp) / "history.jsonl"), patch.object(history.config, "HISTORY_DIR", Path(tmp)):
            first = str(Path(tmp) / "one" / "image.png")
            second = str(Path(tmp) / "two" / "image.png")
            history.record({"files": [first], "seed": 1})
            history.record({"files": [second], "seed": 2})
            self.assertEqual(history.entry_for(first)["seed"], 1)
            self.assertEqual(history.entry_for(second)["seed"], 2)
            history.forget([first])
            self.assertIsNone(history.entry_for(first))
            self.assertEqual(history.entry_for(second)["seed"], 2)

    def test_character_slug_cannot_escape_library(self):
        for slug in ("../escape", "C:/escape", "a/b", "a\\b", ".", ".."):
            with self.subTest(slug=slug):
                self.assertIsNone(characters.get(slug))
                with self.assertRaises(ValueError):
                    characters._file(slug)


if __name__ == "__main__":
    unittest.main()
