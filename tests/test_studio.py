# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Pauza karty (przycisk na gornym pasku) i profile modelek (widoki Kolejki, Historii, Studia)."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import cancel, history, llm, studio
from app import queue_manager as qm


class _IsolatedQueue(unittest.TestCase):
    """Kolejka na plikach tymczasowych - prawdziwa config/queue.json zostaje nietknieta."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.files_dir = root / "queue_files"
        self.patches = [
            patch.object(qm, "_STATE", root / "queue.json"),
            patch.object(qm, "FILES_DIR", self.files_dir),
            patch.object(qm, "_JOBS", []),
            patch.object(qm, "_STATE_INFO", {"next_id": 1, "paused": False}),
            patch.object(qm, "_free_gpu", lambda *a, **k: None),  # bez zwalniania pamieci GPU
            patch.object(cancel, "request", cancel.TOKEN.cancel),
        ]
        for p in self.patches:
            p.start()
        cancel.reset()
        qm._REQUEUE.clear()
        self.ref = root / "ref.png"
        self.ref.write_bytes(b"png")

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        cancel.reset()
        qm._REQUEUE.clear()
        self.tmp.cleanup()

    def start_job(self, runner):
        qm.register("test", runner)
        job = qm.add("test", "test", {}, files={"refs": [str(self.ref)]})
        job.status, job.started = qm.RUNNING, time.time()
        worker = threading.Thread(target=qm._run, args=(job,))
        worker.start()
        return job, worker


class PauseTests(_IsolatedQueue):
    def test_pause_puts_the_running_job_back_with_its_files(self):
        started = threading.Event()

        def runner(params, cb):
            started.set()
            while True:  # liczy, az pauza przerwie (cb -> cancel.check)
                cb(0.5, "liczy")
                time.sleep(0.01)

        job, worker = self.start_job(runner)
        self.assertTrue(started.wait(2))
        self.assertEqual(qm.pause_now(), job.id)
        worker.join(5)
        self.assertEqual((job.status, job.progress, job.started), (qm.PENDING, 0.0, None))
        self.assertTrue((self.files_dir / str(job.id)).exists())  # zdjecia do ponownego liczenia
        self.assertTrue(qm.paused())
        self.assertIsNone(qm._next())  # pauza - nic nie rusza
        qm.set_paused(False)
        self.assertIs(qm._next(), job)  # po wznowieniu wraca na swoje miejsce

    def test_job_finished_at_the_moment_of_pause_stays_done(self):
        started, go = threading.Event(), threading.Event()

        def runner(params, cb):
            started.set()
            go.wait(2)
            return True, ["wynik.png"], [], ""

        job, worker = self.start_job(runner)
        self.assertTrue(started.wait(2))
        qm.pause_now()
        go.set()
        worker.join(5)
        self.assertEqual(job.status, qm.DONE)

    def test_stop_button_still_stops_for_good(self):
        started = threading.Event()

        def runner(params, cb):
            started.set()
            while True:
                cb(0.5, "liczy")
                time.sleep(0.01)

        job, worker = self.start_job(runner)
        self.assertTrue(started.wait(2))
        self.assertTrue(qm.stop_current(job.id))
        worker.join(5)
        self.assertEqual(job.status, qm.STOPPED)
        self.assertFalse((self.files_dir / str(job.id)).exists())

    def test_prompt_helper_stays_off_the_gpu_while_paused(self):
        calls = []
        with patch.object(qm, "paused", return_value=True), \
                patch.object(llm, "_generate",
                             side_effect=lambda s, u, gpu, *a: calls.append(gpu) or (True, "ok")):
            ok, _ = llm.ask("system", "user")
        self.assertTrue(ok)
        self.assertEqual(calls, [False])


CHARS = [{"name": "Virtual Mia", "slug": "mia", "photos": ["mia.png"], "created": 1},
         {"name": "Nova", "slug": "nova", "photos": ["nova.png"], "created": 2}]


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(studio.characters, "all_characters", return_value=CHARS)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_results_belong_to_a_profile_by_slug_or_name(self):
        """An image of the removed LoRA characters named only its LoRA: it belongs to nobody now."""
        entries = [{"files": ["C:/o/d.png"]},
                   {"loras": [{"name": "aistudio_char_mia.safetensors"}], "files": ["C:/o/c.png"]},
                   {"character": "Nova", "files": ["C:/o/b.mp4"]},
                   {"profile": "mia", "character": "Virtual Mia", "files": ["C:/o/a.png"]}]
        self.assertEqual(studio.file_owners(entries),
                         {"a.png": "mia", "b.mp4": "nova", "c.png": "", "d.png": ""})

    def test_queue_view_shows_only_the_selected_models_jobs(self):
        jobs = [SimpleNamespace(id=1, params={"character": "mia"}),
                SimpleNamespace(id=2, params={"character": "nova"}),
                SimpleNamespace(id=3, params={})]
        self.assertEqual([j.id for j in studio.jobs("mia", jobs)], [1])
        self.assertEqual([j.id for j in studio.jobs("", jobs)], [1, 2, 3])

    def test_history_view_lists_only_the_profiles_files(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(studio.config, "OUTPUTS_DIR", Path(tmp)):
            for name in ("a.png", "b.mp4", "c.png", "x_raw.png"):
                (Path(tmp) / name).write_bytes(b"x")
            owners = {"a.png": "mia", "b.mp4": "nova", "c.png": "mia", "x_raw.png": "mia"}
            entries = [{"files": [str(Path(tmp) / name)]} for name in owners]
            with patch.object(studio.history, "recent", return_value=entries):
                mine = studio.results("mia", owners=owners)
                everyone = studio.results("", owners=owners)
        self.assertEqual(sorted(Path(p).name for p in mine), ["a.png", "c.png"])
        self.assertEqual(sorted(Path(p).name for p in everyone), ["a.png", "b.mp4", "c.png"])

    def test_every_recorded_result_that_exists_is_listed_whatever_its_record_says(self):
        """Records of earlier versions carry review fields ("qc.safe", "qc.coverage", a content
        "level"): they are read as any other record and decide nothing. A job's work files, which
        the history does not list, stay out, and so does a recorded file that is gone."""
        with tempfile.TemporaryDirectory() as tmp, patch.object(studio.config, "OUTPUTS_DIR", Path(tmp)):
            paths = {name: Path(tmp) / name for name in ("new.png", "old.png", "refused.png", "adult.mp4",
                                                         "unreviewed.png", "work_part1.mp4", "draft.png")}
            for path in paths.values():
                path.write_bytes(b"preserved")
            entries = [{"files": [str(paths["new.png"])]},
                       {"files": [str(paths["old.png"])], "level": "sfw",
                        "qc": {"ok": True, "safe": True, "coverage": {"people": "none"}}},
                       {"files": [str(paths["refused.png"])], "qc": {"ok": False, "safe": False}},
                       {"files": [str(paths["adult.mp4"])], "qc": {"safe": True}, "level": "nsfw"},
                       {"files": [str(paths["unreviewed.png"])], "qc": {"ok": True}},
                       {"files": [str(Path(tmp) / "gone.png")]}, {"files": "broken"}, {"mode": "no files"}, None]
            listed = ["adult.mp4", "new.png", "old.png", "refused.png", "unreviewed.png"]
            with patch.object(studio.history, "recent", return_value=entries):
                self.assertEqual(sorted(Path(file).name for file in studio.results()), listed)
                self.assertEqual(sorted(Path(file).name for file in studio.results(exts=(".mp4",))), ["adult.mp4"])
                self.assertTrue(studio.is_recorded(str(paths["refused.png"])))
                self.assertFalse(studio.is_recorded(str(paths["draft.png"])))
            rows = {row["slug"]: row for row in studio.overview([], [entry for entry in entries if entry], time.time())}
            self.assertEqual(rows[""]["total"], len(listed))
            self.assertTrue(all(path.read_bytes() == b"preserved" for path in paths.values()))

    def test_overview_counts_queue_and_results_per_model(self):
        now = time.time()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(studio.characters, "DIR", Path(tmp)), \
                patch.object(studio.characters, "usable", return_value={"slug": "x"}):
            files = []
            for name in ("a.png", "b.png", "c.png"):
                (Path(tmp) / name).write_bytes(b"x")
                files.append(str(Path(tmp) / name))
            entries = [{"profile": "mia", "ts": now, "files": [files[0]]},
                       {"profile": "mia", "ts": now - 3 * 86400, "files": [files[1]]},
                       {"profile": "nova", "ts": now, "files": [files[2], str(Path(tmp) / "gone.png")]}]
            snapshot = [qm.Job(id=1, kind="obraz", label="a", params={"character": "mia"}),
                        qm.Job(id=2, kind="obraz", label="b", params={"character": "mia"},
                               status=qm.DONE),
                        qm.Job(id=3, kind="obraz", label="c", params={})]
            rows = {r["slug"]: r for r in studio.overview(snapshot, entries, now)}
        self.assertEqual((rows["mia"]["queued"], rows["mia"]["today"], rows["mia"]["total"]),
                         (1, 1, 2))
        self.assertEqual((rows["nova"]["queued"], rows["nova"]["total"]), (0, 1))  # bez znikłego
        self.assertEqual(rows[""]["queued"], 1)  # zadanie bez postaci

    def test_a_job_belongs_to_its_main_character_only(self):
        """A job of the removed training named its character by "slug"; the creator's draft jobs name none."""
        training = qm.Job(id=4, kind="trening", label="t", params={"slug": "kuba", "steps": 2400})
        draft = qm.Job(id=5, kind="character_photos", label="p", params={"draft": "0" * 32, "count": 4})
        video = qm.Job(id=6, kind="wideo", label="v", params={"character": "kuba"})
        self.assertEqual([studio.job_owner(job) for job in (training, draft, video)], ["", "", "kuba"])
        self.assertEqual([j.id for j in studio.jobs("kuba", [training, draft, video])], [6])
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(studio.characters, "DIR", Path(tmp)), \
                patch.object(studio.characters, "all_characters",
                             return_value=[{"slug": "kuba", "name": "Kuba", "photos": []}]), \
                patch.object(studio.characters, "usable", return_value=None):
            rows = {r["slug"]: r for r in studio.overview([training], [], time.time())}
        self.assertEqual((rows["kuba"]["queued"], rows[""]["queued"]), (0, 1))
        self.assertEqual(rows["kuba"]["status"], "Needs reference photos")

    def test_queue_jobs_record_their_profile_in_history(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(history, "_HISTORY_FILE", Path(tmp) / "history.jsonl"), \
                patch.object(history.config, "HISTORY_DIR", Path(tmp)):
            history.job_context({"character": "mia"})
            entry = history.record({"files": ["a.png"]})
            history.job_context(None)
        self.assertEqual(entry["profile"], "mia")


if __name__ == "__main__":
    unittest.main()
