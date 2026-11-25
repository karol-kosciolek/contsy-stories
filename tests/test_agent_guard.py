# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Agent page's two seams in the old code: downloads switched off for its work, and the owner of a queued job."""
import hashlib
import json
import tempfile
import threading
import unittest
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from app import downloader, model_manager, queue_manager

DATA = b"a model that is not on this disk" * 50


class DownloadGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hits = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.hits.append(self.path)
                self.send_response(200)
                self.send_header("Content-Length", str(len(DATA)))
                self.end_headers()
                self.wfile.write(DATA)

            def log_message(self, *_):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.hits.clear()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def fetch(self):
        with patch.object(downloader, "_resolve_url", return_value=self.url):
            return downloader.download_file("official/model", "model.gguf", self.folder, expected_size=len(DATA),
                                            sha256=hashlib.sha256(DATA).hexdigest())

    def test_the_network_is_never_touched_while_downloads_are_off(self):
        with downloader.no_downloads():
            self.assertTrue(downloader.downloads_blocked())
            with self.assertRaises(downloader.DownloadsBlocked) as caught:
                self.fetch()
        self.assertIn(downloader.BLOCKED_PREFIX, str(caught.exception))
        self.assertEqual(self.hits, [])
        self.assertFalse(downloader.downloads_blocked())
        self.assertEqual(self.fetch().read_bytes(), DATA)  # the same call works once downloads are on again
        self.assertEqual(len(self.hits), 1)

    def test_a_file_that_is_already_there_is_still_used(self):
        self.fetch()
        self.hits.clear()
        with downloader.no_downloads():
            self.assertEqual(self.fetch().read_bytes(), DATA)
        self.assertEqual(self.hits, [])

    def test_the_switch_belongs_to_the_thread_that_set_it(self):
        seen = []
        with downloader.no_downloads():
            worker = threading.Thread(target=lambda: seen.append(downloader.downloads_blocked()))
            worker.start()
            worker.join()
        self.assertEqual(seen, [False])

    def test_ensure_names_the_missing_models_and_downloads_nothing(self):
        catalog = {"id": "m1", "name": "Test Model — a long description"}
        with patch.object(model_manager, "is_installed", return_value=False), \
                patch.object(model_manager, "announce_checks"), \
                patch.object(model_manager.config, "get_model", return_value=catalog), \
                patch.object(model_manager, "download_model") as download:
            with downloader.no_downloads():
                message = model_manager.ensure(["m1"], exact=True)
            download.assert_not_called()
            self.assertEqual(message, f"{downloader.BLOCKED_PREFIX}: Test Model.")
            self.assertIsNone(model_manager.ensure(["m1"], exact=True))  # allowed: it downloads
            download.assert_called_once()

    def test_ensure_with_everything_installed_is_not_a_problem_when_downloads_are_off(self):
        with patch.object(model_manager, "is_installed", return_value=True), \
                patch.object(model_manager, "announce_checks"), downloader.no_downloads():
            self.assertIsNone(model_manager.ensure(["m1"], exact=True))


class QueueOwnerTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        enter = self.context.enter_context
        self.root = Path(enter(tempfile.TemporaryDirectory())).resolve()
        enter(patch.object(queue_manager, "_JOBS", []))
        enter(patch.object(queue_manager, "_STATE", self.root / "queue.json"))
        enter(patch.object(queue_manager, "FILES_DIR", self.root / "queue_files"))
        enter(patch.object(queue_manager, "STAGES_DIR", self.root / "stages"))
        enter(patch.object(queue_manager, "_STATE_INFO", {"next_id": 1, "paused": False}))
        enter(patch.object(queue_manager, "_prompt_refusal", return_value=""))

    def test_a_job_added_inside_the_block_carries_its_owner_and_prefix(self):
        with queue_manager.owner("agent:20261117-120000-abc123", downloads=False, prefix="Agent · "):
            mine = queue_manager.add("obraz", "A street", {"prompt": "x"})
        yours = queue_manager.add("obraz", "A road", {"prompt": "y"})
        self.assertEqual((mine.owner, mine.downloads, mine.label), ("agent:20261117-120000-abc123", False,
                                                                    "Agent · A street"))
        self.assertEqual((yours.owner, yours.downloads, yours.label), ("", True, "A road"))

    def test_blocks_nest_and_restore_and_threads_do_not_share(self):
        other = []
        with queue_manager.owner("outer"):
            with queue_manager.owner("inner", downloads=False):
                thread = threading.Thread(target=lambda: other.append(queue_manager.add("obraz", "t", {}).owner))
                thread.start()
                thread.join()
                self.assertEqual(queue_manager.add("obraz", "a", {}).owner, "inner")
            self.assertEqual(queue_manager.add("obraz", "b", {}).owner, "outer")
        self.assertEqual(queue_manager.add("obraz", "c", {}).owner, "")
        self.assertEqual(other, [""])

    def test_a_bad_tag_is_refused(self):
        for tag in ("", "a b", "x" * 81, "../x", None, 5):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                with queue_manager.owner(tag):
                    pass

    def test_the_owner_survives_a_restart_and_an_old_queue_file_still_loads(self):
        with queue_manager.owner("agent:r1", downloads=False):
            queue_manager.add("obraz", "kept", {"prompt": "p"})
        data = json.loads((self.root / "queue.json").read_text(encoding="utf-8"))
        old = dict(data["jobs"][0])
        del old["owner"], old["downloads"]
        old.update(id=9, label="from before")
        data["jobs"].append(old)
        (self.root / "queue.json").write_text(json.dumps(data), encoding="utf-8")
        queue_manager._JOBS.clear()
        queue_manager._load()
        jobs = {job.label: job for job in queue_manager.snapshot()}
        self.assertEqual((jobs["kept"].owner, jobs["kept"].downloads), ("agent:r1", False))
        self.assertEqual((jobs["from before"].owner, jobs["from before"].downloads), ("", True))

    def test_a_saved_job_with_a_wrong_owner_type_is_skipped(self):
        with queue_manager.owner("agent:r1"):
            queue_manager.add("obraz", "kept", {"prompt": "p"})
        data = json.loads((self.root / "queue.json").read_text(encoding="utf-8"))
        data["jobs"][0]["owner"] = 5
        (self.root / "queue.json").write_text(json.dumps(data), encoding="utf-8")
        queue_manager._JOBS.clear()
        queue_manager._load()
        self.assertEqual(queue_manager.snapshot(), [])

    def test_a_job_without_permission_runs_with_downloads_off_and_the_next_one_does_not(self):
        seen = []

        def runner(params, cb):
            seen.append(downloader.downloads_blocked())
            return True, [], [], ""

        with patch.dict(queue_manager._RUNNERS, {"probe": runner}):
            with queue_manager.owner("agent:r1", downloads=False):
                first = queue_manager.add("probe", "one", {})
            second = queue_manager.add("probe", "two", {})
            queue_manager._run(first)
            queue_manager._run(second)
        self.assertEqual(seen, [True, False])
        self.assertEqual((first.status, second.status), (queue_manager.DONE, queue_manager.DONE))

    def test_a_model_the_job_needs_ends_it_with_the_reason_not_with_a_download(self):
        def runner(params, cb):
            return False, [], [], model_manager.ensure(["m1"], exact=True) or ""

        catalog = {"id": "m1", "name": "Test Model — x"}
        with patch.dict(queue_manager._RUNNERS, {"probe": runner}), \
                patch.object(model_manager, "is_installed", return_value=False), \
                patch.object(model_manager, "announce_checks"), \
                patch.object(model_manager.config, "get_model", return_value=catalog), \
                patch.object(model_manager, "download_model") as download:
            with queue_manager.owner("agent:r1", downloads=False):
                job = queue_manager.add("probe", "needs a model", {})
            queue_manager._run(job)
        download.assert_not_called()
        self.assertEqual(job.status, queue_manager.ERROR)
        self.assertTrue(job.error.startswith(downloader.BLOCKED_PREFIX))


if __name__ == "__main__":
    unittest.main()
