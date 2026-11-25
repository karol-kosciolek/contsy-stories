# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Queue fixes: a job that finishes as Stop is clicked is done, Stop is requested under the queue lock, a requeued
job marks its new attempt in the stage log and a job that fails to queue leaves no attachments behind. No test
touches the live config/queue.json."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app import cancel, queue_manager


class QueueFixTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.patches = [patch.object(queue_manager, "STAGES_DIR", root / "stages"),
                        patch.object(queue_manager, "FILES_DIR", root / "queue_files"),
                        patch.object(queue_manager, "_STATE", root / "queue.json"),
                        patch.object(queue_manager, "_save"),
                        patch.object(queue_manager, "_JOBS", []),
                        patch.object(queue_manager, "_prompt_refusal", return_value="")]
        for p in self.patches:
            p.start()
        self.root = root
        cancel.reset()

    def tearDown(self):
        cancel.reset()
        for p in reversed(self.patches):
            p.stop()
        self.folder.cleanup()

    def test_a_job_that_finishes_as_stop_is_clicked_is_done(self):
        job = queue_manager.Job(id=1, kind="test_kind", label="test", params={})

        def runner(params, cb):
            cancel.request()  # Stop arrives after the last step, before the job returns
            return True, ["out.png"], [], ""

        with patch.dict(queue_manager._RUNNERS, {"test_kind": runner}):
            queue_manager._JOBS.append(job)
            queue_manager._run(job)
        self.assertEqual(job.status, queue_manager.DONE)
        self.assertEqual(job.files, ["out.png"])

    def test_a_stopped_job_that_did_not_finish_is_stopped(self):
        job = queue_manager.Job(id=2, kind="test_kind", label="test", params={})

        def runner(params, cb):
            cancel.request()
            cb(0.5, "working")  # raises Cancelled
            return True, [], [], ""

        with patch.dict(queue_manager._RUNNERS, {"test_kind": runner}):
            queue_manager._JOBS.append(job)
            queue_manager._run(job)
        self.assertEqual(job.status, queue_manager.STOPPED)

    def test_stop_is_requested_while_the_queue_lock_is_held(self):
        job = queue_manager.Job(id=3, kind="test_kind", label="test", params={}, status=queue_manager.RUNNING)
        queue_manager._JOBS.append(job)
        held = []
        real = cancel.request

        def request():
            # _COND is a Condition over an RLock: another thread cannot take it while stop_current holds it
            got = []
            t = threading.Thread(target=lambda: got.append(queue_manager._COND.acquire(timeout=0.05)))
            t.start()
            t.join()
            if got and got[0]:
                queue_manager._COND.release()
            held.append(not got[0])
            real()

        with patch.object(cancel, "request", side_effect=request):
            self.assertTrue(queue_manager.stop_current(3))
        self.assertEqual(held, [True])

    def test_a_job_that_fails_to_queue_leaves_no_attachments(self):
        source = self.root / "photo.png"
        source.write_bytes(b"png")
        with self.assertRaises(FileNotFoundError):
            queue_manager.add("test_kind", "test", {}, files={"photos": [str(source), str(self.root / "gone.png")]})
        self.assertFalse(any((self.root / "queue_files").glob("*/*")))
        self.assertEqual(queue_manager._JOBS, [])

    def test_a_new_attempt_is_marked_in_the_stage_log(self):
        job = queue_manager.Job(id=4, kind="test_kind", label="test", params={})
        queue_manager._stage_log(job)(0.5, "first")
        queue_manager._stage_log(job)(0.5, "second")
        lines = (self.root / "stages" / "4.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual([line.split("\t")[-1] for line in lines], ["first", "# new attempt", "second"])


if __name__ == "__main__":
    unittest.main()
