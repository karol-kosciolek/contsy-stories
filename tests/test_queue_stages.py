# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The queue writes each new step text of a running job with its time to history/stages/<id>.tsv."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import queue_manager


class StageLogTests(unittest.TestCase):
    def test_every_new_step_is_written_once_with_its_time_and_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            job = queue_manager.Job(id="7", kind="test_kind", label="test", params={})

            def runner(params, cb):
                cb(0.1, "Voice: preparing…")
                cb(0.2, "Voice: preparing…")  # same text again: not a new stage
                cb(0.5, "Background audio: generating")
                cb(1.0)  # no text: nothing to log
                return True, [], []

            # _save must be patched: _run saves the queue, and the real config/queue.json holds the user's jobs
            with patch.object(queue_manager, "STAGES_DIR", Path(folder)), \
                    patch.object(queue_manager, "_save"), \
                    patch.object(queue_manager, "_STATE", Path(folder) / "queue.json"), \
                    patch.dict(queue_manager._RUNNERS, {"test_kind": runner}), \
                    patch.object(queue_manager, "_prompt_refusal", return_value=""):
                queue_manager._run(job)
            rows = [line.split("\t") for line in (Path(folder) / "7.tsv").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row[2] for row in rows], ["Voice: preparing…", "Background audio: generating"])
        self.assertEqual([row[1] for row in rows], ["0.100", "0.500"])
        self.assertTrue(all(float(row[0]) >= 0 for row in rows))

    def test_a_log_that_cannot_be_written_does_not_stop_the_job(self):
        job = queue_manager.Job(id="8", kind="test_kind", label="test", params={})
        with patch.object(queue_manager, "STAGES_DIR", Path("Z:/missing/drive/stages")), \
                patch.object(Path, "mkdir", side_effect=OSError("no drive")):
            write = queue_manager._stage_log(job)
            write(0.5, "step")  # must not raise


if __name__ == "__main__":
    unittest.main()
