# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The two Timeline functions the Agent page needs: a save that waits for the project's jobs, and a check of scenes
that queues nothing."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import config, history, queue_manager, scene_parts, timeline


class AgentTimelineTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        enter = self.context.enter_context
        self.root = Path(enter(tempfile.TemporaryDirectory())).resolve()
        enter(patch.object(timeline, "DIR", self.root / "projects"))
        enter(patch("app.items.DIR", self.root / "items"))
        enter(patch.object(config, "OUTPUTS_DIR", self.root / "outputs"))
        enter(patch.object(config, "HISTORY_DIR", self.root / "history"))
        enter(patch.object(history, "_HISTORY_FILE", self.root / "history/history.jsonl"))
        enter(patch.object(timeline.model_manager, "ensure", return_value=None))
        enter(patch.object(timeline.sound, "available", return_value=True))
        enter(patch.object(timeline.tts, "available", return_value=True))
        enter(patch.object(timeline.model_manager, "VERIFIED_FILE", self.root / "verified_models.json"))
        enter(patch.object(timeline.model_manager, "_STORED", None))
        enter(patch.object(queue_manager, "_JOBS", []))
        enter(patch.object(queue_manager, "_STATE", self.root / "queue.json"))
        enter(patch.object(queue_manager, "_STATE_INFO", {"next_id": 1, "paused": False}))
        enter(patch.object(queue_manager, "_save"))

    def save_idle(self, scenes, previous=None, title="Coffee spot"):
        return timeline.save_idle(previous.id if previous else None, previous.revision if previous else None, title,
                                  tuple(scenes), "Landscape 16:9", True, [])

    def job(self, project, status=queue_manager.PENDING, kind=timeline.PREPARE):
        job = queue_manager.Job(len(queue_manager._JOBS) + 1, kind, "x", {"project": project.id}, status=status)
        queue_manager._JOBS.append(job)
        return job

    def test_it_saves_like_save_when_nothing_of_the_project_is_queued(self):
        first = self.save_idle([timeline.Scene("A cup on a table", 4)])
        second = self.save_idle([timeline.Scene("A cup on a table", 4), timeline.Scene("A saucer", 3)], first)
        self.assertEqual((first.revision, second.revision), (1, 2))
        self.assertEqual(timeline.load(first.id), second)

    def test_a_waiting_or_running_job_of_the_project_refuses_the_save_and_leaves_it_as_it_was(self):
        first = self.save_idle([timeline.Scene("A cup on a table", 4)])
        for status in (queue_manager.PENDING, queue_manager.RUNNING):
            queue_manager._JOBS.clear()
            self.job(first, status)
            with self.subTest(status=status), self.assertRaises(timeline.Busy):
                self.save_idle([timeline.Scene("Something else", 2)], first)
            self.assertEqual(timeline.load(first.id), first)

    def test_a_finished_job_or_another_projects_job_does_not_block(self):
        first = self.save_idle([timeline.Scene("A cup on a table", 4)])
        other = self.save_idle([timeline.Scene("A tree", 2)], title="Garden")
        self.job(first, queue_manager.DONE)
        self.job(other, queue_manager.RUNNING)
        self.job(first, queue_manager.PENDING, kind="wideo")  # a clip of the Video page: not the project's job
        saved = self.save_idle([timeline.Scene("A cup on a table", 5)], first)
        self.assertEqual(saved.revision, 2)

    def test_a_stale_revision_and_a_deleted_project_are_still_refused_as_before(self):
        first = self.save_idle([timeline.Scene("A cup on a table", 4)])
        self.save_idle([timeline.Scene("A cup on a table", 5)], first)
        with self.assertRaisesRegex(ValueError, "another tab"):
            self.save_idle([timeline.Scene("Late", 1)], first)
        with self.assertRaises(timeline.Deleted):
            timeline.save_idle("0" * 32, 1, "Gone", (timeline.Scene("x", 1),), "Landscape 16:9", True, [])

    def errors(self, project, indices=None):
        return [(index, issue.text) for index, issue in timeline.check_scenes(project, indices)
                if issue.level == scene_parts.ERROR]

    def test_check_scenes_reports_a_scene_without_a_picture_description_and_queues_nothing(self):
        project = self.save_idle([timeline.Scene("A cup on a table", 4), timeline.Scene("", 3)])
        found = self.errors(project)
        self.assertEqual([index for index, _ in found], [1])
        self.assertIn("description", found[0][1])
        self.assertNotIn("Scene 2:", found[0][1])  # the scene is named by the index, once
        self.assertEqual(queue_manager.snapshot(), [])

    def test_check_scenes_takes_a_choice_of_scenes_and_ignores_numbers_that_are_not_scenes(self):
        project = self.save_idle([timeline.Scene("", 2), timeline.Scene("", 3), timeline.Scene("A cup", 4)])
        self.assertEqual([index for index, _ in self.errors(project, [1])], [1])
        self.assertEqual(timeline.check_scenes(project, [7, -1]), [])
        self.assertEqual({index for index, _ in self.errors(project)}, {0, 1})

    def test_a_scene_that_is_ready_has_no_error_and_nothing_is_taken_from_the_editors_memory(self):
        project = self.save_idle([timeline.Scene("A cup on a table, steam rising", 4)])
        def memo(kind, key, compute, ttl=0.0):
            if kind in ("issues", "voices"):  # what the editor remembers of the scenes and the library's voices
                raise AssertionError(f"the editor's memory of {kind} was used")
            return compute()

        with patch.object(timeline, "_memo", side_effect=memo):
            self.assertEqual(self.errors(project), [])

    def test_the_same_problem_is_not_reported_twice_and_errors_come_first(self):
        project = self.save_idle([timeline.Scene("", 3)])
        issues = [issue for _, issue in timeline.check_scenes(project)]
        self.assertEqual(len({issue.text for issue in issues}), len(issues))
        levels = [issue.level for issue in issues]
        self.assertEqual(levels, sorted(levels, key=lambda level: {"error": 0, "warning": 1, "info": 2}[level]))


if __name__ == "__main__":
    unittest.main()
