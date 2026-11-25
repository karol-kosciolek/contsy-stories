# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Video and Image pages queue through app/quick_jobs.py, which the Agent page uses too."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image

from app import llm, queue_manager, quick_jobs, refs, ui_images, ui_video
from app import cast as cast_mod


class QuickJobsTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        enter = self.context.enter_context
        self.root = Path(enter(tempfile.TemporaryDirectory())).resolve()
        enter(patch.object(queue_manager, "_JOBS", []))
        enter(patch.object(queue_manager, "_STATE", self.root / "queue.json"))
        enter(patch.object(queue_manager, "FILES_DIR", self.root / "queue_files"))
        enter(patch.object(queue_manager, "_STATE_INFO", {"next_id": 1, "paused": False}))
        # no character or voice of the live library is read: nobody is named, nobody is picked
        self.check = enter(patch.object(cast_mod, "check_prompt", return_value=([], "", [])))
        enter(patch.object(cast_mod, "picked", return_value=[]))
        self.picture = self.root / "ref.png"
        Image.new("RGB", (8, 8), (90, 90, 90)).save(self.picture)

    def test_a_clip_is_queued_with_its_settings_and_copies_of_its_files(self):
        queued = quick_jobs.queue_video("A cup on a table [1]", [str(self.picture)], str(self.picture), 5.0,
                                        "Landscape 16:9", 1)
        job = queued.job
        self.assertEqual(job.kind, "wideo")
        self.assertEqual(job.label, "Video 5 s: A cup on a table [1]")
        self.assertEqual({key: job.params[key] for key in ("prompt", "seconds", "aspect", "sound")},
                         {"prompt": "A cup on a table [1]", "seconds": 5, "aspect": "Landscape 16:9", "sound": True})
        self.assertTrue(Path(job.params["image"]).is_file() and str(self.root) in job.params["image"])
        self.assertEqual(len(job.params["refs"]), 1)
        self.assertEqual(queue_manager.snapshot(), [job])

    def test_a_picture_is_queued_in_its_format(self):
        job = quick_jobs.queue_image("A street at night", [], "Portrait 9:16").job
        self.assertEqual((job.kind, job.label), ("obraz", "Image Portrait 9:16: A street at night"))
        self.assertEqual(job.params["format"], "Portrait 9:16")

    def test_every_refusal_queues_nothing_and_says_why(self):
        cases = [
            ("   ", [], "Enter a prompt."),
            ("A cup [2]", [str(self.picture)], "refers to image [2], but only 1 reference images"),
            ("A cup", [str(self.picture)] * (refs.MAX_REFS + 1), f"Up to {refs.MAX_REFS} reference images"),
        ]
        for prompt, paths, expected in cases:
            for queue in (lambda p, r: quick_jobs.queue_video(p, r, None, 5, "Landscape 16:9", True),
                          lambda p, r: quick_jobs.queue_image(p, r, "Square 1:1")):
                with self.subTest(prompt=prompt), self.assertRaises(quick_jobs.Refused) as caught:
                    queue(prompt, paths)
                self.assertIn(expected, str(caught.exception))
        self.assertEqual(queue_manager.snapshot(), [])

    def test_the_content_rule_is_shown_as_it_always_was_and_the_others_get_a_mark(self):
        with patch.object(llm, "blocked", return_value=True), self.assertRaises(quick_jobs.Refused) as caught:
            quick_jobs.queue_image("anything", [], "Square 1:1")
        self.assertEqual(caught.exception.shown, llm.BLOCK_MSG)
        with self.assertRaises(quick_jobs.Refused) as caught:
            quick_jobs.queue_image("", [], "Square 1:1")
        self.assertEqual(caught.exception.shown, "❌ Enter a prompt.")

    def test_a_characters_refusal_comes_through_and_its_notes_come_back(self):
        self.check.return_value = ([], "Up to 5 characters can be in one video or photo.", [])
        with self.assertRaisesRegex(quick_jobs.Refused, "Up to 5 characters"):
            quick_jobs.queue_image("a crowd", [], "Square 1:1")
        self.check.return_value = ([{"slug": "ola", "name": "Ola"}], "", ["No reference photos yet."])
        queued = quick_jobs.queue_image("Ola smiles", [], "Square 1:1", picked=["ola"])
        self.assertEqual((queued.notes, [c["slug"] for c in queued.found]), (["No reference photos yet."], ["ola"]))
        self.assertEqual(queued.job.params["character"], "ola")

    def test_the_owner_block_reaches_a_job_queued_this_way(self):
        with queue_manager.owner("agent:r1", downloads=False, prefix="Agent · "):
            job = quick_jobs.queue_image("A street", [], "Square 1:1").job
        self.assertEqual((job.owner, job.downloads, job.label), ("agent:r1", False, "Agent · Image Square 1:1: A street"))


class PagesUseItTests(unittest.TestCase):
    """The two pages' buttons give the same words as before and queue through the helper."""

    def handler(self, build):
        with gr.Blocks() as demo:
            build()
        return next(block.fn for block in demo.fns.values() if getattr(block.fn, "__name__", "") == "add")

    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        enter = self.context.enter_context
        enter(patch.object(queue_manager, "_JOBS", []))
        enter(patch.object(queue_manager, "_STATE_INFO", {"next_id": 1, "paused": False}))

    def test_the_video_page_shows_a_refusal_with_its_mark_and_a_queued_job_with_its_number(self):
        add = self.handler(ui_video.build_video_tab)
        refused = add("", [], None, 5, "Landscape 16:9", True, [])
        self.assertTrue("st-err" in refused and refused.endswith("Enter a prompt."), refused)
        job = queue_manager.Job(7, "wideo", "x", {})
        with patch.object(quick_jobs, "queue_video", return_value=quick_jobs.Queued(job)) as queue, \
                patch.object(queue_manager, "position", return_value=0):
            shown = add("A cup", [("a.png", "caption")], None, 5, "Landscape 16:9", True, ["ola"])
        self.assertIn("#7", shown)
        queue.assert_called_once_with("A cup", ["a.png"], None, 5, "Landscape 16:9", True, ["ola"])

    def test_the_image_page_does_the_same(self):
        add = self.handler(ui_images.build_images_tab)
        refused = add("", [], "Square 1:1", [])
        self.assertTrue("st-err" in refused and refused.endswith("Enter a prompt."), refused)
        job = queue_manager.Job(8, "obraz", "x", {})
        with patch.object(quick_jobs, "queue_image", return_value=quick_jobs.Queued(job)) as queue, \
                patch.object(queue_manager, "position", return_value=2):
            shown = add("A cup", None, "Square 1:1", None)
        self.assertIn("#8", shown)
        self.assertIn("2 jobs ahead", shown)
        queue.assert_called_once_with("A cup", [], "Square 1:1", None)


if __name__ == "__main__":
    unittest.main()
