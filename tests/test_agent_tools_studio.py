# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Agent's tools for the queue, the PC, the settings and the Gallery (app/agent_tools_studio.py)."""
from types import SimpleNamespace
from unittest.mock import patch

from app import agent_preflight, agent_tools_studio, ffmpeg_utils, history, prefs, queue_manager, vram
from app.downloader import BLOCKED_PREFIX
from tests.agent_base import AgentCase, picture


class QueueToolTests(AgentCase):
    def add(self, host=None, **fields):
        if host is None:
            job = queue_manager.add("obraz", "A road", {"prompt": "p"})
        else:
            with host.queue_scope():
                job = queue_manager.add("obraz", "A street", {"prompt": "p"})
        for name, value in fields.items():
            setattr(job, name, value)
        return job

    def test_the_run_sees_its_own_jobs_and_only_a_count_of_the_owners(self):
        host = self.host()
        mine = self.add(host)
        self.add()
        self.add(status=queue_manager.RUNNING)
        found = self.ok(host, "get_queue")
        self.assertEqual([row["id"] for row in found["mine"]], [mine.id])
        self.assertEqual((found["others_waiting"], found["paused"], "others" in found), (2, False, False))
        self.assertEqual(found["mine"][0]["status"], "queued")
        wide = self.ok(self.host(**self.allow(queue_all=True)), "get_queue")  # another run: it sees all three waiting
        self.assertEqual(len(wide["others"]), 3)

    def test_waiting_for_finished_jobs_answers_at_once_with_their_files(self):
        host = self.host()
        job = self.add(host)
        self.finish_jobs(job.id, files=[str(self.root / "outputs" / "street.png")])
        answer = self.ok(host, "wait_for_jobs", job_ids=[job.id])
        self.assertTrue(answer["all_finished"])
        self.assertEqual((answer["jobs"][0]["status"], answer["jobs"][0]["files"]), ("done", ["street.png"]))
        self.assertEqual(self.clock.slept, [])
        self.assertAlmostEqual(host.store.read()["gpu"]["seconds"], 30.0, delta=1)

    def test_a_wait_that_runs_out_of_time_says_so_and_the_jobs_stay(self):
        host = self.host()
        job = self.add(host)
        answer = self.ok(host, "wait_for_jobs", job_ids=[job.id], timeout_seconds=10)
        self.assertFalse(answer["all_finished"])
        self.assertIn("Time is up", answer["note"])
        self.assertAlmostEqual(sum(self.clock.slept), 10, delta=2.5)
        self.assertEqual(queue_manager.snapshot(), [job])

    def test_a_paused_queue_ends_the_wait_with_the_reason(self):
        host = self.host()
        job = self.add(host)
        with patch.object(queue_manager, "paused", return_value=True):
            answer = self.ok(host, "wait_for_jobs", job_ids=[job.id])
        self.assertIn("paused", answer["note"])
        self.assertEqual(self.clock.slept, [])

    def test_a_stop_ends_the_wait_and_a_strangers_job_cannot_be_waited_for(self):
        host = self.host()
        job = self.add(host)
        host.stop_requested = lambda: True
        self.refused(host, "wait_for_jobs", "OWNER_STOPPED", job_ids=[job.id])
        self.refused(self.host(), "wait_for_jobs", "NOT_OWNED", job_ids=[self.add().id])
        self.refused(self.host(), "wait_for_jobs", "NOT_FOUND", job_ids=[4242])

    def test_a_job_that_needed_a_model_is_marked_so_the_agent_can_tell_the_owner(self):
        host = self.host()
        job = self.add(host)
        self.finish_jobs(job.id, status=queue_manager.ERROR, error=f"{BLOCKED_PREFIX}: Wan 2.2.")
        row = self.ok(host, "wait_for_jobs", job_ids=[job.id])["jobs"][0]
        self.assertEqual((row["status"], row["code"]), ("failed", "MODELS_MISSING"))

    def test_what_the_studio_noticed_during_a_job_is_given_to_the_agent_so_it_can_act_and_report(self):
        host = self.host()
        job = self.add(host)
        self.finish_jobs(job.id)
        job.notes = ["6 clips ready on the timeline.", "Scenes 1 → 2 · The action is not continuous. " + "x" * 900]
        notes = self.ok(host, "wait_for_jobs", job_ids=[job.id])["jobs"][0]["notes"]
        self.assertEqual(len(notes), 2)
        self.assertIn("not continuous", notes[1])
        self.assertLessEqual(len(notes[1]), 500)
        self.finish_jobs(job.id)
        job.notes = []
        self.assertNotIn("notes", self.ok(host, "wait_for_jobs", job_ids=[job.id])["jobs"][0])

    def test_the_run_stops_its_own_jobs_and_the_whole_queue_only_with_permission(self):
        host = self.host()
        mine, theirs = self.add(host), self.add()
        self.assertEqual(self.ok(host, "stop_jobs", job_ids=[mine.id]), {"stopped": [mine.id]})
        self.refused(host, "stop_jobs", "NOT_OWNED", job_ids=[theirs.id])
        self.refused(host, "queue_control", "PERMISSION", action="pause")
        with patch.object(queue_manager, "set_paused") as paused:
            self.assertIn("pause done", self.ok(self.host(**self.allow(queue_all=True)), "queue_control",
                                                action="pause"))
            paused.assert_called_once_with(True)


class StatusAndSettingsToolTests(AgentCase):
    def test_the_status_tells_the_gpu_the_disk_and_the_models_still_missing(self):
        host = self.host(mode="produce")
        with patch.object(vram, "get_gpu_info", return_value=vram.GpuInfo(True, "RTX", 16.0, 12.5)), \
                patch.object(agent_preflight, "missing_models", return_value=[("wan", "Wan 2.2", 3 * 2 ** 30)]), \
                patch.object(agent_tools_studio.agent_store, "free_gb", return_value=420.0):
            status = self.ok(host, "get_status")
        self.assertEqual(status["gpu"], {"name": "RTX", "memory_gb": 16.0, "free_gb": 12.5})
        self.assertEqual((status["mode"], status["disk_free_gb"], status["downloads_allowed"]),
                         ("produce", 420.0, False))
        self.assertNotIn("gpu_minutes_budget", status)
        self.assertEqual(status["missing_models"], [{"name": "Wan 2.2", "download_gb": 3.0}])

    def test_the_settings_are_read_and_changed_only_with_permission_and_never_the_folder(self):
        host = self.host(**self.allow(settings=True))
        self.assertEqual(self.ok(host, "get_settings")["qc"], True)
        self.assertEqual(self.ok(host, "set_preference", key="qc", value="false"), {"key": "qc", "old": True, "new": False})
        self.assertFalse(prefs.load()["qc"])
        self.refused(host, "set_preference", "INVALID", key="qc", value="maybe")
        self.refused(host, "set_preference", "INVALID", key="folder", value="C:/x")  # not a choice of the schema
        self.refused(self.host(), "set_preference", "PERMISSION", key="qc", value="true")
        self.assertEqual(self.ok(host, "set_preference", key="tier", value="lite")["new"], "lite")


class GalleryToolTests(AgentCase):
    def result(self, name, prompt="a cup", suffix=".png"):
        path = self.root / "outputs" / f"{name}{suffix}"
        picture(path, size=(64, 48)) if suffix == ".png" else path.write_bytes(b"not a real video")
        history.record({"files": [str(path)], "kind": "obraz", "prompt": prompt})
        return path

    def test_the_newest_results_are_listed_with_the_prompt_that_made_them(self):
        self.result("one", "A cup on a table")
        self.result("two", "A saucer", ".mp4")
        rows = self.ok(self.host(), "list_gallery")
        self.assertEqual({row["file"]: row["kind"] for row in rows}, {"one.png": "image", "two.mp4": "video"})
        self.assertEqual([row["prompt"] for row in rows if row["file"] == "one.png"], ["A cup on a table"])
        self.assertEqual([row["file"] for row in self.ok(self.host(), "list_gallery", kind="video")], ["two.mp4"])
        self.assertEqual(self.ok(self.host(), "list_gallery", limit=1).__len__(), 1)

    def test_a_picture_is_shown_to_the_model_only_when_the_owner_allowed_sending_pictures(self):
        self.result("one")
        silent = self.call(self.host(send_images=False), "inspect_media", file="one.png")
        self.assertTrue(silent.ok and not silent.images)
        self.assertIn("not shown", silent.text)
        shown = self.call(self.host(send_images=True), "inspect_media", file="one.png")
        self.assertEqual(len(shown.images), 1)
        self.assertEqual(shown.images[0][:4], b"\x89PNG")
        self.assertIn('"width":64', shown.text)

    def test_a_frame_of_a_video_is_shown_with_its_length(self):
        self.result("clip", suffix=".mp4")
        host = self.host(send_images=True)

        def frame(video, seconds, out):
            picture(out, size=(80, 60))
            return True, ""

        with patch.object(ffmpeg_utils, "probe_duration", return_value=6.0), \
                patch.object(ffmpeg_utils, "frame_at", side_effect=frame) as taken:
            shown = self.call(host, "inspect_media", file="clip.mp4", at_seconds=2)
        self.assertEqual(taken.call_args.args[1], 2)
        self.assertEqual((len(shown.images), '"seconds":6.0' in shown.text), (1, True))

    def test_only_a_recorded_result_of_the_gallery_can_be_looked_at(self):
        self.result("one")
        picture(self.root / "outputs" / "stray.png")
        for bad in ("stray.png", "../one.png", "nothing.png", str(self.root / "outputs" / "one.png")):
            self.refused(self.host(), "inspect_media", "NOT_FOUND", file=bad)
