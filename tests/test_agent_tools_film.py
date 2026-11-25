# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Agent's film tools (app/agent_tools_film.py) on real Timeline projects in a temporary folder; the jobs that need
a GPU are queued by the real queue and never run."""
from pathlib import Path
from unittest.mock import patch

from app import llm, queue_manager, timeline, ui_images
from app.agent_tools_film import STILL_FORMATS
from tests.agent_base import AgentCase, picture

SCENE = {"prompt": "A barista hands a cup of coffee over a wooden counter, morning light", "seconds": 5}


class ProjectToolTests(AgentCase):
    def create(self, host, **extra):
        return self.ok(host, "save_project", title="Coffee ad", scenes=[SCENE], **extra)

    def owners_project(self, scenes=None):
        return timeline.save(None, None, "Owner film", tuple(scenes or [timeline.Scene("A tree in wind", 4)]),
                             "Landscape 16:9", True, [])

    def test_a_project_is_created_whole_and_read_back_with_its_dialogue_and_sound(self):
        self.saved_character("Mia")
        host = self.host()
        scenes = [{**SCENE, "cast": ["mia"], "lines": [{"text": "Here you go!", "speaker": "mia"}],
                   "background": "a quiet cafe", "effects": [{"text": "a cup set on wood", "at": 1.0}]},
                  {"prompt": "Close on the cup, steam rising", "seconds": 2.5, "cut": True}]
        made = self.ok(host, "save_project", title="Coffee ad", scenes=scenes, characters=["mia"], aspect="Portrait 9:16",
                       subtitles=True)
        self.assertEqual((made["revision"], made["aspect"], made["subtitles"], made["seconds"]),
                         (1, "Portrait 9:16", True, 7.5))
        self.assertTrue(host.is_mine("project", made["id"]))
        read = self.ok(host, "get_project", project_id=made["id"])
        self.assertEqual(read["scenes"][0]["lines"], [{"text": "Here you go!", "speaker": "mia"}])
        self.assertEqual((read["scenes"][0]["cast"], read["scenes"][1]["cut"], read["scenes"][1]["seconds"]),
                         (["mia"], True, 2.5))
        self.assertEqual(read["state"]["clips"], [False, False])
        self.assertFalse(read["state"]["job_waiting"])
        self.assertEqual([row["title"] for row in self.ok(host, "list_projects")], ["Coffee ad"])

    def test_changing_a_project_needs_the_revision_that_was_read_and_copies_the_owners_project_once(self):
        project = self.owners_project()
        host = self.host()
        self.refused(host, "save_project", "INVALID", project_id=project.id, title="X", scenes=[SCENE])
        self.refused(host, "save_project", "STALE", project_id=project.id, expected_revision=7, title="X",
                     scenes=[SCENE])
        self.assertEqual(host.store.read().get("resources", {}).get("backups", {}), {})
        changed = self.ok(host, "save_project", project_id=project.id, expected_revision=1, title="Owner film 2",
                          scenes=[SCENE])
        self.assertEqual(changed["revision"], 2)
        self.assertTrue((host.store.path("backups", f"001-project-{project.id}") / "project.json").is_file())
        self.ok(host, "save_project", project_id=project.id, expected_revision=2, title="Owner film 3", scenes=[SCENE])
        self.assertEqual(list(host.store.read()["resources"]["backups"]), [f"project:{project.id}"])
        self.assertEqual(len(host.store.read()["changes"]), 2)
        self.refused(host, "save_project", "NOT_FOUND", project_id="f" * 32, expected_revision=1, title="X",
                     scenes=[SCENE])

    def test_a_project_the_run_made_is_changed_without_a_copy(self):
        host = self.host()
        made = self.create(host)
        self.ok(host, "save_project", project_id=made["id"], expected_revision=1, title="Coffee ad v2", scenes=[SCENE])
        self.assertEqual(host.store.read()["resources"]["backups"], {})

    def test_a_project_with_a_job_is_not_changed(self):
        project = self.owners_project()
        queue_manager._JOBS.append(queue_manager.Job(1, timeline.PREPARE, "x", {"project": project.id},
                                                     status=queue_manager.RUNNING))
        self.refused(self.host(), "save_project", "BUSY", project_id=project.id, expected_revision=1, title="X",
                     scenes=[SCENE])
        self.assertEqual(timeline.load(project.id), project)

    def test_the_content_rule_the_write_permission_and_the_scene_limits_apply(self):
        host = self.host()
        with patch.object(llm, "blocked", return_value=True):
            self.refused(host, "save_project", "CONTENT_RULE", title="X", scenes=[SCENE])
        self.refused(self.host(permissions={"write": False}), "save_project", "PERMISSION", title="X", scenes=[SCENE])
        self.refused(host, "save_project", "INVALID", title="X", scenes=[{**SCENE, "seconds": 45}])
        self.refused(host, "save_project", "INVALID", title="X", scenes=[{**SCENE, "cast": ["nobody"]}])
        self.refused(host, "save_project", "INVALID", title="X", scenes=[SCENE], aspect="Wide")
        self.assertEqual(timeline.projects(), [])

    def test_an_attached_picture_becomes_a_scene_reference_and_is_shown_by_its_short_id(self):
        ref = timeline.store_reference(picture(self.root / "product.png", color=(200, 30, 30)))
        host = self.host(brief=[{"name": "product.png", "kind": "image", "ref": ref}])
        scene = {"prompt": "The product [1] on a white table", "seconds": 4, "references": [ref[:12]]}
        made = self.ok(host, "save_project", title="Product", scenes=[scene])
        self.assertEqual(made["scenes"][0]["references"], [ref[:12]])
        self.assertEqual(timeline.load(made["id"]).scenes[0].references, (ref,))
        again = self.ok(host, "save_project", project_id=made["id"], expected_revision=1, title="Product",
                        scenes=[{**scene, "seconds": 5}])
        self.assertEqual(again["scenes"][0]["references"], [ref[:12]])
        self.refused(host, "save_project", "NOT_FOUND", title="P", scenes=[{**scene, "references": ["f" * 12]}])

    def test_check_project_reports_what_generation_would_refuse_and_queues_nothing(self):
        host = self.host()
        made = self.ok(host, "save_project", title="Draft", scenes=[SCENE, {"prompt": " ", "seconds": 3}])
        report = self.ok(host, "check_project", project_id=made["id"])
        self.assertFalse(report["ready"])
        self.assertEqual({row["scene"] for row in report["issues"] if row["level"] == "error"}, {1})
        self.assertEqual(self.ok(host, "check_project", project_id=made["id"], scenes=[0])["ready"], True)
        self.assertEqual(queue_manager.snapshot(), [])

    def test_a_project_that_does_not_exist_is_not_found(self):
        for tool, arguments in (("get_project", {}), ("check_project", {}), ("list_takes", {"scene": 0})):
            self.refused(self.host(), tool, "NOT_FOUND", project_id="e" * 32, **arguments)
        self.refused(self.host(), "get_project", "INVALID", project_id="../x")


class ProductionToolTests(AgentCase):
    def setUp(self):
        super().setUp()
        self.project = timeline.save(None, None, "Owner film", (timeline.Scene("A tree in wind", 4),
                                                               timeline.Scene("A river", 3)), "Landscape 16:9", True, [])
        self.enqueued = []

        def enqueue(project, kind, *, scene_index=None, require_merge=False):
            self.enqueued.append((kind, scene_index))
            return queue_manager.add(kind, f"{kind} · {project.title}", {"project": project.id, "seconds": 4})

        self.context.enter_context(patch.object(timeline, "enqueue", side_effect=enqueue))

    def test_only_the_sound_preview_is_allowed_in_a_run_that_prepares(self):
        host = self.host(mode="prepare")
        for action in ("prepare_missing", "prepare_scene", "export", "retry_failed"):
            self.refused(host, "timeline_job", "MODE", project_id=self.project.id, action=action, scene=0)
        done = self.ok(host, "timeline_job", project_id=self.project.id, action="sound_preview", scene=1)
        self.assertEqual(self.enqueued, [(timeline.SOUND, 1)])
        job = queue_manager.snapshot()[0]
        self.assertEqual((job.id, job.owner, job.downloads), (done["job_id"], host.owner_tag, False))

    def test_a_run_that_produces_queues_the_clips_and_the_export(self):
        host = self.host(mode="produce")
        self.ok(host, "timeline_job", project_id=self.project.id, action="prepare_missing")
        self.ok(host, "timeline_job", project_id=self.project.id, action="prepare_scene", scene=1)
        self.ok(host, "timeline_job", project_id=self.project.id, action="export")
        self.assertEqual(self.enqueued, [(timeline.PREPARE, None), (timeline.PREPARE, 1), (timeline.PRODUCE, None)])
        self.refused(host, "timeline_job", "INVALID", project_id=self.project.id, action="prepare_scene")
        self.refused(host, "timeline_job", "INVALID", project_id=self.project.id, action="retry_failed")
        self.refused(self.host(mode="produce", permissions={"write": False}), "timeline_job", "PERMISSION",
                     project_id=self.project.id, action="export")

    def test_a_failed_job_is_queued_again_as_the_same_kind_of_job(self):
        host = self.host(mode="produce")
        with patch.object(timeline, "failure", return_value={"scene": 1, "kind": timeline.PREPARE, "error": "no face"}):
            self.ok(host, "timeline_job", project_id=self.project.id, action="retry_failed")
        with patch.object(timeline, "failure", return_value={"scene": 0, "kind": timeline.EDIT, "error": "x"}), \
                patch.object(timeline, "retry_edit", side_effect=lambda project: queue_manager.add(
                    timeline.EDIT, "retry", {"project": project.id, "seconds": 3})) as retry:
            self.ok(host, "timeline_job", project_id=self.project.id, action="retry_failed")
            retry.assert_called_once()
        self.assertEqual(self.enqueued, [(timeline.PREPARE, 1)])

    def test_a_scene_is_edited_from_its_own_clip_with_the_scope_and_a_copy_of_the_project(self):
        host = self.host(mode="produce")
        seen = {}

        def edit(saved, index, scene, *, change="", scope="scene"):
            seen.update(index=index, scene=scene, change=change, scope=scope)
            return saved, queue_manager.add(timeline.EDIT, "Edit", {"project": saved.id, "seconds": 4})

        with patch.object(timeline, "edit", side_effect=edit):
            done = self.ok(host, "edit_scene", project_id=self.project.id, expected_revision=1, scene=1, scope="film",
                           prompt="A wide river at dusk", picture_change="the sign says OPEN", background="water")
        self.assertEqual((seen["index"], seen["change"], seen["scope"]), (1, "the sign says OPEN", "film"))
        self.assertEqual((seen["scene"].prompt, seen["scene"].background), ("A wide river at dusk", "water"))
        self.assertTrue(done["job_id"])
        self.assertTrue((host.store.path("backups", f"001-project-{self.project.id}") / "project.json").is_file())
        self.refused(host, "edit_scene", "STALE", project_id=self.project.id, expected_revision=9, scene=0)
        self.refused(host, "edit_scene", "INVALID", project_id=self.project.id, expected_revision=1, scene=5)
        self.refused(self.host(mode="prepare"), "edit_scene", "MODE", project_id=self.project.id, expected_revision=1,
                     scene=0)

    def test_new_dialogue_keeps_who_speaks_and_when(self):
        self.saved_character("Mia")
        host = self.host(mode="produce")
        spoken = timeline.save(None, None, "Talk", (timeline.Scene("Mia at a desk", 5, cast=("mia",), lines=(
            timeline.scene_parts.Cue("Good morning", "mia", 0.5),)),), "Landscape 16:9", True, ["mia"])
        seen = {}

        def edit(saved, index, scene, *, change="", scope="scene"):
            seen["scene"] = scene
            return saved, queue_manager.add(timeline.EDIT, "Edit", {"project": saved.id, "seconds": 4})

        with patch.object(timeline, "edit", side_effect=edit):
            self.ok(host, "edit_scene", project_id=spoken.id, expected_revision=1, scene=0, lines=["Good evening"])
        self.assertEqual([(cue.text, cue.speaker, cue.start) for cue in seen["scene"].lines],
                         [("Good evening", "mia", 0.5)])

    def test_takes_are_listed_and_chosen_and_a_busy_project_keeps_its_take(self):
        host = self.host()
        takes = [{"key": "k1", "label": "Take 1", "current": True, "anchor_ok": True},
                 {"key": "k2", "label": "Take 2", "current": False, "anchor_ok": False}]
        with patch.object(timeline, "takes", return_value=takes):
            self.assertEqual([row["key"] for row in self.ok(host, "list_takes", project_id=self.project.id, scene=0)],
                             ["k1", "k2"])
        with patch.object(timeline, "use_take", return_value=self.project) as use:
            self.ok(host, "choose_take", project_id=self.project.id, scene=0, key="k2")
            use.assert_called_once_with(self.project, 0, "k2")
            queue_manager._JOBS.append(queue_manager.Job(1, timeline.PREPARE, "x", {"project": self.project.id},
                                                         status=queue_manager.PENDING))
            self.refused(host, "choose_take", "BUSY", project_id=self.project.id, scene=0, key="k1")
            self.assertEqual(use.call_count, 1)
        self.refused(host, "list_takes", "INVALID", project_id=self.project.id, scene=9)


class PageToolTests(AgentCase):
    def test_the_still_formats_are_the_image_pages_formats(self):
        self.assertEqual(list(STILL_FORMATS), list(ui_images.SIZES))

    def test_a_clip_is_queued_as_the_video_page_would_and_marked_as_the_runs(self):
        ref = timeline.store_reference(picture(self.root / "start.png", color=(30, 90, 200)))
        host = self.host(mode="produce", brief=[{"name": "start.png", "kind": "image", "ref": ref}])
        done = self.ok(host, "queue_clip", prompt="Rain on a window at night [1]", seconds=4, aspect="Portrait 9:16",
                       pictures=[ref[:12]], start_picture=ref[:12], sound=False)
        job = queue_manager.snapshot()[0]
        self.assertEqual((job.id, job.kind, job.owner, job.downloads), (done["job_id"], "wideo", host.owner_tag, False))
        self.assertEqual((job.params["seconds"], job.params["aspect"], job.params["sound"]),
                         (4, "Portrait 9:16", False))
        self.assertTrue(Path(job.params["image"]).is_file() and len(job.params["refs"]) == 1)
        self.refused(host, "queue_clip", "INVALID", prompt="Rain [3]", seconds=4, aspect="Landscape 16:9")
        self.refused(host, "queue_clip", "NOT_FOUND", prompt="Rain", seconds=4, aspect="Landscape 16:9",
                     start_picture="f" * 12)
        with patch.object(llm, "blocked", return_value=True):
            self.refused(host, "queue_clip", "CONTENT_RULE", prompt="anything", seconds=4, aspect="Landscape 16:9")

    def test_a_picture_may_be_queued_in_a_run_that_only_prepares(self):
        host = self.host(mode="prepare")
        done = self.ok(host, "queue_still", prompt="A poster of a coffee cup", format="Portrait 3:4")
        job = queue_manager.snapshot()[0]
        self.assertEqual((job.id, job.kind, job.params["format"]), (done["job_id"], "obraz", "Portrait 3:4"))
        self.refused(host, "queue_still", "INVALID", prompt="x", format="Huge")
        self.refused(host, "queue_still", "INVALID", prompt="  ", format="Square 1:1")
