# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The agent's exact text over the film (set_overlays, app/agent_tools_film.py): saved with the project, shown when it
reads the project, warned about when a scene asks the picture for text, looked at in the finished film, and listed in the
script. The burning itself is tested in tests/test_overlays.py."""
import json
from unittest.mock import patch

from app import agent_delivery as delivery, agent_prompt, llm, overlays, queue_manager, timeline
from tests.agent_base import AgentCase
from tests.test_agent_eyes import EyesCase, decode

SCENE = {"prompt": "A lamp on a clean wooden table, soft evening light, calm empty space on the right", "seconds": 4}
CARD = {"text": "Lumio\nlumio.example/offer", "last_seconds": 2, "backdrop": "card", "size": "large"}


class OverlayToolTests(AgentCase):
    def create(self, host, **extra):
        return self.ok(host, "save_project", title="Lamp ad", scenes=[SCENE], **extra)

    def test_a_project_is_created_with_its_text_and_reads_it_back_with_when_it_is_shown(self):
        host = self.host()
        made = self.create(host, overlays=[CARD, {"text": "Light that listens", "start": 0.5, "end": 3, "position": "bottom"}])
        self.assertEqual(made["overlays"][0], {"text": "Lumio\nlumio.example/offer", "last_seconds": 2.0,
                                               "size": "large", "backdrop": "card"})
        self.assertEqual(made["overlay_times"], [[2.0, 4.0], [0.5, 3.0]])
        self.assertLess(list(made).index("overlays"), list(made).index("scenes"))  # a cut answer keeps the text
        read = self.ok(host, "get_project", project_id=made["id"])
        self.assertEqual((read["overlays"], read["overlay_times"]), (made["overlays"], made["overlay_times"]))
        self.assertEqual(len(timeline.load(made["id"]).overlays), 2)
        # what the agent reads can be sent back as it is: nothing changes
        again = self.ok(host, "set_overlays", project_id=made["id"], expected_revision=made["revision"],
                        overlays=read["overlays"])
        self.assertEqual(again["revision"], made["revision"])

    def test_a_save_that_names_no_overlays_keeps_them_and_an_empty_list_removes_them(self):
        host = self.host()
        made = self.create(host, overlays=[CARD])
        kept = self.ok(host, "save_project", project_id=made["id"], expected_revision=1, title="Lamp ad 2", scenes=[SCENE])
        self.assertEqual(len(kept["overlays"]), 1)
        gone = self.ok(host, "save_project", project_id=made["id"], expected_revision=kept["revision"], title="Lamp ad 2",
                       scenes=[SCENE], overlays=[])
        self.assertNotIn("overlays", gone)

    def test_set_overlays_replaces_the_list_and_needs_the_revision_that_was_read(self):
        host = self.host()
        made = self.create(host)
        self.refused(host, "set_overlays", "STALE", project_id=made["id"], expected_revision=9, overlays=[CARD])
        done = self.ok(host, "set_overlays", project_id=made["id"], expected_revision=1, overlays=[CARD])
        self.assertEqual((done["revision"], len(done["overlays"])), (2, 1))
        self.assertIn("timeline_job export", done["next"])
        self.assertEqual(list(done)[0], "next")
        same = self.ok(host, "set_overlays", project_id=made["id"], expected_revision=2, overlays=[CARD])
        self.assertEqual(same["revision"], 2)  # nothing changed
        removed = self.ok(host, "set_overlays", project_id=made["id"], expected_revision=2, overlays=[])
        self.assertNotIn("overlays", removed)
        self.assertEqual(timeline.load(made["id"]).overlays, ())

    def test_what_is_not_a_valid_text_is_refused_with_the_reason(self):
        host = self.host()
        made = self.create(host)
        arguments = {"project_id": made["id"], "expected_revision": 1}
        for bad, reason in (({"text": "ok", "start": 3, "end": 2}, "at least"),
                            ({"text": "ok", "start": 1, "end": 3, "last_seconds": 2}, "not both"),
                            ({"text": "a\nb\nc\nd\ne", "start": 0, "end": 2}, "at most 4 lines"),
                            ({"text": "ok", "start": 0, "end": 2, "position": "left"}, "position"),
                            ({"text": "a\x00b", "start": 0, "end": 2}, "control")):
            self.assertIn(reason, self.refused(host, "set_overlays", "INVALID", overlays=[bad], **arguments))
        self.refused(host, "set_overlays", "INVALID", overlays=[{"text": "x", "start": 0, "end": 2}] * 10, **arguments)
        self.assertEqual(timeline.load(made["id"]).overlays, ())

    def test_the_content_rule_the_permission_a_waiting_job_and_a_missing_project(self):
        host = self.host()
        made = self.create(host)
        with patch.object(llm, "blocked", return_value=True):
            self.refused(host, "set_overlays", "CONTENT_RULE", project_id=made["id"], expected_revision=1, overlays=[CARD])
            self.refused(host, "save_project", "CONTENT_RULE", title="X", scenes=[SCENE], overlays=[CARD])
        self.refused(self.host(permissions={"write": False}), "set_overlays", "PERMISSION", project_id=made["id"],
                     expected_revision=1, overlays=[CARD])
        self.refused(host, "set_overlays", "NOT_FOUND", project_id="e" * 32, expected_revision=1, overlays=[CARD])
        queue_manager._JOBS.append(queue_manager.Job(1, timeline.PREPARE, "x", {"project": made["id"]},
                                                     status=queue_manager.RUNNING))
        self.refused(host, "set_overlays", "BUSY", project_id=made["id"], expected_revision=1, overlays=[CARD])

    def test_the_owners_project_is_copied_once_before_the_agent_changes_its_text(self):
        project = timeline.save(None, None, "Owner film", (timeline.Scene("A tree in wind", 4),), "Landscape 16:9", True, [])
        host = self.host()
        self.ok(host, "set_overlays", project_id=project.id, expected_revision=1, overlays=[CARD])
        self.assertTrue((host.store.path("backups", f"001-project-{project.id}") / "project.json").is_file())
        self.assertEqual(len(host.store.read()["changes"]), 1)

    def test_check_project_warns_about_a_scene_that_asks_the_picture_for_text_and_about_texts_that_never_show(self):
        host = self.host()
        sign = {"prompt": 'A shop front, a neon sign reads "LUMIO OPEN" above the door', "seconds": 3}
        made = self.ok(host, "save_project", title="Shop", scenes=[SCENE, sign], overlays=[
            {"text": "A", "start": 0, "end": 3}, {"text": "B", "start": 2, "end": 5}, {"text": "Late", "start": 50, "end": 55}])
        report = self.ok(host, "check_project", project_id=made["id"])
        warnings = [(row["scene"], row["text"]) for row in report["issues"] if row["level"] == "warning"]
        self.assertTrue(report["ready"])  # warnings do not stop a film
        self.assertTrue(any(scene == 1 and "LUMIO OPEN" in text and "set_overlays" in text for scene, text in warnings))
        self.assertTrue(any(scene is None and "Overlays 1 and 2" in text for scene, text in warnings))
        self.assertTrue(any(scene is None and "Overlay 3 starts after the end" in text for scene, text in warnings))
        only = self.ok(host, "check_project", project_id=made["id"], scenes=[0])
        self.assertFalse(any("LUMIO" in row["text"] for row in only["issues"]))

    def test_check_project_warns_when_a_bottom_text_meets_the_subtitles(self):
        host = self.host()
        made = self.ok(host, "save_project", title="Subs", scenes=[SCENE], subtitles=True,
                       overlays=[{"text": "Low", "start": 0, "end": 2, "position": "bottom"}])
        warnings = [row["text"] for row in self.ok(host, "check_project", project_id=made["id"])["issues"]]
        self.assertTrue(any("Subtitles are on" in text and "bottom" in text for text in warnings), warnings)
        top = self.ok(host, "save_project", project_id=made["id"], expected_revision=1, title="Subs", scenes=[SCENE],
                      subtitles=True, overlays=[{"text": "High", "start": 0, "end": 2, "position": "top"}])
        self.assertFalse(any("Subtitles are on" in row["text"]
                             for row in self.ok(host, "check_project", project_id=top["id"])["issues"]))

    def test_the_prompt_rules_tell_the_agent_that_exact_text_is_not_drawn_by_the_model(self):
        rules = self.ok(self.host(), "get_prompt_rules", kind="video")
        self.assertIn("set_overlays", rules if isinstance(rules, str) else json.dumps(rules))

    def test_the_agent_is_told_never_to_ask_a_picture_for_exact_text(self):
        prompt = agent_prompt.system_prompt({})
        for expected in ("set_overlays", "cannot spell", "last_seconds", "word for word"):
            self.assertIn(expected, prompt)


class FilmLookTests(EyesCase):
    def test_looking_at_a_film_with_text_gives_its_times_and_a_frame_of_every_text(self):
        project = self.project()
        project = timeline.set_overlays(project.id, project.revision, [overlays.Overlay("THE END", last=2, backdrop="card"),
                                                                       overlays.Overlay("Late", 50, 55)])
        host = self.host()
        with patch.object(delivery, "export_of", return_value=self.film):
            result = self.call(host, "look_at_film", project_id=project.id)
        self.assertTrue(result.ok, result.text)
        data = json.loads(result.text)
        self.assertEqual([row["shown_from_to"] for row in data["text_burned_in"]], [[4.0, 6.0], None])
        self.assertEqual(len(result.images), 2)  # the scenes, and a frame in the middle of the text that is shown
        self.assertGreater(decode(result.images[1]).width, 100)  # a real sheet

    def test_the_text_comes_before_the_scenes_and_a_missing_room_drops_only_its_sheet(self):
        project = self.project()
        project = timeline.set_overlays(project.id, project.revision, [overlays.Overlay("THE END", last=2, backdrop="card")])
        host = self.host()
        host.showed(79, [])  # one picture of the 80 is left: the scenes' sheet fits, the text's does not
        with patch.object(delivery, "export_of", return_value=self.film):
            result = self.call(host, "look_at_film", project_id=project.id)
        data = json.loads(result.text)
        self.assertLess(list(data).index("text_burned_in"), list(data).index("scenes"))
        self.assertEqual(len(result.images), 1)
        self.assertIn("no room", data["text_frames"])

    def test_a_film_without_text_has_no_extra_sheet(self):
        project = self.project()
        with patch.object(delivery, "export_of", return_value=self.film):
            result = self.call(self.host(), "look_at_film", project_id=project.id)
        self.assertEqual(len(result.images), 1)
        self.assertNotIn("text_burned_in", json.loads(result.text))


class DocumentTests(AgentCase):
    def test_the_script_lists_the_text_that_is_burned_into_the_film(self):
        project = timeline.save(None, None, "Lamp", (timeline.Scene("A lamp", 4),), "Landscape 16:9", True, [])
        project = timeline.set_overlays(project.id, project.revision, [overlays.Overlay("Lumio\nlumio.example", last=2)])
        page = delivery.script_doc(project).decode("utf-8")
        self.assertIn("Text on screen", page)
        self.assertIn("Lumio<br>lumio.example", page)
        self.assertIn("last 2", page)
        plain = timeline.save(None, None, "Plain", (timeline.Scene("A lamp", 4),), "Landscape 16:9", True, [])
        self.assertNotIn("Text on screen", delivery.script_doc(plain).decode("utf-8"))
