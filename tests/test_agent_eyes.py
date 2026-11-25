# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The agent's eyes (app/agent_eyes.py, app/agent_tools_eyes.py): sheets of frames that fall inside their scenes, pictures that
leave the PC only when allowed and never when they come from an uploaded photo, and the correction of a wrong photo."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import agent_delivery as delivery, agent_eyes, agent_host, agent_loop, agent_prompt, characters, history
from app import items, queue_manager, scene_parts, timeline
from engine.media import video
from tests.agent_base import AgentCase, picture
from tests.agent_fake import tool_turn
from tests.test_agent_loop import FINISH, LoopCase

COLORS = [(220, 30, 30), (30, 200, 40), (40, 60, 230), (230, 220, 40)]


def film(path: Path, scene_seconds: list, *, extra: float = 0.0) -> Path:
    """A tiny film whose scene n is all COLORS[n]; `extra` seconds of white are added at the end."""
    with video.Writer(path, 30, 48, 32, preset="ultrafast", comment=video.AI_COMMENT) as out:
        for number, seconds in enumerate(scene_seconds):
            frame = np.full((32, 48, 3), COLORS[number % len(COLORS)], dtype=np.uint8)
            for _ in range(int(seconds * 30)):
                out.write(frame)
        for _ in range(int(extra * 30)):
            out.write(np.full((32, 48, 3), 255, dtype=np.uint8))
    return path


def decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def near(pixel, color, slack=45) -> bool:
    return all(abs(int(a) - int(b)) <= slack for a, b in zip(pixel, color))


class EyesCase(AgentCase):
    def setUp(self):
        super().setUp()
        self.work = self.root / "work"
        self.film = film(self.root / "film.mp4", [2, 2, 2])

    def project(self, speech=False):
        lines = (scene_parts.Cue("Hello there."),) if speech else ()
        scenes = (timeline.Scene("A grey living room, a lamp", 2, lines=lines), timeline.Scene("Warm light", 2),
                  timeline.Scene("A close-up of a hand", 2))
        return timeline.save(None, None, "Lamp film", scenes, "Landscape 16:9", True, [])


class SheetTests(EyesCase):
    def test_every_frame_falls_inside_its_scene_and_the_sheet_is_a_small_jpeg(self):
        project = self.project()
        sheets, covered = agent_eyes.film_sheets(self.film, project.scenes, 1, self.work)
        [data] = sheets
        self.assertEqual(covered, [0, 1, 2])
        self.assertEqual(data[:2], b"\xff\xd8")
        sheet = decode(data)
        self.assertLessEqual(max(sheet.size), agent_eyes.MAX_SIDE)
        width = sheet.width // 3
        for number in range(3):  # the middle of each cell is the colour of its scene
            self.assertTrue(near(sheet.getpixel((number * width + width // 2, sheet.height // 2)), COLORS[number]), number)
        self.assertLess(sum(sheet.getpixel((4, 4))), 60)  # the label strip is dark
        self.assertEqual(list(self.work.iterdir()), [])   # no frame file is left behind

    def test_a_film_with_fades_is_read_at_its_own_length_not_the_projects(self):
        longer = film(self.root / "long.mp4", [2.4, 2.4, 2.4], extra=0.0)
        [data], _ = agent_eyes.film_sheets(longer, self.project().scenes, 1, self.work)
        sheet = decode(data)
        width = sheet.width // 3
        for number in range(3):
            self.assertTrue(near(sheet.getpixel((number * width + width // 2, sheet.height // 2)), COLORS[number]))

    def test_many_frames_make_several_sheets_but_never_more_than_three(self):
        scenes = [SimpleNamespace(seconds=1) for _ in range(9)]
        long = film(self.root / "nine.mp4", [1] * 9)
        sheets, covered = agent_eyes.film_sheets(long, scenes, 3, self.work)  # 27 frames: all nine scenes
        self.assertEqual((len(sheets), covered), (3, list(range(9))))
        self.assertEqual(len(agent_eyes.film_sheets(long, scenes, 1, self.work)[0]), 1)
        crowded, covered = agent_eyes.film_sheets(long, [SimpleNamespace(seconds=0.5)] * 9 * 2, 3, self.work)
        self.assertLessEqual(len(crowded), agent_eyes.MAX_SHEETS)
        self.assertEqual(covered, list(range(9)))  # what did not fit is not claimed

    def test_a_file_that_cannot_be_read_gives_no_picture_instead_of_an_error(self):
        broken = self.root / "broken.mp4"
        broken.write_bytes(b"not a film")
        self.assertEqual(agent_eyes.film_sheets(broken, self.project().scenes, 1, self.work), ([], []))
        self.assertIsNone(agent_eyes.single_frame(broken, 1.0, self.work)[0])
        self.assertIsNone(agent_eyes.clip_sheet(broken, 2.0, 4, self.work, "Scene 1"))
        self.assertIsNone(agent_eyes.photos_sheet([self.root / "nothing.png"], "x"))

    def test_photos_make_one_labelled_sheet(self):
        paths = [picture(self.root / f"p{n}.png", size=(300, 400), color=COLORS[n]) for n in range(3)]
        sheet = decode(agent_eyes.photos_sheet(paths, "Mara"))
        self.assertEqual(sheet.width, 3 * 420)
        self.assertTrue(near(sheet.getpixel((420 + 210, sheet.height // 2)), COLORS[1]))


class FilmToolTests(EyesCase):
    def export(self, project):
        return patch.object(delivery, "export_of", return_value=self.film)

    def test_the_film_is_shown_as_labelled_sheets_with_what_each_scene_should_show(self):
        project = self.project(speech=True)
        host = self.host()
        with self.export(project):
            result = self.call(host, "look_at_film", project_id=project.id)
        self.assertTrue(result.ok, result.text)
        data = json.loads(result.text)
        self.assertEqual([row["scene"] for row in data["scenes"]], [0, 1, 2])
        self.assertEqual(data["scenes"][0]["lines"], ["Hello there."])
        self.assertIn("grey living room", data["scenes"][0]["should_show"])
        self.assertEqual(len(result.images), 1)
        self.assertEqual(result.block("t")["content"][1]["source"]["media_type"], "image/jpeg")

    def test_one_full_size_frame_at_a_chosen_second(self):
        project = self.project()
        with self.export(project):
            result = self.call(self.host(), "look_at_film", project_id=project.id, at_seconds=3.0)
        self.assertEqual(len(result.images), 1)
        self.assertTrue(near(decode(result.images[0]).getpixel((24, 28)), COLORS[1]))
        self.assertEqual(json.loads(result.text)["shown_at_seconds"], 3.0)

    def test_without_the_owners_permission_only_the_facts_come_and_no_frame_is_made(self):
        project = self.project()
        with self.export(project), patch.object(agent_eyes, "film_sheets", side_effect=AssertionError("looked")):
            result = self.call(self.host(send_images=False), "look_at_film", project_id=project.id)
        self.assertTrue(result.ok)
        self.assertEqual(result.images, [])
        self.assertIn("not shown", json.loads(result.text)["pictures"])

    def test_a_film_with_a_character_made_from_an_uploaded_photo_stays_on_this_pc(self):
        self.saved_character("Mia")  # its photos were uploaded
        project = timeline.save(None, None, "Cast", (timeline.Scene("Mia at a desk", 2, cast=("mia",)),), "Landscape 16:9",
                                True, ["mia"])
        with self.export(project):
            result = self.call(self.host(), "look_at_film", project_id=project.id)
        self.assertEqual(result.images, [])
        self.assertIn("uploaded", result.text)

    def test_a_project_without_a_finished_film_is_told_what_to_do(self):
        project = self.project()
        text = self.refused(self.host(), "look_at_film", "INVALID", project_id=project.id)
        self.assertIn("look_at_scene", text)


class SceneToolTests(EyesCase):
    def clip_record(self, seconds=2):
        return {"path": self.film, "seconds": seconds, "qc": {"ok": True, "reasons": []},
                "dialogue_timings": [{"text": "Hello there.", "character": "mia", "start": 0.4, "end": 1.2}]}

    def patches(self, keys=("a" * 64, None, None)):
        return (patch.object(timeline, "prepared", return_value={"assets": list(keys)}),
                patch.object(timeline, "asset", return_value=self.clip_record()),
                patch.object(timeline, "takes", return_value=[{"key": "b" * 64, "label": "Take 2", "current": False,
                                                               "anchor_ok": True}, {"key": "a" * 64, "label": "Take 1",
                                                                                    "current": True, "anchor_ok": True}]))

    def test_a_clip_is_shown_as_frames_with_the_studios_own_checks_and_what_is_said(self):
        project = self.project(speech=True)
        prepared, asset, takes = self.patches()
        with prepared, asset, takes:
            result = self.call(self.host(), "look_at_scene", project_id=project.id, scene=0, frames=4)
        self.assertTrue(result.ok, result.text)
        data = json.loads(result.text)
        # an older clip record says ok, but nothing checked the clip itself: it is "unchecked", never a pass
        self.assertEqual((data["scene"], data["studio_checks"]["status"], data["studio_checks"]["passed"],
                          data["studio_checks"]["record"], data["spoken"][0]["text"]),
                         (0, "unchecked", False, {"ok": True, "reasons": []}, "Hello there."))
        self.assertEqual(len(result.images), 1)
        sheet = decode(result.images[0])
        self.assertTrue(near(sheet.getpixel((sheet.width // 4, sheet.height // 2)), COLORS[0]))

    def test_a_take_may_be_chosen_only_among_the_scenes_own_takes_and_a_scene_needs_a_clip(self):
        project = self.project()
        prepared, asset, takes = self.patches()
        with prepared, asset, takes:
            host = self.host()
            self.assertTrue(self.call(host, "look_at_scene", project_id=project.id, scene=0, take="b" * 64).ok)
            self.refused(host, "look_at_scene", "NOT_FOUND", project_id=project.id, scene=0, take="c" * 64)
            self.refused(host, "look_at_scene", "INVALID", project_id=project.id, scene=1)  # no clip yet
            self.refused(host, "look_at_scene", "INVALID", project_id=project.id, scene=9)

    def test_it_answers_with_facts_only_when_pictures_may_not_be_sent(self):
        project = self.project()
        prepared, asset, takes = self.patches()
        with prepared, asset, takes, patch.object(agent_eyes, "clip_sheet", side_effect=AssertionError("looked")):
            result = self.call(self.host(send_images=False), "look_at_scene", project_id=project.id, scene=0)
        self.assertEqual(result.images, [])
        self.assertIn("not shown", result.text)


class PhotoToolTests(EyesCase):
    def test_a_saved_character_made_from_uploaded_photos_stays_on_this_pc_and_a_generated_one_is_shown(self):
        saved = self.saved_character()
        host = self.host()
        blocked = self.call(host, "look_at_photos", kind="character", slug=saved["slug"])
        self.assertEqual(blocked.images, [])
        self.assertIn("uploaded", blocked.text)
        record = json.loads(characters._file(saved["slug"]).read_text(encoding="utf-8"))
        for photo in record["photos"]:
            photo["origin"] = "generated"
        characters._file(saved["slug"]).write_text(json.dumps(record), encoding="utf-8")
        shown = self.call(host, "look_at_photos", kind="character", slug=saved["slug"])
        self.assertEqual(len(shown.images), 1)
        self.assertEqual(json.loads(shown.text)["name"], "Mia")
        self.assertEqual(self.call(self.host(send_images=False), "look_at_photos", kind="character",
                                   slug=saved["slug"]).images, [])

    def test_a_place_is_looked_at_the_same_way(self):
        place = self.saved_item()
        record = json.loads(items._file(place["slug"]).read_text(encoding="utf-8"))
        for photo in record["photos"]:
            photo["origin"] = "generated"
        items._file(place["slug"]).write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(len(self.call(self.host(), "look_at_photos", kind="item", slug=place["slug"]).images), 1)
        self.refused(self.host(), "look_at_photos", "NOT_FOUND", kind="item", slug="nothing")

    def test_a_draft_of_this_run_is_looked_at_before_it_is_saved_and_one_of_another_run_is_not(self):
        host = self.host()
        made = self.ok(host, "create_character", name="Leo", gender="man", look="Tall, dark curly hair, grey eyes",
                       voice={"preset": "hanna"}, photos=2)
        self.clear_queue()
        paths = [picture(self.root / f"d{n}.png", size=(300, 400), color=COLORS[n]) for n in range(2)]
        held = [(path, {"origin": "generated"}) for path in paths]
        with patch.object(characters, "draft_photos", return_value=held):
            shown = self.call(host, "look_at_photos", kind="character", draft_id=made["draft_id"])
            self.assertEqual(len(shown.images), 1)
            self.assertEqual(json.loads(shown.text)["origins"], ["generated", "generated"])
            self.refused(self.host(), "look_at_photos", "NOT_FOUND", kind="character", draft_id=made["draft_id"])
        with patch.object(characters, "draft_photos", return_value=[(paths[0], {"origin": "uploaded"})]):
            self.assertEqual(self.call(host, "look_at_photos", kind="character", draft_id=made["draft_id"]).images, [])
        for arguments in ({"kind": "character"}, {"kind": "character", "slug": "a", "draft_id": "b"}):
            self.refused(host, "look_at_photos", "INVALID", **arguments)

    def test_a_wrong_photo_is_regenerated_as_the_runs_job_or_removed(self):
        host = self.host()
        made = self.ok(host, "create_character", name="Leo", gender="man", look="Tall, dark curly hair, grey eyes",
                       voice={"preset": "hanna"}, photos=2)
        self.clear_queue()

        def regenerate(draft, index):
            return queue_manager.add("character_photos", "Regenerate", {"draft": draft, "count": 1, "index": index})

        with patch.object(characters, "enqueue_regenerate", side_effect=regenerate) as queued:
            done = self.ok(host, "fix_photo", kind="character", draft_id=made["draft_id"], photo=2, action="regenerate")
        queued.assert_called_once_with(made["draft_id"], 1)
        job = queue_manager.snapshot()[0]
        self.assertEqual((job.id, job.owner, job.label.startswith("Agent: ")), (done["job_id"], host.owner_tag, True))
        with patch.object(characters, "remove_photo") as removed, patch.object(characters, "draft_photos", return_value=[1]):
            self.clear_queue()
            self.assertEqual(self.ok(host, "fix_photo", kind="character", draft_id=made["draft_id"], photo=1,
                                     action="remove"), {"removed": 1, "photos_left": 1})
        removed.assert_called_once_with(made["draft_id"], 0)
        self.refused(self.host(), "fix_photo", "NOT_FOUND", kind="character", draft_id=made["draft_id"], photo=1,
                     action="remove")
        self.refused(self.host(permissions={"write": False}), "fix_photo", "PERMISSION", kind="character",
                     draft_id=made["draft_id"], photo=1, action="remove")

    def test_only_a_generated_photo_can_be_regenerated_and_the_reason_is_told(self):
        host = self.host()
        made = self.ok(host, "create_character", name="Leo", gender="man", look="Tall, dark curly hair, grey eyes",
                       voice={"preset": "hanna"}, photos=2)
        self.clear_queue()
        self.refused(host, "fix_photo", "INVALID", kind="character", draft_id=made["draft_id"], photo=1, action="regenerate")


class PackageAndReportTests(EyesCase):
    def test_a_film_added_to_the_package_brings_a_contact_sheet_and_a_failing_sheet_does_not_stop_it(self):
        project = self.project()
        host = self.host()
        with patch.object(delivery, "export_of", return_value=self.film):
            added = self.ok(host, "add_to_package", source="project_film", project_id=project.id, label="Lamp ad")
        names = [item["name"] for item in delivery.package(host.store)["files"]]
        self.assertEqual((added["added"], names), ("lamp-ad.mp4", ["lamp-ad.mp4", "lamp-ad-contact-sheet.jpg"]))
        sheet = (delivery.folder(host.store) / "lamp-ad-contact-sheet.jpg").read_bytes()
        self.assertEqual(sheet[:2], b"\xff\xd8")
        other = self.host()
        with patch.object(delivery, "export_of", return_value=self.film), \
                patch.object(agent_eyes, "film_sheets", side_effect=OSError("disk")):
            self.ok(other, "add_to_package", source="project_film", project_id=project.id)
        self.assertEqual([item["kind"] for item in delivery.package(other.store)["files"]], ["film"])
        self.assertTrue(any(event["type"] == "contact_sheet_failed" for event in other.store.events(limit=0)))

    def test_the_verdict_on_every_scene_goes_into_the_report(self):
        host = self.host()
        review = [{"scene": 1, "verdict": "ok", "note": "Grey room, lamp visible"},
                  {"scene": 5, "verdict": "fixed", "note": "Second lamp removed with a new take"},
                  {"scene": 6, "verdict": "issue", "note": "<b>Text</b> slightly blurred"}]
        done = {"outcome": "partial", "summary": "Done.", "checklist": [], "review": review}
        self.ok(host, "finish_run", **done)
        delivery.build(host.store)
        report = (delivery.folder(host.store) / "report.html").read_text(encoding="utf-8")
        self.assertIn("What the agent saw, scene by scene", report)
        self.assertIn("Second lamp removed", report)
        self.assertNotIn("<b>Text</b>", report)
        self.assertIn("contact-sheet.jpg", (delivery.folder(host.store) / "README.txt").read_text(encoding="utf-8"))
        self.refused(host, "finish_run", "INVALID", **{**done, "review": [{"scene": 1, "verdict": "great"}]})


class PromptTests(AgentCase):
    def test_the_agent_is_told_to_look_judge_fix_and_look_again_and_pictures_are_on_by_default(self):
        text = agent_prompt.system_prompt({"mode": "produce"})
        for expected in ("look_at_photos", "look_at_scene", "look_at_film", "fix_photo", "look, judge, fix, look again",
                         "You can SEE what you make"):
            self.assertIn(expected, text)
        blind = agent_prompt.system_prompt({"mode": "produce", "send_images": False})
        self.assertIn("NOT sent to you", blind)
        self.assertNotIn("You can SEE", blind)


class LoopWithPicturesTests(LoopCase):
    def test_pictures_the_agent_looked_at_are_kept_as_blobs_and_sent_back_with_the_conversation(self):
        project = timeline.save(None, None, "Lamp film", (timeline.Scene("A grey living room", 2),
                                                          timeline.Scene("Warm light", 2)), "Landscape 16:9", True, [])
        movie = film(self.root / "film.mp4", [2, 2])
        script = [tool_turn(("look_at_film", {"project_id": project.id})), tool_turn(FINISH)]
        with patch.object(delivery, "export_of", return_value=movie):
            self.assertEqual(self.run_loop(*script), "completed")
        records = self.host_.store.records()
        stored = json.dumps(records[2]["content"])
        self.assertIn("$blob", stored)
        self.assertNotIn("/9j/", stored)  # the picture's own bytes are in the blob, not in the line
        sent = self.provider.snapshots[1][2]["content"][0]["content"]
        self.assertEqual([part["type"] for part in sent], ["text", "image"])
        self.assertTrue(sent[1]["source"]["data"].startswith("/9j/"))


class ReviewFixTests(EyesCase):
    """The defects the review of the eyes found."""

    def write_origins(self, kind, slug, origins):
        path = (characters if kind == "character" else items)._file(slug)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["photos"] = [({"origin": origin} if origin is not None else {}) for origin in origins]
        path.write_text(json.dumps(record), encoding="utf-8")

    def test_only_photos_that_are_all_proven_generated_may_leave_the_pc(self):
        saved = self.saved_character()
        cases = [(["generated", "generated"], False), (["generated", "kept"], True), (["kept", "kept"], True),
                 (["generated", "uploaded"], True), (["generated", None], True), (["generated", "whatever"], True), ([], True)]
        for origins, stays in cases:
            with self.subTest(origins=origins):
                self.write_origins("character", saved["slug"], origins)
                self.assertEqual(agent_eyes.photos_were_uploaded("character", saved["slug"]), stays)
        characters._file(saved["slug"]).write_text("not json", encoding="utf-8")
        self.assertTrue(agent_eyes.photos_were_uploaded("character", saved["slug"]))
        self.assertTrue(agent_eyes.photos_were_uploaded("character", "nobody"))
        self.assertTrue(agent_eyes.photos_were_uploaded("item", "nothing"))

    def test_a_draft_opened_for_editing_keeps_its_pictures_on_the_pc(self):
        saved = self.saved_character()
        self.write_origins("character", saved["slug"], ["generated", "generated"])
        host = self.host()
        edit = self.ok(host, "edit_character", slug=saved["slug"], look="Short copper hair, freckles, green eyes")
        shown = self.call(host, "look_at_photos", kind="character", draft_id=edit["draft_id"])  # the draft's photos are "kept"
        self.assertEqual(shown.images, [])
        self.assertIn("stay on this PC", shown.text)

    def test_the_film_guard_follows_who_is_in_the_scenes_not_only_the_projects_list(self):
        self.saved_character("Mia")                       # uploaded photos
        place = self.saved_item("TV studio")              # uploaded photos
        by_text = timeline.save(None, None, "Named", (timeline.Scene("Mia walks into the room", 2),), "Landscape 16:9", True, [])
        self.assertTrue(agent_eyes.film_shows_uploaded(by_text))
        in_place = timeline.save(None, None, "Place", (timeline.Scene("The TV studio, empty, wide shot", 2),),
                                 "Landscape 16:9", True, [])
        self.assertTrue(agent_eyes.film_shows_uploaded(in_place))
        plain = self.project()
        self.assertFalse(agent_eyes.film_shows_uploaded(plain))
        self.assertTrue(agent_eyes.film_shows_uploaded(plain, [{"characters": ["mia"]}]))  # a clip that shows her
        with patch.object(timeline, "_scene_cast", side_effect=ValueError("broken")):
            self.assertTrue(agent_eyes.film_shows_uploaded(plain))  # what cannot be worked out is not shown
        self.write_origins("item", place["slug"], ["generated", "generated"])
        self.assertFalse(agent_eyes.film_shows_uploaded(in_place))

    def test_a_gallery_film_is_checked_through_its_project_and_unknown_names_block_a_picture(self):
        self.saved_character("Mia")
        project = timeline.save(None, None, "Named", (timeline.Scene("Mia walks into the room", 2),), "Landscape 16:9", True, [])
        movie = self.root / "outputs" / "timeline_named.mp4"
        film(movie, [2])
        history.record({"files": [str(movie)], "mode": "video_timeline", "project": project.id})
        shown = self.call(self.host(), "inspect_media", file=movie.name)
        self.assertEqual(shown.images, [])
        self.assertIn("stays on this PC", shown.text)
        other = self.root / "outputs" / "old.png"
        picture(other, size=(64, 48))
        history.record({"files": [str(other)], "kind": "obraz", "prompt": "x", "characters": ["Somebody who was deleted"]})
        self.assertEqual(self.call(self.host(), "inspect_media", file="old.png").images, [])

    def test_a_long_film_is_shown_in_parts_and_never_claims_scenes_it_did_not_show(self):
        scenes = tuple(timeline.Scene(f"Scene text {n}", 1) for n in range(30))
        project = timeline.save(None, None, "Long", scenes, "Landscape 16:9", True, [])
        long = film(self.root / "long30.mp4", [1] * 30)
        host = self.host()
        with patch.object(delivery, "export_of", return_value=long):
            first = self.call(host, "look_at_film", project_id=project.id)
            data = json.loads(first.text)
            self.assertEqual((data["pictures_cover_scenes"], len(first.images)), ("0 to 26 (of 30)", 3))
            self.assertIn("from_scene=27", data["more"])
            rest = self.call(host, "look_at_film", project_id=project.id, from_scene=27)
            self.assertEqual(json.loads(rest.text)["pictures_cover_scenes"], "27 to 29 (of 30)")
            self.assertNotIn("more", json.loads(rest.text))
        self.assertEqual(host.store.read()["saw"], list(range(30)))
        sheets, covered = agent_eyes.film_sheets(long, project.scenes, 1, self.work, max_sheets=10)
        self.assertEqual((len(sheets), covered), (4, list(range(30))))  # the package's contact sheet shows every scene

    def test_a_time_past_the_end_shows_the_last_frame_and_says_so(self):
        project = self.project()
        with patch.object(delivery, "export_of", return_value=self.film):
            result = self.call(self.host(), "look_at_film", project_id=project.id, at_seconds=99)
        self.assertLessEqual(json.loads(result.text)["shown_at_seconds"], 6.0)
        self.assertEqual(len(result.images), 1)

    def test_the_pictures_one_conversation_holds_are_limited_and_the_agent_is_told(self):
        project = self.project()
        host = self.host()
        host.store.update(pictures=agent_host.MAX_PICTURES)
        with patch.object(delivery, "export_of", return_value=self.film), \
                patch.object(agent_eyes, "film_sheets", side_effect=AssertionError("looked")):
            result = self.call(host, "look_at_film", project_id=project.id)
        self.assertEqual(result.images, [])
        self.assertIn(str(agent_host.MAX_PICTURES), json.loads(result.text)["pictures"])
        fresh = self.host()
        with patch.object(delivery, "export_of", return_value=self.film):
            self.call(fresh, "look_at_film", project_id=project.id)
        self.assertEqual(fresh.pictures(), 1)

    def test_the_report_says_which_scenes_the_agent_really_looked_at(self):
        project = self.project()
        host = self.host()
        prepared = patch.object(timeline, "prepared", return_value={"assets": ["a" * 64, None, None]})
        asset = patch.object(timeline, "asset", return_value={"path": self.film, "seconds": 2, "qc": {}})
        with prepared, asset:
            self.assertTrue(self.call(host, "look_at_scene", project_id=project.id, scene=0).images)
        review = [{"scene": 1, "verdict": "ok", "note": "Looked at"}, {"scene": 2, "verdict": "ok", "note": "Not looked at"}]
        self.ok(host, "finish_run", outcome="partial", summary="x", checklist=[], review=review)
        self.assertEqual(host.store.read()["outcome"]["saw_scenes"], [1])
        delivery.build(host.store)
        report = (delivery.folder(host.store) / "report.html").read_text(encoding="utf-8")
        self.assertIn("Pictures looked at", report)
        self.assertRegex(report, r"<td>1</td><td>ok</td><td>yes</td>")
        self.assertRegex(report, r"<td>2</td><td>ok</td><td>no</td>")

    def test_the_tries_at_correcting_a_scene_or_a_photo_are_counted_and_limited(self):
        project = self.project()
        host = self.host(mode="produce")

        def edit(saved, index, scene, *, change="", scope="scene"):
            return saved, queue_manager.add(timeline.EDIT, "Edit", {"project": saved.id, "seconds": 4})

        with patch.object(timeline, "edit", side_effect=edit):
            for _ in range(3):
                self.ok(host, "edit_scene", project_id=project.id, expected_revision=1, scene=0, prompt="A grey room")
            self.refused(host, "edit_scene", "BUDGET", project_id=project.id, expected_revision=1, scene=0,
                         prompt="A grey room")
            self.ok(host, "edit_scene", project_id=project.id, expected_revision=1, scene=1, prompt="Warm light")
        again = self.host(mode="produce")
        made = patch.object(timeline, "prepared", return_value={"assets": ["a" * 64, None, None]})
        enqueue = patch.object(timeline, "enqueue", side_effect=lambda project, kind, **kw: queue_manager.add(
            kind, "Make", {"project": project.id, "seconds": 2}))
        with made, enqueue:
            self.ok(again, "timeline_job", project_id=project.id, action="prepare_scene", scene=1)  # the first making is free
            self.ok(again, "timeline_job", project_id=project.id, action="prepare_scene", scene=1)
            for _ in range(2):
                self.ok(again, "timeline_job", project_id=project.id, action="prepare_scene", scene=0)
                self.clear_queue()
            self.refused(again, "timeline_job", "INVALID", project_id=project.id, action="prepare_scene")  # no scene given
        photo = self.host()
        made_draft = self.ok(photo, "create_character", name="Leo", gender="man", look="Tall, dark curly hair, grey eyes",
                             voice={"preset": "hanna"}, photos=2)
        self.clear_queue()
        with patch.object(characters, "enqueue_regenerate", side_effect=lambda draft, index: queue_manager.add(
                "character_photos", "Again", {"draft": draft, "count": 1, "index": index})):
            for _ in range(2):
                self.ok(photo, "fix_photo", kind="character", draft_id=made_draft["draft_id"], photo=1, action="regenerate")
                self.clear_queue()
            self.refused(photo, "fix_photo", "BUDGET", kind="character", draft_id=made_draft["draft_id"], photo=1,
                         action="regenerate")
            self.ok(photo, "fix_photo", kind="character", draft_id=made_draft["draft_id"], photo=2, action="regenerate")

    def test_a_photo_cannot_be_fixed_while_a_photo_job_of_the_draft_waits(self):
        host = self.host()
        made = self.ok(host, "create_character", name="Leo", gender="man", look="Tall, dark curly hair, grey eyes",
                       voice={"preset": "hanna"}, photos=2)  # its photo job is waiting in the queue
        self.refused(host, "fix_photo", "BUSY", kind="character", draft_id=made["draft_id"], photo=1, action="remove")

    def test_a_scanner_holding_a_frame_file_does_not_lose_the_frame(self):
        real = Path.unlink

        def held(self_, *args, **kwargs):
            if self_.name.startswith("eye_"):
                raise PermissionError("in use")
            return real(self_, *args, **kwargs)

        with patch("pathlib.Path.unlink", held):
            image = agent_eyes.frame_image(self.film, 1.0, self.work)
        self.assertIsNotNone(image)

    def test_contact_sheets_of_two_films_never_overwrite_each_other(self):
        host = self.host()
        first, second = self.project(), self.project()
        with patch.object(delivery, "export_of", return_value=self.film):
            self.ok(host, "add_to_package", source="project_film", project_id=first.id, label="Lamp ad")
            self.ok(host, "add_to_package", source="project_film", project_id=second.id, label="Lamp ad")
        names = [item["name"] for item in delivery.package(host.store)["files"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sum(name.endswith("-contact-sheet.jpg") for name in names), 2)


class BriefPictureTests(AgentCase):
    def test_a_picture_of_the_brief_is_sent_as_a_small_jpeg_and_counted(self):
        from app import agent_brief, agent_config, agent_preflight, agent_runner, agent_store
        from tests.agent_fake import FakeClock, ScriptedProvider

        big = self.root / "big.png"
        Image.effect_noise((3000, 2000), 60).convert("RGB").save(big)
        notes = self.root / "brief.md"
        notes.write_text("# A film\nMake a 10 s film.\n", encoding="utf-8")
        with patch.object(agent_config, "get_key", return_value=("sk-" + "ant-" + "x" * 30, "t")), \
                patch.object(agent_preflight, "blocking_models", return_value=""), \
                patch.object(agent_store, "free_gb", return_value=500.0), \
                patch.object(agent_runner, "make_clock", return_value=FakeClock()), \
                patch.object(agent_runner, "_keep_awake"), \
                patch.object(agent_runner, "make_provider", return_value=ScriptedProvider(AssertionError("stop"))):
            run_id = agent_runner.start([notes, big], "", {"send_images": True})
            agent_runner.wait(run_id)
        store = agent_store.RunStore(run_id)
        blocks = store.wire_messages()[0]["content"]
        image = next(block for block in blocks if block["type"] == "image")
        self.assertEqual(image["source"]["media_type"], "image/jpeg")
        import base64
        self.assertLessEqual(max(Image.open(io.BytesIO(base64.b64decode(image["source"]["data"]))).size), agent_eyes.MAX_SIDE)
        self.assertEqual(store.read()["pictures"], 1)
        self.assertEqual(agent_loop.count_pictures(blocks), 1)

    def test_a_fresh_conversation_holds_only_the_pictures_of_the_brief(self):
        from tests.agent_fake import ScriptedProvider
        host = self.host()
        pic = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}
        host.store.append_message("user", [{"type": "text", "text": "brief"}, pic])
        host.store.update(pictures=60)
        loop = agent_loop.Loop(host.store, host, ScriptedProvider())
        loop.fresh_conversation("context")
        self.assertEqual(host.store.read()["pictures"], 1)
