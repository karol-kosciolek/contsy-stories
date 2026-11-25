# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The package the owner takes away (app/agent_delivery.py and its tools): documents from fixed templates with every text
escaped, names made by the studio, the manifest and the ZIP, and the checks before a run may say "complete"."""
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

from app import agent_delivery as delivery, agent_loop, agent_tools, ffmpeg_utils, history, scene_parts, timeline
from tests.agent_base import AgentCase, picture
from tests.agent_fake import ScriptedProvider, tool_turn

HOSTILE = "<img src=x onerror=alert(1)><script>alert(2)</script>"
FORMULA = "=1+1+cmd|' /C calc'!A0"


class DeliveryCase(AgentCase):
    def project(self, title="Coffee ad", prompt="A barista hands over a cup", speech=True, **scene):
        lines = (scene_parts.Cue("Here you go!"),) if speech else ()
        return timeline.save(None, None, title, (timeline.Scene(prompt, 4, lines=lines, background="a cafe", **scene),
                                                 timeline.Scene("Close on the cup", 2.5, cut=True)),
                             "Landscape 16:9", True, [])

    def result(self, name="frame.png", suffix=None):
        path = self.root / "outputs" / name
        picture(path, size=(160, 90), color=(80, 120, 200))
        history.record({"files": [str(path)], "kind": "obraz", "prompt": "a frame"})
        return path


class DocumentTests(DeliveryCase):
    def test_a_script_shows_every_scene_with_its_dialogue_and_forbids_scripts_and_outside_pictures(self):
        text = delivery.script_doc(self.project(), "A friendly ad.").decode()
        for expected in ("Coffee ad", "Scene 1", "A barista hands over a cup", "Here you go!", "Narrator", "a cafe",
                         "A friendly ad.", "Scene 2", "cut", "2.5 s"):
            self.assertIn(expected, text)
        self.assertIn("Content-Security-Policy", text)
        self.assertIn("default-src 'none'; img-src data:", text)

    def test_what_a_model_wrote_is_escaped_in_every_document(self):
        self.saved_character(HOSTILE[:30].replace("<", "x").replace(">", "x"))
        project = timeline.save(None, None, HOSTILE, (timeline.Scene(HOSTILE, 3, background=HOSTILE,
                                                                     lines=(scene_parts.Cue(HOSTILE),)),),
                                "Landscape 16:9", True, [])
        frames = {0: self.result()}
        for data in (delivery.script_doc(project, HOSTILE), delivery.storyboard_doc(project, frames, HOSTILE),
                     delivery.character_doc(project)):
            text = data.decode()
            self.assertNotIn("<script", text.lower().replace("&lt;script", ""))
            self.assertNotIn("<img src=x", text)
            self.assertIn("&lt;img src=x", text)
        self.assertEqual(delivery.esc(None), "")

    def test_the_shot_list_defuses_spreadsheet_formulas_and_keeps_every_scene(self):
        project = self.project(prompt=FORMULA)
        data = delivery.shot_list(project)
        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        self.assertEqual(rows[0][:3], ["Scene", "Seconds", "Cut"])
        self.assertEqual((len(rows), rows[1][3][0], rows[2][2]), (3, "'", "yes"))
        self.assertEqual(rows[1][4], "Narrator: Here you go!")
        self.assertEqual([delivery.cell(value)[:1] for value in ("=x", "+1", "-1", "@x", "\tx", "ok")],
                         ["'", "'", "'", "'", "'", "o"])

    def test_subtitles_follow_the_scenes_and_the_scripts_own_timing(self):
        scenes = (timeline.Scene("One", 4, lines=(scene_parts.Cue("Hello there.", start=1.0),
                                                  scene_parts.Cue("How are you today?"))),
                  timeline.Scene("Two", 3, lines=(scene_parts.Cue("Fine."),)))
        project = timeline.save(None, None, "Talk", scenes, "Landscape 16:9", True, [])
        text = delivery.subtitles(project).decode()
        blocks = text.strip().split("\r\n\r\n")
        self.assertEqual(len(blocks), 3)
        self.assertTrue(blocks[0].startswith("1\r\n00:00:01,000 --> 00:00:01,"))
        self.assertTrue(blocks[2].startswith("3\r\n00:00:04,300 -->"))  # the second scene starts at 4 s, the first line 0.3 s in
        silent = timeline.save(None, None, "Quiet", (timeline.Scene("A tree", 3),), "Landscape 16:9", True, [])
        self.assertEqual(delivery.subtitles(silent), b"")

    def test_the_storyboard_shows_a_picture_only_for_the_scenes_that_have_one_and_the_sheet_the_cast(self):
        project = timeline.save(self.project().id, 1, "Coffee ad", self.project().scenes, "Landscape 16:9", True, []) \
            if False else self.project()
        text = delivery.storyboard_doc(project, {1: self.result()}).decode()
        self.assertEqual(text.count("data:image/jpeg;base64,"), 1)
        self.assertLess(text.index("Scene 2"), text.index("data:image"))
        self.saved_character("Mia")
        self.saved_item("TV studio")
        cast = timeline.save(None, None, "Cast", (timeline.Scene("Mia in the TV studio", 3, cast=("mia",)),),
                             "Landscape 16:9", True, ["mia"], film_items=("tv_studio",))
        sheet = delivery.character_doc(cast).decode()
        self.assertTrue(all(word in sheet for word in ("Mia", "TV studio", "Short red hair")))
        self.assertEqual(sheet.count("data:image/jpeg;base64,"), 2)


class PackageTests(DeliveryCase):
    def test_names_are_made_here_from_what_the_model_suggested(self):
        host = self.host()
        names = [delivery.put(host.store, name, b"x", "document") for name in
                 ("../../Windows/System32/evil.html", "My Film!.HTML", "My Film!.HTML", "CON.txt", "..", "a/b\\c.csv")]
        self.assertEqual(names, ["evil.html", "my-film.html", "my-film_2.html", "con.txt", "file", "c.csv"])
        for name in names:
            self.assertTrue((delivery.folder(host.store) / name).is_file(), name)
        self.assertEqual({path.name for path in delivery.folder(host.store).iterdir()}, set(names))

    def test_a_document_written_again_replaces_the_one_before_and_a_new_file_unbuilds_the_package(self):
        host = self.host()
        delivery.put(host.store, "script.html", b"one", "document", replace=True)
        delivery.build(host.store)
        self.assertTrue(delivery.package(host.store)["built"])
        delivery.put(host.store, "script.html", b"two", "document", replace=True)
        self.assertEqual([item["name"] for item in delivery.package(host.store)["files"]], ["script.html"])
        self.assertEqual((delivery.folder(host.store) / "script.html").read_bytes(), b"two")
        self.assertEqual(delivery.package(host.store)["built"], "")

    def test_a_result_is_copied_into_the_package_with_its_hash(self):
        host = self.host()
        source = self.result("poster.png")
        name = delivery.add_file(host.store, source, "My Poster", "picture")
        self.assertEqual(name, "my-poster.png")
        item = delivery.package(host.store)["files"][0]
        self.assertEqual(item["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertTrue(source.exists())

    def test_the_package_is_built_with_a_readme_a_report_a_checklist_a_manifest_and_one_zip(self):
        host = self.host(brief=[{"name": "brief.md", "kind": "text"}])
        host.store.update(title="Coffee ad", outcome={
            "kind": "open_items", "summary": HOSTILE, "assumptions": ["A 15 s film"], "deviations": [],
            "owner_should_check": ["The logo"], "failing": [{"code": "G4", "text": "no film"}],
            "checklist": [{"requirement": FORMULA, "status": "partial", "note": HOSTILE}]},
            questions=[HOSTILE], changes=["Setting qc: True -> False."])
        delivery.put(host.store, "coffee-script.html", b"<p>script</p>", "document")
        built = delivery.build(host.store)
        self.assertEqual(built["files"], ["coffee-script.html", "README.txt", "report.html", "brief_checklist.csv",
                                          "manifest.json"])
        folder = delivery.folder(host.store)
        report = (folder / "report.html").read_text(encoding="utf-8")
        self.assertNotIn("<img src=x", report)
        self.assertIn("&lt;img src=x", report)
        self.assertIn("The agent did not see any picture", report)
        self.assertIn("Setting qc", report)
        self.assertIn("480p", (folder / "README.txt").read_text(encoding="utf-8"))
        rows = list(csv.reader(io.StringIO((folder / "brief_checklist.csv").read_text(encoding="utf-8-sig"))))
        self.assertEqual(rows[1][0][0], "'")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            self.assertEqual(item["sha256"], hashlib.sha256((folder / item["name"]).read_bytes()).hexdigest())
        with zipfile.ZipFile(folder / "delivery.zip") as bundle:
            self.assertEqual(sorted(bundle.namelist()), sorted([*built["files"]]))
            self.assertIsNone(bundle.testzip())
            self.assertEqual(bundle.read("coffee-script.html"), b"<p>script</p>")
        self.assertEqual(delivery.downloads(host.store)[0].name, "delivery.zip")
        self.assertFalse(any(path.name.endswith(".part") for path in folder.iterdir()))

    def test_the_downloads_are_listed_zip_first_and_a_folder_without_files_lists_none(self):
        host = self.host()
        self.assertEqual(delivery.downloads(host.store), [])
        delivery.put(host.store, "b.html", b"x", "document")
        delivery.build(host.store)
        (delivery.folder(host.store) / "z.part").write_bytes(b"half")
        names = [path.name for path in delivery.downloads(host.store)]
        self.assertEqual(names[0], "delivery.zip")
        self.assertNotIn("z.part", names)

    def test_a_record_with_nothing_in_it_still_makes_a_package(self):
        host = self.host()
        delivery.build(host.store)
        self.assertTrue((delivery.folder(host.store) / "report.html").is_file())


class GateTests(DeliveryCase):
    def test_an_empty_or_stale_package_stops_a_complete_run(self):
        host = self.host()
        self.assertEqual([code for code, _ in delivery.gate_failures(host)], ["G7"])
        delivery.put(host.store, "script.html", b"x", "document")
        self.assertIn("build_package", delivery.gate_failures(host)[0][1])
        delivery.build(host.store)
        self.assertEqual(delivery.gate_failures(host), [])
        delivery.put(host.store, "more.html", b"y", "document")
        self.assertEqual([code for code, _ in delivery.gate_failures(host)], ["G7"])

    def test_scenes_that_generation_would_refuse_stop_a_complete_run(self):
        host = self.host()
        project = timeline.save(None, None, "Draft", (timeline.Scene("A cup", 3), timeline.Scene(" ", 2)),
                                "Landscape 16:9", True, [])
        host.created("project", project.id)
        delivery.put(host.store, "script.html", b"x", "document")
        delivery.build(host.store)
        failing = delivery.gate_failures(host)
        self.assertEqual([code for code, _ in failing], ["G2"])
        self.assertIn("scene 2", failing[0][1])

    def test_a_run_that_produces_needs_the_finished_film_in_the_package_and_of_the_right_length(self):
        host = self.host(mode="produce")
        project = self.project(speech=False)
        host.created("project", project.id)
        delivery.put(host.store, "script.html", b"x", "document")
        delivery.build(host.store)
        self.assertEqual([code for code, _ in delivery.gate_failures(host)], ["G4"])
        film = self.result("film.png")
        film = film.with_suffix(".mp4")
        film.write_bytes(b"not really a film")
        with patch.object(timeline, "exported", return_value=film), \
                patch.object(ffmpeg_utils, "probe_duration", return_value=6.5):
            self.assertIn("not in the package", delivery.gate_failures(host)[0][1])
            delivery.add_file(host.store, film, "Coffee ad", "film", project=project.id)
            delivery.build(host.store)
            self.assertEqual(delivery.gate_failures(host), [])
        with patch.object(timeline, "exported", return_value=film), \
                patch.object(ffmpeg_utils, "probe_duration", return_value=9.0):
            failing = delivery.gate_failures(host)
        self.assertEqual([code for code, _ in failing], ["G4"])
        self.assertIn("lasts 9.0 s", failing[0][1])

    def test_a_run_that_only_prepares_does_not_need_a_film(self):
        host = self.host(mode="prepare")
        project = self.project(speech=False)
        host.created("project", project.id)
        delivery.put(host.store, "script.html", b"x", "document")
        delivery.build(host.store)
        self.assertEqual(delivery.gate_failures(host), [])


class DeliveryToolTests(DeliveryCase):
    def test_the_documents_are_written_from_a_project_with_names_made_by_the_studio(self):
        host = self.host()
        project = self.project("Coffee Ad!")
        base = f"coffee-ad-{project.id[:6]}"
        for doc, name in (("script", f"{base}-script.html"), ("storyboard", f"{base}-storyboard.html"),
                          ("shot_list", f"{base}-shot-list.csv"), ("character_sheet", f"{base}-characters.html"),
                          ("subtitles", f"{base}-subtitles-estimated.srt")):
            self.assertEqual(self.ok(host, "write_document", doc=doc, project_id=project.id)["document"], name)
        self.ok(host, "write_document", doc="script", project_id=project.id, notes="Second version")
        self.assertEqual(len(delivery.package(host.store)["files"]), 5)  # the script was replaced, not added
        self.assertIn("Second version", (delivery.folder(host.store) / f"{base}-script.html").read_text("utf-8"))

    def test_two_films_with_one_title_and_a_long_title_keep_their_documents_apart(self):
        host = self.host()
        first, second = self.project("Spot 1"), self.project("spot-1")
        long = self.project("A very long title for an advertising film about coffee " * 3)
        names = [self.ok(host, "write_document", doc=doc, project_id=project.id)["document"]
                 for project in (first, second, long) for doc in ("script", "storyboard", "character_sheet")]
        self.assertEqual(len(set(names)), 9)
        self.assertEqual(len(delivery.package(host.store)["files"]), 9)

    def test_the_storyboard_takes_pictures_of_the_gallery_only(self):
        host = self.host()
        project = self.project()
        self.result("frame.png")
        done = self.ok(host, "write_document", doc="storyboard", project_id=project.id,
                       frames=[{"scene": 0, "file": "frame.png"}])
        self.assertIn("data:image/jpeg", (delivery.folder(host.store) / done["document"]).read_text("utf-8"))
        for bad in ("../frame.png", "nothing.png", str(self.root / "outputs" / "frame.png")):
            self.refused(host, "write_document", "NOT_FOUND", doc="storyboard", project_id=project.id,
                         frames=[{"scene": 0, "file": bad}])
        self.refused(host, "write_document", "INVALID", doc="storyboard", project_id=project.id,
                     frames=[{"scene": 9, "file": "frame.png"}])
        self.refused(host, "write_document", "INVALID", doc="subtitles",
                     project_id=timeline.save(None, None, "Quiet", (timeline.Scene("A tree", 3),), "Landscape 16:9",
                                              True, []).id)
        self.refused(host, "write_document", "NOT_FOUND", doc="script", project_id="a" * 32)
        self.refused(host, "write_document", "INVALID", doc="notes", project_id=project.id)

    def test_a_note_that_breaks_the_content_rule_is_refused(self):
        with patch.object(agent_tools.llm, "blocked", return_value=True):
            self.refused(self.host(), "write_document", "CONTENT_RULE", doc="script", project_id=self.project().id,
                         notes="anything")

    def test_a_film_or_a_result_is_added_under_a_name_the_studio_makes(self):
        host = self.host()
        project = self.project()
        self.refused(host, "add_to_package", "INVALID", source="project_film")
        self.refused(host, "add_to_package", "INVALID", source="project_film", project_id=project.id)
        self.refused(host, "add_to_package", "INVALID", source="gallery_file")
        self.refused(host, "add_to_package", "NOT_FOUND", source="gallery_file", file="../x.png")
        poster = self.result("poster.png")
        added = self.ok(host, "add_to_package", source="gallery_file", file="poster.png", label="../../Poster")
        self.assertEqual(added["added"], "poster.png")
        movie = poster.with_suffix(".mp4")
        movie.write_bytes(b"film")
        history.record({"files": [str(movie)], "kind": "wideo", "prompt": "a film"})
        with patch.object(timeline, "exported", return_value=movie):
            film = self.ok(host, "add_to_package", source="project_film", project_id=project.id)
        self.assertEqual(film["added"], "coffee-ad.mp4")
        self.assertEqual([(item["kind"], item.get("project", "")) for item in delivery.package(host.store)["files"]],
                         [("picture", ""), ("film", project.id)])

    def test_finish_run_says_complete_only_for_a_built_package_and_partial_is_always_allowed(self):
        host = self.host()
        done = {"outcome": "complete", "summary": "Done.", "checklist": [{"requirement": "A film", "status": "done"}]}
        self.assertIn("G7", self.refused(host, "finish_run", "REVIEW_REQUIRED", **done))
        self.ok(host, "write_document", doc="shot_list", project_id=self.project().id)
        self.assertIn("build_package", self.refused(host, "finish_run", "REVIEW_REQUIRED", **done))
        listing = self.ok(host, "build_package")
        self.assertIn("delivery.zip", listing["zip"])
        self.ok(host, "finish_run", **done)
        self.assertEqual(host.store.read()["outcome"]["kind"], "complete")
        other = self.host()
        self.ok(other, "finish_run", **{**done, "outcome": "partial"})
        self.assertEqual(other.store.read()["outcome"]["failing"][0]["code"], "G7")


class FinalizeTests(DeliveryCase):
    def test_a_completed_run_has_its_package_with_the_runs_own_report(self):
        host = self.host()
        host.store.append_message("user", [{"type": "text", "text": "Make an ad."}])
        project = self.project()
        script = ("write_document", {"doc": "script", "project_id": project.id})
        finish = ("finish_run", {"outcome": "partial", "summary": "Prepared the script.",
                                 "checklist": [{"requirement": "Film", "status": "skipped", "note": "not produced"}]})
        loop = agent_loop.Loop(host.store, host, ScriptedProvider(tool_turn(script), tool_turn(finish)), clock=self.clock)
        self.assertEqual(loop.run(), "completed")
        folder = delivery.folder(host.store)
        self.assertIn("Prepared the script.", (folder / "report.html").read_text(encoding="utf-8"))
        with zipfile.ZipFile(folder / "delivery.zip") as bundle:
            self.assertTrue(any(name.endswith("-script.html") for name in bundle.namelist()))
        self.assertTrue(host.store.read()["package"]["built"])

    def test_a_package_that_cannot_be_written_does_not_stop_the_run_from_being_finished(self):
        host = self.host()
        host.store.append_message("user", [{"type": "text", "text": "Make an ad."}])
        finish = ("finish_run", {"outcome": "partial", "summary": "x", "checklist": []})
        with patch.object(delivery, "build", side_effect=OSError("disk full")):
            loop = agent_loop.Loop(host.store, host, ScriptedProvider(tool_turn(finish)), clock=self.clock)
            self.assertEqual(loop.run(), "completed")
        self.assertTrue(any(event["type"] == "finalize_failed" for event in host.store.events(limit=0)))
