# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Burned subtitles in the Timeline export (a project option, off by default) and projects whose cast names a
character that no longer exists. Clips are written directly (no generation, no GPU)."""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from app import characters, config, ffmpeg_utils, history, timeline
from engine.media import video

SIZE = (320, 180)


def _clip(path: Path, seconds: int, gray: int) -> None:
    with video.Writer(path, 30, *SIZE, preset="ultrafast", comment=video.AI_COMMENT) as out:
        for _ in range(30 * seconds):
            out.write(np.full((SIZE[1], SIZE[0], 3), gray, dtype=np.uint8))


def _band(frame) -> int:
    """The brightest value of the subtitle band of a frame (white text is about 255, the clips are dark)."""
    return int(frame.to_ndarray(format="rgb24")[100:150].max())


class SubtitleExportTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.root = Path(self.context.enter_context(tempfile.TemporaryDirectory())).resolve()
        for target in (patch.object(timeline, "DIR", self.root / "projects"),
                       patch("app.items.DIR", self.root / "items"),
                       patch.object(config, "OUTPUTS_DIR", self.root / "outputs"),
                       patch.object(config, "HISTORY_DIR", self.root / "history"),
                       patch.object(history, "_HISTORY_FILE", self.root / "history/history.jsonl"),
                       patch.object(characters, "DIR", self.root / "characters")):
            self.context.enter_context(target)

    def project(self, spoken=True):
        project = timeline.save(None, None, "Subtitled", (timeline.Scene("A cup", 1), timeline.Scene("A tree", 2)),
                                "Landscape 16:9", False, [])
        lines = [[{"text": "Hello there", "character": "", "voice": "hanna", "start": 0.2, "end": 0.8}],
                 [{"text": "And goodbye", "character": "", "voice": "hanna", "start": 0.5, "end": 1.5},
                  {"text": "", "start": 1.6, "end": 1.9}, {"text": "late", "start": "x", "end": 2}]]
        lines = lines if spoken else [[], []]  # clips made with Audio off have no spoken lines
        keys = ["1" * 64, "2" * 64]
        for key, seconds, said, gray in zip(keys, (1, 2), lines, (20, 40)):
            clip, metadata = timeline._asset_paths(project, key)
            clip.parent.mkdir(parents=True, exist_ok=True)
            _clip(clip, seconds, gray)
            timeline._write(metadata, {"qc": {"ok": True, "reasons": [], "face": None}, "sha256": timeline._sha(clip),
                                       "seconds": seconds, "width": SIZE[0], "height": SIZE[1], "characters": [],
                                       "voices": [], "transition_note": "", "dialogue_timings": said})
        timeline._write_plan(project, {"assets": keys})
        return project

    def export(self, project):
        ok, files, _, error = timeline.render_job({"project": project.id, "key": project.key}, lambda *_: None)
        self.assertTrue(ok, error)
        return Path(files[0])

    def test_the_lines_are_on_the_movie_clock_and_cut_at_their_scene(self):
        project = self.project()
        state = timeline.prepared(project)
        records = [timeline.asset(project, key) for key in state["assets"]]
        self.assertEqual(timeline.subtitle_cues(project, records),
                         [(0.2, 0.8, "Hello there"), (1.5, 2.5, "And goodbye")])

    def test_subtitles_are_off_by_default_and_turning_them_on_burns_the_lines_at_their_times(self):
        project = self.project()
        self.assertFalse(project.subtitles)
        with patch.object(ffmpeg_utils, "burn_subtitles", side_effect=AssertionError("burned without the option")):
            plain = self.export(project)
        self.assertEqual(timeline.exported(project), plain)

        on = timeline.set_subtitles(project.id, project.revision, True)
        self.assertEqual((on.subtitles, on.key, on.revision), (True, project.key, project.revision + 1))
        self.assertEqual(timeline.prepared(on)["assets"], timeline.prepared(project)["assets"])  # clips stay
        self.assertIsNone(timeline.exported(on))  # the plain export does not count for a subtitled one
        burned = self.export(on)
        self.assertNotEqual(burned, plain)
        self.assertEqual(timeline.exported(on), burned)
        with av.open(str(burned)) as container:
            self.assertEqual(container.metadata.get("comment"), video.AI_COMMENT)
            frames = list(container.decode(video=0))
        self.assertEqual(len(frames), 90)
        self.assertGreater(_band(frames[15]), 200)  # 0.5 s: "Hello there"
        self.assertLess(_band(frames[30]), 120)  # 1.0 s: nothing is said
        self.assertGreater(_band(frames[60]), 200)  # 2.0 s: "And goodbye" (scene 2 from 1 s, line at 0.5 s)
        self.assertLess(_band(frames[80]), 120)

        off = timeline.set_subtitles(on.id, on.revision, False)
        self.assertIsNone(timeline.exported(off))  # the subtitled export does not count for a plain one
        again = self.export(off)
        self.assertEqual(again.read_bytes(), plain.read_bytes())  # a plain export is made exactly as before
        self.assertNotIn("export_subtitles", timeline.prepared(off))

    def test_an_export_with_subtitles_but_no_spoken_line_says_why_it_has_none(self):
        project = timeline.set_subtitles(*(lambda p: (p.id, p.revision))(self.project(spoken=False)), True)
        with patch.object(ffmpeg_utils, "burn_subtitles", side_effect=AssertionError("nothing to burn")):
            ok, files, notes, error = timeline.render_job({"project": project.id, "key": project.key},
                                                          lambda *_: None)
        self.assertTrue(ok, error)
        self.assertEqual(notes, [timeline.NO_SUBTITLES_SILENT])  # this project's Audio is off
        self.assertIn("need Audio on", notes[0])
        movie = Path(files[0])
        state = timeline.prepared(project)
        records = [timeline.asset(project, key) for key in state["assets"]]
        self.assertEqual(timeline._burn(replace(project, audio=True), records, movie, lambda *_: None),
                         (movie, timeline.NO_SUBTITLES))

    def test_the_option_is_saved_only_when_on_and_keeps_the_projects_key(self):
        project = self.project()
        path = timeline.DIR / project.id / "project.json"
        fields = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("subtitles", fields)  # a project without subtitles keeps the fields it always had
        on = timeline.set_subtitles(project.id, project.revision, True)
        self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["subtitles"])
        self.assertTrue(timeline.load(project.id).subtitles)
        edited = timeline.save(on.id, on.revision, "Subtitled", on.scenes, on.aspect, on.audio, [])
        self.assertTrue(edited.subtitles)  # an ordinary save from the editor keeps the option
        with self.assertRaisesRegex(ValueError, "another tab"):
            timeline.set_subtitles(project.id, project.revision, False)
        with self.assertRaisesRegex(ValueError, "Choose whether"):
            timeline.set_subtitles(project.id, None, "yes")
        fields["subtitles"] = "yes"
        path.write_text(json.dumps(fields), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Could not open"):
            timeline.load(project.id)

    def test_an_old_project_file_opens_with_subtitles_off_and_the_same_key(self):
        project = self.project()
        path = timeline.DIR / project.id / "project.json"
        before = path.read_text(encoding="utf-8")
        opened = timeline.load(project.id)
        self.assertEqual((opened.subtitles, opened.key), (False, project.key))
        timeline.save(opened.id, opened.revision, opened.title, opened.scenes, opened.aspect, opened.audio, [])
        self.assertEqual(path.read_text(encoding="utf-8"), before)  # nothing changed: nothing written


class MissingCharacterTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.root = Path(self.context.enter_context(tempfile.TemporaryDirectory())).resolve()
        for target in (patch.object(timeline, "DIR", self.root / "projects"),
                       patch("app.items.DIR", self.root / "items"),
                       patch.object(characters, "DIR", self.root / "characters")):
            self.context.enter_context(target)
        from tests.character_fixtures import store_character

        store_character(self.root / "characters", {"name": "Ada", "slug": "ada"})

    def test_a_project_whose_cast_names_a_missing_character_opens_with_a_message(self):
        scene = timeline.Scene("Ada and Kuba talk", 2, cast=("ada", "kuba"))
        project = timeline.save(None, None, "Old film", (scene,), "Landscape 16:9", True, ["ada", "ola", "kuba"])
        opened = timeline.load(project.id)
        self.assertEqual(opened.characters, ("ada", "ola", "kuba"))  # opening changes nothing
        notice = timeline.cast_notice(opened)
        self.assertIn("ola, kuba", notice)
        self.assertIn("no longer exist", notice)
        with self.assertRaisesRegex(ValueError, "The character kuba no longer exists"):
            timeline._scene_cast(opened, scene)
        self.assertEqual(timeline.cast_notice(timeline.save(None, None, "New", (timeline.Scene("Ada", 1),),
                                                            "Landscape 16:9", True, ["ada"])), "")

    def test_a_universe_with_a_missing_member_names_it(self):
        from app import universes

        with patch.object(universes, "_file", return_value=self.root / "characters" / "universes.json"):
            (self.root / "characters" / "universes.json").write_text(json.dumps({"universes": [
                {"name": "Family", "slug": "family", "members": ["ada", "ola"]}]}), encoding="utf-8")
            family = universes.get("family")
            self.assertEqual([c["slug"] for c in universes.members(family)], ["ada"])
            self.assertIn("The character ola no longer exists", universes.notice(family))


if __name__ == "__main__":
    unittest.main()
