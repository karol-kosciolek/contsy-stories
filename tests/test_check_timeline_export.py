# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""tools/check_timeline_export.py on a tiny synthetic project and export (CPU only). The export's sound has a
known background level, a speech-band tone in a line's window, an effect burst at a known time, an effect
that never sounds, a scene without background and a dropout at a join: the report finds and flags them,
shows the right frames on its contact sheets and reads the project and outputs folders without writing."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import config, history, timeline, video_gen
from app.scene_parts import Cue, Effect
from engine.media import audio, video

ROOT = Path(__file__).resolve().parents[1]
RATE = audio.RATE
FPS = video_gen.FPS
BACKGROUND_RMS = 0.02   # -34 dBFS
SCENE_RED = (200, 100, 20)  # the picture's red channel per scene; green counts frames (index % 15 * 16)


def _tool():
    spec = importlib.util.spec_from_file_location("check_timeline_export", ROOT / "tools" / "check_timeline_export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _tool()


def _noise(seconds, rms, seed):
    """Noise between 100 and 6000 Hz (AAC keeps all of it) at an exact RMS, mono."""
    count = int(round(seconds * RATE))
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(count))
    frequencies = np.fft.rfftfreq(count, 1 / RATE)
    spectrum[(frequencies < 100) | (frequencies > 6000)] = 0
    signal = np.fft.irfft(spectrum, count)
    return (signal * rms / np.sqrt(np.mean(signal ** 2))).astype(np.float32)


def _soundtrack():
    """8 s: background noise in scenes 1-2 (0-6 s), silence in scene 3; the knock of scene 1 at 1.2-1.5 s,
    a 1 kHz tone where scene 2's line is (3.5-4.7 s), nothing for scene 2's slam (5.0 s), 0.1 s of
    silence across the 1 → 2 join (3.0 s)."""
    mono = np.zeros(8 * RATE, np.float32)
    mono[:6 * RATE] = _noise(6.0, BACKGROUND_RMS, 1)
    mono[int(1.2 * RATE):int(1.5 * RATE)] += _noise(0.3, 0.12, 2)
    t = np.arange(int(1.2 * RATE)) / RATE
    start = int(3.5 * RATE)
    mono[start:start + t.size] += (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    mono[int(2.95 * RATE):int(3.05 * RATE)] = 0.0
    return np.stack([mono, mono])


def _frame(index, red):
    return np.full((32, 48, 3), (red, index % 15 * 16, 60), dtype=np.uint8)


def _film(path, scenes, sound=None):
    """scenes: [(seconds, red)]; frame numbers count on across the scenes."""
    with video.Writer(path, FPS, 48, 32, preset="ultrafast", comment=video.AI_COMMENT,
                      audio=(sound, RATE) if sound is not None else None) as out:
        index = 0
        for seconds, red in scenes:
            for _ in range(seconds * FPS):
                out.write(_frame(index, red))
                index += 1


def _snapshot(*folders):
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
            for folder in folders if folder.exists() for path in sorted(folder.rglob("*"))}


class ExportCheckTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        enter = self.context.enter_context
        self.root = Path(enter(tempfile.TemporaryDirectory())).resolve()
        enter(patch.object(timeline, "DIR", self.root / "projects"))
        enter(patch("app.items.DIR", self.root / "items"))
        enter(patch.object(config, "OUTPUTS_DIR", self.root / "outputs"))
        enter(patch.object(config, "HISTORY_DIR", self.root / "history"))
        enter(patch.object(history, "_HISTORY_FILE", self.root / "history" / "history.jsonl"))
        for memo in (timeline._MEMO, timeline._CLIP_INDEX):
            memo.clear()
            self.addCleanup(memo.clear)

    def make_project(self, *, export=True):
        scenes = (
            timeline.Scene("Ada knocks at the door.", 3, cast=("ada",), background="hallway hum",
                           effects=(Effect("door knock", 1.2, 1.0),)),
            timeline.Scene("Kit opens the door.", 3, cut=True, cast=("kit",), background="hallway hum",
                           lines=(Cue("Hello there.", "kit", 0.5),), effects=(Effect("door slam", 2.0, 0.8),)),
            timeline.Scene("The hallway is empty.", 2, cut=True, cast=("ada",)),
        )
        project = timeline.save(None, None, "Export check", scenes, "Landscape 16:9", True,
                                ["ada", "kit"])
        clips = timeline.DIR / project.id / "clips"
        clips.mkdir(parents=True)
        keys = []
        for index, (scene, red) in enumerate(zip(project.scenes, SCENE_RED)):
            key = hashlib.sha256(f"clip {index}".encode()).hexdigest()
            _film(clips / f"{key}.mp4", [(scene.seconds, red)])
            timings = [{"text": "Hello there.", "character": "kit", "voice": "marek", "start": 0.5, "end": 1.7}] \
                if index == 1 else []
            record = {"sha256": hashlib.sha256((clips / f"{key}.mp4").read_bytes()).hexdigest(),
                      "seconds": scene.seconds, "qc": {"ok": True, "reasons": [], "face": None}, "width": 48, "height": 32,
                      "characters": list(scene.cast), "voices": ["Marek"] if timings else [],
                      "dialogue_timings": timings, "transition": "first" if index == 0 else "cut",
                      "transition_note": "First scene." if index == 0 else "Intentional cut.", "continuity_version": 1}
            (clips / f"{key}.json").write_text(json.dumps(record), encoding="utf-8")
            keys.append(key)
        plan = {"assets": keys, "export_assets": keys, "continuity_version": 1}
        if export:
            config.OUTPUTS_DIR.mkdir(parents=True)
            movie = config.OUTPUTS_DIR / "timeline_export_check_0123abcd.mp4"
            _film(movie, [(scene.seconds, red) for scene, red in zip(project.scenes, SCENE_RED)], _soundtrack())
            history.record({"mode": "video_timeline", "project": project.id, "files": [str(movie)],
                            "qc": {"ok": True, "reasons": [], "face": None}})
            plan.update(export=movie.name, export_revision=project.revision, export_merge=None)
        (timeline.DIR / project.id / "plans").mkdir()
        (timeline.DIR / project.id / "plans" / f"{project.key}.json").write_text(json.dumps(plan), encoding="utf-8")
        return project

    def test_the_report_finds_the_known_sound_and_flags_what_is_wrong(self):
        project = self.make_project()
        report = check.build_report(project.id)
        self.assertEqual(report["export"]["source"], "current export")
        self.assertEqual(Path(report["export"]["path"]), timeline.exported(project))
        facts = report["container"]
        self.assertAlmostEqual(facts["duration"], 8.0, delta=1 / FPS)
        self.assertEqual((facts["fps"], facts["frames"], facts["width"], facts["height"]), (30.0, 240, 48, 32))
        self.assertEqual((facts["audio"]["codec"], facts["audio"]["rate"], facts["audio"]["channels"]), ("aac", RATE, 2))
        self.assertAlmostEqual(facts["audio"]["decoded_seconds"], 8.0, delta=1 / FPS)
        first, second, third = report["scenes"]
        self.assertEqual([(scene["start"], scene["end"]) for scene in report["scenes"]], [(0, 3), (3, 6), (6, 8)])
        self.assertEqual((first["cast"], second["cast"], second["voices"]), (["ada"], ["kit"], ["Marek"]))

        # The known background level outside the dialogue, and the scene without any.
        for scene in (first, second):
            self.assertEqual(scene["background"]["status"], "ok")
            self.assertAlmostEqual(scene["background"]["rms_dbfs"], -34.0, delta=1.5)
        self.assertEqual(third["background"]["status"], "flag")
        self.assertLess(third["background"]["rms_dbfs"], check.QUIET_BACKGROUND_DB)

        # The line on the movie clock, well above the background in the speech band.
        [line] = second["lines"]
        self.assertEqual((line["start"], line["end"], line["character"], line["voice"]), (3.5, 4.7, "kit", "marek"))
        self.assertEqual(line["status"], "ok")
        self.assertGreater(line["rise_db"], 10)

        # The knock is found against the second before it; the slam never sounds.
        [knock], [slam] = first["effects"], second["effects"]
        self.assertEqual((knock["at"], knock["status"], knock["reference_from"]), (1.2, "ok", "the second before"))
        self.assertGreater(knock["rise_db"], 10)
        self.assertEqual((slam["at"], slam["status"]), (5.0, "flag"))
        self.assertLess(slam["rise_db"], check.EFFECT_RISE_DB)

        # The dropout at the first join; the second join only goes from sound to silence.
        self.assertEqual([(join["scenes"], join["status"]) for join in report["joins"]],
                         [([1, 2], "flag"), ([2, 3], "ok")])
        self.assertLess(report["joins"][0]["quietest_block_dbfs"], -50)

        loudness = report["loudness"]
        self.assertIsNotNone(loudness["integrated_lufs"])
        self.assertLess(loudness["sample_peak_dbfs"], 0)
        flags = "\n".join(report["flags"])
        for text in ("Scene 3: background too quiet", "Scene 2, effect 1 \"door slam\" at 2.00 s: no clear energy rise",
                     "Join 1 → 2 at 3.00 s: dropout"):
            self.assertIn(text, flags)
        for text in ("door knock", "line 1", "Scene 1: background", "Join 2 → 3", "Export:"):
            self.assertNotIn(text, flags)
        self.assertEqual({item["criterion"]: item["ok"] for item in report["stage5"]},
                         {"30–60 s long": False, "two characters": True, "several shots": True,
                          "at least two lines": False, "audible background": False})
        json.dumps(report)  # the whole report is plain JSON

    def test_contact_sheets_show_the_first_middle_and_last_frame_of_each_scene(self):
        project = self.make_project()
        report = check.build_report(project.id, frames=self.root / "sheets")
        self.assertEqual(len(report["written"]), 3)
        expected = [(0, 45, 89), (90, 135, 179), (180, 210, 239)]
        thumb = round(32 * check.THUMB_WIDTH / 48)
        for scene, indices, red in zip(report["scenes"], expected, SCENE_RED):
            frames = scene["frames"]
            self.assertEqual([item["frame"] for item in frames["picked"]], list(indices))
            self.assertTrue(frames["source"].startswith("export "))
            with Image.open(frames["file"]) as sheet:
                self.assertEqual(sheet.size, (3 * check.THUMB_WIDTH + 4 * check.MARGIN,
                                              check.HEADER + thumb + check.CAPTION + check.MARGIN))
                for column, index in enumerate(indices):
                    x = check.MARGIN + column * (check.THUMB_WIDTH + check.MARGIN) + check.THUMB_WIDTH // 2
                    r, g, _ = sheet.getpixel((x, check.HEADER + thumb // 2))
                    self.assertLess(abs(r - red), 8, (scene["number"], index))
                    self.assertLess(abs(g - index % 15 * 16), 8, (scene["number"], index))

    def test_the_command_line_writes_only_its_own_files_and_always_exits_0(self):
        project = self.make_project()
        protected = _snapshot(timeline.DIR, config.OUTPUTS_DIR)
        out = self.root / "review"
        with redirect_stdout(io.StringIO()) as printed:
            code = check.main([project.id, "--frames", str(out / "frames"), "--json", str(out / "report.json")])
        self.assertEqual(code, 0)
        text = printed.getvalue()
        for needed in ("Export check", "(the project's current export)", "Line 1", "speech band", "Joins",
                       "Stage 5 test film", "Flags (", "background too quiet"):
            self.assertIn(needed, text)
        saved = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["flags"])
        for number, flag in enumerate(saved["flags"], 1):  # the JSON holds the flags the text lists
            self.assertIn(f"  {number}. {flag}", text)
        self.assertEqual(len(list((out / "frames").glob("*.png"))), 3)
        self.assertEqual(len(saved["written"]), 4)  # three contact sheets and the report itself
        self.assertEqual(_snapshot(timeline.DIR, config.OUTPUTS_DIR), protected)  # nothing written there

        # Another file with --export; a report into the outputs folder is refused, the exit code stays 0.
        copy = self.root / "copy.mp4"
        copy.write_bytes(timeline.exported(project).read_bytes())
        target = config.OUTPUTS_DIR / "report.json"
        with redirect_stdout(io.StringIO()) as printed:
            code = check.main([project.id, "--export", str(copy), "--json", str(target)])
        self.assertEqual(code, 0)
        self.assertFalse(target.exists())
        self.assertIn("(--export; not the project's current export)", printed.getvalue())
        self.assertIn("Output: --json was not written", printed.getvalue())
        self.assertEqual(_snapshot(timeline.DIR, config.OUTPUTS_DIR), protected)

    def test_a_shorter_movie_is_flagged_once_and_the_scene_it_lacks_is_not_checked(self):
        project = self.make_project()
        short = self.root / "short.mp4"
        _film(short, [(3, SCENE_RED[0]), (3, SCENE_RED[1])], _soundtrack()[:, :6 * RATE])
        report = check.build_report(project.id, export=short, frames=self.root / "sheets")
        self.assertEqual((report["export"]["source"], report["export"]["current"]), ("--export", False))
        self.assertFalse(report["container"]["duration_ok"])
        export_flags = [flag for flag in report["flags"] if flag.startswith("Export:")]
        self.assertEqual(len(export_flags), 1, export_flags)  # not again for its frames and its sound
        self.assertIn("lasts 6.0 s; its scenes add up to 8 s", export_flags[0])
        third = report["scenes"][2]
        self.assertEqual(third["background"], {"reason": "the movie's sound ends at 6.00 s", "status": "unchecked"})
        self.assertEqual(report["joins"][1]["status"], "unchecked")
        self.assertEqual({item["criterion"]: item["ok"] for item in report["stage5"]}["audible background"], None)
        self.assertEqual(len(report["written"]), 2)  # no contact sheet for the scene the movie lacks
        self.assertNotIn("frames", third)
        self.assertIn("Frames: scene 3 is past the end of the movie.", report["unchecked"])

    def test_without_an_export_the_clip_facts_are_still_reported(self):
        project = self.make_project(export=False)
        report = check.build_report(project.id, frames=self.root / "sheets")
        self.assertIsNone(report["export"]["path"])
        self.assertIsNone(report["container"])
        self.assertTrue(report["flags"][0].startswith("Export: the project has no current export"))
        second = report["scenes"][1]
        self.assertEqual((second["clip"]["fps"], second["clip"]["duration"], second["cast"]), (30.0, 3.0, ["kit"]))
        self.assertEqual((second["lines"][0]["start"], second["lines"][0]["end"]), (3.5, 4.7))
        self.assertIsNone(second["background"])
        self.assertNotIn("status", second["lines"][0])
        self.assertEqual([item["ok"] for item in report["stage5"]], [None, True, True, None, None])
        # Without an export the contact sheets show the scene clips.
        self.assertEqual(len(report["written"]), 3)
        self.assertTrue(second["frames"]["source"].startswith("clip "))
        self.assertEqual([item["frame"] for item in second["frames"]["picked"]], [0, 45, 89])
        with redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(check.main([project.id]), 0)
        text = printed.getvalue()
        self.assertIn("Export      none: the project has no current export", text)
        self.assertIn("Clip        ", text)
        self.assertIn("kit (marek) 0.50–1.70 s · movie 3.50–4.70 s", text)

    def test_an_unknown_project_is_reported_with_exit_code_0(self):
        with redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(check.main(["0123456789abcdef0123456789abcdef"]), 0)
        self.assertIn("cannot be opened", printed.getvalue())
        self.assertIn("Could not open this timeline project.", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
