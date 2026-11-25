# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Timeline editor: track lanes, the scene inspector's editors, production actions and the
refresh that redraws only what changed. Runs on the CPU with temporary folders; the backend's
sound and take functions are mocked."""
from contextlib import ExitStack
from dataclasses import replace
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import unquote

import gradio as gr
from gradio.route_utils import API_PREFIX

from app import characters, config, history, queue_manager, scene_parts, style_library, styles, timeline, ui_theme
from app import ui_timeline
from app import ui_timeline_view as view
from app import universes, voices
from app.scene_parts import Cue, Effect, Issue
from app.ui_timeline import ClipVisual, Draft, Presentation, TrackDialogue, track_html

SKIP = gr.skip()
PROFILES = {
    "ada": {"slug": "ada", "name": "Ada", "gender": "woman"},
    "kit": {"slug": "kit", "name": "Kit", "gender": "man"},
    "rose_marie": {"slug": "rose_marie", "name": "Rose Marie", "gender": "woman"},
}
REFERENCE = "a" * 64


class _Rows(HTMLParser):
    """Tag sequences of every editor row, to check that rows share one skeleton."""

    def __init__(self):
        super().__init__()
        self.rows, self._depth = [], 0

    def handle_starttag(self, tag, attrs):
        names = dict(attrs)
        if tag == "li" and "data-row" in names:
            self.rows.append({"prototype": "data-prototype" in names, "tags": []})
            self._depth = 1
            return
        if self._depth and tag not in ("option", "optgroup"):
            self.rows[-1]["tags"].append(tag)
        if self._depth and tag not in ("input",):
            self._depth += 1

    def handle_endtag(self, tag):
        if self._depth:
            self._depth -= 1


def _rows(markup: str) -> list[dict]:
    parser = _Rows()
    parser.feed(markup)
    return parser.rows


class _Isolated(unittest.TestCase):
    """Temporary project, output and cache folders, a small character and voice library, an
    empty GPU queue. Nothing is queued and no model is loaded."""

    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.root = Path(self.context.enter_context(tempfile.TemporaryDirectory())).resolve()
        enter = self.context.enter_context
        enter(patch.object(timeline, "DIR", self.root / "projects"))
        enter(patch("app.items.DIR", self.root / "items"))
        enter(patch.object(config, "OUTPUTS_DIR", self.root / "outputs"))
        enter(patch.object(config, "HISTORY_DIR", self.root / "history"))
        enter(patch.object(history, "_HISTORY_FILE", self.root / "history/history.jsonl"))
        enter(patch.object(ui_theme, "get_upload_folder", return_value=str(self.root / "cache")))
        enter(patch.object(characters, "all_characters", return_value=list(PROFILES.values())))
        enter(patch.object(characters, "get", side_effect=lambda slug: PROFILES.get(slug)))
        enter(patch.object(voices, "all_voices", return_value=[
            voices.Voice(name="Hanna", slug="hanna", gender="female", path=str(self.root / "hanna.wav"))]))
        enter(patch.object(universes, "all_universes", return_value=[]))
        enter(patch.object(style_library, "all_custom", return_value=[]))
        enter(patch.object(queue_manager, "snapshot", return_value=[]))
        enter(patch.object(queue_manager, "paused", return_value=False))
        enter(patch.object(queue_manager, "add", side_effect=AssertionError("Nothing may be queued in tests.")))
        ui_timeline._forget()
        self.addCleanup(ui_timeline._forget)

    def build(self):
        with patch.object(timeline, "projects", return_value=[]):
            with gr.Blocks() as demo:
                with gr.Tab("Timeline", id="timeline", elem_id="timeline-page") as tab:
                    returned = ui_timeline.build_timeline_tab(tab)
        return demo, returned


class TimelineEditorTests(_Isolated):
    def setUp(self):
        super().setUp()
        enter = self.context.enter_context
        # The backend's sound and take API (app/timeline.py); patched without create=True, so a renamed
        # function fails here instead of being stood in for.
        self.issues = enter(patch.object(timeline, "scene_issues", return_value=[]))
        self.audition = enter(patch.object(timeline, "sound_preview", return_value=None))
        self.mix = enter(patch.object(timeline, "preview_mix", return_value=None))
        self.leveled = enter(patch.object(timeline, "leveled_clip", return_value=None))
        self.takes = enter(patch.object(timeline, "takes", return_value=[]))
        self.failure = enter(patch.object(timeline, "failure", return_value=None))
        enter(patch.object(timeline, "reference_path", return_value=None))

    def project(self, scenes, *, audio=True, characters_=("ada", "kit")):
        return timeline.Project("0" * 32, 3, "Rain <film>", tuple(scenes), audio=audio, characters=characters_)

    # ----------------------------- track -----------------------------

    def test_track_draws_planned_and_measured_lines_overflow_effects_and_alerts(self):
        scene = timeline.Scene("Ada waits at the <door>.", 4, background="", effects=(
            Effect("door <slam>", 1.5, 1.0), Effect("thunder", 3.5, 2.0), Effect("echo", 6.0, 1.0)))
        second = timeline.Scene("Kit answers.", 2, background="rain on the <window>")
        visuals = (
            ClipVisual(lines=(TrackDialogue("Ada", "Hello <you>", 0.3, 1.5, planned=True),
                              TrackDialogue("Ada", "Wait", 3.6, 4.8, planned=True),
                              TrackDialogue("Ada", "Late", 4.2, 5.0, planned=True)),
                       alert="Line 3 starts at 4.2 s, after the scene ends at 4 s."),
            ClipVisual(lines=(TrackDialogue("Kit", "Ready", 0.4, 1.2),)),
        )
        markup = track_html(Draft(scenes=(scene, second)), ("Draft", "Ready"), visuals=visuals)
        self.assertIn("A2</small> Sound", markup)
        self.assertIn("Hello &lt;you&gt;", markup)
        self.assertNotIn("<you>", markup)
        self.assertNotIn("<door>", markup)
        self.assertIn('data-time="0.300" data-planned', markup)
        self.assertIn("runs 0.8 s past the scene end", markup)
        self.assertIn("starts at 4.2 s, after the scene ends at 4 s", markup)
        self.assertEqual(markup.count("data-overflow"), 2 + 2)  # two lines, the cut and the late effect
        self.assertIn('data-line-scene="1" data-time="4.400"', markup)  # measured line seeks on the clock
        self.assertIn(">Auto</span></button>", markup)
        self.assertIn("rain on the &lt;window&gt;", markup)
        self.assertIn('aria-label="Effect 1 in scene 1: door &lt;slam&gt;, at 1.5 s for 1 s"', markup)
        self.assertIn("cut at the scene end", markup)
        self.assertIn('class="timeline-clip-alert"', markup)
        self.assertIn("Problem: Line 3 starts", markup)
        self.assertEqual(markup.count("timeline-clip-alert"), 1)
        self.assertIn("Project audio is off.", track_html(Draft(scenes=(scene,)), audio=False))
        self.assertNotIn("timeline-effect", track_html(Draft(scenes=(scene,)), audio=False))

    def test_track_keeps_legacy_untimed_lines(self):
        markup = track_html(Draft(scenes=(timeline.Scene('Ada says “Hi.”', 5),)),
                            visuals=(ClipVisual(lines=(TrackDialogue("Ada", "Hi."),)),))
        self.assertIn('data-line-scene="0" data-planned', markup)
        self.assertIn("--audio-width:100.0000%", markup)

    # ----------------------------- inspector editors -----------------------------

    def test_editor_markup_escapes_user_text_and_rows_share_one_skeleton(self):
        options = [("", [("Only character", "")]), ("Characters", [('Ada "<b>"', "ada")]),
                   ("Voices", [("@Hanna", "@hanna")])]
        rows = [view.LineRow("ada", 'He said "<script>alert(1)</script>"', "1.5", "1.5–2.6 s", "estimated"),
                view.LineRow("@gone", "", "", note="<i>note</i>")]
        markup = view.lines_editor(rows, options, 'k"<', extras=["", "@gone (no such voice)"])
        self.assertNotIn("<script>", markup)
        self.assertNotIn("<b>", markup)
        self.assertNotIn("<i>", markup)
        self.assertIn("&lt;script&gt;", markup)
        self.assertIn('data-key="k&quot;&lt;"', markup)
        self.assertIn('<option value="@gone" selected>@gone (no such voice)</option>', markup)
        self.assertIn('<optgroup label="Voices"><option value="@hanna">@Hanna</option>', markup)
        parsed = _rows(markup)
        self.assertEqual([row["prototype"] for row in parsed], [False, False, True])
        self.assertEqual(parsed[0]["tags"], parsed[2]["tags"])
        self.assertEqual(parsed[1]["tags"], parsed[2]["tags"])

        effects = view.effects_editor([view.EffectRow("<glass>", "1", "", "1.0–3.0 s"),
                                       view.EffectRow("", "", "")], "key")
        self.assertIn("&lt;glass&gt;", markup + effects)
        self.assertNotIn("<glass>", effects)
        parsed = _rows(effects)
        self.assertEqual(parsed[0]["tags"], parsed[-1]["tags"])
        self.assertIn('placeholder="2"', effects)

        full = view.lines_editor([view.LineRow(text="x")] * scene_parts.MAX_LINES, options, "k")
        self.assertIn("data-add disabled", full)
        references = view.references_html([(REFERENCE, '/file="x'), ("b" * 64, "")], 'k"')
        self.assertIn("[1]", references)
        self.assertIn("Missing file", references)
        self.assertIn('src="/file=&quot;x"', references)
        self.assertIn('aria-label="Remove reference image 2"', references)
        check = view.issues_html([Issue(scene_parts.INFO, "<i>auto</i>"), Issue(scene_parts.ERROR, "Line 1 <b>")])
        self.assertLess(check.index('data-level="error"'), check.index('data-level="info"'))
        self.assertNotIn("<b>", check)
        self.assertIn("<strong>Error</strong>", check)
        players = view.players_html([("Line 1 · <Ada>", "/a.wav"), ("Skipped", "")], "Out <of> date", stale=True)
        self.assertIn("data-stale", players)
        self.assertEqual(players.count("<audio"), 1)
        self.assertNotIn("<Ada>", players)

    def test_edited_reads_editor_rows_into_scene_parts_and_keeps_old_calls_working(self):
        previous = timeline.Scene("Ada at the window", 6, lines=(Cue("Before", "ada"),), background="rain",
                                  effects=(Effect("bell", 1.0),), voice_db=-3.0)
        draft = Draft(None, (previous,), 0, "form-1")
        legacy = ui_timeline._edited(draft, "Ada at the door", 6, False, False, None)
        self.assertEqual(legacy.scenes[0].lines, previous.lines)
        self.assertEqual(legacy.scenes[0].effects, previous.effects)
        self.assertEqual(legacy.scenes[0].voice_db, -3.0)
        self.assertEqual(legacy.scenes[0].prompt, "Ada at the door")

        lines = json.dumps({"key": "form-1", "rows": [
            {"speaker": "ada", "text": " Hi  there "}, {"speaker": "", "text": ""},  # an empty placeholder row
            {"speaker": "@hanna", "text": "Who?", "start": 2.5}]})
        effects = json.dumps({"key": "form-1", "rows": [{"text": "door slam", "at": 1.5, "seconds": 1},
                                                        {"text": "  "}]})
        references = json.dumps({"key": "form-1", "rows": [REFERENCE, REFERENCE]})
        edited = ui_timeline._edited(draft, "Ada at the door", 6, False, False, None, "", lines, effects,
                                     "  wind   and rain ", references, 2.4, -6)
        scene = edited.scenes[0]
        self.assertEqual(scene.lines, (Cue("Hi there", "ada"), Cue("Who?", "@hanna", 2.5)))
        self.assertEqual(scene.effects, (Effect("door slam", 1.5, 1.0),))
        self.assertEqual(scene.background, "wind and rain")
        self.assertEqual(scene.references, (REFERENCE,))
        self.assertEqual((scene.voice_db, scene.background_db), (2.5, -6.0))

        stale = json.dumps({"key": "another-scene", "rows": [{"text": "Wrong scene"}]})
        self.assertEqual(ui_timeline._edited(draft, "x", 6, False, False, None, "", stale).scenes[0].lines,
                         previous.lines)
        self.assertEqual(ui_timeline._edited(draft, "x", 6, False, False, None, "",
                                             '[{"text": "Plain list"}]').scenes[0].lines, (Cue("Plain list"),))
        with self.assertRaisesRegex(ValueError, "Line 2: Choose who says"):
            ui_timeline._edited(draft, "x", 6, False, False, None, "", json.dumps(
                {"key": "form-1", "rows": [{"text": "ok"}, {"text": "bad", "speaker": "Not A Slug"}]}))
        with self.assertRaisesRegex(ValueError, "Effect 1"):
            ui_timeline._edited(draft, "x", 6, False, False, None, "", None, '[{"text": "x", "seconds": 30}]')
        with self.assertRaisesRegex(ValueError, "Dialogue level"):
            ui_timeline._edited(draft, "x", 6, False, False, None, "", None, None, None, None, 40)
        with self.assertRaisesRegex(ValueError, "reference image"):
            ui_timeline._edited(draft, "x", 6, False, False, None, "", None, None, None, '["../etc"]')
        with self.assertRaisesRegex(ValueError, "tenths of a second"):
            ui_timeline._edited(draft, "x", 2.55, False, False, None)
        with self.assertRaisesRegex(ValueError, "tenths of a second"):
            ui_timeline._edited(draft, "x", 0.5, False, False, None)
        # a scene lasts tenths of a second too; a whole number stays an int, so the scene keeps its keys
        self.assertEqual(ui_timeline._edited(draft, "x", 2.5, False, False, None).scenes[0].seconds, 2.5)
        whole = ui_timeline._edited(draft, "x", 6.0, False, False, None).scenes[0].seconds
        self.assertEqual((whole, type(whole)), (6, int))

    def test_typed_character_names_select_the_character(self):
        self.assertEqual(ui_timeline._character_slugs(["Ada", " rose   marie ", "KIT", "kit", "Nobody"]),
                         ["ada", "rose_marie", "kit", "Nobody"])
        self.assertEqual(ui_timeline._typed_names(["ada"]), SKIP)
        self.assertEqual(ui_timeline._typed_names(["Ada", "Rose Marie"]), gr.update(value=["ada", "rose_marie"]))
        draft = Draft(None, (timeline.Scene("Ada and Kit", 5),))
        edited = ui_timeline._edited(draft, "Ada and Kit", 5, False, True, ["Kit", "ADA"])
        self.assertEqual(edited.scenes[0].cast, ("kit", "ada"))
        candidate = ui_timeline._candidate(edited, "Film", "Landscape 16:9", True, "",
                                           ["Rose Marie", "ada"])
        self.assertEqual(candidate.characters, ("rose_marie", "ada"))
        self.assertNotIn("(unavailable)", str(ui_timeline._scene_choices(candidate, edited.scenes[0])))

    def test_editor_keeps_an_empty_new_row_and_shows_windows(self):
        scene = timeline.Scene("Ada waits", 4, lines=(Cue("Hello there", "ada"), Cue("Long line " * 6, "ada")))
        project = self.project([scene])
        draft = Draft(project, project.scenes, 0, "form-2")
        rows = json.dumps({"key": "form-2", "rows": [cue.to_json() for cue in scene.lines] + [{"speaker": "", "text": ""}]})
        display = ui_timeline._presentation(draft, project, ui_timeline._AUTO, Presentation(),
                                            editors=(rows, None, None))
        parsed = _rows(display.lines)
        self.assertEqual([row["prototype"] for row in parsed], [False, False, False, True])
        self.assertIn("0.3–1.3 s", display.lines)
        self.assertIn("past the end", display.lines)
        self.assertIn('data-past-end="true"', display.lines)
        self.assertIn("Ada (only character)", display.lines)
        self.assertIn("@Hanna", display.lines)

        self.audition.return_value = SimpleNamespace(
            key="sound", texts=("Hello there", "Long line"), timings=((0.3, 1.0), (1.5, 3.9)),
            speakers=(SimpleNamespace(name="Ada"), SimpleNamespace(name="Ada")), lines=(), background=None,
            background_prompt="", effects=(), effect_texts=())
        ui_timeline._forget()
        heard = ui_timeline._presentation(draft, project, ui_timeline._AUTO, Presentation(),
                                          editors=(rows, None, None))
        self.assertIn("0.3–1.0 s<small>measured</small>", heard.lines)
        stale = ui_timeline._presentation(Draft(project, project.scenes, 0, "form-3"), project,
                                          ui_timeline._AUTO, Presentation(), editors=(rows, None, None))
        self.assertEqual(len(_rows(stale.lines)), 3)  # rows typed for another scene are not shown

    def test_inspector_shows_sound_check_previews_takes_and_production_state(self):
        wav = self.root / "line.wav"
        wav.write_bytes(b"RIFF....WAVE")
        mix = self.root / "mix.wav"
        mix.write_bytes(b"RIFF....WAVE")
        scene = timeline.Scene("Ada at the window", 5, lines=(Cue("Hi", "ada"),), effects=(Effect("bell", 1),),
                               references=(REFERENCE,))
        project = self.project([scene, timeline.Scene("Kit", 3)])
        self.issues.return_value = [Issue(scene_parts.ERROR, "Line 1: no <voice> sample.", line=0),
                                    Issue(scene_parts.INFO, "Background from the picture.")]
        self.audition.return_value = SimpleNamespace(
            key="k1", texts=("Hi",), timings=((0.3, 1.0),), speakers=(SimpleNamespace(name="Ada"),),
            lines=(wav,), background=wav, background_prompt="rain", effects=(wav,), effect_texts=("bell",))
        self.mix.return_value = mix
        self.takes.return_value = [{"key": "t2", "created": 2, "current": True, "anchor_ok": True, "label": "Take 2"},
                                   {"key": "t1", "created": 1, "current": False, "anchor_ok": False, "label": "Take 1"}]
        self.failure.return_value = {"scene": 1, "error": "Scene 2: Out of <memory>.", "kind": timeline.PRODUCE}
        jobs = [queue_manager.Job(7, timeline.SOUND, "Sound preview · Rain", {"project": project.id,
                                  "key": project.key, "scene_index": 0}, status=queue_manager.RUNNING,
                                  progress=0.4, desc="Lines")]
        draft = Draft(project, project.scenes, 0, "f")
        with patch.object(queue_manager, "snapshot", return_value=jobs), \
                patch.object(timeline, "prepared", return_value={"assets": []}), \
                patch.object(queue_manager, "position", return_value=0):
            display = ui_timeline._presentation(draft, project, ui_timeline._AUTO, Presentation())
        self.assertIn("<strong>Error</strong>", display.check)
        self.assertIn("no &lt;voice&gt; sample", display.check)
        self.assertIn("no &lt;voice&gt; sample", display.lines)  # the row's note
        self.assertIn("Preparing the sound preview… 40%", display.players)
        self.assertEqual(display.players.count("<audio"), 3)
        self.assertIn("Background · rain · starts 0.25 s before the scene", display.players)
        self.assertTrue(display.missing)
        self.assertTrue(display.mix)
        self.assertEqual(display.takes, (("Take 2 · in use", "t2"), ("Take 1 · the previous scene changed", "t1")))
        self.assertEqual(display.take, "t2")
        self.assertEqual(display.retry, "")  # a job is still running
        self.assertEqual(display.queue_label, "Pause queue")
        self.assertIn("Sound preview: 40% · Lines", display.report)
        self.assertIn("Missing file", display.references)
        self.assertIn('class="timeline-clip-alert"', display.track)
        self.assertTrue(display.background_hint.startswith("Auto: "))

        self.audition.return_value = None
        ui_timeline._forget()
        with patch.object(timeline, "prepared", return_value={"assets": []}):
            later = ui_timeline._presentation(draft, project, ui_timeline._AUTO, display)
        self.assertIn("Out of date", later.players)
        self.assertEqual(later.players.count("<audio"), 3)
        self.assertEqual(later.retry, "Retry scene 2")
        self.assertIn("The export failed. Scene 2: Out of &lt;memory&gt;.", later.report)
        self.assertEqual(later.queue_label, "")
        self.assertIn('data-status="Failed"', later.track)
        self.failure.return_value = {"scene": 0, "error": "Scene 1: no voice.", "kind": timeline.SOUND}
        with patch.object(timeline, "prepared", return_value={"assets": []}):
            sound = ui_timeline._presentation(draft, project, ui_timeline._AUTO, later)
        self.assertEqual(sound.retry, "Retry sound of scene 1")
        self.assertNotIn('data-status="Failed"', sound.track)  # the clip did not fail

    def served(self, url: str) -> bytes:
        """The bytes behind a viewer URL (a copy in Gradio's cache)."""
        prefix = API_PREFIX + "/file="
        self.assertTrue(url.startswith(prefix), url)
        return Path(unquote(url[len(prefix):])).read_bytes()

    def viewer_clip(self, project, record):
        """What the viewer plays for scene 1, after the media worker made it."""
        ui_timeline._viewer_clip(project, project, 0, "e" * 64, record)
        view.wait()
        return ui_timeline._viewer_clip(project, project, 0, "e" * 64, record)

    def test_the_viewer_plays_each_clip_as_the_export_mixes_it(self):
        clip, background = self.root / "clip.mp4", self.root / "background.wav"
        leveled = self.root / ("f" * 16 + "-" + "a" * 24 + ".mp4")  # timeline names previews by the clip's checksum
        clip.write_bytes(b"clip")
        leveled.write_bytes(b"leveled")
        background.write_bytes(b"RIFF")
        record = {"path": clip, "sha256": "f" * 64, "audio_sources": (background, None),
                  "transition_note": "First scene."}
        self.leveled.return_value = leveled
        project = self.project([timeline.Scene("Ada waits", 2, voice_db=-3.0, background_db=2.0)])
        with patch.object(timeline, "asset", return_value=record) as asset:
            url, detail = self.viewer_clip(project, record)
        self.assertEqual(self.served(url), b"leveled")
        self.assertIn("Dialogue -3 dB, background +2 dB", detail)
        self.assertIn("First scene.", detail)
        self.leveled.assert_called_once_with(project, 0)
        asset.assert_called_once_with(project, "e" * 64)  # the checksums are checked before the clip is shown
        # at 0 dB / 0 dB too: the scene sounds as the export will (its stems in the export mix)
        flat = self.project([timeline.Scene("Ada waits", 2)])
        self.leveled.reset_mock()
        with patch.object(timeline, "asset", return_value=record):
            url, detail = self.viewer_clip(flat, record)
        self.assertEqual(self.served(url), b"leveled")
        self.assertNotIn("dB", detail)
        self.leveled.assert_called_once_with(flat, 0)
        # without separate stems the clip itself plays; a preview of another clip is never used
        self.leveled.reset_mock()
        ui_timeline._forget()
        with patch.object(timeline, "asset", return_value={**record, "audio_sources": None}):
            url, _ = self.viewer_clip(flat, {**record, "audio_sources": None})
        self.assertEqual(self.served(url), b"clip")
        self.leveled.assert_not_called()
        ui_timeline._forget()
        self.leveled.return_value = self.root / ("0" * 16 + "-other.mp4")
        with patch.object(timeline, "asset", return_value=record):
            url, _ = self.viewer_clip(flat, record)
        self.assertEqual(self.served(url), b"clip")
        # a clip that no longer matches its checksum is not shown
        ui_timeline._forget()
        with patch.object(timeline, "asset", side_effect=lambda project, key, verify=True: None if verify else record):
            url, _ = self.viewer_clip(flat, record)
        self.assertEqual(url, "")

    def test_quoted_picture_text_offers_conversion(self):
        project = self.project([timeline.Scene('Ada says “Hello.” Kit nods.', 5)])
        display = ui_timeline._presentation(Draft(project, project.scenes), project, ui_timeline._AUTO,
                                            Presentation())
        self.assertTrue(display.convert)
        self.assertIn("1 quoted line", display.picture_note)
        both = timeline.Scene('Ada says “Hello.”', 5, lines=(Cue("Hi", "ada"),))
        project = self.project([both])
        display = ui_timeline._presentation(Draft(project, project.scenes), project, ui_timeline._AUTO,
                                            Presentation())
        self.assertIn('data-level="error"', display.picture_note)
        self.assertIn("Move them here", display.lines)
        self.assertFalse(display.convert)  # the conversion only works for a scene without dialogue lines

    def test_a_failed_clip_shows_its_mark_and_errors_lose_their_emoji(self):
        markup = track_html(Draft(scenes=(timeline.Scene("Ada waits", 2), timeline.Scene("Kit runs", 2))),
                            ("Failed", "Ready"),
                            visuals=(ClipVisual(), ClipVisual(alert="⛔ Only adult characters are supported.")))
        self.assertEqual(markup.count('class="timeline-clip-alert"'), 2)
        self.assertIn('title="The last generation of this scene failed."', markup)
        self.assertIn("Problem: Only adult characters are supported.", markup)
        self.assertNotIn("⛔", markup)

    def test_a_clip_that_no_longer_fits_its_scene_is_marked_outdated_not_ready(self):
        """A character re-created with new photos changes the scenes' picture keys: timeline.outdated lists
        them, and their clips must not look Ready in the bin, the track and the inspector."""
        project = self.project([timeline.Scene("Ada waits", 2), timeline.Scene("Kit runs", 2),
                                timeline.Scene("", 2)])
        keys = ["b" * 64, "c" * 64, ""]
        statuses = ui_timeline._scene_statuses(project, project, keys, None, None, frozenset({0, 2}))
        self.assertEqual(statuses, ("Outdated", "Ready", "Empty"))
        self.assertEqual(ui_timeline._scene_statuses(project, project, keys, None), ("Ready", "Ready", "Empty"))
        with patch.object(timeline, "outdated", return_value=[1]):
            self.assertEqual(ui_timeline._outdated_scenes(project, keys), frozenset({1}))
        with patch.object(timeline, "outdated", side_effect=OSError("gone")):
            self.assertEqual(ui_timeline._outdated_scenes(project, keys), frozenset())
        self.assertEqual(ui_timeline._outdated_scenes(replace(project, id=""), keys), frozenset())
        markup = track_html(Draft(scenes=project.scenes[:2]), statuses[:2], visuals=(ClipVisual(), ClipVisual()))
        self.assertIn('data-status="Outdated"', markup)

    def test_the_queue_button_matches_a_queue_paused_before_the_page_was_built(self):
        with patch.object(queue_manager, "paused", return_value=True):
            demo, _ = self.build()
            components = demo.get_config_file()["components"]
            buttons = [component["props"] for component in components
                       if component.get("type") == "button" and component["props"].get("value") == "Resume queue"]
            self.assertEqual(len(buttons), 1)
            self.assertTrue(buttons[0]["visible"])
            initial = ui_timeline._presentation(Draft(), ui_timeline._NEW_PROJECT, ui_timeline._AUTO,
                                                Presentation())
            self.assertEqual(initial.queue_label, "Resume queue")
            first = ui_timeline._refresh(Draft(), "", 5, False, False, [], "", "", "", "", "", 0, 0,
                                         "Untitled movie", "Landscape 16:9", True, "", [], False, "realistic",
                                         ui_timeline._AUTO, initial)
        self.assertTrue(all(value == SKIP for value in first))  # the built button is already right

    def test_a_preview_that_is_not_the_clips_sound_says_so(self):
        clip = self.root / "clip.mp4"
        clip.write_bytes(b"clip")
        scene = timeline.Scene("Ada at the window", 2, lines=(Cue("Hi", "ada"),))
        project = self.project([scene])
        record = {"path": clip, "sha256": "f" * 64, "audio_sources": (clip, None), "transition_note": "",
                  "characters": ["ada"], "sound_key": "older"}
        self.audition.return_value = SimpleNamespace(
            key="k1", texts=("Hi",), timings=((0.3, 1.0),), speakers=(SimpleNamespace(name="Ada"),), lines=(),
            background=None, background_prompt="", effects=(), effect_texts=())
        with patch.object(timeline, "prepared", return_value={"assets": ["e" * 64]}), \
                patch.object(timeline, "asset", return_value=record), \
                patch.object(timeline, "outdated", return_value=[]):
            display = ui_timeline._presentation(Draft(project, project.scenes), project, ui_timeline._AUTO,
                                                Presentation())
            self.assertIn("not the sound of the current clip", display.players)
            ui_timeline._forget()
            with patch.object(timeline, "asset", return_value={**record, "sound_key": "k1"}):
                same = ui_timeline._presentation(Draft(project, project.scenes), project,
                                                 ui_timeline._AUTO, Presentation())
        self.assertIn("matches the scene as it is now", same.players)

    def test_a_ready_sound_preview_or_export_is_not_queued_again(self):
        project = self.project([timeline.Scene("Ada waits", 5)])
        draft = Draft(project, project.scenes, 0, "f")
        movie = self.root / "movie.mp4"
        with patch.object(timeline, "save", return_value=project), \
                patch.object(timeline, "has_active_job", return_value=False), \
                patch.object(timeline, "projects", return_value=[]), \
                patch.object(timeline, "enqueue", side_effect=AssertionError("Queued")):
            self.audition.return_value = SimpleNamespace(key="k1")
            result = ui_timeline._commit(timeline.SOUND, single=True)(*self.values(draft))
            self.assertIn("sound preview of scene 1 is ready", result[2])
            self.assertNotIn("st-err", result[2])
            with patch.object(timeline, "exported", return_value=movie):
                result = ui_timeline._commit(timeline.PRODUCE)(*self.values(draft))
            self.assertIn("already exported", result[2])
            self.assertEqual(result[5], ui_timeline._PROGRAM)

    # ----------------------------- actions -----------------------------

    def values(self, draft, *, take=None):
        scene = draft.scenes[draft.selected]
        form = (scene.prompt, scene.seconds, scene.cut, scene.cast is not None, list(scene.cast or ()), "",
                None, None, scene.background, None, scene.voice_db, scene.background_db)
        project = ("Rain", "Landscape 16:9", True, "", ["ada"], False, "realistic")
        return (draft, *form, *project) + ((take,) if take is not None else ())

    def test_a_scene_whose_dialogue_needs_more_time_asks_before_it_is_made(self):
        """The owner's request of 2026-11-16: render anyway, use the suggested length and render, or cancel."""
        project = self.project([timeline.Scene("Ada waits", 3), timeline.Scene("Kit runs", 2)])
        draft = Draft(project, project.scenes, 0, "f")
        commits = []

        def commit(kind, *, single=False, **_):
            def apply(draft, *values):
                commits.append((kind, single, draft, values))
                return ("draft",) + (SKIP,) * (ui_timeline._CHANGES - 1)
            return apply

        needs = {0: 6, 1: 4}
        with patch.object(timeline, "suggested_length", side_effect=lambda project, index: needs.get(index)), \
                patch.object(timeline, "outdated", return_value=[0, 1]), \
                patch.object(ui_timeline, "_commit", side_effect=commit):
            asked = ui_timeline._guarded(timeline.PREPARE, single=True)(*self.values(draft))
            self.assertEqual(commits, [])  # nothing is made before the answer
            popup, text, use, plan = asked[ui_timeline._CHANGES:]
            self.assertIs(popup["visible"], True)
            self.assertIn("Scene 1 lasts 3 s; its dialogue needs about 6 s.", text)
            self.assertEqual(use["value"], "Use 6 s and render")
            self.assertEqual(plan, {"kind": timeline.PREPARE, "single": True, "fixes": {0: (3, 6)}})

            anyway = ui_timeline._length_choice(False)(plan, *self.values(draft))
            self.assertEqual(commits[-1][3][1], 3)  # rendered with the length it has
            self.assertIs(anyway[ui_timeline._CHANGES]["visible"], False)
            used = ui_timeline._length_choice(True)(plan, *self.values(draft))
            self.assertEqual((commits[-1][3][1], used[-1]), (6, 6))  # the inspector's Duration shows it too

            everything = ui_timeline._guarded(timeline.PRODUCE)(*self.values(draft))[-1]
            self.assertEqual(everything["fixes"], {0: (3, 6), 1: (2, 4)})
            ui_timeline._length_choice(True)(everything, *self.values(draft))
            kind, single, made, values = commits[-1]
            self.assertEqual((kind, single, values[1], made.scenes[1].seconds), (timeline.PRODUCE, False, 6, 4))

            needs.clear()  # the dialogue fits: made at once, the pop-up stays closed
            done = ui_timeline._guarded(timeline.PREPARE, single=True)(*self.values(draft))
            self.assertEqual(done[0], "draft")
            self.assertIs(done[ui_timeline._CHANGES]["visible"], False)

    def test_the_suggested_length_covers_the_dialogue(self):
        project = self.project([timeline.Scene("Ada waits", 2, lines=(Cue("Hello there, how are you today?"),
                                                                       Cue("Fine.")))])
        with patch.object(timeline, "effective", return_value=("Ada waits", project.scenes[0].lines)), \
                patch.object(timeline, "_measured", return_value=[2.0, 1.0]), \
                patch.object(timeline, "_members_cached", return_value=[]):
            self.assertEqual(timeline.suggested_length(project, 0), 4.3)  # 0.3 + 2.0 + 0.5 + 1.0 = 3.8, + 0.5 tail
            with patch.object(timeline, "_measured", return_value=[2.0, 1.7]):
                self.assertEqual(timeline.suggested_length(project, 0), 5)  # 5.0 is a whole number
            with patch.object(timeline, "_measured", return_value=[0.5, 0.3]):
                self.assertIsNone(timeline.suggested_length(project, 0))  # it fits
            self.assertIsNone(timeline.suggested_length(replace(project, audio=False), 0))

    def test_production_bar_actions_queue_the_right_jobs(self):
        project = self.project([timeline.Scene("Ada waits", 5), timeline.Scene("Kit runs", 3)])
        draft = Draft(project, project.scenes, 1, "f")
        saved = self.project(project.scenes)
        job = SimpleNamespace(id=12)
        with patch.object(timeline, "save", return_value=saved) as save, \
                patch.object(timeline, "has_active_job", return_value=False), \
                patch.object(timeline, "enqueue", return_value=job) as enqueue, \
                patch.object(timeline, "projects", return_value=[]), \
                patch.object(timeline, "exported", return_value=None), \
                patch.object(timeline, "use_take") as use_take:
            cases = ((ui_timeline._commit(timeline.PRODUCE), timeline.PRODUCE, None, ui_timeline._PROGRAM),
                     (ui_timeline._commit(timeline.PREPARE), timeline.PREPARE, None, ui_timeline._SCENE),
                     (ui_timeline._commit(timeline.MERGE), timeline.MERGE, None, ui_timeline._PROGRAM),
                     (ui_timeline._commit(timeline.SOUND, single=True), timeline.SOUND, 1, SKIP),
                     (ui_timeline._commit(timeline.PREPARE, single=True), timeline.PREPARE, 1,
                      ui_timeline._SCENE))
            for handler, kind, index, shown in cases:
                with self.subTest(kind=kind, index=index):
                    result = handler(*self.values(draft))
                    self.assertEqual(enqueue.call_args.args, (saved, kind))
                    self.assertEqual(enqueue.call_args.kwargs["scene_index"], index)
                    self.assertEqual(result[0].baseline, saved)
                    self.assertEqual(result[5], shown)
                    self.assertIn("Job #12", result[2])
            self.failure.return_value = {"scene": 0, "error": "Out of memory.", "kind": timeline.PRODUCE}
            ui_timeline._commit(timeline.PREPARE, retry=True)(*self.values(draft))
            self.assertEqual(enqueue.call_args.args[1], timeline.PREPARE)
            self.assertEqual(enqueue.call_args.kwargs["scene_index"], 0)
            self.assertIs(self.failure.call_args.args[0], project)  # read before the edits were saved
            self.failure.return_value = {"scene": 1, "error": "No voice.", "kind": timeline.SOUND}
            ui_timeline._commit(timeline.PREPARE, retry=True)(*self.values(draft))
            self.assertEqual(enqueue.call_args.args[1], timeline.SOUND)
            self.assertEqual(enqueue.call_args.kwargs["scene_index"], 1)
            self.failure.return_value = None
            result = ui_timeline._commit(timeline.PREPARE, retry=True)(*self.values(draft))
            self.assertIn("no failed scene", result[2])
            calls = enqueue.call_count
            result = ui_timeline._commit(take=True)(*self.values(draft, take="t1"))
            use_take.assert_called_once_with(saved, 1, "t1")
            self.assertEqual(enqueue.call_count, calls)
            self.assertIn("now uses the chosen take", result[2])
            self.assertEqual(save.call_args.args[3], draft.scenes)
        with patch.object(timeline, "has_active_job", return_value=True), \
                patch.object(timeline, "enqueue") as enqueue:
            result = ui_timeline._commit(timeline.PRODUCE)(*self.values(draft))
            enqueue.assert_not_called()
            self.assertIn("already has a job", result[2])
            self.assertEqual(result[0], SKIP)

    # ----------------------------- subtitles -----------------------------

    def _new_form(self):
        """The inspector and project values of an untouched new project, as the page sends them."""
        scene = Draft().scenes[0]
        form = (scene.prompt, scene.seconds, scene.cut, scene.cast is not None, list(scene.cast or ()), "",
                None, None, scene.background, None, scene.voice_db, scene.background_db)
        return (*form, "Untitled movie", "Landscape 16:9", True, "", [], False, "realistic")

    def test_subtitles_are_a_project_option_beside_export_movie(self):
        demo, _ = self.build()
        box = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "timeline-subtitles")
        export = next(b for b in demo.blocks.values() if isinstance(b, gr.Button) and b.value == "Export movie")
        self.assertIsInstance(box, gr.Checkbox)
        self.assertEqual((box.label, box.value), ("Subtitles", False))  # off by default
        siblings = export.parent.children
        self.assertEqual(siblings.index(box) + 1, siblings.index(export))  # right where the export starts
        by_name = {fn.api_name: fn for fn in demo.fns.values()}
        for action in ("timeline_export", "timeline_save", "timeline_join", "timeline_prepare"):
            self.assertIn(box, by_name[action].inputs)  # every production action saves the option first
        refresh = next(fn for fn in demo.fns.values() if fn.fn is ui_timeline._refresh)
        self.assertIn((box._id, "input"), refresh.targets)  # a toggle shows "Unsaved changes" at once
        self.assertIn(box, by_name["timeline_open"].outputs)
        audio = next(b for b in demo.blocks.values() if isinstance(b, gr.Checkbox) and b.label == "Audio")
        option = next(fn for fn in demo.fns.values() if fn.fn is ui_timeline._subtitles_option)
        self.assertEqual(((audio._id, "change") in option.targets, option.outputs), (True, [box]))
        self.assertEqual(ui_timeline._subtitles_option(False)["interactive"], False)  # Subtitles need Audio
        self.assertEqual(ui_timeline._subtitles_option(True)["interactive"], True)

    def test_subtitles_are_saved_with_the_project_and_left_out_when_off(self):
        form = self._new_form()[:12]
        start = Draft(None, (timeline.Scene("Ada waits", 5),), 0, "k")
        with patch.object(timeline, "projects", return_value=[]):
            result = ui_timeline._commit()(start, "Ada waits", *form[1:], "Rain", "Landscape 16:9", True, "",
                                           ["ada"], True, "realistic")
        on = result[0].baseline
        stored = timeline._root(on.id) / "project.json"
        self.assertTrue(on.subtitles)
        self.assertIs(json.loads(stored.read_text(encoding="utf-8"))["subtitles"], True)
        draft = Draft(on, on.scenes, 0, "k")
        off = ui_timeline._candidate(draft, "Rain", "Landscape 16:9", True, "", ["ada"], False)
        self.assertNotEqual(off, on)  # an unsaved change of the project...
        self.assertEqual(off.key, on.key)  # ...that no clip depends on
        self.assertIn("Subtitles", ui_timeline._status_html(on, on, False, []))
        self.assertNotIn("Subtitles", ui_timeline._status_html(off, on, True, []))
        silent = replace(on, audio=False)  # a line is shown while it is spoken: without audio there is none
        self.assertIn("Subtitles need Audio", ui_timeline._status_html(silent, silent, False, []))
        self.assertTrue(ui_timeline._candidate(draft, "Rain", "Landscape 16:9", True, "", ["ada"]).subtitles)
        with patch.object(timeline, "projects", return_value=[]):
            result = ui_timeline._commit()(draft, "Ada waits", *form[1:], "Rain", "Landscape 16:9", True, "",
                                           ["ada"], False, "realistic")
        saved = result[0].baseline
        self.assertEqual((saved.subtitles, saved.revision), (False, on.revision + 1))
        self.assertNotIn("subtitles", json.loads(stored.read_text(encoding="utf-8")))  # the keys it always had

    def test_opening_a_project_shows_its_subtitles_option(self):
        with patch.object(timeline, "projects", return_value=[]):
            on = timeline.save(None, None, "Rain", (timeline.Scene("Ada waits", 5),), "Landscape 16:9", True,
                               ["ada"], subtitles=True)
            demo, _ = self.build()
            box = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "timeline-subtitles")
            opener = next(fn for fn in demo.fns.values() if fn.api_name == "timeline_open")
            out = opener.fn(on.id, Draft(), *self._new_form())
        self.assertIs(out[opener.outputs.index(box)], True)
        self.assertEqual(out[0].baseline.id, on.id)

    # ----------------------------- style -----------------------------

    def test_a_film_in_a_style_no_longer_in_the_library_opens_with_that_style_listed(self):
        watercolor = styles.custom_id(styles.ANIME, "soft pastel watercolor")
        with patch.object(timeline, "projects", return_value=[]):
            film = timeline.save(None, None, "Rain", (timeline.Scene("Ada waits", 5),), "Landscape 16:9", True,
                                 ["ada"], style=watercolor)
            demo, _ = self.build()
            menu = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "timeline-style")
            about = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "timeline-style-about")
            opener = next(fn for fn in demo.fns.values() if fn.api_name == "timeline_open")
            out = opener.fn(film.id, Draft(), *self._new_form())
        update = out[opener.outputs.index(menu)]
        self.assertEqual(update["value"], watercolor)
        self.assertEqual(update["choices"], [*[c for c in styles.choices() if c[1] != styles.REALISTIC], ("Custom: soft pastel watercolor", watercolor)])
        hint = next(fn for fn in demo.fns.values() if fn.fn is ui_timeline._style_about)
        self.assertEqual((hint.inputs, hint.outputs), ([menu], [about]))  # it follows every change of the menu
        self.assertIn("Not in your styles any more", ui_timeline._style_about(watercolor))
        self.assertIn("Make your own style on the Styles page", ui_timeline._style_about(styles.CARTOON))
        on_load = [fn for fn in demo.fns.values() if fn.outputs == [menu]
                   and any(event == "load" for _, event in fn.targets)]
        self.assertEqual(len(on_load), 1)  # the owner's styles are read again when the page loads

    def test_the_status_line_names_the_style_by_the_owners_name(self):
        film = replace(self.project([timeline.Scene("Ada waits", 5)]),
                       style=styles.custom_id(styles.OTHER, "ink and wash"))
        with patch.object(style_library, "label", return_value="Ink <wash> by the river at dusk"):
            status = ui_timeline._status_html(film, film, False, [])
        self.assertIn('title="Landscape 16:9 · 480p · Ink &lt;wash&gt; by the river at dusk · Audio on"', status)
        self.assertIn(">16:9 · 480p · Ink &lt;wash&gt; by the river… · Audio on<", status)  # short on the bar

    def test_saving_refuses_a_style_with_childlike_words(self):
        values = list(self.values(Draft(None, (timeline.Scene("Ada waits", 5),), 0, "k")))
        values[-1] = styles.custom_id(styles.OTHER, "chibi, big heads")
        with patch.object(timeline, "projects", return_value=[]):
            result = ui_timeline._commit()(*values)
        self.assertEqual(result[0], SKIP)
        self.assertIn("every character is an adult", result[2])
        self.assertEqual(list(timeline.DIR.glob("*/project.json")), [])

    def test_the_program_says_when_an_export_differs_only_in_subtitles(self):
        plain_project = self.project([timeline.Scene("Ada waits", 5)])
        with_subtitles = replace(plain_project, subtitles=True)
        earlier = timeline.MoviePreview(Path("movie.mp4"), plain_project.key, 3, current=False)
        self.assertIn("burn in the subtitles",
                      ui_timeline._movie_view(with_subtitles, with_subtitles, earlier, False, False).detail)
        self.assertIn("without the subtitles",
                      ui_timeline._movie_view(plain_project, plain_project, earlier, False, True).detail)
        current = timeline.MoviePreview(Path("movie.mp4"), plain_project.key, 3)
        text = ui_timeline._movie_view(plain_project, plain_project, earlier, False, False, "0123456789abcdef")
        self.assertIn("Export again to put the current text over the film.", text.detail)
        self.assertNotIn("current text", ui_timeline._movie_view(plain_project, plain_project, earlier, False, False, "").detail)
        self.assertEqual(ui_timeline._movie_view(plain_project, plain_project, current, False, False).label,
                         "Current export")
        toggled = ui_timeline._movie_view(with_subtitles, plain_project, current, False, False)
        self.assertEqual(toggled.label, "Earlier export")  # the checkbox changed and is not saved yet
        self.assertTrue(toggled.detail.endswith("unsaved changes."))

    def test_queue_button_pauses_and_resumes(self):
        with patch.object(queue_manager, "paused", return_value=False), \
                patch.object(queue_manager, "pause_now", return_value=4) as pause:
            label, message = ui_timeline._toggle_queue()
        pause.assert_called_once_with()
        self.assertEqual(label["value"], "Resume queue")
        self.assertIn("Job #4", message)
        with patch.object(queue_manager, "paused", return_value=True), \
                patch.object(queue_manager, "set_paused") as resume:
            label, _ = ui_timeline._toggle_queue()
        resume.assert_called_once_with(False)
        self.assertEqual(label["value"], "Pause queue")

    def test_generate_missing_clips_is_off_when_every_scene_has_a_clip(self):
        # timeline.enqueue(PREPARE) without a scene refuses when no clip is missing or out of date.
        project = self.project([timeline.Scene("Ada waits", 2), timeline.Scene("Kit runs", 3)])
        inputs = (Draft(project, project.scenes, 0, "f"), "Ada waits", 2, False, False, [], "", None, None, "",
                  None, 0, 0, "Rain <film>", "Landscape 16:9", True, "", ["ada", "kit"], False, "realistic",
                  ui_timeline._AUTO)
        outdated = self.context.enter_context(patch.object(timeline, "outdated", return_value=[]))
        with patch.object(timeline, "prepared", return_value={"assets": ["e" * 64]}):
            some = ui_timeline._refresh(*inputs, Presentation())
        self.assertTrue(some[-1].missing)
        missing = ui_timeline.OUTPUTS.index("missing")
        self.assertEqual(some[missing], SKIP)  # the button starts enabled
        with patch.object(timeline, "prepared", return_value={"assets": ["e" * 64, "f" * 64]}):
            every = ui_timeline._refresh(*inputs, some[-1])
            self.assertFalse(every[-1].missing)
            self.assertEqual(every[missing], gr.update(interactive=False))
            self.assertEqual(ui_timeline._refresh(*inputs, every[-1])[missing], SKIP)
            # every scene has a clip, but scene 2's no longer fits (its previous scene changed)
            outdated.return_value = [1]
            stale = ui_timeline._refresh(*inputs, every[-1])
            self.assertTrue(stale[-1].missing)
            self.assertEqual(stale[missing], gr.update(interactive=True))
            outdated.return_value = []
        # An edit that drops a clip (here a longer scene 1) gives the button its work back.
        longer = ui_timeline._refresh(*inputs[:2], 4, *inputs[3:], every[-1])
        self.assertTrue(longer[-1].missing)
        self.assertEqual(longer[missing], gr.update(interactive=True))

    # ----------------------------- the built tab -----------------------------

    def test_tab_builds_and_refresh_fills_every_output(self):
        demo, (universe_pick, cast_pick) = self.build()
        self.assertIsInstance(universe_pick, gr.Dropdown)
        self.assertTrue(cast_pick.multiselect)
        names = {fn.api_name for fn in demo.fns.values()}
        for name in ("timeline_export", "timeline_prepare", "timeline_join", "timeline_sound", "timeline_retry",
                     "timeline_use_take", "timeline_new_take", "timeline_generate", "timeline_save",
                     "timeline_convert_quotes"):
            self.assertIn(name, names)
        refresh = next(fn for fn in demo.fns.values() if fn.fn is ui_timeline._refresh)
        self.assertEqual(refresh.trigger_mode, "always_last")
        gallery = next(fn for fn in demo.fns.values() if fn.js and "data-page" in fn.js)
        # the page's own tab first: the menu's Gallery button exists only once the Tools accordion was opened
        self.assertIn('#pages > .tab-wrapper [role=tab][data-tab-id="gallery"]', gallery.js)
        self.assertLess(gallery.js.index("data-tab-id"), gallery.js.index('.workspace-nav button[data-page="gallery"]'))
        def triggers(fn):
            return {(getattr(demo.blocks.get(block), "elem_id", None) or getattr(demo.blocks.get(block), "label", None),
                     event) for block, event in fn.targets}

        # the editors' scripts and Gradio's multiselect report their values as change events
        refreshed = triggers(refresh)
        for elem in ("timeline-lines", "timeline-effects", "timeline-line-assignments"):
            self.assertIn((elem, "change"), refreshed)
            self.assertNotIn((elem, "input"), refreshed)
        self.assertIn(("Appearing in this scene", "change"), refreshed)
        typed = [fn for fn in demo.fns.values() if fn.fn is ui_timeline._typed_names]
        self.assertEqual(sorted(event for fn in typed for _, event in fn.targets), ["change", "change"])
        config_text = str(demo.get_config_file())
        for label in ("Picture", "Dialogue", "Sound", "Takes", "Generate missing clips", "Export movie",
                      "Show in Gallery", "Prepare sound preview", "Use this take", "Generate new take",
                      "Convert to dialogue lines", "Add images"):
            self.assertIn(label, config_text)

        scene = timeline.Scene("Ada waits", 4, lines=(Cue("Hi", "ada"),), effects=(Effect("bell", 1.0),))
        project = self.project([scene])
        draft = Draft(project, project.scenes, 0, "form")
        lines = json.dumps({"key": "form", "rows": [{"speaker": "ada", "text": "Hi"}, {"text": ""}]})
        inputs = (draft, "Ada waits", 4, False, False, [], "", lines, None, "", None, 0, 0,
                  "Rain <film>", "Landscape 16:9", True, "", ["ada", "kit"], False, "realistic",
                  ui_timeline._AUTO)
        with patch.object(timeline, "prepared", return_value={"assets": []}):
            first = ui_timeline._refresh(*inputs, Presentation())
            self.assertEqual(len(first), len(refresh.outputs))
            display = first[-1]
            self.assertIsInstance(display, Presentation)
            at = ui_timeline.OUTPUTS.index
            self.assertEqual(first[at("lines")], display.lines)
            self.assertEqual(len(_rows(display.lines)), 3)  # two rows, one of them empty, and the prototype
            # A refresh tick with nothing new leaves the editors (and their focus) alone.
            second = ui_timeline._refresh(*inputs, display)
            self.assertEqual(len(second), len(refresh.outputs))
            self.assertTrue(all(value == SKIP for value in second))
            invalid = ui_timeline._refresh(*inputs[:2], 0, *inputs[3:], display)
            self.assertEqual(len(invalid), len(refresh.outputs))
            self.assertIn("whole duration", invalid[at("heading")])
            broken = json.dumps({"key": "form", "rows": [{"speaker": "Bad Name", "text": "Hi"}]})
            problem = ui_timeline._refresh(*inputs[:7], broken, *inputs[8:], display)
            self.assertIn("Line 1: Choose who says", problem[at("heading")])
            self.assertIn("Choose who says the line", problem[at("lines")])

    def test_initial_refresh_of_a_new_project_matches_the_built_page(self):
        with patch.object(timeline, "projects", return_value=[]):
            initial = ui_timeline._presentation(Draft(), ui_timeline._NEW_PROJECT, ui_timeline._AUTO,
                                                Presentation())
            result = ui_timeline._refresh(Draft(), "", 5, False, False, [], "", "", "", "", "", 0, 0,
                                          "Untitled movie", "Landscape 16:9", True, "", [], False, "realistic",
                                          ui_timeline._AUTO, initial)
        self.assertTrue(all(value == SKIP for value in result))
        self.issues.side_effect = ValueError("Invalid timeline project.")
        ui_timeline._forget()
        scene = timeline.Scene("Ada", 2, lines=(Cue("A long line that cannot fit in two seconds", "ada"),))
        project = replace(ui_timeline._NEW_PROJECT, scenes=(scene,))
        display = ui_timeline._presentation(Draft(None, (scene,)), project, ui_timeline._AUTO, Presentation())
        self.assertIn("should end around", display.check)  # the local timing check stands in


class TimelineEditorBackendTests(_Isolated):
    """The editor's save, reopen, refresh and quote conversion against the real Timeline backend."""

    def form(self, scene, lines=None, effects=None):
        return (scene.prompt, scene.seconds, scene.cut, scene.cast is not None, list(scene.cast or ()), "",
                lines, effects, scene.background, None, scene.voice_db, scene.background_db)

    def test_dialogue_effects_and_levels_survive_save_reopen_and_refresh(self):
        start = Draft(None, (timeline.Scene("Ada waits at the window", 5),), 0, "k")
        lines = json.dumps({"key": "k", "rows": [{"speaker": "ada", "text": "Is it raining?"}, {"text": ""},
                                                 {"speaker": "@hanna", "text": "Yes.", "start": 2.5}]})
        effects = json.dumps({"key": "k", "rows": [{"text": "thunder", "at": 3}]})
        project = ("Rain", "Landscape 16:9", True, "", ["Ada"], False, "realistic")
        with patch.object(timeline, "projects", return_value=[]):
            result = ui_timeline._commit()(start, "Ada waits at the window", 5, False, False, [], "", lines, effects,
                                           " rain on glass ", None, -2, 3.5, *project)
        saved = result[0].baseline
        self.assertIn("Revision 1 saved.", result[2])
        self.assertEqual(saved.characters, ("ada",))  # the typed name selected the character
        scene = timeline.load(saved.id).scenes[0]
        self.assertEqual(scene.lines, (Cue("Is it raining?", "ada"), Cue("Yes.", "@hanna", 2.5)))
        self.assertEqual(scene.effects, (Effect("thunder", 3.0),))
        self.assertEqual((scene.background, scene.voice_db, scene.background_db), ("rain on glass", -2.0, 3.5))

        values = ui_timeline._editor_values(Draft(saved, saved.scenes))
        reopened = values[0]
        self.assertNotEqual(reopened.form, "k")
        self.assertEqual(json.loads(values[7]), {"key": reopened.form, "rows": [cue.to_json() for cue in scene.lines]})
        self.assertEqual((values[9], values[11], values[12]), ("rain on glass", -2.0, 3.5))
        self.assertEqual(json.loads(values[10]), {"key": reopened.form, "rows": []})
        out = ui_timeline._refresh(reopened, *self.form(scene, values[7], values[8]), *project,
                                   ui_timeline._AUTO, Presentation())
        display = out[-1]
        self.assertEqual(len(out), len(ui_timeline.OUTPUTS))  # 28: the places and items picker and its note too
        self.assertIn("Is it raining?", display.lines)
        self.assertIn(f'data-key="{reopened.form}"', display.lines)
        self.assertIn("thunder", display.track)
        self.assertIn("rain on glass", display.track)
        self.assertIn("Revision 1 saved", display.status)
        self.assertNotIn("Unsaved changes", display.status)
        self.assertIn("Scene 1", display.heading)
        self.assertEqual((display.takes, display.retry, display.download), ((), "", ""))
        self.assertIn("No sound preview yet", display.players)

    def test_quoted_lines_convert_to_dialogue_lines(self):
        demo, _ = self.build()
        convert = next(fn for fn in demo.fns.values() if fn.api_name == "timeline_convert_quotes")
        scene = timeline.Scene('Ada opens the door and says “Come in.” Rain falls outside.', 5)
        draft = Draft(None, (scene,), 0, "k")
        result = convert.fn(draft, *self.form(scene), "Rain", "Landscape 16:9", True, "", ["ada"])
        converted = result[0]
        self.assertNotIn("“", result[1])
        self.assertIn("talks", result[1])
        self.assertEqual([cue.text for cue in converted.scenes[0].lines], ["Come in."])
        self.assertNotEqual(converted.form, "k")  # the editors reload the scene
        self.assertEqual(json.loads(result[7])["rows"], [cue.to_json() for cue in converted.scenes[0].lines])


class ProjectDeleteTests(_Isolated):
    """Project → Delete project: the open project's folder goes to the Recycle Bin (here: a temporary folder) and the
    editor starts a new project."""

    def setUp(self):
        super().setUp()
        self.bin = self.root / "recycle-bin"
        self.bin.mkdir()

        def to_bin(path):
            shutil.move(str(path), str(self.bin / Path(path).name))
            return True

        self.to_bin = to_bin
        self.move = self.context.enter_context(patch.object(timeline.trash, "move", side_effect=to_bin))
        self.context.enter_context(patch.object(timeline, "_DELETE_WAIT", 0))

    def saved(self, title):
        with patch.object(timeline, "projects", return_value=[]):
            return timeline.save(None, None, title, (timeline.Scene("Ada waits at the window", 5),), "Landscape 16:9",
                                 True, ["ada"])

    @staticmethod
    def job(project, status):
        return SimpleNamespace(id=7, status=status, kind=timeline.PREPARE, params={"project": project.id})

    @staticmethod
    def form(scene):
        return (scene.prompt, scene.seconds, scene.cut, scene.cast is not None, list(scene.cast or ()), "", None, None,
                scene.background, None, scene.voice_db, scene.background_db)

    def test_a_deleted_project_goes_to_the_recycle_bin_whole_and_leaves_the_list(self):
        keep, film = self.saved("Keep"), self.saved("Rain")
        self.assertEqual(timeline.delete(film.id), "Rain")
        self.assertTrue((self.bin / film.id / "project.json").is_file())  # the whole folder, ready to restore
        self.assertFalse(timeline._root(film.id).exists())
        self.assertEqual([project.id for project in timeline.projects()], [keep.id])
        with self.assertRaisesRegex(timeline.Deleted, "it was deleted"):
            timeline.delete(film.id)  # deleted already
        self.assertEqual(self.move.call_count, 1)

    def test_a_project_with_a_job_waiting_or_running_is_kept(self):
        film = self.saved("Rain")
        for status in (queue_manager.PENDING, queue_manager.RUNNING):
            with self.subTest(status=status), \
                    patch.object(queue_manager, "snapshot", return_value=[self.job(film, status)]):
                with self.assertRaisesRegex(ValueError, "job in the GPU queue.*Tools → GPU Queue"):
                    timeline.delete(film.id)
        self.move.assert_not_called()
        finished = [self.job(film, queue_manager.DONE), self.job(film, queue_manager.ERROR)]
        with patch.object(queue_manager, "snapshot", return_value=finished):
            self.assertEqual(timeline.delete(film.id), "Rain")  # finished jobs only stay in the queue's list

    def test_a_folder_the_recycle_bin_cannot_take_stays_and_wrong_ids_are_refused(self):
        film = self.saved("Rain")
        self.move.side_effect, self.move.return_value = None, False  # a clip stays open the whole time
        with self.assertRaisesRegex(ValueError, "files are still in use"):
            timeline.delete(film.id)
        self.assertEqual(self.move.call_count, timeline._DELETE_TRIES)
        self.assertEqual(timeline.load(film.id).title, "Rain")
        self.move.side_effect = AssertionError("a wrong id reached the Recycle Bin")
        for wrong in ("", None, "0" * 31, "../projects", film.id.upper()):
            with self.subTest(project=wrong), self.assertRaisesRegex(ValueError, "Invalid timeline project"):
                timeline.delete(wrong)

    def test_files_busy_for_a_moment_are_waited_for_and_the_queue_is_checked_on_every_try(self):
        film, other = self.saved("Rain"), self.saved("Snow")
        tries = []

        def busy_twice(path):  # the editor's background readers let go of the clips after two tries
            tries.append(path)
            return len(tries) > 2 and self.to_bin(path)

        self.move.side_effect = busy_twice
        self.assertEqual(timeline.delete(film.id), "Rain")
        self.assertEqual(len(tries), 3)
        self.assertTrue((self.bin / film.id / "project.json").is_file())
        tries.clear()
        queued = [self.job(other, queue_manager.PENDING)]
        with patch.object(queue_manager, "snapshot", side_effect=[[], queued]):  # a job was queued meanwhile
            with self.assertRaisesRegex(ValueError, "job in the GPU queue"):
                timeline.delete(other.id)
        self.assertEqual(len(tries), 1)
        self.assertTrue(timeline._root(other.id).is_dir())

    def test_a_tab_that_still_has_the_deleted_project_open_keeps_working_and_cannot_save_it_again(self):
        film = self.saved("Rain")
        timeline.delete(film.id)
        with self.assertRaisesRegex(timeline.Deleted, "it was deleted"):
            timeline.save(film.id, film.revision, "Rain", film.scenes, film.aspect, film.audio, ["ada"])
        draft = Draft(film, film.scenes, 0, "k")
        project = ("Rain", "Landscape 16:9", True, "", ["ada"], False, "realistic")
        out = ui_timeline._refresh(draft, *self.form(film.scenes[0]), *project, ui_timeline._AUTO, Presentation())
        self.assertEqual(len(out), len(ui_timeline.OUTPUTS))  # the editor still draws it: nothing raised
        result = ui_timeline._commit()(draft, *self.form(film.scenes[0]), *project)
        self.assertIn("it was deleted", result[2])
        self.assertFalse(timeline._root(film.id).exists())  # the save did not make the folder again

    def test_the_project_menu_deletes_the_open_project_once_the_browser_asked(self):
        keep, film = self.saved("Keep"), self.saved("Rain <film>")
        demo, _ = self.build()
        blocks = {getattr(block, "elem_id", None): block for block in demo.blocks.values()}
        button = blocks["timeline-delete-project"]
        parents, parent = [], button.parent
        while parent is not None:
            parents.append(parent)
            parent = parent.parent
        self.assertIn(blocks["timeline-project-menu"], parents)  # in the Project menu, away from Open and New
        self.assertEqual((button.value, button.variant), ("Delete project", "stop"))
        ask = next(fn for fn in demo.fns.values() if (button._id, "click") in fn.targets)
        self.assertIsNone(ask.fn)
        self.assertEqual((ask.inputs, ask.outputs), ([blocks["timeline-title"]], [blocks["timeline-delete-answer"]]))
        self.assertIn("confirm(", ask.js)
        self.assertIn("Recycle Bin", ask.js)
        delete = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "delete_open")
        self.assertEqual(delete.inputs[1], blocks["timeline-delete-answer"])
        self.assertEqual(delete.outputs[-1], blocks["timeline-delete-answer"])
        status = delete.outputs.index(blocks["timeline-action-status"])
        library = delete.outputs.index(blocks["timeline-projects"])
        name = delete.outputs.index(blocks["timeline-title"])
        opened = Draft(film, film.scenes)

        out = delete.fn(opened, "no")  # Cancel in the browser: nothing changes
        self.assertTrue(all(value == SKIP for value in out[:-1]))
        self.assertEqual(out[-1], "")
        self.assertTrue(timeline._root(film.id).is_dir())

        out = delete.fn(opened, "yes")
        self.assertEqual(len(out), len(delete.outputs))
        self.assertFalse(timeline._root(film.id).exists())
        self.assertIsNone(out[0].baseline)  # the editor holds a new project
        self.assertEqual(out[name], "Untitled movie")
        self.assertEqual((out[library]["value"], out[library]["choices"]), (None, [("Keep", keep.id)]))
        self.assertIn("moved to the Recycle Bin", out[status])
        self.assertIn("Rain &lt;film&gt;", out[status])
        self.assertEqual(out[-1], "")  # an earlier yes never counts again

        out = delete.fn(opened, "yes")  # a tab that still had it open: deleted already, its editor lets go too
        self.assertIsNone(out[0].baseline)
        self.assertEqual(out[name], "Untitled movie")
        self.assertEqual(out[library]["choices"], [("Keep", keep.id)])
        self.assertIn("Rain &lt;film&gt;’ was deleted already, perhaps in another tab", out[status])
        self.assertEqual(self.move.call_count, 1)

        busy = [self.job(keep, queue_manager.RUNNING)]
        for draft, reason in ((Draft(), "not saved yet"), (Draft(keep, keep.scenes), "job in the GPU queue")):
            with self.subTest(reason=reason), patch.object(queue_manager, "snapshot", return_value=busy):
                out = delete.fn(draft, "yes")
                self.assertIn(reason, out[status])  # only the reason is shown; the editor stays as it is
                self.assertTrue(all(value == SKIP for index, value in enumerate(out[:-1]) if index != status))
        self.assertEqual([project.id for project in timeline.projects()], [keep.id])


class SceneAssistantTests(_Isolated):
    """The Video page's prompt assistant under a scene's picture text (llm.write_prompt is patched)."""

    def write(self, idea, current, draft=None, references="", style=styles.ANIMATION_3D,
              reply=(True, "Penny waves at the camera from her bicycle, tracking shot.")):
        with patch.object(ui_timeline.llm, "write_prompt", return_value=reply) as writer:
            out = ui_timeline._write_scene_prompt(idea, current, draft or Draft(), references, style,
                                                  progress=lambda *args, **kwargs: None)
        return out, writer

    def test_the_assistant_sits_under_the_picture_text_and_writes_into_it(self):
        demo, _ = self.build()
        blocks = {getattr(block, "elem_id", None): block for block in demo.blocks.values()}
        description = blocks["timeline-description"]
        picture = next(block for block in demo.blocks.values() if isinstance(block, gr.Tab) and block.label == "Picture")
        panel = next(block for block in demo.blocks.values()
                     if isinstance(block, gr.Accordion) and "AI prompt assistant" in str(block.label))
        convert = next(block for block in demo.blocks.values()
                       if isinstance(block, gr.Button) and block.value == "Convert to dialogue lines")
        self.assertEqual(panel.label, "AI prompt assistant — any language")  # one line in the narrow inspector
        self.assertLess(description._id, panel._id)
        self.assertLess(panel._id, convert._id)  # right under the picture text, in the Picture tab
        parent = panel.parent
        while parent is not None and parent is not picture:
            parent = parent.parent
        self.assertIs(parent, picture)
        write = next(block for block in demo.blocks.values()
                     if isinstance(block, gr.Button) and block.value == "Write prompt" and block._id > panel._id)
        handler = next(fn for fn in demo.fns.values() if (write._id, "click") in fn.targets)
        self.assertIs(handler.fn, ui_timeline._write_scene_prompt)
        self.assertEqual(handler.outputs[0], description)
        idea, current, state, references, style, duration, cast_pick = handler.inputs
        self.assertEqual((current, references, style, duration, cast_pick),
                         (description, blocks[view.REFERENCES_ID], blocks["timeline-style"],
                          blocks["timeline-duration"], blocks["timeline-characters"]))
        self.assertIsInstance(state, gr.State)  # the selected scene and its form key
        self.assertIsInstance(idea, gr.Textbox)

    def test_an_idea_becomes_one_shot_in_the_films_style(self):
        (text, status), writer = self.write("Penny jedzie rowerem z górki i macha", "an old description")
        idea, kind, current = writer.call_args.args[:3]
        self.assertEqual((kind, current), ("video", ""))  # an idea replaces the description, as on the Video page
        self.assertTrue(idea.startswith("Penny jedzie rowerem z górki i macha\n"))
        self.assertIn("ONE continuous shot", idea)
        self.assertEqual({key: writer.call_args.kwargs[key] for key in ("n_refs", "style")},
                         {"n_refs": 0, "style": styles.ANIMATION_3D})
        self.assertEqual(text, "Penny waves at the camera from her bicycle, tracking shot.")
        self.assertIn("Generate scene", status)

    def test_without_an_idea_the_description_is_improved(self):
        _, writer = self.write("", "  Penny on a bike  ")
        idea, _, current = writer.call_args.args[:3]
        self.assertEqual((idea, current), (ui_timeline._ONE_SHOT, "Penny on a bike"))
        (_, status), writer = self.write(" ", "")
        writer.assert_not_called()
        self.assertIn("Enter an idea", status)

    def test_the_scenes_reference_images_are_counted_and_an_unknown_style_is_realistic(self):
        scene = timeline.Scene("Penny holds the lamp from [1]", 5, references=("a" * 64, "b" * 64))
        draft = Draft(None, (scene,), 0, "k")
        _, writer = self.write("Penny holds the lamp [1]", "", draft, ui_timeline._envelope(["a" * 64, "b" * 64], "k"),
                               style="no such style")
        self.assertEqual({key: writer.call_args.kwargs[key] for key in ("n_refs", "style")},
                         {"n_refs": 2, "style": styles.DEFAULT})
        _, writer = self.write("Penny holds the lamp [1]", "", draft, "")  # the box not loaded yet: the scene's own
        self.assertEqual(writer.call_args.kwargs["n_refs"], 2)

    def test_the_writer_knows_the_scenes_length_characters_and_the_scene_before(self):
        """The owner's request of 2026-11-16: prompts written for Contsy, not generic ones."""
        first = timeline.Scene('@ada sits at a desk. A screen displays the text "NEWS".', 4)
        second = timeline.Scene("", 3)
        draft = Draft(None, (first, second), 1, "k")
        with patch.object(ui_timeline.llm, "write_prompt", return_value=(True, "Ada smiles.")) as writer:
            ui_timeline._write_scene_prompt("Ada się uśmiecha", "", draft, "", styles.ANIMATION_3D, 3, ["ada"],
                                            progress=lambda *args, **kwargs: None)
        context = writer.call_args.kwargs["context"]
        self.assertIn("@ada (a woman", context)
        self.assertIn("This scene lasts 3 s", context)
        self.assertIn("at most about 5 spoken words", context)  # (3 - 0.3 - 0.5) * 2.6
        self.assertIn('which showed: @ada sits at a desk. A screen displays the text "NEWS".', context)
        cut = Draft(None, (first, replace(second, cut=True)), 1, "k")
        self.assertIn("intentional cut", ui_timeline._scene_context(cut, 1.5, []))
        self.assertIn("better none", ui_timeline._scene_context(cut, 1.5, []))

    def test_a_named_animal_is_described_to_the_writer_as_one(self):
        dog = {"slug": "chain_dog", "name": "Chain Dog", "gender": "man", "kind": characters.CREATURE,
               "look": "an adult corgi in a blue hoodie"}
        with patch.object(characters, "all_characters", return_value=[dog, PROFILES["ada"]]):
            _, writer = self.write("Chain Dog chases a ball", "")
            idea = writer.call_args.args[0]
            self.assertIn("Chain Dog is an animal or creature, not a human. Keep it one, as @chain_dog", idea)
            self.assertNotIn("corgi", idea)  # its looks come from its photos
            _, writer = self.write("Ada waves", "")
            self.assertNotIn("animal or creature", writer.call_args.args[0])  # people: as before

    def test_a_reply_in_shots_becomes_one_description_and_a_refusal_changes_nothing(self):
        reply = (True, "Shot 1: Penny rides down the hill.\nShot 2: Penny waves at the camera.")
        (text, status), _ = self.write("Penny rides and then waves", "", reply=reply)
        self.assertEqual(text, "Penny rides down the hill. Penny waves at the camera.")
        self.assertIn("add a scene (+ Scene)", status)
        (text, status), _ = self.write("idea", "", reply=(False, "The prompt assistant model is missing."))
        self.assertEqual(text, SKIP)
        self.assertIn("The prompt assistant model is missing.", status)
        (text, status), _ = self.write("idea", "", reply=(False, "⛔ Not allowed."))
        self.assertEqual((text, status.count("Not allowed")), (SKIP, 1))


if __name__ == "__main__":
    unittest.main()
