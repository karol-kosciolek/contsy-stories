# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Timeline editor workspace: panels and tab order, the viewer's static markup, the scrub data the
track carries for the viewer, background media work (viewer copies, waveforms, frames) that the 3 s
refresh never waits for, the edge-trim handler, opening a project, escaping, the track's time scale
(zoom, scroll, fit), and the limits Gradio's stylesheet handling puts on the CSS. CPU only, temporary
folders; the page scripts are checked with `node --check`, and the track's zoom maths run in Node.js,
when Node.js is installed."""
from contextlib import ExitStack
from dataclasses import replace
import heapq
from html.parser import HTMLParser
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import unquote

import gradio as gr
from gradio.route_utils import API_PREFIX
import numpy as np

from app import characters, config, history, queue_manager, timeline, ui_theme, ui_timeline, universes, video_gen
from app import ui_timeline_view as view
from app import style_library, voices
from app.scene_parts import Cue, Effect
from app.ui_timeline import ClipVisual, Draft, Presentation, Program, TrackDialogue, track_html
from engine.media import audio

SKIP = gr.skip()
PROFILES = {
    "ada": {"slug": "ada", "name": "Ada", "gender": "woman", "voice": "hanna"},
    "kit": {"slug": "kit", "name": 'Kit <b>"K"</b>', "gender": "man"},
}
HOSTILE = '<script>alert("x")</script> & "quotes"'


class _Elements(HTMLParser):
    """Every element with its (unescaped) attributes, in document order."""

    def __init__(self):
        super().__init__()
        self.items = []

    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))

    handle_startendtag = handle_starttag


def elements(markup: str, cls: str | None = None, tag: str | None = None) -> list[tuple[str, dict]]:
    parser = _Elements()
    parser.feed(markup)
    return [(name, attrs) for name, attrs in parser.items
            if (tag is None or name == tag) and (cls is None or cls in (attrs.get("class") or "").split())]


def served(url: str) -> bytes:
    prefix = API_PREFIX + "/file="
    assert url.startswith(prefix), url
    return Path(unquote(url[len(prefix):])).read_bytes()


class _Isolated(unittest.TestCase):
    """Temporary project, output and cache folders, a small library, an empty GPU queue, the backend's
    sound and take functions mocked. Media workers finish before the folders go away."""

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
        enter(patch.object(characters, "preview_path", return_value=None))
        enter(patch.object(voices, "all_voices", return_value=[
            voices.Voice(name="Hanna", slug="hanna", gender="female", path=str(self.root / "hanna.wav"))]))
        enter(patch.object(universes, "all_universes", return_value=[]))
        enter(patch.object(style_library, "all_custom", return_value=[]))
        enter(patch.object(queue_manager, "snapshot", return_value=[]))
        enter(patch.object(queue_manager, "paused", return_value=False))
        enter(patch.object(queue_manager, "add", side_effect=AssertionError("Nothing may be queued in tests.")))
        enter(patch.object(timeline, "scene_issues", return_value=[]))
        enter(patch.object(timeline, "sound_preview", return_value=None))
        enter(patch.object(timeline, "preview_mix", return_value=None))
        self.leveled = enter(patch.object(timeline, "leveled_clip", return_value=None))
        enter(patch.object(timeline, "takes", return_value=[]))
        enter(patch.object(timeline, "failure", return_value=None))
        enter(patch.object(timeline, "reference_path", return_value=None))
        ui_timeline._forget()
        self.addCleanup(ui_timeline._forget)
        self.addCleanup(view.wait)  # runs first: no worker writes after the folders are gone

    def build(self):
        with patch.object(timeline, "projects", return_value=[]):
            with gr.Blocks() as demo:
                with gr.Tab("Timeline", id="timeline", elem_id="timeline-page") as tab:
                    ui_timeline.build_timeline_tab(tab)
        return demo

    def project(self, scenes, **fields):
        return timeline.Project("0" * 32, 3, "Rain", tuple(scenes), characters=("ada", "kit"), **fields)

    def clip(self, name="clip.mp4", stems=True) -> dict:
        clip = self.root / name
        clip.write_bytes(b"not a real video")
        background = self.root / f"{name}.background.wav"
        background.write_bytes(b"RIFF")
        return {"path": clip, "sha256": "f" * 64, "audio_sources": (background, None) if stems else None,
                "transition_note": "", "characters": ["ada"], "dialogue_timings": [], "seconds": 4}


class WorkspaceLayoutTests(_Isolated):
    def test_panels_follow_the_tab_order_and_the_page_has_no_big_title(self):
        demo = self.build()
        layout_config = demo.get_config_file()
        props = {component["id"]: component.get("props", {}) for component in layout_config["components"]}
        order = []

        def walk(node):
            order.append(props.get(node["id"], {}).get("elem_id"))
            for child in node.get("children", ()):
                walk(child)
        walk(layout_config["layout"])
        sequence = ["timeline-workspace", "timeline-appbar", "timeline-title", "timeline-project-menu",
                    "timeline-bin", "timeline-scene-list", "timeline-viewer", "timeline-inspector",
                    "timeline-inspector-tabs", "timeline-generate", "timeline-timeline", "timeline-zoom",
                    "timeline-track"]
        for name in sequence:
            self.assertIn(name, order)
        self.assertEqual(sorted(sequence, key=order.index), sequence)  # app bar → bin → viewer → inspector → timeline
        self.assertNotIn("page-head", str(layout_config))
        self.assertIn("Generate scene", str(layout_config))
        self.assertIn("Describe the shot: subject, action, camera…", str(layout_config))

    def test_the_viewer_zoom_bar_and_app_bar_tools_never_change_after_the_page_is_built(self):
        demo = self.build()
        blocks = {getattr(block, "elem_id", None): block for block in demo.blocks.values()}
        static = {blocks[name]._id for name in ("timeline-viewer", "timeline-zoom", "timeline-shell")}
        for fn in demo.fns.values():
            written = {getattr(block, "_id", block) for block in fn.outputs}
            self.assertFalse(static & written, fn.api_name)
        self.assertEqual(blocks["timeline-viewer"].js_on_load, view.VIEWER_JS)
        self.assertEqual(blocks["timeline-track"].js_on_load, view.TRACK_JS)
        self.assertEqual(blocks["timeline-zoom"].js_on_load, view.ZOOM_JS)
        self.assertIn("data-viewer", blocks["timeline-viewer"].value)
        # the viewer tells the server only the source the user picked, through a hidden field
        self.assertIn("#timeline-view textarea", view.VIEWER_JS)

    def test_viewer_controls_are_real_buttons_with_names(self):
        markup = view.viewer_html()
        buttons = elements(markup, tag="button")
        self.assertGreaterEqual(len(buttons), 8)
        for _, attrs in buttons:
            self.assertEqual(attrs.get("type"), "button")
        named = {attrs.get("aria-label") for _, attrs in buttons}
        for name in ("Previous scene", "Back one frame", "Play", "Forward one frame", "Next scene", "Full screen"):
            self.assertIn(name, named)
        modes = {attrs["data-mode"]: attrs for _, attrs in buttons if "data-mode" in attrs}
        self.assertEqual(modes["scene"]["aria-pressed"], "true")
        self.assertEqual(modes["program"]["aria-disabled"], "true")
        self.assertEqual(modes["program"]["aria-describedby"], "timeline-program-hint")
        self.assertIn('id="timeline-program-hint"', markup)
        stage = next(attrs for _, attrs in elements(markup) if "data-stage" in attrs)
        self.assertEqual((stage["tabindex"], stage["role"]), ("0", "group"))
        self.assertIn("Space", stage["aria-label"])
        self.assertIn(">00:00:00:00<", markup)
        zoom = view.zoom_html()
        buttons = {attrs["data-zoom"]: attrs for _, attrs in elements(zoom, tag="button")}
        self.assertEqual({name: attrs.get("aria-label") for name, attrs in buttons.items()},
                         {"out": "Zoom out", "in": "Zoom in", "fit": "Zoom to fit"})
        self.assertEqual(buttons["fit"]["aria-keyshortcuts"], "Shift+Z")
        self.assertEqual(elements(zoom, tag="input")[0][1]["aria-label"], "Zoom")
        toggle = elements(view.shell_html(), tag="button")[0][1]
        self.assertEqual((toggle["aria-controls"], toggle["aria-pressed"]), ("timeline-bin", "true"))

    def test_assistive_technology_gets_a_heading_lists_and_the_playhead_time(self):
        headings = elements(view.shell_html(), tag="h1")
        self.assertEqual(len(headings), 1)  # the workspace has no visible title, but the page has a heading
        self.assertIn("tl-sr", headings[0][1]["class"])
        self.assertIn(">Timeline</h1>", view.shell_html())
        markup = view.viewer_html()
        timecode = elements(markup, "tl-timecode")[0][1]
        self.assertNotIn("aria-label", timecode)  # a paragraph cannot be named; its text says what it is
        self.assertIn('<span class="tl-sr">Playhead time </span><span data-now>', markup)
        assets = view.assets_html(2, [view.AssetItem("a" * 64, "/a.png", "[1]")],
                                  [view.AssetItem("b" * 64, "", "Scene 1", usable=True)])
        grids = [attrs for _, attrs in elements(assets, "tl-asset-grid")]
        self.assertEqual([grid.get("role") for grid in grids], ["list", "list"])
        self.assertTrue(all(grid.get("aria-label") for grid in grids))
        self.assertEqual([attrs.get("role") for _, attrs in elements(assets, "tl-asset")], ["listitem", "listitem"])
        self.assertIn("aria-expanded", view.SHELL_JS)  # Gradio's accordion button (the Project menu)

    def test_timecodes_and_labels(self):
        self.assertEqual(view.timecode(0), "00:00:00:00")
        self.assertEqual(view.timecode(12 + 4 / 30), "00:00:12:04")
        self.assertEqual(view.timecode(3725.5), "01:02:05:15")
        self.assertEqual(view.clock(61), "01:01")
        self.assertEqual(view.first_words("one two three", 40), "one two three")
        self.assertEqual(view.first_words("Ada waits at the door of the old house.", 18), "Ada waits at the…")


class TrackDataTests(_Isolated):
    def visuals(self):
        return (
            ClipVisual(images=tuple(f"/f/{index}.jpg" for index in range(12)), media='/file=C:/a "b".mp4',
                       lines=(TrackDialogue("Ada", "Hi", 0.3, 1.2), TrackDialogue("Ada", "Later", 2.0, 2.5,
                                                                                    planned=True)),
                       voice_wave=(0, 3, 15, 9, 2) * 16, bed_wave=(4, 5, 6) * 30, detail="First scene."),
            ClipVisual(pending=True, note="The clip is missing."),
            ClipVisual(),
        )

    def test_every_clip_and_the_program_are_in_the_track_for_the_viewer(self):
        scenes = (timeline.Scene(HOSTILE, 4), timeline.Scene("Kit answers.", 3), timeline.Scene("", 2))
        project = self.project(scenes, aspect="Portrait 9:16")
        program = Program("/file=C:/edit.mp4", "Joined edit", "00:09 · Ready to export", True, "edit")
        markup = track_html(Draft(project, scenes, 1), ("Ready", "Generating", "Empty"), visuals=self.visuals(),
                            program=program, view="program", aspect=project.aspect)
        strip = elements(markup, "timeline-strip")[0][1]
        self.assertEqual((strip["data-total"], strip["data-view"], strip["data-aspect"], strip["data-max-seconds"]),
                         ("9", "program", "0.5625", str(video_gen.MAX_SECONDS)))
        self.assertEqual((strip["data-program"], strip["data-program-label"], strip["data-program-current"],
                          strip["data-program-kind"]), ("/file=C:/edit.mp4", "Joined edit", "true", "edit"))
        self.assertNotIn("data-program-hint", strip)
        blocks = [attrs for _, attrs in elements(markup, "timeline-clip")]
        self.assertEqual([(block["data-start"], block["data-end"]) for block in blocks], [("0", "4"), ("4", "7"),
                                                                                          ("7", "9")])
        self.assertEqual(blocks[0]["data-src"], '/file=C:/a "b".mp4')
        self.assertEqual(blocks[0]["data-slate"], " ".join(HOSTILE.split()))  # the viewer sets it as text
        self.assertEqual(blocks[0]["data-detail"], "First scene.")
        self.assertIn("data-pending", blocks[1])
        self.assertNotIn("data-src", blocks[1])
        self.assertEqual(blocks[1]["data-note"], "The clip is missing.")
        self.assertNotIn("data-pending", blocks[2])
        self.assertEqual([block["aria-pressed"] for block in blocks], ["false", "true", "false"])
        self.assertEqual(len(elements(markup, "timeline-trim")), 3)  # every clip's right edge
        ruler = elements(markup, "timeline-ruler")[0][1]
        self.assertEqual((ruler["role"], ruler["tabindex"], ruler["aria-valuenow"], ruler["aria-valuetext"]),
                         ("slider", "0", "4", "00:00:04:00"))
        joined = elements(markup, "timeline-joined-clip")
        self.assertEqual(len(joined), 1)
        self.assertIn("Joined edit · 00:09", joined[0][1]["aria-label"])
        self.assertIn("data-edit", elements(markup, "timeline-deck")[0][1])
        self.assertEqual(len(elements(markup, "timeline-playhead-head")), 1)
        self.assertEqual(elements(markup, "timeline-trim-tip")[0][1].get("role"), "status")

    def test_the_lanes_run_on_past_the_projects_end(self):
        """Every scene's column is its length at the track's time scale, so the lanes (and the ruler) run on
        through empty time after the project's end instead of stretching the scenes over the canvas."""
        scenes = (timeline.Scene("Ada.", 4), timeline.Scene("Kit.", 3), timeline.Scene("", 30))
        markup = track_html(Draft(scenes=scenes), program=Program("/p.mp4", "Joined edit", "", True, "edit"))
        strip = elements(markup, "timeline-strip")[0][1]
        self.assertEqual(strip["style"], "--scene-columns:calc(var(--tl-pps, 40px) * 4) "
                                         "calc(var(--tl-pps, 40px) * 3) calc(var(--tl-pps, 40px) * 30)")
        self.assertNotIn("fr", strip["style"])
        self.assertEqual(elements(markup, "timeline-end")[0][1]["aria-hidden"], "true")  # the end, after the lanes
        self.assertLess(markup.index('class="timeline-end"'), markup.index('class="timeline-playhead"'))
        # the server's 40 px a second until the script has measured the window is the script's own default
        self.assertEqual(re.search(r"ZOOM_DEFAULT = (\d+)", view.ZOOM_MATH_JS).group(1), "40")
        hint = elements(markup, "timeline-deck-hint")
        self.assertTrue(hint)
        for words in ("Scroll to move", "Ctrl+scroll", "Shift+Z"):
            self.assertIn(words, markup)
        css = StylesheetTests
        self.assertEqual(css.declared_in("#timeline-page .timeline-strip")["width"], "var(--tl-canvas, auto)")
        self.assertIn("var(--tl-length", css.declared_in("#timeline-page .timeline-joined-clip")["width"])
        end = css.declared_in("#timeline-page .timeline-end")
        self.assertEqual((end["left"], end["right"], end["pointer-events"]), ("var(--tl-length, 100%)", "0", "none"))
        scroller = css.declared_in("#timeline-page .timeline-scroll")
        self.assertEqual((scroller["overflow"], scroller["overscroll-behavior-x"], scroller["overflow-anchor"]),
                         ("auto", "contain", "none"))  # a visible scroll bar; no history swipe

    def test_an_earlier_export_is_offered_but_not_drawn_on_the_timeline(self):
        scenes = (timeline.Scene("Ada.", 4),)
        earlier = Program("/file=C:/old.mp4", "Earlier export", "Rendered from revision 1.", False, "earlier")
        markup = track_html(Draft(scenes=scenes), program=earlier)
        strip = elements(markup, "timeline-strip")[0][1]
        self.assertEqual((strip["data-program-label"], strip["data-program-current"]), ("Earlier export", "false"))
        self.assertFalse(elements(markup, "timeline-joined-clip"))
        hint = track_html(Draft(scenes=scenes), hint="Click Join and check.")
        self.assertEqual(elements(hint, "timeline-strip")[0][1]["data-program-hint"], "Click Join and check.")
        self.assertTrue(elements(track_html(Draft(scenes=scenes), joined=True), "timeline-joined-clip"))  # older callers

    def test_filmstrips_and_waveforms(self):
        scenes = (timeline.Scene("Ada.", 4), timeline.Scene("Kit.", 3), timeline.Scene("", 2))
        markup = track_html(Draft(scenes=scenes), ("Ready", "Saved", "Empty"), visuals=self.visuals())
        self.assertEqual(len(elements(markup, tag="img")), 12)
        lines = re.findall(r'<button type="button" class="timeline-audio-segment".*?</button>', markup)
        self.assertIn('class="tl-wave"', lines[0])  # measured: its slice of the voice waveform
        self.assertNotIn("tl-wave", lines[1])  # planned
        beds = re.findall(r'<button type="button" class="timeline-bed".*?</button>', markup)
        self.assertIn("<polygon", beds[0])
        self.assertNotIn("tl-wave", beds[1])
        self.assertEqual(view.wave_slice(tuple(range(80)), 0.3, 1.2), tuple(range(6, 24)))
        self.assertEqual(view.wave_slice((1, 2, 3), 2.0, 1.0), ())
        self.assertEqual(view.wave_svg((0, 0, 0)), "")
        shape = view.wave_svg((0, 15, 7))
        self.assertEqual(shape.count(","), 2 * 3 + 2)
        self.assertIn('viewBox="0 0 6 32"', shape)
        self.assertIn('preserveAspectRatio="none"', shape)

    def test_user_text_is_escaped_in_every_workspace_part(self):
        scene = timeline.Scene(HOSTILE, 4, background=HOSTILE, effects=(Effect(HOSTILE[:40], 1.0),),
                               lines=(Cue("<i>hi</i>", "ada"),))
        project = timeline.Project("0" * 32, 2, HOSTILE, (scene,), characters=("ada", "kit"))
        draft = Draft(project, project.scenes, 0, "f")
        with patch.object(timeline, "prepared", return_value={"assets": []}):
            display = ui_timeline._presentation(draft, project, ui_timeline._AUTO, Presentation(), problem=HOSTILE)
        for name in ("track", "bin", "status", "heading", "cast", "assets", "report", "lines", "effects"):
            with self.subTest(part=name):
                markup = getattr(display, name)
                self.assertNotIn("<script>", markup)
                self.assertNotIn("<b>", markup)
                self.assertNotIn("<i>hi", markup)
        self.assertIn("&lt;script&gt;", display.track)
        self.assertIn("Kit &lt;b&gt;&quot;K&quot;&lt;/b&gt;", display.cast)
        self.assertIn('role="alert"', display.heading)
        self.assertIn("&lt;script&gt;", display.heading)
        self.assertIn("&lt;script&gt;", display.bin)


class BackgroundMediaTests(_Isolated):
    def tone(self, path: Path, quiet: float, loud: float) -> Path:
        rate = audio.RATE
        silence = np.zeros(int(quiet * rate), np.float32)
        t = np.arange(int(loud * rate)) / rate
        wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        samples = np.concatenate([silence, wave])
        audio.write_wav(path, np.stack([samples, samples]), rate)
        return path

    def test_waveforms_are_read_once_per_file_version_on_a_worker(self):
        path = self.tone(self.root / "voice.wav", 1.0, 1.0)
        with patch.object(view, "_decode_peaks", wraps=view._decode_peaks) as decode:
            view.peaks(path, 0.0, 2.0)
            view.wait()
            levels = view.peaks(path, 0.0, 2.0)
            view.peaks(path, 0.0, 2.0)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(len(levels), 2 * view.WAVE_RATE)
            self.assertEqual(set(levels[:view.WAVE_RATE - 1]), {0})
            self.assertGreaterEqual(min(levels[view.WAVE_RATE + 1:]), view.WAVE_LEVELS - 1)
            view.peaks(path, 1.0, 1.0)  # the background stem's lead-in is skipped like this
            view.wait()
            self.assertGreaterEqual(min(view.peaks(path, 1.0, 1.0)[1:]), view.WAVE_LEVELS - 1)
            self.tone(path, 0.0, 2.5)  # a new version of the file is read again
            view.peaks(path, 0.0, 2.0)
            view.wait()
            self.assertGreaterEqual(min(view.peaks(path, 0.0, 2.0)[1:]), view.WAVE_LEVELS - 1)
            self.assertEqual(decode.call_count, 3)
        self.assertEqual(view.peaks(self.root / "missing.wav", 0.0, 2.0), ())
        broken = self.root / "broken.wav"
        broken.write_bytes(b"RIFF....WAVE")
        view.peaks(broken, 0.0, 1.0)
        view.wait()
        self.assertEqual(view.peaks(broken, 0.0, 1.0), ())

    def test_large_files_are_copied_once_per_version_on_a_worker(self):
        movie = self.root / "movie.mp4"
        movie.write_bytes(b"movie")
        with patch.object(ui_theme, "cache_media", wraps=ui_theme.cache_media) as copy:
            view.cached_later(movie)
            view.wait()
            cached = view.cached_later(movie)
            view.cached_later(movie)
            self.assertEqual(copy.call_count, 1)
        self.assertEqual(Path(cached).read_bytes(), b"movie")
        self.assertEqual(served(view.url_for(cached)), b"movie")
        self.assertEqual(view.cached_later(self.root / "missing.mp4"), "")

    def test_the_refresh_never_waits_for_the_viewer_copies(self):
        record = self.clip()
        leveled = self.root / ("f" * 16 + "-" + "b" * 24 + ".mp4")
        leveled.write_bytes(b"export mix")
        gate, calls = threading.Event(), []

        def slow(project, index):
            calls.append(index)
            gate.wait(10)
            return leveled
        self.leveled.side_effect = slow
        project = self.project([timeline.Scene("Ada waits", 4)])
        draft = Draft(project, project.scenes, 0, "f")
        with patch.object(timeline, "prepared", return_value={"assets": ["e" * 64]}), \
                patch.object(timeline, "asset", return_value=record), \
                patch.object(timeline, "outdated", return_value=[]):
            started = time.monotonic()
            first = ui_timeline._presentation(draft, project, ui_timeline._AUTO, Presentation())
            self.assertLess(time.monotonic() - started, 5)
            block = elements(first.track, "timeline-clip")[0][1]
            self.assertIn("data-pending", block)
            self.assertNotIn("data-src", block)
            gate.set()
            view.wait()
            second = ui_timeline._presentation(draft, project, ui_timeline._AUTO, first)
            ui_timeline._presentation(draft, project, ui_timeline._AUTO, second)
        block = elements(second.track, "timeline-clip")[0][1]
        self.assertNotIn("data-pending", block)
        self.assertEqual(served(block["data-src"]), b"export mix")
        self.assertEqual(calls, [0])  # once per clip version and levels

    def test_program_prefers_the_matching_export_then_the_joined_edit(self):
        project = self.project([timeline.Scene("Ada waits", 4)])
        new = ui_timeline._program(ui_timeline._NEW_PROJECT, ui_timeline._NEW_PROJECT, {"assets": []}, False, None,
                                   False, [])
        self.assertIsNone(new[0])
        self.assertIn("Save the project", new[1])
        joined = self.root / "edit.mp4"
        joined.write_bytes(b"joined")
        state = {"assets": ["e" * 64], "merged": {"attention": [1], "notes": []}}
        with patch.object(timeline, "movie_preview", return_value=None), \
                patch.object(timeline, "merged_clip", return_value=joined):
            ui_timeline._program(project, project, state, True, joined, False, ["e" * 64])
            view.wait()
            program, hint, download = ui_timeline._program(project, project, state, True, joined, False, ["e" * 64])
        self.assertEqual((program.label, program.current, program.kind, hint, download),
                         ("Joined edit", True, "edit", "", ""))
        self.assertIn("1 join to review", program.detail)
        self.assertEqual(served(program.source), b"joined")
        export = self.root / "timeline_movie.mp4"
        export.write_bytes(b"export")
        current = timeline.MoviePreview(export, project.key, 3)
        with patch.object(timeline, "movie_preview", return_value=current):
            ui_timeline._forget()
            ui_timeline._program(project, project, state, True, joined, False, ["e" * 64])
            view.wait()
            program, _, download = ui_timeline._program(project, project, state, True, joined, False, ["e" * 64])
        self.assertEqual((program.label, program.current, program.kind), ("Current export", True, "export"))
        self.assertEqual(Path(download).read_bytes(), b"export")
        edited = replace_scenes(project, (timeline.Scene("Ada runs", 4),))
        with patch.object(timeline, "movie_preview", return_value=timeline.MoviePreview(export, "old", 2, False)):
            ui_timeline._forget()
            program, _, _ = ui_timeline._program(edited, project, state, False, None, False, [None])
        self.assertEqual((program.label, program.current, program.kind), ("Earlier export", False, "earlier"))
        with patch.object(timeline, "movie_preview", return_value=None):
            ui_timeline._forget()
            _, hint, _ = ui_timeline._program(edited, project, state, False, None, False, [None])
        self.assertIn("Save the changes", hint)

    def test_a_program_being_copied_is_pending_and_one_that_cannot_play_is_not_current(self):
        project = self.project([timeline.Scene("Ada waits", 4)])
        draft = Draft(project, project.scenes, 0)
        joined = self.root / "edit.mp4"
        joined.write_bytes(b"joined")
        state = {"assets": ["e" * 64], "merged": {"attention": [], "notes": []}}
        gate = threading.Event()
        self.addCleanup(gate.set)

        def slow(saved, plan, verify=False):
            gate.wait(10)
            return joined

        def program():
            return ui_timeline._program(project, project, state, True, joined, False, ["e" * 64])
        with patch.object(timeline, "movie_preview", return_value=None), \
                patch.object(timeline, "merged_clip", side_effect=slow):
            pending, _, _ = program()  # its checksum is being checked and its copy made
            self.assertEqual((pending.source, pending.pending, pending.current), ("", True, True))
            strip = elements(track_html(draft, program=pending, joined=True), "timeline-strip")[0][1]
            self.assertEqual(strip["data-program-pending"], "true")
            gate.set()
            view.wait()
            ready, _, _ = program()
        self.assertEqual((ready.pending, ready.current, served(ready.source)), (False, True, b"joined"))
        self.assertNotIn("data-program-pending", elements(track_html(draft, program=ready), "timeline-strip")[0][1])
        with patch.object(timeline, "movie_preview", return_value=None), \
                patch.object(timeline, "merged_clip", return_value=None):  # the file no longer matches
            ui_timeline._forget()
            program()
            view.wait()
            broken, _, _ = program()
        self.assertEqual((broken.label, broken.source, broken.pending, broken.current), ("Joined edit", "", False, False))
        self.assertIn("Join and check again", broken.detail)
        markup = track_html(draft, program=broken, joined=True)
        strip = elements(markup, "timeline-strip")[0][1]
        self.assertEqual(strip["data-program-current"], "false")  # the viewer shows the scenes by default
        self.assertNotIn("data-program-pending", strip)
        self.assertTrue(elements(markup, "timeline-joined-clip"))  # V2 still offers it, with the reason
        export = self.root / "timeline_movie.mp4"
        export.write_bytes(b"export")
        with patch.object(timeline, "movie_preview", return_value=timeline.MoviePreview(export, project.key, 3)), \
                patch.object(ui_theme, "cache_media", return_value=None):  # its copy could not be made
            ui_timeline._forget()
            ui_timeline._program(project, project, state, True, None, False, ["e" * 64])
            view.wait()
            unreadable, _, download = ui_timeline._program(project, project, state, True, None, False, ["e" * 64])
        self.assertEqual((unreadable.label, unreadable.current, unreadable.pending, download),
                         ("Current export", False, False, ""))
        self.assertIn("could not be read", unreadable.detail)

    def test_a_new_mix_of_a_clip_keeps_the_last_one_on_screen(self):
        record = self.clip()
        gate = threading.Event()
        self.addCleanup(gate.set)

        def mix(project, index):
            level = project.scenes[index].voice_db
            path = self.root / ("f" * 16 + f"-{level:+g}.mp4")
            path.write_bytes(f"mix {level:+g}".encode())
            if level:
                gate.wait(10)
            return path
        self.leveled.side_effect = mix
        project = self.project([timeline.Scene("Ada waits", 4)])
        louder = replace_scenes(project, (replace(project.scenes[0], voice_db=3.0),))

        def block(current, prior):
            shown = ui_timeline._presentation(Draft(project, current.scenes, 0, "f"), current, ui_timeline._AUTO,
                                              prior)
            return shown, elements(shown.track, "timeline-clip")[0][1]
        with patch.object(timeline, "prepared", return_value={"assets": ["e" * 64]}), \
                patch.object(timeline, "asset", return_value=record), \
                patch.object(timeline, "outdated", return_value=[]):
            block(project, Presentation())
            view.wait()
            shown, first = block(project, Presentation())
            self.assertEqual(served(first["data-src"]), b"mix +0")
            shown, during = block(louder, shown)  # the Dialogue level changed: a new mix is being made
            self.assertEqual(served(during["data-src"]), b"mix +0")  # the last one keeps playing
            self.assertNotIn("data-pending", during)
            self.assertIn("Updating the preview…", during["data-detail"])
            gate.set()
            view.wait()
            _, after = block(louder, shown)
        self.assertEqual(served(after["data-src"]), b"mix +3")
        self.assertNotIn("Updating", after["data-detail"])


def replace_scenes(project, scenes):
    return replace(project, scenes=scenes)


class _Listener:
    """A listener with another trigger mode (what Gradio would do with it)."""

    def __init__(self, fn, trigger_mode):
        self.fn, self.trigger_mode = fn, trigger_mode


class _Gradio:
    """Quick clicks replayed the way Gradio 6.30 handles them, without a server.

    The page reads its fields when it sends an event and takes an answer's values when it arrives.
    While an event of a listener is on its way, a new one of that listener is sent ("multiple"),
    dropped ("once") or deferred ("always_last"): when the event on its way is answered, Gradio sends
    THAT event's data again, not the deferred one's (Blocks-*.js: dispatch() and Je()). The server runs
    the events of one function one at a time in arrival order and reads gr.State when it runs one
    (gradio/blocks.py preprocess_data); the track and the bin share track_action's queue."""

    SERVICE = 0.08  # seconds the server takes to answer an event

    def __init__(self, loaded: tuple):
        self.state = loaded[0]                                       # gr.State on the server
        self.page = [self.value(output, None) for output in loaded[1:13]]  # the inspector's fields
        self.received = []
        self._events, self._order = [], itertools.count()
        self._free = 0.0
        self._pending, self._deferred = {}, set()

    @staticmethod
    def value(output, current):
        if isinstance(output, dict) and output.get("__type__") == "update":
            return output.get("value", current)
        return output

    def click(self, at: float, listener, data: dict) -> None:
        heapq.heappush(self._events, (at, next(self._order), self._dispatch, (listener, data)))

    def run(self) -> "_Gradio":
        while self._events:
            now, _, step, args = heapq.heappop(self._events)
            step(now, *args)
        return self

    def _dispatch(self, now, listener, data):
        if self._pending.get(listener) is not None:
            if listener.trigger_mode == "once":
                return
            if listener.trigger_mode == "always_last":
                self._deferred.add(listener)
                return
        token = object()
        self._pending[listener] = token
        self._free = max(now, self._free) + self.SERVICE
        heapq.heappush(self._events, (self._free, next(self._order), self._answer,
                                      (listener, data, tuple(self.page), token)))

    def _answer(self, now, listener, data, inputs, token):
        self.received.append(data)
        result = listener.fn(self.state, *inputs, gr.EventData(None, data))
        self.state = self.value(result[0], self.state)
        self.page = [self.value(output, current) for output, current in zip(result[1:13], self.page)]
        if self._pending.get(listener) is token:
            self._pending[listener] = None
        if listener in self._deferred:
            self._deferred.discard(listener)
            self._dispatch(now, listener, data)


class QuickClickTests(_Isolated):
    """Clicks on the track and in the bin that come faster than the server answers: each one is
    applied in order, the last one wins, and no scene gets the inspector values of another."""

    SCENES = tuple(timeline.Scene(f"Scene {number} picture.", 4) for number in range(1, 5))

    def listeners(self) -> dict:
        demo = self.build()
        found = {}
        for fn in demo.fns.values():
            if getattr(fn.fn, "__name__", "") == "track_action":
                (block, _), = fn.targets
                found[demo.blocks[block].elem_id] = fn
        self.assertEqual(set(found), {"timeline-track", "timeline-scene-list"})
        return found

    def loaded(self, selected: int = 0) -> _Gradio:
        return _Gradio(ui_timeline._editor_values(Draft(None, self.SCENES, selected)))

    def prompts(self, draft) -> list[str]:
        return [scene.prompt for scene in draft.scenes]

    def test_quick_clicks_reach_the_server_in_order_and_the_last_one_wins(self):
        listeners = self.listeners()
        for name, listener in listeners.items():
            with self.subTest(listener=name):
                self.assertEqual(listener.trigger_mode, "multiple")
                gradio = self.loaded()
                for at, index in ((0.0, 1), (0.02, 2), (0.04, 3)):  # e.g. ↓ ↓ ↓ in the bin
                    gradio.click(at, listener, {"index": index})
                gradio.run()
                self.assertEqual(gradio.received, [{"index": 1}, {"index": 2}, {"index": 3}])
                self.assertEqual(gradio.state.selected, 3)
                self.assertEqual(gradio.state.scenes, self.SCENES)
                self.assertEqual(gradio.page[0], "Scene 4 picture.")  # the inspector shows the last choice
        # Why not Gradio's other modes: "always_last" sends the first click's data again, "once" drops
        # the later clicks; either way the inspector ends on scene 2 while the user chose scene 4.
        track = listeners["timeline-track"].fn
        for mode, received in (("always_last", [{"index": 1}, {"index": 1}]), ("once", [{"index": 1}])):
            with self.subTest(mode=mode):
                gradio, listener = self.loaded(), _Listener(track, mode)
                for at, index in ((0.0, 1), (0.02, 2), (0.04, 3)):
                    gradio.click(at, listener, {"index": index})
                gradio.run()
                self.assertEqual(gradio.received, received)
                self.assertEqual(gradio.state.selected, 1)

    def test_overlapping_clicks_never_put_one_scenes_fields_on_another(self):
        listeners = self.listeners()
        bin_list, track = listeners["timeline-scene-list"], listeners["timeline-track"]
        gradio = self.loaded()
        gradio.page[0] = "Scene 1, rewritten."  # typed, then scene 2 in the bin and scene 3 on the track
        gradio.click(0.0, bin_list, {"index": 1})
        gradio.click(0.03, track, {"index": 2})
        gradio.run()
        self.assertEqual(gradio.state.selected, 2)
        self.assertEqual(self.prompts(gradio.state), ["Scene 1, rewritten.", "Scene 2 picture.", "Scene 3 picture.",
                                                      "Scene 4 picture."])
        self.assertEqual(gradio.page[0], "Scene 3 picture.")
        # a length dragged on the track while a selection is on its way
        gradio = self.loaded()
        gradio.page[1] = 6  # the Duration field of scene 1
        gradio.click(0.0, track, {"index": 1})
        gradio.click(0.01, track, {"trim": 3, "seconds": 9})
        gradio.click(0.02, track, {"from": 0, "slot": 4})  # and scene 1 dragged to the end
        gradio.run()
        self.assertEqual([scene.seconds for scene in gradio.state.scenes], [4, 4, 9, 6])
        self.assertEqual(self.prompts(gradio.state)[-1], "Scene 1 picture.")
        self.assertEqual(gradio.state.selected, 3)

    def test_a_click_sent_again_before_its_answer_changes_nothing(self):
        action = self.listeners()["timeline-track"].fn
        gradio = self.loaded()
        first = action(gradio.state, *gradio.page, gr.EventData(None, {"index": 1}))
        again = action(first[0], *gradio.page, gr.EventData(None, {"index": 1}))  # a double-click
        self.assertEqual(first[0].selected, 1)
        self.assertTrue(all(output == SKIP for output in again))  # what was typed meanwhile stays

    def test_values_from_before_the_last_load_are_not_applied(self):
        loaded = ui_timeline._editor_values(Draft(None, self.SCENES, 0))
        page = [_Gradio.value(output, None) for output in loaded[1:13]]
        draft = loaded[0]
        self.assertTrue(ui_timeline._current_form(draft, page))
        moved = replace(draft, selected=1, form="later")  # the server has loaded scene 2 since
        self.assertFalse(ui_timeline._current_form(moved, page))
        self.assertTrue(ui_timeline._current_form(moved, [*page[:6], None, None, page[8], "", *page[10:]]))  # a new page
        rows = '{"key": "later", "rows": []}'  # a row edited after the new load
        self.assertTrue(ui_timeline._current_form(moved, [*page[:6], rows, *page[7:]]))
        self.assertEqual(ui_timeline._apply_form(moved, [*page[:1], 6, *page[2:]]), moved)
        # a refresh sent before the answer arrived previews the draft, not the earlier inspector
        project = timeline.Project("0" * 32, 3, "Rain", self.SCENES)
        typed = ["Typed for scene 1.", *page[1:]]
        with patch.object(timeline, "prepared", return_value={"assets": []}):
            track = ui_timeline._refresh(replace(moved, baseline=project), *typed, "Rain", "Landscape 16:9",
                                         True, "", [], False, "realistic", "", Presentation())[0]
        self.assertNotIn("Typed for scene 1.", track)
        self.assertEqual([attrs["aria-pressed"] for _, attrs in elements(track, "timeline-clip")],
                         ["false", "true", "false", "false"])


class EditorActionTests(_Isolated):
    def form(self, scene):
        return (scene.prompt, scene.seconds, scene.cut, scene.cast is not None, list(scene.cast or ()), "",
                None, None, scene.background, None, scene.voice_db, scene.background_db)

    def test_dragging_a_clip_edge_changes_its_length_like_the_duration_field(self):
        scenes = (timeline.Scene("Ada", 4, lines=(Cue("Hi", "ada"),)), timeline.Scene("Kit", 3))
        draft = Draft(None, scenes, 0, "f")
        trimmed = ui_timeline._trimmed(draft, 1, 9)
        self.assertEqual((trimmed.scenes[1].seconds, trimmed.selected), (9, 1))
        self.assertEqual(trimmed.scenes[0], scenes[0])
        for index, seconds in ((1, 0), (1, 31), (1, 2.5), (1, True), (2, 5), (-1, 5), ("1", 5)):
            with self.subTest(index=index, seconds=seconds), self.assertRaises(ValueError):
                ui_timeline._trimmed(draft, index, seconds)
        demo = self.build()
        action = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "track_action"
                      and any(getattr(demo.blocks.get(block), "elem_id", "") == "timeline-track"
                              for block, _ in fn.targets))
        result = action.fn(draft, *self.form(scenes[0]), gr.EventData(None, {"trim": 1, "seconds": 9}))
        edited = result[0]
        self.assertEqual((edited.scenes[1].seconds, edited.selected), (9, 1))
        self.assertEqual(result[2], 9)  # the Duration field shows the new length
        self.assertEqual(result[-1], SKIP)  # the viewer source stays
        self.assertEqual(edited.scenes[0].lines, scenes[0].lines)
        with self.assertRaises(gr.Error):
            action.fn(draft, *self.form(scenes[0]), gr.EventData(None, {"trim": 5, "seconds": 9}))
        selected = action.fn(draft, *self.form(scenes[0]), gr.EventData(None, {"index": 1}))
        self.assertEqual((selected[0].selected, selected[-1]), (1, SKIP))
        joined = action.fn(draft, *self.form(scenes[0]), gr.EventData(None, {"joined": True}))
        self.assertEqual(joined[-1], ui_timeline._PROGRAM)
        bin_click = [fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "track_action"]
        self.assertEqual(len(bin_click), 2)  # the track and the bin's scene list
        self.assertEqual({fn.trigger_mode for fn in bin_click}, {"multiple"})  # see QuickClickTests

    def test_opening_a_project_fills_the_header_status_cast_and_track_at_once(self):
        saved = timeline.save(None, None, "Rain", (timeline.Scene("Ada waits", 4),), "Landscape 16:9",
                              True, ["ada"])
        demo = self.build()
        opener = next(fn for fn in demo.fns.values() if fn.api_name == "timeline_open")
        fresh = (Draft(), "", 5, False, False, [], "", None, None, "", None, 0, 0,
                 "Untitled movie", "Landscape 16:9", True, "", [])
        with patch.object(timeline, "projects", return_value=[saved]):
            result = opener.fn(saved.id, *fresh)
        values = {getattr(block, "elem_id", None): value for block, value in zip(opener.outputs, result)}
        self.assertEqual(len(result), len(opener.outputs))
        self.assertIn("Scene 1", values["timeline-scene-heading"])
        self.assertIn("Revision 1 saved", values["timeline-status"])
        self.assertIn("Ada", values["timeline-cast-cards"])
        self.assertIn('data-project="' + saved.id, values["timeline-track"])
        self.assertIn("Ada waits", values["timeline-scene-list"])
        self.assertEqual(values["timeline-view"], ui_timeline._AUTO)
        seen = next(value for block, value in zip(opener.outputs, result) if isinstance(value, Presentation))
        self.assertEqual(seen.heading, values["timeline-scene-heading"])
        # every part of the display is sent, also those equal to an empty display (no download, no retry,
        # no conversion): the page may still show the previous project's
        first = len(result) - 2 - len(ui_timeline._OPEN_OUTPUTS)
        display = dict(zip(ui_timeline._OPEN_OUTPUTS, result[first:first + len(ui_timeline._OPEN_OUTPUTS)]))
        for name, update in display.items():
            self.assertNotEqual(update, SKIP, name)
        self.assertEqual(display["download"], gr.update(value=None, visible=False))
        self.assertEqual(display["retry"], gr.update(value="Retry", visible=False))
        self.assertEqual(display["convert"], gr.update(visible=False))

    def test_open_without_a_chosen_project_warns_and_changes_nothing(self):
        demo = self.build()
        opener = next(fn for fn in demo.fns.values() if fn.api_name == "timeline_open")
        fresh = (Draft(), "", 5, False, False, [], "", None, None, "", None, 0, 0,
                 "Untitled movie", "Landscape 16:9", True, "", [])
        for empty in ("", None):
            with self.subTest(value=empty), patch.object(gr, "Warning") as warning:
                result = opener.fn(empty, *fresh)  # no gr.Error: it would mark the project list as failed
            warning.assert_called_once_with("Choose a saved project to open.")
            self.assertEqual(len(result), len(opener.outputs))
            self.assertTrue(all(value == SKIP for value in result))

    def test_refresh_fills_every_output_and_then_skips_them(self):
        demo = self.build()
        refresh = next(fn for fn in demo.fns.values() if fn.fn is ui_timeline._refresh)
        self.assertEqual(len(refresh.outputs), len(ui_timeline.OUTPUTS))
        names = [getattr(block, "elem_id", None) for block in refresh.outputs]
        for name in ("timeline-track", "timeline-scene-list", "timeline-status", "timeline-scene-heading",
                     "timeline-cast-cards", "timeline-assets", "timeline-queue-line"):
            self.assertIn(name, names)
        project = self.project([timeline.Scene("Ada waits", 4), timeline.Scene("Kit", 2)])
        inputs = (Draft(project, project.scenes, 0, "f"), "Ada waits", 4, False, False, [], "", None, None, "",
                  None, 0, 0, "Rain", "Landscape 16:9", True, "", ["ada", "kit"], False, "realistic", "program")
        with patch.object(timeline, "prepared", return_value={"assets": []}):
            first = ui_timeline._refresh(*inputs, Presentation())
            again = ui_timeline._refresh(*inputs, first[-1])
        self.assertEqual(len(first), len(refresh.outputs))
        self.assertIn('data-view="program"', first[0])
        self.assertTrue(all(value == SKIP for value in again))  # nothing new: no redraw, no flicker
        changed = ui_timeline._refresh(*inputs[:-1], "bogus", first[-1])
        self.assertIn('data-view=""', changed[0])  # an unknown viewer source falls back to automatic

    def test_status_heading_bin_cast_assets_and_queue_chip(self):
        project = self.project([timeline.Scene("Ada waits", 4, references=("a" * 64,)),
                                timeline.Scene("Kit", 3, references=("b" * 64, "a" * 64)),
                                timeline.Scene("Ada again", 2, references=("b" * 64,))])
        with patch.object(timeline, "prepared", return_value={"assets": []}), \
                patch.object(timeline, "reference_path", side_effect=lambda name: None):
            clean = ui_timeline._presentation(Draft(project, project.scenes, 0, "f"), project, ui_timeline._AUTO,
                                              Presentation())
            edited = replace_scenes(project, (timeline.Scene("Ada runs", 4), *project.scenes[1:]))
            dirty = ui_timeline._presentation(Draft(project, edited.scenes, 0, "f"), edited, ui_timeline._AUTO,
                                              Presentation())
        self.assertIn('data-state="saved"', clean.status)
        self.assertIn("Revision 3 saved", clean.status)
        self.assertIn("16:9 · 480p · Audio on", clean.status)
        self.assertIn('data-state="dirty"', dirty.status)
        self.assertIn("Unsaved changes", dirty.status)
        self.assertIn("Scene 1 <small>of 3</small>", clean.heading)
        self.assertIn("00:00–00:04", clean.heading)
        cards = [attrs for _, attrs in elements(clean.bin, "tl-bin-scene")]
        self.assertEqual([card["aria-pressed"] for card in cards], ["true", "false", "false"])
        self.assertEqual([card["data-scene"] for card in cards], ["0", "1", "2"])
        self.assertIn("Voice: Hanna", clean.cast)  # Ada's own voice
        self.assertIn("Default voice", clean.cast)
        uses = [attrs for _, attrs in elements(clean.assets, tag="button") if "data-use-reference" in attrs]
        self.assertEqual([use["data-use-reference"] for use in uses], ["b" * 64])  # in scenes 2 and 3, not in 1
        self.assertIn("Scenes 2, 3", clean.assets)
        full = view.assets_html(1, [], [view.AssetItem("c" * 64, "", "Scene 2", usable=False)])
        self.assertIn("disabled", elements(full, tag="button")[0][1])
        job = queue_manager.Job(4, timeline.PREPARE, "Generate clips · Rain", {"project": project.id},
                                status=queue_manager.RUNNING, progress=0.42, desc="Scene 2 of 3")
        chip = ui_timeline._queue_line(project, {"assets": ["e" * 64, None]}, job, False, None)
        self.assertIn('data-state="running"', chip)
        self.assertIn("Generate clips: 42% · Scene 2 of 3", chip)
        self.assertIn("class='bar'", chip)
        self.assertIn("1/3 clips ready", chip)
        failed = ui_timeline._queue_line(project, {"assets": []}, None, False,
                                         {"scene": 1, "error": "Out of <memory>.", "kind": timeline.PREPARE})
        self.assertIn('data-state="failed"', failed)
        self.assertIn("Out of &lt;memory&gt;.", failed)


class StylesheetTests(unittest.TestCase):
    """Gradio copies rules inside @media only with a '.contain' prefix and splits their selector lists
    on every comma; @container, @supports and @layer rules are dropped."""

    @staticmethod
    def media_rules(css: str) -> list[str]:
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        selectors = []
        for match in re.finditer(r"@media[^{]*\{", css):
            depth, position = 1, match.end()
            while depth:
                depth += {"{": 1, "}": -1}.get(css[position], 0)
                position += 1
            body = css[match.end():position - 1]
            selectors += [selector.strip() for selector in re.findall(r"([^{}]+)\{[^{}]*\}", body)]
        return selectors

    def test_rules_inside_media_queries_survive_gradios_prefix(self):
        for css in (view.CSS, ui_theme.CSS):
            for selector in self.media_rules(css):
                with self.subTest(selector=selector):
                    depth = 0
                    for char in selector:
                        depth += {"(": 1, ")": -1}.get(char, 0)
                        self.assertFalse(char == "," and depth > 0, "a comma inside :is()/:not() in @media")
                    if css is view.CSS:
                        for part in selector.split(","):
                            self.assertTrue(part.strip().startswith("#timeline-page"), part)
            for rule in ("@container", "@supports", "@layer"):
                self.assertNotIn(rule, css)

    def test_the_workspace_fills_the_window_below_its_top(self):
        workspace = re.search(r"#timeline-page #timeline-workspace \{([^}]*)\}", view.CSS).group(1)
        self.assertIn("100dvh", workspace)
        self.assertIn("var(--tl-top", workspace)
        # exactly the window's height below its top: no bottom padding to leave room for any more
        self.assertNotRegex(workspace, r"var\(--tl-top[^)]*\)\s*-")
        self.assertIn("--tl-top", view.SHELL_JS)
        self.assertIn("dataset.size", view.SHELL_JS)
        self.assertIn("getElementById('topbar')", view.SHELL_JS)  # a top bar that wraps moves the top
        for size in ("medium", "narrow"):
            self.assertIn(f'#timeline-workspace[data-size="{size}"]', view.CSS)
        # the page reaches the window's edges below the top bar: Gradio's main column has no cap
        # (.fillable) and no padding, and the pages lose their padding while the Timeline is shown
        self.assertIn('#pages:has(> #timeline-page:not([style*="display: none"])) { padding: 0 !important; }',
                      ui_theme.CSS)
        self.assertIn(".gradio-container .main { max-width: none !important; padding: 0 !important; }",
                      ui_theme.CSS)
        self.assertIn(".gradio-container .main > .wrap { padding: 0 !important; }", ui_theme.CSS)
        # no sidebar any more to tell about the page's width
        self.assertNotIn("dispatchEvent(new Event('resize'))", view.SHELL_JS)

    @staticmethod
    def rule(selector: str, css: str = view.CSS) -> str:
        found = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
        assert found, selector
        return found.group(1)

    @staticmethod
    def rules(css: str = view.CSS) -> list[tuple[str, dict, str]]:
        """(selector, declarations, @media query or "") of every style rule; selector lists split at their
        top-level commas."""
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        found = []

        def walk(text, media):
            position = 0
            while (opening := text.find("{", position)) >= 0:
                depth, end = 1, opening + 1
                while depth:
                    depth += {"{": 1, "}": -1}.get(text[end], 0)
                    end += 1
                prelude, body = text[position:opening].strip(), text[opening + 1:end - 1]
                if prelude.startswith("@media"):
                    walk(body, prelude)
                elif not prelude.startswith("@"):
                    declarations = {p.strip(): v.strip() for p, _, v in (d.partition(":") for d in body.split(";"))
                                    if v.strip()}
                    found.extend((" ".join(part.split()), declarations, media)
                                 for part in ui_theme._top_split(prelude))
                position = end
        walk(css, "")
        return found

    @classmethod
    def declared_in(cls, selector: str, media: str = "") -> dict:
        """Every declaration of the rules for exactly this selector (in this @media query), merged."""
        merged = {}
        for name, declarations, query in cls.rules():
            if name == selector and query == media:
                merged.update(declarations)
        return merged

    def declared(self, selector: str, media: str = "") -> dict:
        return self.declared_in(selector, media)

    def test_panels_sit_flush_with_single_hairlines(self):
        """No gap, no rounded corner and no doubled border between the app bar, the bin, the viewer, the
        inspector and the timeline, in every layout."""
        for selector in ("#timeline-page #timeline-workspace", "#timeline-page .tl-body"):
            self.assertEqual(self.declared(selector)["gap"], "0 !important", selector)
        self.assertNotIn("--tl-gap", view.CSS)
        self.assertNotIn("--tl-radius", view.CSS)
        panel = self.declared("#timeline-page .tl-panel")
        self.assertEqual((panel["border"], panel["border-radius"]), ("0 !important", "0 !important"))
        self.assertEqual(panel["padding"], "10px !important")  # room inside, none around
        appbar = self.declared("#timeline-page .tl-appbar")
        self.assertEqual((appbar["border"], appbar["border-radius"]), ("0", "0"))
        hairline = "1px solid var(--c-border)"
        self.assertEqual(appbar["border-bottom"], hairline)
        # wide and medium: the bin and the viewer draw their right edge, the timeline its top edge, the
        # inspector none (its left edge is the viewer's right one; its right edge is the window's)
        self.assertEqual(self.declared("#timeline-page .tl-bin")["border-right"], hairline + " !important")
        self.assertEqual(self.declared("#timeline-page .tl-viewer-panel")["border-right"], hairline + " !important")
        self.assertEqual(self.declared("#timeline-page .tl-timeline-panel")["border-top"], hairline + " !important")
        self.assertFalse([name for name in self.declared("#timeline-page .tl-inspector") if name.startswith("border")])
        # stacked: a hairline under every panel but the timeline, also before the page script has measured
        for stacked, media in (('#timeline-page #timeline-workspace[data-size="narrow"]', ""),
                               ("#timeline-page #timeline-workspace:not([data-size])", "@media (max-width: 767px)")):
            with self.subTest(layout=stacked):
                for name in (".tl-viewer-panel", ".tl-inspector") + ((".tl-bin",) if not media else ()):
                    self.assertEqual(self.declared(f"{stacked} {name}", media)["border-bottom"],
                                     hairline + " !important", name)
                self.assertEqual(self.declared(f"{stacked} .tl-viewer-panel", media)["border-right"], "0 !important")
                self.assertEqual(self.declared(f"{stacked} .tl-timeline-panel", media)["border-top"], "0 !important")
        # no rule anywhere frames or rounds a panel, the app bar or the workspace, or puts a gap between panels
        workspace = r"#timeline-workspace(\[[^\]]*\]|:not\([^)]*\))*"
        framed = re.compile(rf"(\.tl-(panel|bin|viewer-panel|inspector|timeline-panel|appbar|body)|{workspace})$")
        spaced = re.compile(rf"(\.tl-body|{workspace})$")
        for selector, declarations, _ in self.rules():
            last = selector.split()[-1]
            if not framed.search(last):
                continue
            with self.subTest(selector=selector):
                for name in ("border", "border-radius") + (("gap",) if spaced.search(last) else ()):
                    self.assertTrue(declarations.get(name, "0").startswith("0"), (name, declarations.get(name)))

    def test_narrow_panels_keep_their_heights_and_follow_the_tab_order(self):
        # every panel of the body has "height: auto !important"; the narrow heights must win over it
        assert "height: auto !important" in self.rule("#timeline-page .tl-body > *")
        for panel, height in ((".tl-viewer-panel", "min(62vw, 420px)"), (".tl-bin", "340px"),
                              (".tl-timeline-panel", "300px")):
            with self.subTest(panel=panel):
                self.assertIn(f"height: {height} !important",
                              self.rule(f'#timeline-page #timeline-workspace[data-size="narrow"] {panel}'))
        stacked = self.rule('#timeline-page #timeline-workspace[data-size="narrow"][data-bin="open"] .tl-body')
        self.assertIn('"bin" "viewer" "inspector" "timeline"', stacked)  # the DOM's (and Tab's) order
        card = self.rule("#timeline-page .tl-slate-card")
        self.assertIn("box-sizing: border-box", card)  # its padding stays inside a narrow stage
        phone = re.search(r"@media \(max-width: 767px\) \{(.*?)\n\}", view.CSS, re.S).group(1)
        self.assertIn("flex: 1 1 100% !important", self.rule(
            "#timeline-page .tl-appbar-production > #timeline-queue-line:has(.tl-queue)", phone))

    def test_every_lane_fits_the_timeline_panels_floor(self):
        """The sideways scroll bar is always there (the canvas runs a view past the project's end), so the lanes'
        smallest heights, V2 included, must fit the timeline row's floor with it: on a short window no lane
        hides below the view, out of reach of the wheel (which moves along the timeline)."""
        def smallest(rows: str) -> int:  # the fixed rows and the minimum of every minmax()
            return sum(int(value) for value in re.findall(r"(\d+)px", rows))
        plain = smallest(self.declared("#timeline-page .timeline-deck")["--tl-rows"])
        edit = smallest(self.declared("#timeline-page .timeline-deck[data-edit]")["--tl-rows"])
        self.assertEqual((plain, edit), (132, 158))
        self.assertEqual(self.declared("#timeline-page .timeline-board")["min-height"], f"{plain}px")
        rows = self.declared("#timeline-page .tl-body")["grid-template-rows"]
        floor = int(re.fullmatch(r"minmax\(0, 1fr\) clamp\((\d+)px, 40%, 460px\)", rows).group(1))
        # the panel's top hairline, the toolbar (6 + 28 + 6 px and its hairline), the hint line (about 24 px)
        # and a scroll bar (15-17 px on Windows)
        around = 1 + 41 + 24 + 17
        self.assertGreaterEqual(floor - around, edit)
        # stacked, the timeline is 300 px high and its toolbar may wrap to two rows
        narrow = self.declared('#timeline-page #timeline-workspace[data-size="narrow"] .tl-timeline-panel')["height"]
        self.assertEqual(narrow, "300px !important")
        self.assertGreaterEqual(300 - (75 + 24 + 17), edit)

    def test_focus_rings_are_neither_clipped_nor_painted_over(self):
        tabs = self.rule("#timeline-page :is(#timeline-inspector-tabs, #timeline-bin-tabs) > .tabitem")
        top, right, bottom, left = (int(value) for value in
                                    re.search(r"padding: (\d+)px (\d+)px (\d+)px (\d+)px", tabs).groups())
        self.assertGreaterEqual(min(top, left), 4)  # a 2 px ring 2 px outside a control fits
        self.assertIn("margin: 0 0 0 -4px", tabs)  # and the controls stay where they were
        ring = self.rule("#timeline-page .tl-stage:focus-visible::after")
        for part in ('content: ""', "position: absolute", "z-index", "pointer-events: none", "box-shadow"):
            self.assertIn(part, ring)  # above the video, the slate and the caption
        self.assertNotIn("#timeline-page .tl-stage:focus-visible {", view.CSS)


class TimeScaleTests(unittest.TestCase):
    """The track's zoom: a time scale in pixels per second, independent of the window. ZOOM_MATH_JS holds the
    functions TRACK_JS zooms, fits, scrolls and follows with; they run here in Node.js."""

    def run_math(self, expressions: dict) -> dict:
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is not installed")
        body = "".join(f"out[{json.dumps(name)}] = ({code});\n" for name, code in expressions.items())
        script = ("'use strict';\n" + view.ZOOM_MATH_JS + "\nconst out = {};\n" + body
                  + "process.stdout.write(JSON.stringify(out));\n")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "zoom.js"
            path.write_text(script, encoding="utf-8")
            result = subprocess.run([node, str(path)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_zooming_keeps_the_time_under_the_pointer(self):
        # (scale, left edge in seconds, factor, pointer in px from the view's left edge, project seconds)
        cases = [(80, 0, 1.5, 300, 12), (120, 1.25, 1 / 1.5, 300, 12), (40, 10, 3, 900, 60), (300, 20.5, .5, 450, 60),
                 (5, 100, 1.3826, 640, 600), (599, 3, .9, 10, 12)]
        found = self.run_math({"views": "%s.map(([p, l, f, a, t]) => { const v = zoomAround(p, l, p * f, a, t); "
                                         "return [l + a / p, v.left + a / v.pps, v.at, p * f, v.pps]; })"
                                         % json.dumps(cases)})
        for case, (before, after, anchored, wanted, scale) in zip(cases, found["views"]):
            with self.subTest(case=case):
                self.assertAlmostEqual(after, before, places=9)  # the same time stays under the pointer
                self.assertAlmostEqual(anchored, before, places=9)
                self.assertAlmostEqual(scale, wanted, places=9)

    def test_the_scale_and_the_left_edge_stay_in_their_ranges(self):
        found = self.run_math({
            "range": "[ZOOM_MIN, ZOOM_MAX, ZOOM_DEFAULT, ZOOM_STEP]",
            "in": "zoomAround(80, 0, 1e6, 500, 12).pps", "out": "zoomAround(80, 0, 1e-3, 500, 12).pps",
            "start": "zoomAround(80, .5, 20, 500, 12).left",  # zooming out near the start stops at 0
            "end": "zoomAround(100, 11, 400, 900, 12).left",  # the left edge goes no further than the end
            "odd": "[clampPps(NaN), clampPps(-5), clampPps(Infinity), clampPps(1), clampPps(1e4)]",
            # steps that multiply and divide leave rounding errors: a scale that close to a limit is the limit
            "snapped": "[clampPps(ZOOM_MAX * (1 - 1e-12)), clampPps(ZOOM_MIN * (1 + 1e-12)), clampPps(599.9), "
                       "clampPps(2.1)]",
            "levels": "[zoomLevel(ZOOM_MIN), zoomLevel(ZOOM_MAX), zoomLevel(levelPps(.37)), levelPps(.5), "
                      "levelPps(-1), levelPps(2)]",
        })
        low, high, default, step = found["range"]
        self.assertEqual((low, high, default, step), (2, 600, 40, 1.5))
        self.assertEqual((found["in"], found["out"]), (600, 2))
        self.assertEqual((found["start"], found["end"]), (0, 12))
        self.assertEqual(found["odd"], [40, 40, 40, 2, 600])
        self.assertEqual(found["snapped"], [600, 2, 599.9, 2.1])
        bottom, top, middle, half, under, over = found["levels"]
        self.assertEqual((bottom, top), (0, 1))
        self.assertAlmostEqual(middle, .37, places=9)  # the slider is logarithmic
        self.assertAlmostEqual(half, (2 * 600) ** .5, places=9)
        self.assertEqual((under, over), (2, 600))

    def test_fit_shows_the_whole_project_with_a_margin(self):
        cases = [(12, 1000), (30, 1000), (300, 1000), (12, 311), (60, 2400)]
        found = self.run_math({"fits": "%s.map(([t, r]) => fitPps(t, r))" % json.dumps(cases),
                               "edges": "[fitPps(3600, 1000), fitPps(1, 1000), fitPps(0, 1000), fitPps(12, 0)]"})
        self.assertAlmostEqual(found["fits"][0], 80)  # 12 s in 1000 px: 960 px, 40 px to spare
        for (seconds, width), scale in zip(cases, found["fits"]):
            with self.subTest(seconds=seconds, width=width):
                margin = width - seconds * scale
                self.assertGreaterEqual(margin, 16 - 1e-9)  # it all fits, with a little room after the end
                self.assertLessEqual(margin, max(16, width * .04) + 1e-9)  # and fills the view
        # beyond the range a project cannot fit (it scrolls) or is shown at the closest scale
        self.assertEqual(found["edges"], [2, 600, 40, 40])

    def test_the_canvas_runs_one_view_past_the_end(self):
        found = self.run_math({
            "widths": "[canvasWidth(12, 80, 1000), canvasWidth(12, 2, 1000), canvasWidth(0, 80, 1000), "
                      "canvasWidth(30, 600, 311)]",
            "limits": "[leftLimit(12), leftLimit(0)]",
        })
        # far out a 30 s project is a 60 px block; the canvas always holds one view of empty time after it
        self.assertEqual(found["widths"], [12 * 80 + 1000, 12 * 2 + 1000, 1000, 30 * 600 + 311])
        self.assertEqual(found["limits"], [12, 0])  # the view can scroll until the end is at its left edge

    def test_the_wheel_and_a_pinch_zoom_smoothly(self):
        deltas = [-400, -160, -100, -30, -5, 0, 5, 30, 100, 160, 400]
        found = self.run_math({"factors": "%s.map(wheelFactor)" % json.dumps(deltas),
                               "pinch": "wheelFactor(-4)"})
        factors = found["factors"]
        self.assertEqual(factors, sorted(factors, reverse=True))  # up zooms in, down zooms out
        self.assertEqual(factors[0], factors[1])  # one event never jumps further than a fast notch
        self.assertEqual(factors[5], 1)
        notch = factors[2]
        self.assertTrue(1.3 < notch < 1.5, notch)
        self.assertAlmostEqual(notch * factors[8], 1, places=9)  # a notch up and one down cancel out
        self.assertAlmostEqual(found["pinch"], 2.718281828459045 ** .032, places=9)  # a pinch follows the fingers

    def test_the_ruler_labels_stay_apart_and_show_frames_when_zoomed_in(self):
        scales = [2, 7, 40, 80, 300, 600]
        found = self.run_math({"steps": "%s.map(p => rulerStep(p))" % json.dumps(scales)})
        for scale, step in zip(scales, found["steps"]):
            with self.subTest(scale=scale):
                self.assertGreaterEqual(step * scale, 64)  # labels never overlap
        self.assertEqual(found["steps"][3], 1)
        self.assertGreaterEqual(found["steps"][0], 32)
        frames = found["steps"][-1] * 30
        self.assertLess(found["steps"][-1], 1)
        self.assertAlmostEqual(frames, round(frames), places=9)  # whole frames

    def test_playback_turns_the_page_and_a_jump_is_shown(self):
        found = self.run_math({
            "paging": "[followLeft(0, 300, 1000, 2, 12, true), followLeft(0, 300, 1000, 3.4, 12, true), "
                      "followLeft(5, 300, 1000, 1, 12, true), followLeft(0, 300, 1000, 12, 12, true)]",
            "jumps": "[followLeft(0, 80, 1000, 11, 12, false), followLeft(10, 300, 1000, 2, 12, false)]",
        })
        stays, turned, back, end = found["paging"]
        self.assertEqual(stays, 0)  # in view: the view stays
        self.assertAlmostEqual(turned, 3.4 - 40 / 300)  # past the right edge: the playhead starts the next page
        self.assertAlmostEqual(back, 1 - 40 / 300)
        self.assertLessEqual(end, 12)
        self.assertEqual(found["jumps"][0], 0)
        self.assertAlmostEqual(found["jumps"][1], 2 - 1000 / 300 * .15)

    def test_a_drag_held_at_the_edge_moves_the_view_until_the_end_is_in_view(self):
        """A scrub or a dragged clip held at the view's right edge moves the view on (TRACK_JS's canFollow)
        until the project's end is in view, clear of the 36 px edge zone, and no further into the empty time."""
        # (left edge in seconds, scale, view width, project seconds)
        cases = [(10, 600, 1000, 12), (10.5, 600, 1000, 12), (0, 80, 1000, 12), (0, 600, 1000, 12), (40, 2, 900, 600)]
        found = self.run_math({"ends": "%s.map(([l, p, r, t]) => endInView(l, p, r, t, 36))" % json.dumps(cases)})
        self.assertEqual(found["ends"], [False, True, True, False, False])
        self.assertIn("!endInView(leftTime, pps, viewWidth, total, EDGE)", view.TRACK_JS)
        # a dragged clip is accepted on dragenter as on dragover: the view may move another element under it
        self.assertIn("element.addEventListener('dragenter'", view.TRACK_JS)

    def test_fit_is_an_action_and_the_view_is_kept(self):
        """No zoom mode depends on the window: Fit fits once (when asked, or when a project is opened for the
        first time) and a resize keeps the scale; the view (scale and left edge) is kept per project in
        the script and in the browser's storage, which may be unavailable."""
        zoom = view.zoom_html()
        self.assertNotIn("aria-pressed", zoom)
        self.assertNotIn("aria-pressed", view.ZOOM_JS)
        self.assertNotIn("Fit", view.ZOOM_JS)
        track = view.TRACK_JS
        for gone in ("fitScale", "MAX_SCALE", "zoom = clamp(level"):
            self.assertNotIn(gone, track)
        self.assertEqual(track.count("fitPps("), 2)  # its definition and the one layout that fits
        # fitted when the page loads, when another project opens without a view of its own, and by Fit
        self.assertEqual(sorted(re.findall(r"\bneedsFit = (!?\w+)", track)), ["!known", "!settled", "true", "true"])
        self.assertIn("new ResizeObserver(() => sync())", track)  # a resize lays out at the same scale
        self.assertIn("'contsy.timeline.views'", track)
        for match in re.finditer("localStorage", track):
            before = track[:match.start()]
            self.assertGreater(before.rfind("try {"), before.rfind("} catch"), "storage outside try")
        # every zoom input reaches the same handlers; wheel zoom is taken from the browser (Ctrl or a pinch)
        for part in ("'wheel'", "passive: false", "event.ctrlKey || event.metaKey", "requestAnimationFrame(applyView)",
                     "'timeline:zoom'", "key === 'Z' || key === 'z'"):
            self.assertIn(part, track)


class TrackViewTests(unittest.TestCase):
    """What keeps the view (zoom and scroll position) where the user is working. The track's own edits are
    answered like its own clicks: the server's answer selects the edited scene, and the track expects that
    selection (asked) instead of following it into view like a scene chosen elsewhere (the bin, the toolbar, a
    new scene). The behaviour itself is exercised on a scratch DOM and in a browser; these tests hold the
    contract between the script and the server and the pieces that are easy to lose in an edit."""

    @staticmethod
    def wheel_handler() -> str:
        track = view.TRACK_JS
        start = track.index("panel().addEventListener('wheel'")
        return track[start:track.index("}, { passive: false });", start)]

    def test_the_track_expects_the_selection_its_trims_and_drops_bring(self):
        track = view.TRACK_JS
        for asked, sent in (("asked = { index: done.index, until: performance.now() + 4000 };",
                             "trigger('click', { trim: done.index, seconds: done.value });"),
                            ("asked = { index: slot - (slot > drag ? 1 : 0), until: performance.now() + 4000 };",
                             "trigger('click', { from: drag, slot });")):
            with self.subTest(sent=sent):
                self.assertEqual((track.count(asked), track.count(sent)), (1, 1))
                between = track[track.index(asked):track.index(sent)]
                self.assertTrue(between and "trigger(" not in between and "return" not in between, between)
        # the scenes the server selects in its answers are exactly those
        scenes = tuple(timeline.Scene(f"Scene {number}.", 3 + number) for number in range(4))
        draft = Draft(None, scenes, 0)
        for index in range(4):
            self.assertEqual(ui_timeline._trimmed(draft, index, 7).selected, index)
        for source, slot in itertools.product(range(4), range(5)):
            if slot in (source, source + 1):
                continue  # the track sends no drop that changes nothing
            with self.subTest(source=source, slot=slot):
                moved = ui_timeline._scene_order(draft, "drop", (source, slot))
                self.assertEqual(moved.selected, slot - (1 if slot > source else 0))
                self.assertEqual(moved.scenes[moved.selected], scenes[source])

    def test_a_saved_project_with_a_remembered_view_is_never_taken_for_a_new_one_being_saved(self):
        """A new project that is saved gets an id nothing is remembered for and keeps its view; opening a saved
        project whose scene lengths happen to equal the new project's brings back its own view."""
        self.assertEqual(re.findall(r"const saved = ([^;]+);", view.TRACK_JS),
                         ["project === '' && id !== '' && outline === shape && !recall(id)"])

    def test_ctrl_wheel_anywhere_over_the_timeline_panel_zooms_the_timeline(self):
        """Over the toolbar and the zoom bar too (separate components above the track), Ctrl+wheel or a pinch
        zooms the timeline instead of the page; a plain wheel there is left alone."""
        track = view.TRACK_JS
        self.assertIn("const panel = () => element.closest('#timeline-timeline') || element;", track)
        self.assertEqual(track.count("addEventListener('wheel'"), 1)
        handler = self.wheel_handler()
        zoom, track_only = handler.index("event.ctrlKey || event.metaKey"), handler.index("if (!onTrack) return;")
        self.assertLess(zoom, track_only)
        self.assertIn("event.preventDefault();", handler[zoom:track_only])
        self.assertIn("const anchor = onTrack ? ", handler)  # the pointer over the track, else the playhead
        self.assertIn("if (!element.isConnected", handler)  # a track that was replaced leaves the panel alone

    def test_over_the_lane_names_a_vertical_turn_scrolls_lanes_that_do_not_fit(self):
        handler = self.wheel_handler()
        self.assertIn("if (!sideways && target.closest('.timeline-rail') && "
                      "scroll.scrollHeight > scroll.clientHeight + 1) return;", handler)
        self.assertLess(handler.index("target.closest('.timeline-rail')"), handler.index("event.preventDefault();\n    scrollView"))


class ScriptSyntaxTests(unittest.TestCase):
    def test_every_page_script_parses(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is not installed")
        scripts = {name: value for name, value in vars(view).items() if name.endswith("_JS") and isinstance(value, str)}
        scripts["_SPEAKER_JS"] = ui_timeline._SPEAKER_JS
        self.assertGreaterEqual(len(scripts), 10)
        with tempfile.TemporaryDirectory() as folder:
            for name, code in scripts.items():
                with self.subTest(script=name):
                    wrapped = (f"const handler = {code};\n" if code.lstrip().startswith("() =>") else
                               "function onLoad(element, trigger, props, server, upload, watch) {\n" + code + "\n}\n")
                    path = Path(folder) / f"{name}.js"
                    path.write_text(wrapped, encoding="utf-8")
                    result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_scripts_talk_through_the_same_events(self):
        for name in ("data", "seek", "scrub", "mode", "zoomed"):  # the track tells
            self.assertIn(f"emit('{name}'", view.TRACK_JS)
        for name in ("data", "seek", "scrub", "mode"):
            self.assertIn(f"'timeline:{name}'", view.VIEWER_JS)
        for name in ("time", "query"):  # the viewer tells
            self.assertIn(f"emit('{name}'", view.VIEWER_JS)
            self.assertIn(f"'timeline:{name}'", view.TRACK_JS)
        self.assertIn("'timeline:zoom'", view.ZOOM_JS + view.TRACK_JS)
        self.assertIn("'timeline:zoomed'", view.ZOOM_JS)
        self.assertIn("watch('value'", view.TRACK_JS)
        self.assertNotIn("watch(", view.VIEWER_JS)  # the viewer's markup never changes


if __name__ == "__main__":
    unittest.main()
