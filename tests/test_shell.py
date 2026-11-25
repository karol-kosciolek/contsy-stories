# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The app shell: one top bar above every page instead of a sidebar - the brand, the workspace menu and a
Tools menu (the other pages and the character choice) on the left, the engine status and the GPU pause on
the right - with every handler still wired, and the pages right below it. CPU only, no server; the page
script is checked with `node --check` when Node.js is installed."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr

from app import characters, main, model_manager, prefs, queue_manager, studio, style_library, ui_theme
from app.vram import GpuInfo

GPU = GpuInfo(True, "NVIDIA GeForce RTX 4070 Ti SUPER", total_gb=16.0, free_gb=13.6)


class _Elements(HTMLParser):
    """Every element (tag, attributes) and the text inside each element whose class is asked for."""

    def __init__(self, collect: str = ""):
        super().__init__()
        self.items, self.collect, self.text, self._open = [], collect, [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.items.append((tag, attrs))
        self._open.append(self.collect and self.collect in (attrs.get("class") or "").split())

    def handle_endtag(self, tag):
        if self._open:
            self._open.pop()

    def handle_data(self, data):
        if any(self._open):
            self.text.append(data)


def parse(markup: str, collect: str = "") -> _Elements:
    parser = _Elements(collect)
    parser.feed(markup)
    return parser


def _declarations(css: str, selector: str) -> dict:
    """The declarations of every top-level rule whose selector list holds `selector`, merged in order."""
    found = {}
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if selector in (" ".join(part.split()) for part in selectors.split(",")):
            found.update((p.strip(), v.strip()) for p, _, v in (d.partition(":") for d in body.split(";")) if v.strip())
    return found


def _media(css: str, query: str) -> str:
    match = re.search(re.escape(f"@media ({query})") + r"\s*\{((?:[^{}]*\{[^{}]*\})*[^{}]*)\}", css)
    assert match, query
    return match.group(1)


class _Built(unittest.TestCase):
    """The whole app, built once without a server (the GPU, the library and the settings mocked)."""

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(prefs, "_FILE", Path(tmp) / "prefs.json"), \
                patch.object(model_manager, "VERIFIED_FILE", Path(tmp) / "verified_models.json"), \
                patch.object(model_manager, "_STORED", None), \
                patch.object(style_library, "all_custom", return_value=[]), \
                patch.object(characters, "all_characters", return_value=[]), \
                patch("app.items.all_items", return_value=[]), \
                patch.object(studio, "results", return_value=[]), \
                patch("app.ui_settings.model_licenses_html", return_value=""), \
                patch.object(main, "get_gpu_info", return_value=GPU):
            cls.demo = main.build_ui()
        cls.blocks = list(cls.demo.blocks.values())
        cls.by_id = {block.elem_id: block for block in cls.blocks if getattr(block, "elem_id", None)}

    def inside(self, block, ancestor: str) -> bool:
        while block is not None:
            if getattr(block, "elem_id", None) == ancestor:
                return True
            block = getattr(block, "parent", None)
        return False

    def topbar_html(self) -> list[gr.HTML]:
        return [block for block in self.blocks if isinstance(block, gr.HTML) and self.inside(block, "topbar")]


class TopBarTests(_Built):
    def test_there_is_no_sidebar(self):
        self.assertFalse(any(isinstance(block, gr.Sidebar) for block in self.blocks))
        config = self.demo.get_config_file()
        self.assertNotIn("sidebar", {component["type"] for component in config["components"]})
        self.assertFalse(any("sidebar" in (getattr(block, "elem_id", None) or "") for block in self.blocks))

    def test_the_top_bar_comes_first_and_holds_the_whole_shell(self):
        config = self.demo.get_config_file()
        ids = {component["id"]: component["props"].get("elem_id") for component in config["components"]}
        shown = [ids[child["id"]] for child in config["layout"]["children"] if ids.get(child["id"])]
        self.assertEqual(shown, ["topbar", "pages"])  # timers and states have no markup
        self.assertIsInstance(self.by_id["topbar"], gr.Row)
        for name in ("workspace-nav", main.TOOLS_PANEL, "tools-nav", "character", "engine-status", "gpu-pause"):
            with self.subTest(part=name):
                self.assertTrue(self.inside(self.by_id[name], "topbar"))
        self.assertTrue(self.inside(self.by_id["character"], main.TOOLS_PANEL))
        self.assertTrue(self.inside(self.by_id["tools-nav"], main.TOOLS_PANEL))
        # a column, not an accordion: Gradio renders a closed accordion's content only once it is opened
        self.assertIsInstance(self.by_id[main.TOOLS_PANEL], gr.Column)
        self.assertIn(ui_theme.BRAND, self.by_id["workspace-nav"].value)

    def test_the_top_bar_has_one_button_for_every_page(self):
        buttons = [attrs for block in self.topbar_html() for tag, attrs in parse(block.value).items
                   if tag == "button" and "data-page" in attrs]
        pages = [attrs["data-page"] for attrs in buttons]
        self.assertCountEqual(pages, [page for _, page in main.WORKSPACE + main.TOOLS])
        self.assertEqual(len(pages), len(set(pages)))
        self.assertTrue(all(attrs.get("type") == "button" for attrs in buttons))
        current = [attrs["data-page"] for attrs in buttons if attrs.get("aria-current") == "page"]
        self.assertEqual(current, [main.START_PAGE])
        tabs = {block.id for block in self.blocks if isinstance(block, gr.Tab)
                and getattr(getattr(block, "parent", None), "elem_id", None) == "pages"}
        self.assertEqual(set(pages), tabs)  # every button opens a page and every page has a button
        across = re.findall(r'data-page="([^"]+)"', self.by_id["workspace-nav"].value)
        self.assertEqual(across, [page for _, page in main.WORKSPACE])
        listed = re.findall(r'data-page="([^"]+)"', self.by_id["tools-nav"].value)
        self.assertEqual(listed, [page for _, page in main.TOOLS])

    def test_the_tools_button_discloses_its_panel(self):
        toggles = [attrs for tag, attrs in parse(self.by_id["workspace-nav"].value).items
                   if tag == "button" and "data-tools-toggle" in attrs]
        self.assertEqual(len(toggles), 1)
        toggle = toggles[0]
        self.assertEqual((toggle["type"], toggle["aria-expanded"], toggle["aria-controls"]),
                         ("button", "false", main.TOOLS_PANEL))
        self.assertIn(">Tools<", self.by_id["workspace-nav"].value)
        # the page script and the stylesheet open the same panel
        self.assertIn(f"getElementById('{main.TOOLS_PANEL}')", ui_theme.JS)
        self.assertIn(f"#topbar > #{main.TOOLS_PANEL}[data-open]", ui_theme.CSS)
        self.assertEqual(_declarations(ui_theme.CSS, f"#topbar > #{main.TOOLS_PANEL}")["display"], "none !important")

    def test_status_pause_and_the_character_choice_stay_wired(self):
        demo, ids = self.demo, {name: block._id for name, block in self.by_id.items()}
        status, pause, character = ids["engine-status"], ids["gpu-pause"], ids["character"]

        def listeners(event, target=None):
            for fn in demo.fns.values():
                for block, name in fn.targets:
                    if name == event and (target is None or block == target):
                        yield fn
        outputs = lambda fn: [getattr(block, "_id", block) for block in fn.outputs]
        inputs = lambda fn: [getattr(block, "_id", block) for block in fn.inputs]
        ticks = [(demo.blocks[block].value, outputs(fn)) for fn in listeners("tick")
                 for block, name in fn.targets if name == "tick"]
        self.assertIn((10.0, [status, pause]), ticks)  # the engine status and the pause label every 10 s
        loads = list(listeners("load"))
        self.assertIn([status, pause], [outputs(fn) for fn in loads])
        clicks = list(listeners("click", pause))
        self.assertEqual([(fn.fn.__name__, outputs(fn)) for fn in clicks], [("_toggle_gpu", [pause, status])])
        self.assertIn([character], [outputs(fn) for fn in loads])  # fresh choices on every page load
        self.assertTrue(any(inputs(fn) == [character] and not outputs(fn) for fn in listeners("input", character)))
        self.assertIn(6.0, [seconds for seconds, written in ticks if written[:1] == [character]])  # saved elsewhere
        # Contsy Stories: the older pages' character choice is kept for the Characters page but not shown
        self.assertFalse(self.by_id["character"].visible)

    def test_pause_toggles_and_returns_the_new_label_and_status(self):
        toggle = next(fn for fn in self.demo.fns.values() if getattr(fn.fn, "__name__", "") == "_toggle_gpu").fn
        with patch.object(gr, "Info") as told, patch.object(main, "get_gpu_info", return_value=GPU), \
                patch.object(queue_manager, "paused", return_value=False), \
                patch.object(queue_manager, "pause_now", return_value=7) as pause_now, \
                patch.object(queue_manager, "snapshot", return_value=[]):
            label, chip = toggle()
        pause_now.assert_called_once()
        self.assertIn("Job #7", told.call_args.args[0])
        self.assertEqual(label, "Pause and release GPU")  # queue_manager.paused() is mocked as False here
        self.assertIn("engine-chip", chip)
        with patch.object(gr, "Info"), patch.object(main, "get_gpu_info", return_value=GPU), \
                patch.object(queue_manager, "paused", return_value=True), \
                patch.object(queue_manager, "set_paused") as resume, \
                patch.object(queue_manager, "snapshot", return_value=[]):
            toggle()
        resume.assert_called_once_with(False)


class StatusChipTests(unittest.TestCase):
    def chip(self, *, paused=False, up=True, gpu=GPU, jobs=(), seconds=0):
        with patch.object(main.engine_runtime.Engine, "available", return_value=up), \
                patch.object(queue_manager, "paused", return_value=paused), \
                patch.object(main, "get_gpu_info", return_value=gpu), \
                patch.object(queue_manager, "snapshot", return_value=[SimpleNamespace(status=s) for s in jobs]), \
                patch.object(queue_manager, "remaining_seconds", return_value=seconds):
            self.markup = main.status_html()
        page = parse(self.markup, "engine-text")
        chip = next(attrs for tag, attrs in page.items if "engine-chip" in (attrs.get("class") or "").split())
        dot = next(attrs for tag, attrs in page.items if "dot" in (attrs.get("class") or "").split())
        hidden = parse(self.markup, "engine-sr").text
        return chip, dot, "".join(page.text), "".join(hidden)

    def test_one_short_line_and_every_detail_in_the_tooltip(self):
        chip, dot, line, spoken = self.chip(jobs=(queue_manager.RUNNING, queue_manager.DONE), seconds=540)
        self.assertEqual(line, "Engine ready · 13.6 / 16 GB free · Queue 1 · about 9 min")
        self.assertEqual(chip["title"], "Engine ready\nRTX 4070 Ti SUPER · 13.6 of 16 GB free\nQueue 1 · about 9 min")
        self.assertEqual(spoken, "Engine ready. RTX 4070 Ti SUPER · 13.6 of 16 GB free. Queue 1 · about 9 min")
        self.assertEqual((chip["data-state"], dot["class"].split()), ("ready", ["dot"]))
        self.assertEqual(dot["aria-hidden"], "true")

    def test_the_state_is_told_by_shape_not_color(self):
        chip, dot, line, _ = self.chip(paused=True)
        self.assertEqual((chip["data-state"], dot["class"].split()), ("pause", ["dot", "pause"]))
        self.assertTrue(line.startswith("Paused — GPU released · "))
        self.assertTrue(line.endswith("Queue empty"))
        chip, dot, line, _ = self.chip(up=False, gpu=GpuInfo(False))
        self.assertEqual((chip["data-state"], dot["class"].split()), ("off", ["dot", "off"]))
        self.assertEqual(line, "Engine unavailable · No GPU · Queue empty")
        self.assertIn("No NVIDIA GPU detected", chip["title"])

    def test_the_assistant_does_not_change_the_engine_state(self):
        # The assistant loads only when a feature asks for it: a missing assistant runtime is that
        # feature's own message, the engine chip stays "ready" and never asks about it.
        with patch.object(main.llm, "ensure", side_effect=AssertionError("Not asked")), \
                patch.object(main.llm, "runtime_problem", side_effect=AssertionError("Not asked")):
            chip, dot, line, _ = self.chip()
        self.assertEqual((chip["data-state"], dot["class"].split()), ("ready", ["dot"]))
        self.assertTrue(line.startswith("Engine ready · "))

    def test_details_are_escaped_in_the_markup(self):
        name = 'RTX <b>"x"</b>'
        chip, dot, line, spoken = self.chip(gpu=GpuInfo(True, f"NVIDIA {name}", total_gb=16.0, free_gb=13.6))
        self.assertNotIn("RTX", line)  # the short line stays short
        self.assertIn(name, chip["title"])  # the parser unescapes it: the markup has it escaped
        self.assertIn(name, spoken)
        self.assertNotIn('<b>"x"', self.markup)
        self.assertIn("&lt;b&gt;&quot;x&quot;&lt;/b&gt;", self.markup)


class ShellStyleTests(unittest.TestCase):
    def test_the_top_bar_is_a_44_px_strip_across_the_window(self):
        bar = _declarations(ui_theme.CSS, "#topbar")
        self.assertEqual(bar["min-height"], "44px")
        self.assertEqual((bar["position"], bar["top"]), ("sticky", "0"))
        self.assertEqual(bar["border-bottom"], "1px solid var(--c-border) !important")
        self.assertEqual(bar["flex-wrap"], "nowrap !important")
        children = _declarations(ui_theme.CSS, "#topbar > *")
        self.assertEqual((children["flex"], children["min-width"]), ("0 0 auto !important", "0 !important"))
        # Gradio's main column: no cap and no padding; the pages start right under the bar
        self.assertEqual(_declarations(ui_theme.CSS, ".gradio-container .main")["max-width"], "none !important")
        self.assertEqual(_declarations(ui_theme.CSS, ".gradio-container .main > .wrap")["padding"], "0 !important")
        self.assertEqual(_declarations(ui_theme.CSS, ".gradio-container .contain .column:has(> #topbar)")["gap"],
                         "0 !important")
        self.assertEqual(_declarations(ui_theme.CSS, ".gradio-container")["overflow"], "clip !important")

    def test_narrow_windows_wrap_the_bar_instead_of_scrolling_sideways(self):
        self.assertIn("#topbar .brand .b-name { display: none; }", _media(ui_theme.CSS, "max-width: 1100px"))
        phone = _media(ui_theme.CSS, "max-width: 720px")
        self.assertIn("#topbar { flex-wrap: wrap !important;", phone)
        self.assertIn("#topbar > #engine-status { flex: 1 1 0 !important;", phone)
        chip = _declarations(ui_theme.CSS, ".engine-chip .engine-text")
        self.assertEqual((chip["min-width"], chip["text-overflow"]), ("0", "ellipsis"))
        self.assertEqual(_declarations(ui_theme.CSS, ".topbar-start")["flex-wrap"], "wrap")
        self.assertEqual(_declarations(ui_theme.CSS, ".workspace-nav")["flex-wrap"], "wrap")

    def test_the_other_pages_keep_a_readable_width_and_the_timeline_reaches_the_edges(self):
        pages = _declarations(ui_theme.CSS, "#pages")
        self.assertEqual(pages["padding"], "28px clamp(16px, 4vw, 40px) 48px !important")
        self.assertEqual(_declarations(ui_theme.CSS, "#pages > .tabitem")["max-width"], "1240px")
        timeline = '#pages:has(> #timeline-page:not([style*="display: none"]))'
        self.assertEqual(_declarations(ui_theme.CSS, timeline)["padding"], "0 !important")
        self.assertEqual(_declarations(ui_theme.CSS, "#pages > #timeline-page")["max-width"], "none")

    def test_gradios_own_page_tabs_are_out_of_the_tab_order(self):
        """The menus switch pages by clicking Gradio's tab buttons, which sit off screen. They stay laid out
        (Gradio measures them, and a click on a hidden button still selects its page) but a keyboard user never
        tabs through them after the top bar, and screen readers do not announce a second list of pages."""
        tabs = _declarations(ui_theme.CSS, "#pages > .tab-wrapper")
        self.assertEqual((tabs["visibility"], tabs["left"]), ("hidden", "-10000px"))
        self.assertNotIn("display", tabs)
        self.assertIn("if (tab.getAttribute('aria-selected') !== 'true') tab.click();", ui_theme.JS)

    def test_no_sidebar_rules_or_scripts_are_left(self):
        for text in (ui_theme.CSS, ui_theme.JS):
            for gone in ("#sidebar", ".sidebar", "sidebar-tools", "sb-foot", "sb-status", "--overlap-amount",
                         "closeOnPhone", "toggle-button"):
                self.assertNotIn(gone, text)

    def test_the_tools_script_handles_the_keyboard_and_clicks_elsewhere(self):
        js = ui_theme.JS
        for part in ("aria-expanded", "'Escape'", "'pointerdown'", "'focusin'", "toggleAttribute('data-open'",
                     "toggleAttribute('data-current'", "--topbar-height", "focus({preventScroll: true})"):
            self.assertIn(part, js)
        # Escape closes an open list (the character dropdown's) before the panel
        self.assertIn('[aria-expanded="true"]:not([data-tools-toggle])', js)

    def test_the_page_script_parses(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is not installed")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shell.js"
            path.write_text(f"const handler = {ui_theme.JS};\n", encoding="utf-8")
            result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
