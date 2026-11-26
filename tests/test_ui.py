# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Interfejs: menu w gornym pasku ma strone dla kazdej pozycji, komunikaty i nazwy zadan
bez emoji (spokojny wyglad), cala aplikacja buduje sie bez serwera (pasek: tests/test_shell.py)."""
import tempfile
import unittest
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
from fastapi import FastAPI
from starlette.testclient import TestClient

from app import characters, llm, main, prefs, studio, style_library, ui_history, ui_queue, ui_theme, ui_video
from app.vram import GpuInfo


class NoteTests(unittest.TestCase):
    def test_leading_marks_become_status_marks(self):  # their shapes: tests/test_theme.py
        self.assertIn("st-ok", ui_theme.note("✅ Zapisano."))
        self.assertIn("st-err", ui_theme.note("❌ Wpisz prompt."))
        self.assertIn("st-err", ui_theme.note("⛔ Zablokowane."))
        self.assertIn("st-warn", ui_theme.note("⚠ Kontrola jakości: 3 próby."))
        self.assertTrue(ui_theme.note("❌ Wpisz prompt.").endswith("Wpisz prompt."))
        self.assertEqual(ui_theme.note("Zwykły tekst"), "Zwykły tekst")
        self.assertEqual(ui_theme.note(""), "")
        self.assertEqual(ui_theme.note(ui_theme.note("✅ x")), ui_theme.note("✅ x"))  # raz

    def test_model_picker_has_no_emoji(self):
        with patch.object(characters, "all_characters", return_value=[
                {"name": "Mia", "slug": "mia", "photos": ["mia.png"]}, {"name": "Ola", "slug": "ola", "photos": []}]):
            self.assertEqual(characters.choices(), [("No character", ""), ("Mia", "mia")])

    def test_every_timeline_job_kind_has_a_queue_label(self):
        from app import timeline
        for kind in timeline.JOB_KINDS:
            self.assertTrue(ui_queue.KINDS.get(kind, "").startswith("Timeline "), kind)
        self.assertEqual(len({ui_queue.KINDS[kind] for kind in timeline.JOB_KINDS}), len(timeline.JOB_KINDS))


class ShellTests(unittest.TestCase):
    def test_output_media_is_served_to_the_local_page_only(self):
        """A result file is served like any other request to the loopback host: no history lookup decides
        it. Another host name (a DNS-rebinding page) is refused."""
        app = FastAPI()

        @app.get("/gradio_api/file/{path:path}")
        @app.get("/gradio_api/file={path:path}")
        def file(path: str):
            return {"ok": True}

        app.add_middleware(main.LocalRequestGuard)
        with TestClient(app, base_url="http://127.0.0.1:7881") as client:
            for marker in ("/gradio_api/file=", "/gradio_api/file/"):
                url = marker + (main.config.OUTPUTS_DIR / "video_20261114_120000.mp4").as_posix()
                self.assertEqual(client.get(url).status_code, 200)
                self.assertEqual(client.get(url, headers={"Host": "example.com"}).status_code, 403)

    def test_foreign_origins_and_hosts_cannot_execute_ui_actions(self):
        app = FastAPI()
        calls = []

        @app.post("/action")
        def action():
            calls.append(True)
            return {"ok": True}

        app.add_middleware(main.LocalRequestGuard)
        with TestClient(app, base_url="http://127.0.0.1:7881") as client:
            for headers in ({"Origin": "https://example.com"}, {"Origin": "null"},
                            {"Host": "example.com", "Origin": "http://example.com"},
                            {"Sec-Fetch-Site": "cross-site"}):
                self.assertEqual(client.post("/action", headers=headers).status_code, 403)
            self.assertEqual(calls, [])
            self.assertEqual(client.post("/action", headers={"Origin": "http://127.0.0.1:7881"})
                             .status_code, 200)
            self.assertEqual(client.post("/action").status_code, 200)
            self.assertEqual(len(calls), 2)

    def test_launch_is_local_and_excludes_private_data(self):
        for host in ("127.0.0.1", "::1", "localhost"):
            options = main.local_launch_options({"host": host})
            self.assertEqual(options["server_name"], host)
            self.assertFalse(options["share"])
            self.assertEqual(options["allowed_paths"], [str(main.config.OUTPUTS_DIR.resolve())])
            for private in (main.config.CONFIG_DIR, main.config.MODELS_DIR, main.config.HISTORY_DIR,
                            characters.DIR, main.timeline.DIR):
                self.assertIn(str(private.resolve()), options["blocked_paths"])
            self.assertTrue(set(options).issubset(inspect.signature(gr.Blocks.launch).parameters))
        for host in ("0.0.0.0", "::", "192.168.1.10", "example.com"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                main.local_launch_options({"host": host})

    def test_no_output_file_or_folder_is_blocked(self):
        """Every file in outputs/ may be served, earlier results included: nothing inside it is on the
        blocked list and launching changes no file there."""
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp).resolve()
            (outputs / "older").mkdir()
            files = [outputs / "new.png", outputs / "older" / "kept.mp4"]
            for file in files:
                file.write_bytes(b"preserve")
            with patch.object(main.config, "OUTPUTS_DIR", outputs):
                blocked = main.local_launch_options({"host": "127.0.0.1"})["blocked_paths"]
            self.assertFalse([path for path in blocked if Path(path).resolve().is_relative_to(outputs)])
            self.assertTrue(all(file.read_bytes() == b"preserve" for file in files))

    def test_adopted_character_library_caches_only_previews(self):
        from app import ui_characters, voices

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "adopted"
            library, imported = data / "characters", data / "voices"
            folder = library / "sample"
            training = folder / "photos" / "02.png"  # another reference photo: never served
            training.parent.mkdir(parents=True)
            training.write_bytes(b"private training photo")
            preview = folder / "photos" / "01.png"
            preview.write_bytes(b"preview")
            cache = Path(tmp) / "cache"
            character = {"slug": "sample", "photos": [str(preview), str(training)]}
            with (patch.object(characters, "DIR", library), patch.object(voices, "USER_DIR", imported),
                  patch.object(ui_theme, "get_upload_folder", return_value=str(cache))):
                options = main.local_launch_options({"host": "127.0.0.1"})
                self.assertIn(str(library.resolve()), options["blocked_paths"])
                self.assertIn(str(imported.resolve()), options["blocked_paths"])
                cached = Path(ui_characters._cover(character))
                self.assertTrue(cached.is_relative_to(cache))
                self.assertEqual(cached.read_bytes(), preview.read_bytes())

                self.assertEqual(ui_characters._cover({"slug": "sample", "photos": []}), "")
                self.assertEqual(training.read_bytes(), b"private training photo")

    def test_whole_app_builds_and_tabs_match_menu(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(prefs, "_FILE", Path(tmp) / "prefs.json"), \
                patch.object(style_library, "all_custom", return_value=[]), \
                patch.object(characters, "all_characters", return_value=[]), \
                patch("app.items.all_items", return_value=[]), \
                patch.object(studio, "results", return_value=[]), \
                patch("app.ui_settings.model_licenses_html", return_value=""), \
                patch.object(main, "get_gpu_info",
                             return_value=GpuInfo(True, "NVIDIA GeForce RTX 4070", total_gb=16.0,
                                                  free_gb=9.5)):
            demo = main.build_ui()
            # Every list the page refreshes on load keeps the value it starts with: Gradio refuses an input whose
            # value is not among its choices (the Editor's Style once lost Realistic, so no project could be opened).
            checked = 0
            for fn in demo.fns.values():
                # only the page-load refreshes: a click handler without inputs may pause the queue and the like
                if (fn.inputs or not fn.outputs or fn.fn is None or not fn.targets
                        or any(event != "load" for _, event in fn.targets)):
                    continue
                result = fn.fn()
                updates = result if isinstance(result, tuple) else (result,)
                for block, update in zip(fn.outputs, updates):
                    if (isinstance(block, gr.Dropdown) and isinstance(update, dict) and "choices" in update
                            and "value" not in update and block.value not in (None, "", [])):
                        values = {c[1] if isinstance(c, (tuple, list)) else c for c in update["choices"]}
                        starting = block.value if isinstance(block.value, list) else [block.value]
                        self.assertLessEqual(set(starting), values, block.elem_id or block.label)
                        checked += 1
            self.assertGreater(checked, 0)
        # top-level pages only: a page may hold its own tabs (Voices: Women / Men)
        tabs = {b.id for b in demo.blocks.values() if isinstance(b, gr.Tab)
                and getattr(getattr(b, "parent", None), "elem_id", None) == "pages"}
        self.assertEqual(tabs, {page for _, page in main.WORKSPACE + main.TOOLS})
        pages = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "pages")
        self.assertEqual(pages.selected, "stories")
        nav = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "workspace-nav")
        self.assertIn('data-page="stories" aria-current="page"', nav.value)
        self.assertIn('data-page="characters"', nav.value)
        self.assertNotIn("train", nav.value)  # the page of the removed character training has its own id
        self.assertIn('data-page="series"', nav.value)
        # Contsy Stories: the general studio's pages are gone from the menus
        for gone in ("universes", "styles", "gallery", "video", "image", "home"):
            self.assertNotIn(f'data-page="{gone}"', nav.value)
        self.assertIn('aria-controls="topbar-tools"', nav.value)  # the Tools button
        tools = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "topbar-tools")
        self.assertIsInstance(tools, gr.Column)  # rendered at once, unlike a closed accordion
        self.assertTrue(any(isinstance(child, gr.HTML) and 'data-page="agent"' in child.value
                            for child in tools.children))
        self.assertFalse(any(isinstance(child, gr.HTML) and 'data-page="video"' in child.value
                             for child in tools.children))
        self.assertFalse(any(isinstance(b, gr.Sidebar) for b in demo.blocks.values()))
        self.assertEqual(demo.title, "Contsy Stories")  # karta przegladarki
        self.assertIn(">Contsy Stories<", ui_theme.BRAND)  # logo w gornym pasku
        self.assertIn(ui_theme.BRAND, nav.value)
        text = str(demo.get_config_file())
        for unwanted in ("Zaznaczanie wielu",
                         "AI Series", "Autopilot", "Content Packs", "Pack from selection",
                         "Seria z polecenia", "Polecenie dla AI", "Przerwij bieżące"):
            self.assertNotIn(unwanted, text)


class GallerySafetyTests(unittest.TestCase):
    def test_recycle_failure_preserves_file_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.write_bytes(b"keep me")
            with patch.object(ui_history.config, "OUTPUTS_DIR", Path(tmp)), \
                    patch.object(ui_history.studio, "is_recorded", return_value=True), \
                    patch.object(ui_history.trash, "move", return_value=False), \
                    patch.object(ui_history.history, "forget") as forget:
                self.assertEqual(ui_history.recycle_selected([str(path)]), ([], [str(path)]))
                self.assertEqual(path.read_bytes(), b"keep me")
                forget.assert_not_called()

    def test_recycle_partial_failure_forgets_only_successful_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ("ok.png", "keep.png")]
            for path in paths:
                path.touch()
            with patch.object(ui_history.config, "OUTPUTS_DIR", Path(tmp)), \
                    patch.object(ui_history.studio, "is_recorded", return_value=True), \
                    patch.object(ui_history.trash, "move", side_effect=[True, False]), \
                    patch.object(ui_history.history, "forget") as forget:
                self.assertEqual(ui_history.recycle_selected([str(p) for p in paths]),
                                 ([str(paths[0])], [str(paths[1])]))
                forget.assert_called_once_with([str(paths[0])])

    def test_gallery_rejects_outside_files_and_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            outside = Path(tmp) / "private.txt"
            outside.touch()
            self.assertEqual(ui_history._safe([str(root), str(outside)]), [])

    def test_gallery_rejects_selections_the_history_does_not_list(self):
        # The Gallery shows recorded results only: a running job's work file is never selected.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video_20261114_part1.mp4"
            path.touch()
            with patch.object(ui_history.config, "OUTPUTS_DIR", Path(tmp)), \
                    patch.object(ui_history.studio, "is_recorded", return_value=False):
                self.assertEqual(ui_history._safe([str(path)]), [])


class GenerationTests(unittest.TestCase):
    def test_queue_poller_shows_a_finished_result_and_keeps_the_previous_one_for_a_deleted_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.touch()
            job = ui_queue.qm.Job(123, "obraz", "Image 1:1", {}, status=ui_queue.qm.DONE, files=[str(path)])
            with patch.object(ui_queue.qm, "latest_finished", return_value=job):
                result, message, last_id = ui_queue.result_poller("obraz")(None)
                self.assertEqual((result, last_id), ([str(path)], 123))
                path.unlink()
                result, message, last_id = ui_queue.result_poller("obraz")(None)
            self.assertEqual((result, last_id), (gr.update(), 123))
            self.assertEqual(message, ui_queue.finished_message(job))

    def test_a_start_image_goes_straight_to_generation(self):
        # No review of the start image: the assistant is not loaded before the clip is made.
        out = SimpleNamespace(ok=True, files=["clip.mp4"], notes=["Done."], error="")
        with patch.object(llm, "ensure", side_effect=AssertionError("The assistant was loaded")), \
                patch.object(ui_video.cast_mod, "for_job", return_value=([], "")), \
                patch.object(ui_video.video_gen, "generate", return_value=out) as generate, \
                patch.object(ui_video.prefs, "autosave", return_value=""):
            ok, files, notes, error = ui_video.run_job({"prompt": "an adult walking", "image": "start.png",
                                                        "seconds": 5, "aspect": "Portrait 9:16"}, lambda *_: None)
        self.assertEqual((ok, files, notes, error), (True, ["clip.mp4"], ["Done."], ""))
        self.assertEqual(generate.call_args.kwargs["image_path"], "start.png")

    def test_a_video_job_queued_with_a_mode_is_made_in_the_one_mode(self):
        # Jobs queued before 2026-11-14 carry their mode ("speed"); the one mode makes them all.
        out = SimpleNamespace(ok=True, files=["clip.mp4"], notes=[], error="")
        for speed in ("quality", "fast", "preview"):
            with patch.object(ui_video.cast_mod, "for_job", return_value=([], "")), \
                    patch.object(ui_video.video_gen, "generate", return_value=out) as generate, \
                    patch.object(ui_video.prefs, "autosave", return_value=""):
                ok, *_ = ui_video.run_job({"prompt": "rain on a window", "seconds": 5, "aspect": "Landscape 16:9",
                                           "speed": speed}, lambda *_: None)
            self.assertTrue(ok, speed)
            self.assertNotIn("speed", generate.call_args.kwargs)

    def test_failed_engine_video_is_not_autosaved(self):
        out = SimpleNamespace(ok=False, files=[], notes=[], error="Wan could not finish the clip.")
        with patch.object(ui_video.cast_mod, "for_job", return_value=([], "")), \
                patch.object(ui_video.video_gen, "generate", return_value=out), \
                patch.object(ui_video.prefs, "autosave") as autosave:
            ok, files, _, error = ui_video.run_job({"prompt": "an adult walking", "seconds": 5,
                                                   "aspect": "Portrait 9:16"}, lambda *_: None)
        self.assertFalse(ok)
        self.assertEqual(files, [])
        self.assertEqual(error, out.error)
        autosave.assert_not_called()


if __name__ == "__main__":
    unittest.main()
