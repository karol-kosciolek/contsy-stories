# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Small robustness fixes (Windows device names as slugs, the parsed history, a preferences file that is being
replaced, a cut-short brief file) and the launch options of the local studio (upload cap, no telemetry, private
folders never served). Scratch folders only."""
import inspect
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import agent_store, characters, config, history, items, main, music_library, prefs, series, voices


class SmallFixTests(unittest.TestCase):
    def test_reserved_windows_names_are_not_slugs(self):
        self.assertEqual(characters.slugify("CON"), "con_")
        self.assertEqual(items.slugify("lpt1"), "lpt1_")
        self.assertEqual(voices.slugify("Nul"), "nul_")
        self.assertEqual(characters.slugify("Anna"), "anna")

    def test_history_is_parsed_once_per_change_and_keeps_odd_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.jsonl"
            with patch.object(history, "_HISTORY_FILE", path), patch.object(config, "HISTORY_DIR", Path(folder)):
                history.record({"files": ["a.png"], "odd": Path("x")})
                with patch.object(Path, "read_text", wraps=path.read_text) as reads:
                    first = history.recent(10)
                    history.recent(10)
                self.assertEqual(reads.call_count, 1)  # the second call is served from the cache
                self.assertEqual(first[0]["odd"], "x")
                first[0]["odd"] = "changed"  # recent() hands out copies, so the cache stays intact
                self.assertEqual(history.recent(10)[0]["odd"], "x")
                history.record({"files": ["b.png"]})
                self.assertEqual([e["files"] for e in history.recent(10)], [["b.png"], ["a.png"]])

    def test_preferences_survive_a_moment_of_permission_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prefs.json"
            path.write_text(json.dumps({"qc": False}), encoding="utf-8")
            calls = []
            real = Path.read_text

            def flaky(self, *args, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    raise PermissionError("being replaced")
                return real(self, *args, **kwargs)

            with patch.object(prefs, "_FILE", path), patch.object(Path, "read_text", flaky):
                self.assertFalse(prefs.load()["qc"])

    def test_a_cut_short_brief_is_written_again(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(agent_store, "RUNS_DIR", Path(folder)):
            store = agent_store.RunStore.__new__(agent_store.RunStore)
            store.path = lambda *parts: Path(folder).joinpath(*parts)
            (Path(folder) / "brief").mkdir()
            data = b"brief text"
            name = store.put_brief(data, "txt")
            (Path(folder) / "brief" / name).write_bytes(b"brief")  # truncated
            store.put_brief(data, "txt")
            self.assertEqual((Path(folder) / "brief" / name).read_bytes(), data)


class LaunchOptionTests(unittest.TestCase):
    def options(self) -> dict:
        return main.local_launch_options({"host": "127.0.0.1"})

    def test_every_launch_option_is_a_launch_parameter(self):
        import gradio as gr
        accepted = set(inspect.signature(gr.Blocks.launch).parameters)
        options = self.options()
        self.assertEqual(set(options) - accepted, set())

    def test_uploads_are_capped(self):
        self.assertEqual(main.MAX_UPLOAD, "200mb")
        self.assertEqual(self.options()["max_file_size"], main.MAX_UPLOAD)

    def test_gradio_analytics_are_off_before_gradio_is_imported(self):
        source = Path(main.__file__).read_text(encoding="utf-8")
        setting = re.search(r"""os\.environ\.setdefault\(\s*["']GRADIO_ANALYTICS_ENABLED["']\s*,\s*["']False["']""",
                            source)
        gradio = re.search(r"^[ \t]*(?:import gradio|from gradio\b)", source, re.M)
        self.assertIsNotNone(setting)
        self.assertIsNotNone(gradio)
        self.assertLess(setting.start(), gradio.start())
        self.assertIn("analytics_enabled=False", inspect.getsource(main.build_ui))

    def test_series_and_music_folders_are_never_served(self):
        blocked = self.options()["blocked_paths"]
        self.assertIn(str(Path(series.DIR).resolve()), blocked)
        self.assertIn(str(Path(music_library.folder()).resolve()), blocked)


if __name__ == "__main__":
    unittest.main()
