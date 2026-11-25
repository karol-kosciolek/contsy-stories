# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Styles page (app/ui_styles.py): the built-in table, making, renaming, editing, translating and deleting the
owner's styles through the page's handlers, and the page built in gr.Blocks. The style library is a temporary folder
and the prompt assistant is replaced."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from app import characters, llm, style_library, styles, ui_styles

SKIP = gr.skip()


def quiet(*_args, **_kwargs):
    """A stand-in for gr.Progress."""


class StylesPageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for patcher in (patch.object(characters, "DIR", Path(self.tmp.name) / "characters"),
                        patch.dict(style_library._CACHE, clear=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_the_built_in_table_shows_every_starting_point_and_how_it_is_made(self):
        self.assertEqual(set(ui_styles._MADE_WITH), set(styles.BASES))
        html = ui_styles.builtin_table()
        for style in styles.BASES.values():
            self.assertIn(f"<b>{style.label}</b>", html)
            self.assertIn(ui_styles._MADE_WITH[style.id].split(";")[0].replace("'", "&#x27;"), html)
        self.assertIn("Only as the start of a style of your own", html)
        self.assertIn("Japanese / Asian anime", html)

    def test_a_style_is_made_renamed_edited_and_deleted_on_the_page(self):
        table, picker, name, base, look = ui_styles.views("")
        self.assertIn("No styles of your own yet", table)
        self.assertEqual((picker["value"], picker["choices"], name, base, look),
                         ("", [(style_library.NEW_STYLE, "")], "", styles.OTHER, ""))
        status, table, picker, name, base, look, loaded = ui_styles.save_style("", "Story <b>", styles.ANIME,
                                                                               "soft pastel watercolor")
        self.assertIn("saved", status)
        self.assertNotIn("Tip:", status)
        self.assertIn("<b>Story &lt;b&gt;</b>", table)
        self.assertIn("<td>Anime</td>", table)
        [entry] = style_library.all_custom()
        self.assertEqual((picker["value"], name, base, look, loaded),
                         (entry["key"], "Story <b>", styles.ANIME, "soft pastel watercolor", entry["key"]))
        self.assertIn(("Story <b>", entry["key"]), picker["choices"])
        self.assertEqual(ui_styles.views(entry["key"])[2:], ("Story <b>", styles.ANIME, "soft pastel watercolor"))
        status = ui_styles.save_style(entry["key"], "Storybook", styles.ANIME, "soft pastel watercolor")[0]
        self.assertIn("Nothing needs to be made again", status)
        status = ui_styles.save_style(entry["key"], "Storybook", styles.ANIME, "bright gouache")[0]
        self.assertIn("new version", status)
        refused = ui_styles.save_style("", "Ink", styles.OTHER, "chibi, big heads")
        self.assertIn("every character is an adult", refused[0])
        self.assertTrue(all(value == SKIP for value in refused[1:]))  # the form keeps its words
        self.assertIn("Enter a name", ui_styles.save_style("", "", styles.OTHER, "ink wash")[0])
        status = ui_styles.save_style("", "Akwarela", styles.OTHER, "miękkie pastelowe światło, akwarela")[0]
        self.assertIn("Translate to English", status)  # saved, with a tip
        self.assertEqual(len(style_library.all_custom()), 2)
        self.assertEqual(ui_styles.delete_style(entry["key"], "no")[0], "Nothing deleted.")
        status, table, picker, name, base, look, loaded = ui_styles.delete_style(entry["key"], "yes")
        self.assertIn("keep their look", status)
        self.assertNotIn("Storybook", table)
        self.assertEqual((picker["value"], name, look, loaded), ("", "", "", ""))
        self.assertIn("Choose one of your styles", ui_styles.delete_style(entry["key"], "yes")[0])  # gone already
        self.assertIn("Choose one of your styles", ui_styles.delete_style("", "yes")[0])

    def test_the_picker_fills_the_form_only_when_another_style_is_picked(self):
        entry, _ = style_library.save("Storybook", styles.ANIME, "soft pastel watercolor")
        picker, name, base, look, loaded = ui_styles.pick_style(entry["key"], "")
        self.assertEqual((picker["value"], name, base, look, loaded),
                         (entry["key"], "Storybook", styles.ANIME, "soft pastel watercolor", entry["key"]))
        # Gradio sends the picker's input event on every blur: the style in the form keeps what was typed
        self.assertTrue(all(value == SKIP for value in ui_styles.pick_style(entry["key"], entry["key"])))
        self.assertTrue(all(value == SKIP for value in ui_styles.pick_style("", "")))
        picker, name, base, look, loaded = ui_styles.pick_style("", entry["key"])  # New style picked
        self.assertEqual((picker["value"], name, base, look, loaded), ("", "", styles.OTHER, "", ""))
        self.assertEqual(ui_styles.pick_style("typed text", entry["key"])[4], "")  # no such style: a new one

    def test_translate_puts_english_words_into_the_look(self):
        with patch.object(llm, "style_words", return_value=(True, "soft pastel watercolor, paper texture")):
            look, status = ui_styles.translate_look("miękka akwarela", progress=quiet)
        self.assertEqual(look, "soft pastel watercolor, paper texture")
        self.assertIn("In English now", status)
        with patch.object(llm, "style_words", return_value=(False, llm.STYLE_MINOR_MSG)):
            look, status = ui_styles.translate_look("styl chibi", progress=quiet)
        self.assertEqual(look, SKIP)
        self.assertIn("every character is an adult", status)
        self.assertNotIn("⛔", status)

    def test_the_page_builds_with_private_events_and_lists_the_saved_styles(self):
        style_library.save("Storybook", styles.OTHER, "soft pastel watercolor")
        with gr.Blocks() as demo:
            with gr.Tab("Styles", id="styles") as tab:
                ui_styles.build_styles_tab(tab)
        by_id = {block.elem_id: block for block in demo.blocks.values() if getattr(block, "elem_id", None)}
        self.assertIn("Storybook", by_id["style-list"].value)
        self.assertEqual(by_id["style-base"].value, styles.OTHER)
        self.assertEqual([value for _, value in by_id["style-base"].choices], [*styles.STYLES, styles.OTHER])
        self.assertEqual(by_id["style-pick"].value, "")
        self.assertEqual(by_id["style-look"].max_length, styles.MAX_LOOK)
        handlers = {ui_styles.save_style, ui_styles.translate_look, ui_styles.delete_style, ui_styles.pick_style}
        found = [fn for fn in demo.fns.values() if fn.fn in handlers]
        self.assertEqual({fn.fn for fn in found}, handlers)
        self.assertTrue(all(fn.api_visibility == "private" for fn in demo.fns.values()))


if __name__ == "__main__":
    unittest.main()
