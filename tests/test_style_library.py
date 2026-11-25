# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The owner's own styles (app/style_library.py): saved names for custom style ids in characters/styles.json, new
versions on an edit, earlier versions and deleted styles that keep working, the refusals, and a damaged file. The
library folder is a temporary one; no model and no live state is used."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import characters, llm, style_library, styles


class StyleLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for patcher in (patch.object(characters, "DIR", self.root / "characters"),
                        patch.dict(style_library._CACHE, clear=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_the_file_lives_beside_the_characters_and_reading_creates_nothing(self):
        self.assertEqual(style_library._file(), self.root / "characters" / "styles.json")
        self.assertEqual(style_library.all_custom(), [])
        self.assertEqual(style_library.choices(), [c for c in styles.choices() if c[1] != styles.REALISTIC])
        self.assertFalse((self.root / "characters").exists())

    def test_a_saved_style_is_listed_by_name_and_picked_by_its_id(self):
        entry, message = style_library.save("Storybook", styles.OTHER, "soft pastel watercolor, paper texture")
        self.assertIn("saved", message)
        style_id = styles.custom_id(styles.OTHER, "soft pastel watercolor, paper texture")
        self.assertEqual(entry["id"], style_id)
        self.assertEqual(style_library.label(style_id), "Storybook")
        self.assertIn(("Storybook", style_id), style_library.choices())
        self.assertEqual(style_library.choices()[:3], [c for c in styles.choices() if c[1] != styles.REALISTIC])
        self.assertIn("starting from Something else", style_library.about(style_id))
        stored = json.loads(style_library._file().read_text(encoding="utf-8"))
        self.assertEqual(stored["styles"][0]["name"], "Storybook")
        self.assertNotIn("id", stored["styles"][0])  # the id is computed from base and look

    def test_a_rename_keeps_the_id_and_an_edit_makes_a_new_version(self):
        entry, _ = style_library.save("Storybook", styles.ANIME, "soft pastel watercolor")
        first = entry["id"]
        renamed, message = style_library.save("Picture book", styles.ANIME, "soft pastel watercolor", entry["key"])
        self.assertEqual(renamed["id"], first)
        self.assertIn("Nothing needs to be made again", message)
        edited, message = style_library.save("Picture book", styles.ANIME, "bright pastel gouache", entry["key"])
        self.assertNotEqual(edited["id"], first)
        self.assertIn("new version", message)
        self.assertEqual(style_library.label(first), "Picture book (earlier version)")
        self.assertIn("earlier version", style_library.about(first))
        self.assertIn(("Picture book (earlier version)", first), style_library.choices(first))
        self.assertNotIn(first, [value for _, value in style_library.choices()])
        back, _ = style_library.save("Picture book", styles.ANIME, "soft pastel watercolor", entry["key"])
        self.assertEqual(back["id"], first)  # back to the first look: it is current again, not an earlier version
        self.assertEqual(style_library.label(first), "Picture book")
        for number in range(12):
            style_library.save("Picture book", styles.ANIME, f"look number {number}", entry["key"])
        self.assertEqual(len(style_library.get(entry["key"])["earlier"]), style_library.MAX_EARLIER)

    def test_a_deleted_style_keeps_working_where_it_is_used(self):
        entry, _ = style_library.save("Noir", styles.REALISTIC, "black and white film noir, hard shadows")
        ok, message = style_library.delete(entry["key"])
        self.assertTrue(ok)
        self.assertIn("keep their look", message)
        self.assertEqual(style_library.label(entry["id"]), "Custom: black and white film noir, hard shadows")
        self.assertIn("Not in your styles any more", style_library.about(entry["id"]))
        self.assertIn(entry["id"], [value for _, value in style_library.choices(entry["id"])])
        self.assertTrue(styles.valid(entry["id"]))
        self.assertFalse(style_library.delete(entry["key"])[0])

    def test_names_starting_points_and_looks_that_cannot_be_saved(self):
        style_library.save("Storybook", styles.OTHER, "soft pastel watercolor")
        for name, base, look, message in (
                ("", styles.OTHER, "ink", "Enter a name"), ("x" * 41, styles.OTHER, "ink wash", "at most 40"),
                ("Anime", styles.OTHER, "ink wash", "built-in"), ("Something else", styles.OTHER, "ink wash", "built-in"),
                ("storybook", styles.OTHER, "ink wash", "already have a style named"),
                ("Ink", "clay", "ink wash", "starts from"), ("Ink", styles.OTHER, "ink: wash", "only letters"),
                ("Ink", styles.OTHER, "soft pastel watercolor", "already have this style"),
                ("Ink", styles.OTHER, "chibi, big heads", "adult"), ("Ink", styles.OTHER, "kids cartoon", "adult")):
            with self.subTest(name=name, look=look):
                entry, problem = style_library.save(name, base, look)
                self.assertIsNone(entry)
                self.assertIn(message, problem)
        with patch.object(llm, "blocked", return_value=True):
            self.assertEqual(style_library.check("Ink", styles.OTHER, "ink wash"), llm.BLOCK_MSG)

    def test_a_damaged_file_reads_as_no_styles_and_is_set_aside_before_the_next_save(self):
        path = style_library._file()
        path.parent.mkdir(parents=True)
        path.write_text("{ not json", encoding="utf-8")
        self.assertEqual(style_library.all_custom(), [])
        entry, _ = style_library.save("Ink", styles.OTHER, "ink wash, rough paper")
        self.assertIsNotNone(entry)
        self.assertEqual(len(list(path.parent.glob("styles.damaged-*.json"))), 1)
        self.assertEqual([item["name"] for item in style_library.all_custom()], ["Ink"])
        path.write_text(json.dumps({"styles": [{"key": "k", "name": "Bad", "base": "clay", "look": "x"},
                                               {"key": "j", "name": "Good", "base": "other", "look": "ink wash"}]}),
                        encoding="utf-8")
        self.assertEqual([item["name"] for item in style_library.all_custom()], ["Good"])  # a bad entry is left out

    def test_a_file_of_another_shape_reads_as_no_styles_and_is_set_aside_before_the_next_save(self):
        path = style_library._file()
        path.parent.mkdir(parents=True)
        for number, shape in enumerate((None, 5, True, "abc", {"a": 1}, [1, 2])):
            with self.subTest(shape=shape):
                path.write_text(json.dumps({"format": 1, "styles": shape}), encoding="utf-8")
                self.assertEqual(style_library.all_custom(), [])
                self.assertEqual(style_library.choices(), [c for c in styles.choices() if c[1] != styles.REALISTIC])
                entry, message = style_library.save(f"Ink {number}", styles.OTHER, f"ink wash {number}")
                self.assertIsNotNone(entry, message)
                self.assertEqual([item["name"] for item in style_library.all_custom()], [f"Ink {number}"])
                path.write_text(json.dumps({"format": 1, "styles": shape}), encoding="utf-8")
        path.write_text(json.dumps(["not", "a", "library"]), encoding="utf-8")
        self.assertEqual(style_library.all_custom(), [])
        self.assertIsNotNone(style_library.save("Ink", styles.OTHER, "ink wash")[0])
        self.assertEqual(len(list(path.parent.glob("styles.damaged-*.json"))), 7)  # every one kept, none overwritten

    def test_a_file_with_an_entry_it_cannot_read_is_kept_before_a_change(self):
        path = style_library._file()
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"styles": [{"key": "k", "name": "Bad", "base": "clay", "look": "x"},
                                               {"key": "j", "name": "Good", "base": "other", "look": "ink wash"}]}),
                        encoding="utf-8")
        entry, _ = style_library.save("Clay", styles.OTHER, "claymation, plasticine")
        self.assertEqual(sorted(item["name"] for item in style_library.all_custom()), ["Clay", "Good"])
        [kept] = path.parent.glob("styles.damaged-*.json")
        self.assertIn("Bad", kept.read_text(encoding="utf-8"))  # the entry it could not read is still on disk
        style_library.delete(entry["key"])
        self.assertEqual(len(list(path.parent.glob("styles.damaged-*.json"))), 1)  # a whole file is not set aside

    def test_a_read_that_fails_is_not_remembered_and_never_overwrites_the_library(self):
        first, _ = style_library.save("Storybook", styles.OTHER, "soft pastel watercolor")
        style_library._CACHE.clear()
        with patch.object(Path, "read_text", side_effect=OSError("the file is in use")):
            self.assertEqual(style_library.all_custom(), [])  # reads as none for now...
            entry, message = style_library.save("Ink", styles.OTHER, "ink wash")
            self.assertIsNone(entry)
            self.assertIn("Could not read your styles", message)
            self.assertEqual(style_library.delete(first["key"]), (False, "Could not read your styles: the file is in "
                                                                           "use"))
        self.assertEqual([item["name"] for item in style_library.all_custom()], ["Storybook"])  # ...and not after


class StyleWordsTests(unittest.TestCase):
    """The Styles page's Translate to English (llm.style_words), with the language model replaced."""

    def words(self, answer, description="miękka akwarela z fakturą papieru"):
        with patch.object(llm, "ensure", return_value=None), \
                patch.object(llm, "ask", return_value=answer) as ask:
            return llm.style_words(description), ask

    def test_the_answer_comes_back_as_clean_look_words(self):
        result, ask = self.words((True, "Soft pastel watercolor, visible paper texture."))
        self.assertEqual(result, (True, "Soft pastel watercolor, visible paper texture"))
        self.assertIn("miękka akwarela z fakturą papieru", ask.call_args.args[1])
        self.assertTrue(ask.call_args.kwargs["check"]("soft pastel watercolor"))
        self.assertFalse(ask.call_args.kwargs["check"]("miękka akwarela, pastelowe światło"))
        self.assertTrue(styles.valid(styles.custom_id(styles.OTHER, result[1])))

    def test_an_answer_that_is_not_english_or_not_for_adults_is_refused(self):
        ok, message = self.words((True, "miękka akwarela, pastelowe światło"))[0]
        self.assertFalse(ok)
        self.assertIn("English", message)
        self.assertEqual(self.words((True, "chibi style, big heads, pastel"))[0], (False, llm.STYLE_MINOR_MSG))
        self.assertEqual(self.words((False, "The model did not answer."))[0], (False, "The model did not answer."))

    def test_nothing_is_asked_for_an_empty_or_refused_description_or_without_the_assistant(self):
        with patch.object(llm, "ask") as ask:
            self.assertEqual(llm.style_words("  "), (False, "Describe the look first."))
            with patch.object(llm, "blocked", return_value=True):
                self.assertEqual(llm.style_words("ink wash"), (False, llm.BLOCK_MSG))
            with patch.object(llm, "ensure", return_value="The assistant is not installed."):
                self.assertEqual(llm.style_words("ink wash"), (False, "The assistant is not installed."))
        ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
