# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Voice library: the bundled manifest, imported voices, name checks, defaults, the voice of
a character and how speech resolves voice names. No real folder, FFmpeg or GPU is touched."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import characters, prefs, tts, ui_voices, voices


class _Library(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="voices_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.builtin = self.tmp / "assets"
        self.builtin.mkdir()
        for name in ("hanna", "zoe", "marek"):
            (self.builtin / f"{name}.wav").write_bytes(b"RIFF")
        (self.builtin / "voices.json").write_text(json.dumps({"voices": [
            {"name": "Hanna", "file": "hanna.wav", "gender": "female", "description": "20s, bright",
             "seconds": 7.0, "age": "20s", "f0_hz": 275},
            {"name": "Zoe", "file": "zoe.wav", "gender": "female", "age": "teen"},
            {"name": "Marek", "file": "marek.wav", "gender": "male", "f0_hz": 114, "age": "30s"},
            {"name": "Ghost", "file": "missing.wav", "gender": "male"},
            {"name": "Odd", "file": "hanna.wav", "gender": "robot"}]}), encoding="utf-8")
        self.user = self.tmp / "voices"
        (self.tmp / "characters").mkdir()
        for target in (patch.object(voices, "BUILTIN_DIR", self.builtin),
                       patch.object(voices, "MANIFEST", self.builtin / "voices.json"),
                       patch.object(voices, "USER_DIR", self.user),
                       patch.object(characters, "DIR", self.tmp / "characters"),
                       patch.object(prefs, "_FILE", self.tmp / "prefs.json")):
            target.start()
            self.addCleanup(target.stop)

    def add_voice(self, name="Kasia", gender="female", seconds=6.4, description="my voice",
                  age="", f0=201.0):
        sample = self.tmp / "rec.m4a"
        sample.write_bytes(b"x")

        def convert(src, dst):
            dst.write_bytes(b"RIFF")
            return True, ""

        with patch.object(voices, "_convert", convert), \
                patch.object(voices, "measure_pitch", return_value=f0), \
                patch.object(voices.ffmpeg_utils, "probe_duration", return_value=seconds):
            return voices.create(name, gender, str(sample), True, description, age)


class LibraryTests(_Library):
    def test_manifest_lists_bundled_voices_and_skips_broken_entries(self):
        pool = voices.all_voices()
        self.assertEqual([v.name for v in pool], ["Hanna", "Zoe", "Marek"])
        self.assertTrue(all(v.builtin for v in pool))
        self.assertEqual(pool[0].label, "Hanna — Woman, 20s, bright")
        self.assertEqual(pool[2].label, "Marek — Man, 30s")
        self.assertEqual(Path(pool[0].path), self.builtin / "hanna.wav")
        self.assertEqual((pool[0].age, pool[0].f0_hz, pool[1].age, pool[2].f0_hz), ("20s", 275.0, "", 114.0))

    def test_pitch_bands_depend_on_the_gender(self):
        hanna, zoe, marek = voices.all_voices()
        self.assertEqual((voices.pitch(hanna), voices.pitch(zoe), voices.pitch(marek)), ("high", "", "medium"))
        self.assertEqual(voices.pitch(voices.Voice("A", "a", "female", "", f0_hz=160)), "low")
        self.assertEqual(voices.pitch(voices.Voice("B", "b", "male", "", f0_hz=160)), "high")
        self.assertEqual(voices.pitch(voices.Voice("C", "c", "male", "", f0_hz=90)), "low")

    def test_lookup_by_slug_name_or_tag(self):
        self.assertEqual(voices.get("hanna").name, "Hanna")
        self.assertEqual(voices.get("HANNA").slug, "hanna")
        self.assertEqual(voices.get("@Hanna").slug, "hanna")
        self.assertIsNone(voices.get("ghost"))
        self.assertIsNone(voices.get(""))
        self.assertEqual(voices.tokens()["hanna"], ("hanna", "female"))
        self.assertEqual(voices.tokens()["marek"], ("marek", "male"))

    def test_defaults_follow_settings_and_fall_back_to_the_first_voice(self):
        self.assertEqual(voices.defaults(), {"female": "hanna", "male": "marek"})
        prefs.save(default_female="zoe")
        self.assertEqual(voices.defaults()["female"], "zoe")
        self.assertEqual(voices.catalog()["female"], ["zoe", "hanna"])
        prefs.save(default_female="marek")  # wrong gender: ignored
        self.assertEqual(voices.defaults()["female"], "hanna")
        prefs.save(default_male="gone")
        self.assertEqual(voices.defaults()["male"], "marek")

    def test_character_voice_is_used_only_while_it_exists(self):
        self.assertEqual(voices.for_character({"voice": "marek"}), ("marek", "male"))
        self.assertIsNone(voices.for_character({"voice": "gone"}))
        self.assertIsNone(voices.for_character({"name": "Ola"}))
        self.assertIsNone(voices.for_character(None))

    def test_name_checks(self):
        for name, fragment in (("", "Enter"), ("1", "one word"), ("12", "one word"),
                               ("2pac", "one word"), ("3Hanna", "one word"),
                               ("two words", "one word"), ("x" * 25, "one word"),
                               ("woman", "woman or a man"), ("Pan", "woman or a man"),
                               ("Hanna", "already exists"), ("HANNA", "already exists")):
            with self.subTest(name=name):
                self.assertIn(fragment, voices.check_name(name))
        self.assertEqual(voices.check_name("Kasia"), "")
        self.assertEqual(voices.check_name("Małgosia"), "")
        self.assertEqual(voices.slugify("Małgosia"), "malgosia")


class ImportTests(_Library):
    def test_create_converts_and_stores_the_voice(self):
        voice, msg = self.add_voice(age="40s")
        self.assertIsNotNone(voice, msg)
        self.assertEqual((voice.slug, voice.gender, voice.builtin, voice.seconds, voice.age, voice.f0_hz),
                         ("kasia", "female", False, 6.4, "40s", 201.0))
        self.assertEqual(Path(voice.path), self.user / "kasia" / "sample.wav")
        stored = json.loads((self.user / "kasia" / "voice.json").read_text(encoding="utf-8"))
        self.assertEqual((stored["name"], stored["consent"], stored["source"], stored["age"], stored["f0_hz"]),
                         ("Kasia", True, "rec.m4a", "40s", 201))
        self.assertNotIn("path", stored)
        self.assertIn("@Kasia", msg)
        self.assertEqual([v.name for v in voices.all_voices()], ["Hanna", "Zoe", "Marek", "Kasia"])
        self.assertEqual(voices.get("kasia").description, "my voice")
        self.assertEqual(voices.get("kasia").label, "Kasia — Woman, 40s, my voice")
        self.assertEqual(voices.pitch(voices.get("kasia")), "medium")
        self.assertEqual(voices.tokens()["kasia"], ("kasia", "female"))
        self.assertEqual(voices.catalog()["female"], ["hanna", "zoe", "kasia"])
        self.assertEqual(self.add_voice(name="Ola", age="teen")[0].age, "")  # unknown age is dropped

    def test_create_rejects_bad_input_before_converting(self):
        sample = self.tmp / "rec.wav"
        sample.write_bytes(b"x")
        cases = [("Hanna", "female", str(sample), True, "already exists"),
                 ("Kasia", "other", str(sample), True, "woman's or a man's"),
                 ("Kasia", "female", str(sample), False, "Confirm"),
                 ("Kasia", "female", None, True, "Upload or record"),
                 ("Kasia", "female", str(self.tmp / "nope.wav"), True, "Upload or record")]
        for name, gender, path, consent, fragment in cases:
            with self.subTest(fragment=fragment), patch.object(voices, "_convert") as convert:
                voice, msg = voices.create(name, gender, path, consent)
                self.assertIsNone(voice)
                self.assertIn(fragment, msg)
                convert.assert_not_called()
        self.assertFalse(self.user.exists())

    def test_a_name_or_description_that_combines_a_minor_with_sexual_content_is_refused(self):
        sample = self.tmp / "rec.wav"
        sample.write_bytes(b"x")
        with patch.object(voices, "_convert") as convert:
            with patch.object(voices.llm, "blocked", return_value=True):  # one word cannot say both
                self.assertEqual(voices.check_name("Kasia"), voices.llm.BLOCK_MSG)
            self.assertEqual(voices.create("Kasia", "female", str(sample), True,
                                           description="a 15-year-old in a bikini"),
                             (None, voices.llm.BLOCK_MSG))
            convert.assert_not_called()
        self.assertFalse(self.user.exists())

    def test_short_or_unreadable_samples_are_rejected_and_cleaned_up(self):
        voice, msg = self.add_voice(seconds=1.2)
        self.assertIsNone(voice)
        self.assertIn("Only 1.2 s", msg)
        self.assertIn("3–12 seconds", msg)
        self.assertFalse((self.user / "kasia").exists())
        voice, msg = self.add_voice(seconds=None)  # silence trimmed to nothing
        self.assertIsNone(voice)
        self.assertIn("silent or too quiet", msg)
        self.assertFalse((self.user / "kasia").exists())
        sample = self.tmp / "rec.wav"
        sample.write_bytes(b"x")
        with patch.object(voices, "_convert", return_value=(False, "Invalid data found")):
            voice, msg = voices.create("Kasia", "female", str(sample), True)
        self.assertIsNone(voice)
        self.assertIn("Could not read the sample (Invalid data found)", msg)
        self.assertFalse((self.user / "kasia").exists())

    def test_delete_moves_the_voice_to_the_recycle_bin_and_resets_who_used_it(self):
        self.add_voice()
        prefs.save(default_female="kasia")
        self.assertEqual(voices.defaults()["female"], "kasia")
        folder = self.tmp / "characters" / "ola"
        folder.mkdir(parents=True)
        (folder / "character.json").write_text(json.dumps({"name": "Ola", "slug": "ola", "voice": "kasia"}),
                                               encoding="utf-8")
        with patch.object(characters, "DIR", self.tmp / "characters"), \
                patch.object(voices.trash, "move", side_effect=lambda p: shutil.rmtree(p) or True) as move:
            ok, msg = voices.delete("kasia")
            self.assertTrue(ok)
            self.assertIn("Recycle Bin", msg)
            self.assertEqual(characters.get("ola")["voice"], "")
        self.assertEqual(Path(move.call_args.args[0]), self.user / "kasia")
        self.assertIsNone(voices.get("kasia"))
        self.assertEqual(prefs.load()["default_female"], "")
        self.assertEqual(voices.delete("hanna"), (False, "Built-in voices cannot be removed."))
        self.assertEqual(voices.delete("kasia"), (False, "Voice not found."))


class SpeechResolutionTests(_Library):
    def test_names_legacy_names_and_unknown_names_resolve(self):
        self.assertEqual(tts.resolve("Marek").slug, "marek")
        self.assertEqual(tts.resolve("zoe").slug, "zoe")
        self.assertEqual(tts.resolve("Woman 1 — higher voice").slug, "hanna")
        self.assertEqual(tts.resolve("Kobieta 2 — niższy głos").slug, "hanna")
        self.assertEqual(tts.resolve("Man 1 — low voice").slug, "marek")
        self.assertEqual(tts.resolve("Mężczyzna 2 — średni głos").slug, "marek")
        self.assertEqual(tts.resolve("nonsense").slug, "hanna")
        self.assertEqual(tts.resolve(None).slug, "hanna")
        prefs.save(default_female="zoe")
        self.assertEqual(tts.resolve("Woman 2 — lower voice").slug, "zoe")

    def test_without_any_voice_speech_reports_it_instead_of_crashing(self):
        with patch.object(voices, "MANIFEST", self.tmp / "none.json"), \
                patch.object(tts, "available", return_value=True):
            self.assertIsNone(tts.resolve("hanna"))
            ok, msg = tts.synthesize_many([("Hi", "hanna", str(self.tmp / "out.wav"))])
        self.assertFalse(ok)
        self.assertIn("No voice samples", msg)

    def test_leading_parenthesis_is_not_read_as_a_style_instruction(self):
        self.assertEqual(tts._speakable("(sigh) Hello there"), "sigh Hello there")
        self.assertEqual(tts._speakable("Hello (there)"), "Hello (there)")
        self.assertEqual(tts._speakable("  （quiet） ok "), "quiet ok")

    def test_speech_lines_reference_the_correct_voice_samples(self):
        self.add_voice()
        first, last = self.tmp / "a.wav", self.tmp / "b.wav"
        with (patch.object(tts, "available", return_value=True),
              patch.object(tts.engine_runtime, "engine") as engine):
            synthesize = engine.return_value.tts
            synthesize.return_value = last
            self.assertEqual(tts.synthesize_many([("Hi", "Kasia", str(first)),
                                                  ("(ha) Yo", "Man 1 — low voice", str(last))]),
                             (True, str(last)))
        lines = synthesize.call_args.args[1]
        self.assertEqual([line.text for line in lines], ["Hi", "ha Yo"])
        self.assertEqual([Path(line.reference) for line in lines],
                         [self.user / "kasia" / "sample.wav", self.builtin / "marek.wav"])
        self.assertEqual([Path(line.output) for line in lines], [first, last])


class CharacterVoiceTests(_Library):
    def test_a_characters_own_voice_is_found_by_its_key_and_never_listed_in_the_library(self):
        from tests.character_fixtures import store_character

        store_character(self.tmp / "characters", {"name": "Ola", "slug": "ola", "gender": "woman"}, voice=b"RIFF")
        own = voices.get("character:ola")
        self.assertEqual((own.name, own.slug, own.gender), ("Ola", "character:ola", "female"))
        self.assertEqual(Path(own.path), self.tmp / "characters" / "ola" / "voice" / "sample.wav")
        self.assertNotIn("character:ola", [v.slug for v in voices.all_voices()])
        self.assertNotIn("character:ola", voices.tokens())  # "@Ola" is the character, not a voice
        self.assertEqual(voices.for_character(characters.get("ola")), ("character:ola", "female"))
        self.assertIn("name of a character", voices.check_name("Ola"))
        self.assertIsNone(voices.get("character:nobody"))

    def test_an_older_reference_character_keeps_its_library_voice(self):
        folder = self.tmp / "characters" / "ada"
        folder.mkdir()
        (folder / "character.json").write_text(json.dumps({"name": "Ada", "slug": "ada", "gender": "woman",
                                                          "status": "reference", "voice": "marek"}), encoding="utf-8")
        self.assertEqual(characters.get("ada")["voice"], "marek")
        self.assertEqual(voices.for_character(characters.get("ada")), ("marek", "male"))


class VoicesPageTests(_Library):
    def test_voice_views_render_from_the_library(self):
        self.add_voice()
        prefs.save(default_male="marek")
        women_html, men_html, women, men, listen, removable = ui_voices.voice_views()
        self.assertIn("@Hanna", women_html)
        self.assertIn("@Kasia", women_html)
        self.assertIn("Imported", women_html)
        self.assertNotIn("@Marek", women_html)
        self.assertIn("@Marek", men_html)
        self.assertEqual(women["value"], "hanna")
        self.assertEqual(men["value"], "marek")
        self.assertEqual([slug for _, slug in women["choices"]], ["hanna", "zoe", "kasia"])
        self.assertEqual(len(listen["choices"]), 4)
        self.assertEqual([slug for _, slug in removable["choices"]], ["kasia"])

    def test_tables_group_by_pitch_and_filter(self):
        self.add_voice(age="40s", f0=150.0)  # a low woman's voice
        html = ui_voices.render("female")
        self.assertIn("3 of 3 voices", html)
        self.assertLess(html.index("Low voices (1)"), html.index("High voices (1)"))
        self.assertLess(html.index("High voices (1)"), html.index("Pitch not measured (1)"))
        self.assertIn("@Kasia", ui_voices.render("female", pitch="low"))
        self.assertNotIn("@Hanna", ui_voices.render("female", pitch="low"))
        self.assertIn("1 of 3 voices", ui_voices.render("female", age="40s"))
        self.assertIn("@Zoe", ui_voices.render("female", age="unknown"))
        self.assertNotIn("@Hanna", ui_voices.render("female", source="imported"))
        self.assertIn("@Hanna", ui_voices.render("female", query="BRIGHT"))
        self.assertIn("No voice matches", ui_voices.render("female", query="nobody"))
        self.assertIn("No voices yet", ui_voices.render("male", pool=[]))

    def test_empty_library_renders_a_hint(self):
        with patch.object(voices, "MANIFEST", self.tmp / "none.json"):
            women_html, men_html, women, men, listen, removable = ui_voices.voice_views()
        self.assertIn("No voices yet", women_html)
        self.assertIn("No voices yet", men_html)
        self.assertIsNone(women["value"])
        self.assertEqual(listen["choices"], [])


if __name__ == "__main__":
    unittest.main()
