# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Dialogue parsing: @Name tags, bare voice names and gender words choose the voice of each
spoken line; the picture and sound descriptions keep the name but never the @ sign."""
import unittest

from app import dialogue

VOICES = {"hanna": ("hanna", "female"), "zoe": ("zoe", "female"), "marek": ("marek", "male")}
CATALOG = {"female": ["hanna", "zoe", "maja"], "male": ["marek", "adam"]}
# The owner's scene of 2026-11-16: the screen's text became the dog's first spoken line, and "talking" in its place
# was painted on the screen
NEWS = ('chain_dog, a cute small adult dog with a red scarf, sits in a professional crypto news studio. Behind him, a '
        'large screen displays the exact text "CRYPTO NEWS". He smiles adorably, then looks directly at the camera and '
        'cheerfully says "Crypto News!" with accurate lip sync.')


class ShownTextTests(unittest.TestCase):
    """A quote that is text shown in the picture (a screen's, a sign's, a shirt's) stays in the picture text as written
    and is never a spoken line."""

    def test_the_screens_text_is_shown_and_only_the_line_is_spoken(self):
        visual, sound, lines = dialogue.parse(NEWS)
        self.assertEqual([(line.speaker, line.text) for line in lines], [("male", "Crypto News!")])
        self.assertEqual((dialogue.quotes(NEWS), dialogue.shown(NEWS)), (("Crypto News!",), ("CRYPTO NEWS",)))
        self.assertIn('a large screen displays the exact text "CRYPTO NEWS". He smiles', visual)
        self.assertNotIn("text talking", visual)
        self.assertNotIn("CRYPTO NEWS", sound)
        self.assertEqual([match.group(1) for match in dialogue.spoken(NEWS)], ["Crypto News!"])

    def test_writing_and_displays_are_shown_and_speech_stays_speech(self):
        cases = {
            'A sign reads "OPEN".': True,
            'A neon sign that says "BAR" glows above the door.': True,
            'A sign: "Open". Ada smiles.': True,
            'The headline reads "Dogs win".': True,
            'He wears a shirt with the text "BOSS".': True,
            'She holds a book titled "Dune".': True,
            'The text on the screen says "Error 404".': True,
            'A banner with "SALE" hangs above.': True,
            'The screen behind him shows "LIVE".': True,
            'He types "hello" on the keyboard.': True,
            'Na ekranie wyświetla się napis „CRYPTO NEWS”.': True,
            'Kuba looks at the sign and says "Hmm".': False,
            'Ola says the words "I love you".': False,
            'Chain Dog reads the headline aloud: "Dogs win!"': False,
            'Ola mówi „Cześć”.': False,
            'He smiles "Hi!"': False,
            'A woman in a cafe smiles at the camera and says „Cześć, miło Cię widzieć!”': False,
        }
        for text, shown in cases.items():
            with self.subTest(text):
                self.assertEqual(bool(dialogue.shown(text)), shown)
                self.assertEqual(bool(dialogue.quotes(text)), not shown)

    def test_a_sentence_can_hold_a_line_and_a_sign(self):
        text = 'He says "Hi" and points at the sign reading "OPEN".'
        visual, sound, lines = dialogue.parse(text)
        self.assertEqual(([line.text for line in lines], dialogue.shown(text)), (["Hi"], ("OPEN",)))
        self.assertEqual(visual, 'He talks and points at the sign reading "OPEN", natural lip movement')
        self.assertNotIn("OPEN", sound)

    def test_only_shown_text_keeps_the_picture_text_as_written(self):
        visual, sound, lines = dialogue.parse('A sign reads "OPEN" over a quiet street.')
        self.assertEqual((visual, sound, lines), ('A sign reads "OPEN" over a quiet street.',
                                                  "A sign reads over a quiet street.", []))
        # a prompt without quotes is parsed as it always was
        self.assertEqual(dialogue.parse("A quiet street , at night."), ("A quiet street , at night.",) * 2 + ([],))


class VoiceTagTests(unittest.TestCase):
    def test_tag_picks_the_voice_and_its_gender(self):
        visual, sound, lines = dialogue.parse("@Marek says „Good morning!” in a cafe.", VOICES)
        self.assertEqual([(l.voice, l.speaker, l.who, l.text) for l in lines],
                         [("marek", "male", "male#marek", "Good morning!")])
        self.assertNotIn("@", visual)
        self.assertNotIn("@", sound)
        self.assertIn("Marek talks", visual)
        self.assertTrue(visual.endswith(", natural lip movement"))

    def test_tag_wins_over_gender_words_and_bare_names(self):
        _, _, lines = dialogue.parse("A woman called Hanna and @Marek says „Hi”", VOICES)
        self.assertEqual((lines[0].voice, lines[0].speaker), ("marek", "male"))

    def test_bare_registered_name_before_the_quote(self):
        _, _, lines = dialogue.parse("Hanna smiles at the camera and says „Hello there”", VOICES)
        self.assertEqual((lines[0].voice, lines[0].speaker, lines[0].who),
                         ("hanna", "female", "female#hanna"))

    def test_unknown_tag_is_ignored_and_loses_its_at_sign(self):
        visual, _, lines = dialogue.parse("@Nobody, a man, says „Hey”", VOICES)
        self.assertEqual((lines[0].voice, lines[0].speaker), ("", "male"))
        self.assertNotIn("@", visual)
        self.assertIn("Nobody", visual)

    def test_image_marker_with_at_sign_is_not_a_voice(self):
        self.assertEqual(dialogue.tags("a hoodie @2 and @Hanna says „x” @hanna"), ["Hanna", "hanna"])
        _, _, lines = dialogue.parse("a man in the hoodie @2 says „Nice”", VOICES)
        self.assertEqual((lines[0].voice, lines[0].speaker), ("", "male"))

    def test_two_named_speakers_are_different_people(self):
        _, _, lines = dialogue.parse("@Hanna says „Hi”. @Zoe answers „Hey”. @Hanna says „Go”.", VOICES)
        self.assertEqual([l.who for l in lines], ["female#hanna", "female#zoe", "female#hanna"])

    def test_unnamed_lines_keep_the_gender_heuristics(self):
        _, _, lines = dialogue.parse("A woman says „Hi”. A man says „Hey”.", VOICES)
        self.assertEqual([(l.voice, l.who) for l in lines], [("", "female#0"), ("", "male#0")])

    def test_without_a_voice_map_only_gender_is_recognised(self):
        visual, _, lines = dialogue.parse("@Hanna says „Hello”")
        self.assertEqual((lines[0].voice, lines[0].speaker), ("", "female"))
        self.assertNotIn("@", visual)

    def test_polish_prompt_with_a_tag(self):
        visual, sound, lines = dialogue.parse("@Hanna mówi „Dzień dobry” w kawiarni.", VOICES)
        self.assertEqual((lines[0].voice, lines[0].text), ("hanna", "Dzień dobry"))
        self.assertIn("Hanna talks", visual)
        self.assertNotIn("Dzień dobry", sound)

    def test_prompt_without_quotes_keeps_the_name_without_at(self):
        visual, sound, lines = dialogue.parse("@Hanna walks in the park", VOICES)
        self.assertEqual(lines, [])
        self.assertEqual(visual, "Hanna walks in the park")
        self.assertEqual(sound, visual)

    def test_closer_gender_word_beats_a_bare_name_elsewhere_in_the_segment(self):
        _, _, lines = dialogue.parse("Marek nods. The woman replies „Hey there”", VOICES)
        self.assertEqual((lines[0].voice, lines[0].speaker, lines[0].who), ("", "female", "female#0"))
        _, _, lines = dialogue.parse("Hanna watches as the man says „Hello”", VOICES)
        self.assertEqual((lines[0].voice, lines[0].speaker), ("", "male"))
        _, _, lines = dialogue.parse("In Jan 2026 a woman says „Hi”", {"jan": ("jan", "male")})
        self.assertEqual((lines[0].voice, lines[0].speaker), ("", "female"))

    def test_bare_name_after_the_gender_word_still_picks_the_voice(self):
        _, _, lines = dialogue.parse("The woman, Hanna, says „Hi”", VOICES)
        self.assertEqual((lines[0].voice, lines[0].speaker), ("hanna", "female"))
        _, _, lines = dialogue.parse("A woman next to @Marek says „Hi”", VOICES)
        self.assertEqual((lines[0].voice, lines[0].speaker), ("marek", "male"))  # tags always win

    def test_tags_inside_spoken_lines_are_not_voice_tags(self):
        self.assertEqual(dialogue.tags('@Hanna says „follow @contsy on IG” and @Marek nods'),
                         ["Hanna", "Marek"])
        _, _, lines = dialogue.parse('@Hanna says „follow @contsy on IG”', VOICES)
        self.assertEqual(lines[0].text, "follow @contsy on IG")

    def test_same_person_keeps_talking_until_a_second_one_is_announced(self):
        _, _, lines = dialogue.parse("@Hanna says „Hi”. She adds „Bye”. A man says „Yo”.", VOICES)
        self.assertEqual([(l.voice, l.who) for l in lines],
                         [("hanna", "female#hanna"), ("hanna", "female#hanna"), ("", "male#0")])
        _, _, lines = dialogue.parse("Two women: @Hanna says „Hi”. The other says „Bye”.", VOICES)
        self.assertEqual([(l.voice, l.who) for l in lines], [("hanna", "female#hanna"), ("", "female#0")])

    def test_gender_words_are_recognised(self):
        self.assertTrue(dialogue.is_gender_word("woman"))
        self.assertTrue(dialogue.is_gender_word("Pan"))
        self.assertFalse(dialogue.is_gender_word("Hanna"))
        self.assertFalse(dialogue.is_gender_word(""))


class VoicesForTests(unittest.TestCase):
    def test_named_lines_keep_their_voice(self):
        _, _, lines = dialogue.parse("@Zoe says „Hi”. A man says „Hey”.", VOICES)
        self.assertEqual(dialogue.voices_for(lines, CATALOG), ["zoe", "marek"])

    def test_two_unnamed_women_get_the_default_and_the_next_voice(self):
        _, _, lines = dialogue.parse("Two women chat. One says „Hi”, the other says „Hey”.", VOICES)
        self.assertEqual([l.who for l in lines], ["female#0", "female#1"])
        self.assertEqual(dialogue.voices_for(lines, CATALOG), ["hanna", "zoe"])

    def test_character_voice_comes_first_for_its_gender_only(self):
        _, _, lines = dialogue.parse("A woman says „Hi”. A man says „Hey”. @Zoe says „Go”.", VOICES)
        self.assertEqual(dialogue.voices_for(lines, CATALOG, ("maja", "female")),
                         ["maja", "marek", "zoe"])
        self.assertEqual(dialogue.voices_for(lines, CATALOG, ("adam", "male")),
                         ["hanna", "adam", "zoe"])

    def test_named_voice_that_no_longer_exists_falls_back_to_its_gender_default(self):
        _, _, lines = dialogue.parse("@Marek says „Hey”.", VOICES)
        self.assertEqual(dialogue.voices_for(lines, {"female": ["hanna"], "male": ["adam"]}), ["adam"])

    def test_missing_catalog_entries_give_an_empty_slug(self):
        _, _, lines = dialogue.parse("A man says „Hey”.", VOICES)
        self.assertEqual(dialogue.voices_for(lines, {}), [""])
        self.assertEqual(dialogue.voices_for(lines, {"male": []}, ("kasia", "female")), [""])

    def test_a_named_voice_and_an_unnamed_woman_get_different_voices(self):
        _, _, lines = dialogue.parse("Two women: @Hanna says „Hi”. The other says „Bye”.", VOICES)
        self.assertEqual(dialogue.voices_for(lines, CATALOG), ["hanna", "zoe"])


class SpeakerTests(unittest.TestCase):
    def _speakers(self, prompt):
        return [l.speaker for l in dialogue.parse(prompt, VOICES)[2]]

    def test_polish_on_and_pan_still_mark_the_man(self):
        for prompt in ("Kobieta pyta „Gdzie?”. On zapytał „Co?”.",
                       "Kobieta pyta „Gdzie?”, a on z uśmiechem odpowiada „Tutaj”.",
                       "Kobieta mówi „Cześć”. On też mówi „Tutaj”.",
                       "Kobieta mówi „Cześć”. Starszy pan w kapeluszu odpowiada „Dzień dobry”.",
                       "Kobieta mówi „Cześć”. Pan doktor odpowiada „Dzień dobry”.",
                       "Ona: „Gdzie?” On: „Tutaj”."):
            with self.subTest(prompt=prompt):
                self.assertEqual(self._speakers(prompt), ["female", "male"])

    def test_english_on_and_pan_are_not_male(self):
        for prompt in ("A woman sits on a bench and says „Hi”.", "A woman on Main Street says „Hi”.",
                       "A woman holds a pan and says „Dinner is ready”."):
            with self.subTest(prompt=prompt):
                self.assertEqual(self._speakers(prompt), ["female"])

    def test_the_subject_speaks_not_the_person_spoken_to(self):
        for prompt in ("The woman looks at the man and says „Hi”.", "The woman says to the man „Hi”.",
                       "She looks at him and smiles: „Hi”"):
            with self.subTest(prompt=prompt):
                self.assertEqual(self._speakers(prompt), ["female"])

    def test_tags_in_an_earlier_sentence_do_not_beat_the_speaker(self):
        lines = dialogue.parse("@Hanna waves at a man. The man says „Hi”.", VOICES)[2]
        self.assertEqual((lines[0].voice, lines[0].speaker), ("", "male"))
        lines = dialogue.parse("@Hanna walks in. She says „Hi”.", VOICES)[2]
        self.assertEqual((lines[0].voice, lines[0].speaker), ("hanna", "female"))  # she = Hanna
        lines = dialogue.parse("Hanna smiles at him and says „Hello there”", VOICES)[2]
        self.assertEqual(lines[0].voice, "hanna")


class ShownTextWithImageNumbersTests(unittest.TestCase):
    def test_an_image_number_before_the_verb_keeps_the_quote_in_the_picture(self):
        for text, quote in (('The big screen in the TV studio [2] displays "CRYPTO NEWS".', "CRYPTO NEWS"),
                            ('A neon sign [2] reads "OPEN".', "OPEN"), ('The screen [1] shows "SALE".', "SALE")):
            with self.subTest(text=text):
                self.assertEqual(dialogue.shown(text), (quote,))
                self.assertEqual(dialogue.spoken(text), [])
                self.assertEqual(dialogue.shown(text), dialogue.shown(text.replace(" [2]", "").replace(" [1]", "")))

    def test_someone_in_a_tv_studio_speaks(self):
        for text in ('Chain Dog in the TV studio says "Hello".', 'Chain Dog in the TV studio [2] says "Hello".',
                     'Ola in the television studio says "Good evening".'):
            with self.subTest(text=text):
                self.assertEqual(dialogue.shown(text), ())
                self.assertEqual(len(dialogue.spoken(text)), 1)
        self.assertEqual(dialogue.shown('The TV shows "BREAKING NEWS".'), ("BREAKING NEWS",))  # a TV still shows text


if __name__ == "__main__":
    unittest.main()
