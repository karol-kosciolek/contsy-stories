# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Animal and creature characters in the creator's photos (app/char_photos.py) and the creature texts of the styles
(app/styles.py): an animated style draws a creature from its look alone (no gender, no person, no default outfit), a
custom style takes its starting point's creature texts, Realistic draws none, the gate is asked about a creature and
its face is never compared. The models, the gate and the face check are replaced: no GPU and no live state."""
from contextlib import ExitStack
import io
from pathlib import Path
import re
import string
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from app import char_photos, characters, llm, prefs, queue_manager, styles

LOOK = ("An adorable adult anthropomorphic red-orange corgi dog with big brown eyes, wearing a tiny blue hoodie, "
        "standing upright on two legs.")
DRAWN = LOOK.rstrip(".")  # how the look reaches the anchor's prompt
HUMAN = re.compile(r"\{gender\}|\b(?:wo)?m[ae]n\b|\bpersons?\b|\bsweaters?\b|\bcrew-neck\b", re.I)
YOUNG = re.compile(r"\b(?:pupp(?:y|ies)|kittens?|cubs?|foals?|fawns?|chicks?|hatchlings?|bab(?:y|ies)|young|"
                   r"juveniles?)\b", re.I)
REALISTIC_WORDS = ("photoreal", "realistic", "live-action", "real scene", "real-looking", "real life")
ANIMATED = (styles.ANIMATION_3D, styles.CARTOON, styles.ANIME, styles.OTHER)  # every starting point but Realistic


def _fields(template: str) -> set:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


class CreatureTextTests(unittest.TestCase):
    def test_every_animated_starting_point_draws_one_grown_creature_alone_in_its_own_medium(self):
        for base_id in ANIMATED:
            base = styles.BASES[base_id]
            with self.subTest(style=base_id):
                self.assertTrue(base.allows_creatures)
                self.assertEqual(_fields(base.creature_anchor), {"look"})  # no gender
                self.assertEqual(_fields(base.creature_view), set())
                for text in (base.creature_anchor, base.creature_view):
                    self.assertIsNone(HUMAN.search(text), text)
                    self.assertIsNone(YOUNG.search(text), text)
                    self.assertFalse([word for word in REALISTIC_WORDS if word in text.lower()], text)
                    self.assertFalse(llm.blocked(text) or llm.refers_to_minor(text), text)
                    self.assertIn("fully grown non-human creature", text)
                self.assertIn("not a human", base.creature_anchor)
                self.assertIn("Alone, no people and no other animals.", base.creature_anchor)
                # the medium words of the style's person texts: the same opening and finish of the anchor, the
                # same opening of the view
                self.assertEqual(base.creature_anchor.split(" of ")[0], base.anchor.split(" of ")[0])
                self.assertEqual(base.creature_anchor.rsplit(", ", 1)[1], base.anchor.rsplit(", ", 1)[1])
                self.assertEqual(base.creature_view.split(":")[0], base.view.split(":")[0])
        anime = styles.get(styles.ANIME)
        self.assertIn("with an animal head and an animal body, not a human with animal ears", anime.creature_anchor)
        self.assertIn("not a human with animal ears", anime.creature_view)

    def test_realistic_draws_no_creature_and_a_custom_style_follows_its_starting_point(self):
        realistic = styles.get(styles.REALISTIC)
        self.assertEqual((realistic.creature_anchor, realistic.creature_view, realistic.allows_creatures),
                         ("", "", False))
        for base_id, base in styles.BASES.items():
            with self.subTest(base=base_id):
                custom = styles.get(styles.custom_id(base_id, "soft pastel watercolor"))
                self.assertEqual((custom.creature_anchor, custom.creature_view),
                                 (base.creature_anchor, base.creature_view))
                self.assertEqual(custom.allows_creatures, base_id != styles.REALISTIC)
                self.assertEqual(custom.allows_creatures, characters.creature_style_ok(custom.id))

    def test_the_creature_views_follow_the_person_views_slot_for_slot(self):
        views = char_photos.CREATURE_VIEWS
        self.assertEqual(len(views), len(char_photos.VIEWS))  # a regenerated photo keeps its slot's view
        for text in (*views, char_photos._SAME_CREATURE):
            self.assertIsNone(HUMAN.search(text), text)
            self.assertNotIn("clothes", text)
        self.assertIn("wearing exactly the same outfit as in image 1, or nothing if image 1 shows none", views[4])
        for view in views[5:]:  # the cafe and the street
            self.assertIn("empty", view)
            self.assertIn("nobody else in the picture", view)
        self.assertIn("no people and no other animals", char_photos._SAME_CREATURE)


class CreaturePhotoTests(unittest.TestCase):
    """generate() and regenerate() of a creature draft: its own prompts, the gate asked about a creature, and no face
    comparison (3D animation is the style whose faces a person's views are compared by)."""

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.renders, self.composes = [], []
        self.gate = Mock(return_value="")
        self.look_check = Mock(wraps=characters.check_look)
        self.ensure = Mock(return_value=None)
        for target in (patch.object(characters, "DIR", self.root / "characters"),
                       patch.object(characters, "DRAFTS_DIR", self.root / "drafts"),
                       patch.object(queue_manager, "_JOBS", []),
                       patch.object(queue_manager, "_save"),
                       patch.object(characters, "gate_models", return_value=None),
                       patch.object(characters, "check_photo", self.gate),
                       patch.object(characters, "check_look", self.look_check),
                       patch.object(char_photos.model_manager, "ensure", self.ensure),
                       patch.object(char_photos.engine_runtime, "engine", return_value=Mock()),
                       patch.object(char_photos.engine_runtime, "render", side_effect=self.render),
                       patch.object(char_photos.refs, "compose", side_effect=self.compose),
                       patch.object(char_photos.qc, "face_score",
                                    side_effect=AssertionError("a creature's face is never compared"))):
            stack.enter_context(target)

    def render(self, prompt, width, height, seed, steps, runtime=None, **options):
        self.renders.append(prompt)
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (seed % 200, 90, 40)).save(buffer, "PNG")
        return buffer.getvalue()

    def compose(self, client, images, prompt, width, height, out, seed, progress_cb=None, **options):
        self.composes.append((list(images), prompt))
        Image.new("RGB", (8, 8), (seed % 200, 60, 60)).save(out)
        return True, str(out)

    def creature(self, style=styles.ANIMATION_3D, look=LOOK) -> str:
        draft = characters.new_draft()
        characters.update_draft(draft, name="Chain Dog", gender="man", look=look, style=style)
        self.assertEqual(characters.update_draft(draft, kind=characters.CREATURE)["kind"], characters.CREATURE)
        return draft

    def assert_creature_gate(self, style_id):
        self.assertTrue(self.gate.call_args_list)
        for call in self.gate.call_args_list:
            self.assertEqual((call.args[1], call.kwargs), (style_id, {"kind": characters.CREATURE}))

    def test_a_creature_is_drawn_from_its_look_alone_in_every_animated_style(self):
        watercolor = styles.custom_id(styles.OTHER, "soft pastel watercolor")
        anime_ink = styles.custom_id(styles.ANIME, "ink outlines, muted colors")
        for style_id in (styles.ANIMATION_3D, styles.CARTOON, styles.ANIME, watercolor, anime_ink):
            style = styles.get(style_id)
            with self.subTest(style=style_id):
                self.renders.clear()
                self.composes.clear()
                self.gate.reset_mock()
                self.look_check.reset_mock()
                draft = self.creature(style_id)
                progress = []
                notes = char_photos.generate(draft, 3, False, lambda _done, text: progress.append(text))
                self.assertIn("3 reference photos ready", notes[0])
                self.assertEqual(len(characters.draft(draft)["photos"]), 3)
                self.assertEqual(self.renders,
                                 [prefs.styled(style.creature_anchor.format(look=DRAWN), "image", style_id)])
                self.assertIn(f": {DRAWN}. Alone, no people and no other animals.", self.renders[0])
                self.assertEqual([prompt for _, prompt in self.composes],
                                 [char_photos._SAME_CREATURE + view + style.creature_view
                                  for view in char_photos.CREATURE_VIEWS[:2]])
                for prompt in (*self.renders, *(prompt for _, prompt in self.composes)):
                    self.assertIsNone(HUMAN.search(prompt), prompt)
                self.assertIn("Photo 1: the character...", progress)
                self.assertNotIn("Photo 1: the person...", progress)
                self.look_check.assert_called_once_with(LOOK, kind=characters.CREATURE)
                self.assertEqual(self.gate.call_count, 3)  # each photo passed the creature gate at its first take
                self.assert_creature_gate(style_id)

    def test_a_person_keeps_the_person_words_and_the_face_check(self):
        draft = characters.new_draft()
        characters.update_draft(draft, name="Mia", gender="woman", look="Short red hair, freckles",
                                style=styles.ANIMATION_3D)
        self.assertEqual(characters.draft(draft)["kind"], characters.PERSON)
        with patch.object(char_photos.qc, "face_score", return_value=0.9) as score:
            notes = char_photos.generate(draft, 2, False, lambda *_: None)
        self.assertIn("2 reference photos ready", notes[0])
        style = styles.get(styles.ANIMATION_3D)
        self.assertEqual(self.renders, [prefs.styled(style.anchor.format(gender="woman", look="Short red hair, freckles"),
                                                     "image", styles.ANIMATION_3D)])
        self.assertEqual([prompt for _, prompt in self.composes],
                         [char_photos._SAME + char_photos.VIEWS[0] + style.view])
        score.assert_called_once()
        self.look_check.assert_called_once_with("Short red hair, freckles", kind=characters.PERSON)
        self.assertEqual({(call.args[1], call.kwargs["kind"]) for call in self.gate.call_args_list},
                         {(styles.ANIMATION_3D, characters.PERSON)})

    def test_a_creature_is_never_drawn_in_a_realistic_style(self):
        draft = self.creature()
        char_photos.generate(draft, 2, False, lambda *_: None)
        drawn = (len(self.renders), len(self.composes), self.ensure.call_count)
        real = characters.draft
        for style_id in (styles.REALISTIC, styles.custom_id(styles.REALISTIC, "film noir, deep shadows")):
            # a draft refuses this pair (characters.update_draft): a damaged draft or a job that slipped through
            with self.subTest(style=style_id), patch.object(characters, "draft",
                                                            side_effect=lambda d, s=style_id: {**real(d), "style": s}):
                for run in (lambda: char_photos.generate(draft, 3, False, lambda *_: None),
                            lambda: char_photos.generate(draft, 3, True, lambda *_: None),
                            lambda: char_photos.regenerate(draft, 1, lambda *_: None)):
                    with self.assertRaises(ValueError) as caught:
                        run()
                    self.assertEqual(str(caught.exception), characters.CREATURE_STYLE_MSG)
        self.assertEqual((len(self.renders), len(self.composes), self.ensure.call_count), drawn)  # nothing was drawn
        self.assertEqual(len(characters.draft(draft)["photos"]), 2)

    def test_a_creatures_photo_is_generated_again_as_the_same_creature(self):
        draft = self.creature()
        char_photos.generate(draft, 6, False, lambda *_: None)
        style = styles.get(styles.ANIMATION_3D)
        before = characters.draft(draft)["photos"]
        self.gate.reset_mock()
        char_photos.regenerate(draft, 5, lambda *_: None)  # the slot of the full-body view: its own outfit or none
        self.assertEqual(self.composes[-1], ([before[0]], char_photos._SAME_CREATURE + char_photos.CREATURE_VIEWS[4]
                                             + style.creature_view))
        after = characters.draft(draft)["photos"]
        self.assertEqual(after[:5], before[:5])
        self.assertNotEqual(after[5], before[5])
        char_photos.regenerate(draft, 0, lambda *_: None)  # the first photo: a frontal view of photo 2
        self.assertEqual(self.composes[-1], ([after[1]], char_photos._SAME_CREATURE + char_photos._FRONT
                                             + style.creature_view))
        self.assert_creature_gate(styles.ANIMATION_3D)
        # the only photo: a new creature drawn from the look
        only = self.creature(styles.CARTOON)
        char_photos.generate(only, 1, False, lambda *_: None)
        self.renders.clear()
        self.look_check.reset_mock()
        char_photos.regenerate(only, 0, lambda *_: None)
        cartoon = styles.get(styles.CARTOON)
        self.assertEqual(self.renders, [prefs.styled(cartoon.creature_anchor.format(look=DRAWN), "image",
                                                     styles.CARTOON)])
        self.look_check.assert_called_once_with(LOOK, kind=characters.CREATURE)

    def test_views_of_an_uploaded_creature_are_made_with_the_creature_words(self):
        draft = self.creature(styles.ANIME)
        source = self.root / "mine.png"
        Image.new("RGB", (512, 640), (200, 120, 40)).save(source)
        self.assertEqual(characters.add_upload(draft, source, True), (True, "Photo added."))
        mine = characters.draft(draft)["photos"][0]
        self.gate.reset_mock()
        char_photos.generate(draft, 3, True, lambda *_: None)
        self.assertEqual(self.renders, [])  # no new creature was drawn
        anime = styles.get(styles.ANIME)
        self.assertEqual(self.composes, [([mine], char_photos._SAME_CREATURE + view + anime.creature_view)
                                         for view in char_photos.CREATURE_VIEWS[:2]])
        self.look_check.assert_not_called()  # nothing is drawn from the look
        self.assert_creature_gate(styles.ANIME)

    def test_a_creature_the_gate_refuses_is_named_with_the_creature_words(self):
        draft = self.creature()
        self.gate.return_value = "The photo must show exactly one animal or creature."
        with self.assertRaises(ValueError) as caught:
            char_photos.generate(draft, 3, False, lambda *_: None)
        self.assertEqual(str(caught.exception), "No first photo passed the check (no person under 18). "
                                                "Change the look description and try again. Last try: The photo "
                                                "must show exactly one animal or creature.")  # why, for the owner
        self.assertEqual(len(self.renders), char_photos.MAX_TRIES)
        self.assertEqual(list((self.root / "drafts" / draft).glob("p_*.png")), [])
        # the first photo passes and every take of its view shows people: the view is left out
        self.gate.side_effect = ["", *["The photo must show the animal or creature alone, without people."]
                                 * char_photos.MAX_TRIES]
        notes = char_photos.generate(draft, 2, False, lambda *_: None)
        self.assertEqual(len(characters.draft(draft)["photos"]), 1)
        self.assertTrue(notes[-1].startswith("Photo 2 was left out: The photo must show the animal or creature "
                                             "alone, without people."), notes)
        # the only photo drawn again never passes
        self.gate.side_effect = None
        with self.assertRaises(ValueError) as caught:
            char_photos.regenerate(draft, 0, lambda *_: None)
        self.assertEqual(str(caught.exception),
                         "Photo 1 was not replaced: it did not pass the check (no person under 18).")
        # a human child in the picture stops the job, and the draft keeps its photo
        self.gate.return_value = "⛔ The photo appears to show a minor (estimated age 9). Only adults are allowed."
        with self.assertRaisesRegex(ValueError, "^⛔"):
            char_photos.generate(draft, 3, False, lambda *_: None)
        self.assertEqual(len(characters.draft(draft)["photos"]), 1)

    def test_a_look_that_names_a_minor_is_refused_before_anything_is_drawn(self):
        for look in ("A 12-year-old golden retriever wearing a red scarf", "A teen dragon with tiny wings"):
            with self.subTest(look=look):
                draft = self.creature(look=look)
                with self.assertRaisesRegex(ValueError, "^⛔ Only adult characters"):
                    char_photos.generate(draft, 2, False, lambda *_: None)
        self.assertEqual((self.renders, self.composes, self.ensure.call_count), ([], [], 0))


if __name__ == "__main__":
    unittest.main()
