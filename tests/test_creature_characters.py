# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Animal and creature characters (app/characters.py, kind "creature"; app/llm.py check_creature_photo): where a
creature may be drawn, its adults-only look, its photo gate (no people, one creature, never a face check), the kind in
the creator's draft and its refusals, the save round trip, and its scenes without a person's words or a default
outfit. A person's character stays exactly as it was: its stored record, identity, gate calls and prompts.
Everything lives in a temporary folder and the vision and face models are replaced: no GPU, no network."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app import characters, llm, prefs, qc, queue_manager, styles, voices
from engine.media import audio
from tests.character_fixtures import store_character

PERSON, CREATURE = characters.PERSON, characters.CREATURE
ADULT = {"person": True, "age": 30, "gender": "woman", "caption": "a portrait"}
ALONE = {"humans": 0, "youngest_human_age": 0, "creatures": 1, "baby": False, "caption": "a blue hoodie, standing"}
CHAIN_DOG = ("An adorable adult anthropomorphic red-orange corgi dog with a fluffy white chest, big brown eyes and "
             "pointed ears, wearing a tiny blue hoodie, standing upright on two legs")
NOIR = styles.custom_id(styles.REALISTIC, "film noir, high contrast")  # a custom style of the Realistic family
PAPER = styles.custom_id(styles.OTHER, "paper cut-out, bright colors")
WATERCOLOR = styles.custom_id(styles.ANIME, "soft pastel watercolor")
ANIMATED = (styles.ANIMATION_3D, styles.CARTOON, styles.ANIME, PAPER, WATERCOLOR)


def picture(path: Path, color=(200, 120, 40)) -> Path:
    Image.new("RGB", (512, 640), color).save(path)
    return path


def speech(path: Path, seconds: float, rate: int = 48000) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    wave = (0.3 * np.sin(2 * np.pi * 120 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    audio.write_wav(path, wave[None, :], rate)
    return path


class KindTests(unittest.TestCase):
    def test_a_creature_is_drawn_in_every_style_family_but_realistic(self):
        self.assertEqual((PERSON, CREATURE, characters.KINDS), ("person", "creature", ("person", "creature")))
        self.assertTrue(characters.is_creature({"kind": "creature"}))
        for other in ({"kind": "person"}, {}, {"kind": "Creature"}, None, "creature", ["creature"]):
            self.assertFalse(characters.is_creature(other), other)
        for style in ANIMATED + (styles.custom_id(styles.ANIMATION_3D, "clay stop motion"),):
            self.assertTrue(characters.creature_style_ok(style), style)
        for style in (styles.REALISTIC, NOIR, "", None, "clay", styles.OTHER):
            self.assertFalse(characters.creature_style_ok(style), style)
        for style in (styles.REALISTIC, NOIR) + ANIMATED:  # the styles' own word on it (app/styles.py) agrees
            self.assertEqual(characters.creature_style_ok(style), styles.get(style).allows_creatures, style)
        self.assertEqual(characters.CREATURE_STYLE_MSG,
                         "An animal or creature needs an animated style: 3D animation, 2D cartoon, Anime or one of "
                         "your own animated styles.")


class LookTests(unittest.TestCase):
    def test_a_creatures_look_must_describe_a_fully_grown_animal(self):
        for look in ("a corgi puppy", "a fluffy kitten", "a bear cub", "a baby dragon", "a young fox", "a chibi cat"):
            with self.subTest(look=look):  # the owner's decision of 2026-11-16: an animal may be young
                self.assertEqual(characters.check_look(look, CREATURE), "")
        for look in ("a 5-year-old fox", "a teen wolf", "a teen wolf with a_puppy_face"):  # words that name a minor
            with self.subTest(look=look):
                self.assertEqual(characters.check_look(look, CREATURE), characters.CREATURE_ADULT_MSG)
                self.assertEqual(characters.check_look(look, kind=CREATURE), characters.CREATURE_ADULT_MSG)
        self.assertEqual(characters.CREATURE_ADULT_MSG,
                         "⛔ Only adult characters are supported: describe a fully grown animal or creature.")
        grown = (CHAIN_DOG, "A grey wolf in a baby blue scarf", "An owl with a baby-pink bow",
                 "A pug with fawn-colored fur", "A rooster with chicken feet", "A cat in a cubist shirt",
                 "An old tomcat", "An elegant cat in a red dress and kitten heels")
        for look in grown:
            with self.subTest(look=look):
                self.assertEqual(characters.check_look(look, CREATURE), "")
        self.assertIn("what animal or creature it is", characters.check_look("  ", CREATURE))
        self.assertIn("at most 600", characters.check_look("x" * 601, CREATURE))

    def test_a_persons_look_is_checked_exactly_as_before(self):
        self.assertEqual(characters.check_look("Red hair, a puppy tattoo on her arm, young-looking"), "")
        self.assertEqual(characters.check_look("Red hair", kind=PERSON), "")
        self.assertEqual(characters.check_look("a 15-year-old girl with long hair"),
                         "⛔ Only adult characters are supported. This description appears to refer to a minor.")
        self.assertEqual(characters.check_look(" "), "Describe how the character looks: hair, eyes, skin tone, build "
                                                     "and style.")

    def test_young_words_are_found_after_the_same_normalisation_as_the_minors_guard(self):
        self.assertTrue(llm.refers_to_young("A KITTEN"))
        self.assertTrue(llm.refers_to_young("", None, "a ｐｕｐｐｙ"))
        self.assertTrue(llm.refers_to_young("a kawaii kids mascot"))
        self.assertFalse(llm.refers_to_young("a baby blue jacket", "baby-pink boots", "fawn colour"))
        self.assertFalse(llm.refers_to_young(CHAIN_DOG))


class CreatureGateTests(unittest.TestCase):
    """characters.check_photo / same_person with the vision and face models replaced."""

    def setUp(self):
        self.vision = Mock(return_value=ALONE)
        self.person = Mock(return_value=ADULT)
        self.counted = Mock(return_value=1)
        self.compared = Mock(return_value=0.9)
        for target in (patch.object(llm, "check_creature_photo", self.vision),
                       patch.object(llm, "check_photo", self.person),
                       patch.object(qc, "count_faces", self.counted),
                       patch.object(qc, "face_score", self.compared)):
            target.start()
            self.addCleanup(target.stop)

    def gate(self, answer, style=styles.ANIMATION_3D):
        self.vision.return_value = answer
        return characters.check_photo("dog.png", style, kind=CREATURE)

    def test_the_creature_gate_refuses_people_minors_and_anything_but_one_creature(self):
        cases = ((None, "The photo could not be checked. Try again."),
                 ({**ALONE, "humans": 1, "youngest_human_age": 15},
                  "⛔ The photo appears to show a minor (estimated age 15). Only adults are allowed."),
                 ({**ALONE, "humans": 2, "youngest_human_age": 17, "creatures": 0},
                  "⛔ The photo appears to show a minor (estimated age 17). Only adults are allowed."),
                 ({**ALONE, "humans": 1, "youngest_human_age": 18},
                  ""),  # adults with it: allowed
                 ({**ALONE, "baby": True}, ""),  # a cute grown animal kept failing as a baby: no refusal
                 ({**ALONE, "creatures": 0}, ""),  # nothing but a minor is refused (the owner's decision)
                 ({**ALONE, "creatures": 2}, ""),
                 (ALONE, ""))
        for style in ANIMATED:
            for answer, expected in cases:
                with self.subTest(style=style, answer=answer):
                    self.assertEqual(self.gate(answer, style), expected)
            self.assertEqual(self.vision.call_args.args, ("dog.png", style))
        # never the person's gate, never a face count or comparison, a 3D style's faces included
        self.person.assert_not_called()
        self.counted.assert_not_called()
        self.assertEqual(characters.same_person("dog.png", "first.png", styles.ANIMATION_3D, kind=CREATURE), "")
        self.compared.assert_not_called()

    def test_an_unreadable_answer_is_asked_again(self):
        self.vision.side_effect = [None, ALONE]
        self.assertEqual(characters.check_photo("dog.png", styles.ANIMATION_3D, kind=CREATURE), "")
        self.vision.side_effect = [None, None]
        self.assertEqual(characters.check_photo("dog.png", styles.ANIMATION_3D, kind=CREATURE),
                         "The photo could not be checked. Try again.")
        self.vision.side_effect = None

    def test_a_creatures_photo_in_a_realistic_style_is_refused_unseen(self):
        for style in (styles.REALISTIC, NOIR):
            self.assertEqual(self.gate(ALONE, style), characters.CREATURE_STYLE_MSG)
        self.vision.assert_not_called()

    def test_a_persons_photo_passes_the_gate_it_always_did(self):
        self.assertEqual(characters.check_photo("me.png"), "")
        self.assertEqual(self.person.call_args.args, ("me.png",))  # the realistic call, word for word
        self.assertEqual(self.person.call_args.kwargs, {})
        self.assertEqual(characters.check_photo("me.png", styles.ANIMATION_3D, kind=PERSON), "")
        self.assertEqual(self.person.call_args.kwargs, {"style": styles.ANIMATION_3D})
        self.assertEqual(self.counted.call_count, 2)
        self.assertEqual(characters.same_person("me.png", "first.png", styles.ANIMATION_3D), "")
        self.compared.assert_called_once()
        self.vision.assert_not_called()


class CreatureVisionTests(unittest.TestCase):
    """llm.check_creature_photo with the vision call replaced."""

    @staticmethod
    def answer(**fields) -> str:
        return json.dumps({"humans": 0, "youngest_human_age": 0, "creatures": 1, "baby": False,
                           "caption": "A blue hoodie.", **fields})

    def test_the_vision_model_counts_humans_and_creatures(self):
        calls = []

        def vision(path, prompt, max_tokens, schema=None, size=512):
            calls.append((path, prompt, max_tokens, schema, size))
            return "Here it is: " + self.answer()

        with patch.object(llm, "_vision", side_effect=vision):
            for style in (styles.ANIMATION_3D, styles.CARTOON, styles.ANIME, WATERCOLOR, PAPER):
                self.assertEqual(llm.check_creature_photo("dog.png", style),
                                 {"humans": 0, "youngest_human_age": 0, "creatures": 1, "baby": False,
                                  "caption": "A blue hoodie"})
        path, prompt, max_tokens, schema, size = calls[0]
        self.assertEqual((path, max_tokens, size), ("dog.png", 160, 448))  # the same call as the person's gate
        self.assertIs(schema, llm._CREATURE_SCHEMA)
        self.assertEqual(set(schema["required"]), {"humans", "youngest_human_age", "creatures", "baby", "caption"})
        for words in ("anthropomorphic animal", "is NOT a human", "cat ears", "IS a human", "youngest", "real, 3D or "
                      "drawn", "3D animated film", "newborn or baby animal", "even if it looks cute"):
            self.assertIn(words, prompt)
        mediums = [prompt for _, prompt, *_ in calls]
        self.assertIn("2D cartoon, hand-drawn animation", mediums[1])
        self.assertIn("2D anime, hand-drawn cel animation", mediums[2])
        self.assertIn("2D anime, hand-drawn cel animation", mediums[3])  # a custom style: its base's medium...
        self.assertNotIn("watercolor", mediums[3])                       # ...never the owner's words
        self.assertIn("stylized animated film", mediums[4])

    def test_an_answer_of_the_wrong_type_or_out_of_range_is_no_answer(self):
        wrong = (dict(humans=-1), dict(humans=21), dict(humans=True), dict(humans="0"), dict(humans=1.0),
                 dict(humans=None), dict(creatures=-1), dict(creatures=21), dict(creatures=False), dict(creatures="1"),
                 dict(humans=1, youngest_human_age=-1), dict(humans=1, youngest_human_age=131),
                 dict(humans=1, youngest_human_age=True), dict(humans=1, youngest_human_age="30"),
                 dict(humans=1, youngest_human_age=None), dict(baby=None), dict(baby="no"), dict(baby=0))
        for fields in wrong:
            with self.subTest(fields=fields), patch.object(llm, "_vision", return_value=self.answer(**fields)):
                self.assertIsNone(llm.check_creature_photo("dog.png", styles.CARTOON))
        for raw in (None, "", "no JSON here", "[1, 2]", '{"humans": 0', "{not json}"):
            with self.subTest(raw=raw), patch.object(llm, "_vision", return_value=raw):
                self.assertIsNone(llm.check_creature_photo("dog.png", styles.CARTOON))
        # the age matters only when a human is there, and the caption is optional
        with patch.object(llm, "_vision", return_value=json.dumps({"humans": 0, "youngest_human_age": -5,
                                                                     "creatures": 1, "baby": False})):
            self.assertEqual(llm.check_creature_photo("dog.png", styles.CARTOON),
                             {"humans": 0, "youngest_human_age": 0, "creatures": 1, "baby": False, "caption": ""})
        with patch.object(llm, "_vision", return_value=json.dumps({"humans": 0, "youngest_human_age": 0,
                                                                     "creatures": 1})):
            self.assertIsNone(llm.check_creature_photo("dog.png", styles.CARTOON))  # no word on a baby: no answer
        with patch.object(llm, "_vision", return_value=self.answer(humans=20, youngest_human_age=34, creatures=0,
                                                                   caption=7)):
            self.assertEqual(llm.check_creature_photo("dog.png", styles.CARTOON),
                             {"humans": 20, "youngest_human_age": 34, "creatures": 0, "baby": False, "caption": ""})
        with patch.object(llm, "_vision", return_value=self.answer(caption="a 15-year-old in lingerie")):
            self.assertEqual(llm.check_creature_photo("dog.png", styles.CARTOON)["caption"], "")  # blocked: dropped

    def test_the_persons_photo_gate_is_word_for_word_what_it_was(self):
        self.assertEqual(llm._PHOTO_PROMPT, (
            "This photo is a reference photo of a character. Return JSON with: person - true if "
            "one real-looking human is clearly visible; age - your best estimate of that person's age in "
            "years; gender - woman or man; caption - one English sentence of at most 30 words "
            "describing ONLY the clothing, pose, facial expression, framing (close-up, half body or "
            "full body), background and lighting. Do not describe the face, hair, skin, eyes, body "
            "shape or age in the caption."))
        self.assertEqual(llm._PHOTO_SCHEMA, {
            "type": "object",
            "properties": {"person": {"type": "boolean"}, "age": {"type": "integer"},
                           "gender": {"type": "string", "enum": ["woman", "man"]},
                           "caption": {"type": "string"}},
            "required": ["person", "age", "gender", "caption"]})
        with patch.object(llm, "_vision", return_value=json.dumps(ADULT)) as vision:
            self.assertEqual(llm.check_photo("me.png")["age"], 30)
        self.assertEqual(vision.call_args.args, ("me.png", llm._PHOTO_PROMPT, 160))
        self.assertEqual(vision.call_args.kwargs, {"schema": llm._PHOTO_SCHEMA, "size": 448})


class StoredKindTests(unittest.TestCase):
    """Profiles and identities of stored characters (tests/character_fixtures.py writes them)."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.library = Path(folder.name) / "characters"
        target = patch.object(characters, "DIR", self.library)
        target.start()
        self.addCleanup(target.stop)

    def test_a_character_saved_before_kinds_is_a_person_with_the_identity_it_had(self):
        record = store_character(self.library, {"name": "Mia", "slug": "mia", "look": "Short red hair"}, photos=2,
                                 voice=b"RIFF-mia")
        self.assertNotIn("kind", record)
        profile = characters.get("mia")
        self.assertEqual(profile["kind"], PERSON)
        self.assertFalse(characters.is_creature(profile))
        for found in (characters.usable("mia"), characters.timeline_profile("mia"), characters.all_characters()[0]):
            self.assertEqual(found["kind"], PERSON)
        folder = self.library / "mia"
        self.assertEqual(characters.identity(profile), {
            "format": 2, "slug": "mia", "name": "Mia", "gender": "woman", "look": "Short red hair",
            "photos": [characters._digest(folder / "photos" / name) for name in ("01.png", "02.png")],
            "voice": characters._digest(folder / "voice" / "sample.wav")})
        store_character(self.library, {"name": "Odd", "slug": "odd", "kind": "dragon"})  # an unknown kind
        self.assertEqual(characters.get("odd")["kind"], PERSON)
        self.assertNotIn("kind", characters.identity(characters.get("odd")))

    def test_an_older_reference_character_is_a_person_and_keeps_its_stored_identity(self):
        folder = self.library / "rose"
        folder.mkdir(parents=True)
        portrait = picture(folder / "preview.png")
        stored = {"name": "Rose", "slug": "rose", "gender": "woman", "look": "red hair", "status": "reference",
                  "portrait_sha256": characters._digest(portrait), "voice": "", "created": 1}
        (folder / "character.json").write_text(json.dumps(stored), encoding="utf-8")
        rose = characters.get("rose")
        self.assertEqual((rose["kind"], rose["legacy"]), (PERSON, "reference"))
        self.assertEqual(characters.identity(rose), stored)

    def test_a_creatures_profile_and_identity_carry_its_kind(self):
        data = {"name": "Chain Dog", "slug": "chain_dog", "gender": "man", "look": CHAIN_DOG,
                "style": styles.ANIMATION_3D}
        store_character(self.library, data, photos=1)
        as_person = characters.identity(characters.get("chain_dog"))
        store_character(self.library, {**data, "kind": CREATURE}, photos=1)
        dog = characters.get("chain_dog")
        self.assertEqual((dog["kind"], dog["style"]), (CREATURE, styles.ANIMATION_3D))
        self.assertTrue(characters.is_creature(dog))
        self.assertTrue(characters.is_creature(characters.usable("chain_dog")))
        self.assertEqual(characters.identity(dog), {**as_person, "kind": CREATURE})  # its own clip keys


class _Creator(unittest.TestCase):
    """The creator on drafts in a temporary folder: characters, drafts, voices and prefs there, the queue in memory,
    the vision gates replaced and the face models counted (a creature is never face-checked)."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        builtin = self.root / "assets"
        builtin.mkdir()
        speech(builtin / "rex.wav", 4.0)
        (builtin / "voices.json").write_text(json.dumps({"voices": [
            {"name": "Rex", "file": "rex.wav", "gender": "male", "description": "40s, warm"}]}), encoding="utf-8")
        self.creature_gate = Mock(return_value=ALONE)
        self.person_gate = Mock(return_value=ADULT)
        self.count_faces = Mock(return_value=1)
        self.face_score = Mock(return_value=0.9)
        for target in (patch.object(characters, "DIR", self.root / "characters"),
                       patch.object(characters, "DRAFTS_DIR", self.root / "drafts"),
                       patch.object(voices, "BUILTIN_DIR", builtin),
                       patch.object(voices, "MANIFEST", builtin / "voices.json"),
                       patch.object(voices, "USER_DIR", self.root / "voices"),
                       patch.object(voices, "measure_pitch", return_value=120.0),
                       patch.object(prefs, "_FILE", self.root / "prefs.json"),
                       patch.object(characters, "gate_models", return_value=None),
                       patch.object(characters.trash, "move", side_effect=AssertionError("no Recycle Bin in a test")),
                       patch.object(queue_manager, "_JOBS", []),
                       patch.object(queue_manager, "_save"),
                       patch.object(llm, "_vision", side_effect=AssertionError("no vision model in a test")),
                       patch.object(llm, "check_creature_photo", self.creature_gate),
                       patch.object(llm, "check_photo", self.person_gate),
                       patch.object(qc, "count_faces", self.count_faces),
                       patch.object(qc, "face_score", self.face_score)):
            target.start()
            self.addCleanup(target.stop)

    def creature_draft(self, look: str = CHAIN_DOG, style: str = styles.ANIMATION_3D) -> str:
        draft = characters.new_draft()
        characters.update_draft(draft, name="Chain Dog", gender="man", look=look, style=style, kind=CREATURE)
        return draft

    def upload(self, draft: str, name: str = "dog.png", color=(200, 120, 40)):
        return characters.add_upload(draft, picture(self.root / name, color), True)

    def ready_creature(self, photos: int = 2) -> str:
        draft = self.creature_draft()
        for index in range(photos):
            self.assertEqual(self.upload(draft, f"dog{index}.png", (200, 120 + index, 40)), (True, "Photo added."))
        ok, message = characters.use_preset_voice(draft, "rex")
        self.assertTrue(ok, message)
        return draft

    def manifest(self, draft: str) -> dict:
        return json.loads((self.root / "drafts" / draft / "draft.json").read_text(encoding="utf-8"))

    @staticmethod
    def force(draft: str, photo_gate: str = "", **fields) -> None:
        """Write a draft state the form never makes (an older or damaged draft)."""
        data = characters._manifest(draft)
        data.update(fields)
        if photo_gate:
            data["photos"][0]["gate"] = photo_gate
        characters._write_manifest(data)

    def written(self) -> list:
        return sorted(p.relative_to(self.root).as_posix() for p in (self.root / "characters").rglob("*")
                      if p.is_file())

    def assert_no_face_checks(self):
        self.count_faces.assert_not_called()
        self.face_score.assert_not_called()
        self.person_gate.assert_not_called()


class CreatureDraftTests(_Creator):
    def test_a_new_draft_is_a_person_and_becomes_a_creature_only_in_an_animated_style(self):
        draft = characters.new_draft()
        self.assertEqual((characters.draft(draft)["kind"], self.manifest(draft)["kind"]), (PERSON, PERSON))
        for fields in ({"kind": CREATURE}, {"kind": CREATURE, "style": NOIR}):  # the draft is Realistic
            with self.subTest(fields=fields), self.assertRaises(ValueError) as refused:
                characters.update_draft(draft, name="Chain Dog", **fields)
            self.assertEqual(str(refused.exception), characters.CREATURE_STYLE_MSG)
        view = characters.draft(draft)
        self.assertEqual((view["kind"], view["style"], view["name"]), (PERSON, styles.REALISTIC, ""))  # untouched
        view = characters.update_draft(draft, name="Chain Dog", kind=CREATURE, style=styles.ANIMATION_3D)
        self.assertEqual((view["kind"], view["style"], view["name"]), (CREATURE, styles.ANIMATION_3D, "Chain Dog"))
        for style in (styles.REALISTIC, NOIR):
            with self.subTest(style=style), self.assertRaises(ValueError) as refused:
                characters.update_draft(draft, style=style)
            self.assertEqual(str(refused.exception), characters.CREATURE_STYLE_MSG)
        self.assertEqual(characters.update_draft(draft, style=PAPER)["style"], PAPER)
        with self.assertRaisesRegex(ValueError, "Choose a person or an animal or creature"):
            characters.update_draft(draft, kind="dragon")
        view = characters.update_draft(draft, kind=PERSON, style=styles.REALISTIC)  # no photos yet: back to a person
        self.assertEqual((view["kind"], view["style"]), (PERSON, styles.REALISTIC))

    def test_the_kind_changes_only_while_the_draft_has_no_photos_and_makes_none(self):
        draft = self.creature_draft()
        self.assertEqual(self.upload(draft), (True, "Photo added."))
        with self.assertRaises(ValueError) as refused:
            characters.update_draft(draft, kind=PERSON, style=styles.REALISTIC)
        self.assertEqual(str(refused.exception), "Remove the photos before changing what kind of character this is.")
        self.assertEqual(characters.update_draft(draft, kind=CREATURE, name="Rex")["name"], "Rex")  # not a change
        characters.remove_photo(draft, 0)
        queue_manager._JOBS.append(queue_manager.Job(1, characters.PHOTO_KIND, "photos", {"draft": draft}))
        with self.assertRaisesRegex(ValueError, "Wait for the photos"):
            characters.update_draft(draft, kind=PERSON)
        queue_manager._JOBS.clear()
        self.assertEqual(characters.update_draft(draft, kind=PERSON)["kind"], PERSON)

    def test_a_creatures_upload_passes_the_creature_gate_and_is_never_compared_by_face(self):
        draft = self.creature_draft()
        self.assertEqual(self.upload(draft, "a.png"), (True, "Photo added."))
        self.assertEqual(self.upload(draft, "b.png", (10, 20, 30)), (True, "Photo added."))  # no same_person
        self.assertEqual(self.creature_gate.call_count, 2)
        self.assertEqual(self.creature_gate.call_args.args[1], styles.ANIMATION_3D)
        self.assertEqual([entry["gate"] for entry in self.manifest(draft)["photos"]], [CREATURE, CREATURE])
        refusals = ((dict(ALONE, humans=1, youngest_human_age=12), "⛔ The photo appears to show a minor"),
                    (None, "could not be checked"))
        for answer, reason in refusals:
            self.creature_gate.return_value = answer
            with self.subTest(answer=answer):
                ok, message = self.upload(draft, "c.png", (1, 2, 3))
                self.assertFalse(ok)
                self.assertIn(reason, message)
        self.assertEqual(len(characters.draft(draft)["photos"]), 2)
        self.assert_no_face_checks()

    def test_a_persons_upload_records_the_persons_gate(self):
        draft = characters.new_draft()
        self.assertEqual(self.upload(draft, "a.png"), (True, "Photo added."))
        self.assertEqual(self.upload(draft, "b.png", (1, 2, 3)), (True, "Photo added."))
        self.assertEqual([entry["gate"] for entry in self.manifest(draft)["photos"]], [PERSON, PERSON])
        self.assertEqual((self.person_gate.call_count, self.count_faces.call_count, self.face_score.call_count),
                         (2, 2, 1))  # the adult gate and one face each, the second compared with the first
        self.creature_gate.assert_not_called()

    def test_photo_jobs_carry_the_kind_and_draw_only_a_fully_grown_creature(self):
        draft = self.creature_draft(look="A 15-year-old corgi in a blue hoodie")
        with self.assertRaises(ValueError) as refused:
            characters.enqueue_photos(draft, 4)
        self.assertEqual(str(refused.exception), characters.CREATURE_ADULT_MSG)
        self.assertEqual(queue_manager._JOBS, [])
        characters.update_draft(draft, look=CHAIN_DOG)
        job = characters.enqueue_photos(draft, 4)
        self.assertEqual((job.params["kind"], job.params["style"], job.params["gender"]),
                         (CREATURE, styles.ANIMATION_3D, "man"))
        person = characters.new_draft()
        characters.update_draft(person, look="Short red hair")
        self.assertEqual(characters.enqueue_photos(person, 2).params["kind"], PERSON)
        with self.assertRaisesRegex(ValueError, "or generate a new character"):
            characters.enqueue_photos(person, 2, keep_own=True)
        # a generated photo lands with the gate of the draft's kind, and a new take of it keeps the kind
        generated = picture(characters.new_photo_path(draft))
        characters.replace_photos(draft, [(generated, "generated")])
        self.assertEqual(characters.draft_photos(draft)[0][1]["gate"], CREATURE)
        self.assertEqual(characters.enqueue_regenerate(draft, 0).params["kind"], CREATURE)
        characters.update_draft(draft, look="a teen dragon")
        with self.assertRaises(ValueError) as refused:  # the only photo: drawn again from the look
            characters.enqueue_regenerate(draft, 0)
        self.assertEqual(str(refused.exception), characters.CREATURE_ADULT_MSG)
        kept = characters.draft_photos(draft)[0][1]
        characters.replace_photos(draft, [kept])  # a photo that stays keeps its gate
        self.assertEqual(characters.draft_photos(draft)[0][1]["gate"], CREATURE)

    def test_a_creature_in_a_realistic_style_is_refused_everywhere(self):
        draft = self.creature_draft()
        generated = picture(characters.new_photo_path(draft))
        characters.replace_photos(draft, [(generated, "generated")])
        self.force(draft, style=styles.REALISTIC)  # a state update_draft never writes
        for queue in (lambda: characters.enqueue_photos(draft, 2), lambda: characters.enqueue_regenerate(draft, 0)):
            with self.assertRaises(ValueError) as refused:
                queue()
            self.assertEqual(str(refused.exception), characters.CREATURE_STYLE_MSG)
        self.assertEqual(self.upload(draft), (False, characters.CREATURE_STYLE_MSG))
        self.assertEqual(queue_manager._JOBS, [])
        self.creature_gate.assert_not_called()


class CreatureSaveTests(_Creator):
    def test_a_creature_is_saved_and_loaded_with_its_kind(self):
        profile, message = characters.save_draft(self.ready_creature())
        self.assertIsNotNone(profile, message)
        self.assertEqual(message, "Character ‘Chain Dog’ saved with 2 photos and its own voice.")
        record = json.loads((self.root / "characters" / "chain_dog" / "character.json").read_text(encoding="utf-8"))
        self.assertEqual((record["format"], record["kind"], record["style"], record["gender"], record["look"]),
                         (2, CREATURE, styles.ANIMATION_3D, "man", CHAIN_DOG))
        self.assertEqual((profile["kind"], profile["style"], profile["voice"]),
                         (CREATURE, styles.ANIMATION_3D, "character:chain_dog"))
        identity = characters.identity(profile)
        self.assertEqual((identity["kind"], identity["style"]), (CREATURE, styles.ANIMATION_3D))
        edit = characters.load_draft("chain_dog")
        view = characters.draft(edit)
        self.assertEqual((view["kind"], view["style"], view["origins"]), (CREATURE, styles.ANIMATION_3D,
                                                                          ["kept", "kept"]))
        self.assertEqual([entry["gate"] for _, entry in characters.draft_photos(edit)], [CREATURE, CREATURE])
        with self.assertRaisesRegex(ValueError, "Remove the photos"):
            characters.save_draft(edit, kind=PERSON)
        again, message = characters.save_draft(edit)
        self.assertIsNotNone(again, message)
        self.assertEqual(characters.identity(again), identity)  # saved again unchanged: its clips keep their keys
        self.assert_no_face_checks()

    def test_save_checks_the_kind_the_look_and_every_photos_gate_again(self):
        draft = self.ready_creature()
        cases = (({"look": "A 15-year-old fox in a hoodie"}, characters.CREATURE_ADULT_MSG),
                 ({"look": CHAIN_DOG, "style": styles.REALISTIC}, characters.CREATURE_STYLE_MSG),
                 ({"style": NOIR}, characters.CREATURE_STYLE_MSG),
                 ({"style": styles.ANIMATION_3D, "kind": "dragon"}, "Choose a person or an animal or creature."),
                 ({"kind": CREATURE, "photo_gate": PERSON},
                  "Some photos were checked for another kind of character. Remove them, then upload or generate them "
                  "again."),
                 ({"kind": PERSON, "style": styles.REALISTIC, "photo_gate": CREATURE},
                  "Some photos were checked for another kind of character. Remove them, then upload or generate them "
                  "again."))
        for fields, reason in cases:
            with self.subTest(fields=fields):
                self.force(draft, **fields)
                self.assertEqual(characters.save_draft(draft), (None, reason))
        self.assertEqual(self.written(), [])  # nothing reached the library

    def test_a_person_is_saved_with_the_fields_it_always_had(self):
        draft = characters.new_draft()
        characters.update_draft(draft, name="Mia", look="Short red hair")
        self.assertEqual(self.upload(draft), (True, "Photo added."))
        self.assertTrue(characters.use_preset_voice(draft, "rex")[0])
        profile, message = characters.save_draft(draft, kind=PERSON)
        self.assertIsNotNone(profile, message)
        self.assertEqual(set(profile["record"]), {"format", "name", "slug", "gender", "look", "created", "updated",
                                                  "photos", "voice"})  # no kind, no style
        self.assertEqual(profile["kind"], PERSON)
        self.assertNotIn("kind", characters.identity(profile))

    def test_a_draft_from_before_kinds_is_saved_as_a_person(self):
        draft = characters.new_draft()
        characters.update_draft(draft, name="Mia", look="Short red hair")
        self.assertEqual(self.upload(draft), (True, "Photo added."))
        self.assertTrue(characters.use_preset_voice(draft, "rex")[0])
        data = characters._manifest(draft)
        del data["kind"]
        for entry in data["photos"]:
            del entry["gate"]
        characters._write_manifest(data)
        self.assertEqual(characters.draft(draft)["kind"], PERSON)
        profile, message = characters.save_draft(draft)
        self.assertIsNotNone(profile, message)
        self.assertEqual(profile["kind"], PERSON)


class CreatureRenderTests(unittest.TestCase):
    DOG = {"name": "Chain Dog", "slug": "chain_dog", "gender": "man", "kind": CREATURE, "look": CHAIN_DOG,
           "photos": [__file__]}

    @staticmethod
    def render(ch: dict, text: str, style: str):
        with patch.object(characters.refs, "compose", return_value=(True, "frame.png")) as compose:
            result = characters.render(None, ch, text, 832, 480, Path("frame.png"), seed=1, style=style)
        return result, compose

    def test_a_creatures_scene_names_no_person_and_adds_no_outfit(self):
        for style in ANIMATED + (styles.custom_id(styles.ANIMATION_3D, "clay stop motion"),):
            with self.subTest(style=style):
                (ok, _), compose = self.render(self.DOG, "Chain Dog runs on the beach", style)
                self.assertTrue(ok)
                prompt = compose.call_args.args[2]
                self.assertTrue(prompt.startswith(characters._CREATURE_LEAD), prompt)
                self.assertIn("the creature and everything around it", prompt)
                self.assertTrue(prompt.endswith(" Chain Dog runs on the beach"), prompt)
                self.assertNotRegex(prompt, r"(?i)\b(?:person|man|woman|human|adult identity|wearing|swimsuit)\b")
                self.assertEqual(compose.call_args.kwargs, {"style": style})

    def test_a_creature_is_never_drawn_in_a_realistic_scene(self):
        for style in (styles.REALISTIC, NOIR):
            with self.subTest(style=style):
                (ok, message), compose = self.render(self.DOG, "Chain Dog in a cafe", style)
                self.assertFalse(ok)
                self.assertEqual(message, f"Chain Dog cannot be in a realistic scene. {characters.CREATURE_STYLE_MSG}")
                compose.assert_not_called()

    def test_a_persons_scene_is_word_for_word_what_it_was(self):
        mia = {"name": "Mia", "gender": "woman", "photos": [__file__]}
        lead = "Use the exact adult identity in image 1. Preserve the same face, hair and skin tone. "
        outfit = " Wearing a fitted black knit dress with long sleeves."
        for ch in (mia, {**mia, "kind": PERSON}):
            (ok, _), compose = self.render(ch, "Mia in a cafe", styles.REALISTIC)
            self.assertEqual(compose.call_args.args[2],
                             lead + "Make a single live-action scene. Mia in a cafe" + outfit)
            self.assertEqual(compose.call_args.kwargs, {})  # the call it always had
            (ok, _), compose = self.render(ch, "Mia in a cafe", styles.ANIMATION_3D)
            self.assertEqual(compose.call_args.args[2], lead + styles.get(styles.ANIMATION_3D).render + "Mia in a cafe"
                             + outfit)
            self.assertIn("the person and everything around them", compose.call_args.args[2])


class CreatureSafetyTests(_Creator):
    """A fully grown animal or creature with a grown voice, never in sexual or nude content."""

    def test_a_baby_answer_refuses_no_photo(self):
        self.creature_gate.return_value = {**ALONE, "baby": True}  # the owner's decision: cute grown animals passed
        self.assertEqual(characters.check_photo("dog.png", styles.ANIMATION_3D, kind=CREATURE), "")

    def test_sexual_or_nude_words_are_refused_for_an_animal_and_only_for_it(self):
        for look in ("an adult corgi, naked", "a seductive adult fox", "an adult dragon with sex appeal",
                     "nagi dorosły lis", "an adult cat_nude"):
            with self.subTest(look=look):
                self.assertEqual(characters.check_look(look, kind=CREATURE), characters.CREATURE_SEXUAL_MSG)
        self.assertEqual(characters.check_look("an adult corgi in a blue bikini", kind=CREATURE), "")  # clothing alone
        self.assertEqual(characters.check_look("late twenties, seductive smile"), "")  # a person: as before
        self.assertTrue(llm.sexual("Chain Dog lies naked on the bed"))
        self.assertFalse(llm.sexual("Chain Dog on the beach next to Ada in a bikini"))

    def test_an_animal_never_gets_into_a_sexual_scene(self):
        dog = {**CreatureRenderTests.DOG}
        with patch.object(characters.refs, "compose", return_value=(True, "frame.png")) as compose:
            ok, message = characters.render(None, dog, "Chain Dog lies naked on the bed", 832, 480,
                                            Path("frame.png"), seed=1, style=styles.ANIMATION_3D)
        self.assertEqual((ok, message), (False, characters.CREATURE_SEXUAL_MSG))
        compose.assert_not_called()

    def test_an_animals_voice_may_be_young_but_never_names_a_minor(self):
        for text in ("a squeaky puppy voice", "a baby dragon's high voice", "a cub's voice"):  # the owner's decision
            with self.subTest(text=text):
                self.assertEqual(characters.check_voice_description(text, CREATURE), "")
        self.assertIn("refer to a minor", characters.check_voice_description("a 12-year-old boy's voice", CREATURE))


class ReviewFixTests(_Creator):
    """Fixes from the review: an own style's sexual words, undress wording, a voice designed before the kind."""

    def test_an_own_style_with_sexual_words_never_draws_an_animal(self):
        pinup = styles.custom_id(styles.ANIMATION_3D, "sensual pin-up art, seductive poses")
        clean = styles.custom_id(styles.ANIMATION_3D, "clay stop motion, warm light")
        self.assertEqual(characters.creature_style_problem(pinup), characters.CREATURE_SEXUAL_MSG)
        self.assertEqual(characters.creature_style_problem(clean), "")
        self.assertEqual(characters.creature_style_problem(styles.REALISTIC), characters.CREATURE_STYLE_MSG)
        self.assertFalse(characters.creature_style_ok(pinup))
        draft = characters.new_draft()
        with self.assertRaisesRegex(ValueError, re.escape(characters.CREATURE_SEXUAL_MSG)):
            characters.update_draft(draft, name="Chain Dog", gender="man", look=CHAIN_DOG, style=pinup, kind=CREATURE)
        self.assertEqual(characters.check_photo("dog.png", pinup, kind=CREATURE), characters.CREATURE_SEXUAL_MSG)
        with patch.object(characters.refs, "compose", return_value=(True, "frame.png")) as compose:
            ok, message = characters.render(None, CreatureRenderTests.DOG, "Chain Dog sits on a sofa", 832, 480,
                                            Path("frame.png"), seed=1, style=pinup)
        self.assertEqual((ok, message), (False, characters.CREATURE_SEXUAL_MSG))
        compose.assert_not_called()

    def test_undress_wording_is_nude_content(self):
        for text in ("Eve walks Chain Dog on the beach without clothes", "Eve, unclothed, sits by the dog",
                     "Eve disrobes next to Chain Dog", "a bare-breasted woman beside the fox", "swimming in the buff",
                     "Ewa i Tom nadzy bawią się z Reksem", "Ewa bez ubrania", "nagą"):
            with self.subTest(text=text):
                self.assertTrue(llm.sexual(text))
        self.assertFalse(llm.sexual("Chain Dog in a blue hoodie runs through the park"))


class AssistantAnimalTests(unittest.TestCase):
    def test_the_prompt_writer_keeps_animals_as_animals_and_invents_no_speech(self):
        for kind in ("image", "video"):
            for style in (styles.REALISTIC, styles.ANIMATION_3D, styles.CARTOON, PAPER):
                with self.subTest(kind=kind, style=style):
                    text = llm.system_prompt(kind, style)
                    self.assertIn("Animals and creatures stay animals and creatures", text)
                    self.assertIn("describe every person as an adult", text)  # the rule for people stays
        self.assertIn("never invent speech", llm._SYSTEM["video"])


if __name__ == "__main__":
    unittest.main()
