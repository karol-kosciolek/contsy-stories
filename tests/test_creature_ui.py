# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Animal and creature characters on the pages: the creator's Character radio (app/ui_characters.py) goes with the
photo steps and Save, Edit restores it, the creator's words follow it and an animal or creature gets an animated style;
the character list names it; a Realistic film refuses it in a scene before any GPU time (app/timeline.py); the Styles
page says how it talks (app/ui_styles.py). Real drafts in temporary folders with the photo gates and the queue's worker
replaced: no GPU, no network, no app start."""
import json
import re
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from app import (characters, config, history, llm, queue_manager, scene_parts, style_library, styles, timeline,
                 ui_characters, ui_styles, ui_theme, universes)
from app.ui_characters import Creator
from tests.test_characters import _Library, picture
from tests.test_ui_characters import named

SKIP = gr.skip()
CORGI = "An adult anthropomorphic red-orange corgi, big pointed ears, fluffy tail, tiny blue hoodie, stands upright"
# what the creature photo gate's vision model answers for a photo of the corgi alone (llm.check_creature_photo)
ALONE = {"humans": 0, "youngest_human_age": 0, "creatures": 1, "baby": False, "caption": "a corgi in a blue hoodie"}
FORM = [f"form:{name}" for name in ui_characters.FORM_NAMES]
LIBRARY = [*FORM, "cards", "listing", "pick", "assets", "top"]
# step 2's heading and the "keep my own photos" box of a person, as the creator always showed them
PERSON_STEP = ("<div><h3 class='section-title'>2. Reference photos</h3><p class='section-sub'>Generate photos of one "
               "person from the look, then review them. Every photo must show one adult with one clear face.</p></div>")
KEEP_PERSON = ("Keep my own photos and add views of the first up to the number of photos (untick to generate a new "
               "person)")
REFUSED = ("Chain Dog is an animal or creature: choose an animated style for this film (Project → Style), or take "
           "Chain Dog out of this scene.")


class CreatureCreatorTests(_Library):
    def setUp(self):
        super().setUp()
        for patcher in (patch.object(ui_theme, "get_upload_folder", return_value=str(self.root / "cache")),
                        patch.object(llm, "check_creature_photo", return_value=dict(ALONE))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def step(self, out: dict, handler, *values, extra=()):
        """Run `handler` the way the page does: the tab's draft, jobs and display, then the form values."""
        return named(handler(out["draft"], out["watch"], out["shown"], *values), extra)

    def start(self) -> dict:
        return {"draft": "", "watch": {}, "shown": ui_characters._view("", {}, "", "")}

    def drafts(self) -> list:
        return sorted(path.name for path in characters.DRAFTS_DIR.iterdir()) if characters.DRAFTS_DIR.exists() else []

    def test_the_kind_goes_with_the_photo_steps_and_save_and_edit_restores_it(self):
        out = self.step(self.start(), ui_characters.generate_photos, "Chain Dog", "man", CORGI, 2, True, None,
                        styles.ANIMATION_3D, characters.CREATURE)
        self.assertIn("Waiting in the GPU queue", out["photo_status"])
        self.assertEqual(characters.draft(out["draft"])["kind"], characters.CREATURE)
        queue_manager._JOBS.clear()  # withdrawn: the corgi's own photos come instead

        photos = [str(picture(self.root / f"corgi{n}.png", color=(200, 110 + n, 40))) for n in range(2)]
        out = self.step(out, ui_characters.upload_photos, "Chain Dog", "man", CORGI, photos, True, None,
                        styles.ANIMATION_3D, characters.CREATURE)
        self.assertIn("2 photos added", out["photo_status"])
        llm.check_photo.assert_not_called()  # a person's gate (one adult, one face) is never asked
        out = self.step(out, ui_characters.use_preset, "Chain Dog", "man", CORGI, "hanna", None)
        saved = self.step(out, ui_characters.save_character, "Chain Dog", "man", CORGI, None, styles.ANIMATION_3D,
                          characters.CREATURE, extra=LIBRARY)
        self.assertIn("Character ‘Chain Dog’ saved with 2 photos", saved["checklist"])
        self.assertEqual(saved["form:kind"], characters.PERSON)  # the creator starts a new character: a person
        record = json.loads((self.root / "characters" / "chain_dog" / "character.json").read_text(encoding="utf-8"))
        self.assertEqual((record["kind"], record["style"], record["gender"]),
                         (characters.CREATURE, styles.ANIMATION_3D, "man"))
        self.assertEqual(characters.get("chain_dog")["kind"], characters.CREATURE)

        # the list: the style line names what it is, and so do the selected character's badges
        self.assertIn("<b>Chain Dog</b><br><span class='muted'>3D animation · Animal or creature</span>",
                      saved["listing"])
        self.assertIn(">Animal or creature<", ui_characters._character_assets("chain_dog"))

        opened = named(ui_characters.edit_character("yes", "chain_dog", "", {}, Creator()), [*FORM, "info"])
        self.assertEqual((opened["form:kind"], opened["form:gender"], opened["form:style"]["value"]),
                         (characters.CREATURE, "man", styles.ANIMATION_3D))
        self.assertNotIn("data-unsaved", opened["heading"])  # nothing changed: the kind is the saved one
        self.assertFalse(ui_characters._unsaved(characters.draft(opened["draft"]), "Chain Dog", CORGI))

        # its photos stay a creature's: changing the kind needs them removed first, and the status box says so
        person = self.step(opened, ui_characters.generate_photos, "Chain Dog", "man", CORGI, 4, False, None,
                           styles.ANIMATION_3D, characters.PERSON)
        self.assertIn("Remove the photos before changing what kind of character this is.", person["photo_status"])
        self.assertEqual(characters.draft(opened["draft"])["kind"], characters.CREATURE)
        self.assertEqual(queue_manager._JOBS, [])

    def test_a_person_keeps_the_creator_and_the_list_it_had(self):
        self.assertEqual(ui_characters.FORM_NAMES[-1], "kind")
        self.assertEqual(ui_characters._form()[-1], characters.PERSON)  # an empty creator makes a person
        profile, _ = characters.save_draft(self.ready_draft("Mia", photos=1))
        record = json.loads((self.root / "characters" / "mia" / "character.json").read_text(encoding="utf-8"))
        self.assertNotIn("kind", record)
        _gallery, listing, _names = ui_characters._rows()
        self.assertIn("<b>Mia</b><br><span class='muted'>Realistic</span>", listing)
        self.assertNotIn("Animal or creature", listing + ui_characters._character_assets("mia"))
        opened = named(ui_characters.edit_character("yes", "mia", "", {}, Creator()), [*FORM, "info"])
        self.assertEqual((opened["form:kind"], opened["form:gender"]), (characters.PERSON, "woman"))
        self.assertNotIn("data-unsaved", opened["heading"])
        # a photo step called without a kind (an older caller) keeps the draft's
        regen = self.step(opened, ui_characters.regenerate_photo, "Mia", "woman", profile["look"], 0)
        self.assertIn("Only generated photos can be generated again", regen["photo_status"])
        self.assertEqual(characters.draft(opened["draft"])["kind"], characters.PERSON)

    def test_refusals_of_the_draft_show_in_the_status_boxes(self):
        out = self.step(self.start(), ui_characters.generate_photos, "Chain Dog", "man", CORGI, 2, True, None,
                        styles.REALISTIC, characters.CREATURE)
        self.assertIn(characters.CREATURE_STYLE_MSG, out["photo_status"])
        self.assertEqual((out["draft"], self.drafts()), ("", []))  # the draft made for the refused step is gone
        self.assertEqual(queue_manager._JOBS, [])
        own = styles.custom_id(styles.REALISTIC, "film noir, high contrast")  # the owner's style based on Realistic
        out = self.step(out, ui_characters.upload_photos, "Chain Dog", "man", CORGI,
                        [str(picture(self.root / "corgi.png"))], True, None, own, characters.CREATURE)
        self.assertIn(characters.CREATURE_STYLE_MSG, out["shown"].photo_status)  # the same words: not redrawn
        self.assertEqual((out["draft"], self.drafts()), ("", []))

        # a person's draft with a photo: Save as an animal or creature is refused and nothing is written
        draft = self.ready_draft("Mia", photos=1)
        shown = ui_characters._view(draft, {}, "Mia", CORGI)
        refused = named(ui_characters.save_character(draft, {}, shown, "Mia", "woman", CORGI, None,
                                                     styles.ANIMATION_3D, characters.CREATURE), LIBRARY)
        self.assertIn("Remove the photos before changing what kind of character this is.", refused["checklist"])
        self.assertTrue(all(refused[name] == SKIP for name in LIBRARY))
        self.assertEqual(self.written(), [])
        self.assertEqual(characters.draft(draft)["kind"], characters.PERSON)

    def test_the_words_follow_the_kind_and_an_animal_needs_an_animated_style(self):
        gender, look, style, step, keep, consent = ui_characters._kind_view(characters.CREATURE, styles.REALISTIC)
        self.assertEqual(consent["label"], characters.CREATURE_CONSENT)  # an animal's photo needs no person's consent
        self.assertEqual(gender["choices"], [("Female voice (she)", "woman"), ("Male voice (he)", "man")])
        self.assertNotIn("value", gender)  # the values stay: only the labels say voice and pronoun
        self.assertEqual(look["placeholder"], "For example: an adult anthropomorphic red-orange corgi, big pointed "
                                              "ears, fluffy tail, blue hoodie")
        self.assertNotIn("value", look)  # what was typed stays
        self.assertEqual(style["value"], styles.ANIMATION_3D)  # Realistic draws no animals or creatures
        self.assertIn("A photo that shows someone under 18 is refused.", step)
        self.assertNotIn("one adult with one clear face", step)
        self.assertIn("generate a new animal or creature", keep["label"])
        for text in (*(label for label, _ in gender["choices"]), look["placeholder"], step, keep["label"]):
            self.assertIsNone(re.search(r"\b(?:man|woman|person)\b", text, re.I), text)
        own_realistic = styles.custom_id(styles.REALISTIC, "film noir, high contrast")
        self.assertEqual(ui_characters._kind_view(characters.CREATURE, own_realistic)[2]["value"],
                         styles.ANIMATION_3D)
        self.assertEqual(ui_characters._kind_view(characters.CREATURE, None)[2]["value"], styles.ANIMATION_3D)
        for animated in (styles.ANIMATION_3D, styles.CARTOON, styles.ANIME, styles.custom_id(styles.OTHER, "clay"),
                         styles.custom_id(styles.ANIME, "soft pastel watercolor")):
            self.assertEqual(ui_characters._kind_view(characters.CREATURE, animated)[2], SKIP, animated)

        gender, look, style, step, keep, consent = ui_characters._kind_view(characters.PERSON, styles.REALISTIC)
        self.assertEqual(consent["label"], characters.CONSENT)
        self.assertEqual(gender["choices"], [("Woman", "woman"), ("Man", "man")])
        self.assertEqual(look["placeholder"], "For example: late twenties, shoulder-length wavy auburn hair, green "
                                              "eyes, freckles, fair skin, slim build, casual style")
        self.assertEqual(style, SKIP)  # a person keeps any style
        self.assertEqual((step, keep["label"]), (PERSON_STEP, KEEP_PERSON))  # as the creator always said
        self.assertEqual(ui_theme.section("2. Reference photos", ui_characters.PHOTO_RULES[characters.PERSON]).value,
                         PERSON_STEP)
        self.assertEqual(ui_characters._kind_view("anything else", styles.CARTOON)[3], PERSON_STEP)


class CreaturePageTests(unittest.TestCase):
    """The Characters page built in gr.Blocks: the radio's place and the events that carry it."""

    @classmethod
    def setUpClass(cls):
        with patch.object(characters, "all_characters", return_value=[]), \
                patch.object(style_library, "all_custom", return_value=[]), \
                patch.object(universes, "all_universes", return_value=[]):
            with gr.Blocks() as cls.demo:
                top = gr.Dropdown([], allow_custom_value=True)
                universe = gr.Dropdown([], allow_custom_value=True)
                additions = gr.Dropdown([], multiselect=True, allow_custom_value=True)
                with gr.Tab("Characters", id="characters") as tab:
                    cls.built = ui_characters.build_characters_tab(tab, top, (universe, additions))
        cls.blocks = list(cls.demo.blocks.values())
        cls.creator = next(b for b in cls.blocks if getattr(b, "elem_id", None) == ui_characters.CREATOR)
        cls.kind = next(b for b in cls.blocks if getattr(b, "elem_id", None) == "char-kind")

    def inside(self, block, ancestor) -> bool:
        while block is not None:
            if block is ancestor:
                return True
            block = getattr(block, "parent", None)
        return False

    def by_label(self, label):
        return next(block for block in self.blocks if self.inside(block, self.creator)
                    and getattr(block, "label", None) == label)

    def handler(self, fn):
        return next(found for found in self.demo.fns.values() if found.fn is fn)

    @staticmethod
    def ids(blocks) -> list:
        return [block._id for block in blocks]

    def test_the_character_radio_opens_step_one_before_gender(self):
        self.assertIsInstance(self.kind, gr.Radio)
        self.assertEqual((self.kind.label, self.kind.value), ("Character", characters.PERSON))
        self.assertEqual(self.kind.choices, [("Person", "person"), ("Animal or creature", "creature")])
        order = [b for b in self.blocks if self.inside(b, self.creator)]
        steps = [self.kind, self.by_label("Gender"), self.by_label("Style"), self.by_label("Look"),
                 self.by_label("Number of photos")]
        positions = [order.index(block) for block in steps]
        self.assertEqual(positions, sorted(positions))
        step = self.kind.parent
        while "char-step" not in (getattr(step, "elem_classes", None) or []):
            step = step.parent
        self.assertIn("1. Look", step.children[0].value)  # the first thing in step 1
        self.assertTrue(self.inside(self.by_label("Gender"), step))
        self.assertEqual(self.by_label("Look").placeholder, ui_characters.LOOK_EXAMPLES[characters.PERSON])
        self.assertTrue(any(getattr(b, "value", None) == PERSON_STEP for b in order))  # step 2 as it always was

    def test_the_kind_goes_with_every_step_that_sends_the_form_and_comes_back_with_it(self):
        for fn in (ui_characters.generate_photos, ui_characters.regenerate_photo, ui_characters.upload_photos,
                   ui_characters.save_character):
            found = self.handler(fn)
            self.assertIs(found.inputs[-1], self.kind, fn)  # the last input: after the style
            self.assertEqual((found.concurrency_id, found.api_visibility), (ui_characters.SERIAL, "private"), fn)
        for fn in (ui_characters.design_voice, ui_characters.clone_voice, ui_characters.use_preset):
            self.assertNotIn(self.kind._id, self.ids(self.handler(fn).inputs), fn)  # they keep the draft's kind
        for fn in (ui_characters.save_character, ui_characters.start_over, ui_characters.edit_character):
            outputs = self.ids(self.handler(fn).outputs)
            self.assertIn(self.kind._id, outputs, fn)
            self.assertEqual(len(outputs), len(set(outputs)), fn)
        # every refusal or cancel sends one update (or a skip) per output
        self.assertEqual(len(ui_characters.start_over("no", "", {}, Creator())),
                         len(self.handler(ui_characters.start_over).outputs))
        self.assertEqual(len(ui_characters.edit_character("no", "", "", {}, Creator())),
                         len(self.handler(ui_characters.edit_character).outputs))
        leave, inputs, outputs = self.built
        self.assertIn(self.kind._id, self.ids(outputs))  # leaving the page empties it too
        self.assertEqual(len(leave("", {}, Creator())), len(outputs))

    def test_a_change_of_kind_rewords_the_creator(self):
        style = next(b for b in self.blocks if getattr(b, "elem_id", None) == "char-style")
        found = self.handler(ui_characters._kind_view)
        # the owner's choice, and Edit, Save or Discard filling the form
        self.assertEqual([tuple(target) for target in found.targets], [(self.kind._id, "change")])
        self.assertEqual(self.ids(found.inputs), [self.kind._id, style._id])
        gender, look, picked, step, keep, consent = found.outputs
        self.assertEqual((gender.label, look.label, consent.label), ("Gender", "Look", characters.CONSENT))
        self.assertIs(picked, style)
        self.assertEqual((step.value, keep.label), (PERSON_STEP, KEEP_PERSON))
        self.assertEqual(found.api_visibility, "private")


class CreatureFilmTests(unittest.TestCase):
    """A film whose style is Realistic (or based on it) refuses an animal or creature in a scene before any GPU time:
    an ERROR in the scene's issues, and no job is queued; an animated film makes it."""

    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        enter = self.context.enter_context
        self.root = Path(enter(tempfile.TemporaryDirectory())).resolve()
        enter(patch.object(timeline, "DIR", self.root / "projects"))
        enter(patch("app.items.DIR", self.root / "items"))
        enter(patch.object(config, "OUTPUTS_DIR", self.root / "outputs"))
        enter(patch.object(config, "HISTORY_DIR", self.root / "history"))
        enter(patch.object(history, "_HISTORY_FILE", self.root / "history/history.jsonl"))
        enter(patch.object(queue_manager, "_STATE", self.root / "queue.json"))
        enter(patch.object(queue_manager, "FILES_DIR", self.root / "queue_files"))
        enter(patch.object(queue_manager, "_JOBS", []))
        enter(patch.object(queue_manager, "_STATE_INFO", {"next_id": 1, "paused": False}))
        enter(patch.object(queue_manager, "_save"))
        for memo in (timeline._MEMO, timeline._CLIP_INDEX):
            memo.clear()
            self.addCleanup(memo.clear)
        photo = str(self.root / "photo.png")
        self.profiles = {
            "chain_dog": {"slug": "chain_dog", "name": "Chain Dog", "gender": "man", "kind": characters.CREATURE,
                          "style": styles.ANIMATION_3D, "look": CORGI, "voice": "", "photos": [photo]},
            "ada": {"slug": "ada", "name": "Ada", "gender": "woman", "kind": characters.PERSON,
                    "style": styles.REALISTIC, "look": "Short red hair", "voice": "", "photos": [photo]},
            "kit": {"slug": "kit", "name": "Kit", "gender": "man", "kind": characters.PERSON,
                    "style": styles.ANIMATION_3D, "look": "Short dark hair", "voice": "", "photos": [photo]},
        }
        for name, fake in (("timeline_profile", self.profiles.get), ("get", self.profiles.get),
                           ("usable", self.profiles.get), ("all_characters", lambda: list(self.profiles.values()))):
            enter(patch.object(characters, name, side_effect=fake))
        enter(patch.object(timeline.voices, "all_voices", return_value=[]))
        # no picture, sound or model is ever made or loaded here
        for target, name in ((timeline.video_gen, "generate"), (timeline.model_manager, "ensure"),
                             (timeline.scene_sound, "prepare")):
            enter(patch.object(target, name, side_effect=AssertionError(f"{name} was called")))

    def save(self, scenes, previous=None, *, style=None):
        return timeline.save(previous.id if previous else None, previous.revision if previous else None,
                             "Corgi film", tuple(scenes), "Landscape 16:9", False, ["chain_dog", "ada", "kit"],
                             style=style)

    def errors(self, project, index=0) -> list:
        return [issue.text for issue in timeline.scene_issues(project, index) if issue.level == scene_parts.ERROR]

    def test_a_realistic_film_refuses_an_animal_in_a_scene_before_any_gpu_time(self):
        scenes = [timeline.Scene("Chain Dog runs across the yard.", 2, cast=("chain_dog",)),
                  timeline.Scene("Ada waves at the camera.", 2, cut=True, cast=("ada",)),
                  timeline.Scene("Chain Dog barks at the mailman.", 2, cut=True)]  # the cast from the description
        project = self.save(scenes)
        self.assertEqual(project.style, styles.REALISTIC)
        self.assertEqual(self.errors(project, 0), [REFUSED])
        self.assertEqual(self.errors(project, 1), [])  # a person in a Realistic film is fine
        self.assertEqual(self.errors(project, 2), [REFUSED])
        for kind, options, number in ((timeline.PREPARE, {}, 1), (timeline.PREPARE, {"scene_index": 0}, 1),
                                      (timeline.PREPARE, {"scene_index": 2}, 3), (timeline.PRODUCE, {}, 1)):
            with self.assertRaisesRegex(ValueError, re.escape(f"Scene {number}: {REFUSED}")):
                timeline.enqueue(project, kind, **options)
        self.assertEqual(queue_manager._JOBS, [])
        job = timeline.enqueue(project, timeline.PREPARE, scene_index=1)  # the person's scene is made
        self.assertEqual(job.params["scene_index"], 1)
        queue_manager._JOBS.clear()
        # a job that reaches the worker anyway stops before any picture or sound
        ok, _files, _notes, error = timeline.prepare_job({"project": project.id, "key": project.key,
                                                          "scene_index": 0}, lambda *_: None)
        self.assertFalse(ok)
        self.assertIn(REFUSED, error)

        noir = self.save(scenes, project, style=styles.custom_id(styles.REALISTIC, "film noir, high contrast"))
        self.assertEqual(self.errors(noir, 0), [REFUSED])  # the owner's style based on Realistic: the same
        with self.assertRaisesRegex(ValueError, re.escape(REFUSED)):
            timeline.enqueue(noir, timeline.PREPARE, scene_index=0)

        film = noir
        for animated in (styles.ANIMATION_3D, styles.CARTOON, styles.custom_id(styles.ANIME, "soft pastel watercolor")):
            film = self.save(scenes, film, style=animated)
            for index in range(len(scenes)):
                self.assertEqual(self.errors(film, index), [], (animated, index))
        job = timeline.enqueue(film, timeline.PREPARE, scene_index=0)
        self.assertEqual((job.kind, job.params["scene_index"]), (timeline.PREPARE, 0))

    def test_a_scene_with_an_animal_is_never_sexual_or_nude(self):
        scenes = [timeline.Scene("Chain Dog lies naked on the bed.", 2, cast=("chain_dog",)),
                  timeline.Scene("Ada lies naked on the bed.", 2, cut=True, cast=("ada",)),
                  timeline.Scene("Chain Dog runs on the beach next to Ada in a bikini.", 2, cut=True,
                                 cast=("chain_dog", "ada"))]
        film = self.save(scenes, style=styles.ANIMATION_3D)
        self.assertEqual(self.errors(film, 0), [characters.CREATURE_SEXUAL_MSG])
        self.assertEqual(self.errors(film, 1), [])  # people: the owner's choice, as before
        self.assertEqual(self.errors(film, 2), [])  # a bikini alone is no reason
        with self.assertRaisesRegex(ValueError, re.escape(characters.CREATURE_SEXUAL_MSG)):
            timeline.enqueue(film, timeline.PREPARE, scene_index=0)
        self.assertEqual(queue_manager._JOBS, [])
        ok, _files, _notes, error = timeline.prepare_job({"project": film.id, "key": film.key, "scene_index": 0},
                                                         lambda *_: None)
        self.assertFalse(ok)
        self.assertIn(characters.CREATURE_SEXUAL_MSG, error)

    def test_a_co_stars_nude_look_a_continued_shot_and_a_sexual_film_style_are_refused_with_an_animal(self):
        self.profiles["eve"] = {"slug": "eve", "name": "Eve", "gender": "woman", "kind": characters.PERSON,
                                "style": styles.ANIMATION_3D, "look": "nude, long red hair", "voice": "",
                                "photos": self.profiles["ada"]["photos"]}
        scenes = [timeline.Scene("Eve walks Chain Dog through the park.", 2, cast=("chain_dog", "eve")),
                  timeline.Scene("Chain Dog sits by the lake.", 2, cut=True, cast=("chain_dog",)),
                  timeline.Scene("Ada undresses and swims naked in the lake.", 2, cast=("ada",)),
                  timeline.Scene("Ada undresses and swims naked in the lake.", 2, cut=True, cast=("ada",))]
        film = timeline.save(None, None, "Lake", tuple(scenes), "Landscape 16:9", False,
                             ["chain_dog", "ada", "eve"], style=styles.ANIMATION_3D)
        self.assertEqual(self.errors(film, 0), [characters.CREATURE_SEXUAL_MSG])  # the co-star's look reaches it
        self.assertEqual(self.errors(film, 2), [characters.CREATURE_SEXUAL_MSG])  # continues from Chain Dog's shot
        self.assertEqual(self.errors(film, 3), [])  # a cut: people alone, the owner's choice
        with self.assertRaisesRegex(ValueError, re.escape(characters.CREATURE_SEXUAL_MSG)):
            timeline.enqueue(film, timeline.PREPARE, scene_index=2)
        pinup = timeline.save(film.id, film.revision, "Lake", (scenes[1],), "Landscape 16:9", False,
                              ["chain_dog"], style=styles.custom_id(styles.ANIMATION_3D, "sensual pin-up art"))
        self.assertEqual(self.errors(pinup, 0), [characters.CREATURE_SEXUAL_MSG])

    def test_the_note_on_another_style_leaves_out_the_refused_animal(self):
        realistic = self.save([timeline.Scene("Chain Dog and Kit sit on a bench.", 2, cast=("chain_dog", "kit"))])
        members = [self.profiles["chain_dog"], self.profiles["kit"]]
        error, note = timeline._style_issues(realistic, members)
        self.assertEqual((error.level, error.text), (scene_parts.ERROR, REFUSED))
        self.assertEqual(note.level, scene_parts.INFO)
        self.assertIn("this film (Realistic): Kit (3D animation).", note.text)  # the refused animal is not news
        [note] = timeline._style_issues(replace(realistic, style=styles.CARTOON), members)
        self.assertEqual(note.level, scene_parts.INFO)
        self.assertIn("Chain Dog (3D animation), Kit (3D animation)", note.text)  # drawn in another style: news only


class StylesPageCreatureTests(unittest.TestCase):
    def test_the_3d_style_says_how_an_animal_or_creature_talks(self):
        made = ui_styles._MADE_WITH[styles.ANIMATION_3D]
        self.assertIn("An animal or creature never gets MuseTalk: alone it talks through InfiniteTalk, otherwise its "
                      "lines play without mouth movement.", made)
        self.assertIn(made.replace("'", "&#x27;"), ui_styles.builtin_table())


if __name__ == "__main__":
    unittest.main()
