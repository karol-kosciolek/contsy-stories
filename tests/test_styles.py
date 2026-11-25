# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""A film's visual style (app/styles.py): Realistic keeps every text it had, the other styles carry no realistic
words into any prompt, a Timeline project stores its style and a change marks its clips Outdated, characters are
drawn in their style and the face thresholds follow it. No model, no GPU and no live state is used."""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app import char_photos, characters, config, history, llm, prefs, qc, refs, styles, timeline, ui_timeline, video_gen
from engine.media import video

REALISTIC_WORDS = ("photoreal", "realistic", "live-action", "real scene", "real-looking", "real life")
OLD_VIDEO = ("ultra realistic, photorealistic real life footage, natural lighting, cinematic, highly detailed, "
             "sharp focus")
OLD_IMAGE = "ultra realistic photo, photorealistic, natural light, highly detailed, sharp focus"


def _film(path, seconds):
    with video.Writer(path, 30, 48, 32, preset="ultrafast", comment=video.AI_COMMENT) as out:
        for _ in range(30 * seconds):
            out.write(np.full((32, 48, 3), 90, dtype=np.uint8))


# Every text and model choice of the styles stored films and characters use, as they were when "2d" was the one 2D
# style (2026-11-16): a stored film keeps its look, keys and Ready clips only while these stay byte-identical.
PRESET_FIELDS = ("video", "image", "keep", "scene", "render", "anchor", "view", "writer_image", "writer_video",
                 "subject", "edit_lora", "edit_trigger", "image_lora", "video_lora", "musetalk", "video_experts",
                 "draws_start")
PRESET_PINS = {"realistic": "f07cb1124524cf4180d10dd47c5a7c4e0cdfde456662d908ef0c30401ca17b85",
               "3d": "78c4439ae4b838f27a4361967ae753c8c76c9f043b29216353f6cca57c6dfcf9",
               "2d": "465923e7d83053530c9daa6f63307d5fc3974c13db99bf4f1fe6b96b0b1c01cf"}
KEY_PINS = {"realistic": "6825fa540879dddd0329667c730030d68b03a6ecbb26d739b14db47341aeb815",
            "3d": "b83cc5341242fd499f2d21f29a457d0378d43dfe40513ddf94481a5f78a85a16",
            "2d": "ae2d29da840ef64d57f6d90aa9305838eef26c7334e01123304c853a19d088d3"}


def _pin(style) -> str:
    import hashlib

    blob = json.dumps({name: getattr(style, name) for name in PRESET_FIELDS}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class StoredStylePinTests(unittest.TestCase):
    def test_the_styles_stored_films_use_keep_every_text_model_and_key(self):
        for style_id, pin in PRESET_PINS.items():
            with self.subTest(style=style_id):
                self.assertEqual(_pin(styles.get(style_id)), pin)
                project = timeline.Project("0" * 32, 1, "x", (timeline.Scene("A cup", 1),), style=style_id)
                self.assertEqual(project.key, KEY_PINS[style_id])


class PresetTests(unittest.TestCase):
    def test_the_anime_style_is_the_former_2d_style_under_a_new_name(self):
        anime = styles.get(styles.ANIME)
        self.assertEqual((styles.ANIME, styles.CARTOON_2D, anime.label), ("2d", "2d", "Anime"))
        self.assertEqual((anime.edit_lora, anime.image_lora, anime.video_experts), (("anime-2000", 1.0),
                                                                                    ("sk_anime_style", 0.8),
                                                                                    styles.ANISORA_ID))
        self.assertTrue(anime.draws_start)
        self.assertFalse(anime.musetalk)

    def test_the_2d_cartoon_is_a_drawn_cartoon_without_anime(self):
        cartoon = styles.get(styles.CARTOON)
        self.assertEqual(cartoon.label, "2D cartoon")
        texts = [getattr(cartoon, name) for name in PRESET_FIELDS[:10]]
        for text in texts:
            self.assertNotIn("anime", text.lower())
            self.assertNotIn("western", text.lower())
            self.assertFalse(llm.blocked(text))
            self.assertFalse(llm.refers_to_minor(text))
        self.assertIsNone(cartoon.edit_lora)
        self.assertIsNone(cartoon.image_lora)
        self.assertEqual(cartoon.video_experts, styles.ANISORA_ID)
        self.assertTrue(cartoon.draws_start)
        self.assertFalse(cartoon.musetalk)
        self.assertIn("adult", cartoon.anchor)
        self.assertFalse(qc.rules(styles.CARTOON).detects)
        self.assertEqual(styles.for_video_text(cartoon.video).id, styles.CARTOON)

    def test_the_starting_points_are_frozen_and_their_film_texts_cannot_be_confused(self):
        self.assertEqual({key: base.medium for key, base in styles.BASES.items()},
                         {"realistic": "photorealistic live-action film",
                          "3d": "3D animated film, stylized 3D CGI animation",
                          "cartoon": "2D cartoon, hand-drawn animation",
                          "2d": "2D anime, hand-drawn cel animation",
                          "other": "stylized animated film"})
        videos = [style.video for style in styles.STYLES.values()]
        self.assertEqual(len(set(videos)), len(videos))
        tails = [f", {base.medium}, cinematic, highly detailed" for base in styles.BASES.values()]
        for tail in tails:
            self.assertFalse(any(other != tail and other.endswith(tail) for other in tails))
            self.assertFalse(any(video.endswith(tail) for video in videos))
        for key, base in styles.BASES.items():
            self.assertTrue(base.about, key)
            self.assertIn(base.family, qc.FACE_RULES)


class CustomStyleTests(unittest.TestCase):
    LOOK = "soft pastel watercolor, visible paper texture"

    def test_the_look_words_have_one_canonical_form(self):
        self.assertEqual(styles.clean_look("  soft   pastel watercolor,  paper texture.  "),
                         "soft pastel watercolor, paper texture")
        self.assertEqual(styles.clean_look(styles.clean_look("akwarela, ołówkowe kontury")),
                         "akwarela, ołówkowe kontury")  # Polish letters are allowed, and it is idempotent
        for bad, message in (("ab", "at least 3"), ("x" * 201, "at most 200"), ("ink: noir", "only letters"),
                             ("[1] ink", "only letters"), ("ink {look}", "only letters"), ('"ink"', "only letters"),
                             ("ink @mia", "only letters"), ("ink\nnoir", None)):
            with self.subTest(bad=bad):
                if message is None:
                    self.assertEqual(styles.clean_look(bad), "ink noir")  # a newline is only white space
                else:
                    with self.assertRaisesRegex(ValueError, message):
                        styles.clean_look(bad)

    def test_a_custom_style_takes_its_starting_points_models_and_puts_its_words_in_every_picture_text(self):
        for base_id, base in styles.BASES.items():
            with self.subTest(base=base_id):
                style_id = styles.custom_id(base_id, self.LOOK)
                self.assertEqual(style_id, f"custom:1:{base_id}:{self.LOOK}")
                custom = styles.get(style_id)
                self.assertTrue(styles.valid(style_id))
                self.assertEqual((custom.family, custom.custom, custom.about), (base_id, True, self.LOOK))
                for name in ("edit_lora", "edit_trigger", "image_lora", "video_lora", "musetalk", "video_experts",
                             "draws_start", "scene", "anchor", "view", "subject"):
                    self.assertEqual(getattr(custom, name), getattr(base, name), name)
                for name in ("video", "image", "keep", "render"):
                    self.assertIn(self.LOOK, getattr(custom, name), name)
                self.assertFalse(custom.realistic)
                self.assertEqual(styles.for_video_text(custom.video).id, style_id)  # what a clip records
                self.assertEqual(qc.rules(style_id), qc.FACE_RULES[base_id])
        self.assertEqual(styles.base_label(styles.custom_id(styles.ANIME, self.LOOK)), "Anime")
        self.assertEqual(styles.label(styles.custom_id(styles.OTHER, "x" * 60)), "Custom: " + "x" * 39 + "…")

    def test_the_version_1_format_is_frozen(self):
        custom = styles.get("custom:1:other:ink and wash, 1920s print, A&B's look")
        self.assertEqual(custom.video, "ink and wash, 1920s print, A&B's look, stylized animated film, cinematic, "
                                       "highly detailed")
        self.assertEqual(custom.image, "ink and wash, 1920s print, A&B's look, stylized animated film still, highly "
                                       "detailed")
        for bad in ("custom:1:clay:ink", "custom:2:other:ink", "custom:1:other:", "custom:1:other: ink",
                    "custom:1:other:ink.", "custom:1:other:a:b", "custom:1:other", "custom:1:otherink"):
            with self.subTest(bad=bad):
                self.assertFalse(styles.valid(bad))
                self.assertEqual(styles.of(bad).id, styles.REALISTIC)
        looks = {styles.for_video_text(styles.get(styles.custom_id(base, "ink")).video).id for base in styles.BASES}
        self.assertEqual(len(looks), len(styles.BASES))  # the same words on five starting points: five styles

    def test_the_prompts_compositions_and_photos_use_a_custom_style(self):
        style_id = styles.custom_id(styles.ANIME, self.LOOK)
        custom = styles.get(style_id)
        self.assertEqual(prefs.styled("A cafe.", "image", style_id), f"A cafe, {custom.image}")
        self.assertIn(self.LOOK, refs.keep(style_id))
        self.assertIn(custom.writer_video, llm.system_prompt("video", style_id))


class StyleTextTests(unittest.TestCase):
    def test_realistic_keeps_the_exact_texts_of_the_studio_before_styles(self):
        self.assertEqual(prefs.STYLE, {"video": OLD_VIDEO, "image": OLD_IMAGE})
        self.assertEqual(prefs.styled("A cafe.", "image"), f"A cafe, {OLD_IMAGE}")
        self.assertTrue(refs._KEEP.startswith("Photorealistic photo, one coherent real scene. Every person"))
        self.assertEqual(refs.keep(), refs._KEEP)
        self.assertEqual(llm.system_prompt("image"), llm._SYSTEM["image"])
        # every assistant writes for Contsy: @names, quoted text seen in the picture, no style words of its own
        for kind in ("image", "video"):
            text = llm.system_prompt(kind)
            self.assertIn("@chain_dog", text)
            self.assertIn("displays the text", text)
            self.assertIn("do not write style words", text)
        self.assertIn("2.5 words per second", llm.system_prompt("video"))
        self.assertEqual(llm.characters_context([{"slug": "chain_dog", "kind": "creature", "look": "a corgi"},
                                                 {"slug": "ola", "gender": "woman"}]),
                         "Saved characters of the user's studio - when the idea mentions one, write it exactly as here "
                         "and do not describe its looks: @chain_dog (an animal or creature, not a human); "
                         "@ola (a woman).")
        self.assertEqual(llm.characters_context([]), "")
        self.assertEqual(llm.system_prompt("video", styles.REALISTIC), llm._SYSTEM["video"])
        self.assertIn("one coherent real scene", refs.cast_lead([{"name": "Ola", "gender": "woman"}]))
        self.assertTrue(char_photos.ANCHOR.startswith("Photo of an adult {gender}. {look}. Portrait photo"))
        self.assertEqual(styles.for_video_text(OLD_VIDEO).id, styles.REALISTIC)
        self.assertEqual(styles.for_video_text("").id, styles.REALISTIC)
        self.assertEqual(styles.for_video_text("cinematic, film grain").id, styles.REALISTIC)

    def test_animated_styles_carry_no_realistic_words_into_any_prompt(self):
        for style_id in (styles.ANIMATION_3D, styles.CARTOON_2D):
            look = styles.get(style_id)
            with self.subTest(style=style_id):
                texts = [look.video, look.image, look.keep, look.scene, look.render, look.anchor, look.view,
                         prefs.styled("A cafe", "image", style_id), refs.keep(style_id),
                         refs.edit_prompt("A woman [1] in a cafe", [1], False, False, style=style_id),
                         refs.cast_lead([{"name": "Ola", "gender": "woman"}], style=style_id),
                         llm.system_prompt("image", style_id), llm.system_prompt("video", style_id)]
                for text in texts:
                    self.assertFalse([word for word in REALISTIC_WORDS if word in text.lower()], text)
                self.assertEqual(styles.for_video_text(look.video).id, style_id)
                self.assertIn("adult", look.anchor)  # an animated character is drawn as an adult too

    def test_an_unknown_style_is_refused_and_a_damaged_one_reads_as_realistic(self):
        with self.assertRaisesRegex(ValueError, "valid style"):
            styles.get("watercolor")
        self.assertEqual(styles.of("watercolor").id, styles.REALISTIC)
        self.assertEqual(styles.of(None).id, styles.REALISTIC)
        self.assertEqual([value for _, value in styles.choices()], ["realistic", "3d", "cartoon", "2d"])
        self.assertFalse(styles.valid("other"))  # only a starting point of custom styles
        with self.assertRaisesRegex(ValueError, "valid style"):
            styles.get(styles.OTHER)
        self.assertEqual(styles.of("clay").id, styles.REALISTIC)

    def test_the_photo_gate_accepts_a_drawn_adult_only_in_an_animated_style(self):
        prompts = []

        def vision(_path, prompt, *_args, **_kwargs):
            prompts.append(prompt)
            return json.dumps({"person": True, "age": 30, "gender": "woman", "caption": "a plain top"})

        with patch.object(llm, "_vision", side_effect=vision):
            self.assertEqual(llm.check_photo("x.png")["age"], 30)
            llm.check_photo("x.png", style=styles.CARTOON_2D)
        self.assertIn("one real-looking human", prompts[0])
        self.assertNotIn("real-looking", prompts[1])
        self.assertIn("drawing", prompts[1])
        self.assertIn("age", prompts[1])  # the adult gate stays in every style


class StyleExpertTests(unittest.TestCase):
    def test_the_2d_style_runs_the_anisora_experts_with_their_merged_distillation(self):
        from app import engine_runtime

        files = {("anisora-2d", "anisora", "high_noise"): Path("high.safetensors"),
                 ("anisora-2d", "anisora", "low_noise"): Path("low.safetensors")}
        both = {"high": Path("high.safetensors"), "low": Path("low.safetensors")}
        with (patch.object(engine_runtime.model_manager, "model_path", side_effect=lambda *key, **_: files.get(key)),
              patch.object(engine_runtime.model_manager, "tier", return_value="full"),
              patch.dict(engine_runtime.config.SETTINGS["engine"], {"fp8_experts": "matching"})):
            self.assertEqual(engine_runtime.style_experts(styles.CARTOON_2D), both)
            self.assertEqual(engine_runtime.style_experts(styles.CARTOON_2D, lora_strength=0.5), {})
            self.assertEqual(engine_runtime.style_experts(styles.ANIMATION_3D), {})
            self.assertEqual(engine_runtime.style_experts(styles.REALISTIC), {})
            with patch.object(engine_runtime.model_manager, "tier", return_value="lite"):
                self.assertEqual(engine_runtime.style_experts(styles.CARTOON_2D), {})  # FP8 experts: Full set only
            with patch.dict(engine_runtime.config.SETTINGS["engine"], {"fp8_experts": "off"}):
                self.assertEqual(engine_runtime.style_experts(styles.CARTOON_2D), {})
        with (patch.object(engine_runtime.model_manager, "model_path",
                           side_effect=lambda *key, **_: files.get(key) if key[2] == "high_noise" else None),
              patch.object(engine_runtime.model_manager, "tier", return_value="full"),
              patch.dict(engine_runtime.config.SETTINGS["engine"], {"fp8_experts": "matching"})):
            self.assertEqual(engine_runtime.style_experts(styles.CARTOON_2D), {"high": Path("high.safetensors")})
        self.assertEqual(styles.get(styles.CARTOON_2D).video_experts, styles.ANISORA_ID)
        self.assertTrue(styles.get(styles.CARTOON_2D).draws_start)
        self.assertFalse(styles.get(styles.REALISTIC).draws_start)


    def test_a_custom_style_runs_the_models_and_add_ons_of_its_starting_point(self):
        from types import SimpleNamespace

        from app import engine_runtime

        watercolor = styles.custom_id(styles.ANIME, "soft pastel watercolor")
        clay = styles.custom_id(styles.OTHER, "claymation, plasticine texture")
        files = {("anisora-2d", "anisora", "high_noise"): Path("high.safetensors"),
                 ("anisora-2d", "anisora", "low_noise"): Path("low.safetensors"),
                 (styles.STYLE_MODEL_ID, "style_loras", "anime-2000"): Path("anime-2000.safetensors"),
                 (styles.STYLE_MODEL_ID, "style_loras", "sk_anime_style"): Path("sk_anime_style.safetensors")}
        edit = SimpleNamespace(text_encoder="encoder", transformer="transformer", lightning="lightning", vae="vae")
        with (patch.object(engine_runtime.model_manager, "model_path", side_effect=lambda *key, **_: files.get(key)),
              patch.object(engine_runtime.model_manager, "tier", return_value="full"),
              patch.dict(engine_runtime.config.SETTINGS["engine"], {"fp8_experts": "matching"}),
              patch.object(engine_runtime, "qwen_edit_files", return_value=edit)):
            self.assertEqual(engine_runtime.style_experts(watercolor),
                             {"high": Path("high.safetensors"), "low": Path("low.safetensors")})
            self.assertEqual(engine_runtime.style_loras(watercolor, "edit"), [(Path("anime-2000.safetensors"), 1.0)])
            self.assertEqual(engine_runtime.style_loras(watercolor, "image"),
                             [(Path("sk_anime_style.safetensors"), 0.8)])
            self.assertEqual((engine_runtime.style_experts(clay), engine_runtime.style_loras(clay, "edit")), ({}, []))
            by_text = engine_runtime.start_frame_files(styles.get(watercolor).video)  # what a clip records
            self.assertEqual(by_text, engine_runtime.start_frame_files(watercolor))
            self.assertEqual(by_text, ["encoder", "transformer", "lightning", Path("anime-2000.safetensors"), "vae"])
        with patch.object(engine_runtime.model_manager, "model_path", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "The Anime style LoRA"):
                engine_runtime.style_loras(watercolor, "edit")


class StyleSafetyTests(unittest.TestCase):
    def test_childlike_style_words_are_refused_in_every_form_and_adult_looks_are_not(self):
        for look in ("chibi, big heads", "chibis, big heads", "chibified characters", "chibi_style, pastel",
                     "baby faced characters", "baby face characters", "babyfaces", "baby-ish proportions", "babyish",
                     "child like proportions", "childlike", "kawaii kids", "super deformed style, big heads",
                     "super-deformed figures"):
            with self.subTest(look=look):
                self.assertEqual(llm.style_problem(styles.custom_id(styles.OTHER, look)), llm.STYLE_MINOR_MSG)
        for look in ("soft pastel watercolor", "claymation, plasticine models", "retro comic, halftone dots",
                     "noir, deep shadows, faces in profile"):
            with self.subTest(look=look):
                self.assertEqual(llm.style_problem(styles.custom_id(styles.OTHER, look)), "")
        self.assertEqual(llm.style_problem(styles.ANIME), "")  # a preset is never checked

    def test_a_composition_keeps_a_custom_styles_words_and_a_presets_prompt_as_it_was(self):
        look = "storybook watercolor, warm golden glow, pale skin tones, light tan paper"
        custom = styles.get(styles.custom_id(styles.OTHER, look))
        prompt = refs.edit_prompt(f"Anna with tan skin holds the lamp from [2], {custom.video}", [1, 2], False, False,
                                  style=custom.id)
        self.assertIn(f", {custom.video}", prompt)  # the owner's words, word for word
        self.assertNotIn("tan skin", prompt)  # the person's look still comes from the photo
        for preset in styles.STYLES.values():
            with self.subTest(style=preset.id):
                text = f"Anna with tan skin holds the lamp from [2], {preset.video}"
                body = refs.re.sub(r"\s{2,}", " ", refs._MARK.sub(lambda m: " from image 2" if (m.group(1) or m.group(2)) == "2" else "", text)).strip()
                self.assertIn(refs.neutral_look(body), refs.edit_prompt(text, [1, 2], False, False, style=preset.id))


class FaceRuleTests(unittest.TestCase):
    def test_a_check_across_styles_uses_the_most_lenient_thresholds(self):
        with patch.dict(qc.FACE_RULES, {styles.CARTOON_2D: qc.FaceRules(0.2, 0.15)}):
            self.assertEqual(qc.rules(), qc.FaceRules(qc.FACE_MIN, qc.FACE_MATCH))
            self.assertEqual(qc.rules(styles.REALISTIC, styles.CARTOON_2D), qc.FaceRules(0.2, 0.15))
            self.assertTrue(qc.judge(0.3, minimum=qc.rules(styles.CARTOON_2D).minimum).ok)
            self.assertFalse(qc.judge(0.3).ok)
            ola = {"name": "Ola", "slug": "ola", "style": styles.CARTOON_2D}
            with (patch.object(qc.model_manager, "ensure", return_value=None),
                  patch.object(qc, "face_score", return_value=0.3)):
                self.assertTrue(qc.check("x.png", ola).ok)              # the character's own style
                self.assertFalse(qc.check("x.png", {**ola, "style": "realistic"}).ok)
                self.assertTrue(qc.check("x.png", {**ola, "style": "realistic"}, style=styles.CARTOON_2D).ok)


class Drawn2DFaceTests(unittest.TestCase):
    def test_a_2d_photo_is_judged_by_the_vision_model_alone_and_never_by_face(self):
        """YuNet found no face on any 2D anime anchor (2026-11-15): the adult gate of the vision model decides, the face
        count and the SFace comparison are left out; a 3D face is still counted."""
        adult = {"person": True, "age": 25, "gender": "woman", "caption": ""}
        with (patch.object(characters.llm, "check_photo", return_value=adult),
              patch.object(qc, "count_faces", return_value=0) as counted,
              patch.object(qc, "face_score", side_effect=AssertionError("no face to compare"))):
            self.assertEqual(characters.check_photo("x.png", styles.CARTOON_2D), "")
            self.assertEqual(characters.same_person("x.png", "y.png", styles.CARTOON_2D), "")
            counted.assert_not_called()
            self.assertIn("exactly one clear face", characters.check_photo("x.png", styles.ANIMATION_3D))
        with patch.object(characters.llm, "check_photo", return_value={**adult, "age": 15}):
            self.assertTrue(characters.check_photo("x.png", styles.CARTOON_2D).startswith("⛔"))  # the adult gate stays
        self.assertFalse(qc.rules(styles.CARTOON_2D).detects)
        self.assertFalse(qc.rules(styles.REALISTIC, styles.CARTOON_2D).detects)
        self.assertTrue(qc.rules(styles.ANIMATION_3D).detects)


class CharacterStyleTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        stack.enter_context(patch.object(characters, "DIR", root / "characters"))
        stack.enter_context(patch.object(characters, "DRAFTS_DIR", root / "drafts"))

    def test_a_draft_remembers_its_style_and_a_realistic_character_keeps_its_identity(self):
        draft = characters.new_draft()
        self.assertEqual(characters.draft(draft)["style"], styles.REALISTIC)
        self.assertEqual(characters.update_draft(draft, style=styles.ANIMATION_3D)["style"], styles.ANIMATION_3D)
        with self.assertRaisesRegex(ValueError, "valid style"):
            characters.update_draft(draft, style="clay")
        record = {"format": characters.FORMAT, "name": "Mia", "slug": "mia", "gender": "woman", "look": "red hair",
                  "photos": [], "voice": None}
        plain = characters._profile("mia", record)
        drawn = characters._profile("mia", {**record, "style": styles.CARTOON_2D})
        self.assertEqual((plain["style"], drawn["style"]), (styles.REALISTIC, styles.CARTOON_2D))
        self.assertNotIn("style", characters.identity(plain))  # the clips of a realistic character stay valid
        self.assertEqual(characters.identity(drawn)["style"], styles.CARTOON_2D)

    def test_a_character_can_be_drawn_in_a_custom_style_but_never_a_childlike_one(self):
        draft = characters.new_draft()
        watercolor = styles.custom_id(styles.ANIME, "soft pastel watercolor")
        self.assertEqual(characters.update_draft(draft, style=watercolor)["style"], watercolor)
        for unsafe in ("custom:1:cartoon:kids cartoon", "custom:1:2d:chibi, big heads"):
            with self.assertRaisesRegex(ValueError, "adult"):
                characters.update_draft(draft, style=unsafe)
        self.assertEqual(characters.draft(draft)["style"], watercolor)
        record = {"format": characters.FORMAT, "name": "Mia", "slug": "mia", "gender": "woman", "look": "red hair",
                  "photos": [], "voice": None, "style": watercolor}
        self.assertEqual(characters.identity(characters._profile("mia", record))["style"], watercolor)

    def test_a_scene_with_a_character_is_drawn_in_the_films_style(self):
        mia = {"name": "Mia", "gender": "woman", "photos": [__file__]}
        with patch.object(characters.refs, "compose", return_value=(True, "frame.png")) as compose:
            characters.render(None, mia, "Mia in a cafe", 832, 480, Path("frame.png"), seed=1)
            characters.render(None, mia, "Mia in a cafe", 832, 480, Path("frame.png"), seed=1,
                              style=styles.ANIMATION_3D)
        realistic, animated = (call.args[2] for call in compose.call_args_list)
        self.assertIn("Make a single live-action scene. Mia in a cafe", realistic)
        self.assertIn("3D animated film", animated)
        self.assertNotIn("live-action", animated)

    def test_the_creator_draws_the_first_photo_and_its_views_in_the_drafts_style(self):
        renders, composes = [], []

        def render(prompt, *_args, **_kwargs):
            renders.append(prompt)
            return b"png"

        def compose(_client, _images, prompt, *_args, **_kwargs):
            composes.append(prompt)
            return True, "view.png"

        target = Path(tempfile.mkdtemp()) / "a.png"
        self.addCleanup(target.unlink, missing_ok=True)
        with (patch.object(char_photos.engine_runtime, "render", side_effect=render),
              patch.object(char_photos.characters, "check_photo", return_value="") as gate,
              patch.object(char_photos.refs, "compose", side_effect=compose),
              patch.object(char_photos.qc, "face_score", return_value=0.9)):
            self.assertTrue(char_photos._anchor(None, "red hair", "woman", target, 1, [], styles.CARTOON_2D))
            ok, _ = char_photos._view(None, target, {"slug": "x", "photos": [str(target)]}, "half body", target, 1,
                                      styles.CARTOON_2D)
        self.assertTrue(ok)
        self.assertTrue(renders[0].startswith("2D anime cartoon character"))
        self.assertTrue(renders[0].endswith(styles.get(styles.CARTOON_2D).image))
        self.assertTrue(composes[0].endswith(styles.get(styles.CARTOON_2D).view))
        self.assertEqual({call.args[1] for call in gate.call_args_list}, {styles.CARTOON_2D})


class TimelineStyleTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.root = Path(self.context.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.context.enter_context(patch.object(timeline, "DIR", self.root / "projects"))
        self.context.enter_context(patch("app.items.DIR", self.root / "items"))
        self.context.enter_context(patch.object(config, "OUTPUTS_DIR", self.root / "outputs"))
        self.context.enter_context(patch.object(config, "HISTORY_DIR", self.root / "history"))
        self.context.enter_context(patch.object(history, "_HISTORY_FILE", self.root / "history/history.jsonl"))
        self.context.enter_context(patch.object(timeline.model_manager, "ensure", return_value=None))
        self.context.enter_context(patch.object(timeline.model_manager, "VERIFIED_FILE",
                                                self.root / "verified_models.json"))
        self.context.enter_context(patch.object(timeline.model_manager, "_STORED", None))
        self.styles_used = []

    def generate(self, prompt, seconds, _aspect, *, scene_dir, style, **_options):
        self.styles_used.append(style)
        scene_dir.mkdir(parents=True)
        path = scene_dir / "scene.mp4"
        _film(path, seconds)
        notes = ["Lip sync skipped (no face). The dialogue plays over the picture without mouth movement."]
        info = {"qc": {"ok": True}, "width": 48, "height": 32}
        if style != prefs.STYLE["video"]:
            info["lipsync_skipped"] = notes[0]
            if "tree" in prompt:  # a drawn start frame that failed: the clip was made from the text
                info["start_frame_note"] = "The starting frame could not be drawn, so the clip was made from the text."
        return video_gen.VideoOutput(True, [str(path)], info=info, notes=notes)

    def save(self, scenes, previous=None, **options):
        return timeline.save(previous.id if previous else None, previous.revision if previous else None,
                             "Cartoon", tuple(scenes), "Landscape 16:9", False, [], **options)

    def test_a_film_in_a_custom_style_is_made_with_its_words_and_keeps_them_when_the_library_changes(self):
        from app import style_library

        self.context.enter_context(patch.object(characters, "DIR", self.root / "characters"))
        self.context.enter_context(patch.dict(style_library._CACHE, clear=True))
        entry, _ = style_library.save("Storybook", styles.ANIME, "soft pastel watercolor, paper texture")
        project = self.save([timeline.Scene("A cup", 1), timeline.Scene("A tree", 1, cut=True)], style=entry["id"])
        self.assertEqual(timeline.load(project.id).style, entry["id"])
        stored = json.loads((timeline._root(project.id) / "project.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["style"], entry["id"])
        custom = styles.get(entry["id"])
        with patch.object(timeline.video_gen, "generate", side_effect=self.generate):
            ok, _files, notes, error = timeline.prepare_job({"project": project.id, "key": project.key},
                                                            lambda *_: None)
            self.assertTrue(ok, error)
            self.assertEqual(self.styles_used, [custom.video] * 2)  # the renderer gets the style's Wan text
            record = timeline.asset(project, timeline.prepared(project)["assets"][0])
            self.assertEqual(record["style"], custom.video)
            self.assertEqual(styles.for_video_text(record["style"]).id, entry["id"])
            self.assertEqual(timeline.takes(project, 0)[0]["current"], True)
            # a new version of the style: the film keeps the old one until the owner picks it again
            edited, _ = style_library.save("Storybook", styles.ANIME, "bright gouache, paper texture", entry["key"])
            self.assertEqual(timeline.load(project.id).style, entry["id"])
            newer = self.save(project.scenes, project, style=edited["id"])
            self.assertEqual(timeline.outdated(newer), [0, 1])
            [issue] = timeline._style_issues(newer, [{"name": "Mia", "slug": "mia", "style": entry["id"]}])
            self.assertIn("this film (Storybook): Mia (Storybook (earlier version))", issue.text)
            ok, _files, notes, error = timeline.prepare_job({"project": newer.id, "key": newer.key},
                                                            lambda *_: None)
            self.assertTrue(ok, error)
            self.assertEqual(self.styles_used[2:], [styles.get(edited["id"]).video] * 2)
            old_take, new_take = timeline.prepared(project)["assets"][0], timeline.prepared(newer)["assets"][0]
            self.assertNotEqual(old_take, new_take)
            # each version lists only its own takes of the scene
            self.assertEqual([take["key"] for take in timeline.takes(newer, 0)], [new_take])
            back = self.save(project.scenes, newer, style=entry["id"])
            self.assertEqual(back.key, project.key)
            self.assertEqual(timeline.outdated(back), [])  # back to the earlier version: its clips are Ready again
            self.assertEqual([take["key"] for take in timeline.takes(back, 0)], [old_take])
        style_library.delete(entry["key"])
        reopened = timeline.load(project.id)
        self.assertEqual((reopened.style, reopened.video_style, reopened.key), (back.style, custom.video, back.key))

    def test_an_unsafe_style_is_refused_only_when_it_is_chosen(self):
        safe = styles.custom_id(styles.OTHER, "ink and wash")
        project = self.save([timeline.Scene("A cup", 1)], style=safe)
        with self.assertRaisesRegex(ValueError, "adult"):
            self.save(project.scenes, project, style=styles.custom_id(styles.OTHER, "chibi, big heads"))
        with patch.object(timeline.llm, "style_problem", return_value="⛔ refused now"):
            kept = self.save((timeline.Scene("A cup on a table", 1),), project)  # unchanged style: saved
            self.assertEqual(kept.style, safe)
            self.assertEqual(timeline.load(project.id).style, safe)

    def test_a_realistic_project_keeps_its_file_and_key(self):
        project = self.save([timeline.Scene("A cup", 1)])
        stored = json.loads((timeline._root(project.id) / "project.json").read_text(encoding="utf-8"))
        self.assertNotIn("style", stored)
        self.assertEqual(project.style, styles.REALISTIC)
        self.assertEqual(project.key, replace(project, style=styles.REALISTIC).key)
        animated = replace(project, style=styles.ANIMATION_3D)
        self.assertNotEqual(animated.key, project.key)
        with self.assertRaisesRegex(ValueError, "valid style"):
            self.save(project.scenes, project, style="clay")

    def test_a_style_change_marks_the_made_scenes_outdated_and_the_old_style_finds_them_again(self):
        project = self.save([timeline.Scene("A cup", 1), timeline.Scene("A tree", 1, cut=True)])
        with patch.object(timeline.video_gen, "generate", side_effect=self.generate):
            ok, _files, notes, error = timeline.prepare_job({"project": project.id, "key": project.key},
                                                            lambda *_: None)
            self.assertTrue(ok, error)
            self.assertEqual(notes, ["2 clips ready on the timeline."])
            self.assertEqual(timeline.outdated(project), [])
            made = timeline.prepared(project)["assets"]
            picked = replace(project, style=styles.CARTOON_2D)  # chosen in the menu, not saved yet: Outdated already
            self.assertEqual(ui_timeline._stale_scenes(picked, project, made, False, False), frozenset({0, 1}))
            self.assertEqual(ui_timeline._stale_scenes(picked, project, made, False, True), frozenset())
            self.assertEqual(ui_timeline._stale_scenes(replace(picked, audio=not project.audio), project, made,
                                                       False, False), frozenset())  # not only the style changed
            cartoon = self.save(project.scenes, project, style=styles.CARTOON_2D)
            self.assertEqual(timeline.load(project.id).style, styles.CARTOON_2D)
            keys = timeline.prepared(cartoon)["assets"]
            self.assertEqual(keys, timeline.prepared(project)["assets"])  # the clips stay on the timeline...
            self.assertEqual(timeline.outdated(cartoon), [0, 1])           # ...and show Outdated
            statuses = ui_timeline._scene_statuses(cartoon, cartoon, keys, None,
                                                   stale=frozenset(timeline.outdated(cartoon)))
            self.assertEqual(statuses, ("Outdated", "Outdated"))
            ok, _files, notes, error = timeline.prepare_job({"project": cartoon.id, "key": cartoon.key},
                                                            lambda *_: None)
            self.assertTrue(ok, error)
            self.assertEqual(timeline.outdated(cartoon), [])
            self.assertEqual(self.styles_used[2:], [styles.get(styles.CARTOON_2D).video] * 2)
            # the job log names each scene whose lip sync was skipped, and the one whose drawn frame failed
            self.assertEqual(len(notes), 4)
            self.assertTrue(notes[1].startswith("Scene 1: Lip sync skipped"))
            self.assertEqual(notes[3], "Scene 2: The starting frame could not be drawn, so the clip was made from the text.")
            record = timeline.asset(cartoon, timeline.prepared(cartoon)["assets"][0])
            self.assertEqual(record["style"], styles.get(styles.CARTOON_2D).video)
            self.assertIn("Lip sync skipped", record["lipsync_skipped"])
            # Realistic picked again in the menu: before Save the realistic clips already show Ready
            picked = replace(cartoon, style=styles.REALISTIC)
            keys = timeline.plan_for(picked, cartoon, timeline.prepared(cartoon)["assets"])
            self.assertEqual(keys, timeline.prepared(project)["assets"])
            self.assertEqual(ui_timeline._stale_scenes(picked, cartoon, keys, False, False), frozenset())
            # back to Realistic: the realistic clips are on the timeline again, Ready, nothing is made
            back = self.save(project.scenes, cartoon, style=styles.REALISTIC)
            self.assertEqual(back.key, project.key)
            self.assertEqual(timeline.outdated(back), [])
        self.assertEqual(len(self.styles_used), 4)

    def test_the_editor_names_the_style_and_keeps_the_saved_one_when_the_field_is_empty(self):
        project = self.save([timeline.Scene("A cup", 1)], style=styles.ANIMATION_3D)
        draft = ui_timeline.Draft(project, project.scenes)
        self.assertEqual(ui_timeline._candidate(draft, "Cartoon", "Landscape 16:9", False, "", []).style,
                         styles.ANIMATION_3D)
        self.assertEqual(ui_timeline._candidate(draft, "Cartoon", "Landscape 16:9", False, "", [], False,
                                                styles.CARTOON_2D).style, styles.CARTOON_2D)
        self.assertIn("3D animation", ui_timeline._status_html(project, project, False, []))
        plain = replace(project, style=styles.REALISTIC)
        self.assertNotIn("Realistic", ui_timeline._status_html(plain, plain, False, []))

    def test_a_character_drawn_in_another_style_is_named_in_the_scene_notes(self):
        project = self.save([timeline.Scene("Mia waves", 1)], style=styles.CARTOON_2D)
        mia = {"name": "Mia", "slug": "mia", "style": styles.REALISTIC}
        issues = timeline._style_issues(project, [mia, {"name": "Kit", "slug": "kit", "style": styles.CARTOON_2D}])
        self.assertEqual(len(issues), 1)
        self.assertIn("Mia (Realistic)", issues[0].text)
        self.assertEqual(timeline._style_issues(replace(project, style=styles.REALISTIC), [mia]), [])


if __name__ == "__main__":
    unittest.main()
