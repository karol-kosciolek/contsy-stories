# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Animal and creature characters in scenes: how compositions and the video model name them (people's texts stay byte
for byte as they were), the always-Realistic Video and Image pages refusing them, face checks and lip sync leaving them
out, and a scene where one speaks keeping its picture, with a note, in any style. No GPU, no model, no live state."""
from contextlib import ExitStack
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from app import cast, characters, ffmpeg_utils, history, image_gen, lipsync, model_manager, prefs, qc, refs, sound
from app import styles, tts, video_gen, voices
from app.scene_parts import Cue
from engine.media import audio as media_audio
from tests.character_fixtures import store_character

OLA = {"name": "Ola", "slug": "ola", "gender": "woman", "kind": "person", "look": "Fair skin, long wavy auburn hair",
       "photos": ["ola.png"]}
KUBA = {"name": "Kuba", "slug": "kuba", "gender": "man", "kind": "person",
        "look": "Short dark brown hair, a short beard", "photos": ["kuba.png"]}
JAN = {"name": "Jan", "slug": "jan", "gender": "man", "kind": "person", "look": "", "photos": ["jan.png"]}
CHAIN = {"name": "Chain Dog", "slug": "chain_dog", "gender": "man", "kind": "creature", "style": styles.ANIMATION_3D,
         "look": "An adorable adult anthropomorphic red-orange corgi dog wearing a tiny blue hoodie, standing "
                 "upright on two legs.", "photos": ["chain_dog.png"]}
HOOT = {"name": "Hoot", "slug": "hoot", "gender": "woman", "kind": "creature", "look": "A fully grown snowy owl",
        "photos": ["hoot.png"]}
CHAIN_LOOK = CHAIN["look"][:-1]  # as the prompts say it: without the closing period
NOTE = "Lip sync left out for Chain Dog: MuseTalk only moves human mouths, so the lines play over the picture."
REFUSED = "Chain Dog is an animal or creature: use it in a Timeline film in an animated style."
KEEP_CHAIN = ("Keep the creature from image 2 exactly as it is there: the same kind of animal or creature, face, body "
              "shape, colors and markings, and its outfit exactly as image 2 shows it.")


class WordingTests(unittest.TestCase):
    def test_a_creature_is_the_creature_in_compositions_and_labels_in_a_group_of_its_own(self):
        self.assertEqual(cast.mark_names("Ola walks Chain Dog in the park. Chain Dog barks.", [OLA, CHAIN]),
                         "The woman [1] walks the creature [2] in the park. The creature [2] barks.")
        # Chain Dog speaks with a man's voice, yet Kuba stays "the man", not "the first man"
        self.assertEqual(cast.labels([KUBA, CHAIN, OLA]),
                         {"kuba": "the man", "chain_dog": "the creature", "ola": "the woman"})
        self.assertEqual(cast.label_names("Chain Dog chases Hoot past Kuba.", [CHAIN, HOOT, KUBA]),
                         "The first creature chases the second creature past the man.")
        self.assertEqual(cast.look_sentences([OLA, CHAIN, JAN]), ". The woman has fair skin, long wavy auburn hair. "
                                                                 f"The creature looks like this: {CHAIN_LOOK}")

    def test_people_keep_every_word_they_had(self):
        self.assertEqual(cast.mark_names("Ola hugs Kuba near the fountain [3]", [OLA, KUBA]),
                         "The woman [1] hugs the man [2] near the fountain [3]")
        self.assertEqual(cast.look_sentences([OLA, KUBA]), ". The woman has fair skin, long wavy auburn hair. "
                                                           "The man has short dark brown hair, a short beard")
        lead = refs.cast_lead([OLA, KUBA], creatures=[3])  # an image number outside the lead changes nothing
        self.assertEqual(lead, refs.cast_lead([OLA, KUBA]))
        self.assertEqual(refs.edit_prompt("The woman [1] hugs the man [2] near the fountain [3]", [1, 2, 3], False,
                                          False, looks={1: OLA["look"], 2: KUBA["look"], 3: ""}, base_lead=lead),
                         "These are different people: image 1 shows Ola, a woman; image 2 shows Kuba, a man. Show them "
                         "together in one coherent real scene as described, keeping each person's exact face, facial "
                         "features, hairstyle, hair color, skin tone and body from their own image. Do not merge, swap "
                         "or blend their faces. The person from image 1 has Fair skin, long wavy auburn hair - keep "
                         "exactly this look in any light. The person from image 2 has Short dark brown hair, a short "
                         "beard - keep exactly this look in any light. Photorealistic photo, one coherent real scene. "
                         "Every person from the reference images stays exactly the same person: same face, facial "
                         "features, hairstyle, hair color, skin tone and body - only pose, expression, clothing and "
                         "setting change as described. Keep the exact design, colors, patterns, logos, text and "
                         "proportions of the items from the reference images. The woman from image 1 hugs the man from "
                         "image 2 near the fountain from image 3 Her or his look stays exactly: Fair skin, long wavy "
                         "auburn hair, not tanned. Her or his look stays exactly: Short dark brown hair, a short "
                         "beard.")
        self.assertEqual(refs.cast_lead([KUBA], first_image=2, adding=True),
                         "These are different people: image 2 shows Kuba, a man. Add them to the picture from image 1 "
                         "next to the people already there, keeping each person's exact face, facial features, "
                         "hairstyle, hair color, skin tone and body from their own image. Do not merge, swap or blend "
                         "their faces.")
        self.assertEqual(refs.CHARACTER, "Image 1 shows the main person: keep exactly her or his face, facial "
                                         "features, hairstyle, hair color, skin tone and body from image 1. Do not "
                                         "tan, darken, lighten or restyle her or him.")
        self.assertEqual(video_gen._wan_prompt("A woman walks on a beach.", [OLA], {}, False, ""),
                         "A woman walks on a beach.. The person has Fair skin, long wavy auburn hair")

    def test_a_composition_names_the_creature_and_says_what_to_keep_of_it(self):
        self.assertEqual(refs.cast_lead([OLA, CHAIN], creatures=[2]),
                         "These are different characters: image 1 shows Ola, a woman; image 2 shows Chain Dog, an "
                         "animal or creature character (not a human). Show them together in one coherent real scene as "
                         "described, keeping each person's exact face, facial features, hairstyle, hair color, skin "
                         f"tone and body from their own image. {KEEP_CHAIN} Do not merge, swap or blend their faces.")
        animals = refs.cast_lead([HOOT, CHAIN], style=styles.ANIMATION_3D, creatures=[1, 2])
        self.assertTrue(animals.startswith(
            "These are different characters: image 1 shows Hoot, an animal or creature character (not a human); "
            "image 2 shows Chain Dog, an animal or creature character (not a human). Show them together in one "
            "coherent scene of a 3D animated film as described. Keep the creature from image 1 exactly as it is "
            "there"), animals)
        self.assertIn(KEEP_CHAIN, animals)
        self.assertFalse([word for word in ("person", "woman", " man") if word in animals], animals)

    def test_the_edit_prompt_describes_a_creature_image_in_its_own_words(self):
        text = cast.mark_names(refs.renumber("Ola walks Chain Dog past the kiosk [1]", 2), [OLA, CHAIN])
        prompt = refs.edit_prompt(text, [1, 2, 3], False, False, looks={1: OLA["look"], 2: CHAIN["look"], 3: ""},
                                  base_lead=refs.cast_lead([OLA, CHAIN], creatures=[2]), creatures=[2])
        self.assertIn("The woman from image 1 walks the creature from image 2 past the kiosk from image 3", prompt)
        self.assertIn(f"The person from image 1 has {OLA['look']} - keep exactly this look in any light. The creature "
                      f"from image 2 looks like this: {CHAIN_LOOK} - keep exactly this look in any light.", prompt)
        self.assertTrue(prompt.endswith(f"Her or his look stays exactly: {OLA['look']}, not tanned. The look of the "
                                        f"creature from image 2 stays exactly: {CHAIN_LOOK}."), prompt)
        self.assertNotIn("person from image 2", prompt)
        unnamed = refs.edit_prompt("A sunny park", [1, 2], False, False, creatures=[2])
        self.assertIn("Use the exact characters and items shown in image 1 and image 2: the characters and things in "
                      "the description below are them.", unnamed)
        based = refs.edit_prompt("Chain Dog sits on a bench", [1], True, False, base_lead=refs.CREATURE_CHARACTER)
        self.assertTrue(based.startswith("Image 1 shows the main character, an animal or creature (not a human): keep "
                                         "exactly the same kind of animal or creature"), based)
        self.assertIn("Use the exact items, places and any other people shown in image 2; the main character is the "
                      "one from image 1.", based)

    def test_compose_frame_tells_the_composition_which_images_show_creatures(self):
        calls = []

        def compose(client, base, ref_list, text, w, h, out, seed=None, **options):
            calls.append((text, options["pass_leads"], list(options["creatures"])))
            Path(out).write_bytes(b"scene")
            return True, str(out)

        with tempfile.TemporaryDirectory() as tmp, patch("app.refs.compose_refs", side_effect=compose):
            first = cast.compose_frame(None, [OLA, CHAIN], "Ola walks Chain Dog on the beach", 1280, 720,
                                       Path(tmp) / "a.png", 1)
            second = cast.compose_frame(None, [OLA, KUBA, JAN, CHAIN], "Ola, Kuba and Jan meet Chain Dog", 1280, 720,
                                        Path(tmp) / "b.png", 1)
        self.assertTrue(first[0] and second[0], (first, second))
        text, leads, creatures = calls[0]
        self.assertEqual(text, "The woman [1] walks the creature [2] on the beach")
        self.assertEqual((leads, creatures), ([refs.cast_lead([OLA, CHAIN], creatures=[2])], [2]))
        _, leads, creatures = calls[1]
        self.assertEqual(creatures, [4])
        self.assertEqual(leads[0], refs.cast_lead([OLA, KUBA, JAN]))  # the first pass shows people only: their words
        self.assertEqual(leads[1], "These are different characters: image 2 shows Chain Dog, an animal or creature "
                                   "character (not a human). Add them to the picture from image 1 next to the "
                                   f"characters already there. {KEEP_CHAIN} Do not merge, swap or blend their faces.")

    def test_the_video_model_hears_a_creature_in_its_own_words(self):
        self.assertEqual(video_gen._wan_prompt("Chain Dog runs through a sunny park.", [CHAIN], {}, False, ""),
                         f"Chain Dog runs through a sunny park.. Chain Dog looks like this: {CHAIN_LOOK}")
        self.assertEqual(video_gen._wan_prompt("A dog runs through a sunny park", [CHAIN], {1: "a red coat"}, True,
                                               "cinematic"),
                         f"A dog runs through a sunny park. The character looks like this: {CHAIN_LOOK}. The person "
                         "has a red coat, cinematic")
        self.assertEqual(video_gen._wan_prompt("Ola walks Chain Dog on the beach", [OLA, CHAIN], {}, False, ""),
                         "The woman walks the creature on the beach. The woman has fair skin, long wavy auburn hair. "
                         f"The creature looks like this: {CHAIN_LOOK}")

    def test_only_people_are_checked_for_a_face_that_drifted(self):
        with patch.object(qc, "enabled", return_value=True), \
                patch.object(qc, "check", return_value=qc.Verdict(True)) as check:
            self.assertEqual(video_gen._drift(Path("frame.png"), [CHAIN, HOOT]), "")
            check.assert_not_called()
            video_gen._drift(Path("frame.png"), [CHAIN, OLA])
        self.assertIs(check.call_args.args[1], OLA)


class PageTests(unittest.TestCase):
    """The Video and Image pages are always Realistic: an animal or creature is refused before anything runs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="creature_pages_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for number, ch in enumerate((OLA, CHAIN)):
            store_character(self.tmp / "characters", {**ch, "created": number})
        for target in (patch.object(characters, "DIR", self.tmp / "characters"),
                       patch.object(prefs, "_FILE", self.tmp / "prefs.json"),
                       patch.object(voices, "tokens", return_value={})):
            target.start()
            self.addCleanup(target.stop)

    def test_the_pages_refuse_a_creature_that_would_appear(self):
        self.assertTrue(characters.is_creature(characters.get("chain_dog")))
        self.assertEqual(cast.check_prompt("Chain Dog runs through the park", [])[1:], (REFUSED, []))
        self.assertEqual(cast.check_prompt("A sunny park", ["chain_dog"])[1], REFUSED)  # the main character's shot
        found, error, notes = cast.check_prompt("Ola reads a book", ["ola", "chain_dog"])  # picked, never named
        self.assertEqual(([c["slug"] for c in found], error), (["ola"], ""))
        self.assertIn("Not named in the prompt, so not in it: Chain Dog.", notes)
        noir = styles.custom_id(styles.REALISTIC, "black and white film noir")
        self.assertEqual(cast.check_prompt("Chain Dog runs", [], style=noir)[1], REFUSED)

    def test_a_timeline_film_in_an_animated_style_takes_a_creature(self):
        for style in (styles.ANIMATION_3D, styles.CARTOON, styles.ANIME,
                      styles.custom_id(styles.ANIME, "soft pastel watercolor")):
            with self.subTest(style=style):
                found, error, _ = cast.check_prompt("Ola walks Chain Dog", ["ola"], ["Ola walks Chain Dog"],
                                                    style=style)
                self.assertEqual(([c["slug"] for c in found], error), (["ola", "chain_dog"], ""))

    def test_the_image_and_video_jobs_refuse_a_creature_before_any_model_work(self):
        untouched = AssertionError("no model work for a refused creature")
        chain, ola = characters.get("chain_dog"), characters.get("ola")
        with patch.object(model_manager, "ensure", side_effect=untouched), \
                patch.object(image_gen.engine_runtime, "engine", side_effect=untouched):
            alone = image_gen.edit_with_refs([], "Chain Dog in a park", 1.0, character=chain)
            together = image_gen.edit_with_refs([(1, "hat.png")], "Ola and Chain Dog", 1.0, character=ola,
                                                cast=[ola, chain])
            video = video_gen.generate("Chain Dog runs on a beach.", 3, "Landscape 16:9", sound_on=False,
                                       style=prefs.STYLE["video"], character=chain, cast=[chain])
        self.assertEqual([(out.ok, out.error) for out in (alone, together, video)], [(False, REFUSED)] * 3)


class FaceCheckTests(unittest.TestCase):
    def test_a_creature_has_no_reference_faces_and_nothing_of_it_is_read(self):
        untouched = AssertionError("a creature's face is never read")
        with patch.dict(qc._REFS, {"chain_dog": ((), ["a stale embedding"])}), \
                patch.object(qc.characters, "reference_photos", side_effect=untouched), \
                patch.object(qc, "_embed", side_effect=untouched):
            self.assertEqual(qc._references(CHAIN), [])
            self.assertIsNone(qc.face_score("frame.png", CHAIN))

    def test_checks_leave_creatures_out(self):
        untouched = AssertionError("nothing to measure")
        with patch.object(qc.model_manager, "ensure", side_effect=untouched), \
                patch.object(qc, "face_score", side_effect=untouched), \
                patch.object(qc, "face_scores", side_effect=untouched):
            for creatures in (CHAIN, [CHAIN, HOOT]):
                self.assertEqual(qc.check("frame.png", creatures).record(), {"ok": True, "reasons": [], "face": None})
        with patch.object(qc.model_manager, "ensure", return_value=None), \
                patch.object(qc, "face_score", return_value=0.9) as score:
            verdict = qc.check("frame.png", [OLA, CHAIN])
        score.assert_called_once_with("frame.png", OLA)
        self.assertEqual(verdict.record(), {"ok": True, "reasons": [], "face": 0.9})

    def test_scores_and_crop_matches_never_name_a_creature(self):
        face = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        known = {"ola": [face], "chain_dog": [face]}  # even a creature's matching face would not count
        with patch.object(qc, "_references", side_effect=lambda ch: known[ch["slug"]]) as references, \
                patch.object(qc, "_faces", return_value=[((0, 0, 10, 10), face)]), \
                patch.object(qc, "_embed", return_value=face):
            self.assertEqual(qc.face_scores("frame.png", [CHAIN, OLA]), {"ola": 1.0})
            self.assertEqual(qc.match_crops(["crop.png"], [CHAIN, OLA]), {"ola": 0})
        self.assertEqual({call.args[0]["slug"] for call in references.call_args_list}, {"ola"})

    def test_a_creature_s_style_sets_no_face_threshold(self):
        drawn = {**CHAIN, "style": styles.ANIME}  # YuNet finds no 2D face: counted, it would turn detection off
        self.assertTrue(qc.rules(*qc._styles_of([OLA, drawn])).detects)
        animated = {**OLA, "style": styles.ANIMATION_3D}
        with patch.object(qc.model_manager, "ensure", return_value=None), \
                patch.object(qc, "face_score", return_value=0.5):
            verdict = qc.check("frame.png", [animated, {**CHAIN, "style": styles.REALISTIC}],
                               style=styles.ANIMATION_3D)
        self.assertEqual(verdict.reasons, ["different face (similarity 0.50 < 0.70)"])  # 3D faces' threshold


class LipSyncTests(unittest.TestCase):
    TRACKS = [{"crop": "face_0.png", "size": 200}, {"crop": "face_1.png", "size": 150}]

    def test_a_creature_never_takes_a_face(self):
        notes = []
        with patch.object(qc, "match_crops", side_effect=AssertionError("no person to recognise")):
            # one speaker and one free face: no size or gender fallback for a creature
            self.assertEqual(lipsync.assign_faces(["male#char:chain_dog"], self.TRACKS[:1], [CHAIN], "work", notes),
                             {})
        self.assertEqual(notes, [NOTE])

    def test_people_keep_their_faces_and_the_creatures_lines_are_left_out(self):
        notes = []
        with patch.object(qc, "match_crops", return_value={}) as match:
            assigned = lipsync.assign_faces(["male#char:chain_dog", "female#char:ola", "male#char:chain_dog"],
                                            self.TRACKS, [CHAIN, OLA], "work", notes)
        self.assertEqual(list(assigned), ["female#char:ola"])
        self.assertIs(assigned["female#char:ola"], self.TRACKS[0])  # the largest face: the creature spoke first
        self.assertEqual(match.call_args.args[1], [OLA])
        self.assertEqual(notes, [NOTE])

    def test_the_note_names_every_creature(self):
        self.assertEqual(lipsync.creature_note(["Chain Dog", "Hoot"]),
                         "Lip sync left out for Chain Dog and Hoot: MuseTalk only moves human mouths, so the lines "
                         "play over the picture.")
        self.assertIn(" for Rex, Hoot and Chain Dog: ", lipsync.creature_note(["Rex", "Hoot", "Chain Dog"]))

    def test_creature_lines_alone_keep_the_picture_without_looking_for_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            picture, out = Path(tmp) / "talk.mp4", Path(tmp) / "synced.mp4"
            picture.write_bytes(b"picture")
            with patch.object(lipsync.ffmpeg_utils, "video_fps", return_value=25.0), \
                    patch.object(lipsync, "find_faces", side_effect=AssertionError("no face is looked for")):
                result = lipsync.run_dialogue(str(picture), str(Path(tmp) / "speech.wav"), [(0.5, 1.5)],
                                              ["male#char:chain_dog"], str(out), cast=[CHAIN, OLA])
            self.assertEqual(result, (True, str(out), [NOTE]))
            self.assertEqual(out.read_bytes(), b"picture")


RATE = media_audio.RATE
HANDLE = media_audio.TIMELINE_AUDIO_HANDLE
PICTURE = "Chain Dog sits by the rainy window, natural lip movement"
WOOF = Cue("Woof! Hello there.", "chain_dog", 0.5)


def _tone(path, seconds, frequency=220.0):
    t = np.arange(int(round(seconds * RATE))) / RATE
    media_audio.write_wav(path, (0.2 * np.sin(2 * np.pi * frequency * t))[None, :].astype(np.float32), RATE)


def _copy(source, destination, *args, **kwargs):
    shutil.copyfile(source, destination)
    return True, str(destination)


class SceneTests(unittest.TestCase):
    """Timeline scenes with spoken lines (video_gen.generate with scene_dir and cues); speech, sound, Wan, lip sync,
    the start frames and FFmpeg are fakes."""

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        pool = []
        for name, slug, gender in (("Hanna", "hanna", "female"), ("Adam", "adam", "male")):
            _tone(self.root / f"{slug}.wav", 0.5)
            pool.append(voices.Voice(name, slug, gender, str(self.root / f"{slug}.wav"), builtin=True))
        self.events, self.synced, self.part_fps, self.retimed, self.prompts = [], [], [], [], []

        def speech(items, progress_cb=None):
            for _text, _voice, out in items:
                _tone(out, 1.0, 320.0)
            return True, ""

        def ambience(_client, _scene, _custom, seconds, _seed, out_path, progress_cb=None):
            _tone(out_path, seconds + 2 * HANDLE, 1500.0)
            return True, str(out_path)

        def wan(_runtime, _kind, _start, output, *args, **options):
            self.events.append("wan")
            self.part_fps.append(args[1])  # the rate the part is made at
            self.prompts.append(options["prompt"])
            Path(output).write_bytes(b"mp4")

        def lip_sync(video, _voice, timings, who, output, *_args, **options):
            self.events.append("lipsync")
            self.synced.append((list(timings), list(who), [Path(path).name for path in options.get("lines") or ()]))
            Path(output).write_bytes(Path(video).read_bytes() + b" synced")
            return True, output, []

        def frame(*args, **_options):
            Path(args[5]).write_bytes(b"png")
            return True, str(args[5])

        def mix(_voice, _background, out_path, _duration, **_options):
            Path(out_path).write_bytes(b"mix")
            return True, str(out_path)

        def retime(_client, source, output, **options):
            self.retimed.append(options["duration"])
            return _copy(source, output)

        self.ensure = stack.enter_context(patch.object(model_manager, "ensure", return_value=None))
        self.talk = stack.enter_context(patch.object(video_gen.engine_runtime, "talk_installed", return_value=False))
        for target, name, options in (
                (tts, "synthesize_many", {"side_effect": speech}),
                (sound, "generate", {"side_effect": ambience}),
                (history, "record", {}),
                (voices, "all_voices", {"return_value": pool}),
                (video_gen.characters, "render", {"side_effect": frame}),
                (video_gen.cast_mod, "compose_frame", {"side_effect": frame}),
                (video_gen.engine_runtime, "engine", {"return_value": MagicMock()}),
                (video_gen.engine_runtime, "video_part", {"side_effect": wan}),
                (video_gen.lipsync_proc, "run_dialogue", {"side_effect": lip_sync}),
                (video_gen, "_lipsync_to_output_fps", {"side_effect": retime}),
                (ffmpeg_utils, "convert_fps", {"side_effect": _copy}),
                (ffmpeg_utils, "mix_voice_and_background", {"side_effect": mix}),
                (ffmpeg_utils, "combine_video_audio",
                 {"side_effect": lambda source, _audio, output, **_options: _copy(source, output)}),
                (ffmpeg_utils, "probe_duration", {"return_value": 3.0}),
                (ffmpeg_utils, "video_fps", {"return_value": 30.0})):
            stack.enter_context(patch.object(target, name, **options))
        stack.enter_context(patch.object(prefs, "_FILE", self.root / "prefs.json"))

    def scene(self, members, cues, style, folder):
        self.events.clear(), self.synced.clear(), self.part_fps.clear(), self.retimed.clear(), self.prompts.clear()
        self.ensure.reset_mock()
        return video_gen.generate(PICTURE, 3, "Portrait 9:16", seed=123, scene_dir=self.root / folder, cast=members,
                                  exact_cast=True, cues=cues, background_text="steady rain",
                                  style=styles.get(style).video)

    def test_a_creature_speaking_alone_keeps_its_picture_with_a_note_in_any_style(self):
        """Its lines play over a picture made like a silent one: MuseTalk is not even installed for them, and a
        film whose style is Realistic does not fail the scene either."""
        for style in (styles.ANIMATION_3D, styles.REALISTIC, styles.custom_id(styles.REALISTIC, "film noir")):
            with self.subTest(style=style):
                output = self.scene([CHAIN], (WOOF,), style, folder=f"alone_{len(style)}")
                self.assertTrue(output.ok, output.error)
                self.assertNotIn("lipsync", self.events)
                self.assertEqual(output.info["lipsync_skipped"], NOTE)  # what the Timeline shows in its notes
                self.assertIn(NOTE, output.notes)
                self.assertEqual((self.part_fps, self.retimed), ([30], []))
                self.assertEqual(output.info["dialogue"], [WOOF.text])
                self.assertTrue(output.audio_stems[1].is_file())  # the speech still plays
                self.assertNotIn(video_gen.lipsync_proc.MODEL_ID, self.ensure.call_args_list[0].args[0])
                self.assertIn(f"Chain Dog looks like this: {CHAIN_LOOK}", self.prompts[0])
                # a Timeline job encodes its scenes' prompts ahead (video_gen.scene_wan_prompt): the same words
                self.assertEqual(self.prompts, [video_gen.scene_wan_prompt(PICTURE, styles.get(style).video, [CHAIN],
                                                                           exact_cast=True)])

    def test_a_mixed_scene_lip_syncs_only_the_people(self):
        output = self.scene([OLA, CHAIN], (Cue("Hi, Chain Dog.", "ola", 0.2), Cue("Woof!", "chain_dog", 1.6)),
                            styles.REALISTIC, folder="mixed")
        self.assertTrue(output.ok, output.error)
        (timings, who, lines), = self.synced
        self.assertEqual((who, [start for start, _ in timings]), (["female#char:ola"], [0.2]))
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].endswith("_speech1.wav"), lines)  # Ola's own line file, not Chain Dog's
        self.assertEqual(output.info["lipsync_skipped"], NOTE)
        self.assertEqual((self.part_fps, self.retimed), ([25], [3.0]))  # a person speaks: made at the lip-sync rate
        self.assertEqual(output.info["dialogue"], ["Hi, Chain Dog.", "Woof!"])
        self.assertIn(video_gen.lipsync_proc.MODEL_ID, self.ensure.call_args_list[0].args[0])
        self.assertIn("The creature looks like this", self.prompts[0])

    def test_a_narrator_in_an_animated_film_moves_no_mouth(self):
        """Contsy Stories: a library voice without a character is a voice-over in an animated style, so no lip sync
        runs (and no lip-sync frame rate); in a Realistic film it is lip-synced as before."""
        narrator = (Cue("Once upon a time, Ola lived by the sea.", "@adam", 0.2),)
        for style in (styles.ANIMATION_3D, styles.CARTOON):
            with self.subTest(style=style):
                output = self.scene([OLA], narrator, style, folder=f"narrated_{style}")
                self.assertTrue(output.ok, output.error)
                self.assertEqual((self.synced, self.part_fps), ([], [30]))
                self.assertEqual(output.info["dialogue"], [narrator[0].text])
        output = self.scene([OLA], narrator, styles.REALISTIC, folder="narrated_realistic")
        self.assertTrue(output.ok, output.error)
        self.assertEqual(len(self.synced), 1)

    def test_a_people_scene_is_lip_synced_as_before(self):
        output = self.scene([OLA, CHAIN], (Cue("Hi.", "ola", 0.2),), styles.REALISTIC, folder="people")
        self.assertTrue(output.ok, output.error)
        self.assertEqual([who for _, who, _ in self.synced], [["female#char:ola"]])
        self.assertNotIn("lipsync_skipped", output.info)
        self.assertFalse(any("Lip sync left out" in note for note in output.notes), output.notes)

    def test_a_creature_alone_in_an_animated_film_still_talks_through_infinitetalk(self):
        prompts = []

        def talk(_client, _start, _speech, output, **options):
            prompts.append(options["prompt"])
            Path(output).write_bytes(b"talking")

        self.talk.return_value = True
        with patch.object(video_gen.engine_runtime, "talk_part", side_effect=talk):
            output = self.scene([CHAIN], (WOOF,), styles.ANIMATION_3D, folder="talk")
        self.assertTrue(output.ok, output.error)
        self.assertEqual(output.info["lipsync"], "infinitetalk")
        self.assertEqual(self.events, [])  # neither Wan nor MuseTalk
        self.assertNotIn("lipsync_skipped", output.info)
        self.assertIn(f"Chain Dog looks like this: {CHAIN_LOOK}", prompts[0])
        with patch.object(video_gen.engine_runtime, "talk_part", side_effect=RuntimeError("out of memory")):
            fallback = self.scene([CHAIN], (WOOF,), styles.ANIMATION_3D, folder="fallback")
        self.assertTrue(fallback.ok, fallback.error)
        self.assertEqual(self.events, ["wan"])  # MuseTalk never runs for a creature
        self.assertTrue(any(note.endswith("the scene was made with Wan 2.2.") for note in fallback.notes),
                        fallback.notes)
        self.assertEqual(fallback.info["lipsync_skipped"], NOTE)

    def test_a_creatures_scene_is_the_base_of_its_reference_pictures(self):
        hat = self.root / "hat.png"
        hat.write_bytes(b"png")

        def compose(*args, **_options):
            Path(args[6]).write_bytes(b"png")
            return True, str(args[6])

        with patch.object(video_gen.llm, "describe_person", return_value=""), \
                patch.object(video_gen.refs, "compose_refs", side_effect=compose) as composed:
            output = video_gen.generate("Chain Dog wears the hat [1]", 3, "Portrait 9:16", seed=1,
                                        scene_dir=self.root / "hat", cast=[CHAIN], exact_cast=True,
                                        ref_images=[str(hat)], style=styles.get(styles.ANIMATION_3D).video,
                                        sound_on=False)
        self.assertTrue(output.ok, output.error)
        self.assertEqual(composed.call_args.kwargs["base_lead"], refs.CREATURE_CHARACTER)


if __name__ == "__main__":
    unittest.main()
