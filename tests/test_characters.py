# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Characters (app/characters.py): the stored character (photos and its own voice, checked by sha256), older
characters that still load, the creator's draft API (uploads through the adult / one-face gate, voices, an
atomic save, editing) and the photo generator (app/char_photos.py) with the models replaced (no GPU)."""
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app import char_photos, characters, llm, prefs, qc, queue_manager, tts, universes, voices
from engine.media import audio

ADULT = {"person": True, "age": 30, "gender": "woman", "caption": "a portrait"}


def picture(path: Path, size=(512, 640), color=(120, 100, 90)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def speech(path: Path, seconds: float, rate: int = 48000) -> Path:
    """A voiced-like tone with pauses removed by nothing: `seconds` of sound."""
    t = np.arange(int(seconds * rate)) / rate
    wave = (0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    audio.write_wav(path, wave[None, :], rate)
    return path


class _Library(unittest.TestCase):
    """Every store in a temporary folder; the vision gate, the face count and the pitch are replaced."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.builtin = self.root / "assets"
        self.builtin.mkdir()
        speech(self.builtin / "hanna.wav", 5.0)
        (self.builtin / "voices.json").write_text(json.dumps({"voices": [
            {"name": "Hanna", "file": "hanna.wav", "gender": "female", "description": "20s, bright"}]}),
            encoding="utf-8")
        for target in (patch.object(characters, "DIR", self.root / "characters"),
                       patch.object(characters, "DRAFTS_DIR", self.root / "drafts"),
                       patch("app.items.DIR", self.root / "items"),
                       patch.object(voices, "BUILTIN_DIR", self.builtin),
                       patch.object(voices, "MANIFEST", self.builtin / "voices.json"),
                       patch.object(voices, "USER_DIR", self.root / "voices"),
                       patch.object(voices, "measure_pitch", return_value=180.0),
                       patch.object(prefs, "_FILE", self.root / "prefs.json"),
                       patch.object(characters, "gate_models", return_value=None),
                       patch.object(queue_manager, "_JOBS", []),
                       patch.object(queue_manager, "_save")):
            target.start()
            self.addCleanup(target.stop)
        self.photo_check = patch.object(llm, "check_photo", return_value=ADULT)
        self.faces = patch.object(qc, "_faces", return_value=[object()])
        self.same_face = patch.object(qc, "face_score", return_value=0.9)  # an upload shows the first photo's person
        for started in (self.photo_check, self.faces, self.same_face):
            started.start()
            self.addCleanup(started.stop)

    def upload(self, draft, name="me.png", consent=True, **options):
        return characters.add_upload(draft, picture(self.root / name, **options), consent)

    def ready_draft(self, name="Mia", photos=2):
        draft = characters.new_draft()
        characters.update_draft(draft, name=name, gender="woman", look="Short red hair, freckles, green eyes")
        for index in range(photos):
            self.assertEqual(self.upload(draft, f"p{index}.png", color=(100 + index, 90, 80)), (True, "Photo added."))
        ok, message = characters.use_preset_voice(draft, "hanna")
        self.assertTrue(ok, message)
        return draft

    def written(self) -> list:
        """Every file under characters/ and voices/."""
        return sorted(p.relative_to(self.root).as_posix() for base in ("characters", "voices")
                      for p in (self.root / base).rglob("*") if p.is_file())


class StoredCharacterTests(_Library):
    def test_a_saved_character_has_its_photos_and_its_own_voice_in_its_folder(self):
        profile, message = characters.save_draft(self.ready_draft())
        self.assertIsNotNone(profile, message)
        folder = self.root / "characters" / "mia"
        self.assertEqual(self.written(), ["characters/mia/character.json", "characters/mia/photos/01.png",
                                          "characters/mia/photos/02.png", "characters/mia/voice/sample.wav"])
        record = json.loads((folder / "character.json").read_text(encoding="utf-8"))
        self.assertEqual((record["format"], record["name"], record["gender"]), (2, "Mia", "woman"))
        self.assertEqual([p["origin"] for p in record["photos"]], ["uploaded", "uploaded"])
        self.assertEqual((record["voice"]["origin"], record["voice"]["preset"]), ("preset", "hanna"))
        self.assertEqual(profile["photos"], [str(folder / "photos" / "01.png"), str(folder / "photos" / "02.png")])
        self.assertEqual(profile["voice"], "character:mia")
        self.assertEqual(characters.state(profile), "Ready")
        self.assertEqual(characters.reference_portrait(profile), folder / "photos" / "01.png")
        own = voices.get("character:mia")
        self.assertEqual((own.name, own.gender, Path(own.path)), ("Mia", "female", folder / "voice" / "sample.wav"))
        self.assertNotIn("character:mia", [v.slug for v in voices.all_voices()])  # not in the voice library
        self.assertEqual(voices.for_character(profile), ("character:mia", "female"))
        self.assertEqual(tts.resolve("character:mia").path, own.path)

    def test_a_changed_photo_or_voice_file_is_not_used(self):
        characters.save_draft(self.ready_draft())
        folder = self.root / "characters" / "mia"
        picture(folder / "photos" / "01.png", color=(1, 2, 3))
        (folder / "voice" / "sample.wav").write_bytes(b"other")
        profile = characters.get("mia")
        self.assertEqual(profile["photos"], [str(folder / "photos" / "02.png")])
        self.assertEqual(profile["voice"], "")
        self.assertIsNone(voices.get("character:mia"))
        (folder / "photos" / "02.png").unlink()
        self.assertIsNone(characters.usable("mia"))
        self.assertEqual(characters.state(characters.get("mia")), "Needs reference photos")

    def test_older_characters_still_load(self):
        library = self.root / "characters"
        reference = library / "rose"
        reference.mkdir(parents=True)
        portrait = picture(reference / "preview.png")
        digest = characters._digest(portrait)
        (reference / "character.json").write_text(json.dumps({
            "name": "Rose", "slug": "rose", "gender": "woman", "look": "red hair", "status": "reference",
            "portrait_sha256": digest, "voice": "hanna", "created": 1}), encoding="utf-8")
        trained = library / "ola"
        (trained / "photos").mkdir(parents=True)
        picture(trained / "photos" / "01.png")
        picture(trained / "preview.png")
        (trained / "character.json").write_text(json.dumps({
            "name": "Ola", "slug": "ola", "trigger": "ola_aist", "gender": "woman", "status": "gotowa",
            "photos": 16, "steps": 2400, "look": "", "created": 2, "voice": ""}), encoding="utf-8")
        rose, ola = characters.get("rose"), characters.get("ola")
        self.assertEqual((rose["photos"], rose["voice"], rose["legacy"]), ([str(portrait)], "hanna", "reference"))
        self.assertEqual(characters.usable("rose"), rose)
        self.assertEqual((ola["photos"], ola["legacy"]), ([], "lora"))
        self.assertIsNone(characters.usable("ola"))
        self.assertIn("removed LoRA training", characters.state(ola))
        self.assertEqual([c["slug"] for c in characters.all_characters()], ["rose", "ola"])
        self.assertEqual(characters.choices(), [(characters.DEFAULT_LABEL, ""), ("Rose", "rose")])
        # Timeline clip keys hash the stored record, as they did before: an unchanged older character keeps its clips
        stored = json.loads((reference / "character.json").read_text(encoding="utf-8"))
        self.assertEqual(characters.identity(rose), stored)

    def test_a_characters_identity_names_its_files_by_content_not_by_path(self):
        profile, _ = characters.save_draft(self.ready_draft())
        identity = characters.identity(profile)
        self.assertNotIn(str(self.root), json.dumps(identity).replace("\\\\", "\\"))
        folder = self.root / "characters" / "mia"
        self.assertEqual(identity["photos"],
                         [characters._digest(folder / "photos" / name) for name in ("01.png", "02.png")])
        self.assertEqual(identity["voice"], characters._digest(folder / "voice" / "sample.wav"))
        self.assertEqual(characters.identity({"slug": "x"}), {"slug": "x"})  # a bare profile is its own identity
        # saved again without a change (a new save time, every photo now "kept"): the clips keep their keys
        time.sleep(0.01)
        again, message = characters.save_draft(characters.load_draft("mia"))
        self.assertIsNotNone(again, message)
        self.assertNotEqual(again["record"], profile["record"])
        self.assertEqual(characters.identity(again), identity)
        changed, _ = characters.save_draft(characters.load_draft("mia"), look="Long black hair")
        self.assertNotEqual(characters.identity(changed), identity)  # the look changes what is generated

    def test_characters_that_no_longer_exist_are_named(self):
        characters.save_draft(self.ready_draft())
        self.assertEqual(characters.missing(["mia", "kuba", "", "kuba"]), ["kuba"])
        self.assertEqual(characters.missing_message(["mia"]), "")
        self.assertIn("The character kuba no longer exists", characters.missing_message(["mia", "kuba"]))
        self.assertIn("kuba, ola", characters.missing_message(["kuba", "ola"]))

    def test_a_library_voice_deleted_leaves_older_characters_on_the_default(self):
        folder = self.root / "characters" / "rose"
        folder.mkdir(parents=True)
        (folder / "character.json").write_text(json.dumps({"name": "Rose", "slug": "rose", "status": "reference",
                                                           "voice": "kasia"}), encoding="utf-8")
        characters.forget_library_voice("kasia")
        self.assertEqual(characters.get("rose")["voice"], "")

    def test_deleting_moves_the_folder_and_leaves_universes_and_the_top_picker(self):
        characters.save_draft(self.ready_draft("Mia"))
        characters.save_draft(self.ready_draft("Kuba"))
        characters.save_draft(self.ready_draft("Jan"))
        universes.save("Family", ["mia", "kuba", "jan"])
        prefs.save(character="mia")

        def to_bin(path):
            shutil.rmtree(path)
            return True

        with patch.object(characters.trash, "move", side_effect=to_bin):
            self.assertEqual(characters.delete("mia"), "Character ‘Mia’ moved to the Recycle Bin.")
        self.assertIsNone(characters.get("mia"))
        self.assertEqual(universes.get("family")["members"], ["kuba", "jan"])
        self.assertEqual(prefs.load()["character"], "")
        self.assertEqual(characters.delete("mia"), "Character not found.")


class CreatorTests(_Library):
    def test_nothing_reaches_the_library_before_save_and_leaving_discards_the_draft(self):
        draft = self.ready_draft()
        self.assertEqual(self.written(), [])
        view = characters.draft(draft)
        self.assertEqual((view["name"], view["gender"], len(view["photos"]), view["voice_info"]["origin"]),
                         ("Mia", "woman", 2, "preset"))
        self.assertTrue(all(Path(p).is_relative_to(self.root / "drafts") for p in [*view["photos"], view["voice"]]))
        characters.discard_draft(draft)
        self.assertFalse((self.root / "drafts" / draft).exists())
        with self.assertRaisesRegex(ValueError, "draft was closed"):
            characters.draft(draft)
        self.assertEqual(self.written(), [])

    def test_an_upload_needs_consent_and_one_adult_face(self):
        draft = characters.new_draft()
        self.assertFalse(self.upload(draft, consent=False)[0])
        cases = ((dict(ADULT, age=16), 1, "⛔"), (None, 1, "could not be checked"),
                 (dict(ADULT, person=False), 1, "No clear person"), (ADULT, 2, "exactly one clear face"),
                 (ADULT, 0, "exactly one clear face"))
        for answer, faces, reason in cases:
            with self.subTest(answer=answer, faces=faces), patch.object(llm, "check_photo", return_value=answer), \
                    patch.object(qc, "_faces", return_value=[object()] * faces):
                ok, message = self.upload(draft)
                self.assertFalse(ok)
                self.assertIn(reason, message)
        self.assertIn("too small", self.upload(draft, size=(200, 300))[1])
        (self.root / "text.png").write_text("not a picture")
        self.assertFalse(characters.add_upload(draft, self.root / "text.png", True)[0])
        self.assertEqual(characters.draft(draft)["photos"], [])
        self.assertEqual(sorted(p.name for p in (self.root / "drafts" / draft).iterdir()), ["draft.json"])

    def test_up_to_eight_photos_and_one_can_be_removed(self):
        draft = characters.new_draft()
        for index in range(characters.MAX_PHOTOS):
            self.assertTrue(self.upload(draft, f"{index}.png", color=(index, 0, 0))[0])
        self.assertIn("at most 8 photos", self.upload(draft, "9.png")[1])
        first, second = characters.draft(draft)["photos"][:2]
        characters.remove_photo(draft, 0)
        self.assertEqual(characters.draft(draft)["photos"][0], second)
        self.assertFalse(Path(first).exists())
        with self.assertRaisesRegex(ValueError, "no longer in the draft"):
            characters.remove_photo(draft, 9)

    def test_a_cloned_voice_needs_consent_and_three_seconds_of_speech(self):
        draft = characters.new_draft()
        long_sample = speech(self.root / "long.wav", 15.0)
        self.assertEqual(characters.clone_voice(draft, str(long_sample), False), (False, voices.CONSENT_MSG))
        ok, message = characters.clone_voice(draft, str(speech(self.root / "short.wav", 2.0)), True)
        self.assertFalse(ok)
        self.assertIn("Only 2.0 s of speech", message)
        ok, message = characters.clone_voice(draft, str(long_sample), True)
        self.assertTrue(ok, message)
        info = characters.draft(draft)["voice_info"]
        self.assertEqual((info["origin"], info["consent"]), ("cloned", True))
        self.assertLessEqual(info["seconds"], characters.CLONE_SECONDS[1] + 0.05)
        self.assertFalse(characters.use_preset_voice(draft, "nobody")[0])

    def test_save_checks_everything_and_writes_nothing_on_an_error(self):
        draft = self.ready_draft("Mia")
        with patch.object(llm, "blocked", return_value=True):
            self.assertEqual(characters.save_draft(draft), (None, llm.BLOCK_MSG))
        for name, reason in (("", "Enter a character name"), ("Woman", "already means a woman"),
                             ("Hanna", "name of a voice")):
            with self.subTest(name=name):
                profile, message = characters.save_draft(draft, name=name)
                self.assertIsNone(profile)
                self.assertIn(reason, message)
        self.assertIn("Only adult characters", characters.save_draft(draft, name="Mia", look="a 15-year-old girl")[1])
        characters.update_draft(draft, look="Short red hair")
        no_voice = characters.new_draft()
        characters.update_draft(no_voice, name="Ola", look="Black bob")
        self.assertIn("Add 1 to 8 reference photos", characters.save_draft(no_voice)[1])
        self.upload(no_voice, "ola.png")
        self.assertIn("Give the character a voice", characters.save_draft(no_voice)[1])
        self.assertEqual(self.written(), [])
        self.assertIsNotNone(characters.save_draft(draft)[0])
        other = self.ready_draft("mia")
        self.assertIn("already exists", characters.save_draft(other)[1])

    def test_save_waits_for_the_drafts_jobs(self):
        draft = self.ready_draft()
        queue_manager._JOBS.append(queue_manager.Job(1, characters.PHOTO_KIND, "photos", {"draft": draft}))
        self.assertIn("still being made", characters.save_draft(draft)[1])
        characters.discard_draft(draft)  # a waiting job of a discarded draft is withdrawn
        self.assertEqual(queue_manager._JOBS, [])

    def test_discarding_a_draft_stops_its_running_job_only(self):
        draft = self.ready_draft()
        mine = queue_manager.Job(1, characters.VOICE_KIND, "voice", {"draft": draft}, status=queue_manager.RUNNING)
        other = queue_manager.Job(2, characters.PHOTO_KIND, "photos", {"draft": "f" * 32})
        queue_manager._JOBS.extend([mine, other])
        with patch.object(queue_manager.cancel, "request") as stop:
            characters.discard_draft(draft)
        stop.assert_called_once_with()  # the running job of this draft; another draft's job stays queued
        self.assertEqual(queue_manager._JOBS, [mine, other])
        self.assertFalse(characters.DRAFTS_DIR.joinpath(draft).exists())

    def test_editing_replaces_the_character_in_one_step_and_keeps_its_slug(self):
        created, _ = characters.save_draft(self.ready_draft("Mia", photos=3))
        draft = characters.load_draft("mia")
        view = characters.draft(draft)
        self.assertEqual((view["slug"], view["name"], len(view["photos"]), view["voice_info"]["origin"]),
                         ("mia", "Mia", 3, "preset"))
        characters.remove_photo(draft, 2)
        time.sleep(0.01)
        profile, message = characters.save_draft(draft, name="Mia Rose")
        self.assertIsNotNone(profile, message)
        self.assertEqual((profile["slug"], profile["name"], len(profile["photos"])), ("mia", "Mia Rose", 2))
        self.assertEqual(profile["created"], created["created"])
        self.assertGreater(profile["updated"], created["updated"])
        self.assertEqual(sorted(p.name for p in (self.root / "characters").iterdir()), ["mia"])  # no leftover
        self.assertEqual(sorted(p.name for p in (self.root / "characters" / "mia" / "photos").iterdir()),
                         ["01.png", "02.png"])

    def test_an_older_reference_character_is_saved_in_the_new_format_with_its_voice(self):
        folder = self.root / "characters" / "rose"
        folder.mkdir(parents=True)
        portrait = picture(folder / "preview.png")
        (folder / "character.json").write_text(json.dumps({
            "name": "Rose", "slug": "rose", "gender": "woman", "look": "red hair", "status": "reference",
            "portrait_sha256": characters._digest(portrait), "voice": "hanna", "created": 5}), encoding="utf-8")
        profile, message = characters.save_draft(characters.load_draft("rose"))
        self.assertIsNotNone(profile, message)
        self.assertEqual((profile["voice"], len(profile["photos"]), profile["legacy"], profile["created"]),
                         ("character:rose", 1, "", 5))
        self.assertFalse((folder / "preview.png").exists())

    def legacy_lora(self, slug="ola") -> Path:
        """A character of the removed LoRA training: training photos with a caption, and a preview."""
        folder = self.root / "characters" / slug
        (folder / "photos").mkdir(parents=True)
        for number in range(1, 4):
            picture(folder / "photos" / f"{number:02d}.png", color=(number, 0, 0))
        (folder / "photos" / "01.txt").write_text("ola_aist, a woman", encoding="utf-8")
        picture(folder / "preview.png")
        (folder / "character.json").write_text(json.dumps({
            "name": "Ola", "slug": slug, "trigger": "ola_aist", "gender": "woman", "status": "gotowa", "photos": 3,
            "look": "", "created": 2, "voice": "hanna"}), encoding="utf-8")
        return folder

    def test_saving_an_older_lora_character_sends_its_old_folder_to_the_recycle_bin(self):
        folder = self.legacy_lora()
        binned = []

        def to_bin(path):
            binned.append(sorted(p.relative_to(path).as_posix() for p in Path(path).rglob("*") if p.is_file()))
            shutil.rmtree(path)
            return True

        draft = characters.load_draft("ola")
        self.assertEqual(characters.draft(draft)["photos"], [])  # training photos were never checked references
        self.upload(draft, "reference.png")
        with patch.object(characters.trash, "move", side_effect=to_bin):
            profile, message = characters.save_draft(draft, look="Short red hair")
        self.assertIsNotNone(profile, message)
        self.assertEqual(binned, [["character.json", "photos/01.png", "photos/01.txt", "photos/02.png",
                                   "photos/03.png", "preview.png"]])
        self.assertEqual(sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()),
                         ["character.json", "photos/01.png", "voice/sample.wav"])
        self.assertEqual([p.name for p in (self.root / "characters").iterdir()], ["ola"])  # nothing left over

    def test_an_older_folder_the_recycle_bin_refuses_is_kept_and_settled_at_the_next_start(self):
        self.legacy_lora()
        draft = characters.load_draft("ola")
        self.upload(draft, "reference.png")
        with patch.object(characters.trash, "move", return_value=False):
            profile, message = characters.save_draft(draft, look="Short red hair")
        self.assertIsNotNone(profile, message)
        self.assertIn("could not be moved to the Recycle Bin", message)
        kept = [p for p in (self.root / "characters").iterdir() if p.name.startswith(".replaced_ola_")]
        self.assertEqual(len(kept), 1)
        self.assertTrue((kept[0] / "photos" / "01.txt").is_file())  # never deleted
        with patch.object(characters.trash, "move", side_effect=lambda path: shutil.rmtree(path) or True) as move:
            characters.startup()
        move.assert_called_once_with(kept[0])
        self.assertEqual([p.name for p in (self.root / "characters").iterdir()], ["ola"])

    def test_a_save_cut_short_between_its_renames_is_settled_at_the_start(self):
        characters.save_draft(self.ready_draft("Mia"))
        characters.save_draft(self.ready_draft("Kuba"))
        library = self.root / "characters"
        (library / "mia").replace(library / ".replaced_mia_0123abcd")  # the old folder set aside, then the crash
        shutil.copytree(library / "kuba", library / ".replaced_kuba_89abcdef")  # the crash after the new one went in
        characters.new_draft()  # a draft of the previous run: no browser tab can reach it any more
        (characters.DRAFTS_DIR / ".staging_0123456789ab").mkdir()
        self.assertIsNone(characters.get("mia"))
        with patch.object(characters.trash, "move", side_effect=AssertionError("a current folder is not binned")):
            characters.startup()
        self.assertEqual(characters.get("mia")["name"], "Mia")
        self.assertEqual(sorted(p.name for p in library.iterdir()), ["kuba", "mia"])
        self.assertEqual(list(characters.DRAFTS_DIR.iterdir()), [])

    def test_a_character_deleted_while_it_is_edited_is_saved_as_a_new_one(self):
        characters.save_draft(self.ready_draft("Mia"))
        draft = characters.load_draft("mia")
        with patch.object(characters.trash, "move", side_effect=lambda path: shutil.rmtree(path) or True):
            characters.delete("mia")
        profile, message = characters.save_draft(draft)
        self.assertIsNotNone(profile, message)
        self.assertEqual((profile["slug"], profile["name"], len(profile["photos"])), ("mia", "Mia", 2))

    def test_an_upload_must_show_the_person_of_the_first_photo(self):
        draft = characters.new_draft()
        self.assertTrue(self.upload(draft, "a.png")[0])  # the first photo: nothing to compare it with
        first = characters.draft(draft)["photos"][0]
        for score, reason in ((0.2, "different person"), (None, "could not be compared")):
            with self.subTest(score=score), patch.object(qc, "face_score", return_value=score) as compared:
                ok, message = self.upload(draft, "b.png", color=(1, 2, 3))
                self.assertFalse(ok)
                self.assertIn(reason, message)
                self.assertEqual(compared.call_args.args[1]["photos"], [first])
        self.assertEqual(characters.draft(draft)["photos"], [first])

    def test_keeping_own_photos_needs_a_count_above_them(self):
        draft = self.ready_draft(photos=2)
        with self.assertRaisesRegex(ValueError, "already has 2 of its own photos"):
            characters.enqueue_photos(draft, 2, keep_own=True)
        self.assertEqual(queue_manager._JOBS, [])  # no GPU job that would make nothing
        self.assertTrue(characters.enqueue_photos(draft, 3, keep_own=True).params["keep_own"])

    def test_a_voice_description_must_describe_an_adult(self):
        draft = characters.new_draft()
        for text in ("a little girl, about ten years old, high playful voice", "a ｔｅｅｎ boy", "a 12-year-old",
                     "a te‍en"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "Only adult voices"):
                characters.enqueue_voice_design(draft, text)
        self.assertEqual(queue_manager._JOBS, [])
        refused = characters.run_voice_job({"draft": draft, "description": "a young girl", "gender": "woman"},
                                           lambda *_: None)
        self.assertIn("Only adult voices", refused[3])
        # VoxCPM2 always reads that the speaker is an adult; "girl" or "boy" never stand for the gender
        self.assertEqual(characters._voice_description("a girl with a raspy voice", "woman"),
                         "An adult woman, a girl with a raspy voice")
        self.assertEqual(characters._voice_description("a deep male voice", "man"), "An adult, a deep male voice")
        self.assertEqual(characters._voice_description("Adult woman, low alto", "woman"), "Adult woman, low alto")
        with patch.object(tts, "available", side_effect=AssertionError("nothing is designed")):
            self.assertIn("Only adult voices", tts.design_voice("a teenage girl", self.root / "x.wav")[1])

    def test_draft_jobs_do_not_run_after_a_restart(self):
        from app import items

        self.assertEqual(set(queue_manager.SESSION_KINDS), {characters.PHOTO_KIND, characters.VOICE_KIND,
                                                            items.PHOTO_KIND})
        state = self.root / "queue.json"
        draft = {"draft": "a" * 32}
        state.write_text(json.dumps({"jobs": [
            {"id": 1, "kind": characters.PHOTO_KIND, "label": "Photos", "params": {**draft, "count": 4},
             "status": queue_manager.RUNNING},
            {"id": 2, "kind": characters.VOICE_KIND, "label": "Voice", "params": {**draft, "description": "warm"},
             "status": queue_manager.PENDING},
            {"id": 3, "kind": characters.PHOTO_KIND, "label": "Photos", "params": {**draft, "count": 2},
             "status": queue_manager.DONE}]}), encoding="utf-8")
        with patch.object(queue_manager, "_STATE", state), \
                patch.object(queue_manager, "_STATE_INFO", {"next_id": 1, "paused": False}):
            queue_manager._load()
        jobs = {job.id: job for job in queue_manager.snapshot()}
        self.assertEqual([(jobs[i].status, jobs[i].error) for i in (1, 2)],
                         [(queue_manager.STOPPED, queue_manager.SESSION_MSG)] * 2)
        self.assertEqual(jobs[3].status, queue_manager.DONE)
        self.assertIsNone(queue_manager._next())

    def test_the_dashboard_names_universe_members_that_no_longer_exist(self):
        from app import ui_studio

        characters.save_draft(self.ready_draft("Ada"))
        folder = self.root / "characters" / "kit"
        folder.mkdir(parents=True)
        (folder / "character.json").write_text(json.dumps({"name": "Kit", "slug": "kit", "status": "gotowa"}),
                                               encoding="utf-8")
        universes._write([{"name": "Family", "slug": "family", "members": ["ada", "ola", "kit", "kuba"]}])
        head = ui_studio._universe_head("universe:family", ["ada"])
        self.assertIn("Universe <b>Family</b>: Ada. Without reference photos: Kit.", head)
        self.assertIn("These characters no longer exist and were left out: ola, kuba.", head)
        self.assertNotIn(", ,", head)
        none_ready = ui_studio._universe_head("universe:family", ["ada", "ola", "kit", "kuba"])
        self.assertIn("Universe <b>Family</b>: Ada.", none_ready)  # the stored members, when none is ready

    def test_jobs_check_their_input_and_report_a_closed_draft(self):
        draft = characters.new_draft()
        with self.assertRaisesRegex(ValueError, "Describe how the character looks"):
            characters.enqueue_photos(draft, 4)
        characters.update_draft(draft, look="Short red hair", gender="man")
        for count in (0, 9):
            with self.assertRaisesRegex(ValueError, "Choose 1 to 8"):
                characters.enqueue_photos(draft, count)
        with self.assertRaisesRegex(ValueError, "Upload a photo first"):
            characters.enqueue_photos(draft, 4, keep_own=True)
        job = characters.enqueue_photos(draft, 4)
        self.assertEqual((job.kind, job.params["count"], job.params["gender"]), ("character_photos", 4, "man"))
        self.assertEqual(job.estimate(), queue_manager.rate("character_photos") * 4)
        with self.assertRaisesRegex(ValueError, "Describe the voice"):
            characters.enqueue_voice_design(draft, "  ")
        self.assertEqual(characters.enqueue_voice_design(draft, "warm low voice").kind, "character_voice")
        characters.discard_draft(draft)
        self.assertEqual(characters.run_photo_job({"draft": draft, "count": 2}, lambda *_: None)[3],
                         "This character draft was closed. Start again.")
        self.assertFalse(characters.run_voice_job({"draft": draft, "description": "x"}, lambda *_: None)[0])

    def test_a_designed_voice_lands_in_the_draft(self):
        draft = characters.new_draft()
        characters.update_draft(draft, gender="man")
        asked = []

        def design(description, out, seed=None, progress_cb=None, text=None):
            asked.append(description)
            speech(Path(out), 5.0)
            return True, str(out)

        with patch.object(tts, "design_voice", side_effect=design):
            ok, _, notes, error = characters.run_voice_job(
                {"draft": draft, "description": "warm, low and calm", "gender": "man"}, lambda *_: None)
        self.assertTrue(ok, error)
        self.assertEqual(asked, ["An adult man, warm, low and calm"])
        info = characters.draft(draft)["voice_info"]
        self.assertEqual((info["origin"], info["description"]), ("designed", "warm, low and calm"))
        self.assertEqual([p.name for p in (self.root / "drafts" / draft).glob(".*")], [])  # no work files left
        with patch.object(tts, "design_voice", return_value=(False, "no usable take")):
            self.assertEqual(characters.run_voice_job({"draft": draft, "description": "x", "gender": "man"},
                                                      lambda *_: None)[3], "no usable take")


class PhotoGeneratorTests(_Library):
    """app/char_photos.py: the anchor from Z-Image, the views from Qwen-Image-Edit of the anchor, each through the
    gate and the SFace comparison with the anchor."""

    def setUp(self):
        super().setUp()
        self.draft = characters.new_draft()
        characters.update_draft(self.draft, name="Mia", gender="woman", look="Short red hair, freckles")
        self.scores = []
        self.compose_calls = []
        for target in (patch.object(char_photos.model_manager, "ensure", return_value=None),
                       patch.object(char_photos.engine_runtime, "engine", return_value=Mock()),
                       patch.object(char_photos.engine_runtime, "render", side_effect=self.render),
                       patch.object(char_photos.refs, "compose", side_effect=self.compose),
                       patch.object(char_photos.qc, "face_score", side_effect=self.score)):
            target.start()
            self.addCleanup(target.stop)

    def render(self, prompt, width, height, seed, steps, runtime=None):
        self.anchor_prompt = prompt
        import io

        buffer = io.BytesIO()
        Image.new("RGB", (width, height), (seed % 200, 50, 50)).save(buffer, "PNG")
        return buffer.getvalue()

    def compose(self, client, images, prompt, width, height, out, seed, progress_cb=None):
        self.compose_calls.append((images, prompt))
        picture(Path(out), (width, height), (seed % 200, 60, 60))
        return True, str(out)

    def score(self, path, ch):
        self.assertEqual(len(ch["photos"]), 1)  # compared with the anchor only
        return self.scores.pop(0) if self.scores else 0.9

    def test_a_new_person_is_an_anchor_and_views_of_it(self):
        notes = char_photos.generate(self.draft, 4, False, lambda *_: None)
        view = characters.draft(self.draft)
        self.assertEqual((len(view["photos"]), set(view["origins"])), (4, {"generated"}))
        self.assertIn("adult woman. Short red hair, freckles", self.anchor_prompt)
        anchor = view["photos"][0]
        self.assertEqual([images for images, _ in self.compose_calls], [[anchor]] * 3)
        self.assertTrue(all(prompt.startswith("The same person as in image 1") for _, prompt in self.compose_calls))
        self.assertIn("4 reference photos ready", notes[0])

    def test_a_view_that_loses_the_face_is_drawn_again_then_left_out(self):
        self.scores = [0.2, 0.8] + [0.1] * char_photos.MAX_TRIES
        notes = char_photos.generate(self.draft, 3, False, lambda *_: None)
        self.assertEqual(len(characters.draft(self.draft)["photos"]), 2)
        self.assertEqual(len(self.compose_calls), 2 + char_photos.MAX_TRIES)
        self.assertTrue(any("did not match the first photo" in note for note in notes))

    def test_a_photo_that_looks_like_a_minor_stops_the_job_and_keeps_the_draft(self):
        self.upload(self.draft, "mine.png")
        with patch.object(llm, "check_photo", return_value=dict(ADULT, age=15)), \
                self.assertRaisesRegex(ValueError, "⛔"):
            char_photos.generate(self.draft, 3, False, lambda *_: None)
        view = characters.draft(self.draft)
        self.assertEqual(view["origins"], ["uploaded"])
        self.assertEqual(len(list((self.root / "drafts" / self.draft).glob("p_*.png"))), 1)

    def test_own_photos_stay_and_views_are_made_of_the_first(self):
        self.upload(self.draft, "mine.png")
        mine = characters.draft(self.draft)["photos"][0]
        char_photos.generate(self.draft, 3, True, lambda *_: None)
        view = characters.draft(self.draft)
        self.assertEqual((view["photos"][0], view["origins"]), (mine, ["uploaded", "generated", "generated"]))
        self.assertFalse(hasattr(self, "anchor_prompt"))  # no new person was drawn
        self.assertEqual({tuple(images) for images, _ in self.compose_calls}, {(mine,)})

    def test_one_photo_is_generated_again_in_its_place(self):
        char_photos.generate(self.draft, 3, False, lambda *_: None)
        before = characters.draft(self.draft)["photos"]
        char_photos.regenerate(self.draft, 1, lambda *_: None)
        after = characters.draft(self.draft)["photos"]
        self.assertEqual((after[0], after[2]), (before[0], before[2]))
        self.assertNotEqual(after[1], before[1])
        self.assertFalse(Path(before[1]).exists())
        char_photos.regenerate(self.draft, 0, lambda *_: None)  # the anchor: a frontal view of photo 2
        self.assertEqual(self.compose_calls[-1][0], [after[1]])
        self.upload(self.draft, "mine.png")
        with self.assertRaisesRegex(ValueError, "Only generated photos"):
            characters.enqueue_regenerate(self.draft, 3)

    def test_the_only_photo_is_drawn_again_only_from_an_adult_look(self):
        char_photos.generate(self.draft, 1, False, lambda *_: None)
        drawn = self.anchor_prompt
        characters.update_draft(self.draft, look="a 15-year-old girl with long brown hair")  # changed afterwards
        with self.assertRaisesRegex(ValueError, "Only adult characters"):
            characters.enqueue_regenerate(self.draft, 0)
        with self.assertRaisesRegex(ValueError, "Only adult characters"):  # a job queued before the change
            char_photos.regenerate(self.draft, 0, lambda *_: None)
        self.assertEqual(self.anchor_prompt, drawn)  # nothing was drawn from that look


class RenderTests(unittest.TestCase):
    def test_a_character_keeps_the_clothing_the_prompt_states(self):
        prompts = []

        def compose(client, sources, prompt, *args, **kwargs):
            prompts.append(prompt)
            return True, "frame.png"

        mia = {"slug": "mia", "name": "Mia", "gender": "woman", "look": "", "photos": ["01.png"]}
        with tempfile.TemporaryDirectory() as folder, patch.object(characters.refs, "compose", side_effect=compose):
            for text in ("Mia lies nude on the bed", "Mia on the beach"):
                characters.render(Mock(), mia, text, 832, 1216, Path(folder) / "frame.png", seed=3)
        self.assertNotIn("Wearing", prompts[0])
        self.assertTrue(prompts[0].endswith("Mia lies nude on the bed"), prompts[0])
        self.assertIn(" Wearing a black one-piece swimsuit.", prompts[1])  # silent about clothing: the default

    def test_a_man_gets_a_mans_outfit(self):
        self.assertEqual(characters.outfit_for("a walk on the beach", "man"),
                         "black swim shorts and an open short-sleeve linen shirt")
        self.assertEqual(characters.outfit_for("reading at a desk", "man"), "a navy crew-neck sweater and dark jeans")
        self.assertEqual(characters.outfit_for("close-up with natural skin texture"),
                         "a fitted black knit dress with long sleeves")  # "skin" is not skiing

    def test_a_look_that_names_a_minor_is_refused(self):
        self.assertIn("Only adult characters", characters.check_look("a 15-year-old girl with long hair"))
        for hidden in ("a ｔｅｅｎ with long hair", "a te‍en with long hair"):  # full-width, zero-width joiner
            self.assertIn("Only adult characters", characters.check_look(hidden))
        self.assertEqual(characters.check_look("Short red hair, freckles"), "")
        self.assertIn("Describe how the character looks", characters.check_look("  "))


if __name__ == "__main__":
    unittest.main()
