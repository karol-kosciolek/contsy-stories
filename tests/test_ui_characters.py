# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Characters page (app/ui_characters.py): the one-screen creator's layout and wiring, and its handlers run
on a real draft (app/characters.py) in temporary folders, with the photo gate and the queue's worker replaced.
Nothing reaches the character list or a voice list before Save; leaving the page discards the draft."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from app import characters, main, prefs, queue_manager, studio, style_library, styles, ui_characters, ui_theme
from app.ui_characters import Creator
from app.vram import GpuInfo
from tests.test_characters import _Library, picture, speech

SKIP = gr.skip()
STATE = ["draft", "watch", *ui_characters.DISPLAY, "shown"]
LOOK = "Late twenties, short red hair, freckles, green eyes"


def named(result, extra=()) -> dict:
    """A creator handler's outputs by name (STATE, then `extra`)."""
    names = [*STATE, *extra]
    assert len(result) == len(names), (len(result), len(names))
    return dict(zip(names, result))


class CreatorFlowTests(_Library):
    def setUp(self):
        super().setUp()
        cache = patch.object(ui_theme, "get_upload_folder", return_value=str(self.root / "cache"))
        cache.start()
        self.addCleanup(cache.stop)

    def step(self, out: dict, handler, *values, extra=()):
        """Run `handler` the way the page does: the tab's draft, jobs and display, then the form values."""
        return named(handler(out["draft"], out["watch"], out["shown"], *values), extra)

    def start(self) -> dict:
        return {"draft": "", "watch": {}, "shown": ui_characters._view("", {}, "", "")}

    def tick(self, out: dict, name: str, look: str, pick=None) -> dict:
        """The timer, or typing in the name or look: the same draft, a fresh display."""
        return named((out["draft"], *ui_characters.refresh_creator(out["draft"], out["watch"], out["shown"], name,
                                                                    look, pick)))

    def finish_photo_job(self, out: dict, count: int) -> queue_manager.Job:
        """What the queue's photo job does: generated photos land in the draft, the job is done."""
        job = queue_manager._JOBS[-1]
        made = [(picture(characters.new_photo_path(out["draft"]), color=(90 + n, 80, 70)), "generated")
                for n in range(count)]
        characters.replace_photos(out["draft"], made)
        job.status, job.notes = queue_manager.DONE, [f"{count} reference photos ready. Review them before saving.",
                                                    "Photo 3 was left out: the face did not match the first photo."]
        return job

    def test_generating_photos_runs_in_the_queue_and_the_screen_follows_it(self):
        out = self.step(self.start(), ui_characters.generate_photos, "", "woman", "", 4, True, None)
        self.assertIn("Describe how the character looks", out["photo_status"])
        self.assertEqual(queue_manager._JOBS, [])
        out = self.step(out, ui_characters.generate_photos, "Mia", "woman", LOOK, 3, True, None)
        job = queue_manager._JOBS[-1]
        self.assertEqual((job.kind, job.params["count"], job.params["look"]), (characters.PHOTO_KIND, 3, LOOK))
        self.assertIn("Waiting in the GPU queue", out["photo_status"])
        self.assertIs(out["ticker"].active, True)  # the screen follows the job only while it waits or runs
        self.assertIs(out["generate"]["interactive"], False)
        self.assertIs(out["upload"]["interactive"], False)  # an upload now would be lost when the job ends
        self.assertIn("Save is ready once", out["checklist"])

        job.status, job.progress, job.desc = queue_manager.RUNNING, 0.4, "Photo 2/3..."
        out = self.tick(out, "Mia", LOOK)
        self.assertIn("Making the photos… 40% · Photo 2/3", out["photo_status"])

        self.finish_photo_job(out, 2)
        out = self.tick(out, "Mia", LOOK)
        self.assertEqual(len(out["photos"]["value"]), 2)
        self.assertTrue(out["photos"]["visible"])
        self.assertEqual([caption for _, caption in out["photos"]["value"]], ["1 · main", "2"])
        self.assertEqual(out["pick"]["choices"], [("1", 0), ("2", 1)])
        self.assertIn("2 reference photos ready", out["photo_status"])
        self.assertIn("Photo 3 was left out", out["photo_status"])
        self.assertIs(out["ticker"].active, False)
        self.assertEqual(out["generate"]["value"], "Generate all again")
        self.assertIs(out["regenerate"]["visible"], True)
        self.assertEqual(self.written(), [])  # nothing reaches the character list before Save

        again = self.step(out, ui_characters.regenerate_photo, "Mia", "woman", LOOK, None)
        self.assertIn("Select a photo first", again["photo_status"])
        again = self.step(out, ui_characters.regenerate_photo, "Mia", "woman", LOOK, 1)
        self.assertEqual(queue_manager._JOBS[-1].params["index"], 1)
        self.assertIn("Waiting in the GPU queue", again["photo_status"])

    def test_a_progress_update_redraws_only_what_changed(self):
        out = self.step(self.start(), ui_characters.generate_photos, "Mia", "woman", LOOK, 2, True, None)
        same = self.tick(out, "Mia", LOOK)
        for part in ui_characters.DISPLAY:
            self.assertEqual(same[part], SKIP, part)  # a playing voice or an open photo is not reset

    def test_own_photos_need_consent_and_pass_the_gate(self):
        photo = picture(self.root / "me.png")
        out = self.step(self.start(), ui_characters.upload_photos, "", "woman", LOOK, [str(photo)], False, None)
        self.assertIn("Tick the box", out["photo_status"])
        self.assertEqual(out["draft"], "")
        out = self.step(out, ui_characters.upload_photos, "", "woman", LOOK, [str(photo)], True, None)
        self.assertIn("1 photo added", out["photo_status"])
        self.assertEqual([caption for _, caption in out["photos"]["value"]], ["1 · main · your photo"])
        self.assertIs(out["keep"]["visible"], True)  # Generate may keep the own photo and add views of it
        tiny = picture(self.root / "tiny.png", size=(100, 100))
        refused = self.step(out, ui_characters.upload_photos, "", "woman", LOOK, [str(tiny)], True, None)
        self.assertIn("No photo was added", refused["photo_status"])
        self.assertIn("tiny.png: The photo is too small", refused["photo_status"])
        with patch.object(characters.llm, "check_photo", return_value={**self.ADULT, "age": 15}):
            minor = self.step(out, ui_characters.upload_photos, "", "woman", LOOK, [str(photo)], True, None)
        self.assertIn("appears to show a minor", minor["photo_status"])
        self.assertNotIn("⛔", ui_theme.note(minor["photo_status"]))

        removed = self.step(out, ui_characters.remove_photo, "", LOOK, 0)
        self.assertIn("Photo 1 removed", removed["photo_status"])
        self.assertIsNone(removed["pick"]["value"])  # a second click removes nothing by mistake
        self.assertIs(removed["photos"]["visible"], False)
        self.assertEqual(self.written(), [])

    ADULT = {"person": True, "age": 30, "gender": "woman", "caption": "a portrait"}

    def test_the_voice_is_designed_cloned_or_started_from_a_bundled_one(self):
        out = self.step(self.start(), ui_characters.design_voice, "Mia", "woman", LOOK, "  ", None)
        self.assertIn("Describe the voice first", out["voice_status"])
        out = self.step(out, ui_characters.design_voice, "Mia", "man", LOOK, "warm and low", None)
        job = queue_manager._JOBS[-1]
        self.assertEqual((job.kind, job.params["gender"]), (characters.VOICE_KIND, "man"))
        self.assertIn("Waiting in the GPU queue", out["voice_status"])
        self.assertIs(out["clone"]["interactive"], False)
        queue_manager._JOBS.clear()

        sample = speech(self.root / "me.wav", 4.0)
        refused = self.step(out, ui_characters.clone_voice, "Mia", "woman", LOOK, str(sample), False, None,
                            extra=["sample"])
        self.assertIn("Confirm that the sample", refused["voice_status"])
        self.assertEqual(refused["sample"], SKIP)
        cloned = self.step(out, ui_characters.clone_voice, "Mia", "woman", LOOK, str(sample), True, None,
                           extra=["sample"])
        self.assertIn("Voice cloned", cloned["voice_status"])
        self.assertIsNone(cloned["sample"])
        self.assertIn("Cloned from your recording", cloned["voice_about"])
        self.assertTrue(Path(cloned["voice"]).is_file())

        bundled = self.step(cloned, ui_characters.use_preset, "Mia", "woman", LOOK, "hanna", None)
        self.assertIn("Started from Hanna", bundled["voice_about"])
        self.assertNotEqual(bundled["voice"], SKIP)  # the same file name, new content: the player gets it
        self.assertEqual(self.step(bundled, ui_characters.use_preset, "Mia", "woman", LOOK, None, None)
                         ["voice_status"].count("Choose a bundled voice"), 1)
        self.assertEqual(self.written(), [])  # the voice is in the draft, not in any voice list

    def test_save_waits_for_name_look_photo_and_voice_then_writes_everything_together(self):
        out = self.start()
        self.assertIn("To save, add a look description, at least one photo, a name and a voice.",
                      out["shown"].checklist)
        self.assertFalse(out["shown"].can_save)
        out = self.step(out, ui_characters.upload_photos, "", "woman", LOOK, [str(picture(self.root / "a.png"))],
                        True, None)
        out = self.step(out, ui_characters.use_preset, "", "woman", LOOK, "hanna", None)
        self.assertIn("To save, add a name.", out["checklist"])
        self.assertEqual(out["save"], SKIP)  # still not ready: no redraw
        typed = self.tick(out, "Mia", LOOK)
        self.assertIs(typed["save"]["interactive"], True)
        self.assertIn("Ready to save: 1 photo", typed["checklist"])
        self.assertEqual(self.written(), [])

        library = ["form:" + name for name in ui_characters.FORM_NAMES] + ["cards", "listing", "pick", "assets",
                                                                           "top"]
        with patch.object(characters, "check_name", return_value="A character named ‘Mia’ already exists."):
            refused = self.step(typed, ui_characters.save_character, "Mia", "woman", LOOK, None, extra=library)
        self.assertIn("already exists", refused["checklist"])
        self.assertTrue(all(refused[name] == SKIP for name in library))
        self.assertEqual(self.written(), [])
        cleared = self.tick(refused, "Mia Rose", LOOK)
        self.assertNotIn("already exists", cleared["checklist"])  # a new name: the old refusal is gone

        saved = self.step(typed, ui_characters.save_character, "Mia", "woman", LOOK, None, extra=library)
        self.assertEqual(saved["draft"], "")
        self.assertIn("Character ‘Mia’ saved with 1 photo and its own voice", saved["checklist"])
        self.assertEqual(saved["form:name"], "")
        self.assertEqual(saved["pick"]["value"], "mia")
        self.assertIn("Mia", saved["listing"])
        self.assertEqual(self.written(), ["characters/mia/character.json", "characters/mia/photos/01.png",
                                          "characters/mia/voice/sample.wav"])
        self.assertEqual(characters.get("mia")["voice"], "character:mia")
        self.assertFalse(characters.DRAFTS_DIR.joinpath(typed["draft"]).exists())

    def test_editing_uses_the_same_screen_and_marks_unsaved_work(self):
        profile, _ = characters.save_draft(self.ready_draft("Mia", photos=2))
        form = [f"form:{name}" for name in ui_characters.FORM_NAMES]
        cancelled = named(ui_characters.edit_character("no", "mia", "", {}, Creator()), [*form, "info"])
        self.assertTrue(all(value == SKIP for value in cancelled.values()))  # the list keeps its own message
        out = named(ui_characters.edit_character("yes", "mia", "", {}, Creator()), [*form, "info"])
        self.assertIn("“Mia” is open in the creator", out["info"])
        self.assertIn("Edit “Mia”", out["heading"])
        self.assertNotIn("data-unsaved", out["heading"])  # nothing changed yet: leaving does not ask
        self.assertEqual((out["form:name"], out["form:gender"]), ("Mia", "woman"))
        self.assertEqual(out["form:count"], 4)  # above its 2 kept photos (at least the default): Generate adds views
        self.assertEqual(out["generate"]["value"], "Generate photos")  # "keep" decides: views or a new person
        self.assertEqual([caption for _, caption in out["photos"]["value"]], ["1 · main · saved", "2 · saved"])
        self.assertIn("Started from Hanna", out["voice_about"])
        changed = self.step(out, ui_characters.remove_photo, "Mia", LOOK, 1)
        self.assertIn("data-unsaved", changed["heading"])
        self.assertIn(f"#{ui_characters.CREATOR} [data-unsaved]", ui_characters.CONFIRM_JS)
        regen = self.step(changed, ui_characters.regenerate_photo, "Mia", "woman", LOOK, 0)
        self.assertIn("Only generated photos can be generated again", regen["photo_status"])

    def test_a_character_in_a_style_of_the_owners_own_keeps_it_in_the_creator_and_the_list(self):
        storybook = styles.custom_id(styles.ANIME, "soft pastel watercolor")
        with patch.dict(style_library._CACHE, clear=True):
            entry, _ = style_library.save("Storybook", styles.ANIME, "soft pastel watercolor")
            draft = self.ready_draft("Mia", photos=2)
            characters.update_draft(draft, style=storybook)
            characters.save_draft(draft)
            form = [f"form:{name}" for name in ui_characters.FORM_NAMES]
            out = named(ui_characters.edit_character("yes", "mia", "", {}, Creator()), [*form, "info"])
            self.assertEqual(out["form:style"]["value"], storybook)
            self.assertEqual(out["form:style"]["choices"], [*[c for c in styles.choices() if c[1] != styles.REALISTIC], ("Storybook", storybook)])
            _gallery, listing, _names = ui_characters._rows()
            self.assertIn("<b>Mia</b><br><span class='muted'>Storybook</span>", listing)  # the style under the name
            self.assertIn(">Storybook<", ui_characters._character_assets("mia"))
            style_library.delete(entry["key"])  # the character keeps its look, and the menu still lists it
            out = named(ui_characters.edit_character("yes", "mia", "", {}, Creator()), [*form, "info"])
            self.assertEqual(out["form:style"]["choices"][-1], ("Custom: soft pastel watercolor", storybook))

    def test_photos_or_a_voice_being_made_count_as_unsaved_work(self):
        out = self.step(self.start(), ui_characters.generate_photos, "Mia", "woman", LOOK, 3, True, None)
        self.assertIn("data-unsaved", out["heading"])  # leaving now would stop the job: the page asks first
        self.assertIn("still being made are stopped", ui_characters.DISCARD_QUESTION)
        self.assertIn("still being made are stopped", ui_theme.JS)
        queue_manager._JOBS.clear()
        typed = self.tick(out, "", LOOK)
        self.assertIn("data-unsaved", typed["shown"].heading)  # the draft holds a typed look: leaving clears it
        self.assertFalse(ui_characters._unsaved(None, "Mia", LOOK))  # no draft: leaving keeps the fields

    def test_an_older_character_opened_unchanged_is_not_unsaved_work(self):
        folder = self.root / "characters" / "rose"
        folder.mkdir(parents=True)
        portrait = picture(folder / "preview.png")
        (folder / "character.json").write_text(json.dumps({
            "name": "Rose", "slug": "rose", "gender": "woman", "look": "Red hair", "status": "reference",
            "portrait_sha256": characters._digest(portrait), "voice": "hanna", "created": 1}), encoding="utf-8")
        form = [f"form:{name}" for name in ui_characters.FORM_NAMES]
        out = named(ui_characters.edit_character("yes", "rose", "", {}, Creator()), [*form, "info"])
        self.assertIn("Edit “Rose”", out["heading"])
        self.assertNotIn("data-unsaved", out["heading"])  # its library voice was copied, nothing else changed
        changed = self.step(out, ui_characters.use_preset, "Rose", "woman", "Red hair", "hanna", None)
        self.assertNotIn("data-unsaved", changed["heading"])  # the same bundled voice again
        renamed = self.tick(changed, "Rosa", "Red hair")
        self.assertIn("data-unsaved", renamed["heading"])

    def test_leaving_discarding_or_closing_the_tab_throws_the_draft_away(self):
        out = self.step(self.start(), ui_characters.upload_photos, "", "woman", LOOK,
                        [str(picture(self.root / "a.png"))], True, None)
        folder = characters.DRAFTS_DIR / out["draft"]
        self.assertTrue(folder.exists())
        self.assertTrue(all(value == SKIP for value in ui_characters.leave("", {}, Creator())))
        kept = ui_characters.start_over("no", out["draft"], out["watch"], out["shown"])
        self.assertTrue(all(value == SKIP for value in kept))
        self.assertTrue(folder.exists())
        form = [f"form:{name}" for name in ui_characters.FORM_NAMES]
        left = named(ui_characters.leave(out["draft"], out["watch"], out["shown"]), form)
        self.assertEqual((left["draft"], left["watch"], left["form:look"]), ("", {}, ""))
        self.assertIn("New character", left["heading"])
        self.assertFalse(folder.exists())
        other = self.step(self.start(), ui_characters.use_preset, "", "woman", LOOK, "hanna", None)
        ui_characters._forget_draft(other["draft"])  # Gradio's state deletion after the tab closed
        self.assertFalse((characters.DRAFTS_DIR / other["draft"]).exists())
        ui_characters._forget_draft(None)


class CharactersPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(prefs, "_FILE", Path(tmp) / "prefs.json"), \
                patch.object(style_library, "all_custom", return_value=[]), \
                patch.object(characters, "all_characters", return_value=[]), \
                patch.object(studio, "results", return_value=[]), \
                patch("app.ui_settings.model_licenses_html", return_value=""), \
                patch.object(main, "get_gpu_info", return_value=GpuInfo(True, "RTX", total_gb=16.0, free_gb=9.0)):
            cls.demo = main.build_ui()
        cls.blocks = list(cls.demo.blocks.values())
        cls.page = next(b for b in cls.blocks if getattr(b, "elem_id", None) == "characters-page")

    def inside(self, block, ancestor) -> bool:
        while block is not None:
            if block is ancestor:
                return True
            block = getattr(block, "parent", None)
        return False

    def on_page(self):
        return [block for block in self.blocks if self.inside(block, self.page)]

    def by_label(self, label):
        return next(block for block in self.on_page() if getattr(block, "label", None) == label
                    or getattr(block, "value", None) == label)

    def test_the_creator_runs_top_to_bottom_beside_the_list(self):
        creator = next(b for b in self.blocks if getattr(b, "elem_id", None) == ui_characters.CREATOR)
        order = [b for b in self.on_page() if self.inside(b, creator)]
        steps = ["Gender", "Look", "Number of photos", "Generate photos", "Generate this photo again",
                 "Remove this photo", characters.CONSENT, "Upload your own photos", "Name", "Describe the voice",
                 "Generate voice", "Clone voice", "Use this voice", "Character voice", "Save character", "Discard"]
        positions = [order.index(self.by_label(step)) for step in steps]
        self.assertEqual(positions, sorted(positions))
        self.assertIs(self.by_label("Save character").interactive, False)  # nothing to save yet
        self.assertIs(self.by_label("Character voice").interactive, False)  # a player
        upload = self.by_label("Upload your own photos")
        self.assertEqual((upload.file_types, upload.file_count), (["image"], "multiple"))
        listing = next(b for b in self.blocks if getattr(b, "elem_id", None) == "char-cards")
        self.assertIs(listing.parent.parent, creator.parent)  # the list's card is beside the creator
        self.assertIsInstance(listing.parent, gr.Column)

    def test_every_creator_event_runs_one_at_a_time_and_privately(self):
        handlers = {ui_characters.generate_photos, ui_characters.regenerate_photo, ui_characters.remove_photo,
                    ui_characters.upload_photos, ui_characters.design_voice, ui_characters.clone_voice,
                    ui_characters.use_preset, ui_characters.refresh_creator, ui_characters.save_character,
                    ui_characters.start_over, ui_characters.edit_character, ui_characters.leave}
        found = [fn for fn in self.demo.fns.values() if fn.fn in handlers]
        self.assertEqual({fn.fn for fn in found}, handlers)
        for fn in found:
            self.assertEqual((fn.concurrency_id, fn.api_visibility), (ui_characters.SERIAL, "private"), fn.fn)
            outputs = [block._id for block in fn.outputs]
            self.assertEqual(len(outputs), len(set(outputs)), fn.fn)  # one update per component and event
        refresh = next(fn for fn in found if fn.fn is ui_characters.refresh_creator)
        ticker = next(self.demo.blocks[block] for block, event in refresh.targets if event == "tick")
        self.assertEqual((ticker.value, ticker.active), (2.0, False))  # it runs only while a job waits or runs

    def test_leaving_the_page_for_any_other_discards_the_draft(self):
        pages = {b.id: b for b in self.blocks if isinstance(b, gr.Tab)
                 and getattr(getattr(b, "parent", None), "elem_id", None) == "pages"}
        leaving = {block for fn in self.demo.fns.values() if fn.fn is ui_characters.leave
                   for block, event in fn.targets if event == "select"}
        others = {page._id for name, page in pages.items() if name != "characters"}
        self.assertEqual(leaving, others)
        # the page script asks first while the creator holds unsaved work (and before the tab closes)
        self.assertIn(f"document.querySelector('{ui_characters._UNSAVED}')", ui_theme.JS)
        self.assertIn("if (leavesCharacter(tab) && !confirm(LEAVE_CHARACTER)) return;", ui_theme.JS)
        self.assertIn("dataset.tabId === 'characters'", ui_theme.JS)  # asked only when leaving this page
        self.assertIn("beforeunload", ui_theme.JS)

    def test_the_creators_button_scripts_parse(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is not installed")
        scripts = [fn.js for fn in self.demo.fns.values() if fn.js and ui_characters.CREATOR in fn.js]
        self.assertEqual(sorted(scripts), sorted([ui_characters.CONFIRM_JS] * 3
                                                 + [ui_characters.SHOW_CREATOR_JS, ui_characters.SHOW_EDITED_JS]))
        scroll = next(fn for fn in self.demo.fns.values() if fn.js == ui_characters.SHOW_EDITED_JS)
        confirm = next(fn for fn in self.demo.fns.values() if fn.js == ui_characters.CONFIRM_JS)
        self.assertEqual(scroll.inputs, confirm.outputs)  # it scrolls only when the confirmation said yes
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "creator.js"
            path.write_text("\n".join(f"const f{i} = {js};" for i, js in enumerate(
                (ui_characters.CONFIRM_JS, ui_characters.SHOW_CREATOR_JS, ui_characters.SHOW_EDITED_JS,
                 ui_characters.DELETE_JS))), encoding="utf-8")
            result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_page_keeps_a_characters_voice_with_the_character(self):
        text = " ".join(str(value) for block in self.on_page()
                        for value in (getattr(block, "value", ""), getattr(block, "label", ""),
                                      getattr(block, "info", "")) if value)
        for unwanted in ("Voices page", "voice library", 'data-page="voices"', "LoRA", "Train", "training"):
            self.assertNotIn(unwanted, text)
        self.assertIn("It is saved with the character", text)


if __name__ == "__main__":
    unittest.main()
