# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Places and items (app/items.py, app/item_photos.py, app/ui_items.py) and how a Timeline scene uses them
(app/timeline.py, app/ui_timeline.py): the library in temporary folders, the photo gate and the engine replaced, an
empty GPU queue whose state is never saved. Nothing live is read or written."""
import io
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import gradio as gr
from PIL import Image

from app import characters, config, dialogue, engine_runtime, item_photos, items, llm, main, queue_manager, refs
from app import style_library
from app import styles, timeline, ui_items, ui_theme, ui_timeline, voices
from app import ui_timeline_view as view
from app.ui_timeline import Draft
from tests.test_characters import picture
from tests.test_ui_timeline import _Isolated

EMPTY = {"humans": 0, "youngest_human_age": 0}
STUDIO = ("A modern TV news studio, a curved glossy blue anchor desk, a big screen with a world map, blue and white "
          "studio lights")
STATE = ["draft", "watch", *ui_items.DISPLAY, "shown"]


def png(size=(1216, 832), color=(40, 60, 120)) -> bytes:
    data = io.BytesIO()
    Image.new("RGB", size, color).save(data, format="PNG")
    return data.getvalue()


class _Library(unittest.TestCase):
    """The item and character libraries, the drafts and the queue in a temporary folder; the vision gate answers
    EMPTY (no people) unless a test says otherwise."""

    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        enter = self.context.enter_context
        self.root = Path(enter(tempfile.TemporaryDirectory())).resolve()
        enter(patch.object(items, "DIR", self.root / "items"))
        enter(patch.object(items, "DRAFTS_DIR", self.root / "item_drafts"))
        enter(patch.object(characters, "DIR", self.root / "characters"))
        enter(patch.object(items, "gate_models", return_value=None))
        enter(patch.object(queue_manager, "_JOBS", []))
        enter(patch.object(queue_manager, "_STATE", self.root / "queue.json"))
        enter(patch.object(queue_manager, "_STATE_INFO", {"next_id": 1, "paused": False}))
        enter(patch.object(queue_manager, "_save"))
        self.gate = enter(patch.object(llm, "check_item_photo", return_value=dict(EMPTY)))

    def upload(self, draft, name="studio.png", consent=True, **options):
        return items.add_upload(draft, picture(self.root / name, **options), consent)

    def saved_item(self, name="TV studio", kind=items.PLACE, look=STUDIO, photos=2, style=styles.ANIMATION_3D) -> dict:
        draft = items.new_draft()
        items.update_draft(draft, name=name, kind=kind, look=look, style=style)
        for index in range(photos):
            self.assertEqual(self.upload(draft, f"{name}{index}.png", color=(60 + index, 70 + len(name), 140)),
                             (True, "Photo added."))  # every item's photos differ from the others'
        profile, message = items.save_draft(draft)
        self.assertIsNotNone(profile, message)
        return profile


class ItemLibraryTests(_Library):
    def test_an_item_is_saved_with_its_photos_and_opens_again_for_editing(self):
        item = self.saved_item()
        folder = self.root / "items" / "tv_studio"
        record = json.loads((folder / "item.json").read_text(encoding="utf-8"))
        self.assertEqual((record["format"], record["name"], record["kind"], record["style"], record["look"]),
                         (1, "TV studio", items.PLACE, styles.ANIMATION_3D, STUDIO))
        self.assertEqual([entry["file"] for entry in record["photos"]], ["01.png", "02.png"])
        self.assertEqual([entry["origin"] for entry in record["photos"]], ["uploaded", "uploaded"])
        self.assertEqual(item["photos"], [str(folder / "photos" / "01.png"), str(folder / "photos" / "02.png")])
        self.assertEqual(list((self.root / "item_drafts").iterdir()), [])  # the draft is gone with Save
        self.assertEqual([found["slug"] for found in items.all_items()], ["tv_studio"])
        self.assertEqual((items.state(item), items.usable("tv_studio")["slug"]), ("Ready", "tv_studio"))
        self.assertEqual(items.main_photo(item), folder / "photos" / "01.png")
        self.assertEqual(items.identity(item), {"item": "tv_studio", "name": "TV studio",
                                                "photo": characters._digest(folder / "photos" / "01.png")})

        draft = items.load_draft("tv_studio")
        opened = items.draft(draft)
        self.assertEqual((opened["slug"], opened["origins"], opened["kind"]), ("tv_studio", ["kept", "kept"],
                                                                                items.PLACE))
        items.remove_photo(draft, 1)
        profile, message = items.save_draft(draft, name="News studio")
        self.assertEqual((profile["slug"], profile["name"], len(profile["photos"])), ("tv_studio", "News studio", 1))
        self.assertEqual(json.loads((folder / "item.json").read_text(encoding="utf-8"))["created"], record["created"])
        self.assertIn("Place ‘News studio’ saved with 1 photo.", message)

    def test_a_photo_needs_consent_and_one_of_someone_under_18_is_refused(self):
        draft = items.new_draft()
        self.assertEqual(self.upload(draft, consent=False)[0], False)
        self.gate.return_value = {"humans": 1, "youngest_human_age": 12}
        ok, message = self.upload(draft, "kid.png")
        self.assertFalse(ok)
        self.assertTrue(message.startswith("⛔ The photo appears to show a minor (estimated age 12)"))
        # adults in a place are no reason to refuse it: an item has no person rules
        self.gate.return_value = {"humans": 3, "youngest_human_age": 30}
        self.assertEqual(self.upload(draft, "crowd.png"), (True, "Photo added."))
        self.gate.reset_mock()
        self.gate.return_value = None  # an answer that cannot be read is asked again, then refused
        self.assertEqual(self.upload(draft, "unclear.png"), (False, "The photo could not be checked. Try again."))
        self.assertEqual(self.gate.call_count, 2)
        self.assertEqual(len(items.draft(draft)["photos"]), 1)
        self.assertEqual(self.upload(draft, "tiny.png", size=(100, 100))[1],
                         "The photo is too small (at least 384 pixels on each side).")

    def test_only_text_that_combines_minors_with_sexual_content_is_refused(self):
        self.assertEqual(items.check_look("A playground for children, swings and a red slide", items.PLACE), "")
        self.assertEqual(items.check_look("A bedroom of a teenager with nude posters", items.PLACE), llm.BLOCK_MSG)
        self.assertIn("Describe the object", items.check_look("  ", items.OBJECT))
        self.assertIn("at most 600", items.check_look("x" * 601))
        self.assertEqual(items.check_name("A schoolgirl erotic room"), llm.BLOCK_MSG)

    def test_names_stay_apart_from_other_items_and_characters(self):
        self.saved_item()
        self.assertIn("already exists", items.check_name("tv STUDIO"))
        self.assertEqual(items.check_name("TV studio", slug="tv_studio"), "")  # the item being edited
        with patch.object(characters, "all_characters", return_value=[{"name": "Chain Dog", "slug": "chain_dog"}]):
            self.assertIn("A character is named", items.check_name("chain dog"))
        with patch.object(voices, "all_voices", return_value=[]):
            self.assertIn("A place or item is named", characters.check_name("TV Studio"))
            self.assertEqual(characters.check_name("Penny"), "")
        self.assertEqual(items.check_name("  "), "Enter a name for the place or item.")

    def test_a_deleted_item_goes_to_the_recycle_bin(self):
        self.saved_item()
        binned = []

        def to_bin(path):
            binned.append(Path(path).name)
            shutil.move(str(path), str(self.root / "bin"))
            return True

        with patch.object(items.trash, "move", side_effect=to_bin):
            message = items.delete("tv_studio")
        self.assertIn("moved to the Recycle Bin", message)
        self.assertEqual((binned, items.get("tv_studio"), items.all_items()), (["tv_studio"], None, []))
        self.assertEqual(items.delete("tv_studio"), "Place or item not found.")

    def test_startup_settles_a_save_cut_short_and_clears_the_drafts(self):
        self.saved_item()
        folder = self.root / "items" / "tv_studio"
        backup = self.root / "items" / ".replaced_tv_studio_0123abcd"
        folder.replace(backup)  # a crash between the two renames of a save
        items.new_draft()
        items.startup()
        self.assertTrue(folder.is_dir())
        self.assertFalse(backup.exists())
        self.assertEqual(list((self.root / "item_drafts").iterdir()), [])
        self.assertEqual(items.get("tv_studio")["name"], "TV studio")

    def test_a_draft_job_does_not_run_after_a_restart(self):
        state = self.root / "queue.json"
        state.write_text(json.dumps({"jobs": [
            {"id": 1, "kind": items.PHOTO_KIND, "label": "Photos", "params": {"draft": "a" * 32, "count": 3},
             "status": queue_manager.PENDING}]}), encoding="utf-8")
        queue_manager._load()
        job = queue_manager.snapshot()[0]
        self.assertEqual(job.status, queue_manager.STOPPED)
        self.assertIn("Start again on the Items page", job.error)
        self.assertEqual(job.units(), 3.0)  # the estimate counts the photos, as for a character's

    def test_names_in_a_picture_text(self):
        studio = {"slug": "tv_studio", "name": "TV studio"}
        desk = {"slug": "desk", "name": "desk"}
        anchor_desk = {"slug": "studio_desk", "name": "TV studio desk"}
        found = items.mentions('Chain Dog reads the news in the tv-studio. He says "the desk is new"', [studio, desk])
        self.assertEqual(found, [studio])  # a quoted line names nothing
        self.assertEqual(items.mentions("A desktop computer", [desk]), [])  # whole words only
        self.assertEqual(items.mentions("Sitting at the TV studio desk", [studio, anchor_desk]), [anchor_desk])
        self.assertEqual(items.mark("In the TV studio, at the desk, then the TV studio again",
                                    [(studio, 3), (desk, 4)]),
                         "In the TV studio [3], at the desk [4], then the TV studio again")
        self.assertEqual(items.mark('He says "TV studio" in the TV studio', [(studio, 2)]),
                         'He says "TV studio" in the TV studio [2]')


class ItemPhotoTests(_Library):
    """The photo job (app/item_photos.py) with the engine replaced: Z-Image draws the first photo, Qwen-Image-Edit
    every other view of it, and every photo passes the gate."""

    def setUp(self):
        super().setUp()
        enter = self.context.enter_context
        self.render = enter(patch.object(engine_runtime, "render", side_effect=lambda *a, **k: png()))

        def compose(client, images, prompt, width, height, target, seed=None, *rest, **options):
            Path(target).write_bytes(png((width, height), color=(50, 70, seed % 200 if seed else 90)))
            return True, str(target)

        self.compose = enter(patch.object(refs, "compose", side_effect=compose))
        enter(patch.object(item_photos.model_manager, "ensure", return_value=None))
        enter(patch.object(engine_runtime, "engine", return_value=object()))

    def draft(self, kind=items.PLACE, look=STUDIO, style=styles.ANIMATION_3D) -> str:
        draft = items.new_draft()
        items.update_draft(draft, name="TV studio", kind=kind, look=look, style=style)
        return draft

    def test_a_place_is_drawn_empty_and_its_views_keep_it(self):
        draft = self.draft()
        job = items.enqueue_photos(draft, 3)
        self.assertEqual((job.kind, job.params["count"], job.params["kind"], job.params["style"]),
                         (items.PHOTO_KIND, 3, items.PLACE, styles.ANIMATION_3D))
        ok, files, notes, error = items.run_photo_job(job.params, lambda *a: None)
        self.assertTrue(ok, error)
        self.assertEqual(notes[0], "3 reference photos ready. Review them before saving.")
        opened = items.draft(draft)
        self.assertEqual(opened["origins"], ["generated"] * 3)
        prompt, width, height = self.render.call_args.args[:3]
        self.assertEqual((width, height), item_photos.SIZES[items.PLACE])  # landscape, as the scenes it goes into
        self.assertIn("no people and no animals", prompt)
        self.assertIn("stylized 3D CGI render", prompt)
        self.assertNotIn("character", prompt)  # the 3D style's "expressive animated character" is left out
        self.assertEqual(self.render.call_args.kwargs.get("style"), styles.ANIMATION_3D)
        views = [call.args[2] for call in self.compose.call_args_list]
        self.assertTrue(all(text.startswith("The same place as in image 1") for text in views))
        self.assertIn("a reverse angle of the same place", views[0])
        self.assertTrue(all(call.args[1] == [opened["photos"][0]] for call in self.compose.call_args_list))
        self.assertEqual(self.gate.call_count, 3)  # every photo passed the gate

    def test_an_object_is_drawn_alone_and_a_photo_can_be_made_again(self):
        draft = self.draft(items.OBJECT, "A vintage red bicycle with a wicker basket", styles.REALISTIC)
        ok, *_ = items.run_photo_job(items.enqueue_photos(draft, 2).params, lambda *a: None)
        self.assertTrue(ok)
        prompt, width, height = self.render.call_args.args[:3]
        self.assertEqual((width, height), (1024, 1024))
        self.assertIn("One single object alone", prompt)
        self.assertIn("ultra realistic photo", prompt)
        self.assertNotIn("style", self.render.call_args.kwargs)  # Realistic draws as it always did
        before = items.draft(draft)["photos"]
        job = items.enqueue_regenerate(draft, 1)
        ok, _, notes, error = items.run_photo_job(job.params, lambda *a: None)
        self.assertTrue(ok, error)
        self.assertEqual(notes, ["Photo 2 generated again."])
        after = items.draft(draft)["photos"]
        self.assertEqual((after[0], len(after)), (before[0], 2))
        self.assertNotEqual(after[1], before[1])
        self.assertTrue(self.compose.call_args.args[2].startswith("The same object as in image 1"))

    def test_a_photo_of_a_minor_stops_the_job_and_the_draft_keeps_what_it_had(self):
        draft = self.draft()
        self.assertEqual(self.upload(draft, "mine.png"), (True, "Photo added."))
        before = items.draft(draft)["photos"]
        self.gate.return_value = {"humans": 1, "youngest_human_age": 9}
        ok, _, _, error = items.run_photo_job(items.enqueue_photos(draft, 3, keep_own=True).params, lambda *a: None)
        self.assertFalse(ok)
        self.assertTrue(error.startswith("⛔ The photo appears to show a minor"))
        self.assertEqual(items.draft(draft)["photos"], before)
        self.assertEqual(sorted(path.name for path in (self.root / "item_drafts" / draft).glob("p_*.png")),
                         sorted(Path(path).name for path in before))

    def test_what_a_photo_job_needs(self):
        draft = self.draft(look="")
        with self.assertRaisesRegex(ValueError, "Describe the place"):
            items.enqueue_photos(draft, 3)
        with self.assertRaisesRegex(ValueError, "Choose 1 to 8 photos"):
            items.enqueue_photos(self.draft(), 9)
        with self.assertRaisesRegex(ValueError, "Upload a photo first"):
            items.enqueue_photos(self.draft(), 3, keep_own=True)
        own = self.draft()
        self.upload(own)
        with self.assertRaisesRegex(ValueError, "Only generated photos"):
            items.enqueue_regenerate(own, 0)

    def test_the_style_words_of_a_place_or_object_leave_out_characters(self):
        self.assertNotIn("face", item_photos.style_words(styles.CARTOON))
        self.assertIn("bold clean black outlines", item_photos.style_words(styles.CARTOON))
        self.assertEqual(item_photos.style_words(styles.REALISTIC), styles.get(styles.REALISTIC).image)
        custom = styles.custom_id(styles.OTHER, "watercolor storybook, soft pastel colors")
        self.assertIn("watercolor storybook", item_photos.style_words(custom))
        self.assertIn("(2D anime, hand-drawn cel animation)", item_photos.view_prompt("a wide shot", items.PLACE,
                                                                                     styles.ANIME))


class ItemsPageTests(_Library):
    """The Items page: its place in the app, its layout and its handlers on a real draft."""

    def setUp(self):
        super().setUp()
        self.context.enter_context(patch.object(ui_theme, "get_upload_folder", return_value=str(self.root / "cache")))

    def named(self, result, extra=()) -> dict:
        names = [*STATE, *extra]
        self.assertEqual(len(result), len(names))
        return dict(zip(names, result))

    def start(self) -> dict:
        return {"draft": "", "watch": {}, "shown": ui_items._view("", {}, "", "")}

    def test_the_page_sits_after_characters_and_its_folders_are_never_served(self):
        self.assertEqual(main.WORKSPACE[2:5], (("Editor", "timeline"), ("Characters", "characters"),
                                               ("Locations & Props", "items")))  # after Dashboard and Series
        self.assertIn("items", ui_theme.ICONS)
        blocked = main.local_launch_options({"host": "127.0.0.1"})["blocked_paths"]
        self.assertIn(str((self.root / "items").resolve()), blocked)
        self.assertIn(str((self.root / "item_drafts").resolve()), blocked)

    def test_the_creator_reads_top_to_bottom(self):
        with patch.object(style_library, "all_custom", return_value=[]):
            with gr.Blocks() as demo:
                with gr.Tab("Items", id="items", elem_id="items-page") as tab:
                    ui_items.build_items_tab(tab)
        blocks = list(demo.blocks.values())
        creator = next(block for block in blocks if getattr(block, "elem_id", None) == ui_items.CREATOR)

        def inside(block):
            while block is not None:
                if block is creator:
                    return True
                block = getattr(block, "parent", None)
            return False

        labels = [getattr(block, "label", None) for block in blocks if inside(block) and getattr(block, "label", None)]
        order = ["Kind", "Style", "Description", "Number of photos", items.CONSENT, "Name"]
        self.assertEqual([label for label in labels if label in order], order)
        kind = next(block for block in blocks if getattr(block, "elem_id", None) == "item-kind")
        self.assertEqual(kind.choices, [("Place", items.PLACE), ("Object", items.OBJECT)])
        handlers = {found.fn for found in demo.fns.values()}
        for fn in (ui_items.generate_photos, ui_items.upload_photos, ui_items.save_item, ui_items.edit_item,
                   ui_items.start_over, ui_items.delete_item, ui_items.refresh_creator):
            self.assertIn(fn, handlers)

    def test_generate_review_save_edit_and_delete(self):
        out = self.named(ui_items.generate_photos("", {}, self.start()["shown"], "TV studio", items.PLACE, "", 3,
                                                  True, None, styles.ANIMATION_3D))
        self.assertIn("Describe the place", out["photo_status"])
        self.assertEqual(queue_manager._JOBS, [])
        out = self.named(ui_items.generate_photos(out["draft"], out["watch"], out["shown"], "TV studio", items.PLACE,
                                                  STUDIO, 3, True, None, styles.ANIMATION_3D))
        job = queue_manager._JOBS[-1]
        self.assertEqual((job.kind, job.params["count"], job.params["style"]),
                         (items.PHOTO_KIND, 3, styles.ANIMATION_3D))
        self.assertIn("Waiting in the GPU queue", out["photo_status"])
        self.assertIs(out["ticker"].active, True)
        self.assertIn("Save is ready once", out["checklist"])

        made = [(picture(items.new_photo_path(out["draft"]), color=(80 + n, 90, 160)), "generated") for n in range(3)]
        items.replace_photos(out["draft"], made)
        job.status, job.notes = queue_manager.DONE, ["3 reference photos ready. Review them before saving."]
        out = self.named((out["draft"], *ui_items.refresh_creator(out["draft"], out["watch"], out["shown"],
                                                                  "TV studio", STUDIO, None)))
        self.assertIn("3 reference photos ready", out["photo_status"])
        self.assertEqual(len(out["photos"]["value"]), 3)
        self.assertEqual(out["photos"]["value"][0][1], "1 · main")
        self.assertIs(out["save"]["interactive"], True)
        self.assertIs(out["ticker"].active, False)

        saved = ui_items.save_item(out["draft"], out["watch"], out["shown"], "TV studio", items.PLACE, STUDIO, None,
                                   styles.ANIMATION_3D)
        out = self.named(saved, [*ui_items.FORM_NAMES, "cards", "listing", "pick", "assets"])
        self.assertEqual(out["draft"], "")
        self.assertIn("Place ‘TV studio’ saved with 3 photos.", out["checklist"])
        self.assertEqual(out["pick"]["value"], "tv_studio")
        self.assertIn("TV studio", out["listing"])

        edited = ui_items.edit_item("yes", "tv_studio", "", {}, out["shown"])
        out = self.named(edited, [*ui_items.FORM_NAMES, "info"])
        self.assertTrue(out["draft"])
        self.assertEqual((out["name"], out["kind"]), ("TV studio", items.PLACE))
        self.assertIn("is open in the creator", out["info"])
        out = self.named(ui_items.start_over("yes", out["draft"], out["watch"], out["shown"]), ui_items.FORM_NAMES)
        self.assertEqual(out["draft"], "")
        self.assertEqual(list((self.root / "item_drafts").iterdir()), [])

        self.assertIn("Nothing deleted", ui_items.delete_item("tv_studio", "nope")[0])
        with patch.object(items.trash, "move", side_effect=lambda path: shutil.rmtree(path) or True):
            message = ui_items.delete_item("tv_studio", "delete")[0]
        self.assertIn("moved to the Recycle Bin", message)
        self.assertEqual(items.all_items(), [])


class TimelineItemTests(_Library):
    """A scene with a place or item: picked in the inspector (one of its reference images) or named in its picture
    text (used as one when the scene is made, and part of its clip keys)."""

    def setUp(self):
        super().setUp()
        enter = self.context.enter_context
        enter(patch.object(timeline, "DIR", self.root / "projects"))
        enter(patch.object(config, "OUTPUTS_DIR", self.root / "outputs"))
        enter(patch.object(characters, "all_characters", return_value=[]))
        enter(patch.object(voices, "all_voices", return_value=[]))
        for memo in (timeline._MEMO, timeline._ITEM_REFERENCES, timeline._CLIP_INDEX):
            memo.clear()
            self.addCleanup(memo.clear)
        ui_timeline._forget()
        self.addCleanup(ui_timeline._forget)

    def project(self, *prompts, references=()) -> timeline.Project:
        scenes = tuple(timeline.Scene(prompt, 4, references=tuple(references)) for prompt in prompts)
        return timeline.Project("p" * 32, 1, "News", scenes, audio=False, style=styles.ANIMATION_3D)

    def key(self, project, index=0) -> tuple[str, str]:
        timeline._MEMO.clear()
        scene = project.scenes[index]
        return (timeline._clip_key(project, scene.clip_settings(), [], project.video_style, "full", index, None),
                timeline._picture_key(project, scene, [], project.video_style, "full", index, None))

    def test_a_picked_item_is_one_of_the_scenes_reference_images(self):
        item = self.saved_item()
        name = timeline.item_reference(item)
        self.assertEqual(name, timeline.store_reference(items.main_photo(item)))  # what a pick stores
        draft = Draft(scenes=(timeline.Scene("Chain Dog waves at the camera"),), form="k")
        value, message, picker = ui_timeline._pick_items(draft, ui_timeline._envelope([], "k"), ["tv_studio"])
        self.assertEqual(json.loads(value), {"key": "k", "rows": [name]})
        self.assertEqual(picker["value"], ["tv_studio"])
        self.assertIn("Added to the scene's reference images: TV studio [1]", message)
        self.assertIsNotNone(timeline.reference_path(name))
        value, message, picker = ui_timeline._pick_items(draft, ui_timeline._envelope(["b" * 64, name], "k"), [])
        self.assertEqual((json.loads(value)["rows"], picker["value"]), (["b" * 64], []))  # only the item's image leaves
        self.assertIn("Taken out", message)
        full = ui_timeline._envelope([str(n) * 64 for n in range(1, 7)], "k")
        value, message, picker = ui_timeline._pick_items(draft, full, ["tv_studio"])
        self.assertEqual(len(json.loads(value)["rows"]), 6)
        self.assertIn("at most 6 reference images", message)
        self.assertEqual(picker["value"], [])  # the refused pick leaves the picker

    def test_the_inspector_names_the_item_of_each_image_and_the_items_the_text_names(self):
        item = self.saved_item()
        other = self.saved_item("Red bicycle", items.OBJECT, "A vintage red bicycle", photos=1)
        name = timeline.item_reference(item)
        scene = timeline.Scene("Chain Dog rides the red bicycle into the TV studio", references=(name,))
        choices, values, note, labels = ui_timeline._scene_items(scene, [name])
        self.assertEqual(choices, (("TV studio", "tv_studio"), ("Red bicycle", "red_bicycle")))
        self.assertEqual((values, labels), (("tv_studio",), {name: "TV studio"}))
        self.assertIn("Named in the picture text: <b>Red bicycle</b>. Its main photo is used as a reference image",
                      note)
        self.assertEqual(timeline.named_items(scene), [other])  # the picked studio is not named again
        self.assertIn("<figcaption>[1] TV studio</figcaption>", view.references_html([(name, "", "TV studio")], "k"))
        self.assertIn("<figcaption>[1]</figcaption>", view.references_html([(name, "")], "k"))

    def test_a_named_item_is_part_of_the_scenes_keys_and_a_scene_without_one_keeps_its_keys(self):
        plain = self.project("Chain Dog waves at the camera")
        named = self.project("Chain Dog reads the news in the TV studio")
        before = self.key(plain), self.key(named)
        item = self.saved_item()
        self.assertEqual(self.key(plain), before[0])
        self.assertNotEqual(self.key(named)[0], before[1][0])
        self.assertNotEqual(self.key(named)[1], before[1][1])
        made = self.key(named)
        draft = items.load_draft("tv_studio")  # a new main photo: the clip is made again
        items.remove_photo(draft, 0)
        items.save_draft(draft)
        self.assertNotEqual(self.key(named), made)
        # picked as a reference image, the item is in the scene's settings instead
        picked = self.project("Chain Dog reads the news in the TV studio",
                              references=[timeline.item_reference(items.get("tv_studio"))])
        timeline._MEMO.clear()
        self.assertEqual(timeline.named_items(picked.scenes[0]), [])
        self.assertEqual(timeline._item_identities(picked.scenes[0].clip_settings()), [])
        self.assertEqual(timeline._item_identities(named.scenes[0].clip_settings()),
                         [items.identity(items.get(item["slug"]))])

    def test_a_named_item_reaches_the_scene_as_a_reference_image(self):
        item = self.saved_item()
        prompt, paths = timeline._with_named_items("Chain Dog in the TV studio", [], [item])
        stored = timeline.reference_path(timeline.item_reference(item))
        self.assertEqual((prompt, paths), ("Chain Dog in the TV studio", [str(stored)]))
        prompt, paths = timeline._with_named_items("Chain Dog wears the scarf [1] in the TV studio", ["scarf.png"],
                                                   [item])
        self.assertEqual(prompt, "Chain Dog wears the scarf [1] in the TV studio [2]")  # numbered: it gets one too
        self.assertEqual(paths, ["scarf.png", str(stored)])
        # the number it gets must not turn a screen's text into speech (dialogue.shown): the picture keeps it
        prompt, _ = timeline._with_named_items('A news anchor sits at the desk [1]. The big screen in the TV studio '
                                               'displays "CRYPTO NEWS".', ["desk.png"], [item])
        self.assertIn("TV studio [2] displays", prompt)
        picture = dialogue.parse(prompt)[0]
        self.assertIn('"CRYPTO NEWS"', picture)
        self.assertNotIn("talking", picture)

        project = self.project("Chain Dog reads the news in the TV studio")
        captured = {}

        def generate(prompt, seconds, aspect, **options):
            captured.update(options, prompt=prompt)
            return None

        with patch.object(timeline.video_gen, "generate", side_effect=generate), \
                patch.object(timeline.engine_runtime, "read_ahead_start_frame") as read_ahead:
            timeline._generate(project, project.scenes[0], [], 7, project.video_style, self.root, lambda *a: None,
                               number=0)
        self.assertEqual(captured["ref_images"], [str(stored)])
        self.assertEqual(captured["prompt"], "Chain Dog reads the news in the TV studio")
        read_ahead.assert_called_once()

    def test_without_items_a_scene_is_made_as_before(self):
        project = self.project("Chain Dog in the TV studio")
        self.assertEqual(timeline.named_items(project.scenes[0]), [])
        captured = {}
        with patch.object(timeline.video_gen, "generate", side_effect=lambda *a, **k: captured.update(k)):
            timeline._generate(project, project.scenes[0], [], 7, project.video_style, self.root, lambda *a: None,
                               number=0)
        self.assertEqual(captured["ref_images"], [])


class TimelinePickerTests(_Isolated):
    """The picker sits right under "Appearing in this scene" and writes the reference editor's field."""

    def test_the_picker_follows_appearing_in_this_scene_and_writes_the_references(self):
        demo, _ = self.build()
        blocks = list(demo.blocks.values())
        picker = next(block for block in blocks if getattr(block, "elem_id", None) == "timeline-items")
        cast = next(block for block in blocks if getattr(block, "label", None) == "Appearing in this scene")
        self.assertEqual(picker.label, "Places and items in this scene")
        self.assertTrue(picker.multiselect)
        siblings = picker.parent.children
        self.assertEqual(siblings.index(picker), siblings.index(cast) + 1)
        handler = next(found for found in demo.fns.values() if found.fn is ui_timeline._pick_items)
        field = next(block for block in blocks if getattr(block, "elem_id", None) == view.REFERENCES_ID)
        self.assertIn(field._id, [block._id for block in handler.inputs])
        self.assertIn(field._id, [block._id for block in handler.outputs])
        self.assertIn("items", ui_timeline.OUTPUTS)


if __name__ == "__main__":
    unittest.main()
