# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Agent's tools: how they are described, checked and reported (app/agent_tools.py), and each group on the real
functions of Contsy with every store in a temporary folder (tests/agent_base.py)."""
import json
from unittest.mock import patch

from app import agent_config, agent_tools, characters, items, llm, queue_manager, style_library, styles, timeline
from app import universes, voices
from app.agent_host import ToolError
from app.downloader import BLOCKED_PREFIX, DownloadsBlocked
from tests.agent_base import AgentCase, picture

KEY = "sk-" + "ant-" + "api03-" + "x" * 30


class SchemaTests(AgentCase):
    def test_a_value_is_checked_against_its_schema_and_nothing_is_coerced(self):
        schema = agent_tools.obj({"name": agent_tools.text("n", max_length=5, min_length=1),
                                  "count": agent_tools.whole("c", 1, 3), "ratio": agent_tools.number("r", 0, 1),
                                  "on": agent_tools.flag("f"), "kind": agent_tools.choice("k", "a", "b"),
                                  "list": agent_tools.many("l", agent_tools.whole("n", 0, 9), max_items=2),
                                  "code": agent_tools.text("c", pattern=r"[0-9a-f]{4}")}, ("name",))
        agent_tools.validate(schema, {"name": "ok", "count": 2, "ratio": 0.5, "on": True, "kind": "a", "list": [1, 2],
                                      "code": "ab12"})
        bad = [({}, "name"), ({"name": ""}, "name"), ({"name": "toolong"}, "name"), ({"name": 5}, "name"),
               ({"name": "a", "extra": 1}, "extra"), ({"name": "a", "count": 4}, "count"),
               ({"name": "a", "count": 1.5}, "count"), ({"name": "a", "count": True}, "count"),
               ({"name": "a", "count": "2"}, "count"), ({"name": "a", "ratio": float("nan")}, "ratio"),
               ({"name": "a", "ratio": 2}, "ratio"), ({"name": "a", "on": 1}, "on"), ({"name": "a", "kind": "c"}, "kind"),
               ({"name": "a", "list": [1, 2, 3]}, "list"), ({"name": "a", "list": [10]}, "list[0]"),
               ({"name": "a", "list": "x"}, "list"), ({"name": "a", "code": "zz"}, "code"), ("text", "input")]
        for value, where in bad:
            with self.subTest(value=value), self.assertRaises(ToolError) as caught:
                agent_tools.validate(schema, value)
            self.assertEqual(caught.exception.code, "INVALID")
            self.assertIn(where, str(caught.exception))

    def test_every_tool_is_described_the_same_way_every_time(self):
        first = agent_tools.definitions()
        self.assertEqual(json.dumps(first), json.dumps(agent_tools.definitions()))
        names = [one["name"] for one in first]
        self.assertEqual(names, sorted(set(names)))
        for one in first:
            with self.subTest(tool=one["name"]):
                self.assertTrue(one["description"].strip())
                self.assertEqual(one["input_schema"]["type"], "object")
                self.assertIs(one["input_schema"]["additionalProperties"], False)
                self.assertTrue(set(one["input_schema"]["required"]) <= set(one["input_schema"]["properties"]))
        self.assertTrue({"save_project", "create_character", "queue_clip", "wait_for_jobs", "delete_thing",
                         "timeline_job", "edit_scene", "set_preference", "list_gallery"} <= set(names))


class DispatcherTests(AgentCase):
    def test_an_unknown_tool_and_bad_arguments_are_errors_the_model_can_read(self):
        host = self.host()
        self.assertIn("no tool called", self.refused(host, "make_coffee", "INVALID"))
        self.refused(host, "get_character", "INVALID")            # slug is required
        self.refused(host, "get_character", "INVALID", slug=5)
        self.refused(host, "get_character", "INVALID", slug="a", more=1)

    def test_a_tool_of_final_production_is_refused_in_a_run_that_prepares(self):
        host = self.host(mode="prepare")
        self.refused(host, "queue_clip", "MODE", prompt="A cup", seconds=3, aspect="Landscape 16:9")
        self.assertEqual(queue_manager.snapshot(), [])

    def test_what_a_contsy_function_complains_about_becomes_a_code(self):
        host = self.host()
        cases = [(timeline.Busy("This project has a job in the GPU queue."), "BUSY"),
                 (timeline.Deleted(), "NOT_FOUND"),
                 (DownloadsBlocked("x"), "MODELS_MISSING"),
                 (ValueError(f"{BLOCKED_PREFIX}: Wan"), "MODELS_MISSING"),
                 (ValueError(llm.BLOCK_MSG), "CONTENT_RULE"), (ValueError("⛔ Only adult characters"), "CONTENT_RULE"),
                 (ValueError("This project was saved in another tab."), "STALE"),
                 (ValueError("This project already has a job in the queue."), "BUSY"),
                 (ValueError("Choose a valid aspect ratio."), "INVALID"),
                 (RuntimeError("boom"), "INTERNAL")]
        for error, code in cases:
            with self.subTest(error=str(error)[:30]), patch.object(characters, "all_characters", side_effect=error):
                self.refused(host, "list_characters", code)
        crash = [event for event in host.store.events() if event["type"] == "tool_crash"]
        self.assertEqual([event["error"] for event in crash], ["RuntimeError"])

    def test_a_crash_does_not_show_its_details_to_the_model(self):
        with patch.object(characters, "all_characters", side_effect=RuntimeError("C:/secret/path")):
            text = self.refused(self.host(), "list_characters", "INTERNAL")
        self.assertNotIn("secret", text)

    def test_every_call_is_logged_without_the_key_and_a_long_answer_is_cut(self):
        host = self.host()
        self.call(host, "get_character", slug=f"{KEY}")
        event = [e for e in host.store.events() if e["type"] == "tool"][-1]
        self.assertEqual((event["tool"], event["ok"], event["code"]), ("get_character", False, "NOT_FOUND"))
        self.assertNotIn(KEY, json.dumps(host.store.events()))
        long = agent_tools._text("x" * (agent_tools.MAX_RESULT_CHARS + 50))
        self.assertTrue(long.endswith("narrower selection)") and len(long) < agent_tools.MAX_RESULT_CHARS + 100)

    def test_pictures_come_back_as_image_blocks_of_the_tool_result(self):
        result = agent_tools.Result("shown", images=[b"\x89PNG\r\n\x1a\n" + b"0" * 10, b"\xff\xd8\xff" + b"0" * 10])
        block = result.block("toolu_1")
        self.assertEqual(block["tool_use_id"], "toolu_1")
        self.assertNotIn("is_error", block)
        self.assertEqual([part["type"] for part in block["content"]], ["text", "image", "image"])
        self.assertEqual([part["source"]["media_type"] for part in block["content"][1:]], ["image/png", "image/jpeg"])
        self.assertTrue(agent_tools.Result("no", ok=False, code="INVALID").block("t")["is_error"])


class LibraryToolTests(AgentCase):
    def made(self, host, **extra):
        arguments = {"name": "Mia", "gender": "woman", "look": "Short red hair, freckles, green eyes", "photos": 2,
                     "voice": {"preset": "hanna"}, **extra}
        return self.ok(host, "create_character", **arguments)

    def test_a_character_is_made_in_two_steps_and_the_run_may_change_it_later_without_a_backup(self):
        host = self.host()
        made = self.made(host)
        job = queue_manager.snapshot()[0]
        self.assertEqual((job.kind, job.owner, job.label.startswith("Agent: "), made["job_ids"]),
                         ("character_photos", host.owner_tag, True, [job.id]))
        self.assertTrue(host.is_mine("character_draft", made["draft_id"]))
        self.refused(host, "finish_character", "BUSY", draft_id=made["draft_id"])  # its photos are still queued
        self.clear_queue()  # the worker would have made the photos
        for number in range(2):
            ok, message = characters.add_upload(made["draft_id"], picture(self.root / f"m{number}.png",
                                                                          color=(90 + number, 80, 70)), True)
            self.assertTrue(ok, message)
        done = self.ok(host, "finish_character", draft_id=made["draft_id"])
        self.assertEqual((done["character"]["slug"], done["character"]["ready"], done["character"]["photos"]),
                         ("mia", True, 2))
        self.assertTrue(host.is_mine("character", "mia"))
        self.assertEqual(self.ok(host, "get_character", slug="mia")["name"], "Mia")
        self.assertEqual([row["slug"] for row in self.ok(host, "list_characters")["items"]], ["mia"])

    def test_creating_needs_the_write_permission_a_clean_text_and_exactly_one_voice(self):
        self.refused(self.host(permissions={"write": False}), "create_character", "PERMISSION", name="Mia",
                     gender="woman", look="Red hair", voice={"preset": "hanna"})
        host = self.host()
        with patch.object(llm, "blocked", return_value=True):
            self.refused(host, "create_character", "CONTENT_RULE", name="Mia", gender="woman", look="Red hair",
                         voice={"preset": "hanna"})
        for voice in ({}, {"preset": "hanna", "description": "warm"}):
            self.refused(host, "create_character", "INVALID", name="Mia", gender="woman", look="Red hair", voice=voice)
        self.refused(host, "create_character", "INVALID", name="Mia", gender="woman", look="Red hair",
                     voice={"preset": "nobody"})
        self.assertFalse(characters.DRAFTS_DIR.exists() and any(characters.DRAFTS_DIR.iterdir()))  # no draft left
        self.assertEqual(queue_manager.snapshot(), [])

    def test_a_designed_voice_is_queued_with_the_photos(self):
        host = self.host()
        made = self.ok(host, "create_character", name="Leo", gender="man", look="Tall, dark curly hair, grey eyes",
                       voice={"description": "calm, low and warm"})
        self.assertEqual([job.kind for job in queue_manager.snapshot()], ["character_photos", "character_voice"])
        self.assertEqual(len(made["job_ids"]), 2)

    def test_a_character_of_the_owner_is_copied_before_it_is_changed(self):
        saved = self.saved_character()
        host = self.host()
        edit = self.ok(host, "edit_character", slug=saved["slug"], look="Short copper hair, freckles, green eyes")
        self.assertEqual(edit["job_ids"], [])
        self.assertEqual(self.ok(host, "finish_character", draft_id=edit["draft_id"])["character"]["slug"], "mia")
        self.assertEqual(characters.get("mia")["look"], "Short copper hair, freckles, green eyes")
        self.assertTrue((host.store.path("backups", "001-character-mia") / "character.json").is_file())
        self.assertTrue(host.store.read()["changes"][0].startswith("Character mia changed"))

    def test_a_draft_of_another_run_or_one_that_is_gone_is_not_found(self):
        host = self.host()
        self.refused(host, "finish_character", "NOT_FOUND", draft_id="0" * 32)
        self.refused(host, "discard_draft", "NOT_FOUND", kind="character", draft_id="0" * 32)
        made = self.made(host)
        self.refused(self.host(), "finish_character", "NOT_FOUND", draft_id=made["draft_id"])
        self.ok(host, "discard_draft", kind="character", draft_id=made["draft_id"])
        self.refused(host, "finish_character", "NOT_FOUND", draft_id=made["draft_id"])

    def test_places_and_items_go_the_same_way(self):
        host = self.host()
        made = self.ok(host, "create_item", name="TV studio", kind="place", look="A modern TV news studio, blue desk",
                       photos=2)
        self.assertEqual(queue_manager.snapshot()[0].kind, "item_photos")
        self.clear_queue()
        for number in range(2):
            ok, message = items.add_upload(made["draft_id"], picture(self.root / f"s{number}.png",
                                                                     color=(60 + number, 70, 140)), True)
            self.assertTrue(ok, message)
        self.assertEqual(self.ok(host, "finish_item", draft_id=made["draft_id"])["item"]["slug"], "tv_studio")
        self.assertEqual(self.ok(host, "get_item", slug="tv_studio")["kind"], "place")
        edit = self.ok(host, "edit_item", slug="tv_studio", look="A modern TV news studio, red desk")
        self.ok(host, "finish_item", draft_id=edit["draft_id"])
        self.assertEqual(items.get("tv_studio")["look"], "A modern TV news studio, red desk")
        self.assertEqual(host.store.read()["resources"]["backups"], {})  # the run made it itself: no copy
        self.refused(host, "get_item", "NOT_FOUND", slug="nothing")

    def test_a_universe_and_a_style_are_saved_and_one_of_the_owners_is_copied_before_a_change(self):
        self.saved_character("Mia")
        self.saved_character("Leo")
        owner_style, _ = style_library.save("Noir", styles.REALISTIC, "high contrast, deep shadows")
        host = self.host()
        crew = self.ok(host, "save_universe", name="Crew", members=["mia", "leo"])
        self.ok(host, "save_universe", name="The crew", members=["mia", "leo"], slug=crew["slug"])
        self.assertEqual(host.store.read()["resources"]["backups"], {})  # the run made the universe itself
        self.refused(host, "save_universe", "INVALID", name="Solo", members=["mia"])
        self.refused(host, "save_universe", "NOT_FOUND", name="X", members=["mia", "leo"], slug="nope")
        new = self.ok(host, "save_style", name="Noir", base=styles.REALISTIC, look="high contrast, deep shadows, rain",
                      key=owner_style["key"])
        self.assertNotEqual(new["id"], owner_style["id"])
        self.assertEqual(host.store.read()["resources"]["backups"],
                         {"style:" + owner_style["key"]: "001-style-" + owner_style["key"]})
        listed = self.ok(host, "list_styles")
        self.assertTrue(any(row["id"] == styles.REALISTIC for row in listed["styles"]))
        self.assertEqual([row["name"] for row in listed["own_styles"]], ["Noir"])

    def test_the_voices_are_listed(self):
        self.assertEqual([row["slug"] for row in self.ok(self.host(), "list_voices")["items"]], ["hanna"])

    def test_an_attached_picture_becomes_a_photo_only_with_the_owners_permission_to_use_files(self):
        ref = timeline.store_reference(picture(self.root / "logo.png", color=(10, 200, 30)))
        brief = [{"name": "logo.png", "kind": "image", "ref": ref}]
        host = self.host(brief=brief)
        made = self.ok(host, "create_item", name="Logo", kind="object", look="A flat green logo", photos=1)
        self.clear_queue()
        self.refused(host, "add_photo_from_attachment", "PERMISSION", kind="item", draft_id=made["draft_id"],
                     picture_id=ref[:12])
        allowed = self.host(brief=brief, permissions={"write": True, "uploads": True})
        made = self.ok(allowed, "create_item", name="Logo", kind="object", look="A flat green logo", photos=1)
        self.clear_queue()
        self.refused(allowed, "add_photo_from_attachment", "NOT_FOUND", kind="item", draft_id=made["draft_id"],
                     picture_id="f" * 12)
        self.assertEqual(self.ok(allowed, "add_photo_from_attachment", kind="item", draft_id=made["draft_id"],
                                 picture_id=ref[:12]), "Photo added.")
        self.assertEqual(len(items.draft(made["draft_id"])["photos"]), 1)

    def test_deleting_goes_through_the_host(self):
        saved = self.saved_character()
        self.refused(self.host(), "delete_thing", "PERMISSION", kind="character", ref=saved["slug"])
        host = self.host(**self.allow(delete=True))
        self.assertIn("Recycle Bin", self.ok(host, "delete_thing", kind="character", ref=saved["slug"]))
        self.refused(host, "delete_thing", "NOT_FOUND", kind="character", ref=saved["slug"])
