# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The folder of an agent run: its summary, its events, its append-only conversation (with large images and
documents kept as blobs and put back byte for byte), the sweep after a restart, and the guard against the API key."""
import base64
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import agent_config, agent_store

KEY = "sk-" + "ant-" + "api03-" + "Kk1Ll2Mm3Nn4Oo5Pp6Qq7Rr8Ss9"
IMAGE = bytes(range(256)) * 40  # 10 KB of bytes that are not text


def image_block(data=IMAGE):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(data).decode("ascii")}}


class _Isolated(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.root = Path(self.context.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.context.enter_context(patch.object(agent_store, "RUNS_DIR", self.root / "agent_runs"))
        self.context.enter_context(patch.object(agent_config, "CREDENTIALS_FILE", self.root / "credentials.json"))
        env = {name: value for name, value in os.environ.items() if not name.startswith("ANTHROPIC_")}
        self.context.enter_context(patch.dict(os.environ, env, clear=True))
        agent_config._CACHE.update(stamp=None, key="")

    def made(self, **summary) -> agent_store.RunStore:
        store = agent_store.RunStore(agent_store.new_run_id())
        store.create(summary={"config": {"model": "claude-opus-5-5"}, "usage": {"requests": 0}, **summary},
                     system="System text.", tools=[{"name": "t", "input_schema": {"type": "object"}}])
        return store


class RunFolderTests(_Isolated):
    def test_a_new_run_has_its_folders_and_frozen_instructions(self):
        store = self.made()
        for name in agent_store.FOLDERS:
            self.assertTrue(store.path(name).is_dir(), name)
        self.assertEqual(store.system_text(), "System text.")
        self.assertEqual(store.tools(), [{"name": "t", "input_schema": {"type": "object"}}])
        record = store.read()
        self.assertEqual((record["id"], record["state"], record["schema"]), (store.id, "created", 1))
        with self.assertRaisesRegex(ValueError, "already exists"):
            store.create(summary={}, system="", tools=[])

    def test_update_merges_dicts_one_level_deep_and_replaces_the_rest(self):
        store = self.made()
        record = store.update(state="running", usage={"requests": 3, "output": 40}, error=None)
        self.assertEqual((record["state"], record["usage"], record["config"]["model"]),
                         ("running", {"requests": 3, "output": 40}, "claude-opus-5-5"))
        self.assertEqual(store.update(usage={"requests": 4})["usage"], {"requests": 4, "output": 40})
        self.assertGreater(record["updated"], record["created"] - 1)

    def test_run_ids_are_checked_before_any_path_is_made(self):
        for bad in ("", "..", "../x", "20261117-143012-ZZZZZZ", "20261117-143012-3f9a1c/../../x", None, 5):
            with self.subTest(id=bad), self.assertRaises(ValueError):
                agent_store.run_dir(bad)
        made = agent_store.new_run_id()
        self.assertTrue(agent_store.valid_id(made))
        self.assertEqual(agent_store.run_dir(made).parent, agent_store.RUNS_DIR.resolve())

    def test_blob_and_brief_names_are_content_hashes_and_other_names_are_refused(self):
        store = self.made()
        name = store.put_blob(b"abc", "png")
        self.assertEqual(store.read_blob(name), b"abc")
        self.assertEqual(store.put_blob(b"abc", "png"), name)  # the same bytes, the same blob
        brief = store.put_brief(b"hello", "md")
        self.assertEqual(store.path("brief", brief).read_bytes(), b"hello")
        for bad in ("../x.png", "x.png", "a" * 64, "a" * 64 + "/../x.png"):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                store.read_blob(bad)


class EventTests(_Isolated):
    def test_events_are_numbered_and_the_key_is_masked(self):
        store = self.made()
        with patch.dict(os.environ, {agent_config.ENV_KEY: KEY}):
            first = store.log("state", state="running")
            second = store.log("tool_result", name="get_project", text=f"header was Bearer {KEY} and more")
        self.assertEqual((first, second), (1, 2))
        raw = store.path("events.jsonl").read_text(encoding="utf-8")
        self.assertNotIn(KEY, raw)
        self.assertIn("[key removed]", raw)
        self.assertEqual([event["type"] for event in store.events()], ["state", "tool_result"])
        self.assertEqual([event["seq"] for event in store.events(since=1)], [2])
        self.assertEqual(len(store.events(limit=1)), 1)
        # a second reader or a new store object carries on the numbers
        again = agent_store.RunStore(store.id)
        self.assertEqual(again.log("state", state="paused"), 3)

    def test_a_line_cut_short_by_a_crash_is_left_out(self):
        store = self.made()
        store.log("state", state="running")
        with store.path("events.jsonl").open("ab") as stream:
            stream.write(b'{"seq": 2, "type": "tool_res')
        self.assertEqual([event["seq"] for event in store.events()], [1])
        self.assertEqual(store.view()["events"][0]["type"], "state")


class ConversationTests(_Isolated):
    def test_messages_come_back_exactly_with_large_images_kept_as_blobs(self):
        store = self.made()
        thinking = {"type": "thinking", "thinking": "", "signature": "sig-" + "q" * 40}
        user = [{"type": "text", "text": "Brief"}, image_block()]
        result = [{"type": "tool_result", "tool_use_id": "toolu_01",
                   "content": [{"type": "text", "text": "frame"}, image_block(IMAGE[::-1])]}]
        store.append_message("user", user)
        store.append_message("assistant", [thinking, {"type": "tool_use", "id": "toolu_01", "name": "t",
                                                      "input": {"é": "ż"}}], meta={"request_id": "req_1"})
        store.append_message("user", result)

        raw = store.path("transcript.jsonl").read_text(encoding="utf-8")
        self.assertNotIn(base64.b64encode(IMAGE).decode("ascii")[:200], raw)  # the pixels are in blobs
        self.assertEqual(len(list(store.path("blobs").iterdir())), 2)
        self.assertEqual(json.dumps(store.records()[1]["content"][0]), json.dumps(thinking))
        self.assertEqual(store.records()[1]["meta"], {"request_id": "req_1"})

        wire = store.wire_messages()
        self.assertEqual([item["role"] for item in wire], ["user", "assistant", "user"])
        self.assertEqual(json.dumps(wire[0]["content"]), json.dumps(user))  # byte for byte, key order included
        self.assertEqual(json.dumps(wire[2]["content"]), json.dumps(result))
        self.assertEqual(wire[1]["content"][0], thinking)
        self.assertEqual(wire[1]["content"][1]["input"], {"é": "ż"})
        self.assertNotIn("meta", wire[1])

    def test_small_payloads_and_text_stay_in_the_line(self):
        store = self.made()
        small = image_block(b"tiny")
        store.append_message("user", [small, {"type": "text", "text": "x" * 5000}])
        self.assertEqual(list(store.path("blobs").iterdir()), [])
        self.assertEqual(store.wire_messages()[0]["content"][0], small)

    def test_a_message_with_the_key_is_refused_and_nothing_is_written(self):
        store = self.made()
        store.append_message("user", "first")
        with patch.dict(os.environ, {agent_config.ENV_KEY: KEY}):
            with self.assertRaises(agent_store.LeakError):
                store.append_message("assistant", [{"type": "text", "text": f"my key is {KEY}"}])
            with self.assertRaises(agent_store.LeakError):
                store.append_message("assistant", [{"type": "text", "text": "sk-ant-" + "z" * 30}])
        self.assertEqual(len(store.records()), 1)
        self.assertEqual(store.append_message("user", "second"), 2)  # the numbering carries on

    def test_a_cut_last_line_is_ignored_and_the_next_message_goes_on_after_it(self):
        store = self.made()
        store.append_message("user", "one")
        with store.path("transcript.jsonl").open("ab") as stream:
            stream.write(b'{"i": 2, "role": "assistant", "con')
        self.assertEqual([record["content"] for record in store.records()], ["one"])
        self.assertEqual(len(store.wire_messages()), 1)


class SweepTests(_Isolated):
    def test_runs_left_working_by_a_restart_become_interrupted(self):
        running = self.made()
        running.update(state="running")
        waiting = self.made()
        waiting.update(state="waiting", wait_reason="gpu_job")
        done = self.made()
        done.update(state="completed")
        paused = self.made()
        paused.update(state="paused")
        swept = agent_store.startup_sweep()
        self.assertCountEqual(swept, [running.id, waiting.id])
        for store in (running, waiting):
            record = store.read()
            self.assertEqual((record["state"], record["wait_reason"], record["interrupt_reason"]),
                             ("interrupted", "", "app_restart"))
            self.assertEqual(store.events()[-1]["type"], "state")
        self.assertEqual((done.read()["state"], paused.read()["state"]), ("completed", "paused"))
        self.assertEqual(agent_store.startup_sweep(), [])

    def test_list_runs_is_newest_first_and_skips_junk(self):
        old = agent_store.RunStore("20260101-000000-aaaaaa")
        old.create(summary={}, system="", tools=[])
        new = agent_store.RunStore("20261117-120000-bbbbbb")
        new.create(summary={}, system="", tools=[])
        (agent_store.RUNS_DIR / "not-a-run").mkdir()
        broken = agent_store.RunStore("20261117-130000-cccccc")
        broken.path("run.json").parent.mkdir(parents=True)
        broken.path("run.json").write_text("{nope", encoding="utf-8")
        self.assertEqual([run["id"] for run in agent_store.list_runs()], [new.id, old.id])
        shutil_free = agent_store.free_gb()
        self.assertGreater(shutil_free, 0)

    def test_a_finished_run_goes_to_the_recycle_bin_but_a_working_one_stays(self):
        store = self.made()
        store.update(state="running")
        with patch.object(agent_store.trash, "move", return_value=True) as move:
            with self.assertRaisesRegex(ValueError, "still working"):
                agent_store.move_to_trash(store.id)
            move.assert_not_called()
            store.update(state="completed")
            self.assertTrue(agent_store.move_to_trash(store.id))
            move.assert_called_once_with(store.dir)


if __name__ == "__main__":
    unittest.main()
