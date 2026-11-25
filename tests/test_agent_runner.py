# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Starting, pausing, stopping and resuming a run on its thread (app/agent_runner.py), with a scripted provider."""
import threading
from pathlib import Path
from unittest.mock import patch

from app import agent_config, agent_delivery, agent_preflight, agent_runner, agent_store, timeline
from app.agent_runner import RunnerError
from tests.agent_base import AgentCase, picture
from tests.agent_fake import FakeClock, ScriptedProvider, error, text_turn, tool_turn

KEY = "sk-" + "ant-" + "api03-" + "r" * 30
FINISH = ("finish_run", {"outcome": "complete", "summary": "All done.",
                         "checklist": [{"requirement": "A film", "status": "done"}]})
STATUS = ("update_status", {"phase": "concept", "text": "Planning"})


class RunnerCase(AgentCase):
    def setUp(self):
        super().setUp()
        enter = self.context.enter_context
        enter(patch.object(agent_config, "get_key", return_value=(KEY, "test")))
        enter(patch.object(agent_preflight, "blocking_models", return_value=""))
        enter(patch.object(agent_store, "free_gb", return_value=500.0))
        enter(patch.object(agent_runner, "make_clock", return_value=FakeClock()))
        enter(patch.object(agent_runner, "_keep_awake"))
        enter(patch.object(agent_delivery, "gate_failures", return_value=[]))  # tested in tests/test_agent_delivery.py
        self.provider = ScriptedProvider()
        enter(patch.object(agent_runner, "make_provider", side_effect=lambda name: self.provider))
        self.addCleanup(self.end_run)
        self.brief = self.root / "brief.md"
        self.brief.write_text("# Coffee ad\nA 15 s film for Beans & Co. Audience: students.\n", encoding="utf-8")

    def end_run(self):
        if run := agent_runner.active():
            agent_runner.stop(run)
            agent_runner.wait(run, 10)

    def begin(self, *script, extra="", files=None, **settings):
        self.provider.script = list(script)
        run_id = agent_runner.start([self.brief] if files is None else files, extra, settings)
        agent_runner.wait(run_id)
        return run_id

    def record(self, run_id):
        return agent_store.RunStore(run_id).read()


class StartTests(RunnerCase):
    def test_a_run_is_created_from_the_brief_and_works_to_the_end_on_its_own_thread(self):
        run_id = self.begin(tool_turn(STATUS), tool_turn(FINISH))
        record = self.record(run_id)
        self.assertEqual((record["state"], record["title"], record["outcome"]["kind"]), ("completed", "brief", "complete"))
        self.assertEqual([item["name"] for item in record["brief"]], ["brief.md"])
        self.assertEqual((record["settings"]["mode"], record["steps"]), ("prepare", 2))
        store = agent_store.RunStore(run_id)
        first = store.wire_messages()[0]["content"]
        self.assertIn(f"[[client-material {record['nonce']} · file 1 · brief.md]]", first[-1]["text"])
        self.assertIn("Beans & Co.", first[-1]["text"])
        self.assertTrue(any(store.path("brief").iterdir()))
        self.assertIn("PREPARES", store.system_text())
        self.assertTrue(store.tools())
        self.assertFalse(agent_runner.busy())

    def test_the_owners_extra_words_and_a_picture_are_part_of_the_first_message(self):
        logo = picture(self.root / "logo.png", color=(200, 20, 20))
        run_id = self.begin(tool_turn(FINISH), extra="Make it funny.", files=[self.brief, (logo, "logo.png")])
        content = agent_store.RunStore(run_id).wire_messages()[0]["content"]
        text = "\n".join(block.get("text", "") for block in content)
        self.assertIn("Make it funny.", text)
        record = self.record(run_id)
        picture_ref = next(item["ref"] for item in record["brief"] if item["kind"] == "image")
        self.assertIn(picture_ref[:12], text)
        self.assertEqual(timeline.reference_path(picture_ref).is_file(), True)

    def test_it_will_not_start_without_a_key_a_brief_or_room_and_leaves_nothing_behind(self):
        cases = [(patch.object(agent_config, "get_key", return_value=("", "")), self.brief, "", "key"),
                 (patch.object(agent_store, "free_gb", return_value=1.0), self.brief, "", "disk"),
                 (patch.object(agent_preflight, "blocking_models", return_value="Models are missing."), self.brief, "",
                  "Models are missing."),
                 (patch.object(agent_config, "get_key", return_value=(KEY, "t")), None, "", "brief file"),
                 (patch.object(agent_config, "get_key", return_value=(KEY, "t")), None, "x" * 5000, "under")]
        for context, brief, extra, words in cases:
            with self.subTest(words=words), context, self.assertRaises(RunnerError) as caught:
                agent_runner.start([brief] if brief else [], extra, {})
            self.assertIn(words, str(caught.exception))
        stray = self.root / "notes.exe"
        stray.write_bytes(b"MZ")
        with self.assertRaises(RunnerError) as caught:
            agent_runner.start([self.brief, stray], "", {})
        self.assertIn("notes.exe", str(caught.exception))
        self.assertEqual(agent_store.list_runs(), [])

    def test_a_second_run_does_not_start_while_one_works(self):
        gate = threading.Event()
        self.provider.script = [lambda request: (gate.wait(10), tool_turn(FINISH))[1]]
        run_id = agent_runner.start([self.brief], "", {})
        try:
            self.assertEqual(agent_runner.active(), run_id)
            with self.assertRaises(RunnerError):
                agent_runner.start([self.brief], "", {})
            with self.assertRaises(RunnerError):
                agent_runner.resume(run_id)
        finally:
            gate.set()
            agent_runner.wait(run_id)
        self.assertEqual(self.record(run_id)["state"], "completed")

    def test_the_run_keeps_the_computer_awake_only_when_the_owner_wants_it(self):
        self.begin(tool_turn(FINISH))
        self.assertEqual([call.args for call in agent_runner._keep_awake.call_args_list], [(True,), (False,)])
        agent_runner._keep_awake.reset_mock()
        self.begin(tool_turn(FINISH), keep_awake=False)
        agent_runner._keep_awake.assert_not_called()

    def test_a_crash_before_the_first_question_marks_the_run_failed_and_frees_the_runner(self):
        with patch.object(agent_runner, "make_provider", side_effect=RuntimeError("no key store")):
            run_id = agent_runner.start([self.brief], "", {})
            agent_runner.wait(run_id)
        record = self.record(run_id)
        self.assertEqual((record["state"], record["failure"]["kind"]), ("failed", "internal"))
        self.assertFalse(agent_runner.busy())


class ControlTests(RunnerCase):
    def test_a_run_is_paused_and_then_resumed_from_the_same_conversation(self):
        run_id = None

        def pause_now(request):
            agent_runner.pause(run_id)
            return tool_turn(STATUS)

        self.provider.script = [pause_now]
        run_id = agent_runner.start([self.brief], "", {})
        agent_runner.wait(run_id)
        self.assertEqual(self.record(run_id)["state"], "paused")
        self.provider.script = [tool_turn(FINISH)]
        agent_runner.resume(run_id)
        agent_runner.wait(run_id)
        self.assertEqual(self.record(run_id)["state"], "completed")
        self.assertEqual(len(self.provider.snapshots[1]), 3)  # the brief, the first answer, its tool result

    def test_a_stop_ends_the_run_and_a_stopped_run_can_be_resumed(self):
        run_id = None

        def stop_now(request):
            agent_runner.stop(run_id)
            return tool_turn(STATUS)

        self.provider.script = [stop_now]
        run_id = agent_runner.start([self.brief], "", {})
        agent_runner.wait(run_id)
        self.assertEqual(self.record(run_id)["state"], "stopped")
        self.assertFalse(agent_runner.stop(run_id) or agent_runner.pause(run_id))
        self.provider.script = [tool_turn(FINISH)]
        agent_runner.resume(run_id)
        agent_runner.wait(run_id)
        self.assertEqual(self.record(run_id)["state"], "completed")

    def test_a_run_that_failed_for_its_key_is_resumed_when_the_key_is_right_and_forgets_the_failure(self):
        run_id = self.begin(error("auth", "The key was refused."))
        self.assertEqual(self.record(run_id)["failure"]["kind"], "auth")
        self.provider.script = [tool_turn(FINISH)]
        agent_runner.resume(run_id)
        agent_runner.wait(run_id)
        record = self.record(run_id)
        self.assertEqual((record["state"], record["failure"]), ("completed", None))

    def test_new_limits_let_a_run_that_stopped_at_its_budget_go_on(self):
        big = [agent_runner.agent_provider.UsageEntry("claude-opus-5-5", input_tokens=6_000_000)]
        run_id = self.begin(tool_turn(STATUS, usage=big), tool_turn(STATUS), tool_turn(STATUS), tool_turn(STATUS))
        self.assertEqual(self.record(run_id)["state"], "budget_stopped")
        self.provider.script = [tool_turn(FINISH)]
        agent_runner.resume(run_id, budgets={"cost_usd": 100.0})
        agent_runner.wait(run_id)
        record = self.record(run_id)
        self.assertEqual((record["state"], record["settings"]["budgets"]["cost_usd"]), ("completed", 100.0))

    def test_a_run_that_is_working_finished_or_unknown_cannot_be_resumed(self):
        run_id = self.begin(tool_turn(FINISH))
        for bad in (run_id, "nonsense", "20260101-000000-abcdef"):
            with self.subTest(bad=bad), self.assertRaises(RunnerError):
                agent_runner.resume(bad)
        with patch.object(agent_config, "get_key", return_value=("", "")):
            agent_store.RunStore(run_id).update(state="stopped")
            with self.assertRaises(RunnerError):
                agent_runner.resume(run_id)

    def test_a_prepared_job_is_approved_for_production_in_the_same_run(self):
        run_id = self.begin(tool_turn(FINISH))
        self.assertEqual(self.record(run_id)["settings"]["mode"], "prepare")
        self.provider.script = [tool_turn(FINISH)]
        agent_runner.resume(run_id, approve_production=True)
        agent_runner.wait(run_id)
        record = self.record(run_id)
        self.assertEqual((record["state"], record["settings"]["mode"]), ("completed", "produce"))
        sent = self.provider.snapshots[-1]
        self.assertIn("approved production", sent[-1]["content"][0]["text"])
        self.assertEqual(sent[0], self.provider.snapshots[0][0])  # the conversation continues from its first message
        with self.assertRaises(RunnerError):
            agent_runner.resume(run_id, approve_production=True)  # it already produced

    def test_only_a_finished_preparing_run_can_be_approved(self):
        run_id = self.begin(error("auth", "no"))
        with self.assertRaises(RunnerError):
            agent_runner.resume(run_id, approve_production=True)


class LifecycleTests(RunnerCase):
    def test_a_run_the_app_lost_is_found_interrupted_after_a_restart_and_can_be_resumed(self):
        run_id = self.begin(tool_turn(FINISH))
        agent_store.RunStore(run_id).update(state="running")
        self.assertEqual(agent_runner.startup(), [run_id])
        self.assertEqual((self.record(run_id)["state"], self.record(run_id)["interrupt_reason"]),
                         ("interrupted", "app_restart"))
        self.provider.script = [tool_turn(FINISH)]
        agent_runner.resume(run_id)
        agent_runner.wait(run_id)
        self.assertEqual(self.record(run_id)["state"], "completed")

    def test_shutdown_pauses_the_working_run_so_it_can_be_resumed_and_is_nothing_when_idle(self):
        agent_runner.shutdown()
        gate = threading.Event()
        self.provider.script = [lambda request: (gate.wait(10), tool_turn(STATUS))[1]]
        run_id = agent_runner.start([self.brief], "", {})
        threading.Timer(0.3, gate.set).start()
        agent_runner.shutdown(timeout=10)
        self.assertEqual(self.record(run_id)["state"], "paused")
        self.assertFalse(agent_runner.busy())

    def test_the_page_reads_a_run_from_disk(self):
        run_id = self.begin(tool_turn(STATUS), tool_turn(FINISH))
        view = agent_runner.view(run_id)
        self.assertEqual((view["state"], view["working"], view["status"]["phase"]), ("completed", False, "concept"))
        self.assertTrue(any(event["type"] == "tool_done" for event in view["events"]))
        self.assertEqual(agent_runner.view("nonsense"), {})
        self.assertEqual(agent_runner.view(), {})
        self.assertIsInstance(Path(agent_store.RunStore(run_id).dir), Path)
