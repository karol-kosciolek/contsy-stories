# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The run loop (app/agent_loop.py) driven by a scripted provider and a clock that does not wait, on the real tools
with every store in a temporary folder."""
import threading
from unittest.mock import patch

from app import agent_config, agent_delivery, agent_loop, queue_manager
from app.agent_provider import Aborted, UsageEntry
from tests.agent_base import AgentCase
from tests.agent_fake import MODEL, ScriptedProvider, error, refusal_turn, text_turn, tool_turn

FINISH = ("finish_run", {"outcome": "complete", "summary": "All done.",
                         "checklist": [{"requirement": "A 15 s film", "status": "done"}]})
STATUS = ("update_status", {"phase": "concept", "text": "Planning the scenes"})


class LoopCase(AgentCase):
    def setUp(self):
        super().setUp()
        # the checks of the package have their own tests (tests/test_agent_delivery.py)
        self.context.enter_context(patch.object(agent_delivery, "gate_failures", return_value=[]))
        self.context.enter_context(patch.object(agent_delivery, "finalize"))

    def start(self, *script, brief="Make a 15 s coffee ad.", **settings):
        host = self.host(**settings)
        host.store.append_message("user", [{"type": "text", "text": brief}])
        self.provider = ScriptedProvider(*script)
        self.stop, self.pause = threading.Event(), threading.Event()
        self.loop = agent_loop.Loop(host.store, host, self.provider, clock=self.clock, stop=self.stop, pause=self.pause)
        self.host_ = host
        return host

    def run_loop(self, *script, **settings) -> str:
        self.start(*script, **settings)
        return self.loop.run()

    def record(self):
        return self.host_.store.read()

    def states(self):
        return [event["state"] for event in self.host_.store.events(limit=0) if event["type"] == "state"]


class FinishingTests(LoopCase):
    def test_a_run_that_calls_the_tools_and_finishes_is_completed_with_its_cost_and_steps(self):
        state = self.run_loop(tool_turn(STATUS), tool_turn(FINISH))
        record = self.record()
        self.assertEqual(state, "completed")
        self.assertEqual((record["steps"], record["usage"]["turns"], record["outcome"]["kind"]), (2, 2, "complete"))
        self.assertAlmostEqual(record["usage"]["cost_usd"], 2 * (2000 * 4 + 300 * 20) / 1e6)
        self.assertEqual(record["status"], {"phase": "concept", "text": "Planning the scenes"})
        roles = [message["role"] for message in self.host_.store.wire_messages()]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(self.states(), ["running", "completed"])

    def test_the_transcript_keeps_the_models_blocks_exactly_and_the_answers_in_order(self):
        self.run_loop(tool_turn(STATUS, ("list_universes", {})), tool_turn(FINISH))
        first = self.host_.store.wire_messages()[1]["content"]
        self.assertEqual([block["type"] for block in first], ["thinking", "tool_use", "tool_use"])
        answers = self.host_.store.wire_messages()[2]["content"]
        self.assertEqual([block["tool_use_id"] for block in answers], ["toolu_01", "toolu_02"])
        self.assertTrue(all(block["type"] == "tool_result" for block in answers))
        self.assertEqual(self.provider.snapshots[1][1]["content"][0]["signature"], first[0]["signature"])

    def test_a_tool_that_refuses_is_reported_to_the_model_and_the_run_goes_on(self):
        self.run_loop(tool_turn(("get_character", {"slug": "nobody"})), tool_turn(FINISH))
        answer = self.host_.store.wire_messages()[2]["content"][0]
        self.assertTrue(answer["is_error"])
        self.assertIn("NOT_FOUND", answer["content"][0]["text"])
        self.assertEqual(self.record()["state"], "completed")

    def test_finish_run_refuses_a_complete_run_with_open_items_until_it_says_partial(self):
        partial = ("finish_run", {"outcome": "partial", "summary": "Most done.",
                                  "checklist": [{"requirement": "Logo", "status": "skipped", "note": "not attached"}]})
        wrong = ("finish_run", {"outcome": "complete", "summary": "x",
                                "checklist": [{"requirement": "Logo", "status": "skipped"}]})
        self.assertEqual(self.run_loop(tool_turn(wrong), tool_turn(partial)), "completed")
        self.assertIn("REVIEW_REQUIRED", self.host_.store.wire_messages()[2]["content"][0]["content"][0]["text"])
        self.assertEqual(self.record()["outcome"]["kind"], "open_items")

    def test_a_model_that_stops_talking_is_reminded_twice_and_then_the_run_ends_with_open_items(self):
        state = self.run_loop(text_turn("Here is my plan."), text_turn("Still planning."), text_turn("Done, I think."))
        self.assertEqual(state, "completed")
        self.assertEqual(self.record()["outcome"]["kind"], "open_items")
        self.assertTrue(self.record()["outcome"]["ended_without_finish"])
        reminders = [m for m in self.host_.store.wire_messages() if m["role"] == "user" and "finish_run" in str(m)]
        self.assertEqual(len(reminders), 2)

    def test_a_reminder_that_works_ends_the_run_well(self):
        self.assertEqual(self.run_loop(text_turn("Ready."), tool_turn(FINISH)), "completed")
        self.assertEqual(self.record()["outcome"]["kind"], "complete")


class AnswerShapeTests(LoopCase):
    def test_a_cut_off_tool_call_is_not_kept_and_the_model_is_asked_for_smaller_calls(self):
        cut = tool_turn(("save_project", {"title": "x"}))
        cut.stop_reason = "max_tokens"
        self.run_loop(cut, tool_turn(FINISH))
        roles = [message["role"] for message in self.host_.store.wire_messages()]
        self.assertEqual(roles, ["user", "user", "assistant", "user"])
        self.assertIn("cut off", self.host_.store.wire_messages()[1]["content"][0]["text"])
        self.assertEqual(self.record()["steps"], 1)  # only finish_run ran
        self.assertGreater(self.record()["usage"]["cost_usd"], 2 * 0.001)  # the cut answer was paid for

    def test_a_refusal_ends_the_run_failed_and_is_not_kept(self):
        self.assertEqual(self.run_loop(refusal_turn("general_harms")), "failed")
        failure = self.record()["failure"]
        self.assertEqual(failure["kind"], "refusal")
        self.assertIn("general_harms", failure["message"])
        self.assertEqual(len(self.host_.store.wire_messages()), 1)

    def test_the_key_in_a_message_stops_the_run_and_is_never_written(self):
        key = "sk-" + "ant-" + "api03-" + "k" * 30
        state = self.run_loop(tool_turn(("update_status", {"phase": "concept", "text": f"my key is {key}"})))
        self.assertEqual((state, self.record()["failure"]["kind"]), ("failed", "leak"))
        self.assertNotIn(key, self.host_.store.path("transcript.jsonl").read_text(encoding="utf-8"))
        self.assertNotIn(key, self.host_.store.path("run.json").read_text(encoding="utf-8"))

    def test_waiting_for_a_gpu_job_shows_as_waiting_and_then_running_again(self):
        host = self.start(tool_turn(("wait_for_jobs", {"job_ids": [1], "timeout_seconds": 5})), tool_turn(FINISH))
        with host.queue_scope():
            job = queue_manager.add("obraz", "A street", {"prompt": "p"})
        self.finish_jobs(job.id)
        self.loop.run()
        self.assertEqual(self.states(), ["running", "waiting", "running", "completed"])
        reasons = [event.get("wait_reason") for event in host.store.events(limit=0) if event["type"] == "state"]
        self.assertEqual(reasons, ["", "gpu_job", "", ""])


class ErrorTests(LoopCase):
    def test_a_rate_limit_waits_as_long_as_the_api_says_and_carries_on(self):
        state = self.run_loop(error("rate_limit", "slow down", retry_after=42), tool_turn(FINISH))
        self.assertEqual(state, "completed")
        self.assertEqual(self.clock.slept, [42])
        self.assertIn("rate_limit", [event.get("wait_reason") for event in self.host_.store.events(limit=0)])

    def test_server_trouble_is_retried_with_a_growing_wait(self):
        self.run_loop(error("overloaded"), error("server"), error("network"), tool_turn(FINISH))
        self.assertEqual(self.clock.slept, [5, 10, 20])

    def test_a_cut_off_request_sets_a_part_of_its_cost_aside(self):
        self.run_loop(error("network"), tool_turn(FINISH))
        self.assertGreater(self.record()["usage"]["reserved_usd"], 0)

    def test_what_cannot_be_retried_fails_the_run_with_the_reason_and_keeps_it_resumable(self):
        for kind in ("auth", "billing", "model", "bad_request"):
            with self.subTest(kind=kind):
                self.assertEqual(self.run_loop(error(kind, f"{kind} words")), "failed")
                self.assertEqual(self.record()["failure"], {"kind": kind, "message": f"{kind} words"})

    def test_an_outage_that_lasts_too_long_leaves_the_run_interrupted(self):
        state = self.run_loop(*[error("network") for _ in range(40)])
        self.assertEqual((state, self.record()["interrupt_reason"]), ("interrupted", "api_outage"))
        self.assertGreater(sum(self.clock.slept), agent_loop.MAX_OUTAGE * 0.9)

    def test_a_conversation_that_is_too_long_starts_afresh_from_the_brief_and_a_card(self):
        host = self.start(tool_turn(STATUS), error("context", "too long"), tool_turn(FINISH))
        host.created("project", "a" * 32)
        self.assertEqual(self.loop.run(), "completed")
        record = self.record()
        self.assertEqual(record["fresh_conversations"], 1)
        sent = self.provider.snapshots[-1]
        self.assertEqual(len(sent), 1)  # the new conversation: the brief and the card in one message
        self.assertEqual(sent[0]["content"][0]["text"], "Make a 15 s coffee ad.")
        self.assertIn("project:" + "a" * 32, sent[0]["content"][-1]["text"])
        self.assertIn("update_status", sent[0]["content"][-1]["text"])

    def test_a_conversation_that_never_fits_fails_the_run(self):
        state = self.run_loop(error("too_large"), error("too_large"), error("too_large"))
        self.assertEqual((state, self.record()["failure"]["kind"]), ("failed", "too_large"))


class LimitTests(LoopCase):
    def test_at_ninety_percent_the_model_is_told_in_its_next_tool_answer(self):
        big = [UsageEntry(MODEL, input_tokens=4_600_000)]  # $18.4 of $20
        self.run_loop(tool_turn(STATUS, usage=big), tool_turn(FINISH))
        answers = self.host_.store.wire_messages()[4]["content"]  # the answers to the second turn's calls
        self.assertEqual(answers[-1]["type"], "text")
        self.assertIn("92%", answers[-1]["text"])

    def test_at_the_limit_only_finish_run_works_and_a_run_that_still_does_not_finish_stops(self):
        big = [UsageEntry(MODEL, input_tokens=6_000_000)]  # $24 of $20
        self.run_loop(tool_turn(STATUS, usage=big), tool_turn(STATUS), tool_turn(FINISH))
        self.assertIn("cost limit", self.host_.store.wire_messages()[3]["content"][0]["text"])
        answer = self.host_.store.wire_messages()[5]["content"][0]
        self.assertIn("BUDGET", answer["content"][0]["text"])
        self.assertEqual(self.record()["state"], "completed")
        state = self.run_loop(tool_turn(STATUS, usage=big), tool_turn(STATUS), tool_turn(STATUS), tool_turn(STATUS))
        self.assertEqual((state, self.record()["budget_reason"]), ("budget_stopped", "cost"))

    def test_many_tool_calls_and_a_long_time_do_not_stop_a_run_only_its_cost_does(self):
        script = [tool_turn(("list_universes", {}), ("update_status", {"phase": "concept", "text": f"step {n}"}))
                  for n in range(1, 40)] + [tool_turn(FINISH)]

        def slow(request):
            self.clock.now += 5 * 3600
            return tool_turn(STATUS)

        self.assertEqual(self.run_loop(slow, slow, slow, *script), "completed")
        self.assertGreater(self.record()["steps"], 70)
        self.assertGreater(self.record()["elapsed_seconds"], 14 * 3600)

    def test_the_same_call_again_and_again_is_warned_about_and_then_stopped(self):
        script = [tool_turn(("list_universes", {})) for _ in range(6)]
        state = self.run_loop(*script)
        self.assertEqual((state, self.record()["failure"]["kind"]), ("failed", "stuck"))
        self.assertIn("same call", self.host_.store.wire_messages()[6]["content"][-1]["text"])  # after the third

    def test_a_wait_for_jobs_that_is_repeated_is_not_a_loop(self):
        host = self.start(*[tool_turn(("wait_for_jobs", {"job_ids": [1], "timeout_seconds": 5})) for _ in range(7)],
                          tool_turn(FINISH))
        with host.queue_scope():
            queue_manager.add("obraz", "A street", {"prompt": "p"})
        with patch.object(queue_manager, "paused", return_value=False):
            self.assertEqual(self.loop.run(), "completed")

    def test_the_working_time_adds_up_on_the_clock(self):
        def slow(request):
            self.clock.now += 90
            return tool_turn(FINISH)

        self.run_loop(slow)
        self.assertGreaterEqual(self.record()["elapsed_seconds"], 90)


class FlagTests(LoopCase):
    def test_a_stop_ends_the_run_and_withdraws_its_waiting_jobs_but_not_the_owners(self):
        host = self.start(lambda request: (self.stop.set(), tool_turn(STATUS))[1])
        with host.queue_scope():
            mine = queue_manager.add("obraz", "Mine", {"prompt": "p"})
        theirs = queue_manager.add("obraz", "Theirs", {"prompt": "p"})
        self.assertEqual(self.loop.run(), "stopped")
        self.assertEqual([job.id for job in queue_manager.snapshot()], [theirs.id])
        self.assertNotIn(mine, queue_manager.snapshot())
        self.assertEqual(len(self.host_.store.wire_messages()), 3)  # the turn that arrived was kept and answered

    def test_a_pause_leaves_the_run_resumable_and_a_new_loop_carries_on_from_the_same_conversation(self):
        host = self.start(lambda request: (self.pause.set(), tool_turn(STATUS))[1])
        self.assertEqual(self.loop.run(), "paused")
        before = self.host_.store.wire_messages()
        self.provider = ScriptedProvider(tool_turn(FINISH))
        self.pause.clear()
        again = agent_loop.Loop(host.store, host, self.provider, clock=self.clock, stop=self.stop, pause=self.pause)
        self.assertEqual(again.run(), "completed")
        self.assertEqual(self.provider.snapshots[0], before)

    def test_a_stop_while_a_response_streams_leaves_the_run_stopped(self):
        host = self.start(lambda request: (_ for _ in ()).throw(Aborted()))
        self.stop.set()
        self.assertEqual(self.loop.run(), "stopped")
        del host

    def test_an_abort_that_nobody_asked_for_leaves_the_run_interrupted(self):
        self.start(lambda request: (_ for _ in ()).throw(Aborted()))
        self.assertEqual(self.loop.run(), "interrupted")

    def test_a_stop_during_a_wait_ends_it_at_once(self):
        host = self.start(error("overloaded"), tool_turn(FINISH))
        self.clock.sleep = lambda seconds, wake=lambda: False: self.stop.set()
        self.assertEqual(self.loop.run(), "stopped")
        del host


class ResumeTests(LoopCase):
    def interrupted_run(self, *, done=(), started=()):
        host = self.host()
        host.store.append_message("user", [{"type": "text", "text": "Make a 15 s coffee ad."}])
        host.store.append_message("assistant", [
            {"type": "tool_use", "id": "toolu_a", "name": "update_status", "input": {}},
            {"type": "tool_use", "id": "toolu_b", "name": "list_universes", "input": {}},
            {"type": "tool_use", "id": "toolu_c", "name": "list_voices", "input": {}}])
        for call in started:
            host.store.log("tool_start", id=call, tool="x")
        for call in done:
            host.store.log("tool_done", id=call, tool="x", ok=True, code="")
        return host

    def test_a_call_nobody_answered_is_answered_by_what_is_known_of_it(self):
        host = self.interrupted_run(started=["toolu_a", "toolu_b"], done=["toolu_a"])
        provider = ScriptedProvider(tool_turn(FINISH))
        loop = agent_loop.Loop(host.store, host, provider, clock=self.clock)
        self.assertEqual(loop.run(), "completed")
        answers = provider.snapshots[0][-1]["content"]
        texts = {block["tool_use_id"]: block["content"][0]["text"] for block in answers}
        self.assertTrue(texts["toolu_a"].startswith("COMPLETED") and texts["toolu_b"].startswith("INTERRUPTED"))
        self.assertTrue(texts["toolu_c"].startswith("NOT RUN"))
        self.assertTrue(all(block["is_error"] for block in answers))

    def test_only_what_happened_after_the_last_answer_decides_what_became_of_a_call(self):
        host = self.interrupted_run(started=["toolu_a"])
        host.store.log("tool_done", id="toolu_a", tool="x", ok=True, code="")  # an older call that reused the id
        host.store.log("turn", model="m", stop="tool_use")
        host.store.log("tool_start", id="toolu_a", tool="x")
        provider = ScriptedProvider(tool_turn(FINISH))
        agent_loop.Loop(host.store, host, provider, clock=self.clock).run()
        answers = {block["tool_use_id"]: block["content"][0]["text"] for block in provider.snapshots[0][-1]["content"]}
        self.assertTrue(answers["toolu_a"].startswith("INTERRUPTED"))

    def test_a_conversation_that_ended_on_the_models_words_is_asked_to_continue(self):
        host = self.host()
        host.store.append_message("user", [{"type": "text", "text": "Brief"}])
        host.store.append_message("assistant", [{"type": "text", "text": "Working on it."}])
        provider = ScriptedProvider(tool_turn(FINISH))
        agent_loop.Loop(host.store, host, provider, clock=self.clock).run()
        self.assertEqual([m["role"] for m in provider.snapshots[0]], ["user", "assistant", "user"])
        self.assertIn("interrupted", provider.snapshots[0][-1]["content"][0]["text"])

    def test_a_run_without_a_conversation_fails_instead_of_asking_nothing(self):
        host = self.host()
        loop = agent_loop.Loop(host.store, host, ScriptedProvider(), clock=self.clock)
        self.assertEqual(loop.run(), "failed")

    def test_an_unexpected_error_ends_the_run_failed_without_the_details(self):
        self.start(tool_turn(STATUS))
        with patch.object(agent_loop.Loop, "_handle", side_effect=RuntimeError("C:/secret")):
            self.assertEqual(self.loop.run(), "failed")
        self.assertNotIn("secret", str(self.record()["failure"]))
        self.assertIn("RuntimeError", self.record()["failure"]["message"])

    def test_the_key_is_not_in_anything_the_run_wrote(self):
        key = "sk-" + "ant-" + "api03-" + "z" * 30
        with patch.object(agent_config, "get_key", return_value=(key, "test")):
            self.run_loop(tool_turn(STATUS), error("network"), tool_turn(FINISH))
        for path in self.host_.store.dir.rglob("*"):
            if path.is_file():
                self.assertNotIn(key.encode(), path.read_bytes(), path.name)
