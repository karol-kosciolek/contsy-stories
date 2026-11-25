# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Regression tests for the defects found when the Agent page's code was reviewed: each test fails on the code as it was
before the fix."""
import io
import json
import threading
import time
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import httpx2

from app import agent_brief, agent_claude, agent_config, agent_delivery as delivery, agent_loop, agent_runner
from app import agent_store, agent_tools, agent_tools_studio, characters, downloader, history, queue_manager, timeline
from app import ui_agent, ui_timeline, universes
from app.agent_host import ToolError
from app.agent_provider import Aborted, ProviderError, UsageEntry
from tests.agent_base import AgentCase, picture
from tests.agent_fake import MODEL, FakeClock, ScriptedProvider, error, text_turn, tool_turn
from tests.test_agent_loop import FINISH, STATUS, LoopCase
from tests.test_agent_runner import RunnerCase


# --- the run store ------------------------------------------------------------------------------------------------

class StoreTests(AgentCase):
    def test_a_message_with_a_unicode_line_separator_is_not_lost(self):
        store = self.host().store
        text = "line one line two line three\u0085line four"
        store.append_message("user", [{"type": "text", "text": text}])
        store.log("status", text=text)
        self.assertEqual(store.wire_messages()[0]["content"][0]["text"], text)
        self.assertEqual(store.events(limit=0)[-1]["text"], text)

    def test_a_tool_input_that_looks_like_a_blob_marker_comes_back_as_it_was(self):
        store = self.host().store
        tricky = {"type": "tool_use", "id": "t", "name": "x", "input": {
            "$blob": "a" * 64 + ".png", "blob": 1, "$$blob": 2, "nested": [{"$blob": "b" * 64 + ".jpg"}]}}
        store.append_message("assistant", [tricky])
        self.assertEqual(store.wire_messages()[0]["content"][0], tricky)
        self.assertEqual(len(store.wire_messages()), 1)

    def test_a_message_after_a_line_that_was_cut_short_is_kept(self):
        store = self.host().store
        store.append_message("user", [{"type": "text", "text": "one"}])
        with store.path("transcript.jsonl").open("ab") as stream:
            stream.write(b'{"i": 2, "role": "assistant", "content": [{"ty')  # the process died here
        store.append_message("user", [{"type": "text", "text": "three"}])
        self.assertEqual([m["content"][0]["text"] for m in store.wire_messages()], ["one", "three"])

    def test_run_json_is_written_and_read_again_when_windows_holds_the_file_for_a_moment(self):
        store = self.host().store
        real = agent_store.atomic_text
        calls = []

        def busy(path, text):
            calls.append(1)
            if len(calls) < 3:
                raise PermissionError("in use")
            real(path, text)

        with patch.object(agent_store, "atomic_text", side_effect=busy), patch.object(agent_store.time, "sleep"):
            store.update(steps=5)
        self.assertEqual((len(calls), store.read()["steps"]), (3, 5))
        reads = []
        original = type(store.path("run.json")).read_text

        def held(self_, *args, **kwargs):
            reads.append(1)
            if len(reads) < 2:
                raise PermissionError("in use")
            return original(self_, *args, **kwargs)

        with patch("pathlib.Path.read_text", held), patch.object(agent_store.time, "sleep"):
            self.assertEqual(store.read()["steps"], 5)
        with patch.object(agent_store, "atomic_text", side_effect=PermissionError("locked")), \
                patch.object(agent_store.time, "sleep"), self.assertRaises(PermissionError):
            store.update(steps=6)


# --- the Claude client --------------------------------------------------------------------------------------------

class ClientTests(AgentCase):
    def test_a_stream_that_ends_before_the_message_does_is_not_a_turn(self):
        class Stream:
            def __iter__(self):
                return iter([])

            def get_final_message(self):
                return SimpleNamespace(stop_reason=None, content=[], usage=None, model=MODEL, id="m")

            def close(self):
                pass

        class Manager:
            def __enter__(self):
                return Stream()

            def __exit__(self, *args):
                return False

        client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=lambda **params: Manager())))
        provider = agent_claude.ClaudeProvider()
        with patch.object(provider, "_get_client", return_value=client), self.assertRaises(ProviderError) as caught:
            provider._stream({"model": MODEL}, lambda: False, lambda name: None)
        self.assertEqual(caught.exception.kind, "network")

    def test_a_connection_that_breaks_in_the_middle_of_an_answer_is_a_network_error_not_an_unknown_one(self):
        cases = [(httpx2.ReadError("broken"), "network"), (httpx2.RemoteProtocolError("cut"), "network"),
                 (httpx2.ReadTimeout("slow"), "timeout"), (ConnectionResetError("reset"), "network"),
                 (TimeoutError("slow"), "timeout"), (RuntimeError("something else"), "unknown")]
        for exc, kind in cases:
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(agent_claude.map_error(exc).kind, kind)
                self.assertTrue(agent_claude.map_error(exc).retryable or kind == "unknown")


# --- the loop -----------------------------------------------------------------------------------------------------

class LoopFixTests(LoopCase):
    def test_a_turn_that_says_the_context_is_full_starts_a_fresh_conversation_and_is_paid_for(self):
        full = text_turn("x", stop="model_context_window_exceeded")
        state = self.run_loop(tool_turn(STATUS), full, tool_turn(FINISH))
        record = self.record()
        self.assertEqual((state, record["fresh_conversations"]), ("completed", 1))
        self.assertGreaterEqual(record["usage"]["cost_usd"], 3 * 0.001)
        self.assertEqual(record["usage"]["turns"], 2)  # the full one was not kept as a turn

    def test_an_empty_answer_is_not_kept_and_the_model_is_asked_to_go_on(self):
        empty = text_turn("")
        empty.content, empty.text = [], ""
        state = self.run_loop(empty, tool_turn(FINISH))
        self.assertEqual(state, "completed")
        roles = [m["role"] for m in self.host_.store.wire_messages()]
        self.assertEqual(roles, ["user", "user", "assistant", "user"])
        self.assertIn("empty", self.host_.store.wire_messages()[1]["content"][0]["text"])
        three = []
        for _ in range(4):
            turn = text_turn("")
            turn.content, turn.text = [], ""
            three.append(turn)
        self.assertEqual(self.run_loop(*three), "completed")
        self.assertEqual(self.record()["outcome"]["kind"], "open_items")

    def test_nothing_runs_after_finish_run_in_the_same_answer(self):
        state = self.run_loop(tool_turn(FINISH, STATUS, ("list_universes", {})))
        answers = self.host_.store.wire_messages()[2]["content"]
        self.assertEqual(state, "completed")
        self.assertEqual([block["tool_use_id"] for block in answers], ["toolu_01", "toolu_02", "toolu_03"])
        self.assertTrue(all("NOT RUN" in block["content"][0]["text"] for block in answers[1:]))
        self.assertEqual(self.record()["steps"], 1)
        self.assertNotIn("status", self.record())

    def test_the_reminders_start_over_whenever_the_model_works_again(self):
        script = [text_turn("planning"), tool_turn(STATUS), text_turn("still planning"), tool_turn(STATUS),
                  text_turn("thinking"), tool_turn(FINISH)]
        self.assertEqual(self.run_loop(*script), "completed")
        self.assertEqual(self.record()["outcome"]["kind"], "complete")

    def test_unclassified_errors_far_apart_do_not_add_up_to_a_failure(self):
        script = []
        for _ in range(agent_loop.MAX_UNKNOWN + 2):
            script += [error("unknown", "odd"), tool_turn(STATUS)]
        self.assertEqual(self.run_loop(*script, tool_turn(FINISH)), "completed")

    def test_what_is_set_aside_for_cut_requests_ignores_pictures_and_has_a_ceiling(self):
        picture_block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A" * 4_000_000}}
        messages = [{"role": "user", "content": [picture_block, {"type": "text", "text": "x" * 400}]}]
        self.assertLess(agent_loop._estimate_tokens(messages), 3000)
        script = [error("network") for _ in range(30)] + [tool_turn(FINISH)]
        self.run_loop(*script, budgets={"cost_usd": 10.0})
        self.assertLessEqual(self.record()["usage"]["reserved_usd"], 1.0 + 1e-9)

    def test_a_response_aborted_by_stop_is_set_aside_too(self):
        def abort(request):
            self.stop.set()
            raise Aborted()

        host = self.start(abort)
        self.assertEqual(self.loop.run(), "stopped")
        self.assertGreater(host.store.read()["usage"]["reserved_usd"], 0)

    def test_the_card_of_a_fresh_conversation_says_when_the_run_produces(self):
        self.start(tool_turn(FINISH), mode="produce")
        self.assertIn("PRODUCES", self.loop.state_card("context"))
        self.start(tool_turn(FINISH))
        self.assertNotIn("PRODUCES", self.loop.state_card("context"))

    def test_stopping_withdraws_the_jobs_that_are_there_even_when_one_has_vanished(self):
        host = self.start(lambda request: (self.stop.set(), tool_turn(STATUS))[1])
        with host.queue_scope():
            jobs = [queue_manager.add("obraz", f"Job {n}", {"prompt": "p"}) for n in range(3)]
        real = type(host).job
        taken = []

        def vanishing(self_, job_id):
            if not taken:
                taken.append(job_id)
                queue_manager._JOBS.remove(next(j for j in queue_manager._JOBS if j.id == jobs[1].id))
            return real(self_, job_id)

        with patch.object(type(host), "job", vanishing):
            self.assertEqual(self.loop.run(), "stopped")
        self.assertNotIn(jobs[0], queue_manager.snapshot())
        self.assertNotIn(jobs[2], queue_manager.snapshot())


# --- the runner ---------------------------------------------------------------------------------------------------

class PausingClock(FakeClock):
    def __init__(self, on_sleep):
        super().__init__()
        self.on_sleep, self.count = on_sleep, 0

    def sleep(self, seconds, wake=lambda: False):
        self.count += 1
        if self.count == 3:
            self.on_sleep()
        super().sleep(seconds, wake)


class RunnerFixTests(RunnerCase):
    def test_a_pause_ends_a_wait_for_jobs_at_once_and_leaves_the_job_queued(self):
        run_id = None
        clock = PausingClock(lambda: agent_runner.pause(run_id))
        with patch.object(agent_runner, "make_clock", return_value=clock):
            self.provider.script = [tool_turn(("update_status", {"phase": "intake", "text": "x"})),
                                    lambda request: tool_turn(("queue_still", {"prompt": "A cup", "format": "Square 1:1"})),
                                    lambda request: tool_turn(("wait_for_jobs", {"job_ids": [1], "timeout_seconds": 1500})),
                                    tool_turn(FINISH)]
            run_id = agent_runner.start([self.brief], "", {})
            agent_runner.wait(run_id)
        self.assertEqual(self.record(run_id)["state"], "paused")
        self.assertLess(clock.now - 1_000_000.0, 60)  # it did not wait the 25 minutes
        self.assertEqual([job.status for job in queue_manager.snapshot()], [queue_manager.PENDING])

    def test_two_different_files_with_one_name_are_both_kept(self):
        other = self.root / "other"
        other.mkdir()
        second = other / "brief.md"
        second.write_text("# A second brief\nAudience: pensioners.\n", encoding="utf-8")
        run_id = self.begin(tool_turn(("get_brief", {})), tool_turn(FINISH), files=[self.brief, second])
        store = agent_store.RunStore(run_id)
        self.assertEqual(len(list(store.path("brief").iterdir())), 2)
        answer = store.wire_messages()[2]["content"][0]["content"][0]["text"]
        self.assertIn("pensioners", answer)
        self.assertIn("students", answer)

    def test_a_run_resumed_after_a_limit_is_told_the_new_limits(self):
        big = [UsageEntry(MODEL, input_tokens=6_000_000)]
        run_id = self.begin(tool_turn(STATUS, usage=big), tool_turn(STATUS), tool_turn(STATUS), tool_turn(STATUS))
        self.provider.script = [tool_turn(FINISH)]
        agent_runner.resume(run_id, budgets={"cost_usd": 100.0})
        agent_runner.wait(run_id)
        sent = self.provider.snapshots[-1]
        note = next(m for m in sent if m["role"] == "user" and "resumed this run after it reached its cost limit" in str(m))
        self.assertIn("$100", note["content"][0]["text"])


# --- the host and the tools ---------------------------------------------------------------------------------------

class NoteTests(AgentCase):
    def test_an_ai_edit_and_a_chosen_take_of_the_owners_project_are_noted_for_the_report(self):
        project = timeline.save(None, None, "Owner film", (timeline.Scene("A tree in wind", 4),), "Landscape 16:9", True, [])
        roomy = self.host(mode="produce")

        def edit(saved, index, scene, *, change="", scope="scene"):
            return saved, queue_manager.add(timeline.EDIT, "Edit", {"project": saved.id, "seconds": 4})

        with patch.object(timeline, "edit", side_effect=edit):
            self.ok(roomy, "edit_scene", project_id=project.id, expected_revision=1, scene=0, prompt="A big tree")
        self.assertTrue(roomy.store.read()["changes"][0].startswith("Scene 1 of project"))
        with patch.object(timeline, "takes", return_value=[]), patch.object(timeline, "use_take", return_value=project):
            self.clear_queue()
            self.ok(roomy, "choose_take", project_id=project.id, scene=0, key="k")
        self.assertEqual(len(roomy.store.read()["changes"]), 2)


class HostFixTests(AgentCase):
    def test_a_character_that_could_not_be_binned_is_not_reported_deleted(self):
        saved = self.saved_character()
        host = self.host(**self.allow(delete=True))
        with patch("app.trash.move", return_value=False), self.assertRaises(ToolError) as caught:
            host.delete("character", saved["slug"])
        self.assertEqual(caught.exception.code, "BUSY")
        self.assertIsNotNone(characters.get(saved["slug"]))
        self.assertNotIn("changes", host.store.read())
        place = self.saved_item()
        with patch("app.trash.move", return_value=False), self.assertRaises(ToolError) as caught:
            host.delete("item", place["slug"])
        self.assertEqual(caught.exception.code, "BUSY")

    def test_deleting_a_character_copies_the_universes_it_leaves_and_says_so(self):
        self.saved_character("Mia")
        self.saved_character("Leo")
        crew, _ = universes.save("Crew", ["mia", "leo"])
        host = self.host(**self.allow(delete=True))
        host.delete("character", "mia")
        backup = host.store.path("backups", "002-universes-with-mia", "record.json")
        self.assertEqual(json.loads(backup.read_text())[0]["slug"], crew["slug"])
        self.assertTrue(any("left the universes: Crew" in line for line in host.store.read()["changes"]))

    def test_a_project_whose_files_are_in_use_is_busy_not_missing(self):
        project = timeline.save(None, None, "Ad", (timeline.Scene("A cup", 3),), "Landscape 16:9", True, [])
        host = self.host(**self.allow(delete=True))
        with patch.object(timeline, "delete", side_effect=ValueError("The project's files are still in use.")), \
                self.assertRaises(ToolError) as caught:
            host.delete("project", project.id)
        self.assertEqual(caught.exception.code, "BUSY")

    def test_a_refused_save_of_a_universe_or_style_leaves_no_note_of_a_change(self):
        self.saved_character("Mia")
        self.saved_character("Leo")
        crew, _ = universes.save("Crew", ["mia", "leo"])
        host = self.host()
        self.refused(host, "save_universe", "INVALID", name="Crew", members=["mia"], slug=crew["slug"])
        self.assertNotIn("changes", host.store.read())

    def test_a_tool_may_not_download_a_model_unless_the_owner_allowed_it(self):
        seen = []
        probe = agent_tools.Tool("probe_downloads", "probe", agent_tools.obj({}),
                                 lambda host, args: seen.append(downloader.downloads_blocked()) or "ok")
        with patch.dict(agent_tools._TOOLS, {"probe_downloads": probe}):
            self.ok(self.host(), "probe_downloads")
            self.ok(self.host(**self.allow(downloads=True)), "probe_downloads")
        self.assertEqual(seen, [True, False])
        self.assertFalse(downloader.downloads_blocked())

    def test_the_label_prefix_does_not_hide_the_action_in_the_timeline_chip(self):
        host = self.host()
        with host.queue_scope():
            job = queue_manager.add(timeline.PREPARE, "Generate scene 3 · My film", {"project": "x"})
        self.assertNotIn(" · ", job.label.split(" · ")[0].replace("Generate scene 3", ""))
        self.assertEqual(ui_timeline._job_name(job), "Agent: Generate scene 3")


class ToolFixTests(AgentCase):
    def test_a_picture_of_a_character_made_from_an_uploaded_photo_stays_on_this_pc(self):
        saved = self.saved_character()  # its photos were uploaded
        result = self.root / "outputs" / "scene.png"
        picture(result, size=(64, 48))
        history.record({"files": [str(result)], "kind": "obraz", "prompt": "Mia", "profile": saved["slug"]})
        shown = self.call(self.host(send_images=True), "inspect_media", file="scene.png")
        self.assertEqual(shown.images, [])
        self.assertIn("stays on this PC", shown.text)
        record = json.loads(characters._file(saved["slug"]).read_text(encoding="utf-8"))
        for photo in record["photos"]:
            photo["origin"] = "generated"
        characters._file(saved["slug"]).write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(len(self.call(self.host(send_images=True), "inspect_media", file="scene.png").images), 1)
        other = self.root / "outputs" / "plain.png"
        picture(other, size=(64, 48))
        history.record({"files": [str(other)], "kind": "obraz", "prompt": "a cup"})
        self.assertEqual(len(self.call(self.host(send_images=True), "inspect_media", file="plain.png").images), 1)

    def test_a_character_record_that_cannot_be_read_blocks_the_picture(self):
        saved = self.saved_character()
        result = self.root / "outputs" / "scene.png"
        picture(result, size=(64, 48))
        history.record({"files": [str(result)], "kind": "obraz", "prompt": "Mia", "profile": saved["slug"]})
        characters._file(saved["slug"]).write_text("not json", encoding="utf-8")
        self.assertTrue(agent_tools_studio._shows_uploaded_photo(result))

    def test_a_project_without_scenes_is_not_ready_and_a_bad_selection_is_refused(self):
        host = self.host()
        project = timeline.save(None, None, "Empty", (), "Landscape 16:9", True, [])
        answer = self.ok(host, "check_project", project_id=project.id)
        self.assertFalse(answer["ready"])
        self.assertIn("no scenes", answer["issues"][0]["text"])
        other = timeline.save(None, None, "One", (timeline.Scene("A cup", 3),), "Landscape 16:9", True, [])
        self.refused(host, "check_project", "INVALID", project_id=other.id, scenes=[7])
        self.assertTrue(self.ok(host, "check_project", project_id=other.id, scenes=[0])["ready"])


class DeliveryFixTests(AgentCase):
    def test_a_film_in_the_package_that_is_not_the_current_export_is_refused(self):
        host = self.host(mode="produce")
        project = timeline.save(None, None, "Ad", (timeline.Scene("A cup", 4),), "Landscape 16:9", True, [])
        host.created("project", project.id)
        old = self.root / "outputs" / "old.mp4"
        new = self.root / "outputs" / "new.mp4"
        old.write_bytes(b"first cut")
        new.write_bytes(b"second cut")
        delivery.add_file(host.store, old, "Ad", "film", project=project.id)
        delivery.build(host.store)
        with patch.object(timeline, "exported", return_value=new), patch("app.ffmpeg_utils.probe_duration", return_value=4.0):
            failing = delivery.gate_failures(host)
            self.assertEqual([code for code, _ in failing], ["G4"])
            self.assertIn("older cut", failing[0][1])
            delivery.add_file(host.store, new, "Ad", "film", project=project.id)
            delivery.build(host.store)
            self.assertEqual(delivery.gate_failures(host), [])


# --- the page and the brief -----------------------------------------------------------------------------------------

class PageFixTests(AgentCase):
    def build(self):
        with gr.Blocks() as demo:
            with gr.Tab("Agent") as tab:
                wire = ui_agent.build_agent_tab(tab)
            wire(demo)
        return {getattr(fn.fn, "__name__", ""): fn.fn for fn in demo.fns.values()}

    def test_resume_keeps_a_runs_limits_unless_it_stopped_at_one(self):
        handlers = self.build()
        host = self.host(budgets={"cost_usd": 5.0})
        form = [20.0, 250, 40, 240, 12.0]
        with patch.object(agent_runner, "resume") as resume:
            host.store.update(state="paused")
            handlers["resume_handler"](host.store.id, *form)
            host.store.update(state="budget_stopped")
            handlers["resume_handler"](host.store.id, *form)
        self.assertEqual([call.kwargs["budgets"] is None for call in resume.call_args_list], [True, False])
        self.assertEqual(resume.call_args_list[1].kwargs["budgets"]["cost_usd"], 20.0)

    def test_the_timer_refreshes_start_and_the_history_when_the_run_changes_and_not_otherwise(self):
        tick = self.build()["tick"]
        view = {"state": "completed", "title": "x", "settings": {"mode": "prepare", "budgets": {}}, "updated": 1,
                "events": [{"seq": 3}], "working": False}
        with patch.object(agent_runner, "view", return_value=view):
            changed = tick("20260101-000000-abcdef", "", 0, "", "prepare", False)
            same = tick("20260101-000000-abcdef", changed[7], 0, "", "prepare", False)
        self.assertEqual(len(changed), 8 + 5)
        self.assertNotEqual(changed[10], gr.skip())  # the history is drawn again
        self.assertEqual(same[8:], (gr.skip(),) * 5)


class BriefFixTests(AgentCase):
    def docx(self, document: bytes, *, method=zipfile.ZIP_DEFLATED) -> bytes:
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as bundle:
            bundle.writestr("word/document.xml", document, compress_type=method)
        return out.getvalue()

    def test_a_word_file_in_utf16_cannot_smuggle_an_entity_past_the_guard(self):
        xml = ('<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE d [<!ENTITY e "x">]>'
               '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
               '<w:p><w:r><w:t>&e;</w:t></w:r></w:p></w:body></w:document>')
        for encoded in (xml.encode("utf-16"), xml.encode("utf-16-le"), xml.encode("utf-16-be")):
            with self.subTest(head=encoded[:4]), self.assertRaises(agent_brief.BriefError):
                agent_brief.docx_text(self.docx(encoded))

    def test_a_word_file_that_is_odd_inside_is_refused_in_words_never_crashes(self):
        good = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r>'
                '<w:t>Hello</w:t></w:r></w:p></w:body></w:document>').encode()
        self.assertEqual(agent_brief.docx_text(self.docx(good)), "Hello")
        with self.assertRaises(agent_brief.BriefError):
            agent_brief.docx_text(self.docx(good, method=zipfile.ZIP_BZIP2)[:-30] + b"x" * 30)
        data = self.docx(good)
        for failure in (NotImplementedError("compression type 99"), RuntimeError("encrypted"), EOFError("short")):
            with self.subTest(failure=type(failure).__name__), patch("zipfile.ZipFile.open", side_effect=failure), \
                    self.assertRaises(agent_brief.BriefError):
                agent_brief.docx_text(data)
        deep = ("<a>" * 5000 + "</a>" * 5000).encode()
        with self.assertRaises(agent_brief.BriefError):
            agent_brief.docx_text(self.docx(deep))
