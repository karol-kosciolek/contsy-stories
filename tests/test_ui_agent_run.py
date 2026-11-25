# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Agent page's live view of a run (app/ui_agent_run.py): what it shows, that nothing a model or a file wrote reaches
the page unescaped, which buttons work in which state, and what the timer and the clicks do."""
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr

from app import agent_runner, main, ui_agent_run as page
from app.agent_runner import RunnerError

HOSTILE = "<img src=x onerror=alert(1)><script>alert(2)</script>"


def view(state="running", **fields):
    return {"id": "20260101-000000-abcdef", "state": state, "title": "Coffee ad", "created": 1_700_000_000,
            "settings": {"mode": "prepare", "model": "claude-opus-5-5",
                         "budgets": {"cost_usd": 20.0, "steps": 250, "gpu_minutes_prepare": 40,
                                     "gpu_minutes_produce": 240, "hours": 12.0}},
            "usage": {"cost_usd": 3.456}, "steps": 25, "elapsed_seconds": 3725, "gpu": {"seconds": 600},
            "status": {"phase": "timeline", "text": "Saving the scenes"}, "events": [], "working": True, **fields}


class RenderTests(unittest.TestCase):
    def test_no_run_says_so(self):
        self.assertIn("No run selected", page.run_html({}))

    def test_what_the_run_has_used_of_its_limits_and_what_it_is_doing(self):
        text = re.sub(r"<[^>]+>", " ", page.run_html(view()))
        text = " ".join(text.split())
        for expected in ("Coffee ad", "Working", "prepare", "claude-opus-5-5", "$3.46", "of $20", "25", "10 min",
                         "1 h 2 min" if False else "1:02 h", "timeline", "Saving the scenes"):
            self.assertIn(expected, text)
        for gone in ("of 250", "of 40 min", "of 12 h"):  # only the cost has a limit
            self.assertNotIn(gone, text)

    def test_every_state_has_a_word_and_a_shape_never_a_color(self):
        for state in page.STATE_BADGE:
            html = page.run_html(view(state))
            self.assertIn(page.STATE_BADGE[state][0], html)
            self.assertNotRegex(html, r"#[0-9a-fA-F]{3,6}|rgb\(|hsl\(|color:")

    def test_the_reasons_a_run_is_not_working_are_told_in_words(self):
        self.assertIn("Contsy was restarted", page.run_html(view("interrupted", interrupt_reason="app_restart")))
        self.assertIn("cost limit", page.run_html(view("budget_stopped", budget_reason="cost")))
        self.assertIn("GPU to finish", page.run_html(view("waiting", wait_reason="gpu_job")))
        self.assertIn("refused", page.run_html(view("failed", failure={"kind": "auth", "message": "The key was refused."})))
        done = page.run_html(view("completed", outcome={"kind": "open_items", "summary": "Two scenes left."}))
        self.assertIn("Finished with open items", done)
        self.assertIn("Two scenes left.", done)

    def test_nothing_a_model_or_a_file_wrote_reaches_the_page_as_markup(self):
        events = [{"type": "tool_done", "t": 1, "tool": HOSTILE, "ok": False, "code": HOSTILE},
                  {"type": "status", "t": 2, "phase": HOSTILE, "text": HOSTILE},
                  {"type": "question", "t": 3, "text": HOSTILE}, {"type": "api_error", "t": 4, "message": HOSTILE},
                  {"type": "state", "t": 5, "state": HOSTILE}, {"type": "resume", "t": 6, "reason": HOSTILE}]
        html = page.run_html(view(title=HOSTILE, status={"phase": HOSTILE, "text": HOSTILE}, events=events,
                                  questions=[HOSTILE], failure={"message": HOSTILE}, outcome={"summary": HOSTILE},
                                  wait_reason=HOSTILE, settings={"mode": HOSTILE, "model": HOSTILE, "budgets": {}}))
        self.assertNotIn("<img", html)
        self.assertNotIn("<script", html)
        self.assertIn("&lt;img", html)

    def test_the_log_shows_the_newest_first_a_limited_number_of_events_and_skips_the_noise(self):
        events = [{"type": "tool_done", "t": 1_700_000_000 + n, "tool": f"tool_{n}", "ok": True} for n in range(40)]
        events.insert(5, {"type": "turn", "t": 1, "text": "not shown"})
        html = page.run_html(view(events=events))
        self.assertEqual(html.count("<tr>") - 1, page.SHOWN_EVENTS)
        self.assertLess(html.index("tool_39"), html.index("tool_30"))
        self.assertNotIn("not shown", html)
        self.assertNotIn("tool_20<", html)

    def test_a_record_with_missing_or_broken_numbers_still_renders(self):
        html = page.run_html({"id": "x", "state": "running", "settings": {"budgets": {"steps": "many"}},
                              "usage": {"cost_usd": "n/a"}, "steps": None, "elapsed_seconds": float("nan"),
                              "gpu": {"seconds": [1]}})
        self.assertIn("Working", html)

    def test_the_run_list_names_each_run_by_its_time_title_and_state(self):
        runs = [{"id": "20260101-000000-abcdef", "title": "Coffee ad", "state": "completed", "created": 1_700_000_000}]
        with patch.object(page.agent_store, "list_runs", return_value=runs):
            [(label, value)] = page.run_choices()
        self.assertEqual(value, "20260101-000000-abcdef")
        self.assertTrue(label.endswith("Coffee ad · Finished"))


class ButtonTests(unittest.TestCase):
    def enabled(self, state, **fields):
        return [update["interactive"] for update in page.buttons(view(state, **fields))]

    def test_pause_and_stop_work_while_it_works_and_resume_when_it_can_continue(self):
        self.assertEqual(self.enabled("running"), [True, True, False, False])
        self.assertEqual(self.enabled("waiting"), [True, True, False, False])
        for state in ("paused", "interrupted", "stopped", "failed", "budget_stopped"):
            self.assertEqual(self.enabled(state), [False, False, True, False], state)
        self.assertEqual(self.enabled("created"), [False, False, False, False])

    def test_a_finished_run_that_only_prepared_can_be_approved_and_one_that_produced_cannot(self):
        outcome = {"kind": "complete"}
        self.assertEqual(self.enabled("completed", outcome=outcome), [False, False, False, True])
        self.assertEqual(self.enabled("completed", outcome={"kind": "open_items"}), [False, False, False, True])
        produced = {"mode": "produce", "budgets": {}}
        self.assertEqual(self.enabled("completed", outcome=outcome, settings=produced), [False] * 4)
        self.assertEqual([update["interactive"] for update in page.buttons({})], [False] * 4)


class TimerTests(unittest.TestCase):
    def test_the_card_is_drawn_again_only_when_something_changed_and_the_timer_follows_the_work(self):
        with patch.object(agent_runner, "view", return_value=view(events=[{"seq": 4, "type": "state"}])):
            card, *_, timer, signature = page.show("run")
            self.assertIn("Coffee ad", card)
            self.assertTrue(timer.active)
            same = page.show("run", signature=signature)
            self.assertEqual(same[:6], (gr.skip(),) * 6)
            self.assertTrue(same[6].active)
            self.assertNotEqual(page.show("run", force=True, signature=signature)[0], gr.skip())
        with patch.object(agent_runner, "view", return_value=view("completed", working=False)):
            self.assertFalse(page.show("run", force=True)[6].active)
        self.assertIn("No run selected", page.show("", force=True)[0])
        self.assertFalse(page.show("", force=True)[6].active)


class ClickTests(unittest.TestCase):
    def test_start_says_why_it_cannot_and_returns_the_run_when_it_can(self):
        with patch.object(agent_runner, "start", side_effect=RunnerError("Add your Claude API key first.")):
            message, run = page.start_click(["brief.md"], "", {})
        self.assertIn("key first", message)
        self.assertEqual(run, gr.skip())
        with patch.object(agent_runner, "start", side_effect=RuntimeError("secret C:/path")):
            message, _ = page.start_click([], "", {})
        self.assertNotIn("secret", message)
        with patch.object(agent_runner, "start", return_value="20260101-000000-abcdef") as start:
            message, run = page.start_click(["brief.md"], "x", {"mode": "prepare"})
        self.assertEqual(run, "20260101-000000-abcdef")
        start.assert_called_once_with(["brief.md"], "x", {"mode": "prepare"})
        self.assertIn("close this page", message)

    def test_the_other_buttons_report_what_happened(self):
        with patch.object(agent_runner, "pause", return_value=True), patch.object(agent_runner, "stop", return_value=False):
            self.assertIn("Pausing", page.pause_click("r"))
            self.assertIn("not working", page.stop_click("r"))
        with patch.object(agent_runner, "resume", side_effect=RunnerError("Add your Claude API key first.")):
            self.assertIn("key first", page.resume_click("r", {"cost_usd": 5.0}))
            self.assertIn("key first", page.approve_click("r"))
        with patch.object(agent_runner, "resume") as resume:
            self.assertIn("Resumed", page.resume_click("r", {"cost_usd": 50.0}))
            self.assertIn("Approved", page.approve_click("r"))
        self.assertEqual([call.kwargs for call in resume.call_args_list],
                         [{"budgets": {"cost_usd": 50.0}}, {"approve_production": True}])


class ChipTests(unittest.TestCase):
    def test_the_top_bar_says_the_agent_works_only_while_it_does(self):
        gpu = SimpleNamespace(available=False)
        with patch.object(main.engine_runtime.Engine, "available", return_value=True), \
                patch.object(main.queue_manager, "paused", return_value=False), \
                patch.object(main, "get_gpu_info", return_value=gpu), \
                patch.object(main.queue_manager, "snapshot", return_value=[]):
            idle = main.status_html()
            with patch.object(main.agent_runner, "busy", return_value=True):
                working = main.status_html()
        self.assertNotIn("Agent", idle)
        self.assertIn("Agent working", working)
        self.assertIn("working on a job", working)  # in the tooltip and for screen readers too


if __name__ == "__main__":
    unittest.main()
