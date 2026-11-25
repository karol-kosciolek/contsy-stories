# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Agent page: it builds without reading the disk, every event is private, the key is handled and shown masked,
the attached files and Start's reason follow the rules, and the page lives first in the top bar. State and the network
are replaced (a temporary folder, a scripted provider)."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr

from app import agent_config, agent_preflight, agent_store, main, timeline, ui_agent, ui_theme
from app import agent_provider as provider
from app.agent_provider import ProviderError
from tests.agent_fake import ScriptedProvider

KEY = "sk-" + "ant-" + "api03-" + "Hh1Jj2Kk3Ll4Mm5Nn6Oo7Pp8Qq9"
SKIP = gr.skip()


class _Isolated(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.root = Path(self.context.enter_context(tempfile.TemporaryDirectory())).resolve()
        enter = self.context.enter_context
        enter(patch.object(agent_config, "CREDENTIALS_FILE", self.root / "agent_credentials.json"))
        enter(patch.object(agent_config, "SETTINGS_FILE", self.root / "agent_settings.json"))
        enter(patch.object(agent_store, "RUNS_DIR", self.root / "agent_runs"))
        enter(patch.object(timeline, "DIR", self.root / "projects"))
        env = {name: value for name, value in os.environ.items() if not name.startswith("ANTHROPIC_")}
        enter(patch.dict(os.environ, env, clear=True))
        agent_config._CACHE.update(stamp=None, key="")
        # preflight reads the models, the queue and the disk: stood in for
        enter(patch.object(agent_preflight, "model_rows", return_value=[("z-image-turbo", "Z-Image-Turbo", 0)]))
        enter(patch.object(agent_preflight.queue_manager, "snapshot", return_value=[]))
        enter(patch.object(agent_preflight.queue_manager, "paused", return_value=False))
        enter(patch.object(agent_store, "free_gb", return_value=400.0))

    def file(self, name, data) -> str:
        path = self.root / name
        path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
        return str(path)


class BuildTests(_Isolated):
    def build(self):
        with gr.Blocks() as demo:
            with gr.Tab("Agent", id="agent", elem_id="agent-page") as tab:
                wire = ui_agent.build_agent_tab(tab)
            wire(demo)
        return demo, {block.elem_id: block for block in demo.blocks.values() if getattr(block, "elem_id", None)}

    def test_the_page_builds_without_reading_the_disk_and_all_events_are_private(self):
        def forbidden(*_args, **_kwargs):
            raise AssertionError("The page read the disk while it was built.")

        with patch.object(agent_config, "load_settings", forbidden), patch.object(agent_config, "key_status", forbidden), \
                patch.object(agent_store, "list_runs", forbidden), patch.object(agent_preflight, "checks", forbidden):
            demo, by_id = self.build()
        self.assertTrue(all(fn.api_visibility == "private" for fn in demo.fns.values()))
        handlers = {fn.fn for fn in demo.fns.values()}
        for fn in (ui_agent.save_key_click, ui_agent.remove_key_click, ui_agent.test_connection_click,
                   ui_agent.files_change, ui_agent.refresh_status, ui_agent.load_everything, ui_agent.save_defaults,
                   ui_agent.model_hint_html):
            self.assertIn(fn, handlers)
        self.assertEqual([value for _, value in by_id["agent-model"].choices], list(provider.MODELS))
        self.assertEqual(by_id["agent-model"].value, provider.DEFAULT_MODEL)
        self.assertEqual(by_id["agent-key"].type, "password")
        self.assertFalse(by_id["agent-start"].interactive)  # until the page loads and checks Start's conditions

    def test_the_saved_settings_fill_the_form_only_when_the_page_loads_never_when_a_field_changes(self):
        demo, _ = self.build()
        loaders = [fn for fn in demo.fns.values() if fn.fn is ui_agent.load_everything]
        self.assertEqual(len(loaders), 1)
        self.assertEqual([event for _, event in loaders[0].targets], ["load"])  # a load of the page, no field's change
        for name in ("tick", "pick_runs"):
            fns = [fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == name]
            self.assertTrue(fns)

    def test_the_defaults_in_the_form_are_the_safe_ones(self):
        _, by_id = self.build()
        checked = {name: by_id[f"agent-permission-{name.replace('_', '-')}"].value
                   for name in agent_config.PERMISSIONS}
        self.assertEqual(checked, agent_config.DEFAULT_PERMISSIONS)
        self.assertTrue(by_id["agent-send-images"].value)
        self.assertTrue(by_id["agent-keep-awake"].value)
        self.assertEqual(by_id["agent-mode"].value, "prepare")
        self.assertEqual(by_id["agent-cost-usd"].value, 20.0)
        for name in agent_config.PERMISSIONS:
            self.assertTrue(by_id[f"agent-permission-{name.replace('_', '-')}"].info)

    def test_the_page_is_first_in_the_top_bar_with_an_icon_and_its_runs_are_never_served(self):
        self.assertEqual(main.TOOLS[0], ("Agent", "agent"))  # Contsy Stories: the agent's log is a tool
        self.assertEqual(main.START_PAGE, "stories")
        self.assertIn("agent", ui_theme.ICONS)
        blocked = main.local_launch_options({"host": "127.0.0.1"})["blocked_paths"]
        self.assertIn(str((self.root / "agent_runs").resolve()), blocked)


class FormTests(_Isolated):
    def test_the_form_and_the_settings_convert_both_ways(self):
        settings = agent_config.clean_settings({"model": "claude-sonnet-5-5", "effort": "high", "mode": "produce",
                                                "send_images": True, "keep_awake": False,
                                                "budgets": {"cost_usd": 5, "steps": 50},
                                                "permissions": {"delete": True, "uploads": True}})
        values = ui_agent.form_from_settings(settings)
        self.assertEqual(len(values), len(ui_agent.SETTING_FIELDS) + len(ui_agent.BUDGET_FIELDS)
                         + len(ui_agent.PERMISSION_FIELDS))
        self.assertEqual(ui_agent.settings_from_form(*values), settings)
        self.assertEqual(ui_agent.settings_from_form(*([None] * len(values)))["model"], provider.DEFAULT_MODEL)

    def test_defaults_are_saved_without_the_consent_for_attached_files(self):
        settings = agent_config.clean_settings({"mode": "produce", "permissions": {"uploads": True, "queue_all": True}})
        message = ui_agent.save_defaults(*ui_agent.form_from_settings(settings))
        self.assertIn("never saved", message)
        saved = agent_config.load_settings()
        self.assertEqual(saved["mode"], "produce")
        self.assertTrue(saved["permissions"]["queue_all"])
        self.assertFalse(saved["permissions"]["uploads"])

    def test_the_model_hint_names_the_price_as_an_estimate(self):
        hint = ui_agent.model_hint_html("claude-opus-5-5")
        self.assertIn("estimate", hint)
        self.assertIn("$4", hint)
        self.assertIn(provider.MODELS["claude-haiku-5-5"].hint, ui_agent.model_hint_html("claude-haiku-5-5"))
        self.assertIn(provider.MODELS[provider.DEFAULT_MODEL].hint, ui_agent.model_hint_html("nonsense"))


@unittest.skipUnless(sys.platform == "win32", "the key is encrypted with Windows DPAPI")
class KeyTests(_Isolated):
    def test_saving_testing_and_removing_the_key_shows_only_its_end(self):
        status, cleared = ui_agent.save_key_click("hello")
        self.assertIn("does not look like", status)
        self.assertEqual(cleared, SKIP)
        status, cleared = ui_agent.save_key_click(KEY)
        self.assertIn(KEY[-4:], ui_agent.key_status_note())
        self.assertNotIn(KEY, status + ui_agent.key_status_note())
        self.assertEqual((cleared, "Not tested yet." in status), ("", True))
        self.assertIn("encrypted", ui_agent.key_status_note())

        fake = ScriptedProvider()
        with patch.object(provider, "get_provider", return_value=fake):
            result = ui_agent.test_connection_click("claude-sonnet-5-5", progress=lambda *a, **k: None)
        self.assertEqual(fake.pings, ["claude-sonnet-5-5"])
        self.assertIn("Claude Opus 5.5 answered", result)
        fake.ping_result = ProviderError("auth", "Anthropic did not accept the API key.")
        with patch.object(provider, "get_provider", return_value=fake):
            self.assertIn("did not accept", ui_agent.test_connection_click("x", progress=lambda *a, **k: None))
        fake.ping_result = RuntimeError(f"broke with {KEY}")
        with patch.object(provider, "get_provider", return_value=fake):
            failed = ui_agent.test_connection_click("x", progress=lambda *a, **k: None)
        self.assertNotIn(KEY, failed)
        removed, cleared = ui_agent.remove_key_click()
        self.assertIn("removed", removed)
        self.assertIn("No key saved", ui_agent.key_status_note())

    def test_an_environment_key_wins_and_says_so(self):
        agent_config.save_key(KEY)
        with patch.dict(os.environ, {agent_config.ENV_KEY: "sk-" + "ant-" + "api03-" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3"}):
            text = ui_agent.key_status_note()
        self.assertIn("ANTHROPIC_API_KEY", text)
        self.assertIn("not used while it is set", text)

    def test_without_a_key_the_test_asks_for_one_and_calls_nothing(self):
        with patch.object(provider, "get_provider", side_effect=AssertionError("no provider without a key")):
            self.assertIn("Save a key first", ui_agent.test_connection_click("claude-opus-5-5",
                                                                                 progress=lambda *a, **k: None))


class BriefTests(_Isolated):
    def test_attached_files_are_listed_with_their_status_and_names_are_escaped(self):
        good = self.file("brief.md", "Make an ad for a coffee shop.")
        bad = self.file("tool.exe", b"MZ")
        html, accepted = ui_agent.files_change([good, bad], False)
        self.assertEqual(accepted, 1)
        self.assertIn("brief.md", html)
        self.assertIn("Refused", html)
        self.assertIn("not a kind of file", html)
        # names are labels from the owner's disk or a model: escaped wherever they are shown
        from app import agent_brief

        hostile = agent_brief.Attachment("<script>alert(1)</script>.md", "text", 10, "a" * 64, text="x", suffix="md")
        listed = ui_agent.files_html([hostile], [("<img src=x onerror=alert(1)>", "<b>why</b>")])
        self.assertNotIn("<script>", listed)
        self.assertNotIn("<img", listed)
        self.assertNotIn("<b>why</b>", listed)
        self.assertIn("&lt;script&gt;", listed)
        empty, none = ui_agent.files_change(None, False)
        self.assertEqual(none, 0)
        self.assertIn("No files yet", empty)

    def test_a_pdf_is_accepted_with_images_on_or_off_and_the_page_says_where_it_goes(self):
        pdf = self.file("deck.pdf", b"%PDF-1.7 data")
        for images in (False, True):
            table, accepted = ui_agent.files_change([pdf], images)
            self.assertEqual(accepted, 1)
            self.assertIn("Goes to Anthropic whole", table)

    def test_start_states_follow_the_order_the_owner_can_fix_them(self):
        key = {"source": "saved", "hint": "sk-ant-…1234"}
        state = lambda **options: ui_agent.start_state(**{"key": key, "accepted": 1, "extra": "", "mode": "prepare",  # noqa: E731
                                                          "downloads": False, **options})
        self.assertEqual(state(key={"source": "none"})[0], False)
        self.assertIn("Add your Claude key", state(key={"source": "none"}, accepted=0)[1])
        self.assertIn("Add a brief file", state(accepted=0)[1])
        self.assertTrue(state(accepted=0, extra="Make a 10 s ad")[0])
        self.assertIn("content rules", state(accepted=0, extra="make her a naked child")[1])
        self.assertIn("already in progress", state(running=True)[1])
        self.assertEqual(state(ready=False), (False, "⚠ The run loop is not part of this build yet."))
        enabled, reason = state()
        self.assertTrue(enabled)
        self.assertIn("wait for your approval", reason)
        with patch.object(agent_preflight, "missing_models", return_value=[("wan", "Wan 2.2", 37_000_000_000)]):
            enabled, reason = state(mode="produce")
            self.assertEqual((enabled, "Wan 2.2" in reason), (False, True))
            self.assertTrue(state(mode="produce", downloads=True)[0])
            self.assertTrue(state(mode="prepare")[0])


class StatusTests(_Isolated):
    def test_refresh_returns_every_panel_and_start_follows_the_key_and_the_brief(self):
        with patch.object(ui_agent, "runner_available", return_value=True):
            key, preflight, history, start, note = ui_agent.refresh_status(0, "", "prepare", False)
            self.assertEqual((start["interactive"], "Add your Claude key" in note), (False, True))
            agent_config.CREDENTIALS_FILE.write_text("{}", encoding="utf-8")
            with patch.object(agent_config, "key_status", return_value={"source": "saved", "hint": "sk-ant-…1234",
                                                                        "saved": True, "environment": False,
                                                                        "saved_hint": "sk-ant-…1234",
                                                                        "encrypt": True}):
                key, preflight, history, start, note = ui_agent.refresh_status(2, "", "prepare", False)
        self.assertEqual(start["interactive"], True)
        self.assertIn("Key saved", key)
        self.assertIn("Claude key", preflight)
        self.assertIn("No runs yet", history)
        self.assertIn("Ready", note)

    def test_loading_fills_the_form_from_the_saved_settings_and_reads_the_panels(self):
        agent_config.save_settings({"model": "claude-haiku-5-5", "mode": "produce", "effort": "high",
                                    "budgets": {"cost_usd": 3}, "permissions": {"downloads": True}})
        loaded = ui_agent.load_everything(0, "")
        form = ui_agent.form_from_settings(agent_config.load_settings())
        self.assertEqual(loaded[:len(form)], form)
        self.assertEqual(len(loaded), len(form) + 2 + 5)
        self.assertEqual(form[0], "claude-haiku-5-5")
        self.assertIn("Fastest and cheapest", loaded[len(form)])
        self.assertIn("Also generates the clips", loaded[len(form) + 1])

    def test_the_history_lists_runs_with_a_shape_for_their_state_and_escaped_names(self):
        store = agent_store.RunStore("20261117-120000-aaaaaa")
        store.create(summary={"config": {"mode": "produce"}, "usage": {"cost_usd": 1.234},
                              "brief": [{"name": "<b>ad</b>.md"}]}, system="", tools=[])
        store.update(state="failed")
        old = agent_store.RunStore("20261116-120000-bbbbbb")
        old.create(summary={"config": {"mode": "prepare"}, "brief": []}, system="", tools=[])
        old.update(state="completed")
        html = ui_agent.history_html()
        self.assertIn("&lt;b&gt;ad&lt;/b&gt;.md", html)
        self.assertNotIn("<b>ad</b>", html)
        self.assertIn("badge err", html)
        self.assertIn("badge ok", html)
        self.assertIn("$1.23", html)
        self.assertLess(html.index("ad&lt;/b&gt;.md"), html.index("completed") if "completed" in html else len(html))

    def test_the_preflight_table_uses_a_badge_shape_per_state(self):
        html = ui_agent.preflight_html("prepare", False)
        self.assertIn("Claude key", html)
        self.assertIn("badge err", html)  # no key saved
        self.assertIn("badge ok", html)


if __name__ == "__main__":
    unittest.main()
