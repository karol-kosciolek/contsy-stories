# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Agent page's API key (encrypted for the Windows account, masked everywhere, redacted from text) and its
settings (checked, with the consent for attached files never saved). Everything lives in a temporary folder."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from app import agent_config, agent_provider as provider

KEY = "sk-" + "ant-" + "api03-" + "Zx9Yw8Vv7Uu6Tt5Ss4Rr3Qq2Pp1"
OTHER = "sk-" + "ant-" + "api03-" + "Mm1Nn2Oo3Pp4Qq5Rr6Ss7Tt8Uu9"


class _Isolated(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        root = Path(self.context.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.context.enter_context(patch.object(agent_config, "CREDENTIALS_FILE", root / "agent_credentials.json"))
        self.context.enter_context(patch.object(agent_config, "SETTINGS_FILE", root / "agent_settings.json"))
        env = {name: value for name, value in os.environ.items() if name != agent_config.ENV_KEY}
        self.context.enter_context(patch.dict(os.environ, env, clear=True))
        agent_config._CACHE.update(stamp=None, key="")
        self.root = root


@unittest.skipUnless(sys.platform == "win32", "the key is encrypted with Windows DPAPI")
class KeyTests(_Isolated):
    def test_a_saved_key_is_encrypted_masked_and_read_back(self):
        self.assertEqual(agent_config.get_key(), ("", "none"))
        self.assertEqual(agent_config.save_key(f"  {KEY}\n"), "")
        text = agent_config.CREDENTIALS_FILE.read_text(encoding="utf-8")
        self.assertNotIn(KEY, text)
        self.assertNotIn(KEY[-12:], text)
        record = json.loads(text)
        self.assertEqual((record["protect"], record["hint"]), ("dpapi-user", "sk-ant-…" + KEY[-4:]))
        self.assertEqual(agent_config.get_key(), (KEY, "saved"))
        status = agent_config.key_status()
        self.assertEqual((status["source"], status["hint"], status["saved"]), ("saved", "sk-ant-…" + KEY[-4:], True))
        self.assertNotIn(KEY, json.dumps(status))
        # a second save replaces it
        agent_config.save_key(OTHER)
        self.assertEqual(agent_config.get_key()[0], OTHER)

    def test_the_encrypted_file_does_not_open_with_other_entropy_or_damage(self):
        agent_config.save_key(KEY)
        record = json.loads(agent_config.CREDENTIALS_FILE.read_text(encoding="utf-8"))
        record["blob"] = record["blob"][:-8] + "AAAAAAAA"
        agent_config.CREDENTIALS_FILE.write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(agent_config.get_key(), ("", "none"))

    def test_clearing_overwrites_and_removes_the_file(self):
        agent_config.save_key(KEY)
        path = agent_config.CREDENTIALS_FILE
        agent_config.clear_key()
        self.assertFalse(path.exists())
        self.assertEqual(agent_config.get_key(), ("", "none"))
        agent_config.clear_key()  # nothing to clear is no error

    def test_the_environment_variable_wins_over_a_saved_key(self):
        agent_config.save_key(KEY)
        with patch.dict(os.environ, {agent_config.ENV_KEY: f" {OTHER} "}):
            self.assertEqual(agent_config.get_key(), (OTHER, "environment"))
            status = agent_config.key_status()
            self.assertEqual((status["source"], status["environment"], status["saved"]),
                             ("environment", True, True))
            self.assertEqual(status["saved_hint"], "sk-ant-…" + KEY[-4:])


class KeyChecksTests(_Isolated):
    def test_a_key_must_look_like_a_claude_key(self):
        for bad in ("", "   ", "hello", "sk-ant-short", "sk-ant-" + "a" * 20 + " b", "sk-proj-" + "a" * 30):
            with self.subTest(key=bad):
                self.assertNotEqual(agent_config.check_key(bad), "")
        self.assertEqual(agent_config.check_key(KEY), "")
        self.assertEqual(agent_config.save_key("nope"), agent_config.check_key("nope"))
        self.assertFalse(agent_config.CREDENTIALS_FILE.exists())

    def test_redact_removes_the_key_and_anything_shaped_like_one(self):
        with patch.dict(os.environ, {agent_config.ENV_KEY: KEY}):
            text = f"Authorization: Bearer {KEY}; and {OTHER}; and sk-ant-short123; plain words"
            clean = agent_config.redact(text)
        self.assertNotIn(KEY, clean)
        self.assertNotIn(OTHER, clean)
        self.assertNotIn("sk-ant-short123", clean)
        self.assertIn("plain words", clean)
        self.assertEqual(agent_config.redact(None), "")
        self.assertEqual(agent_config.redact("a secret-token here", "secret-token"), "a [key removed] here")

    def test_the_hint_never_shows_more_than_the_end(self):
        self.assertEqual(agent_config.key_hint(KEY), "sk-ant-…" + KEY[-4:])
        self.assertEqual(agent_config.key_hint("short"), "")


class SettingsTests(_Isolated):
    def test_defaults_are_safe_and_complete(self):
        settings = agent_config.load_settings()
        self.assertEqual((settings["provider"], settings["model"], settings["effort"], settings["mode"]),
                         ("anthropic", provider.DEFAULT_MODEL, "medium", "prepare"))
        self.assertTrue(settings["send_images"])
        self.assertTrue(settings["keep_awake"])
        self.assertEqual(settings["budgets"]["cost_usd"], 20.0)
        permissions = settings["permissions"]
        self.assertTrue(permissions["write"])
        self.assertFalse(any(permissions[name] for name in ("delete", "uploads", "queue_all", "settings",
                                                            "downloads")))

    def test_bad_values_fall_back_and_numbers_are_kept_in_range(self):
        saved = agent_config.save_settings({"model": "claude-made-up", "effort": "extreme", "mode": "yolo",
                                            "budgets": {"cost_usd": "9999999", "steps": -4, "hours": "abc",
                                                        "gpu_minutes_prepare": float("nan")},  # old keys: ignored
                                            "permissions": {"delete": 1, "surprise": True}})
        self.assertEqual((saved["model"], saved["effort"], saved["mode"]), (provider.DEFAULT_MODEL, "medium", "prepare"))
        self.assertEqual(saved["budgets"]["cost_usd"], 1000.0)
        self.assertEqual(saved["budgets"], {"cost_usd": 1000.0})  # the cost is the only limit
        self.assertEqual(agent_config.clean_budgets({"cost_usd": "abc"}), {"cost_usd": 20.0})
        self.assertEqual(set(saved["permissions"]), set(agent_config.PERMISSIONS))
        self.assertTrue(saved["permissions"]["delete"])
        self.assertEqual(agent_config.load_settings(), saved)

    def test_the_consent_for_attached_files_is_never_saved(self):
        saved = agent_config.save_settings({"permissions": {"uploads": True, "delete": True, "queue_all": True}})
        self.assertFalse(saved["permissions"]["uploads"])
        self.assertFalse(json.loads(agent_config.SETTINGS_FILE.read_text(encoding="utf-8"))["permissions"]["uploads"])
        self.assertTrue(agent_config.load_settings()["permissions"]["delete"])
        # a run can still be started with it: the cleaned settings of one run keep it
        self.assertTrue(agent_config.clean_settings({"permissions": {"uploads": True}})["permissions"]["uploads"])
        self.assertTrue(agent_config.allowed({"permissions": {"uploads": True}}, "uploads"))
        self.assertFalse(agent_config.allowed({}, "write"))

    def test_a_damaged_settings_file_gives_the_defaults(self):
        agent_config.SETTINGS_FILE.write_text("{not json", encoding="utf-8")
        self.assertEqual(agent_config.load_settings(), agent_config.clean_settings({}))
        agent_config.SETTINGS_FILE.write_text("[1, 2]", encoding="utf-8")
        self.assertEqual(agent_config.load_settings(), agent_config.clean_settings({}))


if __name__ == "__main__":
    unittest.main()
