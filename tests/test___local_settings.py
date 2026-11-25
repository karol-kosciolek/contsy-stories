# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Loaded first by `unittest discover` (the name sorts before every other test module): the tests use settings.json
alone, so the owner's config/settings.local.json (for example another port or state folder) does not change what they
test. Set CONTSY_SETTINGS_LOCAL yourself to test with another override file."""
import os
import unittest

os.environ.setdefault("CONTSY_SETTINGS_LOCAL", "none")


class LocalSettingsTests(unittest.TestCase):
    def test_the_suite_runs_without_the_owners_override(self):
        from app import config
        self.assertTrue(os.environ.get("CONTSY_SETTINGS_LOCAL"))
        if os.environ["CONTSY_SETTINGS_LOCAL"].strip().lower() == "none":
            self.assertEqual(config.SETTINGS, config.load_settings(None))
