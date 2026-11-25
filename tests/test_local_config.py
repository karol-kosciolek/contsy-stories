# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Reject settings that expose the local engine or override protected paths."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import config


class LocalConfigTests(unittest.TestCase):
    def load(self, override):
        defaults = json.loads((config.CONFIG_DIR / 'settings.json').read_text('utf-8'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'settings.json').write_text(json.dumps(defaults), encoding='utf-8')
            (root/'settings.local.json').write_text(json.dumps(override), encoding='utf-8')
            with patch.object(config, 'CONFIG_DIR', root):
                return config.load_settings()

    def test_remote_bind_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'local-only'):
            self.load({'app': {'host': '0.0.0.0'}})

    def test_port_overrides_are_validated(self):
        for value in (True, 80, 65536, '7861'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'port'):
                self.load({'app':{'port':value}})

    def test_source_url_is_empty_or_an_http_address(self):
        self.assertEqual(self.load({})['app']['source_url'], '')
        url = 'https://example.org/contsy/tree/v0.2.0'
        self.assertEqual(self.load({'app':{'source_url':url}})['app']['source_url'], url)
        for value in ('javascript:alert(1)', 'ftp://example.org/src', 'https://a b', "https://x'y", 42):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'source_url'):
                self.load({'app':{'source_url':value}})

    def test_engine_settings_default_and_validate(self):
        self.assertEqual(self.load({})['engine'], {'lora_rounding': 'nearest', 'warm_cache_gib': 'auto',
                                                  'vram_limit_gib': None, 'attention': 'auto',
                                                  'fp8_experts': 'matching'})
        self.assertEqual(self.load({'engine': {'vram_limit_gib': 8}})['engine']['vram_limit_gib'], 8)
        # "sdpa" switches SageAttention off (PyTorch's own attention)
        self.assertEqual(self.load({'engine': {'attention': 'sdpa'}})['engine']['attention'], 'sdpa')
        # the FP8 image-to-video experts: off, or each one that holds its LoRA's distillation (the default)
        self.assertEqual(self.load({'engine': {'fp8_experts': 'off'}})['engine']['fp8_experts'], 'off')
        for value in (0, 36):  # 0 turns the warm cache off, a number of GiB sets it
            self.assertEqual(self.load({'engine': {'warm_cache_gib': value}})['engine']['warm_cache_gib'], value)
        for override in ({'engine': {'vram_limit_gib': 0}}, {'engine': {'warm_cache_gib': -1}},
                         {'engine': {'warm_cache_gib': 'on'}}, {'engine': {'warm_cache_gib': 2.5}},
                         {'engine': {'lora_rounding': 'up'}}, {'engine': {'attention': 'sage'}},
                         {'engine': {'attention': None}}, {'engine': {'fp8_experts': True}},
                         {'engine': {'fp8_experts': 'high'}}, {'engine': {'fp8_experts': 'all'}}, {'engine': 5}):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, 'engine'):
                self.load(override)

    def test_state_directory_defaults_to_project_and_can_be_adopted(self):
        self.assertEqual(self.load({})["paths"]["state_dir"], ".")
        self.assertEqual(self.load({"paths": {"state_dir": "D:/Studio data"}})["paths"]["state_dir"],
                         "D:/Studio data")
        for invalid in ("", 12, None, {}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "paths.state_dir"):
                self.load({"paths": {"state_dir": invalid}})


if __name__ == '__main__':
    unittest.main()
