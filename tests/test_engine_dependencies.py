# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The libraries the own engine imports are pinned (requirements.txt and the pip constraints of install.bat), and the
repository does not hide the engine from git (an unanchored `models/` rule once swallowed engine/models)."""
import importlib.metadata
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class EnginePinTests(unittest.TestCase):
    def test_every_engine_package_is_pinned_exactly_once(self):
        from app import setup_windows

        pins = setup_windows.engine_pins()
        self.assertEqual(sorted(name.split("==")[0] for name in pins), sorted(setup_windows.ENGINE_PACKAGES))
        self.assertTrue(all(len(pin.split("==")) == 2 and pin.split("==")[1][0].isdigit() for pin in pins), pins)

    def test_the_constraints_hold_the_torch_triple_and_the_engine_pins(self):
        from app import setup_windows

        fake = {"torch": "9.0+cu999", "torchvision": "9.1+cu999", "torchaudio": "9.2+cu999"}
        with patch.object(setup_windows.importlib.metadata, "version", side_effect=lambda name: fake[name]):
            lines = setup_windows.constraints_text().splitlines()
        self.assertEqual(lines[:3], ["torch==9.0+cu999", "torchvision==9.1+cu999", "torchaudio==9.2+cu999"])
        self.assertEqual(lines[3:], setup_windows.engine_pins())

    def test_the_installed_versions_are_the_pinned_ones(self):
        from app import setup_windows

        for pin in setup_windows.engine_pins():
            name, wanted = pin.split("==")
            try:
                found = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.skipTest(f"{name} is not installed here")
            with self.subTest(package=name):
                self.assertEqual(found, wanted, f"requirements.txt pins {name}=={wanted}, {found} is installed")


class RepositoryHygieneTests(unittest.TestCase):
    def test_no_unanchored_ignore_rule_hides_a_directory_of_the_engine_or_the_tools(self):
        names = {path.name for top in ("engine", "tests", "tools", "app", "docs") for path in (ROOT / top).rglob("*")
                 if path.is_dir() and path.name != "__pycache__"}
        for number, line in enumerate((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines(), 1):
            rule = line.strip()
            if rule and not rule.startswith(("#", "!", "/")) and rule.endswith("/") and "*" not in rule[:-1]:
                with self.subTest(rule=rule, line=number):
                    self.assertNotIn(rule[:-1], names, f".gitignore line {number} ({rule}) matches a source directory; "
                                                       "anchor it with a leading slash")


if __name__ == "__main__":
    unittest.main()
