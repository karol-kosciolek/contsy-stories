# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Claude SDK behind the optional Agent page: pinned in requirements.txt, listed in the notices, and the only
module that imports it is app/agent_claude.py (so nothing else in the app needs the package or the network)."""
from importlib import metadata
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent.parent
PINS = {"anthropic": "2.10.0", "jiter": "0.19.0", "sniffio": "1.3.1"}


class AgentDependencyTests(unittest.TestCase):
    def test_the_sdk_and_its_two_additions_are_pinned_and_installed_at_the_pinned_versions(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for name, version in PINS.items():
            with self.subTest(package=name):
                self.assertRegex(requirements, rf"(?m)^{name}=={re.escape(version)}\s*$")
                self.assertEqual(metadata.version(name), version)

    def test_the_notices_name_the_packages_and_their_licenses(self):
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        row = next(line for line in notices.splitlines() if line.startswith("| anthropic "))
        for text in ("anthropic 2.10.0", "jiter 0.19.0", "sniffio 1.3.1", "Apache-2.0", "MIT"):
            self.assertIn(text, row)
        self.assertIn("optional Agent page", row)

    def test_only_the_claude_provider_imports_the_sdk(self):
        pattern = re.compile(r"(?m)^\s*(?:import anthropic|from anthropic\b)")
        offenders = [path.name for path in sorted((ROOT / "app").glob("*.py"))
                     if path.name != "agent_claude.py" and pattern.search(path.read_text(encoding="utf-8"))]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
