# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Contsy's documents promised that everything stays on the PC. The optional Agent page sends text to Anthropic once a
key is saved and a run starts, so no document may keep the old promise without saying so."""
from pathlib import Path
import unittest

from app import ui_agent

ROOT = Path(__file__).resolve().parent.parent


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


class DocsHonestyTests(unittest.TestCase):
    def test_the_readme_no_longer_promises_everything_stays_local(self):
        readme = read("README.md")
        for promise in ("100% on your own PC", "runs-100%25%20locally", "Runs 100% locally",
                        "No. Everything runs locally"):
            self.assertNotIn(promise, readme)
        faq = next(line for line in readme.splitlines() if line.startswith("**Do my photos leave my computer?**"))
        for word in ("Not by default", "Agent", "Anthropic", "Send images to the cloud", "never contacts Anthropic"):
            self.assertIn(word, faq)
        self.assertTrue(any(line.startswith("- **Agent**") and "Anthropic" in line for line in readme.splitlines()))

    def test_every_promise_that_generation_never_calls_a_hosted_api_names_the_agent_exception(self):
        for name in ("docs/GUIDE.md", "README.md", "SECURITY.md", "docs/LICENSING.md"):
            for number, line in enumerate(read(name).splitlines(), 1):
                if "never calls a hosted API" in line:
                    with self.subTest(file=name, line=number):
                        self.assertIn("Agent", line)
        guide = read("docs/GUIDE.md")
        self.assertNotIn("Contsy never stores tokens.", guide)
        self.assertIn("Contsy never stores Hugging Face tokens.", guide)
        self.assertIn("## Agent", guide)

    def test_the_security_note_the_settings_text_and_the_package_name_the_key_and_the_cloud(self):
        security = read("SECURITY.md")
        for word in ("Anthropic", "DPAPI", "run folders", "no shell"):
            self.assertIn(word, security)
        self.assertIn("Claude API key of the Agent page", (ROOT / "app" / "ui_settings.py").read_text(encoding="utf-8"))
        init = (ROOT / "app" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("Agent", init)
        self.assertNotIn("(offline)", init)

    def test_the_page_says_exactly_what_leaves_the_pc(self):
        for text in ("What leaves this PC", "Anthropic", "HTTPS", "your brief files", "Send images to the cloud",
                     "uploaded photo", "Nothing is sent until you press Start"):
            self.assertIn(text, ui_agent.PRIVACY)


if __name__ == "__main__":
    unittest.main()
