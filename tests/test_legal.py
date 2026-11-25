# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""License notices stay consistent: GPL-3.0-only for own code, SPDX headers, legal notices in the UI, every
download source named in MODEL_LICENSES.md, and no removed engine or model named where it is not history."""
import html
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from app import config, ui_settings

ROOT = Path(__file__).resolve().parents[1]
# Third-party files kept unchanged under their own license: path -> (SPDX license, the original notice they keep).
VENDORED = {ROOT / "engine" / "kernels" / "sage" / name: ("Apache-2.0", "Copyright (c) 2024 by SageAttention team.")
            for name in ("attn_qk_int8_per_block.py", "quant_per_block.py")}
OWN_CODE = [path for path in (*ROOT.glob("app/*.py"), *ROOT.glob("engine/**/*.py"), *ROOT.glob("tests/*.py"),
                              *ROOT.glob("tools/**/*.py"), *ROOT.glob("*.bat")) if path not in VENDORED]
DOCS = [*(ROOT / name for name in ("README.md", "CONTRIBUTING.md", "SECURITY.md", "MODEL_LICENSES.md",
                                    "THIRD_PARTY_NOTICES.md", "NOTICE")), *ROOT.glob("docs/*.md")]


class LicenseFileTests(unittest.TestCase):
    def test_license_is_the_unmodified_gpl_text(self):
        text = (ROOT / "LICENSE").read_text("utf-8")
        self.assertTrue(text.lstrip().startswith("GNU GENERAL PUBLIC LICENSE"))
        self.assertIn("Version 3, 29 June 2007", text)
        self.assertEqual(text, (ROOT / "LICENSES" / "GPL-3.0-only.txt").read_text("utf-8"))
        self.assertEqual(app.__license__, "GPL-3.0-only")

    def test_own_files_carry_spdx_headers(self):
        self.assertTrue(OWN_CODE)
        for path in OWN_CODE:
            head = path.read_text("utf-8", errors="replace")[:600]
            with self.subTest(path=path.name):
                self.assertIn("SPDX-License-Identifier: GPL-3.0-only", head)
                self.assertIn("SPDX-FileCopyrightText:", head)

    def test_vendored_files_keep_their_license_and_notice_and_are_listed(self):
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text("utf-8")
        reuse = (ROOT / "REUSE.toml").read_text("utf-8")
        for path, (license_id, original) in VENDORED.items():
            relative = path.relative_to(ROOT).as_posix()
            with self.subTest(path=relative):
                text = path.read_text("utf-8")
                self.assertIn(f"SPDX-License-Identifier: {license_id}", text[:600])
                self.assertNotIn("GPL", text)
                self.assertIn(original, text)
                self.assertIn((ROOT / "LICENSES" / f"{license_id}.txt").relative_to(ROOT).as_posix(), text[:600])
                self.assertIn(f"`{relative}`", notices)
                self.assertIn(f'"{relative}"', reuse)

    def test_docs_use_the_current_name_and_license(self):
        for path in DOCS:
            text = path.read_text("utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("Contsy Media Studio", text)
                if path.name != "LICENSING.md":  # explains why earlier licenses were replaced
                    self.assertNotIn("Elastic", text)
                    self.assertNotIn("AGPL", text)


class AboutCardTests(unittest.TestCase):
    def about(self, url):
        with patch.dict(config.SETTINGS["app"], {"source_url": url}):
            return ui_settings.about_html()

    def test_shows_appropriate_legal_notices(self):
        text = self.about("")
        for needed in ("Contsy", app.__version__, "© 2026 Karol Kosciolek", "(GPL-3.0-only)",
                       "no warranty", "LICENSE", "THIRD_PARTY_NOTICES.md", "EzAudio-XL (MIT)",
                       "FLAN-T5-XL (Apache-2.0)", "MuseTalk 1.5 (MIT)"):
            with self.subTest(needed=needed):
                self.assertIn(needed, text)
        self.assertNotIn("Affero", text)
        self.assertIn(html.escape(str(config.ROOT)), text)  # without a URL: the local source folder

    def test_links_the_configured_source_and_escapes_it(self):
        text = self.about("https://example.org/contsy?v=1&x=2")
        self.assertIn("href='https://example.org/contsy?v=1&amp;x=2'", text)
        self.assertNotIn(html.escape(str(config.ROOT)), text)


def _downloads():
    """(model id, download) for everything the installer fetches for the catalog, including the official files
    an install step makes a catalog file from."""
    from app import model_manager

    return [(model["id"], source) for model in config.CATALOG["models"] for spec in model["files"]
            for source in model_manager.sources(spec, model)]


class ModelSourcesTests(unittest.TestCase):
    def test_every_download_source_is_listed_with_its_own_host(self):
        from urllib.parse import urlsplit

        text = ui_settings.model_licenses_html()
        for model_id, source in _downloads():
            with self.subTest(model=model_id, file=source["filename"]):
                if source.get("url"):
                    self.assertIn(f"href='{html.escape(source['url'], quote=True)}'", text)
                    self.assertIn(f"{urlsplit(source['url']).netloc}: {Path(source['filename']).name}", text)
                else:
                    self.assertIn(f"href='https://huggingface.co/{source['repo_id']}'", text)
        self.assertIn("openaipublic.azureedge.net: tiny.pt", text)
        self.assertIn("drive.usercontent.google.com: RIFE_trained_model_v4.9.2.zip", text)

    def test_model_licenses_names_every_repository_and_host_files_come_from(self):
        """MODEL_LICENSES.md must list exactly what is downloaded: each Hugging Face repository, and for a plain
        URL its host (and the GitHub project for a GitHub address)."""
        from urllib.parse import urlsplit

        text = (ROOT / "MODEL_LICENSES.md").read_text(encoding="utf-8")
        aux = json.loads((ROOT / "config" / "auxiliary-models.json").read_text(encoding="utf-8"))
        specs = [(model_id, source.get("repo_id"), source.get("url")) for model_id, source in _downloads()]
        specs += [(key, entry.get("repo_id"), entry.get("url")) for key, entry in aux.items()]
        for owner, repo, url in specs:
            with self.subTest(model=owner, source=url or repo):
                if url:
                    parts = urlsplit(url)
                    github = parts.netloc in ("github.com", "raw.githubusercontent.com")
                    self.assertIn(f"https://github.com/{'/'.join(parts.path.split('/')[1:3])}" if github
                                  else parts.netloc, text)
                else:
                    self.assertIn(f"https://huggingface.co/{repo}", text)


# Engines and models that Contsy no longer uses. Code, tests, tools, configuration and installer files must not
# name them; user documents may only in a line that says what replaced them or which files a user may delete.
# The first alternative is the former node-graph backend, its organization and its package paths; "\x6f" is the
# letter o, so that this guard does not spell the name it forbids.
REMOVED = re.compile(r"c\x6fmfy(?:ui|[-_ ]?org|[_/])|seedvr|moss[-_ ]?sound|(?-i:MOSS)|turbowan|flashvsr|meanaudio"
                     r"|stable[ _-]?audio(?![ _-]?tools)|latentsync", re.I)
USER_DATA = {"queue.json", "user_prefs.json", "settings.local.json"}
# Kept on purpose: the Apache-2.0 origin notice of two modules MuseTalk reuses.
KEPT_IN_CODE = {"engine/models/whisper.py": "LatentSync", "engine/models/sd_unet.py": "LatentSync"}
HISTORY_WORDS = re.compile(r"replaced|removed|delete|first written|no longer|earlier|before \d", re.I)


class RemovedNamesTests(unittest.TestCase):
    def test_code_tests_tools_and_installer_files_do_not_name_removed_engines_or_models(self):
        files = [*ROOT.glob("app/**/*.py"), *ROOT.glob("engine/**/*.py"), *ROOT.glob("tools/**/*.py"),
                 *ROOT.glob("tests/*.py"), *ROOT.glob("*.bat"), ROOT / "requirements.txt",
                 *(path for path in ROOT.glob("config/*.json") if path.name not in USER_DATA | {"models.json"})]
        for path in files:
            if path == Path(__file__).resolve():
                continue
            relative = path.relative_to(ROOT).as_posix()
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                found = REMOVED.search(line)
                if found and KEPT_IN_CODE.get(relative) != found.group(0):
                    self.fail(f"{relative}:{number} names {found.group(0)!r}: {line.strip()[:120]}")

    def test_the_catalog_names_no_removed_project(self):
        """Every file comes from its authors or a named converter: no download source, name, description, license
        or destination of the catalog names a removed engine or model, not even as an old folder."""
        catalog = json.loads((ROOT / "config" / "models.json").read_text(encoding="utf-8"))

        def strings(value, key=""):
            if isinstance(value, dict):
                for name, item in value.items():
                    yield from strings(item, name)
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item, key)
            elif isinstance(value, str):
                yield key, value

        for key, value in strings(catalog):
            self.assertIsNone(REMOVED.search(value), f"{key}: {value}")

    def test_no_file_of_the_repository_names_the_former_node_graph_backend(self):
        """Not even as history, in an internal document or as a download source."""
        import os
        import subprocess

        backend = re.compile("c\x6fmfy", re.I)
        try:  # the files git tracks or would add (a checkout); a copy without git is walked instead
            listed = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                    cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8").split("\0")
            files = [name for name in listed if name]
        except (OSError, subprocess.CalledProcessError):
            # downloaded weights, the Python environment and user data are not part of the repository
            skipped = {".git", ".worktrees", ".claude", "venv", "models", "outputs", "history", "characters",
                       "voices", "timeline_projects", "tmp", "queue_files", "__pycache__", ".ruff_cache", "temp"}
            files = []
            for folder, dirs, names in os.walk(ROOT):
                dirs[:] = [name for name in dirs if name not in skipped]
                files += [(Path(folder) / name).relative_to(ROOT).as_posix() for name in names]
        self.assertIn("app/model_manager.py", files)
        found = []
        for relative in files:
            path = ROOT / relative
            if backend.search(relative):
                found.append(relative)
            if (path.name in USER_DATA or not path.is_file()
                    or path.stat().st_size > 4 << 20 or b"\0" in path.read_bytes()[:8192]):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            found += [f"{relative}:{number}" for number, line in enumerate(text.splitlines(), 1)
                      if backend.search(line)]
        self.assertEqual(found, [])

    def test_user_documents_name_removed_engines_or_models_only_as_history(self):
        docs = [*DOCS]
        catalog_sources = {spec.get("repo_id", model.get("repo_id")) for model in config.CATALOG["models"]
                           for spec in model["files"]}
        catalog_sources |= {"/".join(spec["url"].split("/")[3:5]) for model in config.CATALOG["models"]
                            for spec in model["files"] if spec.get("url")}
        for path in docs:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not REMOVED.search(line) or HISTORY_WORDS.search(line):
                    continue
                with self.subTest(path=path.name, line=number):
                    # A download source the catalog still uses is named where the docs list the sources.
                    self.assertTrue(any(source in line for source in catalog_sources if REMOVED.search(source)),
                                    line.strip()[:160])


if __name__ == "__main__":
    unittest.main()
