# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""A model file verified once is not hashed again after a restart unless it changed (model_manager.VERIFIED_FILE)."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import model_manager

DATA = b"model weights " * 1000


class VerifiedAcrossRestartsTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.store = self.root / "config" / "verified_models.json"
        self.model = self.root / "model.bin"
        self.model.write_bytes(DATA)
        self.spec = {"size_bytes": len(DATA), "sha256": hashlib.sha256(DATA).hexdigest()}
        for patcher in (patch.object(model_manager, "VERIFIED_FILE", self.store),
                        patch.object(model_manager, "VERIFY_CACHE_BYTES", 1024),
                        patch.object(model_manager, "_VERIFIED", {}),
                        patch.object(model_manager, "_STORED", None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.folder.cleanup)
        self.hashed = 0
        real = hashlib.file_digest

        def counting(stream, digest):
            self.hashed += 1
            return real(stream, digest)

        hashing = patch.object(model_manager.hashlib, "file_digest", counting)
        hashing.start()
        self.addCleanup(hashing.stop)

    def restart(self):
        """What a new process sees: nothing in memory, the store on disk."""
        model_manager._VERIFIED.clear()
        model_manager._STORED = None

    def test_a_verified_file_is_not_hashed_again_after_a_restart(self):
        self.assertTrue(model_manager._verified_file(self.model, self.spec))
        self.assertEqual(self.hashed, 1)
        stored = json.loads(self.store.read_text(encoding="utf-8"))
        self.assertEqual(stored["version"], 1)
        self.assertEqual(stored["files"][str(self.model.resolve())]["sha256"], self.spec["sha256"])
        self.restart()
        self.assertTrue(model_manager._verified_file(self.model, self.spec))
        self.assertEqual(self.hashed, 1)

    def test_a_changed_time_means_a_new_hash(self):
        self.assertTrue(model_manager._verified_file(self.model, self.spec))
        self.restart()
        stat = self.model.stat()
        os.utime(self.model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertTrue(model_manager._verified_file(self.model, self.spec))
        self.assertEqual(self.hashed, 2)

    def test_replaced_bytes_with_the_old_times_are_hashed_and_refused(self):
        self.assertTrue(model_manager._verified_file(self.model, self.spec))
        stat = self.model.stat()
        replacement = self.root / "other.bin"
        replacement.write_bytes(DATA[:-1] + b"X")
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        os.replace(replacement, self.model)
        self.restart()
        self.assertFalse(model_manager._verified_file(self.model, self.spec))
        self.assertEqual(self.hashed, 2)

    def test_another_expected_hash_is_checked_again(self):
        self.assertTrue(model_manager._verified_file(self.model, self.spec))
        self.restart()
        self.assertFalse(model_manager._verified_file(self.model, dict(self.spec, sha256="0" * 64)))
        self.assertEqual(self.hashed, 2)

    def test_a_failed_check_is_not_stored(self):
        self.assertFalse(model_manager._verified_file(self.model, dict(self.spec, sha256="0" * 64)))
        self.assertFalse(self.store.exists())

    def test_small_files_are_always_hashed_and_never_stored(self):
        with patch.object(model_manager, "VERIFY_CACHE_BYTES", len(DATA) + 1):
            self.assertTrue(model_manager._verified_file(self.model, self.spec))
            self.assertTrue(model_manager._verified_file(self.model, self.spec))
        self.assertEqual(self.hashed, 2)
        self.assertFalse(self.store.exists())

    def test_a_damaged_store_counts_as_empty(self):
        self.store.parent.mkdir(parents=True)
        self.store.write_text("{not json", encoding="utf-8")
        self.assertTrue(model_manager._verified_file(self.model, self.spec))
        self.assertEqual(self.hashed, 1)
        self.assertEqual(json.loads(self.store.read_text(encoding="utf-8"))["version"], 1)

    def test_a_store_that_cannot_be_written_only_loses_the_memory(self):
        with patch.object(model_manager, "atomic_text", side_effect=OSError("read-only")):
            self.assertTrue(model_manager._verified_file(self.model, self.spec))
        self.restart()
        self.assertTrue(model_manager._verified_file(self.model, self.spec))
        self.assertEqual(self.hashed, 2)


class LiveStoreDefaultTest(unittest.TestCase):
    def test_the_store_is_in_the_state_config_folder(self):
        from app import config

        self.assertEqual(model_manager.VERIFIED_FILE.parent, config.STATE_CONFIG_DIR)


if __name__ == "__main__":
    unittest.main()
