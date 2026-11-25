# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""What the Agent page checks before Start: the key, the queue, the models a mode needs and the disk. The model
manager, the queue and the disk are replaced: nothing is read from the real state."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app import agent_preflight, agent_store, queue_manager

NAMES = {"z-image-turbo": "Z-Image-Turbo", "wan22-a14b-i2v": "Wan 2.2 A14B image to video"}


def entry(model_id, size):
    """A stand-in for model_manager.missing_bytes: every model is installed except those in `size`."""
    return (size.get(model_id, 0), size.get(model_id, 0))


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.missing = {}
        self.jobs = []
        self.free = 500.0
        manager, queue = agent_preflight.model_manager, agent_preflight.queue_manager
        for patcher in (
                patch.object(manager, "tiered", side_effect=lambda model_id: model_id),
                patch.object(manager, "missing_bytes", side_effect=lambda ids: entry(ids[0], self.missing)),
                patch.object(agent_preflight.config, "get_model",
                             side_effect=lambda model_id: {"name": NAMES.get(model_id, model_id)}),
                patch.object(queue, "snapshot", side_effect=lambda: list(self.jobs)),
                patch.object(queue, "paused", return_value=False),
                patch.object(agent_preflight.agent_store, "free_gb", side_effect=lambda: self.free)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def by_name(self, mode="prepare", key=None, **options):
        key = key or {"source": "saved", "hint": "sk-ant-…a1b2"}
        return {check.name: check for check in agent_preflight.checks(mode, key=key, **options)}

    def test_each_mode_lists_the_models_it_uses(self):
        self.assertIn("z-image-turbo", agent_preflight.models_for("prepare"))
        self.assertNotIn("wan22-a14b-i2v", agent_preflight.models_for("prepare"))
        self.assertIn("wan22-a14b-i2v", agent_preflight.models_for("produce"))
        self.assertEqual(agent_preflight.models_for("anything"), agent_preflight.PREPARE_MODELS)
        self.assertEqual(len(agent_preflight.model_rows("produce")), len(agent_preflight.PRODUCE_MODELS))

    def test_everything_installed_and_idle_is_all_ready(self):
        checks = self.by_name("produce")
        self.assertEqual({name: check.state for name, check in checks.items()},
                         {"Claude key": "ok", "GPU queue": "ok", "Models": "ok", "Disk space": "ok"})
        self.assertIn("a1b2", checks["Claude key"].detail)
        self.assertNotIn("sk-ant-…a1b2", checks["Claude key"].detail)  # only the end of the hint is repeated

    def test_the_key_states(self):
        self.assertEqual(self.by_name(key={"source": "none"})["Claude key"].state, "err")
        environment = self.by_name(key={"source": "environment", "hint": "sk-ant-…zz99"})["Claude key"]
        self.assertEqual(environment.state, "ok")
        self.assertIn("ANTHROPIC_API_KEY", environment.detail)

    def test_the_queue_states(self):
        self.jobs = [SimpleNamespace(status=queue_manager.RUNNING), SimpleNamespace(status=queue_manager.PENDING),
                     SimpleNamespace(status=queue_manager.DONE)]
        busy = self.by_name()["GPU queue"]
        self.assertEqual(busy.state, "info")
        self.assertIn("1 running, 1 waiting", busy.detail)
        with patch.object(agent_preflight.queue_manager, "paused", return_value=True):
            self.assertEqual(self.by_name()["GPU queue"].state, "warn")
        with patch.object(agent_preflight.queue_manager, "snapshot", side_effect=RuntimeError("boom")):
            self.assertEqual(self.by_name()["GPU queue"].state, "warn")

    def test_missing_models_are_named_and_an_error_unless_downloads_are_allowed(self):
        self.missing = {"z-image-turbo": 20_000_000_000, "wan22-a14b-i2v": 37_000_000_000}
        prepare = self.by_name("prepare")["Models"]
        self.assertEqual(prepare.state, "err")
        self.assertIn("Z-Image-Turbo (20.0 GB)", prepare.detail)
        self.assertNotIn("Wan 2.2", prepare.detail)  # Prepare does not use it
        produce = self.by_name("produce")["Models"]
        self.assertIn("Wan 2.2 A14B image to video", produce.detail)
        self.assertIn("57.0 GB", produce.detail)
        allowed = self.by_name("produce", downloads_allowed=True)["Models"]
        self.assertEqual(allowed.state, "warn")
        self.assertIn("may download", allowed.detail)

    def test_only_producing_waits_for_a_missing_model_and_only_without_the_permission(self):
        self.missing = {"wan22-a14b-i2v": 37_000_000_000}
        self.assertEqual(agent_preflight.blocking_models("prepare", False), "")
        self.assertEqual(agent_preflight.blocking_models("produce", True), "")
        reason = agent_preflight.blocking_models("produce", False)
        self.assertIn("Wan 2.2 A14B image to video", reason)
        self.assertIn("use Prepare", reason)
        self.missing = {}
        self.assertEqual(agent_preflight.blocking_models("produce", False), "")

    def test_disk_space_levels(self):
        for free, state in ((500, "ok"), (12, "warn"), (3, "err")):
            with self.subTest(free=free):
                self.free = free
                self.assertEqual(self.by_name()["Disk space"].state, state)
        with patch.object(agent_store, "free_gb", side_effect=OSError("no")):
            self.assertEqual(self.by_name()["Disk space"].state, "warn")


if __name__ == "__main__":
    unittest.main()
