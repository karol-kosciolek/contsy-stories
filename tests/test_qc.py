# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Face identity checks: the verdict from measured similarities, checks that do no work when there
is nothing to compare and video sampling (no GPU)."""
import unittest
from unittest.mock import patch

from app import qc

MIA = {"slug": "mia", "name": "Mia"}
WEAK = "different face (similarity 0.30 < 0.40)"


class JudgeTests(unittest.TestCase):
    def test_face_below_threshold_is_another_person(self):
        verdict = qc.judge(0.35)
        self.assertFalse(verdict.ok)
        self.assertIn("different face", verdict.summary())
        self.assertTrue(qc.judge(qc.FACE_MIN).ok)
        self.assertTrue(qc.judge(None).ok)  # no face was measured

    def test_record_holds_the_identity_fields_only(self):
        self.assertEqual(qc.Verdict(True).record(), {"ok": True, "reasons": [], "face": None})
        record = qc.Verdict(False, [WEAK], face=0.30049, faces={"Mia": 0.30049, "Kuba": None}).record()
        self.assertEqual(record, {"ok": False, "reasons": [WEAK], "face": 0.3,
                                  "faces": {"Mia": 0.3, "Kuba": None}})


class NothingToCompareTests(unittest.TestCase):
    """Quality off or no character: no model, no file and no frame is touched."""

    def setUp(self):
        untouched = AssertionError("Nothing should be measured")
        self.patches = [patch.object(qc.model_manager, "ensure", side_effect=untouched),
                        patch.object(qc, "face_score", side_effect=untouched),
                        patch.object(qc, "face_scores", side_effect=untouched)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_a_picture_is_not_opened(self):
        for character, quality in ((None, True), (None, False), ([], True), (MIA, False), ([MIA, MIA], False)):
            with self.subTest(character=character, quality=quality):
                verdict = qc.check("missing.png", character, quality)
                self.assertEqual(verdict.record(), {"ok": True, "reasons": [], "face": None})

    def test_the_checks_never_reach_the_assistant(self):
        # Face identity runs on the CPU: loading the vision model here would swap models mid-generation.
        self.assertFalse(hasattr(qc, "llm"))


class PictureTests(unittest.TestCase):
    def check(self, score):
        with patch.object(qc.model_manager, "ensure", return_value=None), \
                patch.object(qc, "face_score", return_value=score) as measured:
            verdict = qc.check("portrait.png", MIA)
        measured.assert_called_once_with("portrait.png", MIA)
        return verdict

    def test_one_character_is_scored_on_its_face(self):
        self.assertEqual(self.check(0.62).record(), {"ok": True, "reasons": [], "face": 0.62})
        weak = self.check(0.3)
        self.assertEqual((weak.ok, weak.reasons, weak.face), (False, [WEAK], 0.3))
        self.assertTrue(self.check(None).ok)  # no face on the picture

    def test_a_missing_face_model_leaves_the_face_unmeasured(self):
        with patch.object(qc.model_manager, "ensure", return_value="The face model is not installed."), \
                patch.object(qc, "face_score", side_effect=AssertionError("No model")):
            verdict = qc.check("portrait.png", MIA)
        self.assertEqual(verdict.record(), {"ok": True, "reasons": [], "face": None})


if __name__ == "__main__":
    unittest.main()
