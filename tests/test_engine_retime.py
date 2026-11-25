# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Frame retiming puts every output frame on the exact output clock (Wan 16 -> 30, lip sync 25 -> 30)."""
import unittest

from engine.media.retime import frame_rate_plan


class FrameRatePlanTests(unittest.TestCase):
    def test_wan_uses_exact_thirtieths_without_dropping_intermediate_frames(self):
        tasks, layout = frame_rate_plan(81, 16, 30)
        self.assertEqual(len(layout), 151)  # 5 seconds plus its endpoint
        by_pair = {}
        for pair, fraction in tasks:
            by_pair.setdefault(pair, []).append(fraction)
        for index, (pair, slot) in enumerate(layout):
            fraction = 0.0 if slot < 0 else by_pair[pair][slot]
            self.assertAlmostEqual((pair + fraction) / 16, index / 30)

    def test_integer_ratio_reproduces_whole_multiplier_timestamps(self):
        tasks, layout = frame_rate_plan(17, 16, 48)
        self.assertEqual(tasks, [(i, t / 3) for i in range(16) for t in (1, 2)])
        self.assertEqual(len(layout), 49)

    def test_lipsync_keeps_the_last_source_frame_and_never_extrapolates(self):
        tasks, layout = frame_rate_plan(25, 25, 30)
        self.assertEqual(len(layout), 30)
        self.assertEqual(layout[-1], (24, -1))
        self.assertTrue(all(0 < fraction < 1 for _, fraction in tasks))
        self.assertTrue(all(pair < 24 for pair, _ in tasks))

    def test_single_frame_and_invalid_input(self):
        self.assertEqual(frame_rate_plan(1, 25, 30), ([], [(0, -1)]))
        for bad in ((0, 25, 30), (5, 0, 30), (5, 25, -1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                frame_rate_plan(*bad)


if __name__ == "__main__":
    unittest.main()
