# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Spoken lines, effects and levels shared by the Timeline project, sound preparation and editor."""
import json
import unittest

from app import scene_parts as parts


class CueTests(unittest.TestCase):
    def test_normalizes_text_and_rounds_start(self):
        cue = parts.Cue("  Did you   lock it? ", "rose", 1.234)
        self.assertEqual((cue.text, cue.speaker, cue.start), ("Did you lock it?", "rose", 1.23))
        self.assertEqual(cue.character, "rose")
        self.assertEqual(cue.voice, "")

    def test_voice_speaker_and_json_round_trip(self):
        cue = parts.Cue("Hello", "@hanna")
        self.assertEqual((cue.voice, cue.character), ("hanna", ""))
        self.assertEqual(cue.to_json(), {"text": "Hello", "speaker": "@hanna"})
        self.assertEqual(parts.Cue.from_json(cue.to_json()), cue)
        timed = parts.Cue("Later", "", 2.5)
        self.assertEqual(parts.Cue.from_json(json.loads(json.dumps(timed.to_json()))), timed)

    def test_rejects_invalid_values(self):
        for bad in (dict(text=3), dict(text="Hi", speaker="Rose Ann"), dict(text="Hi", start=-1),
                    dict(text="Hi", start=float("nan")), dict(text="Hi", start=True), dict(text="x" * 301)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parts.Cue(**{"text": "", **bad})
        with self.assertRaises(ValueError):
            parts.Cue.from_json({"text": "Hi", "voice": "x"})


class EffectTests(unittest.TestCase):
    def test_defaults_and_limits(self):
        effect = parts.Effect("door slam", 3)
        self.assertEqual((effect.at, effect.seconds), (3.0, parts.EFFECT_SECONDS))
        self.assertEqual(effect.to_json(), {"text": "door slam", "at": 3.0})
        self.assertEqual(parts.Effect.from_json({"text": "thunder", "at": 1, "seconds": 4}).seconds, 4.0)
        with self.assertRaises(ValueError):
            parts.Effect("thunder", 1, 20)
        with self.assertRaises(ValueError):
            parts.Effect("thunder", -0.5)


class TimingTests(unittest.TestCase):
    def test_automatic_lines_follow_each_other(self):
        cues = (parts.Cue("One two three"), parts.Cue("Four five"))
        windows = parts.timings(cues, durations=[1.0, 2.0])
        self.assertEqual(windows, [(0.3, 1.3), (1.8, 3.8)])

    def test_explicit_start_wins_and_next_line_follows_it(self):
        cues = (parts.Cue("A", start=2.0), parts.Cue("B"))
        self.assertEqual(parts.timings(cues, [1.5, 1.0]), [(2.0, 3.5), (4.0, 5.0)])

    def test_an_overlap_with_any_earlier_line_is_found_and_automatic_lines_wait_for_it(self):
        long_line, short, late = parts.Cue("A", start=0.5), parts.Cue("B", start=1.0), parts.Cue("C", start=2.0)
        windows = parts.timings((long_line, short, late), [3.0, 0.5, 1.0])
        self.assertEqual(windows, [(0.5, 3.5), (1.0, 1.5), (2.0, 3.0)])
        issues = parts.timing_issues((long_line, short, late), (), 6, [3.0, 0.5, 1.0])
        self.assertEqual([(issue.text, issue.line) for issue in issues],
                         [("Line 2 starts before line 1 ends.", 1), ("Line 3 starts before line 1 ends.", 2)])
        # An automatic line follows the line that ends last, not only the one before it.
        automatic = parts.timings((long_line, short, parts.Cue("D")), [3.0, 0.5, 1.0])
        self.assertEqual(automatic[2], (4.0, 5.0))
        self.assertEqual([issue.line for issue in parts.timing_issues((long_line, short, parts.Cue("D")), (), 6,
                                                                       [3.0, 0.5, 1.0])], [1])

    def test_estimate_grows_with_words(self):
        self.assertEqual(parts.estimate_seconds(""), parts.MIN_LINE_SECONDS)
        self.assertGreater(parts.estimate_seconds("one two three four five six seven eight nine ten"), 3.5)

    def test_issues_for_lines_past_the_scene_and_overlaps(self):
        cues = (parts.Cue("A", start=1.0), parts.Cue("B", start=1.5), parts.Cue("C", start=5.5))
        effects = (parts.Effect("thud", 5.0), parts.Effect("bell", 6.0))
        estimated = parts.timing_issues(cues, effects, 6)
        self.assertEqual([(i.level, i.line, i.effect) for i in estimated],
                         [("warning", 1, None), ("warning", 2, None), ("info", None, 0), ("warning", None, 1)])
        measured = parts.timing_issues(cues, (), 6, durations=[0.4, 0.4, 0.9])
        self.assertEqual([(i.level, i.line) for i in measured], [("warning", 2)])
        # nothing past the scene end stops it: the scene keeps the length chosen for it, even one second
        self.assertEqual(measured[0].text, "Line 3 ends at 6.4 s, after the scene ends at 6 s, so its end is cut off. "
                                           "Make the scene longer to hear all of it.")
        late = parts.timing_issues((parts.Cue("A"), parts.Cue("B")), (parts.Effect("bell", 1.0),), 1, [2.0, 1.0])
        self.assertEqual([(i.level, i.text) for i in late], [
            (parts.WARNING, "Line 1 ends at 2.3 s, after the scene ends at 1 s, so its end is cut off. Make the scene "
                            "longer to hear all of it."),
            (parts.WARNING, "Line 2 starts at 2.8 s, after the scene ends at 1 s, so it is not heard. Make the scene "
                            "longer to hear it."),
            (parts.WARNING, "Effect 1 is at 1.0 s, after the scene ends at 1 s, so it is not heard.")])
        self.assertFalse([i for i in late if i.level == parts.ERROR])

    def test_check_blocks_empty_text_but_never_the_scene_length(self):
        parts.check((parts.Cue("Hi", start=4.9),), (parts.Effect("thud", 1),), 5)
        # a line or an effect after the scene end is only not heard (timing_issues warns): any length can be saved
        parts.check((parts.Cue("Hi", start=5),), (parts.Effect("thud", 7),), 5)
        parts.check((parts.Cue("Hi", start=3), parts.Cue("Bye")), (parts.Effect("thud", 2),), 1)
        with self.assertRaisesRegex(ValueError, "Line 1: write"):
            parts.check((parts.Cue(""),), (), 5)
        with self.assertRaisesRegex(ValueError, "Effect 1: describe"):
            parts.check((), (parts.Effect("", 1),), 5)
        self.assertEqual(parts.heard((parts.Effect("a", 0.5), parts.Effect("b", 1.0), parts.Effect("c", 3.0)), 1),
                         (parts.Effect("a", 0.5),))


class EditorJsonTests(unittest.TestCase):
    def test_blank_rows_are_placeholders(self):
        text = json.dumps([{"text": "Hi", "speaker": "rose", "start": 1}, {"text": "  "}])
        self.assertEqual(parts.cues_from_json(text), (parts.Cue("Hi", "rose", 1.0),))
        self.assertEqual(parts.effects_from_json(json.dumps([{"text": "", "at": 0}])), ())
        self.assertEqual(parts.cues_from_json(""), ())
        with self.assertRaises(ValueError):
            parts.cues_from_json("{not json")
        cues = (parts.Cue("Hi", "rose", 1.0),)
        self.assertEqual(parts.cues_from_json(json.dumps([cue.to_json() for cue in cues])), cues)


class LevelTests(unittest.TestCase):
    def test_levels_round(self):
        self.assertEqual(parts.level(-3.26), -3.5)
        self.assertEqual(parts.level(-0.1), 0.0)
        for bad in (-30, 13, float("inf"), True, "3"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parts.level(bad)

    def test_reference_names(self):
        self.assertTrue(parts.is_reference("a" * 64))
        self.assertFalse(parts.is_reference("../" + "a" * 61))


if __name__ == "__main__":
    unittest.main()
