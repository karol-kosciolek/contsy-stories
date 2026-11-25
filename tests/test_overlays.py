# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Exact text over the finished film (app/overlays.py, engine/media/overlays.py): what an overlay may be, how the burner
draws it, and how the Timeline saves it and makes the export again when it changes. Clips are written directly (no
generation, no GPU)."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from app import characters, config, ffmpeg_utils, history, overlays, timeline
from engine.media import overlays as burner
from engine.media import video
from engine.media.errors import MediaError

SIZE = (320, 180)


def _clip(path: Path, seconds: int, gray: int) -> None:
    with video.Writer(path, 30, *SIZE, preset="ultrafast", comment=video.AI_COMMENT) as out:
        for _ in range(30 * seconds):
            out.write(np.full((SIZE[1], SIZE[0], 3), gray, dtype=np.uint8))


def _rgb(frame) -> np.ndarray:
    return frame.to_ndarray(format="rgb24")


def _white(frame, rows=slice(None)) -> int:
    """How many pixels of the frame (in the given rows) are bright: the letters, on clips this dark."""
    return int((_rgb(frame)[rows].min(axis=2) > 200).sum())


class OverlayRules(unittest.TestCase):
    def test_a_text_is_tidied_and_its_times_are_in_tenths(self):
        one = overlays.clean(overlays.Overlay("  Lumio \n\n  lumio.example/offer  ", 1.04, 3.26))
        self.assertEqual((one.text, one.start, one.end), ("Lumio\nlumio.example/offer", 1.0, 3.3))
        self.assertEqual(overlays.clean({"text": "Hi", "last": 2}).last, 2.0)

    def test_what_is_refused(self):
        bad = [overlays.Overlay(""), overlays.Overlay("  \n "), overlays.Overlay("x\ny\nz\nw\nv", 0, 2),
               overlays.Overlay("x" * 201, 0, 2), overlays.Overlay("a\x00b", 0, 2), overlays.Overlay("a​b", 0, 2),
               overlays.Overlay("ok", 2, 2), overlays.Overlay("ok", 2, 2.3), overlays.Overlay("ok", -1, 2),
               overlays.Overlay("ok", 0, 99999), overlays.Overlay("ok", 1, 3, last=2), overlays.Overlay("ok", last=0.1),
               overlays.Overlay("ok", 0, 2, position="left"), overlays.Overlay("ok", 0, 2, size="huge"),
               overlays.Overlay("ok", 0, 2, backdrop="red"), overlays.Overlay("ok", float("nan"), 2),
               overlays.Overlay("ok", True, 2), overlays.Overlay(5, 0, 2), {"text": "ok", "color": "red"}, "text"]
        for row in bad:
            with self.assertRaises(ValueError, msg=repr(row)):
                overlays.clean(row)
        with self.assertRaisesRegex(ValueError, "Overlay 2"):
            overlays.cleaned([overlays.Overlay("ok", 0, 2), overlays.Overlay("")])
        with self.assertRaisesRegex(ValueError, "at most 9"):
            overlays.cleaned([overlays.Overlay("ok", 0, 2)] * 10)

    def test_scripts_that_cannot_be_drawn_correctly_are_refused_and_the_rest_is_kept(self):
        for text in ("שלום", "مرحبا", "สวัสดี", "नमस्ते", "Hello שלום"):
            with self.assertRaisesRegex(ValueError, "cannot be drawn correctly", msg=text):
                overlays.clean(overlays.Overlay(text, 0, 2))
        for text in ("Zażółć gęślą jaźń", "Привет, мир", "Ελληνικά", "日本語のテキスト", "한국어", "Price: 19.99 €"):
            self.assertEqual(overlays.clean(overlays.Overlay(text, 0, 2)).text, text)

    def test_only_cards_clash_with_cards_and_text_over_a_card_is_fine(self):
        card, other = overlays.Overlay("A", 0, 4, backdrop="card"), overlays.Overlay("B", 0, 4, backdrop="card", position="top")
        text = overlays.Overlay("C", 1, 3, position="bottom")
        self.assertEqual(len(overlays.clashes([card, other], 10)), 1)
        self.assertEqual(overlays.clashes([card, text], 10), [])
        self.assertEqual(overlays.clashes([text, card], 10), [])
        self.assertEqual(overlays.clashes([card, overlays.Overlay("D", 4, 6, backdrop="card")], 10), [])

    def test_a_huge_number_is_refused_not_a_crash(self):
        with self.assertRaises(ValueError):
            overlays.clean({"text": "a", "end": 10 ** 400})
        kept = overlays.tolerant([{"text": "a", "end": 10 ** 400}, {"text": "b", "start": 0, "end": 2}])
        self.assertEqual([one.text for one in kept], ["b"])

    def test_the_rows_a_tool_takes_are_what_it_shows_and_numbers_count_from_one(self):
        one = overlays.clean(overlays.Overlay("The end", last=3, backdrop="card"))
        self.assertEqual(overlays.rows([one]), [{"text": "The end", "last_seconds": 3.0, "backdrop": "card"}])
        self.assertEqual(overlays.view([one], 10)[0], {"number": 1, "text": "The end", "last_seconds": 3.0,
                                                       "backdrop": "card", "shown_from_to": [7.0, 10.0]})
        with self.assertRaisesRegex(ValueError, "Overlay 1:"):
            overlays.cleaned([overlays.Overlay("")])

    def test_the_stored_form_holds_only_what_differs_and_comes_back_equal(self):
        one = overlays.clean(overlays.Overlay("The end", last=3, backdrop="card", size="large"))
        self.assertEqual(one.to_json(), {"text": "The end", "last": 3.0, "size": "large", "backdrop": "card"})
        self.assertEqual(overlays.clean(one.to_json()), one)

    def test_a_damaged_stored_row_is_left_out_instead_of_breaking_the_project(self):
        rows = [{"text": "ok", "start": 0, "end": 2}, {"text": ""}, "x", {"text": "ok", "end": "soon"}]
        self.assertEqual([one.text for one in overlays.tolerant(rows)], ["ok"])
        self.assertEqual(overlays.tolerant("nonsense"), ())

    def test_the_key_is_empty_without_overlays_and_follows_every_change(self):
        one, two = overlays.Overlay("A", 0, 2), overlays.Overlay("A", 0, 2.5)
        self.assertEqual(overlays.key(()), "")
        self.assertNotEqual(overlays.key((one,)), overlays.key((two,)))
        self.assertEqual(overlays.key((one,)), overlays.key((overlays.Overlay("A", 0, 2),)))

    def test_where_and_when_an_overlay_is_shown(self):
        self.assertEqual(overlays.window(overlays.Overlay("a", 1, 3), 10), (1.0, 3.0))
        self.assertEqual(overlays.window(overlays.Overlay("a", 8, 15), 10), (8.0, 10.0))  # cut at the film's end
        self.assertIsNone(overlays.window(overlays.Overlay("a", 12, 15), 10))
        self.assertEqual(overlays.window(overlays.Overlay("a", last=3), 10), (7.0, 10.0))
        self.assertEqual(overlays.window(overlays.Overlay("a", last=30), 10), (0.0, 10.0))
        rows, notes = overlays.layers([overlays.Overlay("Hi", 1, 3), overlays.Overlay("Late", 12, 15)], 10)
        self.assertEqual(rows, [(1.0, 3.0, "Hi", "center", "medium", "none")])
        self.assertIn("starts after the end of the film", notes[0])

    def test_two_texts_in_the_same_place_at_the_same_time_are_pointed_out(self):
        one, two = overlays.Overlay("A", 0, 3), overlays.Overlay("B", 2, 5)
        self.assertEqual(len(overlays.clashes([one, two], 10)), 1)
        self.assertEqual(overlays.clashes([one, overlays.Overlay("B", 2, 5, position="top")], 10), [])
        self.assertEqual(overlays.clashes([one, overlays.Overlay("B", 3, 5)], 10), [])


class BurnerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.src = self.root / "src.mp4"
        _clip(self.src, 3, 30)

    def burn(self, *layers):
        report = []
        out = self.root / "out.mp4"
        count = burner.burn(self.src, [burner.Layer(*layer) for layer in layers], out, comment=video.AI_COMMENT,
                            report=report)
        with av.open(str(out)) as container:
            self.assertEqual(container.metadata.get("comment"), video.AI_COMMENT)
            frames = list(container.decode(video=0))
        self.assertEqual((count, len(frames)), (90, 90))
        return frames, report

    def test_a_text_is_on_screen_only_between_its_times_and_fades(self):
        frames, report = self.burn((1.0, 2.0, "THE END", "center", "large", "none"))
        self.assertEqual(report, [])
        self.assertEqual(_white(frames[10]), 0)         # 0.3 s: not yet
        self.assertEqual(_white(frames[30]), 0)         # 1.0 s: it is just starting (faded in from nothing)
        self.assertGreater(_white(frames[45]), 100)     # 1.5 s: the letters
        self.assertLess(_white(frames[31]), _white(frames[45]))  # fading in
        self.assertEqual(_white(frames[62]), 0)         # after the end
        # the picture outside the letters keeps its colour
        self.assertTrue((np.abs(_rgb(frames[5]).astype(int) - 30) < 6).all())

    def test_a_text_that_runs_to_the_end_of_the_film_does_not_fade_out_there(self):
        frames, _ = self.burn((1.0, 3.0, "THE END", "center", "large", "card"), (0.2, 1.2, "Early", "top", "medium", "none"))
        self.assertGreater(_white(frames[89]), 100)                  # the very last frame is still the card
        self.assertLess(int(np.median(_rgb(frames[89]))), 25)
        self.assertEqual(burner.fade(2.95, 1.0, 3.0, 3.0), 1.0)
        self.assertLess(burner.fade(1.9, 1.0, 2.0, 3.0), 1.0)
        self.assertEqual(burner.fade(2.95, 1.0, 3.0), burner.fade(2.95, 1.0, 3.0, None))  # unknown end: as before
        self.assertLess(burner.fade(2.95, 1.0, 3.0, None), 0.5)

    def test_characters_no_font_can_draw_are_reported_and_a_text_of_nothing_else_is_refused(self):
        layer = burner.Layer(0.0, 3.0, "Hi \u2717\u2717", "center", "medium", "none")
        with patch.object(burner.subtitles, "_drawable", return_value=("arial.ttf", "Hi")):
            self.assertTrue(burner.compose(layer, SIZE).lost)
            frames, report = self.burn((0.0, 3.0, "Hi \u2717\u2717", "center", "medium", "none"))
        self.assertTrue(any("cannot be drawn" in line for line in report), report)
        with patch.object(burner.subtitles, "_drawable", return_value=("arial.ttf", "")):
            with self.assertRaisesRegex(MediaError, "no font of this PC"):
                burner.compose(layer, SIZE)
        self.assertFalse(burner.compose(burner.Layer(0.0, 3.0, "Hi", "center", "medium", "none"), SIZE).lost)

    def test_text_is_drawn_over_a_card_whatever_the_order_of_the_list(self):
        for layers in (((0.0, 3.0, "Tagline", "bottom", "medium", "none"), (1.0, 3.0, "END", "center", "large", "card")),
                       ((1.0, 3.0, "END", "center", "large", "card"), (0.0, 3.0, "Tagline", "bottom", "medium", "none"))):
            frames, _ = self.burn(*layers)
            self.assertGreater(_white(frames[60], slice(120, 180)), 0, layers)   # 2.0 s: the tagline is on the card
            self.assertGreater(_white(frames[60], slice(60, 120)), 0, layers)    # and so is the card's own text

    def test_a_variation_selector_is_not_a_character_that_was_left_out(self):
        self.assertEqual(burner._letters("\u2764\ufe0f ok"), 3)
        with patch.object(burner.subtitles, "_drawable", return_value=("seguisym.ttf", "\u2764 ok")):
            self.assertFalse(burner.compose(burner.Layer(0.0, 3.0, "\u2764\ufe0f ok", "center", "medium", "none"), SIZE).lost)

    def test_the_position_is_the_top_the_middle_or_the_bottom(self):
        spread = {}
        for where in ("top", "center", "bottom"):
            frames, _ = self.burn((0.0, 3.0, "Where", where, "medium", "none"))
            frame = frames[45]
            spread[where] = [_white(frame, slice(first, first + 60)) for first in (0, 60, 120)]
        self.assertTrue(spread["top"][0] > 0 and spread["top"][2] == 0, spread)
        self.assertTrue(spread["center"][1] > 0 and spread["center"][0] == 0 and spread["center"][2] == 0, spread)
        self.assertTrue(spread["bottom"][2] > 0 and spread["bottom"][0] == 0, spread)

    def test_a_card_hides_the_picture_and_a_panel_darkens_what_is_behind_the_text(self):
        bright = self.root / "bright.mp4"
        _clip(bright, 3, 200)
        self.src = bright
        frames, _ = self.burn((0.0, 3.0, "Card", "center", "large", "card"))
        picture = _rgb(frames[45])
        self.assertLess(int(np.median(picture)), 25)          # black everywhere ...
        self.assertGreater(_white(frames[45]), 50)             # ... with the letters on it
        frames, _ = self.burn((0.0, 3.0, "Panel", "bottom", "medium", "panel"))
        picture = _rgb(frames[45])
        self.assertLess(int(picture[150:175, 20:300].min()), 120)    # the panel behind the text
        self.assertGreater(int(picture[5:20, 5:300].mean()), 180)    # the rest of the picture is untouched

    def test_a_text_too_big_for_the_frame_is_made_smaller_and_a_word_is_never_cut(self):
        frames, report = self.burn((0.0, 3.0, "internationalizationinternationalization.example/offer", "center", "large",
                                    "none"))
        self.assertEqual(len(report), 1)
        self.assertIn("made smaller", report[0])
        portrait = (480, 854)
        big, small = (burner.subtitles.Style(size=units, margin_bottom=36.0, margin_side=18.0) for units in (28.0, 14.0))
        self.assertFalse(burner._words_fit("lumio.example/offer", portrait, big))      # it would be cut in two
        self.assertTrue(burner._words_fit("lumio.example/offer", portrait, small))
        address = burner.compose(burner.Layer(0.0, 3.0, "lumio.example/offer", "center", "large", "none"), portrait)
        self.assertTrue(address.shrunk)
        short = burner.compose(burner.Layer(0.0, 3.0, "THE END", "center", "large", "none"), portrait)
        self.assertFalse(short.shrunk)

    def test_two_texts_at_once_both_show(self):
        frames, _ = self.burn((0.0, 3.0, "Top text", "top", "medium", "none"),
                              (0.0, 3.0, "Bottom text", "bottom", "medium", "none"))
        self.assertGreater(_white(frames[45], slice(0, 60)), 0)
        self.assertGreater(_white(frames[45], slice(120, 180)), 0)

    def test_a_layer_without_time_is_ignored_and_no_layer_copies_the_film(self):
        frames, _ = self.burn((2.0, 2.0, "Nothing", "center", "medium", "none"))
        self.assertEqual(_white(frames[60]), 0)


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.root = Path(self.context.enter_context(tempfile.TemporaryDirectory())).resolve()
        for target in (patch.object(timeline, "DIR", self.root / "projects"),
                       patch("app.items.DIR", self.root / "items"),
                       patch.object(config, "OUTPUTS_DIR", self.root / "outputs"),
                       patch.object(config, "HISTORY_DIR", self.root / "history"),
                       patch.object(history, "_HISTORY_FILE", self.root / "history/history.jsonl"),
                       patch.object(characters, "DIR", self.root / "characters")):
            self.context.enter_context(target)

    def project(self):
        project = timeline.save(None, None, "Text film", (timeline.Scene("A cup", 1), timeline.Scene("A tree", 2)),
                                "Landscape 16:9", False, [])
        keys = ["1" * 64, "2" * 64]
        for key, seconds, gray in zip(keys, (1, 2), (20, 40)):
            clip, metadata = timeline._asset_paths(project, key)
            clip.parent.mkdir(parents=True, exist_ok=True)
            _clip(clip, seconds, gray)
            timeline._write(metadata, {"qc": {"ok": True, "reasons": [], "face": None}, "sha256": timeline._sha(clip),
                                       "seconds": seconds, "width": SIZE[0], "height": SIZE[1], "characters": [],
                                       "voices": [], "transition_note": "", "dialogue_timings": []})
        timeline._write_plan(project, {"assets": keys})
        return project

    def export(self, project):
        ok, files, notes, error = timeline.render_job({"project": project.id, "key": project.key}, lambda *_: None)
        self.assertTrue(ok, error)
        return Path(files[0]), notes

    def test_a_film_without_overlays_is_saved_and_exported_exactly_as_before(self):
        project = self.project()
        self.assertEqual(project.overlays, ())
        fields = json.loads((timeline.DIR / project.id / "project.json").read_text(encoding="utf-8"))
        self.assertNotIn("overlays", fields)
        with patch.object(ffmpeg_utils, "burn_overlays", side_effect=AssertionError("burned without overlays")):
            plain, notes = self.export(project)
        self.assertEqual((notes, timeline.exported(project)), ([], plain))
        self.assertNotIn("export_overlays", timeline.prepared(project))

    def test_overlays_are_saved_keep_the_key_and_survive_an_ordinary_save(self):
        project = self.project()
        card = overlays.Overlay("Lumio\nlumio.example", last=1, backdrop="card")
        saved = timeline.set_overlays(project.id, project.revision, [card])
        self.assertEqual((saved.revision, saved.key, saved.overlays), (project.revision + 1, project.key, (card,)))
        self.assertEqual(timeline.load(project.id).overlays, (card,))
        stored = json.loads((timeline.DIR / project.id / "project.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["overlays"], [{"text": "Lumio\nlumio.example", "last": 1.0, "backdrop": "card"}])
        # an ordinary save (the editor: a title change, the Agent: a scene change) keeps them
        again = timeline.save(saved.id, saved.revision, "Renamed", saved.scenes, saved.aspect, saved.audio, [])
        self.assertEqual(again.overlays, (card,))
        same = timeline.set_overlays(again.id, again.revision, [card])
        self.assertEqual(same.revision, again.revision)  # nothing changed: no new revision
        removed = timeline.set_overlays(same.id, same.revision, [])
        self.assertEqual(removed.overlays, ())
        stored = json.loads((timeline.DIR / project.id / "project.json").read_text(encoding="utf-8"))
        self.assertNotIn("overlays", stored)
        with self.assertRaisesRegex(ValueError, "Overlay 1"):
            timeline.set_overlays(removed.id, removed.revision, [overlays.Overlay("")])
        with self.assertRaisesRegex(ValueError, "another tab"):
            timeline.set_overlays(project.id, project.revision, [card])

    def test_a_damaged_overlay_in_the_file_does_not_make_the_project_unopenable(self):
        project = timeline.set_overlays(*(lambda p: (p.id, p.revision))(self.project()),
                                        [overlays.Overlay("Keep", 0, 2)])
        path = timeline.DIR / project.id / "project.json"
        fields = json.loads(path.read_text(encoding="utf-8"))
        fields["overlays"] = [{"text": "Keep", "end": 2}, {"text": ""}, 7]
        path.write_text(json.dumps(fields), encoding="utf-8")
        self.assertEqual([one.text for one in timeline.load(project.id).overlays], ["Keep"])

    def test_the_export_burns_the_text_and_is_made_again_when_the_text_changes(self):
        project = self.project()
        plain, _ = self.export(project)
        with_card = timeline.set_overlays(project.id, project.revision,
                                          [overlays.Overlay("THE END", last=1, backdrop="card", size="large")])
        self.assertIsNone(timeline.exported(with_card))   # the export without the text is not the current one
        burned, notes = self.export(with_card)
        self.assertNotEqual(burned, plain)
        self.assertEqual((notes, timeline.exported(with_card)), ([], burned))
        self.assertEqual(timeline.prepared(with_card)["export_overlays"], overlays.key(with_card.overlays))
        with av.open(str(burned)) as container:
            self.assertEqual(container.metadata.get("comment"), video.AI_COMMENT)
            frames = list(container.decode(video=0))
        self.assertEqual(len(frames), 90)
        self.assertEqual(_white(frames[30]), 0)            # 1 s: the picture
        self.assertGreater(_white(frames[75]), 100)        # 2.5 s: the card, in the last second
        self.assertLess(int(np.median(_rgb(frames[75]))), 25)

        other = timeline.set_overlays(with_card.id, with_card.revision,
                                      [overlays.Overlay("THE END", last=1, backdrop="card", size="small")])
        self.assertIsNone(timeline.exported(other))
        removed = timeline.set_overlays(other.id, other.revision, [])
        self.assertIsNone(timeline.exported(removed))
        again, _ = self.export(removed)
        self.assertEqual(again.read_bytes(), plain.read_bytes())   # the same film as before any text
        self.assertNotIn("export_overlays", timeline.prepared(removed))

    def test_the_subtitles_are_burned_first_and_the_text_goes_over_them(self):
        project = self.project()
        on = timeline.save(project.id, project.revision, project.title, project.scenes, project.aspect, project.audio, [],
                           subtitles=True, overlays=(overlays.Overlay("X", 0, 3, backdrop="card"),))
        order = []
        real_subtitles, real_text = ffmpeg_utils.burn_subtitles, ffmpeg_utils.burn_overlays

        def subtitles(src, cues, out):
            order.append(("subtitles", Path(src).name))
            return real_subtitles(src, cues, out)

        def text(src, layers, out, report=None):
            order.append(("text", Path(src).name))
            return real_text(src, layers, out, report=report)

        with patch.object(timeline, "subtitle_cues", return_value=[(0.2, 0.8, "Hello")]),                 patch.object(ffmpeg_utils, "burn_subtitles", subtitles), patch.object(ffmpeg_utils, "burn_overlays", text):
            movie, _ = self.export(on)
        self.assertEqual(order, [("subtitles", "render.mp4"), ("text", "subtitled.mp4")])
        self.assertEqual(timeline.exported(on), movie)

    def test_a_text_that_starts_after_the_end_is_left_out_with_a_note(self):
        project = self.project()
        late = timeline.set_overlays(project.id, project.revision, [overlays.Overlay("Late", 50, 55)])
        with patch.object(ffmpeg_utils, "burn_overlays", side_effect=AssertionError("nothing to burn")):
            movie, notes = self.export(late)
        self.assertEqual(len(notes), 1)
        self.assertIn("starts after the end of the film", notes[0])
        self.assertEqual(timeline.exported(late), movie)


if __name__ == "__main__":
    unittest.main()
