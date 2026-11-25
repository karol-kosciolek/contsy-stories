# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Several characters in one picture: each face is matched to its own character (one face
each, look-alikes are not swapped on a guess), quality checks score every character, lip
sync gives each speaking character its own face, and a face that drifts between the parts of a
video is reported. Fake face embeddings - no face models, no GPU."""
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app import lipsync, qc

OLA = {"name": "Ola", "slug": "ola", "gender": "woman"}
KUBA = {"name": "Kuba", "slug": "kuba", "gender": "man"}


def vec(*values):
    return np.array([values], dtype=np.float32)


A, B, C = vec(1, 0, 0), vec(0, 1, 0), vec(0, 0, 1)


class AssignTests(unittest.TestCase):
    def test_each_character_gets_its_best_distinct_face(self):
        self.assertEqual(qc.assign({"ola": [0.2, 0.8], "kuba": [0.7, 0.1]}), {"ola": 1, "kuba": 0})

    def test_weak_and_ambiguous_faces_are_left_out(self):
        self.assertEqual(qc.assign({"ola": [0.3], "kuba": [0.2]}), {})        # below 0.35
        self.assertEqual(qc.assign({"ola": [0.60], "kuba": [0.58]}), {})      # a look-alike
        # once the rival has its own face, the ambiguous one is clear
        self.assertEqual(qc.assign({"ola": [0.60, 0.30], "kuba": [0.58, 0.50]}),
                         {"kuba": 1, "ola": 0})

    def test_without_threshold_and_margin_every_character_gets_a_face_if_one_is_left(self):
        self.assertEqual(qc.assign({"ola": [0.1], "kuba": [0.2]}, threshold=-1.0, margin=0.0),
                         {"kuba": 0})
        self.assertEqual(qc.assign({"ola": [0.1, 0.05], "kuba": [0.2, 0.0]}, -1.0, 0.0),
                         {"kuba": 0, "ola": 1})

    def test_cosine_matches_the_recogniser(self):
        self.assertAlmostEqual(qc._cos(A, A), 1.0)
        self.assertAlmostEqual(qc._cos(A, B), 0.0)
        self.assertEqual(qc._cos(vec(0, 0, 0), A), 0.0)


class MatchTests(unittest.TestCase):
    def setUp(self):
        refs = {"ola": [A], "kuba": [B]}
        target = patch.object(qc, "_references", side_effect=lambda ch: refs.get(ch["slug"], []))
        target.start()
        self.addCleanup(target.stop)

    def faces(self, *embs):
        boxes = [(10 + 100 * i, 10, 60 + 100 * i, 70) for i in range(len(embs))]
        return patch.object(qc, "_faces", return_value=list(zip(boxes, embs)))

    def test_count_faces_counts_every_face_on_the_image(self):
        with self.faces(B, A, C):
            self.assertEqual(qc.count_faces("x.png"), 3)
        with self.faces():
            self.assertEqual(qc.count_faces("x.png"), 0)

    def test_an_absent_character_is_not_scored_against_a_stranger(self):
        with self.faces(A, C):  # Ola and a stranger; Kuba is not in the picture
            self.assertEqual(qc.face_scores("x.png", [OLA, KUBA]), {"ola": 1.0, "kuba": None})

    def test_face_scores_use_distinct_faces_and_skip_characters_without_photos(self):
        nova = {"name": "Nova", "slug": "nova", "gender": "woman"}
        with self.faces(A):  # only Ola is in the picture
            scores = qc.face_scores("x.png", [OLA, KUBA, nova])
        self.assertEqual(scores, {"ola": 1.0, "kuba": None})

    def test_photo_check_reports_every_character(self):
        with self.faces(A), patch.object(qc.model_manager, "ensure", return_value=None):
            verdict = qc.check("x.png", [OLA, KUBA])
            frame = qc.check("x.png", [OLA, KUBA], require_all=False)
        self.assertEqual(verdict.faces, {"Ola": 1.0, "Kuba": None})
        self.assertEqual(verdict.face, 1.0)
        self.assertEqual((verdict.ok, verdict.reasons), (False, ["face of Kuba not found"]))
        self.assertEqual((frame.ok, frame.reasons), (True, []))
        self.assertEqual(verdict.record()["faces"], {"Ola": 1.0, "Kuba": None})
        self.assertNotIn("faces", qc.Verdict(True).record())

    def test_low_score_names_the_character(self):
        v = qc.judge(None, {"Ola": 0.2, "Kuba": 0.9})
        self.assertEqual(v.reasons, ["different face for Ola (similarity 0.20 < 0.40)"])

class LipSyncTests(unittest.TestCase):
    def test_speaking_characters_get_their_own_faces_others_the_unrecognised_ones(self):
        tracks = [{"crop": "face_0.png", "size": 200},
                  {"crop": "face_1.png", "size": 150},
                  {"crop": "face_2.png", "size": 120}]
        who = ["male#char:kuba", "female#char:ola", "female#0"]
        notes = []
        with patch.object(qc, "match_crops", return_value={"ola": 1, "kuba": 2}) as match:
            assigned = lipsync.assign_faces(who, tracks, [OLA, KUBA], "work", notes)
        self.assertIs(assigned["female#char:ola"], tracks[1])     # not the largest woman
        self.assertIs(assigned["male#char:kuba"], tracks[2])
        self.assertIs(assigned["female#0"], tracks[0])
        self.assertEqual([Path(p).name for p in match.call_args.args[0]],
                         ["face_0.png", "face_1.png", "face_2.png"])

    def test_unrecognised_character_falls_back_to_size(self):
        tracks = [{"crop": "face_0.png", "size": 200}]
        with patch.object(qc, "match_crops", return_value={}):
            assigned = lipsync.assign_faces(["female#char:ola"], tracks, [OLA], "work", [])
        self.assertIs(assigned["female#char:ola"], tracks[0])
        notes = []
        assigned = lipsync.assign_faces(["female#0", "male#0"], tracks[:1], [], "work", notes)
        self.assertEqual(len(assigned), 1)
        self.assertIn("more speakers than visible faces", notes[0])

    def test_a_small_face_crop_is_enlarged_and_full_frame_failure_falls_back_to_it(self):
        work = Path(tempfile.mkdtemp(prefix="lipsync_test_"))
        self.addCleanup(shutil.rmtree, work, True)
        faces = {"width": 848, "height": 480, "tracks": [
            {"bbox": [216, 77, 302, 195], "size": 86, "presence": 1.0, "crop": ""}]}
        runs, ffmpeg_calls = [], []

        def run(video, audio, output, cb=None):
            runs.append(Path(video).name)
            if len(runs) == 1:
                return False, "Lip sync covered 0 of 25 speech frames of the speaker (no face found in 25)"
            Path(output).write_bytes(b"mp4")
            return True, output

        with patch.object(lipsync, "find_faces", return_value=faces), \
                patch.object(lipsync, "run", side_effect=run), \
                patch.object(lipsync.dialogue, "speaker_track"), \
                patch.object(lipsync.ffmpeg_utils, "transform", side_effect=lambda src, out, chain, **kw: ffmpeg_calls.append(chain)), \
                patch.object(lipsync.ffmpeg_utils, "overlay_layers", return_value=(True, "")) as overlay:
            notes = []
            ok, out = lipsync._dialogue_clip(work / "v25.mp4", "d.wav", [(0.3, 1.4)], ["female#0"],
                                             str(work / "out.mp4"), 2.0, str(work), None, notes)
        self.assertTrue(ok)
        self.assertEqual(runs, ["v25.mp4", "crop0.mp4"])  # whole frame failed -> the face crop
        self.assertIn("retried on the face only", notes[0])
        crop_filter = ffmpeg_calls[0]
        self.assertRegex(crop_filter, r"^crop=212:212:\d+:\d+,scale=512:512:flags=lanczos$")
        self.assertEqual(overlay.call_args.args[1][0][3:], (212, 212))  # laid back at its size

    def _two_speakers(self, run_tracks, run=None):
        """_dialogue_clip on a shot with two faces and a line for each, with lip sync replaced by fakes."""
        work = Path(tempfile.mkdtemp(prefix="lipsync_test_"))
        self.addCleanup(shutil.rmtree, work, True)
        faces = {"width": 848, "height": 480, "tracks": [
            {"bbox": [100, 80, 220, 230], "size": 150, "presence": 1.0, "crop": ""},
            {"bbox": [560, 70, 655, 210], "size": 140, "presence": 1.0, "crop": ""},
            {"bbox": [400, 300, 450, 350], "size": 50, "presence": 0.5, "crop": ""}]}
        calls = SimpleNamespace(tracks=[], runs=[], transforms=[], overlay=None)

        def tracks(video, speakers, output, cb=None, *, num_faces=0):
            calls.tracks.append((Path(video).name, [Path(path).name for path, _ in speakers],
                                 [list(box) for _, box in speakers], num_faces))
            ok, result = run_tracks()
            if ok:
                Path(output).write_bytes(b"together")
            return ok, result

        def single(video, audio, output, cb=None):
            calls.runs.append((Path(video).name, Path(audio).name))
            ok, message = run(video) if run else (True, output)
            if ok:
                Path(output).write_bytes(b"crop")
            return ok, message

        def overlay(base, layers, output):
            calls.overlay = (Path(base).name, [Path(layer[0]).name for layer in layers])
            Path(output).write_bytes(b"overlaid")
            return True, ""

        notes = []
        with patch.object(lipsync, "find_faces", return_value=faces), \
                patch.object(lipsync, "run_tracks", side_effect=tracks), \
                patch.object(lipsync, "run", side_effect=single), \
                patch.object(lipsync.dialogue, "speaker_track"), \
                patch.object(lipsync.ffmpeg_utils, "transform",
                             side_effect=lambda src, out, chain, **kw: calls.transforms.append(chain)), \
                patch.object(lipsync.ffmpeg_utils, "overlay_layers", side_effect=overlay):
            ok, out = lipsync._dialogue_clip(work / "v25.mp4", "d.wav", [(0.3, 1.4), (1.5, 2.5)],
                                             ["female#0", "male#0"], str(work / "out.mp4"), 3.0, str(work),
                                             None, notes)
        return ok, Path(out), calls, notes

    def test_all_speakers_are_lip_synced_in_one_pass_over_the_whole_frame(self):
        covered = [SimpleNamespace(coverage=1.0), SimpleNamespace(coverage=0.9)]
        ok, out, calls, notes = self._two_speakers(lambda: (True, covered))
        self.assertTrue(ok)
        self.assertEqual(out.read_bytes(), b"together")
        # each face with its own track, anchored at its box; the detector looks for every face of the shot
        self.assertEqual(calls.tracks, [("v25.mp4", ["a0.wav", "a1.wav"],
                                         [[100, 80, 220, 230], [560, 70, 655, 210]], 3)])
        self.assertEqual((calls.runs, calls.transforms, calls.overlay, notes), ([], [], None, []))

    def test_a_face_the_one_pass_missed_is_retried_on_its_crop_over_that_result(self):
        partial = [SimpleNamespace(coverage=1.0), SimpleNamespace(coverage=0.2)]
        ok, out, calls, notes = self._two_speakers(lambda: (True, partial))
        self.assertTrue(ok)
        self.assertEqual(calls.runs, [("crop1.mp4", "a1.wav")])  # only the man, from his own crop
        self.assertEqual(calls.overlay, ("together.mp4", ["ls1.mp4"]))  # laid over the one-pass result
        self.assertIn("missed a face", notes[0])
        # the crop retry fails as well: the one-pass result stays, with a note for that person
        ok, out, calls, notes = self._two_speakers(lambda: (True, partial), run=lambda video: (False, "no face"))
        self.assertTrue(ok)
        self.assertEqual(out.read_bytes(), b"together")
        self.assertIsNone(calls.overlay)
        self.assertIn("Lip sync for person 2 skipped (no face).", notes)

    def test_a_one_pass_that_reached_no_speaker_is_a_failure_when_every_crop_fails(self):
        missed = [SimpleNamespace(coverage=0.1), SimpleNamespace(coverage=0.2)]
        ok, out, calls, notes = self._two_speakers(lambda: (True, missed), run=lambda video: (False, "no face"))
        self.assertFalse(ok)
        self.assertEqual(str(out), "No face could be lip-synced (no face)")
        self.assertEqual(calls.runs, [("crop0.mp4", "a0.wav"), ("crop1.mp4", "a1.wav")])
        self.assertIsNone(calls.overlay)

    def test_a_failed_face_tracking_is_not_reported_as_a_framing_problem(self):
        with patch("engine.models.faces.find_faces", side_effect=OSError("landmarker download failed")), \
                patch("app.aux_models.ensure_landmarker", return_value="landmarker.task"):
            faces = lipsync.find_faces("v25.mp4", "work")
        self.assertEqual(faces, {"tracks": [], "error": "OSError: landmarker download failed"})
        with patch.object(lipsync, "find_faces", return_value=faces):
            ok, message = lipsync._dialogue_clip(Path("v25.mp4"), "d.wav", [(0.3, 1.4)], ["female#0"], "out.mp4",
                                                 2.0, "work", None, [])
        self.assertEqual((ok, message), (False, "Face tracking failed: OSError: landmarker download failed"))

    def test_the_lip_sync_rate_is_the_models(self):
        from engine.pipelines import musetalk

        self.assertEqual(lipsync.FPS, musetalk.FPS)

    def test_a_failed_one_pass_falls_back_to_one_crop_per_face(self):
        ok, out, calls, notes = self._two_speakers(lambda: (False, "CUDA out of memory"))
        self.assertTrue(ok)
        self.assertEqual(calls.runs, [("crop0.mp4", "a0.wav"), ("crop1.mp4", "a1.wav")])
        self.assertEqual(calls.overlay, ("v25.mp4", ["ls0.mp4", "ls1.mp4"]))
        self.assertIn("all speakers at once failed (CUDA out of memory)", notes[0])
        ok, _out, _calls, _notes = self._two_speakers(lambda: (False, "CUDA out of memory"),
                                                      run=lambda video: (False, "no face"))
        self.assertFalse(ok)

    def test_a_silent_character_s_face_is_matched_too_and_kept_from_a_speaker(self):
        nova = {"name": "Nova", "slug": "nova", "gender": "woman"}
        tracks = [{"crop": "face_0.png", "size": 200},
                  {"crop": "face_1.png", "size": 150}]
        # every character in the video is matched, not only the speakers
        with patch.object(qc, "match_crops", return_value={"nova": 0, "ola": 1}) as match:
            assigned = lipsync.assign_faces(["female#char:ola"], tracks, [OLA, nova], "work", [])
        self.assertIs(assigned["female#char:ola"], tracks[1])
        self.assertEqual([c["slug"] for c in match.call_args.args[1]], ["ola", "nova"])
        # Ola is not recognised: her line does not fall back onto Nova's recognised face
        with patch.object(qc, "match_crops", return_value={"nova": 0}):
            assigned = lipsync.assign_faces(["female#char:ola"], tracks, [OLA, nova], "work", [])
        self.assertIs(assigned["female#char:ola"], tracks[1])
        # an unnamed line takes the largest face nobody was recognised in, not the silent Nova's
        with patch.object(qc, "match_crops", return_value={"nova": 0}) as match:
            assigned = lipsync.assign_faces(["female#0"], tracks, [OLA, nova], "work", [])
        match.assert_called_once()
        self.assertIs(assigned["female#0"], tracks[1])

    def test_a_recognised_face_carries_the_gender_stored_in_the_characters_profile(self):
        """No gender is guessed from a face: when every face is a recognised character, an unnamed speaker takes
        the face of a character whose profile has the speaker's gender; another speaker's face is never taken."""
        tracks = [{"crop": "face_0.png", "size": 200}, {"crop": "face_1.png", "size": 150}]
        with patch.object(qc, "match_crops", return_value={"kuba": 0, "ola": 1}):
            assigned = lipsync.assign_faces(["female#0", "male#0"], tracks, [OLA, KUBA], "work", [])
        self.assertIs(assigned["female#0"], tracks[1])  # Ola's profile says woman, though Kuba's face is larger
        self.assertIs(assigned["male#0"], tracks[0])
        # without face models nothing is recognised: faces go by size, in the order the speakers first talk
        with patch.object(qc, "match_crops", side_effect=OSError("no SFace model")):
            assigned = lipsync.assign_faces(["male#0", "female#0"], tracks, [OLA, KUBA], "work", [])
        self.assertEqual((assigned["male#0"], assigned["female#0"]), (tracks[0], tracks[1]))
        # no cast: nothing to recognise and no model is asked
        with patch.object(qc, "match_crops", side_effect=AssertionError("no cast, no matching")):
            assigned = lipsync.assign_faces(["female#0"], tracks, [], "work", [])
        self.assertIs(assigned["female#0"], tracks[0])


class ReferenceAndDriftTests(unittest.TestCase):
    def test_the_reference_faces_are_the_characters_reference_photos(self):
        tmp = Path(tempfile.mkdtemp(prefix="refs_test_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        photos = [tmp / "01.png", tmp / "02.png"]
        for photo in photos:
            photo.write_bytes(b"png")
        ola = {**OLA, "photos": [str(p) for p in photos]}
        embedded = []
        with patch.object(qc, "_embed", side_effect=lambda path: embedded.append(Path(path).name) or A),                 patch.dict(qc._REFS, clear=True):
            self.assertEqual(len(qc._references(ola)), 2)
            qc._references(ola)  # computed once for an unchanged set of files
            self.assertEqual(embedded, ["01.png", "02.png"])
            self.assertEqual(qc._references({**OLA, "photos": []}), [])

    def test_a_face_that_drifted_at_a_part_handoff_is_reported(self):
        from app import video_gen

        drifted = qc.Verdict(False, ["different face (similarity 0.21 < 0.40)"])
        with patch.object(qc, "enabled", return_value=True),                 patch.object(qc, "check", return_value=drifted) as check:
            self.assertEqual(video_gen._drift(Path("last.png"), [OLA]), "different face (similarity 0.21 < 0.40)")
            self.assertEqual(check.call_args.args[1], OLA)  # one character: its own score
            video_gen._drift(Path("last.png"), [OLA, KUBA])
            self.assertEqual((check.call_args.args[1], check.call_args.kwargs), ([OLA, KUBA], {"require_all": False}))
        with patch.object(qc, "enabled", return_value=False), patch.object(qc, "check") as check:
            self.assertEqual(video_gen._drift(Path("last.png"), [OLA]), "")
            check.assert_not_called()
        self.assertEqual(video_gen._drift(Path("last.png"), []), "")


if __name__ == "__main__":
    unittest.main()
