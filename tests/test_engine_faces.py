# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""CPU face utilities with synthetic detections and a real PyAV clip."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "cv2", "av"))


@unittest.skipUnless(HAVE, "face detection and media dependencies are missing")
class FaceTests(unittest.TestCase):
    def test_yunet_supplements_landmarks_and_suppresses_duplicate_boxes(self):
        import numpy as np
        from engine.models.faces.tracks import face_boxes

        rgb = np.full((48, 64, 3), [11, 22, 33], dtype=np.uint8)

        class YuNet:
            def setInputSize(self, size):
                self.size = size

            def detect(self, bgr):
                assert tuple(bgr[0, 0]) == (33, 22, 11)
                return True, [[4, 5, 21, 22], [50, 30, 18, 12]]

        yunet = YuNet()
        boxes = face_boxes(rgb, lambda _: [np.array([[4, 5], [25, 27]])], yunet)
        self.assertEqual(yunet.size, (64, 48))
        self.assertEqual(boxes, [(4, 5, 25, 27), (50, 30, 64, 42)])
        self.assertEqual(face_boxes(rgb, lambda _: [np.array([[4, 5], [25, 27]])]), [(4, 5, 25, 27)])  # no YuNet

    def test_tracking_uses_pyav_and_writes_only_persistent_face_crops(self):
        import numpy as np
        from PIL import Image
        from engine.media import video
        from engine.models.faces import tracks

        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / "source.mp4"
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            with video.Writer(source, 25, 64, 48) as writer:
                for _ in range(5):
                    writer.write(frame)
            main = np.array([[-2, 2], [20, 27]])
            transient = np.array([[40, 3], [58, 21]])
            with patch.object(tracks, "Landmarker") as factory:
                handle = factory.return_value.__enter__.return_value
                handle.points.side_effect = [[main, transient], [main], [main], [], [main]]
                result = tracks.find_faces(source, work, landmarker_path="local.task")
            # no guess about the person: lip sync recognises who a face is from its head crop
            self.assertEqual(result, {"frames": 5, "fps": 25.0, "width": 64, "height": 48, "tracks": [
                {"presence": 0.8, "bbox": [0.0, 2.0, 20.0, 27.0], "size": 20.0, "cx": 10.0,
                 "crop": "face_0.png"}]})
            self.assertFalse((Path(work) / "face_1.png").exists())
            with Image.open(Path(work) / "face_0.png") as picture:
                self.assertEqual(picture.size, (30, 34))

    def test_a_stride_looks_at_every_nth_frame_and_counts_presence_among_them(self):
        import numpy as np
        from engine.media import video
        from engine.models.faces import tracks

        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / "source.mp4"
            with video.Writer(source, 25, 64, 48) as writer:
                for _ in range(7):
                    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            main = np.array([[-2, 2], [20, 27]])
            with patch.object(tracks, "Landmarker") as factory:
                handle = factory.return_value.__enter__.return_value
                handle.points.side_effect = [[main], [], [main + 1]]  # frames 0, 3 and 6
                result = tracks.find_faces(source, work, landmarker_path="local.task", stride=3)
            self.assertEqual(handle.points.call_count, 3)
            self.assertEqual(result["frames"], 7)
            self.assertEqual(result["tracks"], [{"presence": 0.67, "bbox": [0.0, 2.0, 21.0, 28.0], "size": 20.5,
                                                 "cx": 10.25, "crop": "face_0.png"}])
            with self.assertRaises(ValueError):
                tracks.find_faces(source, work, landmarker_path="local.task", stride=0)

    def test_the_call_of_a_studio_started_before_the_change_still_finds_faces(self):
        """Transitional (remove with `clip_dir` after the Studio restarts): its lip sync passes a CLIP folder getter,
        which is neither called nor needed."""
        import numpy as np
        from engine.media import video
        from engine.models.faces import tracks

        def no_clip():
            raise AssertionError("no model guesses a face's gender")

        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / "source.mp4"
            with video.Writer(source, 25, 64, 48) as writer:
                for _ in range(3):
                    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            with patch.object(tracks, "Landmarker") as factory:
                factory.return_value.__enter__.return_value.points.return_value = [np.array([[-2, 2], [20, 27]])]
                result = tracks.find_faces(source, work, landmarker_path="local.task", clip_dir=no_clip,
                                           yunet_path=None, cancel=None)
            self.assertEqual([track["crop"] for track in result["tracks"]], ["face_0.png"])
            self.assertNotIn("gender", result["tracks"][0])
            self.assertTrue((Path(work) / "face_0.png").is_file())

    def test_a_clip_without_faces_gives_no_tracks_and_no_crops(self):
        import numpy as np
        from engine.media import video
        from engine.models.faces import tracks

        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / "blank.mp4"
            with video.Writer(source, 25, 32, 32) as writer:
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            with patch.object(tracks, "Landmarker") as factory:
                factory.return_value.__enter__.return_value.points.return_value = []
                result = tracks.find_faces(source, work, landmarker_path="local.task")
            self.assertEqual(result["tracks"], [])
            self.assertEqual(list(Path(work).glob("face_*.png")), [])

    def test_face_embeddings_keep_original_pixel_coordinates_and_sort(self):
        from unittest.mock import Mock
        from PIL import Image
        from engine.models.faces import cosine_similarity, face_embeddings

        detector = Mock()
        detector.detect.return_value = True, [[2, 3, 4, 5], [50, 40, 25, 20]]
        recognizer = Mock()
        recognizer.alignCrop.side_effect = lambda _img, face: face
        recognizer.feature.side_effect = lambda face: face[:2]
        with tempfile.TemporaryDirectory() as work:
            image = Path(work) / "żółw.png"
            Image.new("RGB", (2560, 960)).save(image)
            self.assertEqual(face_embeddings(image, detector, recognizer), [
                ((100, 80, 150, 120), [50, 40]), ((4, 6, 12, 16), [2, 3])])
            detector.setInputSize.assert_called_once_with((1280, 480))
            self.assertEqual(face_embeddings(Path(work) / "missing.png", detector, recognizer), [])
        self.assertAlmostEqual(cosine_similarity([1, 1], [2, 2]), 1.0)
        self.assertEqual(cosine_similarity([0, 0], [2, 2]), 0.0)

    def test_face_track_cancellation_happens_before_opening_input(self):
        from engine.core.cancel import CancelToken, Cancelled
        from engine.models.faces import find_faces

        token = CancelToken()
        token.cancel()
        with self.assertRaises(Cancelled):
            find_faces("missing.mp4", "missing", landmarker_path="missing.task", cancel=token)
