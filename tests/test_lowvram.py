# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Lip-sync wiring: the reduced-memory mode on smaller GPUs and the one pass for several speakers."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from app import config, lipsync, model_manager
from engine.media import audio, video
from engine.models.musetalk import faces


class LowVramTests(unittest.TestCase):
    def test_lip_sync_selects_the_memory_mode_for_the_configured_limit(self):
        runtime = Mock()
        frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
        runtime.lipsync.return_value = frames
        frame = SimpleNamespace(to_ndarray=lambda **_: frames[0])
        with (patch.object(model_manager, "model_path", return_value=Path("weights")),
              patch.object(model_manager, "small_gpu", return_value=False),
              patch.object(lipsync.engine_runtime, "engine", return_value=runtime),
              patch("app.aux_models.ensure_landmarker", return_value=Path("landmarker")),
              patch.object(faces, "MuseTalkFaceDetector", return_value=MagicMock()),
              patch.object(video, "read_frames", return_value=[frame] * 3),
              patch.object(audio, "read", return_value=(np.zeros((1, 16000), np.float32), 16000)),
              patch.object(video, "Writer") as writer):
            for limit, expected in ((8, True), (12, True), (16, False), (None, False)):
                with (self.subTest(limit=limit),
                      patch.dict(config.SETTINGS["engine"], vram_limit_gib=limit)):
                    self.assertEqual(lipsync.run("in.mp4", "voice.wav", "out.mp4"), (True, "out.mp4"))
                    self.assertIs(runtime.lipsync.call_args.kwargs["low_vram"], expected)
                    self.assertNotIn("audio", writer.call_args.kwargs)  # the clip's mix adds the sound

    def test_one_pass_follows_every_speaker_s_face_and_returns_their_coverage(self):
        from engine.models.musetalk.faces import Face
        from engine.pipelines.musetalk import Report, TrackReport

        frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
        frame = SimpleNamespace(to_ndarray=lambda **_: frames[0])
        left, right = (Face((1, 1, 6, 6), np.zeros((3, 2))), Face((9, 1, 14, 6), np.zeros((3, 2))))
        detector = MagicMock()
        detector.__enter__.return_value.faces.side_effect = [[right, left], [left]]  # largest first
        runtime = Mock()

        def lipsync_tracks(*args, report_cb=None, **kwargs):
            report_cb(Report((TrackReport(frames=2, speech=2, synced=2), TrackReport(frames=2, speech=2, synced=1))))
            return frames

        runtime.lipsync_tracks.side_effect = lipsync_tracks
        with (patch.object(model_manager, "model_path", return_value=Path("weights")),
              patch.object(model_manager, "small_gpu", return_value=False),
              patch.object(lipsync.engine_runtime, "engine", return_value=runtime),
              patch("app.aux_models.ensure_landmarker", return_value=Path("landmarker")),
              patch.object(faces, "MuseTalkFaceDetector", return_value=detector) as made,
              patch.object(video, "read_frames", return_value=[frame] * 2),
              patch.object(audio, "read", return_value=(np.zeros((1, 16000), np.float32), 16000)),
              patch.object(video, "Writer") as writer):
            ok, reports = lipsync.run_tracks("in.mp4", [("a0.wav", (0, 0, 7, 7)), ("a1.wav", (8, 0, 15, 7))],
                                             "out.mp4", num_faces=3)
        self.assertTrue(ok)
        self.assertEqual([report.coverage for report in reports], [1.0, 0.5])
        self.assertEqual(made.call_args.kwargs["num_faces"], 3)
        tracks = runtime.lipsync_tracks.call_args.args[4]
        self.assertEqual([track.faces for track in tracks], [[left, left], [right, None]])
        self.assertNotIn("audio", writer.call_args.kwargs)  # the clip's mix adds the sound
