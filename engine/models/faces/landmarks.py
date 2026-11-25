# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""MediaPipe Face Landmarker in image mode, on the CPU (the face points of lip sync and face tracking)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

Box = tuple[int, int, int, int]


class Landmarker:
    """Load the caller's local model, using MediaPipe's CPU delegate."""

    def __init__(self, path: str | Path, num_faces: int):
        import mediapipe as mp

        self._mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=num_faces)
        self._model = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def points(self, frame: np.ndarray) -> list[np.ndarray]:
        h, w = frame.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame, dtype=np.uint8))
        return [np.asarray([(p.x * w, p.y * h) for p in face]) for face in self._model.detect(image).face_landmarks]

    def close(self) -> None:
        self._model.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

