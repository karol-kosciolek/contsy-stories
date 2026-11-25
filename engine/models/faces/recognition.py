# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""OpenCV YuNet face detection and SFace identity embeddings."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .landmarks import Box


def load_yunet(path: str | Path):
    import cv2

    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.8, 0.3, 5000)


def load_sface(path: str | Path):
    import cv2

    return cv2.FaceRecognizerSF.create(str(path), "")


def face_embeddings(path: str | Path, detector, recognizer) -> list[tuple[Box, np.ndarray]]:
    """Largest-first SFace embeddings and their bounding boxes in source pixels."""
    import cv2

    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return []
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
    if img is None:
        return []
    h, w = img.shape[:2]
    scale = min(1.0, 1280 / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None:
        return []
    result = []
    for face in sorted(faces, key=lambda item: -item[2] * item[3]):
        x, y, fw, fh = (float(v) / scale for v in face[:4])
        box = (int(max(x, 0)), int(max(y, 0)), int(x + fw), int(y + fh))
        result.append((box, recognizer.feature(recognizer.alignCrop(img, face))))
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / norm if norm else 0.0

