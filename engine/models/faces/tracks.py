# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Face tracks for the dialogue pipeline: MediaPipe faces (with YuNet when it is installed) followed through a clip.

A track carries no guess about the person: lip sync (app/lipsync.py) learns who a face is by matching its head crop
to the photos of the cast with SFace, and takes the gender from the recognised character's profile.

The faces can be found on every `stride`-th frame only (the first, then every stride-th): a track's presence,
box, size and head crop then come from those frames. Lip sync finds the face again in every frame for the
mouth itself (engine.models.musetalk.faces), so the tracks only have to tell who is where."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from ...core.cancel import CancelToken
from .landmarks import Landmarker
from .recognition import load_yunet

MIN_PRESENCE = 0.4
Box = tuple[float, float, float, float]


def _iou(a: Box, b: Box) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _bounded(box: Box, width: int, height: int) -> Box:
    x0, y0, x1, y1 = box
    return max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)


def face_boxes(rgb: np.ndarray, landmarks, yunet=None) -> list[Box]:
    """MediaPipe boxes first, then small YuNet faces without an overlapping box."""
    h, w = rgb.shape[:2]
    boxes: list[Box] = []
    for points in landmarks(rgb):
        x0, y0 = map(float, points.min(axis=0))
        x1, y1 = map(float, points.max(axis=0))
        box = _bounded((x0, y0, x1, y1), w, h)
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    if yunet is not None:
        import cv2

        yunet.setInputSize((w, h))
        _, found = yunet.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        for item in found if found is not None else ():
            x, y, fw, fh = (float(v) for v in item[:4])
            box = _bounded((x, y, x + fw, y + fh), w, h)
            if box[2] > box[0] and box[3] > box[1] and all(_iou(box, old) < 0.3 for old in boxes):
                boxes.append(box)
    return boxes


@dataclass
class _Track:
    last: Box
    seen_at: int = -1
    boxes: list[Box] = field(default_factory=list)
    largest: float = 0.0
    wide: np.ndarray | None = None

    def append(self, index: int, box: Box, rgb: np.ndarray) -> None:
        self.boxes.append(box)
        self.last, self.seen_at = box, index
        x0, y0, x1, y1 = box
        width = x1 - x0
        if width > self.largest and int(x1) > int(x0) and int(y1) > int(y0):
            # the head with room around it, from the frame where the face is largest: what face recognition reads
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            half = max(width, y1 - y0) * 0.8
            h, w = rgb.shape[:2]
            self.wide = rgb[int(max(cy - half, 0)):int(min(cy + half, h)),
                            int(max(cx - half, 0)):int(min(cx + half, w))].copy()
            self.largest = width

    def describe(self, count: int) -> dict:
        boxes = np.asarray(self.boxes)
        return {"presence": round(len(self.boxes) / count, 2),
                "bbox": [float(boxes[:, 0].min()), float(boxes[:, 1].min()),
                         float(boxes[:, 2].max()), float(boxes[:, 3].max())],
                "size": float(np.median(boxes[:, 2] - boxes[:, 0])),
                "cx": float(np.median((boxes[:, 0] + boxes[:, 2]) / 2))}


def _tracks(frames: Iterable[np.ndarray], landmarks, yunet, cancel: CancelToken):
    """The persistent tracks of the frames given (each one is looked at), their count and the frame size."""
    tracks: list[_Track] = []
    count = w = h = 0
    for count, rgb in enumerate(frames, start=1):
        cancel.check()
        h, w = rgb.shape[:2]
        for box in face_boxes(rgb, landmarks, yunet):
            best = max(tracks, key=lambda t: _iou(t.last, box), default=None)
            if best is None or best.seen_at == count or _iou(best.last, box) <= 0.2:
                best = _Track(box)
                tracks.append(best)
            best.append(count, box, rgb)
    kept = [t for t in tracks if len(t.boxes) >= MIN_PRESENCE * max(count, 1) and t.wide is not None]
    return kept, count, w, h


def _every(frames, stride: int):
    for index, frame in enumerate(frames):
        if index % stride == 0:
            yield frame


def find_faces(video: str | Path, workdir: str | Path, *, landmarker_path: str | Path,
               yunet_path: str | Path | None = None, cancel: CancelToken | None = None,
               clip_dir: object = None, stride: int = 1) -> dict:
    """Find persistent faces and write a head crop of each (face_<n>.png in `workdir`). `stride`: the faces are
    looked for on every stride-th frame ("frames" counts all of the clip's frames, a track's presence the share of
    the looked-at frames that show it).

    `clip_dir` is transitional and ignored: the lip sync of a running Studio that started before the face tracks lost
    their gender guess still passes it, until it restarts (its speakers then go by face size). Remove it after that
    restart."""
    from ...media import video as media

    cancel = cancel or CancelToken()
    cancel.check()
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be a positive integer")
    detector = None
    if yunet_path is not None and Path(yunet_path).is_file():
        try:
            detector = load_yunet(yunet_path)
        except Exception:
            pass  # YuNet is optional: the tracker also runs with MediaPipe alone.
    fps = media.probe(video).fps or 25.0
    with Landmarker(landmarker_path, num_faces=10) as landmarker:
        frames = media.read_frames(video)
        total = [0]

        def counted():
            for frame in frames:
                total[0] += 1
                yield frame

        try:
            tracks, looked, w, h = _tracks((media.to_rgb(frame) for frame in _every(counted(), stride)),
                                           landmarker.points, detector, cancel)
        finally:
            frames.close()
    results = []
    for index, track in enumerate(tracks):
        cancel.check()
        name = f"face_{index}.png"
        Image.fromarray(track.wide).save(Path(workdir) / name)
        results.append({**track.describe(looked), "crop": name})
    return {"frames": total[0], "fps": fps, "width": w, "height": h, "tracks": results}
