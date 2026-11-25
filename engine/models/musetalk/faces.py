# SPDX-FileCopyrightText: 2024 Tencent Music Entertainment Group
# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only AND MIT
"""Face boxes, blend masks and compositing for MuseTalk 1.5, with our own detectors.

The reference (TMElyralab/MuseTalk, MIT: `musetalk/utils/preprocessing.py`, `blending.py`) takes 68 face
points from DWPose, an S3FD detection and a BiSeNet face parsing. None of those ship here (their licences and
the lack of Windows wheels); this module re-implements the geometry on MediaPipe's 478 points,
with OpenCV YuNet for faces MediaPipe misses, and replaces the parsing with a geometric mask.

Kept from the reference, because the network was trained on it:

* The crop box: as wide as the jaw contour, its bottom at the chin, its vertical centre at a point on the
  lower nose bridge (68-point index 29), so the masked lower half starts there. A margin below the chin.
* The crop is resized to 256 x 256 with Lanczos regardless of its aspect ratio.
* The blend: the generated face is resized to the box, only the region below the middle row of the box
  changes, and the mask is feathered with a Gaussian of about 5 % of the 1.5x blend square.

Our own (each one switchable, see `engine.pipelines.musetalk.Settings`): the mask is the jaw polygon with the
nose cut out instead of a parsed skin region; it fades to zero before the box edges, so no box outline can
show, and the chin margin is sized so that this fade starts below the grown jaw (`blend_margin`); boxes and
polygons are averaged over a few frames against jitter, and short detection gaps are bridged, but only
between overlapping boxes (never across two people); frames without a usable face keep their pixels, with
the change faded in and out around them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

Box = tuple[float, float, float, float]
IntBox = tuple[int, int, int, int]

# MediaPipe face mesh indices (FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL / _NOSE).
JAW = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234]
NOSE = [98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64]
#: Stand-in for point 29 of the 68-point layout (two thirds down the nose bridge, between 195 and 5).
#: A design assumption: the README says the upper bound of the box changes how far the mouth opens.
BRIDGE = (195, 5)
NOSE_TIP = 1
CHEEKS = (234, 454)

JAW_SPREAD = (0.02, 0.04)   # the jaw polygon grows by this share of the box (x, y): an opened chin drops
NOSE_SPREAD = 0.03          # the nose polygon grows by this share of the box width
CHEEK_INSET = 0.06          # the outer cheeks keep their pixels (BiSeNet 'jaw' mode erodes them as well)
YUNET_DROP = 0.2            # nose-bridge point = nose tip - this x the eye-to-mouth distance (YuNet only)
#: Two faces in neighbouring frames (or across a bridged gap) are the same face only when their boxes
#: overlap at least this much; otherwise smoothing and gap filling would average two people.
CONTINUITY_IOU = 0.3


@dataclass(frozen=True)
class Face:
    """One face in one frame, in frame pixels.

    `box` is the reference crop box before the chin margin (x0, y0, x1, y1). `region` is the polygon that
    may change (the jaw contour, or an ellipse around the mouth when only YuNet saw the face); `hole` is a
    polygon kept original inside it (the nose) or None. `turn` is 0 for a frontal face and about 1 for a
    profile."""
    box: Box
    region: np.ndarray
    hole: Optional[np.ndarray] = None
    turn: float = 0.0

    def compatible(self, other: "Face") -> bool:
        """True when the two faces can be averaged point by point (same polygon layouts)."""
        return (self.region.shape == other.region.shape
                and (self.hole is None) == (other.hole is None)
                and (self.hole is None or self.hole.shape == other.hole.shape))

    def continues(self, other: "Face") -> bool:
        """True when `other` can be the same face a few frames later: averageable and overlapping. A
        detector that jumps to another person, or a gap between two people, must not be averaged."""
        return self.compatible(other) and _iou(self.box, other.box) >= CONTINUITY_IOU


def reference_box(left: float, right: float, bottom: float, center_y: float, clamp_top: bool = True) -> Box:
    """The reference box: centred vertically on `center_y`, from the face's sides down to its chin."""
    top = center_y - (bottom - center_y)
    return float(left), float(max(0.0, top) if clamp_top else top), float(right), float(bottom)


def _turn(nose_x: float, left_x: float, right_x: float) -> float:
    span = right_x - left_x
    if not math.isfinite(span) or abs(span) < 1e-6:
        return 1.0
    return float(min(1.0, abs(2.0 * (nose_x - left_x) / span - 1.0)))


def face_from_landmarks(points: np.ndarray) -> Optional[Face]:
    """A Face from MediaPipe's 478 (or 468) points in frame pixels, None when they make no face box."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
        return None
    jaw = points[JAW]
    center_y = float(points[list(BRIDGE), 1].mean())
    left, right, bottom = float(jaw[:, 0].min()), float(jaw[:, 0].max()), float(jaw[:, 1].max())
    if right - left < 1 or bottom - center_y < 1:
        return None
    sides = sorted(points[list(CHEEKS), 0])
    return Face(reference_box(left, right, bottom, center_y, clamp_top=False), jaw.copy(), points[NOSE].copy(),
                _turn(float(points[NOSE_TIP, 0]), sides[0], sides[1]))


def _ellipse(cx: float, cy: float, rx: float, ry: float, count: int = 32) -> np.ndarray:
    angle = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.stack((cx + rx * np.cos(angle), cy + ry * np.sin(angle)), axis=1)


def face_from_yunet(row: np.ndarray) -> Optional[Face]:
    """A Face from one YuNet detection (box, right eye, left eye, nose tip, mouth corners, score).

    YuNet has no jaw contour: the box takes its sides and bottom from the detection, the centre row
    from the nose tip, and the region is an ellipse from just below the nose to the chin."""
    row = np.asarray(row, dtype=np.float64)
    if row.shape[0] < 14 or not np.isfinite(row[:14]).all():
        return None
    x, y, w, h = row[:4]
    eyes, nose, mouth = row[4:8].reshape(2, 2), row[8:10], row[10:14].reshape(2, 2)
    eye_mid, mouth_mid = eyes.mean(axis=0), mouth.mean(axis=0)
    center_y = nose[1] - YUNET_DROP * float(np.linalg.norm(mouth_mid - eye_mid))
    bottom = y + h
    if w < 1 or bottom - center_y < 1:
        return None
    top = nose[1] + 0.15 * max(mouth_mid[1] - nose[1], 0.0)
    region = _ellipse(mouth_mid[0], (top + bottom) / 2, 0.38 * w, max((bottom - top) / 2, 1.0))
    return Face(reference_box(x, x + w, bottom, center_y, clamp_top=False), region, None,
                _turn(nose[0], min(eyes[:, 0]), max(eyes[:, 0])))


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside(face: Sequence[float], area: Sequence[float]) -> float:
    """The share of `face` that lies in `area`."""
    x0, y0, x1, y1 = max(face[0], area[0]), max(face[1], area[1]), min(face[2], area[2]), min(face[3], area[3])
    size = (face[2] - face[0]) * (face[3] - face[1])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / size if size > 0 else 0.0


class MuseTalkFaceDetector:
    """MediaPipe face landmarks on the CPU, with YuNet for the faces MediaPipe does not return.

    Call it with an RGB frame for the largest face (a Face or None), or use `faces` for all of them,
    largest first. YuNet is optional; without it a frame MediaPipe misses has no face."""

    def __init__(self, landmarker_path: str | Path, yunet_path: str | Path | None = None, num_faces: int = 1,
                 min_score: float = 0.6):
        from ..faces.landmarks import Landmarker
        from ..faces.recognition import load_yunet

        self.num_faces = num_faces
        self.min_score = min_score
        self._landmarker = Landmarker(landmarker_path, num_faces=num_faces)
        self._yunet = load_yunet(yunet_path) if yunet_path is not None and Path(yunet_path).is_file() else None

    def faces(self, frame: np.ndarray) -> List[Face]:
        found = [face for face in map(face_from_landmarks, self._landmarker.points(frame)) if face is not None]
        if self._yunet is not None and len(found) < self.num_faces:
            h, w = frame.shape[:2]
            self._yunet.setInputSize((w, h))
            _, rows = self._yunet.detect(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
            for row in rows if rows is not None else ():
                if row[14] < self.min_score:
                    continue
                detection = (row[0], row[1], row[0] + row[2], row[1] + row[3])
                if all(_iou(detection, face.box) < 0.3 for face in found):
                    face = face_from_yunet(row)
                    if face is not None:
                        found.append(face)
        return sorted(found, key=lambda face: face.box[0] - face.box[2])[:self.num_faces]

    def __call__(self, frame: np.ndarray) -> Optional[Face]:
        faces = self.faces(frame)
        return faces[0] if faces else None

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def follow_tracks(frames: Sequence[Sequence[Face]], anchors: Sequence[Box],
                  min_iou: float = 0.2) -> List[List[Optional[Face]]]:
    """Assign the faces of every frame to tracks that start at `anchors` (one box per track, e.g. the union
    box of a face track from `engine.models.faces.find_faces`). A track follows the face that overlaps its
    last box most; while it has no face yet, a face that lies mostly inside its anchor box. Each face goes
    to one track at most; a track without a match in a frame gets None there."""
    last: List[Optional[Box]] = [None] * len(anchors)
    result: List[List[Optional[Face]]] = [[] for _ in anchors]
    for faces in frames:
        pairs = []
        for t, anchor in enumerate(anchors):
            for f, face in enumerate(faces):
                score = _iou(last[t], face.box) if last[t] is not None else 0.0
                if score <= min_iou and _inside(face.box, anchor) > 0.5:
                    score = min_iou + 1e-3 * _inside(face.box, anchor)  # below any real overlap
                if score > min_iou:
                    pairs.append((score, t, f))
        taken_tracks, taken_faces = set(), set()
        chosen: dict[int, Face] = {}
        for _score, t, f in sorted(pairs, reverse=True):
            if t not in taken_tracks and f not in taken_faces:
                taken_tracks.add(t)
                taken_faces.add(f)
                chosen[t] = faces[f]
        for t in range(len(anchors)):
            face = chosen.get(t)
            result[t].append(face)
            if face is not None:
                last[t] = face.box
    return result


# ----------------------------------------------------------------------------------------------
# per-track planning: usable frames, gaps, smoothing, fades
# ----------------------------------------------------------------------------------------------

def usable(faces: Sequence[Optional[Face]], min_face: float, max_turn: float) -> List[Optional[Face]]:
    """None for faces too small (box width in pixels) or turned too far to lip-sync."""
    return [face if face is not None and face.box[2] - face.box[0] >= min_face and face.turn <= max_turn else None
            for face in faces]


def _mix(a: Face, b: Face, t: float) -> Face:
    def lerp(x, y):
        return None if x is None else (1 - t) * np.asarray(x, dtype=np.float64) + t * np.asarray(y, dtype=np.float64)

    return Face(tuple(float(v) for v in lerp(a.box, b.box)), lerp(a.region, b.region), lerp(a.hole, b.hole),
                (1 - t) * a.turn + t * b.turn)


def fill_gaps(faces: Sequence[Optional[Face]], max_gap: int) -> List[Optional[Face]]:
    """Bridge runs of at most `max_gap` missing faces between two faces that continue each other (see
    `Face.continues`) by interpolation (a detector that misses one blurred frame would otherwise make
    the mouth jump back for a frame). A gap between two different faces stays a gap."""
    faces = list(faces)
    if max_gap <= 0:
        return faces
    index = 0
    while index < len(faces):
        if faces[index] is not None:
            index += 1
            continue
        end = index
        while end < len(faces) and faces[end] is None:
            end += 1
        before, after = (faces[index - 1] if index else None), (faces[end] if end < len(faces) else None)
        if before is not None and after is not None and end - index <= max_gap and before.continues(after):
            for k in range(index, end):
                faces[k] = _mix(before, after, (k - index + 1) / (end - index + 1))
        index = end
    return faces


def smooth(faces: Sequence[Optional[Face]], window: int) -> List[Optional[Face]]:
    """Average box and polygons over a centred window of `window` frames, within runs of faces that
    continue each other frame to frame (a jump to another face ends the run on both sides).

    The reference crops every frame on its own landmarks; the landmark noise then moves the crop and the
    mask between frames and the mouth shimmers. A centred average does not lag behind a moving head."""
    faces = list(faces)
    radius = max(0, (window - 1) // 2)
    if radius == 0:
        return faces
    result: List[Optional[Face]] = []
    for i, face in enumerate(faces):
        if face is None:
            result.append(None)
            continue
        group = [face]
        for step in (-1, 1):
            j = i + step
            # chained frame to frame (j - step is the face before j on the walk out from i)
            while (abs(j - i) <= radius and 0 <= j < len(faces) and faces[j] is not None
                   and faces[j - step].continues(faces[j])):
                group.append(faces[j])
                j += step
        box = tuple(float(v) for v in np.mean([g.box for g in group], axis=0))
        region = np.mean([g.region for g in group], axis=0)
        hole = None if face.hole is None else np.mean([g.hole for g in group], axis=0)
        result.append(Face(box, region, hole, face.turn))
    return result


def strengths(valid: Sequence[bool], fade: int) -> np.ndarray:
    """Blend strength per frame: 0 where no face is used, rising over `fade` frames next to such a frame,
    so the mouth does not jump between the generated and the original picture. The ends of the clip
    are not fades: there is nothing to jump from."""
    valid = np.asarray(valid, dtype=bool)
    if fade <= 0 or valid.all():
        return valid.astype(np.float32)
    distance = np.full(len(valid), np.inf)
    last = -np.inf
    for i, ok in enumerate(valid):
        last = last if ok else i
        distance[i] = i - last
    last = np.inf
    for i in range(len(valid) - 1, -1, -1):
        last = last if valid[i] else i
        distance[i] = min(distance[i], last - i)
    return np.where(valid, np.minimum(1.0, distance / (fade + 1)), 0.0).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# crop, mask, paste
# ----------------------------------------------------------------------------------------------

def crop_box(face: Face, frame_shape: Sequence[int], margin: int, pad_outside: bool) -> Optional[IntBox]:
    """The integer crop box with the chin margin. With `pad_outside` it may leave the frame (the crop then
    repeats the edge pixels, so a face cut by the frame keeps the reference's proportions); without it,
    the reference clamps the box to the frame. None when nothing of it is in the frame."""
    height, width = frame_shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in face.box)
    y1 += int(margin)
    if not pad_outside:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 - x0 < 2 or y1 - y0 < 2 or x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
        return None
    return x0, y0, x1, y1


def crop(frame: np.ndarray, box: IntBox, size: int) -> np.ndarray:
    """The box resized to size x size with Lanczos (the reference's crop), edges repeated outside the frame."""
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    inner = frame[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
    if (x0, y0, x1, y1) != (max(0, x0), max(0, y0), min(width, x1), min(height, y1)):
        inner = cv2.copyMakeBorder(inner, max(0, -y0), max(0, y1 - height), max(0, -x0), max(0, x1 - width),
                                   cv2.BORDER_REPLICATE)
    return cv2.resize(np.ascontiguousarray(inner), (size, size), interpolation=cv2.INTER_LANCZOS4)


def _fill(shape: Sequence[int], polygon: np.ndarray, origin: Sequence[int]) -> np.ndarray:
    canvas = np.zeros(tuple(shape), np.uint8)
    points = np.round((np.asarray(polygon, dtype=np.float64) - np.asarray(origin, dtype=np.float64)) * 4)
    cv2.fillPoly(canvas, [points.astype(np.int32)], 255, lineType=cv2.LINE_8, shift=2)  # quarter-pixel vertices
    return canvas


def _grow(mask: np.ndarray, rx: int, ry: int) -> np.ndarray:
    if rx <= 0 and ry <= 0:
        return mask
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1)))


def lower_face_mask(face: Face, box: IntBox) -> np.ndarray:
    """uint8 mask (0/255) over the box: the region polygon, grown a little downwards, minus the nose.

    This replaces BiSeNet's parsing (skin, lips and mouth, nose left out). Unlike the parser it cannot
    leave out hair, a hand or a microphone in front of the mouth."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    mask = _grow(_fill((height, width), face.region, (x0, y0)),
                 round(JAW_SPREAD[0] * width), round(JAW_SPREAD[1] * height))
    if face.hole is not None:
        spread = round(NOSE_SPREAD * width)
        mask[_grow(_fill((height, width), face.hole, (x0, y0)), spread, spread) > 0] = 0
    return mask


def blend_square(box: IntBox) -> int:
    """Half side `s` of the reference's blend square (`get_crop_box` with expand 1.5)."""
    x0, y0, x1, y1 = box
    return int(max(x1 - x0, y1 - y0) // 2 * 1.5)


def blur_kernel(box: IntBox, feather: float = 1.0) -> int:
    """The reference's Gaussian kernel (odd, about 5 % of the blend square's side), scaled by `feather`."""
    return int(0.05 * feather * (2 * blend_square(box)) // 2 * 2) + 1


def blend_margin(box: Box, feather_scale: float = 1.0) -> int:
    """The smallest chin margin (pixels below the chin) at which the inset of `feather` spares the jaw.

    `lower_face_mask` grows the jaw JAW_SPREAD[1] of the crop height below the chin, where an opened jaw
    drops, and `feather(inset=True)` clears the bottom `blur_kernel // 2 + 1` rows so that the weight is
    zero at the box edge. The margin holds both (the last grown row falls into the cleared border; the
    chin row keeps a weight above 0.9 from 40 px faces up), otherwise the clearing eats into the chin: at
    a 4 % margin the chin row kept about 0.1, so the generated jaw could not drop below the original chin,
    which was half blended. Both terms grow with the crop, which grows with the margin, so this iterates
    to the fixed point; it converges because they grow by about a tenth of the margin. About 11 % of the
    box height at the default feather (MuseTalk trains with 0-20 px on faces of at least 200 px)."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    margin = 0
    for _ in range(64):
        crop = (x0, y0, x1, y1 + margin)
        need = round(JAW_SPREAD[1] * (crop[3] - crop[1])) + blur_kernel(crop, feather_scale) // 2 + 1
        if need <= margin:
            break
        margin = need
    return margin


def feather(raw: np.ndarray, box: IntBox, feather_scale: float = 1.0, inset: bool = True) -> np.ndarray:
    """Blend weights in [0, 1] over the box from a 0/255 mask of the box.

    As the reference: rows above the middle row of the blend square (the box centre) are cleared, and the
    rest is blurred as if the mask were zero outside the box. With `inset`, a border as wide as the blur
    radius (and the outer cheeks) is cleared first, so the weights reach zero before the box edge and the
    edge of the generated picture can never show; the chin margin must then be at least `blend_margin`
    or the cleared bottom rows reach the chin."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    mask = raw.astype(np.float32)
    mask[:max(0, (y0 + y1) // 2 - y0)] = 0
    kernel = blur_kernel(box, feather_scale)
    pad = kernel // 2 + 1
    if inset:
        side = max(pad, int(round(CHEEK_INSET * width)))
        mask[:, :side] = 0
        mask[:, max(0, width - side):] = 0
        mask[max(0, height - pad):] = 0
    canvas = np.zeros((height + 2 * pad, width + 2 * pad), np.float32)
    canvas[pad:pad + height, pad:pad + width] = mask
    if kernel > 1:
        canvas = cv2.GaussianBlur(canvas, (kernel, kernel), 0)
    return np.clip(canvas[pad:pad + height, pad:pad + width] / 255.0, 0.0, 1.0)


def paste(frame: np.ndarray, face: np.ndarray, box: IntBox, weights: np.ndarray, shrink_area: bool = True) -> None:
    """Blend the generated face (size x size RGB) into `frame` in place over the box with `weights`.

    The reference resizes with cv2's default (bilinear); `shrink_area` uses area averaging when the box is
    smaller than the face, which does not alias the 256 px detail into a small face. Pixels whose weight
    is 0 keep their exact value."""
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    shrinking = (x1 - x0) < face.shape[1] and (y1 - y0) < face.shape[0]
    method = cv2.INTER_AREA if shrink_area and shrinking else cv2.INTER_LINEAR
    resized = cv2.resize(face, (x1 - x0, y1 - y0), interpolation=method).astype(np.float32)
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    inner = (slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))
    weight = weights[inner][..., None]
    original = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    mixed = original * (1.0 - weight) + resized[inner] * weight
    frame[fy0:fy1, fx0:fx1] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


MaskFn = Callable[[Face, IntBox], np.ndarray]
