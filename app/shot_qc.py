# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The independent shot check (docs/QUALITY_ARCHITECTURE.md, layer 4).

Not the production agent: its own sampling, its own criteria, and a binding verdict. A clip (or a keyframe before it
goes to the video model) is checked against its ShotSpec (app/shot_spec.py) and the series bible (app/bible.py):

1. Frames: 2 per second, plus every 4th frame of the last 1.5 s (where Wan's endings go wrong: defect A), plus the
   first and the last frame.
2. Deterministic detectors on the CPU, from the cheapest:
   - backdrop: the picture around the characters is a flat studio backdrop (B),
   - drift: the colours of the shot move away from its keyframe / first frame (E),
   - palette: each cast member's body colour and accessories are there (D, E) and stay in the frame (J),
   - motion: the characters move (I),
   - framing: a character leaves the frame or is cut off by the camera (J).
3. Yes/no questions to the local vision model (llm.inspect_review) made from the spec: a stranger in the picture (A),
   the cast's look against their reference photos (C, D), each thing the narration needs (F, H), and broken physics
   or objects (F, G). Optional: a second opinion from Claude when the local verdict is uncertain (shot_arbiter).

The verdict is a QcVerdict (pass / fail / unchecked / needs_review) stored in the clip's record ("qc"); an older
record without a status is "unchecked", never a pass.
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from . import bible as bible_mod
from . import palette
from .shot_spec import ShotSpec

PASS, FAIL, UNCHECKED, NEEDS_REVIEW = "pass", "fail", "unchecked", "needs_review"
VERSION = 1
SIZE = (320, 180)              # detectors work on this frame size (16:9; other aspects are padded by resizing)
STEP_SECONDS = 0.5             # 2 frames a second
TAIL_SECONDS = 1.5             # ...and every TAIL_STEP-th frame of the last 1.5 s
TAIL_STEP = 4
MAX_RETRIES = 2                # new seeds for a shot that fails, before it needs the owner's review

# Thresholds, measured on tests/fixtures/quality (tools/qc_eval.py prints each detector's hits and false alarms).
BACKDROP_EDGE = 2.5            # a pixel whose blurred brightness changes by more than this (0-255) is an edge
BACKDROP_BAND = 0.32           # every band of the frame (top, middle, bottom) has fewer edges than this: a backdrop
BACKDROP_FRAMES = 0.5          # ...in at least this share of the frames
DRIFT_LIMIT = 0.3              # chroma distance of a frame from the reference (see _chroma)
DRIFT_FRAMES = 0.3
BODY_MIN = 0.0035              # a character's body colour covers at least this share of the frame when it is there
ACCESSORY_MIN = 0.0006         # an accessory at least this share
MISSING_FRAMES = 0.3           # a character (or its accessory) missing in more than this share of the frames
ACCESSORY_DROP = 0.25          # an accessory under this share of its keyframe/first-frame size is gone
MOTION_MIN = 0.12              # change of the characters' silhouettes (1 - overlap) between frames 0.5 s apart
MOTION_CONFIRM = 0.13          # the vision model's "they stand still" counts only under this silhouette change
                               # (still shots measured 0.06-0.10; a walk to a puddle and a lean over it, 0.12-0.2)
TAIL_LOSS = 0.25               # a character whose size in the tail falls under this share of its first half: gone
EDGE_CUT = 0.45                # ...or that touches the frame edge with under this share of its height left

VLM_SIZES = (896, 768)


@dataclass
class Check:
    name: str
    status: str                  # pass / fail / unchecked
    detail: str = ""
    defect: str = ""             # the defect letter it guards against (docs/QUALITY_ARCHITECTURE.md)
    frames: List[float] = field(default_factory=list)  # seconds of the frames that show the problem
    value: Optional[float] = None


@dataclass
class QcVerdict:
    status: str
    checks: List[Check] = field(default_factory=list)
    frames: List[float] = field(default_factory=list)  # the sampled seconds
    version: int = VERSION
    attempt: int = 0
    seconds: float = 0.0                               # how long the check took
    face: Optional[float] = None                       # the legacy face score, kept for older readers

    @property
    def ok(self) -> bool:
        return self.status == PASS

    @property
    def reasons(self) -> List[str]:
        return [f"{check.name}: {check.detail}" for check in self.checks if check.status == FAIL]

    def record(self) -> dict:
        """The "qc" record of a clip: the legacy fields (ok, reasons, face) and the verdict."""
        return {"ok": self.ok, "reasons": self.reasons, "face": self.face, "status": self.status,
                "checks": [asdict(check) for check in self.checks], "frames": [round(t, 3) for t in self.frames],
                "version": self.version, "attempt": self.attempt, "seconds": round(self.seconds, 2)}

    @classmethod
    def from_record(cls, record) -> "QcVerdict":
        if not isinstance(record, dict) or record.get("status") not in (PASS, FAIL, UNCHECKED, NEEDS_REVIEW):
            reasons = record.get("reasons") if isinstance(record, dict) else None
            checks = [Check("legacy", FAIL, str(reason)) for reason in reasons or []]
            return cls(FAIL if checks else UNCHECKED, checks)
        checks = [Check(**{key: row.get(key) for key in Check.__dataclass_fields__ if key in row})
                  for row in record.get("checks") or [] if isinstance(row, dict) and row.get("name")]
        return cls(record["status"], checks, list(record.get("frames") or []), int(record.get("version") or 0),
                   int(record.get("attempt") or 0), float(record.get("seconds") or 0.0), record.get("face"))


def status_of(record) -> str:
    """The status of a clip's "qc" record: pass, fail, needs_review or unchecked (an older record, or none)."""
    return QcVerdict.from_record(record).status


# --- frames -----------------------------------------------------------------------------------------------------

def sample_indices(count: int, fps: float) -> List[int]:
    """Frame numbers to check in a clip of `count` frames: every STEP_SECONDS, every TAIL_STEP-th frame of the last
    TAIL_SECONDS, and the first and the last frame."""
    if count <= 0:
        return []
    step = max(1, int(round(fps * STEP_SECONDS)))
    picked = set(range(0, count, step))
    tail_start = max(0, count - 1 - int(round(fps * TAIL_SECONDS)))
    picked.update(range(count - 1, tail_start - 1, -TAIL_STEP))
    picked.update((0, count - 1))
    return sorted(picked)


@dataclass
class Frame:
    seconds: float
    rgb: np.ndarray             # SIZE, uint8
    full: Optional[Image.Image] = None  # a larger copy for the vision model and the sheets

    @property
    def hsv(self) -> np.ndarray:
        cached = getattr(self, "_hsv", None)
        if cached is None:
            cached = palette.to_hsv(self.rgb)
            object.__setattr__(self, "_hsv", cached)
        return cached


def read_frames(path, *, full_width: int = 960) -> Tuple[List[Frame], float]:
    """The sampled frames of a clip and its length in seconds. Every frame is decoded (to know the count) but only
    the sampled ones are converted. `full` keeps the frame at up to `full_width` (the clip's own 848x480 stays as it
    is), so a zoomed crop still shows small details."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 30)
        frames = list(container.decode(stream))
    picked = sample_indices(len(frames), fps)
    out = []
    for index in picked:
        frame = frames[index]
        image = frame.to_image()
        rgb = np.asarray(image.resize(SIZE, Image.BILINEAR), dtype=np.uint8)
        if image.width > full_width:
            image = image.resize((full_width, max(1, round(full_width * image.height / image.width))), Image.BILINEAR)
        out.append(Frame(index / fps, rgb, image))
    return out, len(frames) / fps if fps else 0.0


def still_frame(path) -> Frame:
    with Image.open(path) as image:
        image = image.convert("RGB")
        rgb = np.asarray(image.resize(SIZE, Image.BILINEAR), dtype=np.uint8)
        full = image.resize((480, max(1, round(480 * image.height / image.width))), Image.BILINEAR)
    return Frame(0.0, rgb, full)


# --- deterministic detectors ------------------------------------------------------------------------------------

def _character_mask(frame: Frame, cast: Sequence[bible_mod.CharacterSpec]) -> np.ndarray:
    hit = np.zeros(frame.rgb.shape[:2], dtype=bool)
    for member in cast:
        hit |= palette.mask(frame.hsv, member.body_ranges)
        for accessory in member.must_have:
            hit |= palette.mask(frame.hsv, accessory.ranges)
    return hit


def backdrop_share(frame: Frame, cast: Sequence[bible_mod.CharacterSpec]) -> Tuple[float, float, float]:
    """(top, middle, bottom) share of edge pixels around the characters. A painted place has edges everywhere (leaves,
    grass, furniture); a studio backdrop has almost none in every band, even the floor (defect B)."""
    from PIL import ImageFilter
    from scipy.ndimage import binary_dilation

    gray = np.asarray(Image.fromarray(frame.rgb).convert("L").filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32)
    edges = np.hypot(np.abs(np.diff(gray, axis=1))[:-1, :], np.abs(np.diff(gray, axis=0))[:, :-1]) > BACKDROP_EDGE
    around = ~binary_dilation(_character_mask(frame, cast)[:-1, :-1], iterations=6)
    rows = edges.shape[0]
    bands = []
    for lo, hi in ((0, rows // 3), (rows // 3, 2 * rows // 3), (2 * rows // 3, rows)):
        chosen = around[lo:hi]
        bands.append(float(edges[lo:hi][chosen].mean()) if chosen.any() else 1.0)
    return bands[0], bands[1], bands[2]


def check_backdrop(frames: Sequence[Frame], cast) -> Check:
    flagged = [frame.seconds for frame in frames if max(backdrop_share(frame, cast)) < BACKDROP_BAND]
    share = len(flagged) / max(1, len(frames))
    if share >= BACKDROP_FRAMES:
        return Check("backdrop", FAIL, f"plain studio-like backdrop around the characters in {share:.0%} of the "
                     "frames", "B", flagged[:6], round(share, 3))
    return Check("backdrop", PASS, "", "B", value=round(share, 3))


def _chroma(frame: Frame) -> np.ndarray:
    """The frame's mean colour as two opponent chroma values (red-green, yellow-blue), 0-1 scale."""
    rgb = frame.rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.array([float((r - g).mean()), float(((r + g) / 2 - b).mean())])


def check_drift(frames: Sequence[Frame], reference: Optional[Frame] = None) -> Check:
    if not frames:
        return Check("color_drift", UNCHECKED, "no frames", "E")
    base = _chroma(reference if reference is not None else frames[0])
    distances = [(frame.seconds, float(np.linalg.norm(_chroma(frame) - base))) for frame in frames]
    far = [t for t, d in distances if d > DRIFT_LIMIT]
    worst = max(d for _, d in distances)
    if len(far) / len(frames) > DRIFT_FRAMES:
        return Check("color_drift", FAIL, f"the colours drift away from the {'keyframe' if reference else 'first frame'}"
                     f" (distance {worst:.3f} > {DRIFT_LIMIT})", "E", far[:6], round(worst, 4))
    return Check("color_drift", PASS, "", "E", value=round(worst, 4))


@dataclass
class _Presence:
    body: List[float]
    accessories: Dict[str, List[float]]
    boxes: List[Optional[Tuple[int, int, int, int]]]


def _presence(frames: Sequence[Frame], member: bible_mod.CharacterSpec) -> _Presence:
    body, boxes = [], []
    accessories: Dict[str, List[float]] = {one.name: [] for one in member.must_have}
    for frame in frames:
        body_mask = palette.mask(frame.hsv, member.body_ranges) if member.body_ranges else None
        body.append(float(body_mask.mean()) if body_mask is not None else float("nan"))
        whole = body_mask.copy() if body_mask is not None else np.zeros(frame.rgb.shape[:2], dtype=bool)
        for accessory in member.must_have:
            hit = palette.mask(frame.hsv, accessory.ranges)
            accessories[accessory.name].append(float(hit.mean()))
            whole |= hit
        boxes.append(palette.largest_blob_box(whole))
    return _Presence(body, accessories, boxes)


def check_palette(frames: Sequence[Frame], cast: Sequence[bible_mod.CharacterSpec],
                  reference: Optional[Frame] = None) -> List[Check]:
    """Each cast member's body colour and accessories in the frames (defects D, E), relative to the reference frame
    when there is one (a keyframe that passed its check)."""
    checks = []
    for member in cast:
        found = _presence(frames, member)
        known = [value for value in found.body if not math.isnan(value)]
        if known:
            missing = [frame.seconds for frame, value in zip(frames, found.body) if value < BODY_MIN]
            if len(missing) / len(frames) > MISSING_FRAMES:
                checks.append(Check(f"{member.slug}_colour", FAIL,
                                    f"{member.name}'s {member.body} body is missing or another colour in "
                                    f"{len(missing)}/{len(frames)} frames", "D", missing[:6],
                                    round(float(np.median(known)), 5)))
            else:
                checks.append(Check(f"{member.slug}_colour", PASS, "", "D", value=round(float(np.median(known)), 5)))
        for accessory in member.must_have:
            shares = found.accessories[accessory.name]
            base = None
            if reference is not None:
                base = palette.share(reference.hsv, accessory.ranges)
            elif shares:
                base = max(shares[:3])
            floor = max(ACCESSORY_MIN, ACCESSORY_DROP * base) if base else ACCESSORY_MIN
            gone = [frame.seconds for frame, value in zip(frames, shares) if value < floor]
            name = f"{member.slug}_{accessory.name.replace(' ', '_')}"
            if len(gone) / max(1, len(frames)) > MISSING_FRAMES:
                checks.append(Check(name, FAIL, f"{member.name}'s {accessory.color} {accessory.name} is missing or "
                                    f"changed colour in {len(gone)}/{len(frames)} frames", "E", gone[:6],
                                    round(float(np.median(shares)), 5)))
            else:
                checks.append(Check(name, PASS, "", "E", value=round(float(np.median(shares)), 5)))
    return checks


def _silhouette(frame: Frame, member: bible_mod.CharacterSpec) -> Optional[np.ndarray]:
    """The character's colour mask inside its own box, scaled to 32x32 (so a camera move, which shifts and scales the
    box, does not count as the character moving)."""
    hit = palette.mask(frame.hsv, member.body_ranges)
    for accessory in member.must_have:
        hit |= palette.mask(frame.hsv, accessory.ranges)
    box = palette.largest_blob_box(hit)
    if box is None or box[2] - box[0] < 6 or box[3] - box[1] < 6:
        return None
    piece = Image.fromarray((hit[box[1]:box[3], box[0]:box[2]] * 255).astype(np.uint8)).resize((32, 32))
    return np.asarray(piece) > 127


def character_motion(frames: Sequence[Frame], cast: Sequence[bible_mod.CharacterSpec]) -> Optional[float]:
    """The median change of the characters' silhouettes between frames STEP_SECONDS apart (1 - their overlap): a
    character that walks, turns or reaches changes its silhouette; one that stands still facing the camera does not
    (defect I). None when no character is found."""
    regular = [frame for frame in frames if abs(frame.seconds / STEP_SECONDS - round(frame.seconds / STEP_SECONDS))
               < 0.07]
    changes = []
    for member in cast:
        shapes = [_silhouette(frame, member) for frame in regular]
        for a, b in zip(shapes, shapes[1:]):
            if a is None or b is None:
                continue
            union = int((a | b).sum())
            changes.append(1.0 - int((a & b).sum()) / union if union else 0.0)
    return float(np.median(changes)) if changes else None


def check_motion(frames: Sequence[Frame], cast) -> Check:
    if not cast:
        return Check("motion", UNCHECKED, "no characters to follow", "I")
    value = character_motion(frames, cast)
    if value is None:
        return Check("motion", UNCHECKED, "the characters were not found", "I")
    if value < MOTION_MIN:
        return Check("motion", FAIL, f"the characters barely move (silhouette change {value:.2f} < {MOTION_MIN})", "I",
                     value=round(value, 2))
    return Check("motion", PASS, "", "I", value=round(value, 2))


def check_framing(frames: Sequence[Frame], cast: Sequence[bible_mod.CharacterSpec]) -> List[Check]:
    """A character that leaves the frame or is cut off by its edge in the shot's second half (defect J)."""
    checks = []
    if len(frames) < 4:
        return checks
    half = frames[-1].seconds / 2
    for member in cast:
        found = _presence(frames, member)
        sizes = [((box[2] - box[0]) * (box[3] - box[1]) if box else 0, box) for box in found.boxes]
        early = [size for (size, _), frame in zip(sizes, frames) if frame.seconds <= half and size]
        if not early:
            continue
        typical = float(np.median(early))
        heights = [box[3] - box[1] for (_, box), frame in zip(sizes, frames) if box and frame.seconds <= half]
        height = float(np.median(heights)) if heights else 0.0
        lost = []
        for (size, box), frame in zip(sizes, frames):
            if frame.seconds <= half:
                continue
            if size < TAIL_LOSS * typical:
                lost.append(frame.seconds)
            elif box and height and (box[3] >= SIZE[1] - 1 or box[1] <= 0 or box[0] <= 0 or box[2] >= SIZE[0] - 1) \
                    and (box[3] - box[1]) < EDGE_CUT * height:
                lost.append(frame.seconds)
        late = [frame for frame in frames if frame.seconds > half]
        if late and len(lost) / len(late) > 0.4:
            checks.append(Check(f"{member.slug}_framing", FAIL, f"{member.name} leaves the frame or is cut off by "
                                "the camera in the second half of the shot", "J", lost[:6]))
        else:
            checks.append(Check(f"{member.slug}_framing", PASS, "", "J"))
    return checks


def text_box_overlap(frames: Sequence[Frame], cast: Sequence[bible_mod.CharacterSpec],
                     box: Tuple[float, float, float, float]) -> float:
    """The largest share of a text box (x0, y0, x1, y1 as fractions of the frame) that the characters cover in the
    frames (defect K: a title burned over a face)."""
    worst = 0.0
    for frame in frames:
        hit = _character_mask(frame, cast)
        h, w = hit.shape
        x0, y0, x1, y1 = int(box[0] * w), int(box[1] * h), int(math.ceil(box[2] * w)), int(math.ceil(box[3] * h))
        region = hit[y0:y1, x0:x1]
        if region.size:
            worst = max(worst, float(region.mean()))
    return worst


# --- the vision model -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Question:
    id: str
    text: str
    good: str           # answer A: what a right shot shows
    bad: str            # answer B: the defect
    defect: str


def questions(spec: Optional[ShotSpec], cast: Sequence[bible_mod.CharacterSpec], bible: Optional[bible_mod.Bible],
              *, still: bool = False) -> List[Question]:
    """The questions of a shot, made from its spec and the bible (nothing the production agent wrote is taken as a
    verdict). Each has two described answers, A (right) and B (the defect): a small vision model answered yes/no
    questions with negations in them backwards ("Is Lumi the only being, with no human?" -> "no: no humans"),
    measured in the test episode of 2026-11-22."""
    names = " and ".join(member.name for member in cast)
    if bible is not None and bible.only_named_cast:
        out = [Question("only_cast", "Who appears in the frames and zoom crops?",
                        f"only {names} from the reference pictures" if cast else "no living being at all",
                        "also a human, a child, or another animal or creature", "A")]
    else:
        out = [Question("only_cast", "Does a stranger appear (in a doorway, at the edge, in the last frames)?",
                        "nobody unexpected appears", "a stranger appears", "A")]
    for member in cast:
        out.append(Question(f"{member.slug}_identity", f"How does {member.name} in the crops of picture 2 compare "
                            "with its REFERENCE?", "the same body shape and the same parts",
                            "different: an extra part or item it does not have in its REFERENCE (a shell, wings, "
                            "a patch on its back, glasses, a scarf, clothes) or another shape", "C"))
        for accessory in member.must_have:
            out.append(Question(f"{member.slug}_{accessory.name.replace(' ', '_')}",
                                f"What does {member.name} wear in the frames?",
                                f"the {accessory.color} {accessory.name} in every frame where {member.name} is visible",
                                f"the {accessory.name} is missing, another colour, or comes and goes, or "
                                f"{member.name} wears something its reference does not", "D"))
        if member.body:
            out.append(Question(f"{member.slug}_body_colour", f"What colour is {member.name}'s body in the frames?",
                                f"{member.body}, as in its reference", "grey or another colour in some frames", "D"))
    if spec is not None:
        for number, thing in enumerate([*spec.key_visual, *spec.must_show], 1):
            out.append(Question(f"shows_{number}", f"Is this in the shot: {thing}?", "clearly visible",
                                "missing or hard to make out", "F"))
        if not still:
            for number, thing in enumerate(spec.key_visual, 1):
                out.append(Question(f"stays_{number}", f"How does this look over the shot: {thing}?",
                                    "the same shape and look throughout",
                                    "it turns into something else or into a smear", "G"))
        for number, thing in enumerate(spec.must_not_show, 1):
            out.append(Question(f"hides_{number}", f"Is this anywhere inside the pictures of the shot (the "
                                f"inspector's yellow captions do not count): {thing}?", "nowhere", "it is visible",
                                "A"))
        if bible is not None:
            place = bible.place(spec.location)
            if place is not None and place.plate:
                out.append(Question("same_place", f"Is this the place in the REFERENCE picture of {place.name}?",
                                    "the same place (the same buildings, materials and landmarks)",
                                    "a different place", "H"))
    out.append(Question("physics", "Is anything physically impossible in the frames?", "everything is plausible",
                        "a character stands on water or floats without wings, or something passes through a body "
                        "or a face", "F"))
    if not still and cast:
        out.append(Question("action", "What do the characters do in the shot?",
                            "a clear visible action (walk, turn, reach, fly...)",
                            "they just stand still facing the camera", "I"))
    if not still:
        out.append(Question("stable", "Does anything transform during the shot?", "nothing changes into something else",
                            "something turns into something else, dissolves into noise, or appears from nowhere", "G"))
    return out


_ANSWER = {"type": "object", "properties": {
    "answers": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "answer": {"type": "string", "enum": ["A", "B", "unsure"]},
        "reason": {"type": "string"}}, "required": ["id", "answer", "reason"]}}},
    "required": ["answers"]}


def contact_sheet(frames: Sequence[Frame], references: Sequence[Tuple[str, str]] = (), *, columns: int = 4,
                  cell: int = 300, labels: bool = True, zooms: Sequence[Tuple[str, Image.Image]] = ()) -> Image.Image:
    """One picture: the reference pictures (label, path) on the first row, then the frames with their times, then
    the zoomed crops (hotspots)."""
    rows = []
    if references:
        refs = []
        for label, path in references:
            try:
                with Image.open(path) as image:
                    picture = image.convert("RGB")
                    picture.thumbnail((cell, int(cell * 0.75)))
                    refs.append((f"REFERENCE: {label}", picture.copy()))
            except OSError:
                continue
        if refs:
            rows.append(refs)
    cells = [(f"{frame.seconds:.2f} s", (frame.full or Image.fromarray(frame.rgb)).copy()) for frame in frames]
    for start in range(0, len(cells), columns):
        rows.append(cells[start:start + columns])
    if zooms:
        rows.append([(label, image) for label, image in zooms][:columns])
    height = int(cell * 9 / 16)
    row_heights = []
    for row in rows:
        row_heights.append(max(min(int(cell * 0.75), int(cell * img.height / max(1, img.width))) for _, img in row)
                           + (16 if labels else 0))
    sheet = Image.new("RGB", (columns * cell, sum(row_heights) or height), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    top = 0
    for row, row_height in zip(rows, row_heights):
        for column, (label, image) in enumerate(row):
            picture = image.copy()
            picture.thumbnail((cell - 4, row_height - (16 if labels else 0)))
            sheet.paste(picture, (column * cell + 2, top + (16 if labels else 0)))
            if labels:
                draw.text((column * cell + 4, top + 2), label, fill=(255, 230, 120))
        top += row_height
    return sheet


HOTSPOT_DIFF = 22.0             # brightness change (0-255, blurred) that counts as something new
HOTSPOT_MIN = 0.0012           # ...over at least this share of the frame
ZOOM_SIDE = 0.34               # a zoom shows at least this share of the frame's width


def hotspots(frames: Sequence[Frame], cast: Sequence[bible_mod.CharacterSpec], limit: int = 2) -> List[Tuple[str, Image.Image]]:
    """Zoomed crops of what moves in the last TAIL_SECONDS outside the known characters: a small stranger walking in
    a doorway (defect A) is a few dozen pixels in a whole frame, too small for the vision model on a sheet. The tail
    frames are TAIL_STEP frames apart, so a slow camera move changes little between two of them while something that
    walks or appears does; the changes are added up over the tail. Each crop is (label, picture) of the last frame
    around the busiest spot."""
    from PIL import ImageFilter
    from scipy.ndimage import binary_dilation, label

    end = frames[-1].seconds if frames else 0.0
    tail = [frame for frame in frames if frame.seconds >= end - TAIL_SECONDS]
    if len(tail) < 3:
        return []

    def gray(frame: Frame) -> np.ndarray:
        return np.asarray(Image.fromarray(frame.rgb).convert("L").filter(ImageFilter.GaussianBlur(0.8)),
                          dtype=np.float32)

    grays = [gray(frame) for frame in tail]
    busy = np.zeros(grays[0].shape, dtype=np.float32)
    for frame, before, after in zip(tail[1:], grays, grays[1:]):
        moved = np.abs(after - before) > HOTSPOT_DIFF
        moved &= ~binary_dilation(_character_mask(frame, cast), iterations=3)
        busy += moved
    spots = binary_dilation(busy >= 2, iterations=3)
    labels, count = label(spots)
    if not count:
        return []
    sizes = np.bincount(labels.ravel())[1:]
    order = [int(i) + 1 for i in np.argsort(-sizes) if sizes[i] / spots.size >= HOTSPOT_MIN][:limit]
    out = []
    last = tail[-1]
    picture = last.full or Image.fromarray(last.rgb)
    scale = picture.width / SIZE[0]
    for number in order:
        ys, xs = np.nonzero(labels == number)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        cx, cy = (x0 + x1) / 2 * scale, (y0 + y1) / 2 * scale
        half_w = max((x1 - x0) * scale * 0.7, ZOOM_SIDE * picture.width / 2)
        half_h = half_w * 9 / 16
        box = (max(0, int(cx - half_w)), max(0, int(cy - half_h)), min(picture.width, int(cx + half_w)),
               min(picture.height, int(cy + half_h)))
        out.append((f"ZOOM {last.seconds:.2f} s", picture.crop(box)))
    return out


def _encode(image: Image.Image, quality: int = 82) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def character_sheet(frames: Sequence[Frame], cast: Sequence[bible_mod.CharacterSpec], *, per_character: int = 4,
                    cell: int = 260) -> Optional[Image.Image]:
    """One picture per call for the identity questions (defects C, D): each character's reference photo followed by
    crops of that character from frames spread over the shot (its last frames included), found by its colours. A
    shell on the back or a grey owl is much plainer in a crop beside the reference than in a whole frame."""
    rows = []
    for member in cast:
        crops = []
        found = _presence(frames, member)
        picked = [i for i, box in enumerate(found.boxes) if box is not None]
        if not picked:
            continue
        chosen = [picked[round(k * (len(picked) - 1) / max(1, per_character - 1))] for k in range(per_character)]
        for index in dict.fromkeys(chosen):
            frame, box = frames[index], found.boxes[index]
            picture = frame.full or Image.fromarray(frame.rgb)
            scale = picture.width / SIZE[0]
            x0, y0, x1, y1 = box
            pad_x, pad_y = (x1 - x0) * 0.35 + 4, (y1 - y0) * 0.35 + 4
            crop = picture.crop((max(0, int((x0 - pad_x) * scale)), max(0, int((y0 - pad_y) * scale)),
                                 min(picture.width, int((x1 + pad_x) * scale)),
                                 min(picture.height, int((y1 + pad_y) * scale))))
            crops.append((f"{member.name} {frame.seconds:.1f} s", crop))
        refs = []
        if member.photo:
            try:
                with Image.open(member.photo) as image:
                    refs.append((f"REFERENCE {member.name}", image.convert("RGB").copy()))
            except OSError:
                pass
        rows.append(refs + crops)
    if not rows:
        return None
    columns = max(len(row) for row in rows)
    sheet = Image.new("RGB", (columns * cell, len(rows) * (cell + 16)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        for c, (label, image) in enumerate(row):
            picture = image.copy()
            picture.thumbnail((cell - 4, cell - 4))
            sheet.paste(picture, (c * cell + 2, r * (cell + 16) + 16))
            draw.text((c * cell + 4, r * (cell + 16) + 2), label, fill=(255, 230, 120))
    return sheet


def ask_vision(sheet_path: Path, asked: Sequence[Question], prompt_intro: str = "",
               extra: Sequence[Path] = ()) -> Dict[str, Tuple[str, str]]:
    """The local vision model's answers {id: (yes/no/unsure, reason)} for one contact sheet (and `extra` pictures:
    the character sheet). Short answers keep the call fast: the model's output, not the pictures, is most of its
    time. Raises llm.VisualReviewUnavailable when no valid answer came back."""
    from . import llm

    listing = "\n".join(f"- {one.id}: {one.text} A = {one.good}. B = {one.bad}." for one in asked)
    prompt = ((prompt_intro or "You are a strict quality inspector for a children's animated series. ")
              + "Picture 1 shows REFERENCE pictures on its first row (labelled), then frames of one shot in time "
              "order (labelled with seconds), and maybe ZOOM crops: enlarged parts of the last frame where something "
              "moves." + (" Picture 2 shows each character's REFERENCE picture followed by crops of that character "
                          "from the shot: compare them closely (body shape, colours, extra parts, what it wears)."
                          if extra else "")
              + " The yellow captions (times, REFERENCE, ZOOM, names) are written by the inspector on the sheet: "
              "ignore them, they are never text in the shot. Judge only what the pictures show. Look at every frame "
              "carefully, including the small details, the zoom crops and the last frames. "
              "For each question answer A or B (the one that describes what you see), or unsure; the reason is at "
              "most 6 words.\n" + listing)

    def valid(result: dict) -> bool:
        answers = result.get("answers")
        return (isinstance(answers, list) and {row.get("id") for row in answers if isinstance(row, dict)}
                >= {one.id for one in asked})

    result = llm.inspect_review([str(sheet_path), *map(str, extra)], prompt, 80 + 36 * len(asked), _ANSWER, valid,
                                VLM_SIZES)
    return {row["id"]: (str(row.get("answer", "unsure")).strip().upper()[:1] if str(row.get("answer", "")).strip()
                        .upper()[:1] in ("A", "B") else "unsure", str(row.get("reason", ""))[:200])
            for row in result["answers"] if isinstance(row, dict) and row.get("id")}


def judge_answers(asked: Sequence[Question], answers: Dict[str, Tuple[str, str]]) -> Tuple[List[Check], List[str]]:
    """Checks from the answers, and the ids answered "unsure" (for the arbiter)."""
    checks, unsure = [], []
    for one in asked:
        answer, reason = answers.get(one.id, ("unsure", "no answer"))
        if answer == "unsure":
            unsure.append(one.id)
            checks.append(Check(f"vision_{one.id}", UNCHECKED, f"unsure: {reason}", one.defect))
            continue
        good = answer == "A"
        checks.append(Check(f"vision_{one.id}", PASS if good else FAIL,
                            "" if good else f"{one.text} -> {one.bad} ({reason})", one.defect))
    return checks, unsure


# --- the verdict ------------------------------------------------------------------------------------------------

@dataclass
class Context:
    """What a check knows about a shot."""
    cast: Tuple[bible_mod.CharacterSpec, ...] = ()
    spec: Optional[ShotSpec] = None
    bible: Optional[bible_mod.Bible] = None
    reference: Optional[Path] = None        # the keyframe the clip started from (it passed its own check)
    vision: bool = True
    arbiter: Optional[Callable[[Path, Sequence[Question], Dict[str, Tuple[str, str]]], Dict[str, Tuple[str, str]]]] \
        = None
    work: Optional[Path] = None             # where the contact sheet is written


def _references(context: Context) -> List[Tuple[str, str]]:
    found = []
    for member in context.cast:
        if member.photo:
            found.append((member.name, member.photo))
    if context.spec is not None and context.bible is not None:
        place = context.bible.place(context.spec.location)
        if place is not None and place.plate:
            found.append((place.name, place.plate))
    return found[:4]


def combine(checks: Sequence[Check]) -> str:
    if any(check.status == FAIL for check in checks):
        return FAIL
    if not checks or any(check.status == UNCHECKED and (check.name == "vision" or check.name.startswith("vision_"))
                         for check in checks):
        return UNCHECKED
    return PASS


def _run(frames: List[Frame], duration: float, context: Context, *, still: bool) -> QcVerdict:
    started = time.monotonic()
    cast = list(context.cast)
    reference = still_frame(context.reference) if context.reference and Path(context.reference).is_file() else None
    checks: List[Check] = [check_backdrop(frames, cast)]
    if not still:
        checks.append(check_drift(frames, reference))
    checks += check_palette(frames, cast, reference)
    if not still:
        checks.append(check_motion(frames, cast))
        checks += check_framing(frames, cast)
    if context.vision:
        asked = questions(context.spec, cast, context.bible, still=still)
        work = context.work or Path(".")
        work.mkdir(parents=True, exist_ok=True)
        sheet_path = work / ("qc_still.jpg" if still else "qc_sheet.jpg")
        chosen = frames if still else _vision_frames(frames)
        zooms = [] if still else hotspots(frames, cast)
        contact_sheet(chosen, _references(context), zooms=zooms).save(sheet_path, quality=88)
        extra = []
        people = character_sheet(frames if not still else frames, cast) if cast else None
        if people is not None:
            extra.append(work / ("qc_cast_still.jpg" if still else "qc_cast.jpg"))
            people.save(extra[0], quality=88)
        try:
            answers = ask_vision(sheet_path, asked, extra=extra)
        except Exception as error:  # noqa: BLE001 - no verdict is "unchecked", never a pass
            checks.append(Check("vision", UNCHECKED, f"the vision check did not answer: {str(error)[:160]}", ""))
        else:
            found, unsure = judge_answers(asked, answers)
            if unsure and context.arbiter is not None:
                try:
                    second = context.arbiter(sheet_path, [one for one in asked if one.id in unsure], answers,
                                             extra=extra)
                except Exception as error:  # noqa: BLE001 - the arbiter is optional
                    second = {}
                    checks.append(Check("arbiter", UNCHECKED, f"the second opinion failed: {str(error)[:160]}"))
                answers.update(second)
                found, unsure = judge_answers(asked, answers)
            checks += _fuse_motion(found, checks)
    status = combine(checks)
    return QcVerdict(status, checks, [frame.seconds for frame in frames], seconds=time.monotonic() - started)


def _fuse_motion(found: List[Check], cpu: Sequence[Check]) -> List[Check]:
    """The vision model takes a character walking towards the camera for one standing still (test episode,
    2026-11-22): its "they just stand still" counts only when the measured silhouette change agrees."""
    motion = next((check for check in cpu if check.name == "motion" and check.value is not None), None)
    out = []
    for check in found:
        if check.name == "vision_action" and check.status == FAIL and motion is not None                 and motion.value >= MOTION_CONFIRM:
            check = Check(check.name, PASS, f"overruled: the characters' silhouettes change {motion.value:.2f} between "
                          "frames", check.defect)
        out.append(check)
    return out


def _vision_frames(frames: Sequence[Frame], limit: int = 12) -> List[Frame]:
    """The frames the vision model sees: spread over the shot, with the last 1.5 s always in (defect A)."""
    if len(frames) <= limit:
        return list(frames)
    end = frames[-1].seconds
    tail = [frame for frame in frames if frame.seconds >= end - TAIL_SECONDS][-4:]
    tail_ids = {id(frame) for frame in tail}
    body = [frame for frame in frames if id(frame) not in tail_ids]
    count = limit - len(tail)
    picked = {id(body[round(i * (len(body) - 1) / max(1, count - 1))]) for i in range(count)} | tail_ids
    return [frame for frame in frames if id(frame) in picked]


def check_clip(path, context: Context) -> QcVerdict:
    frames, duration = read_frames(path)
    if not frames:
        return QcVerdict(FAIL, [Check("read", FAIL, "the clip has no frames")])
    return _run(frames, duration, context, still=False)


def check_still(path, context: Context) -> QcVerdict:
    """A keyframe before it goes to the video model: the backdrop, the palette and the vision questions."""
    return _run([still_frame(path)], 0.0, context, still=True)


# --- report helpers ---------------------------------------------------------------------------------------------

def sheet_data_uri(path, context: Optional[Context] = None, *, columns: int = 6) -> str:
    """A JPEG contact sheet of a clip's checked frames as a data: URI (the QC report)."""
    frames, _ = read_frames(path, full_width=320)
    image = contact_sheet(frames, _references(context) if context else (), columns=columns, cell=220)
    return "data:image/jpeg;base64," + base64.b64encode(_encode(image, 70)).decode("ascii")


def summary(verdict: QcVerdict) -> str:
    if verdict.status == PASS:
        return "passed"
    failing = [check for check in verdict.checks if check.status == FAIL]
    if failing:
        return "; ".join(f"[{check.defect or '?'}] {check.detail or check.name}" for check in failing[:5])
    return verdict.status


def to_json(verdict: QcVerdict) -> str:
    return json.dumps(verdict.record(), ensure_ascii=False)
