# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Face identity checks for pictures of characters (YuNet detection, SFace recognition).

A character's reference faces come from its reference photos (app/characters.py). The scores tell
which face on a picture belongs to which character (lip sync, the views of the character creator)
and whether a face still is the character's: a start frame, and the frame a video's next part starts
from (its drift is reported, with the "qc" preference on). Everything here runs on the CPU and never
loads the assistant. A weak face is reported in a note: no file is moved or hidden.

SFace was trained on photos of real faces, so the faces of a 3D animated or drawn character score differently: each
style (app/styles.py) has its own thresholds (FACE_RULES, measured on faces of that style), and a check that
involves several styles (a realistic character in an animated film) uses the most lenient of them.

An animal or creature character (characters.is_creature) is never face-detected, scored or compared: it has no
reference faces, every check, score and match leaves it out, and its style sets no threshold.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import characters, config, model_manager, prefs, styles

MODEL_ID = "face-check"
FACE_MIN = 0.40
FACE_MATCH = 0.35    # a face counts as a character's face from this similarity (SFace cosine)
MATCH_MARGIN = 0.05  # a face that fits two characters this closely is left unassigned


@dataclass(frozen=True)
class FaceRules:
    """The SFace thresholds of a style: `minimum` - a generated view or a video frame below it shows another face
    (FACE_MIN for real faces); `match` - from it a face counts as the character's (lip sync, uploads; FACE_MATCH).
    `detects` - whether YuNet finds the faces of the style at all: it finds 3D animated faces, but no 2D anime face
    (0 of the anchors drawn on 2026-11-15, animation_samples/debug), so a 2D character's photos are not counted or
    compared by face (`characters.check_photo`, `char_photos._view`)."""
    minimum: float
    match: float
    detects: bool = True


FACE_RULES: Dict[str, FaceRules] = {
    styles.REALISTIC: FaceRules(FACE_MIN, FACE_MATCH),
    # Measured on the creator's 3D characters (2026-11-15, animation_samples/sface_measurements.json, 2 characters x 4
    # photos): the same character scores 0.81-0.88, two different characters up to 0.58 (mean 0.51): cartoon faces look
    # alike to SFace, so the realistic thresholds would take another 3D character for this one.
    styles.ANIMATION_3D: FaceRules(0.70, 0.65),
    # 2D anime: YuNet found no face on any first photo drawn by Z-Image (3 of 3, 2026-11-15), and on Qwen-Image-Edit's
    # views of one character it did, scoring only 0.46-0.72 against each other: no threshold can be set on that.
    styles.ANIME: FaceRules(FACE_MIN, FACE_MATCH, detects=False),
    # 2D cartoon: on two characters drawn by the creator (4 photos each, 2026-11-16) YuNet found the face on all of one
    # and on none of the other (a mustached cartoon man), and SFace scored the same character 0.12-0.59: as for Anime,
    # the vision model's adult check decides and faces are not compared. The starting point of custom styles with no
    # add-on ("Something else") is drawn too, and follows the same rule.
    styles.CARTOON: FaceRules(FACE_MIN, FACE_MATCH, detects=False),
    styles.OTHER: FaceRules(FACE_MIN, FACE_MATCH, detects=False),
}


def rules(*style_ids) -> FaceRules:
    """The thresholds for faces of these styles (None / "" = Realistic; a custom style follows its starting point):
    the most lenient of them, so a character drawn in one style and shown in a film of another is not called someone
    else for the style change alone."""
    found = [FACE_RULES[styles.of(style).family] for style in style_ids] or [FACE_RULES[styles.REALISTIC]]
    return FaceRules(min(r.minimum for r in found), min(r.match for r in found), all(r.detects for r in found))


def _people(chars) -> List[dict]:
    """The characters whose faces can be compared: an animal or creature (characters.is_creature) never is."""
    return [ch for ch in chars if not characters.is_creature(ch)]


def _styles_of(chars) -> List[str]:
    """The styles of the characters' own faces (rules); an animal's or creature's style sets no threshold."""
    return [ch.get("style") or styles.REALISTIC for ch in _people(chars) if isinstance(ch, dict)]
_IMAGES = (".png", ".jpg", ".jpeg", ".webp")


@dataclass
class Verdict:
    """What a face identity check found. ok is False for a face that is measurably someone else's,
    a character missing from a picture that must show it, or a clip that could not be read; a face
    that was not measured (no check asked for, no face visible) leaves it True.

    status - set for a finished clip's record: "unchecked" when nothing looked at the clip itself (the face check
    reads only the frames between its parts), "fail" for a problem found. A clip's own check (app/shot_qc.py) sets
    "pass" or "fail" later. A record with a status keeps ok == (status == "pass"): a clip nobody checked is never
    reported as passed (defect O)."""
    ok: bool
    reasons: List[str] = field(default_factory=list)
    face: Optional[float] = None
    faces: Optional[Dict[str, Optional[float]]] = None  # several characters: name -> score
    status: Optional[str] = None

    def summary(self) -> str:
        return "; ".join(self.reasons) or "ok"

    def record(self) -> dict:
        """The quality record stored with a result (history entries, Timeline clips)."""
        out = {"ok": self.ok if self.status is None else self.status == "pass", "reasons": self.reasons,
               "face": None if self.face is None else round(self.face, 3)}
        if self.status is not None:
            out["status"] = self.status
        if self.faces is not None:
            out["faces"] = {name: None if s is None else round(s, 3)
                            for name, s in self.faces.items()}
        return out


def enabled() -> bool:
    return bool(prefs.load().get("qc", True))


def judge(face: Optional[float], faces: Optional[Dict[str, Optional[float]]] = None,
          minimum: float = FACE_MIN) -> Verdict:
    """Verdict from measured similarities: a score below `minimum` (FACE_MIN, or a style's FaceRules.minimum) is
    another person's face. face - one character's score (None = not measured). faces - several characters (name ->
    score of their own face, None = not found); then face is their minimum, set by the caller."""
    reasons: List[str] = []
    if faces is not None:
        reasons += [f"different face for {name} (similarity {s:.2f} < {minimum:.2f})"
                    for name, s in faces.items() if s is not None and s < minimum]
    elif face is not None and face < minimum:
        reasons.append(f"different face (similarity {face:.2f} < {minimum:.2f})")
    return Verdict(ok=not reasons, reasons=reasons, face=face, faces=faces)


# ----------------------------- twarz (YuNet + SFace) -----------------------------

_LOCK = threading.Lock()
_NETS: Dict[str, object] = {}
_REFS: Dict[str, Tuple[float, list]] = {}  # slug -> (znacznik zdjec, osadzenia twarzy)


def yunet_path() -> Path:
    """The YuNet face detector of the face check; lip sync's face tracking uses the same file."""
    return config.destination_dir("qc") / "face_detection_yunet_2023mar.onnx"


def _nets():
    from engine.models.faces import load_sface, load_yunet

    if not _NETS:
        detector = load_yunet(yunet_path())
        recognizer = load_sface(config.destination_dir("qc") / "face_recognition_sface_2021dec.onnx")
        _NETS.update(det=detector, rec=recognizer)
    return _NETS["det"], _NETS["rec"]


Box = Tuple[int, int, int, int]


def _faces(path) -> List[Tuple[Box, object]]:
    """Every face on the image: (box x0, y0, x1, y1 in the image's own pixels, SFace
    embedding), largest first. Empty when there is no face or no readable file."""
    from engine.models.faces import face_embeddings

    if not Path(path).is_file():
        return []
    return face_embeddings(path, *_nets())


def _embed(path) -> Optional[object]:
    """Osadzenie najwiekszej twarzy na obrazie albo None (brak twarzy / pliku)."""
    faces = _faces(path)
    return faces[0][1] if faces else None


def _cos(a, b) -> float:
    """Cosine similarity of two SFace embeddings (same value as FaceRecognizerSF FR_COSINE)."""
    from engine.models.faces import cosine_similarity

    return cosine_similarity(a, b)


def _best(emb, refs: list) -> Optional[float]:
    return max(_cos(emb, r) for r in refs) if refs else None


def assign(scores: Dict[str, List[Optional[float]]], threshold: float = FACE_MATCH,
           margin: float = MATCH_MARGIN) -> Dict[str, int]:
    """One face per character: scores[character][face] -> {character: face index}.

    The strongest pair goes first. A pair below the threshold is never used, and a face that
    another still unassigned character fits almost as well (within margin) waits until that
    character has found its own face - so look-alike characters are not swapped on a guess."""
    pairs = sorted(((s, key, i) for key, row in scores.items() for i, s in enumerate(row)
                    if s is not None and s >= threshold), reverse=True)
    out: Dict[str, int] = {}
    while True:
        for s, key, i in pairs:
            if key in out or i in out.values():
                continue
            rival = max((row[i] for other, row in scores.items()
                         if other != key and other not in out and row[i] is not None),
                        default=None)
            if rival is not None and rival > s - margin:
                continue
            out[key] = i
            break
        else:
            return out


def count_faces(path) -> int:
    """How many faces YuNet finds on the image (the gate of a reference photo). Under the lock: the detector is
    shared and takes the size of each image before it runs, so two images at once would mix their sizes."""
    with _LOCK:
        return len(_faces(path))


def match_crops(paths: List[str], chars: List[dict]) -> Dict[str, int]:
    """Which face crop (one person each, e.g. a lip-sync track) shows which character:
    slug -> index in paths. Weak or ambiguous matches are left out, and so is every animal or creature."""
    chars = _people(chars)
    with _LOCK:
        embs = [_embed(p) for p in paths]
        refs = {ch["slug"]: _references(ch) for ch in chars}
    scores = {slug: [None if e is None else _best(e, r) for e in embs] for slug, r in refs.items()}
    return assign(scores, threshold=rules(*_styles_of(chars)).match)


def face_scores(path: str, chars: List[dict], style: Optional[str] = None) -> Dict[str, Optional[float]]:
    """Similarity of each character's own face on the image (best distinct faces, strongest
    first): slug -> score, None when no face on the image is recognisably theirs (below
    FACE_MATCH, or the `match` of the styles involved: the picture's style and the characters'; not in the
    picture, or someone else's face). Characters without reference photos cannot be scored and are left out, and
    so is every animal or creature."""
    chars = _people(chars)
    with _LOCK:
        faces = _faces(path)
        refs = {ch["slug"]: _references(ch) for ch in chars}
    refs = {slug: r for slug, r in refs.items() if r}
    scores = {slug: [_best(emb, r) for _, emb in faces] for slug, r in refs.items()}
    picked = assign(scores, threshold=rules(style, *_styles_of(chars)).match, margin=0.0)
    return {slug: (scores[slug][picked[slug]] if slug in picked else None) for slug in refs}


def _references(ch: dict) -> list:
    """The SFace embeddings of the character's reference photos (computed once per set of files). An animal or
    creature has none: SFace knows human faces only, so its photos are never read or cached."""
    if characters.is_creature(ch):
        return []
    files = [f for f in characters.reference_photos(ch) if f.suffix.lower() in _IMAGES and f.exists()]
    stamp = (tuple(str(f) for f in files), sum(f.stat().st_mtime for f in files))
    cached = _REFS.get(ch["slug"])
    if cached and cached[0] == stamp:
        return cached[1]
    embs = [e for e in (_embed(f) for f in files) if e is not None]
    _REFS[ch["slug"]] = (stamp, embs)
    return embs


def face_score(path: str, ch: dict) -> Optional[float]:
    """Najwieksze podobienstwo twarzy z wyniku do twarzy postaci; None - brak twarzy (an animal or creature: always,
    without opening the picture)."""
    if characters.is_creature(ch):
        return None
    with _LOCK:
        refs = _references(ch)
        emb = _embed(path)
        if emb is None or not refs:
            return None
        return _best(emb, refs)


# ----------------------------- checks -----------------------------

def _cast(ch) -> List[dict]:
    """A character, a list of characters or None -> list of characters."""
    if not ch:
        return []
    return [c for c in ch if c] if isinstance(ch, (list, tuple)) else [ch]


def check(path: str, ch, quality: bool = True, require_all: bool = True, style: Optional[str] = None) -> Verdict:
    """Whether the faces on a picture are the characters' own.

    ch - one character, or a list when several characters share the picture: then each one
    is scored on its own face. require_all - a character whose face is not on the picture
    is a reason (a photo); a video frame passes False (people leave the frame).
    With quality off or without a character there is nothing to compare: the verdict is ok
    and the picture is not opened. An animal or creature is left out (it has no face SFace can compare), so a
    picture of creatures alone is not opened either. style - the picture's style: the thresholds are the most
    lenient of its FaceRules and those of the characters' own styles."""
    cast = _people(_cast(ch))
    if not quality or not cast:
        return Verdict(ok=True)
    face, faces = None, None
    limits = rules(style, *_styles_of(cast))
    if not model_manager.ensure([MODEL_ID]):
        try:
            if len(cast) == 1:
                face = face_score(path, cast[0])
            else:
                scores = face_scores(path, cast, style) if style else face_scores(path, cast)
                faces = {c["name"]: scores[c["slug"]] for c in cast if c["slug"] in scores} or None
        except Exception:  # noqa: BLE001 - no OpenCV / unreadable file: the face stays unmeasured
            face, faces = None, None
    verdict = judge(face, faces, limits.minimum)
    if faces is not None:
        found = [s for s in faces.values() if s is not None]
        verdict.face = min(found) if found else None
        if require_all:
            verdict.reasons += [f"face of {name} not found" for name, s in faces.items() if s is None]
            verdict.ok = not verdict.reasons
    return verdict
