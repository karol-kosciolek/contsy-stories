# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Build the shot-quality regression set (tests/fixtures/quality) from finished episodes.

The set holds COPIES of short pieces of the Lantern Woods episodes 01-08 (each labelled with the defects A-Q of
docs/QUALITY_ARCHITECTURE.md seen in it, or as a good shot) and small copies of the reference pictures. Tests read only
these copies, never a state folder's live data.

    venv\\Scripts\\python.exe tools\\build_quality_fixtures.py --films <folder with "01 - ....mp4"> --refs <state folder>

Pieces are cut on the episodes' shot boundaries (the Timeline projects' scene lengths, listed in CLIPS) and made
smaller (640x360, H.264 CRF 30, AAC 96 kb/s) so the set stays a few megabytes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "quality"

# id, episode, start s, end s, defects seen, the series characters that must be on screen, what the shot must show,
# what the narrator says (or the scene's subject). Times are shot boundaries of the episode's Timeline project.
CLIPS = [
    ("A_02_s10_boy", "02", 69.8, 78.8, ["A"], [], ["Lumi's burrow at night"], "A boy walks out at the end of the shot."),
    ("B_02_s08_flat", "02", 54.8, 62.8, ["B", "I"], ["lumi", "barley"], ["Lumi's burrow"], "Plain orange backdrop."),
    ("C_04_s07_shell", "04", 45.0, 52.5, ["C"], ["lumi"], ["a puddle", "a leaf"], "Lumi grows a shell."),
    ("C_03_s08_back", "03", 56.8, 64.8, ["C"], ["lumi"], ["a flower pot"], "Belly patch on Lumi's back."),
    ("D_04_s04_grey", "04", 22.5, 30.0, ["D"], ["lumi", "barley"], ["a puddle"], "Barley grey, not in the cast."),
    ("D_04_s08_grey", "04", 52.5, 60.0, ["D", "F"], ["lumi", "barley"], ["leaves floating on a puddle"],
     "Barley grey, not in the cast."),
    ("D_07_s07_grey", "07", 45.0, 52.5, ["D"], ["lumi", "barley"], ["a pool with the moon"], "Barley grey."),
    ("E_03_s06_blue", "03", 37.0, 46.9, ["E"], ["lumi"], ["a flower pot by a round window"], "Whole shot turns blue."),
    ("F_04_s09_water", "04", 60.0, 67.5, ["F", "D"], ["lumi", "barley"], ["a small puddle"],
     "The puddle is a lake; characters on the water."),
    ("F_05_s08_float", "05", 55.0, 62.5, ["F"], ["lumi"], ["a kite"], "Lumi floats."),
    ("F_07_s09_pool", "07", 60.0, 67.5, ["F"], ["lumi", "barley"], ["a small pool"], "Pool becomes a stone basin."),
    ("G_05_s02_kite", "05", 7.5, 15.0, ["G"], ["lumi"], ["a kite"], "The kite turns into a red squid."),
    ("G_08_s02_wand", "08", 7.5, 15.0, ["G"], ["lumi"], ["a bubble wand", "soap bubbles"],
     "The bubble wand goes through the face."),
    ("G_08_s08_bubbles", "08", 52.5, 60.0, ["G"], ["lumi", "barley"], ["soap bubbles"], "Bubbles become noise."),
    ("H_06_s06_snowman", "06", 37.5, 45.5, ["H"], ["lumi"], ["a snowman"], "The snowman."),
    ("H_06_s07_owl", "06", 45.5, 53.0, ["H"], ["lumi", "barley"], ["a snowman"], "The snowman becomes a white owl."),
    ("H_06_s03_house", "06", 15.0, 22.5, ["H"], ["lumi"], ["Lumi's burrow"], "The house is wooden here."),
    ("H_06_s10_house", "06", 69.0, 77.0, ["H"], [], ["Lumi's burrow"], "The house is mossy here."),
    ("I_01_s09_static", "01", 61.5, 69.0, ["I"], ["lumi", "barley"], [], "Front-facing, nearly still."),
    ("I_04_s01_static", "04", 0.0, 7.5, ["I", "K"], ["lumi"], ["a mossy path"], "Title over the face; little motion."),
    ("J_07_s08_crop", "07", 52.5, 60.0, ["J"], ["lumi", "barley"], [], "The camera runs off the characters."),
    ("J_07_s10_crop", "07", 67.5, 75.0, ["J"], ["lumi", "barley"], [], "The camera runs off the characters."),
    ("K_01_s01_title", "01", 0.0, 8.0, ["K"], ["lumi"], [], "Title burned in the middle of the frame."),
    ("good_05_s07", "05", 47.0, 55.0, [], ["barley"], [], "A good shot."),
    ("good_06_s04", "06", 22.5, 30.0, [], ["lumi"], [], "A good shot."),
    ("good_06_s05", "06", 30.0, 37.5, [], ["lumi"], [], "A good shot."),
    ("good_08_s03", "08", 15.0, 22.5, [], ["lumi"], [], "A good shot."),
]
# Longer pieces for the sound checks (L, M, N): the narration rhythm, the ambience changing at every cut and the
# loudness are properties of a stretch of film, not of one shot.
AUDIO = [
    ("audio_03_0_40", "03", 0.0, 40.0, ["L", "M", "N"], "Narration starts 0.3 s after each cut; ambience changes; "
                                                       "quiet gaps."),
    ("audio_06_20_60", "06", 20.0, 60.0, ["L", "M", "N"], "Ambience only about 4 dB under the narrator."),
]
REFS = {"lumi.png": "characters/lumi/photos/01.png", "lumi_34.png": "characters/lumi/photos/02.png",
        "barley.png": "characters/barley/photos/01.png", "barley_34.png": "characters/barley/photos/02.png",
        "the_clearing.png": "items/the_clearing/photos/01.png", "lumi_s_burrow.png": "items/lumi_s_burrow/photos/01.png"}


def _film(folder: Path, episode: str) -> Path:
    found = sorted(folder.glob(f"{episode} - *.mp4"))
    if not found:
        raise SystemExit(f"No film {episode} - *.mp4 in {folder}")
    return found[0]


def _cut(source: Path, start: float, end: float, target: Path, *, video: bool = True) -> None:
    command = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}"]
    if video:
        command += ["-vf", "scale=640:360:flags=area,fps=30", "-c:v", "libx264", "-crf", "30", "-preset", "slow",
                    "-pix_fmt", "yuv420p"]
    else:
        command += ["-vn"]
    command += ["-c:a", "aac", "-b:a", "96k", "-ar", "48000", "-movflags", "+faststart", str(target)]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--films", required=True, type=Path)
    parser.add_argument("--refs", required=True, type=Path, help="A state folder with characters/ and items/.")
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH.")
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    (OUT / "refs").mkdir(parents=True, exist_ok=True)
    labels = {"version": 1, "clips": [], "audio": []}
    for clip_id, episode, start, end, defects, cast, must_show, note in CLIPS:
        target = OUT / "clips" / f"{clip_id}.mp4"
        _cut(_film(args.films, episode), start, end, target)
        labels["clips"].append({"id": clip_id, "file": f"clips/{clip_id}.mp4", "episode": episode, "start": start,
                                "end": end, "defects": defects, "cast": cast, "must_show": must_show, "note": note,
                                "expect": "fail" if defects else "pass"})
        print(clip_id, target.stat().st_size // 1024, "KiB")
    for clip_id, episode, start, end, defects, note in AUDIO:
        target = OUT / "clips" / f"{clip_id}.m4a"
        _cut(_film(args.films, episode), start, end, target, video=False)
        labels["audio"].append({"id": clip_id, "file": f"clips/{clip_id}.m4a", "episode": episode, "start": start,
                                "end": end, "defects": defects, "note": note})
        print(clip_id, target.stat().st_size // 1024, "KiB")
    from PIL import Image

    for name, relative in REFS.items():
        with Image.open(args.refs / relative) as picture:
            picture = picture.convert("RGB")
            picture.thumbnail((384, 384))
            picture.save(OUT / "refs" / name)
    (OUT / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
