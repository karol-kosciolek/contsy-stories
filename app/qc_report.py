# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The QC report of an exported episode (docs/QUALITY_ARCHITECTURE.md, layer 4 and 5): what the shot check found in
every shot, what the sound check measured in the finished film, and whether the title covers a character.

The report is written next to the film as `<film>.qc.json` and `<film>.qc.html` (one page with each shot's checked
frames, its verdict and its reasons, and the sound measurements), so the owner sees what the check saw. The delivery
gate (agent_delivery G8/G9) reads the JSON: an episode is complete only when every shot passed and the sound passed.
"""
from __future__ import annotations

import html
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import shot_qc
from .shot_qc import Check, FAIL, PASS, UNCHECKED

TARGET_LUFS, LUFS_TOLERANCE = -14.0, 1.0
TARGET_TP = -1.5
MIN_BED_UNDER_VOICE = 12.0
DEAD_AIR = 0.4               # s of silence (-60 dBFS) that count as dead air
WPM_RANGE = (115.0, 145.0)   # narration pace for ages 3-7 (words per minute of speech)
TITLE_BANDS = {"top": (0.04, 0.03, 0.96, 0.22), "bottom": (0.04, 0.78, 0.96, 0.97), "center": (0.1, 0.38, 0.9, 0.62)}
TITLE_COVER = 0.08           # a title band where characters cover more than this share is over a character
VERSION = 1
_WORD = re.compile(r"[A-Za-z']+")


def report_paths(movie: Path) -> Tuple[Path, Path]:
    return movie.with_name(movie.stem + ".qc.json"), movie.with_name(movie.stem + ".qc.html")


# --- sound ----------------------------------------------------------------------------------------------------

def audio_checks(movie: Path, mix: Optional[dict], records: Sequence[dict], scenes: Sequence) -> Tuple[List[Check], dict]:
    """The sound of the finished film: loudness, true peak, clipping, dead air, the ambience under the narrator, the
    music, the narration fitting its shots, and the narration's pace."""
    from engine.media import audio as media_audio, loudness

    checks: List[Check] = []
    measured: dict = {}
    try:
        samples, rate = media_audio.read(movie, rate=48000)
    except Exception as error:  # noqa: BLE001 - no sound to measure is a failure of the check
        return [Check("audio_read", FAIL, f"the film's sound could not be read: {error}", "N")], measured
    level = loudness.integrated(samples, rate)
    peak = loudness.true_peak(samples, rate)
    clipped = loudness.clipped(samples)
    dead = loudness.silences(samples, rate, longest=DEAD_AIR)
    measured.update(lufs=round(level, 2), true_peak=round(peak, 2), clipped=clipped, dead_air=dead[:20],
                    seconds=round(samples.shape[1] / rate, 2))
    checks.append(Check("loudness", PASS if abs(level - TARGET_LUFS) <= LUFS_TOLERANCE else FAIL,
                        "" if abs(level - TARGET_LUFS) <= LUFS_TOLERANCE else
                        f"{level:.1f} LUFS, the target is {TARGET_LUFS:g} ± {LUFS_TOLERANCE:g}", "N", value=round(level, 2)))
    checks.append(Check("true_peak", PASS if peak <= TARGET_TP + 0.1 else FAIL,
                        "" if peak <= TARGET_TP + 0.1 else f"{peak:.2f} dBTP, at most {TARGET_TP:g}", "N",
                        value=round(peak, 2)))
    checks.append(Check("clipping", PASS if clipped < 1e-5 else FAIL,
                        "" if clipped < 1e-5 else f"{clipped:.4%} of the samples at full scale", "N"))
    checks.append(Check("dead_air", PASS if not dead else FAIL,
                        "" if not dead else f"{len(dead)} silent stretches over {DEAD_AIR:g} s (first at "
                        f"{dead[0][0]:.1f} s)", "M", [start for start, _ in dead[:6]]))
    mix = mix or {}
    under = mix.get("bed_under_voice")
    if under is None:
        checks.append(Check("ambience_under_voice", UNCHECKED, "the film was not made with the episode mix", "M"))
    else:
        checks.append(Check("ambience_under_voice", PASS if under >= MIN_BED_UNDER_VOICE else FAIL,
                            "" if under >= MIN_BED_UNDER_VOICE else
                            f"the ambience is only {under:.1f} LU under the narrator (at least {MIN_BED_UNDER_VOICE:g})",
                            "M", value=round(under, 2)))
    if mix.get("music"):
        checks.append(Check("music", PASS, f"music: {mix['music']}", "M"))
    else:
        checks.append(Check("music", UNCHECKED, "no music: the owner's music library is empty", "M"))
    measured["mix"] = mix
    # the narration in its shots, and its pace
    words, speech, overruns = 0, 0.0, []
    for number, (record, scene) in enumerate(zip(records, scenes), 1):
        for line in record.get("dialogue_timings") or []:
            try:
                start, end = float(line["start"]), float(line["end"])
            except (KeyError, TypeError, ValueError):
                continue
            words += len(_WORD.findall(str(line.get("text", ""))))
            speech += max(0.0, end - start)
            if end > float(scene.seconds) + 0.05:
                overruns.append(number)
    checks.append(Check("narration_fits", PASS if not overruns else FAIL,
                        "" if not overruns else f"the narration runs past the end of shot(s) "
                        f"{', '.join(map(str, overruns))} and is cut", "L"))
    if speech > 0:
        pace = words / speech * 60.0
        measured["words_per_minute"] = round(pace, 1)
        good = WPM_RANGE[0] <= pace <= WPM_RANGE[1]
        checks.append(Check("narration_pace", PASS if good else FAIL, "" if good else
                            f"{pace:.0f} words a minute while speaking; {WPM_RANGE[0]:.0f}-{WPM_RANGE[1]:.0f} for ages "
                            "3-7", "L", value=round(pace, 1)))
    return checks, measured


# --- the title ------------------------------------------------------------------------------------------------

def title_cover(movie: Path, start: float, end: float, cast, band: str) -> float:
    """How much of a title band the characters cover in the film between `start` and `end` (0-1)."""
    frames = _frames_between(movie, start, end)
    return shot_qc.text_box_overlap(frames, cast, TITLE_BANDS[band]) if frames else 0.0


def _frames_between(movie: Path, start: float, end: float, count: int = 5) -> List[shot_qc.Frame]:
    import av
    import numpy as np

    wanted = [start + (end - start) * (i + 0.5) / count for i in range(count)]
    out = []
    with av.open(str(movie)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            moment = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
            if wanted and moment >= wanted[0]:
                image = frame.to_image()
                out.append(shot_qc.Frame(moment, np.asarray(image.resize(shot_qc.SIZE), dtype=np.uint8), image))
                wanted.pop(0)
            if not wanted or moment > end:
                break
    return out


def safe_title_position(movie: Path, start: float, end: float, cast) -> Tuple[str, dict]:
    """The band (top or bottom) where a title covers the fewest characters, with the measured covers."""
    covers = {band: round(title_cover(movie, start, end, cast, band), 3) for band in ("top", "bottom")}
    return min(covers, key=covers.get), covers


# --- the report -----------------------------------------------------------------------------------------------

def build(project, records: Sequence[dict], movie: Path, *, mix: Optional[dict] = None,
          titles: Optional[List[dict]] = None) -> dict:
    """The report of an exported film: every shot's verdict (from its clip record), the sound and the titles."""
    shots = []
    for number, (record, scene) in enumerate(zip(records, project.scenes), 1):
        verdict = shot_qc.QcVerdict.from_record(record.get("qc"))
        shots.append({"scene": number, "status": verdict.status, "attempt": verdict.attempt,
                      "summary": shot_qc.summary(verdict), "checks": [asdict(check) for check in verdict.checks],
                      "frames": verdict.frames, "seconds": scene.seconds, "clip": Path(record["path"]).name,
                      "shot": json.loads(scene.shot) if scene.shot else None})
    sound, measured = audio_checks(movie, mix, records, project.scenes)
    title_checks = []
    for row in titles or []:
        ok = row.get("cover", 0.0) <= TITLE_COVER
        title_checks.append(Check("title_safe_zone", PASS if ok else FAIL, "" if ok else
                                  f"the title “{row.get('text', '')[:40]}” covers characters ({row.get('cover', 0):.0%} "
                                  f"of its band)", "K", value=row.get("cover")))
    failing_shots = [shot["scene"] for shot in shots if shot["status"] != PASS]
    failing_sound = [check.name for check in [*sound, *title_checks] if check.status == FAIL]
    status = PASS if not failing_shots and not failing_sound else FAIL
    return {"version": VERSION, "made": time.strftime("%Y-%m-%d %H:%M:%S"), "project": project.id,
            "title": project.title, "film": movie.name, "status": status, "shots": shots,
            "failing_shots": failing_shots, "sound": [asdict(check) for check in sound],
            "titles": [asdict(check) for check in title_checks], "failing_sound": failing_sound,
            "measured": measured}


def _badge(status: str) -> str:
    colors = {PASS: "#1f7a3a", FAIL: "#a32020", UNCHECKED: "#6b6b6b", shot_qc.NEEDS_REVIEW: "#a36a00"}
    return (f'<span style="background:{colors.get(status, "#555")};color:#fff;padding:2px 8px;border-radius:10px;'
            f'font-size:12px">{html.escape(status)}</span>')


def write(report: dict, movie: Path, records: Sequence[dict]) -> Tuple[Path, Path]:
    """The report as JSON and as one HTML page (each shot's checked frames as a picture inside the page)."""
    json_path, html_path = report_paths(movie)
    json_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>QC {html.escape(report['title'])}</title>"
             "<style>body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:24px}"
             "h1{font-size:22px}h2{font-size:17px;margin-top:28px}td,th{padding:4px 8px;text-align:left}"
             "table{border-collapse:collapse}tr{border-bottom:1px solid #333}img{max-width:100%;border-radius:6px}"
             ".fail{color:#ff8080}.muted{color:#999}</style></head><body>",
             f"<h1>QC report: {html.escape(report['title'])} {_badge(report['status'])}</h1>",
             f"<p class='muted'>{html.escape(report['film'])} · made {html.escape(report['made'])}</p>",
             "<h2>Sound</h2><table>"]
    for check in report["sound"] + report["titles"]:
        parts.append(f"<tr><td>{html.escape(check['name'])}</td><td>{_badge(check['status'])}</td>"
                     f"<td>{html.escape(str(check.get('value') if check.get('value') is not None else ''))}</td>"
                     f"<td class='{'fail' if check['status'] == FAIL else 'muted'}'>{html.escape(check['detail'])}</td>"
                     "</tr>")
    measured = report.get("measured") or {}
    parts.append("</table><p class='muted'>" + html.escape(json.dumps(
        {key: measured.get(key) for key in ("lufs", "true_peak", "words_per_minute", "dead_air")})) + "</p>")
    parts.append("<h2>Shots</h2>")
    for shot, record in zip(report["shots"], records):
        parts.append(f"<h3>Shot {shot['scene']} {_badge(shot['status'])} <span class='muted'>"
                     f"{shot['seconds']} s · try {shot['attempt'] + 1}</span></h3>")
        failing = [check for check in shot["checks"] if check["status"] == FAIL]
        if failing:
            parts.append("<ul>" + "".join(f"<li class='fail'>[{html.escape(check['defect'] or '?')}] "
                                          f"{html.escape(check['detail'] or check['name'])}</li>" for check in failing)
                         + "</ul>")
        elif shot["status"] != PASS:
            parts.append(f"<p class='muted'>{html.escape(shot['summary'])}</p>")
        try:
            parts.append(f"<img alt='frames of shot {shot['scene']}' src='{shot_qc.sheet_data_uri(record['path'])}'>")
        except Exception as error:  # noqa: BLE001 - the page still shows the verdict
            parts.append(f"<p class='fail'>Frames could not be read: {html.escape(str(error))}</p>")
    parts.append("</body></html>")
    html_path.write_text("".join(parts), encoding="utf-8")
    return json_path, html_path


def read(movie: Path) -> Optional[dict]:
    json_path, _ = report_paths(movie)
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("film") == movie.name else None
