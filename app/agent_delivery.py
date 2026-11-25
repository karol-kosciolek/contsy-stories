# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""What the owner takes away from a run: the finished films and pictures, and the documents that go with them (script,
storyboard, shot list, character sheet, subtitles, report), packed in one folder and one ZIP with a manifest.

Nothing a model wrote is ever used as markup or as a name: documents are made from the saved project by fixed templates
whose every text is escaped (and carry a policy that forbids scripts and outside pictures), table cells that could be read
as a formula are defused, and every file name and archive path is made here. The package is written as bytes (not
through the app's text writer, which turns line ends into CR LF)."""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Optional

from PIL import Image

from . import agent_store, characters, config, ffmpeg_utils, items, scene_parts, studio, timeline

DOCUMENTS = ("script", "storyboard", "shot_list", "character_sheet", "subtitles")
GENERATED = ("README.txt", "report.html", "brief_checklist.csv", "manifest.json", "delivery.zip")
FOLDER = "deliverables"
ZIP_NAME = "delivery.zip"
THUMB = 480
DURATION_SLACK = 1.5  # s between an export's length and its project's
_SLUG = re.compile(r"[^a-z0-9]+")
_CSP = ("default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'")
CSS = ("body{font:15px/1.55 Georgia,serif;max-width:860px;margin:2em auto;padding:0 1em;color:#111}"
       "h1,h2,h3{font-family:Arial,Helvetica,sans-serif;font-weight:600}h1{font-size:1.7em}"
       ".scene{border:1px solid #888;padding:.7em 1em;margin:1em 0;page-break-inside:avoid}"
       ".meta{color:#555;font:12px Arial,sans-serif}.who{font:600 13px Arial,sans-serif}"
       "img{max-width:100%;border:1px solid #999}table{border-collapse:collapse;width:100%}"
       "td,th{border:1px solid #999;padding:.3em .5em;text-align:left;vertical-align:top;font-size:13px}"
       "@media print{body{margin:0}}")
LOOK = ("This film is 480p, 30 frames per second, with English speech. It is made by AI models; the people in it are not "
        "real.")


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def slug(text: str, fallback: str = "file", limit: int = 60) -> str:
    return _SLUG.sub("-", str(text).lower()).strip("-")[:limit].strip("-") or fallback


def cell(value) -> str:
    """A table cell that a spreadsheet will not take for a formula."""
    text = str(value if value is not None else "")
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def page(title: str, body: str) -> bytes:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta http-equiv='Content-Security-Policy' content=\"{_CSP}\"><title>{esc(title)}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>").encode("utf-8")


def _seconds(value) -> str:
    return f"{value:g} s"


def _picture(path: Path, side: int = THUMB) -> str:
    """A picture as a data address, scaled down and without its metadata."""
    with Image.open(path) as source:
        picture = source.convert("RGB")
    picture.thumbnail((side, side))
    out = io.BytesIO()
    picture.save(out, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def _speaker(project: timeline.Project, cue: scene_parts.Cue) -> str:
    if cue.character:
        found = characters.get(cue.character)
        return found["name"] if found else cue.character
    return cue.voice or "Narrator"


# --- documents made from a project -----------------------------------------------------------------------------------

def script_doc(project: timeline.Project, notes: str = "") -> bytes:
    scenes = []
    for number, scene in enumerate(project.scenes, 1):
        lines = "".join(f"<p><span class='who'>{esc(_speaker(project, cue))}:</span> {esc(cue.text)}</p>"
                        for cue in scene.lines)
        sound = f"<p class='meta'>Sound: {esc(scene.background)}</p>" if scene.background else ""
        scenes.append(f"<div class='scene'><h3>Scene {number} <span class='meta'>{_seconds(scene.seconds)}"
                      f"{' · cut' if scene.cut else ''}</span></h3><p>{esc(scene.prompt)}</p>{lines}{sound}</div>")
    intro = f"<p>{esc(notes)}</p>" if notes else ""
    return page(f"{project.title} — script",
                f"<h1>{esc(project.title)}</h1><p class='meta'>Script · {_seconds(project.seconds)} · "
                f"{len(project.scenes)} scenes</p>{intro}{''.join(scenes)}{_text_on_screen(project)}"
                f"<p class='meta'>{esc(LOOK)}</p>")


def _text_on_screen(project: timeline.Project) -> str:
    """The exact text burned into the film (Project.overlays) with when it is shown."""
    if not project.overlays:
        return ""
    rows = []
    for overlay in project.overlays:
        when = (f"last {_seconds(overlay.last)} of the film" if overlay.last
                else f"{_seconds(overlay.start)} to {_seconds(overlay.end)}")
        rows.append(f"<p><b>{esc(overlay.text).replace(chr(10), '<br>')}</b> "
                    f"<span class='meta'>{when} · {overlay.position}</span></p>")
    return f"<div class='scene'><h3>Text on screen <span class='meta'>burned into the film</span></h3>{''.join(rows)}</div>"


def storyboard_doc(project: timeline.Project, frames: Optional[dict] = None, notes: str = "") -> bytes:
    """The scenes as cards; `frames` maps a scene number to a picture file that shows it."""
    frames = frames or {}
    cards = []
    for number, scene in enumerate(project.scenes):
        shown = (f"<p><img alt='Scene {number + 1}' src='{_picture(frames[number])}'></p>" if number in frames else "")
        said = "".join(f"<p><span class='who'>{esc(_speaker(project, cue))}:</span> {esc(cue.text)}</p>"
                       for cue in scene.lines)
        cards.append(f"<div class='scene'><h3>Scene {number + 1} <span class='meta'>{_seconds(scene.seconds)}</span></h3>"
                     f"{shown}<p>{esc(scene.prompt)}</p>{said}</div>")
    intro = f"<p>{esc(notes)}</p>" if notes else ""
    return page(f"{project.title} — storyboard", f"<h1>{esc(project.title)}</h1><p class='meta'>Storyboard</p>"
                                                 f"{intro}{''.join(cards)}")


def character_doc(project: timeline.Project) -> bytes:
    blocks = []
    for slug_ in project.cast_slugs:
        person = characters.get(slug_)
        if not person:
            continue
        photo = (f"<p><img alt='{esc(person['name'])}' src='{_picture(Path(person['photos'][0]))}'></p>"
                 if person.get("photos") else "")
        blocks.append(f"<div class='scene'><h3>{esc(person['name'])} <span class='meta'>{esc(person.get('kind', ''))} · "
                      f"{esc(person.get('gender', ''))}</span></h3>{photo}<p>{esc(person.get('look', ''))}</p></div>")
    for slug_ in project.items:
        thing = items.get(slug_)
        if not thing:
            continue
        photo = (f"<p><img alt='{esc(thing['name'])}' src='{_picture(Path(thing['photos'][0]))}'></p>"
                 if thing.get("photos") else "")
        blocks.append(f"<div class='scene'><h3>{esc(thing['name'])} <span class='meta'>{esc(thing.get('kind', ''))}"
                      f"</span></h3>{photo}<p>{esc(thing.get('look', ''))}</p></div>")
    return page(f"{project.title} — characters and places",
                f"<h1>{esc(project.title)}</h1><p class='meta'>Characters, places and items</p>"
                f"{''.join(blocks) or '<p>None.</p>'}")


def shot_list(project: timeline.Project) -> bytes:
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow(["Scene", "Seconds", "Cut", "Description", "Dialogue", "Characters", "Background sound"])
    for number, scene in enumerate(project.scenes, 1):
        said = " | ".join(f"{_speaker(project, cue)}: {cue.text}" for cue in scene.lines)
        writer.writerow([number, scene.seconds, "yes" if scene.cut else "", cell(scene.prompt), cell(said),
                         cell(", ".join(scene.cast or ())), cell(scene.background)])
    return out.getvalue().encode("utf-8-sig")


def _stamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    return f"{millis // 3600000:02d}:{millis // 60000 % 60:02d}:{millis // 1000 % 60:02d},{millis % 1000:03d}"


def subtitles(project: timeline.Project) -> bytes:
    """An SRT from the script, with the lines placed as the film places them. The times are estimates until the film is
    heard: the exported film itself carries measured subtitles when the project has them switched on."""
    rows, offset, number = [], 0.0, 0
    for scene in project.scenes:
        previous = None
        for cue in scene.lines:
            start = cue.start if cue.start is not None else (scene_parts.FIRST_LINE if previous is None
                                                             else previous + scene_parts.LINE_GAP)
            length = max(scene_parts.MIN_LINE_SECONDS, len(cue.text.split()) / scene_parts.WORDS_PER_SECOND)
            previous = start + length
            number += 1
            rows.append(f"{number}\r\n{_stamp(offset + start)} --> {_stamp(offset + min(previous, scene.seconds))}\r\n"
                        f"{cue.text}\r\n")
        offset += scene.seconds
    return ("\r\n".join(rows) + "\r\n").encode("utf-8") if rows else b""


# --- the package -----------------------------------------------------------------------------------------------------

def folder(store: agent_store.RunStore) -> Path:
    target = store.path(FOLDER)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(store: agent_store.RunStore) -> dict:
    found = store.read().get("package")
    found = found if isinstance(found, dict) else {}
    return {"files": [item for item in found.get("files", []) if isinstance(item, dict)],
            "built": found.get("built", ""), "projects": list(found.get("projects", []))}


def _unique(target: Path, name: str) -> str:
    stem, dot, suffix = name.rpartition(".")
    stem, suffix = (stem, "." + suffix) if dot else (name, "")
    candidate, number = name, 2
    while (target / candidate).exists() or candidate in GENERATED:
        candidate = f"{stem}_{number}{suffix}"
        number += 1
    return candidate


def put(store: agent_store.RunStore, name: str, data: bytes, kind: str, *, replace: bool = False,
        project: str = "") -> str:
    """Write a document or a copy of a file into the package; the name is made here from `name`."""
    target = folder(store)
    clean = slug(Path(name).stem, limit=100) + Path(name).suffix.lower()
    current = package(store)
    if replace:
        for old in [item for item in current["files"] if item["name"] == clean]:
            current["files"].remove(old)
    else:
        clean = _unique(target, clean)
    (target / clean).write_bytes(data)
    current["files"].append({"name": clean, "kind": kind, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                             **({"project": project} if project else {})})
    store.update(package={**current, "built": ""})
    return clean


def add_file(store: agent_store.RunStore, source: Path, name: str, kind: str, *, project: str = "") -> str:
    """Copy a result of the studio into the package under a name made here."""
    target = folder(store)
    clean = _unique(target, slug(Path(name).stem) + source.suffix.lower())
    shutil.copy2(source, target / clean)
    current = package(store)
    current["files"].append({"name": clean, "kind": kind, "size": (target / clean).stat().st_size,
                             "sha256": _sha(target / clean), **({"project": project} if project else {})})
    store.update(package={**current, "built": ""})
    return clean


def _signature(files: list) -> str:
    return hashlib.sha256(json.dumps(sorted((item["name"], item["sha256"]) for item in files)).encode()).hexdigest()


def readme(record: dict, files: list) -> bytes:
    outcome = record.get("outcome") or {}
    lines = [f"{record.get('title') or 'Contsy job'}", "=" * 40, "",
             "What is in this folder:", ""]
    lines += [f"  {item['name']}  ({item['kind']})" for item in files]
    lines += ["", "  report.html  what the agent did, what it assumed and what you should check",
              "  brief_checklist.csv  every requirement of the brief and how it went",
              "  *-contact-sheet.jpg  one frame of every scene of the film, to see at a glance whether it follows the brief",
              "  manifest.json  the SHA-256 of every file here", "", LOOK, "",
              "Subtitles in subtitles.srt are timed from the script and are estimates; a film exported with subtitles "
              "has them burned in at the measured times.",
              "Check the film by eye before you hand it on: no AI result is perfect, and the agent could not judge "
              "likeness, logos or on-screen text by itself.", ""]
    if outcome.get("summary"):
        lines += ["Summary:", outcome["summary"], ""]
    return "\r\n".join(lines).encode("utf-8")


def report_doc(record: dict, files: list) -> bytes:
    outcome = record.get("outcome") or {}
    usage = record.get("usage") or {}
    settings = record.get("settings") or {}

    def rows(items_: list, headers: list) -> str:
        if not items_:
            return "<p>None.</p>"
        head = "".join(f"<th>{esc(name)}</th>" for name in headers)
        body = "".join("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>" for row in items_)
        return f"<table><tr>{head}</tr>{body}</table>"

    def bullets(values) -> str:
        return "<ul>" + "".join(f"<li>{esc(value)}</li>" for value in values) + "</ul>" if values else "<p>None.</p>"

    checks = [[item.get("requirement", ""), item.get("status", ""), item.get("note", "")]
              for item in outcome.get("checklist") or []]
    resources = (record.get("resources") or {}).get("created") or []
    advisory = ["Check the logos, products and any person-like face in the film by eye: the agent cannot verify likeness.",
                "Check on-screen text and subtitles by eye."]
    if not settings.get("send_images"):
        advisory.insert(0, "The agent did not see any picture (sending pictures was off): look at every scene yourself.")
    body = (f"<h1>{esc(record.get('title') or 'Report')}</h1>"
            f"<p class='meta'>Mode {esc(settings.get('mode', ''))} · model {esc(settings.get('model', ''))} · estimated cost "
            f"${float(usage.get('cost_usd') or 0):.2f} · {int(record.get('steps') or 0)} steps</p>"
            f"<h2>Summary</h2><p>{esc(outcome.get('summary', 'The run has not finished.'))}</p>"
            f"<h2>Brief checklist</h2>{rows(checks, ['Requirement', 'Status', 'Note'])}"
            f"<h2>What the agent saw, scene by scene</h2>"
            f"{rows([[row.get('scene', ''), row.get('verdict', ''), 'yes' if row.get('scene') in (outcome.get('saw_scenes') or []) else 'no', row.get('note', '')] for row in outcome.get('review') or []], ['Scene', 'Verdict', 'Pictures looked at', 'What it saw'])}"
            f"<h2>Assumptions</h2>{bullets(outcome.get('assumptions'))}"
            f"<h2>Differences from the brief</h2>{bullets(outcome.get('deviations'))}"
            f"<h2>Please check by eye</h2>{bullets([*(outcome.get('owner_should_check') or []), *advisory])}"
            f"<h2>Open points</h2>{bullets([item['text'] for item in outcome.get('failing') or []])}"
            f"<h2>Questions for you</h2>{bullets(record.get('questions'))}"
            f"<h2>Changes to your own library and settings</h2>{bullets(record.get('changes'))}"
            f"<h2>Made by this run</h2>{bullets(resources)}"
            f"<h2>Files</h2>{rows([[item['name'], item['kind'], item['sha256'][:16]] for item in files], ['File', 'Kind', 'SHA-256'])}")
    return page(f"{record.get('title') or 'Contsy'} — report", body)


def checklist_csv(record: dict) -> bytes:
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow(["Requirement", "Status", "Note"])
    for item in (record.get("outcome") or {}).get("checklist") or []:
        writer.writerow([cell(item.get("requirement")), cell(item.get("status")), cell(item.get("note"))])
    return out.getvalue().encode("utf-8-sig")


def build(store: agent_store.RunStore) -> dict:
    """Write the package's own files (README, report, checklist, manifest) and its ZIP from what is in the folder now.
    It can be run again after more files are added; the last run is the one the owner downloads."""
    record = store.read()
    target = folder(store)
    current = package(store)
    files = current["files"]
    for name, data in (("README.txt", readme(record, files)), ("report.html", report_doc(record, files)),
                       ("brief_checklist.csv", checklist_csv(record))):
        (target / name).write_bytes(data)
    listed = [*files, *({"name": name, "kind": "document", "size": (target / name).stat().st_size,
                         "sha256": _sha(target / name)} for name in GENERATED[:3])]
    (target / "manifest.json").write_bytes(json.dumps({"made": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                       "files": listed}, indent=1).encode("utf-8"))
    archive = target / ZIP_NAME
    temporary = target / (ZIP_NAME + ".part")
    stored = (".mp4", ".png", ".jpg", ".jpeg", ".webp")
    with zipfile.ZipFile(temporary, "w") as bundle:
        for item in [*listed, {"name": "manifest.json"}]:
            path = target / item["name"]
            bundle.write(path, arcname=item["name"], compress_type=zipfile.ZIP_STORED if path.suffix.lower() in stored
                         else zipfile.ZIP_DEFLATED)
    temporary.replace(archive)
    store.update(package={**current, "built": _signature(files), "zip": ZIP_NAME})
    return {"files": [item["name"] for item in listed] + ["manifest.json"], "zip": ZIP_NAME,
            "bytes": archive.stat().st_size}


def finalize(store: agent_store.RunStore) -> None:
    """When a run is over: the package is written once more with the run's own report."""
    build(store)


def downloads(store: agent_store.RunStore) -> list[Path]:
    """The files the owner can download, the ZIP first."""
    target = store.path(FOLDER)
    if not target.is_dir():
        return []
    found = sorted((path for path in target.iterdir() if path.is_file() and not path.name.endswith(".part")),
                   key=lambda path: (path.name != ZIP_NAME, path.name))
    return found


# --- the checks before a run may say "complete" ----------------------------------------------------------------------

def quality_failures(project, *, exported: bool = True) -> list[tuple[str, str]]:
    """The independent shot check and sound check of a series episode's project (docs/QUALITY_ARCHITECTURE.md):
    G8 a shot that did not pass (fail, needs_review or unchecked: a shot nobody checked never counts as passed),
    G9 the finished film's QC report missing, or its sound or title checks not passed. Binding: the production agent's
    own review cannot override it."""
    from . import qc_report, shot_qc

    failing = []
    try:
        keys = list(timeline.prepared(project)["assets"])
    except (ValueError, OSError):
        keys = []
    bad = []
    for index in range(len(project.scenes)):
        record = timeline.asset(project, keys[index], verify=False) if index < len(keys) and keys[index] else None
        status = shot_qc.status_of(record.get("qc")) if record else "missing"
        if status != shot_qc.PASS:
            reason = shot_qc.summary(shot_qc.QcVerdict.from_record(record.get("qc"))) if record else "no clip"
            bad.append(f"scene {index + 1} {status} ({reason[:160]})")
    if bad:
        failing.append(("G8", f"Project ‘{project.title}’: shots that did not pass the shot check: "
                              + "; ".join(bad[:8]) + ". Make them again (a new take) or finish partial and list "
                              "them for the owner."))
    if exported:
        movie = timeline.exported(project)
        report = qc_report.read(movie) if movie is not None else None
        if movie is not None and report is None:
            failing.append(("G9", f"The film of ‘{project.title}’ has no QC report: export it again."))
        elif report is not None and report.get("failing_sound"):
            failing.append(("G9", f"The film of ‘{project.title}’ did not pass the sound and title checks: "
                                  + ", ".join(report["failing_sound"]) + "."))
    return failing


def gate_failures(host) -> list[tuple[str, str]]:
    """What stops a run from being complete: (code, text) pairs. G2 scenes that generation would refuse, G4 a finished
    film missing (a run that produces), G7 a package that is not built from what it holds, and for a series episode
    G8/G9 the shot and sound checks (quality_failures)."""
    failing = []
    record = host.store.read()
    created = (record.get("resources") or {}).get("created") or []
    ids = [name.split(":", 1)[1] for name in created if name.startswith("project:")]
    current = package(host.store)
    for project_id in ids:
        try:
            project = timeline.load(project_id)
        except (ValueError, OSError):
            continue
        errors = [f"scene {index + 1}: {issue.text}" for index, issue in timeline.check_scenes(project)
                  if issue.level == scene_parts.ERROR]
        if errors:
            failing.append(("G2", f"Project ‘{project.title}’ has scenes that cannot be made: " + "; ".join(errors[:4])))
        if host.mode != "produce":
            continue
        if _series_run(host):
            failing += quality_failures(project)
        movie = timeline.exported(project)
        if movie is None:
            failing.append(("G4", f"Project ‘{project.title}’ has no finished film (timeline_job export, then "
                                  "add_to_package)."))
            continue
        length = ffmpeg_utils.probe_duration(movie)
        if length is None or abs(length - project.seconds) > DURATION_SLACK:
            failing.append(("G4", f"The film of ‘{project.title}’ lasts {length or 0:.1f} s, but the project is "
                                  f"{project.seconds:g} s: export it again."))
        elif not any(item.get("project") == project_id and item.get("kind") == "film" and item.get("sha256") == _sha(movie)
                     for item in current["files"]):
            failing.append(("G4", f"The finished film of ‘{project.title}’ is not in the package, or the package holds an "
                                  "older cut of it (add_to_package)."))
    if not current["files"]:
        failing.append(("G7", "The package is empty: write the documents and add the files (write_document, "
                              "add_to_package), then build_package."))
    elif current["built"] != _signature(current["files"]):
        failing.append(("G7", "The package was not built after the last file was added: call build_package."))
    return failing


def _series_run(host) -> bool:
    try:
        from .agent_tools_series import bound
    except ImportError:
        return False
    return bound(host) is not None


def export_of(project: timeline.Project) -> Optional[Path]:
    movie = timeline.exported(project)
    return movie if movie and movie.is_file() and studio.is_recorded(str(movie)) else None


def result_file(name: str) -> Optional[Path]:
    """A recorded result of the Gallery by its plain file name, else None."""
    path = config.OUTPUTS_DIR / Path(str(name)).name
    return path if Path(str(name)).name == str(name) and path.is_file() and studio.is_recorded(str(path)) else None
