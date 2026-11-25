# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The agent's tools for what the owner takes away: documents made from a Timeline project, the finished films and pictures,
and the package that holds them (app/agent_delivery.py). Every document comes from a fixed template; the agent chooses
which one and gives a short note, never markup, and never a file name."""
from __future__ import annotations

from pathlib import Path

from . import agent_delivery as delivery, agent_eyes as eyes, llm
from .agent_host import Host, ToolError
from .agent_tools import choice, many, obj, text, tool, whole
from .agent_tools_film import _ID, _load

SOURCES = ("project_film", "gallery_file")


def _need_clean(*texts: str) -> None:
    if llm.blocked(*[value for value in texts if value]):
        raise ToolError("CONTENT_RULE", llm.BLOCK_MSG)


@tool("write_document",
      "Write one document of the package from a Timeline project: script (the film as text), storyboard (scene cards, "
      "with a picture per scene if you give frames from the Gallery), shot_list (a spreadsheet of the scenes), "
      "character_sheet (characters, places and items with their photos) or subtitles (an SRT estimated from the "
      "script). The studio lays the document out; you give the project and, if you like, a short note for its start.",
      obj({"doc": choice("Which document.", *delivery.DOCUMENTS),
           "project_id": text("The project id.", max_length=32, pattern=_ID),
           "notes": text("A short note shown at the start of a script or storyboard (plain text).", max_length=3000),
           "frames": many("For the storyboard: a Gallery picture per scene.",
                          obj({"scene": whole("The scene number (0 is the first).", 0, 59),
                               "file": text("A picture's file name from list_gallery or wait_for_jobs.",
                                            max_length=200, min_length=1)}, ("scene", "file")), max_items=60)},
          ("doc", "project_id")))
def _write_document(host: Host, args: dict):
    project = _load(args["project_id"])
    notes = args.get("notes", "")
    _need_clean(notes)
    base = f"{delivery.slug(project.title, 'film', 30)}-{project.id[:6]}"  # the id keeps two films of one title apart
    kind = args["doc"]
    if kind == "script":
        name, data = f"{base}-script.html", delivery.script_doc(project, notes)
    elif kind == "storyboard":
        frames = {}
        for row in args.get("frames", []):
            path = delivery.result_file(row["file"])
            if path is None or path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                raise ToolError("NOT_FOUND", f"{row['file']} is not a picture of the Gallery.")
            if row["scene"] >= len(project.scenes):
                raise ToolError("INVALID", f"scene {row['scene']} is not a scene of the project.")
            frames[row["scene"]] = path
        name, data = f"{base}-storyboard.html", delivery.storyboard_doc(project, frames, notes)
    elif kind == "shot_list":
        name, data = f"{base}-shot-list.csv", delivery.shot_list(project)
    elif kind == "character_sheet":
        name, data = f"{base}-characters.html", delivery.character_doc(project)
    else:
        data = delivery.subtitles(project)
        if not data:
            raise ToolError("INVALID", "The project has no dialogue, so there is nothing to subtitle.")
        name = f"{base}-subtitles-estimated.srt"
    written = delivery.put(host.store, name, data, "document", replace=True, project=project.id)
    return {"document": written, "bytes": len(data)}


@tool("add_to_package",
      "Put a finished file into the package the owner downloads: the exported film of a Timeline project "
      "(project_film), or a picture or clip of the Gallery (gallery_file). Names are made by the studio.",
      obj({"source": choice("What to add.", *SOURCES),
           "project_id": text("For project_film: the project id.", max_length=32, pattern=_ID),
           "file": text("For gallery_file: the file name.", max_length=200),
           "label": text("A short name for the file (letters and digits).", max_length=60)}, ("source",)))
def _add_to_package(host: Host, args: dict):
    if args["source"] == "project_film":
        if "project_id" not in args:
            raise ToolError("INVALID", "project_film needs project_id.")
        project = _load(args["project_id"])
        movie = delivery.export_of(project)
        if movie is None:
            raise ToolError("INVALID", "This project has no finished film: run timeline_job export and wait for it.")
        name = delivery.add_file(host.store, movie, args.get("label") or project.title, "film", project=project.id)
        try:  # a sheet of frames, one for each scene: the owner sees at a glance whether the film follows the brief
            sheets, _ = eyes.film_sheets(movie, project.scenes, 1, host.store.path("work"), max_sheets=10)  # every scene
            for number, sheet in enumerate(sheets, 1):
                delivery.put(host.store, f"{Path(name).stem}-contact-sheet{f'-{number}' if number > 1 else ''}.jpg", sheet,
                             "document", project=project.id)
        except Exception as error:  # noqa: BLE001 - the film is in the package either way
            host.store.log("contact_sheet_failed", error=type(error).__name__)
    else:
        if "file" not in args:
            raise ToolError("INVALID", "gallery_file needs file.")
        path = delivery.result_file(args["file"])
        if path is None:
            raise ToolError("NOT_FOUND", "No such result in the Gallery (list_gallery shows the file names).")
        kind = "film" if path.suffix.lower() == ".mp4" else "picture"
        name = delivery.add_file(host.store, path, args.get("label") or path.stem, kind)
    return {"added": name, "next": "build_package when everything is in"}


@tool("build_package",
      "Build the package: the studio writes the README, the report, the brief checklist and the manifest and packs "
      "everything in one ZIP. Call it after the last add_to_package or write_document; finish_run checks that it was.",
      obj({}))
def _build_package(host: Host, args: dict):
    return delivery.build(host.store)
