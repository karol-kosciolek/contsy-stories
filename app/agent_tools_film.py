# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The agent's tools for films: Timeline projects and their scenes, the jobs that make clips and the export, AI edits and
takes, and the Video and Image pages' clips and pictures.

A project is saved whole (every scene) with `save_project`, which is how the agent creates, changes, reorders and removes
scenes. Making clips is final production: a run that only prepares for the owner's approval is refused those jobs (MODE)
and checks its scenes with `check_project` instead, which queues nothing and runs no model."""
from __future__ import annotations

import json
from dataclasses import replace

from . import bible as bible_mod
from . import dialogue, llm, overlays as text_overlays, queue_manager, quick_jobs, scene_parts, shot_spec, timeline
from . import video_gen
from .agent_host import Host, ToolError
from .agent_tools import choice, flag, many, number, obj, text, tool, whole

STILL_FORMATS = ("Square 1:1", "Landscape 16:9", "Portrait 9:16", "Portrait 3:4")
TRIES_PER_SCENE = 3  # corrections of one scene (an AI edit or a new take) before the agent must accept it or report it
JOB_ACTIONS = ("prepare_missing", "prepare_scene", "sound_preview", "export", "retry_failed")
_ID = r"[0-9a-f]{32}"
STATUS = {queue_manager.PENDING: "queued", queue_manager.RUNNING: "running", queue_manager.DONE: "done",
          queue_manager.ERROR: "failed", queue_manager.STOPPED: "stopped"}

LINE = obj({"text": text("What is said (English).", max_length=scene_parts.MAX_TEXT, min_length=1),
            "speaker": text("A character slug, or @voice-slug for a library voice; omit for the scene's only character.",
                            max_length=40),
            "start": number("Seconds from the scene's start; omit to place it after the previous line.", 0, 30),
            "pace": number("Set by the studio (the narrator's words a minute); you may send back what get_project "
                           "shows.", 0, scene_parts.PACE_RANGE[1])},
           ("text",))
EFFECT = obj({"text": text("The sound effect described.", max_length=scene_parts.MAX_TEXT, min_length=1),
              "at": number("Seconds from the scene's start.", 0, 30), "seconds": number("How long, 0.5-8 s.", 0.5, 8)},
             ("text",))
SHOT = obj({"location": text("A location of the series bible (its slug), where the shot happens.", max_length=80,
                             min_length=1),
            "time_of_day": choice("The light.", *shot_spec.TIMES),
            "action": text("ONE visible action with its direction, naming who does it: 'Lumi walks from the left to the "
                           "right along the path', 'Barley flutters down and lands beside Lumi'. Never what is NOT "
                           "there.", max_length=300, min_length=3),
            "camera": choice("The camera.", *shot_spec.CAMERAS),
            "cast": many("Character slugs in the shot (every character the action names is added).",
                         text("A slug.", max_length=80), max_items=3),
            "key_visual": many("1-3 things the narration of this shot speaks about that MUST be seen ('a small "
                               "puddle on the path').", text("A short noun phrase.", max_length=60), max_items=3),
            "props": many("Props of the series bible (slugs) in the shot.", text("A slug.", max_length=80), max_items=4),
            "must_show": many("Statements the shot check must find true ('Lumi stands on the grass').",
                              text("A statement.", max_length=100), max_items=5),
            "must_not_show": many("What must NOT be in the shot ('people', 'a second owl'): checked, never written into "
                                  "the prompt.", text("A short phrase.", max_length=100), max_items=5),
            "continuous": flag("Continues the previous shot's last frame (same place and cast, at most 2 in a row); "
                               "default is a cut.")},
           ("location", "time_of_day", "action", "camera", "key_visual"))
SCENE = obj({"prompt": text("The picture and motion of the scene (English), without quoted speech: put what is said in "
                            "lines. Name a character or a place or item to put it in the scene; [1], [2] point to the "
                            "scene's references. In a series episode use shot instead: the studio writes the prompt.",
                            max_length=2000, min_length=1),
             "shot": SHOT,
             "seconds": number("Length in seconds (tenths allowed), 1-30. Fit it to the dialogue (check_project "
                               "suggests one).", 1, video_gen.MAX_SECONDS),
             "cut": flag("An intentional cut: the scene does not continue from the previous one's last frame."),
             "cast": many("Character slugs in this scene (from the project's characters); omit to follow the "
                          "description.", text("A slug.", max_length=80), max_items=5),
             "lines": many("Spoken lines.", LINE, max_items=scene_parts.MAX_LINES),
             "background": text("Background sound; omit to derive it from the picture.", max_length=scene_parts.MAX_TEXT),
             "effects": many("Timed sound effects.", EFFECT, max_items=scene_parts.MAX_EFFECTS),
             "references": many("Ids of pictures the owner attached (from their first message), shown as [1], [2]...",
                                text("A picture id.", max_length=64), max_items=scene_parts.MAX_REFERENCES),
             "voice_db": number("Dialogue level in dB.", *scene_parts.LEVEL_RANGE),
             "background_db": number("Background level in dB.", *scene_parts.LEVEL_RANGE)},
            ("seconds",))
OVERLAY = obj({"text": text("The EXACT text to show, English; up to four lines separated by a line break.",
                            max_length=text_overlays.MAX_TEXT, min_length=1),
               "start": number("Seconds from the film's start where it appears (with end).", 0, text_overlays.MAX_TIME),
               "end": number("Seconds from the film's start where it goes (after start).", 0, text_overlays.MAX_TIME),
               "last_seconds": number("Instead of start and end: show it during the last this many seconds of the film "
                                      "(an end card keeps its place when scenes change length).", text_overlays.MIN_SECONDS,
                                      text_overlays.MAX_TIME),
               "position": choice("center (default), top or bottom.", *text_overlays.POSITIONS),
               "size": choice("medium (default), small or large; a text too big for the frame is made smaller.",
                              *text_overlays.SIZES),
               "backdrop": choice("none (white letters with an outline, default), panel (a dark panel behind them) or card "
                                  "(a plain black card that hides the picture: an end card).", *text_overlays.BACKDROPS)},
              ("text",))


def overlays_from(rows) -> tuple:
    """The Overlay objects of a tool's rows (last_seconds is Overlay.last); ToolError INVALID when one is wrong."""
    try:
        return text_overlays.cleaned(text_overlays.Overlay(
            row["text"], row.get("start", 0.0), row.get("end", 0.0), row.get("last_seconds", 0.0),
            row.get("position", "center"), row.get("size", "medium"), row.get("backdrop", "none")) for row in rows)
    except ValueError as error:
        raise ToolError("INVALID", f"overlays: {error}")


# --- views -------------------------------------------------------------------------------------------------------

def scene_view(scene: timeline.Scene, index: int, project: timeline.Project | None = None) -> dict:
    view = {"index": index, "prompt": scene.prompt, "seconds": scene.seconds, "cut": scene.cut,
            "cast": list(scene.cast) if scene.cast is not None else None,
            "lines": [cue.to_json() for cue in scene.lines], "background": scene.background,
            "effects": [effect.to_json() for effect in scene.effects],
            "references": [name[:12] for name in scene.references]}
    if scene.voice_db:
        view["voice_db"] = scene.voice_db
    if scene.background_db:
        view["background_db"] = scene.background_db
    if scene.shot:
        view["shot"] = json.loads(scene.shot)
    if project is not None:  # what the video model really gets (the compiled shot and the style words)
        view["video_prompt"] = timeline.video_prompt(project, index)
    return view


def project_view(project: timeline.Project, *, state: bool = False) -> dict:
    view = {"id": project.id, "revision": project.revision, "title": project.title, "aspect": project.aspect,
            "audio": project.audio, "style": project.style, "universe": project.universe,
            "characters": list(project.characters), "items": list(project.items), "subtitles": project.subtitles,
            "seconds": project.seconds}
    if project.overlays:  # before the scenes, so that a long answer that is cut keeps them
        view["overlays"] = text_overlays.rows(project.overlays)  # as set_overlays takes them: send them back as they are
        view["overlay_times"] = [list(span) if (span := text_overlays.window(one, project.seconds)) else None
                                 for one in project.overlays]
    view["scenes"] = [scene_view(scene, number, project) for number, scene in enumerate(project.scenes)]
    if state:
        view["state"] = project_state(project)
    return view


def project_state(project: timeline.Project) -> dict:
    """What has been made of a project: which scenes have a clip, which are outdated, the export, the last failure."""
    found: dict = {"job_waiting": timeline.has_active_job(project)}
    try:
        plan = timeline.prepared(project)
        keys = list(plan["assets"])
        found["clips"] = [bool(index < len(keys) and keys[index] and timeline.asset(project, keys[index], verify=False))
                          for index in range(len(project.scenes))]
        found["outdated"] = timeline.outdated(project, keys)
        movie = timeline.exported(project, plan)
        found["exported"] = movie.name if movie else None
        failure = timeline.failure(project)
        if failure:
            found["last_failure"] = {"scene": failure["scene"], "kind": failure["kind"], "error": failure["error"][:400]}
    except (ValueError, OSError) as error:
        found["unreadable"] = str(error)[:200]
    return found


def job_view(job) -> dict:
    view = {"id": job.id, "kind": job.kind, "label": job.label[:160], "status": STATUS.get(job.status, job.status),
            "progress": round(job.progress, 2), "step": job.desc[:160], "files": [name.replace("\\", "/").rsplit("/", 1)[-1]
                                                                                  for name in job.files][:10]}
    if job.error:
        view["error"] = job.error[:600]
    if job.notes:  # what the studio noticed (a join that is not continuous, a lip sync that was skipped...)
        view["notes"] = [note[:500] for note in job.notes][:12]
    if job.status == queue_manager.RUNNING and (left := job.eta()) is not None:
        view["seconds_left"] = int(left)
    elif job.status == queue_manager.PENDING:
        view["estimate_seconds"] = int(job.estimate())
    return view


def _load(project_id: str) -> timeline.Project:
    try:
        return timeline.load(project_id)
    except (ValueError, OSError) as error:
        raise ToolError("NOT_FOUND", f"No such project ({error}). list_projects shows the ids.")


def _fresh(project: timeline.Project, expected) -> None:
    if expected is not None and expected != project.revision:
        raise ToolError("STALE", f"The project is at revision {project.revision}, not {expected}. Read it again "
                        "(get_project) and repeat the change.")


def _backup_project(host: Host, project: timeline.Project) -> None:
    if not host.is_mine("project", project.id):
        host.ensure_backup("project", project.id, source=timeline._inside(timeline._root(project.id), "project.json"))


def _clean_scene_text(args: dict) -> list[str]:
    texts = [args.get("prompt", ""), args.get("background", "")]
    texts += [line["text"] for line in args.get("lines", [])] + [effect["text"] for effect in args.get("effects", [])]
    shot = args.get("shot") if isinstance(args.get("shot"), dict) else {}
    texts += [str(shot.get("action", ""))] + [str(one) for name in ("key_visual", "must_show", "must_not_show")
                                              for one in shot.get(name) or []]
    return texts


def make_scene(host: Host, args: dict, known: set, shot: tuple | None = None) -> timeline.Scene:
    """A scene from a tool's arguments; picture ids become the stored reference names. shot: (spec, compiled prompt,
    cast) of a shot compiled from its ShotSpec (compile_shots): the prompt, the cast and the cut come from it."""
    references = []
    for short in args.get("references", []):
        name = next((one for one in known if one.startswith(short.lower())), None) if len(short) >= 8 else None
        references.append(name or host.reference_name(short))
    if shot is not None:
        spec, prompt, members = shot
        args = {**args, "prompt": prompt, "cast": list(members), "cut": not spec.continuous}
    return timeline.Scene(
        prompt=args["prompt"].strip(), seconds=scene_parts.length(args["seconds"]), cut=bool(args.get("cut", False)),
        cast=tuple(args["cast"]) if "cast" in args else None, shot=shot[0].dumps() if shot is not None else "",
        # a narrator's line in a series shot is read at a children's pace (scene_parts.NARRATOR_PACE)
        lines=tuple(scene_parts.Cue(line["text"], line.get("speaker", ""), line.get("start"),
                                    scene_parts.NARRATOR_PACE if shot is not None
                                    and line.get("speaker", "").startswith(scene_parts.VOICE_PREFIX) else 0.0)
                    for line in args.get("lines", [])),
        background=args.get("background", ""),
        effects=tuple(scene_parts.Effect(effect["text"], effect.get("at", 0.0), effect.get("seconds", 2.0))
                      for effect in args.get("effects", [])),
        references=tuple(references), voice_db=float(args.get("voice_db", 0.0)),
        background_db=float(args.get("background_db", 0.0)))


def compile_shots(host: Host, rows: list) -> list:
    """(spec, prompt, cast) per scene from its `shot`, checked against the series bible; None for a scene without one.
    In a series episode every scene needs a shot, its lines are the narrator's when the series is narrated, and one
    shot is one picture pass (video_gen.part_seconds_limit): ToolError INVALID lists every problem by scene number
    (from 1, as the owner reads them)."""
    from .agent_tools_series import bound

    found = bound(host)
    allowed = bible_mod.for_series(found["series"]) if found else None
    problems, out, previous, chain = [], [], None, 0
    limit = video_gen.part_seconds_limit() if allowed is not None else None
    for scene_number, row in enumerate(rows, 1):
        if "shot" not in row:
            if allowed is not None:
                problems.append(f"scene {scene_number}: a series episode's scene needs a shot (location, time_of_day, "
                                "action, camera, key_visual); the studio writes the prompt from it.")
            elif "prompt" not in row:
                problems.append(f"scene {scene_number}: give a prompt or a shot.")
            out.append(None)
            previous, chain = None, 0
            continue
        if allowed is None:
            problems.append(f"scene {scene_number}: shots are for series episodes; write a prompt instead.")
            out.append(None)
            continue
        try:
            spec = shot_spec.ShotSpec.from_json(row["shot"])
        except shot_spec.ShotError as error:
            problems.append(f"scene {scene_number}: {error}")
            out.append(None)
            continue
        speakers = [line.get("speaker", "") for line in row.get("lines", [])]
        spoken_by = [one for one in speakers if not one.startswith("@")]
        issues = shot_spec.validate(spec, allowed, previous=previous, chain=chain, seconds=float(row["seconds"]),
                                    max_seconds=limit, spoken_by=spoken_by)
        problems += [f"scene {scene_number}: {issue}" for issue in issues]
        if issues:
            out.append(None)
        else:
            out.append((spec, shot_spec.compile_prompt(spec, allowed), shot_spec.shot_cast(spec, allowed)))
        chain = chain + 1 if spec.continuous else 0
        previous = spec
    if problems:
        raise ToolError("INVALID", " ".join(problems[:24]))
    return out


# --- projects ----------------------------------------------------------------------------------------------------

@tool("list_projects", "List the Timeline projects (id, title, revision, number of scenes, length, style).", obj({}))
def _list_projects(host: Host, args: dict):
    return [{"id": one.id, "title": one.title, "revision": one.revision, "scenes": len(one.scenes),
             "seconds": one.seconds, "style": one.style, "aspect": one.aspect} for one in timeline.projects()]


@tool("get_project", "A Timeline project with every scene and what has been made of it (clips, outdated scenes, the "
                     "export, the last failure).",
      obj({"project_id": text("The project id.", max_length=32, pattern=_ID)}, ("project_id",)))
def _get_project(host: Host, args: dict):
    return project_view(_load(args["project_id"]), state=True)


@tool("save_project",
      "Create a Timeline project, or save a new revision of one (give project_id and expected_revision). Send the WHOLE "
      "list of scenes: this is how scenes are added, changed, reordered and removed. Scenes continue from the last "
      "frame of the one before unless cut is true; a scene's characters must be in the project's characters. Saving "
      "is refused while a job of the project is queued. English only; dialogue goes in lines, not in quotes in the "
      "description.",
      obj({"project_id": text("Omit to create a new project.", max_length=32, pattern=_ID),
           "expected_revision": whole("The revision you read; required with project_id.", 1, 1_000_000),
           "title": text("The film's title.", max_length=120, min_length=1),
           "aspect": choice("Format (default Landscape 16:9).", *video_gen.ASPECTS),
           "audio": flag("Sound on (default true)."),
           "style": text("A style id from list_styles (in a series episode the series style is always used).", max_length=200),
           "universe": text("A universe slug, or empty for none.", max_length=80),
           "characters": many("Character slugs in the film.", text("A slug.", max_length=80), max_items=12),
           "items": many("Place and item slugs in the film.", text("A slug.", max_length=80), max_items=50),
           "subtitles": flag("Burn the dialogue into the export as subtitles."),
           "overlays": many("Exact on-screen text burned into the export (set_overlays explains); omit to keep what the "
                            "project has, [] removes it.", OVERLAY, max_items=text_overlays.MAX_OVERLAYS),
           "scenes": many("The scenes, in order.", SCENE, max_items=60)},
          ("title", "scenes")))
def _save_project(host: Host, args: dict):
    host.require("write", "Saving a Timeline project")
    from .agent_tools_series import series_style
    if series_style(host):  # an episode is always drawn in its series' style (app/series.py)
        args = {**args, "style": series_style(host)}
    if llm.blocked(args["title"], *[value for scene in args["scenes"] for value in _clean_scene_text(scene)],
                   *[row["text"] for row in args.get("overlays", [])]):
        raise ToolError("CONTENT_RULE", llm.BLOCK_MSG)
    project_id = args.get("project_id")
    old = None
    if project_id:
        if "expected_revision" not in args:
            raise ToolError("INVALID", "expected_revision is required with project_id.")
        old = _load(project_id)
        _fresh(old, args["expected_revision"])
        _backup_project(host, old)
    known = {name for scene in old.scenes for name in scene.references} if old else set()
    shots = compile_shots(host, args["scenes"])
    scenes = tuple(make_scene(host, scene, known, shot) for scene, shot in zip(args["scenes"], shots))
    options = {name: args[name] for name in ("universe", "subtitles", "style") if name in args}
    if "items" in args:
        options["film_items"] = tuple(args["items"])
    if "overlays" in args:
        options["overlays"] = overlays_from(args["overlays"])
    selected = list(args["characters"]) if "characters" in args else (list(old.characters) if old else [])
    # every character a shot's compiler cast is in the film (the bible decides who is in a shot, not the agent's list)
    selected = list(dict.fromkeys([*selected, *(slug for shot in shots if shot for slug in shot[2])]))
    project = timeline.save_idle(project_id, args.get("expected_revision"), args["title"], scenes,
                                 args.get("aspect", old.aspect if old else "Landscape 16:9"),
                                 args.get("audio", old.audio if old else True), selected, **options)
    if old is None:
        host.created("project", project.id)
    elif project.revision != old.revision and not host.is_mine("project", project.id):
        host.note_change(f"Project ‘{project.title}’ saved as revision {project.revision} "
                         "(the earlier project.json is in the run's backups).")
    return project_view(project)


@tool("set_overlays",
      "Set the EXACT text shown over the finished film: an end card with the brand and address, a tagline, a product name, "
      "a price. The video model cannot spell: letters it draws come out wrong, so never ask a scene's picture to contain text "
      "that must be right. Contsy burns this text into the export itself, in the real letters, from start to end seconds "
      "(or during the last_seconds of the film), at the top, center or bottom, optionally on a dark panel or a black card "
      "that hides the picture. Send the WHOLE list (an empty list removes them; the overlays get_project shows can be "
      "sent back as they are; numbers in messages count from 1). The film shows the text only after the "
      "next export (timeline_job export, which makes the film again from the clips already made). Check it with look_at_film "
      "(at_seconds). Refused while a job of the project is queued.",
      obj({"project_id": text("The project id.", max_length=32, pattern=_ID),
           "expected_revision": whole("The revision you read.", 1, 1_000_000),
           "overlays": many("The overlays.", OVERLAY, max_items=text_overlays.MAX_OVERLAYS)},
          ("project_id", "expected_revision", "overlays")))
def _set_overlays(host: Host, args: dict):
    host.require("write", "Setting the text over a film")
    if llm.blocked(*[row["text"] for row in args["overlays"]]):
        raise ToolError("CONTENT_RULE", llm.BLOCK_MSG)
    overlays = overlays_from(args["overlays"])
    project = _load(args["project_id"])
    _fresh(project, args["expected_revision"])
    _backup_project(host, project)
    saved = timeline.save_idle(project.id, project.revision, project.title, project.scenes, project.aspect, project.audio,
                               list(project.characters), overlays=overlays)
    if saved.revision != project.revision and not host.is_mine("project", project.id):
        host.note_change(f"Project ‘{project.title}’: the text over the film was changed (revision {saved.revision}; the "
                         "earlier project.json is in the run's backups).")
    return {"next": "timeline_job export to show the new text in the film", **project_view(saved)}


@tool("check_project",
      "Check scenes without making anything (no GPU, nothing queued): what generation would refuse (an empty "
      "description, unknown characters or voices, a creature in a realistic style, missing reference images...), "
      "warnings, and the length each scene needs for its dialogue. Do this before producing.",
      obj({"project_id": text("The project id.", max_length=32, pattern=_ID),
           "scenes": many("Scene numbers (0 is the first); omit for all.", whole("A scene number.", 0, 59),
                          max_items=60)}, ("project_id",)))
def _check_project(host: Host, args: dict):
    project = _load(args["project_id"])
    if not project.scenes:
        return {"ready": False, "errors": 1, "issues": [{"scene": None, "level": scene_parts.ERROR,
                                                          "text": "The project has no scenes."}],
                "suggested_seconds": {}}
    if args.get("scenes") is not None and not any(0 <= number < len(project.scenes) for number in args["scenes"]):
        raise ToolError("INVALID", "scenes: none of these is a scene of the project.")
    found = timeline.check_scenes(project, args.get("scenes"))
    suggested = {}
    for index in args.get("scenes") or range(len(project.scenes)):
        try:
            length = timeline.suggested_length(project, index)
        except (ValueError, OSError):
            length = None
        if length is not None:
            suggested[str(index)] = length
    errors = sum(issue.level == scene_parts.ERROR for _, issue in found)
    rows = [{"scene": index, "level": issue.level, "text": issue.text} for index, issue in found]
    rows += _text_issues(project, args.get("scenes"))
    return {"ready": errors == 0, "errors": errors, "issues": rows[:80], "suggested_seconds": suggested}


def _text_issues(project: timeline.Project, scenes=None) -> list[dict]:
    """Warnings about text: a description that asks the picture to show quoted text (a video model cannot spell it; use
    overlays), and overlays that clash or never show."""
    rows = []
    for index, scene in enumerate(project.scenes):
        if scenes is None or index in scenes:
            for shown in dialogue.shown(scene.prompt):
                rows.append({"scene": index, "level": scene_parts.WARNING,
                             "text": f"The description asks the picture to show the text “{shown[:40]}”. A video model "
                                     "cannot spell: the letters will come out wrong. Describe the picture without any "
                                     "letters and put text that must be exact in set_overlays."})
    if scenes is None:
        rows += [{"scene": None, "level": scene_parts.WARNING, "text": text} for text in
                 text_overlays.clashes(project.overlays, project.seconds)]
        for number, overlay in enumerate(project.overlays, 1):
            if text_overlays.window(overlay, project.seconds) is None:
                rows.append({"scene": None, "level": scene_parts.WARNING,
                             "text": f"Overlay {number} starts after the end of the film ({project.seconds} s): it would "
                                     "not be shown."})
        if project.subtitles and any(one.position == "bottom" for one in project.overlays):
            rows.append({"scene": None, "level": scene_parts.WARNING,
                         "text": "Subtitles are on and an overlay is at the bottom, where the subtitles are: they can "
                                 "cover each other. Put the overlay at the top or the center."})
    return rows


# --- jobs ----------------------------------------------------------------------------------------------------------

_KINDS = {"prepare_missing": timeline.PREPARE, "prepare_scene": timeline.PREPARE, "sound_preview": timeline.SOUND,
          "export": timeline.PRODUCE}


@tool("timeline_job",
      "Queue GPU work on a Timeline project: prepare_missing (make every scene's clip), prepare_scene (one scene), "
      "sound_preview (hear one scene's sound without making the clip), export (make what is missing, join and render the "
      "film: the finished MP4), retry_failed (queue the last failed job again). Returns the job id: call wait_for_jobs. "
      "Everything but sound_preview is final production and only a run that produces may do it.",
      obj({"project_id": text("The project id.", max_length=32, pattern=_ID),
           "action": choice("What to queue.", *JOB_ACTIONS),
           "scene": whole("The scene number for prepare_scene and sound_preview (0 is the first).", 0, 59)},
          ("project_id", "action")))
def _timeline_job(host: Host, args: dict):
    host.require("write", "Making clips of a Timeline project")
    action = args["action"]
    if action != "sound_preview":
        host.require_produce(f"timeline_job {action}")
    project = _load(args["project_id"])
    scene = args.get("scene")
    if action == "prepare_scene" and scene is not None:
        made = list(timeline.prepared(project)["assets"])
        if scene < len(made) and made[scene]:  # making a scene again is a try at correcting it; the first making is free
            host.spend_try(f"scene:{project.id}:{scene}", TRIES_PER_SCENE, f"Scene {scene + 1}")
    with host.queue_scope():
        if action == "retry_failed":
            failure = timeline.failure(project)
            if not failure:
                raise ToolError("INVALID", "There is no failed job to retry.")
            if failure["kind"] == timeline.EDIT:
                job = timeline.retry_edit(project)
            else:
                job = timeline.enqueue(project, failure["kind"],
                                       scene_index=failure["scene"] if failure["kind"] in (timeline.PREPARE,
                                                                                            timeline.SOUND) else None)
        else:
            if action in ("prepare_scene", "sound_preview") and scene is None:
                raise ToolError("INVALID", f"{action} needs scene (0 is the first).")
            job = timeline.enqueue(project, _KINDS[action], scene_index=scene if action != "prepare_missing" else None)
    return {"job_id": job.id, "label": job.label, "estimate_seconds": int(job.estimate()),
            "next": "wait_for_jobs, then get_project"}


@tool("edit_scene",
      "Change a scene that already has a clip with AI, from its own clip: new dialogue, a changed sign or caption, a new "
      "detail. Every edit is a new take (the earlier take stays; choose_take switches back). scope carries the edit on "
      "to the scenes after it or through the whole film (for a sign that appears in several scenes). Final production.",
      obj({"project_id": text("The project id.", max_length=32, pattern=_ID),
           "expected_revision": whole("The revision you read.", 1, 1_000_000),
           "scene": whole("The scene number (0 is the first).", 0, 59),
           "scope": choice("scene (default), after or film.", *timeline.EDIT_SCOPES),
           "prompt": text("The scene's new picture description (English), if it changes.", max_length=2000, min_length=1),
           "lines": many("The new spoken lines' texts, in order (each keeps its speaker and time).",
                         text("A line.", max_length=scene_parts.MAX_TEXT, min_length=1), max_items=12),
           "background": text("New background sound.", max_length=scene_parts.MAX_TEXT),
           "picture_change": text("What the first frame should show differently (for example: the sign now says "
                                  "BREAKING NEWS).", max_length=timeline.MAX_CHANGE)},
          ("project_id", "expected_revision", "scene")), produce=True)
def _edit_scene(host: Host, args: dict):
    host.require("write", "Editing a scene")
    project = _load(args["project_id"])
    _fresh(project, args["expected_revision"])
    index = args["scene"]
    if not 0 <= index < len(project.scenes):
        raise ToolError("INVALID", "scene is not a scene of the project.")
    if llm.blocked(args.get("prompt", ""), args.get("background", ""), args.get("picture_change", ""),
                   *args.get("lines", [])):
        raise ToolError("CONTENT_RULE", llm.BLOCK_MSG)
    scene = project.scenes[index]
    if "prompt" in args and scene.shot:
        raise ToolError("INVALID", "This scene's prompt is compiled from its shot: change the shot with save_project "
                        "(or describe the change in picture_change).")
    if "prompt" in args:
        scene = replace(scene, prompt=args["prompt"].strip())
    if "lines" in args:
        scene = timeline.with_lines(scene, args["lines"])
    if "background" in args:
        scene = replace(scene, background=args["background"])
    scope = args.get("scope", "scene")
    host.spend_try(f"scene:{project.id}:{index}", TRIES_PER_SCENE, f"Scene {index + 1}")
    _backup_project(host, project)
    with host.queue_scope():
        saved, job = timeline.edit(project, index, scene, change=args.get("picture_change", ""), scope=scope)
    if not host.is_mine("project", project.id):
        host.note_change(f"Scene {index + 1} of project ‘{project.title}’ edited with AI (scope {scope}): revision "
                         f"{saved.revision}; the earlier take stays.")
    return {"job_id": job.id, "revision": saved.revision, "next": "wait_for_jobs, then get_project"}


@tool("list_takes", "The takes (versions of a scene's clip) of one scene: key, label, which one is shown.",
      obj({"project_id": text("The project id.", max_length=32, pattern=_ID),
           "scene": whole("The scene number.", 0, 59)}, ("project_id", "scene")))
def _list_takes(host: Host, args: dict):
    project = _load(args["project_id"])
    if not 0 <= args["scene"] < len(project.scenes):
        raise ToolError("INVALID", "scene is not a scene of the project.")
    return [{"key": take["key"], "label": take["label"], "current": take["current"], "fits": take["anchor_ok"]}
            for take in timeline.takes(project, args["scene"])]


@tool("choose_take", "Show another take of a scene in the film (from list_takes).",
      obj({"project_id": text("The project id.", max_length=32, pattern=_ID), "scene": whole("The scene number.", 0, 59),
           "key": text("The take's key.", max_length=80)}, ("project_id", "scene", "key")))
def _choose_take(host: Host, args: dict):
    host.require("write", "Choosing a take")
    project = _load(args["project_id"])
    if not 0 <= args["scene"] < len(project.scenes):
        raise ToolError("INVALID", "scene is not a scene of the project.")
    if timeline.has_active_job(project):
        raise ToolError("BUSY", "This project has a job in the GPU queue.")
    _backup_project(host, project)
    saved = timeline.use_take(project, args["scene"], args["key"])
    if not host.is_mine("project", project.id):
        host.note_change(f"Scene {args['scene'] + 1} of project ‘{project.title}’ shows another take.")
    return project_view(saved)


# --- the Video and Image pages ------------------------------------------------------------------------------------

def _pictures(host: Host, ids: list) -> list[str]:
    paths = []
    for short in ids:
        path = timeline.reference_path(host.reference_name(short))
        if path is None:
            raise ToolError("NOT_FOUND", "An attached picture is no longer stored.")
        paths.append(str(path))
    return paths


_PAGE_CHARACTERS = many("Character slugs that appear (their names in the prompt put them in the shot).",
                        text("A slug.", max_length=80), max_items=5)
_PAGE_PICTURES = many("Ids of attached pictures, [1], [2]... in the prompt.", text("A picture id.", max_length=64),
                      max_items=10)


@tool("queue_clip",
      "Queue a single realistic video clip (the Video page: 480p, 30 FPS, up to 30 s, with sound) outside any "
      "project. Film styles belong to Timeline projects. Final production. Returns a job id; its file is listed by "
      "wait_for_jobs.",
      obj({"prompt": text("The scene: subject, action, place, light, camera, sounds. Speech in quotation marks.",
                          max_length=4000, min_length=1),
           "seconds": whole("Length, 1-30.", 1, video_gen.MAX_SECONDS),
           "aspect": choice("Format.", *video_gen.ASPECTS), "sound": flag("With sound (default true)."),
           "characters": _PAGE_CHARACTERS, "pictures": _PAGE_PICTURES,
           "start_picture": text("The id of an attached picture to start from.", max_length=64)},
          ("prompt", "seconds", "aspect")), produce=True)
def _queue_clip(host: Host, args: dict):
    from .agent_tools_series import bound
    if bound(host):
        raise ToolError("INVALID", "An episode is made in its Timeline projects, in the series style; single clips and "
                        "pictures outside a project are not part of Contsy Stories.")
    start = _pictures(host, [args["start_picture"]])[0] if args.get("start_picture") else None
    with host.queue_scope():
        queued = quick_jobs.queue_video(args["prompt"], _pictures(host, args.get("pictures", [])), start,
                                        args["seconds"], args["aspect"], args.get("sound", True),
                                        args.get("characters", []))
    return {"job_id": queued.job.id, "notes": queued.notes}


@tool("queue_still",
      "Queue a single realistic picture (the Image page), for a storyboard frame or a poster. Returns a job id.",
      obj({"prompt": text("The picture.", max_length=4000, min_length=1),
           "format": choice("Format.", *STILL_FORMATS), "characters": _PAGE_CHARACTERS, "pictures": _PAGE_PICTURES},
          ("prompt", "format")))
def _queue_still(host: Host, args: dict):
    from .agent_tools_series import bound
    if bound(host):
        raise ToolError("INVALID", "An episode is made in its Timeline projects, in the series style; single clips and "
                        "pictures outside a project are not part of Contsy Stories.")
    with host.queue_scope():
        queued = quick_jobs.queue_image(args["prompt"], _pictures(host, args.get("pictures", [])), args["format"],
                                        args.get("characters", []))
    return {"job_id": queued.job.id, "notes": queued.notes}
