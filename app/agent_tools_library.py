# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The agent's tools for the owner's library: characters, places and items, universes, styles, voices, and deleting.

A character or a place is made in two steps because its photos take GPU time: `create_character` starts a draft and
queues its photos (and voice), `wait_for_jobs` waits, `finish_character` saves it. Editing an existing one goes the same
way from a copy of it. Every write asks the host for the permission first; a thing the owner made is copied into the
run's backups before it is changed."""
from __future__ import annotations

from . import characters, items, llm, style_library, styles, timeline, universes, voices
from .agent_host import DELETABLE, Host, ToolError
from .agent_tools import choice, many, obj, text, tool, whole

_SLUG = r"[A-Za-z0-9_-]{1,80}"


def _clean(*texts: str) -> None:
    """The content rule for text the agent wrote (the library functions apply it again to what they store)."""
    if llm.blocked(*[value for value in texts if value]):
        raise ToolError("CONTENT_RULE", llm.BLOCK_MSG)


def _paged(rows: list, args: dict) -> dict:
    offset, limit = args.get("offset", 0), args.get("limit", 40)
    return {"total": len(rows), "offset": offset, "items": rows[offset:offset + limit]}


PAGE = {"offset": whole("How many to skip.", 0, 100_000), "limit": whole("How many to return (default 40).", 1, 100)}


# --- characters --------------------------------------------------------------------------------------------------

def character_view(ch: dict) -> dict:
    return {"slug": ch["slug"], "name": ch["name"], "kind": ch.get("kind", characters.PERSON),
            "gender": ch.get("gender"), "style": ch.get("style") or styles.REALISTIC, "look": ch.get("look", ""),
            "photos": len(ch.get("photos") or []), "has_voice": bool(ch.get("voice") or ch.get("voice_info")),
            "ready": characters.usable(ch["slug"]) is not None}


@tool("list_characters", "List the characters in the owner's library (slug, name, kind, gender, style, look).",
      obj(PAGE))
def _list_characters(host: Host, args: dict):
    return _paged([character_view(ch) for ch in characters.all_characters()], args)


@tool("get_character", "One character of the library, by slug.", obj({"slug": text("The character's slug.", max_length=80)},
                                                                       ("slug",)))
def _get_character(host: Host, args: dict):
    found = characters.get(args["slug"])
    if found is None:
        raise ToolError("NOT_FOUND", "No such character. list_characters shows the slugs.")
    return character_view(found)


def _draft_of(host: Host, kind: str, draft_id: str) -> None:
    if not host.is_mine(f"{kind}_draft", draft_id):
        raise ToolError("NOT_FOUND", f"This run has no {kind} draft with that id.")


@tool("create_character",
      "Start a new character: a person or an animal or creature (animals need an animated film style), with a look, a "
      "voice and reference photos. It queues the photos (and a designed voice) on the GPU and returns a draft id; call "
      "wait_for_jobs for the jobs it lists, then finish_character to save it. The look describes the character only "
      "(age, face, hair, build, clothes); a person is always an adult.",
      obj({"name": text("The character's name.", max_length=characters.MAX_NAME, min_length=1),
           "gender": choice("woman or man (for a creature, the voice's gender).", *characters.GENDERS),
           "look": text("What the character looks like.", max_length=characters.MAX_LOOK, min_length=3),
           "kind": choice("person (default) or creature.", *characters.KINDS),
           "style": text("A style id from list_styles; omit for Realistic.", max_length=200),
           "photos": whole("How many reference photos to make, 1-8 (default 4).", characters.MIN_PHOTOS,
                           characters.MAX_PHOTOS),
           "voice": obj({"preset": text("A built-in voice slug from list_voices.", max_length=80),
                         "description": text("Describe a voice to design instead (queues a voice job).",
                                             max_length=characters.MAX_VOICE_DESCRIPTION)})},
          ("name", "gender", "look", "voice")))
def _create_character(host: Host, args: dict):
    host.require("write", "Creating a character")
    voice = args["voice"]
    preset, design = voice.get("preset"), voice.get("description")
    if bool(preset) == bool(design):
        raise ToolError("INVALID", "voice: give exactly one of preset (a built-in voice from list_voices) or "
                        "description (a voice to design).")
    _clean(args["name"], args["look"], design or "")
    from .agent_tools_series import series_character, series_style
    sheet = series_character(host, args["name"])
    if sheet is not None and sheet.get("slug") and characters.get(sheet["slug"]) is not None:
        raise ToolError("INVALID", f"{sheet['name']} already exists in this series as the library character "
                        f"{sheet['slug']}: use that one, do not design it again.")
    if series_style(host):  # a series character is drawn in the series style only
        args = {**args, "style": series_style(host)}
    if problem := characters.check_name(args["name"]):
        raise ToolError("INVALID", problem)
    draft = characters.new_draft()
    try:
        characters.update_draft(draft, name=args["name"], gender=args["gender"], look=args["look"],
                                style=args.get("style"), kind=args.get("kind"))
        if preset:
            ok, message = characters.use_preset_voice(draft, preset)
            if not ok:
                raise ToolError("INVALID", message)
        with host.queue_scope():
            photo_job = characters.enqueue_photos(draft, args.get("photos", 4))
            voice_job = characters.enqueue_voice_design(draft, design) if design else None
    except BaseException:
        characters.discard_draft(draft)
        raise
    host.created("character_draft", draft)
    jobs = [photo_job.id] + ([voice_job.id] if voice_job else [])
    return {"draft_id": draft, "job_ids": jobs, "next": "wait_for_jobs for these jobs, then finish_character"}


@tool("edit_character",
      "Change a character of the library: its name, look, gender or style, or add more views of it. This opens a copy of "
      "it as a draft (the original stays until finish_character saves it; a copy of the original is kept in the run's "
      "backups). Use wait_for_jobs if it queued photos, then finish_character.",
      obj({"slug": text("The character's slug.", max_length=80),
           "name": text("New name.", max_length=characters.MAX_NAME), "look": text("New look.", max_length=characters.MAX_LOOK),
           "gender": choice("woman or man.", *characters.GENDERS), "style": text("A style id from list_styles.", max_length=200),
           "views": whole("Make the character have this many photos in all by adding views of its first photo.",
                          2, characters.MAX_PHOTOS)}, ("slug",)))
def _edit_character(host: Host, args: dict):
    host.require("write", "Changing a character")
    _clean(args.get("name", ""), args.get("look", ""))
    draft = characters.load_draft(args["slug"])
    host.created("character_draft", draft)
    try:
        characters.update_draft(draft, name=args.get("name"), gender=args.get("gender"), look=args.get("look"),
                                style=args.get("style"))
        jobs = []
        if args.get("views"):
            with host.queue_scope():
                job = characters.enqueue_photos(draft, args["views"], keep_own=True)
            jobs.append(job.id)
    except BaseException:
        characters.discard_draft(draft)
        raise
    return {"draft_id": draft, "job_ids": jobs, "next": "finish_character" + (" after wait_for_jobs" if jobs else "")}


@tool("finish_character", "Save a character draft (from create_character or edit_character) into the library once its "
                          "jobs are done.", obj({"draft_id": text("The draft id.", max_length=64)}, ("draft_id",)))
def _finish_character(host: Host, args: dict):
    host.require("write", "Saving a character")
    _draft_of(host, "character", args["draft_id"])
    try:
        info = characters.draft(args["draft_id"])
    except (ValueError, OSError):
        raise ToolError("NOT_FOUND", "That draft is gone (a draft does not survive a restart of Contsy). Start again "
                        "with create_character.")
    slug = info["slug"]
    if slug and characters.get(slug) is not None:
        host.ensure_backup("character", slug, source=characters._folder(slug))
    profile, message = characters.save_draft(args["draft_id"])
    if profile is None:
        raise ToolError("BUSY" if message.startswith("Wait for") else "INVALID", message)
    if slug and not host.is_mine("character", slug):
        host.note_change(f"Character {slug} changed (a copy of the earlier one is in the run's backups).")
    else:
        host.created("character", profile["slug"])
    return {"character": character_view(profile), "message": message}


@tool("discard_draft", "Throw away a draft of this run (a character or place started but not wanted).",
      obj({"kind": choice("character or item.", "character", "item"), "draft_id": text("The draft id.", max_length=64)},
          ("kind", "draft_id")))
def _discard_draft(host: Host, args: dict):
    _draft_of(host, args["kind"], args["draft_id"])
    (characters if args["kind"] == "character" else items).discard_draft(args["draft_id"])
    return "Draft discarded."


@tool("add_photo_from_attachment",
      "Use a picture the owner attached to this run as a reference photo of a character or place draft. Only with the "
      "owner's permission to use attached files (they confirm the people in them agreed and are adults).",
      obj({"kind": choice("character or item.", "character", "item"), "draft_id": text("The draft id.", max_length=64),
           "picture_id": text("The picture's id from the owner's first message.", max_length=64)},
          ("kind", "draft_id", "picture_id")))
def _add_photo(host: Host, args: dict):
    host.require("write", "Using an attached picture")
    host.require("uploads", "Using an attached picture as a photo")
    _draft_of(host, args["kind"], args["draft_id"])
    path = timeline.reference_path(host.reference_name(args["picture_id"]))
    if path is None:
        raise ToolError("NOT_FOUND", "The attached picture is no longer stored.")
    ok, message = (characters if args["kind"] == "character" else items).add_upload(args["draft_id"], path, True)
    if not ok:
        raise ToolError("INVALID", message)
    return message


# --- places and items ---------------------------------------------------------------------------------------------

def item_view(item: dict) -> dict:
    return {"slug": item["slug"], "name": item["name"], "kind": item.get("kind"), "look": item.get("look", ""),
            "style": item.get("style") or styles.REALISTIC, "photos": len(item.get("photos") or []),
            "ready": items.usable(item["slug"]) is not None}


@tool("list_items", "List the owner's places and items (slug, name, kind place or object, look).", obj(PAGE))
def _list_items(host: Host, args: dict):
    return _paged([item_view(one) for one in items.all_items()], args)


@tool("get_item", "One place or item of the library, by slug.",
      obj({"slug": text("The slug.", max_length=80)}, ("slug",)))
def _get_item(host: Host, args: dict):
    found = items.get(args["slug"])
    if found is None:
        raise ToolError("NOT_FOUND", "No such place or item. list_items shows the slugs.")
    return item_view(found)


@tool("create_item",
      "Start a new place or object (a studio set, a product, a street...) with a description and reference photos. "
      "Queues its photos and returns a draft id; call wait_for_jobs, then finish_item. Name the item in a scene's "
      "description to put it in that scene.",
      obj({"name": text("Its name.", max_length=items.MAX_NAME, min_length=1),
           "kind": choice("place or object.", *items.KINDS),
           "look": text("What it looks like.", max_length=items.MAX_LOOK, min_length=3),
           "style": text("A style id from list_styles; omit for Realistic.", max_length=200),
           "photos": whole("How many photos, 1-8 (default 3).", items.MIN_PHOTOS, items.MAX_PHOTOS)},
          ("name", "kind", "look")))
def _create_item(host: Host, args: dict):
    host.require("write", "Creating a place or item")
    _clean(args["name"], args["look"])
    if problem := items.check_name(args["name"]):
        raise ToolError("INVALID", problem)
    draft = items.new_draft()
    try:
        items.update_draft(draft, name=args["name"], kind=args["kind"], look=args["look"], style=args.get("style"))
        with host.queue_scope():
            job = items.enqueue_photos(draft, args.get("photos", 3))
    except BaseException:
        items.discard_draft(draft)
        raise
    host.created("item_draft", draft)
    return {"draft_id": draft, "job_ids": [job.id], "next": "wait_for_jobs for this job, then finish_item"}


@tool("edit_item", "Change a place or item of the library (name, kind, look, style) from a copy; then finish_item. A "
                   "copy of the original is kept in the run's backups.",
      obj({"slug": text("The slug.", max_length=80), "name": text("New name.", max_length=items.MAX_NAME),
           "kind": choice("place or object.", *items.KINDS), "look": text("New look.", max_length=items.MAX_LOOK),
           "style": text("A style id.", max_length=200),
           "views": whole("Make it have this many photos in all by adding views of its first photo.", 2,
                          items.MAX_PHOTOS)}, ("slug",)))
def _edit_item(host: Host, args: dict):
    host.require("write", "Changing a place or item")
    _clean(args.get("name", ""), args.get("look", ""))
    draft = items.load_draft(args["slug"])
    host.created("item_draft", draft)
    try:
        items.update_draft(draft, name=args.get("name"), kind=args.get("kind"), look=args.get("look"),
                           style=args.get("style"))
        jobs = []
        if args.get("views"):
            with host.queue_scope():
                job = items.enqueue_photos(draft, args["views"], keep_own=True)
            jobs.append(job.id)
    except BaseException:
        items.discard_draft(draft)
        raise
    return {"draft_id": draft, "job_ids": jobs, "next": "finish_item" + (" after wait_for_jobs" if jobs else "")}


@tool("finish_item", "Save a place or item draft into the library once its jobs are done.",
      obj({"draft_id": text("The draft id.", max_length=64)}, ("draft_id",)))
def _finish_item(host: Host, args: dict):
    host.require("write", "Saving a place or item")
    _draft_of(host, "item", args["draft_id"])
    try:
        info = items.draft(args["draft_id"])
    except (ValueError, OSError):
        raise ToolError("NOT_FOUND", "That draft is gone (a draft does not survive a restart of Contsy). Start again "
                        "with create_item.")
    slug = info.get("slug") or ""
    if slug and items.get(slug) is not None:
        host.ensure_backup("item", slug, source=items._folder(slug))
    profile, message = items.save_draft(args["draft_id"])
    if profile is None:
        raise ToolError("BUSY" if message.startswith("Wait for") else "INVALID", message)
    if slug and not host.is_mine("item", slug):
        host.note_change(f"Place or item {slug} changed (a copy of the earlier one is in the run's backups).")
    else:
        host.created("item", profile["slug"])
    return {"item": item_view(profile), "message": message}


# --- universes, styles, voices -------------------------------------------------------------------------------------

@tool("list_universes", "List the owner's universes (named groups of characters that appear together).", obj({}))
def _list_universes(host: Host, args: dict):
    return [{"slug": one["slug"], "name": one["name"], "members": one["members"]} for one in universes.all_universes()]


@tool("save_universe", "Create a universe from 2-5 characters, or change one (give its slug).",
      obj({"name": text("Its name.", max_length=universes.MAX_NAME, min_length=1),
           "members": many("Character slugs.", text("A slug.", max_length=80), max_items=5),
           "slug": text("The slug of the universe to change.", max_length=80)}, ("name", "members")))
def _save_universe(host: Host, args: dict):
    host.require("write", "Saving a universe")
    _clean(args["name"])
    slug = args.get("slug") or ""
    changes_owners = False
    if slug:
        found = universes.get(slug)
        if found is None:
            raise ToolError("NOT_FOUND", "No such universe.")
        changes_owners = not host.is_mine("universe", slug)
        if changes_owners:
            host.ensure_backup("universe", slug, data=found)
    saved, message = universes.save(args["name"], args["members"], slug)
    if saved is None:
        raise ToolError("INVALID", message)
    if not slug:
        host.created("universe", saved["slug"])
    elif changes_owners:
        host.note_change(f"Universe {slug} changed.")
    return {"slug": saved["slug"], "message": message}


@tool("list_styles", "List the film styles: the built-in ones and the owner's own. Use an id in a project's style or "
                     "a character's style.", obj({}))
def _list_styles(host: Host, args: dict):
    presets = [{"id": value, "label": label, "about": style_library.about(value)} for label, value in styles.choices()]
    mine = [{"id": one["id"], "key": one["key"], "name": one["name"], "base": one["base"], "look": one["look"]}
            for one in style_library.all_custom()]
    return {"styles": presets, "own_styles": mine, "bases": [{"id": value, "label": label}
                                                             for label, value in styles.base_choices()]}


@tool("save_style", "Save a style of the owner's own on top of a base style (a new one, or a new version of one: give "
                    "its key). A new version leaves films that use the earlier one as they are.",
      obj({"name": text("Its name.", max_length=style_library.MAX_NAME, min_length=1),
           "base": text("A base id from list_styles.", max_length=200),
           "look": text("The look in words (a short phrase list; letters, digits and , . ; ' ( ) / & + ! ? -).",
                        max_length=400, min_length=3),
           "key": text("The key of the style to change.", max_length=40)}, ("name", "base", "look")))
def _save_style(host: Host, args: dict):
    host.require("write", "Saving a style")
    _clean(args["name"], args["look"])
    key = args.get("key") or ""
    changes_owners = False
    if key:
        found = style_library.get(key)
        if found is None:
            raise ToolError("NOT_FOUND", "No such style of the owner's.")
        changes_owners = not host.is_mine("style", key)
        if changes_owners:
            host.ensure_backup("style", key, data=found)
    entry, message = style_library.save(args["name"], args["base"], args["look"], key)
    if entry is None:
        raise ToolError("INVALID", message)
    if not key:
        host.created("style", entry["key"])
    elif changes_owners:
        host.note_change(f"Style {found['name']} saved as a new version.")
    return {"id": entry["id"], "key": entry["key"], "message": message}


@tool("list_voices", "List the voices a character or a spoken line can use (built-in and imported).", obj(PAGE))
def _list_voices(host: Host, args: dict):
    rows = [{"slug": one.slug, "name": one.name, "gender": one.gender, "age": one.age, "description": one.description,
             "builtin": one.builtin} for one in voices.all_voices()]
    return _paged(rows, args)


@tool("delete_thing",
      "Delete something of the owner's (a project, character, place or item, universe, style, imported voice, or a file "
      "of the Gallery), only with the owner's permission to delete. It goes to the Recycle Bin or, for a style or "
      "universe, is kept in the run's backups. Never needed to finish a job; use only when the brief asks for it.",
      obj({"kind": choice("What to delete.", *DELETABLE),
           "ref": text("Its id or slug (a style: its key; a Gallery file: its file name).", max_length=120)},
          ("kind", "ref")))
def _delete_thing(host: Host, args: dict):
    return host.delete(args["kind"], args["ref"])
