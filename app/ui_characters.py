# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Characters page (the character list and the one-screen creator) and the Universes page.

A character is made on one screen, top to bottom: what it is (a person, or an animal or creature: characters.KINDS),
the look, gender and style (Realistic, 3D animation or 2D cartoon / anime: the film's style, app/styles.py; an animal
or creature is drawn only in an animated style, and its gender names its voice and pronoun); the reference photos
(generate 1 to 8 of one person, then review them: generate one again, remove one, upload your own with the consent
box); the name; the voice (describe it and generate it, clone a 3-12 s recording with the consent box, or start from a
bundled voice) with a player; Save. The screen works on a draft of app/characters.py (its API is documented
there): nothing reaches the character list or any voice list before Save, and Discard, leaving the page
(main.py wires the other pages' select events to `leave`) or closing the browser tab (the draft State's
delete callback, an hour later) throws the draft away. Editing a character loads a copy of it into the same
screen. The page never sends the user to the Voices page: a character's voice is made and kept here.

Photo generation and voice design are GPU queue jobs. The screen follows the jobs it started with a timer
that runs only while one of them waits or runs; uploads, cloning and bundled voices run at once. Every
creator event shares one concurrency id, so two events of a browser tab never write its draft state at once.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gradio as gr

from . import cast, characters, prefs, queue_manager, style_library, styles, ui_timeline, ui_voices, universes, voices
from .ui_theme import badge, cache_media, esc, note, page_head, plain, section, table, waveform

NEW_UNIVERSE = "New universe"
GENDERS = [("Woman", "woman"), ("Man", "man")]
# What a character is (characters.KINDS). An animal or creature keeps the values "woman" and "man": they pick its voice
# and pronoun, and its labels say so.
CHARACTER_KINDS = [("Person", characters.PERSON), ("Animal or creature", characters.CREATURE)]
CREATURE_GENDERS = [("Female voice (she)", "woman"), ("Male voice (he)", "man")]
CREATURE_NOTE = " · Animal or creature"  # after the style of an animal or creature in the character list
# The creator's words that follow the kind of character: the look's example, step 2's rule and "keep my own photos".
LOOK_EXAMPLES = {
    characters.PERSON: ("For example: late twenties, shoulder-length wavy auburn hair, green eyes, freckles, fair "
                        "skin, slim build, casual style"),
    characters.CREATURE: ("For example: an adult anthropomorphic red-orange corgi, big pointed ears, fluffy tail, blue "
                          "hoodie"),
}
PHOTO_RULES = {
    characters.PERSON: ("Generate photos of one person from the look, then review them. Every photo must show one "
                        "adult with one clear face."),
    characters.CREATURE: ("Generate photos of the animal or creature from the look, then review them. A photo that "
                          "shows someone under 18 is refused."),
}
KEEP_LABELS = {
    characters.PERSON: ("Keep my own photos and add views of the first up to the number of photos (untick to "
                        "generate a new person)"),
    characters.CREATURE: ("Keep my own photos and add views of the first up to the number of photos (untick to "
                          "generate a new animal or creature)"),
}
DEFAULT_PHOTOS = 4
CREATOR = "char-creator"  # the creator's card (scrolled into view when a character is loaded into it)
# Every creator event runs in this one queue: a tab's draft id, watched jobs and display are read and written
# by one event at a time (a name typed while a Generate click is answered would otherwise drop the job id).
SERIAL = "character-creator"
_BUSY = (queue_manager.PENDING, queue_manager.RUNNING)
_KINDS = {"photos": characters.PHOTO_KIND, "voice": characters.VOICE_KIND}
_ORIGINS = {"uploaded": "your photo", "kept": "saved"}
# The creator's heading carries data-unsaved while leaving would lose work (`_unsaved`: unsaved changes, or photos
# or a voice still being made); the page script (ui_theme.SHELL) and these buttons ask before it is thrown away.
_UNSAVED = f"#{CREATOR} [data-unsaved]"
DISCARD_QUESTION = ("Discard the character you are making? Its photos and voice are not saved, and photos or a voice "
                    "still being made are stopped.")
CONFIRM_JS = (f"() => (!document.querySelector('{_UNSAVED}') || confirm('{DISCARD_QUESTION}')) ? 'yes' : 'no'")
_SCROLL = f"document.getElementById('{CREATOR}')?.scrollIntoView({{behavior: 'smooth', block: 'start'}})"
SHOW_CREATOR_JS = f"() => {_SCROLL}"
SHOW_EDITED_JS = f"(answer) => {{ if (answer === 'yes') {_SCROLL}; }}"  # an edit the confirmation cancelled stays put
DELETE_JS = ("() => prompt('Delete the selected character with its photos and voice? It goes to the "
             "Recycle Bin.\\nTo confirm, type: DELETE') || ''")


# ----------------------------- universes -----------------------------

def universes_html() -> str:
    """Universes as a table: name, characters, how many are ready to generate."""
    names = {c["slug"]: c["name"] for c in characters.all_characters()}
    rows = []
    for u in universes.all_universes():
        ready = sum(bool(characters.timeline_profile(slug)) for slug in u["members"])
        rows.append([f"<b>{esc(u['name'])}</b>",
                     esc(", ".join(names.get(s, s) for s in u["members"])),
                     _state_badge(f"Ready {ready} of {len(u['members'])}")
                     if ready < len(u["members"]) else badge("Ready", "ok")])
    return table(["Universe", "Characters", "Status"], rows,
                 empty=f"<b>No universes yet</b>Pick 2 to {cast.MAX_CAST} characters below, name the "
                       "group and click “Save universe”.")


def universe_views(selected: str = ""):
    """Table, universe picker, name and characters of the selected universe ('' = a new one)."""
    u = universes.get(selected)
    return (universes_html(),
            gr.update(choices=[(NEW_UNIVERSE, "")] + [(x["name"], x["slug"])
                                                       for x in universes.all_universes()],
                      value=u["slug"] if u else ""),
            u["name"] if u else "",
            gr.update(choices=[(c["name"], c["slug"]) for c in characters.all_characters()],
                      value=list(u["members"]) if u else []))


# ----------------------------- the character list -----------------------------

def _cover(ch: dict) -> str:
    return cache_media(characters.preview_path(ch)) or ""


def _state_badge(state: str) -> str:
    """A badge for characters.state() (or a universe's "Ready N of M"): its shape tells the state."""
    return badge(state, "ok" if state == "Ready" else "warn")


def _voice_label(ch: dict) -> str:
    return "Own voice" if ch.get("voice_info") else "Library voice" if ch.get("voice") else "No voice yet"


def _rows():
    """(gallery, listing, choices) for the page."""
    chars = characters.all_characters()
    cards = [(c, _cover(c)) for c in chars]
    gallery = [(path, f"{c['name']} · {characters.state(c)}") for c, path in cards if path]
    # the style under the name (and what an animal or creature is): the list keeps its four columns in the narrow card
    rows = [[f"<b>{esc(c['name'])}</b><br><span class='muted'>{esc(style_library.label(c.get('style')))}"
             f"{CREATURE_NOTE if characters.is_creature(c) else ''}</span>",
             _state_badge(characters.state(c)), esc(_voice_label(c)), f"<span class='num'>{len(c['photos'])}</span>"]
            for c in chars]
    listing = table(["Character", "Status", "Voice", "Photos"], rows,
                    empty="<b>No characters yet</b>Make your first one in the creator.")
    return gallery, listing, [(c["name"], c["slug"]) for c in chars]


def _character_assets(slug):
    ch = characters.get(slug)
    if not ch:
        return "<p class='section-sub'>Select a character to see its photos and voice.</p>"
    photos = len(ch["photos"])
    count = f"{photos} reference photo" + ("" if photos == 1 else "s")
    creature = badge(CREATURE_NOTE.strip(" ·")) if characters.is_creature(ch) else ""
    return (f"<div class='char-assets'><strong>{esc(ch['name'])}</strong><div class='char-asset-badges'>"
            f"{badge(style_library.label(ch.get('style')))}{creature}{badge(count, 'ok' if photos else 'warn')}"
            f"{badge(_voice_label(ch), 'ok' if ch.get('voice_info') else 'warn')}</div></div>")


def _library_views(selected: Optional[str]):
    """The list's gallery, table, picker and the selected character's badges, fresh."""
    gallery, listing, names = _rows()
    slugs = [slug for _, slug in names]
    value = selected if selected in slugs else (slugs[0] if slugs else None)
    return gallery, listing, gr.update(choices=names, value=value), _character_assets(value)


# ----------------------------- the creator: what it shows -----------------------------

@dataclass(frozen=True)
class Creator:
    """What the creator shows, kept per browser tab: a redraw sends only the parts that changed, so a playing
    voice or a photo being looked at is not reset by a progress update."""
    heading: str = ""
    photos: tuple = ()          # (draft path, caption) in order; the first is the main photo
    photo_status: str = ""
    photo_busy: bool = False
    own_photos: bool = False    # uploaded or saved photos are in the draft: Generate may keep them
    # (path, size, modification time) of the draft's voice: every new take has the same file name, so its stamp
    # is what tells the player that the voice changed
    voice: tuple = ()
    voice_about: str = ""
    voice_status: str = ""
    voice_busy: bool = False
    designed: bool = False
    checklist: str = ""
    can_save: bool = False


def _draft(draft_id: str) -> Optional[dict]:
    """characters.draft(draft_id), or None when there is none or it was closed."""
    if not draft_id:
        return None
    try:
        return characters.draft(draft_id)
    except ValueError:
        return None


def _watched(watch: dict) -> dict:
    """The jobs this tab started, by kind ("photos", "voice"), as the queue has them now (None when gone)."""
    ids = {kind: watch.get(kind) for kind in _KINDS}
    found = {job.id: job for job in queue_manager.snapshot() if job.id in ids.values()}
    return {kind: found.get(job_id) for kind, job_id in ids.items()}


def _job_status(job: queue_manager.Job, doing: str) -> str:
    """A job of the draft in plain words: waiting, its progress, its result or why it failed."""
    if job.status == queue_manager.PENDING:
        if queue_manager.paused():
            return note("⚠ Waiting: the GPU queue is paused. Click “Resume generation” in the top bar.")
        ahead = queue_manager.position(job.id)
        behind = f" behind {ahead} job{'s' if ahead != 1 else ''}" if ahead else ""
        return f"Waiting in the GPU queue{behind}. It takes about {queue_manager.fmt_time(job.estimate())}."
    if job.status == queue_manager.RUNNING:
        detail = plain(job.desc).strip().rstrip(".…")
        return f"{doing}… {round(job.progress * 100)}%" + (f" · {esc(detail)}" if detail else "")
    if job.status == queue_manager.DONE:
        first, *rest = [plain(text) for text in job.notes if text] or ["Done."]
        return note(f"✅ {first}") + ("\n\n" + "\n".join(f"- {esc(text)}" for text in rest) if rest else "")
    if job.status == queue_manager.STOPPED:
        return note(f"⏹ Stopped. {job.error}".strip())
    return note(f"❌ {job.error or 'It failed. Try again.'}")


def _voice_about(info: Optional[dict]) -> str:
    """Where the draft's voice comes from, for the line under the player."""
    if not info:
        return "No voice yet. Describe one and generate it, clone a recording or use a bundled voice."
    seconds = float(info.get("seconds") or 0.0)
    length = f" · {seconds:.1f} s sample" if seconds else ""
    origin = info.get("origin")
    if origin == "designed":
        return f"Designed from “{esc(info.get('description') or '')}”{length}."
    if origin == "cloned":
        return f"Cloned from your recording{length}."
    if origin in ("preset", "library"):
        voice = voices.get(info.get("preset") or "")
        name = esc(voice.name) if voice else "a bundled voice"
        return (f"Started from {name}{length}. A copy of it is saved with the character."
                if origin == "preset" else f"The voice it had before ({name}){length}. It is copied into the "
                                           "character when you save.")
    return f"Its own voice{length}."


def _heading(data: Optional[dict], unsaved: bool) -> str:
    saved = characters.get(data["slug"]) if data and data.get("slug") else None
    title = f"Edit “{esc(saved['name'])}”" if saved else "New character"
    marker = " data-unsaved" if unsaved else ""
    return (f"<div class='char-creator-head'{marker}><h2 class='section-title'>{title}</h2>"
            "<p class='section-sub'>Work from top to bottom. Nothing is saved until you click Save character; "
            "Discard or leaving this page throws the draft away.</p></div>")


def _same_voice(data: dict, saved: dict) -> bool:
    """The draft speaks with the saved character's voice: its own sample (the same sha256) or, for an older
    character, the copy of its library voice that load_draft made (origin "library" or "preset" naming it)."""
    own = (saved.get("voice_info") or {}).get("sha256")
    if own:
        return bool(data["voice"]) and characters._digest(Path(data["voice"])) == own
    if not saved.get("voice"):
        return not data["voice"]
    info = data["voice_info"] or {}
    return info.get("origin") in ("library", "preset") and info.get("preset") == saved["voice"]


def _unsaved(data: Optional[dict], name: str = "", look: str = "") -> bool:
    """Leaving the creator would lose work: a photo or voice job of the draft waits or runs (leaving stops it), or
    the draft differs from what is saved in its photos, voice, name, look, style or kind (a new character: any of
    its photos, voice, name or look)."""
    if not data:
        return False  # without a draft, leaving keeps the typed fields (`leave` changes nothing)
    if data["busy"]:
        return True
    name, look = " ".join((name or "").split()), " ".join((look or "").split())
    saved = characters.get(data["slug"]) if data.get("slug") else None
    if saved is None:
        return bool(data["photos"] or data["voice"] or name or look)
    return not (_same_voice(data, saved) and len(data["photos"]) == len(saved["photos"])
                and all(origin == "kept" for origin in data["origins"])
                and name == saved["name"] and look == saved["look"]
                and data.get("style", styles.DEFAULT) == saved.get("style", styles.DEFAULT)
                and data.get("kind", characters.PERSON) == saved.get("kind", characters.PERSON))


def _checklist(data: Optional[dict], name: str, look: str, busy: bool, message: str) -> tuple[str, bool]:
    """(what Save still needs, in plain words; whether Save may be clicked)."""
    photos = len(data["photos"]) if data else 0
    missing = [what for what, ok in (("a look description", (look or "").strip()), ("at least one photo", photos),
                                     ("a name", (name or "").strip()), ("a voice", data and data["voice"])) if not ok]
    if busy:
        text = "Save is ready once the photos and the voice being made are done."
    elif missing:
        text = "To save, add " + (", ".join(missing[:-1]) + " and " if len(missing) > 1 else "") + missing[-1] + "."
    else:
        text = f"Ready to save: {photos} photo{'s' if photos != 1 else ''}, a name, a look and its own voice."
    return (note(message) + "\n\n" + text if message else text), not missing and not busy


def _view(draft_id: str, watch: dict, name: str, look: str) -> Creator:
    data = _draft(draft_id)
    jobs = _watched(watch)
    busy_ids = set(data["busy"]) if data else set()
    busy = {kind for kind, job in jobs.items() if job and job.status in _BUSY and job.id in busy_ids}
    photos = tuple(zip(data["photos"], data["origins"])) if data else ()
    captions = tuple((path, " · ".join(part for part in (str(number), "main" if number == 1 else "",
                                                          _ORIGINS.get(origin, "")) if part))
                     for number, (path, origin) in enumerate(photos, 1))
    voice_path = Path(data["voice"]) if data and data["voice"] else None
    try:
        stat = voice_path.stat() if voice_path else None
    except OSError:
        stat = None
    info = data["voice_info"] if data else None
    checklist, can_save = _checklist(data, name, look, bool(busy), watch.get("note", ""))
    return Creator(
        heading=_heading(data, _unsaved(data, name, look)),
        photos=captions,
        photo_status=(_job_status(jobs["photos"], "Making the photos") if jobs["photos"]
                      else note(watch.get("photo_note", ""))),
        photo_busy="photos" in busy,
        own_photos=any(origin in ("uploaded", "kept") for _, origin in photos),
        voice=(str(voice_path), stat.st_size, stat.st_mtime_ns) if stat else (),
        voice_about=_voice_about(info),
        voice_status=(_job_status(jobs["voice"], "Designing the voice") if jobs["voice"]
                      else note(watch.get("voice_note", ""))),
        voice_busy="voice" in busy,
        designed=bool(info and info.get("origin") == "designed"),
        checklist=checklist,
        can_save=can_save)


# The creator's display, in the order every creator event returns it (then the Creator itself, kept per tab).
DISPLAY = ("heading", "photos", "pick", "photo_status", "generate", "keep", "regenerate", "remove", "upload",
           "voice", "voice_about", "voice_status", "design", "clone", "use_preset", "checklist", "save", "ticker")


def _updates(view: Creator, prior: Creator, pick=None, *, force: bool = False, clear_pick: bool = False) -> tuple:
    """An update for every DISPLAY part whose content changed (all of them with force); gr.skip() keeps the
    others, so the selected photo, a playing voice or an open photo stay as they are. `pick` is the photo
    selected now: when the number of photos changes it is kept if it still names a photo, else cleared (a value
    the radio does not list would be refused by Gradio); clear_pick clears it in any case."""
    def changed(*names):
        return force or any(getattr(view, name) != getattr(prior, name) for name in names)

    skip = gr.skip()
    count, has = len(view.photos), bool(view.photos)
    photo_idle = gr.update(visible=has, interactive=not view.photo_busy)  # the actions on one photo
    voice_idle = gr.update(interactive=not view.voice_busy)
    if clear_pick or len(view.photos) != len(prior.photos) or force:
        value = pick if not clear_pick and isinstance(pick, int) and 0 <= pick < count else None
        pick_update = gr.update(choices=[(str(number + 1), number) for number in range(count)], value=value,
                                visible=has)
    else:
        pick_update = skip
    parts = {
        "heading": view.heading if changed("heading") else skip,
        "photos": (gr.update(value=[(url, caption) for path, caption in view.photos if (url := cache_media(path))],
                             visible=has) if changed("photos") else skip),
        "pick": pick_update,
        "photo_status": view.photo_status if changed("photo_status") else skip,
        # with own photos the "keep" box decides between adding views and a new person, so the label stays neutral
        "generate": (gr.update(value="Generate all again" if has and not view.own_photos else "Generate photos",
                               interactive=not view.photo_busy)
                     if changed("photos", "own_photos", "photo_busy") else skip),
        # a new or loaded draft starts with "keep my photos" ticked; later only its visibility follows the draft
        "keep": (gr.update(visible=view.own_photos, **({"value": True} if force else {}))
                 if changed("own_photos") else skip),
        "regenerate": photo_idle if changed("photos", "photo_busy") else skip,
        "remove": photo_idle if changed("photos", "photo_busy") else skip,
        "upload": gr.update(interactive=not view.photo_busy) if changed("photo_busy") else skip,
        "voice": (cache_media(view.voice[0]) if view.voice else None) if changed("voice") else skip,
        "voice_about": view.voice_about if changed("voice_about") else skip,
        "voice_status": view.voice_status if changed("voice_status") else skip,
        "design": (gr.update(value="Generate again" if view.designed else "Generate voice",
                             interactive=not view.voice_busy) if changed("designed", "voice_busy") else skip),
        "clone": voice_idle if changed("voice_busy") else skip,
        "use_preset": voice_idle if changed("voice_busy") else skip,
        "checklist": view.checklist if changed("checklist") else skip,
        "save": gr.update(interactive=view.can_save) if changed("can_save") else skip,
        "ticker": (gr.Timer(active=view.photo_busy or view.voice_busy)
                   if changed("photo_busy", "voice_busy") else skip),
    }
    return tuple(parts[name] for name in DISPLAY)


def _render(draft_id: str, watch: dict, prior: Creator, name: str, look: str, pick=None, *,
            force: bool = False, clear_pick: bool = False) -> tuple:
    """The DISPLAY updates and the new Creator (for the per-tab State)."""
    view = _view(draft_id, watch, name, look)
    return (*_updates(view, prior or Creator(), pick, force=force, clear_pick=clear_pick), view)


# ----------------------------- the creator: actions -----------------------------

def _ensure(draft_id: str, name: str, gender: str, look: str, style: Optional[str] = None,
            kind: Optional[str] = None) -> str:
    """The tab's draft with the form's fields remembered; a new draft when there is none (or it was closed). style
    and kind None (the voice steps) keep the draft's. ValueError when the draft refuses them (an animal or creature
    in a Realistic style, a new kind while the draft has photos): the handlers show it, and a draft made for this
    call is thrown away again."""
    made = _draft(draft_id) is None
    if made:
        draft_id = characters.new_draft()
    try:
        characters.update_draft(draft_id, name=name or "", gender=gender if gender in characters.GENDERS else "woman",
                                look=look or "", style=style if styles.valid(style) else None,
                                kind=kind if kind in characters.KINDS else None)
    except (ValueError, OSError):
        if made:
            _discard(draft_id)  # nobody could reach it: the tab keeps the draft id it had
        raise
    return draft_id


def _discard(draft_id: str) -> None:
    """Throw the draft away (its queued or running jobs are withdrawn by characters.discard_draft)."""
    if draft_id:
        try:
            characters.discard_draft(draft_id)
        except (ValueError, OSError):
            pass  # already gone, or a file still open: the draft folder is removed after a day anyway


def _presets(gender: str):
    """The bundled voices of the character's gender, for "Start from a bundled voice"."""
    wanted = "male" if gender == "man" else "female"
    return gr.update(choices=[(voice.label, voice.slug) for voice in voices.builtin() if voice.gender == wanted],
                     value=None)


def _form(data: Optional[dict] = None) -> tuple:
    """The creator's inputs for a draft (None: an empty creator), in FORM_NAMES order. "Keep my own photos" is
    part of the display (its visibility follows the draft), so it is not here. The words that follow the kind (the
    gender labels, the look's example, step 2's rule) change with the kind radio (`_kind_view`)."""
    data = data or {}
    info = data.get("voice_info") or {}
    gender = data.get("gender") or "woman"
    # a loaded character keeps its photos ("keep" is ticked), so the count starts one above them: Generate then
    # adds a view instead of finding nothing to make
    count = min(characters.MAX_PHOTOS, max(DEFAULT_PHOTOS, len(data.get("photos") or ()) + 1))
    # the draft's style is always among the choices: an earlier version of a style, or one deleted since, too
    style = data.get("style") if styles.valid(data.get("style")) else styles.DEFAULT
    kind = data.get("kind") if data.get("kind") in characters.KINDS else characters.PERSON
    return (gender, data.get("look", ""), count, data.get("name", ""),
            info.get("description", "") if info.get("origin") == "designed" else "", None, False, False,
            _presets(gender), gr.update(value=style, choices=style_library.choices(style)), kind)


def _kind_view(kind, style):
    """The creator's words for the kind of character chosen, or restored by Edit, Save or Discard: (gender, look,
    style, step 2's heading, "keep my own photos"). For an animal or creature Gender names its voice and pronoun (the
    values stay "woman" and "man"), the look gets a creature's example and step 2 asks for photos of it alone; it
    needs an animated style, so a Realistic one (or one of the owner's own based on Realistic) becomes 3D animation."""
    kind = characters.CREATURE if kind == characters.CREATURE else characters.PERSON
    switch = kind == characters.CREATURE and not characters.creature_style_ok(style)
    return (gr.update(choices=CREATURE_GENDERS if kind == characters.CREATURE else GENDERS),
            gr.update(placeholder=LOOK_EXAMPLES[kind]),
            gr.update(value=styles.ANIMATION_3D) if switch else gr.skip(),
            _photo_step(kind),
            gr.update(label=KEEP_LABELS[kind]),
            gr.update(label=characters.CREATURE_CONSENT if kind == characters.CREATURE else characters.CONSENT))


def _photo_step(kind: str) -> str:
    """Step 2's heading with what its photos must show for this kind of character (ui_theme.section's markup)."""
    rule = PHOTO_RULES.get(kind, PHOTO_RULES[characters.PERSON])
    return f"<div><h3 class='section-title'>2. Reference photos</h3><p class='section-sub'>{rule}</p></div>"


def generate_photos(draft_id, watch, prior, name, gender, look, count, keep, pick, style=None, kind=None):
    """Generate: a new person or animal or creature (or, with own photos kept, views of the first of them) as a GPU
    queue job, drawn in the chosen style."""
    try:
        draft_id = _ensure(draft_id, name, gender, look, style, kind)
        data = characters.draft(draft_id)
        own = any(origin in ("uploaded", "kept") for origin in data["origins"])
        job = characters.enqueue_photos(draft_id, int(count or DEFAULT_PHOTOS), keep_own=bool(keep) and own)
        watch = {**watch, "photos": job.id, "photo_note": ""}
    except (ValueError, OSError) as error:
        watch = {**watch, "photos": None, "photo_note": f"❌ {error}"}
    return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick))


def _picked(pick) -> int:
    if not isinstance(pick, int):
        raise ValueError("Select a photo first: click it in the grid or choose its number.")
    return pick


def regenerate_photo(draft_id, watch, prior, name, gender, look, pick, style=None, kind=None):
    """A new take of the selected generated photo, as a GPU queue job."""
    try:
        draft_id = _ensure(draft_id, name, gender, look, style, kind)
        job = characters.enqueue_regenerate(draft_id, _picked(pick))
        watch = {**watch, "photos": job.id, "photo_note": ""}
    except (ValueError, OSError) as error:
        watch = {**watch, "photos": None, "photo_note": f"❌ {error}"}
    return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick))


def remove_photo(draft_id, watch, prior, name, look, pick):
    """Remove the selected photo; the selection is cleared, so a second click removes nothing by mistake."""
    try:
        index = _picked(pick)
        if _draft(draft_id) is None:
            raise ValueError("There are no photos to remove yet.")
        characters.remove_photo(draft_id, index)
        message = f"✅ Photo {index + 1} removed."
    except (ValueError, OSError) as error:
        message = f"❌ {error}"
    watch = {**watch, "photos": None, "photo_note": message}
    return (draft_id, watch, *_render(draft_id, watch, prior, name, look, clear_pick=True))


def upload_photos(draft_id, watch, prior, name, gender, look, files, consent, pick, style=None, kind=None):
    """Your own photos: each one needs the consent box and passes the photo check (one adult with one face; an
    animal or creature: it alone, fully grown, with no people)."""
    files = [str(file) for file in (files if isinstance(files, list) else [files] if files else [])]
    if not files:
        return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick))
    if not consent:
        watch = {**watch, "photos": None,
                 "photo_note": "❌ Tick the box to confirm that the photos show a fictional character or an adult "
                               "who agreed, then upload them again."}
        return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick))
    try:
        draft_id = _ensure(draft_id, name, gender, look, style, kind)
        added, refused = 0, []
        for file in files:
            ok, message = characters.add_upload(draft_id, file, True)
            if ok:
                added += 1
            else:
                refused.append(f"{Path(file).name}: {plain(message)}")
        lead = (f"✅ {added} photo{'s' if added != 1 else ''} added." if added else
                "❌ No photo was added.")
        message = lead + ("\n\n" + "\n".join(f"- {esc(line)}" for line in refused) if refused else "")
    except (ValueError, OSError) as error:
        message = f"❌ {error}"
    watch = {**watch, "photos": None, "photo_note": message}
    return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick))


def design_voice(draft_id, watch, prior, name, gender, look, description, pick):
    """Generate a voice from its description (VoxCPM2 voice design) as a GPU queue job."""
    try:
        draft_id = _ensure(draft_id, name, gender, look)
        job = characters.enqueue_voice_design(draft_id, description)
        watch = {**watch, "voice": job.id, "voice_note": ""}
    except (ValueError, OSError) as error:
        watch = {**watch, "voice": None, "voice_note": f"❌ {error}"}
    return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick))


def clone_voice(draft_id, watch, prior, name, gender, look, sample, consent, pick):
    """The voice from 3-12 s of a recording (the consent box is required); the sample field empties on success."""
    try:
        draft_id = _ensure(draft_id, name, gender, look)
        ok, message = characters.clone_voice(draft_id, sample, bool(consent))
    except (ValueError, OSError) as error:
        ok, message = False, str(error)
    watch = {**watch, "voice": None, "voice_note": ("✅ " if ok else "❌ ") + message}
    return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick), None if ok else gr.skip())


def use_preset(draft_id, watch, prior, name, gender, look, preset, pick):
    """Start from a bundled voice: a copy of its sample becomes the character's voice."""
    try:
        if not preset:
            raise ValueError("Choose a bundled voice first.")
        draft_id = _ensure(draft_id, name, gender, look)
        ok, message = characters.use_preset_voice(draft_id, preset)
    except (ValueError, OSError) as error:
        ok, message = False, str(error)
    watch = {**watch, "voice": None, "voice_note": ("✅ " if ok else "❌ ") + message}
    return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick))


def refresh_creator(draft_id, watch, prior, name, look, pick):
    """The timer while a job runs, and typing in the name or look (the Save button and its checklist)."""
    if watch.get("note"):  # a refusal of Save is about the values before this change
        watch = {**watch, "note": ""}
    return (watch, *_render(draft_id, watch, prior, name, look, pick))


def save_character(draft_id, watch, prior, name, gender, look, pick, style=None, kind=None):
    """Save: photos, voice, name, look, gender, style and kind together. On success the creator starts a new character
    and the list selects the saved one; on a refusal everything stays as it is."""
    try:
        if _draft(draft_id) is None:
            raise ValueError("Generate or upload photos and give the character a voice first.")
        profile, message = characters.save_draft(draft_id, name=name or "", gender=gender, look=look or "",
                                                 style=style if styles.valid(style) else None,
                                                 kind=kind if kind in characters.KINDS else None)
    except (ValueError, OSError) as error:
        profile, message = None, str(error)
    keep = (gr.skip(),) * (len(FORM_NAMES) + 5)
    if profile is None:
        watch = {**watch, "note": f"❌ {message}"}
        return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick), *keep)
    watch = {"note": f"✅ {message}"}
    return ("", watch, *_render("", watch, prior, "", "", force=True), *_form(),
            *_library_views(profile["slug"]), gr.update(choices=cast.top_picker_choices(),
                                                        value=cast.top_picker_value()))


def start_over(confirmed, draft_id, watch, prior):
    """Discard or New character (after the browser's confirmation when the draft holds unsaved work)."""
    if confirmed != "yes":
        return (gr.skip(),) * (2 + len(DISPLAY) + 1 + len(FORM_NAMES))
    _discard(draft_id)
    return ("", {}, *_render("", {}, prior, "", "", force=True), *_form())


def leave(draft_id, watch, prior):
    """Another page was opened: the draft is thrown away (nothing to do without one)."""
    if not draft_id:
        return (gr.skip(),) * (2 + len(DISPLAY) + 1 + len(FORM_NAMES))
    return start_over("yes", draft_id, watch, prior)


def edit_character(confirmed, slug, draft_id, watch, prior):
    """Load a copy of the selected character into the creator (after the browser's confirmation when the
    current draft holds unsaved work)."""
    skip_all = (gr.skip(),) * (2 + len(DISPLAY) + 1 + len(FORM_NAMES))
    if confirmed != "yes":
        return (*skip_all, gr.skip())
    ch = characters.get(slug or "")
    if not ch:
        return (*skip_all, note("❌ Select a character to edit."))
    _discard(draft_id)
    try:
        draft_id = characters.load_draft(slug)
        data = characters.draft(draft_id)
    except (ValueError, OSError) as error:
        return ("", {}, *_render("", {}, prior, "", "", force=True), *_form(), note(f"❌ {error}"))
    watch = {}
    needs = "" if data["photos"] else " It has no reference photos yet: generate or upload some."
    return (draft_id, watch, *_render(draft_id, watch, prior, data["name"], data["look"], force=True),
            *_form(data), note(f"✅ “{ch['name']}” is open in the creator.{needs}"))


def _select_photo(evt: gr.SelectData):
    """A click on a photo in the grid selects its number below it (the keyboard way to the same choice)."""
    return evt.index if isinstance(evt.index, int) else gr.skip()


def _forget_draft(draft_id) -> None:
    """The browser tab is gone (Gradio deletes its state an hour after it closes): so is its draft."""
    if isinstance(draft_id, str):
        _discard(draft_id)


FORM_NAMES = ("gender", "look", "count", "name", "voice_text", "sample", "voice_consent", "photo_consent", "preset",
              "style", "kind")


# ----------------------------- the page -----------------------------

def build_characters_tab(tab: gr.Tab, top_picker: gr.Dropdown, timeline_cast):
    """The character list beside the one-screen creator. Returns (leave, inputs, outputs): main.py runs it
    when another page is opened, so leaving the screen discards the draft."""
    page_head("Characters", "A character is a look, 1 to 8 reference photos of one adult, a name and its own "
                            "voice. Nothing is trained: a character is ready when you save it.")
    timeline_universe, timeline_additions = timeline_cast
    initial = _view("", {}, "", "")
    with gr.Row(equal_height=False, elem_classes=["char-layout"]):
        with gr.Column(scale=2, min_width=300, elem_classes=["card"]):
            section("Your characters", "Select one to edit it, add it to the Timeline or delete it.")
            gallery0, listing0, names0 = _rows()
            cards = gr.Gallery(value=gallery0, show_label=False, columns=3, height=300,
                               allow_preview=False, object_fit="cover", elem_id="char-cards", buttons=[])
            listing = gr.HTML(listing0)
            # allow_custom_value: the list refreshes every few seconds; the code checks the slug anyway
            pick = gr.Dropdown(names0, value=names0[0][1] if names0 else None,
                               label="Selected character", interactive=True, allow_custom_value=True)
            assets = gr.HTML(_character_assets(names0[0][1] if names0 else ""), padding=False)
            with gr.Row(elem_classes=["toolbar"]):
                edit = gr.Button("Edit", size="sm", variant="primary")
                use = gr.Button("Add to Timeline", size="sm")
                delete = gr.Button("Delete", size="sm", variant="stop")
            info = gr.Markdown(elem_classes=["status-box"])
            section("Timeline universe", "Add a whole group of characters to the Timeline cast.")
            with gr.Row(elem_classes=["toolbar"]):
                universe_pick = gr.Dropdown(ui_timeline.universe_choices(), value="", show_label=False,
                                            allow_custom_value=True, interactive=True, elem_classes=["grow"])
                universe_use = gr.Button("Use in Timeline", size="sm")
            universe_info = gr.Markdown(elem_classes=["status-box"])
        with gr.Column(scale=3, min_width=320, elem_id=CREATOR, elem_classes=["card"]):
            with gr.Row(elem_classes=["toolbar", "char-creator-top"]):
                heading = gr.HTML(initial.heading, padding=False, elem_classes=["grow"])
                new = gr.Button("New character", size="sm")
            with gr.Column(elem_classes=["char-step"]):
                section("1. Look")
                kind = gr.Radio(CHARACTER_KINDS, value=characters.PERSON, label="Character",
                                info="An animal or creature needs an animated style, and its gender picks its voice.",
                                elem_id="char-kind")
                gender = gr.Radio(GENDERS, value="woman", label="Gender")
                # The photos are drawn in this style; pick the style of the films the character will play in
                style = gr.Dropdown(style_library.choices(), value=styles.CARTOON, label="Style",
                                    info="Draw the character in the style of its films. New photos use it. Make your "
                                         "own styles on the Styles page.",
                                    elem_id="char-style")
                look = gr.Textbox(label="Look", lines=3, max_lines=8, max_length=characters.MAX_LOOK,
                                  placeholder=LOOK_EXAMPLES[characters.PERSON])
            with gr.Column(elem_classes=["char-step"]):
                photo_step = section("2. Reference photos", PHOTO_RULES[characters.PERSON])
                with gr.Row(elem_classes=["toolbar"]):
                    count = gr.Slider(characters.MIN_PHOTOS, characters.MAX_PHOTOS, value=DEFAULT_PHOTOS, step=1,
                                      label="Number of photos", elem_classes=["grow"])
                    generate = gr.Button("Generate photos", variant="primary", size="sm")
                keep_own = gr.Checkbox(True, label=KEEP_LABELS[characters.PERSON], visible=False)
                photo_status = gr.Markdown(initial.photo_status, elem_classes=["status-box"])
                photos = gr.Gallery(value=[], label="Reference photos", show_label=False, columns=4,
                                    height="auto", object_fit="cover", allow_preview=True, buttons=["fullscreen"],
                                    visible=False, elem_id="char-photos")
                photo_pick = gr.Radio([], value=None, label="Selected photo", visible=False,
                                      elem_id="char-photo-pick")
                with gr.Row(elem_classes=["toolbar"]):
                    regenerate = gr.Button("Generate this photo again", size="sm", visible=False)
                    remove = gr.Button("Remove this photo", size="sm", variant="stop", visible=False)
                photo_consent = gr.Checkbox(False, label=characters.CONSENT)
                upload = gr.UploadButton("Upload your own photos", file_types=["image"], file_count="multiple",
                                         size="sm", elem_id="char-photo-upload")
            with gr.Column(elem_classes=["char-step"]):
                section("3. Name", "Write it in prompts to put the character in a picture or a scene.")
                name = gr.Textbox(label="Name", max_lines=1, max_length=characters.MAX_NAME,
                                  placeholder="For example, Mia")
            with gr.Column(elem_classes=["char-step"]):
                section("4. Voice", "The character speaks its dialogue lines with this voice. It is saved with "
                                    "the character.")
                with gr.Tabs(elem_id="char-voice-source"):
                    with gr.Tab("Describe", id="char-voice-describe"):
                        voice_text = gr.Textbox(label="Describe the voice", lines=2, max_lines=4,
                                                max_length=characters.MAX_VOICE_DESCRIPTION,
                                                placeholder="For example: a warm, low, calm voice with a slight rasp")
                        design = gr.Button("Generate voice", variant="primary", size="sm")
                    with gr.Tab("Clone a recording", id="char-voice-clone"):
                        sample = gr.Audio(label="3–12 seconds of one person speaking clearly", type="filepath",
                                          sources=["upload", "microphone"], waveform_options=waveform())
                        voice_consent = gr.Checkbox(False, label=ui_voices.VOICE_CONSENT)
                        clone = gr.Button("Clone voice", variant="primary", size="sm")
                    with gr.Tab("Bundled voice", id="char-voice-preset"):
                        preset = gr.Dropdown(_presets("woman")["choices"], value=None, label="Start from a bundled "
                                             "voice", interactive=True, allow_custom_value=True)
                        use_preset_btn = gr.Button("Use this voice", size="sm")
                voice_status = gr.Markdown(initial.voice_status, elem_classes=["status-box"])
                voice_player = gr.Audio(label="Character voice", type="filepath", interactive=False, buttons=[],
                                        waveform_options=waveform(), elem_id="char-voice-player")
                voice_about = gr.Markdown(initial.voice_about, elem_classes=["hint"])
            with gr.Column(elem_classes=["char-step"]):
                section("5. Save")
                checklist = gr.Markdown(initial.checklist, elem_classes=["status-box"])
                with gr.Row(elem_classes=["toolbar"]):
                    save = gr.Button("Save character", variant="primary", size="lg", interactive=False,
                                     elem_classes=["cta"])
                    discard = gr.Button("Discard", size="sm", variant="stop")
    ticker = gr.Timer(2.0, active=False)  # runs only while a photo or voice job of the draft waits or runs

    draft = gr.State("", delete_callback=_forget_draft)
    watch = gr.State({})
    shown = gr.State(initial)
    display = [heading, photos, photo_pick, photo_status, generate, keep_own, regenerate, remove, upload,
               voice_player, voice_about, voice_status, design, clone, use_preset_btn, checklist, save, ticker]
    form = [gender, look, count, name, voice_text, sample, voice_consent, photo_consent, preset, style, kind]
    # the owner's styles are made on the Styles page: read again on every visit, keeping the choice
    tab.select(lambda current: gr.update(choices=style_library.choices(current), value=current), inputs=style,
               outputs=style, api_visibility="private", show_progress="hidden")
    # a change of kind, by the owner or by Edit, Save or Discard filling the form: the words that follow it, and an
    # animated style for an animal or creature (the draft takes the kind with the next photo step or Save)
    kind.change(_kind_view, inputs=[kind, style], outputs=[gender, look, style, photo_step, keep_own, photo_consent],
                api_visibility="private", show_progress="hidden")
    state_out = [draft, watch, *display, shown]
    creator = dict(api_visibility="private", concurrency_id=SERIAL, concurrency_limit=1)

    generate.click(generate_photos, inputs=[draft, watch, shown, name, gender, look, count, keep_own, photo_pick, style,
                                            kind],
                   outputs=state_out, **creator)
    regenerate.click(regenerate_photo, inputs=[draft, watch, shown, name, gender, look, photo_pick, style, kind],
                     outputs=state_out, **creator)
    remove.click(remove_photo, inputs=[draft, watch, shown, name, look, photo_pick], outputs=state_out, **creator)
    upload.upload(upload_photos, inputs=[draft, watch, shown, name, gender, look, upload, photo_consent, photo_pick,
                                         style, kind],
                  outputs=state_out, **creator)
    photos.select(_select_photo, outputs=photo_pick, api_visibility="private", show_progress="hidden")
    design.click(design_voice, inputs=[draft, watch, shown, name, gender, look, voice_text, photo_pick],
                 outputs=state_out, **creator)
    clone.click(clone_voice, inputs=[draft, watch, shown, name, gender, look, sample, voice_consent, photo_pick],
                outputs=[*state_out, sample], **creator)
    use_preset_btn.click(use_preset, inputs=[draft, watch, shown, name, gender, look, preset, photo_pick],
                         outputs=state_out, **creator)
    gender.change(_presets, inputs=gender, outputs=preset, api_visibility="private", show_progress="hidden")
    gr.on([ticker.tick, name.input, look.input], refresh_creator,
          inputs=[draft, watch, shown, name, look, photo_pick], outputs=[watch, *display, shown],
          show_progress="hidden", trigger_mode="always_last", **creator)

    library_out = [cards, listing, pick, assets, top_picker]
    save.click(save_character, inputs=[draft, watch, shown, name, gender, look, photo_pick, style, kind],
               outputs=[*state_out, *form, *library_out], **creator)
    confirm = gr.Textbox(visible=False)
    for button in (discard, new):
        button.click(None, outputs=confirm, js=CONFIRM_JS).then(
            start_over, inputs=[confirm, draft, watch, shown], outputs=[*state_out, *form], **creator)
    edit.click(None, outputs=confirm, js=CONFIRM_JS).then(
        edit_character, inputs=[confirm, pick, draft, watch, shown], outputs=[*state_out, *form, info],
        **creator).then(None, inputs=confirm, js=SHOW_EDITED_JS)
    new.click(None, js=SHOW_CREATOR_JS)

    # ----- the list -----
    def choose_universe(slug):
        group = universes.get(slug or "")
        if slug and not group:
            return note("❌ Choose an existing universe."), gr.skip()
        message = f"“{group['name']}” added to the Timeline cast." if group else "Timeline universe cleared."
        missing = universes.notice(group) if group else ""  # members deleted or archived since it was saved
        return note(" ".join(part for part in (message, missing) if part)), gr.update(
            choices=ui_timeline.universe_choices(), value=slug or "")

    universe_use.click(choose_universe, inputs=universe_pick, outputs=[universe_info, timeline_universe])
    tab.select(lambda slug: gr.update(choices=ui_timeline.universe_choices(slug), value=slug or ""),
               inputs=timeline_universe, outputs=universe_pick)

    def refresh(sig, current):
        """Every few seconds: the library; nothing is redrawn while it is unchanged."""
        gallery, text, names = _rows()
        new_sig = json.dumps([text, gallery, names, _character_assets(current)], ensure_ascii=False)
        if new_sig == sig:
            return (gr.skip(),) * 5
        slugs = [s for _, s in names]
        value = current if current in slugs else (slugs[0] if slugs else None)
        return gallery, text, gr.update(choices=names, value=value), _character_assets(value), new_sig

    sig = gr.State("")
    gr.on([tab.select, gr.Timer(4.0).tick], refresh, inputs=[sig, pick],
          outputs=[cards, listing, pick, assets, sig], api_visibility="private", show_progress="hidden")
    pick.change(_character_assets, inputs=pick, outputs=assets, api_visibility="private")

    def on_card(evt: gr.SelectData):
        chars = [c for c in characters.all_characters() if _cover(c)]
        return chars[evt.index]["slug"] if 0 <= evt.index < len(chars) else gr.skip()

    cards.select(on_card, outputs=pick, api_visibility="private")

    def use_it(slug, selected):
        ch = characters.timeline_profile(slug or "")
        if not ch:
            return note("❌ This character needs at least one reference photo. Edit it and add photos."), gr.skip()
        selected = list(dict.fromkeys([*(selected or []), slug]))
        choices = [(person["name"], person["slug"]) for person in characters.all_characters()]
        return note(f"✅ “{ch['name']}” added to the Timeline cast."), gr.update(choices=choices, value=selected)

    use.click(use_it, inputs=[pick, timeline_additions], outputs=[info, timeline_additions])

    def delete_character(slug, typed):
        """Deletes only after the phrase DELETE is typed - a click alone may be an accident."""
        phrase = (typed or "").strip().upper()
        if not slug:
            return note("❌ Select a character to delete."), *(gr.skip(),) * 5
        if phrase != "DELETE":
            return "Nothing deleted. The confirmation was cancelled or did not match DELETE.", *(gr.skip(),) * 5
        msg = characters.delete(slug)
        return (note(msg), *_library_views(None),
                gr.update(choices=cast.top_picker_choices(), value=cast.top_picker_value()))

    # a browser prompt: Cancel, or type DELETE and OK -> a hidden field -> the deletion
    typed = gr.Textbox(visible=False)
    delete.click(None, outputs=typed, js=DELETE_JS).then(delete_character, inputs=[pick, typed],
                                                         outputs=[info, *library_out])
    return leave, [draft, watch, shown], [*state_out, *form]


def build_universes_tab(tab: gr.Tab, top_picker: gr.Dropdown, timeline_cast) -> None:
    """Manage saved groups of characters."""
    page_head("Universes", "Group characters into a cast for Timeline, Video and Image.")
    timeline_universe, _ = timeline_cast
    table_view, universe_field, name_value, members_field = universe_views()
    with gr.Column(elem_classes=["card"]):
        u_list = gr.HTML(table_view)
        with gr.Row():
            u_pick = gr.Dropdown(universe_field["choices"], value=universe_field["value"],
                                 label="Universe", interactive=True,
                                 allow_custom_value=True, scale=1)
            u_name = gr.Textbox(value=name_value, label="Name", placeholder="For example, Family",
                                max_lines=1, scale=1)
            u_members = gr.Dropdown(members_field["choices"], value=members_field["value"],
                                    multiselect=True, label="Characters",
                                    max_choices=cast.MAX_CAST, interactive=True,
                                    allow_custom_value=True, scale=2)
        with gr.Row(elem_classes=["toolbar"]):
            u_save_btn = gr.Button("Save universe", size="sm", variant="primary")
            u_timeline_btn = gr.Button("Use in Timeline", size="sm")
            u_use_btn = gr.Button("Use in Video and Image", size="sm")
            u_delete_btn = gr.Button("Delete universe", size="sm", variant="stop")
            u_info = gr.Markdown(elem_classes=["status-box", "grow"])
    u_outputs = [u_list, u_pick, u_name, u_members]

    def u_save(slug, name_v, members_v):
        saved, msg = universes.save(name_v, members_v or [], slug or "")
        if not saved:
            return (note(f"❌ {msg}"), *([gr.skip()] * len(u_outputs)), gr.skip())
        picker = gr.update(choices=cast.top_picker_choices(), value=cast.top_picker_value())
        return (note(f"✅ {msg} Click “Use in Video and Image” (or select it in the top bar’s Tools) "
                     "to load its characters there."), *universe_views(saved["slug"]),
                picker)

    u_save_btn.click(u_save, inputs=[u_pick, u_name, u_members],
                     outputs=[u_info, *u_outputs, top_picker])

    def u_use(slug):
        u = universes.get(slug or "")
        if not u:
            return note("❌ Save the universe first, then use it."), gr.skip()
        main = cast.select(cast.UNIVERSE + u["slug"])
        names = ", ".join(c["name"] for c in universes.members(u)) or "none ready yet"
        return (note(f"✅ Universe ‘{u['name']}’ selected ({names})."
                     + ("" if main else " None of its characters has reference photos yet.")
                     + (" " + universes.notice(u) if universes.notice(u) else "")),
                gr.update(choices=cast.top_picker_choices(), value=cast.UNIVERSE + u["slug"]))

    u_use_btn.click(u_use, inputs=u_pick, outputs=[u_info, top_picker])

    def u_timeline(slug):
        group = universes.get(slug or "")
        if not group:
            return note("❌ Choose a saved universe."), gr.skip()
        return note(f"“{group['name']}” added to the Timeline cast."), gr.update(
            choices=ui_timeline.universe_choices(), value=slug)

    u_timeline_btn.click(u_timeline, inputs=u_pick, outputs=[u_info, timeline_universe])

    def u_remove(slug, confirmed_v):
        if not slug:
            return (note("❌ Select a universe to delete."), *([gr.skip()] * len(u_outputs)),
                    gr.skip())
        if confirmed_v != "yes":
            return ("Nothing deleted.", *([gr.skip()] * len(u_outputs)), gr.skip())
        if (cast.current_universe() or {}).get("slug") == slug:
            cast.select(prefs.load().get("character", ""))  # the top picker back to its main character
        ok, msg = universes.delete(slug)
        return (note(("✅ " if ok else "❌ ") + msg), *universe_views(""),
                gr.update(choices=cast.top_picker_choices(), value=cast.top_picker_value()))

    u_confirm = gr.Textbox(visible=False)
    u_delete_btn.click(None, inputs=u_pick, outputs=u_confirm,
                       js="(slug) => slug ? (confirm('Delete this universe? Its characters stay "
                          "in the library.') ? 'yes' : 'no') : 'yes'").then(
        u_remove, inputs=[u_pick, u_confirm], outputs=[u_info, *u_outputs, top_picker])
    u_pick.input(lambda slug: universe_views(slug or "")[1:], inputs=u_pick,
                 outputs=[u_pick, u_name, u_members])
    tab.select(lambda: universe_views(""), outputs=u_outputs)
