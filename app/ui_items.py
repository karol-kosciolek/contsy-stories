# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Items page: the library of places and items beside the one-screen creator, built like the Characters page
(app/ui_characters.py) on the drafts of app/items.py (its API is documented there).

An item is made top to bottom: what it is (a place or an object) with its style and description; the reference photos
(generate 1 to 8, then review them: generate one again, remove one, upload your own with the consent box); the name;
Save. Nothing reaches the library before Save. The draft stays in the creator while other pages are visited and is
thrown away by Discard, New or Edit (the browser asks first when it holds unsaved work) and when the browser tab is
gone (the draft State's delete callback). Photo generation is a GPU queue job; the screen follows it with a timer that
runs only while it waits or runs. Every creator event shares one concurrency id, so two events of a browser tab never
write its draft state at once.

In the Timeline a scene uses an item's main photo as a reference image: picked under "Places and items in this scene",
or named in the scene's picture text (app/timeline.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gradio as gr

from . import items, queue_manager, style_library, styles
from .ui_theme import badge, cache_media, esc, note, page_head, plain, section, table

KINDS = [("Place", items.PLACE), ("Object", items.OBJECT)]
KIND_LABELS = {items.PLACE: "Place", items.OBJECT: "Object"}
LOOK_EXAMPLES = {
    items.PLACE: ("For example: a modern TV news studio, a curved glossy blue anchor desk, a big screen with a world "
                  "map behind it, blue and white studio lights"),
    items.OBJECT: "For example: a vintage red bicycle with a wicker basket and a brass bell",
}
PHOTO_RULES = ("Generate photos from the description, then review them. The first photo is the main one: scenes are "
               "composed from it. A photo that shows someone under 18 is refused.")
KEEP_LABEL = "Keep my own photos and add views of the first up to the number of photos (untick to generate a new one)"
DEFAULT_PHOTOS = 3
CREATOR = "item-creator"  # the creator's card (scrolled into view when an item is loaded into it)
SERIAL = "item-creator"
_BUSY = (queue_manager.PENDING, queue_manager.RUNNING)
_ORIGINS = {"uploaded": "your photo", "kept": "saved"}
_UNSAVED = f"#{CREATOR} [data-unsaved]"
DISCARD_QUESTION = ("Discard the place or item you are making? Its photos are not saved, and photos still being made "
                    "are stopped.")
CONFIRM_JS = f"() => (!document.querySelector('{_UNSAVED}') || confirm('{DISCARD_QUESTION}')) ? 'yes' : 'no'"
_SCROLL = f"document.getElementById('{CREATOR}')?.scrollIntoView({{behavior: 'smooth', block: 'start'}})"
SHOW_CREATOR_JS = f"() => {_SCROLL}"
SHOW_EDITED_JS = f"(answer) => {{ if (answer === 'yes') {_SCROLL}; }}"
DELETE_JS = ("() => prompt('Delete the selected place or item with its photos? It goes to the Recycle Bin.\\nTo "
             "confirm, type: DELETE') || ''")


# ----------------------------- the library -----------------------------

def _cover(item: dict) -> str:
    return cache_media(items.main_photo(item)) or ""


def _rows():
    """(gallery, listing, choices) for the page."""
    found = items.all_items()
    gallery = [(path, f"{item['name']} · {KIND_LABELS[item['kind']]}") for item in found if (path := _cover(item))]
    rows = [[f"<b>{esc(item['name'])}</b><br><span class='muted'>{esc(style_library.label(item.get('style')))}</span>",
             esc(KIND_LABELS[item["kind"]]), badge(items.state(item), "ok" if item.get("photos") else "warn"),
             f"<span class='num'>{len(item['photos'])}</span>"] for item in found]
    listing = table(["Place or item", "Kind", "Status", "Photos"], rows,
                    empty="<b>No places or items yet</b>Make your first one in the creator, for example the TV "
                          "studio your characters often appear in.")
    return gallery, listing, [(item["name"], item["slug"]) for item in found]


def _item_assets(slug) -> str:
    item = items.get(slug or "")
    if not item:
        return "<p class='section-sub'>Select a place or item to see its photos.</p>"
    count = len(item["photos"])
    photos = f"{count} reference photo" + ("" if count == 1 else "s")
    return (f"<div class='char-assets'><strong>{esc(item['name'])}</strong><div class='char-asset-badges'>"
            f"{badge(KIND_LABELS[item['kind']])}{badge(style_library.label(item.get('style')))}"
            f"{badge(photos, 'ok' if count else 'warn')}</div></div>")


def _library_views(selected: Optional[str]):
    """The library's gallery, table, picker and the selected item's badges, fresh."""
    gallery, listing, names = _rows()
    slugs = [slug for _, slug in names]
    value = selected if selected in slugs else (slugs[0] if slugs else None)
    return gallery, listing, gr.update(choices=names, value=value), _item_assets(value)


# ----------------------------- the creator: what it shows -----------------------------

@dataclass(frozen=True)
class Creator:
    """What the creator shows, kept per browser tab: a redraw sends only the parts that changed, so a photo being
    looked at is not reset by a progress update."""
    heading: str = ""
    photos: tuple = ()          # (draft path, caption) in order; the first is the main photo
    photo_status: str = ""
    photo_busy: bool = False
    own_photos: bool = False    # uploaded or saved photos are in the draft: Generate may keep them
    checklist: str = ""
    can_save: bool = False


def _draft(draft_id: str) -> Optional[dict]:
    """items.draft(draft_id), or None when there is none or it was closed."""
    if not draft_id:
        return None
    try:
        return items.draft(draft_id)
    except ValueError:
        return None


def _watched(watch: dict) -> Optional[queue_manager.Job]:
    """The photo job this tab started, as the queue has it now (None when gone)."""
    job_id = watch.get("photos")
    return next((job for job in queue_manager.snapshot() if job.id == job_id), None) if job_id else None


def _job_status(job: queue_manager.Job) -> str:
    """The photo job in plain words: waiting, its progress, its result or why it failed."""
    if job.status == queue_manager.PENDING:
        if queue_manager.paused():
            return note("⚠ Waiting: the GPU queue is paused. Click “Resume generation” in the top bar.")
        ahead = queue_manager.position(job.id)
        behind = f" behind {ahead} job{'s' if ahead != 1 else ''}" if ahead else ""
        return f"Waiting in the GPU queue{behind}. It takes about {queue_manager.fmt_time(job.estimate())}."
    if job.status == queue_manager.RUNNING:
        detail = plain(job.desc).strip().rstrip(".…")
        return f"Making the photos… {round(job.progress * 100)}%" + (f" · {esc(detail)}" if detail else "")
    if job.status == queue_manager.DONE:
        first, *rest = [plain(text) for text in job.notes if text] or ["Done."]
        return note(f"✅ {first}") + ("\n\n" + "\n".join(f"- {esc(text)}" for text in rest) if rest else "")
    if job.status == queue_manager.STOPPED:
        return note(f"⏹ Stopped. {job.error}".strip())
    return note(f"❌ {job.error or 'It failed. Try again.'}")


def _heading(data: Optional[dict], unsaved: bool) -> str:
    saved = items.get(data["slug"]) if data and data.get("slug") else None
    title = f"Edit “{esc(saved['name'])}”" if saved else "New place or item"
    marker = " data-unsaved" if unsaved else ""
    return (f"<div class='char-creator-head'{marker}><h2 class='section-title'>{title}</h2>"
            "<p class='section-sub'>Work from top to bottom. Nothing is saved until you click Save; Discard throws "
            "the draft away.</p></div>")


def _unsaved(data: Optional[dict], name: str = "", look: str = "") -> bool:
    """Discarding the draft would lose work: a photo job of it waits or runs, or it differs from what is saved (a new
    item: any photo, name or description)."""
    if not data:
        return False
    if data["busy"]:
        return True
    name, look = " ".join((name or "").split()), " ".join((look or "").split())
    saved = items.get(data["slug"]) if data.get("slug") else None
    if saved is None:
        return bool(data["photos"] or name or look)
    return not (len(data["photos"]) == len(saved["photos"]) and all(origin == "kept" for origin in data["origins"])
                and name == saved["name"] and look == saved["look"] and data["kind"] == saved["kind"]
                and data["style"] == saved["style"])


def _checklist(data: Optional[dict], name: str, look: str, busy: bool, message: str) -> tuple[str, bool]:
    """(what Save still needs, in plain words; whether Save may be clicked)."""
    photos = len(data["photos"]) if data else 0
    missing = [what for what, ok in (("a description", (look or "").strip()), ("at least one photo", photos),
                                     ("a name", (name or "").strip())) if not ok]
    if busy:
        text = "Save is ready once the photos being made are done."
    elif missing:
        text = "To save, add " + (", ".join(missing[:-1]) + " and " if len(missing) > 1 else "") + missing[-1] + "."
    else:
        text = f"Ready to save: {photos} photo{'s' if photos != 1 else ''}, a name and a description."
    return (note(message) + "\n\n" + text if message else text), not missing and not busy


def _view(draft_id: str, watch: dict, name: str, look: str) -> Creator:
    data = _draft(draft_id)
    job = _watched(watch)
    busy = bool(job and job.status in _BUSY and data and job.id in data["busy"])
    photos = tuple(zip(data["photos"], data["origins"])) if data else ()
    captions = tuple((path, " · ".join(part for part in (str(number), "main" if number == 1 else "",
                                                          _ORIGINS.get(origin, "")) if part))
                     for number, (path, origin) in enumerate(photos, 1))
    checklist, can_save = _checklist(data, name, look, busy, watch.get("note", ""))
    return Creator(heading=_heading(data, _unsaved(data, name, look)), photos=captions,
                   photo_status=_job_status(job) if job else note(watch.get("photo_note", "")),
                   photo_busy=busy, own_photos=any(origin in ("uploaded", "kept") for _, origin in photos),
                   checklist=checklist, can_save=can_save)


# The creator's display, in the order every creator event returns it (then the Creator itself, kept per tab).
DISPLAY = ("heading", "photos", "pick", "photo_status", "generate", "keep", "regenerate", "remove", "upload",
           "checklist", "save", "ticker")


def _updates(view: Creator, prior: Creator, pick=None, *, force: bool = False, clear_pick: bool = False) -> tuple:
    """An update for every DISPLAY part whose content changed (all of them with force); gr.skip() keeps the others,
    so the selected photo and an open photo stay as they are."""
    def changed(*names):
        return force or any(getattr(view, name) != getattr(prior, name) for name in names)

    skip = gr.skip()
    count, has = len(view.photos), bool(view.photos)
    photo_idle = gr.update(visible=has, interactive=not view.photo_busy)
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
        "generate": (gr.update(value="Generate all again" if has and not view.own_photos else "Generate photos",
                               interactive=not view.photo_busy)
                     if changed("photos", "own_photos", "photo_busy") else skip),
        "keep": (gr.update(visible=view.own_photos, **({"value": True} if force else {}))
                 if changed("own_photos") else skip),
        "regenerate": photo_idle if changed("photos", "photo_busy") else skip,
        "remove": photo_idle if changed("photos", "photo_busy") else skip,
        "upload": gr.update(interactive=not view.photo_busy) if changed("photo_busy") else skip,
        "checklist": view.checklist if changed("checklist") else skip,
        "save": gr.update(interactive=view.can_save) if changed("can_save") else skip,
        "ticker": gr.Timer(active=view.photo_busy) if changed("photo_busy") else skip,
    }
    return tuple(parts[name] for name in DISPLAY)


def _render(draft_id: str, watch: dict, prior: Creator, name: str, look: str, pick=None, *,
            force: bool = False, clear_pick: bool = False) -> tuple:
    """The DISPLAY updates and the new Creator (for the per-tab State)."""
    view = _view(draft_id, watch, name, look)
    return (*_updates(view, prior or Creator(), pick, force=force, clear_pick=clear_pick), view)


# ----------------------------- the creator: actions -----------------------------

def _ensure(draft_id: str, name: str, kind: str, look: str, style: Optional[str]) -> str:
    """The tab's draft with the form's fields remembered; a new draft when there is none (or it was closed). A draft
    made for a call that fails is thrown away again."""
    made = _draft(draft_id) is None
    if made:
        draft_id = items.new_draft()
    try:
        items.update_draft(draft_id, name=name or "", kind=kind if kind in items.KINDS else items.PLACE,
                           look=look or "", style=style if styles.valid(style) else None)
    except (ValueError, OSError):
        if made:
            _discard(draft_id)
        raise
    return draft_id


def _discard(draft_id: str) -> None:
    if draft_id:
        try:
            items.discard_draft(draft_id)
        except (ValueError, OSError):
            pass  # already gone, or a file still open: the draft folder is removed after a day anyway


def _form(data: Optional[dict] = None) -> tuple:
    """The creator's inputs for a draft (None: an empty creator), in FORM_NAMES order."""
    data = data or {}
    kind = data.get("kind") if data.get("kind") in items.KINDS else items.PLACE
    count = min(items.MAX_PHOTOS, max(DEFAULT_PHOTOS, len(data.get("photos") or ()) + 1))
    style = data.get("style") if styles.valid(data.get("style")) else styles.DEFAULT
    return (kind, gr.update(value=data.get("look", ""), placeholder=LOOK_EXAMPLES[kind]), count, data.get("name", ""),
            False, gr.update(value=style, choices=style_library.choices(style)))


FORM_NAMES = ("kind", "look", "count", "name", "photo_consent", "style")


def generate_photos(draft_id, watch, prior, name, kind, look, count, keep, pick, style):
    """Generate: a new place or object (or, with own photos kept, views of the first of them) as a GPU queue job."""
    try:
        draft_id = _ensure(draft_id, name, kind, look, style)
        data = items.draft(draft_id)
        own = any(origin in ("uploaded", "kept") for origin in data["origins"])
        job = items.enqueue_photos(draft_id, int(count or DEFAULT_PHOTOS), keep_own=bool(keep) and own)
        watch = {**watch, "photos": job.id, "photo_note": ""}
    except (ValueError, OSError) as error:
        watch = {**watch, "photos": None, "photo_note": f"❌ {error}"}
    return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick))


def _picked(pick) -> int:
    if not isinstance(pick, int):
        raise ValueError("Select a photo first: click it in the grid or choose its number.")
    return pick


def regenerate_photo(draft_id, watch, prior, name, kind, look, pick, style):
    """A new take of the selected generated photo, as a GPU queue job."""
    try:
        draft_id = _ensure(draft_id, name, kind, look, style)
        job = items.enqueue_regenerate(draft_id, _picked(pick))
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
        items.remove_photo(draft_id, index)
        message = f"✅ Photo {index + 1} removed."
    except (ValueError, OSError) as error:
        message = f"❌ {error}"
    watch = {**watch, "photos": None, "photo_note": message}
    return (draft_id, watch, *_render(draft_id, watch, prior, name, look, clear_pick=True))


def upload_photos(draft_id, watch, prior, name, kind, look, files, consent, pick, style):
    """Your own photos: each needs the consent box and passes the photo check (no one who looks under 18)."""
    files = [str(file) for file in (files if isinstance(files, list) else [files] if files else [])]
    if not files:
        return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick))
    if not consent:
        watch = {**watch, "photos": None,
                 "photo_note": "❌ Tick the box to confirm that you may use the photos, then upload them again."}
        return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick))
    try:
        draft_id = _ensure(draft_id, name, kind, look, style)
        added, refused = 0, []
        for file in files:
            ok, message = items.add_upload(draft_id, file, True)
            if ok:
                added += 1
            else:
                refused.append(f"{Path(file).name}: {plain(message)}")
        lead = f"✅ {added} photo{'s' if added != 1 else ''} added." if added else "❌ No photo was added."
        message = lead + ("\n\n" + "\n".join(f"- {esc(line)}" for line in refused) if refused else "")
    except (ValueError, OSError) as error:
        message = f"❌ {error}"
    watch = {**watch, "photos": None, "photo_note": message}
    return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick))


def refresh_creator(draft_id, watch, prior, name, look, pick):
    """The timer while a job runs, and typing in the name or description (the Save button and its checklist)."""
    if watch.get("note"):  # a refusal of Save is about the values before this change
        watch = {**watch, "note": ""}
    return (watch, *_render(draft_id, watch, prior, name, look, pick))


def save_item(draft_id, watch, prior, name, kind, look, pick, style):
    """Save: photos, name, kind, description and style together. On success the creator starts a new item and the
    library selects the saved one; on a refusal everything stays as it is."""
    try:
        if _draft(draft_id) is None:
            raise ValueError("Generate or upload photos first.")
        profile, message = items.save_draft(draft_id, name=name or "", kind=kind if kind in items.KINDS else None,
                                            look=look or "", style=style if styles.valid(style) else None)
    except (ValueError, OSError) as error:
        profile, message = None, str(error)
    keep = (gr.skip(),) * (len(FORM_NAMES) + 4)
    if profile is None:
        watch = {**watch, "note": f"❌ {message}"}
        return (draft_id, watch, *_render(draft_id, watch, prior, name, look, pick), *keep)
    watch = {"note": f"✅ {message}"}
    return ("", watch, *_render("", watch, prior, "", "", force=True), *_form(), *_library_views(profile["slug"]))


def start_over(confirmed, draft_id, watch, prior):
    """Discard or New (after the browser's confirmation when the draft holds unsaved work)."""
    if confirmed != "yes":
        return (gr.skip(),) * (2 + len(DISPLAY) + 1 + len(FORM_NAMES))
    _discard(draft_id)
    return ("", {}, *_render("", {}, prior, "", "", force=True), *_form())


def edit_item(confirmed, slug, draft_id, watch, prior):
    """Load a copy of the selected item into the creator (after the browser's confirmation when the current draft
    holds unsaved work)."""
    skip_all = (gr.skip(),) * (2 + len(DISPLAY) + 1 + len(FORM_NAMES))
    if confirmed != "yes":
        return (*skip_all, gr.skip())
    item = items.get(slug or "")
    if not item:
        return (*skip_all, note("❌ Select a place or item to edit."))
    _discard(draft_id)
    try:
        draft_id = items.load_draft(slug)
        data = items.draft(draft_id)
    except (ValueError, OSError) as error:
        return ("", {}, *_render("", {}, prior, "", "", force=True), *_form(), note(f"❌ {error}"))
    watch = {}
    return (draft_id, watch, *_render(draft_id, watch, prior, data["name"], data["look"], force=True),
            *_form(data), note(f"✅ “{item['name']}” is open in the creator."))


def delete_item(slug, typed):
    """Deletes only after the phrase DELETE is typed - a click alone may be an accident."""
    if not slug:
        return note("❌ Select a place or item to delete."), *(gr.skip(),) * 4
    if (typed or "").strip().upper() != "DELETE":
        return "Nothing deleted. The confirmation was cancelled or did not match DELETE.", *(gr.skip(),) * 4
    message = items.delete(slug)
    return (note(message), *_library_views(None))


def _select_photo(evt: gr.SelectData):
    return evt.index if isinstance(evt.index, int) else gr.skip()


def _forget_draft(draft_id) -> None:
    """The browser tab is gone (Gradio deletes its state an hour after it closes): so is its draft."""
    if isinstance(draft_id, str):
        _discard(draft_id)


def _kind_words(kind):
    """The description's example follows the kind."""
    return gr.update(placeholder=LOOK_EXAMPLES.get(kind, LOOK_EXAMPLES[items.PLACE]))


# ----------------------------- the page -----------------------------

def build_items_tab(tab: gr.Tab) -> None:
    """The library of places and items beside the one-screen creator."""
    page_head("Places and items", "Reference photos of a place your characters often appear in, such as a TV studio, "
                                  "or of an object. Pick them for a scene in the Timeline, or write their name in "
                                  "the scene's picture text.")
    initial = _view("", {}, "", "")
    with gr.Row(equal_height=False, elem_classes=["char-layout"]):
        with gr.Column(scale=2, min_width=300, elem_classes=["card"]):
            section("Your places and items", "Select one to edit or delete it.")
            gallery0, listing0, names0 = _rows()
            cards = gr.Gallery(value=gallery0, show_label=False, columns=3, height=300, allow_preview=False,
                               object_fit="cover", elem_id="item-cards", buttons=[])
            listing = gr.HTML(listing0)
            pick = gr.Dropdown(names0, value=names0[0][1] if names0 else None, label="Selected place or item",
                               interactive=True, allow_custom_value=True)
            assets = gr.HTML(_item_assets(names0[0][1] if names0 else ""), padding=False)
            with gr.Row(elem_classes=["toolbar"]):
                edit = gr.Button("Edit", size="sm", variant="primary")
                delete = gr.Button("Delete", size="sm", variant="stop")
            info = gr.Markdown(elem_classes=["status-box"])
        with gr.Column(scale=3, min_width=320, elem_id=CREATOR, elem_classes=["card"]):
            with gr.Row(elem_classes=["toolbar", "char-creator-top"]):
                heading = gr.HTML(initial.heading, padding=False, elem_classes=["grow"])
                new = gr.Button("New place or item", size="sm")
            with gr.Column(elem_classes=["char-step"]):
                section("1. What it is")
                kind = gr.Radio(KINDS, value=items.PLACE, label="Kind", elem_id="item-kind",
                                info="A place is drawn as an empty wide shot, an object alone on a plain background.")
                style = gr.Dropdown(style_library.choices(), value=styles.CARTOON, label="Style",
                                    info="Draw it in the style of the films it is for. New photos use it.",
                                    elem_id="item-style")
                look = gr.Textbox(label="Description", lines=3, max_lines=8, max_length=items.MAX_LOOK,
                                  placeholder=LOOK_EXAMPLES[items.PLACE], elem_id="item-look")
            with gr.Column(elem_classes=["char-step"]):
                section("2. Reference photos", PHOTO_RULES)
                with gr.Row(elem_classes=["toolbar"]):
                    count = gr.Slider(items.MIN_PHOTOS, items.MAX_PHOTOS, value=DEFAULT_PHOTOS, step=1,
                                      label="Number of photos", elem_classes=["grow"])
                    generate = gr.Button("Generate photos", variant="primary", size="sm", elem_id="item-generate")
                keep_own = gr.Checkbox(True, label=KEEP_LABEL, visible=False)
                photo_status = gr.Markdown(initial.photo_status, elem_classes=["status-box"])
                photos = gr.Gallery(value=[], label="Reference photos", show_label=False, columns=3, height="auto",
                                    object_fit="cover", allow_preview=True, buttons=["fullscreen"], visible=False,
                                    elem_id="item-photos")
                photo_pick = gr.Radio([], value=None, label="Selected photo", visible=False, elem_id="item-photo-pick")
                with gr.Row(elem_classes=["toolbar"]):
                    regenerate = gr.Button("Generate this photo again", size="sm", visible=False)
                    remove = gr.Button("Remove this photo", size="sm", variant="stop", visible=False)
                photo_consent = gr.Checkbox(False, label=items.CONSENT)
                upload = gr.UploadButton("Upload your own photos", file_types=["image"], file_count="multiple",
                                         size="sm", elem_id="item-photo-upload")
            with gr.Column(elem_classes=["char-step"]):
                section("3. Name", "Write it in a scene's picture text to bring the place or item into the scene.")
                name = gr.Textbox(label="Name", max_lines=1, max_length=items.MAX_NAME,
                                  placeholder="For example, TV studio", elem_id="item-name")
            with gr.Column(elem_classes=["char-step"]):
                section("4. Save")
                checklist = gr.Markdown(initial.checklist, elem_classes=["status-box"])
                with gr.Row(elem_classes=["toolbar"]):
                    save = gr.Button("Save", variant="primary", size="lg", interactive=False, elem_classes=["cta"],
                                     elem_id="item-save")
                    discard = gr.Button("Discard", size="sm", variant="stop")
    ticker = gr.Timer(2.0, active=False)  # runs only while a photo job of the draft waits or runs

    draft = gr.State("", delete_callback=_forget_draft)
    watch = gr.State({})
    shown = gr.State(initial)
    display = [heading, photos, photo_pick, photo_status, generate, keep_own, regenerate, remove, upload, checklist,
               save, ticker]
    form = [kind, look, count, name, photo_consent, style]
    state_out = [draft, watch, *display, shown]
    creator = dict(api_visibility="private", concurrency_id=SERIAL, concurrency_limit=1)
    # the owner's styles are made on the Styles page: read again on every visit, keeping the choice
    tab.select(lambda current: gr.update(choices=style_library.choices(current), value=current), inputs=style,
               outputs=style, api_visibility="private", show_progress="hidden")
    kind.change(_kind_words, inputs=kind, outputs=look, api_visibility="private", show_progress="hidden")

    generate.click(generate_photos, inputs=[draft, watch, shown, name, kind, look, count, keep_own, photo_pick, style],
                   outputs=state_out, **creator)
    regenerate.click(regenerate_photo, inputs=[draft, watch, shown, name, kind, look, photo_pick, style],
                     outputs=state_out, **creator)
    remove.click(remove_photo, inputs=[draft, watch, shown, name, look, photo_pick], outputs=state_out, **creator)
    upload.upload(upload_photos, inputs=[draft, watch, shown, name, kind, look, upload, photo_consent, photo_pick,
                                         style], outputs=state_out, **creator)
    photos.select(_select_photo, outputs=photo_pick, api_visibility="private", show_progress="hidden")
    gr.on([ticker.tick, name.input, look.input], refresh_creator, inputs=[draft, watch, shown, name, look, photo_pick],
          outputs=[watch, *display, shown], show_progress="hidden", trigger_mode="always_last", **creator)

    library_out = [cards, listing, pick, assets]
    save.click(save_item, inputs=[draft, watch, shown, name, kind, look, photo_pick, style],
               outputs=[*state_out, *form, *library_out], **creator)
    confirm = gr.Textbox(visible=False)
    for button in (discard, new):
        button.click(None, outputs=confirm, js=CONFIRM_JS).then(
            start_over, inputs=[confirm, draft, watch, shown], outputs=[*state_out, *form], **creator)
    edit.click(None, outputs=confirm, js=CONFIRM_JS).then(
        edit_item, inputs=[confirm, pick, draft, watch, shown], outputs=[*state_out, *form, info],
        **creator).then(None, inputs=confirm, js=SHOW_EDITED_JS)
    new.click(None, js=SHOW_CREATOR_JS)

    # ----- the library -----
    def refresh(sig, current):
        """Every few seconds: the library; nothing is redrawn while it is unchanged."""
        gallery, text, names = _rows()
        new_sig = json.dumps([text, gallery, names, _item_assets(current)], ensure_ascii=False)
        if new_sig == sig:
            return (gr.skip(),) * 5
        slugs = [slug for _, slug in names]
        value = current if current in slugs else (slugs[0] if slugs else None)
        return gallery, text, gr.update(choices=names, value=value), _item_assets(value), new_sig

    sig = gr.State("")
    gr.on([tab.select, gr.Timer(4.0).tick], refresh, inputs=[sig, pick], outputs=[*library_out, sig],
          api_visibility="private", show_progress="hidden")
    pick.change(_item_assets, inputs=pick, outputs=assets, api_visibility="private")

    def on_card(evt: gr.SelectData):
        shown_items = [item for item in items.all_items() if _cover(item)]
        return shown_items[evt.index]["slug"] if 0 <= evt.index < len(shown_items) else gr.skip()

    cards.select(on_card, outputs=pick, api_visibility="private")
    typed = gr.Textbox(visible=False)
    delete.click(None, outputs=typed, js=DELETE_JS).then(delete_item, inputs=[pick, typed],
                                                         outputs=[info, *library_out])
