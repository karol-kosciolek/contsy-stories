# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Image page: a prompt, an aspect ratio and optional reference images.

Each model draws at its own resolution. Without reference images Z-Image-Turbo (8 steps) renders
1080p directly. With reference images (for example a person [1] in a hoodie [2]) or several
characters, Qwen-Image-Edit-2511 composes them by the prompt at about 1 megapixel
(image_gen.EDIT_PIXELS). A selected character (top bar) is composed from its reference photo at that
size too, and with reference images it is the base of the composition. Models download on first use; the
style is fixed - ultra realistic.
"""
from __future__ import annotations

from typing import Optional

import gradio as gr

from . import engine_runtime, image_gen, model_manager, prefs, queue_manager, quick_jobs, refs
from . import ui_assist, ui_queue, ui_video
from . import cast as cast_mod
from .ui_theme import note, page_head, section
from .ui_video import gallery_paths

# The size Z-Image renders (shorter side 1080 px); a composition keeps only the aspect ratio
SIZES = {"Square 1:1": (1080, 1080), "Landscape 16:9": (1920, 1080), "Portrait 9:16": (1080, 1920),
         "Portrait 3:4": (1080, 1440)}
SIZE_INFO = ("Without reference images or characters the image is 1080p. With reference images or characters it "
             "is about 1 megapixel (1360×768 in 16:9), the size the composing model works at.")


def image_size(format_name: str | None) -> tuple[int, int]:
    return SIZES.get(format_name, SIZES["Square 1:1"])


def _up16(v: int) -> int:
    return (v + 15) // 16 * 16


def build_images_tab(tab: Optional[gr.Tab] = None, top_picker: Optional[gr.Dropdown] = None):
    """Buduje strone. Zwraca (funkcja_odswiezania, wejscia, wyjscia) dla auto-odswiezania."""
    page_head("Image", "Describe a scene to create an image in the queue. With reference images, "
                         "AI combines people, clothing and locations according to your prompt.")
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["card"]):
            positive = gr.Textbox(
                label="Image prompt", lines=4, max_lines=12,
                info="Write in any language; the AI assistant can refine your idea into English. Name your "
                     "characters to put them in the picture. Refer to images as [1], [2], [3], for example: the man "
                     "[1] wearing the hoodie [2] in front of the building [3].",
                placeholder="A cozy coffee shop at night with a neon sign, rain on the window")
            wire_assistant = ui_assist.build_assistant("image", positive)
            cast_pick = ui_video.cast_picker("Characters in this photo", ui_video.PHOTO_CAST_INFO)
            fmt = gr.Radio(list(SIZES), value=next(iter(SIZES)), label="Aspect ratio", info=SIZE_INFO)
            with gr.Accordion("Reference images (optional)", open=False):
                ref_gallery = gr.Gallery(
                    label="Reference images (up to 10): [1], [2], [3]… in upload order",
                    type="filepath", interactive=True, columns=5, height=150,
                    file_types=["image"], object_fit="contain")
                wire_assistant(ref_gallery)
            generate = gr.Button("Generate image", variant="primary", size="lg",
                                 elem_classes=["cta"])
        with gr.Column(scale=4, elem_classes=["card", "result-card"]):
            section("Result", "Your latest completed image. Browse all results in Gallery.")
            gallery = gr.Gallery(show_label=False, columns=1, height=440, preview=True,
                                 buttons=["download", "fullscreen"])
            status = gr.Markdown(elem_classes=["status-box"])

    def add(*values):
        return note(_add(*values))

    def _add(pos_v, refs_v, fmt_v, cast_v=None):
        """Tylko dodaje zadanie do kolejki - przycisk od razu jest gotowy na kolejne."""
        try:
            queued = quick_jobs.queue_image(pos_v, gallery_paths(refs_v), fmt_v, cast_v)
        except quick_jobs.Refused as refused:
            return refused.shown
        return ui_queue.added_message(queued.job) + ui_video.cast_reply(queued.found, queued.notes)

    generate.click(add, inputs=[positive, ref_gallery, fmt, cast_pick], outputs=status,
                   trigger_mode="multiple")  # kazde klikniecie = zadanie
    last = gr.State(getattr(queue_manager.latest_finished("obraz"), "id", None))
    gr.Timer(3.0).tick(ui_queue.result_poller("obraz"), inputs=last,
                       outputs=[gallery, status, last])
    if tab is not None:
        tab.select(lambda: ui_video.cast_view(), outputs=cast_pick)
    if top_picker is not None:  # the top picker's new value itself (saving it runs in parallel)
        top_picker.input(lambda v: ui_video.cast_view(v or ""), inputs=top_picker, outputs=cast_pick)
    return lambda: ui_video.cast_view(), [], [cast_pick]


def run_job(p: dict, cb) -> tuple:
    """Wykonanie zadania z kolejki (watek roboczy) -> (ok, pliki, uwagi, blad)."""
    pos_v, ref_paths = p["prompt"], p.get("refs") or []
    width, height = image_size(p.get("format"))
    chs, ch_note = cast_mod.for_job(p)
    # who is in the picture: the characters it names, otherwise the main one (app/cast.py)
    members = cast_mod.shot_casts([pos_v], chs)[0] if chs else []
    ch = members[0] if members else None
    text = cast_mod.untag(pos_v, chs)  # "@kuba" -> "Kuba": models never see the @ syntax
    prompt = prefs.styled(text, "image")
    if ref_paths or members:
        used = [n for n in refs.numbers(pos_v) if n <= len(ref_paths)] or \
            list(range(1, len(ref_paths) + 1))
        used = used[:refs.MAX_REFS]
        out = image_gen.edit_with_refs([(n, ref_paths[n - 1]) for n in used], prompt, width / height, cb,
                                       character=ch, cast=members if len(members) > 1 else None)
        msg = out.error  # with a result: the notes of the composition
    else:
        err = model_manager.ensure([engine_runtime.ZIMAGE_ID], cb)
        if err:
            return False, [], [], err
        out = image_gen.txt2img(prompt, _up16(width), _up16(height), prefs.IMAGE_STEPS, -1, cb)
        for f in out.files if out.ok else []:
            image_gen.crop_to(f, width, height, {"positive": prompt})
        msg = "" if out.ok else out.error
    files = list(out.files)
    if not out.ok:
        return False, [], [], msg
    notes = [n for n in [ch_note, msg, prefs.autosave(files)] if n]
    return True, files, notes, ""


queue_manager.register("obraz", run_job)
