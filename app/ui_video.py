# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Video page: one prompt (picture, sound, dialogue, shots), reference images, a starting image,
duration, aspect ratio and audio.

Every video is made in the one mode of app/video_gen.py (480p, 30 FPS). "Generate" only adds a job to
the queue (app/queue_manager.py); the finished video appears by itself.
"""
from __future__ import annotations

from typing import List, Optional

import gradio as gr

from . import prefs, queue_manager, quick_jobs
from . import ui_assist, ui_queue, video_gen
from . import cast as cast_mod
from .ui_theme import note, page_head, section

ASPECTS = list(video_gen.ASPECTS)


def gallery_paths(items) -> List[str]:
    """Sciezki z galerii Gradio (lista (sciezka, podpis) albo sciezek), w kolejnosci."""
    out = []
    for it in items or []:
        path = it[0] if isinstance(it, (list, tuple)) else it
        if isinstance(path, dict):
            path = path.get("path") or path.get("name")
        if path:
            out.append(str(path))
    return out


CAST_INFO = ("Pre-filled from the top bar’s Tools. Write a character’s name in the prompt to place them "
             "in a shot (names add characters here automatically); a shot without names "
             "continues with the previous shot’s characters.")
PHOTO_CAST_INFO = ("Pre-filled from the top bar’s Tools. Write a character’s name in the prompt to place "
                   "them in the photo (names add characters here automatically); a prompt without "
                   "names shows the first character.")


def cast_picker(label: str, info: str = CAST_INFO) -> gr.Dropdown:
    """'Characters in this video/photo': characters with reference photos, refreshed when the page opens."""
    return gr.Dropdown([], value=[], multiselect=True, max_choices=cast_mod.MAX_CAST,
                       label=label, info=info, interactive=True, allow_custom_value=True)


def cast_view(top_choice: Optional[str] = None):
    """Fresh choices and the pre-fill for a page's characters picker from the top bar's character picker
    (`top_choice`: its value; default: the saved one)."""
    return gr.update(choices=cast_mod.picker_choices(), value=cast_mod.picked(top_choice))


def cast_reply(found: List[dict], notes: List[str]) -> str:
    """The line under 'Queued as #N' that says who is in the video or photo."""
    if not found and not notes:
        return ""
    who = (f"Characters: {' + '.join(c['name'] for c in found)}. " if found else "")
    return "\n\n" + who + " ".join(notes)


def build_video_tab(tab: Optional[gr.Tab] = None, top_picker: Optional[gr.Dropdown] = None):
    """Buduje strone. Zwraca (funkcja_odswiezania, wejscia, wyjscia) dla auto-odswiezania."""
    page_head("Video", "Describe a scene to queue a video with audio. You can immediately "
                       "add more jobs.")
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["card"]):
            positive = gr.Textbox(
                label="Scene prompt", lines=5, max_lines=14,
                info="Describe the subject, action, location, lighting, camera and sounds in any language. Put "
                     "spoken dialogue in quotation marks, preserving your preferred language; pick a voice with "
                     "“@Hanna says …” (see the Voices page). Name your characters to put them in a shot. "
                     "Reference images as [1], [2]; start the next shot with “Shot 2:”.",
                placeholder="Ola hands Kuba a coffee in a sunny cafe and says “Here you go!”\n"
                            "Shot 2: Kuba smiles and answers “Thanks!”")
            wire_assistant = ui_assist.build_assistant("video", positive)
            cast_pick = cast_picker("Characters in this video")
            with gr.Row(equal_height=True):
                seconds = gr.Slider(1, video_gen.MAX_SECONDS, value=5, step=1, label="Duration (s)",
                                    scale=3)
                sound_on = gr.Checkbox(value=True, label="Audio", scale=1, min_width=120,
                                       elem_classes=["switch"])
            aspect = gr.Radio(ASPECTS, value=ASPECTS[0], label=f"Aspect ratio ({video_gen.RESOLUTION} · 30 FPS)")
            with gr.Accordion("Reference images and starting image (optional)", open=False):
                ref_gallery = gr.Gallery(
                    label="Reference images (up to 10): [1], [2], [3]… in upload order (for example, "
                          "a hoodie, a person or a building)", type="filepath", interactive=True, columns=5,
                    height=150, file_types=["image"], object_fit="contain")
                wire_assistant(ref_gallery)
                img_input = gr.Image(type="filepath", label="Starting image (the exact first "
                                                            "frame of the video)", height=170)
            generate = gr.Button("Generate video", variant="primary", size="lg",
                                 elem_classes=["cta"])
        with gr.Column(scale=4, elem_classes=["card", "result-card"]):
            section("Result", "Your latest completed video. Browse all results in Gallery.")
            out_video = gr.Video(show_label=False, height=380)
            status = gr.Markdown(elem_classes=["status-box"])

    def add(*values):
        return note(_add(*values))

    def _add(pos_v, refs_v, img_v, seconds_v, aspect_v, sound_v, cast_v=None):
        """Tylko dodaje zadanie do kolejki - przycisk od razu jest gotowy na kolejne."""
        try:
            queued = quick_jobs.queue_video(pos_v, gallery_paths(refs_v), img_v, seconds_v, aspect_v, sound_v, cast_v)
        except quick_jobs.Refused as refused:
            return refused.shown
        return ui_queue.added_message(queued.job) + cast_reply(queued.found, queued.notes)

    generate.click(add, inputs=[positive, ref_gallery, img_input, seconds, aspect, sound_on, cast_pick],
                   outputs=status, trigger_mode="multiple")  # kazde klikniecie = zadanie
    last = gr.State(getattr(queue_manager.latest_finished("wideo"), "id", None))
    gr.Timer(3.0).tick(ui_queue.result_poller("wideo"), inputs=last,
                       outputs=[out_video, status, last])
    if tab is not None:
        tab.select(lambda: cast_view(), outputs=cast_pick)
    if top_picker is not None:  # the top picker's new value itself (saving it runs in parallel)
        top_picker.input(lambda v: cast_view(v or ""), inputs=top_picker, outputs=cast_pick)
    return lambda: cast_view(), [], [cast_pick]


def run_job(p: dict, cb) -> tuple:
    """Wykonanie zadania z kolejki (watek roboczy) -> (ok, pliki, uwagi, blad). A job queued before the one
    video mode may still carry its mode ("speed"): it is made in the one mode like any other."""
    chs, ch_note = cast_mod.for_job(p)
    out = video_gen.generate(
        p["prompt"], p["seconds"], p["aspect"], image_path=p.get("image"),
        ref_images=p.get("refs") or [], sound_on=p.get("sound", True),
        style=prefs.STYLE["video"], progress_cb=cb, character=chs[0] if chs else None, cast=chs)
    notes = ([ch_note] if ch_note else []) + out.notes
    if out.ok:
        notes += [n for n in [prefs.autosave(out.files)] if n]
    return out.ok, out.files, notes, out.error


queue_manager.register("wideo", run_job)
