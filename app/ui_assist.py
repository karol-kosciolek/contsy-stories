# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Pomocnik promptu na stronach Video i Zdjecie oraz pod opisem sceny w Timeline (app/llm.py).

Pomysl po polsku -> nowy prompt po angielsku wpisany w pole opisu (poprzedni opis nie jest
doklejany - wczesniej stary prompt "przeciekal" do nowego). Puste pole pomyslu = popraw obecny
opis. Pomocnik wie, ile zdjec pomocniczych dolaczono w zakladce, i wskazuje je [1], [2].
Model dziala lokalnie.
"""
from __future__ import annotations

from typing import Callable, Tuple

import gradio as gr

from . import llm
from .ui_theme import note


PLACEHOLDER = ("For example, an adult in a green hoodie jumps between rooftops at sunset, then says “Let’s go!” to "
               "the camera. Pick a voice with “@Hanna says …”. With reference images, refer to “the person in image "
               "[1]”. Leave blank to improve the current prompt.")


LABEL = "AI prompt assistant — describe your idea in any language"


def assistant_panel(placeholder: str = PLACEHOLDER, label: str = LABEL) -> Tuple[gr.Textbox, gr.Button, gr.Markdown]:
    """The assistant's collapsed panel where it is built: the idea, Write prompt and the status line. The caller
    wires the button: build_assistant on the Video and Image pages, the Timeline for a scene's picture text."""
    with gr.Accordion(label, open=False):
        idea = gr.Textbox(show_label=False, lines=2, max_lines=6, placeholder=placeholder)
        with gr.Row(elem_classes=["toolbar"]):
            write = gr.Button("Write prompt", size="sm")
            status = gr.Markdown(elem_classes=["status-box", "grow"])
    return idea, write, status


def build_assistant(kind: str, prompt_box: gr.Textbox) -> Callable[..., None]:
    """kind: 'video' / 'image'; prompt_box - pole opisu, do ktorego trafia wynik. Zwraca
    funkcje, ktora podpina galerie zdjec pomocniczych (tworzona nizej na stronie)."""
    idea, write, status = assistant_panel()

    def wire(ref_gallery: gr.Gallery) -> None:
        from .ui_video import gallery_paths

        def run(idea_v, current_v, refs_v, progress=gr.Progress()):
            def cb(frac, desc):
                progress(min(max(frac, 0.0), 1.0), desc=desc)

            from . import cast, characters

            idea_v = (idea_v or "").strip()
            # the studio's characters by the @names Contsy recognises; these pages are always Realistic, which draws
            # no animal or creature character
            people = [person for person in cast.ready_characters() if not characters.is_creature(person)]
            named = [person["slug"] for person in cast.mentions(f"{idea_v} {current_v or ''}", people)]
            ok, text = llm.write_prompt(idea_v, kind, "" if idea_v else current_v, cb,
                                        n_refs=len(gallery_paths(refs_v)), context=llm.characters_context(people),
                                        names=named)
            if not ok:
                return gr.update(), note(f"❌ {text}" if not text.startswith("⛔") else text)
            return text, note("✅ The prompt has been added above. Review or edit it, then click "
                              "Generate.")

        write.click(run, inputs=[idea, prompt_box, ref_gallery], outputs=[prompt_box, status])

    return wire
