# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Styles page: the built-in styles and the owner's own (app/style_library.py).

A film picks its style in Timeline → Project → Style and a character in Characters → 1. Look. This page shows what
each built-in style looks like and how the studio makes it, and lets the owner make, rename, edit and delete styles of
their own: a name, the built-in style it starts from (or "Something else") and the look in their own words.
"""
from __future__ import annotations

import gradio as gr

from . import llm, style_library, styles
from .ui_theme import esc, note, page_head, section, table

# How the studio makes each starting point; a style of the owner's own is made like the one it starts from. A test
# checks that every starting point has a line.
_MADE_WITH = {
    styles.REALISTIC: "The studio's standard models, no style add-on. Lip sync with MuseTalk; when it fails, the "
                      "scene fails too.",
    styles.ANIMATION_3D: "A 3D style add-on (the Pixar-inspired 3D LoRA) in every picture with characters; Wan 2.2 "
                         "animates the shots. One speaker alone talks through InfiniteTalk, otherwise MuseTalk. An "
                         "animal or creature never gets MuseTalk: alone it talks through InfiniteTalk, otherwise its "
                         "lines play without mouth movement.",
    styles.CARTOON: "Words only, no style add-on. A shot with nothing to start from begins with a drawn frame; the "
                    "drawn-animation model Index-AniSora animates the shots (Full set, otherwise Wan 2.2). No "
                    "MuseTalk: one speaker alone talks through InfiniteTalk.",
    styles.ANIME: "Anime style add-ons (Anime LoRAs) for the pictures; drawn start frames; Index-AniSora animates "
                  "the shots (Full set, otherwise Wan 2.2). No MuseTalk: one speaker alone talks through InfiniteTalk.",
    styles.OTHER: "Words only, no style add-on; drawn start frames; Wan 2.2 animates the shots. No MuseTalk: one "
                  "speaker alone talks through InfiniteTalk.",
}
LOOK_PLACEHOLDER = ("For example: watercolor storybook, soft pastel colors, visible paper texture, gentle warm "
                    "light")
DELETE_CONFIRM = ("(key) => key ? (confirm('Remove this style from your styles? Films and characters that use it "
                  "keep their look.') ? 'yes' : 'no') : 'yes'")


def builtin_table() -> str:
    """The built-in styles, and "Something else" as a starting point of the owner's own styles."""
    rows = [[f"<b>{esc(style.label)}</b>", esc(style.about), esc(_MADE_WITH[style.id])]
            for style in styles.STYLES.values()]
    other = styles.BASES[styles.OTHER]
    rows.append([f"<b>{esc(other.label)}</b>", esc(f"{other.about} Only as the start of a style of your own."),
                 esc(_MADE_WITH[other.id])])
    return table(["Style", "Looks like", "Made with"], rows)


def _yours() -> list[dict]:
    return sorted(style_library.all_custom(), key=lambda entry: entry["name"].casefold())


def custom_table(entries: list[dict]) -> str:
    rows = [[f"<b>{esc(entry['name'])}</b>", esc(styles.base_label(entry["id"])), esc(entry["look"])]
            for entry in entries]
    return table(["Name", "Starts from", "Look"], rows, empty="<b>No styles of your own yet</b>Make one below.")


def views(key: str = "") -> tuple:
    """The table, the picker and the form (name, start, look) for the style `key` ("" = a new style)."""
    entries = _yours()
    entry = next((item for item in entries if item["key"] == key), None) if key else None
    return (custom_table(entries),
            gr.update(choices=[(style_library.NEW_STYLE, "")] + [(item["name"], item["key"]) for item in entries],
                      value=entry["key"] if entry else ""),
            entry["name"] if entry else "",
            entry["base"] if entry else styles.OTHER,
            entry["look"] if entry else "")


def _refused(message: str) -> str:
    """A refusal for the status line (a message with its own mark keeps it)."""
    return note(message if message.lstrip().startswith(("⛔", "❌", "⚠")) else f"❌ {message}")


def pick_style(key, loaded):
    """A style picked fills the form, and `loaded` remembers which one. Gradio sends the picker's input event on every
    blur too: the style already in the form keeps what the owner typed."""
    key = key if style_library.get(key or "") else ""
    if key == (loaded or ""):
        return (gr.skip(),) * 5
    return (*views(key)[1:], key)


def save_style(key, name, base, look):
    """Save the form as a new style (key "") or as the picked one: a rename keeps its id, a new look makes a new
    version."""
    entry, message = style_library.save(name or "", base or "", look or "", key or "")
    if entry is None:
        return (_refused(message), *([gr.skip()] * 6))
    tip = "" if llm.english(entry["look"]) else " Tip: English words work best; use Translate to English."
    return (note(f"✅ {message}{tip}"), *views(entry["key"]), entry["key"])


def translate_look(look, progress=gr.Progress()):
    """The look in English style words (the prompt assistant), for the owner to check before saving."""
    def report(fraction, desc):
        progress(min(max(fraction, 0.0), 1.0), desc=desc)

    ok, words = llm.style_words(look or "", report)
    if not ok:
        return gr.skip(), _refused(words)
    return words, note("✅ In English now. Check the words, then save the style.")


def delete_style(key, confirmed):
    if not key or not style_library.get(key):
        return (_refused("Choose one of your styles to delete."), *([gr.skip()] * 6))
    if confirmed != "yes":
        return ("Nothing deleted.", *([gr.skip()] * 6))
    ok, message = style_library.delete(key)
    return (note(("✅ " if ok else "❌ ") + message), *views(""), "")


def build_styles_tab(tab: gr.Tab) -> None:
    page_head("Styles", "How the studio knows the look of a film: every film has one style (Timeline → Project → "
                        "Style), and every character the style its photos are drawn in (Characters → 1. Look). The "
                        "style decides the words every picture model gets, the style add-ons it loads, the model that "
                        "animates the shots, and how lip sync and the face checks work.")
    with gr.Column(elem_classes=["card"]):
        section("Built-in styles")
        gr.HTML(builtin_table(), padding=False)
    listing_now, picker_now, *_ = views("")
    with gr.Column(elem_classes=["card"]):
        section("Your styles", "Describe a look in your own words and choose the built-in style it starts from. "
                               "Then pick it for a film or a character like a built-in style.")
        listing = gr.HTML(listing_now, padding=False, elem_id="style-list")
        with gr.Row():
            pick = gr.Dropdown(picker_now["choices"], value="", label="Style", interactive=True,
                               allow_custom_value=True, elem_id="style-pick", scale=1)
            name = gr.Textbox(label="Name", max_lines=1, max_length=style_library.MAX_NAME,
                              placeholder="For example, Storybook", elem_id="style-name", scale=1)
        base = gr.Radio(styles.base_choices(), value=styles.OTHER, label="Start from",
                        info="The studio draws, animates and lip-syncs your style like this one. Realistic is for "
                             "filmed looks with real people; Something else lets your words alone decide.",
                        elem_id="style-base")
        look = gr.Textbox(label="Describe the look", lines=3, max_lines=6, max_length=styles.MAX_LOOK,
                          placeholder=LOOK_PLACEHOLDER,
                          info="Colors, lines, light and texture, in English. Describe the pictures, not the people.",
                          elem_id="style-look")
        with gr.Row(elem_classes=["toolbar"]):
            translate = gr.Button("Translate to English", size="sm")
            save = gr.Button("Save style", size="sm", variant="primary")
            delete = gr.Button("Delete style", size="sm", variant="stop")
            status = gr.Markdown(elem_classes=["status-box", "grow"])
    form = [listing, pick, name, base, look]
    loaded = gr.State("")  # the style whose name, start and look the form was filled with ("" = a new style)
    save.click(save_style, inputs=[pick, name, base, look], outputs=[status, *form, loaded], api_visibility="private")
    translate.click(translate_look, inputs=look, outputs=[look, status], api_visibility="private")
    confirmed = gr.Textbox(visible=False)
    delete.click(None, inputs=pick, outputs=confirmed, js=DELETE_CONFIRM, api_visibility="private").then(
        delete_style, inputs=[pick, confirmed], outputs=[status, *form, loaded], api_visibility="private")
    pick.input(pick_style, inputs=[pick, loaded], outputs=[pick, name, base, look, loaded], api_visibility="private")
    # styles saved in another browser tab: the table and the picker are read again on every visit; the form stays
    tab.select(lambda key: views(key if style_library.get(key or "") else "")[:2], inputs=pick,
               outputs=[listing, pick], api_visibility="private", show_progress="hidden")
