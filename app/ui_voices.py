# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The 'Voices' page: the library of standalone voices (a narrator, unnamed speakers: built-in and
imported voices) in Women and Men tabs, grouped by pitch and filterable by name, age and source;
listening to samples; the default voice per gender; importing a new voice from a recording (voice
cloning) and removing imported voices. Data and rules live in app/voices.py. A character's own voice
is not listed here: it is made on the Characters page and stored with the character.

Values are read when the page opens (demo.load and tab.select), not when the server starts,
so the page always shows the current library.
"""
from __future__ import annotations

from typing import List, Optional

import gradio as gr

from . import prefs, voices
from .ui_theme import cache_media, esc, note, page_head, section, table, waveform

VOICE_CONSENT = ("I confirm that this is my own voice, a synthetic voice, or the voice of an adult "
                 "who agreed to its cloning.")
VOICE_TYPES = [("Woman", "female"), ("Man", "male")]
TABS = (("female", "Women"), ("male", "Men"))
PITCH_CHOICES = [("All pitches", ""), ("Low", "low"), ("Medium", "medium"), ("High", "high")]
AGE_CHOICES = [("All ages", "")] + [(a, a) for a in voices.AGES] + [("Age not set", "unknown")]
SOURCE_CHOICES = [("Built-in and imported", ""), ("Built-in", "builtin"), ("Imported", "imported")]
AGE_FORM = [("Age (optional)", "")] + [(a, a) for a in voices.AGES]
HOW_TO = ("Write the voice name with **@** right before a spoken line in a video prompt, for "
          "example **@Hanna says “Welcome back!”** or **@Marek answers “Thanks!”**. A "
          "character’s name speaks with that character’s own voice (made with the character). "
          "Other lines use the default voices below. Works in Timeline and Video.")
_GROUPS = (("low", "Low voices"), ("medium", "Medium voices"), ("high", "High voices"),
           ("", "Pitch not measured"))


def matches(v: voices.Voice, query: str = "", pitch: str = "", age: str = "", source: str = "") -> bool:
    """Filter of the library tables: text in the name, age or description; pitch; age; source."""
    q = (query or "").strip().casefold()
    if q and q not in f"{v.name} {v.age} {v.description}".casefold():
        return False
    if pitch and voices.pitch(v) != pitch:
        return False
    if age == "unknown" and v.age:
        return False
    if age and age != "unknown" and v.age != age:
        return False
    if source == "builtin" and not v.builtin:
        return False
    if source == "imported" and v.builtin:
        return False
    return True


def render(gender: str, query: str = "", pitch: str = "", age: str = "", source: str = "",
           pool: Optional[List[voices.Voice]] = None) -> str:
    """Voices of one gender as tables grouped by pitch, sorted by age and name, filtered."""
    all_of_gender = [v for v in (voices.all_voices() if pool is None else pool) if v.gender == gender]
    shown = [v for v in all_of_gender if matches(v, query, pitch, age, source)]
    if not all_of_gender:
        return "<div class='empty'><b>No voices yet</b>Add one on the right.</div>"
    if not shown:
        return ("<div class='empty'><b>No voice matches the filters</b>Clear the search or choose "
                "other filters.</div>")

    def order(v: voices.Voice):
        return (voices.AGES.index(v.age) if v.age in voices.AGES else len(voices.AGES), v.name.casefold())

    groups = {key: [] for key, _ in _GROUPS}
    for v in sorted(shown, key=order):
        groups[voices.pitch(v)].append([
            f"<b>@{esc(v.name)}</b>", esc(v.age or "—"),
            esc(f"{v.f0_hz:.0f} Hz") if v.f0_hz else "—", esc(v.description),
            f"<span class='num'>{v.seconds:.1f} s</span>" if v.seconds else "",
            "Built-in" if v.builtin else "Imported"])
    parts = [f"<p class='muted'>{len(shown)} of {len(all_of_gender)} voices</p>"]
    for key, title in _GROUPS:
        if groups[key]:
            parts.append(f"<h3 class='section-title'>{title} ({len(groups[key])})</h3>")
            parts.append(table(["Voice", "Age", "Pitch", "Description", "Sample", "Source"], groups[key]))
    return "".join(parts)


def voice_views():
    """Fresh values for the page: both tables (unfiltered), default dropdowns, sample picker,
    removal list."""
    pool = voices.all_voices()
    chosen = voices.defaults(pool)
    women = [(v.label, v.slug) for v in pool if v.gender == "female"]
    men = [(v.label, v.slug) for v in pool if v.gender == "male"]
    return (render("female", pool=pool), render("male", pool=pool),
            gr.update(choices=women, value=chosen["female"] or None),
            gr.update(choices=men, value=chosen["male"] or None),
            gr.update(choices=[(v.label, v.slug) for v in pool], value=None),
            gr.update(choices=[(v.label, v.slug) for v in pool if not v.builtin], value=None))


def build_voices_tab(tab: gr.Tab):
    """Builds the page. Returns (load function, outputs) for demo.load, like Settings."""
    page_head("Voices", "Standalone voices for a narrator or speakers who are not characters: 20 built-in "
                        "voices of different ages, plus voices cloned from your own recordings. A character’s "
                        "own voice is made with the character on the Characters page and stays with it.")
    tables = {}
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["card"]):
            section("Voice library", "Each voice is a short sample that the speech model clones for "
                                     "every spoken line, so a voice sounds the same in every video. "
                                     "Grouped by pitch; search or filter by age and source.")
            gr.Markdown(HOW_TO, elem_classes=["hint"])
            with gr.Tabs():
                for gender, title in TABS:
                    with gr.Tab(title, id=f"voices-{gender}"):
                        with gr.Row():
                            query = gr.Textbox(show_label=False, max_lines=1, scale=3,
                                               placeholder="Search by name or description")
                            pitch = gr.Dropdown(PITCH_CHOICES, value="", show_label=False, scale=2)
                            age = gr.Dropdown(AGE_CHOICES, value="", show_label=False, scale=2)
                            source = gr.Dropdown(SOURCE_CHOICES, value="", show_label=False, scale=2)
                        listing = gr.HTML()
                        tables[gender] = listing
                        for control in (query, pitch, age, source):
                            control.change(lambda q, p, a, s, g=gender: render(g, q, p, a, s),
                                           inputs=[query, pitch, age, source], outputs=listing)
            with gr.Row(equal_height=True):
                listen = gr.Dropdown([], label="Listen to a sample", interactive=True,
                                     allow_custom_value=True, scale=2)
                player = gr.Audio(label="Sample", type="filepath", interactive=False, scale=3,
                                  waveform_options=waveform())
            section("Default voices", "Used for spoken lines that name no voice and are not said by a "
                                      "character with its own voice.")
            with gr.Row():
                default_f = gr.Dropdown([], label="Woman’s lines", interactive=True,
                                        allow_custom_value=True)
                default_m = gr.Dropdown([], label="Man’s lines", interactive=True,
                                        allow_custom_value=True)
        with gr.Column(scale=4, elem_classes=["card"]):
            section("Add a standalone voice (voice cloning)",
                    "Upload or record 3–12 seconds of one person speaking clearly, without music or other "
                    "voices. The sample is converted and kept on this computer; its pitch is measured for the "
                    "library. A character’s voice is cloned in the character’s own Voice step instead.")
            with gr.Row():
                v_name = gr.Textbox(label="Name (one word, used as @Name)", placeholder="For example, Kasia",
                                    max_lines=1, scale=3)
                v_type = gr.Radio(VOICE_TYPES, value="female", label="Type", scale=2)
            with gr.Row():
                v_age = gr.Dropdown(AGE_FORM, value="", label="Age", scale=1)
                v_desc = gr.Textbox(label="Description (optional)", placeholder="For example, warm and calm",
                                    max_lines=1, scale=3)
            v_sample = gr.Audio(label="Voice sample", type="filepath", sources=["upload", "microphone"],
                                waveform_options=waveform())
            v_consent = gr.Checkbox(label=VOICE_CONSENT)
            v_add = gr.Button("Add voice", variant="primary", size="lg", elem_classes=["cta"])
            v_status = gr.Markdown(elem_classes=["status-box"])
            section("Remove an imported voice", "Built-in voices stay. Removed voices go to the Recycle Bin.")
            with gr.Row(elem_classes=["toolbar"]):
                remove_pick = gr.Dropdown([], show_label=False, interactive=True,
                                          allow_custom_value=True, elem_classes=["grow"])
                remove = gr.Button("Remove", size="sm", variant="stop")
            remove_info = gr.Markdown(elem_classes=["status-box"])

    outputs = [tables["female"], tables["male"], default_f, default_m, listen, remove_pick]
    default_f.input(lambda v: prefs.save(default_female=v or "") and None, inputs=default_f)
    default_m.input(lambda v: prefs.save(default_male=v or "") and None, inputs=default_m)
    listen.change(lambda slug: cache_media(voices.reference_path(slug)) if slug else None,
                  inputs=listen, outputs=player)

    form = [v_name, v_sample, v_desc, v_age]

    def add_voice(name_v, type_v, age_v, desc_v, sample_v, consent_v):
        voice, msg = voices.create(name_v, type_v, sample_v, bool(consent_v), desc_v or "", age_v or "")
        if not voice:  # the form keeps its values so the user can correct them
            return (note(f"❌ {msg}"), *([gr.skip()] * (len(outputs) + len(form))))
        return (note(f"✅ {msg}"), *voice_views(), "", None, "", "")

    v_add.click(add_voice, inputs=[v_name, v_type, v_age, v_desc, v_sample, v_consent],
                outputs=[v_status, *outputs, *form])

    def remove_voice(confirmed_v, slug):
        if not slug:
            return note("❌ Select an imported voice to remove."), *([gr.skip()] * len(outputs))
        if confirmed_v != "yes":
            return "Nothing removed.", *([gr.skip()] * len(outputs))
        ok, msg = voices.delete(slug)
        return note(("✅ " if ok else "❌ ") + msg), *voice_views()

    # browser confirmation first: an accidental click would otherwise remove the voice
    confirmed = gr.Textbox(visible=False)
    remove.click(None, inputs=remove_pick, outputs=confirmed,
                 js="(slug) => slug ? (confirm('Remove this voice? It is moved to the Recycle "
                    "Bin.') ? 'yes' : 'no') : 'yes'").then(
        remove_voice, inputs=[confirmed, remove_pick], outputs=[remove_info, *outputs])

    tab.select(voice_views, outputs=outputs)
    return voice_views, outputs
