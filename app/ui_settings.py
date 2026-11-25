# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Strona 'Settings' - kontrola jakosci wynikow modelek, autozapis kopii wynikow, zasady.

Wartosci czytane przy kazdym otwarciu strony (nie z chwili startu serwera) - inaczej
odswiezona karta pokazywala stare; zapis od razu po zmianie (app/prefs.py).
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import gradio as gr

from . import __copyright__, __title__, __version__, config, model_manager, prefs, restart, vram
from .aux_models import CATALOG as AUX_CATALOG
from .ui_theme import esc, note, page_head, section, table

SAFETY_NOTE = ("Create content with fictional characters or consenting adults only. Do not impersonate "
               "real people or use their face, body or voice without permission. Use a person's photos for a character only "
               "when they show an AI character or an adult who has consented. Clone a voice only from your own recording, a synthetic voice or an adult who agreed. "
               "Outputs include AI-generated metadata where supported.")


def _sources(model: dict) -> dict[str, str]:
    """Every place a model's files come from, link -> text: its Hugging Face repos and, for a file
    downloaded from a plain URL, that URL (shown as host and file name). A file that an install step makes
    on this PC is listed by the official files it is made from."""
    sources = {}
    if model.get("repo_id"):
        sources[f"https://huggingface.co/{model['repo_id']}"] = model["repo_id"]
    for spec in model.get("files", []):
        for source in model_manager.sources(spec, model):
            if source.get("url"):
                sources[source["url"]] = f"{urlsplit(source['url']).netloc}: {Path(source['filename']).name}"
            elif source.get("repo_id"):
                sources[f"https://huggingface.co/{source['repo_id']}"] = source["repo_id"]
    return sources


def model_licenses_html() -> str:
    """Render the catalog's license notes and every download source for review."""
    rows = []
    for model in config.CATALOG.get("models", []):
        links = "<br>".join(_link(url, text) for url, text in _sources(model).items())
        rows.append([esc(model.get("name", model.get("id", ""))),
                     esc(model.get("license", "See the model card")), links])
    spec = AUX_CATALOG["landmarker"]
    rows.append(["MediaPipe Face Landmarker", esc(spec["license"]),
                 f"<a href='{esc(spec['source_url'])}' target='_blank' "
                 "rel='noopener noreferrer'>Provider and model terms</a>"])
    return table(["Model", "License notes", "Model cards and weight providers"], rows)


def _link(url: str, text: str) -> str:
    return f"<a href='{esc(url)}' target='_blank' rel='noopener noreferrer'>{esc(text)}</a>"


def about_html() -> str:
    """GPL 'Appropriate Legal Notices': copyright, no warranty, the license and the source code."""
    url = config.SETTINGS["app"].get("source_url", "")
    source = _link(url, url) if url else f"the folder Contsy runs from, <code>{esc(config.ROOT)}</code>"
    return (f"<p><b>{esc(__title__)} {esc(__version__)}</b> · {esc(__copyright__)}</p>"
            "<p>Free software under the GNU General Public License, version 3 only "
            "(GPL-3.0-only). You may use, change and share it under that license. It comes with "
            "no warranty. Read the license in <code>LICENSE</code> in the installation folder or at "
            f"{_link('https://www.gnu.org/licenses/gpl-3.0.html', 'gnu.org')}.</p>"
            f"<p>Source code: {source}.</p>"
            "<p>The AI models and third-party parts keep their own licenses: see "
            "<code>NOTICE</code>, <code>THIRD_PARTY_NOTICES.md</code>, <code>MODEL_LICENSES.md</code> "
            "and <code>docs/LICENSING.md</code> in the installation folder. "
            "EzAudio-XL (MIT) with FLAN-T5-XL (Apache-2.0) makes background sound and effects, "
            "MuseTalk 1.5 (MIT) moves the lips.</p>"
            "<p>Sharing a changed version? Set <code>app.source_url</code> in "
            "<code>config/settings.local.json</code> so this card links to its source code.</p>")


TIER_CHOICES = [("Automatic", ""), ("Full quality (16 GB+ VRAM)", "full"), ("Lite (8–12 GB VRAM)", "lite")]


def tier_hint() -> str:
    """Which model set this PC uses and why."""
    gpu = vram.get_gpu_info()
    auto = model_manager.default_tier()
    card = f"{gpu.name}, {gpu.total_gb:.0f} GB" if gpu.total_gb else "GPU not detected"
    return ("Lite uses smaller versions of the same models: much less GPU memory and RAM, slightly "
            "less detail. Automatic picks Lite below 15 GB of VRAM. Missing models download on first "
            "use.\n\n"
            f"This PC: {esc(card)} → automatic choice **{'Lite' if auto == 'lite' else 'Full quality'}**, "
            f"in use now: **{'Lite' if model_manager.tier() == 'lite' else 'Full quality'}**.")


def build_settings_tab():
    """Zwraca (funkcja, wyjscia) do wczytania ustawien przy otwarciu strony."""
    page_head("Settings", "Changes take effect immediately and persist after restarting the app.")
    with gr.Column(elem_classes=["card"]):
        section("Model set", "Applies to new jobs.")
        tier = gr.Radio(TIER_CHOICES, value="", show_label=False, elem_id="tier")
        tier_note = gr.Markdown(elem_classes=["hint"])
    with gr.Column(elem_classes=["card"]):
        section("Face check", "Compare the faces in videos with the character's reference photos: a face that "
                              "drifts between the parts of a video is reported in the result's notes.")
        qc_on = gr.Checkbox(label="Check faces against reference photos", value=True, container=False,
                            elem_id="qc", elem_classes=["switch"])
    with gr.Column(elem_classes=["card"]):
        section("Autosave", "Copy each completed result to a folder of your choice, such as "
                             "a backup drive.")
        autosave = gr.Checkbox(label="Save copies of results", container=False,
                               elem_id="autosave", elem_classes=["switch"])
        with gr.Row(elem_classes=["toolbar"]):
            folder = gr.Textbox(show_label=False, elem_id="folder", max_lines=1,
                                placeholder="Autosave folder", elem_classes=["grow"])
            pick = gr.Button("Choose folder", size="sm")
    with gr.Column(elem_classes=["card"]):
        section("Restart", "Closes Contsy Studio and starts the app and engine again with start.bat. "
                           "A new browser tab opens in about 20 seconds. Pause the queue before restarting.")
        with gr.Row(elem_classes=["toolbar"]):
            restart_btn = gr.Button("Restart Contsy Studio", size="sm", variant="stop")
            restart_info = gr.Markdown(elem_classes=["status-box", "grow"])
    with gr.Column(elem_classes=["card"]):
        section("Safety and consent")
        gr.Markdown(SAFETY_NOTE, elem_classes=["hint"])
    with gr.Accordion("Model downloads and licenses", open=False,
                      elem_classes=["accordion-card"]):
        gr.Markdown("Missing model weights download automatically when you first use a feature. "
                    "Downloads can be large and require an internet connection; subsequent generation "
                    "uses the local files. Review each model card and license below before use, "
                    "including any commercial-use conditions. Model weights are distributed by their "
                    "providers separately from this app.\n\n"
                    "For a gated repository, accept its access terms on Hugging Face and set "
                    "the **HF_TOKEN** environment variable before starting the app. The app reads the "
                    "token from the environment and does not store it in Settings. The only secret "
                    "Contsy keeps is the Claude API key of the Agent page, encrypted for your Windows "
                    "account.",
                    elem_classes=["hint"])
        gr.HTML(model_licenses_html())
    with gr.Column(elem_classes=["card"]):
        section("About")
        gr.HTML(about_html(), elem_id="about", elem_classes=["hint"])

    tier.input(lambda v: (prefs.save(tier=v if v in model_manager.TIERS else ""), tier_hint())[1],
               inputs=tier, outputs=tier_note)
    qc_on.input(lambda v: prefs.save(qc=bool(v)) and None, inputs=qc_on)
    autosave.input(lambda v: prefs.save(autosave=bool(v)) and None, inputs=autosave)
    folder.input(lambda v: prefs.save(folder=v or "") and None, inputs=folder)

    def _pick(current):
        chosen = prefs.choose_folder(current)
        prefs.save(folder=chosen or "")
        return chosen

    pick.click(_pick, inputs=folder, outputs=folder)

    def do_restart(typed_v):
        ok, msg = restart.request(typed_v)
        if not ok and msg.startswith("Nothing restarted"):
            return msg
        return note(("✅ " if ok else "❌ ") + msg)

    # two steps on purpose: the button, then the word RESTART typed in the browser prompt
    typed = gr.Textbox(visible=False)
    restart_btn.click(None, outputs=typed,
                      js="() => prompt('Restart Contsy Studio now? Unfinished jobs return to the "
                         "queue.\\nTo confirm, type: RESTART') || ''").then(
        do_restart, inputs=typed, outputs=restart_info)

    def load():
        p = prefs.load()
        return p["tier"], tier_hint(), p["qc"], p["autosave"], p["folder"]

    return load, [tier, tier_note, qc_on, autosave, folder]
