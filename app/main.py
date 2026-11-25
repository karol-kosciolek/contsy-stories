# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Contsy's Timeline workspace and local Gradio launcher."""
from __future__ import annotations

import ipaddress
import os
from typing import Callable
from urllib.parse import urlsplit

# Gradio's usage telemetry is switched off before Gradio is imported, so nothing about the studio's use leaves the
# computer (setdefault: an explicit environment setting still wins)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse

from . import characters, config, engine_runtime, llm, queue_manager, ui_characters
from . import timeline, ui_queue, ui_settings, ui_theme, ui_timeline
from . import cast, items, music_library, series, ui_items, ui_series, ui_voices, voices
# The single-clip and single-picture pages are not part of Contsy Stories; their modules still register the "wideo" and
# "obraz" GPU jobs that the agent page's tools and older queued jobs use (docs/CONTSY_STORIES_PRODUCT.md).
from . import ui_images, ui_video  # noqa: F401
from . import agent_runner, agent_store, ui_agent
from .vram import get_gpu_info

START_PAGE = "stories"
# Contsy Stories: the creator's pages. The Editor is the Timeline; Locations & Props are the library's places and items.
WORKSPACE = (("Dashboard", "stories"), ("Series", "series"), ("Editor", "timeline"), ("Characters", "characters"),
             ("Locations & Props", "items"))
TOOLS = (("Agent", "agent"), ("Voices", "voices"), ("GPU Queue", "queue"), ("Settings", "settings"))
# The largest single upload any page accepts (a voice recording, app/voices.MAX_BYTES): Gradio refuses bigger ones
# before they reach the disk, so an oversized file cannot fill it
MAX_UPLOAD = "200mb"
TOOLS_PANEL = "topbar-tools"  # the top bar's Tools panel: its button (ui_theme.topbar_start) opens it
# flatten_selectors: Gradio's own prefixing breaks selector lists inside :is() (see ui_theme)
STYLE = {**ui_theme.STYLE, "css": ui_theme.flatten_selectors(ui_theme.CSS + ui_timeline.CSS)}

def _loopback_host(host: str) -> bool:
    try:
        return host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class LocalRequestGuard:
    """Reject foreign hosts and cross-origin writes before a UI action can execute."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        host = headers.get("host", "")
        try:
            allowed = _loopback_host(urlsplit("//" + host).hostname or "")
            if scope["method"] not in ("GET", "HEAD", "OPTIONS"):
                origin = headers.get("origin")
                if origin:
                    parsed = urlsplit(origin)
                    allowed = allowed and parsed.scheme == scope.get("scheme", "http") \
                        and parsed.netloc.lower() == host.lower()
                allowed = allowed and headers.get("sec-fetch-site") != "cross-site"
        except ValueError:  # a host or origin that cannot be parsed
            allowed = False
        if not allowed:
            return await PlainTextResponse("Local requests only.", status_code=403)(scope, receive, send)
        await self.app(scope, receive, send)


def _private_paths() -> list:
    """Folders the file route must never serve: the library, model, configuration and state folders."""
    return [config.CONFIG_DIR, config.MODELS_DIR, config.HISTORY_DIR, characters.DIR, characters.DRAFTS_DIR,
            items.DIR, items.DRAFTS_DIR, voices.USER_DIR, timeline.DIR, agent_store.RUNS_DIR, series.DIR,
            music_library.folder(), config.ROOT / ".git", config.STATE_CONFIG_DIR,
            *(config.STATE_ROOT / name for name in (".git", "models", "queue_files"))]


def local_launch_options(app_cfg: dict) -> dict:
    """Keep the local studio private: a loopback host, no sharing, and files served only from the
    outputs folder (the library, model, configuration and state folders are blocked)."""
    host = str(app_cfg.get("host", "127.0.0.1")).strip()
    if not _loopback_host(host):
        raise ValueError("The studio host must be a loopback address (127.0.0.1, ::1 or localhost).")
    icon = ui_theme.favicon()
    return {"server_name": host, "share": False, "footer_links": [], "favicon_path": icon,
            "max_file_size": MAX_UPLOAD, "head": ui_theme.favicon_link() if icon else "",
            "app_kwargs": {"middleware": [Middleware(LocalRequestGuard)]},
            "allowed_paths": [str(config.OUTPUTS_DIR.resolve())],
            "blocked_paths": list(dict.fromkeys(str(p.resolve()) for p in _private_paths()))}


def status_html() -> str:
    """The engine chip at the right of the top bar: the engine, the GPU and the queue. A mark whose shape
    tells the state (ready = filled dot, paused = hollow ring, unavailable = square), one short line, and
    every detail (the GPU's name) in its tooltip and for screen readers."""
    up = engine_runtime.Engine.available()
    if queue_manager.paused():
        dot, state = "pause", "Paused — GPU released"
    elif not up:
        dot, state = "off", "Engine unavailable"
    else:
        dot, state = "", "Engine ready"
    short, details = [], [state]
    gpu = get_gpu_info()
    if gpu.available:
        name = gpu.name.replace("NVIDIA ", "").replace("GeForce ", "")
        short.append(f"{gpu.free_gb:.1f} / {gpu.total_gb:.0f} GB free")
        details.append(f"{name} · {gpu.free_gb:.1f} of {gpu.total_gb:.0f} GB free")
    else:
        short.append("No GPU")
        details.append("No NVIDIA GPU detected")
    snap = queue_manager.snapshot()
    busy = sum(1 for j in snap if j.status in (queue_manager.PENDING, queue_manager.RUNNING))
    queue = (f"Queue {busy} · about {queue_manager.fmt_time(queue_manager.remaining_seconds())}"
             if busy else "Queue empty")
    short.append(queue)
    details.append(queue)
    if agent_runner.busy():  # only in memory: nothing is read from the disk for the chip
        short.append("Agent working")
        details.append("The Agent page is working on a job")
    line = " · ".join([f"<b>{ui_theme.esc(state)}</b>", *(ui_theme.esc(part) for part in short)])
    tooltip = ui_theme.esc("\n".join(details))
    spoken = ui_theme.esc(". ".join(part.rstrip(". ") for part in details))
    return (f"<div class='engine-chip' data-state='{dot or 'ready'}' title='{tooltip}'>"
            f"<span class='dot {dot}' aria-hidden='true'></span>"
            f"<span class='engine-text' aria-hidden='true'>{line}</span>"
            f"<span class='engine-sr'>{spoken}</span></div>")


def pause_label() -> str:
    return "Resume generation" if queue_manager.paused() else "Pause and release GPU"


# ----------------------------- Budowa UI -----------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Contsy Stories", analytics_enabled=False) as demo:
        # ----- top bar on every page: the brand, the workspace menu and Tools on the left (the Tools panel
        # holds the other pages and the character choice; it is a column, so Gradio renders it at once), the
        # engine status and the GPU pause on the right. The menus switch pages in the browser (ui_theme.JS).
        with gr.Row(elem_id="topbar"):
            gr.HTML(ui_theme.topbar_start(WORKSPACE, START_PAGE, TOOLS_PANEL), padding=False,
                    apply_default_css=False, elem_id="workspace-nav")
            with gr.Column(min_width=0, elem_id=TOOLS_PANEL):
                gr.HTML(ui_theme.workspace_nav(TOOLS, "Tools"), padding=False, apply_default_css=False,
                        elem_id="tools-nav")
                # the older pages' main character choice: Contsy Stories works per series instead, so it is hidden
                character = gr.Dropdown(cast.top_picker_choices(), value="", label="Character or universe",
                                        elem_id="character", interactive=True, allow_custom_value=True,
                                        visible=False)
            status = gr.HTML(status_html(), padding=False, apply_default_css=False, elem_id="engine-status")
            # only on a click (e.g. before a game): stops the running job - it returns to the queue - and
            # releases GPU memory; a second click resumes
            gpu_pause = gr.Button(pause_label(), size="sm", elem_id="gpu-pause")
        gr.Timer(10.0).tick(lambda: (status_html(), pause_label()), outputs=[status, gpu_pause])

        def _toggle_gpu():
            if queue_manager.paused():
                queue_manager.set_paused(False)
                gr.Info("Generation resumed.")
            else:
                stopped = queue_manager.pause_now()
                gr.Info("Paused. Generation has stopped and GPU memory is being released." + (
                    f" Job #{stopped} will return to the queue and restart when generation resumes."
                    if stopped else ""))
            return pause_label(), status_html()
        gpu_pause.click(_toggle_gpu, outputs=[gpu_pause, status])

        with gr.Tabs(selected=START_PAGE, elem_id="pages"):
            with gr.Tab("Dashboard", id="stories", elem_id="stories-page") as stories_tab:
                stories_fn, stories_outs = ui_series.build_home(stories_tab)
            # the Editor before the Series workspace: the workspace opens an episode's project in it
            with gr.Tab("Editor", id="timeline", elem_id="timeline-page") as timeline_tab:
                timeline_cast = ui_timeline.build_timeline_tab(timeline_tab)
            with gr.Tab("Series", id="series", elem_id="series-page") as series_tab:
                series_fn, series_outs = ui_series.build_series_tab(series_tab)
            with gr.Tab("Characters", id="characters", elem_id="characters-page") as characters_tab:
                leave_characters = ui_characters.build_characters_tab(characters_tab, character, timeline_cast)
            with gr.Tab("Locations & Props", id="items", elem_id="items-page") as items_tab:
                ui_items.build_items_tab(items_tab)
            with gr.Tab("Agent", id="agent", elem_id="agent-page") as agent_tab:
                agent_wire = ui_agent.build_agent_tab(agent_tab)
            with gr.Tab("Voices", id="voices") as voices_tab:
                voices_fn, voices_outs = ui_voices.build_voices_tab(voices_tab)
            with gr.Tab("GPU Queue", id="queue", elem_id="queue-page") as queue_tab:
                ui_queue.build_queue_tab(queue_tab)
            with gr.Tab("Settings", id="settings", elem_id="settings-page") as settings_tab:
                settings_fn, settings_outs = ui_settings.build_settings_tab()

        # Leaving the Characters page discards the character being made there (the page script asks first
        # when it holds unsaved work or photos or a voice still being made): every other page's select event
        # runs the page's `leave`.
        leave_fn, leave_inputs, leave_outputs = leave_characters
        for page in (stories_tab, series_tab, agent_tab, timeline_tab, items_tab, voices_tab, queue_tab, settings_tab):
            page.select(leave_fn, inputs=leave_inputs, outputs=leave_outputs, api_visibility="private",
                        show_progress="hidden", concurrency_id=ui_characters.SERIAL, concurrency_limit=1)

        # Settings czytane przy kazdym otwarciu strony (nie z chwili startu serwera) -
        # inaczej odswiezona karta pokazywala stare wartosci
        def _characters():
            ch = cast.top_picker_choices()
            sel = cast.top_picker_value()
            return gr.update(choices=ch, value=sel if any(v == sel for _, v in ch) else "")

        demo.load(_characters, outputs=character)
        # the engine and queue state now, not from the server's start (the timer waits 10 s)
        demo.load(lambda: (status_html(), pause_label()), outputs=[status, gpu_pause], show_progress="hidden")
        demo.load(settings_fn, outputs=settings_outs)
        agent_wire(demo)
        demo.load(voices_fn, outputs=voices_outs)
        demo.load(stories_fn, outputs=stories_outs, show_progress="hidden")
        demo.load(series_fn, outputs=series_outs, show_progress="hidden")
        # the Dashboard's "Open" shows the chosen series on the Series page
        open_pick, open_btn = ui_series.HANDLES["home_open"]
        open_btn.click(lambda sid: gr.update(choices=ui_series._series_choices(), value=sid), inputs=open_pick,
                       outputs=ui_series.HANDLES["series_pick"]).then(None, js=ui_series.go_js("series"))
        character.input(lambda v: cast.select(v or "") and None, inputs=character)

        def _sync_characters(sig):
            """A character saved or deleted elsewhere -> fresh choices in the top bar's Tools panel."""
            new_sig = repr(cast.top_picker_choices())
            return (gr.skip(), sig) if new_sig == sig else (_characters(), new_sig)
        char_sig = gr.State(repr(cast.top_picker_choices()))
        gr.Timer(6.0).tick(_sync_characters, inputs=char_sig, outputs=[character, char_sig])

    return demo


def main(*, on_ready: Callable[[], None] | None = None) -> None:
    config.ensure_dirs()
    characters.startup()  # a save cut short by a crash is settled before any page lists the characters
    items.startup()  # the same for places and items
    agent_runner.startup()  # a run a restart cut short is marked interrupted, so it can be resumed
    launch_options = local_launch_options(config.SETTINGS["app"])

    app_cfg = config.SETTINGS["app"]
    # The generation libraries load in the background, so the first job after a start does not wait for them.
    engine_runtime.start_warm_imports()
    if on_ready is None:
        queue_manager.start()
    demo = build_ui()
    try:
        demo.launch(
            **launch_options,
            server_port=app_cfg["port"],
            inbrowser=app_cfg.get("open_browser", True),
            **STYLE,  # Gradio 6: motyw i CSS przekazujemy do launch()
            quiet=True,
            prevent_thread_lock=True,
        )
        if on_ready is not None:
            queue_manager.start()
            on_ready()
        demo.block_thread()
    finally:
        try:
            llm.shutdown()
        finally:
            engine_runtime.release()


if __name__ == "__main__":
    main()
