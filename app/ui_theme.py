# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The look of the app: a dark, minimalist grayscale theme and shared page elements.

Rules: only achromatic colors (R = G = B, alpha allowed) anywhere - the Gradio theme in both
modes, the CSS below, inline styles and SVG data URIs (tests/test_theme.py enforces it). A
near-black page, slightly lighter cards with hairline borders, light primary buttons with dark
text, white only for selection, focus and progress; no gradients, glow or emoji in the menu.
Status never depends on color: ok = filled dot, warning = hollow ring, error = filled square and
a bold label. Always dark: the page script keeps the "dark" class that Gradio removes when the
system theme is light, and the light theme variables equal the dark ones, so nothing changes
before the script runs. Gradio's own stylesheets hard-code a few hues (its color palette, toasts,
form defaults, slider thumbs, code highlighting); the CSS below turns them gray. System fonts
(Inter, Segoe UI Variable on Windows 11) - the app works offline.

Layout: one top bar on every page - the brand, the workspace menu and a Tools menu (the other pages
and the character choice) on the left, the engine status and the GPU pause on the right; the pages
below it are gr.Tabs without their own tab bar - the menus switch them in the browser (no server
round trip), so Tab.select events work as before. Fields inside cards (.card) have no frames of
their own - only text inputs do.
"""
from __future__ import annotations

import hashlib
import html
import inspect
import io
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.parse import quote

import gradio as gr
from PIL import Image, ImageDraw
from gradio.processing_utils import save_file_to_cache
from gradio.utils import get_upload_folder


def cache_media(path: str | Path | None) -> Optional[str]:
    """Copy a character preview or a selected voice sample to Gradio's file cache: the private
    library folders themselves are never served (main.local_launch_options)."""
    if not path:
        return None
    try:
        return save_file_to_cache(path, get_upload_folder())
    except OSError:
        return None


def favicon_link() -> str:
    """The page's icon link, versioned by the icon's bytes. A browser keeps /favicon.ico in a cache of its own
    across reloads and restarts; a changed icon gets a new address, so it is fetched again."""
    version = hashlib.sha256(favicon_png()).hexdigest()[:12]
    return f'<link rel="icon" type="image/png" href="/favicon.ico?v={version}">'


# --- colors: one source for the Gradio theme and the --c-* tokens of the CSS below ---
BG, SURFACE, SURFACE_2, MUTED = "#0a0a0a", "#121212", "#181818", "#1c1c1c"
BORDER, BORDER_2 = "#262626", "#3a3a3a"
TEXT, TEXT_2, TEXT_3 = "#f2f2f2", "#bfbfbf", "#8f8f8f"
ACCENT, ACCENT_SOFT = "#ffffff", "#2a2a2a"
OK, WARN, ERR = "#e6e6e6", "#a6a6a6", "#ffffff"  # always paired with a shape (.st-*, .badge.*)
TOKENS = {"bg": BG, "surface": SURFACE, "surface-2": SURFACE_2, "muted": MUTED,
          "border": BORDER, "border-2": BORDER_2, "text": TEXT, "text-2": TEXT_2,
          "text-3": TEXT_3, "accent": ACCENT, "accent-soft": ACCENT_SOFT,
          "ok": OK, "warn": WARN, "err": ERR}
HOVER = "#202020"           # secondary buttons and option chips under the pointer
SECONDARY_BORDER = "#2e2e2e"
STOP_BORDER = "#555555"     # stop buttons: outlined, light text
CONTROL_BORDER = "#666666"  # checkbox frames, 3:1 against the cards
PLACEHOLDER = "#7a7a7a"     # 4.5:1 on inputs
WAVE = "#666666"            # the part of an audio waveform not played yet


def _mark(shape: str) -> str:
    """Gradio's checkbox tick or radio dot, drawn dark on the light 'selected' fill."""
    svg = f"<svg viewBox='0 0 16 16' fill='black' xmlns='http://www.w3.org/2000/svg'>{shape}</svg>"
    return f'url("data:image/svg+xml,{quote(svg)}")'


_TICK = ("<path d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414"
         "L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/>")
_DOT = "<circle cx='8' cy='8' r='3'/>"

_COLORS = dict(
    body_background_fill=BG, body_text_color=TEXT, body_text_color_subdued=TEXT_3,
    background_fill_primary=SURFACE, background_fill_secondary=SURFACE_2,
    border_color_primary=BORDER, border_color_accent=BORDER_2, border_color_accent_subdued=BORDER_2,
    color_accent=ACCENT, color_accent_soft=ACCENT_SOFT, code_background_fill=MUTED,
    link_text_color=TEXT, link_text_color_hover=ACCENT,
    link_text_color_visited=TEXT_2, link_text_color_active=ACCENT,
    block_background_fill="transparent", block_border_color="transparent",
    block_label_background_fill="transparent", block_label_border_color=BORDER,
    block_label_text_color=TEXT_2, block_title_background_fill="transparent",
    block_title_border_color=BORDER, block_title_text_color=TEXT_2, block_info_text_color=TEXT_3,
    panel_background_fill=SURFACE, panel_border_color=BORDER,
    accordion_text_color=TEXT, table_text_color=TEXT,
    input_background_fill=BG, input_background_fill_hover=BG, input_background_fill_focus=BG,
    input_border_color=BORDER, input_border_color_hover=BORDER_2, input_border_color_focus=TEXT_2,
    input_placeholder_color=PLACEHOLDER,
    button_primary_background_fill=TEXT, button_primary_background_fill_hover=ACCENT,
    button_primary_text_color=BG, button_primary_text_color_hover=BG,
    button_primary_border_color=TEXT, button_primary_border_color_hover=ACCENT,
    button_secondary_background_fill=SURFACE_2, button_secondary_background_fill_hover=HOVER,
    button_secondary_text_color=TEXT, button_secondary_text_color_hover=ACCENT,
    button_secondary_border_color=SECONDARY_BORDER, button_secondary_border_color_hover=BORDER_2,
    button_cancel_background_fill="transparent",
    button_cancel_background_fill_hover="rgba(255, 255, 255, .06)",
    button_cancel_text_color=TEXT, button_cancel_text_color_hover=ACCENT,
    button_cancel_border_color=STOP_BORDER, button_cancel_border_color_hover=TEXT_3,
    checkbox_background_color=BG, checkbox_background_color_hover=BG,
    checkbox_background_color_focus=BG, checkbox_background_color_selected=TEXT,
    checkbox_border_color=CONTROL_BORDER, checkbox_border_color_hover=TEXT_3,
    checkbox_border_color_focus=TEXT, checkbox_border_color_selected=TEXT,
    checkbox_check=_mark(_TICK), radio_circle=_mark(_DOT),
    checkbox_label_background_fill=SURFACE_2, checkbox_label_background_fill_hover=HOVER,
    checkbox_label_background_fill_selected=ACCENT_SOFT,
    checkbox_label_border_color=BORDER, checkbox_label_border_color_hover=BORDER_2,
    checkbox_label_border_color_selected=BORDER_2,
    checkbox_label_text_color=TEXT_2, checkbox_label_text_color_selected=TEXT,
    error_background_fill=MUTED, error_border_color=TEXT_2, error_text_color=TEXT,
    error_icon_color=ERR,
    loader_color=ACCENT, slider_color=TEXT, stat_background_fill=ACCENT_SOFT,
    table_border_color=BORDER, table_even_background_fill=SURFACE,
    table_odd_background_fill=SURFACE_2, table_row_focus=ACCENT_SOFT,
    block_shadow="none", block_label_shadow="none", shadow_drop="none",
    shadow_drop_lg="0 12px 32px rgba(0, 0, 0, .55)", shadow_inset="inset 0 2px 4px rgba(0, 0, 0, .3)",
    input_shadow="none", input_shadow_focus="0 0 0 3px rgba(255, 255, 255, .08)",
    button_primary_shadow="none", button_primary_shadow_hover="none",
    button_primary_shadow_active="none", button_secondary_shadow="none",
    button_secondary_shadow_hover="none", button_secondary_shadow_active="none",
    button_cancel_shadow="none", button_cancel_shadow_hover="none", button_cancel_shadow_active="none",
    checkbox_shadow="none", checkbox_label_shadow="none", checkbox_label_shadow_hover="none",
    checkbox_label_shadow_active="none",
)
_SIZES = dict(
    block_border_width="0px", block_radius="12px", block_padding="0px",
    block_label_border_width="0px", block_label_text_weight="500", block_label_text_size="13px",
    block_title_text_weight="500", block_title_text_size="13px", block_title_padding="0 0 6px 0",
    block_label_padding="0 0 6px 0", block_info_text_size="12.5px",
    input_radius="8px", input_padding="9px 12px", input_text_size="14px",
    input_border_width="1px",
    button_border_width="1px",
    button_large_padding="10px 18px", button_large_radius="8px", button_large_text_size="14px",
    button_large_text_weight="600", button_medium_padding="8px 14px", button_medium_radius="8px",
    button_medium_text_size="14px", button_medium_text_weight="500",
    button_small_padding="6px 12px", button_small_radius="8px", button_small_text_size="13px",
    button_small_text_weight="500", checkbox_border_radius="4px", checkbox_label_padding="6px 12px",
    container_radius="12px", layout_gap="16px", form_gap_width="0px", table_radius="10px",
)


def _theme() -> gr.themes.Base:
    gray = gr.themes.colors.neutral  # pure gray; zinc, slate and the others are tinted
    theme = gr.themes.Base(
        primary_hue=gray, secondary_hue=gray, neutral_hue=gray,
        radius_size=gr.themes.sizes.radius_lg, spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_md,
        font=[gr.themes.Font(f) for f in ("Inter", "Segoe UI Variable Text", "Segoe UI",
                                          "system-ui", "-apple-system", "sans-serif")],
        font_mono=[gr.themes.Font(f) for f in ("Cascadia Mono", "Consolas", "monospace")])
    allowed = set(inspect.signature(theme.set).parameters)  # other Gradio versions: no error
    chosen = {k: v for k, v in {**_COLORS, **_SIZES}.items() if k in allowed}
    theme.set(**chosen)
    # Always dark: a variable set above is the same in both modes; every other one takes Gradio's
    # dark default in light mode too, so the page looks the same before the script adds "dark".
    for dark in [name for name in vars(theme) if name.endswith("_dark")]:
        light = dark[:-5]
        if not hasattr(theme, light):
            continue
        if light in chosen or getattr(theme, dark) is None:
            setattr(theme, dark, getattr(theme, light))
        else:
            setattr(theme, light, getattr(theme, dark))
    return theme


THEME = _theme()


def waveform() -> gr.WaveformOptions:
    """Audio waveforms in grays: Gradio draws them on a canvas with tinted defaults CSS can't reach."""
    return gr.WaveformOptions(waveform_color=WAVE, waveform_progress_color=TEXT,
                              trim_region_color="rgba(255, 255, 255, .16)")


# --- ikony menu (24x24, kreska) - jako maska CSS, kolor bierze z tekstu ---
ICONS = {
    "home": '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" '
            'height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/>'
            '<rect x="3" y="16" width="7" height="5" rx="1.5"/>',
    "video": '<rect x="2.5" y="5.5" width="14" height="13" rx="2.5"/>'
             '<path d="m16.5 10.5 5-3v9l-5-3"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2.5"/><circle cx="9" cy="9" r="1.8"/>'
             '<path d="m21 15-4.5-4.5L6 21"/>',
    "universes": '<path d="M21 8 12 3 3 8v8l9 5 9-5z"/><path d="m3 8 9 5 9-5"/><path d="M12 13v8"/>',
    "characters": '<circle cx="10" cy="8" r="4"/><path d="M2.5 21a7.5 7.5 0 0 1 12.8-5.3"/>'
                  '<path d="M19 14.5v6M16 17.5h6"/>',
    "gallery": '<rect x="7" y="7" width="14" height="14" rx="2.5"/>'
               '<path d="M17 3H5.5A2.5 2.5 0 0 0 3 5.5V17"/><path d="m21 17-3.5-3.5L11 20"/>',
    "voices": '<rect x="9" y="2.5" width="6" height="11.5" rx="3"/>'
              '<path d="M5.5 11a6.5 6.5 0 0 0 13 0"/><path d="M12 17.5v4M8.5 21.5h7"/>',
    "queue": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "settings": '<path d="M3 6h11M18 6h3M3 12h4M11 12h10M3 18h13M20 18h1"/>'
                '<circle cx="16" cy="6" r="2"/><circle cx="9" cy="12" r="2"/>'
                '<circle cx="18" cy="18" r="2"/>',
}
ICONS["styles"] = ('<path d="M12 3a9 9 0 0 0 0 18c1.2 0 2-.9 2-2 0-.6-.2-1-.6-1.4-.3-.4-.5-.8-.5-1.3 '
                   '0-1.1.9-2 2-2h2.3a3.8 3.8 0 0 0 3.8-3.8C21 6.4 17 3 12 3z"/><circle cx="7.5" cy="11.5" r="1.3"/>'
                   '<circle cx="10" cy="7.3" r="1.3"/><circle cx="14.6" cy="7.3" r="1.3"/>')
# Places and items (app/ui_items.py): a location pin
ICONS["items"] = '<path d="M12 21s-6.5-6.2-6.5-11a6.5 6.5 0 0 1 13 0c0 4.8-6.5 11-6.5 11z"/><circle cx="12" cy="10" r="2.3"/>'
ICONS["timeline"] = ('<path d="M3 6h18M3 12h18M3 18h18M9 3v18"/>'
                     '<rect x="11" y="9" width="7" height="6" rx="1.5" fill="white"/>')
# The Agent page (app/ui_agent.py): a four-pointed spark and a dot, geometry of its own
ICONS["agent"] = ('<path d="M10.5 3.5 12.3 9l5.5 1.8-5.5 1.8L10.5 18l-1.8-5.4L3.2 10.8 8.7 9z"/>'
                  '<circle cx="18.5" cy="18" r="1.8"/>')
ICONS["stories"] = ICONS["home"]
ICONS["series"] = ('<rect x="3" y="5" width="18" height="12" rx="2"/><path d="M8 21h8M12 17v4"/>'
                   '<path d="m10 9 4 2-4 2z"/>')


def _svg_url(body: str) -> str:
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
           'stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
           f'{body}</svg>')
    return "data:image/svg+xml;utf8," + quote(svg)


_STAR = ((12, 3.5), (13.9, 9), (19.5, 10.9), (13.9, 12.8), (12, 18.5), (10.1, 12.8), (4.5, 10.9), (10.1, 9))


def favicon_png(size: int = 64) -> bytes:
    """The browser tab and app icon: the top bar's logo star, dark on a light rounded tile, in grays only
    (Gradio's default icon is a yellow and orange logo). A grayscale PNG with transparent corners."""
    scale = 4  # drawn larger and reduced, for smooth edges
    big = size * scale
    image = Image.new("LA", (big, big), (0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, big - 1, big - 1), radius=round(big * 0.24), fill=(int(TEXT[1:3], 16), 255))
    unit = big * 17 / 30 / 24  # the star fills 17 of the tile's 30 px, about as in the top bar's logo
    offset = (big - 24 * unit) / 2
    draw.polygon([(offset + x * unit, offset + y * unit) for x, y in _STAR], fill=(int(BG[1:3], 16), 255))
    data = io.BytesIO()
    image.resize((size, size), Image.Resampling.LANCZOS).save(data, format="PNG")
    return data.getvalue()


def favicon(folder: str | Path | None = None) -> Optional[str]:
    """favicon_png() saved in Gradio's cache folder (or `folder`) for launch(favicon_path=...); the file is
    written only when it is missing or differs. Returns its path, or None when it cannot be written (the
    page then keeps working with Gradio's icon)."""
    data = favicon_png()
    target = Path(folder or get_upload_folder()) / "contsy-favicon.png"
    try:
        if not (target.is_file() and target.read_bytes() == data):
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(f".{target.name}.tmp")
            staged.write_bytes(data)
            staged.replace(target)
        return str(target)
    except OSError:
        return None


# the logo in the top bar: a dark star on the light tile, as in favicon_png(), and the name (hidden on a
# narrow window). Its colors are its own: a gr.HTML with Gradio's default CSS gives every element inside
# it the page's text color (.prose *), so currentColor would draw it white.
BRAND = ("<div class='brand'><div class='logo'><svg viewBox='0 0 24 24' aria-hidden='true' focusable='false'>"
         f"<path d='M12 3.5 13.9 9l5.6 1.9-5.6 1.9L12 18.5l-1.9-5.7L4.5 10.9 10.1 9z' fill='{BG}' stroke='{BG}' "
         "stroke-width='1.4' stroke-linejoin='round'/></svg>"
         "</div><span class='b-name'>Contsy Stories</span></div>")


def _tokens_css() -> str:
    """The --c-* tokens (from TOKENS) and page-wide dark defaults for native controls."""
    tokens = " ".join(f"--c-{name}: {value};" for name, value in TOKENS.items())
    return (":root { color-scheme: dark; accent-color: var(--c-text); "
            "scrollbar-color: var(--c-border-2) transparent; " + tokens + " }\n")


# Gradio's frontend has its own Tailwind palette (--color-red-500 ...) that some built-in parts
# use directly (toasts, video controls, recorder); every hue becomes the gray of the same step.
_PALETTE_HUES = ("grey", "slate", "zinc", "stone", "red", "orange", "amber", "yellow", "lime",
                 "green", "teal", "cyan", "sky", "blue", "indigo", "violet", "purple", "fuchsia",
                 "pink", "rose")
_PALETTE_STEPS = ("50", "100", "200", "300", "400", "500", "600", "700", "800", "900", "950")


def _palette_css() -> str:
    gray = gr.themes.colors.neutral
    steps = {"10": "#fdfdfd", **{step: getattr(gray, f"c{step}") for step in _PALETTE_STEPS}}
    names = [f"--color-{hue}-{step}: {value};" for hue in _PALETTE_HUES for step, value in steps.items()]
    names += [f"--color-{hue}: {gray.c500};" for hue in _PALETTE_HUES]
    return ":root { " + " ".join(names) + " }\n"


def chevron() -> str:
    """The arrow of a native select: a gray chevron (Gradio's form defaults draw a tinted one)."""
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='none'><path d='m6 8 4 4 4-4' "
           f"stroke='{TEXT_3}' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/></svg>")
    return f'url("data:image/svg+xml,{quote(svg)}")'


def _select_css() -> str:
    # Gradio's own rule has the same specificity and may come later, hence !important; a rule that sets the
    # background shorthand (Timeline editors) keeps the arrow too.
    return (".gradio-container select { appearance: none; -webkit-appearance: none; "
            f"background-image: {chevron()} !important; background-repeat: no-repeat !important; "
            "background-position: right 8px center !important; background-size: 16px 16px !important; "
            "padding-right: 28px; }\n")


CSS = _tokens_css() + _palette_css() + """
/* Contsy Stories (app/ui_series.py) */
.story-stat { padding: 14px 16px !important; }
.story-stat .stat-label { font-size: 12px; opacity: .75; text-transform: uppercase; letter-spacing: .04em; }
.story-stat .stat-value { font-size: 28px; font-weight: 700; margin: 4px 0; }
.story-stat .stat-sub, .muted { font-size: 12px; opacity: .7; }
.story-chat { display: flex; flex-direction: column; gap: 6px; max-height: 360px; overflow-y: auto; }
.story-chat .chat-msg { padding: 8px 12px; border-radius: 10px; border: 1px solid var(--c-border); }
.story-chat .chat-msg.owner { align-self: flex-end; background: var(--c-surface); }
body, gradio-app, .gradio-container { background: var(--c-bg) !important; }
/* The app fills the window: Gradio's main column has no cap (.fillable: 640 px … 1920 px by window width)
   and no padding, the top bar spans the window and every page sets its own width below it (#pages).
   overflow: clip instead of Gradio's hidden clips the same way, but is no scroll container, so the top
   bar can stick to the window. No @media here: Gradio prefixes rules inside @media with ".contain",
   so they would miss .main and .wrap. */
.gradio-container { max-width: none !important; width: 100% !important; padding: 0 !important;
  overflow: clip !important; }
.gradio-container .main { max-width: none !important; padding: 0 !important; }
.gradio-container .main > .wrap { padding: 0 !important; }
/* Gradio's root column puts its 16 px gap between the top bar and the pages: they touch */
.gradio-container .contain .column:has(> #topbar) { gap: 0 !important; }
/* toasts open below the top bar (--topbar-height: its height, from the page script), not over its
   status and pause button */
.gradio-container .toast-wrap { --toast-top: var(--topbar-height, 44px) !important; }
footer { display: none !important; }
::selection { background: rgba(160, 160, 160, .4); }
.prose, .prose p, .prose li { color: var(--c-text-2); }
.prose strong { color: var(--c-text); font-weight: 600; }
.prose a { text-underline-offset: 2px; }
/* keyboard focus: one white ring everywhere (text fields show focus with their border) */
.gradio-container :is(a, button, summary, select, [role="button"], [role="tab"], [role="option"],
  [tabindex]):focus-visible,
.gradio-container input:is([type="checkbox"], [type="radio"], [type="range"]):focus-visible {
  outline: 2px solid var(--c-accent); outline-offset: 2px; }

/* --- top bar: on every page, 44 px, the window's width; it sticks to the top of the window, above the
   page (Gradio's gallery preview is at 999); Gradio's fullscreen views (1000, later in the page), toasts
   and dropdown lists stay above it --- */
#topbar { position: sticky; top: 0; z-index: 1000; display: flex !important; flex-wrap: nowrap !important;
  align-items: center !important; gap: 10px !important; min-height: 44px; margin: 0 !important;
  padding: 6px 12px 6px 14px !important; background: var(--c-surface) !important;
  border: 0 !important; border-bottom: 1px solid var(--c-border) !important; border-radius: 0 !important;
  box-shadow: none !important; }
/* Gradio stretches every child of a row (flex: 1, a 160 px minimum): here each keeps its own width */
#topbar > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important;
  margin: 0 !important; }
#topbar > #engine-status { flex: 0 1 auto !important; margin-left: auto !important; }
/* Gradio's loading overlay does not fit a 30 px control: the chip dims while it updates */
#topbar [data-testid="status-tracker"] { display: none !important; }
.topbar-start { display: flex; align-items: center; flex-wrap: wrap; gap: 4px 10px; min-width: 0; }
.brand { display: flex; align-items: center; gap: 9px; flex: none; padding-right: 4px; }
.brand .logo { width: 28px; height: 28px; border-radius: 7px; background: var(--c-text);
  color: var(--c-bg); display: flex; align-items: center; justify-content: center; flex: none; }
.brand .logo svg { width: 16px; height: 16px; }
.brand .b-name { font-weight: 650; font-size: 14.5px; color: var(--c-text); letter-spacing: -0.01em;
  line-height: 1.2; white-space: nowrap; }
/* the menus: page buttons with an icon; the workspace menu runs across the bar, the Tools one down
   its panel. The same weight for every state: the bar does not shift when the page changes. */
.workspace-nav { display: flex; flex-wrap: wrap; gap: 2px; min-width: 0; }
.workspace-nav button, .tools-toggle { display: flex; align-items: center; gap: 8px; min-height: 30px;
  margin: 0; padding: 0 10px; border: 0; border-radius: 7px; background: transparent;
  color: var(--c-text-2); font-family: inherit; font-size: 13.5px; font-weight: 550; line-height: 1;
  text-align: left; white-space: nowrap; cursor: pointer; }
.workspace-nav button:hover, .tools-toggle:hover { background: var(--c-muted); color: var(--c-text); }
.workspace-nav button[aria-current="page"] { background: var(--c-accent-soft); color: var(--c-text); }
.workspace-nav button:focus-visible, .tools-toggle:focus-visible {
  outline: 2px solid var(--c-accent); outline-offset: 2px; }
.workspace-nav .nav-icon { width: 16px; height: 16px; flex: none; background: currentColor;
  -webkit-mask: var(--icon) center / contain no-repeat; mask: var(--icon) center / contain no-repeat; }
/* Tools: a disclosure button; it is marked like the current page while a Tools page is shown */
.tools-toggle[aria-expanded="true"] { background: var(--c-muted); color: var(--c-text); }
.tools-toggle[data-current] { background: var(--c-accent-soft); color: var(--c-text); }
.tools-chevron { width: 6px; height: 6px; margin: -3px 1px 0 0; flex: none; border: solid currentColor;
  border-width: 0 1.5px 1.5px 0; transform: rotate(45deg); }
.tools-toggle[aria-expanded="true"] .tools-chevron { margin-top: 3px; transform: rotate(225deg); }
/* its panel (a Gradio column: the Tools pages and the character choice) opens below the button; the
   page script places it (--tools-left, --tools-top) and opens it (data-open) */
#topbar > #topbar-tools { position: absolute !important; z-index: 2; top: var(--tools-top, calc(100% + 4px));
  left: var(--tools-left, 8px); display: none !important; flex-direction: column !important;
  width: min(280px, calc(100vw - 16px)) !important; gap: 6px !important; padding: 6px !important;
  background: var(--c-surface-2); border: 1px solid var(--c-border-2); border-radius: 10px;
  box-shadow: 0 16px 40px rgba(0, 0, 0, .6); }
#topbar > #topbar-tools[data-open] { display: flex !important; }
#topbar-tools .workspace-nav { flex-direction: column; flex-wrap: nowrap; }
#topbar-tools .workspace-nav button { width: 100%; min-height: 32px; font-size: 13px; }
#topbar-tools #character { padding: 10px 6px 4px !important; border-top: 1px solid var(--c-border-2) !important;
  border-radius: 0 !important; }
#character .wrap { border-radius: 8px !important; }
/* the engine chip: the state by shape (ready = filled dot, paused = hollow ring, unavailable = square
   and a bold label), one short line, every detail in its tooltip */
.engine-chip { display: flex; align-items: center; gap: 7px; min-width: 0; height: 30px; padding: 0 11px;
  border: 1px solid var(--c-border); border-radius: 999px; background: var(--c-bg); color: var(--c-text-3);
  font-size: 12.5px; line-height: 1; white-space: nowrap; font-variant-numeric: tabular-nums; cursor: default; }
.engine-chip .engine-text { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.engine-chip b { color: var(--c-text-2); font-weight: 600; }
.engine-chip[data-state="off"] b { color: var(--c-text); font-weight: 700; }
.engine-chip .dot { flex: none; width: 7px; height: 7px; border-radius: 50%; background: var(--c-ok); }
.engine-chip .dot.off { border-radius: 1px; background: var(--c-err); }
.engine-chip .dot.pause { background: none; box-shadow: inset 0 0 0 1.5px var(--c-warn); }
.engine-chip .engine-sr { position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
#gpu-pause { min-height: 30px; padding: 0 12px; font-size: 13px; white-space: nowrap; }

/* --- pages: tabs without their own bar (the menus switch them), a readable width below the top bar --- */
#pages { border: 0 !important; padding: 28px clamp(16px, 4vw, 40px) 48px !important;
  background: none !important; }
/* Gradio's own tab buttons stay laid out (it measures them, the menus click them) but out of the Tab order
   and out of the accessibility tree: the top bar's menus are the one way between pages. */
#pages > .tab-wrapper { position: absolute !important; left: -10000px; top: 0; width: 4000px;
  height: 1px; overflow: hidden; visibility: hidden; }
#pages > .tabitem, #pages .tabitem { border: 0 !important; padding: 0 !important;
  background: none !important; }
#pages > .tabitem { width: 100%; max-width: 1240px; margin: 0 auto !important; }
/* Timeline: an editor workspace from edge to edge below the top bar (laid out by app/ui_timeline_view.py).
   A page that is not shown has an inline "display: none" on its tab panel. */
#pages:has(> #timeline-page:not([style*="display: none"])) { padding: 0 !important; }
#pages > #timeline-page { max-width: none; }
.page-head { padding: 0 0 4px; }
.page-head h1 { font-size: 24px; font-weight: 650; letter-spacing: -0.025em; margin: 0;
  color: var(--c-text); line-height: 1.25; }
.page-head p { margin: 4px 0 0; color: var(--c-text-3); font-size: 14px; }
.page-head p b { color: var(--c-text-2); font-weight: 600; }
.section-title { font-size: 15px; font-weight: 600; color: var(--c-text); margin: 0; }
.section-sub { font-size: 13px; color: var(--c-text-3); margin: 2px 0 0; }

/* --- cards: one step lighter than the page, hairline border; fields inside have no frames --- */
.card { background: var(--c-surface) !important; border: 1px solid var(--c-border) !important;
  border-radius: 12px !important; padding: 20px !important; gap: 16px !important;
  box-shadow: none; }
.card.tight { padding: 14px 16px !important; }
.card > .form, .card .form { border: 0 !important; background: none !important; gap: 16px !important;
  box-shadow: none !important; }
.form { gap: 16px !important; border: 0 !important; background: none !important; }
.block.padded { padding: 0 !important; }
.hint p, .hint li, .hint { font-size: 13px !important; color: var(--c-text-3) !important; }
.status-box p { margin: 2px 0; font-size: 13.5px; color: var(--c-text-2); }
.status-box ul { margin: 6px 0 0; }
/* status marks from note(): the shape carries the meaning, not a hue -
   ok = filled dot, warning = hollow ring, error = filled square and a bold line, info = dash */
.st { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px;
  vertical-align: 1px; background: var(--c-ok); }
.st-ok { background: var(--c-ok); }
.st-warn { background: none; box-shadow: inset 0 0 0 1.5px var(--c-warn); }
.st-err { border-radius: 1.5px; background: var(--c-err); }
.st-info { height: 2px; border-radius: 1px; vertical-align: 3px; background: var(--c-text-2); }
:is(p, li):has(> .st-err) { color: var(--c-text); font-weight: 600; }

/* --- buttons (colors come from the theme: light primary, dark secondary, outlined stop) --- */
button.lg, button.md, button.sm { font-weight: 500; box-shadow: none !important; }
.cta { min-height: 44px !important; font-weight: 600 !important; }
.toolbar { align-items: center !important; gap: 8px !important; flex-wrap: wrap !important; }
.toolbar > * { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
.toolbar .grow, .toolbar > .form:has(.grow) { flex: 1 1 auto !important; }

/* --- choices (radio) as a segmented control --- */
#pages fieldset > span[data-testid="block-info"] { display: block; }
#pages fieldset .wrap:has(> label > input[type="radio"]) { display: inline-flex;
  flex-wrap: wrap; gap: 2px; padding: 3px; background: var(--c-muted); border-radius: 10px;
  width: auto; }
#pages fieldset .wrap > label:has(> input[type="radio"]) { background: none !important;
  border: 0 !important; box-shadow: none !important; padding: 6px 12px !important;
  border-radius: 7px !important; color: var(--c-text-3) !important; font-weight: 500;
  font-size: 13.5px; cursor: pointer; }
#pages fieldset .wrap > label:has(> input[type="radio"]):hover { color: var(--c-text) !important; }
#pages fieldset .wrap > label:has(> input[type="radio"]).selected {
  background: var(--c-accent-soft) !important; color: var(--c-text) !important;
  box-shadow: inset 0 0 0 1px var(--c-border-2) !important; }
#pages fieldset .wrap > label:has(> input[type="radio"]:focus-visible) {
  outline: 2px solid var(--c-accent); outline-offset: 1px; }
#pages fieldset .wrap > label > input[type="radio"] { position: absolute; opacity: 0;
  width: 1px; height: 1px; pointer-events: none; }
#pages fieldset .wrap > label > input[type="radio"] + span { margin: 0 !important; }

/* --- switches (checkbox with the .switch class): off = gray track, on = light track --- */
.switch input[type="checkbox"] { appearance: none; -webkit-appearance: none; width: 34px !important;
  height: 20px !important; border-radius: 999px !important; border: 0 !important;
  background-color: var(--c-border-2) !important; background-image: none !important;
  position: relative; cursor: pointer; flex: none; transition: background-color .15s ease; }
.switch input[type="checkbox"]::after { content: ""; position: absolute; top: 2px; left: 2px;
  width: 16px; height: 16px; border-radius: 50%; background: var(--c-text-2);
  box-shadow: 0 1px 2px rgba(0, 0, 0, .4); transition: transform .15s ease, background-color .15s ease; }
.switch input[type="checkbox"]:checked { background-color: var(--c-text) !important; }
.switch input[type="checkbox"]:checked::after { transform: translateX(14px); background: var(--c-bg); }
.switch label { gap: 10px; font-weight: 500; color: var(--c-text); }

/* --- accordions --- */
.gr-accordion { overflow: visible !important; }
.card .gr-accordion { border-top: 1px solid var(--c-border) !important; border-radius: 0 !important;
  padding-top: 12px !important; }
.card .label-wrap, .accordion-card .label-wrap { font-weight: 600; color: var(--c-text); }
.block.gr-accordion.accordion-card { background: var(--c-surface) !important;
  border: 1px solid var(--c-border) !important; border-radius: 12px !important;
  padding: 14px 20px !important; }

/* --- indicators (dashboard, queue) --- */
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.kpi { background: var(--c-surface); border: 1px solid var(--c-border); border-radius: 12px;
  padding: 14px 16px; min-width: 0; }
.kpi .k-label { font-size: 12.5px; color: var(--c-text-3); font-weight: 500; }
.kpi .k-value { font-size: 22px; font-weight: 650; letter-spacing: -0.02em; margin-top: 4px;
  color: var(--c-text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  font-variant-numeric: tabular-nums; }
.kpi .k-value.small { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; line-height: 1.35;
  white-space: normal; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.kpi .k-sub { font-size: 12.5px; color: var(--c-text-3); margin-top: 3px; }
.bar { height: 6px; border-radius: 999px; background: var(--c-muted); overflow: hidden; margin-top: 10px; }
.bar > div { height: 100%; background: var(--c-accent); border-radius: 999px; transition: width .6s ease; }
.job { background: var(--c-surface); border: 1px solid var(--c-border); border-radius: 12px;
  padding: 16px 18px; }
.job.idle { color: var(--c-text-3); }
.job .j-title { font-weight: 600; color: var(--c-text); }
.job .j-meta { font-size: 13px; color: var(--c-text-3); margin-top: 8px;
  font-variant-numeric: tabular-nums; }

/* --- tables and badges --- */
.tbl-wrap { overflow-x: auto; }
.tbl { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13.5px;
  border: 0 !important; margin: 0 !important; }
.tbl th, .tbl td { border: 0 !important; background: none !important; }
.tbl th { text-align: left; font-weight: 500; color: var(--c-text-3); font-size: 12px;
  padding: 8px 12px; border-bottom: 1px solid var(--c-border) !important; white-space: nowrap; }
.tbl td { padding: 10px 12px; border-bottom: 1px solid var(--c-muted) !important; color: var(--c-text);
  vertical-align: middle; }
.tbl tr:last-child td { border-bottom: 0 !important; }
.tbl .muted { color: var(--c-text-3); }
.tbl .num { font-variant-numeric: tabular-nums; }
/* badge markers follow the status shapes: ok = dot, warn = ring, err = square, info = dash; a badge
   without a status (neutral) has a small hollow square, so it never looks like ok */
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 2px 9px; border-radius: 999px;
  font-size: 12px; font-weight: 500; background: var(--c-muted); color: var(--c-text-2);
  white-space: nowrap; }
.badge::before { content: ""; flex: none; width: 6px; height: 6px; border-radius: 1px;
  background: none; box-shadow: inset 0 0 0 1px currentColor; }
.badge.ok { color: var(--c-ok); }
.badge.ok::before { border-radius: 50%; background: currentColor; box-shadow: none; }
.badge.warn { color: var(--c-warn); }
.badge.warn::before { border-radius: 50%; background: none; box-shadow: inset 0 0 0 1.5px currentColor; }
.badge.err { color: var(--c-err); font-weight: 650; box-shadow: inset 0 0 0 1px var(--c-border-2); }
.badge.err::before { border-radius: 1px; background: currentColor; box-shadow: none; }
.badge.info { background: var(--c-accent-soft); color: var(--c-text); }
.badge.info::before { height: 2px; border-radius: 1px; background: currentColor; box-shadow: none; }
.empty { padding: 28px 16px; text-align: center; color: var(--c-text-3); font-size: 13.5px;
  border: 1px dashed var(--c-border-2); border-radius: 12px; background: var(--c-surface); }
.empty b { color: var(--c-text); display: block; font-size: 14.5px; margin-bottom: 4px; }

/* --- galleries --- */
.gallery-container .grid-wrap { padding: 2px !important; overflow-y: auto !important; }
.grid-container { gap: 10px !important; }
.thumbnail-item.thumbnail-lg { border-radius: 10px !important; border: 1px solid var(--c-border) !important;
  box-shadow: none !important; background: var(--c-muted) !important; transition: transform .15s ease,
  box-shadow .15s ease; }
.thumbnail-item.thumbnail-lg:hover { --ring-color: transparent; border-color: var(--c-border-2) !important;
  filter: none !important; transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, .5) !important; }
.thumbnail-item.thumbnail-lg:has(video)::before { content: ""; position: absolute; left: 8px; top: 8px;
  z-index: 3; width: 26px; height: 26px; border-radius: 999px; background: rgba(10, 10, 10, .72)
  url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M9 7v10l8-5z' fill='white'/%3E%3C/svg%3E")
  center / 16px no-repeat; }
.caption-label { left: 8px !important; right: auto !important; bottom: 8px !important;
  border: 0 !important; border-radius: 6px !important; background: rgba(10, 10, 10, .78) !important;
  color: var(--c-text-2) !important; font-size: 11.5px !important; padding: 2px 7px !important;
  opacity: 0; transition: opacity .15s ease; }
.thumbnail-lg:hover .caption-label { opacity: 1 !important; }
#studio-tiles .caption-label, #char-cards .caption-label, #item-cards .caption-label { opacity: 1 !important;
  font-weight: 600; color: var(--c-text) !important; }
#studio-tiles .grid-container, #char-cards .grid-container, #item-cards .grid-container {
  grid-template-columns: repeat(auto-fill, minmax(150px, 180px)) !important; }
#studio-tiles .grid-wrap { height: auto !important; min-height: 0 !important; max-height: 440px; }
#studio-recent .grid-container, #hist-gallery .grid-container {
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr)) !important; }
.gallery-container .icon-button-wrapper { background: none !important; box-shadow: none !important; }
video { border-radius: 10px; }
.result-card .block label[data-testid="block-label"] { display: none; }

/* --- Characters page: the list beside the one-screen creator (app/ui_characters.py); the steps of the
   creator are separated by a hairline, its photo grid keeps every photo in view with its number --- */
#characters-page .char-assets, #items-page .char-assets { padding: 8px 0; }
#characters-page .char-asset-badges, #items-page .char-asset-badges { display: flex; gap: 4px; flex-wrap: wrap;
  margin-top: 5px; }
#char-creator .char-creator-top, #item-creator .char-creator-top { align-items: flex-start !important; }
#char-creator .char-step, #item-creator .char-step { gap: 12px !important; padding-top: 16px !important;
  border-top: 1px solid var(--c-border) !important; }
#char-photos .grid-wrap, #item-photos .grid-wrap { height: auto !important; min-height: 0 !important; }
#char-photos .grid-container { grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)) !important; }
#item-photos .grid-container { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)) !important; }
#char-photos .caption-label, #item-photos .caption-label { opacity: 1 !important; font-weight: 600;
  color: var(--c-text) !important; }

/* Gallery: a selected thumbnail gets its frame and tick at once (Gradio does not redraw the
   gallery on click, so the browser adds the mark) */
#hist-gallery .thumbnail-lg { position: relative; }
#hist-gallery .thumbnail-lg.hist-sel { outline: 3px solid var(--c-accent) !important; outline-offset: -3px; }
#hist-gallery .thumbnail-lg.hist-sel::after { content: ""; position: absolute; top: 8px; right: 8px;
  width: 24px; height: 24px; border-radius: 999px; z-index: 5; background: var(--c-accent)
  url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m5 12.5 4.5 4.5L19 7.5'/%3E%3C/svg%3E")
  center / 14px no-repeat; box-shadow: 0 1px 4px rgba(0, 0, 0, .5); }

/* --- Gradio's own hard-coded hues, made gray --- */
/* Tailwind form defaults (white fields, blue checkboxes) for inputs no component restyles */
.gradio-container :is([type="text"], [type="email"], [type="url"], [type="password"], [type="number"],
  [type="date"], [type="datetime-local"], [type="month"], [type="search"], [type="tel"],
  [type="time"], [type="week"], [multiple], textarea, select) {
  background-color: var(--input-background-fill); border-color: var(--input-border-color); }
.gradio-container :is([type="checkbox"], [type="radio"]) { color: var(--c-text); }
.gradio-container input[type="checkbox"]:indeterminate::after { background-color: var(--c-bg) !important; }
/* toasts: the icon and the timer bar follow --toast-color; the icons differ by shape */
.toast-body.error { --toast-color: var(--c-err) !important; border-color: var(--c-text-2) !important; }
.toast-body.warning { --toast-color: var(--c-warn) !important; border-color: var(--c-border-2) !important; }
.toast-body.info { --toast-color: var(--c-text-2) !important; }
.toast-body.success { --toast-color: var(--c-ok) !important; }
/* dropdown lists */
div.options:has(> [role="listbox"]) { background: var(--c-surface-2) !important;
  border: 1px solid var(--c-border-2); }
[data-testid="dropdown-option"]:is(:hover, .active) { background: var(--c-accent-soft) !important; }
/* sliders: dark track, a neutral ring around the white thumb */
.slider_input_container input[type="range"]::-webkit-slider-runnable-track {
  background: linear-gradient(to right, var(--slider-color) var(--range_progress, 0%),
  var(--c-border-2) var(--range_progress, 0%)) !important; }
.slider_input_container input[type="range"]::-moz-range-track { background: var(--c-border-2) !important; }
.slider_input_container input[type="range"]::-webkit-slider-thumb {
  box-shadow: 0 0 0 1px rgba(0, 0, 0, .6), 0 1px 3px rgba(0, 0, 0, .5) !important; }
.slider_input_container input[type="range"]::-moz-range-thumb {
  box-shadow: 0 0 0 1px rgba(0, 0, 0, .6), 0 1px 3px rgba(0, 0, 0, .5) !important; }
/* audio recorder (its buttons borrow the transparent block border), the waveform cursor (inside
   its shadow root), video trimming */
:is(.record-button, .pause-button, .resume-button, .stop-button-paused) {
  border-color: var(--c-border-2) !important; }
:is(.stop-button, .spinner-button) { border-color: var(--c-text-2) !important; }
:is(.record-button, .stop-button, .stop-button-paused)::before { background: var(--c-text) !important; }
.spinner-button .icon { fill: var(--c-text) !important; stroke: var(--c-text) !important; }
::part(cursor) { background-color: var(--c-text) !important; }
.opaque-layer { background-color: rgba(255, 255, 255, .12) !important; }
/* errors in gr.HTML scripts, code highlighting in Markdown */
.error-container:has(> .error-title) { color: var(--c-text) !important;
  background-color: var(--c-surface-2) !important; border-color: var(--c-border-2) !important; }
.error-container .error-component-name { background-color: var(--c-muted) !important; }
.md .token { color: inherit !important; }
.md .token:is(.comment, .prolog, .doctype, .cdata) { color: var(--c-text-3) !important; }
.md :is(pre, code)::selection, .md :is(pre, code) ::selection {
  background: rgba(160, 160, 160, .4) !important; }

/* --- narrow window / phone: the top bar keeps one line down to 721 px (the brand shrinks to its logo,
   the workspace menu drops its icons, the status line ends in an ellipsis), then wraps: the menus on
   the first line, the status and the pause button on the second --- */
@media (max-width: 1100px) {
  .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  #topbar .brand .b-name { display: none; }
}
@media (max-width: 900px) { #topbar #workspace-nav .nav-icon { display: none; } }
@media (max-width: 768px) { .kpis { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 720px) {
  #topbar { flex-wrap: wrap !important; row-gap: 6px !important; padding: 6px 8px !important; }
  #topbar > #workspace-nav { flex: 1 1 100% !important; }
  #topbar > #engine-status { flex: 1 1 0 !important; margin-left: 0 !important; }
  #topbar .topbar-start { gap: 4px 6px; }
  #topbar #workspace-nav .workspace-nav button { padding: 0 7px; }
  #topbar .tools-toggle { padding: 0 7px; }
}
""" + _select_css()

# Straznik starej karty. Gradio wysyla zdarzenia po numerze, a aplikacja po restarcie moze miec
# inny uklad - zdarzenie ze starej karty trafiloby do innej funkcji (w logu: "suggest ... needed
# 2, got 0"; rownie dobrze mogla to byc pauza). Gradio sam przeladowuje karte tylko, gdy jakies
# zdarzenie nie przejdzie w trakcie restartu - karta w tle moze restart przespac. Dlatego przed
# kazdym zdarzeniem karta pyta serwer o app_id (nowy proces = nowe): inne - zdarzenie nie
# wychodzi, karta sie przeladowuje. Gdy sprawdzic sie nie da (inna wersja Gradio) - nie blokuje.
STALE_TAB_GUARD = """
  const cfg = window.gradio_config || {};
  const mine = cfg.app_id;
  const send = window.fetch.bind(window);
  const idUrl = location.origin + (cfg.api_prefix || '/gradio_api') + '/app_id';
  const serverId = () => send(idUrl, {cache: 'no-store'})
    .then(r => (r.ok ? r.json().then(j => j.app_id) : mine))
    .catch(() => null);  // serwer nie odpowiada - zdarzenie i tak nie przejdzie
  let stale = false;
  if (mine !== undefined && mine !== null) serverId().then(first => {
    if (first !== mine) return;  // inne juz przy starcie - bez straznika (zadnej petli)
    window.fetch = async (input, init) => {
      const url = String(input instanceof Request ? input.url : input);
      if (url.includes('/queue/join') || url.includes('/run/')) {
        const now = stale ? null : await serverId();
        if (stale || (now !== null && now !== mine)) {
          if (!stale) { stale = true; location.reload(); }
          return new Promise(() => {});  // zdarzenie ze starej karty nie wychodzi
        }
      }
      return send(input, init);
    };
  });
"""
# Always dark: Gradio follows the system theme and toggles "dark" on body (also later, when the
# system theme changes), so the class goes back at once. It is added only when missing - setting
# a class attribute always notifies the observer, which would then wake itself up forever.
# Menus: a page button switches the page by clicking the hidden tab button (instantly in the browser;
# Tab.select works as with a normal click). Tools is a disclosure button with its panel (#topbar-tools,
# a Gradio column): a click elsewhere, Escape, focus moving past it or a chosen page close it; Escape
# and a chosen page give the focus back to the button. The top bar's height goes to --topbar-height.
SHELL = """
  const dark = () => [document.documentElement, document.body].forEach(node => {
    if (!node.classList.contains('dark')) node.classList.add('dark');
  });
  dark();
  new MutationObserver(dark).observe(document.body, {attributes: true, attributeFilter: ['class']});
  const toolsToggle = () => document.querySelector('#topbar [data-tools-toggle]');
  const toolsPanel = () => document.getElementById('topbar-tools');
  const toolsOpen = () => Boolean(toolsPanel()?.hasAttribute('data-open'));
  const syncNavigation = () => {
    const active = document.querySelector('#pages > .tab-wrapper [role=tab][aria-selected="true"]');
    const page = active?.dataset.tabId;
    if (!page) return;
    document.querySelectorAll('.workspace-nav [data-page]').forEach(button => {
      if (button.dataset.page === page) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
    const tool = Array.from(toolsPanel()?.querySelectorAll('[data-page]') || [])
      .some(button => button.dataset.page === page);
    toolsToggle()?.toggleAttribute('data-current', tool);
  };
  // the panel opens below its button, inside the window (the bar is its positioned parent)
  const placeTools = () => {
    const toggle = toolsToggle(), panel = toolsPanel(), bar = panel?.offsetParent;
    if (!toggle || !bar) return;
    const box = toggle.getBoundingClientRect(), frame = bar.getBoundingClientRect();
    const left = Math.min(box.left - frame.left, frame.width - panel.offsetWidth - 8);
    panel.style.setProperty('--tools-left', Math.max(8, Math.round(left)) + 'px');
    panel.style.setProperty('--tools-top', Math.round(box.bottom - frame.top + 6) + 'px');
  };
  const setTools = (open, focusToggle = false) => {
    const toggle = toolsToggle(), panel = toolsPanel();
    if (!toggle || !panel) return;
    panel.toggleAttribute('data-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    if (open) placeTools();
    if (focusToggle) toggle.focus({preventScroll: true});
  };
  // Opening another page discards the character being made on the Characters page (app/ui_characters.py):
  // while its creator holds unsaved photos or a voice, the page asks first, and so does closing the tab.
  const LEAVE_CHARACTER = 'Leave this page? The character you are making is not saved and will be discarded. '
    + 'Photos or a voice still being made are stopped.';
  const unsavedCharacter = () => Boolean(document.querySelector('#char-creator [data-unsaved]'));
  const leavesCharacter = tab => tab.getAttribute('aria-selected') !== 'true' && unsavedCharacter()
    && document.querySelector('#pages > .tab-wrapper [role=tab][aria-selected="true"]')
      ?.dataset.tabId === 'characters';
  window.addEventListener('beforeunload', event => {
    if (unsavedCharacter() || document.querySelector('#item-creator [data-unsaved]')) event.preventDefault();
  });
  document.addEventListener('click', event => {
    if (event.target.closest('#topbar [data-tools-toggle]')) {
      setTools(!toolsOpen());
      return;
    }
    const link = event.target.closest('.workspace-nav button[data-page]');
    if (!link) return;
    const page = link.dataset.page;
    const tab = document.querySelector(`#pages > .tab-wrapper [role=tab][data-tab-id="${page}"]`);
    if (!tab) return;
    if (leavesCharacter(tab) && !confirm(LEAVE_CHARACTER)) return;
    if (tab.getAttribute('aria-selected') !== 'true') tab.click();
    requestAnimationFrame(syncNavigation);
    requestAnimationFrame(() => window.scrollTo({top: 0, left: 0, behavior: 'instant'}));
    if (toolsPanel()?.contains(link)) setTools(false, true);
  });
  const outside = target => !toolsPanel()?.contains(target) && !toolsToggle()?.contains(target);
  document.addEventListener('pointerdown', event => {
    if (toolsOpen() && outside(event.target)) setTools(false);
  }, true);
  document.addEventListener('focusin', event => {
    if (toolsOpen() && outside(event.target)) setTools(false);
  });
  // Escape: an open list (the character dropdown's) closes first, then the panel
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape' || !toolsOpen()) return;
    if (event.target.closest?.('[aria-expanded="true"]:not([data-tools-toggle])')) return;
    setTools(false, !outside(document.activeElement));
  }, true);
  window.addEventListener('resize', () => {
    if (toolsOpen()) placeTools();
  });
  const watchBar = (tries = 120) => {
    const bar = document.getElementById('topbar');
    if (!bar) {
      if (tries) requestAnimationFrame(() => watchBar(tries - 1));
      return;
    }
    new ResizeObserver(() => document.documentElement.style.setProperty(
      '--topbar-height', bar.offsetHeight + 'px')).observe(bar);
  };
  watchBar();
  const tabs = document.querySelector('#pages > .tab-wrapper');
  if (tabs) new MutationObserver(syncNavigation).observe(tabs, {
    subtree: true, attributes: true, attributeFilter: ['aria-selected'],
  });
"""
JS = """() => {
""" + SHELL + """
  const sel = '#hist-gallery .thumbnail-lg';
  const clear = () => document.querySelectorAll(sel + '.hist-sel')
    .forEach(t => t.classList.remove('hist-sel'));
  document.addEventListener('click', (e) => {
    const t = e.target.closest(sel);
    if (t) {
      const multi = document.querySelector('#hist-multi input');
      if (multi && multi.checked) { t.classList.toggle('hist-sel'); }
      else { clear(); t.classList.add('hist-sel'); }
      return;
    }
    if (e.target.closest('#hist-none, #hist-refresh, #hist-filter, #hist-multi')) setTimeout(clear, 50);
    if (e.target.closest('#hist-all')) setTimeout(() => document.querySelectorAll(sel)
      .forEach(x => x.classList.add('hist-sel')), 400);
    if (e.target.closest('#hist-del')) setTimeout(() => {
      if (document.body.innerText.includes('Moved to Recycle Bin:')) clear(); }, 2500);
  }, true);
""" + STALE_TAB_GUARD + "}"


def workspace_nav(items: Sequence[tuple[str, str]], label: str, active: str = "") -> str:
    """A menu of page buttons (title, page id); the page script switches to a button's data-page."""
    buttons = []
    for title, page in items:
        current = ' aria-current="page"' if page == active else ""
        icon = _svg_url(ICONS[page])
        buttons.append(f'<button type="button" data-page="{esc(page)}"{current}>'
                       f"<span class='nav-icon' aria-hidden='true' style='--icon:url(\"{icon}\")'></span>"
                       f'{esc(title)}</button>')
    return f'<nav class="workspace-nav" aria-label="{esc(label)}">{"".join(buttons)}</nav>'


def tools_toggle(panel: str) -> str:
    """The Tools button of the top bar: it opens and closes the panel with the id `panel` (page script)."""
    return (f'<button type="button" class="tools-toggle" data-tools-toggle aria-expanded="false" '
            f'aria-controls="{esc(panel)}">Tools<span class="tools-chevron" aria-hidden="true"></span></button>')


def topbar_start(items: Sequence[tuple[str, str]], active: str, tools_panel: str) -> str:
    """The left of the top bar: the brand, the workspace menu and the Tools button."""
    return f'<div class="topbar-start">{BRAND}{workspace_nav(items, "Workspace", active)}{tools_toggle(tools_panel)}</div>'


# ----------------------------- elementy stron -----------------------------

def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def head_html(title: str, subtitle: str = "") -> str:
    """Naglowek strony. subtitle to gotowy HTML (tresc od uzytkownika - przez esc())."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    return f"<div class='page-head'><h1>{esc(title)}</h1>{sub}</div>"


def page_head(title: str, subtitle: str = "") -> gr.HTML:
    return gr.HTML(head_html(title, subtitle), padding=False)


def section(title: str, subtitle: str = "") -> gr.HTML:
    sub = f"<p class='section-sub'>{subtitle}</p>" if subtitle else ""
    return gr.HTML(f"<div><h3 class='section-title'>{esc(title)}</h3>{sub}</div>", padding=False)


def badge(text: str, kind: str = "") -> str:
    return f"<span class='badge {kind}'>{esc(text)}</span>"


def bar(fraction: float) -> str:
    pct = max(0, min(100, int(round(fraction * 100))))
    return f"<div class='bar'><div style='width:{max(pct, 2)}%'></div></div>"


def table(headers: Iterable[str], rows: List[List[str]], empty: str = "") -> str:
    """Tabela HTML. Komorki to gotowy HTML (tekst od uzytkownika - przez esc())."""
    if not rows:
        return f"<div class='empty'>{empty}</div>" if empty else ""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f"<div class='tbl-wrap'><table class='tbl'><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def kpis(cards: List[dict]) -> str:
    """Rzad wskaznikow: label, value, opcjonalnie sub, small (dluzszy tekst), progress (0-1)."""
    out = []
    for c in cards:
        value_cls = "k-value small" if c.get("small") else "k-value"
        sub = f"<div class='k-sub'>{c['sub']}</div>" if c.get("sub") else ""
        prog = bar(c["progress"]) if c.get("progress") is not None else ""
        out.append(f"<div class='kpi'><div class='k-label'>{esc(c['label'])}</div>"
                   f"<div class='{value_cls}'>{c['value']}</div>{sub}{prog}</div>")
    return f"<div class='kpis'>{''.join(out)}</div>"


# Messages under buttons: a leading mark (from the app's modules) -> a status mark (.st-*)
_MARKS = (("✅", "ok"), ("➕", "ok"), ("🗑️", "ok"), ("🗑", "ok"),
          ("❌", "err"), ("⛔", "err"), ("⚠️", "warn"), ("⚠", "warn"), ("⏹", "info"))
# The same marks anywhere in a text (with the variation selector that asks for the color form): Windows
# draws them with its color emoji font (⛔ is a red disc), so a message keeps only its words.
_EMOJI = re.compile("(?:" + "|".join(sorted({re.escape(mark.rstrip("️")) for mark, _ in _MARKS},
                                             key=len, reverse=True)) + ")️? ?|️")


def plain(text) -> str:
    """The text without the status emoji the app's modules write into messages (job errors, refusals),
    wherever they stand. For text shown outside note(): tables, issue lists, the Timeline's lines."""
    if text is None:
        return ""
    return _EMOJI.sub("", str(text)).strip()


def note(text: Optional[str]) -> str:
    """Status message (Markdown) in a calm form: a leading ✅/❌/⚠… becomes a status mark whose
    shape tells ok, warning, error and info apart; the marks inside the text are left out
    (plain). Without a mark the text stays as it is."""
    if not text:
        return text or ""
    body = text.lstrip()
    for mark, kind in _MARKS:
        if body.startswith(mark):
            return f"<span class='st st-{kind}'></span>{plain(body[len(mark):].lstrip())}"
    return plain(text) if _EMOJI.search(text) else text


# ----------------------------- CSS for Gradio -----------------------------
# Gradio copies every custom rule with a ".gradio-container… .contain" prefix and splits selectors at EVERY comma,
# also inside :is(…). "#a :is(.b, .c)" then becomes ":is(.b, <prefix> .c)", whose specificity jumps by the prefix,
# so a narrower override (for example [data-narrow]) silently loses. flatten_selectors() rewrites such lists into
# plain selector lists before the CSS reaches Gradio; the cascade stays exactly what the CSS says.
_LISTS = re.compile(r":(is|where|has|not)\(")


def _top_split(text: str) -> list[str]:
    """`text` split at commas that are not inside (), [] or quotes."""
    parts, depth, quote, start = [], 0, "", 0
    for index, char in enumerate(text):
        if quote:
            quote = "" if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _closing(text: str, start: int) -> int:
    """Index of the ')' that closes the '(' just before `start`."""
    depth, quote = 1, ""
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            quote = "" if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"Unbalanced selector: {text}")


def _expand_selector(selector: str) -> list[str]:
    """One complex selector without comma lists inside :is/:where/:has; :not(a, b) becomes :not(a):not(b)."""
    for match in _LISTS.finditer(selector):
        stop = _closing(selector, match.end())
        arguments = _top_split(selector[match.end():stop])
        if len(arguments) < 2:
            continue
        head, tail, kind = selector[:match.start()], selector[stop + 1:], match.group(1)
        if kind == "not":
            return _expand_selector(head + "".join(f":not({a.strip()})" for a in arguments) + tail)
        return [variant for a in arguments for variant in _expand_selector(f"{head}:{kind}({a.strip()}){tail}")]
    return [selector]


def flatten_selectors(css: str) -> str:
    """The same style sheet with every selector list inside :is/:where/:has/:not written out (see above)."""
    out, index, stack = [], 0, []
    while index < len(css):
        if css.startswith("/*", index):
            stop = css.find("*/", index + 2)
            stop = len(css) if stop < 0 else stop + 2
            out.append(css[index:stop])
            index = stop
            continue
        if stack and stack[-1] == "rule":  # declarations: copied until the block closes
            stop = css.find("}", index)
            stop = len(css) if stop < 0 else stop
            out.append(css[index:stop + 1])
            stack.pop()
            index = stop + 1
            continue
        if css[index] == "}":
            out.append("}")
            if stack:
                stack.pop()
            index += 1
            continue
        if css[index].isspace():
            out.append(css[index])
            index += 1
            continue
        # a prelude: a selector list or an at-rule, up to its '{' or ';'
        depth, quote, cursor = 0, "", index
        while cursor < len(css):
            char = css[cursor]
            if quote:
                quote = "" if char == quote else quote
            elif char in "\"'":
                quote = char
            elif char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            elif depth == 0 and char in "{;":
                break
            cursor += 1
        prelude = css[index:cursor]
        if prelude.lstrip().startswith("@") or cursor >= len(css) or css[cursor] == ";":
            out.append(css[index:cursor + 1])
            if cursor < len(css) and css[cursor] == "{":
                stack.append("at")
        else:
            selectors = [variant.strip() for part in _top_split(prelude) for variant in _expand_selector(part.strip())]
            out.append(",\n".join(selectors) + " {")
            stack.append("rule")
        index = cursor + 1
    return "".join(out)


# The theme and the page script; the stylesheet (this CSS with the Timeline's, flattened) is built in app/main.py.
STYLE = {"theme": THEME, "js": JS}
