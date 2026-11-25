# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Timeline editor workspace: viewer, track, bin, background media work, the inspector's editors.

The page works like a small editing program. The viewer (a static gr.HTML whose script owns real
<video> elements) plays and scrubs on the timeline clock without asking the server: the track's
HTML carries a browser URL for every scene clip that exists and for the program (the joined edit or
the export), and the scripts talk through DOM events on #timeline-page:

    timeline:data    track -> viewer   scenes, media URLs, program, total (after every redraw)
    timeline:query   viewer -> track   ask for timeline:data (the viewer may load later)
    timeline:seek    track -> viewer   {time, scrub}: playhead dragged or a scene chosen
    timeline:scrub   track -> viewer   a drag started: stop playback
    timeline:mode    track -> viewer   {mode}: the V2 block shows the program
    timeline:time    viewer -> track   {time, playing}: the playhead follows playback
    timeline:zoom    zoom bar -> track {action: in | out | fit} or {level} or {query}
    timeline:zoomed  track -> zoom bar {level, pps, span, min, max}: the scale and the seconds in view

The track's zoom is a time scale in pixels per second (2-600), independent of the window: Ctrl+wheel
or a pinch anywhere over the timeline panel zooms the timeline, never the page (around the pointer over
the track), the zoom bar and the keys zoom around the playhead, Fit fits the whole project once. The
canvas always runs one view's width past the project's end, the wheel scrolls along it, and the view
(scale and left edge) is kept per project across refreshes, selection changes, the track's own edits
and resizes.

Gradio redraws a gr.HTML by patching its DOM in place (attributes the new markup lacks are removed,
children it lacks too) and runs watch('value') callbacks right after that, before the browser
paints. So the scripts re-apply their own state (columns, ticks, filmstrips) in their watch
callbacks; the track's scale, canvas width and playhead are custom properties of the component's own
element, which a redraw never touches. The viewer's markup never changes after the page is built: the
<video> elements it creates stay.

Slow work never runs in the editor's 3 s refresh: filmstrip frames, the viewer's copies of clips and
movies and waveform peaks are made once per file version by a few worker threads (Background).

The line, effect and reference editors are plain HTML controls. Their scripts write the rows as
JSON into hidden textboxes (#timeline-lines, #timeline-effects, #timeline-references) that the
Python side reads like any other form field. The editors are drawn from those rows, so a row the
user has just added stays on screen while it is still empty.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_for
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Sequence
from urllib.parse import quote

import numpy as np
from gradio.processing_utils import save_bytes_to_cache
from gradio.route_utils import API_PREFIX
from gradio.utils import get_upload_folder
from PIL import Image

from engine.media import audio, video
from engine.media.errors import MediaError

from . import scene_parts, ui_theme

LINES_ID, EFFECTS_ID, REFERENCES_ID = "timeline-lines", "timeline-effects", "timeline-references"
FPS = 30            # the timecode and frame steps; Timeline clips are 30 fps (video_gen.FPS)
FRAMES = 12         # filmstrip thumbnails decoded per clip; the track shows as many as fit
WAVE_RATE = 20      # waveform buckets per second
WAVE_LEVELS = 15    # waveform height steps (0 = silence)
_PEAK_RATE = 8000   # audio is read at this rate for waveforms


class Background:
    """Slow results computed once per key on worker threads, so the editor's refresh never waits:
    get() returns `default` until the job is done and the result from then on (a later refresh
    shows it). Keys hold every input of the result, e.g. a file's path, size and modification time."""

    def __init__(self, name: str, workers: int = 1, size: int = 512):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
        self._jobs: OrderedDict[tuple, Future] = OrderedDict()
        self._lock = Lock()
        self._size = size

    def get(self, key: tuple, compute: Callable, *args, default=None):
        with self._lock:
            job = self._jobs.get(key)
            if job is None:
                job = self._pool.submit(compute, *args)
                self._jobs[key] = job
            self._jobs.move_to_end(key)
            while len(self._jobs) > self._size:
                finished = next((old for old, task in self._jobs.items() if task.done()), None)
                if finished is None:
                    break
                self._jobs.pop(finished)
        if not job.done():
            return default
        try:
            return job.result()
        except Exception:  # noqa: BLE001 - a job that failed shows nothing, as a missing file would
            return default

    def wait(self, timeout: float = 60.0) -> None:
        """Block until the jobs started so far are done (tests)."""
        with self._lock:
            jobs = list(self._jobs.values())
        wait_for(jobs, timeout=timeout)

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()


_FRAMES = Background("timeline-frames", 2)   # filmstrip thumbnails
_MEDIA = Background("timeline-media", 2)     # checked clips, export-mix previews, cached copies
_PEAKS = Background("timeline-peaks", 1)     # waveforms


def forget() -> None:
    """Drop every remembered background result and cached copy (tests)."""
    for pool in (_FRAMES, _MEDIA, _PEAKS):
        pool.clear()
    with _CACHED_LOCK:
        _CACHED.clear()


def wait() -> None:
    """Wait for the background work started so far (tests)."""
    for pool in (_FRAMES, _MEDIA, _PEAKS):
        pool.wait()


def stamp(path: str | Path | None) -> tuple:
    """(path, size, modification time) of a file, () when it is missing."""
    if not path:
        return ()
    try:
        info = Path(path).stat()
        return (str(path), info.st_size, info.st_mtime_ns)
    except OSError:
        return ()


def later(key: tuple, compute: Callable, *args, default=None):
    """compute(*args) on a media worker, once per key (see Background)."""
    return _MEDIA.get(key, compute, *args, default=default)


# ----------------------------- filmstrips, media copies and waveforms -----------------------------

def filmstrip(stamp: tuple, digest: str, seconds: int, count: int = FRAMES) -> tuple[str, ...]:
    """Browser URLs of `count` frames spread over a checked clip; () until they are decoded."""
    if not stamp:
        return ()
    return _FRAMES.get(("strip", stamp, digest, seconds, count), _decode_strip, stamp, digest, seconds, count,
                       default=())


def _file_url(cached: str | Path) -> str:
    return f"{API_PREFIX}/file={quote(Path(cached).as_posix(), safe='/:')}"


def _decode_strip(stamp: tuple, digest: str, seconds: int, count: int = FRAMES) -> tuple[str, ...]:
    """Cache `count` frames of a checked clip, sampled in one pass, without loading a generation model."""
    path = Path(stamp[0])
    try:
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                return ()
        targets = iter(seconds * (index + .5) / count for index in range(count))
        target = next(targets)
        thumbs: list[Image.Image] = []
        frames = video.read_frames(path)
        try:
            for frame in frames:
                if frame.time is None or frame.time < target:
                    continue
                small = frame.to_image()
                small.thumbnail((192, 108), Image.Resampling.LANCZOS)
                while target is not None and frame.time >= target:
                    thumbs.append(small)
                    target = next(targets, None)
                if target is None:
                    break
        finally:
            frames.close()
        if not thumbs:
            return ()
        urls = []
        for index in range(count):
            data = io.BytesIO()
            thumbs[min(index, len(thumbs) - 1)].save(data, format="JPEG", quality=76)
            cached = save_bytes_to_cache(data.getvalue(), "timeline-frame.jpg", get_upload_folder())
            urls.append(_file_url(cached))
        return tuple(urls)
    except (OSError, ValueError, IndexError, MediaError):
        return ()


_CACHED: OrderedDict[tuple, str] = OrderedDict()
_CACHED_LOCK = Lock()


def cached_file(path: str | Path | None) -> str:
    """Gradio's cached copy of a local preview file, copied once per file version.

    Copying hashes the file, so the copy is remembered by the file's path, size and modification
    time. Large files (clips, movies) go through cached_later() so the refresh does not wait."""
    if not path:
        return ""
    file = Path(path)
    try:
        info = file.stat()
    except OSError:
        return ""
    key = (str(file), info.st_size, info.st_mtime_ns)
    with _CACHED_LOCK:
        if key in _CACHED:
            _CACHED.move_to_end(key)
            return _CACHED[key]
    cached = ui_theme.cache_media(file) or ""
    if cached:
        with _CACHED_LOCK:
            _CACHED[key] = cached
            while len(_CACHED) > 256:
                _CACHED.popitem(last=False)
    return cached


def media_url(path: str | Path | None) -> str:
    """A browser URL for a local image or sound used inside the editor's HTML."""
    cached = cached_file(path)
    return _file_url(cached) if cached else ""


def cached_later(path: str | Path | None, pending: str | None = "") -> str | None:
    """cached_file() made on a media worker: `pending` until the copy is ready, "" when there is no
    copy (the file is missing or could not be copied)."""
    found = stamp(path)
    return later(("copy", found), cached_file, path, default=pending) if found else ""


def url_for(cached: str) -> str:
    return _file_url(cached) if cached else ""


def peaks(path: str | Path | None, offset: float = 0.0, seconds: float = 0.0) -> tuple[int, ...]:
    """The waveform of `seconds` of a sound file from `offset`: WAVE_RATE values per second from 0
    (silence) to WAVE_LEVELS, each the loudest sample of its bucket. () until it is read."""
    found = stamp(path)
    if not found or seconds <= 0:
        return ()
    return _PEAKS.get(("peaks", found, round(float(offset), 3), round(float(seconds), 3)), _decode_peaks,
                      Path(found[0]), float(offset), float(seconds), default=())


def _decode_peaks(path: Path, offset: float, seconds: float) -> tuple[int, ...]:
    try:
        samples, rate = audio.read(path, rate=_PEAK_RATE, channels=1, seconds=offset + seconds)
    except (MediaError, OSError, ValueError):
        return ()
    mono = np.abs(np.asarray(samples, dtype=np.float32).reshape(-1))
    step = max(1, rate // WAVE_RATE)
    count = max(1, int(math.ceil(seconds * WAVE_RATE - 1e-6)))
    start = int(round(offset * rate))
    window = mono[start:start + count * step]
    if window.size < count * step:
        window = np.pad(window, (0, count * step - window.size))
    buckets = window.reshape(count, step).max(axis=1)
    top = float(buckets.max()) if buckets.size else 0.0
    if top < 1e-4:
        return (0,) * count
    levels = np.rint(np.sqrt(np.clip(buckets / top, 0.0, 1.0)) * WAVE_LEVELS).astype(int)
    return tuple(int(level) for level in levels)


def wave_slice(levels: Sequence[int], start: float, end: float) -> tuple[int, ...]:
    """The part of a scene's waveform between two times on the scene clock."""
    if not levels or end <= start:
        return ()
    first = max(0, int(math.floor(start * WAVE_RATE)))
    last = min(len(levels), max(first + 1, int(math.ceil(end * WAVE_RATE))))
    return tuple(levels[first:last])


def wave_svg(levels: Sequence[int]) -> str:
    """A small mirrored waveform that stretches to its block (sizes are in viewBox units)."""
    if not levels or not any(levels):
        return ""
    count = len(levels)
    top = " ".join(f"{2 * index + 1},{16 - level}" for index, level in enumerate(levels))
    bottom = " ".join(f"{2 * index + 1},{16 + level}" for index, level in reversed(list(enumerate(levels))))
    return (f'<svg class="tl-wave" viewBox="0 0 {2 * count} 32" preserveAspectRatio="none" aria-hidden="true" '
            f'focusable="false"><polygon points="0,16 {top} {2 * count},16 {bottom}"/></svg>')


# ----------------------------- small text helpers -----------------------------

def clock(seconds: int | float) -> str:
    whole = max(0, int(seconds))
    minutes, second = divmod(whole, 60)
    return f"{minutes:02d}:{second:02d}"


def timecode(seconds: float) -> str:
    """HH:MM:SS:FF at FPS frames per second, as the viewer shows it."""
    frames = max(0, int(math.floor(float(seconds) * FPS + 1e-6)))
    whole, frame = divmod(frames, FPS)
    return f"{whole // 3600:02d}:{whole // 60 % 60:02d}:{whole % 60:02d}:{frame:02d}"


def scene_action(prompt: str) -> str:
    """Prefer the first action over production notes and dialogue in the clip label."""
    line = " ".join(prompt.split())
    action = re.split(r'[„“"«]', line, maxsplit=1)[0].strip() or line
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", action, maxsplit=2)
    return " ".join(parts[:2])


def first_words(text: str, limit: int = 42) -> str:
    """The start of a text for a narrow label, cut at a word with an ellipsis."""
    line = " ".join(text.split())
    if len(line) <= limit:
        return line
    cut = line[:limit].rsplit(" ", 1)[0] or line[:limit]
    return cut.rstrip(",.;:–-") + "…"


# ----------------------------- icons (stroke and fill follow the text color) -----------------------------

_ICONS = {
    "play": '<path d="M8 5.5v13l10.5-6.5z" fill="currentColor" stroke="none"/>',
    "pause": '<path d="M8 5.5h3v13H8zM13 5.5h3v13h-3z" fill="currentColor" stroke="none"/>',
    "scene-back": '<path d="M6 5.5v13"/><path d="M18 6 9 12l9 6z"/>',
    "scene-forward": '<path d="M18 5.5v13"/><path d="M6 6l9 6-9 6z"/>',
    "frame-back": '<path d="m14.5 6-6 6 6 6"/>',
    "frame-forward": '<path d="m9.5 6 6 6-6 6"/>',
    "full": '<path d="M4.5 9V4.5H9M15 4.5h4.5V9M19.5 15v4.5H15M9 19.5H4.5V15"/>',
    "video": '<rect x="3.5" y="5.5" width="17" height="13" rx="2"/><path d="M7.5 5.5v13M16.5 5.5v13"/>',
    "dialogue": '<path d="M5 6.5h14v9H11l-4 3v-3H5z"/>',
    "sound": '<path d="M4 12h2M8 8v8M12 5v14M16 9v6M20 11v2"/>',
    "edit": '<path d="m12 4 8 4-8 4-8-4z"/><path d="m4 12 8 4 8-4M4 16l8 4 8-4"/>',
}


def icon(name: str) -> str:
    return ('<svg class="tl-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">{_ICONS[name]}</svg>')


# ----------------------------- static markup: viewer, zoom bar, app bar tools -----------------------------

def viewer_html() -> str:
    """The viewer. It never changes after the page is built: its script adds the <video> elements."""
    steps = (("scene-back", "Previous scene"), ("frame-back", "Back one frame"), ("play", "Play"),
             ("frame-forward", "Forward one frame"), ("scene-forward", "Next scene"))
    buttons = "".join(f'<button type="button" class="tl-tool" data-step="{name}" aria-label="{label}" '
                      f'title="{label}">{icon(name)}{icon("pause") if name == "play" else ""}</button>'
                      for name, label in steps)
    return (
        '<div class="tl-viewer" data-viewer>'
        '<div class="tl-stage" data-stage tabindex="0" role="group" aria-label="Viewer. Space plays or pauses, '
        'arrow keys step through frames.">'
        '<div class="tl-screen" data-screen></div>'
        '<div class="tl-slate" data-slate data-active><div class="tl-slate-card">'
        '<span class="tl-slate-kicker" data-slate-kicker>Viewer</span>'
        '<p class="tl-slate-text" data-slate-text>Open or start a project to see its scenes here.</p>'
        '<span class="tl-slate-note" data-slate-note></span></div></div>'
        '<p class="tl-caption" data-caption></p></div>'
        '<div class="tl-transport" role="group" aria-label="Transport">'
        '<div class="tl-modes" role="group" aria-label="Viewer source">'
        '<button type="button" data-mode="scene" aria-pressed="true" title="Scene clips on the timeline clock">'
        'Scene</button>'
        '<button type="button" data-mode="program" aria-pressed="false" aria-disabled="true" '
        'aria-describedby="timeline-program-hint">Program</button></div>'
        f'<div class="tl-steps">{buttons}</div>'
        '<p class="tl-timecode"><span class="tl-sr">Playhead time </span><span data-now>00:00:00:00</span>'
        '<span class="tl-timecode-total"><span aria-hidden="true">/ </span><span class="tl-sr">of </span>'
        '<span data-total>00:00:00:00</span></span></p>'
        f'<button type="button" class="tl-tool" data-fullscreen aria-label="Full screen" aria-pressed="false" '
        f'title="Full screen">{icon("full")}</button>'
        '<span class="tl-sr" id="timeline-program-hint" data-program-hint></span></div></div>'
    )


def zoom_html() -> str:
    """The timeline's zoom controls (static; the track's script does the zooming). The slider is the time
    scale (logarithmic); Fit is an action that fits the whole project once, not a mode."""
    return ('<div class="timeline-deck tl-zoombar"><div class="timeline-scale" role="group" aria-label="Zoom">'
            '<button type="button" data-zoom="out" aria-label="Zoom out" title="Zoom out (−)">−</button>'
            '<input type="range" min="0" max="100" step="1" value="50" data-zoom-level aria-label="Zoom">'
            '<button type="button" data-zoom="in" aria-label="Zoom in" title="Zoom in (+)">+</button>'
            '<button type="button" data-zoom="fit" aria-label="Zoom to fit" aria-keyshortcuts="Shift+Z" '
            'title="Zoom to fit the whole project (Shift+Z)">Fit</button></div></div>')


def shell_html() -> str:
    """App bar tools whose state lives in the browser (static), and the page's heading for assistive
    technology (the workspace has no visible title)."""
    return ('<div class="tl-shell"><h1 class="tl-sr">Timeline</h1>'
            '<button type="button" class="tl-chip-button" data-bin-toggle aria-pressed="true" '
            'aria-controls="timeline-bin" title="Show or hide the bin: scenes, cast and assets">Bin</button></div>')


# ----------------------------- bin, cast and assets -----------------------------

@dataclass(frozen=True)
class BinScene:
    """A scene in the bin's storyboard list."""
    index: int
    start: int
    end: int
    status: str
    text: str
    thumb: str = ""
    selected: bool = False
    alert: str = ""


def bin_html(items: Sequence[BinScene]) -> str:
    cards = []
    for item in items:
        number = item.index + 1
        words = first_words(item.text, 70)
        thumb = (f'<img src="{ui_theme.esc(item.thumb)}" alt="" loading="lazy" decoding="async" draggable="false">'
                 if item.thumb else f'<span class="tl-bin-slate">{ui_theme.esc(first_words(item.text, 34)) or "—"}</span>')
        label = (f"Scene {number}, {clock(item.start)} to {clock(item.end)}, {item.status}. "
                 + (item.text or "No description yet.") + (f" Problem: {item.alert}" if item.alert else ""))
        alert = '<span class="tl-bin-alert" aria-hidden="true">!</span>' if item.alert else ""
        cards.append(
            f'<li><button type="button" class="tl-bin-scene" data-scene="{item.index}" '
            f'data-status="{ui_theme.esc(item.status)}" aria-pressed="{"true" if item.selected else "false"}" '
            f'aria-label="{ui_theme.esc(label)}" title="{ui_theme.esc(label)}">'
            f'<span class="tl-bin-thumb" aria-hidden="true">{thumb}</span>'
            f'<span class="tl-bin-meta"><span class="tl-bin-row"><strong>{number:02d}</strong>{alert}'
            f'<span class="tl-bin-duration">{item.end - item.start} s</span></span>'
            f'<span class="tl-bin-text">{ui_theme.esc(words) or "No description yet"}</span>'
            f'<span class="tl-bin-status"><span class="timeline-mark" data-status="{ui_theme.esc(item.status)}">'
            f'</span>{ui_theme.esc(item.status)}</span></span></button></li>')
    return (f'<div class="tl-bin-scenes"><ol class="tl-bin-list" aria-label="Scenes">{"".join(cards)}</ol>'
            '<p class="tl-hint">Click a scene to edit it; ↑ and ↓ move between scenes.</p></div>')


@dataclass(frozen=True)
class CastCard:
    name: str
    portrait: str = ""   # image URL
    voice: str = ""      # "Voice: Hanna", "Default woman's voice"…


def cast_html(people: Sequence[CastCard], *, message: str = "",
              empty: str = "No characters in this project yet. Choose a universe or add characters above.",
              label: str = "Project cast") -> str:
    """Cards with a portrait, a name and one line under it: the bin's Cast tab, and (empty, label) its Items tab."""
    note = f'<p class="tl-hint">{ui_theme.esc(message)}</p>' if message else ""
    if not people:
        return f'<div class="tl-cast-panel"><p class="tl-empty">{ui_theme.esc(empty)}</p>{note}</div>'
    cards = []
    for person in people:
        initials = "".join(part[:1] for part in person.name.split()[:2]).upper() or "?"
        portrait = (f'<img src="{ui_theme.esc(person.portrait)}" alt="" loading="lazy" decoding="async">'
                    if person.portrait else f"<span>{ui_theme.esc(initials)}</span>")
        cards.append(f'<li class="tl-cast-card"><span class="tl-cast-portrait" aria-hidden="true">{portrait}</span>'
                     f'<span class="tl-cast-meta"><strong>{ui_theme.esc(person.name)}</strong>'
                     f'<small>{ui_theme.esc(person.voice)}</small></span></li>')
    return (f'<div class="tl-cast-panel"><ul class="tl-cast" aria-label="{ui_theme.esc(label)}">{"".join(cards)}</ul>'
            f'{note}</div>')


@dataclass(frozen=True)
class AssetItem:
    name: str            # stored reference name (sha256)
    url: str             # image URL, "" when the file is missing
    label: str           # "[1]" in this scene, "Scenes 2, 4" elsewhere
    usable: bool = False  # can be added to the selected scene


def assets_html(scene: int, here: Sequence[AssetItem], elsewhere: Sequence[AssetItem]) -> str:
    """The selected scene's reference images and the project's other ones (with Use)."""
    def figure(item: AssetItem, use: bool) -> str:
        thumb = (f'<img src="{ui_theme.esc(item.url)}" alt="" loading="lazy" decoding="async">' if item.url
                 else "<span>Missing file</span>")
        button = ""
        if use:
            disabled = "" if item.usable else " disabled"
            button = (f'<button type="button" data-use-reference="{ui_theme.esc(item.name)}"{disabled} '
                      f'aria-label="Use this image ({ui_theme.esc(item.label)}) in scene {scene}">Use</button>')
        return (f'<figure class="tl-asset" role="listitem"><span class="tl-asset-thumb">{thumb}</span>'
                f'<figcaption>{ui_theme.esc(item.label)}</figcaption>{button}</figure>')

    mine = "".join(figure(item, False) for item in here)
    other = "".join(figure(item, True) for item in elsewhere)
    return ('<div class="tl-assets">'
            f'<p class="tl-section-label">Scene {scene}</p>'
            + (f'<div class="tl-asset-grid" role="list" aria-label="Reference images in scene {scene}">{mine}</div>'
               if mine else '<p class="tl-empty">No reference images in this scene.</p>')
            + '<p class="tl-section-label">Other scenes</p>'
            + (f'<div class="tl-asset-grid" role="list" aria-label="Reference images in other scenes">{other}</div>'
               if other else '<p class="tl-empty">The other scenes have no reference images.</p>')
            + f'<p class="tl-hint">Up to {scene_parts.MAX_REFERENCES} images per scene; the picture text refers to '
              'them as [1], [2]…</p></div>')


# ----------------------------- scene inspector editors -----------------------------

@dataclass(frozen=True)
class LineRow:
    """One row of the dialogue editor as the user typed it."""
    speaker: str = ""
    text: str = ""
    start: str = ""          # seconds as typed; "" places the line automatically
    window: str = ""         # "0.3–1.9 s"
    kind: str = ""           # "estimated", "measured" or "past the end"
    past_end: bool = False
    note: str = ""


@dataclass(frozen=True)
class EffectRow:
    """One row of the effect editor as the user typed it."""
    text: str = ""
    at: str = ""
    seconds: str = ""        # "" uses scene_parts.EFFECT_SECONDS
    window: str = ""
    past_end: bool = False
    note: str = ""


Options = Sequence[tuple[str, Sequence[tuple[str, str]]]]  # (group label or "", ((label, value), ...))


def _speaker_select(options: Options, value: str, extra: str, number: int) -> str:
    known = {item for _, items in options for _, item in items}
    parts = []
    for group, items in options:
        rendered = "".join(f'<option value="{ui_theme.esc(item)}"' + (" selected" if item == value else "")
                           + f">{ui_theme.esc(label)}</option>" for label, item in items)
        parts.append(f'<optgroup label="{ui_theme.esc(group)}">{rendered}</optgroup>' if group else rendered)
    if value not in known:
        label = extra or f"{value} (not in this scene)"
        parts.append(f'<option value="{ui_theme.esc(value)}" selected>{ui_theme.esc(label)}</option>')
    return (f'<select class="tl-row-who" data-field="speaker" data-value="{ui_theme.esc(value)}" '
            f'data-aria="speaker" aria-label="Line {number} speaker">{"".join(parts)}</select>')


def _window(text: str, kind: str, past_end: bool, measured: bool = False) -> str:
    return (f'<span class="tl-row-window" data-measured="{str(measured).lower()}" '
            f'data-past-end="{str(past_end).lower()}">{ui_theme.esc(text or "—")}'
            f'<small>{ui_theme.esc(kind)}</small></span>')


def _note(text: str) -> str:
    return f'<p class="tl-row-note"{"" if text else " hidden"}>{ui_theme.esc(text)}</p>'


def _line_row(number: int, row: LineRow, options: Options, extra: str = "", *, prototype: bool = False) -> str:
    flags = " data-prototype hidden" if prototype else ""
    return (f'<li class="tl-row" data-row{flags}>'
            f'<span class="tl-row-number" data-number-label aria-hidden="true">{number}</span>'
            + _speaker_select(options, row.speaker, extra, number)
            + f'<input type="text" class="tl-row-text" data-field="text" data-aria="text" data-focus '
              f'value="{ui_theme.esc(row.text)}" maxlength="{scene_parts.MAX_TEXT}" autocomplete="off" '
              f'placeholder="What is said" aria-label="Line {number} text">'
              f'<input type="number" class="tl-row-time" data-field="start" data-number '
              f'data-aria="start, seconds from the scene start" value="{ui_theme.esc(row.start)}" min="0" '
              f'step="0.1" inputmode="decimal" placeholder="Auto" '
              f'aria-label="Line {number} start, seconds from the scene start">'
            + _window(row.window, row.kind, row.past_end, row.kind == "measured")
            + f'<button type="button" class="tl-row-remove" data-remove aria-label="Remove line {number}">'
              f'Remove</button>{_note(row.note)}</li>')


def lines_editor(rows: Sequence[LineRow], options: Options, key: str, *, extras: Sequence[str] = (),
                 message: str = "") -> str:
    """The dialogue line editor: speaker, text, start and the window each line plays in."""
    limit = scene_parts.MAX_LINES
    body = "".join(_line_row(number, row, options, extras[number - 1] if number <= len(extras) else "")
                   for number, row in enumerate(rows, 1))
    disabled = " disabled" if len(rows) >= limit else ""
    return (f'<div class="tl-rows" data-rows="lines" data-key="{ui_theme.esc(key)}" data-target="{LINES_ID}" '
            f'data-noun="Line" data-max="{limit}">'
            f'<p class="tl-rows-message"{"" if message else " hidden"}>{ui_theme.esc(message)}</p>'
            '<div class="tl-rows-head" aria-hidden="true"><span>#</span><span>Speaker</span><span>Line</span>'
            '<span>Start (s)</span><span>Plays</span><span></span></div>'
            f'<ol class="tl-rows-list" aria-label="Dialogue lines">{body}</ol>'
            f'<ol class="tl-rows-proto" hidden>{_line_row(len(rows) + 1, LineRow(), options, prototype=True)}</ol>'
            f'<p class="tl-rows-empty" data-empty{" hidden" if rows else ""}>No dialogue yet. Add a line, '
            'choose who says it and, if needed, when it starts.</p>'
            f'<div class="tl-rows-foot"><button type="button" class="tl-rows-add" data-add{disabled}>'
            f'+ Add line</button><span class="tl-rows-hint"><span data-count>{len(rows)}</span> of {limit} '
            '· an empty start follows the previous line</span></div></div>')


def _effect_row(number: int, row: EffectRow, *, prototype: bool = False) -> str:
    flags = " data-prototype hidden" if prototype else ""
    low, high = scene_parts.EFFECT_RANGE
    return (f'<li class="tl-row" data-row{flags}>'
            f'<span class="tl-row-number" data-number-label aria-hidden="true">{number}</span>'
            f'<input type="text" class="tl-row-text" data-field="text" data-aria="sound" data-focus '
            f'value="{ui_theme.esc(row.text)}" maxlength="{scene_parts.MAX_TEXT}" autocomplete="off" '
            f'placeholder="Door slams, glass breaks…" aria-label="Effect {number} sound">'
            f'<input type="number" class="tl-row-time" data-field="at" data-number '
            f'data-aria="time, seconds from the scene start" value="{ui_theme.esc(row.at)}" min="0" step="0.1" '
            f'inputmode="decimal" placeholder="0" aria-label="Effect {number} time, seconds from the scene start">'
            f'<input type="number" class="tl-row-length" data-field="seconds" data-number '
            f'data-aria="length in seconds" value="{ui_theme.esc(row.seconds)}" min="{low:g}" max="{high:g}" '
            f'step="0.1" inputmode="decimal" placeholder="{scene_parts.EFFECT_SECONDS:g}" '
            f'aria-label="Effect {number} length in seconds">'
            + _window(row.window, "past the end" if row.past_end else "", row.past_end)
            + f'<button type="button" class="tl-row-remove" data-remove aria-label="Remove effect {number}">'
              f'Remove</button>{_note(row.note)}</li>')


def effects_editor(rows: Sequence[EffectRow], key: str, *, message: str = "") -> str:
    """The sound effect editor: description, time on the scene clock and length."""
    limit = scene_parts.MAX_EFFECTS
    body = "".join(_effect_row(number, row) for number, row in enumerate(rows, 1))
    disabled = " disabled" if len(rows) >= limit else ""
    low, high = scene_parts.EFFECT_RANGE
    return (f'<div class="tl-rows" data-rows="effects" data-key="{ui_theme.esc(key)}" data-target="{EFFECTS_ID}" '
            f'data-noun="Effect" data-max="{limit}">'
            f'<p class="tl-rows-message"{"" if message else " hidden"}>{ui_theme.esc(message)}</p>'
            '<div class="tl-rows-head" aria-hidden="true"><span>#</span><span>Sound effect</span>'
            '<span>At (s)</span><span>Length (s)</span><span>Plays</span><span></span></div>'
            f'<ol class="tl-rows-list" aria-label="Sound effects">{body}</ol>'
            f'<ol class="tl-rows-proto" hidden>{_effect_row(len(rows) + 1, EffectRow(), prototype=True)}</ol>'
            f'<p class="tl-rows-empty" data-empty{" hidden" if rows else ""}>No sound effects. Add one at the '
            'moment it should be heard.</p>'
            f'<div class="tl-rows-foot"><button type="button" class="tl-rows-add" data-add{disabled}>'
            f'+ Add effect</button><span class="tl-rows-hint"><span data-count>{len(rows)}</span> of {limit} '
            f'· {low:g}–{high:g} s each</span></div></div>')


def references_html(items: Sequence[tuple], key: str) -> str:
    """Reference image thumbnails, numbered as the picture text refers to them: [1], [2]…

    items: (stored name, image URL or "" when the file is missing), or with a third part, the name of the place or
    item (app/items.py) whose main photo it is, shown after the number."""
    figures = []
    for number, (name, url, *rest) in enumerate(items, 1):
        label = rest[0] if rest and rest[0] else ""
        thumb = (f'<img src="{ui_theme.esc(url)}" alt="Reference image {number}" loading="lazy" decoding="async">'
                 if url else '<span>Missing file</span>')
        caption = f"[{number}] {ui_theme.esc(label)}" if label else f"[{number}]"
        figures.append(f'<figure class="tl-ref" role="listitem" data-reference="{ui_theme.esc(name)}">'
                       f'<span class="tl-ref-thumb">{thumb}</span><figcaption>{caption}</figcaption>'
                       f'<button type="button" data-remove-reference="{ui_theme.esc(name)}" '
                       f'aria-label="Remove reference image {number}">Remove</button></figure>')
    names = json.dumps([entry[0] for entry in items])
    return (f'<div class="tl-refs" data-references data-key="{ui_theme.esc(key)}" data-names="{ui_theme.esc(names)}" '
            f'data-max="{scene_parts.MAX_REFERENCES}">'
            f'<div class="tl-refs-list" role="list" aria-label="Reference images">{"".join(figures)}</div>'
            f'<p class="tl-refs-empty"{" hidden" if items else ""}>No reference images. Add photos of a place, '
            'an outfit or an object to keep it consistent.</p></div>')


_LEVELS = {scene_parts.ERROR: (0, "Error"), scene_parts.WARNING: (1, "Check"), scene_parts.INFO: (2, "Note")}


def issues_html(issues: Sequence[scene_parts.Issue], *, empty: str = "No problems found.") -> str:
    """The Sound check list. Each level has its own shape: errors a filled square and a bold label,
    warnings a hollow ring, notes a small dot."""
    if not issues:
        return (f'<p class="tl-check-ok"><span class="tl-mark" data-level="ok" aria-hidden="true"></span>'
                f'{ui_theme.esc(empty)}</p>')
    ordered = sorted(issues, key=lambda issue: _LEVELS.get(issue.level, (3, ""))[0])
    items = "".join(f'<li data-level="{ui_theme.esc(issue.level)}"><span class="tl-mark" '
                    f'data-level="{ui_theme.esc(issue.level)}" aria-hidden="true"></span>'
                    f'<strong>{_LEVELS.get(issue.level, (3, "Note"))[1]}</strong>'
                    f'<span>{ui_theme.esc(ui_theme.plain(issue.text))}</span></li>' for issue in ordered)
    return f'<ul class="tl-check" aria-label="Sound check">{items}</ul>'


def players_html(items: Sequence[tuple[str, str]], state: str, *, stale: bool = False) -> str:
    """Audio players for an audition: (label, URL) per line, the background and each effect."""
    players = "".join(f'<div class="tl-player"><span>{ui_theme.esc(label)}</span>'
                      f'<audio controls preload="none" src="{ui_theme.esc(url)}" '
                      f'aria-label="{ui_theme.esc(label)}"></audio></div>' for label, url in items if url)
    flag = " data-stale" if stale else ""
    return (f'<div class="tl-players"><p class="tl-players-state"{flag}>{ui_theme.esc(state)}</p>'
            f'{players}</div>')


# ----------------------------- stylesheet -----------------------------
# Gradio keeps top-level rules as written, but copies a rule inside @media only with the prefix
# ".gradio-container… .contain " and splits its selector list on every comma: rules inside @media
# therefore target #timeline-page descendants and never put a comma inside :is() or :not().

CSS = """
#timeline-page { --tl-deck: #0f0f0f; --tl-well: #0a0a0a; --tl-edge: #2a2a2a; --tl-text: #f2f2f2;
  --tl-muted: #9a9a9a; --tl-accent: #ffffff; --tl-cursor: #ffffff; --tl-clip: #1c1c1c;
  --tl-clip-ready: #262626; --tl-line: #3a3a3a; --tl-sound: #222222; --tl-fill: #141414;
  --tl-hover: #2e2e2e; --tl-rail: 100px; color-scheme: dark; }
#timeline-page .tl-hidden { display: none !important; }
#timeline-page .tl-sr { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
  clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
#timeline-page :is(.tl-viewer, .tl-zoombar, .tl-shell, .tl-status, .tl-bin-scenes, .tl-cast-panel, .tl-assets,
  .tl-scene-head, .timeline-deck) :is(p, ol, ul, li, figure, button) { margin: 0; }

/* ----- workspace: fills the window below its own top (--tl-top, measured by the page script: the top
   bar's height). The app bar, the panels and the timeline sit flush against each other and the window's
   edges: no gaps, no rounded corners, a single 1 px hairline between two neighbours. ----- */
#timeline-page #timeline-workspace { display: flex !important; flex-direction: column !important;
  flex-wrap: nowrap !important; gap: 0 !important; min-width: 0;
  height: calc(100vh - var(--tl-top, 44px)); height: calc(100dvh - var(--tl-top, 44px));
  min-height: 560px; }
#timeline-page #timeline-workspace > * { flex: 0 0 auto !important; min-width: 0 !important; }
#timeline-page #timeline-workspace > .tl-body { flex: 1 1 auto !important; min-height: 0 !important; }

/* ----- app bar ----- */
#timeline-page .tl-appbar { display: flex !important; flex-wrap: wrap !important; align-items: center !important;
  justify-content: space-between !important; gap: 6px 14px !important; padding: 6px 12px !important;
  background: var(--c-surface); border: 0; border-bottom: 1px solid var(--c-border); border-radius: 0; }
#timeline-page .tl-appbar > * { flex: 0 1 auto !important; width: auto !important; min-width: 0 !important; }
#timeline-page :is(.tl-appbar-project, .tl-appbar-production) { display: flex !important; flex-wrap: wrap !important;
  align-items: center !important; gap: 6px !important; }
#timeline-page .tl-appbar-production { justify-content: flex-end !important; margin-left: auto !important; }
#timeline-page :is(.tl-appbar-project, .tl-appbar-production) > * { flex: 0 0 auto !important; width: auto !important;
  min-width: 0 !important; }
#timeline-page .tl-appbar-project > .form { display: flex !important; flex-direction: row !important;
  flex-wrap: nowrap !important; gap: 6px !important; border: 0 !important; background: none !important; }
#timeline-page .tl-appbar-project > .form > * { min-width: 0 !important; }
#timeline-page #timeline-title { width: clamp(118px, 13vw, 200px) !important; flex: 0 1 auto !important; }
#timeline-page #timeline-projects { width: clamp(128px, 14vw, 220px) !important; flex: 0 1 auto !important; }
#timeline-page #timeline-title input { min-height: 32px; padding: 5px 10px; font-size: 14px; font-weight: 600; }
#timeline-page #timeline-projects input { font-size: 13px; }
#timeline-page .tl-appbar button.sm { min-height: 32px; padding: 0 12px; font-size: 13px; }
/* the Subtitles option beside Export movie: one line as tall as the buttons */
#timeline-page #timeline-subtitles { display: flex; align-items: center; min-height: 32px; padding: 0 4px !important; }
#timeline-page #timeline-subtitles label { gap: 7px; font-size: 13px; font-weight: 550; color: var(--c-text-2);
  white-space: nowrap; cursor: pointer; }
#timeline-page .tl-status { display: flex; align-items: center; gap: 10px; min-width: 0; padding: 0 2px;
  font-size: 12px; color: var(--c-text-3); white-space: nowrap; font-variant-numeric: tabular-nums; }
#timeline-page .tl-status-save { display: inline-flex; align-items: center; gap: 6px; color: var(--c-text-2); }
#timeline-page .tl-status-save::before { content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: var(--c-ok); }
#timeline-page .tl-status-save[data-state="dirty"] { color: var(--c-text); font-weight: 650; }
#timeline-page .tl-status-save[data-state="dirty"]::before { background: none; box-shadow: inset 0 0 0 1.5px var(--c-text); }
#timeline-page .tl-status-save[data-state="new"]::before { border-radius: 1px; height: 2px; background: var(--c-text-2); }
#timeline-page .tl-status-format { overflow: hidden; text-overflow: ellipsis; max-width: 30ch; }
#timeline-page #timeline-project-menu { position: relative; padding: 0 !important; border: 0 !important;
  background: none !important; overflow: visible !important; }
#timeline-page #timeline-project-menu .label-wrap { display: inline-flex; align-items: center; gap: 8px; width: auto;
  min-height: 32px; margin: 0; padding: 0 10px; border: 1px solid var(--c-border-2); border-radius: 8px;
  background: var(--c-surface-2); color: var(--c-text); }
#timeline-page #timeline-project-menu .label-wrap span { font-size: 13px; font-weight: 550; }
#timeline-page #timeline-project-menu .label-wrap .icon { font-size: 9px; color: var(--c-text-3); }
#timeline-page #timeline-project-menu [data-testid="accordion-content"] { position: absolute; top: calc(100% + 6px);
  right: 0; z-index: 60; width: min(340px, 86vw); padding: 12px; background: var(--c-surface-2);
  border: 1px solid var(--c-border-2); border-radius: 10px; box-shadow: 0 16px 40px rgba(0, 0, 0, .6); }
#timeline-page #timeline-project-menu [data-testid="accordion-content"] .form { gap: 10px !important; }
#timeline-page .tl-chip-button { min-height: 32px; padding: 0 11px; border: 1px solid var(--c-border-2);
  border-radius: 8px; background: transparent; color: var(--c-text-2); font-family: inherit; font-size: 13px;
  font-weight: 550; line-height: 1; cursor: pointer; }
#timeline-page .tl-chip-button:hover { color: var(--c-text); background: var(--c-surface-2); }
#timeline-page .tl-chip-button[aria-pressed="true"] { color: var(--c-text); background: var(--c-accent-soft); }
#timeline-page .tl-chip-button:focus-visible { outline: 2px solid var(--c-accent); outline-offset: 2px; }
/* the messages between the app bar and the panels: strips across the workspace, a hairline below */
#timeline-page .tl-switch-warning { padding: 8px 12px !important; border: 0 !important;
  border-bottom: 1px solid var(--c-border-2) !important; border-radius: 0 !important;
  background: var(--c-surface-2) !important; }
/* the question before a scene whose dialogue needs more time is made: a pop-up over the workspace (Gradio gives the
   group's id to its inner box too: only the outer one floats) */
#timeline-page #timeline-workspace > #timeline-length-warning { position: fixed !important; z-index: 1000; top: 22vh; left: 50%;
  transform: translateX(-50%); width: min(560px, calc(100vw - 32px)); height: auto !important; flex: none !important;
  max-height: 70vh; overflow: auto !important; padding: 16px 18px !important;
  border: 1px solid var(--c-border-2) !important; border-radius: 12px !important;
  background: var(--c-surface) !important; box-shadow: 0 18px 48px rgba(0, 0, 0, .6); }
#timeline-page #timeline-action-status { padding: 6px 12px !important; border-bottom: 1px solid var(--c-border) !important;
  background: var(--c-surface); }
#timeline-page #timeline-action-status p { margin: 0; font-size: 13px; }
#timeline-page #timeline-action-status:not(:has(p)) { display: none !important; }

/* queue chip */
#timeline-page .tl-queue { position: relative; display: flex; align-items: center; gap: 8px; min-width: 0;
  max-width: min(420px, 40vw); min-height: 32px; padding: 0 10px; border: 1px solid var(--c-border);
  border-radius: 999px; background: var(--c-bg); font-size: 12.5px; color: var(--c-text-2); }
#timeline-page .tl-queue-mark { flex: none; width: 7px; height: 7px; border-radius: 50%; background: var(--c-text-3); }
#timeline-page .tl-queue[data-state="running"] .tl-queue-mark { background: var(--c-accent); }
#timeline-page .tl-queue[data-state="queued"] .tl-queue-mark { background: none; box-shadow: inset 0 0 0 1.5px var(--c-text-2); }
#timeline-page .tl-queue[data-state="failed"] .tl-queue-mark { border-radius: 1px; background: var(--c-err); }
#timeline-page .tl-queue[data-state="failed"] .tl-queue-text { color: var(--c-text); font-weight: 650; }
#timeline-page .tl-queue-text { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-variant-numeric: tabular-nums; }
#timeline-page .tl-queue .bar { flex: 0 0 64px; height: 4px; margin: 0; }
#timeline-page .tl-queue details { flex: none; }
#timeline-page .tl-queue summary { list-style: none; cursor: pointer; padding: 2px 4px; border-radius: 6px;
  color: var(--c-text-3); font-size: 12px; }
#timeline-page .tl-queue summary::-webkit-details-marker { display: none; }
#timeline-page .tl-queue summary:hover { color: var(--c-text); }
#timeline-page .tl-queue summary:focus-visible { outline: 2px solid var(--c-accent); outline-offset: 1px; }
#timeline-page .tl-queue-panel { position: absolute; right: 0; top: calc(100% + 6px); z-index: 60;
  width: min(380px, 86vw); padding: 10px 12px; display: grid; gap: 6px; border: 1px solid var(--c-border-2);
  border-radius: 10px; background: var(--c-surface-2); box-shadow: 0 16px 40px rgba(0, 0, 0, .6);
  white-space: normal; }
#timeline-page .tl-queue-panel .j-meta { margin: 0; font-size: 12.5px; color: var(--c-text-2); }
#timeline-page .tl-queue-panel ul { margin: 0; padding-left: 18px; font-size: 12px; color: var(--c-text-2); }

/* ----- body: bin | viewer | inspector over a full-width timeline ----- */
#timeline-page .tl-body { display: grid !important; gap: 0 !important; flex-wrap: nowrap !important;
  grid-template-columns: minmax(196px, 236px) minmax(0, 1fr) minmax(320px, 384px);
  /* the timeline's floor holds its toolbar, every lane (V2 too, see .timeline-deck), the sideways scroll
     bar and the hint line */
  grid-template-rows: minmax(0, 1fr) clamp(250px, 40%, 460px);
  grid-template-areas: "bin viewer inspector" "timeline timeline timeline";
  /* Gradio rows align their children to the top; every panel must fill its grid cell instead */
  align-items: stretch !important; }
#timeline-page .tl-body > * { min-width: 0 !important; min-height: 0 !important; width: auto !important;
  max-width: none !important; height: auto !important; align-self: stretch !important; }
#timeline-page #timeline-workspace:not([data-size="narrow"]) .tl-body > * { max-height: 100% !important; }
#timeline-page .tl-bin { grid-area: bin; }
#timeline-page .tl-viewer-panel { grid-area: viewer; }
#timeline-page .tl-inspector { grid-area: inspector; }
#timeline-page .tl-timeline-panel { grid-area: timeline; }
#timeline-page #timeline-workspace[data-bin="closed"] .tl-bin { display: none !important; }
#timeline-page #timeline-workspace[data-bin="closed"] .tl-body {
  grid-template-columns: minmax(0, 1fr) minmax(320px, 384px);
  grid-template-areas: "viewer inspector" "timeline timeline"; }
#timeline-page #timeline-workspace[data-size="medium"]:not([data-bin="open"]) .tl-bin { display: none !important; }
#timeline-page #timeline-workspace[data-size="medium"]:not([data-bin="open"]) .tl-body {
  grid-template-columns: minmax(0, 1fr) minmax(300px, 360px);
  grid-template-areas: "viewer inspector" "timeline timeline"; }
#timeline-page #timeline-workspace[data-size="medium"][data-bin="open"] .tl-body {
  grid-template-columns: 200px minmax(0, 1fr) minmax(290px, 340px); }
#timeline-page #timeline-workspace[data-size="narrow"] { height: auto; min-height: 0; }
#timeline-page #timeline-workspace[data-size="narrow"] .tl-body { grid-template-columns: minmax(0, 1fr);
  grid-template-rows: none; grid-auto-rows: auto; grid-template-areas: "viewer" "inspector" "timeline"; }
#timeline-page #timeline-workspace[data-size="narrow"]:not([data-bin="open"]) .tl-bin { display: none !important; }
/* the stack follows the page's tab order: bin, viewer, inspector, timeline */
#timeline-page #timeline-workspace[data-size="narrow"][data-bin="open"] .tl-body {
  grid-template-areas: "bin" "viewer" "inspector" "timeline"; }
#timeline-page #timeline-workspace[data-size="narrow"] .tl-panel { overflow: visible; }
/* !important: these heights must win over "height: auto !important" on every panel of the body */
#timeline-page #timeline-workspace[data-size="narrow"] .tl-viewer-panel { height: min(62vw, 420px) !important; }
#timeline-page #timeline-workspace[data-size="narrow"] .tl-bin { height: 340px !important; overflow: hidden; }
#timeline-page #timeline-workspace[data-size="narrow"] .tl-timeline-panel { height: 300px !important; }
#timeline-page #timeline-workspace[data-size="narrow"] #timeline-inspector-tabs > .tabitem { overflow: visible; }

#timeline-page .tl-panel { display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important;
  gap: 8px !important; min-width: 0 !important; min-height: 0 !important; overflow: hidden; padding: 10px !important;
  background: var(--c-surface) !important; border: 0 !important; border-radius: 0 !important;
  box-shadow: none !important; }
#timeline-page .tl-panel > * { flex: 0 0 auto; min-width: 0 !important; }
/* Single hairlines: a panel draws only the edge it shares with the panel after it - the bin and the viewer
   their right edge, the timeline its top edge - so two borders never double and none lies on the window's
   edge. A stacked (narrow) workspace draws them under each panel instead, see below. */
#timeline-page .tl-bin, #timeline-page .tl-viewer-panel { border-right: 1px solid var(--c-border) !important; }
#timeline-page .tl-timeline-panel { border-top: 1px solid var(--c-border) !important; }
#timeline-page #timeline-workspace[data-size="narrow"] .tl-bin,
#timeline-page #timeline-workspace[data-size="narrow"] .tl-viewer-panel,
#timeline-page #timeline-workspace[data-size="narrow"] .tl-inspector {
  border-right: 0 !important; border-bottom: 1px solid var(--c-border) !important; }
#timeline-page #timeline-workspace[data-size="narrow"] .tl-timeline-panel { border-top: 0 !important; }
#timeline-page .tl-kicker, #timeline-page .tl-section-label { margin: 0; font: 650 10px/1.4 system-ui, sans-serif;
  letter-spacing: .12em; text-transform: uppercase; color: var(--c-text-3); }
#timeline-page .tl-hint, #timeline-page .timeline-hint, #timeline-page .timeline-hint p { margin: 0; font-size: 12px;
  color: var(--c-text-3); }
#timeline-page .tl-empty { margin: 0; font-size: 12.5px; color: var(--c-text-3); }

/* tabs inside panels: a segmented control on top, the content scrolls below it */
#timeline-page :is(#timeline-inspector-tabs, #timeline-bin-tabs) { display: flex !important; flex-direction: column;
  gap: 0; min-height: 0; }
#timeline-page :is(#timeline-inspector-tabs, #timeline-bin-tabs) > .tab-wrapper { flex: none; height: auto;
  padding: 0; margin: 0 0 4px !important; }
#timeline-page :is(#timeline-inspector-tabs, #timeline-bin-tabs) [role="tablist"] { gap: 2px; height: auto;
  padding: 3px; border: 0; border-radius: 9px; background: var(--c-muted); }
#timeline-page :is(#timeline-inspector-tabs, #timeline-bin-tabs) [role="tablist"]::after { display: none; }
#timeline-page :is(#timeline-inspector-tabs, #timeline-bin-tabs) [role="tab"] { flex: 1 1 0; height: 28px;
  min-height: 28px; padding: 0 8px; justify-content: center; border: 0; border-radius: 7px; background: transparent;
  color: var(--c-text-2); font-size: 12.5px; font-weight: 550; }
#timeline-page :is(#timeline-inspector-tabs, #timeline-bin-tabs) [role="tab"]:hover { color: var(--c-text); }
#timeline-page :is(#timeline-inspector-tabs, #timeline-bin-tabs) [role="tab"][aria-selected="true"] {
  background: var(--c-surface-2); color: var(--c-text); box-shadow: inset 0 0 0 1px var(--c-border-2); }
#timeline-page :is(#timeline-inspector-tabs, #timeline-bin-tabs) [role="tab"]::after { display: none; }
/* the bin's four tabs (Scenes, Cast, Items, Assets) on one row of its 220 px */
#timeline-page #timeline-bin-tabs [role="tab"] { padding: 0 5px; }
/* The scrolling tab content keeps 4 px around its controls for their focus rings (2 px, 1-2 px outside):
   it reaches 4 px into the panel's padding, so the controls stay where they were. */
#timeline-page :is(#timeline-inspector-tabs, #timeline-bin-tabs) > .tabitem { flex: 1 1 auto; min-height: 0;
  overflow: hidden auto; margin: 0 0 0 -4px !important; padding: 4px 4px 6px 4px !important; gap: 10px;
  scrollbar-gutter: stable; }

/* ----- bin ----- */
#timeline-page .tl-bin { padding: 10px 6px 8px 12px !important; }
#timeline-page .tl-bin > #timeline-bin-tabs { flex: 1 1 auto !important; }
#timeline-page .tl-bin-scenes { display: grid; gap: 8px; }
#timeline-page .tl-bin-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; }
#timeline-page .tl-bin-scene { display: grid; grid-template-columns: 70px minmax(0, 1fr); align-items: start; gap: 8px;
  width: 100%; padding: 5px; border: 1px solid transparent; border-radius: 8px; background: transparent;
  color: var(--c-text); text-align: left; font: inherit; cursor: pointer; }
#timeline-page .tl-bin-scene:hover { background: var(--c-surface-2); }
#timeline-page .tl-bin-scene[aria-pressed="true"] { background: var(--c-surface-2); border-color: var(--c-accent); }
#timeline-page .tl-bin-scene:focus-visible { outline: 2px solid var(--c-accent); outline-offset: 1px; }
#timeline-page .tl-bin-thumb { display: grid; place-items: center; aspect-ratio: 16 / 9; overflow: hidden;
  border-radius: 5px; background: var(--c-muted); border: 1px dashed var(--c-border-2); }
#timeline-page .tl-bin-scene[data-status="Ready"] .tl-bin-thumb { border-style: solid; border-color: var(--c-border); }
#timeline-page .tl-bin-scene[data-status="Outdated"] .tl-bin-thumb { border-style: dashed; border-color: var(--c-border); }
#timeline-page .tl-bin-thumb img { width: 100%; height: 100%; object-fit: cover; }
#timeline-page .tl-bin-slate { padding: 3px; font-size: 8.5px; line-height: 1.2; color: var(--c-text-3);
  overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; }
#timeline-page .tl-bin-meta { display: grid; gap: 2px; min-width: 0; }
#timeline-page .tl-bin-row { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--c-text-3);
  font-variant-numeric: tabular-nums; }
#timeline-page .tl-bin-row strong { font: 650 11px ui-monospace, Consolas, monospace; color: var(--c-text); }
#timeline-page .tl-bin-duration { margin-left: auto; }
#timeline-page .tl-bin-alert { display: inline-grid; place-items: center; width: 13px; height: 13px;
  background: var(--c-err); color: var(--c-bg); font: 800 10px/1 system-ui, sans-serif; }
#timeline-page .tl-bin-text { font-size: 12px; line-height: 1.3; color: var(--c-text); overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow-wrap: anywhere; }
#timeline-page .tl-bin-status { display: inline-flex; align-items: center; gap: 5px; font-size: 11px;
  color: var(--c-text-3); }
#timeline-page .tl-bin-scene[data-status="Failed"] .tl-bin-status { color: var(--c-text); font-weight: 700; }
#timeline-page .tl-cast-panel { display: grid; gap: 8px; }
#timeline-page .tl-cast { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
#timeline-page .tl-cast-card { display: grid; grid-template-columns: 40px minmax(0, 1fr); align-items: center; gap: 8px;
  padding: 5px; border-radius: 8px; background: var(--c-surface-2); }
#timeline-page .tl-cast-portrait { display: grid; place-items: center; width: 40px; height: 40px; overflow: hidden;
  border-radius: 50%; background: var(--c-muted); color: var(--c-text-2); font: 650 13px system-ui, sans-serif; }
#timeline-page .tl-cast-portrait img { width: 100%; height: 100%; object-fit: cover; object-position: center 25%; }
#timeline-page .tl-cast-meta { display: grid; gap: 1px; min-width: 0; }
#timeline-page .tl-cast-meta strong { font-size: 13px; color: var(--c-text); font-weight: 600; }
#timeline-page .tl-cast-meta small { font-size: 11.5px; color: var(--c-text-3); overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; }
#timeline-page .tl-assets { display: grid; gap: 8px; }
#timeline-page .tl-asset-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(64px, 1fr)); gap: 8px; }
#timeline-page .tl-asset { display: grid; gap: 3px; margin: 0; min-width: 0; }
#timeline-page .tl-asset-thumb { display: grid; place-items: center; aspect-ratio: 1; overflow: hidden;
  border-radius: 7px; border: 1px solid var(--c-border); background: var(--c-muted); color: var(--c-text-3);
  font-size: 10px; }
#timeline-page .tl-asset img { width: 100%; height: 100%; object-fit: cover; }
#timeline-page .tl-asset figcaption { font: 600 11px ui-monospace, Consolas, monospace; color: var(--c-text-2);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#timeline-page .tl-asset button { min-height: 26px; padding: 0 8px; border-radius: 7px; border: 1px solid var(--c-border-2);
  background: transparent; color: var(--c-text); font-size: 12px; cursor: pointer; }
#timeline-page .tl-asset button:disabled { opacity: .45; cursor: default; }
#timeline-page .tl-asset button:focus-visible { outline: 2px solid var(--c-accent); outline-offset: 1px; }
#timeline-page #timeline-bin-tabs .toolbar { gap: 6px !important; }

/* ----- viewer ----- */
#timeline-page .tl-viewer-panel { padding: 0 !important; gap: 0 !important; background: var(--tl-well) !important; }
#timeline-page .tl-viewer-panel > #timeline-viewer { flex: 1 1 auto !important; min-height: 0 !important;
  display: flex !important; flex-direction: column; }
#timeline-page #timeline-viewer div:has(.tl-viewer) { flex: 1 1 auto; min-height: 0; display: flex;
  flex-direction: column; }
#timeline-page .tl-viewer { --tl-aspect: 1.7778; flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
#timeline-page .tl-stage { position: relative; flex: 1 1 auto; min-height: 140px; overflow: hidden;
  background: var(--tl-well); outline: none; }
/* the focus ring is drawn above the picture, the slate and the caption (an inset shadow of the stage
   itself would be painted under them) */
#timeline-page .tl-stage:focus-visible::after { content: ""; position: absolute; inset: 0; z-index: 5;
  pointer-events: none; box-shadow: inset 0 0 0 2px var(--c-accent); }
#timeline-page .tl-stage:fullscreen { background: #000000; }
#timeline-page .tl-screen { position: absolute; inset: 0; }
#timeline-page .tl-video { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain;
  border-radius: 0; opacity: 0; pointer-events: none; background: transparent; }
#timeline-page .tl-video[data-active] { opacity: 1; }
#timeline-page .tl-slate { position: absolute; inset: 0; display: grid; place-items: center; padding: 18px;
  opacity: 0; pointer-events: none; }
#timeline-page .tl-slate[data-active] { opacity: 1; }
#timeline-page .tl-slate-card { display: grid; align-content: center; gap: 10px; box-sizing: border-box;
  width: min(100%, 700px); max-height: 100%; padding: clamp(14px, 3vw, 30px); overflow: hidden;
  border: 1px dashed var(--tl-line); border-radius: 12px; background: var(--tl-fill); }
#timeline-page .tl-slate-kicker { font: 650 10.5px/1.4 system-ui, sans-serif; letter-spacing: .12em;
  text-transform: uppercase; color: var(--tl-muted); }
#timeline-page .tl-slate-text { margin: 0; color: var(--tl-text); font-size: clamp(15px, 1.6vw, 22px); line-height: 1.35;
  font-weight: 550; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 6; -webkit-box-orient: vertical;
  overflow-wrap: anywhere; }
#timeline-page .tl-slate-note { font-size: 12.5px; color: var(--tl-muted); }
#timeline-page .tl-caption { position: absolute; left: 8px; top: 8px; max-width: calc(100% - 16px); margin: 0;
  padding: 3px 8px; border-radius: 6px; background: rgba(10, 10, 10, .72); color: var(--tl-text);
  font-size: 11.5px; line-height: 1.4; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  pointer-events: none; }
#timeline-page .tl-caption:empty { display: none; }
#timeline-page .tl-transport { flex: none; display: flex; align-items: center; flex-wrap: wrap; gap: 4px 10px;
  padding: 6px 8px; border-top: 1px solid var(--tl-edge); background: var(--c-surface); }
#timeline-page .tl-modes { display: inline-flex; gap: 2px; padding: 3px; border-radius: 9px; background: var(--c-muted); }
#timeline-page .tl-modes button { min-height: 26px; padding: 0 10px; border: 0; border-radius: 7px; background: transparent;
  color: var(--c-text-2); font-family: inherit; font-size: 12.5px;
  font-weight: 550; line-height: 1; cursor: pointer; }
#timeline-page .tl-modes button:hover { color: var(--c-text); }
#timeline-page .tl-modes button[aria-pressed="true"] { background: var(--c-surface-2); color: var(--c-text);
  box-shadow: inset 0 0 0 1px var(--c-border-2); }
#timeline-page .tl-modes button[aria-disabled="true"] { color: var(--c-text-3); cursor: not-allowed; }
#timeline-page .tl-steps { display: inline-flex; align-items: center; gap: 2px; margin: 0 auto; }
#timeline-page .tl-tool { display: inline-grid; place-items: center; width: 32px; height: 32px; padding: 0; border: 0;
  border-radius: 8px; background: transparent; color: var(--c-text); cursor: pointer; }
#timeline-page .tl-tool:hover { background: var(--c-surface-2); }
#timeline-page .tl-tool .tl-icon { width: 18px; height: 18px; }
#timeline-page .tl-tool[data-step="play"] { width: 38px; height: 38px; border-radius: 50%; background: var(--c-text);
  color: var(--c-bg); }
#timeline-page .tl-tool[data-step="play"]:hover { background: var(--c-accent); }
#timeline-page .tl-tool[data-step="play"] .tl-icon + .tl-icon { display: none; }
#timeline-page .tl-tool[data-step="play"][data-playing] .tl-icon:first-child { display: none; }
#timeline-page .tl-tool[data-step="play"][data-playing] .tl-icon + .tl-icon { display: block; }
#timeline-page :is(.tl-tool, .tl-modes button):focus-visible { outline: 2px solid var(--c-accent); outline-offset: 2px; }
#timeline-page .tl-timecode { margin: 0; display: inline-flex; align-items: baseline; gap: 6px; white-space: nowrap;
  font: 600 13px ui-monospace, Consolas, monospace; font-variant-numeric: tabular-nums; color: var(--c-text); }
#timeline-page .tl-timecode-total { color: var(--c-text-3); font-weight: 500; font-size: 12px; }

/* ----- inspector ----- */
#timeline-page .tl-inspector { padding: 12px 10px 10px 12px !important; }
#timeline-page .tl-inspector > #timeline-inspector-tabs { flex: 1 1 auto !important; }
#timeline-page .tl-inspector-actions { flex: none !important; display: flex !important; flex-wrap: nowrap !important;
  gap: 6px !important; padding-top: 8px !important; border-top: 1px solid var(--c-border); }
#timeline-page .tl-inspector-actions > * { flex: 1 1 0 !important; min-width: 0 !important; }
#timeline-page #timeline-generate::before { content: ""; width: 14px; height: 14px; margin-right: 6px; flex: none;
  background: currentColor; -webkit-mask: var(--tl-spark) center / contain no-repeat; mask: var(--tl-spark) center / contain no-repeat; }
#timeline-page { --tl-spark: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 2.5 14.3 9.7 21.5 12l-7.2 2.3L12 21.5l-2.3-7.2L2.5 12l7.2-2.3z' fill='black'/%3E%3C/svg%3E"); }
#timeline-page .tl-scene-head { display: grid; gap: 3px; padding: 0 2px; }
#timeline-page .tl-scene-title { margin: 0; display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
  color: var(--c-text); font-size: 15px; font-weight: 650; }
#timeline-page .tl-scene-title small { color: var(--c-text-3); font-size: 12px; font-weight: 500; }
#timeline-page .tl-scene-meta { margin: 0; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  font-size: 12px; color: var(--c-text-2); font-variant-numeric: tabular-nums; }
#timeline-page .tl-scene-head .timeline-note { margin-top: 4px; }
#timeline-page :is(.tl-inspector, .tl-bin) :is(p, span, li, label, strong) { overflow-wrap: anywhere; }
#timeline-page #timeline-description textarea { min-height: 112px; font-size: 14.5px; line-height: 1.5; }
#timeline-page .tl-inspector .timeline-controls { align-items: end; gap: 10px; flex-wrap: wrap !important; }
#timeline-page .timeline-note { margin: 0; padding: 8px 10px; border-radius: 10px; font-size: 12.5px;
  color: var(--c-text); background: var(--c-surface-2); border: 1px solid var(--c-border-2); }
#timeline-page .timeline-note[data-level="error"] { border-color: #ffffff; font-weight: 600; }
#timeline-page .timeline-field-label { margin: 0; font-size: 13px; font-weight: 550; color: var(--c-text); }
#timeline-page :is(#timeline-line-assignments, #timeline-lines, #timeline-effects, #timeline-references, #timeline-view) {
  display: none !important; }
#timeline-page .slider_input_container .min_value, #timeline-page .slider_input_container .max_value {
  flex: none; white-space: nowrap; }
#timeline-page .timeline-speaker-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(110px, 38%);
  align-items: center; gap: 8px; padding: 6px 0; border-bottom: 1px solid var(--c-border); font-size: 12px; }
#timeline-page .timeline-speaker-row select { border-radius: 8px; padding: 4px 26px 4px 6px; min-width: 0;
  color: var(--c-text); border: 1px solid var(--c-border-2); background-color: var(--c-bg); }
#timeline-page .timeline-speaker-row .timeline-line-number { color: var(--c-text-3); font-variant-numeric: tabular-nums; }
#timeline-page .tl-rows { display: grid; gap: 6px; }
#timeline-page .tl-rows-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
#timeline-page :is(.tl-rows-head, .tl-row) { display: grid; align-items: center; gap: 6px; }
#timeline-page .tl-rows-head { padding: 0 7px; font-size: 11px; color: var(--c-text-3); }
#timeline-page .tl-rows-head span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#timeline-page .tl-rows[data-rows="lines"] :is(.tl-rows-head, .tl-row) {
  grid-template-columns: 20px minmax(92px, 1fr) minmax(0, 2.2fr) 64px 84px 76px; }
#timeline-page .tl-rows[data-rows="effects"] :is(.tl-rows-head, .tl-row) {
  grid-template-columns: 20px minmax(0, 2.2fr) 60px 60px 84px 76px; }
#timeline-page .tl-rows[data-rows="lines"] .tl-row { grid-template-areas:
  "num who text time window remove" ". note note note note note"; }
#timeline-page .tl-rows[data-rows="effects"] .tl-row { grid-template-areas:
  "num text time length window remove" ". note note note note note"; }
#timeline-page .tl-row { margin: 0; padding: 6px; border-radius: 10px; background: var(--c-surface-2);
  border: 1px solid var(--c-border); }
#timeline-page :is(.tl-rows, .tl-refs) [hidden], #timeline-page .tl-row[hidden] { display: none; }
#timeline-page .tl-row-number { grid-area: num; text-align: center; color: var(--c-text-3);
  font: 600 12px/1 ui-monospace, Consolas, monospace; font-variant-numeric: tabular-nums; }
#timeline-page .tl-row-who { grid-area: who; }
#timeline-page .tl-row-text { grid-area: text; }
#timeline-page .tl-row-time { grid-area: time; }
#timeline-page .tl-row-length { grid-area: length; }
#timeline-page .tl-row-remove { grid-area: remove; }
#timeline-page .tl-row-note { grid-area: note; margin: 0; font-size: 12px; color: var(--c-text-2); }
#timeline-page .tl-row :is(input, select) { width: 100%; min-width: 0; min-height: 32px; padding: 5px 8px; margin: 0;
  border-radius: 8px; border: 1px solid var(--c-border-2); background-color: var(--c-bg); color: var(--c-text);
  font: inherit; font-size: 13px; }
#timeline-page .tl-row select { padding-right: 26px; }
#timeline-page .tl-row :is(input, select):hover { border-color: #555555; }
#timeline-page .tl-row input::placeholder { color: var(--c-text-3); }
#timeline-page .tl-row input[type="number"] { font-variant-numeric: tabular-nums; }
#timeline-page .tl-row-window { grid-area: window; display: grid; line-height: 1.25; white-space: nowrap; min-width: 0;
  overflow: hidden; font-size: 12px; color: var(--c-text-2); font-variant-numeric: tabular-nums; }
#timeline-page .tl-row-window small { font-size: 10.5px; color: var(--c-text-3); }
#timeline-page .tl-row-window[data-measured="true"] { color: var(--c-text); font-weight: 600; }
#timeline-page .tl-row-window[data-past-end="true"] { color: #ffffff; font-weight: 700; }
#timeline-page .tl-row-window[data-past-end="true"] small { color: #ffffff; text-decoration: underline; }
#timeline-page :is(.tl-row-remove, .tl-ref button) { min-height: 32px; padding: 0 10px; margin: 0; border-radius: 8px;
  border: 1px solid var(--c-border-2); background: transparent; color: var(--c-text-2); font-size: 12px;
  white-space: nowrap; cursor: pointer; }
#timeline-page :is(.tl-row-remove, .tl-ref button):hover { color: var(--c-text); border-color: #555555; }
#timeline-page .tl-rows-foot { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
#timeline-page .tl-rows-add { min-height: 32px; padding: 0 12px; margin: 0; border-radius: 8px; border: 1px dashed #555555;
  background: transparent; color: var(--c-text); font-size: 13px; font-weight: 550; cursor: pointer; }
#timeline-page .tl-rows-add:hover { background: var(--c-surface-2); border-style: solid; }
#timeline-page .tl-rows-add:disabled { opacity: .45; cursor: default; }
#timeline-page :is(.tl-rows-hint, .tl-rows-empty, .tl-refs-empty, .tl-check-ok, .tl-players-state) {
  margin: 0; font-size: 12px; color: var(--c-text-3); }
#timeline-page .tl-rows-hint { font-variant-numeric: tabular-nums; }
#timeline-page .tl-rows-message { margin: 0; padding: 7px 10px; border-radius: 8px; font-size: 12.5px;
  color: var(--c-text); border: 1px solid var(--c-border-2); background: var(--c-surface-2); }
#timeline-page [data-narrow] .tl-rows-head { display: none; }
#timeline-page [data-narrow] .tl-rows[data-rows="lines"] .tl-row {
  grid-template-columns: 20px minmax(0, 1fr) minmax(0, 1fr) auto;
  grid-template-areas: "num who who remove" ". text text text" ". time window window" ". note note note"; }
#timeline-page [data-narrow] .tl-rows[data-rows="effects"] .tl-row {
  grid-template-columns: 20px minmax(0, 1fr) minmax(0, 1fr) auto;
  grid-template-areas: "num text text remove" ". time length window" ". note note note"; }
#timeline-page .tl-refs { display: grid; gap: 8px; }
#timeline-page .tl-refs-list { display: flex; flex-wrap: wrap; gap: 8px; }
#timeline-page .tl-refs-list:empty { display: none; }
#timeline-page .tl-ref { display: grid; gap: 4px; width: 84px; margin: 0; }
#timeline-page .tl-ref-thumb { display: grid; place-items: center; aspect-ratio: 1; overflow: hidden;
  border-radius: 8px; border: 1px solid var(--c-border); background: var(--c-muted); color: var(--c-text-3);
  font-size: 11px; }
#timeline-page .tl-ref img { width: 100%; height: 100%; object-fit: cover; }
#timeline-page .tl-ref figcaption { font: 600 12px ui-monospace, Consolas, monospace; color: var(--c-text-2); }
#timeline-page .tl-ref button { min-height: 28px; }
#timeline-page .tl-check { display: grid; gap: 6px; margin: 0; padding: 0; list-style: none; font-size: 12.5px; }
#timeline-page .tl-check li { display: grid; grid-template-columns: 10px auto minmax(0, 1fr); gap: 7px;
  align-items: baseline; margin: 0; color: var(--c-text-2); }
#timeline-page .tl-check strong { color: var(--c-text); font-weight: 600; }
#timeline-page .tl-check li[data-level="error"] strong { color: #ffffff; font-weight: 750; }
#timeline-page .tl-mark { display: inline-block; width: 8px; height: 8px; flex: none; }
#timeline-page .tl-mark[data-level="error"] { background: #ffffff; }
#timeline-page .tl-mark[data-level="warning"] { border: 1.5px solid #a6a6a6; border-radius: 50%; }
#timeline-page .tl-mark[data-level="info"] { width: 6px; height: 6px; margin: 1px; border-radius: 50%;
  background: #8f8f8f; }
#timeline-page .tl-mark[data-level="ok"] { border-radius: 50%; background: #e6e6e6; margin-right: 7px; }
#timeline-page .tl-players { display: grid; gap: 8px; }
#timeline-page .tl-players-state[data-stale] { color: var(--c-text); font-weight: 600; }
#timeline-page .tl-player { display: grid; gap: 3px; min-width: 0; }
#timeline-page .tl-player > span { font-size: 12px; color: var(--c-text-2); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
#timeline-page .tl-player audio { width: 100%; height: 32px; }
#timeline-page #timeline-takes .wrap { display: grid !important; gap: 4px !important; padding: 0 !important;
  background: none !important; }
#timeline-page #timeline-takes .wrap > label { justify-content: flex-start; padding: 7px 10px !important;
  border: 1px solid var(--c-border) !important; border-radius: 8px !important; }
#timeline-page #timeline-takes .wrap > label.selected { border-color: var(--c-border-2) !important;
  background: var(--c-surface-2) !important; }
#timeline-page :is(.tl-rows, .tl-refs, .timeline-speaker-row) :is(button, input, select):focus-visible,
#timeline-page .tl-player audio:focus-visible { outline: 2px solid #ffffff; outline-offset: 2px; }

/* ----- timeline panel: toolbar, rail, ruler, lanes ----- */
#timeline-page .tl-timeline-panel { padding: 0 !important; gap: 0 !important; background: var(--tl-deck) !important; }
#timeline-page .tl-timeline-toolbar { flex: none !important; display: flex !important; flex-wrap: wrap !important;
  align-items: center !important; gap: 6px !important; padding: 6px 8px !important;
  border-bottom: 1px solid var(--tl-edge); }
#timeline-page .tl-timeline-toolbar > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
#timeline-page .tl-timeline-toolbar > #timeline-zoom { margin-left: auto !important; }
#timeline-page .tl-timeline-toolbar button.sm { min-height: 28px; padding: 0 10px; font-size: 12px;
  background: var(--tl-sound); border: 1px solid #333333; color: var(--tl-text); }
#timeline-page .tl-timeline-toolbar button.sm:hover { background: var(--tl-hover); }
#timeline-page .tl-timeline-panel > #timeline-track { flex: 1 1 auto !important; min-height: 0 !important;
  display: flex !important; flex-direction: column; }
#timeline-page #timeline-track div:has(.timeline-deck) { flex: 1 1 auto; min-height: 0; display: flex;
  flex-direction: column; }
/* Lanes share the panel's height: picture first, then dialogue and sound, so waveforms stay readable. At
   their smallest (132 px, 158 px with V2) they all fit the timeline's floor above the sideways scroll bar. */
#timeline-page .timeline-deck { --tl-rows: 24px minmax(44px, 1fr) minmax(32px, .42fr) minmax(32px, .34fr);
  flex: 1 1 auto; min-height: 0;
  display: flex; flex-direction: column; background: var(--tl-deck); color: var(--tl-text); }
#timeline-page .timeline-deck[data-edit] {
  --tl-rows: 24px minmax(44px, 1fr) minmax(32px, .42fr) minmax(32px, .34fr) 26px; }
#timeline-page .tl-zoombar { background: transparent; }
#timeline-page .timeline-scale { display: flex; align-items: center; gap: 4px; }
#timeline-page .timeline-scale button { min-width: 28px; min-height: 28px; padding: 0 8px; margin: 0;
  border: 1px solid #333333; border-radius: 6px; background: var(--tl-sound); color: var(--tl-text);
  font: 600 12px system-ui, sans-serif; cursor: pointer; }
#timeline-page .timeline-scale button:hover { background: var(--tl-hover); }
/* − at the smallest scale, + at the largest (aria-disabled: a focused button keeps the focus) */
#timeline-page .timeline-scale button[aria-disabled="true"] { opacity: .4; cursor: default; }
#timeline-page .timeline-scale input[type="range"] { width: clamp(72px, 10vw, 140px); margin: 0 2px;
  accent-color: var(--tl-text); cursor: pointer; }
/* The view scrolls sideways at every zoom (a sideways swipe never goes back in the browser's history). The
   track's script sets the scale (--tl-pps), the project's length (--tl-length), the canvas width
   (--tl-canvas: the project and one view's width of empty time after it), the playhead and the ruler's
   grid on the component's own element, so a redraw of the markup inside it keeps them. */
#timeline-page .timeline-scroll { position: relative; flex: 1 1 auto; min-height: 0; overflow: auto;
  overscroll-behavior-x: contain; overflow-anchor: none; scrollbar-color: #4a4a4a #141414; }
#timeline-page .timeline-board { display: flex; width: max-content; min-width: 100%; height: 100%; min-height: 132px; }
#timeline-page .timeline-rail { position: sticky; left: 0; z-index: 6; flex: none; width: var(--tl-rail);
  display: grid; grid-template-rows: var(--tl-rows); background: var(--tl-fill); border-right: 1px solid var(--tl-edge);
  color: var(--tl-text); font: 650 11px/1 system-ui, sans-serif; }
#timeline-page .timeline-rail > div { display: flex; align-items: center; gap: 6px; min-height: 0; padding: 0 8px;
  overflow: hidden; border-bottom: 1px solid var(--tl-edge); color: var(--tl-text); }
#timeline-page .timeline-rail small { font: 600 10px ui-monospace, Consolas, monospace; color: var(--tl-muted); }
#timeline-page .timeline-rail .tl-icon { flex: none; width: 13px; height: 13px; color: var(--tl-muted); }
#timeline-page .timeline-rail .timeline-ruler-label { justify-content: flex-end; color: var(--tl-muted);
  font: 500 10px ui-monospace, Consolas, monospace; }
#timeline-page .timeline-strip { position: relative; flex: 1 0 auto; width: var(--tl-canvas, auto); min-width: 0;
  display: grid; grid-template-rows: var(--tl-rows); background: var(--tl-well); }
/* the empty time after the project's end: a line at the end, faint stripes after it */
#timeline-page .timeline-end { position: absolute; z-index: 2; top: 0; bottom: 0; left: var(--tl-length, 100%);
  right: 0; border-left: 1px solid #6e6e6e; pointer-events: none;
  background: repeating-linear-gradient(135deg, rgba(255, 255, 255, .03) 0 6px, transparent 6px 12px); }
#timeline-page .timeline-ruler { position: relative; overflow: hidden; touch-action: none; cursor: ew-resize;
  border-bottom: 1px solid var(--tl-edge); background: var(--tl-fill); outline: none; }
#timeline-page .timeline-ruler:focus-visible { box-shadow: inset 0 0 0 2px var(--tl-cursor); }
#timeline-page .timeline-tick { position: absolute; bottom: 0; width: 1px; height: 5px; background: #4d4d4d;
  pointer-events: none; }
#timeline-page .timeline-tick.major { height: 9px; background: #6e6e6e; }
#timeline-page .timeline-tick span { position: absolute; left: 4px; bottom: 7px; white-space: nowrap; color: var(--tl-muted);
  font: 10px ui-monospace, Consolas, monospace; font-variant-numeric: tabular-nums; }
#timeline-page :is(.timeline-track, .timeline-dialogue, .timeline-sound) { display: grid; min-height: 0;
  grid-template-columns: var(--scene-columns); border-bottom: 1px solid var(--tl-edge);
  background-image: linear-gradient(to right, rgba(255, 255, 255, .05) 1px, transparent 1px);
  background-size: var(--tl-grid, 80px) 100%; }
#timeline-page .timeline-clip { position: relative; display: flex; flex-direction: column; min-width: 0; min-height: 0;
  overflow: hidden; margin: 4px 1px; padding: 0; border: 1px solid var(--tl-line); border-radius: 6px;
  background: var(--tl-clip); color: var(--tl-text); text-align: left; font: inherit; cursor: grab; }
#timeline-page .timeline-clip:not([data-frames]) { border-style: dashed; background: var(--tl-fill); }
#timeline-page .timeline-clip[data-frames] { background: var(--tl-clip-ready); }
#timeline-page .timeline-clip[data-cut="true"] { border-left: 3px solid #bfbfbf; }
#timeline-page .timeline-clip:is([data-status="Ready"], [data-status="Generating"]) { border-style: solid; }
#timeline-page .timeline-clip[data-status="Failed"] { border-style: dashed; border-color: #ffffff; }
#timeline-page .timeline-clip[data-status="Empty"] { border-style: dashed; background: var(--tl-fill); }
#timeline-page .timeline-clip[data-status="Outdated"] { border-style: dashed; }
#timeline-page .timeline-clip:hover { border-color: #8f8f8f; }
/* selection is a white band inside the clip; keyboard focus a ring outside it (see below) */
#timeline-page .timeline-clip[aria-pressed="true"] { box-shadow: inset 0 0 0 2px var(--tl-accent);
  border-color: var(--tl-accent); z-index: 1; }
/* status without the label (narrow clips hide it): a strip along the top, its pattern per status */
#timeline-page .timeline-clip::after { content: ""; position: absolute; inset: 0 0 auto 0; height: 3px; z-index: 2;
  pointer-events: none; }
#timeline-page .timeline-clip[data-status="Ready"]::after { background: #e6e6e6; }
#timeline-page .timeline-clip[data-status="Generating"]::after {
  background: repeating-linear-gradient(-45deg, #ffffff 0 3px, transparent 3px 6px); }
#timeline-page .timeline-clip[data-status="Queued"]::after {
  background: repeating-linear-gradient(90deg, #bfbfbf 0 2px, transparent 2px 6px); }
#timeline-page .timeline-clip[data-status="Stopped"]::after {
  background: repeating-linear-gradient(90deg, #8f8f8f 0 8px, transparent 8px 12px); }
#timeline-page .timeline-clip[data-status="Modified"]::after {
  background: linear-gradient(to right, #bfbfbf 0 35%, transparent 35%); }
#timeline-page .timeline-clip[data-status="Outdated"]::after {
  background: repeating-linear-gradient(90deg, #e6e6e6 0 3px, transparent 3px 6px); }
/* a subtle shimmer while a clip is being made */
#timeline-page .timeline-clip:is([data-status="Generating"], [data-status="Queued"])::before { content: "";
  position: absolute; inset: 0; z-index: 1; pointer-events: none; background-size: 220% 100%;
  background-image: linear-gradient(100deg, transparent 30%, rgba(255, 255, 255, .09) 50%, transparent 70%);
  animation: tl-shimmer 1.8s linear infinite; }
@keyframes tl-shimmer { from { background-position: 120% 0; } to { background-position: -120% 0; } }
#timeline-page .timeline-clip[data-drop="before"] { box-shadow: inset 3px 0 #ffffff; }
#timeline-page .timeline-clip[data-drop="after"] { box-shadow: inset -3px 0 #ffffff; }
/* the same marks over the filmstrip, which covers the clip's own background */
#timeline-page .timeline-clip-ring { position: absolute; inset: 0; z-index: 3; border-radius: 5px; pointer-events: none; }
#timeline-page .timeline-clip[aria-pressed="true"] .timeline-clip-ring { box-shadow: inset 0 0 0 2px var(--tl-accent); }
#timeline-page .timeline-clip[data-drop="before"] .timeline-clip-ring { box-shadow: inset 3px 0 #ffffff; }
#timeline-page .timeline-clip[data-drop="after"] .timeline-clip-ring { box-shadow: inset -3px 0 #ffffff; }
#timeline-page .timeline-clip[data-trimming] { cursor: ew-resize; border-color: var(--tl-accent); }
#timeline-page .timeline-clip-frames { position: absolute; inset: 0; display: flex; gap: 1px; background: var(--tl-clip);
  pointer-events: none; }
#timeline-page .timeline-clip-frames img { flex: 1 1 0; min-width: 0; height: 100%; object-fit: cover;
  object-position: center 38%; }
#timeline-page .timeline-clip-frames img[hidden] { display: none; }
#timeline-page .timeline-clip-top { position: relative; z-index: 2; flex: none; display: flex; align-items: center;
  gap: 5px; height: 20px; min-width: 0; padding: 0 12px 0 6px; background: rgba(10, 10, 10, .74);
  font: 600 11px/1 system-ui, sans-serif; color: var(--tl-text); }
#timeline-page .timeline-clip-top strong { flex: none; font: 650 11px ui-monospace, Consolas, monospace; color: var(--tl-text); }
#timeline-page .timeline-clip-label { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; color: var(--tl-text); }
#timeline-page .timeline-clip-label::before { content: "· "; color: var(--tl-muted); }
#timeline-page .timeline-clip:not([data-frames]) .timeline-clip-label { display: none; }
#timeline-page .timeline-clip:not([data-frames]) .timeline-clip-top { background: transparent; }
#timeline-page .timeline-clip-status { flex: none; color: #bfbfbf; white-space: nowrap; }
#timeline-page .timeline-clip[data-status="Generating"] .timeline-clip-status { color: #ffffff; font-weight: 700; }
#timeline-page .timeline-clip[data-status="Failed"] .timeline-clip-status { color: #ffffff; font-weight: 700;
  text-decoration: underline; }
#timeline-page .timeline-clip-duration { flex: none; white-space: nowrap; color: #d9d9d9;
  font: 11px ui-monospace, Consolas, monospace; }
#timeline-page .timeline-clip-alert { flex: none; display: inline-grid; place-items: center; width: 14px; height: 14px;
  background: #ffffff; color: #0a0a0a; font: 800 11px/1 system-ui, sans-serif; }
#timeline-page .timeline-clip-body { position: relative; z-index: 2; flex: 1 1 auto; min-height: 0; overflow: hidden;
  padding: 5px 12px 5px 6px; }
#timeline-page .timeline-clip-title { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
  overflow: hidden; overflow-wrap: anywhere; color: #cfcfcf; font: 500 11px/1.3 system-ui, sans-serif; }
#timeline-page .timeline-clip[data-frames] .timeline-clip-title { display: none; }
#timeline-page .timeline-clip[data-narrow] :is(.timeline-clip-label, .timeline-clip-status) { display: none; }
#timeline-page .timeline-clip[data-tiny] :is(.timeline-clip-duration, .timeline-clip-title, .timeline-mark) { display: none; }
#timeline-page .timeline-clip[data-tiny] .timeline-clip-top { padding: 0 3px; }
#timeline-page .timeline-trim { position: absolute; z-index: 3; top: 0; right: 0; bottom: 0; width: 9px;
  cursor: ew-resize; touch-action: none; }
#timeline-page .timeline-trim::after { content: ""; position: absolute; top: 50%; right: 2px; width: 3px; height: 20px;
  border-radius: 2px; background: rgba(255, 255, 255, .7); transform: translateY(-50%); opacity: 0; }
#timeline-page .timeline-clip:is(:hover, [data-trimming], [aria-pressed="true"]) .timeline-trim::after { opacity: 1; }
#timeline-page .timeline-trim-tip { position: absolute; z-index: 8; top: 30px; padding: 3px 7px; border-radius: 6px;
  background: var(--tl-accent); color: #0a0a0a; font: 650 11px ui-monospace, Consolas, monospace;
  transform: translateX(-50%); white-space: nowrap; pointer-events: none; }
#timeline-page .timeline-trim-tip[hidden] { display: none; }
/* status marks: the shape tells the status (filled dot ready, ring queued, square failed…) */
#timeline-page .timeline-mark { flex: none; display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: none; box-shadow: inset 0 0 0 1.5px #8f8f8f; }
#timeline-page .timeline-mark[data-status="Ready"] { background: #e6e6e6; box-shadow: none; }
#timeline-page .timeline-mark[data-status="Generating"] { background: #ffffff; box-shadow: 0 0 0 2px rgba(255, 255, 255, .25); }
#timeline-page .timeline-mark[data-status="Queued"] { box-shadow: inset 0 0 0 1.5px #bfbfbf; }
#timeline-page .timeline-mark[data-status="Failed"] { border-radius: 1px; background: #ffffff; box-shadow: none; }
#timeline-page .timeline-mark[data-status="Stopped"] { border-radius: 1px; box-shadow: inset 0 0 0 1.5px #bfbfbf; }
#timeline-page .timeline-mark[data-status="Empty"] { box-shadow: inset 0 0 0 1px #6e6e6e; }
#timeline-page .timeline-mark[data-status="Outdated"] { background: linear-gradient(to right, #e6e6e6 50%, transparent 50%);
  box-shadow: inset 0 0 0 1.5px #e6e6e6; }
#timeline-page :is(.timeline-audio-cell, .timeline-sound-cell) { position: relative; min-width: 0; overflow: hidden;
  border-right: 1px solid rgba(255, 255, 255, .08); }
#timeline-page .tl-wave { position: absolute; left: 0; top: 3px; width: 100%; height: calc(100% - 6px); fill: currentColor;
  opacity: .42; pointer-events: none; }
#timeline-page .timeline-audio-segment { position: absolute; top: 6px; bottom: 6px; left: var(--audio-start);
  width: var(--audio-width); min-width: 4px; margin: 0; padding: 0 4px; overflow: hidden; border: 1px solid #8f8f8f;
  border-radius: 5px; background: #2e2e2e; color: var(--tl-text); text-align: left; cursor: pointer;
  font: 600 10px/1 system-ui, sans-serif; }
#timeline-page .timeline-audio-text { position: relative; display: block; overflow: hidden; white-space: nowrap;
  text-overflow: ellipsis; }
#timeline-page .timeline-audio-segment[data-planned] { border-style: dashed; border-color: #6e6e6e;
  background: #1a1a1a; color: #cfcfcf; }
#timeline-page .timeline-audio-segment[data-overflow] { border-right: 0; border-top-right-radius: 0;
  border-bottom-right-radius: 0; padding-right: 14px; }
#timeline-page .timeline-audio-segment[data-overflow]::after { content: "»"; position: absolute; right: 0;
  top: -1px; bottom: -1px; width: 11px; display: grid; place-items: center; background: #ffffff; color: #0a0a0a;
  font: 800 10px/1 system-ui, sans-serif; }
#timeline-page .timeline-audio-segment:hover { border-color: #ffffff; }
#timeline-page .timeline-bed { position: absolute; left: 2px; right: 2px; top: 4px; bottom: 4px; margin: 0;
  padding: 0 6px; border: 0; border-radius: 4px; background: var(--tl-sound); color: var(--tl-muted); text-align: left;
  font: 600 9.5px/1 system-ui, sans-serif; overflow: hidden; cursor: pointer; }
#timeline-page .timeline-bed-text { position: absolute; left: 6px; right: 6px; bottom: 3px; overflow: hidden;
  white-space: nowrap; text-overflow: ellipsis; }
#timeline-page .timeline-bed[data-custom] { color: var(--tl-text); background: #2b2b2b; }
#timeline-page .timeline-bed:hover { background: var(--tl-hover); color: var(--tl-text); }
#timeline-page .timeline-effect { position: absolute; z-index: 1; top: 2px; height: 15px; left: var(--fx-start);
  width: var(--fx-width); min-width: 14px; margin: 0; padding: 0 3px 0 13px; border: 0;
  border-top: 1px solid rgba(255, 255, 255, .55); border-radius: 0; background: rgba(10, 10, 10, .55);
  color: var(--tl-text); text-align: left; font: 600 9.5px/14px system-ui, sans-serif; overflow: hidden;
  white-space: nowrap; text-overflow: ellipsis; cursor: pointer; }
#timeline-page .timeline-effect::before { content: ""; position: absolute; left: 1px; top: 2px; width: 8px; height: 8px;
  background: #ffffff; transform: rotate(45deg); }
#timeline-page .timeline-effect[data-overflow] { border-right: 3px double #ffffff; }
#timeline-page .timeline-effect:hover { background: #333333; }
#timeline-page .timeline-assembly { position: relative; padding: 3px 1px; border-bottom: 1px solid var(--tl-edge); }
#timeline-page .timeline-joined-clip { display: block; width: calc(var(--tl-length, 100%) - 2px); height: 100%;
  margin: 0; padding: 0 8px;
  border: 1px solid #8f8f8f; border-radius: 5px; background: var(--tl-clip-ready); color: var(--tl-text);
  text-align: left; font: 600 11px system-ui, sans-serif; cursor: pointer; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
#timeline-page .timeline-joined-clip[data-kind="earlier"] { border-style: dashed; color: #bfbfbf; }
#timeline-page .timeline-playhead { position: absolute; z-index: 4; top: 0; bottom: 0; left: var(--playhead, 0px);
  width: 0; pointer-events: none; }
#timeline-page .timeline-playhead::before { content: ""; position: absolute; top: 0; bottom: 0; left: -1px; width: 2px;
  background: var(--tl-cursor); box-shadow: 0 0 0 1px rgba(0, 0, 0, .35); }
#timeline-page .timeline-playhead-head { position: absolute; top: 0; left: -7px; width: 14px; height: 15px;
  background: var(--tl-cursor); clip-path: polygon(0 0, 100% 0, 100% 62%, 50% 100%, 0 62%); pointer-events: auto;
  cursor: ew-resize; touch-action: none; }
#timeline-page .timeline-deck :is(button, [role="slider"]):focus-visible {
  outline: 2px solid var(--tl-cursor); outline-offset: -2px; }
/* where there is room, the focus ring stands outside the control: it stays clear of a zoom button's own
   border and tells a focused clip from the selected one */
#timeline-page .timeline-deck :is(.timeline-scale button, .timeline-clip):focus-visible {
  outline-offset: 2px; z-index: 3; }
#timeline-page .timeline-deck-hint { flex: none; margin: 0; padding: 4px 10px 5px; border-top: 1px solid var(--tl-edge);
  color: var(--tl-muted); font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
@media (prefers-reduced-motion: reduce) {
  #timeline-page .timeline-clip::before { animation: none; }
}
@media (max-width: 767px) {
  #timeline-page #timeline-workspace:not([data-size]) { height: auto; min-height: 0; }
  #timeline-page #timeline-workspace:not([data-size]) .tl-body { grid-template-columns: minmax(0, 1fr);
    grid-template-rows: none; grid-auto-rows: auto; grid-template-areas: "viewer" "inspector" "timeline"; }
  #timeline-page #timeline-workspace:not([data-size]) .tl-bin { display: none !important; }
  #timeline-page #timeline-workspace:not([data-size]) .tl-viewer-panel {
    border-right: 0 !important; border-bottom: 1px solid var(--c-border) !important; }
  #timeline-page #timeline-workspace:not([data-size]) .tl-inspector { border-bottom: 1px solid var(--c-border) !important; }
  #timeline-page #timeline-workspace:not([data-size]) .tl-timeline-panel { border-top: 0 !important; }
  #timeline-page .timeline-deck { --tl-rail: 64px; }
  #timeline-page .timeline-rail small { display: none; }
  #timeline-page .timeline-rail .tl-icon { display: none; }
  #timeline-page #timeline-project-menu [data-testid="accordion-content"] { position: fixed; top: auto; left: 12px;
    right: 12px; width: auto; }
  #timeline-page .tl-queue-panel { position: fixed; top: auto; left: 12px; right: 12px; width: auto; }
  /* the chip gets a line of its own, so a long job text ends in an ellipsis instead of overflowing */
  #timeline-page .tl-appbar-production > #timeline-queue-line:has(.tl-queue) { flex: 1 1 100% !important; }
  #timeline-page .tl-queue { max-width: 100%; }
  #timeline-page .tl-steps { margin: 0; }
}
"""


# ----------------------------- scripts -----------------------------

# Shared by the line and effect editors. The rows a user has just changed win over an older
# server render of the same scene (data-key); a different key means another scene was loaded.
ROWS_JS = """
const page = element.closest('#timeline-page');
let latest = null, typed = null;
const box = () => element.querySelector('[data-rows]');
const rowsOf = root => Array.from(root.querySelectorAll('.tl-rows-list > [data-row]'));
let shownKey = box()?.dataset.key ?? null;
function read(row) {
    const data = {};
    for (const field of row.querySelectorAll('[data-field]')) {
        if ('number' in field.dataset) {
            const text = field.value.trim();
            if (text !== '' && Number.isFinite(Number(text))) data[field.dataset.field] = Number(text);
        } else data[field.dataset.field] = field.value;
    }
    return data;
}
function label(root) {
    const noun = root.dataset.noun || 'Row';
    const rows = rowsOf(root);
    rows.forEach((row, index) => {
        const number = String(index + 1);
        const tag = row.querySelector('[data-number-label]');
        if (tag) tag.textContent = number;
        for (const field of row.querySelectorAll('[data-aria]')) {
            field.setAttribute('aria-label', noun + ' ' + number + ' ' + field.dataset.aria);
        }
        row.querySelector('[data-remove]')?.setAttribute('aria-label', 'Remove ' + noun.toLowerCase() + ' ' + number);
    });
    const add = root.querySelector('[data-add]');
    if (add) add.disabled = rows.length >= Number(root.dataset.max || Infinity);
    const empty = root.querySelector('[data-empty]');
    if (empty) empty.hidden = rows.length > 0;
    const count = root.querySelector('[data-count]');
    if (count) count.textContent = String(rows.length);
}
function send(root) {
    latest = { key: root.dataset.key, rows: rowsOf(root).map(read) };
    typed = null;
    const target = root.dataset.target;
    const input = page?.querySelector('#' + target + ' textarea, #' + target + ' input');
    if (!input) return;
    input.value = JSON.stringify(latest);
    input.dispatchEvent(new Event('input', { bubbles: true }));
}
function fill(row, data) {
    for (const field of row.querySelectorAll('[data-field]')) {
        if (field === document.activeElement) continue;
        const value = data[field.dataset.field];
        const text = value === undefined || value === null ? '' : String(value);
        if (field.value === text) continue;
        if (field.tagName === 'SELECT' && !Array.from(field.options).some(option => option.value === text)) continue;
        field.value = text;
    }
}
function reset(root) {
    for (const field of root.querySelectorAll('[data-field]')) {
        field.value = field.tagName === 'SELECT' ? (field.dataset.value ?? '') : (field.getAttribute('value') ?? '');
    }
}
function append(root) {
    const prototype = root.querySelector('.tl-rows-proto > [data-row]');
    const list = root.querySelector('.tl-rows-list');
    if (!prototype || !list) return null;
    const row = prototype.cloneNode(true);
    row.removeAttribute('data-prototype');
    row.hidden = false;
    list.append(row);
    reset(row);
    return row;
}
function playhead(root, row) {
    // A new effect starts where the playhead stands inside the selected scene.
    const field = row.querySelector('[data-field="at"]');
    const ruler = page?.querySelector('#timeline-track .timeline-ruler');
    const clip = page?.querySelector('#timeline-track .timeline-clip[aria-pressed="true"]');
    if (!field || !ruler || !clip) return;
    const now = Number(ruler.getAttribute('aria-valuenow'));
    const start = Number(clip.dataset.start), end = Number(clip.dataset.end);
    if (Number.isFinite(now) && now >= start && now < end) field.value = String(Math.floor((now - start) * 10) / 10);
}
function restoreTyping(root) {
    // Text typed since the last change event stays, even if an older render replaced its row.
    if (!typed) return;
    let field = typed.field, moved = false;
    if (!field.isConnected) {
        const active = document.activeElement;
        if (active && active !== document.body) return;
        field = rowsOf(root)[typed.row]?.querySelector('[data-field="' + typed.name + '"]');
        if (!field) return;
        typed.field = field;
        field.focus({ preventScroll: true });
        moved = true;
    } else if (field !== document.activeElement) return;
    if (field.value !== typed.value) {
        field.value = typed.value;
        moved = true;
    }
    if (moved) try { field.setSelectionRange(typed.start, typed.end); } catch { /* number inputs */ }
}
element.addEventListener('change', event => {
    const root = box();
    if (root && event.target.closest('[data-field]')) send(root);
});
element.addEventListener('input', event => {
    const field = event.target.closest('[data-field]');
    const root = box();
    if (!field || !root || field.tagName === 'SELECT') return;
    let start = null, end = null;
    try { start = field.selectionStart; end = field.selectionEnd; } catch { /* number inputs */ }
    typed = { field, value: field.value, start, end, name: field.dataset.field,
              row: rowsOf(root).indexOf(field.closest('[data-row]')) };
});
element.addEventListener('click', event => {
    const root = box();
    if (!root) return;
    if (event.target.closest('[data-add]')) {
        if (rowsOf(root).length >= Number(root.dataset.max || Infinity)) return;
        const row = append(root);
        if (!row) return;
        if (root.dataset.rows === 'effects') playhead(root, row);
        label(root);
        send(root);
        row.querySelector('[data-focus]')?.focus();
        return;
    }
    const remove = event.target.closest('[data-remove]');
    if (!remove) return;
    const rows = rowsOf(root);
    const index = rows.indexOf(remove.closest('[data-row]'));
    if (index < 0) return;
    rows[index].remove();
    label(root);
    send(root);
    const rest = rowsOf(root);
    const next = (rest[index] || rest[index - 1])?.querySelector('[data-remove]');
    (next || root.querySelector('[data-add]'))?.focus();
});
watch('value', () => {
    const root = box();
    if (!root) return;
    if (root.dataset.key !== shownKey) {
        shownKey = root.dataset.key;
        latest = null;
        typed = null;
        reset(root);
    } else {
        if (latest && latest.key === shownKey) {
            let rows = rowsOf(root);
            while (rows.length < latest.rows.length && append(root)) rows = rowsOf(root);
            rows.slice(latest.rows.length).forEach(row => row.remove());
            rowsOf(root).forEach((row, index) => fill(row, latest.rows[index]));
        }
        restoreTyping(root);
    }
    label(root);
});
new ResizeObserver(() => {
    if (element.clientWidth) element.toggleAttribute('data-narrow', element.clientWidth < 520);
}).observe(element);
"""


REFERENCES_JS = """
const page = element.closest('#timeline-page');
element.addEventListener('click', event => {
    const remove = event.target.closest('[data-remove-reference]');
    const root = element.querySelector('[data-references]');
    if (!remove || !root) return;
    let names = [];
    try { names = JSON.parse(root.dataset.names || '[]'); } catch { names = []; }
    const name = remove.dataset.removeReference;
    const items = Array.from(root.querySelectorAll('[data-reference]'));
    const index = items.findIndex(item => item.dataset.reference === name);
    names = names.filter(item => item !== name);
    root.dataset.names = JSON.stringify(names);
    items[index]?.remove();
    const input = page?.querySelector('#timeline-references textarea, #timeline-references input');
    if (input) {
        input.value = JSON.stringify({ key: root.dataset.key, rows: names });
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    const rest = Array.from(root.querySelectorAll('[data-remove-reference]'));
    (rest[index] || rest[index - 1] || page?.querySelector('#timeline-reference-upload'))?.focus();
});
"""


# The Assets tab adds an image used elsewhere in the project to the selected scene: the same
# hidden field the reference editor writes (#timeline-references), with that editor's form key.
ASSETS_JS = r"""
const page = element.closest('#timeline-page');
element.addEventListener('click', event => {
    const use = event.target.closest('[data-use-reference]');
    if (!use || use.disabled) return;
    const root = page?.querySelector('#timeline-reference-images [data-references]');
    const input = page?.querySelector('#timeline-references textarea, #timeline-references input');
    if (!root || !input) return;
    let names = [];
    try { names = JSON.parse(root.dataset.names || '[]'); } catch { names = []; }
    const name = use.dataset.useReference;
    if (names.includes(name) || names.length >= Number(root.dataset.max || 6)) return;
    names = [...names, name];
    root.dataset.names = JSON.stringify(names);
    use.disabled = true;
    input.value = JSON.stringify({ key: root.dataset.key, rows: names });
    input.dispatchEvent(new Event('input', { bubbles: true }));
});
"""


# The page's own (hidden) tab, as the top bar's menus do; the Gallery button of the Tools panel is the
# fallback.
SHOW_GALLERY_JS = """() => {
    const tab = document.querySelector('#pages > .tab-wrapper [role=tab][data-tab-id="gallery"]');
    if (tab) {
        if (tab.getAttribute('aria-selected') !== 'true') tab.click();
        requestAnimationFrame(() => window.scrollTo({top: 0, left: 0, behavior: 'instant'}));
        return;
    }
    document.querySelector('.workspace-nav button[data-page="gallery"]')?.click();
}"""


# App bar tools that live in the browser: the bin toggle, the workspace size class
# (wide | medium | narrow, from its own width), the workspace's top (--tl-top: it fills the window
# below it, so the top bar's height counts; measured again when the window, the workspace or the
# top bar change size, or the page is shown) and the Project menu (closed with a click elsewhere or
# Esc; Gradio's accordion button gets aria-expanded).
SHELL_JS = r"""
const page = element.closest('#timeline-page');
const toggle = element.querySelector('[data-bin-toggle]');
const workspace = () => page?.querySelector('#timeline-workspace');
function binShown() {
    const bin = page?.querySelector('#timeline-bin');
    return Boolean(bin && bin.getClientRects().length);
}
function syncToggle() {
    toggle?.setAttribute('aria-pressed', String(binShown()));
}
function measure() {
    const box = workspace();
    if (!box || !box.clientWidth) return;
    const width = box.clientWidth;
    const size = width >= 1080 ? 'wide' : width >= 720 ? 'medium' : 'narrow';
    if (box.dataset.size !== size) box.dataset.size = size;
    const top = Math.max(0, Math.round(box.getBoundingClientRect().top + window.scrollY));
    document.documentElement.style.setProperty('--tl-top', top + 'px');
    syncToggle();
}
toggle?.addEventListener('click', () => {
    const box = workspace();
    if (!box) return;
    box.dataset.bin = binShown() ? 'closed' : 'open';
    syncToggle();
});
function menuOpen() {
    const menu = page?.querySelector('#timeline-project-menu');
    const button = menu?.querySelector('.label-wrap');
    return button && button.classList.contains('open') ? { menu, button } : null;
}
document.addEventListener('pointerdown', event => {
    if (!element.isConnected) return;
    const open = menuOpen();
    if (open && !open.menu.contains(event.target)) open.button.click();
}, true);
document.addEventListener('keydown', event => {
    if (!element.isConnected || event.key !== 'Escape') return;
    const open = menuOpen();
    if (open && open.menu.contains(document.activeElement)) {
        open.button.click();
        open.button.focus();
    }
});
// Gradio's accordion button shows its state only with a class: tell assistive technology too.
function syncMenu() {
    const menu = page?.querySelector('#timeline-project-menu');
    const button = menu?.querySelector('.label-wrap');
    if (!button) return false;
    const expanded = String(button.classList.contains('open'));
    if (button.getAttribute('aria-expanded') !== expanded) button.setAttribute('aria-expanded', expanded);
    const panel = menu.querySelector('[data-testid="accordion-content"]');
    if (panel) {
        if (!panel.id) panel.id = 'timeline-project-panel';
        if (button.getAttribute('aria-controls') !== panel.id) button.setAttribute('aria-controls', panel.id);
    }
    const glyph = button.querySelector('.icon');
    if (glyph && glyph.getAttribute('aria-hidden') !== 'true') glyph.setAttribute('aria-hidden', 'true');
    return true;
}
const watchMenu = () => {
    const menu = page?.querySelector('#timeline-project-menu');
    if (!menu || !syncMenu()) return false;
    new MutationObserver(syncMenu).observe(menu, { attributes: true, attributeFilter: ['class'], subtree: true,
                                                   childList: true });
    return true;
};
if (!watchMenu()) requestAnimationFrame(watchMenu);
const sizes = new ResizeObserver(() => measure());
const watchWorkspace = () => {
    const box = workspace();
    if (!box) return false;
    sizes.observe(box);
    const bar = document.getElementById('topbar');  // the app's top bar, above every page
    if (bar) sizes.observe(bar);
    measure();
    return true;
};
if (!watchWorkspace()) requestAnimationFrame(watchWorkspace);
window.addEventListener('resize', measure);
if (page) new MutationObserver(measure)
    .observe(page, { attributes: true, attributeFilter: ['style', 'class', 'hidden'] });
"""


# The queue chip's Details stay open across redraws (Gradio's redraw drops the open attribute).
REPORT_JS = r"""
let open = false;
element.addEventListener('toggle', event => {
    if (event.target.matches?.('details')) open = event.target.open;
}, true);
watch('value', () => {
    const details = element.querySelector('details');
    if (details && open && !details.open) details.open = true;
});
"""


# The bin's storyboard: click a scene to select it, ↑/↓ to move between scenes. Every choice is sent
# (the listener's trigger mode is "multiple": the server applies them in order, so the last one wins);
# `asked` remembers the latest one until the redraw shows it, so going back to the scene the list
# still shows as selected is sent too.
BIN_JS = r"""
let asked = null;  // {index, until}
const shown = () => Number(element.querySelector('[data-scene][aria-pressed="true"]')?.dataset.scene ?? -1);
function select(index) {
    const waiting = asked && performance.now() < asked.until;
    if (index === shown() && !(waiting && asked.index !== index)) return;
    asked = { index, until: performance.now() + 4000 };
    trigger('click', { index });
}
element.addEventListener('click', event => {
    const card = event.target.closest('[data-scene]');
    if (card) select(Number(card.dataset.scene));
});
element.addEventListener('keydown', event => {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
    const cards = Array.from(element.querySelectorAll('[data-scene]'));
    const index = cards.indexOf(event.target.closest('[data-scene]'));
    const next = cards[index + (event.key === 'ArrowDown' ? 1 : -1)];
    if (index < 0 || !next) return;
    event.preventDefault();
    next.focus();
    select(Number(next.dataset.scene));
});
watch('value', () => {
    if (asked && shown() === asked.index) asked = null;
});
"""


# The zoom bar: −, a slider, + and Zoom to fit. It sends what the user asked for and shows the scale
# the track reports (timeline:zoomed); the track does the zooming.
ZOOM_JS = r"""
const page = element.closest('#timeline-page');
const range = element.querySelector('[data-zoom-level]');
const limits = [[element.querySelector('[data-zoom="out"]'), 'min'],
                [element.querySelector('[data-zoom="in"]'), 'max']];
const send = detail => page?.dispatchEvent(new CustomEvent('timeline:zoom', { detail }));
let shown = null, sliding = false;
function inView(seconds) {
    if (!(seconds > 0)) return '';
    if (seconds >= 120) return 'About ' + Math.round(seconds / 60) + ' min in view';
    return 'About ' + (seconds >= 10 ? Math.round(seconds) : Math.round(seconds * 10) / 10) + ' s in view';
}
function show() {
    if (!shown) return;
    const level = Math.min(1, Math.max(0, Number(shown.level) || 0));
    // the slider's thumb is left alone while it is being dragged; it follows every other zoom
    if (range && !sliding) range.value = String(Math.round(level * 100));
    const text = inView(Number(shown.span));
    if (range && text) range.setAttribute('aria-valuetext', text);
    // aria-disabled keeps a focused button focused at the end of the range (and the timeline's keys working)
    for (const [button, limit] of limits) button?.setAttribute('aria-disabled', String(Boolean(shown[limit])));
}
element.addEventListener('click', event => {
    const button = event.target.closest('button[data-zoom]');
    if (button && button.getAttribute('aria-disabled') !== 'true') send({ action: button.dataset.zoom });
});
range?.addEventListener('input', () => {
    sliding = true;
    send({ level: Number(range.value) / 100 });
});
for (const name of ['change', 'blur']) {
    range?.addEventListener(name, () => {
        sliding = false;
        show();
    });
}
page?.addEventListener('timeline:zoomed', event => {
    if (!element.isConnected) return;
    shown = event.detail || {};
    show();
});
send({ query: true });
"""


# The time scale of the track, in pixels per second: pure functions, shared by TRACK_JS and the tests.
ZOOM_MATH_JS = r"""
// The time scale (no DOM; the tests run these functions in Node). The zoom is a number of pixels per
// second: it does not depend on the window's width, so a resize never rescales the timeline.
const FPS = 30;
const ZOOM_MIN = 2, ZOOM_MAX = 600, ZOOM_STEP = 1.5, ZOOM_DEFAULT = 40;
const RULER_STEPS = [1 / FPS, 2 / FPS, 5 / FPS, 10 / FPS, .5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600];
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
// A scale in the range; one a rounding error away from a limit is the limit (steps that multiply and divide
// add such errors up).
function clampPps(value) {
    const scale = clamp(Number.isFinite(value) && value > 0 ? value : ZOOM_DEFAULT, ZOOM_MIN, ZOOM_MAX);
    return scale > ZOOM_MAX * (1 - 1e-9) ? ZOOM_MAX : scale < ZOOM_MIN * (1 + 1e-9) ? ZOOM_MIN : scale;
}
// The zoom slider's 0..1 is logarithmic: each step multiplies the scale by the same factor.
const zoomLevel = pps => Math.log(clampPps(pps) / ZOOM_MIN) / Math.log(ZOOM_MAX / ZOOM_MIN);
const levelPps = level => ZOOM_MIN * Math.pow(ZOOM_MAX / ZOOM_MIN, clamp(Number(level) || 0, 0, 1));
// The canvas: the project, then at least one view's width of empty time, so the view's left edge can go
// from 0 to the project's end and the project's end can be seen, trimmed and dropped onto at any zoom.
const canvasWidth = (total, pps, room) => Math.max(0, total) * pps + Math.max(0, room);
const leftLimit = total => Math.max(0, total);
// "Zoom to fit": the scale that shows the whole project with a small margin after its end.
function fitPps(total, room) {
    if (!(total > 0) || !(room > 0)) return clampPps(ZOOM_DEFAULT);
    return clampPps(Math.max(1, room - Math.max(16, room * .04)) / total);
}
// Another scale that keeps the time under `anchor` (pixels from the view's left edge) where it is, as far
// as the canvas allows. {pps, left, at}: the scale, the new left edge (seconds) and the anchored time.
function zoomAround(pps, left, next, anchor, total) {
    const scale = clampPps(next);
    const at = left + anchor / pps;
    return { pps: scale, left: clamp(at - anchor / scale, 0, leftLimit(total)), at };
}
// Ctrl+wheel (a trackpad pinch arrives as one) as a zoom factor: a mouse notch (about 100) zooms about
// 1.4x, the small steps of a pinch follow the fingers.
function wheelFactor(delta) {
    const size = Math.min(Math.abs(delta), 160);
    const power = size <= 30 ? size * .008 : .24 + (size - 30) * .0012;
    return Math.exp(delta < 0 ? power : -power);
}
// A scrub or a dragged clip held at the view's right edge moves the view on until the project's end is in view,
// `margin` pixels clear of the edge (not on into the empty time after the end).
const endInView = (left, pps, room, total, margin) => left + (room - margin) / pps >= total;
// The ruler's labelled step: labels at least `gap` pixels apart; whole frames when zoomed in.
const rulerStep = (pps, gap = 64) =>
    RULER_STEPS.find(value => value * pps >= gap) || RULER_STEPS[RULER_STEPS.length - 1];
// The left edge that shows the playhead: during playback the view turns a page when the playhead leaves
// it; a jump puts the playhead a little right of the left edge. The left edge stays when it is in view.
function followLeft(left, pps, room, time, total, paging) {
    const span = room / pps;
    if (paging) {
        if (time >= left && time <= left + span - 4 / pps) return left;
        return clamp(time - Math.min(40, room * .05) / pps, 0, leftLimit(total));
    }
    if (time >= left && time <= left + span - Math.min(24, room / 4) / pps) return left;
    return clamp(time - span * .15, 0, leftLimit(total));
}
"""


# The track: scrubbing, selection, reordering, edge trims, and the view (zoom and scroll).
TRACK_JS = ZOOM_MATH_JS + r"""
const page = element.closest('#timeline-page');
const LABEL_GAP = 64, EDGE = 36, STORE = 'contsy.timeline.views';
const emit = (name, detail = {}) => page?.dispatchEvent(new CustomEvent('timeline:' + name, { detail }));
// The view: the scale (pps, pixels per second) and the time at its left edge (kept in seconds, so that
// rounding the scroll position never moves it). Both stay across redraws, refreshes, edits and resizes and
// are remembered per saved project (views, and the browser's storage). restoring: the next layout sets
// the scroll position from leftTime instead of reading it (after a project change, or while the page was
// hidden, which loses it). viewWidth: the width of the view at the last layout.
let pps = 0, leftTime = 0, lastScroll = 0, viewWidth = 0, restoring = true, needsFit = true, settled = false;
let total = 0, time = 0, playing = false, shape = '';
let project = null, selectedIndex = -1, scrub = null, trim = null, pendingTrim = null, suppressUntil = 0;
// {index, until}: the selection this track asked the server for last (a click, or the scene its trim or drop
// selects), until its redraw arrives. Its answer and older ones leave the view and the playhead where they
// are; it never holds a click back (a refused one may be retried).
let asked = null;
let drag = null, dropTarget = null, scrubFrame = 0, scrubX = null, tickFrame = 0, drawn = null;
// pointerX: where a scrub, a trim or a dragged clip is (lastOver: the dragged clip's last dragover event)
let pointerX = null, lastOver = 0, edgeFrame = 0, viewFrame = 0, wantZoom = null, wantScroll = 0, glide = null;
let saveTimer = 0;
let strip = null, scroll = null, ruler = null, rail = null, blocks = [];
const views = new Map(), unsaved = new Map();  // project id -> {pps, left}

const pad = value => String(value).padStart(2, '0');
function timecode(seconds) {
    const frames = Math.max(0, Math.floor(seconds * FPS + 1e-6));
    const whole = Math.floor(frames / FPS);
    return pad(Math.floor(whole / 3600)) + ':' + pad(Math.floor(whole / 60) % 60) + ':' + pad(whole % 60) + ':' +
        pad(frames % FPS);
}
function rulerLabel(seconds, fine) {
    const frames = Math.round(seconds * FPS);
    const whole = Math.floor(frames / FPS);
    const text = (whole >= 3600 ? Math.floor(whole / 3600) + ':' : '') + pad(Math.floor(whole / 60) % 60) + ':' +
        pad(whole % 60);
    return fine ? text + ':' + pad(frames % FPS) : text;
}
function refs() {
    strip = element.querySelector('.timeline-strip');
    scroll = element.querySelector('.timeline-scroll');
    ruler = element.querySelector('.timeline-ruler');
    rail = element.querySelector('.timeline-rail');
    blocks = Array.from(element.querySelectorAll('.timeline-clip[data-scene]'));
    return Boolean(strip && scroll && ruler);
}
const railWidth = () => (rail ? rail.offsetWidth : 0);
const room = () => Math.max(0, scroll.clientWidth - railWidth());
const maxSeconds = () => Number(strip?.dataset.maxSeconds) || 30;
function lengths() {
    const seconds = blocks.map(block => Number(block.dataset.end) - Number(block.dataset.start));
    const live = trim || (pendingTrim && performance.now() < pendingTrim.until ? pendingTrim : null);
    if (live && live.index < seconds.length) seconds[live.index] = live.value;
    return { seconds, live };
}

// ----- the view: remembered per project -----
function stored() {
    try {
        const data = JSON.parse(window.localStorage.getItem(STORE) || '{}');
        return data && typeof data === 'object' && !Array.isArray(data) ? data : {};
    } catch {
        return {};  // no storage (private window, blocked site data): the view is kept for this page only
    }
}
// A project that was never saved has no id to remember its view by: it opens zoomed to fit.
function recall(id) {
    const known = id ? views.get(id) || stored()[id] : null;
    if (!known || !(Number(known.pps) > 0)) return null;
    return { pps: clampPps(Number(known.pps)), left: Math.max(0, Number(known.left) || 0) };
}
function remember() {
    if (!project || !pps) return;
    const view = { pps, left: leftTime };
    views.set(project, view);
    unsaved.set(project, view);
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
        try {
            const data = stored();
            for (const [id, item] of unsaved) {
                data[id] = { pps: Math.round(item.pps * 1000) / 1000, left: Math.round(item.left * 1000) / 1000,
                             at: Date.now() };
            }
            const ids = Object.keys(data).sort((a, b) => (Number(data[b]?.at) || 0) - (Number(data[a]?.at) || 0));
            for (const old of ids.slice(40)) delete data[old];
            window.localStorage.setItem(STORE, JSON.stringify(data));
        } catch { /* the views are kept for this page only */ }
        unsaved.clear();
    }, 400);
}

// ----- geometry -----
// The scale, the project's length and the canvas width are set on the component's own element: a redraw
// patches only what is inside it, so the canvas keeps its width through a refresh and the browser never
// clamps the scroll position before the watch callback runs.
function geometry(seconds, space) {
    element.style.setProperty('--tl-pps', pps + 'px');
    element.style.setProperty('--tl-length', total * pps + 'px');
    element.style.setProperty('--tl-canvas', canvasWidth(total, pps, space) + 'px');
    strip.style.setProperty('--scene-columns', seconds.map(value => value * pps + 'px').join(' '));
}
function showLeft() {
    leftTime = clamp(leftTime, 0, leftLimit(total));
    const target = leftTime * pps;
    if (Math.abs(scroll.scrollLeft - target) > .5) scroll.scrollLeft = target;
    lastScroll = scroll.scrollLeft;
}
// A scroll made elsewhere (the scroll bar, a swipe, the browser clamping a shorter canvas) moves the left
// edge; one this script made is already in leftTime.
function noteScroll() {
    if (!scroll || !pps || restoring || Math.abs(scroll.scrollLeft - lastScroll) <= 1) return false;
    lastScroll = scroll.scrollLeft;
    leftTime = clamp(lastScroll / pps, 0, leftLimit(total));
    return true;
}
function scrollBy(pixels) {
    if (!pps) return false;
    noteScroll();
    const before = scroll.scrollLeft;
    leftTime = (before + pixels) / pps;
    showLeft();
    if (Math.abs(scroll.scrollLeft - before) < .5) return false;
    drawRuler();
    remember();
    held();
    return true;
}
// After the view moved under a pointer that is dragging, the trimmed edge or the playhead follows it.
function held() {
    if (pointerX === null) return;
    if (trim) moveTrim(pointerX);
    else if (scrub) seekTo(timeAt(pointerX), true, false);
}
function layout() {
    if (!refs()) return false;
    const { seconds } = lengths();
    total = seconds.reduce((sum, value) => sum + value, 0) || Number(strip.dataset.total) || 0;
    const space = room();
    if (!space) { restoring = true; return false; }  // hidden: laid out when it is shown
    viewWidth = space;
    noteScroll();
    if (needsFit || !pps) {
        pps = fitPps(total, space);
        leftTime = 0;
        // Until the page has settled after loading, the next layout fits again: the first one may have
        // measured a window that was still being laid out.
        needsFit = !settled;
        remember();
    }
    geometry(seconds, space);
    showLeft();
    restoring = false;
    drawRuler(true);
    decorate(seconds);
    return true;
}
// Clip labels and filmstrips follow each clip's width (known from the scale: no layout read per clip). A
// filmstrip shows as many frames as fit, each the frame nearest its time; frames repeat when zoomed in.
function decorate(seconds) {
    const height = blocks[0]?.clientHeight || 60;
    const each = Math.max(28, height * (Number(strip.dataset.aspect) || 16 / 9));
    blocks.forEach((block, index) => {
        const width = (seconds[index] || 0) * pps;  // a clip fills its column (no margin)
        block.toggleAttribute('data-narrow', width < 96);
        block.toggleAttribute('data-tiny', width < 44);
        filmstrip(block, width, each);
    });
}
function slotOf(image, slot) {
    const value = slot === undefined ? '' : String(slot);
    if (image.style.order !== value) image.style.order = value;
}
function filmstrip(block, width, each) {
    const box = block.querySelector('.timeline-clip-frames');
    if (!box) return;
    const images = Array.from(box.querySelectorAll('img'));
    const own = images.filter(image => !image.hasAttribute('data-copy'));
    const copies = images.filter(image => image.hasAttribute('data-copy'));
    const slots = own.length ? clamp(Math.round(width / each), 1, 120) : 0;
    const first = new Map(), extra = [];
    for (let slot = 0; slot < slots; slot++) {
        const frame = Math.min(own.length - 1, Math.floor((slot + .5) * own.length / slots));
        if (first.has(frame)) extra.push([slot, frame]);
        else first.set(frame, slot);
    }
    own.forEach((image, frame) => {
        image.hidden = !first.has(frame);
        slotOf(image, first.get(frame));
    });
    copies.slice(extra.length).forEach(copy => copy.remove());
    extra.forEach(([slot, frame], index) => {
        const source = own[frame];
        let copy = copies[index];
        if (!copy) {
            copy = source.cloneNode(false);
            copy.setAttribute('data-copy', '');
            box.append(copy);
        } else if (copy.getAttribute('src') !== source.getAttribute('src')) {
            copy.setAttribute('src', source.getAttribute('src'));
        }
        copy.hidden = false;
        slotOf(copy, slot);
    });
}
// Ticks are drawn for the view and one view's width on each side; scrolling inside that range draws nothing.
function drawRuler(force = false) {
    if (!ruler || !scroll || !pps) return;
    const width = room();
    if (!width) return;
    const end = canvasWidth(total, pps, width) / pps;
    const from = scroll.scrollLeft / pps, to = from + width / pps;
    if (!force && drawn && drawn.pps === pps && drawn.end === end && from >= drawn.from && to <= drawn.to) return;
    const step = rulerStep(pps, LABEL_GAP);
    const fine = step < 1;
    const divisions = [10, 6, 5, 4, 3, 2, 1].find(count =>
        (step / count) * pps >= 7 && (!fine || Math.abs(step * FPS / count - Math.round(step * FPS / count)) < 1e-6))
        || 1;
    const minor = step / divisions;
    const first = Math.max(0, Math.floor((from - width / pps) / minor));
    const last = Math.min(Math.floor(end / minor + 1e-6), Math.ceil((to + width / pps) / minor));
    drawn = { pps, end, from: first * minor, to: Math.min(end, (last + 1) * minor) };
    const fragment = document.createDocumentFragment();
    for (let index = first; index <= last; index++) {
        const at = index * minor;
        const tick = document.createElement('span');
        const major = index % divisions === 0;
        tick.className = major ? 'timeline-tick major' : 'timeline-tick';
        tick.style.left = (at * pps) + 'px';
        if (major) {
            const label = document.createElement('span');
            label.textContent = rulerLabel(at, fine);
            tick.append(label);
        }
        fragment.append(tick);
    }
    ruler.replaceChildren(fragment);
    element.style.setProperty('--tl-grid', (step * pps) + 'px');
}
function announce() {
    if (!pps) return;
    emit('zoomed', { level: zoomLevel(pps), pps, span: viewWidth / pps, min: pps <= ZOOM_MIN * 1.0001,
                     max: pps >= ZOOM_MAX * .9999 });
}
function attribute(node, name, value) {
    if (node.getAttribute(name) !== value) node.setAttribute(name, value);
}
function place(follow = false, paging = false) {
    if (!strip || !ruler) return;
    time = clamp(time, 0, total);
    element.style.setProperty('--playhead', time * pps + 'px');
    attribute(ruler, 'aria-valuenow', time.toFixed(2));
    attribute(ruler, 'aria-valuetext', timecode(time));
    attribute(ruler, 'aria-valuemax', String(total));
    if (!follow || !scroll || !pps || !viewWidth) return;
    const next = followLeft(leftTime, pps, viewWidth, time, total, paging);
    if (Math.abs(next - leftTime) < 1e-9) return;
    leftTime = next;
    showLeft();
    drawRuler();
    remember();
}
function data() {
    const set = strip.dataset;
    return {
        project: set.project || '', total: Number(set.total) || 0, view: set.view || '',
        aspect: Number(set.aspect) || 16 / 9, hint: set.programHint || '',
        program: set.programLabel ? { src: set.program || '', label: set.programLabel, detail: set.programDetail || '',
            current: set.programCurrent === 'true', kind: set.programKind || '',
            pending: set.programPending === 'true' } : null,
        scenes: blocks.map(block => ({
            index: Number(block.dataset.scene), start: Number(block.dataset.start), end: Number(block.dataset.end),
            src: block.dataset.src || '', pending: block.hasAttribute('data-pending'), note: block.dataset.note || '',
            status: block.dataset.status || '', text: block.dataset.slate || '', detail: block.dataset.detail || '',
            title: 'Scene ' + (Number(block.dataset.scene) + 1) })),
    };
}
function seekTo(value, scrubbing, follow = !scrubbing) {
    time = clamp(value, 0, total);
    place(follow);
    emit('seek', { time, scrub: scrubbing });
}
function sync(fromServer = false) {
    if (!refs()) return;
    const id = strip.dataset.project || '';
    const outline = blocks.map(block => Number(block.dataset.end) - Number(block.dataset.start)).join(',');
    if (id !== project) {
        // Another project opens as it was left, or zoomed to fit the first time; a new project that was
        // just saved (it gets an id no view is remembered for, its scenes stay) keeps its view. A saved
        // project with a remembered view always opens as it was left, whatever its scene lengths.
        const saved = project === '' && id !== '' && outline === shape && !recall(id);
        if (!saved) {
            // a drag still held while another project opens is let go
            if (trim) endTrim(true);
            scrub = null;
            endDrag();
            glide = null;
            wantZoom = null;
            wantScroll = 0;
            time = 0;
            selectedIndex = -1;
            pendingTrim = null;
            asked = null;
            const known = recall(id);
            pps = known ? known.pps : pps;
            leftTime = known ? known.left : 0;
            needsFit = !known;
            restoring = true;
        }
        project = id;
        if (saved) remember();
    }
    shape = outline;
    if (fromServer && pendingTrim) {
        // A refresh may still show the old length; the edit's own redraw shows the new one.
        const block = blocks[pendingTrim.index];
        if (!block || Number(block.dataset.end) - Number(block.dataset.start) === pendingTrim.value) pendingTrim = null;
    }
    if (!layout()) { emit('data', data()); return; }
    if (trim) {
        // a redraw (the 3 s refresh) during a long trim drops what the trim shows on its clip
        trim.block.setAttribute('draggable', 'false');
        trim.block.setAttribute('data-trimming', '');
        tip();
    }
    const selected = blocks.find(block => block.getAttribute('aria-pressed') === 'true');
    const index = selected ? Number(selected.dataset.scene) : -1;
    const waiting = asked && performance.now() < asked.until;
    if (waiting && asked.index === index) asked = null;
    if (index !== selectedIndex) {
        const previous = selectedIndex;
        selectedIndex = index;
        const start = selected ? Number(selected.dataset.start) : 0, end = selected ? Number(selected.dataset.end) : 0;
        // A scene chosen elsewhere (bin, toolbar, a new scene) comes into the viewer. A click here
        // already moved the playhead; an older answer while a newer click waits is passed over.
        const elsewhere = !waiting && selected && previous !== -1;
        if (elsewhere && !(time >= start && time < end)) {
            time = start;
            emit('seek', { time, scrub: false });
        }
        place(Boolean(elsewhere));
    } else place(false);
    emit('data', data());
    announce();
}

// ----- zoom and scroll: applied in animation frames, never waiting for the server. Small steps (a pinch, a
// trackpad swipe, the slider) are applied at the next frame; a mouse notch or a zoom step glides there over
// a few frames (a third of the rest each frame), its anchor held in place all the way. -----
function focusPoint(space) {
    // the buttons, the slider and the keys zoom around the playhead, or around the view's centre when the
    // playhead is out of view
    const x = (time - leftTime) * pps;
    return x >= 0 && x <= space ? x : space / 2;
}
function zoomTo(next, anchor = null) {
    const space = room();
    if (!space || !pps || trim) return false;
    noteScroll();
    const view = zoomAround(pps, leftTime, next, anchor === null ? focusPoint(space) : clamp(anchor, 0, space), total);
    if (view.pps === pps && Math.abs(view.left - leftTime) < 1e-9) return false;
    pps = view.pps;
    leftTime = view.left;
    layout();
    place(false);
    announce();
    remember();
    held();  // a zoom while scrubbing: the playhead stays under the pointer
    return true;
}
function fit() {
    if (!refs() || trim) return;  // the scale stays put while an edge is dragged
    glide = null;
    wantZoom = null;
    needsFit = true;
    if (layout()) {
        place(false);
        announce();
    }
}
function applyView() {
    viewFrame = 0;
    const zoom = wantZoom, move = wantScroll;
    wantZoom = null;
    wantScroll = 0;
    if (!refs() || !pps || !room()) { glide = null; return; }
    if (zoom) zoomTo(zoom.pps, zoom.anchor);
    if (move) scrollBy(move);
    if (glide) {
        const log = Math.abs(glide.log) < .004 ? glide.log : glide.log / 3;
        const pixels = Math.abs(glide.pixels) < 2 ? glide.pixels : glide.pixels / 3;
        glide.log -= log;
        glide.pixels -= pixels;
        if (log && !zoomTo(pps * Math.exp(log), glide.anchor)) glide.log = 0;  // at a limit, or trimming
        if (pixels && !scrollBy(pixels)) glide.pixels = 0;  // at the start or the end
        if (!glide.log && !glide.pixels) glide = null;
        else viewFrame = requestAnimationFrame(applyView);
    }
}
function frame() {
    if (!viewFrame) viewFrame = requestAnimationFrame(applyView);
}
function queueZoom(next, anchor = null) {
    if (!pps) return;
    wantZoom = { pps: clampPps(next), anchor };
    if (glide) glide.log = 0;  // a scale set directly (the slider) ends a zoom still gliding
    frame();
}
function zoomBy(factor, anchor = null, smooth = true) {
    if (!pps) return;
    if (!smooth) {
        wantZoom = { pps: clampPps((wantZoom ? wantZoom.pps : pps) * factor), anchor };
        frame();
        return;
    }
    glide = glide || { log: 0, pixels: 0, anchor: null };
    // the zoom still to come, within the range: steps past a limit are dropped, not saved up
    const ahead = clamp(Math.log(pps) + glide.log + Math.log(factor), Math.log(ZOOM_MIN), Math.log(ZOOM_MAX));
    glide.log = ahead - Math.log(pps);
    glide.anchor = anchor;
    frame();
}
function scrollView(pixels, smooth) {
    if (smooth) {
        glide = glide || { log: 0, pixels: 0, anchor: null };
        glide.pixels += pixels;
    } else wantScroll += pixels;
    frame();
}
// where the queued and gliding moves will leave the view's left edge, in pixels
const scrollAhead = () => scroll.scrollLeft + wantScroll + (glide ? glide.pixels : 0);

// ----- scrubbing, trimming, and the view following a drag held near its edges -----
const timeAt = clientX => clamp((clientX - strip.getBoundingClientRect().left) / pps, 0, total);
// Pixels a frame the view moves while a drag is held near its left or right edge (faster nearer the edge).
function edgeSpeed(clientX) {
    const box = scroll.getBoundingClientRect(), left = box.left + railWidth(), right = box.left + scroll.clientWidth;
    if (clientX > right - EDGE) return Math.min(1, (clientX - right + EDGE) / EDGE) * 20;
    if (clientX < left + EDGE) return -Math.min(1, (left + EDGE - clientX) / EDGE) * 20;
    return 0;
}
// The view moves only where the drag can still go: a trim until the clip's shortest or longest length, a scrub
// or a dragged clip until the project's end is in view (not on into the empty time after it).
function canFollow(direction) {
    if (trim) return direction > 0 ? trim.value < maxSeconds() : trim.value > 1;
    if (direction > 0) return !endInView(leftTime, pps, viewWidth, total, EDGE);
    return leftTime > 0;
}
// A dragged clip held still gets a dragover event every few hundred milliseconds (or more often); one that
// left the timeline gets none.
const dragging = () => Boolean(scrub || trim || (drag !== null && performance.now() - lastOver < 500));
function follow() {
    if (edgeFrame || pointerX === null || !dragging()) return;
    const speed = edgeSpeed(pointerX);
    if (!speed || !canFollow(speed)) return;
    edgeFrame = requestAnimationFrame(function step() {
        edgeFrame = 0;
        // a dragged clip that left the timeline sends no more dragover events: the view stops
        if (pointerX === null || !dragging() || !refs()) return;
        const speed = edgeSpeed(pointerX);
        if (!speed || !canFollow(speed) || !scrollBy(speed)) return;
        edgeFrame = requestAnimationFrame(step);
    });
}
function stopFollowing() {
    pointerX = null;
    if (edgeFrame) { cancelAnimationFrame(edgeFrame); edgeFrame = 0; }
}
function scheduleScrub(clientX) {
    scrubX = clientX;
    if (scrubFrame) return;
    scrubFrame = requestAnimationFrame(() => {
        scrubFrame = 0;
        if (scrubX === null || !scrub) return;
        seekTo(timeAt(scrubX), true, false);
    });
}
function tip() {
    const box = element.querySelector('.timeline-trim-tip');
    if (!box || !trim) return;
    const rect = trim.block.getBoundingClientRect(), base = strip.getBoundingClientRect();
    box.hidden = false;
    box.textContent = trim.value + ' s' + (trim.value !== trim.from ? '  (was ' + trim.from + ' s)' : '');
    box.style.left = (rect.right - base.left) + 'px';
}
function startTrim(event, handle) {
    const block = handle.closest('.timeline-clip[data-scene]');
    if (!block || !pps) return;
    event.preventDefault();
    event.stopPropagation();
    const seconds = Number(block.dataset.end) - Number(block.dataset.start);
    // the length follows the pointer's travel over the canvas, the view's own moves included
    trim = { id: event.pointerId, index: Number(block.dataset.scene), block, x: event.clientX,
             left: scroll.scrollLeft, from: seconds, value: seconds };
    block.setAttribute('draggable', 'false');
    block.setAttribute('data-trimming', '');
    try { handle.setPointerCapture(event.pointerId); } catch { /* the pointer is gone */ }
    tip();
}
function moveTrim(clientX) {
    if (Math.abs(clientX - trim.x) > 3) trim.moved = true;
    const travel = clientX - trim.x + scroll.scrollLeft - trim.left;
    const value = clamp(Math.round(trim.from + travel / pps), 1, maxSeconds());
    if (value !== trim.value) {
        trim.value = value;
        layout();
        place(false);
    }
    tip();
}
function endTrim(cancel) {
    const done = trim;
    trim = null;
    stopFollowing();
    done.block.setAttribute('draggable', 'true');
    done.block.removeAttribute('data-trimming');
    const box = element.querySelector('.timeline-trim-tip');
    if (box) box.hidden = true;
    // The click that follows a drag of the edge is swallowed; a plain click on the edge selects the clip.
    if (done.moved || done.value !== done.from) suppressUntil = performance.now() + 400;
    if (!cancel && done.value !== done.from) {
        // Kept on screen until the server's redraw carries the new length; a refused change
        // (no redraw) goes back after 5 s.
        const pending = pendingTrim = { index: done.index, value: done.value, until: performance.now() + 5000 };
        // The answer selects the trimmed scene. It is this track's own edit, like its own click: the
        // view and the playhead stay where the user is working.
        asked = { index: done.index, until: performance.now() + 4000 };
        trigger('click', { trim: done.index, seconds: done.value });
        setTimeout(() => {
            if (pendingTrim !== pending) return;
            pendingTrim = null;
            if (layout()) place(false);
        }, 5100);
    }
    layout();
    place(false);
}
element.addEventListener('pointerdown', event => {
    if (event.button !== 0 || !refs() || !pps) return;
    const target = event.target;
    const handle = target.closest('.timeline-trim');
    if (handle) {
        startTrim(event, handle);
        if (trim) pointerX = event.clientX;
        return;
    }
    const onRuler = target.closest('.timeline-ruler, .timeline-playhead-head');
    const onLane = !onRuler && target.closest('.timeline-strip') && !target.closest('button, input, select, textarea, a');
    if (!onRuler && !onLane) return;
    event.preventDefault();
    scrub = { id: event.pointerId };
    pointerX = event.clientX;
    try { (onRuler ? ruler : strip).setPointerCapture(event.pointerId); } catch { /* the pointer is gone */ }
    ruler.focus({ preventScroll: true });
    emit('scrub', {});
    seekTo(timeAt(event.clientX), true, false);
});
element.addEventListener('pointermove', event => {
    if (trim && event.pointerId === trim.id) {
        pointerX = event.clientX;
        moveTrim(event.clientX);
        follow();
    } else if (scrub && event.pointerId === scrub.id) {
        pointerX = event.clientX;
        scheduleScrub(event.clientX);
        follow();
    }
});
function release(event, cancel) {
    if (trim && event.pointerId === trim.id) { endTrim(cancel); return; }
    if (!scrub || event.pointerId !== scrub.id) return;
    scrub = null;
    scrubX = null;
    stopFollowing();
    if (scrubFrame) { cancelAnimationFrame(scrubFrame); scrubFrame = 0; }
    seekTo(cancel ? time : timeAt(event.clientX), false, false);
}
element.addEventListener('pointerup', event => release(event, false));
element.addEventListener('pointercancel', event => release(event, true));
window.addEventListener('keydown', event => {
    if (trim && event.key === 'Escape') { event.preventDefault(); endTrim(true); }
}, true);
function selectScene(index, at = null) {
    const block = blocks[index];
    if (!block) return;
    const start = Number(block.dataset.start), end = Number(block.dataset.end);
    const target = Number.isFinite(at) ? at : (time >= start && time < end ? time : start);
    seekTo(target, false, false);  // a click never scrolls the view the user is looking at
    // Sent unless the server already shows this scene and no other click waits for its answer: the
    // listener's trigger mode is "multiple" and the server applies the clicks in order (the last wins).
    const waiting = asked && performance.now() < asked.until;
    if (index !== selectedIndex || (waiting && asked.index !== index)) {
        asked = { index, until: performance.now() + 4000 };
        trigger('click', { index });
    }
}
element.addEventListener('click', event => {
    if (!refs()) return;
    if (performance.now() < suppressUntil) { event.preventDefault(); return; }
    const target = event.target;
    // Dialogue blocks, background bars and effect markers select their scene at their time.
    const mark = target.closest('[data-line-scene], [data-seek-scene]');
    if (mark) {
        selectScene(Number(mark.dataset.lineScene ?? mark.dataset.seekScene),
                    mark.dataset.time ? Number(mark.dataset.time) : null);
        return;
    }
    if (target.closest('[data-joined]')) {
        emit('mode', { mode: 'program' });
        if (event.detail) seekTo(timeAt(event.clientX), false, false);
        trigger('click', { joined: true });
        return;
    }
    const block = target.closest('.timeline-clip[data-scene]');
    if (block) selectScene(Number(block.dataset.scene));
});

// ----- drag a clip to reorder: onto a clip, or past the last one (the empty time after the end) -----
function clearDrop() {
    dropTarget?.removeAttribute('data-drop');
    dropTarget = null;
}
function dropOn(event) {
    const block = event.target.closest?.('.timeline-clip[data-scene]');
    if (block || !event.target.closest?.('.timeline-track') || !blocks.length) return block || null;
    const x = event.clientX;
    const under = blocks.find(item => {
        const box = item.getBoundingClientRect();
        return x >= box.left - 1 && x <= box.right + 1;
    });
    const last = blocks[blocks.length - 1];
    return under || (x > last.getBoundingClientRect().right ? last : null);
}
function endDrag() {
    clearDrop();
    drag = null;
    stopFollowing();
}
element.addEventListener('dragstart', event => {
    const block = event.target.closest?.('.timeline-clip[data-scene]');
    if (!block || trim) { if (trim) event.preventDefault(); return; }
    drag = Number(block.dataset.scene);
    event.dataTransfer.setData('text/plain', block.dataset.scene);
    event.dataTransfer.effectAllowed = 'move';
});
element.addEventListener('dragover', event => {
    if (drag === null) return;
    // a dragged clip held near the view's edge moves the view, while it is held there
    pointerX = event.clientX;
    lastOver = performance.now();
    follow();
    const block = dropOn(event);
    if (!block) { clearDrop(); return; }
    event.preventDefault();
    if (block !== dropTarget) clearDrop();
    dropTarget = block;
    const bounds = block.getBoundingClientRect();
    block.dataset.drop = event.clientX < bounds.left + bounds.width / 2 ? 'before' : 'after';
    event.dataTransfer.dropEffect = 'move';
});
// A clip can be dropped wherever dragover would take it, also on an element the view has just moved under the
// pointer (the drop may come before its first dragover).
element.addEventListener('dragenter', event => {
    if (drag !== null && dropOn(event)) event.preventDefault();
});
element.addEventListener('dragleave', event => {
    // out of the timeline's view: the view stops following the dragged clip
    const box = drag !== null && scroll ? scroll.getBoundingClientRect() : null;
    if (box && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top ||
                event.clientY > box.bottom)) stopFollowing();
});
element.addEventListener('drop', event => {
    if (drag === null || !dropTarget || dropOn(event) !== dropTarget) return;
    event.preventDefault();
    const slot = Number(dropTarget.dataset.scene) + (dropTarget.dataset.drop === 'after' ? 1 : 0);
    if (slot !== drag && slot !== drag + 1) {
        // The answer selects the moved scene at its new place (the server's "drop" order): this track's own
        // edit, so the view and the playhead stay.
        asked = { index: slot - (slot > drag ? 1 : 0), until: performance.now() + 4000 };
        trigger('click', { from: drag, slot });
    }
    endDrag();
});
element.addEventListener('dragend', endDrag);

// ----- zoom bar, wheel and keys -----
page?.addEventListener('timeline:zoom', event => {
    if (!element.isConnected || !refs()) return;
    const detail = event.detail || {};
    if (detail.query) announce();
    else if (detail.action === 'fit') fit();
    else if (detail.action === 'in') zoomBy(ZOOM_STEP);
    else if (detail.action === 'out') zoomBy(1 / ZOOM_STEP);
    else if (Number.isFinite(detail.level)) queueZoom(levelPps(detail.level));
});
const panel = () => element.closest('#timeline-timeline') || element;  // the toolbar, the zoom bar and the track
// The wheel anywhere over the timeline panel: Ctrl+wheel or a trackpad pinch zooms the timeline, never the
// page, around the pointer over the track and around the playhead (or the view's centre) over the toolbar
// and the zoom bar. Over the track the wheel moves along the timeline (a vertical turn too, Shift+wheel and
// sideways swipes); over the lane names a vertical turn scrolls the lanes up and down when they do not all
// fit. At the start or the end a vertical turn scrolls the page instead; a sideways swipe never goes back in
// the browser's history.
panel().addEventListener('wheel', event => {
    const target = event.target;
    if (!element.isConnected || !(target instanceof Element) || !refs() || !pps) return;
    const onTrack = element.contains(target) && Boolean(target.closest('.timeline-deck'));
    const unit = event.deltaMode === 1 ? 33 : event.deltaMode === 2 ? Math.max(300, room()) : 1;
    if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        const delta = event.deltaY * unit;
        const anchor = onTrack ? event.clientX - scroll.getBoundingClientRect().left - railWidth() : null;
        zoomBy(wheelFactor(delta), anchor, Math.abs(delta) >= 50);
        return;
    }
    if (!onTrack) return;
    let dx = event.deltaX * unit, dy = event.deltaY * unit;
    if (event.shiftKey && !dx) { dx = dy; dy = 0; }
    const sideways = Math.abs(dx) >= Math.abs(dy);
    // the lane names: the browser scrolls the lanes (at their top or bottom, the page)
    if (!sideways && target.closest('.timeline-rail') && scroll.scrollHeight > scroll.clientHeight + 1) return;
    const amount = sideways ? dx : dy;
    if (!amount) return;
    const at = scrollAhead();
    if ((amount < 0 && at <= .5) || (amount > 0 && at >= scroll.scrollWidth - scroll.clientWidth - .5)) {
        if (sideways) event.preventDefault();
        return;
    }
    event.preventDefault();
    scrollView(amount, Math.abs(amount) >= 50);
}, { passive: false });
// Keys while the focus is in the timeline (never while typing in a field): + and − (Ctrl or Cmd with = and
// - too, instead of the browser's zoom) zoom around the playhead, Shift+Z zooms to fit.
page?.addEventListener('keydown', event => {
    if (!element.isConnected || event.defaultPrevented || event.altKey || !refs() || !pps) return;
    const target = event.target;
    if (!(target instanceof Element) || !panel().contains(target)) return;
    const field = target.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])');
    if (field && !(field.tagName === 'INPUT' && field.type === 'range')) return;
    const key = event.key, command = event.ctrlKey || event.metaKey;
    if (key === '+' || key === '=' || key === 'Add') zoomBy(ZOOM_STEP);
    else if (key === '-' || key === '_' || key === 'Subtract' || key === '−') zoomBy(1 / ZOOM_STEP);
    else if (!command && event.shiftKey && (key === 'Z' || key === 'z')) fit();
    else return;
    event.preventDefault();
});
element.addEventListener('scroll', event => {
    if (!scroll || event.target !== scroll) return;
    if (noteScroll()) {
        remember();
        held();
    }
    if (!tickFrame) tickFrame = requestAnimationFrame(() => { tickFrame = 0; drawRuler(); });
}, true);
page?.addEventListener('timeline:time', event => {
    if (!element.isConnected || !strip || scrub || trim) return;
    const detail = event.detail || {};
    time = clamp(Number(detail.time) || 0, 0, total);
    playing = Boolean(detail.playing);
    place(playing || Boolean(detail.reveal), playing);
});
page?.addEventListener('timeline:query', () => {
    if (element.isConnected && refs()) emit('data', data());
});
watch('value', () => sync(true));
new ResizeObserver(() => sync()).observe(element);
sync();
requestAnimationFrame(() => {
    settled = true;
    sync();
});
"""


VIEWER_JS = r"""
const page = element.closest('#timeline-page');
const root = element.querySelector('[data-viewer]');
const stage = root?.querySelector('[data-stage]');
const screen = root?.querySelector('[data-screen]');
const slate = root?.querySelector('[data-slate]');
const slateKicker = root?.querySelector('[data-slate-kicker]');
const slateText = root?.querySelector('[data-slate-text]');
const slateNote = root?.querySelector('[data-slate-note]');
const caption = root?.querySelector('[data-caption]');
const nowLabel = root?.querySelector('[data-now]');
const totalLabel = root?.querySelector('[data-total]');
const playButton = root?.querySelector('[data-step="play"]');
const sceneButton = root?.querySelector('[data-mode="scene"]');
const programButton = root?.querySelector('[data-mode="program"]');
const programHint = root?.querySelector('[data-program-hint]');
const fullButton = root?.querySelector('[data-fullscreen]');
const FPS = 30, FRAME = 1 / FPS, POOL = 3;
const CONTENT = '.timeline-clip, .timeline-audio-segment, .timeline-bed, .timeline-effect, .timeline-joined-clip';
const NOTES = {
    Generating: 'Generating this scene…', Queued: 'Queued for generation.',
    Failed: 'The last generation of this scene failed.', Stopped: 'Generation was stopped.',
    Empty: 'Describe the picture and motion in the inspector.',
};
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const emit = (name, detail = {}) => page?.dispatchEvent(new CustomEvent('timeline:' + name, { detail }));
const pad = value => String(value).padStart(2, '0');
let data = { project: null, total: 0, view: '', hint: '', program: null, scenes: [], aspect: 16 / 9 };
let requested = 'auto', serverView = null, mode = 'scene';
let time = 0, playing = false, current = -1, active = null, slateClock = null, loop = 0;
let shown = slate, waiting = null, revealTimer = 0, wanted = null, seekFrame = 0, spaceDown = false, stageClick = 0;
const pool = [];
let program = null;

function timecode(seconds) {
    const frames = Math.max(0, Math.floor(seconds * FPS + 1e-6));
    const whole = Math.floor(frames / FPS);
    return pad(Math.floor(whole / 3600)) + ':' + pad(Math.floor(whole / 60) % 60) + ':' + pad(whole % 60) + ':' +
        pad(frames % FPS);
}
const span = scene => {
    const minutes = value => pad(Math.floor(value / 60)) + ':' + pad(Math.floor(value % 60));
    return minutes(scene.start) + '–' + minutes(scene.end);
};
function clock() {
    if (nowLabel) nowLabel.textContent = timecode(time);
    if (totalLabel) totalLabel.textContent = timecode(data.total || 0);
}
function makeVideo(kind) {
    const video = document.createElement('video');
    video.className = 'tl-video';
    video.preload = 'auto';
    video.playsInline = true;
    video.setAttribute('playsinline', '');
    video.setAttribute('aria-hidden', 'true');
    video.setAttribute('tabindex', '-1');
    video.dataset.kind = kind;
    for (const name of ['loadedmetadata', 'loadeddata', 'seeked', 'canplay']) {
        video.addEventListener(name, () => settle(video));
    }
    video.addEventListener('error', () => failed(video));
    screen?.append(video);
    return video;
}
function load(video, src) {
    if ((video.dataset.src || '') === src) return;
    video.pause();
    video.dataset.src = src;
    delete video.dataset.seek;
    delete video.dataset.failed;
    if (src) video.src = src;
    else { video.removeAttribute('src'); video.load(); }
}
function distance(video) {
    const index = Number(video.dataset.scene);
    return Number.isFinite(index) ? Math.abs(index - current) : 1e9;
}
function clipVideo(index) {
    const scene = data.scenes[index];
    if (!scene || !scene.src) return null;
    let video = pool.find(item => item.dataset.src === scene.src);
    if (!video) {
        if (pool.length < POOL) {
            video = makeVideo('scene');
            pool.push(video);
        } else {
            const free = pool.filter(item => item !== shown && item !== active && item !== waiting);
            video = (free.length ? free : pool).sort((a, b) => distance(b) - distance(a))[0];
        }
        load(video, scene.src);
    }
    video.dataset.scene = String(index);
    return video;
}
function programVideo(src) {
    if (!program) program = makeVideo('program');
    load(program, src);
    return program;
}
function seek(video, at) {
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : null;
    const target = Math.max(0, duration ? Math.min(at, duration - .001) : at);
    if (video.readyState < 1 || video.seeking) { video.dataset.seek = String(target); return; }
    delete video.dataset.seek;
    if (Math.abs(video.currentTime - target) > .0005) video.currentTime = target;
}
function settle(video) {
    if (video.dataset.seek !== undefined && video.readyState >= 1 && !video.seeking) {
        const at = Number(video.dataset.seek);
        delete video.dataset.seek;
        if (Math.abs(video.currentTime - at) > .0005) { video.currentTime = at; return; }
    }
    if (video === waiting && video.readyState >= 2 && !video.seeking) display(video);
}
function unplayable() {
    showSlate(mode === 'scene' ? data.scenes[current] : null,
              'Could not load this ' + (mode === 'program' ? 'movie' : 'clip') + '.');
}
function failed(video) {
    if (!video.dataset.src) return;
    video.dataset.failed = '';
    if (video === waiting || video === shown || video === active) unplayable();
}
function display(item) {
    waiting = null;
    clearTimeout(revealTimer);
    if (shown === item) return;
    for (const video of pool.concat(program ? [program] : [])) {
        if (video !== item) video.removeAttribute('data-active');
    }
    slate?.toggleAttribute('data-active', item === slate);
    if (item && item !== slate) item.setAttribute('data-active', '');
    shown = item;
}
function reveal(video) {
    // A clip that failed earlier (e.g. while it was preloaded as a neighbour) shows its slate at once.
    if (video.hasAttribute('data-failed')) { unplayable(); return; }
    if (video.readyState >= 2 && !video.seeking) { display(video); return; }
    waiting = video;
    clearTimeout(revealTimer);
    revealTimer = setTimeout(() => { if (waiting === video) display(video); }, 350);
}
function showSlate(scene, note = '') {
    if (!slate) return;
    if (scene) {
        if (slateKicker) slateKicker.textContent = scene.title + ' · ' + (scene.end - scene.start) + ' s' +
            (scene.status ? ' · ' + scene.status : '');
        if (slateText) slateText.textContent = scene.text || 'No picture description yet.';
        // scene.note: the server's reason a clip cannot be shown (missing or changed on disk)
        if (slateNote) slateNote.textContent = note || scene.note || (scene.pending ? 'Preparing the preview…' :
            NOTES[scene.status] || 'No clip yet. Generate this scene to see it here.');
    } else {
        if (slateKicker) slateKicker.textContent = mode === 'program' ? 'Program' : 'Viewer';
        if (slateText) slateText.textContent = note || 'Add a scene to start.';
        if (slateNote) slateNote.textContent = '';
    }
    display(slate);
}
function sceneAt(at) {
    const scenes = data.scenes;
    for (let index = 0; index < scenes.length; index++) if (at < scenes[index].end) return index;
    return scenes.length - 1;
}
function describe() {
    if (!caption) return;
    if (mode === 'program' && data.program) {
        caption.textContent = data.program.label + (data.program.detail ? ' · ' + data.program.detail : '');
        return;
    }
    const scene = data.scenes[current];
    caption.textContent = scene ? scene.title + ' · ' + span(scene) + (scene.status ? ' · ' + scene.status : '') +
        (scene.detail ? ' · ' + scene.detail : '') : '';
}
function preload(index) {
    for (const near of [index + 1, index - 1]) {
        const video = clipVideo(near);
        if (!video || video === shown || video === waiting) continue;
        const scene = data.scenes[near];
        seek(video, near > index ? 0 : Math.max(0, scene.end - scene.start - FRAME));
    }
}
function render(at) {
    time = clamp(at, 0, data.total || 0);
    if (mode === 'program') {
        const info = data.program;
        if (!info) showSlate(null, data.hint || 'No program yet.');
        else if (!info.src) {
            // pending: its copy for the viewer is being made; otherwise the movie cannot be played and
            // its detail says why (e.g. "Join and check again to restore the preview.")
            showSlate(null, info.pending ? 'Preparing the program…' : (info.detail || 'Could not load this movie.'));
        } else {
            const video = programVideo(info.src);
            seek(video, time);
            reveal(video);
        }
    } else {
        const index = sceneAt(time);
        current = index;
        const scene = data.scenes[index];
        if (!scene) showSlate(null);
        else if (scene.src) {
            const video = clipVideo(index);
            seek(video, time - scene.start);
            reveal(video);
            preload(index);
        } else showSlate(scene);
    }
    describe();
    clock();
}
function publish(reveal = false) { emit('time', { time, playing, reveal }); }
function syncPlay() {
    if (!playButton) return;
    playButton.toggleAttribute('data-playing', playing);
    playButton.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    playButton.title = playing ? 'Pause (Space)' : 'Play (Space)';
}
function stopPlayback() {
    playing = false;
    cancelAnimationFrame(loop);
    for (const video of pool) video.pause();
    program?.pause();
    active = null;
    slateClock = null;
    syncPlay();
}
function enter(index, at) {
    const scene = data.scenes[index];
    for (const video of pool) if (!video.paused) video.pause();
    current = index;
    if (!scene) { stopPlayback(); return; }
    if (scene.src) {
        const video = clipVideo(index);
        active = video;
        slateClock = null;
        seek(video, at - scene.start);
        reveal(video);
        video.play().catch(() => {});
    } else {
        active = null;
        slateClock = { base: at, started: performance.now() };
        showSlate(scene);
    }
    preload(index);
    describe();
}
function advance() {
    const next = current + 1;
    if (next >= data.scenes.length) {
        time = data.total;
        stopPlayback();
        clock();
        publish();
        return false;
    }
    time = data.scenes[next].start;
    enter(next, time);
    return true;
}
function step() {
    if (!playing) return;
    let at = time;
    if (mode === 'program') {
        if (!program) { stopPlayback(); publish(); return; }
        // While a seek is still being applied the video's time is the old one: keep the clock.
        const settling = program.seeking || program.dataset.seek !== undefined;
        at = settling ? time : program.currentTime;
        if (!settling && (program.ended || at >= data.total - .001)) {
            time = data.total;
            stopPlayback();
            clock();
            publish();
            return;
        }
    } else {
        const scene = data.scenes[current];
        if (!scene) { stopPlayback(); publish(); return; }
        const length = scene.end - scene.start;
        if (active && active.hasAttribute('data-failed')) {
            // A clip that cannot load plays as its slate for the scene's length.
            slateClock = { base: time, started: performance.now() };
            active = null;
        }
        if (active) {
            const end = Number.isFinite(active.duration) && active.duration > 0 ? Math.min(active.duration, length) : length;
            const settling = active.seeking || active.dataset.seek !== undefined;
            at = settling ? time : scene.start + Math.min(active.currentTime, length);
            if (!settling && (active.ended || active.currentTime >= end - .01)) {
                if (!advance()) return;
                at = time;
            }
        } else if (slateClock) {
            at = slateClock.base + (performance.now() - slateClock.started) / 1000;
            if (at >= scene.end) {
                if (!advance()) return;
                at = time;
            }
        }
    }
    time = clamp(at, 0, data.total);
    clock();
    publish();
    loop = requestAnimationFrame(step);
}
function play() {
    if (playing || !data.total || !data.scenes.length) return;
    if (time >= data.total - FRAME / 2) time = 0;
    if (mode === 'program') {
        if (!data.program || !data.program.src) return;
        const video = programVideo(data.program.src);
        playing = true;
        seek(video, time);
        reveal(video);
        video.play().catch(() => { stopPlayback(); publish(); });
    } else {
        playing = true;
        enter(sceneAt(time), time);
    }
    syncPlay();
    cancelAnimationFrame(loop);
    loop = requestAnimationFrame(step);
}
function holdClick() {  // forget a click on the picture that has not played or paused yet
    clearTimeout(stageClick);
    stageClick = 0;
}
function toggle() {
    holdClick();
    if (playing) { stopPlayback(); publish(); }
    else play();
}
function jump(at) {
    holdClick();
    if (playing) stopPlayback();
    render(at);
    publish(true);
}
const frameOf = value => Math.round(value * FPS) / FPS;
function boundary(direction) {
    const edges = [0].concat(data.scenes.map(scene => scene.end));
    const target = direction < 0 ? edges.filter(edge => edge < time - FRAME / 2).pop()
                                 : edges.find(edge => edge > time + FRAME / 2);
    if (target !== undefined) jump(target);
}
function effective() {
    if (!data.program) return 'scene';
    if (requested === 'program') return 'program';
    if (requested === 'auto' && data.program.current) return 'program';
    return 'scene';
}
function syncModes() {
    sceneButton?.setAttribute('aria-pressed', String(mode === 'scene'));
    programButton?.setAttribute('aria-pressed', String(mode === 'program'));
    programButton?.setAttribute('aria-disabled', String(!data.program));
    const hint = data.program ? data.program.label + (data.program.detail ? ' · ' + data.program.detail : '')
                              : (data.hint || 'There is no program to watch yet.');
    if (programHint) programHint.textContent = data.program ? '' : hint;
    if (programButton) programButton.title = hint;
}
function applyMode() {
    const next = effective();
    if (next !== mode) {
        stopPlayback();
        mode = next;
        if (next === 'scene') program?.pause();
        render(time);
        publish();
    }
    syncModes();
}
function writeView(value) {
    const input = page?.querySelector('#timeline-view textarea, #timeline-view input');
    if (!input || input.value === value) return;
    input.value = value;
    input.dispatchEvent(new Event('input', { bubbles: true }));
}
function choose(value, user = false) {
    if (value === 'program' && !data.program) {
        if (caption) caption.textContent = data.hint || 'There is no program to watch yet.';
        return;
    }
    requested = value;
    if (user) writeView(value);
    applyMode();
}
function update(next) {
    if (!next || !Array.isArray(next.scenes)) return;
    const changedProject = next.project !== data.project;
    const before = data;
    data = next;
    if (root) root.style.setProperty('--tl-aspect', String(next.aspect || 16 / 9));
    if (changedProject) {
        stopPlayback();
        time = 0;
        current = -1;
        serverView = next.view || '';
        requested = serverView || 'auto';
        for (const video of pool) load(video, '');
        if (program) load(program, '');
        mode = effective();
        syncModes();
        render(0);
        publish();
        return;
    }
    if ((next.view || '') !== serverView) {
        serverView = next.view || '';
        requested = serverView || 'auto';
    } else if (requested === 'program' && before.program?.current && !next.program?.current) {
        requested = 'auto';  // an edit left the program behind: show the scenes being edited
    }
    const sources = new Set(next.scenes.map(scene => scene.src).filter(Boolean));
    for (const video of pool) {
        if (video.dataset.src && !sources.has(video.dataset.src) && video !== active && video !== shown) load(video, '');
    }
    time = clamp(time, 0, next.total || 0);
    const wantedMode = effective();
    if (wantedMode !== mode) { applyMode(); return; }
    syncModes();
    if (!playing) { render(time); return; }
    if (mode === 'scene') {
        const scene = next.scenes[current];
        if (!scene || (active ? active.dataset.src !== scene.src : Boolean(scene.src))) enter(sceneAt(time), time);
    } else if (!next.program || !next.program.src) {
        stopPlayback();
        render(time);
        publish();
    } else if (program && program.dataset.src !== next.program.src) {
        const video = programVideo(next.program.src);
        seek(video, time);
        reveal(video);
        video.play().catch(() => { stopPlayback(); publish(); });
    }
    describe();
    clock();
}
function scheduleSeek() {
    if (seekFrame) return;
    seekFrame = requestAnimationFrame(() => {
        seekFrame = 0;
        if (wanted === null) return;
        const at = wanted;
        wanted = null;
        render(at);
    });
}
function fullscreen() {
    if (document.fullscreenElement) document.exitFullscreen?.();
    else stage?.requestFullscreen?.().catch(() => {});
}
document.addEventListener('fullscreenchange', () => {
    const on = document.fullscreenElement === stage;
    fullButton?.setAttribute('aria-pressed', String(on));
    fullButton?.setAttribute('aria-label', on ? 'Exit full screen' : 'Full screen');
});
root?.addEventListener('click', event => {
    const button = event.target.closest('[data-step]');
    if (button) {
        const action = button.dataset.step;
        if (action === 'play') toggle();
        else if (action === 'frame-back') jump(frameOf(time - FRAME));
        else if (action === 'frame-forward') jump(frameOf(time + FRAME));
        else if (action === 'scene-back') boundary(-1);
        else if (action === 'scene-forward') boundary(1);
        return;
    }
    const choice = event.target.closest('[data-mode]');
    if (choice) { choose(choice.dataset.mode, true); return; }
    if (event.target.closest('[data-fullscreen]')) { fullscreen(); return; }
    if (event.target.closest('[data-stage]')) {
        // A click on the picture plays or pauses once it is clear that no second click follows: a
        // double-click is for full screen and leaves playback as it was.
        holdClick();
        if (event.detail > 1) return;
        stageClick = setTimeout(toggle, 250);
    }
});
stage?.addEventListener('dblclick', event => {
    event.preventDefault();
    holdClick();
    fullscreen();
});
page?.addEventListener('timeline:data', event => { if (element.isConnected) update(event.detail); });
page?.addEventListener('timeline:seek', event => {
    if (!element.isConnected) return;
    const detail = event.detail || {};
    holdClick();
    if (playing) { stopPlayback(); publish(); }
    const at = Number(detail.time) || 0;
    if (detail.scrub) { wanted = at; scheduleSeek(); return; }
    wanted = null;
    render(at);
});
page?.addEventListener('timeline:scrub', () => {
    if (!element.isConnected) return;
    holdClick();
    if (playing) { stopPlayback(); publish(); }
});
page?.addEventListener('timeline:mode', event => {
    if (element.isConnected && event.detail?.mode) choose(event.detail.mode);
});
// Keys work while the focus is in the viewer or the timeline, never while typing in a field.
page?.addEventListener('keydown', event => {
    if (!element.isConnected || event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
    const target = event.target;
    if (!(target instanceof Element) || !target.closest('#timeline-viewer, #timeline-timeline')) return;
    if (target.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])')) return;
    const second = event.shiftKey ? 1 : FRAME;
    let handled = true;
    switch (event.key) {
        case ' ':
        case 'Spacebar':
            if (target.closest('button') && !target.closest(CONTENT)) return;  // the button's own action
            spaceDown = true;
            toggle();
            break;
        case 'ArrowLeft': jump(frameOf(time - second)); break;
        case 'ArrowRight': jump(frameOf(time + second)); break;
        case 'Home': jump(0); break;
        case 'End': jump(data.total || 0); break;
        case 'ArrowUp': boundary(-1); break;
        case 'ArrowDown': boundary(1); break;
        default: handled = false;
    }
    if (handled) event.preventDefault();
});
page?.addEventListener('keyup', event => {
    if ((event.key === ' ' || event.key === 'Spacebar') && spaceDown) {
        spaceDown = false;
        event.preventDefault();
    }
});
// The page hides when another page of the app is chosen: stop the sound with it.
if (page) new MutationObserver(() => {
    if (playing && !page.getClientRects().length) { stopPlayback(); publish(); }
}).observe(page, { attributes: true, attributeFilter: ['style', 'class', 'hidden'] });
syncPlay();
syncModes();
clock();
emit('query');
"""
