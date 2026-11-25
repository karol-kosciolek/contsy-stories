# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Timeline editing and project actions in the local Studio panel.

The page is an editor workspace (layout, scripts and styles in app/ui_timeline_view.py): an app bar
with the project and the production actions, a bin (Scenes, Cast, Assets), the viewer, the scene
inspector and a full-width timeline. The inspector has four tabs: Picture (picture and motion,
duration, cast, reference images), Dialogue (spoken lines with a speaker and an entry time), Sound
(background, timed effects, mix levels, the sound check and previews) and Takes (earlier clips of
the scene). The viewer scrubs and plays in the browser: the track's HTML lists a browser URL for
every scene clip and for the program, and the browser seeks without asking the server.

The editor refreshes every few seconds, so anything that reads media or project files is remembered
by file stamps or by key (see _Memo), and slow media work (checking and copying clips, the export
mix of a scene, filmstrips, waveforms) runs on background workers (ui_timeline_view.Background).
"""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock

import gradio as gr

from . import cast, characters, dialogue, items, queue_manager, refs, scene_parts, scene_sound, sound, timeline
from . import overlays as text_overlays
from . import ui_theme
from . import ui_timeline_view as tlv
from . import llm, style_library, styles, ui_assist, universes, video_gen, voices
from .ui_timeline_view import (ASSETS_JS, BIN_JS, CSS, REFERENCES_JS, REPORT_JS, ROWS_JS, SHELL_JS, SHOW_GALLERY_JS,
                               TRACK_JS, VIEWER_JS, ZOOM_JS, AssetItem, BinScene, CastCard, EffectRow, LineRow,
                               assets_html, bin_html, cached_file, cast_html, effects_editor, filmstrip, first_words,
                               issues_html, lines_editor, media_url, players_html, references_html, scene_action,
                               shell_html, viewer_html, wave_slice, wave_svg, zoom_html)

__all__ = ["CSS", "Draft", "build_timeline_tab", "track_html", "universe_choices"]


@dataclass(frozen=True)
class Draft:
    baseline: timeline.Project | None = None
    scenes: tuple[timeline.Scene, ...] = (timeline.Scene(),)
    selected: int = 0
    form: str = ""  # identifies the scene loaded into the line, effect and reference editors
    items: tuple[str, ...] | None = None  # the film's places and items as picked in the bin; None: as saved


_clock = tlv.clock


def _bounds(draft: Draft) -> tuple[int, int]:
    start = sum(scene.seconds for scene in draft.scenes[:draft.selected])
    return start, start + draft.scenes[draft.selected].seconds


class _Memo:
    """A small thread-safe cache for refresh helpers that read project or media files."""

    def __init__(self, size: int = 128, ttl: float | None = None):
        self._items: OrderedDict = OrderedDict()
        self._lock = Lock()
        self._size, self._ttl = size, ttl

    def __call__(self, key, compute):
        now = time.monotonic()
        with self._lock:
            hit = self._items.get(key)
            if hit is not None and (self._ttl is None or now - hit[0] < self._ttl):
                self._items.move_to_end(key)
                return hit[1]
        value = compute()
        with self._lock:
            self._items[key] = (now, value)
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return value

    def put(self, key, value) -> None:
        now = time.monotonic()
        with self._lock:
            self._items[key] = (now, value)
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)

    def peek(self, key, default=None):
        """The value remembered for `key` (nothing is computed)."""
        with self._lock:
            hit = self._items.get(key)
        if hit is None or (self._ttl is not None and time.monotonic() - hit[0] >= self._ttl):
            return default
        return hit[1]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


# Keys hold every input of a result; the time limits only bound how late a change made elsewhere
# (a voice sample, an installed model, a finished job) reaches the editor.
_VOICES = _Memo(4, ttl=10)
_LIBRARY = _Memo(4, ttl=5)
_ISSUES = _Memo(256, ttl=5)
_AUDITIONS = _Memo(256, ttl=5)
_MIXES = _Memo(64, ttl=10)
_TAKES = _Memo(64, ttl=5)
_MOVIES = _Memo(32, ttl=10)
_BACKGROUNDS = _Memo(128)
_ITEMS = _Memo(4, ttl=5)  # the usable places and items (app/items.py)
# The last viewer URL made for each clip (project, plan key, checksum): the viewer keeps playing it
# while a new one is made after the scene's levels or the Audio switch changed.
_SHOWN = _Memo(256)


def _forget() -> None:
    """Drop every remembered result (tests)."""
    for memo in (_VOICES, _LIBRARY, _ISSUES, _AUDITIONS, _MIXES, _TAKES, _MOVIES, _BACKGROUNDS, _SHOWN, _ITEMS):
        memo.clear()
    tlv.forget()


@dataclass(frozen=True)
class TrackDialogue:
    speaker: str
    text: str
    start: float | None = None
    end: float | None = None
    planned: bool = False  # estimated from the text before the line was synthesized


@dataclass(frozen=True)
class ClipVisual:
    images: tuple[str, ...] = ()
    cast_names: tuple[str, ...] = ()
    lines: tuple[TrackDialogue, ...] = ()
    alert: str = ""  # the first error that blocks generating this scene
    media: str = ""  # what the viewer plays for the scene (a browser URL)
    pending: bool = False  # the scene has a clip whose viewer copy is still being made
    note: str = ""  # the viewer slate's note when the clip cannot be shown
    detail: str = ""  # the viewer caption's extra: levels, how the clip continues
    voice_wave: tuple[int, ...] = ()  # waveform of the measured dialogue on the scene clock
    bed_wave: tuple[int, ...] = ()  # waveform of the background on the scene clock


@dataclass(frozen=True)
class Program:
    """What the viewer's Program mode plays: the export or the joined edit."""
    source: str = ""        # browser URL; "" while its copy is being made (pending) or when it cannot be played
    label: str = ""
    detail: str = ""        # without a source and not pending: why it cannot be played
    current: bool = False   # matches the edit on the timeline (and can be played)
    kind: str = ""          # "export", "edit" or "earlier"
    pending: bool = False   # its copy for the viewer is being made


def _dialogue_cell(lines: tuple[TrackDialogue, ...], seconds: int, start: int, index: int,
                   wave: tuple[int, ...] = ()) -> str:
    segments = []
    for number, line in enumerate(lines):
        timed = line.start is not None and line.end is not None
        planned = line.planned or not timed
        shape = ""
        if timed:
            begin = max(0.0, float(line.start))
            end = max(begin, float(line.end))
            late = end - seconds
            if begin >= seconds:
                # Drawn as a stub at the scene end so the line stays visible and clickable.
                offset, width = 0.92, 0.08
                when = f"starts at {begin:.1f} s, after the scene ends at {seconds} s, not heard"
            else:
                offset, width = begin / seconds, (min(end, seconds) - begin) / seconds
                when = f"{begin:.1f}–{end:.1f} s"
                if late > 0.05:
                    when += f", runs {late:.1f} s past the scene end and is cut there"
                if not planned:
                    shape = wave_svg(wave_slice(wave, begin, min(end, seconds)))
            when = ("planned " if planned else "") + when
            mark = f' data-time="{start + min(begin, seconds - 0.1):.3f}"'
            mark += " data-planned" * planned + " data-overflow" * (begin >= seconds or late > 0.05)
        else:
            offset, width = number / len(lines), 1 / len(lines)
            when, mark = "planned", " data-planned"
        title = ui_theme.esc(f"{line.speaker}: {line.text} ({when})")
        segments.append(
            f'<button type="button" class="timeline-audio-segment" data-line-scene="{index}"{mark} '
            f'style="--audio-start:{offset:.4%};--audio-width:{width:.4%}" title="{title}" aria-label="{title}">'
            f'{shape}<span class="timeline-audio-text">{ui_theme.esc(line.speaker + " · " + line.text)}</span></button>'
        )
    return f'<div class="timeline-audio-cell">{"".join(segments)}</div>'


def _sound_cell(scene: timeline.Scene, start: int, index: int, audio: bool, wave: tuple[int, ...] = ()) -> str:
    """Lane A2: the scene's background bar (with its waveform) and a marker per effect at its time."""
    if not audio:
        return '<div class="timeline-sound-cell"></div>'
    seconds = scene.seconds
    custom = " ".join(scene.background.split())
    title = ui_theme.esc(f"Scene {index + 1} background: " + (custom or "automatic, from the picture text"))
    parts = [f'<button type="button" class="timeline-bed" data-seek-scene="{index}" data-time="{start:.3f}"'
             + " data-custom" * bool(custom) + f' title="{title}" aria-label="{title}">'
             f'{wave_svg(wave)}<span class="timeline-bed-text">{ui_theme.esc(custom or "Auto")}</span></button>']
    for number, effect in enumerate(scene.effects, 1):
        position = min(effect.at, seconds * 0.94)
        width = max(min(effect.seconds, seconds - position), 0.0) / seconds
        if effect.at >= seconds:
            when = f"at {effect.at:g} s, after the scene ends at {seconds} s, not heard"
        else:
            when = f"at {effect.at:g} s for {effect.seconds:g} s"
            if effect.at + effect.seconds > seconds + 1e-6:
                when += ", cut at the scene end"
        overflow = effect.at + effect.seconds > seconds + 1e-6
        label = ui_theme.esc(f"Effect {number} in scene {index + 1}: {effect.text}, {when}")
        parts.append(f'<button type="button" class="timeline-effect" data-seek-scene="{index}" '
                     f'data-time="{start + min(effect.at, seconds - 0.1):.3f}"' + " data-overflow" * overflow
                     + f' style="--fx-start:{position / seconds:.4%};--fx-width:{width:.4%}" title="{label}" '
                     f'aria-label="{label}">{ui_theme.esc(effect.text)}</button>')
    return f'<div class="timeline-sound-cell">{"".join(parts)}</div>'


def track_html(draft: Draft, statuses: tuple[str, ...] = (), *, visuals: tuple[ClipVisual, ...] = (),
               joined: bool = False, view: str = "", audio: bool = True, program: Program | None = None,
               hint: str = "", aspect: str = "") -> str:
    """The timeline: ruler, V1 video, A1 dialogue, A2 sound and V2 (the program), with the data the
    viewer plays from (a media URL per scene clip, the program, the slate text of every scene).

    view: the viewer source the server asks for ("", "scene" or "program")."""
    clips, speech, sounds = [], [], []
    start = 0
    for index, scene in enumerate(draft.scenes):
        end = start + scene.seconds
        visual = visuals[index] if index < len(visuals) else ClipVisual()
        status = statuses[index] if index < len(statuses) else ("Draft" if scene.prompt else "Empty")
        prompt = " ".join(scene.prompt.split())
        summary = scene_action(scene.prompt)
        cast_names = ", ".join(visual.cast_names)
        label = ui_theme.esc(f"Scene {index + 1}, {_clock(start)} to {_clock(end)}, {status}. " + prompt
                             + (f" Cast: {cast_names}." if cast_names else "")
                             + (f" Problem: {ui_theme.plain(visual.alert)}" if visual.alert else ""))
        frames = ""
        if visual.images:
            frames = ('<span class="timeline-clip-frames" aria-hidden="true">'
                      + "".join(f'<img src="{ui_theme.esc(url)}" alt="" draggable="false" loading="lazy" '
                                f'decoding="async">' for url in visual.images) + "</span>")
        problem = ui_theme.plain(visual.alert) or ("The last generation of this scene failed." if status == "Failed"
                                                   else "")
        alert = (f'<span class="timeline-clip-alert" aria-hidden="true" title="{ui_theme.esc(problem)}">!</span>'
                 if problem else "")
        media = f' data-src="{ui_theme.esc(visual.media)}"' if visual.media else ""
        media += " data-pending" * (visual.pending and not visual.media)
        media += f' data-note="{ui_theme.esc(visual.note)}"' if visual.note else ""
        media += f' data-detail="{ui_theme.esc(visual.detail)}"' if visual.detail else ""
        clips.append(
            f'<button type="button" class="timeline-clip" draggable="true" data-scene="{index}"'
            + " data-frames" * bool(frames) + f' data-start="{start}" data-end="{end}" '
            f'data-status="{ui_theme.esc(status)}" data-cut="{str(scene.cut).lower()}"' + " data-alert" * bool(visual.alert)
            + f'{media} data-slate="{ui_theme.esc(prompt)}" aria-pressed="{str(index == draft.selected).lower()}" '
            f'aria-label="{label}" title="{label}">{frames}'
            f'<span class="timeline-clip-top"><strong>{index + 1:02d}</strong>'
            f'<span class="timeline-clip-label">{ui_theme.esc(first_words(summary or prompt, 60))}</span>{alert}'
            f'<span class="timeline-clip-status">{ui_theme.esc(status)}</span>'
            f'<span class="timeline-clip-duration">{scene.seconds}s</span>'
            f'<span class="timeline-mark" data-status="{ui_theme.esc(status)}" aria-hidden="true"></span></span>'
            f'<span class="timeline-clip-body"><span class="timeline-clip-title">'
            f'{ui_theme.esc(summary or "Write a scene description")}</span></span>'
            '<span class="timeline-clip-ring" aria-hidden="true"></span>'
            '<span class="timeline-trim" aria-hidden="true" title="Drag to change the length"></span></button>'
        )
        speech.append(_dialogue_cell(visual.lines if audio else (), scene.seconds, start, index, visual.voice_wave))
        sounds.append(_sound_cell(scene, start, index, audio, visual.bed_wave))
        start = end
    # Every scene's column is its length at the track's time scale (--tl-pps, set by the track's script on its
    # own element; 40 px a second until the script has measured the window). The lanes run on past the
    # project's end, so the columns never stretch to fill them.
    columns = " ".join(f"calc(var(--tl-pps, 40px) * {scene.seconds})" for scene in draft.scenes)
    project_id = ui_theme.esc(draft.baseline.id if draft.baseline else "")
    edit = ""
    if program and program.label and program.current:
        edit, kind = f"{program.label} · {_clock(start)}", program.kind
    elif joined:
        edit, kind = f"Joined edit · {_clock(start)}", "edit"
    lane = (f'<div class="timeline-assembly"><button type="button" class="timeline-joined-clip" data-joined '
            f'data-kind="{ui_theme.esc(kind)}" aria-label="{ui_theme.esc(edit)}. Watch it in the viewer." '
            f'title="Watch the whole film in the viewer">{ui_theme.esc(edit)}</button></div>' if edit else "")
    data = {"data-total": start, "data-project": project_id, "data-view": ui_theme.esc(view),
            "data-aspect": f"{video_gen.ASPECTS.get(aspect, 16 / 9):.4f}", "data-max-seconds": video_gen.MAX_SECONDS}
    if program and program.label:
        data.update({"data-program": ui_theme.esc(program.source), "data-program-label": ui_theme.esc(program.label),
                     "data-program-detail": ui_theme.esc(program.detail),
                     "data-program-current": str(program.current).lower(),
                     "data-program-kind": ui_theme.esc(program.kind)})
        if program.pending and not program.source:
            data["data-program-pending"] = "true"
    if hint:
        data["data-program-hint"] = ui_theme.esc(hint)
    attributes = " ".join(f'{name}="{value}"' for name, value in data.items())
    selected = _bounds(draft)[0]
    audio_hint = ("Dashed dialogue is planned, solid is measured; A2 shows each background and its effects."
                  if audio else "Project audio is off.")
    return (
        f'<div class="timeline-deck"' + " data-edit" * bool(lane) + '>'
        f'<div class="timeline-scroll" role="region" aria-label="Timeline, {_clock(start)} total">'
        '<div class="timeline-board"><div class="timeline-rail" aria-hidden="true">'
        f'<div class="timeline-ruler-label">Time</div>'
        f'<div class="timeline-video-label">{tlv.icon("video")}<small>V1</small> Video</div>'
        f'<div class="timeline-audio-label">{tlv.icon("dialogue")}<small>A1</small> Dialogue</div>'
        f'<div class="timeline-sound-label">{tlv.icon("sound")}<small>A2</small> Sound</div>'
        + (f'<div class="timeline-assembly-label">{tlv.icon("edit")}<small>V2</small> Edit</div>' if lane else "")
        + '</div>'
        f'<div class="timeline-strip" style="--scene-columns:{columns}" {attributes}>'
        f'<div class="timeline-ruler" role="slider" tabindex="0" aria-label="Playhead" aria-valuemin="0" '
        f'aria-valuemax="{start}" aria-valuenow="{selected}" aria-valuetext="{tlv.timecode(selected)}"></div>'
        f'<div class="timeline-track" role="group" aria-label="V1 video">{"".join(clips)}</div>'
        f'<div class="timeline-dialogue" role="group" aria-label="A1 dialogue">{"".join(speech)}</div>'
        f'<div class="timeline-sound" role="group" aria-label="A2 background and effects">{"".join(sounds)}</div>'
        f'{lane}<div class="timeline-end" aria-hidden="true"></div>'
        '<div class="timeline-playhead" aria-hidden="true"><span class="timeline-playhead-head"></span></div>'
        '<div class="timeline-trim-tip" role="status" aria-live="polite" hidden></div></div></div></div>'
        '<p class="timeline-deck-hint">Scroll to move along the timeline, Ctrl+scroll or + and − to zoom, Shift+Z to '
        'fit. Drag the ruler to scrub. Click a clip to edit it, drag it to reorder, drag its right edge to change '
        f'its length. {audio_hint}</p></div>'
    )


def _speaker_key(draft: Draft) -> str:
    baseline = draft.baseline
    return f"{baseline.id if baseline else 'new'}:{baseline.revision if baseline else 0}:{draft.selected}"


def _editor_rows(value, key: str) -> list | None:
    """Rows an editor wrote for the scene loaded under `key`, or None to keep the scene's parts.

    The editors write {"key": …, "rows": […]}; rows from another scene (an older key) are
    ignored. A plain JSON list is accepted for scripts and tests."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    data = value
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except ValueError as error:
            raise ValueError("Could not read the scene editor.") from error
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"] if data.get("key") == key else None
    raise ValueError("Could not read the scene editor.")


def _envelope(rows, key: str) -> str:
    return json.dumps({"key": key, "rows": list(rows)}, ensure_ascii=False)


def _parsed(rows: list, parse, noun: str) -> tuple:
    parts = []
    for number, row in enumerate(rows, 1):
        try:
            parts.extend(parse(json.dumps([row], ensure_ascii=False)))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{noun} {number}: {error}") from error
    return tuple(parts)


def _references(rows: list) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(rows))
    if any(not scene_parts.is_reference(name) for name in names):
        raise ValueError("A reference image is not available. Remove it and add it again.")
    if len(names) > scene_parts.MAX_REFERENCES:
        raise ValueError(f"Use at most {scene_parts.MAX_REFERENCES} reference images in a scene.")
    return names


def _scene_parts(draft: Draft, previous: timeline.Scene, lines, effects, background, references,
                 voice_db, background_db) -> dict:
    """The structured parts of the selected scene from the inspector; None keeps a part as it was."""
    parts = {"lines": previous.lines, "effects": previous.effects, "background": previous.background,
             "references": previous.references, "voice_db": previous.voice_db,
             "background_db": previous.background_db}
    if (rows := _editor_rows(lines, draft.form)) is not None:
        parts["lines"] = _parsed(rows, scene_parts.cues_from_json, "Line")
    if (rows := _editor_rows(effects, draft.form)) is not None:
        parts["effects"] = _parsed(rows, scene_parts.effects_from_json, "Effect")
    if (rows := _editor_rows(references, draft.form)) is not None:
        parts["references"] = _references(rows)
    if background is not None:
        text = " ".join(str(background).split())
        if len(text) > scene_parts.MAX_TEXT:
            raise ValueError(f"Keep the background description under {scene_parts.MAX_TEXT} characters.")
        parts["background"] = text
    for name, value, what in (("voice_db", voice_db, "Dialogue"), ("background_db", background_db, "Background")):
        if value is not None:
            try:
                parts[name] = scene_parts.level(value)
            except ValueError as error:
                raise ValueError(f"{what} level: {error}") from error
    return parts


def _edited(draft: Draft, description: str, seconds: int | float | None, cut: bool,
            exact: bool, selected: list[str] | None, assignments: str = "", lines=None, effects=None,
            background: str | None = None, references=None, voice_db: float | None = None,
            background_db: float | None = None) -> Draft:
    """The draft with the inspector's values applied to the selected scene.

    lines, effects and references are the editors' JSON; None (or an older call without them)
    keeps those parts of the scene unchanged."""
    if (not scene_parts.is_length(seconds) or not 1 <= seconds <= video_gen.MAX_SECONDS):
        raise ValueError(f"A scene lasts 1–{video_gen.MAX_SECONDS} s, in tenths of a second (for example 2.5).")
    seconds = scene_parts.length(seconds)  # a whole number stays an int: the scene keeps its keys
    index = draft.selected
    previous = draft.scenes[index]
    prompt = (description or "").strip()
    known = (*(previous.cast or ()), *(draft.baseline.cast_slugs if draft.baseline else ()))
    members = tuple(_character_slugs(selected, known)) if exact else None
    quotes = dialogue.quotes(prompt)
    # Retain assignments only where the same line still occupies the same slot.
    keep = tuple(slug if before == after and members and slug in members else ""
                 for slug, before, after in zip(previous.speakers, dialogue.quotes(previous.prompt),
                                                 quotes))
    if members and assignments:
        try:
            data = json.loads(assignments)
        except (TypeError, ValueError):
            data = None
        if isinstance(data, dict) and data.get("key") == _speaker_key(draft):
            pairs = data.get("assignments")
            if isinstance(pairs, list) and len(pairs) == len(quotes) and all(
                isinstance(pair, list) and len(pair) == 2 and pair[0] == line and isinstance(pair[1], str)
                for pair, line in zip(pairs, quotes)
            ):
                keep = tuple(slug if slug in members else "" for _, slug in pairs)
    speakers = _trim_speakers(keep)
    parts = _scene_parts(draft, previous, lines, effects, background, references, voice_db, background_db)
    scene = timeline.Scene(prompt, seconds, bool(cut and index), members, speakers, **parts)
    return replace(draft, scenes=draft.scenes[:index] + (scene,) + draft.scenes[index + 1:])


def _form_keys(*values) -> set[str]:
    """The form keys in the editors' hidden fields ({"key": …, "rows": […]}, see _editor_rows)."""
    keys = set()
    for value in values:
        data = value
        if isinstance(value, str):
            try:
                data = json.loads(value) if value.strip() else None
            except ValueError:
                continue
        if isinstance(data, dict) and isinstance(data.get("key"), str):
            keys.add(data["key"])
    return keys


def _current_form(draft: Draft, values) -> bool:
    """Whether the inspector's values (the form after the Draft) belong to the scene the draft has
    loaded into the inspector.

    The page reads the fields when it sends an event, the server reads the draft when it runs it. A
    click sent while the answer to an earlier one was on its way therefore carries the inspector as it
    was before that answer: the earlier event applied those values already, and they must not land on
    the scene it selected. The line, effect and reference fields carry the form key of the inspector's
    load; without any key (a new page) the values are taken as current."""
    keys = _form_keys(values[6], values[7], values[9])
    return not keys or draft.form in keys


def _apply_form(draft: Draft, values) -> Draft:
    """_edited() with the inspector's values, unless they belong to an earlier load of the inspector."""
    return _edited(draft, *values) if _current_form(draft, values) else draft


def _trim_speakers(slugs: tuple[str, ...]) -> tuple[str, ...]:
    return slugs[:next((i + 1 for i in range(len(slugs) - 1, -1, -1) if slugs[i]), 0)]


def _scene_order(draft: Draft, action: str, destination: tuple[int, int] | None = None) -> Draft:
    scenes = list(draft.scenes)
    index = draft.selected
    if action == "add":
        index += 1
        scenes.insert(index, timeline.Scene())
    elif action == "duplicate":
        scenes.insert(index + 1, scenes[index])
        index += 1
    elif action == "remove":
        scenes.pop(index)
        scenes = scenes or [timeline.Scene()]
        index = min(index, len(scenes) - 1)
    elif destination is not None:
        source, slot = destination
        index = slot - int(slot > source)
        scenes.insert(index, scenes.pop(source))
    else:
        step = -1 if action == "earlier" else 1
        moved = min(len(scenes) - 1, max(0, index + step))
        scenes.insert(moved, scenes.pop(index))
        index = moved
    if scenes[0].cut:
        scenes[0] = replace(scenes[0], cut=False)
    return replace(draft, scenes=tuple(scenes), selected=index)


def _trimmed(draft: Draft, index, seconds) -> Draft:
    """The draft after a scene's right edge was dragged on the track: scene `index` lasts
    `seconds` and is the selected scene (the inspector shows its new duration)."""
    if type(index) is not int or not 0 <= index < len(draft.scenes):
        raise ValueError("Choose a scene on the timeline.")
    if type(seconds) is not int or not 1 <= seconds <= video_gen.MAX_SECONDS:
        raise ValueError(f"A scene must last 1–{video_gen.MAX_SECONDS} whole seconds.")
    scenes = list(draft.scenes)
    scenes[index] = replace(scenes[index], seconds=seconds)
    return replace(draft, scenes=tuple(scenes), selected=index)


def _project_choices():
    return [(project.title, project.id) for project in timeline.projects()]


def _cast_choices(selected=()):
    choices = [(person["name"], person["slug"]) for person in characters.all_characters()]
    available = {slug for _, slug in choices}
    return choices + [(f"{slug} (unavailable)", slug) for slug in selected if slug not in available]


def _library() -> dict[str, str]:
    """Character slugs by slug, by slug in any case and by name in any case."""
    def build():
        people = characters.all_characters()
        lookup = {person["slug"]: person["slug"] for person in people}
        for person in people:
            lookup.setdefault(person["slug"].casefold(), person["slug"])
        for person in people:
            lookup.setdefault(" ".join(person["name"].split()).casefold(), person["slug"])
        return lookup
    return _LIBRARY("library", build)


def _character_slugs(values, known=()) -> list[str]:
    """Selected characters as slugs: a typed name or slug, in any letter case, selects that
    character. Values that match no character are kept so the user can see and remove them."""
    known = set(known)
    texts = [" ".join(value.split()) for value in values or () if isinstance(value, str) and value.strip()]
    lookup = _library() if any(text not in known for text in texts) else {}
    result = []
    for text in texts:
        slug = text if text in known else lookup.get(text) or lookup.get(text.casefold()) or text
        if slug not in result:
            result.append(slug)
    return result


def _typed_names(values):
    """Turn a name typed into a character picker (Enter) into that character's slug."""
    slugs = _character_slugs(values)
    return gr.skip() if slugs == list(values or ()) else gr.update(value=slugs)


def universe_choices(selected=""):
    choices = [("No universe", "")] + [(item["name"], item["slug"]) for item in universes.all_universes()]
    if selected and all(slug != selected for _, slug in choices):
        choices.append((f"{selected} (saved cast)", selected))
    return choices


_NEW_PROJECT = timeline.Project("", 0, "Untitled movie", Draft().scenes)
# What the viewer shows, kept by the server so an action can switch it (Export shows the program):
# "" picks the program when one matches the edit, else the scene clips.
_AUTO, _SCENE, _PROGRAM = "", "scene", "program"
_FORM_FIELDS = 12  # inspector values after the Draft: description … background level
_PROJECT_FIELDS = 7  # project values after them: name, aspect ratio, audio, universe, cast, subtitles, style
# Project → Delete project: the browser asks first, naming the project as the app bar shows it. The answer goes to the
# server ("yes" or "no"); a yes also closes the menu, so the message and the new project are in view.
DELETE_PROJECT_JS = """(name) => {
    const title = (name || '').trim();
    const yes = confirm('Move ' + (title ? '\u201c' + title + '\u201d' : 'this project') + ' to the Recycle Bin, '
                        + 'with its clips and takes? Exported movies stay in the Gallery. You can restore the project '
                        + 'from the Recycle Bin.');
    if (yes) document.querySelector('#timeline-project-menu .label-wrap.open')?.click();
    return yes ? 'yes' : 'no';
}"""
_SPEAKER_JS = """
const page = element.closest('#timeline-page');
let latest = null;
element.addEventListener('change', event => {
    if (!event.target.closest('select[data-line]')) return;
    const editor = element.querySelector('.timeline-speaker-editor');
    if (!editor || !page) return;
    latest = {
        key: editor.dataset.key,
        assignments: Array.from(editor.querySelectorAll('select[data-line]'), line => [line.dataset.text, line.value])
    };
    const input = page.querySelector('#timeline-line-assignments textarea, #timeline-line-assignments input');
    if (input) {
        input.value = JSON.stringify(latest);
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }
});
watch('value', () => {
    const editor = element.querySelector('.timeline-speaker-editor');
    if (!editor || !latest || editor.dataset.key !== latest.key) { latest = null; return; }
    const selects = Array.from(editor.querySelectorAll('select[data-line]'));
    if (selects.length !== latest.assignments.length ||
        selects.some((select, i) => select.dataset.text !== latest.assignments[i][0] ||
            !Array.from(select.options).some(option => option.value === latest.assignments[i][1]))) {
        latest = null;
        return;
    }
    selects.forEach((select, i) => { select.value = latest.assignments[i][1]; });
});
"""


def _candidate(draft: Draft, name, ratio, with_audio, universe, selected, subtitles=None,
               style=None) -> timeline.Project:
    """The project as the form has it now; subtitles or style None keeps the saved choice (an unknown style too:
    the save would refuse it)."""
    baseline = draft.baseline or _NEW_PROJECT
    slug = universe or ""
    group = baseline.universe_members
    if slug != baseline.universe:
        group = tuple((universes.get(slug) or {}).get("members", ()))
    return replace(baseline, title=(name or "").strip(), scenes=draft.scenes, aspect=ratio, audio=bool(with_audio),
                   universe=slug, universe_members=group,
                   characters=tuple(_character_slugs(selected, (*baseline.characters, *baseline.cast_slugs))),
                   subtitles=baseline.subtitles if subtitles is None else bool(subtitles),
                   style=style if styles.valid(style) else baseline.style,
                   items=baseline.items if draft.items is None else draft.items)


def _editor_values(draft: Draft) -> tuple:
    """Load the selected scene into the inspector. A new form key tells the editors' scripts that
    their rows were replaced by the server."""
    draft = replace(draft, form=uuid.uuid4().hex[:12])
    scene = draft.scenes[draft.selected]
    return (draft, scene.prompt, scene.seconds, gr.update(value=scene.cut, interactive=draft.selected > 0),
            scene.cast is not None, gr.update(value=list(scene.cast or ()), visible=scene.cast is not None), "",
            _envelope((cue.to_json() for cue in scene.lines), draft.form),
            _envelope((effect.to_json() for effect in scene.effects), draft.form),
            scene.background, _envelope(scene.references, draft.form), scene.voice_db, scene.background_db)


@dataclass(frozen=True)
class Preview:
    """What an exported movie is, for the viewer's Program mode (see _movie_view)."""
    label: str = ""
    detail: str = ""


@dataclass(frozen=True)
class Presentation:
    track: str = ""
    heading: str = ""                             # the inspector's header: scene, times, status, problem
    status: str = ""                              # the app bar: save state and project format
    bin: str = ""                                 # the bin's scene list
    cast: str = ""                                # the bin's Cast tab: characters, portraits, voices
    assets: str = ""                              # the bin's Assets tab: reference images
    report: str = ""
    speakers: str = ""
    cast_choices: tuple[tuple[str, str], ...] = ()
    lines: str = ""
    effects: str = ""
    references: str = ""
    picture_note: str = ""
    convert: bool = False
    background_hint: str = ""
    check: str = ""
    players: str = ""
    sound_for: tuple = ()                         # (project id, scene index) the players belong to
    sound_items: tuple[tuple[str, str], ...] = ()  # (label, URL) of the players
    mix: str = ""
    takes: tuple[tuple[str, str], ...] = ()
    take: str | None = None
    takes_note: str = ""
    download: str = ""
    queue_label: str = ""                         # "" hides Pause / Resume queue
    retry: str = ""                               # "" hides Retry scene N
    missing: bool = True                          # a scene has no clip: Generate missing clips has work
    item_choices: tuple[tuple[str, str], ...] = ()  # "Places and items in this scene": (name, slug) of every item
    item_values: tuple[str, ...] = ()             # the items whose main photo is one of the scene's reference images
    items_note: str = ""                          # the items the picture text names (timeline.named_items)
    film_items: str = ""                          # the bin's Items tab: the film's places and items
    edit_note: str = ""                           # the inspector's Edit tab: the take an AI edit starts from


def _media_stamp(path: Path) -> tuple:
    return tlv.stamp(path)


def _voices() -> tuple[voices.Voice, ...]:
    return _VOICES("voices", lambda: tuple(voices.all_voices()))


def _project_jobs(project: timeline.Project) -> list[queue_manager.Job]:
    if not project.id:
        return []
    kinds = set(timeline.JOB_KINDS)
    return [job for job in queue_manager.snapshot() if job.kind in kinds and job.params.get("project") == project.id]


def _scene_members(project: timeline.Project, scene: timeline.Scene, profiles: list[dict]) -> tuple[str, ...]:
    """Characters appearing in the scene: its own cast, or the characters its picture text names
    and its lines give a voice to."""
    if scene.cast is not None:
        return scene.cast
    slugs = [person["slug"] for person in cast.mentions(scene.prompt, profiles)]
    slugs += [cue.character for cue in scene.lines if cue.character]
    slugs = list(dict.fromkeys(slugs))
    if not slugs and not (project.universe or project.reference_characters):
        slugs = list(project.characters[:1])
    return tuple(slugs[:cast.MAX_CAST])


def _scene_issues(project: timeline.Project, index: int, stamp: tuple) -> tuple[scene_parts.Issue, ...]:
    """The scene's sound check, remembered per scene so editing one scene does not re-check all."""
    scene = project.scenes[index]
    key = (project.id, project.audio, project.cast_slugs, project.characters, project.universe, project.style, index,
           scene, stamp)

    def compute():
        try:
            return tuple(timeline.scene_issues(project, index))
        except (ValueError, OSError):
            return tuple(scene_parts.timing_issues(scene.lines, scene.effects, scene.seconds))
    return _ISSUES(key, compute)


def _audition(project: timeline.Project, index: int, stamp: tuple):
    """The scene's sound preview when it matches the scene as edited now, else None."""
    if not project.id or not project.audio:
        return None
    scene = project.scenes[index]
    key = (project.id, project.cast_slugs, project.characters, index, scene, stamp)

    def compute():
        try:
            return timeline.sound_preview(project, index)
        except (ValueError, OSError):
            return None
    return _AUDITIONS(key, compute)


def _measured(scene: timeline.Scene, audition, record: dict | None) -> list[tuple[float, float]] | None:
    """Line windows measured after synthesis, when they belong to the scene's current lines."""
    if audition is not None and len(audition.timings) == len(scene.lines):
        return [(float(start), float(end)) for start, end in audition.timings]
    timings = (record or {}).get("dialogue_timings") or []
    if scene.lines and len(timings) == len(scene.lines) and all(
            isinstance(line, dict) and line.get("text") == cue.text for line, cue in zip(timings, scene.lines)):
        windows = [(line.get("start"), line.get("end")) for line in timings]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
               for window in windows for value in window):
            return [(float(start), float(end)) for start, end in windows]
    return None


def _speaker_name(speaker: str, members: tuple[str, ...], names: dict, voice_names: dict) -> str:
    if speaker.startswith(scene_parts.VOICE_PREFIX):
        return voice_names.get(speaker[1:], speaker)
    if speaker:
        return names.get(speaker, speaker)
    return names.get(members[0], members[0]) if len(members) == 1 else "Speaker"


def _check_clip(project: timeline.Project, saved: timeline.Project, index: int, key: str) -> str:
    """Media worker: what the viewer plays for scene `index` (a browser URL, "" when unusable).

    The clip and its stems must still match their checksums. With separate stems the scene plays
    them mixed as the export mixes them at the scene's levels (timeline.leveled_clip, cached)."""
    try:
        record = timeline.asset(saved, key)
    except (ValueError, OSError):
        record = None
    if not record:
        return ""
    path = Path(record["path"])
    if project.audio and record.get("audio_sources"):
        try:
            mixed = timeline.leveled_clip(project, index)
        except (ValueError, OSError):
            mixed = None
        # The export mix of this very clip: its file name starts with the clip's checksum.
        if mixed and (Path(mixed) == path or Path(mixed).name.startswith(record["sha256"][:16])):
            path = Path(mixed)
    return tlv.url_for(cached_file(path))


def _viewer_clip(project: timeline.Project, saved: timeline.Project, index: int, key: str,
                 record: dict) -> tuple[str | None, str]:
    """(URL or None while it is being made, caption detail) of a scene that has a clip.

    While a new version is made for the same clip (its levels or the Audio switch changed, so the
    export mix is made again), the viewer keeps the last one and the caption says it is updating."""
    scene = project.scenes[index]
    sources = tuple(path for path in record.get("audio_sources") or () if path) if project.audio else ()
    levels = (scene.voice_db, scene.background_db) if sources else None
    key_parts = ("scene", saved.id, key, record["sha256"], _media_stamp(record["path"]),
                 tuple(_media_stamp(path) for path in sources), _media_stamp(record.get("effects_source")),
                 levels, project.audio)
    url = tlv.later(key_parts, _check_clip, project, saved, index, key)
    clip = (saved.id, key, record["sha256"])
    updating = False
    if url:
        _SHOWN.put(clip, url)
    elif url is None and (last := _SHOWN.peek(clip)):
        url, updating = last, True
    detail = []
    if levels and any(levels):
        detail.append(f"Dialogue {levels[0]:+g} dB, background {levels[1]:+g} dB")
    if record.get("transition_note"):
        detail.append(str(record["transition_note"]) if project.key == saved.key else "Saved clip")
    if updating:
        detail.append("Updating the preview…")
    return url, " · ".join(detail)


def _track_visuals(project: timeline.Project, saved: timeline.Project, keys: list,
                   stamp: tuple = (), selected: int = 0) -> tuple[ClipVisual, ...]:
    profiles = [person for slug in project.cast_slugs if (person := characters.get(slug))]
    names = {person["slug"]: person["name"] for person in profiles}
    legacy = project.audio and any(dialogue.quotes(scene.prompt) and not scene.lines for scene in project.scenes)
    voiced = project.audio and any(cue.voice for scene in project.scenes for cue in scene.lines)
    pool = list(_voices()) if legacy or voiced else []
    voice_names = {voice.slug: voice.name for voice in pool}
    tokens = voices.tokens(pool)
    people = cast.people(profiles, pool) if legacy else []
    records = {index: timeline.asset(saved, keys[index], verify=False)
               for index in range(len(project.scenes)) if index < len(keys) and keys[index]}
    # The viewer's copies are made nearest the selected scene first.
    media = {index: _viewer_clip(project, saved, index, keys[index], record)
             for index, record in sorted(records.items(), key=lambda item: abs(item[0] - selected)) if record}
    result = []
    for index, scene in enumerate(project.scenes):
        record = records.get(index)
        slugs = members = _scene_members(project, scene, profiles)
        images, lines, voice_wave, bed_wave = (), [], (), ()
        url, detail, note = (None, "", "")
        if record:
            slugs = record["characters"]
            images = filmstrip(_media_stamp(record["path"]), record["sha256"], scene.seconds)
            url, detail = media[index]
            if url == "":
                note = "The clip is missing or changed on disk. Generate this scene again."
            stems = record.get("audio_sources") or (None, None)
            if project.audio:
                background, voice = stems
                bed_wave = tlv.peaks(background, scene_sound.HANDLE, scene.seconds) if background else ()
                for line in record.get("dialogue_timings", ()):
                    start, end = line.get("start"), line.get("end")
                    if (not all(isinstance(x, (float, int)) and math.isfinite(x) for x in (start, end))
                            or not 0 <= start < min(end, scene.seconds)):
                        continue
                    slug, voice_slug = line.get("character"), line.get("voice")
                    speaker = names.get(slug, slug) or voice_names.get(voice_slug, voice_slug) or "Dialogue"
                    lines.append(TrackDialogue(speaker, line.get("text", ""), start, min(end, scene.seconds)))
                if lines and voice:
                    voice_wave = tlv.peaks(voice, 0.0, scene.seconds)
        elif index < len(keys) and keys[index]:
            note = "The clip is missing. Generate this scene again."
        if project.audio and not lines:
            heard = _audition(project, index, stamp)
            if heard is not None and heard.timings:
                lines = [TrackDialogue(person.name, text, start, end)
                         for person, text, (start, end) in zip(heard.speakers, heard.texts, heard.timings)]
                voice = getattr(heard, "voice", None)
                voice_wave = tlv.peaks(voice, 0.0, scene.seconds) if voice else ()
                if not bed_wave and getattr(heard, "background", None):
                    bed_wave = tlv.peaks(heard.background, scene_sound.HANDLE, scene.seconds)
            elif scene.lines:
                lines = [TrackDialogue(_speaker_name(cue.speaker, members, names, voice_names), cue.text,
                                       start, end, planned=True)
                         for cue, (start, end) in zip(scene.lines, scene_parts.timings(scene.lines))]
            elif dialogue.quotes(scene.prompt) and not record:
                present = [person for person in people if person[1] in slugs]
                _, _, parsed = dialogue.parse(refs.plain(scene.prompt), tokens, present)
                # A missing profile can be restored without preventing the rest of the edit from opening.
                if scene.speakers and all(not slug or slug in names for slug in scene.speakers):
                    parsed = dialogue.assign_speakers(parsed, scene.speakers, present)
                lines = [TrackDialogue(names.get(line.character) or voice_names.get(line.voice)
                                       or ("Woman" if line.speaker == "female" else "Man"), line.text)
                         for line in parsed]
        alert = next((issue.text for issue in _scene_issues(project, index, stamp)
                      if issue.level == scene_parts.ERROR), "")
        result.append(ClipVisual(images, tuple(names.get(slug, slug) for slug in slugs), tuple(lines), alert,
                                 media=url or "", pending=url is None and bool(record), note=note, detail=detail,
                                 voice_wave=tuple(voice_wave), bed_wave=tuple(bed_wave)))
    return tuple(result)


def _scene_choices(project: timeline.Project, scene: timeline.Scene) -> tuple[tuple[str, str], ...]:
    return tuple((person["name"] if (person := characters.get(slug)) else f"{slug} (unavailable)", slug)
                 for slug in dict.fromkeys((*project.cast_slugs, *(scene.cast or ()))))


def _speaker_editor(scene: timeline.Scene, choices: tuple[tuple[str, str], ...], key: str) -> str:
    """Speakers of quoted lines written in the picture text (scenes made before Dialogue lines)."""
    lines = dialogue.quotes(scene.prompt)
    if not lines:
        return ""
    lead = '<p class="timeline-hint">Quoted lines in the picture text:</p>'
    if scene.cast is None:
        return lead + ('<p class="section-sub">Speakers follow the description. Enable Choose characters on the '
                       'Picture tab to assign them here, or convert the quotes to dialogue lines.</p>')
    if not scene.cast:
        return lead + '<p class="section-sub">Add a character appearing in this scene to assign a line.</p>'
    rows = []
    for number, line in enumerate(lines):
        selected = scene.speakers[number] if number < len(scene.speakers) else ""
        options = []
        for label, slug in (("From description", ""), *choices):
            if slug and slug not in scene.cast:
                continue
            options.append(f'<option value="{ui_theme.esc(slug)}"' + (" selected" if slug == selected else "")
                           + f'>{ui_theme.esc(label)}</option>')
        rows.append(f'<label class="timeline-speaker-row"><span><span class="timeline-line-number">'
                    f'{number + 1:02d}</span> “{ui_theme.esc(line)}”</span>'
                    f'<select data-line="{number}" data-text="{ui_theme.esc(line)}" '
                    f'aria-label="Speaker for line {number + 1}">{"".join(options)}</select></label>')
    return lead + f'<div class="timeline-speaker-editor" data-key="{ui_theme.esc(key)}">' + "".join(rows) + '</div>'


def _clip_kinds() -> tuple[str, ...]:
    return timeline.PREPARE, timeline.PRODUCE, timeline.EDIT


def _scene_statuses(project: timeline.Project, saved: timeline.Project, keys: list,
                    job: queue_manager.Job | None, failed: dict | None = None,
                    stale: frozenset = frozenset()) -> tuple[str, ...]:
    """stale: scenes whose clip no longer fits their inputs (timeline.outdated, e.g. a character's photos
    changed); with a clip they show "Outdated" instead of "Ready"."""
    statuses = []
    active = {queue_manager.PENDING: "Queued", queue_manager.RUNNING: "Generating",
              queue_manager.ERROR: "Failed", queue_manager.STOPPED: "Stopped"}
    target = None
    if job and job.kind in _clip_kinds():
        target = job.params.get("scene_index")
        if target is None:
            target = next((i for i, key in enumerate(keys) if not key), len(keys))
    # A failed sound preview leaves the scene's clip as it was.
    failed_scene = (failed or {}).get("scene") if (failed or {}).get("kind") != timeline.SOUND else None
    for index, scene in enumerate(project.scenes):
        if index == target and job.status in active:
            status = active[job.status]
        elif index < len(keys) and keys[index]:
            status = "Outdated" if index in stale else "Ready"
        elif not scene.prompt:
            status = "Empty"
        elif index == failed_scene and index < len(saved.scenes) and scene == saved.scenes[index]:
            status = "Failed"
        elif index < len(saved.scenes) and scene == saved.scenes[index]:
            status = "Saved"
        else:
            status = "Modified" if saved.id else "Draft"
        statuses.append(status)
    return tuple(statuses)


def _movie_view(project: timeline.Project, saved: timeline.Project, movie: timeline.MoviePreview,
                invalid_duration: bool, export_subtitles: bool | None = None,
                export_overlays: str | None = None) -> Preview:
    """export_subtitles: whether the project's latest export has burned subtitles (None: not known). export_overlays: the
    key of the text laid over the project's latest export ('' for none; None: not known; app/overlays.py key)."""
    revision = movie.revision
    unsaved = project.key != saved.key or project.subtitles != saved.subtitles or invalid_duration
    if movie.current and movie.key == project.key and not unsaved:
        label = "Current export"
        detail = f"Matches the current edit · {_clock(project.seconds)}"
        if revision is not None:
            detail += f" · rendered from revision {revision}"
    else:
        label = "Earlier export"
        if revision is not None:
            detail = f"Rendered from revision {revision}. "
        else:
            detail = "Rendered before the current edits. "
        if unsaved:
            detail += "The current edit has unsaved changes."
        elif movie.key != saved.key:
            detail += f"No export matches revision {saved.revision}."
        elif export_subtitles is not None and export_subtitles != saved.subtitles:
            detail += ("Export again to burn in the subtitles." if saved.subtitles
                       else "Export again without the subtitles.")
        elif export_overlays is not None and export_overlays != text_overlays.key(saved.overlays):
            detail += "Export again to put the current text over the film."
        else:
            detail += "Export again to include the current clips."
    return Preview(label, detail)


def _movie(saved: timeline.Project, state: dict) -> timeline.MoviePreview | None:
    """The project's export (the one matching the saved edit, else the latest), checked every few
    seconds at most: looking for an older export reads the project's plans."""
    if not saved.id:
        return None
    merged = state.get("merged") if isinstance(state.get("merged"), dict) else {}
    key = (saved.id, saved.key, state.get("export"), state.get("export_revision"),
           tuple(state.get("export_assets") or ()), tuple(state["assets"]), merged.get("file"),
           state.get("export_merge"), saved.subtitles, bool(state.get("export_subtitles")),
           saved.overlays, state.get("export_overlays"))

    def compute():
        try:
            return timeline.movie_preview(saved, state)
        except (ValueError, OSError):
            return None
    return _MOVIES(key, compute)


def _check_joined(saved: timeline.Project, state: dict) -> str:
    """Media worker: the joined edit's browser URL after its checksum matched, else ""."""
    try:
        path = timeline.merged_clip(saved, state, verify=True)
    except (ValueError, OSError):
        path = None
    return tlv.url_for(cached_file(path)) if path else ""


def _program(current: timeline.Project, saved: timeline.Project, state: dict, current_edit: bool,
             joined: Path | None, invalid_duration: bool, keys: list) -> tuple[Program | None, str, str]:
    """(program, hint, download): what the viewer's Program mode plays (the export matching the edit,
    else the joined edit, else an earlier export), why there is none, and the movie to download.

    A program whose copy for the viewer is still being made is pending; one that cannot be played
    (its file is missing or changed) is offered with the reason, but is not current, so the viewer
    shows the scene clips unless the user picks Program."""
    movie = _movie(saved, state)
    export, download = None, ""
    if movie:
        shown = _movie_view(current, saved, movie, invalid_duration, bool(state.get("export_subtitles")),
                            str(state.get("export_overlays") or ""))
        copy = tlv.cached_later(movie.path, pending=None)  # None while the copy is made, "" without one
        download = copy or ""
        is_current = shown.label == "Current export"
        detail = shown.detail if copy != "" else "The movie file could not be read. Export the movie again."
        export = Program(tlv.url_for(download), shown.label, detail, is_current and copy != "",
                         "export" if is_current else "earlier", pending=copy is None)
        if export.current:
            return export, "", download
    if joined:
        source = tlv.later(("edit", saved.id, _media_stamp(joined)), _check_joined, saved, state)
        if source == "":
            program = Program("", "Joined edit", "Join and check again to restore the preview.", False, "edit")
            return program, "", download
        attention = len(state["merged"].get("attention", []))
        status = f"{attention} join{'s' if attention != 1 else ''} to review" if attention else "Ready to export"
        detail = f"{_clock(current.seconds)} · {status}"
        return Program(source or "", "Joined edit", detail, True, "edit", pending=source is None), "", download
    if export:
        return export, "", download
    if not saved.id:
        hint = "Save the project and generate its clips to watch the whole film here."
    elif not current_edit:
        hint = "Save the changes, then click Join and check to watch the whole film here."
    elif not timeline.all_clips(current, keys):
        hint = "Generate the missing clips, then click Join and check to watch the whole film here."
    else:
        hint = "Click Join and check to assemble the whole film and watch it here."
    return None, hint, download


def _failure(saved: timeline.Project) -> dict | None:
    if not saved.id:
        return None
    try:
        failed = timeline.failure(saved)
    except (ValueError, OSError):
        return None
    return failed if isinstance(failed, dict) else None


def _job_name(job: queue_manager.Job) -> str:
    return job.label.split(" · ")[0] or "Timeline job"


def _queue_line(project: timeline.Project, state: dict, job: queue_manager.Job | None, joined: bool,
                failed: dict | None) -> str:
    """The app bar's queue chip: this project's latest job, the queue state and a failure; the
    rest (clips ready, joins to review, the transition report) opens under Details."""
    details = []
    progress = ""
    kind = "idle"
    if job:
        name = _job_name(job)
        if job.status == queue_manager.RUNNING:
            progress = ui_theme.bar(job.progress)
            kind = "running"
            details.append(f"{name}: {round(job.progress * 100)}% · {job.desc or 'Starting…'}")
        elif job.status == queue_manager.PENDING:
            kind = "queued"
            ahead = queue_manager.position(job.id)
            details.append(f"{name}: queued" + (f" behind {ahead} job{'s' if ahead != 1 else ''}" if ahead else "")
                           + "." + (" The GPU queue is paused." if queue_manager.paused() else ""))
        elif job.status == queue_manager.ERROR:
            kind = "failed"
            details.append(f"{name} failed. {ui_theme.plain(job.error)}")
        elif job.status == queue_manager.STOPPED:
            details.append(f"{name} stopped.")
        elif job.status == queue_manager.DONE:
            details.append(f"{name} finished.")
    elif queue_manager.paused():
        kind = "queued"
        details.append("The GPU queue is paused.")
    if failed and failed.get("error") and not (job and job.status == queue_manager.ERROR):
        what = {timeline.PREPARE: "Clip generation", timeline.SOUND: "The sound preview",
                timeline.MERGE: "Join and check", timeline.EDIT: "The AI edit"}.get(failed.get("kind"), "The export")
        details.append(f"{what} failed. {ui_theme.plain(failed['error'])}")
        if kind not in ("running", "queued"):
            kind = "failed"
    details.append(f"{sum(bool(key) for key in state['assets'])}/{len(project.scenes)} clips ready in the saved edit")
    notes = ""
    if joined and state["merged"].get("notes"):
        attention = len(state["merged"].get("attention", []))
        if attention:
            details.append(f"{attention} join{'s' if attention != 1 else ''} to review.")
        notes = ("<p class=\"j-meta\">Transition report</p><ul>"
                 + "".join(f"<li>{ui_theme.esc(line)}</li>" for line in state["merged"]["notes"]) + "</ul>")
    headline = details[0]
    more = "".join(f'<p class="j-meta">{ui_theme.esc(detail)}</p>' for detail in details) + notes
    return (f'<div class="timeline-job tl-queue" role="status" data-state="{kind}">'
            '<span class="tl-queue-mark" aria-hidden="true"></span>'
            f'<span class="tl-queue-text" title="{ui_theme.esc(headline)}">{ui_theme.esc(headline)}</span>{progress}'
            f'<details><summary aria-label="Queue details">Details</summary><div class="tl-queue-panel">{more}</div>'
            '</details></div>')


def _raw_rows(value, key: str, fallback: list) -> list[dict]:
    """The rows an editor shows: what the user typed for this scene, else the scene's parts."""
    try:
        rows = _editor_rows(value, key)
    except ValueError:
        rows = None
    return fallback if rows is None else [row for row in rows if isinstance(row, dict)]


def _number(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return ""
    return f"{round(float(value), 2):g}"


def _text(value) -> str:
    return value if isinstance(value, str) else ""


def _row_parts(rows: list[dict], parse) -> tuple[list[tuple[int, object]], dict[int, str]]:
    parsed, problems = [], {}
    for position, row in enumerate(rows):
        try:
            found = parse(json.dumps([row], ensure_ascii=False))
        except (TypeError, ValueError) as error:
            problems[position] = str(error)
            continue
        if found:
            parsed.append((position, found[0]))
    return parsed, problems


def _line_rows(rows: list[dict], scene: timeline.Scene, measured, issues) -> list[LineRow]:
    cues, problems = _row_parts(rows, scene_parts.cues_from_json)
    current = tuple(cue for _, cue in cues) == scene.lines
    measured = measured if current and measured is not None and len(measured) == len(cues) else None
    windows = measured or scene_parts.timings([cue for _, cue in cues])
    by_row = {position: window for (position, _), window in zip(cues, windows)}
    notes: dict[int, str] = {}
    if current:
        for issue in issues:
            if issue.line is not None and 0 <= issue.line < len(cues):
                notes.setdefault(cues[issue.line][0], issue.text)
    result = []
    for position, row in enumerate(rows):
        window = by_row.get(position)
        past = bool(window and window[1] > scene.seconds + 1e-6)
        kind = ("past the end" if past else "measured" if measured else "estimated") if window else ""
        result.append(LineRow(_text(row.get("speaker")), _text(row.get("text")), _number(row.get("start")),
                              f"{window[0]:.1f}–{window[1]:.1f} s" if window else "", kind, past,
                              problems.get(position) or notes.get(position, "")))
    return result


def _effect_rows(rows: list[dict], scene: timeline.Scene, issues) -> list[EffectRow]:
    effects, problems = _row_parts(rows, scene_parts.effects_from_json)
    current = tuple(effect for _, effect in effects) == scene.effects
    by_row = dict(effects)
    notes: dict[int, str] = {}
    if current:
        for issue in issues:
            if issue.effect is not None and 0 <= issue.effect < len(effects):
                notes.setdefault(effects[issue.effect][0], issue.text)
    result = []
    for position, row in enumerate(rows):
        effect = by_row.get(position)
        window = f"{effect.at:.1f}–{effect.at + effect.seconds:.1f} s" if effect else ""
        past = bool(effect and effect.at + effect.seconds > scene.seconds + 1e-6)
        result.append(EffectRow(_text(row.get("text")), _number(row.get("at")), _number(row.get("seconds")),
                                window, past, problems.get(position) or notes.get(position, "")))
    return result


def _speaker_options(project: timeline.Project, scene: timeline.Scene, members: tuple[str, ...],
                     names: dict, pool) -> list:
    pickable = scene.cast if scene.cast is not None else project.cast_slugs
    only = f"{names.get(members[0], members[0])} (only character)" if len(members) == 1 else "Only character"
    options = [("", [(only, "")])]
    if pickable:
        options.append(("Characters", [(names.get(slug, slug), slug) for slug in pickable]))
    if pool:
        options.append(("Voices", [(f"@{voice.name}", f"@{voice.slug}") for voice in pool]))
    return options


def _speaker_extra(value: str, names: dict) -> str:
    if value.startswith(scene_parts.VOICE_PREFIX):
        return f"{value} (no such voice)"
    return f"{names.get(value, value)} (not in this scene)"


def _derived_background(prompt: str) -> str:
    """The background description the scene uses when the Background field is empty."""
    def compute():
        _, background, _ = dialogue.parse(refs.plain(prompt))
        return sound.prompt_from_scene(background, "")
    return _BACKGROUNDS(prompt, compute)


def _scene_mix(project: timeline.Project, index: int, audition, record: dict | None) -> str:
    """The scene's dialogue and background mixed at its levels, from the preview or the clip."""
    if not project.id or not project.audio or (audition is None and record is None):
        return ""
    key = (project.id, project.key, index, getattr(audition, "key", None), (record or {}).get("sha256"))

    def compute():
        try:
            path = timeline.preview_mix(project, index)
        except (ValueError, OSError):
            return ""
        return str(path) if path else ""
    path = _MIXES(key, compute)
    return cached_file(path) if path else ""


def _scene_takes(project: timeline.Project, index: int, stamp: tuple, assets: tuple) -> tuple[dict, ...]:
    if not project.id:
        return ()

    def compute():
        try:
            return tuple(take for take in timeline.takes(project, index) if isinstance(take, dict))
        except (ValueError, OSError):
            return ()
    return _TAKES((project.id, project.key, index, stamp, assets), compute)


def _take_label(take: dict) -> str:
    label = str(take.get("label") or take.get("key", "")[:8])
    if take.get("current"):
        label += " · in use"
    if take.get("anchor_ok") is False:
        label += " · the previous scene changed"
    return label


_EDIT_SOURCES = _Memo(64, ttl=5)


def _edit_note(saved: timeline.Project, index: int, stamp: tuple, assets: tuple) -> str:
    """The Edit tab's first line: the take an AI edit of the selected scene starts from, or what to do first."""
    if not saved.id or not 0 <= index < len(saved.scenes):
        return ('<p class="timeline-hint">Save the project and generate this scene first: an AI edit starts from its '
                'clip.</p>')

    def compute():
        try:
            source = timeline.edit_source(saved, index)
        except (ValueError, OSError):
            return None
        if source is None:
            return None
        return next((_take_label(take) for take in _scene_takes(saved, index, stamp, assets)
                     if take.get("key") == source), "the take it showed before its last edit")

    label = _EDIT_SOURCES((saved.id, saved.key, index, stamp, assets), compute)
    if label is None:
        return '<p class="timeline-hint">Generate this scene first: an AI edit starts from its clip.</p>'
    return (f'<p class="timeline-hint">The edit starts from <b>{ui_theme.esc(label)}</b>, which stays in Takes. Say '
            'what should change and click Plan with AI, or change the Picture and Dialogue tabs yourself, then click '
            'Make the edit.</p>')


def _sound_items(audition) -> tuple[tuple[str, str], ...]:
    items = []
    for number, (path, person, text, (start, end)) in enumerate(
            zip(audition.lines, audition.speakers, audition.texts, audition.timings), 1):
        items.append((f"Line {number} · {person.name} · {start:.1f}–{end:.1f} s · {text}", media_url(path)))
    if audition.background:
        # The stem keeps the export's lead-in; Scene mix plays everything on the scene clock.
        items.append((f"Background · {audition.background_prompt} · starts {scene_sound.HANDLE:g} s before the scene",
                       media_url(audition.background)))
    for number, (path, text) in enumerate(zip(audition.effects, audition.effect_texts), 1):
        items.append((f"Effect {number} · {text}", media_url(path)))
    return tuple(items)


def _heading(draft: Draft, status: str, problem: str = "", invalid_duration: bool = False) -> str:
    """The inspector's header: which scene, where it sits on the timeline, its status, a problem."""
    index = draft.selected
    start, end = _bounds(draft)
    note = problem or (f"Enter a whole duration from 1 to {video_gen.MAX_SECONDS} s." if invalid_duration else "")
    return ('<div class="tl-scene-head"><span class="tl-kicker">Scene inspector</span>'
            f'<h3 class="tl-scene-title">Scene {index + 1} <small>of {len(draft.scenes)}</small></h3>'
            f'<p class="tl-scene-meta"><span>{_clock(start)}–{_clock(end)}</span><span aria-hidden="true">·</span>'
            f'<span class="timeline-mark" data-status="{ui_theme.esc(status)}" aria-hidden="true"></span>'
            f'<span>{ui_theme.esc(status)}</span></p>'
            + (f'<p class="timeline-note" data-level="error" role="alert">{ui_theme.esc(note)}</p>' if note else "")
            + "</div>")


def _subtitles_option(with_audio: bool):
    """The Subtitles box can be ticked only with Audio on: without speech no line has a time to be shown at."""
    return gr.update(interactive=bool(with_audio))


def _status_html(current: timeline.Project, saved: timeline.Project, dirty: bool, names: list[str]) -> str:
    """The app bar's save state and the project's format (aspect, resolution, style, audio, cast)."""
    if not saved.id:
        state, text = ("dirty" if dirty else "new"), "New project" + (" · Unsaved changes" if dirty else "")
    elif dirty:
        state, text = "dirty", f"Revision {saved.revision} · Unsaved changes"
    else:
        state, text = "saved", f"Revision {saved.revision} saved"
    audio = "Audio on" if current.audio else "Audio off"
    # a line is shown while it is spoken: without audio there is nothing to show (the export says so too)
    subtitles = ["Subtitles" if current.audio else "Subtitles need Audio"] if current.subtitles else []
    look = [] if current.style == styles.DEFAULT else [style_library.label(current.style)]  # Realistic: as it always read
    short = [name if len(name) <= 24 else name[:23].rstrip() + "…" for name in look]  # the tooltip has it in full
    spec = [current.aspect.split()[-1] if current.aspect else "", video_gen.RESOLUTION, *short, audio, *subtitles]
    full = ([current.aspect, video_gen.RESOLUTION, *look, audio, *subtitles]
            + (["Cast: " + ", ".join(names)] if names else []))
    return (f'<div class="tl-status"><span class="tl-status-save" data-state="{state}">{ui_theme.esc(text)}</span>'
            f'<span class="tl-status-format" title="{ui_theme.esc(" · ".join(full))}">'
            f'{ui_theme.esc(" · ".join(part for part in spec if part))}</span></div>')


def _cast_cards(project: timeline.Project, pool) -> str:
    """The bin's Cast tab: the project's characters with a portrait and the voice they speak with."""
    voice_names = {voice.slug: voice.name for voice in pool}
    people = []
    for slug in project.cast_slugs:
        person = characters.get(slug)
        if not person:
            people.append(CastCard(f"{slug} (unavailable)", voice="This character no longer exists."))
            continue
        own = voices.for_character(person, pool)
        voice = ("Own voice" if own and own[0].startswith(dialogue.CHARACTER_VOICE)
                 else f"Voice: {voice_names.get(own[0], own[0])}" if own
                 else "Default voice" if project.audio else "Audio is off")
        people.append(CastCard(person["name"], media_url(characters.preview_path(person)), voice))
    message = ""
    if project.universe:
        found = universes.get(project.universe) or {}
        message = f"Universe: {found.get('name') or project.universe}. Its characters belong to the cast."
    return cast_html(people, message=message)


def _assets(project: timeline.Project, index: int, names: list[str], here: list[tuple[str, str]]) -> str:
    """The bin's Assets tab: the selected scene's reference images and the rest of the project's."""
    used: OrderedDict[str, list[int]] = OrderedDict()
    for number, scene in enumerate(project.scenes, 1):
        if number - 1 == index:
            continue
        for name in scene.references:
            used.setdefault(name, []).append(number)
    room = len(names) < scene_parts.MAX_REFERENCES
    elsewhere = []
    for name, numbers in used.items():
        if name in names or not scene_parts.is_reference(name):
            continue
        path = timeline.reference_path(name)
        url = tlv.url_for(tlv.cached_later(path)) if path else ""
        label = ("Scene " if len(numbers) == 1 else "Scenes ") + ", ".join(str(number) for number in numbers)
        elsewhere.append(AssetItem(name, url, label, usable=room))
    mine = [AssetItem(name, url, f"[{number}] {rest[0]}" if rest and rest[0] else f"[{number}]")
            for number, (name, url, *rest) in enumerate(here, 1)]
    return assets_html(index + 1, mine, elsewhere)


def _item_pool() -> tuple:
    """The usable places and items (at least one photo), remembered for a few seconds."""
    return _ITEMS("items", lambda: tuple(items.usable_items()))


def _film_item_choices(chosen=()) -> list[tuple[str, str]]:
    """The bin's Items picker: every usable place or item, and the chosen ones that cannot be used now (deleted, or
    without a photo) by their slug, so a pick is never dropped silently."""
    pool = _item_pool()
    known = {item["slug"] for item in pool}
    return ([(item["name"], item["slug"]) for item in pool]
            + [(f"{slug} (unavailable)", slug) for slug in chosen or () if isinstance(slug, str) and slug not in known])


def _film_item_cards(project: timeline.Project) -> str:
    """The bin's Items tab: the film's places and items with their main photo, their kind and the scenes that use
    them (picked as a reference image, or named in the picture text)."""
    pool = {item["slug"]: item for item in _item_pool()}
    cards = []
    for slug in project.items:
        item = pool.get(slug)
        if item is None:
            cards.append(CastCard(f"{slug} (unavailable)", "", "Deleted, or it has no photo yet. Remove it here."))
            continue
        reference = timeline.item_reference(item)
        used = [number for number, scene in enumerate(project.scenes, 1)
                if (reference and reference in scene.references) or items.mentions(scene.prompt, [item])]
        where = (("In scene " if len(used) == 1 else "In scenes ") + ", ".join(str(number) for number in used)
                 if used else "Not in a scene yet")
        photo = items.main_photo(item)
        kind = "Place" if item.get("kind") == items.PLACE else "Object"
        cards.append(CastCard(item["name"], media_url(photo) if photo else "", f"{kind} · {where}"))
    message = ("Write a name in a scene's picture text, or pick it under “Places and items in this scene” in the "
               "Picture tab: its photo then comes into the scene. The prompt assistant knows these names.")
    return cast_html(cards, message=message if cards else "", label="Places and items of this film",
                     empty="No places or items in this film yet. Add them above; make new ones on the Items page.")


def _scene_items(scene: timeline.Scene, names: list[str], film=()) -> tuple[tuple, tuple, str, dict]:
    """For the scene inspector, with the reference images `names` the editor shows: the picker's choices (every usable
    place or item, the film's own first: `film`, the bin's Items), its value (the items whose main photo is one of
    those images), the note on the items the picture text names (they are used as reference images too,
    timeline.named_items) and the item name of each image that is an item's main photo."""
    pool = _item_pool()
    order = {slug: number for number, slug in enumerate(film or ())}
    pool = tuple(sorted(pool, key=lambda item: order.get(item["slug"], len(order))))
    by_reference: dict[str, dict] = {}
    for item in pool:
        if name := timeline.item_reference(item):
            by_reference.setdefault(name, item)
    labels = {name: by_reference[name]["name"] for name in names if name in by_reference}
    values = tuple(dict.fromkeys(by_reference[name]["slug"] for name in names if name in by_reference))
    named = timeline.named_items(replace(scene, references=tuple(names))) if pool else []
    note = ""
    if named:
        listed = ", ".join(f"<b>{ui_theme.esc(item['name'])}</b>" for item in named)
        used = ("Its main photo is used as a reference image" if len(named) == 1 else
                "Their main photos are used as reference images")
        note = f'<p class="timeline-hint">Named in the picture text: {listed}. {used} of this scene too.</p>'
    return tuple((item["name"], item["slug"]) for item in pool), values, note, labels


def _presentation(draft: Draft, current: timeline.Project, view_mode: str, previous: Presentation,
                  invalid_duration: bool = False, problem: str = "", editors: tuple = (None, None, None)):
    saved = draft.baseline or _NEW_PROJECT
    # Read the plan for this tab's saved revision even if another tab has made a newer edit.
    state = timeline.prepared(saved) if saved.id else {"assets": []}
    jobs = _project_jobs(saved)
    stamp = tuple((job.id, job.status) for job in jobs)
    job = max((job for job in jobs if job.params.get("key") == saved.key), key=lambda job: job.id, default=None)
    latest = max(jobs, key=lambda job: job.id, default=None)
    current_edit = bool(saved.id and current.key == saved.key and not invalid_duration)
    # an unsaved edit shows the clips its save keeps (a style the film had before finds the clips made in it)
    keys = (state["assets"] if current_edit else timeline.plan_for(current, saved, state["assets"]) if saved.id
            else timeline.carry_clips(current, saved, state["assets"]))
    failed = _failure(saved)
    stale = _stale_scenes(current, saved, keys, current_edit, invalid_duration)
    statuses = _scene_statuses(current, saved, keys, job if current_edit else None, failed, stale)
    joined = timeline.merged_clip(saved, state) if current_edit else None
    program, hint, download = _program(current, saved, state, current_edit, joined, invalid_duration, keys)
    index = draft.selected
    scene = current.scenes[index]
    dirty = current != saved or invalid_duration or bool(problem)
    heading = _heading(draft, statuses[index], problem, invalid_duration)
    profiles = {slug: person for slug in dict.fromkeys((*current.cast_slugs, *(scene.cast or ())))
                if (person := characters.get(slug))}
    names = {slug: person["name"] for slug, person in profiles.items()}
    status = _status_html(current, saved, dirty, [names.get(slug, slug) for slug in current.cast_slugs])
    visuals = _track_visuals(current, saved, keys, stamp, index)
    track = track_html(draft, statuses, visuals=visuals, joined=bool(joined), view=view_mode, audio=current.audio,
                       program=program, hint=hint, aspect=current.aspect)
    starts = [sum(item.seconds for item in current.scenes[:number]) for number in range(len(current.scenes))]
    bin_items = []
    for number, (item, visual) in enumerate(zip(current.scenes, visuals)):
        thumb = visual.images[min(len(visual.images) - 1, len(visual.images) * 2 // 5)] if visual.images else ""
        bin_items.append(BinScene(number, starts[number], starts[number] + item.seconds, statuses[number],
                                  " ".join(item.prompt.split()), thumb, number == index, ui_theme.plain(visual.alert)))
    scenes_html = bin_html(bin_items)
    report = _queue_line(saved, state, latest, bool(joined), failed) if saved.id else ""
    choices = _scene_choices(current, scene)

    # Scene inspector: the editors are drawn from the rows the user typed for this scene.
    record = timeline.asset(saved, keys[index], verify=False) if index < len(keys) and keys[index] else None
    heard = _audition(current, index, stamp)
    issues = _scene_issues(current, index, stamp)
    members = _scene_members(current, scene, list(profiles.values()))
    pool = _voices()
    cast_cards = _cast_cards(current, pool)
    lines_raw, effects_raw, references_raw = editors
    line_rows = _line_rows(_raw_rows(lines_raw, draft.form, [cue.to_json() for cue in scene.lines]), scene,
                           _measured(scene, heard, record), issues)
    options = _speaker_options(current, scene, members, names, pool)
    known = {value for _, items in options for _, value in items}
    extras = [_speaker_extra(row.speaker, names) if row.speaker not in known else "" for row in line_rows]
    quotes = dialogue.quotes(scene.prompt)
    lines_message = ""
    if quotes and scene.lines:
        lines_message = "The picture text still has quoted lines. Move them here or remove the quotes."
    elif not current.audio:
        lines_message = "Project audio is off, so these lines are not heard."
    lines_html = lines_editor(line_rows, options, draft.form, extras=extras, message=lines_message)
    effect_rows = _effect_rows(_raw_rows(effects_raw, draft.form, [effect.to_json() for effect in scene.effects]),
                               scene, issues)
    effects_html = effects_editor(effect_rows, draft.form,
                                  message="" if current.audio else "Project audio is off, so effects are not heard.")
    reference_names = _reference_names(references_raw, draft.form, scene)
    item_choices, item_values, items_note, item_labels = _scene_items(scene, reference_names, current.items)
    reference_items = [(name, media_url(timeline.reference_path(name)) if scene_parts.is_reference(name) else "",
                        item_labels.get(name, "")) for name in reference_names]
    references = references_html(reference_items, draft.form)
    assets = _assets(current, index, reference_names, reference_items)
    if quotes and scene.lines:
        note = ('<p class="timeline-note" data-level="error">The picture text still has quoted lines. Move them '
                'into Dialogue or remove the quotes before saving.</p>')
    elif quotes:
        count = len(quotes)
        note = (f'<p class="timeline-note">The picture text has {count} quoted line{"s" if count != 1 else ""}. '
                'Convert them to dialogue lines to choose who says each one and when.</p>')
    else:
        note = ""
    if current.audio:
        check = issues_html(issues, empty="No sound problems found.")
    else:
        check = ('<p class="tl-check-ok">Project audio is off. Turn on Audio in the Project menu to hear '
                 'dialogue and sound.</p>')

    sound_job = next((job for job in sorted(jobs, key=lambda job: job.id, reverse=True)
                      if job.kind == timeline.SOUND and job.params.get("scene_index") == index
                      and job.status in (queue_manager.PENDING, queue_manager.RUNNING)), None)
    sound_for, sound_items, stale = (current.id, index), (), False
    if not current.audio:
        players_state = "Project audio is off."
    elif heard is not None:
        sound_items = _sound_items(heard)
        if record and record.get("audio_sources") and record.get("sound_key") != heard.key:
            players_state = ("This preview matches the scene, but it is not the sound of the current clip: the "
                             "scene mix and the export play the clip's sound. Generate a new take to use it.")
        else:
            players_state = "This preview matches the scene as it is now."
    elif previous.sound_for == sound_for and previous.sound_items:
        sound_items, stale = previous.sound_items, True
        players_state = ("Out of date: the scene changed after this preview. Prepare the sound preview again to "
                         "hear the changes.")
    else:
        players_state = ("No sound preview yet. Prepare one to hear the lines and the background before "
                         "generating the clip.")
    if sound_job is not None:
        players_state = ("Preparing the sound preview… " + (f"{round(sound_job.progress * 100)}%"
                         if sound_job.status == queue_manager.RUNNING else "queued")) + ". " + players_state
    players = players_html(sound_items, players_state, stale=stale)
    mix = _scene_mix(current, index, heard, record) if current.audio else ""

    takes = _scene_takes(current, index, stamp, tuple(keys))
    take_choices = tuple((_take_label(take), str(take.get("key", ""))) for take in takes if take.get("key"))
    take = next((str(take["key"]) for take in takes if take.get("current") and take.get("key")), None)
    if take_choices:
        takes_note = ('<p class="timeline-hint">Choose a take and click Use this take. Every take stays on disk, '
                      'and the other scenes keep their clips.'
                      + (' Another version is the scene before or after an AI edit: using it brings back its text '
                         'and dialogue too.' if any(take.get("settings") is not None for take in takes) else '')
                      + '</p>')
    else:
        takes_note = ('<p class="timeline-hint">No takes of this version of the scene yet. Generate the clip to '
                      'make the first take.</p>')

    queue_label = ""
    if queue_manager.paused():
        queue_label = "Resume queue"
    elif any(job.status in (queue_manager.PENDING, queue_manager.RUNNING) for job in jobs):
        queue_label = "Pause queue"
    failed_scene = (failed or {}).get("scene")
    retry = ""
    if (type(failed_scene) is int and 0 <= failed_scene < len(saved.scenes)
            and not (latest and latest.status in (queue_manager.PENDING, queue_manager.RUNNING))):
        what = {timeline.SOUND: "sound of scene", timeline.EDIT: "edit at scene"}.get(failed.get("kind"), "scene")
        retry = f"Retry {what} {failed_scene + 1}"
    return Presentation(
        track, heading, status, scenes_html, cast_cards, assets, report,
        _speaker_editor(scene, choices, _speaker_key(draft)), choices, lines_html, effects_html, references, note,
        bool(quotes) and not scene.lines, f"Auto: {_derived_background(scene.prompt)}", check, players, sound_for,
        sound_items, mix, take_choices, take, takes_note, download, queue_label, retry,
        missing=_has_work(current, keys), item_choices=item_choices, item_values=item_values, items_note=items_note,
        film_items=_film_item_cards(current), edit_note=_edit_note(saved, index, stamp, tuple(keys)))


def _stale_scenes(current: timeline.Project, saved: timeline.Project, keys: list, current_edit: bool,
                  invalid_duration: bool) -> frozenset:
    """The scenes shown Outdated: for the saved edit, those 'Generate missing clips' would make again; before a save
    that changes only the style, also every scene with a clip, as it was made in the saved style."""
    if current_edit:
        return _outdated_scenes(current, keys)
    if (saved.id and not invalid_duration and current.style != saved.style
            and replace(current, style=saved.style).key == saved.key):
        return _outdated_scenes(current, keys)
    return frozenset()


def _style_about(style_id) -> str:
    """The hint under the Project menu's Style: what the style looks like (and, for a built-in one, where to make
    one's own)."""
    if not styles.valid(style_id):
        return ""
    text = style_library.about(style_id)
    if not styles.of(style_id).custom:
        text += " Make your own style on the Styles page."
    return f'<p class="timeline-hint">{ui_theme.esc(text)}</p>'


def _outdated_scenes(project: timeline.Project, keys: list) -> frozenset:
    """The saved project's scenes whose clip 'Generate missing clips' would make again (timeline.outdated)."""
    if not project.id:
        return frozenset()
    try:
        return frozenset(timeline.outdated(project, keys))
    except (ValueError, OSError):
        return frozenset()


def _has_work(project: timeline.Project, keys: list) -> bool:
    """Generate missing clips has work: a scene without a clip, or one whose clip no longer fits
    (timeline.outdated: the scene before it or its cut changed, or its people)."""
    if not timeline.all_clips(project, keys) or not project.id:
        return True
    try:
        return bool(timeline.outdated(project, keys))
    except (ValueError, OSError):
        return False


def _reference_names(value, key: str, scene: timeline.Scene) -> list[str]:
    try:
        rows = _editor_rows(value, key)
    except ValueError:
        rows = None
    rows = list(scene.references) if rows is None else rows
    return list(dict.fromkeys(name for name in rows if isinstance(name, str)))


# The Video page's prompt assistant (app/ui_assist.py) under a scene's picture text. A scene is one shot of the film:
# the writer is told so, and a reply that still came in shots ("Shot 2: ...") is joined into one description.
SCENE_IDEA = ("For example, Penny rides her bicycle down a sunny hill, her scarf flying behind her, tracking shot. "
              "With reference images, refer to [1], [2]. Leave blank to improve the current description.")
_ONE_SHOT = ("(This is one scene of a longer film: write ONE continuous shot as one paragraph, never 'Shot 1:' or "
             "'Shot 2:'.)")
_SHOT_MARK = re.compile(r"(?im)^\s*(?:shot|uj[ęe]cie)\s*\d+\s*:\s*")


_CONTEXT_ITEMS = 12  # places and items named to a prompt writer that knows no film list


def _scene_context(draft: Draft, seconds=None, selected=None) -> str:
    """What the prompt writer is told about the selected scene of the film (llm.write_prompt's context): the film's
    characters by their @names, its places and items by their names (the bin's Items, else the first usable ones),
    how long the scene lasts and so how many words fit in it, and the scene before it when this one continues from
    its last frame."""
    index, scenes = draft.selected, draft.scenes
    known = (*(draft.baseline.cast_slugs if draft.baseline else ()), *_character_slugs(selected or ()))
    people = [person for slug in dict.fromkeys(known) if (person := characters.get(slug))]
    parts = [llm.characters_context(people or characters.all_characters())]
    film = draft.items if draft.items is not None else (draft.baseline.items if draft.baseline else ())
    pool = _item_pool()
    chosen = [item for slug in film for item in pool if item["slug"] == slug]
    parts.append(llm.items_context(chosen or list(pool[:_CONTEXT_ITEMS])))
    if scene_parts.is_length(seconds) and 1 <= seconds <= video_gen.MAX_SECONDS:
        room = seconds - scene_parts.FIRST_LINE - timeline.SPEECH_TAIL
        words = max(0, int(room * scene_parts.WORDS_PER_SECOND))
        parts.append(f"This scene lasts {seconds:g} s: one continuous shot with one main action, and at most about "
                     f"{words} spoken words in total" + (" - better none" if words < 2 else "") + ".")
    if 0 < index < len(scenes):
        before = " ".join(scenes[index - 1].prompt.split())[:400]
        if not scenes[index].cut and before:
            parts.append("This scene continues directly from the last frame of the scene before it, which showed: "
                         f"{before} Keep the same place, characters, positions and light, and describe only what "
                         "happens next.")
        elif scenes[index].cut:
            parts.append("This scene starts a new shot after an intentional cut: describe its place fully.")
    return "\n".join(part for part in parts if part)


def _write_scene_prompt(idea, current, draft, references, style, seconds=None, selected=None,
                        progress=gr.Progress()):
    """Write prompt in a scene's assistant: the idea (or, with none, the current picture text improved) as one shot in
    the film's style, with the scene's reference images numbered as its picture text numbers them."""
    def report(fraction, desc):
        progress(min(max(fraction, 0.0), 1.0), desc=desc)

    idea, current = (idea or "").strip(), (current or "").strip()
    if not idea and not current:
        return gr.skip(), ui_theme.note("❌ Enter an idea in any language, or a description to improve.")
    scene = draft.scenes[draft.selected] if 0 <= draft.selected < len(draft.scenes) else timeline.Scene()
    count = len(_reference_names(references, draft.form, scene))
    # a named animal or creature character is no person: the writer is told so, or its age rule makes one of it
    named = cast.mentions(f"{idea} {current}", [person for person in characters.all_characters()
                                                if characters.is_creature(person)])
    # its looks come from its photos: the writer is not given them (it copied them into the prompt, 2026-11-16)
    rule = " ".join([_ONE_SHOT, *(f"({person['name']} is an animal or creature, not a human. Keep it one, as "
                                  f"@{person['slug']}, with no human age and no description of its looks.)"
                                  for person in named)])
    ok, text = llm.write_prompt(f"{idea}\n{rule}" if idea else rule, "video", "" if idea else current,
                                report, n_refs=count, style=style if styles.valid(style) else styles.DEFAULT,
                                context=_scene_context(draft, seconds, selected),
                                names=[person["slug"] for person in cast.mentions(f"{idea} {current}",
                                                                                   characters.all_characters())])
    if not ok:
        return gr.skip(), ui_theme.note(text if text.startswith("⛔") else f"❌ {text}")
    shots = len(_SHOT_MARK.findall(text))
    text = " ".join(line.strip() for line in _SHOT_MARK.sub("", text).splitlines() if line.strip())
    tip = (" It came as several shots and is one scene now: add a scene (+ Scene) for every shot you want apart."
           if shots > 1 else "")
    return text, ui_theme.note(f"✅ The description has been written above. Review or edit it, then click Generate "
                               f"scene.{tip}")


def _pick_items(draft, value, picked):
    """“Places and items in this scene”: the main photo of an item picked there becomes one of the scene's reference
    images (timeline.store_reference, as an uploaded image would), and the image of an item taken out of it leaves
    them (the images after it move up a number, as with Remove). The reference editor's field gets the new list."""
    scene = draft.scenes[draft.selected] if 0 <= draft.selected < len(draft.scenes) else timeline.Scene()
    names = _reference_names(value, draft.form, scene)
    wanted = [slug for slug in dict.fromkeys(picked or ()) if isinstance(slug, str)]
    pool = {item["slug"]: item for item in items.usable_items()}
    owner = {}
    for slug, item in pool.items():
        if name := timeline.item_reference(item):
            owner.setdefault(name, slug)
    kept = [name for name in names if name not in owner or owner[name] in wanted]
    removed = len(names) - len(kept)
    added, message = [], ""
    for slug in wanted:
        item = pool.get(slug)
        if item is None or timeline.item_reference(item) in kept:
            continue
        if len(kept) >= scene_parts.MAX_REFERENCES:
            message = (f"❌ A scene has at most {scene_parts.MAX_REFERENCES} reference images. Remove one to add "
                       f"{item['name']}.")
            break
        try:
            kept.append(timeline.store_reference(items.main_photo(item)))
        except (ValueError, OSError) as error:
            message = f"❌ {item['name']}: {error}"
            break
        added.append(f"{item['name']} [{len(kept)}]")
    if not message and added:
        message = f"✅ Added to the scene's reference images: {', '.join(added)}."
    elif not message and removed:
        message = "✅ Taken out of the scene's reference images. The images after it moved up a number."
    shown = [owner[name] for name in kept if name in owner]  # a refused pick leaves the picker too
    return _envelope(kept, draft.form), ui_theme.note(message), gr.update(value=list(dict.fromkeys(shown)))


def _pick_film_items(draft, picked):
    """The bin's Items picker: the film's places and items as chosen; saved with the project (Project.items)."""
    return replace(draft, items=tuple(dict.fromkeys(slug for slug in picked or () if isinstance(slug, str) and slug)))


# ----- AI edit of a made scene (the inspector's Edit tab, timeline.edit) -----
EDIT_IDEA = ("For example: change the text on the screen to BREAKING NEWS · make it night · the dog smiles and waves "
             "at the end · he says “Bitcoin is down today” instead.")
EDIT_SCOPES = [("This scene", "scene"), ("This scene and the scenes after it", "after"), ("The whole film", "film")]


def _unspoken(prompt: str) -> str:
    """A picture text without its spoken lines: what the picture is made from apart from what is said."""
    parts, position = [], 0
    for match in dialogue.spoken(prompt or ""):
        parts.append(prompt[position:match.start()])
        position = match.end()
    parts.append((prompt or "")[position:])
    return " ".join("".join(parts).split())


def _plan_summary(before: timeline.Scene, after: timeline.Scene, frame: str) -> str:
    """The Edit tab's status after Plan with AI: what the plan changes, in a few words."""
    parts = []
    if _unspoken(before.prompt) != _unspoken(after.prompt):
        parts.append("a new picture text")
    said = [*dialogue.quotes(after.prompt), *(cue.text for cue in after.lines)]
    if said != [*dialogue.quotes(before.prompt), *(cue.text for cue in before.lines)]:
        parts.append("new dialogue" + (f" (“{said[0]}”)" if len(said) == 1 else ""))
    if frame:
        parts.append("a change to the first frame")
    if not parts:
        return "ℹ️ The assistant changed nothing. Say the change in other words, or change the fields yourself."
    return (f"✅ Plan ready: {', '.join(parts)}. Check the Picture and Dialogue tabs and the picture change below, "
            "then click Make the edit.")


def _plan_scene_edit(idea, draft, *values, progress=gr.Progress()):
    """Plan with AI: the selected scene's text and dialogue as the change asks (llm.plan_edit), loaded into the
    inspector unsaved, and the change for its first frame. Returns the editor values, the picture change, the status
    line and the form key the Edit tab's fields belong to (_edit_fields_for)."""
    form, project_values = values[:_FORM_FIELDS], values[_FORM_FIELDS:_FORM_FIELDS + _PROJECT_FIELDS]
    skip = tuple(gr.skip() for _ in range(_FORM_FIELDS + 1))

    def report(fraction, desc):
        progress(min(max(fraction, 0.0), 1.0), desc=desc)

    try:
        edited = _apply_form(draft, form)
        current = _candidate(edited, *project_values)
    except ValueError as error:
        return (*skip, gr.skip(), ui_theme.note(f"❌ {error}"), gr.skip())
    index = edited.selected
    scene = current.scenes[index]
    # the saved characters the text writes as @name keep that name (the planner is asked again otherwise)
    names = [person["slug"] for person in cast.mentions(scene.prompt, characters.all_characters())
             if re.search(rf"(?<![\w@])@{re.escape(person['slug'])}(?!\w)", scene.prompt, re.I)]
    ok, plan = llm.plan_edit(idea, scene.prompt, [cue.text for cue in scene.lines], style=current.style,
                             context=_scene_context(edited, scene.seconds, project_values[4]), names=names,
                             progress_cb=report)
    if not ok:
        return (*skip, gr.skip(), ui_theme.note(plan if str(plan).startswith("⛔") else f"❌ {plan}"), gr.skip())
    planned = replace(scene, prompt=plan["picture_text"])
    if scene.lines and plan["lines"]:
        try:
            planned = timeline.with_lines(planned, plan["lines"])
        except ValueError as error:
            return (*skip, gr.skip(), ui_theme.note(f"❌ The planned dialogue cannot be used: {error}"), gr.skip())
    loaded = _editor_values(replace(edited, scenes=edited.scenes[:index] + (planned,) + edited.scenes[index + 1:]))
    return (*loaded, plan["frame_change"], ui_theme.note(_plan_summary(scene, planned, plan["frame_change"])),
            loaded[0].form)


def _make_scene_edit(draft, *values):
    """Make the edit: the selected scene as the inspector shows it now (its text and dialogue, perhaps written by Plan
    with AI) becomes its new version, made from the take it showed (timeline.edit) with the Edit tab's picture change
    and scope. Refused while the film has other unsaved changes: the edit saves this scene only. Returns the editor
    values (the saved scene loaded again), the project list, the action status, the viewer mode, the Edit tab's
    fields emptied, its status and their form key."""
    form, project_values = values[:_FORM_FIELDS], values[_FORM_FIELDS:_FORM_FIELDS + _PROJECT_FIELDS]
    change, scope = (list(values[_FORM_FIELDS + _PROJECT_FIELDS:]) + ["", "scene"])[:2]
    try:
        edited = _apply_form(draft, form)
        saved = edited.baseline
        if saved is None or not saved.id:
            raise ValueError("Save the project and generate this scene first: an AI edit starts from its clip.")
        current = _candidate(edited, *project_values)
        index = edited.selected
        if (len(current.scenes) != len(saved.scenes) or replace(current, scenes=saved.scenes).key != saved.key
                or any(mine != kept for number, (mine, kept) in enumerate(zip(current.scenes, saved.scenes))
                       if number != index)):
            raise ValueError("Save the project first: an AI edit saves only this scene, and the film has other "
                             "unsaved changes.")
        project, job = timeline.edit(saved, index, current.scenes[index], change=change or "",
                                     scope=scope or "scene")
    except (ValueError, OSError) as error:
        return (*(gr.skip() for _ in range(_FORM_FIELDS + 1)), *(gr.skip() for _ in range(5)),
                ui_theme.note(f"❌ {error}"), gr.skip())
    loaded = _editor_values(Draft(project, project.scenes, index, items=edited.items))
    message = f"Revision {project.revision} saved. Job #{job.id} added to GPU Queue: AI edit of scene {index + 1}."
    return (*loaded, gr.update(choices=_project_choices(), value=project.id), ui_theme.note(message), _SCENE, "", "",
            ui_theme.note("✅ The edit is in the GPU Queue. Its new take shows here when it is ready; the take it "
                          "starts from stays in Takes."), loaded[0].form)


def _edit_fields_for(draft, form):
    """The Edit tab's fields belong to the scene that was loaded when they were filled: another scene loaded (a new
    form key) empties them."""
    key = draft.form if isinstance(draft, Draft) else ""
    if key == form:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    return "", "", "", key


def _use_take(draft, *values):
    """Use this take (_commit with take): the inspector loads the scene again, as a take of another version brings
    that version's text and dialogue back. Returns the editor values, the project list, the action status, the switch
    warning, the pending switch and the viewer mode."""
    state, library, status, confirm, pending, view, _ = _commit(take=True)(draft, *values)
    if not isinstance(state, Draft):  # refused before anything was saved
        return (*(gr.skip() for _ in range(_FORM_FIELDS + 1)), library, status, confirm, pending, view)
    return (*_editor_values(state), library, status, confirm, pending, view)


# The refresh's outputs, in order (the built tab lists its components the same way).
OUTPUTS = ("track", "bin", "status", "heading", "report", "cast", "assets", "download", "gallery", "queue",
           "retry", "missing", "speakers", "scene_cast", "items", "items_note", "lines", "effects", "references",
           "picture_note", "convert", "background", "check", "players", "mix", "takes", "takes_note", "film_items",
           "edit_note", "seen")
# Opening a project fills these too; the scene cast and the background are part of its form values.
_OPEN_OUTPUTS = tuple(name for name in OUTPUTS if name not in ("scene_cast", "background"))


def _display_updates(display: Presentation, prior: Presentation, exact: bool, *, force: bool = False) -> dict:
    """An update for every output whose content changed; gr.skip() leaves the rest (and their focus,
    scroll and playback) alone. force updates every output (another project was opened: what the
    page shows is not `prior`)."""
    def changed(*names):
        return force or any(getattr(display, name) != getattr(prior, name) for name in names)

    def html(name):
        return getattr(display, name) if changed(name) else gr.skip()

    skip = gr.skip()
    return {
        "track": html("track"),
        "bin": html("bin"),
        "status": html("status"),
        "heading": html("heading"),
        "report": html("report"),
        "cast": html("cast"),
        "assets": html("assets"),
        "download": (gr.update(value=display.download or None, visible=bool(display.download))
                     if changed("download") else skip),
        "gallery": gr.update(visible=bool(display.download)) if changed("download") else skip,
        "queue": (gr.update(value=display.queue_label or "Pause queue", visible=bool(display.queue_label))
                  if changed("queue_label") else skip),
        "retry": gr.update(value=display.retry or "Retry", visible=bool(display.retry)) if changed("retry") else skip,
        "missing": gr.update(interactive=display.missing) if changed("missing") else skip,
        "speakers": html("speakers"),
        "scene_cast": (gr.update(choices=list(display.cast_choices), visible=bool(exact))
                       if changed("cast_choices") else skip),
        "items": (gr.update(choices=list(display.item_choices), value=list(display.item_values))
                  if changed("item_choices", "item_values") else skip),
        "items_note": html("items_note"),
        "lines": html("lines"),
        "effects": html("effects"),
        "references": html("references"),
        "picture_note": html("picture_note"),
        "convert": gr.update(visible=display.convert) if changed("convert") else skip,
        "background": gr.update(placeholder=display.background_hint) if changed("background_hint") else skip,
        "check": html("check"),
        "players": html("players"),
        "mix": gr.update(value=display.mix or None, visible=bool(display.mix)) if changed("mix") else skip,
        "takes": (gr.update(choices=list(display.takes), value=display.take, visible=bool(display.takes))
                  if changed("takes", "take") else skip),
        "takes_note": html("takes_note"),
        "film_items": html("film_items"),
        "edit_note": html("edit_note"),
        "seen": display if force or display != prior else skip,
    }


def _refresh(draft, text, seconds, planned_cut, exact, members, assignments, lines, effects, sound_bed,
             references, voice_db, background_db, name, ratio, with_audio, universe, selected, subtitles, style,
             view_mode, prior):
    """Redraw what changed in the editor; unchanged parts are skipped so focus and playback stay."""
    values = (text, seconds, planned_cut, exact, members, assignments, lines, effects, sound_bed, references,
              voice_db, background_db)
    invalid_duration, problem = False, ""
    if _current_form(draft, values):  # else the inspector still shows an earlier load: preview the draft
        try:
            draft = _edited(draft, *values)
        except ValueError as error:
            try:  # Keep previewing the rest of the scene while one part is being corrected.
                draft = _edited(draft, *values[:6])
                problem = str(error)
            except ValueError:
                invalid_duration = True
    current = _candidate(draft, name, ratio, with_audio, universe, selected, subtitles, style)
    view_mode = view_mode if view_mode in (_SCENE, _PROGRAM) else _AUTO
    display = _presentation(draft, current, view_mode, prior, invalid_duration, problem, (lines, effects, references))
    updates = _display_updates(display, prior, exact)
    return tuple(updates[name] for name in OUTPUTS)


def _commit(kind=None, *, single: bool = False, retry: bool = False, take: bool = False):
    """Save the project, then queue `kind` for the selected scene (single), the failed scene
    (retry) or the whole project, or switch the selected scene to another take (take)."""
    def apply(draft, *values):
        form, project_values = values[:_FORM_FIELDS], values[_FORM_FIELDS:_FORM_FIELDS + _PROJECT_FIELDS]
        rest = values[_FORM_FIELDS + _PROJECT_FIELDS:]
        choice = rest[0] if take and rest else None
        name, ratio, with_audio, universe, selected, subtitles, style = project_values
        try:
            draft = _apply_form(draft, form)
            saved = draft.baseline
            if (kind or take) and saved and timeline.has_active_job(saved):
                raise ValueError("This project already has a job in the queue.")
            if take and not choice:
                raise ValueError("Choose a take first.")
            index, job_kind = (draft.selected if single else None), kind
            if retry:
                # The failure belongs to the saved version; read it before saving the current edits.
                failed = (_failure(saved) or {}) if saved else {}
                index = failed.get("scene")
                if type(index) is not int or not 0 <= index < len(draft.scenes):
                    raise ValueError("There is no failed scene to retry.")
                if failed.get("kind") in (timeline.SOUND, timeline.EDIT):
                    job_kind = failed["kind"]
            if not styles.valid(style):
                raise ValueError("Choose a valid style.")
            current = _candidate(draft, name, ratio, with_audio, universe, selected, subtitles, style)
            project = timeline.save(saved.id if saved else None, saved.revision if saved else None,
                                    name, draft.scenes, ratio, with_audio, list(current.characters),
                                    universe=current.universe, universe_members=current.universe_members,
                                    subtitles=current.subtitles, style=current.style, film_items=current.items)
        except (ValueError, OSError) as error:
            return gr.skip(), gr.skip(), ui_theme.note(f"❌ {error}"), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        draft = replace(draft, baseline=project)
        message = f"Revision {project.revision} saved."
        view_mode = gr.skip()
        if take:
            try:
                shown = timeline.use_take(project, draft.selected, choice)
                if isinstance(shown, timeline.Project) and shown.revision != project.revision:
                    # a take of another version of the scene: its text and dialogue came back with it
                    draft = Draft(shown, shown.scenes, draft.selected, items=draft.items)
                    message = (f"Revision {shown.revision} saved. Scene {draft.selected + 1} is back to the chosen "
                               "version: its take, text and dialogue.")
                else:
                    message = f"Revision {project.revision} saved. Scene {draft.selected + 1} now uses the chosen take."
                view_mode = _SCENE
            except (ValueError, OSError) as error:
                message = f"❌ Revision {project.revision} saved. {error}"
        elif job_kind == timeline.EDIT:
            try:
                job = timeline.retry_edit(project)
                message = f"Revision {project.revision} saved. Job #{job.id} added to GPU Queue."
                view_mode = _SCENE
            except (ValueError, OSError) as error:
                message = f"❌ Revision {project.revision} saved. {error}"
        elif job_kind == timeline.SOUND and timeline.sound_preview(project, index) is not None:
            message = f"Revision {project.revision} saved. The sound preview of scene {index + 1} is ready."
        elif job_kind == timeline.PRODUCE and timeline.exported(project):
            message = f"Revision {project.revision} saved. The movie is already exported: use Download movie."
            view_mode = _PROGRAM
        elif job_kind:
            try:
                job = timeline.enqueue(project, job_kind, scene_index=index,
                                       require_merge=job_kind == timeline.RENDER and not timeline.exported(project))
                message = f"Revision {project.revision} saved. Job #{job.id} added to GPU Queue."
                views = {timeline.PREPARE: _SCENE, timeline.MERGE: _PROGRAM, timeline.RENDER: _PROGRAM,
                         timeline.PRODUCE: _PROGRAM}
                view_mode = views.get(job_kind, gr.skip())
            except (ValueError, OSError) as error:
                message = f"❌ Revision {project.revision} saved. {error}"
        return (draft, gr.update(choices=_project_choices(), value=project.id), ui_theme.note(message),
                gr.update(visible=False), None, view_mode, "")
    return apply


# Before scenes are made: when a scene's dialogue needs more time than the scene lasts, the editor asks first (the
# owner's request of 2026-11-16) - render anyway (the end of the dialogue is cut off), use the suggested length and
# render, or cancel. The commit's outputs are followed by these: the pop-up, its text, the button that uses the
# suggested lengths and the plan it acts on.
_CHANGES = 7  # what _commit returns


def _length_plan(draft: Draft, values, single: bool) -> dict:
    """{scene index: (its length, the length its dialogue needs)} of the scenes the action would make whose dialogue
    needs more time (timeline.suggested_length): the selected one, or every scene Generate missing clips would make.
    {} when the form cannot be read: the action itself then says what is wrong."""
    try:
        edited = _apply_form(draft, values[:_FORM_FIELDS])
        current = _candidate(edited, *values[_FORM_FIELDS:_FORM_FIELDS + _PROJECT_FIELDS])
        if single:
            indices = [edited.selected]
        elif edited.baseline is not None and edited.baseline.id:
            saved = edited.baseline
            indices = timeline.outdated(current, timeline.plan_for(current, saved, timeline.prepared(saved)["assets"]))
        else:
            indices = range(len(current.scenes))
    except (ValueError, OSError):
        return {}
    found = {}
    for index in indices:
        try:
            if (length := timeline.suggested_length(current, index)) is not None:
                found[index] = (current.scenes[index].seconds, length)
        except (ValueError, OSError):
            continue
    return found


def _length_question(fixes: dict) -> tuple[str, str]:
    """(the pop-up's text, the label of the button that uses the suggested lengths)."""
    rows = [f"- Scene {index + 1} lasts {seconds:g} s; its dialogue needs about {length} s."
            for index, (seconds, length) in sorted(fixes.items())]
    text = ("**The dialogue is longer than the scene.** Rendered as it is, the end of the dialogue is cut off.\n\n"
            + "\n".join(rows))
    label = (f"Use {next(iter(fixes.values()))[1]} s and render" if len(fixes) == 1 else
             "Use the suggested lengths and render")
    return text, label


def _guarded(kind, *, single: bool = False):
    """_commit(kind), unless a scene it would make needs more time for its dialogue: then the pop-up asks first."""
    commit = _commit(kind, single=single)

    def apply(draft, *values):
        fixes = _length_plan(draft, values, single)
        if not fixes:
            return (*commit(draft, *values), gr.update(visible=False), gr.skip(), gr.skip(), None)
        text, label = _length_question(fixes)
        return (*(gr.skip() for _ in range(_CHANGES)), gr.update(visible=True), text, gr.update(value=label),
                {"kind": kind, "single": single, "fixes": fixes})
    return apply


def _length_choice(use: bool):
    """The pop-up's Render anyway (use=False) or Use the suggested length (use=True): the action it asked about, with
    the scenes' lengths changed first for the latter (the selected scene's in the inspector too)."""
    def apply(plan, draft, *values):
        skip = (*(gr.skip() for _ in range(_CHANGES)), gr.update(visible=False), gr.skip(), gr.skip(), None)
        if not isinstance(plan, dict):
            return (*skip, gr.skip())
        values, duration = list(values), gr.skip()
        if use:
            scenes = list(draft.scenes)
            for index, (_, length) in plan["fixes"].items():
                if index == draft.selected:
                    values[1] = duration = length  # the inspector's Duration (the form after the Draft)
                elif 0 <= index < len(scenes):
                    scenes[index] = replace(scenes[index], seconds=length)
            draft = replace(draft, scenes=tuple(scenes))
        result = _commit(plan["kind"], single=plan["single"])(draft, *values)
        return (*result, gr.update(visible=False), gr.skip(), gr.skip(), None, duration)
    return apply


def _toggle_queue():
    """Pause stops the running job (it returns to the queue) and releases GPU memory."""
    if queue_manager.paused():
        queue_manager.set_paused(False)
        return gr.update(value="Pause queue"), ui_theme.note("✅ The GPU queue is running again.")
    stopped = queue_manager.pause_now()
    message = "The GPU queue is paused and GPU memory is being released."
    if stopped:
        message += f" Job #{stopped} returns to the queue and restarts when you resume."
    return gr.update(value="Resume queue", visible=True), ui_theme.note(f"⏹ {message}")


def build_timeline_tab(tab: gr.Tab):
    initial = Draft()
    state = gr.State(initial)
    pending = gr.State(None)
    shown = _presentation(initial, _NEW_PROJECT, _AUTO, Presentation())
    seen = gr.State(shown)
    with gr.Column(elem_id="timeline-workspace", elem_classes=["tl-workspace"]):
        # ----- app bar: the project on the left, production on the right -----
        with gr.Row(elem_id="timeline-appbar", elem_classes=["tl-appbar"]):
            with gr.Row(elem_classes=["tl-appbar-project"]):
                title = gr.Textbox(value=_NEW_PROJECT.title, label="Project name", show_label=False, max_lines=1,
                                   container=False, min_width=0, elem_id="timeline-title")
                library = gr.Dropdown(_project_choices(), label="Saved projects", show_label=False, container=False,
                                      min_width=0, elem_id="timeline-projects")
                open_project = gr.Button("Open", size="sm", elem_id="timeline-open")
                from . import ui_series  # the Series workspace opens an episode's project here
                ui_series.HANDLES["timeline_library"] = library
                new_project = gr.Button("New", size="sm", elem_id="timeline-new")
                status_bar = gr.HTML(shown.status, padding=False, apply_default_css=False, elem_id="timeline-status")
                with gr.Accordion("Project", open=False, elem_id="timeline-project-menu"):
                    aspect = gr.Dropdown(list(video_gen.ASPECTS), value=_NEW_PROJECT.aspect, label="Aspect ratio")
                    # The film's look: every scene, start frame and character composition is made in it. A change
                    # marks the scenes already made Outdated (timeline._outdated).
                    style = gr.Dropdown(style_library.choices(_NEW_PROJECT.style), value=_NEW_PROJECT.style,
                                        label="Style", info="How every picture of the film is drawn. Changing it "
                                                            "marks the scenes already made as outdated.",
                                        elem_id="timeline-style")
                    style_about = gr.HTML(_style_about(_NEW_PROJECT.style), padding=False, apply_default_css=False,
                                          elem_id="timeline-style-about")
                    audio = gr.Checkbox(True, label="Audio", info="Dialogue, background and effects")
                    delete_project = gr.Button("Delete project", size="sm", variant="stop",
                                               elem_id="timeline-delete-project")
                gr.HTML(shell_html(), padding=False, apply_default_css=False, js_on_load=SHELL_JS,
                        elem_id="timeline-shell")
            with gr.Row(elem_classes=["tl-appbar-production"]):
                report = gr.HTML(shown.report, padding=False, apply_default_css=False, js_on_load=REPORT_JS,
                                 elem_id="timeline-queue-line")
                queue_toggle = gr.Button(shown.queue_label or "Pause queue", size="sm",
                                         visible=bool(shown.queue_label))
                retry = gr.Button(shown.retry or "Retry", size="sm", visible=bool(shown.retry))
                prepare_all = gr.Button("Generate missing clips", size="sm", interactive=shown.missing)
                join = gr.Button("Join and check", size="sm")
                # Where the export starts: burn the dialogue into the movie (a project option, off by default)
                subtitles = gr.Checkbox(_NEW_PROJECT.subtitles, label="Subtitles", container=False, min_width=0,
                                        interactive=_NEW_PROJECT.audio, elem_id="timeline-subtitles")
                export = gr.Button("Export movie", size="sm", variant="primary")
                download = gr.DownloadButton("Download movie", visible=False, size="sm")
                gallery = gr.Button("Show in Gallery", visible=False, size="sm")
        with gr.Group(visible=False, elem_classes=["tl-switch-warning"]) as confirm:
            gr.Markdown("There are unsaved changes. Save the project before switching, or discard the changes.")
            with gr.Row(elem_classes=["toolbar"]):
                cancel_switch = gr.Button("Keep editing", size="sm")
                discard = gr.Button("Discard and switch", size="sm", variant="stop")
        # the pop-up before a scene whose dialogue needs more time is made (_guarded)
        with gr.Group(visible=False, elem_id="timeline-length-warning") as length_warning:
            length_text = gr.Markdown()
            with gr.Row(elem_classes=["toolbar"]):
                length_anyway = gr.Button("Render anyway", size="sm")
                length_use = gr.Button("Use the suggested length and render", size="sm", variant="primary")
                length_cancel = gr.Button("Cancel", size="sm")
        length_plan = gr.State(None)
        action_status = gr.Markdown(elem_classes=["status-box"], elem_id="timeline-action-status")
        # ----- body: bin | viewer | inspector over the timeline -----
        with gr.Row(elem_classes=["tl-body"]):
            with gr.Column(min_width=0, elem_id="timeline-bin", elem_classes=["tl-panel", "tl-bin"]):
                # wrap: Gradio's overflow menu measured the tabs at its own larger size and hid Cast, Items and Assets
                # behind "..." in this narrow panel; wrapped, every tab stays in sight
                with gr.Tabs(elem_id="timeline-bin-tabs", overflow_behavior="wrap"):
                    with gr.Tab("Scenes", id="timeline-bin-scenes"):
                        scene_list = gr.HTML(shown.bin, padding=False, apply_default_css=False, js_on_load=BIN_JS,
                                             elem_id="timeline-scene-list")
                    with gr.Tab("Cast", id="timeline-bin-cast"):
                        universe_pick = gr.Dropdown(universe_choices(), value="", label="Universe",
                                                    allow_custom_value=True, elem_id="timeline-universe")
                        cast_pick = gr.Dropdown(_cast_choices(), multiselect=True, label="Additional characters",
                                                allow_custom_value=True, elem_id="timeline-characters")
                        cast_cards = gr.HTML(shown.cast, padding=False, apply_default_css=False,
                                             elem_id="timeline-cast-cards")
                    # The film's places and items (Project.items), chosen like its cast: listed first in a scene's
                    # picker and named to the prompt assistant
                    with gr.Tab("Items", id="timeline-bin-items"):
                        film_items_pick = gr.Dropdown(_film_item_choices(), value=[], multiselect=True,
                                                      label="Places and items in this film",
                                                      max_choices=timeline.MAX_FILM_ITEMS,
                                                      elem_id="timeline-film-items")
                        film_item_cards = gr.HTML(shown.film_items, padding=False, apply_default_css=False,
                                                  elem_id="timeline-film-item-cards")
                    with gr.Tab("Assets", id="timeline-bin-assets"):
                        assets = gr.HTML(shown.assets, padding=False, apply_default_css=False, js_on_load=ASSETS_JS,
                                         elem_id="timeline-assets")
                        with gr.Row(elem_classes=["toolbar"]):
                            upload_more = gr.UploadButton("Add images", file_count="multiple", file_types=["image"],
                                                          size="sm", elem_id="timeline-asset-upload")
            with gr.Column(min_width=0, elem_classes=["tl-panel", "tl-viewer-panel"]):
                gr.HTML(viewer_html(), padding=False, apply_default_css=False, js_on_load=VIEWER_JS,
                        elem_id="timeline-viewer")
            with gr.Column(min_width=0, elem_id="timeline-inspector", elem_classes=["tl-panel", "tl-inspector"]):
                scene_heading = gr.HTML(shown.heading, padding=False, apply_default_css=False,
                                        elem_id="timeline-scene-heading")
                with gr.Tabs(elem_id="timeline-inspector-tabs", overflow_behavior="wrap"):  # Edit and Takes in sight
                    with gr.Tab("Picture", id="timeline-picture"):
                        description = gr.Textbox(label="Picture and motion", lines=5, max_lines=12,
                                                 elem_id="timeline-description",
                                                 placeholder="Describe the shot: subject, action, camera…")
                        # the inspector is narrow: a short label stays on one line
                        scene_idea, write_scene, assistant_status = ui_assist.assistant_panel(
                            SCENE_IDEA, "AI prompt assistant — any language")
                        picture_note = gr.HTML(shown.picture_note, padding=False)
                        convert = gr.Button("Convert to dialogue lines", size="sm", visible=False)
                        with gr.Row(elem_classes=["timeline-controls"]):
                            duration = gr.Number(value=5, label="Duration (seconds)", min_width=120, scale=0, step=0.1,
                                                 elem_id="timeline-duration")
                            cut = gr.Checkbox(False, label="Intentional cut from previous scene", interactive=False)
                        exact_cast = gr.Checkbox(False, label="Choose characters for this scene",
                                                 info="Off: follow the description. Use an intentional cut to start "
                                                      "with a different cast.")
                        scene_cast = gr.Dropdown([], value=[], multiselect=True, label="Appearing in this scene",
                                                 max_choices=cast.MAX_CAST, allow_custom_value=True, visible=False)
                        # Places and items (app/items.py): a picked one's main photo is one of the References below
                        scene_items = gr.Dropdown(list(shown.item_choices), value=[], multiselect=True,
                                                  label="Places and items in this scene",
                                                  info="Its main photo becomes one of the reference images below. "
                                                       "Writing its name in the picture text brings it in too. Make "
                                                       "them on the Items page.",
                                                  max_choices=scene_parts.MAX_REFERENCES, elem_id="timeline-items")
                        items_note = gr.HTML(shown.items_note, padding=False, elem_id="timeline-items-note")
                        gr.HTML('<p class="timeline-field-label">References</p>', padding=False)
                        references_view = gr.HTML(shown.references, padding=False, js_on_load=REFERENCES_JS,
                                                  elem_id="timeline-reference-images")
                        references_box = gr.Textbox(value="", show_label=False, elem_id=tlv.REFERENCES_ID)
                        with gr.Row(elem_classes=["toolbar"]):
                            upload = gr.UploadButton("Add images", file_count="multiple", file_types=["image"],
                                                     size="sm", elem_id="timeline-reference-upload")
                        gr.HTML('<p class="timeline-hint">Refer to them as [1], [2] in the picture text. Removing an '
                                'image renumbers the ones after it.</p>', padding=False)
                    with gr.Tab("Dialogue", id="timeline-dialogue"):
                        speaker_editor = gr.HTML(shown.speakers, padding=False, js_on_load=_SPEAKER_JS,
                                                 elem_id="timeline-speakers")
                        line_assignments = gr.Textbox(value="", show_label=False, elem_id="timeline-line-assignments")
                        lines_view = gr.HTML(shown.lines, padding=False, js_on_load=ROWS_JS,
                                             elem_id="timeline-line-editor")
                        lines_box = gr.Textbox(value="", show_label=False, elem_id=tlv.LINES_ID)
                    with gr.Tab("Sound", id="timeline-sound"):
                        background = gr.Textbox(label="Background sound", lines=1, max_lines=3,
                                                placeholder=shown.background_hint,
                                                info="Leave empty to derive it from the picture text.")
                        gr.HTML('<p class="timeline-field-label">Sound effects</p>', padding=False)
                        effects_view = gr.HTML(shown.effects, padding=False, js_on_load=ROWS_JS,
                                               elem_id="timeline-effect-editor")
                        effects_box = gr.Textbox(value="", show_label=False, elem_id=tlv.EFFECTS_ID)
                        voice_level = gr.Slider(*scene_parts.LEVEL_RANGE, value=0, step=0.5,
                                                label="Dialogue level (dB)")
                        background_level = gr.Slider(*scene_parts.LEVEL_RANGE, value=0, step=0.5,
                                                     label="Background level (dB)")
                        gr.HTML('<p class="timeline-field-label">Sound check</p>', padding=False)
                        sound_check = gr.HTML(shown.check, padding=False, elem_id="timeline-sound-check")
                        with gr.Row(elem_classes=["toolbar"]):
                            prepare_sound = gr.Button("Prepare sound preview", size="sm")
                        players = gr.HTML(shown.players, padding=False, elem_id="timeline-sound-players")
                        scene_mix = gr.Audio(label="Scene mix", type="filepath", interactive=False, visible=False,
                                             buttons=[], elem_id="timeline-scene-mix",
                                             waveform_options=ui_theme.waveform())
                    # An AI edit of the made scene (timeline.edit): a new take from the take it shows
                    with gr.Tab("Edit", id="timeline-edit-tab"):
                        edit_note = gr.HTML(shown.edit_note, padding=False, elem_id="timeline-edit-note")
                        edit_idea = gr.Textbox(label="What should change? (any language)", lines=2, max_lines=6,
                                               placeholder=EDIT_IDEA, elem_id="timeline-edit-idea")
                        with gr.Row(elem_classes=["toolbar"]):
                            plan_edit = gr.Button("Plan with AI", size="sm", elem_id="timeline-plan-edit")
                        edit_frame = gr.Textbox(label="Change to the picture", lines=2, max_lines=5,
                                                placeholder="For example: Change the text on the screen to “BREAKING "
                                                            "NEWS”. Leave empty to keep the first frame.",
                                                info="Made on the scene's first frame; the scene then moves again "
                                                     "from it.", elem_id="timeline-edit-frame")
                        edit_scope = gr.Radio(EDIT_SCOPES, value="scene", label="Make the change in",
                                              info="After this scene, the scenes that continue from it are made "
                                                   "again from its new last frame, and a picture change reaches "
                                                   "every new shot too.", elem_id="timeline-edit-scope")
                        edit_status = gr.Markdown(elem_classes=["status-box"], elem_id="timeline-edit-status")
                        with gr.Row(elem_classes=["toolbar"]):
                            make_edit = gr.Button("Make the edit", variant="primary", size="sm",
                                                  elem_id="timeline-make-edit")
                    with gr.Tab("Takes", id="timeline-takes-tab"):
                        takes = gr.Radio([], label="Takes of this scene", elem_id="timeline-takes", visible=False)
                        takes_note = gr.HTML(shown.takes_note, padding=False)
                        with gr.Row(elem_classes=["toolbar"]):
                            use_take = gr.Button("Use this take", size="sm")
                            new_take = gr.Button("Generate new take", size="sm")
                with gr.Row(elem_classes=["toolbar", "tl-inspector-actions"]):
                    generate = gr.Button("Generate scene", variant="primary", size="sm", elem_id="timeline-generate")
                    save = gr.Button("Save project", size="sm", elem_id="timeline-save")
            with gr.Column(min_width=0, elem_id="timeline-timeline", elem_classes=["tl-panel", "tl-timeline-panel"]):
                with gr.Row(elem_classes=["tl-timeline-toolbar"]):
                    add = gr.Button("+ Scene", size="sm")
                    duplicate = gr.Button("Duplicate", size="sm")
                    remove = gr.Button("Delete scene", size="sm")
                    earlier = gr.Button("Move earlier", size="sm")
                    later = gr.Button("Move later", size="sm")
                    gr.HTML(zoom_html(), padding=False, apply_default_css=False, js_on_load=ZOOM_JS,
                            elem_id="timeline-zoom")
                track = gr.HTML(shown.track, padding=False, apply_default_css=False, elem_id="timeline-track",
                                js_on_load=TRACK_JS)
        with gr.Column(elem_classes=["tl-hidden"]):
            view = gr.Textbox(value=_AUTO, label="Viewer source", show_label=False, max_lines=1,
                              elem_id="timeline-view")
            ticks = gr.Timer(3)
            delete_answer = gr.Textbox(value="", show_label=False, max_lines=1, elem_id="timeline-delete-answer")

    form = [state, description, duration, cut, exact_cast, scene_cast, line_assignments, lines_box, effects_box,
            background, references_box, voice_level, background_level]
    settings = form + [title, aspect, audio, universe_pick, cast_pick, subtitles, style]
    editor_outputs = form
    outputs = {"track": track, "bin": scene_list, "status": status_bar, "heading": scene_heading, "report": report,
               "cast": cast_cards, "assets": assets, "download": download, "gallery": gallery, "queue": queue_toggle,
               "retry": retry, "missing": prepare_all, "speakers": speaker_editor, "scene_cast": scene_cast,
               "items": scene_items, "items_note": items_note,
               "lines": lines_view, "effects": effects_view, "references": references_view,
               "picture_note": picture_note, "convert": convert, "background": background, "check": sound_check,
               "players": players, "mix": scene_mix, "takes": takes, "takes_note": takes_note,
               "film_items": film_item_cards, "edit_note": edit_note, "seen": seen}
    view_outputs = [outputs[name] for name in OUTPUTS]
    project_outputs = (editor_outputs + [title, aspect, audio, universe_pick, cast_pick, subtitles, style,
                                         film_items_pick, library, action_status, view]
                       + [outputs[name] for name in _OPEN_OUTPUTS])
    switch_outputs = project_outputs + [confirm, pending]
    edit_form = gr.State("")  # the form key (Draft.form) the Edit tab's fields were filled for

    def open_saved(project_id):
        """The project's form values and its whole display at once, so the header, the cast and the
        track show the opened project immediately (the next refresh only confirms them)."""
        try:
            project = timeline.load(project_id) if project_id else _NEW_PROJECT
        except ValueError as error:
            raise gr.Error(ui_theme.plain(error)) from error
        draft = Draft(project if project.id else None, project.scenes or (timeline.Scene(),))
        values = list(_editor_values(draft))
        draft = values[0]
        current = replace(project, scenes=draft.scenes)
        display = _presentation(draft, current, _AUTO, Presentation(), editors=(values[7], values[8], values[10]))
        scene = draft.scenes[draft.selected]
        values[5] = gr.update(value=list(scene.cast or ()), visible=scene.cast is not None,
                              choices=list(display.cast_choices))
        values[9] = gr.update(value=scene.background, placeholder=display.background_hint)
        updates = _display_updates(display, Presentation(), scene.cast is not None, force=True)
        return (*values, project.title, project.aspect, project.audio,
                gr.update(value=project.universe, choices=universe_choices(project.universe)),
                gr.update(value=list(project.characters), choices=_cast_choices(project.characters)),
                project.subtitles, gr.update(value=project.style, choices=style_library.choices(project.style)),
                gr.update(value=list(project.items), choices=_film_item_choices(project.items)),
                gr.update(value=project.id or None, choices=_project_choices()),
                ui_theme.note(" ".join(("Project opened.", timeline.cast_notice(project))).strip())
                if project.id else "", _AUTO,
                *(updates[name] for name in _OPEN_OUTPUTS))

    def switch_project(target, draft, *values):
        try:
            edited = _apply_form(draft, values[:_FORM_FIELDS])
            dirty = _candidate(edited, *values[_FORM_FIELDS:]) != (draft.baseline or _NEW_PROJECT)
        except ValueError:
            dirty = True
        if dirty:
            return (*(gr.skip() for _ in project_outputs), gr.update(visible=True), target)
        return (*open_saved(target), gr.update(visible=False), None)

    def request_open(project_id, *args):
        if not project_id:
            # a warning, not gr.Error: an error marks every output (the project list too) as failed
            gr.Warning("Choose a saved project to open.")
            return tuple(gr.skip() for _ in switch_outputs)
        return switch_project(project_id, *args)

    def request_new(*args):
        return switch_project("", *args)

    def discard_switch(target):
        result = open_saved(target) if target is not None else tuple(gr.skip() for _ in project_outputs)
        return (*result, gr.update(visible=False), None)

    def delete_open(draft, answer):
        """Project → Delete project, once the browser asked: the open project goes to the Recycle Bin
        (timeline.delete) and the editor starts a new project, as it does when the project was deleted
        already in another tab. A refusal only says why. The answer box is emptied each time, so an
        earlier yes never counts again."""
        if answer != "yes":
            return (*(gr.skip() for _ in switch_outputs), "")
        try:
            if draft.baseline is None:
                raise ValueError("This project is not saved yet, so there is nothing to delete.")
            name = timeline.delete(draft.baseline.id)
            message = f"🗑️ Project ‘{ui_theme.esc(name)}’ moved to the Recycle Bin. You can restore it from there."
        except timeline.Deleted:
            message = (f"🗑️ Project ‘{ui_theme.esc(draft.baseline.title)}’ was deleted already, perhaps in "
                       "another tab.")
        except ValueError as error:
            return (*(ui_theme.note(f"❌ {error}") if output is action_status else gr.skip()
                      for output in switch_outputs), "")
        values = list(open_saved(""))
        values[project_outputs.index(action_status)] = ui_theme.note(message)
        return (*values, gr.update(visible=False), None, "")

    def track_action(draft, text, seconds, planned_cut, exact, members, assignments, lines, effects, sound_bed,
                     references, voice_db, background_db, evt: gr.EventData):
        """A click on the timeline or in the bin: select a scene, reorder (drag and drop), change a
        scene's length (its right edge dragged) or show the program (the V2 block).

        Quick clicks all reach the server (trigger mode "multiple", see below) and run in order: the
        last one wins. A click sent before the answer to an earlier one arrived carries the inspector
        as it was then; its values are not applied again (_current_form)."""
        data = evt._data if isinstance(evt._data, dict) else {}
        index, source, slot = data.get("index"), data.get("from"), data.get("slot")
        if data.get("joined"):
            return (*(gr.skip() for _ in editor_outputs), _PROGRAM)
        if "trim" not in data and type(index) is int and index == draft.selected:
            # The scene is in the inspector already (a click sent again before its answer arrived):
            # loading it again would replace what has been typed since.
            return (*(gr.skip() for _ in editor_outputs), gr.skip())
        values = (text, seconds, planned_cut, exact, members, assignments, lines, effects, sound_bed, references,
                  voice_db, background_db)
        try:
            draft = _apply_form(draft, values)
            if "trim" in data:
                return (*_editor_values(_trimmed(draft, data.get("trim"), data.get("seconds"))), gr.skip())
        except ValueError as error:
            raise gr.Error(ui_theme.plain(error)) from error
        if type(index) is int and 0 <= index < len(draft.scenes):
            return (*_editor_values(replace(draft, selected=index)), gr.skip())
        if (type(source) is int and type(slot) is int
                and 0 <= source < len(draft.scenes) and 0 <= slot <= len(draft.scenes)):
            return (*_editor_values(_scene_order(draft, "drop", (source, slot))), gr.skip())
        return (*(gr.skip() for _ in editor_outputs), gr.skip())

    def change_scene(action):
        def apply(draft, *values):
            try:
                if action != "remove":
                    draft = _apply_form(draft, values)
                return _editor_values(_scene_order(draft, action))
            except ValueError as error:
                raise gr.Error(ui_theme.plain(error)) from error
        return apply

    def convert_quotes(draft, *values):
        try:
            draft = _apply_form(draft, values[:_FORM_FIELDS])
            current = _candidate(draft, *values[_FORM_FIELDS:])
            scene = timeline.quotes_to_cues(current, draft.scenes[draft.selected])
        except ValueError as error:
            raise gr.Error(ui_theme.plain(error)) from error
        index = draft.selected
        return _editor_values(replace(draft, scenes=draft.scenes[:index] + (scene,) + draft.scenes[index + 1:]))

    def add_references(draft, value, files):
        names = _reference_names(value, draft.form, draft.scenes[draft.selected])
        added, message = 0, ""
        for file in files or ():
            if len(names) >= scene_parts.MAX_REFERENCES:
                message = f"❌ Use at most {scene_parts.MAX_REFERENCES} reference images in a scene."
                break
            try:
                name = timeline.store_reference(file)
            except (ValueError, OSError) as error:
                message = f"❌ {Path(str(file)).name}: {error}"
                break
            if name not in names:
                names.append(name)
                added += 1
        if not message:
            message = f"✅ Added {added} reference image{'s' if added != 1 else ''}." if added else ""
        return _envelope(names, draft.form), ui_theme.note(message)

    # Navigation receives the form values, including text the user has not committed to the selected scene yet.
    new_project.click(request_new, inputs=settings, outputs=switch_outputs, api_name="timeline_new")
    open_project.click(request_open, inputs=[library, *settings], outputs=switch_outputs, api_name="timeline_open")
    cancel_switch.click(lambda: (gr.update(visible=False), None), outputs=[confirm, pending], api_visibility="private")
    discard.click(discard_switch, inputs=pending, outputs=switch_outputs, api_visibility="private")
    delete_project.click(None, inputs=title, outputs=delete_answer, js=DELETE_PROJECT_JS,
                         api_visibility="private").then(delete_open, inputs=[state, delete_answer],
                                                        outputs=[*switch_outputs, delete_answer],
                                                        api_visibility="private")
    # Every click reaches the server, in order: Gradio 6.30's "once" drops a click sent while another is
    # on its way, and its "always_last" sends the FIRST click's data again instead of the last one. Both
    # listeners share one queue (same function, concurrency 1), so the last click wins; track_action
    # does not apply the inspector values a click carried from before an earlier answer.
    track.click(track_action, inputs=form, outputs=[*editor_outputs, view], api_visibility="private",
                trigger_mode="multiple")
    scene_list.click(track_action, inputs=form, outputs=[*editor_outputs, view], api_visibility="private",
                     trigger_mode="multiple")
    exact_cast.input(lambda exact: gr.update(visible=bool(exact)), inputs=exact_cast, outputs=scene_cast,
                     queue=False, api_visibility="private", show_progress="hidden")
    for picker in (cast_pick, scene_cast):
        # A name typed and confirmed with Enter reaches the server as a change (Gradio's multiselect sends
        # input only for a picked or removed option); a value that is already a slug is left alone.
        picker.change(_typed_names, inputs=picker, outputs=picker, queue=False, api_visibility="private",
                      show_progress="hidden")
    for button, action in ((add, "add"), (duplicate, "duplicate"), (remove, "remove"),
                           (earlier, "earlier"), (later, "later")):
        button.click(change_scene(action), inputs=form, outputs=editor_outputs, api_name=f"timeline_{action}")
    convert.click(convert_quotes, inputs=settings, outputs=editor_outputs, api_name="timeline_convert_quotes")
    # The prompt assistant writes the selected scene's picture text: one shot, in the film's style
    write_scene.click(_write_scene_prompt, inputs=[scene_idea, description, state, references_box, style, duration,
                                                   cast_pick],
                      outputs=[description, assistant_status], api_visibility="private")
    for uploader in (upload, upload_more):
        uploader.upload(add_references, inputs=[state, references_box, uploader],
                        outputs=[references_box, action_status], api_visibility="private")
    scene_items.input(_pick_items, inputs=[state, references_box, scene_items],
                      outputs=[references_box, action_status, scene_items], api_visibility="private")
    changes = [state, library, action_status, confirm, pending, view, line_assignments]
    save.click(_commit(), inputs=settings, outputs=changes, api_name="timeline_save")
    asked = [*changes, length_warning, length_text, length_use, length_plan]
    generate.click(_guarded(timeline.PREPARE, single=True), inputs=settings, outputs=asked,
                   api_name="timeline_generate")
    new_take.click(_guarded(timeline.PREPARE, single=True), inputs=settings, outputs=asked,
                   api_name="timeline_new_take")
    length_anyway.click(_length_choice(False), inputs=[length_plan, *settings], outputs=[*asked, duration],
                        api_visibility="private")
    length_use.click(_length_choice(True), inputs=[length_plan, *settings], outputs=[*asked, duration],
                     api_visibility="private")
    length_cancel.click(lambda: (gr.update(visible=False), None), outputs=[length_warning, length_plan],
                        api_visibility="private")
    use_take.click(_use_take, inputs=[*settings, takes],
                   outputs=[*editor_outputs, library, action_status, confirm, pending, view],
                   api_name="timeline_use_take")
    # The Edit tab: Plan with AI fills the scene's own fields (unsaved) and the picture change; Make the edit saves the
    # scene and queues its new take from the take it showed. Another scene loaded empties the tab's fields.
    plan_edit.click(_plan_scene_edit, inputs=[edit_idea, *settings],
                    outputs=[*editor_outputs, edit_frame, edit_status, edit_form], api_visibility="private")
    make_edit.click(_make_scene_edit, inputs=[*settings, edit_frame, edit_scope],
                    outputs=[*editor_outputs, library, action_status, view, edit_idea, edit_frame, edit_status,
                             edit_form], api_name="timeline_edit")
    state.change(_edit_fields_for, inputs=[state, edit_form], outputs=[edit_idea, edit_frame, edit_status, edit_form],
                 queue=False, api_visibility="private", show_progress="hidden")
    film_items_pick.input(_pick_film_items, inputs=[state, film_items_pick], outputs=state, api_visibility="private",
                          show_progress="hidden")
    prepare_sound.click(_commit(timeline.SOUND, single=True), inputs=settings, outputs=changes,
                        api_name="timeline_sound")
    prepare_all.click(_guarded(timeline.PREPARE), inputs=settings, outputs=asked, api_name="timeline_prepare")
    join.click(_commit(timeline.MERGE), inputs=settings, outputs=changes, api_name="timeline_join")
    export.click(_guarded(timeline.PRODUCE), inputs=settings, outputs=asked, api_name="timeline_export")
    retry.click(_commit(timeline.PREPARE, retry=True), inputs=settings, outputs=changes, api_name="timeline_retry")
    queue_toggle.click(_toggle_queue, outputs=[queue_toggle, action_status], api_visibility="private")
    gallery.click(None, js=SHOW_GALLERY_JS, api_visibility="private")
    gr.on([tab.select, state.change, view.change, description.blur, duration.input, cut.input,
           exact_cast.input, scene_cast.change, line_assignments.change, lines_box.change, effects_box.change,
           references_box.change, background.blur, voice_level.release, background_level.release, title.blur,
           aspect.input, style.input, audio.input, subtitles.input, cast_pick.change, universe_pick.change, ticks.tick],
          _refresh, inputs=[*settings, view, seen], outputs=view_outputs, queue=False,
          api_visibility="private", show_progress="hidden", trigger_mode="always_last")
    tab.select(lambda current: gr.update(choices=_project_choices(), value=current), inputs=library,
               outputs=library, api_visibility="private")
    # Subtitles show the spoken lines at their times, so they need the project's audio (also when a project opens)
    audio.change(_subtitles_option, inputs=audio, outputs=subtitles, queue=False, api_visibility="private",
                 show_progress="hidden")

    def refresh_choices(universe, selected, look, film):
        return (gr.update(choices=universe_choices(universe)),
                gr.update(choices=_cast_choices(selected or ())),
                gr.update(choices=style_library.choices(look), value=look), _style_about(look),
                gr.update(choices=_film_item_choices(film or ())))

    tab.select(refresh_choices, inputs=[universe_pick, cast_pick, style, film_items_pick],
               outputs=[universe_pick, cast_pick, style, style_about, film_items_pick], api_visibility="private")
    style.change(_style_about, inputs=style, outputs=style_about, queue=False, api_visibility="private",
                 show_progress="hidden")
    # The owner's styles are made on the Styles page: the list is read again on every visit (above) and when the page
    # loads (Timeline is the start page, so a browser reload selects no tab). No inputs: it runs only on load.
    gr.on(None, lambda: gr.update(choices=style_library.choices()), outputs=style, queue=False,
          api_visibility="private", show_progress="hidden")
    # The same for the page's other lists: a browser reload shows the projects, universes, cast and items of now, not
    # the ones from when the server built the page
    gr.on(None, lambda: (gr.update(choices=_project_choices()), gr.update(choices=universe_choices()),
                         gr.update(choices=_cast_choices()), gr.update(choices=_film_item_choices())),
          outputs=[library, universe_pick, cast_pick, film_items_pick], queue=False, api_visibility="private",
          show_progress="hidden")
    return universe_pick, cast_pick
