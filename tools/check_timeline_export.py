# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""An objective quality report for a Timeline project and its exported movie. CPU only, read-only.

    venv/Scripts/python.exe tools/check_timeline_export.py <project id> [--export path.mp4]
        [--frames out_dir] [--json out.json]

The project, its plan and its clip records are read through app.timeline (load, prepared, asset,
exported / movie_preview); nothing in the project folders or the outputs folder is written, and
--frames / --json refuse a target inside them. The movie is the project's current export unless
--export names another file. Its sound is decoded once and every level below is measured on that one
decode (running sums of the power and of the speech-band power).

  container  duration against the sum of the scene durations (within one frame), 30 fps, frame count,
             resolution against the clips, the audio stream's codec, rate and channels
  loudness   integrated loudness, true peak and loudness range as libavfilter's loudnorm measures them
             (the measurement pass of the export's own leveling), overall RMS and sample peak
  per scene  start and end on the movie clock; cast and voices from the clip record; the background
             level in the export outside the dialogue windows (and the effects, when enough is left):
             below -45 dBFS it is 'too quiet'; every line of the clip's dialogue_timings on the movie
             clock with its energy in the 300-3400 Hz speech band over the scene's non-dialogue level
             in that band: under +6 dB 'weak', under +3 dB 'missing'; every effect of the scene
             (Scene.effects): its loudest 50 ms block within at..at+min(seconds, 1.5) against the median
             block of the second before it (the scene's background when that second is taken by
             dialogue or the movie's fade-in): under +4 dB 'no clear energy rise' (a rain bed alone
             varies by up to about +3.5 dB this way)
  joins      RMS within 0.2 s of each scene boundary against the quieter neighbour's background: 6 dB
             lower, or a 50 ms block 15 dB lower, is a dropout (a join in the movie's fades is skipped)
  stage 5    the test film: 30-60 s, two characters, several (3+) shots, at least two lines heard,
             audible background in every scene

Levels are dBFS as the studio counts them: 20*log10(RMS), so an RMS of 0.01 is -40 dBFS. A scene,
line or effect window is cut to the movie (a scene past the end of a shorter movie is not checked);
the export's fade-in (0.5 s) and fade-out (0.6 s) are left out of background levels. --frames writes a
contact sheet per scene (first, middle and last frame, labelled) for a visual review; without an
export the sheets show the scene clips. --json writes the whole report. The exit code is 0 for every
report, also with flags (listed at the end) and with outputs that could not be written; 2 only for a
wrong command line.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import av  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app import config, timeline, video_gen  # noqa: E402
from engine.media import audio as media_audio  # noqa: E402
from engine.media import graph as media_graph  # noqa: E402
from engine.media import video as media_video  # noqa: E402
from engine.media.errors import MediaError  # noqa: E402

FPS = video_gen.FPS               # the export's frame rate
RATE = media_audio.RATE           # ...and sample rate (stereo)
SPEECH_BAND = (300.0, 3400.0)     # Hz: where speech carries its energy (telephone band)
BLOCK = 0.05                      # s: level blocks, as the export's leveling measures (audio._block_rms)
FLOOR_DB = -120.0                 # the level of digital silence in this report
QUIET_BACKGROUND_DB = -45.0       # background RMS outside the dialogue below this: 'too quiet'
AUDIBLE_DB = -60.0                # a 50 ms block above this counts as audible
SPEECH_WEAK_DB = 6.0              # a line's speech band over the scene's non-dialogue level: below 'weak'
SPEECH_MISSING_DB = 3.0           # ...below this the line is no louder than the background: 'missing'
EFFECT_LOOK = 1.5                 # s: at most this much of an effect is looked at
EFFECT_BEFORE = 1.0               # s: ...against the second before it
EFFECT_RISE_DB = 4.0              # its loudest 50 ms block over the level before: below 'no energy rise'
JOIN_HALF = 0.2                   # s on both sides of a scene boundary
JOIN_DIP_DB = 6.0                 # join RMS this far below the quieter neighbour's background: a dropout
JOIN_GAP_DB = 15.0                # ...or one 50 ms block this far below that background's median block
SILENT_DB = -60.0                 # a join this quiet between two unmeasured backgrounds is silence
LINE_GUARD = (0.1, 0.45)          # s before / after a line kept out of background levels (ducking, release)
TALK_GUARD = (0.05, 0.1)          # s before / after a line kept out of an effect's window
EFFECT_GUARD = (0.05, 0.1)        # s before / after an effect kept out of background levels
MOVIE_FADES = (0.5, 0.6)          # s: the export fades the background in and out (audio.timeline_soundtrack)
MIN_SPAN = 0.25                   # s of sound needed for a background or 'second before' level
MIN_WINDOW = 0.1                  # s of sound needed for a line, effect or join window
QUIET_LUFS = -30.0                # an export quieter than this overall is flagged
CLIPPING = 0.999                  # a sample peak at or above this is clipping
STAGE5_SECONDS = (30.0, 60.0)
STAGE5_CHARACTERS, STAGE5_SHOTS, STAGE5_LINES = 2, 3, 2
THUMB_WIDTH, MARGIN, HEADER, CAPTION = 400, 12, 46, 26   # contact sheet layout (pixels)
ROLES = ("first", "middle", "last")

Span = tuple[float, float]


# --- spans on the movie clock (seconds) --------------------------------------------------------------

def _clip(spans: Iterable[Span], low: float, high: float) -> list[Span]:
    return [(max(a, low), min(b, high)) for a, b in spans if min(b, high) > max(a, low)]


def _subtract(spans: Iterable[Span], holes: Iterable[Span]) -> list[Span]:
    """`spans` without the parts covered by `holes`."""
    result = [(a, b) for a, b in spans if b > a]
    for low, high in holes:
        if high <= low:
            continue
        kept = []
        for a, b in result:
            if high <= a or low >= b:
                kept.append((a, b))
                continue
            if a < low:
                kept.append((a, low))
            if high < b:
                kept.append((high, b))
        result = kept
    return result


def _length(spans: Iterable[Span]) -> float:
    return sum(b - a for a, b in spans)


def _db(power: float) -> float:
    """dBFS of a mean power (RMS squared); FLOOR_DB for silence."""
    return max(FLOOR_DB, 10.0 * math.log10(power)) if power > 0 else FLOOR_DB


def _r(value, digits: int = 2):
    return None if value is None else round(float(value), digits)


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# --- the movie's sound -----------------------------------------------------------------------------

def _band(mono: np.ndarray, rate: int, low: float, high: float) -> np.ndarray:
    """`mono` limited to low..high Hz: one FFT of the whole track, raised-cosine edges 50 Hz wide."""
    count = mono.shape[0]
    if not count:
        return mono
    size = 1 << max(1, math.ceil(math.log2(count + rate // 10)))  # the padding keeps the wrap-around away
    frequencies = np.fft.rfftfreq(size, 1.0 / rate)
    edge = 50.0
    rise = np.clip((frequencies - (low - edge)) / edge, 0.0, 1.0)
    fall = np.clip(((high + edge) - frequencies) / edge, 0.0, 1.0)
    weights = (0.5 - 0.5 * np.cos(np.pi * rise)) * (0.5 - 0.5 * np.cos(np.pi * fall))
    return np.fft.irfft(np.fft.rfft(mono, size) * weights, size)[:count]


def _running(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


class Track:
    """The movie's sound, decoded once: running sums of its power (mean of the channels) and of the
    power of its speech band, so any stretch is measured without decoding or filtering again."""

    def __init__(self, samples: np.ndarray, rate: int):
        data = samples.astype(np.float64)
        self.rate, self.count = rate, data.shape[1]
        self.seconds = self.count / rate
        self.peak = float(np.max(np.abs(data))) if data.size else 0.0
        self._sums = {False: _running(np.mean(data * data, axis=0)),
                      True: _running(_band(data.mean(axis=0), rate, *SPEECH_BAND) ** 2)}

    def _ranges(self, spans: Iterable[Span]):
        for a, b in spans:
            start = min(max(int(round(a * self.rate)), 0), self.count)
            end = min(max(int(round(b * self.rate)), 0), self.count)
            if end > start:
                yield start, end

    def level(self, spans: Iterable[Span], band: bool = False) -> float | None:
        """RMS in dBFS over the spans together; None when they hold no sound."""
        sums, total, count = self._sums[band], 0.0, 0
        for start, end in self._ranges(spans):
            total += sums[end] - sums[start]
            count += end - start
        return _db(max(total, 0.0) / count) if count else None

    def blocks(self, spans: Iterable[Span], band: bool = False) -> np.ndarray:
        """dBFS of the 50 ms blocks of each span (a shorter span is one block, a tail of at least half a
        block is one too)."""
        size, sums = max(1, round(BLOCK * self.rate)), self._sums[band]
        starts, ends = [], []
        for start, end in self._ranges(spans):
            whole = (end - start) // size
            if whole:
                first = start + size * np.arange(whole)
                starts.append(first)
                ends.append(first + size)
            rest = (end - start) - whole * size
            if rest and (not whole or rest >= size // 2):
                starts.append(np.array([end - rest]))
                ends.append(np.array([end]))
        if not starts:
            return np.zeros(0)
        first, last = np.concatenate(starts), np.concatenate(ends)
        power = np.maximum(sums[last] - sums[first], 0.0) / (last - first)
        with np.errstate(divide="ignore"):
            return np.maximum(10.0 * np.log10(power), FLOOR_DB)


def _loudness(samples: np.ndarray, rate: int) -> dict | None:
    """Integrated loudness (LUFS), true peak (dBTP) and loudness range (LU) as libavfilter's loudnorm
    measures them (the measurement pass of audio.normalize_loudness); None when it cannot be read."""
    if samples.shape[1] < rate // 2:
        return None
    with media_graph.libav_messages() as lines:
        # loudnorm measures at 192 kHz: aresample keeps the discarded output small
        media_audio.run_chain(samples, rate, f"loudnorm={media_audio.LOUDNESS_TARGET}:print_format=json,"
                                             f"aresample={rate}")
        media_graph.release()
    for _, _, message in lines:
        if "input_i" in message and "{" in message:
            try:
                data = json.loads(message[message.index("{"):message.rindex("}") + 1])
            except ValueError:
                return None
            return {"integrated_lufs": _r(_finite(data.get("input_i")), 1),
                    "true_peak_dbtp": _r(_finite(data.get("input_tp")), 1),
                    "range_lu": _r(_finite(data.get("input_lra")), 1)}
    return None


# --- report ----------------------------------------------------------------------------------------

class _Log:
    """Flags (problems found) and checks that could not be made, in the order they come up."""

    def __init__(self):
        self.flags: list[str] = []
        self.unchecked: list[str] = []

    def flag(self, text: str) -> str:
        self.flags.append(text)
        return "flag"

    def skip(self, text: str) -> str:
        self.unchecked.append(text)
        return "unchecked"


def _timings(record: dict) -> tuple[list[dict], int]:
    """The clip's dialogue lines on the scene clock, in the record's order; and how many are unusable."""
    rows = record.get("dialogue_timings")
    if not isinstance(rows, list):
        return [], 0
    lines, broken = [], 0
    for row in rows:
        try:
            start, end = float(row["start"]), float(row["end"])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            broken += 1
            continue
        lines.append({"text": str(row.get("text") or ""), "character": str(row.get("character") or ""),
                      "voice": str(row.get("voice") or ""), "scene_start": start, "scene_end": end})
    return lines, broken


def _clip_facts(key: str, record: dict, seconds: int, number: int, log: _Log) -> dict:
    sources = record.get("audio_sources") or (None, None)
    stems = [name for name, path in (("background", sources[0]), ("voice", sources[1]),
                                     ("effects", record.get("effects_source"))) if path is not None]
    facts = {"key": key, "file": record["path"].name, "seconds": record["seconds"], "width": record["width"],
             "height": record["height"], "transition": record.get("transition"),
             "note": record.get("transition_note"), "stems": stems, "duration": None, "fps": None,
             "has_audio": None}
    if record["seconds"] != seconds:
        log.flag(f"Scene {number}: the clip lasts {record['seconds']} s, the scene {seconds} s.")
    try:
        info = media_video.probe(record["path"])
    except MediaError as error:
        log.flag(f"Scene {number}: the clip cannot be read ({error}).")
        return facts
    facts.update(duration=_r(info.duration, 3), fps=_r(info.fps, 3), has_audio=info.has_audio)
    if info.fps is None or abs(info.fps - FPS) > 0.01:
        log.flag(f"Scene {number}: the clip plays at {info.fps} fps, not {FPS}.")
    if info.duration is None or abs(info.duration - seconds) > 1 / FPS:
        log.flag(f"Scene {number}: the clip file lasts {info.duration} s, not {seconds} s.")
    return facts


def _scenes(project, keys: list, records: list, log: _Log) -> list[dict]:
    """Every scene on the movie clock with its clip facts, lines and effects (nothing measured yet)."""
    result, start = [], 0.0
    for index, scene in enumerate(project.scenes):
        number, end = index + 1, start + scene.seconds
        key = keys[index] if index < len(keys) else None
        record = records[index]
        entry = {"number": number, "start": _r(start, 3), "end": _r(end, 3), "seconds": scene.seconds,
                 "transition": "first" if index == 0 else "cut" if scene.cut else "continues",
                 "prompt": scene.prompt, "background_text": scene.background,
                 "planned_cast": list(scene.cast) if scene.cast is not None else None,
                 "clip": None, "cast": [], "voices": [], "lines": [], "effects": [], "background": None}
        if record:
            entry["clip"] = _clip_facts(key, record, scene.seconds, number, log)
            entry["cast"] = [str(slug) for slug in record.get("characters") or []]
            entry["voices"] = list(dict.fromkeys(str(voice) for voice in record.get("voices") or []))
            lines, broken = _timings(record)
            if broken:
                log.flag(f"Scene {number}: {broken} dialogue timing(s) of the clip cannot be read.")
            for line_number, line in enumerate(lines, 1):
                line.update(number=line_number, start=_r(start + line["scene_start"], 3),
                            end=_r(start + line["scene_end"], 3))
                if line["scene_end"] > scene.seconds + 1 / FPS:
                    log.flag(f"Scene {number}: line {line_number} ends at {line['scene_end']:.2f} s, after the "
                             f"scene's {scene.seconds} s; the export cuts it off.")
            entry["lines"] = lines
            if scene.lines and not lines:
                log.flag(f"Scene {number}: the clip has no dialogue timings for the scene's {len(scene.lines)} "
                         "line(s), so they cannot be checked.")
        else:
            entry["lines"] = [{"number": line_number, "text": cue.text, "character": cue.character,
                               "voice": cue.voice, "planned_start": cue.start}
                              for line_number, cue in enumerate(scene.lines, 1)]
            log.flag(f"Scene {number}: " + ("the plan's clip is missing or unreadable." if key else "no clip yet."))
        entry["effects"] = [{"number": effect_number, "text": effect.text, "scene_at": effect.at,
                             "seconds": effect.seconds, "at": _r(start + effect.at, 3)}
                            for effect_number, effect in enumerate(scene.effects, 1)]
        result.append(entry)
        start = end
    return result


def _find_export(project, state: dict, export: Path | None, log: _Log) -> tuple[Path | None, dict]:
    try:
        current = timeline.exported(project, state)
    except (ValueError, OSError):
        current = None
    if export is not None:
        path = Path(export)
        info = {"path": str(path), "source": "--export",
                "current": bool(current is not None and path.is_file() and current.resolve() == path.resolve())}
        if not path.is_file():
            log.flag(f"Export: {path} does not exist.")
            return None, info
        return path, info
    if current is not None:
        return current, {"path": str(current), "source": "current export", "current": True}
    try:
        older = timeline.movie_preview(project, state)
    except (ValueError, OSError):
        older = None
    hint = f" An older export exists: {older.path} (check it with --export)." if older is not None else ""
    log.flag(f"Export: the project has no current export, so the movie's sound and picture were not checked.{hint}")
    return None, {"path": None, "source": None, "current": False,
                  "older": str(older.path) if older is not None else None}


def _container(path: Path, project, records: list, log: _Log) -> dict | None:
    """What the movie file says about itself, against the plan."""
    try:
        info = media_video.probe(path)
        with av.open(str(path)) as container:
            video = container.streams.video[0] if container.streams.video else None
            sound = container.streams.audio[0] if container.streams.audio else None
            frames = video.frames if video is not None and video.frames else None
            codec = video.codec_context.name if video is not None else None
            audio = None
            if sound is not None:
                context = sound.codec_context
                audio = {"codec": context.name, "rate": context.sample_rate, "channels": context.layout.nb_channels,
                         "layout": context.layout.name, "decoded_seconds": None}
    except (MediaError, av.error.FFmpegError) as error:
        log.flag(f"Export: cannot read {path} ({error}).")
        return None
    expected = float(project.seconds)
    duration_ok = info.duration is not None and abs(info.duration - expected) <= 1 / FPS + 1e-6
    facts = {"duration": _r(info.duration, 3), "expected_duration": expected, "duration_ok": duration_ok,
             "fps": _r(info.fps, 3), "frames": frames, "expected_frames": int(round(expected * FPS)),
             "width": info.width, "height": info.height, "video_codec": codec, "audio": audio}
    if not duration_ok:
        log.flag(f"Export: the movie lasts {info.duration} s; its scenes add up to {expected:g} s "
                 "(more than one frame apart).")
    if info.fps is None or abs(info.fps - FPS) > 0.01:
        log.flag(f"Export: the movie plays at {info.fps} fps, not {FPS}.")
    # The picture and sound streams are checked on their own only when the length is right: a movie of the
    # wrong length says so once.
    if duration_ok and frames is not None and abs(frames - facts["expected_frames"]) > 1:
        log.flag(f"Export: the movie has {frames} frames, its scenes {facts['expected_frames']}.")
    sizes = {(record["width"], record["height"]) for record in records if record}
    if sizes and (info.width, info.height) not in sizes:
        log.flag(f"Export: the movie is {info.width}×{info.height}, its clips "
                 + ", ".join(f"{width}×{height}" for width, height in sorted(sizes)) + ".")
    if video is None:
        log.flag("Export: the movie has no picture.")
    if audio is None:
        if project.audio:
            log.flag("Export: the movie has no sound although the project's sound is on.")
    else:
        if audio["rate"] != RATE:
            log.flag(f"Export: the sound runs at {audio['rate']} Hz, not {RATE} Hz.")
        if audio["channels"] != 2:
            log.flag(f"Export: the sound has {audio['channels']} channel(s), not 2 (stereo).")
    return facts


def _background(low: float, high: float, core: Span, speech: list, effects: list) -> tuple:
    """(spans, what was left out): the scene's sound outside the dialogue, the effects and the movie's
    fades; the effects, then the fades are kept when too little is left without them."""
    scene = [(low, high)]
    for spans, holes, left_out in ((_clip(scene, *core), speech + effects, "dialogue and effects"),
                                   (_clip(scene, *core), speech, "dialogue"),
                                   (scene, speech, "dialogue")):
        spans = _subtract(spans, holes)
        if _length(spans) >= MIN_SPAN:
            return spans, left_out
    return [], None


def _measure(scenes: list[dict], track: Track, log: _Log) -> list[dict]:
    """Background, lines and effects of every scene and the joins between them; returns the joins."""
    end_of_movie = track.seconds
    core = (min(MOVIE_FADES[0], end_of_movie), max(end_of_movie - MOVIE_FADES[1], 0.0))
    speech = [(line["start"] - LINE_GUARD[0], line["end"] + LINE_GUARD[1])
              for scene in scenes for line in scene["lines"] if "start" in line]
    talk = [(line["start"] - TALK_GUARD[0], line["end"] + TALK_GUARD[1])
            for scene in scenes for line in scene["lines"] if "start" in line]
    effects = [(effect["at"] - EFFECT_GUARD[0], effect["at"] + effect["seconds"] + EFFECT_GUARD[1])
               for scene in scenes for effect in scene["effects"]]
    backgrounds = []
    for scene in scenes:
        low, high = scene["start"], min(scene["end"], end_of_movie)
        spans, left_out = _background(low, high, core, speech, effects)
        backgrounds.append(spans)
        if not spans:
            why = ("the dialogue covers the scene" if high - low >= MIN_SPAN
                   else f"the movie's sound ends at {end_of_movie:.2f} s")
            scene["background"] = {"reason": why,
                                   "status": log.skip(f"Scene {scene['number']}: background not measured; {why}.")}
            continue
        blocks = track.blocks(spans)
        level = track.level(spans)
        result = {"seconds": _r(_length(spans)), "left_out": left_out, "rms_dbfs": _r(level, 1),
                  "band_dbfs": _r(track.level(spans, band=True), 1),
                  "median_block_dbfs": _r(np.median(blocks), 1) if blocks.size else None,
                  "audible_share": _r(np.mean(blocks > AUDIBLE_DB)) if blocks.size else None, "status": "ok"}
        if level < QUIET_BACKGROUND_DB:
            result["status"] = log.flag(f"Scene {scene['number']}: background too quiet: {level:.1f} dBFS RMS outside "
                                        f"the {left_out} (below {QUIET_BACKGROUND_DB:g} dBFS).")
        scene["background"] = result
    everywhere = [span for spans in backgrounds for span in spans]
    movie_band = track.level(everywhere, band=True)
    movie_blocks = track.blocks(everywhere)
    movie_median = float(np.median(movie_blocks)) if movie_blocks.size else None

    for scene, spans in zip(scenes, backgrounds):
        number, low, high = scene["number"], scene["start"], min(scene["end"], end_of_movie)
        background = scene["background"]
        scene_band = background.get("band_dbfs") if spans else None
        scene_median = background.get("median_block_dbfs") if spans else None
        for line in scene["lines"]:
            if "start" not in line:
                continue
            where = f"Scene {number}, line {line['number']}" + (f" ({line['character']})" if line["character"] else "")
            window = _clip([(line["start"], line["end"])], low, high)
            if _length(window) < MIN_WINDOW:
                line["status"] = log.skip(f"{where}: not in the movie's sound.")
                continue
            reference, source = ((scene_band, "the scene's background") if scene_band is not None
                                 else (movie_band, "the movie's background"))
            if reference is None:
                line["status"] = log.skip(f"{where}: no background to compare the speech with.")
                continue
            band = track.level(window, band=True)
            rise = band - reference
            line.update(band_dbfs=_r(band, 1), reference_dbfs=_r(reference, 1), reference_from=source,
                        rise_db=_r(rise, 1), status="ok")
            if rise < SPEECH_MISSING_DB:
                line["status"] = log.flag(f"{where}: speech missing; its speech band is {rise:+.1f} dB over {source} "
                                          f"(needs +{SPEECH_WEAK_DB:g} dB).")
            elif rise < SPEECH_WEAK_DB:
                line["status"] = log.flag(f"{where}: speech weak; its speech band is {rise:+.1f} dB over {source} "
                                          f"(needs +{SPEECH_WEAK_DB:g} dB).")
        for effect in scene["effects"]:
            at = effect["at"]
            where = f"Scene {number}, effect {effect['number']} \"{effect['text']}\" at {effect['scene_at']:.2f} s"
            window = _clip([(at, at + min(effect["seconds"], EFFECT_LOOK))], low, high)
            look = _subtract(window, talk)
            if _length(look) < MIN_WINDOW:
                effect["status"] = log.skip(f"{where}: not checked; " + ("the dialogue covers it." if window
                                                                         else "it is outside the movie."))
                continue
            # the second before, without the movie's fade-in, any dialogue and any effect (its own guard too)
            before = _subtract(_clip([(at - EFFECT_BEFORE, at)], core[0], end_of_movie), speech + effects)
            if _length(before) >= MIN_SPAN:
                reference, source = float(np.median(track.blocks(before))), "the second before"
            elif scene_median is not None:
                reference, source = scene_median, "the scene's background"
            elif movie_median is not None:
                reference, source = movie_median, "the movie's background"
            else:
                effect["status"] = log.skip(f"{where}: no level before it to compare with.")
                continue
            peak = float(np.max(track.blocks(look)))
            rise = peak - reference
            effect.update(window=[_r(look[0][0], 3), _r(look[-1][1], 3)], peak_block_dbfs=_r(peak, 1),
                          window_rms_dbfs=_r(track.level(look), 1),
                          before_rms_dbfs=_r(track.level(before), 1) if before else None,
                          reference_dbfs=_r(reference, 1), reference_from=source, rise_db=_r(rise, 1), status="ok")
            if len(look) > 1 or look[0] != window[0]:
                effect["note"] = "the dialogue during the effect was left out"
            if rise < EFFECT_RISE_DB:
                effect["status"] = log.flag(f"{where}: no clear energy rise; its loudest 50 ms is {rise:+.1f} dB over "
                                            f"{source} (needs +{EFFECT_RISE_DB:g} dB). Listen to it.")

    joins = []
    for left, right in zip(scenes, scenes[1:]):
        at = left["end"]
        where = f"Join {left['number']} → {right['number']} at {at:.2f} s"
        window = [(at - JOIN_HALF, at + JOIN_HALF)]
        if window != _clip(window, core[0], core[1]):  # in the movie's fades, or past its end
            joins.append({"scenes": [left["number"], right["number"]], "at": at,
                          "status": log.skip(f"{where}: not checked; it is in the movie's fade or past its end.")})
            continue
        level = track.level(window)
        quietest = float(np.min(track.blocks(window)))
        sides = [scene["background"] for scene in (left, right) if scene["background"].get("rms_dbfs") is not None]
        join = {"scenes": [left["number"], right["number"]], "at": at, "rms_dbfs": _r(level, 1),
                "quietest_block_dbfs": _r(quietest, 1),
                "backgrounds_dbfs": [scene["background"].get("rms_dbfs") for scene in (left, right)], "status": "ok"}
        if sides:
            dip = min(side["rms_dbfs"] for side in sides) - level
            gap = min(side["median_block_dbfs"] for side in sides) - quietest
            if dip > JOIN_DIP_DB or gap > JOIN_GAP_DB:
                join["status"] = log.flag(f"{where}: dropout; {level:.1f} dBFS RMS within {JOIN_HALF:g} s, quietest 50 ms "
                                          f"{quietest:.1f} dBFS, against a background of "
                                          f"{min(side['rms_dbfs'] for side in sides):.1f} dBFS.")
        elif level < SILENT_DB:
            join["status"] = log.flag(f"{where}: silence ({level:.1f} dBFS RMS within {JOIN_HALF:g} s).")
        joins.append(join)
    return joins


def _stage5(scenes: list[dict], facts: dict | None, measured: bool, log: _Log) -> list[dict]:
    """The stage 5 test film: 30-60 s, two characters, several shots, at least two lines heard, audible
    background. A criterion that needs the export is not checked without one."""
    items = []

    def item(name: str, ok: bool | None, detail: str) -> None:
        items.append({"criterion": name, "ok": ok, "detail": detail})
        if ok is False:
            log.flag(f"Stage 5: {name}: {detail}.")
        elif ok is None:
            log.skip(f"Stage 5: {name}: {detail}.")

    planned = sum(scene["seconds"] for scene in scenes)
    if facts and facts.get("duration") is not None:
        seconds = facts["duration"]
        item("30–60 s long", STAGE5_SECONDS[0] - 1 / FPS <= seconds <= STAGE5_SECONDS[1] + 1 / FPS,
             f"{seconds:.2f} s")
    else:
        item("30–60 s long", None, f"{planned} s planned; no export to measure")
    with_clips = [scene for scene in scenes if scene["clip"]]
    cast = list(dict.fromkeys(slug for scene in with_clips for slug in scene["cast"]))
    if with_clips:
        detail = (", ".join(cast) or "nobody") + ("" if len(with_clips) == len(scenes)
                                                  else f" (in {len(with_clips)} of {len(scenes)} clips)")
        item("two characters", len(cast) >= STAGE5_CHARACTERS, detail)
    else:
        item("two characters", None, "no clips yet")
    item("several shots", len(scenes) >= STAGE5_SHOTS, f"{len(scenes)} scene(s)")
    lines = [line for scene in scenes for line in scene["lines"] if "start" in line]
    if measured:
        heard = [line for line in lines if line.get("rise_db") is not None and line["rise_db"] >= SPEECH_MISSING_DB]
        item("at least two lines", len(heard) >= STAGE5_LINES, f"{len(heard)} of {len(lines)} line(s) heard")
        quiet = [str(scene["number"]) for scene in scenes if scene["background"].get("status") == "flag"]
        unmeasured = [str(scene["number"]) for scene in scenes if scene["background"].get("status") == "unchecked"]
        if quiet:
            item("audible background", False, "too quiet in scene(s) " + ", ".join(quiet))
        elif unmeasured:
            item("audible background", None, "not measured in scene(s) " + ", ".join(unmeasured))
        else:
            item("audible background", True, "every scene")
    else:
        item("at least two lines", None, f"{len(lines)} line(s) in the clips; no export to listen to")
        item("audible background", None, "no export to listen to")
    return items


# --- contact sheets --------------------------------------------------------------------------------

def _font(size: int):
    """A system font with wide coverage (descriptions may be in any language), else Pillow's own."""
    for name in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):
        return ImageFont.load_default()


def _fit(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def _pick_frames(path: Path, wanted: Iterable[int]) -> dict:
    """{index: RGB image} of the frames at those positions, decoded once in order; an index past the
    end gets the last frame."""
    wanted, found, last = set(wanted), {}, None
    with contextlib.closing(media_video.read_frames(path)) as frames:
        for index, frame in enumerate(frames):
            if index in wanted:
                found[index] = frame.to_image().convert("RGB")
                if len(found) == len(wanted):
                    break
            last = frame
    if last is not None:
        for index in wanted - set(found):
            found[index] = last.to_image().convert("RGB")
    return found


def _sheet(title: str, subtitle: str, cells: list, out: Path) -> None:
    """A row of labelled thumbnails (caption, image) under a two-line header, saved as PNG."""
    thumbs = []
    for caption, image in cells:
        height = max(1, round(image.height * THUMB_WIDTH / image.width))
        thumbs.append((caption, image.resize((THUMB_WIDTH, height), Image.Resampling.LANCZOS)))
    tallest = max(image.height for _, image in thumbs)
    sheet = Image.new("RGB", (len(thumbs) * (THUMB_WIDTH + MARGIN) + MARGIN, HEADER + tallest + CAPTION + MARGIN),
                      (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    big, small = _font(16), _font(13)
    width = sheet.width - 2 * MARGIN
    draw.text((MARGIN, 6), _fit(draw, title, big, width), fill=(240, 240, 240), font=big)
    draw.text((MARGIN, 27), _fit(draw, subtitle, small, width), fill=(170, 170, 178), font=small)
    for column, (caption, image) in enumerate(thumbs):
        left = MARGIN + column * (THUMB_WIDTH + MARGIN)
        sheet.paste(image, (left, HEADER))
        draw.text((left, HEADER + tallest + 5), _fit(draw, caption, small, THUMB_WIDTH), fill=(220, 220, 220),
                  font=small)
    sheet.save(out)


def _contact_sheets(project_id: str, scenes: list[dict], records: list, export: Path | None,
                    facts: dict | None, out_dir: Path, log: _Log) -> list[str]:
    """A contact sheet per scene: the first, middle and last frame of the scene in the export (in the
    scene's clip without an export with a picture). Returns the files written."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        log.flag(f"Frames: cannot create {out_dir} ({error}).")
        return []
    plan = []  # (scene, source label, path, [(role, index, seconds on the movie clock)])
    if export is not None and facts is not None and facts.get("video_codec"):
        fps = facts.get("fps") or FPS
        total = facts.get("frames") or int(round((facts.get("duration") or 0.0) * fps))
        for scene in scenes:
            first, count = int(round(scene["start"] * fps)), max(1, int(round(scene["seconds"] * fps)))
            if first >= total:  # a shorter movie: this scene is not in it
                log.skip(f"Frames: scene {scene['number']} is past the end of the movie.")
                continue
            count = min(count, total - first)
            indices = (first, first + count // 2, first + count - 1)
            plan.append((scene, f"export {export.name}", export,
                         [(role, index, index / fps) for role, index in zip(ROLES, indices)]))
    else:
        for scene, record in zip(scenes, records):
            if not record or not scene["clip"]:
                continue
            fps = scene["clip"].get("fps") or FPS
            count = max(1, int(round(scene["seconds"] * fps)))
            indices = (0, count // 2, count - 1)
            plan.append((scene, f"clip {scene['clip']['key'][:12]}", record["path"],
                         [(role, index, scene["start"] + index / fps) for role, index in zip(ROLES, indices)]))
    if not plan:
        log.skip("Frames: no export and no clips to show.")
        return []
    pictures = {}
    for path in dict.fromkeys(item[2] for item in plan):
        wanted = [index for item in plan if item[2] == path for _, index, _ in item[3]]
        try:
            pictures[path] = _pick_frames(path, wanted)
        except MediaError as error:
            log.flag(f"Frames: cannot decode {path} ({error}).")
    written = []
    for scene, label, path, frames in plan:
        if path not in pictures:
            continue
        cast = ", ".join(scene["cast"] or scene["planned_cast"] or [])
        title = f"Scene {scene['number']} · {scene['start']:.2f}-{scene['end']:.2f} s · {label}" + (
            f" · {cast}" if cast else "")
        frame = "frame" if path == export else "clip frame"
        cells = [(f"{role} · {seconds:.2f} s · {frame} {index}", pictures[path][index])
                 for role, index, seconds in frames]
        out = out_dir / f"{project_id[:8]}_scene_{scene['number']:02d}.png"
        try:
            _sheet(title, " ".join(scene["prompt"].split()), cells, out)
        except OSError as error:
            log.flag(f"Frames: cannot write {out} ({error}).")
            continue
        scene["frames"] = {"file": str(out), "source": label,
                           "picked": [{"role": role, "frame": index, "seconds": _r(seconds, 3)}
                                      for role, index, seconds in frames]}
        written.append(str(out))
    return written


# --- the whole report --------------------------------------------------------------------------------

def protected(path: Path) -> Path | None:
    """The project or outputs folder `path` lies in (this check writes nothing there), else None."""
    target = Path(path).resolve()
    for folder in (timeline.DIR, config.OUTPUTS_DIR):
        base = Path(folder).resolve()
        if target == base or base in target.parents:
            return base
    return None


def build_report(project_id: str, export: Path | None = None, frames: Path | None = None) -> dict:
    """The report as a dict (see the module docstring); `frames`: a folder for the contact sheets."""
    log = _Log()
    report = {"project": {"id": project_id}, "export": None, "container": None, "loudness": None, "scenes": [],
              "joins": [], "stage5": [], "written": [], "flags": log.flags, "unchecked": log.unchecked}
    try:
        project = timeline.load(project_id)
    except ValueError as error:
        log.flag(f"Project {project_id}: {error}")
        return report
    state = timeline.prepared(project)
    keys = list(state.get("assets") or [])
    records = [timeline.asset(project, keys[index], verify=False) if index < len(keys) and keys[index] else None
               for index in range(len(project.scenes))]
    failure = timeline.failure(project)
    report["project"] = {"id": project.id, "title": project.title, "revision": project.revision, "key": project.key,
                         "scenes": len(project.scenes), "seconds": project.seconds, "aspect": project.aspect,
                         "audio": project.audio, "clips": sum(bool(record) for record in records),
                         "last_failure": failure}
    path, report["export"] = _find_export(project, state, export, log)
    facts = _container(path, project, records, log) if path is not None else None
    report["container"] = facts
    track = None
    if facts is not None and facts.get("audio"):
        try:
            samples, rate = media_audio.read(path, rate=None, channels=None)  # the one decode of the sound
        except MediaError as error:
            log.flag(f"Export: cannot decode its sound ({error}).")
        else:
            seconds = samples.shape[1] / rate
            facts["audio"]["decoded_seconds"] = _r(seconds, 3)
            if facts["duration_ok"] and abs(seconds - facts["expected_duration"]) > 1 / FPS + 1e-6:
                log.flag(f"Export: its sound lasts {seconds:.3f} s; its scenes add up to "
                         f"{facts['expected_duration']:g} s.")
            track = Track(samples, rate)
            loudness = _loudness(samples, rate) or {}
            del samples
            loudness.update(rms_dbfs=_r(track.level([(0.0, track.seconds)]), 1),
                            sample_peak_dbfs=_r(20 * math.log10(track.peak) if track.peak > 0 else FLOOR_DB, 1))
            report["loudness"] = loudness
            if track.peak >= CLIPPING:
                log.flag(f"Export: the sound clips (sample peak {loudness['sample_peak_dbfs']:+.1f} dBFS).")
            if loudness.get("integrated_lufs") is not None and loudness["integrated_lufs"] < QUIET_LUFS:
                log.flag(f"Export: the movie is very quiet overall ({loudness['integrated_lufs']:.1f} LUFS, below "
                         f"{QUIET_LUFS:g} LUFS).")
    elif facts is not None and not project.audio:
        log.skip("Sound: the project's sound is off, so no sound was checked.")
    scenes = _scenes(project, keys, records, log)
    report["scenes"] = scenes
    if track is not None:
        report["joins"] = _measure(scenes, track, log)
    report["stage5"] = _stage5(scenes, facts, track is not None, log)
    if frames is not None:
        report["written"] += _contact_sheets(project.id, scenes, records, path, facts, Path(frames), log)
    return report


# --- text --------------------------------------------------------------------------------------------

def _db_text(value) -> str:
    return "n/a" if value is None else f"{value:.1f} dBFS"


def _status(entry: dict) -> str:
    return {"ok": "ok", "flag": "FLAG", "unchecked": "not checked"}.get(entry.get("status"), "")


def _short(text: str, limit: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def format_report(report: dict) -> str:
    out: list[str] = []
    add = out.append
    project = report["project"]
    add("Timeline export check")
    if "title" not in project:
        add(f"Project     {project['id']}: cannot be opened")
    else:
        add(f"Project     {project['title']} · {project['id']} · revision {project['revision']}")
        add(f"Plan        {project['scenes']} scenes · {project['seconds']} s · {project['aspect']} · "
            f"sound {'on' if project['audio'] else 'off'} · clips {project['clips']}/{project['scenes']}")
        if project.get("last_failure"):
            failure = project["last_failure"]
            add(f"Last failed job  {failure['kind']}: {_short(failure['error'])}")
    export, facts, loudness = report["export"], report["container"], report["loudness"]
    if export is not None:
        if export.get("path") is None:
            add("Export      none: the project has no current export"
                + (f" (older export: {export['older']})" if export.get("older") else ""))
        elif export.get("source") == "--export":
            current = "the project's current export" if export.get("current") else "not the project's current export"
            add(f"Export      {export['path']} (--export; {current})")
        else:
            add(f"Export      {export['path']} (the project's current export)")
    if facts is not None:
        frames = f"{facts['frames']} frames" if facts["frames"] is not None else "frame count unknown"
        add(f"  Container {facts['duration']} s (scenes {facts['expected_duration']:g} s) · {frames} "
            f"(scenes {facts['expected_frames']}) · {facts['fps']} fps · {facts['width']}×{facts['height']} · "
            f"{facts['video_codec'] or 'no picture'}")
        audio = facts.get("audio")
        if audio:
            add(f"  Audio     {audio['codec']} · {audio['rate']} Hz · {audio['channels']} channel(s) ({audio['layout']})"
                + (f" · {audio['decoded_seconds']} s decoded" if audio.get("decoded_seconds") is not None else ""))
        else:
            add("  Audio     none")
    if loudness:
        parts = []
        if loudness.get("integrated_lufs") is not None:
            parts.append(f"{loudness['integrated_lufs']:.1f} LUFS integrated")
        if loudness.get("true_peak_dbtp") is not None:
            parts.append(f"true peak {loudness['true_peak_dbtp']:.1f} dBTP")
        if loudness.get("range_lu") is not None:
            parts.append(f"range {loudness['range_lu']:.1f} LU")
        parts += [f"RMS {_db_text(loudness.get('rms_dbfs'))}", f"sample peak {_db_text(loudness.get('sample_peak_dbfs'))}"]
        add("  Loudness  " + " · ".join(parts))
    for scene in report["scenes"]:
        add("")
        transition = {"first": "first scene", "cut": "intentional cut", "continues": "continues the previous scene"}
        add(f"Scene {scene['number']} · {scene['start']:.2f}–{scene['end']:.2f} s · {scene['seconds']} s · "
            f"{transition[scene['transition']]}")
        add(f"  Picture     {_short(scene['prompt'])}")
        clip = scene["clip"]
        if clip:
            stems = ", ".join(clip["stems"]) or "none (mixed sound only)"
            add(f"  Clip        {clip['key'][:12]} · {clip['width']}×{clip['height']} · {clip['fps']} fps · "
                f"{clip['duration']} s · sound stems: {stems}" + (f" · {clip['note']}" if clip["note"] else ""))
            add(f"  Cast        {', '.join(scene['cast']) or 'nobody'} · voices: {', '.join(scene['voices']) or 'none'}")
        else:
            planned = scene["planned_cast"]
            add("  Clip        none yet" + (f" · planned cast: {', '.join(planned)}" if planned else ""))
        label = "  Background  "
        if scene["background_text"]:
            add(f"{label}\"{_short(scene['background_text'], 90)}\"")
            label = " " * len(label)
        background = scene.get("background")
        if background and background.get("rms_dbfs") is not None:
            add(f"{label}{_db_text(background['rms_dbfs'])} RMS outside the {background['left_out']} "
                f"({background['seconds']:.1f} s; speech band {_db_text(background['band_dbfs'])}; "
                f"{100 * background['audible_share']:.0f}% of 50 ms blocks audible) · {_status(background)}")
        elif background:
            add(f"{label}not measured: {background['reason']}")
        for line in scene["lines"]:
            speaker = line["character"] or (f"@{line['voice']}" if line.get("voice") else "the scene's character")
            if "start" not in line:
                when = (f"planned at {line['planned_start']:.2f} s" if line.get("planned_start") is not None
                        else "placed automatically")
                add(f"  Line {line['number']:<6} {speaker}, {when} · \"{_short(line['text'], 80)}\"")
                continue
            voice = f" ({line['voice']})" if line.get("voice") and line["character"] else ""
            add(f"  Line {line['number']:<6} {speaker}{voice} {line['scene_start']:.2f}–{line['scene_end']:.2f} s · "
                f"movie {line['start']:.2f}–{line['end']:.2f} s · \"{_short(line['text'], 80)}\"")
            if line.get("rise_db") is not None:
                add(f"              speech band {_db_text(line['band_dbfs'])}, {line['rise_db']:+.1f} dB over "
                    f"{line['reference_from']} · {_status(line)}")
            elif line.get("status"):
                add(f"              {_status(line)}")
        for effect in scene["effects"]:
            add(f"  Effect {effect['number']:<4} \"{_short(effect['text'], 60)}\" at {effect['scene_at']:.2f} s for "
                f"{effect['seconds']:g} s · movie {effect['at']:.2f} s")
            if effect.get("rise_db") is not None:
                note = f" ({effect['note']})" if effect.get("note") else ""
                add(f"              loudest 50 ms {_db_text(effect['peak_block_dbfs'])} vs "
                    f"{_db_text(effect['reference_dbfs'])} {effect['reference_from']}: {effect['rise_db']:+.1f} dB"
                    f"{note} · {_status(effect)}")
            elif effect.get("status"):
                add(f"              {_status(effect)}")
        if scene.get("frames"):
            add(f"  Frames      {scene['frames']['file']}")
    if report["joins"]:
        add("")
        add("Joins")
        for join in report["joins"]:
            left, right = join["scenes"]
            if join.get("rms_dbfs") is None:
                add(f"  {left} → {right} at {join['at']:.2f} s · {_status(join)}")
                continue
            sides = " / ".join(_db_text(value) for value in join["backgrounds_dbfs"])
            add(f"  {left} → {right} at {join['at']:.2f} s · {_db_text(join['rms_dbfs'])} RMS within {JOIN_HALF:g} s · "
                f"quietest 50 ms {_db_text(join['quietest_block_dbfs'])} · backgrounds {sides} · {_status(join)}")
    if report["stage5"]:
        add("")
        add("Stage 5 test film (30–60 s, two characters, several shots, at least two lines, audible background)")
        for item in report["stage5"]:
            mark = {True: "ok  ", False: "FLAG", None: "--  "}[item["ok"]]
            add(f"  {mark}  {item['criterion']}: {item['detail']}")
    if report["written"]:
        add("")
        add("Written")
        for name in report["written"]:
            add(f"  {name}")
    add("")
    if report["unchecked"]:
        add(f"Not checked ({len(report['unchecked'])})")
        for text in report["unchecked"]:
            add(f"  - {text}")
    if report["flags"]:
        add(f"Flags ({len(report['flags'])})")
        for number, text in enumerate(report["flags"], 1):
            add(f"  {number}. {text}")
    else:
        add("Flags: none")
    return "\n".join(out)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", help="the timeline project's id (32 hexadecimal characters)")
    parser.add_argument("--export", type=Path, help="the movie to check instead of the project's current export")
    parser.add_argument("--frames", type=Path, help="a folder for a contact sheet (PNG) per scene")
    parser.add_argument("--json", type=Path, help="write the report as JSON to this file")
    args = parser.parse_args(argv)
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(errors="replace")  # a console without UTF-8 still gets the whole report
    refused = {option: folder for option, target in (("--frames", args.frames), ("--json", args.json))
               if target is not None and (folder := protected(target)) is not None}
    report = build_report(args.project, args.export, None if "--frames" in refused else args.frames)
    for option, folder in refused.items():
        report["flags"].append(f"Output: {option} was not written: {getattr(args, option[2:])} is inside {folder}, "
                               "which this check never writes to. Choose another folder.")
    if args.json is not None and "--json" not in refused:
        report["written"].append(str(args.json))
        try:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as error:
            report["written"].remove(str(args.json))
            report["flags"].append(f"Output: --json was not written: {args.json} ({error}).")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
