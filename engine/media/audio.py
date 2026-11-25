# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Audio on PyAV: files to float arrays and back, and the soundtrack chain of the product.

Samples are float32 arrays shaped (channels, n). The soundtrack chain is the one the ffmpeg program ran
before (same libavfilter filters and settings): narration over a quieter background that ducks while
the narrator speaks, loudness set to -16 LUFS in two passes.
"""
from __future__ import annotations

import json
from fractions import Fraction
from math import isfinite
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import av
import numpy as np

from . import graph as g
from .errors import MediaError

RATE = 48000
LOUDNESS_TARGET = "I=-16:TP=-1.5:LRA=11"  # -16 LUFS: the level of web video services
CHUNK = 4800  # samples per frame pushed into a filter graph (0.1 s at 48 kHz)
TIMELINE_AUDIO_HANDLE = 0.25  # The generator provides half a second of extra material per scene.
LEVEL_CEILING = 0.891  # -1 dBFS: a scene with its own voice or background level is limited below this peak
PREVIEW_FADE = 0.05  # s: a scene preview starts and stops without a click
BED_TARGET = 0.025  # median 50 ms block level the export gives a scene's room tone
BED_MAX_GAIN = 16.0  # ...lifting a quiet one by at most this factor
BACKGROUND_PEAK = 0.2  # the room tone stays below this peak
# The room tone with its effects stays below this one: a click is a few very sharp milliseconds and needs headroom
# above the room tone, or the limiter flattens it into the rain. Each effect's loudness is set by EFFECT_RANGE_DB;
# this ceiling only keeps the clicks' peaks. Stage 5 film, loudest 50 ms over the rain: a light switch +3.7 dB at
# 0.2 and +7.9 dB at 0.7 with the balancing; the movie's true peak stayed at -1.7 dBTP.
EFFECT_PEAK = 0.7
EFFECT_LIFT_DB = 9.0  # effects over a silent bed: their loud part this far above BED_TARGET
# Every effect's loud part (loudest 50 ms) lands this far above the export's room tone (BED_TARGET), so the creator
# never has to balance effects by hand: a quiet click is lifted, a loud door is brought down to dialogue level.
EFFECT_RANGE_DB = (9.0, 15.0)
# Effects are not ducked with the room tone (its release would swallow a click right after a line); while a line is
# spoken they are only softened by this much.
EFFECT_UNDER_SPEECH_DB = -6.0

_LAYOUTS = {1: "mono", 2: "stereo"}
# Look-ahead limiter; latency=true keeps the output aligned with the input and as long as it.
_LIMITER = "alimiter=limit={:.4f}:attack=5:release=50:level=false:latency=true"


class SilentBackground(MediaError):
    """A scene background has no audible blocks after filtering."""


def read(path: str | Path, rate: Optional[int] = RATE, channels: Optional[int] = 2,
         seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """Decode the first audio stream. `rate` and `channels` of None keep the file's own (more than two
    channels are mixed down to stereo); `seconds` stops after that much audio (a long recording is not
    decoded to the end). Returns (samples, rate)."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise MediaError(f"{path} has no audio")
            stream = container.streams.audio[0]
            context = stream.codec_context
            wanted_channels = channels or min(context.layout.nb_channels, 2)
            wanted_rate = rate or context.sample_rate
            resampler = av.AudioResampler(format="fltp", layout=_LAYOUTS[wanted_channels], rate=wanted_rate)
            parts: List[np.ndarray] = []
            have = 0
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    parts.append(out.to_ndarray())
                    have += parts[-1].shape[1]
                if seconds is not None and have >= seconds * wanted_rate:
                    break
            else:
                parts.extend(out.to_ndarray() for out in resampler.resample(None))
    except av.error.FFmpegError as error:
        raise MediaError(f"cannot read audio from {path}: {error}") from error
    samples = np.concatenate(parts, axis=1) if parts else np.zeros((wanted_channels, 0), np.float32)
    if seconds is not None:
        samples = samples[:, :int(round(seconds * wanted_rate))]
    return samples.astype(np.float32, copy=False), wanted_rate


def encode_packets(stream, samples: np.ndarray, rate: int) -> list:
    """Encode (channels, n) float samples with an audio output stream and flush it; returns the packets."""
    packets = []
    for start in range(0, samples.shape[1], CHUNK):
        packets.extend(stream.encode(g.audio_frame(samples[:, start:start + CHUNK], rate, start)))
    packets.extend(stream.encode(None))
    return packets


def encode_into(container, stream, samples: np.ndarray, rate: int) -> None:
    """Encode samples into an audio stream of an open output and write the packets."""
    for packet in encode_packets(stream, samples, rate):
        container.mux(packet)


def write_wav(path: str | Path, samples: np.ndarray, rate: int = RATE) -> None:
    """16-bit PCM WAV."""
    try:
        with av.open(str(path), "w", format="wav") as container:
            stream = container.add_stream("pcm_s16le", rate=rate, layout=_LAYOUTS[samples.shape[0]])
            encode_into(container, stream, samples, rate)
    except av.error.FFmpegError as error:
        raise MediaError(f"cannot write {path}: {error}") from error


def fit(samples: np.ndarray, rate: int, seconds: float) -> np.ndarray:
    """Exactly `seconds` long: shorter audio gets silence after it, longer audio is cut."""
    length = int(round(seconds * rate))
    if samples.shape[1] >= length:
        return samples[:, :length]
    return np.pad(samples, ((0, 0), (0, length - samples.shape[1])))


def _samples(samples: np.ndarray, count: int) -> np.ndarray:
    """Exactly `count` samples: silence after shorter audio, longer audio cut."""
    if samples.shape[1] >= count:
        return samples[:, :count]
    return np.pad(samples, ((0, 0), (0, count - samples.shape[1])))


def gain(db: float) -> float:
    """Linear amplitude factor of a level in dB."""
    return 10.0 ** (float(db) / 20.0)


def limit_peaks(samples: np.ndarray, rate: int = RATE, ceiling: float = LEVEL_CEILING) -> np.ndarray:
    """The samples with no peak above `ceiling`. Audio that already stays below it is returned unchanged; louder audio
    goes through a look-ahead limiter (same length, same clock), so only the peaks lose level."""
    if not samples.size or float(np.max(np.abs(samples))) <= ceiling:
        return samples
    limited, _ = run_chain(samples, rate, _LIMITER.format(ceiling))
    return np.clip(_samples(limited, samples.shape[1]), -ceiling, ceiling)


def run_chain(samples: np.ndarray, rate: int, text: str) -> Tuple[np.ndarray, int]:
    """Pass samples through a linear filter chain written like ffmpeg -af. Returns (samples, rate):
    the rate is the chain output rate (loudnorm, for one, works at 192 kHz)."""
    graph = av.filter.Graph()
    source = graph.add_abuffer(sample_rate=rate, format="fltp", layout=_LAYOUTS[samples.shape[0]],
                               time_base=Fraction(1, rate))
    end = g.add_chain(graph, source, text)
    planar = graph.add("aformat", "sample_fmts=fltp")
    sink = graph.add("abuffersink")
    end.link_to(planar)
    planar.link_to(sink)
    graph.configure()
    parts: List[np.ndarray] = []
    out_rate = rate
    for start in range(0, samples.shape[1], CHUNK):
        wanted = g.push(source, g.audio_frame(samples[:, start:start + CHUNK], rate, start))
        for frame in g.pull_available(sink):
            parts.append(frame.to_ndarray())
            out_rate = frame.sample_rate
        if not wanted:
            break
    g.push(source, None)
    for frame in g.pull_available(sink):
        parts.append(frame.to_ndarray())
        out_rate = frame.sample_rate
    result = np.concatenate(parts, axis=1) if parts else np.zeros((samples.shape[0], 0), np.float32)
    return result, out_rate


def normalize_loudness(samples: np.ndarray, rate: int = RATE, target: str = LOUDNESS_TARGET) -> np.ndarray:
    """Set the loudness in two passes: measure, then apply a constant gain. Single-pass loudnorm starts
    too loud. If the measurement cannot be read, it falls back to the single pass."""
    with g.libav_messages() as lines:
        run_chain(samples, rate, f"loudnorm={target}:print_format=json")
        g.release()
    norm = f"loudnorm={target}"
    for _, _, message in lines:
        if "input_i" in message and "{" in message:
            try:
                measurement = json.loads(message[message.index("{"):message.rindex("}") + 1])
                measured = [float(measurement[key]) for key in
                            ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")]
                if all(isfinite(value) for value in measured):
                    integrated, peak, spread, threshold, offset = measured
                    norm = (f"loudnorm={target}:measured_I={integrated}:measured_TP={peak}:"
                            f"measured_LRA={spread}:measured_thresh={threshold}:offset={offset}:linear=true")
            except (ValueError, KeyError, TypeError):
                pass
            break
    normalized, _ = run_chain(samples, rate, f"{norm},aresample={rate}")
    return normalized


def voice_over_background(voice: Optional[np.ndarray], background: Optional[np.ndarray], seconds: float,
                          rate: int = RATE) -> np.ndarray:
    """The soundtrack: stereo, `rate` Hz, exactly `seconds` long, loudness -16 LUFS.

    The background is quieter (0.8), fades in and out, has its sub-40 Hz rumble cut off to preserve
    headroom, and ducks while the narrator
    speaks (sidechain). The narration is padded with silence to the length of the film first: the
    sidechain ends with its shorter input, which used to cut the background after the last word."""
    if voice is None and background is None:
        raise MediaError("No audio tracks are available.")
    fit_chain = f"apad,atrim=0:{seconds:.3f}"
    fade = f"afade=t=in:d=0.5,afade=t=out:st={max(seconds - 0.6, 0):.2f}:d=0.6"
    low_cut = "highpass=f=40:poles=2"
    if voice is not None and background is not None:
        premix = _ducked_mix(voice, background, rate, fit_chain, f"{low_cut},{fit_chain},{fade},volume=0.8")
    elif voice is not None:
        premix, _ = run_chain(voice, rate, fit_chain)
    else:
        premix, _ = run_chain(background, rate, f"{low_cut},{fit_chain},{fade}")
    return normalize_loudness(premix, rate)


def _ducked_mix(voice: np.ndarray, background: np.ndarray, rate: int, fit_chain: str,
                background_chain: str) -> np.ndarray:
    graph = av.filter.Graph()
    voice_in = graph.add_abuffer(sample_rate=rate, format="fltp", layout=_LAYOUTS[voice.shape[0]],
                                 time_base=Fraction(1, rate), name="voice")
    background_in = graph.add_abuffer(sample_rate=rate, format="fltp", layout=_LAYOUTS[background.shape[0]],
                                      time_base=Fraction(1, rate), name="background")
    split = graph.add("asplit", "2")
    g.add_chain(graph, voice_in, fit_chain).link_to(split)
    ducked = graph.add("sidechaincompress", "threshold=0.05:ratio=3:attack=50:release=400")
    g.add_chain(graph, background_in, background_chain).link_to(ducked, 0, 0)
    split.link_to(ducked, 0, 1)  # first copy of the narration drives the compressor
    mixed = graph.add("amix", "inputs=2:duration=first:normalize=0")
    ducked.link_to(mixed, 0, 0)
    split.link_to(mixed, 1, 1)  # second copy is mixed in
    end = g.add_chain(graph, mixed, fit_chain)
    planar = graph.add("aformat", "sample_fmts=fltp")
    sink = graph.add("abuffersink")
    end.link_to(planar)
    planar.link_to(sink)
    graph.configure()
    parts: List[np.ndarray] = []
    open_inputs = {"voice": True, "background": True}
    for start in range(0, max(voice.shape[1], background.shape[1]), CHUNK):
        for name, source, samples in (("voice", voice_in, voice), ("background", background_in, background)):
            piece = samples[:, start:start + CHUNK]
            if not open_inputs[name]:
                continue
            if piece.shape[1]:
                open_inputs[name] = g.push(source, g.audio_frame(piece, rate, start))
            else:  # this input has ended: apad can start padding it with silence
                g.push(source, None)
                open_inputs[name] = False
        parts.extend(frame.to_ndarray() for frame in g.pull_available(sink))
        if not any(open_inputs.values()):
            break
    for name, source in (("voice", voice_in), ("background", background_in)):
        if open_inputs[name]:
            g.push(source, None)
    parts.extend(frame.to_ndarray() for frame in g.pull_available(sink))
    return np.concatenate(parts, axis=1)


def _block_rms(samples: np.ndarray, block: int) -> np.ndarray:
    length = samples.shape[1] // block * block
    pieces = samples[:, :length].reshape(samples.shape[0], -1, block)
    return np.sqrt(np.mean(pieces * pieces, axis=(0, 2)))


def _fill_background_edges(samples: np.ndarray, rate: int, reference: float) -> None:
    """Cover the model's quiet edges with steady sound from the same recording."""
    block = round(0.05 * rate)
    levels = _block_rms(samples, block)
    edge_blocks = min(24, levels.size // 2)
    length = edge_blocks * block
    # Prefer a contiguous piece without silence or large foreground sounds.
    distance = np.log(np.maximum(levels, 1e-6) / reference) ** 2
    windows = np.lib.stride_tricks.sliding_window_view(distance, edge_blocks)
    start = int(np.argmin(windows.mean(axis=1))) * block
    texture = samples[:, start:start + length].copy()
    centres = (np.arange(edge_blocks) + 0.5) * block
    fade = np.ones(length, dtype=np.float32)
    inside = min(round(2 * TIMELINE_AUDIO_HANDLE * rate), length)
    fade[inside:] = np.linspace(1.0, 0.0, length - inside, dtype=np.float32)
    for edge, edge_levels, taper in ((samples[:, :length], levels[:edge_blocks], fade),
                                      (samples[:, -length:], levels[-edge_blocks:], fade[::-1])):
        needs_tone = np.clip((0.65 - edge_levels / reference) / 0.5, 0.0, 1.0)
        blend = np.interp(np.arange(length), centres, needs_tone).astype(np.float32) * taper
        edge[:] = edge * np.sqrt(1.0 - blend) + texture * np.sqrt(blend)


def _bed(path: Path, seconds: float, rate: int, effects: Optional[Path] = None) -> np.ndarray:
    """A scene's handle-padded room tone, sub-40 Hz rumble cut. `effects`: the effects overlay laid into the
    background (same clock); it is taken out, so only the room tone is measured and leveled."""
    length = seconds + 2 * TIMELINE_AUDIO_HANDLE
    samples, _ = read(path, rate=rate, seconds=length)
    if samples.shape[1] < round(length * rate) - 1:
        raise MediaError("The scene is missing background audio at the join.")
    samples = fit(samples, rate, length)
    if effects is not None:
        overlay, _ = read(effects, rate=rate, seconds=length)
        samples = samples - fit(overlay, rate, length)
    samples, _ = run_chain(samples, rate, "highpass=f=40:poles=2")
    return samples


def _leveled_background(path: Path, seconds: int, rate: int,
                        effects: Optional[Path] = None) -> Tuple[np.ndarray, float]:
    """The scene's room tone leveled for the export, and the gain it took (see _timeline_background)."""
    samples = _bed(path, seconds, rate, effects)
    levels = _block_rms(samples, round(0.05 * rate))
    audible = levels > 1e-4
    if not np.any(audible):
        raise SilentBackground("The scene background audio is silent.")
    # A transient should not determine the level of the room tone.
    bed_gain = min(BED_MAX_GAIN, BED_TARGET / float(np.median(levels[audible])))
    samples, _ = run_chain(samples * bed_gain, rate,
                           f"alimiter=limit={BACKGROUND_PEAK}:attack=5:release=80:level=false:latency=true")
    reference = float(np.median(_block_rms(samples, round(0.05 * rate))[audible]))
    if reference > 1e-6:
        _fill_background_edges(samples, rate, reference)
    return samples, bed_gain


def _timeline_background(path: Path, seconds: int, rate: int, effects: Optional[Path] = None) -> np.ndarray:
    """The handle-padded background leveled to the export's room-tone level, its quiet edges filled with its
    own steady sound. With `effects` (the overlay already inside the background) only the room tone is
    returned: the effects are added back by _effects_layer, after the leveling, the joins and the fades."""
    return _leveled_background(path, seconds, rate, effects)[0]


def background_loudness(path: str | Path, seconds: float, rate: int = RATE,
                        effects: Optional[str | Path] = None) -> Tuple[float, float]:
    """How the export would hear a handle-padded scene background: (the median level of its audible 50 ms
    blocks after the export's leveling - gain capped at BED_MAX_GAIN - and the share of audible blocks).
    (0.0, 0.0) when it is silent. `effects`: the overlay inside the background, left out of the measure."""
    samples = _bed(Path(path), seconds, rate, Path(effects) if effects is not None else None)
    levels = _block_rms(samples, round(0.05 * rate))
    audible = levels > 1e-4
    if not levels.size or not np.any(audible):
        return 0.0, 0.0
    median = float(np.median(levels[audible]))
    return min(BED_MAX_GAIN, BED_TARGET / median) * median, float(np.mean(audible))


def _effects_layer(path: Path, seconds: int, rate: int) -> np.ndarray:
    """A scene's effects overlay on the scene clock (handles trimmed), sub-40 Hz rumble cut like the bed."""
    length = seconds + 2 * TIMELINE_AUDIO_HANDLE
    samples, _ = read(path, rate=rate, seconds=length)
    samples, _ = run_chain(fit(samples, rate, length), rate, "highpass=f=40:poles=2")
    handle = round(TIMELINE_AUDIO_HANDLE * rate)
    return _samples(samples[:, handle:], int(round(seconds * rate))).copy()


def _effects_gain(layer: np.ndarray, rate: int) -> float:
    """Gain for effects whose room tone is silent (no bed level to follow): their loud part EFFECT_LIFT_DB above
    the export's room-tone level, lifted by at most BED_MAX_GAIN."""
    levels = _block_rms(layer, round(0.05 * rate)) if layer.shape[1] else np.zeros(0)
    audible = levels[levels > 1e-4]
    if not audible.size:
        return 0.0
    return min(BED_MAX_GAIN, BED_TARGET * gain(EFFECT_LIFT_DB) / float(np.quantile(audible, 0.9)))


def _timeline_mix(background: np.ndarray, voice: Optional[Path], seconds: int, rate: int,
                  voice_db: float = 0.0) -> np.ndarray:
    if voice is None:
        return background
    speech, _ = read(voice, rate=rate)
    speech = normalize_loudness(speech, rate, "I=-18:TP=-3:LRA=11")
    if voice_db:
        speech = speech * np.float32(gain(voice_db))
    fit_chain = f"apad,atrim=0:{seconds}"
    return fit(_ducked_mix(speech, background, rate, fit_chain, fit_chain), rate, seconds)


def _scene_levels(levels: Optional[Sequence[Optional[Tuple[float, float]]]], index: int) -> Tuple[float, float]:
    """(voice_db, background_db) of one scene; no levels, or None for the scene, is 0/0."""
    if levels is None:
        return 0.0, 0.0
    try:
        entry = levels[index]
    except IndexError as error:
        raise MediaError("A sound level is required for each scene.") from error
    if entry is None:
        return 0.0, 0.0
    try:
        voice_db, background_db = (float(value) for value in entry)
    except (TypeError, ValueError) as error:
        raise MediaError("A scene's sound levels must be two numbers in dB.") from error
    if not (isfinite(voice_db) and isfinite(background_db)):
        raise MediaError("A scene's sound levels must be two numbers in dB.")
    return voice_db, background_db


def _leveled_mix(background: np.ndarray, voice: Optional[Path], seconds: int, rate: int,
                 levels: Tuple[float, float], *, effects: bool = False) -> np.ndarray:
    """The scene's background (already at its level) with its dialogue at the voice level. A scene whose levels were
    changed, or whose effects may peak over the dialogue, is limited below LEVEL_CEILING; any other scene at 0/0 is
    exactly the mix it always was."""
    voice_db, background_db = levels
    mixed = _timeline_mix(background, voice, seconds, rate, voice_db)
    return limit_peaks(mixed, rate) if voice_db or background_db or effects else mixed


def _speech_gain(voice: Path, count: int, rate: int) -> np.ndarray:
    """Per-sample gain for effects: 1 where nobody speaks, gain(EFFECT_UNDER_SPEECH_DB) during a spoken line, with
    short ramps (the dialogue track is silent between lines)."""
    speech, _ = read(voice, rate=rate)
    speech = _samples(speech, count)
    block = max(1, round(0.01 * rate))
    levels = _block_rms(speech, block)
    if not levels.size or float(np.max(levels)) <= 1e-6:
        return np.ones(count, dtype=np.float32)
    active = (levels > float(np.max(levels)) * 0.05).astype(np.float32)  # within 26 dB of the loudest word
    reach = max(1, round(0.05 / 0.01))  # hold 50 ms around every word, then ramp over 40 ms
    active = np.convolve(active, np.ones(2 * reach + 1, dtype=np.float32), mode="same") > 0
    ramp = np.ones(max(1, round(0.04 / 0.01)), dtype=np.float32)
    active = np.convolve(active.astype(np.float32), ramp / ramp.size, mode="same")
    under = np.repeat(active, block)[:count]
    under = np.pad(under, (0, max(0, count - under.size)))
    return (1.0 - (1.0 - gain(EFFECT_UNDER_SPEECH_DB)) * under).astype(np.float32)


def _scene_out(core: np.ndarray, layer: Optional[np.ndarray], voice: Optional[Path], seconds: int, rate: int,
               levels: Tuple[float, float]) -> np.ndarray:
    """A scene's final mix. The room tone ducks under the dialogue (sidechain); the effects are laid over the result
    at the background level and never pumped by the ducking — a click right after a line keeps its level — only
    softened by EFFECT_UNDER_SPEECH_DB while a line is spoken, so the words stay clear."""
    if layer is None:
        return _leveled_mix(core, voice, seconds, rate, levels)
    voice_db, background_db = levels
    scale = np.float32(gain(background_db)) if background_db else np.float32(1.0)
    mixed = _timeline_mix(core, voice, seconds, rate, voice_db)
    effects = _samples(layer, mixed.shape[1]) * scale
    if voice is not None:
        effects = effects * _speech_gain(voice, mixed.shape[1], rate)
        return limit_peaks(mixed + effects, rate)
    # without dialogue the room tone and its effects together keep the effects' own ceiling, as before
    return limit_peaks(mixed + effects, rate, float(EFFECT_PEAK * scale))


def _effect_regions(layer: np.ndarray, rate: int, gap: float = 0.12) -> list[tuple[int, int]]:
    """(start, stop) sample ranges where the overlay holds an effect: it is silent between effects, so every
    stretch of sound separated by more than `gap` seconds of silence is one effect."""
    block = max(1, round(0.01 * rate))
    active = _block_rms(layer, block) > 1e-5 if layer.shape[1] >= block else np.zeros(0, dtype=bool)
    regions, start, silent = [], None, 0
    for index, sounding in enumerate(active):
        if sounding:
            start = index if start is None else start
            silent = 0
        elif start is not None:
            silent += 1
            if silent * block >= gap * rate:
                regions.append((start * block, (index - silent + 1) * block))
                start, silent = None, 0
    if start is not None:
        regions.append((start * block, (len(active) - silent) * block))
    return regions


def _loudest(piece: np.ndarray, rate: int) -> float:
    """The loudest 50 ms (RMS) of a piece, in 10 ms steps; the whole piece when it is shorter."""
    window, step = round(0.05 * rate), max(1, round(0.01 * rate))
    if piece.shape[1] <= window:
        return float(np.sqrt(np.mean(piece * piece))) if piece.size else 0.0
    power = np.cumsum(np.concatenate(([0.0], np.mean(piece * piece, axis=0, dtype=np.float64))))
    sums = power[window::step] - power[:-window:step][:len(power[window::step])]
    return float(np.sqrt(max(float(np.max(sums)), 0.0) / window))


def _balanced_effects(layer: np.ndarray, rate: int) -> np.ndarray:
    """Every effect with its loud part (its loudest 50 ms) EFFECT_RANGE_DB above the export's room tone, so a
    click is heard over rain and a door does not drown the dialogue, whatever level the generator gave them.
    The creator's background level still applies on top (_scene_out)."""
    low, high = BED_TARGET * gain(EFFECT_RANGE_DB[0]), BED_TARGET * gain(EFFECT_RANGE_DB[1])
    out = layer.copy()
    for start, stop in _effect_regions(layer, rate):
        loud = _loudest(out[:, start:stop], rate)
        if loud <= 1e-6:
            continue
        factor = min(max(loud, low), high) / loud
        if abs(factor - 1.0) > 1e-3:
            out[:, start:stop] *= np.float32(min(factor, BED_MAX_GAIN))
    return out


def _scene_effects(effects: Optional[Path], bed_gain: Optional[float], seconds: int,
                   rate: int) -> Optional[np.ndarray]:
    """A scene's effects overlay on the scene clock, each effect balanced into EFFECT_RANGE_DB over the room tone
    (_balanced_effects); None without effects."""
    if effects is None:
        return None
    layer = _effects_layer(effects, seconds, rate)
    layer = layer * np.float32(bed_gain if bed_gain is not None else _effects_gain(layer, rate))
    return _balanced_effects(layer, rate)


def timeline_soundtrack(stems: Iterable[tuple],
                        rate: int = RATE, *,
                        background_fallback: Optional[Callable[[int, int], Path]] = None,
                        on_silent: Optional[Callable[[int], None]] = None,
                        levels: Optional[Sequence[Optional[Tuple[float, float]]]] = None) -> Iterator[np.ndarray]:
    """Mix two adjoining background tracks at a time while keeping dialogue on its scene clock.

    stems: (background, voice, seconds) per scene, or (background, voice, seconds, effects) when the scene's
    effects overlay is known (the effects laid into its background, same clock). Then only the room tone is
    leveled, edge-filled, joined and faded at the movie's start and end; the effects are laid over it
    afterwards at the room tone's gain, so each plays at its own time and level.
    levels: optional (voice_db, background_db) for each scene, in scene order. The background gain applies after the
    background has been leveled (so the cross-fade at a join moves from one scene's level to the next), the voice gain
    after the speech loudness normalization. A scene with a changed level is limited below LEVEL_CEILING; with no levels,
    or 0/0, the soundtrack is exactly what it was without them."""
    handle = round(TIMELINE_AUDIO_HANDLE * rate)
    angle = np.linspace(0.0, np.pi / 2, 2 * handle, dtype=np.float32)
    fade_out, fade_in = np.cos(angle), np.sin(angle)
    pending = None
    for scene_number, stem in enumerate(stems, 1):
        background, voice, seconds = stem[:3]
        effects = Path(stem[3]) if len(stem) > 3 and stem[3] is not None else None
        scene_levels = _scene_levels(levels, scene_number - 1)
        bed_gain = None
        try:
            raw, bed_gain = _leveled_background(background, seconds, rate, effects)
        except SilentBackground as error:
            if background_fallback is not None:
                try:
                    raw = _timeline_background(background_fallback(scene_number, seconds), seconds, rate)
                except MediaError as fallback_error:
                    raise type(fallback_error)(f"Scene {scene_number}: replacement background failed. "
                                               f"{fallback_error}") from fallback_error
            elif on_silent is not None:
                on_silent(scene_number)
                raw = np.zeros((2, int(round(seconds * rate)) + 2 * handle), dtype=np.float32)
            elif effects is not None:  # silent room tone, but the scene's effects are heard
                raw = np.zeros((2, int(round(seconds * rate)) + 2 * handle), dtype=np.float32)
            else:
                raise SilentBackground(f"Scene {scene_number}: {error}") from error
        layer = _scene_effects(effects, bed_gain, seconds, rate)
        if scene_levels[1]:
            raw = raw * np.float32(gain(scene_levels[1]))
        core = raw[:, handle:-handle].copy()
        if pending is None:
            fade = min(core.shape[1], round(0.5 * rate))
            core[:, :fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        else:
            previous, previous_voice, previous_seconds, tail, previous_levels, previous_layer = pending
            joined = tail * fade_out + raw[:, :2 * handle] * fade_in
            previous[:, -handle:] = joined[:, :handle]
            core[:, :handle] = joined[:, handle:]
            yield _scene_out(previous, previous_layer, previous_voice, previous_seconds, rate, previous_levels)
        pending = core, voice, seconds, raw[:, -2 * handle:].copy(), scene_levels, layer
    if pending is not None:
        core, voice, seconds, _, scene_levels, layer = pending
        fade = min(core.shape[1], round(0.6 * rate))
        core[:, -fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        yield _scene_out(core, layer, voice, seconds, rate, scene_levels)


def scene_mix(background: Optional[str | Path], voice: Optional[str | Path], seconds: int,
              voice_db: float = 0.0, background_db: float = 0.0, rate: int = RATE, *,
              effects: Optional[str | Path] = None) -> np.ndarray:
    """One scene as the timeline export mixes it, for previews: stereo, `rate` Hz, exactly `seconds` long.

    `background` is the scene's handle-padded background (it starts TIMELINE_AUDIO_HANDLE before the scene): it is
    leveled as in the export and trimmed to the scene; `effects` is its effects overlay (see timeline_soundtrack),
    laid over the leveled room tone; `voice` is the dialogue on the scene clock, normalized and mixed over the
    background, which ducks under it. Both take their levels and peaks are limited as in the export. A background
    without audible sound plays as silence. Short fades at both ends keep the preview free of clicks."""
    if background is None and voice is None:
        raise MediaError("No audio tracks are available.")
    voice_db, background_db = _scene_levels([(voice_db, background_db)], 0)
    handle = round(TIMELINE_AUDIO_HANDLE * rate)
    count = int(round(seconds * rate))
    core = np.zeros((2, count), dtype=np.float32)
    layer = None
    if background is not None:
        effects = Path(effects) if effects is not None else None
        bed_gain = None
        try:
            raw, bed_gain = _leveled_background(Path(background), seconds, rate, effects)
        except SilentBackground:
            raw = None
        if raw is not None:
            if background_db:
                raw = raw * np.float32(gain(background_db))
            core = _samples(raw[:, handle:], count).copy()
        layer = _scene_effects(effects, bed_gain, seconds, rate)
    mixed = _samples(_scene_out(core, layer, Path(voice) if voice is not None else None, seconds, rate,
                                (voice_db, background_db)), count).copy()
    fade = min(count // 2, round(PREVIEW_FADE * rate))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        mixed[:, :fade] *= ramp
        mixed[:, count - fade:] *= ramp[::-1]
    return mixed


# The old ffmpeg run (`-ac 1 ... -c:a pcm_s16le`) downmixed in 16-bit integers, where FFmpeg scales the matrix so
# that the channels add up to at most 1: L = R keeps its level. In float the same downmix would add 3 dB for such
# channels (a hotter reference for voice cloning), so the normalisation is asked for explicitly.
VOICE_DOWNMIX = "aresample=48000:ochl=mono:rematrix_maxval=1.0"


def prepare_voice_sample(src: str | Path, dst: str | Path, scan_seconds: float, max_seconds: float) -> None:
    """Any audio file to a 48 kHz mono 16-bit WAV for voice cloning: silence cut at both ends, at most
    `max_seconds` long, loudness set to -18 LUFS (one pass). Only the first `scan_seconds` of a long
    recording are read."""
    samples, rate = read(src, rate=None, channels=None, seconds=scan_seconds)
    trim = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence={}"
    chain = ",".join([trim.format(0.15), "areverse", trim.format(0.3), "areverse",
                      "loudnorm=I=-18:TP=-1.5:LRA=11", VOICE_DOWNMIX])
    prepared, out_rate = run_chain(samples, rate, chain)
    write_wav(dst, prepared[:, :int(max_seconds * out_rate)], out_rate)
