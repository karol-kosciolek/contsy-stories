# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Video files on PyAV: probing, decoding, and H.264 encoding with the settings of the product.

Frames travel as `av.VideoFrame`; `to_rgb` / `from_rgb` convert to and from (height, width, 3) uint8 arrays.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Optional, Union

import av
import numpy as np

from . import audio as _audio
from . import graph as g
from .errors import MediaError

AI_COMMENT = "AI-generated; Contsy"


@dataclass(frozen=True)
class Info:
    """What a file says about itself. None: the file does not say."""
    duration: Optional[float]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    has_audio: bool


def _packet_time(packet) -> float:
    stamp = packet.dts if packet.dts is not None else packet.pts
    return float(stamp * packet.time_base) if stamp is not None else 0.0


_UNSPECIFIED = {"colorspace": 2, "color_range": 0, "color_primaries": 2, "color_trc": 2}
COLOR_FIELDS = tuple(_UNSPECIFIED)


def copy_color(source, target) -> None:
    """Copy the colour description (matrix, range, primaries, transfer) of a frame or an encoder context to
    another one, leaving out what is unspecified. The ffmpeg program did this for every re-encode: without
    it a tag-aware reader decodes the picture with the wrong matrix (BT.601 for what is BT.709)."""
    for name in COLOR_FIELDS:
        value = getattr(source, name, None)
        if value is not None and value != _UNSPECIFIED[name]:
            setattr(target, name, value)


def probe(path: str | Path) -> Info:
    try:
        with av.open(str(path)) as container:
            total = container.duration
            stream = container.streams.video[0] if container.streams.video else None
            rate = float(stream.average_rate) if stream is not None and stream.average_rate else None
            return Info(duration=(total / av.time_base) if total else None,
                        width=stream.width if stream is not None else None,
                        height=stream.height if stream is not None else None,
                        fps=rate, has_audio=bool(container.streams.audio))
    except av.error.FFmpegError as error:
        raise MediaError(f"cannot read {path}: {error}") from error


def read_frames(path: str | Path) -> Iterator[av.VideoFrame]:
    """Decoded frames of the first video stream, in presentation order."""
    try:
        with av.open(str(path)) as container:
            yield from container.decode(video=0)
    except av.error.FFmpegError as error:
        raise MediaError(f"cannot decode {path}: {error}") from error


def to_rgb(frame: av.VideoFrame) -> np.ndarray:
    return frame.to_ndarray(format="rgb24")


def from_rgb(array: np.ndarray) -> av.VideoFrame:
    return av.VideoFrame.from_ndarray(np.ascontiguousarray(array, dtype=np.uint8), format="rgb24")


def save_png(frame: av.VideoFrame, path: str | Path) -> None:
    frame.to_image().save(str(path))


class _Interleaver:
    """Writes video packets and queued audio packets to one container in time order."""

    def __init__(self, container):
        self.container, self.audio = container, []

    def video(self, packet) -> None:
        now = _packet_time(packet)
        while self.audio and _packet_time(self.audio[0]) <= now:
            self.container.mux(self.audio.pop(0))
        self.container.mux(packet)

    def finish(self) -> None:
        for packet in self.audio:
            self.container.mux(packet)
        self.audio = []


class Writer:
    """H.264 output, one frame after another on a constant-rate clock (frame n is at n / fps seconds).

    `crf` is the quality (0 is lossless, 17 is visually lossless), `preset` the speed/size trade.
    `comment` goes into the file metadata. The colour description of the first frame (matrix, range,
    primaries, transfer) is written into the stream, unless `colorspace` / `color_range` (FFmpeg's numbers:
    1 = BT.709, 5 = BT.601 PAL, 1 = limited range, 2 = full) say otherwise; they also pick the matrix when
    RGB arrays are converted. Use as a context manager."""

    def __init__(self, path: str | Path, fps: Union[int, Fraction], width: int, height: int, crf: int = 18,
                 preset: str = "medium", pix_fmt: str = "yuv420p", comment: Optional[str] = None,
                 audio: Optional[tuple] = None, audio_from: Optional[Union[str, Path]] = None,
                 colorspace: Optional[int] = None, color_range: Optional[int] = None,
                 color_primaries: Optional[int] = None, color_trc: Optional[int] = None,
                 audio_rate: Optional[int] = None):
        if audio_rate is not None and (audio is not None or audio_from is not None):
            raise ValueError("Choose one audio source.")
        self.path, self.fps, self.count = str(path), Fraction(fps), 0
        self._forced = {"colorspace": colorspace, "color_range": color_range,
                        "color_primaries": color_primaries, "color_trc": color_trc}
        self._tagged = False
        self.width, self.height, self.pix_fmt = width, height, pix_fmt
        self._mixer: Optional[_Interleaver] = None
        self._audio_source = None
        self._audio_track = None
        self._audio_count = 0
        try:
            self.container = av.open(self.path, "w")
            if comment:
                self.container.metadata["comment"] = comment
            self.stream = self.container.add_stream("libx264", rate=self.fps)
            self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, pix_fmt
            self.stream.codec_context.time_base = 1 / self.fps
            self.stream.thread_type = "AUTO"  # PyAV defaults to slice threads, which x264 turns into a worse encode
            self.stream.options = {"crf": str(crf), "preset": preset}
            self._mixer = _Interleaver(self.container)
            if audio is not None:  # (samples, rate): AAC, 192 kbit/s
                samples, rate = audio
                track = self.container.add_stream("aac", rate=rate, layout={1: "mono", 2: "stereo"}[samples.shape[0]])
                track.bit_rate = 192000
                self._mixer.audio = _audio.encode_packets(track, samples, rate)
            elif audio_rate is not None:
                self._audio_track = self.container.add_stream("aac", rate=audio_rate, layout="stereo")
                self._audio_track.bit_rate = 192000
            elif audio_from is not None:  # the first audio stream of that file, packets copied unchanged
                self._audio_source = av.open(str(audio_from))
                if self._audio_source.streams.audio:
                    source_stream = self._audio_source.streams.audio[0]
                    track = self.container.add_stream_from_template(source_stream)
                    for packet in self._audio_source.demux(source_stream):
                        if packet.dts is not None:
                            packet.stream = track
                            self._mixer.audio.append(packet)
        except av.error.FFmpegError as error:
            for opened in (getattr(self, "container", None), self._audio_source):  # do not leak what was opened
                if opened is not None:
                    opened.close()
            raise MediaError(f"cannot write {path}: {error}") from error

    def write_audio(self, samples: np.ndarray) -> None:
        """Feed the next block of stereo PCM to the same AAC encoder and sample clock."""
        track = self._audio_track
        if track is None or samples.ndim != 2 or samples.shape[0] != 2:
            raise ValueError("Streaming audio requires a stereo track and stereo samples.")
        for start in range(0, samples.shape[1], _audio.CHUNK):
            frame = g.audio_frame(samples[:, start:start + _audio.CHUNK], track.rate, self._audio_count + start)
            self._mixer.audio.extend(track.encode(frame))
        self._audio_count += samples.shape[1]

    def write(self, frame: Union[av.VideoFrame, np.ndarray]) -> None:
        if isinstance(frame, np.ndarray):
            frame = from_rgb(frame)
        if (frame.width, frame.height, frame.format.name) != (self.width, self.height, self.pix_fmt):
            options = {name: value for name, value in (("dst_colorspace", self._forced["colorspace"]),
                                                       ("dst_color_range", self._forced["color_range"]))
                       if value is not None}
            frame = frame.reformat(width=self.width, height=self.height, format=self.pix_fmt, **options)
        if not self._tagged:  # the encoder opens with the first frame: the tags must be in place before it
            copy_color(frame, self.stream.codec_context)
            for name, value in self._forced.items():
                if value is not None:
                    setattr(self.stream.codec_context, name, value)
            self._tagged = True
        frame.pts, frame.time_base = self.count, 1 / self.fps
        frame.pict_type = 0  # NONE: a decoded frame carries the picture type of its source; the encoder decides
        self.count += 1
        for packet in self.stream.encode(frame):
            self._mixer.video(packet)

    def close(self) -> None:
        if self.container is None:
            return
        try:
            if self._audio_track is not None:
                self._mixer.audio.extend(self._audio_track.encode(None))
            for packet in self.stream.encode(None):
                self._mixer.video(packet)
            self._mixer.finish()
        finally:
            self.container.close()
            self.container = None
            if self._audio_source is not None:
                self._audio_source.close()

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _decode(container, stream) -> Iterator[av.VideoFrame]:
    """Decoded frames with the start time of the file taken off their timestamps, as the ffmpeg program does for
    every input: a clip whose first frame is at 3 s is a clip that starts at 0 for every trim, fps and overlay."""
    shift = round(Fraction(container.start_time or 0, av.time_base) / stream.time_base)
    for frame in container.decode(stream):
        if shift and frame.pts is not None:
            frame.pts -= shift
        yield frame


def _open_source(stream, graph):
    """A buffer source for a decoded stream with the properties the ffmpeg program gives it: size, pixel
    format, time base, aspect ratio, frame rate (tpad and fps need it) and colour description."""
    context = stream.codec_context
    rate = stream.average_rate or stream.guessed_rate
    base, aspect = stream.time_base, stream.sample_aspect_ratio or Fraction(1, 1)
    args = (f"video_size={context.width}x{context.height}:pix_fmt={context.pix_fmt}:"
            f"time_base={base.numerator}/{base.denominator}:pixel_aspect={aspect.numerator}/{aspect.denominator}")
    if rate:
        args += f":frame_rate={rate.numerator}/{rate.denominator}"
    if context.colorspace != _UNSPECIFIED["colorspace"]:
        args += f":colorspace={context.colorspace}"
    if context.color_range != _UNSPECIFIED["color_range"]:
        args += f":range={context.color_range}"
    return graph.add("buffer", args)


def _drain(sink, writer: "Writer", limit: Optional[int] = None) -> bool:
    """Write every frame the sink has ready. False once `limit` frames (None: no limit) are written."""
    for frame in g.pull_available(sink):
        if limit is not None and writer.count >= limit:
            return False
        writer.write(frame)
    return limit is None or writer.count < limit


def convert_fps(path: str | Path, out_path: str | Path, fps: int, seconds: Optional[float] = None) -> int:
    """Resample the clip to a constant `fps`, picking frames by their timestamps, to a lossless H.264 file.

    The input is already interpolated (RIFE). The last timestamp of an interpolated clip carries no full
    interval, so the end is padded with the final frame and the clip is cut to round(seconds * fps)
    frames: the length of the original clip, and the sound that goes with it, stay in sync.
    Returns the frame count."""
    seconds = seconds if seconds is not None else probe(path).duration
    if not seconds or seconds <= 0 or fps <= 0:
        raise MediaError("Could not determine video duration for FPS conversion.")
    frames = max(1, int(seconds * fps + 0.5))
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            graph = av.filter.Graph()
            source = _open_source(stream, graph)
            end = g.add_chain(graph, source, f"fps={fps},tpad=stop_mode=clone:stop_duration=1,"
                                             f"trim=end_frame={frames},setpts=PTS-STARTPTS,format=yuv420p")
            sink = graph.add("buffersink")
            end.link_to(sink)
            graph.configure()
            with Writer(out_path, fps, stream.width, stream.height, crf=0, preset="fast") as writer:
                wanted = True
                for frame in _decode(container, stream):
                    if not wanted:
                        break
                    g.push(source, frame)
                    wanted = _drain(sink, writer, frames)
                g.push(source, None)
                _drain(sink, writer, frames)
    except av.error.FFmpegError as error:
        raise MediaError(f"FPS conversion failed: {error}") from error
    return frames


def _join_graph(graph, sources, fps: int, overlaps: list, handoffs: Optional[list]):
    """The filter graph that joins the shots; returns its last node.

    Without overlaps each shot after the first starts at the last frame of the one before, so that
    repeated frame is dropped and the motion flows. overlaps[i] > 0: shot i starts that many seconds
    before the end of the previous one (a corrected face); that stretch is cross-faded, but only through
    its first half, where both versions of the motion are still almost the same (over the whole stretch
    the outlines doubled). Where a shot starts is given, not measured: handoffs[i] is the moment, on the
    clock of shot i, of the frame shot i + 1 continues from (its last frame when they do not overlap). A
    probed duration would also count the last frame's interval, which put every overlapped shot one or two
    frames late."""
    count = len(sources)
    if any(overlaps[1:]):
        if handoffs is None or len(handoffs) < count - 1:
            raise MediaError("Overlapping parts need the hand-over moment of every part.")
        ends = []
        for i, source in enumerate(sources):
            trim = "trim=start_frame=1," if i and not overlaps[i] else ""
            ends.append(g.add_chain(graph, source, f"{trim}setpts=PTS-STARTPTS,fps={fps},settb=AVTB,setsar=1,"
                                                   "format=yuv420p"))
        current, start = ends[0], 0.0
        for i in range(1, count):
            start += handoffs[i - 1]  # the output time of shot i's first frame
            over = overlaps[i]
            if over:
                node = graph.add("xfade", f"transition=fade:duration={over / 2:.3f}:offset={start:.4f}")
            else:
                node = graph.add("concat", "n=2:v=1:a=0")
            current.link_to(node, 0, 0)
            ends[i].link_to(node, 0, 1)
            current = node
        return current
    joined = graph.add("concat", f"n={count}:v=1:a=0")
    for i, source in enumerate(sources):
        g.add_chain(graph, source, "trim=start_frame=1,setpts=PTS-STARTPTS" if i else "null").link_to(joined, 0, i)
    return g.add_chain(graph, joined, f"fps={fps}")


def join_segments(parts, out_path: str | Path, fps: int, overlaps: Optional[list] = None,
                  seconds: Optional[float] = None, crf: int = 17, preset: str = "slow",
                  handoffs: Optional[list] = None) -> int:
    """Join shots into one video of `fps` (H.264, CRF 17, slow preset), cut to `seconds` when given.
    Overlapping shots need `handoffs` (see `_join_graph`). Returns the frame count."""
    parts = [str(p) for p in parts]
    if not parts:
        raise MediaError("No video parts to join.")
    overlaps = list(overlaps or []) + [0.0] * (len(parts) - len(overlaps or []))
    limit = None if seconds is None else int(np.ceil(seconds * fps - 1e-9))
    containers = []
    try:
        containers = [av.open(p) for p in parts]
        streams = [c.streams.video[0] for c in containers]
        graph = av.filter.Graph()
        sources = [_open_source(s, graph) for s in streams]
        end = _join_graph(graph, sources, fps, overlaps, handoffs)
        sink = graph.add("buffersink")
        end.link_to(sink)
        graph.configure()
        with Writer(out_path, fps, streams[0].width, streams[0].height, crf=crf, preset=preset) as writer:
            wanted = True
            for container, stream, source in zip(containers, streams, sources):
                for frame in _decode(container, stream):
                    if not wanted:
                        break
                    g.push(source, frame)
                    wanted = _drain(sink, writer, limit)
                g.push(source, None)
                wanted = wanted and _drain(sink, writer, limit)
            return writer.count
    except av.error.FFmpegError as error:
        raise MediaError(f"joining the shots failed: {error}") from error
    finally:
        for container in containers:
            container.close()


def join_timeline(clips, out_path: str | Path, fps: int, progress_cb=None, *, sound_stems=None,
                  background_fallback=None, on_silent_background=None, transitions=None, sound_levels=None,
                  episode=None) -> int:
    """Join complete, equally sized scenes on one frame and stereo sample clock.

    sound_stems: (background, voice) per scene, or (background, voice, effects) with the scene's effects overlay
    (see audio.timeline_soundtrack). sound_levels: optional (voice_db, background_db) per scene for the soundtrack
    mixed from `sound_stems`; clips without separate stems keep their own sound.
    episode: a series episode's mix (engine/media/episode_mix.py) instead of the scene-by-scene one: a dict with
    "locations" (a location key per scene), "music" (a file or None) and "report" (an episode_mix.MixReport that is
    filled in)."""
    clips = list(clips)
    if not clips:
        raise MediaError("The timeline has no scenes.")
    infos = [probe(path) for path, _ in clips]
    width, height = infos[0].width, infos[0].height
    if not width or not height or any((info.width, info.height, info.fps) != (width, height, fps)
                                      for info in infos):
        raise MediaError("Every scene must have the same frame size and frame rate.")
    if sound_stems is not None and len(sound_stems) != len(clips):
        raise MediaError("A soundtrack source is required for each scene.")
    if sound_levels is not None:
        sound_levels = list(sound_levels)
        if len(sound_levels) != len(clips):
            raise MediaError("A sound level is required for each scene.")
    audio_rate = _audio.RATE if sound_stems is not None or any(info.has_audio for info in infos) else None
    soundtrack = None
    if sound_stems is not None and episode is not None:
        from . import episode_mix

        locations = list(episode.get("locations") or [])
        scenes = []
        for index, ((background, voice, *effects), (_, seconds)) in enumerate(zip(sound_stems, clips)):
            voice_db, background_db = (sound_levels[index] if sound_levels is not None else (0.0, 0.0)) or (0.0, 0.0)
            scenes.append(episode_mix.SceneSound(background, voice, seconds, effects[0] if effects else None,
                                                 locations[index] if index < len(locations) else "", voice_db,
                                                 background_db))
        soundtrack = iter(episode_mix.episode_soundtrack(scenes, music=episode.get("music"),
                                                         background_fallback=background_fallback,
                                                         report=episode.get("report")))
    elif sound_stems is not None:
        stems = ((background, voice, seconds, *effects)
                 for (background, voice, *effects), (_, seconds) in zip(sound_stems, clips))
        soundtrack = _audio.timeline_soundtrack(stems, background_fallback=background_fallback,
                                                on_silent=on_silent_background, levels=sound_levels)
    total = sum(round(seconds * fps) for _, seconds in clips)  # a scene may last tenths of a second
    try:
        with Writer(out_path, fps, width, height, crf=17, comment=AI_COMMENT, audio_rate=audio_rate) as writer:
            for number, ((path, seconds), info) in enumerate(zip(clips, infos), 1):
                label = f"Rendering scene {number}/{len(clips)}…"
                if progress_cb:
                    progress_cb(writer.count / total, label)
                if soundtrack is not None:
                    writer.write_audio(next(soundtrack))
                elif audio_rate is not None:
                    if info.has_audio:
                        samples, _ = _audio.read(path, rate=audio_rate, seconds=seconds)
                        samples = _audio.fit(samples, audio_rate, seconds)
                    else:
                        samples = np.zeros((2, int(round(audio_rate * seconds))), dtype=np.float32)
                    writer.write_audio(samples)
                end = writer.count + int(round(seconds * fps))
                transition = (transitions or {}).get(number - 1)
                for frame in _transition_frames(path, transition):
                    if writer.count >= end:
                        raise MediaError(f"Scene {number} is longer than its timeline duration.")
                    writer.write(frame)
                    if progress_cb and writer.count % fps == 0:
                        progress_cb(writer.count / total, label)
                if writer.count != end:
                    raise MediaError(f"Scene {number} is shorter than its timeline duration.")
            return writer.count
    except av.error.FFmpegError as error:
        raise MediaError(f"Timeline export failed: {error}") from error


def _transition_frames(path: str | Path, transition: str | Path | None) -> Iterator[av.VideoFrame]:
    """Replace the opening frames while keeping the source clip's frame clock."""
    with closing(read_frames(path)) as frames:
        if transition is not None:
            with closing(read_frames(transition)) as bridge:
                for frame in bridge:
                    if next(frames, None) is None:
                        raise MediaError("The transition is longer than its scene.")
                    yield frame
        yield from frames


def _frames_from(path: str | Path, seconds: float) -> Iterator[av.VideoFrame]:
    """Decoded frames starting a little before `seconds` (the seek lands on the previous keyframe)."""
    try:
        with av.open(str(path)) as container:
            container.seek(max(int((seconds - 1.0) * av.time_base), 0) + (container.start_time or 0))
            yield from _decode(container, container.streams.video[0])
    except av.error.FFmpegError as error:
        raise MediaError(f"cannot read a frame from {path}: {error}") from error


def last_frame(path: str | Path, png: str | Path) -> None:
    """Save the last frame as PNG (the first frame of the next shot)."""
    frame = None
    for frame in _frames_from(path, max((probe(path).duration or 0.0) - 1.0, 0.0)):
        pass
    if frame is None:
        raise MediaError(f"{path} has no video frames")
    save_png(frame, png)


def frame_at(path: str | Path, seconds: float, png: str | Path) -> None:
    """Save the first frame at or after `seconds` as PNG (the last frame if the video ends sooner)."""
    frame = None
    for frame in _frames_from(path, seconds):
        if frame.pts is not None and float(frame.pts * frame.time_base) >= seconds - 1e-6:
            break
    if frame is None:
        raise MediaError(f"{path} has no video frames")
    save_png(frame, png)


def splice(full: str | Path, patch: str | Path, first: int, stop: int, out_path: str | Path, crf: int = 14) -> int:
    """Put the frames of `patch` in place of frames [first, stop) of `full` (same frame rate), H.264.
    A patch shorter than the gap leaves the video shorter. Returns the frame count."""
    try:
        with av.open(str(full)) as source:
            stream = source.streams.video[0]
            with Writer(out_path, Fraction(stream.average_rate), stream.width, stream.height, crf=crf) as writer:
                for index, frame in enumerate(source.decode(stream)):
                    if index == first:
                        for number, patch_frame in enumerate(read_frames(patch)):
                            if number >= stop - first:
                                break
                            writer.write(patch_frame)
                    if not first <= index < stop:
                        writer.write(frame)
                return writer.count
    except av.error.FFmpegError as error:
        raise MediaError(f"splicing failed: {error}") from error


def _aac_track(target, samples: np.ndarray, rate: int):
    """Add an AAC stream (192 kbit/s) to an output container; returns its encoded packets."""
    track = target.add_stream("aac", rate=rate, layout={1: "mono", 2: "stereo"}[samples.shape[0]])
    track.bit_rate = 192000
    return _audio.encode_packets(track, samples, rate)


def remux(path: str | Path, out_path: str | Path, comment: Optional[str] = None,
          audio: Optional[tuple] = None, keep_audio: bool = True) -> None:
    """Copy the streams of a file into a new one without re-encoding. `comment` sets the file's comment tag;
    `audio` = (samples, rate) adds an AAC track; keep_audio=False leaves the file's own audio out (so `audio`
    replaces it)."""
    try:
        with av.open(str(path)) as source, av.open(str(out_path), "w") as target:
            if comment:
                target.metadata["comment"] = comment
            mapping = {s.index: target.add_stream_from_template(s)
                       for s in [*source.streams.video[:1], *(source.streams.audio if keep_audio else [])]}
            mixer = _Interleaver(target)
            if audio is not None:
                mixer.audio = _aac_track(target, *audio)
            for packet in source.demux(*[s for s in source.streams if s.index in mapping]):
                if packet.dts is None:  # the empty packet that ends a stream
                    continue
                packet.stream = mapping[packet.stream.index]
                mixer.video(packet)
            mixer.finish()
    except av.error.FFmpegError as error:
        raise MediaError(f"cannot copy {path}: {error}") from error


def tag_metadata(path: str | Path) -> None:
    """Add the 'AI-generated' comment to an existing video file (a remux, the file is replaced)."""
    path = Path(path)
    tagged = path.with_name(path.stem + "_tagged" + path.suffix)
    remux(path, tagged, comment=AI_COMMENT)
    tagged.replace(path)


# Filters that change the frame rate (or the spacing of timestamps); with any other chain the output keeps the rate of the
# input stream, which is what the ffmpeg program gave the encoder.
_RATE_FILTERS = frozenset({"fps", "framerate", "framestep", "select", "setpts", "minterpolate", "mpdecimate", "decimate",
                           "telecine", "tinterlace", "interlace", "fieldmatch"})


def _changes_rate(chain: Optional[str]) -> bool:
    return any(name.lower() in _RATE_FILTERS for name, _ in g.split_chain(chain)) if chain else False


def _rate_of(first, second, fallback: Fraction) -> Fraction:
    """Frame rate of a constant-rate stream from the timestamps of its first two frames."""
    if first.pts is None or second.pts is None or second.pts <= first.pts:
        return fallback
    return (1 / ((second.pts - first.pts) * first.time_base)).limit_denominator(1001)


def reencode(path: str | Path, out_path: str | Path, chain: Optional[str] = None, crf: int = 18,
             preset: str = "medium", comment: Optional[str] = None, audio: Optional[tuple] = None,
             max_seconds: Optional[float] = None) -> int:
    """Decode a video, run it through an optional filter chain, and encode it as H.264 yuv420p.
    The output frame rate and size are the ones the chain delivers (fps=25 and crop change them). `audio` = (samples, rate)
    adds an AAC track; `max_seconds` cuts the video. Returns the frame count."""
    writer: Optional[Writer] = None
    held: list = []
    count = 0
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fallback = Fraction(stream.average_rate or stream.guessed_rate)
            changes_rate = _changes_rate(chain)
            graph = av.filter.Graph()
            source = _open_source(stream, graph)
            end = g.add_chain(graph, source, f"{chain + ',' if chain else ''}format=yuv420p")
            sink = graph.add("buffersink")
            end.link_to(sink)
            graph.configure()

            def feed(frames) -> bool:
                """Write frames; the writer starts once the rate is known (two frames, or the end)."""
                nonlocal writer, count
                for frame in frames:
                    held.append(frame)
                    if writer is None and len(held) < 2:
                        continue
                    if writer is None:
                        # a coarse time base (1/1000 in mkv and webm) makes the first interval of a 24 fps clip 42 ms:
                        # measuring is only for chains that really change the rate
                        rate = _rate_of(held[0], held[1], fallback) if changes_rate else fallback
                        writer = Writer(out_path, rate, held[0].width, held[0].height, crf=crf, preset=preset,
                                        comment=comment, audio=audio)
                    limit = None if max_seconds is None else int(np.ceil(max_seconds * writer.fps - 1e-9))
                    while held:
                        if limit is not None and writer.count >= limit:
                            held.clear()
                            return False
                        writer.write(held.pop(0))
                return True

            wanted = True
            for frame in _decode(container, stream):
                if not wanted:
                    break
                g.push(source, frame)
                wanted = feed(g.pull_available(sink))
            g.push(source, None)
            if wanted:
                feed(g.pull_available(sink))
            if writer is None and held:  # a single frame
                writer = Writer(out_path, fallback, held[0].width, held[0].height, crf=crf, preset=preset,
                                comment=comment, audio=audio)
                writer.write(held.pop(0))
            count = writer.count if writer is not None else 0
    except av.error.FFmpegError as error:
        raise MediaError(f"cannot re-encode {path}: {error}") from error
    finally:
        if writer is not None:
            writer.close()
    return count


def combine_video_audio(video: str | Path, audio: Optional[str | Path], out_path: str | Path, tag_ai: bool = True,
                        compress: bool = False) -> None:
    """Put a soundtrack under a video (AAC) and tag the file 'AI-generated'.

    Narration longer than the video must not be cut: the last frame is held until the whole line and a
    short breath (0.3 s) are done. Otherwise the video is copied without re-encoding over the full length
    and a shorter soundtrack gets silence after it. `compress` always re-encodes the picture: the
    lossless intermediate of convert_fps is H.264 High 4:4:4, which phones and Windows players cannot play."""
    comment = AI_COMMENT if tag_ai else None
    if not (audio and Path(str(audio)).exists()):
        if compress:
            reencode(video, out_path, comment=comment)
        else:
            remux(video, out_path, comment=comment)
        return
    v_len, a_len = probe(video).duration or 0.0, probe(audio).duration or 0.0
    samples, rate = _audio.read(audio, rate=None, channels=None)
    if a_len - v_len > 0.05:
        total = round(a_len + 0.3, 2)
        reencode(video, out_path, chain=f"tpad=stop_mode=clone:stop_duration={total - v_len:.2f}", comment=comment,
                 audio=(_audio.fit(samples, rate, total), rate), max_seconds=total)
        return
    track = (_audio.fit(samples, rate, round(v_len, 3)) if v_len else samples, rate)
    if compress:
        reencode(video, out_path, comment=comment, audio=track)
    else:
        try:
            remux(video, out_path, comment=comment, audio=track)
        except MediaError:  # a copy can fail for an odd stream: re-encode the picture instead
            reencode(video, out_path, comment=comment, audio=track)


def _feather_mask(width: int, height: int) -> np.ndarray:
    """Alpha (0..1) of a pasted patch: its inner rectangle opaque, the edge blurred so the seam disappears."""
    from PIL import Image, ImageDraw, ImageFilter

    feather = max(8, int(min(width, height) * 0.12))
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle([feather, feather, width - feather, height - feather], fill=255)
    return np.asarray(mask.filter(ImageFilter.GaussianBlur(feather / 2)), dtype=np.float32) / 255.0


def _planes(array: np.ndarray, width: int, height: int):
    """Y, U, V views of a yuv420p frame as `to_ndarray` returns it: the three planes one after another,
    cut into rows of `width`. U and V can start in the middle of a row (when the height is not a multiple
    of 4), so they are taken from the flat data."""
    flat = array.reshape(-1)
    luma = width * height
    chroma = luma // 4
    return (flat[:luma].reshape(height, width), flat[luma:luma + chroma].reshape(height // 2, width // 2),
            flat[luma + chroma:luma + 2 * chroma].reshape(height // 2, width // 2))


def _blend(base: np.ndarray, patch: np.ndarray, alpha: np.ndarray, x: int, y: int) -> None:
    """base[y:, x:] = base * (1 - alpha) + patch * alpha, clipped to the base; in place."""
    height, width = base.shape
    top, left = max(y, 0), max(x, 0)
    bottom, right = min(y + alpha.shape[0], height), min(x + alpha.shape[1], width)
    if bottom <= top or right <= left:
        return
    window = (slice(top - y, bottom - y), slice(left - x, right - x))
    region = base[top:bottom, left:right].astype(np.float32)
    mixed = region * (1.0 - alpha[window]) + patch[window].astype(np.float32) * alpha[window]
    base[top:bottom, left:right] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)


def overlay_layers(base: str | Path, layers: list, out_path: str | Path) -> int:
    """Paste video patches (path, x, y, width, height) over a base video with soft edges.

    Used for lip sync of several characters: each face is processed on its own and goes back to its
    place; the blurred edge hides the seam between the patch and the rest of the frame. A patch video
    that ends before the base holds its last frame (flashing back to the base would show a different
    mouth). The picture is composited in the planes of the video (no colour conversion), H.264 CRF 16,
    no sound. Returns the frame count."""
    if any(int(w) % 2 or int(h) % 2 for _, _, _, w, h in layers):
        raise MediaError("patch sizes must be even")
    containers, count = [], 0
    try:
        with av.open(str(base)) as source:
            stream = source.streams.video[0]
            width, height = stream.width, stream.height
            if width % 2 or height % 2:
                raise MediaError(f"the base video size {width}x{height} is not supported for overlays")
            containers = [av.open(str(path)) for path, *_ in layers]
            feeds = [iter(_decode(c, c.streams.video[0])) for c in containers]
            masks = [_feather_mask(int(w), int(h)) for _, _, _, w, h in layers]
            masks_c = [m.reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2).mean(axis=(1, 3)) for m in masks]
            current: list = [None] * len(layers)
            upcoming = [next(feed, None) for feed in feeds]

            def at(patch_frame) -> float:
                return float(patch_frame.pts * patch_frame.time_base)

            with Writer(out_path, Fraction(stream.average_rate), width, height, crf=16) as writer:
                for frame in _decode(source, stream):
                    now = float(frame.pts * frame.time_base)
                    array = frame.to_ndarray(format="yuv420p")
                    y_plane, u_plane, v_plane = _planes(array, width, height)
                    for k, (_, x, y, w, h) in enumerate(layers):
                        while upcoming[k] is not None and at(upcoming[k]) <= now + 1e-6:
                            current[k], upcoming[k] = upcoming[k], next(feeds[k], None)
                        if current[k] is None:
                            continue
                        patch = current[k].reformat(width=int(w), height=int(h), format="yuv420p",
                                                    interpolation="LANCZOS").to_ndarray()
                        py, pu, pv = _planes(patch, int(w), int(h))
                        _blend(y_plane, py, masks[k], int(x), int(y))
                        _blend(u_plane, pu, masks_c[k], int(x) // 2, int(y) // 2)
                        _blend(v_plane, pv, masks_c[k], int(x) // 2, int(y) // 2)
                    composed = av.VideoFrame.from_ndarray(array, format="yuv420p")
                    copy_color(frame, composed)
                    writer.write(composed)
                count = writer.count
    except av.error.FFmpegError as error:
        raise MediaError(f"overlay failed: {error}") from error
    finally:
        for container in containers:
            container.close()
    return count
