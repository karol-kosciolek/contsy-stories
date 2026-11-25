# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Narzedzia wideo i audio: laczenie wideo z audio, miks sciezek, metadane, dlugosc.

Calosc dziala w procesie na bibliotekach FFmpeg z pakietu PyAV (engine/media) - bez programu
ffmpeg w PATH. Kazda funkcja zwraca (ok, komunikat), a bledy nie wywracaja pipeline'u - montaz
audio jest etapem opcjonalnym.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from engine.media import audio as _audio
from engine.media import video as _video
from engine.media.errors import MediaError

AI_COMMENT = _video.AI_COMMENT
FPS_TOLERANCE = 0.001
# A Timeline scene's background starts this long before the scene (the export's cross-fade handle).
TIMELINE_AUDIO_HANDLE = _audio.TIMELINE_AUDIO_HANDLE


def _attempt(action, result: str | Path) -> Tuple[bool, str]:
    """(True, result) when `action` works, (False, powod) when the media layer refuses."""
    try:
        action()
    except MediaError as error:
        return False, str(error)
    except Exception as error:  # noqa: BLE001  - montaz nie moze wywrocic generowania
        return False, f"{type(error).__name__}: {error}"
    return True, str(result)


def probe_duration(path: str | Path) -> Optional[float]:
    """Czas trwania pliku w sekundach lub None."""
    try:
        return _video.probe(path).duration or None
    except MediaError:
        return None


def video_fps(path: str | Path) -> Optional[float]:
    """Rzeczywisty sredni FPS pierwszej sciezki wideo."""
    try:
        return _video.probe(path).fps or None
    except MediaError:
        return None


def convert_fps(video: str | Path, out_path: str | Path, fps: int,
                duration: Optional[float] = None) -> Tuple[bool, str]:
    """Przelicza klatki wg czasu, zachowujac dlugosc z dokladnoscia do 1 klatki.

    Wejscie jest juz interpolowane RIFE. Wybor klatek idzie po znacznikach czasu; nie zmieniamy
    szybkosci odtwarzania przez nadpisanie frame_rate. RIFE zapisuje ostatni punkt czasowy, bez jego
    pelnego interwalu: dopelnienie konca zachowuje czas oryginalnego klipu i synchronizacje audio.
    Plik posredni jest kodowany bezstratnie, przed docelowa kompresja.
    """
    duration = duration if duration is not None else probe_duration(video)
    if not duration or duration <= 0 or fps <= 0:
        return False, "Could not determine video duration for FPS conversion."
    ok, msg = _attempt(lambda: _video.convert_fps(video, out_path, fps, duration), out_path)
    if not ok:
        return False, msg
    frames = max(1, int(duration * fps + 0.5))
    actual_fps = video_fps(out_path)
    actual_duration = probe_duration(out_path)
    if actual_fps is None or abs(actual_fps - fps) > FPS_TOLERANCE:
        return False, f"The converted video does not have {fps} FPS."
    if actual_duration is None or abs(actual_duration - frames / fps) > 0.01:
        return False, "FPS conversion changed the clip duration."
    return True, str(out_path)


def last_frame(video: str | Path, out_png: str | Path) -> Tuple[bool, str]:
    """Zapisuje ostatnia klatke wideo jako PNG (pierwsza klatka kolejnego ujecia)."""
    ok, msg = _attempt(lambda: _video.last_frame(video, out_png), out_png)
    return ok and Path(out_png).exists(), msg


def join_segments(parts: List[str], out_path: str | Path, fps: int,
                  overlaps: Optional[List[float]] = None,
                  duration: Optional[float] = None,
                  handoffs: Optional[List[float]] = None) -> Tuple[bool, str]:
    """Skleja ujecia w jedno wideo o podanym FPS (H.264, CRF 17).

    Kazde kolejne ujecie zaczyna sie od ostatniej klatki poprzedniego - ta powtorzona
    klatka jest pomijana, wiec ruch przechodzi plynnie. overlaps[i] > 0: czesc i zaczyna
    sie tyle sekund przed koncem poprzedniej (poprawiona twarz postaci) - ten odcinek
    przenika sie, zamiast skoku twarzy w jednej klatce. handoffs[i]: chwila (na zegarze czesci i)
    klatki, od ktorej startuje czesc i + 1 - potrzebna, gdy jest zakladka.
    """
    return _attempt(lambda: _video.join_segments(parts, out_path, fps, overlaps, duration, handoffs=handoffs),
                    out_path)


def overlay_layers(base: str | Path, layers: List[Tuple[str, int, int, int, int]],
                   out_path: str | Path) -> Tuple[bool, str]:
    """Wkleja wycinki (sciezka, x, y, szer, wys) na wideo bazowe z miekkimi krawedziami.

    Uzywane przy ruchu ust wielu postaci: kazda twarz przetworzona osobno wraca w swoje
    miejsce, a rozmyta maska ukrywa szew miedzy wycinkiem a reszta kadru.
    """
    return _attempt(lambda: _video.overlay_layers(base, layers, out_path), out_path)



def mix_voice_and_background(voice: Optional[str], background: Optional[str],
                             out_path: str | Path, duration: float,
                             background_offset: float = 0.0) -> Tuple[bool, str]:
    """Miks narracji z tlem dzwiekowym do WAV 48 kHz stereo o dlugosci `duration` s.

    Tlo jest ciszej, lagodnie wchodzi i schodzi, i samo przycisza sie, gdy mowi lektor
    (sidechain). Glosnosc ustawiamy w dwoch przebiegach (pomiar, potem stale wzmocnienie),
    bo jednoprzebiegowy loudnorm zaczyna za glosno: -16 LUFS z lektorem (standard
    serwisow internetowych), takze dla samego tla - przy -20 LUFS efekty byly za ciche.
    background_offset: seconds of the background before the picture starts (a Timeline
    background starts TIMELINE_AUDIO_HANDLE early); they are skipped so the sound lines up.
    """
    if not voice and not background:
        return False, "No audio tracks are available."

    def work() -> None:
        bed = None
        if background:
            bed = _audio.read(background)[0]
            if background_offset > 0:
                bed = bed[:, int(round(background_offset * _audio.RATE)):]
        mixed = _audio.voice_over_background(_audio.read(voice)[0] if voice else None, bed, duration)
        _audio.write_wav(out_path, mixed, _audio.RATE)

    return _attempt(work, out_path)


def combine_video_audio(video: str | Path, audio: Optional[str | Path],
                        out_path: str | Path,
                        tag_ai: bool = True, compress: bool = False) -> Tuple[bool, str]:
    """Laczy wideo z audio (lub samo wideo) i dodaje metadane 'AI-generated'.

    Gdy audio (np. narracja) jest dluzsze od wideo, ostatnia klatka jest przedluzana,
    zeby wypowiedz nie zostala ucieta. W przeciwnym razie wideo jest kopiowane bez
    rekompresji w calej dlugosci, a krotsze audio dopelniane cisza.
    compress - obraz zawsze kodowany od nowa: bezstratny plik posredni (convert_fps) ma
    profil H.264 High 4:4:4, ktorego nie odtworzy np. telefon ani odtwarzacz Windows.
    """
    return _attempt(lambda: _video.combine_video_audio(video, audio, out_path, tag_ai, compress), out_path)


def replace_audio(video: str | Path, audio: str | Path, out_path: str | Path) -> Tuple[bool, str]:
    """A copy of a video whose sound is `audio` (fitted to the picture's length) instead of its own;
    the picture is copied without re-encoding and tagged 'AI-generated'."""
    def work() -> None:
        length = _video.probe(video).duration
        samples, rate = _audio.read(audio, rate=None, channels=None)
        if length:
            samples = _audio.fit(samples, rate, round(length, 3))
        _video.remux(video, out_path, comment=AI_COMMENT, audio=(samples, rate), keep_audio=False)

    return _attempt(work, out_path)


def tag_video_metadata(path: str | Path) -> Tuple[bool, str]:
    """Dodaje metadane 'AI-generated' do istniejacego pliku wideo (remux)."""
    return _attempt(lambda: _video.tag_metadata(path), path)


def transform(src: str | Path, out_path: str | Path, chain: str, crf: int = 14,
              preset: str = "medium") -> Tuple[bool, str]:
    """Wideo bez dzwieku przez lancuch filtrow (np. "fps=25", "crop=w:h:x:y,scale=...") do H.264
    yuv420p o podanym CRF - dla krokow posrednich (lip sync)."""
    return _attempt(lambda: _video.reencode(src, out_path, chain=chain, crf=crf, preset=preset), out_path)


def splice_frames(full: str | Path, patch: str | Path, first: int, stop: int,
                  out_path: str | Path) -> Tuple[bool, str]:
    """Wkleja klatki z `patch` w miejsce klatek [first, stop) wideo `full` (ten sam FPS), H.264 CRF 14."""
    return _attempt(lambda: _video.splice(full, patch, first, stop, out_path), out_path)


def frame_at(video: str | Path, seconds: float, out_png: str | Path) -> Tuple[bool, str]:
    """Zapisuje pierwsza klatke od chwili `seconds` jako PNG (kontrola jakosci wideo)."""
    ok, msg = _attempt(lambda: _video.frame_at(video, seconds, out_png), out_png)
    return ok and Path(out_png).exists(), msg


def burn_subtitles(src: str | Path, cues: List[Tuple[float, float, str]],
                   out_path: str | Path) -> Tuple[bool, str]:
    """A copy of the film with the cues (start s, end s, text) burned in: white text with a dark outline at
    the bottom, sized by the frame height and wrapped to the frame and to a reading length of about 42
    characters (subtitles.READABLE). The sound is copied as it is and the file is tagged 'AI-generated'."""
    from engine.media import subtitles

    return _attempt(lambda: subtitles.burn(src, cues, out_path, comment=AI_COMMENT, style=subtitles.READABLE),
                    out_path)


def burn_overlays(src: str | Path, layers: List[tuple], out_path: str | Path,
                  report: Optional[list] = None) -> Tuple[bool, str]:
    """A copy of the film with exact text burned in (app/overlays.py): `layers` are (start s, end s, text, position, size,
    backdrop). The sound is copied as it is and the file is tagged 'AI-generated'; `report` gets a sentence for a text
    that had to be made smaller to fit."""
    from engine.media import overlays

    return _attempt(lambda: overlays.burn(src, [overlays.Layer(*row) for row in layers], out_path, comment=AI_COMMENT,
                                          report=report), out_path)


def prepare_voice_sample(src: str | Path, out_path: str | Path, scan_seconds: float,
                         max_seconds: float) -> Tuple[bool, str]:
    """Dowolne audio -> WAV 48 kHz mono 16 bit: cisza obcieta z obu stron, najwyzej `max_seconds`,
    glosnosc wyrownana (-18 LUFS). Czytane jest tylko pierwsze `scan_seconds` dlugiego nagrania."""
    return _attempt(lambda: _audio.prepare_voice_sample(src, out_path, scan_seconds, max_seconds), out_path)
