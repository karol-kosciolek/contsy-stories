# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Local timeline projects, scene preparation, scene sound previews and the final MP4 export.

A scene keeps its picture text, dialogue lines, background sound, timed effects, reference images
and mix levels (Scene). A prepared clip is cached under everything it is made from except the mix
levels (Scene.clip_settings), so a level change re-mixes the previews, the joined edit and the export
but never generates a clip. A scene whose picture inputs still match an existing clip only gets new
background sound around that picture (the sound-only refresh), and continuity follows the picture
(picture_sha256), so the next scene keeps its clip too. A scene has one sound: its sound preview,
made with the scene's own seed in <project>/sound/<key>/, is also what its clips are made with.
'Generate missing clips' makes every clip that is missing or no longer fits the timeline.

The editor refreshes every few seconds and asks scene_issues, sound_preview, preview_mix,
leveled_clip and takes each time: these are memoized by their inputs and the stamps of the project
folders, and previews are cached files that are mixed once per level setting.

A project can burn its dialogue into the exported movie as subtitles (Project.subtitles, off by default and
set with set_subtitles): each line is shown at the exact time it is spoken. The option is saved only when it
is on and is not part of Project.key, so an older project opens with it off and with the same key, and its
clips and plans stay valid. An export without subtitles is made exactly as before.

A project can also keep exact text for the export (Project.overlays, app/overlays.py: an end card, a tagline, a web
address), because a video model cannot spell: the text is burned into the finished movie after the subtitles, in the
real letters, at the times and place the overlay says. Overlays are saved only when there are any and are not part of
Project.key (no clip depends on them); an export records which it was made with (export_overlays), and one made with
other overlays is no longer the current export, as with the subtitles.

A project has a visual style (Project.style, app/styles.py: Realistic, 3D animation, 2D cartoon, Anime, or one of the
owner's own styles, whose id holds its whole definition). Every picture of the film is made in it: the Wan prompts,
the start frames and the compositions with characters. The style is saved only when it is not Realistic and is part
of Project.key only then, so an older project opens as Realistic with the same key and its clips stay Ready. A clip records the style it was made in (its Wan suffix,
"style"); after a style change the clips carry over but show Outdated until they are made again in the new style. An
animal or creature character is drawn only in animated styles: in a film whose style is Realistic (or based on it) its
scenes have an error issue, so no job is queued for them.

A saved project can be deleted (delete): its folder, with the plans, clips, takes, joined edits and sound previews,
goes to the Recycle Bin, where it can be restored. Exported movies are not in that folder: they stay in the outputs
folder and the Gallery.

Places and items (app/items.py) come into a scene in two ways. Picked in the scene inspector, an item's main photo is
one of the scene's reference images (store_reference, like an uploaded image; item_reference names it). Named in the
picture text, an item that is not among them is used as one too when the scene is made (named_items), the way a
character's name brings the character in: its main photo follows the scene's own images, and its identity
(items.identity) is part of the scene's clip keys, so a clip made before the item existed or before its photo changed
shows Outdated. A scene that names no item keeps the keys it had. A film keeps a list of its places and items too
(Project.items, the bin's Items tab): the editor lists them first and tells the prompt assistant their names; it is not
part of Project.key, so no clip depends on it.

A scene that has a clip can be edited with AI (edit, edit_job): its new text and dialogue are saved, and its new take is
made from the take it showed instead of from nothing. A take whose picture has no mouth movement only gets the new
voice; a Realistic talking picture keeps its frames and MuseTalk moves the mouth to the new lines; otherwise the scene
starts again from its take's first frame (an anchored scene from the previous scene's last frame, as always),
changed first by Qwen-Image-Edit when the edit names a picture change. An edit can carry on through the film (scope):
the scenes after it that continue from it are made again from its new last frame, and the scenes that start a shot
get the same picture change. Every edit is a new take; the take it was made from stays, and the scene's takes list it
as another version (versions.json remembers the text of each), which Use this take brings back with that text.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import functools
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import uuid
import warnings

from PIL import Image, ImageOps

from engine.media import video as media
from engine.media.errors import MediaError

from . import cancel, cast, characters, config, dialogue, ffmpeg_utils, history, items, llm, voices
from . import engine_runtime, model_manager, prefs, queue_manager, refs, scene_parts, scene_sound, sound
from . import overlays as text_overlays
from . import music_library, qc_report, shot_pipeline, shot_qc, studio, style_library, styles, trash, tts, universes, video_gen
from .persistence import atomic_text

DIR = config.STATE_ROOT / "timeline_projects"
PREPARE, MERGE, RENDER = "timeline_prepare", "timeline_merge", "timeline_render"
SOUND, PRODUCE, EDIT = "timeline_sound", "timeline_produce", "timeline_edit"
JOB_KINDS = (PREPARE, MERGE, RENDER, SOUND, PRODUCE, EDIT)
REVIEW_UNAVAILABLE = "visual_review_unavailable"
MAX_REFERENCE_BYTES = 50 * 1024 * 1024
MAX_REFERENCE_PIXELS = 40_000_000
REFERENCE_SIDE = 2048  # longest side of a stored reference image; the composer works far below it
REFERENCE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF")  # JPEG also opens multi-picture (MPO) files
_LOCK = threading.RLock()
_ID = re.compile(r"[0-9a-f]{32}")
_HASH = re.compile(r"[0-9a-f]{64}")
_PREVIEW = re.compile(r"([0-9a-f]{16})-[0-9a-f]{24}\.(wav|mp4)")
_BRIDGE = re.compile(r"[0-9a-f]{64}\.mp4")
_CONTINUITY_VERSION = 1
_SOUND_CACHE_VERSION = "separate-sound-v1"
_PICTURE_VERSION = 1
# Video has one mode since 2026-11-14 (app/video_gen.py). Projects saved before hashed their mode into
# their keys, and the mode that remains was called "preview": keeping that name in the keys keeps those
# projects' keys and clips. A project saved in the removed Fast or Quality mode opens in this one; its
# clips (1080p) no longer match its keys, so they count as outdated and are made again at 480p.
_MODE_IN_KEYS = "preview"
_JOIN_VERSION = 1
_PREVIEW_VERSION = 2  # 2: effects laid over the leveled room tone, the scene preview is the export mix at 0/0 too
_BRIDGE_FRAMES = 12
_SOUND_FIELDS = ("background", "effects")  # sound inputs that leave the picture as it is
_KEEP_PREVIEWS = 4    # cached previews per clip or audition and file type
_KEEP_JOINS = 400     # remembered join verdicts per project
_UI_TTL = 10.0        # s: library changes (voices, characters, models) reach the editor's checks this late
_WRITE_TRIES = 5      # an antivirus or indexer may hold a file for a moment: a rename is tried again
# Delete project: the shell moves a project folder whole or not at all, and an open file stops it. The editor's own
# background readers (filmstrips, waveforms, clip checks) hold the clips for a few seconds after a project opens.
_DELETE_TRIES = 4
_DELETE_WAIT = 0.5    # s between tries; the shell's move itself waits about 1 s before it gives up
# Later scenes whose lines one shared speech batch synthesizes ahead (_SpeechAhead). Each batch saves a VoxCPM2
# load per scene it serves; the lines it made ahead are lost when the job fails or is paused before their scenes,
# so a small window keeps most of the saving and bounds what a failure costs again.
SPEECH_AHEAD_SCENES = 2
# Other scenes still to make whose Wan prompts a job for single scenes encodes in its text-encoder load
# (_prompts_ahead_scenes); with its own prompt they fit the engine's kept encodings (core.memory.EmbeddingCache, 8).
PROMPTS_AHEAD_OTHER_SCENES = 6
# A job that makes several scenes makes their sounds first, this many scenes at a time (_sounds_ahead): each sound model
# loads once per window. The sounds are kept in the scenes' sound folders, so a failed or paused job loses none of them;
# the window only bounds how long the first picture waits.
SOUNDS_AHEAD_SCENES = 8
MAX_FILM_ITEMS = 50  # places and items in a film's list (Project.items)
PLAN_NOT_SAVED = "Could not save the timeline plan. The clip is kept; run the action again."
DELETED = ("This project is not saved any more: it was deleted, perhaps in another tab. Restore it from the Recycle "
           "Bin and open it again, or start a new project.")
NEARLY_SILENT = "Background is nearly silent. Describe the background sound or prepare it again."
# An animal or creature (characters.CREATURE) in a scene of a film whose style is Realistic or based on it
# (_style_issues): an error, so no GPU time is spent on the scene.
CREATURE_NEEDS_ANIMATION = ("{name} is an animal or creature: choose an animated style for this film "
                            "(Project → Style), or take {name} out of this scene.")
# A failed lip sync means the generated picture had no usable speaking face: the same seed would make
# the same picture again, so the creator needs a new take or a description with the face visible.
_FACELESS = ("no face could be lip-synced", "no face in the frame")
_NOT_FRAMING = ("out of memory", "cuda", "memory", "checkpoint", "model")  # a technical cause, not the framing
NO_FACE_HINT = ("If the speaker's face was turned away, hidden or too small, retry the scene for a new take, "
                "or describe the speaker facing the camera.")
_JOIN_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["match", "bridge", "mismatch"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Scene:
    prompt: str = ""                         # Picture and motion. Legacy scenes may contain “quoted lines”.
    seconds: int | float = 5                # whole seconds, or tenths of one (scene_parts.length)
    cut: bool = False
    cast: tuple[str, ...] | None = None      # None follows the description; () is explicitly empty.
    speakers: tuple[str, ...] = ()           # Legacy: a speaker per quoted line (only without lines).
    lines: tuple[scene_parts.Cue, ...] = ()
    background: str = ""                     # Background sound; "" derives it from the picture text.
    effects: tuple[scene_parts.Effect, ...] = ()
    references: tuple[str, ...] = ()         # Stored reference images (store_reference), [1], [2]… in the text.
    voice_db: float = 0.0                    # Dialogue level in previews and the export.
    background_db: float = 0.0               # Background level in previews and the export.
    # The shot contract (app/shot_spec.py) the prompt was compiled from, as canonical JSON; "" for a scene written as
    # free text. It decides the keyframe's place and props and what the shot check (app/shot_qc.py) looks for.
    shot: str = ""

    def settings(self) -> dict:
        """Everything the scene says, as saved. Fields at their defaults are left out, so a scene
        without the newer fields keeps the identity it always had."""
        result = self.clip_settings()
        if self.voice_db:
            result["voice_db"] = self.voice_db
        if self.background_db:
            result["background_db"] = self.background_db
        return result

    def clip_settings(self) -> dict:
        """settings() without the mix levels: what a prepared clip is made from."""
        result = {"prompt": self.prompt, "seconds": self.seconds, "cut": self.cut}
        if self.cast is not None:
            result["cast"] = self.cast
        if self.speakers:
            result["speakers"] = self.speakers
        if self.lines:
            result["lines"] = [cue.to_json() for cue in self.lines]
        if self.background:
            result["background"] = self.background
        if self.effects:
            result["effects"] = [effect.to_json() for effect in self.effects]
        if self.references:
            result["references"] = self.references
        if self.shot:
            result["shot"] = json.loads(self.shot)
        return result

    def footage(self) -> tuple:
        """The inputs of the scene's clip; a clip carries over while they stay the same."""
        return (self.prompt, self.seconds, self.cast, self.speakers, self.lines, self.background, self.effects,
                self.references) + ((self.shot,) if self.shot else ())

    def picture_footage(self) -> tuple:
        """The inputs that need a new picture when they change."""
        return (self.prompt, self.seconds, self.cast, self.speakers, self.lines,
                self.references) + ((self.shot,) if self.shot else ())

    @property
    def spec(self):
        """The scene's ShotSpec, or None for a free-text scene."""
        if not self.shot:
            return None
        from .shot_spec import ShotSpec

        return ShotSpec.from_json(self.shot)

    @classmethod
    def from_json(cls, fields: dict) -> Scene:
        if not isinstance(fields, dict) or not set(fields) <= set(cls.__dataclass_fields__):
            raise ValueError("Invalid scene.")
        members, speakers = fields.get("cast"), fields.get("speakers", [])
        lines, effects, references = fields.get("lines", []), fields.get("effects", []), fields.get("references", [])
        if ((members is not None and not isinstance(members, list)) or not isinstance(speakers, list)
                or any(not isinstance(slug, str) for slug in [*(members or []), *speakers])):
            raise ValueError("Invalid scene cast or speakers.")
        if (not all(isinstance(rows, list) for rows in (lines, effects, references))
                or any(not isinstance(name, str) for name in references)):
            raise ValueError("Invalid scene dialogue, sound or references.")
        shot = fields.get("shot", "")
        if shot:
            from .shot_spec import ShotError, ShotSpec

            try:
                shot = ShotSpec.from_json(shot).dumps()
            except (ShotError, ValueError, TypeError) as error:
                raise ValueError(f"Invalid scene shot: {error}") from error
        return cls(**{**fields, "cast": tuple(members) if members is not None else None, "speakers": tuple(speakers),
                      "lines": tuple(scene_parts.Cue.from_json(row) for row in lines),
                      "effects": tuple(scene_parts.Effect.from_json(row) for row in effects),
                      "references": tuple(references), "shot": shot or ""})


@dataclass(frozen=True)
class Project:
    id: str
    revision: int
    title: str
    scenes: tuple[Scene, ...]
    aspect: str = "Landscape 16:9"
    audio: bool = True
    characters: tuple[str, ...] = ()
    updated: float = 0.0
    reference_characters: tuple[str, ...] = ()
    universe: str = ""
    universe_members: tuple[str, ...] = ()
    subtitles: bool = False  # burn the dialogue into the export; not in `key` (no clip depends on it)
    style: str = styles.DEFAULT  # the film's visual style (app/styles.py); in `key` unless Realistic
    items: tuple[str, ...] = ()  # the film's places and items (app/items.py slugs, the bin's Items tab); not in `key`
    overlays: tuple[text_overlays.Overlay, ...] = ()  # exact text burned into the export (app/overlays.py); not in `key`

    @property
    def look(self) -> styles.Style:
        """The film's style (Realistic for a damaged value, which _validated refuses on save)."""
        return styles.of(self.style)

    @property
    def video_style(self) -> str:
        """The Wan suffix of the film's style: what every clip is made and keyed with (its record's "style")."""
        return self.look.video

    @property
    def seconds(self) -> int | float:
        return round(sum(scene.seconds for scene in self.scenes), 1)  # scenes last tenths of a second too

    @property
    def cast_slugs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.universe_members, *self.reference_characters, *self.characters)))

    @property
    def key(self) -> str:
        settings = [self.aspect, _MODE_IN_KEYS, self.audio, self.characters,
                    [scene.settings() for scene in self.scenes]]
        if self.reference_characters:
            settings.append(self.reference_characters)
        if self.universe_members:
            settings.append({"universe_cast": self.universe_members})
        if self.style != styles.DEFAULT:  # a Realistic project keeps the key it had before styles
            settings.append({"style": self.style})
        return _hash(settings)


@dataclass(frozen=True)
class MoviePreview:
    path: Path
    key: str
    revision: int | None
    current: bool = True


class _SceneError(ValueError):
    """A problem with one scene; `scene` is its index."""

    def __init__(self, scene: int, message: str):
        super().__init__(message)
        self.scene = scene


class _Refused(ValueError):
    """The render cannot start or finish with the current plan."""


class Deleted(ValueError):
    """The project's folder is gone: the project was deleted, perhaps in another tab."""

    def __init__(self):
        super().__init__(DELETED)


class Busy(ValueError):
    """The project has a job in the GPU queue, so it is not changed now (save_idle)."""


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _root(project_id: str) -> Path:
    if not isinstance(project_id, str) or not _ID.fullmatch(project_id):
        raise ValueError("Invalid timeline project.")
    root = DIR.resolve() / project_id
    if root.resolve() != root:
        raise ValueError("The project folder has an unexpected location.")
    return root


def _saved_root(project: Project) -> Path | None:
    """The project's folder, or None for a project that was never saved."""
    try:
        return _root(project.id)
    except ValueError:
        return None


def _inside(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts)
    if path.resolve() != path:
        raise ValueError("The timeline file has an unexpected location.")
    return path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _patiently(action):
    """action() tried again a few times after a PermissionError: on Windows a reader (the editor, an
    antivirus or the search indexer) can hold a file for a moment while it is renamed into place."""
    for attempt in range(_WRITE_TRIES):
        try:
            return action()
        except PermissionError:
            if attempt == _WRITE_TRIES - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _write(path: Path, value) -> None:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2)
        _patiently(lambda: atomic_text(path, text))
    finally:
        _changed()


def _write_bytes(path: Path, data: bytes) -> None:
    """Atomic binary write: a temporary sibling file renamed into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _patiently(lambda: os.replace(name, path))
    finally:
        if name and Path(name).exists():
            Path(name).unlink()
        _changed()


def _valid_index(project: Project, index) -> bool:
    return type(index) is int and 0 <= index < len(project.scenes)


# --- memoized answers for the editor's refresh ---------------------------------------------------------

_MEMO_LOCK = threading.Lock()
_MEMO: dict = {}
_MEMO_LIMIT = 512
_CLIP_INDEX: dict[str, tuple] = {}
_changes = 0  # bumped by every write of this module; folder times alone can miss writes in one clock tick


def _changed() -> None:
    global _changes
    with _MEMO_LOCK:
        _changes += 1


def _memo(kind: str, key, compute, ttl: float = _UI_TTL):
    """compute() remembered for `ttl` seconds under (kind, key)."""
    try:
        hash(key)
    except TypeError:
        return compute()
    now = time.monotonic()
    with _MEMO_LOCK:
        found = _MEMO.get((kind, key))
        if found is not None and found[0] > now:
            return found[1]
    value = compute()
    with _MEMO_LOCK:
        if len(_MEMO) >= _MEMO_LIMIT:
            stale = [entry for entry, (expires, _) in _MEMO.items() if expires <= now]
            for entry in stale or list(_MEMO)[:_MEMO_LIMIT // 2]:
                _MEMO.pop(entry, None)
        _MEMO[(kind, key)] = (now + ttl, value)
    return value


def _folder_stamp(project: Project) -> tuple:
    """Modification times of the saved project's file and folders: they change whenever a plan,
    clip or sound preview is written (every write renames a file into place)."""
    root = _saved_root(project)
    if root is None:
        return ()
    stamps = [_changes]
    for name in ("project.json", "plans", "clips", "sound"):
        try:
            stamps.append((root / name).stat().st_mtime_ns)
        except OSError:
            stamps.append(None)
    return tuple(stamps)


def _cast_fields(project: Project) -> tuple:
    """What decides a scene's cast besides the scene: the style too, as a Realistic film refuses an animal or creature
    (_scene_cast)."""
    return project.characters, project.reference_characters, project.universe, project.universe_members, project.style


def _members_cached(project: Project, scene: Scene) -> list[dict]:
    """_scene_cast for the editor: remembered for a few seconds (it reads the character library)."""
    def compute():
        try:
            return _scene_cast(project, scene), None
        except ValueError as error:
            return None, str(error)

    members, error = _memo("members", (_cast_fields(project), scene), compute)
    if error is not None:
        raise ValueError(error)
    return members


def _installed(part: str) -> bool:
    check = tts.available if part == "voice" else sound.available
    return _memo("installed", part, check, ttl=3 * _UI_TTL)


def _scene_keys(project: Project) -> list:
    """The clip key of every scene: the saved plan, carried over to an edited draft of the project."""
    if _saved_root(project) is None:
        return []

    def compute():
        try:
            saved = load(project.id)
        except ValueError:
            return ()
        keys = prepared(saved)["assets"]
        return tuple(keys if saved.key == project.key else carry_clips(project, saved, keys))

    return list(_memo("keys", (project, _folder_stamp(project)), compute))


def _scene_clip(project: Project, index: int) -> dict | None:
    keys = _scene_keys(project)
    if index < len(keys) and keys[index]:
        return asset(project, keys[index], verify=False)
    return None


def _clip_records(project: Project) -> dict[str, dict]:
    """Every readable clip record of the project by key; read again only when the clips folder changes."""
    root = _saved_root(project)
    if root is None:
        return {}
    folder = _inside(root, "clips")
    try:
        stamp = _changes, folder.stat().st_mtime_ns
    except OSError:
        return {}
    with _MEMO_LOCK:
        found = _CLIP_INDEX.get(project.id)
    if found is not None and found[0] == stamp:
        return found[1]
    records = {}
    for path in folder.glob("*.json"):
        if _HASH.fullmatch(path.stem) and (record := asset(project, path.stem, verify=False)):
            records[path.stem] = record
    with _MEMO_LOCK:
        _CLIP_INDEX[project.id] = (stamp, records)
    return records


def _created(record: dict) -> float:
    value = record.get("created")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    try:
        return record["path"].stat().st_mtime
    except (OSError, KeyError, AttributeError):
        return 0.0


# --- project files ---------------------------------------------------------------------------------------

def _valid_scene(project: Project, number: int, scene: Scene, strict: bool) -> Scene:
    if (not scene_parts.is_length(scene.seconds) or not 1 <= scene.seconds <= video_gen.MAX_SECONDS
            or not isinstance(scene.prompt, str) or type(scene.cut) is not bool):
        raise ValueError(f"Scene {number} must have a description and a duration of 1–{video_gen.MAX_SECONDS} s "
                         "(in tenths of a second).")
    if type(scene.seconds) is not type(scene_parts.length(scene.seconds)):  # 5.0 is stored as 5: the same keys
        scene = replace(scene, seconds=scene_parts.length(scene.seconds))
    if scene.cast is not None:
        if len(scene.cast) != len(set(scene.cast)) or len(scene.cast) > cast.MAX_CAST:
            raise ValueError(f"Scene {number}: select at most {cast.MAX_CAST} different characters.")
        if any(slug not in project.cast_slugs for slug in scene.cast):
            raise ValueError(f"Scene {number}: select characters from the project’s cast or restore them there.")
    if (not isinstance(scene.lines, tuple) or not all(isinstance(cue, scene_parts.Cue) for cue in scene.lines)
            or not isinstance(scene.effects, tuple)
            or not all(isinstance(effect, scene_parts.Effect) for effect in scene.effects)
            or not isinstance(scene.references, tuple) or not isinstance(scene.background, str)):
        raise ValueError(f"Scene {number}: the dialogue, sound or reference settings are invalid.")
    if scene.lines and scene.speakers:
        raise ValueError(f"Scene {number}: choose who speaks in the Dialogue lines.")
    if scene.speakers and (scene.cast is None or len(scene.speakers) > len(dialogue.quotes(scene.prompt))
                           or any(slug and slug not in scene.cast for slug in scene.speakers)):
        raise ValueError(f"Scene {number}: assign dialogue to a character appearing in the scene.")
    if scene.lines and dialogue.quotes(scene.prompt):
        raise ValueError(f"Scene {number}: move the quoted lines into Dialogue or remove the quotes.")
    try:
        scene_parts.check(scene.lines, scene.effects, scene.seconds)
        levels = scene_parts.level(scene.voice_db), scene_parts.level(scene.background_db)
    except ValueError as error:
        raise ValueError(f"Scene {number}: {error}") from error
    allowed = scene.cast if scene.cast is not None else project.cast_slugs
    for line, cue in enumerate(scene.lines, 1):
        if cue.character and cue.character not in allowed:
            where = "this scene" if scene.cast is not None else "the project’s cast"
            raise ValueError(f"Scene {number}: line {line} is spoken by a character who is not in {where}.")
    if len(scene.background) > scene_parts.MAX_TEXT:
        raise ValueError(f"Scene {number}: keep the background sound under {scene_parts.MAX_TEXT} characters.")
    if (len(scene.references) > scene_parts.MAX_REFERENCES or len(set(scene.references)) != len(scene.references)
            or not all(scene_parts.is_reference(name) for name in scene.references)):
        raise ValueError(f"Scene {number}: attach at most {scene_parts.MAX_REFERENCES} different reference images.")
    if strict:
        wrong = [mark for mark in refs.numbers(scene.prompt) if not 1 <= mark <= len(scene.references)]
        if wrong:
            raise ValueError(f"Scene {number}: [{wrong[0]}] does not match an attached reference image.")
    if any(type(value) is not float or value != wanted
           for value, wanted in zip((scene.voice_db, scene.background_db), levels)):
        scene = replace(scene, voice_db=levels[0], background_db=levels[1])
    return scene


def _validated(project: Project, *, strict: bool = True) -> Project:
    """The project with every scene checked and its levels on the 0.5 dB grid. `strict` also checks
    that each [n] in a description has its reference image; an opened project may still lack one."""
    if not isinstance(project.title, str) or not project.title.strip():
        raise ValueError("Enter a project name.")
    if project.aspect not in video_gen.ASPECTS:
        raise ValueError("Choose a valid aspect ratio.")
    if not styles.valid(project.style):
        raise ValueError("Choose a valid style.")
    if (type(project.revision) is not int or type(project.audio) is not bool or type(project.subtitles) is not bool
            or not isinstance(project.updated, (int, float))
            or not isinstance(project.universe, str)
            or any(not isinstance(slug, str)
                   for slug in (*project.characters, *project.reference_characters, *project.universe_members))
            or not isinstance(project.items, tuple) or len(project.items) > MAX_FILM_ITEMS
            or any(not isinstance(slug, str) or not slug for slug in project.items)
            or len(set(project.items)) != len(project.items)):
        raise ValueError("Invalid project settings.")
    if not isinstance(project.overlays, tuple):
        raise ValueError("Invalid project settings.")
    overlays = text_overlays.cleaned(project.overlays)
    scenes = tuple(_valid_scene(project, number, scene, strict) for number, scene in enumerate(project.scenes, 1))
    if any(new is not old for new, old in zip(scenes, project.scenes)) or project.overlays:  # overlays: tidied copies
        return replace(project, scenes=scenes, overlays=overlays)
    return project


def load(project_id: str) -> Project:
    root = _root(project_id)
    try:
        fields = _read(_inside(root, "project.json"))
        fields.pop("speed", None)  # the video mode of a project saved before the one mode (_MODE_IN_KEYS)
        fields["scenes"] = tuple(Scene.from_json(scene) for scene in fields["scenes"])
        fields["characters"] = tuple(fields["characters"])
        fields["reference_characters"] = tuple(fields.get("reference_characters", ()))
        fields["universe_members"] = tuple(fields.get("universe_members", ()))
        film_items = fields.get("items", [])
        if not isinstance(film_items, list):
            raise ValueError("Invalid film items.")
        fields["items"] = tuple(film_items)
        fields["overlays"] = text_overlays.tolerant(fields.get("overlays", []))
        project = Project(**fields)
        if project.id != project_id:
            raise ValueError("The project identifier changed.")
        return _validated(project, strict=False)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Could not open this timeline project.") from error


def projects() -> list[Project]:
    found = []
    for path in DIR.glob("*/project.json"):
        try:
            found.append(load(path.parent.name))
        except ValueError:
            continue
    return sorted(found, key=lambda project: project.updated, reverse=True)


def delete(project_id: str) -> str:
    """Move a saved project's folder to the Recycle Bin and return its title. Refused while one of its jobs waits or
    runs in the GPU queue, because the job writes into that folder; Deleted when the folder is gone already. While a
    file in the folder is open (_DELETE_TRIES) the move is tried again, with the queue checked again each time."""
    root = _root(project_id)  # a wrong id is refused before anything is touched
    for attempt in range(_DELETE_TRIES):
        if attempt:
            time.sleep(_DELETE_WAIT)  # without the lock: saves and other projects' jobs go on meanwhile
        with _LOCK:  # no save or job of the project starts meanwhile: enqueue holds the lock too
            if not _inside(root, "project.json").is_file():
                raise Deleted()
            project = load(project_id)
            if has_active_job(project):
                raise ValueError("This project has a job in the GPU queue. Wait until it ends, or stop or remove it "
                                 "in Tools → GPU Queue, then delete the project.")
            if trash.move(root):
                _changed()
                return project.title
    raise ValueError("The project's files are still in use (the editor may still be loading it, or a player has a "
                     "clip open). Wait a few seconds and try again.")


def save(project_id: str | None, revision: int | None, title: str, scenes: tuple[Scene, ...],
         aspect: str, audio: bool, selected: list[str] | None, *,
         reference_characters: tuple[str, ...] | None = None, universe: str | None = None,
         universe_members: tuple[str, ...] | None = None, subtitles: bool | None = None,
         style: str | None = None, film_items: tuple[str, ...] | None = None,
         overlays: tuple[text_overlays.Overlay, ...] | None = None) -> Project:
    """Save a new revision. reference_characters, universe, universe_members, subtitles, style, film_items (the
    film's places and items, Project.items) and overlays (the exact text burned into the export, Project.overlays) None
    keep what the project had (a new project: none, off, Realistic, no overlays). A style change keeps the clips in the
    plan: they show Outdated (_outdated) until they are made again in the new style."""
    with _LOCK:
        if project_id and not _inside(_root(project_id), "project.json").is_file():
            raise Deleted()  # a tab that still had it open does not make it again
        old = load(project_id) if project_id else None
        if old and old.revision != revision:
            raise ValueError("This project was saved in another tab. Reopen it before saving again.")
        title = (title or "").strip()
        baseline = old or Project(uuid.uuid4().hex, 0, title, scenes)
        slug = baseline.universe if universe is None else universe
        if universe_members is None:
            universe_members = baseline.universe_members
            if slug != baseline.universe:
                universe_members = tuple((universes.get(slug) or {}).get("members", ()))
        project = _validated(Project(
            id=baseline.id, revision=baseline.revision + 1, title=title, scenes=scenes, aspect=aspect, audio=audio,
            characters=tuple(selected or ()), updated=time.time(), universe=slug,
            universe_members=tuple(universe_members),
            reference_characters=baseline.reference_characters if reference_characters is None else tuple(reference_characters),
            subtitles=baseline.subtitles if subtitles is None else subtitles,
            style=baseline.style if style is None else style,
            items=baseline.items if film_items is None else tuple(dict.fromkeys(film_items)),
            overlays=baseline.overlays if overlays is None else tuple(overlays),
        ))
        if project.style != baseline.style:  # a stored film keeps working; only a change is checked
            problem = llm.style_problem(project.style)
            if problem:
                raise ValueError(problem)
        if old and (project.key, project.title, project.universe, project.subtitles, project.items,
                    project.overlays) == (old.key, old.title, old.universe, old.subtitles, old.items, old.overlays):
            return old
        if old and project.key != old.key:
            state = prepared(project)
            previous = prepared(old)
            existing = state["assets"]
            keys = plan_for(project, old, previous["assets"])
            while keys and keys[-1] is None:
                keys.pop()
            # A scene whose clip was dropped only because its sound changed remembers the clip the
            # plan showed: its picture (the chosen take) is reused by the sound-only refresh.
            found = _sound_sources(project, old, previous["assets"], previous.get("sound_sources"))
            kept = state.get("sound_sources") if isinstance(state.get("sound_sources"), list) else []
            sources = [kept[index] if index < len(kept) and _valid_key(kept[index]) else key
                       for index, key in enumerate(found)]
            sources = [key if not (index < len(keys) and keys[index]) else None for index, key in enumerate(sources)]
            update = {}
            if keys != existing:
                update.update(assets=keys, export_assets=state.get("export_assets", existing))
            if any(sources) and sources != kept:
                update["sound_sources"] = sources
            if update:
                _write_plan(project, {**state, **update})
        fields = {**asdict(project), "scenes": [scene.settings() for scene in project.scenes],
                  "overlays": [overlay.to_json() for overlay in project.overlays]}
        if not project.overlays:
            fields.pop("overlays")  # a film without overlays keeps the file it always had
        if not project.subtitles:
            fields.pop("subtitles")  # the file of a project without subtitles keeps the fields it always had
        if project.style == styles.DEFAULT:
            fields.pop("style")  # likewise a Realistic project
        if not project.items:
            fields.pop("items")  # and a film without a list of places and items
        _write(_inside(_root(project.id), "project.json"), fields)
        return project


def save_idle(project_id: str | None, revision: int | None, title: str, scenes: tuple[Scene, ...],
              aspect: str, audio: bool, selected: list[str] | None, **options) -> Project:
    """save() for a caller that cannot see the editor (the Agent page): refused with Busy while a job of the project
    waits or runs, because that job reads the project's files. The check and the save happen under one lock, and
    enqueue() holds the same lock, so no job can start between them."""
    with _LOCK:
        if project_id and _inside(_root(project_id), "project.json").is_file() and has_active_job(load(project_id)):
            raise Busy("This project has a job in the GPU queue. Wait until it ends before changing the project.")
        return save(project_id, revision, title, scenes, aspect, audio, selected, **options)


def set_subtitles(project_id: str, revision: int | None, on: bool) -> Project:
    """Turn burned subtitles in the export on or off. The clips and the joined edit stay; only the export is
    made again (exported() no longer matches)."""
    if type(on) is not bool:
        raise ValueError("Choose whether to burn subtitles.")
    project = load(project_id)
    return save(project.id, project.revision if revision is None else revision, project.title, project.scenes,
                project.aspect, project.audio, list(project.characters),
                reference_characters=project.reference_characters, universe=project.universe,
                universe_members=project.universe_members, subtitles=on)


def set_overlays(project_id: str, revision: int | None, overlays) -> Project:
    """Replace the exact text burned into the export (Project.overlays). The clips and the joined edit stay; only the
    export is made again (exported() no longer matches)."""
    project = load(project_id)
    return save(project.id, project.revision if revision is None else revision, project.title, project.scenes,
                project.aspect, project.audio, list(project.characters),
                reference_characters=project.reference_characters, universe=project.universe,
                universe_members=project.universe_members, overlays=tuple(overlays))


def _prepared_path(project: Project) -> Path:
    return _inside(_root(project.id), "plans", project.key + ".json")


def prepared(project: Project) -> dict:
    try:
        state = _read(_prepared_path(project))
        if (state.get("continuity_version") == _CONTINUITY_VERSION
                and isinstance(state["assets"], list) and len(state["assets"]) <= len(project.scenes)
                and all(key is None or isinstance(key, str) and _HASH.fullmatch(key) for key in state["assets"])):
            return state
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return {"assets": []}


def _write_plan(project: Project, state: dict) -> None:
    _write(_prepared_path(project), {**state, "continuity_version": _CONTINUITY_VERSION})


def plan_for(project: Project, previous: Project, keys: list) -> list:
    """The clips `project`, an edit of `previous` (whose plan is `keys`), has once it is saved: the footage carried
    over (carry_clips), with the plan its own key already has over it, so going back to a style or setting the film
    had before finds the clips made then."""
    existing = prepared(project)["assets"]
    return [existing[index] if index < len(existing) and existing[index] else key
            for index, key in enumerate(carry_clips(project, previous, keys))]


def carry_clips(project: Project, previous: Project, keys: list) -> list:
    """Keep available footage when its scene inputs (not the mix levels) and generation settings match."""
    settings = ("aspect", "audio", "characters", "reference_characters", "universe_members")
    if any(getattr(project, name) != getattr(previous, name) for name in settings):
        return [None] * len(project.scenes)
    same_scene = {scene.footage(): key for scene, key in zip(previous.scenes, keys) if key}
    carried = []
    for index, scene in enumerate(project.scenes):
        key = same_scene.get(scene.footage())
        if index < len(keys) and keys[index]:
            old = previous.scenes[index]
            if scene.footage() == old.footage():
                key = keys[index]
        carried.append(key)
    return carried


def _valid_key(key) -> bool:
    return isinstance(key, str) and bool(_HASH.fullmatch(key))


def _sound_sources(project: Project, previous: Project, keys: list, sources=None) -> list:
    """For each scene of `project`: the clip `previous` showed for a scene with the same picture inputs
    (Scene.picture_footage), or the clip it remembered for one (`sources`, an older sound-only change),
    else None. Used for scenes whose clip did not carry over because only their sound changed."""
    settings = ("aspect", "audio", "characters", "reference_characters", "universe_members")
    if any(getattr(project, name) != getattr(previous, name) for name in settings):
        return [None] * len(project.scenes)
    sources = sources if isinstance(sources, list) else []
    shown = [keys[index] if index < len(keys) and _valid_key(keys[index])
             else sources[index] if index < len(sources) and _valid_key(sources[index]) else None
             for index in range(len(previous.scenes))]
    same_picture = {}
    for scene, key in zip(previous.scenes, shown):
        if key:
            same_picture.setdefault(scene.picture_footage(), key)
    result = []
    for index, scene in enumerate(project.scenes):
        key = None
        if index < len(shown) and shown[index] and previous.scenes[index].picture_footage() == scene.picture_footage():
            key = shown[index]
        result.append(key or same_picture.get(scene.picture_footage()))
    return result


def all_clips(project: Project, keys: list) -> bool:
    return bool(keys) and len(keys) == len(project.scenes) and all(keys)


def _transition(index: int, scene: Scene) -> str:
    return "first" if index == 0 else "cut" if scene.cut else "anchored"


def _outdated(project: Project, index: int, record: dict | None, previous: dict | None,
              members: list[dict] | None) -> bool:
    """True when `record` cannot stay the clip of scene `index`: it is missing or of another length,
    it was made for the other kind of transition (intentional cut or not), it continues from another
    picture than `previous` (the previous scene's plan clip) shows, or its picture inputs changed (its
    picture_key, with its own take, style and tier), or it was made in another style than the film's now (a
    record without one is Realistic). members: the scene's people, None when unknown."""
    scene = project.scenes[index]
    if not record or record["seconds"] != scene.seconds:
        return True
    if record.get("style", prefs.STYLE["video"]) != project.video_style:
        return True
    transition = record.get("transition")
    if transition in ("first", "cut", "anchored") and transition != _transition(index, scene):
        return True
    continuity = None
    if _transition(index, scene) == "anchored":
        if previous is None:
            return True
        anchor = record.get("anchor_sha256")
        if anchor and anchor != _picture_sha(previous):
            return True
        continuity = [_picture_sha(previous), project.scenes[index - 1].prompt]
    if members is not None and record.get("picture_key"):
        style = record.get("style", prefs.STYLE["video"])
        tier = record.get("tier") or _memo("tier", None, model_manager.tier, ttl=3 * _UI_TTL)
        expected = _picture_key(project, scene, members, style, tier, index, continuity,
                                record.get("cache_namespace"), record.get("take"))
        if expected != record["picture_key"]:
            return True
    return False


def _stale(project: Project, keys: list, *, casts: dict | None = None, verify: bool = False) -> list[int]:
    """Indices of the scenes whose clip must be made (again), in order: no usable clip, or one that no
    longer fits (_outdated). An anchored scene after a scene that is made again is counted too, as its
    starting picture will change (a sound-only refresh keeps it, so this is an upper bound; the job
    looks again after each scene). casts: the scenes' people by index (from _validate_generation);
    others are looked up and remembered for the editor."""
    result: list[int] = []
    previous = None
    for index, scene in enumerate(project.scenes):
        key = keys[index] if index < len(keys) else None
        record = asset(project, key, verify=verify) if key else None
        if casts is not None and index in casts:
            members = casts[index]
        else:
            try:
                members = _members_cached(project, scene)
            except ValueError:
                members = None
        redo = _outdated(project, index, record, previous, members)
        if not redo and result and result[-1] == index - 1 and _transition(index, scene) == "anchored":
            redo = True
        if redo:
            result.append(index)
        previous = record
    return result


def outdated(project: Project, keys: list | None = None) -> list[int]:
    """The scenes 'Generate missing clips' would make (again) for `project` with plan `keys` (default:
    its saved plan): see _stale. Remembered per inputs and project files, so the editor can ask on
    every refresh."""
    if _saved_root(project) is None:
        return list(range(len(project.scenes)))
    keys = prepared(project)["assets"] if keys is None else list(keys)
    return list(_memo("outdated", (project, tuple(keys), _folder_stamp(project)),
                      lambda: tuple(_stale(project, keys))))


def _asset_paths(project: Project, key: str) -> tuple[Path, Path]:
    if not _HASH.fullmatch(key):
        raise ValueError("Invalid timeline scene identifier.")
    root = _root(project.id)
    return _inside(root, "clips", key + ".mp4"), _inside(root, "clips", key + ".json")


def _sound_path(clip: Path, kind: str) -> Path:
    return _inside(clip.parent, f"{clip.stem}.{kind}.wav")


def asset(project: Project, key: str | None, verify: bool = True) -> dict | None:
    if not key:
        return None
    try:
        clip, metadata = _asset_paths(project, key)
        record = _read(metadata)
        if not (clip.is_file() and isinstance(record["qc"], dict)
                and all(field in record for field in ("seconds", "width", "height", "characters", "voices",
                                                       "transition_note", "sha256"))
                and (not verify or _sha(clip) == record["sha256"])):
            return None
        sources = effects = None
        if "sound_hashes" in record:
            hashes = record["sound_hashes"]
            background = _sound_path(clip, "background")
            voice = _sound_path(clip, "voice") if hashes.get("voice") else None
            effects = _sound_path(clip, "effects") if hashes.get("effects") else None
            for path, kind in ((background, "background"), (voice, "voice"), (effects, "effects")):
                if path is not None and (not path.is_file() or (verify and _sha(path) != hashes[kind])):
                    return None
            sources = background, voice
        # effects_source: the effects inside the background stem, alone (the export levels the room tone
        # without them); None for clips made before it was kept or without effects
        return {**record, "path": clip, "audio_sources": sources, "effects_source": effects}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return None


def _picture_sha(record: dict) -> str:
    """sha256 of the clip whose picture `record` shows (itself unless only its sound was made again)."""
    return record.get("picture_sha256", record["sha256"])


def exported(project: Project, state: dict | None = None) -> Path | None:
    state = prepared(project) if state is None else state
    if state.get("export_assets", state["assets"]) != state["assets"]:
        return None
    if bool(state.get("export_subtitles")) != project.subtitles:
        return None
    if (state.get("export_overlays") or "") != text_overlays.key(project.overlays):
        return None
    merge = state.get("merged")
    if isinstance(merge, dict) and merge.get("assets") == state["assets"] and state.get("export_merge") != merge.get("file"):
        return None
    return _export_file(state.get("export"))


def _export_file(name) -> Path | None:
    if isinstance(name, str) and Path(name).name == name and name.startswith("timeline_") and name.endswith(".mp4"):
        folder = config.OUTPUTS_DIR.resolve()
        file = _inside(folder, name)
        if file.is_file() and studio.is_recorded(str(file)):  # the Gallery lists the same files
            return file
    return None


def movie_preview(project: Project, state: dict) -> MoviePreview | None:
    """Prefer an export matching the edit, then the latest available export of this project."""
    if path := exported(project, state):
        return MoviePreview(path, project.key, state.get("export_revision"))
    if path := _export_file(state.get("export")):
        return MoviePreview(path, project.key, state.get("export_revision"), current=False)
    plans = _inside(_root(project.id), "plans")
    candidates = []
    for manifest in plans.glob("*.json"):
        if manifest.stem == project.key or not _HASH.fullmatch(manifest.stem):
            continue
        try:
            manifest = _inside(plans, manifest.name)
            record = _read(manifest)
            if isinstance(record, dict) and record.get("export"):
                candidates.append((manifest.stat().st_mtime_ns, manifest.stem, record))
        except (OSError, ValueError):
            continue
    for _, key, record in sorted(candidates, key=lambda item: item[0], reverse=True):
        if path := _export_file(record.get("export")):
            return MoviePreview(path, key, record.get("export_revision"), current=False)
    return None


# --- reference images ------------------------------------------------------------------------------------

def _refs_folder() -> Path:
    return DIR.resolve() / "refs"


def _rgb(image: Image.Image) -> Image.Image:
    """RGB pixels; transparent parts become white rather than whatever color they hide."""
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        layer = image.convert("RGBA")
        canvas = Image.new("RGB", layer.size, (255, 255, 255))
        canvas.paste(layer, mask=layer.getchannel("A"))
        return canvas
    return image.convert("RGB")


def store_reference(path) -> str:
    """Keep an image for scene references and return its name (the SHA-256 of the stored file).

    The file must be an image Pillow can read, at most MAX_REFERENCE_BYTES and MAX_REFERENCE_PIXELS.
    It is stored as an RGB PNG (longest side at most REFERENCE_SIDE) in DIR/refs, outside every
    project folder, so an unsaved draft can refer to it too. The same picture gets the same name."""
    data = _reference_data(path)
    name = hashlib.sha256(data).hexdigest()
    target = _inside(_refs_folder(), name + ".png")
    if not (target.is_file() and target.stat().st_size == len(data)):
        _write_bytes(target, data)
    return name


def _reference_data(path) -> bytes:
    """The PNG that store_reference keeps for the image at `path`; ValueError when it cannot be one."""
    try:
        source = Path(path)
        size = source.stat().st_size
        regular = source.is_file()
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("Could not read the image file.") from error
    if not regular:
        raise ValueError("Could not read the image file.")
    if size > MAX_REFERENCE_BYTES:
        raise ValueError(f"Use an image smaller than {MAX_REFERENCE_BYTES // (1024 * 1024)} MB.")
    too_large = ValueError(f"Use an image of at most {MAX_REFERENCE_PIXELS // 1_000_000} megapixels.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # Only the formats the editor accepts: Pillow picks a reader by content, and some (EPS runs
            # Ghostscript, PSD, FITS...) have no place in a reference upload.
            with Image.open(source, formats=REFERENCE_FORMATS) as image:
                if image.width * image.height > MAX_REFERENCE_PIXELS:
                    raise too_large
                picture = _rgb(ImageOps.exif_transpose(image))  # decodes every pixel: a broken file fails here
        picture.thumbnail((REFERENCE_SIDE, REFERENCE_SIDE), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise too_large from error
    except (OSError, SyntaxError, ValueError) as error:
        if error is too_large:
            raise
        raise ValueError("This file is not an image the studio can read.") from error
    return buffer.getvalue()


def reference_path(name) -> Path | None:
    """The stored reference image `name`, or None for an invalid name or a missing file."""
    if not scene_parts.is_reference(name):
        return None
    try:
        path = _inside(_refs_folder(), name + ".png")
    except ValueError:
        return None
    return path if path.is_file() else None


# --- places and items (app/items.py) -------------------------------------------------------------------------

_ITEM_REFERENCES: dict[str, str] = {}  # sha256 of an item's main photo -> its stored reference name
_ITEM_REFERENCES_LOCK = threading.Lock()


def item_reference(item: dict) -> str:
    """The reference name (store_reference) of the item's main photo, without storing it: a scene whose reference
    images include it has the item. "" when the item has no photo that can be read."""
    photo = items.main_photo(item)
    digest = characters._digest(photo) if photo else ""
    if not digest:
        return ""
    with _ITEM_REFERENCES_LOCK:
        found = _ITEM_REFERENCES.get(digest)
    if found is None:
        try:
            found = hashlib.sha256(_reference_data(photo)).hexdigest()
        except (ValueError, OSError):
            return ""
        with _ITEM_REFERENCES_LOCK:
            if len(_ITEM_REFERENCES) >= 512:
                _ITEM_REFERENCES.clear()
            _ITEM_REFERENCES[digest] = found
    return found


def _named_items(prompt: str, references: tuple) -> list[dict]:
    """The usable places and items a picture text names (items.mentions) whose main photo is not among
    `references`, in the order they are named, as many as the scene has room for (scene_parts.MAX_REFERENCES).
    Remembered for a few seconds: the editor asks on every refresh."""
    room = scene_parts.MAX_REFERENCES - len(references)
    if room <= 0 or not (prompt or "").strip():
        return []

    def compute():
        pool = items.usable_items()
        if not pool:
            return ()
        named = [item for item in items.mentions(prompt, pool) if item_reference(item) not in references]
        return tuple(named[:room])

    return list(_memo("items", (prompt, tuple(references)), compute))


def named_items(scene: Scene) -> list[dict]:
    """The places and items the scene's picture text names that are not already among its reference images: each
    one's main photo is used as a reference image of the scene too when it is made (_with_named_items)."""
    return _named_items(scene.prompt, scene.references)


def _item_identities(settings: dict) -> list:
    """items.identity of every place or item a scene's settings (Scene.clip_settings) name, for its clip keys: [] for
    a scene that names none, whose keys stay as they were."""
    named = _named_items(str(settings.get("prompt") or ""), tuple(settings.get("references") or ()))
    return [items.identity(item) for item in named]


def _with_named_items(prompt: str, references: list[str], named: list[dict]) -> tuple[str, list[str]]:
    """The picture text and the reference image paths of a scene with the places and items its text names (named):
    each one's main photo, kept like a picked reference image (store_reference: the job uses that copy, which a later
    edit of the item cannot change), after the scene's own images. A text that numbers its images ([1], [2]) gets the
    number of each named item after its name, because only numbered images are used then (video_gen._shot_refs); a
    text without numbers uses every image as it is."""
    paths, numbered = list(references), []
    for item in named:
        photo = items.main_photo(item)
        try:
            path = reference_path(store_reference(photo)) if photo else None
        except (ValueError, OSError):
            path = None
        if path is None:
            continue
        paths.append(str(path))
        numbered.append((item, len(paths)))
    if numbered and refs.numbers(prompt):
        prompt = items.mark(prompt, numbered)
    return prompt, paths


# --- scene cast, picture text and cues ------------------------------------------------------------------

def _texts(scene: Scene) -> list[str]:
    return [scene.prompt, *(cue.text for cue in scene.lines), scene.background,
            *(effect.text for effect in scene.effects)]


def _unavailable(slug: str) -> str:
    """Why a character of the cast cannot be in a scene: it is gone (deleted or archived) or has no photos."""
    person = characters.get(slug)
    if person is None:
        return (f"The character {slug} no longer exists. Remove it from this project's cast and scenes, or create "
                "it again on the Characters page.")
    return f"{person['name']} has no reference photos yet. Add photos on the Characters page."


def _profiles(slugs) -> list[dict]:
    """The usable characters of `slugs`, in order; ValueError naming the first one that is not."""
    members = []
    for slug in slugs:
        person = characters.timeline_profile(slug)
        if person is None:
            raise ValueError(_unavailable(slug))
        members.append(person)
    return members


def cast_notice(project: Project) -> str:
    """A message for a project whose cast names characters that no longer exist ("" when every one exists);
    the editor shows it when the project opens. The project itself is never changed by opening it."""
    named = [*project.cast_slugs, *(slug for scene in project.scenes for slug in scene.cast or ())]
    return characters.missing_message(named)


def _scene_cast(project: Project, scene: Scene) -> list[dict]:
    texts = _texts(scene)
    if llm.blocked(*texts):
        raise ValueError(llm.BLOCK_MSG)
    if any(not 1 <= number <= len(scene.references) for number in refs.numbers(scene.prompt)):
        raise ValueError("Attach the reference images the description refers to, or remove their numbers.")
    if scene.cast is not None:
        library = characters.all_characters()
        known = voices.tokens()
        for token in dialogue.tags(cast.strip_mentions(scene.prompt, library)):
            if token.casefold() not in known:
                raise ValueError(f"No voice or character is named @{token}.")
        return _profiles(scene.cast)
    speakers = [cue.character for cue in scene.lines if cue.character]
    if not (project.universe or project.reference_characters):
        _profiles(project.characters)
        # the film's style decides whether an animal or creature may appear (refused in the Timeline's own words)
        members, error, _ = cast.check_prompt(scene.prompt, project.characters, [scene.prompt], style=project.style)
        if error:
            refused = _creatures_refused(project, members)
            raise ValueError(_creature_message(refused[0], project) if refused else error)
        if speakers and not cast.mentions(scene.prompt, members):
            members = []  # nobody is named: the people who speak are the scene's cast, not the main character
        return _with_speakers(project, list(members), speakers)
    return _named_cast(project, scene)


def _with_speakers(project: Project, members: list[dict], speakers: list[str]) -> list[dict]:
    """The members plus every character who speaks a line (cast that follows the description)."""
    chosen = {person["slug"] for person in members}
    for slug in dict.fromkeys(speakers):
        if slug in chosen:
            continue
        if slug not in project.cast_slugs:
            raise ValueError(f"Add {slug} to this project's cast first.")
        members.extend(_profiles([slug]))
        chosen.add(slug)
    if len(members) > cast.MAX_CAST:
        raise ValueError(f"Use at most {cast.MAX_CAST} characters in a scene.")
    return members


def _named_cast(project: Project, scene: Scene) -> list[dict]:
    slugs = project.cast_slugs
    library = characters.all_characters()
    named = cast.mentions(scene.prompt, library)
    outside = [person["name"] for person in named if person["slug"] not in slugs]
    if outside:
        raise ValueError(f"Add {', '.join(outside)} to this project's cast first.")
    unknown = dialogue.tags(cast.strip_mentions(scene.prompt, library))
    known = voices.tokens()
    for token in unknown:
        if token.casefold() not in known:
            raise ValueError(f"No voice or character is named @{token}.")
    speakers = [cue.character for cue in scene.lines if cue.character]
    outside = [slug for slug in speakers if slug not in slugs]
    if outside:
        raise ValueError(f"Add {', '.join(dict.fromkeys(outside))} to this project's cast first.")
    selected = list(dict.fromkeys([*(person["slug"] for person in named), *speakers]))
    if not selected and not (project.universe or project.reference_characters):
        selected = list(project.characters[:1])
    if len(selected) > cast.MAX_CAST:
        raise ValueError(f"Use at most {cast.MAX_CAST} characters in a scene.")
    return _profiles(selected)


def _talking(picture: str, names: list[str], spoken: bool) -> str:
    """The picture text with a "<Name> talks." sentence per speaking character and the lip hint. `spoken` is False
    for a scene where nobody on screen speaks (a narrator's voice-over in an animated film): then the text is left
    as it is, so the video model does not move mouths for a voice that is not theirs (defect I)."""
    if not spoken:
        return picture
    text = picture.strip()
    if names:
        if text and text[-1] not in ".!?":
            text += "."
        text = " ".join([text, *(f"{name} talks." for name in names)]).strip()
    text = text.rstrip(" .!?")
    return f"{text}, natural lip movement" if text else "natural lip movement"


def _legacy_cues(scene: Scene, members: list[dict], lines: list, pool: list, people: list) -> tuple:
    """Quoted lines as fully resolved cues: the character who says each one, else the library voice
    the old pipeline picked for it (dialogue.voices_for). Legacy speaker choices always win."""
    known = {person["slug"] for person in members}
    if scene.speakers:
        lines = dialogue.assign_speakers(lines, [slug if slug in known else "" for slug in scene.speakers], people)
    lead = members[0] if members else None
    chosen = dialogue.voices_for(lines, voices.catalog(pool), voices.for_character(lead, pool))
    cues = []
    for index, (line, voice) in enumerate(zip(lines, chosen)):
        speaker = scene.speakers[index] if index < len(scene.speakers) and scene.speakers[index] else line.character
        if not speaker and voice.startswith(dialogue.CHARACTER_VOICE):
            # The lead's own voice, which the old pipeline gives an unnamed line of the lead's gender. It has no
            # "@" name, so the line goes to the lead, who speaks with that voice.
            owner = voice[len(dialogue.CHARACTER_VOICE):]
            speaker = owner if owner in known else ""
        elif not speaker and voice:
            speaker = scene_parts.VOICE_PREFIX + voice
        cues.append(scene_parts.Cue(line.text, speaker))
    return tuple(cues)


def effective(project: Project, scene: Scene, members: list[dict] | None = None) -> tuple[str, tuple]:
    """(picture text, cues) used for generation and sound previews.

    A structured scene: its picture text with a "<Name> talks." sentence per speaking character
    (in the order they first speak) and ", natural lip movement" when it has lines. A legacy scene
    with quoted lines: dialogue.parse's picture text and its lines as resolved cues (the character,
    or the "@voice" the old pipeline picked) with automatic starts. members: the scene's character
    profiles (_scene_cast)."""
    if members is None:
        members = _scene_cast(project, scene)
    if scene.lines or not dialogue.quotes(scene.prompt):
        names = {person["slug"]: person.get("name") or person["slug"] for person in members}
        only = members[0]["slug"] if len(members) == 1 else ""
        speaking: list[str] = []
        for cue in scene.lines:
            slug = cue.character or (only if not cue.speaker else "")
            if slug and slug not in speaking:
                speaking.append(slug)
        # In an animated film a line without a character is the narrator's voice-over (video_gen: no lip sync for it),
        # so only a speaking character makes the picture talk; a Realistic film's "@voice" line is said by someone in
        # the picture, as it always was.
        narrated = project.look.family != styles.REALISTIC
        spoken = bool(speaking) or (bool(scene.lines) and not narrated)
        return _talking(scene.prompt, [names.get(slug, slug) for slug in speaking], spoken), scene.lines
    pool = voices.all_voices()
    people = cast.people(members, pool)
    picture, _, lines = dialogue.parse(scene.prompt, voices.tokens(pool), people)
    return picture, _legacy_cues(scene, members, lines, pool, people)


def video_prompt(project: Project, index: int) -> str | None:
    """What the video model gets for scene `index` (its picture text, the compiled shot of a ShotSpec scene, with the
    style words), as far as it is known before the scene is made; None when it depends on a reference person's look
    (described only when the scene runs) or the cast cannot be worked out."""
    if not _valid_index(project, index):
        return None
    scene = project.scenes[index]
    try:
        members = _members_cached(project, scene)
        picture = effective(project, scene, members)[0]
    except ValueError:
        return None
    if index > 0 and not scene.cut:
        picture = _CONTINUE + picture
    if scene.shot:
        return video_gen._with_style(refs.strip(picture), project.video_style)
    return video_gen.scene_wan_prompt(picture, project.video_style, members, scene.cast is not None,
                                      _references(scene))


def _without_quotes(prompt: str) -> str:
    """The picture text of a scene whose quoted lines moved to Dialogue: "says „…”" becomes
    "talks", a quote without a speaking verb is left out, the rest stays as written (text shown in the
    picture, dialogue.shown, too)."""
    parts, position = [], 0
    for match in dialogue.spoken(prompt):
        before = prompt[position:match.start()]
        stop = "." if match.group(1).strip()[-1:] in (".", "!", "?") else ""
        if dialogue._SAYS.search(before):
            parts.append(dialogue._SAYS.sub("talks", before) + stop)
        else:
            parts.append(before.rstrip(" :") + stop)
        position = match.end()
    parts.append(prompt[position:])
    text = re.sub(r"[ \t]{2,}", " ", "".join(parts))
    text = re.sub(r"[ \t]+([,.!?;:])", r"\1", text)
    text = re.sub(r"([.!?,;:])[.!?,;:]+", r"\1", text)
    return re.sub(r"^[\s.,;:!?]+", "", text).rstrip(" ,:;")


def quotes_to_cues(project: Project, scene: Scene) -> Scene:
    """The scene with its quoted lines moved into Dialogue lines (who says each one is kept) and
    the quotes taken out of the picture text; the rest of the picture text stays as written."""
    if scene.lines or not dialogue.quotes(scene.prompt):
        return scene
    try:
        members = _scene_cast(project, scene)
    except ValueError:
        members = []
    pool = voices.all_voices()
    people = cast.people(members, pool)
    lines = dialogue.parse(scene.prompt, voices.tokens(pool), people)[2]
    cues = _legacy_cues(scene, members, lines, pool, people)
    return replace(scene, prompt=_without_quotes(scene.prompt), lines=cues, speakers=())


def with_lines(scene: Scene, texts) -> Scene:
    """The scene with its dialogue lines saying `texts` (an AI edit's plan, app/llm.py plan_edit): each line keeps who
    says it and when; a line more is said by the speaker of the last line, after it; lines past the texts go. When a
    kept time no longer fits the new texts, every line is placed after the one before it."""
    texts = [" ".join(str(text).split()) for text in texts or () if str(text).strip()]
    cues = []
    for number, text in enumerate(texts):
        if number < len(scene.lines):
            cues.append(replace(scene.lines[number], text=text))
        else:
            cues.append(scene_parts.Cue(text, scene.lines[-1].speaker if scene.lines else "",
                                        pace=scene.lines[-1].pace if scene.lines else 0.0))
    try:
        scene_parts.check(tuple(cues), scene.effects, scene.seconds)
    except ValueError:
        cues = [replace(cue, start=None) for cue in cues]
    return replace(scene, lines=tuple(cues))


# --- scene sound -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _SoundInputs:
    cues: tuple
    speakers: tuple
    prompt: str   # the sound prompt of the background
    seed: int
    key: str
    talk: bool


@functools.lru_cache(maxsize=1024)
def _bed_prompt(prompt: str, background: str) -> str:
    """The scene's background prompt as the editor derives it (without the cast): what tells two
    scenes' beds apart for _bed_ordinal."""
    return scene_sound.background_prompt(refs.plain(dialogue.parse(refs.plain(prompt))[1]), background)


def _bed_ordinal(project: Project, index: int) -> int:
    """How many earlier scenes would make the same bed (same background prompt, effects and length)."""
    if not _valid_index(project, index):
        return 0
    scenes = project.scenes

    def bed(scene: Scene) -> tuple:
        return _bed_prompt(scene.prompt, scene.background), scene.effects, scene.seconds

    mine = bed(scenes[index])
    return sum(bed(scene) == mine for scene in scenes[:index])


def _sound_seed(project: Project, prompt: str, scene: Scene, ordinal: int) -> int:
    """The sound seed of one scene: from the project, the scene's background prompt, effects and length,
    and how many earlier scenes share them, so two scenes never get the same bed (or the same effect
    renders) by accident. Lines, picture-only edits, levels and moves that keep the order of such
    scenes leave it alone, so a sound preview stays reusable."""
    return int(_hash(["timeline-sound", project.id, prompt, [effect.to_json() for effect in scene.effects],
                      scene.seconds, ordinal])[:8], 16) % (2 ** 31 - 1)


def _index_of(project: Project, scene: Scene) -> int | None:
    found = next((index for index, item in enumerate(project.scenes) if item is scene), None)
    if found is None:
        found = next((index for index, item in enumerate(project.scenes) if item == scene), None)
    return found


def _sound_inputs(project: Project, scene: Scene, members: list[dict], *, voice: bool = True,
                  index: int | None = None) -> _SoundInputs:
    """The scene's sound: its lines, speakers, background prompt, seed and key. index: the scene's place
    in the project (found by the scene itself when not given)."""
    picture, cues = effective(project, scene, members)
    cues = tuple(cues) if voice else ()
    speakers = tuple(scene_sound.resolve_speakers(cues, members)) if cues else ()
    prompt = scene_sound.background_prompt(refs.plain(picture), scene.background)
    if index is None:
        index = _index_of(project, scene)
    seed = _sound_seed(project, prompt, scene, _bed_ordinal(project, index) if index is not None else 0)
    talk = bool(cues)
    key = scene_sound.sound_key(cues, speakers, prompt, scene.effects, scene.seconds, seed, voice=talk,
                                background=True)
    return _SoundInputs(cues, speakers, prompt, seed, key, talk)


def _sound_folder(project: Project, key: str) -> Path:
    if not isinstance(key, str) or not _HASH.fullmatch(key):
        raise ValueError("Invalid scene sound.")
    return _inside(_root(project.id), "sound", key)


def _audition(project: Project, index: int, scene: Scene, members: list[dict]):
    """The prepared sound preview of the scene's current inputs, or None."""
    if not project.audio or _saved_root(project) is None:
        return None
    try:
        inputs = _sound_inputs(project, scene, members, index=index)
        return scene_sound.load(_sound_folder(project, inputs.key), inputs.key)
    except (ValueError, OSError, MediaError):
        return None


def _ensure_sound_models(talk: bool, cb) -> None:
    """Download VoxCPM2 (for lines) and EzAudio-XL on first use, before any sound is made."""
    model_manager.announce_checks(([tts.MODEL_ID] if talk else []) + [sound.MODEL_ID], cb)
    needed = ([tts.MODEL_ID] if talk and not tts.available() else []) + ([] if sound.available() else [sound.MODEL_ID])
    if needed:
        try:
            error = model_manager.ensure(needed, cb)
        finally:
            _MEMO.pop(("installed", "voice"), None)
            _MEMO.pop(("installed", "background"), None)
        if error:
            raise ValueError(error)


class _SpeechAhead:
    """The lines of the scenes a prepare job is about to give a new sound, synthesized together.

    The first scene that has to synthesize its lines also synthesizes, in the same VoxCPM2 batch, those of
    the scenes this job will certainly make after it: the next scenes of the job in an unbroken anchored
    chain (each continues from a picture that is about to be new, so it is made again), at most
    SPEECH_AHEAD_SCENES of them. One model load then serves them all instead of one per scene. A line is seeded
    on its own, so it sounds as it would in its scene's own batch. Use it as a context manager: the files live in
    a temporary folder for the job, so a failed or paused job loses only the lines of that small window."""

    def __init__(self, project: Project, order, casts: dict):
        self.project, self.order, self.casts = project, list(order), casts
        self._lines: dict[tuple[int, str], tuple[Path, ...]] = {}
        self._folder: tempfile.TemporaryDirectory | None = None
        self._batches = 0

    def __enter__(self) -> "_SpeechAhead":
        return self

    def __exit__(self, *_exc) -> None:
        self._lines.clear()
        if self._folder is not None:
            self._folder.cleanup()
            self._folder = None

    def _followers(self, index: int) -> list:
        """(index, sound inputs) of up to SPEECH_AHEAD_SCENES of the job's later scenes anchored to `index` in an
        unbroken chain that will synthesize lines: they speak and have no sound made for their inputs yet."""
        found, previous = [], index
        for number in self.order[self.order.index(index) + 1:] if index in self.order else ():
            scene = self.project.scenes[number]
            if (number != previous + 1 or scene.cut or number not in self.casts
                    or len(found) >= SPEECH_AHEAD_SCENES):
                break
            previous = number
            try:
                inputs = _sound_inputs(self.project, scene, self.casts[number], index=number)
                ready = scene_sound.load(_sound_folder(self.project, inputs.key), inputs.key) is not None
            except (ValueError, OSError, MediaError):
                continue  # the scene reports its own problem when its turn comes
            if inputs.talk and not ready:
                found.append((number, inputs))
        return found

    def lines(self, index: int, inputs, cb) -> tuple[Path, ...] | None:
        """The line files for scene `index` with these sound inputs, or None when the scene synthesizes
        its own (it alone needs speech, or the shared batch failed: each scene then reports its own error)."""
        kept = self._lines.pop((index, inputs.key), None)
        if kept is not None or not inputs.talk:
            return kept
        batch = [(index, inputs)] + self._followers(index)
        if len(batch) == 1:
            return None
        if self._folder is None:
            self._folder = tempfile.TemporaryDirectory(prefix="contsy_speech_", ignore_cleanup_errors=True)
        self._batches += 1
        folder = Path(self._folder.name) / f"batch{self._batches}"
        numbers = ", ".join(str(number + 1) for number, _ in batch)
        try:
            made = scene_sound.synthesize_lines([(item.cues, item.speakers) for _, item in batch], folder,
                                                lambda fraction, message: cb(0.4 * fraction,
                                                                             f"{message} (scenes {numbers})"))
        except ValueError as error:
            print(f"[timeline] Shared speech batch failed, each scene synthesizes its own lines: {error}",
                  flush=True)
            return None
        for (number, item), paths in zip(batch[1:], made[1:]):
            self._lines[(number, item.key)] = paths
        return made[0]


def _sounds_ahead(project: Project, planned, casts: dict, cb, then=()) -> None:
    """A job that will make the pictures of two or more scenes first makes the sounds they still need, together:
    every line in one VoxCPM2 batch and every background in one EzAudio session, so each sound model loads once for
    the job instead of once per scene (EzAudio's text encoder took 14 s in a film's first scene and 4 s in each later
    one, 2026-11-15). Each sound goes to its scene's sound folder, where the scene finds it when its turn comes; every
    line and every background is seeded on its own, so it is the sound the scene would make alone. A sound that cannot
    be made here is left to its scene, which reports its own problem. `then`: files the first scene's picture starts
    with, read after the sound models in the same pass (in a queued job)."""
    if not project.audio or _saved_root(project) is None:
        engine_runtime.read_ahead_sound(False, False, then)
        return
    todo = []
    for number in planned:
        members = casts.get(number)
        if members is None:
            continue
        scene = project.scenes[number]
        try:
            inputs = _sound_inputs(project, scene, members, index=number)
            if scene_sound.load(_sound_folder(project, inputs.key), inputs.key) is None:
                todo.append((number, scene, inputs))
        except (ValueError, OSError, MediaError):
            continue  # the scene reports its own problem when its turn comes
    if len(todo) < 2:
        engine_runtime.read_ahead_sound(False, False, then)
        return
    numbers = ", ".join(str(number + 1) for number, _, _ in todo)

    def report(fraction, message):
        cb(0.0, f"Sound of scenes {numbers} · {message}")

    talking = [(number, inputs) for number, _, inputs in todo if inputs.talk]
    try:
        _ensure_sound_models(bool(talking), report)
        engine_runtime.read_ahead_sound(bool(talking), True, then)
        runtime = engine_runtime.engine()
        with tempfile.TemporaryDirectory(prefix="contsy_sounds_", ignore_cleanup_errors=True) as work:
            lines = {}
            if talking:
                runtime.trim()  # the card for VoxCPM2; models kept warm wait in RAM
                made = scene_sound.synthesize_lines([(item.cues, item.speakers) for _, item in talking], Path(work),
                                                    report)
                lines = {number: paths for (number, _), paths in zip(talking, made)}
                runtime.trim()  # VoxCPM2's leftovers off the card before EzAudio comes in
            with runtime.audio_session():
                for number, scene, inputs in todo:
                    cancel.check()
                    try:
                        scene_sound.prepare(inputs.cues, inputs.speakers, inputs.prompt, scene.effects, scene.seconds,
                                            inputs.seed, _sound_folder(project, inputs.key), report,
                                            voice=inputs.talk, background=True, lines=lines.get(number))
                    except (ValueError, OSError, MediaError) as error:
                        print(f"[timeline] Scene {number + 1} makes its own sound: {error}", flush=True)
    except (ValueError, OSError, MediaError) as error:
        print(f"[timeline] The scenes make their own sound: {error}", flush=True)
    finally:
        _changed()


def _scene_sound(project: Project, index: int, scene: Scene, members: list[dict], cb,
                 speech: _SpeechAhead | None = None):
    """One sound per scene: the scene's current sound preview, else its sound made now with the scene's
    sound seed in <project>/sound/<sound key>/ - the folder 'Prepare sound preview' uses - so the
    preview, the clip and later takes share it. The models are downloaded first when missing. speech:
    the job's shared speech batch (_SpeechAhead), when it runs one."""
    inputs = _sound_inputs(project, scene, members, index=index)
    folder = _sound_folder(project, inputs.key)
    found = scene_sound.load(folder, inputs.key)
    if found is not None:
        return found
    _ensure_sound_models(inputs.talk, lambda fraction, message: cb(0.0, message))
    lines = speech.lines(index, inputs, cb) if speech is not None else None
    try:
        return scene_sound.prepare(inputs.cues, inputs.speakers, inputs.prompt, scene.effects, scene.seconds,
                                   inputs.seed, folder, cb, voice=inputs.talk, background=True, lines=lines)
    finally:
        _changed()


def sound_preview(project: Project, index: int):
    """The scene's current sound preview (scene_sound.SceneSound) when one was prepared for exactly
    these inputs, else None. Cheap: the identity is remembered and only the manifest is read."""
    if not project.audio or not _valid_index(project, index) or _saved_root(project) is None:
        return None
    scene = project.scenes[index]
    ordinal = _bed_ordinal(project, index)

    def compute():
        try:
            return _sound_inputs(project, scene, _members_cached(project, scene), index=index)
        except (ValueError, OSError, MediaError):
            return None

    inputs = _memo("sound", (project.id, _cast_fields(project), scene, ordinal), compute)
    if inputs is None:
        return None
    try:
        return scene_sound.load(_sound_folder(project, inputs.key), inputs.key)
    except (ValueError, OSError):
        return None


def _preview_name(owner: str, *values) -> str:
    return f"{owner[:16]}-{_hash([_PREVIEW_VERSION, *values])[:24]}"


def _prune_previews(folder: Path, name: str) -> None:
    """Keep the newest few previews of the same clip or audition and file type."""
    owner, suffix = name.split("-", 1)[0], Path(name).suffix
    try:
        mine = [path for path in folder.iterdir()
                if (match := _PREVIEW.fullmatch(path.name)) and match.group(1) == owner and path.suffix == suffix]
        mine.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    except OSError:
        return
    for path in mine[_KEEP_PREVIEWS:]:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def _cached_preview(root: Path, name: str, build) -> Path:
    """previews/<name>, made by build(path) once and kept while its inputs stay the same."""
    folder = _inside(root, "previews")
    target = _inside(folder, name)
    if target.is_file():
        return target
    folder.mkdir(parents=True, exist_ok=True)
    temporary = _inside(folder, f".{uuid.uuid4().hex}{target.suffix}")
    try:
        build(temporary)
        if not temporary.is_file():
            raise OSError("The preview was not written.")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    _prune_previews(folder, name)
    return target


def preview_mix(project: Project, index: int) -> Path | None:
    """The scene's sound mixed with its levels as the export mixes it (a cached WAV): the current
    sound preview, unless the scene's clip was made from another sound (its stems are what the
    export plays), else the stems of the scene's clip; None when there is neither."""
    if not project.audio or not _valid_index(project, index) or (root := _saved_root(project)) is None:
        return None
    scene = project.scenes[index]
    levels = scene.voice_db, scene.background_db
    try:
        audition = sound_preview(project, index)
        record = _scene_clip(project, index)
        if record is not None and not record["audio_sources"]:
            record = None
        if audition is not None and (record is None or record.get("sound_key") == audition.key):
            return _cached_preview(root, _preview_name(audition.key, "mix", audition.key, *levels) + ".wav",
                                   lambda out: scene_sound.preview_mix(audition, *levels, out))
        if record is not None:
            background, voice = record["audio_sources"]
            effects = record.get("effects_source")
            return _cached_preview(root, _preview_name(record["sha256"], "clip", record["sha256"], *levels) + ".wav",
                                   lambda out: scene_sound.mix_stems(background, voice, scene.seconds, *levels, out,
                                                                     effects=effects))
    except (ValueError, OSError, MediaError) as error:
        print(f"[timeline] Could not mix the scene preview: {error}", flush=True)
    return None


def leveled_clip(project: Project, index: int) -> Path | None:
    """The scene's clip as the export will play it: a cached copy of its picture with its stems mixed
    by the export chain at the scene's levels (0 dB / 0 dB too, so the scene preview always sounds
    like the export). The clip itself only when it has no separate stems or sound is off."""
    if not _valid_index(project, index) or (root := _saved_root(project)) is None:
        return None
    record = _scene_clip(project, index)
    if not record:
        return None
    if not project.audio or not record["audio_sources"]:
        return record["path"]
    scene = project.scenes[index]
    levels = scene.voice_db, scene.background_db
    background, voice = record["audio_sources"]
    effects = record.get("effects_source")

    def build(out: Path) -> None:
        # The picture is copied as it is; the clip's own mix is replaced by the export mix.
        scene_sound.leveled_clip(record["path"], background, voice, scene.seconds, *levels, out, effects=effects)

    try:
        return _cached_preview(root, _preview_name(record["sha256"], "leveled", record["sha256"], *levels) + ".mp4",
                               build)
    except (ValueError, OSError, MediaError) as error:
        print(f"[timeline] Could not mix the scene preview with the export's sound: {error}", flush=True)
        return record["path"]


# --- sound checks ------------------------------------------------------------------------------------------

_ORDER = {scene_parts.ERROR: 0, scene_parts.WARNING: 1, scene_parts.INFO: 2}


def _speaker_issues(cues: tuple, members: list[dict], pool: list) -> list:
    issues = []
    names = {person["slug"] for person in members}
    for line, cue in enumerate(cues):
        if cue.voice:
            if voices.get(cue.voice, pool) is None:
                issues.append(scene_parts.Issue(scene_parts.ERROR, f"Line {line + 1}: no voice named @{cue.voice}.",
                                                line=line))
        elif cue.character:
            if cue.character not in names:
                try:
                    profile = characters.get(cue.character) or {}
                except (OSError, ValueError):
                    profile = {}
                issues.append(scene_parts.Issue(
                    scene_parts.ERROR, f"Line {line + 1}: {profile.get('name') or cue.character} is not in this scene.",
                    line=line))
        elif len(members) != 1:
            issues.append(scene_parts.Issue(scene_parts.ERROR, f"Line {line + 1}: choose who says it.", line=line))
    if issues or not cues:
        return issues
    try:
        speakers = scene_sound.resolve_speakers(cues, members, pool)
    except ValueError as error:
        return [scene_parts.Issue(scene_parts.ERROR, str(error))]
    missing, told = set(), set()
    for line, speaker in enumerate(speakers):
        voice = voices.get(speaker.voice, pool)
        if (voice is None or not Path(voice.path).is_file()) and speaker.voice not in missing:
            missing.add(speaker.voice)
            label = voice.name if voice else speaker.voice
            issues.append(scene_parts.Issue(scene_parts.ERROR, f"Line {line + 1}: the voice sample of {label} is "
                                                               "missing. Choose another voice or add it again.",
                                            line=line))
        if speaker.character and speaker.character not in told:
            told.add(speaker.character)
            profile = next((person for person in members if person["slug"] == speaker.character), {})
            if voices.get(profile.get("voice") or "", pool) is None:
                label = voice.name if voice else speaker.voice
                issues.append(scene_parts.Issue(scene_parts.INFO, f"{speaker.name} has no voice of their own, "
                                                                  f"so their lines use {label}.", line=line))
    return issues


def _measured(project: Project, index: int, scene: Scene, members: list[dict], cues: tuple, *,
              cached: bool) -> list[float] | None:
    """Measured line lengths: from the current sound preview, else from the scene's clip when it
    speaks exactly these lines."""
    if not cues:
        return None
    audition = sound_preview(project, index) if cached else _audition(project, index, scene, members)
    if audition is not None and len(audition.timings) == len(cues):
        return [max(0.0, float(end) - float(start)) for start, end in audition.timings]
    record = _scene_clip(project, index)
    timings = record.get("dialogue_timings") if record else None
    if isinstance(timings, list) and len(timings) == len(cues):
        try:
            if all(line["text"] == cue.text for line, cue in zip(timings, cues)):
                lengths = [float(line["end"]) - float(line["start"]) for line in timings]
                if all(math.isfinite(length) and length >= 0 for length in lengths):
                    return lengths
        except (KeyError, TypeError, ValueError):
            pass
    return None


def _nearly_silent(project: Project, index: int, scene: Scene, members: list[dict], *, cached: bool) -> bool:
    """The background of the scene's sound preview, or of its clip, is one the export would hear as
    nearly silent (scene_sound.quiet). A preview carries its measure in its manifest; a clip's stem is
    measured once per file version (scene_sound.bed_loudness)."""
    try:
        audition = sound_preview(project, index) if cached else _audition(project, index, scene, members)
        if audition is not None and audition.background is not None:
            if audition.bed_level is not None and audition.bed_audible is not None:
                level, audible = audition.bed_level, audition.bed_audible
            else:
                level, audible = scene_sound.bed_loudness(audition.background, audition.seconds, audition.overlay)
            if scene_sound.quiet(level, audible):
                return True
        record = _scene_clip(project, index)
        if record and record["audio_sources"] and (audition is None or record.get("sound_key") != audition.key):
            level, audible = scene_sound.bed_loudness(record["audio_sources"][0], record["seconds"],
                                                      record.get("effects_source"))
            return scene_sound.quiet(level, audible)
    except (ValueError, OSError, MediaError):
        return False
    return False


def _scene_issues(project: Project, index: int, *, cached: bool) -> list:
    scene = project.scenes[index]
    issues = [scene_parts.Issue(scene_parts.ERROR, f"Reference image {number} is missing. Add it again or "
                                                   "remove it.")
              for number, name in enumerate(scene.references, 1) if reference_path(name) is None]
    try:
        members = _members_cached(project, scene) if cached else _scene_cast(project, scene)
        picture, cues = effective(project, scene, members)
    except ValueError as error:
        issues.append(scene_parts.Issue(scene_parts.ERROR, str(error)))
        return sorted(issues, key=lambda issue: _ORDER.get(issue.level, 3))
    issues += _style_issues(project, members)
    if content := _creature_content(scene, members, _previous_members(project, index)):
        issues.append(scene_parts.Issue(scene_parts.ERROR, content))
    if not project.audio:
        if cues or scene.effects or scene.background.strip():
            issues.append(scene_parts.Issue(scene_parts.WARNING, "Sound is off for this project, so the lines, "
                                                                 "background and effects are not heard."))
        return sorted(issues, key=lambda issue: _ORDER.get(issue.level, 3))
    pool = _memo("voices", None, voices.all_voices) if cached else voices.all_voices()
    issues += _speaker_issues(tuple(cues), members, pool)
    # Models are downloaded on first use (model_manager.ensure), so a missing one is only news.
    if cues and not _installed("voice"):
        issues.append(scene_parts.Issue(scene_parts.INFO, "VoxCPM2 (speech) will be downloaded on first use."))
    if not _installed("background"):
        issues.append(scene_parts.Issue(scene_parts.INFO, "EzAudio-XL (background sound and effects) will be "
                                                          "downloaded on first use."))
    issues += scene_parts.timing_issues(cues, scene.effects, scene.seconds,
                                        _measured(project, index, scene, members, tuple(cues), cached=cached))
    if _nearly_silent(project, index, scene, members, cached=cached):
        issues.append(scene_parts.Issue(scene_parts.WARNING, NEARLY_SILENT))
    if not scene.background.strip():
        derived = scene_sound.background_prompt(refs.plain(picture), "")
        issues.append(scene_parts.Issue(scene_parts.INFO, f"Background sound from the picture: {derived}"))
    return sorted(issues, key=lambda issue: _ORDER.get(issue.level, 3))


def _creatures_refused(project: Project, members) -> list[dict]:
    """The animals and creatures among `members` when the film's style is Realistic or one of the owner's own based on
    it: they are drawn only in animated styles (characters.creature_style_ok), so such a scene is not made."""
    if characters.creature_style_ok(project.style):
        return []
    return [person for person in members if characters.is_creature(person)]


def _creature_message(person: dict, project: Project | None = None) -> str:
    problem = characters.creature_style_problem(project.style) if project is not None else ""
    if problem and problem != characters.CREATURE_STYLE_MSG:
        return problem  # an own style whose words are sexual or nude
    return CREATURE_NEEDS_ANIMATION.format(name=person.get("name") or person.get("slug"))


def _previous_members(project: Project, index: int) -> list[dict]:
    """The characters of the scene before `index` when scene `index` continues from its last frame (no cut): they are
    in its start picture whatever its own cast is."""
    if index <= 0 or index >= len(project.scenes) or project.scenes[index].cut:
        return []
    try:
        return _scene_cast(project, project.scenes[index - 1])
    except ValueError:
        return []


def _creature_content(scene: Scene, members, previous=()) -> str:
    """Why a scene with an animal or creature is not made ("" = it may be): sexual or nude content in anything that
    reaches its picture - its texts, the looks and names of its characters - while a creature is in it or in the
    picture it continues from."""
    everyone = [person for person in (*(members or ()), *previous) if isinstance(person, dict)]
    if not any(characters.is_creature(person) for person in everyone):
        return ""
    words = [*_texts(scene), *(str(person.get("look") or "") for person in everyone),
             *(str(person.get("name") or "") for person in everyone)]
    return characters.CREATURE_SEXUAL_MSG if llm.sexual(*words) else ""


def _style_issues(project: Project, members: list[dict]) -> list:
    """The scene's characters and the film's style. An animal or creature in a film whose style is Realistic or based
    on it is an error (_creatures_refused), so no GPU time is spent on the scene. Characters drawn in another style
    than the film's are news: each scene then redraws them in the film's style, so their look may vary more from
    scene to scene than a character drawn in it."""
    people = [person for person in members if isinstance(person, dict)]
    refused = _creatures_refused(project, people)
    issues = [scene_parts.Issue(scene_parts.ERROR, _creature_message(person, project)) for person in refused]
    other = [person for person in people
             if person not in refused and person.get("style", styles.DEFAULT) != project.style]
    if not other:
        return issues
    names = ", ".join(f"{person['name']} ({style_library.label(person.get('style'))})" for person in other)
    return [*issues, scene_parts.Issue(scene_parts.INFO, f"Drawn in another style than this film "
                                                         f"({style_library.label(project.style)}): "
                                                         f"{names}. Each scene redraws them in the film's style; for "
                                                         "a steadier look, draw them in this style on the Characters "
                                                         "page.")]


SPEECH_TAIL = 0.5  # s of picture after the last line ends, in a suggested scene length


def suggested_length(project: Project, index: int) -> int | float | None:
    """The length scene `index` needs for all of its dialogue to be heard, when that is longer than the scene, else
    None: the end of its last line (measured by its sound preview or clip, else estimated, scene_parts.timings) plus
    SPEECH_TAIL, rounded up to a tenth of a second (scene_parts.length) and at most video_gen.MAX_SECONDS. The editor offers it before a
    scene is made; the scene itself keeps the length set for it. None without audio or dialogue, or when the
    scene's cast cannot be worked out (its issues say why)."""
    if not project.audio or not _valid_index(project, index):
        return None
    scene = project.scenes[index]
    try:
        members = _members_cached(project, scene)
        _, cues = effective(project, scene, members)
    except ValueError:
        return None
    if not cues:
        return None
    windows = scene_parts.timings(cues, _measured(project, index, scene, members, tuple(cues), cached=True))
    end = max(stop for _, stop in windows)
    if end <= scene.seconds + 1e-6:
        return None
    needed = scene_parts.length(min(video_gen.MAX_SECONDS, math.ceil((end + SPEECH_TAIL) * 10 - 1e-6) / 10))
    return needed if needed > scene.seconds else None


def scene_issues(project: Project, index: int) -> list:
    """Sound and dialogue problems of one scene (scene_parts.Issue, errors first): line timing
    (measured once a sound preview or clip exists), who speaks, voices and their samples, missing
    models and reference images, sound that is switched off, and the background derived from the
    picture. Remembered per inputs and project files, so the editor's refresh stays cheap."""
    if not _valid_index(project, index):
        return []
    return list(_memo("issues", (project, index, _folder_stamp(project)),
                      lambda: tuple(_scene_issues(project, index, cached=True))))


def check_scenes(project: Project, indices=None) -> list[tuple[int, scene_parts.Issue]]:
    """What generation would find in these scenes (all by default), as (scene index, issue) pairs, errors first
    within a scene: the checks enqueue() makes (a picture description, reference images, creatures and cast, who
    speaks) and the scene's sound and dialogue issues, measured fresh (not from the editor's memory). Nothing is
    queued and no model runs."""
    found = []
    for index in range(len(project.scenes)) if indices is None else indices:
        if not _valid_index(project, index):
            continue
        issues = []
        try:
            _validate_generation(project, [index])
        except _SceneError as error:
            issues.append(scene_parts.Issue(scene_parts.ERROR, str(error).split(": ", 1)[-1]))
        issues += _scene_issues(project, index, cached=False)
        texts = set()
        for issue in sorted(issues, key=lambda one: _ORDER.get(one.level, 3)):
            if issue.text not in texts:
                texts.add(issue.text)
                found.append((index, issue))
    return found


def _check_issues(project: Project, indices) -> None:
    for index in indices:
        for issue in _scene_issues(project, index, cached=False):
            if issue.level == scene_parts.ERROR:
                raise _SceneError(index, f"Scene {index + 1}: {issue.text}")


# --- takes and failures ------------------------------------------------------------------------------------

def _scene_signature(project: Project, scene: Scene, members: list[dict]) -> str:
    """What every take of one scene shares: project format, the scene's clip inputs and its people."""
    return _hash(["scene", project.aspect, _MODE_IN_KEYS, project.audio, scene.clip_settings(),
                  [person["slug"] for person in members]])


def _expected_anchor(project: Project, index: int, keys: list, records: dict) -> str | None:
    """The picture a take of scene `index` should continue from: None when it needs none."""
    if index == 0 or project.scenes[index].cut:
        return None
    key = keys[index - 1] if index - 1 < len(keys) else None
    previous = (records.get(key) or asset(project, key, verify=False)) if key else None
    return _picture_sha(previous) if previous else ""


def takes(project: Project, index: int) -> list[dict]:
    """The scene's takes, newest first: {"key", "created", "current", "anchor_ok", "label"}.
    anchor_ok: the take continues from the picture the previous scene shows now (always true for the
    first scene and after an intentional cut). After them the takes of the scene's other versions, joined to this one
    by AI edits either way (_other_versions), each with "settings": that version's scene, which use_take brings
    back."""
    if not _valid_index(project, index) or _saved_root(project) is None:
        return []
    scene = project.scenes[index]
    keys = _scene_keys(project)
    current = keys[index] if index < len(keys) else None
    records = _clip_records(project)

    def compute():
        try:
            return _scene_signature(project, scene, _members_cached(project, scene))
        except ValueError:
            return None

    signature = _memo("signature", (project.id, project.aspect, project.audio, _cast_fields(project),
                                    scene), compute)
    # a take made in another style would only show Outdated here: the takes of the film's style are listed
    found = {key: record for key, record in records.items()
             if signature is not None and record.get("scene_signature") == signature
             and record.get("style", prefs.STYLE["video"]) == project.video_style}
    if current and (record := records.get(current) or asset(project, current, verify=False)):
        found[current] = record
    expected = _expected_anchor(project, index, keys, records)
    ordered = sorted(found.items(), key=lambda item: (_created(item[1]), item[0]))
    numbers = {key: number for number, (key, _) in enumerate(ordered, 1)}
    result = []
    for key, record in reversed(ordered):
        created = _created(record)
        if expected is None:
            anchor_ok = True
        elif "anchor_sha256" in record:
            anchor_ok = record["anchor_sha256"] == expected
        else:
            anchor_ok = key == current  # made before takes were tracked: it was chosen for this place
        result.append({"key": key, "created": created, "current": key == current, "anchor_ok": anchor_ok,
                       "label": f"Take {numbers[key]} · {time.strftime('%d %b %H:%M', time.localtime(created))}"})
    if signature is not None:
        for key, record, settings in _other_versions(project, signature, records):
            created = _created(record)
            anchor_ok = True if expected is None else record.get("anchor_sha256") == expected
            result.append({"key": key, "created": created, "current": False, "anchor_ok": anchor_ok,
                           "settings": settings,
                           "label": f"Other version · {time.strftime('%d %b %H:%M', time.localtime(created))}"})
    return result


# --- versions of a scene (AI edits) --------------------------------------------------------------------------
# versions.json in the project folder: for the scene signature (_scene_signature) of every scene an AI edit saved,
# {"settings": that scene's clip settings, "bases": {key of the take the edit started from: {"settings": the scene's
# clip settings then, "signature": that take's scene signature, "time"}}}. Every version of a scene stays reachable
# from every other one, whichever was edited from which.
_VERSIONS_FILE = "versions.json"
_KEEP_VERSIONS = 300     # edited scene versions remembered per project
_OTHER_VERSION_TAKES = 12  # takes of a scene's other versions in its takes list


def _parse_versions(path: Path) -> dict:
    try:
        data = _read(path)
    except (OSError, ValueError):
        return {}
    result = {}
    for signature, entry in (data.items() if isinstance(data, dict) else ()):
        if not (_valid_key(signature) and isinstance(entry, dict) and isinstance(entry.get("settings"), dict)
                and isinstance(entry.get("bases"), dict)):
            continue
        bases = {key: base for key, base in entry["bases"].items()
                 if _valid_key(key) and isinstance(base, dict) and isinstance(base.get("settings"), dict)
                 and _valid_key(base.get("signature")) and isinstance(base.get("time"), (int, float))}
        result[signature] = {"settings": entry["settings"], "bases": bases}
    return result


def _versions(project: Project) -> dict:
    """versions.json as read (see above); {} without one. Remembered while the file stays the same."""
    root = _saved_root(project)
    if root is None:
        return {}
    path = _inside(root, _VERSIONS_FILE)
    try:
        stamp = path.stat()
    except OSError:
        return {}
    return _memo("versions", (project.id, stamp.st_mtime_ns, stamp.st_size), lambda: _parse_versions(path))


def _remember_versions(project: Project, entries: list[tuple[str, dict, str, str, dict]]) -> None:
    """Remember AI edits in versions.json: (scene signature of the edited scene, its clip settings, key of the take
    the edit starts from, that take's scene signature, the scene's clip settings then). Only the newest
    _KEEP_VERSIONS edited versions are kept."""
    if not entries:
        return
    path = _inside(_root(project.id), _VERSIONS_FILE)
    data = _parse_versions(path)
    now = time.time()
    for signature, settings, base, before, settings_before in entries:
        entry = data.pop(signature, None) or {"bases": {}}
        entry["settings"] = settings
        entry["bases"][base] = {"settings": settings_before, "signature": before, "time": now}
        data[signature] = entry  # the newest last
    while len(data) > _KEEP_VERSIONS:
        data.pop(next(iter(data)))
    _write(path, json.loads(json.dumps(data)))  # tuples as lists, as they are read back
    _changed()


def _other_versions(project: Project, signature: str, records: dict) -> list[tuple[str, dict, dict]]:
    """The takes of the scene's other versions, joined to version `signature` by AI edits either way, nearest
    version first and in each the newest take first: (key, record, that version's clip settings). Only takes made in
    the film's style, at most _OTHER_VERSION_TAKES."""
    links: dict[str, dict[str, dict]] = {}
    for edited, entry in _versions(project).items():
        for base in entry["bases"].values():
            if base["signature"] != edited:
                links.setdefault(edited, {})[base["signature"]] = base["settings"]
                links.setdefault(base["signature"], {})[edited] = entry["settings"]
    order, seen, queue = [], {signature}, [signature]
    while queue:
        for neighbour, settings in links.get(queue.pop(0), {}).items():
            if neighbour not in seen:
                seen.add(neighbour)
                order.append((neighbour, settings))
                queue.append(neighbour)
    found = []
    for version, settings in order:
        own = sorted(((key, record) for key, record in records.items()
                      if record.get("scene_signature") == version
                      and record.get("style", prefs.STYLE["video"]) == project.video_style),
                     key=lambda item: (_created(item[1]), item[0]), reverse=True)
        for key, record in own:
            found.append((key, record, settings))
            if len(found) >= _OTHER_VERSION_TAKES:
                return found
    return found


def use_take(project: Project, index: int, key: str) -> Project:
    """Show take `key` in scene `index` of the saved project (a take from takes()) and return the project as saved
    then. An earlier version (a take an AI edit was made from) brings its scene back too: the scene's text, dialogue
    and sound settings as they were (its mix levels stay), saved as a new revision."""
    with _LOCK:
        saved = load(project.id)
        if saved.key != project.key:
            raise ValueError("The timeline changed. Open the saved project again.")
        if has_active_job(saved):
            raise ValueError("Wait for this project's job in the queue to finish.")
        if not _valid_index(saved, index):
            raise ValueError("Choose a scene on the timeline.")
        chosen = next((take for take in takes(saved, index) if take["key"] == key), None) \
            if isinstance(key, str) else None
        if chosen is None:
            raise ValueError("Choose a take of this scene.")
        scene = saved.scenes[index]
        if chosen.get("settings") is not None:
            try:
                scene = Scene.from_json({**chosen["settings"], "voice_db": scene.voice_db,
                                         "background_db": scene.background_db})
            except (ValueError, TypeError) as error:
                raise ValueError("This earlier version cannot be brought back. Choose another take.") from error
        record = asset(saved, key)
        if not record or record["seconds"] != scene.seconds:
            raise ValueError("This take is missing or has changed. Generate a new take.")
        if scene != saved.scenes[index]:
            saved = save(saved.id, saved.revision, saved.title, saved.scenes[:index] + (scene,) + saved.scenes[index + 1:],
                         saved.aspect, saved.audio, list(saved.characters),
                         reference_characters=saved.reference_characters, universe=saved.universe,
                         universe_members=saved.universe_members, subtitles=saved.subtitles, style=saved.style)
        state = prepared(saved)
        keys = list(state["assets"])
        keys.extend([None] * (index + 1 - len(keys)))
        if keys[index] == key:
            return saved
        keys[index] = key
        _write_plan(saved, {**state, "assets": keys, "export_assets": state.get("export_assets", state["assets"])})
        return saved


def _failure_path(project: Project) -> Path:
    return _inside(_root(project.id), "failure.json")


def failure(project: Project) -> dict | None:
    """The last failed job of this version of the project: {"scene": index or None, "error", "kind"}."""
    if _saved_root(project) is None:
        return None
    try:
        record = _read(_failure_path(project))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("key") != project.key:
        return None
    scene, error, kind = record.get("scene"), record.get("error"), record.get("kind")
    if ((scene is not None and not _valid_index(project, scene)) or not isinstance(error, str)
            or kind not in JOB_KINDS):
        return None
    found = {"scene": scene, "error": error, "kind": kind}
    if kind == EDIT and isinstance(record.get("edit"), dict):
        found["edit"] = record["edit"]  # what retry_edit queues again
    return found


def _record_failure(project: Project, kind: str, scene: int | None, error: str, *, retake: bool = False,
                    edit: dict | None = None) -> None:
    """Remember a failed job. retake: the scene's generation itself failed (no usable face, a rejected or
    broken picture…), so the next attempt gets a new take — the same seed would repeat the same picture.
    Interruptions and an unavailable visual review keep the take, so a saved picture can be reused.
    edit: a failed AI edit's job parameters (retry_edit)."""
    record = {"key": project.key, "scene": scene, "error": str(error)[:2000], "kind": kind, "time": time.time()}
    if edit is not None:
        record["edit"] = edit
    if retake and scene is not None:
        record["retake"] = uuid.uuid4().hex
    elif scene is not None and (pending := _retake(project, scene)):
        # A take waiting for this scene stays: the attempt with it was interrupted (review outage, plan write),
        # so its saved picture or clip is reused instead of going back to the seed that already failed.
        record["retake"] = pending
    try:
        _write(_failure_path(project), record)
    except (OSError, ValueError) as problem:
        print(f"[timeline] Could not save the failure record: {problem}", flush=True)


def _interrupted(job: queue_manager.Job) -> bool:
    """A scene job that ended before its picture could be judged: paused, or stopped by an unavailable
    visual review or a plan write. Its take is resumed; a picture that failed gets a new take."""
    return job.status == queue_manager.STOPPED or (job.status == queue_manager.ERROR and (
        REVIEW_UNAVAILABLE in (job.notes or []) or PLAN_NOT_SAVED in (job.error or "")))


def _retake(project: Project, number: int) -> str | None:
    """The new take a failed scene's next attempt uses (see _record_failure), else None."""
    try:
        record = _read(_failure_path(project))
    except (OSError, ValueError):
        return None
    if (isinstance(record, dict) and record.get("key") == project.key and record.get("scene") == number
            and isinstance(record.get("retake"), str) and _ID.fullmatch(record["retake"])):
        return record["retake"]
    return None


def _explain(error) -> str:
    """A generation error with the next step for the creator when there is a clear one."""
    text = str(error)
    lower = text.lower()
    if (any(marker in lower for marker in _FACELESS) and not any(marker in lower for marker in _NOT_FRAMING)
            and NO_FACE_HINT not in text):
        text = f"{text.rstrip('. ')}. {NO_FACE_HINT}"
    return text


def _clear_failure(project: Project, kind: str, scene: int | None = None) -> None:
    """A finished job clears the failure it covers: a whole-timeline job any failure, a scene's
    clip a failure of that scene, a sound preview only a failed preview of that scene."""
    try:
        path = _failure_path(project)
        record = _read(path)
    except (OSError, ValueError):
        return
    if (not isinstance(record, dict) or record.get("key") != project.key or (scene is None and kind != SOUND)
            or (record.get("scene") == scene and (kind != SOUND or record.get("kind") == SOUND))):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


# --- queue -----------------------------------------------------------------------------------------------

def has_active_job(project: Project) -> bool:
    return any(job.status in (queue_manager.PENDING, queue_manager.RUNNING)
               and job.kind in JOB_KINDS and job.params.get("project") == project.id for job in queue_manager.snapshot())


def enqueue(project: Project, kind: str, *, scene_index: int | None = None,
            require_merge: bool = False) -> queue_manager.Job:
    with _LOCK:
        if load(project.id).key != project.key:
            raise ValueError("The timeline changed. Open the saved project again.")
        if has_active_job(project):
            raise ValueError("This project already has a job in the queue.")
        state = prepared(project)
        params = {"project": project.id, "key": project.key, "seconds": project.seconds}
        if kind == PREPARE:
            if scene_index is None:
                casts = _validate_generation(project, range(len(project.scenes)))
                missing = _stale(project, state["assets"], casts=casts)
                if not missing:
                    raise ValueError("Every scene already has a clip.")
                _check_issues(project, missing)
                params["seconds"] = sum(project.scenes[index].seconds for index in missing)
                action = "Generate missing clips"
            else:
                _validate_generation(project, [scene_index])
                _anchor(project, scene_index, state["assets"], verify=False)
                _check_issues(project, [scene_index])
                params.update(scene_index=scene_index, seconds=project.scenes[scene_index].seconds)
                if scene_index < len(state["assets"]) and state["assets"][scene_index]:
                    previous = next((job for job in reversed(queue_manager.snapshot())
                                     if job.kind == PREPARE and job.params.get("project") == project.id
                                     and job.params.get("key") == project.key
                                     and job.params.get("scene_index") == scene_index), None)
                    if retake := _retake(project, scene_index):
                        params["take"] = retake  # the last attempt's picture failed: try another one
                    elif previous and previous.params.get("take") and _interrupted(previous):
                        params["take"] = previous.params.get("take")  # resume an interrupted take
                    else:
                        params["take"] = uuid.uuid4().hex
                action = f"Generate scene {scene_index + 1}"
        elif kind == SOUND:
            if not project.audio:
                raise ValueError("Turn on sound for this project to prepare a sound preview.")
            _validate_generation(project, [scene_index])
            _check_issues(project, [scene_index])
            if sound_preview(project, scene_index) is not None:
                raise ValueError(f"The sound preview of scene {scene_index + 1} is ready.")
            params.update(scene_index=scene_index, seconds=project.scenes[scene_index].seconds)
            action = f"Sound preview · scene {scene_index + 1}"
        elif kind in (MERGE, RENDER):
            if not all_clips(project, state["assets"]):
                raise ValueError("Generate all clips on the timeline first.")
            if kind == RENDER and require_merge and not merged_clip(project, state):
                raise ValueError("Join and check the current timeline before rendering.")
            if require_merge:
                params["require_merge"] = True
            action = "Join and check" if kind == MERGE else "Render timeline"
        elif kind == PRODUCE:
            casts = _validate_generation(project, range(len(project.scenes)))
            if exported(project, state):
                raise ValueError("The movie is already exported. Use Download movie or Show in Gallery.")
            missing = _stale(project, state["assets"], casts=casts)
            _check_issues(project, missing)
            # The queue estimate in seconds of clip generation: missing clips plus joining and writing.
            assembly = (queue_manager.rate(MERGE) + queue_manager.rate(RENDER)) / queue_manager.rate(PREPARE)
            work = sum(project.scenes[index].seconds for index in missing) + assembly * project.seconds
            params["work_seconds"] = round(max(work, 0.1), 2)
            action = "Export movie"
        else:
            raise ValueError("Invalid timeline job.")
        return queue_manager.add(kind, f"{action} · {project.title}", params)


def _validate_generation(project: Project, indices) -> dict:
    if not project.scenes:
        raise ValueError("Add a scene to the timeline first.")
    casts = {}
    for index in indices:
        if not _valid_index(project, index):
            raise ValueError("Choose a scene on the timeline.")
        scene = project.scenes[index]
        if not scene.prompt.strip():
            raise _SceneError(index, f"Scene {index + 1}: write a description of the picture and motion.")
        try:
            casts[index] = _scene_cast(project, scene)
            refused = _creatures_refused(project, casts[index])  # the editor shows it too (_style_issues)
            if refused:
                raise ValueError(_creature_message(refused[0], project))
            if content := _creature_content(scene, casts[index], _previous_members(project, index)):
                raise ValueError(content)
            for number, name in enumerate(scene.references, 1):
                path = reference_path(name)
                if path is None or _sha(path) != name:
                    raise ValueError(f"Reference image {number} is missing or changed. Add it again or remove it.")
            if project.audio and (cues := effective(project, scene, casts[index])[1]):
                scene_sound.resolve_speakers(cues, casts[index])
        except ValueError as error:
            raise _SceneError(index, f"Scene {index + 1}: {error}") from error
    return casts


def _anchor(project: Project, index: int, keys: list, *, verify: bool = True) -> dict | None:
    if index == 0 or project.scenes[index].cut:
        return None
    if index - 1 < len(keys) and (previous := asset(project, keys[index - 1], verify=verify)):
        members = project.scenes[index].cast
        if members is not None and set(members) != set(previous["characters"]):
            raise ValueError("Enable ‘Intentional cut from previous scene’ to start with a different cast.")
        return previous
    raise ValueError(f"Generate scene {index} first to continue from its last frame.")


# --- clips -------------------------------------------------------------------------------------------------

_SOUND_SHARE = 0.12  # of a scene's progress bar when its sound is made before the picture
_CONTINUE = ("Continue directly from the starting image with the same people, positions, clothing, "
             "objects, and lighting. Then follow this action. ")  # before the prompt of an anchored scene


def _references(scene: Scene) -> list[str]:
    return [str(path) for name in scene.references if (path := reference_path(name)) is not None]


class _NextStages:
    """The files of the next scene a job announces before each scene (engine_runtime.next_stage) are withdrawn when the
    job ends, however it ends."""

    def __enter__(self) -> "_NextStages":
        return self

    def __exit__(self, *_exc) -> None:
        engine_runtime.next_stage(())


def _first_files(project: Project, number: int | None, members, *, text: bool = False) -> list:
    """The model files scene `number` reads first when it is made, so a job can read them ahead while something else
    computes (its sounds, or the scene before it): its starting frame's (Qwen-Image-Edit) when it starts a shot with
    people or reference pictures, Z-Image's when it draws its starting frame (styles.Style.draws_start), else the
    video experts it starts with. `text`: the video text encoder's files too, in their place (the job's first scene,
    whose load of the encoder encodes the job's prompts). Nothing for an unknown scene; a file that cannot be found is
    only not listed."""
    if number is None or members is None or not _valid_index(project, number):
        return []
    scene, look = project.scenes[number], project.look
    starts_shot = number == 0 or scene.cut
    try:
        frame = []
        if starts_shot and (members or _references(scene) or named_items(scene)):
            frame = engine_runtime.start_frame_files(look.id)
        elif starts_shot and look.draws_start and engine_runtime.zimage_files() is not None:
            frame = engine_runtime.drawn_frame_files(look.id)
        files, loras = engine_runtime.video_experts("i2v" if frame or not starts_shot else "t2v", 1.0, look.id)
        encoder = video_gen.text_encoder_files(files.text_encoder) if text else []
        if frame:  # the video experts are read by the video stage itself, during the frame (after_start_frame)
            return [*frame, *encoder]
        return [*encoder, files.high, *(path for path, _ in (loras or {}).get("high", ())),
                files.low, *(path for path, _ in (loras or {}).get("low", ()))]
    except (FileNotFoundError, ValueError, OSError):
        return []


class _PromptsAhead:
    """Announces the Wan prompts of the scenes a prepare job is about to make (engine_runtime.expect_video_prompts),
    so the text encoder (UMT5, 10.6 GiB) is loaded once for all of them instead of once per scene. Each prompt is
    the one _generate's video_gen.generate call gives Wan (video_gen.scene_wan_prompt; an anchored scene starts with
    _CONTINUE). A scene whose prompt cannot be known ahead (people in reference pictures, whose looks the language
    model describes when the scene runs) is left out and encodes its own. Only for the default renderer and two or
    more prompts; a prompt announced for a scene that ends up reusing its picture costs one extra encoding (well
    under a second) in a load that happens anyway. A context manager: the announcement ends with the job."""

    def __init__(self, project: Project, numbers, casts: dict, style: str, active: bool = True):
        self.project, self.numbers, self.casts, self.style, self.active = project, list(numbers), casts, style, active
        self.announced = False

    def prompts(self) -> list[str]:
        found = []
        for number in self.numbers:
            scene = self.project.scenes[number]
            members = self.casts.get(number)
            if members is None or named_items(scene):  # reference pictures: their looks are described when it runs
                continue
            picture, _ = effective(self.project, scene, members)
            if number > 0 and not scene.cut:
                picture = _CONTINUE + picture
            prompt = video_gen.scene_wan_prompt(picture, self.style, members, exact_cast=scene.cast is not None,
                                                ref_images=_references(scene))
            if prompt:
                found.append(prompt)
        return list(dict.fromkeys(found))

    def __enter__(self) -> "_PromptsAhead":
        if self.active:
            try:
                prompts = self.prompts()
            except Exception as error:  # noqa: BLE001 - only a speed-up: each scene then encodes its own prompt
                print(f"[timeline] The scene prompts could not be worked out ahead: {error}", flush=True)
                prompts = []
            if len(prompts) > 1:
                engine_runtime.expect_video_prompts(prompts)
                self.announced = True
        return self

    def __exit__(self, *_exc) -> None:
        if self.announced:
            engine_runtime.expect_video_prompts(())
            self.announced = False


def _prompts_ahead_scenes(project: Project, keys: list, order: list, casts: dict) -> tuple[list, dict]:
    """The scenes whose Wan prompts a job for single scenes announces (_PromptsAhead), with their people: its own and
    up to PROMPTS_AHEAD_OTHER_SCENES other scenes of the film that still need a clip, the next ones first. When the
    creator then makes them one by one, their prompts were encoded in this job's text-encoder load: the engine keeps
    the encodings between jobs (Engine.free), so their jobs skip the encoder (about 13 s each, 2026-11-15)."""
    numbers, found = list(order), dict(casts)
    try:
        others = [index for index in _stale(project, keys) if index not in found]
    except (ValueError, OSError, MediaError) as error:
        print(f"[timeline] The other scenes' prompts are not encoded ahead: {error}", flush=True)
        return numbers, found
    first = min(order) if order else 0
    for index in sorted(others, key=lambda index: (index < first, index)):
        if len(numbers) >= len(order) + PROMPTS_AHEAD_OTHER_SCENES:
            break
        try:
            found[index] = _members_cached(project, project.scenes[index])
        except ValueError:
            continue  # the scene reports its own problem when it is made
        numbers.append(index)
    return numbers, found


def _generate(project: Project, scene: Scene, members: list[dict], seed: int, style: str,
              folder: Path, cb, anchor: Path | None = None, *, number: int | None = None,
              speech: _SpeechAhead | None = None, edited: bool = False) -> video_gen.VideoOutput:
    """The default scene renderer (prepare_job's generate_scene). With sound, the scene's one sound
    (_scene_sound: its sound preview, or made now into the project's sound folder) is handed to
    video_gen, so the clip, the preview and later takes share it. number: the scene's index; speech:
    the job's shared speech batch. edited: `anchor` is the starting picture of an AI edit (the first frame of the
    scene's own take, perhaps changed): the scene starts from it as an anchored scene starts from the previous
    scene's last frame, and its reference images are not composed into it again (the picture shows them already)."""
    picture, cues = effective(project, scene, members)
    checked = shot_pipeline.enabled(project, scene) and not edited
    extra: dict = {}
    if checked and anchor is not None and scene.shot:
        # an anchored shot starts from the last frame only when that frame passes the shot check (layer 3)
        problem = shot_pipeline.anchor_problem(project, scene, members, Path(anchor), folder)
        if problem:
            print(f"[timeline] Scene {(number or 0) + 1}: the previous last frame failed the shot check ({problem}); "
                  "the shot starts from a new keyframe.", flush=True)
            anchor = None
    prompt = picture
    if anchor is not None:
        prompt = _CONTINUE + prompt
    if checked:
        extra["keyframe_check"] = shot_pipeline.keyframe_checker(project, scene, members, folder)
        extra["keyframe_tries"] = shot_pipeline.settings()["keyframe_tries"]
    if scene.shot:
        extra["wan_text"] = prompt  # the compiled shot is the video model's exact prompt (app/shot_spec.py)
    references = [] if edited else _references(scene)
    if not edited and (named := named_items(scene)):
        prompt, references = _with_named_items(prompt, references, named)
    if anchor is None and (members or references):
        engine_runtime.read_ahead_start_frame(style)  # its models, read from the disk while the sound is made
    prepared_sound, progress = None, cb
    if project.audio and _saved_root(project) is not None:
        index = number if number is not None else _index_of(project, scene)
        prepared_sound = _audition(project, index, scene, members)
        if prepared_sound is None:
            prepared_sound = _scene_sound(project, index, scene, members,
                                          lambda fraction, message: cb(_SOUND_SHARE * fraction, message), speech)

            def progress(fraction, message):
                cb(_SOUND_SHARE + (1 - _SOUND_SHARE) * fraction, message)
    return video_gen.generate(prompt, scene.seconds, project.aspect, seed=seed, sound_on=project.audio,
                              style=style, progress_cb=progress, cast=members, scene_dir=folder,
                              image_path=str(anchor) if anchor is not None else None,
                              exact_cast=scene.cast is not None, ref_images=references,
                              cues=tuple(cues) if project.audio else (), background_text=scene.background,
                              effects=scene.effects, prepared_sound=prepared_sound, **extra)


def _finish_clip(output: video_gen.VideoOutput, project: Project, scene: Scene) -> Path:
    if not output.ok:
        raise ValueError(output.error)
    clip = Path(output.files[0])
    info = media.probe(clip)
    if (info.fps != video_gen.FPS or abs((info.duration or 0.0) - scene.seconds) > 0.002
            or info.has_audio != project.audio):
        raise ValueError("The generated clip does not match the requested duration or audio.")
    if project.audio and (not output.audio_stems or not output.audio_stems[0].is_file()):
        raise ValueError("The scene's background audio is missing.")
    if output.info.get("voices") and (not output.audio_stems or not output.audio_stems[1]
                                      or not output.audio_stems[1].is_file()):
        raise ValueError("The scene's dialogue audio is missing.")
    overlay = getattr(output, "sound_overlay", None)
    if overlay is not None and not Path(overlay).is_file():
        raise ValueError("The scene's sound effects are missing.")
    return clip


def _build_clip(project: Project, number: int, scene: Scene, members: list[dict], previous: dict | None,
                style: str, key: str, cb, candidate: dict | None = None, *,
                generate_scene=None, meta: dict | None = None, speech: _SpeechAhead | None = None,
                start: Path | None = None) -> dict:
    """start: an AI edit's starting picture (_generate's `edited`), used instead of the previous scene's last frame."""
    clip_path, metadata_path = _asset_paths(project, key)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=clip_path.parent) as folder:
        work = Path(folder)
        seed = int(key[:8], 16) % (2 ** 31 - 1)
        last = None
        continuation = previous is not None and not scene.cut
        if candidate is None and start is not None:
            last = start
        elif continuation and candidate is None:
            last = work / "last.png"
            ok, message = ffmpeg_utils.last_frame(previous["path"], last)
            if not ok:
                raise ValueError(message)

        verdict = None
        if candidate is None:
            render_scene = generate_scene or _generate
            options = {}
            if render_scene is _generate:
                options["number"] = number  # a custom renderer keeps the call it always had
                if speech is not None:
                    options["speech"] = speech
                if start is not None:
                    options["edited"] = True
            checked = render_scene is _generate and shot_pipeline.enabled(project, scene)
            retries = shot_pipeline.settings()["retries"] if checked else 0
            for attempt in range(retries + 1):
                folder_name = "initial" if attempt == 0 else f"retry{attempt}"
                output = render_scene(project, scene, members, seed + 104729 * attempt, style, work / folder_name, cb,
                                      anchor=last, **options)
                clip, info, sources = _finish_clip(output, project, scene), output.info, output.audio_stems
                overlay = getattr(output, "sound_overlay", None) if sources is not None else None
                if not checked:
                    break
                cb(0.99, f"Scene {number + 1}: checking the shot…")
                verdict = shot_pipeline.check_clip(project, scene, members, clip, keyframe=info.get("keyframe"),
                                                   work=work / folder_name, attempt=attempt)
                shot_pipeline.keep_sheets(project, number, attempt, work / folder_name)
                legacy = info.get("qc") if isinstance(info.get("qc"), dict) else {}
                if legacy.get("status") == "fail":  # the face drift found between the clip's parts
                    verdict.checks.append(shot_qc.Check("face_drift", shot_qc.FAIL, "; ".join(legacy.get("reasons")
                                                                                            or []), "C"))
                    verdict.status = shot_qc.FAIL
                if verdict.status != shot_qc.FAIL:
                    break
                print(f"[timeline] Scene {number + 1}, try {attempt + 1}: the shot check failed "
                      f"({shot_qc.summary(verdict)}).", flush=True)
                if attempt < retries:
                    cb(0.0, f"Scene {number + 1}: the shot check failed; making it again with a new seed…")
                else:
                    verdict.status = shot_qc.NEEDS_REVIEW  # the owner decides; the episode is not "complete"
        else:
            clip, info, sources = candidate["path"], candidate, candidate["audio_sources"]
            overlay = candidate.get("effects_source") if sources is not None else None
        transition = "first" if number == 0 else "cut"
        note = "First scene." if number == 0 else "Intentional cut."
        if continuation:
            transition, note = "anchored", "Generated from the preceding scene’s last frame."
        cancel.check()
        quality = shot_pipeline.record(verdict, info.get("qc")) if verdict is not None else info["qc"]
        record = {"sha256": _sha(clip), "seconds": scene.seconds, "qc": quality,
                  "width": info["width"], "height": info["height"], "voices": info.get("voices", []),
                  "dialogue_timings": info.get("dialogue_timings", []),
                  "background_model": info.get("background_model"),
                  "characters": [person["slug"] for person in members], "transition": transition,
                  "transition_note": note, "continuity_version": _CONTINUITY_VERSION, **(meta or {})}
        if isinstance(info.get("sound_key"), str) and _HASH.fullmatch(info["sound_key"]):
            record["sound_key"] = info["sound_key"]  # the scene sound the clip was made with
        if isinstance(info.get("lipsync_skipped"), str):
            record["lipsync_skipped"] = info["lipsync_skipped"]  # a stylized face lip sync could not use
        if isinstance(info.get("start_frame_note"), str):
            record["start_frame_note"] = info["start_frame_note"]  # a drawn start frame failed: made from the text
        if info.get("lipsync") == "infinitetalk":
            record["lipsync"] = "infinitetalk"  # the talking picture was made by InfiniteTalk
        record.setdefault("picture_sha256", record["sha256"])
        if clip != clip_path:
            clip.replace(clip_path)
        stored_sources = stored_effects = None
        if sources is not None:
            hashes, stored = {}, []
            for kind, source in zip(("background", "voice"), sources):
                target = _sound_path(clip_path, kind) if source is not None else None
                if source is not None:
                    hashes[kind] = _sha(source)
                    if source != target:
                        source.replace(target)
                stored.append(target)
            if overlay is not None:
                stored_effects = _sound_path(clip_path, "effects")
                hashes["effects"] = _sha(Path(overlay))
                if Path(overlay) != stored_effects:
                    Path(overlay).replace(stored_effects)
            stored_sources = tuple(stored)
            record["sound_hashes"] = hashes
        _write(metadata_path, record)
        return {**record, "path": clip_path, "audio_sources": stored_sources, "effects_source": stored_effects}


def _picture_settings(scene: Scene) -> dict:
    return {name: value for name, value in scene.clip_settings().items() if name not in _SOUND_FIELDS}


def _clip_key(project: Project, settings: dict, members: list[dict], style: str, tier: str, number: int,
              continuity, cache_namespace=None, take=None) -> str:
    values = [project.aspect, _MODE_IN_KEYS, project.audio, style, tier, settings,
              [characters.identity(person) for person in members], number == 0, continuity]
    if project.audio:
        values.extend((_SOUND_CACHE_VERSION, sound.MODEL_ID))
    if cache_namespace is not None:
        values.append(cache_namespace)
    if take:
        values.append(take)
    if named := _item_identities(settings):  # a scene that names no place or item keeps its key
        values.append({"items": named})
    if dialogue.shown(str(settings.get("prompt") or "")):
        values.append(_SHOWN_TEXT)
    return _hash(values)


# A scene whose picture text shows quoted text (dialogue.shown: a screen's "CRYPTO NEWS"): until 2026-11-16 that text
# was spoken as a dialogue line and replaced by "talking" in the picture text, so its clips show Outdated and are made
# again. Every other scene keeps its keys.
_SHOWN_TEXT = {"shown_text": 1}


def _picture_key(project: Project, scene: Scene, members: list[dict], style: str, tier: str, number: int,
                 continuity, cache_namespace=None, take=None) -> str:
    """Identity of a clip's picture: its clip inputs without the background sound and effects."""
    settings = _picture_settings(scene)
    values = ["picture", _PICTURE_VERSION, project.aspect, _MODE_IN_KEYS, project.audio, style, tier, settings,
              [characters.identity(person) for person in members], number == 0, continuity, cache_namespace,
              take or None]
    if named := _item_identities(settings):  # a scene that names no place or item keeps its key
        values.append({"items": named})
    if dialogue.shown(scene.prompt):  # likewise a scene without quoted text that the picture shows
        values.append(_SHOWN_TEXT)
    return _hash(values)


def _picture_source(project: Project, number: int, scene: Scene, members: list[dict], style: str, tier: str,
                    continuity, cache_namespace, picture_key: str, preferred=None) -> dict | None:
    """A verified clip that already shows this scene's picture and whose sound can be made again
    around it: `preferred` (the clip the plan showed before a sound-only edit, a chosen take
    included) when its picture still fits, else the clip of the same scene without background text
    and effects, else the newest clip with the same picture_key."""
    candidates = []
    if _valid_key(preferred) and (record := asset(project, preferred, verify=False)) and record.get("picture_key") \
            and record["picture_key"] == _picture_key(project, scene, members, style, tier, number, continuity,
                                                      cache_namespace, record.get("take")):
        candidates.append(preferred)
    candidates.append(_clip_key(project, _picture_settings(scene), members, style, tier, number, continuity,
                                cache_namespace))
    records = _clip_records(project)
    candidates += [key for key, record in sorted(records.items(), key=lambda item: _created(item[1]), reverse=True)
                   if record.get("picture_key") == picture_key]
    for key in dict.fromkeys(candidates):
        record = asset(project, key)
        if not record or record["seconds"] != scene.seconds:
            continue
        stems = record["audio_sources"]
        if project.audio and record.get("voices") and not (stems and stems[1] is not None):
            continue  # its dialogue exists only inside the mixed soundtrack
        return record
    return None


def _refreshed_background(project: Project, number: int, scene: Scene, members: list[dict], cb):
    """The scene's background for a sound-only refresh (a SceneSound with a background): the current
    sound preview's, else a newly generated background with its effects (no speech)."""
    audition = _audition(project, number, scene, members)
    if audition is not None and audition.background is not None:
        return audition
    inputs = _sound_inputs(project, scene, members, voice=False, index=number)
    folder = _sound_folder(project, inputs.key)
    made = scene_sound.load(folder, inputs.key)
    if made is None:
        _ensure_sound_models(False, lambda fraction, message: cb(0.0, message))
        try:
            made = scene_sound.prepare((), (), inputs.prompt, scene.effects, scene.seconds, inputs.seed, folder, cb,
                                       voice=False, background=True)
        finally:
            _changed()
    if made.background is None:
        raise ValueError("The new background sound is missing.")
    return made


def _refresh_clip(project: Project, number: int, scene: Scene, members: list[dict], previous: dict | None,
                  style: str, key: str, cb, source: dict, meta: dict) -> dict:
    """A clip with the picture of `source` (copied, not encoded again), its dialogue stem and a new
    background with the scene's effects: the sound-only refresh."""
    clip_path, _ = _asset_paths(project, key)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=clip_path.parent) as folder:
        work = Path(folder)
        clip, stems, overlay, sound_key = work / "clip.mp4", None, None, None
        if project.audio:
            cb(0.05, "Reusing the picture, preparing the new background sound…")
            made = _refreshed_background(project, number, scene, members,
                                         lambda fraction, message: cb(0.05 + 0.8 * fraction, message))
            background = work / "background.wav"
            shutil.copyfile(made.background, background)
            if made.overlay is not None:
                overlay = work / "effects.wav"
                shutil.copyfile(made.overlay, overlay)
            voice = None
            if source["audio_sources"] and source["audio_sources"][1] is not None:
                voice = work / "voice.wav"
                shutil.copyfile(source["audio_sources"][1], voice)
            # The clip's sound is the preview's when it also speaks exactly the preview's lines.
            if (made.voice is None) == (voice is None) and (voice is None or _sha(Path(made.voice)) == _sha(voice)):
                sound_key = made.key
            cb(0.9, "Mixing the scene sound…")
            # Mixed exactly as video_gen mixes a new clip, so a refreshed clip plays as loud as a generated one.
            scene_sound.clip_sound(source["path"], background, voice, clip)
            stems = background, voice
        else:
            cb(0.5, "Reusing the picture…")
            shutil.copyfile(source["path"], clip)
        candidate = {name: source[name] for name in ("qc", "width", "height", "voices", "dialogue_timings")
                     if name in source}
        candidate.update(path=clip, audio_sources=stems, effects_source=overlay,
                         background_model=sound.MODEL_ID if project.audio else source.get("background_model"))
        if sound_key:
            candidate["sound_key"] = sound_key
        meta = {**meta, "picture_sha256": _picture_sha(source),
                "picture_key": source.get("picture_key", meta.get("picture_key")),
                "anchor_sha256": source.get("anchor_sha256", meta.get("anchor_sha256")), "take": source.get("take"),
                "created": time.time()}
        return _build_clip(project, number, scene, members, previous, style, key, cb, candidate, meta=meta)


def _current(params: dict) -> Project:
    project = load(params["project"])
    if project.key != params["key"]:
        raise ValueError("This project was edited. Run the action again on the current version.")
    return project


def _prepare(project: Project, params: dict, cb, *, indices=None, generate_scene=None,
             cache_namespace=None) -> tuple[bool, list, str, int | None, bool]:
    """Prepare the scenes at `indices`: each one's clip key is computed with its continuity and a clip
    that is not stored yet is made. Without indices, every scene whose plan clip is missing or no
    longer fits (_outdated): the scenes are walked in order and each is looked at again once the
    scenes before it are done, so a chosen take that still fits is kept.
    Returns (ok, notes, error, index of the failed scene, retake): retake is True when the scene failed
    while a new picture was being made, so its next attempt should use a new take (seed). Scenes made one
    after another in an anchored chain synthesize their lines in one shared batch (_SpeechAhead)."""
    state = prepared(project)
    keys = list(state["assets"])
    walk = indices is None
    order = list(range(len(project.scenes))) if walk else list(indices)
    try:
        casts = _validate_generation(project, order) if order else {}
    except ValueError as error:
        return False, [], str(error), getattr(error, "scene", None), False
    planned = _stale(project, keys, casts=casts) if walk else order
    ahead, ahead_casts = (planned, casts) if walk else _prompts_ahead_scenes(project, keys, order, casts)
    total = max(len(planned), 1)
    sources = state.get("sound_sources") if isinstance(state.get("sound_sources"), list) else []
    style, tier = project.video_style, model_manager.tier()
    take = params.get("take")
    done = 0
    scene_notes: list[str] = []  # what the job log should say about single scenes (a skipped lip sync)
    with _SpeechAhead(project, order, casts) as speech, \
            _PromptsAhead(project, ahead, ahead_casts, style, active=generate_scene is None), _NextStages():
        # the sounds of the scenes still to make, a window of them at a time, before their pictures (_sounds_ahead)
        windows = {} if generate_scene is not None else {
            planned[start]: planned[start:start + SOUNDS_AHEAD_SCENES]
            for start in range(0, len(planned), SOUNDS_AHEAD_SCENES)}
        for number in order:
            scene = project.scenes[number]
            if number in windows:  # the scene's first models are read after the sound models, while the sounds run
                first = _first_files(project, number, casts.get(number), text=True) if queue_manager.in_worker() else ()
                _sounds_ahead(project, windows[number], casts,
                              lambda fraction, message, number=number:
                              cb(min(done / total, 1.0), f"Scene {number + 1}/{len(project.scenes)} · {message}"),
                              then=first)
            if generate_scene is None and queue_manager.in_worker():  # read by this scene's last picture stage
                following = next((later for later in planned if later > number), None)
                engine_runtime.next_stage(_first_files(project, following, casts.get(following)))
            if walk:
                record = asset(project, keys[number]) if number < len(keys) and keys[number] else None
                before = (asset(project, keys[number - 1], verify=False)
                          if 0 < number <= len(keys) and keys[number - 1] else None)
                if not _outdated(project, number, record, before, casts[number]):
                    continue

            def progress(fraction, message, step=min(done, total - 1), number=number):
                cb(min((step + fraction) / total, 1.0), f"Scene {number + 1}/{len(project.scenes)} · {message}")

            _current(params)
            progress(0, "Checking saved clip…")
            making_picture = False
            try:
                previous = _anchor(project, number, keys)
                members = casts[number]
                continuity = [_picture_sha(previous), project.scenes[number - 1].prompt] if previous else None
                # After a failed generation of this scene the next attempt is a new take (new seed); otherwise
                # a take the plan already shows for these inputs is kept (also when every key is re-evaluated).
                scene_take = (take or _retake(project, number)
                              or _kept_take(project, number, keys, scene, members, style, tier, continuity,
                                            cache_namespace))
                key = _clip_key(project, scene.clip_settings(), members, style, tier, number, continuity,
                                cache_namespace, scene_take)
                cached = asset(project, key)
                if not cached or (continuity and cached.get("continuity_version") != _CONTINUITY_VERSION):
                    meta = {"scene_signature": _scene_signature(project, scene, members),
                            "picture_key": _picture_key(project, scene, members, style, tier, number, continuity,
                                                        cache_namespace, scene_take),
                            "anchor_sha256": continuity[0] if continuity else None, "take": scene_take or None,
                            "created": time.time(), "style": style, "tier": tier}
                    if cache_namespace is not None:
                        meta["cache_namespace"] = cache_namespace
                    if cached:
                        meta.update({name: cached[name] for name in ("created", "take", "picture_sha256", "anchor_sha256",
                                                                     "picture_key", "style", "tier", "cache_namespace")
                                     if name in cached})
                        _build_clip(project, number, scene, members, previous, style, key, progress, cached, meta=meta)
                    elif not scene_take and (source := _picture_source(
                            project, number, scene, members, style, tier, continuity, cache_namespace, meta["picture_key"],
                            preferred=sources[number] if number < len(sources) else None)):
                        _refresh_clip(project, number, scene, members, previous, style, key, progress, source, meta)
                    else:
                        making_picture = True
                        built = _build_clip(project, number, scene, members, previous, style, key, progress,
                                            generate_scene=generate_scene, meta=meta, speech=speech)
                        for note in (built.get("lipsync_skipped"), built.get("start_frame_note")):
                            if note:
                                scene_notes.append(f"Scene {number + 1}: {note}")
                                print(f"[timeline] Scene {number + 1}: {note}", flush=True)
            except llm.VisualReviewUnavailable as error:
                return False, [REVIEW_UNAVAILABLE], f"Scene {number + 1}: {error}", number, False
            except (ValueError, OSError, MediaError) as error:
                return False, [], f"Scene {number + 1}: {_explain(error)}", number, making_picture
            done += 1
            keys.extend([None] * (number + 1 - len(keys)))
            keys[number] = key
            with _LOCK:
                cancel.check()
                _current(params)
                if keys != state["assets"]:
                    update = {**state, "assets": list(keys), "export_assets": state.get("export_assets", state["assets"])}
                    try:
                        _write_plan(project, update)
                    except OSError as error:
                        # The clip is stored under its key, so running the action again only writes the plan.
                        print(f"[timeline] Could not save the plan: {error}", flush=True)
                        return False, [], PLAN_NOT_SAVED, number, False
                    state = update
    count = sum(bool(key) for key in keys)
    cb(1, f"{count}/{len(project.scenes)} clips ready.")
    return True, [f"{count} clips ready on the timeline.", *scene_notes], "", None, False


def _kept_take(project: Project, number: int, keys: list, scene: Scene, members, style: str, tier: str,
               continuity, cache_namespace) -> str | None:
    """The take of the clip the plan shows for scene `number` when that clip still matches its inputs."""
    if number >= len(keys) or not keys[number]:
        return None
    record = asset(project, keys[number], verify=False)
    take = record.get("take") if record else None
    if (isinstance(take, str) and take
            and _clip_key(project, scene.clip_settings(), members, style, tier, number, continuity,
                          cache_namespace, take) == keys[number]):
        return take
    return None


def prepare_job(params: dict, cb, *, generate_scene=None, cache_namespace=None) -> tuple:
    """Generate one scene (params["scene_index"]), or every scene whose clip is missing or no longer
    fits. With a cache_namespace (production scripts) every scene's key is evaluated as it always was:
    a scene whose key has no stored clip is made again."""
    project = _current(params)
    single = "scene_index" in params
    if single:
        indices = [params["scene_index"]]
    else:
        indices = range(len(project.scenes)) if cache_namespace is not None else None
    ok, notes, error, failed, retake = _prepare(project, params, cb, indices=indices,
                                                generate_scene=generate_scene, cache_namespace=cache_namespace)
    if not ok:
        _record_failure(project, PREPARE, failed, error, retake=retake)
        return False, [], notes, error
    _clear_failure(project, PREPARE, params["scene_index"] if single else None)
    return True, [], notes, ""


def sound_job(params: dict, cb) -> tuple:
    """Prepare the sound preview of one scene in <project>/sound/<sound key>/ (kept and reused)."""
    project = _current(params)
    index = params.get("scene_index")
    if not _valid_index(project, index):
        return False, [], [], "Choose a scene on the timeline."
    scene = project.scenes[index]
    try:
        if not project.audio:
            raise ValueError("Turn on sound for this project to prepare a sound preview.")
        members = _validate_generation(project, [index])[index]
        inputs = _sound_inputs(project, scene, members, index=index)
        folder = _sound_folder(project, inputs.key)
        cb(0, f"Scene {index + 1} · checking the sound preview…")
        if scene_sound.load(folder, inputs.key) is None:
            _ensure_sound_models(inputs.talk, lambda fraction, message: cb(0.0, f"Scene {index + 1} · {message}"))
            try:
                scene_sound.prepare(inputs.cues, inputs.speakers, inputs.prompt, scene.effects, scene.seconds,
                                    inputs.seed, folder,
                                    lambda fraction, message: cb(fraction, f"Scene {index + 1} · {message}"),
                                    voice=inputs.talk, background=True)
            finally:
                _changed()
        else:
            queue_manager.reused()  # nothing was made: the queue does not learn its speed from this job
    except _SceneError as error:
        _record_failure(project, SOUND, index, str(error))
        return False, [], [], str(error)
    except (ValueError, OSError, MediaError) as error:
        message = f"Scene {index + 1}: {error}"
        _record_failure(project, SOUND, index, message)
        return False, [], [], message
    _clear_failure(project, SOUND, index)
    cb(1, f"Scene {index + 1} sound preview ready.")
    return True, [], [f"Scene {index + 1} sound preview ready."], ""


def merged_clip(project: Project, state: dict, *, verify: bool = False) -> Path | None:
    record = state.get("merged")
    if not isinstance(record, dict) or record.get("assets") != state["assets"]:
        return None
    name = record.get("file")
    if not isinstance(name, str) or not re.fullmatch(r"[0-9a-f]{32}\.mp4", name):
        return None
    try:
        path = _inside(_root(project.id), "edits", name)
        return path if path.is_file() and (not verify or _sha(path) == record["sha256"]) else None
    except (OSError, KeyError, ValueError):
        return None


def _review_join(frames: list[Path]) -> dict:
    prompt = (
        "These ordered frames show the last frame of a clip, the first frame of the next clip, "
        "and a frame shortly after that. They are meant to continue the same action. Compare only "
        "visible continuity at this join: identity, clothing, people and object positions, "
        "location and lighting. Ignore artistic quality. Return 'match' if the continuation is "
        "plausible, 'bridge' if a brief interpolation could connect the same subjects and framing "
        "through a small jump, or 'mismatch' if identity, staging or setting changes too much for "
        "interpolation. Give a short, specific reason in English. Never recommend interpolation "
        "for different people, unrelated shots or major reframing."
    )

    def valid(result):
        return (result.get("verdict") in ("match", "bridge", "mismatch")
                and isinstance(result.get("reason"), str) and bool(result["reason"].strip()))

    return llm.inspect_review([str(path) for path in frames], prompt, 170, _JOIN_SCHEMA, valid, (512, 768))


def _joins_path(project: Project) -> Path:
    return _inside(_root(project.id), "edits", "joins.json")


def _joins(project: Project) -> dict:
    """Remembered join verdicts by (previous picture, next picture, bridge frames, version)."""
    try:
        data = _read(_joins_path(project))
        entries = data["joins"] if isinstance(data, dict) and data.get("version") == _JOIN_VERSION else {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    if not isinstance(entries, dict):
        return {}
    return {key: entry for key, entry in entries.items()
            if _HASH.fullmatch(str(key)) and isinstance(entry, dict)
            and entry.get("verdict") in ("match", "bridge", "mismatch")
            and isinstance(entry.get("reason"), str) and entry["reason"].strip()}


def _save_joins(project: Project, entries: dict) -> None:
    if len(entries) > _KEEP_JOINS:
        ordered = sorted(entries.items(), key=lambda item: item[1].get("time", 0), reverse=True)
        for _, entry in ordered[_KEEP_JOINS:]:
            if path := _stored_bridge(project, entry, verify=False):
                path.unlink(missing_ok=True)
        entries = dict(ordered[:_KEEP_JOINS])
    try:
        _write(_joins_path(project), {"version": _JOIN_VERSION, "joins": entries})
    except OSError as error:
        print(f"[timeline] Could not save the join checks: {error}", flush=True)


def _stored_bridge(project: Project, entry: dict, *, verify: bool = True) -> Path | None:
    stored = entry.get("bridge")
    if not isinstance(stored, dict) or not isinstance(stored.get("file"), str) or not _BRIDGE.fullmatch(stored["file"]):
        return None
    try:
        path = _inside(_root(project.id), "edits", "bridges", stored["file"])
        return path if path.is_file() and (not verify or _sha(path) == stored.get("sha256")) else None
    except (OSError, ValueError):
        return None


def _store_bridge(project: Project, key: str, bridge: Path) -> dict | None:
    """Keep a smoothed transition for the next join of the same two pictures."""
    temporary = None
    try:
        folder = _inside(_root(project.id), "edits", "bridges")
        folder.mkdir(parents=True, exist_ok=True)
        target = _inside(folder, key + ".mp4")
        temporary = _inside(folder, f".{uuid.uuid4().hex}.mp4")
        shutil.copyfile(bridge, temporary)
        os.replace(temporary, target)
        return {"file": target.name, "sha256": _sha(target)}
    except (OSError, ValueError) as error:
        print(f"[timeline] Could not keep the transition: {error}", flush=True)
        return None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _join_frames(previous: dict, record: dict, frames: int, first: Path, start: Path, end: Path) -> None:
    media.last_frame(previous["path"], first)
    media.frame_at(record["path"], 0, start)
    media.frame_at(record["path"], max(1, frames) / video_gen.FPS, end)


def _review_boundaries(project: Project, records: list[dict], folder: Path, cb):
    """Check every join that is not an intentional cut. Returns (repairs by scene index, notes, the joins
    to review, the joins among them that may turn out otherwise on another try: the review or the
    smoothing could not finish - a remembered verdict gives the same answer again)."""
    repairs, notes, attention, retry = {}, [], [], []
    joins = len(records) - 1
    remembered = _joins(project)
    changed = False
    for index in range(1, len(records)):
        label = f"Scenes {index} → {index + 1}"
        cb(0.7 * (index - 1) / joins, f"Checking {label.lower()}…")
        if project.scenes[index].cut:
            notes.append(f"{label} · intentional cut.")
            continue
        record = records[index]
        timings = record.get("dialogue_timings") or []
        first_word = min((float(line["start"]) for line in timings), default=1.0)
        frames = min(_BRIDGE_FRAMES, int(first_word * video_gen.FPS))
        first, start, end = (folder / f"{index}-{suffix}.png" for suffix in ("before", "start", "after"))
        join = _hash([_JOIN_VERSION, _picture_sha(records[index - 1]), _picture_sha(record), frames])
        entry = remembered.get(join)
        if entry is None:
            _join_frames(records[index - 1], record, frames, first, start, end)
            try:
                verdict = _review_join([first, start, end])
            except llm.VisualReviewUnavailable:
                attention.append(index)
                retry.append(index)
                notes.append(f"{label} · the model could not check this join. Original frames kept.")
                continue
            entry = {"verdict": verdict["verdict"], "reason": " ".join(verdict["reason"].split())[:280],
                     "time": time.time()}
            remembered[join] = entry
            changed = True
        reason = entry["reason"]
        if entry["verdict"] == "mismatch":
            attention.append(index)
            notes.append(f"{label} · {reason} Review the join or regenerate scene {index + 1}.")
            continue
        if entry["verdict"] == "match":
            notes.append(f"{label} · {reason}")
            continue
        if frames < 3 or record.get("voices") and not timings:
            attention.append(index)
            notes.append(f"{label} · {reason} Original frames kept to preserve dialogue sync.")
            continue
        bridge = _stored_bridge(project, entry)
        if bridge is None:
            cb(0.7 * (index - 1) / joins, f"Smoothing {label.lower()}…")
            if not first.is_file():
                _join_frames(records[index - 1], record, frames, first, start, end)
            bridge = folder / f"{index}-bridge.mp4"
            try:
                engine_runtime.transition_frames(first, end, bridge, frames, video_gen.FPS)
            except (OSError, RuntimeError, ValueError) as error:
                notes.append(f"{label} · smoothing could not finish. {str(error)[:180]}")
                attention.append(index)
                retry.append(index)
                continue
            if (stored := _store_bridge(project, join, bridge)) is not None:
                entry["bridge"] = stored
                changed = True
        repairs[index] = bridge
        notes.append(f"{label} · opening frames smoothed. {reason}")
    if changed:
        _save_joins(project, remembered)
    return repairs, notes, attention, retry


def _ready_records(project: Project, state: dict) -> list[dict]:
    if not all_clips(project, state["assets"]):
        raise ValueError("Generate all clips on the timeline first.")
    records = []
    for index, (key, scene) in enumerate(zip(state["assets"], project.scenes), 1):
        record = asset(project, key)
        if not record or record["seconds"] != scene.seconds:
            raise ValueError(f"Scene {index} is missing or has changed. Generate it again.")
        records.append(record)
    return records


def _levels(project: Project) -> list[tuple[float, float]] | None:
    """(voice dB, background dB) per scene, or None when every scene plays at 0 dB / 0 dB."""
    levels = [(scene.voice_db, scene.background_db) for scene in project.scenes]
    return levels if any(voice or background for voice, background in levels) else None


def episode_mix_plan(project: Project) -> dict | None:
    """The episode mix of a series episode (engine/media/episode_mix.py: one narration track, ambience per place,
    music from the owner's library, the -14 LUFS master), or None for a film outside a series, which keeps the
    scene-by-scene mix it always had. The location of a scene is its shot's place, else its background description
    (scenes with the same ambience in a row share one bed)."""
    if not project.audio or not (any(scene.shot for scene in project.scenes)
                                 or shot_pipeline.bible_for(project) is not None):
        return None
    from engine.media import episode_mix

    locations = []
    for scene in project.scenes:
        spec = scene.spec if scene.shot else None
        locations.append(f"place:{spec.location.lower()}" if spec else " ".join(scene.background.lower().split()))
    return {"locations": locations, "music": music_library.pick(project.id), "report": episode_mix.MixReport()}


def _assemble(records: list[dict], out: Path, cb, *, transitions=None, background_fallback=None,
              on_silent_background=None, levels=None, episode=None) -> None:
    stems = [record["audio_sources"] for record in records]
    if any(stems) and not all(stems) and on_silent_background is None:
        raise ValueError("Prepare the missing scene audio before rendering.")
    # (background, voice, effects overlay): the export levels each room tone without its effects
    sound = [(*record["audio_sources"], record.get("effects_source")) for record in records] if all(stems) else None
    media.join_timeline([(record["path"], record["seconds"]) for record in records], out, video_gen.FPS, cb,
                        sound_stems=sound, transitions=transitions,
                        background_fallback=background_fallback, on_silent_background=on_silent_background,
                        sound_levels=levels, **({"episode": episode} if episode is not None and sound else {}))


def _merge(project: Project, params: dict, cb) -> tuple[list[str], list[int]]:
    """Join the prepared clips, check every join and keep the result as the project's joined edit."""
    state = prepared(project)
    records = _ready_records(project, state)
    if merged_clip(project, state, verify=True):
        merged = state["merged"]
        attention = merged.get("attention") or []
        # Joins to review whose verdict is remembered come out the same: the edit is joined again only
        # for those that may turn out otherwise (an older edit without that list: for any join to review).
        retry = merged.get("retry", attention)
        if not retry:
            queue_manager.reused()
            cb(1, f"Joined timeline · {len(attention)} joins to review." if attention
               else "Joined timeline ready to render.")
            return merged.get("notes", []), list(attention)
    folder = _inside(_root(project.id), "edits")
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as temporary:
        work = Path(temporary)
        repairs, notes, attention, retry = _review_boundaries(project, records, work, cb)
        stems = [bool(record["audio_sources"]) for record in records]
        if any(stems) and not all(stems):
            notes.append("Audio · some clips have mixed audio only. The original clip soundtracks were used.")
        temp = work / "joined.mp4"
        mix = episode_mix_plan(project)
        if mix is not None:
            notes.append("Audio · episode mix: one narration track, ambience per place, -14 LUFS master.")
        _assemble(records, temp, lambda fraction, desc: cb(0.7 + 0.3 * fraction, desc), transitions=repairs,
                  **({"episode": mix} if mix is not None else {}),
                  on_silent_background=lambda number: notes.append(
                      f"Scene {number} · no audible background. The scene's dialogue track was preserved."),
                  levels=_levels(project))
        final = _inside(folder, uuid.uuid4().hex + ".mp4")
        digest = _sha(temp)
        with _LOCK:
            cancel.check()
            _current(params)
            if prepared(project)["assets"] != state["assets"]:
                raise ValueError("The clips changed during the join. Join and check again.")
            _patiently(lambda: temp.replace(final))
            try:
                merged = {"file": final.name, "sha256": digest, "assets": state["assets"],
                          "notes": notes, "attention": attention, "retry": retry}
                if mix is not None:
                    merged["audio_mix"] = mix["report"].to_json()
                _write_plan(project, {**state, "merged": merged})
            except OSError as error:
                final.unlink(missing_ok=True)
                print(f"[timeline] Could not save the plan: {error}", flush=True)
                raise OSError("Could not save the timeline plan. Join and check again.") from error
    cb(1, f"Joined timeline · {len(attention)} joins to review." if attention else "Joined timeline ready to render.")
    return notes, attention


def merge_job(params: dict, cb) -> tuple:
    project = _current(params)
    try:
        notes, _ = _merge(project, params, cb)
    except Exception as error:
        _record_failure(project, MERGE, None, str(error))
        raise
    _clear_failure(project, MERGE)
    return True, [], notes, ""


def subtitle_cues(project: Project, records: list[dict]) -> list[tuple[float, float, str]]:
    """Every spoken line of the movie as (start s, end s, text) on the movie's clock: the scene's own line times
    (its clip's dialogue_timings) moved by the length of the scenes before it, cut at the end of the scene."""
    cues, offset = [], 0.0
    for scene, record in zip(project.scenes, records):
        for line in record.get("dialogue_timings") or []:
            try:
                text, start, end = " ".join(str(line["text"]).split()), float(line["start"]), float(line["end"])
            except (KeyError, TypeError, ValueError):
                continue
            end = min(end, float(scene.seconds))
            if text and 0.0 <= start < end:
                cues.append((offset + start, offset + end, text))
        offset += scene.seconds
    return cues


NO_SUBTITLES_SILENT = ("Subtitles need Audio on: a line is shown while it is spoken, and this project has no "
                       "speech. The movie was exported without subtitles.")
NO_SUBTITLES = "Subtitles are on, but no line is spoken in the movie. The movie was exported without subtitles."


def _burn(project: Project, records: list[dict], movie: Path, cb) -> tuple[Path, str]:
    """(the movie with its dialogue burned in, a new file beside it; or the movie itself without any line, with a
    note why)."""
    cues = subtitle_cues(project, records)
    if not cues:
        return movie, NO_SUBTITLES if project.audio else NO_SUBTITLES_SILENT
    cb(0.9, "Burning the subtitles…")
    burned = movie.with_name("subtitled.mp4")
    ok, message = ffmpeg_utils.burn_subtitles(movie, cues, burned)
    if not ok:
        raise MediaError(f"Could not burn the subtitles: {message}")
    return burned, ""


def _overlay(project: Project, movie: Path, cb) -> tuple[Path, list[str]]:
    """(the movie with the project's exact text burned in, a new file beside it; or the movie itself when no overlay is
    shown, with the notes: an overlay that starts after the movie's end, a text made smaller to fit)."""
    length = ffmpeg_utils.probe_duration(movie) or project.seconds
    rows, notes = text_overlays.layers(project.overlays, length)
    if not rows:
        return movie, notes
    cb(0.95, "Adding the text…")
    burned = movie.with_name("overlaid.mp4")
    ok, message = ffmpeg_utils.burn_overlays(movie, rows, burned, report=notes)
    if not ok:
        raise MediaError(f"Could not add the text to the movie: {message}")
    return burned, notes


def _render(project: Project, params: dict, cb, *, background_fallback=None,
            require_merge: bool = False) -> tuple[Path, list[str]]:
    """Write the movie to the outputs folder and the Gallery history; returns the file and notes for the job."""
    state = prepared(project)
    try:
        records = _ready_records(project, state)
    except ValueError as error:
        raise _Refused(str(error)) from error
    joined = merged_clip(project, state, verify=True)
    if require_merge and not joined:
        raise _Refused("Join and check the current timeline before rendering.")
    if existing := exported(project, state):
        queue_manager.reused()
        return existing, []

    notes = []
    output_dir = config.OUTPUTS_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = characters.slugify(project.title) or "movie"
    final = output_dir / f"timeline_{name}_{uuid.uuid4().hex[:8]}.mp4"
    checked = shot_pipeline.bible_for(project) is not None or any(scene.shot for scene in project.scenes)
    mix_report = (state.get("merged") or {}).get("audio_mix") if joined else None
    titles: list = []
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".timeline-") as folder:
        temp = Path(folder) / "render.mp4"
        if joined:
            cb(0.6, "Writing the joined edit to MP4…")
            media.remux(joined, temp, comment=media.AI_COMMENT)
        else:
            mix = episode_mix_plan(project)
            _assemble(records, temp, cb, background_fallback=background_fallback, levels=_levels(project),
                      **({"episode": mix} if mix is not None else {}))
            if mix is not None:
                mix_report = mix["report"].to_json()
        if project.subtitles:
            temp, note = _burn(project, records, temp, cb)
            notes += [note] if note else []
        if project.overlays:
            shown = project
            if checked:  # a series episode's titles go where they cover no character (defect K)
                shown, titles, more = _safe_titles(project, temp)
                notes += more
            temp, more = _overlay(shown, temp, cb)
            notes += more
        cancel.check()
        with _LOCK:
            _current(params)
            latest = prepared(project)
            if latest["assets"] != state["assets"] or latest.get("merged") != state.get("merged"):
                raise _Refused("The prepared clips changed. Render the current timeline again.")
            _patiently(lambda: temp.replace(final))
            try:
                history.record({"mode": "video_timeline", "positive": project.title, "project": project.id,
                                "width": records[0]["width"], "height": records[0]["height"],
                                "seconds": project.seconds, "fps": video_gen.FPS, "shots": len(records),
                                "cast": list(dict.fromkeys(
                                    slug for record in records for slug in record["characters"])),
                                "voices": list(dict.fromkeys(voice for record in records for voice in record["voices"])),
                                "background_model": records[0].get("background_model"),
                                **({"style": project.style} if project.style != styles.DEFAULT else {}),
                                "files": [str(final)]})
                plan = {**state, "export": final.name, "export_revision": project.revision,
                        "export_assets": state["assets"], "export_merge": joined.name if joined else None}
                plan.pop("export_subtitles", None)
                if project.subtitles:
                    plan["export_subtitles"] = True
                plan.pop("export_overlays", None)
                if project.overlays:
                    plan["export_overlays"] = text_overlays.key(project.overlays)
                _write_plan(project, plan)
            except OSError as error:  # the Gallery entry or the plan
                final.unlink(missing_ok=True)
                history.forget([str(final)])
                print(f"[timeline] Could not save the export: {error}", flush=True)
                raise OSError("Could not save the export. Export the movie again.") from error
            except Exception:
                final.unlink(missing_ok=True)
                history.forget([str(final)])
                raise
    if checked:
        try:
            cb(0.98, "Writing the QC report…")
            report = qc_report.build(project, records, final, mix=mix_report, titles=titles)
            qc_report.write(report, final, records)
            notes.append(f"QC report: {report['status']} ({len(report['failing_shots'])} shots and "
                         f"{len(report['failing_sound'])} sound or title checks not passed) — "
                         f"{qc_report.report_paths(final)[1].name}")
        except Exception as error:  # noqa: BLE001 - the film stays; the gate sees the missing report
            print(f"[timeline] The QC report could not be written: {error}", flush=True)
            notes.append(f"The QC report could not be written: {str(error)[:200]}")
    cb(1, "Movie ready to download.")
    return final, notes


def _safe_titles(project: Project, movie: Path) -> tuple:
    """(the project with each centred title moved to the band where it covers no character, the measured covers,
    notes). A title already at the top or bottom stays there; its cover is measured all the same."""
    bible = shot_pipeline.bible_for(project)
    try:
        members = _profiles(list(project.cast_slugs))
    except ValueError:
        members = []
    cast = shot_pipeline.cast_specs(bible, members)
    moved, titles, notes = [], [], []
    for overlay in project.overlays:
        span = text_overlays.window(overlay, project.seconds)
        if span is None or overlay.backdrop == "card":
            moved.append(overlay)
            continue
        position = overlay.position
        if position == "center":
            position, covers = qc_report.safe_title_position(movie, span[0], span[1], cast)
            notes.append(f"Title “{overlay.text.splitlines()[0][:40]}” placed at the {position} (characters cover "
                         f"{covers[position]:.0%} there).")
        cover = qc_report.title_cover(movie, span[0], span[1], cast, position)
        titles.append({"text": overlay.text, "position": position, "cover": round(cover, 3)})
        moved.append(replace(overlay, position=position))
    return replace(project, overlays=tuple(moved)), titles, notes


def render_job(params: dict, cb, *, background_fallback=None) -> tuple:
    project = _current(params)
    try:
        final, notes = _render(project, params, cb, background_fallback=background_fallback,
                               require_merge=bool(params.get("require_merge")))
    except _Refused as error:
        _record_failure(project, RENDER, None, str(error))
        return False, [], [], str(error)
    except Exception as error:
        _record_failure(project, RENDER, None, str(error))
        raise
    _clear_failure(project, RENDER)
    return True, [str(final)], notes, ""


def produce_job(params: dict, cb, *, background_fallback=None) -> tuple:
    """Generate every missing clip, join and check the edit, then export the movie to the Gallery."""
    project = _current(params)
    state = prepared(project)
    if existing := exported(project, state):
        queue_manager.reused()
        cb(1, "Movie ready to download.")
        return True, [str(existing)], [], ""
    try:
        missing = _stale(project, state["assets"], casts=_validate_generation(project, range(len(project.scenes))))
    except ValueError as error:
        _record_failure(project, PRODUCE, getattr(error, "scene", None), str(error))
        return False, [], [], str(error)
    weights = [queue_manager.rate(PREPARE) * sum(project.scenes[index].seconds for index in missing),
               queue_manager.rate(MERGE) * project.seconds, queue_manager.rate(RENDER) * project.seconds]
    total = sum(weights) or 1.0
    starts = [sum(weights[:step]) / total for step in range(3)]

    def stage(step):
        return lambda fraction, message: cb(starts[step] + weights[step] / total * min(max(fraction, 0.0), 1.0),
                                            message)

    notes = []
    if missing:
        ok, notes, error, failed, retake = _prepare(project, params, stage(0))
        if not ok:
            _record_failure(project, PRODUCE, failed, error, retake=retake)
            return False, [], notes, error
    try:
        joined_notes, _ = _merge(project, params, stage(1))
        final, export_notes = _render(project, params, stage(2), background_fallback=background_fallback,
                                      require_merge=True)
    except (ValueError, OSError, MediaError) as error:
        _record_failure(project, PRODUCE, None, str(error))
        return False, [], notes, str(error)
    _clear_failure(project, PRODUCE)
    cb(1, "Movie ready to download.")
    return True, [str(final)], [*notes, *joined_notes, *export_notes], ""


# --- AI edits of a made scene ----------------------------------------------------------------------------------

EDIT_SCOPES = ("scene", "after", "film")  # the edited scene alone; it and the scenes after it; the whole film
MAX_CHANGE = 500  # characters of an edit's picture change
_EDIT_FRAME_SHARE = 0.15  # of a scene's progress bar for its starting frame, when the edit changes the picture
# Asked of Qwen-Image-Edit after the picture change: whatever the change does not name stays as the take shows it.
_KEEP_REST = ("Keep everything else exactly as it is in the picture: the same characters with the same faces, bodies, "
              "clothing, poses and expressions, the same place and objects, the same camera angle, framing and light.")
_RETRY_FIELDS = ("scene_index", "change", "scope", "sources", "changed", "take", "retakes", "seconds")


def _edit_source(project: Project, index: int, keys: list | None = None) -> str | None:
    """The take an AI edit of scene `index` starts from: the clip the plan shows for it, else (an edit saved the
    scene's text and its clip is not made yet: it waits in the queue, or it failed) the newest take that version of
    the scene was edited from."""
    keys = prepared(project)["assets"] if keys is None else keys
    if index < len(keys) and _valid_key(keys[index]) and asset(project, keys[index], verify=False):
        return keys[index]
    try:
        members = _members_cached(project, project.scenes[index])
    except ValueError:
        return None
    entry = _versions(project).get(_scene_signature(project, project.scenes[index], members)) or {}
    for key, _ in sorted(entry.get("bases", {}).items(), key=lambda item: item[1]["time"], reverse=True):
        if asset(project, key, verify=False):
            return key
    return None


def edit_source(project: Project, index: int) -> str | None:
    """The key of the take an AI edit of scene `index` of the saved project would start from; None when the scene
    has no clip to start from (it must be generated first)."""
    if not _valid_index(project, index) or _saved_root(project) is None:
        return None
    return _edit_source(project, index)


def _shown_changes(before: str, after: str) -> list[tuple[str, str]]:
    """The texts the picture shows (dialogue.shown) that an edit of a picture text changed: (old, new) of each that
    differs, when the edit kept their number."""
    old, new = dialogue.shown(before), dialogue.shown(after)
    return [(a, b) for a, b in zip(old, new) if a != b] if len(old) == len(new) else []


def _edit_steps(project: Project, index: int, scope: str, change: str, changed, sources: dict) -> list[tuple[int, str]]:
    """The scenes an AI edit makes, in order, and how: "edit" (again from its own take in `sources`, with the picture
    change) or "follow" (made again from the new last frame of the scene before it, as any anchored scene is). The
    edit makes its own scene; with scope "after" or "film" also the scenes of the scope whose text it changed and, with
    a picture change, the ones that start a shot; and after a scene it made, every scene that continues from it. A
    scene without a clip ends such a run: an edit makes no clip that was not there."""
    first = 0 if scope == "film" else index
    last = index if scope == "scene" else len(project.scenes) - 1
    targets = {index} | {number for number in changed if first <= number <= last}
    if change and scope != "scene":
        targets |= {number for number in range(first, last + 1) if number == 0 or project.scenes[number].cut}
    steps, made = [], False
    for number in range(min(targets), last + 1):
        if number not in sources:
            made = False
        elif made and number > 0 and not project.scenes[number].cut:
            steps.append((number, "follow"))
        elif number in targets:
            steps.append((number, "edit"))
            made = True
        else:
            made = False
    return steps


def edit(saved: Project, index: int, scene: Scene, *, change: str = "",
         scope: str = "scene") -> tuple[Project, queue_manager.Job]:
    """Save `scene`, the edited scene `index` of the saved project `saved` (its new text, dialogue or sound), and queue
    the job that makes its new take from the take the scene shows (edit_job), so the picture stays as far as the
    change allows. change: what the first frame should show differently (an instruction for Qwen-Image-Edit, "" for
    none). scope (EDIT_SCOPES, _edit_steps): "scene" makes this scene alone (a scene after it that continues from it
    then shows Outdated, as after any new take); "after" carries the edit on to the end of the film and "film" through
    the whole film, and a text the picture shows that the edit changed ("CRYPTO NEWS" to "BREAKING NEWS") changes in
    their scenes too. Every scene whose text the edit changes keeps the text it had in versions.json, with the take it
    starts from. Returns (the saved project, the job)."""
    change = " ".join(str(change or "").split())
    if scope not in EDIT_SCOPES:
        raise ValueError("Choose which scenes the edit changes.")
    if len(change) > MAX_CHANGE:
        raise ValueError(f"Keep the picture change under {MAX_CHANGE} characters.")
    if llm.blocked(change, *_texts(scene)):
        raise ValueError(llm.BLOCK_MSG)
    with _LOCK:
        current = load(saved.id)
        if current.key != saved.key:
            raise ValueError("The timeline changed. Open the saved project again.")
        if has_active_job(current):
            raise ValueError("This project already has a job in the queue.")
        if not _valid_index(current, index):
            raise ValueError("Choose a scene on the timeline.")
        keys = list(prepared(current)["assets"])
        source = _edit_source(current, index, keys)
        if source is None:
            raise ValueError(f"Generate scene {index + 1} first: an edit starts from its clip.")
        before = current.scenes[index]
        if scene == before and not change:
            raise ValueError("There is nothing to change yet. Describe the change, or change the scene's text or "
                             "dialogue first.")
        scenes = list(current.scenes)
        scenes[index] = scene
        first, last = (0 if scope == "film" else index), (index if scope == "scene" else len(scenes) - 1)
        if scope != "scene":
            for old, new in _shown_changes(before.prompt, scene.prompt):
                for number in range(first, last + 1):
                    if number != index:
                        scenes[number] = replace(scenes[number],
                                                 prompt=dialogue.replace_shown(scenes[number].prompt, old, new))
        changed = [number for number, (new, old) in enumerate(zip(scenes, current.scenes)) if new != old]
        candidate = _validated(replace(current, scenes=tuple(scenes)))
        casts = _validate_generation(candidate, changed)
        _check_issues(candidate, changed)
        sources = {str(index): source}
        for number in range(first, last + 1):
            if (number != index and number < len(keys) and _valid_key(keys[number])
                    and asset(current, keys[number], verify=False)):
                sources[str(number)] = keys[number]
        before_signatures = {}
        for number in changed:
            record = asset(current, sources[str(number)], verify=False) if str(number) in sources else None
            signature = record.get("scene_signature") if record else None
            if not _valid_key(signature) and record:
                try:
                    signature = _scene_signature(current, current.scenes[number],
                                                 _members_cached(current, current.scenes[number]))
                except ValueError:
                    signature = None
            before_signatures[number] = signature
        project = save(current.id, current.revision, current.title, tuple(scenes), current.aspect, current.audio,
                       list(current.characters), reference_characters=current.reference_characters,
                       universe=current.universe, universe_members=current.universe_members,
                       subtitles=current.subtitles, style=current.style)
        _remember_versions(project, [
            (_scene_signature(project, scenes[number], casts[number]), scenes[number].clip_settings(),
             sources[str(number)], before_signatures[number], current.scenes[number].clip_settings())
            for number in changed if before_signatures.get(number)])
        params = {"project": project.id, "key": project.key, "scene_index": index, "change": change, "scope": scope,
                  "sources": sources, "changed": changed, "take": uuid.uuid4().hex}
        steps = _edit_steps(project, index, scope, change, changed, {int(n): key for n, key in sources.items()})
        params["seconds"] = round(sum(project.scenes[number].seconds for number, _ in steps), 1) or scene.seconds
        return project, queue_manager.add(EDIT, f"Edit scene {index + 1} · {project.title}", params)


def retry_edit(project: Project) -> queue_manager.Job:
    """Queue the saved project's failed AI edit again (failure()["edit"]): the takes it made already are found again
    under their keys, and the scene it failed at gets a new take (seed)."""
    with _LOCK:
        if load(project.id).key != project.key:
            raise ValueError("The timeline changed. Open the saved project again.")
        if has_active_job(project):
            raise ValueError("This project already has a job in the queue.")
        record = failure(project)
        if not record or record["kind"] != EDIT or not isinstance(record.get("edit"), dict):
            raise ValueError("There is no failed edit to retry.")
        params = {name: record["edit"][name] for name in _RETRY_FIELDS if name in record["edit"]}
        params.update(project=project.id, key=project.key)
        if type(record.get("scene")) is int:
            retakes = dict(params["retakes"]) if isinstance(params.get("retakes"), dict) else {}
            retakes[str(record["scene"])] = uuid.uuid4().hex
            params["retakes"] = retakes
        index = _edit_plan(project, params)[0]
        return queue_manager.add(EDIT, f"Edit scene {index + 1} · {project.title}", params)


def _edit_plan(project: Project, params: dict) -> tuple[int, list, dict, str, dict]:
    """An edit job's parameters checked against its project: (the edited scene, the steps (_edit_steps), the take
    each scene starts from, the picture change, the take each "edit" step makes). The steps' takes follow from the
    edit's own, so a job that is paused and runs again finds the takes it made."""
    index = params.get("scene_index")
    if type(index) is not int or not _valid_index(project, index):
        raise ValueError("Choose a scene on the timeline.")
    scope = params.get("scope") if params.get("scope") in EDIT_SCOPES else "scene"
    change = params.get("change") if isinstance(params.get("change"), str) else ""
    raw = params.get("sources") if isinstance(params.get("sources"), dict) else {}
    sources = {int(number): key for number, key in raw.items()
               if isinstance(number, str) and number.isdigit() and _valid_index(project, int(number))
               and _valid_key(key)}
    changed = [number for number in (params.get("changed") if isinstance(params.get("changed"), list) else [])
               if type(number) is int and _valid_index(project, number)]
    take = params.get("take")
    if index not in sources or not (isinstance(take, str) and _ID.fullmatch(take)):
        raise ValueError(f"Scene {index + 1}: this edit lost the take it starts from. Make the edit again.")
    retakes = params.get("retakes") if isinstance(params.get("retakes"), dict) else {}
    steps = _edit_steps(project, index, scope, change, changed, sources)
    made = {}
    for number, how in steps:
        if how == "edit":
            again = retakes.get(str(number))
            made[number] = (again if isinstance(again, str) and _ID.fullmatch(again)
                            else take if number == index else hashlib.sha256(f"{take}:{number}".encode()).hexdigest()[:32])
    return index, steps, sources, change, made


def _still(scene: Scene) -> tuple:
    """What a scene's picture is made from apart from what is said: an edit that leaves it may keep the picture."""
    return _without_quotes(scene.prompt), scene.cut, scene.cast, scene.references


def _edit_mode(project: Project, scene: Scene, members: list[dict], source_key: str, source: dict, change: str) -> str:
    """How an AI edit makes its own scene's new take from take `source`: "voice" keeps the picture and makes only the
    sound again (the edit changed only what is heard, and the picture moves no mouth: lip sync was left out of it, or
    nobody speaks before or after); "redub" keeps a Realistic talking picture and lets MuseTalk move the mouths to the
    new lines; "animate" makes the picture again from the take's first frame. The scene as it was when the take was
    made is the one versions.json remembers for it."""
    if change or scene.seconds != source["seconds"]:
        return "animate"
    entry = _versions(project).get(_scene_signature(project, scene, members)) or {}
    base = entry.get("bases", {}).get(source_key)
    try:
        before = Scene.from_json(base["settings"]) if base else None
    except (ValueError, TypeError):
        before = None
    if before is None or _still(before) != _still(scene):
        return "animate"
    cues = effective(project, scene, members)[1] if project.audio else ()
    spoke = bool(source.get("voices"))
    if not cues:
        return "animate" if spoke else "voice"  # lines taken out: the mouth must stop moving
    said_before = {row.get("character") for row in source.get("dialogue_timings") or () if isinstance(row, dict)}
    said_now = {speaker.character for speaker in scene_sound.resolve_speakers(cues, members)}
    if source.get("lipsync_skipped") and said_now == said_before:
        return "voice"
    look = project.look
    if (spoke and look.family == styles.REALISTIC and look.musetalk and source.get("lipsync") != "infinitetalk"
            and not source.get("lipsync_skipped")):
        return "redub"
    return "animate"


def _show(project: Project, number: int, key: str) -> None:
    """Put clip `key` in the plan as scene `number`'s clip (a new take of an AI edit)."""
    with _LOCK:
        cancel.check()
        _current({"project": project.id, "key": project.key})
        state = prepared(project)
        keys = list(state["assets"])
        keys.extend([None] * (number + 1 - len(keys)))
        if keys[number] == key:
            return
        keys[number] = key
        try:
            _write_plan(project, {**state, "assets": keys,
                                  "export_assets": state.get("export_assets", state["assets"])})
        except OSError as error:
            print(f"[timeline] Could not save the plan: {error}", flush=True)
            raise ValueError(PLAN_NOT_SAVED) from error


def _changed_frame(project: Project, frame: Path, change: str, seed: int, out: Path, cb) -> Path:
    """The starting picture with an AI edit's picture change, made by Qwen-Image-Edit in the film's style
    (refs.compose) at the size a starting frame is drawn at (video_gen.sizes); the rest is asked to stay as it is."""
    look = project.look
    error = model_manager.ensure([engine_runtime.QWEN_EDIT_ID] + ([styles.STYLE_MODEL_ID] if look.has_loras else []),
                                 cb)
    if error:
        raise ValueError(error)
    cb(0.05, "Changing the first frame…")
    _, (width, height) = video_gen.sizes(video_gen.ASPECTS.get(project.aspect, 16 / 9))
    ok, message = refs.compose(engine_runtime.engine(), [str(frame)], f"{change.rstrip(' .')}. {_KEEP_REST}", width,
                               height, out, seed,
                               lambda fraction, text: cb(0.05 + 0.95 * fraction, f"First frame: {text}"),
                               style=project.style)
    if not ok:
        raise ValueError(message)
    return out


def _animated_take(project: Project, number: int, scene: Scene, members: list[dict], previous: dict | None,
                   style: str, key: str, cb, source: dict, change: str, meta: dict) -> dict:
    """An AI edit's take made again from a starting picture: the first frame of take `source` (for an anchored scene
    the previous scene's last frame, as for any take of it), changed first by Qwen-Image-Edit when the edit names a
    picture change."""
    clips = _asset_paths(project, key)[0].parent
    clips.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=clips) as folder:
        start = Path(folder) / "start.png"
        if previous is not None and not scene.cut:
            ok, message = ffmpeg_utils.last_frame(previous["path"], start)
        else:
            ok, message = ffmpeg_utils.frame_at(source["path"], 0.0, start)
        if not ok:
            raise ValueError(message or "The take's first frame could not be read.")
        progress = cb
        if change:
            share = _EDIT_FRAME_SHARE
            start = _changed_frame(project, start, change, int(meta["take"][:8], 16) % (2 ** 31 - 1),
                                   Path(folder) / "changed.png", lambda fraction, text: cb(share * fraction, text))

            def progress(fraction, text):
                cb(share + (1 - share) * fraction, text)
        return _build_clip(project, number, scene, members, previous, style, key, progress, meta=meta, start=start)


def _voice_take(project: Project, number: int, scene: Scene, members: list[dict], previous: dict | None,
                style: str, key: str, cb, source: dict, meta: dict) -> dict:
    """An AI edit's take with the picture of take `source` (copied, not encoded again) and the scene's new sound: its
    lines, background and effects, mixed as video_gen mixes a clip (the edit changed only what is heard)."""
    clip_path, _ = _asset_paths(project, key)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=clip_path.parent) as folder:
        work = Path(folder)
        clip, stems, overlay = work / "clip.mp4", None, None
        candidate = {"voices": [], "dialogue_timings": [], "background_model": source.get("background_model")}
        if project.audio:
            made = _audition(project, number, scene, members)
            if made is None:
                made = _scene_sound(project, number, scene, members, lambda fraction, text: cb(0.85 * fraction, text))
            if made.background is None:
                raise ValueError("The scene's new background sound is missing.")
            background = work / "background.wav"
            shutil.copyfile(made.background, background)
            voice = None
            if made.voice is not None:
                voice = work / "voice.wav"
                shutil.copyfile(made.voice, voice)
            if made.overlay is not None:
                overlay = work / "effects.wav"
                shutil.copyfile(made.overlay, overlay)
            cb(0.9, "Reusing the picture, mixing the new sound…")
            scene_sound.clip_sound(source["path"], background, voice, clip)
            stems = background, voice
            said = list(made.texts)
            candidate = {"voices": [getattr(voices.get(speaker.voice), "name", speaker.voice)
                                    for speaker in made.speakers],
                         "dialogue_timings": [{"text": text, "character": speaker.character, "voice": speaker.voice,
                                               "start": begin, "end": end}
                                              for text, speaker, (begin, end) in zip(said, made.speakers, made.timings)],
                         "background_model": sound.MODEL_ID, "sound_key": made.key}
        else:
            cb(0.5, "Reusing the picture…")
            shutil.copyfile(source["path"], clip)
        candidate.update({name: source[name] for name in ("qc", "width", "height") if name in source})
        if source.get("lipsync_skipped"):
            candidate["lipsync_skipped"] = source["lipsync_skipped"]
        candidate.update(path=clip, audio_sources=stems, effects_source=overlay)
        meta = {**meta, "picture_sha256": _picture_sha(source),
                "anchor_sha256": source.get("anchor_sha256", meta.get("anchor_sha256"))}
        return _build_clip(project, number, scene, members, previous, style, key, cb, candidate, meta=meta)


def _redubber(number: int, source: dict):
    """_build_clip's renderer for a "redub" take: the scene's new sound, and the picture of take `source` with the
    mouths moved to it (video_gen.redub)."""
    def render(project, scene, members, seed, style, folder, cb, anchor=None):
        package = _audition(project, number, scene, members)
        report = cb
        if package is None:
            package = _scene_sound(project, number, scene, members, lambda fraction, text: cb(0.2 * fraction, text))

            def report(fraction, text):
                cb(0.2 + 0.8 * fraction, text)
        return video_gen.redub(source["path"], package, members, scene.seconds, seed,
                               (source["width"], source["height"]), folder, report)
    return render


def _edit_scene(project: Project, number: int, source_key: str, change: str, take: str, cb, *,
                own: bool) -> dict:
    """Make take `take` of scene `number` for an AI edit, from the scene's take `source_key`, and show it in the
    plan; returns its record. own: the scene the edit was asked for, whose new take may keep the source's picture
    (_edit_mode); any other scene of the edit starts again from its take's first frame with the picture change."""
    scene = project.scenes[number]
    members = _scene_cast(project, scene)
    style, tier = project.video_style, model_manager.tier()
    previous = _anchor(project, number, list(prepared(project)["assets"]))
    continuity = [_picture_sha(previous), project.scenes[number - 1].prompt] if previous else None
    key = _clip_key(project, scene.clip_settings(), members, style, tier, number, continuity, None, take)
    record = asset(project, key)
    if record is None:
        source = asset(project, source_key)
        if source is None:
            raise ValueError("The take this edit starts from is missing or has changed. Generate the scene instead.")
        cb(0.0, "Preparing the edit…")
        mode = _edit_mode(project, scene, members, source_key, source, change) if own else "animate"
        meta = {"scene_signature": _scene_signature(project, scene, members),
                "picture_key": _picture_key(project, scene, members, style, tier, number, continuity, None, take),
                "anchor_sha256": continuity[0] if continuity else None, "take": take, "created": time.time(),
                "style": style, "tier": tier,
                "edit": {"kind": mode, "from": source_key, **({"change": change} if change else {})}}
        if mode == "voice":
            record = _voice_take(project, number, scene, members, previous, style, key, cb, source, meta)
        elif mode == "redub":
            meta["anchor_sha256"] = source.get("anchor_sha256", meta["anchor_sha256"])
            record = _build_clip(project, number, scene, members, previous, style, key, cb, meta=meta,
                                 generate_scene=_redubber(number, source))
        else:
            record = _animated_take(project, number, scene, members, previous, style, key, cb, source, change, meta)
    _show(project, number, key)
    return record


def edit_job(params: dict, cb) -> tuple:
    """Make an AI edit's takes (edit): its scene's new take from the take the scene showed, then the scenes the edit
    carries on to (_edit_steps), each shown in the plan as soon as it is made. A failure is remembered with the edit
    (retry_edit); the takes made before it stay in the plan."""
    project = _current(params)
    retry = {name: params[name] for name in _RETRY_FIELDS if name in params}
    try:
        index, steps, sources, change, made = _edit_plan(project, params)
    except ValueError as error:
        _record_failure(project, EDIT, None, str(error))
        return False, [], [], str(error)
    total = sum(project.scenes[number].seconds for number, _ in steps) or 1
    done, notes = 0.0, []
    for number, how in steps:
        share = project.scenes[number].seconds / total

        def share_of(fraction, message, start=done, share=share):
            cb(min(start + share * min(max(fraction, 0.0), 1.0), 1.0), message)

        def progress(fraction, message, number=number):
            share_of(fraction, f"Scene {number + 1}/{len(project.scenes)} · {message}")

        retake = False
        try:
            if how == "follow":  # _prepare names the scene in its messages itself
                ok, more, error, _, retake = _prepare(project, {"project": project.id, "key": project.key},
                                                      share_of, indices=[number])
                if not ok:
                    raise ValueError(error)
                notes.extend(more[1:])
            else:
                record = _edit_scene(project, number, sources[number], change, made[number], progress,
                                     own=number == index)
                for note in (record.get("lipsync_skipped"), record.get("start_frame_note")):
                    if note:
                        notes.append(f"Scene {number + 1}: {note}")
        except (ValueError, OSError, MediaError) as error:
            text = str(error)
            message = text if text.startswith(f"Scene {number + 1}:") else f"Scene {number + 1}: {_explain(error)}"
            _record_failure(project, EDIT, number, message, retake=retake, edit=retry)
            return False, [], notes, message
        done += share
    _clear_failure(project, EDIT)
    cb(1, "The edit is ready.")
    count = len(steps)
    return True, [], [f"Scene {index + 1} edited: {count} scene{'s' if count != 1 else ''} made.", *notes], ""


queue_manager.register(PREPARE, prepare_job)
queue_manager.register(MERGE, merge_job)
queue_manager.register(RENDER, render_job)
queue_manager.register(SOUND, sound_job)
queue_manager.register(PRODUCE, produce_job)
queue_manager.register(EDIT, edit_job)
