# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Resolve model files and route product operations to the shared Engine."""
from __future__ import annotations

import io
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from engine.api import Engine, QwenEditFiles, TalkFiles, WanFiles, ZImageFiles

from . import cancel, config, model_manager

ZIMAGE_ID = "z-image-turbo"
QWEN_EDIT_ID = "qwen-image-edit-2511"
TALK_ID = "infinitetalk"  # talking animated characters (engine.pipelines.talk)
TALK_FPS = 25
BACKGROUND_AUDIO_ID = "ezaudio-xl"
# Where each Z-Image file of the active set lives: (catalog folder, name fragments...) in order. The Full set has
# the official bf16 files in their own folders, the Lite set its GGUFs; both use the official VAE.
ZIMAGE_ROLES = {"transformer": (("z_image_turbo_transformer",), ("diffusion_models",)),
                "text_encoder": (("z_image_turbo_text_encoder",), ("text_encoders",)),
                "vae": (("z_image_turbo_vae",),)}
# The Wan entries that carry the RIFE weights (all four list the same file).
RIFE_MODELS = ("wan22-a14b-i2v", "wan22-a14b-t2v")
# The Wan 2.2 image-to-video experts in scaled FP8 (FP8 tensor-core products on RTX 40-series and newer), each the
# Full set's GGUF expert with its _1022 LoRA merged in at strength 1.0: the low-noise one is LightX2V's published
# file, the high-noise one is made from the GGUF expert and LoRA at installation (engine.models.wan22.convert), because
# LightX2V's high-noise FP8 file holds another distillation, which changes composition and motion. An installed FP8
# expert replaces its GGUF expert and LoRA only when a check finds that LoRA merged in, value for value
# (engine.models.wan22.transformer.lora_merged_share), so a file that holds anything else is never used in its place;
# engine.fp8_experts "off" uses neither. The text encoder, VAE and RIFE still come from the image-to-video entry; the
# Lite set keeps its own smaller files.
FAST_I2V_ID = "wan22-a14b-i2v-fp8"
FAST_I2V_FOLDERS = {"high": "wan_i2v_fp8_local", "low": "wan_i2v_fp8"}  # catalog destination of each FP8 expert
# The text-to-video experts the same way: each the Full set's GGUF expert with its _1217 LoRA merged in, made at
# installation. A step of the GGUF expert dequantizes Q6_K and runs 400 LoRA hooks: about 15 s against 10 s in FP8.
FAST_T2V_ID = "wan22-a14b-t2v-fp8"
FAST_T2V_FOLDERS = {"high": "wan_t2v_fp8_local", "low": "wan_t2v_fp8_local"}
FAST_EXPERTS = {"i2v": (FAST_I2V_ID, FAST_I2V_FOLDERS), "t2v": (FAST_T2V_ID, FAST_T2V_FOLDERS)}
MATCHING_SHARE = 0.9  # a merged LoRA scores 0.97-1.0, another distillation about half (lora_merged_share)
_MERGE_CHECKS: dict = {}  # (FP8 file, GGUF base, LoRA) identities -> share: one check per process and file version
# engine.warm_cache_gib "auto": room for Wan's two experts and its VAE (Full set: 2 x 11.2 GiB + 0.24 GiB;
# the Lite set's Q4_K_M experts are 9 GiB each), so the parts of a clip and the scenes of a Timeline job load
# them once while they run the same experts (image-to-video after image-to-video). Qwen-Image-Edit (the
# starting frame of a later shot with a character or reference images) and Z-Image clear the cache, so such a
# shot loads the experts again. The text encoder (10.6 GiB of bf16 in the Full set) does not push them out;
# its encodings are kept. A pair of experts whose files do not fit together (both FP8 experts: 28 GiB; a GGUF and
# an FP8 expert: 25.2 GiB) is not kept at all (WanModels.from_files), so every part and scene loads both from their
# files again.
AUTO_WARM_CACHE_GIB = 24
# Only where that leaves room for everything else: 64 GB machines report about 63.4 GiB. With 32 GB the
# warm models would push the next stage's into the page file, so nothing is kept there.
AUTO_WARM_CACHE_MIN_RAM_GIB = 60
# Triton keeps the GPU kernels it compiles (SageAttention, the FP8 helpers) here, so only the first run of an
# installation compiles them (unless TRITON_CACHE_DIR names another folder).
TRITON_CACHE_DIR = config.MODELS_DIR / ".cache" / "triton"
_ENGINE: Optional[Engine] = None
_EXPECTED_PROMPTS: tuple = ()  # the Wan prompts the running Timeline job announced (expect_video_prompts)
_NEXT_STAGE: tuple = ()  # the files the running Timeline job's next scene reads first (next_stage)
_AFTER_FRAME: tuple = ()  # the files a scene reads right after its starting frame (after_start_frame)
_LOCK = threading.Lock()


def tokenizer_dir() -> Path:
    return config.destination_dir("tokenizer_z_image")


def _first_path(model_id: str, candidates, tier_name: Optional[str] = None) -> Optional[Path]:
    for dest, *fragments in candidates:
        path = model_manager.model_path(model_id, dest, *fragments, tier_name=tier_name)
        if path is not None:
            return path
    return None


def zimage_file(role: str, tier_name: Optional[str] = None) -> Optional[Path]:
    """The installed Z-Image file of `role` (see ZIMAGE_ROLES) in the given (default: current) model set."""
    return _first_path(ZIMAGE_ID, ZIMAGE_ROLES[role], tier_name)


def zimage_files() -> Optional[ZImageFiles]:
    """Resolve the Z-Image weights for the selected tier."""
    paths = [zimage_file(role) for role in ("transformer", "text_encoder", "vae")]
    if not all(paths):
        return None
    files = ZImageFiles(paths[0], paths[1], paths[2], tokenizer_dir())
    return None if files.missing() else files


def small_gpu() -> bool:
    limit = config.SETTINGS["engine"].get("vram_limit_gib")
    return (limit is not None and limit < 15) or model_manager.small_gpu()


def _rife_weights() -> Path:
    """The RIFE 4.9 weights: the author's flownet.pkl, or the same tensors kept by an earlier installation."""
    path = _first_path(RIFE_MODELS[0], (("rife",),)) or _first_path(RIFE_MODELS[1], (("rife",),))
    if path is None:
        raise FileNotFoundError(f"RIFE weights are missing in {config.destination_dir('rife')}")
    return path


def _encode_frames(path, frames, fps, crf):
    from engine.media.video import Writer

    height, width = frames.shape[-2:]
    with Writer(path, fps, width, height, crf=crf, colorspace=1, color_range=1,
                color_primaries=1, color_trc=1) as writer:
        for frame in frames:
            cancel.check()
            writer.write(_frame_array(frame))


def _frame_array(frame):
    import torch

    return (frame.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).numpy()


def _frame_tensors(path):
    import torch
    from engine.media.video import read_frames

    for frame in read_frames(path):
        cancel.check()
        yield torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)


def _read_frames(path):
    import torch

    frames = list(_frame_tensors(path))
    if not frames:
        raise ValueError("The input contains no video frames.")
    return torch.stack(frames).float().div_(255.0)


def _holds_the_lora(expert: str, fp8_file: Path, kind: str = "i2v") -> bool:
    """Whether the FP8 expert is the Full set's GGUF expert of `kind` with its LoRA merged in (checked once per file
    version)."""
    from engine.core.memory import file_identity
    from engine.models.wan22.transformer import lora_merged_share

    base = model_manager.model_path(f"wan22-a14b-{kind}", "diffusion_models", f"{expert}noise")
    lora = model_manager.model_path(f"wan22-a14b-{kind}", "loras", f"{expert}_noise")
    if base is None or lora is None:
        return False
    try:
        key = tuple(file_identity(path) for path in (fp8_file, base, lora))
        if key not in _MERGE_CHECKS:
            _MERGE_CHECKS[key] = lora_merged_share(fp8_file, base, lora)
            if _MERGE_CHECKS[key] < MATCHING_SHARE:
                print(f"[video] {fp8_file.name} holds another distillation than {lora.name} "
                      f"({_MERGE_CHECKS[key]:.2f} of the checked values match): its GGUF expert and LoRA run.",
                      file=sys.stderr, flush=True)
        return _MERGE_CHECKS[key] >= MATCHING_SHARE
    except Exception as error:  # noqa: BLE001 - an unreadable or foreign file only means the GGUF expert runs
        print(f"[video] {fp8_file.name} could not be checked ({error}): its GGUF expert and LoRA run.",
              file=sys.stderr, flush=True)
        return False


def fast_experts(kind: str, lora_strength: float = 1.0) -> Dict[str, Path]:
    """The FP8 experts (FAST_EXPERTS: FAST_I2V_ID, FAST_T2V_ID) that replace the Full set's GGUF experts of `kind` and
    their LoRAs, by expert ("high", "low"). Empty for the Lite set, a LoRA strength other than the 1.0 a merged
    distillation stands for, or engine.fp8_experts "off". "matching" (the default) takes an installed FP8 expert only
    when it is the GGUF expert with that LoRA merged in (_holds_the_lora)."""
    if kind not in FAST_EXPERTS or config.SETTINGS["engine"].get("fp8_experts", "matching") == "off" \
            or lora_strength != 1.0 or model_manager.tier() != "full":
        return {}
    model_id, folders = FAST_EXPERTS[kind]
    chosen = {}
    for expert, folder in folders.items():
        path = model_manager.model_path(model_id, folder, f"{expert}_noise")
        if path is not None and _holds_the_lora(expert, path, kind):
            chosen[expert] = path
    return chosen


def fast_i2v_experts(lora_strength: float = 1.0) -> Dict[str, Path]:
    """fast_experts of image to video."""
    return fast_experts("i2v", lora_strength)


def video_experts(kind: str, lora_strength: float = 1.0, style: str = "realistic") -> Tuple[WanFiles, Optional[dict]]:
    """The Wan files a video part of `kind` runs with, and its LoRAs by expert (None for none): the FP8 experts
    (fast_experts) or the GGUF ones with their LoRAs, an animated style's own experts (style_experts) and its motion
    LoRA (style_loras)."""
    def required(dest, *matches):
        path = model_manager.model_path(f"wan22-a14b-{kind}", dest, *matches)
        if path is None:
            description = " ".join((dest, *matches))
            raise FileNotFoundError(f"Wan {kind} model bundle is missing {description}.")
        return path

    # The Full set reads the official sharded UMT5 through its index, the Lite set its GGUF.
    text_encoder = model_manager.model_path(f"wan22-a14b-{kind}", "wan_text_encoder", "index") \
        or required("text_encoders")
    fast = fast_experts(kind, lora_strength)  # FP8 experts carry their distillation: no LoRA
    # an animated style's own experts (the 2D style's AniSora) carry the same distillation and take their place
    fast = {**fast, **(style_experts(style, lora_strength) if kind == "i2v" else {})}
    experts = {expert: fast.get(expert) or required("diffusion_models", f"{expert}noise") for expert in ("high", "low")}
    files = WanFiles(text_encoder, experts["high"], experts["low"], required("wan_vae"), kind)
    loras = {expert: [(required("loras", f"{expert}_noise"), lora_strength)]
             for expert in ("high", "low") if lora_strength and expert not in fast}
    for path, strength in style_loras(style, "video"):
        loras.setdefault("high", []).append((path, strength))
    return files, loras or None


def video_part(runtime: Engine, kind: str, start_path: str | None, part_path: Path, native_fps: int, fps: int, *,
               prompt: str, width: int, height: int, frame_count: int, last_index: int,
               seed: int, steps: int, shift: float, split: int, lora_strength: float,
               progress_cb=None, style: str = "realistic", then=()) -> bytes:
    """Write one RIFE-interpolated segment and return its lossless native-rate handoff frame. `style` (app/styles.py):
    an animated style's motion LoRA joins the high-noise expert's LoRAs (`style_loras`). `then`: files the next stage
    reads first, read ahead while the low-noise expert computes (next_stage_files)."""
    from PIL import Image, ImageOps

    files, loras = video_experts(kind, lora_strength, style)
    weights = _rife_weights()
    picture = None
    if start_path:
        with Image.open(start_path) as source:
            picture = ImageOps.exif_transpose(source).convert("RGB")

    def sample_progress(frac, desc):
        if progress_cb is not None:
            progress_cb(frac * 0.9, desc)

    def rife_progress(frac, desc):
        if progress_cb is not None:
            progress_cb(0.9 + frac * 0.1, desc)

    frames = runtime.video(files, prompt, width, height, frame_count, seed, steps, shift, split,
                           start_image=picture, loras=loras, progress_cb=sample_progress, then=then)
    handoff = _png(Image.fromarray(_frame_array(frames[last_index])))
    frames = runtime.interpolate(weights, frames, native_fps, fps, rife_progress)
    _encode_frames(part_path, frames, fps, crf=19)
    return handoff


def talk_files() -> TalkFiles:
    """The installed InfiniteTalk bundle, with the UMT5 encoder and VAE of Wan 2.2's image-to-video entry."""
    model = config.get_model(TALK_ID) or {}
    base = tuple(model_manager._file_target(spec) for spec in model.get("files", ())
                 if spec.get("dest") == "infinitetalk_base")
    text_encoder = model_manager.model_path("wan22-a14b-i2v", "wan_text_encoder", "index")
    vae_file = model_manager.model_path("wan22-a14b-i2v", "wan_vae")
    adapter = model_manager.model_path(TALK_ID, "infinitetalk", "infinitetalk")
    clip = model_manager.model_path(TALK_ID, "infinitetalk", "clip_vision")
    if not base or text_encoder is None or vae_file is None or adapter is None or clip is None:
        raise FileNotFoundError("The InfiniteTalk model bundle is incomplete.")
    return TalkFiles(text_encoder, base, adapter, clip, vae_file, config.destination_dir("infinitetalk_wav2vec"))


def talk_installed() -> bool:
    """Whether InfiniteTalk can run here (its files and those it shares with Wan 2.2 image to video)."""
    return model_manager.is_installed(TALK_ID, exact=True) and model_manager.is_installed("wan22-a14b-i2v")


def talk_part(runtime: Engine, start_path: str, speech_path: str, part_path: Path, *, prompt: str, width: int,
              height: int, seconds: float, seed: int, audio_scale: float = 1.0, progress_cb=None, then=()) -> None:
    """A talking clip at TALK_FPS of exactly `seconds` (InfiniteTalk): the character in the picture `start_path`
    saying the speech of `speech_path` (on the clip's clock from 0), written to `part_path`. `then`: files the next
    stage reads first, read ahead while the last chunk computes (next_stage_files)."""
    from PIL import Image, ImageOps

    from engine.media import audio as media_audio

    samples, _rate = media_audio.read(speech_path, rate=16000, channels=1)
    with Image.open(start_path) as source:
        picture = ImageOps.exif_transpose(source).convert("RGB")
    frames = runtime.talk(talk_files(), prompt, picture, samples[0], round(seconds * TALK_FPS), width, height, seed,
                          audio_scale=audio_scale, progress_cb=progress_cb, then=then)
    _encode_frames(part_path, frames, TALK_FPS, crf=19)


def expect_video_prompts(prompts) -> None:
    """Announce the Wan prompts a job is about to use (video_gen.scene_wan_prompt of each scene it will make),
    or end the announcement with an empty sequence. The first video_part that loads the text encoder then
    encodes them all in that load, so the later scenes skip the encoder (Engine.expect_video_prompts). An
    engine made later (the first model the job runs) gets the announcement when it is made."""
    global _EXPECTED_PROMPTS
    with _LOCK:
        _EXPECTED_PROMPTS = tuple(prompts)
        current = _ENGINE
    if current is not None:
        current.expect_video_prompts(_EXPECTED_PROMPTS)


def interpolate_video(runtime: Engine, source, output, source_fps: int, target_fps: int, progress_cb=None) -> None:
    frames = runtime.interpolate(_rife_weights(), _read_frames(source), source_fps, target_fps, progress_cb)
    _encode_frames(output, frames, target_fps, crf=0)


def transition_frames(first: Path, last: Path, output: Path, frames: int, fps: int) -> None:
    """Use RIFE to fill only the frame slots between two existing endpoints."""
    import numpy as np
    from PIL import Image
    import torch

    pictures = []
    for path in (first, last):
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"))
        pictures.append(torch.from_numpy(pixels).permute(2, 0, 1))
    pair = torch.stack(pictures).float().div_(255)
    bridge = engine().interpolate(_rife_weights(), pair, 1, frames + 1)[1:-1]
    _encode_frames(output, bridge, fps, crf=17)


def background_audio(prompt, seconds, seed, output, progress_cb=None, *, runtime: Engine) -> None:
    """An EzAudio-XL sound for `prompt` in `output` (WAV). The negative text is EzAudio's unconditional one, the
    empty prompt it was trained with (engine.pipelines.ezaudio.NEGATIVE)."""
    from engine.media.audio import write_wav
    from engine.pipelines.ezaudio import NEGATIVE

    bundle = config.destination_dir("ezaudio")
    samples, rate = runtime.audio_bg(bundle, prompt, NEGATIVE, seconds, seed, progress_cb)
    write_wav(output, samples.numpy(), rate)


def sound_files(voice: bool, background: bool) -> list:
    """The weight files a scene's sound reads, in the order it loads them: VoxCPM2's (the lines) and EzAudio-XL's
    (the background: the FLAN-T5 shards with encoder tensors, the diffusion transformer, the waveform decoder)."""
    files = []
    if voice:
        folder = config.destination_dir("voxcpm2")
        files += [folder / "model.safetensors", folder / "audiovae.pth"]
    if background:
        from engine.models.ezaudio.text import shard_files
        from engine.pipelines.ezaudio import EzAudioFiles

        bundle = EzAudioFiles.from_bundle(config.destination_dir("ezaudio"))
        try:
            files += shard_files(bundle.text_encoder)
        except (OSError, ValueError, KeyError):
            pass  # the load reports it
        files += [bundle.transformer, bundle.vae]
    return files


def read_ahead_sound(voice: bool, background: bool, then=()) -> Optional[threading.Thread]:
    """Read the sound models' files (sound_files) into the file cache at the disk's speed, in the order they load,
    while they load: they are memory-mapped and come in by page faults otherwise (FLAN-T5's 4.9 GB of float32 encoder
    took 14 s to load in a scene's first sound, 4 s once cached, 2026-11-15); EzAudio's are read while VoxCPM2 speaks.
    Normal disk priority: nothing else of the scene runs before its sound. `then`: files read after them in the same
    pass (the files the first picture stage starts with), so the sound models never wait behind them. Only a queued
    job reads ahead (not a test or a page that calls the same code)."""
    from engine.core import readahead

    from . import queue_manager

    if not queue_manager.in_worker() or not (voice or background or then):
        return None
    try:
        files = sound_files(voice, background)
    except Exception as error:  # noqa: BLE001 - only a speed-up
        print(f"[sound] The sound models are not read ahead: {error}", file=sys.stderr, flush=True)
        files = []
    return readahead.ahead([*files, *then], name="sound-read-ahead")


def qwen_edit_files() -> QwenEditFiles:
    """Find the active model bundle."""
    transformer = model_manager.model_path(QWEN_EDIT_ID, "diffusion_models")
    text = model_manager.model_path(QWEN_EDIT_ID, "qwen_edit_text_encoder")
    vae = model_manager.model_path(QWEN_EDIT_ID, "qwen_edit_vae")
    lightning = model_manager.model_path(QWEN_EDIT_ID, "loras", "lightning")
    if transformer is None or text is None or vae is None or lightning is None:
        raise FileNotFoundError("The active Qwen-Image-Edit model bundle is incomplete.")
    tokenizer = config.destination_dir("tokenizer_qwen_edit")
    return QwenEditFiles(transformer, text, vae, tokenizer, lightning)


def read_ahead(paths) -> Optional[threading.Thread]:
    """Read model files into Windows' file cache on a background thread, one after another in big sequential reads at
    background disk priority (engine.core.readahead.ahead), while the job does something else, so the stage that
    needs them next finds them in memory instead of reading the disk page by page. A file another read-ahead is
    reading already, or finished reading a moment ago, is skipped; a file that cannot be read is only not read ahead.
    Returns the thread, or None when there was nothing to read."""
    from engine.core import readahead

    return readahead.ahead(paths, background=True)


def next_stage(paths) -> None:
    """Announce the files the running job's next scene reads first (a Timeline job sets them before each scene, ()
    when none): the last picture stage of the scene being made reads them ahead while its last expert or talking chunk
    computes and the disk is idle (next_stage_files, Engine.video / Engine.talk `then`)."""
    global _NEXT_STAGE
    _NEXT_STAGE = tuple(path for path in paths if path)


def after_start_frame(paths) -> None:
    """Announce the files a scene reads first after its starting frame (its video experts, or InfiniteTalk's network
    for a talking scene); () when done. The composition or drawing of the frame reads them ahead while its steps run
    (Engine.edit / Engine.txt2img `then`)."""
    global _AFTER_FRAME
    _AFTER_FRAME = tuple(path for path in paths if path)


def after_start_frame_files() -> tuple:
    """The files announced by after_start_frame; () outside a queued job."""
    from . import queue_manager

    return _AFTER_FRAME if queue_manager.in_worker() else ()


def next_stage_files() -> tuple:
    """The files announced by next_stage; () outside a queued job (a test or a page calling the same code)."""
    from . import queue_manager

    return _NEXT_STAGE if queue_manager.in_worker() else ()


def drawn_frame_files(style: str) -> list:
    """What a starting frame drawn by Z-Image in an animated style reads (styles.Style.draws_start): its transformer,
    text encoder, VAE and the style's image LoRA (about 20 GB)."""
    files = zimage_files()
    if files is None:
        return []
    try:
        style_files = [path for path, _ in style_loras(style, "image")]
    except FileNotFoundError:
        style_files = []
    return [files.text_encoder, files.transformer, *style_files, files.vae]


def start_frame_files(style: str = "realistic") -> list:
    """What a character's or a composed starting frame reads, in the order it reads it: Qwen-Image-Edit's text
    encoder, its transformer, the Lightning LoRA, the style's edit LoRA and the VAE (about 31 GB in the Full set).
    `style`: a style id or a film's video style text (what a Timeline clip records)."""
    from . import styles

    files = qwen_edit_files()
    look = styles.of(style) if styles.valid(style) else styles.for_video_text(style)
    try:
        style_files = [path for path, _ in style_loras(look.id, "edit")]
    except FileNotFoundError:
        style_files = []  # the composition reports it
    return [files.text_encoder, files.transformer, files.lightning, *style_files, files.vae]


def read_ahead_start_frame(style: str = "realistic") -> Optional[threading.Thread]:
    """read_ahead of the starting frame's files (start_frame_files): a scene starts it before its sound, and a job
    for the scene before it while that scene's video computes (next_stage). Wan's experts push them out of the file
    cache in every scene, and the composition then waited about 20 s for the disk (2026-11-15). Only a queued job
    reads ahead (not a test or a page that calls the same code)."""
    from . import queue_manager

    if not queue_manager.in_worker():
        return None
    try:
        files = start_frame_files(style)
    except FileNotFoundError:
        return None
    return read_ahead(files)


def style_experts(style: str, lora_strength: float = 1.0) -> Dict[str, Path]:
    """The image-to-video experts of an animated style that replace Wan 2.2's (app/styles.py `video_experts`: the 2D
    style's Index-AniSora V3.2 with LightX2V's distillation merged in), by expert ("high", "low"); only the installed
    ones, none for Realistic, and none for a LoRA strength other than the 1.0 the merged distillation stands for.
    Like the FP8 experts of FAST_I2V_ID they run only in the Full set and not with engine.fp8_experts "off"."""
    from . import styles

    entry = styles.get(style).video_experts
    if not entry or lora_strength != 1.0 or model_manager.tier() != "full" \
            or config.SETTINGS["engine"].get("fp8_experts", "matching") == "off":
        return {}
    found = {}
    for expert in ("high", "low"):
        path = model_manager.model_path(entry, "anisora", f"{expert}_noise")
        if path is not None:
            found[expert] = path
    return found


def style_loras(style: str, role: str) -> List[Tuple[Path, float]]:
    """The (file, strength) of an animated style's LoRA for `role` ("edit", "image" or "video", app/styles.py), [] for
    Realistic or a role the style has none for. A missing file is an error: the film would silently lose its look."""
    from . import styles

    look = styles.get(style)
    found = getattr(look, f"{role}_lora")
    if not found:
        return []
    match, strength = found
    path = model_manager.model_path(styles.STYLE_MODEL_ID, "style_loras", match)
    if path is None:
        raise FileNotFoundError(f"The {styles.base_label(look.id)} style LoRA ({match}) is missing. Install the "
                                "animation styles.")
    return [(path, strength)]


def edit(images, prompt: str, width: int, height: int, seed: int, progress_cb=None, *, runtime: Engine,
         style: str = "realistic") -> bytes:
    """A Qwen-Image-Edit composition; an animated `style` adds its edit LoRA (`style_loras`)."""
    from PIL import Image, ImageOps

    if not Engine.available():
        raise RuntimeError("The own image editing engine requires CUDA and its Python dependencies.")
    files = qwen_edit_files()
    pictures = []
    for path in images:
        with Image.open(path) as source:
            pictures.append(ImageOps.exif_transpose(source).convert("RGB"))
    extra = style_loras(style, "edit")
    then = after_start_frame_files()
    options = {"then": then} if then else {}
    if extra:
        return _png(runtime.edit(files, pictures, prompt, width, height, seed, progress_cb, loras=extra, **options))
    return _png(runtime.edit(files, pictures, prompt, width, height, seed, progress_cb, **options))


def _files() -> ZImageFiles:
    """Look up the weights for the model set selected for this render."""
    files = zimage_files()
    if files is None:
        raise FileNotFoundError("The Z-Image weights for the active model set are missing.")
    return files


def _ram_gib() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 2 ** 30
    except Exception:  # noqa: BLE001 - unknown RAM counts as small: nothing is kept warm
        return 0.0


def warm_cache_gib() -> int:
    """How much host RAM the engine may keep loaded models in between stages and jobs (engine.warm_cache_gib:
    a number of GiB, or "auto" for AUTO_WARM_CACHE_GIB on machines with AUTO_WARM_CACHE_MIN_RAM_GIB of RAM
    and none on smaller ones). The queue gives the memory back when it runs out of jobs."""
    setting = config.SETTINGS["engine"].get("warm_cache_gib", "auto")
    if setting != "auto":
        return int(setting)
    return AUTO_WARM_CACHE_GIB if _ram_gib() >= AUTO_WARM_CACHE_MIN_RAM_GIB else 0


# The libraries the stages of a scene import on first use (speech, sound, starting frame, lip sync, video), imported
# at the start of the app (warm_imports) instead of inside the first job: in tools/bench_scene.py the first scene of
# a process spent about 6 s more than the next one loading VoxCPM2 (3 s), EzAudio's text encoder (3 s, the
# transformers library) and the lip-sync face finder (2026-11-15).
WARM_IMPORTS = ("voxcpm", "transformers.models.t5.modeling_t5", "transformers.models.qwen2_5_vl", "diffusers",
                "mediapipe", "engine.pipelines.speech", "engine.pipelines.ezaudio", "engine.pipelines.edit",
                "engine.pipelines.musetalk", "engine.pipelines.video", "engine.models.rife")


_WARMING: Optional[threading.Thread] = None


def warm_imports() -> list:
    """Import WARM_IMPORTS and the diffusers networks of Qwen Image Edit (engine.pipelines.edit.import_networks). A
    library that fails to import is left for the stage that needs it to report. Returns the names that failed."""
    import importlib

    failed = []
    for name in WARM_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - the stage that uses it reports the real error
            failed.append(name)
    try:
        from engine.pipelines.edit import import_networks

        import_networks()
    except Exception:  # noqa: BLE001 - as above
        failed.append("engine.pipelines.edit.import_networks")
    return failed


def start_warm_imports() -> None:
    """warm_imports on a background thread (once, when the app starts). engine() waits for it, so a job never
    imports the same libraries at the same time as it: two threads importing diffusers' lazily loaded modules at once
    can see one of them half initialised."""
    global _WARMING
    with _LOCK:
        if _WARMING is None:
            _WARMING = threading.Thread(target=warm_imports, name="warm-imports", daemon=True)
            _WARMING.start()


def engine() -> Engine:
    global _ENGINE
    warming = _WARMING
    if warming is not None and warming is not threading.current_thread():
        warming.join()
    with _LOCK:
        if _ENGINE is None:
            settings = config.SETTINGS["engine"]
            from engine import kernels

            kernels.use_cache_dir(TRITON_CACHE_DIR)  # compiled GPU kernels, kept between runs
            _ENGINE = Engine(cancel=cancel.TOKEN, lora_rounding=settings["lora_rounding"],
                             warm_cache_bytes=warm_cache_gib() << 30,
                             vram_limit_gib=settings.get("vram_limit_gib"),
                             attention=settings.get("attention", "auto"))
            if _EXPECTED_PROMPTS:
                _ENGINE.expect_video_prompts(_EXPECTED_PROMPTS)
        return _ENGINE


def release() -> None:
    """Give back every model the engine keeps and its unused GPU memory (a pause, an idle queue, exit)."""
    with _LOCK:
        current = _ENGINE
    if current is not None:
        current.free()


def _png(picture) -> bytes:
    buffer = io.BytesIO()
    picture.save(buffer, "PNG")
    return buffer.getvalue()


def render(positive: str, width: int, height: int, seed: int, steps: int, progress_cb=None, *,
           runtime: Engine | None = None, style: str = "realistic", keep: bool = True) -> bytes:
    """Text to picture on the engine; PNG bytes. An animated `style` adds its image LoRA (`style_loras`). `keep` False:
    Z-Image is not kept warm afterwards (Engine.txt2img)."""
    extra = style_loras(style, "image")
    options = {} if keep else {"keep": False}
    if not keep and after_start_frame_files():  # a scene's drawn starting frame
        options["then"] = after_start_frame_files()
    if extra:
        return _png((runtime or engine()).txt2img(_files(), positive, width, height, seed, steps, progress_cb,
                                                  loras=extra, **options))
    return _png((runtime or engine()).txt2img(_files(), positive, width, height, seed, steps, progress_cb, **options))
