# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Generate images and record the results."""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from . import config, engine_runtime, history, llm, metadata, model_manager
from .vram import low_vram_hint

ProgressCb = Callable[[float, str], None]

_OOM_HINTS = ("out of memory", "allocation on device")
_OOM_WORD = re.compile(r"\boom\b", re.IGNORECASE)  # the word alone: "Zoom", "Bloom", "Room" in a file path are not it
# Qwen-Image-Edit composes reference images at about 1 megapixel; the picture keeps that size (each
# model draws at its own resolution, nothing is enlarged afterwards).
EDIT_PIXELS = 1024 * 1024


@dataclass
class GenOutput:
    ok: bool
    files: List[str] = field(default_factory=list)   # absolutne sciezki zapisanych plikow
    seed: int = -1
    error: str = ""


def available_models() -> List[str]:
    """The Z-Image Turbo file of the active model set (the model set picks full or Lite files)."""
    model = engine_runtime.zimage_file("transformer")
    return [model.name] if model else []


def _resolve_seed(seed: int) -> int:
    return random.randint(0, 2 ** 31 - 1) if seed is None or seed < 0 else int(seed)


def _save_images(images: List[bytes], prefix: str, info: dict) -> List[str]:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved: List[str] = []
    for i, data in enumerate(images):
        path = config.OUTPUTS_DIR / f"{prefix}_{stamp}_{i}.png"
        path.write_bytes(data)
        metadata.embed_metadata(path, info)
        saved.append(str(path))
    return saved


def _friendly_error(exc: Exception, required_gb: float = 14.0) -> str:
    msg = str(exc)
    if any(h in msg.lower() for h in _OOM_HINTS) or _OOM_WORD.search(msg):
        return "Not enough GPU memory.\n" + low_vram_hint(required_gb)
    return f"Generation failed: {msg}"


def _finish(images: List[bytes], prefix: str, info: dict) -> GenOutput:
    """Save the pictures and write the history entry."""
    saved = _save_images(images, prefix, info)
    history.record(dict(info, files=saved))
    return GenOutput(ok=True, files=saved, seed=info.get("seed", -1))


def _run_engine(make, prefix: str, info: dict, required_gb: float = 14.0) -> GenOutput:
    """Generate PNG bytes, save them and record the result."""
    try:
        png = make()
    except Exception as e:  # noqa: BLE001 - the message goes to the UI (cancel.Cancelled is not an Exception)
        return GenOutput(ok=False, error=_friendly_error(e, required_gb))
    return _finish([png], prefix, info)


def txt2img(positive: str, width: int, height: int, steps: int, seed: int,
            progress_cb: Optional[ProgressCb] = None) -> GenOutput:
    """One Z-Image Turbo picture (it draws at CFG 1, without a negative prompt), saved and recorded."""
    if llm.blocked(positive):
        return GenOutput(ok=False, error=llm.BLOCK_MSG)
    models = available_models()
    if not models:
        return GenOutput(ok=False, error="Select Z-Image Turbo for the active model set.")
    seed = _resolve_seed(seed)
    info = {"mode": "txt2img", "model": models[0], "positive": positive, "width": width, "height": height,
            "steps": steps, "seed": seed}
    return _run_engine(lambda: engine_runtime.render(positive, width, height, seed, steps, progress_cb),
                       "img", info)


def crop_to(path: str, width: int, height: int, info: dict) -> None:
    """Przycina obraz do dokladnego rozmiaru (srodek) i odnawia metadane AI."""
    from PIL import Image

    with Image.open(path) as im:
        if (im.width, im.height) == (width, height):
            return
        left, top = (im.width - width) // 2, (im.height - height) // 2
        im.crop((left, top, left + width, top + height)).save(path)
    metadata.embed_metadata(path, dict(info, width=width, height=height))


def edit_with_refs(ref_list: List[tuple], positive: str, ratio: float,
                   progress_cb: Optional[ProgressCb] = None,
                   character: Optional[dict] = None,
                   cast: Optional[List[dict]] = None) -> GenOutput:
    """A picture composed by Qwen-Image-Edit-2511 from reference images [(number, path)] - up to 10, in
    passes - in the proportions `ratio` and at the size the model composes at (EDIT_PIXELS).
    positive - the description with numbers [n]. character - a character: first its scene composed from its
    primary reference photo (characters.render), which is the base of the composition (the main person); the
    reference images add things and places. Without reference images that scene is the picture.
    cast - two or more characters named in positive: each from its own photo, composed into the scene with the
    reference images (app/cast.py); reference images are optional then. The picture is Realistic, so an animal or
    creature character is refused (cast.creature_error, as the Image page's cast.check_prompt does) before any
    model work."""
    from PIL import Image

    from . import cast as cast_mod
    from . import refs
    from .video_gen import frame_size

    cast = [c for c in (cast or []) if c]
    refused = cast_mod.creature_error([*cast, character])
    if len(cast) < 2:
        cast = []
    if llm.blocked(positive):
        return GenOutput(ok=False, error=llm.BLOCK_MSG)
    if refused:
        return GenOutput(ok=False, error=refused)
    if not ref_list and not cast and not character:
        return GenOutput(ok=False, error="No reference images were provided.")
    err = model_manager.ensure([engine_runtime.QWEN_EDIT_ID], progress_cb)
    if err:
        return GenOutput(ok=False, error=err)
    client = engine_runtime.engine()
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    draft = config.OUTPUTS_DIR / f"edit_{stamp}_draft.png"
    out = config.OUTPUTS_DIR / f"img_{stamp}_0.png"
    seed = _resolve_seed(-1)
    gw, gh = frame_size(ratio, EDIT_PIXELS, 16)

    def ecb(frac: float, desc: str) -> None:
        if progress_cb:
            progress_cb(0.1 + 0.85 * frac, f"Composing references: {desc}")

    base = config.OUTPUTS_DIR / f"edit_{stamp}_character.png"
    try:
        looks = {n: llm.describe_person(path, ecb) for n, path in ref_list}
        start = None
        if cast:
            ok, msg = cast_mod.compose_frame(client, cast, positive, gw, gh, draft, seed, ecb,
                                             extra_refs=list(ref_list), looks=looks)
            if not ok:
                return GenOutput(ok=False, error=msg)
        elif character:
            from . import characters

            ok, msg = characters.render(client, character, positive, gw, gh, base, seed,
                                        lambda f, d: ecb(0.0, f"character {character['name']}..."))
            if not ok:
                return GenOutput(ok=False, error=msg)
            start = msg
            if not ref_list:  # the character's own scene is the picture
                Path(start).replace(draft)
        if not cast and ref_list:
            ok, msg = refs.compose_refs(client, start, ref_list, positive, gw, gh, draft, seed,
                                        progress_cb=ecb, looks=looks,
                                        base_lead=refs.CHARACTER if start else "")
            if not ok:
                return GenOutput(ok=False, error=msg)
        if not draft.is_file():
            return GenOutput(ok=False, error="The composition returned no image.")
        draft.replace(out)
    finally:
        draft.unlink(missing_ok=True)
        base.unlink(missing_ok=True)
    with Image.open(out) as picture:
        width, height = picture.size
    info = {"mode": "edit", "model": "Qwen-Image-Edit-2511", "positive": positive,
            "width": width, "height": height, "seed": seed, "ref_images": len(ref_list)}
    main = cast[0] if cast else character
    if main:
        info["character"] = main["name"]
    if cast:
        info["characters"] = [c["name"] for c in cast]
    metadata.embed_metadata(out, info)
    history.record(dict(info, files=[str(out)]))
    if progress_cb:
        progress_cb(1.0, "Done")
    return GenOutput(ok=True, files=[str(out)], seed=seed)
