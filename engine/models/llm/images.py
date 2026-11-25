# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Pictures for the vision model: shrunk before the model sees them ("several times faster, same answer").

The EXIF orientation is applied, the picture becomes RGB, its longest side is reduced to `max_side` (never enlarged,
the aspect ratio is kept) and it is saved as JPEG at quality 92. The sizes the product uses are 448 px for the check
of a character's reference photo, 512 px for the description of a person and 512 or 768 px for the Timeline's join
review.
The bytes travel to the worker process and from there, as a data URL, to the multimodal chat handler. No temporary
file is written.
"""
from __future__ import annotations

import io
import os
from typing import Any

from .errors import LlmError

MIN_SIDE, MAX_SIDE = 64, 4096
JPEG_QUALITY = 92

ImageSource = Any  # a path, the bytes of an encoded image, or a PIL image


def prepare_image(source: ImageSource, max_side: int) -> bytes:
    """JPEG bytes of `source`, upright, RGB, with its longest side at most `max_side` pixels."""
    if not MIN_SIDE <= int(max_side) <= MAX_SIDE:
        raise LlmError(f"The image size {max_side} is outside {MIN_SIDE}-{MAX_SIDE} pixels.")
    from PIL import Image, ImageOps

    try:
        if isinstance(source, Image.Image):  # the caller's picture is only read: exif_transpose and convert copy it
            image = ImageOps.exif_transpose(source).convert("RGB")
        elif isinstance(source, (bytes, bytearray, memoryview, str, os.PathLike)):
            data = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray, memoryview)) else source
            with Image.open(data) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        else:
            raise LlmError(f"Cannot use {type(source).__name__} as an image.")
        image.thumbnail((int(max_side), int(max_side)))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    except LlmError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise LlmError(f"Could not read the image: {exc}") from None
    return buffer.getvalue()
