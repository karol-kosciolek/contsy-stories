# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""CPU face landmarks, video tracking and identity embeddings."""

from .recognition import cosine_similarity, face_embeddings, load_sface, load_yunet
from .tracks import find_faces

__all__ = ["cosine_similarity", "face_embeddings", "find_faces", "load_sface", "load_yunet"]
