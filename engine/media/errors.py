# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Errors of the media layer."""


class MediaError(RuntimeError):
    """A media file could not be read, written or processed. The message says what and why."""
