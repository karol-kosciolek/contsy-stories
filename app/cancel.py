# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Przerywanie biezacego zadania ("Stop current job" na stronie GPU Queue, "Pause and release GPU"
na gornym pasku).

Kolejka, potoki silnika i procesy robocze sprawdzaja ten sam token miedzy krokami.
Cancelled przechodzi przez zwykle "except Exception" az do kolejki, a bloki finally
sprzataja pliki tymczasowe i oddaja pamiec.
"""
from __future__ import annotations

from engine.core.cancel import Cancelled, CancelToken  # noqa: F401  (Cancelled: zadanie przerwane przez uzytkownika)

TOKEN = CancelToken()


def reset() -> None:
    TOKEN.reset()


def requested() -> bool:
    return TOKEN.cancelled


def check() -> None:
    TOKEN.check()


def request() -> None:
    """Przerywa biezace zadanie."""
    TOKEN.cancel()
