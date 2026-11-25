# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""What the Video and Image pages do on "Generate": check the prompt, the reference images and the characters, then add
the job to the queue. The pages and the Agent page both call this, so a clip or a picture the agent asks for is
checked exactly as one made by hand. Both kinds are made in the one realistic look of those pages (the film styles
belong to the Timeline)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import llm, queue_manager, refs
from . import cast as cast_mod


class Refused(ValueError):
    """The prompt or its inputs cannot be queued; str() says why in plain words."""

    def __init__(self, message: str, *, plain: bool = False):
        super().__init__(message)
        self.plain = plain  # the message already starts like a finished sentence of its own (the content rule)

    @property
    def shown(self) -> str:
        """The text a page shows for it."""
        return str(self) if self.plain else f"❌ {self}"


@dataclass
class Queued:
    job: queue_manager.Job
    found: list = field(default_factory=list)   # the characters that appear
    notes: list = field(default_factory=list)   # what the page tells about them


def _checked(prompt: str, ref_paths: Sequence[str]) -> None:
    if not (prompt or "").strip():
        raise Refused("Enter a prompt.")
    if llm.blocked(prompt):
        raise Refused(llm.BLOCK_MSG, plain=True)
    if len(ref_paths) > refs.MAX_REFS:
        raise Refused(f"Up to {refs.MAX_REFS} reference images are supported ({len(ref_paths)} provided).")
    missing = [n for n in refs.numbers(prompt) if n > len(ref_paths)]
    if missing:
        raise Refused(f"The prompt refers to image [{missing[0]}], but only {len(ref_paths)} reference images were "
                      "provided.")


def queue_video(prompt: str, ref_paths: Sequence[str], image: Optional[str], seconds, aspect: str, sound: bool,
                picked: Optional[Sequence[str]] = None) -> Queued:
    """Queue a clip (job kind "wideo"). `picked`: the characters chosen for it, None = the one chosen in the top bar."""
    ref_paths = list(ref_paths)
    _checked(prompt, ref_paths)
    chosen = cast_mod.picked() if picked is None else list(picked or [])
    shots = refs.split_shots(prompt)
    found, error, notes = cast_mod.check_prompt(prompt, chosen, shots)
    if error:
        raise Refused(error)
    params = {"prompt": prompt, "seconds": int(seconds), "aspect": aspect, "sound": bool(sound)}
    who = cast_mod.tag(params, found, shots)
    job = queue_manager.add("wideo", f"Video {int(seconds)} s{who}: {prompt}", params,
                            files={"refs": ref_paths, "image": image})
    return Queued(job, found, notes)


def queue_image(prompt: str, ref_paths: Sequence[str], format_name: str,
                picked: Optional[Sequence[str]] = None) -> Queued:
    """Queue a picture (job kind "obraz"); `format_name` is one of ui_images.SIZES."""
    ref_paths = list(ref_paths)
    _checked(prompt, ref_paths)
    chosen = cast_mod.picked() if picked is None else list(picked or [])
    found, error, notes = cast_mod.check_prompt(prompt, chosen)
    if error:
        raise Refused(error)
    params = {"prompt": prompt, "format": format_name}
    who = cast_mod.tag(params, found, [prompt])
    job = queue_manager.add("obraz", f"Image {format_name}{who}: {prompt}", params, files={"refs": ref_paths})
    return Queued(job, found, notes)
