# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The agent's tools for Contsy Stories: the series bible and the story memory of the episode the run works on
(app/series.py). A run that is not an episode's (a brief from the Agent page) gets NOT_FOUND from them."""
from __future__ import annotations

from typing import Optional

from . import series
from .agent_host import Host, ToolError
from .agent_tools import many, obj, text, tool


def bound(host: Host) -> Optional[dict]:
    """{"series", "episode"} of the episode this run belongs to, or None."""
    found = host.store.read().get("series")
    if isinstance(found, dict) and series.valid_id(found.get("series")) and series.valid_id(found.get("episode")):
        return found
    return None


def _require(host: Host) -> dict:
    found = bound(host)
    if found is None or series.get(found["series"]) is None:
        raise ToolError("NOT_FOUND", "This run does not belong to a series episode.")
    return found


def series_character(host: Host, name: str) -> Optional[dict]:
    """The sheet of the run's series whose name is `name` (case-insensitive), or None."""
    found = bound(host)
    data = series.get(found["series"]) if found else None
    if not data:
        return None
    return next((c for c in data["characters"] if c["name"].casefold() == (name or "").strip().casefold()), None)


def series_style(host: Host) -> str:
    """The engine style id of the run's series ("" when the run has no series)."""
    found = bound(host)
    data = series.get(found["series"]) if found else None
    return series.engine_style(data["look"]) if data else ""


@tool("get_series_bible",
      "The bible of the series this episode belongs to: style, cast with their design sheets and library slugs, "
      "locations and props, relationships, established facts, the story so far and the summaries of earlier "
      "episodes. Read it again instead of guessing when you are unsure about continuity.", obj({}))
def _get_series_bible(host: Host, args: dict):
    found = _require(host)
    return {"bible": series.story_context(found["series"], found["episode"])}


@tool("save_episode_memory",
      "Save the story memory of this episode before finish_run: what happened (a short summary), the facts that later "
      "episodes must respect (who met whom, who lost or got what, what changed in the world), and the story state "
      "after it. An owner's own corrections are never overwritten.",
      obj({"summary": text("What happened in this episode, in 2-8 sentences.", max_length=series.MAX_SUMMARY,
                           min_length=10),
           "facts": many("New facts that stay true after this episode.", text("One fact.", max_length=300),
                         max_items=20),
           "story_state": text("Where the story stands now, in 1-4 sentences.", max_length=1500)},
          ("summary",)))
def _save_episode_memory(host: Host, args: dict):
    found = _require(host)
    sid, eid = found["series"], found["episode"]
    try:
        episode = series.set_summary(sid, eid, args["summary"], source="auto")
        for fact in args.get("facts") or []:
            series.add_fact(sid, fact, episode=eid, source="auto")
        state = (args.get("story_state") or "").strip()
        current = series.get(sid)
        if state and current and current["story_state_source"] != "owner":
            with series._LOCK:  # the agent's state; an owner's state stays until the owner changes it
                data = series.get(sid)
                data["story_state"], data["story_state_source"] = state[:series.MAX_TEXT], "auto"
                series._save(data)
    except series.SeriesError as error:
        raise ToolError("INVALID", str(error)) from None
    kept = episode["summary_source"] == "owner"
    return {"saved": not kept, "note": "The owner's own summary is kept." if kept else "Saved."}
