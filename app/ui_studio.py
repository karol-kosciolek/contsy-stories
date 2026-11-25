# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Dashboard for the active cast, queue and recent results."""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence

import gradio as gr

from . import cast, characters, queue_manager, studio, ui_queue, universes
from .ui_theme import badge, cache_media, esc, head_html, kpis, section, table

_STATUS = {"Ready": ("Ready", "ok")}  # characters.state(); any other state needs the user's attention


def _ago(ts: float) -> str:
    if not ts:
        return "—"
    secs = time.time() - ts
    if secs < 3600:
        return f"{max(int(secs // 60), 1)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return time.strftime("%d.%m %H:%M", time.localtime(ts))


def _fmt(secs) -> str:
    return queue_manager.fmt_time(secs)


def _profile(profile: str) -> str:
    """The character a page shows for the top picker's value (a universe: its main character)."""
    if profile and profile.startswith(cast.UNIVERSE):
        ready = universes.members(universes.get(profile[len(cast.UNIVERSE):]))
        return ready[0]["slug"] if ready else ""
    return profile if profile and characters.usable(profile) else ""


def _members(profile: str) -> List[str]:
    """Every character the Dashboard shows for the top picker's value (a universe: all members)."""
    if profile and profile.startswith(cast.UNIVERSE):
        u = universes.get(profile[len(cast.UNIVERSE):])
        return [ch["slug"] for ch in universes.members(u)] or (u or {}).get("members", [])
    single = _profile(profile)
    return [single] if single else []


# ----------------------------- Pulpit -----------------------------

def _universe_head(value: str, members: List[str]) -> str:
    """The Dashboard heading of a universe: its ready characters, those still without reference photos, and the
    members that no longer exist (deleted or archived), which are named as missing, not as nameless characters."""
    u = universes.get(value[len(cast.UNIVERSE):]) or {"name": "", "members": []}
    ready = [s for s in members if characters.usable(s)]
    names = ", ".join(esc(studio.name(s)) for s in ready) or "no characters"
    gone = set(characters.missing(u.get("members", [])))
    waiting = [s for s in u.get("members", []) if s not in ready and s not in gone]
    later = (f" Without reference photos: {', '.join(esc(studio.name(s)) for s in waiting)}." if waiting else "")
    later += f" {esc(universes.notice(u))}" if gone else ""
    return head_html("Dashboard", f"Universe <b>{esc(u['name'])}</b>: {names}.{later} Gallery and "
                                  "GPU Queue show this group; its first character is the main one "
                                  "for new jobs. Select “All characters” in the top bar’s Tools to see "
                                  "everything.")


def _dash_head(profile: str) -> str:
    sub = (f"Character <b>{esc(studio.name(profile))}</b>. Gallery and GPU Queue also show only "
           "this character. Select “All characters” in the top bar’s Tools to see everything." if profile else
           "All characters. Click a card to open a character profile.")
    return head_html("Dashboard", sub)


def _kpis(profile: str, rows: List[dict], members: Sequence[str] = (),
          universe: bool = False) -> str:
    """Summarize the queue and results for the selected cast."""
    snap = queue_manager.snapshot()
    run = next((j for j in snap if j.status == queue_manager.RUNNING), None)
    pending = [j for j in snap if j.status == queue_manager.PENDING]
    if profile:
        mine = [j for j in studio.jobs(list(members) or profile, snap)
                if j.status in (queue_manager.PENDING, queue_manager.RUNNING)]
        eta = sum((j.eta() or 0.0) if j.status == queue_manager.RUNNING else j.estimate()
                  for j in mine)
        queue = {"label": "Universe jobs" if universe else "Character jobs", "value": str(len(mine)),
                 "sub": f"about {_fmt(eta)} · entire queue: {len(pending) + bool(run)}"}
    else:
        total = len(pending) + bool(run)
        queue = {"label": "Queued", "value": str(total),
                 "sub": f"estimated time remaining {_fmt(queue_manager.remaining_seconds())}" if total
                 else "queue empty"}
    if queue_manager.paused():
        now = {"label": "Now", "value": "Paused", "sub": "GPU released — no generation running"}
    elif run:
        now = {"label": f"Now · #{run.id}",
               "value": esc(ui_queue.short(run.label, 80)), "small": True,
               "sub": f"{int(run.progress * 100)}% · estimated time remaining {_fmt(run.eta())}",
               "progress": run.progress}
    else:
        now = {"label": "Now", "value": "Idle", "sub": "GPU waiting for jobs"}
    today, total = sum(r["today"] for r in rows), sum(r["total"] for r in rows)
    return kpis([queue, now, {"label": "Results today", "value": str(today), "sub": f"total: {total}"}])


def _table(rows: List[dict]) -> str:
    out = []
    for r in rows:
        if r["slug"]:
            text, kind = _STATUS.get(r["status"], (r["status"] or "—", "warn" if r["status"] else ""))
            state = badge(text, kind)
            name = f"<b>{esc(r['name'])}</b>"
        else:
            state, name = "<span class='muted'>—</span>", esc(r["name"])
        queued = (f"{r['queued']}" + (" · running" if r["running"] else "")
                  + (f" <span class='muted'>(about {_fmt(r['eta'])})</span>" if r["queued"] else ""))
        out.append([name, state, queued, f"<span class='num'>{r['today']}</span>",
                    f"<span class='num'>{r['total']}</span>",
                    f"<span class='muted'>{esc(_ago(r['last']))}</span>",
                    f"<span class='muted'>{esc((r['note'] or '—')[:90])}</span>"])
    return table(["Character", "Status", "Queued", "Today", "Total", "Latest result",
                  "Notes"], out, empty="<b>No characters</b>Add a character on the Characters page.")


def _mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def build_dashboard(tab: gr.Tab, character: gr.Dropdown):
    """character - wybor postaci w panelu Tools na gornym pasku (to on jest profilem). Zwraca (funkcja,
    wejscia, wyjscia) do odswiezenia przy otwarciu strony."""
    head = gr.HTML(_dash_head(""), padding=False)
    stats = gr.HTML(padding=False)
    with gr.Column(elem_classes=["card"]):
        with gr.Row(elem_classes=["toolbar"]):
            with gr.Column(elem_classes=["grow"]):
                section("Characters", "Click a card to open a character profile.")
            show_all = gr.Button("Show all", size="sm", visible=False)
            refresh_btn = gr.Button("Refresh", size="sm")
        tiles = gr.Gallery(show_label=False, columns=6, height="auto", object_fit="cover",
                           allow_preview=False, elem_id="studio-tiles", buttons=[])
        rows_html = gr.HTML()
    with gr.Column(elem_classes=["card"]):
        section("Latest results", "Browse all results in Gallery.")
        recent = gr.Gallery(show_label=False, columns=6, height=360, object_fit="cover",
                            allow_preview=True, elem_id="studio-recent", buttons=["download"])
    slugs = gr.State([])
    rows_st = gr.State([])  # wiersze z ostatniego odswiezenia - dla timera wskaznikow
    outs = [head, stats, tiles, rows_html, recent, slugs, show_all, rows_st]

    def refresh(value):
        members = _members(value)
        profile = _profile(value)
        rows = studio.overview()
        if members:
            rows = [r for r in rows if r["slug"] in members]
        else:  # "bez postaci" tylko, gdy cos ma
            rows = [r for r in rows if r["slug"] or r["queued"] or r["total"]]
        cards, card_slugs = [], []
        for row in rows:
            path = cache_media(row["preview"])
            if path:
                cards.append((path, row["name"]))
                card_slugs.append(row["slug"])
        recent_v = [(f, time.strftime("%d.%m %H:%M", time.localtime(_mtime(f))))
                    for f in studio.results(members, limit=12)]
        universe = bool(value) and value.startswith(cast.UNIVERSE)
        head = _universe_head(value, members) if universe else _dash_head(profile)
        return (head, _kpis(profile, rows, members, universe), cards, _table(rows), recent_v,
                card_slugs, gr.update(visible=bool(members)), rows)

    def pick(slugs_v, evt: gr.SelectData):
        if not (0 <= evt.index < len(slugs_v)) or not characters.usable(slugs_v[evt.index]):
            return (gr.skip(),) * (len(outs) + 1)
        slug = slugs_v[evt.index]
        cast.select(slug)  # profil = modelka do generowania (jak wybor w panelu)
        return (gr.update(choices=cast.top_picker_choices(), value=slug), *refresh(slug))

    def everyone():
        cast.select("")
        return (gr.update(choices=cast.top_picker_choices(), value=""), *refresh(""))

    tab.select(refresh, inputs=character, outputs=outs)
    refresh_btn.click(refresh, inputs=character, outputs=outs)
    character.change(refresh, inputs=character, outputs=outs)
    tiles.select(pick, inputs=slugs, outputs=[character] + outs)
    show_all.click(everyone, outputs=[character] + outs)
    # biezace zadanie i kolejka co kilka sekund; wyniki (historia) - przy odswiezeniu strony
    def _tick(p, rows):
        universe = bool(p) and p.startswith(cast.UNIVERSE)
        return _kpis(_profile(p), rows or [], _members(p), universe)

    gr.Timer(5.0).tick(_tick, inputs=[character, rows_st], outputs=stats)
    return refresh, [character], outs
