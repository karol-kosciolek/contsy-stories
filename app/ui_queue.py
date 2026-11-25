# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Strona 'GPU Queue' i wspolne elementy kolejki dla stron Video / Image.

Biezace zadanie z paskiem postepu i przyciskami "Stop current job" / "Pause after current job",
lista oczekujacych z zaznaczaniem (usun zaznaczone, wyczysc kolejke) i lista zakonczonych.
Odswiezanie co 2 s; liste oczekujacych przebudowujemy tylko, gdy sie zmienila - inaczej znikalyby
zaznaczenia. Przy wybranej postaci (panel Tools na gornym pasku: "Character or universe") lista,
zakonczone i "Clear pending jobs" dotycza tylko jej zadan (app/studio.py); biezace zadanie widac
zawsze - karta jest jedna.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import gradio as gr

from . import queue_manager as qm
from . import studio
from .ui_theme import badge, bar, esc, note, page_head, plain, section, table

KINDS = {"wideo": "Video", "obraz": "Image", "character_photos": "Character photos",
         "character_voice": "Character voice", "item_photos": "Place or item photos", **qm.REMOVED_KINDS,
         "timeline_prepare": "Timeline generation", "timeline_merge": "Timeline join",
         "timeline_render": "Timeline export", "timeline_sound": "Timeline sound preview",
         "timeline_produce": "Timeline movie export", "timeline_edit": "Timeline AI edit"}
STATUS = {qm.DONE: ("Complete", "ok"), qm.ERROR: ("Failed", "err"), qm.STOPPED: ("Stopped", "warn"),
          qm.PENDING: ("Queued", ""), qm.RUNNING: ("Running", "info")}

def short(text: str, n: int = 70) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def added_message(job: qm.Job) -> str:
    ahead = qm.position(job.id)
    where = "starting now" if ahead == 0 else f"{ahead} job{'' if ahead == 1 else 's'} ahead"
    return note(f"➕ Queued as **#{job.id}** ({where}). You can add more jobs immediately. "
                "The result will appear here; track progress in **GPU Queue**.")


def finished_message(job: qm.Job) -> str:
    if job.status == qm.DONE:
        head = f"✅ #{job.id} complete ({qm.fmt_time(job.elapsed())})."
    elif job.status == qm.STOPPED:
        head = f"⏹ #{job.id} stopped."
    else:
        head = f"❌ #{job.id}: {job.error}"
    return note(head + ("\n\n" + "\n".join(f"- {n}" for n in job.notes)
                        if job.notes else ""))


def result_poller(kind: str):
    """Funkcja dla gr.Timer: najnowszy zakonczony wynik danego rodzaju (tylko gdy nowy)."""
    def tick(last_id):
        job = qm.latest_finished(kind)
        if job is None or job.id == last_id:
            return gr.update(), gr.update(), last_id
        files = [f for f in job.files if Path(f).exists()]
        # failed, stopped, or its files were deleted since: the previous result stays on the page
        if job.status != qm.DONE or not files:
            return gr.update(), finished_message(job), job.id
        return (files if kind == "obraz" else files[0]), finished_message(job), job.id
    return tick


def _current_html() -> str:
    run = next((j for j in qm.snapshot() if j.status == qm.RUNNING), None)
    if run is None:
        text = "Queue paused — no generation is running." if qm.paused() else \
            "No generation is running."
        return f"<div class='job idle'>{text}</div>"
    pct = int(run.progress * 100)
    return (f"<div class='job'><div class='j-title'>{badge(KINDS.get(run.kind, run.kind), 'info')} "
            f"#{run.id} · {esc(short(run.label, 90))}</div>{bar(run.progress)}"
            f"<div class='j-meta'>{pct}% · {esc(run.desc or 'start…')} · elapsed "
            f"{qm.fmt_time(run.elapsed())} · about {qm.fmt_time(run.eta())} remaining</div></div>")


def _pending_choices():
    jobs = [j for j in studio.jobs(studio.current_set()) if j.status in (qm.RUNNING, qm.PENDING)]
    # nazwa zadania mowi juz, co to jest ("Video 6 s: ...") - bez dopisku rodzaju
    return [(f"#{j.id} · {short(j.label, 90)}"
             + (" — running (removing will stop it)" if j.status == qm.RUNNING else ""), str(j.id))
            for j in jobs]


def _summary() -> str:
    snap = qm.snapshot()
    pending = [j for j in snap if j.status == qm.PENDING]
    paused = " · **paused**" if qm.paused() else ""
    profile = studio.current()
    from . import cast

    universe = cast.current_universe()  # even with no member ready: the list shows the group
    if profile or universe:
        mine = [j for j in studio.jobs(studio.current_set(), snap)
                if j.status in (qm.PENDING, qm.RUNNING)]
        eta = sum((j.eta() or 0.0) if j.status == qm.RUNNING else j.estimate() for j in mine)
        who = (f"Universe **{universe['name']}**" if universe
               else f"Character **{studio.name(profile)}**")
        return (f"{who} — **{len(mine)}** job{'' if len(mine) == 1 else 's'}, about "
                f"**{qm.fmt_time(eta)}** · entire queue: {len(pending)}, about "
                f"{qm.fmt_time(qm.remaining_seconds())}{paused}")
    if not pending and not any(j.status == qm.RUNNING for j in snap):
        return "The queue is empty. Each click of Generate adds a job here."
    return (f"Queued: **{len(pending)}** · estimated total time remaining "
            f"**{qm.fmt_time(qm.remaining_seconds())}**{paused}")


def _finished_html() -> str:
    done = [j for j in studio.jobs(studio.current_set())
            if j.status in (qm.DONE, qm.ERROR, qm.STOPPED)]
    rows = []
    for j in sorted(done, key=lambda j: -(j.finished or 0))[:30]:
        text, kind = STATUS[j.status]
        state = badge(text, kind) + (f" <span class='muted'>{esc(short(plain(j.error), 70))}</span>"
                                     if j.status == qm.ERROR else "")
        file = esc(Path(j.files[0]).name) if j.files else "—"
        rows.append([f"<span class='muted num'>{j.id}</span>",
                     esc(short(j.label, 80)),
                     state, f"<span class='num'>{qm.fmt_time(j.elapsed())}</span>",
                     f"<span class='muted'>{file}</span>"])
    return table(["#", "Job", "Status", "Time", "File"], rows,
                 empty="Completed jobs will appear here.")


def _pause_label() -> str:
    return "Resume queue" if qm.paused() else "Pause after current job"


def build_queue_tab(tab: gr.Tab) -> None:
    """tab - strona Kolejka (odswiezenie przy kazdym wejsciu)."""
    page_head("GPU Queue", "Jobs run one at a time on one GPU.")
    current = gr.HTML(_current_html(), padding=False)
    with gr.Row(elem_classes=["toolbar"]):
        stop = gr.Button("Stop current job", variant="stop", size="sm")
        pause = gr.Button(_pause_label(), size="sm")
        summary = gr.Markdown(_summary(), elem_classes=["hint", "grow"])
    with gr.Column(elem_classes=["card"]):
        section("Pending jobs", "Select jobs to remove them.")
        pending = gr.CheckboxGroup(_pending_choices(), show_label=False, elem_id="queue-list")
        with gr.Row(elem_classes=["toolbar"]):
            delete = gr.Button("Remove selected", size="sm")
            select_all = gr.Button("Select all", size="sm")
            clear = gr.Button("Clear pending jobs", size="sm", variant="stop")
    with gr.Column(elem_classes=["card"]):
        with gr.Row(elem_classes=["toolbar"]):
            with gr.Column(elem_classes=["grow"]):
                section("Finished jobs", "The latest 30 jobs.")
            clear_done = gr.Button("Clear finished jobs", size="sm")
        done = gr.HTML(_finished_html())
    shown = gr.State([label for label, _ in _pending_choices()])

    def refresh(selected: Optional[List[str]], shown_labels: Optional[List[str]]):
        """shown_labels=None wymusza przebudowe listy (po usunieciu / wyczyszczeniu)."""
        choices = _pending_choices()
        labels = [label for label, _ in choices]
        if labels == shown_labels:  # bez zmian - nie ruszamy listy (zaznaczenia zostaja)
            pend = gr.update()
        else:
            ids = [v for _, v in choices]
            pend = gr.update(choices=choices, value=[v for v in (selected or []) if v in ids])
        run = next((j for j in qm.snapshot() if j.status == qm.RUNNING), None)
        return (_current_html(), _summary(), pend, _finished_html(), _pause_label(), labels,
                run.id if run else None)

    running = gr.State(None)  # zadanie widoczne na karcie - tylko je zatrzymuje "Stop current job"
    outs = [current, summary, pending, done, pause, shown, running]
    gr.Timer(2.0).tick(refresh, inputs=[pending, shown], outputs=outs)
    # Timer rusza dopiero z pierwszym otwarciem strony - bez tego przez 2 s widac stan ze startu
    tab.select(refresh, inputs=[pending, shown], outputs=outs)

    def _stop(selected, shown_ids, running_id):
        # tylko zadanie, ktore uzytkownik widzial: drugie klikniecie (gdy wideo jeszcze sie
        # zatrzymywalo) przerywalo nastepne zadanie, ktore wlasnie ruszylo
        if running_id is not None:
            qm.stop_current(running_id)
        return refresh(selected, shown_ids)

    def _delete(selected):
        qm.remove([int(v) for v in selected or []])
        return refresh([], None)

    def _clear():
        # tylko to, co widac - przy wybranej postaci jej zadania, inne zostaja
        qm.remove([j.id for j in studio.jobs(studio.current_set()) if j.status == qm.PENDING])
        return refresh([], None)

    def _toggle(selected, shown_ids):
        qm.set_paused(not qm.paused())
        return refresh(selected, shown_ids)

    def _clear_done(selected, shown_ids):
        qm.clear_finished()
        return refresh(selected, shown_ids)

    stop.click(_stop, inputs=[pending, shown, running], outputs=outs)
    delete.click(_delete, inputs=pending, outputs=outs)
    clear.click(_clear, outputs=outs)
    for btn, fn in ((pause, _toggle), (clear_done, _clear_done)):
        btn.click(fn, inputs=[pending, shown], outputs=outs)
    select_all.click(lambda: [v for _, v in _pending_choices()], outputs=pending)
