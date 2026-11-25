# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Contsy Stories pages (docs/CONTSY_STORIES_PRODUCT.md): the Dashboard with the first-series wizard, and the
Series workspace (Series Bible, Characters, Locations & Props, Episodes, Editor). The data is app/series.py; the
episodes are written by the agent (app/agent_runner.start_episode / follow_up) and made in the Timeline (the Editor).
"""
from __future__ import annotations

import time
from typing import Optional

import gradio as gr

from . import agent_config, agent_runner, agent_store, characters, queue_manager, series, timeline
from . import ui_theme, voices
from .ui_theme import esc

NEW = "__new__"
NONE = ""
STATUS_LABELS = {"draft": ("Draft", ""), "writing": ("Writing storyboard…", "info"),
                 "storyboard": ("Storyboard ready", "ok"), "approved": ("Approved", "info"),
                 "producing": ("Generating…", "info"), "done": ("Done", "ok"), "failed": ("Needs attention", "err")}
HANDLES: dict = {}  # components other pages use (the series picker, the episode picker)


def go_js(page: str) -> str:
    """Browser code that opens a page of the app (the Tabs of app/main.py)."""
    return ("(...args) => { const t = document.querySelector('#pages > .tab-wrapper [role=tab][data-tab-id=\"" + page
            + "\"]'); if (t && t.getAttribute('aria-selected') !== 'true') t.click(); "
            "window.scrollTo({top: 0}); return args; }")


def _agent_ready() -> tuple[bool, str]:
    key, _ = agent_config.get_key()
    if not key:
        return False, ("The story agent needs your Claude API key (Agent page → Connection); until then storyboards "
                       "are built by hand in the Editor.")
    return True, ""


def _queue_html() -> str:
    """The GPU queue card of the Dashboard: how many jobs wait or run, and about how long they still take."""
    busy = len(_my_jobs())
    text = f"{busy} job{'s' if busy != 1 else ''}" if busy else "Idle"
    sub = (f"about {queue_manager.fmt_time(queue_manager.remaining_seconds())} left" if busy
           else "nothing is generating")
    return (f"<div class='card story-stat'><div class='stat-label'>GPU queue</div>"
            f"<div class='stat-value'>{esc(text)}</div><div class='stat-sub'>{esc(sub)}</div></div>")


def _my_jobs() -> list:
    return [job for job in queue_manager.snapshot() if job.status in (queue_manager.PENDING, queue_manager.RUNNING)]


def _jobs_html() -> str:
    jobs = _my_jobs()
    if not jobs:
        return "<div class='empty'>Nothing is generating right now.</div>"
    rows = []
    for job in jobs[:12]:
        state = "Generating" if job.status == queue_manager.RUNNING else f"Waiting ({queue_manager.position(job.id)} ahead)"
        rows.append([esc(job.label[:70]), ui_theme.badge(state, "info"),
                     ui_theme.bar(job.progress) if job.status == queue_manager.RUNNING else "",
                     esc(queue_manager.fmt_time(job.estimate()))])
    return ui_theme.table(["Job", "State", "Progress", "Est. GPU time"], rows)


def _series_cards() -> str:
    found = series.all_series()
    if not found:
        return "<div class='empty'>No series yet. Create your first animated series below.</div>"
    rows = []
    for item in found:
        eps = series.episodes(item["id"])
        rows.append([f"<b>{esc(item['name'])}</b><br><span class='muted'>{esc(item['logline'][:120])}</span>",
                     esc(series.look_label(item["look"])), esc(str(len(item["characters"]))), esc(str(len(eps))),
                     esc(time.strftime("%Y-%m-%d", time.localtime(item["updated"] or time.time())))])
    return ui_theme.table(["Series", "Style", "Characters", "Episodes", "Updated"], rows)


def _recent_episodes_html() -> str:
    rows = []
    for item in series.all_series():
        for episode in series.episodes(item["id"]):
            rows.append((episode["updated"], item, episode))
    rows.sort(key=lambda row: -row[0])
    out = []
    for _, item, episode in rows[:8]:
        label, kind = STATUS_LABELS[episode["status"]]
        out.append([esc(item["name"]), esc(f"S{episode['season']}E{episode['number']} {episode['title']}"),
                    ui_theme.badge(label, kind), esc(f"{episode['seconds']} s · {episode['format']}")])
    return ui_theme.table(["Series", "Episode", "Status", "Length"], out, "No episodes yet.")


def _series_choices():
    return [(item["name"], item["id"]) for item in series.all_series()]


def _voice_choices():
    return [("Choose later", "")] + voices.choices()


# --- the Dashboard ---------------------------------------------------------------------------------------------------

def build_home(tab: gr.Tab):
    """The Dashboard: the GPU queue, Current Generations, My Series, Recent Episodes and Create New Series (the
    wizard). Returns (load fn, outputs) for the page load."""
    ui_theme.page_head("Contsy Stories", "Create animated characters, build entire episodes, and produce your own "
                                         "original cartoon series.")
    with gr.Row(equal_height=True):
        queue_box = gr.HTML(_queue_html(), padding=False)
        jobs_box = gr.HTML(_jobs_html(), padding=False, elem_classes=["card"])
    ui_theme.section("My Series")
    series_box = gr.HTML(_series_cards(), padding=False)
    with gr.Row():
        open_pick = gr.Dropdown(allow_custom_value=True, choices=_series_choices(), label="Open a series", value=None,
                                scale=3)
        open_btn = gr.Button("Open", variant="primary", scale=1)
    ui_theme.section("Recent Episodes")
    recent_box = gr.HTML(_recent_episodes_html(), padding=False)

    with gr.Accordion("Create New Series", open=not series.all_series(), elem_id="series-wizard") as wizard:
        gr.Markdown("### Create your first animated series\nSeven short steps; you can change everything later.")
        w_name = gr.Textbox(label="1. Name your series", max_length=series.MAX_NAME,
                            placeholder="e.g. Pip and the Lantern Forest")
        w_logline = gr.Textbox(label="What is it about? (optional)", lines=2, max_length=600,
                               placeholder="A curious fox and his robot friend solve small mysteries in a forest town.")
        w_look = gr.Radio(series.look_choices(), value=series.DEFAULT_LOOK, label="2. Choose animation style")
        gr.Markdown("**3. Create your main character** (an animal, a creature, a robot… original, not from an existing "
                    "show)")
        with gr.Row():
            w_char = gr.Textbox(label="Name", max_length=series.MAX_NAME, placeholder="Pip")
            w_kind = gr.Radio([("Animal / creature / robot", "creature"), ("Person (adult)", "person")],
                              value="creature", label="Kind")
        w_desc = gr.Textbox(label="What it looks like", lines=2, max_length=600,
                            placeholder="A small fox with a big fluffy tail and round glasses")
        with gr.Row():
            w_colors = gr.Textbox(label="Signature colors", max_length=200, placeholder="orange fur, white chest")
            w_outfit = gr.Textbox(label="Outfit", max_length=200, placeholder="green scarf")
        w_personality = gr.Textbox(label="Personality", max_length=400, placeholder="curious, brave, a bit clumsy")
        w_voice = gr.Dropdown(allow_custom_value=True, choices=_voice_choices(), value="", label="4. Select a voice")
        w_idea = gr.Textbox(label="5. Describe your first episode", lines=3, max_length=2000,
                            placeholder="Pip finds a lantern that glows only when someone tells the truth.")
        with gr.Row():
            w_format = gr.Radio([("YouTube 16:9", "16:9"), ("YouTube Shorts 9:16", "9:16")], value="16:9",
                                label="Format")
            w_seconds = gr.Slider(series.MIN_EPISODE_SECONDS, series.MAX_EPISODE_SECONDS, value=60, step=15,
                                  label="Episode length (seconds)")
            w_narration = gr.Radio([("Dialogue", "dialogue"), ("Narrator", "narrator"), ("Both", "both")],
                                   value="dialogue", label="Narration")
        gr.Markdown("**6. Review the storyboard** and **7. Generate** happen on the series' Episodes tab: the agent "
                    "writes the storyboard first, nothing is generated before you approve it.")
        w_create = gr.Button("Create series and write the storyboard", variant="primary")
        w_msg = gr.Markdown()

    def _create(name, logline, look, char, kind, desc, colors, outfit, personality, voice, idea, fmt, seconds,
                narration):
        try:
            created = series.create(name, look=look, logline=logline, fmt=fmt, narration=narration)
            if (char or "").strip():
                series.save_character(created["id"], {"name": char, "kind": kind, "description": desc,
                                                      "colors": colors, "outfit": outfit,
                                                      "personality": personality, "voice": voice, "role": "main"})
            message = f"✅ Created **{created['name']}**."
            if (idea or "").strip():
                episode = series.create_episode(created["id"], idea, seconds=int(seconds), fmt=fmt,
                                                narration=narration)
                ready, why = _agent_ready()
                if ready:
                    try:
                        agent_runner.start_episode(created["id"], episode["id"])
                        message += " The agent is writing the storyboard of episode 1; review it on the Episodes tab."
                    except agent_runner.RunnerError as error:
                        message += f" ⚠ The storyboard could not start: {error}"
                else:
                    _manual_storyboard(created["id"], episode["id"])
                    message += f" {why} An empty storyboard was made for you."
        except series.SeriesError as error:
            return (ui_theme.note(f"❌ {error}"), gr.skip(), gr.skip(), gr.skip(), gr.skip())
        return (ui_theme.note(message), _series_cards(), gr.update(choices=_series_choices(), value=created["id"]),
                _recent_episodes_html(), gr.update(open=False))

    w_create.click(_create, inputs=[w_name, w_logline, w_look, w_char, w_kind, w_desc, w_colors, w_outfit,
                                    w_personality, w_voice, w_idea, w_format, w_seconds, w_narration],
                   outputs=[w_msg, series_box, open_pick, recent_box, wizard])

    def _refresh():
        return _queue_html(), _jobs_html(), _series_cards(), _recent_episodes_html()

    gr.Timer(8.0).tick(_refresh, outputs=[queue_box, jobs_box, series_box, recent_box], show_progress="hidden")
    HANDLES["home_open"] = (open_pick, open_btn)

    def _load():
        return (*_refresh(), gr.update(choices=_series_choices()), gr.update(open=not series.all_series()),
                gr.update(choices=_voice_choices()))

    return _load, [queue_box, jobs_box, series_box, recent_box, open_pick, wizard, w_voice]


def _manual_storyboard(series_id: str, episode_id: str) -> str:
    """Without the agent: a Timeline project in the series' style and format with the series characters and one
    scene holding the idea, to build by hand in the Editor."""
    data = series.get(series_id)
    episode = series.get_episode(series_id, episode_id)
    cast = [c["slug"] for c in data["characters"] if c.get("slug") and characters.get(c["slug"])][:12]
    scene = timeline.Scene(prompt=episode["idea"][:900], seconds=5)
    project = timeline.save(None, None, f"{episode['title']} — {data['name']}"[:120], (scene,),
                            series.FORMATS[episode["format"]], True, cast, style=series.engine_style(data["look"]))
    series.update_episode(series_id, episode_id, projects=[project.id], status="storyboard")
    return project.id


# --- the Series workspace ------------------------------------------------------------------------------------------

def _bible_values(data: Optional[dict]):
    data = data or {}
    return [data.get(k, "") for k in ("name", "logline", "world", "rules", "tone", "narration_style", "story_state")] + [
        data.get("look") or series.DEFAULT_LOOK, data.get("format") or "16:9", data.get("narration") or "dialogue",
        data.get("audience") or "family", data.get("narrator_voice") or ""]


def _facts_html(data: Optional[dict]) -> str:
    if not data or not data["facts"]:
        return "<div class='empty'>No facts yet. The agent adds them after each episode; you can add and correct them.</div>"
    rows = [[esc(f["text"]), ui_theme.badge("owner" if f["source"] == "owner" else "auto",
                                            "ok" if f["source"] == "owner" else "info"),
             ui_theme.badge("active" if f.get("active", True) else "retired", "" if f.get("active", True) else "warn")]
            for f in data["facts"]]
    return ui_theme.table(["Fact", "Source", "State"], rows)


def _fact_choices(data):
    return [(f["text"][:80], f["id"]) for f in (data or {}).get("facts", [])]


def _cast_html(data: Optional[dict]) -> str:
    if not data or not data["characters"]:
        return "<div class='empty'>No characters yet.</div>"
    rows = []
    for c in data["characters"]:
        made = characters.get(c["slug"]) if c.get("slug") else None
        state = ui_theme.badge("reference pictures ready", "ok") if made else ui_theme.badge("design sheet only", "warn")
        voice = (made or {}).get("voice") or c.get("voice") or "—"
        rows.append([f"<b>{esc(c['name'])}</b><br><span class='muted'>{esc(c.get('role', ''))}</span>",
                     esc(c.get("description", "")[:140]), esc(c.get("colors", "")), esc(voice), state])
    return ui_theme.table(["Character", "Look", "Colors", "Voice", "State"], rows)


def _char_choices(data):
    return [("New character", NEW)] + [(c["name"], c["id"]) for c in (data or {}).get("characters", [])]


def _places_html(data: Optional[dict]) -> str:
    if not data or not data["places"]:
        return "<div class='empty'>No locations or props yet.</div>"
    rows = [[f"<b>{esc(p['name'])}</b>", esc(p["kind"]), esc(p.get("description", "")[:160]),
             ui_theme.badge("reference ready", "ok") if p.get("slug") else ui_theme.badge("description only", "")]
            for p in data["places"]]
    return ui_theme.table(["Name", "Kind", "Description", "State"], rows)


def _place_choices(data):
    return [("New location or prop", NEW)] + [(p["name"], p["id"]) for p in (data or {}).get("places", [])]


def _episodes_html(series_id: str) -> str:
    rows = []
    for e in series.episodes(series_id) if series.valid_id(series_id) else []:
        label, kind = STATUS_LABELS[e["status"]]
        memory = {"owner": "edited by you", "auto": "saved by the agent", "": "—"}.get(e["summary_source"], "—")
        rows.append([esc(f"S{e['season']}E{e['number']}"), f"<b>{esc(e['title'])}</b>", ui_theme.badge(label, kind),
                     esc(f"{e['seconds']} s · {e['format']}"), esc(memory)])
    return ui_theme.table(["#", "Episode", "Status", "Length", "Story memory"], rows,
                          "No episodes yet. Describe the next one below.")


def _episode_choices(series_id: str):
    return [(f"S{e['season']}E{e['number']} {e['title']}", e["id"])
            for e in (series.episodes(series_id) if series.valid_id(series_id) else [])]


def _storyboard_html(episode: Optional[dict]) -> str:
    if not episode:
        return ""
    scenes = series.episode_scenes(episode)
    if not scenes:
        return "<div class='empty'>No storyboard yet.</div>"
    rows, total = [], 0.0
    for number, scene in enumerate(scenes, start=1):
        total += scene["seconds"]
        lines = "<br>".join(f"<i>{esc(characters.get(l['speaker'])['name'] if l['speaker'] and not l['speaker'].startswith('@') and characters.get(l['speaker']) else (l['speaker'] or 'Speaker'))}:</i> {esc(l['text'])}"
                            for l in scene["lines"])
        state = (ui_theme.badge("outdated", "warn") if scene["outdated"] else
                 ui_theme.badge(f"{len(scene['takes'])} take(s)", "ok") if scene["has_clip"] else
                 ui_theme.badge("not made", ""))
        rows.append([esc(str(number)), esc(f"{scene['seconds']:g} s"), esc(scene["prompt"][:300]), lines, state])
    estimate = total * queue_manager.rate("timeline_produce")
    head = (f"<p class='muted'>{len(scenes)} shots, {total:.0f} s. Generating every shot takes about "
            f"{esc(queue_manager.fmt_time(estimate))} of GPU time (an estimate from the measured speed; shots already "
            "made are reused).</p>")
    return head + ui_theme.table(["#", "Length", "Picture", "Dialogue", "Shot"], rows)


def _chat_html(episode: Optional[dict]) -> str:
    if not episode or not episode["chat"]:
        return "<div class='empty'>No conversation yet.</div>"
    out = []
    for message in episode["chat"][-30:]:
        who = "You" if message["role"] == "owner" else "Agent"
        out.append(f"<div class='chat-msg {esc(message['role'])}'><b>{who}:</b> {esc(message['text'])}</div>")
    return "<div class='story-chat'>" + "".join(out) + "</div>"


def _run_status(episode: Optional[dict]) -> str:
    if not episode or not episode["run"] or not agent_store.valid_id(episode["run"]):
        return ""
    record = agent_store.RunStore(episode["run"]).read() or {}
    state = record.get("state", "")
    text = (record.get("status") or {}).get("text") or ""
    cost = float((record.get("usage") or {}).get("cost_usd") or 0.0)
    if state in agent_store.LIVE_STATES:
        return ui_theme.note(f"⏳ The agent is working: {text or state}. (agent cost so far ${cost:.2f})")
    failure = (record.get("failure") or {}).get("message")
    if failure:
        return ui_theme.note(f"⚠ The agent stopped: {failure}")
    return ui_theme.note(f"ℹ Agent {state or 'idle'} (agent cost so far ${cost:.2f}).")


def _sync_chat(series_id: str, episode: dict) -> dict:
    """When the agent's run has ended after the owner's last message, its final summary becomes its reply."""
    if not episode["run"] or not agent_store.valid_id(episode["run"]) or not episode["chat"]:
        return episode
    if episode["chat"][-1]["role"] != "owner":
        return episode
    record = agent_store.RunStore(episode["run"]).read() or {}
    if record.get("state") in agent_store.LIVE_STATES:
        return episode
    answer = (record.get("outcome") or {}).get("summary") or (record.get("failure") or {}).get("message")
    return series.add_chat(series_id, episode["id"], "agent", answer) if answer else episode


def _warnings_md(series_id: str, episode: Optional[dict]) -> str:
    if not episode:
        return ""
    found = series.warnings(series_id, episode["id"])
    if not found:
        return ui_theme.note("✅ No continuity or completeness warnings. (The look of motion and lip sync still needs "
                             "your eyes: the checks read the plan, the files and the agent's frame review.)")
    return "**Check before publishing:**\n" + "\n".join(f"- ⚠ {esc(w)}" for w in found[:40])


def build_series_tab(tab: gr.Tab):
    """The Series workspace. Returns (load fn, outputs)."""
    ui_theme.page_head("Series", "Your series bible, cast, places and episodes.")
    with gr.Row():
        pick = gr.Dropdown(allow_custom_value=True, choices=_series_choices(), label="Series", value=None, scale=4,
                           elem_id="series-pick")
        new_series = gr.Button("New series", size="sm", scale=1)
        delete_series = gr.Button("Delete series", size="sm", variant="stop", scale=1)
    msg = gr.Markdown()
    with gr.Tabs(elem_id="series-tabs"):
        # ----- Series Bible -----
        with gr.Tab("Series Bible", id="bible"):
            with gr.Row():
                b_name = gr.Textbox(label="Name", max_length=series.MAX_NAME)
                b_look = gr.Radio(series.look_choices(), label="Animation style")
            b_logline = gr.Textbox(label="Premise", lines=2, max_length=series.MAX_TEXT)
            b_world = gr.Textbox(label="World", lines=3, max_length=series.MAX_TEXT)
            b_rules = gr.Textbox(label="Rules of the world", lines=2, max_length=series.MAX_TEXT)
            with gr.Row():
                b_tone = gr.Textbox(label="Tone", max_length=series.MAX_SHORT, placeholder="warm, funny, gentle")
                b_storytelling = gr.Textbox(label="Storytelling", max_length=series.MAX_SHORT,
                                            placeholder="each episode ends with a small lesson")
            with gr.Row():
                b_format = gr.Radio([("YouTube 16:9", "16:9"), ("Shorts 9:16", "9:16")], label="Default format")
                b_narration = gr.Radio([("Dialogue", "dialogue"), ("Narrator", "narrator"), ("Both", "both")],
                                       label="Narration")
                b_audience = gr.Radio(list(series.AUDIENCES), label="Audience")
                b_narrator = gr.Dropdown(allow_custom_value=True, choices=_voice_choices(), label="Narrator voice")
            b_state = gr.Textbox(label="Story so far (the agent updates it after each episode; your edit wins)",
                                 lines=3, max_length=series.MAX_TEXT)
            b_save = gr.Button("Save bible", variant="primary")
            ui_theme.section("Facts the story must keep", "What earlier episodes established. Correct a wrong one or "
                                                          "retire it so it no longer reaches the agent.")
            facts_box = gr.HTML(padding=False)
            with gr.Row():
                fact_new = gr.Textbox(label="New fact", max_length=series.MAX_SHORT, scale=4,
                                      placeholder="Pip lost his green scarf in the river (episode 2).")
                fact_add = gr.Button("Add fact", scale=1)
            with gr.Row():
                fact_pick = gr.Dropdown(allow_custom_value=True, choices=[], label="Fact", scale=3)
                fact_text = gr.Textbox(label="Corrected text", max_length=series.MAX_SHORT, scale=3)
                fact_fix = gr.Button("Save correction", scale=1)
                fact_retire = gr.Button("Retire / restore", scale=1)
            ui_theme.section("Relationships")
            rel_box = gr.HTML(padding=False)
            with gr.Row():
                rel_a = gr.Dropdown(allow_custom_value=True, choices=[], label="Character", scale=2)
                rel_b = gr.Dropdown(allow_custom_value=True, choices=[], label="and", scale=2)
                rel_text = gr.Textbox(label="Relationship (empty removes it)", max_length=series.MAX_SHORT, scale=3)
                rel_save = gr.Button("Save", scale=1)
        # ----- Characters -----
        with gr.Tab("Characters", id="cast"):
            cast_box = gr.HTML(padding=False)
            c_pick = gr.Dropdown(allow_custom_value=True, choices=[("New character", NEW)], value=NEW, label="Edit")
            with gr.Row():
                c_name = gr.Textbox(label="Name", max_length=series.MAX_NAME)
                c_kind = gr.Radio([("Animal / creature / robot", "creature"), ("Person (adult)", "person")],
                                  value="creature", label="Kind")
                c_role = gr.Textbox(label="Role", max_length=series.MAX_SHORT, placeholder="main, best friend…")
            c_desc = gr.Textbox(label="Look", lines=2, max_length=series.MAX_TEXT)
            with gr.Row():
                c_colors = gr.Textbox(label="Signature colors", max_length=series.MAX_SHORT)
                c_silhouette = gr.Textbox(label="Silhouette", max_length=series.MAX_SHORT,
                                          placeholder="small, round, big tail")
            with gr.Row():
                c_outfit = gr.Textbox(label="Outfit", max_length=series.MAX_SHORT)
                c_marks = gr.Textbox(label="Distinctive marks", max_length=series.MAX_SHORT,
                                     placeholder="white tail tip, round glasses")
            c_personality = gr.Textbox(label="Personality", lines=2, max_length=series.MAX_TEXT)
            c_outfits = gr.Textbox(label="Outfit variants (one per line, optional)", lines=2, max_length=1500)
            with gr.Row():
                c_voice = gr.Dropdown(allow_custom_value=True, choices=_voice_choices(), label="Voice")
                c_link = gr.Dropdown(allow_custom_value=True, choices=[],
                                     label="Library character (reference pictures and voice)")
            with gr.Row():
                c_save = gr.Button("Save character", variant="primary")
                c_remove = gr.Button("Remove from series", variant="stop")
                c_studio = gr.Button("Make reference pictures in the character creator")
            gr.Markdown("A character becomes ready for scenes when it has reference pictures and a voice: the agent "
                        "makes them from this sheet when an episode needs the character, or you make them in the "
                        "character creator with the same name.", elem_classes=["muted"])
        # ----- Locations & Props -----
        with gr.Tab("Locations & Props", id="places"):
            places_box = gr.HTML(padding=False)
            p_pick = gr.Dropdown(allow_custom_value=True, choices=[("New location or prop", NEW)], value=NEW,
                                 label="Edit")
            with gr.Row():
                p_name = gr.Textbox(label="Name", max_length=series.MAX_NAME)
                p_kind = gr.Radio([("Location", "location"), ("Prop", "prop")], value="location", label="Kind")
            p_desc = gr.Textbox(label="Description", lines=2, max_length=series.MAX_TEXT)
            with gr.Row():
                p_save = gr.Button("Save", variant="primary")
                p_remove = gr.Button("Remove", variant="stop")
                p_studio = gr.Button("Make reference pictures")
        # ----- Episodes -----
        with gr.Tab("Episodes", id="episodes"):
            episodes_box = gr.HTML(padding=False)
            with gr.Accordion("Create the next episode", open=False) as new_ep_box:
                e_idea = gr.Textbox(label="What happens in this episode?", lines=3, max_length=2000)
                with gr.Row():
                    e_title = gr.Textbox(label="Title (optional)", max_length=series.MAX_NAME)
                    e_seconds = gr.Slider(series.MIN_EPISODE_SECONDS, series.MAX_EPISODE_SECONDS, value=60, step=15,
                                          label="Length (seconds)")
                with gr.Row():
                    e_format = gr.Radio([("YouTube 16:9", "16:9"), ("Shorts 9:16", "9:16")], value="16:9",
                                        label="Format")
                    e_narration = gr.Radio([("Dialogue", "dialogue"), ("Narrator", "narrator"), ("Both", "both")],
                                           value="dialogue", label="Narration")
                e_create = gr.Button("Write the storyboard", variant="primary")
            e_pick = gr.Dropdown(allow_custom_value=True, choices=[], label="Episode", elem_id="episode-pick")
            run_box = gr.Markdown()
            ui_theme.section("Storyboard")
            board_box = gr.HTML(padding=False)
            with gr.Row():
                e_approve = gr.Button("Approve storyboard and generate", variant="primary")
                e_editor = gr.Button("Open in Editor")
                e_stop = gr.Button("Stop the agent", variant="stop")
            warn_box = gr.Markdown()
            ui_theme.section("Talk to the agent about this episode",
                             "For example: “Change the dialogue in scene 4 and make the fox look happier.” Only the "
                             "scenes that need it are changed.")
            chat_box = gr.HTML(padding=False)
            with gr.Row():
                chat_text = gr.Textbox(label="Your request", lines=2, max_length=2000, scale=5)
                chat_send = gr.Button("Send", variant="primary", scale=1)
            ui_theme.section("Story memory of this episode", "What later episodes will remember. Correct it if it is "
                                                             "wrong; your text is never overwritten.")
            summary = gr.Textbox(label="Summary", lines=4, max_length=series.MAX_SUMMARY)
            summary_save = gr.Button("Save summary")
        # ----- Editor -----
        with gr.Tab("Editor", id="editor"):
            gr.Markdown("The Editor is the Timeline: preview, scene fixes, takes, subtitles, title cards and the MP4 "
                        "export. Open a chapter of the episode chosen on the Episodes tab:")
            editor_pick = gr.Dropdown(allow_custom_value=True, choices=[], label="Chapter")
            editor_open = gr.Button("Open in Editor", variant="primary")

    HANDLES["series_pick"] = pick

    # ----- loading a series -----
    def _show(series_id):
        data = series.sync_links(series.get(series_id)) if series.valid_id(series_id) and series.get(series_id) else None
        sid = data["id"] if data else ""
        rel_names = {c["id"]: c["name"] for c in (data or {}).get("characters", [])}
        rels = ui_theme.table(["", "", ""], [[esc(rel_names.get(r["a"], "?")), esc(rel_names.get(r["b"], "?")),
                                              esc(r["text"])] for r in (data or {}).get("relationships", [])],
                              "No relationships yet.")
        pairs = [(c["name"], c["id"]) for c in (data or {}).get("characters", [])]
        library = [("—", "")] + [(c["name"], c["slug"]) for c in characters.all_characters()]
        episodes = _episode_choices(sid)
        latest = episodes[-1][1] if episodes else None
        return (*_bible_values(data), _facts_html(data), gr.update(choices=_fact_choices(data), value=None), rels,
                gr.update(choices=pairs, value=None), gr.update(choices=pairs, value=None), _cast_html(data),
                gr.update(choices=_char_choices(data), value=NEW), gr.update(choices=library, value=""),
                _places_html(data), gr.update(choices=_place_choices(data), value=NEW), _episodes_html(sid),
                gr.update(choices=episodes, value=latest),
                gr.update(value=(data or {}).get("format", "16:9")), gr.update(value=(data or {}).get("narration",
                                                                                                       "dialogue")))

    bible_outputs = [b_name, b_logline, b_world, b_rules, b_tone, b_storytelling, b_state, b_look, b_format,
                     b_narration, b_audience, b_narrator]
    show_outputs = bible_outputs + [facts_box, fact_pick, rel_box, rel_a, rel_b, cast_box, c_pick, c_link,
                                    places_box, p_pick, episodes_box, e_pick, e_format, e_narration]
    pick.change(_show, inputs=pick, outputs=show_outputs, show_progress="hidden")
    new_series.click(None, js=go_js("stories"))

    def _delete(series_id):
        try:
            series.delete(series_id)
        except series.SeriesError as error:
            return ui_theme.note(f"❌ {error}"), gr.skip()
        return ui_theme.note("✅ The series was set aside (its films, characters and projects stay)."), gr.update(
            choices=_series_choices(), value=None)
    delete_series.click(_delete, inputs=pick, outputs=[msg, pick])

    # ----- bible -----
    def _save_bible(sid, name, logline, world, rules, tone, storytelling, state, look, fmt, narration, audience,
                    narrator):
        try:
            current = series.get(sid)
            if current is None:
                raise series.SeriesError("Choose a series first.")
            fields = dict(name=name, logline=logline, world=world, rules=rules, tone=tone,
                          narration_style=storytelling, look=look, format=fmt, narration=narration,
                          audience=audience, narrator_voice=narrator or "")
            if (state or "") != current["story_state"]:
                fields["story_state"] = state or ""
            series.update(sid, **fields)
        except series.SeriesError as error:
            return ui_theme.note(f"❌ {error}"), gr.skip()
        return ui_theme.note("✅ Saved."), gr.update(choices=_series_choices(), value=sid)
    b_save.click(_save_bible, inputs=[pick] + bible_outputs, outputs=[msg, pick])

    def _facts_after(sid, action):
        try:
            action()
        except series.SeriesError as error:
            return ui_theme.note(f"❌ {error}"), gr.skip(), gr.skip()
        data = series.get(sid)
        return ui_theme.note("✅ Saved."), _facts_html(data), gr.update(choices=_fact_choices(data), value=None)

    fact_add.click(lambda sid, text: _facts_after(sid, lambda: series.add_fact(sid, text)),
                   inputs=[pick, fact_new], outputs=[msg, facts_box, fact_pick])
    fact_fix.click(lambda sid, fid, text: _facts_after(sid, lambda: series.edit_fact(sid, fid or "", text=text)),
                   inputs=[pick, fact_pick, fact_text], outputs=[msg, facts_box, fact_pick])

    def _toggle(sid, fid):
        data = series.get(sid) or {"facts": []}
        fact = next((f for f in data["facts"] if f["id"] == fid), None)
        return _facts_after(sid, lambda: series.edit_fact(sid, fid or "", active=not (fact or {}).get("active", True)))
    fact_retire.click(_toggle, inputs=[pick, fact_pick], outputs=[msg, facts_box, fact_pick])

    def _rel(sid, a, b, text):
        try:
            series.set_relationship(sid, a or "", b or "", text or "")
        except series.SeriesError as error:
            return (ui_theme.note(f"❌ {error}"), *[gr.skip()] * (len(show_outputs)))
        return (ui_theme.note("✅ Saved."), *_show(sid))
    rel_save.click(_rel, inputs=[pick, rel_a, rel_b, rel_text], outputs=[msg] + show_outputs)

    # ----- characters -----
    char_fields = [c_name, c_kind, c_role, c_desc, c_colors, c_silhouette, c_outfit, c_marks, c_personality,
                   c_outfits, c_voice, c_link]

    def _char_values(sid, cid):
        data = series.get(sid) or {"characters": []}
        c = next((c for c in data["characters"] if c["id"] == cid), None) or {}
        return (c.get("name", ""), c.get("kind", "creature"), c.get("role", ""), c.get("description", ""),
                c.get("colors", ""), c.get("silhouette", ""), c.get("outfit", ""), c.get("marks", ""),
                c.get("personality", ""), "\n".join(c.get("outfits") or []), c.get("voice", ""), c.get("slug", ""))
    c_pick.change(_char_values, inputs=[pick, c_pick], outputs=char_fields, show_progress="hidden")

    def _save_char(sid, cid, name, kind, role, desc, colors, silhouette, outfit, marks, personality, outfits, voice,
                   link):
        try:
            sheet = series.save_character(sid, {"name": name, "kind": kind, "role": role, "description": desc,
                                                "colors": colors, "silhouette": silhouette, "outfit": outfit,
                                                "marks": marks, "personality": personality, "voice": voice or "",
                                                "slug": link or ""}, "" if cid == NEW else cid)
            series.set_outfits(sid, sheet["id"], (outfits or "").splitlines())
        except series.SeriesError as error:
            return ui_theme.note(f"❌ {error}"), gr.skip(), gr.skip()
        data = series.get(sid)
        return ui_theme.note("✅ Saved."), _cast_html(data), gr.update(choices=_char_choices(data), value=sheet["id"])
    c_save.click(_save_char, inputs=[pick, c_pick] + char_fields, outputs=[msg, cast_box, c_pick])

    def _remove_char(sid, cid):
        if cid in (NEW, None, ""):
            return ui_theme.note("Choose a character."), gr.skip(), gr.skip()
        series.remove_character(sid, cid)
        data = series.get(sid)
        return ui_theme.note("✅ Removed from the series (the library character stays)."), _cast_html(data), \
            gr.update(choices=_char_choices(data), value=NEW)
    c_remove.click(_remove_char, inputs=[pick, c_pick], outputs=[msg, cast_box, c_pick])
    c_studio.click(None, js=go_js("characters"))

    # ----- places -----
    def _place_values(sid, pid):
        data = series.get(sid) or {"places": []}
        p = next((p for p in data["places"] if p["id"] == pid), None) or {}
        return p.get("name", ""), p.get("kind", "location"), p.get("description", "")
    p_pick.change(_place_values, inputs=[pick, p_pick], outputs=[p_name, p_kind, p_desc], show_progress="hidden")

    def _save_place(sid, pid, name, kind, desc):
        try:
            entry = series.save_place(sid, name, desc, kind, "" if pid == NEW else pid)
        except series.SeriesError as error:
            return ui_theme.note(f"❌ {error}"), gr.skip(), gr.skip()
        data = series.get(sid)
        return ui_theme.note("✅ Saved."), _places_html(data), gr.update(choices=_place_choices(data), value=entry["id"])
    p_save.click(_save_place, inputs=[pick, p_pick, p_name, p_kind, p_desc], outputs=[msg, places_box, p_pick])

    def _remove_place(sid, pid):
        if pid in (NEW, None, ""):
            return ui_theme.note("Choose a location or prop."), gr.skip(), gr.skip()
        series.remove_place(sid, pid)
        data = series.get(sid)
        return ui_theme.note("✅ Removed."), _places_html(data), gr.update(choices=_place_choices(data), value=NEW)
    p_remove.click(_remove_place, inputs=[pick, p_pick], outputs=[msg, places_box, p_pick])
    p_studio.click(None, js=go_js("items"))

    # ----- episodes -----
    def _episode_view(sid, eid):
        episode = series.refresh_episode(sid, eid) if series.valid_id(sid) and series.valid_id(eid or "") else None
        if episode:
            episode = _sync_chat(sid, episode)
        chapters = [(f"Chapter {i}" if len(episode["projects"]) > 1 else "Episode", p)
                    for i, p in enumerate(episode["projects"], 1)] if episode else []
        return (_run_status(episode), _storyboard_html(episode), _warnings_md(sid, episode), _chat_html(episode),
                (episode or {}).get("summary", ""),
                gr.update(choices=chapters, value=chapters[0][1] if chapters else None))
    episode_outputs = [run_box, board_box, warn_box, chat_box, summary, editor_pick]
    e_pick.change(_episode_view, inputs=[pick, e_pick], outputs=episode_outputs, show_progress="hidden")

    def _tick(sid, eid):
        episode = series.get_episode(sid, eid or "") if series.valid_id(sid) and series.valid_id(eid or "") else None
        live = bool(episode and episode["run"] and agent_store.valid_id(episode["run"]) and
                    (agent_store.RunStore(episode["run"]).read() or {}).get("state") in agent_store.LIVE_STATES)
        if not live and not (episode and episode["chat"] and episode["chat"][-1]["role"] == "owner"):
            return (*[gr.skip()] * len(episode_outputs), gr.skip())
        out = _episode_view(sid, eid)
        return (*out[:4], gr.skip(), out[5], _episodes_html(sid))
    gr.Timer(6.0).tick(_tick, inputs=[pick, e_pick], outputs=episode_outputs + [episodes_box],
                       show_progress="hidden")

    def _new_episode(sid, idea, title, seconds, fmt, narration):
        try:
            episode = series.create_episode(sid, idea, title=title, seconds=int(seconds), fmt=fmt, narration=narration)
        except series.SeriesError as error:
            return ui_theme.note(f"❌ {error}"), gr.skip(), gr.skip(), gr.skip()
        ready, why = _agent_ready()
        note = f"✅ {episode['title']} created."
        if ready:
            try:
                agent_runner.start_episode(sid, episode["id"])
                note += " The agent is writing its storyboard (you can watch it here)."
            except agent_runner.RunnerError as error:
                note += f" ⚠ {error}"
        else:
            _manual_storyboard(sid, episode["id"])
            note += f" {why}"
        return (ui_theme.note(note), _episodes_html(sid), gr.update(choices=_episode_choices(sid), value=episode["id"]),
                gr.update(open=False))
    e_create.click(_new_episode, inputs=[pick, e_idea, e_title, e_seconds, e_format, e_narration],
                   outputs=[msg, episodes_box, e_pick, new_ep_box])

    def _approve(sid, eid):
        episode = series.get_episode(sid, eid or "") if series.valid_id(sid) and series.valid_id(eid or "") else None
        if not episode:
            return ui_theme.note("Choose an episode.")
        try:
            if episode["run"] and agent_store.valid_id(episode["run"]):
                record = agent_store.RunStore(episode["run"]).read() or {}
                if record.get("state") == "completed" and (record.get("settings") or {}).get("mode") == "prepare":
                    agent_runner.resume(episode["run"], approve_production=True)
                else:
                    agent_runner.follow_up(episode["run"], "The owner approves the storyboard: generate every shot "
                                                            "that is missing, check it and export the episode.")
                series.update_episode(sid, eid, status="producing")
                return ui_theme.note("✅ Approved. The agent is generating the episode; shots that already exist are "
                                     "reused.")
            for project_id in episode["projects"]:
                timeline_job_for(project_id)
            series.update_episode(sid, eid, status="producing")
            return ui_theme.note("✅ Approved. Every missing shot is queued and the episode is exported when they are "
                                 "done.")
        except (agent_runner.RunnerError, ValueError) as error:
            return ui_theme.note(f"❌ {error}")
    e_approve.click(_approve, inputs=[pick, e_pick], outputs=run_box)

    def _stop(sid, eid):
        episode = series.get_episode(sid, eid or "") if series.valid_id(sid) and series.valid_id(eid or "") else None
        if episode and episode["run"] and agent_runner.stop(episode["run"]):
            return ui_theme.note("Stopping the agent…")
        return ui_theme.note("The agent is not working on this episode.")
    e_stop.click(_stop, inputs=[pick, e_pick], outputs=run_box)

    def _send(sid, eid, text):
        episode = series.get_episode(sid, eid or "") if series.valid_id(sid) and series.valid_id(eid or "") else None
        if not episode:
            return ui_theme.note("Choose an episode."), gr.skip(), gr.skip()
        if not episode["run"]:
            return ui_theme.note("This episode has no agent conversation; change it in the Editor."), gr.skip(), \
                gr.skip()
        try:
            agent_runner.follow_up(episode["run"], text)
        except agent_runner.RunnerError as error:
            return ui_theme.note(f"❌ {error}"), gr.skip(), gr.skip()
        return ui_theme.note("⏳ The agent is on it."), _chat_html(series.get_episode(sid, eid)), ""
    chat_send.click(_send, inputs=[pick, e_pick, chat_text], outputs=[run_box, chat_box, chat_text])

    def _save_summary(sid, eid, text):
        try:
            series.set_summary(sid, eid or "", text or "", source="owner")
        except series.SeriesError as error:
            return ui_theme.note(f"❌ {error}"), gr.skip()
        return ui_theme.note("✅ The story memory of this episode is saved."), _episodes_html(sid)
    summary_save.click(_save_summary, inputs=[pick, e_pick, summary], outputs=[msg, episodes_box])

    # ----- editor -----
    def _open(project_id):
        library = HANDLES.get("timeline_library")
        if not project_id or library is None:
            return gr.skip()
        from .ui_timeline import _project_choices
        return gr.update(choices=_project_choices(), value=project_id)
    for button, source in ((editor_open, editor_pick), (e_editor, editor_pick)):
        if HANDLES.get("timeline_library") is not None:
            button.click(_open, inputs=source, outputs=HANDLES["timeline_library"]).then(
                None, js="() => { document.querySelector('#timeline-open')?.click(); }").then(None,
                                                                                            js=go_js("timeline"))

    def _load():
        return gr.update(choices=_series_choices()), gr.update(choices=_voice_choices()), gr.update(
            choices=_voice_choices())
    return _load, [pick, c_voice, b_narrator]


def timeline_job_for(project_id: str) -> None:
    """Without the agent: queue the export of a project, which makes every missing clip first (timeline_produce)."""
    timeline.enqueue(timeline.load(project_id), timeline.PRODUCE)
