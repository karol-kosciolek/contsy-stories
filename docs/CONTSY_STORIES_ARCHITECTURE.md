# Contsy Stories — architecture (as implemented)

Contsy Stories is a single-user app that runs on one PC. Only what exists in the code is described here; planned work
is marked **not implemented**.

## Layers

```
Browser ── Gradio app (app/main.py, one process, 127.0.0.1 only) ────────────────────────────────────────────
             │  pages: Dashboard + wizard, Series workspace (app/ui_series.py), Editor = Timeline (app/ui_timeline.py),
             │         Characters, Locations & Props (items), Agent, Voices, GPU Queue, Settings
             │
             ├── Series data (app/series.py)  ── Series → Seasons → Episodes ── projects → Timeline (Scenes → Takes)
             ├── Story agent (app/agent_*.py) ── LLM provider interface (app/agent_provider.py; Anthropic built in)
             └── GPU queue (app/queue_manager.py) ── each job's runner in this process, on this PC's GPU (engine/)
```

## Data model (app/series.py)

Stored in the state folder (`paths.state_dir`, the Contsy folder by default), next to the characters, items, voices and
Timeline projects:

- `series/<id>/series.json` — the **Series Bible**: name, premise, world, rules, tone, storytelling, animation style
  (`look`, one of the product catalog), default format (16:9 / 9:16), narration (dialogue / narrator / both), audience,
  narrator voice; the **cast** as design sheets (stable id `c…`, name, kind, role, look, signature colors, silhouette,
  outfit, distinctive marks, personality, voice, outfit variants, link to the library character `slug`); **locations
  and props** (`p…`, link to the library item); **relationships**; **facts** (`f…`, source auto/owner, active/retired);
  **story state** (source auto/owner); seasons.
- `series/<id>/episodes/<id>.json` — season, number, title, idea, length, format, narration, status
  (draft → writing → storyboard → producing → done / failed), the Timeline **projects** (one per chapter), the agent
  **run** id, the **chat** shown on the page, the **summary** (source auto/owner).
- **Scenes and takes are not copied**: `episode_scenes()` reads them from the episode's Timeline projects, the same
  scenes and takes the editor, the agent and the export use. Takes are the existing Timeline versions (`timeline.takes`).
- Long episodes are split into **chapters** of at most 180 s (`series.chapters`), each its own Timeline project
  (a project holds at most 60 scenes). The agent is told to build and check one chapter at a time, so a failure costs
  one chapter. Resuming uses the existing mechanisms: a stopped/failed agent run resumes from its transcript, and the
  Timeline makes only missing or outdated clips ("Generate missing clips" / `prepare_missing`).
- Deleting a series or episode moves it to `series/.deleted/` (portable; no Windows Recycle Bin dependency).

## Story memory

- `story_context(series, episode)` builds the text the agent reads: the bible, the cast sheets with their library
  slugs (or NOT CREATED YET), places, relationships, **active** facts, the story so far and the summaries of the
  **earlier** episodes only (the latest three in full, older ones shortened, the oldest dropped to stay under 14 000
  characters).
- `episode_brief()` = the creator's request + format/length/language/narration + chapter plan + context; it is the
  brief of the episode's agent run (`agent_runner.start_episode`).
- After an episode the agent calls `save_episode_memory` (summary, new facts, story state). If it did not, the run's
  final summary is kept as the episode summary ("auto") when the episode is produced (`refresh_episode`).
- The creator can correct every summary, fact and the story state; such a text (source `owner`) is never overwritten
  by the agent, and a fact can be retired so it no longer reaches the agent.

## Character and voice consistency (what is enforced, what is not)

Enforced in code:
- An episode run is **bound** to its series (`run.json` → `series`). In a bound run: `save_project` always uses the
  series style; `create_character` refuses to design again a character the series already has in the library;
  `queue_clip` / `queue_still` (single realistic clips/pictures) are refused.
- The cast sheet (colors, silhouette, outfit, marks) is in every brief and the agent is told to repeat it in each scene
  text. Library characters keep their reference photos and their own voice (`character:<slug>`), which the Timeline
  uses for every scene they are in (existing reference mechanism: Qwen-Image-Edit composition, start frames).
- `series.warnings()` checks each episode: characters not in the series cast (extra characters), a speaker without
  a saved voice or with a voice other than the sheet's, lines by a library voice in a dialogue series, a chapter in
  another style or format, shots not made yet or outdated, no current export, storyboard length far from the plan,
  sheets without a library character, the agent's own review findings (`failing`, `owner_should_check`), and a
  finished episode without memory.

Since 2026-11-22 (docs/QUALITY_ARCHITECTURE.md): an episode's shots are ShotSpecs compiled by the studio from the
bible (`app/shot_spec.py`, `app/bible.py`), every keyframe and clip is checked by an independent shot check
(`app/shot_qc.py`: per-frame colour, backdrop, motion and framing detectors plus the local vision model; failing clips
are made again with a new seed), the episode is mixed as one programme at -14 LUFS (`engine/media/episode_mix.py`),
each export gets a QC report, and the delivery gate refuses "complete" while a shot or the sound did not pass. The
video model can still go wrong; the check's measured sensitivity per defect is in docs/QUALITY_ARCHITECTURE.md
("Regression set and measurements").

## The agent (app/agent_*.py)

- System prompt specialised for animated series (`agent_prompt.py`): showrunner role, continuity rules, short shots,
  narrator lines need no lip sync, no photorealism, original designs only, how to handle a change request after an
  episode.
- New tools (`agent_tools_series.py`): `get_series_bible`, `save_episode_memory`.
- **Conversation after a run** (`agent_runner.follow_up`): a finished (or failed/stopped) run takes the creator's new
  message as the next message of the same transcript and runs again with an extra cost allowance (the run's cost
  limit becomes what it has spent plus at most $2 per request). The agent finds the scene, changes only it
  (save_project / edit_scene / new take / choose_take) and re-exports only if a clip changed. The episode keeps its chat; the run's transcript keeps the whole
  conversation.
- LLM provider: `agent_provider.Provider` protocol with `register_provider(name, label, factory)`. Anthropic is built in,
  with the creator's own Claude API key. A local or self-hosted model can be added as another provider; **not
  implemented** and not quality-tested — the current agent is not replaced.

## GPU queue

Every generation is a job in the GPU queue (`app/queue_manager.py`), run in turn by its runner in the app's own
process on this PC's GPU. Jobs survive a restart; a failed job keeps its reason, and the clips a Timeline job already
made stay. **Pause and release GPU** in the top bar frees the card at once.

## What the product hides and what the engine keeps

Removed from the product's pages (the modules stay, with their tests, because other parts use them):

| Removed page / input | Why it stays in the code |
|---|---|
| Video page (single realistic clip), Image page (single picture) | their modules register the `wideo` / `obraz` GPU jobs used by the Agent page tools and old queued jobs |
| Gallery, old Dashboard (`ui_history`, `ui_studio`) | history and results are still written; the Dashboard is replaced by the Stories Dashboard |
| Universes, Styles pages | universes and custom styles still load in old projects; the series look catalog uses custom styles internally |
| Character/universe picker in the top bar | hidden; the Characters page still takes it |
| Realistic style in style pickers | only offered where a stored record already uses it (old projects open unchanged) |

The engine (Z-Image, Qwen-Image-Edit, Wan 2.2 + AniSora, VoxCPM2, MuseTalk, InfiniteTalk, EzAudio, RIFE, ffmpeg export,
subtitles, overlays, QC) is unchanged.
