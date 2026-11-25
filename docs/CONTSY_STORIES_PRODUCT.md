# Contsy Stories — AI Animated Series Studio

*Create animated characters, build entire episodes, and produce your own original cartoon series with AI.*

For YouTube cartoon creators, animated YouTube Shorts creators, independent producers of animated series and small
studios with their own series. Not a general AI video platform, not realistic ads, not AI influencers.

Contsy Stories runs on your own PC: one user, no account. This document describes what is implemented; see also
[CONTSY_STORIES_ARCHITECTURE.md](CONTSY_STORIES_ARCHITECTURE.md) and [QUALITY_ARCHITECTURE.md](QUALITY_ARCHITECTURE.md).

## What the creator sees

Top bar: **Dashboard · Series · Editor · Characters · Locations & Props**, and a Tools menu with Agent (the agent's
page and log), Voices, GPU Queue and Settings.

**Dashboard** — a GPU queue card (Current generations: the queued and running jobs with their progress), My Series
(open one), Recent Episodes and **Create New Series**, open by default when there is no series yet:

1. Name your series (and an optional premise)
2. Choose animation style
3. Create your main character (name, kind, look, colors, outfit, personality)
4. Select a voice
5. Describe your first episode (format 16:9 or Shorts 9:16, length 15 s – 15 min, dialogue / narrator / both)
6. Review the storyboard — the agent writes it; nothing is generated before approval
7. Generate — "Approve storyboard and generate" on the episode

No sampler, model or weight settings appear in this path. The agent runs on the creator's own Claude API key (saved on
the Agent page); without a key the Dashboard says so and the storyboard is built by hand in the Editor.

**Series workspace** (per series):

1. **Series Bible** — name, premise, world, rules, tone, storytelling, animation style, default format and narration,
   audience, narrator voice, story so far (editable), facts the story must keep (add, correct, retire),
   relationships between characters.
2. **Characters** — design sheets with a stable id: name, kind, role, look, signature colors, silhouette, outfit,
   distinctive marks, personality, outfit variants, voice, and the link to the library character (reference pictures
   and own voice). "Make reference pictures in the character creator" opens the Characters page.
3. **Locations & Props** — recurring places and props, linked to the library items by name.
4. **Episodes** — list with status and story-memory source; create the next episode (the agent writes its
   storyboard with the whole series context); the storyboard (shots, length, dialogue, takes, outdated shots);
   continuity and completeness warnings; **chat with the agent about the episode** after it is done; the
   episode's story memory (editable).
5. **Editor** — opens the episode's Timeline project(s): preview, scene fixes, takes, subtitles, title cards, export.

## Animation styles

| Style | Status | Engine path |
|---|---|---|
| 2D Cartoon | ready | preset `cartoon` (style LoRAs, AniSora drawn experts) |
| Stylized 3D Cartoon | ready | preset `3d` |
| Anime | ready | preset `2d` |
| Storybook Illustration | experimental | custom style over 2D cartoon |
| Clay Animation | experimental | custom style over 3D |
| Paper Cutout | experimental | custom style over 2D cartoon |
| Comic Animation | experimental | custom style over 2D cartoon |

"Ready" means the preset's own models and tuning exist in the engine (`app/styles.py`); the experimental ones use
the same models with style words and were not tuned or quality-checked. The episodes of *Lumi and the Lantern Woods*
were made in Stylized 3D Cartoon; the other styles have not been checked on whole episodes. No style names or imitates
a franchise. Realistic is no longer offered in any picker (old projects that
use it still open).

## Content safety

The studio's protections apply unchanged: minors combined with sexual content are refused everywhere
(`llm.blocked`), sexual content with animals/creatures is refused, voice cloning needs consent.

The character rules also refuse *harmless* child characters, by design:
- a **person** character must be an adult (`characters.check_look`: any word naming a minor — child, kid, boy/girl
  with an age, teen…);
- an **animal or creature** must be fully grown: words like *puppy, kitten, cub, baby, young, chibi, childlike* in its
  look are refused (`llm.refers_to_young`, `llm._CHILDLIKE`);
- the agent's rules say "characters are adults (or fully grown animals and creatures)".

Effect for Contsy Stories: "a baby dragon", "a little kitten" or "a 7-year-old girl" cannot be a character; "a small
fox", "a tiny robot", "a little bunny" can. The cause is the photo pipeline's safety design (reference photos of a
character are generated and edited repeatedly), not a bug, and it is deliberately not relaxed. Child characters in
stylized cartoons would need a narrow change: allow young-looking characters only in non-realistic styles, keep the
sexual-content refusal and add an image moderation model (a new model, not part of Contsy yet).

## Design decisions (all reversible)

1. **Same stack** (Gradio, one process); no frontend rewrite. New pages in `app/ui_series.py`.
2. **Scenes and takes stay in the Timeline**; an episode links its projects instead of copying them, so the editor,
   the agent and the export keep one source of truth.
3. **Long episodes = chapters of ≤ 180 s**, each one Timeline project (a project allows 60 scenes).
4. **Character sheets live in the series**, the reference pictures and voice in the existing library character with
   the same name (linked automatically). The agent creates missing library characters from the sheet.
5. **Speech in English only** (the language list of a series has English only): the voice engine is driven in English
   by the existing pipeline. Other dialogue languages were not tested, so they are not offered.
6. **Old pages hidden, not deleted**: Video, Image, Gallery, old Dashboard, Universes, Styles. Their modules stay
   because the agent page's tools and stored data use them (see the architecture doc).
7. **Realistic removed from pickers** but still readable, so no existing project changes.
8. **Failed jobs** (an engine or model error, out of GPU memory) end as failed with the reason in the GPU queue;
   **Retry scene** runs a failed scene again and keeps the clips already made.
9. **Tests ignore `config/settings.local.json`** (`CONTSY_SETTINGS_LOCAL=none` set by `tests/test___local_settings.py`),
   so a developer's local overrides (another port or state folder) cannot change what the suite tests.
10. **Child characters**: the existing adult-only rule is kept (see Content safety).
11. **Narrator = voice-over in animated styles**: a line spoken by a library voice without a character moves no
    mouth (no InfiniteTalk, no MuseTalk, no 25 FPS lip-sync pass) in any non-realistic style
    (`video_gen.generate`); in a Realistic film it is lip-synced as before. Saves GPU time on narrated scenes.
12. **Episode title cards** are exact text overlays set by the agent (never drawn by the video model).
13. **The Editor's own "New" project still starts in the Realistic style** (changing the editor's blank-project
    default broke its unsaved-changes logic in 4 tests, so it was reverted). Episodes never use it: their projects are
    created in the series style by the agent or the wizard. To fix later in `ui_timeline._NEW_PROJECT` together with
    the editor's draft comparison.

## Progress

| Stage | State |
|---|---|
| 1. Dependency audit | done |
| 2. Simpler product structure and UI | done (new Dashboard, Series workspace, wizard; old pages hidden) |
| 3. Series Bible | done (data, UI, story memory, the creator's corrections) |
| 4. Characters, voices, episodes | done in code (sheets, links, guards, warnings); visual identity not guaranteed |
| 5. Agent and Timeline | done (episode runs, follow-up chat, memory tool, style/cast guards) |
| 6. Onboarding | done (7-step wizard) |
| 7. Quality layer | done (ShotSpec, independent shot QC with retries, QC report, -14 LUFS episode mix, music library; see [QUALITY_ARCHITECTURE.md](QUALITY_ARCHITECTURE.md)) |
| 8. Tests | unit tests for series, story memory, the series-bound agent and the quality layer (`tests/test_stories.py`, `tests/test_quality_architecture.py`); no GPU needed |
| 9. Docs | done |

Next steps (not done): whole-episode runs in the styles other than Stylized 3D Cartoon, an image moderation model
(needed before child characters could be allowed), profile and back views in the character model sheets, an ASR check
of the narration text.
