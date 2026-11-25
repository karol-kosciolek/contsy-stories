# Quality architecture of series episodes

Written on 2026-11-22, after the frame-by-frame review of episodes 01–08 of "Lumi and the Lantern Woods". Quality no
longer depends on the agent writing good prose and grading itself: code, data and an independent check guard it. Only
what exists in the code is described; open points are listed at the end.

## Defects A–Q and where each is handled

| # | Defect | Cause fixed in | Caught by |
|---|---|---|---|
| A | A stranger (a boy) at the end of a shot | negations never reach the prompt (`shot_spec.negations`, `bible.positive`); `must_not_show` goes to the check only | vision question `only_cast` over frames including the last 1.5 s every 4th frame, plus zoomed crops of what moves in the tail (`shot_qc.hotspots`) |
| B | Plain studio backdrop | every shot's keyframe is composed with the location plate (compiled prompt names the place → library item photo is a reference) | `shot_qc.check_backdrop` (edge density in all three bands of the frame) on the keyframe and on the clip |
| C | Character anatomy changes (shell, belly patch on the back) | shots are one picture pass (≤ 6 s), cut by default, chains ≤ 2 and only from a last frame that passes the check; canonical words say "belly patch on the front only" | vision `<slug>_identity` on picture 2: the reference beside crops of the character from the whole shot |
| D | Barley grey / missing photo | the compiler derives the cast from every name and alias (Barley = "owl"): a named character always gets its photo and canonical words | `<slug>_colour` / accessory colour masks (CPU) and vision `<slug>_body_colour`, `<slug>_<accessory>` |
| E | Colour drift (blue shot, green hat) | keyframe from references; no long I2V chains | `check_drift` against the keyframe (or the first frame) and the accessory masks |
| F | Physics / meaning (stands on water, puddle → lake) | `key_visual` from the narration is in the prompt | vision `shows_n` (every key visual and must_show) and `physics` |
| G | Prop turns into a monster | no second part re-playing an action from a mid-motion frame (single pass) | vision `stays_n` per key visual and `stable` |
| H | Snowman → owl; house wooden vs mossy | locations and props as library items with plates, used in every keyframe | vision `same_place` (against the plate), `shows_n` |
| I | Front-facing, motionless | `_talking` adds "natural lip movement" only for a speaking character (never the narrator in an animated film); `ShotSpec.action` needs a motion verb, a static action needs a moving camera | `check_motion` (change of the characters' silhouettes, camera-move tolerant) and vision `action` |
| J | Camera runs off the characters | camera from a fixed list ("…keeping the characters centered") | `check_framing` (a character's size collapses or it is cut by the edge in the second half) |
| K | Title burned over a face | export moves a centred title of a series episode to the top or bottom band that the characters cover least (`timeline._safe_titles`) | `title_safe_zone` in the QC report |
| L | Narration rhythm: 0.3 s in, 2–3 s of silence before each cut | shot length is fitted to the measured narration (sound preview → `check_project.suggested_seconds`); narrator lines are read at ≤ 135 words a minute (`Cue.pace`, `atempo`, at most 15 % slower); J-cuts up to 0.35 s in the episode mix | QC report `narration_fits`, `narration_pace` |
| M | No music, ambience changing at every cut, dead air, ambience too loud | `engine/media/episode_mix.py`: one bed per place, cross-faded only when the place changes, room tone under it, bed −14 LU under the narrator and ducked; music from the music library (`music/` in the state folder) | `dead_air`, `ambience_under_voice`, `music` |
| N | −19 LUFS | master to −14 LUFS integrated, true peak ≤ −1.5 dBTP (4× oversampled limiter) | `loudness`, `true_peak` measured on the exported file (`engine/media/loudness.py`, BS.1770-4) |
| O | QC always ok=True | a clip nobody checked is `unchecked` (`qc.Verdict(status=...)`, `shot_qc.status_of`) | — |
| P | Agent graded all episodes OK | the check is independent and binding: gates G8 (every shot `pass`) and G9 (QC report, sound, title) in `agent_delivery.quality_failures`; `finish_run partial` must list the failures | — |
| Q | "golden glow" → "skin" | `neutral_look` applies only when a person (with a photo) is in the shot | unit test |

## Layers

1. **Shot contract** (`app/shot_spec.py`). A series episode's scene carries a `shot`: location, time of day, one action,
   camera (fixed list), cast, 1–3 key visuals, props, must_show, must_not_show, continuous. `save_project` refuses a
   series scene without one, validates it against the bible and the series rules, and stores the compiled prompt in
   `Scene.prompt` and the spec in `Scene.shot` (canonical JSON; a scene without a shot keeps its old identity and key).
   `get_project` shows each scene's `video_prompt`: what Wan really gets. `edit_scene` refuses a new free prompt for
   a shot scene.
2. **Bible as data** (`app/bible.py`). Canonical words, aliases, accessories with colours and the body colour of each
   character, derived from the series sheet (or given explicitly: `canonical`, `aliases`, `must_have`, `palette`);
   places and props with their plates (library items). Colour words map to HSV ranges in `app/palette.py`.
3. **Drift-resistant generation** (`app/video_gen.py`, `app/timeline.py`, `app/shot_pipeline.py`). The keyframe is
   checked before Wan (backdrop, palette, vision) and composed again with a new seed when it fails (3 tries). An
   anchored shot starts from the last frame only if that frame passes the check. Shots are one pass (a series shot
   longer than `video_gen.part_seconds_limit()` is refused): the measured choice "a" of the brief (see Measurements).
4. **Independent check** (`app/shot_qc.py`, `app/qc_report.py`, `app/shot_arbiter.py`). Frames every 0.5 s, every 4th
   frame of the last 1.5 s, first and last; CPU detectors; one local vision call (Qwen3.5-9B, the app's own model) with
   two pictures (the shot sheet with references and tail zooms; the character crops beside their references) and
   yes-means-right questions made from the spec. Optional Claude arbiter (`config "qc": {"arbiter": true}`, the
   Claude API key saved on the Agent page, Claude Sonnet 5.5 by default) for questions the local model is unsure
   about; it sends those questions with the shot sheet and the character crops to Anthropic, and its tokens and price
   are counted in `shot_arbiter.spent`. A failing clip is made again with a new seed (2 retries), then `needs_review`. The export
   writes `<film>.qc.json` and `<film>.qc.html` (frames of every shot, verdicts, sound measurements).
5. **Sound** (`engine/media/episode_mix.py`, `engine/media/loudness.py`). Used for every project of a series (and every
   project with shot scenes); other Timeline films keep their scene-by-scene mix.

Settings (`config/settings.json`, key `qc`, all optional): `vision` (true), `arbiter` (false), `arbiter_model`,
`retries` (2), `keyframe_tries` (3). The existing "qc" preference turns the check off; then clips stay `unchecked`
and an episode cannot be delivered as complete.

## Regression set and measurements

`tests/fixtures/quality/` holds copies (640×360) of 27 shots of episodes 01–08 labelled with their defects, two
40-second sound pieces, the reference pictures and the series bible (`tools/build_quality_fixtures.py` rebuilt them;
tests never read live state). `tests/test_quality_architecture.py` runs the CPU detectors, the compiler, the
mix and the gates on them. `tools/qc_eval.py [--vision]` prints the sensitivity per defect and the false alarms per
detector family.

Measured on 2026-11-22 (`tools/qc_eval.py --vision`, local Qwen3.5-9B, RTX 4070 Ti SUPER), final question format:

| | CPU detectors | + vision |
|---|---|---|
| defective shots failed | 9/23 (B, E, J ×2, the still ones) | 16/23 |
| good shots failed (false alarms) | 0/4 | 0/4 |
| missed | — | A (a boy ~40 px tall in a doorway), C shell (04), D grey Barley (04 s4), G bubble wand and bubbles, H wooden vs mossy house (×2: the fixtures have no location, so `same_place` is not asked) |

What the live test episode taught (all fixed and covered by tests):
- Yes/no questions with negations inside were answered backwards by the small model ("Is Lumi the only being, with no
  human?" → "no: no humans"), failing a good shot. Every question now has two described answers, A (right) and B (the
  defect). An earlier yes/no version caught 19/23 but with false alarms; the A/B version is the one in use.
- The model read the sheet's own yellow captions as "text in the shot"; the prompt now says to ignore them.
- It took a character walking towards the camera for one standing still; its "stands still" counts only when the
  measured silhouette change agrees (`MOTION_CONFIRM`).
- Naming example items in a generation text draws them: the composition lead said "a hat, a bow tie, a scarf,
  glasses" and Lumi got glasses and a scarf. Generation texts never list examples; only the check's questions do.
- A follow-up after a hard stop left an unanswered tool call (API 400); `agent_runner.follow_up` answers it first.

One vision call takes 15–35 s. Production of a ≤6 s shot in the test episode took about 3.5–4 min including the
keyframe, its check, the clip and the clip check.

Sound, on the stems of episodes 03, 04 and 06 mixed again: −14.0 LUFS, true peak −1.5 to −1.65 dBTP, no dead air,
ambience 17.8–18.6 LU under the narrator (the old films: −18 to −20 LUFS).

## Test episode (2026-11-22)

"The Big Puddle (QC test)" (series Lantern Woods, project `8474bdd758de44259f75f1680efc5c5b`, 18 shots, 83.1 s), made
by the episode agent (Claude Sonnet 5.5, $2.39 of API for the whole run including the restarts below) with the shot
check on:

- 14/18 shots passed (11 at the first try); 4 are `needs_review` after 3 tries, each rightly: shot 4 Lumi stands
  still facing the camera; shot 7 Barley's landing is not shown and Lumi wears Barley's red bow tie; shots 14-15 the
  leaf boats are tiny and on a stream, not on the puddle.
- The delivery gate refused "complete": the run ended `open_items` with G8 (the 4 shots) and G9 (sound and title).
- Sound measured on the film: −14.01 LUFS, no dead air, ambience 18.1 LU under the narrator, 16 J-cuts, no music
  (empty library). Failed: true peak −1.34 dBTP after the AAC encode (fixed: the mix now aims at −2.0, re-measured
  after AAC at −1.64), narration cut at the end of shots 8-9 (the agent's lengths), 149 words a minute (the slowest
  tempo is now 0.8), the title band covered by Lumi's hat (23 %).
- Frame review every second and the last frame of every shot: no stranger, no colour drift, no plain backdrop, no
  runaway camera, no anatomy change; the puddle reads as a stream in shots 5-6 and 13 (passed by the check).
- Production took about 3.5-5 min of GPU per shot try (keyframe, its check, the clip, the clip check); the run made
  18 shots with 12 retries in about 2.5 h of GPU (plus about 40 min lost to the restarts for fixes), well over the 1-1.2 h estimated, mostly because of the retries.

Open from the test: a shot whose action brings a character in ("Barley flutters down and lands") cannot work while
the keyframe already holds the whole cast; two creatures composed together can swap accessories.

## Open points

- J/L-cuts: a line may start up to 0.35 s before its cut; a line never runs on after its shot (the per-shot speech
  stems are cut at the scene end).
- Model sheets: the identity comes from the front photo (plus the 3/4 photo when the library has one) and the
  canonical words; no profile/back views are generated.
- ASR check of the narration text: needs a new package, not added yet.
- The local vision model misses small or subtle defects; see the measured sensitivity. The Claude arbiter is the
  stronger second opinion and is off until you turn it on.
