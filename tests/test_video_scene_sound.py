# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Timeline scenes with structured sound in video_gen: lines with speakers and entry times, the
background description and timed effects, a prepared audition reused without touching its files,
and the clip's own mix lined up with the picture. Wan, lip sync and the start frames are fakes."""
from contextlib import ExitStack
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from app import ffmpeg_utils, history, model_manager, prefs, scene_sound, sound, tts, video_gen, voices
from app.scene_parts import Cue, Effect
from engine.media import audio as media_audio

RATE = media_audio.RATE
HANDLE = media_audio.TIMELINE_AUDIO_HANDLE
STYLE = "high quality realistic sound effects recording"
ROSE = {"name": "Rose", "slug": "rose", "gender": "woman", "look": "Short grey hair", "photos": ["rose.png"]}
PICTURE = "Rose stands by the rainy window. Rose talks, natural lip movement"


def _tone(path, seconds, frequency, level=0.2):
    t = np.arange(int(round(seconds * RATE))) / RATE
    media_audio.write_wav(path, (level * np.sin(2 * np.pi * frequency * t))[None, :].astype(np.float32), RATE)


def _copy(source, destination, *args, **kwargs):
    shutil.copyfile(source, destination)
    return True, str(destination)


class StructuredSceneTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        pool = []
        for name, slug, gender in (("Hanna", "hanna", "female"), ("Adam", "adam", "male")):
            _tone(self.root / f"{slug}.wav", 0.5, 180)
            pool.append(voices.Voice(name, slug, gender, str(self.root / f"{slug}.wav"), builtin=True))
        self.events, self.lipsync, self.mixes, self.sounds, self.lines = [], [], [], [], []
        client = MagicMock()
        client.free.side_effect = lambda: self.events.append("free")
        client.trim.side_effect = lambda: self.events.append("trim")

        def speech(items, progress_cb=None):
            self.events.append("speech")
            for _text, _voice, out in items:
                _tone(out, 1.0, 320)
            return True, ""

        def ambience(_client, _scene, custom, seconds, seed, out_path, progress_cb=None):
            self.events.append("ambience")
            self.sounds.append((custom, seconds, seed))
            _tone(out_path, seconds + 2 * HANDLE, 220 if custom.endswith(STYLE) else 1500, level=0.05)
            return True, str(out_path)

        self.part_fps, self.retimed = [], []

        self.wan_prompts = []

        self.wan_calls = []

        def wan(_runtime, _kind, _start, output, *args, **options):
            self.events.append("wan")
            self.wan_calls.append((_kind, _start, options.get("style")))
            self.part_fps.append(args[1])  # the rate RIFE brings the part to
            self.wan_prompts.append(options["prompt"])
            Path(output).write_bytes(b"mp4")

        def lip_sync(video, voice, timings, who, output, *_args, **kwargs):
            self.events.append("lipsync")
            track, rate = sf.read(voice, dtype="float32")
            self.lipsync.append({"seconds": track.size / rate, "timings": list(timings), "who": list(who)})
            self.lines.append([Path(path).read_bytes() for path in kwargs.get("lines") or ()])
            Path(output).write_bytes(Path(video).read_bytes() + b" synced")
            return True, output, []

        def mix(voice, background, out_path, duration, **kwargs):
            self.mixes.append((voice, background, duration, kwargs))
            Path(out_path).write_bytes(b"mix")
            return True, str(out_path)

        def frame(_client, _character, _text, _w, _h, out, *_args, **_kwargs):
            Path(out).write_bytes(b"png")
            return True, str(out)

        self.speech = stack.enter_context(patch.object(tts, "synthesize_many", side_effect=speech))
        self.ambience = stack.enter_context(patch.object(sound, "generate", side_effect=ambience))
        self.ensure = stack.enter_context(patch.object(model_manager, "ensure", return_value=None))
        self.record = stack.enter_context(patch.object(history, "record"))
        for target, name, options in (
                (voices, "all_voices", {"return_value": pool}),
                (video_gen.characters, "render", {"side_effect": frame}),
                (video_gen.engine_runtime, "engine", {"return_value": client}),
                (video_gen.engine_runtime, "video_part", {"side_effect": wan}),
                (video_gen.lipsync_proc, "run_dialogue", {"side_effect": lip_sync}),
                (video_gen, "_lipsync_to_output_fps",
                 {"side_effect": lambda _c, source, output, **kw: self.retimed.append(kw["duration"]) or _copy(source, output)}),
                (ffmpeg_utils, "convert_fps", {"side_effect": _copy}),
                (ffmpeg_utils, "mix_voice_and_background", {"side_effect": mix}),
                (ffmpeg_utils, "combine_video_audio", {"side_effect": lambda source, _audio, output, **kw: _copy(source, output)}),
                (ffmpeg_utils, "probe_duration", {"return_value": 3.0}),
                (ffmpeg_utils, "video_fps", {"return_value": 30.0})):
            stack.enter_context(patch.object(target, name, **options))
        stack.enter_context(patch.object(prefs, "_FILE", self.root / "prefs.json"))

    def generate(self, folder="scene", prompt=PICTURE, **options):
        options.setdefault("cues", (Cue("Hello there.", "rose", 0.5),))
        options.setdefault("background_text", "steady rain")
        options.setdefault("effects", (Effect("door slam", 1.0, 1.0),))
        return video_gen.generate(prompt, 3, "Portrait 9:16", seed=123,
                                  scene_dir=self.root / folder, cast=[ROSE], exact_cast=True, **options)

    def test_the_scene_sound_is_made_before_the_picture_and_drives_lip_sync(self):
        output = self.generate()
        self.assertTrue(output.ok, output.error)
        # the talking picture is made at the lip-sync rate and brought to 30 FPS once, to the scene's length
        self.assertEqual((self.part_fps, self.retimed), ([25], [3.0]))
        self.assertLess(self.events.index("speech"), self.events.index("wan"))
        self.assertLess(max(i for i, event in enumerate(self.events) if event == "ambience"), self.events.index("wan"))
        # VoxCPM2 and the lip-sync model get the card with trim(), which keeps the warm models and prompt
        # encodings in RAM; free() would drop them in the middle of every talking clip
        self.assertNotIn("free", self.events)
        self.assertEqual(self.events[self.events.index("speech") - 1], "trim")
        self.assertEqual(self.events[self.events.index("lipsync") - 1], "trim")
        self.speech.assert_called_once()
        self.assertEqual([voice for _, voice, _ in self.speech.call_args.args[0]], ["hanna"])
        self.assertEqual(self.sounds, [(f"steady rain, {STYLE}", 3, 123), ("door slam", 1.0, 123 + 1009)])
        self.assertEqual(self.lipsync, [{"seconds": 3.0, "timings": [(0.5, 1.5)], "who": ["female#char:rose"]}])
        self.assertEqual(output.info["dialogue_timings"],
                         [{"text": "Hello there.", "character": "rose", "voice": "hanna", "start": 0.5, "end": 1.5}])
        self.assertEqual((output.info["dialogue"], output.info["voices"]), (["Hello there."], ["Hanna"]))
        self.assertRegex(output.info["sound_key"], r"^[0-9a-f]{64}$")
        self.assertEqual(output.info["background_prompt"], f"steady rain, {STYLE}")
        needed = self.ensure.call_args_list[0].args[0]
        self.assertTrue({"voxcpm2", "musetalk-15", sound.MODEL_ID} <= set(needed))
        # the clip's own mix skips the background's handle; the stems keep it for the export
        (_voice, _background, duration, options), = self.mixes
        self.assertEqual((duration, options), (3.0, {"background_offset": HANDLE}))
        background, voice = output.audio_stems
        self.assertEqual(sf.info(str(background)).frames, int(3.5 * RATE))
        self.assertEqual(sf.info(str(voice)).frames, 3 * RATE)
        # the effects alone on the background's clock, for the export to lay over the leveled room tone
        self.assertEqual(sf.info(str(output.sound_overlay)).frames, int(3.5 * RATE))
        # lip sync hears each speaker's own line files, not only the summed dialogue track
        self.assertEqual(len(self.lines[0]), 1)
        self.record.assert_not_called()
        leftovers = sorted(path.name for path in (self.root / "scene").iterdir())
        self.assertEqual(len(leftovers), 4, leftovers)  # the clip, its two stems and the effects, nothing else
        self.assertTrue(all(re.fullmatch(r"video_\d{8}_\d{6}(\.mp4|_bg\.wav|_voice\.wav|_overlay\.wav)", name)
                            for name in leftovers))

    def test_a_line_past_the_scene_end_is_cut_and_the_scene_keeps_its_length(self):
        """The owner sets the scene's length (2026-11-16): a line that runs past the end is cut there, the clip is
        made with the length chosen for it."""
        output = self.generate(cues=(Cue("Hello there.", "rose", 2.5),))
        self.assertTrue(output.ok, output.error)
        self.assertIn("wan", self.events)
        self.assertIn("ambience", self.events)
        self.assertEqual([(line["start"], line["end"]) for line in output.info["dialogue_timings"]], [(2.5, 3.5)])

    def test_a_scene_of_tenths_of_a_second_is_made_in_whole_seconds_and_cut_to_its_length(self):
        output = video_gen.generate(PICTURE, 2.5, "Portrait 9:16", seed=123, scene_dir=self.root / "short",
                                    cast=[ROSE], exact_cast=True, cues=(Cue("Hello there.", "rose", 0.5),),
                                    background_text="steady rain")
        self.assertTrue(output.ok, output.error)
        self.assertEqual(output.info["seconds"], 2.5)
        self.assertEqual(self.retimed, [2.5])  # the picture, made for 3 s, is cut to the scene's 2.5 s

    def test_a_speaker_outside_the_scene_fails_at_once(self):
        output = self.generate(cues=(Cue("Hi.", "@nobody"),))
        self.assertEqual((output.ok, output.error), (False, "Line 1: no voice named @nobody."))
        self.assertEqual(self.events, [])

    def test_a_prepared_audition_is_copied_and_never_changed(self):
        cues = (Cue("Hello there.", "rose", 0.5),)
        speakers = scene_sound.resolve_speakers(cues, [ROSE])
        audition = scene_sound.prepare(cues, speakers, f"steady rain, {STYLE}", (Effect("door slam", 1.0),), 3, 9,
                                       self.root / "audition", client=MagicMock())
        before = {path.name: path.read_bytes() for path in audition.folder.iterdir()}
        self.events.clear()
        output = self.generate(prepared_sound=audition)
        self.assertTrue(output.ok, output.error)
        self.assertNotIn("speech", self.events)
        self.assertNotIn("ambience", self.events)
        needed = set(self.ensure.call_args_list[0].args[0])
        self.assertIn("musetalk-15", needed)
        self.assertFalse({"voxcpm2", sound.MODEL_ID} & needed)
        self.assertEqual({path.name: path.read_bytes() for path in audition.folder.iterdir()}, before)
        self.assertEqual(scene_sound.load(audition.folder, audition.key, verify=True), audition)
        background, voice = output.audio_stems
        self.assertEqual(background.read_bytes(), audition.background.read_bytes())
        self.assertEqual(voice.read_bytes(), audition.voice.read_bytes())
        self.assertEqual(output.sound_overlay.read_bytes(), audition.overlay.read_bytes())
        self.assertEqual(output.info["sound_key"], audition.key)
        self.assertEqual(self.lipsync[0]["who"], ["female#char:rose"])
        self.assertEqual(self.lines[0], [path.read_bytes() for path in audition.lines])

    def test_an_audition_of_other_lines_is_not_used(self):
        cues = (Cue("Something else.", "rose", 0.5),)
        audition = scene_sound.prepare(cues, scene_sound.resolve_speakers(cues, [ROSE]), f"steady rain, {STYLE}",
                                       (), 3, 9, self.root / "audition", client=MagicMock())
        self.events.clear()
        output = self.generate(prepared_sound=audition)
        self.assertTrue(output.ok, output.error)
        self.assertIn("speech", self.events)
        self.assertIn("no longer matched", " ".join(output.notes))
        self.assertEqual(output.info["dialogue"], ["Hello there."])

    def test_without_sound_the_lines_are_ignored(self):
        output = self.generate(sound_on=False)
        self.assertTrue(output.ok, output.error)
        self.assertEqual(self.events, ["wan"])  # nothing frees or trims the engine around a silent picture
        self.assertEqual((self.part_fps, self.retimed), ([30], []))  # silent: RIFE 16 -> 30 FPS, nothing more
        self.assertIsNone(output.audio_stems)
        self.assertEqual((output.info["dialogue"], output.info["dialogue_timings"]), ([], []))

    def test_a_background_description_and_effects_need_the_scenes_cues(self):
        output = self.generate(prompt="Rain runs down the window.", cues=None)
        self.assertFalse(output.ok)
        self.assertIn("come with its cues", output.error)
        self.assertEqual(self.events, [])

    def test_every_spoken_and_sound_text_passes_the_minors_guard(self):
        with patch.object(video_gen.llm, "blocked", return_value=True) as blocked:
            output = self.generate()
        self.assertEqual((output.ok, output.error), (False, video_gen.llm.BLOCK_MSG))
        self.assertEqual(blocked.call_args.args[2:], ("Hello there.", "steady rain", "door slam"))
        self.assertEqual(self.events, [])

    def test_the_wan_prompt_of_a_scene_is_known_before_it_runs(self):
        """video_gen.scene_wan_prompt gives exactly the prompt the scene's Wan part gets (a Timeline job encodes
        the prompts of its scenes ahead with it), with one character, several, none, and a style; a scene with
        reference pictures has none ahead (their people's looks come from the language model when it runs)."""
        ben = {"name": "Ben", "slug": "ben", "gender": "man", "look": "A short black beard", "photos": ["ben.png"]}
        cases = (([ROSE], True, PICTURE, ""), ([ROSE, ben], True, "Rose and Ben sit at a table. Rose talks", ""),
                 ([], False, "Rain runs down the window.", ""), ([ROSE], False, PICTURE, "cinematic, film grain"))
        for cast, exact, prompt, style in cases:
            with self.subTest(prompt=prompt, style=style):
                self.wan_prompts.clear()
                with patch.object(video_gen.cast_mod, "compose_frame",
                                  side_effect=lambda *args, **kw: (Path(args[5]).write_bytes(b"png"), (True, str(args[5])))[1]):
                    output = video_gen.generate(prompt, 3, "Portrait 9:16", seed=123, scene_dir=self.root / "s",
                                                cast=cast, exact_cast=exact, style=style, sound_on=False)
                self.assertTrue(output.ok, output.error)
                self.assertEqual(self.wan_prompts, [video_gen.scene_wan_prompt(prompt, style, cast, exact_cast=exact)])
        picture = self.root / "ref.png"
        picture.write_bytes(b"png")
        self.assertIsNone(video_gen.scene_wan_prompt("A man [1] waves.", "", [ROSE], True, [str(picture)]))
        self.assertIsNotNone(video_gen.scene_wan_prompt("A man waves.", "", [ROSE], True, [str(self.root / "gone.png")]))

    def test_the_video_tab_does_not_take_scene_sound(self):
        output = video_gen.generate("A cat on a sofa.", 3, "Portrait 9:16", cues=(Cue("Hi."),))
        self.assertFalse(output.ok)
        self.assertIn("need a Timeline scene", output.error)

    # ----------------------------- styles (app/styles.py) -----------------------------

    def test_a_stylized_scene_keeps_its_picture_when_lip_sync_finds_no_face(self):
        """MuseTalk was trained on real faces: an animated scene whose face it cannot use keeps its picture and its
        speech, and says so in its notes (the Timeline job log)."""
        from app import styles

        failed = (False, "No face could be lip-synced (no face found)", [])
        with patch.object(video_gen.lipsync_proc, "run_dialogue", return_value=failed), \
                patch.object(video_gen.engine_runtime, "talk_installed", return_value=False):
            output = self.generate(style=styles.get(styles.ANIMATION_3D).video)
            self.assertTrue(output.ok, output.error)
            self.assertIn("Lip sync skipped (No face could be lip-synced", output.info["lipsync_skipped"])
            self.assertIn(output.info["lipsync_skipped"], output.notes)
            self.assertEqual(output.info["style"], styles.ANIMATION_3D)
            self.assertEqual(output.info["dialogue"], ["Hello there."])  # the speech still plays
            self.assertTrue(output.audio_stems[1].is_file())
            # a realistic scene still fails, so the next attempt is a new take with a usable face
            realistic = self.generate("realistic", style=styles.get(styles.REALISTIC).video)
        self.assertFalse(realistic.ok)
        self.assertIn("Lip sync failed", realistic.error)

    def test_a_stylized_scene_is_composed_and_prompted_in_its_style(self):
        from app import styles

        look = styles.get(styles.CARTOON_2D)
        with patch.object(video_gen.engine_runtime, "talk_installed", return_value=False):
            output = self.generate(style=look.video)
        self.assertTrue(output.ok, output.error)
        self.assertTrue(self.wan_prompts[0].endswith(look.video))
        render = video_gen.characters.render
        self.assertEqual(render.call_args.kwargs.get("style"), styles.CARTOON_2D)
        # MuseTalk would paint a real mouth on a 2D face: it does not run, the speech plays, the notes say so
        self.assertNotIn("lipsync", self.events)
        self.assertIn("MuseTalk cannot animate a drawn mouth", output.info["lipsync_skipped"])
        self.assertEqual(output.info["dialogue"], ["Hello there."])
        self.generate("again", style=styles.get(styles.REALISTIC).video)
        self.assertNotIn("style", render.call_args.kwargs)  # a realistic start frame keeps the call it had

    def test_an_animated_speaker_alone_talks_through_infinitetalk(self):
        """One speaker in an animated scene: InfiniteTalk makes the whole talking picture at the lip-sync rate from
        the start frame and the speech, so neither Wan nor MuseTalk runs; a failure falls back to them."""
        from app import styles

        calls = []

        def talk(_client, start, speech, output, **options):
            calls.append((Path(start).name, Path(speech).read_bytes()[:4], options))
            Path(output).write_bytes(b"talking")

        look = styles.get(styles.ANIMATION_3D)
        with patch.object(video_gen.engine_runtime, "talk_installed", return_value=True), \
                patch.object(video_gen.engine_runtime, "talk_part", side_effect=talk):
            output = self.generate(style=look.video)
        self.assertTrue(output.ok, output.error)
        self.assertNotIn("wan", self.events)
        self.assertNotIn("lipsync", self.events)
        self.assertEqual(output.info["lipsync"], "infinitetalk")
        self.assertEqual(self.retimed, [3.0])  # 25 -> 30 FPS as every talking picture
        start, riff, options = calls[0]
        self.assertEqual((riff, options["seconds"], options["seed"]), (b"RIFF", 3, 123))
        self.assertTrue(options["prompt"].endswith(look.video))
        self.events.clear()
        with patch.object(video_gen.engine_runtime, "talk_installed", return_value=True), \
                patch.object(video_gen.engine_runtime, "talk_part", side_effect=RuntimeError("out of memory")):
            fallback = self.generate("fallback", style=look.video)
        self.assertTrue(fallback.ok, fallback.error)
        self.assertIn("wan", self.events)
        self.assertIn("lipsync", self.events)
        self.assertTrue(any("could not be made with InfiniteTalk" in note for note in fallback.notes))
        self.assertNotIn("lipsync", fallback.info)
        with patch.object(video_gen.engine_runtime, "talk_part") as unused:  # a realistic film never uses it
            self.generate("realistic", style=styles.get(styles.REALISTIC).video)
        unused.assert_not_called()


    def test_a_2d_shot_with_nothing_to_start_from_is_drawn_in_the_style_then_made_from_the_picture(self):
        """The 2D style's AniSora experts are image to video only: a shot without a character, picture or reference gets
        a start frame drawn by Z-Image with the style's image text and LoRA; a frame that cannot be drawn leaves the shot
        to text to video, with a note. Other styles keep text to video."""
        from app import styles

        drawn = []

        def render(prompt, width, height, seed, steps, progress=None, *, runtime=None, **options):
            drawn.append((prompt, (width, height), options.get("style")))
            return b"png"

        look = styles.get(styles.CARTOON_2D)
        with patch.object(video_gen.engine_runtime, "zimage_files", return_value=object()),                 patch.object(video_gen.engine_runtime, "render", side_effect=render):
            output = video_gen.generate("A city street at sunset, a train passes", 3, "Landscape 16:9", seed=5,
                                        scene_dir=self.root / "drawn", cast=[], exact_cast=True, style=look.video,
                                        sound_on=False)
            self.assertTrue(output.ok, output.error)
            prompt, size, style = drawn[0]
            self.assertEqual((size, style), ((1280, 720), styles.CARTOON_2D))
            self.assertTrue(prompt.endswith(look.image))
            kind, start, wan_style = self.wan_calls[-1]
            self.assertEqual((kind, wan_style), ("i2v", styles.CARTOON_2D))
            self.assertTrue(start.endswith("_start1.png"))
            plain = video_gen.generate("A city street at sunset", 3, "Landscape 16:9", seed=5,
                                       scene_dir=self.root / "plain", cast=[], exact_cast=True,
                                       style=styles.get(styles.ANIMATION_3D).video, sound_on=False)
        self.assertTrue(plain.ok, plain.error)
        self.assertEqual(len(drawn), 1)  # the 3D style keeps text to video
        self.assertEqual(self.wan_calls[-1][:2], ("t2v", None))
        with patch.object(video_gen.engine_runtime, "zimage_files", return_value=object()),                 patch.object(video_gen.engine_runtime, "render", side_effect=RuntimeError("out of memory")):
            fallback = video_gen.generate("A city street", 3, "Landscape 16:9", seed=5, scene_dir=self.root / "fb",
                                          cast=[], exact_cast=True, style=look.video, sound_on=False)
        self.assertTrue(fallback.ok, fallback.error)
        self.assertEqual(self.wan_calls[-1][0], "t2v")
        self.assertTrue(any("could not be drawn" in note for note in fallback.notes))
        self.assertIn("could not be drawn", fallback.info["start_frame_note"])  # a Timeline job reports it
        self.assertNotIn("start_frame_note", output.info)

    def test_the_2d_cartoon_and_the_owners_styles_follow_their_starting_point(self):
        """The 2D cartoon draws its start frames and animates them like Anime; a style of the owner's own runs its
        starting point's models with its own words, and a Realistic-based one keeps the realistic lip sync."""
        from app import styles

        drawn = []

        def render(prompt, width, height, seed, steps, progress=None, *, runtime=None, **options):
            drawn.append((prompt, options.get("style")))
            return b"png"

        watercolor = styles.get(styles.custom_id(styles.ANIME, "soft pastel watercolor"))
        for look in (styles.get(styles.CARTOON), watercolor):
            with self.subTest(style=look.id), \
                    patch.object(video_gen.engine_runtime, "zimage_files", return_value=object()), \
                    patch.object(video_gen.engine_runtime, "render", side_effect=render):
                output = video_gen.generate("A city street at sunset", 3, "Landscape 16:9", seed=5,
                                            scene_dir=self.root / look.family, cast=[], exact_cast=True,
                                            style=look.video, sound_on=False)
                self.assertTrue(output.ok, output.error)
                prompt, style = drawn[-1]
                self.assertEqual(style, look.id)
                self.assertTrue(prompt.endswith(look.image))
                self.assertEqual(self.wan_calls[-1][::2], ("i2v", look.id))
                self.assertEqual(output.info["style"], look.id)
        noir = styles.get(styles.custom_id(styles.REALISTIC, "black and white film noir, hard shadows"))
        failed = (False, "No face could be lip-synced (no face found)", [])
        with patch.object(video_gen.lipsync_proc, "run_dialogue", return_value=failed), \
                patch.object(video_gen.engine_runtime, "talk_installed", return_value=True), \
                patch.object(video_gen.engine_runtime, "talk_part") as talk:
            output = self.generate("noir", style=noir.video)
        talk.assert_not_called()  # a real-looking face talks through MuseTalk...
        self.assertFalse(output.ok)
        self.assertIn("Lip sync failed", output.error)  # ...and a face it cannot use needs a new take
        self.assertEqual(video_gen.characters.render.call_args.kwargs.get("style"), noir.id)

if __name__ == "__main__":
    unittest.main()
