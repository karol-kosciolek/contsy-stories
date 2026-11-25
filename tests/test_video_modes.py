# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The one video mode: Wan 2.2 A14B with the LightX2V 4-step distillation at 480p, nothing enlarged
afterwards - frame sizes, sampling, the whole pipeline with the engine faked, and the queue."""
import inspect
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app import config, ffmpeg_utils, history, model_manager, prefs
from app import qc, queue_manager, tts, video_gen, voices

# a clip nobody checked is unchecked, never a pass (docs/QUALITY_ARCHITECTURE.md, defect O)
UNCHECKED = {"ok": False, "reasons": [], "face": None, "status": "unchecked"}


def _native_part(runtime, kind, start_image, output, native_fps, fps, **options):
    output.write_bytes(b"mp4")
    return b"png"  # the lossless frame the next part starts from


def _run(seconds=5, small_gpu=False, **options):
    """The whole _generate with the engine and FFmpeg faked: (output, history info, Wan requests,
    ensure mock, combine_video_audio mock). small_gpu=None keeps the real card check."""
    client = MagicMock()
    recorded, built = {}, []
    real_small_gpu = video_gen.engine_runtime.small_gpu

    def render_part(runtime, kind, start_image, output, native_fps, fps, **part):
        built.append({**part, "fps": fps})
        return _native_part(runtime, kind, start_image, output, native_fps, fps, **part)

    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(config, "OUTPUTS_DIR", Path(tmp)), \
            patch.object(video_gen.engine_runtime, "engine", return_value=client), \
            patch.object(model_manager, "ensure", return_value=None) as ensure, \
            patch.object(video_gen.engine_runtime, "video_part", side_effect=render_part), \
            patch.object(video_gen.engine_runtime, "small_gpu",
                         side_effect=real_small_gpu if small_gpu is None else lambda: small_gpu), \
            patch.object(ffmpeg_utils, "convert_fps", return_value=(True, "")), \
            patch.object(ffmpeg_utils, "join_segments", return_value=(True, "")), \
            patch.object(ffmpeg_utils, "combine_video_audio", return_value=(True, "")) as combine, \
            patch.object(ffmpeg_utils, "video_fps", return_value=30.0), \
            patch.object(history, "record", side_effect=recorded.update):
        out = video_gen._generate("A woman walks on a beach.", seconds, "Landscape 16:9", sound_on=False, **options)
    return out, recorded, built, ensure, combine


class ModeTests(unittest.TestCase):
    def test_the_one_mode_renders_480p_in_four_distilled_steps(self):
        self.assertEqual(video_gen.SAMPLING, {"steps": 4, "split": 2, "shift": 5.0, "lora": 1.0})
        self.assertEqual(video_gen.sizes(16 / 9), ((848, 480), (1280, 720)))
        self.assertEqual(video_gen.sizes(9 / 16), ((480, 848), (720, 1280)))
        self.assertEqual(video_gen.RESOLUTION, "480p")

    def test_every_aspect_ratio_has_a_codec_friendly_frame_and_a_larger_starting_frame(self):
        for name, ratio in video_gen.ASPECTS.items():
            (w, h), (start_w, start_h) = video_gen.sizes(ratio)
            self.assertEqual((w % 16, h % 16), (0, 0), name)
            self.assertLess(abs(w / h - ratio), 0.015, name)  # Wan renders nearly the same proportions
            self.assertAlmostEqual(w * h, video_gen.VIDEO_PIXELS, delta=video_gen.VIDEO_PIXELS * 0.05)
            self.assertGreater(start_w * start_h, w * h, name)  # faces are drawn larger, Wan downsizes

    def test_the_service_takes_no_mode_or_step_count(self):
        parameters = inspect.signature(video_gen._generate).parameters
        self.assertNotIn("speed", parameters)
        self.assertNotIn("steps", parameters)


class PartPlanTests(unittest.TestCase):
    def test_parts_are_even_and_never_end_in_a_short_tail(self):
        cases = {1: [1], 5: [5], 6: [6], 7: [4, 3], 8: [4, 4], 11: [6, 5], 12: [6, 6], 13: [5, 4, 4],
                 18: [6, 6, 6], 19: [5, 5, 5, 4], 30: [6] * 5}
        for seconds, parts in cases.items():
            with self.subTest(seconds=seconds):
                self.assertEqual(video_gen._segments(seconds), parts)
        for seconds in range(1, video_gen.MAX_SECONDS + 1):
            for longest in (video_gen.MAX_PART_SECONDS, video_gen.SMALL_GPU_PART_SECONDS):
                parts = video_gen._segments(seconds, longest)
                self.assertEqual(sum(parts), seconds)
                self.assertLessEqual(max(parts) - min(parts), 1)  # as even as whole seconds allow
                self.assertEqual(len(parts), -(-seconds // longest))  # and no more parts than needed
                self.assertTrue(all((16 * part + 1 - 1) % 4 == 0 for part in parts))  # 4k + 1 Wan frames
        # a card under 15 GB keeps the 81 frames it was measured with: 6 s = 3 + 3, not 5 + 1
        self.assertEqual(video_gen._segments(6, video_gen.SMALL_GPU_PART_SECONDS), [3, 3])

    def test_a_6_second_clip_is_one_97_frame_pass(self):
        out, _info, built, _ensure, _combine = _run(seconds=6)
        self.assertTrue(out.ok, out.error)
        self.assertEqual([part["frame_count"] for part in built], [97])
        self.assertEqual([part["fps"] for part in built], [30])  # silent: RIFE 16 -> 30 directly

    def test_longer_clips_are_split_evenly_and_small_cards_keep_81_frames(self):
        for seconds, small, frames in ((12, False, [97, 97]), (7, False, [65, 49]), (6, True, [49, 49]),
                                       (11, True, [65, 65, 49])):
            with self.subTest(seconds=seconds, small_gpu=small):
                out, info, built, _ensure, _combine = _run(seconds=seconds, small_gpu=small)
                self.assertTrue(out.ok, out.error)
                self.assertEqual([part["frame_count"] for part in built], frames)
                self.assertEqual(info["seconds"], seconds)
                self.assertEqual([part["seed"] for part in built], [built[0]["seed"] + k for k in range(len(frames))])

    def test_a_vram_limit_under_15_gib_keeps_81_frames_on_a_large_card(self):
        with patch.dict(config.SETTINGS["engine"], vram_limit_gib=12),                 patch.object(model_manager, "_gpu_total_gb", return_value=24.0):
            self.assertEqual(video_gen.part_seconds_limit(), video_gen.SMALL_GPU_PART_SECONDS)
            out, _info, built, _ensure, _combine = _run(seconds=6, small_gpu=None)
        self.assertTrue(out.ok, out.error)
        self.assertEqual([part["frame_count"] for part in built], [49, 49])


OLA = {"name": "Ola", "slug": "ola", "gender": "woman", "look": "Fair skin, long auburn hair",
       "photos": ["ola.png"]}
KUBA = {"name": "Kuba", "slug": "kuba", "gender": "man", "look": "Short dark hair, a short beard",
        "photos": ["kuba.png"]}


def _run_cast(prompt: str, cast, seconds: int = 5, verdict=None):
    """_generate with characters, the engine and FFmpeg faked: (output, history info, Wan
    requests, compose_frame calls, characters.render mock). verdict: what the face check finds at a handoff."""
    client = MagicMock()
    recorded, built, composed = {}, [], []

    def render_part(runtime, kind, start_image, output, native_fps, fps, **options):
        built.append(options)
        return _native_part(runtime, kind, start_image, output, native_fps, fps, **options)

    def compose(client_, members, text, w, h, out, seed, cb=None, extra_refs=None, looks=None):
        composed.append(([m["slug"] for m in members], text, (w, h)))
        Path(out).write_bytes(b"png")
        return True, str(out)

    def render(client_, ch, text, w, h, out, seed, cb=None):
        Path(out).write_bytes(b"png")
        return True, str(out)

    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(config, "OUTPUTS_DIR", Path(tmp)), \
            patch.object(prefs, "_FILE", Path(tmp) / "prefs.json"), \
            patch.object(video_gen.engine_runtime, "engine", return_value=client), \
            patch.object(model_manager, "ensure", return_value=None), \
            patch.object(video_gen.engine_runtime, "video_part", side_effect=render_part), \
            patch.object(ffmpeg_utils, "convert_fps", return_value=(True, "")), \
            patch.object(ffmpeg_utils, "combine_video_audio", return_value=(True, "")), \
            patch.object(ffmpeg_utils, "video_fps", return_value=30.0), \
            patch.object(video_gen.characters, "render", side_effect=render) as render_frame, \
            patch.object(video_gen.cast_mod, "compose_frame", side_effect=compose), \
            patch.object(ffmpeg_utils, "join_segments", return_value=(True, "")), \
            patch.object(qc, "enabled", return_value=True), \
            patch.object(qc, "check", return_value=verdict or qc.Verdict(True)), \
            patch.object(history, "record", side_effect=recorded.update):
        out = video_gen._generate(prompt, seconds, "Landscape 16:9", sound_on=False,
                                  character=cast[0] if cast else None, cast=cast)
    return out, recorded, built, composed, render_frame


class CastPipelineTests(unittest.TestCase):
    def test_two_named_characters_start_from_one_composed_frame(self):
        out, info, built, composed, render_frame = _run_cast(
            "Ola hands Kuba a coffee in a cafe.", [OLA, KUBA])
        self.assertTrue(out.ok, out.error)
        # the composed starting frame keeps its 1280x720 class size; Wan renders 480p from it
        self.assertEqual(composed, [(["ola", "kuba"], "Ola hands Kuba a coffee in a cafe.", (1280, 720))])
        self.assertEqual((built[0]["width"], built[0]["height"]), (848, 480))
        render_frame.assert_not_called()
        positive = built[0]["prompt"]
        self.assertIn("The woman hands the man a coffee in a cafe", positive)
        self.assertIn("The woman has fair skin, long auburn hair", positive)
        self.assertIn("The man has short dark hair, a short beard", positive)
        self.assertEqual(info["qc"], UNCHECKED)
        self.assertEqual((info["character"], info["characters"]), ("Ola", ["Ola", "Kuba"]))

    def test_a_shot_naming_only_the_second_character_uses_that_character_alone(self):
        out, info, built, composed, render_frame = _run_cast(
            "Kuba walks along the beach.", [OLA, KUBA])
        self.assertTrue(out.ok, out.error)
        self.assertEqual(composed, [])
        self.assertIs(render_frame.call_args.args[1], KUBA)
        self.assertEqual(info["qc"], UNCHECKED)
        self.assertEqual((info["character"], info["characters"]), ("Ola", ["Kuba"]))

    def test_one_character_works_as_before(self):
        out, info, built, composed, render_frame = _run_cast(
            "A woman walks on a beach.", [OLA])
        self.assertTrue(out.ok, out.error)
        self.assertEqual(composed, [])
        self.assertIs(render_frame.call_args.args[1], OLA)
        self.assertEqual(render_frame.call_args.args[3:5], (1280, 720))  # the character's starting frame
        self.assertEqual(built[0]["prompt"],  # exactly as before the cast existed
                         "A woman walks on a beach.. The person has Fair skin, long auburn hair")
        self.assertEqual(info["qc"], UNCHECKED)
        self.assertEqual(info["character"], "Ola")
        self.assertNotIn("characters", info)


    def test_a_longer_shot_continues_from_its_last_frame_and_reports_a_face_that_drifted(self):
        drifted = qc.Verdict(False, ["different face (similarity 0.20 < 0.40)"])
        out, info, built, composed, render_frame = _run_cast("A woman walks on a beach.", [OLA], seconds=12,
                                                               verdict=drifted)
        self.assertTrue(out.ok, out.error)
        self.assertEqual(len(built), 2)
        self.assertEqual(render_frame.call_count, 1)  # the second part starts from the first part's frame
        self.assertTrue(any("moved away from the character between parts (part 2: different face" in note
                            for note in out.notes), out.notes)
        self.assertEqual(info["qc"], {"ok": False, "face": None, "status": "fail",
                                      "reasons": ["face drift at part 2: different face (similarity 0.20 < 0.40)"]})
        out, info, *_ = _run_cast("A woman walks on a beach.", [OLA], seconds=12)
        self.assertEqual(info["qc"], UNCHECKED)
        self.assertFalse(any("moved away" in note for note in out.notes))


class PipelineTests(unittest.TestCase):
    def test_a_starting_frame_hook_draws_the_start_and_its_vision_outage_leaves_no_files(self):
        """A production script draws a Timeline scene's start image itself (starting_frame). Wan starts from
        that image; a vision outage the hook reports stops the clip before Wan and leaves no files."""
        def copy_video(source, destination, *args, **kwargs):
            shutil.copyfile(source, destination)
            return True, ""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outage = MagicMock(side_effect=video_gen.llm.VisualReviewUnavailable("Test vision outage"))
            start = MagicMock(return_value=str(root / "start.png"))
            with (patch.object(model_manager, "ensure", return_value=None),
                  patch.object(video_gen.engine_runtime, "engine", return_value=MagicMock()),
                  patch.object(video_gen.engine_runtime, "video_part", side_effect=_native_part) as wan,
                  patch.object(ffmpeg_utils, "convert_fps", side_effect=copy_video),
                  patch.object(ffmpeg_utils, "combine_video_audio",
                               side_effect=lambda source, _audio, output, **kw: copy_video(source, output)),
                  patch.object(ffmpeg_utils, "probe_duration", return_value=3.0),
                  patch.object(ffmpeg_utils, "video_fps", return_value=30.0),
                  patch.object(history, "record") as record):
                with self.assertRaisesRegex(video_gen.llm.VisualReviewUnavailable, "Test vision outage"):
                    video_gen.generate("A quiet room.", 3, "Portrait 9:16", seed=123, sound_on=False,
                                       scene_dir=root / "first", starting_frame=outage)
                outage.assert_called_once()
                wan.assert_not_called()
                self.assertEqual([path for path in root.rglob("*") if path.is_file()], [])

                output = video_gen.generate("A quiet room.", 3, "Portrait 9:16", seed=123, sound_on=False,
                                            scene_dir=root / "second", starting_frame=start)
                self.assertTrue(output.ok, output.error)
                start.assert_called_once()
                self.assertEqual((wan.call_count, wan.call_args.args[1:3]), (1, ("i2v", str(root / "start.png"))))
                record.assert_not_called()  # a Timeline scene stays private
                self.assertEqual((output.info["width"], output.info["height"]), (480, 848))

    def test_named_and_character_voices_reach_speech_and_history(self):
        """@Marek keeps his voice and the unnamed woman gets the default woman's voice; when
        speech fails the video is still made and history records no voices."""
        client = MagicMock()
        recorded, spoken, events = {}, [], []
        client.trim.side_effect = lambda: events.append("trim")
        client.free.side_effect = lambda: events.append("free")

        def synth(batch, cb=None):
            events.append("speech")
            spoken.extend(batch)
            return False, "Speech synthesis failed: test"

        library = [voices.Voice("Hanna", "hanna", "female", "h.wav", builtin=True),
                   voices.Voice("Zoe", "zoe", "female", "z.wav", builtin=True),
                   voices.Voice("Marek", "marek", "male", "m.wav", builtin=True)]
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(config, "OUTPUTS_DIR", Path(tmp)), \
                patch.object(prefs, "_FILE", Path(tmp) / "prefs.json"), \
                patch.object(video_gen.engine_runtime, "engine", return_value=client), \
                patch.object(model_manager, "ensure", return_value=None), \
                patch.object(video_gen.engine_runtime, "video_part", side_effect=_native_part), \
                patch.object(ffmpeg_utils, "convert_fps", return_value=(True, "")), \
                patch.object(ffmpeg_utils, "combine_video_audio", return_value=(True, "")), \
                patch.object(ffmpeg_utils, "video_fps", return_value=30.0), \
                patch.object(history, "record", side_effect=recorded.update), \
                patch.object(tts, "synthesize_many", side_effect=synth), \
                patch.object(video_gen.sound, "generate", return_value=(False, "no background")), \
                patch.object(voices, "all_voices", return_value=library):
            prefs.save(default_female="zoe")
            out = video_gen._generate("@Marek says „Hi there”. A woman says „Hey you”.", 5,
                                      "Landscape 16:9", sound_on=True)
        self.assertTrue(out.ok, out.error)
        self.assertEqual([(text, voice) for text, voice, _ in spoken],
                         [("Hi there", "marek"), ("Hey you", "zoe")])
        self.assertEqual(recorded["dialogue"], ["Hi there", "Hey you"])
        self.assertEqual(recorded["voices"], [])
        self.assertIn("Speech synthesis failed", " ".join(out.notes))
        self.assertEqual(events, ["trim", "speech"])  # the card for VoxCPM2; warm models stay in RAM (no free())

    def test_a_video_is_made_at_480p_with_the_distilled_sampling(self):
        out, info, built, ensure, combine = _run()
        self.assertTrue(out.ok, out.error)
        self.assertEqual(len(built), 1)
        part = built[0]
        self.assertEqual((part["width"], part["height"], part["frame_count"]), (848, 480, 81))
        self.assertEqual((part["steps"], part["split"], part["shift"], part["lora_strength"]), (4, 2, 5.0, 1.0))
        self.assertNotIn("negative", part)
        self.assertTrue(combine.call_args.kwargs["compress"])  # the lossless intermediate is encoded
        self.assertEqual((info["width"], info["height"], info["fps"], info["steps"]), (848, 480, 30, 4))
        self.assertEqual(info["engine"], "Wan 2.2 A14B")
        self.assertNotIn("speed", info)
        self.assertEqual(out.notes, [])
        # only the Wan bundle is needed: no other model is downloaded for a plain text-to-video
        self.assertEqual([call.args[0] for call in ensure.call_args_list], [[], ["wan22-a14b-t2v"]])


RATES = {"wideo": 30.0, "obraz": 30.0}


class QueueTests(unittest.TestCase):
    def test_every_video_job_shares_one_time_average(self):
        with patch.dict(queue_manager._RATE, RATES):
            new = queue_manager.Job(1, "wideo", "a", {"seconds": 5})
            old = queue_manager.Job(2, "wideo", "b", {"seconds": 5, "speed": "fast"})  # queued with a mode
            self.assertEqual((new.estimate(), old.estimate()), (150.0, 150.0))

    def test_a_saved_queue_keeps_the_preview_average_as_the_video_rate(self):
        """The Preview average becomes the video rate; the Quality and Fast averages and the image
        upscaler's are dropped. An old video job that carries its mode still loads."""
        saved = {"next_id": 8, "paused": False,
                 "rates": {"wideo": 201.0, "wideo-szybkie": 151.0, "wideo-podglad": 85.0, "upscale": 66.0,
                           "obraz": 72.0},
                 "jobs": [{"id": 7, "kind": "wideo", "label": "Video 5 s · fast: rain",
                           "params": {"prompt": "rain", "seconds": 5, "aspect": "Landscape 16:9",
                                      "sound": True, "speed": "fast"},
                           "status": queue_manager.DONE, "progress": 1.0}]}
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "queue.json"
            state.write_text(json.dumps(saved), encoding="utf-8")
            with patch.object(queue_manager, "_STATE", state), \
                    patch.object(queue_manager, "_JOBS", []), \
                    patch.dict(queue_manager._STATE_INFO, {"next_id": 1, "paused": False}), \
                    patch.dict(queue_manager._RATE, {}, clear=True):
                queue_manager._load()
                rates, jobs = dict(queue_manager._RATE), list(queue_manager._JOBS)
        self.assertEqual(rates, {"wideo": 85.0, "obraz": 72.0})
        self.assertEqual([(job.id, job.params["speed"]) for job in jobs], [(7, "fast")])


if __name__ == "__main__":
    unittest.main()
