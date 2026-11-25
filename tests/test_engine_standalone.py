# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The standalone product wiring, using local files and CPU doubles at model boundaries."""
import hashlib
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from app import cancel, config, engine_runtime, llm, model_manager, queue_manager, refs, setup_windows


class StandaloneTests(unittest.TestCase):
    def test_catalog_uses_the_configured_model_directory(self):
        catalog = config.load_model_catalog("local-models")
        self.assertTrue(all(Path(path).parts[0] == "local-models" for path in catalog["destinations"].values()))
        files = {model["id"]: model["files"] for model in catalog["models"]}
        self.assertTrue(any(spec["filename"] == "z-image-turbo-Q8_0.gguf" for spec in files["z-image-turbo-lite"]))
        rife = next(spec for spec in files["wan22-a14b-i2v-lite"] if spec["dest"] == "rife")
        self.assertEqual((rife["save_as"], rife["equivalents"][0]["filename"]), ("flownet_v4.9.pkl", "rife49.pth"))
        self.assertTrue(any(spec["dest"] == "tokenizer_qwen_edit" for spec in files[engine_runtime.QWEN_EDIT_ID]))
        self.assertEqual(Path(catalog["destinations"]["wan_text_encoder"]).parts[0], "local-models")

    def test_standalone_installer_skips_git_checkout_and_pins_llama_wheel(self):
        with (tempfile.TemporaryDirectory() as folder,
              patch.object(setup_windows, "ROOT", Path(folder)), patch.dict(setup_windows.os.environ),
              patch.object(setup_windows, "constraints_text", return_value=""),
              patch.object(setup_windows, "voxcpm_requirements", return_value=[]),
              patch.object(setup_windows, "check_environment", return_value=True),
              patch.object(config, "ensure_dirs"), patch.object(setup_windows, "run") as run):
            setup_windows.install()
        self.assertTrue(all(call.args[0][0] == setup_windows.sys.executable for call in run.call_args_list))
        self.assertTrue(any(setup_windows.LLAMA_WHEEL in call.args[0] for call in run.call_args_list))
        self.assertIn("#sha256=", setup_windows.LLAMA_WHEEL)

    def test_native_refs_write_the_engine_picture_without_a_review(self):
        client = Mock()
        unused = AssertionError("A composition needs no assistant")
        with (tempfile.TemporaryDirectory() as folder,
              patch.object(engine_runtime, "edit", return_value=b"native png") as edit,
              patch.object(llm, "ensure", side_effect=unused), patch.object(llm._OWN_LLM, "ask", side_effect=unused)):
            output = Path(folder) / "nested" / "reference.png"
            self.assertEqual(refs.compose(client, ["source.png"], "A tree in spring", 32, 32, output, 42),
                             (True, str(output)))
            self.assertEqual(output.read_bytes(), b"native png")
            self.assertEqual(sorted(item.name for item in Path(folder).rglob("*")), ["nested", "reference.png"])
        edit.assert_called_once_with(["source.png"], "A tree in spring", 32, 32, 42, None, runtime=client)

    def test_musetalk_keeps_whisper_and_the_vae_in_its_own_folders(self):
        """MuseTalk's Whisper tiny and VAE live in its own folders and are installed only there: the two folders of
        earlier installations that docs/GUIDE.md names are not looked in, and nothing is linked from them (an
        upgrader moves the files by hand)."""
        guide = (Path(config.ROOT) / "docs" / "GUIDE.md").read_text(encoding="utf-8")
        folders = re.search(r"no longer looks for them in the old folders `models\\(\w+)` and `models\\(\w+)`", guide)
        self.assertIsNotNone(folders)
        model = config.get_model("musetalk-15")
        earlier = {"lipsync_whisper": folders[1], "lipsync_vae": folders[2]}
        specs = [spec for spec in model["files"] if spec["dest"] in earlier]
        self.assertEqual(len(specs), 3)
        for spec in specs:
            self.assertEqual(Path(config.CATALOG["destinations"][spec["dest"]]).parts[-2], "musetalk")
        with (tempfile.TemporaryDirectory() as folder, patch.object(config, "ROOT", Path(folder)),
              patch.object(config, "STATE_ROOT", Path(folder)),
              patch.dict(config.CATALOG, {"destinations": config.load_model_catalog(folder)["destinations"]})):
            pinned = []
            for spec in model["files"]:
                name = Path(spec.get("save_as") or spec["filename"]).name
                pinned.append(dict(spec, size_bytes=len(name), sha256=hashlib.sha256(name.encode()).hexdigest()))
                old = Path(folder) / "models" / earlier[spec["dest"]] / name if spec in specs else None
                path = old or model_manager._file_target(spec)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            self.assertTrue(all((Path(folder) / "models" / old).is_dir() for old in earlier.values()))
            self.assertFalse(model_manager.installed({"files": pinned}))  # the three files are not in their folders
            self.assertFalse(any(model_manager._file_target(spec).exists() for spec in specs))  # nothing linked

    def test_running_queue_does_not_give_a_background_vision_request_its_gpu_or_cancel_token(self):
        with (patch.object(llm, "ensure", return_value=None),
              patch.object(llm, "_catalog_file", return_value="model.gguf"),
              patch.object(llm, "_OWN_LLM") as worker,
              patch.object(queue_manager, "paused", return_value=False),
              patch.object(queue_manager, "snapshot", return_value=[SimpleNamespace(status=queue_manager.RUNNING)])):
            llm._vision("photo.png", "check", 30)
            options = worker.ask.call_args.kwargs
            self.assertFalse(options["gpu"])
            self.assertIsNone(options["cancel"])
            with patch.object(queue_manager, "in_worker", return_value=True):
                llm._vision("photo.png", "check", 30)
                options = worker.ask.call_args.kwargs
                self.assertTrue(options["gpu"])
                self.assertIs(options["cancel"], cancel.TOKEN)

    def test_native_clip_continues_from_lossless_vae_frame_and_exports_at_30_fps(self):
        """7 s are two even parts (4 + 3 s, 65 and 49 Wan frames), not 5 + 2; the second starts from the
        first one's last VAE frame and the joined clip has exactly 7 x 30 frames."""
        try:
            import av
            import torch
        except ImportError:
            self.skipTest("torch is needed for synthetic frames")
        from app import ffmpeg_utils, history, video_gen
        from engine.media.retime import frame_rate_plan

        model = MagicMock()  # the engine

        def generate(_files, _prompt, width, height, length, *_args, **_kwargs):
            color = torch.linspace(0, 0.8, length).reshape(-1, 1, 1, 1)
            return color.expand(length, 3, height, width)

        def interpolate(_weights, frames, source, target, _progress):
            _tasks, layout = frame_rate_plan(len(frames), source, target)
            return frames[[index for index, _slot in layout]]

        model.video.side_effect = generate
        model.interpolate.side_effect = interpolate
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {("wan_text_encoder", "index"): "model.safetensors.index.json",
                     ("diffusion_models", "highnoise"): "high.gguf",
                     ("diffusion_models", "lownoise"): "low.gguf", ("wan_vae",): "vae.safetensors",
                     ("loras", "high_noise"): "high_lora.safetensors",
                     ("loras", "low_noise"): "low_lora.safetensors"}
            for name in paths.values():
                (root / name).touch()
            with (patch.object(config, "OUTPUTS_DIR", root / "outputs"),
                  patch.object(model_manager, "ensure", return_value=None),
                  # nothing else is installed (the optional FP8 image-to-video experts among it)
                  patch.object(model_manager, "model_path",
                               side_effect=lambda _, *key: root / paths[key] if key in paths else None),
                  patch.object(video_gen, "sizes", return_value=((32, 32), (32, 32))),
                  patch.object(engine_runtime, "engine", return_value=model),
                  patch.object(engine_runtime, "_rife_weights", return_value=root / "rife49.pth"),
                  patch.object(engine_runtime, "small_gpu", return_value=False),
                  patch.object(history, "record") as record):
                result = video_gen.generate("A blue cup sits on a table.", 7, "Landscape 16:9", sound_on=False)
            self.assertTrue(result.ok, result.error)
            self.assertEqual(ffmpeg_utils.video_fps(result.files[0]), 30.0)
            with av.open(result.files[0]) as clip:
                self.assertEqual(sum(1 for _ in clip.decode(video=0)), 7 * 30)
            # The clip is recorded as soon as it is assembled: no frame is sampled for a check, so it is unchecked,
            # never a pass (defect O).
            self.assertEqual(record.call_args.args[0]["qc"], {"ok": False, "reasons": [], "face": None,
                                                              "status": "unchecked"})
        first, second = model.video.call_args_list
        self.assertEqual((first.args[0].text_encoder.name, first.args[0].vae.name),
                         ("model.safetensors.index.json", "vae.safetensors"))
        self.assertEqual((first.args[4], second.args[4]), (65, 49))
        self.assertEqual([call.args[3] for call in model.interpolate.call_args_list], [30, 30])  # 16 -> 30 once
        self.assertEqual((first.args[0].kind, second.args[0].kind), ("t2v", "i2v"))
        self.assertIsNone(first.kwargs["start_image"])
        self.assertEqual(second.kwargs["start_image"].getpixel((0, 0)), (204, 204, 204))

    def test_a_talking_clip_is_interpolated_once_to_25_fps_and_once_to_30_fps(self):
        """Wan 16 FPS -> RIFE 25 FPS per part -> lip sync on exactly 7 x 25 frames, never resampled -> RIFE
        25 -> 30 FPS as the last picture step: 7 x 30 frames with the dialogue in sync. When lip sync is
        skipped, the picture still reaches 30 FPS the same way."""
        try:
            import av
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("torch is needed for synthetic frames")
        from app import ffmpeg_utils, history, sound, tts, video_gen, voices
        from engine.media.audio import write_wav
        from engine.media.retime import frame_rate_plan

        def tone(path, seconds):
            t = np.arange(int(seconds * 48000)) / 48000
            write_wav(path, (0.2 * np.sin(2 * np.pi * 330 * t))[None, :].astype(np.float32), 48000)
            return True, str(path)

        def frames_of(path):
            with av.open(str(path)) as clip:
                return sum(1 for _ in clip.decode(video=0))

        for synced in (True, False):
            model, seen = MagicMock(), []

            def generate(_files, _prompt, width, height, length, *_args, **_kwargs):
                return torch.linspace(0, 0.8, length).reshape(-1, 1, 1, 1).expand(length, 3, height, width)

            def interpolate(_weights, frames, source, target, _progress):
                _tasks, layout = frame_rate_plan(len(frames), source, target)
                return frames[[index for index, _slot in layout]]

            def lip_sync(video, _voice, timings, _who, output, *_args, **_kwargs):
                seen.append((ffmpeg_utils.video_fps(video), frames_of(video), list(timings)))
                if not synced:
                    return False, "No face in the frame is large enough for lip sync", []
                shutil.copyfile(video, output)  # lip sync keeps every frame and the rate
                return True, output, []

            def speak(items, _progress=None):
                for _text, _voice, out in items:
                    tone(out, 1.0)
                return True, ""

            model.video.side_effect = generate
            model.interpolate.side_effect = interpolate
            with self.subTest(lip_sync=synced), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                sample = root / "hanna.wav"
                tone(sample, 1.0)
                hanna = voices.Voice("Hanna", "hanna", "female", str(sample), builtin=True)
                with (patch.object(config, "OUTPUTS_DIR", root / "outputs"),
                      patch.object(model_manager, "ensure", return_value=None),
                      patch.object(model_manager, "model_path", side_effect=lambda _, *key: root / "weights"),
                      patch.object(video_gen, "sizes", return_value=((32, 32), (32, 32))),
                      patch.object(engine_runtime, "engine", return_value=model),
                      patch.object(engine_runtime, "_rife_weights", return_value=root / "rife49.pth"),
                      patch.object(engine_runtime, "small_gpu", return_value=False),
                      patch.object(voices, "all_voices", return_value=[hanna]),
                      patch.object(tts, "synthesize_many", side_effect=speak),
                      patch.object(sound, "generate",
                                   side_effect=lambda _c, _s, _x, seconds, _seed, out, _p=None: tone(out, seconds)),
                      patch.object(video_gen.lipsync_proc, "run_dialogue", side_effect=lip_sync),
                      patch.object(history, "record")):
                    result = video_gen.generate('@Hanna says "Keep it dry."', 7, "Landscape 16:9")
                self.assertTrue(result.ok, result.error)
                self.assertEqual(seen, [(25.0, 7 * 25, [(0.3, 1.3)])])
                self.assertEqual([call.args[2:4] for call in model.interpolate.call_args_list],
                                 [(16, 25), (16, 25), (25, 30)])  # each part, then the whole picture once
                self.assertEqual(ffmpeg_utils.video_fps(result.files[0]), 30.0)
                self.assertEqual(frames_of(result.files[0]), 7 * 30)
                self.assertAlmostEqual(ffmpeg_utils.probe_duration(result.files[0]), 7.0, places=1)
                with av.open(result.files[0]) as clip:
                    self.assertEqual(len(clip.streams.audio), 1)
                self.assertEqual(result.info["fps"], 30)
                self.assertEqual(any("Lip sync skipped" in note for note in result.notes), not synced)


class MemoryTests(unittest.TestCase):
    def test_cache_reuses_weights_evicts_by_size_and_invalidates_replacements(self):
        try:
            import torch
            from engine.core.memory import ModelCache
        except ImportError:
            self.skipTest("engine libraries unavailable")

        loader = Mock(side_effect=lambda *_: torch.nn.Linear(4, 4, bias=False))
        with tempfile.TemporaryDirectory() as folder:
            first, second = (Path(folder) / name for name in ("first.safetensors", "second.safetensors"))
            first.write_bytes(b"original")
            second.write_bytes(b"second")
            cache = ModelCache(capacity_bytes=64)
            cache.prepare("zimage")

            def load(path):
                return cache.load(path, "dit", loader, "cpu", torch.float32)

            original = load(first)
            cache.release(original)
            self.assertEqual(cache.bytes, 64)
            self.assertIs(load(first), original)
            cache.release(original)
            first.write_bytes(b"changed weight file")
            updated = load(first)
            self.assertIsNot(updated, original)
            cache.release(updated)
            cache.release(load(second))
            self.assertIsNot(load(first), updated)
            cache.prepare("wan")
            self.assertEqual(cache.bytes, 0)

    def test_a_spare_text_encoder_never_pushes_the_experts_out(self):
        """Wan's sequence with room for the experts and the VAE but not also the text encoder: the encoder of
        the next scene's new prompt is used and dropped, the experts stay for every later part."""
        try:
            import torch
            from engine.core.memory import ModelCache
        except ImportError:
            self.skipTest("engine libraries unavailable")

        sizes = {"text": 6, "vae": 1, "high": 5, "low": 5}  # x 16 bytes: a Linear(4, units) in float32
        loads = []

        def loader(path, _device, _dtype):
            loads.append(Path(path).stem)
            return torch.nn.Linear(4, sizes[Path(path).stem], bias=False)

        with tempfile.TemporaryDirectory() as folder:
            files = {}
            for name, units in sizes.items():
                files[name] = Path(folder) / f"{name}.bin"
                files[name].write_bytes(b"x" * 16 * units)
            cache = ModelCache(capacity_bytes=16 * 12)  # the experts and the VAE (11 units), not the encoder too
            cache.prepare("wan")

            def part(with_text: bool):
                if with_text:
                    cache.release(cache.load(files["text"], "text", loader, "cpu", torch.float32, spare=True))
                for name in ("vae", "high", "low", "vae"):
                    cache.release(cache.load(files[name], name, loader, "cpu", torch.float32))

            part(True)  # the first scene: the encoder is kept while it fits, then gives way to the low expert
            self.assertEqual(loads, ["text", "vae", "high", "low"])
            loads.clear()
            part(False)  # the next part of the clip: its prompt's encoding is kept, everything else is warm
            self.assertEqual(loads, [])
            part(True)  # the next scene: a new prompt loads the encoder, which is not kept beside the experts
            part(False)
            self.assertEqual(loads, ["text"])
            self.assertLessEqual(cache.bytes, cache.capacity_bytes)

    def test_experts_that_cannot_both_be_kept_are_dropped_after_use_and_the_vae_stays(self):
        """Two FP8 experts (14 GiB each) and a 24 GiB cache, scaled down: kept, the idle low expert would only be
        pushed out by the next scene's high one. WanModels asks `fits` and loads them with keep=False instead, so
        nothing idles uselessly, the cache's bound still covers the expert that runs, and the VAE is reused."""
        try:
            import torch
            from engine.core.memory import ModelCache
        except ImportError:
            self.skipTest("engine libraries unavailable")

        sizes = {"vae": 1, "high": 14, "low": 14}  # x 16 bytes
        loads = []

        def loader(path, _device, _dtype):
            loads.append(Path(path).stem)
            return torch.nn.Linear(4, sizes[Path(path).stem], bias=False)

        with tempfile.TemporaryDirectory() as folder:
            files = {}
            for name, units in sizes.items():
                files[name] = Path(folder) / f"{name}.bin"
                files[name].write_bytes(b"x" * 16 * units)
            cache = ModelCache(capacity_bytes=16 * 24)
            cache.prepare("wan")
            self.assertFalse(cache.fits(files["high"], files["low"]))
            self.assertTrue(cache.fits(files["high"], files["vae"]))
            for _scene in range(3):
                for name in ("vae", "high", "low", "vae"):
                    model = cache.load(files[name], name, loader, "cpu", torch.float32, keep=name == "vae")
                    self.assertLessEqual(cache.bytes + 16 * sizes[name], cache.capacity_bytes)
                    cache.release(model)
                self.assertEqual(cache.bytes, 16 * sizes["vae"])  # no idle expert between scenes
            self.assertEqual(loads, ["vae", "high", "low", "high", "low", "high", "low"])

    def test_wan_keeps_its_experts_only_when_both_fit(self):
        try:
            import torch
            from engine.core.memory import ModelCache
            from engine.pipelines import video
        except ImportError:
            self.skipTest("engine libraries unavailable")

        with tempfile.TemporaryDirectory() as folder:
            files = {}
            for name, units in {"high": 5, "low": 7, "vae": 1}.items():
                files[name] = Path(folder) / f"{name}.bin"
                files[name].write_bytes(b"x" * units)
            for capacity, kept in ((12, True), (11, False)):
                with self.subTest(capacity=capacity):
                    cache = Mock(wraps=ModelCache(capacity))
                    network = Mock(side_effect=lambda *_args, **_options: torch.nn.Linear(1, 1))
                    with patch.object(video.text, "check_format"), patch.object(video.text, "files", return_value=()), \
                            patch.object(video.transformer, "load_transformer", network), \
                            patch.object(video.vae, "load_vae", network):
                        models = video.WanModels.from_files("text", files["high"], files["low"], files["vae"], "i2v",
                                                            cache=cache)
                        for expert in ("high", "low"):
                            models.load_transformer(expert, "cpu", None)
                            self.assertEqual(cache.load.call_args.kwargs["keep"], kept)
                        models.load_vae("cpu", None)
                        self.assertTrue(cache.load.call_args.kwargs["keep"])

    def test_speech_sound_and_lip_sync_leave_warm_models_and_trim_keeps_them(self):
        from engine.api import Engine

        engine = Engine(device="cpu")
        engine._cache = Mock()
        engine._embeddings = Mock()
        with patch("engine.pipelines.speech.synthesize_many", return_value="line.wav"), \
                patch("engine.pipelines.ezaudio.generate", return_value=("samples", 48000)), \
                patch("engine.pipelines.musetalk.lipsync", return_value="frames"), \
                patch("engine.pipelines.musetalk.musetalk", return_value="frames"), \
                patch("engine.pipelines.musetalk.MuseTalkModels.from_files"):
            engine.tts(Path("voxcpm2"), [], isolated=False)
            engine.audio_bg(Path("ezaudio"), "rain", "", 3.0, 1)
            engine.lipsync(Path("unet"), Path("whisper"), Path("vae"), "frames", "audio", None, low_vram=False)
            engine.lipsync_tracks(Path("unet"), Path("whisper"), Path("vae"), "frames", [], low_vram=False)
            engine.trim()
        engine._cache.prepare.assert_not_called()
        engine._cache.clear.assert_not_called()
        engine._embeddings.clear.assert_not_called()
        engine.free()  # a pause, an idle queue, an error: the models go, the prompt encodings (a few MB) stay
        engine._cache.clear.assert_called_once_with()
        engine._embeddings.clear.assert_not_called()
        engine.forget_prompts()
        engine._embeddings.clear.assert_called_once_with()

    def test_too_little_face_for_lip_sync_keeps_the_warm_models(self):
        """MuseTalk refuses a clip whose speech mostly lacks a usable face before it loads anything (the app then
        tries a crop of the face): that answer must not cost Wan's warm experts. Any other failure still frees."""
        from engine.api import Engine
        from engine.pipelines import musetalk

        engine = Engine(device="cpu")
        engine._cache = Mock()
        refused = musetalk.CoverageError("Lip sync covered 0 of 3 speech frames", musetalk.Report())
        with patch("engine.pipelines.musetalk.lipsync", side_effect=refused), \
                patch("engine.pipelines.musetalk.MuseTalkModels.from_files"):
            with self.assertRaises(musetalk.CoverageError):
                engine.lipsync(Path("unet"), Path("whisper"), Path("vae"), "frames", "audio", None, low_vram=False)
        engine._cache.clear.assert_not_called()
        with patch("engine.pipelines.musetalk.lipsync", side_effect=RuntimeError("CUDA out of memory")), \
                patch("engine.pipelines.musetalk.MuseTalkModels.from_files"):
            with self.assertRaises(RuntimeError):
                engine.lipsync(Path("unet"), Path("whisper"), Path("vae"), "frames", "audio", None, low_vram=False)
        engine._cache.clear.assert_called_once_with()

    def test_an_audio_session_keeps_ezaudio_in_its_own_cache_until_it_ends(self):
        """A scene's bed and effects share one host cache of EzAudio's networks, apart from the warm cache (which
        keeps Wan's experts); the networks are dropped when the session ends, also when it is nested."""
        from engine.api import SOUND_SESSION_BYTES, Engine

        engine = Engine(device="cpu")
        engine._cache = Mock()
        caches = []
        with patch("engine.pipelines.ezaudio.generate",
                   side_effect=lambda *args, cache=None, **kwargs: caches.append(cache) or ("samples", 48000)):
            engine.audio_bg(Path("ezaudio"), "rain", "", 3.0, 1)
            with engine.audio_session():
                engine.audio_bg(Path("ezaudio"), "rain", "", 3.0, 1)
                with engine.audio_session():
                    engine.audio_bg(Path("ezaudio"), "door", "", 1.0, 2)
                session = engine._sound_cache
            self.assertIsNone(engine._sound_cache)
        self.assertIsNone(caches[0])
        self.assertIs(caches[1], session)
        self.assertIs(caches[2], session)
        self.assertEqual(session.capacity_bytes, SOUND_SESSION_BYTES)
        self.assertGreaterEqual(SOUND_SESSION_BYTES, 4_400_000_000)  # FLAN-T5-XL, the transformer and the decoder
        engine._cache.prepare.assert_not_called()
        engine._cache.clear.assert_not_called()

    def test_models_stay_warm_automatically_only_on_machines_with_64_gb(self):
        for setting, ram, expected in (("auto", 63.4, engine_runtime.AUTO_WARM_CACHE_GIB), ("auto", 31.8, 0),
                                       ("auto", 0.0, 0), (0, 63.4, 0), (36, 31.8, 36)):
            with self.subTest(setting=setting, ram=ram), \
                    patch.dict(config.SETTINGS["engine"], {"warm_cache_gib": setting}), \
                    patch.object(engine_runtime, "_ram_gib", return_value=ram):
                self.assertEqual(engine_runtime.warm_cache_gib(), expected)
        self.assertGreaterEqual(engine_runtime.AUTO_WARM_CACHE_GIB << 30,
                                2 * 12003652096 + 253815318)  # the Full set's two Q6_K experts and the VAE

    def test_the_queue_gives_warm_models_back_only_when_nothing_waits(self):
        job = queue_manager.Job(5, "wideo", "next", {"seconds": 5})
        for jobs, paused, released in (([], False, True), ([job], False, False), ([job], True, True)):
            job.status = queue_manager.PENDING
            with self.subTest(waiting=len(jobs), paused=paused), \
                    patch.object(queue_manager, "_JOBS", list(jobs)), \
                    patch.dict(queue_manager._STATE_INFO, {"paused": paused}), \
                    patch.object(engine_runtime, "release") as release:
                queue_manager._release_when_idle()
                self.assertEqual(release.called, released)

    def test_the_worker_gives_warm_models_back_after_the_last_job(self):
        """The worker's step for one job (queue_manager._process, what _worker runs) ends by giving the warm models
        back when no job waits, and keeps them for a job that is waiting."""
        for waiting, released in ((False, True), (True, False)):
            job = queue_manager.Job(1, "test", "now", {}, status=queue_manager.RUNNING)
            jobs = [job] + ([queue_manager.Job(2, "test", "next", {})] if waiting else [])
            runner = Mock(return_value=(True, [], [], ""))
            with self.subTest(waiting=waiting), tempfile.TemporaryDirectory() as folder, \
                    patch.object(queue_manager, "_JOBS", jobs), \
                    patch.dict(queue_manager._STATE_INFO, {"paused": False}), \
                    patch.dict(queue_manager._RUNNERS, {"test": runner}), \
                    patch.object(queue_manager, "_save"), \
                    patch.object(queue_manager, "FILES_DIR", Path(folder) / "files"), \
                    patch.object(llm, "release_gpu"), \
                    patch.object(engine_runtime, "release") as release:
                queue_manager._process(job)
                runner.assert_called_once()
                self.assertEqual(job.status, queue_manager.DONE)
                self.assertEqual(release.call_count, int(released))

    def test_a_simulated_gpu_limit_never_increases_an_existing_ceiling(self):
        try:
            import torch
            from engine.core.memory import limit_cuda
        except ImportError:
            self.skipTest("engine libraries unavailable")
        for device, index in (("cuda", None), ("cuda:2", 2)):
            with (self.subTest(device=device),
                  patch.object(torch.cuda, "mem_get_info", return_value=(14 << 30, 16 << 30)) as memory,
                  patch.object(torch.cuda, "get_per_process_memory_fraction", side_effect=[1.0, 0.5]) as current,
                  patch.object(torch.cuda, "set_per_process_memory_fraction") as set_limit):
                limit_cuda(device, 8)
                limit_cuda(device, 12)
                self.assertEqual([entry.args for entry in memory.call_args_list], [(index,), (index,)])
                self.assertEqual([entry.args for entry in current.call_args_list], [(index,), (index,)])
                set_limit.assert_called_once_with(0.5, index)
