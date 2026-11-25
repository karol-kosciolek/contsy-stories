# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The product's bridge to the own engine: model file lookup and the shared image helpers."""
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import cancel, characters, engine_runtime, image_gen
from engine.api import REQUIRED_PACKAGES
from engine.core import readahead
from PIL import Image

def png_bytes(size=(64, 48), color=(10, 20, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return buffer.getvalue()


class RuntimeTests(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", *REQUIRED_PACKAGES)),
                         "the engine libraries are not installed here")
    def test_the_engine_needs_a_gpu_and_every_library_the_pipelines_import(self):
        real_find_spec = importlib.util.find_spec
        missing = {"name": None}

        def find_spec(name, *args, **kwargs):
            return None if name == missing["name"] else real_find_spec(name, *args, **kwargs)

        with patch("torch.cuda.is_available", return_value=True), patch("importlib.util.find_spec", find_spec):
            self.assertTrue(engine_runtime.Engine.available())
            for package in ("diffusers", "transformers", "safetensors", "gguf"):
                missing["name"] = package
                with self.subTest(missing=package):
                    self.assertFalse(engine_runtime.Engine.available())
        missing["name"] = None
        with patch("torch.cuda.is_available", return_value=False):
            self.assertFalse(engine_runtime.Engine.available())

    def test_the_files_are_those_of_the_full_model_set_and_all_must_exist(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("t.safetensors", "e.safetensors", "v.safetensors"):
                (root / name).write_bytes(b"x")
            (root / "tok").mkdir()
            (root / "tok" / "vocab.json").write_text("{}")
            found = {"z_image_turbo_transformer": root / "t.safetensors",
                     "z_image_turbo_text_encoder": root / "e.safetensors", "z_image_turbo_vae": root / "v.safetensors"}
            lookup = lambda model_id, dest, tier_name=None: found.get(dest)  # noqa: E731
            with patch.object(engine_runtime.model_manager, "model_path", side_effect=lookup), \
                    patch.object(engine_runtime.model_manager, "tier", return_value="full"), \
                    patch.object(engine_runtime, "tokenizer_dir", return_value=root / "tok"):
                files = engine_runtime.zimage_files()
                self.assertEqual((files.transformer.name, files.text_encoder.name, files.vae.name),
                                 ("t.safetensors", "e.safetensors", "v.safetensors"))
                found["z_image_turbo_vae"] = None
                self.assertIsNone(engine_runtime.zimage_files())
                found["z_image_turbo_vae"] = root / "gone.safetensors"
                self.assertIsNone(engine_runtime.zimage_files())
                found["z_image_turbo_vae"] = root / "v.safetensors"
                # the Lite set lists its GGUFs in the shared folders and no official bf16 files
                del found["z_image_turbo_transformer"], found["z_image_turbo_text_encoder"]
                found["diffusion_models"] = root / "lite-dit.gguf"
                found["text_encoders"] = root / "lite-text.gguf"
                for key in ("diffusion_models", "text_encoders"):
                    found[key].touch()
                with patch.object(engine_runtime.model_manager, "tier", return_value="lite"):
                    files = engine_runtime.zimage_files()
                    self.assertEqual((files.transformer.name, files.text_encoder.name),
                                     ("lite-dit.gguf", "lite-text.gguf"))

    def test_background_sound_uses_ezaudios_unconditional_text(self):
        """The negative text is the empty prompt EzAudio was trained with; any other is outside its training."""
        import torch

        runtime = SimpleNamespace(audio_bg=Mock(return_value=(torch.zeros(1, 4), 48000)))
        with patch("engine.media.audio.write_wav") as written:
            engine_runtime.background_audio("rain", 2.0, 5, "out.wav", runtime=runtime)
        self.assertEqual(runtime.audio_bg.call_args.args[1:5], ("rain", "", 2.0, 5))
        self.assertEqual(written.call_args.args[0], "out.wav")

    def test_render_hands_the_job_to_the_engine_and_returns_png(self):
        picture = Image.new("RGB", (64, 48), (1, 2, 3))
        fake = SimpleNamespace(txt2img=Mock(return_value=picture))
        with patch.object(engine_runtime, "engine", return_value=fake), \
                patch.object(engine_runtime, "zimage_files", return_value="FILES"):
            data = engine_runtime.render("a cat", 64, 48, 7, 8)
            fake.txt2img.assert_called_once_with("FILES", "a cat", 64, 48, 7, 8, None)
            self.assertEqual(Image.open(io.BytesIO(data)).size, (64, 48))

    def test_the_engine_polls_the_apps_own_cancel_token(self):
        from engine.core.cancel import Cancelled
        with patch.object(engine_runtime, "_ENGINE", None):
            self.assertIs(engine_runtime.engine().cancel, cancel.TOKEN)
        self.assertIs(cancel.Cancelled, Cancelled)


class EngineGuardTests(unittest.TestCase):
    """After a failure the cache is emptied only once nothing holds the networks any more."""

    def run_guarded(self, failure):
        import weakref

        from engine.api import Engine

        refs, alive = [], []

        class Network:
            pass

        def run():
            network = Network()
            refs.append(weakref.ref(network))

            def forward(net):  # the error is raised deep inside, so the traceback holds this frame and `net`
                raise failure

            forward(network)

        engine = Engine("cpu")
        engine.free = lambda: alive.append(refs[0]() is not None)
        with self.assertRaises(type(failure)) as caught:
            engine._guarded(run)
        return caught.exception, alive

    def test_an_out_of_memory_error_frees_the_networks_before_the_cache_is_emptied(self):
        error, alive = self.run_guarded(RuntimeError("CUDA out of memory"))
        self.assertEqual(alive, [False])  # the cache was emptied once, with the network already gone
        self.assertIn("out of memory", str(error))
        self.assertTrue(any("engine traceback" in note and "forward" in note for note in error.__notes__))

    def test_a_cancel_passes_through_untouched_and_also_frees(self):
        error, alive = self.run_guarded(cancel.Cancelled())
        self.assertEqual(alive, [False])
        self.assertIsInstance(error, cancel.Cancelled)
        self.assertFalse(getattr(error, "__notes__", []))

    def test_free_gives_the_pinned_host_cache_back_only_when_there_is_a_gpu(self):
        import importlib.util

        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("gguf") is None:
            self.skipTest("torch or gguf is not installed")
        from engine.api import Engine

        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache") as empty, \
                patch("engine.core.streaming.release_host_cache") as release:
            Engine().free()
        empty.assert_called_once()
        release.assert_called_once()
        with patch("torch.cuda.is_available", return_value=False), patch("engine.core.streaming.release_host_cache") as release:
            Engine().free()
        release.assert_not_called()


class ImageHelperTests(unittest.TestCase):
    def test_text_to_image_on_the_engine_is_saved_and_errors_are_friendly(self):
        taken = []

        def finish(images, prefix, info):
            taken.append((images, prefix, info["seed"]))
            return image_gen.GenOutput(ok=True, seed=info["seed"])

        with patch.object(image_gen, "available_models", return_value=["z.safetensors"]), \
                patch.object(image_gen.llm, "blocked", return_value=False), \
                patch.object(image_gen, "_finish", side_effect=finish):
            with patch.object(image_gen.engine_runtime, "render", return_value=b"engine"):
                out = image_gen.txt2img("a cat", 64, 48, 8, 11)
            self.assertTrue(out.ok)
            self.assertEqual(taken, [([b"engine"], "img", 11)])
            with patch.object(image_gen.engine_runtime, "render", side_effect=RuntimeError("CUDA out of memory")):
                bad = image_gen.txt2img("a cat", 64, 48, 8, 11)
            self.assertFalse(bad.ok)
            self.assertIn("Not enough GPU memory", bad.error)
            with patch.object(image_gen.engine_runtime, "render", side_effect=cancel.Cancelled()):
                with self.assertRaises(cancel.Cancelled):  # a cancel is not an error message
                    image_gen.txt2img("a cat", 64, 48, 8, 11)

    def test_without_the_model_sets_image_model_nothing_is_rendered(self):
        with (patch.object(image_gen, "available_models", return_value=[]),
              patch.object(image_gen.llm, "blocked", return_value=False),
              patch.object(engine_runtime, "render") as render):
            out = image_gen.txt2img("a cat", 64, 48, 8, 11)
        self.assertEqual((out.ok, out.error), (False, "Select Z-Image Turbo for the active model set."))
        render.assert_not_called()

    def test_a_file_that_disappears_after_the_check_is_reported_by_name_not_as_an_attribute_error(self):
        with patch.object(engine_runtime, "zimage_files", return_value=None):
            with self.assertRaises(FileNotFoundError) as raised:
                engine_runtime.render("a cat", 64, 48, 1, 8)
            self.assertIn("Z-Image", str(raised.exception))

    def test_only_the_word_oom_makes_a_memory_error_not_names_that_contain_it(self):
        for message in ("CUDA out of memory. Tried to allocate 2 GiB", "torch.OutOfMemoryError: Allocation on device",
                        "OOM while decoding"):
            self.assertIn("Not enough GPU memory", image_gen._friendly_error(RuntimeError(message)), message)
        for message in ("FileNotFoundError: C:/models/zoom_lens.safetensors", "Bloom and Room are characters",
                        "no_oom_here"):
            self.assertTrue(image_gen._friendly_error(RuntimeError(message)).startswith("Generation failed: "), message)

    def test_a_character_is_composed_from_its_primary_photo_on_the_given_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            photo, out = Path(tmp) / "01.png", Path(tmp) / "out.png"
            photo.write_bytes(png_bytes((64, 64)))
            ch = {"slug": "ola", "name": "Ola", "look": "red hair", "gender": "woman", "photos": [str(photo)]}
            client = Mock()
            with (patch.object(characters.llm, "blocked", return_value=False),
                  patch.object(engine_runtime, "edit", return_value=png_bytes((100, 60))) as edit):
                ok, path = characters.render(client, ch, "sitting in a cafe", 100, 60, out, seed=3)
            self.assertTrue(ok, path)
            with Image.open(out) as picture:
                self.assertEqual(picture.size, (100, 60))
            images, prompt, width, height, seed = edit.call_args.args[:5]
            self.assertEqual((images, width, height, seed), ([str(photo)], 100, 60, 3))
            self.assertIn("exact adult identity in image 1", prompt)
            self.assertIs(edit.call_args.kwargs["runtime"], client)
            missing = {**ch, "photos": []}
            self.assertFalse(characters.render(client, missing, "a cafe", 100, 60, out, seed=3)[0])


if __name__ == "__main__":
    unittest.main()


class ReadAheadTests(unittest.TestCase):
    """The starting frame's model files are read into the file cache while a queued scene makes its sound."""

    def test_a_file_is_read_whole_at_background_priority(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.bin"
            path.write_bytes(b"x" * (readahead.READ_BYTES + 5))
            with patch.object(readahead, "_thread_mode") as mode:
                with readahead.background_io():
                    self.assertEqual(readahead.read_file(path), readahead.READ_BYTES + 5)
            self.assertEqual([call.args[0] for call in mode.call_args_list], [0x00010000, 0x00020000])

    def test_files_are_read_once_at_a_time_and_released_after(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "a.bin", Path(folder) / "b.bin"
            first.write_bytes(b"a" * (readahead.READ_BYTES + 3))
            second.write_bytes(b"b" * 5)
            read = []
            real = readahead.read_file
            with patch.object(readahead, "read_file", side_effect=lambda path: read.append(path) or real(path)), \
                    patch.dict(readahead._FINISHED, clear=True):
                thread = engine_runtime.read_ahead([first, second, first, None, Path(folder) / "missing.bin"])
                thread.join(10)
                self.assertEqual([Path(path) for path in read], [first, second])  # a missing file is skipped
                self.assertEqual(readahead._READING, set())
                self.assertTrue(readahead.recently_read(first))
                self.assertIsNone(engine_runtime.read_ahead([first]))  # read a moment ago: not read again

    def test_a_file_being_read_is_not_read_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            busy = Path(folder) / "busy.bin"
            busy.write_bytes(b"x")
            with patch.object(readahead, "_READING", {str(busy)}):
                self.assertIsNone(engine_runtime.read_ahead([busy]))

    def test_only_a_queued_job_reads_the_starting_frame_models_ahead(self):
        from app import styles

        files = SimpleNamespace(text_encoder=Path("text.safetensors"), transformer=Path("model.gguf"),
                                lightning=Path("lightning.safetensors"), vae=Path("vae.safetensors"))
        anime = Path("anime.safetensors")
        with patch.object(engine_runtime, "qwen_edit_files", return_value=files), \
                patch.object(engine_runtime, "style_loras", side_effect=lambda style, role:
                             [(anime, 1.0)] if style == styles.CARTOON_2D else []), \
                patch.object(engine_runtime, "read_ahead") as read_ahead:
            self.assertIsNone(engine_runtime.read_ahead_start_frame())
            read_ahead.assert_not_called()
            from app import queue_manager

            queue_manager._CONTEXT.active = True
            try:
                engine_runtime.read_ahead_start_frame()
                # a film's video style text (what a Timeline scene passes) or a style id
                engine_runtime.read_ahead_start_frame(styles.get(styles.CARTOON_2D).video)
            finally:
                queue_manager._CONTEXT.active = False
            self.assertEqual([call.args[0] for call in read_ahead.call_args_list],
                             [[files.text_encoder, files.transformer, files.lightning, files.vae],
                              [files.text_encoder, files.transformer, files.lightning, anime, files.vae]])
            with patch.object(engine_runtime, "qwen_edit_files", side_effect=FileNotFoundError("incomplete")):
                queue_manager._CONTEXT.active = True
                try:
                    self.assertIsNone(engine_runtime.read_ahead_start_frame())
                finally:
                    queue_manager._CONTEXT.active = False


class WarmImportTests(unittest.TestCase):
    def test_a_library_that_does_not_import_is_only_reported(self):
        with patch.object(engine_runtime, "WARM_IMPORTS", ("json", "no_such_library_for_contsy")):
            self.assertEqual(engine_runtime.warm_imports(), ["no_such_library_for_contsy"])


class WarmImportOrderTests(unittest.TestCase):
    def test_the_engine_waits_for_the_warm_imports(self):
        import threading

        finished = threading.Event()

        def slow_imports():
            finished.wait(5)
            return []

        warming = threading.Thread(target=slow_imports)
        with patch.object(engine_runtime, "_WARMING", warming), patch.object(engine_runtime, "_ENGINE", object()):
            warming.start()
            threading.Timer(0.2, finished.set).start()
            engine_runtime.engine()
            self.assertFalse(warming.is_alive())
