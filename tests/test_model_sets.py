# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Full and Lite model sets: catalog pins, file selection and downloads."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import config, download_models, engine_runtime, image_gen, llm, model_manager


def _install(root: Path, model_ids):
    """Create empty files for every catalog file of the given entries."""
    for model_id in model_ids:
        for spec in config.get_model(model_id)["files"]:
            target = root / spec["dest"] / Path(spec.get("save_as") or spec["filename"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")


class _BothSetsInstalled(unittest.TestCase):
    """Full and Lite files side by side, as after switching sets on one PC."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        _install(self.root, [m["id"] for m in config.CATALOG["models"]])
        for target in (patch.object(config, "destination_dir", lambda dest: self.root / dest),
                       patch.object(model_manager.prefs, "load", lambda: {"tier": self.tier})):
            target.start()
            self.addCleanup(target.stop)
        self.tier = "full"


def _assert_pinned(test, item, label):
    """A download: an https address or a Hugging Face revision, and its exact size and SHA-256."""
    if "url" in item:
        test.assertTrue(item["url"].startswith("https://"), label)
    else:
        test.assertRegex(item["revision"], r"^[0-9a-f]{40}$", label)
    test.assertGreater(item["size_bytes"], 0, label)
    test.assertRegex(item["sha256"], r"^[0-9a-f]{64}$", label)


class CatalogTests(unittest.TestCase):
    def test_every_lite_entry_is_a_pinned_variant_of_a_full_entry(self):
        ids = {m["id"] for m in config.CATALOG["models"]}
        lite = [m for m in config.CATALOG["models"] if m["id"].endswith("-lite")]
        self.assertEqual({m["id"] for m in lite}, {"z-image-turbo-lite", "wan22-a14b-t2v-lite",
                                                    "wan22-a14b-i2v-lite", "qwen-image-edit-2511-lite",
                                                    "prompt-llm-lite"})
        for model in lite:
            self.assertIn(model["id"][:-5], ids)

    def test_every_download_and_every_installed_file_is_pinned(self):
        """Each file the installer downloads has its address or revision, size and SHA-256; a file made by an
        install step also has its own size and SHA-256, which the step must reproduce."""
        for model in config.CATALOG["models"]:
            for spec in model["files"]:
                label = f"{model['id']}: {spec['filename']}"
                with self.subTest(file=label):
                    self.assertIn(spec["dest"], config.CATALOG["destinations"])
                    self.assertGreater(spec["size_bytes"], 0)
                    self.assertRegex(spec["sha256"], r"^[0-9a-f]{64}$")
                    for source in model_manager.sources(spec, model):
                        _assert_pinned(self, source, label)
                        self.assertTrue(source.get("url") or source.get("repo_id"), label)
                    for other in spec.get("equivalents", ()):
                        self.assertGreater(other["size_bytes"], 0, label)
                        self.assertRegex(other["sha256"], r"^[0-9a-f]{64}$", label)
                        self.assertIn(other.get("dest", spec["dest"]), config.CATALOG["destinations"], label)
                        self.assertTrue(set(other) <= {"dest", "filename", "size_bytes", "sha256", "_comment"},
                                        set(other))
                        # an equivalent is an earlier file, never another catalog file or one of its own sources
                        self.assertNotEqual(model_manager._equivalent_path(spec, other),
                                            model_manager._file_target(spec), label)

    def test_install_steps_are_known_and_read_their_own_downloads(self):
        steps = [(model, spec) for model in config.CATALOG["models"] for spec in model["files"] if "convert" in spec]
        self.assertEqual({spec["convert"]["tool"] for _, spec in steps}, set(model_manager.CONVERTERS))
        for model, spec in steps:
            convert = spec["convert"]
            name = spec.get("save_as") or Path(spec["filename"]).name
            with self.subTest(model=model["id"], file=name):
                found = model_manager.sources(spec, model)
                for source in found:
                    self.assertIn(source["dest"], config.CATALOG["destinations"])
                    self.assertTrue(set(source) <= {"filename", "url", "repo_id", "revision", "dest", "size_bytes",
                                                    "sha256"}, set(source))
                inputs = convert["input"] if isinstance(convert["input"], list) else [convert["input"]]
                self.assertTrue(set(inputs) <= {Path(source["filename"]).name for source in convert["sources"]})
                self.assertEqual([path.name for path in model_manager.step_inputs(spec, model)], inputs)
                self.assertIsInstance(convert["delete_sources_after"], bool)
                # the result never overwrites one of its own sources
                self.assertNotIn((spec["dest"], name), [(s["dest"], Path(s["filename"]).name) for s in found])
                self.assertTrue(set(convert) <= {"tool", "source_dest", "input", "member", "sources",
                                                 "delete_sources_after"}, set(convert))

    def test_a_step_source_that_another_entry_installs_has_that_files_pins_and_stays(self):
        """The FP8 experts are made from the Wan entries' GGUF experts and LoRAs (AniSora's from its own float32
        files and the image-to-video LoRAs): a step names such a file with that entry's repository, revision, size
        and SHA-256, and keeps it (it is that entry's file) even when it deletes its other sources."""
        files = {(spec["dest"], Path(spec.get("save_as") or spec["filename"]).name): (model, spec)
                 for model in config.CATALOG["models"] for spec in model["files"] if "convert" not in spec}
        shared = 0
        for model in config.CATALOG["models"]:
            for spec in model["files"]:
                if "convert" not in spec:
                    continue
                for source in model_manager.sources(spec, model):
                    owner = files.get((source["dest"], Path(source["filename"]).name))
                    if owner is None:
                        continue
                    shared += 1
                    other = model_manager.sources(owner[1], owner[0])[0]
                    with self.subTest(file=source["filename"]):
                        self.assertEqual({key: source.get(key) for key in ("repo_id", "filename", "revision",
                                                                          "size_bytes", "sha256")},
                                         {key: other.get(key) for key in ("repo_id", "filename", "revision",
                                                                         "size_bytes", "sha256")})
                        target = config.destination_dir(source["dest"]) / Path(source["filename"]).name
                        self.assertIn(target.resolve(), model_manager._catalog_targets())  # never deleted
        # the image-to-video high-noise expert's GGUF and LoRA, AniSora's two LoRAs, the text-to-video GGUFs and LoRAs
        self.assertEqual(shared, 8)

    def test_lite_entries_are_smaller_than_the_full_ones(self):
        for model in config.CATALOG["models"]:
            if model["id"].endswith("-lite"):
                full = config.get_model(model["id"][:-5])
                self.assertLess(model["size_gb"], full["size_gb"], model["id"])

    def test_size_gb_is_the_installed_size_of_the_files(self):
        for model in config.CATALOG["models"]:
            with self.subTest(model=model["id"]):
                self.assertAlmostEqual(model["size_gb"], sum(s["size_bytes"] for s in model["files"]) / 1e9,
                                       delta=0.01)


class TierTests(unittest.TestCase):
    def tearDown(self):
        model_manager._gpu_total_gb.cache_clear()

    def test_automatic_choice_follows_vram(self):
        with patch.object(model_manager.prefs, "load", return_value={"tier": ""}):
            for total, expected in ((8.0, "lite"), (12.0, "lite"), (16.0, "full"), (24.0, "full"), (0.0, "full")):
                model_manager._gpu_total_gb.cache_clear()
                with patch.object(model_manager.vram, "get_gpu_info",
                                  return_value=model_manager.vram.GpuInfo(available=total > 0, total_gb=total)):
                    self.assertEqual(model_manager.tier(), expected, total)

    def test_settings_override_the_automatic_choice(self):
        with patch.object(model_manager, "_gpu_total_gb", return_value=24.0), \
                patch.object(model_manager.prefs, "load", return_value={"tier": "lite"}):
            self.assertEqual(model_manager.tier(), "lite")
            self.assertEqual(model_manager.tiered("wan22-a14b-i2v"), "wan22-a14b-i2v-lite")
            self.assertEqual(model_manager.tiered("voxcpm2"), "voxcpm2")  # no Lite variant
            self.assertEqual(model_manager.tiered("wan22-a14b-i2v", "full"), "wan22-a14b-i2v")


class SelectionTests(_BothSetsInstalled):
    def test_video_uses_the_files_of_the_active_set(self):
        def wan_file(dest, *fragments):
            path = model_manager.model_path("wan22-a14b-i2v", dest, *fragments)
            return path.name if path else None

        self.assertIn("Q6_K", wan_file("diffusion_models", "highnoise"))
        # the official bf16 UMT5 shards, read through their index; no Lite GGUF in the Full set
        self.assertEqual(wan_file("wan_text_encoder", "index"), "model.safetensors.index.json")
        self.assertIsNone(wan_file("text_encoders"))
        full_lora, vae = wan_file("loras", "high_noise"), wan_file("wan_vae")
        self.tier = "lite"
        self.assertIn("HighNoise-Q4_K_M", wan_file("diffusion_models", "highnoise"))
        self.assertIn("LowNoise-Q4_K_M", wan_file("diffusion_models", "lownoise"))
        self.assertTrue(wan_file("text_encoders").endswith(".gguf"))
        self.assertIsNone(wan_file("wan_text_encoder", "index"))
        self.assertEqual(wan_file("loras", "high_noise"), full_lora)
        self.assertEqual(wan_file("wan_vae"), vae)  # one official VAE for both sets

    def test_image_model_and_encoder_follow_the_set(self):
        self.assertIn("bf16", image_gen.available_models()[0])
        files = engine_runtime.zimage_files()
        self.assertEqual((files.text_encoder.name, files.vae.parent.name),
                         ("qwen3-4b-text-encoder-bf16.safetensors", "z_image_turbo_vae"))
        self.tier = "lite"
        model = image_gen.available_models()[0]
        self.assertEqual(model, "z-image-turbo-Q8_0.gguf")
        self.assertEqual(engine_runtime.zimage_files().text_encoder.name, "Qwen3-4B-Q8_0.gguf")

    def test_reference_edit_uses_the_quantization_of_the_selected_set(self):
        files = engine_runtime.qwen_edit_files()
        self.assertEqual(files.transformer.name, "qwen-image-edit-2511-Q8_0.gguf")
        self.assertEqual((files.text_encoder.name, files.vae.parent.name),
                         ("qwen_2.5_vl_7b_edit_2511_fp8.safetensors", "qwen_edit_vae"))
        self.tier = "lite"
        self.assertEqual(engine_runtime.qwen_edit_files().transformer.name, "qwen-image-edit-2511-Q4_K_M.gguf")

    def test_assistant_uses_the_lighter_quantization(self):
        self.tier = "lite"
        self.assertEqual(llm._model_file(), "Qwen3.5-9B-Q4_K_M.gguf")
        self.assertEqual(llm._catalog_file(projector=True), "Qwen3.5-9B-mmproj-F16.gguf")

    def test_frame_interpolation_uses_the_authors_file_or_the_same_weights_kept_by_an_earlier_installation(self):
        rife = self.root / "rife"
        self.assertEqual(engine_runtime._rife_weights(), rife / "flownet_v4.9.pkl")
        (rife / "flownet_v4.9.pkl").unlink()
        self.assertRaises(FileNotFoundError, engine_runtime._rife_weights)
        spec = next(s for s in config.get_model("wan22-a14b-i2v")["files"] if s["dest"] == "rife")
        old = spec["equivalents"][0]
        (rife / old["filename"]).write_bytes(b"x" * old["size_bytes"])
        self.assertEqual(engine_runtime._rife_weights(), rife / "rife49.pth")

    def test_ensure_maps_to_the_set_unless_exact(self):
        self.tier = "lite"
        with patch.object(model_manager, "is_installed", return_value=True) as installed:
            self.assertIsNone(model_manager.ensure(["z-image-turbo", "voxcpm2"]))
            self.assertEqual([c.args[0] for c in installed.call_args_list], ["z-image-turbo-lite", "voxcpm2"])
            installed.reset_mock()
            model_manager.ensure(["z-image-turbo"], exact=True)
            self.assertEqual(installed.call_args.args[0], "z-image-turbo")


class SmallGpuTests(unittest.TestCase):
    def test_gpus_under_15_gb_count_as_small(self):
        for total, expected in ((8.0, True), (12.0, True), (16.0, False), (0.0, False)):
            with patch.dict(config.SETTINGS["engine"], vram_limit_gib=None),                     patch.object(model_manager, "_gpu_total_gb", return_value=total):
                self.assertEqual(engine_runtime.small_gpu(), expected, total)

    def test_a_configured_vram_limit_under_15_gib_counts_as_small(self):
        for limit, total, expected in ((12, 24.0, True), (16, 24.0, False), (None, 12.0, True)):
            with self.subTest(limit=limit, total=total),                     patch.dict(config.SETTINGS["engine"], vram_limit_gib=limit),                     patch.object(model_manager, "_gpu_total_gb", return_value=total):
                self.assertEqual(engine_runtime.small_gpu(), expected)


class DownloadBundleTests(unittest.TestCase):
    def test_lite_bundles_replace_entries_and_count_shared_files_once(self):
        full, lite = download_models.model_ids("all", "full"), download_models.model_ids("all", "lite")
        self.assertFalse([i for i in full if i.endswith("-lite")])
        self.assertIn("wan22-a14b-i2v-lite", lite)
        self.assertNotIn("wan22-a14b-i2v", lite)
        # the FP8 experts (image to video, text to video and the 2D style's AniSora) run only in the Full set, so
        # Lite does not install them; AniSora is installed from the model list, not with the video bundle
        from app import styles

        full_only = [i for i in full if config.get_model(i).get("full_only")]
        self.assertEqual(full_only, [engine_runtime.FAST_I2V_ID, engine_runtime.FAST_T2V_ID, styles.ANISORA_ID])
        for fast in (engine_runtime.FAST_I2V_ID, engine_runtime.FAST_T2V_ID):
            self.assertIn(fast, download_models.model_ids("video", "full"))
            self.assertNotIn(fast, download_models.model_ids("video", "lite"))
        self.assertNotIn(styles.ANISORA_ID, download_models.model_ids("video", "full"))
        self.assertEqual(len(full) - len(full_only), len(lite))
        self.assertLess(download_models.size_gb(lite), download_models.size_gb(full))
        both_wan = download_models.size_gb(["wan22-a14b-i2v", "wan22-a14b-t2v"])
        self.assertLess(both_wan, config.get_model("wan22-a14b-i2v")["size_gb"]
                        + config.get_model("wan22-a14b-t2v")["size_gb"])

    def test_a_converted_file_downloads_its_sources_and_keeps_only_the_result(self):
        model = config.get_model("z-image-turbo")
        merged = [spec for spec in model["files"] if "convert" in spec]
        self.assertEqual(len(merged), 2)
        sources = sum(s["size_bytes"] for spec in merged for s in spec["convert"]["sources"])
        plain = sum(spec["size_bytes"] for spec in model["files"] if "convert" not in spec)
        self.assertAlmostEqual(download_models.size_gb(["z-image-turbo"]), (sources + plain) / 1e9)
        self.assertAlmostEqual(download_models.disk_gb(["z-image-turbo"]), model["size_gb"], delta=0.01)

    def test_a_step_that_reads_files_of_another_entry_counts_them_once(self):
        i2v, fp8 = "wan22-a14b-i2v", engine_runtime.FAST_I2V_ID
        made = [spec for spec in config.get_model(fp8)["files"] if "convert" in spec]
        published = [spec for spec in config.get_model(fp8)["files"] if "convert" not in spec]
        self.assertEqual(len(made), 1)
        sources = sum(source["size_bytes"] for source in made[0]["convert"]["sources"])
        self.assertEqual(model_manager.download_bytes([i2v, fp8]),
                         model_manager.download_bytes([i2v]) + sum(spec["size_bytes"] for spec in published))
        self.assertEqual(model_manager.disk_bytes([i2v, fp8]),
                         model_manager.disk_bytes([i2v]) + sum(spec["size_bytes"] for spec in made + published))
        # alone, the entry downloads and keeps the GGUF expert and LoRA it is made from
        self.assertEqual(model_manager.download_bytes([fp8]), sources + sum(spec["size_bytes"] for spec in published))
        self.assertEqual(model_manager.disk_bytes([fp8]), sources + sum(s["size_bytes"] for s in made + published))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(config, "destination_dir", lambda dest: Path(folder) / dest):
            download, need = model_manager.missing_bytes([i2v, fp8])
            archive = model_manager.download_bytes([i2v]) - sum(
                spec["size_bytes"] for spec in config.get_model(i2v)["files"] if "convert" not in spec)
            self.assertEqual(download, model_manager.download_bytes([i2v, fp8]))
            self.assertEqual(need, model_manager.disk_bytes([i2v, fp8]) + archive)  # RIFE's archive, until extracted

    def test_disk_check_counts_missing_sources_and_results_at_the_peak(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(config, "destination_dir", lambda dest: Path(folder) / dest):
            download, need = model_manager.missing_bytes(["z-image-turbo"])
            model = config.get_model("z-image-turbo")
            largest = max(sum(s["size_bytes"] for s in spec["convert"]["sources"])
                          for spec in model["files"] if "convert" in spec)
            self.assertEqual(download, model_manager.download_bytes(["z-image-turbo"]))
            self.assertEqual(need, model_manager.disk_bytes(["z-image-turbo"]) + largest)
            _install(Path(folder), ["z-image-turbo"])
            # every result on disk with its size (the sizes are not written out: they are GBs)
            with patch.object(model_manager, "_present", lambda path, size: path.is_file()):
                self.assertEqual(model_manager.missing_bytes(["z-image-turbo"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
