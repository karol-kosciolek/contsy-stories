# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Qwen-Image-Edit in the engine on CPU: pinned reference values of seeded tiny networks, the prompt layout, the
composition pipeline, the token ids of real edit prompts, the install step that turns the official bfloat16 encoder
shards into the scaled-fp8 file, and the layout of the official files on disk (when they are there)."""
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app import config as app_config
from tests import seeded
from tests.seeded import FIXTURES

HAVE = all(importlib.util.find_spec(name) for name in ("torch", "diffusers", "transformers", "safetensors", "gguf"))
TOKENIZER_DIR = app_config.destination_dir("tokenizer_qwen_edit")
OFFICIAL_REPO = "Qwen/Qwen-Image-Edit-2511"
# Tiny Qwen2.5-VL with the structure of the real encoder (transformers config).
TINY_VL = {"text_config": {"vocab_size": 151680, "hidden_size": 32, "intermediate_size": 48,
                           "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                           "max_position_embeddings": 128,
                           "rope_parameters": {"rope_type": "default", "mrope_section": [2, 3, 3]}},
           "vision_config": {"depth": 2, "hidden_size": 32, "intermediate_size": 48, "num_heads": 4,
                             "out_hidden_size": 32, "fullatt_block_indexes": [1]}}


def fixture(kind):
    return json.loads((FIXTURES / f"engine_qwen_{kind}.json").read_text(encoding="utf-8"))


def catalog_file(dest: str):
    """The installed path of the Qwen-Image-Edit catalog file in `dest` that comes from the official repository (or is
    made from its files), None while the catalog lists none."""
    model = app_config.get_model("qwen-image-edit-2511") or {}
    for spec in model.get("files", []):
        official = spec.get("repo_id") == OFFICIAL_REPO or "convert" in spec
        if spec["dest"] == dest and official:
            return app_config.destination_dir(dest) / (spec.get("save_as") or Path(spec["filename"]).name)
    return None


def safetensors_header(path: Path) -> dict:
    with open(path, "rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size))


@unittest.skipUnless(HAVE, "requires torch, diffusers and safetensors")
class QwenEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from safetensors.torch import save_file

        from engine.models.qwen_edit import transformer, vae

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "qwen.safetensors"
            cls.dit = fixture("dit")
            save_file(seeded.seeded_state_dict(cls.dit["shapes"]), str(checkpoint))
            cls.transformer = transformer.load_transformer(
                checkpoint, dtype=torch.float32, config=dict(cls.dit["config"], zero_cond_t=True))
            cls.vae_data = fixture("vae")
            # The fixture seeds its tiny weights by their original Wan names; diffusers' renamer gives them the names
            # of the official file, which the loader reads as they are.
            from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
            save_file(convert_wan_vae_to_diffusers(seeded.seeded_state_dict(cls.vae_data["shapes"])), str(checkpoint))
            cfg = json.loads(vae.CONFIG_FILE.read_text(encoding="utf-8"))
            cls.vae = vae.load_vae(checkpoint, dtype=torch.float32,
                                   config=dict(cfg, base_dim=cls.vae_data["base_dim"]))

    def test_transformer_keeps_its_pinned_outputs_with_zero_one_and_two_references(self):
        import torch
        from engine.models.qwen_edit import transformer

        target = seeded.seeded_inputs(self.dit["target"], 61)
        context = seeded.seeded_inputs(self.dit["context"], 62)
        refs = [seeded.seeded_inputs(shape, 63 + index) for index, shape in enumerate(self.dit["refs"])]
        with torch.inference_mode():
            for case in self.dit["cases"]:
                with self.subTest(references=case["count"]):
                    result = transformer.predict_velocity(self.transformer, target, case["sigma"], context,
                                                          refs[:case["count"]])
                    expected = torch.tensor(case["output"]).reshape(target.shape)
                    torch.testing.assert_close(result, expected, atol=4e-6, rtol=1e-5)

    def test_vae_keeps_its_pinned_reference_latent_and_output_pixels(self):
        import torch
        from engine.models.qwen_edit import vae

        data = self.vae_data
        picture = seeded.seeded_inputs(data["picture_shape"], 71).sigmoid()
        latent = vae.encode(self.vae, picture)
        expected = torch.tensor(data["latent"]).reshape(data["latent_shape"])
        torch.testing.assert_close(latent, expected, atol=4e-6, rtol=1e-5)
        pixels = vae.decode(self.vae, expected)
        torch.testing.assert_close(pixels, torch.tensor(data["decoded"]).reshape(picture.shape), atol=4e-6, rtol=1e-5)

    def test_pipeline_denoises_target_not_reference_and_reports_progress(self):
        import torch
        from PIL import Image
        from engine.pipelines import edit

        refs = [Image.new("RGB", (24, 16), "red"), Image.new("RGB", (16, 24), "blue")]
        encoder = Mock(return_value=seeded.seeded_inputs(self.dit["context"], 62))
        models = edit.EditModels(lambda *_: (None, None), lambda *_: self.transformer, lambda *_: self.vae, encoder)
        report = Mock()
        with patch.object(edit, "REFERENCE_AREA", 24 * 16):
            first = edit.edit(models, refs, "move the person", 40, 24, 9,
                              device="cpu", dtype=torch.float32, progress_cb=report)
            second = edit.edit(models, refs[::-1], "move the person", 40, 24, 9,
                               device="cpu", dtype=torch.float32)
        self.assertEqual(first.size, (40, 24))
        self.assertNotEqual(first.tobytes(), second.tobytes())
        self.assertEqual(encoder.call_args_list[0].args[2:], ("move the person", refs))
        self.assertEqual(report.call_args_list[-1].args[0], 1.0)

    def test_cancel_from_progress_stops_before_second_denoise_step(self):
        import torch
        from PIL import Image
        from engine.core.cancel import CancelToken, Cancelled
        from engine.pipelines import edit

        token = CancelToken()
        models = edit.EditModels(lambda *_: (None, None), lambda *_: self.transformer, lambda *_: self.vae,
                                 lambda *_: seeded.seeded_inputs(self.dit["context"], 62))

        def report(_fraction, message):
            if message == "Step 1/4":
                token.cancel()

        with (patch.object(edit, "REFERENCE_AREA", 16 * 16),
              patch.object(edit.transformer, "predict_denoised", wraps=edit.transformer.predict_denoised) as denoise,
              self.assertRaises(Cancelled)):
            edit.edit(models, [Image.new("RGB", (16, 16))], "move the person", 32, 24, 9,
                      device="cpu", dtype=torch.float32, progress_cb=report, cancel=token)
        self.assertEqual(denoise.call_count, 1)


@unittest.skipUnless(HAVE, "requires torch, diffusers and safetensors")
@unittest.skipUnless(HAVE, "torch, diffusers, transformers, safetensors or gguf missing")
class QwenEditPreloadTests(unittest.TestCase):
    """The streamed transformer is read on a worker thread while the encoders run (edit._Preload)."""

    def test_pages_of_quantized_weights_are_touched_and_others_left(self):
        import torch
        from engine.core.quant import QLinear
        from engine.formats.ggml import QTensor
        from engine.pipelines import edit

        weight = QTensor("w", 8, (4, 32), torch.zeros(3 * edit.PAGE_BYTES, dtype=torch.uint8))
        network = torch.nn.Sequential(QLinear(weight, None), torch.nn.Linear(4, 4))
        self.assertEqual(edit.touch_pages(network), 3 * edit.PAGE_BYTES)
        self.assertEqual(edit.touch_pages(torch.nn.Linear(2, 2)), 0)

    def test_take_waits_for_the_network_and_passes_on_an_error(self):
        import threading
        import torch
        from engine.pipelines import edit

        release = threading.Event()
        built = torch.nn.Linear(2, 2)

        def load():
            release.wait(5)
            return built

        preload = edit._Preload(load)
        release.set()
        self.assertIs(preload.take(), built)

        def broken():
            raise OSError("unreadable file")

        with self.assertRaisesRegex(OSError, "unreadable"):
            edit._Preload(broken).take()

    def test_the_file_is_read_ahead_before_the_network_loads(self):
        """The worker reads the file before it builds the network, unless the scene's own read-ahead finished it a
        moment ago (then it is not copied again) or is reading it now (then it waits for that reader)."""
        import tempfile
        import threading

        import torch
        from engine.core import readahead
        from engine.pipelines import edit

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.gguf"
            path.write_bytes(b"x" * 16)
            order = []
            with patch.object(readahead, "read_file", side_effect=lambda file: order.append(("read", Path(file)))), \
                    patch.dict(readahead._FINISHED, clear=True):
                network = edit._Preload(lambda: order.append("load") or torch.nn.Linear(2, 2), path).take()
                self.assertIsInstance(network, torch.nn.Linear)
                self.assertEqual(order, [("read", path), "load"])
                order.clear()
                edit._Preload(lambda: order.append("load") or torch.nn.Linear(2, 2), path).take()
                self.assertEqual(order, ["load"])  # read ahead a moment ago: not read again
            readahead._FINISHED.pop(str(path), None)
            release, order = threading.Event(), []
            with patch.object(readahead, "read_file", side_effect=lambda file: release.wait(5) and order.append("read")):
                reader = readahead.ahead([path])  # the scene's own reader is still at it
                preload = edit._Preload(lambda: order.append("load") or torch.nn.Linear(2, 2), path)
                release.set()
                preload.take()
                reader.join(5)
            self.assertEqual(order, ["read", "load"])  # waited for that reader instead of reading again

    def test_the_network_classes_are_imported_before_the_worker_starts(self):
        """Two threads importing diffusers' lazy modules at once failed with "cannot import name
        'AutoencoderKLQwenImage'" in the first composition of a fresh process (2026-11-15)."""
        from PIL import Image
        from engine.pipelines import edit

        order = []

        class Stop(Exception):
            pass

        def preload(*_args):
            order.append("preload")
            raise Stop

        models = edit.EditModels(load_text=None, load_transformer=None, load_vae=None)
        with (patch.object(edit, "import_networks", side_effect=lambda: order.append("import")),
              patch.object(edit, "_Preload", side_effect=preload)):
            with self.assertRaises(Stop):
                edit.edit(models, [Image.new("RGB", (16, 16))], "a cup", 16, 16, 1, stream={"pin": False},
                          device="cuda")
        self.assertEqual(order, ["import", "preload"])
        edit.import_networks()  # the real classes exist in the pinned diffusers

    def test_a_dropped_preload_keeps_nothing(self):
        import threading
        import torch
        from engine.pipelines import edit

        release = threading.Event()
        preload = edit._Preload(lambda: release.wait(5) and torch.nn.Linear(2, 2))
        preload.drop()
        release.set()
        preload._thread.join(5)
        self.assertIsNone(preload._network)


class QwenTextTests(unittest.TestCase):
    def test_image_preparation_keeps_its_pinned_patch_rows(self):
        import numpy as np
        import torch
        from PIL import Image
        from engine.models.qwen_edit import text

        data = fixture("text")
        pixels = (np.arange(32 * 48 * 3) % 256).astype(np.uint8).reshape(32, 48, 3)
        with patch.object(text, "AREA", 56 * 56):
            prepared = text.prepare_images([Image.fromarray(pixels)])
        self.assertEqual(list(prepared.pixel_values.shape), data["shape"])
        self.assertEqual(prepared.grid_thw.tolist(), data["grid"])
        torch.testing.assert_close(prepared.pixel_values.flatten()[::53], torch.tensor(data["sample"]))

    def test_rope_positions_follow_the_multimodal_rule(self):
        """Text counts 0, 1, 2 ...; a picture with a merged grid of gh x gw gets (P, P + row, P + column); the text after
        it continues at P + max(gh, gw). Masked tokens are left out of the count (their own values are not used)."""
        import torch
        from engine.models.qwen_edit import text

        config = {"text_config": {"vocab_size": 151680, "hidden_size": 32, "intermediate_size": 48,
                                  "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
                                  "max_position_embeddings": 128,
                                  "rope_parameters": {"rope_type": "default", "mrope_section": [2, 3, 3]}},
                  "vision_config": {"depth": 1, "hidden_size": 32, "intermediate_size": 48, "num_heads": 4,
                                    "out_hidden_size": 32, "fullatt_block_indexes": [0]}}
        model = text._build(config).model
        ids = [5, 6, 7, text.IMAGE_PAD, 8, 9, 10, text.IMAGE_PAD, 11, 12, 13, 14, 15]
        grids = torch.tensor([[1, 4, 6], [1, 4, 2]])  # merged grids 2 x 3 (6 embeddings) and 2 x 1 (2 embeddings)
        mask = torch.ones(1, 19, dtype=torch.long)
        mask[0, -2:] = 0
        got = text._rope_positions(model, ids, [3, 7], [6, 2], grids, mask)
        expected = [[0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7, 8, 9, 9, 11, 12, 13],
                    [0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7, 8, 9, 10, 11, 12, 13],
                    [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 9, 11, 12, 13]]
        self.assertEqual(got[:, :17].tolist(), expected)

    def test_the_picture_limit_is_defined_once(self):
        from app import refs
        from engine.models.qwen_edit import text
        from engine.pipelines import edit

        self.assertEqual((edit.MAX_IMAGES, refs.MAX_IMAGES), (text.MAX_IMAGES, text.MAX_IMAGES))

    def test_the_template_and_the_end_marker(self):
        from engine.models.qwen_edit import text

        ids = [text.IM_START, 1, 2, text.IM_START, text.USER, text.NEWLINE, 9, text.PAD, 4]
        self.assertEqual(text._template_end(ids), 6)
        self.assertEqual(text._token_mask(ids, []), [1, 1, 1, 1, 1, 1, 1, 0, 0])
        self.assertEqual(text._token_mask([text.PAD, 3, text.PAD, 5], []), [0, 1, 0, 0])  # a left pad, then the end
        with self.assertRaises(ValueError):
            text._template_end([text.IM_START, 1, 2])
        for count in (0, 4):
            with self.assertRaises(ValueError):
                text._compose("x", count)
        self.assertEqual(text._compose("x", 1), text.TEMPLATE_EDIT.format(text.PICTURE.format(1) + "x"))

    def test_local_text_checkpoint_runs_with_two_vision_inputs(self):
        import torch
        from PIL import Image
        from safetensors.torch import save_file
        from engine.models.qwen_edit import text

        config = TINY_VL
        shapes = {key: tensor.shape for key, tensor in text._build(config).state_dict().items()
                  if key != "lm_head.weight"}
        ids = [text.IM_START, 13, text.IM_START, text.USER, text.NEWLINE, 10, text.IMAGE_PAD,
               11, text.IMAGE_PAD, 12]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen_vl.safetensors"
            save_file(seeded.seeded_state_dict(shapes), str(path))
            encoder = text.load_text_encoder(path, "cpu", torch.float32, config)
        pictures = [Image.new("RGB", (56, 56), "red"), Image.new("RGB", (84, 42), "blue")]
        with patch.object(text, "token_ids", return_value=ids), patch.object(text, "AREA", 56 * 56):
            result = text.encode_prompt(None, encoder, "change the setting", pictures)
            swapped = text.encode_prompt(None, encoder, "change the setting", pictures[::-1])
        self.assertEqual(result.shape[-1], 32)
        self.assertEqual(result.shape, swapped.shape)
        self.assertGreater(float((result - swapped).abs().max()), 0.001)
        self.assertTrue(torch.isfinite(result).all())


@unittest.skipUnless(HAVE and (TOKENIZER_DIR / "vocab.json").is_file(),
                     "transformers or the tokenizer files are missing")
class QwenTokenizerTests(unittest.TestCase):
    """The token ids of real edit prompts are pinned (tests/fixtures/engine_qwen_tokens.json, made with the official
    processor/ files and equal to the official tokenizer.json); they must not change with the tokenizer files."""

    @classmethod
    def setUpClass(cls):
        from engine.models.qwen_edit import text

        cls.tokenizer = text.load_tokenizer(TOKENIZER_DIR)
        cls.cases = fixture("tokens")["cases"]

    def test_edit_prompts_keep_their_token_ids(self):
        from engine.models.qwen_edit import text

        plain = [case for case in self.cases if not case["literal"]]
        self.assertGreaterEqual(len(plain), 10)
        self.assertEqual({case["images"] for case in plain}, {1, 2, 3})
        for case in plain:
            with self.subTest(prompt=case["prompt"][:40], images=case["images"]):
                self.assertEqual(text.token_ids(self.tokenizer, case["prompt"], case["images"]), case["ids"])

    def test_pictures_are_announced_with_the_placeholder_tokens(self):
        from engine.models.qwen_edit import text

        names = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
        self.assertEqual(self.tokenizer.convert_tokens_to_ids(names),
                         [text.PAD, text.IM_START, 151645, 151652, text.IMAGE_PAD, 151653])
        announcement = len(self.tokenizer("Picture 1: ")["input_ids"])
        for count in (1, 2, 3):
            with self.subTest(images=count):
                ids = text.token_ids(self.tokenizer, "Put the hat from image 2 on the person in image 1.", count)
                starts = [i for i, token in enumerate(ids) if token == 151652]
                self.assertEqual([ids[i:i + 3] for i in starts], [[151652, text.IMAGE_PAD, 151653]] * count)
                self.assertEqual(ids.count(text.IMAGE_PAD), count)
                # diffusers' QwenImageEditPlusPipeline drops the first 64 tokens (prompt_template_encode_start_idx)
                self.assertEqual(text._template_end(ids), 64)
                self.assertEqual(starts[0], 64 + announcement)

    def test_think_and_tool_response_text_stays_text(self):
        """The official files add 22 tokens (151643 to 151664); `<think>` and `<tool_response>` are not among them."""
        from engine.models.qwen_edit import text

        added = self.tokenizer.get_added_vocab()
        if "<think>" in added:
            self.skipTest("the installed tokenizer_config.json is not the official processor/ file yet")
        self.assertEqual(sorted(added.values()), list(range(151643, 151665)))
        literal = [case for case in self.cases if case["literal"]]
        self.assertGreaterEqual(len(literal), 5)
        for case in literal:
            with self.subTest(prompt=case["prompt"][:40]):
                ids = text.token_ids(self.tokenizer, case["prompt"], case["images"])
                self.assertEqual(ids, case["ids"])
                self.assertFalse(any(151665 <= token <= 151668 for token in ids))


def _official_names(state: dict) -> dict:
    """transformers-5 names of a Qwen2.5-VL state dict -> the transformers-4 names of the official shards."""
    renamed = {}
    for key, value in state.items():
        if key.startswith("model.language_model."):
            key = "model." + key[len("model.language_model."):]
        elif key.startswith("model.visual."):
            key = key[len("model."):]
        renamed[key] = value
    return renamed


@unittest.skipUnless(HAVE, "requires torch, transformers and safetensors")
class QwenEncoderConversionTests(unittest.TestCase):
    """convert.quantize_text_encoder on two tiny bfloat16 shards with the names of the official ones."""

    @classmethod
    def setUpClass(cls):
        import torch
        from safetensors.torch import save_file
        from engine.models.qwen_edit import convert, text

        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        shapes = {key: tuple(tensor.shape) for key, tensor in text._build(TINY_VL).state_dict().items()}
        cls.source = {key: value.to(torch.bfloat16)
                      for key, value in _official_names(seeded.seeded_state_dict(shapes)).items()}
        keys = sorted(cls.source)
        cls.folder = root / "text_encoder"
        cls.folder.mkdir()
        cls.shards = [cls.folder / "model-00001-of-00002.safetensors", cls.folder / "model-00002-of-00002.safetensors"]
        save_file({key: cls.source[key] for key in keys[0::2]}, str(cls.shards[0]), metadata={"format": "pt"})
        save_file({key: cls.source[key] for key in keys[1::2]}, str(cls.shards[1]), metadata={"format": "pt"})
        cls.weight_map = {key: cls.shards[position % 2].name for position, key in enumerate(keys)}
        (cls.folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": cls.weight_map}),
                                                                 encoding="utf-8")
        cls.output = root / "encoder_fp8.safetensors"
        cls.sha256 = convert.quantize_text_encoder(cls.folder, cls.output)  # as the installer calls it

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_linear_matrices_become_fp8_with_a_scale_and_the_rest_is_copied(self):
        import torch
        from safetensors import safe_open
        from engine.models.qwen_edit import convert

        codes = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        values = codes[torch.isfinite(codes)].unique()  # every float8 number, ascending
        with safe_open(str(self.output), "pt") as handle:
            keys = set(handle.keys())
            scaled = {key[:-len("scale_weight")] + "weight" for key in keys if key.endswith(".scale_weight")}
            self.assertEqual(len(scaled), 2 * 7 + 2 * 5 + 2)  # decoder layers, vision blocks, merger
            self.assertTrue(all(convert.QUANTIZED.match(key) for key in scaled))
            self.assertEqual(keys, (set(self.source) - {"lm_head.weight"})
                             | {key[:-len("weight")] + "scale_weight" for key in scaled})
            for key in sorted(set(self.source) - scaled - {"lm_head.weight"}):
                copied = handle.get_tensor(key)
                self.assertEqual(copied.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(copied, self.source[key]), key)
            for key in sorted(scaled):
                with self.subTest(matrix=key):
                    weight = self.source[key].float()
                    stored = handle.get_tensor(key)
                    scale = handle.get_tensor(key[:-len("weight")] + "scale_weight")
                    self.assertEqual((stored.dtype, scale.dtype, tuple(scale.shape)),
                                     (torch.float8_e4m3fn, torch.float32, ()))
                    self.assertEqual(float(scale), float(weight.abs().max() / 416.0))
                    self.assertEqual(float(stored.float().abs().max()), 416.0)
                    # every stored value is a float8 number nearest to weight / scale
                    target = (weight / scale).reshape(-1)
                    index = torch.searchsorted(values, target).clamp(1, len(values) - 1)
                    best = torch.minimum((target - values[index - 1]).abs(), (values[index] - target).abs())
                    self.assertTrue(bool(((stored.float().reshape(-1) - target).abs() <= best).all()))

    def test_file_layout_is_fixed_and_the_hash_does_not_depend_on_the_shard_order(self):
        from engine.models.qwen_edit import convert

        header = safetensors_header(self.output)
        self.assertEqual(header.pop("__metadata__"), convert.METADATA)
        size = struct.unpack("<Q", self.output.read_bytes()[:8])[0]
        self.assertEqual(size % 8, 0)
        widths = {"F32": 4, "BF16": 2, "F8_E4M3": 1}
        order = sorted(header, key=lambda name: header[name]["data_offsets"][0])
        self.assertEqual(order, sorted(header, key=lambda name: (-widths[header[name]["dtype"]], name)))
        end = 0
        for name in order:
            self.assertEqual(header[name]["data_offsets"][0], end)
            end = header[name]["data_offsets"][1]
        self.assertEqual(self.output.stat().st_size, 8 + size + end)
        self.assertEqual(hashlib.sha256(self.output.read_bytes()).hexdigest(), self.sha256)
        again = Path(self.directory.name) / "again.safetensors"
        size = self.output.stat().st_size
        self.assertEqual(convert.quantize_text_encoder(self.shards[::-1], again, expected_size=size,
                                                       expected_sha256=self.sha256), self.sha256)
        self.assertEqual(again.read_bytes(), self.output.read_bytes())
        index = self.folder / "model.safetensors.index.json"
        self.assertEqual(convert.quantize_text_encoder(str(index), again, expected_sha256=self.sha256), self.sha256)

    def test_a_wrong_hash_keeps_the_partial_file_and_publishes_nothing(self):
        from engine.models.qwen_edit import convert

        target = Path(self.directory.name) / "wrong.safetensors"
        with self.assertRaises(convert.ConversionError):
            convert.quantize_text_encoder(self.shards, target, expected_sha256="0" * 64)
        self.assertFalse(target.exists())
        self.assertTrue(target.with_name(target.name + ".part").is_file())
        with self.assertRaises(convert.ConversionError):
            convert.quantize_text_encoder(self.shards, target, expected_size=123)
        with self.assertRaises(convert.ConversionError):
            convert.quantize_text_encoder(self.shards, target, cancel_flag=lambda: True)
        self.assertFalse(target.exists())

    def test_refuses_shards_that_are_not_the_official_bfloat16_ones(self):
        import torch
        from safetensors.torch import save_file
        from engine.models.qwen_edit import convert

        root = Path(self.directory.name)
        float32 = root / "float32.safetensors"
        save_file({"model.layers.0.mlp.up_proj.weight": torch.ones(4, 4)}, str(float32))
        with self.assertRaises(convert.ConversionError):
            convert.quantize_text_encoder([float32], root / "x.safetensors")
        with self.assertRaises(convert.ConversionError):  # the same tensor in two shards
            convert.quantize_text_encoder([self.shards[0], self.shards[0]], root / "x.safetensors")
        for name, weight_map in (("outside", {key: "../" + shard for key, shard in self.weight_map.items()}),
                                 ("swapped", {key: self.shards[0].name if shard == self.shards[1].name
                                              else self.shards[1].name for key, shard in self.weight_map.items()})):
            with self.subTest(index=name), self.assertRaises(convert.ConversionError):
                index = self.folder / f"{name}.safetensors.index.json"
                index.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
                try:
                    convert.quantize_text_encoder(index, root / "x.safetensors")
                finally:
                    index.unlink()
        with self.assertRaises(convert.ConversionError):
            convert.quantize(torch.tensor([[1.0, float("nan")]]))
        matrix, scale = convert.quantize(torch.zeros(2, 2))
        self.assertEqual((float(scale), float(matrix.float().abs().max())), (1.0, 0.0))

    def test_the_previous_catalog_layout_loads_the_same_weights(self):
        """Transitional (remove after the Studio restarts on the new catalog): the previous encoder file adds a
        `scaled_fp8` marker, an `X.scale_input` per scaled matrix and `lm_head.weight`; none of them is loaded."""
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from engine.models.qwen_edit import text

        with safe_open(str(self.output), "pt") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        previous = dict(tensors, scaled_fp8=torch.zeros(0, dtype=torch.float8_e4m3fn),
                        **{"lm_head.weight": self.source["lm_head.weight"]},
                        **{key[:-len("scale_weight")] + "scale_input": torch.ones(())
                           for key in tensors if key.endswith(".scale_weight")})
        path = Path(self.directory.name) / "previous_layout.safetensors"
        save_file(previous, str(path))
        header, expected = text._read_header(path), text._read_header(self.output)
        self.assertEqual(header.scaled, expected.scaled)
        self.assertEqual({key: value for key, value in header.tensors.items() if key != "lm_head.weight"},
                         expected.tensors)
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                old = text.load_text_encoder(path, "cpu", dtype, TINY_VL).state_dict()
                new = text.load_text_encoder(self.output, "cpu", dtype, TINY_VL).state_dict()
                self.assertEqual(old.keys(), new.keys())
                for key, value in new.items():
                    self.assertTrue(torch.equal(old[key], value), key)

    def test_converted_file_loads_and_stays_close_to_the_bfloat16_encoder(self):
        import torch
        from PIL import Image
        from safetensors.torch import save_file
        from engine.models.qwen_edit import text

        reference_file = Path(self.directory.name) / "bf16.safetensors"
        save_file(self.source, str(reference_file))
        kept = text.load_text_encoder(self.output, "cpu", torch.bfloat16, TINY_VL)  # fp8 stays fp8
        expanded = text.load_text_encoder(self.output, "cpu", torch.float32, TINY_VL)
        reference = text.load_text_encoder(reference_file, "cpu", torch.float32, TINY_VL)
        fp8_layers = [name for name, module in kept.named_modules() if isinstance(module, text._Fp8Linear)]
        self.assertEqual(len(fp8_layers), 2 * 7 + 2 * 5 + 2)
        ids = [text.IM_START, 13, text.IM_START, text.USER, text.NEWLINE, 10, text.IMAGE_PAD, 11, 12]
        pictures = [Image.new("RGB", (56, 84), "red")]
        with patch.object(text, "token_ids", return_value=ids), patch.object(text, "AREA", 56 * 56):
            results = [text.encode_prompt(None, model, "x", pictures) for model in (kept, expanded, reference)]
        torch.testing.assert_close(results[0], results[1], atol=1e-5, rtol=1e-5)
        cosine = torch.nn.functional.cosine_similarity(results[0].flatten(), results[2].flatten(), dim=0)
        self.assertGreater(float(cosine), 0.99)  # a random tiny network: wrong bytes or scales give far less


@unittest.skipUnless(HAVE, "requires torch, diffusers and safetensors")
class QwenOfficialFileTests(unittest.TestCase):
    """The official VAE and the converted encoder on disk, from their headers only (skipped while they are missing)."""

    def test_vae_in_the_original_wan_layout_gives_the_same_weights(self):
        """Transitional (remove after the Studio restarts on the new catalog): the previous catalog's VAE holds the
        official tensors under their original Wan names."""
        import torch
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
        from safetensors.torch import save_file
        from engine.models.qwen_edit import vae

        data = fixture("vae")
        original = seeded.seeded_state_dict(data["shapes"])
        config = dict(json.loads(vae.CONFIG_FILE.read_text(encoding="utf-8")), base_dim=data["base_dim"])
        with tempfile.TemporaryDirectory() as directory:
            paths = Path(directory) / "wan_layout.safetensors", Path(directory) / "official_layout.safetensors"
            save_file(original, str(paths[0]))
            save_file(convert_wan_vae_to_diffusers(dict(original)), str(paths[1]))
            old, new = (vae.load_vae(path, dtype=torch.float32, config=config).state_dict() for path in paths)
        self.assertEqual(old.keys(), new.keys())
        for key, value in new.items():
            self.assertTrue(torch.equal(old[key], value), key)

    def test_official_vae_has_the_names_and_shapes_of_the_network(self):
        import diffusers
        import torch
        from safetensors import safe_open
        from engine.models.qwen_edit import vae

        path = catalog_file("qwen_edit_vae")
        if path is None or not path.is_file():
            self.skipTest("the official Qwen-Image VAE is not installed")
        with torch.device("meta"):
            network = diffusers.AutoencoderKLQwenImage.from_config(vae._config(None))
        expected = {key: tuple(value.shape) for key, value in network.state_dict().items()}
        with safe_open(str(path), "pt") as handle:
            disk = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
            self.assertEqual({handle.get_slice(key).get_dtype() for key in handle.keys()}, {"BF16"})
        self.assertEqual(disk, expected)

    def test_converted_encoder_has_the_names_and_shapes_of_the_network(self):
        import torch
        from engine.models.qwen_edit import text

        path = catalog_file("qwen_edit_text_encoder")
        if path is None or not path.is_file():
            self.skipTest("the converted Qwen2.5-VL encoder is not installed")
        header = text._read_header(path)
        self.assertEqual(len(header.scaled), 28 * 7 + 32 * 5 + 2)
        self.assertNotIn("lm_head.weight", header.tensors)
        expected = {key: tuple(value.shape) for key, value in text._build().state_dict().items()
                    if key != "lm_head.weight"}
        self.assertEqual({key: shape for key, (shape, _) in header.tensors.items()}, expected)
        for key, (_, dtype) in header.tensors.items():
            module = key[:key.rfind(".")]
            self.assertIs(dtype, torch.float8_e4m3fn if module in header.scaled and key.endswith(".weight")
                          else torch.bfloat16, key)
