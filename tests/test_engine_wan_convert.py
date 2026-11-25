# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The install step that merges a LightX2V LoRA into a Wan 2.2 GGUF expert as scaled FP8 (engine.models.wan22.convert),
on a tiny Wan expert (Q6_K block linears, F16 rest, F32 patch embedding, as QuantStack's files) and a LoRA in
LightX2V's key style, both written by the test into a temporary folder: the layout of LightX2V's FP8 files, the merged
values, bit-identical results whatever the chunking or the number of threads, the FP8 loader and the distillation
check reading the result, the .part file, cancelling and the catalog checks. Everything runs on the CPU."""
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE = all(importlib.util.find_spec(name) is not None for name in ("torch", "safetensors", "gguf", "diffusers"))

DIM, FFN, TEXT, FREQ, LAYERS, RANK = 256, 512, 64, 32, 2, 4
ATTENTION = ("self_attn", "cross_attn")


def tiny_shapes() -> dict:
    """The tensors of a tiny Wan 2.2 I2V expert in the original key names (A14B: 5120, 13824, 4096, 256, 40)."""
    shapes = {"patch_embedding.weight": (DIM, 36, 1, 2, 2), "patch_embedding.bias": (DIM,),
              "text_embedding.0.weight": (DIM, TEXT), "text_embedding.0.bias": (DIM,),
              "text_embedding.2.weight": (DIM, DIM), "text_embedding.2.bias": (DIM,),
              "time_embedding.0.weight": (DIM, FREQ), "time_embedding.0.bias": (DIM,),
              "time_embedding.2.weight": (DIM, DIM), "time_embedding.2.bias": (DIM,),
              "time_projection.1.weight": (6 * DIM, DIM), "time_projection.1.bias": (6 * DIM,),
              "head.head.weight": (64, DIM), "head.head.bias": (64,), "head.modulation": (1, 2, DIM)}
    for block in range(LAYERS):
        prefix = f"blocks.{block}."
        shapes[prefix + "modulation"] = (1, 6, DIM)
        for attention in ATTENTION:
            for part in "qkvo":
                shapes[f"{prefix}{attention}.{part}.weight"] = (DIM, DIM)
                shapes[f"{prefix}{attention}.{part}.bias"] = (DIM,)
            for norm in ("norm_q", "norm_k"):
                shapes[f"{prefix}{attention}.{norm}.weight"] = (DIM,)
        shapes[prefix + "norm3.weight"] = (DIM,)
        shapes[prefix + "norm3.bias"] = (DIM,)
        shapes[prefix + "ffn.0.weight"] = (FFN, DIM)
        shapes[prefix + "ffn.0.bias"] = (FFN,)
        shapes[prefix + "ffn.2.weight"] = (DIM, FFN)
        shapes[prefix + "ffn.2.bias"] = (DIM,)
    return shapes


def tiny_config() -> dict:
    """The diffusers WanTransformer3DModel config of the tiny expert."""
    return {"patch_size": [1, 2, 2], "num_attention_heads": 2, "attention_head_dim": DIM // 2, "in_channels": 36,
            "out_channels": 16, "text_dim": TEXT, "freq_dim": FREQ, "ffn_dim": FFN, "num_layers": LAYERS,
            "cross_attn_norm": True, "qk_norm": "rms_norm_across_heads", "eps": 1e-6, "image_dim": None,
            "added_kv_proj_dim": None, "rope_max_seq_len": 1024, "pos_embed_seq_len": None}


def q6_k_rows(rows: int, columns: int, generator):
    """Random Q6_K blocks (210 bytes per 256 values) with a small float16 scale `d`, as uint8 [rows, bytes]."""
    import numpy as np
    import torch

    blocks = rows * columns // 256
    raw = torch.randint(0, 256, (blocks, 210), generator=generator, dtype=torch.int64).to(torch.uint8).numpy()
    scale = (torch.rand(blocks, generator=generator) * 2e-4 + 1e-4).numpy().astype(np.float16)
    raw[:, 208:210] = scale.view(np.uint8).reshape(blocks, 2)
    return raw.reshape(rows, columns // 256 * 210)


def write_inputs(folder: Path, seed: int = 0):
    """A tiny expert as GGUF and a LoRA that changes every tensor of it (as LightX2V's _1022 LoRA does)."""
    import gguf
    import numpy as np
    import torch
    from safetensors.torch import save_file

    from engine.models.wan22.convert import BLOCK_LINEAR

    generator = torch.Generator().manual_seed(seed)
    gguf_path, lora_path = folder / "expert-Q6_K.gguf", folder / "lora_rank4.safetensors"
    writer = gguf.GGUFWriter(str(gguf_path), "wan")
    lora = {}
    for name, shape in tiny_shapes().items():
        stem = name.rsplit(".", 1)[0]
        if BLOCK_LINEAR.match(name):
            writer.add_tensor(name, q6_k_rows(*shape, generator), raw_dtype=gguf.GGMLQuantizationType.Q6_K)
        else:
            value = torch.randn(shape, generator=generator) * 0.05 + (1.0 if "norm" in name else 0.0)
            writer.add_tensor(name, value.numpy().astype(np.float32 if name.startswith("patch") else np.float16))
        change = lambda size: (torch.randn(size, generator=generator) * 0.01).to(torch.bfloat16)  # noqa: E731
        if name.endswith("modulation"):
            lora[f"diffusion_model.{stem}.diff_m"] = change(shape)
        elif name.endswith(".bias"):
            lora[f"diffusion_model.{stem}.diff_b"] = change(shape)
        elif len(shape) == 2 and not name.startswith("head."):  # every linear but the head gets LoRA matrices
            lora[f"diffusion_model.{stem}.lora_down.weight"] = change((RANK, shape[1])) * 4
            lora[f"diffusion_model.{stem}.lora_up.weight"] = change((shape[0], RANK)) * 4
        else:  # norms, the patch embedding and the head: full changes
            lora[f"diffusion_model.{stem}.diff"] = change(shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    save_file(lora, str(lora_path))
    return gguf_path, lora_path


def header(path: Path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        raw = handle.read(size)
    return size, raw, json.loads(raw)


@unittest.skipUnless(HAVE, "torch, safetensors, gguf or diffusers is missing")
class MergeLoraToFP8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from engine.models.wan22 import convert

        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.gguf, cls.lora = write_inputs(cls.root)
        cls.output = cls.root / "expert_fp8.safetensors"
        cls.sha256 = convert.merge_lora_to_fp8(cls.gguf, cls.lora, cls.output)  # as the installer calls it

    def expected(self):
        """The merged tensors in float64 from the inputs, by an independent route (GGUF expanded by gguf-py, the LoRA
        product as one matrix product)."""
        import gguf
        import torch
        from safetensors import safe_open

        reader = gguf.GGUFReader(str(self.gguf))
        merged = {}
        with safe_open(str(self.lora), framework="pt") as changes:
            keys = set(changes.keys())
            for tensor in reader.tensors:
                shape = tuple(int(n) for n in reversed(tensor.shape.tolist()))
                value = torch.from_numpy(gguf.quants.dequantize(tensor.data, tensor.tensor_type).copy())
                value = value.reshape(shape).to(torch.float64)
                stem = "diffusion_model." + tensor.name.rsplit(".", 1)[0]
                suffix = {"weight": ".diff", "bias": ".diff_b", "modulation": ".diff_m"}[tensor.name.rsplit(".", 1)[1]]
                if tensor.name.endswith("modulation"):
                    stem = "diffusion_model." + tensor.name[:-len(".modulation")]
                if stem + suffix in keys:
                    value = value + changes.get_tensor(stem + suffix).double().reshape(shape)
                if stem + ".lora_down.weight" in keys and tensor.name.endswith(".weight"):
                    value = value + changes.get_tensor(stem + ".lora_up.weight").double() \
                        @ changes.get_tensor(stem + ".lora_down.weight").double()
                merged[tensor.name] = value
        return merged

    def test_a_safetensors_base_with_the_same_values_gives_the_same_bytes(self):
        """The official Index-AniSora experts are float32 safetensors in the original Wan names: read that way, the same
        values give the same file, byte for byte."""
        import torch
        from safetensors.torch import save_file

        from engine.formats.ggml import GGUFFile
        from engine.models.wan22 import convert

        with GGUFFile(self.gguf) as gguf_file:
            values = {name: tensor.to(torch.float32).contiguous() for name, tensor in gguf_file.tensors.items()}
        base = self.root / "expert_float32.safetensors"
        save_file(values, str(base))
        self.assertEqual(convert.merge_lora_to_fp8(base, self.lora, self.root / "from_safetensors.safetensors"),
                         self.sha256)
        save_file({name: value.to(torch.int32) for name, value in list(values.items())[:1]}, str(self.root / "int.st"))
        with self.assertRaisesRegex(convert.ConversionError, "float32, float16 or bfloat16"):
            convert.merge_lora_to_fp8(self.root / "int.st", self.lora, self.root / "never.safetensors")

    def test_the_wan21_image_branch_of_a_lora_is_left_out(self):
        """LightX2V's low-noise `_1022` LoRA also changes Wan 2.1's image key/value branch, which Wan 2.2 lacks: those
        keys are left out (as formats.lora leaves them out at run time) and the result is the same file."""
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        from engine.models.wan22 import convert

        with safe_open(str(self.lora), framework="pt", device="cpu") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        tensors["diffusion_model.blocks.0.cross_attn.k_img.lora_down.weight"] = torch.zeros(4, 8, dtype=torch.bfloat16)
        tensors["diffusion_model.blocks.0.cross_attn.k_img.lora_up.weight"] = torch.zeros(8, 4, dtype=torch.bfloat16)
        tensors["diffusion_model.blocks.0.cross_attn.norm_k_img.diff"] = torch.zeros(8, dtype=torch.bfloat16)
        wider = self.root / "lora_with_image_branch.safetensors"
        save_file(tensors, str(wider))
        self.assertEqual(convert.merge_lora_to_fp8(self.gguf, wider, self.root / "wider.safetensors"), self.sha256)

    def test_the_layout_is_that_of_lightx2vs_fp8_experts(self):
        from engine.models.wan22.convert import BLOCK_LINEAR

        size, raw, entries = header(self.output)
        self.assertNotIn("__metadata__", entries)
        self.assertEqual(size % 8, 0)
        self.assertEqual(raw, raw.rstrip(b" ") + b" " * (len(raw) - len(raw.rstrip(b" "))))
        self.assertEqual(raw.rstrip(b" "), json.dumps(entries, separators=(",", ":")).encode())  # compact JSON
        shapes = tiny_shapes()
        linears = sorted(name for name in shapes if BLOCK_LINEAR.match(name))
        self.assertEqual(len(linears), 10 * LAYERS)
        expected = {name: ("F32", list(shape)) for name, shape in shapes.items() if name not in linears}
        expected.update({name: ("F8_E4M3", list(shapes[name])) for name in linears})
        expected.update({name + "_scale": ("BF16", [shapes[name][0], 1]) for name in linears})
        self.assertEqual({name: (entry["dtype"], entry["shape"]) for name, entry in entries.items()}, expected)
        # float32, then bfloat16, then float8, each by name, back to back from offset 0
        order = sorted(entries, key=lambda name: ({"F32": 0, "BF16": 1, "F8_E4M3": 2}[entries[name]["dtype"]], name))
        self.assertEqual(list(entries), order)
        offset = 0
        for name in order:
            self.assertEqual(entries[name]["data_offsets"][0], offset, name)
            offset = entries[name]["data_offsets"][1]
        self.assertEqual(self.output.stat().st_size, 8 + size + offset)
        self.assertEqual(hashlib.sha256(self.output.read_bytes()).hexdigest(), self.sha256)

    def test_the_values_are_the_merged_expert_rounded_as_lightx2v_stores_them(self):
        import torch
        from safetensors import safe_open

        from engine.models.wan22.convert import BLOCK_LINEAR

        merged = self.expected()
        with safe_open(str(self.output), framework="pt") as result:
            for name, value in merged.items():
                with self.subTest(name=name):
                    if not BLOCK_LINEAR.match(name):
                        stored = result.get_tensor(name)
                        self.assertEqual(stored.dtype, torch.float32)
                        self.assertTrue(torch.equal(stored, stored.to(torch.bfloat16).float()))  # bfloat16 values
                        # within the bfloat16 rounding of the merged value (exact for a sum of two tensors)
                        torch.testing.assert_close(stored.double(), value, rtol=2 ** -8, atol=0)
                        continue
                    weight, scale = result.get_tensor(name), result.get_tensor(name + "_scale")
                    self.assertEqual((weight.dtype, scale.dtype), (torch.float8_e4m3fn, torch.bfloat16))
                    quantized = weight.float()
                    self.assertTrue(bool((quantized.abs().amax(dim=1) == 448).all()))  # each row's largest is 448
                    peak = value.abs().amax(dim=1)
                    torch.testing.assert_close(scale.double().reshape(-1), peak / 448, rtol=2 ** -8, atol=0)
                    restored = quantized.double() * scale.double()
                    # every value within half an e4m3 step (3 mantissa bits) of the merged one
                    error = (restored - value).abs()
                    bound = torch.maximum(value.abs() * 2 ** -4, scale.double() * 2 ** -10)
                    self.assertTrue(bool((error <= bound * 1.0001).all()), float((error / bound).max()))

    def test_the_bytes_do_not_depend_on_the_chunks_or_the_number_of_threads(self):
        import torch

        from engine.models.wan22 import convert

        threads = torch.get_num_threads()
        try:
            for chunk, count in ((300, 1), (1 << 14, max(2, threads)), (1 << 22, 3)):
                with self.subTest(chunk=chunk, threads=count), patch.object(convert, "CHUNK_VALUES", chunk):
                    torch.set_num_threads(count)
                    again = self.root / f"again_{chunk}.safetensors"
                    self.assertEqual(convert.merge_lora_to_fp8(self.gguf, self.lora, again), self.sha256)
                    self.assertEqual(again.read_bytes(), self.output.read_bytes())
                    again.unlink()
        finally:
            torch.set_num_threads(threads)

    def test_the_fp8_loader_reads_it_and_the_network_matches_the_gguf_expert_with_its_lora(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        from engine.core.fp8_linear import FP8Linear
        from engine.formats import fp8, lora
        from engine.models.wan22 import transformer

        self.assertTrue(fp8.channel_scaled(self.output))
        config = tiny_config()
        with safe_open(str(self.output), framework="pt") as stored:  # the same values as a plain float32 file
            expanded = {name: stored.get_tensor(name).float() * stored.get_tensor(name + "_scale").float()
                        if name + "_scale" in stored.keys() else stored.get_tensor(name)
                        for name in stored.keys() if not name.endswith("_scale")}
        dense_path = self.root / "expanded.safetensors"
        save_file(expanded, str(dense_path))
        models = {"packed": transformer.load_transformer(self.output, "cpu", torch.float32, kind="i2v", config=config),
                  "expanded": transformer.load_transformer(dense_path, "cpu", torch.float32, kind="i2v", config=config),
                  "gguf": transformer.load_transformer(self.gguf, "cpu", torch.float32, kind="i2v", config=config),
                  "gguf_lora": transformer.load_transformer(self.gguf, "cpu", torch.float32, kind="i2v", config=config)}
        self.assertEqual(sum(isinstance(module, FP8Linear) for module in models["packed"].modules()), 10 * LAYERS)
        names = [name for name, _ in models["gguf_lora"].named_modules()]
        lora.attach(models["gguf_lora"], [(lora.load_lora(self.lora, names), 1.0)])
        generator = torch.Generator().manual_seed(3)
        latent = torch.randn(1, 16, 2, 8, 8, generator=generator)
        concat = torch.randn(1, 20, 2, 8, 8, generator=generator)
        context = torch.randn(1, 12, TEXT, generator=generator)
        with torch.no_grad():
            out = {name: transformer.predict_velocity(model, latent, torch.tensor(0.7), context, concat)
                   for name, model in models.items()}
        del models

        def distance(a, b):
            return float((out[a] - out[b]).norm() / out[b].norm())

        # every tensor is read under its own name, each FP8 row with its own scale
        self.assertEqual(distance("packed", "expanded"), 0.0)
        # the LoRA is merged in: the FP8 rounding (6% on this random network) is far below what the LoRA changes
        self.assertLess(distance("packed", "gguf_lora"), 0.1)
        self.assertGreater(distance("gguf", "gguf_lora"), 3 * distance("packed", "gguf_lora"))

    def test_the_distillation_check_finds_the_lora_merged_in(self):
        from engine.models.wan22.transformer import lora_merged_share

        self.assertEqual(lora_merged_share(self.output, self.gguf, self.lora, blocks=tuple(range(LAYERS))), 1.0)

    def test_a_lora_of_low_rank_matrices_only_is_checked_on_sampled_rows(self):
        """LightX2V's text-to-video _1217 LoRAs hold no full changes of biases or norms: the check makes sampled rows
        of the changed matrices again and compares the FP8 bytes and scales."""
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        from engine.models.wan22 import convert
        from engine.models.wan22.transformer import lora_merged_share

        with safe_open(str(self.lora), framework="pt") as handle:
            low_rank = {key: handle.get_tensor(key) for key in handle.keys() if ".lora_" in key}
        generator = torch.Generator().manual_seed(7)
        other = {key: (torch.randn(value.shape, generator=generator) * 0.04).to(torch.bfloat16)
                 for key, value in low_rank.items()}
        files = {}
        for name, tensors in (("low_rank", low_rank), ("other", other)):
            lora = self.root / f"{name}.safetensors"
            save_file(tensors, str(lora))
            output = self.root / f"{name}_fp8.safetensors"
            convert.merge_lora_to_fp8(self.gguf, lora, output)
            files[name] = lora, output
        blocks = tuple(range(LAYERS))
        lora, made = files["low_rank"]
        self.assertEqual(lora_merged_share(made, self.gguf, lora, blocks=blocks), 1.0)
        self.assertEqual(lora_merged_share(self.output, self.gguf, lora, blocks=blocks), 1.0)  # same matrices merged
        self.assertLess(lora_merged_share(files["other"][1], self.gguf, lora, blocks=blocks), 0.2)  # another LoRA

    def test_the_part_file_cancelling_and_the_catalog_checks(self):
        from engine.models.wan22 import convert

        size = self.output.stat().st_size
        target = self.root / "checked.safetensors"
        part = target.with_name(target.name + ".part")
        with self.assertRaisesRegex(convert.ConversionError, "catalog expects"):
            convert.merge_lora_to_fp8(self.gguf, self.lora, target, expected_size=size + 8)
        self.assertFalse(part.exists() or target.exists())  # refused before writing
        with self.assertRaisesRegex(convert.ConversionError, "cancelled"):
            convert.merge_lora_to_fp8(self.gguf, self.lora, target, cancel_flag=lambda: True)
        self.assertTrue(part.is_file())
        self.assertFalse(target.exists())
        with self.assertRaisesRegex(convert.ConversionError, "Integrity"):
            convert.merge_lora_to_fp8(self.gguf, self.lora, target, expected_sha256="0" * 64)
        self.assertEqual(part.stat().st_size, size)  # kept for inspection, never published
        self.assertFalse(target.exists())
        progress = []
        self.assertEqual(convert.merge_lora_to_fp8(self.gguf, self.lora, target, expected_size=size,
                                                   expected_sha256=self.sha256.upper(),
                                                   progress_cb=lambda done, total: progress.append((done, total))),
                         self.sha256)
        self.assertFalse(part.exists())
        self.assertEqual(target.read_bytes(), self.output.read_bytes())
        self.assertEqual(progress[-1], (size, size))
        self.assertEqual([done for done, _ in progress], sorted(done for done, _ in progress))

    def test_inputs_that_do_not_fit_are_refused(self):
        import gguf
        import torch
        from safetensors.torch import save_file

        from engine.models.wan22 import convert

        cases = {
            "change no tensor": {"diffusion_model.blocks.7.ffn.0.diff_b": torch.zeros(FFN)},
            "do not fit": {"diffusion_model.blocks.0.ffn.0.lora_down.weight": torch.zeros(RANK, DIM),
                           "diffusion_model.blocks.0.ffn.0.lora_up.weight": torch.zeros(DIM, RANK)},
            "only one of the two": {"diffusion_model.blocks.0.ffn.0.lora_down.weight": torch.zeros(RANK, DIM)},
            "has shape": {"diffusion_model.blocks.0.ffn.0.diff_b": torch.zeros(FFN + 1)},
            "not bfloat16": {"diffusion_model.blocks.0.ffn.0.diff_b": torch.zeros(FFN, dtype=torch.float64)},
        }
        for message, tensors in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(convert.ConversionError, message):
                bad = self.root / "bad_lora.safetensors"
                save_file(tensors, str(bad))
                convert.merge_lora_to_fp8(self.gguf, bad, self.root / "never.safetensors")
        for arch, names in (("llama", {"blocks.0.ffn.0.weight": (32, 32)}), ("wan", {"head.head.bias": (64,)})):
            with self.subTest(arch=arch), self.assertRaises(convert.ConversionError):
                path = self.root / f"other_{arch}.gguf"
                writer = gguf.GGUFWriter(str(path), arch)
                for name, shape in names.items():
                    writer.add_tensor(name, torch.zeros(shape).numpy())
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                save_file({"diffusion_model.head.head.diff_b": torch.zeros(64)}, str(self.root / "small.safetensors"))
                convert.merge_lora_to_fp8(path, self.root / "small.safetensors", self.root / "never.safetensors")
        self.assertFalse((self.root / "never.safetensors").exists())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(HAVE, "torch, safetensors, gguf or diffusers is missing")
class InstallStepTests(unittest.TestCase):
    """The step through the installer as the catalog names it: its two inputs are files of another entry, each in its
    own folder and from its own repository and revision; those on disk are read in place and kept, a missing one is
    downloaded at its own pin, and the result is published only with the catalog's size and SHA-256."""

    @classmethod
    def setUpClass(cls):
        from engine.models.wan22 import convert

        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.inputs = Path(cls.tmp.name)
        cls.gguf, cls.lora = write_inputs(cls.inputs, seed=1)
        made = cls.inputs / "made.safetensors"
        cls.sha256 = convert.merge_lora_to_fp8(cls.gguf, cls.lora, made)
        cls.size = made.stat().st_size
        made.unlink()

    def setUp(self):
        from app import config, model_manager

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.served = {("quant/expert-GGUF", "High/expert-Q6_K.gguf", "a" * 40): self.gguf,
                       ("distill/loras", "lora_rank4.safetensors", "b" * 40): self.lora}
        sources = [{"repo_id": repo, "filename": filename, "dest": dest, "revision": revision,
                    "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                   for (repo, filename, revision), path, dest in zip(self.served, self.served.values(),
                                                                     ("diffusion_models", "loras"))]
        self.spec = {"filename": "expert_fp8.safetensors", "dest": "fp8_local", "size_bytes": self.size,
                     "sha256": self.sha256,
                     "convert": {"tool": "engine.models.wan22.convert.merge_lora_to_fp8",
                                 "input": ["expert-Q6_K.gguf", "lora_rank4.safetensors"], "sources": sources,
                                 "delete_sources_after": False}}
        self.model = {"id": "fast", "name": "Fast — test", "repo_id": "published/models", "files": [self.spec]}
        self.fetched = []
        for target in (patch.object(config, "destination_dir", lambda dest: self.root / dest),
                       patch.dict(config.CATALOG, {"models": [self.model]}),
                       patch.object(model_manager, "get_hf_token", return_value=None),
                       patch.object(model_manager, "download_file", side_effect=self.download)):
            target.start()
            self.addCleanup(target.stop)

    def download(self, repo_id, filename, dest_dir, token=None, progress_cb=None, cancel_flag=None, save_as=None,
                 revision="main", expected_size=0, sha256=None):
        """A pinned Hugging Face download: keeps a file that is there, else copies the served one."""
        final = Path(dest_dir) / (save_as or filename.rsplit("/", 1)[-1])
        if not final.is_file():
            self.fetched.append((repo_id, filename, revision, Path(dest_dir).name))
            final.parent.mkdir(parents=True, exist_ok=True)
            final.write_bytes(self.served[(repo_id, filename, revision)].read_bytes())
        self.assertEqual((final.stat().st_size, globals()["sha256"](final)), (expected_size, sha256))
        return final

    def place(self, path: Path, folder: str) -> Path:
        target = self.root / folder / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        return target

    def test_inputs_on_disk_are_read_where_they_are_and_kept(self):
        from app import model_manager

        gguf, lora = self.place(self.gguf, "diffusion_models"), self.place(self.lora, "loras")
        self.assertEqual(model_manager.missing_bytes(["fast"]), (0, self.size))
        self.assertEqual(model_manager.step_inputs(self.spec, self.model), [gguf, lora])
        model_manager.download_model("fast")
        self.assertEqual(self.fetched, [])
        result = self.root / "fp8_local" / "expert_fp8.safetensors"
        self.assertEqual(sha256(result), self.sha256)
        self.assertTrue(gguf.is_file() and lora.is_file())
        self.assertTrue(model_manager.installed(self.model))
        self.assertEqual(model_manager.model_path("fast", "fp8_local", "fp8"), result)

    def test_missing_inputs_are_downloaded_at_their_own_pins_into_their_own_folders(self):
        from app import model_manager

        self.place(self.lora, "loras")
        model_manager.download_model("fast")
        self.assertEqual(self.fetched, [("quant/expert-GGUF", "High/expert-Q6_K.gguf", "a" * 40, "diffusion_models")])
        self.assertEqual(sha256(self.root / "fp8_local" / "expert_fp8.safetensors"), self.sha256)

    def test_a_result_that_does_not_match_the_catalog_is_not_published(self):
        from app import downloader, model_manager

        self.spec["sha256"] = "0" * 64
        with self.assertRaisesRegex(downloader.DownloadError, "Integrity"):
            model_manager.download_model("fast")
        self.assertFalse((self.root / "fp8_local" / "expert_fp8.safetensors").exists())
        self.assertTrue((self.root / "fp8_local" / "expert_fp8.safetensors.part").is_file())


@unittest.skipUnless(HAVE, "torch, safetensors, gguf or diffusers is missing")
class RoundingTests(unittest.TestCase):
    """The roundings are done with integer arithmetic on the bits, so no CPU can round differently; they agree with
    torch's casts (round to nearest, ties to even) on every float32 whose upper 16 bits are any pattern."""

    def values(self):
        import torch

        upper = torch.arange(0, 1 << 16, dtype=torch.int64) << 16
        lows = torch.tensor([0, 1, 0x7FFF, 0x8000, 0x8001, 0xFFFF, 0x1234, 0xBEEF], dtype=torch.int64)
        bits = (upper[:, None] | lows[None, :]).reshape(-1)
        values = torch.where(bits >= 1 << 31, bits - (1 << 32), bits).to(torch.int32).view(torch.float32)
        return values[torch.isfinite(values)]

    def test_bfloat16(self):
        import torch

        from engine.models.wan22.convert import round_bf16

        values = self.values()
        self.assertTrue(torch.equal(round_bf16(values).view(torch.int32),
                                    values.to(torch.bfloat16).float().view(torch.int32)))

    def test_float8_e4m3fn(self):
        import torch

        from engine.models.wan22.convert import e4m3_bytes

        values = self.values()
        values = values[values.abs() <= 448]
        self.assertTrue(torch.equal(e4m3_bytes(values), values.to(torch.float8_e4m3fn).view(torch.uint8)))
        self.assertEqual(e4m3_bytes(torch.tensor([1.0, 448.0, -448.0, 2.0 ** -9, 2.0 ** -10, 0.0])).tolist(),
                         [0x38, 0x7E, 0xFE, 0x01, 0x00, 0x00])

    def test_the_low_rank_product_rounds_once_per_term_whether_or_not_it_is_fused(self):
        import torch

        from engine.models.wan22.convert import low_rank_product

        generator = torch.Generator().manual_seed(1)
        up = torch.randn(64, 8, generator=generator).to(torch.bfloat16)
        down = (torch.randn(8, 48, generator=generator) * 1e-3).to(torch.bfloat16)
        product = low_rank_product(up, down)
        unfused = up[:, 0:1].double() * down[0:1].double()
        for r in range(1, 8):
            unfused = unfused + up[:, r:r + 1].double() * down[r:r + 1].double()
        self.assertTrue(torch.equal(product, unfused))
        torch.testing.assert_close(product, up.double() @ down.double(), rtol=1e-12, atol=0)


if __name__ == "__main__":
    unittest.main()
