# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The official Z-Image-Turbo files (Tongyi-MAI/Z-Image-Turbo): sharded reading, the installer's bf16 merge, and
the loaders on the official names.

On tiny deterministic networks (tests/seeded.py, the fixtures of test_engine_zimage): a network loaded from files in
the official layout (diffusers names, sharded, merged to bf16) must equal the one loaded from the single-file layout
it replaces. The header tests at the end read only the headers of the real official files and skip when they are
not installed.
"""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from app.config import MODELS_DIR
from tests import seeded

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
OFFICIAL = MODELS_DIR / "official" / "Z-Image-Turbo"
TRANSFORMER_INDEX = OFFICIAL / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
TEXT_ENCODER_INDEX = OFFICIAL / "text_encoder" / "model.safetensors.index.json"
VAE_FILE = OFFICIAL / "vae" / "diffusion_pytorch_model.safetensors"
TRANSFORMER_BF16 = OFFICIAL / "transformer_bf16" / "z-image-turbo-transformer-bf16.safetensors"
TEXT_ENCODER_BF16 = OFFICIAL / "text_encoder_bf16" / "qwen3-4b-text-encoder-bf16.safetensors"

HAVE = all(importlib.util.find_spec(name) for name in ("torch", "diffusers", "transformers", "safetensors"))


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"engine_zimage_{name}.json").read_text(encoding="utf-8"))


def save_sharded(folder: Path, tensors: dict, shards: int, index_name: str) -> Path:
    """`tensors` as `shards` safetensors files plus a Hugging Face index, like the official release."""
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    weight_map = {}
    for number in range(shards):
        part = names[number::shards]
        shard = f"model-{number + 1:05d}-of-{shards:05d}.safetensors"
        save_file({k: tensors[k].contiguous() for k in part}, str(folder / shard), metadata={"format": "pt"})
        weight_map.update({k: shard for k in part})
    index = folder / index_name
    index.write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}), encoding="utf-8")
    return index


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bf16_bits(values):
    """float32 -> bfloat16 bit patterns by the textbook rule (round to nearest, ties to even), for finite values."""
    import torch

    bits = values.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.to(torch.int32).to(torch.int16)


@unittest.skipUnless(HAVE, "torch, diffusers, transformers or safetensors are not installed")
class MergeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def tensors(self):
        import torch

        generator = torch.Generator().manual_seed(4)
        return {"b.weight": torch.randn(5, 7, generator=generator), "a.bias": torch.randn(7, generator=generator),
                "c.scale": torch.randn(1, 3, generator=generator), "a.weight": torch.randn(3, 2, 4, generator=generator)}

    def test_sharded_float32_becomes_one_bf16_file_in_the_save_file_layout(self):
        import torch
        from engine.models.zimage import checkpoint
        from safetensors import safe_open
        from safetensors.torch import save_file

        source = self.tensors()
        index = save_sharded(self.root / "src", source, 3, "x.safetensors.index.json")
        output = self.root / "out" / "merged.safetensors"
        digest = checkpoint.merge(index, output)
        self.assertEqual(digest, sha256(output))
        self.assertFalse(output.with_name(output.name + ".part").exists())
        reference = self.root / "reference.safetensors"
        save_file({k: v.to(torch.bfloat16) for k, v in source.items()}, str(reference))
        self.assertEqual(output.read_bytes(), reference.read_bytes())  # the same tensors give the same bytes
        with safe_open(str(output), framework="pt") as handle:
            self.assertEqual(list(handle.keys()), sorted(source))
            self.assertIsNone(handle.metadata())
            for name, value in source.items():
                got = handle.get_tensor(name)
                self.assertEqual(got.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(got.view(torch.int16), bf16_bits(value)))

    def test_the_cast_rounds_to_nearest_with_ties_to_even(self):
        import torch
        from engine.models.zimage import checkpoint
        from safetensors import safe_open
        from safetensors.torch import save_file

        # 1 + 2^-8 lies halfway between two bf16 values: it goes to the even one (1.0); 1 + 3 * 2^-8 goes up.
        values = torch.tensor([1 + 2 ** -8, 1 + 3 * 2 ** -8, -(1 + 2 ** -8), 1 + 2 ** -8 + 2 ** -20, 3.0e-41, 65504.0])
        source = self.root / "ties.safetensors"
        save_file({"t": values}, str(source))
        output = self.root / "ties_bf16.safetensors"
        checkpoint.merge(source, output)
        with safe_open(str(output), framework="pt") as handle:
            got = handle.get_tensor("t")
        self.assertTrue(torch.equal(got.view(torch.int16), bf16_bits(values)))
        self.assertEqual(got[:4].float().tolist(), [1.0, 1 + 2 ** -6, -1.0, 1 + 2 ** -7])

    def test_bf16_sources_are_copied_unchanged(self):
        import torch
        from engine.models.zimage import checkpoint
        from safetensors.torch import save_file

        source = {k: v.to(torch.bfloat16) for k, v in self.tensors().items()}
        index = save_sharded(self.root / "src", source, 2, "model.safetensors.index.json")
        output = self.root / "merged.safetensors"
        checkpoint.merge(self.root / "src", output)  # a folder with one index
        reference = self.root / "reference.safetensors"
        save_file(source, str(reference))
        self.assertEqual(sha256(output), sha256(reference))
        self.assertEqual(dict(checkpoint.tensors(index)).keys(), source.keys())

    def test_expected_size_and_hash_guard_the_result(self):
        from engine.models.zimage import checkpoint

        import torch
        from safetensors.torch import save_file

        source = self.tensors()
        index = save_sharded(self.root / "src", source, 2, "x.safetensors.index.json")
        reference = self.root / "reference.safetensors"  # merge writes exactly the bytes of save_file (see above)
        save_file({k: v.to(torch.bfloat16) for k, v in source.items()}, str(reference))
        size = reference.stat().st_size
        output = self.root / "merged.safetensors"
        part = output.with_name(output.name + ".part")
        with self.assertRaisesRegex(checkpoint.CheckpointError, "bytes"):
            checkpoint.merge(index, output, expected_size=size + 1)
        self.assertFalse(part.exists())  # refused before writing anything
        with self.assertRaisesRegex(checkpoint.CheckpointError, "SHA-256"):
            checkpoint.merge(index, output, expected_sha256="0" * 64)
        self.assertFalse(output.exists())
        self.assertTrue(part.exists())  # kept for inspection, overwritten by the next run
        digest = checkpoint.merge(index, output, expected_size=size)
        self.assertEqual(checkpoint.merge(index, output, expected_sha256=digest.upper()), digest)
        self.assertFalse(part.exists())

    def test_cancel_stops_before_the_file_is_published(self):
        from engine.models.zimage import checkpoint

        index = save_sharded(self.root / "src", self.tensors(), 2, "x.safetensors.index.json")
        output = self.root / "merged.safetensors"
        calls = []
        with self.assertRaisesRegex(checkpoint.CheckpointError, "cancelled"):
            checkpoint.merge(index, output, cancel_flag=lambda: calls.append(1) or len(calls) > 2)
        self.assertFalse(output.exists())

    def test_damaged_or_hostile_indexes_are_refused(self):
        import torch
        from engine.models.zimage import checkpoint
        from safetensors.torch import save_file

        folder = self.root / "src"
        index = save_sharded(folder, self.tensors(), 2, "x.safetensors.index.json")
        data = json.loads(index.read_text(encoding="utf-8"))
        for shard in ("../outside.safetensors", "sub/model.safetensors", "..\\model.safetensors", "model.bin"):
            with self.subTest(shard=shard):
                bad = folder / "bad.safetensors.index.json"
                bad.write_text(json.dumps({"weight_map": {"a.bias": shard}}), encoding="utf-8")
                with self.assertRaisesRegex(checkpoint.CheckpointError, "invalid shard"):
                    checkpoint.names(bad)
                bad.unlink()
        missing = dict(data["weight_map"], extra=data["weight_map"]["a.bias"])  # listed but not stored
        index.write_text(json.dumps({"weight_map": missing}), encoding="utf-8")
        with self.assertRaisesRegex(checkpoint.CheckpointError, "does not match its index"):
            list(checkpoint.tensors(index))
        index.write_text(json.dumps({"weight_map": {}}), encoding="utf-8")
        with self.assertRaisesRegex(checkpoint.CheckpointError, "empty"):
            checkpoint.names(index)
        (folder / "second.safetensors.index.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(checkpoint.CheckpointError, "exactly one"):
            checkpoint.resolve(folder)
        save_file({"ids": torch.arange(3)}, str(self.root / "ints.safetensors"))
        with self.assertRaisesRegex(checkpoint.CheckpointError, "only float"):
            checkpoint.merge(self.root / "ints.safetensors", self.root / "ints_bf16.safetensors")


@unittest.skipUnless(HAVE, "torch, diffusers, transformers or safetensors are not installed")
class OfficialLayoutLoadsTheSameNetworkTests(unittest.TestCase):
    """The tiny networks of the pinned tests, saved in the official layout, must load to the same weights."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def assert_same(self, a, b):
        import torch

        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(sorted(sa), sorted(sb))
        for name in sa:
            self.assertEqual(sa[name].dtype, sb[name].dtype, name)
            self.assertTrue(torch.equal(sa[name], sb[name]), name)

    def test_transformer(self):
        import torch
        from engine.models.zimage import checkpoint, transformer
        from safetensors.torch import save_file

        data = fixture("transformer")
        single = seeded.seeded_state_dict(data["shapes"])  # the fused-qkv layout of the pinned test
        single_file = self.root / "single.safetensors"
        save_file({k: v.contiguous() for k, v in single.items()}, str(single_file))
        official = {k: v.contiguous() for k, v in transformer.diffusers_tensors(iter(single.items()))}
        self.assertIn(transformer.OFFICIAL_NAME, official)
        index = save_sharded(self.root / "official", official, 3, "diffusion_pytorch_model.safetensors.index.json")
        expected = transformer.load_transformer(single_file, "cpu", torch.float32, config=data["config"])
        self.assert_same(transformer.load_transformer(index, "cpu", torch.float32, config=data["config"]), expected)
        merged = self.root / "merged.safetensors"
        checkpoint.merge(index, merged)
        expected16 = transformer.load_transformer(single_file, "cpu", torch.bfloat16, config=data["config"])
        self.assert_same(transformer.load_transformer(merged, "cpu", torch.bfloat16, config=data["config"]), expected16)

    def test_text_encoder(self):
        import torch
        from engine.models.zimage import checkpoint, text
        from safetensors.torch import save_file

        data = fixture("text")
        tensors = {k: v.to(torch.bfloat16).contiguous() for k, v in seeded.seeded_state_dict(data["shapes"]).items()}
        single_file = self.root / "single.safetensors"
        save_file(tensors, str(single_file))
        index = save_sharded(self.root / "official", tensors, 3, "model.safetensors.index.json")
        merged = self.root / "merged.safetensors"
        checkpoint.merge(index, merged)
        self.assertEqual(sha256(merged), sha256(single_file))  # bf16 shards merge into the very same file
        expected = text.load_text_encoder(single_file, "cpu", torch.bfloat16, config=data["config"])
        self.assert_same(text.load_text_encoder(index, "cpu", torch.bfloat16, config=data["config"]), expected)
        self.assert_same(text.load_text_encoder(merged, "cpu", torch.bfloat16, config=data["config"]), expected)

    def test_vae_in_diffusers_names_decodes_like_the_original_layout(self):
        import numpy as np
        import torch
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
        from engine.models.zimage import vae
        from safetensors.torch import save_file

        data = fixture("vae")
        original = seeded.seeded_state_dict(data["shapes"])
        original_file = self.root / "original.safetensors"
        save_file({k: v.contiguous() for k, v in original.items()}, str(original_file))
        renamed = convert_ldm_vae_checkpoint(dict(original), vae._config(data["config"]))
        self.assertIn(vae.DIFFUSERS_NAME, renamed)
        official_file = self.root / "official.safetensors"
        save_file({k: v.contiguous() for k, v in renamed.items()}, str(official_file))
        expected = vae.load_vae(original_file, "cpu", torch.float32, config=data["config"])
        got = vae.load_vae(official_file, "cpu", torch.float32, config=data["config"])
        self.assert_same(got, expected)
        latent = seeded.seeded_inputs((1, 16, 6, 8), data["latent_seed"])
        self.assertTrue(np.array_equal(np.asarray(vae.decode(got, latent)), np.asarray(vae.decode(expected, latent))))


@unittest.skipUnless(HAVE, "torch, diffusers, transformers or safetensors are not installed")
class OfficialFileHeaderTests(unittest.TestCase):
    """The key and shape sets of the installed official files against the networks of the shipped configs, on the
    meta device: only the headers are read."""

    @staticmethod
    def shapes(path):
        from engine.models.zimage import checkpoint
        from safetensors import safe_open

        found = {}
        files = set(checkpoint.shard_map(path).values()) if path.name.endswith(checkpoint.INDEX_SUFFIX) else {path}
        for file in sorted(files):
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                found.update({key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()})
        return found

    def assert_matches(self, model, shapes):
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        self.assertEqual(sorted(set(expected) ^ set(shapes)), [])
        self.assertEqual({k for k in expected if expected[k] != shapes[k]}, set())

    def need(self, path: Path) -> Path:
        if not path.is_file():
            self.skipTest(f"{path.name} is not installed")
        return path

    @staticmethod
    def installed(path: Path) -> bool:
        """The file, or for an index the index and every shard it names, is on disk."""
        from engine.models.zimage import checkpoint

        if not path.is_file():
            return False
        return not path.name.endswith(checkpoint.INDEX_SUFFIX) or all(
            shard.is_file() for shard in checkpoint.shard_map(path).values())

    def each_installed(self, *paths: Path):
        found = [path for path in paths if self.installed(path)]
        if not found:
            self.skipTest("the official files are not installed")
        for path in found:
            with self.subTest(file=path.name):
                yield path

    def test_transformer(self):
        from engine.models.zimage import transformer

        for path in self.each_installed(TRANSFORMER_INDEX, TRANSFORMER_BF16):
            self.assert_matches(transformer._build(), self.shapes(path))

    def test_text_encoder(self):
        from engine.models.zimage import text

        for path in self.each_installed(TEXT_ENCODER_INDEX, TEXT_ENCODER_BF16):
            self.assert_matches(text._build(), {k.removeprefix("model."): v for k, v in self.shapes(path).items()})

    def test_vae(self):
        import torch
        from diffusers import AutoencoderKL
        from engine.models.zimage import vae

        with torch.device("meta"):
            model = AutoencoderKL.from_config(vae._config(None))
        self.assert_matches(model, self.shapes(self.need(VAE_FILE)))


if __name__ == "__main__":
    unittest.main()
