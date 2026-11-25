# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Catalog files made on the user's PC from official files ("convert"), files of earlier installations that hold the
same weights ("equivalents") and downloads from plain HTTPS addresses, against a real local HTTP server."""
import hashlib
import importlib.util
import io
import json
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import config, downloader, model_manager

MEMBER = b"author's weights " * 300
ARCHIVE = io.BytesIO()
with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as _bundle:
    _bundle.writestr("train_log/flownet.pkl", MEMBER)
    _bundle.writestr("train_log/IFNet_HDv3.py", b"# code\n")
ARCHIVE = ARCHIVE.getvalue()
OLD = b"same weights, earlier name " * 100


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Server:
    files: dict = {}
    requests: list = []

    @classmethod
    def start(cls):
        files, requests = cls.files, cls.requests

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append((self.path, dict(self.headers)))
                data = files.get(self.path.split("?")[0])
                if data is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                start = int(self.headers.get("Range", "bytes=0-")[6:-1] or 0)
                if start:
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                self.wfile.write(data[start:])

            def log_message(self, *_):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        return f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def stop(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()


class InstallStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = _Server.start()
        _Server.files["/archive.zip"] = ARCHIVE

    @classmethod
    def tearDownClass(cls):
        _Server.stop()

    def setUp(self):
        _Server.requests.clear()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.spec = {
            "filename": "archive.zip", "save_as": "flownet_v4.9.pkl", "dest": "rife",
            "size_bytes": len(MEMBER), "sha256": sha(MEMBER),
            "convert": {"tool": "app.downloader.extract_member", "source_dest": "rife", "input": "archive.zip",
                        "member": "train_log/flownet.pkl",
                        "sources": [{"filename": "archive.zip", "url": self.base + "/archive.zip?export=download",
                                     "size_bytes": len(ARCHIVE), "sha256": sha(ARCHIVE)}],
                        "delete_sources_after": True},
            "equivalents": [{"filename": "rife49.pth", "size_bytes": len(OLD), "sha256": sha(OLD)}],
        }
        self.model = {"id": "video", "name": "Video — test", "repo_id": "official/video", "files": [self.spec]}
        for target in (patch.object(config, "destination_dir", lambda dest: self.root / dest),
                       patch.dict(config.CATALOG, {"models": [self.model]}),
                       patch.object(model_manager, "get_hf_token", return_value="hf_secret")):
            target.start()
            self.addCleanup(target.stop)

    def test_the_member_is_extracted_verified_and_the_archive_deleted(self):
        self.assertFalse(model_manager.installed(self.model))
        model_manager.download_model("video")
        self.assertEqual((self.root / "rife" / "flownet_v4.9.pkl").read_bytes(), MEMBER)
        self.assertEqual(sorted(p.name for p in (self.root / "rife").iterdir()), ["flownet_v4.9.pkl"])
        self.assertTrue(model_manager.installed(self.model))
        self.assertEqual(model_manager.model_path("video", "rife"), self.root / "rife" / "flownet_v4.9.pkl")

    def test_a_plain_address_never_receives_the_hugging_face_token(self):
        model_manager.download_model("video")
        self.assertEqual(len(_Server.requests), 1)
        self.assertFalse({"authorization", "Authorization"} & set(_Server.requests[0][1]))

    def test_the_step_runs_only_while_its_result_is_missing(self):
        model_manager.download_model("video")
        _Server.requests.clear()
        with patch.object(downloader, "extract_member", side_effect=AssertionError("ran again")):
            model_manager.download_model("video")
            self.assertTrue(model_manager.installed(self.model))
        self.assertEqual(_Server.requests, [])

    def test_sources_already_on_disk_are_verified_not_downloaded(self):
        (self.root / "rife").mkdir()
        (self.root / "rife" / "archive.zip").write_bytes(ARCHIVE)
        self.spec["convert"]["delete_sources_after"] = False
        model_manager.download_model("video")
        self.assertEqual(_Server.requests, [])
        self.assertEqual((self.root / "rife" / "archive.zip").read_bytes(), ARCHIVE)  # kept as told
        self.assertTrue(model_manager.installed(self.model))

    def test_an_earlier_installations_file_with_the_same_weights_counts_and_stays(self):
        old = self.root / "rife" / "rife49.pth"
        old.parent.mkdir()
        old.write_bytes(OLD)
        self.assertTrue(model_manager.installed(self.model))
        model_manager.download_model("video")
        self.assertEqual(_Server.requests, [])
        self.assertEqual(old.read_bytes(), OLD)
        self.assertFalse((self.root / "rife" / "flownet_v4.9.pkl").exists())
        self.assertEqual(model_manager.model_path("video", "rife"), old)
        old.write_bytes(b"x" * len(OLD))  # same size, other bytes: not the weights
        self.assertFalse(model_manager.installed(self.model))

    def test_an_earlier_file_with_the_same_bytes_elsewhere_is_linked_into_place(self):
        """An equivalent with the catalog file's own SHA-256 (another folder, an earlier name): hard-linked under the
        catalog name, so neither the archive is downloaded nor the step run; the earlier file stays as it was."""
        old = self.root / "old_folder" / "earlier_name.pkl"
        old.parent.mkdir()
        old.write_bytes(MEMBER)
        self.spec["equivalents"] = [{"dest": "old_folder", "filename": old.name, "size_bytes": len(MEMBER),
                                     "sha256": sha(MEMBER)}]
        target = self.root / "rife" / "flownet_v4.9.pkl"
        with patch.object(downloader, "extract_member", side_effect=AssertionError("step ran")):
            self.assertTrue(model_manager.installed(self.model))
            self.assertEqual(target.read_bytes(), MEMBER)
            self.assertEqual(target.stat().st_ino, old.stat().st_ino)  # a link, no copy
            model_manager.download_model("video")
        self.assertEqual(_Server.requests, [])
        self.assertEqual(old.read_bytes(), MEMBER)
        self.assertEqual(model_manager.model_path("video", "rife"), target)

    def test_an_earlier_file_elsewhere_is_used_where_it_is_when_it_cannot_be_linked(self):
        old = self.root / "old_folder" / "earlier_name.pkl"
        old.parent.mkdir()
        old.write_bytes(MEMBER)
        self.spec["equivalents"] = [{"dest": "old_folder", "filename": old.name, "size_bytes": len(MEMBER),
                                     "sha256": sha(MEMBER)}]
        with patch.object(model_manager.os, "link", side_effect=OSError("another volume")):
            self.assertTrue(model_manager.installed(self.model))
            model_manager.download_model("video")
        self.assertEqual(_Server.requests, [])
        self.assertFalse((self.root / "rife" / "flownet_v4.9.pkl").exists())
        self.assertEqual(model_manager.model_path("video", "rife"), old)
        old.write_bytes(b"y" * len(MEMBER))  # same size, other bytes: neither linked nor counted
        self.assertFalse(model_manager.installed(self.model))
        self.assertFalse((self.root / "rife" / "flownet_v4.9.pkl").exists())

    def test_a_result_that_does_not_match_the_catalog_is_never_published(self):
        self.spec["sha256"] = "0" * 64
        with self.assertRaisesRegex(downloader.DownloadError, "Integrity"):
            model_manager.download_model("video")
        self.assertFalse((self.root / "rife" / "flownet_v4.9.pkl").exists())
        self.assertTrue((self.root / "rife" / "archive.zip").is_file())  # sources stay for the next attempt
        self.assertFalse(model_manager.installed(self.model))

    def test_a_member_of_another_size_is_refused_before_writing(self):
        archive = self.root / "a.zip"
        archive.write_bytes(ARCHIVE)
        with self.assertRaisesRegex(downloader.DownloadError, "catalog expects"):
            downloader.extract_member(archive, self.root / "out.pkl", member="train_log/flownet.pkl",
                                      expected_size=len(MEMBER) + 1, expected_sha256=sha(MEMBER))
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["a.zip"])
        with self.assertRaisesRegex(downloader.DownloadError, "has no"):
            downloader.extract_member(archive, self.root / "out.pkl", member="../flownet.pkl",
                                      expected_size=len(MEMBER), expected_sha256=sha(MEMBER))

    def test_the_free_space_check_counts_the_part_file_the_step_overwrites(self):
        """A cancelled or failed run leaves <target>.part; the next run rewrites it from its first byte, so its bytes
        are room for the result, and a refusal names it."""
        folder = self.root / "rife"
        folder.mkdir()
        part = folder / "flownet_v4.9.pkl.part"
        part.write_bytes(b"x" * 3000)
        size = len(MEMBER)

        def free(bytes_free):
            return patch.object(model_manager.shutil, "disk_usage", return_value=SimpleNamespace(free=bytes_free))

        with free(size - 3001), self.assertRaisesRegex(downloader.DownloadError,
                                                       r"needs .* counting .* flownet_v4\.9\.pkl\.part"):
            model_manager.download_model("video")
        self.assertEqual(part.read_bytes(), b"x" * 3000)  # refused before the step ran
        with free(size - 3000):
            model_manager.download_model("video")
        self.assertEqual((folder / "flownet_v4.9.pkl").read_bytes(), MEMBER)
        self.assertFalse(part.exists())
        (folder / "flownet_v4.9.pkl").unlink()
        with free(size - 1), self.assertRaisesRegex(downloader.DownloadError, r"free disk space\.$"):
            model_manager.download_model("video")

    def test_an_unknown_step_is_refused(self):
        self.spec["convert"]["tool"] = "os.remove"
        with self.assertRaisesRegex(downloader.DownloadError, "unknown install step"):
            model_manager.download_model("video")
        self.assertEqual(_Server.requests, [])

    def test_the_settings_and_the_size_math_name_the_official_archive(self):
        from app import ui_settings

        url = self.base + "/archive.zip?export=download"
        self.assertEqual(model_manager.sources(self.spec, self.model)[0]["url"], url)
        self.assertIn("127.0.0.1", ui_settings._sources(self.model)[url])
        self.assertEqual(model_manager.download_bytes(["video"]), len(ARCHIVE))
        self.assertEqual(model_manager.disk_bytes(["video"]), len(MEMBER))
        self.assertEqual(model_manager.missing_bytes(["video"]), (len(ARCHIVE), len(MEMBER) + len(ARCHIVE)))


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("torch", "safetensors")), "requires torch")
class OfficialShardsTests(unittest.TestCase):
    """The Z-Image merge step through the installer: official shards from a Hugging Face revision (served locally),
    merged into one bf16 file whose size and SHA-256 the catalog pins; the shards are deleted afterwards."""

    @classmethod
    def setUpClass(cls):
        import torch
        from safetensors.torch import save_file

        cls.base = _Server.start()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            generator = torch.Generator().manual_seed(0)
            shards = {"model-00001-of-00002.safetensors": {"a.weight": torch.randn(4, 3, generator=generator)},
                      "model-00002-of-00002.safetensors": {"b.weight": torch.randn(5, generator=generator)}}
            weight_map = {}
            for name, tensors in shards.items():
                save_file(tensors, str(folder / name))
                weight_map.update({key: name for key in tensors})
            (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
            cls.sources = {p.name: p.read_bytes() for p in folder.iterdir()}
            from engine.models.zimage.checkpoint import merge

            merge(folder / "model.safetensors.index.json", folder / "merged.safetensors")
            cls.merged = (folder / "merged.safetensors").read_bytes()
        for name, data in cls.sources.items():
            _Server.files[f"/official/model/text_encoder/{name}"] = data

    @classmethod
    def tearDownClass(cls):
        _Server.stop()

    def test_shards_are_downloaded_at_the_pinned_revision_merged_and_removed(self):
        spec = {"repo_id": "official/model", "filename": "text_encoder/model.safetensors.index.json",
                "save_as": "encoder-bf16.safetensors", "dest": "encoder", "revision": "a" * 40,
                "size_bytes": len(self.merged), "sha256": sha(self.merged),
                "convert": {"tool": "engine.models.zimage.checkpoint.merge", "source_dest": "encoder_source",
                            "input": "model.safetensors.index.json",
                            "sources": [{"filename": f"text_encoder/{name}", "size_bytes": len(data),
                                         "sha256": sha(data)} for name, data in sorted(self.sources.items())],
                            "delete_sources_after": True}}
        model = {"id": "image", "name": "Image — test", "repo_id": "official/model", "files": [spec]}
        resolved = []

        def resolve(repo_id, filename, revision="main"):
            resolved.append((repo_id, filename, revision))
            return f"{self.base}/{repo_id}/{filename}"

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(config, "destination_dir", lambda dest: Path(folder) / dest), \
                patch.dict(config.CATALOG, {"models": [model]}), \
                patch.object(downloader, "_resolve_url", side_effect=resolve):
            model_manager.download_model("image")
            self.assertEqual((Path(folder) / "encoder" / "encoder-bf16.safetensors").read_bytes(), self.merged)
            self.assertEqual(list((Path(folder) / "encoder_source").iterdir()), [])
            self.assertTrue(model_manager.installed(model))
        self.assertEqual({revision for _, _, revision in resolved}, {"a" * 40})
        self.assertEqual(len(resolved), 3)


if __name__ == "__main__":
    unittest.main()
