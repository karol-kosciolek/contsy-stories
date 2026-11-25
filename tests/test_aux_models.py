# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Auxiliary downloads must verify bytes before exposing model files to workers."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from app import aux_models, downloader
from app.downloader import DownloadError


class AuxiliaryModelTests(unittest.TestCase):
    """ensure_landmarker goes through the same verified, resumable download as the catalog's plain-address files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = Path(self.tmp.name) / 'models' / 'mediapipe' / 'face_landmarker.task'
        self.data = b'verified face model'
        self.spec = {'url': 'https://storage.googleapis.com/pinned/model.task',
                     'size_bytes': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        for target in (patch.object(aux_models, 'ROOT', Path(self.tmp.name)),
                       patch.dict(aux_models.CATALOG, {'landmarker': self.spec})):
            target.start()
            self.addCleanup(target.stop)

    def response(self, data):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.headers = {}
        response.iter_content.return_value = [data]
        return response

    def test_existing_valid_model_requires_no_network(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(self.data)
        with patch.object(downloader.requests, 'get') as get:
            self.assertEqual(aux_models.ensure_landmarker(), self.target)
        get.assert_not_called()

    def test_download_is_pinned_to_the_catalog_address_without_a_token(self):
        with patch.object(downloader.requests, 'get', return_value=self.response(self.data)) as get:
            self.assertEqual(aux_models.ensure_landmarker(), self.target)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertEqual(get.call_args.args[0], self.spec['url'])
        self.assertNotIn('Authorization', get.call_args.kwargs['headers'])

    def test_existing_corruption_is_preserved_before_repair(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b'x' * len(self.data))
        with patch.object(downloader.requests, 'get', return_value=self.response(self.data)):
            aux_models.ensure_landmarker()
        self.assertEqual(self.target.read_bytes(), self.data)
        saved = list(self.target.parent.glob('face_landmarker.task.invalid_*'))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_bytes(), b'x' * len(self.data))

    def test_bad_remote_hash_never_publishes(self):
        with patch.object(downloader.requests, 'get', return_value=self.response(b'x' * len(self.data))):
            with self.assertRaisesRegex(DownloadError, 'Integrity'):
                aux_models.ensure_landmarker()
        self.assertFalse(self.target.exists())

    def test_excess_download_is_stopped_before_publication(self):
        with patch.object(downloader.requests, 'get', return_value=self.response(self.data + b'extra')):
            with self.assertRaisesRegex(DownloadError, 'exceeded'):
                aux_models.ensure_landmarker()
        self.assertFalse(self.target.exists())

class AuxiliaryCatalogTests(unittest.TestCase):
    def test_the_catalog_pins_only_the_face_landmarker(self):
        """Lip sync takes each face's gender from the recognised character's profile: no model guesses it."""
        self.assertEqual(set(aux_models.CATALOG), {'landmarker'})
        spec = aux_models.CATALOG['landmarker']
        self.assertEqual((len(spec['sha256']), spec['license']), (64, 'Apache-2.0'))
        self.assertGreater(spec['size_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
