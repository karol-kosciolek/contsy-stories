# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""MuseTalk 1.5 on the CPU: our code against fixtures from the reference code (tools/musetalk_parity.py
--make-fixture) and against diffusers, plus the pipeline's own guarantees (no dropped or shifted frames,
untouched pixels outside the mask, cancellation, several speakers)."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("fixtures") / "engine_musetalk.json"
HAVE = all(importlib.util.find_spec(package) is not None
           for package in ("torch", "diffusers", "cv2", "safetensors", "gguf"))
MODELS = Path(os.environ.get("CONTSY_MODELS_DIR", ROOT / "models"))
UNET_FILE = MODELS / "musetalk" / "unet.pth"
TENSOR_COUNT = 686  # tensors in musetalkV15/unet.pth (all float32, 849,947,844 parameters)


def checkpoint_fingerprint(path):
    """Sorted (name, shape) of the tensors in a checkpoint, read through the memory map (no tensor data)."""
    import gc

    import torch

    checkpoint = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
    try:
        return sorted((name, tuple(t.shape)) for name, t in checkpoint.items())
    finally:
        checkpoint = None
        gc.collect()


def expected_fingerprint(config=None):
    """Sorted (name, shape) of every parameter and buffer the network of `config` (default MuseTalk 1.5) needs."""
    import torch
    from engine.models.musetalk import unet

    with torch.device("meta"):
        return sorted((name, tuple(t.shape)) for name, t in unet.build_unet(config).state_dict().items())


def _models(state_unet, state_whisper, vae=None):
    from engine.pipelines import musetalk as pipeline
    from engine.models.musetalk import unet
    from tools import musetalk_parity as ref

    network = unet.build_unet(unet.unet_config(ref.TINY_UNET_2D)).eval()
    network.load_state_dict(state_unet)
    encoder = ref.our_tiny_encoder(state_whisper)
    return pipeline.MuseTalkModels(lambda *_: network, lambda *_: encoder, lambda *_: vae or ref.TinyVAE())


@unittest.skipUnless(HAVE and FIXTURE.is_file(), "MuseTalk dependencies or the fixture are missing")
class MuseTalkReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from engine.models.musetalk import audio, faces, unet
        from engine.pipelines import musetalk as pipeline
        from tools import musetalk_parity as ref
        from tests.seeded import seeded_state_dict

        cls.np, cls.torch, cls.audio, cls.faces, cls.unet, cls.pipeline, cls.ref = (
            np, torch, audio, faces, unet, pipeline, ref)
        cls.data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.unet_state = seeded_state_dict(cls.data["unet_shapes"])
        cls.whisper_state = seeded_state_dict(cls.data["whisper_shapes"])
        # The fixture was made at ref.THREADS threads; float sums split by thread count, so our side runs
        # at the same count or a 1-level tolerance could tip over on another CPU.
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(ref.THREADS)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.threads)

    def test_unet_matches_diffusers_unet2d(self):
        from diffusers import UNet2DConditionModel

        torch, ref = self.torch, self.ref
        reference = UNet2DConditionModel(**{k: v for k, v in ref.TINY_UNET_2D.items() if not k.startswith("_")})
        reference.load_state_dict(self.unet_state)
        ours = self.unet.build_unet(self.unet.unet_config(ref.TINY_UNET_2D)).eval()
        ours.load_state_dict(self.unet_state, strict=True)
        generator = torch.Generator().manual_seed(3)
        latents = torch.randn((3, 8, 16, 16), generator=generator)
        audio = torch.randn((3, 30, 16), generator=generator)
        with torch.inference_mode():
            expected = reference.eval()(latents, torch.tensor([0]), encoder_hidden_states=audio).sample
            actual = self.unet.predict(ours, latents, audio)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_musetalk_config_builds_the_checkpoint_layout(self):
        fingerprint = dict(expected_fingerprint())
        self.assertEqual(len(fingerprint), TENSOR_COUNT)
        self.assertEqual(fingerprint["conv_in.weight"], (320, 8, 3, 3))
        self.assertEqual(fingerprint["down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight"], (320, 384))
        self.assertFalse(any("motion" in name for name in fingerprint))
        with self.assertRaisesRegex(ValueError, "class_embed_type"):
            self.unet.unet_config({**self.unet.MUSETALK_JSON, "class_embed_type": "timestep"})
        with self.assertRaisesRegex(ValueError, "block types"):
            self.unet.unet_config({**self.unet.MUSETALK_JSON, "down_block_types": ["SimpleCrossAttnDownBlock2D"]})

    def test_unet_loader_checks_every_tensor(self):
        torch, ref = self.torch, self.ref
        config = self.unet.unet_config(ref.TINY_UNET_2D)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unet.pth"
            torch.save(self.unet_state, path)
            model = self.unet.load_unet(path, dtype=torch.float32, config=config)
            self.assertEqual(checkpoint_fingerprint(path), expected_fingerprint(config))
            torch.testing.assert_close(model.conv_in.weight, self.unet_state["conv_in.weight"], atol=0, rtol=0)
            cases = {"missing": ({k: v for k, v in self.unet_state.items() if k != "conv_out.bias"}, KeyError),
                     "extra": ({**self.unet_state, "pe": torch.zeros(1, 50, 384)}, KeyError),
                     "shape": ({**self.unet_state, "conv_in.weight": torch.zeros(32, 13, 3, 3)}, ValueError)}
            for name, (state, error) in cases.items():
                with self.subTest(name):
                    torch.save(state, path)
                    with self.assertRaises(error):
                        self.unet.load_unet(path, dtype=torch.float32, config=config)
            torch.save({"state_dict": self.unet_state}, path)
            with self.assertRaisesRegex(ValueError, "state_dict"):
                self.unet.load_unet(path, dtype=torch.float32, config=config)

    def test_whisper_windows_match_get_whisper_chunk(self):
        np, ref = self.np, self.ref
        encoder = ref.our_tiny_encoder(self.whisper_state)
        for name, (seconds, fps) in ref.AUDIO_CASES.items():
            with self.subTest(name):
                item = self.data["audio"][name]
                windows = self.audio.windows(encoder, ref.waveform(seconds), fps, 10 ** 6)
                self.assertEqual(len(windows), item["count"])
                np.testing.assert_allclose(windows[item["frames"]].numpy(), ref.unpack(item["windows"]),
                                           atol=3e-6, rtol=3e-6)
        self.assertEqual(len(self.audio.windows(encoder, ref.waveform(1.0), 25, 7)), 7)

    def test_exact_audio_lengths_keep_their_last_frame(self):
        frame_count, rate = self.audio.frame_count, self.audio.SAMPLE_RATE
        # (n * 640 / 16000) * 25 is n - 1e-15 for 1145 of these n in floats (29, 57, 58, 113, ...)
        lost = [n for n in range(1, 20001) if frame_count(n * 640, 25) != n or frame_count(n * 640 - 1, 25) != n]
        self.assertEqual(lost, [])
        lost_30 = [n for n in range(1, 3001)
                   if frame_count(-(-n * rate // 30), 30) != n or frame_count(n * rate // 30, 30) != n]
        self.assertEqual(lost_30, [])
        # beyond the 1 ms tolerance a frame short is a frame short, and other lengths count as the reference
        self.assertEqual(frame_count(29 * 640 - 17, 25), 28)
        self.assertEqual([frame_count(4800, 25), frame_count(8000, 30), frame_count(496000, 25)], [7, 15, 775])
        encoder = self.ref.our_tiny_encoder(self.whisper_state)
        wave = self.np.resize(self.ref.waveform(0.5), 29 * 640)
        self.assertEqual(len(self.audio.windows(encoder, wave, 25, 10 ** 6)), 29)

    def test_hugging_face_whisper_file_loads_through_the_key_map(self):
        from safetensors.torch import save_file

        torch, ref = self.torch, self.ref
        expected = ref.our_tiny_encoder(self.whisper_state).state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file({k: v.contiguous() for k, v in self.whisper_state.items()}, str(path))
            encoder = self.audio.load_audio_encoder(path, "cpu", dims=ref.TINY_DIMS)
            for name, tensor in encoder.state_dict().items():
                torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)
            partial = {k: v.contiguous() for k, v in self.whisper_state.items() if k != "encoder.layer_norm.bias"}
            save_file(partial, str(path))
            with self.assertRaisesRegex(KeyError, "ln_post.bias"):
                self.audio.load_audio_encoder(path, "cpu", dims=ref.TINY_DIMS)
        with self.assertRaises(KeyError):
            self.audio.hf_encoder_name("encoder.layers.0.self_attn.rotary.weight")
        self.assertIsNone(self.audio.hf_encoder_name("decoder.layers.0.fc1.weight"))

    def test_audio_positions_match_reference(self):
        item = self.data["positions"]
        table = self.audio.positions()
        self.assertEqual(tuple(table.shape), (50, 384))
        self.np.testing.assert_array_equal(table[item["rows"]].numpy(), self.ref.unpack(item["values"]))

    def test_boxes_match_get_landmark_and_bbox(self):
        actual = [list(self.ref.reference_box68(points)) for points in self.ref.box_cases()]
        self.assertEqual(actual, self.data["boxes"])

    def test_whole_loop_matches_reference_inference(self):
        np, ref, pipeline = self.np, self.ref, self.pipeline
        self.assertEqual(self.torch.get_num_threads(), ref.THREADS)  # the fixture's thread count
        video = ref.frames(3)
        faces = [self.faces.Face(tuple(float(v) for v in box), np.zeros((3, 2))) for box in self.data["loop"]["boxes"]]
        settings = pipeline.Settings(**{**pipeline.REFERENCE.__dict__, "resolution": ref.RESOLUTION})
        events = []
        output = pipeline.musetalk(_models(self.unet_state, self.whisper_state), video,
                                   [pipeline.FaceTrack(ref.waveform(3 / ref.FPS + 0.01, seed=11), faces)],
                                   settings=settings, mask_fn=ref.fake_mask_fn, device="cpu", batch_size=2,
                                   stats=lambda *event: events.append(event))
        expected = ref.unpack(self.data["loop"]["frames"])
        self.assertTrue((expected != video).any())
        self.assertLessEqual(int(np.abs(output.astype(np.int16) - expected.astype(np.int16)).max()), 1)
        self.assertEqual([event[0] for event in events],
                         ["audio", "load", "crop", "vae_encode", "unet", "vae_decode", "blend"])


@unittest.skipUnless(HAVE and FIXTURE.is_file(), "MuseTalk dependencies or the fixture are missing")
class MuseTalkPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from engine.models.musetalk import faces
        from engine.pipelines import musetalk as pipeline
        from tools import musetalk_parity as ref
        from tests.seeded import seeded_state_dict

        cls.np, cls.torch, cls.faces, cls.pipeline, cls.ref = np, torch, faces, pipeline, ref
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.models = _models(seeded_state_dict(data["unet_shapes"]), seeded_state_dict(data["whisper_shapes"]))
        cls.settings = pipeline.Settings(resolution=ref.RESOLUTION, smooth=1, fill_gap=0, fade=0, min_face=8)

    def _face(self, x0, y0, x1, y1):
        np = self.np
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        region = self.faces._ellipse(cx, cy + 0.25 * (y1 - y0), 0.4 * (x1 - x0), 0.25 * (y1 - y0))
        hole = np.array([[cx - 3, cy - 2], [cx + 3, cy - 2], [cx, cy + 4]])
        return self.faces.Face((x0, y0, x1, y1), region, hole, 0.1)

    def _run(self, video, tracks, **kwargs):
        kwargs = {"settings": self.settings, "device": "cpu", "batch_size": 1, **kwargs}
        return self.pipeline.musetalk(self.models, video, tracks, **kwargs)

    def test_frame_without_face_passes_through_and_nothing_shifts(self):
        np, ref, pipeline = self.np, self.ref, self.pipeline
        video, wave = ref.frames(4), ref.waveform(4 / 25 + 0.01)
        face = self._face(20, 8, 70, 70)
        full = self._run(video, [pipeline.FaceTrack(wave, [face] * 4)])
        holed = self._run(video, [pipeline.FaceTrack(wave, [face, None, face, face])])
        np.testing.assert_array_equal(holed[1], video[1])
        np.testing.assert_array_equal(holed[[0, 2, 3]], full[[0, 2, 3]])
        self.assertTrue((full[1] != video[1]).any())

    def test_pixels_outside_the_mask_keep_their_values(self):
        np, ref, pipeline = self.np, self.ref, self.pipeline
        video, wave = ref.frames(2), ref.waveform(2 / 25 + 0.01)
        face = self._face(20, 8, 70, 66)
        output = self._run(video, [pipeline.FaceTrack(wave, [face] * 2)],
                           settings=pipeline.Settings(resolution=ref.RESOLUTION, min_face=8))
        x0, y0, x1, y1 = self.faces.crop_box(face, video.shape[1:3], pipeline.Settings().margin(face.box), True)
        changed = (output != video).any(axis=-1)
        self.assertTrue(changed.any())
        outside = np.ones(changed.shape, bool)
        # only the lower half of the box may change; the feather softens the middle row by its radius
        radius = self.faces.blur_kernel((x0, y0, x1, y1), pipeline.Settings().feather) // 2
        outside[:, (y0 + y1) // 2 - radius:y1, x0:x1] = False
        self.assertFalse(changed[outside].any())
        # with the inset, the box edges themselves keep their pixels: no outline of the generated face
        self.assertFalse(changed[:, y1 - 1, x0:x1].any() or changed[:, :, x0].any() or changed[:, :, x1 - 1].any())

    def test_frames_after_the_audio_pass_through(self):
        np, ref, pipeline = self.np, self.ref, self.pipeline
        video = ref.frames(4)
        face = self._face(20, 8, 70, 70)
        output = self._run(video, [pipeline.FaceTrack(ref.waveform(2 / 25 + 0.01), [face] * 4)])
        self.assertTrue((output[:2] != video[:2]).any())
        np.testing.assert_array_equal(output[2:], video[2:])

    def test_two_speakers_in_one_run_equal_two_runs(self):
        np, ref, pipeline = self.np, self.ref, self.pipeline
        video = np.concatenate((ref.frames(3), ref.frames(3)[:, :, ::-1]), axis=2)  # two faces side by side
        left, right = self._face(20, 8, 70, 70), self._face(118, 8, 168, 70)
        a, b = ref.waveform(3 / 25 + 0.01, seed=1), ref.waveform(3 / 25 + 0.01, seed=2)
        together = self._run(video, [pipeline.FaceTrack(a, [left] * 3), pipeline.FaceTrack(b, [right] * 3)],
                             batch_size=4)
        first = self._run(video, [pipeline.FaceTrack(a, [left] * 3)])
        second = self._run(first, [pipeline.FaceTrack(b, [right] * 3)])
        self.assertLessEqual(int(np.abs(together.astype(np.int16) - second.astype(np.int16)).max()), 1)
        self.assertTrue((together[:, :, 96:] != video[:, :, 96:]).any())

    def test_cancellation_stops_between_batches(self):
        from engine.core.cancel import Cancelled, CancelToken

        ref, pipeline = self.ref, self.pipeline
        token = CancelToken()
        messages = []

        def progress(_fraction, message):
            messages.append(message)
            if "frame 2/" in message:
                token.cancel()

        with self.assertRaises(Cancelled):
            self._run(ref.frames(3), [pipeline.FaceTrack(ref.waveform(0.13), [self._face(20, 8, 70, 70)] * 3)],
                      cancel=token, progress_cb=progress)
        self.assertIn("Lip sync: frame 1/3", messages)

    def test_lipsync_wrapper_detects_the_face_and_needs_enough_coverage(self):
        ref, pipeline = self.ref, self.pipeline
        face = self._face(20, 8, 70, 70)
        events = []
        video = ref.frames(3)
        output = pipeline.lipsync(self.models, video, ref.waveform(0.13), lambda frame: face, low_vram=True,
                                  progress_cb=lambda *_: None, cancel=None, device="cpu",
                                  settings=self.settings, stats=lambda *event: events.append(event[0]))
        self.assertEqual(output.shape, video.shape)
        self.assertEqual(events[:2], ["detect", "audio"])
        # no face at all: it fails (the app then retries on a crop of the face) before any model is loaded;
        # with min_coverage=0 the frames pass through
        loads = []

        def refuse(*_args):
            loads.append(1)
            raise AssertionError("no model may load")

        idle = pipeline.MuseTalkModels(refuse, refuse, refuse)
        reports = []
        with self.assertRaises(pipeline.CoverageError) as caught:
            pipeline.lipsync(idle, video, ref.waveform(0.13), lambda frame: None, device="cpu",
                             report_cb=reports.append)
        self.assertIn("covered 0 of 3 speech frames of the speaker (no face found in 3)", str(caught.exception))
        self.assertEqual((reports[0].tracks[0].speech, reports[0].tracks[0].synced), (3, 0))
        self.assertIs(caught.exception.report, reports[0])
        unchanged = pipeline.lipsync(idle, video, ref.waveform(0.13), lambda frame: None, device="cpu",
                                     min_coverage=0)
        self.np.testing.assert_array_equal(unchanged, video)
        self.assertEqual(loads, [])

    def test_coverage_is_reported_with_reasons_and_enforced(self):
        np, ref, pipeline, faces = self.np, self.ref, self.pipeline, self.faces
        video, wave = ref.frames(4), ref.waveform(4 / 25 + 0.01)
        face = self._face(20, 8, 70, 70)
        small = faces.Face((20, 8, 25, 14), face.region, face.hole, 0.0)
        turned = faces.Face(face.box, face.region, face.hole, 0.95)
        reports = []
        output = self._run(video, [pipeline.FaceTrack(wave, [face, None, small, turned])], report_cb=reports.append)
        track = reports[0].tracks[0]
        self.assertEqual((track.frames, track.speech, track.generated, track.synced, track.no_face, track.small,
                          track.turned, track.invalid), (4, 4, 1, 1, 1, 1, 1, 0))
        self.assertAlmostEqual(track.coverage, 0.25)
        np.testing.assert_array_equal(output[1:], video[1:])
        with self.assertRaises(pipeline.CoverageError) as caught:
            self._run(video, [pipeline.FaceTrack(wave, [face, None, small, turned])], min_coverage=0.5)
        self.assertEqual(str(caught.exception), "Lip sync covered 1 of 4 speech frames of the speaker (no face found "
                         "in 1, face too small in 1, face turned away in 1); at least 50% are needed.")
        # a silent speaker without a face has nothing to lip-sync: no failure, and only the speech counts
        silent = np.zeros_like(wave)
        two = [pipeline.FaceTrack(wave, [face] * 4), pipeline.FaceTrack(silent, [None] * 4)]
        self._run(video, two, min_coverage=1.0, report_cb=reports.append)
        self.assertEqual([(t.speech, t.synced) for t in reports[-1].tracks], [(4, 4), (0, 0)])
        quiet = wave.copy()
        quiet[:2 * 640] = 0  # this speaker is silent for the first two frames
        self.assertEqual(self.pipeline.speech.speech_frames(quiet, 25, 4).tolist(), [False, False, True, True])
        with self.assertRaisesRegex(ValueError, "min_coverage"):
            self._run(video, two, min_coverage=1.5)

    def test_nonfinite_faces_are_never_pasted(self):
        from types import SimpleNamespace

        np, ref, pipeline = self.np, self.ref, self.pipeline

        class OverflowVAE(ref.TinyVAE):
            """The first crop of every batch encodes to NaN, as an fp16 overflow would."""

            def encode(self, pixels):
                latent = super().encode(pixels).latent_dist.mode().clone()
                latent[0] = float("nan")
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent, sample=lambda *_: latent))

        models =pipeline.MuseTalkModels(self.models.load_unet, self.models.load_audio, lambda *_: OverflowVAE())
        video, face = ref.frames(3), self._face(20, 8, 70, 70)
        track = pipeline.FaceTrack(ref.waveform(3 / 25 + 0.01), [face] * 3)
        reports = []
        output = pipeline.musetalk(models, video, [track], settings=self.settings, device="cpu", batch_size=2,
                                   report_cb=reports.append)
        np.testing.assert_array_equal(output[[0, 2]], video[[0, 2]])  # never a black mouth
        self.assertTrue((output[1] != video[1]).any())
        report = reports[0]
        self.assertEqual((report.nonfinite_latents, report.nonfinite_faces), (2, 2))
        self.assertEqual((report.tracks[0].generated, report.tracks[0].synced, report.tracks[0].invalid), (1, 1, 2))
        with self.assertRaisesRegex(pipeline.CoverageError, r"1 of 3 .*invalid network output in 2"):
            pipeline.musetalk(models, video, [track], settings=self.settings, device="cpu", batch_size=2,
                              min_coverage=0.5)

    def test_default_blend_reaches_the_chin(self):
        np, faces, pipeline = self.np, self.faces, self.pipeline
        for scale in (0.5, 1.0, 1.6, 3.0):
            with self.subTest(face_height=80 * scale):
                points = np.full((478, 2), 100.0)
                points[faces.JAW, 0] = np.linspace(60, 140, len(faces.JAW))
                points[faces.JAW, 1] = 100 + 40 * np.sin(np.linspace(0, np.pi, len(faces.JAW)))
                points[list(faces.BRIDGE), 1] = (95, 105)
                points[list(faces.CHEEKS), 0] = (60, 140)
                face = faces.face_from_landmarks(points * scale)
                settings = pipeline.DEFAULT
                margin = settings.margin(face.box)
                box = faces.crop_box(face, (int(300 * scale), int(300 * scale)), margin, settings.pad_outside)
                weights = faces.feather(faces.lower_face_mask(face, box), box, settings.feather, settings.inset)
                chin, column = round(140 * scale) - box[1], round(100 * scale) - box[0]
                # the chin itself is replaced (a lower jaw can then drop below the original one) ...
                self.assertGreaterEqual(weights[chin, column], 0.9)
                self.assertGreater(weights[chin + round(faces.JAW_SPREAD[1] * (box[3] - box[1])) // 2, column], 0.5)
                # ... while the weights still reach zero at the box edges, and the margin stays within
                # what MuseTalk saw in training (0-20 px on faces of about 200 px)
                self.assertEqual(float(weights[-1].max() + weights[:, 0].max() + weights[:, -1].max()), 0.0)
                self.assertLessEqual(margin, 0.13 * (face.box[3] - face.box[1]))
        self.assertEqual(pipeline.REFERENCE.margin(face.box), 10)

    def test_every_stage_reports_its_own_peak(self):
        from unittest import mock

        torch, pipeline = self.torch, self.pipeline
        peaks = iter([5, 3, 7, 2])  # allocated peak at the end of a, b, a, b; the counter is reset in between
        resets = []
        events = []
        cuda = torch.cuda
        with mock.patch.object(cuda, "synchronize", lambda *_: None), \
                mock.patch.object(cuda, "reset_peak_memory_stats", lambda *_: resets.append(1)), \
                mock.patch.object(cuda, "max_memory_allocated", lambda *_: next(peaks)):
            clock = pipeline._Clock(torch.device("cuda"), lambda *event: events.append(event))
            for name in ("a", "b", "a", "b"):
                with clock(name):
                    pass
            clock.report("a", "b")
        self.assertEqual(len(resets), 4)
        self.assertEqual([(name, peak) for name, _seconds, peak in events], [("a", 7), ("b", 3)])

    def test_last_frame_of_an_exact_length_clip_is_synced(self):
        pipeline = self.pipeline
        face = self._face(20, 8, 70, 70)
        count = pipeline.speech.frame_count(29 * 640, 25)
        plan = pipeline.plan_track([face] * 29, (80, 96), count, pipeline.Settings(min_face=8))
        self.assertEqual([step[2] for step in plan[-4:]], [1.0] * 4)

    def test_turned_small_and_missing_faces_fade(self):
        np, faces = self.np, self.faces
        face = self._face(20, 8, 70, 70)
        turned = faces.Face(face.box, face.region, face.hole, 0.95)
        small = faces.Face((20, 8, 30, 20), face.region, face.hole, 0.0)
        kept = faces.usable([face, turned, small, None], min_face=32, max_turn=0.8)
        self.assertEqual([f is not None for f in kept], [True, False, False, False])
        np.testing.assert_allclose(faces.strengths([True] * 6 + [False] + [True] * 3, 3),
                                   [1, 1, 1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75])
        np.testing.assert_array_equal(faces.strengths([True, True], 3), [1, 1])

    def test_gaps_are_bridged_and_boxes_smoothed(self):
        faces = self.faces
        a, b = self._face(20, 8, 70, 70), self._face(30, 8, 80, 70)
        filled = faces.fill_gaps([a, None, None, b, None, None, None, None, a], 3)
        self.assertAlmostEqual(filled[1].box[0], 20 + 10 / 3)
        self.assertAlmostEqual(filled[2].box[0], 20 + 20 / 3)
        self.assertTrue(all(f is None for f in filled[4:8]))
        jitter = [self._face(20 + (2 if k % 2 else -2), 8, 70, 70) for k in range(7)]
        smoothed = faces.smooth(jitter, 5)
        self.assertLess(abs(smoothed[3].box[0] - 20), 0.5)
        self.assertEqual(faces.smooth(jitter, 1)[3].box, jitter[3].box)
        yunet_like = faces.Face(a.box, faces._ellipse(40, 50, 10, 8), None, 0.0)
        self.assertIs(faces.fill_gaps([a, None, yunet_like], 3)[1], None)  # incompatible polygons

    def test_gaps_and_smoothing_never_join_two_faces(self):
        faces = self.faces
        a, b = self._face(100, 10, 180, 90), self._face(600, 10, 680, 90)
        self.assertEqual([f is None for f in faces.fill_gaps([a, None, None, b], 3)], [False, True, True, False])
        self.assertEqual([f.box[0] for f in faces.smooth([a, a, b, a, a], 5)], [100, 100, 600, 100, 100])
        # a head that moves still is one face
        moving = [self._face(100 + 6 * k, 10, 180 + 6 * k, 90) for k in range(5)]
        self.assertAlmostEqual(faces.smooth(moving, 5)[2].box[0], 112)
        self.assertAlmostEqual(faces.fill_gaps([moving[0], None, None, moving[3]], 3)[1].box[0], 106)

    def test_tracks_follow_their_own_face(self):
        faces = self.faces
        left = [self._face(10 + k, 10, 50 + k, 60) for k in range(4)]
        right = [self._face(100 - k, 12, 140 - k, 62) for k in range(4)]
        per_frame = [[right[0], left[0]], [left[1], right[1]], [right[2]], [right[3], left[3]]]
        tracks = faces.follow_tracks(per_frame, [(0, 0, 70, 80), (90, 0, 150, 80)])
        self.assertEqual(tracks[0], [left[0], left[1], None, left[3]])
        self.assertEqual(tracks[1], right)

    def test_face_geometry_from_landmarks_and_yunet(self):
        np, faces = self.np, self.faces
        points = np.zeros((478, 2))
        points[:, 0], points[:, 1] = 100, 100
        jaw_x = np.linspace(60, 140, len(faces.JAW))
        points[faces.JAW, 0] = jaw_x
        points[faces.JAW, 1] = 100 + 40 * np.sin(np.linspace(0, np.pi, len(faces.JAW)))
        points[list(faces.BRIDGE), 1] = (95, 105)
        points[list(faces.CHEEKS), 0] = (60, 140)
        points[faces.NOSE_TIP, 0] = 100
        face = faces.face_from_landmarks(points)
        self.assertEqual(face.box, (60.0, 60.0, 140.0, 140.0))
        self.assertAlmostEqual(face.turn, 0.0)
        self.assertEqual(face.hole.shape, (len(faces.NOSE), 2))
        row = np.array([50, 40, 60, 80, 65, 70, 95, 70, 80, 90, 68, 105, 92, 105, 0.9])
        found = faces.face_from_yunet(row)
        self.assertAlmostEqual(found.box[1], 90 - 0.2 * 35 - (120 - (90 - 0.2 * 35)))
        self.assertEqual(found.box[2:], (110.0, 120.0))
        self.assertAlmostEqual(found.turn, 0.0)
        self.assertIsNone(found.hole)
        self.assertIsNone(faces.face_from_landmarks(np.zeros((478, 2))))

    def test_crop_repeats_edges_outside_the_frame(self):
        np, faces = self.np, self.faces
        frame = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
        face = faces.Face((-4.0, -6.0, 12.0, 10.0), np.zeros((3, 2)))
        self.assertEqual(faces.crop_box(face, frame.shape, 0, pad_outside=False), (0, 0, 12, 10))
        box = faces.crop_box(face, frame.shape, 2, pad_outside=True)
        self.assertEqual(box, (-4, -6, 12, 12))
        crop = faces.crop(frame, box, 16)
        self.assertEqual(crop.shape, (16, 16, 3))
        np.testing.assert_array_equal(faces.crop(frame, (0, 0, 16, 16), 16), frame[:16, :16])


@unittest.skipUnless(HAVE and UNET_FILE.is_file(), "the MuseTalk 1.5 UNet is not installed (CONTSY_MODELS_DIR)")
class MuseTalkRealWeightsTests(unittest.TestCase):
    def test_checkpoint_is_safe_to_load_and_matches_our_network(self):
        # weights_only=True and the memory map: only the index is read, no tensor data
        self.assertEqual(checkpoint_fingerprint(UNET_FILE), expected_fingerprint())


if __name__ == "__main__":
    unittest.main()
