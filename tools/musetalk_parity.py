# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""MuseTalk 1.5 against the reference code (TMElyralab/MuseTalk, MIT), on the CPU.

`--make-fixture` writes tests/fixtures/engine_musetalk.json from a code-only checkout of the reference
(`--reference`, commit 0a89dec) with tiny seeded networks: the Whisper windows of `get_whisper_chunk`
(transformers' WhisperModel and WhisperFeatureExtractor), the audio position table, the crop boxes of
`get_landmark_and_bbox`, and the whole inference loop of `scripts/inference.py` (VAE wrapper, `datagen`, UNet,
`decode_latents`, `blending.get_image`) on three synthetic frames. The reference's helper models are not
used: DWPose and S3FD are stubbed with fixed 68-point sets and boxes, BiSeNet with a fixed mask.

`--real` compares our code with the reference classes on the real weights (`--models`, the installed
models folder) and prints max abs error and PSNR: the UNet against diffusers' UNet2DConditionModel built
from musetalk.json, Whisper tiny against transformers' WhisperModel through the reference's audio code, and
the whole loop on a real face (`--face`) with identical boxes and mask on both sides. CUDA is hidden.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import math
import os
import sys
import tempfile
import time
import zlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "engine_musetalk.json"

TINY_UNET_2D = {
    "_class_name": "UNet2DConditionModel", "act_fn": "silu", "attention_head_dim": 2, "block_out_channels": [32, 64],
    "center_input_sample": False, "cross_attention_dim": 16,
    "down_block_types": ["CrossAttnDownBlock2D", "DownBlock2D"],
    "downsample_padding": 1, "flip_sin_to_cos": True, "freq_shift": 0, "in_channels": 8, "layers_per_block": 1,
    "mid_block_scale_factor": 1, "norm_eps": 1e-5, "norm_num_groups": 8, "out_channels": 4, "sample_size": 8,
    "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D"],
}
TINY_WHISPER = {"num_mel_bins": 80, "d_model": 16, "encoder_layers": 2, "encoder_attention_heads": 2,
                "encoder_ffn_dim": 64, "decoder_layers": 1, "decoder_attention_heads": 2, "decoder_ffn_dim": 64,
                "max_source_positions": 1500, "vocab_size": 64, "max_target_positions": 8,
                "pad_token_id": 1, "bos_token_id": 2, "eos_token_id": 3, "decoder_start_token_id": 2}
TINY_DIMS = (80, 1500, 16, 2, 2)  # our AudioEncoder for TINY_WHISPER
RESOLUTION = 32
FPS = 25
FRAME_SHAPE = (80, 96)
AUDIO_CASES = {"short_25": (0.3, 25), "half_30": (0.5, 30), "long_25": (31.0, 25)}
#: CPU threads for making the fixture and for the tests that compare with it: float sums split by thread
#: count, so the reference's own frames move by 1 level between 8 and 20 threads.
THREADS = 8
LONG_FRAMES = [0, 1, 2, 748, 749, 750, 751, 774]
POSITION_ROWS = [0, 1, 2, 25, 49]


# ----------------------------------------------------------------------------------------------
# shared inputs (the tests import these)
# ----------------------------------------------------------------------------------------------

def pack(array) -> dict:
    array = np.ascontiguousarray(array)
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "data": base64.b64encode(zlib.compress(array.tobytes(), 9)).decode("ascii")}


def unpack(item: dict) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(item["data"]))
    return np.frombuffer(raw, dtype=np.dtype(item["dtype"])).reshape(item["shape"]).copy()


def waveform(seconds: float, seed: int = 318) -> np.ndarray:
    """Deterministic speech-like noise: a few tones with a moving envelope plus a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * 16000))) / 16000
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t) ** 2
    tones = sum(np.sin(2 * np.pi * f * t + p) for f, p in ((180, 0.3), (720, 1.1), (2300, 2.0)))
    return (0.08 * envelope * tones + 0.01 * rng.standard_normal(t.shape)).astype(np.float32)


def frames(count: int = 3) -> np.ndarray:
    """Smooth synthetic RGB frames (they compress well and stay within the VAE's comfortable range)."""
    h, w = FRAME_SHAPE
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    out = []
    for k in range(count):
        r = 120 + 80 * np.sin(x / 9.0 + k) * np.cos(y / 13.0)
        g = 110 + 70 * np.cos(x / 15.0 - 0.5 * k) + 30 * np.sin(y / 7.0)
        b = 90 + 60 * np.sin((x + y) / 11.0 + 0.3 * k)
        out.append(np.stack((r, g, b), axis=-1))
    return np.clip(np.rint(np.stack(out)), 0, 255).astype(np.uint8)


def landmarks68(index: int) -> np.ndarray:
    """A plausible 68-point face (int32, as the reference truncates DWPose's output) for frame `index`."""
    cx, cy, half = 46 + 2 * index, 38 + index, 24 - index
    jaw_angle = np.linspace(np.pi * 1.05, np.pi * -0.05, 17)  # ear to ear through the chin (image y points down)
    jaw = np.stack((cx - half * np.cos(jaw_angle), cy + 0.2 * half + 1.1 * half * np.sin(jaw_angle)), axis=1)
    brows = np.stack((np.linspace(cx - 0.8 * half, cx + 0.8 * half, 10), np.full(10, cy - 0.55 * half)), axis=1)
    nose = np.stack((np.full(9, cx + 0.0), np.concatenate((np.linspace(cy - 0.4 * half, cy + 0.2 * half, 4),
                                                          np.full(5, cy + 0.3 * half)))), axis=1)
    nose[4:, 0] = np.linspace(cx - 0.25 * half, cx + 0.25 * half, 5)
    eyes = np.stack((np.concatenate((np.linspace(cx - 0.6 * half, cx - 0.2 * half, 6),
                                     np.linspace(cx + 0.2 * half, cx + 0.6 * half, 6))),
                     np.full(12, cy - 0.3 * half)), axis=1)
    mouth = np.stack((cx + 0.45 * half * np.cos(np.linspace(0, 2 * np.pi, 20, endpoint=False)),
                      cy + 0.65 * half + 0.15 * half * np.sin(np.linspace(0, 2 * np.pi, 20, endpoint=False))), axis=1)
    return np.concatenate((jaw, brows, nose, eyes, mouth)).astype(np.int32)


#: 68-point sets for the box test: the three frames, one whose box would start above the image (clamped to
#: row 0) and one with a lower bridge point.
def box_cases():
    cases = [landmarks68(k) for k in range(3)]
    high = landmarks68(0).copy()
    high[:, 1] -= 30
    low = landmarks68(1).copy()
    low[29, 1] += 6
    return cases + [high, low]


def reference_box68(points: np.ndarray):
    """Our box arithmetic on 68 points: the jaw extremes from all points, bridge point 29."""
    from engine.models.musetalk.faces import reference_box

    box = reference_box(points[:, 0].min(), points[:, 0].max(), points[:, 1].max(), points[29, 1], clamp_top=True)
    return tuple(int(v) for v in box)


def fake_parse(side: int) -> np.ndarray:
    """The stand-in for BiSeNet: a fixed ellipse over the blend square (0/255, side x side)."""
    import cv2

    mask = np.zeros((side, side), np.uint8)
    cv2.ellipse(mask, (side // 2, int(side * 0.62)), (int(side * 0.3), int(side * 0.22)), 0, 0, 360, 255, -1)
    return mask


def fake_mask_fn(_face, box):
    """`fake_parse` cut to the box, the way `blending.get_image` crops the parsed square."""
    from engine.models.musetalk.faces import blend_square

    x0, y0, x1, y1 = box
    s = blend_square(box)
    xs, ys = (x0 + x1) // 2 - s, (y0 + y1) // 2 - s
    return fake_parse(2 * s)[y0 - ys:y1 - ys, x0 - xs:x1 - xs].copy()


class TinyVAE:
    """Deterministic stand-in for AutoencoderKL: average pooling down, bilinear up (diffusers' interface).

    `sample()` returns the mean as well, so the reference's random sample and our `mode()` agree."""
    config = SimpleNamespace(scaling_factor=0.18215, block_out_channels=(4, 4, 4, 4), latent_channels=4)
    dtype = None
    device = None

    def __init__(self):
        import torch

        self.dtype, self.device = torch.float32, torch.device("cpu")

    def encode(self, pixels):
        from torch.nn import functional as F
        import torch

        average = F.avg_pool2d(pixels, 8)
        latent = torch.cat((average, average.mean(dim=1, keepdim=True) * 0.5), dim=1)
        dist = SimpleNamespace(mode=lambda: latent, sample=lambda generator=None: latent)
        return SimpleNamespace(latent_dist=dist)

    def decode(self, latents):
        from torch.nn import functional as F

        sample = F.interpolate(latents[:, :3] * 1.3, scale_factor=8, mode="bilinear", align_corners=False)
        return SimpleNamespace(sample=sample.tanh())

    def to(self, *_args, **_kwargs):
        return self

    def half(self):
        return self

    def eval(self):
        return self


def tiny_whisper_state():
    """Seeded transformers WhisperModel (TINY_WHISPER) state dict and its shapes."""
    import torch
    from transformers import WhisperConfig, WhisperModel
    from tests.seeded import seeded_state_dict, shapes as _shapes

    with torch.device("cpu"):
        model = WhisperModel(WhisperConfig(**TINY_WHISPER)).eval()
    shapes = _shapes(model)
    state = seeded_state_dict(shapes)
    model.load_state_dict(state)
    return model, state, shapes


def our_tiny_encoder(state):
    """Our AudioEncoder with the HF-layout `state` (encoder tensors only, through the key map)."""
    import torch
    from engine.models.whisper import AudioEncoder
    from engine.models.musetalk.audio import hf_encoder_name

    encoder = AudioEncoder(*TINY_DIMS).eval()
    mapped = {}
    for name, tensor in state.items():
        ours = hf_encoder_name(name)
        if ours is not None:
            mapped[ours] = tensor
    encoder.load_state_dict(mapped, strict=True)
    return encoder.to(torch.float32)


# ----------------------------------------------------------------------------------------------
# the reference
# ----------------------------------------------------------------------------------------------

class _Landmarks:
    """Feeds the stubbed DWPose / S3FD with fixed answers, in call order."""
    points: list = []
    boxes: list = []


def _stub_reference_helpers():
    """Modules the reference imports at load time that cannot (mmpose) or must not (S3FD) run here."""
    face_detection = ModuleType("face_detection")
    face_detection.LandmarksType = SimpleNamespace(_2D=2)

    class FaceAlignment:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_detections_for_batch(self, images):
            return [_Landmarks.boxes.pop(0) for _ in images]

    face_detection.FaceAlignment = FaceAlignment
    mmpose = ModuleType("mmpose")
    apis = ModuleType("mmpose.apis")
    apis.init_model = lambda *_args, **_kwargs: object()

    def inference_topdown(_model, _image):
        keypoints = np.zeros((1, 133, 2), np.float32)
        keypoints[0, 23:91] = _Landmarks.points.pop(0)
        return [keypoints]

    apis.inference_topdown = inference_topdown
    structures = ModuleType("mmpose.structures")
    structures.merge_data_samples = lambda results: SimpleNamespace(
        pred_instances=SimpleNamespace(keypoints=results[0]))
    mmpose.apis, mmpose.structures = apis, structures
    for module in (face_detection, mmpose, apis, structures):
        sys.modules[module.__name__] = module


def _reference(path: Path):
    if not (path / "musetalk" / "utils" / "audio_processor.py").is_file():
        raise SystemExit(f"{path} is not a MuseTalk checkout (musetalk/utils/audio_processor.py is missing)")
    _stub_reference_helpers()
    sys.path.insert(0, str(path))
    from musetalk.models.unet import PositionalEncoding
    from musetalk.models.vae import VAE
    from musetalk.utils import blending, preprocessing
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.utils import datagen

    return SimpleNamespace(PositionalEncoding=PositionalEncoding, VAE=VAE, blending=blending,
                           preprocessing=preprocessing, AudioProcessor=AudioProcessor, datagen=datagen)


def _audio_processor(ref, directory: Path):
    from transformers import WhisperFeatureExtractor

    WhisperFeatureExtractor(feature_size=80).save_pretrained(str(directory))
    return ref.AudioProcessor(feature_extractor_path=str(directory))


def _reference_windows(processor, whisper, wave: np.ndarray, fps: int, directory: Path):
    import soundfile as sf
    import torch

    path = directory / f"audio_{len(wave)}.wav"
    sf.write(str(path), wave, 16000, subtype="FLOAT")
    features, length = processor.get_audio_feature(str(path))
    assert length == len(wave)
    return processor.get_whisper_chunk(features, torch.device("cpu"), torch.float32, whisper, length, fps=fps)


def _check_hidden_states(whisper):
    """transformers 5.x records hidden states through hooks; the last one must still follow the final norm."""
    import torch

    mel = torch.randn((1, 80, 3000), generator=torch.Generator().manual_seed(5))
    out = whisper.encoder(mel, output_hidden_states=True)
    assert len(out.hidden_states) == whisper.config.encoder_layers + 1
    torch.testing.assert_close(out.hidden_states[-1], out.last_hidden_state)


def _reference_loop(ref, unet, vae_model, whisper, processor, video: np.ndarray, wave: np.ndarray,
                    points: list, directory: Path, resolution: int, fp, mode="jaw"):
    """scripts/inference.py's loop body (v15) on in-memory frames with the stubbed helpers."""
    import cv2
    import torch

    paths = []
    for k, frame in enumerate(video):
        path = directory / f"{k:08d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        paths.append(str(path))
    _Landmarks.points = [p.astype(np.float32) for p in points]
    _Landmarks.boxes = [(0, 0, video.shape[2], video.shape[1])] * len(points)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        coord_list, frame_list = ref.preprocessing.get_landmark_and_bbox(paths, 0)
    vae = ref.VAE.__new__(ref.VAE)
    vae.vae, vae.scaling_factor, vae._use_float16 = vae_model, vae_model.config.scaling_factor, False
    from torchvision import transforms

    vae.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    vae._resized_img = resolution
    vae._mask_tensor = vae.get_mask_tensor()
    if hasattr(vae_model, "encode") and not isinstance(vae_model, TinyVAE):
        # diffusers' posterior sample is random per frame; the parity run compares the mean on both sides
        def encode_latents(image):
            return vae.scaling_factor * vae.vae.encode(image.to(vae.vae.dtype)).latent_dist.mode()

        vae.encode_latents = encode_latents
    whisper_chunks = _reference_windows(processor, whisper, wave, FPS, directory)
    extra_margin = 10
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + extra_margin, frame.shape[0])
        crop_frame = cv2.resize(frame[y1:y2, x1:x2], (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
        input_latent_list.append(vae.get_latents_for_unet(crop_frame))
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    pe = ref.PositionalEncoding(d_model=whisper.config.d_model)
    timesteps = torch.tensor([0])
    res_frame_list = []
    for whisper_batch, latent_batch in ref.datagen(whisper_chunks, input_latent_list_cycle, 2, 0, "cpu"):
        pred = unet(latent_batch, timesteps, encoder_hidden_states=pe(whisper_batch)).sample
        res_frame_list.extend(vae.decode_latents(pred))
    out = []
    for i, res_frame in enumerate(res_frame_list):
        x1, y1, x2, y2 = coord_list_cycle[i % len(coord_list_cycle)]
        ori_frame = frame_list_cycle[i % len(frame_list_cycle)].copy()
        y2 = min(y2 + extra_margin, ori_frame.shape[0])
        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        combined = ref.blending.get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=mode, fp=fp)
        out.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    return np.stack(out), [tuple(int(v) for v in box) for box in coord_list]


def _fake_parser():
    from PIL import Image

    def parse(image, mode="raw"):
        del mode
        return Image.fromarray(fake_parse(image.size[0]))

    return parse


def make_fixture(reference: Path) -> dict:
    from diffusers import UNet2DConditionModel
    from tests.seeded import seeded_state_dict, shapes as _shapes

    ref = _reference(reference)
    whisper, _state, whisper_shapes = tiny_whisper_state()
    _check_hidden_states(whisper)
    unet = UNet2DConditionModel(**{k: v for k, v in TINY_UNET_2D.items() if not k.startswith("_")}).eval()
    unet_shapes = _shapes(unet)
    unet.load_state_dict(seeded_state_dict(unet_shapes))
    data = {"whisper_shapes": whisper_shapes, "unet_shapes": unet_shapes, "audio": {}}
    with tempfile.TemporaryDirectory(prefix="contsy-musetalk-parity-") as temporary:
        directory = Path(temporary)
        processor = _audio_processor(ref, directory)
        for name, (seconds, fps) in AUDIO_CASES.items():
            windows = _reference_windows(processor, whisper, waveform(seconds), fps, directory).float().numpy()
            picked = list(range(len(windows))) if name != "long_25" else LONG_FRAMES
            data["audio"][name] = {"count": len(windows), "frames": picked, "windows": pack(windows[picked])}
        data["positions"] = {"rows": POSITION_ROWS,
                             "values": pack(ref.PositionalEncoding(d_model=384).pe[0, POSITION_ROWS].numpy())}
        boxes = []
        for points in box_cases():
            _Landmarks.points = [points.astype(np.float32)]
            _Landmarks.boxes = [(0, 0, FRAME_SHAPE[1], FRAME_SHAPE[0])]
            path = directory / "box.png"
            import cv2

            cv2.imwrite(str(path), np.zeros((*FRAME_SHAPE, 3), np.uint8))
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                coords, _ = ref.preprocessing.get_landmark_and_bbox([str(path)], 0)
            boxes.append([int(v) for v in coords[0]])
        data["boxes"] = boxes
        video = frames(3)
        wave = waveform(3 / FPS + 0.01, seed=11)
        loop, coords = _reference_loop(ref, unet, TinyVAE(), whisper, processor, video, wave,
                                       [landmarks68(k) for k in range(3)], directory, RESOLUTION, _fake_parser())
        data["loop"] = {"frames": pack(loop), "boxes": [list(c) for c in coords]}
    return data


# ----------------------------------------------------------------------------------------------
# real weights
# ----------------------------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(peak ** 2 / mse)


def _real_unet(models: Path, report: dict):
    import torch
    from diffusers import UNet2DConditionModel
    from engine.models.musetalk import unet as ours

    path = models / "musetalk" / "unet.pth"
    config = json.loads((models / "musetalk" / "musetalk.json").read_text(encoding="utf-8"))
    start = time.perf_counter()
    network = ours.load_unet(path, "cpu", torch.float32)
    report["unet_load_seconds_cpu_float32"] = round(time.perf_counter() - start, 2)
    reference = UNet2DConditionModel(**{k: v for k, v in config.items() if not k.startswith("_")}).eval()
    reference.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True, mmap=True), strict=True)
    generator = torch.Generator().manual_seed(7)
    latents = torch.randn((2, 8, 32, 32), generator=generator)
    audio = torch.randn((2, 50, 384), generator=generator)
    with torch.inference_mode():
        expected = reference(latents, torch.tensor([0]), encoder_hidden_states=audio).sample
        actual = ours.predict(network, latents, audio)
    diff = (actual - expected).abs()
    report["unet"] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                      "reference_abs_max": float(expected.abs().max()),
                      "psnr_db_over_reference_range": psnr(actual.numpy(), expected.numpy(),
                                                           float(expected.max() - expected.min()))}
    return network, reference


def _catalog_folder(models: Path, role: str) -> Path:
    """The folder of a catalog destination (`lipsync_whisper`, `lipsync_vae`, `musetalk`) inside `models`."""
    from app.config import MODELS_DIR, destination_dir

    return models / destination_dir(role).relative_to(MODELS_DIR)


def _real_whisper(ref, models: Path, voice: Path, directory: Path, report: dict):
    import soundfile as sf
    import torch
    from transformers import WhisperConfig, WhisperModel
    from engine.models.musetalk import audio as ours

    encoder = ours.load_audio_encoder(_catalog_folder(models, "lipsync_whisper") / "tiny.pt", "cpu")
    # The same weights in transformers' layout: invert our key map over transformers' own key names.
    whisper = WhisperModel(WhisperConfig(num_mel_bins=80, d_model=384, encoder_layers=4, encoder_attention_heads=6,
                                         encoder_ffn_dim=1536, decoder_layers=4, decoder_attention_heads=6,
                                         decoder_ffn_dim=1536, max_source_positions=1500)).eval()
    state = whisper.state_dict()
    ours_state = encoder.state_dict()
    for name in list(state):
        mapped = ours.hf_encoder_name(name)
        if mapped is not None:
            state[name] = ours_state[mapped].clone()
    whisper.load_state_dict(state)
    _check_hidden_states(whisper)
    wave, rate = sf.read(str(voice), dtype="float32", always_2d=True)
    if rate != 16000:
        import librosa

        wave = librosa.resample(wave.mean(axis=1), orig_sr=rate, target_sr=16000)
    else:
        wave = wave.mean(axis=1)
    wave = np.ascontiguousarray(wave[:5 * 16000], dtype=np.float32)
    processor = _audio_processor(ref, directory)
    with torch.inference_mode():
        expected = _reference_windows(processor, whisper, wave, FPS, directory).float()
    actual = ours.windows(encoder, wave, FPS, 10 ** 6)
    diff = (actual - expected).abs()
    report["whisper"] = {"frames": len(actual), "reference_frames": len(expected), "max_abs": float(diff.max()),
                         "mean_abs": float(diff.mean()), "reference_abs_max": float(expected.abs().max())}
    return whisper, processor, wave


def _face_frames(face: Path, count: int) -> np.ndarray:
    """`count` frames of a real portrait scaled to 480 px high, drifting one pixel per frame."""
    from PIL import Image

    with Image.open(face) as picture:
        picture = picture.convert("RGB")
        scale = 480 / picture.height
        picture = picture.resize((round(picture.width * scale) // 2 * 2, 480), Image.LANCZOS)
    base = np.asarray(picture)
    width = min(base.shape[1] - count, 480) // 2 * 2
    out = []
    for k in range(count):
        x = (base.shape[1] - count - width) // 2 + k
        out.append(base[:, x:x + width])
    return np.stack(out)


def run_real(reference: Path, models: Path, face: Path, voice: Path, landmarker: Path, out: Path,
             frame_count: int = 8) -> dict:
    import torch
    from diffusers import AutoencoderKL
    from engine.models.musetalk import faces as geometry
    from engine.pipelines import musetalk as pipeline

    ref = _reference(reference)
    report: dict = {"torch": torch.__version__}
    network, reference_unet = _real_unet(models, report)
    print(json.dumps(report["unet"]))
    with tempfile.TemporaryDirectory(prefix="contsy-musetalk-real-") as temporary:
        directory = Path(temporary)
        whisper, processor, wave = _real_whisper(ref, models, voice, directory, report)
        print(json.dumps(report["whisper"]))
        video = _face_frames(face, frame_count)
        with geometry.MuseTalkFaceDetector(landmarker) as detector:
            found = [detector(frame) for frame in video]
        if any(f is None for f in found):
            raise SystemExit("MediaPipe found no face in some frames of the portrait")
        report["face_turn"] = [round(f.turn, 3) for f in found]
        # The same boxes on both sides: our MediaPipe box, written as a 68-point set whose extremes and
        # point 29 give exactly that box in the reference's arithmetic.
        points = []
        for f in found:
            x0, _top, x1, y1 = (int(round(v)) for v in f.box)
            center = int(round((f.box[1] + f.box[3]) / 2))
            p = np.tile(np.array([[(x0 + x1) // 2, center]], np.int32), (68, 1))
            p[0], p[16], p[8], p[29] = (x0, center), (x1, center), ((x0 + x1) // 2, y1), ((x0 + x1) // 2, center)
            points.append(p)
        vae_dir = _catalog_folder(models, "lipsync_vae")
        vae_reference = AutoencoderKL.from_pretrained(str(vae_dir), local_files_only=True, use_safetensors=True).eval()
        start = time.perf_counter()
        with torch.inference_mode():
            expected, coords = _reference_loop(ref, reference_unet, vae_reference, whisper, processor, video,
                                               wave[:int(16000 * (frame_count / FPS + 0.01))], points, directory,
                                               256, _fake_parser())
        report["reference_loop_seconds_cpu"] = round(time.perf_counter() - start, 1)
        faces = [geometry.Face(tuple(float(v) for v in box), np.zeros((3, 2))) for box in coords]
        encoder = pipeline.speech.load_audio_encoder(_catalog_folder(models, "lipsync_whisper") / "tiny.pt", "cpu")
        vae_ours = AutoencoderKL.from_pretrained(str(vae_dir), local_files_only=True, use_safetensors=True).eval()
        models_ = pipeline.MuseTalkModels(lambda *_: network, lambda *_: encoder, lambda *_: vae_ours)
        settings = pipeline.REFERENCE
        start = time.perf_counter()
        actual = pipeline.musetalk(models_, video, [pipeline.FaceTrack(wave[:int(16000 * (frame_count / FPS + 0.01))],
                                                                       faces)],
                                   settings=settings, mask_fn=fake_mask_fn, device="cpu", dtype=torch.float32,
                                   batch_size=2)
        report["our_loop_seconds_cpu"] = round(time.perf_counter() - start, 1)
        diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        changed = np.abs(expected.astype(np.int16) - video.astype(np.int16)).max(axis=-1) > 0
        report["loop"] = {"frames": int(len(actual)), "max_level_diff": int(diff.max()),
                          "share_pixels_differing": float((diff.max(axis=-1) > 0).mean()),
                          "psnr_db_full_frame": psnr(actual, expected),
                          "psnr_db_changed_region": psnr(actual[changed], expected[changed]),
                          "changed_pixels": int(changed.sum())}
        out.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(np.concatenate((video[0], expected[0], actual[0]), axis=1)).save(out / "loop_frame0.png")
    print(json.dumps(report["loop"]))
    (out / "musetalk_real_parity.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True, help="code-only MuseTalk checkout (commit 0a89dec)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-fixture", action="store_true")
    mode.add_argument("--real", action="store_true")
    parser.add_argument("--models", type=Path,
                        help="models folder with the catalog's MuseTalk, lip-sync Whisper and lip-sync VAE folders")
    parser.add_argument("--face", type=Path, help="a portrait (any size) for the whole-loop check")
    parser.add_argument("--voice", type=Path, help="a speech recording (wav)")
    parser.add_argument("--landmarker", type=Path, help="MediaPipe face_landmarker.task")
    parser.add_argument("--out", type=Path, help="folder for the report and a comparison picture")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(ROOT))
    import torch

    torch.set_num_threads(THREADS)
    if args.make_fixture:
        with torch.inference_mode():
            data = make_fixture(args.reference.resolve())
        FIXTURE.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
        print(FIXTURE)
        return 0
    missing = [name for name in ("models", "face", "voice", "landmarker", "out") if getattr(args, name) is None]
    if missing:
        parser.error("--real needs " + ", ".join("--" + name for name in missing))
    run_real(args.reference.resolve(), args.models, args.face, args.voice, args.landmarker, args.out, args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
