# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Measure Wan 2.2 image-to-video with the GGUF + LoRA experts or the FP8 experts, with SDPA or SageAttention, and
compare the pictures.

Configurations (each runs in a fresh process, one after the other, so every run starts with an empty GPU and
first-run costs are its own):

    baseline       GGUF Q6_K experts + LightX2V's _1022 LoRAs, PyTorch SDPA (the product without the FP8 entry)
    fp8            the two scaled FP8 experts of the FP8 entry, no LoRA, SDPA: LightX2V's low-noise file (the GGUF
                   expert with its _1022 LoRA merged in) and the high-noise expert the installer makes the same way
                   from the GGUF expert and its LoRA (engine.models.wan22.convert); --fp8-high runs another file
    sage           GGUF Q6_K + LoRAs, SageAttention
    fp8_sage       both FP8 experts, SageAttention (the product's default with the FP8 entry installed)
    matching_sage  GGUF high-noise expert + its LoRA, FP8 low-noise expert, SageAttention (the product while only the
                   low-noise FP8 expert is installed)

Example (the GPU belongs to the live Studio: pause its queue first; a CUDA run refuses to start while the queue on
--status-port is running or not paused, or while other processes hold more than --max-busy-gib of the GPU):

    venv\\Scripts\\python.exe tools\\bench_wan_fast.py --models models --out bench\\wan ^
        --start-clip timeline_projects\\PROJECT\\clips\\CLIP.mp4 --prompt "..." --configs baseline,fp8,sage,fp8_sage
    venv\\Scripts\\python.exe tools\\bench_wan_fast.py --out bench\\wan --compare --qc-dir models\\qc

A run writes into --out/<config>/: result.json (seconds per stage and per step, the expert loads, peak VRAM per
stage, the process's peak RAM), frames.npy (uint8 [N, H, W, 3]; with --repeat, of the last clip, and frames_1.npy of
the first) and video.mp4 (16 fps). The engine runs as the product runs it (blocks streamed from host memory, the warm
cache of --warm-cache-gib, the text encoder streamed); `--repeat 2` runs the clip again in the same process, as the
next scene of a Timeline job would (warm cache and compiled kernels).

--prompt2 gives the second and later clips another prompt (the next scene of a Timeline job), and --expect announces
all prompts before the first clip as a Timeline job does, so one text-encoder load serves them all. --unfused runs
diffusers' block forward instead of the fused elementwise kernels (engine/models/wan22/block.py), for their A/B;
--encode-latents changes how many latent frames of the start sequence the VAE computes, and --prefault touches every
page of a loaded expert's weights in a background thread (experiments).

--compile-probe times the first and the second call of every Triton kernel at the video's shapes with an empty
Triton cache (the one-off compilation an installation pays once). --compare writes contact sheets (first, middle and
last frame) of every configuration, the SFace similarity of the face in the output to the face in the start picture,
and the per-frame PSNR of fp8_sage against fp8 (the error of the attention alone), into --out/compare.json and
--out/*.png.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIGS = {  # name -> (expert weights, attention)
    "baseline": ("gguf", "sdpa"),
    "fp8": ("fp8", "sdpa"),
    "sage": ("gguf", "sage"),
    "fp8_sage": ("fp8", "sage"),
    "matching_sage": ("matching", "sage"),
}
WEIGHTS = {"gguf": ("gguf", "gguf"), "fp8": ("fp8", "fp8"), "matching": ("gguf", "fp8")}  # (high, low) expert files
GGUF = "diffusion_models/Wan2.2-I2V-A14B-{}Noise-Q6_K.gguf"
LORA = "loras/wan2.2_i2v_A14b_{}_noise_lora_rank64_lightx2v_4step_1022.safetensors"
FP8 = {"high": "wan_i2v_fp8_local/wan2.2_i2v_A14b_high_noise_1022_fp8_e4m3.safetensors",
       "low": "official/Wan2.2-Distill-Models/wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"}
TEXT = "official/Wan2.2-T2V-A14B-Diffusers/text_encoder/model.safetensors.index.json"
VAE = "official/Wan2.2-T2V-A14B-Diffusers/vae/diffusion_pytorch_model.safetensors"
GIB = 1 << 30


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, default=ROOT / "models", help="the models folder of an installation")
    parser.add_argument("--out", type=Path, required=True, help="folder for the results")
    parser.add_argument("--configs", default="baseline,fp8,sage,fp8_sage", help="comma-separated: " + ", ".join(CONFIGS))
    parser.add_argument("--fp8-high", type=Path, help="another high-noise FP8 file for the fp8 configurations "
                        "(default: the converted one in --models)")
    parser.add_argument("--config", choices=tuple(CONFIGS), help=argparse.SUPPRESS)  # one run, in this process
    parser.add_argument("--start", type=Path, help="the start picture")
    parser.add_argument("--start-clip", type=Path, help="or a video whose first frame is the start picture")
    parser.add_argument("--prompt", default="A person at a desk looks up into the camera, warm lamp light")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=97, help="4k + 1")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--split", type=int, default=2)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1, help="clips per process (the next ones run warm)")
    parser.add_argument("--prompt2", help="the prompt of the second and later clips (the next scene of a Timeline job)")
    parser.add_argument("--expect", action="store_true",
                        help="announce every clip's prompt before the first, as a Timeline job does: one text-encoder "
                             "load encodes them all (core.memory.EmbeddingCache.expect)")
    parser.add_argument("--encode-latents", type=int,
                        help="the start sequence's latent frames the VAE computes, the last repeated for the rest "
                             "(engine.pipelines.video.START_LATENT_FRAMES; a number above 25 encodes all of them)")
    parser.add_argument("--prefault", action="store_true",
                        help="touch every page of a loaded expert's host weights in a background thread, block by "
                             "block (an experiment: is the first step's extra time the memory map's page faults?)")
    parser.add_argument("--unfused", action="store_true",
                        help="diffusers' block forward instead of the fused one (the elementwise kernels' A/B)")
    parser.add_argument("--warm-cache-gib", type=int, default=24, help="the engine's host cache (the app's auto: 24)")
    parser.add_argument("--compile-probe", action="store_true", help="time the Triton compilation and stop")
    parser.add_argument("--compare", action="store_true", help="compare the finished runs in --out")
    parser.add_argument("--qc-dir", type=Path, help="folder with the YuNet and SFace .onnx files (for --compare)")
    parser.add_argument("--status-port", type=int, default=7860, help="the live Studio's port for the queue check")
    parser.add_argument("--skip-queue-check", action="store_true", help="no live Studio on this machine")
    parser.add_argument("--max-busy-gib", type=float, default=2.5,
                        help="refuse to start when the GPU already has more than this in use")
    return parser.parse_args(argv)


def _start_picture(args):
    from PIL import Image, ImageOps

    if args.start is not None:
        with Image.open(args.start) as picture:
            return ImageOps.exif_transpose(picture).convert("RGB")
    if args.start_clip is not None:
        from engine.media.video import read_frames

        for frame in read_frames(args.start_clip):
            return frame.to_image().convert("RGB")
    raise SystemExit("--start or --start-clip is required")


def _gpu_ready(args) -> str | None:
    """None when the GPU may be used, else the reason it may not."""
    import torch

    if not args.skip_queue_check:
        try:
            from native_stage import queue_idle_and_paused
        except ImportError:
            return ("the live-queue check needs tools/gpu_stage, which is not part of this repository; "
                    "pass --skip-queue-check")
        if not queue_idle_and_paused(args.status_port):
            return f"the Studio queue on port {args.status_port} is not paused and idle"
    free, total = torch.cuda.mem_get_info()
    busy = (total - free) / GIB
    if busy > args.max_busy_gib:
        return f"{busy:.1f} GiB of the GPU are already in use (limit {args.max_busy_gib} GiB)"
    return None


class Timeline:
    """Synchronised timestamps of the pipeline's progress messages and expert loads."""

    def __init__(self):
        import torch

        self.torch = torch
        self.origin = time.perf_counter()
        self.events = []

    def mark(self, label: str) -> None:
        self.torch.cuda.synchronize()
        self.events.append((round(time.perf_counter() - self.origin, 3), label))


def _prefault(network) -> None:
    """Read one byte of every 4 KiB page of the network's host weights, block by block, in a daemon thread."""
    import threading

    import torch

    def touch():
        with torch.inference_mode():
            for block in network.blocks:
                for tensor in [*block.parameters(), *block.buffers()]:
                    if tensor.device.type == "cpu" and tensor.numel():
                        flat = tensor.detach().reshape(-1).view(torch.uint8) if tensor.is_contiguous() else None
                        if flat is not None:
                            int(flat[::4096].sum())

    threading.Thread(target=touch, daemon=True).start()


def _prompt(args, repeat: int) -> str:
    return args.prompt2 if repeat and args.prompt2 else args.prompt


def _run_clip(args, config: str, picture, timeline: Timeline, cache, embeddings, prompt: str):
    import torch

    from engine.pipelines import video

    weights, attention = CONFIGS[config]
    models_dir = args.models
    kinds = dict(zip(("high", "low"), WEIGHTS[weights]))
    experts = {name: models_dir / GGUF.format(name.capitalize()) if kind == "gguf"
               else args.fp8_high if name == "high" and args.fp8_high else models_dir / FP8[name]
               for name, kind in kinds.items()}
    models = video.WanModels.from_files(models_dir / TEXT, experts["high"], experts["low"], models_dir / VAE, "i2v",
                                        cache=cache)
    load = models.load_transformer
    loads = []

    def timed_load(which, device, dtype):
        timeline.mark(f"load {which} start")
        start = time.perf_counter()
        network = load(which, device, dtype)
        torch.cuda.synchronize()
        loads.append((which, round(time.perf_counter() - start, 3)))
        if args.prefault:
            _prefault(network)
        timeline.mark(f"load {which} end")
        return network

    models = video.WanModels(models.load_text, timed_load, models.load_vae, models.kind, models.recycle,
                             models.text_id)
    from engine.models.wan22 import transformer

    predict = transformer.predict_denoised

    def timed_predict(*arguments, **options):  # one sampling step is one forward pass
        timeline.mark("forward start")
        result = predict(*arguments, **options)
        timeline.mark("forward end")
        return result

    transformer.predict_denoised = timed_predict
    release = video.release

    def timed_release(device):  # the garbage collection and CUDA cache emptying between the networks
        timeline.mark("release start")
        release(device)
        timeline.mark("release end")

    video.release = timed_release
    loras = {name: [(models_dir / LORA.format(name), 1.0)] for name, kind in kinds.items() if kind == "gguf"} or None
    stages = []
    try:
        clip = video.generate(
            models, prompt, args.width, args.height, args.frames, args.seed, args.steps, args.shift,
            start_image=picture, split=args.split, loras=loras, device="cuda", dtype=torch.bfloat16,
            progress_cb=lambda fraction, message: timeline.mark(message),
            stats=lambda name, seconds, peak: stages.append({"stage": name, "seconds": round(seconds, 3),
                                                             "peak_gib": round(peak / GIB, 2)}),
            stream={"pin": cache is None}, stream_text=True, embeddings=embeddings, attention=attention)
    finally:
        transformer.predict_denoised = predict
        video.release = release
    return clip, stages, loads


class MemoryWatch:
    """The process's peak working set and commit (private bytes) and the system's lowest available memory while the
    block runs, sampled every quarter second (psutil)."""

    def __init__(self):
        import threading

        import psutil

        self.process, self.psutil = psutil.Process(), psutil
        self.peak_working_set = self.peak_private = 0
        self.min_available = float("inf")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        info = self.process.memory_info()
        self.peak_working_set = max(self.peak_working_set, info.rss)
        self.peak_private = max(self.peak_private, getattr(info, "private", 0))
        self.min_available = min(self.min_available, self.psutil.virtual_memory().available)

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            self._sample()

    def __enter__(self) -> "MemoryWatch":
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    def report(self) -> dict:
        return {"peak_working_set_gib": round(self.peak_working_set / GIB, 2),
                "peak_private_gib": round(self.peak_private / GIB, 2),
                "min_system_available_gib": round(self.min_available / GIB, 2),
                "system_total_gib": round(self.psutil.virtual_memory().total / GIB, 2)}


def _steps(events) -> list:
    """Seconds of each sampling step (one forward pass of an expert, synchronised at both ends). The first step of
    an expert also reads the expert's weights from the file as its blocks stream in for the first time."""
    steps, started = [], None
    for at, label in events:
        if label == "forward start":
            started = at
        elif label == "forward end" and started is not None:
            steps.append(round(at - started, 3))
            started = None
    return steps


def run_one(args) -> int:
    import numpy as np
    import torch

    from engine import kernels
    from engine.core import attention as attention_backend
    from engine.core.memory import EmbeddingCache, ModelCache
    from engine.media.video import Writer

    reason = _gpu_ready(args)
    if reason:
        print(f"Not starting: {reason}.", file=sys.stderr)
        return 2
    config = args.config
    out = args.out / config
    out.mkdir(parents=True, exist_ok=True)
    kernels.use_cache_dir(args.out / "triton-cache")
    picture = _start_picture(args)
    cache = ModelCache(args.warm_cache_gib * GIB) if args.warm_cache_gib else None
    embeddings = EmbeddingCache()
    prompts = [_prompt(args, repeat) for repeat in range(args.repeat)]
    if args.expect and hasattr(embeddings, "expect"):
        embeddings.expect(prompts)
    if args.encode_latents:
        from engine.pipelines import video

        video.START_LATENT_FRAMES = args.encode_latents
    if args.unfused:
        from engine.models.wan22 import transformer

        if hasattr(transformer, "fused_block"):
            transformer.fused_block.install = lambda model: 0
    report = {"config": config, "weights": CONFIGS[config][0], "fp8_files": {name: str(path) for name, path in (
                  ("high", args.fp8_high or args.models / FP8["high"]), ("low", args.models / FP8["low"]))},
              "attention": attention_backend.resolve(CONFIGS[config][1], "cuda"),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "frames": args.frames,
              "size": [args.width, args.height], "seed": args.seed, "prompts": prompts, "expect": args.expect,
              "unfused": args.unfused, "encode_latents": args.encode_latents, "prefault": args.prefault,
              "busy_gib_at_start": round((torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / GIB, 2),
              "runs": []}
    for repeat in range(args.repeat):
        timeline = Timeline()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with MemoryWatch() as memory:
            clip, stages, loads = _run_clip(args, config, picture, timeline, cache, embeddings, prompts[repeat])
        total = time.perf_counter() - start
        if repeat == 0 and args.repeat > 1:  # the first clip's frames too (frames.npy holds the last one's)
            np.save(out / "frames_1.npy", (clip.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8).numpy())
        report["runs"].append({"seconds": round(total, 3), "seconds_per_clip_second": round(total / (args.frames / 16), 2),
                               "stages": stages, "expert_loads": loads, "steps": _steps(timeline.events),
                               "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 2),
                               "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 2),
                               "ram": memory.report(),
                               "events": timeline.events})
        print(json.dumps({k: v for k, v in report["runs"][-1].items() if k != "events"}), flush=True)
    frames = (clip.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    np.save(out / "frames.npy", frames)
    with Writer(out / "video.mp4", 16, args.width, args.height, crf=17, colorspace=1, color_range=1,
                color_primaries=1, color_trc=1) as writer:
        for frame in frames:
            writer.write(frame)
    picture.save(out / "start.png")
    (out / "result.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


def compile_probe(args) -> int:
    """First and second call of every Triton kernel at the shapes of a 848x480, 97-frame clip, empty cache."""
    import torch

    from engine import kernels

    # a new, empty cache folder (left in place: Windows keeps the compiled launcher modules open until exit)
    folder = args.out / f"triton-cache-probe-{time.strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True)
    from engine.core import attention
    from engine.kernels import fp8, norms

    os.environ.pop(kernels.CACHE_ENV, None)
    kernels.use_cache_dir(folder)
    tokens = ((args.frames - 1) // 4 + 1) * (args.height // 16) * (args.width // 16)
    x = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
    wide = torch.randn(tokens, 13824, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(1, tokens, 40, 128, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(1, 512, 40, 128, device="cuda", dtype=torch.bfloat16)
    vector = torch.randn(5120, device="cuda") * 0.1
    features = torch.randn(1, 96, 4, args.height, args.width, device="cuda", dtype=torch.bfloat16)
    gain = torch.ones(96, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    cases = {
        "quantize_rows 5120": lambda: fp8.quantize_rows(x),
        "quantize_rows 13824": lambda: fp8.quantize_rows(wide),
        "scale_output": lambda: fp8.scale_output_(x, torch.ones(tokens, device="cuda"),
                                                  torch.ones(5120, device="cuda"), None),
        "sage self-attention": lambda: attention.sage(q, q, q),
        "sage cross-attention": lambda: attention.sage(q, text, text),
        "scale_output + gelu": lambda: fp8.scale_output_(wide, torch.ones(tokens, device="cuda"),
                                                         torch.ones(13824, device="cuda"), None, activation="gelu_tanh"),
        "add_norm modulated": lambda: norms.add_norm(x, x, vector, norm=norms.MODULATED, a=vector, b=vector),
        "vae norm + silu": lambda: norms.rms_channels(features, gain, 96 ** 0.5, True),
    }
    result = {}
    for name, call in cases.items():
        times = []
        for _ in range(2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            call()
            torch.cuda.synchronize()
            times.append(round(time.perf_counter() - start, 3))
        result[name] = {"first_call_s": times[0], "second_call_s": times[1]}
        print(name, result[name], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "compile_probe.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


def _faces(frame_rgb, detector, recognizer):
    """SFace features of the largest face in an RGB uint8 frame, or None."""
    import numpy as np

    bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
    height, width = bgr.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(bgr)
    if faces is None:
        return None
    face = max(faces, key=lambda item: item[2] * item[3])
    return recognizer.feature(recognizer.alignCrop(bgr, face))


def compare(args) -> int:
    import numpy as np
    from PIL import Image, ImageDraw

    from engine.models.faces.recognition import cosine_similarity, load_sface, load_yunet

    runs = {name: args.out / name for name in CONFIGS if (args.out / name / "frames.npy").is_file()}
    if not runs:
        raise SystemExit(f"no finished runs in {args.out}")
    frames = {name: np.load(folder / "frames.npy") for name, folder in runs.items()}
    results = {name: json.loads((folder / "result.json").read_text(encoding="utf-8")) for name, folder in runs.items()}
    summary = {"runs": {}}
    detector = recognizer = None
    if args.qc_dir is not None:
        detector = load_yunet(args.qc_dir / "face_detection_yunet_2023mar.onnx")
        recognizer = load_sface(args.qc_dir / "face_recognition_sface_2021dec.onnx")
    for name, clip in frames.items():
        entry = {"seconds": [run["seconds"] for run in results[name]["runs"]],
                 "steps": [run["steps"] for run in results[name]["runs"]],
                 "peak_allocated_gib": [run["peak_allocated_gib"] for run in results[name]["runs"]]}
        if recognizer is not None:
            start = np.asarray(Image.open(runs[name] / "start.png").convert("RGB"))
            reference = _faces(start, detector, recognizer)
            scores = []
            for index in range(0, len(clip), 4):
                found = _faces(clip[index], detector, recognizer)
                scores.append(None if found is None or reference is None
                              else round(cosine_similarity(reference, found), 4))
            valid = [score for score in scores if score is not None]
            entry["sface_to_start"] = {"every_4th_frame": scores, "mean": round(float(np.mean(valid)), 4) if valid
                                       else None, "min": min(valid) if valid else None,
                                       "frames_without_face": scores.count(None)}
        summary["runs"][name] = entry
    if "fp8" in frames and "fp8_sage" in frames:
        a, b = frames["fp8"].astype(np.float64), frames["fp8_sage"].astype(np.float64)
        mse = ((a - b) ** 2).reshape(len(a), -1).mean(axis=1)
        psnr = [round(float(10 * np.log10(255.0 ** 2 / value)), 2) if value > 0 else None for value in mse]
        valid = [value for value in psnr if value is not None]
        summary["psnr_fp8_sage_vs_fp8"] = {"per_frame": psnr, "mean": round(float(np.mean(valid)), 2),
                                           "min": min(valid), "first_frame": psnr[0], "last_frame": psnr[-1]}
    for other in ("fp8", "sage", "matching_sage"):
        if "baseline" in frames and other in frames:
            a, b = frames["baseline"].astype(np.float64), frames[other].astype(np.float64)
            mse = ((a - b) ** 2).reshape(len(a), -1).mean(axis=1)
            summary[f"psnr_{other}_vs_baseline_mean"] = round(
                float(np.mean(10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-9)))), 2)
    # contact sheets: one row per configuration, first / middle / last frame
    names = list(frames)
    first = frames[names[0]]
    height, width = first.shape[1:3]
    label = 22
    sheet = Image.new("RGB", (width * 3, (height + label) * len(names)), "black")
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(names):
        clip = frames[name]
        for column, index in enumerate((0, len(clip) // 2, len(clip) - 1)):
            sheet.paste(Image.fromarray(clip[index]), (column * width, row * (height + label) + label))
        seconds = results[name]["runs"][0]["seconds"]
        draw.text((6, row * (height + label) + 4), f"{name}: frames 0 / {len(clip) // 2} / {len(clip) - 1}, "
                  f"{seconds:.0f} s", fill="white")
    sheet.save(args.out / "contact_sheet_all.png")
    for pair in (("baseline", "fp8_sage"), ("fp8", "fp8_sage"), ("baseline", "matching_sage")):
        if all(name in frames for name in pair):
            rows = Image.new("RGB", (width * 3, (height + label) * 2), "black")
            draw = ImageDraw.Draw(rows)
            for row, name in enumerate(pair):
                clip = frames[name]
                for column, index in enumerate((0, len(clip) // 2, len(clip) - 1)):
                    rows.paste(Image.fromarray(clip[index]), (column * width, row * (height + label) + label))
                draw.text((6, row * (height + label) + 4), name, fill="white")
            rows.save(args.out / f"contact_sheet_{pair[0]}_vs_{pair[1]}.png")
    (args.out / "compare.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if k != "runs" else {n: {kk: vv for kk, vv in e.items() if kk != "sface_to_start"}
                                                  | ({"sface_mean": e["sface_to_start"]["mean"]}
                                                     if "sface_to_start" in e else {})
                                                  for n, e in v.items()})
                      for k, v in summary.items() if k != "psnr_fp8_sage_vs_fp8"}, indent=1))
    if "psnr_fp8_sage_vs_fp8" in summary:
        print("PSNR fp8_sage vs fp8:", {k: v for k, v in summary["psnr_fp8_sage_vs_fp8"].items() if k != "per_frame"})
    return 0


def main(argv=None) -> int:
    args = _arguments(argv)
    if args.compare:
        return compare(args)
    if args.compile_probe:
        reason = _gpu_ready(args)
        if reason:
            print(f"Not starting: {reason}.", file=sys.stderr)
            return 2
        return compile_probe(args)
    if args.config:
        return run_one(args)
    status = 0
    for name in [item.strip() for item in args.configs.split(",") if item.strip()]:
        if name not in CONFIGS:
            raise SystemExit(f"unknown configuration {name!r}")
        command = [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:]),
                   "--config", name]
        print(f"== {name}", flush=True)
        status = subprocess.call(command) or status
    return status


if __name__ == "__main__":
    sys.exit(main())
