# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Measure MuseTalk 1.5 lip sync: model load, seconds per stage, peak VRAM, compute seconds per output second.

Example (the GPU belongs to the live Studio: pause its queue first; the script refuses a CUDA run while the
queue on --status-port is running or not paused):

    venv\\Scripts\\python.exe tools\\bench_musetalk.py --unet models\\musetalk\\unet.pth ^
        --whisper WHISPER_FOLDER\\tiny.pt --vae-dir VAE_FOLDER ^
        --landmarker models\\mediapipe\\face_landmarker.task --yunet models\\qc\\face_detection_yunet_2023mar.onnx ^
        --video clip_25fps.mp4 --audio line.wav --out bench\\musetalk --batch 8,16,32 --dtype float16,float32

WHISPER_FOLDER and VAE_FOLDER are the catalog's lip-sync Whisper and VAE folders under models\\ (the destinations
`lipsync_whisper` and `lipsync_vae` of config/models.json).

Two speakers in one shot: --faces 2 --audio left.wav right.wav (faces are ordered left to right in the
first frame that shows both, then followed by `faces.follow_tracks`).

Every run writes its video (with the first audio) and `bench_musetalk.json` into --out. The face detection
(MediaPipe/YuNet on the CPU) is timed once and counted in the per-second figure of every run. Per run the
report has the free VRAM at the start (memory the paused Studio still holds shows there), each stage's own
peak, the faces the network returned as NaN or inf (left out by the pipeline, so they show as unchanged
frames, never as black mouths) and each face track's coverage of its speech frames.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DTYPES = ("float16", "float32", "bfloat16")


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unet", type=Path, required=True, help="musetalkV15 unet.pth")
    parser.add_argument("--whisper", type=Path, required=True, help="Whisper tiny: tiny.pt or model.safetensors")
    parser.add_argument("--vae-dir", type=Path, required=True, help="sd-vae-ft-mse folder (config.json + safetensors)")
    parser.add_argument("--landmarker", type=Path, required=True, help="MediaPipe face_landmarker.task")
    parser.add_argument("--yunet", type=Path, help="OpenCV YuNet .onnx for faces MediaPipe misses (optional)")
    parser.add_argument("--video", type=Path, required=True, help="the clip to lip-sync (25 fps recommended)")
    parser.add_argument("--audio", type=Path, nargs="+", required=True,
                        help="the speech (any format, resampled to 16 kHz); one file per face, left to right "
                             "(the last one is reused for the remaining faces)")
    parser.add_argument("--faces", type=int, default=1, help="faces to lip-sync in each frame")
    parser.add_argument("--out", type=Path, required=True, help="folder for the videos and bench_musetalk.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", help="comma-separated: " + ", ".join(DTYPES))
    parser.add_argument("--batch", default="16", help="comma-separated batch sizes, e.g. 8,16,32")
    parser.add_argument("--frames", type=int, default=0, help="use only the first N frames (0: all)")
    parser.add_argument("--settings", choices=("default", "reference"), default="default")
    parser.add_argument("--low-vram", action="store_true", help="the pipeline's low_vram mode (VAE slicing)")
    parser.add_argument("--repeat", type=int, default=1, help="runs per configuration (the first includes warm-up)")
    parser.add_argument("--status-port", type=int, default=7860, help="the live Studio's port for the queue check")
    parser.add_argument("--skip-queue-check", action="store_true", help="no live Studio on this machine")
    args = parser.parse_args(argv)
    args.dtypes = [name.strip() for name in args.dtype.split(",") if name.strip()]
    args.batches = [int(value) for value in args.batch.split(",") if value.strip()]
    if not args.dtypes or any(name not in DTYPES for name in args.dtypes) or not args.batches \
            or min(args.batches) < 1 or args.repeat < 1 or args.faces < 1:
        parser.error("invalid --dtype, --batch, --repeat or --faces")
    for name in ("unet", "whisper", "landmarker", "video"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name.replace('_', '-')} {getattr(args, name)} does not exist")
    for path in args.audio:
        if not path.is_file():
            parser.error(f"--audio {path} does not exist")
    if not (args.vae_dir / "config.json").is_file():
        parser.error(f"--vae-dir {args.vae_dir} has no config.json")
    return args


def _queue_refusal(args) -> str | None:
    """None when the GPU may be used, else the reason it may not."""
    if not args.device.startswith("cuda") or args.skip_queue_check:
        return None
    try:
        from native_stage import queue_idle_and_paused
    except ImportError:
        return ("the live-queue check needs tools/gpu_stage, which is not part of this repository; "
                "pass --skip-queue-check")
    if not queue_idle_and_paused(args.status_port):
        return (f"the Studio queue on port {args.status_port} is not paused and idle; pause it first "
                "(or --skip-queue-check when no Studio runs here)")
    return None


def _read(args):
    import numpy as np
    from engine.media import audio as media_audio
    from engine.media import video as media_video

    info = media_video.probe(args.video)
    pictures = []
    for frame in media_video.read_frames(args.video):
        pictures.append(media_video.to_rgb(frame))
        if args.frames and len(pictures) >= args.frames:
            break
    if not pictures:
        raise SystemExit(f"{args.video} has no frames")
    audios = []
    for path in args.audio:
        samples, rate = media_audio.read(path, rate=16000, channels=1)
        audios.append(samples[0])
    fps = round(float(info.fps or 25))
    return np.stack(pictures), audios, rate, fps


def face_tracks(per_frame, count: int):
    """`count` face tracks (a Face or None per frame) from the faces found in every frame (largest first):
    one track takes the largest face; several start left to right in the first frame that shows them all."""
    from engine.models.musetalk.faces import follow_tracks

    if count == 1:
        return [[faces[0] if faces else None for faces in per_frame]]
    first = next((faces for faces in per_frame if len(faces) >= count), None)
    if first is None:
        raise SystemExit(f"no frame shows {count} faces")
    anchors = [face.box for face in sorted(first[:count], key=lambda face: face.box[0])]
    return follow_tracks(per_frame, anchors)


def _detect(args, frames):
    from engine.models.musetalk.faces import MuseTalkFaceDetector

    start = time.perf_counter()
    with MuseTalkFaceDetector(args.landmarker, args.yunet, num_faces=args.faces) as detector:
        per_frame = [detector.faces(frame) for frame in frames]
    return face_tracks(per_frame, args.faces), time.perf_counter() - start


def _write(path: Path, frames, samples, rate: int, fps: int) -> None:
    from engine.media import video as media_video

    height, width = frames.shape[1:3]
    samples = samples[None, :len(frames) * rate // fps]
    with media_video.Writer(path, fps, width, height, crf=18, audio=(samples, rate)) as writer:
        for frame in frames:
            writer.write(frame)


def _gib(value) -> float:
    return round(value / 2 ** 30, 3)


def _run(args, frames, audios, tracks, fps, dtype_name: str, batch: int) -> dict:
    import dataclasses

    import torch
    from engine.pipelines import musetalk as pipeline

    dtype = getattr(torch, dtype_name)
    models = pipeline.MuseTalkModels.from_files(args.unet, args.whisper, args.vae_dir)
    settings = pipeline.REFERENCE if args.settings == "reference" else pipeline.Settings()
    stages: dict = {}
    reports: list = []
    reserved = [0]
    cuda = args.device.startswith("cuda")
    memory: dict = {}
    if cuda:
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(args.device)
        memory = {"free_vram_gib_at_start": _gib(free), "total_vram_gib": _gib(total),
                  "allocated_gib_at_start": _gib(torch.cuda.memory_allocated(args.device))}

    def stat(name, seconds, peak):
        # every stage reports its own peak (the pipeline resets the counter per stage); the reserved
        # peak is sampled here, the allocator's cache rarely shrinks within a run
        stages[name] = {"seconds": round(seconds, 3), "peak_gib": _gib(peak)}
        if cuda:
            reserved[0] = max(reserved[0], torch.cuda.max_memory_reserved(args.device))

    faces = [pipeline.FaceTrack(audios[min(k, len(audios) - 1)], track) for k, track in enumerate(tracks)]
    start = time.perf_counter()
    output = pipeline.musetalk(models, frames, faces, fps=fps, settings=settings, batch_size=batch,
                               low_vram=args.low_vram, device=args.device, dtype=dtype, stats=stat,
                               report_cb=reports.append)
    seconds = time.perf_counter() - start
    report = reports[0]
    result = {"dtype": dtype_name, "batch": batch, "seconds_total": round(seconds, 3), "stages": stages, **memory,
              "peak_allocated_gib": max((stage["peak_gib"] for stage in stages.values()), default=0.0),
              "peak_reserved_gib": _gib(reserved[0]),
              "nonfinite_latents": report.nonfinite_latents, "nonfinite_faces": report.nonfinite_faces,
              "tracks": [{**dataclasses.asdict(track), "coverage": round(track.coverage, 3)}
                         for track in report.tracks]}
    return result, output


def main(argv=None) -> int:
    args = _arguments(argv)
    sys.path.insert(0, str(ROOT))
    refusal = _queue_refusal(args)
    if refusal:
        print(f"Not starting: {refusal}.", file=sys.stderr)
        return 2
    import numpy as np

    args.out.mkdir(parents=True, exist_ok=True)
    frames, audios, rate, fps = _read(args)
    seconds = len(frames) / fps
    tracks, detect_seconds = _detect(args, frames)
    found = [sum(face is not None for face in track) for track in tracks]
    report = {"video": str(args.video), "frames": len(frames), "fps": fps, "frame_size": list(frames.shape[1:3]),
              "output_seconds": round(seconds, 3), "faces": args.faces, "faces_found_per_track": found,
              "detect_seconds_cpu": round(detect_seconds, 3), "settings": args.settings, "device": args.device,
              "runs": []}
    print(f"{len(frames)} frames at {fps} fps ({seconds:.2f} s), faces per track {found}; "
          f"detection {detect_seconds:.2f} s")
    if fps != 25:
        print("note: MuseTalk was trained at 25 fps; this clip runs at its own rate")
    for dtype_name in args.dtypes:
        for batch in args.batches:
            for attempt in range(args.repeat):
                result, output = _run(args, frames, audios, tracks, fps, dtype_name, batch)
                compute = result["seconds_total"] + detect_seconds
                load = result["stages"].get("load", {}).get("seconds", 0.0)
                result.update({"attempt": attempt + 1,
                               "compute_seconds_per_output_second": round(compute / seconds, 3),
                               "without_load_per_output_second": round((compute - load) / seconds, 3),
                               "changed_frames": int((output != frames).any(axis=(1, 2, 3)).sum())})
                name = f"musetalk_{dtype_name}_b{batch}_{attempt + 1}.mp4"
                _write(args.out / name, output, audios[0], rate, fps)
                result["video"] = name
                report["runs"].append(result)
                stages = ", ".join(f"{key} {value['seconds']:.2f}s" for key, value in result["stages"].items())
                print(f"{dtype_name} batch {batch} run {attempt + 1}: total {result['seconds_total']:.2f} s "
                      f"(load {load:.2f} s) | {stages} | peak {result['peak_allocated_gib']:.2f} GiB allocated, "
                      f"{result['peak_reserved_gib']:.2f} GiB reserved | "
                      f"{result['compute_seconds_per_output_second']:.2f} s per output second "
                      f"({result['without_load_per_output_second']:.2f} without load)")
                coverage = ", ".join(f"{track['synced']}/{track['speech']}" for track in result["tracks"])
                generated = sum(track["generated"] for track in result["tracks"])
                print(f"  speech frames synced per track: {coverage}; changed frames {result['changed_frames']} "
                      f"(faces pasted: {generated}); free VRAM at start "
                      f"{result.get('free_vram_gib_at_start', '-')} GiB")
                if result["nonfinite_faces"] or result["nonfinite_latents"]:
                    print(f"  WARNING: {result['nonfinite_latents']} UNet outputs and {result['nonfinite_faces']} "
                          f"decoded faces were NaN or inf ({dtype_name}); those frames kept their pixels")
                del output
    (args.out / "bench_musetalk.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(args.out / "bench_musetalk.json")
    del np
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
