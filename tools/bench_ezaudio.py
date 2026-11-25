# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Speed and memory of the EzAudio pipeline on the GPU, with the files given explicitly.

    venv\\Scripts\\python.exe tools\\bench_ezaudio.py --transformer models\\ezaudio\\ezaudio_s3_xl.pt
        --config models\\ezaudio\\ezaudio-xl.yml --vae models\\ezaudio\\vae\\1m.pt
        --text-encoder models\\ezaudio\\text_encoder --out <folder> [--steps 25,50,100] [--repeats 3]
        [--cases window,scene,long,warm] [--dtype float16] [--device cuda] [--min-seconds 10]

--min-seconds 10 makes every sound in a full 10 s window and keeps its beginning (the listening comparison with
exact-length windows); float16 is the product's precision on CUDA, --dtype bfloat16 the alternative it replaced.

Cases (each at every --steps value, --repeats times), with the length of the clip they serve:
  window  one 10 s window (9.5 s + the two 0.25 s handles = 500 latent frames); clip 9.5 s
  scene   a 5 s scene in one call: the bed and two 2 s effects layered over it (5.5 + 2.5 + 2.5 s generated);
          clip 5 s
  long    a 30 s bed (30.5 s generated, four windows with continuation); clip 30 s
  warm    the scene case with the engine's ModelCache: the first call loads from disk, the next ones from RAM
Without the cache every call loads all three networks from disk (the operating system's file cache may still
hold the files after the first run; the first repeat is the honest cold load).

Printed and written to <out>/bench.json:
- load time and compute time per stage (text encoder, transformer, decoder);
- seconds of compute per second of the finished clip (the owner's goal is at most 30), with and without loading;
  sounds layered over the same clip count once, by the clip's length. Per second of generated audio (all sounds
  with their handles) as a secondary figure;
- per stage the peak of allocated and of reserved CUDA memory (the caching allocator's pool, which is what the
  16 GB budget has to hold; the CUDA context comes on top);
- allocated and reserved CUDA memory before and after each call, flagged when the call left anything behind (the
  pipeline releases every network, also into the warm cache, which keeps them in host RAM).
The sounds are written as 48 kHz WAV files to <out> for listening.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BED = "light rain on a window with distant thunder, high quality realistic sound effects recording"
EFFECTS = ("a door closing, high quality realistic sound effects recording",
           "footsteps on a wooden floor, high quality realistic sound effects recording")
LOADS = ("load text encoder", "load transformer", "load vae")
COMPUTE = ("encode", "sample", "decode")
GIB = 2 ** 30


def cases(selected, seed) -> dict:
    """{name: (seconds of the clip the call serves, the sounds of the call)}."""
    from engine.pipelines.ezaudio import Sound

    scene = [Sound(BED, 5.0, seed), Sound(EFFECTS[0], 2.0, seed + 1), Sound(EFFECTS[1], 2.0, seed + 2)]
    every = {"window": (9.5, [Sound(BED, 9.5, seed)]), "scene": (5.0, scene), "long": (30.0, [Sound(BED, 30.0, seed)]),
             "warm": (5.0, scene)}
    unknown = set(selected) - set(every)
    if unknown:
        raise SystemExit(f"unknown cases: {sorted(unknown)}")
    return {name: every[name] for name in selected}


def figures(stages: dict, total: float, clip_seconds: float, sounds, handle: float) -> dict:
    """Times of one call per second of the clip (the headline) and per generated second."""
    generated = sum(sound.seconds + 2 * handle for sound in sounds)
    load = sum(stages[name]["seconds"] for name in LOADS if name in stages)
    compute = sum(stages[name]["seconds"] for name in COMPUTE if name in stages)
    return {"clip_seconds": clip_seconds, "generated_seconds": generated, "total_seconds": total,
            "load_seconds": load, "compute_seconds": compute,
            "compute_per_clip_second": compute / clip_seconds, "total_per_clip_second": total / clip_seconds,
            "compute_per_generated_second": compute / generated}


def memory_now(device: str):
    """Allocated and reserved CUDA bytes after all queued work, or None on the CPU."""
    import torch

    if not device.startswith("cuda"):
        return None
    torch.cuda.synchronize(device)
    return {"allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device)}


def memory_change(before, after) -> dict:
    """What a call left on the GPU: any allocated or reserved byte more than before it is flagged."""
    if before is None or after is None:
        return {}
    left_allocated = after["allocated"] - before["allocated"]
    left_reserved = after["reserved"] - before["reserved"]
    return {"before": before, "after": after, "left_allocated_bytes": left_allocated,
            "left_reserved_bytes": left_reserved, "left_behind": left_allocated > 0 or left_reserved > 0}


def _run(models, clip_seconds, sounds, args, steps, out: Path, label: str) -> dict:
    """One generate_many call with its stage times and memory peaks."""
    import torch

    from engine.media.audio import write_wav
    from engine.pipelines import ezaudio

    cuda = args.device.startswith("cuda")
    stages = {}

    def stats(name, seconds, peak):
        # The pipeline resets the peak statistics when a stage starts and calls this when it ends, so the
        # reserved peak read here belongs to the stage too.
        reserved = torch.cuda.max_memory_reserved(args.device) if cuda else 0
        stages[name] = {"seconds": seconds, "peak_allocated_gib": peak / GIB, "peak_reserved_gib": reserved / GIB}

    before = memory_now(args.device)
    started = time.perf_counter()
    waveforms = ezaudio.generate_many(models, sounds, args.negative, device=args.device,
                                      dtype=getattr(torch, args.dtype),
                                      steps=steps, cfg=args.cfg, rescale=args.rescale, eta=args.eta,
                                      min_seconds=args.min_seconds, stats=stats)
    total = time.perf_counter() - started
    after = memory_now(args.device)
    for number, waveform in enumerate(waveforms):
        write_wav(out / f"{label}_{number}.wav", waveform.numpy(), ezaudio.OUTPUT_RATE)
    return {"label": label, "steps": steps, **figures(stages, total, clip_seconds, sounds, ezaudio.HANDLE),
            "peak_allocated_gib": max((stage["peak_allocated_gib"] for stage in stages.values()), default=0.0),
            "peak_reserved_gib": max((stage["peak_reserved_gib"] for stage in stages.values()), default=0.0),
            "memory": memory_change(before, after), "stages": stages}


def _print(result: dict) -> None:
    stages = result["stages"]
    parts = "  ".join(f"{name} {stage['seconds']:.2f}s/{stage['peak_allocated_gib']:.2f}"
                      f"({stage['peak_reserved_gib']:.2f})GiB" for name, stage in stages.items())
    print(f"{result['label']:28s} total {result['total_seconds']:6.2f}s  load {result['load_seconds']:5.2f}s  "
          f"compute {result['compute_seconds']:6.2f}s = {result['compute_per_clip_second']:.2f} s per clip s "
          f"({result['total_per_clip_second']:.2f} with loading; {result['compute_per_generated_second']:.2f} per "
          f"generated s)  peak {result['peak_allocated_gib']:.2f} GiB allocated, "
          f"{result['peak_reserved_gib']:.2f} GiB reserved")
    print(f"{'':28s} {parts}")
    memory = result["memory"]
    if memory.get("left_behind"):
        print(f"{'':28s} WARNING: the call left {memory['left_allocated_bytes'] / 2 ** 20:.1f} MiB allocated and "
              f"{memory['left_reserved_bytes'] / 2 ** 20:.1f} MiB reserved on the GPU")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--transformer", type=Path, required=True, help="ezaudio_s3_xl.pt")
    parser.add_argument("--config", type=Path, required=True, help="ezaudio-xl.yml")
    parser.add_argument("--vae", type=Path, required=True, help="vae/1m.pt (config.json next to it)")
    parser.add_argument("--text-encoder", type=Path, required=True, help="the google/flan-t5-xl folder")
    parser.add_argument("--out", type=Path, required=True, help="folder for bench.json and the WAV files")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--steps", default="25,50,100", help="comma-separated step counts")
    parser.add_argument("--cases", default="window,scene,long,warm")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--cfg", type=float, default=5.0, help="guidance; 1 = the conditional pass only")
    parser.add_argument("--rescale", type=float, default=0.75)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--negative", default="",
                        help="unconditional text (the reference: empty; the app sends: music, speech)")
    parser.add_argument("--min-seconds", type=float, default=0.0,
                        help="generate at least this long and keep the beginning (10 = the training length)")
    parser.add_argument("--cache-gib", type=float, default=12.0, help="host RAM for the warm case's ModelCache")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    import torch

    from engine.core.memory import ModelCache
    from engine.pipelines import ezaudio

    files = ezaudio.EzAudioFiles(args.transformer, args.config, args.vae, args.text_encoder)
    steps_list = [int(value) for value in args.steps.split(",") if value.strip()]
    selected = cases([name.strip() for name in args.cases.split(",") if name.strip()], args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info()
        print(f"{torch.cuda.get_device_name()}: {free / GIB:.1f} of {total / GIB:.1f} GiB free; "
              f"torch {torch.__version__}, dtype {args.dtype}")
    results = []
    for name, (clip_seconds, sounds) in selected.items():
        for steps in steps_list:
            cache = None
            if name == "warm":
                cache = ModelCache(int(args.cache_gib * GIB))
                cache.prepare("audio")
            models = ezaudio.EzAudioModels.from_files(files, cache)
            runs = []
            for repeat in range(args.repeats):
                label = f"{name}_{steps}steps_{repeat + 1}"
                runs.append(_run(models, clip_seconds, sounds, args, steps, args.out, label))
                _print(runs[-1])
            later = runs[1:] or runs
            summary = {"case": name, "steps": steps, "clip_seconds": clip_seconds, "first": runs[0],
                       "median_total_seconds": statistics.median(run["total_seconds"] for run in later),
                       "median_compute_per_clip_second": statistics.median(run["compute_per_clip_second"]
                                                                           for run in later),
                       "median_total_per_clip_second": statistics.median(run["total_per_clip_second"]
                                                                         for run in later),
                       "any_call_left_memory": any(run["memory"].get("left_behind", False) for run in runs),
                       "runs": runs}
            results.append(summary)
            print(f"== {name}, {steps} steps: median of repeats 2.. {summary['median_total_seconds']:.2f}s, "
                  f"{summary['median_compute_per_clip_second']:.2f} s compute per clip s, "
                  f"{summary['median_total_per_clip_second']:.2f} with loading"
                  + ("; MEMORY LEFT ON THE GPU" if summary["any_call_left_memory"] else ""))
            if cache is not None:
                cache.clear()
    (args.out / "bench.json").write_text(json.dumps({"settings": {k: str(v) for k, v in vars(args).items()},
                                                     "results": results}, indent=1), encoding="utf-8")
    print(f"wrote {args.out / 'bench.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
