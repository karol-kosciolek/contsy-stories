# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Make the bundled voice samples (assets/voices) with VoxCPM2 - a development tool.

Each voice in SPECS is described in words (gender, age, character of the voice). VoxCPM2's
voice design reads a description in parentheses at the start of the text, so for every spec
the tool generates several candidates with different seeds, scores them with UTMOS
(speech-quality model, torch.hub tarepan/SpeechMOS) and measures the pitch (librosa.pyin),
keeps the best candidate whose pitch fits the spec, trims silence and writes a 48 kHz mono
16-bit WAV plus the manifest assets/voices/voices.json read by app/voices.py.

The samples are synthetic: no recording of a real person is used at any point.

Usage (from the project root, with the app's venv; the GPU must be free):
    venv\\Scripts\\python.exe tools\\make_voices.py            # all voices
    venv\\Scripts\\python.exe tools\\make_voices.py hanna marek  # only these
    venv\\Scripts\\python.exe tools\\make_voices.py --candidates 2 --out <folder>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "voxcpm2"
OUT_DIR = ROOT / "assets" / "voices"
SENTENCES = [
    "Thanks for stopping by today. Before we get started, let me show you what we have been "
    "working on this week, step by step.",
    "It's a beautiful morning out here. Grab a coffee, sit back, and let's walk through the "
    "plan together, one piece at a time.",
    "Alright, here is the quick version. We'll start with the basics, then look at a few "
    "details that really make a difference.",
]
# name, gender, description shown in the app, voice-design instruction, pitch range in Hz
SPECS = [
    ("Hanna", "female", "20s, bright and lively",
     "A woman in her early twenties, bright, lively, cheerful, youthful high voice", (205, 290)),
    ("Zoe", "female", "20s, soft and gentle",
     "A young woman in her twenties, soft, gentle, slightly breathy voice", (190, 265)),
    ("Maja", "female", "30s, warm and friendly",
     "A woman in her thirties, warm, friendly and relaxed voice", (180, 245)),
    ("Lena", "female", "30s, clear and confident",
     "A woman in her thirties, clear, confident, articulate news-presenter voice", (175, 235)),
    ("Nina", "female", "30s, low and smoky",
     "A woman in her thirties, low, smoky, velvety, deep alto voice", (135, 185)),
    ("Klara", "female", "40s, calm storyteller",
     "A woman in her forties, calm, composed, soothing storytelling voice", (160, 220)),
    ("Sara", "female", "40s, energetic presenter",
     "A woman in her forties, energetic, upbeat, enthusiastic presenter voice", (185, 255)),
    ("Ewa", "female", "50s, mature and warm",
     "A woman in her fifties, mature, warm, reassuring voice", (150, 205)),
    ("Julia", "female", "60s, gentle and unhurried",
     "A woman in her sixties, gentle, kind, grandmotherly voice, unhurried pace", (145, 205)),
    ("Rita", "female", "70s, soft and slightly husky",
     "An elderly woman in her seventies, soft, slightly husky, slow, aged voice", (135, 195)),
    ("Adam", "male", "20s, young and energetic",
     "A young man in his twenties, energetic, upbeat, friendly voice, clearly masculine "
     "medium pitch", (105, 165)),
    ("Leon", "male", "20s, soft-spoken and calm",
     "A young man in his twenties, soft-spoken, calm, relaxed voice", (95, 145)),
    ("Marek", "male", "30s, deep and warm",
     "A man in his thirties, deep, warm, rich voice", (80, 125)),
    ("Tomasz", "male", "30s, clear and confident",
     "A man in his thirties, clear, confident, articulate presenter voice", (100, 162)),
    ("Filip", "male", "30s, light and friendly",
     "A man in his thirties, light tenor, friendly, smiling, higher male voice", (118, 175)),
    ("Oskar", "male", "40s, deep narrator",
     "A man in his forties, very deep, authoritative documentary-narrator bass voice", (72, 115)),
    ("Igor", "male", "40s, rough and gravelly",
     "A man in his forties, rough, gravelly, slightly raspy voice", (85, 130)),
    ("Piotr", "male", "50s, mature and steady",
     "A man in his fifties, mature, steady, measured voice", (90, 135)),
    ("Jan", "male", "60s, warm and slow",
     "A man in his sixties, warm, kind, slow grandfatherly voice", (92, 145)),
    ("Henryk", "male", "70s, elderly and thin",
     "An elderly man in his seventies, thin, slightly shaky, slow, aged voice", (100, 160)),
]
CFG, STEPS = 2.0, 20          # the app's settings (engine/pipelines/speech.py)
MIN_LEN, MAX_LEN = 4.0, 12.0  # usable sample length in seconds


def slugify(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum() or c == "_")


def load_utmos(device: str):
    """UTMOS22 strong scorer from torch.hub (needs internet once); None when unavailable."""
    import torch
    try:
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        return predictor.to(device).eval()
    except Exception as exc:  # noqa: BLE001 - scoring is optional, pitch still decides
        print(f"[WARN] UTMOS unavailable ({exc}); candidates are ranked by pitch only.")
        return None


def pitch_hz(y16: np.ndarray) -> float:
    """Median fundamental frequency of the voiced frames (Hz), 0 when nothing is voiced."""
    import librosa
    f0, voiced, _ = librosa.pyin(y16.astype(np.float32), fmin=60, fmax=420, sr=16000)
    values = f0[voiced & np.isfinite(f0)] if f0 is not None else np.array([])
    return float(np.median(values)) if values.size else 0.0


def trim(wav: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    y, _ = librosa.effects.trim(wav, top_db=38)
    pad = np.zeros(int(sr * 0.15), dtype=wav.dtype)
    return np.concatenate([pad, y, pad])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="voice names to (re)make; default: all")
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="manifest to update (default: <out>/voices.json)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress is visible when stdout is a file
    wanted = {n.lower() for n in args.names}
    specs = [s for s in SPECS if not wanted or s[0].lower() in wanted]
    if not specs:
        print("No matching voice names. Known:", ", ".join(s[0] for s in SPECS))
        return 1
    import soundfile as sf
    import torch
    import librosa
    from voxcpm import VoxCPM

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (out_dir / "voices.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    entries = {e["name"]: e for e in manifest.get("voices", []) if isinstance(e, dict) and "name" in e}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading VoxCPM2 from {MODEL_DIR} on {device}...")
    model = VoxCPM.from_pretrained(str(MODEL_DIR), load_denoiser=False, optimize=False)
    sr = model.tts_model.sample_rate
    scorer = load_utmos(device)
    started = time.time()
    for n, (name, gender, description, design, (lo, hi)) in enumerate(specs, 1):
        sentence = SENTENCES[(n - 1) % len(SENTENCES)]
        best = None
        for k in range(args.candidates):
            seed = 1000 * (SPECS.index((name, gender, description, design, (lo, hi))) + 1) + k
            torch.manual_seed(seed)
            np.random.seed(seed)
            text = f"({design}, natural conversational English) {sentence}"
            try:
                wav = model.generate(text=text, cfg_value=CFG, inference_timesteps=STEPS)
            except Exception as exc:  # noqa: BLE001 - one bad candidate must not stop the run
                print(f"  {name} #{k}: generation failed ({exc})")
                continue
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            wav = trim(wav, sr)
            seconds = len(wav) / sr
            y16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            f0 = pitch_hz(y16)
            score = None
            if scorer is not None:
                with torch.inference_mode():
                    score = float(scorer(torch.from_numpy(y16).unsqueeze(0).to(device), 16000).item())
            in_range = lo <= f0 <= hi and MIN_LEN <= seconds <= MAX_LEN
            distance = 0.0 if lo <= f0 <= hi else min(abs(f0 - lo), abs(f0 - hi))
            rank = (in_range, -(distance), score if score is not None else 0.0)
            print(f"  {name} #{k}: {seconds:.1f} s, F0 {f0:.0f} Hz, UTMOS "
                  f"{score if score is None else round(score, 2)}, fits={in_range}")
            if best is None or rank > best["rank"]:
                best = {"rank": rank, "wav": wav, "seconds": seconds, "f0": f0, "utmos": score,
                        "seed": seed}
        if best is None:
            print(f"[ERROR] {name}: no candidate could be generated")
            continue
        slug = slugify(name)
        target = out_dir / f"{slug}.wav"
        sf.write(str(target), best["wav"], sr, subtype="PCM_16")
        entries[name] = {"name": name, "slug": slug, "file": target.name, "gender": gender,
                         "age": description.split(",")[0].strip(),
                         "description": description, "seconds": round(best["seconds"], 2),
                         "f0_hz": round(best["f0"]),
                         "utmos": None if best["utmos"] is None else round(best["utmos"], 2),
                         "design": design, "seed": best["seed"], "sentence": sentence}
        print(f"[{n}/{len(specs)}] {name}: {best['seconds']:.1f} s, F0 {best['f0']:.0f} Hz, "
              f"UTMOS {entries[name]['utmos']} -> {target.name}")
    order = [s[0] for s in SPECS]
    manifest = {
        "about": ("Synthetic voices generated with VoxCPM2 (openbmb/VoxCPM2, Apache-2.0) by "
                  "tools/make_voices.py from a text description of each voice; no recording "
                  "of a real person was used. Candidates were ranked by pitch range and UTMOS."),
        "format": "48 kHz, mono, 16-bit PCM WAV",
        "voices": [entries[k] for k in sorted(entries, key=lambda k: order.index(k)
                                               if k in order else len(order))],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    print(f"Manifest: {manifest_path} ({len(manifest['voices'])} voices, "
          f"{(time.time() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
