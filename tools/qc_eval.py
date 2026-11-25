# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Measure the shot check (app/shot_qc.py) on the regression set (tests/fixtures/quality).

    venv\\Scripts\\python.exe tools\\qc_eval.py              # the CPU detectors only (seconds)
    venv\\Scripts\\python.exe tools\\qc_eval.py --vision     # plus the local vision model (GPU, ~30 s a shot)

For every labelled shot it prints the verdict and the failing checks, then per defect letter how many labelled shots
were caught (sensitivity) and how many good shots were failed (false alarms), per detector family. --out writes the
whole result as JSON. Nothing here writes to config/ or a state folder: the vision model's answers go to a scratch folder.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import bible, shot_qc  # noqa: E402
from app.shot_spec import ShotSpec  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "quality"


def fixture_bible() -> bible.Bible:
    data = json.loads((FIXTURES / "series.json").read_text(encoding="utf-8"))
    refs = FIXTURES / "refs"
    return bible.from_series(data, photos={"lumi": [str(refs / "lumi.png"), str(refs / "lumi_34.png")],
                                           "barley": [str(refs / "barley.png"), str(refs / "barley_34.png")]},
                             plates={"the_clearing": str(refs / "the_clearing.png"),
                                     "lumi_s_burrow": str(refs / "lumi_s_burrow.png")})


def spec_for(row: dict, allowed: bible.Bible) -> ShotSpec:
    """A spec for a labelled shot: its place is unknown to the fixture (no plate question), what it must show is the
    label's must_show."""
    return ShotSpec(location="", time_of_day="day", action="", camera="static medium", cast=tuple(row["cast"]),
                    key_visual=tuple(row.get("must_show") or ()))


def family(check_name: str) -> str:
    return "vision" if check_name.startswith("vision") else "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vision", action="store_true", help="Ask the local vision model too (GPU).")
    parser.add_argument("--only", nargs="*", help="Only these shot ids.")
    parser.add_argument("--out", type=Path, help="Write the results as JSON here.")
    args = parser.parse_args()
    allowed = fixture_bible()
    labels = json.loads((FIXTURES / "labels.json").read_text(encoding="utf-8"))
    rows, results = [row for row in labels["clips"] if not args.only or row["id"] in args.only], []
    with tempfile.TemporaryDirectory() as scratch:
        for row in rows:
            cast = tuple(allowed.character(slug) for slug in row["cast"])
            context = shot_qc.Context(cast=cast, spec=spec_for(row, allowed), bible=allowed, vision=args.vision,
                                      work=Path(scratch) / row["id"])
            started = time.monotonic()
            verdict = shot_qc.check_clip(FIXTURES / row["file"], context)
            failing = [check for check in verdict.checks if check.status == shot_qc.FAIL]
            results.append({"id": row["id"], "defects": row["defects"], "expect": row["expect"],
                            "status": verdict.status, "seconds": round(time.monotonic() - started, 1),
                            "failing": [{"name": c.name, "defect": c.defect, "detail": c.detail} for c in failing],
                            "unchecked": [c.name for c in verdict.checks if c.status == shot_qc.UNCHECKED]})
            print(f"{row['id']:18s} expect={row['expect']:4s} got={verdict.status:9s} "
                  f"{time.monotonic() - started:5.1f}s  " + "; ".join(f"{c.name}" for c in failing), flush=True)
    print()
    letters = sorted({letter for row in rows for letter in row["defects"]})
    for kind in ("cpu", "vision", "all"):
        if kind == "vision" and not args.vision:
            continue
        print(f"== {kind} detectors")
        for letter in letters:
            labelled = [r for r in results if letter in r["defects"]]
            caught = [r for r in labelled if any(kind == "all" or family(f["name"]) == kind for f in r["failing"])]
            print(f"  {letter}: shot failed {len(caught)}/{len(labelled)}  "
                  + ", ".join(r["id"] for r in labelled if r not in caught))
        good = [r for r in results if r["expect"] == "pass"]
        false = [r for r in good if any(kind == "all" or family(f["name"]) == kind for f in r["failing"])]
        print(f"  false alarms on good shots: {len(false)}/{len(good)} " + ", ".join(
            f"{r['id']} ({', '.join(f['name'] for f in r['failing'])})" for r in false))
    if args.out:
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
