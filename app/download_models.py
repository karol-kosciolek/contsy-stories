# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Download model bundles in advance instead of on first use.

    venv\\Scripts\\python.exe -m app.download_models                 list bundles and sizes
    venv\\Scripts\\python.exe -m app.download_models all             every model, full quality
    venv\\Scripts\\python.exe -m app.download_models all --lite      every model, Lite set
    venv\\Scripts\\python.exe -m app.download_models images|video    one bundle

Without --full/--lite the model set chosen in Settings (or suited to this GPU) is used.
Downloads resume after interruptions and are verified like downloads started from the app.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import config, model_manager

BUNDLES = {"images": "obrazy", "video": "wideo"}  # set keys in config/models.json


def model_ids(bundle: str, tier_name: str = "full") -> List[str]:
    """Catalog entries of a bundle in the given model set. An entry marked "full_only" (the FP8 image-to-video
    experts, which only the Full set runs) is left out of the Lite set."""
    if bundle == "all":
        base = [m["id"] for m in config.CATALOG.get("models", []) if not m["id"].endswith("-lite")]
    else:
        base = list(config.CATALOG.get("sets", {}).get(BUNDLES[bundle], {}).get("models", []))
    if tier_name == "lite":
        base = [i for i in base if not (config.get_model(i) or {}).get("full_only")]
    return list(dict.fromkeys(model_manager.tiered(i, tier_name) for i in base))


def size_gb(ids: List[str]) -> float:
    """Download size of a clean installation; files shared by several entries (VAE, LoRAs) count once, and a
    file made by an install step counts as the official files it is made from."""
    return model_manager.download_bytes(ids) / 1e9


def disk_gb(ids: List[str]) -> float:
    """Disk space once installed (the sources of install steps are deleted after them)."""
    return model_manager.disk_bytes(ids) / 1e9


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download Contsy model bundles in advance.")
    parser.add_argument("bundle", nargs="?", choices=[*BUNDLES, "all"])
    chosen = parser.add_mutually_exclusive_group()
    chosen.add_argument("--full", action="store_const", const="full", dest="tier",
                        help="full-quality models (16 GB+ VRAM)")
    chosen.add_argument("--lite", action="store_const", const="lite", dest="tier",
                        help="Lite models (8-12 GB VRAM)")
    args = parser.parse_args(argv)
    if not args.bundle:
        for tier_name in model_manager.TIERS:
            for bundle in (*BUNDLES, "all"):
                ids = model_ids(bundle, tier_name)
                print(f"{tier_name:5s} {bundle:7s} {size_gb(ids):6.1f} GB to download, {disk_gb(ids):6.1f} GB "
                      f"installed  ({len(ids)} models)")
        print(f"This PC uses the '{model_manager.tier()}' set. Run again with a bundle name "
              "(and --full or --lite) to download it.")
        return 0

    tier_name = args.tier or model_manager.tier()

    def progress(frac: float, desc: str) -> None:
        print(f"\r{frac * 100:5.1f}%  {desc[:100]:100s}", end="", flush=True)

    error = model_manager.ensure(model_ids(args.bundle, tier_name), progress, exact=True)
    print()
    if error:
        print(error, file=sys.stderr)
        return 1
    print(f"Ready: every model in '{args.bundle}' ({tier_name}) is installed and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
