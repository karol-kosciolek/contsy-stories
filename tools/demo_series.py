# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Contsy Stories demo series: an original animal character, a shared location and prop, two episodes with
storyboards (16:9 and 9:16) and the story memory between them.

    venv\\Scripts\\python.exe tools\\demo_series.py --state <scratch folder>

It writes only into the state folder given (never the live app's unless you point it there) and runs nothing on the
GPU: generating the shots, the voice and the exports is the manual part, done in the app (it takes GPU time).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EPISODE_1 = [
    ("Wide shot of a moonlit forest clearing with tall blue trees and floating fireflies. Pip, a small orange fox with a "
     "white chest, a green scarf and round glasses, tiptoes into the clearing.", 5, []),
    ("Close-up: Pip finds an old silver lantern half buried in the moss and brushes the dirt away with his paw.", 5,
     ["Oh! Who left you here, little lantern?"]),
    ("Pip holds the silver lantern up; it flickers with a warm golden glow as he speaks.", 5,
     ["I promise I will find your owner."]),
    ("Pip jumps over a stream with the lantern; his green scarf slips off and floats away down the water.", 5, []),
    ("Pip looks back at the stream, ears down, then smiles and walks on with the glowing lantern.", 5,
     ["My scarf! ...Never mind. The lantern comes first."]),
]
EPISODE_2 = [
    ("Morning in the same moonlit forest clearing, now golden with sunrise. Pip, a small orange fox with a white chest "
     "and round glasses, without his scarf, shivers a little and holds the silver lantern.", 5,
     ["Brr. I miss my scarf."]),
    ("Pip follows the stream downhill, the silver lantern glowing brighter with every step.", 5, []),
    ("Pip finds his green scarf caught on a reed and wraps it around his neck, beaming.", 5,
     ["There you are! And look, the lantern is glowing!"]),
]


def build() -> dict:
    from app import series, styles, timeline
    data = series.create("Pip and the Lantern Forest", look="cartoon_2d", fmt="16:9", narration="dialogue",
                         logline="A curious little fox and an old silver lantern that glows when someone tells the "
                                 "truth.")
    sid = data["id"]
    series.update(sid, world="The Lantern Forest: tall blue trees, fireflies, a stream that runs downhill to a "
                             "reed meadow.", rules="Animals talk. The silver lantern glows only when someone "
                                                  "tells the truth.",
                  tone="warm, gentle, a little funny", narration_style="each episode ends with a small kindness")
    series.save_character(sid, {"name": "Pip", "kind": "creature", "role": "main",
                                "description": "a small grown-up orange fox with a big "
                                               "fluffy tail", "colors": "orange fur, white chest, white tail tip",
                                "silhouette": "small, round head, big fluffy tail", "outfit": "green scarf",
                                "marks": "round glasses", "personality": "curious, kind, a bit clumsy",
                                "voice": ""})
    series.save_place(sid, "Lantern Forest clearing", "a moonlit clearing with tall blue trees and fireflies")
    series.save_place(sid, "Silver lantern", "an old silver lantern with a warm golden glow", kind="prop")
    made = []
    for number, (shots, fmt, idea, summary, facts) in enumerate((
            (EPISODE_1, "16:9", "Pip finds a silver lantern in the forest and promises to find its owner.",
             "Pip found an old silver lantern in the Lantern Forest clearing and promised to find its owner. "
             "Jumping over the stream he lost his green scarf, which floated downhill.",
             ["Pip carries the silver lantern.", "Pip lost his green scarf in the stream."]),
            (EPISODE_2, "9:16", "Pip follows the stream to find his lost scarf.", "", [])), start=1):
        episode = series.create_episode(sid, idea, title=["The Silver Lantern", "The Lost Scarf"][number - 1],
                                        seconds=sum(s for _, s, _ in shots), fmt=fmt)
        scenes = tuple(timeline.with_lines(timeline.Scene(prompt=p, seconds=s, cut=i == 0), lines)
                       for i, (p, s, lines) in enumerate(shots))
        project = timeline.save(None, None, f"{episode['title']} — Pip and the Lantern Forest", scenes,
                                series.FORMATS[fmt], True, [], style=styles.CARTOON)
        series.update_episode(sid, episode["id"], projects=[project.id], status="storyboard")
        if summary:
            series.set_summary(sid, episode["id"], summary)
            for fact in facts:
                series.add_fact(sid, fact, episode=episode["id"])
        made.append((episode["id"], project.id))
    series.update(sid, story_state="Pip is looking for his lost green scarf and still carries the silver lantern.")
    return {"series": sid, "episodes": made, "context": series.story_context(sid, made[1][0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True, help="the state folder to write into")
    args = parser.parse_args()
    os.environ.setdefault("CONTSY_SETTINGS_LOCAL", "none")
    from app import config
    # Every state path moves before build() imports the stores, which read these folders when they are imported.
    state = Path(args.state).resolve()
    config.STATE_ROOT = state
    config.STATE_CONFIG_DIR = state / "config"
    config.OUTPUTS_DIR = state / config.SETTINGS["paths"]["outputs_dir"]
    config.HISTORY_DIR = state / config.SETTINGS["paths"]["history_dir"]
    found = build()
    print(f"Demo series {found['series']} with episodes {found['episodes']} in {config.STATE_ROOT}")


if __name__ == "__main__":
    main()
