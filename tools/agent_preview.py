# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Open Contsy on another port with a scratch state folder, to look at any page without touching your real studio:
no GPU queue worker runs and no engine is warmed up.

    venv/Scripts/python.exe tools/agent_preview.py <scratch folder> [port]

The scratch folder is one that tools/agent_e2e_fake.py --keep left behind (its "state" folder holds a finished run). Every
state path of the app is moved there right after app.config is imported; the script stops when one is still elsewhere."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    scratch = Path(sys.argv[1]).resolve()
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 7899
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    os.environ.setdefault("CONTSY_SETTINGS_LOCAL", "none")  # the scratch app ignores the real studio's local settings
    from app import config

    state = scratch / "state"
    config.STATE_ROOT = state
    config.STATE_CONFIG_DIR = state / "config"
    config.OUTPUTS_DIR = state / "outputs"
    config.HISTORY_DIR = state / "history"
    config.ensure_dirs()
    config.SETTINGS["app"]["port"] = port
    config.SETTINGS["app"]["open_browser"] = False
    from app import (agent_config, agent_store, characters, history, items, main, model_manager, music_library, prefs,
                     queue_manager, series, timeline, voices)

    for name, path in (("characters", characters.DIR), ("character drafts", characters.DRAFTS_DIR),
                       ("items", items.DIR), ("item drafts", items.DRAFTS_DIR), ("timeline", timeline.DIR),
                       ("series", series.DIR), ("music", music_library.folder()), ("voices", voices.USER_DIR),
                       ("prefs", prefs._FILE), ("agent credentials", agent_config.CREDENTIALS_FILE),
                       ("agent settings", agent_config.SETTINGS_FILE), ("queue", queue_manager._STATE),
                       ("queue files", queue_manager.FILES_DIR), ("history", history._HISTORY_FILE),
                       ("outputs", config.OUTPUTS_DIR), ("runs", agent_store.RUNS_DIR),
                       ("verified", model_manager.VERIFIED_FILE)):
        if scratch not in Path(path).resolve().parents:
            print(f"Stopping: {name} points outside the scratch folder: {path}")
            return 3
    queue_manager.start = lambda: None  # nothing is generated in a preview
    main.engine_runtime.start_warm_imports = lambda: None
    main.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
