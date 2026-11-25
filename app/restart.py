# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Restart Contsy Studio from the Settings page (Windows).

The server cannot restart in place: the launcher holds a single-instance mutex and the Gradio
server owns its port. request() therefore starts this file as a small detached helper (standard
library only) that waits until the current process is gone and then runs start.bat, which
opens a new console window and the browser exactly like a manual start. The native assistant
worker closes before the server exits. A generating job blocks the restart.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
START = ROOT / "start.bat"
TITLE = "Contsy Studio"  # the new console window's title
PHRASE = "RESTART"
# DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW: the helper outlives the server
_DETACHED = 0x00000008 | 0x00000200 | 0x08000000
_NEW_CONSOLE = 0x00000010


def request(typed: str) -> Tuple[bool, str]:
    """Restart after the user typed PHRASE in the browser prompt. Returns (ok, message)."""
    from . import queue_manager

    if (typed or "").strip().upper() != PHRASE:
        return False, f"Nothing restarted. The confirmation was cancelled or did not match {PHRASE}."
    running = [j for j in queue_manager.snapshot() if j.status == queue_manager.RUNNING]
    if running:
        return False, (f"Job #{running[0].id} is generating. Wait for it to finish or click "
                       "“Pause and release GPU” in the top bar, then restart.")
    from . import agent_runner

    if agent_runner.busy():
        return False, ("The Agent is working on a job. Stop it on the Agent page (it can be resumed later), then "
                       "restart.")
    if os.name != "nt" or not START.exists():
        return False, "start.bat was not found next to the app, so it cannot restart itself."
    threading.Thread(target=_restart, name="restart", daemon=True).start()
    return True, ("Restarting Contsy Studio. This tab stops responding for a moment; a new "
                  "console window and a new browser tab open in about 20 seconds.")


def port() -> int:
    from . import config

    return int(config.SETTINGS["app"]["port"])


def _restart(delay: float = 2.0) -> None:
    from . import llm

    time.sleep(delay)  # the reply must reach the browser first
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(os.getpid()), str(port())],
                     cwd=str(ROOT), creationflags=_DETACHED, close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        from . import agent_runner

        agent_runner.shutdown()  # a run that began meanwhile is paused, so it can be resumed after the restart
        llm.shutdown()
    finally:
        os._exit(0)  # the detached restart helper waits for the mutex and port to become free


# ----------------------------- helper process -----------------------------

def wait_for_exit(pid: int, timeout: float = 90.0) -> bool:
    """True once the process is gone (Windows: OpenProcess + WaitForSingleObject)."""
    if os.name != "nt":
        return True
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE
    if not handle:
        return True  # already gone
    try:
        return kernel.WaitForSingleObject(handle, int(timeout * 1000)) == 0  # WAIT_OBJECT_0
    finally:
        kernel.CloseHandle(handle)


def wait_for_port(app_port: int, timeout: float = 30.0) -> bool:
    """True once nothing listens on the app port any more."""
    deadline = time.time() + timeout
    while True:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", int(app_port))) != 0:
                return True
        if time.time() >= deadline:
            return False
        time.sleep(0.5)


def start_command() -> str:
    """The command line that runs start.bat in a new console window titled TITLE. `start` takes its first
    QUOTED argument as the title; an unquoted one-word title ("Contsy") is read as the program to run, and
    subprocess quotes a list item only when it has a space, so the command line is written out in full."""
    return f'cmd.exe /c start "{TITLE}" "{START}"'


def launch_after(pid: int, app_port: int) -> int:
    """Helper entry point: wait for the old process, then run start.bat in a new console."""
    wait_for_exit(pid)
    wait_for_port(app_port)
    subprocess.Popen(start_command(), cwd=str(ROOT), creationflags=_NEW_CONSOLE)
    return 0


if __name__ == "__main__":
    sys.exit(launch_after(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 7860))
