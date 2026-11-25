# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""An end-to-end check of the Agent page without a Claude key: a scripted model drives the real tools, the real queue and
(with --gpu) one real picture on the GPU, in a scratch state folder, and the run is killed in the middle and resumed.

    venv/Scripts/python.exe tools/agent_e2e_fake.py [--gpu] [--keep]

Nothing of the owner's is touched: the child processes move every state path (config, characters, items, projects, queue,
outputs, history, agent runs) into a scratch folder right after app.config is imported and stop if any path of the app
still points elsewhere. With --gpu the script first checks that the card is free (under 3 GiB used), takes the shared
GPU lock (mkdir contsy_gpu.lock with owner.txt, removed at the end) and watches the owner's live queues: when one of them
has a waiting or running job, the check stops itself. The models are only read (the verified-models list is copied
into the scratch folder, so nothing is hashed again). Exit code 0 = every check passed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "contsy_gpu.lock"
LIVE_QUEUES = (ROOT / "config" / "queue.json",)
BUSY = ("w kolejce", "generowanie")
MAX_USED_MIB = 3072
FAKE_KEY = "sk-" + "ant-" + "api03-" + "e2e" * 12


# --- the child: one process of the app on scratch state -------------------------------------------------------------

def child(phase: str, scratch: Path, gpu: bool) -> int:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from app import config

    state = scratch / "state"
    config.STATE_ROOT = state
    config.STATE_CONFIG_DIR = state / "config"
    config.OUTPUTS_DIR = state / "outputs"
    config.HISTORY_DIR = state / "history"
    config.ensure_dirs()
    import app.main  # noqa: F401 - registers every job runner of the app
    from app import (agent_config, agent_runner, agent_store, characters, history, items, model_manager, prefs,
                     queue_manager, timeline, voices)

    inside = {"characters": characters.DIR, "items": items.DIR, "timeline": timeline.DIR, "voices": voices.USER_DIR,
              "queue": queue_manager._STATE, "queue_files": queue_manager.FILES_DIR, "prefs": prefs._FILE,
              "history": history._HISTORY_FILE, "runs": agent_store.RUNS_DIR, "settings": agent_config.SETTINGS_FILE,
              "verified": model_manager.VERIFIED_FILE, "outputs": config.OUTPUTS_DIR}
    stray = {name: str(path) for name, path in inside.items() if scratch.resolve() not in Path(path).resolve().parents}
    if stray:
        print(json.dumps({"abort": "a state path is outside the scratch folder", "paths": stray}))
        return 3

    from unittest.mock import patch

    sys.path.insert(0, str(ROOT))
    from tests.agent_fake import ScriptedProvider, tool_turn

    voice = next(row.slug for row in voices.all_voices())
    scene = {"prompt": "A barista hands a cup of coffee over a wooden counter in morning light, slow push-in",
             "seconds": 5, "background": "a quiet cafe with a coffee machine"}
    finish = ("finish_run", {"outcome": "complete", "summary": "The end-to-end check ran.", "checklist": [
        {"requirement": "A package", "status": "done"}]})

    def last_result(request) -> dict:
        text = request.messages[-1]["content"][0]["content"][0]["text"]
        return json.loads(text)

    def die_soon(delay: float):
        def turn(request):
            threading.Timer(delay, lambda: os._exit(7)).start()
            return tool_turn(("wait_for_jobs", {"job_ids": [1], "timeout_seconds": 600}))
        return turn

    if phase == "run1":
        script = [tool_turn(("update_status", {"phase": "intake", "text": "Reading the brief"})),
                  tool_turn(("list_voices", {})),
                  tool_turn(("save_project", {"title": "Coffee ad", "scenes": [scene, {**scene, "seconds": 4}]})),
                  lambda request: tool_turn(("check_project", {"project_id": last_result(request)["id"]})),
                  tool_turn(("list_projects", {}))]
        if gpu:
            script += [tool_turn(("queue_still", {"prompt": "A white coffee cup on a wooden table, soft window light",
                                                  "format": "Square 1:1"})), die_soon(4.0)]
        else:
            script += [lambda request: os._exit(7)]  # the process dies at a clean boundary, between two turns
    else:
        memo = {}

        def remember(request):
            memo["project"] = last_result(request)[0]["id"]
            return tool_turn(("get_queue", {}))

        script = [tool_turn(("list_projects", {})), remember]
        if gpu:
            script += [tool_turn(("wait_for_jobs", {"job_ids": [1], "timeout_seconds": 900})),
                       lambda request: tool_turn(("add_to_package", {
                           "source": "gallery_file", "file": last_result(request)["jobs"][0]["files"][0]}))]
        script += [lambda request: tool_turn(("write_document", {"doc": "script", "project_id": memo["project"]})),
                   lambda request: tool_turn(("write_document", {"doc": "shot_list", "project_id": memo["project"]})),
                   tool_turn(("build_package", {})), tool_turn(finish)]

    provider = ScriptedProvider(*script)
    brief = scratch / "brief.md"
    if phase == "run1":
        brief.write_text("# Coffee ad\nA 10 s film for Beans & Co.\n", encoding="utf-8")
    with patch.object(agent_runner, "make_provider", return_value=provider), \
            patch.object(agent_config, "get_key", return_value=(FAKE_KEY, "test")), \
            patch.object(agent_runner, "_keep_awake"):
        if gpu:
            queue_manager.start()
        interrupted = agent_runner.startup()
        if phase == "run1":
            run_id = agent_runner.start([brief], "", {"mode": "prepare"})
        else:
            run_id = interrupted[0]
            agent_runner.resume(run_id)
        agent_runner.wait(run_id, 1800)
    record = agent_store.RunStore(run_id).read()
    print(json.dumps({"run": run_id, "state": record.get("state"), "interrupted": interrupted,
                      "failure": record.get("failure")}))
    return 0 if record.get("state") == "completed" or phase == "run1" else 1


# --- the parent: checks, the lock, the two phases and the verdict ------------------------------------------------------

def used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout.split()
    return int(out[0]) if out else 10 ** 9


def live_busy() -> list:
    found = []
    for path in LIVE_QUEUES:
        try:
            jobs = json.loads(path.read_text(encoding="utf-8")).get("jobs", [])
        except (OSError, ValueError):
            continue
        found += [f"{path}: job {job.get('id')} {job.get('status')}" for job in jobs if job.get("status") in BUSY]
    return found


def fingerprint() -> dict:
    """What the live folders hold now (their names and sizes), to show that nothing of the owner's changed."""
    out = {}
    for name in ("characters", "items", "timeline_projects", "voices", "outputs", "agent_runs", "history"):
        base = ROOT / name
        out[name] = sorted((str(p.relative_to(base)), p.stat().st_size) for p in base.rglob("*") if p.is_file()) \
            if base.exists() else []
    for path in (ROOT / "config" / "user_prefs.json", ROOT / "config" / "queue.json", ROOT / "config" / "agent_settings.json",
                 ROOT / "config" / "agent_credentials.json"):
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
    return out


def run_child(phase: str, scratch: Path, gpu: bool, stop: threading.Event) -> tuple[int, str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--child", phase, "--scratch", str(scratch)]
    if gpu:
        command.append("--gpu")
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    lines: list[str] = []
    threading.Thread(target=lambda: lines.extend(process.stdout), daemon=True).start()
    while process.poll() is None:
        if stop.is_set():
            process.kill()
        time.sleep(1.0)
    time.sleep(0.5)
    return process.returncode, "".join(lines)


def verify(scratch: Path, gpu: bool, outputs: list) -> list[str]:
    problems = []
    state = scratch / "state"
    runs = sorted((state / "agent_runs").iterdir()) if (state / "agent_runs").exists() else []
    if len(runs) != 1:
        return [f"expected one run folder, found {len(runs)}"]
    run = runs[0]
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if record.get("state") != "completed" or (record.get("outcome") or {}).get("kind") != "complete":
        problems.append(f"the run ended {record.get('state')}, outcome {(record.get('outcome') or {}).get('kind')}")
    transcript = (run / "transcript.jsonl").read_text(encoding="utf-8")
    if gpu and "INTERRUPTED" not in transcript:
        problems.append("the call that was running when the process was killed was not reported as interrupted")
    package = run / "deliverables"
    try:
        with zipfile.ZipFile(package / "delivery.zip") as bundle:
            names = bundle.namelist()
            for wanted in ("README.txt", "report.html", "manifest.json", "brief_checklist.csv"):
                if wanted not in names:
                    problems.append(f"{wanted} is not in the ZIP")
            for suffix in ("-script.html", "-shot-list.csv"):
                if not any(name.startswith("coffee-ad-") and name.endswith(suffix) for name in names):
                    problems.append(f"no *{suffix} document is in the ZIP")
            if gpu and not any(name.endswith(".png") for name in names):
                problems.append("the picture the GPU made is not in the ZIP")
            manifest = json.loads(bundle.read("manifest.json"))
            for item in manifest["files"]:
                if hashlib.sha256(bundle.read(item["name"])).hexdigest() != item["sha256"]:
                    problems.append(f"{item['name']} does not match the manifest")
    except (OSError, KeyError, zipfile.BadZipFile, ValueError) as error:
        problems.append(f"the ZIP could not be read: {error}")
    for path in state.rglob("*"):
        if path.is_file() and FAKE_KEY.encode() in path.read_bytes():
            problems.append(f"the key is in {path.relative_to(state)}")
    queue = state / "config" / "queue.json"
    if gpu:
        jobs = json.loads(queue.read_text(encoding="utf-8")).get("jobs", []) if queue.exists() else []
        mine = [job for job in jobs if job.get("owner") == f"agent:{run.name}"]
        if not mine or not all(job["label"].startswith("Agent: ") and job["downloads"] is False for job in mine):
            problems.append("the run's job is not marked as the agent's")
        if not mine or mine[0].get("status") != "gotowe":
            problems.append(f"the picture job did not finish: {[job.get('status') for job in mine]}")
        if not list((state / "outputs").glob("*.png")):
            problems.append("no picture in the scratch outputs")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", action="store_true", help="make one real picture on the GPU")
    parser.add_argument("--keep", action="store_true", help="keep the scratch folder")
    parser.add_argument("--child", choices=("run1", "run2"))
    parser.add_argument("--scratch")
    args = parser.parse_args()
    if args.child:
        return child(args.child, Path(args.scratch), args.gpu)

    before = fingerprint()
    scratch = Path(tempfile.mkdtemp(prefix="contsy_agent_e2e_"))
    (scratch / "state" / "config").mkdir(parents=True)
    verified = ROOT / "config" / "verified_models.json"
    if verified.exists():
        shutil.copy2(verified, scratch / "state" / "config" / "verified_models.json")
    locked, stop, watcher = False, threading.Event(), None
    ok = False
    try:
        if args.gpu:
            if busy := live_busy():
                print("The owner's queue is busy, so nothing was run:", *busy, sep="\n  ")
                return 4
            if (used := used_mib()) >= MAX_USED_MIB:
                print(f"The GPU holds {used} MiB (limit {MAX_USED_MIB}), so nothing was run.")
                return 4
            try:
                LOCK.mkdir()
            except FileExistsError:
                print("The GPU lock is held by:", (LOCK / "owner.txt").read_text(errors="replace"))
                return 4
            locked = True
            (LOCK / "owner.txt").write_text(f"agent_e2e_fake.py pid {os.getpid()} {time.strftime('%H:%M:%S')}\n")

            def watch():
                while not stop.wait(3.0):
                    if live_busy():
                        print("An owner's job started: stopping.")
                        stop.set()
            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
        code1, out1 = run_child("run1", scratch, args.gpu, stop)
        print("--- phase 1 (killed on purpose, exit", code1, ")\n" + out1[-1500:])
        code2, out2 = run_child("run2", scratch, args.gpu, stop)
        print("--- phase 2 (resumed, exit", code2, ")\n" + out2[-1500:])
        problems = [] if not stop.is_set() else ["stopped: an owner's job started"]
        problems += [f"phase 1 ended with {code1}, not the planned kill (7)"] if code1 != 7 else []
        problems += [f"phase 2 ended with {code2}"] if code2 else []
        problems += verify(scratch, args.gpu, [])
        after = fingerprint()
        changed = [name for name in before if before[name] != after[name]]
        if changed:
            print("Note: these live folders changed meanwhile (the owner's app may have been working):", changed)
        for problem in problems:
            print("PROBLEM:", problem)
        ok = not problems
        print("PASSED" if ok else "FAILED")
        return 0 if ok else 1
    finally:
        stop.set()
        if locked:
            shutil.rmtree(LOCK, ignore_errors=True)
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)
        else:
            print("Scratch folder kept:", scratch)


if __name__ == "__main__":
    sys.exit(main())
