# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Read a model file ahead into the operating system's file cache, behind every other disk read.

A memory-mapped weight file whose pages are not in the cache is read by page faults, in small pieces (about
0.8 GB/s for Qwen-Image-Edit's 21 GB file on the test PC, 2026-11-15); one sequential pass with big reads brings it in
at the disk's speed. A reading thread can run at background priority (Windows' THREAD_MODE_BACKGROUND_BEGIN), so the
models the job loads meanwhile get the disk first. `ahead` reads the files of the next stages in the order they are
needed, while the current stage computes.
"""
from __future__ import annotations

import json
import re
import sys
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterable, Iterator, Optional

READ_BYTES = 64 << 20  # one read
WAIT_LIMIT = 600.0  # s an `ahead` thread waits for its start event at most (a stage that failed never sets it)


def _thread_mode(mode: int) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), mode)
    except (AttributeError, OSError):
        pass


def _normal_memory_priority() -> None:
    """Give the calling thread the normal memory priority (MEMORY_PRIORITY_NORMAL, 5) again (Windows)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class _Priority(ctypes.Structure):
            _fields_ = [("MemoryPriority", wintypes.ULONG)]

        kernel32 = ctypes.WinDLL("kernel32")  # its own prototypes: ctypes.windll's are shared with other code
        kernel32.GetCurrentThread.restype = wintypes.HANDLE
        kernel32.SetThreadInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        priority = _Priority(5)
        kernel32.SetThreadInformation(kernel32.GetCurrentThread(), 0,  # ThreadMemoryPriority
                                      ctypes.byref(priority), ctypes.sizeof(priority))
    except (AttributeError, OSError):
        pass


@contextmanager
def background_io() -> Iterator[None]:
    """The calling thread's disk reads go behind everyone else's inside the block (Windows; elsewhere unchanged).
    The pages it reads keep the normal memory priority: background mode also lowers that to very low (1, measured on
    2026-11-15), so Windows would hand the pages read ahead to the next allocation first."""
    _thread_mode(0x00010000)  # THREAD_MODE_BACKGROUND_BEGIN
    _normal_memory_priority()
    try:
        yield
    finally:
        _thread_mode(0x00020000)  # THREAD_MODE_BACKGROUND_END


_BLOCK = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


def network_order(path: str | Path) -> Optional[list]:
    """For a safetensors file of a network of numbered blocks (`blocks.N.`, the Wan experts), the (offset, length)
    byte ranges of its contents in the order the network uses them: the header and the tensors outside the blocks,
    then block 0, 1, 2 ... with each block's tensors (its matrices and their scales) together, adjacent ranges joined.
    The files store their tensors by name (blocks 0, 1, 10-19, 2, 20-29 ...), so a reader going from start to end
    would leave blocks 2-9 for last although the first step needs them first. None for any other file."""
    if Path(path).suffix.lower() != ".safetensors":
        return None
    try:
        with open(path, "rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 0 < size < (100 << 20):
                return None
            header = json.loads(stream.read(size))
    except (OSError, ValueError):
        return None
    start = 8 + size
    rest, blocks = [], {}
    for name, entry in header.items():
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        try:
            begin, end = (int(value) for value in entry["data_offsets"])
        except (KeyError, TypeError, ValueError):
            return None
        match = _BLOCK.search(name)
        (blocks.setdefault(int(match.group(1)), []) if match else rest).append((start + begin, end - begin))
    if not blocks:
        return None
    ranges = [(0, start), *sorted(rest)] + [piece for block in sorted(blocks) for piece in sorted(blocks[block])]
    joined: list = []
    for offset, length in ranges:
        if joined and joined[-1][0] + joined[-1][1] == offset:
            joined[-1] = (joined[-1][0], joined[-1][1] + length)
        elif length:
            joined.append((offset, length))
    return joined


def read_file(path: str | Path) -> int:
    """Read the file in READ_BYTES pieces (the data is dropped: only the cache keeps it): a network's safetensors file
    in the order the network uses it (`network_order`), any other from start to end. The reads leave the GIL to the
    other threads. Returns the bytes read."""
    buffer = memoryview(bytearray(READ_BYTES))
    total = 0
    ranges = network_order(path)
    with open(path, "rb", buffering=0) as stream:
        if ranges is None:
            while count := stream.readinto(buffer):
                total += count
            return total
        for offset, length in ranges:
            stream.seek(offset)
            while length > 0:
                count = stream.readinto(buffer[:min(length, READ_BYTES)])
                if not count:
                    break
                total += count
                length -= count
    return total


def _memory_status() -> tuple:
    """(total, available) bytes of physical memory (Windows: GlobalMemoryStatusEx; elsewhere sysconf, with nothing
    known as available); (0, 0) when they cannot be told. Available memory is the free and the cached (standby) pages:
    what reading a file can take without trimming a process's working set."""
    if sys.platform == "win32":
        try:
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_extended", ctypes.c_ulonglong)]

            status = _Status()
            status.length = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_phys), int(status.avail_phys)
        except (AttributeError, OSError):
            return 0, 0
        return 0, 0
    try:
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), 0
    except (AttributeError, OSError, ValueError):
        return 0, 0


def physical_memory() -> int:
    """Bytes of physical memory of this machine (0 when it cannot be told)."""
    return _memory_status()[0]


def available_memory() -> int:
    """Bytes of physical memory a file read can take now without trimming a process (0 when it cannot be told)."""
    return _memory_status()[1]


AHEAD_RESERVE = 8 << 30  # available memory a read-ahead leaves untouched (the other studio, the job, the system)


def _room_for(size: int) -> bool:
    """Whether reading `size` more bytes ahead leaves AHEAD_RESERVE of the available memory. Available memory is what
    no process holds (free pages and cached files), so a read never pushes a working set out, the other studio's or
    the job's; it does push out the oldest cached files, which are those of stages that already ran. Unknown memory
    (another system) never holds a read back."""
    total, available = _memory_status()
    if not total or sys.platform != "win32":
        return True
    return available >= size + AHEAD_RESERVE


_READING: set = set()  # files an `ahead` thread has listed and not finished (being read, or waiting their turn)
_ACTIVE: set = set()  # files an `ahead` thread is reading this moment
_CLAIMED: set = set()  # listed files a stage read itself (`settle`): their `ahead` thread leaves them out
_READING_LOCK = threading.Lock()
_FINISHED: dict = {}  # file -> (size, mtime_ns, time.monotonic()) of its last complete read by `ahead`
RECENT_SECONDS = 60.0  # a file read ahead this recently is taken as cached (`recently_read`): longer, and a
# later stage of the job (a 20-30 GB model) may have pushed it out again


def recently_read(path: str | Path, within: float = RECENT_SECONDS) -> bool:
    """Whether an `ahead` thread finished reading this file (the same size and modification time) within the last
    `within` seconds, so its pages are most likely still in the file cache and reading it again would only copy it."""
    import time

    try:
        stat = Path(path).stat()
    except OSError:
        return False
    with _READING_LOCK:
        done = _FINISHED.get(str(path))
    return done is not None and done[:2] == (stat.st_size, stat.st_mtime_ns) and time.monotonic() - done[2] <= within


def _record(path: str) -> None:
    import time

    try:
        stat = Path(path).stat()
    except OSError:
        return
    with _READING_LOCK:
        _FINISHED[path] = (stat.st_size, stat.st_mtime_ns, time.monotonic())


def _log(name: str, path: str, size: int, seconds: Optional[float]) -> None:
    """One line per file an `ahead` thread read or left out, with the memory it saw (the job log shows it)."""
    available = available_memory() / (1 << 30)
    what = f"read in {seconds:.1f} s" if seconds is not None else "left out: no room"
    print(f"[read-ahead] {name}: {Path(path).name} ({size / (1 << 30):.1f} GiB) {what}, {available:.1f} GiB "
          "available", flush=True)


def ahead(paths: Iterable[str | Path], *, wait: Optional[threading.Event] = None, background: bool = False,
          name: str = "read-ahead", skip_recent: bool = True, patience: float = 0.0) -> Optional[threading.Thread]:
    """Read `paths` into the file cache one after another, in their order, on a daemon thread (the data is dropped),
    so the stage that maps them next finds them in memory instead of reading the disk page by page. A path that is
    missing, empty, already being read by another `ahead` thread, read completely within RECENT_SECONDS (unless
    `skip_recent` is False) or unreadable is left out, and so is a file the available memory has no room for when its
    turn comes (`_room_for`: the other studio's and the job's memory is never pushed out by a read-ahead), unless
    room appears within `patience` seconds (the stage computing meanwhile gives its memory back when it ends). `wait`: the thread starts reading once the event is set
    (another stage's files are read first); `background`: at background disk priority (`background_io`), behind the
    reads the job makes meanwhile. Returns the thread, or None when there is nothing to read."""
    candidates = [path for path in dict.fromkeys(str(path) for path in paths if path)
                  if not (skip_recent and recently_read(path))]
    files = []
    with _READING_LOCK:
        for path in candidates:
            try:
                if path not in _READING and Path(path).is_file() and Path(path).stat().st_size:
                    files.append(path)
            except OSError:
                continue
        _READING.update(files)
    if not files:
        return None

    def run() -> None:
        import time

        try:
            if wait is not None:
                wait.wait(WAIT_LIMIT)
            with background_io() if background else nullcontext():
                for path in files:
                    try:
                        size = Path(path).stat().st_size
                        deadline = time.monotonic() + patience
                        while not _room_for(size) and time.monotonic() < deadline and not _claimed(path):
                            time.sleep(0.5)  # the stage computing now may give its memory back
                        with _READING_LOCK:
                            if path in _CLAIMED:  # a stage is reading it itself
                                _CLAIMED.discard(path)
                                continue
                            if _room_for(size):
                                _ACTIVE.add(path)
                        if path not in _ACTIVE:  # the stage reads it by page faults
                            _log(name, path, size, None)
                            continue
                        started = time.monotonic()
                        read_file(path)
                        _record(path)
                        _log(name, path, size, time.monotonic() - started)
                    except OSError:
                        pass
                    finally:
                        with _READING_LOCK:
                            _READING.discard(path)
                            _ACTIVE.discard(path)
        finally:
            with _READING_LOCK:
                _READING.difference_update(files)

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread


def _claimed(path: str) -> bool:
    with _READING_LOCK:
        return path in _CLAIMED


def settle(path: str | Path, *, background: bool = True) -> None:
    """Return once `path` is in the file cache as far as reading ahead can tell: an `ahead` thread reading it this
    moment is waited for, a file read completely within RECENT_SECONDS is left alone, and any other file is read by
    this call (at background disk priority unless `background` is False), also one that an `ahead` thread has only
    listed (waiting for its turn or for memory room): that thread then leaves it out. A stage about to map the file
    calls it, so it neither reads a file a second time beside a reader bringing it in nor waits for a read that has
    not started."""
    import time

    key = str(path)
    with _READING_LOCK:
        active = key in _ACTIVE
        if not active and key in _READING:
            _CLAIMED.add(key)
    if active:
        while True:
            with _READING_LOCK:
                if key not in _ACTIVE:
                    return
            time.sleep(0.05)
    if recently_read(key):
        return
    try:
        with background_io() if background else nullcontext():
            read_file(key)
        _record(key)
    except OSError:
        pass
