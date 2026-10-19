import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from ragsearch import fileutil
from ragsearch.fileutil import LockTimeout, atomic_write_bytes, file_lock


def assert_free(lock):
    """The lock can be taken again at once, so nothing is still holding it."""
    with file_lock(lock, timeout=0.5):
        pass


def test_atomic_write_replaces_file_contents(tmp_path):
    path = tmp_path / "out.bin"
    atomic_write_bytes(path, b"first")
    atomic_write_bytes(path, b"second\n")
    assert path.read_bytes() == b"second\n"  # binary: no newline translation on Windows
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]  # no temp files left


def test_atomic_write_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "out.bin"
    atomic_write_bytes(path, b"x")
    assert path.read_bytes() == b"x"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_atomic_write_gives_normal_permissions(tmp_path):
    umask = os.umask(0)
    os.umask(umask)
    path = tmp_path / "out.bin"
    atomic_write_bytes(path, b"x")
    assert path.stat().st_mode & 0o777 == 0o666 & ~umask  # not mkstemp's owner-only 0600


def test_atomic_write_retries_while_another_program_holds_the_file(tmp_path, monkeypatch):
    # On Windows os.replace fails while any process has the old file open.
    real_replace = os.replace
    failures = []

    def busy_twice(src, dst):
        if len(failures) < 2:
            failures.append(dst)
            raise PermissionError(13, "file in use")
        real_replace(src, dst)

    monkeypatch.setattr(fileutil.os, "replace", busy_twice)
    atomic_write_bytes(tmp_path / "out.bin", b"data")
    assert (tmp_path / "out.bin").read_bytes() == b"data"
    assert len(failures) == 2


def test_atomic_write_gives_up_and_cleans_up_if_the_file_stays_busy(tmp_path, monkeypatch):
    def always_busy(src, dst):
        raise PermissionError(13, "file in use")

    monkeypatch.setattr(fileutil.os, "replace", always_busy)
    with pytest.raises(PermissionError):
        atomic_write_bytes(tmp_path / "out.bin", b"data", retry_for=0.2)
    assert list(tmp_path.iterdir()) == []


def test_file_lock_is_exclusive_between_threads(tmp_path):
    lock = tmp_path / "x.lock"
    inside, overlaps = [], []

    def worker():
        for _ in range(20):
            with file_lock(lock):
                if inside:
                    overlaps.append(True)
                inside.append(True)
                time.sleep(0.001)
                inside.pop()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlaps == []
    assert_free(lock)


def test_file_lock_times_out_when_held(tmp_path):
    lock = tmp_path / "x.lock"
    with file_lock(lock), pytest.raises(LockTimeout, match="x.lock"), file_lock(lock, timeout=0.2):
        pass


HOLDER = """
import sys, time
sys.path.insert(0, sys.argv[2])
from ragsearch.fileutil import file_lock
with file_lock(sys.argv[1]):
    print("held", flush=True)
    time.sleep(60)
"""


def test_lock_is_shared_between_processes_and_freed_if_the_holder_dies(tmp_path):
    lock = tmp_path / "x.lock"
    root = str(Path(__file__).parent.parent)
    holder = subprocess.Popen([sys.executable, "-c", HOLDER, str(lock), root],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(LockTimeout), file_lock(lock, timeout=0.3):
            pass  # another process holds it
    finally:
        holder.kill()  # a crash: the process never releases the lock itself
        holder.wait()
        holder.stdout.close()
    assert_free(lock)  # the operating system released it


def test_an_old_lock_file_left_behind_is_simply_reused(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text("", encoding="utf-8")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    assert_free(lock)


def test_lock_is_released_when_the_body_raises(tmp_path):
    lock = tmp_path / "x.lock"
    with pytest.raises(RuntimeError), file_lock(lock):
        raise RuntimeError("boom")
    assert_free(lock)


def test_an_unopenable_lock_path_fails_at_once_instead_of_spinning(tmp_path):
    lock = tmp_path / "x.lock"
    lock.mkdir()  # a directory where the lock file should be
    started = time.monotonic()
    with pytest.raises(OSError), file_lock(lock, timeout=30):
        pass
    assert time.monotonic() - started < 2


def test_a_permission_error_fails_at_once_instead_of_spinning(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError(13, "read-only folder")

    monkeypatch.setattr(fileutil.os, "open", denied)
    started = time.monotonic()
    with pytest.raises(PermissionError), file_lock(tmp_path / "x.lock", timeout=30):
        pass
    assert time.monotonic() - started < 2


def test_reader_lock_reads_anyway_when_the_folder_cannot_hold_a_lock(tmp_path):
    from ragsearch.fileutil import reader_lock

    blocked = tmp_path / "x.lock"
    blocked.mkdir()  # the lock file cannot be created, as in a read-only folder
    with reader_lock(blocked):
        pass  # no error: reading goes ahead unlocked


def test_reader_lock_still_waits_for_a_writer(tmp_path):
    from ragsearch.fileutil import reader_lock

    lock = tmp_path / "x.lock"
    with file_lock(lock), pytest.raises(LockTimeout), reader_lock(lock, timeout=0.2):
        pass
