import os
import threading
import time

import pytest

from ragsearch.fileutil import LockTimeout, atomic_write_bytes, file_lock


def test_atomic_write_replaces_file_contents(tmp_path):
    path = tmp_path / "out.bin"
    atomic_write_bytes(path, b"first")
    atomic_write_bytes(path, b"second")
    assert path.read_bytes() == b"second"
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]  # no temp files left


def test_atomic_write_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "out.bin"
    atomic_write_bytes(path, b"x")
    assert path.read_bytes() == b"x"


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
    assert not lock.exists()


def test_file_lock_times_out_when_held(tmp_path):
    lock = tmp_path / "x.lock"
    with file_lock(lock), pytest.raises(LockTimeout, match="x.lock"), file_lock(lock, timeout=0.2):
        pass


def test_stale_lock_left_by_a_crashed_process_is_broken(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text("", encoding="utf-8")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    with file_lock(lock, timeout=0.5, stale_after=60):
        pass
    assert not lock.exists()


def test_lock_is_released_when_the_body_raises(tmp_path):
    lock = tmp_path / "x.lock"
    with pytest.raises(RuntimeError), file_lock(lock):
        raise RuntimeError("boom")
    assert not lock.exists()
