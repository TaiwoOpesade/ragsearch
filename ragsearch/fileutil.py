"""Small, dependency-free helpers for safe file writes shared by the store and budget.

* ``atomic_write_bytes`` writes to a unique temp file and swaps it in with
  ``os.replace``, so a reader never sees a half-written file.
* ``file_lock`` is a cross-process (and cross-thread) lock built on
  ``O_CREAT | O_EXCL``, which works the same on Windows and Linux.
"""

import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class LockTimeout(RuntimeError):
    """Raised when another process holds a lock for longer than the timeout."""


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextmanager
def file_lock(
    path: str | Path,
    timeout: float = 30.0,
    stale_after: float = 300.0,
    poll: float = 0.02,
) -> Iterator[None]:
    """Hold ``path`` as a lock file; a lock older than ``stale_after`` s is broken."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > stale_after:
                    path.unlink(missing_ok=True)  # left behind by a crashed process
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise LockTimeout(
                    f"{path} is held by another ragsearch process; try again shortly"
                ) from None
            time.sleep(poll)
        except PermissionError:  # Windows: the holder is deleting the file right now
            time.sleep(poll)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
