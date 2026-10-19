"""Small, dependency-free helpers for safe file writes shared by the store and budget.

* ``atomic_write_bytes`` writes a unique temp file and swaps it in with ``os.replace``,
  so a reader never sees a half-written file.
* ``file_lock`` is a cross-process (and cross-thread) lock using the operating
  system's own file locks: ``msvcrt.locking`` on Windows, ``fcntl.flock`` elsewhere.
  The OS releases them if a process dies, so a crash can never leave a stale lock,
  and nobody has to guess when a lock is "too old" to break.
"""

import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_BINARY = getattr(os, "O_BINARY", 0)  # Windows: no newline translation


class LockTimeout(RuntimeError):
    """Raised when another process holds a lock for longer than the timeout."""


def atomic_write_bytes(path: str | Path, data: bytes, retry_for: float = 5.0) -> None:
    """Replace ``path`` with ``data`` in one step.

    The temp file is created with mode 0o666 so permissions follow the umask, as a
    plain open() would (mkstemp's owner-only 0600 locked out other accounts). On
    Windows, ``os.replace`` fails while another program (a reader, an antivirus scan)
    has the old file open, so it is retried for up to ``retry_for`` seconds.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        deadline = time.monotonic() + retry_for
        while True:
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _try_lock(fd: int) -> bool:
    if os.name == "nt":
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:  # held by someone else
            return False
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(fd: int) -> None:
    if os.name == "nt":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def file_lock(path: str | Path, timeout: float = 30.0, poll: float = 0.02) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` (created if needed and left in place).

    Raises ``LockTimeout`` if another holder keeps it longer than ``timeout``.
    A lock file that cannot be opened at all (read-only folder, a directory in the
    way) raises that ``OSError`` at once rather than waiting.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | _BINARY, 0o666)
    try:
        deadline = time.monotonic() + timeout
        while not _try_lock(fd):
            if time.monotonic() > deadline:
                raise LockTimeout(f"{path} is held by another ragsearch process; try again shortly")
            time.sleep(poll)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)
