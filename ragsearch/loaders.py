"""Find and read .md, .txt and .pdf files without loading a whole file at once.

A ``Document`` is a source name plus a ``segments()`` callable that lazily yields
bounded pieces of text: blocks of at most ``segment_chars`` characters for text
files and one page at a time for PDFs. ``segments()`` can be called repeatedly,
so the pipeline can hash a source first and only embed it when it has changed.
"""

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from pypdf import PdfReader

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
SEGMENT_CHARS = 1_000_000

_LAST_WHITESPACE = re.compile(r"\s(?=\S*\Z)")


class LoaderError(RuntimeError):
    """Raised when a file cannot be read or parsed."""


@dataclass(frozen=True)
class Document:
    source: str
    segments: Callable[[], Iterator[str]]

    @classmethod
    def from_text(cls, source: str, text: str) -> "Document":
        return cls(source=source, segments=lambda: iter([text]))


def _text_segments(path: Path, segment_chars: int) -> Iterator[str]:
    try:
        with path.open(encoding="utf-8-sig") as f:
            carry = ""
            while block := f.read(segment_chars - len(carry)):
                block = carry + block
                match = _LAST_WHITESPACE.search(block)
                if match is None:  # one enormous "word": emit it as-is
                    yield block
                    carry = ""
                    continue
                yield block[: match.start()]
                carry = block[match.end() :]
            if carry:
                yield carry
    except (OSError, UnicodeDecodeError) as err:
        raise LoaderError(f"Could not read {path}: {err}") from err


def _pdf_segments(path: Path) -> Iterator[str]:
    try:
        # Passing an open file (not a path) lets pypdf seek instead of reading it all.
        with path.open("rb") as f:
            for page in PdfReader(f).pages:
                yield page.extract_text() or ""
    except Exception as err:  # a parser of untrusted files can fail in many ways
        raise LoaderError(f"Could not read PDF {path}: {err}") from err


def _find_files(path: Path) -> list[Path]:
    if not path.exists():
        log.warning("Skipping %s: path does not exist", path)
        return []
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            return [path]
        log.warning("Skipping %s: unsupported file type", path)
        return []
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_documents(
    path: str | Path, segment_chars: int = SEGMENT_CHARS
) -> Iterator[Document]:
    """Yield a ``Document`` for every supported file at ``path`` (recursive)."""
    for file in _find_files(Path(path)):
        if file.suffix.lower() == ".pdf":
            segments = partial(_pdf_segments, file)
        else:
            segments = partial(_text_segments, file, segment_chars)
        yield Document(source=file.resolve().as_posix(), segments=segments)
