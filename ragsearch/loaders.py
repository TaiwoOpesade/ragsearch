"""Find and read .md, .txt and .pdf files without loading a whole file at once.

A ``Document`` is a source name plus a ``segments()`` callable that lazily yields
bounded pieces of text: blocks of at most ``segment_chars`` characters for text
files and one page at a time for PDFs. ``segments()`` can be called repeatedly,
so the pipeline can hash a source first and only embed it when it has changed.
"""

import codecs
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
_SURROGATE = re.compile("[\ud800-\udfff]")


class LoaderError(RuntimeError):
    """Raised when a file cannot be read or parsed."""


@dataclass(frozen=True)
class Document:
    source: str
    segments: Callable[[], Iterator[str]]

    @classmethod
    def from_text(cls, source: str, text: str) -> "Document":
        return cls(source=source, segments=lambda: iter([text]))


def clean_text(text: str) -> str:
    """Replace lone UTF-16 surrogates (from broken PDF fonts) so the text is valid UTF-8."""
    return _SURROGATE.sub("�", text)


def _detect_encoding(path: Path) -> str:
    """``utf-8-sig`` if the file is valid UTF-8, else ``cp1252``; scanned in blocks."""
    with path.open("rb") as f:
        if f.read(2) in (b"\xff\xfe", b"\xfe\xff"):
            raise LoaderError(f"Could not read {path}: it is UTF-16; save it as UTF-8")
        f.seek(0)
        decoder = codecs.getincrementaldecoder("utf-8")()
        valid = True
        for block in iter(lambda: f.read(1 << 20), b""):
            if b"\x00" in block:
                raise LoaderError(f"Could not read {path}: it looks like a binary file")
            if valid:
                try:
                    decoder.decode(block)
                except UnicodeDecodeError:
                    valid = False
        if valid:
            try:
                decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                valid = False
    if not valid:
        log.warning("%s is not valid UTF-8; reading it as Windows-1252", path)
    return "utf-8-sig" if valid else "cp1252"


def _text_segments(path: Path, segment_chars: int) -> Iterator[str]:
    try:
        encoding = _detect_encoding(path)
        with path.open(encoding=encoding, errors="replace") as f:
            carry = ""
            while True:
                want = segment_chars - len(carry)
                piece = f.read(want)
                block = carry + piece
                if len(piece) < want:  # end of file: nothing can follow, keep it whole
                    if block:
                        yield block
                    return
                match = _LAST_WHITESPACE.search(block)
                if match is None:  # one enormous "word": emit it as-is
                    yield block
                    carry = ""
                    continue
                yield block[: match.start()]
                carry = block[match.end() :]
    except OSError as err:
        raise LoaderError(f"Could not read {path}: {err}") from err


def _pdf_segments(path: Path) -> Iterator[str]:
    try:
        # Passing an open file (not a path) lets pypdf seek instead of reading it all.
        with path.open("rb") as f:
            for page in PdfReader(f).pages:
                yield clean_text(page.extract_text() or "")
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
        source = file.resolve().as_posix()
        if _SURROGATE.search(source):  # an undecodable file name: unstorable and unprintable
            log.warning("Skipping %s: its name is not valid Unicode; rename it", ascii(source))
            continue
        if file.suffix.lower() == ".pdf":
            segments = partial(_pdf_segments, file)
        else:
            segments = partial(_text_segments, file, segment_chars)
        yield Document(source=source, segments=segments)
