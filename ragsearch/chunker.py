"""Split text into overlapping word-based chunks."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int


def chunk_text(
    text: str, source: str = "", size: int = 200, overlap: int = 40
) -> list[Chunk]:
    """Split ``text`` into chunks of ``size`` words, sharing ``overlap`` words.

    Overlap stops a sentence that straddles a boundary from being lost to both
    neighbouring chunks.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    words = text.split()
    if not words:
        return []

    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    while True:
        window = words[start : start + size]
        chunks.append(Chunk(text=" ".join(window), source=source, index=len(chunks)))
        if start + size >= len(words):
            break
        start += step
    return chunks


def chunk_stream(
    segments: Iterable[str], source: str = "", size: int = 200, overlap: int = 40
) -> Iterator[Chunk]:
    """Chunk a stream of text segments exactly as ``chunk_text`` would chunk their join.

    Words flow across segment boundaries, so a page break or a 1 MB read boundary
    never cuts a chunk short, and at most ``size + 1`` words are held at a time.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    step = size - overlap
    buffer: list[str] = []
    index = 0
    for segment in segments:
        for word in segment.split():
            buffer.append(word)
            if len(buffer) > size:  # more follows, so this window is not the last
                yield Chunk(text=" ".join(buffer[:size]), source=source, index=index)
                index += 1
                del buffer[:step]
    if buffer:
        yield Chunk(text=" ".join(buffer), source=source, index=index)
