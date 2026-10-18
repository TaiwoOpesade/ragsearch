"""Split text into overlapping word-based chunks."""

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
