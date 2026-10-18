"""Ingest documents into a store and search it.

Each source is hashed first; unchanged sources are skipped before anything is
embedded. Changed or new sources are chunked and embedded in small batches so
memory use stays bounded on modest hardware.
"""

import hashlib
import logging
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field

import numpy as np

from ragsearch.chunker import Chunk, chunk_text
from ragsearch.embedder import Embedder
from ragsearch.index import SearchResult
from ragsearch.loaders import Document, LoaderError
from ragsearch.store import Store

log = logging.getLogger(__name__)

BATCH_SIZE = 32


@dataclass
class IngestReport:
    added: int = 0
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def content_hash(doc: Document) -> str:
    digest = hashlib.sha256()
    for segment in doc.segments():
        digest.update(segment.encode("utf-8"))
    return digest.hexdigest()


def _chunks(doc: Document, size: int, overlap: int) -> Iterator[Chunk]:
    offset = 0
    for segment in doc.segments():
        chunks = chunk_text(segment, source=doc.source, size=size, overlap=overlap)
        for chunk in chunks:
            yield Chunk(text=chunk.text, source=chunk.source, index=offset + chunk.index)
        offset += len(chunks)


def _embed_chunks(
    chunks: Iterable[Chunk], embedder: Embedder, batch_size: int
) -> tuple[np.ndarray, list[dict]]:
    vectors: list[np.ndarray] = []
    items: list[dict] = []
    batch: list[Chunk] = []

    def flush() -> None:
        vectors.append(embedder.embed([c.text for c in batch]))
        items.extend(asdict(c) for c in batch)
        batch.clear()

    for chunk in chunks:
        batch.append(chunk)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    if not vectors:
        return np.zeros((0, embedder.dim), dtype=np.float32), []
    return np.vstack(vectors), items


def ingest(
    documents: Iterable[Document],
    embedder: Embedder,
    store: Store,
    *,
    chunk_size: int = 200,
    overlap: int = 40,
    batch_size: int = BATCH_SIZE,
) -> IngestReport:
    """Add new or changed ``documents`` to ``store`` and save it."""
    report = IngestReport()
    try:
        for doc in documents:
            try:
                digest = content_hash(doc)
                if store.is_current(doc.source, digest):
                    report.skipped.append(doc.source)
                    continue
                chunks = _chunks(doc, chunk_size, overlap)
                vectors, items = _embed_chunks(chunks, embedder, batch_size)
            except LoaderError as err:
                log.warning("Skipping %s: %s", doc.source, err)
                report.failed.append(doc.source)
                continue
            store.replace(doc.source, digest, vectors, items)
            report.updated.append(doc.source)
            report.added += len(items)
    finally:
        # Keep whatever finished, even if e.g. Ollama goes away part-way through.
        if report.updated:
            store.save()
    return report


def search(query: str, embedder: Embedder, store: Store, k: int = 5) -> list[SearchResult]:
    return store.search(embedder.embed([query])[0], k)
