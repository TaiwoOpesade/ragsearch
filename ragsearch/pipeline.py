"""Ingest documents into a store and search it.

Each source is hashed first; unchanged sources are skipped before anything is
embedded. Changed or new sources are chunked (across segment boundaries) and
embedded in small batches so memory use stays bounded on modest hardware.

With ``prune_under``, files that were indexed under that folder but no longer
exist there (deleted, renamed or moved) are removed, so their old text stops
appearing in results.
"""

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ragsearch.chunker import Chunk, chunk_stream
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
    removed: list[str] = field(default_factory=list)


def content_hash(doc: Document) -> str:
    digest = hashlib.sha256()
    for segment in doc.segments():
        digest.update(segment.encode("utf-8"))
    return digest.hexdigest()


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
    prune_under: str | Path | None = None,
) -> IngestReport:
    """Add new or changed ``documents`` to ``store`` and save it."""
    report = IngestReport()
    seen: set[str] = set()
    try:
        for doc in documents:
            seen.add(doc.source)
            try:
                digest = content_hash(doc)
                if store.is_current(doc.source, digest):
                    report.skipped.append(doc.source)
                    continue
                chunks = chunk_stream(doc.segments(), doc.source, chunk_size, overlap)
                vectors, items = _embed_chunks(chunks, embedder, batch_size)
            except (LoaderError, UnicodeError) as err:
                log.warning("Skipping %s: %s", doc.source, err)
                report.failed.append(doc.source)
                continue
            store.replace(doc.source, digest, vectors, items)
            report.updated.append(doc.source)
            report.added += len(items)
        if prune_under is not None:
            report.removed = _prune(store, Path(prune_under), seen)
    finally:
        # Keep whatever finished, even if e.g. Ollama goes away part-way through.
        if report.updated or report.removed:
            store.save()
    return report


def _prune(store: Store, root: Path, seen: set[str]) -> list[str]:
    """Remove indexed files under ``root`` that this ingest did not see and that are gone.

    Checking the disk too matters: a folder that could not be listed this time
    (an offline share, a permission change) is skipped silently by the walk.
    """
    base = root.resolve().as_posix()
    prefix = base.rstrip("/") + "/"
    gone = [s for s in store.sources
            if s not in seen and (s == base or s.startswith(prefix)) and not Path(s).exists()]
    for source in gone:
        store.remove(source)
    return gone


def search(query: str, embedder: Embedder, store: Store, k: int = 5) -> list[SearchResult]:
    return store.search(embedder.embed_query(query), k)
