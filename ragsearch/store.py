"""Persist chunk vectors, their text and a per-source manifest in ``index.npz``.

The manifest records a content hash for every source, so unchanged sources can
be skipped on re-ingest and changed sources replace their old chunks. It lives
*inside* ``index.npz`` so one atomic write saves everything together;
``manifest.json`` is a human-readable copy that is never trusted on load.

Several processes may share a data dir (say the API server and a CLI ingest):
saves take a lock file, and if another process saved since this store loaded,
its sources are merged in rather than overwritten.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ragsearch.fileutil import LockTimeout, atomic_write_bytes, file_lock
from ragsearch.index import (
    Index,
    SearchResult,
    VectorIndex,
    decode_json,
    encode_json,
    npz_bytes,
)

log = logging.getLogger(__name__)


class StoreError(RuntimeError):
    """Raised when stored data is damaged or inconsistent with what is being added."""


class Store:
    def __init__(
        self,
        data_dir: str | Path,
        embedder_name: str,
        index_factory: Callable[[int], Index] = VectorIndex,
    ):
        self.data_dir = Path(data_dir)
        self.embedder_name = embedder_name
        self._index_factory = index_factory
        self._sources: dict[str, dict] = {}
        self._items: list[dict] = []
        self._blocks: list[np.ndarray] = []  # vector rows, concatenated only when needed
        self._dim: int | None = None
        self._index: Index | None = None
        self._dirty: set[str] = set()  # sources changed here since the last load/save
        self._signature: tuple | None = None
        self._load()

    @property
    def _index_path(self) -> Path:
        return self.data_dir / "index.npz"

    @property
    def _manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    def __len__(self) -> int:
        return len(self._items)

    @property
    def sources(self) -> dict[str, dict]:
        return dict(self._sources)

    def is_current(self, source: str, digest: str) -> bool:
        return self._sources.get(source, {}).get("hash") == digest

    def replace(
        self, source: str, digest: str, vectors: np.ndarray, items: list[dict]
    ) -> None:
        """Drop any old chunks for ``source`` and store ``items`` in their place."""
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if vectors.shape[0] != len(items):
            raise StoreError(f"Got {vectors.shape[0]} vectors but {len(items)} items")
        if items and self._dim is not None and vectors.shape[1] != self._dim:
            raise StoreError(f"Store holds dim={self._dim} vectors, got dim={vectors.shape[1]}")
        if self._sources.get(source, {}).get("chunks"):
            self._drop_rows(source)
        if items:
            self._dim = vectors.shape[1]
            self._blocks.append(vectors)
            self._items.extend(items)
        self._sources[source] = {"hash": digest, "chunks": len(items)}
        self._changed(source)

    def remove(self, source: str) -> None:
        """Forget ``source`` and its chunks (e.g. the file was deleted)."""
        if source in self._sources:
            self._drop_rows(source)
            del self._sources[source]
            self._changed(source)

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        if not self._items:
            return []
        if self._index is None:
            self._index = self._index_factory(self._vectors().shape[1])
            self._index.add(self._vectors(), self._items)
        return self._index.search(query, k)

    def refresh(self) -> None:
        """Pick up saves made by other processes since this store last loaded or saved."""
        if self._disk_signature() != self._signature:
            if self._dirty:
                self._merge_from_disk()
            else:
                self._load()

    def save(self) -> None:
        try:
            with file_lock(self.data_dir / ".lock"):
                if self._disk_signature() != self._signature:
                    self._merge_from_disk()  # another process saved: keep its sources too
                manifest = {"embedder": self.embedder_name, "sources": self._sources}
                vectors = self._vectors() if self._items else np.zeros((0, 0), np.float32)
                atomic_write_bytes(self._index_path, npz_bytes(
                    vectors=vectors, items=encode_json(self._items),
                    manifest=encode_json(manifest)))
                atomic_write_bytes(self._manifest_path,
                                   json.dumps(manifest, indent=2).encode("utf-8"))
                self._signature = self._disk_signature()
                self._dirty.clear()
        except LockTimeout as err:
            raise StoreError(str(err)) from err

    # --- internals --------------------------------------------------------------

    def _changed(self, source: str) -> None:
        self._dirty.add(source)
        self._index = None

    def _vectors(self) -> np.ndarray:
        if len(self._blocks) != 1:
            dim = self._dim or 0
            self._blocks = [np.concatenate(self._blocks) if self._blocks
                            else np.zeros((0, dim), dtype=np.float32)]
        return self._blocks[0]

    def _drop_rows(self, source: str) -> None:
        keep = [i for i, item in enumerate(self._items) if item["source"] != source]
        if len(keep) != len(self._items):
            self._blocks = [self._vectors()[keep]]
            self._items = [self._items[i] for i in keep]

    def _disk_signature(self) -> tuple | None:
        try:
            st = self._index_path.stat()
        except FileNotFoundError:
            return None
        return st.st_mtime_ns, st.st_size, st.st_ino

    def _read_disk(self) -> tuple[dict, list[dict], np.ndarray] | None:
        if not self._index_path.exists():
            if self._manifest_path.exists():
                log.warning("%s is missing; every source will be ingested again",
                            self._index_path)
            return None
        try:
            with np.load(self._index_path, allow_pickle=False) as data:
                vectors = data["vectors"]
                items = decode_json(data["items"])
                manifest = (decode_json(data["manifest"]) if "manifest" in data.files
                            else json.loads(self._manifest_path.read_text(encoding="utf-8")))
            sources = dict(manifest["sources"])
            embedder = manifest["embedder"]
        except (OSError, ValueError, KeyError, TypeError) as err:
            raise StoreError(
                f"{self.data_dir} is damaged ({type(err).__name__}: {err}). "
                "Delete it to start again, or use another --data-dir."
            ) from err
        if sources and embedder != self.embedder_name:
            raise StoreError(
                f"{self.data_dir} was built with embedder {embedder!r}, "
                f"not {self.embedder_name!r}. Use the same --embedder or another data dir."
            )
        counts: dict[str, int] = {}
        for item in items:
            counts[item["source"]] = counts.get(item["source"], 0) + 1
        for source in set(counts) | set(sources):
            if sources.get(source, {}).get("chunks") != counts.get(source, 0):
                sources[source] = {"hash": None, "chunks": counts.get(source, 0)}  # re-ingest
        return sources, items, vectors

    def _load(self) -> None:
        disk = self._read_disk()
        self._sources, self._items, vectors = disk if disk else ({}, [], None)
        self._blocks = [vectors] if vectors is not None and vectors.size else []
        self._dim = vectors.shape[1] if self._blocks else None
        self._dirty.clear()
        self._index = None
        self._signature = self._disk_signature()

    def _merge_from_disk(self) -> None:
        disk = self._read_disk()
        if disk is not None:
            disk_sources, disk_items, disk_vectors = disk
            theirs = [i for i, it in enumerate(disk_items) if it["source"] not in self._dirty]
            ours = [i for i, it in enumerate(self._items) if it["source"] in self._dirty]
            blocks = [b for b in (disk_vectors[theirs] if disk_vectors.size else None,
                                  self._vectors()[ours] if ours else None)
                      if b is not None and b.size]
            if len({b.shape[1] for b in blocks}) > 1:
                raise StoreError(f"{self.data_dir} holds vectors of a different size")
            sources = {s: v for s, v in disk_sources.items() if s not in self._dirty}
            sources.update({s: self._sources[s] for s in self._dirty if s in self._sources})
            self._items = [disk_items[i] for i in theirs] + [self._items[i] for i in ours]
            self._blocks, self._sources = blocks, sources
            self._dim = blocks[0].shape[1] if blocks else self._dim
        self._index = None
        self._signature = self._disk_signature()
