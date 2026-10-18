"""Persist chunk vectors to ``index.npz`` and a per-source ``manifest.json``.

The manifest records a content hash for every source, so unchanged sources can
be skipped on re-ingest and changed sources replace their old chunks. The
store owns the vectors and items; the search index is rebuilt from them
lazily, which keeps replacement simple for any index backend.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ragsearch.index import Index, SearchResult, VectorIndex, load_npz, save_npz


class StoreError(RuntimeError):
    """Raised when stored data is inconsistent with what is being added."""


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
        self._vectors: np.ndarray | None = None
        self._items: list[dict] = []
        self._index: Index | None = None
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
        if self._vectors is None:
            self._vectors = np.zeros((0, vectors.shape[1]), dtype=np.float32)
        if items and vectors.shape[1] != self._vectors.shape[1]:
            raise StoreError(
                f"Store holds dim={self._vectors.shape[1]} vectors, got dim={vectors.shape[1]}"
            )

        keep = [i for i, item in enumerate(self._items) if item["source"] != source]
        self._vectors = np.vstack([self._vectors[keep], vectors]) if items else self._vectors[keep]
        self._items = [self._items[i] for i in keep] + list(items)
        self._sources[source] = {"hash": digest, "chunks": len(items)}
        self._index = None

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        if not self._items:
            return []
        if self._index is None:
            assert self._vectors is not None
            self._index = self._index_factory(self._vectors.shape[1])
            self._index.add(self._vectors, self._items)
        return self._index.search(query, k)

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        vectors = self._vectors if self._vectors is not None else np.zeros((0, 0), np.float32)
        tmp = self._index_path.with_suffix(".tmp")
        save_npz(tmp, vectors, self._items)
        os.replace(tmp, self._index_path)  # never leave a half-written index behind
        manifest = {"embedder": self.embedder_name, "sources": self._sources}
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._manifest_path.exists():
            return
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if manifest["sources"] and manifest["embedder"] != self.embedder_name:
            raise StoreError(
                f"{self.data_dir} was built with embedder {manifest['embedder']!r}, "
                f"not {self.embedder_name!r}. Use the same --embedder or another data dir."
            )
        self._sources = manifest["sources"]
        if self._index_path.exists():
            vectors, self._items = load_npz(self._index_path)
            self._vectors = vectors if vectors.size else None
