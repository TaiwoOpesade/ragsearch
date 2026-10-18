"""FAISS-backed index: exact inner-product search on normalised vectors.

``IndexFlatIP`` is still brute force, but FAISS uses optimised SIMD/BLAS
kernels, so it is much faster than NumPy as the index grows. The file format
is the same ``.npz`` as ``VectorIndex``, so saved indexes are interchangeable.
"""

from pathlib import Path

import faiss
import numpy as np

from ragsearch.index import SearchResult, check_add, check_query, load_npz, save_npz


class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._items: list[dict] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, vectors: np.ndarray, items: list[dict]) -> None:
        self._index.add(np.ascontiguousarray(check_add(self.dim, vectors, items)))
        self._items.extend(items)

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        query = check_query(self.dim, query, k)
        if not self._items:
            return []
        scores, ids = self._index.search(query, min(k, len(self._items)))
        return [
            SearchResult(score=float(score), item=self._items[i])
            for score, i in zip(scores[0], ids[0], strict=True)
        ]

    def save(self, path: str | Path) -> None:
        save_npz(path, self._index.reconstruct_n(0, len(self)), self._items)

    @classmethod
    def load(cls, path: str | Path) -> "FaissIndex":
        vectors, items = load_npz(path)
        index = cls(dim=vectors.shape[1])
        if items:
            index.add(vectors, items)
        return index
