"""In-memory vector index using brute-force cosine similarity (NumPy only).

Vectors are normalised on the way in, so cosine similarity is just a dot
product. Stage 3 swaps this for FAISS behind the same interface.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    score: float
    item: dict


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._items: list[dict] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, vectors: np.ndarray, items: list[dict]) -> None:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors with dim={self.dim}, got {vectors.shape[1]}")
        if vectors.shape[0] != len(items):
            raise ValueError(f"Got {vectors.shape[0]} vectors but {len(items)} items")
        self._vectors = np.vstack([self._vectors, self._normalise(vectors)])
        self._items.extend(items)

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"Expected query with dim={self.dim}, got {query.shape[0]}")
        if not self._items:
            return []

        scores = self._vectors @ self._normalise(query[None, :])[0]
        k = min(k, len(self._items))
        top = np.argsort(-scores, kind="stable")[:k]
        return [SearchResult(score=float(scores[i]), item=self._items[i]) for i in top]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            np.savez_compressed(
                f, vectors=self._vectors, items=np.array(json.dumps(self._items))
            )

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        with np.load(path, allow_pickle=False) as data:
            vectors = data["vectors"]
            items = json.loads(data["items"].item())
        index = cls(dim=vectors.shape[1])
        index._vectors = vectors
        index._items = items
        return index

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
