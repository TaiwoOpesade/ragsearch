"""Vector indexes behind one ``Index`` protocol.

``VectorIndex`` is brute-force cosine similarity in NumPy. ``FaissIndex`` (in
``faiss_index.py``) does the same search with FAISS. Vectors are normalised on
the way in, so cosine similarity is just a dot product.
"""

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    score: float
    item: dict


@runtime_checkable
class Index(Protocol):
    dim: int

    def __len__(self) -> int: ...

    def add(self, vectors: np.ndarray, items: list[dict]) -> None: ...

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> "Index": ...


def normalise_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def check_add(dim: int, vectors: np.ndarray, items: list[dict]) -> np.ndarray:
    """Validate an ``add`` call and return normalised float32 rows."""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    if vectors.shape[1] != dim:
        raise ValueError(f"Expected vectors with dim={dim}, got {vectors.shape[1]}")
    if vectors.shape[0] != len(items):
        raise ValueError(f"Got {vectors.shape[0]} vectors but {len(items)} items")
    return normalise_rows(vectors)


def check_query(dim: int, query: np.ndarray, k: int) -> np.ndarray:
    """Validate a ``search`` call and return the normalised query as one row."""
    if k <= 0:
        raise ValueError("k must be positive")
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    if query.shape[0] != dim:
        raise ValueError(f"Expected query with dim={dim}, got {query.shape[0]}")
    return normalise_rows(query[None, :])


def encode_json(value: object) -> np.ndarray:
    """JSON as a UTF-8 byte array: about 4x smaller than a NumPy unicode string."""
    data = json.dumps(value, ensure_ascii=False).encode("utf-8")
    return np.frombuffer(data, dtype=np.uint8)


def decode_json(array: np.ndarray) -> object:
    if array.dtype.kind == "U":  # files written before the UTF-8 format
        return json.loads(array.item())
    return json.loads(array.tobytes().decode("utf-8"))


def npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def save_npz(path: str | Path, vectors: np.ndarray, items: list[dict]) -> None:
    Path(path).write_bytes(npz_bytes(vectors=vectors, items=encode_json(items)))


def load_npz(path: str | Path) -> tuple[np.ndarray, list[dict]]:
    with np.load(path, allow_pickle=False) as data:
        return data["vectors"], decode_json(data["items"])


def index_backend(name: str) -> type[Index]:
    """Look up an index class by its config name: ``"numpy"`` or ``"faiss"``."""
    if name == "numpy":
        return VectorIndex
    if name == "faiss":
        from ragsearch.faiss_index import FaissIndex  # only import FAISS when used

        return FaissIndex
    raise ValueError(f"unknown index backend {name!r}; expected 'numpy' or 'faiss'")


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._items: list[dict] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, vectors: np.ndarray, items: list[dict]) -> None:
        self._vectors = np.vstack([self._vectors, check_add(self.dim, vectors, items)])
        self._items.extend(items)

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        query = check_query(self.dim, query, k)
        if not self._items:
            return []

        scores = self._vectors @ query[0]
        k = min(k, len(self._items))
        top = np.argsort(-scores, kind="stable")[:k]
        return [SearchResult(score=float(scores[i]), item=self._items[i]) for i in top]

    def save(self, path: str | Path) -> None:
        save_npz(path, self._vectors, self._items)

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        vectors, items = load_npz(path)
        index = cls(dim=vectors.shape[1])
        index._vectors = vectors
        index._items = items
        return index
