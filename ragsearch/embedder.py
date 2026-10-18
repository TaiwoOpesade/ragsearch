"""Turn text into unit-length vectors.

Two implementations share one interface (``name``, ``dim``, ``embed`` for
documents and ``embed_query`` for search queries):

* ``HashEmbedder``: offline bag-of-words hashing. Not truly semantic, but fast
  and deterministic, so it is used for tests and as a fallback.
* ``OllamaEmbedder``: real semantic embeddings from a local Ollama server.
"""

import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Anything with a ``name``, a vector ``dim``, ``embed`` and ``embed_query``."""

    name: str

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed documents (chunks to be indexed) as rows of unit vectors."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one search query as a unit vector."""
        ...


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot produce vectors."""


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


class HashEmbedder:
    """Signed feature hashing over lowercase word tokens."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, bucket] += sign
        return _normalise(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# Models trained with task prefixes: (document prefix, query prefix).
TASK_PREFIXES = {"nomic-embed-text": ("search_document: ", "search_query: ")}


class OllamaEmbedder:
    """Embeddings from a local Ollama server (``POST /api/embed``).

    Some models expect a task prefix on every input (``nomic-embed-text`` was
    trained with ``search_document:`` / ``search_query:``); ``make_embedder``
    sets these from ``TASK_PREFIXES``.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        timeout: float = 60,
        document_prefix: str = "",
        query_prefix: str = "",
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.name = f"ollama-{model}"
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed(["dimension probe"])
        assert self._dim is not None
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._request([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._request([self.query_prefix + text])[0]

    def _request(self, inputs: list[str]) -> np.ndarray:
        request = urllib.request.Request(
            f"{self.host}/api/embed",
            data=json.dumps({"model": self.model, "input": inputs}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as err:
            detail = _error_detail(err)
            err.close()  # HTTPError holds an open response; avoid a ResourceWarning
            raise EmbeddingError(f"Ollama returned HTTP {err.code}{detail}") from err
        except ValueError as err:  # not JSON: probably not an Ollama server
            raise EmbeddingError(
                f"Ollama at {self.host} sent an unreadable reply; is OLLAMA_HOST right?"
            ) from err
        except OSError as err:
            raise EmbeddingError(
                f"Could not reach Ollama at {self.host}. Is it running? ({err})"
            ) from err

        if not isinstance(payload, dict) or "embeddings" not in payload:
            raise EmbeddingError("Ollama response had no 'embeddings' field")
        try:
            vectors = np.asarray(payload["embeddings"], dtype=np.float32)
        except (TypeError, ValueError) as err:
            raise EmbeddingError("Ollama returned malformed embeddings") from err
        if vectors.ndim != 2 or vectors.shape[0] != len(inputs):
            raise EmbeddingError(
                f"Ollama returned {vectors.shape[0] if vectors.ndim else 0} vectors "
                f"for {len(inputs)} texts"
            )
        self._dim = vectors.shape[1]
        return _normalise(vectors)


def _error_detail(err: urllib.error.HTTPError) -> str:
    try:
        message = json.loads(err.read()).get("error")
    except (ValueError, AttributeError, OSError):
        return ""
    return f": {message}" if message else ""


def make_embedder(
    kind: str,
    host: str = "http://localhost:11434",
    model: str = "nomic-embed-text",
    hash_dim: int = 512,
) -> HashEmbedder | OllamaEmbedder:
    """Build an embedder by name: ``"hash"`` (offline) or ``"ollama"``."""
    if kind == "hash":
        return HashEmbedder(dim=hash_dim)
    if kind == "ollama":
        document_prefix, query_prefix = TASK_PREFIXES.get(model.split(":")[0], ("", ""))
        return OllamaEmbedder(model=model, host=host, document_prefix=document_prefix,
                              query_prefix=query_prefix)
    raise ValueError(f"unknown embedder {kind!r}; expected 'hash' or 'ollama'")
