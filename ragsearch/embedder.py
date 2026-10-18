"""Turn text into unit-length vectors.

Two implementations share one interface (``dim`` and ``embed``):

* ``HashEmbedder``: offline bag-of-words hashing. Not truly semantic, but fast
  and deterministic, so it is used for tests and as a fallback.
* ``OllamaEmbedder``: real semantic embeddings from a local Ollama server.
"""

import hashlib
import json
import re
import urllib.error
import urllib.request

import numpy as np


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot produce vectors."""


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


class HashEmbedder:
    """Signed feature hashing over lowercase word tokens."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, bucket] += sign
        return _normalise(vectors)


class OllamaEmbedder:
    """Embeddings from a local Ollama server (``POST /api/embed``)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        timeout: float = 60,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
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

        request = urllib.request.Request(
            f"{self.host}/api/embed",
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as err:
            err.close()  # HTTPError holds an open response; avoid a ResourceWarning
            raise EmbeddingError(f"Ollama returned HTTP {err.code}") from err
        except OSError as err:
            raise EmbeddingError(
                f"Could not reach Ollama at {self.host}. Is it running? ({err})"
            ) from err

        if "embeddings" not in payload:
            raise EmbeddingError("Ollama response had no 'embeddings' field")

        vectors = np.asarray(payload["embeddings"], dtype=np.float32)
        self._dim = vectors.shape[1]
        return _normalise(vectors)
