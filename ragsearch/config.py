"""Settings read from environment variables, with defaults that suit small local models."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    ollama_host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    gen_model: str = "llama3.2:3b"
    data_dir: Path = Path("data")
    index_backend: str = "numpy"
    # repr=False keeps the key out of logs, tracebacks and print(settings).
    tavily_api_key: str | None = field(default=None, repr=False)


def _with_scheme(host: str) -> str:
    return host if "://" in host else f"http://{host}"


INDEX_BACKENDS = ("numpy", "faiss")


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build ``Settings`` from ``env`` (defaults to ``os.environ``)."""
    env = os.environ if env is None else env
    default = Settings()
    backend = env.get("INDEX_BACKEND", default.index_backend).lower()
    if backend not in INDEX_BACKENDS:
        raise ValueError(f"INDEX_BACKEND must be one of {INDEX_BACKENDS}, got {backend!r}")
    return Settings(
        ollama_host=_with_scheme(env.get("OLLAMA_HOST", default.ollama_host)),
        embed_model=env.get("EMBED_MODEL", default.embed_model),
        gen_model=env.get("GEN_MODEL", default.gen_model),
        data_dir=Path(env.get("DATA_DIR", str(default.data_dir))),
        index_backend=backend,
        tavily_api_key=env.get("TAVILY_API_KEY") or None,
    )
