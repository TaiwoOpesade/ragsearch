"""Settings read from environment variables (and an optional ``.env`` file), with
defaults that suit small local models."""

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
    embedder: str = "ollama"
    index_backend: str = "numpy"
    tavily_monthly_cap: int = 800
    web_fallback_threshold: float = 0.5
    # repr=False keeps the key out of logs, tracebacks and print(settings).
    tavily_api_key: str | None = field(default=None, repr=False)


def _with_scheme(host: str) -> str:
    return host if "://" in host else f"http://{host}"


INDEX_BACKENDS = ("numpy", "faiss")
EMBEDDERS = ("ollama", "hash")


def read_dotenv(path: str | Path) -> dict[str, str]:
    """Parse simple ``KEY=value`` lines from a ``.env`` file (missing file -> ``{}``).

    Blank lines, ``#`` comments, lines without ``=`` and empty values are skipped;
    matching single or double quotes around a value are removed.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or key.startswith("#"):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value:
            values[key] = value
    return values


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build ``Settings`` from ``env``.

    By default that is ``.env`` in the current directory overlaid with the real
    environment, so a variable set in your shell always wins over the file.
    """
    if env is None:
        env = {**read_dotenv(".env"), **os.environ}
    default = Settings()
    backend = env.get("INDEX_BACKEND", default.index_backend).lower()
    if backend not in INDEX_BACKENDS:
        raise ValueError(f"INDEX_BACKEND must be one of {INDEX_BACKENDS}, got {backend!r}")
    embedder = env.get("EMBEDDER", default.embedder).lower()
    if embedder not in EMBEDDERS:
        raise ValueError(f"EMBEDDER must be one of {EMBEDDERS}, got {embedder!r}")
    return Settings(
        ollama_host=_with_scheme(env.get("OLLAMA_HOST", default.ollama_host)),
        embed_model=env.get("EMBED_MODEL", default.embed_model),
        gen_model=env.get("GEN_MODEL", default.gen_model),
        data_dir=Path(env.get("DATA_DIR", str(default.data_dir))),
        embedder=embedder,
        index_backend=backend,
        tavily_monthly_cap=int(env.get("TAVILY_MONTHLY_CAP", default.tavily_monthly_cap)),
        web_fallback_threshold=float(
            env.get("WEB_FALLBACK_THRESHOLD", default.web_fallback_threshold)
        ),
        tavily_api_key=env.get("TAVILY_API_KEY") or None,
    )
