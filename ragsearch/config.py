"""Settings read from environment variables (and an optional ``.env`` file), with
defaults that suit small local models.

Every value is stripped, an empty value means "use the default", and a bad value
raises ``ConfigError`` naming the variable, so a typo never becomes a traceback.
"""

import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlsplit, urlunsplit

T = TypeVar("T")

_INLINE_COMMENT = re.compile(r"\s#")
# A host name or IPv4 address (optionally "*." in front), or a bracketed IPv6 address.
_HOST_ENTRY = re.compile(r"^(\*\.)?[A-Za-z0-9.-]+$|^\[[0-9A-Fa-f:.]+\]$")

INDEX_BACKENDS = ("numpy", "faiss")
EMBEDDERS = ("ollama", "hash")


class ConfigError(ValueError):
    """Raised for a setting that cannot be used, with the variable's name."""


@dataclass(frozen=True)
class Settings:
    ollama_host: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text"
    gen_model: str = "llama3.2:3b"
    data_dir: Path = Path("data")
    embedder: str = "ollama"
    index_backend: str = "numpy"
    tavily_monthly_cap: int = 800
    # Measured with nomic-embed-text: on-topic best scores 0.61-0.81, off-topic 0.46-0.49.
    web_fallback_threshold: float = 0.55
    # Host names the API answers to; anything else gets HTTP 400 (blocks DNS rebinding).
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost")
    # repr=False keeps the key out of logs, tracebacks and print(settings).
    tavily_api_key: str | None = field(default=None, repr=False)


def _ollama_url(host: str) -> str:
    """Add a scheme, and use 127.0.0.1 for localhost.

    On Windows "localhost" resolves to IPv6 ``::1`` first. Ollama listens on IPv4
    only, so every request waited ~5 s for that attempt to fail before falling back
    (measured: ingest 9.6 s via localhost vs 1.4 s via 127.0.0.1).
    """
    url = host if "://" in host else f"http://{host}"
    parts = urlsplit(url)
    if (parts.hostname or "").lower() == "localhost":
        port = f":{parts.port}" if parts.port else ""
        url = urlunsplit(parts._replace(netloc=f"127.0.0.1{port}"))
    return url


def _decode_dotenv(raw: bytes, path: Path) -> str:
    try:
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):  # PowerShell 5.1 '>' writes UTF-16
            return raw.decode("utf-16")
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as err:
        raise ConfigError(f"{path} is not UTF-8 text; save it as UTF-8") from err


def read_dotenv(path: str | Path) -> dict[str, str]:
    """Parse simple ``KEY=value`` lines from a ``.env`` file (missing file -> ``{}``).

    Blank lines, ``#`` comments, lines without ``=`` and empty values are skipped.
    A quoted value ends at its closing quote (anything after it, such as a
    comment, is ignored); an unquoted value ends at whitespace followed by ``#``.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in _decode_dotenv(path.read_bytes(), path).splitlines():
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or key.startswith("#"):
            continue
        quote = value[:1]
        if quote in ("'", '"') and value.find(quote, 1) > 0:
            value = value[1 : value.find(quote, 1)]
        else:
            value = _INLINE_COMMENT.split(value, maxsplit=1)[0].rstrip()
        if value:
            values[key] = value
    return values


def _setting(env: Mapping[str, str], name: str, default: T, parse: Callable[[str], T]) -> T:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return parse(raw)
    except ValueError as err:
        raise ConfigError(f"{name}={raw!r} is not valid: {err}") from None


def _choice(options: tuple[str, ...]) -> Callable[[str], str]:
    def parse(raw: str) -> str:
        if raw.lower() not in options:
            raise ValueError(f"expected one of {', '.join(options)}")
        return raw.lower()

    return parse


def _cap(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise ValueError("must be 0 or more")
    return value


def _hosts(raw: str) -> tuple[str, ...]:
    hosts = tuple(h.strip() for h in raw.split(",") if h.strip())
    if not hosts:
        raise ValueError("list at least one host name, separated by commas")
    for host in hosts:
        if not _HOST_ENTRY.match(host):
            raise ValueError(f"{host!r} is not usable; write host names or IP addresses, "
                             "[::1] for IPv6, and wildcards only as a leading *.")
    return hosts


def _threshold(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("must be a finite number")
    return value


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build ``Settings`` from ``env``.

    By default that is ``.env`` in the current directory overlaid with the real
    environment, so a variable set in your shell always wins over the file.
    """
    if env is None:
        env = {**read_dotenv(".env"), **os.environ}
    d = Settings()
    return Settings(
        ollama_host=_ollama_url(_setting(env, "OLLAMA_HOST", d.ollama_host, str)),
        embed_model=_setting(env, "EMBED_MODEL", d.embed_model, str),
        gen_model=_setting(env, "GEN_MODEL", d.gen_model, str),
        data_dir=_setting(env, "DATA_DIR", d.data_dir, Path),
        embedder=_setting(env, "EMBEDDER", d.embedder, _choice(EMBEDDERS)),
        index_backend=_setting(env, "INDEX_BACKEND", d.index_backend, _choice(INDEX_BACKENDS)),
        tavily_monthly_cap=_setting(env, "TAVILY_MONTHLY_CAP", d.tavily_monthly_cap, _cap),
        web_fallback_threshold=_setting(
            env, "WEB_FALLBACK_THRESHOLD", d.web_fallback_threshold, _threshold
        ),
        allowed_hosts=_setting(env, "ALLOWED_HOSTS", d.allowed_hosts, _hosts),
        tavily_api_key=_setting(env, "TAVILY_API_KEY", None, str),
    )
