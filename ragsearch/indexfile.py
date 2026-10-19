"""The on-disk format of a data dir.

``index.npz`` holds everything in one file, so one atomic write saves it all:
``vectors`` (float32 rows), ``items`` (chunk text and metadata as UTF-8 JSON) and
``manifest`` (the embedder name plus a content hash and chunk count per source).
``manifest.json`` is a human-readable copy that is never trusted on load.
"""

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from ragsearch.fileutil import atomic_write_bytes
from ragsearch.index import decode_json, encode_json, npz_bytes

# Problems with a file's contents (as opposed to not being able to read it at all).
DAMAGE = (ValueError, KeyError, TypeError, zipfile.BadZipFile, EOFError)


def signature(st: os.stat_result) -> tuple:
    """A fingerprint that changes whenever the file is replaced."""
    return st.st_mtime_ns, st.st_size, st.st_ino


def read_bytes(path: Path) -> tuple[bytes, tuple] | None:
    """The file's bytes and the fingerprint of that same file, or None if it is missing."""
    try:
        with open(path, "rb") as f:
            return f.read(), signature(os.fstat(f.fileno()))
    except FileNotFoundError:
        return None


def parse(raw: bytes, manifest_copy: Path) -> tuple[np.ndarray, list[dict], dict]:
    """Vectors, items and manifest from index.npz bytes (raises one of ``DAMAGE``).

    Files written before the manifest moved inside index.npz fall back to the copy.
    """
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        vectors = data["vectors"]
        items = decode_json(data["items"])
        manifest = (decode_json(data["manifest"]) if "manifest" in data.files
                    else json.loads(manifest_copy.read_text(encoding="utf-8")))
    return vectors, items, manifest


def reconcile(sources: dict, items: list[dict]) -> dict:
    """Mark any source whose chunk count disagrees with the items for re-ingesting."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item["source"]] = counts.get(item["source"], 0) + 1
    sources = dict(sources)
    for source in set(counts) | set(sources):
        if sources.get(source, {}).get("chunks") != counts.get(source, 0):
            sources[source] = {"hash": None, "chunks": counts.get(source, 0)}
    return sources


def write(index_path: Path, manifest_copy: Path, vectors: np.ndarray, items: list[dict],
          manifest: dict) -> None:
    atomic_write_bytes(index_path, npz_bytes(
        vectors=vectors, items=encode_json(items), manifest=encode_json(manifest)))
    # The readable copy escapes non-ASCII, so odd file names cannot break it.
    atomic_write_bytes(manifest_copy, json.dumps(manifest, indent=2).encode("ascii"))
