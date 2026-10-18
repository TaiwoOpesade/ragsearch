import json

import numpy as np
import pytest

from ragsearch.store import Store, StoreError


def vecs(*rows):
    return np.array(rows, dtype=np.float32)


def items(source, n):
    return [{"text": f"{source} chunk {i}", "source": source, "index": i} for i in range(n)]


def test_new_store_is_empty(tmp_path):
    store = Store(tmp_path, embedder_name="hash-3")
    assert len(store) == 0
    assert store.search(vecs([1, 0, 0])[0], k=3) == []


def test_replace_adds_chunks_and_records_hash(tmp_path):
    store = Store(tmp_path, embedder_name="hash-3")
    store.replace("a.md", "h1", vecs([1, 0, 0], [0, 1, 0]), items("a.md", 2))
    assert len(store) == 2
    assert store.is_current("a.md", "h1")
    assert not store.is_current("a.md", "h2")
    assert not store.is_current("b.md", "h1")


def test_replace_swaps_old_chunks_of_the_same_source_only(tmp_path):
    store = Store(tmp_path, embedder_name="hash-3")
    store.replace("a.md", "h1", vecs([1, 0, 0], [0, 1, 0]), items("a.md", 2))
    store.replace("b.md", "h1", vecs([0, 0, 1]), items("b.md", 1))

    store.replace("a.md", "h2", vecs([0, 1, 0]), [{"text": "new", "source": "a.md", "index": 0}])

    assert len(store) == 2
    texts = {r.item["text"] for r in store.search(vecs([1, 1, 1])[0], k=10)}
    assert texts == {"new", "b.md chunk 0"}


def test_search_ranks_by_cosine_similarity(tmp_path):
    store = Store(tmp_path, embedder_name="hash-3")
    store.replace("a.md", "h", vecs([1, 0, 0], [0, 1, 0]), items("a.md", 2))
    results = store.search(vecs([0, 1, 0])[0], k=2)
    assert results[0].item["index"] == 1
    assert results[0].score == pytest.approx(1.0)


def test_save_then_reopen_round_trips(tmp_path):
    store = Store(tmp_path / "data", embedder_name="hash-3")
    store.replace("a.md", "h1", vecs([1, 0, 0]), items("a.md", 1))
    store.save()

    assert (tmp_path / "data" / "index.npz").exists()
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert manifest["sources"]["a.md"] == {"hash": "h1", "chunks": 1}

    reopened = Store(tmp_path / "data", embedder_name="hash-3")
    assert len(reopened) == 1
    assert reopened.is_current("a.md", "h1")
    assert reopened.search(vecs([1, 0, 0])[0], k=1)[0].item["source"] == "a.md"


def test_reopening_with_a_different_embedder_raises(tmp_path):
    store = Store(tmp_path, embedder_name="hash-3")
    store.replace("a.md", "h1", vecs([1, 0, 0]), items("a.md", 1))
    store.save()
    with pytest.raises(StoreError, match="hash-3"):
        Store(tmp_path, embedder_name="ollama-nomic-embed-text")


def test_replace_with_wrong_dimension_raises(tmp_path):
    store = Store(tmp_path, embedder_name="hash-3")
    store.replace("a.md", "h1", vecs([1, 0, 0]), items("a.md", 1))
    with pytest.raises(StoreError, match="dim"):
        store.replace("b.md", "h1", vecs([1, 0, 0, 0]), items("b.md", 1))


def test_replace_with_no_chunks_removes_source_but_remembers_hash(tmp_path):
    store = Store(tmp_path, embedder_name="hash-3")
    store.replace("a.md", "h1", vecs([1, 0, 0]), items("a.md", 1))
    store.replace("a.md", "h2", np.zeros((0, 3), dtype=np.float32), [])
    assert len(store) == 0
    assert store.is_current("a.md", "h2")
