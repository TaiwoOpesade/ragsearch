import json

import numpy as np
import pytest

from ragsearch.faiss_index import FaissIndex
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


def test_store_can_search_with_the_faiss_backend(tmp_path):
    store = Store(tmp_path, embedder_name="hash-3", index_factory=FaissIndex)
    store.replace("a.md", "h", vecs([1, 0, 0], [0, 1, 0]), items("a.md", 2))
    assert store.search(vecs([0, 1, 0])[0], k=1)[0].item["index"] == 1


# --- removal, atomic saves, damage recovery, two processes ----------------------


def saved(tmp_path, *sources):
    store = Store(tmp_path, embedder_name="hash-3")
    for i, source in enumerate(sources):
        store.replace(source, f"h-{source}", vecs([1, 0, 0])[0:1] * (i + 1), items(source, 1))
    store.save()
    return store


def test_remove_drops_a_source_and_its_chunks(tmp_path):
    store = saved(tmp_path, "a.md", "b.md")
    store.remove("a.md")
    assert set(store.sources) == {"b.md"}
    assert len(store) == 1
    store.save()
    assert set(Store(tmp_path, embedder_name="hash-3").sources) == {"b.md"}


def test_manifest_is_stored_inside_the_index_file(tmp_path):
    saved(tmp_path, "a.md")
    (tmp_path / "manifest.json").unlink()  # the readable copy is optional
    assert Store(tmp_path, embedder_name="hash-3").is_current("a.md", "h-a.md")


def test_lost_index_makes_every_source_re_ingest(tmp_path, caplog):
    import logging

    saved(tmp_path, "a.md")
    (tmp_path / "index.npz").unlink()
    with caplog.at_level(logging.WARNING):
        store = Store(tmp_path, embedder_name="hash-3")
    assert not store.is_current("a.md", "h-a.md")
    assert "index.npz" in caplog.text


def test_corrupt_manifest_copy_is_ignored(tmp_path):
    saved(tmp_path, "a.md")
    (tmp_path / "manifest.json").write_text('{"sources": {"a', encoding="utf-8")
    assert len(Store(tmp_path, embedder_name="hash-3")) == 1


def test_corrupt_index_is_a_clear_store_error(tmp_path):
    saved(tmp_path, "a.md")
    (tmp_path / "index.npz").write_bytes(b"not a zip file")
    with pytest.raises(StoreError, match="damaged"):
        Store(tmp_path, embedder_name="hash-3")


def test_old_format_index_files_still_load(tmp_path):
    old_items = [{"text": "café", "source": "a.md", "index": 0}]
    with open(tmp_path / "index.npz", "wb") as f:
        np.savez_compressed(f, vectors=vecs([1, 0, 0]), items=np.array(json.dumps(old_items)))
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"embedder": "hash-3", "sources": {"a.md": {"hash": "h", "chunks": 1}}}), encoding="utf-8")
    store = Store(tmp_path, embedder_name="hash-3")
    assert store.is_current("a.md", "h")
    assert store.search(vecs([1, 0, 0])[0], k=1)[0].item["text"] == "café"


def test_non_ascii_text_round_trips(tmp_path):
    store = Store(tmp_path, embedder_name="hash-3")
    store.replace("a.md", "h", vecs([1, 0, 0]),
                  [{"text": "naïve café 中文", "source": "a.md", "index": 0}])
    store.save()
    reopened = Store(tmp_path, embedder_name="hash-3")
    assert reopened.search(vecs([1, 0, 0])[0], k=1)[0].item["text"] == "naïve café 中文"


def test_save_leaves_no_temporary_files(tmp_path):
    saved(tmp_path, "a.md")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [".lock", "index.npz", "manifest.json"]  # the lock file stays by design


def test_two_stores_on_one_data_dir_do_not_lose_each_others_sources(tmp_path):
    api = Store(tmp_path, embedder_name="hash-3")  # e.g. the API server, loaded at startup
    cli = Store(tmp_path, embedder_name="hash-3")  # e.g. a CLI ingest in another process
    cli.replace("from-cli.md", "h1", vecs([1, 0, 0]), items("from-cli.md", 1))
    cli.save()
    api.replace("from-api.md", "h2", vecs([0, 1, 0]), items("from-api.md", 1))
    api.save()
    assert set(Store(tmp_path, embedder_name="hash-3").sources) == {"from-cli.md", "from-api.md"}


def test_merge_keeps_this_stores_own_replacement_of_a_shared_source(tmp_path):
    saved(tmp_path, "shared.md")
    first, second = (Store(tmp_path, embedder_name="hash-3") for _ in range(2))
    second.replace("other.md", "h", vecs([0, 0, 1]), items("other.md", 1))
    second.save()
    first.replace("shared.md", "new", vecs([0, 1, 0]), items("shared.md", 1))
    first.save()
    final = Store(tmp_path, embedder_name="hash-3")
    assert final.is_current("shared.md", "new")
    assert "other.md" in final.sources
    assert len(final) == 2


def test_refresh_picks_up_another_process_save(tmp_path):
    api = Store(tmp_path, embedder_name="hash-3")
    saved(tmp_path, "late.md")
    assert "late.md" not in api.sources
    api.refresh()
    assert "late.md" in api.sources
    assert api.search(vecs([1, 0, 0])[0], k=1)[0].item["source"] == "late.md"


# --- concurrency and error reporting (second review) --------------------------------


def save_another(tmp_path, source):
    other = Store(tmp_path, embedder_name="hash-3")
    other.replace(source, "h", vecs([0, 0, 1]), items(source, 1))
    other.save()


def test_a_save_during_a_load_is_never_missed(tmp_path, monkeypatch):
    import threading

    from ragsearch import store as store_module

    saved(tmp_path, "first.md")
    real_load = store_module.np.load
    writer: list[threading.Thread] = []

    def slow_load(*args, **kwargs):
        data = real_load(*args, **kwargs)
        if not writer:  # first load only: another process saves while we read
            writer.append(threading.Thread(target=save_another, args=(tmp_path, "second.md")))
            writer[0].start()
            writer[0].join(timeout=1.0)
        return data

    monkeypatch.setattr(store_module.np, "load", slow_load)
    reader = Store(tmp_path, embedder_name="hash-3")
    monkeypatch.undo()
    writer[0].join(timeout=10)
    reader.refresh()
    assert "second.md" in reader.sources


def test_save_waits_for_a_reader_that_has_the_index_open(tmp_path):
    import threading

    saved(tmp_path, "a.md")
    handle = open(tmp_path / "index.npz", "rb")  # noqa: SIM115 - held open on purpose
    threading.Timer(0.3, handle.close).start()  # on Windows os.replace fails until then
    save_another(tmp_path, "b.md")
    assert set(Store(tmp_path, embedder_name="hash-3").sources) == {"a.md", "b.md"}


def test_unreadable_index_is_not_called_damaged(tmp_path, monkeypatch):
    from ragsearch import store as store_module

    saved(tmp_path, "a.md")

    def denied(path):
        raise PermissionError(13, "Access is denied", str(path))

    monkeypatch.setattr(store_module, "_read_bytes", denied)
    with pytest.raises(StoreError) as exc:
        Store(tmp_path, embedder_name="hash-3")
    assert "could not be read" in str(exc.value)
    assert "Delete" not in str(exc.value)


def test_a_failed_save_is_a_store_error(tmp_path, monkeypatch):
    from ragsearch import store as store_module

    def denied(path, data, **kwargs):
        raise PermissionError(13, "Access is denied", str(path))

    store = Store(tmp_path, embedder_name="hash-3")
    store.replace("a.md", "h", vecs([1, 0, 0]), items("a.md", 1))
    monkeypatch.setattr(store_module, "atomic_write_bytes", denied)
    with pytest.raises(StoreError, match="could not be saved"):
        store.save()


def test_source_names_with_lone_surrogates_still_save(tmp_path):
    name = "caf\udce9.md"  # an undecodable file name, as Linux reports it
    store = Store(tmp_path, embedder_name="hash-3")
    store.replace(name, "h", vecs([1, 0, 0]), items(name, 1))
    store.save()
    assert name in Store(tmp_path, embedder_name="hash-3").sources


def test_reading_a_fresh_data_dir_creates_nothing(tmp_path):
    Store(tmp_path / "new", embedder_name="hash-3")
    assert not (tmp_path / "new").exists()
