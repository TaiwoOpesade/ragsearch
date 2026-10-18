import logging
from pathlib import Path

import numpy as np

from ragsearch.embedder import HashEmbedder
from ragsearch.loaders import Document, load_documents
from ragsearch.pipeline import ingest, search
from ragsearch.store import Store

FIXTURES = Path(__file__).parent / "fixtures"


class BatchRecordingEmbedder(HashEmbedder):
    """A real HashEmbedder that also remembers the size of every batch."""

    def __init__(self, dim: int = 512):
        super().__init__(dim)
        self.batches: list[int] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.batches.append(len(texts))
        return super().embed(texts)


def make_store(tmp_path, embedder):
    return Store(tmp_path / "data", embedder_name=embedder.name)


def top_name(query, embedder, store):
    return search(query, embedder, store, k=1)[0].item["source"].rsplit("/", 1)[-1]


def test_ingested_fixtures_are_searchable(tmp_path):
    emb = HashEmbedder(dim=512)
    store = make_store(tmp_path, emb)
    report = ingest(load_documents(FIXTURES), emb, store)

    assert len(report.updated) == 4
    assert report.added == len(store) > 0
    assert top_name("last in first out pushed popped", emb, store) == "stacks.md"
    assert top_name("central banks interest rates", emb, store) == "inflation.pdf"
    assert top_name("chlorophyll light glucose", emb, store) == "photosynthesis.md"


def test_reingesting_unchanged_sources_adds_zero_chunks(tmp_path):
    emb = HashEmbedder(dim=512)
    ingest(load_documents(FIXTURES), emb, make_store(tmp_path, emb))

    store = make_store(tmp_path, emb)  # fresh store, reloaded from disk
    before = len(store)
    report = ingest(load_documents(FIXTURES), emb, store)

    assert report.added == 0
    assert report.updated == []
    assert len(report.skipped) == 4
    assert len(store) == before


def test_changed_source_replaces_its_old_chunks(tmp_path):
    emb = HashEmbedder(dim=512)
    store = make_store(tmp_path, emb)
    ingest([Document.from_text("a.md", "old words about stacks")], emb, store)
    ingest([Document.from_text("b.md", "queues")], emb, store)

    report = ingest([Document.from_text("a.md", "brand new text about photosynthesis")], emb, store)

    assert report.updated == ["a.md"]
    assert len(store) == 2
    texts = {r.item["text"] for r in search("anything", emb, store, k=10)}
    assert texts == {"brand new text about photosynthesis", "queues"}


def test_unreadable_source_is_skipped_with_warning(tmp_path, caplog):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "good.txt").write_text("readable text", encoding="utf-8")
    (tmp_path / "docs" / "bad.pdf").write_bytes(b"not a pdf")
    emb = HashEmbedder(dim=64)
    store = make_store(tmp_path, emb)

    with caplog.at_level(logging.WARNING):
        report = ingest(load_documents(tmp_path / "docs"), emb, store)

    assert [s.rsplit("/", 1)[-1] for s in report.failed] == ["bad.pdf"]
    assert [s.rsplit("/", 1)[-1] for s in report.updated] == ["good.txt"]
    assert "bad.pdf" in caplog.text


def test_embeds_in_batches_of_at_most_32(tmp_path):
    emb = BatchRecordingEmbedder()
    text = " ".join(f"w{i}" for i in range(100 * 10))  # 100 chunks of 10 words
    ingest([Document.from_text("big.txt", text)], emb, make_store(tmp_path, emb),
           chunk_size=10, overlap=0)
    assert emb.batches == [32, 32, 32, 4]


def test_chunk_indexes_continue_across_segments(tmp_path):
    emb = HashEmbedder(dim=64)
    store = make_store(tmp_path, emb)
    doc = Document("two.txt", lambda: iter(["a b c d", "e f g h"]))
    ingest([doc], emb, store, chunk_size=2, overlap=0)
    indexes = sorted(r.item["index"] for r in search("a", emb, store, k=10))
    assert indexes == [0, 1, 2, 3]


def test_ingest_saves_so_a_new_store_sees_the_chunks(tmp_path):
    emb = HashEmbedder(dim=64)
    ingest([Document.from_text("a.md", "saved text")], emb, make_store(tmp_path, emb))
    assert len(make_store(tmp_path, emb)) == 1


# --- cross-segment chunks, broken text, pruning deleted files -------------------


def test_chunks_flow_across_segment_boundaries(tmp_path):
    emb = HashEmbedder(dim=64)
    store = make_store(tmp_path, emb)
    doc = Document("two.txt", lambda: iter(["a b c", "d e"]))
    ingest([doc], emb, store, chunk_size=4, overlap=1)
    texts = sorted(r.item["text"] for r in search("a", emb, store, k=10))
    assert texts == ["a b c d", "d e"]


def test_document_with_broken_unicode_fails_alone(tmp_path):
    emb = HashEmbedder(dim=64)
    store = make_store(tmp_path, emb)
    report = ingest([Document.from_text("bad", "half a pair \ud835"),
                     Document.from_text("good", "fine text")], emb, store)
    assert report.failed == ["bad"]
    assert report.updated == ["good"]


def test_search_embeds_the_query_with_embed_query(tmp_path):
    class QueryRecorder(HashEmbedder):
        def __init__(self, dim):
            super().__init__(dim)
            self.queries: list[str] = []

        def embed_query(self, text):
            self.queries.append(text)
            return super().embed_query(text)

    emb = QueryRecorder(dim=64)
    store = make_store(tmp_path, emb)
    ingest([Document.from_text("a.md", "stacks")], emb, store)
    search("what is a stack", emb, store)
    assert emb.queries == ["what is a stack"]


def write(folder, name, text):
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_deleted_file_is_pruned_when_its_folder_is_re_ingested(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    write(notes, "draft.md", "exam is Monday in room 12")
    write(notes, "keep.md", "stacks are LIFO")
    emb = HashEmbedder(dim=64)
    store = make_store(tmp_path, emb)
    ingest(load_documents(notes), emb, store, prune_under=notes)

    (notes / "draft.md").unlink()
    report = ingest(load_documents(notes), emb, store, prune_under=notes)

    assert [s.rsplit("/", 1)[-1] for s in report.removed] == ["draft.md"]
    assert [s.rsplit("/", 1)[-1] for s in store.sources] == ["keep.md"]
    assert all("room 12" not in r.item["text"] for r in search("exam", emb, store, k=10))


def test_renamed_file_does_not_leave_a_stale_copy(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    write(notes, "draft.md", "exam is Monday in room 12")
    emb = HashEmbedder(dim=64)
    store = make_store(tmp_path, emb)
    ingest(load_documents(notes), emb, store, prune_under=notes)

    (notes / "draft.md").rename(notes / "final.md")
    ingest(load_documents(notes), emb, store, prune_under=notes)
    assert [s.rsplit("/", 1)[-1] for s in store.sources] == ["final.md"]
    assert len(store) == 1


def test_pruning_never_touches_other_folders_web_pages_or_failed_files(tmp_path):
    notes, other = tmp_path / "notes", tmp_path / "notes-archive"
    notes.mkdir()
    other.mkdir()
    write(other, "old.md", "archived")
    bad = notes / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    emb = HashEmbedder(dim=64)
    store = make_store(tmp_path, emb)
    ingest(load_documents(other), emb, store)
    ingest([Document.from_text("https://web.example/page", "a web page")], emb, store)
    store.replace(bad.resolve().as_posix(), "old", emb.embed(["old pdf text"]),
                  [{"text": "old pdf text", "source": bad.resolve().as_posix(), "index": 0}])

    report = ingest(load_documents(notes), emb, store, prune_under=notes)

    assert report.removed == []
    assert len(store.sources) == 3  # the archive file, the web page and the failed PDF
