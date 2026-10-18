import logging

import pytest

from ragsearch.loaders import Document, LoaderError, load_documents


def read_all(doc: Document) -> str:
    return " ".join(doc.segments())


def test_directory_is_loaded_recursively_in_sorted_order(tmp_path):
    (tmp_path / "b.txt").write_text("bravo text", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_text("# alpha notes", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("x,y", encoding="utf-8")

    docs = list(load_documents(tmp_path))

    assert [d.source for d in docs] == sorted(d.source for d in docs)
    assert {d.source.rsplit("/", 1)[-1] for d in docs} == {"b.txt", "a.md"}
    texts = {d.source.rsplit("/", 1)[-1]: read_all(d) for d in docs}
    assert texts == {"b.txt": "bravo text", "a.md": "# alpha notes"}


def test_single_file_source_is_absolute_posix_path(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("hello", encoding="utf-8")
    [doc] = load_documents(path)
    assert doc.source == path.resolve().as_posix()


def test_utf8_bom_is_stripped(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes("﻿hello".encode())
    [doc] = load_documents(path)
    assert read_all(doc) == "hello"


def test_pdf_text_is_read_one_page_at_a_time(tmp_path, make_pdf):
    path = make_pdf(tmp_path / "doc.pdf", ["first page words", "second page words"])
    [doc] = load_documents(path)
    segments = list(doc.segments())
    assert len(segments) == 2
    assert "first page" in segments[0]
    assert "second page" in segments[1]


def test_large_text_is_split_into_bounded_segments_on_word_boundaries(tmp_path):
    words = [f"word{i}" for i in range(500)]
    path = tmp_path / "big.txt"
    path.write_text(" ".join(words), encoding="utf-8")

    [doc] = load_documents(path, segment_chars=100)
    segments = list(doc.segments())

    assert len(segments) > 1
    assert all(len(s) <= 100 for s in segments)
    assert " ".join(segments).split() == words


def test_segments_can_be_read_more_than_once(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("same text", encoding="utf-8")
    [doc] = load_documents(path)
    assert read_all(doc) == read_all(doc)


def test_undecodable_text_file_raises_loader_error_when_read(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe\xfa not utf8 \x80")
    [doc] = load_documents(path)
    with pytest.raises(LoaderError, match="bad.txt"):
        read_all(doc)


def test_corrupt_pdf_raises_loader_error_when_read(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")
    [doc] = load_documents(path)
    with pytest.raises(LoaderError, match="broken.pdf"):
        read_all(doc)


def test_missing_path_warns_and_yields_nothing(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        docs = list(load_documents(tmp_path / "nope"))
    assert docs == []
    assert "does not exist" in caplog.text


def test_unsupported_single_file_warns_and_yields_nothing(tmp_path, caplog):
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert list(load_documents(path)) == []
    assert "unsupported" in caplog.text


def test_document_from_text_wraps_in_memory_text():
    doc = Document.from_text("https://example.com", "page text")
    assert doc.source == "https://example.com"
    assert list(doc.segments()) == ["page text"]
