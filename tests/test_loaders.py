import logging

import pytest

from ragsearch.loaders import Document, LoaderError, clean_text, load_documents


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
    path.write_bytes(b"\xef\xbb\xbfhello")  # UTF-8 byte order mark + text
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


# --- end-of-file handling, encodings, PDF text cleaning ------------------------


def test_file_without_trailing_newline_keeps_its_last_word(tmp_path):
    path = tmp_path / "note.txt"
    path.write_bytes(b"The password for the lab wifi is tangerine")  # no final newline
    [doc] = load_documents(path)
    assert list(doc.segments()) == ["The password for the lab wifi is tangerine"]


def test_small_segments_lose_no_words_without_a_final_newline(tmp_path):
    words = [f"w{i}" for i in range(40)]
    path = tmp_path / "n.txt"
    path.write_text(" ".join(words), encoding="utf-8")  # no final newline
    [doc] = load_documents(path, segment_chars=50)
    segments = list(doc.segments())
    assert " ".join(segments).split() == words
    assert all(len(s) <= 50 for s in segments)


def test_windows_1252_text_is_read_with_a_warning(tmp_path, caplog):
    path = tmp_path / "ansi.txt"
    path.write_bytes("Café notes – smart “quotes”".encode("cp1252"))
    [doc] = load_documents(path)
    with caplog.at_level(logging.WARNING):
        assert read_all(doc) == "Café notes – smart “quotes”"
    assert "Windows-1252" in caplog.text


def test_binary_file_with_a_text_extension_is_rejected(tmp_path):
    path = tmp_path / "image.txt"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    [doc] = load_documents(path)
    with pytest.raises(LoaderError, match="image.txt"):
        read_all(doc)


def test_clean_text_replaces_lone_surrogates():
    assert clean_text("integral \ud835 over x") == "integral � over x"
    assert clean_text("plain") == "plain"


def test_file_names_that_are_not_valid_unicode_are_skipped(tmp_path, monkeypatch, caplog):
    # On Linux an undecodable name arrives as a lone surrogate (e.g. "caf\udce9.md").
    # Such a name cannot be stored or shown, so the file is skipped with a warning.
    from ragsearch import loaders

    good = tmp_path / "good.md"
    good.write_text("fine", encoding="utf-8")

    class UndecodableName:
        suffix = ".md"

        def resolve(self):
            return self

        def as_posix(self):
            return str(tmp_path.as_posix()) + "/caf\udce9.md"

    monkeypatch.setattr(loaders, "_find_files", lambda path: [good, UndecodableName()])
    with caplog.at_level(logging.WARNING):
        docs = list(load_documents(tmp_path))
    assert [d.source.rsplit("/", 1)[-1] for d in docs] == ["good.md"]
    assert "not valid Unicode" in caplog.text
    assert "caf\\udce9.md" in caplog.text  # shown escaped...
    assert "\udce9" not in caplog.text  # ...never as a raw surrogate
