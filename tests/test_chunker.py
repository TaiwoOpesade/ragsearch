import pytest

from ragsearch.chunker import Chunk, chunk_text


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n  ", source="a.md") == []


def test_short_text_is_one_chunk():
    chunks = chunk_text("hello world", source="a.md", size=10, overlap=2)
    assert chunks == [Chunk(text="hello world", source="a.md", index=0)]


def test_long_text_splits_into_overlapping_chunks():
    chunks = chunk_text(words(10), source="a.md", size=4, overlap=1)
    assert [c.text for c in chunks] == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
    ]


def test_chunk_indexes_are_sequential():
    chunks = chunk_text(words(10), source="a.md", size=4, overlap=1)
    assert [c.index for c in chunks] == [0, 1, 2]


def test_last_chunk_covers_final_word_without_trailing_duplicate():
    chunks = chunk_text(words(8), source="a.md", size=4, overlap=1)
    assert chunks[-1].text.endswith("w7")
    assert len(chunks) == 3  # w0-3, w3-6, w6-7


def test_whitespace_is_normalised():
    chunks = chunk_text("a\n\nb\t c", source="x", size=10, overlap=0)
    assert chunks[0].text == "a b c"


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (4, 4), (4, 5), (4, -1)])
def test_invalid_size_or_overlap_raises(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c d e", source="x", size=size, overlap=overlap)
