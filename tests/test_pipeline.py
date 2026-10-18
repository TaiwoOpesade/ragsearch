from dataclasses import asdict

from ragsearch.chunker import chunk_text
from ragsearch.embedder import HashEmbedder
from ragsearch.index import VectorIndex

DOCS = {
    "stacks.md": (
        "A stack is a last in first out data structure. Items are pushed onto "
        "the top and popped from the top. Stacks are used for undo and recursion."
    ),
    "queues.md": (
        "A queue is a first in first out data structure. Items join at the rear "
        "and leave from the front. Queues are used for scheduling and buffering."
    ),
    "inflation.md": (
        "Inflation is a sustained rise in the general price level. Central banks "
        "raise interest rates to reduce consumer demand and control inflation."
    ),
}


def build_index(embedder):
    index = VectorIndex(dim=embedder.dim)
    for source, text in DOCS.items():
        chunks = chunk_text(text, source=source, size=40, overlap=5)
        index.add(embedder.embed([c.text for c in chunks]), [asdict(c) for c in chunks])
    return index


def top_source(index, embedder, query):
    q = embedder.embed([query])[0]
    return index.search(q, k=1)[0].item["source"]


def test_end_to_end_search_finds_the_right_document():
    emb = HashEmbedder(dim=512)
    index = build_index(emb)
    assert top_source(index, emb, "which structure is last in first out") == "stacks.md"
    assert top_source(index, emb, "how do banks fight rising prices") == "inflation.md"
    assert top_source(index, emb, "first in first out front and rear") == "queues.md"


def test_search_results_carry_chunk_metadata():
    emb = HashEmbedder(dim=512)
    index = build_index(emb)
    item = index.search(emb.embed(["undo and recursion"])[0], k=1)[0].item
    assert set(item) == {"text", "source", "index"}
