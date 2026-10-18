import logging

import pytest

from ragsearch.embedder import HashEmbedder
from ragsearch.loaders import Document
from ragsearch.pipeline import ingest
from ragsearch.retrieval import hybrid_search
from ragsearch.store import Store
from ragsearch.web.budget import CreditBudget
from ragsearch.web.tavily import TavilyClient

WEB_TEXT = "photosynthesis converts light energy into glucose in chloroplasts"


def tavily_results(request):
    return 200, {}, {"results": [
        {"title": "Photosynthesis", "url": "https://web.example/photo", "content": WEB_TEXT,
         "score": 0.9},
        {"title": "Other", "url": "https://web.example/other", "content": "unrelated football",
         "score": 0.2},
    ]}


@pytest.fixture
def emb():
    return HashEmbedder(dim=512)


@pytest.fixture
def store(tmp_path, emb):
    store = Store(tmp_path / "data", embedder_name=emb.name)
    ingest([Document.from_text("stacks.md", "a stack is last in first out")], emb, store)
    return store


@pytest.fixture
def web(stub_server, tmp_path):
    stub_server.routes["/search"] = tavily_results
    budget = CreditBudget(tmp_path / "usage.json", cap=10)
    return TavilyClient(api_key="tvly-test", budget=budget, base_url=stub_server.url)


def test_confident_local_match_does_not_call_the_web(store, emb, web, stub_server):
    hits = hybrid_search("a stack is last in first out", emb, store, web=web, threshold=0.5)
    assert stub_server.requests == []
    assert [(h.source, h.origin) for h in hits] == [("stacks.md", "local")]


def test_weak_local_match_adds_labelled_web_results_ranked_together(store, emb, web,
                                                                     stub_server):
    hits = hybrid_search("how does photosynthesis make glucose", emb, store, k=3, web=web,
                         threshold=0.5)
    assert len(stub_server.requests) == 1
    assert hits[0].source == "https://web.example/photo"
    assert hits[0].origin == "web"
    assert {h.origin for h in hits} == {"local", "web"}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_web_scores_are_our_own_cosine_scores_not_tavilys(store, emb, web):
    hits = hybrid_search(WEB_TEXT, emb, store, web=web, threshold=0.99)
    top = next(h for h in hits if h.source == "https://web.example/photo")
    assert top.score == pytest.approx(1.0, abs=1e-5)


def test_k_limits_combined_results(store, emb, web):
    assert len(hybrid_search("glucose", emb, store, k=2, web=web)) == 2


def test_empty_store_goes_to_the_web(tmp_path, emb, web, stub_server):
    empty = Store(tmp_path / "empty", embedder_name=emb.name)
    hits = hybrid_search("glucose", emb, empty, web=web)
    assert hits and all(h.origin == "web" for h in hits)


def test_without_a_web_client_only_local_results_are_returned(store, emb):
    hits = hybrid_search("glucose", emb, store, web=None)
    assert all(h.origin == "local" for h in hits)


def test_web_results_are_not_persisted_by_default(store, emb, web):
    before = len(store)
    hybrid_search("glucose photosynthesis", emb, store, web=web)
    assert len(store) == before


def test_web_results_are_persisted_when_asked(store, emb, web):
    hybrid_search("glucose photosynthesis", emb, store, web=web, persist=True)
    assert "https://web.example/photo" in store.sources
    reopened = Store(store.data_dir, embedder_name=emb.name)
    assert "https://web.example/photo" in reopened.sources


def test_web_failure_falls_back_to_local_results_with_warning(store, emb, web,
                                                              stub_server, caplog):
    stub_server.routes["/search"] = (500, {}, {"detail": {"error": "boom"}})
    with caplog.at_level(logging.WARNING):
        hits = hybrid_search("glucose", emb, store, web=web)
    assert [h.origin for h in hits] == ["local"]
    assert "web search failed" in caplog.text
