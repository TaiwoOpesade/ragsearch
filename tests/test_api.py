import warnings
from pathlib import Path

import pytest

from ragsearch.api import create_app
from ragsearch.config import Settings
from ragsearch.generator import NO_ANSWER, GenerationError
from ragsearch.web import tavily

with warnings.catch_warnings():
    # Starlette 1.9 wants the new `httpx2` package for its TestClient. httpx2 is not on
    # this project's approved dependency list, so accept the (still working) httpx path.
    # Only this exact message is ignored; every other warning still fails the run.
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
    from fastapi.testclient import TestClient

FIXTURES = Path(__file__).parent / "fixtures"


class StubGenerator:
    """Answers from the first source and cites it, recording what it was given."""

    def __init__(self, answer="A stack removes the newest item first [1]."):
        self.answer = answer
        self.calls: list[tuple[str, list]] = []

    def generate(self, question, sources):
        self.calls.append((question, sources))
        return self.answer


class FailingGenerator:
    def generate(self, question, sources):
        raise GenerationError("Could not reach Ollama at http://127.0.0.1:1")


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", embedder="hash", tavily_api_key="tvly-test")


@pytest.fixture
def generator():
    return StubGenerator()


@pytest.fixture
def client(settings, generator):
    return TestClient(create_app(settings=settings, generator=generator))


@pytest.fixture
def ingested(client):
    response = client.post("/ingest", json={"path": str(FIXTURES)})
    assert response.status_code == 200
    return client


def test_health_reports_status_and_index_size(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "chunks": 0, "embedder": "hash-512",
                    "index_backend": "numpy"}


def test_ingest_path_reports_counts(client):
    body = client.post("/ingest", json={"path": str(FIXTURES)}).json()
    assert len(body["updated"]) == 4
    assert body["added"] == body["total_chunks"] > 0
    again = client.post("/ingest", json={"path": str(FIXTURES)}).json()
    assert again["added"] == 0
    assert len(again["skipped"]) == 4


def test_ingest_needs_exactly_one_of_path_or_url(client):
    assert client.post("/ingest", json={}).status_code == 422
    both = client.post("/ingest", json={"path": "a", "url": "https://x.example"})
    assert both.status_code == 422


def test_search_returns_labelled_results(ingested):
    body = ingested.post("/search", json={"query": "last in first out stack", "k": 2}).json()
    first = body["results"][0]
    assert first["source"].endswith("stacks.md")
    assert first["origin"] == "local"
    assert set(first) == {"score", "source", "text", "origin"}
    assert len(body["results"]) == 2


def test_search_validates_input(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "q", "k": 0}).status_code == 422


def test_ask_returns_answer_with_citations(ingested, generator):
    body = ingested.post("/ask", json={"question": "Which item does a stack remove first?",
                                       "k": 3}).json()
    assert body["answer"] == "A stack removes the newest item first [1]."
    [citation] = body["citations"]
    assert citation["number"] == 1
    assert citation["source"].endswith("stacks.md")
    assert citation["origin"] == "local"
    assert [r["number"] for r in body["results"]] == [1, 2, 3]
    question, sources = generator.calls[0]
    assert question == "Which item does a stack remove first?"
    assert len(sources) == 3


def test_ask_with_empty_index_says_it_does_not_know_without_calling_the_model(client,
                                                                              generator):
    body = client.post("/ask", json={"question": "anything?"}).json()
    assert body["answer"] == NO_ANSWER
    assert body["citations"] == []
    assert generator.calls == []


def test_ask_reports_unavailable_model_as_503(settings):
    app = create_app(settings=settings, generator=FailingGenerator())
    client = TestClient(app)
    client.post("/ingest", json={"path": str(FIXTURES)})
    response = client.post("/ask", json={"question": "what is a stack"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_ask_with_web_cites_web_sources(stub_server, monkeypatch, settings, generator):
    monkeypatch.setattr(tavily, "BASE_URL", stub_server.url)
    stub_server.routes["/search"] = lambda r: (200, {}, {"results": [
        {"title": "Heaps", "url": "https://web.example/heaps",
         "content": "A binary heap keeps the smallest item at the root.", "score": 0.9},
    ]})
    client = TestClient(create_app(settings=settings, generator=generator))
    body = client.post("/ask", json={"question": "binary heap smallest item root",
                                     "web": True}).json()
    assert body["citations"] == [{"number": 1, "source": "https://web.example/heaps",
                                  "origin": "web"}]


def test_budget_exhaustion_is_reported_as_429(stub_server, monkeypatch, tmp_path):
    monkeypatch.setattr(tavily, "BASE_URL", stub_server.url)
    settings = Settings(data_dir=tmp_path / "data", embedder="hash",
                        tavily_api_key="tvly-test", tavily_monthly_cap=0)
    client = TestClient(create_app(settings=settings, generator=StubGenerator()))
    response = client.post("/ingest", json={"url": "https://x.example", "provider": "tavily"})
    assert response.status_code == 429
    assert stub_server.requests == []
