"""HTTP API (FastAPI). Run with:

    uvicorn --factory ragsearch.api:create_app

Endpoints: GET /health, POST /ingest, POST /search, POST /ask.
"""

import threading
from dataclasses import asdict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ragsearch.config import Settings, load_settings
from ragsearch.embedder import EmbeddingError, make_embedder
from ragsearch.generator import (
    NO_ANSWER,
    GenerationError,
    Generator,
    OllamaGenerator,
    cited_numbers,
)
from ragsearch.index import index_backend
from ragsearch.loaders import load_documents
from ragsearch.pipeline import ingest
from ragsearch.retrieval import Hit, hybrid_search
from ragsearch.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    Health,
    IngestRequest,
    IngestResponse,
    NumberedHit,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from ragsearch.store import Store, StoreError
from ragsearch.web.budget import BudgetExceeded
from ragsearch.web.fetcher import FetchError
from ragsearch.web.sources import make_tavily, web_documents
from ragsearch.web.tavily import TavilyError

# Which HTTP status each domain error becomes.
ERROR_STATUS: dict[type[Exception], int] = {
    EmbeddingError: 503,
    GenerationError: 503,
    TavilyError: 502,
    BudgetExceeded: 429,
    FetchError: 422,
    StoreError: 409,
}


def create_app(settings: Settings | None = None, generator: Generator | None = None) -> FastAPI:
    settings = settings or load_settings()
    embedder = make_embedder(settings.embedder, host=settings.ollama_host,
                             model=settings.embed_model)
    store = Store(settings.data_dir, embedder_name=embedder.name,
                  index_factory=index_backend(settings.index_backend))
    generator = generator or OllamaGenerator(model=settings.gen_model,
                                             host=settings.ollama_host)
    lock = threading.Lock()  # the store is shared by request threads
    app = FastAPI(title="ragsearch", description="Local-first semantic search and RAG")

    for error, status in ERROR_STATUS.items():
        def handler(request: Request, exc: Exception, status: int = status) -> JSONResponse:
            return JSONResponse(status_code=status, content={"detail": str(exc)})

        app.add_exception_handler(error, handler)

    def retrieve(query: str, k: int, web: bool, save_web: bool) -> list[Hit]:
        client = make_tavily(settings, store.data_dir) if web else None
        with lock:
            return hybrid_search(query, embedder, store, k=k, web=client,
                                 threshold=settings.web_fallback_threshold, persist=save_web)

    @app.get("/health")
    def health() -> Health:
        return Health(status="ok", chunks=len(store), embedder=embedder.name,
                      index_backend=settings.index_backend)

    @app.post("/ingest")
    def ingest_endpoint(request: IngestRequest) -> IngestResponse:
        if request.url:
            # Fetch before taking the lock: crawling waits 1 s+ per page.
            documents = list(web_documents(
                request.url, settings, store.data_dir, crawl=request.crawl,
                max_pages=request.max_pages, max_depth=request.max_depth,
                provider=request.provider,
            ))
        else:
            documents = load_documents(request.path or "")
        with lock:
            report = ingest(documents, embedder, store)
            return IngestResponse(**asdict(report), total_chunks=len(store))

    @app.post("/search")
    def search_endpoint(request: SearchRequest) -> SearchResponse:
        hits = retrieve(request.query, request.k, request.web, request.save_web)
        return SearchResponse(results=[SearchHit(**asdict(hit)) for hit in hits])

    @app.post("/ask")
    def ask_endpoint(request: AskRequest) -> AskResponse:
        hits = retrieve(request.question, request.k, request.web, request.save_web)
        if not hits:
            return AskResponse(answer=NO_ANSWER, citations=[], results=[])
        answer = generator.generate(request.question, hits)  # outside the lock: slow
        return AskResponse(
            answer=answer,
            citations=[
                Citation(number=n, source=hits[n - 1].source, origin=hits[n - 1].origin)
                for n in cited_numbers(answer, len(hits))
            ],
            results=[NumberedHit(number=n, **asdict(hit))
                     for n, hit in enumerate(hits, start=1)],
        )

    return app
