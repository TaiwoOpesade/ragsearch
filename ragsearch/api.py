"""HTTP API (FastAPI). Run with:

    uvicorn --factory ragsearch.api:create_app

Endpoints: GET /health, POST /ingest, POST /search, POST /ask, and the browser
interface at GET / (static files in ``ragsearch/ui``).

One process may serve the API while the CLI ingests into the same data dir:
the store is refreshed before each request and saves merge with each other.
The API has no authentication and can read local files, so it only answers
connections from this computer, and only requests whose Host header is in
``ALLOWED_HOSTS`` (which stops a web page that re-points its domain at 127.0.0.1,
known as DNS rebinding, from using it).
"""

import ipaddress
import mimetypes
import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from ragsearch.config import ConfigError, Settings, load_settings
from ragsearch.embedder import EmbeddingError, make_embedder
from ragsearch.generator import (
    NO_ANSWER,
    GenerationError,
    Generator,
    OllamaGenerator,
    cited_numbers,
)
from ragsearch.index import index_backend
from ragsearch.loaders import LoaderError, load_documents
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
from ragsearch.web.budget import BudgetExceeded, CreditBudget
from ragsearch.web.fetcher import Fetcher, FetchError
from ragsearch.web.sources import make_tavily, web_documents
from ragsearch.web.tavily import TavilyClient, TavilyError

UI_DIR = Path(__file__).parent / "ui"
# The Windows registry can map .js to text/plain, which browsers refuse to run.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
        "form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}



def _from_this_computer(request: Request) -> bool:
    try:
        return ipaddress.ip_address(request.client.host if request.client else "").is_loopback
    except ValueError:  # not an IP address at all
        return False


# Which HTTP status each domain error becomes.
ERROR_STATUS: dict[type[Exception], int] = {
    EmbeddingError: 503,
    GenerationError: 503,
    TavilyError: 502,
    BudgetExceeded: 429,
    FetchError: 422,
    LoaderError: 422,
    StoreError: 409,
}


def create_app(
    settings: Settings | None = None,
    generator: Generator | None = None,
    fetcher: Fetcher | None = None,
) -> FastAPI:
    try:
        settings = settings or load_settings()
        embedder = make_embedder(settings.embedder, host=settings.ollama_host,
                                 model=settings.embed_model)
        store = Store(settings.data_dir, embedder_name=embedder.name,
                      index_factory=index_backend(settings.index_backend))
    except (ConfigError, StoreError) as err:
        raise SystemExit(f"ragsearch: {err}") from err  # one clear line, no traceback
    generator = generator or OllamaGenerator(model=settings.gen_model,
                                             host=settings.ollama_host)
    fetcher = fetcher or Fetcher()  # one crawler, so its 1 s delay holds across requests
    lock = threading.Lock()  # the store is shared by request threads
    fetch_lock = threading.Lock()
    app = FastAPI(title="ragsearch", description="Local-first semantic search and RAG")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

    @app.middleware("http")
    async def guard(request: Request, call_next) -> Response:
        # A network client can fake the Host header, so also check where it connects from.
        if not _from_this_computer(request):
            return JSONResponse(status_code=403, content={
                "detail": "ragsearch only answers requests from this computer"})
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/ui/"):
            response.headers.update(PAGE_HEADERS)  # every copy of the page, not just "/"
        return response

    for error, status in ERROR_STATUS.items():
        def handler(request: Request, exc: Exception, status: int = status) -> JSONResponse:
            return JSONResponse(status_code=status, content={"detail": str(exc)})

        app.add_exception_handler(error, handler)

    def retrieve(query: str, k: int, web: bool, save_web: bool) -> tuple[list[Hit], list[str]]:
        warnings: list[str] = []
        client: TavilyClient | None = None
        if web:
            try:
                client = make_tavily(settings, store.data_dir)
            except TavilyError as err:
                warnings.append(f"web search is not configured: {err}")
        with lock:
            store.refresh()
            hits = hybrid_search(query, embedder, store, k=k, web=client,
                                 threshold=settings.web_fallback_threshold,
                                 persist=save_web, warnings=warnings)
        return hits, warnings

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    @app.get("/health")
    def health() -> Health:
        # Never wait for an ingest or web search: refresh only if the store is free.
        if lock.acquire(blocking=False):
            try:
                store.refresh()
            finally:
                lock.release()
        budget = CreditBudget(store.data_dir / "tavily_usage.json",
                              cap=settings.tavily_monthly_cap)
        try:
            used: int | None = budget.used
        except BudgetExceeded:  # unreadable usage file
            used = None
        return Health(status="ok", chunks=len(store), embedder=embedder.name,
                      index_backend=settings.index_backend,
                      web_configured=bool(settings.tavily_api_key),
                      web_credits_used=used, web_credits_cap=settings.tavily_monthly_cap,
                      web_fallback_threshold=settings.web_fallback_threshold)

    @app.post("/ingest")
    def ingest_endpoint(request: IngestRequest) -> IngestResponse:
        prune_under = None
        if request.url is not None:
            # Fetch before taking the store lock: crawling waits 1 s+ per page.
            with fetch_lock:
                documents = list(web_documents(
                    request.url, settings, store.data_dir, crawl=request.crawl,
                    max_pages=request.max_pages, max_depth=request.max_depth,
                    provider=request.provider, fetcher=fetcher,
                ))
        else:
            path = Path(request.path or "")
            if not path.exists():
                raise LoaderError(f"{path} does not exist")
            documents = load_documents(path)
            prune_under = path if path.is_dir() else None
        with lock:
            store.refresh()
            report = ingest(documents, embedder, store, prune_under=prune_under)
            return IngestResponse(**asdict(report), total_chunks=len(store))

    @app.post("/search")
    def search_endpoint(request: SearchRequest) -> SearchResponse:
        hits, warnings = retrieve(request.query, request.k, request.web, request.save_web)
        return SearchResponse(results=[SearchHit(**asdict(hit)) for hit in hits],
                              warnings=warnings)

    @app.post("/ask")
    def ask_endpoint(request: AskRequest) -> AskResponse:
        hits, warnings = retrieve(request.question, request.k, request.web, request.save_web)
        if not hits:
            return AskResponse(answer=NO_ANSWER, citations=[], results=[], warnings=warnings)
        answer = generator.generate(request.question, hits)  # outside the lock: slow
        return AskResponse(
            answer=answer,
            citations=[
                Citation(number=n, source=hits[n - 1].source, origin=hits[n - 1].origin)
                for n in cited_numbers(answer, len(hits))
            ],
            results=[NumberedHit(number=n, **asdict(hit))
                     for n, hit in enumerate(hits, start=1)],
            warnings=warnings,
        )

    return app
