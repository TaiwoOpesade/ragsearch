"""Hybrid retrieval: local index first, live web search only when local results are weak.

If the best local score is below ``threshold``, Tavily is searched, the returned
snippets are embedded in memory with the *same* embedder, and everything is
ranked together by our own cosine score. Each hit is labelled "local" or "web".
Web results are only written to the store when ``persist=True``.
"""

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from ragsearch.embedder import Embedder
from ragsearch.loaders import Document
from ragsearch.pipeline import ingest
from ragsearch.store import Store
from ragsearch.web.budget import BudgetExceeded
from ragsearch.web.tavily import TavilyError, WebResult

log = logging.getLogger(__name__)


class WebSearcher(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[WebResult]: ...


@dataclass(frozen=True)
class Hit:
    score: float
    source: str
    text: str
    origin: Literal["local", "web"]


def hybrid_search(
    query: str,
    embedder: Embedder,
    store: Store,
    k: int = 5,
    *,
    web: WebSearcher | None = None,
    threshold: float = 0.5,
    persist: bool = False,
    warnings: list[str] | None = None,
) -> list[Hit]:
    """Local results, plus labelled web results when the best local score is weak.

    A web failure never fails the search: it is logged and, if ``warnings`` is
    given, appended to it so callers can tell the user.
    """
    query_vector = embedder.embed_query(query)
    hits = [
        Hit(r.score, r.item["source"], r.item["text"], "local")
        for r in store.search(query_vector, k)
    ]
    if web is None or (hits and hits[0].score >= threshold):
        return hits

    try:
        results = [r for r in web.search(query, max_results=k) if r.content.strip()]
    except (TavilyError, BudgetExceeded) as err:
        message = f"web search failed, using local results only: {err}"
        log.warning(message)
        if warnings is not None:
            warnings.append(message)
        return hits
    local_sources = {h.source for h in hits}
    results = [r for r in results if r.url not in local_sources]  # no duplicates
    if not results:
        return hits

    vectors = embedder.embed([r.content for r in results])
    scores = vectors @ query_vector / (np.linalg.norm(query_vector) or 1.0)
    hits += [
        Hit(float(score), r.url, r.content, "web")
        for r, score in zip(results, scores, strict=True)
    ]
    if persist:
        # Never let a short snippet replace a page that is already indexed in full.
        new = [r for r in results if r.url not in store.sources]
        ingest([Document.from_text(r.url, r.content) for r in new], embedder, store)
    return sorted(hits, key=lambda h: h.score, reverse=True)[:k]
