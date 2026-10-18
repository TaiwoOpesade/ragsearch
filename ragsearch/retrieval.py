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
) -> list[Hit]:
    query_vector = embedder.embed([query])[0]
    hits = [
        Hit(r.score, r.item["source"], r.item["text"], "local")
        for r in store.search(query_vector, k)
    ]
    if web is None or (hits and hits[0].score >= threshold):
        return hits

    try:
        results = [r for r in web.search(query, max_results=k) if r.content.strip()]
    except (TavilyError, BudgetExceeded) as err:
        log.warning("web search failed, using local results only: %s", err)
        return hits
    if not results:
        return hits

    vectors = embedder.embed([r.content for r in results])
    scores = vectors @ query_vector / (np.linalg.norm(query_vector) or 1.0)
    hits += [
        Hit(float(score), r.url, r.content, "web")
        for r, score in zip(results, scores, strict=True)
    ]
    if persist:
        ingest([Document.from_text(r.url, r.content) for r in results], embedder, store)
    return sorted(hits, key=lambda h: h.score, reverse=True)[:k]
