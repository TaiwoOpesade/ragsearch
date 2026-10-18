"""A small client for Tavily's REST API (search, extract, crawl) using httpx.

Contracts follow docs.tavily.com (API reference, read 2026-10-18):
* ``POST /search`` {query, max_results, search_depth} -> results[] {title, url, content, score}
* ``POST /extract`` {urls (max 20), extract_depth, format} -> results[] {url, raw_content},
  failed_results[] {url, error}
* ``POST /crawl`` {url, max_depth, limit, allow_external, extract_depth, format}
  -> results[] {url, raw_content}
* Auth: ``Authorization: Bearer <TAVILY_API_KEY>``; errors look like {"detail": {"error": ...}}.

Every call is checked against the local ``CreditBudget`` *before* it is sent.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from ragsearch.web.budget import CreditBudget, crawl_cost, extract_cost, search_cost
from ragsearch.web.fetcher import Page

log = logging.getLogger(__name__)

BASE_URL = "https://api.tavily.com"
MAX_EXTRACT_URLS = 20
# Only failures where a retry cannot double-charge: the request never arrived, or
# Tavily said it is temporarily unavailable (it does not charge failed calls).
RETRY_STATUSES = {502, 503, 504}
RETRY_DELAYS = (0.5, 1.0)  # seconds to wait before the 2nd and 3rd attempts


class TavilyError(RuntimeError):
    """Raised when Tavily cannot be reached or rejects a request."""


@dataclass(frozen=True)
class WebResult:
    url: str
    title: str
    content: str
    score: float


class TavilyClient:
    def __init__(
        self,
        api_key: str | None,
        budget: CreditBudget,
        base_url: str | None = None,
        timeout: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise TavilyError(
                "TAVILY_API_KEY is not set. Get a free key at tavily.com and add "
                "TAVILY_API_KEY=... to .env (or your environment), or use --provider local."
            )
        self._api_key = api_key
        self._budget = budget
        self._base_url = (base_url or BASE_URL).rstrip("/")
        self._timeout = timeout
        self._sleep = sleep

    def search(
        self, query: str, max_results: int = 5, search_depth: str = "basic"
    ) -> list[WebResult]:
        body = {"query": query, "max_results": max_results, "search_depth": search_depth}
        payload = self._post("/search", body, search_cost(search_depth))
        return [
            WebResult(r["url"], r.get("title", ""), r.get("content", ""), float(r["score"]))
            for r in payload.get("results", [])
        ]

    def extract(self, urls: list[str]) -> list[Page]:
        pages: list[Page] = []
        for start in range(0, len(urls), MAX_EXTRACT_URLS):
            batch = urls[start : start + MAX_EXTRACT_URLS]
            body = {"urls": batch, "extract_depth": "basic", "format": "text"}
            payload = self._post("/extract", body, extract_cost(len(batch)),
                                 actual=lambda p: extract_cost(len(p.get("results", []))))
            for failed in payload.get("failed_results", []):
                log.warning("Tavily could not extract %s: %s", failed["url"], failed["error"])
            pages += [Page(r["url"], r["raw_content"]) for r in payload.get("results", [])
                      if r.get("raw_content")]
        return pages

    def crawl(self, url: str, max_pages: int = 30, max_depth: int = 2) -> list[Page]:
        body = {
            "url": url,
            "max_depth": max(1, min(max_depth, 5)),  # Tavily allows 1-5
            "limit": max_pages,
            "allow_external": False,
            "extract_depth": "basic",
            "format": "text",
        }
        payload = self._post("/crawl", body, crawl_cost(max_pages),
                             actual=lambda p: crawl_cost(len(p.get("results", []))))
        return [Page(r["url"], r["raw_content"]) for r in payload.get("results", [])
                if r.get("raw_content")]

    def _post(
        self,
        path: str,
        body: dict,
        worst_case: int,
        actual: Callable[[dict], int] | None = None,
    ) -> dict:
        """POST ``body`` if ``worst_case`` credits fit the budget, then record the real cost."""
        self._budget.ensure_available(worst_case)
        response = self._send(path, body)
        if response.status_code != 200:
            raise TavilyError(_describe_error(response))

        payload = response.json()
        charged = (payload.get("usage") or {}).get("credits")
        if charged is None:
            charged = actual(payload) if actual else worst_case
        self._budget.record(int(charged))
        return payload


    def _send(self, path: str, body: dict) -> httpx.Response:
        """POST once, retrying only failures that cannot have been charged."""
        for attempt, delay in enumerate((*RETRY_DELAYS, None)):
            try:
                response = httpx.post(
                    self._base_url + path,
                    json=body,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=self._timeout,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as err:
                if delay is None:
                    raise TavilyError(
                        f"Could not reach Tavily at {self._base_url}: {err}"
                    ) from err
            except httpx.HTTPError as err:  # sent but no answer: retrying could double-charge
                raise TavilyError(f"Could not reach Tavily at {self._base_url}: {err}") from err
            else:
                if response.status_code not in RETRY_STATUSES or delay is None:
                    return response
            log.warning("Tavily request failed (attempt %d), retrying in %.1fs", attempt + 1,
                        delay)
            self._sleep(delay)
        raise AssertionError("unreachable")  # the loop always returns or raises


def _describe_error(response: httpx.Response) -> str:
    status = response.status_code
    if status == 401:
        return "Tavily rejected the API key (401): it is missing or invalid."
    if status == 429:
        wait = response.headers.get("Retry-After", "a few")
        return f"Tavily rate limit hit (429); retry after {wait} seconds."
    if status in (432, 433):
        return f"Tavily plan or usage limit exceeded ({status})."
    try:
        detail = response.json().get("detail")
        message = detail.get("error") if isinstance(detail, dict) else detail
    except ValueError:
        message = response.text[:200]
    return f"Tavily returned HTTP {status}: {message}"
