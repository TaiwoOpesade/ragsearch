"""A small client for Tavily's REST API (search, extract, crawl) using httpx.

Contracts follow docs.tavily.com (API reference, read 2026-10-18):
* ``POST /search`` {query, max_results (0-20), search_depth} -> results[] {title, url,
  content, score}
* ``POST /extract`` {urls (max 20), extract_depth, format} -> results[] {url, raw_content},
  failed_results[] {url, error}
* ``POST /crawl`` {url, max_depth (1-5), limit, allow_external, extract_depth, format}
  -> results[] {url, raw_content}
* ``include_usage: true`` adds usage.credits (what the call really cost) to a reply.
* Auth: ``Authorization: Bearer <TAVILY_API_KEY>``; errors look like {"detail": {"error": ...}}.

Credits are reserved in the local ``CreditBudget`` *before* a call is sent and
settled to the real cost afterwards. A failure that Tavily cannot have charged
gives the reservation back; one that it might have charged keeps it.
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
MAX_SEARCH_RESULTS = 20
CRAWL_TIMEOUT = 180.0  # Tavily's own crawl timeout defaults to 150 s
# Only failures where a retry cannot double-charge: the request never arrived, or
# Tavily said it is temporarily unavailable (it does not charge failed calls).
RETRY_STATUSES = {502, 503, 504}
RETRY_DELAYS = (0.5, 1.0)  # seconds to wait before the 2nd and 3rd attempts


class TavilyError(RuntimeError):
    """Raised when Tavily cannot be reached or rejects a request."""


class _NotCharged(TavilyError):
    """The request never reached Tavily, or Tavily refused it: nothing was charged."""


class _MaybeCharged(TavilyError):
    """The request was sent but no usable answer came back: it may have been charged."""


@dataclass(frozen=True)
class WebResult:
    url: str
    title: str
    content: str
    score: float


def _valid_key(key: str) -> bool:
    return key.isascii() and key.isprintable() and not any(c.isspace() for c in key)


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
        if not _valid_key(api_key):
            # Never echo the key: an HTTP library error about a bad header would.
            raise TavilyError(
                "TAVILY_API_KEY contains spaces or invisible characters; copy it again "
                "from tavily.com with nothing before or after it."
            )
        self._api_key = api_key
        self._budget = budget
        self._base_url = (base_url or BASE_URL).rstrip("/")
        self._timeout = timeout
        self._sleep = sleep

    def search(
        self, query: str, max_results: int = 5, search_depth: str = "basic"
    ) -> list[WebResult]:
        body = {
            "query": query,
            "max_results": max(0, min(max_results, MAX_SEARCH_RESULTS)),
            "search_depth": search_depth,
        }
        payload = self._post("/search", body, search_cost(search_depth))
        return [
            WebResult(r["url"], r.get("title") or "", r.get("content") or "",
                      float(r.get("score") or 0.0))
            for r in payload.get("results", [])
            if isinstance(r, dict) and r.get("url")
        ]

    def extract(self, urls: list[str]) -> list[Page]:
        pages: list[Page] = []
        for start in range(0, len(urls), MAX_EXTRACT_URLS):
            batch = urls[start : start + MAX_EXTRACT_URLS]
            body = {"urls": batch, "extract_depth": "basic", "format": "text"}
            payload = self._post("/extract", body, extract_cost(len(batch)),
                                 actual=lambda p: extract_cost(len(p.get("results", []))))
            for failed in payload.get("failed_results", []):
                log.warning("Tavily could not extract %s: %s", failed.get("url"),
                            failed.get("error"))
            pages += _pages(payload)
        return pages

    def crawl(self, url: str, max_pages: int = 30, max_depth: int = 2) -> list[Page]:
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        body = {
            "url": url,
            "max_depth": max(1, min(max_depth, 5)),  # Tavily allows 1-5
            "limit": max_pages,
            "allow_external": False,
            "extract_depth": "basic",
            "format": "text",
        }
        payload = self._post("/crawl", body, crawl_cost(max_pages),
                             actual=lambda p: crawl_cost(len(p.get("results", []))),
                             timeout=max(self._timeout, CRAWL_TIMEOUT))
        return _pages(payload)

    def _post(
        self,
        path: str,
        body: dict,
        worst_case: int,
        actual: Callable[[dict], int] | None = None,
        timeout: float | None = None,
    ) -> dict:
        """Reserve ``worst_case`` credits, POST ``body``, then settle to the real cost."""
        reserved = self._budget.reserve(worst_case)
        try:
            payload = self._send(path, {**body, "include_usage": True}, timeout or self._timeout)
        except _NotCharged:
            self._budget.adjust(-reserved)
            raise
        # _MaybeCharged propagates with the reservation kept: better to over-count.
        charged = (payload.get("usage") or {}).get("credits")
        if not isinstance(charged, int):
            charged = actual(payload) if actual else worst_case
        self._budget.adjust(charged - reserved)
        return payload

    def _send(self, path: str, body: dict, timeout: float) -> dict:
        """POST, retrying only failures that cannot have been charged; return the JSON."""
        for attempt, delay in enumerate((*RETRY_DELAYS, None)):
            try:
                response = httpx.post(
                    self._base_url + path,
                    json=body,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=timeout,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as err:
                if delay is None:
                    raise _NotCharged(self._redact(
                        f"Could not reach Tavily at {self._base_url}: {err}")) from err
            except httpx.HTTPError as err:  # sent, no answer: a retry could double-charge
                raise _MaybeCharged(self._redact(
                    f"No answer from Tavily at {self._base_url} ({type(err).__name__}); "
                    "the call may have been charged, so it was counted.")) from err
            else:
                if response.status_code == 200:
                    return self._parse(response)
                if response.status_code not in RETRY_STATUSES or delay is None:
                    raise _NotCharged(self._redact(_describe_error(response)))
            log.warning("Tavily request failed (attempt %d), retrying in %.1fs", attempt + 1,
                        delay)
            self._sleep(delay)
        raise AssertionError("unreachable")  # the loop always returns or raises

    def _parse(self, response: httpx.Response) -> dict:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            raise _MaybeCharged("Tavily sent an unreadable reply; the call may have been "
                                "charged, so it was counted.")
        return payload

    def _redact(self, message: str) -> str:
        return message.replace(self._api_key, "***")


def _pages(payload: dict) -> list[Page]:
    return [Page(r["url"], r["raw_content"]) for r in payload.get("results", [])
            if isinstance(r, dict) and r.get("url") and r.get("raw_content")]


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
    except (ValueError, AttributeError):
        message = response.text[:200]
    return f"Tavily returned HTTP {status}: {message}"
