"""A free, polite fallback crawler: requests for HTTP, trafilatura for main-text extraction.

Rules it always follows:
* obeys robots.txt (a server error on robots.txt means "do not crawl");
* never leaves the host it started on, including via redirects;
* stops after ``max_pages`` requests and ``max_depth`` link hops;
* waits at least one second between any two requests.
"""

import logging
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import NamedTuple
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests
import trafilatura

log = logging.getLogger(__name__)

USER_AGENT = "ragsearch/0.1 (local research crawler)"
MIN_DELAY = 1.0
MAX_BYTES = 5_000_000


class FetchError(RuntimeError):
    """Raised when a page cannot be fetched or has no readable text."""


@dataclass(frozen=True)
class Page:
    url: str
    text: str


class _Response(NamedTuple):
    status: int
    content_type: str
    body: bytes
    location: str | None


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href")
        if tag == "a" and href:
            self.links.append(href)


def _host(url: str) -> str:
    return urlsplit(url).netloc.lower()


class Fetcher:
    def __init__(
        self,
        session: requests.Session | None = None,
        delay: float = MIN_DELAY,
        timeout: float = 10.0,
        max_bytes: int = MAX_BYTES,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        if delay < MIN_DELAY:
            raise ValueError("delay must be at least 1 second between requests")
        self._session = session or requests.Session()
        self._delay, self._timeout, self._max_bytes = delay, timeout, max_bytes
        self._sleep, self._clock = sleep, clock
        self._last_request: float | None = None
        self._robots: dict[str, RobotFileParser] = {}

    def fetch(self, url: str) -> Page:
        """Fetch one page (following same-host redirects) and return its main text."""
        pages = list(self.crawl(url, max_pages=6, max_depth=0))
        if not pages:
            raise FetchError(f"No readable text fetched from {url} (see warnings above)")
        return pages[0]

    def crawl(self, start_url: str, max_pages: int = 30, max_depth: int = 2) -> Iterator[Page]:
        """Breadth-first crawl of ``start_url``'s host, yielding pages with text."""
        host = _host(start_url)
        start = urldefrag(start_url).url
        queue, seen, requests_made = deque([(start, 0)]), {start}, 0
        while queue and requests_made < max_pages:
            url, depth = queue.popleft()
            if not self._allowed(url):
                log.warning("Skipping %s: disallowed by robots.txt", url)
                continue
            try:
                requests_made += 1
                response = self._get(url)
            except FetchError as err:
                log.warning("Skipping %s: %s", url, err)
                continue

            targets: list[tuple[str, int]] = []
            if response.location:  # redirect: same depth, still same-host only
                targets = [(urljoin(url, response.location), depth)]
            elif response.status != 200:
                log.warning("Skipping %s: HTTP %d", url, response.status)
            elif "html" not in response.content_type:
                log.info("Skipping %s: not HTML (%s)", url, response.content_type)
            else:
                html = response.body.decode("utf-8", errors="replace")
                text = trafilatura.extract(html, url=url)
                if text:
                    yield Page(url=url, text=text)
                if depth < max_depth:
                    targets = [(urljoin(url, link), depth + 1) for link in _links(html)]

            for target, target_depth in targets:
                target = urldefrag(target).url
                if target in seen or urlsplit(target).scheme not in ("http", "https"):
                    continue
                if _host(target) != host:
                    log.info("Not following %s: different host", target)
                    continue
                seen.add(target)
                queue.append((target, target_depth))

    def _allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self._robots:
            parser = RobotFileParser()
            try:
                response = self._get(root + "/robots.txt")
                status = response.status
            except FetchError:
                status = 503
            if status in (401, 403) or status >= 500:
                parser.disallow_all = True  # be conservative when unsure
            elif status >= 300:
                parser.allow_all = True  # no robots.txt: crawling is allowed
            else:
                parser.parse(response.body.decode("utf-8", errors="replace").splitlines())
            self._robots[root] = parser
        return self._robots[root].can_fetch(USER_AGENT, url)

    def _wait_politely(self) -> None:
        if self._last_request is not None:
            remaining = self._delay - (self._clock() - self._last_request)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request = self._clock()

    def _get(self, url: str) -> _Response:
        self._wait_politely()
        try:
            with self._session.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=self._timeout,
                allow_redirects=False, stream=True,
            ) as r:
                body = bytearray()
                for block in r.iter_content(64 * 1024):
                    body += block
                    if len(body) > self._max_bytes:
                        raise FetchError(f"response larger than {self._max_bytes} bytes")
                location = r.headers.get("Location") if r.is_redirect else None
                return _Response(r.status_code, r.headers.get("Content-Type", ""),
                                 bytes(body), location)
        except requests.RequestException as err:
            raise FetchError(f"request failed: {err}") from err


def _links(html: str) -> list[str]:
    parser = _LinkParser()
    parser.feed(html)
    return parser.links
