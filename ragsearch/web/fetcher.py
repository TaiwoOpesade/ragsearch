"""A free, polite fallback crawler: requests for HTTP, trafilatura for main-text extraction.

Rules it always follows:
* obeys robots.txt using RFC 9309 matching (``robots.py``); a robots.txt that is
  rate-limited or erroring means "do not crawl", and redirects to it are followed;
* never leaves the host it started on, including via redirects;
* stops after ``max_pages`` requests and ``max_depth`` link hops;
* waits at least one second between the end of one response and the next request.
"""

import codecs
import logging
import re
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import NamedTuple
from urllib.parse import urldefrag, urljoin, urlsplit

import requests
import trafilatura

from ragsearch.web.robots import RobotsRules

log = logging.getLogger(__name__)

USER_AGENT = "ragsearch/0.1 (local research crawler)"
MIN_DELAY = 1.0
MAX_BYTES = 5_000_000
MAX_ROBOTS_REDIRECTS = 5  # RFC 9309 asks crawlers to follow at least five

_CHARSET = re.compile(r"charset=[\"']?([\w.:-]+)", re.IGNORECASE)


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

    def charset(self) -> str | None:
        match = _CHARSET.search(self.content_type)
        if not match:
            return None
        try:
            return codecs.lookup(match.group(1)).name
        except LookupError:
            return None


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


def _resolve(base: str, link: str) -> str | None:
    """Absolute, fragment-free http(s) URL for ``link``, or None if it is unusable."""
    try:
        target = urldefrag(urljoin(base, link.strip())).url
        scheme = urlsplit(target).scheme
    except ValueError:  # e.g. an unbalanced IPv6 bracket: "http://[oops"
        log.info("Skipping malformed link %r on %s", link, base)
        return None
    return target if scheme in ("http", "https") else None


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
        self._robots: dict[str, RobotsRules] = {}

    def fetch(self, url: str) -> Page:
        """Fetch one page (following same-host redirects) and return its main text."""
        pages = list(self.crawl(url, max_pages=6, max_depth=0))
        if not pages:
            raise FetchError(f"No readable text fetched from {url} (see warnings above)")
        return pages[0]

    def crawl(self, start_url: str, max_pages: int = 30, max_depth: int = 2) -> Iterator[Page]:
        """Breadth-first crawl of ``start_url``'s host, yielding pages with text."""
        host = _host(start_url)
        start = _resolve(start_url, start_url)
        if start is None:
            raise FetchError(f"Not an http(s) URL: {start_url!r}")
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

            links: list[tuple[str, int]] = []
            if response.location:  # redirect: same depth, still same-host only
                links = [(response.location, depth)]
            elif response.status != 200:
                log.warning("Skipping %s: HTTP %d", url, response.status)
            elif "html" not in response.content_type:
                log.info("Skipping %s: not HTML (%s)", url, response.content_type)
            else:
                charset = response.charset()
                # With no declared charset, give trafilatura the bytes so it can detect it.
                html = response.body.decode(charset, errors="replace") if charset else None
                text = trafilatura.extract(html if html is not None else response.body, url=url)
                if text:
                    yield Page(url=url, text=text)
                if depth < max_depth:
                    source = html or response.body.decode("utf-8", errors="replace")
                    links = [(link, depth + 1) for link in _links(source)]

            for link, link_depth in links:
                target = _resolve(url, link)
                if target is None or target in seen:
                    continue
                if _host(target) != host:
                    log.info("Not following %s: different host", target)
                    continue
                seen.add(target)
                queue.append((target, link_depth))

    def _allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self._robots:
            self._robots[root] = self._fetch_robots(root + "/robots.txt")
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        return self._robots[root].allowed(path)

    def _fetch_robots(self, url: str) -> RobotsRules:
        for _ in range(MAX_ROBOTS_REDIRECTS + 1):
            try:
                response = self._get(url)
            except FetchError:
                return RobotsRules.disallow_all()  # unreachable: be conservative
            if response.location:
                target = _resolve(url, response.location)
                if target is None:
                    return RobotsRules.allow_all()
                url = target
                continue
            status = response.status
            if status == 200:
                body = response.body.decode(response.charset() or "utf-8", errors="replace")
                return RobotsRules.parse(body, USER_AGENT)
            if status in (401, 403, 429) or status >= 500:
                return RobotsRules.disallow_all()  # forbidden, rate-limited or erroring
            return RobotsRules.allow_all()  # 404 and other 4xx: no robots.txt
        return RobotsRules.allow_all()  # too many redirects: treated as unavailable (RFC 9309)

    def _wait_politely(self) -> None:
        if self._last_request is not None:
            remaining = self._delay - (self._clock() - self._last_request)
            if remaining > 0:
                self._sleep(remaining)

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
        finally:
            # Measure the gap from when this response finished, not when it started.
            self._last_request = self._clock()


def _links(html: str) -> list[str]:
    parser = _LinkParser()
    parser.feed(html)
    return parser.links
