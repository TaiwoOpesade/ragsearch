"""Reading HTTP responses and HTML for the crawler: charsets, links and URLs.

Kept apart from ``fetcher.py`` so that module holds only the crawling policy
(robots.txt, same host, limits, politeness).
"""

import codecs
import logging
import re
from html.parser import HTMLParser
from typing import NamedTuple
from urllib.parse import urldefrag, urljoin, urlsplit

log = logging.getLogger(__name__)

_CHARSET = re.compile(r"charset=[\"']?([\w.:-]+)", re.IGNORECASE)


class Response(NamedTuple):
    status: int
    content_type: str
    body: bytes
    location: str | None

    def charset(self) -> str | None:
        """The declared text encoding, or None (also for bytes codecs like base64)."""
        match = _CHARSET.search(self.content_type)
        if not match:
            return None
        try:
            info = codecs.lookup(match.group(1))
        except LookupError:
            return None
        return info.name if getattr(info, "_is_text_encoding", True) else None


def decode(body: bytes, charset: str) -> str:
    """Decode with ``charset``, falling back to UTF-8 if that codec cannot decode."""
    try:
        return body.decode(charset, errors="replace")
    except (LookupError, UnicodeError):  # e.g. charset=undefined
        return body.decode("utf-8", errors="replace")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href")
        if tag == "a" and href:
            self.links.append(href)


def links(html: str, url: str) -> list[str]:
    """Every ``<a href>`` in ``html``; markup the parser chokes on ends the list early."""
    parser = LinkParser()
    try:
        parser.feed(html)
    except (AssertionError, ValueError) as err:  # older HTMLParser releases, odd markup
        log.info("Stopped reading links on %s: %s", url, err)
    return parser.links


def host(url: str) -> str:
    return urlsplit(url).netloc.lower()


def resolve(base: str, link: str) -> str | None:
    """Absolute, fragment-free http(s) URL for ``link``, or None if it is unusable."""
    try:
        target = urldefrag(urljoin(base, link.strip())).url
        scheme = urlsplit(target).scheme
    except ValueError:  # e.g. an unbalanced IPv6 bracket: "http://[oops"
        log.info("Skipping malformed link %r on %s", link, base)
        return None
    return target if scheme in ("http", "https") else None
