import pytest

from ragsearch.web.fetcher import Fetcher, FetchError

HTML = {"Content-Type": "text/html; charset=utf-8"}


def page(title, body, links=()):
    anchors = "".join(f'<a href="{href}">link</a> ' for href in links)
    return (
        f"<html><head><title>{title}</title></head><body><article><h1>{title}</h1>"
        f"<p>{body}</p><p>{anchors}</p></article></body></html>"
    )


class FakeTime:
    """Deterministic clock: time only moves when the fetcher sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_time():
    return FakeTime()


@pytest.fixture
def site(stub_server):
    other_host = f"http://localhost:{stub_server.port}/external"  # same server, other host
    stub_server.routes.update({
        "/robots.txt": (200, {}, "User-agent: *\nDisallow: /private\n"),
        "/": (200, HTML, page("Home", "Welcome to the stacks site.",
                              ["/a", "/b", "/a#section", "/private/secret", other_host,
                               "/doc.pdf", "mailto:me@example.com"])),
        "/a": (200, HTML, page("Page A", "A stack is last in first out.", ["/c"])),
        "/b": (200, HTML, page("Page B", "Queues are first in first out.")),
        "/c": (200, HTML, page("Page C", "Depth two page about recursion.", ["/d"])),
        "/d": (200, HTML, page("Page D", "Depth three page.")),
        "/private/secret": (200, HTML, page("Secret", "must never be fetched")),
        "/external": (200, HTML, page("External", "must never be fetched")),
        "/doc.pdf": (200, {"Content-Type": "application/pdf"}, b"%PDF-1.4"),
    })
    return stub_server


def make_fetcher(fake_time, **kwargs):
    return Fetcher(sleep=fake_time.sleep, clock=fake_time.clock, **kwargs)


def test_fetch_single_page_returns_extracted_text(site, fake_time):
    result = make_fetcher(fake_time).fetch(site.url + "/a")
    assert result.url == site.url + "/a"
    assert "last in first out" in result.text
    assert "<p>" not in result.text


def test_crawl_respects_robots_txt(site, fake_time):
    list(make_fetcher(fake_time).crawl(site.url + "/", max_pages=30, max_depth=5))
    assert "/robots.txt" in site.paths
    assert "/private/secret" not in site.paths


def test_crawl_stays_on_the_same_host(site, fake_time):
    list(make_fetcher(fake_time).crawl(site.url + "/", max_pages=30, max_depth=5))
    assert "/external" not in site.paths


def test_crawl_honours_max_depth(site, fake_time):
    pages = list(make_fetcher(fake_time).crawl(site.url + "/", max_pages=30, max_depth=1))
    assert {p.url.removeprefix(site.url) for p in pages} == {"/", "/a", "/b"}
    assert "/c" not in site.paths


def test_crawl_honours_max_pages(site, fake_time):
    list(make_fetcher(fake_time).crawl(site.url + "/", max_pages=2, max_depth=5))
    assert [p for p in site.paths if p != "/robots.txt"] == ["/", "/a"]


def test_crawl_skips_non_html_and_fetches_each_url_once(site, fake_time):
    pages = list(make_fetcher(fake_time).crawl(site.url + "/", max_pages=30, max_depth=5))
    assert all(not p.url.endswith(".pdf") for p in pages)
    assert site.paths.count("/a") == 1
    assert len(pages) == 5  # /, /a, /b, /c, /d


def test_waits_at_least_one_second_between_every_request(site, fake_time):
    list(make_fetcher(fake_time).crawl(site.url + "/", max_pages=30, max_depth=5))
    assert len(fake_time.sleeps) == len(site.requests) - 1
    assert all(s >= 1.0 for s in fake_time.sleeps)


def test_delay_below_one_second_is_rejected():
    with pytest.raises(ValueError, match="1 second"):
        Fetcher(delay=0.5)


def test_robots_server_error_means_nothing_is_fetched(stub_server, fake_time):
    stub_server.routes["/robots.txt"] = (503, {}, "down")
    stub_server.routes["/"] = (200, HTML, page("Home", "text"))
    assert list(make_fetcher(fake_time).crawl(stub_server.url + "/")) == []
    assert stub_server.paths == ["/robots.txt"]


def test_missing_robots_txt_allows_crawling(stub_server, fake_time):
    stub_server.routes["/"] = (200, HTML, page("Home", "Plain page text here."))
    pages = list(make_fetcher(fake_time).crawl(stub_server.url + "/"))
    assert len(pages) == 1


def test_fetch_of_disallowed_page_raises(site, fake_time):
    with pytest.raises(FetchError):
        make_fetcher(fake_time).fetch(site.url + "/private/secret")
    assert "/private/secret" not in site.paths


def test_http_error_page_raises_on_fetch(site, fake_time):
    with pytest.raises(FetchError):
        make_fetcher(fake_time).fetch(site.url + "/missing")


def test_redirect_to_same_host_is_followed(site, fake_time):
    site.routes["/old"] = (301, {"Location": "/a"}, "")
    assert "last in first out" in make_fetcher(fake_time).fetch(site.url + "/old").text


def test_redirect_to_another_host_is_not_followed(site, fake_time):
    site.routes["/away"] = (302, {"Location": f"http://localhost:{site.port}/external"}, "")
    with pytest.raises(FetchError):
        make_fetcher(fake_time).fetch(site.url + "/away")
    assert "/external" not in site.paths


def test_oversized_page_is_skipped(site, fake_time):
    site.routes["/huge"] = (200, HTML, page("Huge", "x " * 5000))
    with pytest.raises(FetchError):
        make_fetcher(fake_time, max_bytes=1000).fetch(site.url + "/huge")


def test_unreachable_host_raises_fetch_error(fake_time):
    with pytest.raises(FetchError):
        make_fetcher(fake_time, timeout=1).fetch("http://127.0.0.1:1/")
