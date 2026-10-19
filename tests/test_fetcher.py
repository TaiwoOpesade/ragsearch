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


# --- robots.txt edge cases, encodings, bad links, politeness -------------------


def test_longest_robots_rule_is_obeyed_during_a_crawl(stub_server, fake_time):
    stub_server.routes.update({
        "/robots.txt": (200, {}, "User-agent: *\nAllow: /\nDisallow: /private\n"),
        "/": (200, HTML, page("Home", "Home page text.", ["/private/secret", "/ok"])),
        "/ok": (200, HTML, page("OK", "Allowed page text.")),
        "/private/secret": (200, HTML, page("Secret", "must never be fetched")),
    })
    list(make_fetcher(fake_time).crawl(stub_server.url + "/", max_depth=2))
    assert "/ok" in stub_server.paths
    assert "/private/secret" not in stub_server.paths


def test_redirected_robots_txt_is_followed(stub_server, fake_time):
    stub_server.routes.update({
        "/robots.txt": (301, {"Location": "/real-robots.txt"}, ""),
        "/real-robots.txt": (200, {}, "User-agent: *\nDisallow: /\n"),
        "/": (200, HTML, page("Home", "must never be fetched")),
    })
    assert list(make_fetcher(fake_time).crawl(stub_server.url + "/")) == []
    assert "/" not in stub_server.paths


def test_rate_limited_robots_txt_means_do_not_crawl(stub_server, fake_time):
    stub_server.routes.update({
        "/robots.txt": (429, {"Retry-After": "60"}, "slow down"),
        "/": (200, HTML, page("Home", "must never be fetched")),
    })
    assert list(make_fetcher(fake_time).crawl(stub_server.url + "/")) == []
    assert stub_server.paths == ["/robots.txt"]


def test_non_utf8_page_is_decoded_using_its_charset(stub_server, fake_time):
    html = page("Menu", "Crème brûlée at the café is delicious.")
    stub_server.routes["/"] = (200, {"Content-Type": "text/html; charset=iso-8859-1"},
                               html.encode("latin-1"))
    text = make_fetcher(fake_time).fetch(stub_server.url + "/").text
    assert "café" in text
    assert "�" not in text


@pytest.mark.parametrize("charset", ["base64", "zlib", "hex", "rot13", "undefined"])
def test_a_charset_that_cannot_decode_text_is_ignored(stub_server, fake_time, charset):
    # Python's codec registry also holds bytes-to-bytes codecs such as base64,
    # which bytes.decode refuses, and "undefined", which always fails.
    declared = {"Content-Type": f"text/html; charset={charset}"}
    stub_server.routes.update({
        "/robots.txt": (200, {"Content-Type": f"text/plain; charset={charset}"},
                        "User-agent: *\nDisallow: /private\n"),
        "/": (200, declared, page("Home", "Home page text.", ["/private/secret", "/next"])),
        "/next": (200, declared, page("Next", "Next page text.")),
        "/private/secret": (200, HTML, page("Secret", "must never be fetched")),
    })
    pages = list(make_fetcher(fake_time).crawl(stub_server.url + "/", max_depth=1))
    assert [p.url.removeprefix(stub_server.url) for p in pages] == ["/", "/next"]
    assert "/private/secret" not in stub_server.paths  # robots.txt was still obeyed


def test_malformed_link_is_skipped_and_the_crawl_continues(stub_server, fake_time):
    stub_server.routes.update({
        "/": (200, HTML, page("Home", "Home page text.", ["http://[oops", "/good"])),
        "/good": (200, HTML, page("Good", "The good page text.")),
    })
    pages = list(make_fetcher(fake_time).crawl(stub_server.url + "/", max_depth=1))
    assert [p.url.removeprefix(stub_server.url) for p in pages] == ["/", "/good"]


def test_politeness_gap_is_measured_from_the_end_of_a_slow_response(stub_server, fake_time):
    def slow_home(request):
        fake_time.now += 1.5  # the server takes 1.5 s to answer
        return 200, HTML, page("Home", "Home page text.", ["/next"])

    stub_server.routes.update({
        "/": slow_home,
        "/next": (200, HTML, page("Next", "Next page text.")),
    })
    list(make_fetcher(fake_time).crawl(stub_server.url + "/", max_depth=1))
    # robots.txt, then a full 1 s wait before "/", then a full 1 s wait before "/next"
    assert fake_time.sleeps == [1.0, 1.0]


# --- robustness and robots.txt caching (second review) --------------------------------


def test_robots_redirect_to_an_unparseable_host_does_not_crash(stub_server, fake_time):
    bad_host = "http://" + "a" * 70 + ".example/robots.txt"  # a DNS label over 63 chars
    stub_server.routes.update({
        "/robots.txt": (301, {"Location": bad_host}, ""),
        "/": (200, HTML, page("Home", "must never be fetched")),
    })
    assert list(make_fetcher(fake_time).crawl(stub_server.url + "/")) == []
    assert "/" not in stub_server.paths  # unsure about robots.txt, so nothing is crawled


def test_a_page_that_breaks_the_html_parser_does_not_stop_the_crawl(stub_server, fake_time,
                                                                     monkeypatch):
    from ragsearch.web import fetcher as fetcher_module

    def broken_feed(self, data):
        raise AssertionError("unknown status keyword 'foo' in marked section")

    monkeypatch.setattr(fetcher_module._LinkParser, "feed", broken_feed)
    stub_server.routes["/"] = (200, HTML, page("Home", "Readable home page text.", ["/next"]))
    pages = list(make_fetcher(fake_time).crawl(stub_server.url + "/", max_depth=1))
    assert [p.url.removeprefix(stub_server.url) for p in pages] == ["/"]


def test_robots_txt_is_fetched_again_after_a_day(stub_server, fake_time):
    fetcher = make_fetcher(fake_time)
    stub_server.routes["/robots.txt"] = (200, {}, "User-agent: *\nAllow: /\n")
    stub_server.routes["/"] = (200, HTML, page("Home", "Home page text."))
    assert len(list(fetcher.crawl(stub_server.url + "/"))) == 1

    stub_server.routes["/robots.txt"] = (200, {}, "User-agent: *\nDisallow: /\n")
    fake_time.now += 3600  # an hour later: the cached copy still applies
    assert len(list(fetcher.crawl(stub_server.url + "/"))) == 1
    fake_time.now += 24 * 3600  # a day later: robots.txt is read again
    assert list(fetcher.crawl(stub_server.url + "/")) == []
    assert stub_server.paths.count("/robots.txt") == 2


def test_a_robots_txt_error_is_only_cached_for_a_few_minutes(stub_server, fake_time):
    fetcher = make_fetcher(fake_time)
    stub_server.routes["/robots.txt"] = (503, {}, "down")
    stub_server.routes["/"] = (200, HTML, page("Home", "Home page text."))
    assert list(fetcher.crawl(stub_server.url + "/")) == []

    stub_server.routes["/robots.txt"] = (404, {}, "none")  # the site recovered
    fake_time.now += 600
    assert len(list(fetcher.crawl(stub_server.url + "/"))) == 1
