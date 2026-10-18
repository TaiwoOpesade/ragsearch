"""CLI web features against local stub servers only (no internet, no real Tavily)."""

import pytest

from ragsearch.cli import main
from ragsearch.config import Settings
from ragsearch.web import tavily

HTML = {"Content-Type": "text/html"}


def page(body, links=()):
    anchors = "".join(f'<a href="{h}">more</a> ' for h in links)
    return f"<html><body><article><p>{body}</p><p>{anchors}</p></article></body></html>"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", tavily_api_key="tvly-test")


@pytest.fixture
def site(stub_server):
    stub_server.routes.update({
        "/": (200, HTML, page("Binary search halves a sorted list each step.", ["/next"])),
        "/next": (200, HTML, page("Merge sort splits and merges lists.")),
    })
    return stub_server


@pytest.fixture
def fake_tavily(stub_server, monkeypatch):
    monkeypatch.setattr(tavily, "BASE_URL", stub_server.url)
    return stub_server


def run(argv, settings, capsys):
    code = main(argv, settings=settings)
    out, err = capsys.readouterr()
    return code, out, err


def test_ingest_single_url_with_local_fetcher(site, settings, capsys):
    code, out, _ = run(["ingest", site.url + "/", "--embedder", "hash"], settings, capsys)
    assert code == 0
    assert "1 sources" in out
    assert "/next" not in site.paths  # a single page, not a crawl

    _, out, _ = run(["search", "binary search sorted list", "--embedder", "hash"],
                    settings, capsys)
    assert out.splitlines()[0].endswith(site.url + "/")


def test_crawl_with_local_provider_honours_max_pages(site, settings, capsys):
    code, out, _ = run(["ingest", site.url + "/", "--crawl", "--max-pages", "2",
                        "--provider", "local", "--embedder", "hash"], settings, capsys)
    assert code == 0
    assert "2 sources" in out


def test_ingest_url_with_tavily_uses_extract(fake_tavily, settings, capsys):
    fake_tavily.routes["/extract"] = lambda r: (200, {}, {
        "results": [{"url": r["json"]["urls"][0], "raw_content": "Hash tables map keys."}],
        "failed_results": [],
    })
    code, out, _ = run(["ingest", "https://docs.example/hash", "--provider", "tavily",
                        "--embedder", "hash"], settings, capsys)
    assert code == 0
    assert "1 sources" in out
    assert fake_tavily.paths == ["/extract"]
    assert (settings.data_dir / "tavily_usage.json").exists()


def test_crawl_with_tavily_passes_limits(fake_tavily, settings, capsys):
    fake_tavily.routes["/crawl"] = lambda r: (200, {}, {"base_url": "docs.example", "results": [
        {"url": "https://docs.example/a", "raw_content": "Trees have nodes."},
        {"url": "https://docs.example/b", "raw_content": "Graphs have edges."},
    ]})
    code, out, _ = run(["ingest", "https://docs.example", "--crawl", "--max-pages", "30",
                        "--max-depth", "2", "--provider", "tavily", "--embedder", "hash"],
                       settings, capsys)
    assert code == 0
    assert fake_tavily.requests[0]["json"]["limit"] == 30
    assert fake_tavily.requests[0]["json"]["max_depth"] == 2
    assert "2 sources" in out


def test_tavily_without_api_key_is_a_clear_error(tmp_path, capsys):
    settings = Settings(data_dir=tmp_path / "data")
    code, _, err = run(["ingest", "https://docs.example", "--provider", "tavily",
                        "--embedder", "hash"], settings, capsys)
    assert code == 1
    assert "TAVILY_API_KEY" in err


def test_search_with_web_labels_each_result(fake_tavily, settings, capsys):
    fake_tavily.routes["/search"] = lambda r: (200, {}, {"results": [
        {"title": "Dijkstra", "url": "https://web.example/dijkstra",
         "content": "Dijkstra finds shortest paths in weighted graphs.", "score": 0.8},
    ]})
    code, out, _ = run(["search", "shortest paths weighted graphs", "--web",
                        "--embedder", "hash"], settings, capsys)
    assert code == 0
    assert "[web]" in out.splitlines()[0]
    assert out.splitlines()[0].endswith("https://web.example/dijkstra")


def test_search_without_web_flag_never_calls_tavily(fake_tavily, settings, capsys):
    run(["search", "anything", "--embedder", "hash"], settings, capsys)
    assert fake_tavily.requests == []
