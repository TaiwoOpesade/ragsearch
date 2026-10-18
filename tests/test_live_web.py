"""Opt-in tests against the real internet (RAGSEARCH_LIVE=1).

WARNING: the Tavily test spends 1 real credit from your monthly free plan and
records it in your real usage file (DATA_DIR/tavily_usage.json), so the local
budget guard stays accurate. Never enabled in CI.
"""

import pytest

from ragsearch.config import load_settings
from ragsearch.web.fetcher import Fetcher
from ragsearch.web.sources import make_tavily

pytestmark = pytest.mark.live


def test_real_tavily_search_returns_results():
    settings = load_settings()
    if not settings.tavily_api_key:
        pytest.skip("TAVILY_API_KEY not set")
    client = make_tavily(settings, settings.data_dir)
    results = client.search("what is a stack data structure", max_results=2)
    assert results and results[0].url.startswith("http")
    assert results[0].content


def test_real_fetcher_reads_example_dot_com():
    page = Fetcher().fetch("https://example.com/")
    # trafilatura keeps the main body text; the <h1> heading is treated as the title.
    assert page.url == "https://example.com/"
    assert "documentation examples" in page.text
