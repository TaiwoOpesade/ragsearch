"""Opt-in tests against the real internet (RAGSEARCH_LIVE=1).

WARNING: the Tavily test spends 1 real credit from your monthly free plan.
Never enabled in CI.
"""

import pytest

from ragsearch.config import load_settings
from ragsearch.web.budget import CreditBudget
from ragsearch.web.fetcher import Fetcher
from ragsearch.web.tavily import TavilyClient

pytestmark = pytest.mark.live


def test_real_tavily_search_returns_results(tmp_path):
    settings = load_settings()
    if not settings.tavily_api_key:
        pytest.skip("TAVILY_API_KEY not set")
    client = TavilyClient(settings.tavily_api_key, CreditBudget(tmp_path / "usage.json"))
    results = client.search("what is a stack data structure", max_results=2)
    assert results and results[0].url.startswith("http")


def test_real_fetcher_reads_example_dot_com():
    page = Fetcher().fetch("https://example.com/")
    assert "Example Domain" in page.text
