"""Turn a URL into ``Document``s using the free local crawler or Tavily."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from ragsearch.config import Settings
from ragsearch.loaders import Document
from ragsearch.web.budget import CreditBudget
from ragsearch.web.fetcher import Fetcher
from ragsearch.web.tavily import TavilyClient


def make_tavily(settings: Settings, data_dir: Path) -> TavilyClient:
    budget = CreditBudget(data_dir / "tavily_usage.json", cap=settings.tavily_monthly_cap)
    return TavilyClient(api_key=settings.tavily_api_key, budget=budget)


def web_documents(
    url: str,
    settings: Settings,
    data_dir: Path,
    *,
    crawl: bool = False,
    max_pages: int = 30,
    max_depth: int = 2,
    provider: Literal["local", "tavily"] = "local",
    fetcher: Fetcher | None = None,
) -> Iterable[Document]:
    """One page, or a same-site crawl, as documents keyed by page URL.

    Pass a long-lived ``fetcher`` (as the API does) so the politeness delay and
    the robots.txt cache hold across calls, not just within one.
    """
    if provider == "tavily":
        client = make_tavily(settings, data_dir)
        pages = client.crawl(url, max_pages, max_depth) if crawl else client.extract([url])
    else:
        fetcher = fetcher or Fetcher()
        pages = fetcher.crawl(url, max_pages, max_depth) if crawl else [fetcher.fetch(url)]
    return (Document.from_text(page.url, page.text) for page in pages)
