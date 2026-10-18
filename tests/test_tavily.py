"""TavilyClient against a local stub server. No test here touches the real API."""

import pytest

from ragsearch.web.budget import BudgetExceeded, CreditBudget
from ragsearch.web.tavily import TavilyClient, TavilyError

KEY = "tvly-test-key-not-real"


def search_response(request):
    return 200, {}, {
        "query": request["json"]["query"],
        "results": [
            {"title": "Stacks", "url": "https://a.example/stacks",
             "content": "A stack is LIFO.", "score": 0.91, "raw_content": None},
            {"title": "Queues", "url": "https://b.example/queues",
             "content": "A queue is FIFO.", "score": 0.42, "raw_content": None},
        ],
        "response_time": 0.5,
    }


@pytest.fixture
def budget(tmp_path):
    return CreditBudget(tmp_path / "tavily_usage.json", cap=20)


@pytest.fixture
def client(stub_server, budget):
    return TavilyClient(api_key=KEY, budget=budget, base_url=stub_server.url)


def test_search_sends_bearer_auth_and_documented_fields(stub_server, client):
    stub_server.routes["/search"] = search_response
    client.search("what is a stack", max_results=3)
    [request] = stub_server.requests
    assert request["method"] == "POST"
    assert request["headers"]["Authorization"] == f"Bearer {KEY}"
    assert request["json"] == {"query": "what is a stack", "max_results": 3,
                               "search_depth": "basic"}


def test_search_parses_results(stub_server, client):
    stub_server.routes["/search"] = search_response
    results = client.search("stack")
    assert [r.url for r in results] == ["https://a.example/stacks", "https://b.example/queues"]
    assert results[0].title == "Stacks"
    assert results[0].content == "A stack is LIFO."
    assert results[0].score == pytest.approx(0.91)


@pytest.mark.parametrize("depth,credits", [("basic", 1), ("advanced", 2)])
def test_search_records_credits_by_depth(stub_server, client, budget, depth, credits):
    stub_server.routes["/search"] = search_response
    client.search("stack", search_depth=depth)
    assert budget.used == credits


def test_usage_reported_by_tavily_is_preferred(stub_server, client, budget):
    stub_server.routes["/search"] = lambda r: (200, {}, {"results": [], "usage": {"credits": 2}})
    client.search("stack")
    assert budget.used == 2


def test_exhausted_budget_refuses_before_any_request(stub_server, client, budget):
    budget.record(20)
    stub_server.routes["/search"] = search_response
    with pytest.raises(BudgetExceeded):
        client.search("stack")
    assert stub_server.requests == []


def test_missing_api_key_is_a_clear_error(budget):
    with pytest.raises(TavilyError, match="TAVILY_API_KEY"):
        TavilyClient(api_key=None, budget=budget)


def test_extract_returns_pages_and_charges_for_successes_only(stub_server, client, budget):
    stub_server.routes["/extract"] = lambda r: (200, {}, {
        "results": [{"url": "https://a.example/1", "raw_content": "Page one text."}],
        "failed_results": [{"url": "https://a.example/2", "error": "Failed to retrieve"}],
        "response_time": 0.3,
    })
    pages = client.extract(["https://a.example/1", "https://a.example/2"])
    assert [(p.url, p.text) for p in pages] == [("https://a.example/1", "Page one text.")]
    body = stub_server.requests[0]["json"]
    assert body == {"urls": ["https://a.example/1", "https://a.example/2"],
                    "extract_depth": "basic", "format": "text"}
    assert budget.used == 1


def test_extract_sends_at_most_20_urls_per_request(stub_server, client):
    stub_server.routes["/extract"] = lambda r: (200, {}, {
        "results": [{"url": u, "raw_content": "t"} for u in r["json"]["urls"]],
        "failed_results": [],
    })
    pages = client.extract([f"https://a.example/{i}" for i in range(25)])
    assert [len(r["json"]["urls"]) for r in stub_server.requests] == [20, 5]
    assert len(pages) == 25


def test_crawl_sends_documented_fields_and_stays_on_site(stub_server, client, budget):
    stub_server.routes["/crawl"] = lambda r: (200, {}, {
        "base_url": "docs.example",
        "results": [{"url": f"https://docs.example/{i}", "raw_content": f"page {i}"}
                    for i in range(10)],
        "response_time": 1.2,
    })
    pages = client.crawl("https://docs.example", max_pages=30, max_depth=2)
    body = stub_server.requests[0]["json"]
    assert body == {"url": "https://docs.example", "max_depth": 2, "limit": 30,
                    "allow_external": False, "extract_depth": "basic", "format": "text"}
    assert len(pages) == 10
    assert budget.used == 1 + 2  # map 10 pages + extract 10 pages


def test_crawl_refuses_when_worst_case_cost_exceeds_budget(stub_server, client, budget):
    budget.record(15)
    with pytest.raises(BudgetExceeded):
        client.crawl("https://docs.example", max_pages=30)  # worst case 9 credits
    assert stub_server.requests == []


@pytest.mark.parametrize(
    "status,headers,message",
    [
        (401, {}, "invalid"),
        (429, {"Retry-After": "7"}, "7"),
        (432, {}, "plan"),
        (500, {}, "500"),
    ],
)
def test_http_errors_become_clear_tavily_errors(stub_server, client, budget,
                                                status, headers, message):
    stub_server.routes["/search"] = (status, headers, {"detail": {"error": "nope"}})
    with pytest.raises(TavilyError, match=message) as exc:
        client.search("stack")
    assert KEY not in str(exc.value)
    assert budget.used == 0  # failed calls are not charged


def test_unreachable_server_is_a_tavily_error(budget):
    client = TavilyClient(api_key=KEY, budget=budget, base_url="http://127.0.0.1:1", timeout=1)
    with pytest.raises(TavilyError, match="reach"):
        client.search("stack")
