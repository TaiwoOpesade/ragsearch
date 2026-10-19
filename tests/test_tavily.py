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
                               "search_depth": "basic", "include_usage": True}


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
                    "extract_depth": "basic", "format": "text",
                    "include_usage": True}
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
                    "allow_external": False, "extract_depth": "basic", "format": "text",
                    "include_usage": True}
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
    client = TavilyClient(api_key=KEY, budget=budget, base_url="http://127.0.0.1:1", timeout=1,
                          sleep=lambda seconds: None)  # retry timing is tested separately
    with pytest.raises(TavilyError, match="reach"):
        client.search("stack")


# --- retries for transient failures ------------------------------------------


class RecordingSleep:
    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def flaky(statuses):
    """Route that answers with each status in turn, then succeeds."""
    remaining = list(statuses)

    def route(request):
        if remaining:
            return remaining.pop(0), {}, {"detail": {"error": "temporarily unavailable"}}
        return search_response(request)

    return route


def test_temporarily_unavailable_is_retried_then_succeeds(stub_server, budget):
    stub_server.routes["/search"] = flaky([503, 503])
    sleep = RecordingSleep()
    client = TavilyClient(api_key=KEY, budget=budget, base_url=stub_server.url, sleep=sleep)
    results = client.search("stack")
    assert len(results) == 2
    assert len(stub_server.requests) == 3
    assert sleep.calls == [0.5, 1.0]
    assert budget.used == 1  # charged once, for the call that succeeded


def test_gives_up_after_the_retries_are_used(stub_server, budget):
    stub_server.routes["/search"] = flaky([503, 503, 503, 503])
    client = TavilyClient(api_key=KEY, budget=budget, base_url=stub_server.url,
                          sleep=RecordingSleep())
    with pytest.raises(TavilyError, match="503"):
        client.search("stack")
    assert len(stub_server.requests) == 3
    assert budget.used == 0


@pytest.mark.parametrize("status", [502, 504])
def test_gateway_errors_keep_the_charge_and_are_never_retried(stub_server, budget, status):
    # The request reached Tavily's gateway, so the backend may still finish and charge
    # it. A long crawl is the likeliest case, and a retry could pay for it twice.
    stub_server.routes["/crawl"] = (status, {}, {"detail": {"error": "gateway"}})
    sleep = RecordingSleep()
    client = TavilyClient(api_key=KEY, budget=budget, base_url=stub_server.url, sleep=sleep)
    with pytest.raises(TavilyError, match="may have been charged"):
        client.crawl("https://docs.example", max_pages=30)
    assert budget.used == 9  # the worst case for 30 pages stays counted
    assert len(stub_server.requests) == 1
    assert sleep.calls == []


@pytest.mark.parametrize("status", [400, 401, 429, 432, 500])
def test_errors_that_a_retry_cannot_fix_are_not_retried(stub_server, budget, status):
    stub_server.routes["/search"] = (status, {}, {"detail": {"error": "no"}})
    sleep = RecordingSleep()
    client = TavilyClient(api_key=KEY, budget=budget, base_url=stub_server.url, sleep=sleep)
    with pytest.raises(TavilyError):
        client.search("stack")
    assert len(stub_server.requests) == 1
    assert sleep.calls == []


def test_connection_failures_are_retried(budget):
    sleep = RecordingSleep()
    client = TavilyClient(api_key=KEY, budget=budget, base_url="http://127.0.0.1:1",
                          timeout=1, sleep=sleep)
    with pytest.raises(TavilyError, match="reach"):
        client.search("stack")
    assert sleep.calls == [0.5, 1.0]


def test_missing_key_message_mentions_the_env_file(budget):
    with pytest.raises(TavilyError, match=r"\.env"):
        TavilyClient(api_key=None, budget=budget)


# --- key safety, limits and unreadable replies ---------------------------------


@pytest.mark.parametrize("bad_key", ["tvly-abc ", "tvly-abc\n", "tvly a", "tvly-é"])
def test_malformed_key_is_rejected_without_echoing_it(stub_server, budget, bad_key):
    with pytest.raises(TavilyError, match="TAVILY_API_KEY") as exc:
        TavilyClient(api_key=bad_key, budget=budget, base_url=stub_server.url)
    assert bad_key.strip() not in str(exc.value)
    assert stub_server.requests == []


def test_key_is_redacted_if_the_server_echoes_it(stub_server, client):
    stub_server.routes["/search"] = (400, {}, {"detail": {"error": f"bad key {KEY}"}})
    with pytest.raises(TavilyError) as exc:
        client.search("stack")
    assert KEY not in str(exc.value)
    assert "***" in str(exc.value)


def test_search_asks_for_at_most_20_results(stub_server, client):
    stub_server.routes["/search"] = search_response
    client.search("stack", max_results=50)
    assert stub_server.requests[0]["json"]["max_results"] == 20


def test_result_without_a_score_does_not_crash(stub_server, client):
    stub_server.routes["/search"] = lambda r: (200, {}, {"results": [
        {"url": "https://a.example", "title": "A", "content": "text"}]})
    assert client.search("stack")[0].score == 0.0


def test_unreadable_200_reply_is_a_tavily_error_and_stays_counted(stub_server, client, budget):
    stub_server.routes["/search"] = (200, {"Content-Type": "text/html"}, "<html>portal</html>")
    with pytest.raises(TavilyError, match="unreadable"):
        client.search("stack")
    assert budget.used == 1  # Tavily may have charged it, so keep the reservation


def test_timeout_after_sending_keeps_the_worst_case_counted(stub_server, budget):
    import time

    def slow(request):
        time.sleep(1.0)
        return search_response(request)

    stub_server.routes["/search"] = slow
    client = TavilyClient(api_key=KEY, budget=budget, base_url=stub_server.url, timeout=0.2,
                          sleep=RecordingSleep())
    with pytest.raises(TavilyError, match="may have been charged"):
        client.search("stack", search_depth="advanced")
    assert budget.used == 2
    assert len(stub_server.requests) == 1  # never retried: that could double-charge


def test_crawl_gets_a_longer_timeout_than_search(stub_server, budget, monkeypatch):
    import time

    from ragsearch.web.tavily import CRAWL_TIMEOUT

    assert CRAWL_TIMEOUT >= 150  # Tavily's own crawl timeout defaults to 150 s

    def slow_crawl(request):
        time.sleep(0.5)  # longer than the client's 0.2 s timeout
        return 200, {}, {"results": [{"url": "https://docs.example/1", "raw_content": "text"}]}

    monkeypatch.setattr("ragsearch.web.tavily.CRAWL_TIMEOUT", 2.0)  # a quick stand-in
    stub_server.routes["/crawl"] = slow_crawl
    client = TavilyClient(api_key=KEY, budget=budget, base_url=stub_server.url, timeout=0.2,
                          sleep=RecordingSleep())
    pages = client.crawl("https://docs.example", max_pages=5)
    assert [p.url for p in pages] == ["https://docs.example/1"]


def test_crawl_rejects_a_non_positive_page_limit(client):
    with pytest.raises(ValueError, match="max_pages"):
        client.crawl("https://docs.example", max_pages=0)
