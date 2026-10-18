import json
from datetime import date

import pytest

from ragsearch.web.budget import (
    BudgetExceeded,
    CreditBudget,
    crawl_cost,
    extract_cost,
    map_cost,
    search_cost,
)


def budget(tmp_path, cap=10, today=date(2026, 10, 1)):
    return CreditBudget(tmp_path / "tavily_usage.json", cap=cap, today=lambda: today)


@pytest.mark.parametrize("depth,cost", [("basic", 1), ("advanced", 2)])
def test_search_cost(depth, cost):
    assert search_cost(depth) == cost


@pytest.mark.parametrize("urls,cost", [(1, 1), (5, 1), (6, 2), (10, 2), (11, 3)])
def test_extract_costs_one_credit_per_five_urls(urls, cost):
    assert extract_cost(urls) == cost


@pytest.mark.parametrize("pages,cost", [(1, 1), (10, 1), (11, 2)])
def test_map_costs_one_credit_per_ten_pages(pages, cost):
    assert map_cost(pages) == cost


def test_crawl_costs_map_plus_extract():
    assert crawl_cost(30) == map_cost(30) + extract_cost(30) == 3 + 6


def test_zero_urls_or_pages_cost_nothing():
    assert extract_cost(0) == map_cost(0) == crawl_cost(0) == 0


def test_records_usage_and_persists_between_instances(tmp_path):
    budget(tmp_path).record(3)
    again = budget(tmp_path)
    assert again.used == 3
    assert again.remaining == 7


def test_call_that_fits_exactly_under_the_cap_is_allowed(tmp_path):
    b = budget(tmp_path, cap=10)
    b.record(8)
    b.ensure_available(2)  # 8 + 2 == cap: fine


def test_refuses_calls_beyond_the_cap_with_a_clear_error(tmp_path):
    b = budget(tmp_path, cap=10)
    b.record(9)
    with pytest.raises(BudgetExceeded, match=r"9/10 .*needs 2"):
        b.ensure_available(2)


def test_spending_over_the_cap_is_still_recorded_and_blocks_further_calls(tmp_path):
    # Credits already charged by Tavily must never be under-counted.
    b = budget(tmp_path, cap=5)
    b.record(6)
    assert b.used == 6
    assert b.remaining == 0
    with pytest.raises(BudgetExceeded):
        b.ensure_available(1)


def test_new_month_starts_from_zero(tmp_path):
    budget(tmp_path, today=date(2026, 10, 31)).record(10)
    november = budget(tmp_path, today=date(2026, 11, 1))
    assert november.used == 0
    november.ensure_available(10)


def test_file_records_month_and_credits(tmp_path):
    budget(tmp_path).record(4)
    data = json.loads((tmp_path / "tavily_usage.json").read_text())
    assert data == {"month": "2026-10", "credits": 4}


def test_corrupt_usage_file_refuses_rather_than_resetting(tmp_path):
    (tmp_path / "tavily_usage.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(BudgetExceeded, match="usage file"):
        budget(tmp_path).ensure_available(1)


# --- reservations: check-and-book atomically, then settle ---------------------


def test_reserve_books_credits_immediately(tmp_path):
    b = budget(tmp_path, cap=10)
    assert b.reserve(3) == 3
    assert budget(tmp_path, cap=10).used == 3  # visible to other instances at once


def test_reserve_refuses_beyond_the_cap_and_books_nothing(tmp_path):
    b = budget(tmp_path, cap=10)
    b.record(9)
    with pytest.raises(BudgetExceeded, match=r"9/10"):
        b.reserve(2)
    assert b.used == 9


def test_adjust_settles_a_reservation_to_the_real_cost(tmp_path):
    b = budget(tmp_path, cap=10)
    b.reserve(9)  # worst case for a crawl
    b.adjust(3 - 9)  # Tavily actually charged 3
    assert b.used == 3


def test_adjust_never_goes_below_zero(tmp_path):
    b = budget(tmp_path, cap=10)
    b.adjust(-5)
    assert b.used == 0


def test_concurrent_reservations_never_exceed_the_cap(tmp_path):
    import threading

    b = budget(tmp_path, cap=10)
    granted, refused = [], []

    def worker():
        try:
            granted.append(b.reserve(1))
        except BudgetExceeded:
            refused.append(True)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(granted) == 10
    assert len(refused) == 10
    assert b.used == 10


def test_usage_file_is_written_atomically_with_no_leftovers(tmp_path):
    b = budget(tmp_path)
    b.record(1)
    b.reserve(2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tavily_usage.json"]


def test_a_budget_locked_by_another_process_refuses_cleanly(tmp_path):
    from ragsearch.fileutil import file_lock

    b = CreditBudget(tmp_path / "tavily_usage.json", cap=10, lock_timeout=0.2)
    with file_lock(tmp_path / "tavily_usage.json.lock"), \
            pytest.raises(BudgetExceeded, match="locked"):
        b.reserve(1)
