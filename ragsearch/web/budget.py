"""A local monthly counter that stops Tavily calls before the free plan runs out.

Credit costs (Tavily free plan, 1,000 credits/month):
search 1 (basic) or 2 (advanced); extract 1 per 5 URLs; map 1 per 10 pages;
crawl = map + extract. The default cap of 800 leaves a safety margin.
"""

import json
import math
from collections.abc import Callable
from datetime import date
from pathlib import Path

from ragsearch.fileutil import atomic_write_bytes, file_lock

DEFAULT_CAP = 800


class BudgetExceeded(RuntimeError):
    """Raised instead of making a Tavily call that would pass the monthly cap."""


def search_cost(depth: str = "basic") -> int:
    return 2 if depth == "advanced" else 1


def extract_cost(n_urls: int) -> int:
    return math.ceil(n_urls / 5)


def map_cost(n_pages: int) -> int:
    return math.ceil(n_pages / 10)


def crawl_cost(n_pages: int) -> int:
    return map_cost(n_pages) + extract_cost(n_pages)


class CreditBudget:
    """Monthly credit counter in a JSON file, safe across threads and processes.

    Callers ``reserve`` the worst-case cost before a call (an atomic check-and-book
    under a lock file), then ``adjust`` by the difference once the real cost is known.
    """

    def __init__(
        self,
        path: str | Path,
        cap: int = DEFAULT_CAP,
        today: Callable[[], date] = date.today,
    ):
        self.path = Path(path)
        self.cap = cap
        self._today = today

    @property
    def _lock(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    def _month(self) -> str:
        return self._today().strftime("%Y-%m")

    @property
    def used(self) -> int:
        if not self.path.exists():
            return 0
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return int(data["credits"]) if data["month"] == self._month() else 0
        except (ValueError, KeyError, TypeError) as err:
            # Never silently reset to zero: that could overspend real credits.
            raise BudgetExceeded(
                f"Tavily usage file {self.path} is unreadable ({err}); fix or delete it"
            ) from err

    @property
    def remaining(self) -> int:
        return max(self.cap - self.used, 0)

    def _check(self, used: int, credits: int) -> None:
        if used + credits > self.cap:
            raise BudgetExceeded(
                f"Tavily budget: {used}/{self.cap} credits used this month and this call "
                f"needs {credits}. Raise TAVILY_MONTHLY_CAP or use --provider local."
            )

    def _write(self, credits: int) -> None:
        data = {"month": self._month(), "credits": max(credits, 0)}
        atomic_write_bytes(self.path, json.dumps(data).encode("utf-8"))

    def ensure_available(self, credits: int) -> None:
        self._check(self.used, credits)

    def reserve(self, credits: int) -> int:
        """Book ``credits`` now, or raise ``BudgetExceeded`` and book nothing."""
        with file_lock(self._lock):
            used = self.used
            self._check(used, credits)
            self._write(used + credits)
        return credits

    def adjust(self, delta: int) -> None:
        """Add ``delta`` (negative to give back part of a reservation); never below 0."""
        if delta:
            with file_lock(self._lock):
                self._write(self.used + delta)

    def record(self, credits: int) -> None:
        """Add credits actually spent. Never refuses: real spending must be counted."""
        self.adjust(credits)
