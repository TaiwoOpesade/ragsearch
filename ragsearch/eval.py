"""Retrieval evaluation: recall@k and MRR@k over labelled question -> source pairs.

The labelled file is JSON: ``[{"question": "...", "expected_source": "stacks.md"}, ...]``.
A retrieved source matches when it equals ``expected_source`` or ends with
``"/" + expected_source``, so labels can be short file names or full URLs.

* recall@k: fraction of questions whose expected source is in the top k sources.
* MRR@k: mean of 1/rank of the first matching source (0 when not in the top k).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ragsearch.embedder import Embedder
from ragsearch.store import Store

CHUNKS_PER_SOURCE = 5  # fetch extra chunks so k *distinct* sources are usually found


class EvalError(ValueError):
    """Raised for a missing or malformed labelled dataset."""


@dataclass(frozen=True)
class Case:
    question: str
    expected_source: str


@dataclass(frozen=True)
class EvalReport:
    k: int
    n: int
    recall: float
    mrr: float
    ranks: list[int | None]  # 1-based rank of the expected source per question


def matches(source: str, expected: str) -> bool:
    return source == expected or source.endswith("/" + expected)


def _first_rank(ranked: list[str], expected: str) -> int | None:
    return next((i for i, s in enumerate(ranked, start=1) if matches(s, expected)), None)


def _check(ranked: list[list[str]], expected: list[str]) -> None:
    if not expected or len(ranked) != len(expected):
        raise ValueError("need one ranking per expected source, and at least one")


def recall_at_k(ranked: list[list[str]], expected: list[str], k: int) -> float:
    _check(ranked, expected)
    hits = sum(_first_rank(r[:k], e) is not None for r, e in zip(ranked, expected, strict=True))
    return hits / len(expected)


def mean_reciprocal_rank(ranked: list[list[str]], expected: list[str]) -> float:
    _check(ranked, expected)
    ranks = [_first_rank(r, e) for r, e in zip(ranked, expected, strict=True)]
    return sum(1 / r for r in ranks if r is not None) / len(expected)


def ranked_sources(query: str, embedder: Embedder, store: Store, k: int) -> list[str]:
    """Top ``k`` distinct sources for ``query``, best first."""
    results = store.search(embedder.embed([query])[0], k * CHUNKS_PER_SOURCE)
    unique = list(dict.fromkeys(r.item["source"] for r in results))
    return unique[:k]


def load_cases(path: str | Path) -> list[Case]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cases = [Case(str(row["question"]), str(row["expected_source"])) for row in data]
    except (OSError, ValueError) as err:
        raise EvalError(f"Could not read labelled dataset {path}: {err}") from err
    except (KeyError, TypeError) as err:
        raise EvalError(
            f"{path}: every entry needs 'question' and 'expected_source' ({err!r})"
        ) from err
    if not cases:
        raise EvalError(f"{path} contains no questions")
    return cases


def evaluate(cases: list[Case], embedder: Embedder, store: Store, k: int = 5) -> EvalReport:
    ranked = [ranked_sources(c.question, embedder, store, k) for c in cases]
    expected = [c.expected_source for c in cases]
    return EvalReport(
        k=k,
        n=len(cases),
        recall=recall_at_k(ranked, expected, k),
        mrr=mean_reciprocal_rank(ranked, expected),
        ranks=[_first_rank(r, e) for r, e in zip(ranked, expected, strict=True)],
    )
