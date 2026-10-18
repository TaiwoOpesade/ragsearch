import json
from pathlib import Path

import pytest

from ragsearch.cli import main
from ragsearch.config import Settings
from ragsearch.embedder import HashEmbedder
from ragsearch.eval import (
    Case,
    EvalError,
    evaluate,
    load_cases,
    matches,
    mean_reciprocal_rank,
    ranked_sources,
    recall_at_k,
)
from ragsearch.loaders import Document, load_documents
from ragsearch.pipeline import ingest
from ragsearch.store import Store

FIXTURES = Path(__file__).parent / "fixtures"

# Hand-worked example:
#   q1 ranks a,b,c expecting a -> rank 1
#   q2 ranks c,a,b expecting a -> rank 2
#   q3 ranks x,y,z expecting b -> not found
RANKED = [["a", "b", "c"], ["c", "a", "b"], ["x", "y", "z"]]
EXPECTED = ["a", "a", "b"]


@pytest.mark.parametrize("k,expected", [(1, 1 / 3), (2, 2 / 3), (3, 2 / 3)])
def test_recall_at_k_hand_computed(k, expected):
    assert recall_at_k(RANKED, EXPECTED, k) == pytest.approx(expected)


def test_mrr_hand_computed():
    # (1/1 + 1/2 + 0) / 3 = 0.5
    assert mean_reciprocal_rank(RANKED, EXPECTED) == pytest.approx(0.5)


def test_mrr_second_hand_computed_example():
    # (1/2 + 1/1) / 2 = 0.75 ; recall@1 = 1/2
    ranked, expected = [["b", "a"], ["a"]], ["a", "a"]
    assert mean_reciprocal_rank(ranked, expected) == pytest.approx(0.75)
    assert recall_at_k(ranked, expected, 1) == pytest.approx(0.5)


def test_mrr_only_counts_the_first_match():
    assert mean_reciprocal_rank([["x", "a", "a"]], ["a"]) == pytest.approx(0.5)


def test_metrics_reject_mismatched_or_empty_input():
    with pytest.raises(ValueError):
        recall_at_k([["a"]], ["a", "b"], 1)
    with pytest.raises(ValueError):
        mean_reciprocal_rank([], [])


def test_expected_source_matches_by_path_suffix():
    full = "C:/Users/me/project/tests/fixtures/stacks.md"
    assert matches(full, "stacks.md")
    assert matches(full, "fixtures/stacks.md")
    assert matches("https://web.example/a", "https://web.example/a")
    assert not matches("C:/x/mystacks.md", "stacks.md")


def test_ranked_sources_collapses_chunks_of_the_same_source(tmp_path):
    emb = HashEmbedder(dim=128)
    store = Store(tmp_path, embedder_name=emb.name)
    ingest([Document.from_text("a.md", "apple " * 30), Document.from_text("b.md", "pear")],
           emb, store, chunk_size=5, overlap=0)
    assert ranked_sources("apple", emb, store, k=2) == ["a.md", "b.md"]


def test_load_cases_reads_labelled_json():
    cases = load_cases(FIXTURES / "labelled.json")
    assert cases[0] == Case("Which data structure is last in first out?", "stacks.md")
    assert len(cases) == 6


@pytest.mark.parametrize("content", ["not json", "[]", '[{"question": "q"}]'])
def test_load_cases_rejects_bad_files(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(EvalError):
        load_cases(path)


def test_evaluate_on_fixtures_finds_every_expected_source_in_top_3(tmp_path):
    emb = HashEmbedder(dim=512)
    store = Store(tmp_path, embedder_name=emb.name)
    ingest(load_documents(FIXTURES), emb, store)
    report = evaluate(load_cases(FIXTURES / "labelled.json"), emb, store, k=3)
    assert report.n == 6
    assert report.recall == pytest.approx(1.0)
    assert report.mrr == pytest.approx(
        sum(1 / r for r in report.ranks) / 6
    )


def test_eval_cli_prints_metrics(tmp_path, capsys):
    settings = Settings(data_dir=tmp_path / "data", embedder="hash")
    main(["ingest", str(FIXTURES)], settings=settings)
    capsys.readouterr()
    code = main(["eval", str(FIXTURES / "labelled.json"), "-k", "3"], settings=settings)
    out = capsys.readouterr().out
    assert code == 0
    assert "recall@3: 1.000" in out
    assert "MRR@3:" in out


def test_eval_cli_reports_bad_file(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"nope": 1}]), encoding="utf-8")
    code = main(["eval", str(bad)], settings=Settings(data_dir=tmp_path, embedder="hash"))
    assert code == 1
    assert "expected_source" in capsys.readouterr().err
