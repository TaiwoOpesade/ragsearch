import numpy as np
import pytest

from ragsearch.index import VectorIndex


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def make_index():
    idx = VectorIndex(dim=3)
    idx.add(
        np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(1, 1, 0)]),
        [{"id": "x"}, {"id": "y"}, {"id": "xy"}],
    )
    return idx


def test_len_counts_added_vectors():
    assert len(make_index()) == 3


def test_search_returns_best_match_first_with_cosine_score():
    results = make_index().search(unit(1, 0, 0), k=3)
    assert [r.item["id"] for r in results] == ["x", "xy", "y"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.7071, abs=1e-3)
    assert results[2].score == pytest.approx(0.0, abs=1e-6)


def test_search_k_larger_than_index_returns_everything():
    assert len(make_index().search(unit(1, 0, 0), k=50)) == 3


def test_search_k_limits_results():
    assert len(make_index().search(unit(1, 0, 0), k=1)) == 1


def test_search_empty_index_returns_empty_list():
    assert VectorIndex(dim=3).search(unit(1, 0, 0), k=5) == []


def test_add_normalises_vectors_so_scale_does_not_matter():
    idx = VectorIndex(dim=3)
    idx.add(np.array([[10.0, 0.0, 0.0]]), [{"id": "big"}])
    assert idx.search(unit(1, 0, 0), k=1)[0].score == pytest.approx(1.0)


def test_add_wrong_dimension_raises():
    with pytest.raises(ValueError, match="dim"):
        VectorIndex(dim=3).add(np.zeros((1, 4)), [{"id": "a"}])


def test_add_mismatched_item_count_raises():
    with pytest.raises(ValueError):
        VectorIndex(dim=3).add(np.zeros((2, 3)), [{"id": "a"}])


def test_search_wrong_query_dimension_raises():
    with pytest.raises(ValueError, match="dim"):
        make_index().search(np.zeros(4), k=1)


def test_search_with_non_positive_k_raises():
    with pytest.raises(ValueError):
        make_index().search(unit(1, 0, 0), k=0)


def test_zero_vector_is_stored_without_nan_scores():
    idx = VectorIndex(dim=3)
    idx.add(np.zeros((1, 3)), [{"id": "empty"}])
    assert idx.search(unit(1, 0, 0), k=1)[0].score == 0.0


def test_add_can_be_called_repeatedly():
    idx = make_index()
    idx.add(unit(0, 0, 1)[None, :], [{"id": "z"}])
    assert len(idx) == 4
    assert idx.search(unit(0, 0, 1), k=1)[0].item["id"] == "z"


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "index.npz"
    original = make_index()
    original.save(path)
    loaded = VectorIndex.load(path)
    assert len(loaded) == 3
    assert loaded.dim == 3
    a = original.search(unit(1, 0.2, 0), k=3)
    b = loaded.search(unit(1, 0.2, 0), k=3)
    assert [r.item for r in a] == [r.item for r in b]
    assert [r.score for r in a] == pytest.approx([r.score for r in b])


def test_save_and_load_empty_index(tmp_path):
    path = tmp_path / "empty.npz"
    VectorIndex(dim=5).save(path)
    loaded = VectorIndex.load(path)
    assert len(loaded) == 0
    assert loaded.dim == 5
