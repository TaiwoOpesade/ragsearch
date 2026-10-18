"""Behaviour every Index backend must share, run against each implementation."""

import numpy as np
import pytest

from ragsearch.faiss_index import FaissIndex
from ragsearch.index import Index, VectorIndex, index_backend


@pytest.fixture(params=[VectorIndex, FaissIndex], ids=["numpy", "faiss"])
def backend(request):
    return request.param


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def make(backend):
    idx = backend(dim=3)
    idx.add(
        np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(1, 1, 0)]),
        [{"id": "x"}, {"id": "y"}, {"id": "xy"}],
    )
    return idx


def test_satisfies_index_protocol(backend):
    assert isinstance(backend(dim=3), Index)


def test_dim_and_len(backend):
    idx = make(backend)
    assert idx.dim == 3
    assert len(idx) == 3


def test_best_match_first_with_cosine_scores(backend):
    results = make(backend).search(unit(1, 0, 0), k=3)
    assert [r.item["id"] for r in results] == ["x", "xy", "y"]
    assert [r.score for r in results] == pytest.approx([1.0, 0.7071, 0.0], abs=1e-3)


def test_k_is_capped_at_index_size(backend):
    assert len(make(backend).search(unit(1, 0, 0), k=50)) == 3


def test_k_limits_results(backend):
    assert len(make(backend).search(unit(1, 0, 0), k=2)) == 2


def test_empty_index_returns_empty_list(backend):
    assert backend(dim=3).search(unit(1, 0, 0), k=5) == []


def test_vectors_are_normalised_on_add(backend):
    idx = backend(dim=3)
    idx.add(np.array([[10.0, 0.0, 0.0]]), [{"id": "big"}])
    assert idx.search(np.array([5.0, 0.0, 0.0]), k=1)[0].score == pytest.approx(1.0)


def test_zero_vector_scores_zero_not_nan(backend):
    idx = backend(dim=3)
    idx.add(np.zeros((1, 3)), [{"id": "empty"}])
    assert idx.search(unit(1, 0, 0), k=1)[0].score == 0.0


def test_repeated_add_appends(backend):
    idx = make(backend)
    idx.add(unit(0, 0, 1)[None, :], [{"id": "z"}])
    assert len(idx) == 4
    assert idx.search(unit(0, 0, 1), k=1)[0].item["id"] == "z"


@pytest.mark.parametrize(
    "vectors,items",
    [(np.zeros((1, 4)), [{"id": "a"}]), (np.zeros((2, 3)), [{"id": "a"}])],
    ids=["wrong-dim", "count-mismatch"],
)
def test_bad_add_raises(backend, vectors, items):
    with pytest.raises(ValueError):
        backend(dim=3).add(vectors, items)


def test_wrong_query_dimension_raises(backend):
    with pytest.raises(ValueError, match="dim"):
        make(backend).search(np.zeros(4), k=1)


def test_non_positive_k_raises(backend):
    with pytest.raises(ValueError):
        make(backend).search(unit(1, 0, 0), k=0)


def test_save_and_load_round_trip(backend, tmp_path):
    path = tmp_path / "index.npz"
    original = make(backend)
    original.save(path)
    loaded = backend.load(path)
    assert (loaded.dim, len(loaded)) == (3, 3)
    a = original.search(unit(1, 0.2, 0), k=3)
    b = loaded.search(unit(1, 0.2, 0), k=3)
    assert [r.item for r in a] == [r.item for r in b]
    assert [r.score for r in a] == pytest.approx([r.score for r in b])


def test_save_and_load_empty(backend, tmp_path):
    path = tmp_path / "empty.npz"
    backend(dim=5).save(path)
    loaded = backend.load(path)
    assert (loaded.dim, len(loaded)) == (5, 0)


def test_backends_agree_on_random_data():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(200, 16)).astype(np.float32)
    items = [{"id": i} for i in range(200)]
    numpy_idx, faiss_idx = VectorIndex(16), FaissIndex(16)
    numpy_idx.add(vectors, items)
    faiss_idx.add(vectors, items)
    for query in rng.normal(size=(5, 16)):
        a, b = numpy_idx.search(query, k=10), faiss_idx.search(query, k=10)
        assert [r.item for r in a] == [r.item for r in b]
        assert [r.score for r in a] == pytest.approx([r.score for r in b], abs=1e-5)


def test_index_backend_looks_up_classes_by_name():
    assert index_backend("numpy") is VectorIndex
    assert index_backend("faiss") is FaissIndex
    with pytest.raises(ValueError, match="annoy"):
        index_backend("annoy")
