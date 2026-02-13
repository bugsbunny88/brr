"""Tests for the vector index."""

import tempfile
from pathlib import Path

import pytest

from brr.core.errors import DimensionMismatchError
from brr.core.errors import IndexCorruptedError
from brr.index.vector_index import VectorIndex


def test_add_and_search():
    idx = VectorIndex(dimension=4, embedder_id="test")
    idx.add("doc1", [1.0, 0.0, 0.0, 0.0])
    idx.add("doc2", [0.0, 1.0, 0.0, 0.0])
    idx.add("doc3", [0.7, 0.7, 0.0, 0.0])

    results = idx.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert len(results) == 2
    assert results[0].doc_id == "doc1"
    assert results[0].score == pytest.approx(1.0)


def test_add_batch():
    idx = VectorIndex(dimension=3)
    idx.add_batch(
        ["a", "b", "c"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert idx.count == 3
    assert idx.doc_ids == ["a", "b", "c"]


def test_dimension_mismatch_on_add():
    idx = VectorIndex(dimension=4)
    with pytest.raises(DimensionMismatchError):
        idx.add("doc1", [1.0, 0.0])


def test_dimension_mismatch_on_search():
    idx = VectorIndex(dimension=4)
    idx.add("doc1", [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(DimensionMismatchError):
        idx.search([1.0, 0.0], k=1)


def test_save_load_roundtrip():
    idx = VectorIndex(dimension=3, embedder_id="test-emb", use_f16=True)
    idx.add("doc1", [1.0, 0.0, 0.0])
    idx.add("doc2", [0.0, 1.0, 0.0])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_index"
        idx.save(path)

        loaded = VectorIndex.load(path)
        assert loaded.dimension == 3
        assert loaded.embedder_id == "test-emb"
        assert loaded.count == 2
        assert loaded.doc_ids == ["doc1", "doc2"]

        # Verify search still works after load
        results = loaded.search([1.0, 0.0, 0.0], k=1)
        assert results[0].doc_id == "doc1"


def test_save_load_f32():
    idx = VectorIndex(dimension=2, use_f16=False)
    idx.add("a", [0.5, 0.5])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "f32_index"
        idx.save(path)

        loaded = VectorIndex.load(path)
        results = loaded.search([0.5, 0.5], k=1)
        assert results[0].doc_id == "a"


def test_empty_index_search():
    idx = VectorIndex(dimension=4)
    results = idx.search([1.0, 0.0, 0.0, 0.0], k=5)
    assert results == []


def test_load_missing_file():
    with pytest.raises(IndexCorruptedError):
        VectorIndex.load("/nonexistent/path/index")


def test_k_larger_than_count():
    idx = VectorIndex(dimension=2)
    idx.add("a", [1.0, 0.0])
    idx.add("b", [0.0, 1.0])
    results = idx.search([1.0, 0.0], k=100)
    assert len(results) == 2
