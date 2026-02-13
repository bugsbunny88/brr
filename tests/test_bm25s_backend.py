"""Tests for BM25 lexical backend (requires bm25s)."""

import pytest


try:
    import bm25s  # noqa: F401

    HAS_BM25S = True
except ImportError:
    HAS_BM25S = False

pytestmark = pytest.mark.skipif(not HAS_BM25S, reason="bm25s not installed")


@pytest.fixture
def backend():
    from brr.lexical.bm25s_backend import BM25SBackend

    b = BM25SBackend()
    b.index_documents(
        doc_ids=["d1", "d2", "d3"],
        texts=[
            "the quick brown fox jumps over the lazy dog",
            "machine learning algorithms for natural language processing",
            "the fox and the hound are friends",
        ],
    )
    return b


def test_search_returns_results(backend):
    results = backend.search("fox", limit=3)
    assert len(results) > 0
    doc_ids = [r[0] for r in results]
    assert "d1" in doc_ids or "d3" in doc_ids


def test_search_scores_are_positive(backend):
    results = backend.search("fox", limit=3)
    for _, score in results:
        assert score >= 0


def test_search_limit(backend):
    results = backend.search("the", limit=1)
    assert len(results) <= 1


def test_empty_index():
    from brr.lexical.bm25s_backend import BM25SBackend

    b = BM25SBackend()
    results = b.search("anything", limit=5)
    assert results == []
