"""Tests for the TwoTierSearcher progressive generator."""

from brr.config import TwoTierConfig
from brr.core.types import SearchPhase
from brr.embed.hash_embedder import FnvHashEmbedder
from brr.fusion.two_tier import TwoTierSearcher
from brr.index.vector_index import VectorIndex


def _build_index(embedder: FnvHashEmbedder) -> VectorIndex:
    idx = VectorIndex(dimension=embedder.dimension, embedder_id=embedder.model_id)
    docs = [
        ("d1", "distributed consensus algorithm"),
        ("d2", "quick brown fox"),
        ("d3", "machine learning models"),
        ("d4", "search engine optimization"),
        ("d5", "python programming language"),
    ]
    for doc_id, text in docs:
        idx.add(doc_id, embedder.embed(text))
    return idx


def test_yields_initial_phase():
    emb = FnvHashEmbedder(dim=64)
    idx = _build_index(emb)
    config = TwoTierConfig(fast_only=True)
    searcher = TwoTierSearcher(index=idx, fast_embedder=emb, config=config)

    results = list(searcher.search("consensus algorithm", k=3))
    assert len(results) == 1
    assert results[0].phase == SearchPhase.INITIAL
    assert len(results[0].hits) <= 3


def test_fast_only_skips_refinement():
    emb = FnvHashEmbedder(dim=64)
    idx = _build_index(emb)
    config = TwoTierConfig(fast_only=True)
    searcher = TwoTierSearcher(index=idx, fast_embedder=emb, config=config)

    phases = [r.phase for r in searcher.search("test", k=5)]
    assert SearchPhase.REFINED not in phases
    assert SearchPhase.REFINEMENT_FAILED not in phases


def test_with_quality_yields_two_phases():
    emb = FnvHashEmbedder(dim=64)
    idx = _build_index(emb)
    # Use a second hash embedder as "quality" for testing
    quality_emb = FnvHashEmbedder(dim=64, ngram_size=4)
    config = TwoTierConfig(fast_only=False)
    searcher = TwoTierSearcher(
        index=idx, fast_embedder=emb, quality_embedder=quality_emb, config=config
    )

    results = list(searcher.search("machine learning", k=3))
    assert len(results) == 2
    assert results[0].phase == SearchPhase.INITIAL
    assert results[1].phase == SearchPhase.REFINED


def test_empty_query_returns_empty():
    emb = FnvHashEmbedder(dim=64)
    idx = _build_index(emb)
    searcher = TwoTierSearcher(index=idx, fast_embedder=emb)

    results = list(searcher.search("", k=5))
    assert len(results) == 1
    assert results[0].phase == SearchPhase.INITIAL
    assert results[0].hits == []


def test_search_empty_index():
    emb = FnvHashEmbedder(dim=64)
    idx = VectorIndex(dimension=64)
    searcher = TwoTierSearcher(index=idx, fast_embedder=emb)

    results = list(searcher.search("test query", k=5))
    assert len(results) >= 1
    assert results[0].phase == SearchPhase.INITIAL
