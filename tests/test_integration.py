"""Integration tests: end-to-end search pipeline with hash embedder."""

from brr import FnvHashEmbedder
from brr import SearchPhase
from brr import TwoTierConfig
from brr import TwoTierSearcher
from brr import VectorIndex
from brr import canonicalize
from brr import classify_query
from brr import reciprocal_rank_fusion
from tests.conftest import SAMPLE_DOCS


def test_full_pipeline_hash_only():
    """End-to-end: index → search → fuse → results with hash embedder."""
    embedder = FnvHashEmbedder(dim=128)
    index = VectorIndex(dimension=128, embedder_id=embedder.model_id)

    # Index all sample docs with canonicalization
    for doc_id, text in SAMPLE_DOCS:
        clean = canonicalize(text)
        vec = embedder.embed(clean)
        index.add(doc_id, vec)

    assert index.count == len(SAMPLE_DOCS)

    # Search
    config = TwoTierConfig(fast_only=True)
    searcher = TwoTierSearcher(index=index, fast_embedder=embedder, config=config)

    results = list(searcher.search("distributed consensus fault tolerant", k=5))
    assert len(results) == 1
    assert results[0].phase == SearchPhase.INITIAL
    assert len(results[0].hits) > 0

    # doc-1 should rank well for this query
    hit_ids = [h.doc_id for h in results[0].hits]
    assert "doc-1" in hit_ids


def test_rrf_standalone():
    """Test RRF fusion directly with mock ranked lists."""
    lexical = [("doc-1", 10.0), ("doc-4", 8.0), ("doc-9", 6.0)]
    semantic = [("doc-4", 0.95), ("doc-1", 0.90), ("doc-8", 0.85)]

    fused = reciprocal_rank_fusion(lexical, semantic, k=60.0)

    # doc-1 and doc-4 in both → should be top-ranked
    top_ids = [h.doc_id for h in fused[:2]]
    assert "doc-1" in top_ids
    assert "doc-4" in top_ids


def test_two_tier_with_quality():
    """Two-tier search with hash embedders simulating fast + quality."""
    fast = FnvHashEmbedder(dim=64, ngram_size=3)
    quality = FnvHashEmbedder(dim=64, ngram_size=5)

    index = VectorIndex(dimension=64, embedder_id=fast.model_id)
    for doc_id, text in SAMPLE_DOCS:
        index.add(doc_id, fast.embed(text))

    config = TwoTierConfig(fast_only=False, quality_weight=0.7)
    searcher = TwoTierSearcher(
        index=index, fast_embedder=fast, quality_embedder=quality, config=config
    )

    results = list(searcher.search("vector similarity cosine search", k=5))
    assert len(results) == 2
    assert results[0].phase == SearchPhase.INITIAL
    assert results[1].phase == SearchPhase.REFINED


def test_query_classification_integration():
    """Query classification feeds into the searcher correctly."""
    assert classify_query("").name == "EMPTY"
    assert classify_query("br-123").name == "IDENTIFIER"
    assert classify_query("error handling").name == "SHORT_KEYWORD"
    assert classify_query("how does search work in this system").name == "NATURAL_LANGUAGE"
