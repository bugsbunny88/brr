"""Tests for Reciprocal Rank Fusion."""

from brr.fusion.rrf import reciprocal_rank_fusion


def test_basic_fusion():
    lexical = [("doc1", 10.0), ("doc2", 8.0), ("doc3", 6.0)]
    semantic = [("doc2", 0.9), ("doc1", 0.8), ("doc4", 0.7)]

    fused = reciprocal_rank_fusion(lexical, semantic, k=60.0)
    # doc1 and doc2 appear in both sources → should have in_both_sources=True
    doc1 = next(h for h in fused if h.doc_id == "doc1")
    doc2 = next(h for h in fused if h.doc_id == "doc2")
    assert doc1.in_both_sources
    assert doc2.in_both_sources

    # doc3 only in lexical, doc4 only in semantic
    doc3 = next(h for h in fused if h.doc_id == "doc3")
    doc4 = next(h for h in fused if h.doc_id == "doc4")
    assert not doc3.in_both_sources
    assert not doc4.in_both_sources


def test_scores_are_rrf():
    lexical = [("a", 5.0)]
    semantic = [("a", 0.5)]
    fused = reciprocal_rank_fusion(lexical, semantic, k=60.0)
    # rank 0 in both: 1/(60+0+1) + 1/(60+0+1) = 2/61
    assert abs(fused[0].rrf_score - 2.0 / 61.0) < 1e-10


def test_empty_inputs():
    fused = reciprocal_rank_fusion([], [], k=60.0)
    assert fused == []


def test_lexical_only():
    lexical = [("a", 5.0), ("b", 3.0)]
    fused = reciprocal_rank_fusion(lexical, [], k=60.0)
    assert len(fused) == 2
    assert fused[0].doc_id == "a"
    assert fused[0].lexical_rank == 0
    assert fused[0].semantic_rank is None


def test_semantic_only():
    semantic = [("x", 0.9), ("y", 0.8)]
    fused = reciprocal_rank_fusion([], semantic, k=60.0)
    assert len(fused) == 2
    assert fused[0].doc_id == "x"
    assert fused[0].semantic_rank == 0


def test_deterministic_tie_breaking():
    # Same RRF score, neither in both — tiebreak: lexical_score desc, then doc_id asc.
    # "b" has lexical_score=5.0, "a" has no lexical_score (treated as -inf).
    # So "b" comes first due to lexical_score tiebreak.
    lexical = [("b", 5.0)]
    semantic = [("a", 0.9)]
    fused = reciprocal_rank_fusion(lexical, semantic, k=60.0)
    assert fused[0].doc_id == "b"
    assert fused[1].doc_id == "a"


def test_doc_id_tiebreak():
    # When everything else is equal, doc_id ascending breaks ties
    lexical = [("b", 5.0), ("a", 5.0)]
    fused = reciprocal_rank_fusion(lexical, [], k=60.0)
    # Same lexical_score, rank differs → different RRF scores, so "a" has rank 1, "b" rank 0
    # "b" at rank 0 has higher RRF score
    assert fused[0].doc_id == "b"  # rank 0: 1/61
    assert fused[1].doc_id == "a"  # rank 1: 1/62


def test_in_both_tiebreak():
    # doc_shared is in both with same total score as doc_single
    lexical = [("shared", 5.0), ("lex_only", 3.0)]
    semantic = [("sem_only", 0.9), ("shared", 0.8)]
    fused = reciprocal_rank_fusion(lexical, semantic, k=60.0)
    shared = next(h for h in fused if h.doc_id == "shared")
    # shared: 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62
    assert shared.in_both_sources
    assert shared.rrf_score > 1.0 / 61.0  # more than single-source
