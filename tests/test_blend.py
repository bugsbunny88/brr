"""Tests for two-tier score blending."""

from brr.core.types import FusedHit
from brr.fusion.blend import blend_scores


def _hit(doc_id: str, rrf_score: float, **kwargs) -> FusedHit:
    return FusedHit(doc_id=doc_id, rrf_score=rrf_score, **kwargs)


def test_basic_blend():
    fast = [_hit("a", 1.0), _hit("b", 0.5)]
    quality = [_hit("a", 0.8), _hit("c", 0.9)]

    blended = blend_scores(fast, quality, quality_weight=0.7)
    by_id = {h.doc_id: h for h in blended}

    # "a" in both: 0.7*0.8 + 0.3*1.0 = 0.56 + 0.30 = 0.86
    assert abs(by_id["a"].rrf_score - 0.86) < 1e-10
    # "b" fast only: 0.3*0.5 = 0.15
    assert abs(by_id["b"].rrf_score - 0.15) < 1e-10
    # "c" quality only: 0.7*0.9 = 0.63
    assert abs(by_id["c"].rrf_score - 0.63) < 1e-10


def test_blend_preserves_order():
    fast = [_hit("x", 0.1)]
    quality = [_hit("y", 1.0)]

    blended = blend_scores(fast, quality, quality_weight=0.7)
    assert blended[0].doc_id == "y"  # 0.7*1.0 = 0.7
    assert blended[1].doc_id == "x"  # 0.3*0.1 = 0.03


def test_blend_empty():
    assert blend_scores([], [], quality_weight=0.7) == []


def test_blend_equal_weight():
    fast = [_hit("a", 1.0)]
    quality = [_hit("a", 1.0)]
    blended = blend_scores(fast, quality, quality_weight=0.5)
    assert abs(blended[0].rrf_score - 1.0) < 1e-10
