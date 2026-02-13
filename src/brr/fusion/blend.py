"""Two-tier score blending (quality_weight * quality + (1-quality_weight) * fast)."""

from __future__ import annotations

from brr.core.types import FusedHit


def blend_scores(
    fast_hits: list[FusedHit],
    quality_hits: list[FusedHit],
    quality_weight: float = 0.7,
) -> list[FusedHit]:
    """Blend fast-tier and quality-tier RRF scores.

    For documents in both tiers: blended = quality_weight * quality + (1 - quality_weight) * fast.
    For documents in only one tier: use that tier's score scaled by its weight.
    Results are re-sorted by blended score with the same 4-level tie-breaking.

    Returns:
        Sorted list of FusedHit with blended scores.
    """
    fast_weight = 1.0 - quality_weight

    fast_by_id: dict[str, FusedHit] = {hit.doc_id: hit for hit in fast_hits}
    quality_by_id: dict[str, FusedHit] = {hit.doc_id: hit for hit in quality_hits}

    all_doc_ids = set(fast_by_id.keys()) | set(quality_by_id.keys())
    blended = [
        _blend_single(
            doc_id,
            fast_by_id.get(doc_id),
            quality_by_id.get(doc_id),
            quality_weight,
            fast_weight,
        )
        for doc_id in all_doc_ids
    ]

    blended.sort(key=_blend_sort_key)
    return blended


def _blend_single(
    doc_id: str,
    fast_hit: FusedHit | None,
    qual_hit: FusedHit | None,
    quality_weight: float,
    fast_weight: float,
) -> FusedHit:
    """Build a blended FusedHit from fast and quality tier hits.

    Returns:
        FusedHit with blended RRF score.

    Raises:
        ValueError: If both hits are None.
    """
    fast_score = fast_hit.rrf_score if fast_hit else 0.0
    qual_score = qual_hit.rrf_score if qual_hit else 0.0
    score = quality_weight * qual_score + fast_weight * fast_score

    # Prefer quality-tier metadata if available, else fast
    ref = qual_hit or fast_hit
    if ref is None:  # pragma: no cover
        msg = "both hits cannot be None"
        raise ValueError(msg)

    return FusedHit(
        doc_id=doc_id,
        rrf_score=score,
        lexical_rank=ref.lexical_rank,
        semantic_rank=ref.semantic_rank,
        lexical_score=ref.lexical_score,
        semantic_score=ref.semantic_score,
        in_both_sources=ref.in_both_sources,
    )


def _blend_sort_key(hit: FusedHit) -> tuple[float, int, float, str]:
    """Sort key for blended results: same 4-level tie-breaking as RRF.

    Returns:
        Tuple for sorting by score desc, in_both desc, lexical_score desc, doc_id asc.
    """
    return (
        -hit.rrf_score,
        -(1 if hit.in_both_sources else 0),
        -(hit.lexical_score if hit.lexical_score is not None else float("-inf")),
        hit.doc_id,
    )
