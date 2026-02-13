"""Reciprocal Rank Fusion (RRF) with 4-level tie-breaking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from brr.core.types import FusedHit


_DEFAULT_K: Final = 60.0


@dataclass
class _RRFAccumulator:
    doc_id: str
    rrf_score: float = 0.0
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None
    in_both: bool = False


def _sort_key(entry: _RRFAccumulator) -> tuple[float, int, float, str]:
    return (
        -entry.rrf_score,
        -(1 if entry.in_both else 0),
        -(entry.lexical_score if entry.lexical_score is not None else float("-inf")),
        entry.doc_id,
    )


def reciprocal_rank_fusion(
    lexical_results: list[tuple[str, float]],
    semantic_results: list[tuple[str, float]],
    k: float = _DEFAULT_K,
) -> list[FusedHit]:
    """Fuse lexical and semantic ranked lists via RRF.

    RRF score for each source: 1 / (K + rank + 1), where rank is 0-based.
    4-level tie-breaking: score desc, in_both desc, lexical_score desc, doc_id asc.

    Returns:
        Sorted list of FusedHit combining both sources.
    """
    accum: dict[str, _RRFAccumulator] = {}

    for rank, (doc_id, score) in enumerate(lexical_results):
        if doc_id not in accum:
            accum[doc_id] = _RRFAccumulator(doc_id=doc_id)
        entry = accum[doc_id]
        entry.rrf_score += 1.0 / (k + rank + 1)
        entry.lexical_rank = rank
        entry.lexical_score = score

    for rank, (doc_id, score) in enumerate(semantic_results):
        if doc_id not in accum:
            accum[doc_id] = _RRFAccumulator(doc_id=doc_id)
        entry = accum[doc_id]
        entry.rrf_score += 1.0 / (k + rank + 1)
        entry.semantic_rank = rank
        entry.semantic_score = score

    for entry in accum.values():
        entry.in_both = entry.lexical_rank is not None and entry.semantic_rank is not None

    entries = sorted(accum.values(), key=_sort_key)
    return [_to_fused_hit(accumulated) for accumulated in entries]


def _to_fused_hit(accumulated: _RRFAccumulator) -> FusedHit:
    """Convert an RRF accumulator to a FusedHit.

    Returns:
        FusedHit with accumulated scores and ranks.
    """
    return FusedHit(
        doc_id=accumulated.doc_id,
        rrf_score=accumulated.rrf_score,
        lexical_rank=accumulated.lexical_rank,
        semantic_rank=accumulated.semantic_rank,
        lexical_score=accumulated.lexical_score,
        semantic_score=accumulated.semantic_score,
        in_both_sources=accumulated.in_both,
    )
