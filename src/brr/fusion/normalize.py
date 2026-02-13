"""Min-max score normalization."""

from __future__ import annotations

from math import isclose
from typing import TypeAlias


_ScoredPairs: TypeAlias = list[tuple[str, float]]


def min_max_normalize(scored: _ScoredPairs) -> _ScoredPairs:
    """Normalize scores to [0, 1] via min-max scaling.

    If all scores are equal (or list has <= 1 element), all scores become 1.0.

    Returns:
        List of (doc_id, normalized_score) tuples.
    """
    if len(scored) <= 1:
        return [(doc_id, 1.0) for doc_id, _ in scored]

    scores = [score for _, score in scored]
    lo = min(scores)
    hi = max(scores)
    span = hi - lo

    if isclose(span, 0.0):
        return [(doc_id, 1.0) for doc_id, _ in scored]

    return [(doc_id, (score - lo) / span) for doc_id, score in scored]
