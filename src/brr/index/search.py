"""NumPy brute-force top-k dot product search."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray


def top_k_dot_product(
    query: NDArray[np.floating],
    vectors: NDArray[np.floating],
    k: int,
) -> list[tuple[int, float]]:
    """Return top-k (index, score) pairs by dot product similarity.

    Uses numpy argpartition for O(N) selection instead of full sort.
    NaN-safe: NaN scores are treated as -inf.

    Returns:
        List of (index, score) tuples sorted by descending score.
    """
    if vectors.shape[0] == 0 or k <= 0:
        return []

    query_f32 = query.astype(np.float32, copy=False)
    vecs_f32 = vectors.astype(np.float32, copy=False)
    scores = vecs_f32 @ query_f32

    nan_mask = np.isnan(scores)
    if nan_mask.any():
        scores = np.where(nan_mask, -np.inf, scores)

    k = min(k, len(scores))
    if k == len(scores):
        top_indices = np.argsort(-scores)
    else:
        part_indices = np.argpartition(-scores, k)[:k]
        top_indices = part_indices[np.argsort(-scores[part_indices])]

    return [(int(idx), float(scores[idx])) for idx in top_indices]
