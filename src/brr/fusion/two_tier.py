"""TwoTierSearcher — progressive generator orchestrating the full pipeline."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from brr.config import TwoTierConfig
from brr.core.canonicalize import canonicalize_query
from brr.core.query_class import QueryClass
from brr.core.query_class import adaptive_budget
from brr.core.query_class import classify_query
from brr.core.types import FusedHit
from brr.core.types import SearchPhase
from brr.core.types import SearchResult
from brr.fusion.blend import blend_scores
from brr.fusion.rrf import reciprocal_rank_fusion


if TYPE_CHECKING:
    from collections.abc import Generator

    from brr.core.protocols import Embedder
    from brr.core.protocols import LexicalBackend
    from brr.index.vector_index import VectorIndex


logger = logging.getLogger(__name__)


class TwoTierSearcher:
    """Orchestrates two-tier progressive hybrid search.

    Yields SearchResult objects via a generator:
    1. Initial: fast embedder + lexical -> RRF
    2. Refined: quality embedder re-scores top candidates -> blend -> re-rank

    Usage:
        for result in searcher.search("query", k=10):
            if result.phase == SearchPhase.INITIAL:
                display(result.hits)  # fast, show immediately
            elif result.phase == SearchPhase.REFINED:
                update_display(result.hits)  # better rankings
    """

    def __init__(
        self,
        index: VectorIndex,
        fast_embedder: Embedder,
        config: TwoTierConfig | None = None,
        quality_embedder: Embedder | None = None,
        lexical_backend: LexicalBackend | None = None,
    ) -> None:
        self._index = index
        self._fast_embedder = fast_embedder
        self._quality_embedder = quality_embedder
        self._lexical = lexical_backend
        self._config = config or TwoTierConfig()

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> Generator[SearchResult, None, None]:
        """Progressive search generator.

        Yields:
            SearchResult for each phase (INITIAL, then optionally REFINED).
        """
        query_text = canonicalize_query(query)
        qclass = classify_query(query_text)
        if qclass == QueryClass.EMPTY:
            yield SearchResult(phase=SearchPhase.INITIAL, hits=[])
            return

        semantic_k, lexical_k = _compute_budgets(qclass, k, self._config.candidate_multiplier)
        start = time.monotonic()
        semantic_results = _run_semantic(self._fast_embedder, self._index, query_text, semantic_k)
        lexical_results = _run_lexical(self._lexical, query_text, lexical_k)
        initial_hits = _fuse(lexical_results, semantic_results, self._config.rrf_k)[:k]

        logger.info(
            "Phase 1: %.1fms, %d hits", (time.monotonic() - start) * 1000, len(initial_hits)
        )
        yield SearchResult(phase=SearchPhase.INITIAL, hits=initial_hits)

        if self._config.fast_only or self._quality_embedder is None:
            return

        yield from self._run_refinement(
            query_text,
            semantic_k,
            lexical_results=lexical_results,
            initial_hits=initial_hits,
            k=k,
        )

    def _compute_quality_blend(
        self,
        query_text: str,
        semantic_k: int,
        lexical_results: list[tuple[str, float]],
        initial_hits: list[FusedHit],
        k: int,
    ) -> list[FusedHit]:
        """Score with quality embedder and blend with initial results.

        Returns:
            Blended and re-ranked hits.
        """
        quality_vec = self._quality_embedder.embed(query_text)  # type: ignore[union-attr]
        quality_hits = self._index.search(quality_vec, k=semantic_k)
        quality_semantic = [(hit.doc_id, hit.score) for hit in quality_hits]
        quality_fused = _fuse(lexical_results, quality_semantic, self._config.rrf_k)
        return blend_scores(initial_hits, quality_fused[:k], self._config.quality_weight)

    def _run_refinement(
        self,
        query_text: str,
        semantic_k: int,
        *,
        lexical_results: list[tuple[str, float]],
        initial_hits: list[FusedHit],
        k: int,
    ) -> Generator[SearchResult, None, None]:
        """Run quality refinement phase.

        Yields:
            SearchResult with REFINED or REFINEMENT_FAILED phase.
        """
        start = time.monotonic()
        try:
            blended = self._compute_quality_blend(
                query_text,
                semantic_k,
                lexical_results,
                initial_hits,
                k,
            )
        except Exception:
            logger.warning("Quality refinement failed", exc_info=True)
            yield SearchResult(phase=SearchPhase.REFINEMENT_FAILED, hits=initial_hits)
        else:
            logger.info(
                "Phase 2: %.1fms, %d hits", (time.monotonic() - start) * 1000, len(blended)
            )
            yield SearchResult(phase=SearchPhase.REFINED, hits=blended[:k])


def _compute_budgets(
    qclass: QueryClass,
    k: int,
    multiplier: int,
) -> tuple[int, int]:
    """Compute semantic and lexical retrieval budgets.

    Returns:
        Tuple of (semantic_k, lexical_k).
    """
    budget = adaptive_budget(qclass, multiplier)
    return k * budget.semantic_multiplier, k * budget.lexical_multiplier


def _run_semantic(
    embedder: Embedder,
    index: VectorIndex,
    query_text: str,
    semantic_k: int,
) -> list[tuple[str, float]]:
    """Run semantic search with the fast embedder.

    Returns:
        List of (doc_id, score) pairs, empty on failure.
    """
    if semantic_k <= 0:
        return []
    try:
        query_vec = embedder.embed(query_text)
        hits = index.search(query_vec, k=semantic_k)
    except Exception:
        logger.warning("Fast embedding failed, continuing without semantic", exc_info=True)
        return []
    else:
        return [(hit.doc_id, hit.score) for hit in hits]


def _run_lexical(
    backend: LexicalBackend | None,
    query_text: str,
    lexical_k: int,
) -> list[tuple[str, float]]:
    """Run lexical search if a backend is available.

    Returns:
        List of (doc_id, score) pairs, empty on failure or no backend.
    """
    if backend is None or lexical_k <= 0:
        return []
    try:
        return backend.search(query_text, limit=lexical_k)
    except Exception:
        logger.warning("Lexical search failed, continuing without lexical", exc_info=True)
        return []


def _fuse(
    lexical: list[tuple[str, float]],
    semantic: list[tuple[str, float]],
    rrf_k: float,
) -> list[FusedHit]:
    """Fuse lexical and semantic results via RRF or single-source fallback.

    Returns:
        Sorted list of FusedHit.
    """
    if lexical and semantic:
        return reciprocal_rank_fusion(lexical, semantic, k=rrf_k)
    if semantic:
        return [
            FusedHit(doc_id=doc_id, rrf_score=score, semantic_rank=idx, semantic_score=score)
            for idx, (doc_id, score) in enumerate(semantic)
        ]
    if lexical:
        return [
            FusedHit(doc_id=doc_id, rrf_score=score, lexical_rank=idx, lexical_score=score)
            for idx, (doc_id, score) in enumerate(lexical)
        ]
    return []
