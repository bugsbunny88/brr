"""EmbedderStack: auto-detection with graceful fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brr.embed.hash_embedder import FnvHashEmbedder


if TYPE_CHECKING:
    from brr.core.protocols import Embedder


logger = logging.getLogger(__name__)


@dataclass
class EmbedderStack:
    """Holds the fast and quality embedder pair.

    Auto-detection probes for available ML libraries and falls back
    to the hash embedder if nothing else is installed.
    """

    fast: Embedder
    quality: Embedder | None

    @classmethod
    def auto_detect(cls) -> EmbedderStack:
        """Probe for available embedders and build the best stack.

        Priority: fastembed (quality) > model2vec (fast) > hash (fallback).

        Returns:
            EmbedderStack with best available embedders.
        """
        fast_embedder: Embedder | None = None
        quality_embedder: Embedder | None = None

        fast_embedder = _try_model2vec()
        quality_embedder = _try_fastembed()

        return _build_stack(cls, fast_embedder, quality_embedder)


def _try_model2vec() -> Embedder | None:
    """Attempt to load model2vec for the fast tier.

    Returns:
        Model2VecEmbedder instance or None if unavailable.
    """
    try:
        from brr.embed.model2vec_embedder import Model2VecEmbedder

        m2v = Model2VecEmbedder()
        m2v.dimension  # noqa: B018  # triggers load
    except Exception as exc:  # noqa: BLE001
        logger.debug("Model2Vec not available: %s", exc)
        return None
    else:
        logger.info("Model2Vec embedder available (fast tier)")
        return m2v


def _try_fastembed() -> Embedder | None:
    """Attempt to load fastembed for the quality tier.

    Returns:
        FastEmbedEmbedder instance or None if unavailable.
    """
    try:
        from brr.embed.fastembed_embedder import FastEmbedEmbedder

        fem = FastEmbedEmbedder()
        fem.dimension  # noqa: B018  # triggers load
    except Exception as exc:  # noqa: BLE001
        logger.debug("FastEmbed not available: %s", exc)
        return None
    else:
        logger.info("FastEmbed embedder available (quality tier)")
        return fem


def _build_stack(
    cls: type[EmbedderStack],
    fast_embedder: Embedder | None,
    quality_embedder: Embedder | None,
) -> EmbedderStack:
    """Build an EmbedderStack from discovered embedders with fallbacks.

    Returns:
        EmbedderStack with best available configuration.
    """
    if fast_embedder is None and quality_embedder is None:
        logger.info("No ML embedders available, using hash embedder fallback")
        return cls(fast=FnvHashEmbedder(), quality=None)

    if fast_embedder is None and quality_embedder is not None:
        logger.info("Only quality embedder available, using for both tiers")
        return cls(fast=quality_embedder, quality=None)

    if quality_embedder is None and fast_embedder is not None:
        logger.info("Only fast embedder available, no refinement possible")
        return cls(fast=fast_embedder, quality=None)

    # Both available — should not be None at this point
    return cls(fast=fast_embedder, quality=quality_embedder)  # type: ignore[arg-type]
