"""Core data types for brr."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import auto


@dataclass(frozen=True, slots=True)
class VectorHit:
    """Raw vector similarity result."""

    index: int
    score: float
    doc_id: str


@dataclass(frozen=True, slots=True)
class FusedHit:
    """Hybrid search result combining lexical and semantic signals."""

    doc_id: str
    rrf_score: float
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None
    in_both_sources: bool = False


class SearchPhase(Enum):
    """Identifies which phase of progressive search produced results."""

    INITIAL = auto()
    REFINED = auto()
    REFINEMENT_FAILED = auto()


@dataclass(slots=True)
class SearchResult:
    """A progressive search result yielded by the two-tier searcher."""

    phase: SearchPhase
    hits: list[FusedHit] = field(default_factory=list)
