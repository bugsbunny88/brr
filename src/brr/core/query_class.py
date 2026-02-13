"""Query classification and adaptive budget allocation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Final


class QueryClass(Enum):
    """Classifies a query for adaptive retrieval strategy."""

    EMPTY = auto()
    IDENTIFIER = auto()
    SHORT_KEYWORD = auto()
    NATURAL_LANGUAGE = auto()


# Patterns that indicate identifiers
_IDENTIFIER_RE: Final = re.compile(r"^[\w./-]+$")
_PATH_RE: Final = re.compile(r"[/\\]")
_TICKET_ID_RE: Final = re.compile(r"^[a-zA-Z]+-\d+$")

_SHORT_KEYWORD_MAX_WORDS: Final = 3


def classify_query(query: str) -> QueryClass:
    """Classify a query for adaptive retrieval budgets.

    Returns:
        The QueryClass describing the query type.
    """
    stripped = query.strip()
    if not stripped:
        return QueryClass.EMPTY

    words = stripped.split()
    if len(words) == 1:
        token = words[0]
        if _TICKET_ID_RE.match(token) or _PATH_RE.search(token) or _IDENTIFIER_RE.match(token):
            return QueryClass.IDENTIFIER

    if len(words) <= _SHORT_KEYWORD_MAX_WORDS:
        return QueryClass.SHORT_KEYWORD

    return QueryClass.NATURAL_LANGUAGE


@dataclass(frozen=True, slots=True)
class CandidateBudget:
    """Per-source candidate multipliers."""

    lexical_multiplier: int
    semantic_multiplier: int


def adaptive_budget(
    query_class: QueryClass,
    base_multiplier: int = 3,
) -> CandidateBudget:
    """Return per-source candidate multipliers based on query classification.

    Returns:
        CandidateBudget with lexical and semantic multipliers.
    """
    match query_class:
        case QueryClass.EMPTY:
            return CandidateBudget(lexical_multiplier=0, semantic_multiplier=0)
        case QueryClass.IDENTIFIER:
            return CandidateBudget(
                lexical_multiplier=base_multiplier * 2,
                semantic_multiplier=max(1, base_multiplier // 2),
            )
        case QueryClass.SHORT_KEYWORD:
            return CandidateBudget(
                lexical_multiplier=base_multiplier,
                semantic_multiplier=base_multiplier,
            )
        case QueryClass.NATURAL_LANGUAGE:
            return CandidateBudget(
                lexical_multiplier=max(1, base_multiplier // 2),
                semantic_multiplier=base_multiplier * 2,
            )
