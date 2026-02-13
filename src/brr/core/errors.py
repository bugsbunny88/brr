"""SearchError hierarchy for brr."""

from __future__ import annotations


class SearchError(Exception):
    """Base exception for all brr errors."""


class EmbedderUnavailableError(SearchError):
    """Requested embedder model is not available."""

    def __init__(self, model: str, reason: str) -> None:
        self.model = model
        self.reason = reason
        super().__init__(f"Embedder unavailable: {model} — {reason}")


class EmbeddingFailedError(SearchError):
    """Embedding computation failed."""

    def __init__(self, model: str, cause: Exception | None = None) -> None:
        self.model = model
        self.cause = cause
        msg = (
            f"Embedding failed for model {model}: {cause}"
            if cause
            else f"Embedding failed for model {model}"
        )
        super().__init__(msg)


class IndexCorruptedError(SearchError):
    """Vector index file is corrupted or invalid."""

    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"Index corrupted at {path}: {detail}")


class DimensionMismatchError(SearchError):
    """Embedding dimension does not match the index."""

    def __init__(self, expected: int, found: int) -> None:
        self.expected = expected
        self.found = found
        super().__init__(f"Dimension mismatch: expected {expected}, found {found}")


class QueryParseError(SearchError):
    """Could not parse the search query."""

    def __init__(self, query: str, detail: str) -> None:
        self.query = query
        self.detail = detail
        super().__init__(f"Query parse error for '{query}': {detail}")


class SearchTimeoutError(SearchError):
    """Search exceeded the configured time budget."""

    def __init__(self, elapsed_ms: float, budget_ms: float) -> None:
        self.elapsed_ms = elapsed_ms
        self.budget_ms = budget_ms
        super().__init__(f"Search timeout: {elapsed_ms:.1f}ms exceeded {budget_ms:.1f}ms budget")
