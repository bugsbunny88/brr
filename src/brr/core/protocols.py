"""Protocol definitions for embedders and lexical backends."""

from __future__ import annotations

from enum import Enum
from enum import auto
from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Sequence


class ModelCategory(Enum):
    """Categorizes an embedder by quality tier."""

    HASH = auto()
    FAST = auto()
    QUALITY = auto()


@runtime_checkable
class Embedder(Protocol):
    """Structural interface for text embedding models."""

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts into vectors."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def model_id(self) -> str:
        """Return the model identifier string."""
        ...

    @property
    def is_semantic(self) -> bool:
        """Return True if this embedder produces semantic embeddings."""
        ...

    @property
    def category(self) -> ModelCategory:
        """Return the model quality tier category."""
        ...


@runtime_checkable
class LexicalBackend(Protocol):
    """Structural interface for lexical (BM25) search backends."""

    def index_documents(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Index documents for lexical retrieval."""
        ...

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Return list of (doc_id, score) tuples ranked by relevance."""
        ...
