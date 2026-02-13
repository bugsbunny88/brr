"""BM25 lexical search backend using bm25s."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence


class BM25SBackend:
    """Wrapper around bm25s implementing the LexicalBackend protocol.

    bm25s is imported lazily so that `import brr` works without it.
    """

    def __init__(self) -> None:
        self._index: object | None = None
        self._doc_ids: list[str] = []

    def index_documents(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Tokenize and index documents for BM25 retrieval."""
        import bm25s  # lazy import

        self._doc_ids = list(doc_ids)
        corpus_tokens = bm25s.tokenize(list(texts), stopwords="en")
        self._index = bm25s.BM25()
        self._index.index(corpus_tokens)  # type: ignore[union-attr]

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Search the BM25 index.

        Returns:
            List of (doc_id, score) pairs ranked by relevance.
        """
        if self._index is None or not self._doc_ids:
            return []

        import bm25s  # lazy import

        query_tokens = bm25s.tokenize([query], stopwords="en")
        bound_k = min(limit, len(self._doc_ids))
        indices, scores = self._index.retrieve(  # type: ignore[union-attr]
            query_tokens,
            k=bound_k,
        )

        ranked_pairs: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            idx_int = int(idx)
            if 0 <= idx_int < len(self._doc_ids):
                ranked_pairs.append((self._doc_ids[idx_int], float(score)))
        return ranked_pairs

    def save(self, path: str) -> None:
        """Save the BM25 index to disk."""
        if self._index is not None:
            self._index.save(path)  # type: ignore[union-attr]

    @classmethod
    def load(cls, path: str, doc_ids: list[str]) -> BM25SBackend:
        """Load a BM25 index from disk.

        Returns:
            BM25SBackend with restored index.
        """
        import bm25s

        backend = cls()
        backend._index = bm25s.BM25.load(path)
        backend._doc_ids = doc_ids
        return backend
