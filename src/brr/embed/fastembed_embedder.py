"""FastEmbed (MiniLM-L6-v2) embedder — quality tier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brr.core.errors import EmbedderUnavailableError
from brr.core.errors import EmbeddingFailedError
from brr.core.protocols import ModelCategory


if TYPE_CHECKING:
    from collections.abc import Sequence


class FastEmbedEmbedder:
    """Wrapper around fastembed for the all-MiniLM-L6-v2 quality tier.

    fastembed is imported lazily so that `import brr` works without it.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, loading the model if needed.

        Raises:
            RuntimeError: If dimension is not set after loading.
        """
        self._ensure_loaded()
        if self._dim is None:  # pragma: no cover
            msg = "dimension not set after loading"
            raise RuntimeError(msg)
        return self._dim

    @property
    def model_id(self) -> str:
        """Return the model identifier string."""
        return self._model_name

    @property
    def is_semantic(self) -> bool:
        """Return True — fastembed produces semantic embeddings."""
        return True

    @property
    def category(self) -> ModelCategory:
        """Return QUALITY model category."""
        return ModelCategory.QUALITY

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector.

        Returns:
            Float vector of size `dimension`.

        Raises:
            EmbeddingFailedError: If encoding fails.
        """
        self._ensure_loaded()
        try:
            vectors = list(self._model.embed([text]))  # type: ignore[union-attr]
        except Exception as exc:
            raise EmbeddingFailedError(self._model_name, exc) from exc
        else:
            return list(vectors[0])

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts into vectors.

        Returns:
            List of float vectors.

        Raises:
            EmbeddingFailedError: If encoding fails.
        """
        self._ensure_loaded()
        try:
            vectors = list(self._model.embed(list(texts)))  # type: ignore[union-attr]
        except Exception as exc:
            raise EmbeddingFailedError(self._model_name, exc) from exc
        else:
            return [list(row) for row in vectors]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding  # type: ignore[import-untyped]

            self._model = TextEmbedding(model_name=self._model_name)
            dummy = list(self._model.embed(["test"]))  # type: ignore[union-attr]
            self._dim = len(dummy[0])
        except ImportError as exc:
            raise EmbedderUnavailableError(
                self._model_name, "fastembed not installed (pip install brr[fastembed])"
            ) from exc
        except Exception as exc:
            raise EmbedderUnavailableError(self._model_name, str(exc)) from exc
