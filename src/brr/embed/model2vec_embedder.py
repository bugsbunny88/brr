"""Model2Vec (potion-128M) embedder — fast tier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brr.core.errors import EmbedderUnavailableError
from brr.core.errors import EmbeddingFailedError
from brr.core.protocols import ModelCategory


if TYPE_CHECKING:
    from collections.abc import Sequence


class Model2VecEmbedder:
    """Wrapper around model2vec for the potion-multilingual-128M fast tier.

    model2vec is imported lazily so that `import brr` works without it.
    """

    def __init__(self, model_name: str = "minishlab/potion-base-8M") -> None:
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
        """Return True — model2vec produces semantic embeddings."""
        return True

    @property
    def category(self) -> ModelCategory:
        """Return FAST model category."""
        return ModelCategory.FAST

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector.

        Returns:
            Float vector of size `dimension`.

        Raises:
            EmbeddingFailedError: If encoding fails.
        """
        self._ensure_loaded()
        try:
            vectors = self._model.encode([text])  # type: ignore[union-attr]
        except Exception as exc:
            raise EmbeddingFailedError(self._model_name, exc) from exc
        else:
            return vectors[0].tolist()

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts into vectors.

        Returns:
            List of float vectors.

        Raises:
            EmbeddingFailedError: If encoding fails.
        """
        self._ensure_loaded()
        try:
            vectors = self._model.encode(list(texts))  # type: ignore[union-attr]
        except Exception as exc:
            raise EmbeddingFailedError(self._model_name, exc) from exc
        else:
            return [row.tolist() for row in vectors]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from model2vec import StaticModel  # type: ignore[import-untyped]

            self._model = StaticModel.from_pretrained(self._model_name)
            dummy = self._model.encode(["test"])  # type: ignore[union-attr]
            self._dim = dummy.shape[1]
        except ImportError as exc:
            raise EmbedderUnavailableError(
                self._model_name, "model2vec not installed (pip install brr[model2vec])"
            ) from exc
        except Exception as exc:
            raise EmbedderUnavailableError(self._model_name, str(exc)) from exc
