"""FNV-1a hash embedder — zero ML dependencies, always available."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from brr.core.protocols import ModelCategory


if TYPE_CHECKING:
    from collections.abc import Sequence

# FNV-1a constants for 64-bit
_FNV_OFFSET: Final = 0xCBF29CE484222325
_FNV_PRIME: Final = 0x100000001B3
_BIT_WIDTH: Final = 64
_MASK64: Final = (1 << _BIT_WIDTH) - 1
_DEFAULT_DIM: Final = 384
_DEFAULT_NGRAM: Final = 3
_SIGN_BIT: Final = 32


def _fnv1a_hash(raw_bytes: bytes) -> int:
    """Compute FNV-1a 64-bit hash of raw bytes.

    Returns:
        64-bit hash integer.
    """
    hash_val = _FNV_OFFSET
    for single_byte in raw_bytes:
        hash_val ^= single_byte
        hash_val = (hash_val * _FNV_PRIME) & _MASK64
    return hash_val


def _scatter_ngram(vec: list[float], gram: bytes) -> None:
    """Hash a single n-gram and scatter into the vector."""
    hash_val = _fnv1a_hash(gram)
    bucket = hash_val % len(vec)
    sign = 1.0 if (hash_val >> _SIGN_BIT) & 1 else -1.0
    vec[bucket] += sign


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector in-place, returning it.

    Returns:
        Normalized vector (same list if norm > 0).
    """
    norm = sum(coord * coord for coord in vec) ** 0.5
    if norm > 0:
        return [coord / norm for coord in vec]
    return vec


class FnvHashEmbedder:
    """Deterministic hash-based embedder using FNV-1a.

    Produces a fixed-dimension embedding by hashing overlapping character
    n-grams (trigrams by default) into buckets. Not semantic, but useful
    as a zero-dependency fallback and for exact/fuzzy string matching.
    """

    def __init__(self, dim: int = _DEFAULT_DIM, ngram_size: int = _DEFAULT_NGRAM) -> None:
        self._dim = dim
        self._ngram_size = ngram_size

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dim

    @property
    def model_id(self) -> str:
        """Return the model identifier string."""
        return f"fnv1a-{self._dim}d"

    @property
    def is_semantic(self) -> bool:
        """Return False — hash embedder is not semantic."""
        return False

    @property
    def category(self) -> ModelCategory:
        """Return HASH model category."""
        return ModelCategory.HASH

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector.

        Returns:
            L2-normalized float vector of size `dimension`.
        """
        vec = [0.0] * self._dim  # noqa: WPS435,WPS358
        text_bytes = text.encode("utf-8")
        ngram = self._ngram_size
        if len(text_bytes) < ngram:
            _scatter_ngram(vec, text_bytes)
        else:
            for start in range(len(text_bytes) - ngram + 1):
                _scatter_ngram(vec, text_bytes[start : start + ngram])
        return _l2_normalize(vec)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts into vectors.

        Returns:
            List of L2-normalized float vectors.
        """
        return [self.embed(text) for text in texts]
