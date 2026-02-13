"""Vector index with f16 quantization, save/load as .npz+JSON sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Final

import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray

from brr.core.errors import DimensionMismatchError
from brr.core.errors import IndexCorruptedError
from brr.core.types import VectorHit
from brr.index.search import top_k_dot_product


_META_VERSION: Final = 1
_DEFAULT_TOP_K: Final = 10


class VectorIndex:
    """In-memory vector index with f16 quantization and brute-force search."""

    def __init__(
        self,
        dimension: int,
        embedder_id: str = "",
        *,
        use_f16: bool = True,
    ) -> None:
        self._dimension = dimension
        self._embedder_id = embedder_id
        self._use_f16 = use_f16
        self._doc_ids: list[str] = []
        self._vectors: NDArray | None = None

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for this index."""
        return self._dimension

    @property
    def count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._doc_ids)

    @property
    def embedder_id(self) -> str:
        """Return the embedder identifier used to build this index."""
        return self._embedder_id

    @property
    def doc_ids(self) -> list[str]:
        """Return a copy of all document IDs in index order."""
        return list(self._doc_ids)

    def add(self, doc_id: str, embedding: list[float] | NDArray) -> None:
        """Add a single document embedding to the index.

        Raises:
            DimensionMismatchError: If embedding dimension doesn't match index.
        """
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.shape != (self._dimension,):
            actual_dim = vec.shape[0] if vec.ndim == 1 else -1
            raise DimensionMismatchError(self._dimension, actual_dim)

        self._doc_ids.append(doc_id)
        row = vec.reshape(1, -1)
        if self._vectors is None:
            self._vectors = row
        else:
            self._vectors = np.vstack([self._vectors, row])

    def add_batch(self, doc_ids: list[str], embeddings: list[list[float]] | NDArray) -> None:
        """Add multiple document embeddings at once.

        Raises:
            DimensionMismatchError: If embedding dimension doesn't match index.
            ValueError: If doc_ids length doesn't match embeddings rows.
        """
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._dimension:
            raise DimensionMismatchError(
                self._dimension, matrix.shape[1] if matrix.ndim == 2 else -1
            )
        if matrix.shape[0] != len(doc_ids):
            raise ValueError(
                f"doc_ids length ({len(doc_ids)}) != embeddings rows ({matrix.shape[0]})"
            )

        self._doc_ids.extend(doc_ids)
        if self._vectors is None:
            self._vectors = matrix
        else:
            self._vectors = np.vstack([self._vectors, matrix])

    def search(self, query: list[float] | NDArray, k: int = _DEFAULT_TOP_K) -> list[VectorHit]:
        """Brute-force top-k search by dot product similarity.

        Returns:
            List of VectorHit sorted by descending similarity.

        Raises:
            DimensionMismatchError: If query dimension doesn't match index.
        """
        if self._vectors is None or self.count == 0:
            return []

        query_arr = np.asarray(query, dtype=np.float32)
        if query_arr.shape != (self._dimension,):
            raise DimensionMismatchError(
                self._dimension, query_arr.shape[0] if query_arr.ndim == 1 else -1
            )

        top_hits = top_k_dot_product(query_arr, self._vectors, k)
        return [
            VectorHit(
                index=idx,
                score=score,
                doc_id=self._doc_ids[idx],
            )
            for idx, score in top_hits
        ]

    def save(self, path: str | Path) -> None:
        """Save index to .npz (vectors) + .json sidecar (metadata)."""
        resolved = Path(path)
        npz_path = resolved.with_suffix(".npz")
        json_path = resolved.with_suffix(".json")

        if self._vectors is None:
            np.savez_compressed(npz_path, vectors=np.empty((0, self._dimension)))
        else:
            save_vecs = self._vectors.astype(np.float16) if self._use_f16 else self._vectors
            np.savez_compressed(npz_path, vectors=save_vecs)

        meta: dict = {
            "version": _META_VERSION,
            "embedder_id": self._embedder_id,
            "dimension": self._dimension,
            "quantization": "f16" if self._use_f16 else "f32",
            "record_count": len(self._doc_ids),
            "doc_ids": self._doc_ids,
        }
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> VectorIndex:
        """Load index from .npz + .json sidecar.

        Returns:
            Loaded VectorIndex ready for search.

        Raises:
            IndexCorruptedError: If files are missing or data is invalid.
        """
        resolved = Path(path)
        meta, vectors = _load_index_files(
            resolved.with_suffix(".json"),
            resolved.with_suffix(".npz"),
        )
        dimension = meta["dimension"]

        if vectors.ndim == 2 and vectors.shape[1] != dimension:
            raise IndexCorruptedError(
                str(resolved),
                f"dimension mismatch: header={dimension}, data={vectors.shape[1]}",
            )

        idx = cls(
            dimension=dimension,
            embedder_id=meta.get("embedder_id", ""),
            use_f16=(meta.get("quantization", "f16") == "f16"),
        )
        idx._doc_ids = list(meta["doc_ids"])
        idx._vectors = vectors if vectors.shape[0] > 0 else None
        return idx


def _load_index_files(
    json_path: Path,
    npz_path: Path,
) -> tuple[dict, NDArray]:
    """Read and validate index files from disk.

    Returns:
        Tuple of (metadata dict, vectors array).

    Raises:
        IndexCorruptedError: If files are missing or corrupted.
    """
    if not json_path.exists():
        raise IndexCorruptedError(str(json_path), "metadata sidecar not found")
    if not npz_path.exists():
        raise IndexCorruptedError(str(npz_path), "vector data not found")

    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise IndexCorruptedError(str(json_path), f"invalid metadata: {exc}") from exc

    npz_data = np.load(npz_path)
    return meta, npz_data["vectors"].astype(np.float32)
