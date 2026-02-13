"""brr — Two-tier hybrid search with progressive results."""

from brr.config import TwoTierConfig
from brr.core.canonicalize import canonicalize
from brr.core.canonicalize import canonicalize_query
from brr.core.errors import DimensionMismatchError
from brr.core.errors import EmbedderUnavailableError
from brr.core.errors import EmbeddingFailedError
from brr.core.errors import IndexCorruptedError
from brr.core.errors import QueryParseError
from brr.core.errors import SearchError
from brr.core.errors import SearchTimeoutError
from brr.core.protocols import Embedder
from brr.core.protocols import LexicalBackend
from brr.core.protocols import ModelCategory
from brr.core.query_class import QueryClass
from brr.core.query_class import adaptive_budget
from brr.core.query_class import classify_query
from brr.core.types import FusedHit
from brr.core.types import SearchPhase
from brr.core.types import SearchResult
from brr.core.types import VectorHit
from brr.embed.hash_embedder import FnvHashEmbedder
from brr.embed.stack import EmbedderStack
from brr.fusion.rrf import reciprocal_rank_fusion
from brr.fusion.two_tier import TwoTierSearcher
from brr.index.vector_index import VectorIndex


__all__ = [
    "DimensionMismatchError",
    "Embedder",
    "EmbedderStack",
    "EmbedderUnavailableError",
    "EmbeddingFailedError",
    "FnvHashEmbedder",
    "FusedHit",
    "IndexCorruptedError",
    "LexicalBackend",
    "ModelCategory",
    "QueryClass",
    "QueryParseError",
    "SearchError",
    "SearchPhase",
    "SearchResult",
    "SearchTimeoutError",
    "TwoTierConfig",
    "TwoTierSearcher",
    "VectorHit",
    "VectorIndex",
    "adaptive_budget",
    "canonicalize",
    "canonicalize_query",
    "classify_query",
    "reciprocal_rank_fusion",
]
