"""Shared test fixtures."""

from __future__ import annotations

import pytest

from brr.embed.hash_embedder import FnvHashEmbedder
from brr.index.vector_index import VectorIndex


# Sample corpus for integration tests
SAMPLE_DOCS = [
    ("doc-1", "distributed consensus algorithm for fault tolerant systems"),
    ("doc-2", "quick brown fox jumps over the lazy dog"),
    ("doc-3", "machine learning model training with gradient descent"),
    ("doc-4", "reciprocal rank fusion combines lexical and semantic search"),
    ("doc-5", "python programming language for data science applications"),
    ("doc-6", "error handling and retry logic in distributed systems"),
    ("doc-7", "the cat sat on the mat in the sunny garden"),
    ("doc-8", "vector similarity search using cosine distance metrics"),
    ("doc-9", "building search engines with inverted indices and bm25"),
    ("doc-10", "neural network embeddings for natural language processing"),
]


@pytest.fixture
def hash_embedder() -> FnvHashEmbedder:
    return FnvHashEmbedder(dim=64)


@pytest.fixture
def sample_index(hash_embedder: FnvHashEmbedder) -> VectorIndex:
    """Build a small VectorIndex from SAMPLE_DOCS using the hash embedder."""
    idx = VectorIndex(dimension=hash_embedder.dimension, embedder_id=hash_embedder.model_id)
    for doc_id, text in SAMPLE_DOCS:
        embedding = hash_embedder.embed(text)
        idx.add(doc_id, embedding)
    return idx
