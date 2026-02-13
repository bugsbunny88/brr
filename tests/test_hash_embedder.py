"""Tests for the FNV-1a hash embedder."""

import math

from brr.core.protocols import Embedder
from brr.core.protocols import ModelCategory
from brr.embed.hash_embedder import FnvHashEmbedder


def test_dimension():
    emb = FnvHashEmbedder(dim=128)
    assert emb.dimension == 128


def test_output_length():
    emb = FnvHashEmbedder(dim=64)
    vec = emb.embed("hello world")
    assert len(vec) == 64


def test_deterministic():
    emb = FnvHashEmbedder(dim=64)
    v1 = emb.embed("test string")
    v2 = emb.embed("test string")
    assert v1 == v2


def test_different_inputs_differ():
    emb = FnvHashEmbedder(dim=64)
    v1 = emb.embed("hello")
    v2 = emb.embed("world")
    assert v1 != v2


def test_l2_normalized():
    emb = FnvHashEmbedder(dim=128)
    vec = emb.embed("some text for embedding")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6


def test_batch():
    emb = FnvHashEmbedder(dim=64)
    results = emb.embed_batch(["hello", "world"])
    assert len(results) == 2
    assert results[0] == emb.embed("hello")
    assert results[1] == emb.embed("world")


def test_protocol_compliance():
    emb = FnvHashEmbedder()
    assert isinstance(emb, Embedder)
    assert emb.category == ModelCategory.HASH
    assert emb.is_semantic is False
    assert "fnv1a" in emb.model_id


def test_short_text():
    emb = FnvHashEmbedder(dim=64)
    vec = emb.embed("ab")  # shorter than trigram
    assert len(vec) == 64
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6


def test_empty_string():
    emb = FnvHashEmbedder(dim=64)
    vec = emb.embed("")
    assert len(vec) == 64
    # Empty string is shorter than ngram size, hashes the whole (empty) bytes
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6
