# brr: State of Thinking

*Snapshot after completing the Python implementation. What exists, what doesn't, and what's next.*

## What brr Solves Today

The core hybrid retrieval pipeline is done. This covers Stage 1 from the architecture doc:

- **BM25 via `bm25s`** for lexical retrieval (500x faster than rank_bm25, scipy sparse, mmap save/load)
- **Dense embeddings via EmbedderStack** with auto-detection and graceful fallback: fastembed (quality tier), model2vec/potion (fast tier), FNV-1a hash (zero-dep fallback)
- **RRF fusion** with 4-level tie-breaking (score, in-both-sources, lexical score, doc ID)
- **Two-tier progressive generator** that yields fast results immediately, then refines with the quality embedder asynchronously. Most retrieval libraries skip this

Additional pieces that go beyond what the architecture doc called for:

- **Query classification with adaptive budgets** that biases toward lexical for identifiers (`AAPL`, file paths) and toward semantic for natural language queries. Right behavior for a financial research corpus or mixed-modality input.
- **Canonicalization pipeline**: NFC normalization, markdown stripping, code block collapsing, import streak filtering. Accounts for a real chunk of retrieval quality on markdown-heavy input.
- **Persistence** -- `VectorIndex.save/load` with f16 quantization (.npz + JSON sidecar), `BM25SBackend.save/load`. The index isn't ephemeral.

## The Gap: What "Two Tiers" Actually Means

The current "two tiers" are two *embedding models* (fast like model2vec, quality like fastembed/bge-small). Both tiers do single-vector cosine/dot-product similarity. The quality tier just uses a better model. The blend weighting (0.7 quality + 0.3 fast) is a good way to combine them, but it's not the same as changing the *scoring function itself*.

The architecture doc's Stage 2 -- where a cross-encoder or late-interaction model re-scores candidates using a fundamentally different mechanism -- is the piece that isn't here yet.

The DeepMind paper's formal result is relevant here: single-vector models can't scale to combinatorial query complexity. As the corpus grows, the number of meaningful distinctions a single vector needs to encode grows combinatorially while embedding capacity grows polynomially. A query like "blue trail-running shoes, size 10, under $100" compresses to a single point, and results become a mix of partial matches. Distance stops being meaningful when too many conceptually different things occupy the same neighborhood.

## Three Paths Forward

### Path 1: Cross-Encoder as a Third Tier

The `TwoTierSearcher` generator pattern already supports this structurally. Add a Stage 3 yield that takes the top-k from blended results, runs them through a cross-encoder, and re-sorts. The consumer just gets a third `SearchResult` with `phase=RERANKED`.

The `Embedder` protocol doesn't fit cross-encoders (they take a query-document pair, not a single text). Needs a new protocol:

```python
@runtime_checkable
class Reranker(Protocol):
    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]: ...
```

**Models to consider:**

| Model | Size | Strength |
|---|---|---|
| `BAAI/bge-reranker-v2-m3` | 568M | Multilingual + code |
| `mixedbread-ai/mxbai-rerank-large-v1` | 435M | SOTA quality |
| `Xenova/ms-marco-MiniLM-L-6-v2` | 22M | Fast, lightweight |
| `jinaai/jina-reranker-v2-base-multilingual` | 278M | Code-aware |

Cross-encoders are the nuclear option: full attention between all query and document tokens. Highest accuracy, but nothing can be pre-computed. Practical only for reranking 10-100 candidates from the fast pipeline.

### Path 2: ColBERT / Late Interaction

ColBERT produces a vector *per token* (typically 128-dim each). At query time, it computes MaxSim -- for each query token, find its maximum similarity to any document token, then sum. This is different from cosine similarity between two single vectors. Over 100x more efficient than a cross-encoder at comparable scale.

This also doesn't fit the current `Embedder` protocol because it produces a matrix of token-level vectors, not a single vector. And `VectorIndex` stores one vector per doc.

Two integration options:

1. **Separate index type** that stores per-token matrices. Significant storage cost -- 768GB for 10M passages vs 61GB for single-vector. But enables ColBERT for retrieval, not just reranking.
2. **Reranker-only** (more practical). Embed the query into token vectors at query time, load precomputed doc token matrices for just the candidates, compute MaxSim. Avoids changing the index format.

**Key models:** ColBERTv2 (Stanford original), Jina-ColBERT-v2 (multilingual, 8192 token context), answerai-colbert-small (33M params, punches above its weight). ColBERT variants on ModernBERT are emerging.

**Database support:** Vespa has the deepest native ColBERT integration (32x compression, multi-phase ranking). Qdrant supports multi-vector natively. Weaviate added multi-vector in v1.29. LanceDB supports ColPali for visual late interaction.

### Path 3: SPLADE / Learned Sparse

Instead of multiple dense vectors, SPLADE produces high-dimensional sparse vectors where each dimension maps to a vocabulary term. Uses BERT's MLM head for term expansion ("automobiles" activates "cars"). Solves vocabulary mismatch while staying sparse enough for inverted indexes.

Storage efficient -- documents expand to ~100 tokens on average, approximately the same as a normal text index. Elasticsearch's ELSER model is built on this architecture. Pinecone and Vespa support it natively.

Complementary to ColBERT rather than competing. Vespa can combine SPLADE + ColBERT in the same ranking pipeline.

## Impact Estimates

From the architecture research:

| Component | Retrieval Quality Impact |
|---|---|
| Embedding model choice | ~10-15% |
| Reranker (cross-encoder) | ~20-30% |
| Chunking strategy | ~15-20% |

The reranker is the single biggest lever. brr currently captures the embedding and fusion components. Adding a reranker tier would address the largest remaining quality gap.

## What's Not brr's Problem

- **Incremental indexing from filesystem.** `add`/`add_batch` exist but there's no watcher or diff logic. That's glue between the filesystem and the index -- belongs in the application layer.
- **FTS5 on the provenance store.** That's a migration on the SQLite schema in the main codebase, not something this library owns.
- **Chunking strategy.** brr indexes whatever you give it. How you split documents is upstream.

## The Honest Assessment

brr handles Stage 1 (hybrid retrieval with fusion) well. The progressive generator pattern means Stage 2 (reranking with a fundamentally different scoring model) can be added as another yield phase without restructuring anything.

The question is whether to extend with reranker support now or ship what exists and add that tier when the corpus outgrows single-vector retrieval.

The common production pattern is layered: BM25 + dense ANN + optional SPLADE for retrieval, ColBERT MaxSim for reranking, cross-encoder for fine reranking of the final 10-50. In the DeepMind framing, the single-vector embedding is a fast L1 cache for semantics, and the system needs a higher-rank L2 component for precise combinatorial logic. brr is the L1. The L2 slot is open.
