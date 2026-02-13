# brr

**Search goes vroom.**

brr is a two-tier hybrid search library for Python. It fuses lexical (BM25) and semantic (vector) search via Reciprocal Rank Fusion, then yields fast results immediately while a quality model refines rankings in the background. Your code gets results before the user finishes reading the loading spinner.

```
pip install brr            # numpy only, hash embedder, already fast
pip install brr[full]      # model2vec + fastembed + bm25s, absurdly fast
```

---

## The Idea

Most search libraries make you choose: fast or good. brr says no. It runs a fast static embedder first (sub-millisecond), hands you results, then re-scores with a quality transformer model and hands you *better* results. You display the first batch instantly. The second batch arrives before the user notices.

That's the vroom.

```python
from brr import TwoTierSearcher, EmbedderStack, VectorIndex

stack = EmbedderStack.auto_detect()
index = VectorIndex(dimension=stack.fast.dimension, embedder_id=stack.fast.model_id)

# Index some documents
for doc_id, text in documents:
    index.add(doc_id, stack.fast.embed(text))

# Search -- results come in phases
searcher = TwoTierSearcher(index, stack.fast, quality_embedder=stack.quality)
for result in searcher.search("distributed consensus algorithm", k=10):
    print(f"[{result.phase.name}]")
    for hit in result.hits:
        print(f"  {hit.doc_id}  (rrf: {hit.rrf_score:.4f})")
```

Phase 1 lands fast. Phase 2 lands better. Your UI never blocks.

---

## CLI

brr ships a `brr` command for quick indexing and search from the terminal.

```bash
# Index documents (one per line from stdin)
cat docs.txt | brr index my_index

# Search
brr search my_index "how does authentication work"

# Index info
brr info my_index
```

---

## What It Does

| Feature | How It Works |
|---------|-------------|
| **Progressive search** | Generator yields INITIAL results, then REFINED results. Display the first, swap in the second |
| **Hybrid fusion** | BM25 lexical + vector semantic, combined with RRF (K=60) |
| **Two-tier embedding** | Fast tier (model2vec/potion, sub-ms) + quality tier (fastembed/bge-small) |
| **Graceful fallback** | No ML models? Hash embedder works everywhere with zero dependencies |
| **Query classification** | Identifiers bias lexical, natural language biases semantic |
| **f16 quantization** | Half the memory, negligible quality loss |
| **Zero-dep core** | `pip install brr` works with just numpy -- ML models are optional extras |

---

## Install

```bash
# Core only (numpy + hash embedder)
pip install brr

# Fast semantic search (model2vec/potion)
pip install brr[model2vec]

# Quality semantic search (fastembed/bge-small)
pip install brr[fastembed]

# Lexical BM25 search
pip install brr[lexical]

# Everything -- full vroom
pip install brr[full]
```

---

## Zero-Dep Mode

Don't want to download models? The hash embedder produces deterministic vectors from FNV-1a hashing. No semantic understanding, but it works everywhere and never needs a network connection.

```python
from brr import FnvHashEmbedder, VectorIndex

embedder = FnvHashEmbedder(dimension=384)
index = VectorIndex(dimension=384, embedder_id=embedder.model_id)

index.add("doc-1", embedder.embed("hello world"))
index.add("doc-2", embedder.embed("goodbye world"))

hits = index.search(embedder.embed("hello"), k=2)
for hit in hits:
    print(f"{hit.doc_id}: {hit.score:.4f}")
```

---

## Architecture

```
Query
  |
  v
Canonicalize (NFC + markdown strip + code collapse)
  |
  v
Classify (Empty | Identifier | Short | NaturalLanguage)
  |
  +---> Fast Embed (model2vec, <1ms) --> Vector Search --+
  |                                                      |
  +---> BM25 Lexical Search ----------------------------+
                                                         |
                                                         v
                                                  RRF Fusion (K=60)
                                                         |
                                                    yield INITIAL
                                                         |
                                                         v
                                                  Quality Embed (fastembed)
                                                         |
                                                         v
                                                  Two-Tier Blend (0.7/0.3)
                                                         |
                                                    yield REFINED
```

---

## Configuration

All knobs are available via `TwoTierConfig` or environment variables:

```python
from brr import TwoTierConfig

config = TwoTierConfig(
    quality_weight=0.7,       # blend: 70% quality, 30% fast
    rrf_k=60.0,               # RRF constant
    candidate_multiplier=3,   # fetch 3x candidates from each source
    fast_only=False,          # skip quality refinement entirely
)
```

| Environment Variable | Default | What It Does |
|---------------------|---------|-------------|
| `BRR_QUALITY_WEIGHT` | `0.7` | Blend factor for quality tier |
| `BRR_RRF_K` | `60.0` | RRF constant |
| `BRR_FAST_ONLY` | `false` | Skip quality refinement |
| `BRR_QUALITY_TIMEOUT_MS` | `500` | Max wait for quality model |

---

## Save / Load

Indices persist as `.npz` (vectors, f16 quantized) + `.json` (metadata sidecar).

```python
# Save
index.save("my_index")

# Load
index = VectorIndex.load("my_index")
```

---

## Project Structure

```
src/brr/
  core/           # types, errors, protocols, canonicalization, query classification
  embed/          # hash, model2vec, fastembed embedders + auto-detection stack
  index/          # vector index, numpy brute-force top-k search
  lexical/        # bm25s wrapper
  fusion/         # RRF, blending, normalization, two-tier orchestrator
  cli.py          # brr command
  config.py       # TwoTierConfig + env vars
```

---

## License

MIT
