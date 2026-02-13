"""Basic search demo: index documents and search with progressive results."""

from brr import FnvHashEmbedder
from brr import TwoTierConfig
from brr import TwoTierSearcher
from brr import VectorIndex
from brr import canonicalize


DOCS = [
    ("doc-1", "Distributed consensus algorithms ensure fault tolerance in replicated systems"),
    ("doc-2", "The quick brown fox jumped gracefully over the lazy sleeping dog"),
    ("doc-3", "Machine learning models are trained using stochastic gradient descent"),
    ("doc-4", "Reciprocal rank fusion combines lexical and semantic search signals"),
    ("doc-5", "Python is a versatile programming language for data science"),
    ("doc-6", "Error handling and retry logic prevent cascading failures"),
    ("doc-7", "Vector similarity search finds nearest neighbors in embedding space"),
    ("doc-8", "Building inverted indices powers full-text search engines"),
    ("doc-9", "Neural network embeddings capture semantic meaning of text"),
    ("doc-10", "Two-tier search delivers fast initial results then refines rankings"),
]


def main() -> None:
    # Auto-detect best embedder (falls back to hash if no ML libs installed)
    embedder = FnvHashEmbedder(dim=128)

    # Build the vector index
    print("Indexing documents...")
    index = VectorIndex(dimension=embedder.dimension, embedder_id=embedder.model_id)
    for doc_id, text in DOCS:
        clean = canonicalize(text)
        vec = embedder.embed(clean)
        index.add(doc_id, vec)

    print(f"Indexed {index.count} documents ({embedder.model_id})")

    # Search with progressive results
    config = TwoTierConfig(fast_only=True)
    searcher = TwoTierSearcher(index=index, fast_embedder=embedder, config=config)

    query = "distributed consensus fault tolerant"
    print(f"\nSearching: '{query}'")

    for result in searcher.search(query, k=5):
        phase_name = result.phase.name
        print(f"\n--- {phase_name} ---")
        for hit in result.hits:
            src = []
            if hit.lexical_rank is not None:
                src.append(f"lex#{hit.lexical_rank}")
            if hit.semantic_rank is not None:
                src.append(f"sem#{hit.semantic_rank}")
            sources = ", ".join(src) if src else "n/a"
            both = " [BOTH]" if hit.in_both_sources else ""
            print(f"  {hit.doc_id}  rrf={hit.rrf_score:.4f}  ({sources}){both}")


if __name__ == "__main__":
    main()
