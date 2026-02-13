"""Zero-dependency demo: hash embedder only, no ML models needed."""

from brr.embed.hash_embedder import FnvHashEmbedder
from brr.index.vector_index import VectorIndex


def main() -> None:
    # FNV-1a hash embedder: deterministic, zero ML dependencies
    embedder = FnvHashEmbedder(dim=384)
    print(f"Embedder: {embedder.model_id}")
    print(f"Dimension: {embedder.dimension}")
    print(f"Semantic: {embedder.is_semantic}")
    print(f"Category: {embedder.category.name}")

    # Embed some text
    vec = embedder.embed("hello world")
    print(f"\nEmbedding for 'hello world': [{vec[0]:.4f}, {vec[1]:.4f}, ..., {vec[-1]:.4f}]")

    # Build a tiny index and search
    index = VectorIndex(dimension=384, embedder_id=embedder.model_id)
    texts = [
        ("greeting", "hello world"),
        ("farewell", "goodbye world"),
        ("code", "print hello world in python"),
    ]
    for doc_id, text in texts:
        index.add(doc_id, embedder.embed(text))

    query_vec = embedder.embed("hello")
    results = index.search(query_vec, k=3)

    print(f"\nSearch for 'hello' (top {len(results)}):")
    for hit in results:
        print(f"  {hit.doc_id}: {hit.score:.4f}")


if __name__ == "__main__":
    main()
