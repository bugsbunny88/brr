"""brr CLI -- search goes vroom."""

from __future__ import annotations

import argparse
import sys
from typing import Final

from brr.embed.hash_embedder import FnvHashEmbedder
from brr.index.vector_index import VectorIndex


_DEFAULT_K: Final = 10


def main() -> None:
    """Entry point for the brr CLI."""
    parser = argparse.ArgumentParser(
        prog="brr",
        description="brr -- hybrid search that goes vroom",
    )
    sub = parser.add_subparsers(dest="command")

    _add_index_cmd(sub)
    _add_search_cmd(sub)
    _add_info_cmd(sub)

    args = parser.parse_args()

    if args.command == "index":
        _run_index(args)
    elif args.command == "search":
        _run_search(args)
    elif args.command == "info":
        _run_info(args)
    else:
        parser.print_help()


def _add_index_cmd(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the index subcommand."""
    cmd = sub.add_parser("index", help="Index documents from stdin (one per line)")
    cmd.add_argument("path", help="Path to save the index")


def _add_search_cmd(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the search subcommand."""
    cmd = sub.add_parser("search", help="Search an index")
    cmd.add_argument("path", help="Path to the index")
    cmd.add_argument("query", help="Search query")
    cmd.add_argument("-k", type=int, default=_DEFAULT_K, help="Number of results")


def _add_info_cmd(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the info subcommand."""
    cmd = sub.add_parser("info", help="Show index info")
    cmd.add_argument("path", help="Path to the index")


def _run_index(args: argparse.Namespace) -> None:
    """Build an index from stdin lines."""
    embedder = FnvHashEmbedder()
    index = VectorIndex(dimension=embedder.dimension, embedder_id=embedder.model_id)

    for line_num, line in enumerate(sys.stdin):
        text = line.strip()
        if text:
            doc_id = f"doc-{line_num}"
            embedding = embedder.embed(text)
            index.add(doc_id, embedding)

    index.save(args.path)
    sys.stdout.write(f"Indexed {index.count} documents -> {args.path}\n")


def _run_search(args: argparse.Namespace) -> None:
    """Search an existing index."""
    embedder = FnvHashEmbedder()
    index = VectorIndex.load(args.path)
    query_vec = embedder.embed(args.query)
    hits = index.search(query_vec, k=args.k)

    for hit in hits:
        sys.stdout.write(f"{hit.score:.4f}  {hit.doc_id}\n")


def _run_info(args: argparse.Namespace) -> None:
    """Show index metadata."""
    index = VectorIndex.load(args.path)
    sys.stdout.write(f"Documents: {index.count}\n")
    sys.stdout.write(f"Dimension: {index.dimension}\n")
    sys.stdout.write(f"Embedder:  {index.embedder_id}\n")


if __name__ == "__main__":
    main()
