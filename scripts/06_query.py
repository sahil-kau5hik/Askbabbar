#!/usr/bin/env python3
"""
Phase 6 Script: Interactive CLI Query Tool
Test the full RAG pipeline from the command line before launching the API.

Usage:
  python scripts/06_query.py "What is a linked list?"
  python scripts/06_query.py "Explain binary search tree" --top-k 3
  python scripts/06_query.py --search-only "recursion explained"
  python scripts/06_query.py --interactive
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.config import settings
from backend.llm.generator import generate_answer
from backend.models.schemas import QueryRequest
from backend.retrieval.retriever import search
from backend.utils.logging import logger


def run_query(
    query: str,
    top_k: int = settings.TOP_K,
    threshold: float = settings.SIMILARITY_THRESHOLD,
    search_only: bool = False,
    expand_adjacent: bool = True,
) -> None:
    request = QueryRequest(
        query=query,
        top_k=top_k,
        similarity_threshold=threshold,
        expand_adjacent=expand_adjacent,
    )

    sources = search(request)

    if not sources:
        print("\n[No results found above similarity threshold.]\n")
        return

    if search_only:
        print(f"\n=== Search Results for: '{query}' ===\n")
        for i, src in enumerate(sources, 1):
            lec = f"Lecture {src.lecture_number}" if src.lecture_number else src.title
            print(f"[{i}] {lec} | {src.timestamp_range} | score={src.relevance_score}")
            print(f"     URL: {src.youtube_url}")
            print(f"     {src.matched_transcript_snippet[:200]}...")
            print()
        return

    response = generate_answer(query=query, sources=sources)

    print(f"\n{'=' * 60}")
    print(f"QUESTION: {response.query}")
    print(f"{'=' * 60}")
    print(f"\nANSWER (confidence={response.confidence}):\n")
    print(response.answer)
    print(f"\n{'─' * 60}")
    print(f"SOURCES ({len(response.sources)}):")
    for i, src in enumerate(response.sources, 1):
        lec = f"Lecture {src.lecture_number}" if src.lecture_number else src.title
        print(f"  [{i}] {lec} | {src.timestamp_range} | score={src.relevance_score}")
        print(f"       {src.youtube_url}")
    print(f"{'=' * 60}\n")


def interactive_mode(top_k: int, threshold: float, expand_adjacent: bool) -> None:
    print("\n=== Love Babbar DSA RAG - Interactive Query Mode ===")
    print("Type your question and press Enter. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        run_query(query, top_k=top_k, threshold=threshold, expand_adjacent=expand_adjacent)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Query the Love Babbar DSA RAG system"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Your DSA/C++ question",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.TOP_K,
        help=f"Number of sources to retrieve (default: {settings.TOP_K})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.SIMILARITY_THRESHOLD,
        help=f"Minimum cosine similarity score (default: {settings.SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Skip LLM generation; show raw search results only",
    )
    parser.add_argument(
        "--no-adjacent",
        action="store_true",
        help="Disable adjacent chunk expansion",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start interactive query session",
    )
    args = parser.parse_args()

    expand = not args.no_adjacent

    if args.interactive:
        interactive_mode(top_k=args.top_k, threshold=args.threshold, expand_adjacent=expand)
    elif args.query:
        run_query(
            query=args.query,
            top_k=args.top_k,
            threshold=args.threshold,
            search_only=args.search_only,
            expand_adjacent=expand,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
