"""Command-line interface: ingest, chat, stats, clear."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.chunker import chunk_document
from src.config import Config, load_config
from src.document_loader import load_documents
from src.embeddings import EmbeddingError, OllamaEmbedder
from src.llm import LLMError, OllamaLLM
from src.rag import RAGPipeline
from src.retriever import Retriever
from src.vector_store import VectorStore, VectorStoreError

BANNER = """========================================
       Simple RAG Knowledge Agent
========================================

Type your question.
Type 'sources' to show the sources of the last answer.
Type 'exit' to quit.
"""


def _setup_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    # Silence noisy libraries unless debugging.
    if not debug:
        for noisy in ("chromadb", "httpx", "urllib3", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_components(cfg: Config):
    embedder = OllamaEmbedder(model=cfg.embedding_model, base_url=cfg.ollama_base_url)
    store = VectorStore(path=cfg.chroma_path, collection=cfg.chroma_collection, embedder=embedder)
    retriever = Retriever(vector_store=store, top_k=cfg.top_k, threshold=cfg.relevance_threshold)
    llm = OllamaLLM(model=cfg.llm_model, base_url=cfg.ollama_base_url)
    return embedder, store, retriever, llm


def _print_sources(sources) -> None:
    from src.prompts import format_sources

    if not sources:
        print("\nSources: (none)")
        return
    print("\nSources:")
    print(format_sources(sources))


# ---------------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------------- #
def cmd_ingest(cfg: Config) -> int:
    try:
        embedder, store, _, _ = _build_components(cfg)
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Quick health check on the embedding model.
    try:
        embedder.health_check()
    except EmbeddingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    documents_dir = cfg.documents_path
    if not documents_dir.exists():
        print(f"ERROR: documents directory not found: {documents_dir}", file=sys.stderr)
        return 2

    logging.info("Loading documents from %s ...", documents_dir)
    total_docs = 0
    total_chunks_created = 0
    total_chunks_stored = 0

    for doc in load_documents(documents_dir):
        total_docs += 1
        chunks = chunk_document(doc, cfg.chunk_size, cfg.chunk_overlap)
        total_chunks_created += len(chunks)
        logging.info("Created %d chunks from %s", len(chunks), doc.filename)

        try:
            stored = store.add_chunks(chunks)
        except VectorStoreError as exc:
            logging.error("Failed to store chunks for %s: %s", doc.filename, exc)
            continue
        total_chunks_stored += stored

    print()
    print(f"Documents processed: {total_docs}")
    print(f"Chunks created: {total_chunks_created}")
    print(f"Chunks stored: {total_chunks_stored}")
    print(f"Total chunks in collection: {store.count()}")
    print("Ingestion completed successfully.")
    return 0


def cmd_stats(cfg: Config) -> int:
    try:
        embedder, store, _, _ = _build_components(cfg)
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        chunk_count = store.count()
        sources = store.distinct_sources()
    except VectorStoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Configuration")
    print(f"  Documents directory : {cfg.documents_path}")
    print(f"  Chroma path         : {cfg.chroma_path}")
    print(f"  Collection          : {cfg.chroma_collection}")
    print(f"  Embedding model     : {cfg.embedding_model}")
    print(f"  LLM model           : {cfg.llm_model}")
    print(f"  Chunk size          : {cfg.chunk_size}")
    print(f"  Chunk overlap       : {cfg.chunk_overlap}")
    print(f"  Top K               : {cfg.top_k}")
    print(f"  Relevance threshold : {cfg.relevance_threshold}")
    print()
    print("Knowledge base")
    print(f"  Documents           : {len(sources)}")
    print(f"  Chunks              : {chunk_count}")
    if sources:
        print("  Sources:")
        for s in sorted(sources):
            print(f"    - {s}")
    return 0


def cmd_clear(cfg: Config) -> int:
    answer = input("Are you sure you want to delete the knowledge base? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Aborted.")
        return 0

    try:
        _, store, _, _ = _build_components(cfg)
        store.reset()
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Knowledge base deleted.")
    return 0


def cmd_chat(cfg: Config) -> int:
    try:
        _, store, retriever, llm = _build_components(cfg)
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        chunk_count = store.count()
    except VectorStoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if chunk_count == 0:
        print(
            "The knowledge base is empty. Run `python main.py ingest` first.",
            file=sys.stderr,
        )
        return 2

    # Best-effort LLM health check.
    try:
        llm.health_check()
    except LLMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        max_history_messages=cfg.max_history_messages,
        debug=cfg.debug,
    )

    print(BANNER)
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        lowered = question.lower()
        if lowered in {"exit", "quit"}:
            break
        if lowered == "sources":
            print()
            print(pipeline.format_last_sources())
            print()
            continue
        if lowered == "clear":
            pipeline.clear_history()
            print("Conversation history cleared.")
            continue

        try:
            result = pipeline.answer(question)
        except Exception as exc:  # last-resort safety net
            logging.error("Unexpected error: %s", exc)
            print("AI: Something went wrong while answering. See logs for details.")
            continue

        print()
        print(f"AI: {result.answer}")
        if result.sources:
            print()
            print("Sources:")
            from src.prompts import format_sources

            print(format_sources(result.sources))
        print()

    return 0


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simple-rag",
        description="Simple RAG PoC CLI (Ollama + ChromaDB).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Load documents and update the vector store.")
    sub.add_parser("chat", help="Start the interactive RAG chat.")
    sub.add_parser("stats", help="Show knowledge base and model information.")
    sub.add_parser("clear", help="Delete the vector collection (with confirmation).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.debug:
        cfg = cfg.__class__(**{**cfg.__dict__, "debug": True})

    _setup_logging(cfg.debug)

    if args.command == "ingest":
        return cmd_ingest(cfg)
    if args.command == "chat":
        return cmd_chat(cfg)
    if args.command == "stats":
        return cmd_stats(cfg)
    if args.command == "clear":
        return cmd_clear(cfg)

    parser.print_help()
    return 1