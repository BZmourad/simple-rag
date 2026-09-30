"""CLI v2: streaming, collections, persistent history."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.chunker import chunk_document
from src.config import Config, load_config
from src.document_loader import load_documents
from src.embeddings import EmbeddingError
from src.history_store import HistoryStore
from src.hybrid_retriever import HybridRetriever
from src.llm import LLMError, OllamaLLM
from src.prompts import format_sources
from src.rag import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.vector_store import VectorStoreError, create_vector_store

BANNER = """========================================
     Simple RAG Knowledge Agent v2
========================================

Commands:
  sources   - show sources of last answer
  history   - show conversation history
  clear     - clear conversation history
  exit      - quit
"""


def _setup_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    if not debug:
        for noisy in ("chromadb", "httpx", "urllib3", "sentence_transformers", "transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_components(cfg: Config, collection: str | None = None):
    store = create_vector_store(cfg)
    retriever = HybridRetriever(
        vector_store=store,
        top_k=cfg.top_k,
        dense_weight=cfg.hybrid_dense_weight,
        sparse_weight=cfg.hybrid_sparse_weight,
        rrf_k=cfg.rrf_k,
        collection=collection,
    )
    reranker = CrossEncoderReranker(cfg.reranker_model) if cfg.reranker_enabled else None
    llm = OllamaLLM(model=cfg.llm_model, base_url=cfg.ollama_base_url)
    return store, retriever, reranker, llm


def cmd_ingest(cfg: Config, collection: str | None = None) -> int:
    collection = collection or cfg.default_collection
    try:
        store, retriever, _, _ = _build_components(cfg, collection)
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not cfg.documents_path.exists():
        print(f"ERROR: documents directory not found: {cfg.documents_path}", file=sys.stderr)
        return 2

    logging.info("Loading documents into collection '%s' from %s", collection, cfg.documents_path)
    total_docs = total_created = total_stored = 0

    for doc in load_documents(cfg.documents_path, cfg=cfg):
        total_docs += 1
        chunks = chunk_document(doc, cfg.chunk_size, cfg.chunk_overlap)
        total_created += len(chunks)
        logging.info("Created %d chunks from %s", len(chunks), doc.filename)
        try:
            stored = store.add_chunks(chunks, collection=collection)
        except VectorStoreError as exc:
            logging.error("Failed to store chunks for %s: %s", doc.filename, exc)
            continue
        total_stored += stored

    retriever.invalidate_bm25()
    print()
    print(f"Collection: {collection}")
    print(f"Documents processed: {total_docs}")
    print(f"Chunks created: {total_created}")
    print(f"Chunks stored: {total_stored}")
    print(f"Total chunks: {store.count(collection=collection)}")
    print("Ingestion completed successfully.")
    return 0


def cmd_stats(cfg: Config, collection: str | None = None) -> int:
    collection = collection or cfg.default_collection
    try:
        store, _, _, _ = _build_components(cfg, collection)
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Configuration")
    print(f"  Backend             : {cfg.vector_backend}")
    print(f"  Collection          : {collection}")
    print(f"  Embedding model     : {cfg.embedding_model}")
    print(f"  LLM model           : {cfg.llm_model}")
    print(f"  Hybrid enabled      : {cfg.hybrid_enabled}")
    print(f"  Reranker            : {'enabled' if cfg.reranker_enabled else 'disabled'} ({cfg.reranker_model})")
    print(f"  OCR                 : {'enabled' if cfg.ocr_enabled else 'disabled'}")
    print(f"  Top K / Rerank Top K: {cfg.top_k} / {cfg.rerank_top_k}")
    print(f"  Relevance threshold : {cfg.relevance_threshold}")
    print()
    print("Knowledge base")
    print(f"  Collections         : {', '.join(store.list_collections()) or '(none)'}")
    print(f"  Chunks in '{collection}': {store.count(collection=collection)}")
    sources = store.distinct_sources(collection=collection)
    print(f"  Documents           : {len(sources)}")
    for s in sorted(sources):
        print(f"    - {s}")
    return 0


def cmd_clear(cfg: Config, collection: str | None = None) -> int:
    collection = collection or cfg.default_collection
    answer = input(f"Delete collection '{collection}'? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Aborted.")
        return 0
    try:
        store, _, _, _ = _build_components(cfg, collection)
        store.reset(collection=collection)
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Collection '{collection}' deleted.")
    return 0


def cmd_chat(cfg: Config, collection: str | None = None, session: str = "default") -> int:
    collection = collection or cfg.default_collection
    try:
        store, retriever, reranker, llm = _build_components(cfg, collection)
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if store.count(collection=collection) == 0:
        print(f"Collection '{collection}' is empty. Run: python main.py ingest", file=sys.stderr)
        return 2

    try:
        llm.health_check()
    except LLMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    history_store = HistoryStore(cfg.history_dir, session_id=session)
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        reranker=reranker,
        max_history_messages=cfg.max_history_messages,
        rerank_top_k=cfg.rerank_top_k,
        relevance_threshold=cfg.relevance_threshold,
        history_store=history_store,
        collection=collection,
        debug=cfg.debug,
    )

    print(BANNER)
    print(f"Collection: {collection} | Session: {session}\n")

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
        if lowered == "history":
            for m in history_store.messages:
                print(f"  {m['role']}: {m['content'][:120]}")
            print()
            continue
        if lowered == "clear":
            pipeline.clear_history()
            print("History cleared.")
            continue

        print()
        print("AI: ", end="", flush=True)
        sources = []
        try:
            for event in pipeline.stream_answer(question):
                if event["type"] == "token":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "done":
                    sources = event.get("sources", [])
        except Exception as exc:
            logging.error("Unexpected error: %s", exc)
            print("\nAI: Something went wrong. See logs.")
            continue
        print()
        if sources:
            print()
            print("Sources:")
            print(format_sources(sources))
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simple-rag")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--collection", default=None, help="Collection / namespace name")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    sub.add_parser("chat")
    sub.add_parser("stats")
    sub.add_parser("clear")
    sub.add_parser("serve", help="Start the FastAPI server")
    sub.add_parser("eval", help="Run the evaluation harness")
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
        return cmd_ingest(cfg, args.collection)
    if args.command == "chat":
        return cmd_chat(cfg, args.collection)
    if args.command == "stats":
        return cmd_stats(cfg, args.collection)
    if args.command == "clear":
        return cmd_clear(cfg, args.collection)
    if args.command == "serve":
        from src.api.server import run_server
        run_server(cfg)
        return 0
    if args.command == "eval":
        from src.eval.runner import run_eval
        return run_eval(cfg)
    parser.print_help()
    return 1