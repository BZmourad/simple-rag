"""CLI v4: streaming, collections, JWT auth, graph, RAGAS tracking."""

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
from src.query_expander import QueryExpander
from src.rag import RAGPipeline
from src.reranker import get_reranker
from src.vector_store import VectorStoreError, create_vector_store

BANNER = """========================================
     Simple RAG Knowledge Agent v4
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
        for noisy in (
            "chromadb", "httpx", "httpcore", "urllib3",
            "sentence_transformers", "transformers",
            "posthog", "chromadb.telemetry",
            "falkordb", "redislite", "qdrant_client",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)
            logging.getLogger(noisy).propagate = False


def _build_components(cfg: Config, collection: str | None = None, with_graph: bool = False):
    store = create_vector_store(cfg)

    retriever = HybridRetriever(
        vector_store=store,
        top_k=cfg.top_k,
        dense_weight=cfg.hybrid_dense_weight,
        sparse_weight=cfg.hybrid_sparse_weight,
        rrf_k=cfg.rrf_k,
        collection=collection,
    )
    reranker = get_reranker(cfg.reranker_model) if cfg.reranker_enabled else None
    llm = OllamaLLM(model=cfg.llm_model, base_url=cfg.ollama_base_url)

    expander = None
    if cfg.query_expansion_enabled:
        expander = QueryExpander(
            llm=llm,
            mode=cfg.query_expansion_mode,
            num_queries=cfg.query_expansion_num_queries,
            hyde_temp=cfg.query_expansion_hyde_temp,
            redis_url=cfg.rate_limit_redis_url,
            cache_ttl=3600,
        )

    graph_store = None
    if with_graph and cfg.graph_rag_enabled:
        try:
            from src.graph_store_falkor import FalkorGraphStore
            graph_store = FalkorGraphStore(path=cfg.falkordb_path)
        except Exception as exc:
            logging.warning("Graph store unavailable: %s", exc)

    return store, retriever, reranker, llm, expander, graph_store


# ---------------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------------- #
def cmd_ingest(cfg: Config, collection: str | None = None) -> int:
    collection = collection or cfg.default_collection
    try:
        store, retriever, _, _, _, graph_store = _build_components(
            cfg, collection, with_graph=cfg.graph_rag_enabled
        )
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not cfg.documents_path.exists():
        print(f"ERROR: documents directory not found: {cfg.documents_path}", file=sys.stderr)
        return 2

    logging.info("Loading documents into collection '%s' from %s", collection, cfg.documents_path)
    total_docs = total_created = total_stored = 0

    graph_extractor = None
    if graph_store is not None:
        from src.graph_rag import GraphRAGExtractor
        graph_extractor = GraphRAGExtractor(
            method=cfg.graph_extraction_method,
            threshold=cfg.graph_entity_threshold,
        )

    for doc in load_documents(cfg.documents_path, cfg=cfg):
        total_docs += 1
        chunks = chunk_document(doc, cfg.chunk_size, cfg.chunk_overlap)
        total_created += len(chunks)
        logging.info("Created %d chunks from %s", len(chunks), doc.filename)

        try:
            stored = store.add_chunks(chunks, collection=collection)
            total_stored += stored
        except VectorStoreError as exc:
            logging.error("Failed to store chunks for %s: %s", doc.filename, exc)
            continue

        # Populate graph store
        if graph_extractor is not None and graph_store is not None:
            for chunk in chunks:
                try:
                    entities, relations = graph_extractor.extract(chunk.text)
                    for e in entities:
                        graph_store.add_entity(e.name, e.entity_type)
                    for r in relations:
                        graph_store.add_relation(r.source, r.relation, r.target)
                except Exception as exc:
                    logging.warning("Graph extraction failed for chunk: %s", exc)

    retriever.invalidate_bm25()

    print()
    print(f"Collection: {collection}")
    print(f"Documents processed: {total_docs}")
    print(f"Chunks created: {total_created}")
    print(f"Chunks stored: {total_stored}")
    try:
        print(f"Total chunks: {store.count(collection=collection)}")
    except VectorStoreError as exc:
        logging.error("Could not read final chunk count: %s", exc)
        print("Total chunks: unknown")
    if graph_store is not None:
        print(f"Graph: {graph_store.stats()}")
    print("Ingestion completed successfully.")
    return 0


def cmd_stats(cfg: Config, collection: str | None = None) -> int:
    collection = collection or cfg.default_collection
    try:
        store, _, _, _, _, graph_store = _build_components(
            cfg, collection, with_graph=cfg.graph_rag_enabled
        )
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
    print(f"  Query expansion     : {cfg.query_expansion_mode if cfg.query_expansion_enabled else 'disabled'}")
    print(f"  Graph RAG           : {cfg.graph_backend if cfg.graph_rag_enabled else 'disabled'}")
    print(f"  Auth mode           : {cfg.auth_mode}")
    print(f"  Rate limit          : {cfg.rate_limit_per_minute}/min, {cfg.rate_limit_per_day}/day")
    print(f"  Top K / Rerank Top K: {cfg.top_k} / {cfg.rerank_top_k}")
    print(f"  Relevance threshold : {cfg.relevance_threshold}")
    print()
    print("Knowledge base")
    try:
        collections = store.list_collections()
    except Exception:
        collections = []
    print(f"  Collections         : {', '.join(collections) or '(none)'}")
    try:
        print(f"  Chunks in '{collection}': {store.count(collection=collection)}")
        sources = store.distinct_sources(collection=collection)
        print(f"  Documents           : {len(sources)}")
        for s in sorted(sources):
            print(f"    - {s}")
    except VectorStoreError as exc:
        print(f"  (error reading collection: {exc})")

    if graph_store is not None:
        print()
        print("Knowledge graph")
        try:
            stats = graph_store.stats()
            print(f"  Nodes               : {stats.get('nodes', 0)}")
            print(f"  Edges               : {stats.get('edges', 0)}")
        except Exception as exc:
            print(f"  (error reading graph: {exc})")
    return 0


def cmd_clear(cfg: Config, collection: str | None = None) -> int:
    collection = collection or cfg.default_collection
    answer = input(f"Delete collection '{collection}'? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Aborted.")
        return 0
    try:
        store, _, _, _, _, _ = _build_components(cfg, collection)
        store.reset(collection=collection)
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Collection '{collection}' deleted.")
    return 0


def cmd_chat(cfg: Config, collection: str | None = None, session: str = "default") -> int:
    collection = collection or cfg.default_collection
    try:
        store, retriever, reranker, llm, expander, graph_store = _build_components(
            cfg, collection, with_graph=cfg.graph_rag_enabled
        )
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        if store.count(collection=collection) == 0:
            print(f"Collection '{collection}' is empty. Run: python main.py ingest", file=sys.stderr)
            return 2
    except VectorStoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
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
        query_expander=expander,
        knowledge_graph=graph_store,
        graph_max_depth=cfg.graph_max_depth,
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


# ---------------------------------------------------------------------- #
# Parser and entrypoint
# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simple-rag")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--collection", default=None, help="Collection / namespace name")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Load documents and update the vector store")
    sub.add_parser("chat", help="Start the interactive RAG chat")
    sub.add_parser("stats", help="Show knowledge base and model information")
    sub.add_parser("clear", help="Delete the vector collection (with confirmation)")
    sub.add_parser("serve", help="Start the FastAPI server")
    sub.add_parser("eval", help="Run the evaluation harness")
    sub.add_parser("ragas", help="Run RAGAS evaluation and track regression")
    sub.add_parser("graph-stats", help="Show knowledge graph statistics")
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
    if args.command == "ragas":
        from src.eval.ragas_runner import run_ragas_eval, print_ragas_report
        summary = run_ragas_eval(cfg)
        print_ragas_report(cfg, summary)
        return 0
    if args.command == "graph-stats":
        from src.graph_store_falkor import FalkorGraphStore
        try:
            store = FalkorGraphStore(path=cfg.falkordb_path)
            stats = store.stats()
            print(f"Nodes: {stats.get('nodes', 0)}")
            print(f"Edges: {stats.get('edges', 0)}")
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        return 0

    parser.print_help()
    return 1