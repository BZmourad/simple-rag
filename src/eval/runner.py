"""Evaluation harness: retrieval hit-rate + answer faithfulness."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from src.eval.tracker import RagasTracker
from src.config import Config
from src.history_store import HistoryStore
from src.hybrid_retriever import HybridRetriever
from src.llm import OllamaLLM
from src.rag import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.vector_store import create_vector_store

logger = logging.getLogger(__name__)


@dataclass
class EvalItem:
    question: str
    expected_sources: list[str]
    expected_answer_criteria: str = ""


def _load_dataset(path: Path) -> tuple[str, list[EvalItem]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    collection = data.get("collection", "default")
    items = [
        EvalItem(
            question=it["question"],
            expected_sources=it.get("expected_sources", []),
            expected_answer_criteria=it.get("expected_answer_criteria", ""),
        )
        for it in data.get("items", [])
    ]
    return collection, items


def _hit_rate(item: EvalItem, retrieved_sources: list[str]) -> bool:
    """Retrieval hit: at least one expected source appears in retrieved."""
    if not item.expected_sources:
        return not retrieved_sources  # refusal case: no sources = correct
    return any(src in retrieved_sources for src in item.expected_sources)


def run_eval(cfg: Config) -> int:
    if not cfg.eval_dataset_path.exists():
        print(f"ERROR: dataset not found: {cfg.eval_dataset_path}")
        return 2

    collection, items = _load_dataset(cfg.eval_dataset_path)
    print(f"Running eval on {len(items)} items (collection='{collection}')")

    store = create_vector_store(cfg)
    retriever = HybridRetriever(
        vector_store=store, top_k=cfg.top_k,
        dense_weight=cfg.hybrid_dense_weight, sparse_weight=cfg.hybrid_sparse_weight,
        rrf_k=cfg.rrf_k, collection=collection,
    )
    reranker = CrossEncoderReranker(cfg.reranker_model) if cfg.reranker_enabled else None
    llm = OllamaLLM(model=cfg.llm_model, base_url=cfg.ollama_base_url)

    hits = 0
    results = []
    for item in items:
        pipeline = RAGPipeline(
            retriever=retriever, llm=llm, reranker=reranker,
            max_history_messages=0,  # no history in eval
            rerank_top_k=cfg.rerank_top_k,
            relevance_threshold=cfg.relevance_threshold,
            history_store=None, collection=collection,
        )
        result = pipeline.answer(item.question)
        retrieved_sources = [c.metadata.get("filename", "") for c in result.sources]
        hit = _hit_rate(item, retrieved_sources)
        hits += int(hit)
        results.append({
            "question": item.question,
            "hit": hit,
            "answer_preview": result.answer[:120],
            "retrieved_sources": retrieved_sources,
        })
        print(f"  [{'HIT' if hit else 'MISS'}] {item.question[:60]}")

    hit_rate = hits / len(items) if items else 0.0
    print()
    print(f"Retrieval hit-rate: {hit_rate:.2%} ({hits}/{len(items)})")
    if cfg.ragas_tracking_enabled:
        tracker = RagasTracker(cfg.ragas_db_path)
        # Get git commit if available
        import subprocess
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd="/opt/simple-rag").decode().strip()
        except Exception:
            commit = None

        run_id = tracker.record(
            collection=collection,
            num_items=len(items),
            faithfulness=summary["faithfulness"],
            context_precision=summary["context_precision"],
            context_recall=summary["context_recall"],
            git_commit=commit,
        )
        comparison = tracker.compare(collection)
        print(f"\nRun #{run_id} recorded.")
        print(f"Regression check: {comparison['status']}")
        if comparison.get("regressions"):
            print("⚠️  Regressions detected:")
            for metric, delta in comparison["regressions"].items():
                print(f"    {metric}: {delta:+.4f}")
        tracker.close()