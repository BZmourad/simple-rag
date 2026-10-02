"""RAGAS-based evaluation: Faithfulness, ContextPrecision, ContextRecall."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)


def _load_dataset(path: Path) -> list[dict]:
    """Load a JSON file with items containing question/ground_truth/answer/contexts."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("items", [])


def run_ragas_eval(cfg) -> dict:
    """Run RAGAS evaluation against the configured dataset.

    The dataset must contain items with:
      - question: str
      - ground_truth: str
      - answer: str (from your RAG pipeline)
      - contexts: list[str] (retrieved chunks)
    """
    if not cfg.ragas_dataset_path.exists():
        raise FileNotFoundError(f"RAGAS dataset not found: {cfg.ragas_dataset_path}")

    items = _load_dataset(cfg.ragas_dataset_path)
    if not items:
        raise ValueError("RAGAS dataset is empty.")

    # Wrap Ollama as the judge LLM and embeddings
    judge_llm = ChatOllama(model=cfg.ragas_judge_model)
    judge_emb = OllamaEmbeddings(model=cfg.ragas_judge_embedding)
    wrapped_llm = LangchainLLMWrapper(judge_llm)
    wrapped_emb = LangchainEmbeddingsWrapper(judge_emb)

    # Configure the metrics with the judge
    metrics = [faithfulness, context_precision, context_recall]
    for m in metrics:
        m.llm = wrapped_llm
        if hasattr(m, "embeddings"):
            m.embeddings = wrapped_emb

    dataset = Dataset.from_list(items)
    results = evaluate(dataset=dataset, metrics=metrics, llm=wrapped_llm, embeddings=wrapped_emb)

    # Convert to plain dict
    summary = {
        "faithfulness": float(results.get("faithfulness", 0.0)),
        "context_precision": float(results.get("context_precision", 0.0)),
        "context_recall": float(results.get("context_recall", 0.0)),
    }
    logger.info("RAGAS results: %s", summary)
    return summary

def print_ragas_report(cfg, summary: dict) -> None:
    """Print RAGAS summary and record it in the SQLite tracker."""
    print()
    print("RAGAS evaluation results")
    print("=" * 40)
    print(f"  Faithfulness      : {summary['faithfulness']:.4f}")
    print(f"  Context precision : {summary['context_precision']:.4f}")
    print(f"  Context recall    : {summary['context_recall']:.4f}")

    if not cfg.ragas_tracking_enabled:
        return

    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd="/opt/simple-rag",
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = None

    from src.eval.tracker import RagasTracker
    tracker = RagasTracker(cfg.ragas_db_path)
    run_id = tracker.record(
        collection=cfg.default_collection,
        num_items=len(json.loads(cfg.ragas_dataset_path.read_text()).get("items", [])),
        faithfulness=summary["faithfulness"],
        context_precision=summary["context_precision"],
        context_recall=summary["context_recall"],
        git_commit=commit,
    )
    comparison = tracker.compare(cfg.default_collection)
    tracker.close()

    print()
    print(f"Run #{run_id} recorded (commit: {commit or 'unknown'})")
    print(f"Regression status: {comparison['status']}")
    if comparison.get("regressions"):
        print("⚠️  Regressions detected:")
        for metric, delta in comparison["regressions"].items():
            print(f"    {metric}: {delta:+.4f}")