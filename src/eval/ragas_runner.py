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