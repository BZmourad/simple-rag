"""Cross-encoder re-ranking with process-wide singleton cache."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RerankedChunk:
    text: str
    metadata: dict
    rerank_score: float
    original_score: float


_RERANKER_INSTANCES: dict[str, "CrossEncoderReranker"] = {}
_RERANKER_LOCK = threading.Lock()


class CrossEncoderReranker:
    """Re-rank candidate chunks using a cross-encoder model.

    The model is loaded lazily on first use and cached at module level so that
    multiple requests (FastAPI, Celery workers, CLI) share the same instance.
    This avoids repeatedly loading the ~90MB model per request.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading cross-encoder model: %s", self.model_name)
                self._model = CrossEncoder(self.model_name, max_length=512)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load reranker model '{self.model_name}'. "
                    f"Run: pip install sentence-transformers. Error: {exc}"
                ) from exc

    def rerank(self, query: str, candidates: list, top_k: int) -> list[RerankedChunk]:
        if not candidates:
            return []
        self._load()
        pairs = [(query, c.text) for c in candidates]
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            logger.error("Reranker inference failed: %s", exc)
            return [
                RerankedChunk(
                    text=c.text, metadata=c.metadata,
                    rerank_score=getattr(c, "similarity", 0.0),
                    original_score=getattr(c, "similarity", 0.0),
                )
                for c in candidates[:top_k]
            ]
        scored = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        return [
            RerankedChunk(
                text=c.text, metadata=c.metadata,
                rerank_score=float(score),
                original_score=getattr(c, "similarity", 0.0),
            )
            for c, score in scored[:top_k]
        ]


def get_reranker(model_name: str) -> CrossEncoderReranker:
    """Return a cached singleton for the given model name.

    Thread-safe. The cache key is the model name, so different models get
    different instances while repeated calls with the same name reuse one.
    """
    with _RERANKER_LOCK:
        if model_name not in _RERANKER_INSTANCES:
            logger.info("Creating new reranker singleton for model: %s", model_name)
            _RERANKER_INSTANCES[model_name] = CrossEncoderReranker(model_name)
        return _RERANKER_INSTANCES[model_name]


def reset_reranker_cache() -> None:
    """Clear the singleton cache (for tests)."""
    with _RERANKER_LOCK:
        _RERANKER_INSTANCES.clear()