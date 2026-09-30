"""Cross-encoder re-ranking to improve retrieval precision."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RerankedChunk:
    text: str
    metadata: dict
    rerank_score: float
    original_score: float


class CrossEncoderReranker:
    """Re-rank candidate chunks using a cross-encoder model.

    Cross-encoders jointly encode (query, passage) pairs, which is slower
    but far more accurate than cosine similarity on independent embeddings.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
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
        """Re-rank candidates and return top_k by cross-encoder score."""
        if not candidates:
            return []

        self._load()
        pairs = [(query, c.text) for c in candidates]

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            logger.error("Reranker inference failed: %s", exc)
            # Fallback to original ordering
            return [
                RerankedChunk(
                    text=c.text,
                    metadata=c.metadata,
                    rerank_score=getattr(c, "similarity", 0.0),
                    original_score=getattr(c, "similarity", 0.0),
                )
                for c in candidates[:top_k]
            ]

        scored = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        return [
            RerankedChunk(
                text=c.text,
                metadata=c.metadata,
                rerank_score=float(score),
                original_score=getattr(c, "similarity", 0.0),
            )
            for c, score in scored[:top_k]
        ]