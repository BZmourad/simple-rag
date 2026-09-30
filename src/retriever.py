"""Retrieval with relevance threshold on top of the vector store."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    similarity: float
    distance: float


class Retriever:
    def __init__(self, vector_store: VectorStore, top_k: int, threshold: float) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return only chunks above the similarity threshold, ordered by similarity."""
        raw = self.vector_store.query(question, top_k=self.top_k)
        kept: list[RetrievedChunk] = []
        for hit in raw:
            if hit["similarity"] >= self.threshold:
                kept.append(
                    RetrievedChunk(
                        text=hit["text"],
                        metadata=hit["metadata"],
                        similarity=hit["similarity"],
                        distance=hit["distance"],
                    )
                )
        kept.sort(key=lambda c: c.similarity, reverse=True)
        logger.debug(
            "Retrieved %d/%d chunks above threshold %.2f",
            len(kept),
            len(raw),
            self.threshold,
        )
        return kept