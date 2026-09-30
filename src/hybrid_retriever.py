"""Hybrid retrieval: BM25 (sparse) + vector (dense) with RRF fusion."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


@dataclass
class HybridChunk:
    text: str
    metadata: dict
    similarity: float
    distance: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    chunk_id: str = ""


class HybridRetriever:
    """Combines dense vector search with BM25 keyword search via RRF."""

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 20,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        collection: str | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.collection = collection
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[str] = []
        self._bm25_metadatas: list[dict] = []
        self._bm25_ids: list[str] = []

    def _ensure_bm25_index(self) -> None:
        """Build BM25 index from the vector store's documents (lazy, cached)."""
        if self._bm25 is not None:
            return
        try:
            all_data = self.vector_store.get_all_documents(collection=self.collection)
        except Exception as exc:
            logger.warning("Could not build BM25 index: %s", exc)
            self._bm25 = None
            return

        ids = all_data.get("ids", [])
        docs = all_data.get("documents", [])
        metas = all_data.get("metadatas", [])

        if not docs:
            self._bm25 = None
            return

        tokenized = [_tokenize(d) for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus = docs
        self._bm25_metadatas = metas
        self._bm25_ids = ids
        logger.info("BM25 index built with %d documents", len(docs))

    def invalidate_bm25(self) -> None:
        """Call after ingestion so the index rebuilds on next query."""
        self._bm25 = None
        self._bm25_corpus = []
        self._bm25_metadatas = []
        self._bm25_ids = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[HybridChunk]:
        """Retrieve using hybrid search. Returns fused, ranked chunks."""
        k = top_k or self.top_k

        # Dense retrieval
        dense_hits = self.vector_store.query(query, top_k=k, collection=self.collection)

        # Sparse (BM25) retrieval
        self._ensure_bm25_index()
        sparse_hits: list[dict] = []
        if self._bm25 is not None:
            query_tokens = _tokenize(query)
            if query_tokens:
                bm25_scores = self._bm25.get_scores(query_tokens)
                indexed = sorted(
                    enumerate(bm25_scores),
                    key=lambda x: float(x[1]),
                    reverse=True,
                )[:k]
                for idx, score in indexed:
                    if idx < len(self._bm25_corpus):
                        sparse_hits.append({
                            "text": self._bm25_corpus[idx],
                            "metadata": self._bm25_metadatas[idx] if idx < len(self._bm25_metadatas) else {},
                            "bm25_score": float(score),
                            "chunk_id": self._bm25_ids[idx] if idx < len(self._bm25_ids) else "",
                        })

        # Fuse via Reciprocal Rank Fusion
        fused = self._rrf_fuse(dense_hits, sparse_hits, k)
        return fused

    def _rrf_fuse(self, dense: list[dict], sparse: list[dict], top_k: int) -> list[HybridChunk]:
        """Combine dense and sparse results using weighted RRF."""
        scores: dict[str, dict] = {}

        for rank, hit in enumerate(dense, start=1):
            key = hit.get("metadata", {}).get("chunk_id") or hit.get("text", "")[:80]
            rrf = self.dense_weight * (1.0 / (self.rrf_k + rank))
            if key not in scores:
                scores[key] = {
                    "text": hit["text"],
                    "metadata": hit.get("metadata", {}),
                    "similarity": hit.get("similarity", 0.0),
                    "distance": hit.get("distance", 0.0),
                    "bm25_score": 0.0,
                    "rrf_score": 0.0,
                }
            scores[key]["rrf_score"] += rrf

        for rank, hit in enumerate(sparse, start=1):
            key = hit.get("chunk_id") or hit.get("text", "")[:80]
            rrf = self.sparse_weight * (1.0 / (self.rrf_k + rank))
            if key not in scores:
                scores[key] = {
                    "text": hit["text"],
                    "metadata": hit.get("metadata", {}),
                    "similarity": 0.0,
                    "distance": 1.0,
                    "bm25_score": hit.get("bm25_score", 0.0),
                    "rrf_score": 0.0,
                }
            scores[key]["rrf_score"] += rrf
            scores[key]["bm25_score"] = hit.get("bm25_score", 0.0)

        ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]
        return [
            HybridChunk(
                text=r["text"],
                metadata=r["metadata"],
                similarity=r["similarity"],
                distance=r["distance"],
                bm25_score=r["bm25_score"],
                rrf_score=r["rrf_score"],
                chunk_id=r["metadata"].get("chunk_id", ""),
            )
            for r in ranked
        ]