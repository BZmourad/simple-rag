"""ChromaDB-backed vector store."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.chunker import Chunk
from src.embeddings import OllamaEmbedder

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    pass


class VectorStore:
    """Thin wrapper around a Chroma collection.

    Chroma is configured with a cosine space so that "distance" is
    `1 - cosine_similarity`. Similarity is what we use for thresholds.
    """

    def __init__(self, path: Path, collection: str, embedder: OllamaEmbedder) -> None:
        self.path = path
        self.collection_name = collection
        self.embedder = embedder

        try:
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialise ChromaDB: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks, skipping IDs that already exist. Returns number inserted."""
        if not chunks:
            return 0

        existing_ids = self._existing_ids([c.chunk_id for c in chunks])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            return 0

        texts = [c.text for c in new_chunks]
        try:
            embeddings = self.embedder.embed(texts)
        except Exception as exc:
            raise VectorStoreError(f"Failed to embed chunks: {exc}") from exc

        try:
            self._collection.add(
                ids=[c.chunk_id for c in new_chunks],
                documents=texts,
                embeddings=embeddings,
                metadatas=[c.metadata for c in new_chunks],
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to store chunks in ChromaDB: {exc}") from exc

        return len(new_chunks)

    def _existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        try:
            result = self._collection.get(ids=ids, include=[])
        except Exception:
            # Older Chroma versions may not accept include=[] — fall back.
            result = self._collection.get(ids=ids)
        return set(result.get("ids", []) or [])

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def query(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Return a list of {text, metadata, similarity, distance} dicts."""
        if not query_text.strip():
            return []

        try:
            query_vec = self.embedder.embed_one(query_text)
        except Exception as exc:
            raise VectorStoreError(f"Failed to embed query: {exc}") from exc

        try:
            result = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for text, meta, dist in zip(documents, metadatas, distances):
            distance = float(dist) if dist is not None else 1.0
            # Cosine space: distance in [0, 2] -> similarity in [-1, 1].
            similarity = max(0.0, 1.0 - distance)
            hits.append(
                {
                    "text": text,
                    "metadata": dict(meta or {}),
                    "distance": distance,
                    "similarity": similarity,
                }
            )
        return hits

    # ------------------------------------------------------------------ #
    # Metadata / admin
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise VectorStoreError(f"Failed to count collection: {exc}") from exc

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to reset collection: {exc}") from exc

    def distinct_sources(self) -> set[str]:
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Failed to list sources: {exc}") from exc
        metadatas = result.get("metadatas") or []
        return {m.get("source", "") for m in metadatas if m and m.get("source")}