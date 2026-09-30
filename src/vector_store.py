"""Vector store abstraction: Chroma and Qdrant backends with multi-collection support."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Protocol

import chromadb
from chromadb.config import Settings

from src.chunker import Chunk
from src.embeddings import OllamaEmbedder

logger = logging.getLogger(__name__)
_VALID_NAME = re.compile(r"[^a-zA-Z0-9._-]")

def sanitize_collection_name(name: str) -> str:
    """Return a ChromaDB/Qdrant-safe collection name.

    - Replaces invalid characters with '-'
    - Pads names shorter than 3 chars
    - Truncates names longer than 512 chars
    """
    if not name:
        return "default"
    cleaned = _VALID_NAME.sub("-", name.strip())
    cleaned = cleaned.strip("-._")  # must start/end with alnum
    if len(cleaned) < 3:
        cleaned = f"col-{cleaned}" if cleaned else "col-default"
    cleaned = cleaned[:512].rstrip("-._")
    if len(cleaned) < 3:
        cleaned = "col-default"
    return cleaned

class VectorStoreError(RuntimeError):
    pass


class VectorStoreProtocol(Protocol):
    def add_chunks(self, chunks: list[Chunk], collection: str | None = None) -> int: ...
    def query(self, query_text: str, top_k: int, collection: str | None = None) -> list[dict]: ...
    def count(self, collection: str | None = None) -> int: ...
    def reset(self, collection: str | None = None) -> None: ...
    def distinct_sources(self, collection: str | None = None) -> set[str]: ...
    def get_all_documents(self, collection: str | None = None) -> dict: ...
    def list_collections(self) -> list[str]: ...


# --------------------------------------------------------------------------- #
# Chroma backend
# --------------------------------------------------------------------------- #
class ChromaVectorStore:
    """Chroma-backed store with per-collection isolation."""

    def __init__(self, path: Path, default_collection: str, embedder: OllamaEmbedder) -> None:
        self.path = path
        self.default_collection = default_collection
        self.embedder = embedder
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialise ChromaDB: {exc}") from exc

    def _collection(self, collection: str | None = None):
        name = sanitize_collection_name(collection or self.default_collection)
        try:
            return self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to access collection '{name}': {exc}") from exc

    def add_chunks(self, chunks: list[Chunk], collection: str | None = None) -> int:
        if not chunks:
            return 0
        coll = self._collection(collection)
        existing = self._existing_ids(coll, [c.chunk_id for c in chunks])
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            return 0
        texts = [c.text for c in new_chunks]
        try:
            embeddings = self.embedder.embed(texts)
        except Exception as exc:
            raise VectorStoreError(f"Failed to embed chunks: {exc}") from exc
        try:
            coll.add(
                ids=[c.chunk_id for c in new_chunks],
                documents=texts,
                embeddings=embeddings,
                metadatas=[c.metadata for c in new_chunks],
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to store chunks: {exc}") from exc
        return len(new_chunks)

    def _existing_ids(self, coll, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        try:
            result = coll.get(ids=ids, include=[])
        except Exception:
            result = coll.get(ids=ids)
        return set(result.get("ids", []) or [])

    def query(self, query_text: str, top_k: int, collection: str | None = None) -> list[dict]:
        if not query_text.strip():
            return []
        coll = self._collection(collection)
        try:
            query_vec = self.embedder.embed_one(query_text)
        except Exception as exc:
            raise VectorStoreError(f"Failed to embed query: {exc}") from exc
        try:
            result = coll.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Query failed: {exc}") from exc

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        hits = []
        for doc, meta, dist, cid in zip(documents, metadatas, distances, ids):
            distance = float(dist) if dist is not None else 1.0
            similarity = max(0.0, 1.0 - distance)
            m = dict(meta or {})
            m.setdefault("chunk_id", cid)
            hits.append({
                "text": doc,
                "metadata": m,
                "distance": distance,
                "similarity": similarity,
                "chunk_id": cid,
            })
        return hits

    def count(self, collection: str | None = None) -> int:
        return int(self._collection(collection).count())

    def reset(self, collection: str | None = None) -> None:
        name = sanitize_collection_name(collection or self.default_collection)
        try:
            self._client.delete_collection(name)
            self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        except Exception as exc:
            raise VectorStoreError(f"Failed to reset collection: {exc}") from exc

    def distinct_sources(self, collection: str | None = None) -> set[str]:
        coll = self._collection(collection)
        try:
            result = coll.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Failed to list sources: {exc}") from exc
        return {m.get("source", "") for m in (result.get("metadatas") or []) if m and m.get("source")}

    def get_all_documents(self, collection: str | None = None) -> dict:
        coll = self._collection(collection)
        try:
            return coll.get(include=["documents", "metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Failed to fetch documents: {exc}") from exc

    def list_collections(self) -> list[str]:
        try:
            return [c.name for c in self._client.list_collections()]
        except Exception as exc:
            raise VectorStoreError(f"Failed to list collections: {exc}") from exc


# --------------------------------------------------------------------------- #
# Qdrant backend
# --------------------------------------------------------------------------- #
class QdrantVectorStore:
    """Qdrant-backed store (for horizontal scaling)."""

    def __init__(self, url: str, api_key: str, default_collection: str, embedder: OllamaEmbedder) -> None:
        self.url = url
        self.api_key = api_key
        self.default_collection = default_collection
        self.embedder = embedder
        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=url, api_key=api_key or None)
        except Exception as exc:
            raise VectorStoreError(f"Failed to connect to Qdrant: {exc}") from exc

    def _ensure_collection(self, collection: str, vector_size: int = 768):
        from qdrant_client import models
        try:
            self._client.get_collection(collection)
        except Exception:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def add_chunks(self, chunks: list[Chunk], collection: str | None = None) -> int:
        if not chunks:
            return 0
        from qdrant_client import models
        name = collection or self.default_collection
        texts = [c.text for c in chunks]
        try:
            embeddings = self.embedder.embed(texts)
        except Exception as exc:
            raise VectorStoreError(f"Failed to embed chunks: {exc}") from exc
        self._ensure_collection(name, vector_size=len(embeddings[0]))
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector=emb,
                payload={**c.metadata, "text": c.text},
            )
            for c, emb in zip(chunks, embeddings)
        ]
        try:
            self._client.upsert(collection_name=name, points=points)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc
        return len(points)

    def query(self, query_text: str, top_k: int, collection: str | None = None) -> list[dict]:
        name = collection or self.default_collection
        try:
            query_vec = self.embedder.embed_one(query_text)
            hits = self._client.search(collection_name=name, query_vector=query_vec, limit=top_k)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant query failed: {exc}") from exc
        results = []
        for h in hits:
            payload = dict(h.payload or {})
            text = payload.pop("text", "")
            results.append({
                "text": text,
                "metadata": payload,
                "distance": 1.0 - float(h.score),
                "similarity": float(h.score),
                "chunk_id": str(h.id),
            })
        return results

    def count(self, collection: str | None = None) -> int:
        name = collection or self.default_collection
        try:
            return int(self._client.count(collection_name=name).count)
        except Exception:
            return 0

    def reset(self, collection: str | None = None) -> None:
        name = collection or self.default_collection
        try:
            self._client.delete_collection(name)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant reset failed: {exc}") from exc

    def distinct_sources(self, collection: str | None = None) -> set[str]:
        return set()

    def get_all_documents(self, collection: str | None = None) -> dict:
        return {"ids": [], "documents": [], "metadatas": []}

    def list_collections(self) -> list[str]:
        try:
            return [c.name for c in self._client.get_collections().collections]
        except Exception:
            return []


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def create_vector_store(cfg) -> VectorStoreProtocol:
    """Factory: returns the configured vector store backend."""
    from src.embeddings import OllamaEmbedder
    embedder = OllamaEmbedder(model=cfg.embedding_model, base_url=cfg.ollama_base_url)
    if cfg.vector_backend == "qdrant":
        return QdrantVectorStore(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key,
            default_collection=cfg.qdrant_collection,
            embedder=embedder,
        )
    return ChromaVectorStore(
        path=cfg.chroma_path,
        default_collection=cfg.chroma_collection,
        embedder=embedder,
    )