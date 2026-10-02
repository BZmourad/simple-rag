"""Celery tasks for document ingestion."""

from __future__ import annotations

import logging

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="ingest.run")
def run_ingest(self, collection: str | None = None) -> dict:
    """Run the full ingestion pipeline asynchronously."""
    from src.chunker import chunk_document
    from src.config import load_config
    from src.document_loader import load_documents
    from src.hybrid_retriever import HybridRetriever
    from src.vector_store import create_vector_store

    cfg = load_config()
    coll = collection or cfg.default_collection

    store = create_vector_store(cfg)
    retriever = HybridRetriever(
        vector_store=store,
        top_k=cfg.top_k,
        dense_weight=cfg.hybrid_dense_weight,
        sparse_weight=cfg.hybrid_sparse_weight,
        rrf_k=cfg.rrf_k,
        collection=coll,
    )

    total_docs = total_created = total_stored = 0
    for doc in load_documents(cfg.documents_path, cfg=cfg):
        total_docs += 1
        chunks = chunk_document(doc, cfg.chunk_size, cfg.chunk_overlap)
        total_created += len(chunks)
        try:
            stored = store.add_chunks(chunks, collection=coll)
            total_stored += stored
        except Exception as exc:
            logger.error("Failed to store chunks for %s: %s", doc.filename, exc)

    retriever.invalidate_bm25()
    return {
        "collection": coll,
        "documents_processed": total_docs,
        "chunks_created": total_created,
        "chunks_stored": total_stored,
        "total_chunks": store.count(collection=coll),
    }