"""SSE endpoint that streams ingestion progress."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.chunker import chunk_document
from src.config import Config
from src.document_loader import discover_documents, load_document
from src.hybrid_retriever import HybridRetriever
from src.vector_store import VectorStoreError, create_vector_store

logger = logging.getLogger(__name__)


def register_ingest_stream(app: FastAPI, cfg: Config, auth) -> None:
    @app.post("/ingest/stream")
    async def ingest_stream(collection: str | None = None, user=Depends(auth.require_auth)):
        """Stream ingestion progress as Server-Sent Events."""
        coll = collection or cfg.default_collection

        async def event_generator():
            try:
                # 1. Discover files
                files = discover_documents(cfg.documents_path)
                total = len(files)
                yield _sse("start", {"total": total, "collection": coll})

                if total == 0:
                    yield _sse("done", {
                        "documents_processed": 0, "chunks_created": 0,
                        "chunks_stored": 0, "total_chunks": 0,
                    })
                    return

                # 2. Build store in a thread
                loop = asyncio.get_running_loop()
                store = await loop.run_in_executor(None, create_vector_store, cfg)
                retriever = HybridRetriever(
                    vector_store=store, top_k=cfg.top_k,
                    dense_weight=cfg.hybrid_dense_weight,
                    sparse_weight=cfg.hybrid_sparse_weight,
                    rrf_k=cfg.rrf_k, collection=coll,
                )

                total_docs = total_created = total_stored = 0

                for idx, path in enumerate(files, start=1):
                    yield _sse("file", {
                        "index": idx, "total": total,
                        "filename": path.name,
                        "stage": "loading",
                    })

                    doc = await loop.run_in_executor(
                        None, load_document, path, cfg.documents_path, cfg
                    )
                    if doc is None:
                        yield _sse("file", {
                            "index": idx, "total": total,
                            "filename": path.name,
                            "stage": "skipped",
                        })
                        continue

                    yield _sse("file", {
                        "index": idx, "total": total,
                        "filename": path.name,
                        "stage": "chunking",
                    })

                    chunks = await loop.run_in_executor(
                        None, chunk_document, doc, cfg.chunk_size, cfg.chunk_overlap
                    )
                    total_docs += 1
                    total_created += len(chunks)

                    yield _sse("file", {
                        "index": idx, "total": total,
                        "filename": path.name,
                        "stage": "embedding",
                        "chunks": len(chunks),
                    })

                    try:
                        stored = await loop.run_in_executor(
                            None, store.add_chunks, chunks, coll
                        )
                        total_stored += stored
                    except VectorStoreError as exc:
                        logger.error("Store failed for %s: %s", path.name, exc)
                        yield _sse("file", {
                            "index": idx, "total": total,
                            "filename": path.name,
                            "stage": "error",
                            "error": str(exc),
                        })
                        continue

                    yield _sse("file", {
                        "index": idx, "total": total,
                        "filename": path.name,
                        "stage": "done",
                        "stored": stored,
                    })

                retriever.invalidate_bm25()
                total_chunks = await loop.run_in_executor(
                    None, store.count, coll
                )
                yield _sse("done", {
                    "documents_processed": total_docs,
                    "chunks_created": total_created,
                    "chunks_stored": total_stored,
                    "total_chunks": total_chunks,
                })
            except Exception as exc:
                logger.exception("Ingest stream failed")
                yield _sse("error", {"error": str(exc)})

        return StreamingResponse(event_generator(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"