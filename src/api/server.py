"""FastAPI wrapper exposing /ingest, /ask, /stats, /ask/stream."""

from __future__ import annotations

import json
import logging

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    AskRequest,
    AskResponse,
    IngestResponse,
    SourceItem,
    StatsResponse,
)
from src.chunker import chunk_document
from src.config import Config
from src.document_loader import load_documents
from src.embeddings import EmbeddingError
from src.history_store import HistoryStore
from src.hybrid_retriever import HybridRetriever
from src.llm import LLMError, OllamaLLM
from src.rag import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.vector_store import VectorStoreError, create_vector_store
from src.api.auth import AuthManager

logger = logging.getLogger(__name__)


def _build_pipeline(cfg: Config, collection: str, session_id: str = "api"):
    store = create_vector_store(cfg)
    retriever = HybridRetriever(
        vector_store=store,
        top_k=cfg.top_k,
        dense_weight=cfg.hybrid_dense_weight,
        sparse_weight=cfg.hybrid_sparse_weight,
        rrf_k=cfg.rrf_k,
        collection=collection,
    )
    reranker = CrossEncoderReranker(cfg.reranker_model) if cfg.reranker_enabled else None
    llm = OllamaLLM(model=cfg.llm_model, base_url=cfg.ollama_base_url)
    history = HistoryStore(cfg.history_dir, session_id=session_id)
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        reranker=reranker,
        max_history_messages=cfg.max_history_messages,
        rerank_top_k=cfg.rerank_top_k,
        relevance_threshold=cfg.relevance_threshold,
        history_store=history,
        collection=collection,
        debug=cfg.debug,
    )
    return store, retriever, pipeline


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="Simple RAG API", version="3.0.0")
    auth = AuthManager(
        enabled=cfg.auth_enabled,
        valid_keys=cfg.api_keys,
        rate_limit_enabled=cfg.rate_limit_enabled,
        redis_url=cfg.rate_limit_redis_url,
        per_minute=cfg.rate_limit_per_minute,
        per_day=cfg.rate_limit_per_day,
    )
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(collection: str | None = None, user=Depends(auth.require_auth)):
        coll = collection or cfg.default_collection
        try:
            store, retriever, _ = _build_pipeline(cfg, coll)
        except (EmbeddingError, VectorStoreError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        total_docs = total_created = total_stored = 0
        try:
            for doc in load_documents(cfg.documents_path, cfg=cfg):
                total_docs += 1
                chunks = chunk_document(doc, cfg.chunk_size, cfg.chunk_overlap)
                total_created += len(chunks)
                total_stored += store.add_chunks(chunks, collection=coll)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

        retriever.invalidate_bm25()
        return IngestResponse(
            collection=coll,
            documents_processed=total_docs,
            chunks_created=total_created,
            chunks_stored=total_stored,
            total_chunks=store.count(collection=coll),
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, user=Depends(auth.require_auth)):
        coll = req.collection or cfg.default_collection
        try:
            _, _, pipeline = _build_pipeline(cfg, coll, session_id=req.session_id)
            result = pipeline.answer(req.question)
        except (EmbeddingError, VectorStoreError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        sources = [
            SourceItem(
                filename=c.metadata.get("filename"),
                page=c.metadata.get("page"),
                similarity=c.similarity,
                rrf_score=c.rrf_score,
            )
            for c in result.sources
        ]
        return AskResponse(answer=result.answer, sources=sources, collection=coll)

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest):
        coll = req.collection or cfg.default_collection

        def event_generator():
            try:
                _, _, pipeline = _build_pipeline(cfg, coll, session_id=req.session_id)
                for event in pipeline.stream_answer(req.question):
                    if event["type"] == "token":
                        yield f"event: token\ndata: {json.dumps({'token': event['content']})}\n\n"
                    elif event["type"] == "done":
                        sources = [
                            {
                                "filename": c.metadata.get("filename"),
                                "page": c.metadata.get("page"),
                                "similarity": c.similarity,
                            }
                            for c in event.get("sources", [])
                        ]
                        yield f"event: done\ndata: {json.dumps({'sources': sources})}\n\n"
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/stats", response_model=StatsResponse)
    def stats(collection: str | None = None):
        coll = collection or cfg.default_collection
        try:
            store, _, _ = _build_pipeline(cfg, coll)
        except (EmbeddingError, VectorStoreError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        sources = store.distinct_sources(collection=coll)
        return StatsResponse(
            backend=cfg.vector_backend,
            collection=coll,
            chunks=store.count(collection=coll),
            documents=len(sources),
            embedding_model=cfg.embedding_model,
            llm_model=cfg.llm_model,
            hybrid_enabled=cfg.hybrid_enabled,
            reranker_enabled=cfg.reranker_enabled,
            ocr_enabled=cfg.ocr_enabled,
        )

    return app


def run_server(cfg: Config) -> None:
    import uvicorn
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.api_host, port=cfg.api_port)