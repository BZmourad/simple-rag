"""FastAPI wrapper v4: JWT auth, rate limiting, streaming, graph augmentation."""

from __future__ import annotations

import json
import logging

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from src.api.auth import AuthManager
from src.api.routes_auth import create_auth_router
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
from src.llm import LLMError, create_llm
from src.query_expander import QueryExpander
from src.rag import RAGPipeline
from src.reranker import get_reranker
from src.vector_store import VectorStoreError, create_vector_store


logger = logging.getLogger(__name__)

def _build_pipeline(cfg: Config, collection: str, session_id: str = "api", with_graph: bool = True):
    """Build a fresh RAG pipeline for one request.

    Components are cheap except the reranker, which is a process-wide singleton
    via get_reranker() so the cross-encoder model is loaded once.
    """
    store = create_vector_store(cfg)

    retriever = HybridRetriever(
        vector_store=store,
        top_k=cfg.top_k,
        dense_weight=cfg.hybrid_dense_weight,
        sparse_weight=cfg.hybrid_sparse_weight,
        rrf_k=cfg.rrf_k,
        collection=collection,
    )
    reranker = get_reranker(cfg.reranker_model) if cfg.reranker_enabled else None
    llm = create_llm(cfg)

    expander = None
    if cfg.query_expansion_enabled:
        expander = QueryExpander(
            llm=llm,
            mode=cfg.query_expansion_mode,
            num_queries=cfg.query_expansion_num_queries,
            hyde_temp=cfg.query_expansion_hyde_temp,
            redis_url=cfg.rate_limit_redis_url,
            cache_ttl=3600,
        )

    graph_store = None
    if with_graph and cfg.graph_rag_enabled:
        try:
            from src.graph_store_falkor import FalkorGraphStore
            graph_store = FalkorGraphStore(path=cfg.falkordb_path)
        except Exception as exc:
            logger.warning("Graph store unavailable: %s", exc)

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
        query_expander=expander,
        knowledge_graph=graph_store,
        graph_max_depth=cfg.graph_max_depth,
        debug=cfg.debug,
    )
    return store, retriever, pipeline


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="Simple RAG API", version="7.0.0")

    auth = AuthManager(
        mode=cfg.auth_mode,
        valid_keys=cfg.api_keys,
        jwt_secret=cfg.jwt_secret,
        jwt_algorithm=cfg.jwt_algorithm,
        access_expire_minutes=cfg.jwt_access_expire_minutes,
        refresh_expire_days=cfg.jwt_refresh_expire_days,
        rate_limit_enabled=cfg.rate_limit_enabled,
        redis_url=cfg.rate_limit_redis_url,
        per_minute=cfg.rate_limit_per_minute,
        per_day=cfg.rate_limit_per_day,
    )

    # Auth endpoints (login/refresh) are always mounted.
    app.include_router(create_auth_router(auth))

    # ------------------------------------------------------------------ #
    # Health (no auth)
    # ------------------------------------------------------------------ #
    @app.get("/health")
    def health():
        return {"status": "ok", "version": "7.0.0", "auth_mode": cfg.auth_mode, "llm_provider": cfg.llm_provider}

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    @app.post("/ingest", response_model=IngestResponse)
    def ingest(collection: str | None = None, user=Depends(auth.require_auth)):
        coll = collection or cfg.default_collection
        try:
            store, retriever, _ = _build_pipeline(cfg, coll, with_graph=False)
        except (EmbeddingError, VectorStoreError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Optional graph population
        graph_extractor = None
        graph_store = None
        if cfg.graph_rag_enabled:
            try:
                from src.graph_rag import GraphRAGExtractor
                from src.graph_store_falkor import FalkorGraphStore
                graph_store = FalkorGraphStore(path=cfg.falkordb_path)
                graph_extractor = GraphRAGExtractor(
                    method=cfg.graph_extraction_method,
                    threshold=cfg.graph_entity_threshold,
                )
            except Exception as exc:
                logger.warning("Graph store unavailable for ingest: %s", exc)

        total_docs = total_created = total_stored = 0
        try:
            for doc in load_documents(cfg.documents_path, cfg=cfg):
                total_docs += 1
                chunks = chunk_document(doc, cfg.chunk_size, cfg.chunk_overlap)
                total_created += len(chunks)
                total_stored += store.add_chunks(chunks, collection=coll)

                if graph_extractor is not None and graph_store is not None:
                    for chunk in chunks:
                        try:
                            entities, relations = graph_extractor.extract(chunk.text)
                            for e in entities:
                                graph_store.add_entity(e.name, e.entity_type)
                            for r in relations:
                                graph_store.add_relation(r.source, r.relation, r.target)
                        except Exception as exc:
                            logger.warning("Graph extraction failed: %s", exc)
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

    # ------------------------------------------------------------------ #
    # Ask (non-streaming)
    # ------------------------------------------------------------------ #
    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest, user=Depends(auth.require_auth)):
        coll = req.collection or cfg.default_collection
        _, _, pipeline = await run_in_threadpool(_build_pipeline, cfg, coll, req.session_id)
        try:
            result = await pipeline.aanswer(req.question)
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        sources = [
            SourceItem(
                filename=c.metadata.get("filename"),
                page=c.metadata.get("page"),
                similarity=getattr(c, "similarity", None),
                rrf_score=getattr(c, "rrf_score", None),
            )
            for c in result.sources
        ]
        return AskResponse(answer=result.answer, sources=sources, collection=coll)

    # ------------------------------------------------------------------ #
    # Ask (streaming SSE)
    # ------------------------------------------------------------------ #
    @app.post("/ask/stream")
    async def ask_stream(req: AskRequest, user=Depends(auth.require_auth)):
        coll = req.collection or cfg.default_collection
        _, _, pipeline = await run_in_threadpool(_build_pipeline, cfg, coll, req.session_id)

        async def event_generator():
            try:
                async for event in pipeline.astream_answer(req.question):
                    if event["type"] == "token":
                        yield f"event: token\ndata: {json.dumps({'token': event['content']})}\n\n"
                    elif event["type"] == "done":
                        sources = [{"filename": c.metadata.get("filename"),
                                    "page": c.metadata.get("page"),
                                    "similarity": getattr(c, "similarity", None)}
                                for c in event.get("sources", [])]
                        yield f"event: done\ndata: {json.dumps({'sources': sources})}\n\n"
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    @app.get("/stats", response_model=StatsResponse)
    def stats(collection: str | None = None, user=Depends(auth.require_auth)):
        coll = collection or cfg.default_collection
        try:
            store, _, _ = _build_pipeline(cfg, coll, with_graph=False)
        except (EmbeddingError, VectorStoreError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        try:
            sources = store.distinct_sources(collection=coll)
            chunk_count = store.count(collection=coll)
        except VectorStoreError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return StatsResponse(
            backend=cfg.vector_backend,
            collection=coll,
            chunks=chunk_count,
            documents=len(sources),
            embedding_model=cfg.embedding_model,
            llm_model=cfg.llm_model,
            hybrid_enabled=cfg.hybrid_enabled,
            reranker_enabled=cfg.reranker_enabled,
            ocr_enabled=cfg.ocr_enabled,
        )

    # ------------------------------------------------------------------ #
    # Clear (admin)
    # ------------------------------------------------------------------ #
    @app.post("/clear")
    def clear(collection: str | None = None, user=Depends(auth.require_auth)):
        coll = collection or cfg.default_collection
        if "ingest" not in user.get("scopes", []):
            raise HTTPException(status_code=403, detail="Missing 'ingest' scope")
        try:
            store, _, _ = _build_pipeline(cfg, coll, with_graph=False)
            store.reset(collection=coll)
        except VectorStoreError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "cleared", "collection": coll}

    # ------------------------------------------------------------------ #
    # Graph stats (admin)
    # ------------------------------------------------------------------ #
    @app.get("/graph/stats")
    def graph_stats(user=Depends(auth.require_auth)):
        if not cfg.graph_rag_enabled:
            raise HTTPException(status_code=404, detail="Graph RAG is disabled")
        try:
            from src.graph_store_falkor import FalkorGraphStore
            store = FalkorGraphStore(path=cfg.falkordb_path)
            return {"backend": cfg.graph_backend, **store.stats()}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # --- Mount static UI ---
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import RedirectResponse

    WEB_DIR = Path(__file__).resolve().parents[2] / "web"
    if WEB_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")

        @app.get("/")
        async def root():
            return RedirectResponse(url="/ui/login.html")

    # --- Register SSE ingest stream ---
    from src.api.ingest_stream import register_ingest_stream
    register_ingest_stream(app, cfg, auth)

    # Settings
    from src.settings_store import SettingsStore
    from src.api.settings import register_settings_routes
    from src.api.files import register_file_routes

    settings_store = SettingsStore(cfg.env_path) 
    register_settings_routes(app, settings_store, auth)
    register_file_routes(app, cfg, auth)

    return app

def run_server(cfg: Config) -> None:
    import uvicorn
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.uvicorn_host, port=cfg.uvicorn_port)