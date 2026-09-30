"""Pydantic schemas for the FastAPI layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str | None = None
    top_k: int | None = None
    stream: bool = False
    session_id: str = "api"


class SourceItem(BaseModel):
    filename: str | None = None
    page: int | None = None
    similarity: float | None = None
    rrf_score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []
    collection: str


class IngestResponse(BaseModel):
    collection: str
    documents_processed: int
    chunks_created: int
    chunks_stored: int
    total_chunks: int


class StatsResponse(BaseModel):
    backend: str
    collection: str
    chunks: int
    documents: int
    embedding_model: str
    llm_model: str
    hybrid_enabled: bool
    reranker_enabled: bool
    ocr_enabled: bool