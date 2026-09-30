"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be an integer, got {raw!r}") from exc


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be a float, got {raw!r}") from exc


@dataclass(frozen=True)
class Config:
    """Application configuration."""

    ollama_base_url: str
    llm_model: str
    embedding_model: str

    chroma_path: Path
    chroma_collection: str

    documents_path: Path

    chunk_size: int
    chunk_overlap: int

    top_k: int
    relevance_threshold: float

    max_history_messages: int

    debug: bool


def load_config() -> Config:
    """Load configuration from environment variables."""

    chunk_size = _get_int("CHUNK_SIZE", 800)
    chunk_overlap = _get_int("CHUNK_OVERLAP", 150)

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("CHUNK_OVERLAP must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    top_k = _get_int("TOP_K", 5)
    if top_k <= 0:
        raise ValueError("TOP_K must be greater than 0")

    threshold = _get_float("RELEVANCE_THRESHOLD", 0.70)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("RELEVANCE_THRESHOLD must be between 0.0 and 1.0")

    return Config(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        llm_model=os.getenv("LLM_MODEL", "qwen3:8b"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        chroma_path=Path(os.getenv("CHROMA_PATH", "./chroma_db")).resolve(),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "knowledge_base"),
        documents_path=Path(os.getenv("DOCUMENTS_PATH", "./documents")).resolve(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        relevance_threshold=threshold,
        max_history_messages=_get_int("MAX_HISTORY_MESSAGES", 6),
        debug=_get_bool("DEBUG", False),
    )