"""Configuration loaded from environment variables (v2)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return default if raw is None or raw.strip() == "" else int(raw)


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return default if raw is None or raw.strip() == "" else float(raw)


@dataclass(frozen=True)
class Config:
    # Ollama
    ollama_base_url: str
    llm_model: str
    embedding_model: str

    # Vector backend
    vector_backend: str
    chroma_path: Path
    chroma_collection: str
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str

    # Documents
    documents_path: Path
    default_collection: str

    # Chunking
    chunk_size: int
    chunk_overlap: int

    # Retrieval
    top_k: int
    rerank_top_k: int
    relevance_threshold: float
    hybrid_enabled: bool
    hybrid_dense_weight: float
    hybrid_sparse_weight: float
    rrf_k: int

    # Reranker
    reranker_enabled: bool
    reranker_model: str

    # OCR
    ocr_enabled: bool
    ocr_lang: str
    ocr_dpi: int
    ocr_min_chars: int

    # History
    history_dir: Path
    max_history_messages: int

    # API
    api_host: str
    api_port: int

    # Eval
    eval_dataset_path: Path

    debug: bool


def load_config() -> Config:
    cfg = Config(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        llm_model=os.getenv("LLM_MODEL", "qwen3:8b"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        vector_backend=os.getenv("VECTOR_BACKEND", "chroma").lower(),
        chroma_path=Path(os.getenv("CHROMA_PATH", "./chroma_db")).resolve(),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "knowledge_base"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "knowledge_base"),
        documents_path=Path(os.getenv("DOCUMENTS_PATH", "./documents")).resolve(),
        default_collection=os.getenv("DEFAULT_COLLECTION", "default"),
        chunk_size=_get_int("CHUNK_SIZE", 800),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
        top_k=_get_int("TOP_K", 20),
        rerank_top_k=_get_int("RERANK_TOP_K", 5),
        relevance_threshold=_get_float("RELEVANCE_THRESHOLD", 0.70),
        hybrid_enabled=_get_bool("HYBRID_ENABLED", True),
        hybrid_dense_weight=_get_float("HYBRID_DENSE_WEIGHT", 0.7),
        hybrid_sparse_weight=_get_float("HYBRID_SPARSE_WEIGHT", 0.3),
        rrf_k=_get_int("RRF_K", 60),
        reranker_enabled=_get_bool("RERANKER_ENABLED", True),
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        ocr_enabled=_get_bool("OCR_ENABLED", True),
        ocr_lang=os.getenv("OCR_LANG", "eng+fra"),
        ocr_dpi=_get_int("OCR_DPI", 300),
        ocr_min_chars=_get_int("OCR_MIN_CHARS", 30),
        history_dir=Path(os.getenv("HISTORY_DIR", "./history")).resolve(),
        max_history_messages=_get_int("MAX_HISTORY_MESSAGES", 6),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=_get_int("API_PORT", 8000),
        eval_dataset_path=Path(os.getenv("EVAL_DATASET_PATH", "./eval/golden_dataset.json")).resolve(),
        debug=_get_bool("DEBUG", False),
    )

    # Validation
    if cfg.vector_backend not in {"chroma", "qdrant"}:
        raise ValueError("VECTOR_BACKEND must be 'chroma' or 'qdrant'")
    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    if not 0.0 <= cfg.relevance_threshold <= 1.0:
        raise ValueError("RELEVANCE_THRESHOLD must be between 0.0 and 1.0")
    return cfg