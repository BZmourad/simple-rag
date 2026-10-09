"""Configuration loaded from environment variables (v2)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

def _clean_env_value(raw: str) -> str:
    """Strip whitespace and any inline '# comment' from a value."""
    if raw is None:
        return ""
    import re
    stripped = re.split(r"\s+#", raw, maxsplit=1)[0]
    return stripped.strip()


def _get_bool(key: str, default: bool = False) -> bool:
    raw = _clean_env_value(os.getenv(key, ""))
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _get_int(key: str, default: int) -> int:
    raw = _clean_env_value(os.getenv(key, ""))
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {key} must be an integer, got {raw!r}. "
            f"Hint: inline comments in .env are not supported."
        ) from exc


def _get_float(key: str, default: float) -> float:
    raw = _clean_env_value(os.getenv(key, ""))
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {key} must be a float, got {raw!r}. "
            f"Hint: inline comments in .env are not supported."
        ) from exc


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

    # RAGAS
    ragas_enabled: bool
    ragas_judge_model: str
    ragas_judge_embedding: str
    ragas_dataset_path: Path

    # Celery
    celery_broker_url: str
    celery_result_backend: str
    celery_task_time_limit: int
    celery_task_soft_time_limit: int

    # Query expansion
    query_expansion_enabled: bool
    query_expansion_mode: str
    query_expansion_num_queries: int
    query_expansion_hyde_temp: float

    # Graph RAG
    graph_rag_enabled: bool
    graph_extraction_method: str
    graph_entity_threshold: float
    graph_max_depth: int

    # Auth
    auth_enabled: bool
    api_keys: list[str]
    jwt_secret: str
    jwt_algorithm: str
    jwt_expire_minutes: int

    # Rate limiting
    rate_limit_enabled: bool
    rate_limit_redis_url: str
    rate_limit_per_minute: int
    rate_limit_per_day: int

    # Reranker cache
    reranker_cache_size: int

    # Auth
    auth_mode: str
    jwt_access_expire_minutes: int
    jwt_refresh_expire_days: int

    # Graph
    graph_backend: str
    falkordb_path: Path

    # RAGAS tracking
    ragas_tracking_enabled: bool
    ragas_db_path: Path

    # Celery beat
    celery_beat_enabled: bool
    celery_ingest_schedule_minutes: int

    # Flower
    flower_port: int
    flower_basic_auth: str

    # LLM Provider
    llm_provider: str
    llamacpp_base_url: str
    llamacpp_model: str
    llamacpp_timeout: float
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    gemini_api_key: str
    gemini_model: str
    embedding_provider: str
    workers: int
    uvicorn_host: str
    uvicorn_port: int

    env_path: Path


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
        ragas_enabled=_get_bool("RAGAS_ENABLED", True),
        ragas_judge_model=os.getenv("RAGAS_JUDGE_MODEL", os.getenv("LLM_MODEL", "qwen3:8b")),
        ragas_judge_embedding=os.getenv("RAGAS_JUDGE_EMBEDDING", os.getenv("EMBEDDING_MODEL", "nomic-embed-text")),
        ragas_dataset_path=Path(os.getenv("RAGAS_DATASET_PATH", "./eval/ragas_dataset.json")).resolve(),
        celery_broker_url=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
        celery_result_backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),
        celery_task_time_limit=_get_int("CELERY_TASK_TIME_LIMIT", 3600),
        celery_task_soft_time_limit=_get_int("CELERY_TASK_SOFT_TIME_LIMIT", 3300),
        query_expansion_enabled=_get_bool("QUERY_EXPANSION_ENABLED", True),
        query_expansion_mode=os.getenv("QUERY_EXPANSION_MODE", "hyde"),
        query_expansion_num_queries=_get_int("QUERY_EXPANSION_NUM_QUERIES", 3),
        query_expansion_hyde_temp=_get_float("QUERY_EXPANSION_HYDE_TEMP", 0.7),
        graph_rag_enabled=_get_bool("GRAPH_RAG_ENABLED", False),
        graph_extraction_method=os.getenv("GRAPH_EXTRACTION_METHOD", "gliner"),
        graph_entity_threshold=_get_float("GRAPH_ENTITY_THRESHOLD", 0.5),
        graph_max_depth=_get_int("GRAPH_MAX_DEPTH", 2),
        auth_enabled=_get_bool("AUTH_ENABLED", True),
        api_keys=[k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()],
        jwt_secret=os.getenv("JWT_SECRET", "change-me"),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        jwt_expire_minutes=_get_int("JWT_EXPIRE_MINUTES", 60),
        rate_limit_enabled=_get_bool("RATE_LIMIT_ENABLED", True),
        rate_limit_redis_url=os.getenv("RATE_LIMIT_REDIS_URL", "redis://localhost:6379/0"),
        rate_limit_per_minute=_get_int("RATE_LIMIT_PER_MINUTE", 30),
        rate_limit_per_day=_get_int("RATE_LIMIT_PER_DAY", 500),
        reranker_cache_size=_get_int("RERANKER_CACHE_SIZE", 1),
        #falkordb_path=Path(os.getenv("FALKORDB_PATH", "./data/falkordb.db")).resolve(),
        auth_mode=os.getenv("AUTH_MODE", "jwt").lower(),
        jwt_access_expire_minutes=_get_int("JWT_ACCESS_EXPIRE_MINUTES", 30),
        jwt_refresh_expire_days=_get_int("JWT_REFRESH_EXPIRE_DAYS", 7),
        graph_backend=os.getenv("GRAPH_BACKEND", "falkordb").lower(),
        falkordb_path=Path(os.getenv("FALKORDB_PATH", "./data/falkordb.db")).resolve(),
        ragas_tracking_enabled=_get_bool("RAGAS_TRACKING_ENABLED", True),
        ragas_db_path=Path(os.getenv("RAGAS_DB_PATH", "./data/ragas_history.db")).resolve(),
        celery_beat_enabled=_get_bool("CELERY_BEAT_ENABLED", True),
        celery_ingest_schedule_minutes=_get_int("CELERY_INGEST_SCHEDULE_MINUTES", 60),
        flower_port=_get_int("FLOWER_PORT", 5555),
        flower_basic_auth=os.getenv("FLOWER_BASIC_AUTH", "monitoring:change-me"),
        llm_provider=os.getenv("LLM_PROVIDER", "llamacpp").lower(),
        llamacpp_base_url=os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8080"),
        llamacpp_model=os.getenv("LLAMACPP_MODEL", "local"),
        llamacpp_timeout=_get_float("LLAMACPP_TIMEOUT", 180.0),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama").lower(),
        workers=_get_int("WORKERS", 4),
        uvicorn_host=os.getenv("UVICORN_HOST", "127.0.0.1"),
        uvicorn_port=_get_int("UVICORN_PORT", 8000),
        env_path=Path(os.getenv("ENV_PATH", "/opt/simple-rag/.env")).resolve(),
    )

    # Validation
    if cfg.vector_backend not in {"chroma", "qdrant"}:
        raise ValueError("VECTOR_BACKEND must be 'chroma' or 'qdrant'")
    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    if not 0.0 <= cfg.relevance_threshold <= 1.0:
        raise ValueError("RELEVANCE_THRESHOLD must be between 0.0 and 1.0")
    if cfg.auth_enabled and cfg.auth_mode == "apikey" and not cfg.api_keys:
        raise ValueError("API_KEY mode requires at least one key in API_KEYS")
    return cfg