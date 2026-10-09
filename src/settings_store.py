"""Runtime settings: overrides .env values and persists back to .env."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# Keys that can be changed from the settings page.
MUTABLE_KEYS = {
    "LLM_PROVIDER",
    "LLAMACPP_BASE_URL",
    "LLAMACPP_MODEL",
    "LLAMACPP_TIMEOUT",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_BASE_URL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "OLLAMA_BASE_URL",
    "TOP_K",
    "RERANK_TOP_K",
    "RELEVANCE_THRESHOLD",
    "HYBRID_ENABLED",
    "RERANKER_ENABLED",
    "QUERY_EXPANSION_ENABLED",
    "QUERY_EXPANSION_MODE",
    "MAX_HISTORY_MESSAGES",
    "RATE_LIMIT_PER_MINUTE",
    "RATE_LIMIT_PER_DAY",
    "DEBUG",
}

# Keys whose values must be masked in API responses.
SECRET_KEYS = {"OPENROUTER_API_KEY", "GEMINI_API_KEY", "JWT_SECRET", "ADMIN_PASSWORD_HASH"}


class SettingsStore:
    """Thread-safe store for runtime overrides.

    Values set at runtime take precedence over .env and are persisted to a
    sidecar file (.env.runtime) so they survive restarts without rewriting
    the user's .env.
    """

    def __init__(self, env_path: Path) -> None:
        self.env_path = env_path
        self.runtime_path = env_path.parent / ".env.runtime"
        self._lock = Lock()
        self._overrides: dict[str, str] = {}
        self._load_runtime()

    def _load_runtime(self) -> None:
        if not self.runtime_path.exists():
            return
        try:
            for line in self.runtime_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k in MUTABLE_KEYS:
                    self._overrides[k] = v
                    os.environ[k] = v
            logger.info("Loaded %d runtime overrides from %s", len(self._overrides), self.runtime_path)
        except Exception as exc:
            logger.warning("Failed to load runtime overrides: %s", exc)

    def _save_runtime(self) -> None:
        try:
            lines = [f"{k}={v}" for k, v in sorted(self._overrides.items())]
            self.runtime_path.write_text("\n".join(lines) + "\n")
            # Restrict permissions — the file may contain API keys
            self.runtime_path.chmod(0o600)
        except Exception as exc:
            logger.error("Failed to save runtime overrides: %s", exc)

    def get(self, key: str, default: str = "") -> str:
        with self._lock:
            if key in self._overrides:
                return self._overrides[key]
            return os.getenv(key, default)

    def set_many(self, values: dict[str, Any]) -> dict[str, str]:
        """Set multiple keys. Returns the applied key→value map (masked values excluded)."""
        applied: dict[str, str] = {}
        with self._lock:
            for k, v in values.items():
                if k not in MUTABLE_KEYS:
                    continue
                if v is None:
                    continue
                sval = str(v)
                self._overrides[k] = sval
                os.environ[k] = sval
                applied[k] = sval
            self._save_runtime()
        return applied

    def all_mutable(self) -> dict[str, str]:
        """Return all mutable keys with their current values (secrets masked)."""
        out: dict[str, str] = {}
        for k in sorted(MUTABLE_KEYS):
            raw = self.get(k, "")
            if k in SECRET_KEYS and raw:
                out[k] = "****" + raw[-4:] if len(raw) > 4 else "****"
            else:
                out[k] = raw
        return out

    def clear(self) -> None:
        with self._lock:
            for k in self._overrides:
                os.environ.pop(k, None)
            self._overrides.clear()
            if self.runtime_path.exists():
                self.runtime_path.unlink()