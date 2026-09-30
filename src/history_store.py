"""Persist conversation history to disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HistoryStore:
    """JSON-file-backed conversation history."""

    def __init__(self, directory: Path, session_id: str = "default") -> None:
        self.directory = directory
        self.session_id = session_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{session_id}.json"
        self.messages: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data.get("messages", [])
        except Exception as exc:
            logger.warning("Could not load history from %s: %s", self.path, exc)
            return []

    def _save(self) -> None:
        try:
            payload = {"session_id": self.session_id, "messages": self.messages}
            self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.error("Could not save history: %s", exc)

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._save()

    def get_recent(self, n: int) -> list[dict]:
        return self.messages[-n:] if n > 0 else []

    def clear(self) -> None:
        self.messages = []
        self._save()