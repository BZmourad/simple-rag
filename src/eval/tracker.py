"""SQLite-backed RAGAS regression tracker."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    git_commit TEXT,
    collection TEXT NOT NULL,
    num_items INTEGER NOT NULL,
    faithfulness REAL,
    context_precision REAL,
    context_recall REAL,
    metadata TEXT
);
"""


class RagasTracker:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def record(
        self,
        collection: str,
        num_items: int,
        faithfulness: float,
        context_precision: float,
        context_recall: float,
        git_commit: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs (run_at, git_commit, collection, num_items, "
            "faithfulness, context_precision, context_recall, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                git_commit,
                collection,
                num_items,
                faithfulness,
                context_precision,
                context_recall,
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def history(self, collection: str, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, run_at, git_commit, num_items, faithfulness, "
            "context_precision, context_recall FROM runs "
            "WHERE collection = ? ORDER BY id DESC LIMIT ?",
            (collection, limit),
        ).fetchall()
        return [
            {
                "id": r[0], "run_at": r[1], "git_commit": r[2], "num_items": r[3],
                "faithfulness": r[4], "context_precision": r[5], "context_recall": r[6],
            }
            for r in rows
        ]

    def compare(self, collection: str) -> dict:
        """Compare the latest run to the previous one."""
        hist = self.history(collection, limit=2)
        if len(hist) < 2:
            return {"status": "insufficient_history", "runs": len(hist)}
        latest, previous = hist[0], hist[1]
        deltas = {
            "faithfulness": (latest["faithfulness"] or 0) - (previous["faithfulness"] or 0),
            "context_precision": (latest["context_precision"] or 0) - (previous["context_precision"] or 0),
            "context_recall": (latest["context_recall"] or 0) - (previous["context_recall"] or 0),
        }
        regressions = {k: v for k, v in deltas.items() if v < -0.05}
        return {
            "latest_run": latest,
            "previous_run": previous,
            "deltas": deltas,
            "regressions": regressions,
            "status": "regression_detected" if regressions else "ok",
        }

    def close(self) -> None:
        self._conn.close()