"""Persistent graph store backed by FalkorDBLite (embedded)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FalkorGraphStore:
    """Cypher-capable persistent graph store using FalkorDBLite.

    The database file is a single file on disk; no server to manage.
    """

    def __init__(self, path: Path, graph_name: str = "knowledge") -> None:
        self.path = path
        self.graph_name = graph_name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from redislite.falkordb_client import FalkorDB
            self._db = FalkorDB(str(self.path))
            self._graph = self._db.select_graph(self.graph_name)
            logger.info("FalkorDB opened at %s (graph=%s)", self.path, self.graph_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open FalkorDB at {self.path}. "
                f"Install with: pip install falkordblite. Error: {exc}"
            ) from exc

    def add_entity(self, name: str, entity_type: str = "Unknown", **props: Any) -> None:
        """Upsert an entity node."""
        try:
            self._graph.query(
                "MERGE (e:Entity {name: $name}) "
                "SET e.type = $type, e += $props",
                {"name": name, "type": entity_type, "props": props},
            )
        except Exception as exc:
            logger.error("Failed to add entity %s: %s", name, exc)

    def add_relation(self, source: str, relation: str, target: str, **props: Any) -> None:
        """Create a relationship between two entities."""
        rel_type = relation.upper().replace(" ", "_").replace("-", "_")
        try:
            self._graph.query(
                "MERGE (a:Entity {name: $src}) "
                "MERGE (b:Entity {name: $tgt}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                "SET r += $props",
                {"src": source, "tgt": target, "props": props},
            )
        except Exception as exc:
            logger.error("Failed to add relation %s->%s: %s", source, target, exc)

    def find_entities(self, text: str) -> list[str]:
        """Return entity names that appear as substrings in the given text."""
        try:
            result = self._graph.query("MATCH (e:Entity) RETURN e.name")
            names = [row[0] for row in result.result_set]
        except Exception as exc:
            logger.error("find_entities failed: %s", exc)
            return []
        lowered = text.lower()
        return [n for n in names if n and n.lower() in lowered]

    def traverse(self, entities: list[str], max_depth: int = 2) -> list[tuple[str, str, str]]:
        """Return (source, relation, target) triples reachable from the given entities."""
        if not entities:
            return []
        try:
            result = self._graph.query(
                f"MATCH (a:Entity)-[r*1..{max_depth}]->(b:Entity) "
                "WHERE a.name IN $names "
                "UNWIND r AS rel "
                "RETURN startNode(rel).name, type(rel), endNode(rel).name",
                {"names": entities},
            )
            return [(row[0], row[1], row[2]) for row in result.result_set]
        except Exception as exc:
            logger.error("traverse failed: %s", exc)
            return []

    def stats(self) -> dict:
        try:
            nodes = self._graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
            edges = self._graph.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
            return {"nodes": nodes, "edges": edges}
        except Exception:
            return {"nodes": 0, "edges": 0}

    def clear(self) -> None:
        try:
            self._graph.delete()
            self._graph = self._db.select_graph(self.graph_name)
        except Exception as exc:
            logger.error("clear failed: %s", exc)