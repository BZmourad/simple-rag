"""Graph RAG: entity extraction + relationship traversal with networkx."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    entity_type: str


@dataclass
class Relation:
    source: str
    relation: str
    target: str
    context: str = ""


@dataclass
class KnowledgeGraph:
    graph: nx.Graph = field(default_factory=nx.Graph)
    entity_index: dict[str, list[str]] = field(default_factory=dict)

    def add_entity(self, entity: Entity) -> None:
        self.graph.add_node(entity.name, entity_type=entity.entity_type)
        key = entity.name.lower()
        self.entity_index.setdefault(key, []).append(entity.name)

    def add_relation(self, rel: Relation) -> None:
        self.graph.add_edge(rel.source, rel.target, relation=rel.relation, context=rel.context)

    def find_entities(self, text: str) -> list[str]:
        """Find entity names mentioned in a query."""
        found: list[str] = []
        for key, names in self.entity_index.items():
            if key in text.lower():
                found.extend(names)
        return list(set(found))

    def traverse(self, entities: list[str], max_depth: int = 2) -> list[tuple[str, str, str]]:
        """Return triples reachable from the given entities within max_depth."""
        triples: list[tuple[str, str, str]] = []
        visited: set[str] = set()
        frontier = [(e, 0) for e in entities]

        while frontier:
            node, depth = frontier.pop(0)
            if node in visited or depth > max_depth:
                continue
            visited.add(node)
            if node not in self.graph:
                continue
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor) or {}
                relation = edge_data.get("relation", "related_to")
                triples.append((node, relation, neighbor))
                if neighbor not in visited:
                    frontier.append((neighbor, depth + 1))
        return triples


class GraphRAGExtractor:
    """Extract entities and relations from chunks using configurable backends."""

    def __init__(self, method: str = "heuristic", threshold: float = 0.5) -> None:
        self.method = method
        self.threshold = threshold
        self._gliner_model = None

    def extract(self, text: str) -> tuple[list[Entity], list[Relation]]:
        if self.method == "gliner":
            return self._extract_gliner(text)
        if self.method == "llm":
            # Placeholder for LLM-based extraction (requires passing an LLM client)
            logger.warning("LLM extraction not yet wired; falling back to heuristic")
        return self._extract_heuristic(text)

    def _extract_heuristic(self, text: str) -> tuple[list[Entity], list[Relation]]:
        """Regex-based extraction for common patterns (offline, fast)."""
        entities: list[Entity] = []
        relations: list[Relation] = []

        # Capitalized multi-word sequences as candidate entities
        candidates = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b", text)
        seen: set[str] = set()
        for c in candidates:
            if c not in seen:
                seen.add(c)
                entities.append(Entity(name=c, entity_type="Unknown"))

        # Simple relation patterns
        patterns = [
            (r"([\w\s]+) is (?:the |a )?(?:CEO|president|founder) of ([\w\s]+)", "leads"),
            (r"([\w\s]+) founded ([\w\s]+)", "founded"),
            (r"([\w\s]+) developed ([\w\s]+)", "developed"),
            (r"([\w\s]+) is backed by ([\w\s]+)", "backed_by"),
            (r"([\w\s]+) invested in ([\w\s]+)", "invested_in"),
            (r"([\w\s]+) (?:works at|works for|is at) ([\w\s]+)", "works_at"),
            (r"([\w\s]+) headquartered in ([\w\s]+)", "located_in"),
        ]
        for pattern, rel_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                src = match.group(1).strip()
                tgt = match.group(2).strip()
                if src and tgt and len(src) > 1 and len(tgt) > 1:
                    relations.append(Relation(source=src, relation=rel_type, target=tgt))

        return entities, relations

    def _extract_gliner(self, text: str) -> tuple[list[Entity], list[Relation]]:
        """Use GLiNER ONNX for fast offline NER."""
        try:
            from gliner import GLiNER
            if self._gliner_model is None:
                logger.info("Loading GLiNER model (urchade/gliner_medium-v2.1)")
                self._gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
            labels = ["person", "company", "technology", "location", "project"]
            predictions = self._gliner_model.predict_entities(text, labels, threshold=self.threshold)
            entities = [
                Entity(name=p["text"], entity_type=p["label"])
                for p in predictions
            ]
            # Relations require LLM or heuristic post-processing
            _, relations = self._extract_heuristic(text)
            return entities, relations
        except Exception as exc:
            logger.warning("GLiNER extraction failed (%s); falling back to heuristic", exc)
            return self._extract_heuristic(text)


def build_knowledge_graph(chunks: list, extractor: GraphRAGExtractor) -> KnowledgeGraph:
    """Build a knowledge graph from a list of chunks."""
    kg = KnowledgeGraph()
    for chunk in chunks:
        entities, relations = extractor.extract(chunk.text)
        for e in entities:
            kg.add_entity(e)
        for r in relations:
            kg.add_relation(r)
    logger.info("Knowledge graph built: %d nodes, %d edges", kg.graph.number_of_nodes(), kg.graph.number_of_edges())
    return kg


def graph_augment_context(kg: KnowledgeGraph, query: str, max_depth: int = 2) -> str:
    """Given a query, find related entities and return their triples as context."""
    query_entities = kg.find_entities(query)
    if not query_entities:
        return ""
    triples = kg.traverse(query_entities, max_depth=max_depth)
    if not triples:
        return ""
    lines = [f"{s} --{r}--> {t}" for s, r, t in triples]
    return "Knowledge graph context:\n" + "\n".join(lines)