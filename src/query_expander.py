"""Query expansion via HyDE and Multi-Query generation."""

from __future__ import annotations

import logging

from src.llm import LLMError, OllamaLLM

logger = logging.getLogger(__name__)

HYDE_PROMPT = """Write a short, factual passage (3-5 sentences) that would answer the following question. Do not mention that you are generating a hypothetical document. Just write the passage.

Question: {question}

Passage:"""

MULTI_QUERY_PROMPT = """Generate {num_queries} different versions of the following question, each phrased from a different perspective. Return one per line, numbered 1., 2., 3. Do not add explanations.

Question: {question}"""


class QueryExpander:
    """Expand a user query into one or more retrieval-friendly queries."""

    def __init__(self, llm: OllamaLLM, mode: str = "hyde", num_queries: int = 3, hyde_temp: float = 0.7) -> None:
        self.llm = llm
        self.mode = mode
        self.num_queries = num_queries
        self.hyde_temp = hyde_temp

    def expand(self, question: str) -> list[str]:
        """Return a list of queries to run against the retriever.

        The original question is always included first so that even if
        expansion fails, retrieval still has a baseline query.
        """
        queries: list[str] = [question]

        if self.mode in {"hyde", "both"}:
            try:
                queries.append(self._hyde(question))
            except LLMError as exc:
                logger.warning("HyDE expansion failed: %s", exc)

        if self.mode in {"multi_query", "both"}:
            try:
                queries.extend(self._multi_query(question))
            except LLMError as exc:
                logger.warning("Multi-query expansion failed: %s", exc)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for q in queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(q.strip())
        logger.debug("Query expansion produced %d unique queries", len(unique))
        return unique

    def _hyde(self, question: str) -> str:
        prompt = HYDE_PROMPT.format(question=question)
        return self.llm.generate(
            system="You are a helpful assistant that writes factual passages.",
            messages=[{"role": "user", "content": prompt}],
        )

    def _multi_query(self, question: str) -> list[str]:
        prompt = MULTI_QUERY_PROMPT.format(question=question, num_queries=self.num_queries)
        raw = self.llm.generate(
            system="You rewrite questions into multiple phrasings. Output only the numbered list.",
            messages=[{"role": "user", "content": prompt}],
        )
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        queries: list[str] = []
        for line in lines:
            cleaned = line.lstrip("0123456789.-) ").strip()
            if cleaned and len(cleaned) > 3:
                queries.append(cleaned)
        return queries[: self.num_queries]