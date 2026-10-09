"""Query expansion via HyDE and Multi-Query generation, with Redis caching."""

from __future__ import annotations
from typing import Any
import hashlib
import json
import logging

import redis

from src.llm import LLMError

logger = logging.getLogger(__name__)

HYDE_PROMPT = """Write a short, factual passage (3-5 sentences) that would answer the following question. Do not mention that you are generating a hypothetical document. Just write the passage.

Question: {question}

Passage:"""

MULTI_QUERY_PROMPT = """Generate {num_queries} different versions of the following question, each phrased from a different perspective. Return one per line, numbered 1., 2., 3. Do not add explanations.

Question: {question}"""


class QueryExpander:
    """Expand a user query into one or more retrieval-friendly queries.

    Modes:
      - "hyde": generate one hypothetical answer document (HyDE)
      - "multi_query": generate N paraphrased questions
      - "both": both of the above

    Results are cached in Redis keyed by (mode, question). Caching avoids
    redundant LLM calls when the same or similar question is asked repeatedly
    (common in RAG evaluation and API traffic).
    """

    def __init__(
        self,
        llm: Any,
        mode: str = "hyde",
        num_queries: int = 3,
        hyde_temp: float = 0.7,
        redis_url: str | None = None,
        cache_ttl: int = 3600,
        cache_prefix: str = "qe",
    ) -> None:
        self.llm = llm
        self.mode = mode
        self.num_queries = num_queries
        self.hyde_temp = hyde_temp
        self.cache_ttl = cache_ttl
        self.cache_prefix = cache_prefix

        self._redis: redis.Redis | None = None
        if redis_url:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("Query expansion cache connected to Redis (%s)", redis_url)
            except Exception as exc:
                logger.warning(
                    "Query expansion cache Redis unavailable (%s); proceeding without cache",
                    exc,
                )
                self._redis = None

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #
    def _cache_key(self, question: str) -> str:
        normalized = question.lower().strip()
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
        return f"{self.cache_prefix}:{self.mode}:{digest}"

    def _get_cached(self, question: str) -> list[str] | None:
        if not self._redis:
            return None
        try:
            raw = self._redis.get(self._cache_key(question))
            if raw:
                logger.debug("Query expansion cache HIT for: %s", question[:60])
                return json.loads(raw)
        except Exception as exc:
            logger.warning("Cache read failed: %s", exc)
        return None

    def _set_cached(self, question: str, queries: list[str]) -> None:
        if not self._redis:
            return
        try:
            self._redis.setex(
                self._cache_key(question),
                self.cache_ttl,
                json.dumps(queries),
            )
        except Exception as exc:
            logger.warning("Cache write failed: %s", exc)

    def invalidate(self, question: str | None = None) -> int:
        """Clear cache for a specific question, or all expansion keys if None."""
        if not self._redis:
            return 0
        try:
            if question is not None:
                return int(self._redis.delete(self._cache_key(question)))
            count = 0
            for key in self._redis.scan_iter(match=f"{self.cache_prefix}:*"):
                count += int(self._redis.delete(key))
            return count
        except Exception as exc:
            logger.warning("Cache invalidation failed: %s", exc)
            return 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def expand(self, question: str) -> list[str]:
        """Return a list of queries to run against the retriever.

        The original question is always included first, so retrieval has a
        baseline query even if expansion fails.
        """
        cached = self._get_cached(question)
        if cached is not None:
            return cached

        queries = self._do_expand(question)
        self._set_cached(question, queries)
        return queries

    def _do_expand(self, question: str) -> list[str]:
        queries: list[str] = [question]

        if self.mode in {"hyde", "both"}:
            try:
                hyde_result = self._hyde(question)
                if hyde_result:
                    queries.append(hyde_result)
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

    # ------------------------------------------------------------------ #
    # Expansion strategies
    # ------------------------------------------------------------------ #
    def _hyde(self, question: str) -> str:
        prompt = HYDE_PROMPT.format(question=question)
        return self.llm.generate(
            system="You are a helpful assistant that writes factual passages.",
            messages=[{"role": "user", "content": prompt}],
        )

    def _multi_query(self, question: str) -> list[str]:
        prompt = MULTI_QUERY_PROMPT.format(
            question=question, num_queries=self.num_queries
        )
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