"""RAG orchestration v7: hybrid retrieval + rerank + persistent history (async)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.graph_store_falkor import FalkorGraphStore
from src.history_store import HistoryStore
from src.hybrid_retriever import HybridChunk, HybridRetriever
from src.llm import LLMError
from src.prompts import (
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    build_context_block,
    format_sources,
)
from src.query_expander import QueryExpander
from src.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    answer: str
    sources: list = field(default_factory=list)
    used_context: bool = False


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm,
        reranker: CrossEncoderReranker | None,
        max_history_messages: int,
        rerank_top_k: int = 5,
        relevance_threshold: float = 0.70,
        history_store: HistoryStore | None = None,
        collection: str | None = None,
        query_expander: QueryExpander | None = None,
        knowledge_graph: FalkorGraphStore | None = None,
        graph_max_depth: int = 2,
        debug: bool = False,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.max_history_messages = max_history_messages
        self.rerank_top_k = rerank_top_k
        self.relevance_threshold = relevance_threshold
        self.history_store = history_store
        self.collection = collection
        self.query_expander = query_expander
        self.knowledge_graph = knowledge_graph
        self.graph_max_depth = graph_max_depth
        self.debug = debug
        self.last_sources: list = []

    # ------------------------------------------------------------------ #
    # Retrieval helpers (sync — vector DB is sync)
    # ------------------------------------------------------------------ #
    def _retrieve_candidates(self, question: str) -> list[HybridChunk]:
        queries = [question]
        if self.query_expander:
            try:
                queries = self.query_expander.expand(question)
            except Exception as exc:
                logger.warning("Query expansion failed: %s", exc)

        merged: dict[str, HybridChunk] = {}
        for q in queries:
            for c in self.retriever.retrieve(q):
                key = c.metadata.get("chunk_id") or c.text[:80]
                if key not in merged or c.rrf_score > merged[key].rrf_score:
                    merged[key] = c
        return list(merged.values())

    def _rerank_and_filter(self, question: str, candidates: list[HybridChunk]) -> list[HybridChunk]:
        if self.reranker and candidates:
            reranked = self.reranker.rerank(question, candidates, top_k=self.rerank_top_k)
            final = [
                HybridChunk(
                    text=r.text, metadata=r.metadata,
                    similarity=r.original_score, distance=1.0 - r.original_score,
                    bm25_score=0.0, rrf_score=r.rerank_score,
                )
                for r in reranked
            ]
        else:
            final = candidates[: self.rerank_top_k]

        kept = [c for c in final if c.similarity >= self.relevance_threshold or c.rrf_score > 0]

        if self.debug:
            for i, c in enumerate(kept, 1):
                logger.info("  [%d] sim=%.3f rrf=%.4f src=%s",
                            i, c.similarity, c.rrf_score, c.metadata.get("filename"))
        return kept

    def _graph_context(self, question: str) -> str:
        if not self.knowledge_graph:
            return ""
        try:
            ents = self.knowledge_graph.find_entities(question)
            if not ents:
                return ""
            triples = self.knowledge_graph.traverse(ents, max_depth=self.graph_max_depth)
            if not triples:
                return ""
            return "Knowledge graph context:\n" + "\n".join(
                f"{s} --{r}--> {t}" for s, r, t in triples
            )
        except Exception as exc:
            logger.warning("Graph augmentation failed: %s", exc)
            return ""

    # ------------------------------------------------------------------ #
    # Sync answer (used by CLI eval)
    # ------------------------------------------------------------------ #
    def answer(self, question: str) -> RAGAnswer:
        question = (question or "").strip()
        if not question:
            return RAGAnswer(answer="Please provide a non-empty question.")

        candidates = self._retrieve_candidates(question)
        kept = self._rerank_and_filter(question, candidates)

        if not kept:
            self.last_sources = []
            self._push_history(question, NO_CONTEXT_ANSWER)
            return RAGAnswer(answer=NO_CONTEXT_ANSWER, sources=[], used_context=False)

        context_block = build_context_block(kept)
        graph_ctx = self._graph_context(question)
        if graph_ctx:
            context_block = context_block + "\n\n" + graph_ctx

        system_prompt = SYSTEM_PROMPT.format(context=context_block)
        history = self.history_store.get_recent(self.max_history_messages) if self.history_store else []
        messages = [*history, {"role": "user", "content": question}]

        try:
            answer_text = self.llm.generate(system=system_prompt, messages=messages)
        except LLMError as exc:
            logger.error("LLM error: %s", exc)
            return RAGAnswer(
                answer="The language model is currently unavailable.",
                sources=kept,
                used_context=True,
            )

        refusal_markers = ("not available in the knowledge base", "could not find", "cannot find")
        refused = any(m in answer_text.lower() for m in refusal_markers)
        sources = [] if refused else kept
        self.last_sources = sources
        self._push_history(question, answer_text)
        return RAGAnswer(answer=answer_text, sources=sources, used_context=True)

    # ------------------------------------------------------------------ #
    # Sync streaming (used by CLI chat)
    # ------------------------------------------------------------------ #
    def stream_answer(self, question: str):
        question = (question or "").strip()
        if not question:
            yield {"type": "token", "content": "Please provide a non-empty question."}
            yield {"type": "done", "sources": []}
            return

        candidates = self._retrieve_candidates(question)
        kept = self._rerank_and_filter(question, candidates)

        if not kept:
            yield {"type": "token", "content": NO_CONTEXT_ANSWER}
            yield {"type": "done", "sources": []}
            return

        context_block = build_context_block(kept)
        graph_ctx = self._graph_context(question)
        if graph_ctx:
            context_block = context_block + "\n\n" + graph_ctx

        system_prompt = SYSTEM_PROMPT.format(context=context_block)
        history = self.history_store.get_recent(self.max_history_messages) if self.history_store else []
        messages = [*history, {"role": "user", "content": question}]

        full: list[str] = []
        try:
            for token in self.llm.stream_generate(system=system_prompt, messages=messages):
                full.append(token)
                yield {"type": "token", "content": token}
        except LLMError as exc:
            yield {"type": "token", "content": f"\n[Error: {exc}]"}

        answer_text = "".join(full).strip()
        refusal_markers = ("not available in the knowledge base", "could not find", "cannot find")
        refused = any(m in answer_text.lower() for m in refusal_markers)
        sources = [] if refused else kept
        self.last_sources = sources
        self._push_history(question, answer_text)
        yield {"type": "done", "sources": sources}

    # ------------------------------------------------------------------ #
    # Async variants (used by FastAPI)
    # ------------------------------------------------------------------ #
    async def aanswer(self, question: str) -> RAGAnswer:
        question = (question or "").strip()
        if not question:
            return RAGAnswer(answer="Please provide a non-empty question.")

        candidates = self._retrieve_candidates(question)
        kept = self._rerank_and_filter(question, candidates)

        if not kept:
            self.last_sources = []
            self._push_history(question, NO_CONTEXT_ANSWER)
            return RAGAnswer(answer=NO_CONTEXT_ANSWER, sources=[], used_context=False)

        context_block = build_context_block(kept)
        graph_ctx = self._graph_context(question)
        if graph_ctx:
            context_block = context_block + "\n\n" + graph_ctx

        system_prompt = SYSTEM_PROMPT.format(context=context_block)
        history = self.history_store.get_recent(self.max_history_messages) if self.history_store else []
        messages = [*history, {"role": "user", "content": question}]

        try:
            answer_text = await self.llm.agenerate(system=system_prompt, messages=messages)
        except LLMError as exc:
            logger.error("LLM error: %s", exc)
            return RAGAnswer(
                answer="The language model is currently unavailable.",
                sources=kept, used_context=True,
            )

        refusal_markers = ("not available in the knowledge base", "could not find", "cannot find")
        refused = any(m in answer_text.lower() for m in refusal_markers)
        sources = [] if refused else kept
        self.last_sources = sources
        self._push_history(question, answer_text)
        return RAGAnswer(answer=answer_text, sources=sources, used_context=True)

    async def astream_answer(self, question: str):
        question = (question or "").strip()
        if not question:
            yield {"type": "token", "content": "Please provide a non-empty question."}
            yield {"type": "done", "sources": []}
            return

        candidates = self._retrieve_candidates(question)
        kept = self._rerank_and_filter(question, candidates)

        if not kept:
            yield {"type": "token", "content": NO_CONTEXT_ANSWER}
            yield {"type": "done", "sources": []}
            return

        context_block = build_context_block(kept)
        graph_ctx = self._graph_context(question)
        if graph_ctx:
            context_block = context_block + "\n\n" + graph_ctx

        system_prompt = SYSTEM_PROMPT.format(context=context_block)
        history = self.history_store.get_recent(self.max_history_messages) if self.history_store else []
        messages = [*history, {"role": "user", "content": question}]

        full: list[str] = []
        try:
            async for token in self.llm.astream_generate(system=system_prompt, messages=messages):
                full.append(token)
                yield {"type": "token", "content": token}
        except LLMError as exc:
            yield {"type": "token", "content": f"\n[Error: {exc}]"}

        answer_text = "".join(full).strip()
        refusal_markers = ("not available in the knowledge base", "could not find", "cannot find")
        refused = any(m in answer_text.lower() for m in refusal_markers)
        sources = [] if refused else kept
        self.last_sources = sources
        self._push_history(question, answer_text)
        yield {"type": "done", "sources": sources}

    # ------------------------------------------------------------------ #
    def _push_history(self, question: str, answer: str) -> None:
        if not self.history_store:
            return
        self.history_store.append("user", question)
        self.history_store.append("assistant", answer)

    def clear_history(self) -> None:
        if self.history_store:
            self.history_store.clear()
        self.last_sources = []

    def format_last_sources(self) -> str:
        if not self.last_sources:
            return "No sources available for the last answer."
        return format_sources(self.last_sources)