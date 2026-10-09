"""RAG orchestration v2: hybrid retrieval + rerank + persistent history."""

from __future__ import annotations
from src.graph_store_falkor import FalkorGraphStore
import logging
from dataclasses import dataclass, field
from src.query_expander import QueryExpander
from src.graph_rag import KnowledgeGraph, graph_augment_context
from src.history_store import HistoryStore
from src.hybrid_retriever import HybridChunk, HybridRetriever
from src.llm import LLMError, OllamaLLM
from src.prompts import (
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    build_context_block,
    format_sources,
)
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
        llm: OllamaLLM,
        reranker: CrossEncoderReranker | None,
        max_history_messages: int,
        rerank_top_k: int = 5,
        relevance_threshold: float = 0.70,
        history_store: HistoryStore | None = None,
        collection: str | None = None,
        query_expander: QueryExpander | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
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
        self.debug = debug
        self.last_sources: list = []
        self.query_expander = query_expander
        self.graph_max_depth = graph_max_depth
        self.knowledge_graph: FalkorGraphStore | None = knowledge_graph

    async def answer(self, question: str) -> RAGAnswer:
        """Async version of answer()."""
        # ... same retrieval/rerank logic, but await ...
        answer_text = await self.llm.agenerate(system=system_prompt, messages=messages)
        question = (question or "").strip()
        if not question:
            return RAGAnswer(answer="Please provide a non-empty question.")

        # 1. Query expansion
        queries = [question]
        if self.query_expander:
            queries = self.query_expander.expand(question)

        # 2. Retrieve for each expanded query and merge
        all_candidates: dict[str, HybridChunk] = {}
        for q in queries:
            for c in self.retriever.retrieve(q):
                key = c.metadata.get("chunk_id") or c.text[:80]
                if key not in all_candidates or c.rrf_score > all_candidates[key].rrf_score:
                    all_candidates[key] = c
        candidates = list(all_candidates.values())

        # 3. Graph augmentation
        graph_context = ""
        if self.knowledge_graph and candidates:
            graph_context = graph_augment_context(self.knowledge_graph, question, self.graph_max_depth)

        if self.knowledge_graph:
            ents = self.knowledge_graph.find_entities(question)
            if ents:
                triples = self.knowledge_graph.traverse(ents, max_depth=self.graph_max_depth)
                if triples:
                    graph_context = "Knowledge graph context:\n" + "\n".join(
                        f"{s} --{r}--> {t}" for s, r, t in triples
                    )

        # 4. Re-rank with cross-encoder
        if self.reranker and candidates:
            reranked = self.reranker.rerank(question, candidates, top_k=self.rerank_top_k)
            final_chunks = [
                HybridChunk(
                    text=r.text,
                    metadata=r.metadata,
                    similarity=r.original_score,
                    distance=1.0 - r.original_score,
                    bm25_score=0.0,
                    rrf_score=r.rerank_score,
                )
                for r in reranked
            ]
        else:
            final_chunks = candidates[: self.rerank_top_k]

        # 5. Threshold filter
        kept = [c for c in final_chunks if c.similarity >= self.relevance_threshold or c.rrf_score > 0]
        if self.debug:
            for i, c in enumerate(kept, 1):
                logger.info("  [%d] sim=%.3f rrf=%.4f src=%s", i, c.similarity, c.rrf_score, c.metadata.get("filename"))

        if not kept:
            self.last_sources = []
            self._push_history(question, NO_CONTEXT_ANSWER)
            return RAGAnswer(answer=NO_CONTEXT_ANSWER, sources=[], used_context=False)

        # 6. Build prompt and generate
        context_block = build_context_block(kept)
        system_prompt = SYSTEM_PROMPT.format(context=context_block)
        history = self.history_store.get_recent(self.max_history_messages) if self.history_store else []
        messages = [*history, {"role": "user", "content": question}]

        try:
            answer_text = self.llm.generate(system=system_prompt, messages=messages)
        except LLMError as exc:
            logger.error("LLM error: %s", exc)
            return RAGAnswer(
                answer="The language model is currently unavailable. Check Ollama.",
                sources=kept,
                used_context=True,
            )

        refusal_markers = ("not available in the knowledge base", "could not find", "cannot find")
        refused = any(m in answer_text.lower() for m in refusal_markers)
        sources = [] if refused else kept
        self.last_sources = sources
        self._push_history(question, answer_text)
        return RAGAnswer(answer=answer_text, sources=sources, used_context=True)

    async def stream_answer(self, question: str):
        """Async streaming version."""
        async for token in self.llm.astream_generate(system=system_prompt, messages=messages):
            yield {"type": "token", "content": token}
        yield {"type": "done", "sources": sources}
        question = (question or "").strip()
        if not question:
            yield {"type": "token", "content": "Please provide a non-empty question."}
            yield {"type": "done", "sources": []}
            return

        candidates = self.retriever.retrieve(question)
        if self.reranker and candidates:
            reranked = self.reranker.rerank(question, candidates, top_k=self.rerank_top_k)
            final_chunks = [
                HybridChunk(
                    text=r.text, metadata=r.metadata,
                    similarity=r.original_score, distance=1.0 - r.original_score,
                    bm25_score=0.0, rrf_score=r.rerank_score,
                )
                for r in reranked
            ]
        else:
            final_chunks = candidates[: self.rerank_top_k]

        kept = [c for c in final_chunks if c.similarity >= self.relevance_threshold or c.rrf_score > 0]
        if not kept:
            yield {"type": "token", "content": NO_CONTEXT_ANSWER}
            yield {"type": "done", "sources": []}
            return

        context_block = build_context_block(kept)
        system_prompt = SYSTEM_PROMPT.format(context=context_block)
        history = self.history_store.get_recent(self.max_history_messages) if self.history_store else []
        messages = [*history, {"role": "user", "content": question}]

        full = []
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