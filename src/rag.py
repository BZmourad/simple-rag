"""RAG orchestration: retrieval + LLM + history + sources."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.llm import LLMError, OllamaLLM
from src.prompts import (
    NO_CONTEXT_ANSWER,
    OFF_TOPIC_ANSWER,
    SYSTEM_PROMPT,
    build_context_block,
    format_sources,
)
from src.retriever import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    used_context: bool = False


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: OllamaLLM,
        max_history_messages: int,
        debug: bool = False,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.max_history_messages = max_history_messages
        self.debug = debug
        # History is a list of {"role": "user"|"assistant", "content": str}
        self.history: list[dict] = []
        self.last_sources: list[RetrievedChunk] = []

    # ------------------------------------------------------------------ #
    def answer(self, question: str) -> RAGAnswer:
        question = (question or "").strip()
        if not question:
            return RAGAnswer(answer="Please provide a non-empty question.")

        retrieved = self.retriever.retrieve(question)

        if self.debug:
            logger.info("Question: %s", question)
            for i, c in enumerate(retrieved, 1):
                logger.info(
                    "  [%d] sim=%.3f dist=%.3f src=%s p=%s",
                    i,
                    c.similarity,
                    c.distance,
                    c.metadata.get("filename"),
                    c.metadata.get("page"),
                )

        if not retrieved:
            # No chunk above threshold -> strict refusal.
            self.last_sources = []
            self._push_history(question, NO_CONTEXT_ANSWER)
            return RAGAnswer(answer=NO_CONTEXT_ANSWER, sources=[], used_context=False)

        context_block = build_context_block(retrieved)
        system_prompt = SYSTEM_PROMPT.format(context=context_block)

        # Build chat history (trimmed), then append the current question.
        history = self.history[-self.max_history_messages :] if self.max_history_messages > 0 else []
        messages = [*history, {"role": "user", "content": question}]

        try:
            answer_text = self.llm.generate(system=system_prompt, messages=messages)
        except LLMError as exc:
            logger.error("LLM error: %s", exc)
            answer_text = (
                "The language model is currently unavailable. "
                "Please check that Ollama is running and the configured model is installed."
            )
            return RAGAnswer(answer=answer_text, sources=retrieved, used_context=True)

        # Detect if the LLM explicitly refused (best-effort, for source display).
        refusal_markers = (
            "not available in the knowledge base",
            "cannot find",
            "could not find",
            "no information",
            "only answer questions related",
        )
        refused = any(marker in answer_text.lower() for marker in refusal_markers)

        sources = [] if refused else retrieved
        self.last_sources = sources
        self._push_history(question, answer_text)

        return RAGAnswer(answer=answer_text, sources=sources, used_context=True)

    # ------------------------------------------------------------------ #
    def _push_history(self, question: str, answer: str) -> None:
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        # Hard cap so memory doesn't grow unbounded.
        max_entries = max(self.max_history_messages * 2, 2)
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    def clear_history(self) -> None:
        self.history.clear()
        self.last_sources = []

    def format_last_sources(self) -> str:
        if not self.last_sources:
            return "No sources available for the last answer."
        return format_sources(self.last_sources)