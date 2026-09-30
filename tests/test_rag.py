"""Tests for the RAG pipeline (mocked LLM and embedder)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.rag import RAGPipeline
from src.retriever import RetrievedChunk


class _FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]):
        self._chunks = chunks
        self.calls: list[str] = []

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        self.calls.append(question)
        return list(self._chunks)


@dataclass
class _FakeLLM:
    reply: str = "Employees receive 25 days of annual leave."
    error: Exception | None = None

    def generate(self, system: str, messages) -> str:
        self.last_system = system
        self.last_messages = list(messages)
        if self.error is not None:
            raise self.error
        return self.reply

    def health_check(self) -> None:  # pragma: no cover
        pass


def _chunk(text: str, sim: float = 0.9, source: str = "HR.pdf", page: int = 4) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata={"source": source, "filename": source, "page": page},
        similarity=sim,
        distance=1.0 - sim,
    )


def test_empty_retrieval_returns_refusal():
    retriever = _FakeRetriever([])
    llm = _FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm, max_history_messages=4)

    result = pipeline.answer("What is the annual leave policy?")

    assert "could not find" in result.answer.lower()
    assert result.sources == []
    assert result.used_context is False


def test_relevant_context_is_used_and_sources_returned():
    chunks = [_chunk("Employees get 25 days of annual leave.", sim=0.85)]
    retriever = _FakeRetriever(chunks)
    llm = _FakeLLM(reply="Employees receive 25 days of annual leave.")
    pipeline = RAGPipeline(retriever=retriever, llm=llm, max_history_messages=4)

    result = pipeline.answer("How many leave days?")
    assert "25 days" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].metadata["filename"] == "HR.pdf"


def test_irrelevant_answer_hides_sources():
    """If the LLM says the info isn't available, sources should not be shown."""
    chunks = [_chunk("Cafeteria menu for Monday.", sim=0.75)]
    retriever = _FakeRetriever(chunks)
    llm = _FakeLLM(reply="This information is not available in the knowledge base.")
    pipeline = RAGPipeline(retriever=retriever, llm=llm, max_history_messages=4)

    result = pipeline.answer("What is the company revenue?")
    assert result.sources == []


def test_llm_error_is_handled():
    from src.llm import LLMError

    chunks = [_chunk("Some relevant text.", sim=0.9)]
    retriever = _FakeRetriever(chunks)
    llm = _FakeLLM(error=LLMError("boom"))
    pipeline = RAGPipeline(retriever=retriever, llm=llm, max_history_messages=4)

    result = pipeline.answer("Anything?")
    assert "unavailable" in result.answer.lower()


def test_history_is_passed_and_capped():
    chunks = [_chunk("Some text.", sim=0.9)]
    retriever = _FakeRetriever(chunks)
    llm = _FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm, max_history_messages=2)

    pipeline.answer("Q1")
    pipeline.answer("Q2")
    pipeline.answer("Q3")

    # The last call should have at most max_history_messages * 2 + 1 entries:
    # (history entries) + current user message.
    assert len(llm.last_messages) <= 2 * 2 + 1


def test_empty_question_short_circuits():
    retriever = _FakeRetriever([_chunk("x", sim=0.9)])
    llm = _FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm, max_history_messages=4)
    result = pipeline.answer("   ")
    assert "empty" in result.answer.lower()
    assert retriever.calls == []


def test_format_last_sources_deduplicates():
    chunks = [
        _chunk("a", source="HR.pdf", page=4),
        _chunk("b", source="HR.pdf", page=4),
        _chunk("c", source="Handbook.pdf", page=12),
    ]
    pipeline = RAGPipeline(
        retriever=_FakeRetriever(chunks), llm=_FakeLLM(), max_history_messages=4
    )
    pipeline.answer("q")
    # Force sources so we can check the formatter directly.
    pipeline.last_sources = chunks
    rendered = pipeline.format_last_sources()
    assert rendered.count("HR.pdf") == 1
    assert "Handbook.pdf" in rendered