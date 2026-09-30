"""System and user prompt templates for the RAG agent."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a specialized knowledge-base assistant.

Your job is to answer questions using ONLY the context provided by the retrieval system.

Rules:
- Use only the provided context.
- Do not invent information.
- Do not rely on your general knowledge.
- If the context does not contain the answer, say clearly that the information is not available in the knowledge base.
- If the question is unrelated to the knowledge base topic, politely explain that you can only answer questions related to the configured topic.
- Keep answers clear and concise.
- When possible, reference the source documents that support the answer.

Context:
{context}
"""


NO_CONTEXT_ANSWER = (
    "I could not find sufficient information in the knowledge base to answer this question."
)

OFF_TOPIC_ANSWER = (
    "I can only answer questions related to the configured knowledge base."
)


def build_context_block(chunks: list) -> str:
    """Render retrieved chunks into a numbered context block."""
    if not chunks:
        return ""
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        source = meta.get("filename") or meta.get("source") or "unknown"
        page = meta.get("page")
        header = f"[{i}] {source}" + (f" — page {page}" if page else "")
        lines.append(f"{header}\n{chunk.text}")
    return "\n\n".join(lines)


def format_sources(chunks: list) -> str:
    """Format a deduplicated, ordered list of sources."""
    seen: set[tuple[str, int | None]] = set()
    lines: list[str] = []
    for chunk in chunks:
        meta = chunk.metadata or {}
        source = meta.get("filename") or meta.get("source") or "unknown"
        page = meta.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        if page:
            lines.append(f"- {source} — page {page}")
        else:
            lines.append(f"- {source}")
    return "\n".join(lines)