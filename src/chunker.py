"""Deterministic text chunking with overlap and metadata preservation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from src.document_loader import LoadedDocument


@dataclass
class Chunk:
    """A single chunk of text with attached metadata."""

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _stable_chunk_id(source: str, page: int, index: int, text: str) -> str:
    """Deterministic ID based on document + page + index + text hash."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{source}::p{page}::c{index}::{digest}"


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks.

    Uses a simple sliding window over characters. Prefers breaking on
    paragraph / sentence / word boundaries when possible for cleaner chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)
    step = chunk_size - chunk_overlap

    while start < length:
        end = min(start + chunk_size, length)
        piece = text[start:end]

        # Try to break on a paragraph or sentence boundary when possible.
        if end < length:
            for boundary in ("\n\n", "\n", ". ", "! ", "? "):
                cut = piece.rfind(boundary)
                # Only accept the boundary if it keeps at least 60% of the window.
                if cut != -1 and cut >= int(chunk_size * 0.6):
                    piece = piece[: cut + len(boundary)]
                    end = start + len(piece)
                    break

        piece = piece.strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break
        start = end - chunk_overlap

    return chunks


def chunk_document(doc: LoadedDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk every page of a loaded document."""
    chunks: list[Chunk] = []
    for page in doc.pages:
        page_number = int(page.metadata.get("page", 1))
        pieces = split_text(page.text, chunk_size, chunk_overlap)
        for idx, piece in enumerate(pieces):
            chunk_id = _stable_chunk_id(doc.source, page_number, idx, piece)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    metadata={
                        "source": doc.source,
                        "filename": doc.filename,
                        "file_type": doc.file_type,
                        "page": page_number,
                        "chunk_index": idx,
                    },
                )
            )
    return chunks