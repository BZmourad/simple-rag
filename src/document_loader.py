"""Load supported documents (PDF, DOCX, TXT, MD) into structured records."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


@dataclass
class DocumentPage:
    """A single unit of extracted text (page or whole file for txt/md)."""

    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class LoadedDocument:
    """A loaded document with its pages."""

    source: str
    filename: str
    file_type: str
    pages: list[DocumentPage]


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_documents(directory: Path) -> list[Path]:
    """Return all supported files under the given directory (non-recursive + recursive)."""
    if not directory.exists():
        logger.warning("Documents directory does not exist: %s", directory)
        return []

    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            if is_supported(path):
                files.append(path)
            elif path.name != ".gitkeep":
                logger.warning("Skipping unsupported file: %s", path)
    return files


def _load_pdf(path: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    try:
        with fitz.open(path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                text = page.get_text("text") or ""
                if text.strip():
                    pages.append(
                        DocumentPage(
                            text=text,
                            metadata={"page": page_index},
                        )
                    )
    except Exception as exc:  # pragma: no cover - depends on bad file
        logger.error("Failed to read PDF %s: %s", path, exc)
        raise
    return pages


def _load_docx(path: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    try:
        doc = DocxDocument(str(path))
        # Merge all paragraphs into a single "page" (DOCX has no real pages).
        text = "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
        # Include table text as well.
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells:
                    text += "\n" + " | ".join(cells)
        if text.strip():
            pages.append(DocumentPage(text=text, metadata={"page": 1}))
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to read DOCX %s: %s", path, exc)
        raise
    return pages


def _load_text_file(path: Path) -> list[DocumentPage]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to read text file %s: %s", path, exc)
        raise

    if not text.strip():
        return []
    return [DocumentPage(text=text, metadata={"page": 1})]


def load_document(path: Path, documents_root: Path) -> LoadedDocument | None:
    """Load a single document. Returns None if empty or on failure."""
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            pages = _load_pdf(path)
        elif ext == ".docx":
            pages = _load_docx(path)
        elif ext in {".txt", ".md", ".markdown"}:
            pages = _load_text_file(path)
        else:
            logger.warning("Unsupported file type: %s", path)
            return None
    except Exception:
        return None

    if not pages:
        logger.warning("Document is empty or could not be parsed: %s", path)
        return None

    try:
        relative = path.relative_to(documents_root)
    except ValueError:
        relative = path.name

    return LoadedDocument(
        source=str(relative),
        filename=path.name,
        file_type=ext.lstrip("."),
        pages=pages,
    )


def load_documents(directory: Path) -> Iterable[LoadedDocument]:
    """Yield loaded documents from a directory."""
    for path in discover_documents(directory):
        logger.info("Loading document: %s", path.name)
        doc = load_document(path, directory)
        if doc is not None:
            yield doc