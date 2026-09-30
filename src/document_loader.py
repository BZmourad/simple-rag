"""Load supported documents with optional OCR for scanned PDFs."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import pytesseract
from docx import Document as DocxDocument
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


@dataclass
class DocumentPage:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class LoadedDocument:
    source: str
    filename: str
    file_type: str
    pages: list[DocumentPage]


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_documents(directory: Path) -> list[Path]:
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


def _ocr_page(page: fitz.Page, lang: str, dpi: int) -> str:
    """Render a PDF page to image and OCR it."""
    try:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang=lang)
    except Exception as exc:
        logger.warning("OCR failed on page: %s", exc)
        return ""


def _load_pdf(path: Path, ocr_enabled: bool, ocr_lang: str, ocr_dpi: int, ocr_min_chars: int) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    try:
        with fitz.open(path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                text = (page.get_text("text") or "").strip()

                # OCR fallback for scanned pages
                if ocr_enabled and len(text) < ocr_min_chars:
                    logger.debug("Running OCR on %s page %d (text len=%d)", path.name, page_index, len(text))
                    ocr_text = _ocr_page(page, lang=ocr_lang, dpi=ocr_dpi).strip()
                    if ocr_text:
                        text = ocr_text
                        logger.info("OCR extracted %d chars from %s page %d", len(text), path.name, page_index)

                if text:
                    pages.append(DocumentPage(text=text, metadata={"page": page_index}))
    except Exception as exc:
        logger.error("Failed to read PDF %s: %s", path, exc)
        raise
    return pages


def _load_docx(path: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    try:
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells:
                    text += "\n" + " | ".join(cells)
        if text.strip():
            pages.append(DocumentPage(text=text, metadata={"page": 1}))
    except Exception as exc:
        logger.error("Failed to read DOCX %s: %s", path, exc)
        raise
    return pages


def _load_text_file(path: Path) -> list[DocumentPage]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.error("Failed to read text file %s: %s", path, exc)
        raise
    return [DocumentPage(text=text, metadata={"page": 1})] if text.strip() else []


def load_document(path: Path, documents_root: Path, cfg=None) -> LoadedDocument | None:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            ocr_enabled = getattr(cfg, "ocr_enabled", True) if cfg else True
            ocr_lang = getattr(cfg, "ocr_lang", "eng") if cfg else "eng"
            ocr_dpi = getattr(cfg, "ocr_dpi", 300) if cfg else 300
            ocr_min_chars = getattr(cfg, "ocr_min_chars", 30) if cfg else 30
            pages = _load_pdf(path, ocr_enabled, ocr_lang, ocr_dpi, ocr_min_chars)
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


def load_documents(directory: Path, cfg=None) -> Iterable[LoadedDocument]:
    for path in discover_documents(directory):
        logger.info("Loading document: %s", path.name)
        doc = load_document(path, directory, cfg=cfg)
        if doc is not None:
            yield doc