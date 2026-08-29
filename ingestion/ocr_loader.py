"""
ingestion/ocr_loader.py
------------------------
Loads PDFs page-by-page. For every page it first tries to pull the native
text layer with PyMuPDF (fast, exact). If a page has little/no extractable
text (e.g. a scanned page, or a page that's mostly a table rendered as an
image) it falls back to OCR via pytesseract, using Arabic + English
language packs.

Output: a list of LangChain `Document` objects, one per page, with rich
metadata (source file, page number, extraction method) that later powers
citations in the generation step.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when running this script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import fitz  # PyMuPDF
import pytesseract
# pyrefly: ignore [missing-import]
from PIL import Image
from langchain_core.documents import Document

from config import OCR_CACHE_DIR, OCR_DPI, OCR_LANGUAGES, MIN_CHARS_FOR_NATIVE_TEXT

logger = logging.getLogger(__name__)


def _cache_path(pdf_path: Path) -> Path:
    """Each PDF gets a cache file keyed by its content hash so re-running
    ingestion on unchanged files never re-runs (slow) OCR."""
    file_hash = hashlib.md5(pdf_path.read_bytes()).hexdigest()
    return OCR_CACHE_DIR / f"{pdf_path.stem}_{file_hash}.json"


def _page_to_image(page: "fitz.Page", dpi: int = OCR_DPI) -> Image.Image:
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _ocr_page(page: "fitz.Page") -> str:
    image = _page_to_image(page)
    text = pytesseract.image_to_string(image, lang=OCR_LANGUAGES)
    return text.strip()


def load_pdf_with_ocr(pdf_path: str | Path, use_cache: bool = True) -> List[Document]:
    """Load a single PDF, returning one Document per page.

    Extraction strategy per page:
      1. Try native text extraction (fitz) -- fast & exact.
      2. If that yields < MIN_CHARS_FOR_NATIVE_TEXT characters, treat the
         page as image-based and run Tesseract OCR on a rasterized version.
    """
    pdf_path = Path(pdf_path)
    cache_file = _cache_path(pdf_path)

    if use_cache and cache_file.exists():
        logger.info("Loading cached OCR/text extraction for %s", pdf_path.name)
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in cached
        ]

    documents: List[Document] = []
    with fitz.open(pdf_path) as pdf:
        for page_number in range(len(pdf)):
            page = pdf[page_number]
            native_text = page.get_text("text").strip()

            if len(native_text) >= MIN_CHARS_FOR_NATIVE_TEXT:
                text = native_text
                method = "native"
            else:
                logger.info(
                    "Page %s of %s has little native text -> running OCR",
                    page_number + 1,
                    pdf_path.name,
                )
                text = _ocr_page(page)
                method = "ocr"

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number + 1,
                        "extraction_method": method,
                    },
                )
            )

    if use_cache:
        cache_file.write_text(
            json.dumps(
                [{"page_content": d.page_content, "metadata": d.metadata} for d in documents],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return documents


def load_all_pdfs(pdf_dir: str | Path, use_cache: bool = True) -> List[Document]:
    """Load every .pdf file found in `pdf_dir`."""
    pdf_dir = Path(pdf_dir)
    all_docs: List[Document] = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)

    for pdf_file in pdf_files:
        logger.info("Loading %s", pdf_file.name)
        all_docs.extend(load_pdf_with_ocr(pdf_file, use_cache=use_cache))

    return all_docs
