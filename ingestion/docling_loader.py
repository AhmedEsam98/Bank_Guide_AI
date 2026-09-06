"""
ingestion/docling_loader.py
---------------------------
Docling-based PDF loader.

Docling performs full layout analysis: reading-order correction, table
extraction (as Markdown tables), RTL/Arabic-aware text ordering, and
handles scanned pages via an embedded OCR pipeline.

Output: same `List[Document]` interface as the old `load_all_pdfs()` so the
rest of the pipeline (chunking, embedding) is unaffected.

Caching: JSON cache keyed by MD5 of the PDF bytes so unchanged files are
never re-processed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.documents import Document

from config import (
    OCR_CACHE_DIR,
    DOCLING_MODE,
    DOCLING_DO_OCR,
    DOCLING_OCR_ENGINE,
    DOCLING_OCR_MODEL_NAME,
    DOCLING_OCR_LANGUAGES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Docling pipeline (lazy-init singleton — model weights load once per config)
# ---------------------------------------------------------------------------
_converters = {}


def _get_converter(do_ocr: bool = DOCLING_DO_OCR):
    global _converters
    if do_ocr in _converters:
        return _converters[do_ocr]

    import torch
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        AcceleratorOptions,
        AcceleratorDevice,
        EasyOcrOptions,
    )

    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_table_structure = True        # Extract tables as Markdown via TableFormer
    pipeline_opts.generate_picture_images = True   # Generate picture images (prevents 'Image not available' placeholder)

    if bool(do_ocr):
        pipeline_opts.do_ocr = True
        # Explicitly configure OCR model: EasyOCR (CRAFT detection + CRNN Arabic/English recognition)
        pipeline_opts.ocr_options = EasyOcrOptions(
            lang=DOCLING_OCR_LANGUAGES,
            use_gpu=torch.cuda.is_available(),
        )
        logger.info("Configured OCR model: %s (engine=%s, languages=%s)", DOCLING_OCR_MODEL_NAME, DOCLING_OCR_ENGINE, DOCLING_OCR_LANGUAGES)
    else:
        pipeline_opts.do_ocr = False
        logger.info("OCR disabled (digital text extraction only).")

    if torch.cuda.is_available():
        pipeline_opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CUDA,
        )
        logger.info("Docling using CUDA acceleration (do_ocr=%s).", do_ocr)
    else:
        pipeline_opts.accelerator_options = AcceleratorOptions(num_threads=4)
        logger.info("Docling using CPU (do_ocr=%s).", do_ocr)

    if DOCLING_MODE == "accurate":
        pipeline_opts.table_structure_options.mode = TableFormerMode.ACCURATE
    else:
        pipeline_opts.table_structure_options.mode = TableFormerMode.FAST

    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    _converters[do_ocr] = converter
    logger.info("Docling DocumentConverter ready (mode=%s, ocr_model=%s).", DOCLING_MODE, DOCLING_OCR_MODEL_NAME if do_ocr else "none")
    return converter


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_path(pdf_path: Path) -> Path:
    file_hash = hashlib.md5(pdf_path.read_bytes()).hexdigest()
    return OCR_CACHE_DIR / f"docling_{pdf_path.stem}_{file_hash}.json"


def _clean_markdown_text(text: str) -> str:
    """Strip Docling missing image boilerplate comments and collapse extra newlines."""
    import re
    text = re.sub(r"<!--\s*🖼️?[^>]*PdfPipelineOptions[^>]*-->", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_cache(cache_file: Path) -> List[Document] | None:
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return [
            Document(
                page_content=_clean_markdown_text(d["page_content"]),
                metadata=d["metadata"],
            )
            for d in data
        ]
    except Exception as exc:
        logger.warning("Could not read Docling cache %s: %s", cache_file, exc)
        return None


def _save_cache(cache_file: Path, docs: List[Document]) -> None:
    try:
        cache_file.write_text(
            json.dumps(
                [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Could not write Docling cache %s: %s", cache_file, exc)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------
def load_pdf_with_docling(
    pdf_path: str | Path,
    use_cache: bool = True,
    do_ocr: bool = DOCLING_DO_OCR,
) -> List[Document]:
    """Convert one PDF with Docling and return one Document per page.

    Each Document has metadata:
        source, page, extraction_method, has_tables, word_count
    """
    pdf_path = Path(pdf_path)
    cache_file = _cache_path(pdf_path)

    if use_cache:
        cached = _load_cache(cache_file)
        if cached is not None:
            logger.info("Loaded Docling cache for %s (%d pages).", pdf_path.name, len(cached))
            return cached

    logger.info("Running Docling on %s (do_ocr=%s) …", pdf_path.name, do_ocr)
    t0 = time.time()

    try:
        converter = _get_converter(do_ocr=do_ocr)
        result = converter.convert(str(pdf_path))
    except Exception as exc:
        logger.error("Docling failed on %s: %s", pdf_path.name, exc)
        return []

    elapsed = time.time() - t0
    logger.info("Docling finished %s in %.1fs.", pdf_path.name, elapsed)

    # Export full document as Markdown (preserves tables, headings, reading order)
    full_md = result.document.export_to_markdown()

    # --- Split into per-page Documents in a single pass ---
    docs: List[Document] = []
    page_texts_map: dict[int, list[str]] = {}

    for item, _ in result.document.iterate_items():
        page_no = 1
        if hasattr(item, "prov") and item.prov:
            for p in item.prov:
                if hasattr(p, "page_no"):
                    page_no = p.page_no
                    break

        if hasattr(item, "export_to_markdown"):
            try:
                txt = item.export_to_markdown(doc=result.document)
            except Exception:
                txt = getattr(item, "text", "")
        elif hasattr(item, "text") and item.text:
            txt = item.text
        else:
            txt = ""

        if txt and txt.strip():
            page_texts_map.setdefault(page_no, []).append(txt.strip())

    if page_texts_map:
        for page_no in sorted(page_texts_map.keys()):
            text = _clean_markdown_text("\n\n".join(page_texts_map[page_no]))
            if not text:
                continue

            has_tables = "| --- |" in text or "---|" in text or "|:---" in text
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_no,
                    "extraction_method": "docling",
                    "ocr_model": DOCLING_OCR_MODEL_NAME if do_ocr else "none",
                    "has_tables": has_tables,
                    "word_count": len(text.split()),
                },
            ))
    else:
        # Fallback: treat the whole document as one block split by form-feed markers
        sections = full_md.split("\f") if "\f" in full_md else [full_md]
        for i, section in enumerate(sections, start=1):
            text = section.strip()
            if not text:
                continue
            has_tables = "| --- |" in text or "---|" in text or "|:---" in text
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": i,
                    "extraction_method": "docling",
                    "ocr_model": DOCLING_OCR_MODEL_NAME if do_ocr else "none",
                    "has_tables": has_tables,
                    "word_count": len(text.split()),
                },
            ))

    logger.info("Docling extracted %d page-documents from %s.", len(docs), pdf_path.name)

    if use_cache and docs:
        _save_cache(cache_file, docs)

    return docs


def load_all_pdfs(
    pdf_dir: str | Path,
    use_cache: bool = True,
    do_ocr: bool = DOCLING_DO_OCR,
) -> List[Document]:
    """Load every .pdf in `pdf_dir` using Docling."""
    pdf_dir = Path(pdf_dir)
    all_docs: List[Document] = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)

    for pdf_file in pdf_files:
        all_docs.extend(load_pdf_with_docling(pdf_file, use_cache=use_cache, do_ocr=do_ocr))

    return all_docs
