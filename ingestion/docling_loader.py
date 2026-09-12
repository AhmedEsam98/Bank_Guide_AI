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

Table-of-contents filtering: pages that look like a TOC (dense with
dot-leaders or pipe-table rows mapping a title to a page number) are
tagged with metadata["is_toc"] = True and excluded from load_all_pdfs()'s
output. TOC pages mention every section title in the document, so if left
in the retrieval index they act as false attractors -- scoring as
"relevant" to almost any query and crowding out the actual content page.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.documents import Document

from config import (
    OCR_CACHE_DIR,
    OCR_DPI,
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


def _get_converter(do_ocr: bool = DOCLING_DO_OCR, force_fast_mode: bool = False, force_full_page_ocr: bool = False, dpi_override: int | None = None, force_cpu: bool = False):
    global _converters
    cache_key = (do_ocr, force_fast_mode, force_full_page_ocr, dpi_override, force_cpu)
    if cache_key in _converters:
        return _converters[cache_key]

    # IMPORTANT: only ever keep ONE converter's models loaded at a time.
    # Each distinct config (do_ocr/force_fast_mode/force_full_page_ocr/
    # dpi_override/force_cpu) loads its own full set of EasyOCR + layout +
    # table-structure models onto the GPU. A single load_pdf_with_docling
    # call can use up to 5 different configs across its fallback tiers
    # (main pass, FAST fallback, full-page-OCR-for-tables, low-DPI, CPU),
    # and previously every one of those stayed cached forever, so VRAM
    # usage only ever went up across a run -- on a small (e.g. 4GB) GPU
    # this alone was enough to make later pages/later fallback tiers fail
    # even though each individual model isn't that large. Evicting the
    # previous converter(s) and forcing a real GPU memory release before
    # building the next one keeps peak usage to "one converter's worth"
    # regardless of how many fallback tiers a document needs.
    if _converters:
        import gc
        import torch
        logger.info(
            "Evicting %d previously-cached converter(s) before loading a "
            "new config, to free GPU/CPU memory (config changed to %s).",
            len(_converters), cache_key,
        )
        _converters.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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
    # Picture image generation is memory-heavy; skip it on the low-memory
    # fallback pass (used to retry pages that OOM'd on the first pass).
    pipeline_opts.generate_picture_images = not force_fast_mode

    # Docling renders pages to images internally for OCR at a resolution
    # controlled by `images_scale`, where 1.0 = Docling's baseline of 72
    # DPI. OCR_DPI is expressed in real DPI (e.g. 300), so convert it to
    # the scale factor Docling expects. Low DPI blurs small text in dense
    # tables, causing OCR misreads (U+FFFD substitutions) and reading-
    # order confusion, but high DPI costs proportionally more memory
    # (roughly with the square of the DPI, since it scales both
    # dimensions of the rendered page image). dpi_override lets a final,
    # lowest-memory fallback tier trade quality for simply not losing the
    # page's content entirely when even the FAST-mode fallback OOMs.
    if dpi_override is not None:
        effective_dpi = dpi_override
    elif force_full_page_ocr:
        effective_dpi = 150
    elif force_fast_mode:
        effective_dpi = 150
    else:
        effective_dpi = OCR_DPI
    pipeline_opts.images_scale = effective_dpi / 72.0

    if bool(do_ocr):
        pipeline_opts.do_ocr = True
        # Explicitly configure OCR model: EasyOCR (CRAFT detection + CRNN Arabic/English recognition)
        # force_full_page_ocr: some pages (esp. dense Arabic tables) have a
        # native embedded text layer built with a broken/non-standard font
        # encoding (missing or wrong ToUnicode CMap -- a known issue with
        # many Arabic Word-to-PDF exporters). Docling's default behavior
        # trusts that native text layer wherever it thinks one exists and
        # skips OCR for it entirely, which means images_scale/DPI has no
        # effect there. Forcing full-page OCR makes Docling always render
        # the page and recognize text via EasyOCR pixel-level detection,
        # bypassing any broken native encoding -- but it costs noticeably
        # more time/memory, so it's only applied selectively (see
        # load_pdf_with_docling's quality-check retry pass) rather than
        # globally for every page of every PDF.
        pipeline_opts.ocr_options = EasyOcrOptions(
            lang=DOCLING_OCR_LANGUAGES,
            use_gpu=(torch.cuda.is_available() and not force_cpu),
            force_full_page_ocr=force_full_page_ocr,
        )
        logger.info(
            "Configured OCR model: %s (engine=%s, languages=%s, force_full_page_ocr=%s)",
            DOCLING_OCR_MODEL_NAME, DOCLING_OCR_ENGINE, DOCLING_OCR_LANGUAGES, force_full_page_ocr,
        )
    else:
        pipeline_opts.do_ocr = False
        logger.info("OCR disabled (digital text extraction only).")

    # Low-memory fallback pass uses fewer threads to reduce peak concurrent
    # memory usage (this is the pass used to recover pages that failed with
    # std::bad_alloc on the first, higher-memory pass).
    thread_count = 2 if force_fast_mode else 4

    use_cuda = torch.cuda.is_available() and not force_cpu
    if use_cuda:
        pipeline_opts.accelerator_options = AcceleratorOptions(
            num_threads=thread_count,
            device=AcceleratorDevice.CUDA,
        )
        logger.info("Docling using CUDA acceleration (do_ocr=%s, force_fast_mode=%s).", do_ocr, force_fast_mode)
    else:
        pipeline_opts.accelerator_options = AcceleratorOptions(num_threads=thread_count)
        logger.info(
            "Docling using CPU (do_ocr=%s, force_fast_mode=%s, force_cpu=%s).",
            do_ocr, force_fast_mode, force_cpu,
        )

    if force_fast_mode or DOCLING_MODE != "accurate":
        pipeline_opts.table_structure_options.mode = TableFormerMode.FAST
    else:
        pipeline_opts.table_structure_options.mode = TableFormerMode.ACCURATE

    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    _converters[cache_key] = converter
    logger.info("Docling DocumentConverter ready (mode=%s, ocr_model=%s).", DOCLING_MODE, DOCLING_OCR_MODEL_NAME if do_ocr else "none")
    return converter


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_path(pdf_path: Path) -> Path:
    file_hash = hashlib.md5(pdf_path.read_bytes()).hexdigest()
    return OCR_CACHE_DIR / f"docling_{pdf_path.stem}_{file_hash}.json"


def _clean_markdown_text(text: str) -> str:
    """Strip Docling missing image boilerplate comments, collapse extra
    newlines, and normalize Arabic text.

    NFKC normalization is important here: EasyOCR (and some PDF text
    layers with non-standard font encodings) can emit Arabic Presentation
    Form glyphs (U+FB50-U+FEFF -- isolated/initial/medial/final glyph
    shapes meant for rendering) instead of standard logical Arabic
    characters (U+0600-U+06FF). Left uncorrected, this text looks
    garbled/reversed everywhere downstream (chunking, embeddings, and
    especially in retrieved excerpts shown to the user), even though the
    underlying reading order is usually fine. NFKC maps each presentation
    form back to its standard logical-form equivalent.
    """
    text = re.sub(r"<!--\s*🖼️?[^>]*PdfPipelineOptions[^>]*-->", "", text)
    # Strip embedded base64 image data (Docling's picture fallback when it
    # can't extract text from an image element). These blobs are pure
    # noise for retrieval -- no searchable content -- but at hundreds of
    # KB each they get sliced into dozens of meaningless chunks by the
    # downstream chunker, inflating the index and wasting embedding
    # compute for zero retrieval value.
    text = re.sub(r"!\[Image\]\(data:image/[^)]+\)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = unicodedata.normalize("NFKC", text)
    return text.strip()


def _looks_corrupted(text: str) -> bool:
    """Detect genuine OCR/text-extraction corruption: literal U+FFFD
    replacement characters, which mean some character truly could not be
    decoded/recognized (unlike encoding-normalization issues, this is not
    fixable via NFKC -- the only real fix is re-extracting via a
    different path, e.g. force_full_page_ocr to bypass a broken native
    PDF text layer). A small handful can be an isolated OCR misread; a
    meaningful density indicates the whole page's extraction is broken."""
    if not text:
        return False
    replacement_count = text.count("\ufffd")
    if replacement_count == 0:
        return False
    return replacement_count >= 3 or (replacement_count / max(1, len(text))) > 0.002


def _has_scrambled_arabic(text: str) -> bool:
    """Detect if Arabic text has reversed/scrambled character order or broken ligatures.
    Common symptoms in PDFs exported with broken RTL glyph mapping:
      - Taa Marbuta (ة) followed by space/period at the start of a token (e.g. 'ة. ايدار' for 'إدارة', 'ة دائر' for 'دائرة')
      - Definite article prefix (ال) appearing detached or as a suffix (e.g. 'افقة المو', 'كزي المر')
    """
    if not text:
        return False
    if re.search(r"(?:^|\s)ة[\.\s]+[\u0600-\u06FF]", text):
        return True
    if re.search(r"[\u0600-\u06FF]+\s+(?:المر|المو|الاد|الان)\b", text):
        return True
    return False


def _looks_like_toc(text: str) -> bool:
    """Heuristic: does this page look like a table of contents rather than
    actual procedure content? TOC pages are dense with dot-leaders
    ('....') connecting a title to a page number (often inside a
    markdown table with many dots per cell), and/or contain an explicit
    "Contents" heading. Including them in the retrieval index causes
    them to act as false attractors -- since they mention EVERY section
    title in the document, they can score as relevant to almost any
    query, crowding out the actual content page."""
    if not text or len(text) < 20:
        return False

    stripped = text.strip()

    dot_leader_chars = len(re.findall(r"\.{3,}", stripped))
    total_dots = stripped.count(".")

    trailing_page_number_rows = len(re.findall(r"\d{1,3}\s*\|", stripped))

    has_toc_heading = bool(re.search(
        r"(حﺘﻮ|المحتويات|الفهرس|جدول المحتويات|Table of Contents|"
        r"^Contents\b)",
        stripped,
    ))

    dot_density = total_dots / max(1, len(stripped))

    return (
        has_toc_heading
        or dot_leader_chars >= 5
        or (dot_density > 0.05 and trailing_page_number_rows >= 3)
    )


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
def _convert_single_pass(pdf_path: Path, converter, do_ocr: bool) -> List[Document]:
    """Run one Docling conversion pass and return the resulting per-page
    Documents. Raises if Docling itself raises; pages that fail inside
    Docling's own internal stages (e.g. std::bad_alloc on a preprocess
    stage) are simply absent from the result -- that's the case the
    caller (load_pdf_with_docling) checks for and retries."""
    t0 = time.time()
    result = converter.convert(str(pdf_path))
    elapsed = time.time() - t0
    logger.info("Docling finished %s in %.1fs.", pdf_path.name, elapsed)

    full_md = result.document.export_to_markdown()

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
                    "is_toc": _looks_like_toc(text),
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
                    "is_toc": _looks_like_toc(text),
                },
            ))

    return docs


def _extract_single_page_pdf(pdf_path: Path, page_no: int, out_path: Path) -> bool:
    """Slice a single 1-indexed page from a PDF into a temporary standalone PDF."""
    try:
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(str(pdf_path))
        if page_no < 1 or page_no > len(reader.pages):
            return False
        writer = PdfWriter()
        writer.add_page(reader.pages[page_no - 1])
        with open(out_path, "wb") as f:
            writer.write(f)
        return True
    except Exception as exc:
        logger.warning("Failed to slice page %d from %s: %s", page_no, pdf_path.name, exc)
        return False


def _convert_single_page(pdf_path: Path, page_no: int, converter, do_ocr: bool) -> Document | None:
    """Convert a single page of a PDF by isolating it into a 1-page PDF.
    
    This avoids whole-document memory buildup (preventing std::bad_alloc on
    heavy pages) and executes in seconds rather than reprocessing the entire document.
    """
    import gc
    import tempfile
    import torch

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if not _extract_single_page_pdf(pdf_path, page_no, tmp_path):
            return None

        t0 = time.time()
        result = converter.convert(str(tmp_path))
        elapsed = time.time() - t0
        logger.info("Docling converted page %d of %s in %.1fs.", page_no, pdf_path.name, elapsed)

        text_parts: list[str] = []
        for item, _ in result.document.iterate_items():
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
                text_parts.append(txt.strip())

        if text_parts:
            text = _clean_markdown_text("\n\n".join(text_parts))
        else:
            text = _clean_markdown_text(result.document.export_to_markdown())

        if not text:
            return None

        has_tables = "| --- |" in text or "---|" in text or "|:---" in text
        return Document(
            page_content=text,
            metadata={
                "source": pdf_path.name,
                "page": page_no,
                "extraction_method": "docling",
                "ocr_model": DOCLING_OCR_MODEL_NAME if do_ocr else "none",
                "has_tables": has_tables,
                "word_count": len(text.split()),
                "is_toc": _looks_like_toc(text),
            },
        )
    except Exception as exc:
        logger.warning("Failed to convert single page %d of %s: %s", page_no, pdf_path.name, exc)
        return None
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _get_pdf_page_count(pdf_path: Path) -> int | None:
    """Best-effort real page count of the PDF, used to detect pages that
    Docling silently dropped (e.g. due to std::bad_alloc on a preprocess
    stage). Returns None if pypdf isn't available -- in that case missing-
    page detection/retry is simply skipped."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    except Exception as exc:
        logger.warning(
            "Could not determine real page count for %s (%s) -- skipping "
            "missing-page detection/retry. Install 'pypdf' to enable it.",
            pdf_path.name, exc,
        )
        return None


def load_pdf_with_docling(
    pdf_path: str | Path,
    use_cache: bool = True,
    do_ocr: bool = DOCLING_DO_OCR,
) -> List[Document]:
    """Convert one PDF with Docling and return one Document per page.

    If any pages come back missing after the first pass (Docling can
    silently drop a page if a native preprocessing stage OOMs -- this
    happens with std::bad_alloc on memory-heavy pages, e.g. large
    embedded images or complex tables under TableFormerMode.ACCURATE),
    this automatically retries ONLY the conversion (Docling doesn't
    support true per-page re-conversion, so the whole document is
    reprocessed) using a lighter-weight, lower-memory pipeline
    (TableFormerMode.FAST, generate_picture_images=False, fewer threads),
    and merges in whichever previously-missing pages recover. Pages that
    succeeded on the first (higher-quality) pass are kept as-is.

    A separate, later pass checks every extracted page for genuine OCR/
    extraction corruption (literal U+FFFD replacement characters, which
    indicate a broken native PDF text layer -- common with Arabic
    Word-to-PDF exporters using non-standard font encodings). If any
    pages are corrupted, the whole document is reprocessed once more
    with force_full_page_ocr=True (forces genuine pixel-level OCR,
    bypassing the broken native text layer), and only the previously-
    corrupted pages are swapped in from that pass. This keeps the fast
    default path for the vast majority of pages/documents that don't
    need it, while still fixing the ones that do.

    Each Document has metadata:
        source, page, extraction_method, has_tables, word_count, is_toc
    """
    pdf_path = Path(pdf_path)
    cache_file = _cache_path(pdf_path)

    if use_cache:
        cached = _load_cache(cache_file)
        if cached is not None:
            logger.info("Loaded Docling cache for %s (%d pages).", pdf_path.name, len(cached))
            return cached

    logger.info("Running Docling on %s (do_ocr=%s) …", pdf_path.name, do_ocr)

    try:
        converter = _get_converter(do_ocr=do_ocr)
        docs = _convert_single_pass(pdf_path, converter, do_ocr)
    except Exception as exc:
        logger.error("Docling failed on %s: %s", pdf_path.name, exc)
        return []

    # --- Detect and retry pages that silently dropped out (e.g. OOM) -----
    real_page_count = _get_pdf_page_count(pdf_path)
    if real_page_count is not None:
        found_pages = {d.metadata["page"] for d in docs}
        expected_pages = set(range(1, real_page_count + 1))
        missing_pages = sorted(expected_pages - found_pages)

        if missing_pages:
            logger.warning(
                "%s: %d page(s) missing after first pass (likely OOM'd "
                "internally): %s. Retrying individually with a lighter-weight pipeline "
                "(FAST table mode, no picture rendering, fewer threads)...",
                pdf_path.name, len(missing_pages), missing_pages,
            )
            try:
                fallback_converter = _get_converter(do_ocr=do_ocr, force_fast_mode=True)
                recovered = 0
                for page_no in missing_pages:
                    doc = _convert_single_page(pdf_path, page_no, fallback_converter, do_ocr)
                    if doc is not None:
                        docs.append(doc)
                        recovered += 1

                still_missing = [p for p in missing_pages if p not in {d.metadata["page"] for d in docs}]
                if recovered:
                    logger.info(
                        "%s: recovered %d/%d previously-missing page(s) via "
                        "single-page fallback pass.", pdf_path.name, recovered, len(missing_pages),
                    )
                if still_missing:
                    logger.warning(
                        "%s: %d page(s) still missing after the FAST-mode "
                        "fallback: %s. Trying at lower DPI (120)...",
                        pdf_path.name, len(still_missing), still_missing,
                    )
                    try:
                        low_dpi_converter = _get_converter(do_ocr=do_ocr, force_fast_mode=True, dpi_override=120)
                        for page_no in list(still_missing):
                            doc = _convert_single_page(pdf_path, page_no, low_dpi_converter, do_ocr)
                            if doc is not None:
                                docs.append(doc)
                                still_missing.remove(page_no)
                    except Exception as exc:
                        logger.error(
                            "%s: low-DPI fallback pass failed: %s.",
                            pdf_path.name, exc,
                        )

                if still_missing:
                    logger.warning(
                        "%s: %d page(s) still missing even at low DPI: %s. "
                        "Trying on CPU...",
                        pdf_path.name, len(still_missing), still_missing,
                    )
                    try:
                        cpu_converter = _get_converter(do_ocr=do_ocr, force_fast_mode=True, dpi_override=150, force_cpu=True)
                        for page_no in list(still_missing):
                            doc = _convert_single_page(pdf_path, page_no, cpu_converter, do_ocr)
                            if doc is not None:
                                docs.append(doc)
                                still_missing.remove(page_no)
                    except Exception as exc:
                        logger.error(
                            "%s: CPU fallback pass failed: %s.",
                            pdf_path.name, exc,
                        )

                if still_missing:
                    logger.error(
                        "%s: %d page(s) STILL missing after all fallback tiers: %s.",
                        pdf_path.name, len(still_missing), still_missing,
                    )
            except Exception as exc:
                logger.error(
                    "%s: fallback retry pass itself failed: %s. Missing "
                    "pages %s could not be recovered.",
                    pdf_path.name, exc, missing_pages,
                )

    docs.sort(key=lambda d: d.metadata["page"])
    logger.info("Docling extracted %d page-documents from %s.", len(docs), pdf_path.name)

    # --- Detect and fix genuinely corrupted / unreliable pages via ------
    # --- targeted full-page OCR ------------------------------------------
    if do_ocr:
        # Two signals trigger targeted single-page re-OCR:
        #  1. Literal U+FFFD replacement characters (genuine decode failure).
        #  2. Scrambled Arabic detection (_has_scrambled_arabic): detects reversed
        #     glyph sequences (e.g. 'ة. ايدار', 'كزي المر') caused by broken RTL font mappings.
        pages_needing_fix = [
            d.metadata["page"] for d in docs
            if _looks_corrupted(d.page_content) or _has_scrambled_arabic(d.page_content)
        ]
        if pages_needing_fix:
            logger.warning(
                "%s: %d page(s) need targeted full-page OCR (corruption/scrambled "
                "Arabic detected): %s. Re-processing these specific pages individually "
                "with force_full_page_ocr=True (150 DPI, CPU mode)...",
                pdf_path.name, len(pages_needing_fix), pages_needing_fix,
            )
            try:
                # Use CPU mode to avoid GPU stalling/hanging on per-page OCR
                full_ocr_converter = _get_converter(
                    do_ocr=do_ocr, force_full_page_ocr=True,
                    dpi_override=150, force_cpu=True,
                )
                fixed = 0
                still_bad = []
                docs_by_page = {d.metadata["page"]: d for d in docs}
                total_fix = len(pages_needing_fix)
                for idx, page_no in enumerate(pages_needing_fix, 1):
                    logger.info(
                        "%s: OCR-fixing page %d (%d/%d)...",
                        pdf_path.name, page_no, idx, total_fix,
                    )
                    sys.stderr.flush()
                    sys.stdout.flush()
                    candidate = _convert_single_page(pdf_path, page_no, full_ocr_converter, do_ocr)
                    original = docs_by_page.get(page_no)
                    if candidate is not None and not _looks_corrupted(candidate.page_content):
                        docs_by_page[page_no] = candidate
                        fixed += 1
                        logger.info(
                            "%s: page %d fixed successfully (%d words).",
                            pdf_path.name, page_no, len(candidate.page_content.split()),
                        )
                    elif candidate is not None and original is not None:
                        if candidate.page_content.count("\ufffd") < original.page_content.count("\ufffd"):
                            docs_by_page[page_no] = candidate
                            fixed += 1
                        else:
                            still_bad.append(page_no)
                    elif candidate is not None:
                        docs_by_page[page_no] = candidate
                        fixed += 1
                    else:
                        still_bad.append(page_no)
                    sys.stderr.flush()
                    sys.stdout.flush()

                docs = sorted(docs_by_page.values(), key=lambda d: d.metadata["page"])

                if fixed:
                    logger.info(
                        "%s: applied targeted full-page OCR to %d/%d page(s).",
                        pdf_path.name, fixed, len(pages_needing_fix),
                    )
                if still_bad:
                    logger.warning(
                        "%s: %d page(s) could not be improved via targeted full-page OCR: %s.",
                        pdf_path.name, len(still_bad), still_bad,
                    )
            except Exception as exc:
                logger.error(
                    "%s: targeted full-page-OCR retry failed: %s.",
                    pdf_path.name, exc,
                )

    if use_cache and docs:
        _save_cache(cache_file, docs)

    return docs


def load_all_pdfs(
    pdf_dir: str | Path,
    use_cache: bool = True,
    do_ocr: bool = DOCLING_DO_OCR,
) -> List[Document]:
    """Load every .pdf in `pdf_dir` using Docling.

    Pages flagged as table-of-contents (metadata["is_toc"] == True) are
    filtered out here, after loading/caching, so the cache still keeps
    the original TOC pages on disk (useful for debugging / inspection)
    while the pipeline (chunking, embedding, retrieval) never sees them.
    """
    pdf_dir = Path(pdf_dir)
    all_docs: List[Document] = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)

    for pdf_file in pdf_files:
        all_docs.extend(load_pdf_with_docling(pdf_file, use_cache=use_cache, do_ocr=do_ocr))

    toc_count = sum(1 for d in all_docs if d.metadata.get("is_toc"))
    if toc_count:
        logger.info(
            "Filtering out %d likely table-of-contents page(s) from %d total pages.",
            toc_count, len(all_docs),
        )
        all_docs = [d for d in all_docs if not d.metadata.get("is_toc")]

    return all_docs