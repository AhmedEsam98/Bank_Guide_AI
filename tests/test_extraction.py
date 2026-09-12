"""
test_extraction.py
--------------------
Inspect what the ingestion loader actually extracted from your PDFs,
page by page, BEFORE it gets chunked or embedded. Useful for catching:
  - pages that came back empty/near-empty (OOM'd, scanned badly, etc.)
  - pages that fell back to OCR vs native text extraction
  - garbled Arabic OCR output
  - which pages are missing entirely from the ingested corpus

Usage:
    python test_extraction.py
    python test_extraction.py --pdf-dir data/pdfs
    python test_extraction.py --min-chars 40 --output extraction_report.md

Output: a Markdown report (default: extraction_report.md) with one section
per PDF, a per-page table (page #, char count, extraction method, first
120 chars preview), and a summary of flagged/short/missing pages.

NOTE: this imports your project's PDF loader. If your ingestion pipeline
uses a different module/function name than `ingestion.ocr_loader.load_all_pdfs`
(e.g. you've since switched to a docling-based loader), just edit the
import block below -- everything else in this script only needs a list
of LangChain `Document` objects with `.page_content` and metadata keys
`source` / `page` (and optionally `extraction_method`).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DATA_DIR
from ingestion.docling_loader import load_all_pdfs


def _preview(text: str, n: int = 120) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:n] + ("…" if len(cleaned) > n else "")


def build_report(pdf_dir: Path, min_chars: int, use_cache: bool) -> str:
    print(f"Loading & extracting PDFs from {pdf_dir} ...")
    docs = load_all_pdfs(pdf_dir, use_cache=use_cache)
    print(f"Extracted {len(docs)} page-documents.\n")

    by_source: dict[str, List] = defaultdict(list)
    for d in docs:
        by_source[d.metadata.get("source", "unknown")].append(d)

    lines = [
        "# PDF Extraction Report",
        "",
        f"- **PDF directory**: `{pdf_dir}`",
        f"- **Total page-documents extracted**: `{len(docs)}`",
        f"- **Flag threshold**: pages with fewer than `{min_chars}` characters are flagged",
        "",
        "---",
        "",
    ]

    all_flagged = []

    for source, pages in sorted(by_source.items()):
        pages_sorted = sorted(pages, key=lambda d: d.metadata.get("page", 0))
        page_numbers = [d.metadata.get("page") for d in pages_sorted]

        # detect gaps in the page sequence (missing pages)
        if page_numbers and all(isinstance(p, int) for p in page_numbers):
            expected = set(range(min(page_numbers), max(page_numbers) + 1))
            missing = sorted(expected - set(page_numbers))
        else:
            missing = []

        lines.append(f"## {source}")
        lines.append("")
        lines.append(f"- Pages extracted: `{len(pages_sorted)}`")
        lines.append(f"- Missing page numbers (gaps in sequence): `{missing or 'none'}`")
        lines.append("")
        lines.append("| Page | Chars | Method | Preview |")
        lines.append("|---|---|---|---|")

        for d in pages_sorted:
            page = d.metadata.get("page", "?")
            method = d.metadata.get("extraction_method", "?")
            n_chars = len(d.page_content)
            flag = " ⚠️" if n_chars < min_chars else ""
            if flag:
                all_flagged.append((source, page, n_chars, method))
            lines.append(f"| {page}{flag} | {n_chars} | {method} | {_preview(d.page_content)} |")

        if missing:
            for m in missing:
                lines.append(f"| {m} | 0 | **MISSING** | _(not present in extracted corpus)_ |")
                all_flagged.append((source, m, 0, "MISSING"))

        lines.append("")
        lines.append("---")
        lines.append("")

    lines.insert(
        7,
        f"- **Flagged pages (short or missing)**: `{len(all_flagged)}`",
    )

    if all_flagged:
        lines.append("## ⚠️ Flagged pages summary")
        lines.append("")
        lines.append("| Source | Page | Chars | Method |")
        lines.append("|---|---|---|---|")
        for source, page, n_chars, method in all_flagged:
            lines.append(f"| {source} | {page} | {n_chars} | {method} |")
        lines.append("")

    return "\n".join(lines)


def _parse_args():
    parser = argparse.ArgumentParser(description="Inspect raw PDF text extraction quality.")
    parser.add_argument("--pdf-dir", default=str(DATA_DIR))
    parser.add_argument("--min-chars", type=int, default=40, help="Flag pages with fewer chars than this")
    parser.add_argument("--no-cache", action="store_true", help="Disable OCR cache reuse")
    parser.add_argument("--output", default="extraction_report.md")
    return parser.parse_args()


def main():
    args = _parse_args()
    report = build_report(Path(args.pdf_dir), args.min_chars, use_cache=not args.no_cache)
    tests_dir = Path(__file__).resolve().parent
    output_path = Path(args.output) if Path(args.output).is_absolute() else tests_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Done. Report saved to: {output_path}")


if __name__ == "__main__":
    main()
