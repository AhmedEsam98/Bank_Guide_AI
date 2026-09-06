"""
ingestion/benchmark_ocr.py
---------------------------
Comprehensive benchmark for Docling OCR & Data Extraction pipeline:
  1. Extraction performance & throughput (time, pages/sec)
  2. Language & quality metrics (Arabic %, English %, word counts)
  3. Layout & structure preservation (tables detected and formatted)
  4. Chunking efficiency & size distribution
  5. End-to-end retrieval validation on sample banking SOP queries

Usage:
    python -m ingestion.benchmark_ocr                  # full extraction + chunking benchmark
    python -m ingestion.benchmark_ocr --retrieval      # also run test retrieval queries
    python -m ingestion.benchmark_ocr --no-cache       # force fresh OCR extraction without cache
    python -m ingestion.benchmark_ocr --pdf path.pdf   # benchmark single file
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DATA_DIR, EVAL_OUTPUT_DIR, DOCLING_DO_OCR
from ingestion.docling_loader import load_pdf_with_docling
from ingestion.chunking import chunk_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_arabic_char(c: str) -> bool:
    """Check if unicode character belongs to Arabic script block."""
    code = ord(c)
    return (
        0x0600 <= code <= 0x06FF  # Arabic
        or 0x0750 <= code <= 0x077F  # Arabic Supplement
        or 0x08A0 <= code <= 0x08FF  # Arabic Extended-A
        or 0xFB50 <= code <= 0xFDFF  # Arabic Presentation Forms-A
        or 0xFE70 <= code <= 0xFEFF  # Arabic Presentation Forms-B
    )


def compute_text_metrics(text: str) -> Dict[str, Any]:
    """Calculate character counts, word counts, Arabic/English percentage, and tables."""
    total_chars = len(text)
    if total_chars == 0:
        return {
            "total_chars": 0,
            "total_words": 0,
            "arabic_pct": 0.0,
            "english_pct": 0.0,
            "table_count": 0,
        }

    arabic_chars = sum(1 for c in text if is_arabic_char(c))
    english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    words = text.split()
    table_count = text.count("\n|")  # Count markdown table rows

    return {
        "total_chars": total_chars,
        "total_words": len(words),
        "arabic_pct": round((arabic_chars / total_chars) * 100, 2),
        "english_pct": round((english_chars / total_chars) * 100, 2),
        "table_rows": table_count,
    }


def benchmark_extraction(
    pdf_paths: List[Path],
    use_cache: bool = True,
    do_ocr: bool = DOCLING_DO_OCR,
) -> Dict[str, Any]:
    """Run Docling extraction on PDF files and measure quality metrics."""
    print("\n" + "=" * 80)
    print(f"  DOCLING OCR & EXTRACTION BENCHMARK (do_ocr={do_ocr})")
    print("=" * 80)

    pdf_results = []
    total_start = time.time()
    total_pages = 0
    total_words = 0
    total_chars = 0
    total_arabic_chars = 0
    total_table_rows = 0

    for pdf in pdf_paths:
        print(f"\n📂 Processing: {pdf.name}")
        t0 = time.time()
        docs = load_pdf_with_docling(pdf, use_cache=use_cache, do_ocr=do_ocr)
        elapsed = time.time() - t0

        pages_count = len(docs)
        combined_text = "\n\n".join(d.page_content for d in docs)
        metrics = compute_text_metrics(combined_text)

        total_pages += pages_count
        total_words += metrics["total_words"]
        total_chars += metrics["total_chars"]
        total_arabic_chars += sum(1 for c in combined_text if is_arabic_char(c))
        total_table_rows += metrics["table_rows"]

        words_per_page = round(metrics["total_words"] / max(pages_count, 1), 1)
        sec_per_page = round(elapsed / max(pages_count, 1), 2)

        file_stat = {
            "file_name": pdf.name,
            "pages": pages_count,
            "extraction_time_sec": round(elapsed, 2),
            "sec_per_page": sec_per_page,
            "words": metrics["total_words"],
            "words_per_page": words_per_page,
            "arabic_pct": metrics["arabic_pct"],
            "english_pct": metrics["english_pct"],
            "table_rows": metrics["table_rows"],
        }
        pdf_results.append(file_stat)

        print(f"   ├─ Pages: {pages_count} | Extraction Time: {elapsed:.2f}s ({sec_per_page}s/page)")
        print(f"   ├─ Words: {metrics['total_words']:,} (avg {words_per_page} words/page)")
        print(f"   ├─ Language Balance: {metrics['arabic_pct']}% Arabic | {metrics['english_pct']}% English")
        print(f"   └─ Markdown Table Rows: {metrics['table_rows']}")

    total_time = time.time() - total_start
    overall_arabic_pct = round((total_arabic_chars / max(total_chars, 1)) * 100, 2)

    summary = {
        "total_files": len(pdf_paths),
        "total_pages": total_pages,
        "total_words": total_words,
        "total_chars": total_chars,
        "overall_arabic_pct": overall_arabic_pct,
        "total_table_rows": total_table_rows,
        "total_time_sec": round(total_time, 2),
        "avg_sec_per_page": round(total_time / max(total_pages, 1), 2),
        "do_ocr": do_ocr,
        "files": pdf_results,
    }

    return summary


def benchmark_chunking(pdf_paths: List[Path], do_ocr: bool = DOCLING_DO_OCR) -> Dict[str, Any]:
    """Evaluate chunking distribution across Docling extracted pages."""
    print("\n" + "=" * 80)
    print("  DOCLING CHUNKING & STRUCTURAL DISTRIBUTION")
    print("=" * 80)

    # Load all pages via cache
    all_docs = []
    for pdf in pdf_paths:
        all_docs.extend(load_pdf_with_docling(pdf, use_cache=True, do_ocr=do_ocr))

    strategies = ["recursive_character", "markdown_heading", "arabic_paragraph"]
    chunk_results = {}

    for strat in strategies:
        chunks = chunk_documents(all_docs, strategy=strat, chunk_size=800, chunk_overlap=100)
        lengths = [len(c.page_content) for c in chunks] if chunks else [0]
        word_counts = [len(c.page_content.split()) for c in chunks] if chunks else [0]

        stat = {
            "strategy": strat,
            "chunk_count": len(chunks),
            "avg_char_length": round(sum(lengths) / max(len(lengths), 1), 1),
            "min_char_length": min(lengths),
            "max_char_length": max(lengths),
            "avg_word_count": round(sum(word_counts) / max(len(word_counts), 1), 1),
        }
        chunk_results[strat] = stat
        print(f"\n Strategy: {strat}")
        print(f"   ├─ Chunks: {stat['chunk_count']}")
        print(f"   ├─ Avg Character Length: {stat['avg_char_length']} (min: {stat['min_char_length']}, max: {stat['max_char_length']})")
        print(f"   └─ Avg Words/Chunk: {stat['avg_word_count']}")

    return chunk_results


def benchmark_retrieval() -> List[Dict[str, Any]]:
    """Validate retrieval accuracy with Docling indexed chunks."""
    print("\n" + "=" * 80)
    print("  RETRIEVAL VALIDATION ON DOCLING VECTOR STORE")
    print("=" * 80)

    from retrieval.retriever import get_retriever

    test_queries = [
        "ما هي إجراءات التعامل مع البريد السري وطرود كبار العملاء؟",
        "ما هي شروط وإجراءات تفريغ كاميرات المراقبة CCTV؟",
        "ما هي معايير الجرد الدوري وإتلاف الموجودات الثابتة في المستودعات؟",
    ]

    retriever = get_retriever(mode="hybrid", top_k=3)
    query_results = []

    for q in test_queries:
        print(f"\n🔍 Query: {q}")
        t0 = time.time()
        docs = retriever.invoke(q)
        latency = time.time() - t0
        print(f"   Retrieved {len(docs)} chunks in {latency:.2f}s:")

        top_sources = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "?")
            top_sources.append({"source": src, "page": page})
            snippet = d.page_content[:120].replace("\n", " ")
            print(f"     [{i}] {src} (p.{page}) -> {snippet}...")

        query_results.append({
            "query": q,
            "latency_sec": round(latency, 2),
            "retrieved_count": len(docs),
            "top_sources": top_sources,
        })

    return query_results


def run_benchmark(
    pdf_path: Path | None = None,
    use_cache: bool = True,
    do_ocr: bool = DOCLING_DO_OCR,
    run_retrieval_test: bool = False,
    save_report: bool = True,
) -> Dict[str, Any]:
    """Execute complete benchmark suite and save JSON report."""
    if pdf_path:
        pdf_files = [pdf_path]
    else:
        pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in data/pdfs.")
        return {}

    extraction_summary = benchmark_extraction(pdf_files, use_cache=use_cache, do_ocr=do_ocr)
    chunking_summary = benchmark_chunking(pdf_files, do_ocr=do_ocr)

    retrieval_summary = []
    if run_retrieval_test:
        retrieval_summary = benchmark_retrieval()

    final_report = {
        "engine": "Docling (Accurate Mode)",
        "do_ocr": do_ocr,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "extraction": extraction_summary,
        "chunking": chunking_summary,
        "retrieval": retrieval_summary,
    }

    if save_report:
        out_path = EVAL_OUTPUT_DIR / "docling_benchmark.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Benchmark report saved to: {out_path}")

    return final_report


def _parse_args():
    parser = argparse.ArgumentParser(description="Docling OCR & Extraction Benchmark")
    parser.add_argument("--pdf", help="Path to a single PDF to benchmark")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh extraction without cache")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR (benchmark digital text extraction only)")
    parser.add_argument("--do-ocr", action="store_true", help="Enable OCR for scanned pages")
    parser.add_argument("--retrieval", action="store_true", help="Include retrieval validation test")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    target_pdf = Path(args.pdf) if args.pdf else None
    do_ocr_val = False if args.no_ocr else (True if args.do_ocr else DOCLING_DO_OCR)
    run_benchmark(
        pdf_path=target_pdf,
        use_cache=not args.no_cache,
        do_ocr=do_ocr_val,
        run_retrieval_test=args.retrieval,
    )
