"""
test_chunking.py
-------------------
Run every chunking strategy from ingestion/chunking.py over the same
extracted PDF text and compare them side by side: number of chunks,
average/min/max chunk length, and a handful of sample chunks per
strategy -- so you can eyeball which strategy actually produces
sensible, self-contained chunks for these Arabic procedure manuals
before you commit to one for ingestion.

Usage:
    python test_chunking.py
    python test_chunking.py --strategy arabic_paragraph --chunk-size 800 --chunk-overlap 100
    python test_chunking.py --samples 5 --output chunking_report.md

If --strategy is omitted, ALL strategies in config.CHUNK_STRATEGIES are
run and compared. If given, only that one strategy is run (useful once
you've narrowed it down and want to inspect it in more detail with more
sample chunks).
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import statistics
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import (
    CHUNK_STRATEGIES,
    DATA_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)
from ingestion.docling_loader import load_all_pdfs
from ingestion.chunking import chunk_documents


def _preview(text: str, n: int = 200) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:n] + ("…" if len(cleaned) > n else "")


def _stats_for(chunks) -> dict:
    lengths = [len(c.page_content) for c in chunks]
    if not lengths:
        return {"count": 0, "avg": 0, "min": 0, "max": 0, "stdev": 0}
    return {
        "count": len(lengths),
        "avg": round(statistics.mean(lengths), 1),
        "min": min(lengths),
        "max": max(lengths),
        "stdev": round(statistics.pstdev(lengths), 1) if len(lengths) > 1 else 0,
    }


def build_report(
    pdf_dir: Path,
    strategies: List[str],
    chunk_size: int,
    chunk_overlap: int,
    samples: int,
    use_cache: bool,
) -> str:
    print(f"Loading & extracting PDFs from {pdf_dir} ...")
    raw_docs = load_all_pdfs(pdf_dir, use_cache=use_cache)
    print(f"Loaded {len(raw_docs)} page-documents. Running {len(strategies)} strategy(ies)...\n")

    lines = [
        "# Chunking Strategy Comparison Report",
        "",
        f"- **Source pages**: `{len(raw_docs)}`",
        f"- **Chunk size / overlap**: `{chunk_size}` / `{chunk_overlap}`",
        f"- **Strategies compared**: `{', '.join(strategies)}`",
        "",
        "## Summary",
        "",
        "| Strategy | # Chunks | Avg len | Min len | Max len | Std dev |",
        "|---|---|---|---|---|---|",
    ]

    per_strategy_chunks = {}
    for strategy in strategies:
        print(f"  -> {strategy}")
        chunks = chunk_documents(
            raw_docs, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        per_strategy_chunks[strategy] = chunks
        s = _stats_for(chunks)
        lines.append(
            f"| {strategy} | {s['count']} | {s['avg']} | {s['min']} | {s['max']} | {s['stdev']} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    for strategy, chunks in per_strategy_chunks.items():
        lines.append(f"## Sample chunks — `{strategy}`")
        lines.append("")
        for i, chunk in enumerate(chunks[:samples], start=1):
            src = chunk.metadata.get("source", "?")
            page = chunk.metadata.get("page", "?")
            lines.append(f"**Chunk {i}** (`{src}`, page {page}, {len(chunk.page_content)} chars)")
            lines.append("")
            lines.append("```")
            lines.append(chunk.page_content)
            lines.append("```")
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _parse_args():
    parser = argparse.ArgumentParser(description="Compare chunking strategies side by side.")
    parser.add_argument("--pdf-dir", default=str(DATA_DIR))
    parser.add_argument(
        "--strategy",
        default=None,
        help="Run only this strategy (default: run all strategies in config.CHUNK_STRATEGIES)",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--samples", type=int, default=3, help="Sample chunks to print per strategy")
    parser.add_argument("--no-cache", action="store_true", help="Disable OCR cache reuse")
    parser.add_argument("--output", default="chunking_report.md")
    return parser.parse_args()


def main():
    args = _parse_args()
    strategies = [args.strategy] if args.strategy else list(CHUNK_STRATEGIES)

    report = build_report(
        pdf_dir=Path(args.pdf_dir),
        strategies=strategies,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        samples=args.samples,
        use_cache=not args.no_cache,
    )
    tests_dir = Path(__file__).resolve().parent
    output_path = Path(args.output) if Path(args.output).is_absolute() else tests_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nDone. Report saved to: {output_path}")


if __name__ == "__main__":
    main()
