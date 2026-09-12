"""
test_retrieval.py
--------------------
Run a batch of test queries through your RETRIEVER ONLY (no LLM call --
fast, free, and isolates retrieval quality from generation quality).
Reports, per query, the ranked chunks that came back: source, page,
chunking strategy/metadata, similarity score, and a content preview.

If your queries file is a Markdown ground-truth table (category, question,
expected answer, reference/page), the script also extracts an expected
page number from the reference column and auto-flags each query as a
HIT or MISS depending on whether that page shows up in the retrieved
chunks -- plus an overall summary with hit rate, latency, and a
per-category breakdown.

Queries come from:
  - a Markdown ground-truth table like test_rag.md (--queries-file test_rag.md)
  - a plain .txt file, one query per line (--queries-file queries.txt)

Usage:
    python tests/test_retrieval.py
    python tests/test_retrieval.py --queries-file test_rag.md
    python tests/test_retrieval.py --top-k 5 --retrieval-mode semantic
    python tests/test_retrieval.py --retrieval-mode hybrid --semantic-weight 0.7 --bm25-weight 0.3
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

# Ensure project root is in sys.path BEFORE any project imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import (
    DEFAULT_TOP_K,
    DEFAULT_SEARCH_TYPE,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_BM25_WEIGHT,
)
from retrieval.retriever import retrieve
from retrieval.vectorstore import vectorstore_is_ready


# --------------------------------------------------------------------------
# Query loading
# --------------------------------------------------------------------------

@dataclass
class QueryItem:
    question: str
    category: Optional[str] = None
    expected_answer: Optional[str] = None
    reference: Optional[str] = None
    expected_pages: List[int] = field(default_factory=list)


_SEPARATOR_ROW_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
_PAGE_NUM_RE = re.compile(r"\d+")


def load_queries_from_txt(path: Path) -> List[QueryItem]:
    return [
        QueryItem(question=line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_queries_from_md(path: Path) -> List[QueryItem]:
    """Parse a ground-truth table: category | question | expected answer | reference.

    Falls back gracefully if the table has fewer columns -- only `question`
    is required, the rest are used for auto hit/miss checking when present.
    """
    text = path.read_text(encoding="utf-8")
    table_lines = [line.strip() for line in text.splitlines()
                   if line.strip().startswith("|")]
    if not table_lines:
        return []

    data_lines = table_lines[1:]  # drop header
    if data_lines and _SEPARATOR_ROW_RE.match(data_lines[0]):
        data_lines = data_lines[1:]  # drop separator row

    items: List[QueryItem] = []
    for line in data_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or not cells[1]:
            continue
        category = cells[0] if len(cells) > 0 and cells[0] else None
        question = cells[1]
        expected_answer = cells[2] if len(cells) > 2 and cells[2] else None
        reference = cells[3] if len(cells) > 3 and cells[3] else None
        expected_pages = [int(n) for n in _PAGE_NUM_RE.findall(
            reference)] if reference else []
        items.append(
            QueryItem(
                question=question,
                category=category,
                expected_answer=expected_answer,
                reference=reference,
                expected_pages=expected_pages,
            )
        )
    return items


def load_queries(path: Path) -> List[QueryItem]:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return load_queries_from_md(path)
    elif suffix == ".txt":
        return load_queries_from_txt(path)
    else:
        raise ValueError(
            f"Unsupported queries file type '{suffix}'. Use .md or .txt.")


# --------------------------------------------------------------------------
# Retrieval execution
# --------------------------------------------------------------------------

def _filter_supported_kwargs(func: Callable, candidate_kwargs: dict) -> dict:
    """Only keep kwargs that `func` actually declares. Computed once, not per-query."""
    sig = inspect.signature(func)
    supported = {k: v for k, v in candidate_kwargs.items()
                 if k in sig.parameters}
    dropped = sorted(set(candidate_kwargs) - set(supported))
    if dropped:
        print(
            f"(note: retrieve() doesn't accept {dropped}, skipping those for all queries)\n")
    return supported


def _normalize_doc(doc):
    """Accept either a Document-like object or a (doc, score) tuple."""
    if isinstance(doc, tuple) and len(doc) == 2:
        d, score = doc
        return d, score
    return doc, None


def _extract_score(doc, fallback_score) -> str:
    if fallback_score is not None:
        try:
            return f"{float(fallback_score):.4f}"
        except (TypeError, ValueError):
            return str(fallback_score)
    metadata = getattr(doc, "metadata", None) or {}
    for key in ("score", "relevance_score", "similarity", "distance"):
        if key in metadata:
            val = metadata[key]
            try:
                return f"{float(val):.4f}"
            except (TypeError, ValueError):
                return str(val)
    return "n/a"


def _extract_page(doc) -> Optional[int]:
    metadata = getattr(doc, "metadata", None) or {}
    page = metadata.get("page")
    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def run_query(query: str, supported_kwargs: dict) -> dict:
    t0 = time.time()
    try:
        raw_docs = retrieve(query, **supported_kwargs)
        error = None
    except Exception as exc:  # noqa: BLE001
        raw_docs = []
        error = str(exc)
    elapsed = time.time() - t0
    docs = [_normalize_doc(d) for d in raw_docs]
    return {"docs": docs, "elapsed": elapsed, "error": error}


# --------------------------------------------------------------------------
# Report building
# --------------------------------------------------------------------------

def build_report(items: List[QueryItem], candidate_kwargs: dict, supported_kwargs: dict) -> str:
    lines = [
        "# Retrieval Test Report",
        "",
        f"- **Queries**: `{len(items)}`",
        f"- **Requested settings**: `{candidate_kwargs}`",
        f"- **Settings actually used**: `{supported_kwargs}`",
        f"- **Similarity Metric**: `Cosine Similarity (Dense)`",
        f"- **Generated**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
    ]

    total_latency = 0.0
    error_count = 0
    checkable = 0
    hits = 0
    category_stats: dict = {}
    per_query_rows = []

    for i, item in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {item.question[:60]}...")
        result = run_query(item.question, supported_kwargs)
        total_latency += result["elapsed"]

        hit_marker = "N/A"
        if item.expected_pages:
            checkable += 1
            retrieved_pages = {
                p for doc, _ in result["docs"] if (p := _extract_page(doc)) is not None
            }
            is_hit = bool(retrieved_pages & set(item.expected_pages))
            hit_marker = "✅ HIT" if is_hit else "❌ MISS"
            if is_hit:
                hits += 1
            if item.category:
                cat = category_stats.setdefault(
                    item.category, {"total": 0, "hits": 0})
                cat["total"] += 1
                cat["hits"] += 1 if is_hit else 0

        if result["error"]:
            error_count += 1

        per_query_rows.append((i, item, result, hit_marker))

    # --- Summary section ---
    avg_latency = total_latency / len(items) if items else 0.0
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Avg latency**: `{avg_latency:.3f}s`")
    lines.append(f"- **Errors**: `{error_count}/{len(items)}`")
    if checkable:
        lines.append(
            f"- **Ground-truth hit rate**: `{hits}/{checkable}` ({hits/checkable:.0%})")
        if category_stats:
            lines.append("")
            lines.append("| Category | Hits | Total | Rate |")
            lines.append("|---|---|---|---|")
            for cat, stat in category_stats.items():
                rate = stat["hits"] / stat["total"] if stat["total"] else 0
                lines.append(
                    f"| {cat} | {stat['hits']} | {stat['total']} | {rate:.0%} |")
    else:
        lines.append(
            "- **Ground-truth hit rate**: `n/a (no reference pages found in queries file)`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Per-query detail ---
    for i, item, result, hit_marker in per_query_rows:
        lines.append(f"## Q{i}. {item.question}")
        lines.append("")
        if item.category:
            lines.append(f"- **Category**: `{item.category}`")
        if item.reference:
            lines.append(f"- **Expected reference**: `{item.reference}`")
        if item.expected_pages:
            lines.append(
                f"- **Ground-truth check**: {hit_marker} (expected page(s): {item.expected_pages})")
        lines.append(f"- **Latency**: `{result['elapsed']:.3f}s`")

        if result["error"]:
            lines.append(f"- **ERROR**: `{result['error']}`")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        docs = result["docs"]
        lines.append(f"- **Chunks retrieved**: `{len(docs)}`")
        lines.append("")

        if not docs:
            lines.append("_(no chunks retrieved)_")
        else:
            lines.append(
                "| Rank | Source | Page | Strategy | Score | Preview |")
            lines.append("|---|---|---|---|---|---|")
            for rank, (doc, fallback_score) in enumerate(docs, start=1):
                metadata = getattr(doc, "metadata", None) or {}
                src = metadata.get("source", "?")
                page = metadata.get("page", "?")
                strat = metadata.get("chunk_strategy", "?")
                score = _extract_score(doc, fallback_score)
                content = getattr(doc, "page_content", "") or ""
                preview = " ".join(content.split())[:150]
                lines.append(
                    f"| {rank} | {src} | {page} | {strat} | {score} | {preview} |")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Test retrieval quality against a set of queries (Cosine similarity, no LLM call).")
    parser.add_argument("--queries-file", default="test_rag.md",
                        help="Path to a .md (ground-truth table) or .txt file (default: test_rag.md)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--search-type", default=DEFAULT_SEARCH_TYPE, choices=[
                        "similarity"], help="Vector search type (default: similarity - Cosine)")
    parser.add_argument("--retrieval-mode", default=DEFAULT_RETRIEVAL_MODE, choices=[
                        "hybrid", "semantic", "keyword"], help="Retrieval mode: hybrid (Dense+BM25), semantic (Cosine), or keyword (default: %(default)s)")
    parser.add_argument("--semantic-weight", type=float,
                        default=DEFAULT_SEMANTIC_WEIGHT)
    parser.add_argument("--bm25-weight", type=float,
                        default=DEFAULT_BM25_WEIGHT)
    parser.add_argument("--source-filter", default=None,
                        help="Restrict to a single source PDF filename")
    parser.add_argument("--output", default="retrieval_report.md")
    return parser.parse_args()


def main():
    args = _parse_args()

    if not vectorstore_is_ready():
        print("Vectorstore is not ready. Run ingestion first.")
        sys.exit(1)

    queries_path = Path(args.queries_file)
    if not queries_path.exists():
        if (Path(__file__).resolve().parent / args.queries_file).exists():
            queries_path = Path(__file__).resolve().parent / args.queries_file
        elif (project_root / "tests" / args.queries_file).exists():
            queries_path = project_root / "tests" / args.queries_file
        elif (project_root / args.queries_file).exists():
            queries_path = project_root / args.queries_file

    if not queries_path.exists():
        print(f"File not found: {queries_path}")
        sys.exit(1)

    items = load_queries(queries_path)
    if not items:
        print(
            f"No queries could be parsed from {queries_path.name}. Check the file format.")
        sys.exit(1)
    print(f"Loaded {len(items)} queries from {queries_path.name}.\n")

    candidate_kwargs = {
        "top_k": args.top_k,
        "search_type": args.search_type,
        "mode": args.retrieval_mode,
        "semantic_weight": args.semantic_weight,
        "bm25_weight": args.bm25_weight,
        "source_filter": args.source_filter,
    }
    candidate_kwargs = {k: v for k,
                        v in candidate_kwargs.items() if v is not None}
    supported_kwargs = _filter_supported_kwargs(retrieve, candidate_kwargs)

    report = build_report(items, candidate_kwargs, supported_kwargs)

    tests_dir = Path(__file__).resolve().parent
    output_path = Path(args.output) if Path(args.output).is_absolute() else tests_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nDone. Report saved to: {output_path}")


if __name__ == "__main__":
    main()
