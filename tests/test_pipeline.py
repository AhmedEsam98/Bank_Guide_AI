"""
test_pipeline.py
----------------
Automated test script for the Bank Guide AI pipeline:
1. Tests document extraction -> outputs 'extracted_data.md'
2. Tests chunking (in an isolated subprocess -- avoids a native library
   conflict between docling_loader and langchain_text_splitters when both
   are imported in the same process) -> outputs 'chunking_test.md'

Usage:
    python test_pipeline.py
    python -m tests.test_pipeline
"""

from __future__ import annotations

import os

# Must be set BEFORE torch / onnxruntime / chromadb / faiss get imported
# anywhere below (directly or transitively). On Windows, having more than
# one bundled OpenMP runtime loaded in the same process (common with
# torch + onnxruntime/chromadb/faiss together) causes a native access
# violation (exit code 3221225477 / 0xC0000005) with no Python traceback.
# This tells the runtime to tolerate the duplicate instead of crashing.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import pickle
import subprocess
from pathlib import Path

# Ensure project root is in sys.path -- this MUST run before any imports
# from our own packages (config, ingestion, etc.), since this script is
# sometimes launched directly (python test_pipeline.py) rather than as a
# module, in which case Python only puts this file's own folder on
# sys.path, not the project root.
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_DIR,
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DOCLING_OCR_MODEL_NAME,
)


def run_extraction_test(output_path: Path) -> list:
    from ingestion.docling_loader import load_all_pdfs
    print("\n[1/2] Running Document Extraction Test...")
    t0 = time.time()
    docs = load_all_pdfs(DATA_DIR, use_cache=True)
    elapsed = time.time() - t0

    print(f"  -> Extracted {len(docs)} pages across all PDFs in {elapsed:.2f}s.")

    # Group pages by PDF
    pdf_groups: dict[str, list] = {}
    for doc in docs:
        pdf_groups.setdefault(doc.metadata.get("source", "Unknown"), []).append(doc)

    lines = [
        "# Bank Guide AI — Extracted Documents Report",
        "",
        f"- **Extraction Engine**: Docling Layout & Table Parsing",
        f"- **OCR Model**: `{DOCLING_OCR_MODEL_NAME}`",
        f"- **Total Extracted Pages**: `{len(docs)}`",
        f"- **PDF Documents Count**: `{len(pdf_groups)}`",
        f"- **Generated At**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "---",
        "",
    ]

    for pdf_name, page_docs in pdf_groups.items():
        lines.append(f"## Document: `{pdf_name}` ({len(page_docs)} pages)")
        lines.append("")
        for doc in sorted(page_docs, key=lambda d: d.metadata.get("page", 0)):
            page_no = doc.metadata.get("page", "?")
            word_count = doc.metadata.get("word_count", len(doc.page_content.split()))
            has_tables = "Yes" if doc.metadata.get("has_tables") else "No"
            lines.append(f"### Page {page_no} (Words: {word_count} | Tables: {has_tables})")
            lines.append("")
            lines.append(doc.page_content)
            lines.append("")
            lines.append("---")
            lines.append("")

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    print(f"  -> [OK] Saved extracted data report to: {output_path.name}")
    return docs


def _write_chunking_report(chunks: list, output_path: Path, input_page_count: int) -> None:
    """Write the markdown chunking report from an already-computed list of
    chunks. Split out from run_chunking_test so it can be called after
    chunks come back from the isolated subprocess (no `docs` needed at
    this point)."""
    lengths = [len(c.page_content) for c in chunks]
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    avg_len = int(sum(lengths) / len(lengths)) if lengths else 0
    median_len = sorted(lengths)[len(lengths) // 2] if lengths else 0

    print(f"  -> Generated {len(chunks)} chunks.")
    print(f"  -> Lengths (chars): Min={min_len}, Max={max_len}, Avg={avg_len}, Median={median_len}")

    md_lines = [
        "# 📑 Bank Guide AI — Chunking Test Report",
        "",
        "> **Status**: ✅ Ingestion & Chunking Complete  ",
        f"> **Chunking Strategy**: `{DEFAULT_CHUNK_STRATEGY}`  ",
        f"> **Input Pages**: `{input_page_count}`  ",
        f"> **Total Output Chunks**: `{len(chunks):,}`  ",
        "",
        "## 📊 Chunking Statistics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Chunking Strategy** | `{DEFAULT_CHUNK_STRATEGY}` |",
        f"| **Input Pages** | `{input_page_count}` |",
        f"| **Total Chunks** | `{len(chunks):,}` |",
        f"| **Shortest Chunk** | `{min_len}` characters |",
        f"| **Longest Chunk** | `{max_len}` characters |",
        f"| **Average Length** | `{avg_len}` characters |",
        f"| **Median Length** | `{median_len}` characters |",
        "",
        "---",
        "",
        "## 🔍 Detailed Chunk Inspection",
        "",
    ]

    for idx, chunk in enumerate(chunks, start=1):
        src = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "?")
        strategy = chunk.metadata.get("chunk_strategy", DEFAULT_CHUNK_STRATEGY)
        h1 = chunk.metadata.get("header_1", "")
        h2 = chunk.metadata.get("header_2", "")
        header_path = f"{h1} > {h2}".strip(" >") if (h1 or h2) else "None"

        md_lines.append(f"### 🔹 Chunk #{idx} of {len(chunks)}")
        md_lines.append(f"- **Document**: `{src}`")
        md_lines.append(f"- **Page**: `{page}`")
        md_lines.append(f"- **Strategy**: `{strategy}`")
        md_lines.append(f"- **Header Path**: `{header_path}`")
        md_lines.append(f"- **Length**: `{len(chunk.page_content)}` characters")
        md_lines.append("")
        md_lines.append("```markdown")
        md_lines.append(chunk.page_content.strip())
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    md_path = output_path.with_suffix(".md")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  -> [OK] Saved chunking test report (MD) to: {md_path.name}")


def run_chunking_test_isolated(docs: list, output_path: Path) -> list:
    """Run chunking in a SEPARATE PROCESS to avoid a native library
    conflict between docling_loader and langchain_text_splitters when
    imported in the same process (causes silent crashes on Windows --
    manifests as stack overflow / access violation with no Python
    traceback)."""
    print(f"\n[2/2] Running Chunking Test (Strategy: '{DEFAULT_CHUNK_STRATEGY}') [isolated subprocess]...")

    docs_pickle_path = TESTS_DIR / "_docs_temp.pkl"
    chunks_pickle_path = TESTS_DIR / "_chunks_temp.pkl"

    with open(docs_pickle_path, "wb") as f:
        pickle.dump(docs, f)

    chunk_script = f'''
import sys, pickle
sys.path.insert(0, r"{PROJECT_ROOT}")
from ingestion.chunking import chunk_documents
from config import DEFAULT_CHUNK_STRATEGY, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

with open(r"{docs_pickle_path}", "rb") as f:
    docs = pickle.load(f)

chunks = chunk_documents(
    docs,
    strategy=DEFAULT_CHUNK_STRATEGY,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
)

with open(r"{chunks_pickle_path}", "wb") as f:
    pickle.dump(chunks, f)

print("CHUNK_OK")
'''

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-c", chunk_script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.time() - t0

    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0 or "CHUNK_OK" not in (result.stdout or ""):
        print(f"  -> [ERROR] Chunking subprocess failed (exit code {result.returncode}).")
        if result.stderr:
            print(result.stderr)
        docs_pickle_path.unlink(missing_ok=True)
        sys.exit(1)

    with open(chunks_pickle_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"  -> Chunking subprocess completed in {elapsed:.2f}s.")

    _write_chunking_report(chunks, output_path, input_page_count=len(docs))

    docs_pickle_path.unlink(missing_ok=True)
    chunks_pickle_path.unlink(missing_ok=True)

    return chunks


def main():
    print("=" * 80, flush=True)
    print("BANK GUIDE AI — END-TO-END PIPELINE TEST RUNNER", flush=True)
    print("=" * 80, flush=True)

    extracted_md_path = TESTS_DIR / "extracted_data.md"
    chunking_base_path = TESTS_DIR / "chunking_test"

    # Step 1: Extraction
    docs = run_extraction_test(extracted_md_path)

    # Step 2: Chunking (isolated subprocess -- see run_chunking_test_isolated)
    run_chunking_test_isolated(docs, chunking_base_path)

    # Explicit memory cleanup
    del docs
    import gc
    gc.collect()

    print("\n" + "=" * 80, flush=True)
    print("ALL TESTS COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 80, flush=True)
    print(f"1. Extracted Data Markdown : {extracted_md_path.name}", flush=True)
    print(f"2. Chunking Test Markdown  : {chunking_base_path.name}.md", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()