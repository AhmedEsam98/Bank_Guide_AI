"""
ingestion/ingest.py
---------------------
Top-level orchestrator: PDFs -> (OCR-aware) text extraction -> chunking ->
embeddings -> persisted Chroma vector store.

Can be run standalone:
    python -m ingestion.ingest --strategy recursive_character --chunk-size 1000

...or imported and called from the Streamlit app.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when running this script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.documents import Document

from config import (
    DATA_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_STRATEGY,
)
from ingestion.ocr_loader import load_all_pdfs
from ingestion.chunking import chunk_documents
from retrieval.vectorstore import build_vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_ingestion(
    pdf_dir: str = str(DATA_DIR),
    strategy: str = DEFAULT_CHUNK_STRATEGY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    use_cache: bool = True,
    reset_collection: bool = True,
) -> dict:
    """Run the full ingestion pipeline and return a small summary dict
    (used by the Streamlit UI to show progress/results)."""

    logger.info("Step 1/3 - Loading & OCR-extracting PDFs from %s", pdf_dir)
    raw_docs: List[Document] = load_all_pdfs(pdf_dir, use_cache=use_cache)
    logger.info("Loaded %d page-documents", len(raw_docs))

    if not raw_docs:
        return {"pages": 0, "chunks": 0, "strategy": strategy}

    logger.info("Step 2/3 - Chunking with strategy='%s'", strategy)
    chunks = chunk_documents(
        raw_docs, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    logger.info("Produced %d chunks", len(chunks))

    logger.info("Step 3/3 - Embedding + writing to Chroma vector store")
    build_vectorstore(chunks, reset=reset_collection)
    logger.info("Ingestion complete.")

    return {
        "pages": len(raw_docs),
        "chunks": len(chunks),
        "strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "sources": sorted({d.metadata.get("source", "unknown") for d in raw_docs}),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG vector store.")
    parser.add_argument("--pdf-dir", default=str(DATA_DIR))
    parser.add_argument("--strategy", default=DEFAULT_CHUNK_STRATEGY)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--no-cache", action="store_true", help="Disable OCR cache reuse")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run_ingestion(
        pdf_dir=args.pdf_dir,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_cache=not args.no_cache,
    )
    print(summary)
