"""
retrieval/vectorstore.py
--------------------------
Builds / loads the persisted Chroma vector store using a multilingual
embedding model (BGE-M3), which handles the Arabic source documents well.
"""

from __future__ import annotations

import logging
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when running this script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import COLLECTION_NAME, VECTORSTORE_DIR
from ingestion.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def build_vectorstore(chunks: List[Document], reset: bool = True) -> Chroma:
    """Embed `chunks` and persist them to the Chroma collection on disk.

    If `reset` is True, any existing collection data is wiped first -- this
    is what you want when re-ingesting with a different chunking strategy,
    so stale chunks from a previous strategy don't linger in the index.
    """
    if reset and VECTORSTORE_DIR.exists():
        logger.info("Resetting existing vector store at %s", VECTORSTORE_DIR)
        shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )

    # Batch to keep GPU VRAM safely below 4GB capacity
    batch_size = 4
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectordb.add_documents(batch)
        logger.info("Embedded %d/%d chunks", min(i + batch_size, len(chunks)), len(chunks))
        
        # Periodically clear GPU cache every 20 chunks
        if i % 20 == 0:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return vectordb


def load_vectorstore() -> Chroma:
    """Load the existing persisted vector store (read-only usage from the
    retrieval/generation side)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )


def vectorstore_is_ready() -> bool:
    """Quick check used by the Streamlit app to decide whether to prompt
    the user to run ingestion first."""
    try:
        vectordb = load_vectorstore()
        return vectordb._collection.count() > 0  # noqa: SLF001
    except Exception:
        return False
