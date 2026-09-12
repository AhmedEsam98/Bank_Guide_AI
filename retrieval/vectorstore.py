"""
retrieval/vectorstore.py
--------------------------
Builds / loads the persisted Chroma vector store using a multilingual
embedding model (BGE-M3), which handles the Arabic source documents well.
"""

from __future__ import annotations

import os

# Must be set BEFORE torch / onnxruntime get imported below. On Windows,
# chromadb (bundles its own onnxruntime) and torch/sentence-transformers
# (pulled in via ingestion.embeddings) each bring their own OpenMP runtime.
# Having both loaded in one process can crash with a native access
# violation (exit code 3221225477 / 0xC0000005, no Python traceback).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

# Import torch (via embeddings) BEFORE chromadb. On Windows, whichever of
# torch / onnxruntime registers its OpenMP runtime first tends to "win",
# letting the second library bind to the already-loaded symbols instead
# of crashing. Importing in this order is a known workaround for the
# torch <-> onnxruntime access-violation conflict described above.
from ingestion.embeddings import get_embeddings

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import COLLECTION_NAME, VECTORSTORE_DIR

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """Get persistent Chroma client pointing to data/chroma directory."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )


def build_vectorstore(chunks: List[Document], reset: bool = True) -> Chroma:
    """Embed `chunks` and persist them to the Chroma collection on disk."""
    get_chroma_client.cache_clear()
    load_vectorstore.cache_clear()
    client = get_chroma_client()

    if reset:
        logger.info("Resetting existing collection '%s'", COLLECTION_NAME)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    vectordb = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Batch size 16 balances speed vs. memory on CPU with BGE-M3 (1024-dim)
    import gc
    batch_size = 16
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectordb.add_documents(batch)
        gc.collect()
        logger.info("Embedded batch %d/%d chunks", i + len(batch), len(chunks))

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return vectordb


@lru_cache(maxsize=1)
def load_vectorstore() -> Chroma:
    """Load the existing persisted vector store."""
    client = get_chroma_client()
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


def vectorstore_is_ready() -> bool:
    """Quick check used by the Streamlit app to decide whether to prompt
    the user to run ingestion first."""
    try:
        client = get_chroma_client()
        col = client.get_collection(COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False