"""
ingestion/embeddings.py
-----------------------
Dedicated module for loading and managing the text embedding model.
Uses HuggingFace BGE-M3 (multilingual) by default, optimized for Arabic and English.
"""

from __future__ import annotations

import logging
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when running this script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings(
    model_name: str = EMBEDDING_MODEL_NAME,
    device: str = EMBEDDING_DEVICE,
    normalize: bool = True,
) -> HuggingFaceEmbeddings:
    """Load and cache the HuggingFace embedding model on GPU.

    Cached with lru_cache so the model weights are loaded into memory
    only once per process across ingestion, retrieval, and UI sessions.
    """
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    logger.info("Loading embedding model: %s (device=%s)", model_name, device)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize, "batch_size": 4},
    )


def embed_query(text: str) -> List[float]:
    """Generate an embedding vector for a single query string."""
    embeddings = get_embeddings()
    return embeddings.embed_query(text)


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Generate embedding vectors for a list of document strings."""
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(f"Testing embedding model: {EMBEDDING_MODEL_NAME}")

    sample_texts = [
        "إجراءات وحدة البريد والملفات المركزية",
        "Central Mail and Files Unit Procedures",
    ]

    t0 = time.time()
    model = get_embeddings()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    for text in sample_texts:
        t_embed = time.time()
        vec = embed_query(text)
        print(f"Text: '{text}' -> Vector dimension: {len(vec)} (took {time.time() - t_embed:.4f}s)")
