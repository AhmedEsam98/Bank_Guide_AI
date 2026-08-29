"""
retrieval/retriever.py
------------------------
Thin wrapper around the Chroma retriever so the rest of the app doesn't
need to know about search_type / kwargs plumbing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when running this script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config import DEFAULT_SEARCH_TYPE, DEFAULT_TOP_K
from retrieval.vectorstore import load_vectorstore


def get_retriever(
    top_k: int = DEFAULT_TOP_K,
    search_type: str = DEFAULT_SEARCH_TYPE,
    source_filter: str | None = None,
) -> VectorStoreRetriever:
    """Build a retriever.

    search_type: "similarity" (pure cosine similarity) or "mmr" (Maximal
    Marginal Relevance, which reduces redundant chunks in the result set --
    useful given these manuals repeat boilerplate headers/footers a lot).

    source_filter: optionally restrict retrieval to a single source PDF
    (e.g. "Central_Mail_and_Files_Unit_Procedures_Manual.pdf").
    """
    vectordb = load_vectorstore()

    search_kwargs: dict = {"k": top_k}
    if search_type == "mmr":
        search_kwargs.update({"fetch_k": max(top_k * 4, 20), "lambda_mult": 0.5})
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}

    return vectordb.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    search_type: str = DEFAULT_SEARCH_TYPE,
    source_filter: str | None = None,
) -> List[Document]:
    """Convenience one-shot retrieval call (used directly by the generation
    step and for debugging in the Streamlit sidebar)."""
    retriever = get_retriever(top_k=top_k, search_type=search_type, source_filter=source_filter)
    return retriever.invoke(query)


def list_available_sources() -> List[str]:
    """Return the distinct source PDF filenames currently indexed, for
    populating a filter dropdown in the UI."""
    vectordb = load_vectorstore()
    try:
        data = vectordb.get(include=["metadatas"])
        sources = {m.get("source") for m in data.get("metadatas", []) if m}
        return sorted(s for s in sources if s)
    except Exception:
        return []
