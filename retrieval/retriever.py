"""
retrieval/retriever.py
------------------------
Unified retriever module supporting 3 retrieval paradigms:
1. Semantic / Dense: ChromaDB vector search (Cosine similarity)
2. Keyword / Sparse: BM25 lexical search over indexed document chunks
3. Hybrid / Ensemble: Combines Semantic + BM25 using Reciprocal Rank Fusion (RRF) with configurable weights
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, List, Sequence

# Ensure project root is in sys.path when running this script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pydantic import Field
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_SEARCH_TYPE,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_TOP_K,
)
from retrieval.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


class HybridEnsembleRetriever(BaseRetriever):
    """Ensemble retriever combining multiple retrievers using Reciprocal Rank Fusion (RRF) with weights."""
    
    retrievers: List[BaseRetriever]
    weights: List[float] = Field(default_factory=lambda: [0.5, 0.5])
    c: int = 60  # RRF rank smoothing constant
    top_k: int = DEFAULT_TOP_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        all_results = [r.invoke(query) for r in self.retrievers]
        
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for weight, doc_list in zip(self.weights, all_results):
            for rank, doc in enumerate(doc_list):
                # Unique key identifier for chunk deduplication
                src = doc.metadata.get("source", "")
                page = str(doc.metadata.get("page", ""))
                snippet = doc.page_content[:100]
                key = f"{src}_{page}_{snippet}"
                
                if key not in doc_map:
                    doc_map[key] = doc
                    doc_scores[key] = 0.0
                
                # Weighted RRF score formula
                doc_scores[key] += weight * (1.0 / (self.c + rank + 1))

        # Rank all combined chunks by highest fusion score
        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        top_docs = []
        for rank, k in enumerate(sorted_keys[:self.top_k], 1):
            doc = doc_map[k]
            doc.metadata["score"] = round(doc_scores[k], 4)
            doc.metadata["rank"] = rank
            top_docs.append(doc)
        return top_docs


def get_all_documents_from_vectorstore(source_filter: str | None = None) -> List[Document]:
    """Extract all indexed Document objects from the Chroma store.
    
    If source_filter is specified, only documents matching that source filename are returned.
    """
    vectordb = load_vectorstore()
    try:
        data = vectordb._collection.get(include=["documents", "metadatas"])
        raw_docs = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        
        docs: List[Document] = []
        for text, meta in zip(raw_docs, metadatas):
            if not text or not text.strip():
                continue
            meta_dict = meta if isinstance(meta, dict) else {}
            if source_filter and meta_dict.get("source") != source_filter:
                continue
            docs.append(Document(page_content=text, metadata=meta_dict))
        return docs
    except Exception as e:
        logger.warning("Failed to fetch documents from vectorstore: %s", e)
        return []


def get_semantic_retriever(
    top_k: int = DEFAULT_TOP_K,
    search_type: str = DEFAULT_SEARCH_TYPE,
    source_filter: str | None = None,
) -> BaseRetriever:
    """Build a semantic / dense vector retriever backed by ChromaDB (Cosine similarity)."""
    vectordb = load_vectorstore()
    search_kwargs: dict[str, Any] = {"k": top_k}
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}

    return vectordb.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def get_bm25_retriever(
    top_k: int = DEFAULT_TOP_K,
    source_filter: str | None = None,
) -> BaseRetriever:
    """Build a BM25 sparse keyword retriever from documents stored in Chroma."""
    docs = get_all_documents_from_vectorstore(source_filter=source_filter)
    if not docs:
        logger.warning("No documents available to build BM25 retriever. Using fallback.")
        docs = [Document(page_content="دليل إجراءات البنك والمستندات", metadata={"source": "fallback"})]
    
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = top_k
    return bm25


def get_hybrid_retriever(
    top_k: int = DEFAULT_TOP_K,
    search_type: str = DEFAULT_SEARCH_TYPE,
    source_filter: str | None = None,
    weights: Sequence[float] | None = None,
) -> BaseRetriever:
    """Build a hybrid retriever combining Semantic vector search and BM25 keyword search."""
    if weights is None:
        weights = [DEFAULT_SEMANTIC_WEIGHT, DEFAULT_BM25_WEIGHT]

    semantic_retriever = get_semantic_retriever(
        top_k=top_k, search_type=search_type, source_filter=source_filter
    )
    bm25_retriever = get_bm25_retriever(
        top_k=top_k, source_filter=source_filter
    )

    return HybridEnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=list(weights),
        top_k=top_k,
    )


def get_retriever(
    mode: str = DEFAULT_RETRIEVAL_MODE,
    top_k: int = DEFAULT_TOP_K,
    search_type: str = DEFAULT_SEARCH_TYPE,
    source_filter: str | None = None,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
) -> BaseRetriever:
    """Factory function to build a retriever for the requested mode:
    - 'semantic': Dense vector search (Chroma + BGE-M3, Cosine similarity)
    - 'keyword': BM25 exact keyword / lexical matching
    - 'hybrid': Ensemble combining Dense + BM25 with custom weights
    """
    mode_clean = (mode or DEFAULT_RETRIEVAL_MODE).lower().strip()

    if mode_clean == "keyword":
        return get_bm25_retriever(top_k=top_k, source_filter=source_filter)
    elif mode_clean == "semantic":
        return get_semantic_retriever(
            top_k=top_k, search_type=search_type, source_filter=source_filter
        )
    else:  # default "hybrid"
        return get_hybrid_retriever(
            top_k=top_k,
            search_type=search_type,
            source_filter=source_filter,
            weights=[semantic_weight, bm25_weight],
        )


def retrieve(
    query: str,
    mode: str = DEFAULT_RETRIEVAL_MODE,
    top_k: int = DEFAULT_TOP_K,
    search_type: str = DEFAULT_SEARCH_TYPE,
    source_filter: str | None = None,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
) -> List[Document]:
    """Convenience one-shot retrieval call."""
    retriever = get_retriever(
        mode=mode,
        top_k=top_k,
        search_type=search_type,
        source_filter=source_filter,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
    )
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
