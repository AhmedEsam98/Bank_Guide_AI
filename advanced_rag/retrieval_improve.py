"""
advanced_rag/retrieval_improve.py
---------------------------------
Three post-retrieval improvement strategies:

1. Reranking      — cross-encoder re-scores retrieved chunks (local GPU)
2. Compression    — LLM extracts only relevant sentences from each chunk
3. CRAG           — LLM evaluator checks retrieval quality and decides
                    whether to proceed, refine, or abort
"""

from __future__ import annotations
from evaluation.cost_tracker import CostTracker
from generation.generator import get_llm
from config import RERANKER_MODEL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


logger = logging.getLogger(__name__)


# =========================================================================
# 1. Reranking  (cross-encoder, runs locally on GPU — no LLM API call)
# =========================================================================

_reranker_model = None  # lazy-loaded singleton


def _get_reranker():
    """Lazy-load the cross-encoder reranker model once."""
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder(
                RERANKER_MODEL,
                max_length=512,
                device="cuda",
            )
            logger.info(
                "Loaded cross-encoder reranker (%s) on CUDA.", RERANKER_MODEL)
        except Exception as exc:
            logger.warning(
                "Failed to load cross-encoder (%s). Reranking disabled.", exc)
    return _reranker_model


def rerank_documents(
    query: str,
    docs: List[Document],
    top_k: int = 5,
) -> List[Document]:
    """Re-score *docs* using a cross-encoder and return the top-*top_k*.

    If the reranker model is unavailable, returns the original list unchanged.
    """
    reranker = _get_reranker()
    if reranker is None or not docs:
        return docs[:top_k]

    t0 = time.time()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    latency = time.time() - t0

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    result = [doc for _, doc in scored[:top_k]]

    logger.info(
        "Reranked %d → %d docs in %.2fs (top score=%.4f, bottom=%.4f)",
        len(docs), len(result), latency,
        scored[0][0] if scored else 0,
        scored[-1][0] if scored else 0,
    )
    return result


# =========================================================================
# 2. Contextual Compression  (LLM-based)
# =========================================================================

_COMPRESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a context compressor. Given a question and a document chunk, "
     "extract ONLY the sentences that are directly relevant to answering the "
     "question. Remove boilerplate, headers, page numbers, and irrelevant "
     "content. If nothing is relevant, return the string 'NOT_RELEVANT'. "
     "Return the compressed text only — no explanation."),
    ("human",
     "Question: {question}\n\n"
     "Document chunk:\n{chunk}"),
])


def compress_documents(
    question: str,
    docs: List[Document],
    model_name: str = "openai/gpt-oss-20b",
    tracker: Optional[CostTracker] = None,
) -> List[Document]:
    """Compress each document to only the relevant sentences.

    Drops chunks that the LLM judges as entirely irrelevant.
    """
    if not docs:
        return docs

    llm = get_llm(model_name=model_name, temperature=0.0)
    compressed: List[Document] = []
    total_original = 0
    total_compressed = 0

    t0 = time.time()
    for doc in docs:
        total_original += len(doc.page_content)
        response = llm.invoke(
            _COMPRESS_PROMPT.format_messages(
                question=question, chunk=doc.page_content[:1500]
            )
        )
        text = response.content.strip() if hasattr(
            response, "content") else str(response).strip()
        if text and text.upper() != "NOT_RELEVANT":
            total_compressed += len(text)
            compressed.append(
                Document(page_content=text, metadata=dict(doc.metadata))
            )
    latency = time.time() - t0

    if tracker:
        # Record a single aggregated entry for compression
        tracker.record("compression", model_name, response, latency)

    logger.info(
        "Compressed %d → %d docs, %d → %d chars in %.2fs",
        len(docs), len(compressed), total_original, total_compressed, latency,
    )
    return compressed if compressed else docs  # fallback to uncompressed


# =========================================================================
# 3. CRAG — Corrective RAG
# =========================================================================

_CRAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a retrieval quality evaluator. Given a question and the "
     "retrieved context chunks, assess whether the chunks contain enough "
     "information to answer the question.\n\n"
     "Respond ONLY with valid JSON — no markdown:\n"
     '{{"verdict": "<correct|ambiguous|incorrect>", '
     '"reason": "<brief explanation>"}}\n\n'
     "Verdicts:\n"
     '- "correct": chunks clearly contain the answer\n'
     '- "ambiguous": chunks are partially relevant but may be insufficient\n'
     '- "incorrect": chunks are irrelevant or do not address the question'),
    ("human",
     "Question: {question}\n\n"
     "Retrieved chunks:\n{context}"),
])


def evaluate_retrieval_crag(
    question: str,
    docs: List[Document],
    model_name: str = "openai/gpt-oss-20b",
    tracker: Optional[CostTracker] = None,
) -> Tuple[str, str]:
    """Evaluate retrieval quality and return ``(verdict, reason)``.

    Verdict is one of ``"correct"``, ``"ambiguous"``, ``"incorrect"``.
    """
    if not docs:
        if tracker:
            tracker.record_skipped("crag_evaluator")
        return "incorrect", "No documents retrieved"

    # Build a compact context string for the evaluator
    context_str = "\n---\n".join(
        f"[Chunk {i+1}] {doc.page_content[:500]}" for i, doc in enumerate(docs)
    )

    llm = get_llm(model_name=model_name, temperature=0.0)
    t0 = time.time()
    response = llm.invoke(
        _CRAG_PROMPT.format_messages(question=question, context=context_str)
    )
    latency = time.time() - t0

    if tracker:
        tracker.record("crag_evaluator", model_name, response, latency)

    raw = response.content if hasattr(response, "content") else str(response)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split(
                "\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        data = json.loads(cleaned)
        verdict = data.get("verdict", "correct")
        if verdict not in ("correct", "ambiguous", "incorrect"):
            verdict = "correct"
        return verdict, data.get("reason", "")
    except (json.JSONDecodeError, TypeError):
        logger.warning("CRAG parse failed, defaulting to 'correct'.")
        return "correct", "parse_fallback"
