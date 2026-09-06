"""
advanced_rag/query_transform.py
-------------------------------
Five query-understanding / transformation techniques, each callable
independently by the pipeline orchestrator based on the router's decision.

Every function:
  - Accepts the original question (+ optional extras)
  - Uses the Groq LLM with a specialised prompt
  - Logs its cost via CostTracker
  - Returns the transformed query / queries
  - Falls back gracefully on LLM errors
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.prompts import ChatPromptTemplate

from generation.generator import get_llm
from evaluation.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


# =========================================================================
# 1. Query Rewriting
# =========================================================================

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query-rewriting assistant for a bank procedure manuals RAG system. "
     "The documents are in Arabic. Rewrite the user's question to be clearer, more "
     "specific, and better suited for semantic search. If the question is vague or "
     "conversational, make it explicit. If it is in English, keep it in English but "
     "add key Arabic terms that might appear in the manuals. "
     "Return ONLY the rewritten query — no explanation."),
    ("human", "{question}"),
])


def rewrite_query(
    question: str,
    model_name: str = "openai/gpt-oss-20b",
    tracker: Optional[CostTracker] = None,
) -> str:
    """Return a single improved version of *question*."""
    llm = get_llm(model_name=model_name, temperature=0.3)
    t0 = time.time()
    response = llm.invoke(_REWRITE_PROMPT.format_messages(question=question))
    latency = time.time() - t0

    if tracker:
        tracker.record("rewriter", model_name, response, latency)

    rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()
    return rewritten or question  # fallback to original


# =========================================================================
# 2. Multi-Query Generation
# =========================================================================

_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a search-query generator. Given the user's question, generate {n} "
     "alternative phrasings that approach the information need from different angles. "
     "These will be used to search a bank procedure manuals knowledge base. "
     "Return a JSON array of strings — no markdown, no explanation.\n"
     'Example: ["query 1", "query 2", "query 3"]'),
    ("human", "{question}"),
])


def generate_multi_queries(
    question: str,
    n: int = 3,
    model_name: str = "openai/gpt-oss-20b",
    tracker: Optional[CostTracker] = None,
) -> List[str]:
    """Return *n* alternative phrasings of *question*."""
    llm = get_llm(model_name=model_name, temperature=0.5)
    t0 = time.time()
    response = llm.invoke(
        _MULTI_QUERY_PROMPT.format_messages(question=question, n=n)
    )
    latency = time.time() - t0

    if tracker:
        tracker.record("multi_query", model_name, response, latency)

    raw = response.content if hasattr(response, "content") else str(response)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        queries = json.loads(cleaned)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:n]
    except (json.JSONDecodeError, TypeError):
        logger.warning("Multi-query JSON parse failed, falling back to original query.")

    return [question]  # fallback


# =========================================================================
# 3. Question Decomposition
# =========================================================================

_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a question decomposition assistant. Break the user's complex "
     "question into simpler, independent sub-questions that can each be "
     "answered separately and then combined. "
     "Return a JSON array of strings — no markdown, no explanation.\n"
     'Example: ["sub-question 1", "sub-question 2"]'),
    ("human", "{question}"),
])


def decompose_question(
    question: str,
    model_name: str = "openai/gpt-oss-20b",
    tracker: Optional[CostTracker] = None,
) -> List[str]:
    """Decompose a complex question into sub-questions."""
    llm = get_llm(model_name=model_name, temperature=0.2)
    t0 = time.time()
    response = llm.invoke(_DECOMPOSE_PROMPT.format_messages(question=question))
    latency = time.time() - t0

    if tracker:
        tracker.record("decomposition", model_name, response, latency)

    raw = response.content if hasattr(response, "content") else str(response)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        subs = json.loads(cleaned)
        if isinstance(subs, list) and all(isinstance(q, str) for q in subs):
            return subs
    except (json.JSONDecodeError, TypeError):
        logger.warning("Decomposition parse failed, using original question.")

    return [question]


# =========================================================================
# 4. HyDE — Hypothetical Document Embeddings
# =========================================================================

_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Write a short, factual passage (in Arabic "
     "if the topic relates to bank procedures, otherwise in the user's language) "
     "that would answer the following question, as if it were an excerpt from a "
     "bank procedure manual. This passage will be used for semantic search — "
     "make it dense with relevant terminology. "
     "Return ONLY the passage — no title, no explanation."),
    ("human", "{question}"),
])


def generate_hyde_passage(
    question: str,
    model_name: str = "openai/gpt-oss-20b",
    tracker: Optional[CostTracker] = None,
) -> str:
    """Generate a hypothetical answer passage to use as a search query."""
    llm = get_llm(model_name=model_name, temperature=0.4)
    t0 = time.time()
    response = llm.invoke(_HYDE_PROMPT.format_messages(question=question))
    latency = time.time() - t0

    if tracker:
        tracker.record("hyde", model_name, response, latency)

    passage = response.content.strip() if hasattr(response, "content") else str(response).strip()
    return passage or question


# =========================================================================
# 5. Self-Query — extract semantic query + metadata filters
# =========================================================================

_SELF_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a metadata-filter extraction assistant for a bank procedure manuals "
     "knowledge base stored in ChromaDB.\n\n"
     "Available metadata fields:\n"
     '  - "source" (string): PDF filename, one of:\n'
     '      "Assets and wearhouse operation Tasks and Procedures Manual.pdf"\n'
     '      "Central Alarm Tasks And Procedures Manual.pdf"\n'
     '      "Central Mail and Files Unit Procedures Manual.pdf"\n'
     '  - "page" (integer): page number in the PDF\n'
     '  - "extraction_method" (string): "native" or "ocr"\n'
     '  - "chunk_strategy" (string): chunking strategy used\n\n'
     "Given the user's question, extract:\n"
     '1. "query": the semantic search query (the actual information need)\n'
     '2. "filters": a dict of Chroma metadata filters to apply, or empty dict {{}}\n\n'
     "Return ONLY valid JSON — no markdown, no explanation.\n"
     'Example: {{"query": "alarm maintenance procedures", "filters": '
     '{{"source": "Central Alarm Tasks And Procedures Manual.pdf"}}}}'),
    ("human", "{question}"),
])


def extract_self_query(
    question: str,
    model_name: str = "openai/gpt-oss-20b",
    tracker: Optional[CostTracker] = None,
) -> Dict[str, Any]:
    """Extract semantic query and metadata filters from *question*.

    Returns ``{"query": str, "filters": dict}``.
    """
    llm = get_llm(model_name=model_name, temperature=0.0)
    t0 = time.time()
    response = llm.invoke(_SELF_QUERY_PROMPT.format_messages(question=question))
    latency = time.time() - t0

    if tracker:
        tracker.record("self_query", model_name, response, latency)

    raw = response.content if hasattr(response, "content") else str(response)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        data = json.loads(cleaned)
        return {
            "query": data.get("query", question),
            "filters": data.get("filters", {}),
        }
    except (json.JSONDecodeError, TypeError):
        logger.warning("Self-query parse failed, using original question with no filters.")
        return {"query": question, "filters": {}}
