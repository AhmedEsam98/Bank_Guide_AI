"""
advanced_rag/pipeline.py
------------------------
Central orchestrator that ties routing, query transforms, retrieval
improvements, and final generation into a single call.

    question
      → Router (classify route + techniques)
      → IF simple  : LLM answer directly (no retrieval)
      → IF basic   : standard retrieve() → generate
      → IF advanced : selected transforms → retrieve → improve → generate
      → Return AdvancedRAGResult with answer, docs, cost, route info
"""

from __future__ import annotations
from advanced_rag.retrieval_improve import (
    rerank_documents,
    compress_documents,
    evaluate_retrieval_crag,
)
from advanced_rag.query_transform import (
    rewrite_query,
    generate_multi_queries,
    decompose_question,
    generate_hyde_passage,
    extract_self_query,
)
from routing.router import RouteDecision, route_question
from retrieval.retriever import retrieve
from generation.generator import get_llm, _format_context, _prompt as generation_prompt
from evaluation.cost_tracker import CostTracker
from config import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_GROQ_MODEL,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_SEARCH_TYPE,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_TEMPERATURE,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


logger = logging.getLogger(__name__)


@dataclass
class AdvancedRAGResult:
    """Everything the UI needs from one pipeline run."""
    answer: str
    docs: List[Document]
    route: RouteDecision
    cost_summary: Dict[str, Any]
    total_latency: float = 0.0
    rewritten_query: Optional[str] = None
    sub_questions: List[str] = field(default_factory=list)

    def __iter__(self):
        """Allow unpacking: answer, docs, meta = result"""
        meta = {
            "route": self.route.route if hasattr(self.route, "route") else str(self.route),
            "reason": getattr(self.route, "reason", ""),
            "techniques": getattr(self.route, "techniques", []),
            "cost_summary": self.cost_summary,
            "latency": self.total_latency,
            "rewritten_query": self.rewritten_query,
            "sub_questions": self.sub_questions,
        }
        return iter((self.answer, self.docs, meta))

    def __getitem__(self, index):
        return tuple(self)[index]


def _deduplicate_docs(docs: List[Document]) -> List[Document]:
    """Remove duplicate chunks by (source, page, first-100-chars)."""
    seen = set()
    unique = []
    for doc in docs:
        key = (
            doc.metadata.get("source", ""),
            str(doc.metadata.get("page", "")),
            doc.page_content[:100],
        )
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def _simple_answer(
    question: str,
    model_name: str,
    temperature: float,
    tracker: CostTracker,
) -> str:
    """Answer a general-knowledge or small-talk question directly (no retrieval)."""
    llm = get_llm(model_name=model_name, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant for a bilingual banking Standard Operating Procedures (SOP) "
         "knowledge base. Answer the user's question clearly, accurately, and concisely in the same "
         "language as the user's message (Arabic or English).\n\n"
         "- For general questions, conceptual inquiries (e.g. what is RAG, reranking, vectors), or greetings, "
         "provide a clear and helpful explanation.\n"
         "- If the user is asking about specific internal bank procedures or workflows, provide a helpful general "
         "summary and mention that the internal manuals can be searched for exact unit-level procedures."),
        ("human", "{question}"),
    ])

    t0 = time.time()
    response = llm.invoke(prompt.format_messages(question=question))
    latency = time.time() - t0
    tracker.record("final_generation", model_name, response, latency)

    return response.content if hasattr(response, "content") else str(response)


def _generate_answer(
    question: str,
    docs: List[Document],
    model_name: str,
    temperature: float,
    tracker: CostTracker,
) -> str:
    """Final grounded generation using the existing bilingual prompt."""
    max_chars = 4500 if "allam" not in model_name.lower() else 3000
    context = _format_context(docs, max_total_chars=max_chars)

    llm = get_llm(model_name=model_name, temperature=temperature)

    t0 = time.time()
    response = llm.invoke(
        generation_prompt.format_messages(context=context, question=question)
    )
    latency = time.time() - t0
    tracker.record("final_generation", model_name, response, latency)

    return response.content if hasattr(response, "content") else str(response)


def advanced_rag_answer(
    question: str,
    model_name: str = DEFAULT_GROQ_MODEL,
    top_k: int = 5,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    search_type: str = DEFAULT_SEARCH_TYPE,
    source_filter: Optional[str] = None,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> AdvancedRAGResult:
    """Run the full Advanced RAG pipeline and return a rich result."""

    tracker = CostTracker()
    pipeline_t0 = time.time()

    # ── Step 1: Route ──────────────────────────────────────────────────
    route = route_question(question, model_name=model_name, tracker=tracker)
    logger.info("Route: %s | Techniques: %s | Reason: %s",
                route.route, route.techniques, route.reason)

    # ── Step 2: Simple route (no retrieval) ────────────────────────────
    if route.route == "simple":
        # Mark all advanced steps as skipped
        for step in ["rewriter", "multi_query", "decomposition", "hyde",
                     "self_query", "reranking", "compression", "crag_evaluator"]:
            tracker.record_skipped(step)

        answer = _simple_answer(question, model_name, temperature, tracker)
        return AdvancedRAGResult(
            answer=answer,
            docs=[],
            route=route,
            cost_summary=tracker.summary(),
            total_latency=time.time() - pipeline_t0,
        )

    # ── Step 3: Basic RAG route ────────────────────────────────────────
    if route.route == "basic_rag":
        for step in ["rewriter", "multi_query", "decomposition", "hyde",
                     "self_query", "reranking", "compression", "crag_evaluator"]:
            tracker.record_skipped(step)

        effective_top_k = min(top_k, 5)
        docs = retrieve(
            question,
            mode=retrieval_mode,
            top_k=effective_top_k,
            search_type=search_type,
            source_filter=source_filter,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
        )
        answer = _generate_answer(
            question, docs, model_name, temperature, tracker)
        return AdvancedRAGResult(
            answer=answer,
            docs=docs,
            route=route,
            cost_summary=tracker.summary(),
            total_latency=time.time() - pipeline_t0,
        )

    # ── Step 4: Advanced RAG route ─────────────────────────────────────
    techniques = set(route.techniques)
    effective_top_k = min(top_k, 5)
    all_docs: List[Document] = []

    # -- Query transformations --

    active_query = question
    active_source_filter = source_filter
    rewritten_query_str: Optional[str] = None
    sub_questions_list: List[str] = []

    # 4a. Rewriting
    if "rewriting" in techniques:
        active_query = rewrite_query(
            question, model_name=model_name, tracker=tracker)
        rewritten_query_str = active_query
        logger.info("Rewritten query: %s", active_query[:100])
    else:
        tracker.record_skipped("rewriter")

    # 4b. Self-Query (metadata extraction)
    if "self_query" in techniques:
        sq = extract_self_query(
            question, model_name=model_name, tracker=tracker)
        active_query = sq["query"]
        if sq["filters"].get("source"):
            active_source_filter = sq["filters"]["source"]
        logger.info("Self-query: query=%s, filters=%s",
                    active_query[:80], sq["filters"])
    else:
        tracker.record_skipped("self_query")

    # 4c. Multi-Query
    if "multi_query" in techniques:
        queries = generate_multi_queries(
            active_query, n=3, model_name=model_name, tracker=tracker)
        for q in queries:
            docs = retrieve(
                q, mode=retrieval_mode, top_k=effective_top_k,
                search_type=search_type, source_filter=active_source_filter,
                semantic_weight=semantic_weight, bm25_weight=bm25_weight,
            )
            all_docs.extend(docs)
    else:
        tracker.record_skipped("multi_query")

    # 4d. Decomposition
    if "decomposition" in techniques:
        sub_questions = decompose_question(
            question, model_name=model_name, tracker=tracker)
        sub_questions_list = sub_questions
        for sq in sub_questions:
            docs = retrieve(
                sq, mode=retrieval_mode, top_k=effective_top_k,
                search_type=search_type, source_filter=active_source_filter,
                semantic_weight=semantic_weight, bm25_weight=bm25_weight,
            )
            all_docs.extend(docs)
    else:
        tracker.record_skipped("decomposition")

    # 4e. HyDE
    if "hyde" in techniques:
        hyde_passage = generate_hyde_passage(
            question, model_name=model_name, tracker=tracker)
        docs = retrieve(
            hyde_passage, mode=retrieval_mode, top_k=effective_top_k,
            search_type=search_type, source_filter=active_source_filter,
            semantic_weight=semantic_weight, bm25_weight=bm25_weight,
        )
        all_docs.extend(docs)
    else:
        tracker.record_skipped("hyde")

    # If no multi-query/decomposition/hyde ran, do a standard retrieval with
    # the (possibly rewritten) query
    if not all_docs:
        all_docs = retrieve(
            active_query, mode=retrieval_mode, top_k=effective_top_k,
            search_type=search_type, source_filter=active_source_filter,
            semantic_weight=semantic_weight, bm25_weight=bm25_weight,
        )

    # Deduplicate
    all_docs = _deduplicate_docs(all_docs)

    # -- Retrieval improvements --

    # 4f. Reranking
    if "reranking" in techniques:
        all_docs = rerank_documents(
            active_query, all_docs, top_k=effective_top_k)
    else:
        tracker.record_skipped("reranking")

    # 4g. Contextual Compression
    if "compression" in techniques:
        all_docs = compress_documents(
            question, all_docs, model_name=model_name, tracker=tracker
        )
    else:
        tracker.record_skipped("compression")

    # 4h. CRAG
    if "crag" in techniques:
        verdict, reason = evaluate_retrieval_crag(
            question, all_docs, model_name=model_name, tracker=tracker
        )
        logger.info("CRAG verdict: %s — %s", verdict, reason)

        if verdict == "incorrect":
            # Retrieval failed — try rewriting and re-retrieving once
            logger.info("CRAG: incorrect — attempting rewrite + re-retrieve")
            rewritten = rewrite_query(
                question, model_name=model_name, tracker=tracker)
            if not rewritten_query_str:
                rewritten_query_str = rewritten
            all_docs = retrieve(
                rewritten, mode=retrieval_mode, top_k=effective_top_k,
                search_type=search_type, source_filter=active_source_filter,
                semantic_weight=semantic_weight, bm25_weight=bm25_weight,
            )
        elif verdict == "ambiguous":
            # Supplement with additional retrieval
            logger.info("CRAG: ambiguous — supplementing with rewritten query")
            rewritten = rewrite_query(
                question, model_name=model_name, tracker=tracker)
            if not rewritten_query_str:
                rewritten_query_str = rewritten
            extra_docs = retrieve(
                rewritten, mode=retrieval_mode, top_k=effective_top_k,
                search_type=search_type, source_filter=active_source_filter,
                semantic_weight=semantic_weight, bm25_weight=bm25_weight,
            )
            all_docs = _deduplicate_docs(all_docs + extra_docs)
        # verdict == "correct" → proceed with current docs
    else:
        tracker.record_skipped("crag_evaluator")

    # Trim to top_k for generation
    all_docs = all_docs[:effective_top_k]

    # ── Step 5: Final generation ───────────────────────────────────────
    answer = _generate_answer(
        question, all_docs, model_name, temperature, tracker)

    return AdvancedRAGResult(
        answer=answer,
        docs=all_docs,
        route=route,
        cost_summary=tracker.summary(),
        total_latency=time.time() - pipeline_t0,
        rewritten_query=rewritten_query_str,
        sub_questions=sub_questions_list,
    )
