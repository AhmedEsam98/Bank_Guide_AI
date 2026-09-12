"""
generation/generator.py
--------------------------
Builds the final RAG answer: retrieved chunks + chat history + question ->
LLM (Groq via langchain-groq) -> answer with source citations.

The prompt is bilingual-aware: it instructs the model to answer in the same
language the user asked in (these manuals are Arabic, users may ask in
Arabic or English), and to ground every claim in the provided context only.

Also provides `advanced_rag_answer`, a heavier pipeline that adds:
  - routing        (decide if retrieval is even needed)
  - query rewrite   (turn the raw question into a better search query)
  - relevance grading / CRAG (drop irrelevant retrieved chunks, one LLM call)
  - compression     (condense the surviving chunks before the final answer)
"""

from __future__ import annotations
from retrieval.retriever import retrieve
from config import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_GROQ_MODEL,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_SEARCH_TYPE,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_TEMPERATURE,
    GROQ_API_KEY,
)
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is in sys.path when running this script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


SYSTEM_PROMPT = """You are a precise assistant answering questions about internal bank \
procedure manuals (Central Mail & Files, Central Alarm, and Assets/Warehouse Operations).

Rules:
- Answer ONLY using the information in the provided context chunks below.
- The source documents are in Arabic. If the user asks in Arabic, answer in Arabic. \
If the user asks in English, answer in English (translating/summarizing the relevant \
Arabic content faithfully).
- If the answer is not contained in the context, say clearly that the manuals do not \
cover it -- do not invent procedures, names, or numbers.
- Ground every fact in the provided context and cite the source document name and page number \
inline where relevant (e.g., [اسم الدليل، صفحة X] or [Document Name, Page X]). Do not add a separate "المراجع" or "Sources" section at the end of your answer.
- Keep answers structured (numbered steps) when the original content is a procedure.

Context:
{context}
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

# ---------------------------------------------------------------------------
# Advanced RAG prompts (routing / rewrite / grading)
# ---------------------------------------------------------------------------

_ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Decide how to handle a message sent to an internal bank-procedures assistant. "
            "Respond with ONLY one word, nothing else:\n"
            "- retrieve  -> the message is a real question about bank procedures, policies, "
            "or the manuals, and needs looking up\n"
            "- direct    -> the message is small talk, a greeting, thanks, or something that "
            "does not require looking anything up",
        ),
        ("human", "{question}"),
    ]
)

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's question into a clear, standalone search query optimized for "
            "retrieving relevant passages from internal bank procedure manuals (written in "
            "Arabic). Preserve the original language of the question. Respond with ONLY the "
            "rewritten query, nothing else, no quotes.",
        ),
        ("human", "{question}"),
    ]
)

_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade whether retrieved context chunks are relevant to a user's question "
            "about internal bank procedure manuals. You will be given numbered chunks and a "
            "question. Respond with ONLY a comma-separated list of the relevant chunk numbers "
            "(e.g. '1,3,4'). If none are relevant, respond with exactly: none",
        ),
        ("human", "Question: {question}\n\nChunks:\n{chunks}"),
    ]
)

_COMPRESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Condense the numbered context chunks below, keeping only the sentences relevant "
            "to the question. Preserve each chunk's [Chunk N | source: ... | page: ...] header "
            "exactly as given, followed by the condensed text for that chunk only. Do not add "
            "commentary, do not merge chunks, do not translate. If a chunk has nothing relevant, "
            "keep its header with the text '(nothing relevant)'.",
        ),
        ("human", "Question: {question}\n\nChunks:\n{chunks}"),
    ]
)


def _format_context(docs: List[Document], max_total_chars: int = 3000) -> str:
    blocks = []
    total_chars = 0
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        content = doc.page_content.strip()

        remaining_budget = max_total_chars - total_chars
        if remaining_budget <= 150:
            break
        if len(content) > remaining_budget:
            content = content[:remaining_budget] + "... [truncated]"

        block = f"[Chunk {i} | source: {src} | page: {page}]\n{content}"
        blocks.append(block)
        total_chars += len(block)
        if total_chars >= max_total_chars:
            break

    return "\n\n---\n\n".join(blocks) if blocks else "(no relevant context found)"


def get_llm(model_name: str = DEFAULT_GROQ_MODEL, temperature: float = DEFAULT_TEMPERATURE):
    api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    return ChatGroq(
        groq_api_key=api_key,
        model=model_name or DEFAULT_GROQ_MODEL,
        temperature=temperature,
    )


def answer_question(
    question: str,
    model_name: str = DEFAULT_GROQ_MODEL,
    top_k: int = 5,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    search_type: str = DEFAULT_SEARCH_TYPE,
    source_filter: str | None = None,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tuple[str, List[Document]]:
    effective_top_k = top_k if top_k <= 5 else 5
    max_chars = 4500

    retrieved_docs = retrieve(
        question,
        mode=retrieval_mode,
        top_k=effective_top_k,
        search_type=search_type,
        source_filter=source_filter,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
    )

    llm = get_llm(model_name=model_name, temperature=temperature)
    chain = _prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": _format_context(retrieved_docs, max_total_chars=max_chars),
            "question": question,
        }
    )

    return answer, retrieved_docs


# ---------------------------------------------------------------------------
# Advanced RAG helpers
# ---------------------------------------------------------------------------

def _route(llm, question: str) -> str:
    chain = _ROUTE_PROMPT | llm | StrOutputParser()
    try:
        raw = chain.invoke({"question": question}).strip().lower()
    except Exception:  # noqa: BLE001
        return "retrieve"  # fail safe: default to doing the real work
    return "direct" if "direct" in raw else "retrieve"


def _rewrite_query(llm, question: str) -> str:
    chain = _REWRITE_PROMPT | llm | StrOutputParser()
    try:
        rewritten = chain.invoke({"question": question}).strip()
    except Exception:  # noqa: BLE001
        return question
    return rewritten or question


def _numbered_chunks(docs: List[Document], max_total_chars: int = 4000) -> str:
    blocks = []
    total_chars = 0
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        content = doc.page_content.strip()
        remaining = max_total_chars - total_chars
        if remaining <= 150:
            break
        if len(content) > remaining:
            content = content[:remaining] + "... [truncated]"
        block = f"[Chunk {i} | source: {src} | page: {page}]\n{content}"
        blocks.append(block)
        total_chars += len(block)
    return "\n\n---\n\n".join(blocks)


def _grade_relevance(llm, question: str, docs: List[Document]) -> List[Document]:
    if not docs:
        return []
    chunks_text = _numbered_chunks(docs)
    chain = _GRADE_PROMPT | llm | StrOutputParser()
    try:
        raw = chain.invoke(
            {"question": question, "chunks": chunks_text}).strip().lower()
    except Exception:  # noqa: BLE001
        return docs  # fail safe: keep everything if grading breaks

    if "none" in raw and not any(ch.isdigit() for ch in raw):
        return []

    keep_indices = set()
    for tok in raw.replace(" ", "").split(","):
        if tok.isdigit():
            keep_indices.add(int(tok))

    if not keep_indices:
        return docs  # couldn't parse -> fail safe, keep everything

    return [doc for i, doc in enumerate(docs, start=1) if i in keep_indices]


def _compress_chunks(llm, question: str, docs: List[Document], max_total_chars: int = 4000) -> str:
    if not docs:
        return "(no relevant context found)"
    chunks_text = _numbered_chunks(docs, max_total_chars=max_total_chars)
    chain = _COMPRESS_PROMPT | llm | StrOutputParser()
    try:
        compressed = chain.invoke(
            {"question": question, "chunks": chunks_text}).strip()
    except Exception:  # noqa: BLE001
        return chunks_text  # fail safe: fall back to uncompressed chunks
    return compressed or chunks_text


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
) -> Tuple[str, List[Document], dict]:
    """
    Heavier RAG pipeline: routing -> query rewrite -> retrieval -> CRAG-style
    relevance grading -> compression -> final answer.

    Returns (answer, docs_used, meta) where meta describes the route decision
    and which techniques actually ran, for display in the UI.
    """
    llm = get_llm(model_name=model_name, temperature=temperature)
    meta: dict = {"techniques": []}

    # 1. Route
    route = _route(llm, question)
    meta["route"] = route

    if route == "direct":
        direct_chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a friendly assistant for an internal bank procedures tool. "
                    "Reply briefly and naturally. Do not invent bank procedure content.",
                ),
                ("human", "{question}"),
            ]
        ) | llm | StrOutputParser()
        answer = direct_chain.invoke({"question": question})
        meta["techniques"].append("routing")
        return answer, [], meta

    meta["techniques"].append("routing")

    # 2. Query rewrite
    rewritten_query = _rewrite_query(llm, question)
    meta["rewritten_query"] = rewritten_query
    meta["techniques"].append("query_rewrite")

    # 3. Retrieve (over-fetch a bit to give grading/reranking something to work with)
    fetch_k = min(max(top_k * 2, top_k + 3), 15)
    retrieved_docs = retrieve(
        rewritten_query,
        mode=retrieval_mode,
        top_k=fetch_k,
        search_type=search_type,
        source_filter=source_filter,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
    )
    meta["techniques"].append(
        "hybrid_retrieval" if retrieval_mode == "hybrid" else "retrieval")

    # 4. CRAG-style relevance grading
    graded_docs = _grade_relevance(llm, question, retrieved_docs)
    meta["techniques"].append("relevance_grading")

    if not graded_docs:
        # Fallback: nothing graded relevant, try once more with the raw question
        # and the requested retrieval mode before giving up.
        fallback_docs = retrieve(
            question,
            mode=retrieval_mode,
            top_k=top_k,
            search_type=search_type,
            source_filter=source_filter,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
        )
        graded_docs = _grade_relevance(llm, question, fallback_docs)
        meta["techniques"].append("fallback_retrieval")

    final_docs = graded_docs[:top_k]

    if not final_docs:
        answer = (
            "The manuals do not appear to cover this. "
            "Please rephrase the question or check that the right documents were ingested."
        )
        return answer, [], meta

    # 5. Compression
    compressed_context = _compress_chunks(
        llm, question, final_docs, max_total_chars=4500)
    meta["techniques"].append("compression")

    # 6. Final answer
    chain = _prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {"context": compressed_context, "question": question})

    return answer, final_docs, meta
