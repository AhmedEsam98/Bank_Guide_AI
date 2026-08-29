"""
generation/generator.py
--------------------------
Builds the final RAG answer: retrieved chunks + chat history + question ->
Groq LLM (via langchain-groq) -> answer with source citations.

The prompt is bilingual-aware: it instructs the model to answer in the same
language the user asked in (these manuals are Arabic, users may ask in
Arabic or English), and to ground every claim in the provided context only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in sys.path when running this script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import DEFAULT_GROQ_MODEL, GROQ_API_KEY
from retrieval.retriever import retrieve

SYSTEM_PROMPT = """You are a precise assistant answering questions about internal bank \
procedure manuals (Central Mail & Files, Central Alarm, and Assets/Warehouse Operations).

Rules:
- Answer ONLY using the information in the provided context chunks below.
- The source documents are in Arabic. If the user asks in Arabic, answer in Arabic. \
If the user asks in English, answer in English (translating/summarizing the relevant \
Arabic content faithfully).
- If the answer is not contained in the context, say clearly that the manuals do not \
cover it -- do not invent procedures, names, or numbers.
- When useful, mention which document/section the information came from (use the \
"source" and "page" metadata shown with each chunk).
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


def _format_context(docs: List[Document], max_total_chars: int = 3000) -> str:
    blocks = []
    total_chars = 0
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        content = doc.page_content.strip()
        
        # Prevent context length exceeded errors on small context models
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


def get_llm(model_name: str = DEFAULT_GROQ_MODEL, temperature: float = 0.1) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or the Streamlit sidebar."
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
    search_type: str = "mmr",
    source_filter: str | None = None,
    temperature: float = 0.1,
) -> Tuple[str, List[Document]]:
    """Full RAG call: retrieve relevant chunks, then generate a grounded
    answer with Groq. Returns (answer_text, retrieved_documents) so the UI
    can show sources alongside the answer."""

    # Keep context balanced for high accuracy across models without hitting TPM limits
    effective_top_k = top_k if top_k <= 5 else 5
    max_chars = 4500 if "allam" not in model_name.lower() else 3000

    retrieved_docs = retrieve(
        question, top_k=effective_top_k, search_type=search_type, source_filter=source_filter
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
