"""
app.py
------
Streamlit front-end for the RAG pipeline (local test version).

Sidebar:
  - Pick a chunking strategy + chunk size/overlap
  - Run ingestion (OCR -> chunk -> embed -> Chroma) over whatever PDFs
    already exist in data/pdfs/ (no upload widget -- add files there
    manually, this build is for local testing only)
  - Pick Groq model, retrieval settings (top_k, search_type, source filter)

GROQ_API_KEY is read only from the environment / .env file -- it is never
entered or shown in the UI.

Main area:
  - Chat interface. Each answer shows the retrieved source chunks used.
"""

from __future__ import annotations

import os
import time

import streamlit as st

from config import (
    AVAILABLE_GROQ_MODELS,
    CHUNK_STRATEGIES,
    DATA_DIR,
    DEFAULT_BM25_WEIGHT,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_GROQ_MODEL,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_SEARCH_TYPE,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_TOP_K,
    RETRIEVAL_MODES,
)
from ingestion.ingest import run_ingestion
from retrieval.vectorstore import vectorstore_is_ready
from generation.generator import answer_question

st.set_page_config(page_title="Bank Manuals RAG", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar - configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    if not os.getenv("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY not set. Add it to a .env file next to app.py.")

    model_name = st.selectbox(
        "LLM Model",
        AVAILABLE_GROQ_MODELS,
        index=AVAILABLE_GROQ_MODELS.index(DEFAULT_GROQ_MODEL) if DEFAULT_GROQ_MODEL in AVAILABLE_GROQ_MODELS else 0,
        help="llama-3.3-70b / qwen models offer strong Arabic/English comprehension."
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    st.divider()
    st.subheader("1. Chunking strategy")
    strategy = st.selectbox("Strategy", CHUNK_STRATEGIES, index=CHUNK_STRATEGIES.index(DEFAULT_CHUNK_STRATEGY))
    chunk_size = st.number_input("Chunk size", min_value=100, max_value=4000, value=DEFAULT_CHUNK_SIZE, step=50)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=DEFAULT_CHUNK_OVERLAP, step=25)
    use_cache = st.checkbox("Reuse OCR cache (fast re-runs)", value=True)

    run_button = st.button("🚀 Run ingestion", type="primary", use_container_width=True)

    if run_button:
        existing_pdfs = list(DATA_DIR.glob("*.pdf"))
        if not existing_pdfs:
            st.error("No PDFs found in data/pdfs. Add at least one PDF to that folder first.")
        else:
            with st.spinner("Running OCR → chunking → embedding... this can take a while on first run."):
                start = time.time()
                summary = run_ingestion(
                    strategy=strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    use_cache=use_cache,
                    reset_collection=True,
                )
                elapsed = time.time() - start
            st.success(
                f"Ingested {summary['pages']} pages -> {summary['chunks']} chunks "
                f"using '{summary['strategy']}' in {elapsed:.1f}s"
            )
            st.session_state["ingestion_summary"] = summary

    st.divider()
    st.subheader("2. Retrieval")
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        RETRIEVAL_MODES,
        index=RETRIEVAL_MODES.index(DEFAULT_RETRIEVAL_MODE),
        format_func=lambda m: {
            "hybrid": "🔀 Hybrid (Dense + BM25)",
            "semantic": "🧠 Semantic (Chroma Vector)",
            "keyword": "🔍 Keyword (BM25 Lexical)",
        }.get(m, m),
        help="Hybrid combines semantic understanding with exact keyword matching.",
    )

    top_k = st.slider("Top-K chunks", 1, 15, DEFAULT_TOP_K)

    # Mode-specific options
    search_type = DEFAULT_SEARCH_TYPE
    semantic_weight = DEFAULT_SEMANTIC_WEIGHT
    bm25_weight = DEFAULT_BM25_WEIGHT

    if retrieval_mode in ("semantic", "hybrid"):
        search_type = st.radio(
            "Vector search type",
            ["mmr", "similarity"],
            index=0 if DEFAULT_SEARCH_TYPE == "mmr" else 1,
            help="MMR diversifies chunks to reduce repeated boilerplate headers/footers.",
        )

    if retrieval_mode == "hybrid":
        sem_ratio = st.slider(
            "Dense / Keyword balance",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="0.0 = pure BM25 keyword, 1.0 = pure semantic vector, 0.5 = balanced",
        )
        semantic_weight = sem_ratio
        bm25_weight = 1.0 - sem_ratio

    source_filter = None  # Search across all source documents by default

# ---------------------------------------------------------------------------
# Main area - chat
# ---------------------------------------------------------------------------
st.title("📄 Bank Procedure Manuals — RAG Assistant")
st.caption(
    "OCR-aware ingestion • configurable chunking • Semantic / Keyword / Hybrid retrieval • Groq LLM generation"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content, sources, mode)

for item in st.session_state.chat_history:
    role = item[0]
    content = item[1]
    srcs = item[2] if len(item) > 2 else None
    mode_used = item[3] if len(item) > 3 else None

    with st.chat_message(role):
        st.markdown(content)
        if srcs:
            mode_label = f" (via {mode_used} retrieval)" if mode_used else ""
            with st.expander(f"📚 Sources used{mode_label}"):
                for i, doc in enumerate(srcs, start=1):
                    st.markdown(
                        f"**{i}. {doc.metadata.get('source')} — page {doc.metadata.get('page')}** "
                        f"(`{doc.metadata.get('extraction_method', '?')}`, "
                        f"`{doc.metadata.get('chunk_strategy', '?')}`)"
                    )
                    st.text(doc.page_content[:800])

question = st.chat_input("Ask a question about the manuals (Arabic or English)...")

if question:
    st.session_state.chat_history.append(("user", question, None, None))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not vectorstore_is_ready():
            answer = "⚠️ The vector store is empty. Please run ingestion first (sidebar)."
            docs = []
        elif not os.getenv("GROQ_API_KEY"):
            answer = "⚠️ GROQ_API_KEY is not set. Add it to your .env file and restart the app."
            docs = []
        else:
            with st.spinner(f"Retrieving chunks ({retrieval_mode}) and generating answer..."):
                try:
                    answer, docs = answer_question(
                        question,
                        model_name=model_name,
                        top_k=top_k,
                        retrieval_mode=retrieval_mode,
                        search_type=search_type,
                        source_filter=source_filter,
                        semantic_weight=semantic_weight,
                        bm25_weight=bm25_weight,
                        temperature=temperature,
                    )
                except Exception as e:  # noqa: BLE001
                    answer = f"❌ Error while generating the answer: {e}"
                    docs = []

        st.markdown(answer)
        if docs:
            with st.expander(f"📚 Sources used (via {retrieval_mode} retrieval)"):
                for i, doc in enumerate(docs, start=1):
                    st.markdown(
                        f"**{i}. {doc.metadata.get('source')} — page {doc.metadata.get('page')}** "
                        f"(`{doc.metadata.get('extraction_method', '?')}`, "
                        f"`{doc.metadata.get('chunk_strategy', '?')}`)"
                    )
                    st.text(doc.page_content[:800])

    st.session_state.chat_history.append(("assistant", answer, docs, retrieval_mode))
