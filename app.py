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
  - **Pipeline mode**: Basic RAG vs Advanced RAG

GROQ_API_KEY is read only from the environment / .env file -- it is never
entered or shown in the UI.

Main area:
  - Chat interface. Each answer shows the retrieved source chunks used.
  - Advanced RAG mode shows route decision, techniques, and cost breakdown.
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
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DOCLING_DO_OCR,
    RETRIEVAL_MODES,
)
from ingestion.ingest import run_ingestion
from retrieval.vectorstore import vectorstore_is_ready
from advanced_rag.pipeline import advanced_rag_answer, AdvancedRAGResult

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Procedure Manuals — RAG Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar: ingestion controls & retrieval settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    model_name = st.selectbox(
        "LLM Model",
        AVAILABLE_GROQ_MODELS,
        index=AVAILABLE_GROQ_MODELS.index(
            DEFAULT_GROQ_MODEL) if DEFAULT_GROQ_MODEL in AVAILABLE_GROQ_MODELS else 0,
        help="llama-3.3-70b / qwen models offer strong Arabic/English comprehension."
    )
    if not os.getenv("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY not set. Add it to a .env file next to app.py.")

    temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05)

    st.divider()
    st.subheader("1. Chunking strategy")
    strategy = st.selectbox("Strategy", CHUNK_STRATEGIES,
                            index=CHUNK_STRATEGIES.index(DEFAULT_CHUNK_STRATEGY))
    chunk_size = st.number_input(
        "Chunk size", min_value=100, max_value=4000, value=DEFAULT_CHUNK_SIZE, step=50)
    chunk_overlap = st.number_input(
        "Chunk overlap", min_value=0, max_value=1000, value=DEFAULT_CHUNK_OVERLAP, step=25)
    do_ocr = st.checkbox("Enable OCR for scanned pages", value=DOCLING_DO_OCR,
                         help="Uncheck for 10x faster extraction on digital PDFs.")
    use_cache = st.checkbox(
        "Reuse extraction cache (fast re-runs)", value=True)

    run_button = st.button(
        "🚀 Run ingestion", type="primary", use_container_width=True)

    if run_button:
        existing_pdfs = list(DATA_DIR.glob("*.pdf"))
        if not existing_pdfs:
            st.error(
                "No PDFs found in data/pdfs. Add at least one PDF to that folder first.")
        else:
            with st.spinner("Running Docling extraction → chunking → embedding..."):
                start = time.time()
                summary = run_ingestion(
                    strategy=strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    use_cache=use_cache,
                    reset_collection=True,
                    do_ocr=do_ocr,
                )
                elapsed = time.time() - start
            st.success(
                f"Ingested {summary['pages']} pages → {summary['chunks']} chunks "
                f"via Docling (OCR={'ON' if do_ocr else 'OFF'}) using '{summary['strategy']}' in {elapsed:.1f}s"
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
    "OCR-aware ingestion • configurable chunking • Semantic / Keyword / Hybrid retrieval • "
    "Intelligent Dynamic Routing (Simple / Basic RAG / Advanced RAG) • Groq generation"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def render_sources_expander(docs: list, mode_label: str = "", key_prefix: str = "src"):
    if not docs:
        return
    with st.expander(f"📚 Sources used ({len(docs)} chunks{mode_label})"):
        unique_sources: dict[str, set[str]] = {}
        for doc in docs:
            s_name = doc.metadata.get("source", "Unknown")
            p_num = str(doc.metadata.get("page", "?"))
            unique_sources.setdefault(s_name, set()).add(p_num)

        for s_name, pages in unique_sources.items():
            sorted_pages = ", ".join(sorted(pages, key=lambda x: int(x) if x.isdigit() else 999))
            st.markdown(f"- 📄 **Document:** `{s_name}` — 📑 **Page:** `{sorted_pages}`")


# --- Replay previous turns ---
for turn_idx, item in enumerate(st.session_state.chat_history):
    if isinstance(item, dict):
        role = item.get("role", "assistant")
        content = item.get("content", "")
        srcs = item.get("docs")
        mode_used = item.get("retrieval_mode")
        route = item.get("route")
        reason = item.get("reason")
        techniques = item.get("techniques", [])
        rewritten_q = item.get("rewritten_query")
        latency = item.get("latency")
        cost = item.get("cost")
    else:
        role = item[0]
        content = item[1]
        srcs = item[2] if len(item) > 2 else None
        mode_used = item[3] if len(item) > 3 else None
        route = None
        reason = None
        techniques = []
        rewritten_q = None
        latency = None
        cost = None

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant" and route:
            if route == "simple":
                st.markdown(
                    '<span style="background-color: #d1fae5; color: #065f46; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">🟢 Route: Simple (Direct Response • No Document Search)</span>',
                    unsafe_allow_html=True,
                )
            elif route == "basic_rag":
                mode_str = f" • {mode_used}" if mode_used else ""
                st.markdown(
                    f'<span style="background-color: #dbeafe; color: #1e40af; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">🔵 Route: Basic RAG (Standard Retrieval{mode_str})</span>',
                    unsafe_allow_html=True,
                )
            elif route == "advanced_rag":
                tech_str = ", ".join(techniques) if techniques else "transforms"
                st.markdown(
                    f'<span style="background-color: #f3e8ff; color: #6b21a8; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">🟣 Route: Advanced RAG ({tech_str})</span>',
                    unsafe_allow_html=True,
                )

            meta_line = []
            if reason:
                meta_line.append(f"💡 **Reason:** {reason}")
            if latency:
                meta_line.append(f"⏱️ **Latency:** {latency:.2f}s")
            if cost is not None and cost > 0:
                meta_line.append(f"💰 **Est. Cost:** ${cost:.6f}")

            if meta_line:
                st.caption(" • ".join(meta_line))

            if rewritten_q:
                st.caption(f"🔎 **Rewritten Query:** _{rewritten_q}_")

        if srcs:
            render_sources_expander(
                srcs,
                mode_label=f" (via {mode_used} retrieval)" if mode_used else "",
                key_prefix=f"prev_{turn_idx}",
            )

# --- The question bar itself ---
question = st.chat_input(
    "Ask a question about the manuals (Arabic or English)...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        docs = []
        route = None
        reason = None
        techniques = []
        rewritten_q = None
        latency = None
        cost = None

        if not os.getenv("GROQ_API_KEY"):
            answer = "⚠️ GROQ_API_KEY is not set. Add it to your .env file and restart the app."
        else:
            with st.spinner(f"Routing question & generating response ({retrieval_mode} retrieval)..."):
                try:
                    result: AdvancedRAGResult = advanced_rag_answer(
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
                    answer = result.answer
                    docs = result.docs
                    route = result.route.route if hasattr(result.route, "route") else str(result.route)
                    reason = getattr(result.route, "reason", "")
                    techniques = getattr(result.route, "techniques", [])
                    rewritten_q = getattr(result, "rewritten_query", None)
                    latency = getattr(result, "total_latency", None)
                    cost = result.cost_summary.get("total_cost_usd", 0.0) if hasattr(result, "cost_summary") else 0.0

                except Exception as e:  # noqa: BLE001
                    answer = f"❌ Error while generating the answer: {e}"
                    docs = []

        st.markdown(answer)

        if route:
            if route == "simple":
                st.markdown(
                    '<span style="background-color: #d1fae5; color: #065f46; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">🟢 Route: Simple (Direct Response • No Document Search)</span>',
                    unsafe_allow_html=True,
                )
            elif route == "basic_rag":
                st.markdown(
                    f'<span style="background-color: #dbeafe; color: #1e40af; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">🔵 Route: Basic RAG (Standard Retrieval • {retrieval_mode})</span>',
                    unsafe_allow_html=True,
                )
            elif route == "advanced_rag":
                tech_str = ", ".join(techniques) if techniques else "transforms"
                st.markdown(
                    f'<span style="background-color: #f3e8ff; color: #6b21a8; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">🟣 Route: Advanced RAG ({tech_str})</span>',
                    unsafe_allow_html=True,
                )

            meta_line = []
            if reason:
                meta_line.append(f"💡 **Reason:** {reason}")
            if latency:
                meta_line.append(f"⏱️ **Latency:** {latency:.2f}s")
            if cost is not None and cost > 0:
                meta_line.append(f"💰 **Est. Cost:** ${cost:.6f}")

            if meta_line:
                st.caption(" • ".join(meta_line))

            if rewritten_q:
                st.caption(f"🔎 **Rewritten Query:** _{rewritten_q}_")

        if docs:
            render_sources_expander(
                docs,
                mode_label=f" via {retrieval_mode}",
                key_prefix=f"curr_{len(st.session_state.chat_history)}",
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "docs": docs,
            "retrieval_mode": retrieval_mode,
            "route": route,
            "reason": reason,
            "techniques": techniques,
            "rewritten_query": rewritten_q,
            "latency": latency,
            "cost": cost,
        })

