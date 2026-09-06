"""
config.py
---------
Central configuration for the Bank Guide AI RAG app: paths, chunking
strategies, retrieval defaults, and LLM model choices.

GROQ_API_KEY is loaded from the environment / a .env file next to this
file and is never hard-coded here.
"""

from __future__ import annotations

import os
from pathlib import Path
import torch
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "pdfs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_DIR = BASE_DIR / "data" / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Vector store (retrieval/vectorstore.py)
# ---------------------------------------------------------------------------
VECTORSTORE_DIR = CHROMA_DIR
COLLECTION_NAME = "bank_manuals"

# ---------------------------------------------------------------------------
# Embeddings (ingestion/embeddings.py)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cross-encoder reranker (multilingual for Arabic & English)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ---------------------------------------------------------------------------
# Docling (OCR + layout-aware extraction)
# ---------------------------------------------------------------------------
DOCLING_MODE = "accurate"   # "accurate" (TableFormer mode) | "standard" (fast)
DOCLING_DO_OCR = True       # True = run OCR on images/scans | False = digital text only (fast)

# OCR Model specifications:
# Engine: EasyOCR (Deep learning CRAFT text detector + CRNN recognizer)
# Languages: Arabic ('ar') and English ('en')
DOCLING_OCR_ENGINE = "EasyOCR"
DOCLING_OCR_MODEL_NAME = "EasyOCR (CRAFT detector + CRNN recognizer [ar, en])"
DOCLING_OCR_LANGUAGES = ["ar", "en"]

OCR_CACHE_DIR = BASE_DIR / "data" / "ocr_cache"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Evaluation output
# ---------------------------------------------------------------------------
EVAL_OUTPUT_DIR = BASE_DIR / "evaluation" / "results"
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API keys (read from environment / .env — never hard-coded)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# LLM models (Groq only)
# ---------------------------------------------------------------------------
AVAILABLE_GROQ_MODELS = [
    "openai/gpt-oss-20b",   # fastest, cheapest, highest daily quota
    "openai/gpt-oss-120b",  # better quality, lower quota
    "qwen/qwen3.6-27b",     # strong multilingual (good for Arabic), lower quota
]
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_TEMPERATURE = 0.0



# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_STRATEGIES = [
    "markdown_heading",
    "recursive_character",
    "character",
    "token_based",
    "arabic_paragraph",
]
DEFAULT_CHUNK_STRATEGY = "markdown_heading"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_MODES = ["hybrid", "semantic", "keyword"]
DEFAULT_RETRIEVAL_MODE = "hybrid"
DEFAULT_SEARCH_TYPE = "mmr"  # "mmr" or "similarity"
DEFAULT_TOP_K = 3
DEFAULT_SEMANTIC_WEIGHT = 0.5
DEFAULT_BM25_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# Pipeline mode (Basic vs Advanced RAG)
# ---------------------------------------------------------------------------
PIPELINE_MODES = ["basic", "advanced"]
DEFAULT_PIPELINE_MODE = "basic"