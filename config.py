"""
config.py
---------
Central configuration for the whole RAG pipeline (ingestion, retrieval,
generation). Keeping every path / model name / default parameter here means
the Streamlit app and the individual modules never hard-code values twice.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "pdfs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "chroma_db"
OCR_CACHE_DIR = BASE_DIR / "vectorstore" / "ocr_cache"

DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Groq / LLM
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

AVAILABLE_GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
    "openai/gpt-oss-120b",
]
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Multilingual embedding model -- important since the source PDFs are
# Arabic. BGE-M3 handles Arabic well and works fully offline via
# sentence-transformers.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda"  # Dedicated GPU acceleration (NVIDIA CUDA)

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
# Tesseract language codes: Arabic + English (manuals mix both).
OCR_LANGUAGES = "ara+eng"

# DPI used to rasterize PDF pages before running OCR. Higher = more accurate
# but slower.
OCR_DPI = 300

# If a PDF page already has more than this many extractable characters via
# direct text extraction, OCR is skipped for that page (huge speed win).
MIN_CHARS_FOR_NATIVE_TEXT = 40

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_STRATEGIES = [
    "recursive_character",
    "character",
    "token_based",
    "arabic_paragraph",
    "markdown_heading",
]

DEFAULT_CHUNK_STRATEGY = "recursive_character"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 5
DEFAULT_SEARCH_TYPE = "mmr"  # "similarity" | "mmr"
COLLECTION_NAME = "bank_manuals"
