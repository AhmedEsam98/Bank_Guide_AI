# 📄 Bank_Guide_AI — Bilingual Banking SOP Assistant

A high-performance, modular **Retrieval-Augmented Generation (RAG)** pipeline designed for Arabic & English banking procedure manuals (Central Mail & Files, Central Alarm, and Assets/Warehouse Operations).

Powered by **GPU-accelerated BGE-M3 Embeddings**, **ChromaDB Vector Store**, and **Groq LLMs** (Qwen 3.8 27B / ALLaM 7B / GPT-OSS).

## 🚀 Key Features
- **GPU-Accelerated Ingestion (`CUDA`)**: Blazing fast embeddings using `BAAI/bge-m3` on NVIDIA GPUs with memory-safe batching.
- **OCR-Aware Extraction**: PyMuPDF native layer extraction with automated Tesseract OCR fallback for scanned pages.
- **5 Configurable Chunking Strategies**: `recursive_character`, `arabic_paragraph`, `markdown_heading`, `character`, and `token_based`.
- **Bilingual Grounded Generation**: Ask questions in Arabic or English with verified source chunk citations (Document, Page, Strategy).
- **Interactive Streamlit UI**: Intuitive web chat interface with live parameter tuning (top-k, temperature, search type, and LLM selection).

## Project layout

```
rag_project/
├── config.py                 # all paths/models/defaults in one place
├── app.py                    # Streamlit UI
├── data/pdfs/                # put your source PDFs here
├── vectorstore/              # Chroma DB + OCR cache (auto-created)
├── ingestion/
│   ├── ocr_loader.py         # PyMuPDF text extraction + Tesseract OCR fallback
│   ├── chunking.py           # 5 interchangeable chunking strategies
│   └── ingest.py             # orchestrates: load -> chunk -> embed -> store
├── retrieval/
│   ├── vectorstore.py        # Chroma build/load with BGE-M3 embeddings
│   └── retriever.py          # similarity/MMR retriever wrapper
└── generation/
    └── generator.py          # Groq LLM + grounded prompt -> answer
```

## 1. System dependencies (needed for OCR)

Tesseract OCR (with Arabic language data) and Poppler (for PDF rasterization
via `pdf2image`, used internally by PyMuPDF's renderer path is native, but
`pdf2image` is kept as an optional utility):

- **Ubuntu/Debian**
  ```bash
  sudo apt-get update
  sudo apt-get install -y tesseract-ocr tesseract-ocr-ara poppler-utils
  ```
- **macOS (Homebrew)**
  ```bash
  brew install tesseract tesseract-lang poppler
  ```
- **Windows**: install the Tesseract installer (UB-Mannheim build, includes
  language packs) and add it to PATH; install Poppler and add its `bin/` to
  PATH.

## 2. Python setup

```bash
cd rag_project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Configure your Groq API key

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY=...
```
(You can also just paste it into the Streamlit sidebar at runtime.)

## 4. Add your PDFs

Drop the PDF files into `data/pdfs/` (or upload them via the Streamlit
sidebar once the app is running).

## 5. Run

```bash
streamlit run app.py
```

In the sidebar:
1. Enter your Groq API key.
2. Confirm your PDFs are listed.
3. Pick a chunking strategy (`recursive_character` is a good default;
   `arabic_paragraph` and `markdown_heading` are tailored to these
   numbered-procedure manuals).
4. Click **Run ingestion** (first run does OCR where needed and is the
   slowest step; results are cached by file hash so re-running is fast).
5. Ask questions in the chat box, in Arabic or English.

## CLI ingestion (without Streamlit)

```bash
python -m ingestion.ingest --strategy arabic_paragraph --chunk-size 800 --chunk-overlap 100
```

## Notes on chunking strategies

| Strategy               | Best for |
|-------------------------|----------|
| `recursive_character`   | General purpose default |
| `character`              | Simple/fast baseline, fixed-size |
| `token_based`            | Aligning chunk size with LLM token budget |
| `arabic_paragraph`       | These manuals' numbered Arabic steps ("-1", "-2"...) |
| `markdown_heading`       | Grouping whole numbered sections ("1. إجراءات...") together |

## Notes on embeddings

`BAAI/bge-m3` is used because it's strong on Arabic and works fully
offline via `sentence-transformers` (no extra API calls/cost beyond Groq
for generation).
