"""
ingestion/chunking.py
----------------------
Several interchangeable chunking strategies. Pick one by name via
`chunk_documents(docs, strategy=...)`. Keeping them side by side (instead of
committing to a single splitter) makes it easy to experiment and see which
one gives the best retrieval quality for a given document set.

Strategies:
  - recursive_character : LangChain's RecursiveCharacterTextSplitter.
                           General purpose, respects paragraph/sentence
                           boundaries where possible. Good default.
  - character           : Plain fixed-size character splitter on a single
                           separator. Simple & fast baseline.
  - token_based          : Splits by token count (tiktoken) instead of raw
                           characters -- keeps chunks aligned with what the
                           LLM actually "sees" per request.
  - arabic_paragraph     : Splits on Arabic/Latin paragraph & numbered-list
                           boundaries (e.g. "-1", "-2", "أولاً", "ثانياً").
                           Tailored to these procedure manuals, which are
                           structured as numbered steps.
  - markdown_heading     : Splits on the documents' own section headers
                           (lines that look like "N. <title>") so each
                           chunk maps to one procedure/section.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when running this script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)

from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------
def _recursive_character(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", "،", "؛", " ", ""],
    )
    return splitter.split_documents(docs)


def _character(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )
    return splitter.split_documents(docs)


def _token_based(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = TokenTextSplitter(
        chunk_size=chunk_size // 4 if chunk_size > 400 else 100,  # ~chars->tokens
        chunk_overlap=chunk_overlap // 4 if chunk_overlap > 40 else 20,
    )
    return splitter.split_documents(docs)


_ARABIC_STEP_PATTERN = re.compile(
    r"(?=\n?-\s?\d{1,2}\s)|(?=\n?\d{1,2}[\.\)]\s)|(?=\nأولاً)|(?=\nثانياً)|(?=\nثالثاً)"
)


def _arabic_paragraph(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split on numbered-step / paragraph boundaries typical of Arabic
    procedure manuals, then re-merge tiny pieces up to ~chunk_size so we
    don't end up with hundreds of one-line chunks."""
    out: List[Document] = []
    for doc in docs:
        raw_pieces = [p.strip() for p in _ARABIC_STEP_PATTERN.split(doc.page_content) if p.strip()]
        if not raw_pieces:
            raw_pieces = [doc.page_content]

        buffer = ""
        for piece in raw_pieces:
            if len(buffer) + len(piece) <= chunk_size:
                buffer = f"{buffer}\n{piece}".strip()
            else:
                if buffer:
                    out.append(Document(page_content=buffer, metadata=dict(doc.metadata)))
                buffer = piece
        if buffer:
            out.append(Document(page_content=buffer, metadata=dict(doc.metadata)))

    # apply overlap by stitching a tail of the previous chunk onto the next
    if chunk_overlap > 0:
        for i in range(1, len(out)):
            tail = out[i - 1].page_content[-chunk_overlap:]
            out[i].page_content = f"{tail}\n{out[i].page_content}"

    return out


_HEADING_PATTERN = re.compile(r"^\s*\.?\d{1,2}[\.\)]?\s+\S")


def _markdown_heading(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Group lines under the nearest numbered section heading (e.g. '1.
    إجراءات الجرد السنوي...'), then further split oversized sections with
    the recursive splitter so nothing exceeds chunk_size."""
    sections: List[Document] = []
    current_lines: List[str] = []
    current_meta = None

    def flush():
        if current_lines and current_meta is not None:
            sections.append(
                Document(page_content="\n".join(current_lines).strip(), metadata=current_meta)
            )

    for doc in docs:
        for line in doc.page_content.split("\n"):
            if _HEADING_PATTERN.match(line) and current_lines:
                flush()
                current_lines = [line]
                current_meta = dict(doc.metadata)
            else:
                if current_meta is None:
                    current_meta = dict(doc.metadata)
                current_lines.append(line)
        flush()
        current_lines, current_meta = [], None

    # Now cap section size
    refine_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return refine_splitter.split_documents([s for s in sections if s.page_content.strip()])


_STRATEGY_MAP = {
    "recursive_character": _recursive_character,
    "character": _character,
    "token_based": _token_based,
    "arabic_paragraph": _arabic_paragraph,
    "markdown_heading": _markdown_heading,
}


def chunk_documents(
    docs: List[Document],
    strategy: str = "recursive_character",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Dispatch to the requested chunking strategy."""
    if strategy not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. Choose from {list(_STRATEGY_MAP)}"
        )

    chunks = _STRATEGY_MAP[strategy](docs, chunk_size, chunk_overlap)

    # tag every chunk with strategy + a stable chunk index for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_strategy"] = strategy
        chunk.metadata["chunk_index"] = i

    return chunks
