"""
evaluation/evaluator.py
-----------------------
LLM-as-Judge evaluator implementing 4 metrics from Section 6:
  - Context Relevance
  - Faithfulness
  - Answer Relevance
  - Correctness

Each metric returns a score (1-5) and a brief justification.
Uses a consistent rubric across all evaluations.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from generation.generator import get_llm
from evaluation.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


@dataclass
class EvalScore:
    metric: str
    score: int  # 1-5
    justification: str


# ---------------------------------------------------------------------------
# Shared rubric preamble
# ---------------------------------------------------------------------------
_RUBRIC_PREAMBLE = (
    "Score on a scale of 1-5:\n"
    "  5 = Excellent - fully satisfies the criterion\n"
    "  4 = Good - mostly satisfies with minor gaps\n"
    "  3 = Adequate - partially satisfies\n"
    "  2 = Poor - significant deficiencies\n"
    "  1 = Very Poor - fails the criterion entirely\n\n"
    "Respond ONLY with valid JSON - no markdown:\n"
    '{{"score": <1-5>, "justification": "<brief explanation>"}}'
)


# ---------------------------------------------------------------------------
# 1. Context Relevance
# ---------------------------------------------------------------------------
_CTX_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an evaluation judge. Assess whether the retrieved context "
     "chunks are relevant to answering the question.\n\n" + _RUBRIC_PREAMBLE),
    ("human",
     "Question: {question}\n\n"
     "Retrieved Context:\n{context}"),
])


def evaluate_context_relevance(
    question: str,
    docs: List[Document],
    model_name: str = "qwen/qwen3.6-27b",
    tracker: Optional[CostTracker] = None,
) -> EvalScore:
    context = "\n---\n".join(
        f"[Chunk {i+1}] {d.page_content[:500]}" for i, d in enumerate(docs)
    ) or "(no context)"

    return _run_eval(
        "context_relevance",
        _CTX_RELEVANCE_PROMPT,
        {"question": question, "context": context},
        model_name, tracker,
    )


# ---------------------------------------------------------------------------
# 2. Faithfulness
# ---------------------------------------------------------------------------
_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an evaluation judge. Assess whether every claim in the answer "
     "is supported by (grounded in) the provided context. Penalise any "
     "fabricated or unsupported information.\n\n" + _RUBRIC_PREAMBLE),
    ("human",
     "Answer:\n{answer}\n\n"
     "Context:\n{context}"),
])


def evaluate_faithfulness(
    answer: str,
    docs: List[Document],
    model_name: str = "qwen/qwen3.6-27b",
    tracker: Optional[CostTracker] = None,
) -> EvalScore:
    context = "\n---\n".join(
        f"[Chunk {i+1}] {d.page_content[:500]}" for i, d in enumerate(docs)
    ) or "(no context)"

    return _run_eval(
        "faithfulness",
        _FAITHFULNESS_PROMPT,
        {"answer": answer, "context": context},
        model_name, tracker,
    )


# ---------------------------------------------------------------------------
# 3. Answer Relevance
# ---------------------------------------------------------------------------
_ANSWER_REL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an evaluation judge. Assess whether the answer actually "
     "addresses the user's question without irrelevant content.\n\n"
     + _RUBRIC_PREAMBLE),
    ("human",
     "Question: {question}\n\n"
     "Answer:\n{answer}"),
])


def evaluate_answer_relevance(
    question: str,
    answer: str,
    model_name: str = "qwen/qwen3.6-27b",
    tracker: Optional[CostTracker] = None,
) -> EvalScore:
    return _run_eval(
        "answer_relevance",
        _ANSWER_REL_PROMPT,
        {"question": question, "answer": answer},
        model_name, tracker,
    )


# ---------------------------------------------------------------------------
# 4. Correctness
# ---------------------------------------------------------------------------
_CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an evaluation judge. Compare the answer against the reference "
     "(ground-truth) answer and assess factual correctness. If no reference "
     "is provided, assess based on general accuracy.\n\n" + _RUBRIC_PREAMBLE),
    ("human",
     "Question: {question}\n\n"
     "Answer:\n{answer}\n\n"
     "Reference answer:\n{ground_truth}"),
])


def evaluate_correctness(
    question: str,
    answer: str,
    ground_truth: str = "(no reference available)",
    model_name: str = "qwen/qwen3.6-27b",
    tracker: Optional[CostTracker] = None,
) -> EvalScore:
    return _run_eval(
        "correctness",
        _CORRECTNESS_PROMPT,
        {"question": question, "answer": answer, "ground_truth": ground_truth},
        model_name, tracker,
    )


# ---------------------------------------------------------------------------
# Run all 4 metrics in one call
# ---------------------------------------------------------------------------

def evaluate_all(
    question: str,
    answer: str,
    docs: List[Document],
    ground_truth: str = "(no reference available)",
    model_name: str = "qwen/qwen3.6-27b",
    tracker: Optional[CostTracker] = None,
) -> List[EvalScore]:
    """Run all four evaluation metrics and return a list of EvalScore."""
    return [
        evaluate_context_relevance(question, docs, model_name, tracker),
        evaluate_faithfulness(answer, docs, model_name, tracker),
        evaluate_answer_relevance(question, answer, model_name, tracker),
        evaluate_correctness(question, answer, ground_truth, model_name, tracker),
    ]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _run_eval(
    metric_name: str,
    prompt: ChatPromptTemplate,
    variables: dict,
    model_name: str,
    tracker: Optional[CostTracker],
) -> EvalScore:
    """Execute one evaluation prompt and parse the JSON response."""
    llm = get_llm(model_name=model_name, temperature=0.0)

    t0 = time.time()
    response = llm.invoke(prompt.format_messages(**variables))
    latency = time.time() - t0

    if tracker:
        tracker.record(f"eval_{metric_name}", model_name, response, latency)

    raw = response.content if hasattr(response, "content") else str(response)
    if isinstance(raw, list):
        raw = "".join(str(part) for part in raw)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        data = json.loads(cleaned)
        score = int(data.get("score", 3))
        score = max(1, min(5, score))
        return EvalScore(
            metric=metric_name,
            score=score,
            justification=data.get("justification", ""),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Eval parse failed for %s: %s", metric_name, exc)
        return EvalScore(metric=metric_name, score=3, justification="parse_error")