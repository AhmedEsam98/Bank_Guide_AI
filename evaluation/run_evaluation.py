"""
evaluation/run_evaluation.py
----------------------------
Batch runner that executes Q1-Q10 through both Basic RAG and Advanced RAG
pipelines, evaluates each with 4 metrics, collects cost/latency data,
and outputs a results table + comparison report.

Usage:
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --model qwen/qwen3.8-27b --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DEFAULT_GROQ_MODEL, BASE_DIR
from generation.generator import answer_question
from advanced_rag.pipeline import advanced_rag_answer
from evaluation.evaluator import evaluate_all
from evaluation.cost_tracker import CostTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The 10 evaluation questions (from advanced_rag_task.md Section 8)
# ---------------------------------------------------------------------------
EVAL_QUESTIONS: List[Dict[str, str]] = [
    {
        "id": "Q1",
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "focus": "Simple/general knowledge question.",
    },
    {
        "id": "Q2",
        "question": "What are the main limitations of RAG?",
        "focus": "Document-grounded factual question.",
    },
    {
        "id": "Q3",
        "question": "Why is it bad?",
        "focus": "Ambiguous/conversational query.",
    },
    {
        "id": "Q4",
        "question": "What are the challenges and failure modes of RAG?",
        "focus": "Good candidate for Multi-Query.",
    },
    {
        "id": "Q5",
        "question": "Compare RAG and fine-tuning, explain their advantages and disadvantages, and state when each should be used.",
        "focus": "Complex multi-part question; good for Decomposition.",
    },
    {
        "id": "Q6",
        "question": "How does RAG reduce hallucination?",
        "focus": "Good candidate for HyDE.",
    },
    {
        "id": "Q7",
        "question": "Find documents about RAG published after 2024.",
        "focus": "Good candidate for Self-Query.",
    },
    {
        "id": "Q8",
        "question": "What is reranking and why is it useful in RAG?",
        "focus": "Retrieval/re-ranking knowledge question.",
    },
    {
        "id": "Q9",
        "question": "Which approach should be used for a simple question versus a complex multi-part question?",
        "focus": "Routing decision question.",
    },
    {
        "id": "Q10",
        "question": "ما هي إجراءات التعامل مع البريد السري في وحدة البريد والملفات المركزية؟",
        "focus": "CRAG/retrieval-correction test; corpus-specific question.",
    },
]


def run_basic_rag(question: str, model_name: str) -> Dict[str, Any]:
    """Run a question through the basic RAG pipeline and return results."""
    tracker = CostTracker()

    t0 = time.time()
    answer, docs = answer_question(
        question,
        model_name=model_name,
        top_k=5,
        retrieval_mode="hybrid",
        search_type="similarity",
    )
    latency = time.time() - t0

    # Basic RAG doesn't have native cost tracking, so we estimate
    tracker.record(
        step="basic_rag_generation",
        model=model_name,
        llm_response=answer,  # string, so tokens will be estimated
        latency=latency,
    )

    return {
        "answer": answer,
        "docs": [{"source": d.metadata.get("source"), "page": d.metadata.get("page"),
                  "content": d.page_content[:500]}
                 for d in docs],
        "num_docs": len(docs),
        "cost_summary": tracker.summary(),
        "latency": round(latency, 2),
    }


def run_advanced_rag(question: str, model_name: str) -> Dict[str, Any]:
    """Run a question through the advanced RAG pipeline and return results."""
    t0 = time.time()
    answer, docs, meta = advanced_rag_answer(
        question,
        model_name=model_name,
        top_k=5,
        retrieval_mode="hybrid",
        search_type="similarity",
    )
    latency = time.time() - t0

    return {
        "answer": answer,
        "docs": [{"source": d.metadata.get("source"), "page": d.metadata.get("page"),
                  "content": d.page_content[:500]}
                 for d in docs],
        "num_docs": len(docs),
        "route": meta.get("route", "basic_rag"),
        "techniques": meta.get("techniques", []),
        "reason": meta.get("reason", ""),
        "cost_summary": meta.get("cost_summary", {}),
        "latency": round(latency, 2),
    }


def evaluate_results(
    question: str,
    answer: str,
    docs: list,
    model_name: str,
) -> Dict[str, Any]:
    """Run all 4 evaluation metrics on a result."""
    from langchain_core.documents import Document
    doc_objects = []
    if docs:
        for d in docs:
            if isinstance(d, dict):
                doc_objects.append(
                    Document(page_content=d.get("content", "(no content)"), metadata=d)
                )
            elif hasattr(d, "page_content"):
                doc_objects.append(d)

    scores = evaluate_all(
        question=question,
        answer=answer,
        docs=doc_objects,
        model_name=model_name,
    )

    return {s.metric: {"score": s.score, "justification": s.justification} for s in scores}


def run_full_evaluation(
    model_name: str = DEFAULT_GROQ_MODEL,
    output_path: str | None = None,
) -> Dict[str, Any]:
    """Run the complete evaluation suite and return/save results."""
    results = []

    for q in EVAL_QUESTIONS:
        qid = q["id"]
        question = q["question"]
        logger.info("=" * 60)
        logger.info("Evaluating %s: %s", qid, question[:80])

        # Basic RAG
        logger.info("[%s] Running Basic RAG...", qid)
        try:
            basic = run_basic_rag(question, model_name)
        except Exception as e:
            logger.error("[%s] Basic RAG failed: %s", qid, e)
            basic = {"answer": f"ERROR: {e}", "docs": [], "num_docs": 0,
                     "cost_summary": {}, "latency": 0}

        # Advanced RAG
        logger.info("[%s] Running Advanced RAG...", qid)
        try:
            advanced = run_advanced_rag(question, model_name)
        except Exception as e:
            logger.error("[%s] Advanced RAG failed: %s", qid, e)
            advanced = {"answer": f"ERROR: {e}", "docs": [], "num_docs": 0,
                        "route": "error", "techniques": [], "reason": str(e),
                        "cost_summary": {}, "latency": 0}

        # Evaluation (on advanced RAG answer)
        logger.info("[%s] Evaluating...", qid)
        try:
            eval_scores = evaluate_results(question, advanced["answer"], advanced["docs"], model_name)
        except Exception as e:
            logger.error("[%s] Evaluation failed: %s", qid, e)
            eval_scores = {}

        results.append({
            "id": qid,
            "question": question,
            "focus": q["focus"],
            "basic_rag": basic,
            "advanced_rag": advanced,
            "evaluation": eval_scores,
        })

        logger.info("[%s] Done. Route=%s, Techniques=%s",
                    qid, advanced.get("route", "?"), advanced.get("techniques", []))

    # Build summary
    output = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "questions": results,
        "summary": _build_summary(results),
    }

    # Save to file
    if output_path is None:
        output_dir = BASE_DIR / "evaluation" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json")

    Path(output_path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", output_path)

    # Print summary table
    _print_results_table(results)

    return output


def _build_summary(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate summary statistics across all questions."""
    routes = {}
    total_basic_cost = 0
    total_advanced_cost = 0

    for r in results:
        route = r["advanced_rag"].get("route", "unknown")
        routes[route] = routes.get(route, 0) + 1
        total_basic_cost += r["basic_rag"].get("cost_summary", {}).get("total_cost_usd", 0)
        total_advanced_cost += r["advanced_rag"].get("cost_summary", {}).get("total_cost_usd", 0)

    eval_scores = {}
    for r in results:
        for metric, data in r.get("evaluation", {}).items():
            if metric not in eval_scores:
                eval_scores[metric] = []
            eval_scores[metric].append(data.get("score", 0))

    avg_scores = {
        metric: round(sum(scores) / len(scores), 2) if scores else 0
        for metric, scores in eval_scores.items()
    }

    return {
        "route_distribution": routes,
        "total_basic_cost_usd": round(total_basic_cost, 6),
        "total_advanced_cost_usd": round(total_advanced_cost, 6),
        "avg_eval_scores": avg_scores,
    }


def _print_results_table(results: List[Dict]) -> None:
    """Print the results table from Section 14."""
    header = f"{'ID':<5} {'Route':<12} {'Techniques':<30} {'Ctx':<5} {'Faith':<6} {'Ans':<5} {'Corr':<5} {'Cost($)':<10} {'Latency':<8}"
    print("\n" + "=" * len(header))
    print("RESULTS TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        qid = r["id"]
        adv = r["advanced_rag"]
        ev = r.get("evaluation", {})

        route = adv.get("route", "?")
        techniques = ", ".join(adv.get("techniques", [])) or "-"
        ctx = ev.get("context_relevance", {}).get("score", "-")
        faith = ev.get("faithfulness", {}).get("score", "-")
        ans = ev.get("answer_relevance", {}).get("score", "-")
        corr = ev.get("correctness", {}).get("score", "-")
        cost = adv.get("cost_summary", {}).get("total_cost_usd", 0)
        latency = adv.get("latency", 0)

        print(f"{qid:<5} {route:<12} {techniques:<30} {ctx!s:<5} {faith!s:<6} {ans!s:<5} {corr!s:<5} {cost:<10.6f} {latency:<8.2f}")

    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Run Advanced RAG evaluation suite.")
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL, help="Groq model to use")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_full_evaluation(model_name=args.model, output_path=args.output)


