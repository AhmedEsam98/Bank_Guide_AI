"""
tests/test_pipeline.py
----------------------
End-to-end RAG pipeline testing runner.
Tests dynamic routing, retrieval, and Arabic answer generation across sample procedure questions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
for p in (str(project_root), str(tests_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from advanced_rag.pipeline import advanced_rag_answer

TEST_QUESTIONS = [
    "ما هي إجراءات الدخول وصلاحيات الموافقة في دليل الإنذار المركزي؟",
    "ما هي مهام قسم البريد المركزي؟",
    "ما هي إجراءات جرد الموجودات في المستودعات؟",
]


def run_pipeline_test(output_path: Path | None = None):
    if output_path is None:
        output_path = tests_dir / "rag_answers_output.txt"

    out_lines = []

    for q in TEST_QUESTIONS:
        out_lines.append(f"\n{'='*70}\nQuestion: {q}\n{'='*70}")
        try:
            res = advanced_rag_answer(q)
            route_str = res.route.route if hasattr(res.route, "route") else str(res.route)
            reason_str = getattr(res.route, "reason", "")
            out_lines.append(f"Route: {route_str}")
            out_lines.append(f"Reason: {reason_str}")
            out_lines.append(f"Num docs retrieved: {len(res.docs)}")
            for idx, doc in enumerate(res.docs[:3], 1):
                src = doc.metadata.get("source", "?")
                pg = doc.metadata.get("page", "?")
                out_lines.append(f"  Doc {idx}: [{src}] page {pg}")
            out_lines.append(f"\n--- Answer ---\n{res.answer}\n")
        except Exception as e:
            out_lines.append(f"Error: {e}")

    output_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Saved to {output_path}")


def main():
    run_pipeline_test()


if __name__ == "__main__":
    main()
