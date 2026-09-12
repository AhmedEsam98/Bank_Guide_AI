"""
tests/verify_routing.py
Quick verification of dynamic routing and pipeline execution.
"""
from advanced_rag.pipeline import advanced_rag_answer
from routing.router import route_question
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("Testing Router...")
    q1 = "Hello, what is RAG?"
    r1 = route_question(q1)
    print(f"Q1: '{q1}' -> Route: {r1.route} | Reason: {r1.reason}")
    assert r1.route == "simple", f"Expected simple, got {r1.route}"

    q2 = "ما هي مهام وحدة البريد المركزي والملفات؟"
    r2 = route_question(q2)
    print(f"Q2: '{q2}' -> Route: {r2.route} | Reason: {r2.reason}")
    assert r2.route == "basic_rag", f"Expected basic_rag, got {r2.route}"

    q3 = "قارن بين إجراءات إتلاف الموجودات وإجراءات التبرع بها بالتفصيل"
    r3 = route_question(q3)
    print(
        f"Q3: '{q3}' -> Route: {r3.route} | Reason: {r3.reason} | Techniques: {r3.techniques}")
    assert r3.route == "advanced_rag", f"Expected advanced_rag, got {r3.route}"

    print("\nTesting Pipeline execution on simple route...")
    res = advanced_rag_answer("What is RAG?")
    print(f"Answer length: {len(res.answer)} chars")
    print(f"Route: {res.route.route}")
    print(f"Docs count: {len(res.docs)}")
    assert len(res.docs) == 0, "Simple route should have 0 docs"

    # Test tuple unpacking compatibility
    ans, docs, meta = res
    print(
        f"Unpack successful: route={meta['route']}, latency={meta['latency']:.2f}s")

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
