"""
routing/router.py
-----------------
LLM-based question classifier.  Given a user question, decides whether it
needs:
  - "simple"       → general knowledge, no document retrieval
  - "basic_rag"    → straightforward lookup, standard retrieval suffices
  - "advanced_rag" → complex/multi-part/ambiguous; also selects which
                     advanced techniques to apply

Returns a structured ``RouteDecision`` dataclass so the pipeline
orchestrator can act on it deterministically.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_core.prompts import ChatPromptTemplate
from generation.generator import get_llm
from evaluation.cost_tracker import CostTracker

logger = logging.getLogger(__name__)

# All techniques the router may recommend
VALID_TECHNIQUES = {
    "rewriting", "multi_query", "decomposition",
    "hyde", "self_query", "reranking", "compression", "crag",
}

ROUTER_SYSTEM_PROMPT = """\
You are a routing classifier for a RAG (Retrieval-Augmented Generation) system \
that answers questions about internal bank procedure manuals.

Given a user question, classify it into exactly ONE route and, if the route is \
"advanced_rag", select which techniques to apply.

Routes:
- "simple": The question is general knowledge that does NOT require searching \
the bank manuals.  Examples: "What is RAG?", "What is reranking?"
- "basic_rag": The question requires information from the bank manuals but is \
straightforward — a single standard retrieval + generation is enough.
- "advanced_rag": The question is complex, ambiguous, multi-part, or has \
metadata constraints.  Select one or more techniques from: \
rewriting, multi_query, decomposition, hyde, self_query, reranking, compression, crag.

Technique selection guidance:
- "rewriting": when the query is vague, conversational, or poorly worded.
- "multi_query": when the question has multiple information needs that can be \
searched from different angles.
- "decomposition": when the question contains multiple distinct sub-questions.
- "hyde": when there is likely a semantic mismatch between the query wording \
and the document content (e.g. user asks in English, docs are in Arabic).
- "self_query": when the question mentions specific metadata like document name, \
page number, date, or department.
- "reranking": when high precision is needed and initial retrieval may return \
partially relevant chunks.
- "compression": when retrieved chunks are likely to contain irrelevant \
boilerplate alongside the answer.
- "crag": when retrieval quality is uncertain and a quality check is warranted.

Respond ONLY with a valid JSON object — no markdown, no explanation:
{{"route": "<simple|basic_rag|advanced_rag>", "reason": "<brief explanation>", "techniques": [<list of technique strings or empty>]}}
"""

_router_prompt = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM_PROMPT),
    ("human", "{question}"),
])


@dataclass
class RouteDecision:
    route: str  # "simple" | "basic_rag" | "advanced_rag"
    reason: str
    techniques: List[str] = field(default_factory=list)


def route_question(
    question: str,
    model_name: str = "openai/gpt-oss-20b",
    tracker: Optional[CostTracker] = None,
) -> RouteDecision:
    """Classify *question* into a route and optional techniques list.

    On parse failure, falls back to ``basic_rag`` so the pipeline never
    breaks.
    """
    llm = get_llm(model_name=model_name, temperature=0.0)

    t0 = time.time()
    response = (llm).invoke(_router_prompt.format_messages(question=question))
    latency = time.time() - t0

    # Track cost
    if tracker:
        tracker.record(step="router", model=model_name, llm_response=response, latency=latency)

    # Parse JSON from response
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        # Strip markdown fences if the model wraps its answer
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        data = json.loads(cleaned)
        route = data.get("route", "basic_rag")
        if route not in ("simple", "basic_rag", "advanced_rag"):
            route = "basic_rag"
        techniques = [t for t in data.get("techniques", []) if t in VALID_TECHNIQUES]
        reason = data.get("reason", "")
        return RouteDecision(route=route, reason=reason, techniques=techniques)

    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        logger.warning("Router JSON parse failed (%s), falling back to basic_rag. Raw: %s", exc, raw[:200])
        return RouteDecision(route="basic_rag", reason="parse_fallback", techniques=[])
