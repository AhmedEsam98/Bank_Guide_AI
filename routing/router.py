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
from evaluation.cost_tracker import CostTracker
from generation.generator import get_llm
from config import DEFAULT_GROQ_MODEL
from langchain_core.prompts import ChatPromptTemplate

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


logger = logging.getLogger(__name__)

# All techniques the router may recommend
VALID_TECHNIQUES = {
    "rewriting", "multi_query", "decomposition",
    "hyde", "self_query", "reranking", "compression", "crag",
}

ROUTER_SYSTEM_PROMPT = """\
You are an intelligent routing classifier for a bilingual RAG (Retrieval-Augmented Generation) \
system that answers questions about internal bank procedure manuals (Central Mail & Files, \
Central Alarm, and Assets & Warehouse Operations).

Given a user question, classify it into exactly ONE route and, if the route is \
"advanced_rag", select which techniques to apply.

Routes:
- "simple": The question is general knowledge, greetings, casual small talk, or conceptual \
AI/technical definitions that do NOT require searching the internal bank manuals.
  Examples:
  - "Hello", "Hi", "السلام عليكم", "صباح الخير", "Who are you?", "شكراً لك"
  - "What is RAG?", "What is reranking?", "ما هو نموذج اللغة الضخم؟"
- "basic_rag": The question is a straightforward factual or procedural inquiry from the internal \
bank manuals that can be answered accurately with a single standard retrieval.
  Examples:
  - "ما هي مهام وحدة البريد المركزي والملفات؟"
  - "What is the procedure for incoming mail registration?"
  - "ما هو رقم نموذج إتلاف المواد والموجودات؟"
  - "Who verifies branch alarm signals?"
- "advanced_rag": The question is complex, ambiguous, conversational (uses ambiguous pronouns), \
contains multiple distinct sub-questions, requires comparisons across policies, mentions metadata constraints \
(like specific document names, years, pages), or requires deep procedural reasoning. Select one or more techniques.
  Examples:
  - "Why is it bad?" (vague pronoun, needs rewriting)
  - "قارن بين إجراءات إتلاف الموجودات وإجراءات التبرع بها بالتفصيل" (comparison, needs decomposition)
  - "Find alarm bypass procedures updated in 2026 for branch security" (metadata constraint, needs self_query)
  - "How does the bank prevent fraudulent mail delivery across branches?" (multi-faceted, needs multi_query / hyde)

Technique selection guidance (for advanced_rag):
- "rewriting": when the query is vague, conversational, uses pronouns without context, or is poorly worded.
- "multi_query": when the question has multiple information needs that can be searched from different angles.
- "decomposition": when the question contains multiple distinct sub-questions or asks for comparison between two or more procedures.
- "hyde": when there is a semantic or linguistic mismatch (e.g. English query where manuals are in Arabic, or high-level abstract query).
- "self_query": when the question explicitly specifies metadata like document name, page number, date, year, or department.
- "reranking": when high precision is needed among several retrieved procedural chunks.
- "compression": when retrieved chunks contain boilerplate or extraneous administrative text.
- "crag": when retrieval quality is uncertain and a quality verification check is warranted.

Respond ONLY with a valid JSON object — no markdown fences, no explanatory text:
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
    model_name: str = DEFAULT_GROQ_MODEL,
    tracker: Optional[CostTracker] = None,
) -> RouteDecision:
    """Classify *question* into a route and optional techniques list.

    On parse failure, falls back to ``basic_rag`` so the pipeline never
    breaks.
    """
    llm = get_llm(model_name=model_name, temperature=0.0)

    t0 = time.time()
    response = llm.invoke(_router_prompt.format_messages(question=question))
    latency = time.time() - t0

    # Track cost
    if tracker:
        tracker.record(step="router", model=model_name,
                       llm_response=response, latency=latency)

    # Parse JSON from response
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        # Extract substring between first '{' and last '}' to tolerate model preambles
        cleaned = raw.strip()
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx: end_idx + 1]
        elif cleaned.startswith("```"):
            cleaned = cleaned.split(
                "\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        data = json.loads(cleaned)
        route = data.get("route", "basic_rag")
        if route not in ("simple", "basic_rag", "advanced_rag"):
            route = "basic_rag"
        techniques = [t for t in data.get(
            "techniques", []) if t in VALID_TECHNIQUES]
        reason = data.get("reason", "")
        return RouteDecision(route=route, reason=reason, techniques=techniques)

    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        logger.warning(
            "Router JSON parse failed (%s), falling back to basic_rag. Raw: %s", exc, raw[:200])
        return RouteDecision(route="basic_rag", reason="parse_fallback", techniques=[])
