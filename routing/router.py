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
You are an expert routing classifier for a bilingual enterprise RAG system querying internal bank procedure manuals:
- Central Mail & Files Procedures Manual
- Central Alarm Tasks & Procedures Manual
- Assets & Warehouse Operations Tasks & Procedures Manual

Given a user question, analyze its linguistic and structural intent, and classify it into exactly ONE route:

1. "simple":
   - Greetings, chit-chat, thanks ("مرحبا", "السلام عليكم", "صباح الخير", "شكراً", "Who are you?").
   - Out-of-corpus general AI / technical definitions that do NOT belong to internal bank manuals ("What is RAG?", "What is an LLM?", "ما هو نموذج اللغة؟", "How does vector search work?").

2. "basic_rag":
   - Use ONLY for narrow, targeted, single-fact lookups that have an exact, localized answer in a single paragraph or table row:
     - Exact form numbers or document IDs (e.g., "ما هو رقم نموذج إتلاف المواد؟")
     - A specific job role or title for a single task (e.g., "من هو الموظف المسؤول عن فتح القاصة؟", "Who verifies branch alarm signals at closing?")
     - Specific single-point contact or definition (e.g., "ما هو رمز وحدة البريد؟", "ما هي ساعات تسليم البريد؟")
   - Do NOT use basic_rag for multi-step procedures, broad workflows, or open-ended policy questions.

3. "advanced_rag":
   - Use for ANY question that requires comprehensive retrieval across multiple steps, policies, or manual sections:
     - Procedural workflows & handling cases: Questions asking "how" to handle processes or situations ("كيف يتم التعامل مع...", "ما هي خطوات وإجراءات...", "كيف يُعالج...", "How to handle inventory cases?").
     - Multi-faceted operational topics: Topics like inventory (الجرد), asset disposal (الإتلاف), loans filing (ملفات القروض), alarms and false alerts (الإنذارات الوهمية), restocking (إعادة التغذية), which involve committees, approvals, documentation, and discrepancy handling across multiple pages.
     - Ambiguous, terse, or conversational queries: Using vague keywords or pronouns (e.g., "حالات الجرد", "البريد السري", "Why is it bad?").
     - Multi-part or comparative questions: Asking for comparison, differences, or multiple requirements ("قارن بين...", "ما الفرق بين...", multi-question sentences).
     - Explicit constraints: Mentioning specific years, dates, departments, or document titles.

Technique selection guidance (for advanced_rag):
- "multi_query": Essential for broad procedural topics ("الجرد", "الإتلاف", "الإنذار", etc.) to search from multiple semantic angles and retrieve all related steps.
- "reranking": Essential whenever high precision is needed to rank chunks from multiple pages.
- "rewriting": When query is ambiguous, conversational, uses pronouns, or is very brief.
- "decomposition": When the question contains multiple separate questions, comparative clauses ("قارن", "compare"), or asks about relationships, coordination, or interactions between two or more distinct departments, units, or manuals (e.g. "بين وحدة X ووحدة Y").
- "hyde": When there is a linguistic or conceptual gap (e.g., English query searching Arabic bank procedures).
- "self_query": When specific document name, year, or page is explicitly cited.
- "compression": When chunks contain repetitive administrative headers/tables.
- "crag": When retrieval correctness needs verification.

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
