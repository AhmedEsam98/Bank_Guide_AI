"""
evaluation/cost_tracker.py
--------------------------
Lightweight tracker that wraps every Groq LLM call to capture tokens,
cost, latency, and model info.  Used by all Advanced RAG components
(router, query transforms, CRAG, compression, final generation) so we
can report a per-question cost breakdown (Section 7 of the task).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Groq pricing (USD per 1M tokens) - updated mid-2025.
# Source: https://groq.com/pricing/
# If a model is not listed here we fall back to a conservative estimate.
# ---------------------------------------------------------------------------
_PRICE_PER_1M: Dict[str, Dict[str, float]] = {
    "openai/gpt-oss-20b":     {"input": 0.10, "output": 0.10},
    "openai/gpt-oss-120b":    {"input": 0.30, "output": 0.30},
    "qwen/qwen3.6-27b":       {"input": 0.18, "output": 0.18},
    "qwen/qwen3.8-27b":       {"input": 0.18, "output": 0.18},
}
_DEFAULT_PRICE = {"input": 0.20, "output": 0.20}  # conservative fallback


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = _PRICE_PER_1M.get(model, _DEFAULT_PRICE)
    return (
        input_tokens * prices["input"] / 1_000_000
        + output_tokens * prices["output"] / 1_000_000
    )


@dataclass
class LLMCallRecord:
    """One logged LLM invocation."""
    step: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    executed: bool = True


@dataclass
class CostTracker:
    """Accumulates LLM-call records for a single question / pipeline run."""

    records: List[LLMCallRecord] = field(default_factory=list)

    # -- public helpers -----------------------------------------------------

    def record(
        self,
        step: str,
        model: str,
        llm_response: Any,
        latency: float,
    ) -> LLMCallRecord:
        """Extract token usage from a LangChain/Groq response and log it.

        ``llm_response`` can be:
        * An ``AIMessage`` with ``response_metadata`` / ``usage_metadata``
        * A plain string (when piped through StrOutputParser â€” no usage
          info available, so we estimate from string lengths).
        """
        input_tokens, output_tokens = _extract_tokens(llm_response)
        cost = _estimate_cost(model, input_tokens, output_tokens)
        rec = LLMCallRecord(
            step=step,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_s=round(latency, 4),
        )
        self.records.append(rec)
        return rec

    def record_skipped(self, step: str) -> LLMCallRecord:
        """Log a step that was *not* executed (for the cost table)."""
        rec = LLMCallRecord(
            step=step, model="", input_tokens=0, output_tokens=0,
            cost_usd=0.0, latency_s=0.0, executed=False,
        )
        self.records.append(rec)
        return rec

    # -- aggregation --------------------------------------------------------

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records if r.executed)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records if r.executed)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records if r.executed)

    @property
    def total_latency(self) -> float:
        return sum(r.latency_s for r in self.records if r.executed)

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary dict."""
        rows = []
        for r in self.records:
            rows.append({
                "step": r.step,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": round(r.cost_usd, 8),
                "latency_s": r.latency_s,
                "executed": r.executed,
            })
        return {
            "steps": rows,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 8),
            "total_latency_s": round(self.total_latency, 4),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_tokens(llm_response: Any) -> tuple[int, int]:
    """Best-effort token extraction from various LangChain response types."""
    # AIMessage with usage_metadata (langchain-google-genai)
    if hasattr(llm_response, "usage_metadata"):
        um = llm_response.usage_metadata
        if um:
            return (
                um.get("input_tokens", 0) if isinstance(um, dict) else getattr(um, "input_tokens", 0),
                um.get("output_tokens", 0) if isinstance(um, dict) else getattr(um, "output_tokens", 0),
            )

    # AIMessage with response_metadata.token_usage (older langchain)
    if hasattr(llm_response, "response_metadata"):
        rm = llm_response.response_metadata or {}
        tu = rm.get("token_usage") or rm.get("usage") or {}
        if tu:
            return (
                tu.get("prompt_tokens", 0),
                tu.get("completion_tokens", 0),
            )

    # Plain string â€” rough estimate (4 chars â‰ˆ 1 token)
    if isinstance(llm_response, str):
        return 0, max(1, len(llm_response) // 4)

    return 0, 0

