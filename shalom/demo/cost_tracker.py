"""Thread-safe LLM API cost tracking."""

import threading
import time
from typing import Any, Dict, List


class CostTracker:
    """Accumulates token usage across LLM API calls and estimates cost.

    Thread-safe: designed to be used as a ``usage_callback`` with
    :class:`~shalom.core.llm_provider.LLMProvider` which may invoke it from
    multiple evaluator threads simultaneously.
    """

    # Per-million-token pricing (USD), updated Feb 2026.
    PRICING: Dict[str, Dict[str, float]] = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: List[Dict[str, Any]] = []

    def record(self, usage: Dict[str, Any]) -> None:
        """Record a single API call's token usage (callback signature)."""
        with self._lock:
            self._calls.append({**usage, "timestamp": time.time()})

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self._calls)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            calls = list(self._calls)
        return sum(
            c.get("input_tokens", 0) + c.get("output_tokens", 0) for c in calls
        )

    @property
    def total_cost(self) -> float:
        with self._lock:
            calls = list(self._calls)
        total = 0.0
        for c in calls:
            pricing = self.PRICING.get(c.get("model", ""), {"input": 0, "output": 0})
            total += c.get("input_tokens", 0) * pricing["input"] / 1_000_000
            total += c.get("output_tokens", 0) * pricing["output"] / 1_000_000
        return total

    def summary(self) -> Dict[str, Any]:
        """Return a dict summarizing accumulated usage."""
        with self._lock:
            calls = list(self._calls)
        return {
            "total_calls": len(calls),
            "total_tokens": sum(
                c.get("input_tokens", 0) + c.get("output_tokens", 0) for c in calls
            ),
            "total_cost_usd": self.total_cost,
            "calls": calls,
        }
