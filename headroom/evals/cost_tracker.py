"""Cost tracking for evaluation suite runs.

Tracks actual API spend per benchmark, enforces budget limits,
and estimates remaining cost before running expensive benchmarks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (as of Feb 2026)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-3-5-20241022": {"input": 1.00, "output": 5.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    # Fallback for unknown models
    "_default": {"input": 1.00, "output": 5.00},
}


@dataclass
class UsageRecord:
    """Record of a single API call's token usage."""

    benchmark: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Track and enforce API spend budget for eval runs.

    Usage:
        tracker = CostTracker(budget_usd=20.0)
        tracker.record("gsm8k", "gpt-4o-mini", input_tokens=5000, output_tokens=1000)
        if not tracker.check_budget():
            print("Budget exceeded!")
    """

    budget_usd: float = 20.0
    records: list[UsageRecord] = field(default_factory=list)

    @property
    def spent_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.spent_usd)

    def _get_pricing(self, model: str) -> dict[str, float]:
        """Get pricing for a model, falling back to default."""
        return MODEL_PRICING.get(model, MODEL_PRICING["_default"])

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Compute cost in USD for given token counts."""
        pricing = self._get_pricing(model)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def record(
        self, benchmark: str, model: str, input_tokens: int, output_tokens: int
    ) -> UsageRecord:
        """Record token usage from an API call."""
        cost = self._compute_cost(model, input_tokens, output_tokens)
        rec = UsageRecord(
            benchmark=benchmark,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self.records.append(rec)
        logger.debug(f"Cost: ${cost:.4f} ({benchmark}, {input_tokens} in / {output_tokens} out)")
        return rec

    def check_budget(self) -> bool:
        """Return True if within budget."""
        return self.spent_usd < self.budget_usd

    def estimate_cost(
        self,
        model: str,
        n_samples: int,
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 100,
        multiplier: int = 2,  # baseline + headroom = 2x calls
    ) -> float:
        """Estimate cost for a benchmark run before executing it."""
        total_input = n_samples * avg_input_tokens * multiplier
        total_output = n_samples * avg_output_tokens * multiplier
        return self._compute_cost(model, total_input, total_output)

    def can_afford(
        self,
        model: str,
        n_samples: int,
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 100,
        multiplier: int = 2,
    ) -> bool:
        """Check if we can afford a benchmark run within remaining budget."""
        estimated = self.estimate_cost(
            model, n_samples, avg_input_tokens, avg_output_tokens, multiplier
        )
        return estimated <= self.remaining_usd

    def summary(self) -> dict[str, Any]:
        """Return summary of spending."""
        by_benchmark: dict[str, float] = {}
        total_input = 0
        total_output = 0
        for r in self.records:
            by_benchmark[r.benchmark] = by_benchmark.get(r.benchmark, 0) + r.cost_usd
            total_input += r.input_tokens
            total_output += r.output_tokens

        return {
            "budget_usd": self.budget_usd,
            "spent_usd": round(self.spent_usd, 4),
            "remaining_usd": round(self.remaining_usd, 4),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "by_benchmark": {k: round(v, 4) for k, v in by_benchmark.items()},
            "n_calls": len(self.records),
        }

    def print_summary(self) -> None:
        """Print a formatted cost summary."""
        s = self.summary()
        print(f"\n{'Cost Summary':=^50}")
        print(f"  Budget:    ${s['budget_usd']:.2f}")
        print(f"  Spent:     ${s['spent_usd']:.4f}")
        print(f"  Remaining: ${s['remaining_usd']:.4f}")
        print(f"  API calls: {s['n_calls']}")
        print(f"  Tokens:    {s['total_input_tokens']:,} in / {s['total_output_tokens']:,} out")
        if s["by_benchmark"]:
            print(f"  {'Breakdown:':-^40}")
            for name, cost in sorted(s["by_benchmark"].items(), key=lambda x: -x[1]):
                print(f"    {name:<25} ${cost:.4f}")
        print("=" * 50)
