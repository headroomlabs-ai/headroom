#!/usr/bin/env python3
"""Latency benchmark for Headroom compression pipeline.

Measures compression overhead across content types and input sizes,
profiles individual transform stages, and computes cost-benefit analysis
to answer: "Does the token savings outweigh added processing time?"

Usage:
    # Run with terminal output (default)
    python benchmarks/bench_latency.py

    # Save markdown report
    python benchmarks/bench_latency.py --output docs/LATENCY_BENCHMARKS.md

    # Save JSON results
    python benchmarks/bench_latency.py --json latency_results.json

    # Custom iterations
    python benchmarks/bench_latency.py --iterations 50

    # Run specific content type only
    python benchmarks/bench_latency.py --scenario json
    python benchmarks/bench_latency.py --scenario code
    python benchmarks/bench_latency.py --scenario text
    python benchmarks/bench_latency.py --scenario logs
    python benchmarks/bench_latency.py --scenario agentic

Scenarios:
    json     - JSON arrays via SmartCrusher (100-5K items)
    code     - Python source via CodeCompressor (50-1000 lines)
    text     - Plain text/RAG via Kompress fallback (1K-50K tokens)
    logs     - Structured logs via LogCompressor (100-5K entries)
    agentic  - Multi-turn agent conversations (10-100 turns)
    rag      - RAG conversations with large context (5K-50K tokens)
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure benchmarks package is importable when running as script
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from benchmarks.scenarios.conversations import (  # noqa: E402
    generate_agentic_conversation,
    generate_rag_conversation,
)
from benchmarks.scenarios.tool_outputs import (  # noqa: E402
    generate_api_responses,
    generate_database_rows,
    generate_log_entries,
    generate_search_results,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LLM prefill rates (ms per input token) for cost-benefit analysis.
# These are conservative estimates based on published benchmarks and represent
# the incremental TTFT contribution per additional input token.
MODEL_PROFILES: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {
        "ms_per_token": 0.01,
        "price_per_mtok_input": 0.15,
        "label": "GPT-4o Mini",
    },
    "gpt-4o": {
        "ms_per_token": 0.03,
        "price_per_mtok_input": 2.50,
        "label": "GPT-4o",
    },
    "claude-sonnet-4-5": {
        "ms_per_token": 0.03,
        "price_per_mtok_input": 3.00,
        "label": "Claude Sonnet 4.5",
    },
    "claude-opus-4": {
        "ms_per_token": 0.08,
        "price_per_mtok_input": 15.00,
        "label": "Claude Opus 4",
    },
}

# Reference model for the main report table
REFERENCE_MODEL = "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A benchmark scenario to measure."""

    name: str
    content_type: str  # json, code, text, logs, agentic, rag
    size_label: str  # Human-readable size (e.g., "100 items", "50 turns")
    messages: list[dict[str, Any]]
    model_limit: int = 200_000  # Context limit for pipeline


@dataclass
class TransformTiming:
    """Timing for a single transform within the pipeline."""

    name: str
    durations_ms: list[float] = field(default_factory=list)

    @property
    def p50_ms(self) -> float:
        if not self.durations_ms:
            return 0.0
        s = sorted(self.durations_ms)
        return s[len(s) // 2]

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.durations_ms) if self.durations_ms else 0.0


@dataclass
class LatencyResult:
    """Result of benchmarking a single scenario."""

    scenario_name: str
    content_type: str
    size_label: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    compression_ratio: float
    num_messages: int
    timings_ms: list[float]  # Full pipeline timings per iteration
    transform_timings: dict[str, TransformTiming] = field(default_factory=dict)
    transforms_applied: list[str] = field(default_factory=list)

    @property
    def p50_ms(self) -> float:
        s = sorted(self.timings_ms)
        return s[len(s) // 2]

    @property
    def p95_ms(self) -> float:
        s = sorted(self.timings_ms)
        idx = int(math.ceil(0.95 * len(s))) - 1
        return s[max(0, idx)]

    @property
    def p99_ms(self) -> float:
        s = sorted(self.timings_ms)
        idx = int(math.ceil(0.99 * len(s))) - 1
        return s[max(0, idx)]

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.timings_ms)

    @property
    def stddev_ms(self) -> float:
        return statistics.stdev(self.timings_ms) if len(self.timings_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.timings_ms)

    @property
    def max_ms(self) -> float:
        return max(self.timings_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "content_type": self.content_type,
            "size_label": self.size_label,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "compression_ratio": self.compression_ratio,
            "num_messages": self.num_messages,
            "iterations": len(self.timings_ms),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "stddev_ms": round(self.stddev_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "transforms_applied": self.transforms_applied,
            "transform_breakdown": {
                name: {
                    "p50_ms": round(tt.p50_ms, 3),
                    "mean_ms": round(tt.mean_ms, 3),
                }
                for name, tt in self.transform_timings.items()
            },
        }


# ---------------------------------------------------------------------------
# Code generation (for CodeCompressor scenarios)
# ---------------------------------------------------------------------------


def _generate_python_function(name: str, lines: int) -> str:
    """Generate a realistic Python function."""
    parts = [f"def {name}(data: list[dict], config: dict | None = None) -> dict:"]
    parts.append(f'    """Process {name.replace("_", " ")} and return results."""')
    parts.append("    if config is None:")
    parts.append("        config = {}")
    parts.append(f'    results = {{"function": "{name}", "items": []}}')
    parts.append("    errors = []")
    parts.append("")

    # Fill body to target line count
    for i in range(max(0, lines - 12)):
        kind = i % 5
        if kind == 0:
            parts.append(f"    for item in data[{i}:{i + 10}]:")
            parts.append(f'        value = item.get("field_{i}", None)')
        elif kind == 1:
            parts.append(f"    if len(results['items']) > {i * 10}:")
            parts.append('        results["overflow"] = True')
        elif kind == 2:
            parts.append("    try:")
            parts.append(f"        computed = sum(x.get('value', 0) for x in data[:{i + 5}])")
            parts.append(f'        results["computed_{i}"] = computed')
        elif kind == 3:
            parts.append("    except (KeyError, TypeError) as exc:")
            parts.append(f'        errors.append({{"step": {i}, "error": str(exc)}})')
        else:
            parts.append(f"    # Step {i}: aggregate intermediate results")
            parts.append(f'    results["step_{i}"] = len(data)')

    parts.append("")
    parts.append('    results["errors"] = errors')
    parts.append("    return results")
    return "\n".join(parts)


def generate_python_code(target_lines: int) -> str:
    """Generate a realistic Python module of approximately `target_lines` lines."""
    sections = [
        '"""Auto-generated benchmark module for code compression testing."""',
        "",
        "from __future__ import annotations",
        "",
        "import json",
        "import logging",
        "import os",
        "from dataclasses import dataclass, field",
        "from typing import Any",
        "",
        "logger = logging.getLogger(__name__)",
        "",
        "",
        "@dataclass",
        "class ProcessingConfig:",
        '    """Configuration for data processing."""',
        "",
        "    batch_size: int = 100",
        "    max_retries: int = 3",
        "    timeout_seconds: float = 30.0",
        "    output_format: str = 'json'",
        "    debug: bool = False",
        "",
        "",
    ]

    current_lines = len(sections)
    func_idx = 0

    while current_lines < target_lines:
        remaining = target_lines - current_lines
        func_lines = min(remaining, random.randint(15, 40))
        func_name = f"process_batch_{func_idx}"
        func_code = _generate_python_function(func_name, func_lines)
        sections.append(func_code)
        sections.append("")
        sections.append("")
        current_lines += func_lines + 2
        func_idx += 1

    return "\n".join(sections[:target_lines])


# ---------------------------------------------------------------------------
# Plain text generation
# ---------------------------------------------------------------------------


def generate_plain_text(target_tokens: int) -> str:
    """Generate realistic plain text content (technical documentation)."""
    # ~4 chars per token
    target_chars = target_tokens * 4

    paragraphs = [
        "The system architecture follows a microservices pattern with clear separation of concerns. "
        "Each service owns its data store and communicates through well-defined APIs. Event-driven "
        "messaging handles asynchronous workflows, while synchronous REST APIs serve real-time "
        "requests. The API gateway handles routing, authentication, and rate limiting at the edge.",
        "Database optimization is critical for maintaining low-latency responses under load. "
        "We use connection pooling with a minimum of 10 and maximum of 100 connections per service. "
        "Read replicas handle analytics queries to avoid impacting transactional workloads. "
        "Indexes are maintained on frequently queried columns with regular analysis of query plans.",
        "The caching layer uses a tiered approach: L1 in-memory caches with a 60-second TTL for "
        "hot data, L2 Redis caches with a 5-minute TTL for frequently accessed resources, and L3 "
        "CDN caching for static assets. Cache invalidation follows a pub/sub pattern to ensure "
        "consistency across service instances without requiring cache stampede protection.",
        "Monitoring and observability are built into every service from day one. Structured logging "
        "with correlation IDs enables distributed tracing across service boundaries. Metrics are "
        "collected via Prometheus and visualized in Grafana dashboards. Alerts are configured for "
        "SLO violations with appropriate severity levels and escalation paths.",
        "The deployment pipeline uses blue-green deployments with automated canary analysis. Each "
        "deployment is validated against health checks, latency percentiles, and error rate thresholds "
        "before traffic is shifted. Rollback is automated if any SLO is breached during the canary "
        "window, typically set to 15 minutes for non-critical services.",
        "Security follows a defense-in-depth strategy with multiple layers of protection. All "
        "inter-service communication uses mTLS with certificate rotation every 90 days. API "
        "authentication uses short-lived JWT tokens with refresh token rotation. Secrets are "
        "managed through HashiCorp Vault with automatic rotation policies.",
        "Error handling follows a consistent pattern across all services. Transient errors trigger "
        "exponential backoff with jitter, starting at 100ms and capping at 30 seconds. Circuit "
        "breakers prevent cascade failures by opening after 5 consecutive failures and attempting "
        "a half-open state after 60 seconds. All errors are classified by severity and tracked "
        "as structured events for post-incident analysis.",
        "Performance testing is integrated into the CI/CD pipeline. Load tests run against a "
        "staging environment that mirrors production topology. Baseline metrics are captured for "
        "each release candidate and compared against the previous stable release. Regressions "
        "greater than 10% in p99 latency automatically block the deployment.",
    ]

    result: list[str] = []
    current_chars = 0
    while current_chars < target_chars:
        para = random.choice(paragraphs)
        result.append(para)
        result.append("")
        current_chars += len(para) + 1

    return "\n".join(result)[:target_chars]


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------


def _wrap_as_tool_message(content: str) -> list[dict[str, Any]]:
    """Wrap content as a minimal tool-call conversation."""
    return [
        {"role": "system", "content": "You are a helpful assistant.\n\nCurrent date: 2025-01-06"},
        {"role": "user", "content": "Analyze the following data."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_bench_1",
                    "type": "function",
                    "function": {"name": "get_data", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_bench_1", "content": content},
    ]


def generate_scenarios(content_types: list[str] | None = None) -> list[Scenario]:
    """Generate all benchmark scenarios.

    Args:
        content_types: Limit to specific types. None = all.

    Returns:
        List of Scenario objects ready for benchmarking.
    """
    all_types = {"json", "code", "text", "logs", "agentic", "rag"}
    types = set(content_types) if content_types else all_types

    scenarios: list[Scenario] = []
    random.seed(42)

    # --- JSON arrays (SmartCrusher path) ---
    if "json" in types:
        for n, label in [
            (100, "100 items"),
            (500, "500 items"),
            (1_000, "1K items"),
            (5_000, "5K items"),
        ]:
            data = generate_search_results(n)
            msgs = _wrap_as_tool_message(json.dumps(data))
            scenarios.append(
                Scenario(
                    name=f"JSON: Search Results ({label})",
                    content_type="json",
                    size_label=label,
                    messages=msgs,
                )
            )

        # Also test API responses and database rows
        data = generate_api_responses(500)
        msgs = _wrap_as_tool_message(json.dumps(data))
        scenarios.append(
            Scenario(
                name="JSON: API Responses (500 items)",
                content_type="json",
                size_label="500 items",
                messages=msgs,
            )
        )

        data = generate_database_rows(1_000, table_type="metrics")
        msgs = _wrap_as_tool_message(json.dumps(data))
        scenarios.append(
            Scenario(
                name="JSON: Database Rows (1K rows)",
                content_type="json",
                size_label="1K rows",
                messages=msgs,
            )
        )

        # --- String arrays (NEW: universal JSON) ---
        for n, label in [(100, "100 strings"), (500, "500 strings"), (1_000, "1K strings")]:
            strings = [f"GET /api/endpoint_{i % 20} 200 OK" for i in range(n)]
            # Inject some errors
            for j in range(0, n, max(1, n // 5)):
                strings[j] = f"GET /api/endpoint_{j} 500 error: internal server error"
            msgs = _wrap_as_tool_message(json.dumps(strings))
            scenarios.append(
                Scenario(
                    name=f"JSON: String Array ({label})",
                    content_type="json",
                    size_label=label,
                    messages=msgs,
                )
            )

        # --- Number arrays (NEW: universal JSON) ---
        for n, label in [(200, "200 numbers"), (1_000, "1K numbers")]:
            numbers = [42.0 + random.gauss(0, 5) for _ in range(n)]
            # Inject outliers
            numbers[n // 4] = 999.9
            numbers[3 * n // 4] = -500.0
            msgs = _wrap_as_tool_message(json.dumps(numbers))
            scenarios.append(
                Scenario(
                    name=f"JSON: Number Array ({label})",
                    content_type="json",
                    size_label=label,
                    messages=msgs,
                )
            )

        # --- Mixed arrays (NEW: universal JSON) ---
        mixed = (
            [{"id": i, "status": "active"} for i in range(100)]
            + [f"log: request {i} completed" for i in range(100)]
            + [random.gauss(50, 10) for _ in range(50)]
        )
        msgs = _wrap_as_tool_message(json.dumps(mixed))
        scenarios.append(
            Scenario(
                name="JSON: Mixed Array (250 items)",
                content_type="json",
                size_label="250 items",
                messages=msgs,
            )
        )

        # --- Flat objects (NEW: object compression) ---
        flat_obj = {f"config_{i}": f"value_{i} " * 20 for i in range(100)}
        msgs = _wrap_as_tool_message(json.dumps(flat_obj))
        scenarios.append(
            Scenario(
                name="JSON: Flat Object (100 keys)",
                content_type="json",
                size_label="100 keys",
                messages=msgs,
            )
        )

        # --- Nested objects with arrays (recursion) ---
        nested = {
            "search_results": generate_search_results(200),
            "log_entries": [f"INFO: processed request {i}" for i in range(100)],
            "metrics": [random.gauss(50, 5) for _ in range(300)],
            "metadata": {"total": 600, "query": "benchmark test"},
        }
        msgs = _wrap_as_tool_message(json.dumps(nested))
        scenarios.append(
            Scenario(
                name="JSON: Nested Object (3 arrays)",
                content_type="json",
                size_label="600 items nested",
                messages=msgs,
            )
        )

    # --- Code (CodeCompressor path) ---
    if "code" in types:
        for lines, label in [
            (50, "~50 lines"),
            (200, "~200 lines"),
            (500, "~500 lines"),
            (1_000, "~1K lines"),
        ]:
            code = generate_python_code(lines)
            msgs = _wrap_as_tool_message(code)
            scenarios.append(
                Scenario(
                    name=f"Code: Python ({label})",
                    content_type="code",
                    size_label=label,
                    messages=msgs,
                )
            )

    # --- Plain text (Kompress fallback path) ---
    if "text" in types:
        for tokens, label in [
            (1_000, "1K tokens"),
            (5_000, "5K tokens"),
            (20_000, "20K tokens"),
            (50_000, "50K tokens"),
        ]:
            text = generate_plain_text(tokens)
            msgs = _wrap_as_tool_message(text)
            scenarios.append(
                Scenario(
                    name=f"Text: Documentation ({label})",
                    content_type="text",
                    size_label=label,
                    messages=msgs,
                )
            )

    # --- Log entries (LogCompressor path) ---
    if "logs" in types:
        for n, label in [
            (100, "100 entries"),
            (500, "500 entries"),
            (1_000, "1K entries"),
            (5_000, "5K entries"),
        ]:
            logs = generate_log_entries(n)
            msgs = _wrap_as_tool_message(json.dumps(logs))
            scenarios.append(
                Scenario(
                    name=f"Logs: Structured ({label})",
                    content_type="logs",
                    size_label=label,
                    messages=msgs,
                )
            )

    # --- Agentic conversations (full pipeline) ---
    if "agentic" in types:
        for turns, items, label in [
            (10, 50, "10 turns"),
            (25, 50, "25 turns"),
            (50, 50, "50 turns"),
            (100, 30, "100 turns"),
        ]:
            random.seed(42)
            msgs = generate_agentic_conversation(
                turns=turns, tool_calls_per_turn=2, items_per_tool_response=items
            )
            # Set a model_limit that forces IntelligentContext to kick in
            limit = max(50_000, turns * 2_000)
            scenarios.append(
                Scenario(
                    name=f"Agentic: Multi-tool ({label})",
                    content_type="agentic",
                    size_label=label,
                    messages=msgs,
                    model_limit=limit,
                )
            )

    # --- RAG conversations ---
    if "rag" in types:
        for tokens, queries, label in [
            (5_000, 3, "5K context"),
            (20_000, 5, "20K context"),
            (50_000, 5, "50K context"),
        ]:
            random.seed(42)
            msgs = generate_rag_conversation(context_tokens=tokens, num_queries=queries)
            scenarios.append(
                Scenario(
                    name=f"RAG: Document QA ({label})",
                    content_type="rag",
                    size_label=label,
                    messages=msgs,
                )
            )

    return scenarios


# ---------------------------------------------------------------------------
# Profiling pipeline
# ---------------------------------------------------------------------------


class ProfilingPipeline:
    """Wraps TransformPipeline to record per-transform timing."""

    def __init__(self) -> None:
        from headroom.config import HeadroomConfig
        from headroom.transforms.pipeline import TransformPipeline

        self.config = HeadroomConfig()
        self.pipeline = TransformPipeline(config=self.config)
        self.last_transform_timings: dict[str, float] = {}

    def apply(
        self,
        messages: list[dict[str, Any]],
        model: str = "benchmark-model",
        model_limit: int = 200_000,
    ) -> Any:
        """Apply pipeline with per-transform timing.

        Returns the TransformResult from the pipeline.
        Per-transform timings are stored in self.last_transform_timings.
        """
        from headroom.tokenizer import Tokenizer
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import deep_copy_messages

        tokenizer = Tokenizer(get_tokenizer(model), model)
        current_messages = deep_copy_messages(messages)
        self.last_transform_timings = {}

        for transform in self.pipeline.transforms:
            if not transform.should_apply(current_messages, tokenizer, model_limit=model_limit):
                self.last_transform_timings[transform.name] = 0.0
                continue

            t0 = time.perf_counter_ns()
            result = transform.apply(current_messages, tokenizer, model_limit=model_limit)
            t1 = time.perf_counter_ns()

            self.last_transform_timings[transform.name] = (t1 - t0) / 1_000_000  # ns → ms
            current_messages = result.messages

        # Compute final token counts
        tokens_before = tokenizer.count_messages(messages)
        tokens_after = tokenizer.count_messages(current_messages)

        # Return a lightweight result object
        return _PipelineResult(
            messages=current_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=[
                name for name, dur in self.last_transform_timings.items() if dur > 0
            ],
        )


@dataclass
class _PipelineResult:
    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    transforms_applied: list[str]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_scenario(
    pipeline: ProfilingPipeline,
    scenario: Scenario,
    iterations: int = 20,
    warmup: int = 3,
) -> LatencyResult:
    """Run a single scenario through the pipeline multiple times.

    Args:
        pipeline: Profiling pipeline instance.
        scenario: The scenario to benchmark.
        iterations: Number of measured iterations.
        warmup: Number of warmup iterations (not counted).

    Returns:
        LatencyResult with all timing data.
    """
    # Warmup (exercises JIT, caches, lazy inits)
    for _ in range(warmup):
        pipeline.apply(scenario.messages, model_limit=scenario.model_limit)

    # Measured runs
    timings_ms: list[float] = []
    transform_timings: dict[str, TransformTiming] = {}
    last_result = None

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        result = pipeline.apply(scenario.messages, model_limit=scenario.model_limit)
        t1 = time.perf_counter_ns()

        total_ms = (t1 - t0) / 1_000_000
        timings_ms.append(total_ms)
        last_result = result

        # Record per-transform timings
        for name, dur_ms in pipeline.last_transform_timings.items():
            if name not in transform_timings:
                transform_timings[name] = TransformTiming(name=name)
            transform_timings[name].durations_ms.append(dur_ms)

    assert last_result is not None

    tokens_saved = last_result.tokens_before - last_result.tokens_after
    ratio = tokens_saved / last_result.tokens_before if last_result.tokens_before > 0 else 0.0

    return LatencyResult(
        scenario_name=scenario.name,
        content_type=scenario.content_type,
        size_label=scenario.size_label,
        tokens_before=last_result.tokens_before,
        tokens_after=last_result.tokens_after,
        tokens_saved=tokens_saved,
        compression_ratio=ratio,
        num_messages=len(scenario.messages),
        timings_ms=timings_ms,
        transform_timings=transform_timings,
        transforms_applied=last_result.transforms_applied,
    )


def run_all(
    scenarios: list[Scenario],
    iterations: int = 20,
    warmup: int = 3,
    verbose: bool = False,
) -> list[LatencyResult]:
    """Run all scenarios and return results.

    Args:
        scenarios: List of scenarios to benchmark.
        iterations: Measured iterations per scenario.
        warmup: Warmup iterations per scenario.
        verbose: Print progress.

    Returns:
        List of LatencyResult objects.
    """
    pipeline = ProfilingPipeline()
    results: list[LatencyResult] = []

    for i, scenario in enumerate(scenarios, 1):
        if verbose:
            print(f"  [{i}/{len(scenarios)}] {scenario.name}...", end=" ", flush=True)

        result = run_scenario(pipeline, scenario, iterations=iterations, warmup=warmup)
        results.append(result)

        if verbose:
            print(
                f"{result.p50_ms:.1f}ms (p50), "
                f"{result.compression_ratio:.0%} compression, "
                f"{result.tokens_saved:,} tokens saved"
            )

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _fmt_ms(ms: float) -> str:
    """Format milliseconds with appropriate precision."""
    if ms < 0:
        return f"-{_fmt_ms(-ms)}"
    if ms < 0.01:
        return "<0.01"
    if ms < 1.0:
        return f"{ms:.2f}"
    if ms < 100.0:
        return f"{ms:.1f}"
    return f"{ms:.0f}"


def _fmt_tokens(n: int) -> str:
    """Format token count with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_terminal_report(results: list[LatencyResult]) -> str:
    """Format results as a terminal-friendly report."""
    lines: list[str] = []

    lines.append("")
    lines.append("=" * 100)
    lines.append("  HEADROOM LATENCY BENCHMARK")
    lines.append("=" * 100)
    lines.append("")

    # --- Compression Overhead Table ---
    lines.append("COMPRESSION OVERHEAD BY SCENARIO")
    lines.append("-" * 100)
    header = (
        f"{'Scenario':<40} {'Tokens In':>10} {'Saved':>8} {'Ratio':>7} "
        f"{'p50':>8} {'p95':>8} {'p99':>8} {'Mean':>8}"
    )
    lines.append(header)
    lines.append("-" * 100)

    current_type = ""
    for r in results:
        if r.content_type != current_type:
            if current_type:
                lines.append("")
            current_type = r.content_type

        row = (
            f"{r.scenario_name:<40} "
            f"{_fmt_tokens(r.tokens_before):>10} "
            f"{_fmt_tokens(r.tokens_saved):>8} "
            f"{r.compression_ratio:>6.0%} "
            f"{_fmt_ms(r.p50_ms) + 'ms':>8} "
            f"{_fmt_ms(r.p95_ms) + 'ms':>8} "
            f"{_fmt_ms(r.p99_ms) + 'ms':>8} "
            f"{_fmt_ms(r.mean_ms) + 'ms':>8}"
        )
        lines.append(row)

    lines.append("")
    lines.append("")

    # --- Per-Transform Breakdown (for agentic/rag scenarios) ---
    pipeline_results = [r for r in results if r.transform_timings]
    if pipeline_results:
        lines.append("PER-TRANSFORM BREAKDOWN (selected scenarios)")
        lines.append("-" * 80)
        header = f"{'Scenario':<40} {'Transform':<20} {'p50 (ms)':>10} {'% Total':>10}"
        lines.append(header)
        lines.append("-" * 80)

        for r in pipeline_results:
            total_p50 = r.p50_ms
            for tname, tt in r.transform_timings.items():
                pct = (tt.p50_ms / total_p50 * 100) if total_p50 > 0 else 0
                lines.append(
                    f"{r.scenario_name:<40} {tname:<20} {_fmt_ms(tt.p50_ms):>9}ms {pct:>9.0f}%"
                )
            lines.append("")

    # --- Cost-Benefit Analysis ---
    lines.append("")
    lines.append("COST-BENEFIT ANALYSIS")
    lines.append("-" * 100)
    model = MODEL_PROFILES[REFERENCE_MODEL]
    lines.append(f"Reference model: {model['label']} ({model['ms_per_token']}ms/token prefill)")
    lines.append("")

    header = (
        f"{'Scenario':<40} {'Compress':>10} {'LLM Saved':>10} {'Net Benefit':>12} {'$/1K Reqs':>10}"
    )
    lines.append(header)
    lines.append("-" * 100)

    for r in results:
        compress_ms = r.p50_ms
        llm_saved_ms = r.tokens_saved * model["ms_per_token"]
        net_ms = llm_saved_ms - compress_ms
        cost_saved = r.tokens_saved / 1_000_000 * model["price_per_mtok_input"] * 1000

        net_str = f"+{net_ms:.1f}ms" if net_ms >= 0 else f"{net_ms:.1f}ms"

        lines.append(
            f"{r.scenario_name:<40} "
            f"{_fmt_ms(compress_ms) + 'ms':>10} "
            f"{_fmt_ms(llm_saved_ms) + 'ms':>10} "
            f"{net_str:>12} "
            f"${cost_saved:>8.2f}"
        )

    lines.append("")
    lines.append("")

    # --- Break-even summary ---
    lines.append("BREAK-EVEN ANALYSIS")
    lines.append("-" * 80)
    lines.append("Minimum tokens saved for compression to pay for itself in latency:")
    lines.append("")

    for _model_name, profile in MODEL_PROFILES.items():
        lines.append(f"  {profile['label']:<25} ({profile['ms_per_token']}ms/token):")
        for r in results:
            if r.tokens_saved == 0:
                continue
            # Break-even: compress_ms = tokens_needed * ms_per_token
            # tokens_needed = compress_ms / ms_per_token
            tokens_needed = r.p50_ms / profile["ms_per_token"]
            if tokens_needed <= r.tokens_saved:
                lines.append(
                    f"    {r.scenario_name:<38} "
                    f"need {_fmt_tokens(int(tokens_needed)):>6}, "
                    f"save {_fmt_tokens(r.tokens_saved):>6} -> ALWAYS WINS"
                )
            else:
                lines.append(
                    f"    {r.scenario_name:<38} "
                    f"need {_fmt_tokens(int(tokens_needed)):>6}, "
                    f"save {_fmt_tokens(r.tokens_saved):>6} -> OVERHEAD > SAVINGS"
                )
        lines.append("")

    return "\n".join(lines)


def format_markdown_report(results: list[LatencyResult]) -> str:
    """Format results as a publishable markdown report."""
    lines: list[str] = []

    lines.append("# Headroom Latency Benchmarks")
    lines.append("")
    lines.append(
        "Measured compression overhead across content types and sizes to answer: "
        "**does the token savings outweigh the processing time?**"
    )
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # Environment
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- **Platform**: {platform.platform()}")
    lines.append(f"- **Processor**: {platform.processor() or platform.machine()}")
    lines.append(f"- **Python**: {platform.python_version()}")
    lines.append("- **Headroom**: v0.3.7")
    lines.append("")

    # TL;DR
    if results:
        all_savings = [r for r in results if r.tokens_saved > 0]
        if all_savings:
            avg_ratio = statistics.mean(r.compression_ratio for r in all_savings)
            max_overhead = max(r.p50_ms for r in all_savings)
            model = MODEL_PROFILES[REFERENCE_MODEL]
            all_net = [r.tokens_saved * model["ms_per_token"] - r.p50_ms for r in all_savings]
            wins = sum(1 for n in all_net if n > 0)
            lines.append("## TL;DR")
            lines.append("")
            lines.append(f"- Average compression: **{avg_ratio:.0%}** token reduction")
            lines.append(f"- Maximum compression overhead: **{_fmt_ms(max_overhead)}ms** (p50)")
            lines.append(
                f"- Net latency win: **{wins}/{len(all_savings)}** scenarios "
                f"against {model['label']}"
            )
            lines.append("")

    # Main results table
    lines.append("## Compression Overhead by Scenario")
    lines.append("")
    lines.append(
        "| Scenario | Tokens In | Tokens Out | Saved | Ratio | p50 (ms) | p95 (ms) | Mean (ms) |"
    )
    lines.append(
        "|----------|-----------|------------|-------|-------|----------|----------|-----------|"
    )

    for r in results:
        lines.append(
            f"| {r.scenario_name} | {_fmt_tokens(r.tokens_before)} | "
            f"{_fmt_tokens(r.tokens_after)} | {_fmt_tokens(r.tokens_saved)} | "
            f"{r.compression_ratio:.0%} | {_fmt_ms(r.p50_ms)} | "
            f"{_fmt_ms(r.p95_ms)} | {_fmt_ms(r.mean_ms)} |"
        )

    lines.append("")

    # Per-transform breakdown
    pipeline_results = [r for r in results if len(r.transform_timings) > 1]
    if pipeline_results:
        lines.append("## Per-Transform Latency Breakdown")
        lines.append("")
        lines.append("| Scenario | Transform | p50 (ms) | % of Total |")
        lines.append("|----------|-----------|----------|------------|")

        for r in pipeline_results:
            total_p50 = r.p50_ms
            for tname, tt in r.transform_timings.items():
                pct = (tt.p50_ms / total_p50 * 100) if total_p50 > 0 else 0
                lines.append(f"| {r.scenario_name} | {tname} | {_fmt_ms(tt.p50_ms)} | {pct:.0f}% |")

        lines.append("")

    # Cost-benefit analysis
    lines.append("## Cost-Benefit Analysis")
    lines.append("")
    lines.append("Net latency benefit = LLM time saved from fewer tokens - compression overhead.")
    lines.append("")
    lines.append("| Scenario | Compress (ms) | LLM Saved (ms)* | Net Benefit | $/1K Requests** |")
    lines.append("|----------|---------------|-----------------|-------------|-----------------|")

    model = MODEL_PROFILES[REFERENCE_MODEL]
    for r in results:
        if r.tokens_saved <= 0:
            continue
        compress_ms = r.p50_ms
        llm_saved_ms = r.tokens_saved * model["ms_per_token"]
        net_ms = llm_saved_ms - compress_ms
        cost_saved = r.tokens_saved / 1_000_000 * model["price_per_mtok_input"] * 1000

        net_str = f"+{net_ms:.1f}ms" if net_ms >= 0 else f"{net_ms:.1f}ms"
        lines.append(
            f"| {r.scenario_name} | {_fmt_ms(compress_ms)} | "
            f"{_fmt_ms(llm_saved_ms)} | {net_str} | ${cost_saved:.2f} |"
        )

    lines.append("")
    lines.append(
        f"\\* LLM time saved based on {model['label']} prefill rate "
        f"({model['ms_per_token']}ms/token)"
    )
    lines.append(f"\\*\\* Cost savings at ${model['price_per_mtok_input']}/MTok input pricing")
    lines.append("")

    # Multi-model comparison
    lines.append("## Break-Even Across Models")
    lines.append("")
    lines.append("Compression overhead (p50) vs. LLM time saved for different model speed tiers:")
    lines.append("")

    header = "| Scenario | Compress (ms) |"
    separator = "|----------|---------------|"
    for profile in MODEL_PROFILES.values():
        header += f" {profile['label']} |"
        separator += "------------|"
    lines.append(header)
    lines.append(separator)

    for r in results:
        if r.tokens_saved <= 0:
            continue
        row = f"| {r.scenario_name} | {_fmt_ms(r.p50_ms)} |"
        for profile in MODEL_PROFILES.values():
            llm_saved = r.tokens_saved * profile["ms_per_token"]
            net = llm_saved - r.p50_ms
            if net > 0:
                row += f" +{_fmt_ms(net)}ms |"
            else:
                row += f" {_fmt_ms(net)}ms |"
        lines.append(row)

    lines.append("")

    # Data-driven key takeaways
    lines.append("## Key Takeaways")
    lines.append("")

    compressing = [r for r in results if r.tokens_saved > 0]
    model = MODEL_PROFILES[REFERENCE_MODEL]
    pt = 0  # point counter
    if compressing:
        # Where compression wins on latency
        latency_wins = [r for r in compressing if r.tokens_saved * model["ms_per_token"] > r.p50_ms]

        pt += 1
        if latency_wins:
            win_types = list(dict.fromkeys(r.content_type for r in latency_wins))
            win_names = ", ".join(win_types[:4])
            lines.append(
                f"{pt}. **Compression pays for itself in latency** for "
                f"{len(latency_wins)}/{len(compressing)} compressing scenarios "
                f"({win_names}). For these, the LLM prefill time saved exceeds "
                f"compression overhead."
            )
        else:
            lines.append(
                f"{pt}. **Compression adds latency in all scenarios** at "
                f"{model['label']} prefill rates. The value is in cost savings, "
                f"not speed."
            )

        # ContentRouter dominance
        cr_pcts = []
        for r in compressing:
            if "content_router" in r.transform_timings:
                cr_pct = (
                    r.transform_timings["content_router"].p50_ms / r.p50_ms * 100
                    if r.p50_ms > 0
                    else 0
                )
                cr_pcts.append(cr_pct)
        if cr_pcts:
            avg_cr = statistics.mean(cr_pcts)
            pt += 1
            lines.append(
                f"{pt}. **ContentRouter is {avg_cr:.0f}% of pipeline cost** on average "
                f"— it does the actual compression work. CacheAligner and context "
                f"management are <2% of total time."
            )

        # Cost savings are always significant
        best_cost = max(compressing, key=lambda r: r.tokens_saved)
        cost_per_1k = best_cost.tokens_saved / 1_000_000 * model["price_per_mtok_input"] * 1000
        pt += 1
        lines.append(
            f"{pt}. **Cost savings are substantial regardless of latency.** "
            f"The highest-compression scenario ({best_cost.scenario_name}) "
            f"saves ${cost_per_1k:.0f}/1K requests at {model['label']} pricing."
        )

        # Where it doesn't help
        no_compress = [r for r in results if r.tokens_saved <= 0]
        if no_compress:
            types = sorted({r.content_type for r in no_compress})
            pt += 1
            lines.append(
                f"{pt}. **No compression for**: {', '.join(types)}. "
                f"These content types pass through the pipeline with only "
                f"tokenization overhead ({_fmt_ms(min(r.p50_ms for r in no_compress))}"
                f"-{_fmt_ms(max(r.p50_ms for r in no_compress))}ms)."
            )

        # Opus always wins
        opus = MODEL_PROFILES.get("claude-opus-4")
        if opus:
            opus_wins = [r for r in compressing if r.tokens_saved * opus["ms_per_token"] > r.p50_ms]
            if len(opus_wins) > len(latency_wins):
                pt += 1
                lines.append(
                    f"{pt}. **Slower/pricier models benefit most.** Claude Opus shows "
                    f"a net latency win in {len(opus_wins)}/{len(compressing)} "
                    f"scenarios vs {len(latency_wins)} for {model['label']}, "
                    f"with {opus['ms_per_token']}ms/token prefill."
                )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*Benchmarks run with `python benchmarks/bench_latency.py`. "
        "Results vary based on hardware, Python version, and content characteristics.*"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Headroom latency benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save markdown report to this path",
    )
    parser.add_argument(
        "--json",
        "-j",
        help="Save JSON results to this path",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=20,
        help="Number of measured iterations per scenario (default: 20)",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        choices=["json", "code", "text", "logs", "agentic", "rag"],
        action="append",
        help="Run specific content type(s) only. Can be repeated.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show progress during benchmark run",
    )

    args = parser.parse_args()

    print("Headroom Latency Benchmark")
    print("=" * 40)
    print()

    # Generate scenarios
    content_types = args.scenario  # None means all
    print("Generating test scenarios...", flush=True)
    scenarios = generate_scenarios(content_types)
    print(f"  {len(scenarios)} scenarios ready")
    print()

    # Run benchmarks
    print(f"Running benchmarks ({args.iterations} iterations, {args.warmup} warmup)...")
    print()
    results = run_all(
        scenarios,
        iterations=args.iterations,
        warmup=args.warmup,
        verbose=True,
    )

    # Terminal report (always printed)
    report = format_terminal_report(results)
    print(report)

    # Save markdown report
    if args.output:
        md = format_markdown_report(results)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(md)
        print(f"Markdown report saved to: {args.output}")

    # Save JSON results
    if args.json:
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "results": [r.to_dict() for r in results],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(data, indent=2))
        print(f"JSON results saved to: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
