#!/usr/bin/env python3
"""
Real-world benchmark for DynamicContentDetector.

Tests the detector against realistic system prompts from AI coding agents,
chatbots, and enterprise applications.
"""

import statistics
import time
from dataclasses import dataclass
from typing import Any

from headroom.cache.dynamic_detector import (
    DetectorConfig,
    DynamicContentDetector,
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    content_length: int
    spans_found: int
    categories: list[str]
    static_length: int
    dynamic_length: int
    latency_ms: float
    tiers_used: list[str]
    warnings: list[str]


# Real-world system prompts
REAL_WORLD_PROMPTS = {
    "claude_code_style": """You are Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest.

Today is Tuesday, January 7, 2026.
Current time: 10:30:45 AM PST.

You are operating in a software development environment with access to:
- File system operations
- Terminal commands
- Web search

Session ID: sess_abc123def456ghi789jkl012
Request ID: req_xyz789abc123def456ghi789
User: tchopra
Workspace: /Users/tchopra/claude-projects/headroom

Be concise, accurate, and helpful. Follow the user's instructions carefully.""",
    "enterprise_assistant": """You are an enterprise AI assistant for Acme Corporation.

Current Date: 2026-01-07T10:30:00Z
Last Updated: 2026-01-07T09:00:00Z

User Profile:
- Name: John Smith
- Employee ID: EMP-2024-00542
- Department: Engineering
- Manager: Sarah Johnson
- Location: San Francisco, CA
- Hire Date: March 15, 2023

System Status:
- API Version: v2.3.1-beta
- Server Load: 45%
- Active Users: 1,247
- Queue Length: 23

Budget Information:
- Monthly Allowance: $5,000.00
- Used This Month: $2,341.67
- Remaining: $2,658.33

Help the user with their work tasks while following company policies.""",
    "coding_agent": """You are an autonomous coding agent with access to tools.

Environment:
- OS: macOS Darwin 25.1.0
- Working Directory: /Users/developer/projects/myapp
- Git Branch: feature/JIRA-1234-add-auth
- Last Commit: a1b2c3d4e5f6 (2 hours ago)
- Node Version: v20.10.0
- Python Version: 3.11.7

Current Task Context:
- Task ID: 550e8400-e29b-41d4-a716-446655440000
- Created: 2026-01-07T08:15:30Z
- Priority: High
- Estimated Time: 2 hours

API Keys Available:
- OPENAI_API_KEY: sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
- ANTHROPIC_API_KEY: sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
- DATABASE_URL: postgresql://user:pass@localhost:5432/mydb

Execute tasks step by step, verify each action, and report progress.""",
    "customer_support": """You are a customer support agent for TechStore Inc.

Current Time: January 7, 2026, 3:45 PM EST
Support Ticket: #TKT-2026-0107-4521

Customer Information:
- Name: Alice Chen
- Email: alice.chen@email.com
- Phone: (555) 123-4567
- Customer Since: August 2021
- Loyalty Tier: Gold
- Total Purchases: $12,456.78

Recent Orders:
- Order #ORD-2026-0105-7823 - iPhone 15 Pro - $1,199.00 - Delivered
- Order #ORD-2025-1220-3456 - AirPods Pro - $249.00 - Delivered
- Order #ORD-2025-1115-9012 - MacBook Air - $1,299.00 - Returned

Active Issues:
- Case #CS-2026-0107-001 - Battery drain issue - Open since today

Provide helpful, empathetic support while following company guidelines.""",
    "data_analysis": """You are a data analysis assistant.

Report Generated: 2026-01-07 10:30:00 UTC
Report ID: RPT-550e8400-e29b-41d4-a716-446655440000
Data Range: 2025-12-01 to 2025-12-31

Summary Statistics:
- Total Revenue: $1,234,567.89
- Total Orders: 45,678
- Average Order Value: $27.03
- Top Product: Widget Pro ($234,567.00)
- Top Region: California (23.4%)

Key Metrics:
- DAU: 125,000
- MAU: 890,000
- Churn Rate: 2.3%
- NPS Score: 67

Anomalies Detected:
- Spike on Dec 15: 3.2x normal traffic
- Drop on Dec 25: 0.4x normal (expected - holiday)

Help analyze the data and provide insights.""",
    "minimal_static": """You are a helpful AI assistant.

Your role is to:
1. Answer questions accurately
2. Be concise and clear
3. Follow instructions carefully
4. Admit when you don't know something

Always be helpful, harmless, and honest.""",
    "heavy_dynamic": """Session started at 2026-01-07T10:30:45.123Z
Request ID: req_abc123def456ghi789jkl012mno345pqr678
Trace ID: 550e8400-e29b-41d4-a716-446655440000
Parent Span: span_xyz789abc123
User Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)
IP Address: 192.168.1.100
Geo: San Francisco, CA, USA (37.7749, -122.4194)

Auth Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Token Expires: 2026-01-07T11:30:45Z
Refresh Token: rt_abc123def456

Last Login: 2026-01-06T18:45:30Z
Login Count: 1,247
Account Balance: $5,432.10
Credit Limit: $10,000.00

Real-time Stock Prices (as of 10:30 AM):
- AAPL: $185.42 (+1.2%)
- GOOGL: $142.89 (-0.5%)
- MSFT: $378.23 (+0.8%)
- AMZN: $156.78 (+2.1%)

Process this request.""",
}


def run_benchmark(
    prompts: dict[str, str],
    tiers: list[str],
    iterations: int = 10,
) -> dict[str, Any]:
    """Run benchmark on prompts with specified tiers."""

    config = DetectorConfig(tiers=tiers)  # type: ignore
    detector = DynamicContentDetector(config)

    results: dict[str, list[BenchmarkResult]] = {}

    for name, content in prompts.items():
        results[name] = []

        for _ in range(iterations):
            start = time.perf_counter()
            result = detector.detect(content)
            elapsed = (time.perf_counter() - start) * 1000

            categories = list({s.category.value for s in result.spans})

            results[name].append(
                BenchmarkResult(
                    name=name,
                    content_length=len(content),
                    spans_found=len(result.spans),
                    categories=categories,
                    static_length=len(result.static_content),
                    dynamic_length=len(result.dynamic_content),
                    latency_ms=elapsed,
                    tiers_used=result.tiers_used,
                    warnings=result.warnings,
                )
            )

    return results


def print_results(
    results: dict[str, list[BenchmarkResult]],
    tier_name: str,
):
    """Print benchmark results."""

    print(f"\n{'=' * 80}")
    print(f"BENCHMARK RESULTS: {tier_name}")
    print(f"{'=' * 80}")

    for name, runs in results.items():
        latencies = [r.latency_ms for r in runs]
        avg_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0

        # Use first run for span info (consistent across runs)
        first = runs[0]

        compression = (
            (1 - first.static_length / first.content_length) * 100
            if first.content_length > 0
            else 0
        )

        print(f"\n📄 {name}")
        print(f"   Content: {first.content_length:,} chars")
        print(f"   Spans found: {first.spans_found}")
        print(f"   Categories: {', '.join(first.categories) if first.categories else 'none'}")
        print(f"   Static: {first.static_length:,} chars | Dynamic: {first.dynamic_length:,} chars")
        print(f"   Compression: {compression:.1f}% removed")
        print(f"   Latency: {avg_latency:.2f}ms ± {std_latency:.2f}ms")
        print(f"   Tiers used: {', '.join(first.tiers_used)}")
        if first.warnings:
            print(f"   ⚠️  Warnings: {len(first.warnings)}")


def print_comparison(all_results: dict[str, dict[str, list[BenchmarkResult]]]):
    """Print comparison across tiers."""

    print(f"\n{'=' * 80}")
    print("TIER COMPARISON")
    print(f"{'=' * 80}")

    prompts = list(REAL_WORLD_PROMPTS.keys())
    tiers = list(all_results.keys())

    # Header
    header = f"{'Prompt':<25}"
    for tier in tiers:
        header += f" | {tier:>12} spans | {'latency':>8}"
    print(header)
    print("-" * len(header))

    for prompt in prompts:
        row = f"{prompt:<25}"
        for tier in tiers:
            if prompt in all_results[tier]:
                runs = all_results[tier][prompt]
                spans = runs[0].spans_found
                latency = statistics.mean([r.latency_ms for r in runs])
                row += f" | {spans:>12} | {latency:>7.2f}ms"
            else:
                row += f" | {'N/A':>12} | {'N/A':>8}"
        print(row)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    for tier in tiers:
        all_latencies = []
        total_spans = 0
        for runs in all_results[tier].values():
            all_latencies.extend([r.latency_ms for r in runs])
            total_spans += runs[0].spans_found

        avg = statistics.mean(all_latencies)
        p50 = statistics.median(all_latencies)
        p99 = (
            sorted(all_latencies)[int(len(all_latencies) * 0.99)] if len(all_latencies) > 1 else avg
        )

        print(f"\n{tier}:")
        print(f"   Total spans detected: {total_spans}")
        print(f"   Avg latency: {avg:.2f}ms")
        print(f"   P50 latency: {p50:.2f}ms")
        print(f"   P99 latency: {p99:.2f}ms")


def show_detection_details(prompt_name: str, content: str):
    """Show detailed detection for a specific prompt."""

    print(f"\n{'=' * 80}")
    print(f"DETECTION DETAILS: {prompt_name}")
    print(f"{'=' * 80}")

    config = DetectorConfig(tiers=["regex"])
    detector = DynamicContentDetector(config)
    result = detector.detect(content)

    print(f"\nOriginal content ({len(content)} chars):")
    print("-" * 40)
    print(content[:500] + "..." if len(content) > 500 else content)

    print(f"\n\nDetected spans ({len(result.spans)}):")
    print("-" * 40)
    for span in result.spans:
        print(
            f"  [{span.category.value:12}] '{span.text[:50]}{'...' if len(span.text) > 50 else ''}'"
        )

    print(f"\n\nStatic content ({len(result.static_content)} chars):")
    print("-" * 40)
    print(
        result.static_content[:500] + "..."
        if len(result.static_content) > 500
        else result.static_content
    )

    print(f"\n\nDynamic content ({len(result.dynamic_content)} chars):")
    print("-" * 40)
    print(result.dynamic_content if result.dynamic_content else "(none)")


def main():
    """Run the benchmark."""

    print("🚀 Dynamic Content Detector - Real World Benchmark")
    print("=" * 80)

    iterations = 20

    # Test each tier configuration
    tier_configs = {
        "regex_only": ["regex"],
        # "regex+ner": ["regex", "ner"],  # Uncomment if spacy installed
        # "all_tiers": ["regex", "ner", "semantic"],  # Uncomment if all deps installed
    }

    all_results: dict[str, dict[str, list[BenchmarkResult]]] = {}

    for tier_name, tiers in tier_configs.items():
        print(f"\n⏱️  Running {tier_name} ({iterations} iterations per prompt)...")
        results = run_benchmark(REAL_WORLD_PROMPTS, tiers, iterations)
        all_results[tier_name] = results
        print_results(results, tier_name)

    # Print comparison if multiple tiers tested
    if len(all_results) > 1:
        print_comparison(all_results)

    # Show detailed detection for a few prompts
    print("\n" + "=" * 80)
    print("DETAILED DETECTION EXAMPLES")
    print("=" * 80)

    for name in ["claude_code_style", "enterprise_assistant", "heavy_dynamic"]:
        show_detection_details(name, REAL_WORLD_PROMPTS[name])


if __name__ == "__main__":
    main()
