#!/usr/bin/env python3
"""
Prefix Cache Strategy Benchmark — Real API Calls

Sends a 25-turn conversation through 4 different caching strategies and measures
actual cache_read_input_tokens vs cache_creation_input_tokens from the Anthropic API.

Strategies:
  1. Baseline — no Headroom, no markers
  2. Headroom compression — full pipeline, CompressionCache keeps bytes stable
  3. Headroom + prefix freeze — pipeline skips frozen (already-cached) messages
  4. Headroom + explicit markers — pipeline + 4 cache_control breakpoints

Usage:
    # Load API key from .env and run
    source .env && python benchmarks/prefix_cache_benchmark.py

    # Quick test with fewer turns
    source .env && python benchmarks/prefix_cache_benchmark.py --turns 5

    # With specific model
    source .env && python benchmarks/prefix_cache_benchmark.py --model claude-sonnet-4-6

Estimated cost: ~$0.50-1.00 total across all strategies.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Pricing (per token)
# ---------------------------------------------------------------------------
PRICING = {
    "claude-sonnet-4-6": {
        "input": 3.00 / 1_000_000,
        "output": 15.00 / 1_000_000,
        "cache_read": 0.30 / 1_000_000,
        "cache_write": 3.75 / 1_000_000,
    },
    "claude-haiku-4-5-20251001": {
        "input": 0.80 / 1_000_000,
        "output": 4.00 / 1_000_000,
        "cache_read": 0.08 / 1_000_000,
        "cache_write": 1.00 / 1_000_000,
    },
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TurnMetrics:
    turn: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class StrategyResult:
    name: str
    turns: list[TurnMetrics] = field(default_factory=list)

    @property
    def total_input(self) -> int:
        return sum(
            t.input_tokens + t.cache_read_tokens + t.cache_creation_tokens for t in self.turns
        )

    @property
    def total_cache_read(self) -> int:
        return sum(t.cache_read_tokens for t in self.turns)

    @property
    def total_cache_write(self) -> int:
        return sum(t.cache_creation_tokens for t in self.turns)

    @property
    def total_output(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_cost(self) -> float:
        return sum(t.cost_usd for t in self.turns)

    @property
    def cache_hit_rate(self) -> float:
        total = (
            self.total_cache_read + self.total_cache_write + sum(t.input_tokens for t in self.turns)
        )
        return (self.total_cache_read / total * 100) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Conversation builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert software engineering assistant. You help users debug code,
analyze logs, query databases, and search codebases. You have access to several tools.

When analyzing data, be thorough but concise. Focus on anomalies, errors, and actionable insights.
Always explain your reasoning step by step.

Important guidelines:
- When you see error patterns, highlight them immediately
- For database queries, suggest optimizations if the result set is large
- For code analysis, focus on potential bugs and security issues
- Always provide actionable next steps

You are working in a large Python monorepo with FastAPI services, PostgreSQL databases,
and Redis caching. The codebase uses pytest for testing and has CI/CD via GitHub Actions."""

TOOLS = [
    {
        "name": "search_codebase",
        "description": "Search the codebase for patterns, function definitions, or references.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search pattern or keyword"},
                "file_pattern": {"type": "string", "description": "Glob pattern for files"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"},
                "offset": {"type": "integer", "description": "Line offset to start from"},
                "limit": {"type": "integer", "description": "Number of lines to read"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "query_database",
        "description": "Execute a read-only SQL query against the application database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL SELECT query"},
                "database": {"type": "string", "description": "Database name"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_logs",
        "description": "Search application logs for patterns within a time range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Log pattern to search"},
                "service": {"type": "string", "description": "Service name"},
                "hours": {"type": "integer", "description": "Hours to look back"},
            },
            "required": ["pattern"],
        },
    },
]

USER_QUERIES = [
    "Can you search for all usages of the `authenticate_user` function?",
    "Read the file src/auth/middleware.py so I can understand the auth flow.",
    "Query the database for failed login attempts in the last hour: SELECT user_id, attempt_time, error_code FROM auth_logs WHERE status='failed' AND attempt_time > NOW() - INTERVAL '1 hour' ORDER BY attempt_time DESC LIMIT 50",
    "Search the logs for 'ConnectionRefused' errors in the auth-service from the past 2 hours.",
    "Read src/auth/token_validator.py — I think the bug might be there.",
    "Search for all files that import from `auth.middleware`.",
    "Query for users who had more than 5 failed attempts: SELECT user_id, COUNT(*) as fails FROM auth_logs WHERE status='failed' AND attempt_time > NOW() - INTERVAL '24 hours' GROUP BY user_id HAVING COUNT(*) > 5",
    "Search logs for 'JWT expired' in auth-service.",
    "Read the test file tests/test_auth.py to see what's covered.",
    "Search for `rate_limit` in the codebase.",
    "Read src/config/settings.py to check the rate limit configuration.",
    "Query the metrics table: SELECT endpoint, avg_latency_ms, p99_latency_ms, error_rate FROM api_metrics WHERE timestamp > NOW() - INTERVAL '1 hour' ORDER BY error_rate DESC",
    "Search logs for any 5xx errors across all services.",
    "Read src/api/routes.py to check the endpoint definitions.",
    "Search for usages of the Redis cache client.",
    "Read src/cache/redis_client.py for the connection pooling setup.",
    "Query cache hit rates: SELECT cache_key_prefix, hit_count, miss_count, hit_count::float/(hit_count+miss_count) as hit_rate FROM cache_stats WHERE period='hourly' ORDER BY miss_count DESC LIMIT 20",
    "Search logs for 'cache eviction' warnings.",
    "Read the Dockerfile to check the base image version.",
    "Search for any TODO or FIXME comments in the auth module.",
    "Read .github/workflows/ci.yml for the CI pipeline config.",
    "Query deployment history: SELECT version, deployed_at, deployed_by, status FROM deployments WHERE service='auth-service' ORDER BY deployed_at DESC LIMIT 10",
    "Search for error handling patterns — look for bare `except:` blocks.",
    "Read src/auth/oauth.py for the OAuth integration.",
    "Search logs for memory usage spikes in the last 4 hours.",
]

# Fake tool responses (JSON data that would come from tools)
TOOL_RESPONSES = {
    "search_codebase": lambda q: json.dumps(
        [
            {
                "file": f"src/auth/{f}.py",
                "line": 10 + i * 5,
                "match": f"def authenticate_user(request): # {q}",
            }
            for i, f in enumerate(["middleware", "token_validator", "oauth", "session", "utils"])
        ]
        + [
            {
                "file": f"tests/test_{f}.py",
                "line": 20 + i * 3,
                "match": f"from auth.middleware import {q.split()[0] if q.split() else 'auth'}",
            }
            for i, f in enumerate(["auth", "api", "cache"])
        ]
    ),
    "read_file": lambda q: (
        "# File contents (simulated)\nimport logging\nfrom typing import Optional\n\nlogger = logging.getLogger(__name__)\n\n"
        + "\n".join(
            [
                f"def function_{i}(arg: str) -> Optional[dict]:\n    \"\"\"Process {q}.\"\"\"\n    result = {{}}\n    for key in ['id', 'name', 'status']:\n        result[key] = f'value_{{key}}_{{arg}}'\n    logger.info(f'Processed {{arg}}')\n    return result\n"
                for i in range(8)
            ]
        )
    ),
    "query_database": lambda q: json.dumps(
        [
            {
                "user_id": f"user_{i:04d}",
                "attempt_time": f"2025-01-15T10:{i:02d}:00Z",
                "error_code": [
                    "INVALID_PASSWORD",
                    "EXPIRED_TOKEN",
                    "RATE_LIMITED",
                    "ACCOUNT_LOCKED",
                ][i % 4],
                "status": "failed",
            }
            for i in range(15)
        ]
    ),
    "search_logs": lambda q: "\n".join(
        [
            f"2025-01-15T10:{i:02d}:{j:02d}Z [ERROR] auth-service: {q} - connection to db-primary:5432 refused (attempt {j + 1}/3)"
            for i in range(5)
            for j in range(3)
        ]
    ),
}


def build_turn_messages(
    turn_idx: int,
    history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Build messages for a specific turn, return (messages, user_query)."""
    query = USER_QUERIES[turn_idx % len(USER_QUERIES)]
    messages = list(history) + [{"role": "user", "content": query}]
    return messages, query


# ---------------------------------------------------------------------------
# API call helper
# ---------------------------------------------------------------------------


def call_anthropic(
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    max_tokens: int = 100,
) -> dict[str, Any]:
    """Make a real Anthropic API call, return the full response JSON."""
    # Separate system from messages (Anthropic format)
    system_content = None
    api_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            api_messages.append(msg)

    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": api_messages,
    }
    if system_content:
        body["system"] = system_content
    if tools:
        body["tools"] = tools

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(
            "https://api.anthropic.com/v1/messages",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()


def extract_metrics(resp: dict, turn: int, pricing: dict) -> TurnMetrics:
    """Extract cache metrics from Anthropic response."""
    usage = resp.get("usage", {})
    cr = usage.get("cache_read_input_tokens", 0)
    cw = usage.get("cache_creation_input_tokens", 0)
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)

    cost = (
        cr * pricing["cache_read"]
        + cw * pricing["cache_write"]
        + inp * pricing["input"]
        + out * pricing["output"]
    )

    return TurnMetrics(
        turn=turn,
        cache_read_tokens=cr,
        cache_creation_tokens=cw,
        input_tokens=inp,
        output_tokens=out,
        cost_usd=cost,
    )


def extract_assistant_content(resp: dict) -> dict[str, Any]:
    """Convert Anthropic response to a message dict for conversation history."""
    content = resp.get("content", [])
    # Check for tool use
    has_tool_use = any(b.get("type") == "tool_use" for b in content if isinstance(b, dict))

    if has_tool_use:
        return {"role": "assistant", "content": content}
    else:
        # Extract text
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        return {"role": "assistant", "content": text}


def make_tool_result(assistant_msg: dict) -> list[dict[str, Any]]:
    """Generate fake tool results for any tool_use blocks in the assistant message."""
    results = []
    content = assistant_msg.get("content", [])
    if not isinstance(content, list):
        return results

    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_name = block.get("name", "search_codebase")
            tool_id = block.get("id", "")
            query = json.dumps(block.get("input", {}))

            gen = TOOL_RESPONSES.get(tool_name, TOOL_RESPONSES["search_codebase"])
            fake_output = gen(query)

            results.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": fake_output,
                        }
                    ],
                }
            )
    return results


# ---------------------------------------------------------------------------
# Strategy: inject explicit cache_control markers
# ---------------------------------------------------------------------------


def inject_cache_markers(
    system_content: str | None,
    api_messages: list[dict[str, Any]],
) -> tuple[str | list | None, list[dict[str, Any]]]:
    """Inject up to 4 cache_control breakpoints at strategic positions.

    Marker 1: End of system prompt
    Marker 2: ~1/3 through messages
    Marker 3: ~2/3 through messages
    Marker 4: Last message
    """
    # Marker 1: system prompt
    if system_content and isinstance(system_content, str):
        system_content = [
            {"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}
        ]

    if not api_messages:
        return system_content, api_messages

    msgs = copy.deepcopy(api_messages)
    n = len(msgs)

    # Pick positions for markers 2-4 (indices into msgs)
    positions = set()
    if n >= 3:
        positions.add(n // 3)  # Marker 2: ~1/3
        positions.add(2 * n // 3)  # Marker 3: ~2/3
    positions.add(n - 1)  # Marker 4: last message

    markers_placed = 1  # Already placed marker 1 on system
    for pos in sorted(positions):
        if markers_placed >= 4:
            break
        msg = msgs[pos]
        content = msg.get("content")

        if isinstance(content, str):
            msg["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
            markers_placed += 1
        elif isinstance(content, list) and content:
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}
                markers_placed += 1

    return system_content, msgs


# ---------------------------------------------------------------------------
# Run a full conversation for one strategy
# ---------------------------------------------------------------------------


def inject_cc_style_markers(
    system_content: str | None,
    api_messages: list[dict[str, Any]],
) -> tuple[str | list | None, list[dict[str, Any]]]:
    """Simulate Claude Code's caching strategy.

    Claude Code places cache_control on:
    - The system prompt (stable, always cached)
    - The last ~2 user/assistant messages (growing prefix)
    This uses 2-3 of the 4 available breakpoints.
    """
    # Marker on system prompt
    if system_content and isinstance(system_content, str):
        system_content = [
            {"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}
        ]

    if not api_messages:
        return system_content, api_messages

    msgs = copy.deepcopy(api_messages)
    n = len(msgs)

    # Marker on last message (the new user query)
    markers_placed = 1  # system already has one
    if n >= 1 and markers_placed < 4:
        msg = msgs[-1]
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
            markers_placed += 1
        elif isinstance(content, list) and content:
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}
                markers_placed += 1

    # Marker on second-to-last user message (if exists)
    if n >= 3 and markers_placed < 4:
        # Find second-to-last user message
        for i in range(n - 2, -1, -1):
            if msgs[i].get("role") == "user":
                content = msgs[i].get("content")
                if isinstance(content, str):
                    msgs[i]["content"] = [
                        {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                    ]
                    markers_placed += 1
                elif isinstance(content, list) and content:
                    last_block = content[-1]
                    if isinstance(last_block, dict):
                        last_block["cache_control"] = {"type": "ephemeral"}
                        markers_placed += 1
                break

    return system_content, msgs


# ---------------------------------------------------------------------------
# Caching mode enum
# ---------------------------------------------------------------------------
CACHE_MODE_NONE = "none"
CACHE_MODE_CC_STYLE = "cc_style"  # Claude Code's strategy
CACHE_MODE_EXPLICIT = "explicit_4"  # Headroom's 4 strategic breakpoints


def run_strategy(
    name: str,
    api_key: str,
    model: str,
    num_turns: int,
    pricing: dict,
    use_tools: bool = True,
    cache_mode: str = CACHE_MODE_NONE,
    delay: float = 1.0,
) -> StrategyResult:
    """Run a full multi-turn conversation and collect cache metrics."""
    result = StrategyResult(name=name)

    history: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    tools = TOOLS if use_tools else None

    for turn in range(num_turns):
        # Build messages for this turn
        query = USER_QUERIES[turn % len(USER_QUERIES)]
        history.append({"role": "user", "content": query})

        # Prepare API call
        system_content: str | list | None = None
        api_messages: list[dict[str, Any]] = []
        for msg in history:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                api_messages.append(msg)

        # Apply caching strategy
        if cache_mode == CACHE_MODE_CC_STYLE:
            system_content, api_messages = inject_cc_style_markers(system_content, api_messages)
        elif cache_mode == CACHE_MODE_EXPLICIT:
            system_content, api_messages = inject_cache_markers(system_content, api_messages)

        # Build request body
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": 100,
            "messages": api_messages,
        }
        if system_content:
            body["system"] = system_content if isinstance(system_content, list) else system_content
        if tools:
            body["tools"] = tools

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Make the API call
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=body,
                    headers=headers,
                )
                resp.raise_for_status()
                resp_json = resp.json()
        except Exception as e:
            print(f"  [!] Turn {turn + 1} failed: {e}")
            break

        # Extract metrics
        metrics = extract_metrics(resp_json, turn + 1, pricing)
        result.turns.append(metrics)

        total_cached = (
            metrics.cache_read_tokens + metrics.cache_creation_tokens + metrics.input_tokens
        )
        hit_pct = (metrics.cache_read_tokens / total_cached * 100) if total_cached > 0 else 0

        print(
            f"  Turn {turn + 1:2d}: "
            f"read={metrics.cache_read_tokens:6d}  "
            f"write={metrics.cache_creation_tokens:6d}  "
            f"input={metrics.input_tokens:5d}  "
            f"hit={hit_pct:5.1f}%  "
            f"${metrics.cost_usd:.4f}"
        )

        # Add assistant response to history
        assistant_msg = extract_assistant_content(resp_json)
        history.append(assistant_msg)

        # If assistant used tools, add fake tool results
        tool_results = make_tool_result(assistant_msg)
        history.extend(tool_results)

        # Delay to let cache settle (Anthropic needs the first response to complete
        # before subsequent requests can hit the cache)
        if delay > 0:
            time.sleep(delay)

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[StrategyResult], num_turns: int) -> None:
    """Print comparison report."""
    print()
    print("=" * 72)
    print(f"  Prefix Cache Strategy Benchmark ({num_turns} turns)")
    print("=" * 72)

    baseline_cost = results[0].total_cost if results else 0

    for r in results:
        total = r.total_cache_read + r.total_cache_write + sum(t.input_tokens for t in r.turns)
        hit_rate = (r.total_cache_read / total * 100) if total > 0 else 0

        print(f"\n  Strategy: {r.name}")
        print(f"    Total prompt tokens:  {total:>10,}")
        print(f"    Cache reads (hit):    {r.total_cache_read:>10,}  ({hit_rate:.1f}%)")
        print(f"    Cache writes (miss):  {r.total_cache_write:>10,}")
        print(f"    Output tokens:        {r.total_output:>10,}")
        print(f"    Total cost:           ${r.total_cost:>9.4f}")
        if baseline_cost > 0 and r is not results[0]:
            savings = (1 - r.total_cost / baseline_cost) * 100
            print(f"    Savings vs baseline:  {savings:>9.1f}%")

    # Per-turn hit rate table
    print("\n  Per-turn cache hit rate:")
    header = "  Turn |"
    for r in results:
        short_name = r.name[:12].ljust(12)
        header += f" {short_name} |"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for turn_idx in range(num_turns):
        row = f"  {turn_idx + 1:4d} |"
        for r in results:
            if turn_idx < len(r.turns):
                t = r.turns[turn_idx]
                total = t.cache_read_tokens + t.cache_creation_tokens + t.input_tokens
                hit = (t.cache_read_tokens / total * 100) if total > 0 else 0
                row += f" {hit:>10.1f}% |"
            else:
                row += f" {'N/A':>10s}  |"
        print(row)

    print()
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Prefix cache strategy benchmark")
    parser.add_argument(
        "--turns", type=int, default=15, help="Number of conversation turns (default: 15)"
    )
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6", help="Model to use")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between turns (seconds)")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["all"],
        choices=["baseline", "cc", "markers", "all"],
        help="Which strategies to run (default: all)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        print("  source .env && python benchmarks/prefix_cache_benchmark.py")
        sys.exit(1)

    model = args.model
    pricing = PRICING.get(model, PRICING["claude-sonnet-4-6"])

    strategies_to_run = set(args.strategies)
    if "all" in strategies_to_run:
        strategies_to_run = {"baseline", "cc", "markers"}

    num_strategies = len(strategies_to_run)
    print(f"Prefix Cache Benchmark: {args.turns} turns, model={model}")
    print(f"Strategies: {', '.join(sorted(strategies_to_run))}")
    print(f"Estimated cost: ~${args.turns * num_strategies * 0.01:.2f}")
    print()

    results: list[StrategyResult] = []
    step = 0

    # Strategy 1: Baseline (no markers, no caching at all)
    if "baseline" in strategies_to_run:
        step += 1
        print(f"[{step}/{num_strategies}] Baseline (no markers, no caching)...")
        r = run_strategy(
            "No Cache",
            api_key,
            model,
            args.turns,
            pricing,
            cache_mode=CACHE_MODE_NONE,
            delay=args.delay,
        )
        results.append(r)
        print()

    # Strategy 2: Claude Code-style (system + last 2 messages)
    if "cc" in strategies_to_run:
        step += 1
        print(f"[{step}/{num_strategies}] Claude Code-style (system + last 2 msgs)...")
        r = run_strategy(
            "CC-Style",
            api_key,
            model,
            args.turns,
            pricing,
            cache_mode=CACHE_MODE_CC_STYLE,
            delay=args.delay,
        )
        results.append(r)
        print()

    # Strategy 3: Headroom explicit markers (4 strategic breakpoints)
    if "markers" in strategies_to_run:
        step += 1
        print(f"[{step}/{num_strategies}] Headroom explicit (4 strategic breakpoints)...")
        r = run_strategy(
            "Headroom 4x",
            api_key,
            model,
            args.turns,
            pricing,
            cache_mode=CACHE_MODE_EXPLICIT,
            delay=args.delay,
        )
        results.append(r)
        print()

    # Report
    if results:
        print_report(results, args.turns)


if __name__ == "__main__":
    main()
