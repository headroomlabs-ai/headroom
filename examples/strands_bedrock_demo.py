#!/usr/bin/env python3
"""Comprehensive Strands + Bedrock Demo for Headroom SDK.

This demo showcases two Headroom integration patterns for AWS Strands Agents:

1. **HeadroomHookProvider** - Compresses tool outputs as they happen
   - Intercepts tool results via Strands hooks
   - Applies SmartCrusher compression to large JSON outputs
   - Shows per-tool compression metrics

2. **HeadroomStrandsModel** - Optimizes entire conversation context
   - Wraps BedrockModel for automatic context optimization
   - Applies message-level transforms before API calls
   - Tracks cumulative savings across the session

Run with:
    python examples/strands_bedrock_demo.py          # Run both demos
    python examples/strands_bedrock_demo.py --hook   # Hook provider demo only
    python examples/strands_bedrock_demo.py --model  # Model wrapper demo only

Requirements:
    - AWS credentials configured (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)
    - pip install strands-agents headroom-ai[strands]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Any

# ============================================================================
# Check Dependencies
# ============================================================================


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    missing = []

    # Check strands-agents
    try:
        from strands import Agent  # noqa: F401
        from strands.models import BedrockModel  # noqa: F401
    except ImportError:
        missing.append("strands-agents")

    # Check headroom
    try:
        from headroom.integrations.strands import (  # noqa: F401
            HeadroomHookProvider,
            HeadroomStrandsModel,
        )
    except ImportError:
        missing.append("headroom-ai[strands]")

    if missing:
        print_box(
            "Missing Dependencies",
            [
                "The following packages are required but not installed:",
                "",
                *[f"  - {pkg}" for pkg in missing],
                "",
                "Install with:",
                f"  pip install {' '.join(missing)}",
            ],
            style="error",
        )
        return False

    return True


def check_aws_credentials() -> bool:
    """Check if AWS credentials are available."""
    has_env_keys = os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")
    has_profile = os.environ.get("AWS_PROFILE")
    has_creds_file = os.path.exists(os.path.expanduser("~/.aws/credentials"))

    if not (has_env_keys or has_profile or has_creds_file):
        print_box(
            "AWS Credentials Not Found",
            [
                "This demo requires AWS credentials to access Bedrock.",
                "",
                "Configure credentials using one of these methods:",
                "",
                "1. Environment variables:",
                "   export AWS_ACCESS_KEY_ID='your-access-key'",
                "   export AWS_SECRET_ACCESS_KEY='your-secret-key'",
                "   export AWS_DEFAULT_REGION='us-west-2'",
                "",
                "2. AWS Profile:",
                "   export AWS_PROFILE='your-profile-name'",
                "",
                "3. AWS credentials file:",
                "   ~/.aws/credentials",
            ],
            style="error",
        )
        return False

    return True


# ============================================================================
# Pretty Printing Utilities
# ============================================================================


def print_box(title: str, lines: list[str], style: str = "normal", width: int = 76) -> None:
    """Print a box with title and content using box drawing characters."""
    if style == "error":
        top_left, top_right = "\u2554", "\u2557"  # Double line
        bot_left, bot_right = "\u255a", "\u255d"
        horiz, vert = "\u2550", "\u2551"
    elif style == "success":
        top_left, top_right = "\u256d", "\u256e"  # Rounded
        bot_left, bot_right = "\u2570", "\u256f"
        horiz, vert = "\u2500", "\u2502"
    else:
        top_left, top_right = "\u250c", "\u2510"  # Normal single
        bot_left, bot_right = "\u2514", "\u2518"
        horiz, vert = "\u2500", "\u2502"

    print()
    print(f"{top_left}{horiz * (width - 2)}{top_right}")

    # Title
    title_padding = (width - 4 - len(title)) // 2
    print(
        f"{vert} {' ' * title_padding}{title}{' ' * (width - 4 - title_padding - len(title))} {vert}"
    )
    print(f"{vert}{horiz * (width - 2)}{vert}")

    # Content lines
    for line in lines:
        # Handle lines longer than width
        if len(line) > width - 4:
            line = line[: width - 7] + "..."
        padding = width - 4 - len(line)
        print(f"{vert} {line}{' ' * padding} {vert}")

    print(f"{bot_left}{horiz * (width - 2)}{bot_right}")
    print()


def print_metrics_table(
    metrics: list[dict[str, Any]],
    headers: list[str],
    keys: list[str],
    title: str = "Metrics",
) -> None:
    """Print metrics in a formatted table."""
    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(header)
        for m in metrics:
            val = m.get(keys[i], "")
            max_width = max(max_width, len(str(val)))
        col_widths.append(min(max_width + 2, 25))

    total_width = sum(col_widths) + len(col_widths) + 1

    print(f"\n  {title}")
    print("  " + "\u2500" * (total_width - 2))

    # Header row
    header_row = "\u2502"
    for i, header in enumerate(headers):
        header_row += f" {header:<{col_widths[i] - 2}} \u2502"
    print("  " + header_row)
    print("  " + "\u2502" + "\u2500" * (total_width - 2) + "\u2502")

    # Data rows
    for m in metrics:
        row = "\u2502"
        for i, key in enumerate(keys):
            val = str(m.get(key, ""))
            if len(val) > col_widths[i] - 2:
                val = val[: col_widths[i] - 5] + "..."
            row += f" {val:<{col_widths[i] - 2}} \u2502"
        print("  " + row)

    print("  " + "\u2500" * total_width)


def print_comparison(before: int, after: int, label: str = "Tokens") -> None:
    """Print a before/after comparison with savings."""
    saved = before - after
    pct = (saved / before * 100) if before > 0 else 0

    bar_width = 40
    before_bar = int((before / max(before, 1)) * bar_width)
    after_bar = int((after / max(before, 1)) * bar_width)

    print(f"\n  {label} Comparison:")
    print(f"  BEFORE: {before:>8,} \u2502{'=' * before_bar}")
    print(f"  AFTER:  {after:>8,} \u2502{'=' * after_bar}")
    print(f"  SAVED:  {saved:>8,} ({pct:.1f}%)")


# ============================================================================
# Mock Tools - Generate Verbose Output
# ============================================================================


def search_documentation(query: str, limit: int = 25) -> str:
    """Search documentation for matching articles.

    Returns search results with titles, snippets, URLs, and metadata.
    Simulates a real documentation search API returning verbose results.
    """
    results = []
    categories = [
        "getting-started",
        "api-reference",
        "tutorials",
        "troubleshooting",
        "best-practices",
    ]
    sources = ["internal-docs", "confluence", "notion", "github-wiki", "readme"]

    for i in range(limit):
        result = {
            "id": f"doc-{random.randint(10000, 99999)}",
            "title": f"{query.title()} Guide - Part {i + 1}",
            "snippet": f"This comprehensive guide covers {query} implementation. "
            f"Learn how to configure, deploy, and maintain {query} in production. "
            f"Includes examples, best practices, and troubleshooting tips for {query}.",
            "url": f"https://docs.example.com/{query.replace(' ', '-')}/section-{i + 1}",
            "category": random.choice(categories),
            "source": random.choice(sources),
            "relevance_score": round(random.uniform(0.5, 1.0), 3),
            "last_updated": (datetime.now() - timedelta(days=random.randint(1, 180))).isoformat(),
            "author": f"Author {random.randint(1, 20)}",
            "word_count": random.randint(500, 5000),
            "views": random.randint(100, 10000),
            "helpful_votes": random.randint(10, 500),
            "tags": random.sample(
                ["aws", "python", "deployment", "security", "performance", "monitoring"],
                k=random.randint(2, 4),
            ),
        }
        results.append(result)

    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    return json.dumps(
        {
            "query": query,
            "total_results": limit * 5,  # Simulate more results available
            "page": 1,
            "per_page": limit,
            "results": results,
        },
        indent=2,
    )


def get_server_logs(server: str, lines: int = 100) -> str:
    """Fetch server logs for analysis.

    Returns JSON log entries with timestamps, levels, messages, and context.
    Simulates verbose application logs with mostly INFO entries and some errors.
    """
    entries = []
    levels = ["DEBUG", "INFO", "INFO", "INFO", "INFO", "WARN", "ERROR"]
    services = ["api-gateway", "auth-service", "data-processor", "cache-layer", "message-queue"]

    for _i in range(lines):
        timestamp = datetime.now() - timedelta(minutes=random.randint(1, 1440))
        level = random.choice(levels)

        if level == "ERROR":
            message = random.choice(
                [
                    f"Connection timeout to {server}-db after 30000ms",
                    "Failed to authenticate request: invalid JWT signature",
                    "Rate limit exceeded for client IP 10.0.0.42",
                    "Database query failed: connection pool exhausted",
                    f"Service {server} health check failed: connection refused",
                ]
            )
        elif level == "WARN":
            message = random.choice(
                [
                    f"Slow query detected on {server}: execution time 2.5s",
                    "Memory usage at 85% - consider scaling",
                    "Retry attempt 2/3 for downstream service call",
                    "Certificate expires in 7 days - renewal required",
                ]
            )
        else:
            message = f"Request processed successfully - endpoint=/api/v1/{server}/data"

        entry = {
            "timestamp": timestamp.isoformat(),
            "level": level,
            "server": server,
            "service": random.choice(services),
            "message": message,
            "trace_id": f"trace-{random.randint(100000, 999999):06x}",
            "span_id": f"span-{random.randint(1000, 9999):04x}",
            "request_id": f"req-{random.randint(10000000, 99999999)}",
            "client_ip": f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}",
            "user_agent": random.choice(
                [
                    "Mozilla/5.0 (compatible; MonitorBot/1.0)",
                    "python-requests/2.31.0",
                    "curl/8.1.2",
                    "PostmanRuntime/7.32.0",
                ]
            ),
            "response_time_ms": random.randint(5, 2000),
            "status_code": 200
            if level in ["DEBUG", "INFO"]
            else random.choice([400, 500, 502, 503]),
            "metadata": {
                "pod": f"{server}-{random.randint(1, 5)}-abc123",
                "node": f"ip-10-0-{random.randint(0, 255)}-{random.randint(1, 254)}.ec2.internal",
                "region": random.choice(["us-west-2", "us-east-1", "eu-west-1"]),
                "version": f"v1.{random.randint(0, 9)}.{random.randint(0, 20)}",
            },
        }
        entries.append(entry)

    entries.sort(key=lambda x: x["timestamp"], reverse=True)

    return json.dumps(
        {
            "server": server,
            "log_count": lines,
            "time_range": {
                "start": entries[-1]["timestamp"] if entries else None,
                "end": entries[0]["timestamp"] if entries else None,
            },
            "entries": entries,
        },
        indent=2,
    )


def query_database(sql: str, limit: int = 50) -> str:
    """Execute a database query and return results.

    Returns rows of data as if from a real database query.
    Simulates customer/order/transaction data.
    """
    # Parse table name from SQL (simple simulation)
    table = "records"
    for word in sql.lower().split():
        if word in ["users", "orders", "transactions", "customers", "products", "events"]:
            table = word
            break

    rows = []
    statuses = ["active", "pending", "completed", "cancelled", "refunded"]

    for i in range(limit):
        if table == "users":
            row = {
                "user_id": f"usr-{random.randint(100000, 999999)}",
                "email": f"user{i}@example.com",
                "name": f"Customer {i}",
                "status": random.choice(["active", "inactive", "suspended"]),
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                "last_login": (
                    datetime.now() - timedelta(hours=random.randint(1, 720))
                ).isoformat(),
                "plan": random.choice(["free", "basic", "pro", "enterprise"]),
                "country": random.choice(["US", "UK", "DE", "FR", "JP", "AU"]),
            }
        elif table == "orders":
            row = {
                "order_id": f"ord-{random.randint(100000, 999999)}",
                "customer_id": f"usr-{random.randint(100000, 999999)}",
                "total": round(random.uniform(10, 1000), 2),
                "currency": random.choice(["USD", "EUR", "GBP"]),
                "status": random.choice(statuses),
                "items_count": random.randint(1, 10),
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
                "shipped_at": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
                if random.random() > 0.3
                else None,
            }
        else:
            row = {
                "id": i + 1,
                "record_type": table,
                "value": random.randint(100, 10000),
                "status": random.choice(statuses),
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                "metadata": {
                    "source": random.choice(["web", "api", "import", "sync"]),
                    "version": f"v{random.randint(1, 5)}",
                },
            }
        rows.append(row)

    return json.dumps(
        {
            "query": sql,
            "table": table,
            "row_count": limit,
            "total_available": limit * 10,
            "execution_time_ms": random.randint(10, 500),
            "rows": rows,
        },
        indent=2,
    )


def get_system_metrics(timerange: str = "1h", service: str = "all") -> str:
    """Get system metrics for monitoring.

    Returns time-series data points for CPU, memory, latency, and error rates.
    Simulates Prometheus/CloudWatch style metrics.
    """
    # Parse timerange to determine number of points
    points = {"5m": 10, "15m": 30, "1h": 60, "6h": 72, "24h": 144}.get(timerange, 60)

    data_points = []
    services_list = ["api", "worker", "cache", "database"] if service == "all" else [service]

    for svc in services_list:
        for i in range(points):
            timestamp = datetime.now() - timedelta(minutes=i * (60 // min(points, 60)))

            # Inject some anomalies
            is_anomaly = random.random() < 0.05

            point = {
                "timestamp": timestamp.isoformat(),
                "service": svc,
                "metrics": {
                    "cpu_percent": round(
                        random.uniform(70, 95) if is_anomaly else random.uniform(20, 45), 2
                    ),
                    "memory_percent": round(
                        random.uniform(80, 95) if is_anomaly else random.uniform(40, 65), 2
                    ),
                    "memory_mb": random.randint(2000, 4000)
                    if is_anomaly
                    else random.randint(500, 1500),
                    "latency_p50_ms": random.randint(100, 500)
                    if is_anomaly
                    else random.randint(10, 50),
                    "latency_p95_ms": random.randint(500, 2000)
                    if is_anomaly
                    else random.randint(50, 150),
                    "latency_p99_ms": random.randint(1000, 5000)
                    if is_anomaly
                    else random.randint(100, 300),
                    "request_rate_per_sec": random.randint(500, 2000)
                    if is_anomaly
                    else random.randint(50, 200),
                    "error_rate_percent": round(
                        random.uniform(5, 15) if is_anomaly else random.uniform(0, 1), 3
                    ),
                    "active_connections": random.randint(200, 500)
                    if is_anomaly
                    else random.randint(20, 80),
                },
                "health": "degraded" if is_anomaly else "healthy",
                "region": random.choice(["us-west-2", "us-east-1", "eu-west-1"]),
            }
            data_points.append(point)

    # Calculate summary statistics
    all_cpu = [p["metrics"]["cpu_percent"] for p in data_points]
    all_mem = [p["metrics"]["memory_percent"] for p in data_points]
    all_latency = [p["metrics"]["latency_p50_ms"] for p in data_points]

    return json.dumps(
        {
            "timerange": timerange,
            "service": service,
            "data_points_count": len(data_points),
            "summary": {
                "cpu": {
                    "min": min(all_cpu),
                    "max": max(all_cpu),
                    "avg": sum(all_cpu) / len(all_cpu),
                },
                "memory": {
                    "min": min(all_mem),
                    "max": max(all_mem),
                    "avg": sum(all_mem) / len(all_mem),
                },
                "latency_p50": {
                    "min": min(all_latency),
                    "max": max(all_latency),
                    "avg": sum(all_latency) / len(all_latency),
                },
            },
            "data_points": data_points,
        },
        indent=2,
    )


# ============================================================================
# Demo 1: HeadroomHookProvider
# ============================================================================


def run_hook_provider_demo(region: str = "us-west-2") -> dict[str, Any]:
    """Demonstrate HeadroomHookProvider for tool output compression.

    Returns metrics from the demo run.
    """
    from strands import Agent, tool
    from strands.models import BedrockModel

    from headroom.integrations.strands import HeadroomHookProvider

    print_box(
        "Demo 1: HeadroomHookProvider",
        [
            "The HeadroomHookProvider intercepts tool outputs and compresses",
            "them BEFORE they're added to the conversation context.",
            "",
            "This reduces token usage for subsequent LLM calls by eliminating",
            "redundant data from verbose tool outputs.",
            "",
            "Using: Claude 3 Haiku (anthropic.claude-3-haiku-20240307-v1:0)",
        ],
    )

    # Define tools with @tool decorator
    @tool
    def search_docs_tool(query: str) -> str:
        """Search documentation for articles matching the query.

        Args:
            query: The search query to find relevant documentation

        Returns:
            JSON array of search results with titles, snippets, and URLs
        """
        return search_documentation(query, limit=25)

    @tool
    def get_logs_tool(server: str, lines: int = 100) -> str:
        """Fetch server logs for analysis and troubleshooting.

        Args:
            server: Name of the server to fetch logs from
            lines: Number of log lines to retrieve (default: 100)

        Returns:
            JSON array of log entries with timestamps and messages
        """
        return get_server_logs(server, lines=lines)

    @tool
    def query_db_tool(sql: str) -> str:
        """Execute a database query and return results.

        Args:
            sql: SQL query to execute (e.g., SELECT * FROM users)

        Returns:
            JSON array of database rows
        """
        return query_database(sql, limit=50)

    @tool
    def get_metrics_tool(timerange: str = "1h") -> str:
        """Get system metrics for the specified time range.

        Args:
            timerange: Time range for metrics (5m, 15m, 1h, 6h, 24h)

        Returns:
            JSON object with time-series metrics data
        """
        return get_system_metrics(timerange)

    # Create BedrockModel
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name=region,
        temperature=0.1,
    )

    # Create HeadroomHookProvider
    hook_provider = HeadroomHookProvider(
        compress_tool_outputs=True,
        min_tokens_to_compress=100,  # Compress outputs with 100+ tokens
        preserve_errors=True,
    )

    # Create agent with hook
    agent = Agent(
        model=model,
        tools=[search_docs_tool, get_logs_tool, query_db_tool, get_metrics_tool],
        hooks=[hook_provider],
    )

    print("\n  Running agent queries that trigger tools with verbose output...")
    print("  " + "-" * 60)

    # Query 1: Search documentation
    print("\n  Query 1: Searching documentation...")
    result1 = agent(
        "Search the documentation for 'authentication setup' and summarize "
        "the top 3 most relevant articles you find."
    )
    print(f"  Response: {str(result1)[:200]}...")

    # Query 2: Get server logs
    print("\n  Query 2: Fetching server logs...")
    result2 = agent(
        "Get the logs from server 'api-gateway' (100 lines) and tell me "
        "how many ERROR and WARN level entries there are."
    )
    print(f"  Response: {str(result2)[:200]}...")

    # Query 3: Query database
    print("\n  Query 3: Running database query...")
    result3 = agent(
        "Query the orders table and tell me how many orders have status 'completed' "
        "and what the average order total is."
    )
    print(f"  Response: {str(result3)[:200]}...")

    # Query 4: Get metrics
    print("\n  Query 4: Fetching system metrics...")
    result4 = agent(
        "Get the system metrics for the last hour and identify if there are "
        "any services with high CPU usage (>70%) or memory issues."
    )
    print(f"  Response: {str(result4)[:200]}...")

    # Get metrics
    metrics = hook_provider.get_savings_summary()

    # Display results
    print_box(
        "HeadroomHookProvider Results",
        [
            f"Tool calls processed:    {metrics['total_requests']}",
            f"Compressions applied:    {metrics['compressed_requests']}",
            "",
            f"Tokens BEFORE compression: {metrics['total_tokens_before']:,}",
            f"Tokens AFTER compression:  {metrics['total_tokens_after']:,}",
            f"Tokens SAVED:              {metrics['total_tokens_saved']:,}",
            "",
            f"Average savings:           {metrics['average_savings_percent']:.1f}%",
        ],
        style="success",
    )

    # Show per-tool breakdown
    if hook_provider.metrics_history:
        tool_metrics = []
        for m in hook_provider.metrics_history:
            tool_metrics.append(
                {
                    "tool": m.tool_name[:20],
                    "before": f"{m.tokens_before:,}",
                    "after": f"{m.tokens_after:,}",
                    "saved": f"{m.tokens_saved:,}",
                    "pct": f"{m.savings_percent:.1f}%",
                }
            )

        print_metrics_table(
            tool_metrics,
            headers=["Tool", "Before", "After", "Saved", "%"],
            keys=["tool", "before", "after", "saved", "pct"],
            title="Per-Tool Compression Breakdown",
        )

    print_comparison(
        metrics["total_tokens_before"],
        metrics["total_tokens_after"],
        "Tool Output Tokens",
    )

    return metrics


# ============================================================================
# Demo 2: HeadroomStrandsModel
# ============================================================================


def run_model_wrapper_demo(region: str = "us-west-2") -> dict[str, Any]:
    """Demonstrate HeadroomStrandsModel for conversation optimization.

    Returns metrics from the demo run.
    """
    from strands import Agent, tool
    from strands.models import BedrockModel

    from headroom import HeadroomConfig
    from headroom.integrations.strands import HeadroomStrandsModel

    print_box(
        "Demo 2: HeadroomStrandsModel",
        [
            "HeadroomStrandsModel wraps the Bedrock model to optimize the",
            "ENTIRE conversation context before each API call.",
            "",
            "As conversations grow with tool outputs and history, the",
            "model wrapper applies transforms to reduce context size.",
            "",
            "Using: Claude 3 Haiku wrapped with HeadroomStrandsModel",
        ],
    )

    # Define tools
    @tool
    def verbose_search(query: str) -> str:
        """Search for information with verbose results.

        Args:
            query: Search query

        Returns:
            Detailed search results
        """
        return search_documentation(query, limit=30)

    @tool
    def verbose_logs(server: str) -> str:
        """Get verbose server logs.

        Args:
            server: Server name

        Returns:
            Detailed log entries
        """
        return get_server_logs(server, lines=150)

    @tool
    def verbose_metrics(timerange: str = "1h") -> str:
        """Get verbose metrics data.

        Args:
            timerange: Time range

        Returns:
            Detailed metrics
        """
        return get_system_metrics(timerange)

    @tool
    def verbose_database(table: str) -> str:
        """Query database with verbose results.

        Args:
            table: Table name to query

        Returns:
            Database records
        """
        return query_database(f"SELECT * FROM {table}", limit=60)

    # Create base Bedrock model
    base_model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name=region,
        temperature=0.1,
    )

    # Configure Headroom
    config = HeadroomConfig()
    config.smart_crusher.enabled = True
    config.smart_crusher.min_tokens_to_crush = 100
    config.smart_crusher.max_items_after_crush = 20

    # Wrap with HeadroomStrandsModel
    optimized_model = HeadroomStrandsModel(
        wrapped_model=base_model,
        config=config,
        auto_detect_provider=True,
    )

    # Create agent
    agent = Agent(
        model=optimized_model,
        tools=[verbose_search, verbose_logs, verbose_metrics, verbose_database],
    )

    print("\n  Building up a multi-turn conversation with verbose tool outputs...")
    print("  " + "-" * 60)

    # Simulate a multi-turn conversation
    turns = [
        ("Turn 1", "Search for documentation about 'kubernetes deployment' and give me a summary."),
        ("Turn 2", "Now get the logs from the 'worker-service' server and identify any errors."),
        ("Turn 3", "Query the orders database and tell me the distribution of order statuses."),
        ("Turn 4", "Get the system metrics for the last hour and highlight any anomalies."),
        ("Turn 5", "Based on everything you've found, what's the overall system health status?"),
    ]

    for turn_name, query in turns:
        print(f"\n  {turn_name}: {query[:60]}...")
        result = agent(query)
        print(f"  Response: {str(result)[:150]}...")

    # Get metrics
    metrics = optimized_model.get_savings_summary()

    # Display results
    print_box(
        "HeadroomStrandsModel Results",
        [
            f"API calls made:            {metrics['total_requests']}",
            "",
            f"Total tokens BEFORE opt:   {metrics['total_tokens_before']:,}",
            f"Total tokens AFTER opt:    {metrics['total_tokens_after']:,}",
            f"Total tokens SAVED:        {metrics['total_tokens_saved']:,}",
            "",
            f"Average savings per call:  {metrics['average_savings_percent']:.1f}%",
        ],
        style="success",
    )

    # Show per-request breakdown
    if optimized_model.metrics_history:
        request_metrics = []
        for i, m in enumerate(optimized_model.metrics_history):
            request_metrics.append(
                {
                    "request": f"Request {i + 1}",
                    "before": f"{m.tokens_before:,}",
                    "after": f"{m.tokens_after:,}",
                    "saved": f"{m.tokens_saved:,}",
                    "pct": f"{m.savings_percent:.1f}%",
                }
            )

        print_metrics_table(
            request_metrics,
            headers=["Request", "Before", "After", "Saved", "%"],
            keys=["request", "before", "after", "saved", "pct"],
            title="Per-Request Optimization Breakdown",
        )

    print_comparison(
        metrics["total_tokens_before"],
        metrics["total_tokens_after"],
        "Conversation Tokens",
    )

    return metrics


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    """Run the Strands Bedrock demo."""
    parser = argparse.ArgumentParser(
        description="Headroom + Strands Bedrock Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/strands_bedrock_demo.py          # Run both demos
  python examples/strands_bedrock_demo.py --hook   # Hook provider only
  python examples/strands_bedrock_demo.py --model  # Model wrapper only

Environment Variables:
  AWS_ACCESS_KEY_ID       AWS access key
  AWS_SECRET_ACCESS_KEY   AWS secret key
  AWS_DEFAULT_REGION      AWS region (default: us-west-2)
  AWS_PROFILE             AWS profile name (alternative to keys)
        """,
    )
    parser.add_argument(
        "--hook",
        action="store_true",
        help="Run only the HeadroomHookProvider demo",
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help="Run only the HeadroomStrandsModel demo",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
        help="AWS region for Bedrock (default: us-west-2)",
    )

    args = parser.parse_args()

    # If neither flag is set, run both
    run_hook = args.hook or (not args.hook and not args.model)
    run_model = args.model or (not args.hook and not args.model)

    # Print header
    print_box(
        "Headroom + Strands Bedrock Demo",
        [
            "This demo showcases Headroom's integration with AWS Strands Agents.",
            "",
            "Headroom provides two integration patterns:",
            "  1. HeadroomHookProvider - Compress tool outputs in real-time",
            "  2. HeadroomStrandsModel - Optimize entire conversation context",
            "",
            f"Region: {args.region}",
            "Model:  Claude 3 Haiku (fast and cost-effective for demos)",
        ],
    )

    # Check dependencies
    if not check_dependencies():
        return 1

    # Check AWS credentials
    if not check_aws_credentials():
        return 1

    print("\n  All checks passed. Starting demos...\n")

    all_metrics = {}

    try:
        # Run hook provider demo
        if run_hook:
            hook_metrics = run_hook_provider_demo(region=args.region)
            all_metrics["hook_provider"] = hook_metrics

        # Run model wrapper demo
        if run_model:
            model_metrics = run_model_wrapper_demo(region=args.region)
            all_metrics["model_wrapper"] = model_metrics

        # Print final summary
        if run_hook and run_model:
            total_before = all_metrics.get("hook_provider", {}).get(
                "total_tokens_before", 0
            ) + all_metrics.get("model_wrapper", {}).get("total_tokens_before", 0)
            total_after = all_metrics.get("hook_provider", {}).get(
                "total_tokens_after", 0
            ) + all_metrics.get("model_wrapper", {}).get("total_tokens_after", 0)
            total_saved = total_before - total_after
            total_pct = (total_saved / total_before * 100) if total_before > 0 else 0

            # Estimate cost savings (Claude 3 Haiku pricing)
            # Input: $0.25 / 1M tokens, Output: $1.25 / 1M tokens
            cost_per_token = 0.25 / 1_000_000
            cost_saved = total_saved * cost_per_token

            print_box(
                "Session Summary",
                [
                    "Combined metrics from both demos:",
                    "",
                    f"Total tokens processed:    {total_before:,}",
                    f"Total tokens after opt:    {total_after:,}",
                    f"Total tokens saved:        {total_saved:,} ({total_pct:.1f}%)",
                    "",
                    f"Estimated cost savings:    ${cost_saved:.6f}",
                    "(At scale, these savings compound significantly!)",
                    "",
                    "Integration patterns demonstrated:",
                    "  [x] HeadroomHookProvider - Real-time tool output compression",
                    "  [x] HeadroomStrandsModel - Full context optimization",
                ],
                style="success",
            )

        return 0

    except Exception as e:
        print_box(
            "Error",
            [
                f"An error occurred: {type(e).__name__}",
                "",
                str(e)[:200],
                "",
                "Common issues:",
                "  - Invalid AWS credentials",
                "  - Bedrock not enabled in your AWS account",
                "  - Model not available in selected region",
                "  - Rate limiting from too many requests",
            ],
            style="error",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
