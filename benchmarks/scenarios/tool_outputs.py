"""Data generators for tool output benchmarks.

This module provides realistic data generators that simulate common tool
output patterns encountered in agentic workflows:

- Search results: Elasticsearch/vector search responses with scores
- Log entries: Structured logs with timestamps and severity levels
- API responses: Paginated REST API responses
- Database rows: Query results with various data types

All generators produce data that exercises SmartCrusher's pattern detection
and compression strategies.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Any


def generate_search_results(
    n: int,
    include_uuid_needles: int = 2,
    include_errors: int = 1,
) -> list[dict[str, Any]]:
    """Generate search results with relevance scores.

    Simulates Elasticsearch-style search results with varying scores,
    occasional error entries, and UUID "needle" records for relevance
    testing.

    Args:
        n: Number of results to generate.
        include_uuid_needles: Number of items to mark with UUIDs.
        include_errors: Number of error items to include.

    Returns:
        List of search result dictionaries.

    Example:
        results = generate_search_results(100)
        # [{"id": "doc_1", "score": 0.95, "title": "...", ...}, ...]
    """
    results = []

    # Generate base results
    for i in range(n):
        # Score decreases with position (realistic search behavior)
        base_score = max(0.1, 1.0 - (i * 0.8 / n))
        jitter = random.uniform(-0.05, 0.05)
        score = max(0.0, min(1.0, base_score + jitter))

        result = {
            "id": f"doc_{i}",
            "score": round(score, 4),
            "title": _generate_title(),
            "snippet": _generate_snippet(),
            "source": random.choice(["web", "internal", "docs", "api"]),
            "metadata": {
                "author": _generate_name(),
                "created_at": _generate_timestamp(i),
                "category": random.choice(["technical", "guide", "reference", "tutorial"]),
            },
        }
        results.append(result)

    # Insert UUID needles at random positions
    needle_indices = random.sample(range(len(results)), min(include_uuid_needles, len(results)))
    for idx in needle_indices:
        results[idx]["uuid"] = str(uuid.uuid4())
        results[idx]["is_needle"] = True

    # Insert error items
    error_indices = random.sample(
        [i for i in range(len(results)) if i not in needle_indices],
        min(include_errors, len(results) - len(needle_indices)),
    )
    for idx in error_indices:
        results[idx]["error"] = random.choice(
            [
                "Index out of range",
                "Document not found",
                "Permission denied",
                "Timeout exceeded",
            ]
        )
        results[idx]["status"] = "failed"

    return results


def generate_log_entries(
    n: int,
    include_errors: int = 5,
    include_critical: int = 1,
) -> list[dict[str, Any]]:
    """Generate log entries with timestamps and severity.

    Simulates structured logging output with various severity levels,
    realistic timestamps, and clusterable message patterns.

    Args:
        n: Number of log entries to generate.
        include_errors: Number of ERROR level entries.
        include_critical: Number of CRITICAL level entries.

    Returns:
        List of log entry dictionaries.

    Example:
        logs = generate_log_entries(1000)
        # [{"timestamp": "...", "level": "INFO", "message": "...", ...}, ...]
    """
    entries = []
    base_time = datetime(2025, 1, 6, 0, 0, 0)

    # Message templates for clustering detection
    message_templates = [
        "Request processed successfully for user {user}",
        "Database query completed in {ms}ms",
        "Cache hit for key {key}",
        "API call to {service} returned {status}",
        "Background job {job} started",
        "Background job {job} completed",
        "Health check passed for {component}",
        "Metrics exported: {count} datapoints",
    ]

    error_templates = [
        "Connection failed to {service}: timeout after {ms}ms",
        "Database error: {error_type}",
        "Failed to process request: {error}",
        "Rate limit exceeded for user {user}",
        "Invalid input: {validation_error}",
    ]

    critical_templates = [
        "CRITICAL: Database connection pool exhausted",
        "CRITICAL: Memory usage exceeded threshold (95%)",
        "CRITICAL: Service {service} unresponsive for 60s",
        "CRITICAL: Data corruption detected in table {table}",
    ]

    for i in range(n):
        timestamp = base_time + timedelta(seconds=i * 0.5)

        if i < include_critical:
            level = "CRITICAL"
            template = random.choice(critical_templates)
        elif i < include_critical + include_errors:
            level = "ERROR"
            template = random.choice(error_templates)
        elif random.random() < 0.1:
            level = "WARNING"
            template = random.choice(error_templates)
        elif random.random() < 0.1:
            level = "DEBUG"
            template = random.choice(message_templates)
        else:
            level = "INFO"
            template = random.choice(message_templates)

        message = _format_template(template)

        entry = {
            "timestamp": timestamp.isoformat() + "Z",
            "level": level,
            "logger": random.choice(["app", "api", "worker", "scheduler"]),
            "message": message,
            "service": "headroom-benchmark",
            "hostname": f"worker-{random.randint(1, 10):02d}",
            "trace_id": f"trace_{uuid.uuid4().hex[:16]}",
        }

        # Add exception info for errors
        if level in ("ERROR", "CRITICAL"):
            entry["exception"] = {
                "type": random.choice(
                    ["TimeoutError", "ConnectionError", "ValueError", "RuntimeError"]
                ),
                "message": message,
                "stacktrace": _generate_stacktrace(),
            }

        entries.append(entry)

    # Shuffle to make errors appear randomly throughout
    random.shuffle(entries)

    return entries


def generate_api_responses(
    n: int,
    include_pagination: bool = True,
) -> list[dict[str, Any]]:
    """Generate API response items.

    Simulates paginated REST API responses with varying data types
    and nested structures.

    Args:
        n: Number of items to generate.
        include_pagination: Include pagination metadata.

    Returns:
        List of API response item dictionaries.

    Example:
        items = generate_api_responses(100)
        # [{"id": 1, "type": "user", "attributes": {...}, ...}, ...]
    """
    items = []

    for i in range(n):
        item_type = random.choice(["user", "product", "order", "event", "metric"])

        item = {
            "id": i + 1,
            "type": item_type,
            "attributes": _generate_attributes(item_type),
            "links": {
                "self": f"/api/v1/{item_type}s/{i + 1}",
            },
            "meta": {
                "created_at": _generate_timestamp(i),
                "updated_at": _generate_timestamp(i + random.randint(0, 100)),
            },
        }

        # Add relationships for some items
        if random.random() < 0.3:
            item["relationships"] = {
                "parent": {"id": random.randint(1, max(1, i)), "type": item_type},
            }

        items.append(item)

    return items


def generate_database_rows(
    n: int,
    table_type: str = "mixed",
) -> list[dict[str, Any]]:
    """Generate database query results.

    Simulates SQL query results with realistic data types, NULL values,
    and numeric fields for anomaly detection testing.

    Args:
        n: Number of rows to generate.
        table_type: Type of data ("users", "metrics", "transactions", "mixed").

    Returns:
        List of row dictionaries.

    Example:
        rows = generate_database_rows(1000, table_type="metrics")
        # [{"id": 1, "metric_name": "cpu_usage", "value": 45.2, ...}, ...]
    """
    rows = []

    # Generate with some anomalies for variance detection
    mean_value = 100.0
    std_value = 15.0

    for i in range(n):
        if table_type == "users":
            row = _generate_user_row(i)
        elif table_type == "metrics":
            row = _generate_metric_row(i, mean_value, std_value)
        elif table_type == "transactions":
            row = _generate_transaction_row(i)
        else:  # mixed
            generator = random.choice(
                [
                    _generate_user_row,
                    lambda i: _generate_metric_row(i, mean_value, std_value),
                    _generate_transaction_row,
                ]
            )
            row = generator(i)

        rows.append(row)

    # Insert anomalies at specific positions
    if n > 20:
        anomaly_indices = [n // 4, n // 2, 3 * n // 4]
        for idx in anomaly_indices:
            if "value" in rows[idx]:
                # Insert anomaly (> 3 std from mean)
                rows[idx]["value"] = mean_value + (4 * std_value * random.choice([-1, 1]))
                rows[idx]["is_anomaly"] = True

    return rows


# Helper functions


def _generate_title() -> str:
    """Generate a realistic document title."""
    prefixes = ["How to", "Guide to", "Understanding", "Introduction to", "Advanced"]
    topics = ["Python", "API Design", "Database Optimization", "Caching", "Async Programming"]
    suffixes = ["Best Practices", "in Production", "for Beginners", "Deep Dive", "Tutorial"]
    return f"{random.choice(prefixes)} {random.choice(topics)} - {random.choice(suffixes)}"


def _generate_snippet() -> str:
    """Generate a document snippet."""
    sentences = [
        "This comprehensive guide covers the essential concepts.",
        "Learn how to implement this pattern effectively.",
        "Performance optimization techniques are discussed in detail.",
        "Step-by-step instructions for getting started.",
        "Common pitfalls and how to avoid them.",
    ]
    return " ".join(random.sample(sentences, k=2))


def _generate_name() -> str:
    """Generate a person name."""
    first_names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"


def _generate_timestamp(offset_days: int = 0) -> str:
    """Generate an ISO timestamp."""
    base = datetime(2025, 1, 1, 12, 0, 0)
    dt = base + timedelta(
        days=offset_days, hours=random.randint(0, 23), minutes=random.randint(0, 59)
    )
    return dt.isoformat() + "Z"


def _format_template(template: str) -> str:
    """Fill in template placeholders with random values."""
    replacements = {
        "{user}": f"user_{random.randint(1000, 9999)}",
        "{ms}": str(random.randint(10, 5000)),
        "{key}": f"cache:{random.randint(1, 100)}",
        "{service}": random.choice(["auth", "users", "payments", "notifications"]),
        "{status}": random.choice(["200", "201", "204"]),
        "{job}": f"job_{random.randint(1, 100)}",
        "{component}": random.choice(["database", "cache", "queue", "api"]),
        "{count}": str(random.randint(100, 10000)),
        "{error_type}": random.choice(["ConnectionError", "QueryTimeout", "IntegrityError"]),
        "{error}": random.choice(["Invalid input", "Not found", "Unauthorized"]),
        "{validation_error}": random.choice(["Missing field", "Invalid format", "Out of range"]),
        "{table}": random.choice(["users", "orders", "metrics"]),
    }

    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def _generate_stacktrace() -> str:
    """Generate a fake stacktrace."""
    files = ["app.py", "handlers.py", "services.py", "database.py", "utils.py"]
    lines = []
    for _ in range(random.randint(3, 6)):
        file = random.choice(files)
        lineno = random.randint(10, 500)
        func = random.choice(["handle_request", "process", "execute", "query", "validate"])
        lines.append(f'  File "{file}", line {lineno}, in {func}')
    return "\n".join(lines)


def _generate_attributes(item_type: str) -> dict[str, Any]:
    """Generate attributes based on item type."""
    if item_type == "user":
        return {
            "name": _generate_name(),
            "email": f"user_{random.randint(1000, 9999)}@example.com",
            "status": random.choice(["active", "inactive", "pending"]),
        }
    elif item_type == "product":
        return {
            "name": f"Product {random.randint(1, 1000)}",
            "price": round(random.uniform(9.99, 999.99), 2),
            "category": random.choice(["electronics", "books", "clothing"]),
        }
    elif item_type == "order":
        return {
            "total": round(random.uniform(10, 1000), 2),
            "status": random.choice(["pending", "shipped", "delivered"]),
            "items_count": random.randint(1, 10),
        }
    elif item_type == "event":
        return {
            "name": random.choice(["click", "view", "purchase", "signup"]),
            "source": random.choice(["web", "mobile", "api"]),
        }
    else:  # metric
        return {
            "name": random.choice(["cpu_usage", "memory", "latency", "requests"]),
            "value": round(random.uniform(0, 100), 2),
            "unit": random.choice(["percent", "ms", "count"]),
        }


def _generate_user_row(i: int) -> dict[str, Any]:
    """Generate a user table row."""
    return {
        "id": i + 1,
        "username": f"user_{random.randint(1000, 9999)}",
        "email": f"user{i}@example.com",
        "created_at": _generate_timestamp(i),
        "status": random.choice(["active", "inactive", "pending"]),
        "login_count": random.randint(0, 1000),
        "is_verified": random.choice([True, False, None]),
    }


def _generate_metric_row(i: int, mean: float, std: float) -> dict[str, Any]:
    """Generate a metrics table row with normal distribution."""
    value = random.gauss(mean, std)
    return {
        "id": i + 1,
        "metric_name": random.choice(["cpu_usage", "memory_mb", "request_count", "latency_ms"]),
        "value": round(value, 2),
        "timestamp": _generate_timestamp(i // 100),
        "host": f"server-{random.randint(1, 20):02d}",
        "region": random.choice(["us-east-1", "us-west-2", "eu-west-1"]),
    }


def _generate_transaction_row(i: int) -> dict[str, Any]:
    """Generate a transactions table row."""
    return {
        "id": i + 1,
        "user_id": random.randint(1, 10000),
        "amount": round(random.uniform(-1000, 1000), 2),
        "currency": random.choice(["USD", "EUR", "GBP"]),
        "status": random.choice(["completed", "pending", "failed"]),
        "created_at": _generate_timestamp(i),
        "reference": f"TXN-{uuid.uuid4().hex[:8].upper()}",
    }
