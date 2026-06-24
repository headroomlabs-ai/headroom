"""Mock tools that generate realistic large outputs.

These simulate real-world API responses that benefit from Headroom compression:
- Database queries returning many rows
- Search APIs returning many results
- Log analysis tools returning many entries
- Monitoring tools returning many metrics
"""

import json
import random
from datetime import datetime, timedelta


def generate_user_database_results(query: str, count: int = 100) -> str:
    """Simulate a database query returning user records.

    Real-world scenario: Agent searches for users matching criteria,
    database returns 100+ records but only a few are actually relevant.
    """
    users = []
    departments = ["Engineering", "Sales", "Marketing", "Support", "HR", "Finance"]
    statuses = ["active", "inactive", "pending", "suspended"]

    for i in range(count):
        user = {
            "id": f"usr_{random.randint(100000, 999999)}",
            "email": f"user{i}@example.com",
            "name": f"User {i} {'Smith' if i % 3 == 0 else 'Johnson' if i % 3 == 1 else 'Williams'}",
            "department": random.choice(departments),
            "status": random.choice(statuses),
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            "last_login": (datetime.now() - timedelta(hours=random.randint(1, 720))).isoformat(),
            "role": random.choice(["admin", "user", "viewer", "editor"]),
            "metadata": {
                "preferences": {
                    "theme": random.choice(["dark", "light"]),
                    "notifications": random.choice([True, False]),
                    "timezone": random.choice(["UTC", "PST", "EST", "CST"]),
                },
                "tags": random.sample(
                    ["premium", "verified", "beta", "enterprise"], k=random.randint(0, 3)
                ),
                "login_count": random.randint(1, 500),
            },
        }
        users.append(user)

    return json.dumps({"results": users, "total": count, "query": query}, indent=2)


def generate_search_results(query: str, count: int = 50) -> str:
    """Simulate a search API returning many results.

    Real-world scenario: Agent searches documentation/knowledge base,
    returns many results ranked by relevance.
    """
    results = []
    categories = ["documentation", "tutorial", "api-reference", "faq", "blog", "changelog"]

    for i in range(count):
        result = {
            "id": f"doc_{random.randint(10000, 99999)}",
            "title": f"Document {i}: {query.title()} Guide",
            "snippet": f"This document covers {query}. " * random.randint(2, 5)
            + f"Learn more about implementing {query} in your application...",
            "url": f"https://docs.example.com/{query.replace(' ', '-')}/{i}",
            "category": random.choice(categories),
            "relevance_score": round(random.uniform(0.5, 1.0), 3),
            "last_updated": (datetime.now() - timedelta(days=random.randint(1, 180))).isoformat(),
            "author": f"Author {random.randint(1, 20)}",
            "views": random.randint(100, 10000),
            "helpful_votes": random.randint(0, 500),
        }
        results.append(result)

    # Sort by relevance
    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    return json.dumps({"results": results, "total": count, "query": query}, indent=2)


def generate_log_entries(service: str, count: int = 200) -> str:
    """Simulate a log analysis tool returning many entries.

    Real-world scenario: Agent investigates an issue by searching logs,
    returns many entries but only a few show the actual error.
    """
    entries = []
    levels = ["DEBUG", "INFO", "INFO", "INFO", "WARN", "ERROR"]  # Most are INFO

    for _i in range(count):
        timestamp = datetime.now() - timedelta(minutes=random.randint(1, 1440))
        level = random.choice(levels)

        if level == "ERROR":
            message = random.choice(
                [
                    f"Connection refused to {service}-db: timeout after 30s",
                    "Failed to process request: NullPointerException at line 42",
                    "Authentication failed for user: invalid token",
                    "Rate limit exceeded: 429 Too Many Requests",
                ]
            )
        elif level == "WARN":
            message = random.choice(
                [
                    "Slow query detected: took 2.5s",
                    "Memory usage high: 85% of heap",
                    "Retrying request after transient failure",
                ]
            )
        else:
            message = f"Processing request {random.randint(1000, 9999)} for {service}"

        entry = {
            "timestamp": timestamp.isoformat(),
            "level": level,
            "service": service,
            "message": message,
            "trace_id": f"trace_{random.randint(100000, 999999)}",
            "span_id": f"span_{random.randint(1000, 9999)}",
            "host": f"{service}-{random.randint(1, 5)}.prod.internal",
            "metadata": {
                "request_id": f"req_{random.randint(100000, 999999)}",
                "user_agent": "Mozilla/5.0" if random.random() > 0.5 else "API-Client/1.0",
                "duration_ms": random.randint(1, 5000),
            },
        }
        entries.append(entry)

    # Sort by timestamp
    entries.sort(key=lambda x: x["timestamp"], reverse=True)

    return json.dumps({"entries": entries, "total": count, "service": service}, indent=2)


def generate_metrics_data(service: str, count: int = 100) -> str:
    """Simulate a monitoring tool returning time-series metrics.

    Real-world scenario: Agent checks service health metrics,
    returns many data points but only anomalies matter.
    """
    metrics = []
    now = datetime.now()

    for i in range(count):
        timestamp = now - timedelta(minutes=i * 5)

        # Inject some anomalies
        is_anomaly = random.random() < 0.05

        metric = {
            "timestamp": timestamp.isoformat(),
            "service": service,
            "cpu_percent": random.uniform(60, 95) if is_anomaly else random.uniform(20, 40),
            "memory_percent": random.uniform(80, 98) if is_anomaly else random.uniform(40, 60),
            "request_rate": random.randint(800, 2000) if is_anomaly else random.randint(100, 300),
            "error_rate": random.uniform(5, 15) if is_anomaly else random.uniform(0, 1),
            "latency_p50_ms": random.randint(200, 500) if is_anomaly else random.randint(10, 50),
            "latency_p99_ms": random.randint(1000, 3000) if is_anomaly else random.randint(50, 200),
            "active_connections": random.randint(500, 1000)
            if is_anomaly
            else random.randint(50, 150),
        }
        metrics.append(metric)

    return json.dumps({"metrics": metrics, "service": service, "interval": "5m"}, indent=2)


def generate_api_response(endpoint: str, count: int = 75) -> str:
    """Simulate a generic API returning paginated data.

    Real-world scenario: Agent fetches data from an external API,
    receives large paginated response.
    """
    items = []

    for i in range(count):
        item = {
            "id": i + 1,
            "uuid": f"{random.randint(10000000, 99999999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(100000000000, 999999999999)}",
            "name": f"Item {i}",
            "description": f"This is item {i} from the {endpoint} endpoint. " * 3,
            "status": random.choice(["active", "pending", "completed", "archived"]),
            "priority": random.choice(["low", "medium", "high", "critical"]),
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
            "updated_at": (datetime.now() - timedelta(hours=random.randint(1, 168))).isoformat(),
            "owner": {
                "id": random.randint(1, 100),
                "name": f"Owner {random.randint(1, 100)}",
                "email": f"owner{random.randint(1, 100)}@example.com",
            },
            "tags": random.sample(
                ["urgent", "review", "approved", "blocked", "in-progress"], k=random.randint(1, 3)
            ),
            "metadata": {
                "source": random.choice(["web", "api", "mobile", "import"]),
                "version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
            },
        }
        items.append(item)

    return json.dumps(
        {
            "data": items,
            "pagination": {
                "page": 1,
                "per_page": count,
                "total": count * 10,  # Simulate more pages available
                "total_pages": 10,
            },
            "endpoint": endpoint,
        },
        indent=2,
    )


# Tool definitions for LangChain
TOOL_FUNCTIONS = {
    "search_users": lambda query: generate_user_database_results(query, count=100),
    "search_docs": lambda query: generate_search_results(query, count=50),
    "search_logs": lambda service: generate_log_entries(service, count=200),
    "get_metrics": lambda service: generate_metrics_data(service, count=100),
    "fetch_api_data": lambda endpoint: generate_api_response(endpoint, count=75),
}


if __name__ == "__main__":
    # Test output sizes
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")

    print("Tool Output Token Counts:")
    print("=" * 50)

    for name, func in TOOL_FUNCTIONS.items():
        output = func("test")
        tokens = len(enc.encode(output))
        print(f"{name}: {tokens:,} tokens ({len(output):,} chars)")
