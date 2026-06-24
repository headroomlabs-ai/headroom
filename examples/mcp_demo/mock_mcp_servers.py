"""Mock MCP server outputs for demonstration.

These simulate real MCP tool results from servers like:
- Slack search
- Database queries
- GitHub issues
- Log analysis
"""

import json
import random
from datetime import datetime, timedelta


def generate_slack_search_results(query: str, count: int = 150) -> str:
    """Simulate Slack MCP server search results."""
    channels = ["#engineering", "#incidents", "#support", "#general", "#alerts", "#platform"]
    users = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace"]

    messages = []
    for i in range(count):
        # 15% chance of error-related message
        is_error = random.random() < 0.15
        if is_error:
            text = random.choice(
                [
                    "ERROR: Database connection pool exhausted at 3:45am",
                    "CRITICAL: Memory usage at 95% on prod-api-01",
                    "Exception in PaymentService.processTransaction()",
                    "FAILED: Deploy pipeline broke - rolling back",
                    "ALERT: Latency spike detected on /api/users endpoint",
                ]
            )
        else:
            text = random.choice(
                [
                    f"Reviewed the PR for {query}, looks good to merge",
                    f"Updated the docs with new {query} endpoints",
                    "Meeting notes from standup attached",
                    "Can someone review my changes to the auth module?",
                    "Deployed v2.3.1 to staging environment",
                    "Thanks for the feedback on the design doc!",
                    "Working on the feature request from yesterday",
                ]
            )

        messages.append(
            {
                "id": f"msg_{i}",
                "channel": random.choice(channels),
                "user": random.choice(users),
                "text": text,
                "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                "reactions": random.randint(0, 15),
                "thread_replies": random.randint(0, 10),
                "permalink": f"https://slack.com/archives/C123/p{i}",
            }
        )

    return json.dumps(
        {
            "query": query,
            "messages": messages,
            "total": count,
            "has_more": count > 100,
        },
        indent=2,
    )


def generate_database_query_results(query: str, count: int = 200) -> str:
    """Simulate database MCP server query results."""
    rows = []
    for i in range(count):
        # 5% error rate, 10% null rate
        has_error = random.random() < 0.05
        has_null = random.random() < 0.10

        row = {
            "id": i + 1,
            "user_id": f"usr_{random.randint(10000, 99999)}",
            "email": f"user{i}@example.com",
            "full_name": f"User {i}",
            "status": "ERROR: validation_failed"
            if has_error
            else random.choice(["active", "inactive", "pending"]),
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            "last_login": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            "balance": None if has_null else round(random.uniform(0, 10000), 2),
            "subscription_tier": random.choice(["free", "pro", "enterprise"]),
            "metadata": {"source": random.choice(["web", "mobile", "api"]), "version": "2.0"},
        }
        rows.append(row)

    return json.dumps(
        {
            "query": query,
            "rows": rows,
            "count": count,
            "execution_time_ms": random.randint(50, 500),
        },
        indent=2,
    )


def generate_log_search_results(service: str, count: int = 300) -> str:
    """Simulate log analysis MCP server results."""
    services = [service, f"{service}-worker", f"{service}-scheduler", "auth-service"]

    entries = []
    for i in range(count):
        # 20% error rate (ERROR or FATAL)
        if random.random() < 0.20:
            level = random.choice(["ERROR", "FATAL"])
            message = random.choice(
                [
                    "Connection timeout to primary database",
                    "Failed to process message from queue",
                    "Authentication failed: invalid token",
                    "Out of memory error in request handler",
                    "Unhandled exception: NullPointerException",
                    "Circuit breaker open for external-api",
                ]
            )
        else:
            level = random.choice(["DEBUG", "INFO", "INFO", "INFO", "WARN"])
            message = random.choice(
                [
                    "Request processed successfully",
                    "Cache hit for user session",
                    "Starting scheduled job: cleanup",
                    "Connection pool stats: 10/20 active",
                    "Metrics exported to datadog",
                    "Health check passed",
                ]
            )

        entries.append(
            {
                "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                "level": level,
                "service": random.choice(services),
                "message": message,
                "trace_id": f"trace_{random.randint(100000, 999999)}",
                "span_id": f"span_{random.randint(1000, 9999)}",
                "host": f"prod-{random.choice(['api', 'worker', 'web'])}-{random.randint(1, 10):02d}",
            }
        )

    return json.dumps({"entries": entries, "service": service}, indent=2)


def generate_github_issues_results(repo: str, count: int = 100) -> str:
    """Simulate GitHub MCP server issue results."""
    labels_pool = ["enhancement", "documentation", "question", "good first issue", "help wanted"]
    bug_labels = ["bug", "critical", "urgent", "blocker", "security"]

    issues = []
    for i in range(count):
        # 25% bug rate
        is_bug = random.random() < 0.25
        labels = (
            random.sample(bug_labels, k=random.randint(1, 2))
            if is_bug
            else random.sample(labels_pool, k=random.randint(0, 2))
        )

        issues.append(
            {
                "number": i + 1,
                "title": f"{'[BUG] ' if is_bug else ''}{random.choice(['Fix auth flow', 'Add dark mode', 'Update docs', 'Improve perf'])}",
                "state": random.choice(["open", "open", "closed"]),
                "labels": labels,
                "author": f"contributor{random.randint(1, 50)}",
                "assignee": f"maintainer{random.randint(1, 5)}" if random.random() > 0.3 else None,
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
                "updated_at": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                "comments": random.randint(0, 30),
                "body": "Lorem ipsum dolor sit amet..." if random.random() > 0.5 else "",
                "milestone": f"v{random.randint(1, 3)}.{random.randint(0, 9)}"
                if random.random() > 0.7
                else None,
            }
        )

    return json.dumps(
        {
            "repository": repo,
            "issues": issues,
            "total_count": count,
            "open_count": sum(1 for i in issues if i["state"] == "open"),
        },
        indent=2,
    )
