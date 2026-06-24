"""Real-World MCP Agent Evaluation.

This eval simulates an agent with multiple MCP tools and tests whether
Headroom compression preserves the information needed to answer correctly.

Run with:
    PYTHONPATH=. python -m examples.mcp_demo.run_agent_eval

Requires: OPENAI_API_KEY environment variable
"""

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from openai import OpenAI

from headroom.integrations.mcp import compress_tool_result_with_metrics
from headroom.providers import OpenAIProvider

# ============================================================================
# Test Data Generators (Deterministic for eval reproducibility)
# ============================================================================


def generate_slack_with_specific_errors(seed: int = 42) -> tuple[str, list[dict]]:
    """Generate Slack messages with SPECIFIC errors we'll query for."""
    random.seed(seed)

    # These are the "needle" errors we'll ask the agent to find
    critical_errors = [
        {
            "id": "msg_17",
            "channel": "#incidents",
            "user": "alice",
            "text": "CRITICAL: Payment service is DOWN - customers cannot checkout. Error: ConnectionRefused to payment-db-01",
            "timestamp": "2025-01-06T03:45:00Z",
        },
        {
            "id": "msg_42",
            "channel": "#alerts",
            "user": "bob",
            "text": "ERROR: Auth service returning 500s. Stack trace shows NullPointerException in TokenValidator.java:127",
            "timestamp": "2025-01-06T02:30:00Z",
        },
        {
            "id": "msg_89",
            "channel": "#engineering",
            "user": "charlie",
            "text": "FAILED: Deploy to prod-us-east failed. Reason: Health check timeout after 300s on api-gateway-03",
            "timestamp": "2025-01-05T23:15:00Z",
        },
    ]

    # Generate noise messages
    channels = ["#engineering", "#incidents", "#support", "#general", "#alerts"]
    users = ["alice", "bob", "charlie", "diana", "eve", "frank"]
    noise_messages = [
        "Reviewed the PR, looks good to merge",
        "Updated the docs with new API endpoints",
        "Meeting notes from standup attached",
        "Thanks for the code review feedback!",
        "Deployed v2.3.1 to staging - all tests passing",
        "Working on the feature request from yesterday",
        "Can someone review my changes to the auth module?",
        "Just finished the database migration script",
    ]

    messages = []
    error_idx = 0
    for i in range(150):
        if i in [17, 42, 89]:  # Insert critical errors at specific positions
            messages.append(critical_errors[error_idx])
            error_idx += 1
        else:
            messages.append(
                {
                    "id": f"msg_{i}",
                    "channel": random.choice(channels),
                    "user": random.choice(users),
                    "text": random.choice(noise_messages),
                    "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                }
            )

    return json.dumps({"messages": messages, "total": 150}), critical_errors


def generate_logs_with_specific_errors(seed: int = 43) -> tuple[str, list[dict]]:
    """Generate log entries with SPECIFIC errors we'll query for."""
    random.seed(seed)

    # These are the "needle" errors
    critical_logs = [
        {
            "timestamp": "2025-01-06T03:44:58Z",
            "level": "FATAL",
            "service": "payment-service",
            "message": "Cannot connect to payment-db-01: Connection refused",
            "trace_id": "trace_payment_001",
        },
        {
            "timestamp": "2025-01-06T02:29:55Z",
            "level": "ERROR",
            "service": "auth-service",
            "message": "NullPointerException in TokenValidator.validate() at line 127",
            "trace_id": "trace_auth_001",
        },
        {
            "timestamp": "2025-01-05T23:14:30Z",
            "level": "ERROR",
            "service": "api-gateway",
            "message": "Health check failed: timeout after 300000ms",
            "trace_id": "trace_gateway_001",
        },
        {
            "timestamp": "2025-01-06T01:00:00Z",
            "level": "ERROR",
            "service": "user-service",
            "message": "Database query timeout: SELECT * FROM users WHERE last_login > ?",
            "trace_id": "trace_user_001",
        },
    ]

    services = [
        "api-gateway",
        "auth-service",
        "payment-service",
        "user-service",
        "notification-service",
    ]
    info_messages = [
        "Request processed successfully",
        "Cache hit for user session",
        "Health check passed",
        "Connection pool stats: 10/20 active",
        "Metrics exported to datadog",
    ]

    entries = []
    error_idx = 0
    for i in range(300):
        if i in [15, 45, 120, 200]:  # Insert critical errors
            entries.append(critical_logs[error_idx])
            error_idx += 1
        else:
            entries.append(
                {
                    "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                    "level": random.choice(["DEBUG", "INFO", "INFO", "INFO", "WARN"]),
                    "service": random.choice(services),
                    "message": random.choice(info_messages),
                    "trace_id": f"trace_{random.randint(100000, 999999)}",
                }
            )

    return json.dumps({"entries": entries}), critical_logs


def generate_database_with_anomalies(seed: int = 44) -> tuple[str, list[dict]]:
    """Generate database results with SPECIFIC anomalies."""
    random.seed(seed)

    # Anomalous records we'll ask about
    anomalies = [
        {
            "id": 23,
            "user_id": "usr_99999",
            "email": "admin@internal.com",
            "status": "ERROR: account_locked",
            "balance": 999999.99,
            "login_attempts": 47,
            "last_login": "2025-01-06T04:00:00Z",
        },
        {
            "id": 156,
            "user_id": "usr_00001",
            "email": "test@test.com",
            "status": "ERROR: validation_failed",
            "balance": -500.00,
            "login_attempts": 0,
            "last_login": None,
        },
    ]

    rows = []
    anomaly_idx = 0
    for i in range(200):
        if i in [23, 156]:
            rows.append(anomalies[anomaly_idx])
            anomaly_idx += 1
        else:
            rows.append(
                {
                    "id": i,
                    "user_id": f"usr_{random.randint(10000, 99999)}",
                    "email": f"user{i}@example.com",
                    "status": random.choice(["active", "active", "active", "inactive", "pending"]),
                    "balance": round(random.uniform(0, 5000), 2),
                    "login_attempts": random.randint(0, 5),
                    "last_login": (
                        datetime.now() - timedelta(days=random.randint(0, 30))
                    ).isoformat(),
                }
            )

    return json.dumps({"rows": rows, "count": 200}), anomalies


# ============================================================================
# Eval Test Cases
# ============================================================================


@dataclass
class EvalCase:
    """A single evaluation case."""

    name: str
    tool_name: str
    tool_output: str
    user_query: str
    expected_findings: list[str]  # Substrings that MUST appear in answer
    critical_data: list[dict]  # The actual critical records


def create_eval_cases() -> list[EvalCase]:
    """Create evaluation test cases."""

    slack_output, slack_errors = generate_slack_with_specific_errors()
    logs_output, log_errors = generate_logs_with_specific_errors()
    db_output, db_anomalies = generate_database_with_anomalies()

    return [
        EvalCase(
            name="Slack: Find Payment Outage",
            tool_name="mcp__slack__search",
            tool_output=slack_output,
            user_query="What's causing the payment issues? Find any errors related to payments or checkout.",
            expected_findings=["payment", "DOWN", "ConnectionRefused", "payment-db-01"],
            critical_data=slack_errors,
        ),
        EvalCase(
            name="Slack: Find Auth Errors",
            tool_name="mcp__slack__search",
            tool_output=slack_output,
            user_query="Are there any authentication or auth service errors?",
            expected_findings=["Auth service", "500", "NullPointerException", "TokenValidator"],
            critical_data=slack_errors,
        ),
        EvalCase(
            name="Logs: Find All Errors",
            tool_name="mcp__logs__search",
            tool_output=logs_output,
            user_query="List all ERROR and FATAL log entries with their services and messages.",
            expected_findings=[
                "payment-service",
                "auth-service",
                "api-gateway",
                "Connection refused",
                "NullPointerException",
            ],
            critical_data=log_errors,
        ),
        EvalCase(
            name="Logs: Find Database Issues",
            tool_name="mcp__logs__search",
            tool_output=logs_output,
            user_query="Are there any database connection or query issues in the logs?",
            expected_findings=["Database", "timeout", "Connection refused"],
            critical_data=log_errors,
        ),
        EvalCase(
            name="Database: Find Anomalous Accounts",
            tool_name="mcp__database__query",
            tool_output=db_output,
            user_query="Find any suspicious or anomalous user accounts - unusual balances, error statuses, or high login attempts.",
            expected_findings=["account_locked", "999999", "47", "negative", "-500"],
            critical_data=db_anomalies,
        ),
    ]


# ============================================================================
# Agent Simulation
# ============================================================================


def run_agent_with_tool_output(
    client: OpenAI,
    user_query: str,
    tool_name: str,
    tool_output: str,
    model: str = "gpt-4o-mini",
) -> tuple[str, int]:
    """Simulate agent receiving tool output and answering query.

    Returns: (answer, tokens_used)
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant analyzing tool outputs. Be specific and cite exact details from the data.",
        },
        {"role": "user", "content": user_query},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": tool_output, "tool_call_id": "call_1"},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000,
    )

    return response.choices[0].message.content, response.usage.total_tokens


def evaluate_answer(answer: str, expected_findings: list[str]) -> tuple[int, int, list[str]]:
    """Check if answer contains expected findings.

    Returns: (found_count, total_expected, missing_findings)
    """
    answer_lower = answer.lower()
    found = 0
    missing = []

    for finding in expected_findings:
        if finding.lower() in answer_lower:
            found += 1
        else:
            missing.append(finding)

    return found, len(expected_findings), missing


# ============================================================================
# Main Eval Runner
# ============================================================================


def main():
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n" + "=" * 70)
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("=" * 70)
        print("\nTo run this eval, set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nThen run:")
        print("  PYTHONPATH=. python -m examples.mcp_demo.run_agent_eval")
        return

    client = OpenAI()
    provider = OpenAIProvider()
    tokenizer = provider.get_token_counter("gpt-4o")

    print("\n" + "=" * 70)
    print("MCP AGENT EVALUATION: BEFORE vs AFTER HEADROOM COMPRESSION")
    print("=" * 70)
    print("\nThis eval tests whether an agent can still find critical information")
    print("after Headroom compresses large MCP tool outputs.")
    print("\nModel: gpt-4o-mini")

    eval_cases = create_eval_cases()

    results = []

    for case in eval_cases:
        print(f"\n{'─' * 70}")
        print(f"EVAL: {case.name}")
        print(f'Query: "{case.user_query}"')
        print(f"{'─' * 70}")

        # Measure original tokens
        original_tokens = tokenizer.count_text(case.tool_output)

        # Compress with Headroom
        compression = compress_tool_result_with_metrics(
            content=case.tool_output,
            tool_name=case.tool_name,
            user_query=case.user_query,
        )

        print("\n  Tool Output:")
        print(f"    Original:   {original_tokens:,} tokens")
        print(f"    Compressed: {compression.compressed_tokens:,} tokens")
        print(f"    Saved:      {compression.tokens_saved:,} ({compression.compression_ratio:.1%})")

        # Run agent BEFORE (with original output)
        print("\n  Running agent with ORIGINAL output...")
        try:
            answer_before, tokens_before = run_agent_with_tool_output(
                client, case.user_query, case.tool_name, case.tool_output
            )
            found_before, total, missing_before = evaluate_answer(
                answer_before, case.expected_findings
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            answer_before = ""
            found_before, total, missing_before = (
                0,
                len(case.expected_findings),
                case.expected_findings,
            )
            tokens_before = 0

        # Run agent AFTER (with compressed output)
        print("  Running agent with COMPRESSED output...")
        try:
            answer_after, tokens_after = run_agent_with_tool_output(
                client, case.user_query, case.tool_name, compression.compressed_content
            )
            found_after, _, missing_after = evaluate_answer(answer_after, case.expected_findings)
        except Exception as e:
            print(f"    ERROR: {e}")
            answer_after = ""
            found_after, missing_after = 0, case.expected_findings
            tokens_after = 0

        # Results
        print("\n  Results:")
        print(f"    BEFORE: Found {found_before}/{total} expected findings")
        if missing_before:
            print(f"            Missing: {missing_before}")
        print(f"    AFTER:  Found {found_after}/{total} expected findings")
        if missing_after:
            print(f"            Missing: {missing_after}")

        # Token usage comparison
        print("\n  API Token Usage:")
        print(f"    BEFORE: {tokens_before:,} tokens")
        print(f"    AFTER:  {tokens_after:,} tokens")
        if tokens_before > 0:
            print(
                f"    Saved:  {tokens_before - tokens_after:,} ({(tokens_before - tokens_after) / tokens_before:.1%})"
            )

        # Pass/Fail
        passed = found_after >= found_before
        status = "PASS" if passed else "FAIL"
        print(f"\n  Status: {status}")
        if not passed:
            print("    Reason: Compressed output lost information")
            print(f"    Lost findings: {set(missing_after) - set(missing_before)}")

        results.append(
            {
                "name": case.name,
                "passed": passed,
                "found_before": found_before,
                "found_after": found_after,
                "total": total,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "compression_ratio": compression.compression_ratio,
            }
        )

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r["passed"])
    total_cases = len(results)

    print(f"\n  Tests Passed: {passed}/{total_cases}")
    print("\n  Detailed Results:")
    print(f"  {'Test Name':<35} {'Before':<10} {'After':<10} {'Compress':<10} {'Status':<8}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['name']:<35} {r['found_before']}/{r['total']:<8} {r['found_after']}/{r['total']:<8} {r['compression_ratio']:.0%}{'':>6} {status:<8}"
        )

    # Token savings
    total_tokens_before = sum(r["tokens_before"] for r in results)
    total_tokens_after = sum(r["tokens_after"] for r in results)

    print("\n  Total API Tokens:")
    print(f"    Before: {total_tokens_before:,}")
    print(f"    After:  {total_tokens_after:,}")
    print(
        f"    Saved:  {total_tokens_before - total_tokens_after:,} ({(total_tokens_before - total_tokens_after) / total_tokens_before:.1%})"
    )

    # Cost estimate
    cost_before = total_tokens_before * 0.15 / 1_000_000  # gpt-4o-mini input
    cost_after = total_tokens_after * 0.15 / 1_000_000
    print("\n  Cost (gpt-4o-mini):")
    print(f"    Before: ${cost_before:.4f}")
    print(f"    After:  ${cost_after:.4f}")
    print(f"    Saved:  ${cost_before - cost_after:.4f}")

    print("\n" + "=" * 70)

    if passed == total_cases:
        print("SUCCESS: All tests passed - Headroom compression preserves critical info!")
    else:
        print(f"WARNING: {total_cases - passed} tests failed - some information was lost")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
