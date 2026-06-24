"""Demonstrate Headroom MCP compression on real-world tool outputs.

Run with:
    PYTHONPATH=. python -m examples.mcp_demo.show_compression
"""

import random

from headroom.integrations.mcp import (
    compress_tool_result_with_metrics,
)
from headroom.providers import OpenAIProvider

from .mock_mcp_servers import (
    generate_database_query_results,
    generate_github_issues_results,
    generate_log_search_results,
    generate_slack_search_results,
)


def main():
    random.seed(42)

    print("\n" + "=" * 70)
    print("HEADROOM MCP INTEGRATION - COMPRESSION DEMO")
    print("=" * 70)

    # Get token counter
    provider = OpenAIProvider()
    provider.get_token_counter("gpt-4o")

    # Test scenarios
    scenarios = [
        {
            "name": "Slack Search",
            "tool_name": "mcp__slack__search_messages",
            "tool_args": {"query": "production errors", "limit": 150},
            "user_query": "find production errors from last week",
            "content": generate_slack_search_results("production errors", count=150),
        },
        {
            "name": "Database Query",
            "tool_name": "mcp__database__query",
            "tool_args": {"sql": "SELECT * FROM users WHERE status != 'active'"},
            "user_query": "find users with issues",
            "content": generate_database_query_results("users", count=200),
        },
        {
            "name": "Log Analysis",
            "tool_name": "mcp__logs__search",
            "tool_args": {"service": "api-gateway", "level": "ERROR"},
            "user_query": "find errors in api-gateway",
            "content": generate_log_search_results("api-gateway", count=300),
        },
        {
            "name": "GitHub Issues",
            "tool_name": "mcp__github__list_issues",
            "tool_args": {"repo": "myorg/myrepo", "state": "open"},
            "user_query": "find open bugs",
            "content": generate_github_issues_results("myorg/myrepo", count=100),
        },
    ]

    total_before = 0
    total_after = 0

    for scenario in scenarios:
        print(f"\n{'─' * 70}")
        print(f"Tool: {scenario['name']}")
        print(f"MCP Server: {scenario['tool_name']}")
        print(f'User Query: "{scenario["user_query"]}"')
        print(f"{'─' * 70}")

        result = compress_tool_result_with_metrics(
            content=scenario["content"],
            tool_name=scenario["tool_name"],
            tool_args=scenario["tool_args"],
            user_query=scenario["user_query"],
        )

        print(f"\n  Original tokens:   {result.original_tokens:>8,}")
        print(f"  Compressed tokens: {result.compressed_tokens:>8,}")
        print(f"  Tokens saved:      {result.tokens_saved:>8,} ({result.compression_ratio:.1%})")

        if result.items_before and result.items_after:
            print(f"\n  Items before:      {result.items_before:>8}")
            print(f"  Items after:       {result.items_after:>8}")
            print(f"  Errors preserved:  {result.errors_preserved:>8}")

        total_before += result.original_tokens
        total_after += result.compressed_tokens

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Total original tokens:   {total_before:>8,}")
    print(f"  Total compressed tokens: {total_after:>8,}")
    print(f"  Total tokens saved:      {total_before - total_after:>8,}")
    print(f"  Overall compression:     {(total_before - total_after) / total_before:.1%}")

    # Cost savings at GPT-4o rates ($2.50/1M input)
    cost_before = total_before * 2.50 / 1_000_000
    cost_after = total_after * 2.50 / 1_000_000
    print(f"\n  Cost before (GPT-4o):    ${cost_before:.4f}")
    print(f"  Cost after (GPT-4o):     ${cost_after:.4f}")
    print(f"  Cost saved per request:  ${cost_before - cost_after:.4f}")

    # At scale
    daily_requests = 1000
    monthly_savings = (cost_before - cost_after) * daily_requests * 30
    print(f"\n  At {daily_requests:,} requests/day:")
    print(f"  Monthly savings:         ${monthly_savings:,.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
