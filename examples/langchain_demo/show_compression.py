"""Demonstrate Headroom compression on LangChain tool outputs.

This script shows EXACTLY what Headroom does to large tool outputs:
- Before: Full 100-item JSON array
- After: Compressed to ~20 relevant items

No API key required - runs locally.

Run:
    python -m examples.langchain_demo.show_compression
"""

import json
import sys

try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken required. Run: uv pip install tiktoken")
    sys.exit(1)

from headroom.providers import OpenAIProvider
from headroom.transforms import SmartCrusher

from .mock_tools import TOOL_FUNCTIONS

ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens."""
    return len(ENCODER.encode(text))


def demonstrate_compression(tool_name: str, tool_arg: str, context: str):
    """Show before/after compression for a tool output."""

    print(f"\n{'=' * 70}")
    print(f"TOOL: {tool_name}({tool_arg!r})")
    print(f"CONTEXT: {context!r}")
    print(f"{'=' * 70}")

    # Generate tool output
    raw_output = TOOL_FUNCTIONS[tool_name](tool_arg)
    raw_tokens = count_tokens(raw_output)

    # Parse to count items
    data = json.loads(raw_output)
    if "results" in data:
        item_count = len(data["results"])
    elif "entries" in data:
        item_count = len(data["entries"])
    elif "metrics" in data:
        item_count = len(data["metrics"])
    elif "data" in data:
        item_count = len(data["data"])
    else:
        item_count = "?"

    print("\n--- BEFORE COMPRESSION ---")
    print(f"Items: {item_count}")
    print(f"Tokens: {raw_tokens:,}")
    print(f"Chars: {len(raw_output):,}")
    print("\nFirst 500 chars:")
    print(raw_output[:500] + "...")

    # Create SmartCrusher with context
    from headroom.config import SmartCrusherConfig

    smart_config = SmartCrusherConfig(
        enabled=True,
        min_tokens_to_crush=200,
        max_items_after_crush=20,
    )

    provider = OpenAIProvider()
    tokenizer = provider.get_token_counter("gpt-4o")

    crusher = SmartCrusher(config=smart_config)

    # Build messages with tool output (simulating agent conversation)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": context},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps({tool_name.split("_")[-1]: tool_arg}),
                    },
                }
            ],
        },
        {"role": "tool", "content": raw_output, "tool_call_id": "call_1"},
    ]

    # Apply SmartCrusher (tokenizer is passed to apply())
    result = crusher.apply(messages, tokenizer=tokenizer)
    compressed_messages = result.messages

    # Get compressed output
    compressed_output = compressed_messages[-1]["content"]
    compressed_tokens = count_tokens(compressed_output)

    # Parse compressed to count items
    try:
        compressed_data = json.loads(compressed_output)
        if "results" in compressed_data:
            compressed_items = len(compressed_data["results"])
        elif "entries" in compressed_data:
            compressed_items = len(compressed_data["entries"])
        elif "metrics" in compressed_data:
            compressed_items = len(compressed_data["metrics"])
        elif "data" in compressed_data:
            compressed_items = len(compressed_data["data"])
        else:
            compressed_items = "?"
    except json.JSONDecodeError:
        compressed_items = "N/A"

    print("\n--- AFTER COMPRESSION ---")
    print(f"Items: {compressed_items}")
    print(f"Tokens: {compressed_tokens:,}")
    print(f"Chars: {len(compressed_output):,}")
    print("\nFirst 500 chars:")
    print(compressed_output[:500] + "...")

    # Calculate savings
    tokens_saved = raw_tokens - compressed_tokens
    pct_saved = (tokens_saved / raw_tokens * 100) if raw_tokens > 0 else 0

    print("\n--- SAVINGS ---")
    print(f"Tokens saved: {tokens_saved:,} ({pct_saved:.1f}%)")
    print(f"Items reduced: {item_count} -> {compressed_items}")

    return {
        "tool": tool_name,
        "before_tokens": raw_tokens,
        "after_tokens": compressed_tokens,
        "saved_tokens": tokens_saved,
        "saved_pct": pct_saved,
    }


def main():
    """Run compression demonstrations."""

    print("\n" + "=" * 70)
    print("HEADROOM SMARTCRUSHER: BEFORE/AFTER COMPRESSION")
    print("=" * 70)
    print("""
This demonstrates how Headroom's SmartCrusher compresses large tool outputs.

Key techniques:
1. Pattern detection (logs, time-series, search results)
2. Keep first/last items for context
3. Keep ERROR/anomaly items (important!)
4. Keep items matching the user's query (relevance scoring)
5. Statistical sampling for remaining slots
""")

    results = []

    # Demo 1: User database search
    results.append(
        demonstrate_compression(
            tool_name="search_users",
            tool_arg="Engineering users",
            context="Find all users in the Engineering department who are currently active",
        )
    )

    # Demo 2: Log search with errors
    results.append(
        demonstrate_compression(
            tool_name="search_logs",
            tool_arg="payment-service",
            context="Check the payment-service logs for any ERROR entries",
        )
    )

    # Demo 3: Metrics with anomalies
    results.append(
        demonstrate_compression(
            tool_name="get_metrics",
            tool_arg="api-gateway",
            context="Look for any CPU spikes or high error rates in the api-gateway metrics",
        )
    )

    # Demo 4: Documentation search
    results.append(
        demonstrate_compression(
            tool_name="search_docs",
            tool_arg="authentication",
            context="Find documentation about authentication troubleshooting",
        )
    )

    # Demo 5: API data
    results.append(
        demonstrate_compression(
            tool_name="fetch_api_data",
            tool_arg="orders",
            context="Get recent orders with status 'pending'",
        )
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: TOKEN SAVINGS ACROSS ALL TOOLS")
    print("=" * 70)

    print(f"\n{'Tool':<20} {'Before':>12} {'After':>12} {'Saved':>12} {'%':>8}")
    print("-" * 66)

    total_before = 0
    total_after = 0

    for r in results:
        print(
            f"{r['tool']:<20} {r['before_tokens']:>12,} {r['after_tokens']:>12,} {r['saved_tokens']:>12,} {r['saved_pct']:>7.1f}%"
        )
        total_before += r["before_tokens"]
        total_after += r["after_tokens"]

    total_saved = total_before - total_after
    total_pct = (total_saved / total_before * 100) if total_before > 0 else 0

    print("-" * 66)
    print(
        f"{'TOTAL':<20} {total_before:>12,} {total_after:>12,} {total_saved:>12,} {total_pct:>7.1f}%"
    )

    # Cost savings
    input_cost_per_1m = 2.50  # gpt-4o pricing
    cost_before = total_before * input_cost_per_1m / 1_000_000
    cost_after = total_after * input_cost_per_1m / 1_000_000
    cost_saved = cost_before - cost_after

    print("\n--- COST IMPACT (at gpt-4o $2.50/1M input tokens) ---")
    print(f"Before: ${cost_before:.4f}")
    print(f"After:  ${cost_after:.4f}")
    print(f"Saved:  ${cost_saved:.4f} per request")
    print(
        f"\nAt 1000 requests/day: ${cost_saved * 1000:.2f}/day = ${cost_saved * 1000 * 30:.2f}/month"
    )


if __name__ == "__main__":
    main()
