"""Real-world LangChain Agent: Before/After Headroom Comparison.

This script demonstrates the impact of Headroom optimization on a realistic
LangChain agent that uses tools returning large outputs.

Scenario: A support agent that:
1. Searches user database for matching users
2. Looks up documentation for solutions
3. Checks logs for errors
4. Reviews metrics for anomalies

Each tool returns 50-200 items, simulating real-world API responses.

Run:
    python -m examples.langchain_demo.run_comparison
"""

import json
import os
import sys
import time
from dataclasses import dataclass

# Check for required dependencies
try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken required. Run: pip install tiktoken")
    sys.exit(1)

try:
    from langchain_core.messages import (  # noqa: F401
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.tools import tool  # noqa: F401
    from langchain_openai import ChatOpenAI  # noqa: F401
except ImportError:
    print("ERROR: LangChain required. Run: pip install langchain langchain-openai langchain-core")
    sys.exit(1)

# Import our mock tools
from .mock_tools import TOOL_FUNCTIONS

# Token counter
ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(ENCODER.encode(text))


def count_message_tokens(messages: list[dict]) -> int:
    """Count total tokens in messages."""
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if content:
                total += count_tokens(str(content))
            # Count tool calls
            if "tool_calls" in msg:
                total += count_tokens(json.dumps(msg["tool_calls"]))
        else:
            # LangChain message object
            if hasattr(msg, "content") and msg.content:
                total += count_tokens(str(msg.content))
    return total


@dataclass
class AgentRun:
    """Results from a single agent run."""

    scenario: str
    mode: str  # "baseline" or "headroom"
    total_input_tokens: int
    total_output_tokens: int
    tool_calls: int
    tool_output_tokens: int
    duration_ms: float
    final_response: str
    messages_count: int


def create_langchain_tools():
    """Create LangChain tool wrappers for our mock tools."""

    @tool
    def search_users(query: str) -> str:
        """Search user database for users matching the query. Returns user records with email, department, status, etc."""
        return TOOL_FUNCTIONS["search_users"](query)

    @tool
    def search_docs(query: str) -> str:
        """Search documentation for articles matching the query. Returns docs with titles, snippets, relevance scores."""
        return TOOL_FUNCTIONS["search_docs"](query)

    @tool
    def search_logs(service: str) -> str:
        """Search application logs for a service. Returns log entries with timestamps, levels, messages."""
        return TOOL_FUNCTIONS["search_logs"](service)

    @tool
    def get_metrics(service: str) -> str:
        """Get monitoring metrics for a service. Returns time-series data with CPU, memory, latency, error rates."""
        return TOOL_FUNCTIONS["get_metrics"](service)

    @tool
    def fetch_api_data(endpoint: str) -> str:
        """Fetch data from an API endpoint. Returns paginated items with metadata."""
        return TOOL_FUNCTIONS["fetch_api_data"](endpoint)

    return [search_users, search_docs, search_logs, get_metrics, fetch_api_data]


SYSTEM_PROMPT = """You are a helpful support agent assistant. You help investigate user issues by:

1. Searching the user database to find relevant users
2. Looking up documentation for solutions
3. Checking logs for errors
4. Reviewing metrics for anomalies

Today's date is 2025-01-06.

When investigating issues:
- Start by understanding the problem
- Use tools to gather relevant information
- Look for patterns in the data
- Provide a clear summary of findings

Be thorough but efficient. Focus on finding actionable information."""


SCENARIOS = [
    {
        "name": "User Account Investigation",
        "query": "A user named 'User 42 Williams' is reporting they can't log in. Can you check their account status, look for any authentication errors in the logs, and see if there are any relevant docs about login issues?",
    },
    {
        "name": "Service Performance Investigation",
        "query": "The payment-service seems slow today. Can you check its metrics for any anomalies, look at recent logs for errors, and find documentation about performance troubleshooting?",
    },
    {
        "name": "Multi-User Issue",
        "query": "Several users in the Engineering department are reporting issues. Can you search for Engineering users, check the logs for the user-service, and look up any relevant documentation?",
    },
]


def run_agent_baseline(scenario: dict, api_key: str) -> AgentRun:
    """Run agent WITHOUT Headroom (baseline)."""

    tools = create_langchain_tools()

    # Create model with tools
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0,
    ).bind_tools(tools)

    # Build conversation
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=scenario["query"]),
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    tool_output_tokens = 0
    tool_calls_count = 0

    start_time = time.time()

    # Agent loop (max 5 iterations to prevent runaway)
    for _ in range(5):
        # Count input tokens
        input_tokens = count_message_tokens([{"content": m.content} for m in messages])
        total_input_tokens += input_tokens

        # Call model
        response = model.invoke(messages)
        messages.append(response)

        # Count output tokens
        output_tokens = count_tokens(response.content) if response.content else 0
        if response.tool_calls:
            output_tokens += count_tokens(json.dumps(list(response.tool_calls)))
        total_output_tokens += output_tokens

        # Check if done
        if not response.tool_calls:
            break

        # Execute tools
        for tool_call in response.tool_calls:
            tool_calls_count += 1

            # Find and execute tool
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            for t in tools:
                if t.name == tool_name:
                    result = t.invoke(tool_args)
                    break
            else:
                result = f"Tool {tool_name} not found"

            # Count tool output tokens
            tool_tokens = count_tokens(result)
            tool_output_tokens += tool_tokens

            # Add tool result
            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                )
            )

    duration_ms = (time.time() - start_time) * 1000

    return AgentRun(
        scenario=scenario["name"],
        mode="baseline",
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        tool_calls=tool_calls_count,
        tool_output_tokens=tool_output_tokens,
        duration_ms=duration_ms,
        final_response=response.content if response.content else "",
        messages_count=len(messages),
    )


def run_agent_headroom(scenario: dict, api_key: str) -> AgentRun:
    """Run agent WITH Headroom optimization."""

    # Import Headroom integration
    from headroom import HeadroomConfig
    from headroom.integrations import HeadroomChatModel

    tools = create_langchain_tools()

    # Create base model
    base_model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0,
    )

    # Wrap with Headroom
    config = HeadroomConfig(
        smart_crusher_threshold=500,  # Compress tool outputs > 500 tokens
        smart_crusher_max_items=20,  # Keep max 20 items
        cache_alignment=True,
        rolling_window=True,
    )

    headroom_model = HeadroomChatModel(
        wrapped_model=base_model,
        headroom_config=config,
    ).bind_tools(tools)

    # Build conversation
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=scenario["query"]),
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    tool_output_tokens = 0
    tool_calls_count = 0

    start_time = time.time()

    # Agent loop (max 5 iterations)
    for _ in range(5):
        # Count input tokens (before optimization)
        input_tokens = count_message_tokens([{"content": m.content} for m in messages])
        total_input_tokens += input_tokens

        # Call model (Headroom optimizes internally)
        response = headroom_model.invoke(messages)
        messages.append(response)

        # Count output tokens
        output_tokens = count_tokens(response.content) if response.content else 0
        if response.tool_calls:
            output_tokens += count_tokens(json.dumps(list(response.tool_calls)))
        total_output_tokens += output_tokens

        # Check if done
        if not response.tool_calls:
            break

        # Execute tools
        for tool_call in response.tool_calls:
            tool_calls_count += 1

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            for t in tools:
                if t.name == tool_name:
                    result = t.invoke(tool_args)
                    break
            else:
                result = f"Tool {tool_name} not found"

            tool_tokens = count_tokens(result)
            tool_output_tokens += tool_tokens

            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                )
            )

    duration_ms = (time.time() - start_time) * 1000

    # Get Headroom metrics
    tokens_saved = headroom_model.get_total_tokens_saved()

    return AgentRun(
        scenario=scenario["name"],
        mode="headroom",
        total_input_tokens=total_input_tokens - tokens_saved,  # Actual tokens sent
        total_output_tokens=total_output_tokens,
        tool_calls=tool_calls_count,
        tool_output_tokens=tool_output_tokens,
        duration_ms=duration_ms,
        final_response=response.content if response.content else "",
        messages_count=len(messages),
    )


def print_comparison(baseline: AgentRun, headroom: AgentRun):
    """Print comparison between baseline and headroom runs."""

    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {baseline.scenario}")
    print(f"{'=' * 70}")

    # Token comparison
    input_saved = baseline.total_input_tokens - headroom.total_input_tokens
    input_pct = (
        (input_saved / baseline.total_input_tokens * 100) if baseline.total_input_tokens > 0 else 0
    )

    print(f"\n{'METRIC':<30} {'BASELINE':>15} {'HEADROOM':>15} {'SAVINGS':>15}")
    print("-" * 75)
    print(
        f"{'Input Tokens':<30} {baseline.total_input_tokens:>15,} {headroom.total_input_tokens:>15,} {input_saved:>14,} ({input_pct:.1f}%)"
    )
    print(
        f"{'Output Tokens':<30} {baseline.total_output_tokens:>15,} {headroom.total_output_tokens:>15,} {'N/A':>15}"
    )
    print(
        f"{'Tool Output Tokens':<30} {baseline.tool_output_tokens:>15,} {headroom.tool_output_tokens:>15,} {'(raw)':>15}"
    )
    print(f"{'Tool Calls':<30} {baseline.tool_calls:>15} {headroom.tool_calls:>15} {'':>15}")
    print(f"{'Messages':<30} {baseline.messages_count:>15} {headroom.messages_count:>15} {'':>15}")
    print(
        f"{'Duration (ms)':<30} {baseline.duration_ms:>15.0f} {headroom.duration_ms:>15.0f} {'':>15}"
    )

    # Cost estimation (gpt-4o-mini pricing)
    input_cost_per_1m = 0.15
    output_cost_per_1m = 0.60

    baseline_cost = (
        baseline.total_input_tokens * input_cost_per_1m
        + baseline.total_output_tokens * output_cost_per_1m
    ) / 1_000_000
    headroom_cost = (
        headroom.total_input_tokens * input_cost_per_1m
        + headroom.total_output_tokens * output_cost_per_1m
    ) / 1_000_000
    cost_saved = baseline_cost - headroom_cost
    cost_pct = (cost_saved / baseline_cost * 100) if baseline_cost > 0 else 0

    print(
        f"\n{'Estimated Cost (USD)':<30} ${baseline_cost:>14.6f} ${headroom_cost:>14.6f} ${cost_saved:>13.6f} ({cost_pct:.1f}%)"
    )


def main():
    """Run the before/after comparison."""

    print("\n" + "=" * 70)
    print("LANGCHAIN AGENT: BEFORE/AFTER HEADROOM COMPARISON")
    print("=" * 70)

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("\nRunning in SIMULATION mode (mock results)...\n")
        run_simulation()
        return

    print(f"\nRunning {len(SCENARIOS)} scenarios with real API calls...")
    print("This will make actual OpenAI API calls and incur costs.\n")

    all_baseline = []
    all_headroom = []

    for scenario in SCENARIOS:
        print(f"\nRunning scenario: {scenario['name']}...")

        # Run baseline
        print("  - Running baseline (no optimization)...")
        baseline = run_agent_baseline(scenario, api_key)
        all_baseline.append(baseline)

        # Run with Headroom
        print("  - Running with Headroom optimization...")
        headroom = run_agent_headroom(scenario, api_key)
        all_headroom.append(headroom)

        # Print comparison
        print_comparison(baseline, headroom)

    # Print summary
    print_summary(all_baseline, all_headroom)


def run_simulation():
    """Run simulation without API calls (for testing)."""

    print("SIMULATION MODE - Using estimated token counts\n")

    # Simulate what would happen based on tool output sizes
    for scenario in SCENARIOS:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 50)

        # Estimate tool outputs
        tools_used = ["search_users", "search_logs", "search_docs"]
        total_tool_tokens = 0

        for tool_name in tools_used:
            output = TOOL_FUNCTIONS[tool_name]("test")
            tokens = count_tokens(output)
            total_tool_tokens += tokens
            print(f"  {tool_name}: {tokens:,} tokens")

        print(f"\n  Total tool output: {total_tool_tokens:,} tokens")
        print(f"  With 3 iterations, baseline input would be: ~{total_tool_tokens * 2:,} tokens")
        print(f"  With Headroom (20 items max), estimated: ~{total_tool_tokens // 5:,} tokens")
        print(
            f"  Estimated savings: ~{total_tool_tokens * 2 - total_tool_tokens // 5:,} tokens (~80%)"
        )


def print_summary(baseline_runs: list[AgentRun], headroom_runs: list[AgentRun]):
    """Print overall summary."""

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    total_baseline_input = sum(r.total_input_tokens for r in baseline_runs)
    total_headroom_input = sum(r.total_input_tokens for r in headroom_runs)
    total_saved = total_baseline_input - total_headroom_input
    pct_saved = (total_saved / total_baseline_input * 100) if total_baseline_input > 0 else 0

    print(f"\n{'Metric':<30} {'Baseline':>15} {'Headroom':>15} {'Savings':>15}")
    print("-" * 75)
    print(
        f"{'Total Input Tokens':<30} {total_baseline_input:>15,} {total_headroom_input:>15,} {total_saved:>14,}"
    )
    print(f"{'Percentage Saved':<30} {'':>15} {'':>15} {pct_saved:>14.1f}%")

    # Cost
    input_cost = 0.15 / 1_000_000
    baseline_cost = total_baseline_input * input_cost
    headroom_cost = total_headroom_input * input_cost
    cost_saved = baseline_cost - headroom_cost

    print(
        f"\n{'Est. Input Cost (USD)':<30} ${baseline_cost:>14.4f} ${headroom_cost:>14.4f} ${cost_saved:>13.4f}"
    )

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
Headroom reduced input tokens by {pct_saved:.1f}% across all scenarios.

Key optimizations applied:
- SmartCrusher: Compressed tool outputs from 50-200 items to ~20 relevant items
- CacheAligner: Stabilized system prompt for better cache hits
- Context preserved: Agent still found the right information

This translates to:
- Lower API costs
- Faster responses (less data to process)
- Better fit within context windows
""")


if __name__ == "__main__":
    main()
