"""Pytest fixtures for Headroom benchmarks.

This module provides shared fixtures for benchmark tests including:
- Generated data arrays of various sizes
- Conversation fixtures with tool calls
- System prompts with/without dynamic dates
- Mock tokenizers for consistent measurement

All fixtures are designed to produce deterministic data for reliable
benchmark comparisons across runs.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from benchmarks.scenarios.conversations import (
    generate_agentic_conversation,
    generate_rag_conversation,
)
from benchmarks.scenarios.tool_outputs import (
    generate_api_responses,
    generate_database_rows,
    generate_log_entries,
    generate_search_results,
)

# Set seed for reproducible benchmarks
random.seed(42)


# =============================================================================
# Mock Tokenizer
# =============================================================================


class MockTokenCounter:
    """Mock token counter for benchmarks.

    Uses simple character-based estimation (4 chars = 1 token) for
    fast, consistent token counting without model dependencies.
    """

    def count_text(self, text: str) -> int:
        """Estimate tokens in text (4 chars = 1 token)."""
        return max(1, len(text) // 4)

    def count_message(self, message: dict[str, Any]) -> int:
        """Estimate tokens in a message."""
        content = message.get("content", "")
        if isinstance(content, str):
            return self.count_text(content) + 4  # Overhead for role
        elif isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total += self.count_text(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        total += self.count_text(str(block.get("content", "")))
                    elif block.get("type") == "tool_use":
                        total += self.count_text(json.dumps(block.get("input", {})))
            return total + 4
        else:
            return 10  # Default estimate

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Estimate tokens in message list."""
        return sum(self.count_message(m) for m in messages)


@pytest.fixture
def mock_token_counter() -> MockTokenCounter:
    """Provide mock token counter for benchmarks."""
    return MockTokenCounter()


@pytest.fixture
def mock_tokenizer(mock_token_counter: MockTokenCounter):
    """Provide mock Tokenizer wrapper."""
    from headroom.tokenizer import Tokenizer

    return Tokenizer(token_counter=mock_token_counter, model="benchmark-model")


# =============================================================================
# Data Array Fixtures (various sizes)
# =============================================================================


@pytest.fixture
def items_100() -> list[dict[str, Any]]:
    """Generate 100 search result items."""
    random.seed(42)
    return generate_search_results(100)


@pytest.fixture
def items_1000() -> list[dict[str, Any]]:
    """Generate 1000 search result items."""
    random.seed(42)
    return generate_search_results(1000)


@pytest.fixture
def items_10000() -> list[dict[str, Any]]:
    """Generate 10000 search result items."""
    random.seed(42)
    return generate_search_results(10000)


@pytest.fixture
def log_entries_100() -> list[dict[str, Any]]:
    """Generate 100 log entries."""
    random.seed(42)
    return generate_log_entries(100)


@pytest.fixture
def log_entries_1000() -> list[dict[str, Any]]:
    """Generate 1000 log entries."""
    random.seed(42)
    return generate_log_entries(1000)


@pytest.fixture
def database_rows_100() -> list[dict[str, Any]]:
    """Generate 100 database rows with metrics (for anomaly detection)."""
    random.seed(42)
    return generate_database_rows(100, table_type="metrics")


@pytest.fixture
def database_rows_1000() -> list[dict[str, Any]]:
    """Generate 1000 database rows with metrics."""
    random.seed(42)
    return generate_database_rows(1000, table_type="metrics")


@pytest.fixture
def api_responses_100() -> list[dict[str, Any]]:
    """Generate 100 API response items."""
    random.seed(42)
    return generate_api_responses(100)


# =============================================================================
# Conversation Fixtures
# =============================================================================


@pytest.fixture
def conversation_10_turns() -> list[dict[str, Any]]:
    """Generate 10-turn agentic conversation with tool calls."""
    random.seed(42)
    return generate_agentic_conversation(
        turns=10, tool_calls_per_turn=1, items_per_tool_response=50
    )


@pytest.fixture
def conversation_50_turns() -> list[dict[str, Any]]:
    """Generate 50-turn agentic conversation with tool calls."""
    random.seed(42)
    return generate_agentic_conversation(
        turns=50, tool_calls_per_turn=2, items_per_tool_response=50
    )


@pytest.fixture
def conversation_200_turns() -> list[dict[str, Any]]:
    """Generate 200-turn agentic conversation (stress test)."""
    random.seed(42)
    return generate_agentic_conversation(
        turns=200, tool_calls_per_turn=1, items_per_tool_response=30
    )


@pytest.fixture
def rag_conversation_5k() -> list[dict[str, Any]]:
    """Generate RAG conversation with ~5K context tokens."""
    random.seed(42)
    return generate_rag_conversation(context_tokens=5000, num_queries=3)


@pytest.fixture
def rag_conversation_20k() -> list[dict[str, Any]]:
    """Generate RAG conversation with ~20K context tokens."""
    random.seed(42)
    return generate_rag_conversation(context_tokens=20000, num_queries=5)


@pytest.fixture
def rag_conversation_50k() -> list[dict[str, Any]]:
    """Generate RAG conversation with ~50K context tokens."""
    random.seed(42)
    return generate_rag_conversation(context_tokens=50000, num_queries=5)


# =============================================================================
# System Prompt Fixtures
# =============================================================================


@pytest.fixture
def system_prompt_with_date() -> str:
    """System prompt containing dynamic date."""
    return """You are a helpful AI assistant.

Current date: 2025-01-06
Today is Monday, January 6th, 2025.

You have access to various tools for searching and querying data.
Always provide accurate and helpful responses."""


@pytest.fixture
def system_prompt_without_date() -> str:
    """System prompt without dynamic date (stable)."""
    return """You are a helpful AI assistant.

You have access to various tools for searching and querying data.
Always provide accurate and helpful responses.

Guidelines:
1. Be concise and accurate
2. Use tools when appropriate
3. Cite sources when available"""


@pytest.fixture
def system_prompt_long() -> str:
    """Long system prompt for cache alignment testing."""
    sections = [
        "You are an expert AI assistant with deep knowledge in software engineering.",
        "\n\n## Capabilities\n- Code analysis and review\n- Debugging and troubleshooting\n- Architecture recommendations\n- Performance optimization",
        "\n\n## Guidelines\n1. Always explain your reasoning\n2. Provide code examples when helpful\n3. Consider edge cases\n4. Suggest best practices",
        "\n\n## Tools Available\n- search_code: Search code repositories\n- query_database: Query application databases\n- get_logs: Retrieve service logs\n- run_tests: Execute test suites",
        "\n\n## Response Format\n- Use markdown for formatting\n- Include code blocks with syntax highlighting\n- Organize long responses with headers\n- Summarize key points at the end",
    ]
    return "".join(sections)


@pytest.fixture
def messages_with_tool_output(items_100) -> list[dict[str, Any]]:
    """Messages containing a tool output for crushing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Search for recent users"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "search_users", "arguments": '{"limit": 100}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": json.dumps(items_100),
        },
    ]


@pytest.fixture
def messages_with_system_date(system_prompt_with_date) -> list[dict[str, Any]]:
    """Messages with system prompt containing date."""
    return [
        {"role": "system", "content": system_prompt_with_date},
        {"role": "user", "content": "What's the current date?"},
        {"role": "assistant", "content": "Today is January 6th, 2025."},
    ]


# =============================================================================
# Transform Configuration Fixtures
# =============================================================================


@pytest.fixture
def smart_crusher_config():
    """SmartCrusher config optimized for benchmarks."""
    from headroom.config import SmartCrusherConfig

    return SmartCrusherConfig(
        enabled=True,
        min_items_to_analyze=5,
        min_tokens_to_crush=0,  # Always crush
        max_items_after_crush=15,
        variance_threshold=2.0,
    )


@pytest.fixture
def cache_aligner_config():
    """CacheAligner config for benchmarks."""
    from headroom.config import CacheAlignerConfig

    return CacheAlignerConfig(
        enabled=True,
        normalize_whitespace=True,
        collapse_blank_lines=True,
    )


# =============================================================================
# JSON String Fixtures (for relevance benchmarks)
# =============================================================================


@pytest.fixture
def json_items_100(items_100) -> list[str]:
    """100 items as JSON strings."""
    return [json.dumps(item) for item in items_100]


@pytest.fixture
def json_items_1000(items_1000) -> list[str]:
    """1000 items as JSON strings."""
    return [json.dumps(item) for item in items_1000]


@pytest.fixture
def query_context_uuid() -> str:
    """Query context containing a UUID (for BM25 testing)."""
    return "Find the record with UUID 550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def query_context_semantic() -> str:
    """Query context requiring semantic understanding."""
    return "Show me all the failed requests and errors"


@pytest.fixture
def query_context_mixed() -> str:
    """Query context with both exact match and semantic terms."""
    return "Find user 12345 and show any associated errors"
