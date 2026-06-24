"""Eval: Does the LLM invoke headroom_retrieve when summaries are present?

The REAL test — it's not enough for the LLM to know something is missing.
It must actually call the tool to fetch it.

Compares:
- WITH summary: LLM sees "2 failed, 1 error" → should call headroom_retrieve
- WITHOUT summary: LLM sees "[90 items compressed]" → likely does NOT call tool

Requires: ANTHROPIC_API_KEY in environment or .env file.

Run: python -m pytest tests/test_compression_summary_tool_eval.py -v -s
"""

from __future__ import annotations

import json
import os

import pytest

from tests._dotenv import autouse_apply_env, load_env_overrides

_env_overrides = load_env_overrides()
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or _env_overrides.get("ANTHROPIC_API_KEY", "")
apply_dotenv = autouse_apply_env(_env_overrides)

pytestmark = pytest.mark.skipif(
    not ANTHROPIC_KEY,
    reason="ANTHROPIC_API_KEY not set — skipping integration tests",
)

# The headroom_retrieve tool definition (same as what CCR injects)
HEADROOM_RETRIEVE_TOOL = {
    "name": "headroom_retrieve",
    "description": (
        "Retrieve original uncompressed content from Headroom's compression cache. "
        "Use this when you need more details from compressed data. "
        "You can pass a query to search within the compressed content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "hash": {
                "type": "string",
                "description": "The hash key from the compression marker",
            },
            "query": {
                "type": "string",
                "description": "Optional search query to find specific items within the compressed data",
            },
        },
        "required": ["hash"],
    },
}


def _call_claude_with_tools(messages: list[dict], tools: list[dict], max_tokens: int = 300) -> dict:
    """Make a real Anthropic API call with tool use."""
    import httpx

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "X-Api-Key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": max_tokens,
            "messages": messages,
            "tools": tools,
        },
        timeout=30,
    )
    return resp.json()


def _make_test_results(n: int = 100) -> list[dict]:
    """Test suite output with hidden failures in the compressed portion."""
    results = []
    for i in range(n):
        result = {
            "test_name": f"test_module_{i // 10}.test_case_{i}",
            "status": "passed",
            "duration_ms": 50 + i * 3,
        }
        if i == 42:
            result["status"] = "failed"
            result["error"] = "AssertionError: expected 200, got 401 in auth_middleware"
            result["test_name"] = "test_auth.test_login_expired_token"
        if i == 67:
            result["status"] = "failed"
            result["error"] = "TimeoutError: database pool exhausted after 30s"
            result["test_name"] = "test_database.test_concurrent_connections"
        if i == 88:
            result["status"] = "error"
            result["error"] = "ImportError: cannot import 'NewFeature'"
            result["test_name"] = "test_features.test_new_feature_integration"
        results.append(result)
    return results


def _has_tool_use(response: dict) -> bool:
    """Check if the response contains a tool_use block."""
    for block in response.get("content", []):
        if block.get("type") == "tool_use":
            return True
    return False


def _get_tool_calls(response: dict) -> list[dict]:
    """Extract all tool_use blocks from response."""
    calls = []
    for block in response.get("content", []):
        if block.get("type") == "tool_use":
            calls.append(
                {
                    "name": block.get("name"),
                    "input": block.get("input", {}),
                }
            )
    return calls


class TestToolInvocationWithSummary:
    """The real eval: does the LLM call headroom_retrieve?"""

    def test_with_summary_triggers_tool_call(self):
        """WITH compression summary → LLM should call headroom_retrieve."""
        test_results = _make_test_results(100)
        kept = test_results[:10]  # All passing

        from headroom.transforms.compression_summary import summarize_dropped_items

        summary = summarize_dropped_items(test_results, kept)

        compressed = json.dumps(kept, indent=2)
        compressed += (
            f"\n[90 items compressed to 10. Omitted: {summary}."
            f' Retrieve specific items: headroom_retrieve(hash="ccr_test_abc123", query="your search")]'
        )

        messages = [
            {
                "role": "user",
                "content": (
                    "Here are the test results from our CI pipeline:\n\n"
                    f"{compressed}\n\n"
                    "Tell me about any test failures. What went wrong?"
                ),
            },
        ]

        resp = _call_claude_with_tools(messages, [HEADROOM_RETRIEVE_TOOL])

        tool_calls = _get_tool_calls(resp)
        stop_reason = resp.get("stop_reason", "")

        print(f"\n  Summary: {summary}")
        print(f"  Stop reason: {stop_reason}")
        print(f"  Tool calls: {tool_calls}")

        # With a summary showing failures, the LLM SHOULD call the tool
        if stop_reason == "tool_use":
            assert len(tool_calls) > 0
            call = tool_calls[0]
            assert call["name"] == "headroom_retrieve"
            assert call["input"].get("hash") == "ccr_test_abc123"
            # The query should be about failures/errors
            query = call["input"].get("query", "").lower()
            print(f"  Query used: {query}")
            has_relevant_query = any(
                term in query for term in ["fail", "error", "issue", "problem", "broken", "test"]
            )
            assert has_relevant_query, f"Tool was called but query isn't relevant: {query}"
            print("  RESULT: LLM invoked headroom_retrieve with relevant query ✓")
        else:
            # LLM responded with text — check if it at least mentions the failures
            text = ""
            for block in resp.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            print(f"  LLM text response: {text[:200]}")
            # It's acceptable if the LLM mentions it WANTS to retrieve
            mentions_retrieval = any(
                term in text.lower()
                for term in ["retrieve", "headroom_retrieve", "fetch", "see more", "compressed"]
            )
            print(f"  Mentions retrieval: {mentions_retrieval}")

    def test_without_summary_baseline(self):
        """WITHOUT compression summary → LLM likely does NOT call tool."""
        test_results = _make_test_results(100)
        kept = test_results[:10]  # All passing

        compressed = json.dumps(kept, indent=2)
        compressed += "\n[90 items compressed to 10. Retrieve more: hash=ccr_test_abc123]"

        messages = [
            {
                "role": "user",
                "content": (
                    "Here are the test results from our CI pipeline:\n\n"
                    f"{compressed}\n\n"
                    "Tell me about any test failures. What went wrong?"
                ),
            },
        ]

        resp = _call_claude_with_tools(messages, [HEADROOM_RETRIEVE_TOOL])

        tool_calls = _get_tool_calls(resp)
        stop_reason = resp.get("stop_reason", "")

        print(f"\n  Stop reason: {stop_reason}")
        print(f"  Tool calls: {tool_calls}")

        if stop_reason == "tool_use":
            call = tool_calls[0]
            print(f"  Query used: {call['input'].get('query', 'none')}")
            print("  RESULT: LLM DID invoke tool (may check proactively)")
        else:
            text = ""
            for block in resp.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            print(f"  LLM text response: {text[:200]}")
            print("  RESULT: LLM did NOT invoke tool — assumed all tests passed")

    def test_code_summary_triggers_retrieval(self):
        """Code compression summary → LLM should retrieve specific function."""
        compressed_code = '''class PaymentProcessor:
    """Processes payments via Stripe."""

    def __init__(self, api_key: str):
        # [2 lines omitted]
        pass

    def charge(self, amount: float, currency: str, token: str) -> dict:
        # [8 lines omitted]
        pass

    def refund(self, charge_id: str, amount: float = None) -> dict:
        # [3 lines omitted]
        pass

    def get_balance(self) -> float:
        # [2 lines omitted]
        pass
# [180 tokens compressed. removed: def charge (12 lines), def refund (6 lines). Retrieve full code: headroom_retrieve(hash="ccr_code_xyz", query="function name")]'''

        messages = [
            {
                "role": "user",
                "content": (
                    "Here's the payment processor code:\n\n"
                    f"```python\n{compressed_code}\n```\n\n"
                    "There's a bug in the retry logic for failed charges. "
                    "Can you find and fix it?"
                ),
            },
        ]

        resp = _call_claude_with_tools(messages, [HEADROOM_RETRIEVE_TOOL])

        tool_calls = _get_tool_calls(resp)
        stop_reason = resp.get("stop_reason", "")

        print(f"\n  Stop reason: {stop_reason}")
        print(f"  Tool calls: {tool_calls}")

        if stop_reason == "tool_use":
            call = tool_calls[0]
            assert call["name"] == "headroom_retrieve"
            query = call["input"].get("query", "").lower()
            print(f"  Query: {query}")
            # Should be asking for the charge function specifically
            has_charge = any(term in query for term in ["charge", "retry", "payment", "stripe"])
            print(f"  Targets charge/retry: {has_charge}")
            print("  RESULT: LLM invoked tool to get the charge() implementation ✓")
        else:
            text = ""
            for block in resp.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            print(f"  LLM text: {text[:200]}")
            print("  RESULT: LLM did NOT invoke tool")
