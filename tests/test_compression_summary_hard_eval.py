"""Hard eval: Cases where the LLM has NO reason to check compressed data.

The previous eval asked "are there failures?" — that's too easy, the LLM
will proactively check regardless of summary.

This eval tests the SUBTLE case: the user asks a DIFFERENT question,
but the answer is in the compressed data. The summary is the only hint.

Requires: ANTHROPIC_API_KEY in environment or .env file.
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
    reason="ANTHROPIC_API_KEY not set",
)

HEADROOM_RETRIEVE_TOOL = {
    "name": "headroom_retrieve",
    "description": "Retrieve uncompressed content. Pass a query to search within it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "hash": {"type": "string"},
            "query": {"type": "string"},
        },
        "required": ["hash"],
    },
}


def _call_claude(messages, tools, max_tokens=300):
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


def _get_tool_calls(resp):
    return [
        {"name": b["name"], "input": b.get("input", {})}
        for b in resp.get("content", [])
        if b.get("type") == "tool_use"
    ]


def _get_text(resp):
    return " ".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")


class TestHardCases:
    """Cases where the LLM wouldn't naturally check compressed data."""

    def test_config_lookup_with_summary(self):
        """User asks about a config value that's in compressed data.

        The visible items are all about 'production' env.
        The compressed items include 'staging' configs.
        Summary mentions this. LLM should retrieve.
        """
        visible = [
            {"env": "production", "key": "DATABASE_URL", "value": "postgres://prod-db:5432/app"},
            {"env": "production", "key": "REDIS_URL", "value": "redis://prod-cache:6379"},
            {"env": "production", "key": "API_RATE_LIMIT", "value": "1000"},
        ]
        # Hidden in compressed: staging configs
        all_items = (
            visible
            + [
                {
                    "env": "staging",
                    "key": "DATABASE_URL",
                    "value": "postgres://staging-db:5432/app",
                },
                {"env": "staging", "key": "REDIS_URL", "value": "redis://staging-cache:6379"},
                {"env": "staging", "key": "DEBUG_MODE", "value": "true"},
                {"env": "staging", "key": "LOG_LEVEL", "value": "debug"},
            ]
            * 10
            + [
                {
                    "env": "development",
                    "key": "DATABASE_URL",
                    "value": "postgres://localhost:5432/dev",
                },
            ]
            * 5
        )

        from headroom.transforms.compression_summary import summarize_dropped_items

        summary = summarize_dropped_items(all_items, visible)

        compressed_output = json.dumps(visible, indent=2)
        compressed_output += (
            f"\n[{len(all_items) - len(visible)} items compressed to {len(visible)}."
            f" Omitted: {summary}."
            f' Retrieve specific items: headroom_retrieve(hash="config_hash", query="search")]'
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Here are the application configs:\n\n{compressed_output}\n\n"
                    "What is the staging database URL?"
                ),
            }
        ]

        resp = _call_claude(messages, [HEADROOM_RETRIEVE_TOOL])
        tool_calls = _get_tool_calls(resp)
        text = _get_text(resp)

        print(f"\n  Summary: {summary}")
        print(f"  Stop reason: {resp.get('stop_reason')}")
        print(f"  Tool calls: {tool_calls}")
        if text:
            print(f"  Text: {text[:200]}")

        # WITH summary mentioning "staging" → should retrieve
        if resp.get("stop_reason") == "tool_use":
            assert tool_calls[0]["name"] == "headroom_retrieve"
            query = tool_calls[0]["input"].get("query", "").lower()
            assert "staging" in query or "database" in query
            print("  RESULT: Retrieved staging config ✓")
        else:
            # If LLM didn't retrieve, it should at least mention the data is compressed
            assert "compressed" in text.lower() or "staging" in text.lower()
            print("  RESULT: Mentioned compressed data but didn't retrieve")

    def test_config_lookup_without_summary(self):
        """Same question, but NO summary. LLM only sees production configs."""
        visible = [
            {"env": "production", "key": "DATABASE_URL", "value": "postgres://prod-db:5432/app"},
            {"env": "production", "key": "REDIS_URL", "value": "redis://prod-cache:6379"},
            {"env": "production", "key": "API_RATE_LIMIT", "value": "1000"},
        ]

        compressed_output = json.dumps(visible, indent=2)
        compressed_output += "\n[45 items compressed to 3. Retrieve more: hash=config_hash]"

        messages = [
            {
                "role": "user",
                "content": (
                    f"Here are the application configs:\n\n{compressed_output}\n\n"
                    "What is the staging database URL?"
                ),
            }
        ]

        resp = _call_claude(messages, [HEADROOM_RETRIEVE_TOOL])
        tool_calls = _get_tool_calls(resp)
        text = _get_text(resp)

        print(f"\n  Stop reason: {resp.get('stop_reason')}")
        print(f"  Tool calls: {tool_calls}")
        if text:
            print(f"  Text: {text[:200]}")

        if resp.get("stop_reason") == "tool_use":
            print("  RESULT: LLM proactively retrieved (smart)")
        else:
            print("  RESULT: LLM did NOT retrieve staging config")

    def test_specific_user_in_large_list_with_summary(self):
        """Find a specific user in a compressed user list.

        Summary mentions user roles. User asks about admins.
        """
        visible = [
            {"id": i, "name": f"user_{i}", "role": "member", "email": f"user{i}@co.com"}
            for i in range(5)
        ]
        all_items = (
            visible
            + [
                {"id": i, "name": f"user_{i}", "role": "member", "email": f"user{i}@co.com"}
                for i in range(5, 95)
            ]
            + [
                {"id": 96, "name": "admin_sarah", "role": "admin", "email": "sarah@co.com"},
                {"id": 97, "name": "admin_mike", "role": "admin", "email": "mike@co.com"},
                {"id": 98, "name": "superadmin_jane", "role": "superadmin", "email": "jane@co.com"},
            ]
        )

        from headroom.transforms.compression_summary import summarize_dropped_items

        summary = summarize_dropped_items(all_items, visible)

        compressed_output = json.dumps(visible, indent=2)
        compressed_output += (
            f"\n[{len(all_items) - len(visible)} items compressed to {len(visible)}."
            f" Omitted: {summary}."
            f' Retrieve: headroom_retrieve(hash="users_hash", query="search")]'
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Here's our user list:\n\n{compressed_output}\n\n"
                    "Who are the admin users? I need to contact them."
                ),
            }
        ]

        resp = _call_claude(messages, [HEADROOM_RETRIEVE_TOOL])
        tool_calls = _get_tool_calls(resp)
        text = _get_text(resp)

        print(f"\n  Summary: {summary}")
        print(f"  Stop reason: {resp.get('stop_reason')}")
        print(f"  Tool calls: {tool_calls}")
        if text:
            print(f"  Text: {text[:200]}")

        if resp.get("stop_reason") == "tool_use":
            query = tool_calls[0]["input"].get("query", "").lower()
            assert "admin" in query
            print(f"  RESULT: Retrieved admin users (query='{query}') ✓")
        else:
            print("  RESULT: Did not retrieve admin users")
