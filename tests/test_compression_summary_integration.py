"""Integration eval: Compression summaries with real LLM calls.

Tests whether compression summaries actually help the LLM find information
in compressed data. Compares behavior with and without summaries.

Requires: ANTHROPIC_API_KEY in environment or .env file.

Run: python -m pytest tests/test_compression_summary_integration.py -v -s
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


def _call_claude(messages: list[dict], max_tokens: int = 200) -> dict:
    """Make a real Anthropic API call."""
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
        },
        timeout=30,
    )
    return resp.json()


# ============================================================================
# Test data: realistic tool output that gets compressed
# ============================================================================


def _make_test_suite_output(n: int = 100) -> list[dict]:
    """Simulate a large test suite result (like from a CI/CD tool)."""
    results = []
    for i in range(n):
        result = {
            "test_name": f"test_module_{i // 10}.test_case_{i}",
            "status": "passed",
            "duration_ms": 50 + i * 3,
            "file": f"tests/test_module_{i // 10}.py",
        }
        # Inject specific failures that the LLM should find
        if i == 42:
            result["status"] = "failed"
            result["error"] = "AssertionError: expected status 200, got 401 in auth_middleware"
            result["test_name"] = "test_auth.test_login_with_expired_token"
        if i == 67:
            result["status"] = "failed"
            result["error"] = "TimeoutError: database connection pool exhausted after 30s"
            result["test_name"] = "test_database.test_concurrent_connections"
        if i == 88:
            result["status"] = "error"
            result["error"] = "ImportError: cannot import name 'NewFeature' from 'app.features'"
            result["test_name"] = "test_features.test_new_feature_integration"
        results.append(result)
    return results


class TestSummaryHelpfulness:
    """Compare LLM accuracy with vs without compression summaries."""

    def test_find_failures_with_summary(self):
        """LLM can identify failure types from the summary alone."""
        test_results = _make_test_suite_output(100)

        # Simulate compression: keep first 10, compress rest with summary
        kept = test_results[:10]
        from headroom.transforms.compression_summary import summarize_dropped_items

        summary = summarize_dropped_items(test_results, kept)

        compressed_output = json.dumps(kept, indent=2)
        compressed_output += f"\n[90 items compressed to 10. Omitted: {summary}. "
        compressed_output += (
            'Retrieve specific items: headroom_retrieve(hash="abc123", query="your search")]'
        )

        messages = [
            {
                "role": "user",
                "content": (
                    "Here are the test results from CI:\n\n"
                    f"{compressed_output}\n\n"
                    "Are there any test failures? What types of failures are there? "
                    "Answer concisely."
                ),
            },
        ]

        resp = _call_claude(messages)
        text = resp.get("content", [{}])[0].get("text", "").lower()

        # The LLM should mention failures (from the summary info)
        has_failure_info = any(
            word in text for word in ["fail", "error", "timeout", "assert", "import"]
        )
        print(f"\n  Summary: {summary}")
        print(f"  LLM response: {text[:200]}")
        print(f"  Detected failure info: {has_failure_info}")

        assert has_failure_info, f"LLM didn't detect failures from summary. Response: {text[:300]}"

    def test_find_failures_without_summary(self):
        """Baseline: LLM with NO summary — just '[90 items compressed]'."""
        test_results = _make_test_suite_output(100)

        kept = test_results[:10]
        compressed_output = json.dumps(kept, indent=2)
        compressed_output += "\n[90 items compressed to 10. Retrieve more: hash=abc123]"

        messages = [
            {
                "role": "user",
                "content": (
                    "Here are the test results from CI:\n\n"
                    f"{compressed_output}\n\n"
                    "Are there any test failures? What types of failures are there? "
                    "Answer concisely."
                ),
            },
        ]

        resp = _call_claude(messages)
        text = resp.get("content", [{}])[0].get("text", "").lower()

        # The LLM may or may not detect failures (it only sees 10 passing tests)
        has_failure_info = any(
            word in text for word in ["fail", "error", "timeout", "assert", "import"]
        )
        print(f"\n  LLM response (no summary): {text[:200]}")
        print(f"  Detected failure info: {has_failure_info}")

        # We're NOT asserting here — this is the baseline.
        # We expect this to often MISS failures since the summary is generic.

    def test_code_summary_helps_identify_functions(self):
        """LLM can identify which functions were removed from compressed code."""
        compressed_code = '''
class PaymentProcessor:
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
'''
        from headroom.transforms.compression_summary import summarize_compressed_code

        # Use AST-based summary (language-agnostic)
        bodies = [
            ("def charge(self, amount: float, currency: str, token: str) -> dict:", "...", 10),
            ("def refund(self, charge_id: str, amount: float = None) -> dict:", "...", 20),
            ("def get_balance(self) -> float:", "...", 30),
        ]
        code_summary = summarize_compressed_code(bodies, 3)

        prompt = f"Here is a compressed Python file:\n\n```python\n{compressed_code}\n```\n\n"
        if code_summary:
            prompt += f"[Compression info: {code_summary}]\n\n"
        prompt += "I need to understand the retry logic. Which function should I look at? Answer in one sentence."

        messages = [{"role": "user", "content": prompt}]
        resp = _call_claude(messages, max_tokens=100)
        text = resp.get("content", [{}])[0].get("text", "").lower()

        print(f"\n  Code summary: {code_summary}")
        print(f"  LLM response: {text[:200]}")

        # The LLM should identify the charge() function
        assert "charge" in text, f"LLM didn't identify charge() function. Response: {text}"
