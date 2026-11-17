"""Integration eval: Compression summaries with real LLM calls.

Tests whether the code-compression summary actually helps the LLM tell which
functions were removed from compressed code.

Requires: ANTHROPIC_API_KEY in environment or .env file.

Run: python -m pytest tests/test_compression_summary_integration.py -v -s
"""

from __future__ import annotations

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


class TestSummaryHelpfulness:
    """Check that the code summary carries information the LLM can use."""

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
