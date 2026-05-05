"""Python parity tests for ``headroom.proxy.auth_mode.classify_auth_mode``.

Mirrors the Rust matrix in ``crates/headroom-core/tests/auth_mode.rs``
byte-for-byte. The two implementations MUST agree on every header set
covered here. Any divergence is a bug — we catch it at PR review by
running both suites side by side.
"""

from __future__ import annotations

import time

import pytest

from headroom.proxy.auth_mode import (
    SUBSCRIPTION_UA_PREFIXES,
    AuthMode,
    classify_auth_mode,
)


def _h(**pairs: str) -> dict[str, str]:
    """Build a plain-dict header set in one expression."""
    return dict(pairs)


# ── Required matrix ──────────────────────────────────────────────


def test_api_key_classified_payg() -> None:
    """Anthropic PAYG: ``Authorization: Bearer sk-ant-api03-...``."""
    headers = {"authorization": "Bearer sk-ant-api03-abc123def456"}
    assert classify_auth_mode(headers) is AuthMode.PAYG


def test_oauth_jwt_classified_oauth() -> None:
    """Codex / Cursor OAuth bearer: classic 3-segment JWT."""
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0In0.signaturepart"
    headers = {"authorization": f"Bearer {jwt}"}
    assert classify_auth_mode(headers) is AuthMode.OAUTH


def test_oauth_sk_ant_oat_classified_oauth() -> None:
    """Claude Pro / Max OAuth: ``Bearer sk-ant-oat-...``."""
    headers = {"authorization": "Bearer sk-ant-oat-01-abc123def456"}
    assert classify_auth_mode(headers) is AuthMode.OAUTH


def test_claude_code_ua_classified_subscription() -> None:
    """Claude Code CLI: ``User-Agent: claude-code/1.2.3 ...``."""
    headers = {"user-agent": "claude-code/1.2.3 (darwin; arm64)"}
    assert classify_auth_mode(headers) is AuthMode.SUBSCRIPTION


def test_cursor_ua_classified_subscription() -> None:
    """Cursor CLI: ``User-Agent: cursor/1.0``."""
    headers = {"user-agent": "cursor/1.0"}
    assert classify_auth_mode(headers) is AuthMode.SUBSCRIPTION


def test_no_auth_no_user_agent_default_payg() -> None:
    """Empty headers → safest default is PAYG.

    The OAuth/bedrock branch fires only when there's a positive
    non-Bearer auth signal. Choosing PAYG by default favors the
    OSS-default workload (per-token cost saving).
    """
    assert classify_auth_mode({}) is AuthMode.PAYG


def test_bedrock_no_auth_classified_oauth() -> None:
    """Bedrock SigV4: ``Authorization: AWS4-HMAC-SHA256 Credential=...``.

    Not a Bearer scheme; we treat all non-Bearer Authorization as
    OAuth (passthrough-prefer).
    """
    headers = {
        "authorization": (
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20260501/"
            "us-east-1/bedrock/aws4_request, SignedHeaders=host;x-amz-date, "
            "Signature=fe5f80f77d5fa3beca038a248ff027"
        ),
    }
    assert classify_auth_mode(headers) is AuthMode.OAUTH


# ── Bonus matrix ──────────────────────────────────────────────────


def test_openai_payg_sk_classified_payg() -> None:
    """OpenAI PAYG: ``Authorization: Bearer sk-proj-...``."""
    headers = {"authorization": "Bearer sk-proj-abcdef0123456789"}
    assert classify_auth_mode(headers) is AuthMode.PAYG


def test_gemini_x_goog_api_key_classified_payg() -> None:
    """Google Gemini API key as ``x-goog-api-key``."""
    headers = {"x-goog-api-key": "AIzaSyDUMMYKEY1234567890"}
    assert classify_auth_mode(headers) is AuthMode.PAYG


def test_subscription_takes_precedence_over_oauth_token() -> None:
    """Claude Code CLI sends ``Bearer sk-ant-oat-...`` but is subscription.

    UA wins because the rate-limit / fingerprint policy is bound to
    the CLI, not the token shape. The same token via a non-CLI UA
    would be OAuth.
    """
    headers = {
        "user-agent": "claude-code/1.5.0 (linux; x86_64)",
        "authorization": "Bearer sk-ant-oat-01-abc123",
    }
    assert classify_auth_mode(headers) is AuthMode.SUBSCRIPTION


# ── Edge cases (defensive coverage) ──────────────────────────────


def test_anthropic_x_api_key_classified_payg() -> None:
    """Anthropic API key style: ``x-api-key: sk-ant-...``."""
    headers = {"x-api-key": "sk-ant-api03-abcdef"}
    assert classify_auth_mode(headers) is AuthMode.PAYG


@pytest.mark.parametrize("prefix", SUBSCRIPTION_UA_PREFIXES)
def test_every_subscription_prefix_classified_subscription(prefix: str) -> None:
    """Each entry in the prefix list classifies as Subscription on its own."""
    headers = {"user-agent": f"{prefix}1.0 (test)"}
    assert classify_auth_mode(headers) is AuthMode.SUBSCRIPTION


def test_unparseable_authorization_does_not_raise() -> None:
    """Bytes-valued non-UTF-8 authorization warns and falls through.

    Mirrors the Rust ``classify`` behaviour: never panic, always
    return a valid ``AuthMode``. With nothing else to disambiguate,
    we land on PAYG (the safe default).
    """
    headers = {"authorization": b"\xffnope"}
    # Should not raise.
    result = classify_auth_mode(headers)
    assert result is AuthMode.PAYG


def test_case_insensitive_header_lookup() -> None:
    """Header names are matched case-insensitively (Starlette parity)."""
    headers = {"Authorization": "Bearer sk-ant-api03-test"}
    assert classify_auth_mode(headers) is AuthMode.PAYG


def test_enum_values_match_rust_as_str() -> None:
    """The string form of each enum member matches the Rust ``as_str()``."""
    assert AuthMode.PAYG.value == "payg"
    assert AuthMode.OAUTH.value == "oauth"
    assert AuthMode.SUBSCRIPTION.value == "subscription"


# ── Performance ──────────────────────────────────────────────────


def test_classify_under_100us_per_call() -> None:
    """Smoke perf check.

    Python is slower than Rust by an order of magnitude on string
    work, so the budget here is 100us (vs 10us in Rust). The proxy
    only calls this once per request, and the absolute ceiling is
    well under 1ms; this guards against pathological regressions.
    """
    headers = {
        "user-agent": "claude-code/1.5.0 (linux; x86_64) anthropic/0.42.0",
        "authorization": "Bearer sk-ant-oat-01-abcdefghijklmnopqrstuv",
        "content-type": "application/json",
        "accept": "application/json",
        "host": "api.anthropic.com",
    }

    # Warmup
    for _ in range(1000):
        classify_auth_mode(headers)

    iters = 10_000
    start = time.perf_counter()
    for _ in range(iters):
        classify_auth_mode(headers)
    elapsed = time.perf_counter() - start
    per_call_us = (elapsed / iters) * 1_000_000

    assert per_call_us < 100, f"classify_auth_mode took {per_call_us:.2f} us/call (limit: 100 us)"
