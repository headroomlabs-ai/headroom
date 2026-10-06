from __future__ import annotations

from headroom.proxy.wire_debug_redaction_policy import (
    WIRE_DEBUG_REDACTED,
    redact_for_wire_debug,
    should_redact_key,
)


def test_wire_debug_redacts_direct_secret_keys() -> None:
    redacted = redact_for_wire_debug(
        {
            "Authorization": "Bearer test-token",
            "x-api-key": "sk-test",
            "safe": "visible",
        }
    )

    assert redacted == {
        "Authorization": WIRE_DEBUG_REDACTED,
        "x-api-key": WIRE_DEBUG_REDACTED,
        "safe": "visible",
    }


def test_wire_debug_redacts_nested_secret_suffixes() -> None:
    redacted = redact_for_wire_debug(
        {
            "messages": [
                {"content": "visible", "service_access_token": "secret-token"},
                {"metadata": {"database_password": "secret-password", "trace_id": "abc"}},
            ]
        }
    )

    assert redacted["messages"][0]["content"] == "visible"
    assert redacted["messages"][0]["service_access_token"] == WIRE_DEBUG_REDACTED
    assert redacted["messages"][1]["metadata"]["database_password"] == WIRE_DEBUG_REDACTED
    assert redacted["messages"][1]["metadata"]["trace_id"] == "abc"


def test_wire_debug_key_matching_normalizes_dashes_and_case() -> None:
    assert should_redact_key("Anthropic-API-Key")
    assert should_redact_key("custom-refresh-token")
    assert not should_redact_key("token_count")


def test_wire_debug_redacts_proxy_authorization() -> None:
    redacted = redact_for_wire_debug(
        {"Proxy-Authorization": "Basic dXNlcjpwYXNz", "user-agent": "codex/1.0"}
    )

    assert redacted["Proxy-Authorization"] == WIRE_DEBUG_REDACTED
    assert redacted["user-agent"] == "codex/1.0"


def test_wire_debug_redacts_any_token_suffix() -> None:
    for key in (
        "auth_token",
        "session_token",
        "api_token",
        "github_token",
        "x-amz-security-token",
        "google_id_token",
    ):
        assert should_redact_key(key), key


def test_wire_debug_redacts_credentials_and_secret_key() -> None:
    for key in ("credentials", "aws_credentials", "secret_key", "aws_secret_key"):
        assert should_redact_key(key), key


def test_wire_debug_keeps_token_usage_counters_visible() -> None:
    for key in (
        "max_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "token_count",
    ):
        assert not should_redact_key(key), key
