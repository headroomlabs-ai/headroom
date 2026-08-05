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
