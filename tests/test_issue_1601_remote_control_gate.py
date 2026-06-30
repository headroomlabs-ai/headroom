"""Issue #1601: Claude Remote Control is unavailable with a custom ANTHROPIC_BASE_URL."""

from __future__ import annotations

from headroom.providers.claude.runtime import (
    REMOTE_CONTROL_BASE_URL_ENV,
    is_custom_anthropic_base_url,
    remote_control_gate_message,
)


def test_custom_anthropic_base_url_is_remote_control_gated() -> None:
    assert is_custom_anthropic_base_url("http://127.0.0.1:8787")
    assert is_custom_anthropic_base_url("https://gateway.internal.example")


def test_native_anthropic_base_url_is_not_remote_control_gated() -> None:
    assert not is_custom_anthropic_base_url("https://api.anthropic.com")


def test_remote_control_gate_message_mentions_warning_and_source() -> None:
    message = remote_control_gate_message(source=REMOTE_CONTROL_BASE_URL_ENV)
    assert "Remote Control" in message
    assert REMOTE_CONTROL_BASE_URL_ENV in message
    assert "launch Claude without Headroom for sessions that need this feature" in message
