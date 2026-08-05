"""Tests for pure auth and client classification policy."""

from __future__ import annotations

from headroom.proxy.auth_policy import (
    AuthMode,
    AuthSignals,
    classify_auth_signals,
    classify_client_signals,
    should_stamp_codex_client_signals,
)


def test_subscription_user_agent_wins_over_oauth_token() -> None:
    signals = AuthSignals(
        user_agent="claude-code/1.5.0 (linux; x86_64)",
        authorization="Bearer sk-ant-oat01-abc123",
    )

    assert classify_auth_signals(signals) is AuthMode.SUBSCRIPTION


def test_oauth_bearer_token_shapes_are_oauth() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.signature"

    assert classify_auth_signals(AuthSignals(authorization="Bearer sk-ant-oat01-abc")) is (
        AuthMode.OAUTH
    )
    assert classify_auth_signals(AuthSignals(authorization=f"Bearer {jwt}")) is AuthMode.OAUTH


def test_payg_key_shapes_are_payg() -> None:
    assert classify_auth_signals(AuthSignals(authorization="Bearer sk-ant-api03-abc")) is (
        AuthMode.PAYG
    )
    assert classify_auth_signals(AuthSignals(x_api_key="sk-ant-api03-abc")) is AuthMode.PAYG
    assert classify_auth_signals(AuthSignals(x_goog_api_key="AIzaSyDUMMY")) is AuthMode.PAYG


def test_client_explicit_override_wins_over_user_agent() -> None:
    signals = AuthSignals(user_agent="claude-code/1.2.3", x_client=" AIDER ")

    assert classify_client_signals(signals) == "aider"


def test_grok_build_user_agent_is_subscription_client() -> None:
    signals = AuthSignals(user_agent="grok/1.2.3")

    assert classify_auth_signals(signals) is AuthMode.SUBSCRIPTION
    assert classify_client_signals(signals) == "grok_build"


def test_codex_stamp_only_for_unidentified_responses_callers() -> None:
    assert should_stamp_codex_client_signals("/v1/responses", AuthSignals()) is True
    assert (
        should_stamp_codex_client_signals(
            "/v1/responses/foo",
            AuthSignals(user_agent="codex-cli/0.5"),
        )
        is False
    )
    assert should_stamp_codex_client_signals("/v1/chat/completions", AuthSignals()) is False
