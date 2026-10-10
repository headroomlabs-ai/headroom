"""Inline editor OAuth must not be replaced with the proxy operator's seat."""

from __future__ import annotations

import pytest

from headroom import copilot_auth


@pytest.fixture(autouse=True)
def operator_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "tid_operator_seat_fixture")
    monkeypatch.delenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_PROXY_URL", raising=False)


@pytest.mark.asyncio
async def test_inline_editor_uses_its_own_seat_and_integration() -> None:
    headers = await copilot_auth.apply_copilot_api_auth(
        {
            "authorization": "Bearer ghu_editor_seat_fixture",
            "Copilot-Integration-Id": "vscode",
            "x-api-key": "unrelated-provider-secret",
        },
        url="https://copilot-proxy.githubusercontent.com/v1/engines/gpt-41-copilot/completions",
    )

    assert headers["authorization"] == "Bearer ghu_editor_seat_fixture"
    assert headers["Copilot-Integration-Id"] == "vscode"
    assert "x-api-key" not in headers


@pytest.mark.asyncio
async def test_inline_oauth_is_not_relabelled_as_the_chat_integration() -> None:
    headers = await copilot_auth.apply_copilot_api_auth(
        {"Authorization": "Bearer ghu_editor_seat_fixture"},
        url="https://proxy.business.githubcopilot.com/v1/engines/gpt-41-copilot/completions",
    )

    assert headers["Authorization"] == "Bearer ghu_editor_seat_fixture"
    assert all(name.lower() != "copilot-integration-id" for name in headers)


@pytest.mark.asyncio
async def test_inline_oauth_exception_does_not_widen_chat_auth() -> None:
    headers = await copilot_auth.apply_copilot_api_auth(
        {"authorization": "Bearer ghu_editor_seat_fixture"},
        url="https://api.githubcopilot.com/chat/completions",
    )

    assert headers["Authorization"] == "Bearer tid_operator_seat_fixture"
    assert "authorization" not in headers


@pytest.mark.asyncio
async def test_inline_editor_keeps_its_seat_through_a_prefixed_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_PROXY_URL", "https://gateway.example.invalid/copilot")
    headers = await copilot_auth.apply_copilot_api_auth(
        {
            "authorization": "Bearer ghu_editor_seat_fixture",
            "Copilot-Integration-Id": "vscode",
        },
        url="https://gateway.example.invalid/copilot/v1/engines/gpt-41-copilot/completions",
    )

    assert headers["authorization"] == "Bearer ghu_editor_seat_fixture"
    assert headers["Copilot-Integration-Id"] == "vscode"
