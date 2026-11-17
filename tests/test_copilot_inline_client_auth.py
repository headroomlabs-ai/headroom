"""Inline editor OAuth must not be replaced with the proxy operator's seat."""

from __future__ import annotations

import logging
import re

import httpx
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
            "X-Api-Key": "another-provider-secret",
        },
        url="https://copilot-proxy.githubusercontent.com/v1/engines/gpt-41-copilot/completions",
    )

    assert headers["authorization"] == "Bearer ghu_editor_seat_fixture"
    assert headers["Copilot-Integration-Id"] == "vscode"
    assert "x-api-key" not in httpx.Headers(headers)


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
        {
            "authorization": "Bearer ghu_editor_seat_fixture",
            "AUTHORIZATION": "Bearer stale-other-seat",
            "x-api-key": "unrelated-provider-secret",
            "X-Api-Key": "another-provider-secret",
        },
        url="https://api.githubcopilot.com/chat/completions",
    )

    assert headers["Authorization"] == "Bearer tid_operator_seat_fixture"
    assert "authorization" not in headers
    assert "AUTHORIZATION" not in headers
    assert "x-api-key" not in httpx.Headers(headers)


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gateway", "upstream"),
    [
        (
            "https://gateway.example.invalid/v1",
            "https://proxy.business.githubcopilot.com/v1/engines/gpt-41-copilot/completions",
        ),
        (
            "https://proxy.business.githubcopilot.com/v1",
            "http://proxy.business.githubcopilot.com/v1/engines/gpt-41-copilot/completions",
        ),
        (
            "https://proxy.business.githubcopilot.com:8443/v1",
            "https://proxy.business.githubcopilot.com/v1/engines/gpt-41-copilot/completions",
        ),
        (
            "https://gateway.example.invalid/v",
            "https://gateway.example.invalid/v1/engines/gpt-41-copilot/completions",
        ),
    ],
    ids=["other-host", "other-scheme", "other-port", "segment-boundary"],
)
async def test_gateway_prefix_cannot_rewrite_an_unrelated_native_path(
    monkeypatch: pytest.MonkeyPatch, gateway: str, upstream: str
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_PROXY_URL", gateway)
    headers = await copilot_auth.apply_copilot_api_auth(
        {
            "authorization": "Bearer ghu_editor_seat_fixture",
            "Copilot-Integration-Id": "vscode",
            "x-api-key": "unrelated-provider-secret",
        },
        url=upstream,
    )

    header_view = httpx.Headers(headers)
    assert header_view["authorization"] == "Bearer ghu_editor_seat_fixture"
    assert header_view["Copilot-Integration-Id"] == "vscode"
    assert "x-api-key" not in header_view


@pytest.mark.asyncio
async def test_inline_debug_kind_never_discloses_the_editor_credential(caplog) -> None:
    secret = "ghu_sensitive_editor_credential_do_not_log"
    caplog.set_level(logging.INFO, logger="headroom.copilot_auth")
    await copilot_auth.apply_copilot_api_auth(
        {"authorization": f"Bearer {secret}"},
        url="https://copilot-proxy.githubusercontent.com/v1/engines/gpt-41-copilot/completions",
    )
    assert set(re.findall(r"\bkind=(\S+)", caplog.text)) == {"ghu_***"}
    assert secret not in caplog.text
    assert secret.removeprefix("ghu_") not in caplog.text
