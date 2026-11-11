"""The operator's Copilot credential is only attached over https on the default port.

``is_copilot_upstream_url`` matches by host, so ``http://api.githubcopilot.com``
was treated as Copilot and received the operator's token in cleartext. A client
can choose its upstream with ``x-headroom-base-url``.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from headroom import copilot_auth


@pytest.fixture(autouse=True)
def _copilot_session_token(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_api_token(
        *, integration_id: str | None = None
    ) -> copilot_auth.CopilotAPIToken:
        return copilot_auth.CopilotAPIToken(
            token="copilot-session",
            expires_at=time.time() + 3600,
            api_url=copilot_auth.DEFAULT_API_URL,
        )

    monkeypatch.setattr(
        copilot_auth.get_copilot_token_provider(), "get_api_token", fake_get_api_token
    )


def _apply(url: str) -> dict[str, str]:
    return asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {"authorization": "Bearer downstream", "x-api-key": "sk-unrelated"}, url=url
        )
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://api.githubcopilot.com/v1/chat/completions",
        "https://api.githubcopilot.com:443/v1/chat/completions",
        "wss://api.githubcopilot.com/responses",
    ],
)
def test_credential_is_attached_over_https(url: str) -> None:
    assert _apply(url)["Authorization"] == "Bearer copilot-session"


@pytest.mark.parametrize(
    "url",
    [
        "http://api.githubcopilot.com/v1/chat/completions",
        "http://api.githubcopilot.com:8080/v1/chat/completions",
        "https://api.githubcopilot.com:8443/v1/chat/completions",
        "ws://api.githubcopilot.com/responses",
    ],
)
def test_credential_is_not_attached_to_cleartext_or_odd_ports(url: str) -> None:
    headers = _apply(url)
    assert "copilot-session" not in str(headers)
    assert headers["authorization"] == "Bearer downstream"
    assert "x-api-key" not in headers


def test_an_operator_configured_http_host_keeps_working(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "http://127.0.0.1:9999")
    assert _apply("http://127.0.0.1:9999/v1/chat/completions")["Authorization"] == (
        "Bearer copilot-session"
    )
