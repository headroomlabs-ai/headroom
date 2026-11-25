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


@pytest.mark.parametrize(
    ("configured", "websocket_url"),
    [
        ("https://api.githubcopilot.com:8443", "wss://api.githubcopilot.com:8443/responses"),
        ("http://127.0.0.1:9999", "ws://127.0.0.1:9999/responses"),
    ],
)
def test_the_websocket_form_of_a_configured_origin_is_trusted(
    monkeypatch: pytest.MonkeyPatch, configured: str, websocket_url: str
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", configured)
    assert _apply(websocket_url)["Authorization"] == "Bearer copilot-session"


def test_the_websocket_form_does_not_widen_the_configured_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://api.githubcopilot.com:8443")
    headers = _apply("wss://api.githubcopilot.com:9000/responses")
    assert "copilot-session" not in str(headers)


def test_a_seeded_operator_token_is_not_forwarded_to_an_untrusted_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "tid_operator_seed")
    monkeypatch.setenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", "gho-refresh")
    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {"authorization": "Bearer tid_operator_seed"},
            url="http://api.githubcopilot.com/v1/chat/completions",
        )
    )
    assert "tid_operator_seed" not in str(headers)
    assert "copilot-session" not in str(headers)


@pytest.mark.parametrize(
    ("configured", "url"),
    [
        ("http://api.githubcopilot.com:80", "http://api.githubcopilot.com/v1/chat/completions"),
        ("http://api.githubcopilot.com", "http://api.githubcopilot.com:80/v1/chat/completions"),
        ("https://api.githubcopilot.com:443", "https://api.githubcopilot.com/v1/x"),
        ("http://API.githubcopilot.com:80", "http://api.GITHUBCOPILOT.com/v1/x"),
        ("http://api.githubcopilot.com:80", "ws://api.githubcopilot.com/responses"),
    ],
)
def test_a_configured_origin_matches_by_effective_port(
    monkeypatch: pytest.MonkeyPatch, configured: str, url: str
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", configured)
    assert _apply(url)["Authorization"] == "Bearer copilot-session"


@pytest.mark.parametrize(
    ("configured", "url"),
    [
        ("http://api.githubcopilot.com:80", "http://api.githubcopilot.com:8080/v1/x"),
        ("http://api.githubcopilot.com:80", "http://evil.example/v1/x"),
        ("https://api.githubcopilot.com:443", "http://api.githubcopilot.com/v1/x"),
        ("http://gateway.invalid:0", "http://gateway.invalid/v1/x"),
        ("http://api.githubcopilot.com:80", "http://api.githubcopilot.com:0/v1/x"),
    ],
)
def test_effective_port_matching_does_not_widen_the_configured_origin(
    monkeypatch: pytest.MonkeyPatch, configured: str, url: str
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", configured)
    headers = _apply(url)
    assert "copilot-session" not in str(headers)
    assert headers["authorization"] == "Bearer downstream"


def test_explicit_port_zero_is_not_the_default_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "http://gateway.invalid:0")
    # The configuration never authorized port 80, so the omitted port must not match it.
    assert not copilot_auth._is_trusted_copilot_origin("http://gateway.invalid")
    assert not copilot_auth._is_trusted_copilot_origin("http://gateway.invalid:0")
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "http://gateway.invalid:80")
    assert copilot_auth._is_trusted_copilot_origin("http://gateway.invalid")
    assert not copilot_auth._is_trusted_copilot_origin("http://gateway.invalid:0")
