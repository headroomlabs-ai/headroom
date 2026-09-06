from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from urllib import error as urllib_error

import pytest

from headroom import copilot_auth
from headroom.proxy import ssl_context


def test_device_authorization_uses_form_encoded_request(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    captured = {}

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        captured["request"] = request
        return io.BytesIO(b'{"device_code":"d","user_code":"u"}')

    monkeypatch.setattr(copilot_auth.urllib_request, "urlopen", fake_urlopen)
    copilot_auth.start_copilot_device_authorization()

    request = captured["request"]
    assert request.headers["Content-type"] == "application/x-www-form-urlencoded"
    assert b"client_id=" in request.data
    assert not request.data.startswith(b"{")


def test_device_poll_uses_form_encoded_request(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    captured = {}

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        captured["request"] = request
        return io.BytesIO(b'{"access_token":"gho-test"}')

    monkeypatch.setattr(copilot_auth.urllib_request, "urlopen", fake_urlopen)
    assert copilot_auth.poll_copilot_device_authorization("device-test") == "gho-test"

    request = captured["request"]
    assert request.headers["Content-type"] == "application/x-www-form-urlencoded"
    assert b"device_code=device-test" in request.data
    assert not request.data.startswith(b"{")


@pytest.fixture(autouse=True)
def _isolated_copilot_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep Copilot auth tests away from user secret stores and real auth files."""

    for var in (
        "GITHUB_COPILOT_API_TOKEN",
        "COPILOT_PROVIDER_BEARER_TOKEN",
        "GITHUB_COPILOT_GITHUB_TOKEN",
        "GITHUB_COPILOT_TOKEN",
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GITHUB_COPILOT_API_URL",
        "GITHUB_COPILOT_HOST",
        "GITHUB_COPILOT_ENTERPRISE_URL",
        "GITHUB_COPILOT_ENTERPRISE_DOMAIN",
        "GITHUB_COPILOT_TOKEN_EXCHANGE_URL",
        "GITHUB_COPILOT_USER_INFO_URL",
        "GITHUB_COPILOT_USER_AGENT",
        "GITHUB_COPILOT_EDITOR_VERSION",
        "GITHUB_COPILOT_EDITOR_PLUGIN_VERSION",
        "GITHUB_COPILOT_INTEGRATION_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(copilot_auth, "_provider", None)
    monkeypatch.setenv("HEADROOM_COPILOT_AUTH_FILE", str(tmp_path / "copilot_auth.json"))
    monkeypatch.setattr(copilot_auth, "read_macos_keychain_token", lambda *, host: None)
    monkeypatch.setattr(copilot_auth, "read_linux_secret_token", lambda *, host: None)


def test_read_cached_oauth_token_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-env")
    assert copilot_auth.read_cached_oauth_token() == "gho-env"


def test_read_cached_oauth_token_prefers_headroom_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-env")
    copilot_auth.save_headroom_copilot_oauth_token("gho-headroom")

    assert copilot_auth.read_cached_oauth_token() == "gho-headroom"


def test_default_oauth_domain_uses_enterprise_url_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_URL", "https://ghe.example.com")

    assert copilot_auth.default_oauth_domain() == "ghe.example.com"


def test_default_oauth_domain_falls_back_to_github_com_when_env_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_URL", "   ")
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_DOMAIN", "")

    assert copilot_auth.default_oauth_domain() == "github.com"


@pytest.mark.parametrize(
    ("api_url", "expected"),
    [
        ("https://api.GHE.Example.com:8443/copilot", "ghe.example.com"),
        ("https://copilot-api.GHE.Example.com", "ghe.example.com"),
        ("https://api.githubcopilot.com", "github.com"),
        ("https://api.business.githubcopilot.com", "github.com"),
        ("https://api.enterprise.githubcopilot.com", "github.com"),
        ("https://api.individual.githubcopilot.com", "github.com"),
        ("https://api.ghe.example.com:invalid/copilot", "github.com"),
        ("https://[", "github.com"),
        ("not a URL", "github.com"),
        ("https://", "github.com"),
    ],
)
def test_github_host_derives_from_api_url(
    monkeypatch: pytest.MonkeyPatch, api_url: str, expected: str
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", api_url)
    assert copilot_auth._github_host() == expected


def test_github_host_explicit_value_wins_over_api_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_HOST", "  Explicit.GHE.COM ")
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://other.example.com/api")
    assert copilot_auth._github_host() == "explicit.ghe.com"


def test_github_host_uses_configured_enterprise_domain_before_api_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_URL", "https://enterprise.ghe.example.com")
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://api.other.example.com/api")
    assert copilot_auth._github_host() == "enterprise.ghe.example.com"


def test_github_host_invalid_enterprise_url_falls_back_to_github_com(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_URL", "https://[")
    assert copilot_auth._github_host() == "github.com"


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("GITHUB_COPILOT_ENTERPRISE_URL", "https://api.business.githubcopilot.com"),
        ("GITHUB_COPILOT_ENTERPRISE_DOMAIN", "enterprise.githubcopilot.com"),
    ],
)
def test_public_enterprise_config_keeps_public_defaults(
    monkeypatch: pytest.MonkeyPatch, env_name: str, env_value: str
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN_EXCHANGE_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_USER_INFO_URL", raising=False)
    monkeypatch.setenv(env_name, env_value)

    assert copilot_auth._github_host() == "github.com"
    assert copilot_auth.default_oauth_domain() == "github.com"
    assert copilot_auth._token_exchange_url() == copilot_auth.DEFAULT_TOKEN_EXCHANGE_URL
    assert copilot_auth._user_info_url() == copilot_auth.DEFAULT_USER_INFO_URL


def test_enterprise_hostname_blank_returns_empty() -> None:
    assert copilot_auth._enterprise_hostname("   ") == ""


def test_configured_url_hostname_blank_returns_empty() -> None:
    assert copilot_auth._configured_url_hostname("   ") == ""


def test_copilot_subdomain_enterprise_host_rejects_blank_and_public_hosts() -> None:
    assert copilot_auth._copilot_subdomain_enterprise_host("   ") is None
    assert copilot_auth._copilot_subdomain_enterprise_host("https://github.com") is None
    assert (
        copilot_auth._copilot_subdomain_enterprise_host("https://api.business.githubcopilot.com")
        is None
    )


def test_read_cached_oauth_token_prefers_copilot_cli_before_generic_github_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-generic")
    monkeypatch.setattr(
        copilot_auth,
        "_read_windows_copilot_cli_oauth_token",
        lambda: None,
    )
    monkeypatch.setattr(copilot_auth, "_read_macos_keychain_oauth_token", lambda: "gho-keychain")
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)

    assert copilot_auth.read_cached_oauth_token() == "gho-keychain"


def test_iter_oauth_token_candidates_preserves_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-generic")
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_macos_keychain_oauth_token", lambda: "gho-keychain")
    monkeypatch.setattr(copilot_auth, "_read_file_oauth_token_candidates", lambda: [])
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)

    candidates = copilot_auth.iter_oauth_token_candidates()

    assert [(candidate.source, candidate.token) for candidate in candidates] == [
        ("macos-keychain:copilot-cli", "gho-keychain"),
        ("env:GITHUB_TOKEN", "ghp-generic"),
    ]


def test_resolve_subscription_bearer_token_skips_invalid_generic_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_API_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_PROVIDER_BEARER_TOKEN", raising=False)
    monkeypatch.setattr(
        copilot_auth, "_subscription_resolution_from_token_exchange", lambda _: None
    )
    monkeypatch.setattr(
        copilot_auth,
        "iter_oauth_token_candidates",
        lambda **_kwargs: [
            copilot_auth.CopilotTokenCandidate(
                token="ghp-generic",
                source="env:GITHUB_TOKEN",
                confidence="generic-github",
            ),
            copilot_auth.CopilotTokenCandidate(
                token="tid_copilot",
                source="macos-keychain:copilot-cli",
                confidence="high",
            ),
        ],
    )
    monkeypatch.setattr(
        copilot_auth,
        "_fetch_copilot_user_info",
        lambda token: (
            {"endpoints": {"api": "https://api.individual.githubcopilot.com"}}
            if token == "tid_copilot"
            else None
        ),
    )

    assert copilot_auth.resolve_subscription_bearer_token() == "tid_copilot"


def test_resolve_subscription_candidates_skips_unvalidated_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        copilot_auth,
        "_subscription_resolution_from_token_exchange",
        lambda candidate: pytest.fail(f"must not exchange {candidate.source}"),
    )

    assert (
        copilot_auth._resolve_subscription_oauth_token_candidates(
            [
                copilot_auth.CopilotTokenCandidate(
                    token="gho-unvalidated",
                    source="untrusted",
                    confidence="low",
                    validate_for_subscription=False,
                )
            ]
        )
        is None
    )


def test_resolve_subscription_bearer_token_does_not_fallback_to_unexchanged_oauth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_API_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_PROVIDER_BEARER_TOKEN", raising=False)
    monkeypatch.setattr(
        copilot_auth, "_subscription_resolution_from_token_exchange", lambda _: None
    )
    monkeypatch.setattr(
        copilot_auth,
        "iter_oauth_token_candidates",
        lambda **_kwargs: [
            copilot_auth.CopilotTokenCandidate(
                token="gho-copilot",
                source="macos-keychain:copilot-cli",
                confidence="high",
            ),
        ],
    )
    monkeypatch.setattr(
        copilot_auth,
        "_fetch_copilot_user_info",
        lambda token: (
            {"endpoints": {"api": "https://api.individual.githubcopilot.com"}}
            if token == "gho-copilot"
            else None
        ),
    )

    assert copilot_auth.resolve_subscription_bearer_token() is None


def test_resolve_subscription_bearer_token_defers_keychain_when_saved_token_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copilot_auth.save_headroom_copilot_oauth_token("gho-saved")
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(
        copilot_auth,
        "_read_macos_keychain_oauth_token",
        lambda: pytest.fail("Keychain must not be read when saved OAuth resolves"),
    )
    monkeypatch.setattr(copilot_auth, "_read_linux_secret_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_file_oauth_token_candidates", lambda: [])
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)
    monkeypatch.setattr(
        copilot_auth,
        "_subscription_resolution_from_token_exchange",
        lambda candidate: (
            copilot_auth._subscription_resolution(
                token="copilot-api",
                source=f"{candidate.source}:token-exchange",
                confidence="copilot-token-exchange",
                api_url=copilot_auth.DEFAULT_API_URL,
                refresh_oauth_token=candidate.token,
            )
            if candidate.token == "gho-saved"
            else None
        ),
    )

    assert copilot_auth.resolve_subscription_bearer_token() == "copilot-api"


def test_resolve_subscription_bearer_token_reads_keychain_after_noninteractive_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copilot_auth.save_headroom_copilot_oauth_token("gho-rejected")
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(
        copilot_auth,
        "_read_macos_keychain_oauth_token",
        lambda: "gho-keychain",
    )
    monkeypatch.setattr(copilot_auth, "_read_linux_secret_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_file_oauth_token_candidates", lambda: [])
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)
    monkeypatch.setattr(
        copilot_auth,
        "_subscription_resolution_from_token_exchange",
        lambda candidate: (
            copilot_auth._subscription_resolution(
                token="copilot-api",
                source=f"{candidate.source}:token-exchange",
                confidence="copilot-token-exchange",
                api_url=copilot_auth.DEFAULT_API_URL,
                refresh_oauth_token=candidate.token,
            )
            if candidate.token == "gho-keychain"
            else None
        ),
    )

    assert copilot_auth.resolve_subscription_bearer_token() == "copilot-api"


def test_resolve_subscription_bearer_token_reads_windows_store_after_noninteractive_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copilot_auth.save_headroom_copilot_oauth_token("gho-rejected")
    monkeypatch.setattr(
        copilot_auth,
        "_read_windows_copilot_cli_oauth_token",
        lambda: "gho-windows",
    )
    monkeypatch.setattr(copilot_auth, "_read_macos_keychain_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_linux_secret_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_file_oauth_token_candidates", lambda: [])
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)
    monkeypatch.setattr(
        copilot_auth,
        "_subscription_resolution_from_token_exchange",
        lambda candidate: (
            copilot_auth._subscription_resolution(
                token="copilot-api",
                source=f"{candidate.source}:token-exchange",
                confidence="copilot-token-exchange",
                api_url=copilot_auth.DEFAULT_API_URL,
                refresh_oauth_token=candidate.token,
            )
            if candidate.token == "gho-windows"
            else None
        ),
    )

    assert copilot_auth.resolve_subscription_bearer_token() == "copilot-api"


def test_resolve_subscription_bearer_token_skips_duplicate_keychain_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copilot_auth.save_headroom_copilot_oauth_token("gho-duplicate")
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(
        copilot_auth,
        "_read_macos_keychain_oauth_token",
        lambda: "gho-duplicate",
    )
    monkeypatch.setattr(copilot_auth, "_read_linux_secret_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_file_oauth_token_candidates", lambda: [])
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)
    attempted_tokens: list[str] = []
    monkeypatch.setattr(
        copilot_auth,
        "_subscription_resolution_from_token_exchange",
        lambda candidate: attempted_tokens.append(candidate.token) or None,
    )

    assert copilot_auth.resolve_subscription_bearer_token() is None
    assert attempted_tokens == ["gho-duplicate"]


def test_subscription_enterprise_host_repro(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_API_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_API_TOKEN_EXPIRES_AT", raising=False)
    monkeypatch.delenv("COPILOT_PROVIDER_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_API_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_ENTERPRISE_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_ENTERPRISE_DOMAIN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN_EXCHANGE_URL", raising=False)
    monkeypatch.setattr(
        copilot_auth,
        "iter_oauth_token_candidates",
        lambda **_kwargs: [
            copilot_auth.CopilotTokenCandidate(
                token="gho-oauth",
                source="headroom-copilot-auth:/tmp/copilot_auth.json",
                confidence="copilot-oauth",
            ),
        ],
    )
    captured: dict[str, str] = {}

    def fake_exchange(headers: dict[str, str]) -> dict[str, object]:
        captured.update(headers)
        return {
            "token": "copilot-api",
            "expires_at": int(time.time()) + 3600,
            "endpoints": {"api": "https://api.enterprise.githubcopilot.com"},
        }

    monkeypatch.setattr(
        copilot_auth.CopilotTokenProvider,
        "_exchange_token_sync",
        staticmethod(fake_exchange),
    )

    resolution = copilot_auth.resolve_subscription_bearer_token_details()

    assert resolution is not None
    assert resolution.token == "copilot-api"
    assert resolution.source == "headroom-copilot-auth:/tmp/copilot_auth.json:token-exchange"
    assert resolution.confidence == "copilot-token-exchange"
    assert resolution.api_url == copilot_auth.DEFAULT_API_URL
    assert resolution.token_fingerprint == copilot_auth.token_fingerprint("copilot-api")
    assert resolution.refresh_oauth_token == "gho-oauth"
    assert isinstance(resolution.api_token_expires_at, float)
    assert captured == {
        "Accept": "application/json",
        "Authorization": "Bearer gho-oauth",
        "User-Agent": "GitHubCopilotChat/0.35.0",
        "Editor-Version": "vscode/1.107.0",
        "Editor-Plugin-Version": "copilot-chat/0.35.0",
        "Copilot-Integration-Id": "vscode-chat",
    }


def test_resolve_subscription_exchange_uses_cloud_enterprise_advertised_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_API_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_PROVIDER_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_API_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_ENTERPRISE_DOMAIN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN_EXCHANGE_URL", raising=False)
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_URL", "github.com/enterprises/cbcrc")
    monkeypatch.setattr(
        copilot_auth,
        "iter_oauth_token_candidates",
        lambda **_kwargs: [
            copilot_auth.CopilotTokenCandidate(
                token="gho-oauth",
                source="env:GITHUB_COPILOT_TOKEN",
                confidence="explicit",
            ),
        ],
    )
    monkeypatch.setattr(
        copilot_auth.CopilotTokenProvider,
        "_exchange_token_sync",
        staticmethod(lambda _headers: {"token": "copilot-api"}),
    )
    monkeypatch.setattr(
        copilot_auth,
        "_fetch_copilot_user_info",
        lambda _token: {"endpoints": {"api": "https://api.enterprise.githubcopilot.com"}},
    )

    resolution = copilot_auth.resolve_subscription_bearer_token_details()

    assert resolution is not None
    assert resolution.api_url == copilot_auth.DEFAULT_API_URL
    assert copilot_auth._token_exchange_url() == "https://api.github.com/copilot_internal/v2/token"


def test_api_url_from_exchange_payload_rejects_non_copilot_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_API_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_ENTERPRISE_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_ENTERPRISE_DOMAIN", raising=False)
    monkeypatch.setattr(
        copilot_auth,
        "_fetch_copilot_user_info",
        lambda _token: {"endpoints": {"api": "https://api.business.githubcopilot.com"}},
    )

    resolved = copilot_auth._api_url_from_exchange_payload(
        {"endpoints": {"api": "https://api.openai.com/v1"}},
        oauth_token="gho-oauth",
    )

    assert resolved == copilot_auth.DEFAULT_API_URL


def _resolve_subscription_producer_path(
    monkeypatch: pytest.MonkeyPatch,
    producer: str,
    payload_host: str,
    configured_api_url: str | None = None,
    enterprise_domain: str | None = None,
) -> str:
    with monkeypatch.context() as patch:
        patch.delenv("GITHUB_COPILOT_API_TOKEN", raising=False)
        patch.delenv("GITHUB_COPILOT_API_URL", raising=False)
        patch.delenv("GITHUB_COPILOT_ENTERPRISE_DOMAIN", raising=False)
        if configured_api_url is not None:
            patch.setenv("GITHUB_COPILOT_API_URL", configured_api_url)
        if enterprise_domain is not None:
            patch.setenv("GITHUB_COPILOT_ENTERPRISE_DOMAIN", enterprise_domain)

        payload = {"endpoints": {"api": payload_host}}
        if producer == "exchange":
            patch.setattr(
                copilot_auth,
                "iter_oauth_token_candidates",
                lambda **_kwargs: [
                    copilot_auth.CopilotTokenCandidate(
                        token="gho-oauth", source="test", confidence="test"
                    )
                ],
            )
            patch.setattr(
                copilot_auth.CopilotTokenProvider,
                "_exchange_token_sync",
                staticmethod(lambda _headers: {"token": "tid-api", **payload}),
            )
        elif producer == "explicit":
            patch.setenv("GITHUB_COPILOT_API_TOKEN", "tid-api")
            patch.setattr(copilot_auth, "_fetch_copilot_user_info", lambda _token: payload)
        else:
            patch.setattr(
                copilot_auth,
                "iter_oauth_token_candidates",
                lambda **_kwargs: [
                    copilot_auth.CopilotTokenCandidate(
                        token="tid_api", source="test", confidence="test"
                    )
                ],
            )
            patch.setattr(copilot_auth, "_fetch_copilot_user_info", lambda _token: payload)

        resolution = copilot_auth.resolve_subscription_bearer_token_details()
        assert resolution is not None
        return resolution.api_url


@pytest.mark.parametrize("producer", ["exchange", "explicit", "candidate"])
def test_subscription_api_url_pin_precedence(
    monkeypatch: pytest.MonkeyPatch, producer: str
) -> None:
    assert (
        _resolve_subscription_producer_path(
            monkeypatch,
            producer,
            "https://api.enterprise.githubcopilot.com",
            configured_api_url="https://api.pinned.example.com",
        )
        == "https://api.pinned.example.com"
    )
    assert (
        _resolve_subscription_producer_path(
            monkeypatch,
            producer,
            "https://api.other.githubcopilot.com",
            configured_api_url=copilot_auth.DEFAULT_API_URL,
        )
        == copilot_auth.DEFAULT_API_URL
    )


@pytest.mark.parametrize("producer", ["exchange", "explicit", "candidate"])
def test_subscription_enterprise_domain_precedence(
    monkeypatch: pytest.MonkeyPatch, producer: str
) -> None:
    assert (
        _resolve_subscription_producer_path(
            monkeypatch,
            producer,
            "https://api.business.githubcopilot.com",
            enterprise_domain="ghe.example.com",
        )
        == "https://copilot-api.ghe.example.com"
    )


def test_subscription_unknown_host_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        copilot_auth._subscription_api_url_from_user_info_payload(
            {"endpoints": {"api": "https://api.other.githubcopilot.com"}}
        )
        == "https://api.other.githubcopilot.com"
    )


@pytest.mark.parametrize(
    "payload_host",
    [
        "https://api.githubcopilot.com",
        "https://api.individual.githubcopilot.com",
        "https://api.business.githubcopilot.com",
        "https://api.enterprise.githubcopilot.com",
    ],
)
def test_subscription_known_hosts_normalize_to_default(payload_host: str) -> None:
    assert (
        copilot_auth._subscription_api_url_from_user_info_payload(
            {"endpoints": {"api": payload_host}}
        )
        == copilot_auth.DEFAULT_API_URL
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"endpoints": {}},
        {"endpoints": {"api": " "}},
        {"endpoints": {"api": 4}},
        {"endpoints": {"api": "https://api.openai.com/v1"}},
    ],
)
def test_subscription_payload_host_fallback(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, object] | None
) -> None:
    assert copilot_auth._subscription_api_url_from_user_info_payload(payload) == (
        copilot_auth.DEFAULT_API_URL
    )


def test_api_url_from_exchange_payload_normalizes_individual_public_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_API_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_ENTERPRISE_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_ENTERPRISE_DOMAIN", raising=False)

    resolved = copilot_auth._api_url_from_exchange_payload(
        {"endpoints": {"api": "https://api.individual.githubcopilot.com"}},
        oauth_token="gho-oauth",
    )

    assert resolved == copilot_auth.DEFAULT_API_URL


def test_api_url_from_exchange_payload_rejects_non_copilot_host_without_user_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_API_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_ENTERPRISE_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_ENTERPRISE_DOMAIN", raising=False)
    monkeypatch.setattr(copilot_auth, "_fetch_copilot_user_info", lambda _token: None)

    resolved = copilot_auth._api_url_from_exchange_payload(
        {"endpoints": {"api": "https://api.openai.com/v1"}},
        oauth_token="gho-oauth",
    )

    assert resolved == copilot_auth.DEFAULT_API_URL


def test_enterprise_domain_routes_token_exchange_and_user_info_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN_EXCHANGE_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_USER_INFO_URL", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_ENTERPRISE_URL", raising=False)
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_DOMAIN", "ghe.example.com")

    assert (
        copilot_auth._token_exchange_url()
        == "https://api.ghe.example.com/copilot_internal/v2/token"
    )
    assert copilot_auth._user_info_url() == "https://api.ghe.example.com/copilot_internal/user"


def test_user_info_url_override_wins_over_enterprise_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_DOMAIN", "ghe.example.com")
    monkeypatch.setenv(
        "GITHUB_COPILOT_USER_INFO_URL",
        "https://custom.example.com/copilot_internal/user",
    )

    assert copilot_auth._user_info_url() == "https://custom.example.com/copilot_internal/user"


def test_copilot_api_url_from_enterprise_url_supports_enterprise_server_domain() -> None:
    assert (
        copilot_auth.copilot_api_url_from_enterprise_url("https://ghe.example.com/")
        == "https://copilot-api.ghe.example.com"
    )
    assert (
        copilot_auth.copilot_api_url_from_enterprise_url("https://api.ghe.example.com/")
        == "https://copilot-api.ghe.example.com"
    )
    assert (
        copilot_auth.copilot_api_url_from_enterprise_url("https://copilot-api.ghe.example.com/")
        == "https://copilot-api.ghe.example.com"
    )


def test_copilot_api_url_from_enterprise_url_ignores_github_cloud_enterprise_path() -> None:
    assert (
        copilot_auth.copilot_api_url_from_enterprise_url("https://github.com/enterprises/cbcrc/")
        == copilot_auth.DEFAULT_API_URL
    )


def test_should_exchange_oauth_token_supports_truthy_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for raw in ("1", "true", "YES", "On"):
        monkeypatch.setenv("GITHUB_COPILOT_USE_TOKEN_EXCHANGE", raw)
        assert copilot_auth._should_exchange_oauth_token() is True

    monkeypatch.setenv("GITHUB_COPILOT_USE_TOKEN_EXCHANGE", "off")
    assert copilot_auth._should_exchange_oauth_token() is False


def test_resolve_token_file_paths_prefers_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_FILE", "~/custom-token.json")

    paths = copilot_auth._resolve_token_file_paths()

    assert paths == [Path("~/custom-token.json").expanduser()]


def test_resolve_token_file_paths_includes_localappdata_and_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN_FILE", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(copilot_auth.Path, "home", staticmethod(lambda: tmp_path / "home"))

    paths = copilot_auth._resolve_token_file_paths()

    assert paths == [
        tmp_path / "local" / "github-copilot" / "apps.json",
        tmp_path / "local" / "github-copilot" / "hosts.json",
        tmp_path / "home" / ".config" / "github-copilot" / "apps.json",
        tmp_path / "home" / ".config" / "github-copilot" / "hosts.json",
    ]


def test_read_cached_oauth_token_falls_back_to_gh_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_macos_keychain_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: "gho-gh-cli")
    # No cached token files — a developer machine may have a real
    # ~/.config/github-copilot token that would win before the gh fallback.
    monkeypatch.setattr(copilot_auth, "_resolve_token_file_paths", lambda: [])

    assert copilot_auth.read_cached_oauth_token() == "gho-gh-cli"


def test_read_cached_oauth_token_prefers_copilot_cli_windows_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(
        copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: "gho-copilot"
    )
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: "gho-gh-cli")

    assert copilot_auth.read_cached_oauth_token() == "gho-copilot"


def test_read_cached_oauth_token_prefers_macos_keychain_before_gh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_macos_keychain_oauth_token", lambda: "gho-keychain")
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: "gho-gh-cli")

    assert copilot_auth.read_cached_oauth_token() == "gho-keychain"


def test_read_macos_keychain_oauth_token_uses_security(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_read(*, host: str) -> str:
        calls.append(host)
        return "gho-keychain"

    monkeypatch.setattr(copilot_auth, "read_macos_keychain_token", fake_read)
    assert copilot_auth._read_macos_keychain_oauth_token() == "gho-keychain"
    assert calls == ["github.com"]


def test_keychain_and_secret_service_use_derived_host(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_macos(*, host: str) -> None:
        calls.append(("macos", host))
        return None

    def fake_linux(*, host: str) -> None:
        calls.append(("linux", host))
        return None

    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://ghe.example.com/api")
    monkeypatch.setattr(copilot_auth, "read_macos_keychain_token", fake_macos)
    monkeypatch.setattr(copilot_auth, "read_linux_secret_token", fake_linux)
    assert copilot_auth._read_macos_keychain_oauth_token() is None
    assert copilot_auth._read_linux_secret_oauth_token() is None
    assert calls == [("macos", "ghe.example.com"), ("linux", "ghe.example.com")]


def test_read_cached_oauth_token_reads_hosts_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hosts = tmp_path / "hosts.json"
    hosts.write_text(
        json.dumps(
            {
                "github.com": {
                    "oauth_token": "gho-file",
                    "expires_at": "2999-01-01T00:00:00Z",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_FILE", str(hosts))
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_macos_keychain_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)

    assert copilot_auth.read_cached_oauth_token() == "gho-file"


def test_read_cached_oauth_token_reads_custom_api_host_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hosts = tmp_path / "hosts.json"
    hosts.write_text(
        json.dumps(
            {
                "ghe.example.com": {"oauth_token": "gho-ghe"},
                "adjacent.example.com": {"oauth_token": "gho-adjacent"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://api.ghe.example.com:8443/copilot")
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_FILE", str(hosts))
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_macos_keychain_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)

    candidates = copilot_auth._read_file_oauth_token_candidates()
    assert [candidate.token for candidate in candidates] == ["gho-ghe"]
    assert copilot_auth.read_cached_oauth_token() == "gho-ghe"


def test_read_cached_oauth_token_skips_expired_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hosts = tmp_path / "hosts.json"
    hosts.write_text(
        json.dumps({"github.com": {"oauthToken": "gho-old", "expiresAt": 1}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_FILE", str(hosts))
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_macos_keychain_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)

    assert copilot_auth.read_cached_oauth_token() is None


def test_read_gh_cli_oauth_token_uses_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class CompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = "gho-gh-cli\n"

    def fake_run(*args: object, **kwargs: object) -> CompletedProcess:
        calls.append(list(args[0]))
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        return CompletedProcess()

    monkeypatch.setenv("GITHUB_COPILOT_HOST", "example.ghe.com")
    monkeypatch.setattr(copilot_auth, "run", fake_run)

    assert copilot_auth._read_gh_cli_oauth_token() == "gho-gh-cli"
    assert calls == [["gh", "auth", "token", "--hostname", "example.ghe.com"]]


def test_read_gh_cli_oauth_token_uses_api_url_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class CompletedProcess:
        returncode = 0
        stdout = "gho-gh-cli\n"

    def fake_run(*args: object, **kwargs: object) -> CompletedProcess:
        calls.append(list(args[0]))
        return CompletedProcess()

    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://api.ghe.example.com/api")
    monkeypatch.setattr(copilot_auth, "run", fake_run)
    assert copilot_auth._read_gh_cli_oauth_token() == "gho-gh-cli"
    assert calls == [["gh", "auth", "token", "--hostname", "ghe.example.com"]]


def test_read_gh_cli_oauth_token_returns_none_when_invocation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:  # noqa: ANN002, ANN003
        raise OSError("gh missing")

    monkeypatch.setattr(copilot_auth, "run", fake_run)

    assert copilot_auth._read_gh_cli_oauth_token() is None


def test_read_gh_cli_oauth_token_returns_none_for_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        copilot_auth,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="ignored"),
    )

    assert copilot_auth._read_gh_cli_oauth_token() is None


def test_read_gh_cli_oauth_token_returns_none_for_blank_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        copilot_auth,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=" \n"),
    )

    assert copilot_auth._read_gh_cli_oauth_token() is None


def test_resolve_client_bearer_token_prefers_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "copilot-api")
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-oauth")

    assert copilot_auth.resolve_client_bearer_token() == "copilot-api"


def test_has_oauth_auth_false_when_no_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(copilot_auth, "resolve_client_bearer_token", lambda: None)

    assert copilot_auth.has_oauth_auth() is False


def test_is_copilot_api_url_matches_expected_hosts() -> None:
    assert copilot_auth.is_copilot_api_url("https://api.githubcopilot.com/v1/chat/completions")
    assert copilot_auth.is_copilot_api_url("wss://api.githubcopilot.com/v1/responses")
    assert not copilot_auth.is_copilot_api_url("https://api.openai.com/v1/chat/completions")


def test_is_copilot_api_url_matches_ghe_copilot_hosts() -> None:
    assert copilot_auth.is_copilot_api_url("https://copilot-api.acme.ghe.com/v1/responses")
    assert copilot_auth.is_copilot_api_url("https://copilot-api.ghe.com/v1/chat/completions")
    assert not copilot_auth.is_copilot_api_url("https://api.acme.ghe.com/v1/responses")
    assert not copilot_auth.is_copilot_api_url("https://not-copilot-api.acme.ghe.com/v1/responses")


def test_is_copilot_api_url_trusts_configured_enterprise_api_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_DOMAIN", "ghe.example.com")

    assert copilot_auth.is_copilot_api_url("https://copilot-api.ghe.example.com/v1/responses")
    assert not copilot_auth.is_copilot_api_url("https://copilot-api.other.example.com/v1/responses")


def test_build_copilot_upstream_url_strips_v1_only_for_copilot_hosts() -> None:
    assert (
        copilot_auth.build_copilot_upstream_url(
            "https://api.githubcopilot.com",
            "/v1/chat/completions",
        )
        == "https://api.githubcopilot.com/chat/completions"
    )
    assert (
        copilot_auth.build_copilot_upstream_url(
            "https://api.openai.com",
            "/v1/chat/completions",
        )
        == "https://api.openai.com/v1/chat/completions"
    )


def test_build_copilot_upstream_url_preserves_v1_messages_for_copilot() -> None:
    # Copilot's Anthropic surface for Claude models is /v1/messages (with the
    # /v1); stripping it forwarded /messages and Copilot 404'd (#2409).
    assert (
        copilot_auth.build_copilot_upstream_url(
            "https://api.githubcopilot.com",
            "/v1/messages",
        )
        == "https://api.githubcopilot.com/v1/messages"
    )
    # Batches under the messages endpoint keep /v1 too.
    assert (
        copilot_auth.build_copilot_upstream_url(
            "https://api.githubcopilot.com",
            "/v1/messages/batches",
        )
        == "https://api.githubcopilot.com/v1/messages/batches"
    )
    # A GHE Copilot host keeps /v1/messages as well.
    assert (
        copilot_auth.build_copilot_upstream_url(
            "https://copilot-api.acme.ghe.com",
            "/v1/messages",
        )
        == "https://copilot-api.acme.ghe.com/v1/messages"
    )


def test_build_copilot_upstream_url_strips_v1_for_ghe_copilot_hosts() -> None:
    assert (
        copilot_auth.build_copilot_upstream_url(
            "https://copilot-api.acme.ghe.com",
            "/v1/responses",
        )
        == "https://copilot-api.acme.ghe.com/responses"
    )
    assert (
        copilot_auth.build_copilot_upstream_url(
            "https://api.acme.ghe.com",
            "/v1/responses",
        )
        == "https://api.acme.ghe.com/v1/responses"
    )


def test_build_copilot_upstream_url_strips_v1_for_configured_enterprise_api_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_DOMAIN", "ghe.example.com")

    assert (
        copilot_auth.build_copilot_upstream_url(
            "https://copilot-api.ghe.example.com",
            "/v1/responses",
        )
        == "https://copilot-api.ghe.example.com/responses"
    )


def test_apply_copilot_api_auth_replaces_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_api_token(
        *, integration_id: str | None = None
    ) -> copilot_auth.CopilotAPIToken:
        return copilot_auth.CopilotAPIToken(
            token="copilot-session",
            expires_at=time.time() + 3600,
            api_url=copilot_auth.DEFAULT_API_URL,
        )

    monkeypatch.setattr(
        copilot_auth.get_copilot_token_provider(),
        "get_api_token",
        fake_get_api_token,
    )

    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {"authorization": "Bearer downstream-token", "x-api-key": "sk-downstream"},
            url="https://api.githubcopilot.com/v1/chat/completions",
        )
    )

    assert headers["Authorization"] == "Bearer copilot-session"
    assert "authorization" not in headers
    assert "x-api-key" not in headers
    assert headers["User-Agent"] == "GitHubCopilotChat/0.35.0"
    assert headers["Editor-Version"] == "vscode/1.107.0"
    assert headers["Editor-Plugin-Version"] == "copilot-chat/0.35.0"
    assert headers["Copilot-Integration-Id"] == "vscode-chat"


def test_apply_copilot_api_auth_passes_through_existing_api_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_api_token(
        *, integration_id: str | None = None
    ) -> copilot_auth.CopilotAPIToken:
        raise AssertionError("provider should not be called for existing API token")

    monkeypatch.setattr(
        copilot_auth.get_copilot_token_provider(),
        "get_api_token",
        fake_get_api_token,
    )

    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {
                "authorization": "Bearer tid_existing_copilot_token",
                "x-api-key": "sk-downstream",
            },
            url="https://api.githubcopilot.com/v1/chat/completions",
        )
    )

    assert headers["authorization"] == "Bearer tid_existing_copilot_token"
    assert "x-api-key" not in headers


def test_apply_copilot_api_auth_replaces_managed_seeded_api_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_api_token(
        *, integration_id: str | None = None
    ) -> copilot_auth.CopilotAPIToken:
        return copilot_auth.CopilotAPIToken(
            token="copilot-refreshed",
            expires_at=time.time() + 3600,
            api_url=copilot_auth.DEFAULT_API_URL,
        )

    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "tid_existing_copilot_token")
    monkeypatch.setenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", "gho-refresh")
    monkeypatch.setattr(
        copilot_auth.get_copilot_token_provider(),
        "get_api_token",
        fake_get_api_token,
    )

    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {
                "authorization": "Bearer tid_existing_copilot_token",
                "x-api-key": "sk-downstream",
            },
            url="https://api.githubcopilot.com/v1/chat/completions",
        )
    )

    assert headers["Authorization"] == "Bearer copilot-refreshed"
    assert "authorization" not in headers
    assert "x-api-key" not in headers


def test_apply_copilot_api_auth_passes_through_github_oauth_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-supplied GitHub OAuth (gho_/ghs_/ghp_/github_pat_) bearer
    token must be forwarded unchanged, not replaced.

    Regression test for headroomlabs-ai/headroom#1813: this function used
    to treat any gho_-prefixed token as "not a suitable Copilot API
    token" and silently replace it with Headroom's own independently
    fetched/exchanged credential. That broke both:
    - a live Copilot CLI session (its own gho_ token worked directly for
      model "claude-sonnet-5", but got 400 model_not_supported once
      Headroom substituted a differently-entitled token), and
    - OpenCode's native GitHub Copilot integration (#1813): replacing its
      gho_ token changed the effective client/integrator lane Copilot's
      backend sees, breaking model discovery/inference parity with
      native (non-proxied) behavior.
    """

    async def fail_if_called() -> copilot_auth.CopilotAPIToken:
        raise AssertionError(
            "get_api_token() must not be called for a forwardable gho_ bearer token"
        )

    monkeypatch.setattr(
        copilot_auth.get_copilot_token_provider(),
        "get_api_token",
        fail_if_called,
    )

    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {"authorization": "Bearer gho_liveClientToken123"},
            url="https://api.githubcopilot.com/v1/chat/completions",
        )
    )

    assert headers["authorization"] == "Bearer gho_liveClientToken123"


def test_apply_copilot_api_auth_replaces_non_bearer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_api_token(
        *, integration_id: str | None = None
    ) -> copilot_auth.CopilotAPIToken:
        return copilot_auth.CopilotAPIToken(
            token="copilot-session",
            expires_at=time.time() + 3600,
            api_url=copilot_auth.DEFAULT_API_URL,
        )

    monkeypatch.setattr(
        copilot_auth.get_copilot_token_provider(),
        "get_api_token",
        fake_get_api_token,
    )

    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {"authorization": "Basic abc123"},
            url="https://api.githubcopilot.com/v1/chat/completions",
        )
    )

    assert headers["Authorization"] == "Bearer copilot-session"
    assert "authorization" not in headers


def test_is_copilot_api_token_matches_expected_prefixes() -> None:
    assert copilot_auth._is_copilot_api_token("tid_session_token") is True
    assert copilot_auth._is_copilot_api_token("gho_oauth") is False
    assert copilot_auth._is_copilot_api_token("ghs_oauth") is False
    assert copilot_auth._is_copilot_api_token("ghp_oauth") is False
    assert copilot_auth._is_copilot_api_token("github_pat_example") is False
    assert copilot_auth._is_copilot_api_token("Bearer maybe") is False


def test_is_forwardable_copilot_bearer_token_matches_expected_prefixes() -> None:
    """The inference-path helper (unlike _is_copilot_api_token, which is
    scoped only to subscription/user-info resolution) accepts BOTH
    short-lived Copilot API tokens (tid_) AND GitHub OAuth tokens
    (gho_/ghs_/ghp_/github_pat_) as forwardable -- see
    _is_forwardable_copilot_bearer_token()'s docstring and
    headroomlabs-ai/headroom#1813 for why GitHub OAuth tokens must be
    forwardable for chat-completion/inference requests.
    """
    assert copilot_auth._is_forwardable_copilot_bearer_token("tid_session_token") is True
    assert copilot_auth._is_forwardable_copilot_bearer_token("gho_oauth") is True
    assert copilot_auth._is_forwardable_copilot_bearer_token("ghs_oauth") is True
    assert copilot_auth._is_forwardable_copilot_bearer_token("ghp_oauth") is True
    assert copilot_auth._is_forwardable_copilot_bearer_token("github_pat_example") is True
    assert copilot_auth._is_forwardable_copilot_bearer_token("sk-unrelated-anthropic-key") is False
    assert copilot_auth._is_forwardable_copilot_bearer_token("") is False
    assert copilot_auth._is_forwardable_copilot_bearer_token("   ") is False


def test_apply_copilot_api_auth_injects_required_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_api_token(
        *, integration_id: str | None = None
    ) -> copilot_auth.CopilotAPIToken:
        return copilot_auth.CopilotAPIToken(
            token="copilot-session",
            expires_at=time.time() + 3600,
            api_url=copilot_auth.DEFAULT_API_URL,
        )

    monkeypatch.setattr(
        copilot_auth.get_copilot_token_provider(),
        "get_api_token",
        fake_get_api_token,
    )
    monkeypatch.delenv("GITHUB_COPILOT_INTEGRATION_ID", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_EDITOR_VERSION", raising=False)

    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {},
            url="https://api.githubcopilot.com/v1/chat/completions",
        )
    )

    assert headers["Authorization"] == "Bearer copilot-session"
    assert headers["Copilot-Integration-Id"] == "vscode-chat"
    assert headers["Editor-Version"] == "vscode/1.107.0"
    assert headers["Editor-Plugin-Version"] == "copilot-chat/0.35.0"


def test_apply_copilot_api_auth_preserves_existing_copilot_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_api_token(
        *, integration_id: str | None = None
    ) -> copilot_auth.CopilotAPIToken:
        return copilot_auth.CopilotAPIToken(
            token="copilot-session",
            expires_at=time.time() + 3600,
            api_url=copilot_auth.DEFAULT_API_URL,
        )

    monkeypatch.setattr(
        copilot_auth.get_copilot_token_provider(),
        "get_api_token",
        fake_get_api_token,
    )
    monkeypatch.setenv("GITHUB_COPILOT_INTEGRATION_ID", "should-not-override")
    monkeypatch.setenv("GITHUB_COPILOT_EDITOR_VERSION", "should-not-override")

    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {
                "Authorization": "Bearer downstream-token",
                "Copilot-Integration-Id": "custom-integration",
                "Editor-Version": "custom-editor",
            },
            url="https://api.githubcopilot.com/v1/chat/completions",
        )
    )

    assert headers["Copilot-Integration-Id"] == "custom-integration"
    assert headers["Editor-Version"] == "custom-editor"
    assert headers["Authorization"] == "Bearer copilot-session"


def test_apply_copilot_api_auth_preserves_existing_headers_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_api_token(
        *, integration_id: str | None = None
    ) -> copilot_auth.CopilotAPIToken:
        return copilot_auth.CopilotAPIToken(
            token="copilot-session",
            expires_at=time.time() + 3600,
            api_url=copilot_auth.DEFAULT_API_URL,
        )

    monkeypatch.setattr(
        copilot_auth.get_copilot_token_provider(),
        "get_api_token",
        fake_get_api_token,
    )

    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {
                "authorization": "Bearer downstream-token",
                "user-agent": "custom-agent",
                "editor-version": "custom-editor",
                "editor-plugin-version": "custom-plugin",
                "copilot-integration-id": "custom-integration",
            },
            url="https://api.githubcopilot.com/v1/chat/completions",
        )
    )

    assert headers["Authorization"] == "Bearer copilot-session"
    assert "authorization" not in headers
    assert headers["user-agent"] == "custom-agent"
    assert headers["editor-version"] == "custom-editor"
    assert headers["editor-plugin-version"] == "custom-plugin"
    assert headers["copilot-integration-id"] == "custom-integration"
    assert "User-Agent" not in headers
    assert "Editor-Version" not in headers
    assert "Editor-Plugin-Version" not in headers
    assert "Copilot-Integration-Id" not in headers


def test_token_provider_reuses_oauth_token_without_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-oauth")

    provider = copilot_auth.CopilotTokenProvider()
    calls = {"count": 0}

    def fake_exchange(headers: dict[str, str]) -> dict[str, object]:
        calls["count"] += 1
        return {
            "token": "copilot-api",
            "expires_at": int(time.time()) + 3600,
            "refresh_in": 1200,
            "endpoints": {"api": "https://api.githubcopilot.com"},
            "sku": "copilot_individual",
        }

    monkeypatch.setattr(provider, "_exchange_token_sync", staticmethod(fake_exchange))

    first = asyncio.run(provider.get_api_token())
    second = asyncio.run(provider.get_api_token())

    assert first.token == "gho-oauth"
    assert second.token == "gho-oauth"
    assert calls["count"] == 0


def test_token_provider_can_exchange_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-oauth")
    monkeypatch.setenv("GITHUB_COPILOT_USE_TOKEN_EXCHANGE", "true")

    provider = copilot_auth.CopilotTokenProvider()
    calls = {"count": 0}
    captured: dict[str, str] = {}

    def fake_exchange(headers: dict[str, str]) -> dict[str, object]:
        calls["count"] += 1
        captured.update(headers)
        return {
            "token": "copilot-api",
            "expires_at": int(time.time()) + 3600,
            "refresh_in": 1200,
            "endpoints": {"api": "https://api.githubcopilot.com"},
            "sku": "copilot_individual",
        }

    monkeypatch.setattr(provider, "_exchange_token_sync", staticmethod(fake_exchange))

    first = asyncio.run(provider.get_api_token())
    second = asyncio.run(provider.get_api_token())

    assert first.token == "copilot-api"
    assert second.token == "copilot-api"
    assert calls["count"] == 1
    assert captured["Authorization"] == "Bearer gho-oauth"
    assert captured["User-Agent"] == "GitHubCopilotChat/0.35.0"
    assert captured["Editor-Version"] == "vscode/1.107.0"
    assert captured["Editor-Plugin-Version"] == "copilot-chat/0.35.0"
    assert captured["Copilot-Integration-Id"] == "vscode-chat"


def test_token_provider_prefers_explicit_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "copilot-api")
    monkeypatch.delenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_API_TOKEN_EXPIRES_AT", raising=False)
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://api.githubcopilot.com")

    token = asyncio.run(copilot_auth.CopilotTokenProvider().get_api_token())

    assert token.token == "copilot-api"
    assert token.api_url == "https://api.githubcopilot.com"


def test_token_provider_raises_without_oauth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_API_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_API_TOKEN_EXPIRES_AT", raising=False)
    monkeypatch.setattr(copilot_auth, "read_cached_oauth_token", lambda: None)

    with pytest.raises(RuntimeError, match="No GitHub Copilot OAuth token"):
        asyncio.run(copilot_auth.CopilotTokenProvider().get_api_token())


def test_exchange_token_raises_when_exchange_returns_empty_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = copilot_auth.CopilotTokenProvider()
    monkeypatch.setattr(
        provider,
        "_exchange_token_sync",
        staticmethod(lambda headers: {"token": "", "expires_at": int(time.time()) + 1}),
    )

    with pytest.raises(RuntimeError, match="empty token"):
        asyncio.run(provider._exchange_token("gho-oauth"))


def test_token_provider_refreshes_expired_seeded_explicit_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "copilot-expired")
    monkeypatch.setenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", "gho-refresh")
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN_EXPIRES_AT", str(time.time() - 120))
    monkeypatch.delenv("GITHUB_COPILOT_USE_TOKEN_EXCHANGE", raising=False)

    provider = copilot_auth.CopilotTokenProvider()
    calls = {"count": 0}
    captured: dict[str, str] = {}

    def fake_exchange(headers: dict[str, str]) -> dict[str, object]:
        calls["count"] += 1
        captured.update(headers)
        return {
            "token": "copilot-refreshed",
            "expires_at": int(time.time()) + 3600,
            "endpoints": {"api": "https://api.business.githubcopilot.com"},
        }

    monkeypatch.setattr(provider, "_exchange_token_sync", staticmethod(fake_exchange))

    token = asyncio.run(provider.get_api_token())

    assert token.token == "copilot-refreshed"
    assert captured["Authorization"] == "Bearer gho-refresh"
    assert calls["count"] == 1


def test_token_provider_reuses_valid_seeded_explicit_token_without_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "copilot-valid")
    monkeypatch.setenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", "gho-refresh")
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN_EXPIRES_AT", str(time.time() + 3600))

    provider = copilot_auth.CopilotTokenProvider()
    calls = {"count": 0}

    def fake_exchange(_headers: dict[str, str]) -> dict[str, object]:
        calls["count"] += 1
        return {
            "token": "copilot-refreshed",
            "expires_at": int(time.time()) + 3600,
            "endpoints": {"api": "https://api.githubcopilot.com"},
        }

    monkeypatch.setattr(provider, "_exchange_token_sync", staticmethod(fake_exchange))

    first = asyncio.run(provider.get_api_token())
    second = asyncio.run(provider.get_api_token())

    assert first.token == "copilot-valid"
    assert second.token == "copilot-valid"
    assert calls["count"] == 0


def test_token_provider_refreshes_seeded_token_when_expiry_is_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "copilot-expired")
    monkeypatch.setenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", "gho-refresh")
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN_EXPIRES_AT", "inf")

    provider = copilot_auth.CopilotTokenProvider()
    calls = {"count": 0}

    def fake_exchange(_headers: dict[str, str]) -> dict[str, object]:
        calls["count"] += 1
        return {
            "token": "copilot-refreshed",
            "expires_at": int(time.time()) + 3600,
            "endpoints": {"api": "https://api.githubcopilot.com"},
        }

    monkeypatch.setattr(provider, "_exchange_token_sync", staticmethod(fake_exchange))

    token = asyncio.run(provider.get_api_token())

    assert token.token == "copilot-refreshed"
    assert calls["count"] == 1


def test_token_provider_refreshes_seeded_token_even_when_exchange_flag_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "copilot-expired")
    monkeypatch.setenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", "gho-refresh")
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN_EXPIRES_AT", str(time.time() - 120))
    monkeypatch.setenv("GITHUB_COPILOT_USE_TOKEN_EXCHANGE", "false")

    provider = copilot_auth.CopilotTokenProvider()
    calls = {"count": 0}

    def fake_exchange(_headers: dict[str, str]) -> dict[str, object]:
        calls["count"] += 1
        return {
            "token": "copilot-refreshed",
            "expires_at": int(time.time()) + 3600,
            "endpoints": {"api": "https://api.githubcopilot.com"},
        }

    monkeypatch.setattr(provider, "_exchange_token_sync", staticmethod(fake_exchange))

    token = asyncio.run(provider.get_api_token())

    assert token.token == "copilot-refreshed"
    assert calls["count"] == 1


def test_token_provider_preserves_configured_api_url_when_refreshing_seeded_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "copilot-expired")
    monkeypatch.setenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", "gho-refresh")
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN_EXPIRES_AT", str(time.time() - 120))
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://proxy.internal.example.com")

    provider = copilot_auth.CopilotTokenProvider()

    monkeypatch.setattr(
        provider,
        "_exchange_token_sync",
        staticmethod(
            lambda _headers: {
                "token": "copilot-refreshed",
                "expires_at": int(time.time()) + 3600,
                "endpoints": {"api": "https://api.other.githubcopilot.com"},
            }
        ),
    )

    token = asyncio.run(provider.get_api_token())

    assert token.token == "copilot-refreshed"
    assert token.api_url == "https://proxy.internal.example.com"


def test_exchange_token_sync_raises_for_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        def read(self) -> bytes:
            return b'{"message":"Not Found"}'

        def close(self) -> None:
            return None

    def fake_urlopen(request, timeout: float):  # noqa: ANN001, ANN202
        raise urllib_error.HTTPError(
            url=request.full_url,
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=DummyResponse(),
        )

    monkeypatch.setattr(copilot_auth.urllib_request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 404"):
        copilot_auth.CopilotTokenProvider._exchange_token_sync({"Authorization": "token test"})


def test_apply_copilot_api_auth_returns_original_headers_for_non_copilot_url() -> None:
    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {"authorization": "Bearer downstream-token"},
            url="https://api.openai.com/v1/chat/completions",
        )
    )

    assert headers == {"authorization": "Bearer downstream-token"}


def test_read_windows_copilot_cli_oauth_token_returns_none_without_windll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(copilot_auth.os, "name", "nt")
    monkeypatch.delattr(copilot_auth.ctypes, "WinDLL", raising=False)

    assert copilot_auth._read_windows_copilot_cli_oauth_token() is None


def test_is_copilot_api_token_returns_false_for_empty_string() -> None:
    assert copilot_auth._is_copilot_api_token("") is False
    assert copilot_auth._is_copilot_api_token("   ") is False


def test_token_kind_returns_known_prefixes() -> None:
    assert copilot_auth._token_kind("tid_x") == "tid_***"  # noqa: S105
    assert copilot_auth._token_kind("gho_x") == "gho_***"  # noqa: S105
    assert copilot_auth._token_kind("ghs_x") == "ghs_***"  # noqa: S105
    assert copilot_auth._token_kind("ghp_x") == "ghp_***"  # noqa: S105
    assert copilot_auth._token_kind("github_pat_x") == "github_pat_***"  # noqa: S105


def test_token_kind_returns_unknown_for_unrecognised_token() -> None:
    assert copilot_auth._token_kind("some_random_token") == "unknown"


def test_token_kind_returns_empty_for_blank_token() -> None:
    assert copilot_auth._token_kind("") == "empty"
    assert copilot_auth._token_kind("   ") == "empty"


def test_exchange_token_sync_returns_payload_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"token": "copilot-api", "expires_at": int(time.time()) + 3600}

    class FakeResponse:
        def read(self) -> bytes:
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        copilot_auth.urllib_request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    result = copilot_auth.CopilotTokenProvider._exchange_token_sync(
        {"Authorization": "Bearer gho_test"}  # noqa: S105
    )

    assert result == payload


def test_exchange_token_sync_uses_configured_corporate_tls_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copilot refresh must use the same corporate trust config as upstream I/O."""
    payload = {"token": "copilot-api", "expires_at": int(time.time()) + 3600}
    tls_context = object()
    captured: dict[str, object] = {}

    class FakeResponse:
        def read(self) -> bytes:
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(ssl_context, "build_urlopen_context", lambda: tls_context)
    monkeypatch.setattr(copilot_auth.urllib_request, "urlopen", fake_urlopen)

    result = copilot_auth.CopilotTokenProvider._exchange_token_sync(
        {"Authorization": "Bearer gho_test"}  # noqa: S105
    )

    assert result == payload
    assert captured["context"] is tls_context


def test_iter_oauth_token_candidates_includes_linux_secret_service_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_macos_keychain_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_linux_secret_oauth_token", lambda: "gho-linux")
    monkeypatch.setattr(copilot_auth, "_read_file_oauth_token_candidates", lambda: [])
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)

    candidates = copilot_auth.iter_oauth_token_candidates()

    assert [(candidate.source, candidate.token) for candidate in candidates] == [
        ("linux-secret-service:copilot-cli", "gho-linux"),
    ]


def test_iter_oauth_token_candidates_skips_platform_secret_stores_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(
        copilot_auth,
        "_read_windows_copilot_cli_oauth_token",
        lambda: pytest.fail("Windows Credential Manager should not be read"),
    )
    monkeypatch.setattr(
        copilot_auth,
        "_platform_secret_store_oauth_token_candidates",
        lambda: pytest.fail("platform secret stores should not be read"),
    )
    monkeypatch.setattr(copilot_auth, "_read_file_oauth_token_candidates", lambda: [])
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)

    assert copilot_auth.iter_oauth_token_candidates(include_platform_secret_stores=False) == []


def test_iter_oauth_token_candidates_reads_windows_store_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_var in (
        "GITHUB_COPILOT_GITHUB_TOKEN",
        "GITHUB_COPILOT_TOKEN",
        "COPILOT_GITHUB_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(
        copilot_auth,
        "_read_windows_copilot_cli_oauth_token",
        lambda: "gho-windows",
    )
    monkeypatch.setattr(copilot_auth, "_read_macos_keychain_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_linux_secret_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_file_oauth_token_candidates", lambda: [])
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)

    candidates = copilot_auth.iter_oauth_token_candidates()

    assert [(candidate.source, candidate.token) for candidate in candidates] == [
        ("windows-credential-manager:copilot-cli", "gho-windows"),
    ]


def test_platform_secret_store_candidates_include_macos_keychain_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_macos_keychain_oauth_token", lambda: "gho-macos")
    monkeypatch.setattr(copilot_auth, "_read_linux_secret_oauth_token", lambda: None)

    candidates = copilot_auth._platform_secret_store_oauth_token_candidates()

    assert [(candidate.source, candidate.token) for candidate in candidates] == [
        ("macos-keychain:copilot-cli", "gho-macos"),
    ]
