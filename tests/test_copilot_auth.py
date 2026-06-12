from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from urllib import error as urllib_error

import pytest

from headroom import copilot_auth


def test_read_cached_oauth_token_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-env")
    assert copilot_auth.read_cached_oauth_token() == "gho-env"


def test_read_cached_oauth_token_prefers_copilot_cli_before_generic_github_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-generic")
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
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
        copilot_auth,
        "iter_oauth_token_candidates",
        lambda: [
            copilot_auth.CopilotTokenCandidate(
                token="ghp-generic",
                source="env:GITHUB_TOKEN",
                confidence="generic-github",
            ),
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

    assert copilot_auth.resolve_subscription_bearer_token() == "gho-copilot"


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
    monkeypatch.setattr(copilot_auth.subprocess, "run", fake_run)

    assert copilot_auth._read_gh_cli_oauth_token() == "gho-gh-cli"
    assert calls == [["gh", "auth", "token", "--hostname", "example.ghe.com"]]


def test_read_gh_cli_oauth_token_returns_none_when_invocation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:  # noqa: ANN002, ANN003
        raise OSError("gh missing")

    monkeypatch.setattr(copilot_auth.subprocess, "run", fake_run)

    assert copilot_auth._read_gh_cli_oauth_token() is None


def test_read_gh_cli_oauth_token_returns_none_for_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        copilot_auth.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="ignored"),
    )

    assert copilot_auth._read_gh_cli_oauth_token() is None


def test_read_gh_cli_oauth_token_returns_none_for_blank_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        copilot_auth.subprocess,
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


def test_apply_copilot_api_auth_replaces_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_api_token() -> copilot_auth.CopilotAPIToken:
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
            {"authorization": "Bearer downstream-token"},
            url="https://api.githubcopilot.com/v1/chat/completions",
        )
    )

    assert headers["Authorization"] == "Bearer copilot-session"
    assert "authorization" not in headers


def test_apply_copilot_api_auth_passes_through_existing_api_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_api_token() -> copilot_auth.CopilotAPIToken:
        raise AssertionError("provider should not be called for existing API token")

    monkeypatch.setattr(
        copilot_auth.get_copilot_token_provider(),
        "get_api_token",
        fake_get_api_token,
    )

    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {"authorization": "Bearer tid_existing_copilot_token"},
            url="https://api.githubcopilot.com/v1/chat/completions",
        )
    )

    assert headers["authorization"] == "Bearer tid_existing_copilot_token"


def test_apply_copilot_api_auth_replaces_github_oauth_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_api_token() -> copilot_auth.CopilotAPIToken:
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
            {"authorization": "Bearer gho_downstream_oauth"},
            url="https://api.githubcopilot.com/v1/chat/completions",
        )
    )

    assert headers["Authorization"] == "Bearer copilot-session"
    assert "authorization" not in headers


def test_apply_copilot_api_auth_replaces_non_bearer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_api_token() -> copilot_auth.CopilotAPIToken:
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


def test_apply_copilot_api_auth_injects_required_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_api_token() -> copilot_auth.CopilotAPIToken:
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
    assert headers["editor-version"] == "vscode/1.104.1"


def test_apply_copilot_api_auth_preserves_existing_copilot_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_api_token() -> copilot_auth.CopilotAPIToken:
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

    assert first.token == "copilot-api"
    assert second.token == "copilot-api"
    assert calls["count"] == 1


def test_token_provider_prefers_explicit_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "copilot-api")
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://api.githubcopilot.com")

    token = asyncio.run(copilot_auth.CopilotTokenProvider().get_api_token())

    assert token.token == "copilot-api"
    assert token.api_url == "https://api.githubcopilot.com"


def test_token_provider_raises_without_oauth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_API_TOKEN", raising=False)
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
