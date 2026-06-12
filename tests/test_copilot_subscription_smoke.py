"""Cross-platform smoke test for GitHub Copilot subscription routing.

The subscription flow has to behave identically on macOS, Linux, and Windows
(and in headless Docker/CI), but the only OS-specific part — reading the
Copilot CLI token from the platform secret store — is impossible to exercise
portably. This suite proves the *portable* contract instead:

1. With an explicit Copilot API token in the environment, resolution + API-URL
   discovery succeed on every platform without touching any secret store. This
   is the deterministic escape hatch (``GITHUB_COPILOT_API_TOKEN``) for
   headless CI. OAuth tokens still need successful token exchange before
   subscription mode can use them.
2. Each OS-specific secret reader is inert on a foreign platform — so on any
   given OS only that OS's reader can fire, and a missing/foreign secret store
   degrades to ``None`` rather than crashing.
3. The proxy injects exactly the token the wrapper validated (the
   deterministic-handoff fix), never a different discoverable one.
4. The full wrapper→proxy chain carries one consistent token end to end.

Everything here is hermetic: no Keychain, no ``secret-tool``, no Credential
Manager, no network. It runs the same on every OS.
"""

from __future__ import annotations

import asyncio

import pytest

from headroom import copilot_auth, copilot_linux_secret, copilot_macos_keychain

BUSINESS_API = "https://api.business.githubcopilot.com"


def _stub_all_secret_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate 'no OS secret store / not logged in' on every platform."""
    monkeypatch.setattr(copilot_auth, "read_headroom_copilot_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_windows_copilot_cli_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_macos_keychain_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_linux_secret_oauth_token", lambda: None)
    monkeypatch.setattr(copilot_auth, "_read_file_oauth_token_candidates", lambda: [])
    monkeypatch.setattr(copilot_auth, "_read_gh_cli_oauth_token", lambda: None)


def _clear_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        *copilot_auth._COPILOT_OAUTH_TOKEN_ENV_VARS,
        *copilot_auth._GENERIC_GITHUB_TOKEN_ENV_VARS,
        *copilot_auth._API_TOKEN_ENV_VARS,
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# 1. The explicit API-token env path resolves on any platform with no secret store.
# ---------------------------------------------------------------------------
def test_api_token_env_resolves_subscription_without_secret_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_all_secret_stores(monkeypatch)
    _clear_token_env(monkeypatch)
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "tid_env_universal")
    monkeypatch.setattr(
        copilot_auth, "_subscription_resolution_from_token_exchange", lambda _: None
    )
    monkeypatch.setattr(
        copilot_auth,
        "_fetch_copilot_user_info",
        lambda token: (
            {"endpoints": {"api": BUSINESS_API}} if token == "tid_env_universal" else None
        ),
    )

    assert copilot_auth.resolve_subscription_bearer_token() == "tid_env_universal"
    # Routing is override -> generic; the account host advertised by user-info is
    # NOT used (it regressed newer models on the responses API, #610). With no
    # GITHUB_COPILOT_API_URL pin set, the generic public host is returned.
    monkeypatch.delenv("GITHUB_COPILOT_API_URL", raising=False)
    assert copilot_auth.resolve_copilot_api_url("tid_env_universal") == copilot_auth.DEFAULT_API_URL


def test_api_url_falls_back_to_default_when_user_info_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_token_env(monkeypatch)
    monkeypatch.delenv("GITHUB_COPILOT_API_URL", raising=False)
    monkeypatch.setattr(copilot_auth, "_fetch_copilot_user_info", lambda token: None)

    # No network / no endpoints advertised → safe default, never a crash.
    assert copilot_auth.resolve_copilot_api_url("gho-anything") == copilot_auth.DEFAULT_API_URL


def test_subscription_rejects_generic_token_and_accepts_api_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_all_secret_stores(monkeypatch)
    _clear_token_env(monkeypatch)
    monkeypatch.setattr(
        copilot_auth, "_subscription_resolution_from_token_exchange", lambda _: None
    )
    # A generic GitHub token is present but cannot be exchanged for a Copilot
    # API token; a valid Copilot API token is discoverable behind it.
    monkeypatch.setattr(
        copilot_auth,
        "iter_oauth_token_candidates",
        lambda: [
            copilot_auth.CopilotTokenCandidate(
                token="ghp-generic-pat", source="env:GITHUB_TOKEN", confidence="generic-github"
            ),
            copilot_auth.CopilotTokenCandidate(
                token="tid_real_copilot",
                source="macos-keychain:copilot-cli",
                confidence="high",
            ),
        ],
    )
    monkeypatch.setattr(
        copilot_auth,
        "_fetch_copilot_user_info",
        lambda token: {"endpoints": {"api": BUSINESS_API}} if token == "tid_real_copilot" else None,
    )

    assert copilot_auth.resolve_subscription_bearer_token() == "tid_real_copilot"


# ---------------------------------------------------------------------------
# 2. Each OS reader is inert on a foreign platform.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("foreign_platform", ["linux", "win32"])
def test_macos_reader_noop_off_darwin(
    monkeypatch: pytest.MonkeyPatch, foreign_platform: str
) -> None:
    monkeypatch.setattr(copilot_macos_keychain.sys, "platform", foreign_platform)
    assert copilot_macos_keychain.read_copilot_oauth_token(host="github.com") is None


@pytest.mark.parametrize("foreign_platform", ["darwin", "win32"])
def test_linux_reader_noop_off_linux(
    monkeypatch: pytest.MonkeyPatch, foreign_platform: str
) -> None:
    monkeypatch.setattr(copilot_linux_secret.sys, "platform", foreign_platform)
    assert copilot_linux_secret.read_copilot_oauth_token(host="github.com") is None


def test_windows_reader_noop_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(copilot_auth.os, "name", "posix")
    assert copilot_auth._read_windows_copilot_cli_oauth_token() is None


# ---------------------------------------------------------------------------
# 3. The proxy injects exactly the wrapper-validated token (determinism).
# ---------------------------------------------------------------------------
def test_proxy_injects_explicit_token_over_discovered_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reset the cached module-level provider so this test is self-contained.
    monkeypatch.setattr(copilot_auth, "_provider", None)
    # What `wrap copilot --subscription` exports for the proxy:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "gho-validated")
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", BUSINESS_API)
    monkeypatch.setenv("GITHUB_COPILOT_USE_TOKEN_EXCHANGE", "false")
    # A *different* token is discoverable — it must be ignored entirely.
    monkeypatch.setattr(
        copilot_auth, "read_cached_oauth_token", lambda: "gho-WRONG-should-not-be-used"
    )

    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {"authorization": "Bearer placeholder"},
            url=f"{BUSINESS_API}/v1/chat/completions",
        )
    )

    assert headers["Authorization"] == "Bearer gho-validated"
    assert "authorization" not in headers


# ---------------------------------------------------------------------------
# 4. Full wrapper→proxy chain carries one consistent token to a pinned host.
# ---------------------------------------------------------------------------
def test_end_to_end_subscription_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(copilot_auth, "_provider", None)

    # (a) wrapper side: resolve + validate the subscription token. The API host
    #     comes from the GITHUB_COPILOT_API_URL pin — the supported way to target
    #     a dedicated enterprise / data-residency host. user-info is NOT used to
    #     route (#610), so it advertises a *different* host here to prove it is
    #     ignored when picking the upstream.
    _stub_all_secret_stores(monkeypatch)
    _clear_token_env(monkeypatch)
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-seat-token")
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", BUSINESS_API)
    monkeypatch.setattr(
        copilot_auth,
        "_subscription_resolution_from_token_exchange",
        lambda _candidate: copilot_auth._subscription_resolution(
            token="tid-seat-token",
            source="env:GITHUB_COPILOT_TOKEN:token-exchange",
            confidence="copilot-token-exchange",
            api_url=BUSINESS_API,
        ),
    )
    monkeypatch.setattr(
        copilot_auth,
        "_fetch_copilot_user_info",
        lambda token: (
            {"endpoints": {"api": "https://api.individual.githubcopilot.com"}}
            if token == "gho-seat-token"
            else None
        ),
    )
    resolved_token = copilot_auth.resolve_subscription_bearer_token()
    resolved_url = copilot_auth.resolve_copilot_api_url(resolved_token)
    assert resolved_token == "tid-seat-token"
    assert resolved_url == BUSINESS_API  # the pin wins; the user-info host is ignored

    # (b) hand-off: the wrapper exports exactly these for the proxy.
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", resolved_token)

    # (c) proxy side: build the upstream URL (Copilot has no /v1 prefix) and
    #     inject the same token onto the outbound request.
    upstream = copilot_auth.build_copilot_upstream_url(resolved_url, "/v1/chat/completions")
    assert upstream == "https://api.business.githubcopilot.com/chat/completions"

    headers = asyncio.run(
        copilot_auth.apply_copilot_api_auth({"authorization": "Bearer placeholder"}, url=upstream)
    )
    assert headers["Authorization"] == f"Bearer {resolved_token}"
