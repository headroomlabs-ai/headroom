"""A Copilot token and its integration ID must leave together.

GitHub binds a Copilot API token to the ``Copilot-Integration-Id`` it was
minted under and verifies the pairing with an HMAC. Presenting a token minted
for one integration alongside a header naming another fails with:

    401 unauthorized: unable to validate HMAC for the given
        Copilot-Integration-ID

Reported from a Copilot CLI session:

    [CopilotCLISession] Failed to fetch models: Error: 401 "unauthorized:
        unable to validate HMAC for the given Copilot-Integration-ID"
    [CopilotCLISession] Proxy URL configured (authType=hmac), skipping
        client-side token validation

``apply_copilot_api_auth`` applied the integration ID with *set-default*
semantics (``_set_header_default`` returns early when the header is already
present) BEFORE deciding whose token to use. The client always sends one, so
when Headroom replaced the token — the common case, logged as ``incoming token
not suitable (kind=unknown), will replace`` — the request went out carrying the
CLIENT's integration ID next to HEADROOM's token, minted under ``vscode-chat``.

The second log line is why nothing caught it sooner: seeing a proxy URL, the
Copilot client reports ``authType=hmac`` and skips its own token validation,
deferring to the proxy. Nobody checks the pairing until GitHub rejects it.

The failing call was model discovery, so the client fell back to its built-in
model list — which is why a user's selected model never appeared in telemetry.
"""

from __future__ import annotations

import asyncio

import pytest

from headroom import copilot_auth
from headroom.copilot_auth import (
    CopilotAPIToken,
    apply_copilot_api_auth,
    resolve_copilot_integration_id,
)

CAPI = "https://api.githubcopilot.com/chat/completions"
CLI_ID = "copilot-cli-chat"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "GITHUB_COPILOT_INTEGRATION_ID",
        "GITHUB_COPILOT_API_TOKEN",
        "GITHUB_COPILOT_REFRESH_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    copilot_auth._provider = None
    yield
    copilot_auth._provider = None


class _RecordingProvider:
    """Stands in for the token provider; records what it was asked to mint."""

    def __init__(self) -> None:
        self.asked: list[str | None] = []

    async def get_api_token(self, *, integration_id: str | None = None):
        self.asked.append(integration_id)
        return CopilotAPIToken(
            token=f"minted-for-{integration_id}",
            expires_at=9_999_999_999.0,
            api_url="https://api.githubcopilot.com",
        )


def _install(monkeypatch) -> _RecordingProvider:
    provider = _RecordingProvider()
    monkeypatch.setattr(copilot_auth, "get_copilot_token_provider", lambda: provider)
    return provider


def _apply(headers: dict, url: str = CAPI) -> dict:
    return asyncio.run(apply_copilot_api_auth(headers, url=url))


# --------------------------------------------------------------------------- #
# The reported failure
# --------------------------------------------------------------------------- #
def test_token_is_minted_under_the_clients_integration_id(monkeypatch) -> None:
    provider = _install(monkeypatch)

    _apply({"Authorization": "Bearer unusable", "Copilot-Integration-Id": CLI_ID})

    assert provider.asked == [CLI_ID], (
        "the replacement token must be minted for the surface that made the "
        "call, not for the proxy's default"
    )


def test_forwarded_header_matches_the_minted_token(monkeypatch) -> None:
    """The invariant. This is what GitHub HMAC-verifies."""
    provider = _install(monkeypatch)

    out = _apply({"Authorization": "Bearer unusable", "Copilot-Integration-Id": CLI_ID})

    minted_for = provider.asked[0]
    assert out["Authorization"] == f"Bearer minted-for-{minted_for}"
    assert out["Copilot-Integration-Id"] == minted_for


def test_no_duplicate_integration_id_header_is_emitted(monkeypatch) -> None:
    """Overwriting must replace the client's casing, not sit beside it."""
    _install(monkeypatch)

    out = _apply({"Authorization": "Bearer unusable", "copilot-integration-id": CLI_ID})

    matching = [k for k in out if k.lower() == "copilot-integration-id"]
    assert len(matching) == 1
    # Written through the client's own key, not beside it.
    assert matching[0] == "copilot-integration-id"


# --------------------------------------------------------------------------- #
# The pass-through branch keeps the client's own matched pair
# --------------------------------------------------------------------------- #
def test_a_forwardable_client_token_keeps_the_clients_id(monkeypatch) -> None:
    """When we don't replace the credential, we must not touch its pairing."""
    provider = _install(monkeypatch)
    monkeypatch.setattr(copilot_auth, "_is_forwardable_copilot_bearer_token", lambda _t: True)
    monkeypatch.setattr(copilot_auth, "_is_managed_copilot_seeded_bearer", lambda _t: False)

    out = _apply({"Authorization": "Bearer tid=real;exp=1", "Copilot-Integration-Id": CLI_ID})

    assert provider.asked == [], "no token should have been minted"
    assert out["Authorization"] == "Bearer tid=real;exp=1"
    assert out["Copilot-Integration-Id"] == CLI_ID


# --------------------------------------------------------------------------- #
# Resolution order
# --------------------------------------------------------------------------- #
def test_a_client_that_states_its_identity_beats_the_configured_default(
    monkeypatch,
) -> None:
    """``GITHUB_COPILOT_INTEGRATION_ID`` configures the DEFAULT, it does not
    override a client that named itself — the long-standing contract pinned by
    ``test_apply_copilot_api_auth_preserves_existing_copilot_headers``. What
    matters here is that whichever value wins is used for BOTH halves.
    """
    monkeypatch.setenv("GITHUB_COPILOT_INTEGRATION_ID", "enterprise-shim")
    provider = _install(monkeypatch)

    out = _apply({"Authorization": "Bearer unusable", "Copilot-Integration-Id": CLI_ID})

    assert provider.asked == [CLI_ID]
    assert out["Copilot-Integration-Id"] == CLI_ID


def test_the_configured_default_applies_when_the_client_sends_none(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_INTEGRATION_ID", "enterprise-shim")
    provider = _install(monkeypatch)

    out = _apply({"Authorization": "Bearer unusable"})

    assert provider.asked == ["enterprise-shim"]
    assert out["Copilot-Integration-Id"] == "enterprise-shim"


def test_client_value_wins_over_the_default(monkeypatch) -> None:
    assert resolve_copilot_integration_id(CLI_ID) == CLI_ID


def test_default_when_the_client_sends_none(monkeypatch) -> None:
    provider = _install(monkeypatch)

    out = _apply({"Authorization": "Bearer unusable"})

    assert provider.asked == [copilot_auth._DEFAULT_COPILOT_INTEGRATION_ID]
    assert out["Copilot-Integration-Id"] == copilot_auth._DEFAULT_COPILOT_INTEGRATION_ID


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_client_values_fall_back(blank) -> None:
    assert resolve_copilot_integration_id(blank) == copilot_auth._DEFAULT_COPILOT_INTEGRATION_ID


def test_non_copilot_upstream_is_untouched(monkeypatch) -> None:
    provider = _install(monkeypatch)
    headers = {"Authorization": "Bearer sk-openai", "Copilot-Integration-Id": CLI_ID}

    out = _apply(dict(headers), url="https://api.openai.com/v1/chat/completions")

    assert out == headers
    assert provider.asked == []


# --------------------------------------------------------------------------- #
# The cache must not hand one integration another's token
# --------------------------------------------------------------------------- #
def test_tokens_are_cached_per_integration_id(monkeypatch) -> None:
    from headroom.copilot_auth import CopilotTokenProvider

    provider = CopilotTokenProvider()
    exchanged: list[str | None] = []

    async def _fake_exchange(oauth_token, *, integration_id=None):  # noqa: ANN001
        exchanged.append(integration_id)
        return CopilotAPIToken(
            token=f"tok-{integration_id}",
            expires_at=9_999_999_999.0,
            api_url="https://api.githubcopilot.com",
        )

    monkeypatch.setattr(provider, "_exchange_token", _fake_exchange)
    monkeypatch.setattr(copilot_auth, "read_cached_oauth_token", lambda: "oauth")
    monkeypatch.setattr(copilot_auth, "_should_exchange_oauth_token", lambda: True)

    a = asyncio.run(provider.get_api_token(integration_id="vscode-chat"))
    b = asyncio.run(provider.get_api_token(integration_id=CLI_ID))
    a_again = asyncio.run(provider.get_api_token(integration_id="vscode-chat"))

    assert a.token == "tok-vscode-chat"
    assert b.token == f"tok-{CLI_ID}"
    # Distinct integrations must not share a slot...
    assert a.token != b.token
    # ...and the same one must still be served from cache.
    assert a_again.token == a.token
    assert exchanged == ["vscode-chat", CLI_ID]
