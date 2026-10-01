"""VS Code Copilot inline completions must reach Copilot, not OpenAI (#3076).

When `github.copilot.advanced.debug.overrideProxyUrl` points at Headroom, the
Copilot extension sends its "ghost text" completions to
``/v1/engines/<engine>/completions``. Headroom registers no route for that path,
so it lands in the catch-all passthrough — which resolves an upstream from the
auth headers alone and therefore fell through to the OpenAI target. Editor
keystrokes were forwarded to ``api.openai.com``, a host that has not served the
Engines API for years and that corporate networks routinely block.

Two things have to hold for the round trip: the path has to select the Copilot
API, and it has to survive Copilot's ``/v1``-stripping intact, because the
extension already built the exact path Copilot serves.
"""

from __future__ import annotations

import asyncio

import pytest

from headroom import copilot_auth
from headroom.copilot_auth import (
    build_copilot_upstream_url,
    copilot_completions_base_url,
    is_copilot_completions_path,
    reset_observed_completions_endpoint,
)
from headroom.providers.proxy_targets import select_passthrough_base_url

COPILOT_API = "https://api.githubcopilot.com"
# GitHub serves inline completions from a *different* host than chat. Verified
# unauthenticated against the live endpoints:
#   POST copilot-proxy.githubusercontent.com/v1/engines/<e>/completions -> 401
#   POST api.githubcopilot.com/v1/engines/<e>/completions               -> 404
# and proxy.<sku>.githubcopilot.com is a CNAME to the former. 401 means "exists,
# needs auth"; 404 means the CAPI host does not serve this path at all (#3076).
COMPLETIONS_PROXY = "https://copilot-proxy.githubusercontent.com"
COMPLETIONS = "/v1/engines/gpt-41-copilot/completions"


def _proxy(**legacy_targets: str):
    class Runtime:
        @staticmethod
        def api_target(provider: str) -> str:
            return f"https://runtime.{provider}.test"

        @staticmethod
        def model_metadata_provider(headers) -> str:  # type: ignore[no-untyped-def]
            return "anthropic" if headers.get("x-api-key") else "openai"

    return type("Proxy", (), {**legacy_targets, "provider_runtime": Runtime()})()


@pytest.fixture(autouse=True)
def _no_ambient_copilot_config(monkeypatch: pytest.MonkeyPatch):
    """Resolve the Copilot URL from a clean environment, not the dev's own."""
    for var in (
        "GITHUB_COPILOT_API_URL",
        "GITHUB_COPILOT_ENTERPRISE_URL",
        "GITHUB_COPILOT_PROXY_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_observed_completions_endpoint()
    yield
    reset_observed_completions_endpoint()


# --------------------------------------------------------------------------- #
# Path recognition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path",
    [
        COMPLETIONS,
        "/v1/engines/copilot-codex/completions",
        # A trailing slash is still the same endpoint.
        "/v1/engines/gpt-41-copilot/completions/",
    ],
)
def test_copilot_completions_paths_are_recognised(path: str) -> None:
    assert is_copilot_completions_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        # The OpenAI-compatible surface, which must keep its existing routing.
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/messages",
        "/models",
        # Shape-alike paths that are not the completions endpoint. Matching
        # these would divert unrelated traffic to Copilot.
        "/v1/engines/gpt-41-copilot",
        "/v1/engines//completions",
        "/v1/engines/a/b/completions",
        "/v2/engines/gpt-41-copilot/completions",
    ],
)
def test_other_paths_are_not_mistaken_for_completions(path: str) -> None:
    assert is_copilot_completions_path(path) is False


# --------------------------------------------------------------------------- #
# Upstream selection
# --------------------------------------------------------------------------- #
def test_completions_do_not_fall_through_to_the_openai_target() -> None:
    """The reported bug: keystrokes forwarded to api.openai.com."""
    proxy = _proxy(OPENAI_API_URL="https://api.openai.com")

    # ...and they must land on the completions host, not the chat host, which
    # answers this path with 404.
    assert select_passthrough_base_url(proxy, {}, COMPLETIONS) == COMPLETIONS_PROXY


def test_a_chat_host_is_not_treated_as_a_completions_host() -> None:
    """A CAPI host must still be redirected, because it does not serve this path.

    `headroom wrap vscode` points the OpenAI target at the resolved subscription
    URL, which is the *chat* surface (it is what `GITHUB_COPILOT_API_URL` is set
    to). Leaving it alone — as an "it's already a Copilot host" guard did — sent
    `/v1/engines/.../completions` to a host that answers 404.
    """
    proxy = _proxy(OPENAI_API_URL="https://api.business.githubcopilot.com")

    assert select_passthrough_base_url(proxy, {}, COMPLETIONS) == COMPLETIONS_PROXY


def test_an_account_specific_completions_host_is_left_alone() -> None:
    """A host that already serves completions is never rewritten.

    These are the per-SKU hosts GitHub hands out through `endpoints.proxy`, so
    replacing one with the generic default would move a subscriber off the host
    their own token named.
    """
    proxy = _proxy(OPENAI_API_URL="https://proxy.business.githubcopilot.com")

    assert (
        select_passthrough_base_url(proxy, {}, COMPLETIONS)
        == "https://proxy.business.githubcopilot.com"
    )


def test_enterprise_deployments_keep_their_own_copilot_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The redirect is env-resolved, so a GHE tenant is not sent to github.com."""
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://copilot-api.acme.ghe.com")
    proxy = _proxy(OPENAI_API_URL="https://api.openai.com")

    assert select_passthrough_base_url(proxy, {}, COMPLETIONS) == (
        "https://copilot-api.acme.ghe.com"
    )


def test_non_copilot_paths_keep_their_existing_upstream() -> None:
    """The redirect is scoped to the one path; nothing else may move."""
    proxy = _proxy(
        OPENAI_API_URL="https://legacy.openai.test",
        ANTHROPIC_API_URL="https://legacy.anthropic.test",
        GEMINI_API_URL="https://legacy.gemini.test",
    )

    assert select_passthrough_base_url(proxy, {}, "/v1/chat/completions") == (
        "https://legacy.openai.test"
    )
    assert select_passthrough_base_url(proxy, {}, "/v1/embeddings") == "https://legacy.openai.test"
    # Callers that pass no path at all behave exactly as before.
    assert select_passthrough_base_url(proxy, {}) == "https://legacy.openai.test"


def test_explicit_provider_auth_is_never_hijacked() -> None:
    """Only the OpenAI fall-through is redirected.

    The Copilot extension sends none of these headers, so a request that
    selected an upstream through one of them is not Copilot's — and silently
    diverting a caller who authenticated to a named provider would be worse
    than the bug being fixed.
    """
    proxy = _proxy(
        OPENAI_API_URL="https://api.openai.com",
        ANTHROPIC_API_URL="https://legacy.anthropic.test",
        GEMINI_API_URL="https://legacy.gemini.test",
    )

    assert select_passthrough_base_url(proxy, {"x-api-key": "k"}, COMPLETIONS) == (
        "https://legacy.anthropic.test"
    )
    assert select_passthrough_base_url(proxy, {"x-goog-api-key": "k"}, COMPLETIONS) == (
        "https://legacy.gemini.test"
    )
    assert select_passthrough_base_url(proxy, {"chatgpt-account-id": "acct"}, COMPLETIONS) == (
        "https://chatgpt.com"
    )


# --------------------------------------------------------------------------- #
# Where completions are sent
# --------------------------------------------------------------------------- #
def test_completions_host_defaults_to_githubs_completions_proxy() -> None:
    """The default is GitHub's own default for this endpoint, not the CAPI host.

    `@vscode/copilot-api` resolves it as
    ``token?.endpoints.proxy || DEFAULT_PROXY_BASE_URL`` where
    ``DEFAULT_PROXY_BASE_URL = "https://copilot-proxy.githubusercontent.com"``.
    """
    assert copilot_completions_base_url() == COMPLETIONS_PROXY


def test_github_advertised_completions_host_wins_over_the_default() -> None:
    """GitHub names the completions host in the token exchange; believe it.

    This is what keeps the destination from being an assumption about which
    host serves inline completions — if GitHub says they live elsewhere, that
    is where they go.
    """
    copilot_auth._remember_completions_endpoint(
        {
            "token": "tid=x",
            "endpoints": {
                "api": COPILOT_API,
                "proxy": "https://copilot-proxy.githubusercontent.com",
            },
        }
    )

    assert copilot_completions_base_url() == "https://copilot-proxy.githubusercontent.com"


def test_an_operator_override_beats_everything() -> None:
    """A network fronting Copilot through its own gateway needs no code change."""
    copilot_auth._remember_completions_endpoint(
        {"endpoints": {"proxy": "https://copilot-proxy.githubusercontent.com"}}
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GITHUB_COPILOT_PROXY_URL", "https://copilot.internal.acme/")

        assert copilot_completions_base_url() == "https://copilot.internal.acme"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"endpoints": {}},
        {"endpoints": {"proxy": "   "}},
        {"endpoints": {"proxy": 7}},
        {"endpoints": "not-a-dict"},
        "not-a-dict",
    ],
)
def test_a_payload_without_a_usable_proxy_host_changes_nothing(payload) -> None:  # type: ignore[no-untyped-def]
    copilot_auth._remember_completions_endpoint(payload)

    assert copilot_completions_base_url() == COMPLETIONS_PROXY


def test_the_advertised_host_is_used_for_routing() -> None:
    proxy = _proxy(OPENAI_API_URL="https://api.openai.com")
    copilot_auth._remember_completions_endpoint(
        {"endpoints": {"proxy": "https://copilot-proxy.githubusercontent.com"}}
    )

    assert select_passthrough_base_url(proxy, {}, COMPLETIONS) == (
        "https://copilot-proxy.githubusercontent.com"
    )


# --------------------------------------------------------------------------- #
# URL construction
# --------------------------------------------------------------------------- #
def test_completions_keep_their_v1_prefix() -> None:
    """Copilot built this path itself, so rewriting it can only break it.

    ``/v1`` is stripped for clients speaking generic-OpenAI at Copilot's
    unprefixed surface. Applying that to a Copilot-native path turns a working
    request into a 404.
    """
    assert build_copilot_upstream_url(COPILOT_API, COMPLETIONS) == f"{COPILOT_API}{COMPLETIONS}"


def test_the_v1_strip_still_applies_to_the_openai_surface() -> None:
    """Guard the behaviour the carve-out sits next to."""
    assert (
        build_copilot_upstream_url(COPILOT_API, "/v1/chat/completions")
        == f"{COPILOT_API}/chat/completions"
    )
    assert build_copilot_upstream_url(COPILOT_API, "/v1/messages") == f"{COPILOT_API}/v1/messages"
    assert build_copilot_upstream_url(COPILOT_API, "/models") == f"{COPILOT_API}/models"


def test_a_non_copilot_upstream_is_never_rewritten() -> None:
    assert (
        build_copilot_upstream_url("https://api.openai.com", COMPLETIONS)
        == f"https://api.openai.com{COMPLETIONS}"
    )


# --------------------------------------------------------------------------- #
# Telling the two Copilot surfaces apart
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url",
    [
        "https://copilot-proxy.githubusercontent.com",
        "https://copilot-proxy.githubusercontent.com/",
        # Scheme-less, as a hand-written config value can be. Failing to
        # recognise it means forwarding with no credentials.
        "copilot-proxy.githubusercontent.com",
        "proxy.individual.githubcopilot.com/v1/engines/x/completions",
        "https://proxy.individual.githubcopilot.com",
        "https://proxy.business.githubcopilot.com",
        "https://proxy.enterprise.githubcopilot.com",
    ],
)
def test_completions_hosts_are_recognised(url: str) -> None:
    assert copilot_auth.is_copilot_completions_host(url) is True


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        # The chat surface. Recognising it as a completions host is the bug this
        # function exists to prevent: it answers this path with 404.
        "https://api.githubcopilot.com",
        "https://api.business.githubcopilot.com",
        "https://copilot-api.acme.ghe.com",
        "https://api.openai.com",
        # Not a Copilot host merely because "proxy" appears somewhere.
        "https://proxy.example.com",
        "https://notproxy.githubcopilot.com",
    ],
)
def test_non_completions_hosts_are_rejected(url) -> None:  # type: ignore[no-untyped-def]
    assert copilot_auth.is_copilot_completions_host(url) is False


def test_an_operator_override_counts_as_a_completions_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise the redirect would fight the operator's own configuration."""
    monkeypatch.setenv("GITHUB_COPILOT_PROXY_URL", "https://copilot.internal.acme/")

    assert copilot_auth.is_copilot_completions_host("https://copilot.internal.acme") is True
    proxy = _proxy(OPENAI_API_URL="https://copilot.internal.acme")
    assert select_passthrough_base_url(proxy, {}, COMPLETIONS) == "https://copilot.internal.acme"


# --------------------------------------------------------------------------- #
# An enterprise tenant's keystrokes must not leave their deployment
# --------------------------------------------------------------------------- #
def test_an_enterprise_deployment_is_never_sent_to_the_public_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public default applies only when no custom deployment is configured.

    For a GHE tenant, defaulting to ``copilot-proxy.githubusercontent.com``
    would forward editor keystrokes to a host outside their deployment. Staying
    on their own host may still be the wrong surface, but it keeps the traffic
    inside the tenant; ``GITHUB_COPILOT_PROXY_URL`` is the exact fix.
    """
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://copilot-api.github.acme.com")

    resolved = copilot_completions_base_url()

    assert resolved == "https://copilot-api.github.acme.com"
    assert "githubusercontent.com" not in resolved
    assert "githubcopilot.com" not in resolved


def test_the_advertised_host_still_wins_for_an_enterprise_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitHub naming the host beats any inference from the configured API URL."""
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://copilot-api.github.acme.com")
    copilot_auth._remember_completions_endpoint(
        {"endpoints": {"proxy": "https://copilot-proxy.github.acme.com"}}
    )

    assert copilot_completions_base_url() == "https://copilot-proxy.github.acme.com"


# --------------------------------------------------------------------------- #
# Routing to the right host is only half of it: it needs credentials
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url",
    [
        "https://copilot-proxy.githubusercontent.com/v1/engines/gpt-41-copilot/completions",
        "https://proxy.business.githubcopilot.com/v1/engines/gpt-41-copilot/completions",
        # The chat surface must keep working exactly as before.
        "https://api.githubcopilot.com/chat/completions",
    ],
)
def test_a_copilot_upstream_is_authenticated(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """Both Copilot surfaces get credentials.

    Gating auth on the chat host alone routed completions to the correct host
    with no Authorization header at all, which that host answers 401 — the fix
    for the destination would have been inert without this.
    """

    class _Token:
        token = "test-copilot-token"

    class _Provider:
        async def get_api_token(self, *, integration_id=None):  # noqa: ANN001, ANN202
            return _Token()

    monkeypatch.setattr(copilot_auth, "get_copilot_token_provider", lambda: _Provider())

    resolved = asyncio.run(copilot_auth.apply_copilot_api_auth({}, url=url))

    assert resolved.get("Authorization") == "Bearer test-copilot-token"


def test_a_non_copilot_upstream_is_never_given_copilot_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The widened gate must not start handing Copilot tokens to other hosts."""

    class _Provider:
        async def get_api_token(self, *, integration_id=None):  # noqa: ANN001, ANN202
            raise AssertionError("must not mint a Copilot token for a non-Copilot host")

    monkeypatch.setattr(copilot_auth, "get_copilot_token_provider", lambda: _Provider())

    for url in (
        "https://api.openai.com/v1/engines/x/completions",
        "https://proxy.example.com/v1/engines/x/completions",
        "https://evil.githubcopilot.com.attacker.test/v1/engines/x/completions",
    ):
        assert asyncio.run(copilot_auth.apply_copilot_api_auth({}, url=url)) == {}


def test_a_completions_host_is_marked_as_copilot_routed() -> None:
    """`build_copilot_upstream_url` is the chokepoint that labels the provider."""
    url = copilot_auth.build_copilot_upstream_url(
        "https://copilot-proxy.githubusercontent.com", COMPLETIONS
    )

    assert url == f"https://copilot-proxy.githubusercontent.com{COMPLETIONS}"
    assert copilot_auth.is_copilot_upstream_url(url) is True


@pytest.mark.parametrize(
    "configured_api_url",
    [
        # What `headroom wrap vscode` actually exports (wrap.py sets
        # GITHUB_COPILOT_API_URL to the resolved subscription URL).
        "https://api.business.githubcopilot.com",
        "https://api.individual.githubcopilot.com",
        "https://api.githubcopilot.com",
    ],
)
def test_a_public_capi_url_does_not_become_the_completions_host(
    monkeypatch: pytest.MonkeyPatch, configured_api_url: str
) -> None:
    """A configured *chat* URL must not drag completions back onto the 404 host.

    The in-tenant rule for a custom deployment has to exclude public Copilot
    hosts, or the single most common setup — `headroom wrap vscode`, which
    exports GITHUB_COPILOT_API_URL — lands right back where it started.
    """
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", configured_api_url)

    assert copilot_completions_base_url() == COMPLETIONS_PROXY


@pytest.mark.parametrize(
    "configured_api_url",
    [
        "https://copilot-api.acme.ghe.com",
        "https://copilot-api.github.acme.com",
    ],
)
def test_a_custom_deployment_still_keeps_its_own_host(
    monkeypatch: pytest.MonkeyPatch, configured_api_url: str
) -> None:
    """Only a host outside *.githubcopilot.com marks a deployment to stay put."""
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", configured_api_url)

    assert copilot_completions_base_url() == configured_api_url


# --------------------------------------------------------------------------- #
# The two callers pass different shapes of URL
# --------------------------------------------------------------------------- #
def test_an_operator_override_is_matched_on_the_full_request_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routing sees a base URL; auth sees the base URL *plus the path*.

    Matching the override by whole-string equality answered True for the first
    and False for the second, so an operator gateway was routed to correctly and
    then forwarded with no credentials — a 401 on the one configuration that is
    the documented remedy for a custom deployment.
    """
    monkeypatch.setenv("GITHUB_COPILOT_PROXY_URL", "https://gw.corp.internal")

    base = "https://gw.corp.internal"
    full = f"https://gw.corp.internal{COMPLETIONS}"

    assert copilot_auth.is_copilot_completions_host(base) is True
    assert copilot_auth.is_copilot_completions_host(full) is True
    assert copilot_auth.is_copilot_completions_host("https://gw.corp.internal/") is True
    assert copilot_auth.is_copilot_upstream_url(full) is True
    # A different host is still not the override.
    assert copilot_auth.is_copilot_completions_host(f"https://elsewhere.test{COMPLETIONS}") is False


def test_an_operator_override_gateway_receives_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End of the same chain: the gateway must actually be authenticated."""

    class _Token:
        token = "test-copilot-token"

    class _Provider:
        async def get_api_token(self, *, integration_id=None):  # noqa: ANN001, ANN202
            return _Token()

    monkeypatch.setenv("GITHUB_COPILOT_PROXY_URL", "https://gw.corp.internal")
    monkeypatch.setattr(copilot_auth, "get_copilot_token_provider", lambda: _Provider())

    resolved = asyncio.run(
        copilot_auth.apply_copilot_api_auth({}, url=f"https://gw.corp.internal{COMPLETIONS}")
    )

    assert resolved.get("Authorization") == "Bearer test-copilot-token"


@pytest.mark.parametrize(
    "configured",
    ["api.githubcopilot.com", "api.business.githubcopilot.com"],
)
def test_a_scheme_less_public_capi_url_is_still_recognised(
    monkeypatch: pytest.MonkeyPatch, configured: str
) -> None:
    """A hand-written value without "https://" must not read as a custom host."""
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", configured)

    assert copilot_completions_base_url() == COMPLETIONS_PROXY
