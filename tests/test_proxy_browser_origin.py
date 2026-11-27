"""Requests from a browser page the proxy does not trust never reach the upstream.

The proxy adds operator-held credentials (``*_extra_headers``, the Copilot
token, provider env keys) to what it forwards, and loopback callers skip
``HEADROOM_PROXY_TOKEN``. A page open in the user's browser reaches the proxy as
a loopback caller and can send a ``text/plain`` JSON body without a CORS
preflight, so the HTTP routes apply the same Origin rule as the WebSocket
routes. Clients that send no ``Origin`` (CLI tools, SDKs, coding agents) and
loopback pages (the dashboard) are unaffected.
"""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import (  # noqa: E402
    FRONT_RELAY_PROOF_HEADER,
    ProxyConfig,
    create_app,
    front_relay_proof,
)

OPERATOR_SECRET = "operator-held-secret"
FOREIGN_ORIGIN = "https://page.example"


@pytest.fixture(autouse=True)
def _default_origin_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_WS_ORIGINS", raising=False)
    monkeypatch.delenv("HEADROOM_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("HEADROOM_PROXY_TOKEN", raising=False)


class _Capturing(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async for _ in request.stream:
            pass
        self.requests.append(request)
        if request.url.path.endswith("/chat/completions"):
            body = {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        else:
            body = {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        return httpx.Response(200, json=body)


def _client(
    base_url: str = "http://127.0.0.1:8787", peer: str = "127.0.0.1"
) -> tuple[TestClient, _Capturing]:
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
        anthropic_extra_headers={"x-api-key": OPERATOR_SECRET},
        openai_extra_headers={"Authorization": f"Bearer {OPERATOR_SECRET}"},
    )
    app = create_app(config)
    transport = _Capturing()
    app.state.proxy.http_client = httpx.AsyncClient(transport=transport)
    # The browser and the proxy share the host, so the peer is loopback.
    # base_url sets the Host a local client sends (TestClient defaults to "testserver").
    return TestClient(app, client=(peer, 51000), base_url=base_url), transport


_ROUTES = {
    "/v1/messages": {
        "model": "claude-sonnet-4-6",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    },
    "/v1/chat/completions": {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    },
}


def _simple_post(client: TestClient, path: str, headers: dict[str, str]) -> httpx.Response:
    # A browser "simple" request: text/plain body, no custom headers, no preflight.
    return client.post(
        path,
        content=json.dumps(_ROUTES[path]),
        headers={"Content-Type": "text/plain", **headers},
    )


@pytest.mark.parametrize("path", sorted(_ROUTES))
@pytest.mark.parametrize(
    "browser_headers",
    [
        {"Origin": FOREIGN_ORIGIN},
        {"Origin": "null"},
        {"Origin": "http://localhost.page.example"},
    ],
)
def test_foreign_page_request_is_refused_before_the_upstream(
    path: str, browser_headers: dict[str, str]
) -> None:
    client, transport = _client()

    resp = _simple_post(client, path, browser_headers)

    assert resp.status_code == 403
    assert transport.requests == []


@pytest.mark.parametrize("path", sorted(_ROUTES))
def test_request_without_origin_still_forwards(path: str) -> None:
    client, transport = _client()

    resp = _simple_post(client, path, {})

    assert resp.status_code == 200, resp.text
    assert len(transport.requests) == 1
    assert any(OPERATOR_SECRET in value for value in transport.requests[0].headers.values())


@pytest.mark.parametrize("origin", ["http://127.0.0.1:8787", "http://localhost:3000"])
def test_loopback_page_still_forwards(origin: str) -> None:
    client, transport = _client()

    resp = _simple_post(client, "/v1/messages", {"Origin": origin})

    assert resp.status_code == 200, resp.text
    assert len(transport.requests) == 1


def test_configured_origin_still_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_WS_ORIGINS", "https://dash.example")
    client, transport = _client()

    allowed = _simple_post(client, "/v1/messages", {"Origin": "https://dash.example"})
    refused = _simple_post(client, "/v1/messages", {"Origin": FOREIGN_ORIGIN})

    assert allowed.status_code == 200, allowed.text
    assert refused.status_code == 403
    assert len(transport.requests) == 1


def test_cross_site_subresource_without_origin_is_refused() -> None:
    client, transport = _client()

    resp = client.get(
        "/v1/models",
        headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"},
    )

    assert resp.status_code == 403
    assert transport.requests == []


def test_cross_site_navigation_to_the_dashboard_still_loads() -> None:
    client, _ = _client()

    resp = client.get(
        "/health",
        headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"},
    )

    assert resp.status_code == 200


def test_loopback_peer_with_a_repointed_hostname_is_refused() -> None:
    # Origin matches Host, but Host is not loopback while the peer is.
    client, transport = _client()

    resp = _simple_post(
        client,
        "/v1/messages",
        {"Origin": "http://page.example:8787", "Host": "page.example:8787"},
    )

    assert resp.status_code == 403
    assert transport.requests == []


REBOUND_HOST = "page.example:8787"
_PATH = "/v1/messages/batches"  # a passthrough GET forwarded with the operator credential


@pytest.mark.parametrize(
    "browser_headers",
    [
        # fetch() from the page itself: same-origin, so no Origin header.
        {"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"},
        # A window the page opened through a cross-site redirect.
        {
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
        },
        # A browser that sends no Sec-Fetch-* headers.
        {"User-Agent": "Mozilla/5.0 (Macintosh) AppleWebKit/605.1.15 Version/15.6 Safari/605.1.15"},
    ],
)
def test_loopback_peer_under_a_foreign_host_is_refused(browser_headers: dict[str, str]) -> None:
    client, transport = _client()

    resp = client.get(_PATH, headers={"Host": REBOUND_HOST, **browser_headers})

    assert resp.status_code == 403
    assert "page.example" in resp.json()["hint"]
    assert "HEADROOM_WS_ORIGINS" in resp.json()["hint"]
    assert transport.requests == []


@pytest.mark.parametrize(
    ("listed", "host"),
    [
        ("http://page.example:8787", REBOUND_HOST),
        # TLS ends at a reverse proxy on this host: the page is https, the proxy sees http.
        ("https://dash.example", "dash.example"),
        ("https://Dash.Example.", "dash.example:443"),
        # An explicit port is kept on both sides, even a scheme's other default.
        ("https://dash.example:80", "dash.example:80"),
    ],
)
def test_listed_host_is_accepted_whatever_the_scheme(
    monkeypatch: pytest.MonkeyPatch, listed: str, host: str
) -> None:
    monkeypatch.setenv("HEADROOM_WS_ORIGINS", listed)
    client, transport = _client()

    resp = client.get(_PATH, headers={"Host": host})

    assert resp.status_code == 200, resp.text
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1:8787",  # CLIs, SDKs, coding agents, the dashboard
        "localhost:8787",
        "[::1]:8787",
        "host.docker.internal:8787",  # a container on Docker Desktop
    ],
)
def test_local_hosts_still_forward(host: str) -> None:
    client, transport = _client()

    resp = client.get(_PATH, headers={"Host": host})

    assert resp.status_code == 200, resp.text
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "headers",
    [
        # A network's DNS can answer the alias too: a page under it is not local.
        {"Host": "host.docker.internal:8787", "Sec-Fetch-Site": "same-origin"},
        {"Host": "host.docker.internal:8787", "Referer": "http://host.docker.internal:8787/"},
        # Not this proxy's port.
        {"Host": "host.docker.internal:9999"},
    ],
)
def test_container_alias_counts_only_without_browser_provenance_on_our_port(
    headers: dict[str, str],
) -> None:
    client, transport = _client()

    resp = client.get(_PATH, headers=headers)

    assert resp.status_code == 403
    assert transport.requests == []


@pytest.mark.parametrize(
    ("base_url", "host"),
    [
        ("http://127.0.0.1:443", "host.docker.internal:443"),
        ("https://127.0.0.1", "host.docker.internal"),  # no port: the scheme's default
    ],
)
def test_container_alias_matches_the_listener_port(base_url: str, host: str) -> None:
    client, transport = _client(base_url)

    resp = client.get(_PATH, headers={"Host": host})

    assert resp.status_code == 200, resp.text
    assert len(transport.requests) == 1


@pytest.mark.parametrize("host", ["host.docker.internal:8787", "127.0.0.1:8787", "localhost:8787"])
def test_rust_front_remote_marker_is_never_local(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    # A front relaying a remote caller to this proxy as host.docker.internal
    # arrives from loopback (Docker Desktop) under a name trusted as local;
    # its remote marker takes that trust away: refused at the local-Host
    # check like any foreign name (a front with --python-upstream sends the
    # relay Host and proof instead).
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "proxy-token")
    client, transport = _client()

    marked = client.get(_PATH, headers={"Host": host, "X-Headroom-Front-Remote": "1"})
    unmarked = client.get(_PATH, headers={"Host": host})

    assert marked.status_code == 403, marked.text
    assert unmarked.status_code == 200, unmarked.text
    assert len(transport.requests) == 1


RELAY_HOST = "headroom-remote.invalid"


@pytest.mark.parametrize("proof", [None, "0" * 64, front_relay_proof("another-token")])
def test_rust_front_relay_host_without_its_proof_is_refused(
    monkeypatch: pytest.MonkeyPatch, proof: str | None
) -> None:
    # Without the proof the relay Host is just a foreign name: a front that
    # relays a rebound page must not turn it into a trusted caller.
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "proxy-token")
    client, transport = _client()
    headers = {"Host": RELAY_HOST}
    if proof is not None:
        headers[FRONT_RELAY_PROOF_HEADER] = proof

    resp = client.get(_PATH, headers=headers)

    assert resp.status_code == 403
    assert transport.requests == []


def test_rust_front_relay_without_a_token_anywhere_is_a_remote_caller() -> None:
    # A token-less front (acknowledged open bind) has nothing to prove with:
    # its remote callers reach the data plane but never the local-only routes.
    client, transport = _client()

    resp = client.get(_PATH, headers={"Host": RELAY_HOST})
    admin = client.get("/debug/tasks", headers={"Host": RELAY_HOST})

    assert resp.status_code == 200, resp.text
    assert len(transport.requests) == 1
    assert admin.status_code == 404


@pytest.mark.parametrize(
    ("path", "statuses", "payload_key"),
    [
        ("/health", {200}, "checks"),
        ("/livez", {200}, "alive"),
        # Not ready until startup has run, which this client skips.
        ("/readyz", {200, 503}, "ready"),
    ],
)
def test_health_probe_relayed_without_a_proof_still_answers(
    monkeypatch: pytest.MonkeyPatch, path: str, statuses: set[int], payload_key: str
) -> None:
    # The Rust front relays a remote monitor's probe without the token, so
    # without a proof; it gets the probe's own answer, minus local-only detail.
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "proxy-token")
    client, _ = _client()
    local = client.get(path)

    resp = client.get(path, headers={"Host": RELAY_HOST})

    assert resp.status_code in statuses, resp.text
    assert resp.status_code == local.status_code
    body = resp.json()
    assert body["service"] == local.json()["service"]
    assert payload_key in body
    assert "config" not in body  # operator config is for local callers only


def test_rust_front_relay_from_a_container_gateway_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A front in a container on Linux reaches this proxy from the bridge
    # gateway, not loopback; its proof stands in for the token all the same.
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "proxy-token")
    client, transport = _client(peer="172.17.0.1")

    resp = client.get(
        _PATH,
        headers={
            "Host": RELAY_HOST,
            "X-Headroom-Front-Remote": "1",
            FRONT_RELAY_PROOF_HEADER: front_relay_proof("proxy-token"),
        },
    )

    assert resp.status_code == 200, resp.text
    assert len(transport.requests) == 1
    assert FRONT_RELAY_PROOF_HEADER not in {k.lower() for k in transport.requests[0].headers}


def test_front_relay_proof_matches_the_rust_front() -> None:
    # The same vector as relay_proof_matches_python in
    # crates/headroom-proxy/src/inbound_auth.rs: both processes must agree.
    assert (
        front_relay_proof("abc")
        == "33d9db219abd5b185118a8980948fe91930384fdf1a44fa3a964fcdb670be2e3"
    )


@pytest.mark.parametrize("host", ["host.docker.internal:8787", "host.containers.internal:8787"])
def test_rust_front_relay_to_a_container_alias_upstream_is_accepted(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    # A containerised front dialling this proxy as a container alias sends
    # that alias as Host on a WebSocket handshake, with the marker and the
    # proof (the Rust test relay_identity_for_a_container_alias_upstream).
    # The marker takes local trust away; the proof authenticates the caller.
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "abc")
    client, transport = _client()
    marked = {"Host": host, "X-Headroom-Front-Remote": "1"}

    proved = client.get(
        _PATH, headers={**marked, FRONT_RELAY_PROOF_HEADER: front_relay_proof("abc")}
    )
    unproved = client.get(_PATH, headers=marked)

    assert proved.status_code == 200, proved.text
    assert unproved.status_code == 403, unproved.text
    assert len(transport.requests) == 1


def test_rust_front_relay_with_its_proof_forwards_without_the_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "proxy-token")
    client, transport = _client()

    resp = client.get(
        _PATH,
        headers={"Host": RELAY_HOST, FRONT_RELAY_PROOF_HEADER: front_relay_proof("proxy-token")},
    )

    assert resp.status_code == 200, resp.text
    assert len(transport.requests) == 1
    assert FRONT_RELAY_PROOF_HEADER not in transport.requests[0].headers


def test_proxy_token_is_required_for_a_listed_non_loopback_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A same-host reverse proxy relays remote callers from loopback: its listed
    # name lets them past the Host check but not past the token.
    monkeypatch.setenv("HEADROOM_WS_ORIGINS", "https://dash.example")
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "proxy-token")
    client, transport = _client()

    refused = client.get(_PATH, headers={"Host": "dash.example"})
    local = client.get(_PATH)
    with_token = client.get(
        _PATH, headers={"Host": "dash.example", "X-Headroom-Proxy-Token": "proxy-token"}
    )

    assert refused.status_code == 401
    assert local.status_code == 200, local.text
    assert with_token.status_code == 200, with_token.text
    assert len(transport.requests) == 2


def test_origin_listed_only_for_cors_still_forwards_when_both_lists_are_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_WS_ORIGINS", "https://ws.example")
    monkeypatch.setenv("HEADROOM_CORS_ORIGINS", "https://dash.example")
    client, transport = _client()

    resp = _simple_post(client, "/v1/messages", {"Origin": "https://dash.example"})

    assert resp.status_code == 200, resp.text
    assert len(transport.requests) == 1
