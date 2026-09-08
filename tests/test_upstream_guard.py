"""Tests for the SSRF upstream guard (WEB-01).

All cases use IP literals or ``localhost`` so no external network is required.
"""

from __future__ import annotations

import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from headroom.providers.proxy_targets import select_passthrough_base_url
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.upstream_guard import is_safe_upstream_url


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://127.0.0.1:8080/admin",  # loopback
        "http://10.0.0.1:8080/",  # RFC1918
        "http://192.168.1.10/",  # RFC1918
        "http://172.16.0.1/",  # RFC1918
        "https://localhost/v1",  # resolves to loopback
        "http://[::1]/",  # IPv6 loopback
        "ftp://example.com/",  # non-http(s)/ws scheme
        "not-a-url",
        "",
    ],
)
def test_blocks_internal_and_invalid(url: str) -> None:
    assert is_safe_upstream_url(url) is False


@pytest.mark.parametrize("url", ["https://8.8.8.8/v1", "https://1.1.1.1/", "wss://9.9.9.9/rt"])
def test_allows_public(url: str) -> None:
    assert is_safe_upstream_url(url) is True


@pytest.mark.parametrize(
    ("label", "url"),
    [
        # RFC 6598 shared address space: `is_private` does not flag it, but it
        # routes to ISP and cloud-internal infrastructure.
        ("shared address space", "http://100.64.0.1/"),
        ("shared address space top", "http://100.127.255.254/"),
        ("benchmarking", "http://198.18.0.1/"),
        ("TEST-NET-1", "http://192.0.2.1/"),
        ("TEST-NET-3", "http://203.0.113.1/"),
        ("reserved 240/4", "http://240.0.0.1/"),
        ("IETF protocol assignments", "http://192.0.0.1/"),
        # IPv6 forms that embed an internal IPv4 address.
        ("6to4 embedding loopback", "http://[2002:7f00:1::]/"),
        ("6to4 embedding RFC1918", "http://[2002:a00:1::]/"),
        ("NAT64 embedding loopback", "http://[64:ff9b::7f00:1]/"),
        ("NAT64 local-use prefix", "http://[64:ff9b:1::7f00:1]/"),
        ("teredo", "http://[2001:0::7f00:1]/"),
        ("IPv4-mapped metadata", "http://[::ffff:169.254.169.254]/"),
        ("IPv4-mapped loopback", "http://[::ffff:127.0.0.1]/"),
        # Credential-prefix confusion: the authority is what counts.
        ("userinfo before loopback", "http://api.openai.com@127.0.0.1/"),
    ],
)
def test_blocks_non_globally_routable_and_embedded_forms(label: str, url: str) -> None:
    assert is_safe_upstream_url(url) is False, label


def test_multicast_is_still_blocked() -> None:
    """`is_global` is True for multicast, so the category checks must remain."""
    assert is_safe_upstream_url("http://224.0.0.1/") is False


def test_dns_failure_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_resolution(*args: object, **kwargs: object) -> list[object]:
        raise socket.gaierror("temporary failure")

    monkeypatch.setattr(socket, "getaddrinfo", fail_resolution)
    assert is_safe_upstream_url("https://temporarily-unresolved.example/v1") is False


def test_allowlist_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "api.internal.example, https://llm.corp:8443")
    # Allowlisted hosts pass — including internal ones the operator opted into,
    # without a DNS lookup.
    assert is_safe_upstream_url("https://api.internal.example/v1") is True
    assert is_safe_upstream_url("https://llm.corp:8443/v1") is True
    # URL entries are exact origins, not implicit host-wide grants.
    assert is_safe_upstream_url("https://llm.corp:22/v1") is False
    assert is_safe_upstream_url("http://llm.corp:8443/v1") is False
    assert is_safe_upstream_url("https://llm.corp/v1") is False
    # Anything not on the list is rejected in allowlist mode, even public hosts.
    assert is_safe_upstream_url("https://8.8.8.8/v1") is False
    assert is_safe_upstream_url("https://api.openai.com/v1") is False


# ---------------------------------------------------------------------------
# Enforcement at the sinks (CVE-2026-77775).
#
# The tests above cover `is_safe_upstream_url` in isolation. They passed while
# `/v1/alpha/search` still forwarded to any caller-named host, because nothing
# asserted the guard was actually *reached*. `select_passthrough_base_url`
# returns the `x-headroom-base-url` value whenever an `api-key` header is
# present -- both attacker-supplied -- so every caller of it is a sink.
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _InternalService:
    """Stands in for an internal host the caller should never be able to reach."""

    def __init__(self) -> None:
        self.hits: list[str] = []
        self.port = _free_port()
        hits = self.hits

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                hits.append(self.path)
                body = b'{"secret":"internal-only"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST  # noqa: N815

            def log_message(self, *args: object) -> None:
                return

        self._server = HTTPServer(("127.0.0.1", self.port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> _InternalService:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _app():  # noqa: ANN202
    return create_app(
        ProxyConfig(
            host="127.0.0.1",
            port=_free_port(),
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
        )
    )


def test_alpha_search_rejects_a_caller_named_loopback_upstream() -> None:
    """The route that shipped unguarded. 400, and the host is never contacted."""
    with _InternalService() as internal, TestClient(_app()) as client:
        response = client.post(
            "/v1/alpha/search",
            headers={
                "api-key": "attacker-supplied",
                "Authorization": "Bearer client-token",
                "x-headroom-base-url": internal.url,
            },
            json={"query": "x"},
        )

    assert response.status_code == 400
    assert internal.hits == [], "proxy forwarded to a loopback address"
    assert "internal-only" not in response.text


def test_no_route_forwards_to_a_loopback_upstream() -> None:
    """Sweep the whole route table -- the guard must hold everywhere.

    This is the generalisation of the fix: a future route that resolves a
    caller-named upstream without validating it fails here rather than in a
    CVE.
    """
    app = _app()
    probes: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if not path:
            continue
        path = re.sub(r"\{[^}]+\}", "probe", path)
        for method in ("POST", "GET"):
            if method in methods:
                probes.add((method, path))
                break

    assert len(probes) > 50, "route discovery found suspiciously few routes"

    with _InternalService() as internal, TestClient(app) as client:
        for method, path in sorted(probes):
            for unlock in ({"api-key": "x"}, {"x-goog-api-key": "x"}):
                headers = {**unlock, "x-headroom-base-url": internal.url}
                try:
                    client.request(method, path, headers=headers, json={"q": "x"})
                except Exception:  # noqa: BLE001 - route errors are not the subject
                    pass
        reached = list(internal.hits)

    assert reached == [], f"routes forwarded to a loopback upstream: {reached}"


class _StubProxy:
    """Minimal stand-in for the proxy object `select_passthrough_base_url` reads."""

    class provider_runtime:  # noqa: N801
        @staticmethod
        def model_metadata_provider(headers: object) -> str:
            return "openai"

        @staticmethod
        def api_target(name: str) -> str:
            return "https://api.openai.com"


def test_passthrough_base_url_ignores_an_unsafe_azure_override() -> None:
    headers = {"api-key": "x", "x-headroom-base-url": "http://169.254.169.254"}

    resolved = select_passthrough_base_url(_StubProxy(), headers)

    assert "169.254.169.254" not in resolved


def test_passthrough_base_url_still_honours_a_safe_azure_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legitimate BYOK must keep working -- this is not a blanket block."""

    def public_resolution(*args: object, **kwargs: object) -> list[object]:
        return [(None, None, None, None, ("20.10.10.10", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", public_resolution)
    headers = {
        "api-key": "x",
        "x-headroom-base-url": "https://my-resource.openai.azure.com/",
    }

    resolved = select_passthrough_base_url(_StubProxy(), headers)

    assert resolved == "https://my-resource.openai.azure.com"


def test_operator_allowlist_still_permits_an_internal_azure_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On-prem/split-horizon deployments opt in explicitly rather than being stuck."""
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "gateway.internal")
    headers = {"api-key": "x", "x-headroom-base-url": "https://gateway.internal/v1"}

    assert select_passthrough_base_url(_StubProxy(), headers) == "https://gateway.internal/v1"


def test_slow_resolution_is_bounded_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hostile hostname must not hold the caller for the resolver's timeout.

    `socket.getaddrinfo` takes no timeout and runs on the calling thread, which
    for the proxy is the event loop -- so an unbounded lookup is an
    unauthenticated stall of every in-flight request.
    """
    import time as _time

    def slow_resolution(*args: object, **kwargs: object) -> list[object]:
        _time.sleep(5.0)
        return [(None, None, None, None, ("8.8.8.8", 443))]

    monkeypatch.setenv("HEADROOM_UPSTREAM_RESOLVE_TIMEOUT_S", "0.25")
    monkeypatch.setattr(socket, "getaddrinfo", slow_resolution)

    started = _time.perf_counter()
    result = is_safe_upstream_url("https://slow.example/v1")
    elapsed = _time.perf_counter() - started

    assert result is False, "a lookup that overruns its budget must fail closed"
    assert elapsed < 2.0, f"resolution was not bounded (took {elapsed:.2f}s)"


async def test_async_guard_matches_the_sync_policy() -> None:
    """The off-loop wrapper must not diverge from the blocking form."""
    from headroom.proxy.upstream_guard import is_safe_upstream_url_async

    assert await is_safe_upstream_url_async("http://127.0.0.1/") is False
    assert await is_safe_upstream_url_async("http://169.254.169.254/") is False
    assert await is_safe_upstream_url_async("https://8.8.8.8/v1") is True
