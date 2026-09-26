"""Tests for the SSRF upstream guard (WEB-01).

All cases use IP literals or ``localhost`` so no external network is required.
"""

from __future__ import annotations

import contextlib
import re
import socket
import ssl
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpcore
import httpx
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


def _app(**overrides: object):  # noqa: ANN202
    return create_app(
        ProxyConfig(
            host="127.0.0.1",
            port=_free_port(),
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            **overrides,  # type: ignore[arg-type]
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


# ---------------------------------------------------------------------------
# DNS rebinding (A-4).
#
# The guard used to resolve the hostname, judge the answer, and then hand the
# *name* to httpx -- which resolves it a second time when it opens the socket.
# An attacker who controls the authoritative DNS for that name simply answers
# the two lookups differently: a public address for the check, an internal one
# for the connection. Every check above still passes while the proxy talks to
# 127.0.0.1, so the tests here assert on where the socket actually went rather
# than on what `is_safe_upstream_url` returned.
# ---------------------------------------------------------------------------


# A path no route claims, so it lands on the catch-all passthrough -- the sink
# that forwards to `x-headroom-base-url` verbatim once the guard has cleared it.
_PROBE_PATH = "/v1/rebind-probe"


class _RebindingResolver:
    """A `getaddrinfo` that answers the first lookup differently from the rest.

    Only the attacker-controlled hostname is intercepted; every other name is
    delegated to the real resolver, so patching this in globally -- which is
    what it takes to reach asyncio's own resolution, not just the guard's --
    leaves the rest of the process alone.
    """

    def __init__(self, hostname: str, first: str, then: str) -> None:
        self.hostname = hostname
        self.first = first
        self.then = then
        self.calls = 0
        self._real = socket.getaddrinfo
        self._lock = threading.Lock()

    def __call__(self, host: object, port: object, *args: object, **kwargs: object) -> list:
        # anyio hands the resolver an ASCII/IDNA-encoded name, the guard a str.
        name = host.decode("ascii") if isinstance(host, bytes) else host
        if name != self.hostname:
            return self._real(host, port, *args, **kwargs)  # type: ignore[arg-type]
        with self._lock:
            self.calls += 1
            address = self.first if self.calls == 1 else self.then
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (address, port or 0),
            )
        ]


def test_rebinding_after_the_check_cannot_move_the_forward_to_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The validated address is the one dialled, not whatever DNS says later.

    The first lookup (the guard's) answers with a public address so the URL
    passes validation; every later one answers with the internal service.
    Before the pin, httpx re-resolved and delivered both the request and the
    response to loopback.
    """
    resolver = _RebindingResolver("rebind.example", first="8.8.8.8", then="127.0.0.1")
    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    with (
        _InternalService() as internal,
        TestClient(_app(connect_timeout_seconds=1)) as client,
    ):
        try:
            response = client.post(
                _PROBE_PATH,
                headers={"x-headroom-base-url": f"http://rebind.example:{internal.port}"},
                json={"query": "x"},
            )
            status, text = response.status_code, response.text
        except Exception as exc:  # noqa: BLE001 - an upstream failure is a pass here
            status, text = 0, repr(exc)

    assert resolver.calls >= 1, "the guard never resolved the hostname"
    assert status != 400, f"the guard rejected the URL; the pin was never exercised: {text}"
    assert status != 404, f"the route did not forward at all: {text}"
    assert internal.hits == [], "the connection followed the rebound answer to loopback"
    assert "internal-only" not in text


def _self_signed_cert(tmp_path: Path, hostname: str) -> tuple[str, str]:
    """Write a self-signed leaf for ``hostname``; return (cert path, key path).

    The certificate is its own issuer, so the same file doubles as the trust
    bundle the proxy is pointed at through ``SSL_CERT_FILE``.
    """
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return str(cert_path), str(key_path)


class _TLSUpstream:
    """An HTTPS upstream on loopback that records the SNI and Host it was sent."""

    def __init__(self, cert_pem: str, key_pem: str) -> None:
        self.sni: list[str | None] = []
        self.host_headers: list[str] = []
        host_headers = self.host_headers

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                host_headers.append(self.headers.get("Host", ""))
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST  # noqa: N815

            def log_message(self, *args: object) -> None:
                return

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        # Nothing here is about protocol versions -- the assertions are on SNI,
        # the Host header, and certificate verification -- but a bare
        # `PROTOCOL_TLS_SERVER` context nominally admits TLS 1.0/1.1, which is a
        # finding in its own right. Pinning the floor keeps the test's subject
        # unchanged and says out loud which versions it means.
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert_pem, key_pem)
        context.sni_callback = lambda _sock, name, _ctx: self.sni.append(name)
        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self._server.socket = context.wrap_socket(self._server.socket, server_side=True)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> _TLSUpstream:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()


def _tls_rebinding_app(
    monkeypatch: pytest.MonkeyPatch,
    resolver: _RebindingResolver,
    ca_pem: str,
):  # noqa: ANN202
    """Point the proxy at ``ca_pem`` and at ``resolver``'s answers.

    ``_is_internal_address`` is neutered so a loopback answer passes
    validation: the subject of these two tests is what happens *after* an
    address is accepted, and faking it this way keeps the upstream on 127.0.0.1
    instead of requiring real egress to a genuinely public address.
    """
    from headroom.proxy import upstream_guard

    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    monkeypatch.setattr(upstream_guard, "_is_internal_address", lambda _ip: False)
    monkeypatch.setenv("SSL_CERT_FILE", ca_pem)
    return _app(connect_timeout_seconds=1)


def test_pinned_connection_presents_the_original_hostname_to_tls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dialling an IP literal must not cost us SNI, the Host header, or the cert.

    The upstream's certificate names `pinned.example` only, so a handshake
    verified against the pinned address -- or one that relaxed verification to
    make pinning work at all -- fails here. And because the second DNS answer
    points off-box, a request arriving at all is proof the first, validated
    answer is what was dialled.
    """
    cert_pem, key_pem = _self_signed_cert(tmp_path, "pinned.example")
    resolver = _RebindingResolver("pinned.example", first="127.0.0.1", then="8.8.8.8")

    with _TLSUpstream(cert_pem, key_pem) as upstream:
        app = _tls_rebinding_app(monkeypatch, resolver, cert_pem)
        with TestClient(app) as client:
            try:
                status = client.post(
                    _PROBE_PATH,
                    headers={"x-headroom-base-url": f"https://pinned.example:{upstream.port}"},
                    json={"query": "x"},
                ).status_code
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"pinned TLS request never reached the upstream: {exc!r}")

    assert status == 200, "the pinned upstream was not reached"
    assert upstream.sni == ["pinned.example"], f"wrong SNI: {upstream.sni}"
    assert upstream.host_headers, "no request arrived at the pinned upstream"
    assert upstream.host_headers[0].startswith("pinned.example"), (
        f"Host header was rewritten to the pinned address: {upstream.host_headers[0]}"
    )


def test_pinning_does_not_disable_certificate_hostname_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The mirror of the test above: a cert for the wrong name must still fail.

    Identical setup with one difference -- the upstream's certificate names
    `other.example` while the caller asked for `pinned.example`. Pinning
    implemented by rewriting the URL to the address without carrying the
    hostname into the handshake, or by loosening `verify`, would pass this
    request straight through.
    """
    cert_pem, key_pem = _self_signed_cert(tmp_path, "other.example")
    resolver = _RebindingResolver("pinned.example", first="127.0.0.1", then="8.8.8.8")

    with _TLSUpstream(cert_pem, key_pem) as upstream:
        app = _tls_rebinding_app(monkeypatch, resolver, cert_pem)
        with TestClient(app) as client:
            try:
                status = client.post(
                    _PROBE_PATH,
                    headers={"x-headroom-base-url": f"https://pinned.example:{upstream.port}"},
                    json={"query": "x"},
                ).status_code
            except Exception:  # noqa: BLE001 - a refused handshake is the pass
                status = 0

    assert status != 200, "a certificate for the wrong hostname was accepted"
    assert upstream.host_headers == [], "the request reached a mis-named upstream"


def test_a_check_pins_every_address_it_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """All of them, in resolver order -- see the IPv6-fallback note on the pin."""
    from headroom.proxy import upstream_guard

    def two_answers(*args: object, **kwargs: object) -> list[object]:
        return [
            (None, None, None, None, ("8.8.8.8", 443)),
            (None, None, None, None, ("1.1.1.1", 443)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", two_answers)
    upstream_guard.clear_validated_addresses()

    assert is_safe_upstream_url("https://Multi.Homed.Example/v1") is True
    # Hostnames are case-insensitive; the connection will ask in lower case.
    assert upstream_guard.validated_addresses("multi.homed.example") == ("8.8.8.8", "1.1.1.1")


def test_a_rejected_or_allowlisted_destination_pins_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an address that was judged may be pinned.

    A rejected destination is never dialled, and an allowlisted one is admitted
    by name without resolving at all -- pinning either would be recording a
    verdict that was not reached.
    """
    from headroom.proxy import upstream_guard

    def internal_answer(*args: object, **kwargs: object) -> list[object]:
        return [(None, None, None, None, ("10.1.2.3", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", internal_answer)
    upstream_guard.clear_validated_addresses()

    assert is_safe_upstream_url("https://rejected.example/v1") is False
    assert upstream_guard.validated_addresses("rejected.example") is None

    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "gateway.internal")
    assert is_safe_upstream_url("https://gateway.internal/v1") is True
    assert upstream_guard.validated_addresses("gateway.internal") is None


def test_an_expired_pin_still_marks_the_destination_as_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lapsed addresses, but the record that the guard ran must survive.

    An aged-out pin that simply vanished would be indistinguishable from a host
    that was never checked, and "never checked" resolves by name. Keeping the
    record is what lets the connect path deny instead.
    """
    from headroom.proxy import upstream_guard

    def public_answer(*args: object, **kwargs: object) -> list[object]:
        return [(None, None, None, None, ("8.8.8.8", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", public_answer)
    monkeypatch.setattr(upstream_guard, "_PIN_TTL_SECONDS", 0.0)
    upstream_guard.clear_validated_addresses()

    assert is_safe_upstream_url("https://briefly.example/v1") is True
    # No usable addresses...
    assert upstream_guard.validated_addresses("briefly.example") is None
    # ...but still, unmistakably, a destination this request had guarded.
    pin = upstream_guard.guarded_pin("briefly.example")
    assert pin is not None, "an expired pin must not decay into 'never checked'"
    assert pin.is_expired() is True
    assert pin.addresses == ("8.8.8.8",)


def test_a_pin_is_scoped_to_the_request_that_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """One caller's pin is not visible to another caller's connection.

    The check runs inside a copied context, so the pin it writes stays there.
    A process-global store keyed by hostname leaked it to every other in-flight
    request instead, which is both a cross-tenant leak and the reason expiry
    could not be made to mean anything.
    """
    import contextvars

    from headroom.proxy import upstream_guard

    def public_answer(*args: object, **kwargs: object) -> list[object]:
        return [(None, None, None, None, ("8.8.8.8", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", public_answer)
    upstream_guard.clear_validated_addresses()

    other_request = contextvars.copy_context()
    assert other_request.run(is_safe_upstream_url, "https://tenant-a.example/v1") is True
    assert other_request.run(upstream_guard.validated_addresses, "tenant-a.example") == ("8.8.8.8",)
    assert upstream_guard.guarded_pin("tenant-a.example") is None, (
        "another request's pin was visible here"
    )


def test_the_pin_scope_is_bounded_and_overflow_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hostnames are caller-supplied, so a scope cannot grow without limit.

    Overflow is a rejection rather than an eviction: an evicted host would pass
    validation while leaving the connection to resolve the name itself, which is
    the failure mode this whole module exists to remove.
    """
    from headroom.proxy import upstream_guard

    def public_answer(*args: object, **kwargs: object) -> list[object]:
        return [(None, None, None, None, ("8.8.8.8", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", public_answer)
    monkeypatch.setattr(upstream_guard, "_PIN_MAX_ENTRIES", 4)
    upstream_guard.clear_validated_addresses()

    accepted = [index for index in range(20) if is_safe_upstream_url(f"https://f{index}.example/")]

    assert accepted == [0, 1, 2, 3]
    assert len(upstream_guard.begin_pin_scope()) == 4
    assert upstream_guard.validated_addresses("f0.example") == ("8.8.8.8",)
    assert upstream_guard.guarded_pin("f19.example") is None
    # Re-checking a host already in the scope refreshes it rather than
    # overflowing, so a retry of the same upstream is never spuriously refused.
    assert is_safe_upstream_url("https://f0.example/") is True


# ---------------------------------------------------------------------------
# The connect-time half: what the backend does with each of the three states it
# can find a destination in -- unguarded, pinned, guarded-but-lapsed -- and what
# happens when the route cannot honour a pin at all.
# ---------------------------------------------------------------------------


class _RecordingBackend(httpcore.AsyncNetworkBackend):
    """Inner backend that records what it was asked to dial and never dials it."""

    def __init__(self) -> None:
        self.targets: list[str] = []

    async def connect_tcp(self, host: str, port: int, *args: object, **kwargs: object) -> object:
        self.targets.append(host)
        return object()


async def _connect(host: str) -> tuple[_RecordingBackend, object | None, Exception | None]:
    from headroom.proxy.upstream_pinning import PinnedAddressBackend

    inner = _RecordingBackend()
    backend = PinnedAddressBackend(inner)
    try:
        return inner, await backend.connect_tcp(host, 443), None
    except Exception as exc:  # noqa: BLE001 - the refusal is the subject
        return inner, None, exc


async def test_a_pin_expiring_before_the_socket_opens_denies_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewer's case: a guarded request outliving the pin must be refused.

    A request can legitimately sit between the guard's verdict and its socket
    for longer than the TTL -- pool saturation, connection limits, a slow
    upstream ahead of it in the queue. When it finally connects, the hostname is
    still attacker-controlled, so re-resolving it is exactly the rebinding this
    module prevents. Denial is the only safe answer; the caller can retry, which
    re-checks and re-pins.
    """
    from headroom.proxy import upstream_guard
    from headroom.proxy.upstream_pinning import UnpinnableUpstreamError

    def public_answer(*args: object, **kwargs: object) -> list[object]:
        return [(None, None, None, None, ("8.8.8.8", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", public_answer)
    upstream_guard.clear_validated_addresses()
    assert is_safe_upstream_url("https://rebind.example/v1") is True

    # The socket is opened while the request waits; wind the pin past its TTL.
    scope = upstream_guard.begin_pin_scope()
    lapsed = scope["rebind.example"]
    scope["rebind.example"] = upstream_guard.UpstreamPin(
        lapsed.addresses, lapsed.expires_at - upstream_guard._PIN_TTL_SECONDS - 1.0
    )

    inner, stream, error = await _connect("rebind.example")

    assert stream is None, "an expired pin was treated as an unchecked destination"
    assert isinstance(error, UnpinnableUpstreamError), f"wrong failure: {error!r}"
    assert inner.targets == [], (
        f"the hostname was handed to the resolver a second time: {inner.targets}"
    )


async def test_a_live_pin_dials_the_address_and_an_unguarded_host_dials_the_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two states either side of the denial, so the denial is not blanket."""
    from headroom.proxy import upstream_guard

    def public_answer(*args: object, **kwargs: object) -> list[object]:
        return [(None, None, None, None, ("8.8.8.8", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", public_answer)
    upstream_guard.clear_validated_addresses()
    assert is_safe_upstream_url("https://pinned.example/v1") is True

    inner, stream, error = await _connect("pinned.example")
    assert error is None and stream is not None
    assert inner.targets == ["8.8.8.8"], "a live pin did not reach the socket"

    # An operator-configured upstream never passes through the guard, has no
    # pin, and must keep resolving by name exactly as it always has.
    inner, stream, error = await _connect("api.anthropic.com")
    assert error is None and stream is not None
    assert inner.targets == ["api.anthropic.com"]


def _client_through_proxy(proxy_url: str) -> httpx.AsyncClient:
    """A client routing everything through ``proxy_url``, SOCKS included.

    All three transports build a real pool. SOCKS needs httpx's ``socks`` extra
    (``socksio``), which is in the dev dependencies for exactly this test: without
    it httpcore substitutes a stub ``AsyncSOCKSProxy`` that cannot be constructed,
    and the case degrades into asserting on a type instead of exercising the pool
    the refusal actually has to classify.
    """
    return httpx.AsyncClient(proxy=proxy_url)


@pytest.mark.parametrize(
    ("label", "proxy_url"),
    [
        ("http", "http://proxy.internal:3128"),
        ("https", "https://proxy.internal:3129"),
        ("socks", "socks5://proxy.internal:1080"),
    ],
)
async def test_a_guarded_upstream_through_a_proxy_is_refused(
    monkeypatch: pytest.MonkeyPatch, label: str, proxy_url: str
) -> None:
    """Every supported proxy transport refuses a guarded upstream, loudly.

    A proxy is handed the target *by name* -- in the request line, in CONNECT,
    in the SOCKS5 address -- and resolves it itself, on its own network, after
    the guard ran. Pinning the proxy's address does nothing about that, so the
    honest answer is a refusal rather than a forward on a verdict nothing
    enforces.
    """
    from headroom.proxy import upstream_guard
    from headroom.proxy.upstream_pinning import (
        GuardedUpstreamRefusingTransport,
        UnpinnableUpstreamError,
        install_upstream_pinning,
    )

    def public_answer(*args: object, **kwargs: object) -> list[object]:
        return [(None, None, None, None, ("8.8.8.8", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", public_answer)
    upstream_guard.clear_validated_addresses()

    client = install_upstream_pinning(_client_through_proxy(proxy_url))
    try:
        transport = client._transport_for_url(httpx.URL("https://rebind.example/v1"))
        assert isinstance(transport, GuardedUpstreamRefusingTransport), (
            f"the {label} proxy transport was left able to resolve the target itself"
        )

        # Unguarded traffic is untouched: it never reaches the refusal at all.
        assert upstream_guard.guarded_pin("rebind.example") is None

        assert is_safe_upstream_url("https://rebind.example/v1") is True
        with pytest.raises(UnpinnableUpstreamError) as refused:
            await transport.handle_async_request(httpx.Request("POST", "https://rebind.example/v1"))
    finally:
        # The stubbed SOCKS pool above was never initialised, so it has no
        # connection state to close.
        with contextlib.suppress(AttributeError):
            await client.aclose()

    assert "rebind.example" in str(refused.value)
    assert "proxy" in str(refused.value)


class _RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.hosts: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.hosts.append(request.url.host)
        return httpx.Response(200)


async def test_opt_in_forwards_a_guarded_upstream_through_a_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HEADROOM_ALLOW_PROXIED_GUARDED_UPSTREAMS restores the name verdict on proxy routes.

    A single-user proxy whose ``x-headroom-base-url`` callers are local agent
    integrations cannot allowlist every provider its user configures, so behind
    a corporate proxy the refusal broke them outright. The opt-in forwards the
    request through the proxy once the guard has judged the name.
    """
    from headroom.proxy import upstream_guard
    from headroom.proxy.upstream_pinning import (
        GuardedUpstreamRefusingTransport,
        install_upstream_pinning,
    )

    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("8.8.8.8", 443))]
    )
    monkeypatch.setenv("HEADROOM_ALLOW_PROXIED_GUARDED_UPSTREAMS", "1")
    upstream_guard.clear_validated_addresses()

    client = install_upstream_pinning(_client_through_proxy("http://proxy.internal:3128"))
    try:
        transport = client._transport_for_url(httpx.URL("https://api.example/v1"))
        assert isinstance(transport, GuardedUpstreamRefusingTransport)
        recorder = _RecordingTransport()
        transport._inner = recorder

        assert is_safe_upstream_url("https://api.example/v1") is True
        response = await transport.handle_async_request(
            httpx.Request("POST", "https://api.example/v1")
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    assert recorder.hosts == ["api.example"]


async def test_opt_in_keeps_every_other_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in covers proxy routes only, and never admits an internal answer."""
    from headroom.proxy import upstream_guard
    from headroom.proxy.upstream_pinning import (
        GuardedUpstreamRefusingTransport,
        UnpinnableUpstreamError,
    )

    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("8.8.8.8", 443))]
    )
    monkeypatch.setenv("HEADROOM_ALLOW_PROXIED_GUARDED_UPSTREAMS", "1")
    upstream_guard.clear_validated_addresses()

    assert is_safe_upstream_url("https://api.example/v1") is True
    poolless = GuardedUpstreamRefusingTransport(_RecordingTransport(), "no pool")
    with pytest.raises(UnpinnableUpstreamError):
        await poolless.handle_async_request(httpx.Request("POST", "https://api.example/v1"))

    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("10.0.0.5", 443))]
    )
    upstream_guard.clear_validated_addresses()
    assert is_safe_upstream_url("https://internal.example/v1") is False


async def test_a_proxied_client_still_pins_its_direct_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuring a proxy for some traffic must not unpin the rest.

    httpx keeps the direct pool as the primary transport and mounts the proxy
    per URL pattern, so a `NO_PROXY`-style carve-out still leaves through the
    pinned path -- and the refusal above applies only to what actually routes
    via the proxy.
    """
    from headroom.proxy import upstream_guard
    from headroom.proxy.upstream_pinning import (
        GuardedUpstreamRefusingTransport,
        PinnedAddressBackend,
        install_upstream_pinning,
    )

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
    monkeypatch.setenv("NO_PROXY", "direct.example")
    upstream_guard.clear_validated_addresses()

    client = install_upstream_pinning(httpx.AsyncClient())
    try:
        direct = client._transport_for_url(httpx.URL("https://direct.example/v1"))
        assert not isinstance(direct, GuardedUpstreamRefusingTransport)
        assert isinstance(direct._pool._network_backend, PinnedAddressBackend)

        proxied = client._transport_for_url(httpx.URL("https://elsewhere.example/v1"))
        assert isinstance(proxied, GuardedUpstreamRefusingTransport)
    finally:
        await client.aclose()
