from __future__ import annotations

import asyncio
import json
import logging
import socket
import ssl
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlencode

import httpx
import pytest
import websockets
from fastapi.testclient import TestClient
from pydantic import ValidationError

from headroom.proxy.gateway.config import CredentialConfig, GatewayConfigSnapshot
from headroom.proxy.gateway.credentials import CredentialBroker, CredentialLease, SecretHandle
from headroom.proxy.gateway.egress import EgressPolicy
from headroom.proxy.gateway.errors import GatewayEgressDenied
from headroom.proxy.gateway.oauth import BrowserAuthorizationTransaction, OAuthValidationError
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app

EXAMPLES = Path(__file__).parents[2] / "docs/proposals/unified-api-gateway/examples"


def example():
    return json.loads((EXAMPLES / "gateway.api-keys.json").read_text())


@pytest.mark.parametrize(
    "provider", ["openai", "anthropic", "gemini", "compatible", "vertex", "bedrock"]
)
@pytest.mark.parametrize("kind", ["env", "gcp-adc", "aws-chain", "none"])
def test_config_rejects_every_unadmitted_provider_source_pair(provider, kind, monkeypatch):
    allowed = {
        "openai": "env",
        "anthropic": "env",
        "gemini": "env",
        "compatible": "env",
        "vertex": "gcp-adc",
        "bedrock": "aws-chain",
    }
    if allowed[provider] == kind:
        return

    def forbidden(*args, **kwargs):
        pytest.fail("offline validation accessed DNS or credentials")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    raw = example()["credentials"][0]
    raw.update(
        provider=provider,
        source={
            "env": {"kind": "env", "ref": "KEY"},
            "gcp-adc": {"kind": "gcp-adc", "project": "p"},
            "aws-chain": {"kind": "aws-chain", "region": "us-east-1", "profile": "p"},
            "none": {"kind": "none"},
        }[kind],
        refresh_owner="sdk" if kind in {"gcp-adc", "aws-chain"} else "none",
    )
    with pytest.raises(ValidationError):
        CredentialConfig.model_validate(raw)


@pytest.mark.parametrize(
    "mutation",
    ["public-host", "private-public", "vertex-project", "vertex-region", "bedrock-region"],
)
def test_config_rejects_incompatible_provider_audience(mutation):
    raw = (
        example()
        if mutation in {"public-host", "private-public"}
        else json.loads((EXAMPLES / "gateway.cloud-identities.json").read_text())
    )
    if mutation == "public-host":
        raw["credentials"][0]["allowed_origins"] = ["https://attacker.example:443"]
        raw["routes"][0]["upstream_origin"] = "https://attacker.example:443"
    elif mutation == "private-public":
        raw["routes"][0]["private_network"] = True
    elif mutation == "vertex-project":
        raw["credentials"][0]["source"]["project"] = "other-project"
    elif mutation == "vertex-region":
        raw["credentials"][0]["allowed_path_prefixes"] = [
            "/v1/projects/REPLACE_WITH_AUTHORIZED_PROJECT/locations/europe-west1/"
        ]
        raw["routes"][0]["upstream_path_prefix"] = raw["credentials"][0]["allowed_path_prefixes"][0]
    else:
        raw["credentials"][1]["source"]["region"] = "eu-west-1"
    with pytest.raises(ValidationError):
        GatewayConfigSnapshot.model_validate(raw)


def lease():
    return CredentialLease(
        "openai-api",
        "openai",
        "openai-api",
        ("https://api.openai.com:443",),
        ("/v1/",),
        None,
        1,
        SecretHandle("upstream-secret-sentinel"),
    )


def test_egress_without_explicit_addresses_resolves_and_denies_metadata(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))],
    )
    with pytest.raises(GatewayEgressDenied):
        EgressPolicy().authorize(lease(), "https://api.openai.com/v1/responses")


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1/../admin",
        "https://api.openai.com/v1/%2e%2e/admin",
        "https://api.openai.com/v1/%2fadmin",
        "https://api.openai.com/v1/responses#secret",
        "https://api.openai.com/v1/\\admin",
    ],
)
def test_egress_rejects_ambiguous_final_path(url):
    with pytest.raises(GatewayEgressDenied):
        EgressPolicy().authorize(lease(), url, resolved_addresses=("8.8.8.8",))


@pytest.mark.parametrize(
    "url", ["https://api.openai.com:0/v1/responses", "https://api.openai.com/v1"]
)
def test_zero_port_and_path_prefix_root_are_not_authorized(url):
    with pytest.raises(GatewayEgressDenied):
        EgressPolicy().authorize(lease(), url, resolved_addresses=("8.8.8.8",))


@pytest.mark.parametrize(
    "base",
    [
        "http://attacker@127.0.0.1:43119/callback",
        "http://127.0.0.1:43119/callback#fragment",
        "http://127.0.0.1:43119/%63allback",
        "HTTP://127.0.0.1:43119/callback",
        "http://127.0.0.1:043119/callback",
        "http://127.0.0.1:43119/callback?extra=1",
    ],
)
def test_callback_requires_exact_redirect_and_only_protocol_query(base):
    tx = BrowserAuthorizationTransaction.create(
        authorization_endpoint="https://issuer.example/authorize",
        issuer="https://issuer.example",
        client_id="test",
        redirect_uri="http://127.0.0.1:43119/callback",
        scopes=("inference",),
        allowed_account="account",
        lifetime_seconds=60,
        now=100,
    )
    query = urlencode(
        {"code": "code", "state": tx.state, "issuer": tx.issuer, "account": "account"}
    )
    callback = base + ("&" if "?" in base else "?") + query
    if "#" in base:
        callback = base.split("#")[0] + "?" + query + "#fragment"
    with pytest.raises(OAuthValidationError):
        tx.consume(callback_uri=callback, now=101)


def test_callback_rejects_encoded_protocol_query_names():
    tx = BrowserAuthorizationTransaction.create(
        authorization_endpoint="https://issuer.example/authorize",
        issuer="https://issuer.example",
        client_id="test",
        redirect_uri="http://127.0.0.1:43119/callback",
        scopes=("inference",),
        allowed_account="account",
        lifetime_seconds=60,
        now=100,
    )
    query = urlencode(
        {"code": "code", "state": tx.state, "issuer": tx.issuer, "account": "account"}
    ).replace("state=", "%73tate=")
    with pytest.raises(OAuthValidationError):
        tx.consume(callback_uri=tx.redirect_uri + "?" + query, now=101)


@pytest.mark.asyncio
async def test_stale_invalidation_cannot_evict_newer_generation():
    class Source:
        calls = 0
        invalidated = []

        async def acquire(self, *, now):
            self.calls += 1
            return replace(
                lease(), generation=self.calls, expires_at=now + (1 if self.calls == 1 else 3600)
            )

        async def invalidate(self, old, reason):
            self.invalidated.append(old.generation)

    source = Source()
    broker = CredentialBroker({"openai-api": source})
    route = GatewayConfigSnapshot.model_validate(example()).routes[0]
    old = await broker.acquire(route)
    current = await broker.acquire(route)
    await broker.invalidate(old, "late failure")
    assert await broker.acquire(route) is current
    assert source.invalidated == []


@pytest.mark.asyncio
async def test_refresh_publication_and_old_invalidation_are_serialized():
    refreshing, release = asyncio.Event(), asyncio.Event()

    class Source:
        calls = 0
        invalidated = []

        async def acquire(self, *, now):
            self.calls += 1
            if self.calls == 2:
                refreshing.set()
                await release.wait()
            return replace(
                lease(), generation=self.calls, expires_at=now + (1 if self.calls == 1 else 3600)
            )

        async def invalidate(self, old, reason):
            self.invalidated.append(old.generation)

    source = Source()
    broker = CredentialBroker({"openai-api": source})
    route = GatewayConfigSnapshot.model_validate(example()).routes[0]
    old = await broker.acquire(route)
    refresh = asyncio.create_task(broker.acquire(route))
    await refreshing.wait()
    invalidation = asyncio.create_task(broker.invalidate(old, "late failure"))
    await asyncio.sleep(0)
    release.set()
    current = await refresh
    await invalidation
    assert await broker.acquire(route) is current
    assert source.invalidated == []


def app_with_transport(monkeypatch, handler):
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret-sentinel")
    monkeypatch.setenv("OPENAI_API_KEY", "upstream-secret-sentinel")
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.model_validate(example())))
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    return app


@pytest.mark.parametrize("mode", ["body", "exception", "success-header", "redirect"])
def test_real_dispatch_never_reflects_provider_error_or_secret_header(monkeypatch, caplog, mode):
    sentinel = "upstream-secret-sentinel client-secret-sentinel provider-secret-sentinel"

    async def upstream(request):
        if mode == "exception":
            raise httpx.ConnectError(sentinel, request=request)
        return httpx.Response(
            200 if mode == "success-header" else 302 if mode == "redirect" else 400,
            content=b'{"ok":true}' if mode == "success-header" else sentinel.encode(),
            headers={
                "x-provider-debug": sentinel,
                "set-cookie": sentinel,
                "location": "https://evil.example/" + sentinel,
            },
        )

    app = app_with_transport(monkeypatch, upstream)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/v1/chat/completions",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret-sentinel"},
        json={"model": "REPLACE_WITH_ENABLED_OPENAI_MODEL", "messages": []},
    )
    assert response.status_code == (200 if mode == "success-header" else 502)
    combined = (
        response.text
        + str(response.headers)
        + caplog.text
        + str(app.state.gateway_runtime.status())
    )
    assert "secret-sentinel" not in combined


def test_invalid_ca_bundle_rejected_before_app_serves(monkeypatch, tmp_path):
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client")
    invalid_ca = tmp_path / "invalid.pem"
    invalid_ca.write_text("not a CA certificate")
    raw = example()
    raw["transport"] = {"ca_bundle": str(invalid_ca)}
    with pytest.raises(ValueError):
        create_app(ProxyConfig(gateway=GatewayConfigSnapshot.model_validate(raw)))


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "169.254.169.254",
        "10.0.0.1",
        "224.0.0.1",
        "0.0.0.0",
        "::1",
        "fc00::1",
        "::ffff:127.0.0.1",
    ],
)
def test_dispatch_denies_resolved_destinations_before_outbound(monkeypatch, address):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))],
    )
    reached = []

    async def upstream(request):
        reached.append(request)
        return httpx.Response(200, json={"ok": True})

    app = app_with_transport(monkeypatch, upstream)
    response = TestClient(app).post(
        "/v1/chat/completions",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret-sentinel"},
        json={"model": "REPLACE_WITH_ENABLED_OPENAI_MODEL", "messages": []},
    )
    assert response.status_code == 502
    assert reached == []


def test_private_compatible_uses_its_own_key_and_configured_path(monkeypatch):
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret-sentinel")
    monkeypatch.setenv("INTERNAL_LLM_API_KEY", "private-secret")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.10.1.2", 443))],
    )
    raw = json.loads((EXAMPLES / "gateway.private-upstream.json").read_text())
    raw.pop("transport")
    raw["routes"][0]["upstream_path_prefix"] = "/tenant/v1/"
    raw["credentials"][0]["allowed_path_prefixes"] = ["/tenant/v1/"]
    reached = []

    async def upstream(request):
        reached.append(request)
        return httpx.Response(200, json={"ok": True})

    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.model_validate(raw)))
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        headers={
            "host": "127.0.0.1:8787",
            "authorization": "Bearer client-secret-sentinel",
            "x-headroom-base-url": "https://evil.example",
            "x-headroom-api-key": "evil-key",
        },
        json={"model": "REPLACE_WITH_ENABLED_INTERNAL_MODEL", "messages": []},
    )
    assert response.status_code == 200
    assert str(reached[0].url) == "https://llm.internal.example/tenant/v1/chat/completions"
    assert reached[0].headers["authorization"] == "Bearer private-secret"
    assert "evil" not in str(reached[0].headers)


@pytest.mark.parametrize("kind", ["stateful", "websocket"])
def test_secondary_forwarding_paths_also_deny_dns_before_upstream(monkeypatch, kind):
    from headroom.proxy.gateway.resources import ResourceBinding

    reached = []

    async def upstream(request):
        reached.append(request)
        return httpx.Response(200, json={"id": "resp-a"})

    app = app_with_transport(monkeypatch, upstream)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))],
    )
    client = TestClient(app)
    headers = {"host": "127.0.0.1:8787", "authorization": "Bearer client-secret-sentinel"}
    if kind == "stateful":
        asyncio.run(
            app.state.gateway_runtime.resources.bind(
                ResourceBinding(
                    "resp-a",
                    "local-app",
                    "openai-native",
                    "openai-api",
                    "openai-responses",
                    None,
                    app.state.gateway_runtime.capture().account_key("openai-api"),
                    app.state.gateway_runtime.capture().target_key("openai-native"),
                    1,
                )
            )
        )
        response = client.get("/v1/responses/resp-a", headers=headers)
        assert response.status_code == 502
    else:
        monkeypatch.setattr(websockets, "connect", lambda *a, **kw: reached.append(kw))
        with client.websocket_connect("/v1/responses", headers=headers) as ws:
            ws.send_json({"type": "response.create", "model": "REPLACE_WITH_ENABLED_OPENAI_MODEL"})
            assert ws.receive_json()["error"]["code"] == "gateway_egress_denied"
    assert reached == []


@pytest.mark.asyncio
async def test_http_transport_pins_ip_and_retains_tls_hostname(monkeypatch):
    from headroom.proxy.gateway.egress import EgressPolicy
    from headroom.proxy.gateway.transport import http_client

    received = []

    async def connect(self, request):
        received.append(
            (str(request.url), request.headers["host"], request.extensions["sni_hostname"])
        )
        return httpx.Response(200, content=b"ok")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", connect)
    snapshot = GatewayConfigSnapshot.model_validate(example())
    target = "https://api.openai.com/v1/responses"
    destination = EgressPolicy(resolver=lambda *_: ("8.8.8.8",)).authorize(lease(), target)
    async with http_client(snapshot) as client:
        response = await client.get(target, extensions={"gateway_destination": destination})
    assert response.status_code == 200
    assert received == [("https://8.8.8.8/v1/responses", "api.openai.com", "api.openai.com")]


def test_websocket_transport_pins_tls_and_disables_redirects(monkeypatch):
    from headroom.proxy.gateway.transport import tls_context, websocket_connection

    class Connector:
        pass

    received = []

    def connect(uri, **kwargs):
        received.append((uri, kwargs))
        return Connector()

    monkeypatch.setattr(websockets, "connect", connect)
    snapshot = GatewayConfigSnapshot.model_validate(example())
    destination = EgressPolicy(resolver=lambda *_: ("8.8.8.8",)).authorize(
        lease(), "https://api.openai.com/v1/responses"
    )
    context = tls_context(snapshot)
    connector = websocket_connection(destination, {"authorization": "Bearer secret"}, context)
    _, kwargs = received[0]
    assert kwargs["host"] == "8.8.8.8"
    assert kwargs["server_hostname"] == "api.openai.com"
    assert kwargs["ssl"].verify_mode == ssl.CERT_REQUIRED
    assert kwargs["ssl"].check_hostname
    assert kwargs["proxy"] is None
    redirect = RuntimeError("redirect")
    assert connector.process_redirect(redirect) is redirect


@pytest.mark.asyncio
async def test_stream_transport_exception_is_sanitized():
    from headroom.proxy.gateway.dispatch import _iter_upstream_bytes

    class Upstream:
        is_stream_consumed = False

        async def aiter_raw(self):
            yield b"first"
            raise httpx.ReadError("provider-secret-sentinel")

    stream = _iter_upstream_bytes(Upstream())
    assert await anext(stream) == b"first"
    try:
        await anext(stream)
    except Exception as error:
        assert "secret-sentinel" not in str(error)
        assert error.__suppress_context__
    else:
        pytest.fail("upstream failure was silently swallowed")


def test_websocket_provider_error_frame_is_sanitized(monkeypatch):
    class Upstream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def send(self, frame):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            return json.dumps({"type": "error", "error": {"message": "provider-secret-sentinel"}})

    monkeypatch.setattr(websockets, "connect", lambda *a, **kw: Upstream())
    app = app_with_transport(monkeypatch, lambda request: httpx.Response(200))
    with TestClient(app).websocket_connect(
        "/v1/responses",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret-sentinel"},
    ) as ws:
        ws.send_json({"type": "response.create", "model": "REPLACE_WITH_ENABLED_OPENAI_MODEL"})
        frame = ws.receive_text()
        assert "secret-sentinel" not in frame
        assert json.loads(frame)["error"]["code"] == "gateway_upstream_error"


@pytest.mark.asyncio
async def test_gateway_transport_debug_logs_cannot_expose_headers(monkeypatch, caplog):
    from headroom.proxy.gateway.transport import http_client

    async def connect(self, request):
        logging.getLogger("httpcore.http11").debug("received headers: provider-secret-sentinel")
        return httpx.Response(200, content=b"ok")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", connect)
    caplog.set_level(logging.DEBUG, logger="httpcore.http11")
    destination = EgressPolicy(resolver=lambda *_: ("8.8.8.8",)).authorize(
        lease(), "https://api.openai.com/v1/responses"
    )
    async with http_client(GatewayConfigSnapshot.model_validate(example())) as client:
        await client.get(destination.url, extensions={"gateway_destination": destination})
    assert "secret-sentinel" not in caplog.text
    # The same logger must continue to work for the legacy profile.
    logging.getLogger("httpcore.http11").debug("legacy-transport-debug")
    assert "legacy-transport-debug" in caplog.text


@pytest.mark.asyncio
async def test_native_sse_provider_error_is_not_forwarded_even_when_fragmented():
    from headroom.proxy.gateway.dispatch import _iter_upstream_bytes

    class Upstream:
        is_stream_consumed = False
        headers = {"content-type": "text/event-stream"}

        async def aiter_raw(self):
            yield b'data: {"type":"error","error":{"message":"provider-secret-'
            yield b'sentinel"}}\n\n'

    output = []
    with pytest.raises(Exception, match="Upstream request failed"):
        async for chunk in _iter_upstream_bytes(Upstream()):
            output.append(chunk)
    assert output == []


@pytest.mark.parametrize(
    ("content_type", "event_field"),
    [
        ("Text/Event-Stream", b"event: error"),
        ("text/event-stream ", b"event: error"),
        ("text/event-stream", b"event:error"),
    ],
)
def test_real_dispatch_redacts_sse_error_for_valid_wire_variants(
    monkeypatch, content_type, event_field
):
    reached = []

    async def upstream(request):
        reached.append(str(request.url))
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            content=event_field + b'\ndata: {"message":"provider-secret-sentinel"}\n\n',
        )

    app = app_with_transport(monkeypatch, upstream)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/v1/chat/completions",
        headers={
            "host": "127.0.0.1:8787",
            "authorization": "Bearer client-secret-sentinel",
        },
        json={"model": "REPLACE_WITH_ENABLED_OPENAI_MODEL", "messages": [], "stream": True},
    )
    assert reached == ["https://api.openai.com/v1/chat/completions"]
    # HTTP headers are committed before iteration; the failed stream closes without its error body.
    assert response.status_code == 200
    assert "provider-secret-sentinel" not in response.text
    assert b'"code":"gateway_upstream_error"' in response.content


@pytest.mark.parametrize("failure", ["body-read", "credential-source"])
def test_dispatch_sanitizes_late_body_and_source_exceptions(monkeypatch, caplog, failure):
    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise httpx.ReadError("provider-secret-sentinel")
            yield b""

    async def upstream(request):
        return httpx.Response(200, stream=BrokenBody())

    app = app_with_transport(monkeypatch, upstream)
    if failure == "credential-source":

        async def acquire(*args, **kwargs):
            raise RuntimeError("provider-secret-sentinel")

        app.state.gateway_runtime.broker.acquire = acquire
    response = TestClient(app, raise_server_exceptions=False).post(
        "/v1/responses",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret-sentinel"},
        json={"model": "REPLACE_WITH_ENABLED_OPENAI_MODEL", "input": "hello"},
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "gateway_upstream_error"
    assert "secret-sentinel" not in response.text + caplog.text


def test_gateway_startup_does_not_configure_inherited_external_telemetry(monkeypatch):
    import headroom.proxy.server as server

    configured = []
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    monkeypatch.setattr(
        server, "configure_otel_metrics", lambda config: configured.append("metrics")
    )
    monkeypatch.setattr(
        server, "configure_langfuse_tracing", lambda config: configured.append("tracing")
    )
    app = app_with_transport(monkeypatch, lambda request: httpx.Response(200))

    async def no_startup():
        pass

    monkeypatch.setattr(app.state.proxy, "startup", no_startup)
    with TestClient(app) as client:
        assert client.get("/livez").status_code == 200
    assert configured == []


@pytest.mark.asyncio
async def test_stream_logging_guard_does_not_leak_into_legacy_caller(caplog):
    from headroom.proxy.gateway.transport import _PrivateStream

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            logging.getLogger("httpcore.http11").debug("provider-secret-sentinel")
            yield b"first"
            yield b"second"

    caplog.set_level(logging.DEBUG, logger="httpcore.http11")
    stream = _PrivateStream(Stream()).__aiter__()
    try:
        assert await anext(stream) == b"first"
        logging.getLogger("httpcore.http11").debug("legacy-during-stream")
        assert "provider-secret-sentinel" not in caplog.text
        assert "legacy-during-stream" in caplog.text
    finally:
        await stream.aclose()
