"""Tests for headroom.proxy.agy_dispatch.AgyDispatchServer.

All tests use ephemeral loopback ports; ~/.headroom is never touched.
The upstream Gemini/CloudCode network is mocked via monkeypatching
HeadroomProxy._stream_response so no real network calls are made.

Test coverage:
  (a) TLS client (verifying against root CA, SNI=daily-cloudcode-pa.googleapis.com)
      connects to hypercorn port, POSTs /v1internal:streamGenerateContent, gets 200.
  (b) ALPN negotiates h2.
  (c) End-to-end: agy-side CONNECT terminator → tunnel → hypercorn → app → 200.
  (d) Authorization + x-goog-api-key NOT present in headroom logs (caplog).
  (e) All pre-existing T8 terminator tests still pass (those remain in
      test_agy_terminator.py; this file covers the dispatch-server side only).
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import ssl
import tempfile
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.x509 import Certificate
from cryptography.x509.oid import NameOID

from headroom.proxy.agy_dispatch import AgyDispatchServer
from headroom.proxy.agy_terminator import DEFAULT_ALLOWLIST, AgyCONNECTTerminator

# ---------------------------------------------------------------------------
# CA fixture
# ---------------------------------------------------------------------------

ALLOWLIST_HOST = "daily-cloudcode-pa.googleapis.com"

_AGY_REQUEST_BODY = json.dumps(
    {
        "model": "gemini-2.5-pro",
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
        },
    }
).encode()


def _make_test_ca() -> tuple[RSAPrivateKey, Certificate, bytes]:
    """Generate 2048-bit RSA root CA (never touches disk)."""
    key: RSAPrivateKey = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Headroom Test CA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    return key, cert, cert_pem


@pytest.fixture
def tmp_ca() -> tuple[RSAPrivateKey, Certificate, bytes]:
    return _make_test_ca()


# ---------------------------------------------------------------------------
# SSL context helpers
# ---------------------------------------------------------------------------


def _build_client_ssl_ctx(ca_cert_pem: bytes) -> ssl.SSLContext:
    """Build a TLS client context that trusts only the test CA."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    with tempfile.NamedTemporaryFile(suffix=".pem", delete=True, mode="wb") as f:
        f.write(ca_cert_pem)
        f.flush()
        ctx.load_verify_locations(f.name)
    return ctx


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_sse_mock_response() -> bytes:
    """Minimal SSE response payload that handle_google_cloudcode_stream can relay."""
    lines = [
        b'data: {"candidates":[{"content":{"parts":[{"text":"pong"}]}}]}\r\n',
        b"\r\n",
        b"data: [DONE]\r\n",
        b"\r\n",
    ]
    return b"".join(lines)


# ---------------------------------------------------------------------------
# Tests: (a) + (b) direct TLS → dispatch server → 200 + h2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_server_tls_and_route(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) TLS client verifying against root CA connects to hypercorn port,
    POSTs /v1internal:streamGenerateContent, gets 200.
    No real upstream network: _stream_response is monkeypatched.
    """
    ca_key, ca_cert, ca_cert_pem = tmp_ca

    # Patch HeadroomProxy._stream_response so no upstream call is made.
    from fastapi.responses import StreamingResponse

    from headroom.proxy.server import HeadroomProxy

    async def _fake_stream(self: Any, *args: Any, **kwargs: Any) -> StreamingResponse:
        async def _body() -> bytes:
            yield b'data: {"candidates":[]}\n\ndata: [DONE]\n\n'

        return StreamingResponse(
            _body(),
            status_code=200,
            media_type="text/event-stream",
        )

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    async with AgyDispatchServer(ca_key=ca_key, ca_cert=ca_cert) as srv:
        host, port = srv.address
        assert host == "127.0.0.1"
        assert port > 0

        # Build an HTTPS client that trusts the test CA.
        ssl_ctx = _build_client_ssl_ctx(ca_cert_pem)
        # Use HTTP/1.1 for the direct request (simpler to compose manually).
        ssl_ctx.set_alpn_protocols(["http/1.1"])

        conn_reader, conn_writer = await asyncio.open_connection(
            "127.0.0.1",
            port,
            ssl=ssl_ctx,
            server_hostname=ALLOWLIST_HOST,
        )
        try:
            body = _AGY_REQUEST_BODY
            request = (
                f"POST /v1internal:streamGenerateContent HTTP/1.1\r\n"
                f"Host: {ALLOWLIST_HOST}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"\r\n"
            ).encode() + body

            conn_writer.write(request)
            await conn_writer.drain()

            # Read enough response to confirm 200.
            response_line = await asyncio.wait_for(conn_reader.readline(), timeout=10.0)
            assert b"200" in response_line, f"Expected 200, got {response_line!r}"
        finally:
            conn_writer.close()
            try:
                await conn_writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass


@pytest.mark.asyncio
async def test_dispatch_server_alpn_h2(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) ALPN negotiates h2 when client offers ["h2", "http/1.1"]."""
    ca_key, ca_cert, ca_cert_pem = tmp_ca

    from fastapi.responses import StreamingResponse

    from headroom.proxy.server import HeadroomProxy

    async def _fake_stream(self: Any, *args: Any, **kwargs: Any) -> StreamingResponse:
        async def _body() -> bytes:
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_body(), status_code=200, media_type="text/event-stream")

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    async with AgyDispatchServer(ca_key=ca_key, ca_cert=ca_cert) as srv:
        _, port = srv.address

        ssl_ctx = _build_client_ssl_ctx(ca_cert_pem)
        ssl_ctx.set_alpn_protocols(["h2", "http/1.1"])

        conn_reader, conn_writer = await asyncio.open_connection(
            "127.0.0.1",
            port,
            ssl=ssl_ctx,
            server_hostname=ALLOWLIST_HOST,
        )
        try:
            ssl_obj = conn_writer.get_extra_info("ssl_object")
            alpn = ssl_obj.selected_alpn_protocol() if ssl_obj else None
            assert alpn == "h2", f"Expected h2 ALPN, got {alpn!r}"
        finally:
            conn_writer.close()
            try:
                await conn_writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Test: (c) end-to-end via terminator → tunnel → hypercorn → 200
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminator_tunnel_to_dispatch_server(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) agy-side CONNECT terminator → byte-splice tunnel → hypercorn → app → 200."""
    ca_key, ca_cert, ca_cert_pem = tmp_ca

    from fastapi.responses import StreamingResponse

    from headroom.proxy.server import HeadroomProxy

    async def _fake_stream(self: Any, *args: Any, **kwargs: Any) -> StreamingResponse:
        async def _body() -> bytes:
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_body(), status_code=200, media_type="text/event-stream")

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    async with AgyDispatchServer(ca_key=ca_key, ca_cert=ca_cert) as dispatch_srv:
        _, dispatch_port = dispatch_srv.address

        async with AgyCONNECTTerminator(
            allowlist=DEFAULT_ALLOWLIST,
            ca_key=ca_key,
            ca_cert=ca_cert,
            dispatch_port=dispatch_port,
        ) as terminator:
            proxy_host, proxy_port = terminator.address

            # Step 1: TCP CONNECT to terminator.
            raw_reader, raw_writer = await asyncio.open_connection(proxy_host, proxy_port)
            raw_writer.write(
                f"CONNECT {ALLOWLIST_HOST}:443 HTTP/1.1\r\n"
                f"Host: {ALLOWLIST_HOST}:443\r\n"
                "\r\n".encode()
            )
            await raw_writer.drain()
            resp = await asyncio.wait_for(raw_reader.readline(), timeout=5.0)
            assert b"200" in resp, f"Expected 200 tunnel ACK, got {resp!r}"

            # Step 2: TLS handshake over the tunnel (to hypercorn's SNI cert).
            ssl_ctx = _build_client_ssl_ctx(ca_cert_pem)
            ssl_ctx.set_alpn_protocols(["http/1.1"])
            loop = asyncio.get_event_loop()
            raw_writer.transport.pause_reading()
            tls_transport = await asyncio.wait_for(
                loop.start_tls(
                    raw_writer.transport,
                    raw_writer.transport.get_protocol(),
                    ssl_ctx,
                    server_hostname=ALLOWLIST_HOST,
                ),
                timeout=10.0,
            )

            # Step 3: POST through TLS tunnel and check 200.
            # Re-wrap tls_transport in a StreamReader so we can readline().
            tls_reader = asyncio.StreamReader()
            tls_proto = asyncio.StreamReaderProtocol(tls_reader)
            tls_transport.set_protocol(tls_proto)
            tls_proto.connection_made(tls_transport)

            body = _AGY_REQUEST_BODY
            tls_transport.write(
                (
                    f"POST /v1internal:streamGenerateContent HTTP/1.1\r\n"
                    f"Host: {ALLOWLIST_HOST}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    f"\r\n"
                ).encode()
                + body
            )

            # Read the HTTP status line through the terminator tunnel.
            status_line = await asyncio.wait_for(tls_reader.readline(), timeout=10.0)
            assert b"200" in status_line, (
                f"Expected HTTP 200 through CONNECT tunnel, got: {status_line!r}"
            )

            tls_transport.close()


# ---------------------------------------------------------------------------
# Test: (d) Authorization + x-goog-api-key NOT in headroom logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_secret_headers_not_logged(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """(d) Authorization and x-goog-api-key must not appear in headroom logs."""
    ca_key, ca_cert, ca_cert_pem = tmp_ca

    from fastapi.responses import StreamingResponse

    from headroom.proxy.server import HeadroomProxy

    async def _fake_stream(self: Any, *args: Any, **kwargs: Any) -> StreamingResponse:
        async def _body() -> bytes:
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_body(), status_code=200, media_type="text/event-stream")

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    with caplog.at_level(logging.DEBUG, logger="headroom"):
        async with AgyDispatchServer(ca_key=ca_key, ca_cert=ca_cert) as srv:
            _, port = srv.address

            ssl_ctx = _build_client_ssl_ctx(ca_cert_pem)
            ssl_ctx.set_alpn_protocols(["http/1.1"])

            conn_reader, conn_writer = await asyncio.open_connection(
                "127.0.0.1",
                port,
                ssl=ssl_ctx,
                server_hostname=ALLOWLIST_HOST,
            )
            try:
                body = _AGY_REQUEST_BODY
                secret_auth = "Bearer supersecret-token-xyz"
                secret_api_key = "AIzaSySecret1234"
                request = (
                    f"POST /v1internal:streamGenerateContent HTTP/1.1\r\n"
                    f"Host: {ALLOWLIST_HOST}\r\n"
                    f"Authorization: {secret_auth}\r\n"
                    f"x-goog-api-key: {secret_api_key}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    f"\r\n"
                ).encode() + body

                conn_writer.write(request)
                await conn_writer.drain()
                # Read enough to let the handler log.
                await asyncio.wait_for(conn_reader.readline(), timeout=10.0)
            finally:
                conn_writer.close()
                try:
                    await conn_writer.wait_closed()
                except Exception:  # noqa: BLE001
                    pass

    log_text = "\n".join(r.getMessage() for r in caplog.records)
    # Default path (log_outbound_headers) only logs counts, never values.
    # Assert no header VALUE leaks into any log record on the default path.
    assert "supersecret-token-xyz" not in log_text, "Bearer token leaked into headroom logs"
    assert "AIzaSySecret1234" not in log_text, "x-goog-api-key leaked into headroom logs"


# ---------------------------------------------------------------------------
# Test: (d2) redaction unit — _should_redact_key / redact_for_wire_debug
# ---------------------------------------------------------------------------


def test_redaction_is_load_bearing() -> None:
    """Redaction of authorization and x-goog-api-key is structurally enforced.

    This test is deliberately coupled to _should_redact_key and
    redact_for_wire_debug so that removing or weakening either function
    causes a failure here, making this a load-bearing regression guard.
    """
    from headroom.proxy.helpers import (
        _CODEX_WIRE_REDACTED,
        _should_redact_key,
        redact_for_wire_debug,
    )

    # 1. _should_redact_key must flag both sensitive header names.
    assert _should_redact_key("authorization"), "authorization must be redacted"
    assert _should_redact_key("Authorization"), "Authorization (mixed case) must be redacted"
    assert _should_redact_key("x-goog-api-key"), "x-goog-api-key must be redacted"
    assert _should_redact_key("X-Goog-Api-Key"), "X-Goog-Api-Key (mixed case) must be redacted"

    # 2. redact_for_wire_debug must replace values with _CODEX_WIRE_REDACTED.
    secret_auth = "Bearer supersecret-token-xyz"
    secret_api_key = "AIzaSySecret1234"
    headers = {
        "authorization": secret_auth,
        "x-goog-api-key": secret_api_key,
        "content-type": "application/json",
    }
    redacted = redact_for_wire_debug(headers)
    assert redacted["authorization"] == _CODEX_WIRE_REDACTED, (
        f"authorization must be {_CODEX_WIRE_REDACTED!r}, got {redacted['authorization']!r}"
    )
    assert redacted["x-goog-api-key"] == _CODEX_WIRE_REDACTED, (
        f"x-goog-api-key must be {_CODEX_WIRE_REDACTED!r}, got {redacted['x-goog-api-key']!r}"
    )
    # Non-secret headers must pass through unchanged.
    assert redacted["content-type"] == "application/json"

    # 3. Secret VALUES must not appear in the redacted output at all.
    import json as _json

    redacted_str = _json.dumps(redacted)
    assert secret_auth not in redacted_str, "Bearer token survived redact_for_wire_debug"
    assert secret_api_key not in redacted_str, "API key survived redact_for_wire_debug"


# ---------------------------------------------------------------------------
# Tests: dispatch server lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_server_loopback_only(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
) -> None:
    """AgyDispatchServer binds 127.0.0.1 only."""
    ca_key, ca_cert, _ = tmp_ca
    async with AgyDispatchServer(ca_key=ca_key, ca_cert=ca_cert) as srv:
        host, port = srv.address
        assert host == "127.0.0.1"
        assert port > 0


@pytest.mark.asyncio
async def test_dispatch_server_start_stop_idempotent(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
) -> None:
    """stop() after stop() does not raise."""
    ca_key, ca_cert, _ = tmp_ca
    srv = AgyDispatchServer(ca_key=ca_key, ca_cert=ca_cert)
    await srv.start()
    await srv.stop()
    await srv.stop()  # idempotent


def test_dispatch_server_address_raises_before_start(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
) -> None:
    """address property raises RuntimeError before start()."""
    ca_key, ca_cert, _ = tmp_ca
    srv = AgyDispatchServer(ca_key=ca_key, ca_cert=ca_cert)
    with pytest.raises(RuntimeError, match="not started"):
        _ = srv.address
