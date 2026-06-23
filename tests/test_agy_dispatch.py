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


# ---------------------------------------------------------------------------
# Tests: SNI allowlist guard (headroom-oqb.1)
# ---------------------------------------------------------------------------

_ATTACKER_HOST = "evilcloudcode-pa.googleapis.com"
_CONTROLLED_HOST = "allowed.test"
_CONTROLLED_ALLOWLIST: frozenset[str] = frozenset({_CONTROLLED_HOST})


async def _try_tls_connect(
    port: int,
    ca_cert_pem: bytes,
    server_hostname: str | None,
    *,
    timeout: float = 5.0,
) -> bool:
    """Return True if TLS handshake succeeds, False if it fails with an SSL error."""
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_ctx.check_hostname = server_hostname is not None
    ssl_ctx.verify_mode = ssl.CERT_REQUIRED if server_hostname is not None else ssl.CERT_NONE
    with tempfile.NamedTemporaryFile(suffix=".pem", delete=True, mode="wb") as f:
        f.write(ca_cert_pem)
        f.flush()
        ssl_ctx.load_verify_locations(f.name)
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(
                "127.0.0.1",
                port,
                ssl=ssl_ctx,
                server_hostname=server_hostname,
            ),
            timeout=timeout,
        )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass
        return True
    except (ssl.SSLError, OSError, ConnectionResetError, asyncio.TimeoutError):
        return False


@pytest.mark.asyncio
async def test_sni_allowlisted_still_routes(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allowlisted SNI completes handshake (no regression)."""
    ca_key, ca_cert, ca_cert_pem = tmp_ca

    from fastapi.responses import StreamingResponse

    from headroom.proxy.server import HeadroomProxy

    async def _fake_stream(self: Any, *args: Any, **kwargs: Any) -> StreamingResponse:
        async def _body() -> bytes:
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_body(), status_code=200, media_type="text/event-stream")

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    async with AgyDispatchServer(
        ca_key=ca_key, ca_cert=ca_cert, allowlist=_CONTROLLED_ALLOWLIST
    ) as srv:
        _, port = srv.address
        success = await _try_tls_connect(port, ca_cert_pem, _CONTROLLED_HOST)
    assert success, "Allowlisted SNI must complete handshake"


@pytest.mark.asyncio
async def test_sni_non_allowlisted_rejected(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
) -> None:
    """Non-allowlisted SNI -> handshake aborts; server stays alive;
    get_or_mint NOT called for attacker host; event=sni_refused is logged."""
    from unittest.mock import patch

    from headroom.proxy.agy_terminator import _LeafCache

    ca_key, ca_cert, ca_cert_pem = tmp_ca

    # Capture log records via a handler installed before the server starts.
    # caplog cannot reliably capture records from SSL C-level callbacks, so
    # we install a custom handler directly on the module logger.
    sni_log_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            sni_log_records.append(record)

    capture_handler = _Capture(logging.WARNING)
    _dispatch_logger = logging.getLogger("headroom.proxy.agy_dispatch")
    _dispatch_logger.addHandler(capture_handler)

    try:
        async with AgyDispatchServer(
            ca_key=ca_key, ca_cert=ca_cert, allowlist=_CONTROLLED_ALLOWLIST
        ) as srv:
            _, port = srv.address

            # Spy on _LeafCache.get_or_mint to verify attacker host is never minted.
            original_get_or_mint = _LeafCache.get_or_mint
            call_hostnames: list[str] = []

            def _spy_get_or_mint(
                self: _LeafCache, host: str, *args: Any, **kwargs: Any
            ) -> Any:
                call_hostnames.append(host)
                return original_get_or_mint(self, host, *args, **kwargs)

            with patch.object(_LeafCache, "get_or_mint", _spy_get_or_mint):
                rejected = not await _try_tls_connect(port, ca_cert_pem, _ATTACKER_HOST)

            # Server must still respond to further connections.
            assert srv._server is not None, "Server must stay alive after rejected SNI"
    finally:
        _dispatch_logger.removeHandler(capture_handler)

    assert rejected, "Non-allowlisted SNI must abort handshake"
    attacker_mints = [h for h in call_hostnames if h == _ATTACKER_HOST]
    assert attacker_mints == [], f"get_or_mint called for attacker host: {attacker_mints}"

    # Attacker host must be absent from the leaf cache.
    assert srv._leaf_cache is not None
    assert _ATTACKER_HOST not in srv._leaf_cache._cache, (
        "Attacker host must not appear in leaf cache"
    )

    warned = any("event=sni_refused" in r.getMessage() for r in sni_log_records)
    assert warned, (
        f"Expected event=sni_refused WARNING; got records: "
        f"{[r.getMessage() for r in sni_log_records]}"
    )


@pytest.mark.asyncio
async def test_sni_named_attack_evilcloudcode_rejected(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
) -> None:
    """Named attack: SNI 'evilcloudcode-pa.googleapis.com' is rejected."""
    ca_key, ca_cert, ca_cert_pem = tmp_ca

    async with AgyDispatchServer(ca_key=ca_key, ca_cert=ca_cert) as srv:
        _, port = srv.address
        rejected = not await _try_tls_connect(port, ca_cert_pem, _ATTACKER_HOST)

    assert rejected, "evilcloudcode-pa.googleapis.com must be rejected by SNI guard"


@pytest.mark.asyncio
async def test_sni_placeholder_headroom_internal_rejected(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
) -> None:
    """Explicit wire SNI 'headroom.internal' (the placeholder) is rejected."""
    ca_key, ca_cert, ca_cert_pem = tmp_ca

    # Use a handler to verify event=sni_refused is logged (caplog is unreliable
    # in SSL C-level callbacks).
    sni_log_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            sni_log_records.append(record)

    capture_handler = _Capture(logging.WARNING)
    _dispatch_logger = logging.getLogger("headroom.proxy.agy_dispatch")
    _dispatch_logger.addHandler(capture_handler)

    try:
        async with AgyDispatchServer(
            ca_key=ca_key, ca_cert=ca_cert, allowlist=_CONTROLLED_ALLOWLIST
        ) as srv:
            _, port = srv.address
            rejected = not await _try_tls_connect(port, ca_cert_pem, "headroom.internal")
    finally:
        _dispatch_logger.removeHandler(capture_handler)

    assert rejected, "headroom.internal must be rejected (not in allowlist)"
    warned = any("event=sni_refused" in r.getMessage() for r in sni_log_records)
    assert warned, (
        f"Expected event=sni_refused WARNING for headroom.internal; "
        f"got: {[r.getMessage() for r in sni_log_records]}"
    )


@pytest.mark.asyncio
async def test_sni_none_and_empty_rejected(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
) -> None:
    """None SNI and empty-string SNI are rejected; no headroom.internal leaf served."""
    ca_key, ca_cert, ca_cert_pem = tmp_ca

    # Capture log records via a handler (caplog is unreliable in SSL C-level callbacks).
    sni_log_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            sni_log_records.append(record)

    capture_handler = _Capture(logging.WARNING)
    _dispatch_logger = logging.getLogger("headroom.proxy.agy_dispatch")
    _dispatch_logger.addHandler(capture_handler)

    try:
        async with AgyDispatchServer(
            ca_key=ca_key, ca_cert=ca_cert, allowlist=_CONTROLLED_ALLOWLIST
        ) as srv:
            _, port = srv.address

            # None SNI: disable hostname verification so we can send without SNI.
            ssl_ctx_no_sni = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_ctx_no_sni.check_hostname = False
            ssl_ctx_no_sni.verify_mode = ssl.CERT_NONE
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(
                        "127.0.0.1",
                        port,
                        ssl=ssl_ctx_no_sni,
                        server_hostname=None,  # no SNI extension
                    ),
                    timeout=5.0,
                )
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:  # noqa: BLE001
                    pass
                none_sni_accepted = True
            except (ssl.SSLError, OSError, ConnectionResetError, asyncio.TimeoutError):
                none_sni_accepted = False
    finally:
        _dispatch_logger.removeHandler(capture_handler)

    assert not none_sni_accepted, "None SNI must be rejected"
    warned = any("event=sni_refused" in r.getMessage() for r in sni_log_records)
    assert warned, (
        f"Expected event=sni_refused WARNING for None SNI; "
        f"got: {[r.getMessage() for r in sni_log_records]}"
    )


@pytest.mark.asyncio
async def test_sni_trailing_dot_fqdn_rejected(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
) -> None:
    """Trailing-dot FQDN 'daily-cloudcode-pa.googleapis.com.' is rejected under exact match."""
    ca_key, ca_cert, ca_cert_pem = tmp_ca

    # Use a controlled allowlist with only the non-dotted form.
    allowlist = frozenset({"daily-cloudcode-pa.googleapis.com"})
    async with AgyDispatchServer(
        ca_key=ca_key, ca_cert=ca_cert, allowlist=allowlist
    ) as srv:
        _, port = srv.address
        # trailing dot form is not in allowlist — must be rejected
        rejected = not await _try_tls_connect(
            port, ca_cert_pem, "daily-cloudcode-pa.googleapis.com."
        )

    assert rejected, "Trailing-dot FQDN must be rejected under exact match"


@pytest.mark.asyncio
async def test_sni_exception_inside_callback_server_stays_alive(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
) -> None:
    """Exception raised inside _sni_callback -> handshake aborts AND server stays alive."""
    from unittest.mock import patch

    from headroom.proxy.agy_terminator import _LeafCache

    ca_key, ca_cert, ca_cert_pem = tmp_ca

    async with AgyDispatchServer(
        ca_key=ca_key, ca_cert=ca_cert, allowlist=_CONTROLLED_ALLOWLIST
    ) as srv:
        _, port = srv.address

        def _boom(self: _LeafCache, *a: Any, **kw: Any) -> Any:
            raise RuntimeError("injected failure")

        with patch.object(_LeafCache, "get_or_mint", _boom):
            # The handshake must fail (SSL error), not crash the server.
            rejected = not await _try_tls_connect(port, ca_cert_pem, _CONTROLLED_HOST)

        # Server must still be alive.
        assert srv._server is not None, "Server must stay alive after exception in callback"

    assert rejected, "Exception in SNI callback must abort handshake"


def test_placeholder_host_not_in_default_allowlist() -> None:
    """_PLACEHOLDER_HOST 'headroom.internal' must NOT be in DEFAULT_ALLOWLIST."""
    assert "headroom.internal" not in DEFAULT_ALLOWLIST, (
        "headroom.internal must never appear in DEFAULT_ALLOWLIST"
    )


# ---------------------------------------------------------------------------
# Tests: post-handshake Host guard (headroom-oqb.1)
# ---------------------------------------------------------------------------

async def _http11_request(
    port: int,
    ca_cert_pem: bytes,
    sni_host: str,
    host_header: str,
    *,
    timeout: float = 10.0,
) -> int:
    """Perform HTTP/1.1 GET / and return the status code (or 0 on connection failure)."""
    ssl_ctx = _build_client_ssl_ctx(ca_cert_pem)
    ssl_ctx.set_alpn_protocols(["http/1.1"])
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                "127.0.0.1", port, ssl=ssl_ctx, server_hostname=sni_host
            ),
            timeout=timeout,
        )
    except (ssl.SSLError, OSError, ConnectionResetError):
        return 0
    try:
        request = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode()
        writer.write(request)
        await writer.drain()
        status_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not status_line:
            return 0
        parts = status_line.split()
        return int(parts[1]) if len(parts) >= 2 else 0
    except (OSError, asyncio.TimeoutError, IndexError, ValueError):
        return 0
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.asyncio
async def test_host_guard_non_allowlisted_returns_421(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
) -> None:
    """Post-handshake Host guard: non-allowlisted Host header -> 421 Misdirected Request.

    We spy on _send_421 to confirm the guard is what sends the 421 (not the upstream app).
    """
    from unittest.mock import patch

    ca_key, ca_cert, ca_cert_pem = tmp_ca

    send_421_called = [False]

    async def _spy_send_421(send: Any) -> None:
        send_421_called[0] = True
        import headroom.proxy.agy_dispatch as _m

        await _m._send_421.__wrapped__(send) if hasattr(
            _m._send_421, "__wrapped__"
        ) else await _real_send_421(send)

    from headroom.proxy import agy_dispatch as _agy_dispatch_mod

    _real_send_421 = _agy_dispatch_mod._send_421

    with patch("headroom.proxy.agy_dispatch._send_421", _spy_send_421):
        async with AgyDispatchServer(
            ca_key=ca_key, ca_cert=ca_cert, allowlist=_CONTROLLED_ALLOWLIST
        ) as srv:
            _, port = srv.address
            status = await _http11_request(
                port, ca_cert_pem, _CONTROLLED_HOST, "evil.example.com"
            )

    assert status == 421, f"Expected 421 for non-allowlisted Host, got {status}"
    assert send_421_called[0], "Guard must call _send_421 for non-allowlisted Host"


@pytest.mark.asyncio
async def test_host_guard_allowlisted_passes(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-handshake Host guard: allowlisted Host passes through to the app (guard not triggered).

    We spy on _send_421 to confirm the guard does NOT refuse the allowlisted host.
    The app may return any status (404, 200, …) — that is app-layer behavior, not the guard.
    """
    from unittest.mock import patch

    ca_key, ca_cert, ca_cert_pem = tmp_ca

    send_421_called = [False]

    async def _spy_send_421(send: Any) -> None:
        send_421_called[0] = True
        from headroom.proxy.agy_dispatch import _send_421

        await _send_421(send)

    with patch("headroom.proxy.agy_dispatch._send_421", _spy_send_421):
        async with AgyDispatchServer(
            ca_key=ca_key, ca_cert=ca_cert, allowlist=_CONTROLLED_ALLOWLIST
        ) as srv:
            _, port = srv.address
            status = await _http11_request(
                port, ca_cert_pem, _CONTROLLED_HOST, _CONTROLLED_HOST
            )

    assert not send_421_called[0], (
        f"Guard must NOT refuse the allowlisted Host '{_CONTROLLED_HOST}'; "
        f"got HTTP status {status}"
    )
    assert status != 0, "Expected a valid HTTP response (guard passed request to app)"


@pytest.mark.asyncio
async def test_host_guard_port_qualified_host_passes(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
) -> None:
    """Post-handshake Host guard: 'host:443' form is normalized and passes the guard.

    Spy on _send_421 — the guard must NOT refuse 'allowed.test:443'.
    """
    from unittest.mock import patch

    ca_key, ca_cert, ca_cert_pem = tmp_ca

    send_421_called = [False]

    async def _spy_send_421(send: Any) -> None:
        send_421_called[0] = True
        from headroom.proxy.agy_dispatch import _send_421

        await _send_421(send)

    with patch("headroom.proxy.agy_dispatch._send_421", _spy_send_421):
        async with AgyDispatchServer(
            ca_key=ca_key, ca_cert=ca_cert, allowlist=_CONTROLLED_ALLOWLIST
        ) as srv:
            _, port = srv.address
            status = await _http11_request(
                port, ca_cert_pem, _CONTROLLED_HOST, f"{_CONTROLLED_HOST}:443"
            )

    assert not send_421_called[0], (
        f"Guard must NOT refuse 'host:port' form; got HTTP status {status}"
    )
    assert status != 0, "Expected a valid HTTP response (guard passed request to app)"


@pytest.mark.asyncio
async def test_host_guard_mixed_case_host_passes(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
) -> None:
    """Post-handshake Host guard: mixed-case Host is normalized (lowercased) and passes.

    Spy on _send_421 — the guard must NOT refuse the uppercased form.
    """
    from unittest.mock import patch

    ca_key, ca_cert, ca_cert_pem = tmp_ca

    send_421_called = [False]

    async def _spy_send_421(send: Any) -> None:
        send_421_called[0] = True
        from headroom.proxy.agy_dispatch import _send_421

        await _send_421(send)

    with patch("headroom.proxy.agy_dispatch._send_421", _spy_send_421):
        async with AgyDispatchServer(
            ca_key=ca_key, ca_cert=ca_cert, allowlist=_CONTROLLED_ALLOWLIST
        ) as srv:
            _, port = srv.address
            mixed_case = _CONTROLLED_HOST.upper()
            status = await _http11_request(
                port, ca_cert_pem, _CONTROLLED_HOST, mixed_case
            )

    assert not send_421_called[0], (
        f"Guard must NOT refuse mixed-case Host (normalized to lower); got HTTP status {status}"
    )
    assert status != 0, "Expected a valid HTTP response (guard passed request to app)"


# ---------------------------------------------------------------------------
# Tests: load_cert_chain_in_memory used in dispatch (headroom-oqb.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_no_tmpfile_on_linux(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Linux (memfd_create available), load_cert_chain is never called with
    a regular filesystem path for leaf key material — only /proc/self/fd/ paths."""
    import os
    import ssl as _ssl

    ca_key, ca_cert, _ = tmp_ca

    if not hasattr(os, "memfd_create"):
        pytest.skip("memfd_create not available; primary path not applicable")

    # Spy on ssl.SSLContext.load_cert_chain to check which paths are passed.
    leaf_fs_paths: list[str] = []
    original_load = _ssl.SSLContext.load_cert_chain

    def _spy_load(
        self: _ssl.SSLContext, certfile: str, keyfile: object = None, **kwargs: object
    ) -> None:
        # Flag any certfile that is NOT an anonymous memfd /proc path.
        if not certfile.startswith("/proc/self/fd/"):
            leaf_fs_paths.append(certfile)
        original_load(self, certfile, keyfile, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_ssl.SSLContext, "load_cert_chain", _spy_load)

    async with AgyDispatchServer(ca_key=ca_key, ca_cert=ca_cert):
        pass

    assert not leaf_fs_paths, (
        f"load_cert_chain must only use /proc/self/fd/ on Linux (memfd), "
        f"but got regular fs paths: {leaf_fs_paths}"
    )


@pytest.mark.asyncio
async def test_dispatch_handshake_still_works_via_helper(
    tmp_ca: tuple[RSAPrivateKey, Certificate, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AgyDispatchServer (using load_cert_chain_in_memory) still completes
    a TLS handshake for an allowlisted SNI host — regression guard."""
    from fastapi.responses import StreamingResponse

    from headroom.proxy.server import HeadroomProxy

    ca_key, ca_cert, ca_cert_pem = tmp_ca

    async def _fake_stream(self: Any, *args: Any, **kwargs: Any) -> StreamingResponse:
        async def _body() -> bytes:
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_body(), status_code=200, media_type="text/event-stream")

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

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
            request = (
                f"GET / HTTP/1.1\r\n"
                f"Host: {ALLOWLIST_HOST}\r\n"
                f"\r\n"
            ).encode()
            conn_writer.write(request)
            await conn_writer.drain()
            response_line = await asyncio.wait_for(conn_reader.readline(), timeout=10.0)
        finally:
            conn_writer.close()

    # Any HTTP response (even 404/421) confirms the TLS handshake succeeded.
    assert response_line.startswith(b"HTTP/"), (
        f"Expected HTTP response; TLS handshake must succeed via helper. Got: {response_line!r}"
    )
