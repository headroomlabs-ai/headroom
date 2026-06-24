"""Tests for headroom.proxy.agy_terminator.

All tests use ephemeral ports and tmp_path; real ~/.headroom is never touched.
Tests use real asyncio connections over loopback to verify behavior.
"""

from __future__ import annotations

import asyncio
import datetime
import ssl
import tempfile

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.x509 import Certificate
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from headroom.proxy.agy_terminator import (
    DEFAULT_ALLOWLIST,
    AgyCONNECTTerminator,
    _is_loopback,
    _LeafCache,
    _parse_connect,
    mint_leaf,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWLIST_HOST = "daily-cloudcode-pa.googleapis.com"
NON_ALLOWLIST_HOST = "example.com"


def _make_test_ca() -> tuple[RSAPrivateKey, Certificate, bytes]:
    """Generate a fast 2048-bit RSA root CA for tests (never touches disk)."""
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
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    ca_cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    return key, cert, ca_cert_pem


def _build_client_ssl_context(ca_cert_pem: bytes) -> ssl.SSLContext:
    """Build a verifying TLS client context that trusts only our test root CA."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    with tempfile.NamedTemporaryFile(suffix=".pem", delete=True, mode="wb") as tf:
        tf.write(ca_cert_pem)
        tf.flush()
        ctx.load_verify_locations(tf.name)
    ctx.set_alpn_protocols(["h2", "http/1.1"])
    return ctx


@pytest.fixture(scope="module")
def tmp_ca() -> tuple[RSAPrivateKey, Certificate, bytes]:
    """Return (ca_key, ca_cert, ca_cert_pem) — module-scoped; generated once."""
    return _make_test_ca()


# ---------------------------------------------------------------------------
# Unit: _parse_connect
# ---------------------------------------------------------------------------


def test_parse_connect_basic() -> None:
    host, port = _parse_connect("CONNECT example.com:443 HTTP/1.1")
    assert host == "example.com"
    assert port == 443


def test_parse_connect_lowercase() -> None:
    host, port = _parse_connect("connect api.example.com:8443 HTTP/1.1")
    assert host == "api.example.com"
    assert port == 8443


def test_parse_connect_invalid_raises() -> None:
    with pytest.raises(ValueError):
        _parse_connect("GET / HTTP/1.1")


def test_parse_connect_missing_port_raises() -> None:
    with pytest.raises(ValueError):
        _parse_connect("CONNECT example.com HTTP/1.1")


# ---------------------------------------------------------------------------
# Unit: _is_loopback
# ---------------------------------------------------------------------------


def test_is_loopback_127() -> None:
    assert _is_loopback("127.0.0.1") is True


def test_is_loopback_localhost() -> None:
    assert _is_loopback("localhost") is True


def test_is_loopback_ipv6() -> None:
    assert _is_loopback("::1") is True


def test_is_loopback_public() -> None:
    assert _is_loopback("8.8.8.8") is False


def test_is_loopback_hostname() -> None:
    assert _is_loopback("example.com") is False


# ---------------------------------------------------------------------------
# Unit: mint_leaf
# ---------------------------------------------------------------------------


def test_mint_leaf_san(tmp_ca: tuple) -> None:
    """Minted leaf must have SAN=dNSName for the host. (f)"""
    ca_key, ca_cert, _ = tmp_ca
    cert_pem, _ = mint_leaf("api.example.com", ca_key, ca_cert)
    cert = x509.load_pem_x509_certificate(cert_pem)
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    dns_names = san.value.get_values_for_type(x509.DNSName)
    assert "api.example.com" in dns_names


def test_mint_leaf_eku_server_auth(tmp_ca: tuple) -> None:
    """Minted leaf must have EKU=serverAuth only. (f)"""
    ca_key, ca_cert, _ = tmp_ca
    cert_pem, _ = mint_leaf("api.example.com", ca_key, ca_cert)
    cert = x509.load_pem_x509_certificate(cert_pem)
    eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
    assert list(eku.value) == [ExtendedKeyUsageOID.SERVER_AUTH]


def test_mint_leaf_validity_lte_72h(tmp_ca: tuple) -> None:
    """Minted leaf validity must be <= 72 hours. (f)"""
    ca_key, ca_cert, _ = tmp_ca
    cert_pem, _ = mint_leaf("api.example.com", ca_key, ca_cert)
    cert = x509.load_pem_x509_certificate(cert_pem)
    delta = cert.not_valid_after_utc - cert.not_valid_before_utc
    assert delta <= datetime.timedelta(hours=72)


def test_mint_leaf_not_ca(tmp_ca: tuple) -> None:
    """Minted leaf must not have CA:TRUE."""
    ca_key, ca_cert, _ = tmp_ca
    cert_pem, _ = mint_leaf("api.example.com", ca_key, ca_cert)
    cert = x509.load_pem_x509_certificate(cert_pem)
    bc = cert.extensions.get_extension_for_class(x509.BasicConstraints)
    assert bc.value.ca is False


def test_mint_leaf_signed_by_root(tmp_ca: tuple) -> None:
    """Leaf issuer must match the root CA subject."""
    ca_key, ca_cert, _ = tmp_ca
    cert_pem, _ = mint_leaf("api.example.com", ca_key, ca_cert)
    cert = x509.load_pem_x509_certificate(cert_pem)
    assert cert.issuer == ca_cert.subject


# ---------------------------------------------------------------------------
# Unit: _LeafCache
# ---------------------------------------------------------------------------


def test_leaf_cache_reuse(tmp_ca: tuple) -> None:
    """Same host returns same cert PEM (serial equality). (b)"""
    ca_key, ca_cert, _ = tmp_ca
    cache = _LeafCache(max_size=10)
    cert1, _ = cache.get_or_mint("api.example.com", ca_key, ca_cert)
    cert2, _ = cache.get_or_mint("api.example.com", ca_key, ca_cert)
    obj1 = x509.load_pem_x509_certificate(cert1)
    obj2 = x509.load_pem_x509_certificate(cert2)
    assert obj1.serial_number == obj2.serial_number


def test_leaf_cache_different_hosts(tmp_ca: tuple) -> None:
    """Different hosts get different leaf certs."""
    ca_key, ca_cert, _ = tmp_ca
    cache = _LeafCache(max_size=10)
    cert1, _ = cache.get_or_mint("host-a.example.com", ca_key, ca_cert)
    cert2, _ = cache.get_or_mint("host-b.example.com", ca_key, ca_cert)
    obj1 = x509.load_pem_x509_certificate(cert1)
    obj2 = x509.load_pem_x509_certificate(cert2)
    assert obj1.serial_number != obj2.serial_number


def test_leaf_cache_bound_evicts(tmp_ca: tuple) -> None:
    """Cache with max_size=1 evicts oldest on second host."""
    ca_key, ca_cert, _ = tmp_ca
    cache = _LeafCache(max_size=1)
    cache.get_or_mint("host-a.example.com", ca_key, ca_cert)
    cache.get_or_mint("host-b.example.com", ca_key, ca_cert)
    assert len(cache._cache) == 1
    assert "host-b.example.com" in cache._cache


# ---------------------------------------------------------------------------
# Integration: listener bind address
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_listener_bound_to_loopback_only(tmp_ca: tuple) -> None:
    """Listener must be bound to 127.0.0.1, not 0.0.0.0. (d)"""
    ca_key, ca_cert, _ = tmp_ca
    terminator = AgyCONNECTTerminator(
        allowlist=DEFAULT_ALLOWLIST,
        ca_key=ca_key,
        ca_cert=ca_cert,
    )
    await terminator.start()
    try:
        bound_host, bound_port = terminator.address
        assert bound_host == "127.0.0.1", f"Expected 127.0.0.1 but got {bound_host}"
        assert bound_port > 0

        # Connecting via 127.0.0.1 succeeds.
        reader, writer = await asyncio.open_connection("127.0.0.1", bound_port)
        writer.close()
        await writer.wait_closed()

        # 0.0.0.0 is NOT a valid bind address assertion;
        # verify sockets don't list 0.0.0.0.
        for sock in terminator._server.sockets:
            sock_host = sock.getsockname()[0]
            assert sock_host != "0.0.0.0", "Server must not bind to 0.0.0.0"
    finally:
        await terminator.stop()


# ---------------------------------------------------------------------------
# Integration: CONNECT → TLS termination + ALPN (a)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tls_termination_and_alpn(tmp_ca: tuple) -> None:
    """CONNECT to allowlisted host: TLS terminates, leaf chains to root, ALPN=h2. (a)"""
    ca_key, ca_cert, ca_cert_pem = tmp_ca

    tls_reader_captured: list[asyncio.StreamReader] = []
    tls_writer_captured: list[asyncio.StreamWriter] = []
    alpn_captured: list[str | None] = []

    async def capture_dispatch(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        host: str,
        port: int,
    ) -> None:
        ssl_obj = writer.get_extra_info("ssl_object")
        alpn = ssl_obj.selected_alpn_protocol() if ssl_obj else None
        alpn_captured.append(alpn)
        tls_reader_captured.append(reader)
        tls_writer_captured.append(writer)
        # Keep alive briefly so client can complete handshake reads.
        await asyncio.sleep(0.05)

    terminator = AgyCONNECTTerminator(
        allowlist=frozenset({ALLOWLIST_HOST}),
        dispatch=capture_dispatch,
        ca_key=ca_key,
        ca_cert=ca_cert,
    )
    await terminator.start()
    try:
        proxy_host, proxy_port = terminator.address

        # Step 1: TCP CONNECT.
        raw_reader, raw_writer = await asyncio.open_connection(proxy_host, proxy_port)
        connect_req = f"CONNECT {ALLOWLIST_HOST}:443 HTTP/1.1\r\nHost: {ALLOWLIST_HOST}:443\r\n\r\n"
        raw_writer.write(connect_req.encode())
        await raw_writer.drain()
        response = await raw_reader.readline()
        assert b"200" in response, f"Expected 200, got {response!r}"

        # Step 2: TLS handshake on the now-tunnelled connection.
        # We must detach the raw socket from the existing asyncio transport
        # before wrapping it in a new TLS transport — reusing the fd while
        # owned by another transport raises RuntimeError on Python 3.14.
        raw_writer.transport.pause_reading()

        client_ssl_ctx = _build_client_ssl_context(ca_cert_pem)
        loop = asyncio.get_event_loop()

        # Use start_tls to upgrade the existing transport.
        new_transport = await loop.start_tls(
            raw_writer.transport,
            raw_writer.transport.get_protocol(),
            client_ssl_ctx,
            server_hostname=ALLOWLIST_HOST,
        )
        alpn = new_transport.get_extra_info("ssl_object").selected_alpn_protocol()
        assert alpn == "h2", f"Expected h2 ALPN, got {alpn!r}"

        new_transport.close()
    finally:
        await terminator.stop()


# ---------------------------------------------------------------------------
# Integration: leaf cert cache reuse (b)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leaf_cache_reuse_across_connections(tmp_ca: tuple) -> None:
    """Two sequential CONNECT to same allowlisted host reuse the same leaf cert. (b)"""
    ca_key, ca_cert, ca_cert_pem = tmp_ca

    async def serial_dispatch(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        host: str,
        port: int,
    ) -> None:
        await asyncio.sleep(0.05)

    terminator = AgyCONNECTTerminator(
        allowlist=frozenset({ALLOWLIST_HOST}),
        dispatch=serial_dispatch,
        ca_key=ca_key,
        ca_cert=ca_cert,
    )
    await terminator.start()

    try:
        proxy_host, proxy_port = terminator.address

        async def do_connect_and_tls() -> int:
            raw_reader, raw_writer = await asyncio.open_connection(proxy_host, proxy_port)
            raw_writer.write(
                f"CONNECT {ALLOWLIST_HOST}:443 HTTP/1.1\r\nHost: {ALLOWLIST_HOST}:443\r\n\r\n".encode()
            )
            await raw_writer.drain()
            await raw_reader.readline()  # 200 response

            client_ssl_ctx = _build_client_ssl_context(ca_cert_pem)
            loop = asyncio.get_event_loop()
            # Upgrade existing transport to TLS via start_tls (avoids fd reuse error).
            raw_writer.transport.pause_reading()
            new_transport = await loop.start_tls(
                raw_writer.transport,
                raw_writer.transport.get_protocol(),
                client_ssl_ctx,
                server_hostname=ALLOWLIST_HOST,
            )
            ssl_obj = new_transport.get_extra_info("ssl_object")
            cert_der = ssl_obj.getpeercert(binary_form=True)
            cert = x509.load_der_x509_certificate(cert_der)
            serial = cert.serial_number
            new_transport.close()
            return serial

        serial1 = await do_connect_and_tls()
        serial2 = await do_connect_and_tls()
        assert serial1 == serial2, f"Expected same serial, got {serial1} vs {serial2}"
    finally:
        await terminator.stop()


# ---------------------------------------------------------------------------
# Integration: non-allowlist → blind tunnel (c)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blind_tunnel_byte_faithful(tmp_ca: tuple) -> None:
    """Non-allowlisted CONNECT: bytes round-trip unmodified via plain TCP echo server. (c)"""
    ca_key, ca_cert, _ = tmp_ca

    # Spin up a plain TCP echo server.
    echo_host = "127.0.0.1"

    async def echo_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            if data:
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()

    echo_server = await asyncio.start_server(echo_handler, echo_host, 0)
    echo_port = echo_server.sockets[0].getsockname()[1]

    terminator = AgyCONNECTTerminator(
        allowlist=frozenset({ALLOWLIST_HOST}),  # echo host NOT in allowlist
        ca_key=ca_key,
        ca_cert=ca_cert,
    )
    await terminator.start()

    try:
        proxy_host, proxy_port = terminator.address

        raw_reader, raw_writer = await asyncio.open_connection(proxy_host, proxy_port)
        connect_req = (
            f"CONNECT {echo_host}:{echo_port} HTTP/1.1\r\nHost: {echo_host}:{echo_port}\r\n\r\n"
        )
        raw_writer.write(connect_req.encode())
        await raw_writer.drain()
        response = await raw_reader.readline()
        assert b"200" in response, f"Expected 200 for blind tunnel, got {response!r}"
        # Drain the blank line separating HTTP status from body.
        await raw_reader.readline()

        # Send payload and expect it echoed back verbatim — no TLS wrapping.
        payload = b"hello blind tunnel \x00\x01\x02"
        raw_writer.write(payload)
        await raw_writer.drain()

        received = await asyncio.wait_for(raw_reader.read(len(payload)), timeout=5.0)
        assert received == payload, f"Echo mismatch: {received!r} != {payload!r}"
    finally:
        await terminator.stop()
        echo_server.close()
        await echo_server.wait_closed()


# ---------------------------------------------------------------------------
# Integration: self-loop guard (e)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_loop_guard_via_https_proxy_env(
    tmp_ca: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTPS_PROXY pointing at loopback must be refused. (e)"""
    ca_key, ca_cert, _ = tmp_ca
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")

    terminator = AgyCONNECTTerminator(
        allowlist=frozenset({ALLOWLIST_HOST}),
        ca_key=ca_key,
        ca_cert=ca_cert,
    )
    await terminator.start()

    try:
        proxy_host, proxy_port = terminator.address
        raw_reader, raw_writer = await asyncio.open_connection(proxy_host, proxy_port)
        connect_req = (
            f"CONNECT {NON_ALLOWLIST_HOST}:443 HTTP/1.1\r\nHost: {NON_ALLOWLIST_HOST}:443\r\n\r\n"
        )
        raw_writer.write(connect_req.encode())
        await raw_writer.drain()
        response = await raw_reader.readline()
        assert b"403" in response, f"Expected 403 when HTTPS_PROXY is loopback, got {response!r}"
    finally:
        await terminator.stop()


# ---------------------------------------------------------------------------
# Integration: AgyCONNECTTerminator context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminator_context_manager(tmp_ca: tuple) -> None:
    """async with AgyCONNECTTerminator works correctly."""
    ca_key, ca_cert, _ = tmp_ca
    async with AgyCONNECTTerminator(ca_key=ca_key, ca_cert=ca_cert) as t:
        host, port = t.address
        assert host == "127.0.0.1"
        assert port > 0
    assert t._server is None


# ---------------------------------------------------------------------------
# Integration: bad CONNECT request → 400
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bad_connect_returns_400(tmp_ca: tuple) -> None:
    """Malformed (non-CONNECT) request returns 400."""
    ca_key, ca_cert, _ = tmp_ca
    async with AgyCONNECTTerminator(ca_key=ca_key, ca_cert=ca_cert) as t:
        proxy_host, proxy_port = t.address
        reader, writer = await asyncio.open_connection(proxy_host, proxy_port)
        writer.write(b"GET / HTTP/1.1\r\n\r\n")
        await writer.drain()
        response = await reader.readline()
        assert b"400" in response
        writer.close()


# ---------------------------------------------------------------------------
# Tests: load_cert_chain_in_memory used in terminator (headroom-oqb.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminator_no_tmpfile_on_linux(
    tmp_ca: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Linux (memfd_create available), the self-terminate TLS path calls
    load_cert_chain only with /proc/self/fd/ paths — never a regular fs path."""
    import os
    import ssl as _ssl

    ca_key, ca_cert, ca_cert_pem = tmp_ca

    if not hasattr(os, "memfd_create"):
        pytest.skip("memfd_create not available; primary path not applicable")

    leaf_fs_paths: list[str] = []
    original_load = _ssl.SSLContext.load_cert_chain

    def _spy_load(
        self: _ssl.SSLContext, certfile: str, keyfile: object = None, **kwargs: object
    ) -> None:
        if not certfile.startswith("/proc/self/fd/"):
            leaf_fs_paths.append(certfile)
        original_load(self, certfile, keyfile, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_ssl.SSLContext, "load_cert_chain", _spy_load)

    dispatch_called = [False]

    async def _capture_dispatch(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        host: str,
        port: int,
    ) -> None:
        dispatch_called[0] = True
        await asyncio.sleep(0.05)

    terminator = AgyCONNECTTerminator(
        allowlist=frozenset({ALLOWLIST_HOST}),
        dispatch=_capture_dispatch,
        ca_key=ca_key,
        ca_cert=ca_cert,
    )
    await terminator.start()
    try:
        proxy_host, proxy_port = terminator.address
        raw_reader, raw_writer = await asyncio.open_connection(proxy_host, proxy_port)
        connect_req = (
            f"CONNECT {ALLOWLIST_HOST}:443 HTTP/1.1\r\nHost: {ALLOWLIST_HOST}:443\r\n\r\n"
        )
        raw_writer.write(connect_req.encode())
        await raw_writer.drain()
        response = await raw_reader.readline()
        assert b"200" in response

        # Perform TLS handshake using the self-terminate path.
        client_ssl_ctx = _build_client_ssl_context(ca_cert_pem)
        loop = asyncio.get_event_loop()
        raw_writer.transport.pause_reading()
        new_transport = await loop.start_tls(
            raw_writer.transport,
            raw_writer.transport.get_protocol(),
            client_ssl_ctx,
            server_hostname=ALLOWLIST_HOST,
        )
        new_transport.close()
    finally:
        await terminator.stop()

    assert not leaf_fs_paths, (
        f"load_cert_chain must only use /proc/self/fd/ on Linux (memfd), "
        f"but got regular fs paths: {leaf_fs_paths}"
    )


@pytest.mark.asyncio
async def test_terminator_self_terminate_path_works_via_helper(tmp_ca: tuple) -> None:
    """Self-terminate TLS path (legacy dispatch callback) completes handshake
    via load_cert_chain_in_memory — regression guard."""
    ca_key, ca_cert, ca_cert_pem = tmp_ca

    dispatch_called = [False]

    async def _capture_dispatch(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        host: str,
        port: int,
    ) -> None:
        dispatch_called[0] = True
        await asyncio.sleep(0.05)

    terminator = AgyCONNECTTerminator(
        allowlist=frozenset({ALLOWLIST_HOST}),
        dispatch=_capture_dispatch,
        ca_key=ca_key,
        ca_cert=ca_cert,
    )
    await terminator.start()
    try:
        proxy_host, proxy_port = terminator.address
        raw_reader, raw_writer = await asyncio.open_connection(proxy_host, proxy_port)
        raw_writer.write(
            f"CONNECT {ALLOWLIST_HOST}:443 HTTP/1.1\r\nHost: {ALLOWLIST_HOST}:443\r\n\r\n".encode()
        )
        await raw_writer.drain()
        response = await raw_reader.readline()
        assert b"200" in response, f"Expected 200, got {response!r}"

        client_ssl_ctx = _build_client_ssl_context(ca_cert_pem)
        loop = asyncio.get_event_loop()
        raw_writer.transport.pause_reading()
        new_transport = await loop.start_tls(
            raw_writer.transport,
            raw_writer.transport.get_protocol(),
            client_ssl_ctx,
            server_hostname=ALLOWLIST_HOST,
        )
        # Handshake completed successfully; tear down.
        new_transport.close()
    finally:
        await terminator.stop()

    assert dispatch_called[0], "Dispatch callback must have been invoked"


# ---------------------------------------------------------------------------
# Regression: header-drain timeout aborts (no splice)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_header_timeout_aborts(tmp_ca: tuple) -> None:
    """Client stalls mid-headers after CONNECT line → connection aborted, no splice.

    Verifies defect fix: asyncio.TimeoutError in header drain must close
    client_writer and return, never proceeding to _handle_mitm/_handle_blind_tunnel.
    """
    import unittest.mock as mock

    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _handle_connect, _LeafCache

    ca_key, ca_cert, _ = tmp_ca
    leaf_cache = _LeafCache(max_size=4)

    mitm_called = False
    blind_called = False

    async def _fake_mitm(*args: object, **kwargs: object) -> None:
        nonlocal mitm_called
        mitm_called = True

    async def _fake_blind(*args: object, **kwargs: object) -> None:
        nonlocal blind_called
        blind_called = True

    # Feed CONNECT line, then nothing — header-drain readline will block.
    client_reader = asyncio.StreamReader()
    client_reader.feed_data(b"CONNECT notallowlisted.example.com:443 HTTP/1.1\r\n")

    close_called = False

    class _TrackingWriter:
        def get_extra_info(self, key: str, default: object = None) -> object:  # noqa: ANN401
            if key == "peername":
                return ("127.0.0.1", 1234)
            return default

        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

        def close(self) -> None:
            nonlocal close_called
            close_called = True

        async def wait_closed(self) -> None:
            pass

    client_writer = _TrackingWriter()  # type: ignore[assignment]

    # Tiny timeout so the header-drain readline genuinely times out fast.
    with (
        mock.patch.object(_mod, "_handle_mitm", _fake_mitm),
        mock.patch.object(_mod, "_handle_blind_tunnel", _fake_blind),
        mock.patch.object(_mod, "_CONNECT_TIMEOUT", 0.01),
    ):
        await _handle_connect(
            client_reader,
            client_writer,  # type: ignore[arg-type]
            allowlist=frozenset(),
            leaf_cache=leaf_cache,
            ca_key=ca_key,
            ca_cert=ca_cert,
            dispatch=None,  # type: ignore[arg-type]
        )

    assert close_called, "client_writer.close() must be called on header timeout"
    assert not mitm_called, "_handle_mitm must NOT be called on header timeout"
    assert not blind_called, "_handle_blind_tunnel must NOT be called on header timeout"


# ---------------------------------------------------------------------------
# Regression: upstream proxy header-drain timeout closes upstream writer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upstream_proxy_timeout_closes_writer() -> None:
    """Header-drain readline in _connect_via_upstream_proxy times out → upstream writer closed.

    A fake upstream proxy sends the 200 response line then stalls (never sends
    the blank-line header terminator).  With a tiny _CONNECT_TIMEOUT the
    header-drain readline times out and the upstream writer must be closed.
    """
    import unittest.mock as mock

    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _connect_via_upstream_proxy

    closed_writers: list[object] = []
    # Gate that the proxy releases when it has sent the 200 response.
    proxy_sent_200 = asyncio.Event()
    # Gate the proxy waits on so teardown can unblock it cleanly.
    proxy_release = asyncio.Event()

    async def _stall_proxy_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Accept CONNECT, reply 200, then stall without the blank-line terminator."""
        try:
            await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=2.0)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            pass
        writer.write(b"HTTP/1.1 200 Connection Established\r\n")
        await writer.drain()
        proxy_sent_200.set()
        # Block until teardown releases us (or until cancelled).
        try:
            await proxy_release.wait()
        except asyncio.CancelledError:
            pass
        finally:
            writer.close()

    proxy_server = await asyncio.start_server(
        _stall_proxy_handler, host="127.0.0.1", port=0
    )
    proxy_addr = proxy_server.sockets[0].getsockname()
    proxy_host, proxy_port = proxy_addr[0], proxy_addr[1]

    orig_open_conn = asyncio.open_connection

    async def _spy_open_conn(
        host: str, port: int, **kwargs: object
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        r, w = await orig_open_conn(host, port, **kwargs)
        orig_close = w.close

        def _tracked_close() -> None:
            closed_writers.append(w)
            orig_close()

        w.close = _tracked_close  # type: ignore[method-assign]
        return r, w

    try:
        with (
            mock.patch.object(_mod, "_CONNECT_TIMEOUT", 0.2),
            mock.patch("headroom.proxy.agy_terminator.asyncio.open_connection", _spy_open_conn),
        ):
            try:
                r, w = await _connect_via_upstream_proxy(
                    proxy_host, proxy_port, "target.example.com", 443, None
                )
                w.close()
                pytest.fail("Expected asyncio.TimeoutError from stalled header drain")
            except (asyncio.TimeoutError, OSError):
                pass  # expected path
    finally:
        proxy_release.set()  # unblock any stalled handler
        proxy_server.close()
        try:
            await asyncio.wait_for(proxy_server.wait_closed(), timeout=2.0)
        except asyncio.TimeoutError:
            pass

    assert closed_writers, "upstream writer must be closed when header-drain readline times out"


# ---------------------------------------------------------------------------
# Regression: blind tunnel drain error closes target writer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blind_tunnel_drain_error_closes_target() -> None:
    """client_writer.drain() raises before _blind_splice → target_writer is closed.

    Verifies defect fix: if the 200-response drain raises (client disconnected),
    target_writer must be closed to avoid fd leak.
    """
    import unittest.mock as mock

    from headroom.proxy.agy_terminator import _handle_blind_tunnel

    target_release = asyncio.Event()

    async def _idle_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            await target_release.wait()
        except asyncio.CancelledError:
            pass
        finally:
            writer.close()

    target_server = await asyncio.start_server(_idle_handler, host="127.0.0.1", port=0)
    target_addr = target_server.sockets[0].getsockname()
    target_host, target_port = target_addr[0], target_addr[1]

    target_writer_closed = False
    orig_open_conn = asyncio.open_connection

    async def _spy_target_conn(
        host: str, port: int, **kwargs: object
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        r, w = await orig_open_conn(host, port, **kwargs)
        orig_close = w.close

        def _tracked_close() -> None:
            nonlocal target_writer_closed
            target_writer_closed = True
            orig_close()

        w.close = _tracked_close  # type: ignore[method-assign]
        return r, w

    class _DrainFailWriter:
        """client_writer stub whose drain() always raises ConnectionResetError."""

        def get_extra_info(self, key: str, default: object = None) -> object:  # noqa: ANN401
            return ("127.0.0.1", 9999) if key == "peername" else default

        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            raise ConnectionResetError("client gone")

        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            pass

    client_reader = asyncio.StreamReader()

    try:
        with mock.patch(
            "headroom.proxy.agy_terminator.asyncio.open_connection", _spy_target_conn
        ):
            try:
                await _handle_blind_tunnel(
                    client_reader,
                    _DrainFailWriter(),  # type: ignore[arg-type]
                    target_host,
                    target_port,
                    None,
                )
            except Exception:  # noqa: BLE001
                pass  # any propagated exception is acceptable
    finally:
        target_release.set()
        target_server.close()
        try:
            await asyncio.wait_for(target_server.wait_closed(), timeout=2.0)
        except asyncio.TimeoutError:
            pass

    assert target_writer_closed, (
        "target_writer.close() must be called when client_writer.drain() raises before splice"
    )
