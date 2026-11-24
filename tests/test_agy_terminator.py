"""Tests for headroom.proxy.agy_terminator.

All tests use ephemeral ports and tmp_path; real ~/.headroom is never touched.
Tests use real asyncio connections over loopback to verify behavior.
"""

from __future__ import annotations

import asyncio
import datetime
import ssl

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


@pytest.mark.parametrize(
    "target",
    [
        "CloudCode-PA.googleapis.com:443",  # mixed case
        "cloudcode-pa.googleapis.com.:443",  # trailing root dot
    ],
)
def test_parse_connect_normalizes_equivalent_host_forms(target: str) -> None:
    """Equivalent spellings must reach the allowlist in one canonical form.

    The allowlist check is exact match, so an un-normalized target would fall
    through to the blind tunnel: the request still works but silently skips TLS
    termination and compression, with no signal that it was bypassed.
    """
    host, port = _parse_connect(f"CONNECT {target} HTTP/1.1")
    assert host == "cloudcode-pa.googleapis.com"
    assert port == 443
    assert host in DEFAULT_ALLOWLIST


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


def test_is_loopback_ipv4_shorthand_dotted() -> None:
    """127.1 is a valid inet_aton shorthand for 127.0.0.1."""
    assert _is_loopback("127.1") is True


def test_is_loopback_ipv4_decimal() -> None:
    """2130706433 is the decimal encoding of 127.0.0.1."""
    assert _is_loopback("2130706433") is True


def test_is_loopback_zero_shorthand() -> None:
    """0 is inet_aton shorthand for 0.0.0.0 (unspecified, treated as loopback)."""
    assert _is_loopback("0") is True


def test_is_loopback_unspecified() -> None:
    """0.0.0.0 is is_unspecified, not is_loopback, but Linux connect() reaches localhost."""
    assert _is_loopback("0.0.0.0") is True


def test_is_loopback_localhost_trailing_dot() -> None:
    assert _is_loopback("localhost.") is True


def test_is_loopback_ipv4_mapped_ipv6() -> None:
    """Must not depend on interpreter version (CPython gh-103365, fixed in 3.13)."""
    assert _is_loopback("::ffff:127.0.0.1") is True


def test_is_loopback_still_no_dns_for_shorthand_lookalike() -> None:
    """example.com must still return False — the function stays DNS-free."""
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
    # Exact SAN match (not substring/membership) — the leaf carries exactly one dNSName.
    assert dns_names == ["api.example.com"]


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
    # After max_size=1 eviction the sole cached key is exactly host-b.
    assert list(cache._cache) == ["host-b.example.com"]


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
        dispatch_port=1,
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
        dispatch_port=1,
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
        raw_writer.close()
        await raw_writer.wait_closed()
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
        dispatch_port=1,
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


@pytest.mark.asyncio
async def test_non_http_upstream_proxy_scheme_rejected(
    tmp_ca: tuple, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A non-http(s) HTTPS_PROXY (e.g. socks5://) must be refused before any

    HTTP CONNECT text is written into it, and the refusal must never leak
    the credential embedded in the proxy URL's userinfo.
    """
    import headroom.proxy.agy_terminator as _mod

    ca_key, ca_cert, _ = tmp_ca
    monkeypatch.setenv("HTTPS_PROXY", "socks5://user:s3cr3t@proxy:1080")

    called = False

    async def _spy(*args: object, **kwargs: object) -> tuple[object, object]:
        nonlocal called
        called = True
        raise AssertionError("_connect_via_upstream_proxy must not be reached")

    monkeypatch.setattr(_mod, "_connect_via_upstream_proxy", _spy)

    terminator = AgyCONNECTTerminator(
        allowlist=frozenset({ALLOWLIST_HOST}),
        ca_key=ca_key,
        ca_cert=ca_cert,
        dispatch_port=1,
    )
    await terminator.start()

    try:
        with caplog.at_level("WARNING"):
            proxy_host, proxy_port = terminator.address
            raw_reader, raw_writer = await asyncio.open_connection(proxy_host, proxy_port)
            connect_req = (
                f"CONNECT {NON_ALLOWLIST_HOST}:443 HTTP/1.1\r\n"
                f"Host: {NON_ALLOWLIST_HOST}:443\r\n\r\n"
            )
            raw_writer.write(connect_req.encode())
            await raw_writer.drain()
            response = await raw_reader.readline()
        assert b"403" in response, f"Expected 403 for socks5:// upstream, got {response!r}"
        assert called is False
        assert "s3cr3t" not in caplog.text
    finally:
        await terminator.stop()


# ---------------------------------------------------------------------------
# Integration: AgyCONNECTTerminator context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminator_context_manager(tmp_ca: tuple) -> None:
    """async with AgyCONNECTTerminator works correctly."""
    ca_key, ca_cert, _ = tmp_ca
    async with AgyCONNECTTerminator(dispatch_port=1, ca_key=ca_key, ca_cert=ca_cert) as t:
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
    async with AgyCONNECTTerminator(dispatch_port=1, ca_key=ca_key, ca_cert=ca_cert) as t:
        proxy_host, proxy_port = t.address
        reader, writer = await asyncio.open_connection(proxy_host, proxy_port)
        writer.write(b"GET / HTTP/1.1\r\n\r\n")
        await writer.drain()
        response = await reader.readline()
        assert b"400" in response
        writer.close()


# ---------------------------------------------------------------------------
# Regression: header-drain timeout aborts (no splice)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("first_eof", ["client", "target"])
async def test_blind_splice_drains_normal_half_closes(first_eof: str) -> None:
    from headroom.proxy.agy_terminator import _blind_splice

    request = b"request before sending EOF"
    response = b"response after receiving EOF"
    received = []
    upstream_done = asyncio.Event()
    relay_done = asyncio.Event()

    async def upstream(reader, writer):
        try:
            if first_eof == "target":
                writer.write(request)
                await writer.drain()
                writer.write_eof()
                received.append(await reader.read())
            else:
                received.append(await reader.read())
                writer.write(response)
                await writer.drain()
                writer.write_eof()
        finally:
            writer.close()
            await writer.wait_closed()
            upstream_done.set()

    target = await asyncio.start_server(upstream, "127.0.0.1", 0)
    target_port = target.sockets[0].getsockname()[1]

    async def relay(reader, writer):
        try:
            target_reader, target_writer = await asyncio.open_connection("127.0.0.1", target_port)
            await _blind_splice(reader, writer, target_reader, target_writer)
        finally:
            relay_done.set()

    proxy = await asyncio.start_server(relay, "127.0.0.1", 0)
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy.sockets[0].getsockname()[1])
    try:
        if first_eof == "client":
            writer.write(request)
            await writer.drain()
            writer.write_eof()
            assert await asyncio.wait_for(reader.read(), 2) == response
        else:
            assert await asyncio.wait_for(reader.read(), 2) == request
            writer.write(response)
            await writer.drain()
            writer.write_eof()
        await asyncio.wait_for(upstream_done.wait(), 2)
        await asyncio.wait_for(relay_done.wait(), 2)
        assert received == [request if first_eof == "client" else response]
    finally:
        writer.close()
        await writer.wait_closed()
        proxy.close()
        target.close()
        await proxy.wait_closed()
        await target.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        b"X-Header: value\r\n" * 101,
        (b"X-Header: " + b"x" * 2048 + b"\r\n") * 20,
        b"X-Header: " + b"x" * 70000 + b"\r\n",
    ],
    ids=["count", "total-bytes", "oversized-line"],
)
async def test_connect_rejects_excessive_headers(tmp_ca: tuple, headers: bytes) -> None:
    key, cert, _ = tmp_ca
    async with AgyCONNECTTerminator(ca_key=key, ca_cert=cert, dispatch_port=1) as server:
        reader, writer = await asyncio.open_connection(*server.address)
        try:
            writer.write(
                b"CONNECT daily-cloudcode-pa.googleapis.com:443 HTTP/1.1\r\n" + headers + b"\r\n"
            )
            await writer.drain()
            status = await asyncio.wait_for(reader.readline(), 2)
            assert b"431" in status
        finally:
            writer.close()
            await writer.wait_closed()


@pytest.mark.asyncio
async def test_connect_header_deadline_is_not_reset_by_each_line(
    tmp_ca: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("headroom.proxy.agy_terminator._CONNECT_TIMEOUT", 0.12)
    key, cert, _ = tmp_ca
    async with AgyCONNECTTerminator(ca_key=key, ca_cert=cert, dispatch_port=1) as server:
        reader, writer = await asyncio.open_connection(*server.address)

        async def send_headers():
            writer.write(b"CONNECT daily-cloudcode-pa.googleapis.com:443 HTTP/1.1\r\n")
            for _ in range(6):
                writer.write(b"X-Header: value\r\n")
                await writer.drain()
                await asyncio.sleep(0.04)
            writer.write(b"\r\n")
            await writer.drain()

        sender = asyncio.create_task(send_headers())
        try:
            try:
                status = await asyncio.wait_for(reader.readline(), 1)
            except ConnectionResetError:
                # Windows may reset a timed-out socket with unread headers.
                status = b""
            assert status == b""
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionResetError:
                pass


@pytest.mark.asyncio
async def test_connect_header_timeout_aborts() -> None:
    """Client stalls mid-headers after CONNECT line → connection aborted, no splice.

    Verifies defect fix: asyncio.TimeoutError in header drain must close
    client_writer and return, never proceeding to _handle_mitm/_handle_blind_tunnel.
    """
    import unittest.mock as mock

    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _handle_connect

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
            dispatch_port=1,
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

    proxy_server = await asyncio.start_server(_stall_proxy_handler, host="127.0.0.1", port=0)
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
                    proxy_host, proxy_port, "target.example.com", 443, None, None
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

    async def _idle_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
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
        with mock.patch("headroom.proxy.agy_terminator.asyncio.open_connection", _spy_target_conn):
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


# ---------------------------------------------------------------------------
# Coverage: _LeafCache re-mints an expired leaf in place
# ---------------------------------------------------------------------------


def test_leaf_cache_expired_entry_remints(tmp_ca: tuple) -> None:
    """An expired cache entry (not_valid_after in the past) is re-minted."""
    ca_key, ca_cert, _ = tmp_ca
    cache = _LeafCache(max_size=10)
    cert1, _ = cache.get_or_mint("expiring.example.com", ca_key, ca_cert)
    obj1 = x509.load_pem_x509_certificate(cert1)

    # Force the cached entry to look expired.
    cert_pem, key_pem, _ = cache._cache["expiring.example.com"]
    past = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=1)
    cache._cache["expiring.example.com"] = (cert_pem, key_pem, past)

    cert2, _ = cache.get_or_mint("expiring.example.com", ca_key, ca_cert)
    obj2 = x509.load_pem_x509_certificate(cert2)
    assert obj1.serial_number != obj2.serial_number, "Expired leaf must be re-minted"


# ---------------------------------------------------------------------------
# Coverage: _splice_half swallows writer.write_eof() exceptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_splice_half_write_eof_exception_swallowed() -> None:
    """_splice_half must swallow exceptions raised by writer.write_eof()."""
    from headroom.proxy.agy_terminator import _splice_half

    reader = asyncio.StreamReader()
    reader.feed_data(b"payload")
    reader.feed_eof()

    written = bytearray()

    class _EofRaisingWriter:
        def write(self, data: bytes) -> None:
            written.extend(data)

        async def drain(self) -> None:
            pass

        def write_eof(self) -> None:
            raise RuntimeError("eof boom")

    # Must not raise, despite write_eof() raising internally.
    await _splice_half(reader, _EofRaisingWriter())  # type: ignore[arg-type]
    assert bytes(written) == b"payload"


# ---------------------------------------------------------------------------
# Coverage: _blind_splice except-branch cancels both pump tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blind_splice_wait_exception_cancels_both_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If awaiting the pump tasks raises, both tasks are cancelled and both
    writers are still closed via the finally block."""
    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _blind_splice

    client_reader = asyncio.StreamReader()
    target_reader = asyncio.StreamReader()
    closed = {"client": False, "target": False}

    class _W:
        def __init__(self, name: str) -> None:
            self._name = name

        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

        def write_eof(self) -> None:
            pass

        def close(self) -> None:
            closed[self._name] = True

        async def wait_closed(self) -> None:
            pass

    client_writer = _W("client")
    target_writer = _W("target")

    async def _raise_wait(*args: object, **kwargs: object) -> None:
        raise RuntimeError("pump wait boom")

    monkeypatch.setattr(_mod.asyncio, "wait", _raise_wait)

    await _blind_splice(
        client_reader,
        client_writer,  # type: ignore[arg-type]
        target_reader,
        target_writer,  # type: ignore[arg-type]
    )

    assert closed["client"] is True
    assert closed["target"] is True


# ---------------------------------------------------------------------------
# Coverage: _handle_connect first-line CONNECT read timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_connect_first_line_timeout_closes_writer() -> None:
    """First readline() (the CONNECT line itself) times out -> client_writer
    is closed and neither MITM nor blind-tunnel dispatch runs."""
    import unittest.mock as mock

    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _handle_connect

    client_reader = asyncio.StreamReader()  # No data fed -> readline() blocks forever.

    close_called = False

    class _TrackingWriter:
        def get_extra_info(self, key: str, default: object = None) -> object:  # noqa: ANN401
            return ("127.0.0.1", 1234) if key == "peername" else default

        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

        def close(self) -> None:
            nonlocal close_called
            close_called = True

        async def wait_closed(self) -> None:
            pass

    client_writer = _TrackingWriter()

    with mock.patch.object(_mod, "_CONNECT_TIMEOUT", 0.01):
        await _handle_connect(
            client_reader,
            client_writer,  # type: ignore[arg-type]
            allowlist=frozenset(),
            dispatch_port=1,
        )

    assert close_called, "client_writer.close() must be called on first-line CONNECT timeout"


# ---------------------------------------------------------------------------
# Coverage: Proxy-Authorization header is parsed off the CONNECT request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_authorization_header_parsed(tmp_ca: tuple) -> None:
    """CONNECT with a Proxy-Authorization header is accepted and tunnels bytes."""
    ca_key, ca_cert, _ = tmp_ca
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
        allowlist=frozenset({ALLOWLIST_HOST}),  # echo host NOT allowlisted -> blind tunnel
        ca_key=ca_key,
        ca_cert=ca_cert,
        dispatch_port=1,
    )
    await terminator.start()
    try:
        proxy_host, proxy_port = terminator.address
        raw_reader, raw_writer = await asyncio.open_connection(proxy_host, proxy_port)
        connect_req = (
            f"CONNECT {echo_host}:{echo_port} HTTP/1.1\r\n"
            f"Host: {echo_host}:{echo_port}\r\n"
            "Proxy-Authorization: Basic dXNlcjpwYXNz\r\n"
            "\r\n"
        )
        raw_writer.write(connect_req.encode())
        await raw_writer.drain()
        response = await raw_reader.readline()
        assert b"200" in response, f"Expected 200, got {response!r}"
        await raw_reader.readline()  # Drain the blank line separating status from body.

        payload = b"auth header parsed ok"
        raw_writer.write(payload)
        await raw_writer.drain()
        received = await asyncio.wait_for(raw_reader.read(len(payload)), timeout=5.0)
        assert received == payload
    finally:
        raw_writer.close()
        await raw_writer.wait_closed()
        await terminator.stop()
        echo_server.close()
        await echo_server.wait_closed()


# ---------------------------------------------------------------------------
# Coverage: dispatch_port SUCCESS — ACK + blind-splice to loopback dispatch server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_port_success_splices_to_dispatch_server(tmp_ca: tuple) -> None:
    """Allowlisted CONNECT with a reachable dispatch_port: ACK written and raw
    bytes are byte-spliced to the loopback dispatch server (no TLS)."""
    ca_key, ca_cert, _ = tmp_ca
    echo_host = "127.0.0.1"

    async def echo_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            if data:
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()

    dispatch_server = await asyncio.start_server(echo_handler, echo_host, 0)
    dispatch_port = dispatch_server.sockets[0].getsockname()[1]

    terminator = AgyCONNECTTerminator(
        allowlist=frozenset({ALLOWLIST_HOST}),
        ca_key=ca_key,
        ca_cert=ca_cert,
        dispatch_port=dispatch_port,
    )
    await terminator.start()
    try:
        proxy_host, proxy_port = terminator.address
        raw_reader, raw_writer = await asyncio.open_connection(proxy_host, proxy_port)
        connect_req = f"CONNECT {ALLOWLIST_HOST}:443 HTTP/1.1\r\nHost: {ALLOWLIST_HOST}:443\r\n\r\n"
        raw_writer.write(connect_req.encode())
        await raw_writer.drain()
        response = await raw_reader.readline()
        assert b"200" in response, f"Expected ACK 200, got {response!r}"
        await raw_reader.readline()  # Drain the blank line separating status from body.

        payload = b"raw bytes over dispatch splice"
        raw_writer.write(payload)
        await raw_writer.drain()
        received = await asyncio.wait_for(raw_reader.read(len(payload)), timeout=5.0)
        assert received == payload, f"Splice mismatch: {received!r} != {payload!r}"
        # Let the target-side echo connection close, unblocking the server-side
        # _blind_splice call so it runs to completion (its own return statement)
        # before teardown — otherwise the background handler task may be torn
        # down mid-flight.
        await asyncio.sleep(0.05)
    finally:
        raw_writer.close()
        await raw_writer.wait_closed()
        await terminator.stop()
        dispatch_server.close()
        await dispatch_server.wait_closed()


@pytest.mark.asyncio
async def test_dispatch_port_connect_failed_close_exception_swallowed() -> None:
    """dispatch_connect_failed handling: if client_writer.close() itself also
    raises, the inner except swallows it (headroom-vro.2: lines 461-462)."""
    from headroom.proxy.agy_terminator import _handle_mitm

    # Bind then immediately close an ephemeral port so connecting to it
    # deterministically raises ConnectionRefusedError (an OSError subclass).
    probe = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    dead_port = probe.sockets[0].getsockname()[1]
    probe.close()
    await probe.wait_closed()

    close_called = False

    class _RaisingCloseWriter:
        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

        def close(self) -> None:
            nonlocal close_called
            close_called = True
            raise RuntimeError("close boom")

        async def wait_closed(self) -> None:
            pass

    client_reader = asyncio.StreamReader()
    client_writer = _RaisingCloseWriter()

    # Must not raise, despite client_writer.close() raising inside the handler.
    await _handle_mitm(
        client_reader,
        client_writer,  # type: ignore[arg-type]
        dead_port,
    )
    assert close_called


# ---------------------------------------------------------------------------
# Coverage: dispatch_port connect failure closes the client after ACK
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_port_unreachable_closes_client_after_ack(tmp_ca: tuple) -> None:
    """Allowlisted CONNECT with an unreachable dispatch_port: ACK is still sent,
    then the connect attempt fails and client_writer is closed."""
    ca_key, ca_cert, _ = tmp_ca

    # Bind then immediately close an ephemeral port so connecting to it
    # deterministically raises ConnectionRefusedError (an OSError subclass).
    probe = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    dead_port = probe.sockets[0].getsockname()[1]
    probe.close()
    await probe.wait_closed()

    terminator = AgyCONNECTTerminator(
        allowlist=frozenset({ALLOWLIST_HOST}),
        ca_key=ca_key,
        ca_cert=ca_cert,
        dispatch_port=dead_port,
    )
    await terminator.start()
    try:
        proxy_host, proxy_port = terminator.address
        raw_reader, raw_writer = await asyncio.open_connection(proxy_host, proxy_port)
        connect_req = f"CONNECT {ALLOWLIST_HOST}:443 HTTP/1.1\r\nHost: {ALLOWLIST_HOST}:443\r\n\r\n"
        raw_writer.write(connect_req.encode())
        await raw_writer.drain()
        response = await raw_reader.readline()
        assert b"200" in response, (
            f"Expected ACK 200 before dispatch-connect attempt, got {response!r}"
        )
        await raw_reader.readline()  # Drain the blank line separating status from body.

        # dispatch connect failed -> client_writer.close() -> EOF, no more data.
        data = await asyncio.wait_for(raw_reader.read(10), timeout=5.0)
        assert data == b""
    finally:
        await terminator.stop()


# ---------------------------------------------------------------------------
# Coverage: blind tunnel upstream connect failure -> 502 Bad Gateway
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blind_tunnel_connect_failure_returns_502(
    tmp_ca: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-allowlisted CONNECT whose upstream open_connection raises OSError
    must result in a 502 Bad Gateway response."""
    import headroom.proxy.agy_terminator as _mod

    ca_key, ca_cert, _ = tmp_ca

    terminator = AgyCONNECTTerminator(
        allowlist=frozenset({ALLOWLIST_HOST}),  # target NOT allowlisted -> blind tunnel
        ca_key=ca_key,
        ca_cert=ca_cert,
        dispatch_port=1,
    )
    await terminator.start()
    try:
        proxy_host, proxy_port = terminator.address
        # Establish the client<->proxy connection BEFORE patching open_connection,
        # since that patch also covers this very call target.
        raw_reader, raw_writer = await asyncio.open_connection(proxy_host, proxy_port)

        async def _raise_oserror(
            host: str, port: int, **kwargs: object
        ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
            raise OSError("connection refused")

        monkeypatch.setattr(_mod.asyncio, "open_connection", _raise_oserror)

        connect_req = (
            f"CONNECT {NON_ALLOWLIST_HOST}:443 HTTP/1.1\r\nHost: {NON_ALLOWLIST_HOST}:443\r\n\r\n"
        )
        raw_writer.write(connect_req.encode())
        await raw_writer.drain()
        response = await raw_reader.readline()
        assert b"502 Bad Gateway" in response, f"Expected 502, got {response!r}"
    finally:
        await terminator.stop()


# ---------------------------------------------------------------------------
# Coverage: AgyCONNECTTerminator lifecycle edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminator_start_without_ca_key_uses_ensure_root_ca(tmp_path: object) -> None:
    """Omitting ca_key/ca_cert triggers the ensure_root_ca(base_dir=...) start path
    (local CA key generation under tmp_path; never touches real ~/.headroom)."""
    terminator = AgyCONNECTTerminator(dispatch_port=1, base_dir=tmp_path)  # type: ignore[arg-type]
    await terminator.start()
    try:
        host, port = terminator.address
        assert host == "127.0.0.1"
        assert port > 0
        assert terminator._ca_key is not None
        assert terminator._ca_cert is not None
    finally:
        await terminator.stop()


def test_address_before_start_raises_runtime_error() -> None:
    """Reading .address before .start() raises RuntimeError."""
    terminator = AgyCONNECTTerminator(dispatch_port=1)
    with pytest.raises(RuntimeError):
        _ = terminator.address


@pytest.mark.asyncio
async def test_stop_before_start_is_noop() -> None:
    """Calling .stop() before .start() (no server) is a no-op and does not raise."""
    terminator = AgyCONNECTTerminator(dispatch_port=1)
    await terminator.stop()
    assert terminator._server is None


# ---------------------------------------------------------------------------
# Regression: blind-tunnel target guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blind_tunnel_refuses_self_connect() -> None:
    """CONNECT to the terminator's own port must be refused, not tunnelled.

    Without the guard each nesting level costs two fds: a client that keeps
    re-CONNECTing through the terminator to itself exhausts them.
    """
    async with AgyCONNECTTerminator(dispatch_port=1) as term:
        _, port = term.address
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
        await writer.drain()
        response = await asyncio.wait_for(reader.read(64), timeout=5)
        writer.close()

    assert b"403" in response, f"expected 403 for self-connect, got {response!r}"


@pytest.mark.asyncio
async def test_blind_tunnel_refuses_link_local_metadata_host() -> None:
    """169.254.169.254 (cloud instance metadata) must not be reachable."""
    async with AgyCONNECTTerminator(dispatch_port=1) as term:
        _, port = term.address
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"CONNECT 169.254.169.254:80 HTTP/1.1\r\n\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(64), timeout=5)
        writer.close()

    assert b"403" in response, f"expected 403 for link-local target, got {response!r}"


@pytest.mark.parametrize(
    ("proxy_url", "expected_port"),
    [
        ("http://proxy.corp", 80),
        ("https://proxy.corp", 443),
        ("http://proxy.corp:3128", 3128),
    ],
)
@pytest.mark.asyncio
async def test_upstream_proxy_port_defaults_follow_scheme(
    monkeypatch: pytest.MonkeyPatch, proxy_url: str, expected_port: int
) -> None:
    """A port-less HTTPS_PROXY must be dialled per scheme, not always on :443."""
    import headroom.proxy.agy_terminator as _mod

    dialled: list[int] = []

    async def _spy(
        proxy_host: str,
        proxy_port: int,
        target_host: str,
        target_port: int,
        proxy_auth: str | None,
        ssl_context: ssl.SSLContext | None,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        dialled.append(proxy_port)
        raise OSError("stop here — the dialled port is what matters")

    monkeypatch.setenv("HTTPS_PROXY", proxy_url)
    monkeypatch.setattr(_mod, "_connect_via_upstream_proxy", _spy)

    async with AgyCONNECTTerminator(dispatch_port=1) as term:
        _, port = term.address
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n")
        await writer.drain()
        await asyncio.wait_for(reader.read(64), timeout=5)
        writer.close()

    assert dialled == [expected_port]


# ---------------------------------------------------------------------------
# Coverage: upstream CONNECT wire bytes + https-proxy TLS parameters
# ---------------------------------------------------------------------------


class _SpliceCompatWriter:
    """client_writer stub satisfying the subset of StreamWriter used by
    _handle_blind_tunnel and _blind_splice: write/drain/close/wait_closed for
    the 200-response, write_eof for the splice's finally-block once the
    target side reaches EOF.
    """

    def write(self, data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass

    def write_eof(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


async def _start_fake_upstream_proxy() -> tuple[asyncio.AbstractServer, str, int, list[bytes]]:
    """Bind a fake upstream proxy on an ephemeral loopback port.

    Records the raw bytes of the CONNECT request it receives, replies
    200 Connection Established, then closes -- the resulting target-side EOF
    is what lets _blind_splice's FIRST_COMPLETED race return promptly instead
    of hanging on an idle tunnel.
    """
    received: list[bytes] = []

    async def _handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            data = b""
        received.append(data)
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(_handler, host="127.0.0.1", port=0)
    host, port = server.sockets[0].getsockname()[:2]
    return server, host, port, received


@pytest.mark.asyncio
async def test_blind_tunnel_upstream_proxy_url_credential_emitted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """(a) HTTPS_PROXY URL userinfo is derived into Proxy-Authorization."""
    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _handle_blind_tunnel

    server, proxy_host, proxy_port, received = await _start_fake_upstream_proxy()
    # HARNESS TRAP: a fake proxy on 127.0.0.1 is refused by the self-loop
    # guard (_is_loopback) before _connect_via_upstream_proxy is ever
    # reached. This test targets auth-derivation, not the loopback guard
    # (which has its own dedicated tests), so bypass it explicitly.
    monkeypatch.setattr(_mod, "_is_loopback", lambda host: False)
    monkeypatch.setenv("HTTPS_PROXY", f"http://user:secret@{proxy_host}:{proxy_port}")
    client_reader = asyncio.StreamReader()
    client_reader.feed_eof()

    try:
        with caplog.at_level("DEBUG"):
            await asyncio.wait_for(
                _handle_blind_tunnel(
                    client_reader, _SpliceCompatWriter(), "target.example.com", 443, None
                ),
                timeout=5.0,
            )
    finally:
        server.close()
        await server.wait_closed()

    assert received, "fake proxy must have received a CONNECT request"
    assert b"Proxy-Authorization: Basic dXNlcjpzZWNyZXQ=\r\n" in received[0]
    assert "secret" not in caplog.text


@pytest.mark.asyncio
async def test_blind_tunnel_upstream_proxy_percent_decodes_credential(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """(b) Percent-encoded userinfo in HTTPS_PROXY is decoded before Basic auth."""
    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _handle_blind_tunnel

    server, proxy_host, proxy_port, received = await _start_fake_upstream_proxy()
    monkeypatch.setattr(_mod, "_is_loopback", lambda host: False)
    # user%40corp:p%40ss decodes to "user@corp:p@ss".
    monkeypatch.setenv("HTTPS_PROXY", f"http://user%40corp:p%40ss@{proxy_host}:{proxy_port}")
    client_reader = asyncio.StreamReader()
    client_reader.feed_eof()

    try:
        with caplog.at_level("DEBUG"):
            await asyncio.wait_for(
                _handle_blind_tunnel(
                    client_reader, _SpliceCompatWriter(), "target.example.com", 443, None
                ),
                timeout=5.0,
            )
    finally:
        server.close()
        await server.wait_closed()

    assert received, "fake proxy must have received a CONNECT request"
    assert b"Proxy-Authorization: Basic dXNlckBjb3JwOnBAc3M=\r\n" in received[0]
    assert "p@ss" not in caplog.text


@pytest.mark.asyncio
async def test_blind_tunnel_upstream_proxy_url_credential_beats_inbound_header(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """(c) HTTPS_PROXY URL userinfo takes precedence over an inbound header."""
    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _handle_blind_tunnel

    server, proxy_host, proxy_port, received = await _start_fake_upstream_proxy()
    monkeypatch.setattr(_mod, "_is_loopback", lambda host: False)
    monkeypatch.setenv("HTTPS_PROXY", f"http://user:secret@{proxy_host}:{proxy_port}")
    client_reader = asyncio.StreamReader()
    client_reader.feed_eof()
    inbound = "Basic aW5ib3VuZDpzZWNyZXQ="  # "inbound:secret" -- must be shadowed by the URL cred

    try:
        with caplog.at_level("DEBUG"):
            await asyncio.wait_for(
                _handle_blind_tunnel(
                    client_reader, _SpliceCompatWriter(), "target.example.com", 443, inbound
                ),
                timeout=5.0,
            )
    finally:
        server.close()
        await server.wait_closed()

    assert received, "fake proxy must have received a CONNECT request"
    assert b"Proxy-Authorization: Basic dXNlcjpzZWNyZXQ=\r\n" in received[0]
    assert inbound.encode() not in received[0]
    assert "secret" not in caplog.text


@pytest.mark.asyncio
async def test_blind_tunnel_upstream_proxy_inbound_header_used_when_no_userinfo(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """(d) No userinfo in HTTPS_PROXY -> the inbound header is forwarded as-is."""
    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _handle_blind_tunnel

    server, proxy_host, proxy_port, received = await _start_fake_upstream_proxy()
    monkeypatch.setattr(_mod, "_is_loopback", lambda host: False)
    monkeypatch.setenv("HTTPS_PROXY", f"http://{proxy_host}:{proxy_port}")
    client_reader = asyncio.StreamReader()
    client_reader.feed_eof()
    inbound = "Basic aW5ib3VuZDpzZWNyZXQ="  # "inbound:secret"

    try:
        with caplog.at_level("DEBUG"):
            await asyncio.wait_for(
                _handle_blind_tunnel(
                    client_reader, _SpliceCompatWriter(), "target.example.com", 443, inbound
                ),
                timeout=5.0,
            )
    finally:
        server.close()
        await server.wait_closed()

    assert received, "fake proxy must have received a CONNECT request"
    assert f"Proxy-Authorization: {inbound}\r\n".encode() in received[0]
    assert "secret" not in caplog.text


@pytest.mark.asyncio
async def test_blind_tunnel_upstream_proxy_no_credential_no_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(e) No URL userinfo and no inbound header -> no Proxy-Authorization line at all."""
    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _handle_blind_tunnel

    server, proxy_host, proxy_port, received = await _start_fake_upstream_proxy()
    monkeypatch.setattr(_mod, "_is_loopback", lambda host: False)
    monkeypatch.setenv("HTTPS_PROXY", f"http://{proxy_host}:{proxy_port}")
    client_reader = asyncio.StreamReader()
    client_reader.feed_eof()

    try:
        await asyncio.wait_for(
            _handle_blind_tunnel(
                client_reader, _SpliceCompatWriter(), "target.example.com", 443, None
            ),
            timeout=5.0,
        )
    finally:
        server.close()
        await server.wait_closed()

    assert received, "fake proxy must have received a CONNECT request"
    assert b"Proxy-Authorization" not in received[0]


@pytest.mark.asyncio
async def test_blind_tunnel_upstream_proxy_empty_password_no_crash(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """(e2) Username with an EMPTY password -> Basic b64("user:"), no crash from `password or ''`."""
    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _handle_blind_tunnel

    server, proxy_host, proxy_port, received = await _start_fake_upstream_proxy()
    monkeypatch.setattr(_mod, "_is_loopback", lambda host: False)
    monkeypatch.setenv("HTTPS_PROXY", f"http://user:@{proxy_host}:{proxy_port}")
    client_reader = asyncio.StreamReader()
    client_reader.feed_eof()

    try:
        with caplog.at_level("DEBUG"):
            await asyncio.wait_for(
                _handle_blind_tunnel(
                    client_reader, _SpliceCompatWriter(), "target.example.com", 443, None
                ),
                timeout=5.0,
            )
    finally:
        server.close()
        await server.wait_closed()

    assert received, "fake proxy must have received a CONNECT request"
    assert b"Proxy-Authorization: Basic dXNlcjo=\r\n" in received[0]
    assert "dXNlcjo=" not in caplog.text


@pytest.mark.asyncio
async def test_blind_tunnel_https_upstream_proxy_tls_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(f) https:// upstream proxy dials over TLS with verified defaults, SNI to the
    proxy host, and ALPN pinned to http/1.1.

    ALPN cannot be read back off a real ssl.SSLContext (set_alpn_protocols()
    forwards to the C layer and stores nothing readable; selected_alpn_protocol()
    only exists on a post-handshake SSLObject, which a mocked open_connection
    never produces). So this test proves the two halves separately:
    verify_mode/check_hostname/server_hostname off a REAL default context
    (first act), and the ALPN pin via set_alpn_protocols() call-args on a
    MOCKED context (second act).
    """
    import unittest.mock as mock

    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _handle_blind_tunnel

    monkeypatch.setattr(_mod, "_is_loopback", lambda host: False)
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.corp.example:9443")

    # --- act 1: real default context -> verify_mode / check_hostname / SNI ---
    captured: dict[str, object] = {}

    async def _fake_open_connection_real_ctx(
        host: str, port: int, *, ssl: object = None, server_hostname: object = None, **_: object
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        captured["ssl"] = ssl
        captured["server_hostname"] = server_hostname
        raise OSError("stop before real I/O -- the TLS context passed in is what's under test")

    with mock.patch.object(_mod.asyncio, "open_connection", _fake_open_connection_real_ctx):
        client_reader = asyncio.StreamReader()
        client_reader.feed_eof()
        await asyncio.wait_for(
            _handle_blind_tunnel(
                client_reader, _SpliceCompatWriter(), "target.example.com", 443, None
            ),
            timeout=5.0,
        )

    ctx = captured["ssl"]
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    assert captured["server_hostname"] == "proxy.corp.example"

    # --- act 2: mocked context -> ALPN pin asserted via call-args ---
    mock_ctx = mock.MagicMock(spec=ssl.SSLContext)

    async def _fake_open_connection_mock_ctx(
        host: str, port: int, **_: object
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        raise OSError("stop before real I/O -- ALPN pinning is what's under test")

    with (
        mock.patch.object(_mod.ssl, "create_default_context", lambda: mock_ctx),
        mock.patch.object(_mod.asyncio, "open_connection", _fake_open_connection_mock_ctx),
    ):
        client_reader = asyncio.StreamReader()
        client_reader.feed_eof()
        await asyncio.wait_for(
            _handle_blind_tunnel(
                client_reader, _SpliceCompatWriter(), "target.example.com", 443, None
            ),
            timeout=5.0,
        )

    mock_ctx.set_alpn_protocols.assert_called_once_with(["http/1.1"])


@pytest.mark.asyncio
async def test_blind_tunnel_http_upstream_proxy_dials_without_tls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(g) http:// upstream proxy dials plaintext -- ssl=None, no server_hostname/SNI."""
    import unittest.mock as mock

    import headroom.proxy.agy_terminator as _mod
    from headroom.proxy.agy_terminator import _handle_blind_tunnel

    monkeypatch.setattr(_mod, "_is_loopback", lambda host: False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp.example:3128")

    captured: dict[str, object] = {}

    async def _fake_open_connection(
        host: str, port: int, *, ssl: object = None, server_hostname: object = None, **_: object
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        captured["ssl"] = ssl
        captured["server_hostname"] = server_hostname
        raise OSError("stop before real I/O -- the ssl kwarg is what's under test")

    client_reader = asyncio.StreamReader()
    client_reader.feed_eof()

    with mock.patch.object(_mod.asyncio, "open_connection", _fake_open_connection):
        await asyncio.wait_for(
            _handle_blind_tunnel(
                client_reader, _SpliceCompatWriter(), "target.example.com", 443, None
            ),
            timeout=5.0,
        )

    assert captured["ssl"] is None
    assert captured["server_hostname"] is None


def test_upstream_proxy_auth_rejects_socks5_scheme() -> None:
    """(h) socks5:// upstream proxy URL never derives a Proxy-Authorization header.

    _upstream_proxy_auth is pure string/URL logic (no socket I/O), per its own
    docstring, so it is unit-tested directly rather than through
    _handle_blind_tunnel (which 403s a non-http(s) scheme before this helper's
    return value would even be used).
    """
    import urllib.parse

    from headroom.proxy.agy_terminator import _upstream_proxy_auth

    parsed = urllib.parse.urlparse("socks5://user:secret@proxy.corp.example:1080")
    assert _upstream_proxy_auth(parsed, None) is None
    assert _upstream_proxy_auth(parsed, "Basic aW5ib3VuZDpzZWNyZXQ=") is None
