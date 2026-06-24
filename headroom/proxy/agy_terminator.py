"""Selective TLS-MITM forward-proxy listener for the agy MITM transport.

Binds to 127.0.0.1 ONLY. Accepts HTTP CONNECT:
- Allowlisted hosts: when a ``dispatch_port`` is configured, ACK the CONNECT
  and byte-splice the raw connection to the in-process hypercorn HTTPS server
  at that loopback port (AgyDispatchServer).  The hypercorn server owns TLS
  termination and ASGI routing.  When no ``dispatch_port`` is set (legacy /
  test path), self-terminate TLS and hand decrypted streams to the caller-
  supplied async ``dispatch`` callback.
- Non-allowlisted hosts: raw bidirectional byte-splice (blind tunnel).
  If HTTPS_PROXY is set, forward CONNECT through that upstream proxy.
  NEVER chain to a loopback address (self-loop guard).

Security invariants:
- Leaf private keys are loaded from anonymous memory (memfd) on Linux and
  never touch the filesystem; on platforms without memfd, a 0600 temp file
  is written and unlinked immediately after load (perms asserted).
- Proxy-Authorization is never logged.
- Listener bind address is 127.0.0.1, never 0.0.0.0.
"""

from __future__ import annotations

import asyncio
import datetime
import ipaddress
import logging
import os
import ssl
import urllib.parse
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.x509 import Certificate
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from headroom.proxy.agy_ca import ensure_root_ca, load_cert_chain_in_memory

logger = logging.getLogger("headroom.proxy.agy_terminator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LEAF_KEY_BITS = 2048
_LEAF_VALIDITY_HOURS = 72
_BIND_HOST = "127.0.0.1"
_CONNECT_TIMEOUT = 10.0
_SPLICE_BUF = 65536

DEFAULT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "daily-cloudcode-pa.googleapis.com",
        "cloudcode-pa.googleapis.com",
    }
)

# Callback type: receives (reader, writer, host, port) for terminated TLS connections.
# Return value is ignored.
DispatchCallback = Callable[
    [asyncio.StreamReader, asyncio.StreamWriter, str, int],
    Awaitable[Any],
]


async def _noop_dispatch(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    host: str,
    port: int,
) -> None:
    """Default no-op dispatch: drain and close."""
    try:
        writer.close()
        await writer.wait_closed()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Leaf certificate minting
# ---------------------------------------------------------------------------


def mint_leaf(
    host: str,
    ca_key: RSAPrivateKey,
    ca_cert: Certificate,
) -> tuple[bytes, bytes]:
    """Mint a leaf TLS certificate for *host* signed by the root CA.

    Parameters
    ----------
    host:
        Hostname for SAN=dNSName entry.
    ca_key:
        Root CA private key (in-memory, never written).
    ca_cert:
        Root CA certificate object.

    Returns
    -------
    (cert_pem, key_pem)
        Both as PEM bytes. Leaf private keys are loaded from anonymous memory
        (memfd) on Linux and never touch the filesystem; on platforms without
        memfd, a 0600 temp file is written and unlinked immediately after load
        (perms asserted).
    """
    leaf_key: RSAPrivateKey = rsa.generate_private_key(
        public_exponent=65537,
        key_size=_LEAF_KEY_BITS,
    )
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    not_after = now + datetime.timedelta(hours=_LEAF_VALIDITY_HOURS)

    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)]))
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(not_after)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(host)]),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=True,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = leaf_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert_pem, key_pem


# ---------------------------------------------------------------------------
# Leaf cert cache
# ---------------------------------------------------------------------------


class _LeafCache:
    """Fixed-bound leaf cert cache keyed by hostname.

    Bound to allowlist size (small dict). Entries are reused within
    validity; expired entries are replaced in-place.
    """

    def __init__(self, max_size: int) -> None:
        self._max = max(max_size, 1)
        # host -> (cert_pem, key_pem, not_after_utc)
        self._cache: dict[str, tuple[bytes, bytes, datetime.datetime]] = {}

    def get_or_mint(
        self,
        host: str,
        ca_key: RSAPrivateKey,
        ca_cert: Certificate,
    ) -> tuple[bytes, bytes]:
        """Return cached leaf or mint a fresh one."""
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        if host in self._cache:
            cert_pem, key_pem, not_after = self._cache[host]
            if now < not_after - datetime.timedelta(minutes=5):
                return cert_pem, key_pem
            # Expired — re-mint in place.
            del self._cache[host]

        if len(self._cache) >= self._max:
            # Evict oldest entry (FIFO; dict preserves insertion order in Python 3.7+).
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        cert_pem, key_pem = mint_leaf(host, ca_key, ca_cert)
        # Parse just-minted cert to get its not_valid_after.
        cert_obj = x509.load_pem_x509_certificate(cert_pem)
        self._cache[host] = (cert_pem, key_pem, cert_obj.not_valid_after_utc)
        logger.debug("event=leaf_minted host=%s", host)
        return cert_pem, key_pem


# ---------------------------------------------------------------------------
# Loopback guard helper
# ---------------------------------------------------------------------------


def _is_loopback(host: str) -> bool:
    """Return True if *host* resolves to a loopback address."""
    if host.lower() == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_loopback
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Byte-splice helpers
# ---------------------------------------------------------------------------


async def _splice_half(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Forward bytes from reader to writer until EOF."""
    try:
        while True:
            data = await reader.read(_SPLICE_BUF)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.write_eof()
        except Exception:  # noqa: BLE001
            pass


async def _blind_splice(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_reader: asyncio.StreamReader,
    target_writer: asyncio.StreamWriter,
) -> None:
    """Bidirectional byte-splice until either side closes.

    Waits until the FIRST half-stream closes (one side EOF'd / connection
    dropped), then cancels the other.  This avoids a hang when the target
    closes after echoing but the client hasn't sent EOF yet.
    """
    t1 = asyncio.create_task(_splice_half(client_reader, target_writer))
    t2 = asyncio.create_task(_splice_half(target_reader, client_writer))
    try:
        done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception:  # noqa: BLE001
        t1.cancel()
        t2.cancel()
        await asyncio.gather(t1, t2, return_exceptions=True)
    finally:
        for w in (client_writer, target_writer):
            try:
                w.close()
                await w.wait_closed()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# CONNECT request parser
# ---------------------------------------------------------------------------


def _parse_connect(line: str) -> tuple[str, int]:
    """Parse 'CONNECT host:port HTTP/1.x' → (host, port). Raises ValueError."""
    parts = line.strip().split()
    if len(parts) < 2 or parts[0].upper() != "CONNECT":
        raise ValueError(f"Not a CONNECT request: {line!r}")
    hostport = parts[1]
    if ":" not in hostport:
        raise ValueError(f"Missing port in CONNECT target: {hostport!r}")
    host, port_str = hostport.rsplit(":", 1)
    return host, int(port_str)


# ---------------------------------------------------------------------------
# Upstream proxy (HTTPS_PROXY) tunnel
# ---------------------------------------------------------------------------


async def _connect_via_upstream_proxy(
    proxy_host: str,
    proxy_port: int,
    target_host: str,
    target_port: int,
    proxy_auth: str | None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a TCP connection through an upstream HTTP proxy using CONNECT."""
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(proxy_host, proxy_port),
        timeout=_CONNECT_TIMEOUT,
    )
    connect_line = (
        f"CONNECT {target_host}:{target_port} HTTP/1.1\r\nHost: {target_host}:{target_port}\r\n"
    )
    if proxy_auth:
        connect_line += f"Proxy-Authorization: {proxy_auth}\r\n"
    connect_line += "\r\n"
    writer.write(connect_line.encode())
    await writer.drain()

    # Read response — look for 200 Connection Established.
    try:
        response_line = await asyncio.wait_for(reader.readline(), timeout=_CONNECT_TIMEOUT)
        if b"200" not in response_line:
            raise OSError(f"Upstream proxy refused CONNECT: {response_line!r}")
        # Drain remaining headers.
        while True:
            hdr = await asyncio.wait_for(reader.readline(), timeout=_CONNECT_TIMEOUT)
            if hdr in (b"\r\n", b"\n", b""):
                break
    except (OSError, asyncio.TimeoutError):
        writer.close()
        raise
    return reader, writer


# ---------------------------------------------------------------------------
# SSL context builder for TLS termination
# ---------------------------------------------------------------------------


def _build_server_ssl_context(cert_pem: bytes, key_pem: bytes) -> ssl.SSLContext:
    """Build an ssl.SSLContext for server-side TLS with ALPN h2+http/1.1."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    load_cert_chain_in_memory(ctx, cert_pem, key_pem)
    ctx.set_alpn_protocols(["h2", "http/1.1"])
    return ctx


# ---------------------------------------------------------------------------
# Main connection handler
# ---------------------------------------------------------------------------


async def _handle_connect(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    allowlist: frozenset[str],
    leaf_cache: _LeafCache,
    ca_key: RSAPrivateKey,
    ca_cert: Certificate,
    dispatch: DispatchCallback,
    dispatch_port: int | None = None,
) -> None:
    """Handle one incoming TCP connection carrying an HTTP CONNECT request."""
    peer = client_writer.get_extra_info("peername", ("?", 0))
    try:
        first_line_bytes = await asyncio.wait_for(
            client_reader.readline(), timeout=_CONNECT_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.debug("event=connect_timeout peer=%s", peer)
        client_writer.close()
        return

    first_line = first_line_bytes.decode("latin-1")
    try:
        target_host, target_port = _parse_connect(first_line)
    except ValueError as exc:
        logger.debug("event=parse_error peer=%s err=%s", peer, exc)
        client_writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        await client_writer.drain()
        client_writer.close()
        return

    # Drain remaining CONNECT request headers.
    proxy_auth: str | None = None
    while True:
        try:
            hdr_bytes = await asyncio.wait_for(client_reader.readline(), timeout=_CONNECT_TIMEOUT)
        except asyncio.TimeoutError:
            logger.debug("event=connect_header_timeout peer=%s", peer)
            client_writer.close()
            return
        if hdr_bytes in (b"\r\n", b"\n", b""):
            break
        hdr = hdr_bytes.decode("latin-1")
        if hdr.lower().startswith("proxy-authorization:"):
            proxy_auth = hdr.split(":", 1)[1].strip()

    logger.debug(
        "event=connect_received peer=%s target=%s:%d allowlisted=%s",
        peer,
        target_host,
        target_port,
        target_host in allowlist,
    )

    if target_host in allowlist:
        await _handle_mitm(
            client_reader,
            client_writer,
            target_host,
            target_port,
            leaf_cache,
            ca_key,
            ca_cert,
            dispatch,
            dispatch_port=dispatch_port,
        )
    else:
        await _handle_blind_tunnel(
            client_reader,
            client_writer,
            target_host,
            target_port,
            proxy_auth,
        )


async def _handle_mitm(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    host: str,
    port: int,
    leaf_cache: _LeafCache,
    ca_key: RSAPrivateKey,
    ca_cert: Certificate,
    dispatch: DispatchCallback,
    dispatch_port: int | None = None,
) -> None:
    """Handle an allowlisted CONNECT: tunnel to hypercorn or TLS-terminate.

    When *dispatch_port* is set (production path with AgyDispatchServer),
    ACK the CONNECT and byte-splice the raw connection to the loopback
    hypercorn HTTPS port — the hypercorn server owns TLS termination, ALPN
    negotiation, and ASGI routing.

    When *dispatch_port* is None (legacy / test path), TLS is terminated
    here and decrypted streams are forwarded to the *dispatch* callback.
    """
    # --- Production path: byte-splice to hypercorn loopback HTTPS port ---
    if dispatch_port is not None:
        # ACK the CONNECT so the client believes the tunnel is up.
        client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await client_writer.drain()
        try:
            dispatch_reader, dispatch_writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", dispatch_port),
                timeout=_CONNECT_TIMEOUT,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            logger.error("event=dispatch_connect_failed port=%d err=%s", dispatch_port, exc)
            try:
                client_writer.close()
            except Exception:  # noqa: BLE001
                pass
            return
        await _blind_splice(client_reader, client_writer, dispatch_reader, dispatch_writer)
        return

    # --- Legacy path: self-terminate TLS + dispatch callback ---
    # Acknowledge the CONNECT.
    client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await client_writer.drain()

    # Mint/reuse leaf cert.
    cert_pem, key_pem = leaf_cache.get_or_mint(host, ca_key, ca_cert)
    ssl_ctx = _build_server_ssl_context(cert_pem, key_pem)

    # Upgrade the existing raw TCP connection to TLS.
    loop = asyncio.get_event_loop()
    transport = client_writer.transport
    raw_sock = transport.get_extra_info("socket")
    if raw_sock is None:
        logger.error("event=mitm_no_socket host=%s", host)
        client_writer.close()
        return

    # Use start_tls on the existing transport.
    # We need to drain and then do TLS upgrade via StreamReader/Writer wrap.
    try:
        tls_reader, tls_writer = await asyncio.wait_for(
            _upgrade_to_tls_server(client_reader, client_writer, ssl_ctx, loop),
            timeout=15.0,
        )
    except (ssl.SSLError, asyncio.TimeoutError, OSError) as exc:
        logger.debug("event=tls_handshake_failed host=%s err=%s", host, exc)
        try:
            client_writer.close()
        except Exception:  # noqa: BLE001
            pass
        return

    logger.debug(
        "event=tls_terminated host=%s alpn=%s",
        host,
        tls_writer.get_extra_info("ssl_object")
        and tls_writer.get_extra_info("ssl_object").selected_alpn_protocol(),
    )

    try:
        await dispatch(tls_reader, tls_writer, host, port)
    except Exception as exc:  # noqa: BLE001
        logger.debug("event=dispatch_error host=%s err=%s", host, exc)
    finally:
        try:
            tls_writer.close()
            await tls_writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


async def _upgrade_to_tls_server(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    ssl_ctx: ssl.SSLContext,
    loop: asyncio.AbstractEventLoop,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Perform server-side TLS handshake on an existing plain connection.

    Uses asyncio.StreamReaderProtocol + start_tls to upgrade in-place.
    """
    transport = writer.transport
    protocol = transport.get_protocol()

    new_transport = await loop.start_tls(
        transport,
        protocol,
        ssl_ctx,
        server_side=True,
    )
    # Rebind writer's transport reference so subsequent writes go through TLS.
    writer._transport = new_transport  # type: ignore[attr-defined]

    return reader, writer


async def _handle_blind_tunnel(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_host: str,
    target_port: int,
    proxy_auth: str | None,
) -> None:
    """Byte-splice tunnel for non-allowlisted targets."""
    upstream_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    try:
        if upstream_proxy:
            parsed = urllib.parse.urlparse(upstream_proxy)
            proxy_host = parsed.hostname or ""
            proxy_port = parsed.port or 443

            # Self-loop guard: never chain through a loopback upstream proxy.
            if _is_loopback(proxy_host):
                # Log host:port only — never the full URL, which may embed
                # user:pass@ credentials.
                logger.warning(
                    "event=self_loop_blocked_proxy proxy=%s:%s",
                    proxy_host,
                    proxy_port,
                )
                client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                await client_writer.drain()
                client_writer.close()
                return

            target_reader, target_writer = await _connect_via_upstream_proxy(
                proxy_host,
                proxy_port,
                target_host,
                target_port,
                proxy_auth,
            )
        else:
            target_reader, target_writer = await asyncio.wait_for(
                asyncio.open_connection(target_host, target_port),
                timeout=_CONNECT_TIMEOUT,
            )
    except (OSError, asyncio.TimeoutError) as exc:
        logger.debug(
            "event=tunnel_connect_failed target=%s:%d err=%s",
            target_host,
            target_port,
            exc,
        )
        client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        await client_writer.drain()
        client_writer.close()
        return

    try:
        client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await client_writer.drain()
    except OSError:
        target_writer.close()
        raise

    await _blind_splice(client_reader, client_writer, target_reader, target_writer)


# ---------------------------------------------------------------------------
# Public API: Terminator server
# ---------------------------------------------------------------------------


class AgyCONNECTTerminator:
    """Asyncio forward-proxy listener implementing selective TLS-MITM.

    Parameters
    ----------
    allowlist:
        Set of hostnames to TLS-terminate. Defaults to ``DEFAULT_ALLOWLIST``.
    dispatch:
        Async callback invoked for each terminated connection (legacy path,
        used when *dispatch_port* is None).
        Signature: ``async (reader, writer, host, port) -> None``.
        Default: no-op.
    dispatch_port:
        When set, allowlisted CONNECT connections are ACK-ed and byte-spliced
        raw to ``127.0.0.1:<dispatch_port>`` (the in-process AgyDispatchServer).
        When None, the old TLS-terminate + dispatch-callback path is used.
    base_dir:
        Headroom state directory (for CA; defaults to ~/.headroom).
        Inject a ``tmp_path``-derived path in tests.
    ca_key / ca_cert:
        Pre-built CA key+cert. When provided, ``base_dir`` is not used for
        CA loading. Intended for tests.
    port:
        Listener port. 0 = OS-assigned ephemeral (default; tests use this).
    host:
        Bind address. Hardcoded to ``127.0.0.1``; parameter exists only for
        testing internal assertion — callers may not override to non-loopback.
    """

    def __init__(
        self,
        allowlist: frozenset[str] | None = None,
        dispatch: DispatchCallback | None = None,
        base_dir: Path | None = None,
        ca_key: RSAPrivateKey | None = None,
        ca_cert: Certificate | None = None,
        port: int = 0,
        dispatch_port: int | None = None,
    ) -> None:
        self._allowlist = allowlist if allowlist is not None else DEFAULT_ALLOWLIST
        self._dispatch: DispatchCallback = dispatch or _noop_dispatch
        self._dispatch_port = dispatch_port
        self._base_dir = base_dir
        self._ca_key_init = ca_key
        self._ca_cert_init = ca_cert
        self._port = port
        self._server: asyncio.Server | None = None
        self._ca_key: RSAPrivateKey | None = None
        self._ca_cert: Certificate | None = None
        self._leaf_cache: _LeafCache | None = None

    async def start(self) -> None:
        """Start the listener. Must be called before :meth:`address`."""
        if self._ca_key_init is not None and self._ca_cert_init is not None:
            self._ca_key = self._ca_key_init
            self._ca_cert = self._ca_cert_init
        else:
            ca_key, ca_cert, _, _ = ensure_root_ca(base_dir=self._base_dir)
            self._ca_key = ca_key
            self._ca_cert = ca_cert

        self._leaf_cache = _LeafCache(max_size=max(len(self._allowlist), 1))

        self._server = await asyncio.start_server(
            self._connection_handler,
            host=_BIND_HOST,
            port=self._port,
        )
        addr = self._server.sockets[0].getsockname()
        logger.info("event=terminator_started address=%s:%d", addr[0], addr[1])

    async def _connection_handler(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        assert self._ca_key is not None
        assert self._ca_cert is not None
        assert self._leaf_cache is not None
        await _handle_connect(
            reader,
            writer,
            self._allowlist,
            self._leaf_cache,
            self._ca_key,
            self._ca_cert,
            self._dispatch,
            dispatch_port=self._dispatch_port,
        )

    @property
    def address(self) -> tuple[str, int]:
        """Return (host, port) the server is bound to. Requires :meth:`start`."""
        if self._server is None:
            raise RuntimeError("Terminator not started")
        sock = self._server.sockets[0]
        host, port = sock.getsockname()[:2]
        return host, port

    async def stop(self) -> None:
        """Stop the listener and wait for all connections to close."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("event=terminator_stopped")

    async def __aenter__(self) -> AgyCONNECTTerminator:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()
