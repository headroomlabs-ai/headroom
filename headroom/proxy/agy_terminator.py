"""Selective TLS-MITM forward-proxy listener for the agy MITM transport.

Binds to 127.0.0.1 ONLY. Accepts HTTP CONNECT:
- Allowlisted hosts: ACK the CONNECT and byte-splice the raw connection to the
  in-process hypercorn HTTPS server at ``dispatch_port`` (AgyDispatchServer).
  The hypercorn server owns TLS termination and ASGI routing.
- Non-allowlisted hosts: raw bidirectional byte-splice (blind tunnel).
  If HTTPS_PROXY is set, forward CONNECT through that upstream proxy, deriving
  Proxy-Authorization from its userinfo when present. An ``https://`` upstream
  proxy is dialled over TLS (``ssl.create_default_context()``, default
  certificate validation, SNI set to the proxy's own hostname, ALPN pinned to
  ``http/1.1``) instead of plaintext on :443.
  NEVER chain to a loopback address (self-loop guard).

Security invariants:
- Leaf private keys are loaded from anonymous memory (memfd) on Linux and
  never touch the filesystem; on platforms without memfd, a 0600 temp file
  is written and unlinked immediately after load (perms asserted).
- Proxy-Authorization is never logged.
- Upstream proxy auth: when HTTPS_PROXY carries `user:pass@` userinfo, it is
  percent-decoded and sent as HTTP Basic auth, only when the proxy scheme is
  `http`/`https` (never to e.g. `socks5://`), and only the URL-derived
  credential is used when the URL carries one (it takes precedence over an
  inbound Proxy-Authorization header). This is sent in cleartext when the
  upstream scheme is `http://` — same as curl, Go and requests do to a plain
  HTTP proxy; the credential already lives in an env var every tool on the
  box can read, and refusing to send it would break most corporate proxies.
- Listener bind address is 127.0.0.1, never 0.0.0.0.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import ipaddress
import logging
import os
import socket
import ssl
import urllib.parse
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.x509 import Certificate
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from headroom.proxy.agy_ca import ensure_root_ca

logger = logging.getLogger("headroom.proxy.agy_terminator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LEAF_KEY_BITS = 2048
_LEAF_VALIDITY_HOURS = 72
_BIND_HOST = "127.0.0.1"
_CONNECT_TIMEOUT = 10.0
_MAX_CONNECT_HEADERS = 100
_MAX_CONNECT_HEADER_BYTES = 32 * 1024
_MAX_CONNECT_LINE_BYTES = 8 * 1024
_SPLICE_BUF = 65536

DEFAULT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "daily-cloudcode-pa.googleapis.com",
        "cloudcode-pa.googleapis.com",
    }
)

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
    """Return True if *host* is a loopback address, IP-literal forms only.

    Catches dotted-quad, IPv4-shorthand (``127.1``, decimal ``2130706433``),
    the unspecified address (``0.0.0.0`` / ``0``, which Linux ``connect()``
    treats as loopback), IPv6 ``::1``, IPv4-mapped IPv6 (``::ffff:127.0.0.1``,
    normalised via ``ipv4_mapped`` so the result does not depend on the
    interpreter version — see CPython gh-103365, fixed in 3.13), and
    ``localhost``/``localhost.`` case-insensitively. Does NOT resolve DNS: a
    hostname that resolves to loopback (e.g. an attacker-controlled
    ``/etc/hosts`` entry) is not detected here and returns False.
    """
    if host.lower() in ("localhost", "localhost."):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        try:
            packed = socket.inet_aton(host)
        except (OSError, UnicodeError):
            return False
        addr = ipaddress.IPv4Address(packed)
    if isinstance(addr, ipaddress.IPv6Address):
        addr = addr.ipv4_mapped or addr
    return bool(addr.is_loopback or addr.is_unspecified)


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
    """Drain both streams on normal EOF; cancel peers on error or shutdown."""
    t1 = asyncio.create_task(_splice_half(client_reader, target_writer))
    t2 = asyncio.create_task(_splice_half(target_reader, client_writer))
    try:
        # Normal EOF is a half-close, so drain the opposite direction. A
        # connection error completes the wait early and cancels its peer.
        done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            task.result()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception:  # noqa: BLE001
        t1.cancel()
        t2.cancel()
        await asyncio.gather(t1, t2, return_exceptions=True)
    finally:
        t1.cancel()
        t2.cancel()
        await asyncio.gather(t1, t2, return_exceptions=True)
        for w in (client_writer, target_writer):
            try:
                w.close()
                await w.wait_closed()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# CONNECT request parser
# ---------------------------------------------------------------------------


def normalize_host(value: str) -> str:
    """Return *value* as a bare, comparable hostname.

    Strips an optional ``:port``, a trailing root dot, and case. Hostnames are
    case-insensitive and ``example.com.`` names the same host as ``example.com``,
    so every exact-match allowlist check in the agy path — CONNECT target, SNI,
    Host header, passthrough base — must compare this form. Otherwise the layers
    disagree: ``CloudCode-PA.googleapis.com`` skips TLS termination and passes
    through uncompressed with no signal that anything was bypassed.
    """
    host = value.strip()
    # Strip a trailing ``:port`` only when it IS a port. ``example.com:abc`` is
    # not a host with a port, so it stays whole and simply fails the allowlist;
    # requiring exactly one colon leaves IPv6 literals such as ``::1`` alone.
    if host.count(":") == 1:
        left, _, right = host.rpartition(":")
        if right.isdigit():
            host = left
    return host.rstrip(".").lower()


def _parse_connect(line: str) -> tuple[str, int]:
    """Parse 'CONNECT host:port HTTP/1.x' → (host, port). Raises ValueError."""
    parts = line.strip().split()
    if len(parts) < 2 or parts[0].upper() != "CONNECT":
        raise ValueError(f"Not a CONNECT request: {line!r}")
    hostport = parts[1]
    if ":" not in hostport:
        raise ValueError(f"Missing port in CONNECT target: {hostport!r}")
    host, port_str = hostport.rsplit(":", 1)
    return normalize_host(host), int(port_str)


# ---------------------------------------------------------------------------
# Upstream proxy (HTTPS_PROXY) tunnel
# ---------------------------------------------------------------------------


async def _connect_via_upstream_proxy(
    proxy_host: str,
    proxy_port: int,
    target_host: str,
    target_port: int,
    proxy_auth: str | None,
    ssl_context: ssl.SSLContext | None = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a TCP connection through an upstream HTTP proxy using CONNECT.

    *ssl_context* is non-None only for an ``https://`` upstream proxy: the
    CONNECT dial itself is then wrapped in TLS to the proxy (SNI ==
    ``proxy_host``, the proxy's own name — the tunnelled payload carries the
    target's TLS handshake and SNI separately, inside the tunnel).
    """
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(
            proxy_host,
            proxy_port,
            ssl=ssl_context,
            server_hostname=proxy_host if ssl_context is not None else None,
        ),
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
# Main connection handler
# ---------------------------------------------------------------------------


async def _handle_connect(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    allowlist: frozenset[str],
    dispatch_port: int,
    self_port: int | None = None,
) -> None:
    """Handle one incoming TCP connection carrying an HTTP CONNECT request.

    *self_port* is the terminator's own listening port, used to refuse a tunnel
    that would loop back into this very listener.
    """
    peer = client_writer.get_extra_info("peername", ("?", 0))
    deadline = asyncio.get_running_loop().time() + _CONNECT_TIMEOUT

    async def reject_headers() -> None:
        client_writer.write(b"HTTP/1.1 431 Request Header Fields Too Large\r\n\r\n")
        await client_writer.drain()
        client_writer.close()

    try:
        first_line_bytes = await asyncio.wait_for(
            client_reader.readline(), timeout=_CONNECT_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.debug("event=connect_timeout peer=%s", peer)
        client_writer.close()
        return
    except ValueError:
        await reject_headers()
        return
    if len(first_line_bytes) > _MAX_CONNECT_LINE_BYTES:
        await reject_headers()
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
    header_count = 0
    header_bytes = len(first_line_bytes)
    while True:
        try:
            hdr_bytes = await asyncio.wait_for(
                client_reader.readline(),
                timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
            )
        except asyncio.TimeoutError:
            logger.debug("event=connect_header_timeout peer=%s", peer)
            client_writer.close()
            return
        except ValueError:
            await reject_headers()
            return
        if hdr_bytes in (b"\r\n", b"\n", b""):
            break
        header_count += 1
        header_bytes += len(hdr_bytes)
        if (
            header_count > _MAX_CONNECT_HEADERS
            or header_bytes > _MAX_CONNECT_HEADER_BYTES
            or len(hdr_bytes) > _MAX_CONNECT_LINE_BYTES
        ):
            await reject_headers()
            return
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
        await _handle_mitm(client_reader, client_writer, dispatch_port)
    else:
        await _handle_blind_tunnel(
            client_reader,
            client_writer,
            target_host,
            target_port,
            proxy_auth,
            self_port=self_port,
        )


async def _handle_mitm(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    dispatch_port: int,
) -> None:
    """Handle an allowlisted CONNECT: ACK it and byte-splice to hypercorn.

    The raw connection is spliced to the loopback hypercorn HTTPS port
    (AgyDispatchServer), which owns TLS termination, ALPN negotiation and
    ASGI routing.
    """
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


async def _resolve_tunnel_target(host: str, port: int, self_port: int | None) -> str:
    """Return an address for *host* that is safe to tunnel to, else raise ValueError.

    The terminator is an unauthenticated CONNECT proxy on loopback for the life of
    an agy session, so anything running as the user can drive it. Two targets must
    never be reachable through it:

    * itself — ``CONNECT 127.0.0.1:<terminator_port>`` makes the terminator tunnel
      into itself, burning two fds per nesting level until they run out;
    * link-local — 169.254.0.0/16 carries the cloud instance-metadata service.

    Other loopback ports are deliberately still reachable: a local process could
    open them directly, so refusing them buys no security and would break plain
    local tunnelling. The check runs on the *resolved* addresses, not the literal
    (a name resolving to 127.0.0.1 is the same self-connect), and the vetted
    address is what we connect to, so no second lookup can substitute another.
    """
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not infos:
        raise ValueError(f"no address for {host}")
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if addr.is_link_local:
            raise ValueError(f"{host} resolves to link-local {addr}")
        if addr.is_loopback and self_port is not None and port == self_port:
            raise ValueError("self-connect to the terminator's own port")
    return str(ipaddress.ip_address(infos[0][4][0]))


def _upstream_proxy_auth(parsed: urllib.parse.ParseResult, inbound: str | None) -> str | None:
    """Resolve the Proxy-Authorization value to send to the upstream proxy.

    Only pure string/URL logic — no socket I/O — so this is unit-testable
    without spinning up a listener.
    """
    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.username:
        return inbound
    password = urllib.parse.unquote(parsed.password or "")
    userinfo = f"{urllib.parse.unquote(parsed.username)}:{password}".encode()
    return "Basic " + base64.b64encode(userinfo).decode()


async def _handle_blind_tunnel(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_host: str,
    target_port: int,
    proxy_auth: str | None,
    self_port: int | None = None,
) -> None:
    """Byte-splice tunnel for non-allowlisted targets.

    When chaining through an upstream HTTPS_PROXY, Proxy-Authorization is
    derived from that URL's userinfo (percent-decoded) and takes precedence
    over any inbound Proxy-Authorization header: the URL is the operator's
    configuration for this specific upstream proxy and is the only source
    that can carry a working credential, since the child process is handed a
    userinfo-free loopback URL and never sends a header of its own. The
    inbound header remains a fallback for a caller that does supply one.

    Trust boundary, chaining branch: the target (``target_host``) is NOT
    vetted here — it is forwarded to the upstream proxy by name, which
    re-resolves it in its own DNS view, so the upstream proxy's own egress
    policy is the actual boundary, not anything checked in this process.
    Only the proxy side is guarded (self-loop, scheme). The self-loop guard
    (``_is_loopback``) covers IP-literal forms only; a DNS name that
    resolves to loopback (e.g. a local ``/etc/hosts`` entry) is not detected
    and is out of scope — see ``_is_loopback``'s docstring.
    """
    upstream_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    try:
        if upstream_proxy:
            parsed = urllib.parse.urlparse(upstream_proxy)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"unsupported upstream proxy scheme: {parsed.scheme}")
            proxy_host = parsed.hostname or ""
            # Default per scheme: a scheme-less-port HTTPS_PROXY like
            # "http://proxy.corp" speaks plain HTTP on :80, not :443.
            proxy_port = parsed.port or (443 if parsed.scheme == "https" else 80)

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

            proxy_ssl_context: ssl.SSLContext | None = None
            if parsed.scheme == "https":
                proxy_ssl_context = ssl.create_default_context()
                proxy_ssl_context.set_alpn_protocols(["http/1.1"])

            target_reader, target_writer = await _connect_via_upstream_proxy(
                proxy_host,
                proxy_port,
                target_host,
                target_port,
                _upstream_proxy_auth(parsed, proxy_auth),
                proxy_ssl_context,
            )
        else:
            target_addr = await asyncio.wait_for(
                _resolve_tunnel_target(target_host, target_port, self_port),
                timeout=_CONNECT_TIMEOUT,
            )
            target_reader, target_writer = await asyncio.wait_for(
                asyncio.open_connection(target_addr, target_port),
                timeout=_CONNECT_TIMEOUT,
            )
    except ValueError as exc:
        logger.warning(
            "event=tunnel_target_refused target=%s:%d reason=%s",
            target_host,
            target_port,
            exc,
        )
        client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
        await client_writer.drain()
        client_writer.close()
        return
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
    dispatch_port:
        Allowlisted CONNECT connections are ACK-ed and byte-spliced raw to
        ``127.0.0.1:<dispatch_port>`` (the in-process AgyDispatchServer).
    allowlist:
        Set of hostnames to TLS-terminate. Defaults to ``DEFAULT_ALLOWLIST``.
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
        dispatch_port: int,
        allowlist: frozenset[str] | None = None,
        base_dir: Path | None = None,
        ca_key: RSAPrivateKey | None = None,
        ca_cert: Certificate | None = None,
        port: int = 0,
    ) -> None:
        self._allowlist = allowlist if allowlist is not None else DEFAULT_ALLOWLIST
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
        await _handle_connect(
            reader,
            writer,
            self._allowlist,
            self._dispatch_port,
            self_port=self.address[1],
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
