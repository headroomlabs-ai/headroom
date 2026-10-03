"""Explain upstream TLS failures in words a user (and their IT team) can act on.

Behind a corporate TLS-inspection gateway every certificate Headroom sees is
re-signed by the company's root. When that root is not trusted, the only thing
that reaches the user today is ``CERTIFICATE_VERIFY_FAILED`` wrapped in a 502.
This module turns that into: *who* signed the certificate (e.g. "Zscaler
Intermediate Root CA"), what that means, and the exact fix.

It is used in two places:

* the proxy's upstream error paths, so the agent's error line names the cause
  (:func:`describe_upstream_failure`), and
* ``headroom doctor --network`` (:func:`probe_endpoint`).

To learn the issuer, the probe performs a second, *unverified* TLS handshake
purely to read the presented certificate. Nothing is sent over that
connection: it is closed straight after the handshake.
"""

from __future__ import annotations

import base64
import logging
import os
import socket
import ssl
import tempfile
import threading
import time
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger("headroom.proxy")

DOCS_URL = "https://docs.headroomlabs.ai/docs/corporate-networks"

# Issuer substrings (lower-cased) of common TLS-inspection products. Matching is
# on the issuer's organization / common name, so a customer-uploaded
# intermediate ("Acme Corp Zscaler SubCA") still resolves to the product.
_INSPECTION_VENDORS: tuple[tuple[str, str], ...] = (
    ("zscaler", "Zscaler"),
    ("netskope", "Netskope"),
    ("palo alto", "Palo Alto Networks"),
    ("globalprotect", "Palo Alto Networks"),
    ("prisma", "Palo Alto Networks"),
    ("cisco umbrella", "Cisco Umbrella"),
    ("cisco secure access", "Cisco Secure Access"),
    ("fortinet", "Fortinet"),
    ("fortigate", "Fortinet"),
    ("forcepoint", "Forcepoint"),
    ("websense", "Forcepoint"),
    ("blue coat", "Symantec / Broadcom"),
    ("bluecoat", "Symantec / Broadcom"),
    ("symantec", "Symantec / Broadcom"),
    ("cloudflare gateway", "Cloudflare Gateway"),
    ("cloudflare for teams", "Cloudflare Gateway"),
    ("iboss", "iboss"),
    ("check point", "Check Point"),
    ("checkpoint", "Check Point"),
    ("sophos", "Sophos"),
    ("skyhigh", "Skyhigh Security"),
    ("mcafee web gateway", "Skyhigh Security"),
    ("menlo security", "Menlo Security"),
    ("barracuda", "Barracuda"),
)

# ``_ssl.Certificate.public_bytes`` format flag (Python 3.13+ chain API).
_ENCODING_DER = getattr(ssl._ssl, "ENCODING_DER", 1)  # type: ignore[attr-defined]

_PROBE_TIMEOUT_S = 4.0
_PROBE_CACHE_TTL_S = 300.0
_probe_cache: dict[tuple[str, int, bool], tuple[float, ChainInfo]] = {}
_probe_lock = threading.Lock()
# One in-flight probe per (host, port): a burst of failing requests waits for
# the first handshake instead of each opening its own.
_probe_inflight: dict[tuple[str, int, bool], threading.Lock] = {}
_logged_hosts: set[str] = set()


@dataclass
class ChainInfo:
    """What the network presented for ``host`` — no trust decision implied."""

    host: str
    port: int
    reachable: bool
    issuer: str | None = None
    subject: str | None = None
    inspection_vendor: str | None = None
    via_proxy: str | None = None
    error: str | None = None
    chain_issuers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Exception inspection
# ---------------------------------------------------------------------------


def find_cert_verify_error(exc: BaseException | None) -> ssl.SSLCertVerificationError | None:
    """Return the certificate-verification error buried in ``exc``'s chain.

    httpx wraps httpcore, which wraps the ``ssl`` error, and websockets
    re-raises it directly; walk ``__cause__``/``__context__`` (bounded) and the
    exception args to find it.
    """
    seen: set[int] = set()
    stack: list[BaseException] = [exc] if exc is not None else []
    while stack:
        cur = stack.pop()
        if id(cur) in seen or len(seen) > 32:
            continue
        seen.add(id(cur))
        if isinstance(cur, ssl.SSLCertVerificationError):
            return cur
        for nxt in (cur.__cause__, cur.__context__):
            if isinstance(nxt, BaseException):
                stack.append(nxt)
        for arg in getattr(cur, "args", ()):
            if isinstance(arg, BaseException):
                stack.append(arg)
    return None


def _looks_like_cert_failure(exc: BaseException) -> bool:
    # Some stacks (httpcore on certain paths) stringify the ssl error into a
    # plain ConnectError without chaining it.
    text = str(exc)
    return "CERTIFICATE_VERIFY_FAILED" in text or "certificate verify failed" in text


# ---------------------------------------------------------------------------
# Presented-chain probe
# ---------------------------------------------------------------------------


def identify_inspection_vendor(*names: str | None) -> str | None:
    """Map certificate issuer / subject strings to a TLS-inspection product."""
    for name in names:
        if not name:
            continue
        lowered = name.lower()
        for needle, vendor in _INSPECTION_VENDORS:
            if needle in lowered:
                return vendor
    return None


def _name_to_str(name: Any) -> str | None:
    """Render a decoded ``getpeercert()``-style name tuple as ``O=..., CN=...``."""
    if not name:
        return None
    parts: list[str] = []
    wanted = {"organizationName": "O", "organizationalUnitName": "OU", "commonName": "CN"}
    for rdn in name:
        for key, value in rdn:
            if key in wanted:
                parts.append(f"{wanted[key]}={value}")
    return ", ".join(parts) or None


def _decode_der(der: bytes) -> dict[str, Any]:
    """Decode a DER certificate into ``getpeercert()`` form without verifying it.

    Uses ``cryptography`` when installed, else CPython's own decoder (the one
    ``ssl`` uses internally), which needs a PEM file on disk.
    """
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID

        cert = x509.load_der_x509_certificate(der)

        def _rdns(n: Any) -> tuple[tuple[tuple[str, str], ...], ...]:
            mapping = {
                NameOID.ORGANIZATION_NAME: "organizationName",
                NameOID.ORGANIZATIONAL_UNIT_NAME: "organizationalUnitName",
                NameOID.COMMON_NAME: "commonName",
            }
            return tuple(((mapping[a.oid], str(a.value)),) for a in n if a.oid in mapping)

        return {"issuer": _rdns(cert.issuer), "subject": _rdns(cert.subject)}
    except Exception:
        pass
    pem = ssl.DER_cert_to_PEM_cert(der)
    fd, path = tempfile.mkstemp(suffix=".pem")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(pem)
        decoded: dict[str, Any] = ssl._ssl._test_decode_cert(path)  # type: ignore[attr-defined]
        return decoded
    except Exception:
        return {}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _proxy_for(host: str) -> str | None:
    """The HTTPS proxy the upstream client would use for ``host``, if any.

    Environment only, like httpx's ``trust_env``: the OS proxy settings that
    ``urllib.request.getproxies`` also reads on macOS/Windows are not what the
    proxy's requests use, so probing through them would diagnose another path.
    """
    try:
        if urllib.request.proxy_bypass_environment(host):  # type: ignore[attr-defined]
            return None
        env = urllib.request.getproxies_environment()
    except Exception:
        return None
    return env.get("https") or env.get("all") or None


def _open_tunnel(host: str, port: int, proxy_url: str, timeout: float) -> socket.socket:
    """Open a TCP connection to ``host:port`` through an HTTP CONNECT proxy."""
    parts = urlsplit(proxy_url if "://" in proxy_url else f"http://{proxy_url}")
    if parts.scheme not in ("http", ""):
        raise OSError(f"unsupported proxy scheme {parts.scheme!r} for the TLS probe")
    sock = socket.create_connection((parts.hostname or "", parts.port or 8080), timeout=timeout)
    request = f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n"
    if parts.username:
        token = base64.b64encode(f"{parts.username}:{parts.password or ''}".encode()).decode()
        request += f"Proxy-Authorization: Basic {token}\r\n"
    sock.sendall((request + "\r\n").encode())
    reply = b""
    while b"\r\n\r\n" not in reply and len(reply) < 8192:
        chunk = sock.recv(1024)
        if not chunk:
            break
        reply += chunk
    status_line = reply.split(b"\r\n", 1)[0].decode("latin-1", "replace")
    if " 200" not in status_line:
        sock.close()
        raise OSError(f"proxy refused CONNECT: {status_line or 'no reply'}")
    return sock


def _public_address(host: str, port: int, timeout: float) -> tuple[str, int] | None:
    """Resolve ``host`` once and return a public address, or None if it is not public.

    The request-path probe can be pointed at a per-request upstream, so it must
    not become a way to read the certificate of an internal service. Returning
    the checked address (instead of re-resolving) keeps DNS rebinding out too.
    """
    import ipaddress

    # No timeout knob exists for getaddrinfo; the caller already runs off the
    # event loop, and the process-wide socket default must not be touched here.
    del timeout
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    for _family, _type, _proto, _canon, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if not ip.is_global:
            return None
    return (str(infos[0][4][0]), port) if infos else None


def probe_presented_chain(
    host: str,
    port: int = 443,
    *,
    timeout: float = _PROBE_TIMEOUT_S,
    use_cache: bool = True,
    allow_private: bool = False,
) -> ChainInfo:
    """Handshake with ``host`` *without verification* and report who signed it.

    Goes through the same explicit HTTPS proxy the environment configures, so an
    explicit-proxy deployment sees the same re-signed chain the proxy would.
    Results are cached for a few minutes so a burst of failing requests costs a
    single probe. Hosts resolving to private, loopback or link-local addresses
    are skipped unless ``allow_private`` (``doctor``, which the user runs on
    purpose, sets it).
    """
    # allow_private is part of the key so a doctor/test probe of a private host
    # can never be served to the request path.
    key = (host, port, allow_private)
    if not use_cache:
        return _probe_uncached(host, port, timeout=timeout, allow_private=allow_private)
    with _probe_lock:
        gate = _probe_inflight.setdefault(key, threading.Lock())
    with gate:
        with _probe_lock:
            cached = _probe_cache.get(key)
        if cached and time.monotonic() - cached[0] < _PROBE_CACHE_TTL_S:
            return cached[1]
        info = _probe_uncached(host, port, timeout=timeout, allow_private=allow_private)
        with _probe_lock:
            _probe_cache[key] = (time.monotonic(), info)
        return info


def _probe_uncached(host: str, port: int, *, timeout: float, allow_private: bool) -> ChainInfo:
    proxy = _proxy_for(host)
    info = ChainInfo(host=host, port=port, reachable=False, via_proxy=proxy)
    # Certificate-inspection handshake only: it reads the presented chain and
    # closes without sending a byte, so it deliberately skips verification (the
    # chain is what we are diagnosing). It still refuses TLS < 1.2, and it only
    # connects through the public-address / env-proxy CONNECT guard below.
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        if proxy:
            raw = _open_tunnel(host, port, proxy, timeout)
        else:
            target = (host, port) if allow_private else _public_address(host, port, timeout)
            if target is None:
                info.error = "skipped: host resolves to a non-public address"
                return info
            raw = socket.create_connection(target, timeout=timeout)
        with ctx.wrap_socket(raw, server_hostname=host) as tls:
            info.reachable = True
            ders: list[bytes] = []
            get_chain = getattr(tls, "get_unverified_chain", None)  # Python 3.13+
            if get_chain is not None:
                try:
                    ders = [
                        c if isinstance(c, bytes) else c.public_bytes(_ENCODING_DER)
                        for c in get_chain() or []
                    ]
                except Exception:
                    ders = []
            if not ders:
                leaf = tls.getpeercert(binary_form=True)
                ders = [leaf] if leaf else []
        decoded = [_decode_der(d) for d in ders]
        if decoded:
            info.subject = _name_to_str(decoded[0].get("subject"))
            info.issuer = _name_to_str(decoded[0].get("issuer"))
            info.chain_issuers = [s for d in decoded if (s := _name_to_str(d.get("issuer")))]
        info.inspection_vendor = identify_inspection_vendor(info.issuer, *info.chain_issuers)
    except Exception as exc:
        info.error = f"{type(exc).__name__}: {exc}"
    return info


# ---------------------------------------------------------------------------
# User-facing explanation
# ---------------------------------------------------------------------------


def _fix_lines(vendor: str | None) -> list[str]:
    who = f"{vendor} " if vendor else "your network's "
    return [
        f"Headroom trusts the operating system's certificate store, where IT normally "
        f"installs the {who}root certificate. If your browser and Claude Code work but "
        f"Headroom does not, that root is not in this machine's store.",
        "Fix (pick one): ask IT to install the root in the OS certificate store; or export "
        "it to a PEM file and set HEADROOM_CA_BUNDLE=/path/to/root.pem (NODE_EXTRA_CA_CERTS "
        "also works); or ask IT to exempt the provider domains from TLS inspection.",
        "Run `headroom doctor --network` for a full report.",
    ]


def explain_cert_failure(host: str, verify_message: str | None, info: ChainInfo | None) -> str:
    """One paragraph a user can act on (and forward to IT)."""
    reason = verify_message or "certificate verify failed"
    head = f"Headroom could not verify the TLS certificate for {host} ({reason})."
    if info is not None and info.issuer:
        if info.inspection_vendor:
            head += (
                f" The certificate was issued by '{info.issuer}': {info.inspection_vendor} "
                f"is inspecting this connection."
            )
        else:
            head += f" The certificate was issued by '{info.issuer}'."
    return " ".join([head, *_fix_lines(info.inspection_vendor if info else None), DOCS_URL])


def describe_upstream_failure(
    exc: BaseException, url: str | None, *, probe: bool = True, allow_private: bool = False
) -> str | None:
    """Actionable message for a certificate-verification failure, else ``None``.

    ``None`` means "not a TLS trust problem" and the caller keeps its existing
    message. Safe to call from a request's error path: the probe is bounded by a
    short timeout and cached per host. Blocking — async callers should use
    :func:`describe_upstream_failure_async`.
    """
    cert_err = find_cert_verify_error(exc)
    if cert_err is None and not _looks_like_cert_failure(exc):
        return None
    parts = urlsplit(url or "")
    host = parts.hostname or "the upstream API"
    port = parts.port or (443 if parts.scheme in ("https", "wss", "") else 80)
    info = (
        probe_presented_chain(host, port, allow_private=allow_private)
        if probe and parts.hostname
        else None
    )
    verify_message = getattr(cert_err, "verify_message", None) if cert_err else None
    message = explain_cert_failure(host, verify_message, info)
    if host not in _logged_hosts:
        _logged_hosts.add(host)
        logger.error("event=upstream_tls_untrusted host=%s detail=%s", host, message)
    return message


async def describe_upstream_failure_async(
    exc: BaseException, url: str | None, *, probe: bool = True
) -> str | None:
    """:func:`describe_upstream_failure` off the event loop."""
    import asyncio

    if find_cert_verify_error(exc) is None and not _looks_like_cert_failure(exc):
        return None
    return await asyncio.to_thread(describe_upstream_failure, exc, url, probe=probe)


# ---------------------------------------------------------------------------
# Endpoint check for `headroom doctor --network`
# ---------------------------------------------------------------------------


@dataclass
class EndpointReport:
    """Result of checking one endpoint Headroom needs."""

    name: str
    url: str
    ok: bool
    chain: ChainInfo
    verified_with: str | None = None
    error: str | None = None
    status: int | None = None
    block_page: bool = False
    elapsed_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["chain"] = self.chain.to_dict()
        return d


def _looks_like_block_page(status: int, content_type: str, body: bytes) -> bool:
    """A TLS-inspection gateway's interstitial rather than the real API."""
    if "text/html" not in content_type.lower():
        return False
    text = body[:8192].decode("utf-8", "replace").lower()
    if identify_inspection_vendor(text) is not None:
        return True
    return status in (403, 407, 451) and ("access denied" in text or "blocked" in text)


def probe_endpoint(name: str, url: str, *, timeout: float = 8.0) -> EndpointReport:
    """Check ``url`` end to end with Headroom's real trust policy.

    Reports the presented chain (and any inspection vendor), whether the chain
    verifies under the policy Headroom will use, and whether the response is a
    gateway block page instead of the service.
    """
    import httpx

    from headroom.proxy.ssl_context import build_httpx_verify

    parts = urlsplit(url)
    chain = probe_presented_chain(
        parts.hostname or "", parts.port or 443, use_cache=False, allow_private=True
    )
    started = time.monotonic()
    try:
        with httpx.Client(verify=build_httpx_verify(), timeout=timeout) as client:
            resp = client.get(url, headers={"User-Agent": "headroom-doctor"})
        elapsed = (time.monotonic() - started) * 1000
        block = _looks_like_block_page(
            resp.status_code, resp.headers.get("content-type", ""), resp.content
        )
        return EndpointReport(
            name=name,
            url=url,
            ok=not block,
            chain=chain,
            verified_with="headroom trust policy",
            status=resp.status_code,
            block_page=block,
            elapsed_ms=round(elapsed, 1),
            error="gateway returned a block/warning page instead of the service" if block else None,
        )
    except Exception as exc:
        cert_err = find_cert_verify_error(exc)
        if cert_err is not None or _looks_like_cert_failure(exc):
            error = explain_cert_failure(
                parts.hostname or url, getattr(cert_err, "verify_message", None), chain
            )
        else:
            error = f"{type(exc).__name__}: {exc}"
        return EndpointReport(name=name, url=url, ok=False, chain=chain, error=error)


def loopback_no_proxy_gap(environ: Mapping[str, str]) -> str | None:
    """Name the proxy var that would capture loopback traffic, if NO_PROXY misses it.

    With ``HTTP(S)_PROXY`` set and no loopback entry in ``NO_PROXY``, an agent's
    requests to ``http://127.0.0.1:8787`` can be sent to the corporate proxy,
    which cannot reach this laptop.
    """
    proxy_var = next(
        (
            v
            for v in (
                "HTTP_PROXY",
                "http_proxy",
                "HTTPS_PROXY",
                "https_proxy",
                "ALL_PROXY",
                "all_proxy",
            )
            if environ.get(v)
        ),
        None,
    )
    if proxy_var is None:
        return None
    no_proxy = ",".join(environ.get(v, "") for v in ("NO_PROXY", "no_proxy")).lower()
    entries = {e.strip() for e in no_proxy.split(",") if e.strip()}
    if "*" in entries or {"127.0.0.1", "localhost"} <= entries:
        return None
    return proxy_var
