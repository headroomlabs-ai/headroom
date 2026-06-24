"""Root CA lifecycle + combined trust bundle for the agy MITM transport.

Process-scoped: the CA is generated once and persisted under
``~/.headroom/ca/`` (or an injectable base dir for tests). The combined
bundle (system CAs + headroom root CA + filtered corporate CAs) is written
to ``~/.headroom/combined-ca-bundle.pem`` with strict permissions and is
intended for injection into the wrapped agy process via environment
variables (CACERT_PATH / SSL_CERT_FILE / NODE_EXTRA_CA_CERTS).

Security invariants (enforced by assertion):
- CA private key: 0600, parent dir: 0700.
- Combined bundle: 0600, parent dir: 0700.
- CA is NEVER written to any OS trust-store path.
- Only PEM objects with basicConstraints CA:TRUE are included from
  corporate CA files (per-object parse-then-filter).
"""

from __future__ import annotations

import datetime
import logging
import os
import ssl
import stat
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.x509 import Certificate
from cryptography.x509.oid import NameOID

logger = logging.getLogger("headroom.proxy.agy_ca")

# CA validity: 10 years; regeneration triggers when less than 30 days remain.
_CA_VALIDITY_DAYS = 3650
_REGEN_THRESHOLD_DAYS = 30

# Key size for the root CA.
_RSA_KEY_BITS = 4096

# Well-known OS trust store paths — CA must never be written here.
_OS_TRUST_PATHS: tuple[str, ...] = (
    "/etc/ssl/certs",
    "/etc/pki/ca-trust",
    "/usr/local/share/ca-certificates",
    "/etc/ca-certificates",
    "/usr/share/ca-certificates",
    "/System/Library/Keychains",
    "/Library/Keychains",
)

# Candidate system CA bundle paths (ordered by prevalence).
_SYSTEM_BUNDLE_CANDIDATES: tuple[str, ...] = (
    "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu/Alpine
    "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL/CentOS/Fedora
    "/etc/ssl/ca-bundle.pem",  # openSUSE
    "/usr/share/ssl/certs/ca-bundle.crt",  # legacy RHEL
    "/usr/local/etc/openssl/cert.pem",  # macOS Homebrew OpenSSL
    "/etc/ssl/cert.pem",  # macOS system / BSDs
    "/usr/local/share/certs/ca-root-nss.crt",  # FreeBSD
    "/etc/pki/tls/cacert.pem",  # older RHEL
)

# Environment variables that may point at a corporate CA bundle.
_CORP_CA_ENV_VARS: tuple[str, ...] = ("SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS")

# File names under the CA directory.
_CA_KEY_NAME = "ca.key"
_CA_CERT_NAME = "ca.crt"
_BUNDLE_NAME = "combined-ca-bundle.pem"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_perms(path: Path, expected_mode: int) -> None:
    """Raise PermissionError if *path* does not have exactly *expected_mode* bits.

    No-op on non-POSIX platforms (Windows) where mode bits are not meaningful.
    """
    if os.name != "posix":
        return
    actual = stat.S_IMODE(path.stat().st_mode)
    if actual != expected_mode:
        raise PermissionError(
            f"Permission check failed for {path}: expected {oct(expected_mode)}, got {oct(actual)}"
        )


def _secure_dir(path: Path) -> None:
    """Create *path* with 0700 if absent; enforce 0700 on return.

    ``parents=True`` only applies the mode to the leaf directory on some
    platforms — intermediate parents get the umask-filtered mode.  We
    therefore chmod the leaf explicitly after mkdir so pre-existing or
    newly-created paths are always corrected to 0700.
    """
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    _assert_perms(path, 0o700)


def _write_secure(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically with 0600; assert afterwards.

    The temp file is created with mode 0o600 from the start via ``os.open``
    so there is never a world-readable window while data is on disk.

    ``O_BINARY`` (a no-op 0 on POSIX, defined only on Windows) prevents the
    Windows text-mode ``\n``->``\r\n`` translation that would otherwise corrupt
    the PEM bytes written here (CA key/cert and the combined trust bundle).
    """
    tmp = path.with_suffix(".tmp")
    fd = os.open(
        str(tmp),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    _assert_perms(path, 0o600)


def _not_in_os_trust(path: Path) -> None:
    """Raise RuntimeError if *path* resides under any known OS trust location."""
    resolved = str(path.resolve())
    for trust_path in _OS_TRUST_PATHS:
        if resolved.startswith(trust_path):
            raise RuntimeError(
                f"CA file {path} resolves to {resolved}, which is inside OS trust path {trust_path}"
            )


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.timezone.utc)


# ---------------------------------------------------------------------------
# CA generation
# ---------------------------------------------------------------------------


def _generate_root_ca() -> tuple[RSAPrivateKey, Certificate]:
    """Generate a new RSA root CA key + self-signed certificate."""
    key: RSAPrivateKey = rsa.generate_private_key(
        public_exponent=65537,
        key_size=_RSA_KEY_BITS,
    )
    now = _now_utc()
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "Headroom Local CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Headroom MITM"),
        ]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=_CA_VALIDITY_DAYS))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=0),
            critical=True,
        )
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
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    return key, cert


def _is_ca_cert(cert: Certificate) -> bool:
    """Return True iff the certificate has basicConstraints CA:TRUE."""
    try:
        bc = cert.extensions.get_extension_for_class(x509.BasicConstraints)
        return bool(bc.value.ca)
    except x509.ExtensionNotFound:
        return False


def _cert_near_expiry(cert: Certificate) -> bool:
    """Return True if the certificate expires within the regen threshold."""
    threshold = _now_utc() + datetime.timedelta(days=_REGEN_THRESHOLD_DAYS)
    return bool(cert.not_valid_after_utc <= threshold)


# ---------------------------------------------------------------------------
# Bundle helpers
# ---------------------------------------------------------------------------


def _detect_system_bundle() -> Path:
    """Return path to the system CA bundle; raise RuntimeError if not found."""
    for candidate in _SYSTEM_BUNDLE_CANDIDATES:
        p = Path(candidate)
        if p.is_file() and p.stat().st_size > 0:
            logger.debug("event=system_bundle_found path=%s", p)
            return p
    raise RuntimeError(
        "No system CA bundle found. Searched: " + ", ".join(_SYSTEM_BUNDLE_CANDIDATES)
    )


def _windows_trust_pem() -> bytes:
    """Collect CA:TRUE certs from the Windows system trust stores as PEM bytes.

    Windows has no single on-disk CA bundle file, so ``_SYSTEM_BUNDLE_CANDIDATES``
    never matches there. ``ssl.enum_certificates`` (Windows-only) enumerates the
    ROOT and CA stores but returns *all* certs including leaf certs, so the
    result is run through the same ``_parse_ca_certs_from_pem`` CA:TRUE filter
    used for corporate bundles — never trust a non-CA cert as an anchor.
    """
    pem = b""
    for store in ("ROOT", "CA"):
        for der, _enc, _trust in ssl.enum_certificates(store):  # type: ignore[attr-defined,unused-ignore]
            pem += ssl.DER_cert_to_PEM_cert(der).encode("ascii")
    return b"".join(_parse_ca_certs_from_pem(pem))


def _system_trust_pem() -> tuple[bytes, str]:
    """Return ``(system trust PEM bytes, source label)`` for this platform.

    POSIX/macOS read the detected on-disk bundle; Windows enumerates the
    system trust stores via stdlib ``ssl`` (no certifi dependency).
    """
    if sys.platform == "win32":
        return _windows_trust_pem(), "windows-cert-store"
    path = _detect_system_bundle()
    return path.read_bytes(), str(path)


def _parse_ca_certs_from_pem(pem_data: bytes) -> list[bytes]:
    """Parse a multi-cert PEM file, returning PEM bytes for CA:TRUE certs only."""
    results: list[bytes] = []
    # Split on BEGIN CERTIFICATE boundaries; preserve header+body per cert.
    parts = pem_data.split(b"-----BEGIN CERTIFICATE-----")
    for part in parts[1:]:  # skip leading empty fragment
        pem_block = b"-----BEGIN CERTIFICATE-----" + part
        # Trim trailing noise after END CERTIFICATE.
        end_marker = b"-----END CERTIFICATE-----"
        end_idx = pem_block.find(end_marker)
        if end_idx == -1:
            continue
        pem_block = pem_block[: end_idx + len(end_marker)] + b"\n"
        try:
            cert = x509.load_pem_x509_certificate(pem_block)
        except Exception:  # noqa: BLE001
            logger.debug("event=pem_parse_skip reason=invalid_cert")
            continue
        if _is_ca_cert(cert):
            results.append(pem_block)
        else:
            logger.debug(
                "event=corp_ca_filter_drop subject=%s reason=not_ca",
                cert.subject.rfc4514_string(),
            )
    return results


def _collect_corporate_ca_pems(env_vars: Sequence[str] = _CORP_CA_ENV_VARS) -> list[bytes]:
    """
    Collect CA-only PEM blocks from any pre-existing corporate CA env vars.

    Reads SSL_CERT_FILE and NODE_EXTRA_CA_CERTS (if set and pointing at a
    file), parses each PEM object, and retains only those with
    basicConstraints CA:TRUE.
    """
    ca_pems: list[bytes] = []
    for var in env_vars:
        path_str = os.environ.get(var)
        if not path_str:
            continue
        p = Path(path_str)
        if not p.is_file():
            logger.warning("event=corp_ca_env_missing var=%s path=%r", var, path_str)
            continue
        data = p.read_bytes()
        filtered = _parse_ca_certs_from_pem(data)
        logger.info(
            "event=corp_ca_loaded var=%s path=%s ca_count=%d",
            var,
            path_str,
            len(filtered),
        )
        ca_pems.extend(filtered)
    return ca_pems


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_root_ca(
    base_dir: Path | None = None,
) -> tuple[RSAPrivateKey, Certificate, Path, Path]:
    """Ensure the headroom root CA exists and is valid; regenerate if expired.

    Parameters
    ----------
    base_dir:
        Root of the headroom state directory. Defaults to ``~/.headroom``.
        Tests must pass a ``tmp_path``-derived value to avoid touching the
        real home directory.

    Returns
    -------
    (private_key, certificate, key_path, cert_path)
        The in-memory key + cert objects and their on-disk paths.
    """
    if base_dir is None:
        base_dir = Path.home() / ".headroom"

    _secure_dir(base_dir)
    ca_dir = base_dir / "ca"
    _secure_dir(ca_dir)
    _not_in_os_trust(ca_dir)

    key_path = ca_dir / _CA_KEY_NAME
    cert_path = ca_dir / _CA_CERT_NAME

    # --- load existing if present ---
    if key_path.exists() and cert_path.exists():
        _assert_perms(key_path, 0o600)
        _assert_perms(cert_path, 0o600)
        try:
            existing_cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        except Exception as exc:
            logger.warning("event=ca_load_failed reason=%s; regenerating", exc)
            existing_cert = None

        if existing_cert is not None and not _cert_near_expiry(existing_cert):
            try:
                key_bytes = key_path.read_bytes()
                existing_key = serialization.load_pem_private_key(key_bytes, password=None)
                logger.info("event=ca_reused path=%s", cert_path)
                return existing_key, existing_cert, key_path, cert_path  # type: ignore[return-value]
            except Exception as exc:
                logger.warning("event=ca_key_load_failed reason=%s; regenerating", exc)

        # Regenerate — delete stale artifacts.
        logger.info("event=ca_regenerate reason=expired_or_corrupt path=%s", cert_path)
        _delete_stale_artifacts(base_dir)

    # --- generate fresh CA ---
    key, cert = _generate_root_ca()

    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)

    _write_secure(key_path, key_pem)
    _write_secure(cert_path, cert_pem)
    _assert_perms(ca_dir, 0o700)
    _not_in_os_trust(key_path)
    _not_in_os_trust(cert_path)
    logger.info("event=ca_generated path=%s", cert_path)
    return key, cert, key_path, cert_path


def _delete_stale_artifacts(base_dir: Path) -> None:
    """Remove old combined bundle and any leaf certs on CA regeneration."""
    bundle = base_dir / _BUNDLE_NAME
    if bundle.exists():
        bundle.unlink()
        logger.info("event=stale_bundle_deleted path=%s", bundle)
    # Leaf certs would live under base_dir/leaves/ (T8). Delete the dir if present.
    leaves_dir = base_dir / "leaves"
    if leaves_dir.is_dir():
        import shutil

        shutil.rmtree(leaves_dir)
        logger.info("event=stale_leaves_deleted path=%s", leaves_dir)


def build_combined_bundle(
    base_dir: Path | None = None,
    corp_env_vars: Sequence[str] = _CORP_CA_ENV_VARS,
) -> Path:
    """Build (or rebuild) the combined CA trust bundle.

    Combines:
    1. System CA bundle (detected cross-distro; fail-fast if absent).
    2. Headroom root CA certificate.
    3. Any pre-existing corporate CAs from env (CA:TRUE-only, per-object filter).

    Writes to ``<base_dir>/combined-ca-bundle.pem`` with 0600 perms.
    Parent dir is asserted 0700.

    Parameters
    ----------
    base_dir:
        Headroom state directory. Defaults to ``~/.headroom``.
    corp_env_vars:
        Environment variable names to scan for corporate CA files.
        Override in tests to inject fixture paths without touching the env.

    Returns
    -------
    Path to the combined bundle.
    """
    if base_dir is None:
        base_dir = Path.home() / ".headroom"

    _secure_dir(base_dir)

    system_pem, system_source = _system_trust_pem()

    _, ca_cert, _, ca_cert_path = ensure_root_ca(base_dir)
    headroom_pem = ca_cert.public_bytes(serialization.Encoding.PEM)

    corp_pems = _collect_corporate_ca_pems(corp_env_vars)

    combined = system_pem
    if not combined.endswith(b"\n"):
        combined += b"\n"
    combined += headroom_pem
    for pem in corp_pems:
        combined += pem

    bundle_path = base_dir / _BUNDLE_NAME
    _write_secure(bundle_path, combined)
    _assert_perms(bundle_path, 0o600)
    _assert_perms(base_dir, 0o700)
    _not_in_os_trust(bundle_path)

    logger.info(
        "event=bundle_written path=%s system=%s corp_ca_count=%d",
        bundle_path,
        system_source,
        len(corp_pems),
    )
    return bundle_path


# ---------------------------------------------------------------------------
# In-memory leaf cert/key loader
# ---------------------------------------------------------------------------


def load_cert_chain_in_memory(
    ctx: ssl.SSLContext,
    cert_pem: bytes,
    key_pem: bytes,
) -> None:
    """Load *cert_pem* + *key_pem* into *ctx* without writing a persistent key file.

    Leaf private keys are loaded from anonymous memory (memfd) on Linux and
    never touch the filesystem; on platforms without memfd, a 0600 temp file
    is written and unlinked immediately after load (perms asserted).

    Primary path (Linux, ``os.memfd_create`` available):
        An anonymous, unnamed in-kernel file descriptor is created via
        ``memfd_create``.  The combined ``cert_pem + key_pem`` PEM is written
        into it (looping on ``os.write`` to handle short-writes).
        ``load_cert_chain`` reads it through ``/proc/self/fd/{fd}``; the fd is
        closed in a ``finally`` block *after* the load (the ``/proc`` path dies
        the moment the fd is closed).

    Fallback path (memfd absent or ``/proc`` unusable):
        ``tempfile.mkstemp`` creates a 0600 temp file.  The combined PEM is
        written in full (loop on ``os.write``).  ``_assert_perms`` validates
        the 0600 mode (fail-loud; no silent chmod since mkstemp already yields
        0600).  ``load_cert_chain`` is called; ``os.unlink`` removes the file
        in a ``finally`` block even if ``load_cert_chain`` raises.  Residual
        risk: a hard crash (SIGKILL/OOM) during the brief load window could
        orphan a 0600 temp until OS temp cleanup.  This fallback only runs on
        platforms without ``memfd_create``; Linux never writes the key to disk.

    Parameters
    ----------
    ctx:
        Target ``ssl.SSLContext`` (must be server-side, ``PROTOCOL_TLS_SERVER``).
    cert_pem:
        Leaf certificate in PEM encoding.
    key_pem:
        Leaf private key in PEM encoding (unencrypted).
    """
    combined = cert_pem + key_pem

    if sys.platform == "linux" and hasattr(os, "memfd_create"):
        fd = os.memfd_create("hr_leaf")  # type: ignore[attr-defined]
        try:
            _write_all_fd(fd, combined)
            ctx.load_cert_chain(f"/proc/self/fd/{fd}")
            return
        except (FileNotFoundError, PermissionError):
            # /proc not mounted / inaccessible (some containers) — fall through
            # to mkstemp. Caught narrowly ON PURPOSE: ssl.SSLError is a subclass
            # of OSError, so a broad `except OSError` would swallow a malformed
            # cert/key and silently disk-fall-back. Those propagate instead.
            pass
        finally:
            os.close(fd)

    _load_via_mkstemp(ctx, combined)


def _write_all_fd(fd: int, data: bytes) -> None:
    """Write all of *data* to *fd*, handling short-writes."""
    view = memoryview(data)
    written = 0
    total = len(data)
    while written < total:
        n = os.write(fd, view[written:])
        if n == 0:
            raise OSError("os.write wrote 0 bytes; cannot persist leaf PEM")
        written += n


def _load_via_mkstemp(ctx: ssl.SSLContext, combined: bytes) -> None:
    """Write *combined* to a 0600 mkstemp file, load it, then unlink."""
    fd, path = tempfile.mkstemp(prefix="hr_leaf_", suffix=".pem")
    try:
        _write_all_fd(fd, combined)
        os.close(fd)
        fd = -1  # prevent double-close in finally
        _assert_perms(Path(path), 0o600)
        ctx.load_cert_chain(path)
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(path)
        except OSError:
            pass
