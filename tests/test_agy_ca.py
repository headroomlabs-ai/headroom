"""Tests for headroom.proxy.agy_ca — root CA lifecycle + combined bundle.

All tests use pytest's tmp_path; real ~/.headroom is never touched.
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from headroom.proxy.agy_ca import (
    _BUNDLE_NAME,
    _CA_CERT_NAME,
    _CA_KEY_NAME,
    _OS_TRUST_PATHS,
    _assert_perms,
    _cert_near_expiry,
    _collect_corporate_ca_pems,
    _is_ca_cert,
    _parse_ca_certs_from_pem,
    _windows_trust_pem,
    build_combined_bundle,
    ensure_root_ca,
    load_cert_chain_in_memory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cert(
    is_ca: bool,
    days_valid: int = 3650,
    path_length: int | None = 0,
) -> bytes:
    """Generate a minimal PEM certificate for testing."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=days_valid)
        )
        .add_extension(
            x509.BasicConstraints(ca=is_ca, path_length=path_length if is_ca else None),
            critical=True,
        )
    )
    cert = builder.sign(key, hashes.SHA256())
    return cert.public_bytes(serialization.Encoding.PEM)


def _fake_system_bundle(tmp_path: Path, pem_data: bytes | None = None) -> Path:
    """Write a minimal fake system bundle, returning its path."""
    if pem_data is None:
        pem_data = _make_cert(is_ca=True)
    p = tmp_path / "system-ca-bundle.pem"
    p.write_bytes(pem_data)
    return p


# ---------------------------------------------------------------------------
# ensure_root_ca — generation
# ---------------------------------------------------------------------------


def test_ca_generated_on_first_call(tmp_path: Path) -> None:
    """First call creates key + cert under base_dir/ca/."""
    key, cert, key_path, cert_path = ensure_root_ca(base_dir=tmp_path)
    assert key_path.exists()
    assert cert_path.exists()
    assert _is_ca_cert(cert)


def test_ca_dir_is_0700(tmp_path: Path) -> None:
    ensure_root_ca(base_dir=tmp_path)
    ca_dir = tmp_path / "ca"
    _assert_perms(ca_dir, 0o700)


def test_ca_key_is_0600(tmp_path: Path) -> None:
    _, _, key_path, _ = ensure_root_ca(base_dir=tmp_path)
    _assert_perms(key_path, 0o600)


def test_ca_cert_is_0600(tmp_path: Path) -> None:
    _, _, _, cert_path = ensure_root_ca(base_dir=tmp_path)
    _assert_perms(cert_path, 0o600)


def test_ca_has_basic_constraints_ca_true(tmp_path: Path) -> None:
    _, cert, _, _ = ensure_root_ca(base_dir=tmp_path)
    bc = cert.extensions.get_extension_for_class(x509.BasicConstraints)
    assert bc.value.ca is True
    assert bc.value.path_length == 0


def test_ca_has_long_validity(tmp_path: Path) -> None:
    """Cert must be valid for at least 9 years (allowing some clock skew)."""
    _, cert, _, _ = ensure_root_ca(base_dir=tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)
    delta = cert.not_valid_after_utc - now
    assert delta.days >= 365 * 9


# ---------------------------------------------------------------------------
# ensure_root_ca — idempotency (reuse)
# ---------------------------------------------------------------------------


def test_second_call_reuses_existing_ca(tmp_path: Path) -> None:
    """Second call with valid existing CA returns same cert (by serial)."""
    _, cert1, _, _ = ensure_root_ca(base_dir=tmp_path)
    _, cert2, _, _ = ensure_root_ca(base_dir=tmp_path)
    assert cert1.serial_number == cert2.serial_number


def test_second_call_key_object_matches(tmp_path: Path) -> None:
    key1, _, _, _ = ensure_root_ca(base_dir=tmp_path)
    key2, _, _, _ = ensure_root_ca(base_dir=tmp_path)
    pub1 = key1.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    pub2 = key2.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    assert pub1 == pub2


# ---------------------------------------------------------------------------
# ensure_root_ca — regeneration on expiry
# ---------------------------------------------------------------------------


def _write_expiring_ca(base_dir: Path, days_valid: int = 1) -> None:
    """Overwrite the CA with a cert that expires soon (within regen threshold)."""
    ca_dir = base_dir / "ca"
    ca_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "expiring")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=days_valid))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_path = ca_dir / _CA_KEY_NAME
    cert_path = ca_dir / _CA_CERT_NAME
    key_path.write_bytes(key_pem)
    key_path.chmod(0o600)
    cert_path.write_bytes(cert_pem)
    cert_path.chmod(0o600)
    return cert.serial_number  # type: ignore[return-value]


def test_regen_on_expiry_produces_new_serial(tmp_path: Path) -> None:
    old_serial = _write_expiring_ca(tmp_path, days_valid=1)
    _, new_cert, _, _ = ensure_root_ca(base_dir=tmp_path)
    assert new_cert.serial_number != old_serial


def test_regen_deletes_old_bundle(tmp_path: Path) -> None:
    """Stale combined bundle is removed when CA is regenerated."""
    old_bundle = tmp_path / _BUNDLE_NAME
    old_bundle.write_bytes(b"stale")
    old_bundle.chmod(0o600)
    _write_expiring_ca(tmp_path, days_valid=1)
    ensure_root_ca(base_dir=tmp_path)
    # Bundle was deleted; new content would need build_combined_bundle.
    assert not old_bundle.exists()


def test_regen_deletes_old_leaves(tmp_path: Path) -> None:
    """Leaf cert directory is cleaned up on regeneration."""
    leaves_dir = tmp_path / "leaves"
    leaves_dir.mkdir(mode=0o700)
    (leaves_dir / "example.com.crt").write_bytes(b"leaf")
    _write_expiring_ca(tmp_path, days_valid=1)
    ensure_root_ca(base_dir=tmp_path)
    assert not leaves_dir.exists()


# ---------------------------------------------------------------------------
# _is_ca_cert
# ---------------------------------------------------------------------------


def test_is_ca_cert_true_for_ca() -> None:
    pem = _make_cert(is_ca=True)
    cert = x509.load_pem_x509_certificate(pem)
    assert _is_ca_cert(cert) is True


def test_is_ca_cert_false_for_leaf() -> None:
    pem = _make_cert(is_ca=False)
    cert = x509.load_pem_x509_certificate(pem)
    assert _is_ca_cert(cert) is False


# ---------------------------------------------------------------------------
# _parse_ca_certs_from_pem — per-object filter
# ---------------------------------------------------------------------------


def test_parse_filters_non_ca_leaves() -> None:
    """Multi-cert PEM: only CA:TRUE objects survive."""
    ca_pem = _make_cert(is_ca=True)
    leaf_pem = _make_cert(is_ca=False)
    combined = ca_pem + leaf_pem
    results = _parse_ca_certs_from_pem(combined)
    assert len(results) == 1
    cert = x509.load_pem_x509_certificate(results[0])
    assert _is_ca_cert(cert) is True


def test_parse_all_ca_certs_included() -> None:
    ca1 = _make_cert(is_ca=True)
    ca2 = _make_cert(is_ca=True)
    combined = ca1 + ca2
    results = _parse_ca_certs_from_pem(combined)
    assert len(results) == 2


def test_parse_empty_pem_returns_empty() -> None:
    assert _parse_ca_certs_from_pem(b"") == []


def test_parse_skips_invalid_pem_blocks() -> None:
    ca_pem = _make_cert(is_ca=True)
    garbage = b"-----BEGIN CERTIFICATE-----\nZZZZZZ\n-----END CERTIFICATE-----\n"
    combined = ca_pem + garbage
    results = _parse_ca_certs_from_pem(combined)
    # Only the valid CA cert should come through.
    assert len(results) == 1


# ---------------------------------------------------------------------------
# _collect_corporate_ca_pems
# ---------------------------------------------------------------------------


def test_collect_corp_ca_from_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Corporate CA file with one CA + one leaf → only CA returned."""
    ca_pem = _make_cert(is_ca=True)
    leaf_pem = _make_cert(is_ca=False)
    corp_file = tmp_path / "corp.pem"
    corp_file.write_bytes(ca_pem + leaf_pem)

    monkeypatch.setenv("SSL_CERT_FILE", str(corp_file))
    results = _collect_corporate_ca_pems(("SSL_CERT_FILE",))
    assert len(results) == 1
    cert = x509.load_pem_x509_certificate(results[0])
    assert _is_ca_cert(cert) is True


def test_collect_corp_ca_missing_file_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing corporate CA file → empty result (no crash)."""
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "nonexistent.pem"))
    results = _collect_corporate_ca_pems(("SSL_CERT_FILE",))
    assert results == []


def test_collect_corp_ca_unset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("NODE_EXTRA_CA_CERTS", raising=False)
    results = _collect_corporate_ca_pems(("SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS"))
    assert results == []


# ---------------------------------------------------------------------------
# build_combined_bundle
# ---------------------------------------------------------------------------


def test_bundle_is_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys_bundle = _fake_system_bundle(tmp_path)
    monkeypatch.setattr(
        "headroom.proxy.agy_ca._SYSTEM_BUNDLE_CANDIDATES",
        (str(sys_bundle),),
    )
    bundle_path = build_combined_bundle(base_dir=tmp_path, corp_env_vars=())
    assert bundle_path.exists()
    assert bundle_path.stat().st_size > 0


def test_bundle_is_0600(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys_bundle = _fake_system_bundle(tmp_path)
    monkeypatch.setattr(
        "headroom.proxy.agy_ca._SYSTEM_BUNDLE_CANDIDATES",
        (str(sys_bundle),),
    )
    bundle_path = build_combined_bundle(base_dir=tmp_path, corp_env_vars=())
    _assert_perms(bundle_path, 0o600)


def test_parent_dir_is_0700(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys_bundle = _fake_system_bundle(tmp_path)
    monkeypatch.setattr(
        "headroom.proxy.agy_ca._SYSTEM_BUNDLE_CANDIDATES",
        (str(sys_bundle),),
    )
    build_combined_bundle(base_dir=tmp_path, corp_env_vars=())
    _assert_perms(tmp_path, 0o700)


def test_bundle_contains_system_ca(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys_ca_pem = _make_cert(is_ca=True)
    sys_bundle = _fake_system_bundle(tmp_path, pem_data=sys_ca_pem)
    monkeypatch.setattr(
        "headroom.proxy.agy_ca._SYSTEM_BUNDLE_CANDIDATES",
        (str(sys_bundle),),
    )
    bundle_path = build_combined_bundle(base_dir=tmp_path, corp_env_vars=())
    bundle_data = bundle_path.read_bytes()
    # The system CA PEM bytes must appear verbatim in the bundle.
    assert sys_ca_pem in bundle_data


def test_bundle_contains_headroom_ca(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys_bundle = _fake_system_bundle(tmp_path)
    monkeypatch.setattr(
        "headroom.proxy.agy_ca._SYSTEM_BUNDLE_CANDIDATES",
        (str(sys_bundle),),
    )
    _, ca_cert, _, _ = ensure_root_ca(base_dir=tmp_path)
    headroom_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    bundle_path = build_combined_bundle(base_dir=tmp_path, corp_env_vars=())
    bundle_data = bundle_path.read_bytes()
    assert headroom_pem in bundle_data


def test_bundle_contains_corp_ca_but_not_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corporate CA:TRUE cert appears in bundle; leaf cert does not."""
    sys_bundle = _fake_system_bundle(tmp_path)
    monkeypatch.setattr(
        "headroom.proxy.agy_ca._SYSTEM_BUNDLE_CANDIDATES",
        (str(sys_bundle),),
    )
    corp_ca_pem = _make_cert(is_ca=True)
    leaf_pem = _make_cert(is_ca=False)
    corp_file = tmp_path / "corp.pem"
    corp_file.write_bytes(corp_ca_pem + leaf_pem)

    # First call without corp CAs to seed the CA on disk.
    build_combined_bundle(
        base_dir=tmp_path,
        corp_env_vars=(),
    )
    # Call again using a custom corp_env_vars pointing at our fixture file.
    monkeypatch.setenv("_TEST_CORP_CA", str(corp_file))
    bundle_path2 = build_combined_bundle(
        base_dir=tmp_path,
        corp_env_vars=("_TEST_CORP_CA",),
    )
    bundle_data = bundle_path2.read_bytes()
    assert corp_ca_pem in bundle_data
    assert leaf_pem not in bundle_data


def test_windows_trust_pem_filters_non_ca(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Windows ssl.enum_certificates path must drop non-CA (leaf) certs.

    ssl.enum_certificates returns every cert in the store including leaf certs;
    _windows_trust_pem must run them through the CA:TRUE filter so only CA
    anchors end up in the trust bundle. Mock-driven so it runs on every OS
    (ssl.enum_certificates does not exist off Windows).
    """
    ca_pem = _make_cert(is_ca=True)
    leaf_pem = _make_cert(is_ca=False)
    ca_cert = x509.load_pem_x509_certificate(ca_pem)
    leaf_cert = x509.load_pem_x509_certificate(leaf_pem)
    ca_der = ca_cert.public_bytes(serialization.Encoding.DER)
    leaf_der = leaf_cert.public_bytes(serialization.Encoding.DER)

    def fake_enum(store: str) -> list[tuple[bytes, str, bool]]:
        # Return the CA + leaf only for ROOT; CA store empty (avoid double count).
        if store == "ROOT":
            return [(ca_der, "x509_asn", True), (leaf_der, "x509_asn", True)]
        return []

    monkeypatch.setattr("ssl.enum_certificates", fake_enum, raising=False)

    result = _windows_trust_pem()
    # Parse EVERY cert block in the raw result (NOT via the CA filter) so a
    # regression that dropped the internal filter would surface the leaf here.
    marker = b"-----BEGIN CERTIFICATE-----"
    present = {
        x509.load_pem_x509_certificate(marker + block).serial_number
        for block in result.split(marker)[1:]
    }
    assert ca_cert.serial_number in present, "CA anchor must be present"
    assert leaf_cert.serial_number not in present, "leaf cert must be filtered out"


def test_bundle_not_in_os_trust_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bundle path must not reside under any known OS trust store location."""
    sys_bundle = _fake_system_bundle(tmp_path)
    monkeypatch.setattr(
        "headroom.proxy.agy_ca._SYSTEM_BUNDLE_CANDIDATES",
        (str(sys_bundle),),
    )
    bundle_path = build_combined_bundle(base_dir=tmp_path, corp_env_vars=())
    resolved = str(bundle_path.resolve())
    for trust_path in _OS_TRUST_PATHS:
        assert not resolved.startswith(trust_path), (
            f"Bundle {resolved} is inside OS trust path {trust_path}"
        )


def test_ca_never_written_to_os_trust_store(
    tmp_path: Path,
) -> None:
    """CA key + cert paths must not reside under OS trust store directories."""
    _, _, key_path, cert_path = ensure_root_ca(base_dir=tmp_path)
    for path in (key_path, cert_path):
        resolved = str(path.resolve())
        for trust_path in _OS_TRUST_PATHS:
            assert not resolved.startswith(trust_path), (
                f"{path} is inside OS trust path {trust_path}"
            )


# ---------------------------------------------------------------------------
# fail-fast: no system bundle
# ---------------------------------------------------------------------------


def test_no_system_bundle_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "headroom.proxy.agy_ca._SYSTEM_BUNDLE_CANDIDATES",
        (),
    )
    with pytest.raises(RuntimeError, match="No system CA bundle found"):
        build_combined_bundle(base_dir=tmp_path, corp_env_vars=())


# ---------------------------------------------------------------------------
# _cert_near_expiry
# ---------------------------------------------------------------------------


def test_cert_near_expiry_true_for_expiring() -> None:
    pem = _make_cert(is_ca=True, days_valid=1)
    cert = x509.load_pem_x509_certificate(pem)
    assert _cert_near_expiry(cert) is True


def test_cert_near_expiry_false_for_valid() -> None:
    pem = _make_cert(is_ca=True, days_valid=3650)
    cert = x509.load_pem_x509_certificate(pem)
    assert _cert_near_expiry(cert) is False


# ---------------------------------------------------------------------------
# Bundle idempotency
# ---------------------------------------------------------------------------


def test_build_bundle_twice_same_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the bundle twice without CA regen produces identical content."""
    sys_bundle = _fake_system_bundle(tmp_path)
    monkeypatch.setattr(
        "headroom.proxy.agy_ca._SYSTEM_BUNDLE_CANDIDATES",
        (str(sys_bundle),),
    )
    path1 = build_combined_bundle(base_dir=tmp_path, corp_env_vars=())
    data1 = path1.read_bytes()
    path2 = build_combined_bundle(base_dir=tmp_path, corp_env_vars=())
    data2 = path2.read_bytes()
    assert data1 == data2


# ---------------------------------------------------------------------------
# Regression: clean-install with nested base_dir (parents must be created)
# ---------------------------------------------------------------------------


def test_clean_install_nested_base_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_root_ca then build_combined_bundle on a completely fresh nested
    base_dir must succeed and leave base_dir at 0o700.

    Constructs base_dir as tmp_path / "sub" / ".headroom" so the code itself
    must create all intermediate directories — none are pre-created.
    """
    base_dir = tmp_path / "sub" / ".headroom"
    # Sanity: must not exist before the call.
    assert not base_dir.exists()

    sys_bundle = _fake_system_bundle(tmp_path)
    monkeypatch.setattr(
        "headroom.proxy.agy_ca._SYSTEM_BUNDLE_CANDIDATES",
        (str(sys_bundle),),
    )

    # This must not raise AssertionError or PermissionError.
    ensure_root_ca(base_dir=base_dir)
    bundle_path = build_combined_bundle(base_dir=base_dir, corp_env_vars=())

    # base_dir itself must be 0o700 (the root cause of the original bug).
    _assert_perms(base_dir, 0o700)
    assert bundle_path.exists()


# ---------------------------------------------------------------------------
# Helpers shared by load_cert_chain_in_memory tests
# ---------------------------------------------------------------------------


def _make_leaf_pem_pair() -> tuple[bytes, bytes]:
    """Return (cert_pem, key_pem) for a minimal self-signed leaf."""
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "leaf.test")]))
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "leaf.test")]))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(hours=72))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("leaf.test")]), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=True)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    return cert_pem, key_pem


# ---------------------------------------------------------------------------
# load_cert_chain_in_memory — primary path (memfd on Linux)
# ---------------------------------------------------------------------------


def test_load_cert_chain_in_memory_loads_usable_ctx() -> None:
    """Combined cert+key is loaded into a usable SSLContext; no exception."""
    import ssl

    cert_pem, key_pem = _make_leaf_pem_pair()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # Must not raise.
    load_cert_chain_in_memory(ctx, cert_pem, key_pem)


@pytest.mark.skipif(sys.platform != "linux", reason="requires /proc/self/fd")
def test_load_cert_chain_in_memory_no_fd_leak() -> None:
    """After load, the memfd (or temp file) is closed — no leaked descriptors."""
    import ssl

    cert_pem, key_pem = _make_leaf_pem_pair()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    before = set(os.listdir("/proc/self/fd"))
    load_cert_chain_in_memory(ctx, cert_pem, key_pem)
    after = set(os.listdir("/proc/self/fd"))

    # The only new fd allowed is the /proc/self/fd dirfd opened by listdir itself.
    new_fds = after - before
    # Filter out the dirfd from the listdir call above (it closes immediately).
    assert len(new_fds) == 0, f"Leaked file descriptors after load: {new_fds}"


def test_load_cert_chain_in_memory_no_tmpfile_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Linux (memfd available), mkstemp and NamedTemporaryFile are NOT called."""
    import ssl
    import tempfile as _tempfile

    cert_pem, key_pem = _make_leaf_pem_pair()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    mkstemp_called = [False]
    named_tmp_called = [False]

    original_mkstemp = _tempfile.mkstemp
    original_named = _tempfile.NamedTemporaryFile

    def _spy_mkstemp(*args: object, **kwargs: object) -> object:
        mkstemp_called[0] = True
        return original_mkstemp(*args, **kwargs)

    def _spy_named(*args: object, **kwargs: object) -> object:
        named_tmp_called[0] = True
        return original_named(*args, **kwargs)

    monkeypatch.setattr(_tempfile, "mkstemp", _spy_mkstemp)
    monkeypatch.setattr(_tempfile, "NamedTemporaryFile", _spy_named)

    if sys.platform == "linux" and hasattr(os, "memfd_create"):
        load_cert_chain_in_memory(ctx, cert_pem, key_pem)
        assert not mkstemp_called[0], "mkstemp must NOT be called when memfd_create is available"
        assert not named_tmp_called[0], (
            "NamedTemporaryFile must NOT be called when memfd_create is available"
        )


# ---------------------------------------------------------------------------
# load_cert_chain_in_memory — short-write safety
# ---------------------------------------------------------------------------


def test_load_cert_chain_in_memory_short_write_handled() -> None:
    """Helper writes all bytes even if os.write short-writes (1 byte at a time)."""
    import ssl

    cert_pem, key_pem = _make_leaf_pem_pair()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    if not hasattr(os, "memfd_create"):
        pytest.skip("memfd_create not available on this platform")

    original_write = os.write
    written_chunks: list[int] = []

    def _one_byte_write(fd: int, data: bytes | bytearray) -> int:
        # Only short-write to memfd fds; pass through others.
        try:
            path = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            path = ""
        if "memfd" in path or "anon" in path.lower():
            n = original_write(fd, data[:1])
            written_chunks.append(n)
            return n
        return original_write(fd, data)

    import unittest.mock

    with unittest.mock.patch("os.write", side_effect=_one_byte_write):
        # Must succeed despite 1-byte writes.
        load_cert_chain_in_memory(ctx, cert_pem, key_pem)

    total = sum(written_chunks)
    expected = len(cert_pem + key_pem)
    assert total == expected, f"Expected {expected} bytes written in chunks, got {total}"


# ---------------------------------------------------------------------------
# load_cert_chain_in_memory — fallback path (memfd absent/unavailable)
# ---------------------------------------------------------------------------


def test_load_cert_chain_in_memory_fallback_when_no_memfd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When memfd_create is absent, fallback uses mkstemp (0600) and unlinks it."""
    import ssl
    import tempfile as _tempfile

    cert_pem, key_pem = _make_leaf_pem_pair()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    # Force fallback: remove memfd_create from os.
    monkeypatch.delattr(os, "memfd_create", raising=False)

    tmp_paths_created: list[str] = []
    tmp_paths_unlinked: list[str] = []
    original_mkstemp = _tempfile.mkstemp
    original_unlink = os.unlink

    def _spy_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        fd, path = original_mkstemp(*args, **kwargs)
        tmp_paths_created.append(path)
        return fd, path

    def _spy_unlink(path: str, *args: object, **kwargs: object) -> None:
        if any(path == p for p in tmp_paths_created):
            tmp_paths_unlinked.append(path)
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(_tempfile, "mkstemp", _spy_mkstemp)
    monkeypatch.setattr(os, "unlink", _spy_unlink)

    load_cert_chain_in_memory(ctx, cert_pem, key_pem)

    assert tmp_paths_created, "Fallback must call mkstemp"
    for p in tmp_paths_created:
        assert not os.path.exists(p), f"Temp file {p} must be unlinked after load"
    assert set(tmp_paths_created) == set(tmp_paths_unlinked), (
        "Every temp file created must be unlinked"
    )


def test_load_cert_chain_in_memory_fallback_0600(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback temp file has 0600 permissions (asserted by helper)."""
    import ssl
    import stat as _stat
    import tempfile as _tempfile

    cert_pem, key_pem = _make_leaf_pem_pair()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    monkeypatch.delattr(os, "memfd_create", raising=False)

    observed_modes: list[int] = []
    original_mkstemp = _tempfile.mkstemp

    def _spy_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        fd, path = original_mkstemp(*args, **kwargs)
        mode = _stat.S_IMODE(os.stat(path).st_mode)
        observed_modes.append(mode)
        return fd, path

    monkeypatch.setattr(_tempfile, "mkstemp", _spy_mkstemp)

    load_cert_chain_in_memory(ctx, cert_pem, key_pem)

    assert observed_modes, "Fallback must call mkstemp"
    if sys.platform != "win32":
        for mode in observed_modes:
            assert mode == 0o600, f"Temp file mode must be 0600, got {oct(mode)}"


def test_load_cert_chain_in_memory_fallback_unlinks_on_load_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback unlinks temp file even when load_cert_chain raises."""
    import ssl
    import tempfile as _tempfile

    cert_pem, key_pem = _make_leaf_pem_pair()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    monkeypatch.delattr(os, "memfd_create", raising=False)

    tmp_paths_created: list[str] = []
    original_mkstemp = _tempfile.mkstemp

    def _spy_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        fd, path = original_mkstemp(*args, **kwargs)
        tmp_paths_created.append(path)
        return fd, path

    monkeypatch.setattr(_tempfile, "mkstemp", _spy_mkstemp)

    # Patch load_cert_chain to always raise.
    monkeypatch.setattr(
        ctx, "load_cert_chain", lambda *a, **kw: (_ for _ in ()).throw(ssl.SSLError("injected"))
    )

    with pytest.raises(ssl.SSLError):
        load_cert_chain_in_memory(ctx, cert_pem, key_pem)

    # Temp file must still be cleaned up.
    assert tmp_paths_created, "mkstemp must have been called"
    for p in tmp_paths_created:
        assert not os.path.exists(p), f"Temp file {p} must be unlinked even after load exception"


def test_load_cert_chain_in_memory_fallback_via_proc_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When memfd exists but the /proc path is missing (FileNotFoundError),
    the helper falls back to mkstemp.

    Real ``/proc``-absent failure surfaces as FileNotFoundError (ENOENT), which
    is what the helper catches narrowly — a bare OSError/SSLError must NOT
    trigger the disk fallback (see test_..._bad_cert_propagates_without_disk).
    """
    import ssl
    import tempfile as _tempfile

    cert_pem, key_pem = _make_leaf_pem_pair()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    if not hasattr(os, "memfd_create"):
        pytest.skip("memfd_create not available; fallback-via-/proc path not applicable")

    # Patch load_cert_chain to raise FileNotFoundError on first call (simulating
    # /proc not mounted), then succeed on the second (fallback's mkstemp path).
    calls: list[int] = [0]
    original_load = ctx.__class__.load_cert_chain

    def _raise_once(self: ssl.SSLContext, *args: object, **kwargs: object) -> None:
        calls[0] += 1
        if calls[0] == 1:
            raise FileNotFoundError("simulated /proc not mounted")
        original_load(self, *args, **kwargs)

    monkeypatch.setattr(ssl.SSLContext, "load_cert_chain", _raise_once)

    tmp_paths_created: list[str] = []
    original_mkstemp = _tempfile.mkstemp

    def _spy_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        fd, path = original_mkstemp(*args, **kwargs)
        tmp_paths_created.append(path)
        return fd, path

    monkeypatch.setattr(_tempfile, "mkstemp", _spy_mkstemp)

    load_cert_chain_in_memory(ctx, cert_pem, key_pem)

    assert tmp_paths_created, "Fallback (mkstemp) must trigger when /proc path is FileNotFoundError"
    for p in tmp_paths_created:
        assert not os.path.exists(p), f"Fallback temp {p} must be unlinked"


def test_load_cert_chain_in_memory_bad_cert_propagates_without_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed cert/key (ssl.SSLError, an OSError subclass) must propagate
    and NOT silently disk-fall-back via mkstemp."""
    import ssl
    import tempfile as _tempfile

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    if not hasattr(os, "memfd_create"):
        pytest.skip("memfd_create not available; primary path not exercised")

    mkstemp_called = [False]
    original_mkstemp = _tempfile.mkstemp

    def _spy_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        mkstemp_called[0] = True
        return original_mkstemp(*args, **kwargs)

    monkeypatch.setattr(_tempfile, "mkstemp", _spy_mkstemp)

    # Garbage PEM -> load_cert_chain raises ssl.SSLError (subclass of OSError).
    with pytest.raises(ssl.SSLError):
        load_cert_chain_in_memory(ctx, b"-----BEGIN CERTIFICATE-----\nnope\n", b"not-a-key")

    assert not mkstemp_called[0], "bad cert must NOT trigger the disk fallback"


# ---------------------------------------------------------------------------
# ensure_root_ca: corrupt CA key → regenerate (not crash)
# ---------------------------------------------------------------------------


def test_ensure_root_ca_corrupt_key_regenerates(tmp_path: Path) -> None:
    """Valid cert + corrupt key file → ensure_root_ca regenerates, not raises."""
    # First call creates a valid CA on disk.
    _, cert1, key_path, cert_path = ensure_root_ca(base_dir=tmp_path)

    # Overwrite the key with garbage so the parse fails.
    key_path.write_bytes(
        b"-----BEGIN RSA PRIVATE KEY-----\nGARBAGE\n-----END RSA PRIVATE KEY-----\n"
    )

    # Must not raise; must produce a fresh (different serial) CA.
    key2, cert2, _, _ = ensure_root_ca(base_dir=tmp_path)
    assert cert2.serial_number != cert1.serial_number, (
        "corrupt key must trigger regeneration, yielding a new cert"
    )
    # Returned key must be usable (public_bytes does not raise).
    key2.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


# ---------------------------------------------------------------------------
# _assert_perms: skipped on non-POSIX
# ---------------------------------------------------------------------------


def test_assert_perms_skipped_on_non_posix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On non-POSIX platforms _assert_perms must be a no-op (never raise)."""
    p = tmp_path / "file.bin"
    p.write_bytes(b"x")
    # Monkeypatch os.name inside the module under test.
    monkeypatch.setattr("headroom.proxy.agy_ca.os.name", "nt")
    # Any expected_mode value; on real POSIX the mode would differ and raise.
    _assert_perms(p, 0o600)  # must not raise
    _assert_perms(p, 0o700)  # must not raise


# ---------------------------------------------------------------------------
# _write_secure: uses os.replace (atomic cross-platform rename)
# ---------------------------------------------------------------------------


def test_write_secure_uses_os_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_write_secure must call os.replace instead of Path.rename."""
    import headroom.proxy.agy_ca as _mod

    replace_calls: list[tuple[object, object]] = []
    original_replace = os.replace

    def _spy_replace(src: object, dst: object) -> None:
        replace_calls.append((src, dst))
        original_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(_mod.os, "replace", _spy_replace)

    dest = tmp_path / "out.key"
    _mod._write_secure(dest, b"hello")

    assert replace_calls, "os.replace must have been called by _write_secure"
    assert dest.read_bytes() == b"hello"
