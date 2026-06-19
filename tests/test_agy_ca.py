"""Tests for headroom.proxy.agy_ca — root CA lifecycle + combined bundle.

All tests use pytest's tmp_path; real ~/.headroom is never touched.
"""

from __future__ import annotations

import datetime
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
    build_combined_bundle,
    ensure_root_ca,
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
