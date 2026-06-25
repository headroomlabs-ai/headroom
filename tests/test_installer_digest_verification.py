"""Supply-chain integrity tests for release-binary installers.

``rtk``, ``lean-ctx`` and ``codebase-memory-mcp`` are downloaded and then
executed by ``headroom wrap``. These tests assert that every installer
verifies the downloaded bytes against a pinned SHA-256 digest before
unpacking, and fails closed (refuses to install) on a mismatch or an
unpinned asset unless the operator explicitly opts out.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from headroom.graph import installer as cbm_installer
from headroom.lean_ctx import installer as lean_installer
from headroom.rtk import installer as rtk_installer

# (module, allow-unverified env var, pinned-asset filename, error-label)
INSTALLERS = [
    pytest.param(
        rtk_installer,
        "HEADROOM_RTK_ALLOW_UNVERIFIED",
        "rtk-x86_64-unknown-linux-musl.tar.gz",
        "rtk",
        id="rtk",
    ),
    pytest.param(
        lean_installer,
        "HEADROOM_LEAN_CTX_ALLOW_UNVERIFIED",
        "lean-ctx-x86_64-unknown-linux-gnu.tar.gz",
        "lean-ctx",
        id="lean-ctx",
    ),
    pytest.param(
        cbm_installer,
        "HEADROOM_CBM_ALLOW_UNVERIFIED",
        "codebase-memory-mcp-linux-amd64.tar.gz",
        "codebase-memory-mcp",
        id="cbm",
    ),
]


@pytest.mark.parametrize(("mod", "allow_env", "filename", "label"), INSTALLERS)
def test_matching_digest_passes(mod, allow_env, filename, label) -> None:
    data = b"trusted release bytes"
    digest = hashlib.sha256(data).hexdigest()
    # Returns None (no raise) when the bytes match the pinned digest.
    assert mod._verify_asset_digest(filename, data, digest) is None


@pytest.mark.parametrize(("mod", "allow_env", "filename", "label"), INSTALLERS)
def test_tampered_bytes_are_rejected(mod, allow_env, filename, label) -> None:
    pinned = hashlib.sha256(b"trusted release bytes").hexdigest()
    tampered = b"malicious substituted artifact"
    with pytest.raises(RuntimeError, match="failed integrity check"):
        mod._verify_asset_digest(filename, tampered, pinned)


@pytest.mark.parametrize(("mod", "allow_env", "filename", "label"), INSTALLERS)
def test_unpinned_asset_refused_by_default(mod, allow_env, filename, label, monkeypatch) -> None:
    monkeypatch.delenv(allow_env, raising=False)
    with pytest.raises(RuntimeError, match="no pinned SHA-256 digest"):
        mod._verify_asset_digest(filename, b"some bytes", None)


@pytest.mark.parametrize(("mod", "allow_env", "filename", "label"), INSTALLERS)
def test_unpinned_asset_allowed_with_optout(mod, allow_env, filename, label, monkeypatch) -> None:
    monkeypatch.setenv(allow_env, "1")
    # Opt-out lets an unpinned (e.g. version-overridden) asset through.
    assert mod._verify_asset_digest(filename, b"some bytes", None) is None


@pytest.mark.parametrize(("mod", "allow_env", "filename", "label"), INSTALLERS)
def test_every_pinned_digest_is_a_sha256_hex(mod, allow_env, filename, label) -> None:
    digests = {
        rtk_installer: rtk_installer.RTK_ASSET_DIGESTS,
        lean_installer: lean_installer.LEAN_CTX_ASSET_DIGESTS,
        cbm_installer: cbm_installer.CBM_ASSET_DIGESTS,
    }[mod]
    assert digests, "expected at least one pinned asset digest"
    for name, digest in digests.items():
        assert len(digest) == 64, f"{name}: not a sha256 hex digest"
        assert all(c in "0123456789abcdef" for c in digest), f"{name}: non-hex digest"


def _tar_gz_with(member: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=member)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class _Response:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._data


def test_download_rtk_rejects_tampered_pinned_asset(monkeypatch, tmp_path: Path) -> None:
    """End-to-end: a tampered archive for a pinned target aborts the install."""
    monkeypatch.delenv("HEADROOM_RTK_TARGET", raising=False)
    monkeypatch.delenv("HEADROOM_RTK_ALLOW_UNVERIFIED", raising=False)
    tampered = _tar_gz_with("rtk", b"not the real rtk binary")

    with patch.object(rtk_installer, "RTK_BIN_DIR", tmp_path):
        with patch.object(
            rtk_installer, "_get_target_triple", return_value="x86_64-unknown-linux-musl"
        ):
            with patch.object(rtk_installer, "urlopen", return_value=_Response(tampered)):
                with pytest.raises(RuntimeError, match="failed integrity check"):
                    # Pin to the current release so this exercises the verified
                    # path (not the unpinned-override branch) regardless of the
                    # value of RTK_VERSION.
                    rtk_installer.download_rtk(rtk_installer.RTK_VERSION)

    # Nothing was unpacked.
    assert not (tmp_path / "rtk").exists()


def test_download_rtk_bare_version_still_hits_pinned_path(monkeypatch, tmp_path: Path) -> None:
    """A bare ``X.Y.Z`` for the current release normalizes to the pinned path.

    Without version normalization, ``download_rtk("0.42.4")`` would not equal
    ``RTK_VERSION`` (``v0.42.4``), be treated as an override, and skip the digest
    map — an accidental bypass. The tampered bytes must therefore fail the
    integrity check, not the "no pinned digest" override branch.
    """
    monkeypatch.delenv("HEADROOM_RTK_TARGET", raising=False)
    monkeypatch.delenv("HEADROOM_RTK_ALLOW_UNVERIFIED", raising=False)
    bare_version = rtk_installer.RTK_VERSION.lstrip("v")
    tampered = _tar_gz_with("rtk", b"not the real rtk binary")

    with patch.object(rtk_installer, "RTK_BIN_DIR", tmp_path):
        with patch.object(
            rtk_installer, "_get_target_triple", return_value="x86_64-unknown-linux-musl"
        ):
            with patch.object(rtk_installer, "urlopen", return_value=_Response(tampered)):
                with pytest.raises(RuntimeError, match="failed integrity check"):
                    rtk_installer.download_rtk(bare_version)

    assert not (tmp_path / "rtk").exists()
