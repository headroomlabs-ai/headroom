"""Tests for the tokensave release-binary installer."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.graph import tokensave_installer as ts


def _tar_archive(member_name: str = ts.TOKENSAVE_BIN_NAME) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        data = b"#!/bin/sh\necho version\n"
        info = tarfile.TarInfo(name=member_name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return payload.getvalue()


def _zip_archive(member_name: str = "tokensave.exe") -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr(member_name, b"binary")
    return payload.getvalue()


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._data


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("darwin", "arm64", ("tokensave-v9-aarch64-macos.tar.gz", "tar.gz")),
        ("linux", "aarch64", ("tokensave-v9-aarch64-linux.tar.gz", "tar.gz")),
        ("linux", "arm64", ("tokensave-v9-aarch64-linux.tar.gz", "tar.gz")),
        ("linux", "x86_64", ("tokensave-v9-x86_64-linux.tar.gz", "tar.gz")),
        ("windows", "amd64", ("tokensave-v9-x86_64-windows.zip", "zip")),
        ("windows", "arm64", ("tokensave-v9-aarch64-windows.zip", "zip")),
    ],
)
def test_detect_asset_variants(monkeypatch, system, machine, expected) -> None:
    monkeypatch.setattr(ts.platform, "system", lambda: system)
    monkeypatch.setattr(ts.platform, "machine", lambda: machine)
    assert ts._detect_asset("v9") == expected


def test_detect_asset_returns_none_for_intel_mac_and_unknown(monkeypatch) -> None:
    monkeypatch.setattr(ts.platform, "system", lambda: "darwin")
    monkeypatch.setattr(ts.platform, "machine", lambda: "x86_64")
    assert ts._detect_asset("v9") is None  # no x86_64-macos asset is published

    monkeypatch.setattr(ts.platform, "system", lambda: "solaris")
    monkeypatch.setattr(ts.platform, "machine", lambda: "sparc")
    assert ts._detect_asset("v9") is None


def test_get_tokensave_path_prefers_path_then_install_dir(monkeypatch, tmp_path: Path) -> None:
    on_path = tmp_path / "on-path"
    installed = tmp_path / ts.TOKENSAVE_BIN_NAME
    installed.write_text("bin")
    monkeypatch.setattr(ts, "TOKENSAVE_BIN_DIR", tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: str(on_path))
    assert ts.get_tokensave_path() == on_path

    monkeypatch.setattr("shutil.which", lambda name: None)
    assert ts.get_tokensave_path() == installed

    installed.unlink()
    assert ts.get_tokensave_path() is None


def test_ensure_offline_returns_none_when_absent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_BIN_DIR", tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setenv("HEADROOM_BINARIES_OFFLINE", "1")

    def _boom(*a, **k):
        raise AssertionError("download must not run when offline")

    monkeypatch.setattr(ts, "download_tokensave", _boom)
    assert ts.ensure_tokensave() is None


def test_ensure_returns_existing_without_download(monkeypatch, tmp_path: Path) -> None:
    existing = tmp_path / ts.TOKENSAVE_BIN_NAME
    existing.write_text("bin")
    monkeypatch.setattr(ts, "get_tokensave_path", lambda: existing)

    def _boom(*a, **k):
        raise AssertionError("download must not run when binary present")

    monkeypatch.setattr(ts, "download_tokensave", _boom)
    assert ts.ensure_tokensave() == existing


def test_ensure_returns_none_on_unsupported_platform(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ts, "get_tokensave_path", lambda: None)
    monkeypatch.delenv("HEADROOM_BINARIES_OFFLINE", raising=False)
    monkeypatch.setattr(ts.platform, "system", lambda: "darwin")
    monkeypatch.setattr(ts.platform, "machine", lambda: "x86_64")  # no asset
    assert ts.ensure_tokensave() is None


def test_download_tokensave_tarball(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_BIN_DIR", tmp_path)
    monkeypatch.setattr(ts.platform, "system", lambda: "linux")
    monkeypatch.setattr(ts.platform, "machine", lambda: "x86_64")
    # Synthetic archive bytes won't match the pinned digest; this test covers
    # extraction, not integrity, so opt out of verification explicitly.
    monkeypatch.setenv("HEADROOM_TOKENSAVE_ALLOW_UNVERIFIED", "1")
    monkeypatch.setattr(ts, "urlopen", lambda url, timeout=60: FakeResponse(_tar_archive()))
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="tokensave 6\n")
    )
    path = ts.download_tokensave(version="v0.0.0-test")
    assert path == tmp_path / ts.TOKENSAVE_BIN_NAME
    assert path.exists()


def test_download_tokensave_zip_windows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_BIN_DIR", tmp_path)
    monkeypatch.setattr(ts.platform, "system", lambda: "windows")
    monkeypatch.setattr(ts.platform, "machine", lambda: "amd64")
    monkeypatch.setenv("HEADROOM_TOKENSAVE_ALLOW_UNVERIFIED", "1")
    monkeypatch.setattr(ts, "urlopen", lambda url, timeout=60: FakeResponse(_zip_archive()))
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="tokensave 6\n")
    )
    path = ts.download_tokensave(version="v0.0.0-test")
    assert path == tmp_path / "tokensave.exe"
    assert path.exists()


def test_download_raises_for_unsupported_platform(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_BIN_DIR", tmp_path)
    monkeypatch.setattr(ts.platform, "system", lambda: "darwin")
    monkeypatch.setattr(ts.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="no prebuilt tokensave asset"):
        ts.download_tokensave(version="v7.0.0")


def test_download_wraps_network_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_BIN_DIR", tmp_path)
    monkeypatch.setattr(ts.platform, "system", lambda: "linux")
    monkeypatch.setattr(ts.platform, "machine", lambda: "x86_64")

    def _boom(url, timeout=60):
        raise OSError("connection refused")

    monkeypatch.setattr(ts, "urlopen", _boom)
    with pytest.raises(RuntimeError, match="Failed to download tokensave"):
        ts.download_tokensave(version="v7.0.0")


def test_download_raises_when_binary_missing_from_tarball(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_BIN_DIR", tmp_path)
    monkeypatch.setattr(ts.platform, "system", lambda: "linux")
    monkeypatch.setattr(ts.platform, "machine", lambda: "x86_64")
    # Archive contains an unrelated member, not the tokensave binary.
    monkeypatch.setenv("HEADROOM_TOKENSAVE_ALLOW_UNVERIFIED", "1")
    monkeypatch.setattr(
        ts, "urlopen", lambda url, timeout=60: FakeResponse(_tar_archive("README.md"))
    )
    with pytest.raises(RuntimeError, match="binary not found in archive"):
        ts.download_tokensave(version="v0.0.0-test")


def test_download_raises_when_binary_missing_from_zip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_BIN_DIR", tmp_path)
    monkeypatch.setattr(ts.platform, "system", lambda: "windows")
    monkeypatch.setattr(ts.platform, "machine", lambda: "amd64")
    monkeypatch.setenv("HEADROOM_TOKENSAVE_ALLOW_UNVERIFIED", "1")
    monkeypatch.setattr(
        ts, "urlopen", lambda url, timeout=60: FakeResponse(_zip_archive("notes.txt"))
    )
    with pytest.raises(RuntimeError, match="binary not found in archive"):
        ts.download_tokensave(version="v0.0.0-test")


def test_download_tolerates_failed_version_check(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_BIN_DIR", tmp_path)
    monkeypatch.setattr(ts.platform, "system", lambda: "linux")
    monkeypatch.setattr(ts.platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("HEADROOM_TOKENSAVE_ALLOW_UNVERIFIED", "1")
    monkeypatch.setattr(ts, "urlopen", lambda url, timeout=60: FakeResponse(_tar_archive()))
    # Non-zero return code and a raising probe must both be non-fatal.
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="x")
    )
    assert ts.download_tokensave(version="v0.0.0-test") == tmp_path / ts.TOKENSAVE_BIN_NAME

    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe boom"))
    )
    assert ts.download_tokensave(version="v0.0.0-test") == tmp_path / ts.TOKENSAVE_BIN_NAME


def test_verify_asset_digest_accepts_matching_hash(monkeypatch) -> None:
    import hashlib

    data = b"some-release-bytes"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(ts, "TOKENSAVE_ASSET_DIGESTS", {"asset.tar.gz": digest})
    # No exception => verification passed.
    ts._verify_asset_digest("asset.tar.gz", data)


def test_verify_asset_digest_rejects_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_ASSET_DIGESTS", {"asset.tar.gz": "00" * 32})
    with pytest.raises(RuntimeError, match="failed integrity check"):
        ts._verify_asset_digest("asset.tar.gz", b"tampered")


def test_verify_asset_digest_refuses_unpinned_without_optout(monkeypatch) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_ASSET_DIGESTS", {})
    monkeypatch.delenv("HEADROOM_TOKENSAVE_ALLOW_UNVERIFIED", raising=False)
    with pytest.raises(RuntimeError, match="no pinned SHA-256 digest"):
        ts._verify_asset_digest("unknown.tar.gz", b"bytes")


def test_verify_asset_digest_allows_unpinned_with_optout(monkeypatch) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_ASSET_DIGESTS", {})
    monkeypatch.setenv("HEADROOM_TOKENSAVE_ALLOW_UNVERIFIED", "1")
    ts._verify_asset_digest("unknown.tar.gz", b"bytes")  # no exception


def test_download_aborts_on_digest_mismatch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_BIN_DIR", tmp_path)
    monkeypatch.setattr(ts.platform, "system", lambda: "linux")
    monkeypatch.setattr(ts.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("HEADROOM_TOKENSAVE_ALLOW_UNVERIFIED", raising=False)
    # Pin a digest that the synthetic archive cannot match.
    monkeypatch.setattr(
        ts, "TOKENSAVE_ASSET_DIGESTS", {"tokensave-v7.0.0-x86_64-linux.tar.gz": "00" * 32}
    )
    monkeypatch.setattr(ts, "urlopen", lambda url, timeout=60: FakeResponse(_tar_archive()))
    with pytest.raises(RuntimeError, match="failed integrity check"):
        ts.download_tokensave(version="v7.0.0")
    # The unverified binary must not have been written.
    assert not (tmp_path / ts.TOKENSAVE_BIN_NAME).exists()


def test_download_honors_invalid_url_scheme(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ts, "TOKENSAVE_BIN_DIR", tmp_path)
    monkeypatch.setattr(ts.platform, "system", lambda: "linux")
    monkeypatch.setattr(ts.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(ts, "GITHUB_RELEASE_URL", "ftp://example.test/releases")
    with pytest.raises(RuntimeError, match="Failed to download tokensave"):
        ts.download_tokensave(version="v7.0.0")


def test_ensure_returns_none_when_download_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ts, "get_tokensave_path", lambda: None)
    monkeypatch.delenv("HEADROOM_BINARIES_OFFLINE", raising=False)

    def _raise(version=None):
        raise RuntimeError("download failed")

    monkeypatch.setattr(ts, "download_tokensave", _raise)
    assert ts.ensure_tokensave() is None


def test_pinned_version_env_override(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_TOKENSAVE_VERSION", "v9.9.9")
    assert ts._pinned_version() == "v9.9.9"
    monkeypatch.delenv("HEADROOM_TOKENSAVE_VERSION", raising=False)
    assert ts._pinned_version() == ts.TOKENSAVE_VERSION
