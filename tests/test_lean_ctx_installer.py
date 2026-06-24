"""Tests for managed lean-ctx installation."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from headroom.lean_ctx import get_lean_ctx_path, installer


def test_get_lean_ctx_path_finds_windows_managed_binary(tmp_path: Path) -> None:
    managed_dir = tmp_path / ".headroom" / "bin"
    managed_dir.mkdir(parents=True)
    managed_path = managed_dir / "lean-ctx.exe"
    managed_path.write_bytes(b"binary")

    with patch("headroom.lean_ctx.LEAN_CTX_BIN_DIR", managed_dir):
        with patch("headroom.lean_ctx.LEAN_CTX_BIN_PATH", managed_dir / "lean-ctx"):
            with patch("headroom.lean_ctx.shutil.which", return_value=None):
                assert get_lean_ctx_path() == managed_path


def test_get_target_triple_uses_override(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_LEAN_CTX_TARGET", "x86_64-pc-windows-msvc")
    assert installer._get_target_triple() == "x86_64-pc-windows-msvc"


def test_detect_runtime_target_triple_handles_linux_gnu() -> None:
    with patch.object(installer.platform, "system", return_value="Linux"):
        with patch.object(installer.platform, "machine", return_value="x86_64"):
            with patch.object(installer, "_is_musl", return_value=False):
                assert installer._detect_runtime_target_triple() == "x86_64-unknown-linux-gnu"


def test_detect_runtime_target_triple_handles_linux_musl_arm() -> None:
    with patch.object(installer.platform, "system", return_value="Linux"):
        with patch.object(installer.platform, "machine", return_value="aarch64"):
            with patch.object(installer, "_is_musl", return_value=True):
                assert installer._detect_runtime_target_triple() == "aarch64-unknown-linux-musl"


def test_get_download_url_uses_windows_zip(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_LEAN_CTX_TARGET", raising=False)
    monkeypatch.setenv("LEAN_CTX_TARGET", "x86_64-pc-windows-msvc")

    url, ext = installer._get_download_url("v1.2.3")

    assert url == f"{installer.GITHUB_RELEASE_URL}/v1.2.3/lean-ctx-x86_64-pc-windows-msvc.zip"
    assert ext == "zip"
    assert installer._binary_name_for_target("x86_64-pc-windows-msvc") == "lean-ctx.exe"


def test_download_lean_ctx_skips_verify_for_non_native_target(monkeypatch, tmp_path: Path) -> None:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="lean-ctx")
        payload = b"fake-binary"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    archive_bytes = archive.getvalue()

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return archive_bytes

    monkeypatch.setenv("HEADROOM_LEAN_CTX_TARGET", "x86_64-apple-darwin")

    with patch.object(installer, "LEAN_CTX_BIN_DIR", tmp_path):
        with patch.object(installer, "urlopen", return_value=_Response()):
            with patch.object(installer.subprocess, "run") as subprocess_run:
                installed_path = installer.download_lean_ctx("v3.4.7")

    assert installed_path == tmp_path / "lean-ctx"
    assert installed_path.exists()
    subprocess_run.assert_not_called()


def test_download_lean_ctx_extracts_zip_binary(monkeypatch, tmp_path: Path) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w") as zf:
        zf.writestr("lean-ctx.exe", b"fake-windows-binary")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return archive.getvalue()

    monkeypatch.setenv("HEADROOM_LEAN_CTX_TARGET", "x86_64-pc-windows-msvc")

    with patch.object(installer, "LEAN_CTX_BIN_DIR", tmp_path):
        with patch.object(installer, "urlopen", return_value=_Response()):
            installed_path = installer.download_lean_ctx("v3.4.7")

    assert installed_path == tmp_path / "lean-ctx.exe"
    assert installed_path.read_bytes() == b"fake-windows-binary"


def test_download_lean_ctx_verifies_native_target(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HEADROOM_LEAN_CTX_TARGET", raising=False)
    monkeypatch.delenv("LEAN_CTX_TARGET", raising=False)

    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="dist/lean-ctx")
        payload = b"fake-native-binary"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return archive.getvalue()

    run_result = SimpleNamespace(returncode=0, stdout="lean-ctx 3.4.7", stderr="")

    with patch.object(installer, "LEAN_CTX_BIN_DIR", tmp_path):
        with patch.object(
            installer, "_detect_runtime_target_triple", return_value="x86_64-unknown-linux-gnu"
        ):
            with patch.object(installer, "urlopen", return_value=_Response()):
                with patch.object(
                    installer.subprocess, "run", return_value=run_result
                ) as subprocess_run:
                    installed_path = installer.download_lean_ctx("v3.4.7")

    assert installed_path == tmp_path / "lean-ctx"
    subprocess_run.assert_called_once_with(
        [str(installed_path), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
    )


def test_download_lean_ctx_rejects_invalid_download_url(tmp_path: Path) -> None:
    with patch.object(installer, "LEAN_CTX_BIN_DIR", tmp_path):
        with patch.object(installer, "_get_target_triple", return_value="x86_64-unknown-linux-gnu"):
            with patch.object(
                installer,
                "_get_download_url",
                return_value=("file:///tmp/lean-ctx.tar.gz", "tar.gz"),
            ):
                with pytest.raises(RuntimeError, match="Invalid URL scheme"):
                    installer.download_lean_ctx("v3.4.7")


def test_ensure_lean_ctx_returns_none_when_download_fails() -> None:
    with patch("headroom.lean_ctx.get_lean_ctx_path", return_value=None):
        with patch.object(
            installer, "download_lean_ctx", side_effect=RuntimeError("download failed")
        ):
            assert installer.ensure_lean_ctx() is None
