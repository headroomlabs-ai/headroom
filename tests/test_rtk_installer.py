"""Tests for host-target rtk installation overrides."""

from __future__ import annotations

import io
import stat
import tarfile
from pathlib import Path
from unittest.mock import patch

from headroom.rtk import get_rtk_path, installer


def test_get_rtk_path_finds_windows_managed_binary(tmp_path: Path) -> None:
    managed_dir = tmp_path / ".headroom" / "bin"
    managed_dir.mkdir(parents=True)
    managed_path = managed_dir / "rtk.exe"
    managed_path.write_bytes(b"binary")

    with patch("headroom.rtk.RTK_BIN_DIR", managed_dir):
        with patch("headroom.rtk.RTK_BIN_PATH", managed_dir / "rtk"):
            with patch("headroom.rtk.shutil.which", return_value=None):
                assert get_rtk_path() == managed_path


def test_get_target_triple_uses_override(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_RTK_TARGET", "x86_64-pc-windows-msvc")
    assert installer._get_target_triple() == "x86_64-pc-windows-msvc"


def test_download_rtk_skips_verify_for_non_native_target(monkeypatch, tmp_path: Path) -> None:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="rtk")
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

    monkeypatch.setenv("HEADROOM_RTK_TARGET", "x86_64-apple-darwin")

    with patch.object(installer, "RTK_BIN_DIR", tmp_path):
        with patch.object(installer, "urlopen", return_value=_Response()):
            with patch.object(installer.subprocess, "run") as subprocess_run:
                installed_path = installer.download_rtk("v0.28.2")

    assert installed_path == tmp_path / "rtk"
    assert installed_path.exists()
    subprocess_run.assert_not_called()


def test_register_claude_hooks_survives_forked_daemon(tmp_path: Path) -> None:
    """rtk init that exits fast but leaves a child holding stdout must not hang.

    Regression: capturing through pipes made subprocess.run drain until EOF,
    which a lingering grandchild deferred past the 10s timeout even though the
    hooks were already registered. Output now goes to a temp file, so we wait
    only on the direct child.
    """
    fake_rtk = tmp_path / "rtk"
    fake_rtk.write_text("#!/bin/bash\n( sleep 30 ) &\necho done\nexit 0\n")
    fake_rtk.chmod(fake_rtk.stat().st_mode | stat.S_IEXEC)

    assert installer.register_claude_hooks(fake_rtk) is True
