"""Locating the codebase-memory-mcp binary for the code-graph watcher."""

from __future__ import annotations

import platform
import shutil
import stat
from pathlib import Path

from headroom.graph import installer


def test_installed_binary_is_found_by_a_real_path_lookup(monkeypatch, tmp_path: Path) -> None:
    """A binary named the way ``get_cbm_path`` expects must be what
    ``shutil.which`` resolves on this host.

    Nothing about the file name is mocked: on Windows this needs the .exe
    extension, elsewhere the extensionless name with its executable bit.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    path = bin_dir / installer._installed_name(platform.system().lower())
    path.write_bytes(b"#!/bin/sh\necho codebase-memory-mcp test\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setattr(installer, "CBM_BIN_DIR", bin_dir)

    monkeypatch.setenv("PATH", str(bin_dir))
    found = shutil.which("codebase-memory-mcp")
    assert found is not None
    assert Path(found).samefile(path)
    assert installer.get_cbm_path() == Path(found)


def test_get_cbm_path_finds_the_installed_binary_off_path(monkeypatch, tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))
    monkeypatch.setattr(installer, "CBM_BIN_DIR", bin_dir)
    monkeypatch.setattr(installer.platform, "system", lambda: "Windows")
    (bin_dir / "codebase-memory-mcp.exe").write_bytes(b"MZ fake binary")

    assert installer.get_cbm_path() == bin_dir / "codebase-memory-mcp.exe"

    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    assert installer.get_cbm_path() is None
    (bin_dir / "codebase-memory-mcp").write_bytes(b"#!/bin/sh\n")
    assert installer.get_cbm_path() == bin_dir / "codebase-memory-mcp"
