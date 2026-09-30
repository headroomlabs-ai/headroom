"""Release asset selection and extraction for the codebase-memory-mcp installer."""

from __future__ import annotations

import io
import shutil
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom import binaries
from headroom.graph import installer


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _zip_archive(member_name: str, data: bytes = b"MZ fake binary") -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(member_name, data)
    return payload.getvalue()


@pytest.mark.parametrize(
    ("plat", "extension"),
    [
        ("darwin-arm64", ".tar.gz"),
        ("darwin-amd64", ".tar.gz"),
        ("linux-arm64", ".tar.gz"),
        ("linux-amd64", ".tar.gz"),
        ("windows-amd64", ".zip"),
    ],
)
def test_asset_filename_matches_the_registry_asset(plat: str, extension: str) -> None:
    filename = installer._asset_filename(plat)

    assert filename == f"codebase-memory-mcp-{plat}{extension}"
    pinned_urls = {
        asset["url"] for asset in binaries._tool_entry("codebase-memory-mcp")["assets"].values()
    }
    assert f"{installer.GITHUB_RELEASE_URL}/{installer.CBM_VERSION}/{filename}" in pinned_urls


@pytest.mark.parametrize("member_name", ["codebase-memory-mcp.exe", "codebase-memory-mcp"])
def test_download_cbm_on_windows_fetches_and_extracts_the_zip(
    monkeypatch, tmp_path: Path, member_name: str
) -> None:
    monkeypatch.setenv("HEADROOM_BINARIES_ALLOW_UNVERIFIED", "1")
    monkeypatch.setattr(installer, "CBM_BIN_DIR", tmp_path)
    monkeypatch.setattr(installer, "_detect_platform", lambda: "windows-amd64")
    requested: list[str] = []

    def fake_urlopen(url: str, timeout: int = 60) -> FakeResponse:
        requested.append(url)
        return FakeResponse(_zip_archive(member_name))

    monkeypatch.setattr(installer, "urlopen", fake_urlopen)
    probes: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> SimpleNamespace:
        probes.append(command)
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr("headroom._subprocess.run", fake_run)

    path = installer.download_cbm()

    assert requested == [
        f"{installer.GITHUB_RELEASE_URL}/{installer.CBM_VERSION}/"
        "codebase-memory-mcp-windows-amd64.zip"
    ]
    # A PATH lookup on Windows only finds the binary through its .exe extension.
    assert path == tmp_path / "codebase-memory-mcp.exe"
    assert path.read_bytes() == b"MZ fake binary"
    assert probes == [[str(path), "--version"]]


def _host_release_archive(filename: str) -> bytes:
    """Build a release archive in the format ``download_cbm`` requests on this host."""
    if filename.endswith(".zip"):
        return _zip_archive("codebase-memory-mcp.exe")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        data = b"#!/bin/sh\necho codebase-memory-mcp test\n"
        info = tarfile.TarInfo(name="codebase-memory-mcp")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return payload.getvalue()


def test_installed_binary_is_found_by_a_real_path_lookup(monkeypatch, tmp_path: Path) -> None:
    """The installed file must be what ``shutil.which`` resolves on this host.

    Nothing about the file name is mocked: on Windows this needs the .exe
    extension, elsewhere the extensionless name with its executable bit.
    """
    bin_dir = tmp_path / "bin"
    monkeypatch.setenv("HEADROOM_BINARIES_ALLOW_UNVERIFIED", "1")
    monkeypatch.setattr(installer, "CBM_BIN_DIR", bin_dir)
    monkeypatch.setattr(
        installer,
        "urlopen",
        lambda url, timeout=60: FakeResponse(_host_release_archive(url.rsplit("/", 1)[-1])),
    )
    probes: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> SimpleNamespace:
        probes.append(command)
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr("headroom._subprocess.run", fake_run)

    path = installer.download_cbm()

    assert probes == [[str(path), "--version"]]
    monkeypatch.setenv("PATH", str(bin_dir))
    found = shutil.which("codebase-memory-mcp")
    assert found is not None
    assert Path(found).samefile(path)
    assert Path(probes[0][0]).samefile(found)
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


def test_download_cbm_zip_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEADROOM_BINARIES_ALLOW_UNVERIFIED", "1")
    monkeypatch.setattr(installer, "CBM_BIN_DIR", tmp_path)
    monkeypatch.setattr(installer, "_detect_platform", lambda: "windows-amd64")

    monkeypatch.setattr(
        installer, "urlopen", lambda url, timeout=60: FakeResponse(_zip_archive("README.md"))
    )
    with pytest.raises(RuntimeError, match="binary not found in archive"):
        installer.download_cbm()

    monkeypatch.setattr(installer, "urlopen", lambda url, timeout=60: FakeResponse(b"not a zip"))
    with pytest.raises(RuntimeError, match="Failed to extract archive"):
        installer.download_cbm()
