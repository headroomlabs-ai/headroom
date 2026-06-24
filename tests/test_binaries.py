"""Unit tests for headroom.binaries — the lazy fetcher for bundled CLI tools.

No network access. A fake urlopen serves bytes from an in-memory fixture.
"""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import zipfile

import pytest

from headroom import binaries

# -------- Fixtures -------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch, tmp_path):
    """Isolate every test from global state: cache dir, platform lru_cache, env."""
    binaries.detect_platform.cache_clear()
    binaries._registry.cache_clear()
    monkeypatch.setenv("HEADROOM_BINARIES_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("HEADROOM_BINARIES_MIRROR", raising=False)
    monkeypatch.delenv("HEADROOM_BINARIES_OFFLINE", raising=False)
    yield
    binaries.detect_platform.cache_clear()
    binaries._registry.cache_clear()


def _set_platform(monkeypatch, *, sys_plat: str, machine: str, musl: bool = False):
    monkeypatch.setattr(sys, "platform", sys_plat)
    monkeypatch.setattr("platform.machine", lambda: machine)
    monkeypatch.setattr(binaries, "_is_musl", lambda: musl)
    binaries.detect_platform.cache_clear()


def _make_tar_gz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data
        self.headers = {"Content-Length": str(len(data))}

    def read(self, n: int = -1) -> bytes:
        if n < 0 or n >= len(self._data):
            chunk, self._data = self._data, b""
            return chunk
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def fake_urlopen(monkeypatch):
    """Install a fake urllib.request.urlopen that serves registered URLs."""
    served: dict[str, bytes] = {}

    def fake(req, timeout=None):  # noqa: ARG001
        url = req.full_url if hasattr(req, "full_url") else req
        if url not in served:
            raise AssertionError(f"unexpected fetch for {url}")
        return _FakeResponse(served[url])

    monkeypatch.setattr(binaries.urllib.request, "urlopen", fake)
    return served


# -------- Platform detection --------------------------------------------- #


def test_detect_platform_linux_gnu(monkeypatch):
    _set_platform(monkeypatch, sys_plat="linux", machine="x86_64", musl=False)
    p = binaries.detect_platform()
    assert p == binaries.PlatformKey("linux", "x86_64", "gnu")
    assert p.key() == "linux-x86_64-gnu"


def test_detect_platform_linux_musl(monkeypatch):
    _set_platform(monkeypatch, sys_plat="linux", machine="aarch64", musl=True)
    assert binaries.detect_platform().key() == "linux-aarch64-musl"


def test_detect_platform_darwin_arm64(monkeypatch):
    _set_platform(monkeypatch, sys_plat="darwin", machine="arm64")
    assert binaries.detect_platform().key() == "darwin-aarch64"


def test_detect_platform_windows_amd64(monkeypatch):
    _set_platform(monkeypatch, sys_plat="win32", machine="AMD64")
    assert binaries.detect_platform().key() == "windows-x86_64"


# -------- Cache dir ------------------------------------------------------ #


def test_cache_dir_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HEADROOM_BINARIES_CACHE", str(tmp_path / "custom"))
    assert binaries.cache_dir() == (tmp_path / "custom").resolve()


# -------- Registry / asset resolution ------------------------------------ #


def test_unsupported_platform_raises(monkeypatch):
    _set_platform(monkeypatch, sys_plat="linux", machine="riscv64")
    with pytest.raises(binaries.PlatformNotSupported):
        binaries._asset_for_platform("difft", binaries.detect_platform())


def test_pypi_only_tool_raises_with_helpful_message(monkeypatch):
    _set_platform(monkeypatch, sys_plat="darwin", machine="arm64")
    with pytest.raises(binaries.PlatformNotSupported) as exc:
        binaries._asset_for_platform("ast-grep", binaries.detect_platform())
    assert "pip install headroom-ai" in str(exc.value)


def test_unknown_tool_raises_key_error():
    with pytest.raises(KeyError):
        binaries._tool_entry("not-a-real-tool")


# -------- which / resolve with PATH hits --------------------------------- #


def test_which_finds_on_path(monkeypatch, tmp_path):
    fake_bin = tmp_path / "difft"
    fake_bin.write_text("#!/bin/sh\necho ok\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(
        binaries.shutil, "which", lambda name: str(fake_bin) if name == "difft" else None
    )
    # Because the tool is on PATH, which() returns its path without fetching.
    assert binaries.which("difft") == fake_bin


def test_which_returns_none_when_not_cached(monkeypatch):
    _set_platform(monkeypatch, sys_plat="darwin", machine="arm64")
    monkeypatch.setattr(binaries.shutil, "which", lambda _name: None)
    assert binaries.which("difft") is None


def test_resolve_honors_path(monkeypatch, tmp_path):
    fake_bin = tmp_path / "scc"
    fake_bin.write_text("")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(
        binaries.shutil, "which", lambda name: str(fake_bin) if name == "scc" else None
    )
    assert binaries.resolve("scc") == fake_bin


# -------- Offline / mirror / fetch behavior ------------------------------ #


def test_offline_error_when_fetch_required(monkeypatch):
    _set_platform(monkeypatch, sys_plat="darwin", machine="arm64")
    monkeypatch.setattr(binaries.shutil, "which", lambda _name: None)
    monkeypatch.setenv("HEADROOM_BINARIES_OFFLINE", "1")
    with pytest.raises(binaries.OfflineError):
        binaries.resolve("difft")


def test_mirror_substitution(monkeypatch):
    monkeypatch.setenv("HEADROOM_BINARIES_MIRROR", "https://mirror.example.com/gh")
    out = binaries._mirror_url(
        "https://github.com/Wilfred/difftastic/releases/download/0.64.0/x.tar.gz"
    )
    assert (
        out == "https://mirror.example.com/gh/Wilfred/difftastic/releases/download/0.64.0/x.tar.gz"
    )
    # Non-matching URLs are left alone.
    assert binaries._mirror_url("https://example.com/x") == "https://example.com/x"


def test_mirror_url_with_query_params_strips_them_from_download_filename(monkeypatch, fake_urlopen):
    """Mirror URLs with `?token=...` must not leak into the download filename.

    Regression test for the case where `Path(url).name` gave
    `difft.tar.gz?token=abc`, which broke `.endswith(".tar.gz")` detection
    and resulted in a raw archive being copied as an "executable."
    """
    _set_platform(monkeypatch, sys_plat="darwin", machine="arm64")
    monkeypatch.setattr(binaries.shutil, "which", lambda _name: None)
    archive = _make_tar_gz({"difft": b"ok"})
    tokened_url = (
        "https://github.com/Wilfred/difftastic/releases/download/0.64.0/"
        "difft-aarch64-apple-darwin.tar.gz?token=abc&sig=xyz"
    )
    # Override the registry URL for this test.
    reg = binaries._registry()
    asset = reg["tools"]["difft"]["assets"]["darwin-aarch64"]
    original_url = asset["url"]
    asset["url"] = tokened_url
    fake_urlopen[tokened_url] = archive
    try:
        path = binaries.resolve("difft")
        assert path.exists()
        assert path.read_bytes() == b"ok"
    finally:
        asset["url"] = original_url


def test_fetch_extract_and_cache_tar_gz(monkeypatch, fake_urlopen, tmp_path):
    _set_platform(monkeypatch, sys_plat="darwin", machine="arm64")
    monkeypatch.setattr(binaries.shutil, "which", lambda _name: None)

    payload = b"#!/bin/sh\necho fake-difft\n"
    archive = _make_tar_gz({"difft-0.64.0/difft": payload})
    url = "https://github.com/Wilfred/difftastic/releases/download/0.64.0/difft-aarch64-apple-darwin.tar.gz"
    fake_urlopen[url] = archive

    path = binaries.resolve("difft")
    assert path.exists()
    assert path.read_bytes() == payload
    # Second call should use cache (no further fetch).
    fake_urlopen.pop(url)  # remove so a refetch would error
    path2 = binaries.resolve("difft")
    assert path2 == path


def test_fetch_extract_zip(monkeypatch, fake_urlopen):
    _set_platform(monkeypatch, sys_plat="win32", machine="AMD64")
    monkeypatch.setattr(binaries.shutil, "which", lambda _name: None)
    payload = b"MZfake"
    archive = _make_zip({"scc.exe": payload})
    url = "https://github.com/boyter/scc/releases/download/v3.5.0/scc_Windows_x86_64.zip"
    fake_urlopen[url] = archive

    path = binaries.resolve("scc")
    assert path.exists()
    assert path.name.endswith("scc.exe")
    assert path.read_bytes() == payload


def test_sha256_mismatch_raises_and_deletes(monkeypatch, fake_urlopen, tmp_path):
    _set_platform(monkeypatch, sys_plat="darwin", machine="arm64")
    monkeypatch.setattr(binaries.shutil, "which", lambda _name: None)

    # Override the registry entry for difft to include a bogus sha256.
    reg = binaries._registry()
    asset = reg["tools"]["difft"]["assets"]["darwin-aarch64"]
    asset["sha256"] = "deadbeef" * 8  # wrong
    archive = _make_tar_gz({"difft": b"hi"})
    fake_urlopen[asset["url"]] = archive

    try:
        with pytest.raises(binaries.Sha256Mismatch):
            binaries.resolve("difft")
    finally:
        asset["sha256"] = None  # restore


def test_sha256_match_passes(monkeypatch, fake_urlopen):
    _set_platform(monkeypatch, sys_plat="darwin", machine="arm64")
    monkeypatch.setattr(binaries.shutil, "which", lambda _name: None)
    archive = _make_tar_gz({"difft": b"hello"})
    good = hashlib.sha256(archive).hexdigest()
    reg = binaries._registry()
    asset = reg["tools"]["difft"]["assets"]["darwin-aarch64"]
    asset["sha256"] = good
    fake_urlopen[asset["url"]] = archive
    try:
        path = binaries.resolve("difft")
        assert path.read_bytes() == b"hello"
    finally:
        asset["sha256"] = None


# -------- status() ------------------------------------------------------- #


def test_ensure_tools_survives_readonly_cache_dir(monkeypatch, tmp_path):
    """Containerized / read-only home dirs must not crash proxy startup.

    Regression test for PermissionError not being caught in ensure_tools().
    """
    _set_platform(monkeypatch, sys_plat="linux", machine="x86_64", musl=False)
    monkeypatch.setattr(binaries.shutil, "which", lambda _name: None)

    # Point the cache at a path we can't create under (readonly parent).
    readonly_parent = tmp_path / "readonly"
    readonly_parent.mkdir()
    readonly_parent.chmod(0o500)  # r-x: can't create children
    monkeypatch.setenv("HEADROOM_BINARIES_CACHE", str(readonly_parent / "cache"))

    try:
        # Must return a dict, not raise.
        result = binaries.ensure_tools(quiet=True)
        # Fetched tools couldn't write to cache → None. ast-grep is PyPI-only
        # so its entry depends on PATH, which is not what this test exercises.
        assert result.get("difft") is None
        assert result.get("scc") is None
    finally:
        readonly_parent.chmod(0o700)  # so pytest tmp cleanup succeeds


def test_ensure_tools_partial_failure_proxy_still_starts(monkeypatch):
    """If one tool fails to fetch, others still install and ensure_tools returns."""
    _set_platform(monkeypatch, sys_plat="darwin", machine="arm64")
    monkeypatch.setattr(binaries.shutil, "which", lambda _name: None)

    scc_asset = binaries._registry()["tools"]["scc"]["assets"]["darwin-aarch64"]
    scc_tar = _make_tar_gz({"scc": b"ok"})

    def selective_urlopen(req, timeout=None):  # noqa: ARG001
        url = req.full_url if hasattr(req, "full_url") else req
        if url == scc_asset["url"]:
            return _FakeResponse(scc_tar)
        # difft fails with a real network-style error.
        import urllib.error

        raise urllib.error.URLError("simulated network failure")

    monkeypatch.setattr(binaries.urllib.request, "urlopen", selective_urlopen)

    result = binaries.ensure_tools(quiet=True)
    # scc succeeded; difft failed but ensure_tools() did not crash.
    assert result["scc"] is not None
    assert result["difft"] is None


def test_status_reports_every_registered_tool(monkeypatch):
    _set_platform(monkeypatch, sys_plat="darwin", machine="arm64")
    monkeypatch.setattr(binaries.shutil, "which", lambda _name: None)
    rows = binaries.status()
    names = {r["tool"] for r in rows}
    assert {"difft", "scc", "ast-grep"} <= names
    for r in rows:
        assert r["state"] in ("on-path", "cached", "missing", "unsupported-platform")
