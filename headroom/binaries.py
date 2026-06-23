"""Fetcher for bundled CLI tool binaries.

`pip install headroom-ai` pulls `ast-grep-cli` as a proper PyPI binary wheel
(core dependency), so ast-grep is always on PATH. The other two high-value
tools — `difft` (difftastic) and `scc` — are fetched from pinned upstream
GitHub releases at proxy startup, verified, cached per-user, and exec'd.

Supported platforms: linux (glibc + musl) x86_64/aarch64, macOS x86_64/arm64,
Windows x86_64. Unsupported platforms raise PlatformNotSupported; callers in
the compression pipeline should fall back to their non-accelerated path.

Env vars:
    HEADROOM_BINARIES_MIRROR   base URL that replaces https://github.com
    HEADROOM_BINARIES_CACHE    override cache dir
    HEADROOM_BINARIES_OFFLINE  if set, never reach the network
"""

from __future__ import annotations

import functools
import glob
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "BinaryError",
    "BinaryFetchError",
    "PlatformNotSupported",
    "Sha256Mismatch",
    "OfflineError",
    "PlatformKey",
    "detect_platform",
    "cache_dir",
    "resolve",
    "which",
    "status",
    "ensure_tools",
]


# ---------- Exceptions ---------------------------------------------------- #


class BinaryError(Exception):
    """Base exception for the binaries module."""


class BinaryFetchError(BinaryError):
    """Raised when a download fails or an archive cannot be extracted."""


class PlatformNotSupported(BinaryError):
    """Raised when the current OS/arch is not covered by a tool's registry."""


class Sha256Mismatch(BinaryError):
    """Raised when a downloaded asset's SHA256 does not match the pin."""


class OfflineError(BinaryError):
    """Raised when a network fetch is required but HEADROOM_BINARIES_OFFLINE is set."""


# ---------- Platform detection -------------------------------------------- #


@dataclass(frozen=True)
class PlatformKey:
    os: str  # "linux" | "darwin" | "windows"
    arch: str  # "x86_64" | "aarch64"
    libc: str  # "gnu" | "musl" | "n/a"

    def key(self) -> str:
        # Compact form used as registry lookup key and cache subdirectory.
        if self.os == "linux":
            return f"{self.os}-{self.arch}-{self.libc}"
        return f"{self.os}-{self.arch}"


def _machine_to_arch(machine: str) -> str:
    m = machine.lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    return m  # return as-is; lookup will fail cleanly with PlatformNotSupported


def _is_musl() -> bool:
    """Best-effort musl detection on Linux. Never raises.

    First: ask `ldd --version` (returns 'musl' on Alpine/Void).
    Fallback: check for the musl dynamic loader at /lib/ld-musl-*.so.1,
    which is present on Alpine even when `ldd` is absent.
    """
    try:
        out = subprocess.run(
            ["ldd", "--version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if "musl" in (out.stdout + out.stderr).lower():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    # Fallback: musl ships a distinctive dynamic loader.
    if glob.glob("/lib/ld-musl-*.so.1") or glob.glob("/lib64/ld-musl-*.so.1"):
        return True
    return False


@functools.lru_cache(maxsize=1)
def detect_platform() -> PlatformKey:
    arch = _machine_to_arch(platform.machine())
    if sys.platform.startswith("linux"):
        return PlatformKey("linux", arch, "musl" if _is_musl() else "gnu")
    if sys.platform == "darwin":
        return PlatformKey("darwin", arch, "n/a")
    if sys.platform.startswith("win"):
        return PlatformKey("windows", arch, "n/a")
    return PlatformKey(sys.platform, arch, "n/a")


# ---------- Cache dir ----------------------------------------------------- #


def _is_writable_dir(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return path.is_dir() and bool(mode & 0o222)


def _has_writable_existing_parent(path: Path) -> bool:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return False
        current = parent
    return _is_writable_dir(current)


def cache_dir() -> Path:
    override = os.environ.get("HEADROOM_BINARIES_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "headroom" / "bin"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "headroom" / "bin"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "headroom" / "bin"


# ---------- Registry ------------------------------------------------------ #


_REGISTRY_PATH = Path(__file__).parent / "tools.json"


@functools.lru_cache(maxsize=1)
def _registry() -> dict[str, Any]:
    with _REGISTRY_PATH.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def _tool_entry(tool: str) -> dict[str, Any]:
    reg = _registry()
    tools: dict[str, Any] = reg.get("tools", {})
    if tool not in tools:
        raise KeyError(f"unknown tool {tool!r}; known: {sorted(tools)}")
    entry: dict[str, Any] = tools[tool]
    return entry


def _is_pypi_tool(tool: str) -> bool:
    entry = _tool_entry(tool)
    return entry.get("version") == "pypi" or not entry.get("assets")


def _asset_for_platform(tool: str, plat: PlatformKey) -> dict[str, Any]:
    entry = _tool_entry(tool)
    if _is_pypi_tool(tool):
        raise PlatformNotSupported(
            f"{tool}: distributed via PyPI only; `pip install headroom-ai` "
            f"should have placed `{entry.get('binary', tool)}` on PATH."
        )
    assets: dict[str, Any] = entry.get("assets", {})
    asset: dict[str, Any] | None = assets.get(plat.key())
    if asset is None:
        supported = sorted(assets.keys())
        raise PlatformNotSupported(
            f"{tool}: no prebuilt binary for {plat.key()}; supported: {supported}"
        )
    return asset


def _mirror_url(url: str) -> str:
    mirror = os.environ.get("HEADROOM_BINARIES_MIRROR")
    if not mirror:
        return url
    # Only substitute the github.com host so that paths remain intact.
    for prefix in ("https://github.com", "https://objects.githubusercontent.com"):
        if url.startswith(prefix):
            return mirror.rstrip("/") + url[len(prefix) :]
    return url


# ---------- Download + verify --------------------------------------------- #


def _download(url: str, dest: Path, *, progress: bool = True) -> None:
    if os.environ.get("HEADROOM_BINARIES_OFFLINE"):
        raise OfflineError(f"offline mode (HEADROOM_BINARIES_OFFLINE=1) but fetch required: {url}")
    if not _has_writable_existing_parent(dest.parent):
        raise OSError(f"binary cache directory parent is not writable: {dest.parent}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not _is_writable_dir(dest.parent):
        raise OSError(f"binary cache directory is not writable: {dest.parent}")
    final_url = _mirror_url(url)
    req = urllib.request.Request(final_url, headers={"User-Agent": "headroom-binaries/1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (https)
            total = int(resp.headers.get("Content-Length") or 0)
            _stream_to(resp, dest, total, label=dest.name, show_progress=progress)
    except urllib.error.URLError as e:
        raise BinaryFetchError(f"failed to download {final_url}: {e}") from e


def _stream_to(src: Any, dest: Path, total: int, *, label: str, show_progress: bool) -> None:
    # Rich progress if available and stderr is a tty; otherwise silent chunked copy.
    try:
        if show_progress and sys.stderr.isatty():
            from rich.progress import (
                BarColumn,
                DownloadColumn,
                Progress,
                TextColumn,
                TimeRemainingColumn,
                TransferSpeedColumn,
            )

            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            ) as prog:
                task = prog.add_task(label, total=total or None)
                with dest.open("wb") as out:
                    while chunk := src.read(1024 * 64):
                        out.write(chunk)
                        prog.update(task, advance=len(chunk))
            return
    except ImportError:
        pass
    with dest.open("wb") as out:
        shutil.copyfileobj(src, out)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        # Upstream release not SHA-pinned in registry. HTTPS + the GitHub CDN
        # is the only integrity check. Log at INFO so verbose runs can see
        # this state; `doctor` surfaces the same fact via `sha_pinned=False`.
        logger.info("binary %s downloaded without sha256 pin (HTTPS trust only)", path.name)
        return
    got = _sha256_file(path)
    if got.lower() != expected.lower():
        path.unlink(missing_ok=True)
        raise Sha256Mismatch(f"sha256 mismatch for {path.name}: expected {expected}, got {got}")


# ---------- Archive extraction ------------------------------------------- #


def _extract(archive: Path, member: str, dest: Path) -> None:
    """Extract `member` from archive into `dest` (single-file binary)."""
    if not _has_writable_existing_parent(dest.parent):
        raise OSError(f"binary cache directory parent is not writable: {dest.parent}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not _is_writable_dir(dest.parent):
        raise OSError(f"binary cache directory is not writable: {dest.parent}")
    name = archive.name.lower()
    try:
        if name.endswith(".tar.gz") or name.endswith(".tgz"):
            with tarfile.open(archive, "r:gz") as tf:
                _extract_member_from_tar(tf, member, dest)
        elif name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                _extract_member_from_zip(zf, member, dest)
        elif name.endswith(".gz") and not (name.endswith(".tar.gz") or name.endswith(".tgz")):
            # bare .gz of a single binary (e.g. `scc-linux-x86_64.gz`)
            import gzip

            with gzip.open(archive, "rb") as gz, dest.open("wb") as out:
                shutil.copyfileobj(gz, out)
        else:
            # Not an archive — treat the downloaded file itself as the binary.
            shutil.copy2(archive, dest)
    except (tarfile.TarError, zipfile.BadZipFile, OSError) as e:
        raise BinaryFetchError(f"failed to extract {archive.name}: {e}") from e


def _extract_member_from_tar(tf: tarfile.TarFile, member: str, dest: Path) -> None:
    # Match by basename so that registries can specify "difft" even though the
    # upstream tar may include a leading directory like "difft-0.64.0/difft".
    wanted = member.lower()
    for m in tf.getmembers():
        base = m.name.rsplit("/", 1)[-1].lower()
        if base == wanted and m.isfile():
            extracted = tf.extractfile(m)
            if extracted is None:
                continue
            with dest.open("wb") as out:
                shutil.copyfileobj(extracted, out)
            return
    raise BinaryFetchError(f"archive did not contain expected member {member!r}")


def _extract_member_from_zip(zf: zipfile.ZipFile, member: str, dest: Path) -> None:
    wanted = member.lower()
    for info in zf.infolist():
        base = info.filename.rsplit("/", 1)[-1].lower()
        if base == wanted and not info.is_dir():
            with zf.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            return
    raise BinaryFetchError(f"archive did not contain expected member {member!r}")


# ---------- Public API ---------------------------------------------------- #


def _binary_name(tool: str, plat: PlatformKey) -> str:
    entry = _tool_entry(tool)
    base = entry.get("binary", tool)
    return f"{base}.exe" if plat.os == "windows" else base


def _cached_path(tool: str, version: str, plat: PlatformKey) -> Path:
    return cache_dir() / f"{tool}-{version}-{plat.key()}" / _binary_name(tool, plat)


def _in_registry(tool: str) -> bool:
    return tool in _registry().get("tools", {})


def _path_lookup(tool: str) -> Path | None:
    """Find `tool` on PATH or in this interpreter's Scripts/bin directory.

    PyPI binary wheels (e.g. ast-grep-cli) install their console scripts into
    sys.prefix/bin (or sys.prefix/Scripts on Windows). That directory is on
    PATH when the venv is activated, but subprocesses started by a non-active
    interpreter can miss it, so we check it explicitly as a fallback.
    """
    candidates = [tool]
    if _in_registry(tool):
        alias = _tool_entry(tool).get("binary")
        if alias and alias != tool:
            candidates.append(alias)

    for name in candidates:
        found = shutil.which(name)
        if found:
            return Path(found)

    scripts_dir = Path(sys.prefix) / ("Scripts" if sys.platform.startswith("win") else "bin")
    for name in candidates:
        exe = scripts_dir / (name + (".exe" if sys.platform.startswith("win") else ""))
        if exe.exists():
            return exe
    return None


def which(tool: str) -> Path | None:
    """Return a path to `tool` if it is on PATH or already cached, else None.

    Never triggers a network fetch. Callers that want the tool to be installed
    on demand should use `resolve()` instead.
    """
    on_path = _path_lookup(tool)
    if on_path:
        return on_path
    if not _in_registry(tool):
        return None
    try:
        plat = detect_platform()
        _asset_for_platform(tool, plat)  # raises if unsupported
    except PlatformNotSupported:
        return None
    path = _cached_path(tool, _tool_entry(tool)["version"], plat)
    return path if path.exists() else None


def resolve(tool: str) -> Path:
    """Return a path to the tool binary, fetching it on first use.

    Raises PlatformNotSupported if the tool is unavailable on this platform,
    OfflineError if a fetch is required but HEADROOM_BINARIES_OFFLINE is set,
    Sha256Mismatch if verification fails, BinaryFetchError on other IO errors.
    """
    on_path = _path_lookup(tool)
    if on_path:
        return on_path
    if not _in_registry(tool):
        raise KeyError(f"unknown tool {tool!r}")

    plat = detect_platform()
    entry = _tool_entry(tool)
    asset = _asset_for_platform(tool, plat)
    version = entry["version"]
    binary_path = _cached_path(tool, version, plat)
    if binary_path.exists():
        return binary_path

    # Not cached — fetch, verify, extract.
    url = asset["url"]
    sha256 = asset.get("sha256")
    member = asset.get("member", _binary_name(tool, plat))

    with tempfile.TemporaryDirectory(prefix="headroom-fetch-") as tmp:
        tmp_dir = Path(tmp)
        # Strip query params so mirror URLs like `.../difft.tar.gz?token=...`
        # don't produce filenames that break archive-type detection.
        url_path = urllib.parse.urlparse(url).path
        download_path = tmp_dir / (Path(url_path).name or "download")
        _download(url, download_path)
        _verify_sha256(download_path, sha256)
        staging = tmp_dir / "out"
        _extract(download_path, member, staging)
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic move with a PID-scoped partial name so two concurrent
        # `headroom proxy` starts don't race on the same .partial path.
        tmp_final = binary_path.with_name(f"{binary_path.name}.{os.getpid()}.partial")
        shutil.move(str(staging), tmp_final)
        try:
            tmp_final.chmod(0o755)
        except OSError as e:
            if not sys.platform.startswith("win"):
                logger.warning("chmod +x failed for %s: %s", tmp_final, e)
            # On Windows the .exe is already executable; elsewhere we logged.
        os.replace(tmp_final, binary_path)
    return binary_path


def ensure_tools(quiet: bool = False) -> dict[str, Path | None]:
    if not _has_writable_existing_parent(cache_dir()):
        return {name: which(name) for name in _registry().get("tools", {})}
    """Install every tool in the registry if missing. Safe to call repeatedly.

    Called at proxy startup and on first `headroom` CLI invocation so that no
    tool fetch ever happens inside a live request. Skips tools that are on
    PATH, already cached, or distributed via PyPI-only (ast-grep).

    Returns a map of tool_name -> resolved Path (or None if unsupported).
    Never raises; unsupported platforms or offline errors are logged via
    stderr and the tool is skipped.
    """
    out: dict[str, Path | None] = {}
    for name in _registry().get("tools", {}):
        try:
            if _is_pypi_tool(name):
                # ast-grep ships via pip; just record whether it's on PATH.
                out[name] = _path_lookup(name)
                continue
            out[name] = resolve(name)
        except (
            PlatformNotSupported,
            OfflineError,
            BinaryFetchError,
            Sha256Mismatch,
            # Catch readonly / sandboxed filesystems (e.g. containerized
            # home dirs) so proxy startup never fails because the cache dir
            # can't be created. The interceptor fall back to no-op.
            OSError,
        ) as e:
            out[name] = None
            if not quiet:
                print(f"headroom: skipping {name}: {e}", file=sys.stderr)
    return out


def status() -> list[dict[str, Any]]:
    """Return a list of status dicts for every tool in the registry.

    Used by `headroom tools doctor`. Never fetches — only inspects.
    """
    out: list[dict[str, Any]] = []
    plat = detect_platform()
    for name, entry in _registry().get("tools", {}).items():
        row: dict[str, Any] = {
            "tool": name,
            "version": entry.get("version"),
            "platform": plat.key(),
            "source": entry.get("source", "fetched"),
            "path": None,
            "state": "missing",
        }
        # Honor PATH.
        on_path = shutil.which(name) or (
            shutil.which(entry["binary"]) if entry.get("binary") else None
        )
        if on_path:
            row["path"] = on_path
            row["state"] = "on-path"
            out.append(row)
            continue
        try:
            asset = _asset_for_platform(name, plat)
        except PlatformNotSupported as e:
            row["state"] = "unsupported-platform"
            row["detail"] = str(e)
            out.append(row)
            continue
        row["sha_pinned"] = bool(asset.get("sha256"))
        cached = _cached_path(name, entry["version"], plat)
        if cached.exists():
            row["path"] = str(cached)
            row["state"] = "cached"
        out.append(row)
    return out
