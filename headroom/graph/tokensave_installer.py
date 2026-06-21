"""Download and install the ``tokensave`` binary from GitHub releases.

tokensave (https://github.com/aovestdipaperino/tokensave) is the primary
coding-task compressor: a local semantic code-graph MCP server. It is a
single self-contained Rust binary, so — like ``codebase-memory-mcp`` and
``rtk`` — Headroom fetches the prebuilt release asset for the current
platform, caches it under ``~/.local/bin``, and registers it as an MCP
server.

Release-binary only. tokensave is also published to crates.io
(``cargo install tokensave``), but we never shell out to cargo here: a
multi-minute compile is the wrong thing to trigger from ``headroom wrap``.
When no prebuilt asset exists for the platform (e.g. x86_64 macOS, which
tokensave does not currently publish) or the download fails, this module
returns ``None`` and the caller falls back to Serena, the backup compressor.

Env vars:
    HEADROOM_BINARIES_OFFLINE  if set, never reach the network (returns the
                               already-installed binary or ``None``).
    HEADROOM_TOKENSAVE_VERSION override the pinned release tag.
"""

from __future__ import annotations

import io
import logging
import os
import platform
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)

#: Pinned release. Override with HEADROOM_TOKENSAVE_VERSION.
TOKENSAVE_VERSION = "v6.4.4"
TOKENSAVE_REPO = "aovestdipaperino/tokensave"
TOKENSAVE_BIN_DIR = Path.home() / ".local" / "bin"
TOKENSAVE_BIN_NAME = "tokensave"

GITHUB_RELEASE_URL = f"https://github.com/{TOKENSAVE_REPO}/releases/download"


def _pinned_version() -> str:
    return os.environ.get("HEADROOM_TOKENSAVE_VERSION", "").strip() or TOKENSAVE_VERSION


def _detect_asset(version: str) -> tuple[str, str] | None:
    """Return ``(asset_filename, archive_kind)`` for this platform.

    ``archive_kind`` is ``"tar.gz"`` or ``"zip"``. Returns ``None`` when
    tokensave publishes no prebuilt asset for the current platform (the
    caller then falls back to Serena). Release assets are named
    ``tokensave-<version>-<arch>-<os>.<ext>``.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        if machine == "arm64":
            return f"tokensave-{version}-aarch64-macos.tar.gz", "tar.gz"
        # No x86_64-macos release asset is published — fall back to Serena.
        return None
    if system == "linux":
        arch = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
        return f"tokensave-{version}-{arch}-linux.tar.gz", "tar.gz"
    if system == "windows":
        return f"tokensave-{version}-x86_64-windows.zip", "zip"

    return None


def get_tokensave_path() -> Path | None:
    """Find the tokensave binary on PATH or in our install dir; else ``None``."""
    import shutil

    found = shutil.which(TOKENSAVE_BIN_NAME)
    if found:
        return Path(found)

    for name in (TOKENSAVE_BIN_NAME, f"{TOKENSAVE_BIN_NAME}.exe"):
        installed = TOKENSAVE_BIN_DIR / name
        if installed.exists() and installed.is_file():
            return installed

    return None


def download_tokensave(version: str | None = None) -> Path:
    """Download and unpack the tokensave release binary. Returns its path.

    Raises ``RuntimeError`` when no asset exists for this platform, or when
    the download / extraction / verification fails.
    """
    version = version or _pinned_version()
    asset = _detect_asset(version)
    if asset is None:
        raise RuntimeError(
            f"no prebuilt tokensave asset for {platform.system()} {platform.machine()}"
        )
    filename, kind = asset
    url = f"{GITHUB_RELEASE_URL}/{version}/{filename}"

    TOKENSAVE_BIN_DIR.mkdir(parents=True, exist_ok=True)
    bin_name = f"{TOKENSAVE_BIN_NAME}.exe" if kind == "zip" else TOKENSAVE_BIN_NAME
    target_path = TOKENSAVE_BIN_DIR / bin_name

    logger.info("Downloading tokensave %s for %s ...", version, filename)

    try:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")
        with urlopen(url, timeout=60) as response:  # noqa: S310
            data = response.read()
    except Exception as e:
        raise RuntimeError(f"Failed to download tokensave from {url}: {e}") from e

    try:
        if kind == "tar.gz":
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name == TOKENSAVE_BIN_NAME or member.name.endswith(
                        f"/{TOKENSAVE_BIN_NAME}"
                    ):
                        member.name = target_path.name
                        tar.extract(member, TOKENSAVE_BIN_DIR)
                        break
                else:
                    raise RuntimeError("tokensave binary not found in archive")
        else:  # zip
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    if name.endswith(f"{TOKENSAVE_BIN_NAME}.exe") or name.endswith(
                        f"/{TOKENSAVE_BIN_NAME}"
                    ):
                        with zf.open(name) as src, open(target_path, "wb") as dst:
                            dst.write(src.read())
                        break
                else:
                    raise RuntimeError("tokensave binary not found in archive")
    except (tarfile.TarError, zipfile.BadZipFile) as e:
        raise RuntimeError(f"Failed to extract tokensave archive: {e}") from e

    if kind != "zip":
        target_path.chmod(target_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    try:
        result = subprocess.run(
            [str(target_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("Installed tokensave: %s", result.stdout.strip())
        else:
            logger.warning("tokensave installed but version check failed")
    except Exception:
        pass

    return target_path


def ensure_tokensave(version: str | None = None) -> Path | None:
    """Ensure tokensave is available, downloading the release binary if needed.

    Returns the binary path, or ``None`` when the binary is absent and cannot
    be fetched (offline, unsupported platform, or download failure). Callers
    treat ``None`` as "tokensave unavailable → fall back to Serena".
    """
    existing = get_tokensave_path()
    if existing:
        return existing

    if os.environ.get("HEADROOM_BINARIES_OFFLINE"):
        logger.info("tokensave not installed and HEADROOM_BINARIES_OFFLINE set — skipping download")
        return None

    try:
        return download_tokensave(version)
    except RuntimeError as e:
        logger.warning("Could not install tokensave: %s", e)
        return None
