"""Download and install codebase-memory-mcp binary from GitHub releases."""

from __future__ import annotations

import io
import logging
import platform
import shutil
import stat
import tarfile
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)

CBM_VERSION = "v0.6.0"
CBM_REPO = "DeusData/codebase-memory-mcp"
CBM_BIN_DIR = Path.home() / ".local" / "bin"
CBM_BIN_NAME = "codebase-memory-mcp"

GITHUB_RELEASE_URL = f"https://github.com/{CBM_REPO}/releases/download"


def _detect_platform() -> str:
    """Detect platform and return the release asset suffix."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        arch = "arm64" if machine == "arm64" else "amd64"
        return f"darwin-{arch}"
    elif system == "linux":
        arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"
        return f"linux-{arch}"
    elif system == "windows":
        return "windows-amd64"

    raise RuntimeError(f"Unsupported platform: {system} {machine}")


def get_cbm_path() -> Path | None:
    """Find codebase-memory-mcp binary, return path or None."""
    # Check PATH first
    found = shutil.which(CBM_BIN_NAME)
    if found:
        return Path(found)

    # Check our install location
    installed = CBM_BIN_DIR / CBM_BIN_NAME
    if installed.exists() and installed.is_file():
        return installed

    return None


def download_cbm(version: str | None = None) -> Path:
    """Download codebase-memory-mcp binary from GitHub releases.

    Returns path to installed binary.
    """
    version = version or CBM_VERSION
    plat = _detect_platform()
    filename = f"codebase-memory-mcp-{plat}.tar.gz"
    url = f"{GITHUB_RELEASE_URL}/{version}/{filename}"

    CBM_BIN_DIR.mkdir(parents=True, exist_ok=True)
    target_path = CBM_BIN_DIR / CBM_BIN_NAME

    logger.info("Downloading codebase-memory-mcp %s for %s ...", version, plat)

    try:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        with urlopen(url, timeout=60) as response:  # noqa: S310
            data = response.read()
    except Exception as e:
        raise RuntimeError(f"Failed to download codebase-memory-mcp from {url}: {e}") from e

    # Extract binary from tar.gz
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(CBM_BIN_NAME) or member.name == CBM_BIN_NAME:
                    member.name = target_path.name
                    tar.extract(member, CBM_BIN_DIR)
                    break
            else:
                raise RuntimeError("codebase-memory-mcp binary not found in archive")
    except tarfile.TarError as e:
        raise RuntimeError(f"Failed to extract archive: {e}") from e

    # Make executable
    target_path.chmod(target_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # Verify
    try:
        from headroom._subprocess import run

        result = run(
            [str(target_path), "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            ver = result.stdout.strip()
            logger.info("Installed: %s", ver)
        else:
            logger.warning("Binary installed but version check failed")
    except Exception:
        pass

    return target_path


def ensure_cbm() -> Path | None:
    """Ensure codebase-memory-mcp is available. Download if needed.

    Returns path to binary, or None if download failed.
    """
    existing = get_cbm_path()
    if existing:
        return existing

    try:
        return download_cbm()
    except RuntimeError as e:
        logger.warning("Failed to install codebase-memory-mcp: %s", e)
        return None
