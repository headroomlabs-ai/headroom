"""Download and install codebase-memory-mcp binary from GitHub releases."""

from __future__ import annotations

import io
import logging
import platform
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)

CBM_VERSION = "v0.8.1"
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


def _asset_filename(plat: str) -> str:
    """Return the release asset name for ``plat``, with the extension tools.json pins.

    Upstream publishes a .zip for Windows and a .tar.gz for every other platform.
    """
    from headroom.binaries import _tool_entry

    stem = f"codebase-memory-mcp-{plat}"
    for asset in _tool_entry("codebase-memory-mcp").get("assets", {}).values():
        name = str(asset.get("url", "")).rsplit("/", 1)[-1]
        if name in (f"{stem}.tar.gz", f"{stem}.zip"):
            return name
    return f"{stem}.tar.gz"


def _extract_from_zip(data: bytes, target_path: Path) -> None:
    """Write the binary member of a .zip release archive to ``target_path``."""
    wanted = (CBM_BIN_NAME, f"{CBM_BIN_NAME}.exe")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if not info.is_dir() and info.filename.rsplit("/", 1)[-1] in wanted:
                target_path.write_bytes(archive.read(info))
                return
    raise RuntimeError("codebase-memory-mcp binary not found in archive")


def _installed_name(plat: str) -> str:
    """Return the file name to install the binary under for ``plat``.

    On Windows, ``shutil.which`` (and so a PATH lookup) only finds a binary
    whose name ends in a PATHEXT extension such as ``.exe``.
    """
    return f"{CBM_BIN_NAME}.exe" if plat.startswith("windows") else CBM_BIN_NAME


def get_cbm_path() -> Path | None:
    """Find codebase-memory-mcp binary, return path or None."""
    # Check PATH first
    found = shutil.which(CBM_BIN_NAME)
    if found:
        return Path(found)

    # Check our install location
    installed = CBM_BIN_DIR / _installed_name(platform.system().lower())
    if installed.exists() and installed.is_file():
        return installed

    return None


def download_cbm(version: str | None = None) -> Path:
    """Download codebase-memory-mcp binary from GitHub releases.

    Returns path to installed binary.
    """
    version = version or CBM_VERSION
    plat = _detect_platform()
    filename = _asset_filename(plat)
    url = f"{GITHUB_RELEASE_URL}/{version}/{filename}"

    CBM_BIN_DIR.mkdir(parents=True, exist_ok=True)
    target_path = CBM_BIN_DIR / _installed_name(plat)

    logger.info("Downloading codebase-memory-mcp %s for %s ...", version, plat)

    try:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        with urlopen(url, timeout=60) as response:  # noqa: S310
            data = response.read()
    except Exception as e:
        raise RuntimeError(f"Failed to download codebase-memory-mcp from {url}: {e}") from e

    from headroom.binaries import verify_download_bytes

    verify_download_bytes(data, url=url, name="codebase-memory-mcp")

    # Extract binary from the .zip (Windows) or .tar.gz (everything else)
    try:
        if filename.endswith(".zip"):
            _extract_from_zip(data, target_path)
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(CBM_BIN_NAME) or member.name == CBM_BIN_NAME:
                        member.name = target_path.name
                        tar.extract(member, CBM_BIN_DIR, filter="data")
                        break
                else:
                    raise RuntimeError("codebase-memory-mcp binary not found in archive")
    except (tarfile.TarError, zipfile.BadZipFile) as e:
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

    # Local import to match this module's lazy-import style (see download_cbm).
    from headroom.binaries import UnpinnedDownload

    try:
        return download_cbm()
    except RuntimeError as e:
        logger.warning("Failed to install codebase-memory-mcp: %s", e)
        return None
    except UnpinnedDownload as e:
        # Documented contract is "path, or None if the download failed", and a
        # refusal is a failure to install -- the feature is simply unavailable.
        # Deliberately NOT widened to BinaryError: Sha256Mismatch is a tamper
        # signal and must keep propagating rather than becoming a quiet None.
        logger.warning("Refusing to install codebase-memory-mcp: %s", e)
        return None
