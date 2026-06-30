"""Download and install codebase-memory-mcp binary from GitHub releases.

Supply-chain integrity:
    Headroom downloads and then executes this binary, so every pinned
    release asset is verified against a SHA-256 digest in
    ``CBM_ASSET_DIGESTS`` below before the archive is unpacked. A mismatch
    aborts the install rather than running a tampered or substituted
    artifact. When the version is overridden there is no pinned digest, so
    the download is refused unless the operator opts out via
    ``HEADROOM_CBM_ALLOW_UNVERIFIED=1``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import platform
import shutil
import stat
import tarfile
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)

CBM_VERSION = "v0.8.1"
CBM_REPO = "DeusData/codebase-memory-mcp"
CBM_BIN_DIR = Path.home() / ".local" / "bin"
CBM_BIN_NAME = "codebase-memory-mcp"

GITHUB_RELEASE_URL = f"https://github.com/{CBM_REPO}/releases/download"

#: SHA-256 of each pinned release asset (``CBM_VERSION``), keyed by asset
#: filename. The binary is downloaded and executed, so its bytes are verified
#: against this map before extraction. Regenerate when bumping CBM_VERSION:
#:   gh api repos/DeusData/codebase-memory-mcp/releases/tags/$CBM_VERSION \
#:     --jq '.assets[]|"\(.name) \(.digest)"'
CBM_ASSET_DIGESTS: dict[str, str] = {
    "codebase-memory-mcp-darwin-arm64.tar.gz": (
        "fbd047509852021b5446a11141bcb0a3d1dcaebf6e5112460960f29f052c1c58"
    ),
    "codebase-memory-mcp-darwin-amd64.tar.gz": (
        "fb62da3016ea12b948351208759b5c083fb1446cf6e78d6db8b7cd28fe86fd54"
    ),
    "codebase-memory-mcp-linux-arm64.tar.gz": (
        "d2f842d1365da5c35d9c5796f57a821c9745267350994346735e1e6e04d46091"
    ),
    "codebase-memory-mcp-linux-amd64.tar.gz": (
        "dbd3b92ea870ef240b63059f26bda15015f76ef9978931bebc3a0f9d09470973"
    ),
}


def _verify_asset_digest(filename: str, data: bytes, expected: str | None) -> None:
    """Verify downloaded bytes against the pinned SHA-256 digest.

    ``expected`` is the pinned digest for this asset, or ``None`` when no
    digest is pinned (an overridden version). An unpinned asset is refused
    unless ``HEADROOM_CBM_ALLOW_UNVERIFIED`` is set, so unverified code is
    never executed by default. Raises ``RuntimeError`` on a digest mismatch
    or an unpinned-without-opt-out.
    """
    if expected is None:
        if os.environ.get("HEADROOM_CBM_ALLOW_UNVERIFIED"):
            logger.warning(
                "codebase-memory-mcp asset %s has no pinned digest; installing unverified "
                "(HEADROOM_CBM_ALLOW_UNVERIFIED is set)",
                filename,
            )
            return
        raise RuntimeError(
            f"no pinned SHA-256 digest for codebase-memory-mcp asset {filename!r}; refusing to "
            "install unverified. Set HEADROOM_CBM_ALLOW_UNVERIFIED=1 to override."
        )
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"codebase-memory-mcp asset {filename!r} failed integrity check: "
            f"expected sha256 {expected}, got {actual}"
        )
    logger.debug("Verified codebase-memory-mcp asset %s (sha256 %s)", filename, actual)


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


def _normalize_version(version: str) -> str:
    """Canonicalize a release version to the ``v``-prefixed tag form.

    The pinned digest map is keyed to ``CBM_VERSION`` (``vX.Y.Z``). Comparing
    canonical forms stops a bare ``X.Y.Z`` for the current release from being
    treated as a version override and routed through the unverified path.
    """
    version = version.strip()
    return version if version.startswith("v") else f"v{version}"


def download_cbm(version: str | None = None) -> Path:
    """Download codebase-memory-mcp binary from GitHub releases.

    Returns path to installed binary.
    """
    version = _normalize_version(version or CBM_VERSION)
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

    # Verify integrity before unpacking — this binary is later executed.
    expected = CBM_ASSET_DIGESTS.get(filename) if version == CBM_VERSION else None
    _verify_asset_digest(filename, data, expected)

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
