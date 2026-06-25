"""Download and install rtk binary from GitHub releases.

Supply-chain integrity:
    ``headroom wrap`` auto-downloads and then *executes* this binary, so
    every pinned release asset is verified against a SHA-256 digest in
    ``RTK_ASSET_DIGESTS`` below before the archive is unpacked. A mismatch
    aborts the install rather than running a tampered or substituted
    artifact. When the version is overridden (``download_rtk(version=...)``)
    or a cross-target asset is requested (``HEADROOM_RTK_TARGET``) there is
    no pinned digest, so the download is refused unless the operator opts
    out via ``HEADROOM_RTK_ALLOW_UNVERIFIED=1``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import platform
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from headroom._subprocess import run

from . import RTK_BIN_DIR, RTK_BIN_PATH, RTK_VERSION

logger = logging.getLogger(__name__)

GITHUB_RELEASE_URL = "https://github.com/rtk-ai/rtk/releases/download"

#: SHA-256 of each pinned release asset (``RTK_VERSION``), keyed by asset
#: filename. The binary is downloaded and executed, so its bytes are verified
#: against this map before extraction. Regenerate when bumping RTK_VERSION:
#:   gh api repos/rtk-ai/rtk/releases/tags/$RTK_VERSION --jq '.assets[]|"\(.name) \(.digest)"'
RTK_ASSET_DIGESTS: dict[str, str] = {
    "rtk-aarch64-apple-darwin.tar.gz": (
        "f223ca074a0215af002679bc1d34ca92b93e25b3e8ae16aace6e84c06e586802"
    ),
    "rtk-x86_64-apple-darwin.tar.gz": (
        "84121316867613e61925c209607f033b2113bb0ce312c267a79d3e3e8f221e49"
    ),
    "rtk-aarch64-unknown-linux-gnu.tar.gz": (
        "cc2b91c064eb670c097c184913c8fbcb1a943d53d7fe505375e96ba0c5b6459f"
    ),
    "rtk-x86_64-unknown-linux-musl.tar.gz": (
        "34975116da11e09e502501daf758143e0b22ed3a42a10eb67fb693a6270d9e36"
    ),
    "rtk-x86_64-pc-windows-msvc.zip": (
        "f0ec18963581657173bd6a51f5ba012b093823f844db749fec218581af30a568"
    ),
}


def _verify_asset_digest(filename: str, data: bytes, expected: str | None) -> None:
    """Verify downloaded bytes against the pinned SHA-256 digest.

    ``expected`` is the pinned digest for this asset, or ``None`` when no
    digest is pinned (an overridden version or cross-target download). An
    unpinned asset is refused unless ``HEADROOM_RTK_ALLOW_UNVERIFIED`` is
    set, so unverified code is never executed by default. Raises
    ``RuntimeError`` on a digest mismatch or an unpinned-without-opt-out.
    """
    if expected is None:
        if os.environ.get("HEADROOM_RTK_ALLOW_UNVERIFIED"):
            logger.warning(
                "rtk asset %s has no pinned digest; installing unverified "
                "(HEADROOM_RTK_ALLOW_UNVERIFIED is set)",
                filename,
            )
            return
        raise RuntimeError(
            f"no pinned SHA-256 digest for rtk asset {filename!r}; refusing to install "
            "unverified. Set HEADROOM_RTK_ALLOW_UNVERIFIED=1 to override."
        )
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"rtk asset {filename!r} failed integrity check: "
            f"expected sha256 {expected}, got {actual}"
        )
    logger.debug("Verified rtk asset %s (sha256 %s)", filename, actual)


def _detect_runtime_target_triple() -> str:
    """Detect platform and return the rtk release target triple."""
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        arch = "aarch64" if machine == "arm64" else "x86_64"
        return f"{arch}-apple-darwin"
    elif system == "Linux":
        arch = "aarch64" if machine == "aarch64" else "x86_64"
        suffix = "unknown-linux-gnu" if arch == "aarch64" else "unknown-linux-musl"
        return f"{arch}-{suffix}"
    elif system == "Windows":
        return "x86_64-pc-windows-msvc"

    raise RuntimeError(f"Unsupported platform: {system} {machine}")


def _get_target_triple() -> str:
    """Return the requested rtk target triple, honoring explicit overrides."""
    return os.environ.get("HEADROOM_RTK_TARGET", "").strip() or _detect_runtime_target_triple()


def _binary_name_for_target(target: str) -> str:
    """Return the expected binary name for a target triple."""
    return "rtk.exe" if "windows" in target else "rtk"


def _should_verify_target(target: str) -> bool:
    """Verify only when the requested target matches the current runtime."""
    return target == _detect_runtime_target_triple()


def _get_download_url(version: str) -> tuple[str, str]:
    """Get download URL and extension for this platform.

    Returns (url, extension) where extension is 'tar.gz' or 'zip'.
    """
    target = _get_target_triple()

    if "windows" in target:
        ext = "zip"
    else:
        ext = "tar.gz"

    url = f"{GITHUB_RELEASE_URL}/{version}/rtk-{target}.{ext}"
    return url, ext


def _normalize_version(version: str) -> str:
    """Canonicalize a release version to the ``v``-prefixed tag form.

    The pinned digest map is keyed to ``RTK_VERSION`` (``vX.Y.Z``). Without
    this, a bare ``X.Y.Z`` for the current release would not equal
    ``RTK_VERSION``, get treated as a version override, and route the pinned
    release through the unverified path — letting an operator bypass the digest
    map by accident. Comparing canonical forms closes that gap.
    """
    version = version.strip()
    return version if version.startswith("v") else f"v{version}"


def download_rtk(version: str | None = None) -> Path:
    """Download rtk binary from GitHub releases.

    Args:
        version: Version to download (e.g., "v0.42.4"). Defaults to pinned version.

    Returns:
        Path to the installed binary.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    version = _normalize_version(version or RTK_VERSION)
    target = _get_target_triple()
    url, ext = _get_download_url(version)
    target_path = RTK_BIN_DIR / _binary_name_for_target(target)

    RTK_BIN_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading rtk %s from %s ...", version, url)

    try:
        # Validate URL scheme to prevent B310 warning
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL scheme in {url}")

        # Fail closed on TLS errors rather than executing an unverifiable download.
        try:
            with urlopen(url, timeout=30) as response:
                data = response.read()
        except Exception as download_err:
            if "CERTIFICATE_VERIFY_FAILED" in str(download_err):
                raise RuntimeError(
                    "TLS verification failed downloading rtk; fix the local trust store and retry."
                ) from download_err
            raise
    except Exception as e:
        raise RuntimeError(f"Failed to download rtk from {url}: {e}") from e

    # Verify integrity before unpacking — this binary is later executed.
    filename = f"rtk-{target}.{ext}"
    expected = RTK_ASSET_DIGESTS.get(filename) if version == RTK_VERSION else None
    _verify_asset_digest(filename, data, expected)

    # Extract binary
    try:
        if ext == "tar.gz":
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                # Find the rtk binary inside the archive
                for member in tar.getmembers():
                    if member.name.endswith("/rtk") or member.name == "rtk":
                        member.name = target_path.name  # Flatten path
                        tar.extract(member, RTK_BIN_DIR)
                        break
                else:
                    raise RuntimeError("rtk binary not found in archive")
        elif ext == "zip":
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    if name.endswith("rtk.exe") or name.endswith("/rtk"):
                        with zf.open(name) as src, open(target_path, "wb") as dst:
                            dst.write(src.read())
                        break
                else:
                    raise RuntimeError("rtk binary not found in archive")
    except (tarfile.TarError, zipfile.BadZipFile) as e:
        raise RuntimeError(f"Failed to extract rtk archive: {e}") from e

    # Make executable (skip on Windows — no Unix permissions)
    if "windows" not in target:
        target_path.chmod(target_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    if _should_verify_target(target):
        try:
            result = run(
                [str(target_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError(f"rtk verification failed: {result.stderr}")
            logger.info("rtk installed: %s", result.stdout.strip())
        except FileNotFoundError as e:
            raise RuntimeError("rtk binary not found after extraction") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("rtk verification timed out") from e
    else:
        logger.info("rtk installed for target %s at %s (verification skipped)", target, target_path)

    return target_path


def register_claude_hooks(rtk_path: Path | None = None) -> bool:
    """Register rtk hooks in Claude Code settings.

    Runs `rtk init --global` which adds a PreToolUse hook to
    ~/.claude/settings.json that rewrites Bash commands through rtk.

    Returns True if hooks were registered successfully.
    """
    rtk_path = rtk_path or RTK_BIN_PATH

    # Capture output to a temp file rather than pipes: `rtk init` may fork a
    # background process that inherits our stdout/stderr, and a piped
    # `subprocess.run` drains those pipes until EOF — which never arrives while
    # the daemon holds them open, so it blocks to the timeout even though
    # `rtk init` itself exited and already registered the hooks. A file fd has
    # no such reader, so we wait only on the direct child. stdin is DEVNULL so a
    # stray prompt can never block either.
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as out:
            try:
                result = subprocess.run(
                    [str(rtk_path), "init", "--global", "--auto-patch"],
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=out,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                # Read the temp file while it is still open — the outer handler
                # runs after the `with` closes it, so any captured diagnostics
                # would be gone by then.
                out.seek(0)
                logger.warning("rtk init timed out: %s", out.read().strip())
                return False
            if result.returncode == 0:
                logger.info("rtk hooks registered in Claude Code")
                return True
            out.seek(0)
            logger.warning("rtk init failed: %s", out.read().strip())
            return False
    except Exception as e:
        logger.warning("Failed to register rtk hooks: %s", e)
        return False


def ensure_rtk(version: str | None = None) -> Path | None:
    """Ensure rtk is installed — download if needed.

    Returns path to rtk binary, or None if installation failed.
    """
    from . import get_rtk_path

    existing = get_rtk_path()
    if existing:
        return existing

    try:
        return download_rtk(version)
    except RuntimeError as e:
        logger.warning("Could not install rtk: %s", e)
        return None
