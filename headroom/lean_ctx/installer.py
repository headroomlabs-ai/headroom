"""Download and install lean-ctx binary from GitHub releases.

Supply-chain integrity:
    Headroom downloads and then executes this binary, so every pinned
    release asset is verified against a SHA-256 digest in
    ``LEAN_CTX_ASSET_DIGESTS`` below before the archive is unpacked. A
    mismatch aborts the install rather than running a tampered or
    substituted artifact. When the version is overridden or a cross-target
    asset is requested (``HEADROOM_LEAN_CTX_TARGET`` / ``LEAN_CTX_TARGET``)
    there is no pinned digest, so the download is refused unless the
    operator opts out via ``HEADROOM_LEAN_CTX_ALLOW_UNVERIFIED=1``.
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
import zipfile
from pathlib import Path
from urllib.request import urlopen

from headroom._subprocess import run

from . import LEAN_CTX_BIN_DIR, LEAN_CTX_VERSION

logger = logging.getLogger(__name__)

GITHUB_RELEASE_URL = "https://github.com/yvgude/lean-ctx/releases/download"

#: SHA-256 of each pinned release asset (``LEAN_CTX_VERSION``), keyed by asset
#: filename. The binary is downloaded and executed, so its bytes are verified
#: against this map before extraction. Regenerate when bumping LEAN_CTX_VERSION:
#:   gh api repos/yvgude/lean-ctx/releases/tags/$LEAN_CTX_VERSION \
#:     --jq '.assets[]|"\(.name) \(.digest)"'
LEAN_CTX_ASSET_DIGESTS: dict[str, str] = {
    "lean-ctx-aarch64-apple-darwin.tar.gz": (
        "c4db95966f80ab47aadfca296d0f95937085cf601833f0288eeec8b9f02872cd"
    ),
    "lean-ctx-x86_64-apple-darwin.tar.gz": (
        "9d55d9ed24d3b3726c16eea3cc16255538f286a880531b7fa90e7fb00361e2e2"
    ),
    "lean-ctx-aarch64-unknown-linux-gnu.tar.gz": (
        "72435a42bb33afc3d3cd5a62426955c6488192826a3a84d57e26f587740534d9"
    ),
    "lean-ctx-aarch64-unknown-linux-musl.tar.gz": (
        "be68c45ebb19e30ae6fc4713ec56f148ef2dfa08669b2db4abe57706e625c0e8"
    ),
    "lean-ctx-x86_64-unknown-linux-gnu.tar.gz": (
        "ec405e643a4c4cb3e7fdd2818801f11a6d0209cbcfe0ce085df1d62335a5053b"
    ),
    "lean-ctx-x86_64-unknown-linux-musl.tar.gz": (
        "d2cb70294044a04edc32b7bb9ba2e81f826c042db4840226058d2bd4941e0034"
    ),
    "lean-ctx-x86_64-pc-windows-msvc.zip": (
        "57ff7ff936228828ffc94e0803e1727c5ad03d92791283614406b7e4f66706b0"
    ),
}


def _verify_asset_digest(filename: str, data: bytes, expected: str | None) -> None:
    """Verify downloaded bytes against the pinned SHA-256 digest.

    ``expected`` is the pinned digest for this asset, or ``None`` when no
    digest is pinned (an overridden version or cross-target download). An
    unpinned asset is refused unless ``HEADROOM_LEAN_CTX_ALLOW_UNVERIFIED``
    is set, so unverified code is never executed by default. Raises
    ``RuntimeError`` on a digest mismatch or an unpinned-without-opt-out.
    """
    if expected is None:
        if os.environ.get("HEADROOM_LEAN_CTX_ALLOW_UNVERIFIED"):
            logger.warning(
                "lean-ctx asset %s has no pinned digest; installing unverified "
                "(HEADROOM_LEAN_CTX_ALLOW_UNVERIFIED is set)",
                filename,
            )
            return
        raise RuntimeError(
            f"no pinned SHA-256 digest for lean-ctx asset {filename!r}; refusing to install "
            "unverified. Set HEADROOM_LEAN_CTX_ALLOW_UNVERIFIED=1 to override."
        )
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"lean-ctx asset {filename!r} failed integrity check: "
            f"expected sha256 {expected}, got {actual}"
        )
    logger.debug("Verified lean-ctx asset %s (sha256 %s)", filename, actual)


def _detect_runtime_target_triple() -> str:
    """Detect platform and return the lean-ctx release target triple."""
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        arch = "aarch64" if machine == "arm64" else "x86_64"
        return f"{arch}-apple-darwin"
    if system == "Linux":
        arch = "aarch64" if machine == "aarch64" else "x86_64"
        suffix = "unknown-linux-musl" if _is_musl() else "unknown-linux-gnu"
        return f"{arch}-{suffix}"
    if system == "Windows":
        return "x86_64-pc-windows-msvc"

    raise RuntimeError(f"Unsupported platform: {system} {machine}")


def _is_musl() -> bool:
    try:
        result = run(
            ["ldd", "--version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return "musl" in (result.stdout + result.stderr).lower()
    except Exception:
        return False


def _get_target_triple() -> str:
    """Return the requested lean-ctx target triple, honoring explicit overrides."""
    return _get_explicit_target_triple() or _detect_runtime_target_triple()


def _get_explicit_target_triple() -> str:
    """Return the explicitly requested lean-ctx target triple, if any."""
    return (
        os.environ.get("HEADROOM_LEAN_CTX_TARGET", "").strip()
        or os.environ.get("LEAN_CTX_TARGET", "").strip()
    )


def _binary_name_for_target(target: str) -> str:
    """Return the expected binary name for a target triple."""
    return "lean-ctx.exe" if "windows" in target else "lean-ctx"


def _should_verify_target(target: str) -> bool:
    """Verify runtime-detected targets; explicit overrides may be cross-target."""
    if _get_explicit_target_triple():
        return False
    return target == _detect_runtime_target_triple()


def _get_download_url(version: str) -> tuple[str, str]:
    """Get download URL and extension for this platform."""
    target = _get_target_triple()
    ext = "zip" if "windows" in target else "tar.gz"
    url = f"{GITHUB_RELEASE_URL}/{version}/lean-ctx-{target}.{ext}"
    return url, ext


def _normalize_version(version: str) -> str:
    """Canonicalize a release version to the ``v``-prefixed tag form.

    The pinned digest map is keyed to ``LEAN_CTX_VERSION`` (``vX.Y.Z``).
    Comparing canonical forms stops a bare ``X.Y.Z`` for the current release
    from being treated as a version override and routed through the unverified
    path.
    """
    version = version.strip()
    return version if version.startswith("v") else f"v{version}"


def download_lean_ctx(version: str | None = None) -> Path:
    """Download lean-ctx binary from GitHub releases."""
    version = _normalize_version(version or LEAN_CTX_VERSION)
    target = _get_target_triple()
    url, ext = _get_download_url(version)
    target_path = LEAN_CTX_BIN_DIR / _binary_name_for_target(target)

    LEAN_CTX_BIN_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading lean-ctx %s from %s ...", version, url)

    try:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL scheme in {url}")
        try:
            with urlopen(url, timeout=30) as response:
                data = response.read()
        except Exception as download_err:
            if "CERTIFICATE_VERIFY_FAILED" in str(download_err):
                raise RuntimeError(
                    "TLS verification failed downloading lean-ctx; "
                    "fix the local trust store and retry."
                ) from download_err
            raise
    except Exception as e:
        raise RuntimeError(f"Failed to download lean-ctx from {url}: {e}") from e

    # Verify integrity before unpacking — this binary is later executed.
    filename = f"lean-ctx-{target}.{ext}"
    expected = LEAN_CTX_ASSET_DIGESTS.get(filename) if version == LEAN_CTX_VERSION else None
    _verify_asset_digest(filename, data, expected)

    try:
        if ext == "tar.gz":
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("/lean-ctx") or member.name == "lean-ctx":
                        member.name = target_path.name
                        tar.extract(member, LEAN_CTX_BIN_DIR)
                        break
                else:
                    raise RuntimeError("lean-ctx binary not found in archive")
        elif ext == "zip":
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    if name.endswith("lean-ctx.exe") or name.endswith("/lean-ctx"):
                        with zf.open(name) as src, open(target_path, "wb") as dst:
                            dst.write(src.read())
                        break
                else:
                    raise RuntimeError("lean-ctx binary not found in archive")
    except (tarfile.TarError, zipfile.BadZipFile) as e:
        raise RuntimeError(f"Failed to extract lean-ctx archive: {e}") from e

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
                raise RuntimeError(f"lean-ctx verification failed: {result.stderr}")
            logger.info("lean-ctx installed: %s", result.stdout.strip())
        except FileNotFoundError as e:
            raise RuntimeError("lean-ctx binary not found after extraction") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("lean-ctx verification timed out") from e
    else:
        logger.info(
            "lean-ctx installed for target %s at %s (verification skipped)",
            target,
            target_path,
        )

    return target_path


def ensure_lean_ctx(version: str | None = None) -> Path | None:
    """Ensure lean-ctx is installed — download if needed."""
    from . import get_lean_ctx_path

    existing = get_lean_ctx_path()
    if existing:
        return existing

    try:
        return download_lean_ctx(version)
    except RuntimeError as e:
        logger.warning("Could not install lean-ctx: %s", e)
        return None
