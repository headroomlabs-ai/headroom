"""Download and install lean-ctx binary from GitHub releases."""

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

from headroom._subprocess import run

from . import LEAN_CTX_BIN_DIR, LEAN_CTX_VERSION

logger = logging.getLogger(__name__)

GITHUB_RELEASE_URL = "https://github.com/yvgude/lean-ctx/releases/download"


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


def download_lean_ctx(version: str | None = None) -> Path:
    """Download lean-ctx binary from GitHub releases."""
    version = version or LEAN_CTX_VERSION
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
