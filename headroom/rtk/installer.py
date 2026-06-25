"""Download and install rtk binary from GitHub releases."""

from __future__ import annotations

import io
import json
import logging
import os
import platform
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from headroom._subprocess import run

from . import RTK_BIN_DIR, RTK_BIN_PATH, RTK_VERSION

logger = logging.getLogger(__name__)

GITHUB_RELEASE_URL = "https://github.com/rtk-ai/rtk/releases/download"


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


def download_rtk(version: str | None = None) -> Path:
    """Download rtk binary from GitHub releases.

    Args:
        version: Version to download (e.g., "v0.28.2"). Defaults to pinned version.

    Returns:
        Path to the installed binary.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    version = version or RTK_VERSION
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

    try:
        result = run(
            [str(rtk_path), "init", "--global", "--auto-patch"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("rtk hooks registered in Claude Code")
            return True
        else:
            logger.warning("rtk init failed: %s", result.stderr)
            return False
    except Exception as e:
        logger.warning("Failed to register rtk hooks: %s", e)
        return False


def register_codebuddy_hooks(rtk_path: Path | None = None) -> bool:
    """Register rtk hooks in CodeBuddy settings.

    Tries ``rtk init --agent codebuddy --global --auto-patch`` first.
    If the installed rtk does not support ``--agent codebuddy``, falls back
    to manually writing a PreToolUse hook to ``~/.codebuddy/settings.json``
    that calls ``rtk hook claude`` (which works even with older rtk versions).

    Returns True if hooks were registered successfully.
    """
    rtk_path = rtk_path or RTK_BIN_PATH

    # Try the native --agent codebuddy path first
    try:
        result = run(
            [str(rtk_path), "init", "--agent", "codebuddy", "--global", "--auto-patch"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("rtk hooks registered in CodeBuddy (native --agent codebuddy)")
            return True
        logger.debug(
            "rtk init --agent codebuddy failed, falling back to manual: %s", result.stderr.strip()
        )
    except Exception as e:
        logger.debug("rtk init --agent codebuddy error, falling back to manual: %s", e)

    # Fallback: manually write PreToolUse hook using "rtk hook claude"
    return _register_codebuddy_hooks_manual(rtk_path)


def _register_codebuddy_hooks_manual(rtk_path: Path | None = None) -> bool:
    """Manually write rtk hooks into ~/.codebuddy/settings.json.

    Uses ``rtk hook claude`` command which is compatible with older rtk
    versions that don't support ``--agent codebuddy``.
    """
    from headroom.install.paths import codebuddy_settings_path

    rtk_path = rtk_path or RTK_BIN_PATH
    if not rtk_path or not rtk_path.exists():
        return False

    path = codebuddy_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}

    rtk_cmd = str(rtk_path)
    hook_command = f"{rtk_cmd} hook codebuddy"

    hooks = payload.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}

    pre_tool_use = hooks.get("PreToolUse")
    if not isinstance(pre_tool_use, list):
        pre_tool_use = []

    # Check if already registered (either "rtk hook codebuddy" or "rtk hook claude")
    already_registered = any(
        isinstance(entry, dict)
        and isinstance(entry.get("hooks"), list)
        and any(
            isinstance(h, dict)
            and "rtk" in str(h.get("command", ""))
            and "hook" in str(h.get("command", ""))
            and ("claude" in str(h.get("command", "")) or "codebuddy" in str(h.get("command", "")))
            for h in entry["hooks"]
        )
        for entry in pre_tool_use
    )

    if not already_registered:
        pre_tool_use.append(
            {
                "matcher": "Bash",
                "hooks": [
                    {
                        "type": "command",
                        "command": hook_command,
                    }
                ],
            }
        )
        hooks["PreToolUse"] = pre_tool_use
        payload["hooks"] = hooks
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("rtk hooks manually registered in CodeBuddy (rtk hook claude fallback)")
    else:
        logger.debug("rtk hooks already present in CodeBuddy settings")

    return True


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
