"""Path helpers for persistent deployments."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import click

from headroom import paths as _paths

_PROFILE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_profile_name(profile: str) -> str:
    """Validate and normalize a deployment profile name."""

    if profile in {".", ".."} or not _PROFILE_RE.fullmatch(profile):
        raise click.ClickException(f"Invalid profile name '{profile}'")
    return profile


def deploy_root() -> Path:
    """Return the root directory for deployment state."""

    return _paths.deploy_root()


def profile_root(profile: str) -> Path:
    """Return the directory for a named deployment profile."""

    return deploy_root() / validate_profile_name(profile)


def manifest_path(profile: str) -> Path:
    """Return the manifest path for a named profile."""

    return profile_root(profile) / "manifest.json"


def log_path(profile: str) -> Path:
    """Return the log path used by persistent runner scripts."""

    return profile_root(profile) / "runner.log"


def pid_path(profile: str) -> Path:
    """Return the pid file for the raw runtime process."""

    return profile_root(profile) / "runner.pid"


def unix_run_script_path(profile: str) -> Path:
    """Return the foreground runner shell script path."""

    return profile_root(profile) / "run-headroom.sh"


def unix_ensure_script_path(profile: str) -> Path:
    """Return the watchdog shell script path."""

    return profile_root(profile) / "ensure-headroom.sh"


def windows_run_script_path(profile: str) -> Path:
    """Return the foreground runner PowerShell script path."""

    return profile_root(profile) / "run-headroom.ps1"


def windows_run_cmd_path(profile: str) -> Path:
    """Return the foreground runner CMD shim path."""

    return profile_root(profile) / "run-headroom.cmd"


def windows_ensure_script_path(profile: str) -> Path:
    """Return the watchdog PowerShell script path."""

    return profile_root(profile) / "ensure-headroom.ps1"


def windows_ensure_cmd_path(profile: str) -> Path:
    """Return the watchdog CMD shim path."""

    return profile_root(profile) / "ensure-headroom.cmd"


def unix_user_env_targets() -> list[Path]:
    """Return user shell files that can carry the persistent env block."""

    home = Path.home()
    return [home / ".bashrc", home / ".zshrc", home / ".profile"]


def unix_system_env_targets() -> list[Path]:
    """Return system shell files that can carry the persistent env block."""

    if sys.platform == "darwin":
        return [Path("/etc/profile"), Path("/etc/zprofile"), Path("/etc/bashrc")]
    return [Path("/etc/profile.d/headroom.sh")]


def claude_settings_path() -> Path:
    """Return the Claude user settings path."""

    return Path.home() / ".claude" / "settings.json"


def codex_config_path() -> Path:
    """Return the Codex config path."""

    return Path.home() / ".codex" / "config.toml"


def openclaw_config_path() -> Path:
    """Return the OpenClaw config path."""

    return Path.home() / ".openclaw" / "openclaw.json"


def opencode_config_path() -> Path:
    """Return the OpenCode config path.

    Resolves ``~/.config/opencode/opencode.json`` when ``OPENCODE_CONFIG``
    is unset; otherwise the value of that environment variable. Checks for
    ``opencode.jsonc`` as well.
    """

    env_path = os.environ.get("OPENCODE_CONFIG", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    base_dir = Path.home() / ".config" / "opencode"
    jsonc_path = base_dir / "opencode.jsonc"

    if jsonc_path.exists():
        return jsonc_path

    return base_dir / "opencode.json"


def zcode_config_dir() -> Path:
    """Return the ZCode user configuration directory."""

    return Path.home() / ".zcode"
