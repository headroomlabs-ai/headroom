"""Path helpers for persistent deployments."""

from __future__ import annotations

import logging
import os
import re
import stat
import sys
from pathlib import Path

import click

from headroom import paths as _paths

logger = logging.getLogger(__name__)

_PROFILE_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# A deployment profile directory holds secrets. `headroom install --env
# KEY=VALUE` is the supported way to give a supervised proxy a provider API key
# — supervisors (launchd, systemd, Task Scheduler, cron) start from a bare
# environment, so nothing else reaches the process — and every such value is
# persisted verbatim, both into `manifest.json` and into the generated runner
# scripts that `export` it before the exec. Those files are therefore written
# owner-only rather than at the process umask (which leaves them world-readable
# at the common 022). The supervisor always runs them as the installing user
# (user scope) or as root (system scope), and neither needs the group/other
# bits, so this is not a functional restriction.
#
# SECURITY.md documents these modes; `tests/test_packaging_extras_and_security_docs.py`
# reads the octal values back out of that document and compares them with what
# an install actually writes, so the policy and the code cannot drift apart.
# Named for what they ARE -- a permission mode -- not for what the files they
# protect contain. The previous SECRET_* spelling read as a credential to
# CodeQL's sensitive-data heuristic, so `logger.warning("... 0o%o", mode)`
# tripped py/clear-text-logging-sensitive-data on a diagnostic that logs a
# permission bitmask and no secret at all.
OWNER_ONLY_FILE_MODE = 0o600
OWNER_ONLY_SCRIPT_MODE = 0o700
OWNER_ONLY_DIR_MODE = 0o700


#: Whether this platform actually enforces POSIX permission bits. On Windows
#: access is governed by ACLs and ``chmod`` only toggles the read-only flag, so
#: it SUCCEEDS without establishing the requested mode — which is why the check
#: below reads the mode back rather than trusting the call not to raise.
POSIX_MODES_ENFORCED = os.name == "posix"


def chmod_owner_only(path: Path, mode: int) -> bool:
    """Narrow ``path`` to an owner-only ``mode``. Returns whether that took.

    This used to swallow every ``OSError`` and return nothing, so a caller had
    no way to tell a restricted file from an unrestricted one and continued as
    if the documented mode had been established. That is the wrong default for
    the two places it matters, because both NARROW A PRE-EXISTING INODE rather
    than create one: a profile directory made before these modes existed is
    0755, and a runner script rewritten through ``O_TRUNC`` keeps whatever mode
    it already had. In both cases this chmod is the only thing standing between
    a provider API key and every local user, and a silent failure left the
    caller asserting a guarantee it did not have.

    The result is verified with ``stat`` instead of inferred from ``chmod`` not
    raising, because on Windows it does not raise and does not work either.

    Returns ``True`` only when the mode is now exactly ``mode``. A caller
    holding secret-bearing content must act on ``False``; see
    :func:`headroom.install.supervisors._write_private_text`, which refuses to
    leave the file behind.
    """
    try:
        path.chmod(mode)
        actual = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        logger.warning(
            "Could not restrict %s to mode 0o%o (%s); it may be readable by other local users.",
            path,
            mode,
            exc,
        )
        return False
    if actual != mode:
        # Expected on Windows, where the bits are advisory -- noise there, a
        # real finding on POSIX.
        logger.log(
            logging.WARNING if POSIX_MODES_ENFORCED else logging.DEBUG,
            "%s is mode 0o%o after asking for 0o%o; this platform does not "
            "enforce POSIX permission bits, so the file's protection comes "
            "from its directory and the system ACLs instead.",
            path,
            actual,
            mode,
        )
        return False
    return True


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
