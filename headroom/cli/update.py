"""`headroom update` — self-update across the supported install methods.

Detects how Headroom was installed and runs the matching upgrade command, or
refuses with clear guidance when in-tool self-update isn't appropriate (git
checkout, editable install, Docker image, system package manager).

The upgrade always runs through ``sys.executable -m pip`` for the pip path so
it can never touch a different interpreter than the one actually running
Headroom.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click

from headroom._subprocess import run as _subprocess_run
from headroom.update_check import (
    PACKAGE_NAME,
    fetch_latest_version,
    installed_version,
    write_cache,
)

from .main import main


@dataclass(frozen=True)
class InstallMethod:
    """How Headroom is installed and how (or whether) to upgrade it."""

    kind: str
    can_self_update: bool
    argv: list[str] | None = None
    guidance: str | None = None


def _is_source_checkout() -> bool:
    try:
        from headroom._version import _source_root

        return _source_root() is not None
    except Exception:
        return False


def _is_editable_install() -> bool:
    """Detect a PEP 660 editable install via its ``direct_url.json``."""
    try:
        from importlib.metadata import distribution

        raw = distribution(PACKAGE_NAME).read_text("direct_url.json")
        if not raw:
            return False
        data = json.loads(raw)
        dir_info = data.get("dir_info")
        return bool(isinstance(dir_info, dict) and dir_info.get("editable"))
    except Exception:
        return False


def _in_docker() -> bool:
    try:
        from pathlib import Path

        return Path("/.dockerenv").exists() or bool(
            os.environ.get("HEADROOM_IN_DOCKER", "").strip()
        )
    except Exception:
        return False


def _in_virtualenv() -> bool:
    """True inside a venv/virtualenv or a conda environment."""
    if getattr(sys, "prefix", "") != getattr(sys, "base_prefix", ""):
        return True
    # Conda envs often share prefix==base_prefix for `base`; pip -U is still safe
    # inside any conda env, so treat an active CONDA_PREFIX as an environment.
    return bool(os.environ.get("CONDA_PREFIX", "").strip())


def _norm(path: str | os.PathLike[str] | None) -> str:
    """Resolve + lowercase a path for cross-platform substring matching."""
    if not path:
        return ""
    try:
        from pathlib import Path

        return str(Path(path).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(path).replace("\\", "/").lower()


def _package_location() -> str | None:
    """Normalized base directory the headroom-ai distribution is installed in."""
    try:
        from importlib.metadata import distribution

        return _norm(str(distribution(PACKAGE_NAME).locate_file("")))
    except Exception:
        return None


def _user_site() -> str | None:
    try:
        import site

        return _norm(site.getusersitepackages())
    except Exception:
        return None


def _is_user_site_install(location: str | None) -> bool:
    user = _user_site()
    if not user or not location:
        return False
    # Path-segment containment so "/.../site" never matches "/.../site-packages".
    return location == user or location.startswith(user.rstrip("/") + "/")


def _format_cmd(argv: list[str]) -> str:
    """Render argv as a copy-pasteable shell string (handles spaces in paths)."""
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _windows_pip_update_needs_handoff(method: InstallMethod) -> bool:
    return sys.platform.startswith("win") and method.kind in {"pip", "pip-user"}


def _build_windows_handoff_argv(argv: list[str]) -> list[str]:
    helper = (
        "import subprocess,sys,time; time.sleep(1); "
        "raise SystemExit(subprocess.run([sys.executable,*sys.argv[1:]], check=False).returncode)"
    )
    return [sys.executable, "-c", helper, *argv[1:]]


def _is_externally_managed() -> bool:
    """Detect a PEP 668 ``EXTERNALLY-MANAGED`` marker (Homebrew, Debian, etc.)."""
    try:
        import sysconfig
        from pathlib import Path

        for key in ("stdlib", "platstdlib", "purelib"):
            base = sysconfig.get_path(key)
            if base and Path(base, "EXTERNALLY-MANAGED").exists():
                return True
    except Exception:
        return False
    return False


def _spec(extras: str | None) -> str:
    extras = (extras or "").strip().strip("[]")
    return f"{PACKAGE_NAME}[{extras}]" if extras else PACKAGE_NAME


def _managed_env_guidance() -> str:
    if sys.platform == "darwin":
        hint = "`brew upgrade headroom-ai` (if installed via Homebrew), or reinstall with pipx"
    elif sys.platform.startswith("win"):
        hint = "reinstall with pipx (`pipx install headroom-ai`) or use a virtualenv"
    else:
        hint = "use your distro package manager, or reinstall with pipx / a virtualenv"
    return (
        "Headroom is installed in an externally-managed system Python (PEP 668). "
        f"Don't pip into it — {hint}."
    )


def _find_core_pyd() -> Path | None:
    """Locate the _core.pyd file inside the headroom site-packages directory."""
    try:
        import headroom

        headroom_path = Path(headroom.__file__).parent
        core_pyd = headroom_path / "_core.pyd"
        return core_pyd if core_pyd.exists() else None
    except Exception:
        return None


def _is_pyd_locked(pyd_path: Path) -> bool:
    """Verify whether the .pyd file is locked on Windows by attempting to open it in r+b mode."""
    try:
        with open(pyd_path, "r+b") as f:
            f.close()
        return False
    except (PermissionError, OSError):
        return True


def _make_backup(pyd_path: Path) -> Path:
    """Create a backup of the .pyd file as .pyd.bak."""
    backup_path = pyd_path.with_suffix(pyd_path.suffix + ".bak")
    shutil.copy2(pyd_path, backup_path)
    return backup_path


def _restore_backup(pyd_path: Path, backup_path: Path) -> None:
    """Restore the .pyd file atomically by replacing the original with the backup."""
    os.replace(backup_path, pyd_path)


def _test_core_integrity() -> bool:
    """Test if headroom._core can be imported successfully."""
    try:
        result = _subprocess_run(
            [sys.executable, "-c", "from headroom._core import hello"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def safe_update(argv: list[str]) -> int:
    """Execute update with protection against .pyd corruption on Windows.

    Detects if _core.pyd is locked (indicating that headroom proxy is running),
    makes a backup, executes the update, tests integrity, and restores the
    backup in case of failure.

    Returns 0 if successful, error code otherwise.
    """
    core_pyd = _find_core_pyd()
    backup_path = None

    try:
        # Windows: proactive backup when the .pyd is accessible (not locked).
        # If locked, pip will fail before touching the file — just warn.
        # If not locked, pip will attempt to replace it — back it up as a safety net.
        if sys.platform.startswith("win") and core_pyd:
            if _is_pyd_locked(core_pyd):
                click.secho(
                    "Warning: headroom._core is locked (headroom proxy is running).",
                    fg="yellow",
                )
                click.echo("If the upgrade fails, stop the proxy and try again.")
            else:
                click.echo("Creating a backup of headroom._core before upgrading...")
                try:
                    backup_path = _make_backup(core_pyd)
                    click.echo(f"Backup created: {backup_path}")
                except Exception as e:
                    raise click.ClickException(f"Could not back up {core_pyd}: {e}") from None

        # Executing the update
        result = subprocess.run(argv)  # noqa: S603

        if result.returncode != 0:
            # Upgrade failed: restore backup if available
            if core_pyd and backup_path and backup_path.exists():
                click.secho("Restoring backup after upgrade failure...", fg="yellow")
                try:
                    _restore_backup(core_pyd, backup_path)
                    click.echo("Backup restored successfully.")
                except Exception as e:
                    click.secho(
                        f"ERROR: Could not restore backup: {e}",
                        fg="red",
                    )
            return result.returncode

        # Upgrade successful: test integrity
        if core_pyd:
            click.echo("Testing integrity of headroom._core module...")
            if not _test_core_integrity():
                click.secho(
                    "ERROR: headroom._core could not be imported after upgrade.",
                    fg="red",
                )
                if backup_path and backup_path.exists():
                    click.secho("Restoring backup...", fg="yellow")
                    try:
                        _restore_backup(core_pyd, backup_path)
                        click.echo("Backup restored successfully.")
                    except Exception as e:
                        click.secho(
                            f"CRITICAL ERROR: Could not restore: {e}",
                            fg="red",
                        )
                return 1
            click.echo("✓ Module tested successfully.")

        # Cleanup: remove backup if everything went well
        if backup_path and backup_path.exists():
            try:
                backup_path.unlink()
            except Exception:
                pass  # Failed to delete backup is not critical

        return 0

    except FileNotFoundError:
        if core_pyd and backup_path and backup_path.exists():
            try:
                _restore_backup(core_pyd, backup_path)
            except Exception:
                pass
        raise
    except click.ClickException:
        if core_pyd and backup_path and backup_path.exists():
            try:
                _restore_backup(core_pyd, backup_path)
            except Exception:
                pass
        raise


def detect_install_method(extras: str | None = None) -> InstallMethod:
    """Classify the install and build the appropriate upgrade plan.

    Resolution order (first match wins), covering every supported install path
    on macOS / Linux / Windows:

      1. git checkout            → refuse (`git pull`)
      2. editable install        → refuse (reinstall from source)
      3. Docker                  → refuse (pull a new image)
      4. pipx                    → `pipx upgrade`
      5. uv tool                 → `uv tool upgrade`
      6. venv / virtualenv / conda → `sys.executable -m pip install -U`
      7. user-site (`pip --user`)  → `sys.executable -m pip install -U --user`
      8. externally-managed system Python (PEP 668) → refuse with guidance
      9. writable global Python  → `sys.executable -m pip install -U` (last resort)
    """
    if _is_source_checkout():
        return InstallMethod(
            kind="checkout",
            can_self_update=False,
            guidance="Running from a source checkout — update with `git pull`.",
        )
    if _is_editable_install():
        return InstallMethod(
            kind="editable",
            can_self_update=False,
            guidance=(
                "Editable install detected — update your source tree, or "
                "reinstall with `pip install -U --force-reinstall .`."
            ),
        )
    if _in_docker():
        return InstallMethod(
            kind="docker",
            can_self_update=False,
            guidance=(
                "Running inside a container — pull a newer Headroom image instead of self-updating."
            ),
        )

    # pipx / uv tool own their venvs; match against executable, prefix, and the
    # distribution location so detection works regardless of platform layout.
    haystack = "::".join(
        filter(
            None,
            (
                _norm(getattr(sys, "executable", "")),
                _norm(getattr(sys, "prefix", "")),
                _package_location(),
            ),
        )
    )

    pipx_home = _norm(os.environ.get("PIPX_HOME"))
    if "/pipx/venvs/" in haystack or "/pipx/" in haystack or (pipx_home and pipx_home in haystack):
        return InstallMethod(
            kind="pipx",
            can_self_update=True,
            argv=["pipx", "upgrade", PACKAGE_NAME],
        )

    uv_tool_dir = _norm(os.environ.get("UV_TOOL_DIR"))
    if "/uv/tools/" in haystack or (uv_tool_dir and uv_tool_dir in haystack):
        return InstallMethod(
            kind="uv-tool",
            can_self_update=True,
            argv=["uv", "tool", "upgrade", PACKAGE_NAME],
        )

    if _in_virtualenv():
        return InstallMethod(
            kind="pip",
            can_self_update=True,
            argv=[sys.executable, "-m", "pip", "install", "-U", _spec(extras)],
        )

    location = _package_location()
    if _is_user_site_install(location):
        return InstallMethod(
            kind="pip-user",
            can_self_update=True,
            argv=[sys.executable, "-m", "pip", "install", "-U", "--user", _spec(extras)],
        )

    if _is_externally_managed():
        return InstallMethod(
            kind="system",
            can_self_update=False,
            guidance=_managed_env_guidance(),
        )

    # Writable global interpreter (e.g. Windows python.org, some Linux setups):
    # pip -U works without admin in the common case; if not, the manual command
    # is surfaced on failure.
    return InstallMethod(
        kind="pip",
        can_self_update=True,
        argv=[sys.executable, "-m", "pip", "install", "-U", _spec(extras)],
    )


@main.command("update")
@click.option("--check", "check_only", is_flag=True, help="Report only; do not upgrade.")
@click.option("-y", "--yes", "assume_yes", is_flag=True, help="Skip the confirmation prompt.")
@click.option("--pre", "allow_pre", is_flag=True, help="Include pre-releases.")
@click.option(
    "--extras",
    default=None,
    help="Re-request extras for the pip path, e.g. 'all' or 'proxy'.",
)
def update(check_only: bool, assume_yes: bool, allow_pre: bool, extras: str | None) -> None:
    """Update Headroom to the latest release.

    Detects pipx / uv tool / pip installs and runs the right upgrade. Refuses
    (with guidance) for git checkouts, editable installs, Docker, and system
    Python.
    """
    current = installed_version()

    click.echo("Checking PyPI for the latest Headroom release...")
    latest = fetch_latest_version(allow_pre=allow_pre)
    if latest is None:
        raise click.ClickException("Could not reach PyPI to check for updates. Try again later.")

    # Refresh the banner cache as a side effect of an explicit check.
    write_cache(latest)

    if current:
        from packaging.version import InvalidVersion, Version

        try:
            if Version(latest) <= Version(current):
                click.echo(f"Headroom is up to date ({current}).")
                return
        except InvalidVersion:
            pass
        click.echo(f"Update available: {current} → {latest}")
    else:
        click.echo(f"Latest Headroom release: {latest}")

    method = detect_install_method(extras)

    if not method.can_self_update:
        click.echo(method.guidance or "Automatic update is not available for this install.")
        return

    assert method.argv is not None
    cmd_str = _format_cmd(method.argv)
    click.echo(f"Upgrade command: {cmd_str}")

    if check_only:
        return

    if not assume_yes and not click.confirm("Proceed with the upgrade?", default=True):
        click.echo("Aborted.")
        return

    click.echo(f"Running: {cmd_str}")
    if _windows_pip_update_needs_handoff(method):
        handoff_argv = _build_windows_handoff_argv(method.argv)
        try:
            subprocess.Popen(handoff_argv)  # noqa: S603 - fixed allowlist plus helper argv
        except FileNotFoundError:
            raise click.ClickException(
                f"`{handoff_argv[0]}` was not found on PATH. Upgrade manually: {cmd_str}"
            ) from None
        click.echo("Upgrade started in a child process so pip can replace headroom.exe.")
        click.echo("Wait for the pip output to finish, then rerun `headroom --version`.")
        return

    try:
        returncode = safe_update(method.argv)
    except FileNotFoundError:
        raise click.ClickException(
            f"`{method.argv[0]}` was not found on PATH. Install it or upgrade manually: {cmd_str}"
        ) from None

    if returncode != 0:
        raise click.ClickException(f"Upgrade failed (exit {returncode}). Run manually: {cmd_str}")

    # ASCII-only output — emoji can raise UnicodeEncodeError on some Windows consoles.
    click.echo(f"Headroom upgraded to {latest}.")
    click.echo("Restart any running `headroom proxy` to pick up the new version.")


__all__ = [
    "InstallMethod",
    "_build_windows_handoff_argv",
    "_windows_pip_update_needs_handoff",
    "detect_install_method",
    "update",
]
