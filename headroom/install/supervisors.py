"""Supervisor installation helpers for persistent deployments."""

from __future__ import annotations

import getpass
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree
from xml.sax.saxutils import escape as _xml_escape

import click

from headroom._subprocess import run

from .models import ArtifactRecord, DeploymentManifest, SupervisorKind
from .paths import (
    unix_ensure_script_path,
    unix_run_script_path,
    windows_ensure_cmd_path,
    windows_ensure_script_path,
    windows_run_cmd_path,
    windows_run_script_path,
)
from .providers import _powershell_literal
from .runtime import resolve_headroom_command

# After `launchctl bootout`, a follow-up `bootstrap` of the same label can
# return EIO (error 5) for several seconds while launchd releases it. Retry the
# bootstrap up to ~15s (30 attempts x 0.5s) to ride out that settle window.
_MACOS_BOOTSTRAP_RETRIES = 30
_MACOS_BOOTSTRAP_RETRY_DELAY = 0.5

# `launchctl bootout` of an already-absent job exits with ESRCH ("No such
# process"). That single code is the only failure we treat as already-stopped.
_LAUNCHCTL_ESRCH = 3
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _validated_env_items(env: dict[str, str] | None) -> list[tuple[str, str]]:
    items = list((env or {}).items())
    for name, _value in items:
        if not _ENV_NAME_RE.fullmatch(name):
            raise click.ClickException(
                f"Invalid environment variable name {name!r}; expected [A-Za-z_][A-Za-z0-9_]*."
            )
    return items


def _bootstrap_with_retry(domain: str, plist_path: Path, *, action: str = "bootstrap") -> None:
    """Bootstrap ``plist_path`` into ``domain``, riding out launchd's EIO window.

    After a `launchctl bootout`, a follow-up `bootstrap` of the same label can
    return EIO (error 5) for several seconds while launchd releases it. Retry
    for ~15s before giving up. Shared by `install_supervisor` (bootout+bootstrap
    on every apply) and `start_supervisor` (bootstrap after a failed kickstart)
    so both self-heal instead of requiring the manual bootout+rm+reapply
    recovery previously documented for this race.
    """
    last: subprocess.CompletedProcess[str] | None = None
    for _ in range(_MACOS_BOOTSTRAP_RETRIES):
        boot = run(
            ["launchctl", "bootstrap", domain, str(plist_path)],
            capture_output=True,
            text=True,
        )
        if boot.returncode == 0:
            return
        last = boot
        time.sleep(_MACOS_BOOTSTRAP_RETRY_DELAY)
    detail = (last.stderr or last.stdout or "").strip() if last is not None else ""
    raise click.ClickException(
        f"launchctl could not {action} {domain}/{plist_path.stem}: {detail or 'unknown error'}"
    )


def _command_for_script(*parts: str) -> list[str]:
    return [*resolve_headroom_command(), *parts]


def _render_unix_runner(
    path: Path, command: list[str], env: dict[str, str] | None = None
) -> ArtifactRecord:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Supervisors (launchd, systemd, cron) invoke this script with a bare
    # environment — they do not inherit the interactive shell's exports (e.g.
    # AWS_PROFILE, a custom HEADROOM_WORKSPACE_DIR). Export base_env here, before
    # the exec, so `headroom install agent run` itself (which loads the manifest
    # from HEADROOM_WORKSPACE_DIR) sees the same environment `install apply` was
    # run under, not just the proxy subprocess it spawns.
    export_lines = "".join(
        f"export {name}={shlex.quote(value)}\n" for name, value in _validated_env_items(env)
    )
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + export_lines
        + "exec "
        + " ".join(shlex.quote(x) for x in command)
        + "\n"
    )
    path.chmod(0o755)
    return ArtifactRecord(kind="script", path=str(path))


def _render_windows_runner(
    ps1_path: Path, cmd_path: Path, command: list[str], env: dict[str, str] | None = None
) -> list[ArtifactRecord]:
    ps1_path.parent.mkdir(parents=True, exist_ok=True)
    escaped = " ".join(
        [f'"{item}"' if (" " in item or item.endswith(".cmd")) else item for item in command]
    )
    # See _render_unix_runner: Windows services/tasks also start with a bare
    # environment, so base_env must be set explicitly before invoking headroom.
    env_lines = "".join(
        f"$env:{name} = {_powershell_literal(value)}\n" for name, value in _validated_env_items(env)
    )
    ps1_path.write_text(
        f"$ErrorActionPreference = 'Stop'\n{env_lines}& {escaped}\nexit $LASTEXITCODE\n"
    )
    cmd_path.write_text(
        '@echo off\r\npowershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0'
        + ps1_path.name
        + '" %*\r\n'
    )
    return [
        ArtifactRecord(kind="script", path=str(ps1_path)),
        ArtifactRecord(kind="script", path=str(cmd_path)),
    ]


def render_runner_scripts(manifest: DeploymentManifest) -> list[ArtifactRecord]:
    """Render runner/watchdog scripts for the deployment profile."""

    if _is_windows():
        records = []
        records.extend(
            _render_windows_runner(
                windows_run_script_path(manifest.profile),
                windows_run_cmd_path(manifest.profile),
                _command_for_script("install", "agent", "run", "--profile", manifest.profile),
                manifest.base_env,
            )
        )
        records.extend(
            _render_windows_runner(
                windows_ensure_script_path(manifest.profile),
                windows_ensure_cmd_path(manifest.profile),
                _command_for_script("install", "agent", "ensure", "--profile", manifest.profile),
                manifest.base_env,
            )
        )
        return records

    return [
        _render_unix_runner(
            unix_run_script_path(manifest.profile),
            _command_for_script("install", "agent", "run", "--profile", manifest.profile),
            manifest.base_env,
        ),
        _render_unix_runner(
            unix_ensure_script_path(manifest.profile),
            _command_for_script("install", "agent", "ensure", "--profile", manifest.profile),
            manifest.base_env,
        ),
    ]


def _linux_service_unit(manifest: DeploymentManifest, run_script: Path) -> tuple[Path, str]:
    if manifest.scope == "system":
        unit_path = Path("/etc/systemd/system") / f"{manifest.service_name}.service"
    else:
        unit_path = (
            Path.home() / ".config" / "systemd" / "user" / f"{manifest.service_name}.service"
        )
    content = f"""[Unit]
Description=Headroom ({manifest.profile})
After=network-online.target

[Service]
Type=simple
ExecStart={run_script}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
    return unit_path, content


def _macos_launchd_plist(
    manifest: DeploymentManifest, command_path: Path, *, interval: int | None = None
) -> tuple[Path, str]:
    if manifest.supervisor_kind == SupervisorKind.SERVICE.value:
        base_dir = (
            Path("/Library/LaunchDaemons")
            if manifest.scope == "system"
            else Path.home() / "Library" / "LaunchAgents"
        )
    else:
        base_dir = Path.home() / "Library" / "LaunchAgents"
    plist_path = base_dir / f"com.headroom.{manifest.profile}.plist"
    program = str(command_path)
    keys = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
        '<plist version="1.0">',
        "<dict>",
        "  <key>Label</key>",
        f"  <string>com.headroom.{manifest.profile}</string>",
        "  <key>ProgramArguments</key>",
        "  <array>",
        f"    <string>{program}</string>",
        "  </array>",
        "  <key>RunAtLoad</key>",
        "  <true/>",
    ]
    if interval is not None:
        keys.extend(["  <key>StartInterval</key>", f"  <integer>{interval}</integer>"])
    else:
        keys.extend(["  <key>KeepAlive</key>", "  <true/>"])
    keys.extend(["</dict>", "</plist>"])
    return plist_path, "\n".join(keys) + "\n"


def _linux_task_spec(manifest: DeploymentManifest, ensure_script: Path) -> tuple[Path | None, str]:
    if manifest.scope == "system":
        cron_path = Path("/etc/cron.d") / manifest.service_name
        content = f"@reboot root {ensure_script}\n*/5 * * * * root {ensure_script}\n"
        return cron_path, content

    marker_start = f"# >>> headroom {manifest.profile} >>>"
    marker_end = f"# <<< headroom {manifest.profile} <<<"
    content = (
        f"{marker_start}\n@reboot {ensure_script}\n*/5 * * * * {ensure_script}\n{marker_end}\n"
    )
    return None, content


def _windows_current_user() -> str:
    """Best-effort ``DOMAIN\\USER`` for the S4U task principal."""

    user = os.environ.get("USERNAME") or getpass.getuser()
    domain = os.environ.get("USERDOMAIN")
    return f"{domain}\\{user}" if domain else user


class _WindowsTaskRegistrationError(RuntimeError):
    """Task registration or read-back verification failed."""

    def __init__(self, message: str, *, task_created: bool = False) -> None:
        super().__init__(message)
        self.task_created = task_created


def _windows_task_xml(
    command: str, *, trigger_xml: str, scope: str, logon_type: str | None = None
) -> str:
    """Render Task Scheduler XML for ``command`` and its requested principal.

    User-scope tasks use an S4U principal ("run whether user is logged on or
    not", no stored password) so each run happens in a non-interactive session
    and never draws a console window (issue #2453). Callers can explicitly use
    InteractiveToken as a session-scoped compatibility mode for locked-down
    user accounts.
    System-scope tasks keep the LocalSystem service account, which already has
    no desktop.
    """

    if scope == "system":
        if logon_type not in (None, "ServiceAccount"):
            raise ValueError("System-scope Windows tasks must use ServiceAccount")
        principal = (
            "    <UserId>S-1-5-18</UserId>\n"
            "    <LogonType>ServiceAccount</LogonType>\n"
            "    <RunLevel>HighestAvailable</RunLevel>"
        )
    else:
        logon_type = logon_type or "S4U"
        if logon_type not in ("S4U", "InteractiveToken"):
            raise ValueError(f"Unsupported user-scope Windows task logon type: {logon_type}")
        principal = (
            f"    <UserId>{_xml_escape(_windows_current_user())}</UserId>\n"
            f"    <LogonType>{logon_type}</LogonType>\n"
            "    <RunLevel>LeastPrivilege</RunLevel>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-16"?>\n'
        '<Task version="1.2" '
        'xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n'
        "  <Triggers>\n"
        f"{trigger_xml}\n"
        "  </Triggers>\n"
        '  <Principals>\n    <Principal id="Author">\n'
        f"{principal}\n"
        "    </Principal>\n  </Principals>\n"
        "  <Settings>\n"
        "    <Hidden>true</Hidden>\n"
        "    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n"
        "    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\n"
        "    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n"
        "    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>\n"
        "    <StartWhenAvailable>true</StartWhenAvailable>\n"
        "  </Settings>\n"
        '  <Actions Context="Author">\n'
        f"    <Exec>\n      <Command>{_xml_escape(command)}</Command>\n    </Exec>\n"
        "  </Actions>\n"
        "</Task>\n"
    )


def _windows_boot_trigger() -> str:
    return "    <BootTrigger>\n      <Enabled>true</Enabled>\n    </BootTrigger>"


def _windows_health_trigger() -> str:
    start = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    return (
        "    <TimeTrigger>\n"
        f"      <StartBoundary>{start}</StartBoundary>\n"
        "      <Enabled>true</Enabled>\n"
        "      <Repetition>\n"
        "        <Interval>PT5M</Interval>\n"
        "        <StopAtDurationEnd>false</StopAtDurationEnd>\n"
        "      </Repetition>\n"
        "    </TimeTrigger>"
    )


def _register_windows_task(name: str, xml: str, *, expected_logon_type: str) -> None:
    """Register a task and verify its principal from Task Scheduler's XML."""

    # schtasks reads the XML from a file; UTF-16 matches the declared encoding.
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", encoding="utf-16", delete=False)
    try:
        tmp.write(xml)
        tmp.close()
        task_created = False
        try:
            subprocess.run(
                ["schtasks", "/Create", "/TN", name, "/XML", tmp.name, "/F"],
                check=True,
            )
            task_created = True
            query = subprocess.run(
                ["schtasks", "/Query", "/TN", name, "/XML"],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise _WindowsTaskRegistrationError(
                f"Could not register or verify Windows task {name!r}: {exc}",
                task_created=task_created,
            ) from exc

        try:
            task = ElementTree.fromstring(query.stdout or "")
        except ElementTree.ParseError as exc:
            raise _WindowsTaskRegistrationError(
                f"Could not parse registered Windows task {name!r}", task_created=True
            ) from exc

        logon_element = task.find(".//{*}LogonType")
        actual_logon_type = (
            logon_element.text.strip()
            if logon_element is not None and logon_element.text is not None
            else None
        )
        if actual_logon_type != expected_logon_type:
            raise _WindowsTaskRegistrationError(
                f"Windows task {name!r} requested {expected_logon_type} logon type, "
                f"but Task Scheduler registered {actual_logon_type or 'no logon type'}",
                task_created=True,
            )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _delete_windows_task(name: str) -> subprocess.CompletedProcess[str] | None:
    """Best-effort removal of a scheduled task."""

    try:
        return run(
            ["schtasks", "/Delete", "/TN", name, "/F"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None


def _cleanup_windows_tasks_for_fallback(names: list[str], *, must_delete: set[str]) -> None:
    """Remove preferred tasks and fail if Task Scheduler still reports one."""

    for name in names:
        deletion = _delete_windows_task(name)
        if name in must_delete and (deletion is None or deletion.returncode != 0):
            raise _WindowsTaskRegistrationError(
                f"Could not clean up Windows task {name!r} before compatibility fallback"
            )

        try:
            query = run(
                ["schtasks", "/Query", "/TN", name, "/XML"],
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise _WindowsTaskRegistrationError(
                f"Could not verify cleanup of Windows task {name!r}: {exc}"
            ) from exc

        if query.returncode == 0:
            raise _WindowsTaskRegistrationError(
                f"Could not clean up Windows task {name!r}; it remains registered"
            )


def install_supervisor(manifest: DeploymentManifest) -> list[ArtifactRecord]:
    """Install service/task artifacts for the deployment."""

    records = render_runner_scripts(manifest)
    artifact_paths = {Path(item.path).name: Path(item.path) for item in records}

    if manifest.supervisor_kind == SupervisorKind.NONE.value:
        return records

    if (
        sys.platform.startswith("linux")
        and manifest.supervisor_kind == SupervisorKind.SERVICE.value
    ):
        unit_path, content = _linux_service_unit(manifest, artifact_paths["run-headroom.sh"])
        unit_path.parent.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(content)
        flags = [] if manifest.scope == "system" else ["--user"]
        subprocess.run(["systemctl", *flags, "daemon-reload"], check=True)
        subprocess.run(["systemctl", *flags, "enable", manifest.service_name], check=True)
        records.append(ArtifactRecord(kind="service-unit", path=str(unit_path)))
        return records

    if sys.platform.startswith("linux") and manifest.supervisor_kind == SupervisorKind.TASK.value:
        cron_path, content = _linux_task_spec(manifest, artifact_paths["ensure-headroom.sh"])
        if cron_path is not None:
            cron_path.parent.mkdir(parents=True, exist_ok=True)
            cron_path.write_text(content)
            records.append(ArtifactRecord(kind="cron", path=str(cron_path)))
        else:
            current = run(
                ["crontab", "-l"],
                capture_output=True,
                text=True,
            )
            existing = current.stdout if current.returncode == 0 else ""
            marker_start = f"# >>> headroom {manifest.profile} >>>"
            marker_end = f"# <<< headroom {manifest.profile} <<<"
            pattern = re.compile(
                re.escape(marker_start) + r".*?" + re.escape(marker_end), re.DOTALL
            )
            merged = pattern.sub("", existing).strip()
            new_content = (merged + "\n\n" + content).strip() + "\n"
            run(
                ["crontab", "-"],
                input=new_content,
                text=True,
                check=True,
            )
            records.append(ArtifactRecord(kind="crontab", path=f"user:{manifest.profile}"))
        return records

    if sys.platform == "darwin":
        if manifest.supervisor_kind == SupervisorKind.SERVICE.value:
            plist_path, content = _macos_launchd_plist(manifest, artifact_paths["run-headroom.sh"])
        else:
            plist_path, content = _macos_launchd_plist(
                manifest, artifact_paths["ensure-headroom.sh"], interval=300
            )
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(content)
        domain = (
            f"system/{plist_path.stem}"
            if manifest.scope == "system"
            and manifest.supervisor_kind == SupervisorKind.SERVICE.value
            else f"gui/{os.getuid()}/{plist_path.stem}"
        )
        run(
            ["launchctl", "bootout", domain],
            capture_output=True,
            text=True,
        )
        bootstrap_domain = (
            "system"
            if manifest.scope == "system"
            and manifest.supervisor_kind == SupervisorKind.SERVICE.value
            else f"gui/{os.getuid()}"
        )
        _bootstrap_with_retry(bootstrap_domain, plist_path)
        records.append(ArtifactRecord(kind="plist", path=str(plist_path)))
        return records

    if _is_windows() and manifest.supervisor_kind == SupervisorKind.SERVICE.value:
        # sc.exe's binPath= value embeds its own quotes (cmd.exe /c "<path>").
        # Passing this as an argv list lets subprocess.list2cmdline re-quote the
        # token and sc.exe mis-tokenizes it (issue #1654), so build the exact
        # command line ourselves and hand subprocess a string.
        run_cmd = windows_run_cmd_path(manifest.profile)
        create_cmd = (
            f"sc.exe create {manifest.service_name} "
            f'binPath= "cmd.exe /c \\"{run_cmd}\\"" start= auto'
        )
        subprocess.run(create_cmd, check=True)
        subprocess.run(
            ["sc.exe", "failure", manifest.service_name, "reset= 0", "actions= restart/5000"],
            check=True,
        )
        records.append(ArtifactRecord(kind="windows-service", path=manifest.service_name))
        return records

    if _is_windows() and manifest.supervisor_kind == SupervisorKind.TASK.value:
        startup_name = f"{manifest.service_name}-startup"
        health_name = f"{manifest.service_name}-health"
        startup_cmd = str(windows_ensure_cmd_path(manifest.profile))
        # Register from task XML (not schtasks flags) so the principal is S4U /
        # hidden — flag-created tasks use an interactive token and flash a
        # focus-stealing console on every run (issue #2453).
        preferred_logon_type = "ServiceAccount" if manifest.scope == "system" else "S4U"
        preferred_tasks = [
            (startup_name, _windows_boot_trigger()),
            (health_name, _windows_health_trigger()),
        ]
        registered_preferred_tasks: list[str] = []

        try:
            for name, trigger_xml in preferred_tasks:
                _register_windows_task(
                    name,
                    _windows_task_xml(
                        startup_cmd,
                        trigger_xml=trigger_xml,
                        scope=manifest.scope,
                        logon_type=preferred_logon_type,
                    ),
                    expected_logon_type=preferred_logon_type,
                )
                registered_preferred_tasks.append(name)
        except _WindowsTaskRegistrationError as exc:
            if manifest.scope == "system":
                raise
            must_delete = set(registered_preferred_tasks)
            if exc.task_created:
                must_delete.add(name)
            _cleanup_windows_tasks_for_fallback(
                [name for name, _trigger_xml in preferred_tasks], must_delete=must_delete
            )
            _register_windows_task(
                health_name,
                _windows_task_xml(
                    startup_cmd,
                    trigger_xml=_windows_health_trigger(),
                    scope=manifest.scope,
                    logon_type="InteractiveToken",
                ),
                expected_logon_type="InteractiveToken",
            )
            click.echo(
                "Warning: Windows Task Scheduler rejected logged-off recovery; "
                "using session-scoped health recovery. The health task runs every "
                "five minutes while you are logged on and will not run while you are logged off."
            )
            records.append(ArtifactRecord(kind="windows-task", path=health_name))
            return records

        records.extend(
            [
                ArtifactRecord(kind="windows-task", path=startup_name),
                ArtifactRecord(kind="windows-task", path=health_name),
            ]
        )
        return records

    raise click.ClickException(
        f"Persistent {manifest.supervisor_kind} mode is not supported on this platform."
    )


def start_supervisor(manifest: DeploymentManifest) -> None:
    """Start the installed supervisor or runtime for a deployment."""

    if manifest.supervisor_kind == SupervisorKind.NONE.value:
        return
    if sys.platform.startswith("linux"):
        flags = [] if manifest.scope == "system" else ["--user"]
        subprocess.run(["systemctl", *flags, "restart", manifest.service_name], check=True)
        return
    if sys.platform == "darwin":
        label = f"com.headroom.{manifest.profile}"
        domain = (
            "system"
            if manifest.scope == "system"
            and manifest.supervisor_kind == SupervisorKind.SERVICE.value
            else f"gui/{os.getuid()}"
        )
        # Fast path: when the job is already bootstrapped (e.g. `start` right
        # after `install apply`, or `start` on a running service), `kickstart`
        # restarts it in place.
        kick = run(
            ["launchctl", "kickstart", "-k", f"{domain}/{label}"],
            capture_output=True,
            text=True,
        )
        if kick.returncode == 0:
            return
        # Otherwise the job is not registered in the domain. This is the state
        # `stop`/`restart` leave behind, since they `bootout` the job, and
        # `kickstart` cannot recover it (launchctl error 113). Bootstrap fresh
        # instead — a successful bootstrap also starts the job via RunAtLoad.
        plist_dir = (
            Path("/Library/LaunchDaemons")
            if manifest.scope == "system"
            and manifest.supervisor_kind == SupervisorKind.SERVICE.value
            else Path.home() / "Library" / "LaunchAgents"
        )
        plist_path = plist_dir / f"{label}.plist"
        _bootstrap_with_retry(domain, plist_path, action="start")
        return
    if _is_windows() and manifest.supervisor_kind == SupervisorKind.SERVICE.value:
        subprocess.run(["sc.exe", "start", manifest.service_name], check=True)


def stop_supervisor(manifest: DeploymentManifest) -> None:
    """Stop the installed supervisor for a deployment."""

    if manifest.supervisor_kind == SupervisorKind.NONE.value:
        return
    if sys.platform.startswith("linux"):
        flags = [] if manifest.scope == "system" else ["--user"]
        subprocess.run(["systemctl", *flags, "stop", manifest.service_name], check=True)
        return
    if sys.platform == "darwin":
        label = f"com.headroom.{manifest.profile}"
        domain = (
            "system"
            if manifest.scope == "system"
            and manifest.supervisor_kind == SupervisorKind.SERVICE.value
            else f"gui/{os.getuid()}"
        )
        # `bootout` exits with ESRCH ("No such process") when the job is already
        # absent — tolerate only that, so `restart` can proceed to start again.
        # Any other non-zero result is a real failure (permissions, malformed
        # domain, launchd error) and must surface; otherwise `restart` could
        # report success while a stale job is still running.
        result = run(
            ["launchctl", "bootout", f"{domain}/{label}"],
            capture_output=True,
            text=True,
        )
        if result.returncode not in (0, _LAUNCHCTL_ESRCH):
            detail = (result.stderr or result.stdout or "").strip()
            raise click.ClickException(
                f"launchctl bootout failed for {domain}/{label}: {detail or 'unknown error'}"
            )
        return
    if _is_windows() and manifest.supervisor_kind == SupervisorKind.SERVICE.value:
        subprocess.run(["sc.exe", "stop", manifest.service_name], check=True)


def remove_supervisor(manifest: DeploymentManifest) -> None:
    """Remove installed service/task artifacts."""

    if manifest.supervisor_kind == SupervisorKind.NONE.value:
        return

    if sys.platform.startswith("linux"):
        if manifest.supervisor_kind == SupervisorKind.SERVICE.value:
            flags = [] if manifest.scope == "system" else ["--user"]
            run(
                ["systemctl", *flags, "disable", "--now", manifest.service_name],
                capture_output=True,
                text=True,
            )
            unit_path, _ = _linux_service_unit(manifest, unix_run_script_path(manifest.profile))
            if unit_path.exists():
                unit_path.unlink()
            run(
                ["systemctl", *flags, "daemon-reload"],
                capture_output=True,
                text=True,
            )
            return
        cron_path, _ = _linux_task_spec(manifest, unix_ensure_script_path(manifest.profile))
        if cron_path and cron_path.exists():
            cron_path.unlink()
            return
        current = run(
            ["crontab", "-l"],
            capture_output=True,
            text=True,
        )
        if current.returncode != 0:
            return
        marker_start = f"# >>> headroom {manifest.profile} >>>"
        marker_end = f"# <<< headroom {manifest.profile} <<<"
        pattern = re.compile(re.escape(marker_start) + r".*?" + re.escape(marker_end), re.DOTALL)
        content = pattern.sub("", current.stdout).strip()
        run(
            ["crontab", "-"],
            input=(content + "\n") if content else "",
            text=True,
            check=True,
        )
        return

    if sys.platform == "darwin":
        plist_path, _ = _macos_launchd_plist(
            manifest,
            unix_run_script_path(manifest.profile)
            if manifest.supervisor_kind == SupervisorKind.SERVICE.value
            else unix_ensure_script_path(manifest.profile),
            interval=300 if manifest.supervisor_kind == SupervisorKind.TASK.value else None,
        )
        label = f"com.headroom.{manifest.profile}"
        domain = (
            "system"
            if manifest.scope == "system"
            and manifest.supervisor_kind == SupervisorKind.SERVICE.value
            else f"gui/{os.getuid()}"
        )
        run(
            ["launchctl", "bootout", f"{domain}/{label}"],
            capture_output=True,
            text=True,
        )
        if plist_path.exists():
            plist_path.unlink()
        return

    if _is_windows():
        if manifest.supervisor_kind == SupervisorKind.SERVICE.value:
            run(
                ["sc.exe", "stop", manifest.service_name],
                capture_output=True,
                text=True,
            )
            run(
                ["sc.exe", "delete", manifest.service_name],
                capture_output=True,
                text=True,
            )
            return
        _delete_windows_task(f"{manifest.service_name}-startup")
        _delete_windows_task(f"{manifest.service_name}-health")
