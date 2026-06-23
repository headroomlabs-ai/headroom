"""Supervisor installation helpers for persistent deployments."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import click

from .models import ArtifactRecord, DeploymentManifest, SupervisorKind
from .paths import (
    unix_ensure_script_path,
    unix_run_script_path,
    windows_ensure_cmd_path,
    windows_ensure_script_path,
    windows_run_cmd_path,
    windows_run_script_path,
)
from .runtime import resolve_headroom_command

# After `launchctl bootout`, a follow-up `bootstrap` of the same label can
# return EIO (error 5) for several seconds while launchd releases it. Retry the
# bootstrap up to ~15s (30 attempts x 0.5s) to ride out that settle window.
_MACOS_BOOTSTRAP_RETRIES = 30
_MACOS_BOOTSTRAP_RETRY_DELAY = 0.5

# `launchctl bootout` of an already-absent job exits with ESRCH ("No such
# process"). That single code is the only failure we treat as already-stopped.
_LAUNCHCTL_ESRCH = 3


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _command_for_script(*parts: str) -> list[str]:
    return [*resolve_headroom_command(), *parts]


def _render_unix_runner(path: Path, command: list[str]) -> ArtifactRecord:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nexec "
        + " ".join(shlex.quote(x) for x in command)
        + "\n"
    )
    path.chmod(0o755)
    return ArtifactRecord(kind="script", path=str(path))


def _render_windows_runner(
    ps1_path: Path, cmd_path: Path, command: list[str]
) -> list[ArtifactRecord]:
    ps1_path.parent.mkdir(parents=True, exist_ok=True)
    escaped = " ".join(
        [f'"{item}"' if (" " in item or item.endswith(".cmd")) else item for item in command]
    )
    ps1_path.write_text(f"$ErrorActionPreference = 'Stop'\n& {escaped}\nexit $LASTEXITCODE\n")
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
            )
        )
        records.extend(
            _render_windows_runner(
                windows_ensure_script_path(manifest.profile),
                windows_ensure_cmd_path(manifest.profile),
                _command_for_script("install", "agent", "ensure", "--profile", manifest.profile),
            )
        )
        return records

    return [
        _render_unix_runner(
            unix_run_script_path(manifest.profile),
            _command_for_script("install", "agent", "run", "--profile", manifest.profile),
        ),
        _render_unix_runner(
            unix_ensure_script_path(manifest.profile),
            _command_for_script("install", "agent", "ensure", "--profile", manifest.profile),
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
            current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
            existing = current.stdout if current.returncode == 0 else ""
            marker_start = f"# >>> headroom {manifest.profile} >>>"
            marker_end = f"# <<< headroom {manifest.profile} <<<"
            pattern = re.compile(
                re.escape(marker_start) + r".*?" + re.escape(marker_end), re.DOTALL
            )
            merged = pattern.sub("", existing).strip()
            new_content = (merged + "\n\n" + content).strip() + "\n"
            subprocess.run(["crontab", "-"], input=new_content, text=True, check=True)
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
        subprocess.run(["launchctl", "bootout", domain], capture_output=True, text=True)
        bootstrap_domain = (
            "system"
            if manifest.scope == "system"
            and manifest.supervisor_kind == SupervisorKind.SERVICE.value
            else f"gui/{os.getuid()}"
        )
        subprocess.run(["launchctl", "bootstrap", bootstrap_domain, str(plist_path)], check=True)
        records.append(ArtifactRecord(kind="plist", path=str(plist_path)))
        return records

    if _is_windows() and manifest.supervisor_kind == SupervisorKind.SERVICE.value:
        service_bin = f'cmd.exe /c "{windows_run_cmd_path(manifest.profile)}"'
        subprocess.run(
            ["sc.exe", "create", manifest.service_name, f"binPath= {service_bin}", "start= auto"],
            check=True,
        )
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
        user_args = ["/RU", "SYSTEM"] if manifest.scope == "system" else []
        start_schedule = [
            "schtasks",
            "/Create",
            "/TN",
            startup_name,
            "/TR",
            startup_cmd,
            "/SC",
            "ONSTART",
            "/F",
            *user_args,
        ]
        health_schedule = [
            "schtasks",
            "/Create",
            "/TN",
            health_name,
            "/TR",
            startup_cmd,
            "/SC",
            "MINUTE",
            "/MO",
            "5",
            "/F",
            *user_args,
        ]
        subprocess.run(start_schedule, check=True)
        subprocess.run(health_schedule, check=True)
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
        kick = subprocess.run(
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
        # launchd can return EIO (error 5) from bootstrap for several seconds
        # after a bootout while it releases the label, so retry for ~15s.
        plist_dir = (
            Path("/Library/LaunchDaemons")
            if manifest.scope == "system"
            and manifest.supervisor_kind == SupervisorKind.SERVICE.value
            else Path.home() / "Library" / "LaunchAgents"
        )
        plist_path = plist_dir / f"{label}.plist"
        last = kick
        for _ in range(_MACOS_BOOTSTRAP_RETRIES):
            boot = subprocess.run(
                ["launchctl", "bootstrap", domain, str(plist_path)],
                capture_output=True,
                text=True,
            )
            if boot.returncode == 0:
                return
            last = boot
            time.sleep(_MACOS_BOOTSTRAP_RETRY_DELAY)
        detail = (last.stderr or last.stdout or "").strip()
        raise click.ClickException(
            f"launchctl could not start {domain}/{label}: {detail or 'unknown error'}"
        )
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
        result = subprocess.run(
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
            subprocess.run(
                ["systemctl", *flags, "disable", "--now", manifest.service_name],
                capture_output=True,
                text=True,
            )
            unit_path, _ = _linux_service_unit(manifest, unix_run_script_path(manifest.profile))
            if unit_path.exists():
                unit_path.unlink()
            subprocess.run(["systemctl", *flags, "daemon-reload"], capture_output=True, text=True)
            return
        cron_path, _ = _linux_task_spec(manifest, unix_ensure_script_path(manifest.profile))
        if cron_path and cron_path.exists():
            cron_path.unlink()
            return
        current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if current.returncode != 0:
            return
        marker_start = f"# >>> headroom {manifest.profile} >>>"
        marker_end = f"# <<< headroom {manifest.profile} <<<"
        pattern = re.compile(re.escape(marker_start) + r".*?" + re.escape(marker_end), re.DOTALL)
        content = pattern.sub("", current.stdout).strip()
        subprocess.run(
            ["crontab", "-"], input=(content + "\n") if content else "", text=True, check=True
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
        subprocess.run(
            ["launchctl", "bootout", f"{domain}/{label}"], capture_output=True, text=True
        )
        if plist_path.exists():
            plist_path.unlink()
        return

    if _is_windows():
        if manifest.supervisor_kind == SupervisorKind.SERVICE.value:
            subprocess.run(
                ["sc.exe", "stop", manifest.service_name], capture_output=True, text=True
            )
            subprocess.run(
                ["sc.exe", "delete", manifest.service_name], capture_output=True, text=True
            )
            return
        subprocess.run(
            ["schtasks", "/Delete", "/TN", f"{manifest.service_name}-startup", "/F"],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["schtasks", "/Delete", "/TN", f"{manifest.service_name}-health", "/F"],
            capture_output=True,
            text=True,
        )
