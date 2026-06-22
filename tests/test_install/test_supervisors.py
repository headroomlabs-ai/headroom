from __future__ import annotations

from pathlib import Path

import click
import pytest

from headroom.install.models import DeploymentManifest, SupervisorKind
from headroom.install.supervisors import (
    _command_for_script,
    _linux_service_unit,
    _linux_task_spec,
    _macos_launchd_plist,
    _render_unix_runner,
    _render_windows_runner,
    install_supervisor,
    remove_supervisor,
    render_runner_scripts,
    start_supervisor,
    stop_supervisor,
)


def _manifest(
    *, profile: str = "default", scope: str = "user", supervisor: str = "service"
) -> DeploymentManifest:
    return DeploymentManifest(
        profile=profile,
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind=supervisor,
        scope=scope,
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        service_name=f"headroom-{profile}",
    )


def test_linux_service_unit_uses_user_systemd_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _manifest()

    unit_path, content = _linux_service_unit(manifest, tmp_path / "run-headroom.sh")

    assert unit_path == tmp_path / ".config" / "systemd" / "user" / "headroom-default.service"
    assert "ExecStart=" + str(tmp_path / "run-headroom.sh") in content
    assert "Restart=on-failure" in content


def test_command_for_script_and_unix_runner(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "headroom.install.supervisors.resolve_headroom_command",
        lambda: ["python", "-m", "headroom"],
    )

    assert _command_for_script("install", "agent", "run") == [
        "python",
        "-m",
        "headroom",
        "install",
        "agent",
        "run",
    ]

    record = _render_unix_runner(
        tmp_path / "scripts" / "run-headroom.sh", ["headroom", "run", "--flag"]
    )
    assert record.kind == "script"
    content = Path(record.path).read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env bash")
    assert "exec headroom run --flag" in content


def test_linux_task_spec_for_user_scope_includes_crontab_markers(tmp_path: Path) -> None:
    manifest = _manifest(profile="smoke", supervisor=SupervisorKind.TASK.value)

    cron_path, content = _linux_task_spec(manifest, tmp_path / "ensure-headroom.sh")

    assert cron_path is None
    assert "# >>> headroom smoke >>>" in content
    assert "# <<< headroom smoke <<<" in content
    assert "@reboot" in content
    assert "*/5 * * * *" in content


def test_macos_launchd_plist_switches_between_keepalive_and_interval(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    service_manifest = _manifest(supervisor=SupervisorKind.SERVICE.value)
    service_path, service_content = _macos_launchd_plist(
        service_manifest, tmp_path / "run-headroom.sh"
    )
    assert service_path == tmp_path / "Library" / "LaunchAgents" / "com.headroom.default.plist"
    assert "<key>KeepAlive</key>" in service_content
    assert "<key>StartInterval</key>" not in service_content

    task_manifest = _manifest(profile="tasky", supervisor=SupervisorKind.TASK.value)
    task_path, task_content = _macos_launchd_plist(
        task_manifest, tmp_path / "ensure-headroom.sh", interval=300
    )
    assert task_path == tmp_path / "Library" / "LaunchAgents" / "com.headroom.tasky.plist"
    assert "<key>StartInterval</key>" in task_content
    assert "<integer>300</integer>" in task_content


def test_render_windows_runner_writes_ps1_and_cmd_wrappers(tmp_path: Path) -> None:
    ps1_path = tmp_path / "run-headroom.ps1"
    cmd_path = tmp_path / "run-headroom.cmd"

    records = _render_windows_runner(
        ps1_path,
        cmd_path,
        ["C:\\Program Files\\Python\\python.exe", "headroom", "install", "agent", "run"],
    )

    assert [record.path for record in records] == [str(ps1_path), str(cmd_path)]
    ps1_content = ps1_path.read_text(encoding="utf-8")
    cmd_content = cmd_path.read_text(encoding="utf-8")
    assert '& "C:\\Program Files\\Python\\python.exe" headroom install agent run' in ps1_content
    assert (
        'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-headroom.ps1" %*'
        in cmd_content
    )


def test_render_runner_scripts_writes_unix_scripts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "linux")
    monkeypatch.setattr(
        "headroom.install.supervisors.resolve_headroom_command", lambda: ["headroom"]
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _manifest()

    records = render_runner_scripts(manifest)

    assert {record.path.split("\\")[-1].split("/")[-1] for record in records} == {
        "run-headroom.sh",
        "ensure-headroom.sh",
    }


def test_render_runner_scripts_writes_windows_scripts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "win32")
    monkeypatch.setattr(
        "headroom.install.supervisors.resolve_headroom_command", lambda: ["headroom.exe"]
    )
    monkeypatch.setattr(
        "headroom.install.supervisors.windows_run_script_path",
        lambda profile: tmp_path / "run-headroom.ps1",
    )
    monkeypatch.setattr(
        "headroom.install.supervisors.windows_run_cmd_path",
        lambda profile: tmp_path / "run-headroom.cmd",
    )
    monkeypatch.setattr(
        "headroom.install.supervisors.windows_ensure_script_path",
        lambda profile: tmp_path / "ensure-headroom.ps1",
    )
    monkeypatch.setattr(
        "headroom.install.supervisors.windows_ensure_cmd_path",
        lambda profile: tmp_path / "ensure-headroom.cmd",
    )

    records = render_runner_scripts(_manifest(profile="win"))

    assert [Path(record.path).name for record in records] == [
        "run-headroom.ps1",
        "run-headroom.cmd",
        "ensure-headroom.ps1",
        "ensure-headroom.cmd",
    ]


def test_install_supervisor_none_returns_runner_records(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "linux")
    monkeypatch.setattr(
        "headroom.install.supervisors.resolve_headroom_command", lambda: ["headroom"]
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _manifest(supervisor=SupervisorKind.NONE.value)

    records = install_supervisor(manifest)

    assert len(records) == 2
    assert all(record.kind == "script" for record in records)


def test_start_and_stop_supervisor_use_linux_systemctl(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "linux")
    monkeypatch.setattr(
        "headroom.install.supervisors.subprocess.run",
        lambda command, **kwargs: calls.append(command),
    )
    manifest = _manifest()

    start_supervisor(manifest)
    stop_supervisor(manifest)

    assert calls == [
        ["systemctl", "--user", "restart", "headroom-default"],
        ["systemctl", "--user", "stop", "headroom-default"],
    ]


def test_install_supervisor_linux_service_and_tasks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "linux")
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "linux")
    run_script = tmp_path / "run-headroom.sh"
    ensure_script = tmp_path / "ensure-headroom.sh"
    monkeypatch.setattr(
        "headroom.install.supervisors.render_runner_scripts",
        lambda manifest: [
            type("Record", (), {"kind": "script", "path": run_script.as_posix()})(),
            type("Record", (), {"kind": "script", "path": ensure_script.as_posix()})(),
        ],
    )
    unit_path = tmp_path / "headroom-default.service"
    monkeypatch.setattr(
        "headroom.install.supervisors._linux_service_unit",
        lambda manifest, script: (unit_path, "UNIT"),
    )
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append((command, kwargs))
        return type("Result", (), {"returncode": 0, "stdout": "# old cron\n"})()

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)

    service_records = install_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    assert unit_path.read_text(encoding="utf-8") == "UNIT"
    assert ["systemctl", "--user", "daemon-reload"] in [call[0] for call in calls]
    assert ["systemctl", "--user", "enable", "headroom-default"] in [call[0] for call in calls]
    assert service_records[-1].kind == "service-unit"

    cron_path = tmp_path / "headroom-system"
    monkeypatch.setattr(
        "headroom.install.supervisors._linux_task_spec",
        lambda manifest, script: (cron_path, "@reboot root ensure\n"),
    )
    system_task_records = install_supervisor(
        _manifest(profile="system-task", scope="system", supervisor=SupervisorKind.TASK.value)
    )
    assert cron_path.read_text(encoding="utf-8") == "@reboot root ensure\n"
    assert system_task_records[-1].kind == "cron"

    monkeypatch.setattr(
        "headroom.install.supervisors._linux_task_spec",
        lambda manifest, script: (
            None,
            "# >>> headroom default >>>\n@reboot ensure\n# <<< headroom default <<<\n",
        ),
    )
    user_task_records = install_supervisor(_manifest(supervisor=SupervisorKind.TASK.value))
    assert user_task_records[-1].kind == "crontab"
    assert calls[-1][0] == ["crontab", "-"]
    assert "@reboot ensure" in calls[-1][1]["input"]


def test_install_supervisor_darwin_windows_and_unsupported(monkeypatch, tmp_path: Path) -> None:
    run_script = tmp_path / "run-headroom.sh"
    ensure_script = tmp_path / "ensure-headroom.sh"
    monkeypatch.setattr(
        "headroom.install.supervisors.render_runner_scripts",
        lambda manifest: [
            type("Record", (), {"kind": "script", "path": run_script.as_posix()})(),
            type("Record", (), {"kind": "script", "path": ensure_script.as_posix()})(),
        ],
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "headroom.install.supervisors.subprocess.run",
        lambda command, **kwargs: calls.append(command),
    )
    monkeypatch.setattr("headroom.install.supervisors.os.getuid", lambda: 123, raising=False)

    plist_path = tmp_path / "com.headroom.default.plist"
    monkeypatch.setattr(
        "headroom.install.supervisors._macos_launchd_plist",
        lambda manifest, script, interval=None: (plist_path, f"plist-{interval}"),
    )
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "darwin")
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "darwin")
    service_records = install_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    task_records = install_supervisor(_manifest(supervisor=SupervisorKind.TASK.value))
    assert plist_path.read_text(encoding="utf-8") == "plist-300"
    assert service_records[-1].kind == "plist"
    assert task_records[-1].kind == "plist"
    assert ["launchctl", "bootstrap", "gui/123", str(plist_path)] in calls

    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "win32")
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "win32")
    monkeypatch.setattr(
        "headroom.install.supervisors.windows_run_cmd_path",
        lambda profile: Path(f"C:\\tmp\\{profile}\\run-headroom.cmd"),
    )
    monkeypatch.setattr(
        "headroom.install.supervisors.windows_ensure_cmd_path",
        lambda profile: Path(f"C:\\tmp\\{profile}\\ensure-headroom.cmd"),
    )
    win_service = install_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    win_task = install_supervisor(_manifest(supervisor=SupervisorKind.TASK.value))
    assert win_service[-1].kind == "windows-service"
    assert win_task[-2].path.endswith("-startup")
    assert [
        "sc.exe",
        "create",
        "headroom-default",
        'binPath= cmd.exe /c "C:\\tmp\\default\\run-headroom.cmd"',
        "start= auto",
    ] in calls
    assert [
        "schtasks",
        "/Create",
        "/TN",
        "headroom-default-health",
        "/TR",
        "C:\\tmp\\default\\ensure-headroom.cmd",
        "/SC",
        "MINUTE",
        "/MO",
        "5",
        "/F",
    ] in calls

    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "plan9")
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "plan9")
    with pytest.raises(click.ClickException, match="not supported"):
        install_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))


class _LaunchctlResult:
    def __init__(self, returncode: int = 0, stderr: str = "", stdout: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


def test_start_and_stop_supervisor_darwin_windows_and_none(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "headroom.install.supervisors.subprocess.run",
        lambda command, **kwargs: calls.append(command) or _LaunchctlResult(0),
    )
    monkeypatch.setattr("headroom.install.supervisors.os.getuid", lambda: 77, raising=False)

    start_supervisor(_manifest(supervisor=SupervisorKind.NONE.value))
    stop_supervisor(_manifest(supervisor=SupervisorKind.NONE.value))
    assert calls == []

    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "darwin")
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "darwin")
    # Warm path: kickstart succeeds (job already bootstrapped), so start does
    # not fall through to bootstrap.
    start_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    stop_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    assert calls == [
        ["launchctl", "kickstart", "-k", "gui/77/com.headroom.default"],
        ["launchctl", "bootout", "gui/77/com.headroom.default"],
    ]

    calls.clear()
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "win32")
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "win32")
    start_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    stop_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    assert calls == [
        ["sc.exe", "start", "headroom-default"],
        ["sc.exe", "stop", "headroom-default"],
    ]


def test_macos_start_bootstraps_when_job_not_registered(monkeypatch, tmp_path: Path) -> None:
    # Post-`stop`/`restart` state: the job was booted out, so `kickstart` fails
    # (launchctl 113) and start must bootstrap the plist instead.
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "darwin")
    monkeypatch.setattr("headroom.install.supervisors.os.getuid", lambda: 77, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == "kickstart":
            return _LaunchctlResult(113, stderr="Could not find service")
        return _LaunchctlResult(0)

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)

    start_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))

    plist_path = tmp_path / "Library" / "LaunchAgents" / "com.headroom.default.plist"
    assert calls == [
        ["launchctl", "kickstart", "-k", "gui/77/com.headroom.default"],
        ["launchctl", "bootstrap", "gui/77", str(plist_path)],
    ]


def test_macos_start_retries_bootstrap_until_launchd_settles(monkeypatch, tmp_path: Path) -> None:
    # launchd returns EIO (error 5) from bootstrap for a while after a bootout;
    # start should retry until it succeeds.
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "darwin")
    monkeypatch.setattr("headroom.install.supervisors.os.getuid", lambda: 77, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.supervisors.time.sleep", lambda _s: None)
    bootstrap_attempts = 0

    def fake_run(command, **kwargs):
        nonlocal bootstrap_attempts
        if command[1] == "kickstart":
            return _LaunchctlResult(113)
        bootstrap_attempts += 1
        if bootstrap_attempts < 3:
            return _LaunchctlResult(5, stderr="Bootstrap failed: 5: Input/output error")
        return _LaunchctlResult(0)

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)

    start_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    assert bootstrap_attempts == 3


def test_macos_start_raises_after_bootstrap_keeps_failing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "darwin")
    monkeypatch.setattr("headroom.install.supervisors.os.getuid", lambda: 77, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.supervisors.time.sleep", lambda _s: None)
    monkeypatch.setattr("headroom.install.supervisors._MACOS_BOOTSTRAP_RETRIES", 3)

    def fake_run(command, **kwargs):
        if command[1] == "kickstart":
            return _LaunchctlResult(113)
        return _LaunchctlResult(5, stderr="Bootstrap failed: 5: Input/output error")

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)

    with pytest.raises(click.ClickException, match="could not start"):
        start_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))


def test_macos_stop_tolerates_missing_job(monkeypatch) -> None:
    # `bootout` of an absent job exits with ESRCH (3); stop must not raise so
    # that `restart` can proceed to start again.
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "darwin")
    monkeypatch.setattr("headroom.install.supervisors.os.getuid", lambda: 77, raising=False)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs.get("check") is not True
        return _LaunchctlResult(3, stderr="Boot-out failed: 3: No such process")

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)

    stop_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    assert calls == [["launchctl", "bootout", "gui/77/com.headroom.default"]]


def test_macos_stop_raises_on_non_esrch_failure(monkeypatch) -> None:
    # A non-3 `bootout` failure (e.g. permissions) is a real error and must
    # surface — otherwise `restart` could report success with a stale job still
    # running.
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "darwin")
    monkeypatch.setattr("headroom.install.supervisors.os.getuid", lambda: 77, raising=False)
    monkeypatch.setattr(
        "headroom.install.supervisors.subprocess.run",
        lambda command, **kwargs: _LaunchctlResult(
            9, stderr="Boot-out failed: 9: Operation not permitted"
        ),
    )

    with pytest.raises(click.ClickException, match="bootout failed"):
        stop_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))


def test_remove_supervisor_removes_user_crontab_block(monkeypatch) -> None:
    calls: list[tuple[list[str], str | None]] = []
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "linux")

    class Result:
        def __init__(self, returncode: int = 0, stdout: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(command: list[str], **kwargs):
        calls.append((command, kwargs.get("input")))
        if command == ["crontab", "-l"]:
            return Result(
                stdout="# >>> headroom default >>>\n@reboot /tmp/ensure\n# <<< headroom default <<<\n"
            )
        return Result()

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)
    manifest = _manifest(supervisor=SupervisorKind.TASK.value)

    remove_supervisor(manifest)

    assert calls[0][0] == ["crontab", "-l"]
    assert calls[1][0] == ["crontab", "-"]


def test_remove_supervisor_linux_service_cron_path_and_missing_crontab(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "linux")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append(command)
        return type("Result", (), {"returncode": 1, "stdout": ""})()

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)
    unit_path = tmp_path / "headroom-default.service"
    unit_path.write_text("unit", encoding="utf-8")
    monkeypatch.setattr(
        "headroom.install.supervisors._linux_service_unit",
        lambda manifest, script: (unit_path, "unit"),
    )
    remove_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    assert not unit_path.exists()
    assert ["systemctl", "--user", "disable", "--now", "headroom-default"] in calls
    assert ["systemctl", "--user", "daemon-reload"] in calls

    cron_path = tmp_path / "headroom-task"
    cron_path.write_text("cron", encoding="utf-8")
    monkeypatch.setattr(
        "headroom.install.supervisors._linux_task_spec",
        lambda manifest, script: (cron_path, "cron"),
    )
    remove_supervisor(_manifest(supervisor=SupervisorKind.TASK.value))
    assert not cron_path.exists()

    monkeypatch.setattr(
        "headroom.install.supervisors._linux_task_spec",
        lambda manifest, script: (None, "cron"),
    )
    remove_supervisor(_manifest(supervisor=SupervisorKind.TASK.value))
    assert calls[-1] == ["crontab", "-l"]


def test_remove_supervisor_darwin_and_windows(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "headroom.install.supervisors.subprocess.run",
        lambda command, **kwargs: calls.append(command),
    )
    monkeypatch.setattr("headroom.install.supervisors.os.getuid", lambda: 55, raising=False)

    plist_path = tmp_path / "com.headroom.default.plist"
    plist_path.write_text("plist", encoding="utf-8")
    monkeypatch.setattr(
        "headroom.install.supervisors.unix_run_script_path",
        lambda profile: tmp_path / "run-headroom.sh",
    )
    monkeypatch.setattr(
        "headroom.install.supervisors.unix_ensure_script_path",
        lambda profile: tmp_path / "ensure-headroom.sh",
    )
    monkeypatch.setattr(
        "headroom.install.supervisors._macos_launchd_plist",
        lambda manifest, script, interval=None: (plist_path, "plist"),
    )
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "darwin")
    remove_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    assert not plist_path.exists()
    assert calls[0] == ["launchctl", "bootout", "gui/55/com.headroom.default"]

    calls.clear()
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "win32")
    remove_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    remove_supervisor(_manifest(supervisor=SupervisorKind.TASK.value))
    assert calls == [
        ["sc.exe", "stop", "headroom-default"],
        ["sc.exe", "delete", "headroom-default"],
        ["schtasks", "/Delete", "/TN", "headroom-default-startup", "/F"],
        ["schtasks", "/Delete", "/TN", "headroom-default-health", "/F"],
    ]
