from __future__ import annotations

import subprocess
from pathlib import Path

import click
import pytest

from headroom.install.models import DeploymentManifest, SupervisorKind
from headroom.install.supervisors import (
    _command_for_script,
    _linux_service_unit,
    _linux_task_spec,
    _macos_launchd_plist,
    _register_windows_task,
    _render_unix_runner,
    _render_windows_runner,
    _windows_boot_trigger,
    _windows_health_trigger,
    _windows_task_xml,
    _WindowsTaskRegistrationError,
    install_supervisor,
    remove_supervisor,
    render_runner_scripts,
    start_supervisor,
    stop_supervisor,
)


def test_windows_task_xml_user_scope_is_hidden_s4u() -> None:
    # #2453: user-scope tasks must run S4U (non-interactive, no window) and
    # hidden so the 5-minute health run never steals keyboard focus.
    xml = _windows_task_xml(
        "C:\\tmp\\default\\ensure-headroom.cmd",
        trigger_xml=_windows_health_trigger(),
        scope="user",
    )
    assert "<LogonType>S4U</LogonType>" in xml
    assert "<Hidden>true</Hidden>" in xml
    assert "<Interval>PT5M</Interval>" in xml
    assert "<Command>C:\\tmp\\default\\ensure-headroom.cmd</Command>" in xml


def test_windows_task_xml_user_scope_supports_interactive_token() -> None:
    xml = _windows_task_xml(
        "C:\\tmp\\default\\ensure-headroom.cmd",
        trigger_xml=_windows_health_trigger(),
        scope="user",
        logon_type="InteractiveToken",
    )

    assert "<TimeTrigger>" in xml
    assert "<LogonType>InteractiveToken</LogonType>" in xml
    assert "<RunLevel>LeastPrivilege</RunLevel>" in xml
    assert "<Interval>PT5M</Interval>" in xml


def test_windows_task_xml_system_scope_uses_localsystem() -> None:
    xml = _windows_task_xml(
        "C:\\tmp\\default\\ensure-headroom.cmd",
        trigger_xml=_windows_boot_trigger(),
        scope="system",
    )
    assert "<UserId>S-1-5-18</UserId>" in xml
    assert "<LogonType>ServiceAccount</LogonType>" in xml
    assert "<BootTrigger>" in xml


def test_register_windows_task_verifies_queried_logon_type(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append(command)
        if command[1] == "/Query":
            return _LaunchctlResult(
                stdout=(
                    '<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">'
                    "<Principals><Principal><LogonType>S4U</LogonType></Principal></Principals>"
                    "</Task>"
                )
            )
        return _LaunchctlResult()

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)

    _register_windows_task("headroom-default-health", "<Task />", expected_logon_type="S4U")

    assert calls[0][:4] == [
        "schtasks",
        "/Create",
        "/TN",
        "headroom-default-health",
    ]
    assert calls[1] == ["schtasks", "/Query", "/TN", "headroom-default-health", "/XML"]


def test_register_windows_task_rejects_logon_type_downgrade(monkeypatch) -> None:
    def fake_run(command: list[str], **kwargs):
        if command[1] == "/Query":
            return _LaunchctlResult(
                stdout=(
                    '<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">'
                    "<Principals><Principal>"
                    "<LogonType>InteractiveToken</LogonType>"
                    "</Principal></Principals></Task>"
                )
            )
        return _LaunchctlResult()

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)

    with pytest.raises(
        _WindowsTaskRegistrationError, match="requested S4U.*InteractiveToken"
    ) as exc_info:
        _register_windows_task("headroom-default-health", "<Task />", expected_logon_type="S4U")

    assert exc_info.value.task_created is True


@pytest.mark.parametrize("failed_action", ["/Create", "/Query"])
def test_register_windows_task_wraps_schtasks_failures(monkeypatch, failed_action: str) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append(command)
        if command[1] == failed_action:
            raise subprocess.CalledProcessError(1, command)
        return _LaunchctlResult()

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)

    with pytest.raises(
        _WindowsTaskRegistrationError, match="Could not register or verify"
    ) as exc_info:
        _register_windows_task("headroom-default-health", "<Task />", expected_logon_type="S4U")

    assert calls[-1][1] == failed_action
    assert exc_info.value.task_created is (failed_action == "/Query")


def test_register_windows_task_rejects_malformed_queried_xml(monkeypatch) -> None:
    monkeypatch.setattr(
        "headroom.install.supervisors.subprocess.run",
        lambda command, **kwargs: _LaunchctlResult(stdout="<Task>"),
    )

    with pytest.raises(_WindowsTaskRegistrationError, match="Could not parse registered"):
        _register_windows_task("headroom-default-health", "<Task />", expected_logon_type="S4U")


def _manifest(
    *,
    profile: str = "default",
    scope: str = "user",
    supervisor: str = "service",
    base_env: dict[str, str] | None = None,
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
        base_env=base_env or {},
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


def test_render_unix_runner_exports_env_before_exec(tmp_path: Path) -> None:
    record = _render_unix_runner(
        tmp_path / "run-headroom.sh",
        ["headroom", "run"],
        {"HEADROOM_WORKSPACE_DIR": "/Users/x/.headroom-workspace", "AWS_PROFILE": "sso-bedrock"},
    )

    content = Path(record.path).read_text(encoding="utf-8")
    export_index = content.index("export HEADROOM_WORKSPACE_DIR=")
    exec_index = content.index("exec headroom run")

    assert "export HEADROOM_WORKSPACE_DIR=/Users/x/.headroom-workspace" in content
    assert "export AWS_PROFILE=sso-bedrock" in content
    assert export_index < exec_index


def test_render_unix_runner_rejects_invalid_env_name(tmp_path: Path) -> None:
    with pytest.raises(click.ClickException, match="Invalid environment variable name"):
        _render_unix_runner(tmp_path / "run-headroom.sh", ["headroom", "run"], {"BAD-NAME": "x"})


def test_render_unix_runner_omits_export_block_without_env(tmp_path: Path) -> None:
    record = _render_unix_runner(tmp_path / "run-headroom.sh", ["headroom", "run"])

    content = Path(record.path).read_text(encoding="utf-8")

    assert "export" not in content
    assert content == "#!/usr/bin/env bash\nset -euo pipefail\nexec headroom run\n"


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


def test_render_windows_runner_rejects_invalid_env_name(tmp_path: Path) -> None:
    with pytest.raises(click.ClickException, match="Invalid environment variable name"):
        _render_windows_runner(
            tmp_path / "run-headroom.ps1",
            tmp_path / "run-headroom.cmd",
            ["headroom", "run"],
            {"BAD-NAME": "x"},
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


def test_render_runner_scripts_threads_base_env_into_both_scripts(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "linux")
    monkeypatch.setattr(
        "headroom.install.supervisors.resolve_headroom_command", lambda: ["headroom"]
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _manifest(base_env={"HEADROOM_WORKSPACE_DIR": "/custom/workspace"})

    records = render_runner_scripts(manifest)

    for record in records:
        content = Path(record.path).read_text(encoding="utf-8")
        assert "export HEADROOM_WORKSPACE_DIR=/custom/workspace" in content


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


def _stub_windows_task_install(monkeypatch, tmp_path: Path) -> None:
    run_script = tmp_path / "run-headroom.cmd"
    ensure_script = tmp_path / "ensure-headroom.cmd"
    monkeypatch.setattr(
        "headroom.install.supervisors.render_runner_scripts",
        lambda manifest: [
            type("Record", (), {"kind": "script", "path": str(run_script)})(),
            type("Record", (), {"kind": "script", "path": str(ensure_script)})(),
        ],
    )
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "win32")
    monkeypatch.setattr(
        "headroom.install.supervisors.windows_ensure_cmd_path",
        lambda profile: ensure_script,
    )


def test_install_windows_user_tasks_prefers_verified_s4u(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _stub_windows_task_install(monkeypatch, tmp_path)
    registrations: list[tuple[str, str, str]] = []

    def fake_register(name: str, xml: str, *, expected_logon_type: str) -> None:
        registrations.append((name, xml, expected_logon_type))

    monkeypatch.setattr("headroom.install.supervisors._register_windows_task", fake_register)

    records = install_supervisor(_manifest(supervisor=SupervisorKind.TASK.value))

    assert [(name, logon_type) for name, _xml, logon_type in registrations] == [
        ("headroom-default-startup", "S4U"),
        ("headroom-default-health", "S4U"),
    ]
    assert "<BootTrigger>" in registrations[0][1]
    assert "<TimeTrigger>" in registrations[1][1]
    assert [record.path for record in records if record.kind == "windows-task"] == [
        "headroom-default-startup",
        "headroom-default-health",
    ]
    assert "session-scoped" not in capsys.readouterr().out


@pytest.mark.parametrize("failed_task", ["headroom-default-startup", "headroom-default-health"])
def test_install_windows_user_task_failure_cleans_up_and_falls_back(
    monkeypatch, tmp_path: Path, capsys, failed_task: str
) -> None:
    _stub_windows_task_install(monkeypatch, tmp_path)
    registrations: list[tuple[str, str, str]] = []
    deletions: list[str] = []

    def fake_register(name: str, xml: str, *, expected_logon_type: str) -> None:
        registrations.append((name, xml, expected_logon_type))
        if name == failed_task and expected_logon_type == "S4U":
            raise _WindowsTaskRegistrationError("registration denied")

    monkeypatch.setattr("headroom.install.supervisors._register_windows_task", fake_register)
    monkeypatch.setattr(
        "headroom.install.supervisors._cleanup_windows_tasks_for_fallback",
        lambda names, must_delete: deletions.extend(names),
    )

    records = install_supervisor(_manifest(supervisor=SupervisorKind.TASK.value))

    assert deletions == ["headroom-default-startup", "headroom-default-health"]
    fallback_name, fallback_xml, fallback_logon_type = registrations[-1]
    assert fallback_name == "headroom-default-health"
    assert fallback_logon_type == "InteractiveToken"
    assert "<TimeTrigger>" in fallback_xml
    assert "<LogonType>InteractiveToken</LogonType>" in fallback_xml
    assert "<RunLevel>LeastPrivilege</RunLevel>" in fallback_xml
    assert "<Interval>PT5M</Interval>" in fallback_xml
    assert [record.path for record in records if record.kind == "windows-task"] == [
        "headroom-default-health"
    ]
    warning = capsys.readouterr().out
    assert "session-scoped" in warning
    assert "will not run while you are logged off" in warning


def test_install_windows_user_task_silent_downgrade_uses_fallback(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _stub_windows_task_install(monkeypatch, tmp_path)
    registered_xml: dict[str, str] = {}
    query_count = 0
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs):
        nonlocal query_count
        calls.append(command)
        name = command[command.index("/TN") + 1]
        if command[1] == "/Create":
            registered_xml[name] = Path(command[command.index("/XML") + 1]).read_text(
                encoding="utf-16"
            )
            return _LaunchctlResult()
        if command[1] == "/Query":
            if name not in registered_xml:
                return _LaunchctlResult(
                    1, stderr="ERROR: The system cannot find the file specified."
                )
            query_count += 1
            xml = registered_xml[name]
            if query_count == 1:
                xml = xml.replace(
                    "<LogonType>S4U</LogonType>", "<LogonType>InteractiveToken</LogonType>"
                )
            return _LaunchctlResult(stdout=xml)
        registered_xml.pop(name, None)
        return _LaunchctlResult()

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)

    records = install_supervisor(_manifest(supervisor=SupervisorKind.TASK.value))

    deletes = [call for call in calls if call[1] == "/Delete"]
    assert [call[call.index("/TN") + 1] for call in deletes] == [
        "headroom-default-startup",
        "headroom-default-health",
    ]
    assert "headroom-default-startup" not in registered_xml
    assert "<LogonType>InteractiveToken</LogonType>" in registered_xml["headroom-default-health"]
    assert [record.path for record in records if record.kind == "windows-task"] == [
        "headroom-default-health"
    ]
    assert "session-scoped" in capsys.readouterr().out


def test_install_windows_user_task_fallback_failure_propagates(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _stub_windows_task_install(monkeypatch, tmp_path)

    def fake_register(name: str, xml: str, *, expected_logon_type: str) -> None:
        raise _WindowsTaskRegistrationError(f"{expected_logon_type} registration denied")

    monkeypatch.setattr("headroom.install.supervisors._register_windows_task", fake_register)
    monkeypatch.setattr(
        "headroom.install.supervisors._cleanup_windows_tasks_for_fallback",
        lambda names, must_delete: None,
    )

    with pytest.raises(_WindowsTaskRegistrationError, match="InteractiveToken registration denied"):
        install_supervisor(_manifest(supervisor=SupervisorKind.TASK.value))

    assert "session-scoped" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("query_state", "error_match"),
    [
        ("survives", "remains registered"),
        ("delete_denied", "Could not clean up"),
    ],
)
def test_install_windows_user_task_cleanup_failure_prevents_fallback(
    monkeypatch, tmp_path: Path, capsys, query_state: str, error_match: str
) -> None:
    _stub_windows_task_install(monkeypatch, tmp_path)
    registrations: list[str] = []

    def fake_register(name: str, xml: str, *, expected_logon_type: str) -> None:
        registrations.append(expected_logon_type)
        raise _WindowsTaskRegistrationError("S4U registration denied", task_created=True)

    def fake_run(command: list[str], **kwargs):
        if query_state == "survives" and command[1] in ("/Delete", "/Query"):
            return _LaunchctlResult(stdout="<Task />")
        return _LaunchctlResult(1, stderr="ERROR: Access is denied.")

    monkeypatch.setattr("headroom.install.supervisors._register_windows_task", fake_register)
    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)

    with pytest.raises(_WindowsTaskRegistrationError, match=error_match):
        install_supervisor(_manifest(supervisor=SupervisorKind.TASK.value))

    assert registrations == ["S4U"]
    assert "session-scoped" not in capsys.readouterr().out


def test_install_windows_system_tasks_use_verified_service_account(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _stub_windows_task_install(monkeypatch, tmp_path)
    registrations: list[tuple[str, str, str]] = []

    def fake_register(name: str, xml: str, *, expected_logon_type: str) -> None:
        registrations.append((name, xml, expected_logon_type))

    monkeypatch.setattr("headroom.install.supervisors._register_windows_task", fake_register)

    records = install_supervisor(_manifest(scope="system", supervisor=SupervisorKind.TASK.value))

    assert [(name, logon_type) for name, _xml, logon_type in registrations] == [
        ("headroom-default-startup", "ServiceAccount"),
        ("headroom-default-health", "ServiceAccount"),
    ]
    assert "<BootTrigger>" in registrations[0][1]
    assert "<TimeTrigger>" in registrations[1][1]
    assert all(
        "<LogonType>ServiceAccount</LogonType>" in xml for _name, xml, _type in registrations
    )
    assert all("<UserId>S-1-5-18</UserId>" in xml for _name, xml, _type in registrations)
    assert all(
        "<RunLevel>HighestAvailable</RunLevel>" in xml for _name, xml, _type in registrations
    )
    assert [record.path for record in records if record.kind == "windows-task"] == [
        "headroom-default-startup",
        "headroom-default-health",
    ]
    assert "session-scoped" not in capsys.readouterr().out


def test_install_windows_system_task_failure_does_not_fall_back(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _stub_windows_task_install(monkeypatch, tmp_path)
    registrations: list[tuple[str, str]] = []

    def fake_register(name: str, xml: str, *, expected_logon_type: str) -> None:
        registrations.append((name, expected_logon_type))
        raise _WindowsTaskRegistrationError("system registration denied")

    monkeypatch.setattr("headroom.install.supervisors._register_windows_task", fake_register)

    with pytest.raises(_WindowsTaskRegistrationError, match="system registration denied"):
        install_supervisor(_manifest(scope="system", supervisor=SupervisorKind.TASK.value))

    assert registrations == [("headroom-default-startup", "ServiceAccount")]
    assert "session-scoped" not in capsys.readouterr().out


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

    def fake_run(command, **kwargs):
        calls.append(command)
        if isinstance(command, list) and len(command) > 1 and command[1] == "/Query":
            return _LaunchctlResult(
                stdout=(
                    '<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">'
                    "<Principals><Principal><LogonType>S4U</LogonType></Principal></Principals>"
                    "</Task>"
                )
            )
        return _LaunchctlResult(0)

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)
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
    # Regression for #1654: the create command must be a single pre-quoted
    # string (bypassing list2cmdline) with the inner quotes backslash-escaped
    # and `start= auto` as a separate trailing token.
    assert (
        "sc.exe create headroom-default "
        'binPath= "cmd.exe /c \\"C:\\tmp\\default\\run-headroom.cmd\\"" start= auto'
    ) in calls
    # #2453: tasks are registered from S4U/hidden XML via `schtasks /XML`, not
    # interactive-token flag creation. Assert the startup and health tasks are
    # each created from an XML file (the temp path varies).
    task_creates = [
        c for c in calls if isinstance(c, list) and c[:2] == ["schtasks", "/Create"] and "/XML" in c
    ]
    created_names = {c[c.index("/TN") + 1] for c in task_creates}
    assert {"headroom-default-startup", "headroom-default-health"} <= created_names
    for c in task_creates:
        assert c[-1] == "/F"

    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "plan9")
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "plan9")
    with pytest.raises(click.ClickException, match="not supported"):
        install_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))


def test_install_supervisor_retries_bootstrap_until_launchd_settles(
    monkeypatch, tmp_path: Path
) -> None:
    # Same EIO-after-bootout race start_supervisor already rides out, but hit
    # via install_supervisor's own unconditional bootout+bootstrap sequence on
    # every apply (issue: this call site had no retry at all before).
    run_script = tmp_path / "run-headroom.sh"
    monkeypatch.setattr(
        "headroom.install.supervisors.render_runner_scripts",
        lambda manifest: [
            type("Record", (), {"kind": "script", "path": run_script.as_posix()})(),
        ],
    )
    plist_path = tmp_path / "com.headroom.default.plist"
    monkeypatch.setattr(
        "headroom.install.supervisors._macos_launchd_plist",
        lambda manifest, script, interval=None: (plist_path, "plist"),
    )
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "darwin")
    monkeypatch.setattr("headroom.install.supervisors.os.getuid", lambda: 123, raising=False)
    monkeypatch.setattr("headroom.install.supervisors.time.sleep", lambda _s: None)
    bootstrap_attempts = 0

    def fake_run(command, **kwargs):
        nonlocal bootstrap_attempts
        if command[1] == "bootout":
            return _LaunchctlResult(0)
        bootstrap_attempts += 1
        if bootstrap_attempts < 3:
            return _LaunchctlResult(5, stderr="Bootstrap failed: 5: Input/output error")
        return _LaunchctlResult(0)

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)

    install_supervisor(_manifest(supervisor=SupervisorKind.SERVICE.value))
    assert bootstrap_attempts == 3


def test_install_supervisor_raises_after_bootstrap_keeps_failing(
    monkeypatch, tmp_path: Path
) -> None:
    run_script = tmp_path / "run-headroom.sh"
    monkeypatch.setattr(
        "headroom.install.supervisors.render_runner_scripts",
        lambda manifest: [
            type("Record", (), {"kind": "script", "path": run_script.as_posix()})(),
        ],
    )
    plist_path = tmp_path / "com.headroom.default.plist"
    monkeypatch.setattr(
        "headroom.install.supervisors._macos_launchd_plist",
        lambda manifest, script, interval=None: (plist_path, "plist"),
    )
    monkeypatch.setattr("headroom.install.supervisors.sys.platform", "darwin")
    monkeypatch.setattr("headroom.install.supervisors.os.getuid", lambda: 123, raising=False)
    monkeypatch.setattr("headroom.install.supervisors.time.sleep", lambda _s: None)
    monkeypatch.setattr("headroom.install.supervisors._MACOS_BOOTSTRAP_RETRIES", 3)

    def fake_run(command, **kwargs):
        if command[1] == "bootout":
            return _LaunchctlResult(0)
        return _LaunchctlResult(5, stderr="Bootstrap failed: 5: Input/output error")

    monkeypatch.setattr("headroom.install.supervisors.subprocess.run", fake_run)

    with pytest.raises(click.ClickException, match="could not bootstrap"):
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
