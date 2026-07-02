from __future__ import annotations

import os
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

from headroom.install.models import DeploymentManifest, InstallPreset
from headroom.install.runtime import (
    _clear_pid,
    _deployment_env,
    _mount_source,
    _read_pid,
    _runtime_env,
    _write_pid,
    acquire_runtime_start_lock,
    build_runtime_command,
    resolve_headroom_command,
    run_foreground,
    runtime_status,
    start_detached_agent,
    start_persistent_docker,
    stop_runtime,
    wait_ready,
)


def test_build_runtime_command_for_docker_includes_deployment_env(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-docker",
        runtime_kind="docker",
        supervisor_kind="none",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        image="ghcr.io/chopratejas/headroom:latest",
        base_env={"HEADROOM_PORT": "8787"},
        proxy_args=["--host", "127.0.0.1", "--port", "8787"],
    )

    command = build_runtime_command(manifest)

    joined = " ".join(command)
    assert command[:3] == ["docker", "run", "--rm"]
    assert "HEADROOM_DEPLOYMENT_PROFILE=default" in joined
    assert "HEADROOM_DEPLOYMENT_PRESET=persistent-docker" in joined
    assert "127.0.0.1:8787:8787" in joined
    assert "ghcr.io/chopratejas/headroom:latest" in command
    # Canonical Headroom filesystem contract (issue #175) forwarded into
    # the container.
    assert "HEADROOM_WORKSPACE_DIR=/tmp/headroom-home/.headroom" in command
    assert "HEADROOM_CONFIG_DIR=/tmp/headroom-home/.headroom/config" in command


def test_build_runtime_command_for_docker_matches_wrapper_parity(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-docker",
        runtime_kind="docker",
        supervisor_kind="none",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        image="ghcr.io/chopratejas/headroom:latest",
        base_env={"HEADROOM_PORT": "8787"},
        proxy_args=["--host", "127.0.0.1", "--port", "8787"],
    )

    command = build_runtime_command(manifest)

    assert (tmp_path / ".headroom").is_dir()
    assert (tmp_path / ".claude").is_dir()
    assert (tmp_path / ".codex").is_dir()
    assert (tmp_path / ".gemini").is_dir()
    assert "--env" in command
    joined = " ".join(command)
    assert "ANTHROPIC_API_KEY" in joined
    assert "OPENAI_API_KEY" in joined


def test_build_runtime_command_for_docker_does_not_duplicate_entrypoint(
    monkeypatch, tmp_path: Path
) -> None:
    """The image ENTRYPOINT is already ``["headroom", "proxy"]`` (Dockerfile),
    so the args appended after the image name must NOT re-add ``headroom proxy``
    or Docker runs ``headroom proxy headroom proxy ...`` and Click aborts with
    "Got unexpected extra arguments (headroom proxy)" (issue #833)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-docker",
        runtime_kind="docker",
        supervisor_kind="none",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        image="ghcr.io/chopratejas/headroom:latest",
        base_env={"HEADROOM_PORT": "8787"},
        proxy_args=["--host", "127.0.0.1", "--port", "8787", "--backend", "anthropic"],
    )

    command = build_runtime_command(manifest)

    # Everything after the image name is what Docker appends to the ENTRYPOINT.
    image_idx = command.index(manifest.image)
    container_args = command[image_idx + 1 :]
    assert "headroom" not in container_args, (
        f"container args re-add the ENTRYPOINT — got {container_args}"
    )
    assert "proxy" not in container_args, (
        f"container args re-add the ENTRYPOINT — got {container_args}"
    )
    # The container must still bind on all interfaces and keep the real flags.
    assert container_args[:2] == ["--host", "0.0.0.0"]
    assert container_args[2:] == ["--port", "8787", "--backend", "anthropic"]


def test_resolve_headroom_command_prefers_headroom_binary(monkeypatch) -> None:
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/headroom" if name == "headroom" else None
    )

    assert resolve_headroom_command() == ["/usr/bin/headroom"]


def test_resolve_headroom_command_falls_back_to_python_module(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("headroom.install.runtime.sys.executable", "/usr/bin/python")
    assert resolve_headroom_command() == ["/usr/bin/python", "-m", "headroom.cli"]


def test_runtime_env_and_mount_source(monkeypatch) -> None:
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        base_env={"EXTRA": "1"},
    )
    monkeypatch.setattr("headroom.install.runtime.os.environ", {"BASE": "x"})

    assert _deployment_env(manifest) == {
        "HEADROOM_DEPLOYMENT_PROFILE": "default",
        "HEADROOM_DEPLOYMENT_PRESET": "persistent-service",
        "HEADROOM_DEPLOYMENT_RUNTIME": "python",
        "HEADROOM_DEPLOYMENT_SUPERVISOR": "service",
        "HEADROOM_DEPLOYMENT_SCOPE": "user",
    }
    assert _runtime_env(manifest)["BASE"] == "x"
    assert _runtime_env(manifest)["EXTRA"] == "1"
    assert _runtime_env(manifest)["HEADROOM_DEPLOYMENT_PROFILE"] == "default"

    monkeypatch.setattr("headroom.install.runtime.sys.platform", "win32")
    assert _mount_source("C:\\Users\\me", ".headroom") == "C:\\Users\\me\\.headroom"
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "linux")
    assert _mount_source("/home/me", ".headroom") == "/home/me/.headroom"


def test_build_runtime_command_python_and_docker_user(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("headroom.install.runtime.sys.executable", "/usr/bin/python")
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        proxy_args=["--host", "127.0.0.1", "--port", "8787"],
    )
    assert build_runtime_command(manifest) == [
        "/usr/bin/python",
        "-m",
        "headroom.cli",
        "proxy",
        "--host",
        "127.0.0.1",
        "--port",
        "8787",
    ]

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "linux")
    monkeypatch.setattr("headroom.install.runtime.os.getuid", lambda: 1000, raising=False)
    monkeypatch.setattr("headroom.install.runtime.os.getgid", lambda: 1001, raising=False)
    docker_manifest = DeploymentManifest(
        profile="default",
        preset="persistent-docker",
        runtime_kind="docker",
        supervisor_kind="none",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        image="ghcr.io/chopratejas/headroom:latest",
        base_env={"HEADROOM_PORT": "8787"},
        proxy_args=["--host", "127.0.0.1", "--port", "8787"],
    )
    command = build_runtime_command(docker_manifest)
    assert "--user" in command
    assert "1000:1001" in command


def test_read_pid_handles_invalid_content(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pid_file = tmp_path / ".headroom" / "deploy" / "default" / "runner.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("not-a-pid", encoding="utf-8")

    assert _read_pid("default") is None
    _clear_pid("default")
    assert not pid_file.exists()


def test_write_read_and_clear_pid(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_pid("default", 456)
    assert _read_pid("default") == 456
    _clear_pid("default")
    assert _read_pid("default") is None


def test_runtime_start_lock_is_nonblocking(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with acquire_runtime_start_lock("default") as first_acquired:
        assert first_acquired is True
        with acquire_runtime_start_lock("default") as second_acquired:
            assert second_acquired is False

    with acquire_runtime_start_lock("default") as acquired_after_release:
        assert acquired_after_release is True


def test_runtime_start_lock_blocks_another_process(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    script = (
        "from headroom.install.runtime import acquire_runtime_start_lock\n"
        "with acquire_runtime_start_lock('default') as acquired:\n"
        "    print(acquired)\n"
    )
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(Path.cwd())}

    with acquire_runtime_start_lock("default") as acquired:
        assert acquired is True
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=True,
            env=env,
            text=True,
        )

    assert result.stdout.strip() == "False"


def test_run_foreground_and_detached_helpers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "headroom.install.runtime.build_runtime_command", lambda manifest: ["headroom", "proxy"]
    )
    monkeypatch.setattr("headroom.install.runtime._runtime_env", lambda manifest: {"ENV": "1"})
    signal_calls: list[int] = []
    monkeypatch.setattr(
        "headroom.install.runtime.signal.signal", lambda sig, fn: signal_calls.append(sig)
    )

    class FakeProc:
        def __init__(self, returncode: int = 0, pid: int = 321) -> None:
            self.returncode = returncode
            self.pid = pid
            self.terminated = False
            self.killed = False

        def wait(self, timeout: int | None = None) -> int:
            return self.returncode

        def poll(self):
            return None if not self.terminated else self.returncode

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    fake_proc = FakeProc(returncode=7)
    popen_calls: list[tuple[list[str], dict]] = []

    def fake_popen(command: list[str], **kwargs):
        popen_calls.append((command, kwargs))
        return fake_proc

    monkeypatch.setattr("headroom.install.runtime.subprocess.Popen", fake_popen)
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
    )
    assert run_foreground(manifest) == 7
    assert popen_calls[0][0] == ["headroom", "proxy"]
    assert signal.SIGINT in signal_calls
    assert signal.SIGTERM in signal_calls
    assert _read_pid("default") is None

    monkeypatch.setattr("headroom.install.runtime.resolve_headroom_command", lambda: ["headroom"])
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "win32")
    monkeypatch.setattr("headroom.install.runtime.subprocess.DETACHED_PROCESS", 1, raising=False)
    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.CREATE_NEW_PROCESS_GROUP", 2, raising=False
    )
    fake_proc_nt = FakeProc()
    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.Popen", lambda command, **kwargs: fake_proc_nt
    )
    assert start_detached_agent("demo") is fake_proc_nt

    monkeypatch.setattr("headroom.install.runtime.sys.platform", "linux")
    fake_proc_posix = FakeProc()
    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.Popen", lambda command, **kwargs: fake_proc_posix
    )
    assert start_detached_agent("demo") is fake_proc_posix


def test_start_detached_agent_closes_parent_log_fd(monkeypatch, tmp_path: Path) -> None:
    """The parent must close its copy of the log file after Popen.

    The child inherits the descriptor, so leaving the parent's copy open
    leaks one fd per call and pins the log file open against rotation.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.runtime.resolve_headroom_command", lambda: ["headroom"])
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "linux")

    captured: dict[str, object] = {}

    class FakeProc:
        pid = 999

    def fake_popen(command: list[str], **kwargs):
        captured["stdout"] = kwargs["stdout"]
        captured["stderr"] = kwargs["stderr"]
        return FakeProc()

    monkeypatch.setattr("headroom.install.runtime.subprocess.Popen", fake_popen)

    start_detached_agent("demo")

    log_handle = captured["stdout"]
    # Same handle is passed to both streams, and the parent closed it.
    assert captured["stderr"] is log_handle
    assert log_handle.closed is True


def test_start_detached_agent_closes_log_fd_when_popen_raises(monkeypatch, tmp_path: Path) -> None:
    """A Popen failure must not leak the just-opened log file handle."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.runtime.resolve_headroom_command", lambda: ["headroom"])
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "linux")

    captured: dict[str, object] = {}

    def boom(command: list[str], **kwargs):
        captured["stdout"] = kwargs["stdout"]
        raise OSError("spawn failed")

    monkeypatch.setattr("headroom.install.runtime.subprocess.Popen", boom)

    with pytest.raises(OSError, match="spawn failed"):
        start_detached_agent("demo")

    assert captured["stdout"].closed is True


def test_start_stop_wait_and_runtime_status_branches(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.run",
        lambda command, **kwargs: calls.append(command) or type("Result", (), {"stdout": ""})(),
    )
    monkeypatch.setattr(
        "headroom.install.runtime.build_runtime_command",
        lambda manifest: [
            "docker",
            "run",
            "--rm",
            "--name",
            "demo",
            "-p",
            "127.0.0.1:8787:8787",
            "image",
        ],
    )
    manifest = DeploymentManifest(
        profile="default",
        preset=InstallPreset.PERSISTENT_DOCKER.value,
        runtime_kind="docker",
        supervisor_kind="none",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        container_name="headroom-default",
    )
    start_persistent_docker(manifest)
    assert calls == [
        ["docker", "rm", "-f", "headroom-default"],
        [
            "docker",
            "run",
            "-d",
            "--restart",
            "unless-stopped",
            "--name",
            "headroom-default",
            "-p",
            "127.0.0.1:8787:8787",
            "image",
        ],
    ]

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    python_manifest = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        health_url="http://127.0.0.1:8787/health",
    )
    _write_pid("default", 123)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "headroom.install.runtime.os.kill", lambda pid, sig: killed.append((pid, sig))
    )
    stop_runtime(python_manifest)
    assert killed == [(123, signal.SIGTERM)]
    assert _read_pid("default") is None

    _write_pid("default", 124)
    monkeypatch.setattr(
        "headroom.install.runtime.os.kill",
        lambda pid, sig: (_ for _ in ()).throw(OSError("gone")),
    )
    stop_runtime(python_manifest)
    assert _read_pid("default") is None

    probe_results = iter([False, False, True])
    sleeps: list[int] = []
    monkeypatch.setattr("headroom.install.runtime.probe_ready", lambda url: next(probe_results))
    monkeypatch.setattr(
        "headroom.install.runtime.time.sleep", lambda seconds: sleeps.append(seconds)
    )
    assert wait_ready(python_manifest, timeout_seconds=3) is True
    assert sleeps == [1, 1]

    monkeypatch.setattr("headroom.install.runtime.probe_ready", lambda url: False)
    sleeps.clear()
    assert wait_ready(python_manifest, timeout_seconds=2) is False
    assert sleeps == [1, 1]

    class Result:
        def __init__(self, stdout: str = "") -> None:
            self.stdout = stdout

    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.run",
        lambda command, **kwargs: Result(stdout=""),
    )
    assert runtime_status(manifest) == "stopped"
    assert runtime_status(python_manifest) == "stopped"

    _write_pid("default", 125)
    monkeypatch.setattr("headroom.install.runtime.pid_alive", lambda pid: False)
    assert runtime_status(python_manifest) == "stopped"


def test_stop_runtime_for_docker_stops_and_removes_container(monkeypatch) -> None:
    calls: list[list[str]] = []
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-docker",
        runtime_kind="docker",
        supervisor_kind="none",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        container_name="headroom-default",
    )

    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.run",
        lambda command, **kwargs: calls.append(command),
    )

    stop_runtime(manifest)

    assert calls == [
        ["docker", "stop", "headroom-default"],
        ["docker", "rm", "-f", "headroom-default"],
    ]


def test_runtime_status_reads_container_and_pid_state(monkeypatch, tmp_path: Path) -> None:
    docker_manifest = DeploymentManifest(
        profile="default",
        preset="persistent-docker",
        runtime_kind="docker",
        supervisor_kind="none",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        container_name="headroom-default",
    )

    class Result:
        def __init__(self, stdout: str = "") -> None:
            self.stdout = stdout

    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.run",
        lambda command, **kwargs: Result(stdout="headroom-default\n"),
    )
    assert runtime_status(docker_manifest) == "running"

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pid_file = tmp_path / ".headroom" / "deploy" / "default" / "runner.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("123", encoding="utf-8")
    monkeypatch.setattr("headroom.install.runtime.pid_alive", lambda pid: True)
    python_manifest = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
    )
    assert runtime_status(python_manifest) == "running"


def _python_service_manifest() -> DeploymentManifest:
    return DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
    )


def test_runtime_status_reports_live_pid_without_terminating(monkeypatch, tmp_path: Path) -> None:
    """#1544: status on a live detached PID stays 'running' and never signals it."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pid_file = tmp_path / ".headroom" / "deploy" / "default" / "runner.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("25212", encoding="utf-8")

    def fail_kill(pid: int, sig: int) -> None:
        raise AssertionError(f"status must not signal the live proxy (pid={pid}, sig={sig})")

    monkeypatch.setattr("headroom.install.runtime.os.kill", fail_kill)
    monkeypatch.setattr("headroom.install.runtime.pid_alive", lambda pid: True)

    assert runtime_status(_python_service_manifest()) == "running"
    assert pid_file.exists()  # status left the deployment untouched


def test_runtime_status_survives_winerror87_systemerror(monkeypatch, tmp_path: Path) -> None:
    """#1544: a WinError 87 SystemError from the liveness probe yields 'stopped', not a crash."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pid_file = tmp_path / ".headroom" / "deploy" / "default" / "runner.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("25212", encoding="utf-8")

    # Force the psutil fast-path to bail so the os.kill fallback runs...
    fake_psutil = types.SimpleNamespace(
        pid_exists=lambda pid: (_ for _ in ()).throw(RuntimeError("no psutil"))
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    # ...where Windows surfaces WinError 87 as a SystemError, not an OSError.
    monkeypatch.setattr(
        "headroom._subprocess.os.kill",
        lambda pid, sig: (_ for _ in ()).throw(SystemError("WinError 87")),
    )

    assert runtime_status(_python_service_manifest()) == "stopped"
