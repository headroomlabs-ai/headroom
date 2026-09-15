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
    _process_identity,
    _process_matches_runtime,
    _read_pid,
    _runtime_env,
    _write_pid,
    acquire_runtime_start_lock,
    build_runtime_command,
    resolve_headroom_command,
    run_foreground,
    runtime_ownership,
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
        image="ghcr.io/headroomlabs-ai/headroom:latest",
        base_env={"HEADROOM_PORT": "8787"},
        proxy_args=["--host", "127.0.0.1", "--port", "8787"],
    )

    command = build_runtime_command(manifest)

    joined = " ".join(command)
    assert command[:3] == ["docker", "run", "--rm"]
    assert "HEADROOM_DEPLOYMENT_PROFILE=default" in joined
    assert "HEADROOM_DEPLOYMENT_PRESET=persistent-docker" in joined
    assert "127.0.0.1:8787:8787" in joined
    assert "ghcr.io/headroomlabs-ai/headroom:latest" in command
    # Canonical Headroom filesystem contract (issue #175) forwarded into
    # the container.
    assert "HEADROOM_WORKSPACE_DIR=/tmp/headroom-home/.headroom" in command
    assert "HEADROOM_CONFIG_DIR=/tmp/headroom-home/.headroom/config" in command


def test_build_runtime_command_for_docker_includes_gpu_passthrough(
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
        base_env={"HEADROOM_PORT": "8787", "HEADROOM_DOCKER_GPUS": "all"},
        proxy_args=["--host", "127.0.0.1", "--port", "8787"],
    )

    command = build_runtime_command(manifest)

    assert command[command.index("--gpus") + 1] == "all"


def test_build_runtime_command_docker_manifest_env_beats_host_passthrough(
    monkeypatch, tmp_path: Path
) -> None:
    """A manifest value must win over a conflicting host export.

    The manifest pins ``HEADROOM_BACKEND=anthropic``; the host process has a
    stale ``HEADROOM_BACKEND=anyllm`` exported. Both start with the
    ``HEADROOM_`` passthrough prefix, so without the dedupe the command emits
    ``--env HEADROOM_BACKEND=anthropic`` and then a bare ``--env
    HEADROOM_BACKEND``. Docker resolves duplicate ``--env`` last-wins, so the
    bare passthrough reads the host value and silently overrides the manifest,
    diverging the container from its deployment config.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HEADROOM_BACKEND", "anyllm")
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
        base_env={"HEADROOM_PORT": "8787", "HEADROOM_BACKEND": "anthropic"},
        proxy_args=["--host", "127.0.0.1", "--port", "8787"],
    )

    command = build_runtime_command(manifest)

    # The manifest value is emitted...
    assert "HEADROOM_BACKEND=anthropic" in command
    # ...and no bare `--env HEADROOM_BACKEND` follows it to pull in the host
    # value. `in` on the list matches the exact token, so the pinned
    # `HEADROOM_BACKEND=anthropic` element does not count here.
    assert "HEADROOM_BACKEND" not in command


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
        image="ghcr.io/headroomlabs-ai/headroom:latest",
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
        image="ghcr.io/headroomlabs-ai/headroom:latest",
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
    # Force the Docker path deterministically regardless of the test host's
    # `docker` binary (it might resolve to a podman shim).
    monkeypatch.setenv("HEADROOM_CONTAINER_RUNTIME", "docker")
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
        image="ghcr.io/headroomlabs-ai/headroom:latest",
        base_env={"HEADROOM_PORT": "8787"},
        proxy_args=["--host", "127.0.0.1", "--port", "8787"],
    )
    command = build_runtime_command(docker_manifest)
    assert "--user" in command
    assert "1000:1001" in command
    assert "--userns=keep-id" not in command


def test_build_runtime_command_podman_uses_keep_id_not_user(monkeypatch, tmp_path: Path) -> None:
    """Under rootless Podman, --user <host-uid>:<host-gid> selects a subordinate
    UID that owns none of the bind mounts, so writes into ~/.headroom fail. The
    command must use --userns=keep-id and drop --user instead (#2804)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "linux")
    monkeypatch.setattr("headroom.install.runtime.os.getuid", lambda: 1000, raising=False)
    monkeypatch.setattr("headroom.install.runtime.os.getgid", lambda: 1001, raising=False)
    monkeypatch.setenv("HEADROOM_CONTAINER_RUNTIME", "podman")
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
        image="ghcr.io/headroomlabs-ai/headroom:latest",
        base_env={"HEADROOM_PORT": "8787"},
        proxy_args=["--host", "127.0.0.1", "--port", "8787"],
    )
    command = build_runtime_command(manifest)
    assert "--userns=keep-id" in command
    assert "--user" not in command
    assert "1000:1001" not in command


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
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "PYTHONPATH": str(Path.cwd()),
    }

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


def test_non_launchd_foreground_and_detached_helpers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "win32")
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
    monkeypatch.setattr("headroom.install.runtime.subprocess.CREATE_NO_WINDOW", 4, raising=False)
    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.CREATE_NEW_PROCESS_GROUP", 2, raising=False
    )
    nt_calls: list[tuple[list[str], dict]] = []
    fake_proc_nt = FakeProc()

    def fake_popen_nt(command: list[str], **kwargs):
        nt_calls.append((command, kwargs))
        return fake_proc_nt

    monkeypatch.setattr("headroom.install.runtime.subprocess.Popen", fake_popen_nt)
    assert start_detached_agent("demo") is fake_proc_nt
    # DETACHED_PROCESS is not used: it makes CREATE_NO_WINDOW a no-op on
    # Windows, so a detached console child would pop up a visible window.
    assert nt_calls[0][1]["creationflags"] == 4 | 2

    monkeypatch.setattr("headroom.install.runtime.sys.platform", "linux")
    fake_proc_posix = FakeProc()
    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.Popen", lambda command, **kwargs: fake_proc_posix
    )
    assert start_detached_agent("demo") is fake_proc_posix


def test_launchd_exec_handoff_uses_current_pid_and_never_popens(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "darwin")
    monkeypatch.setattr(
        "headroom.install.runtime.build_runtime_command",
        lambda manifest: ["/usr/bin/python3", "-m", "headroom.cli", "proxy"],
    )
    expected_env = {"HEADROOM_TEST": "exec"}
    monkeypatch.setattr("headroom.install.runtime._runtime_env", lambda manifest: expected_env)
    monkeypatch.setattr("headroom.install.runtime.os.getpid", lambda: 4321)

    exec_calls: list[tuple[str, list[str], dict[str, str]]] = []
    pid_at_exec: list[int | None] = []

    def fake_execvpe(file: str, args: list[str], env: dict[str, str]) -> None:
        exec_calls.append((file, args, env))
        pid_at_exec.append(_read_pid("default"))
        raise OSError("exec failed")

    monkeypatch.setattr("headroom.install.runtime.os.execvpe", fake_execvpe)
    monkeypatch.setattr("headroom.install.runtime.os.dup2", lambda source, target: None)
    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.Popen",
        lambda *args, **kwargs: pytest.fail("launchd handoff must not spawn a child"),
    )

    with pytest.raises(OSError, match="exec failed"):
        run_foreground(_python_service_manifest())

    assert exec_calls == [
        (
            "/usr/bin/python3",
            ["/usr/bin/python3", "-m", "headroom.cli", "proxy"],
            expected_env,
        )
    ]
    assert pid_at_exec == [4321]
    assert _read_pid("default") is None


def test_launchd_exec_preserves_runtime_contract(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "darwin")
    command = ["/usr/bin/python3", "-m", "headroom.cli", "proxy", "--port", "8787"]
    env = {"HEADROOM_DEPLOYMENT_PROFILE": "default", "HEADROOM_SECRET": "test"}
    monkeypatch.setattr("headroom.install.runtime.build_runtime_command", lambda manifest: command)
    monkeypatch.setattr("headroom.install.runtime._runtime_env", lambda manifest: env)
    monkeypatch.setattr("headroom.install.runtime.os.getpid", lambda: 9876)

    dup2_calls: list[tuple[int, int]] = []
    exec_calls: list[tuple[str, list[str], dict[str, str]]] = []

    class FakeLog:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def fileno(self) -> int:
            return 42

    monkeypatch.setattr(
        "headroom.install.runtime.open", lambda *args, **kwargs: FakeLog(), raising=False
    )

    def fake_dup2(source: int, target: int) -> None:
        dup2_calls.append((source, target))

    def fake_execvpe(file: str, args: list[str], runtime_env: dict[str, str]) -> None:
        exec_calls.append((file, args, runtime_env))
        raise OSError("exec failed")

    monkeypatch.setattr("headroom.install.runtime.os.dup2", fake_dup2)
    monkeypatch.setattr("headroom.install.runtime.os.execvpe", fake_execvpe)
    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.Popen",
        lambda *args, **kwargs: pytest.fail("launchd handoff must not spawn a child"),
    )

    with pytest.raises(OSError, match="exec failed"):
        run_foreground(_python_service_manifest())

    assert exec_calls == [(command[0], command, env)]
    assert len(dup2_calls) == 2
    assert dup2_calls[0][1:] == (1,)
    assert dup2_calls[1][1:] == (2,)
    assert dup2_calls == [(42, 1), (42, 2)]


def test_docker_runtime_rejects_run_foreground_split_owner(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "darwin")
    command = ["docker", "run", "--rm", "headroom:test"]
    env = {"HEADROOM_DEPLOYMENT_RUNTIME": "docker"}
    monkeypatch.setattr("headroom.install.runtime.build_runtime_command", lambda manifest: command)
    monkeypatch.setattr("headroom.install.runtime._runtime_env", lambda manifest: env)

    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.Popen",
        lambda *args, **kwargs: pytest.fail("Docker supervisor must own the container lifecycle"),
    )
    monkeypatch.setattr(
        "headroom.install.runtime.os.execvpe",
        lambda *args: pytest.fail("Docker launchd service must keep the daemon-owned Popen path"),
    )

    manifest = _python_service_manifest()
    manifest.runtime_kind = "docker"
    with pytest.raises(RuntimeError, match="Docker deployments must be started"):
        run_foreground(manifest)


def test_runtime_ownership_classifier_covers_launchd_docker_and_foreground(monkeypatch) -> None:
    python_manifest = _python_service_manifest()
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "darwin")
    assert runtime_ownership(python_manifest) == "launchd-exec"

    for preset, supervisor in (
        (InstallPreset.PERSISTENT_DOCKER.value, "none"),
        (InstallPreset.PERSISTENT_SERVICE.value, "service"),
        (InstallPreset.PERSISTENT_TASK.value, "task"),
    ):
        docker_manifest = _python_service_manifest()
        docker_manifest.preset = preset
        docker_manifest.runtime_kind = "docker"
        docker_manifest.supervisor_kind = supervisor
        assert runtime_ownership(docker_manifest) == "docker-supervisor"

    python_manifest.preset = "persistent-task"
    python_manifest.supervisor_kind = "task"
    assert runtime_ownership(python_manifest) == "popen"


def test_launchd_transition_keeps_one_listener_owner(monkeypatch, tmp_path: Path) -> None:
    """A launchd restart replaces the listener in place instead of orphaning a child."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "darwin")
    monkeypatch.setattr(
        "headroom.install.runtime.build_runtime_command",
        lambda manifest: ["/usr/bin/python3", "-m", "headroom.cli", "proxy"],
    )
    monkeypatch.setattr("headroom.install.runtime._runtime_env", lambda manifest: {})
    monkeypatch.setattr("headroom.install.runtime.os.getpid", lambda: 4321)
    monkeypatch.setattr("headroom.install.runtime.os.dup2", lambda source, target: None)
    monkeypatch.setattr("headroom.install.runtime.subprocess.Popen", lambda *a, **k: pytest.fail())

    exec_pids: list[int | None] = []

    class ExecHandoff(Exception):
        pass

    monkeypatch.setattr(
        "headroom.install.runtime.os.execvpe",
        lambda file, args, env: (exec_pids.append(_read_pid("default")), raise_exec()),
    )
    monkeypatch.setattr("headroom.install.runtime._clear_pid", lambda *args, **kwargs: None)

    def raise_exec() -> None:
        raise ExecHandoff

    manifest = _python_service_manifest()
    assert runtime_ownership(manifest) == "launchd-exec"
    with pytest.raises(ExecHandoff):
        run_foreground(manifest)
    with pytest.raises(ExecHandoff):
        run_foreground(manifest)

    assert exec_pids == [4321, 4321]
    assert _read_pid("default") == 4321


def test_status_and_stop_reject_a_reused_pid_without_signaling(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _python_service_manifest()
    _write_pid("default", 123)
    monkeypatch.setattr("headroom.install.runtime.pid_alive", lambda pid: True)
    monkeypatch.setattr("headroom.install.runtime.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.install.runtime._process_identity", lambda pid, current: False)
    monkeypatch.setattr(
        "headroom.install.runtime.os.kill",
        lambda *args: pytest.fail("reused PID must not be signaled"),
    )
    assert runtime_status(manifest) == "stopped"
    _write_pid("default", 123)
    stop_runtime(manifest)
    assert _read_pid("default") is None


def test_process_identity_must_match_headroom_command(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def cmdline(self) -> list[str]:
            return ["python", "unrelated-service.py", "--port", "8787"]

        def environ(self) -> dict[str, str]:
            return {
                "HEADROOM_DEPLOYMENT_PROFILE": "default",
                "HEADROOM_DEPLOYMENT_RUNTIME": "python",
            }

    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(Process=FakeProcess))
    assert not _process_matches_runtime(4321, _python_service_manifest())

    class MatchingProcess(FakeProcess):
        def cmdline(self) -> list[str]:
            return ["python", "-m", "headroom.cli", "proxy", "--port", "8787"]

        def environ(self) -> dict[str, str]:
            return {
                "HEADROOM_DEPLOYMENT_PROFILE": "default",
                "HEADROOM_DEPLOYMENT_RUNTIME": "python",
            }

    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(Process=MatchingProcess))
    assert _process_matches_runtime(4321, _python_service_manifest())


def test_process_identity_falls_back_without_psutil(monkeypatch) -> None:
    manifest = _python_service_manifest()
    monkeypatch.setattr(
        "headroom.install.runtime._read_proc_metadata",
        lambda pid: (
            ["python", "-m", "headroom.cli", "proxy", "--port", "8787"],
            {
                "HEADROOM_DEPLOYMENT_PROFILE": "default",
                "HEADROOM_DEPLOYMENT_RUNTIME": "python",
            },
        ),
    )
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert _process_matches_runtime(4321, manifest)


def test_process_identity_uses_macos_ps_fallback_without_psutil(monkeypatch) -> None:
    manifest = _python_service_manifest()
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "darwin")
    monkeypatch.setitem(sys.modules, "psutil", None)

    class Result:
        returncode = 0
        stdout = (
            "/usr/bin/python3 -m headroom.cli proxy --port 8787 "
            "HEADROOM_DEPLOYMENT_PROFILE=default "
            "HEADROOM_DEPLOYMENT_RUNTIME=python"
        )

    monkeypatch.setattr("headroom.install.runtime.run", lambda *a, **k: Result())
    assert _process_matches_runtime(4321, manifest)


def test_process_identity_uses_windows_wmi_command_line_without_psutil(monkeypatch) -> None:
    manifest = _python_service_manifest()
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "win32")
    monkeypatch.setitem(sys.modules, "psutil", None)

    class Result:
        returncode = 0
        stdout = (
            '"C:\\Python\\python.exe" -m headroom.cli proxy --port 8787 '
            "--headroom-deployment-profile default --headroom-deployment-runtime python"
        )

    calls: list[list[str]] = []
    monkeypatch.setattr(
        "headroom.install.runtime.run",
        lambda command, **kwargs: calls.append(command) or Result(),
    )

    assert _process_matches_runtime(4321, manifest)
    assert calls[0][:3] == ["powershell.exe", "-NoProfile", "-NonInteractive"]


def test_windows_runtime_status_and_stop_use_wmi_identity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "win32")
    monkeypatch.setitem(sys.modules, "psutil", None)
    manifest = _python_service_manifest()
    _write_pid("default", 4321)
    pid_states = iter([True, False, False])
    monkeypatch.setattr("headroom.install.runtime.pid_alive", lambda pid: next(pid_states))

    class Result:
        returncode = 0
        stdout = (
            '"C:\\Python\\python.exe" -m headroom.cli proxy --port 8787 '
            "--headroom-deployment-profile default --headroom-deployment-runtime python"
        )

    monkeypatch.setattr("headroom.install.runtime.run", lambda *a, **k: Result())
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "headroom.install.runtime.os.kill", lambda pid, sig: killed.append((pid, sig))
    )

    assert runtime_status(manifest) == "running"
    stop_runtime(manifest)
    assert killed == [(4321, signal.SIGTERM)]
    assert _read_pid("default") is None


def test_windows_wmi_same_port_without_deployment_markers_is_unknown_and_safe(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "win32")
    monkeypatch.setitem(sys.modules, "psutil", None)
    manifest = _python_service_manifest()

    class Result:
        returncode = 0
        stdout = '"C:\\Python\\python.exe" -m headroom.cli proxy --port 8787'

    monkeypatch.setattr("headroom.install.runtime.run", lambda *a, **k: Result())
    _write_pid("default", 4321)
    monkeypatch.setattr("headroom.install.runtime.pid_alive", lambda pid: True)
    monkeypatch.setattr(
        "headroom.install.runtime.os.kill",
        lambda *args: pytest.fail("unidentified same-port process must not be signaled"),
    )

    assert _process_identity(4321, manifest) is None
    assert runtime_status(manifest) == "unknown"
    with pytest.raises(RuntimeError, match="identity unknown"):
        stop_runtime(manifest)
    assert _read_pid("default") == 4321


def test_process_identity_without_supported_metadata_is_unknown_and_safe(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr("headroom.install.runtime._read_proc_metadata", lambda pid: None)
    manifest = _python_service_manifest()
    _write_pid("default", 4321)
    monkeypatch.setattr("headroom.install.runtime.pid_alive", lambda pid: True)
    monkeypatch.setattr(
        "headroom.install.runtime.os.kill",
        lambda *args: pytest.fail("unknown identity must not be signaled"),
    )

    assert _process_identity(4321, manifest) is None
    assert runtime_status(manifest) == "unknown"
    with pytest.raises(RuntimeError, match="identity unknown"):
        stop_runtime(manifest)
    assert _read_pid("default") == 4321


def test_runtime_status_does_not_use_ready_probe_for_failed_identity(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _python_service_manifest()
    _write_pid("default", 4321)
    monkeypatch.setattr("headroom.install.runtime.pid_alive", lambda pid: True)
    monkeypatch.setattr("headroom.install.runtime._process_identity", lambda pid, m: False)
    monkeypatch.setattr("headroom.install.runtime.probe_ready", lambda url: True)

    assert runtime_status(manifest) == "stopped"
    assert _read_pid("default") is None


def test_process_identity_rejects_current_pid(monkeypatch) -> None:
    monkeypatch.setattr("headroom.install.runtime.os.getpid", lambda: 4321)
    assert not _process_matches_runtime(4321, _python_service_manifest())


def test_darwin_task_preserves_popen_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.runtime.sys.platform", "darwin")
    monkeypatch.setattr(
        "headroom.install.runtime.build_runtime_command", lambda manifest: ["headroom", "proxy"]
    )
    monkeypatch.setattr("headroom.install.runtime._runtime_env", lambda manifest: {"ENV": "1"})

    class FakeProc:
        pid = 8765

        def wait(self, timeout: int | None = None) -> int:
            return 0

    popen_calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(
        "headroom.install.runtime.subprocess.Popen",
        lambda command, **kwargs: popen_calls.append((command, kwargs)) or FakeProc(),
    )
    monkeypatch.setattr(
        "headroom.install.runtime.os.execvpe",
        lambda *args: pytest.fail("Darwin task must preserve Popen"),
    )

    task_manifest = _python_service_manifest()
    task_manifest.preset = "persistent-task"
    task_manifest.supervisor_kind = "task"
    assert run_foreground(task_manifest) == 0
    assert popen_calls[0][0] == ["headroom", "proxy"]
    assert _read_pid("default") is None


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
        lambda command, **kwargs: (
            calls.append(command)
            or type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        ),
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
    monkeypatch.setattr("headroom.install.runtime._process_identity", lambda pid, manifest: True)
    monkeypatch.setattr("headroom.install.runtime.pid_alive", lambda pid: False)
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
    assert _read_pid("default") is None


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
        lambda command, **kwargs: (
            calls.append(command)
            or type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        ),
    )

    stop_runtime(manifest)

    assert calls == [
        ["docker", "stop", "headroom-default"],
        ["docker", "rm", "-f", "headroom-default"],
    ]


def test_stop_runtime_rejects_unknown_identity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _python_service_manifest()
    _write_pid("default", 4321)
    monkeypatch.setattr("headroom.install.runtime._process_identity", lambda pid, current: None)
    monkeypatch.setattr("headroom.install.runtime.pid_alive", lambda pid: True)

    with pytest.raises(RuntimeError, match="identity unknown"):
        stop_runtime(manifest)
    assert _read_pid("default") == 4321


def test_stop_runtime_rejects_malformed_pid_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _python_service_manifest()
    pid_file = tmp_path / ".headroom" / "deploy" / "default" / "runner.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("not-a-pid", encoding="utf-8")

    with pytest.raises(RuntimeError, match="runner.pid is invalid"):
        stop_runtime(manifest)
    assert pid_file.exists()


def test_runtime_status_reports_unknown_for_malformed_pid_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _python_service_manifest()
    pid_file = tmp_path / ".headroom" / "deploy" / "default" / "runner.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("not-a-pid", encoding="utf-8")

    assert runtime_status(manifest) == "unknown"
    assert pid_file.exists()


def test_stop_runtime_rejects_identity_change_before_clearing_pid(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _python_service_manifest()
    _write_pid("default", 4321)
    identities = iter([True, False])
    monkeypatch.setattr(
        "headroom.install.runtime._process_identity", lambda pid, current: next(identities)
    )
    monkeypatch.setattr("headroom.install.runtime.pid_alive", lambda pid: True)
    monkeypatch.setattr("headroom.install.runtime.os.kill", lambda pid, sig: None)

    with pytest.raises(RuntimeError, match="identity changed"):
        stop_runtime(manifest)
    assert _read_pid("default") == 4321


def test_stop_runtime_keeps_pid_when_sigterm_does_not_stop_process(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _python_service_manifest()
    _write_pid("default", 4321)
    monkeypatch.setattr("headroom.install.runtime._process_identity", lambda pid, current: True)
    monkeypatch.setattr("headroom.install.runtime.pid_alive", lambda pid: True)
    monkeypatch.setattr("headroom.install.runtime.os.kill", lambda pid, sig: None)
    monkeypatch.setattr("headroom.install.runtime.time.sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="remained alive"):
        stop_runtime(manifest)
    assert _read_pid("default") == 4321


def test_stop_runtime_surfaces_docker_stop_failure(monkeypatch) -> None:
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

    class Result:
        returncode = 1
        stdout = ""
        stderr = "permission denied"

    monkeypatch.setattr("headroom.install.runtime.run", lambda *args, **kwargs: Result())

    with pytest.raises(RuntimeError, match="docker stop failed"):
        stop_runtime(manifest)


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
    monkeypatch.setattr("headroom.install.runtime._process_identity", lambda pid, manifest: True)
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
    monkeypatch.setattr("headroom.install.runtime._process_identity", lambda pid, manifest: True)

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


class TestRestartCurrentDeployment:
    """detect_current_deployment / restart_current_deployment (settings apply)."""

    def _clear_deployment_env(self, monkeypatch) -> None:
        monkeypatch.delenv("HEADROOM_DEPLOYMENT_PROFILE", raising=False)
        monkeypatch.delenv("HEADROOM_DEPLOYMENT_PRESET", raising=False)

    def test_foreground_when_no_deployment_env(self, monkeypatch) -> None:
        from headroom.install import runtime as rt

        self._clear_deployment_env(monkeypatch)
        popen_calls: list = []
        monkeypatch.setattr(rt.subprocess, "Popen", lambda *a, **k: popen_calls.append(a))

        manifest, mode = rt.detect_current_deployment()
        assert manifest is None
        assert mode == "foreground"

        result = rt.restart_current_deployment()
        assert result["restarted"] is False
        assert result["mode"] == "foreground"
        assert "instruction" in result
        assert popen_calls == []  # never restarts a foreground proxy

    def test_docker_returns_host_command_without_restarting(self, monkeypatch) -> None:
        from headroom.install import runtime as rt

        monkeypatch.setenv("HEADROOM_DEPLOYMENT_PROFILE", "default")
        monkeypatch.setenv("HEADROOM_DEPLOYMENT_PRESET", InstallPreset.PERSISTENT_DOCKER.value)
        monkeypatch.setattr(rt, "load_manifest", lambda profile: None)
        popen_calls: list = []
        monkeypatch.setattr(rt.subprocess, "Popen", lambda *a, **k: popen_calls.append(a))

        _manifest, mode = rt.detect_current_deployment()
        assert mode == "docker"

        result = rt.restart_current_deployment()
        assert result["restarted"] is False
        assert result["mode"] == "docker"
        assert result["command"] == "headroom install restart --profile default"
        assert popen_calls == []  # cannot run docker from inside the container

    def test_runtime_env_docker_returns_host_command_without_restarting(self, monkeypatch) -> None:
        from headroom.install import runtime as rt

        monkeypatch.setenv("HEADROOM_DEPLOYMENT_PROFILE", "default")
        monkeypatch.setenv("HEADROOM_DEPLOYMENT_RUNTIME", "docker")
        monkeypatch.delenv("HEADROOM_DEPLOYMENT_PRESET", raising=False)
        monkeypatch.setattr(rt, "load_manifest", lambda profile: None)

        _manifest, mode = rt.detect_current_deployment()
        assert mode == "docker"

    def test_task_mode_detected_and_not_restarted(self, monkeypatch) -> None:
        """A persistent-task deployment must not be told it's a self-restartable 'service'.

        ``headroom install start/stop/restart`` all reject SupervisorKind.TASK
        deployments (see cli/install.py:_reject_task_lifecycle); a detached
        ``headroom install restart`` against a task manifest would previously
        fail silently (stdout/stderr to DEVNULL) after the API had already
        returned ``{"restarted": true}``.
        """
        from headroom.install import runtime as rt

        monkeypatch.setenv("HEADROOM_DEPLOYMENT_PROFILE", "default")
        monkeypatch.setenv("HEADROOM_DEPLOYMENT_PRESET", "persistent-task")
        stub = types.SimpleNamespace(profile="default", supervisor_kind="task")
        monkeypatch.setattr(rt, "load_manifest", lambda profile: stub)
        popen_calls: list = []
        monkeypatch.setattr(rt.subprocess, "Popen", lambda *a, **k: popen_calls.append(a))

        _manifest, mode = rt.detect_current_deployment()
        assert mode == "task"

        result = rt.restart_current_deployment()
        assert result["restarted"] is False
        assert result["mode"] == "task"
        assert "instruction" in result
        assert popen_calls == []  # never spawns `headroom install restart` for task deployments

    def test_service_spawns_detached_restart(self, monkeypatch) -> None:
        from headroom.install import runtime as rt

        monkeypatch.setenv("HEADROOM_DEPLOYMENT_PROFILE", "default")
        monkeypatch.setenv("HEADROOM_DEPLOYMENT_PRESET", "persistent-service")
        stub = types.SimpleNamespace(profile="default", supervisor_kind="service")
        monkeypatch.setattr(rt, "load_manifest", lambda profile: stub)
        recorded: dict = {}

        def fake_popen(command, **kwargs):
            recorded["command"] = command
            recorded["kwargs"] = kwargs
            return object()

        monkeypatch.setattr(rt.subprocess, "Popen", fake_popen)

        _manifest, mode = rt.detect_current_deployment()
        assert mode == "service"

        result = rt.restart_current_deployment()
        assert result["restarted"] is True
        assert result["mode"] == "service"
        assert recorded["command"][-4:] == ["install", "restart", "--profile", "default"]
