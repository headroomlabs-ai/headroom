"""Runtime helpers for persistent deployments."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .health import probe_ready
from .models import DeploymentManifest, InstallPreset, RuntimeKind
from .paths import log_path, pid_path

PASSTHROUGH_ENV_PREFIXES = (
    "HEADROOM_",
    "ANTHROPIC_",
    "OPENAI_",
    "GEMINI_",
    "AWS_",
    "AZURE_",
    "VERTEX_",
    "GOOGLE_",
    "GOOGLE_CLOUD_",
    "MISTRAL_",
    "GROQ_",
    "OPENROUTER_",
    "XAI_",
    "TOGETHER_",
    "COHERE_",
    "OLLAMA_",
    "LITELLM_",
    "OTEL_",
    "SUPABASE_",
    "QDRANT_",
    "NEO4J_",
    "LANGSMITH_",
)


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _deployment_env(manifest: DeploymentManifest) -> dict[str, str]:
    return {
        "HEADROOM_DEPLOYMENT_PROFILE": manifest.profile,
        "HEADROOM_DEPLOYMENT_PRESET": manifest.preset,
        "HEADROOM_DEPLOYMENT_RUNTIME": manifest.runtime_kind,
        "HEADROOM_DEPLOYMENT_SUPERVISOR": manifest.supervisor_kind,
        "HEADROOM_DEPLOYMENT_SCOPE": manifest.scope,
    }


def resolve_headroom_command() -> list[str]:
    """Resolve the most reliable command to invoke headroom."""

    headroom_bin = shutil.which("headroom")
    if headroom_bin:
        return [headroom_bin]
    return [sys.executable, "-m", "headroom.cli"]


def _runtime_env(manifest: DeploymentManifest) -> dict[str, str]:
    env = os.environ.copy()
    env.update(manifest.base_env)
    env.update(_deployment_env(manifest))
    return env


def _ensure_host_dirs() -> None:
    for subdir in (".headroom", ".claude", ".codex", ".gemini"):
        (Path.home() / subdir).mkdir(parents=True, exist_ok=True)


def _mount_source(home: str, subdir: str) -> str:
    if _is_windows():
        return f"{home}\\{subdir}"
    return f"{home}/{subdir}"


def build_runtime_command(manifest: DeploymentManifest) -> list[str]:
    """Build the raw foreground command that runs the proxy."""

    if manifest.runtime_kind == RuntimeKind.PYTHON.value:
        return [sys.executable, "-m", "headroom.cli", "proxy", *manifest.proxy_args]

    _ensure_host_dirs()
    home = str(Path.home())
    container_home = "/tmp/headroom-home"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        manifest.container_name,
        "-p",
        f"127.0.0.1:{manifest.port}:{manifest.port}",
        "--workdir",
        container_home,
        "--env",
        f"HOME={container_home}",
        "--env",
        "PYTHONUNBUFFERED=1",
        # Canonical Headroom filesystem contract (issue #175).
        "--env",
        f"HEADROOM_WORKSPACE_DIR={container_home}/.headroom",
        "--env",
        f"HEADROOM_CONFIG_DIR={container_home}/.headroom/config",
        "--volume",
        f"{_mount_source(home, '.headroom')}:{container_home}/.headroom",
        "--volume",
        f"{_mount_source(home, '.claude')}:{container_home}/.claude",
        "--volume",
        f"{_mount_source(home, '.codex')}:{container_home}/.codex",
        "--volume",
        f"{_mount_source(home, '.gemini')}:{container_home}/.gemini",
    ]
    if not _is_windows():
        getuid = getattr(os, "getuid", None)
        getgid = getattr(os, "getgid", None)
        if callable(getuid) and callable(getgid):
            command.extend(["--user", f"{getuid()}:{getgid()}"])
    runtime_env = {**manifest.base_env, **_deployment_env(manifest)}
    for name, value in runtime_env.items():
        command.extend(["--env", f"{name}={value}"])
    for name in sorted(os.environ):
        if name.startswith(PASSTHROUGH_ENV_PREFIXES):
            command.extend(["--env", name])
    command.extend(
        [
            manifest.image,
            "headroom",
            "proxy",
            "--host",
            "0.0.0.0",
            *manifest.proxy_args[2:],
        ]
    )
    return command


def _write_pid(profile: str, pid: int) -> None:
    path = pid_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid))


def _agent_pid_path(profile: str) -> Path:
    return pid_path(profile).with_name("agent.pid")


def _write_agent_pid(profile: str, pid: int) -> None:
    path = _agent_pid_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid))


def _read_pid_file(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def _read_pid(profile: str) -> int | None:
    return _read_pid_file(pid_path(profile))


def _read_agent_pid(profile: str) -> int | None:
    return _read_pid_file(_agent_pid_path(profile))


def _clear_pid(profile: str) -> None:
    path = pid_path(profile)
    if path.exists():
        path.unlink()


def _clear_agent_pid(profile: str) -> None:
    path = _agent_pid_path(profile)
    if path.exists():
        path.unlink()


def _terminate_pid(pid: int, *, process_group: bool = False) -> None:
    try:
        if process_group and not _is_windows():
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        if not process_group:
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return


def run_foreground(manifest: DeploymentManifest) -> int:
    """Run the raw runtime command in the foreground."""

    command = build_runtime_command(manifest)
    env = _runtime_env(manifest)
    log_file_path = log_path(manifest.profile)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file_path, "a", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(command, env=env, stdout=log_file, stderr=log_file)
        _write_pid(manifest.profile, proc.pid)

        def _cleanup(signum: int | None = None, frame: Any = None) -> None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

        signal.signal(signal.SIGINT, _cleanup)
        signal.signal(signal.SIGTERM, _cleanup)
        try:
            return proc.wait()
        finally:
            _clear_pid(manifest.profile)
            _clear_agent_pid(manifest.profile)


def start_detached_agent(profile: str) -> subprocess.Popen[str]:
    """Start `headroom install agent run` detached for the given profile."""

    command = [*resolve_headroom_command(), "install", "agent", "run", "--profile", profile]
    log_file_path = log_path(profile)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_file_path, "a", encoding="utf-8", errors="replace")  # noqa: SIM115

    kwargs: dict[str, Any] = {"stdout": log_file, "stderr": log_file}
    if _is_windows():
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(command, **kwargs)
    _write_agent_pid(profile, proc.pid)
    return proc


def start_persistent_docker(manifest: DeploymentManifest) -> None:
    """Start a persistent Docker container with restart policy."""

    command = build_runtime_command(manifest)
    docker_cmd = [
        "docker",
        "run",
        "-d",
        "--restart",
        "unless-stopped",
        "--name",
        manifest.container_name,
        *command[5:],  # drop initial `docker run --rm --name ...`
    ]
    subprocess.run(["docker", "rm", "-f", manifest.container_name], capture_output=True, text=True)
    subprocess.run(docker_cmd, check=True)


def stop_runtime(manifest: DeploymentManifest) -> None:
    """Stop the raw runtime for the deployment."""

    if manifest.preset == InstallPreset.PERSISTENT_DOCKER.value:
        subprocess.run(["docker", "stop", manifest.container_name], capture_output=True, text=True)
        subprocess.run(
            ["docker", "rm", "-f", manifest.container_name], capture_output=True, text=True
        )
        return

    agent_pid = _read_agent_pid(manifest.profile)
    if agent_pid is not None:
        _terminate_pid(agent_pid, process_group=True)

    pid = _read_pid(manifest.profile)
    if pid is not None and pid != agent_pid:
        _terminate_pid(pid)

    _clear_pid(manifest.profile)
    _clear_agent_pid(manifest.profile)


def wait_ready(manifest: DeploymentManifest, timeout_seconds: int = 30) -> bool:
    """Wait for the deployment to report ready."""

    for _ in range(timeout_seconds):
        if probe_ready(manifest.health_url):
            return True
        time.sleep(1)
    return False


def runtime_status(manifest: DeploymentManifest) -> str:
    """Return a short status string for the deployment runtime."""

    if manifest.preset == InstallPreset.PERSISTENT_DOCKER.value:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
        )
        if manifest.container_name in result.stdout.splitlines():
            return "running"
        return "stopped"
    for pid in (_read_agent_pid(manifest.profile), _read_pid(manifest.profile)):
        if pid is None:
            continue
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        return "running"
    return "stopped"
