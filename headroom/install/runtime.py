"""Runtime helpers for persistent deployments."""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from headroom._subprocess import pid_alive, run

from .health import probe_ready
from .models import DeploymentManifest, InstallPreset, RuntimeKind, SupervisorKind
from .paths import log_path, pid_path, profile_root
from .state import load_manifest

_STOP_POLL_ATTEMPTS = 30
_STOP_POLL_DELAY = 0.1

# Inside the container the proxy must listen on every interface so the
# host-side published port (127.0.0.1:<port>) can reach it.
CONTAINER_BIND_HOST = "0.0.0.0"  # noqa: S104 — container-internal bind, published only on 127.0.0.1
# proxy_args always starts with the host flag/value pair (see planner.py); we
# drop it and substitute CONTAINER_BIND_HOST for the in-container bind.
_PROXY_ARGS_HOST_PAIR_LEN = 2

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
    "QDRANT_",
    "NEO4J_",
    "LANGSMITH_",
)


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _container_runtime_is_podman() -> bool:
    """Best-effort: is the ``docker`` command actually Podman?

    Rootless Podman maps the host user to container UID 0, so the
    ``--user <host-uid>:<host-gid>`` flag that is correct for Docker instead
    selects a subordinate UID that owns none of the bind-mounted host
    directories, and every write into ``~/.headroom`` fails (#2804). Detect the
    common ``docker -> podman`` shim (e.g. NixOS
    ``/run/current-system/sw/bin/docker -> podman``) by resolving the binary and
    checking its real name. ``HEADROOM_CONTAINER_RUNTIME`` (``podman`` / ``docker``)
    is an explicit override for setups the symlink heuristic cannot see, such as a
    wrapper script. No subprocess is spawned.
    """
    override = os.environ.get("HEADROOM_CONTAINER_RUNTIME", "").strip().lower()
    if override:
        return override == "podman"
    resolved = shutil.which("docker")
    if not resolved:
        return False
    try:
        real = os.path.realpath(resolved)
    except OSError:
        real = resolved
    return "podman" in os.path.basename(real).lower()


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
    for subdir in (".headroom", ".claude", ".codex", ".gemini", ".config/opencode"):
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
        "--volume",
        f"{_mount_source(home, '.config/opencode')}:{container_home}/.config/opencode",
    ]
    docker_gpus = manifest.base_env.get("HEADROOM_DOCKER_GPUS", "").strip()
    if docker_gpus:
        command.extend(["--gpus", docker_gpus])
    if not _is_windows():
        if _container_runtime_is_podman():
            # Rootless Podman maps the host user to container UID 0, so --user
            # would map to a subordinate UID that owns none of the bind mounts and
            # every write into ~/.headroom fails (#2804). keep-id maps the host
            # user to the same UID inside the container, keeping the mounts
            # writable. Docker maps UIDs 1:1, so --user stays correct there.
            command.append("--userns=keep-id")
        else:
            getuid = getattr(os, "getuid", None)
            getgid = getattr(os, "getgid", None)
            if callable(getuid) and callable(getgid):
                command.extend(["--user", f"{getuid()}:{getgid()}"])
    runtime_env = {**manifest.base_env, **_deployment_env(manifest)}
    for name, value in runtime_env.items():
        command.extend(["--env", f"{name}={value}"])
    for name in sorted(os.environ):
        # Skip any name the manifest already pinned above: Docker resolves
        # duplicate `--env` last-wins, so a bare `--env HEADROOM_BACKEND`
        # passthrough (which reads the host process env at
        # `start_persistent_docker` time) would silently override the manifest's
        # `--env HEADROOM_BACKEND=<value>`, diverging the container from its
        # deployment config.
        if name.startswith(PASSTHROUGH_ENV_PREFIXES) and name not in runtime_env:
            command.extend(["--env", name])
    # The image ENTRYPOINT already runs `headroom proxy` (see Dockerfile), so
    # the args appended after the image name are only the proxy flags — never
    # `headroom proxy` again, or Docker would run `headroom proxy headroom
    # proxy ...` and Click aborts on the extra arguments (issue #833).
    command.extend(
        [
            manifest.image,
            "--host",
            CONTAINER_BIND_HOST,
            *manifest.proxy_args[_PROXY_ARGS_HOST_PAIR_LEN:],
        ]
    )
    return command


def runtime_ownership(manifest: DeploymentManifest) -> str:
    """Classify the owner that must launch and supervise this runtime."""

    runtime_kind = getattr(manifest, "runtime_kind", None)
    if (
        runtime_kind == RuntimeKind.DOCKER.value
        or manifest.preset == InstallPreset.PERSISTENT_DOCKER.value
    ):
        return "docker-supervisor"
    if (
        sys.platform == "darwin"
        and manifest.preset == InstallPreset.PERSISTENT_SERVICE.value
        and manifest.runtime_kind == RuntimeKind.PYTHON.value
        and manifest.supervisor_kind == SupervisorKind.SERVICE.value
    ):
        return "launchd-exec"
    return "popen"


def _write_pid(profile: str, pid: int) -> None:
    path = pid_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid))


def _read_pid(profile: str) -> int | None:
    path = pid_path(profile)
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_proc_metadata(pid: int) -> tuple[list[str], dict[str, str]] | None:
    """Read process identity without requiring the optional psutil package."""

    if Path(f"/proc/{pid}/cmdline").exists():
        try:
            cmdline = [
                part.decode(errors="replace")
                for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                if part
            ]
            environ = {
                part.split(b"=", 1)[0].decode(errors="replace"): part.split(b"=", 1)[1].decode(
                    errors="replace"
                )
                for part in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
                if b"=" in part
            }
            return cmdline, environ
        except (OSError, ValueError):
            return None

    if _is_windows():
        # WMI is part of Windows and exposes the full command line without
        # requiring psutil. Environment blocks are not exposed reliably, so
        # the command-line identity is checked together with the PID file.
        try:
            result = run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        raw = result.stdout.strip()
        if result.returncode != 0 or not raw:
            return None
        try:
            return shlex.split(raw, posix=False), {}
        except ValueError:
            return None

    # macOS does not expose /proc by default. `ps -wwE` is the supported
    # system interface that returns the complete command and inherited env.
    if sys.platform != "darwin":
        return None
    try:
        result = run(
            ["ps", "-wwE", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = result.stdout.strip()
    if result.returncode != 0 or not raw:
        return None
    try:
        fields = shlex.split(raw)
    except ValueError:
        return None
    environ = {}
    for field in fields:
        if field.startswith("HEADROOM_") and "=" in field:
            name, value = field.split("=", 1)
            environ[name] = value
    return fields, environ


def _proxy_command_matches(cmdline: list[str], manifest: DeploymentManifest) -> bool:
    """Require the real Headroom proxy argv shape and configured port."""
    shape = any(
        cmdline[index : index + 3] == ["-m", "headroom.cli", "proxy"]
        or cmdline[index : index + 2] == ["headroom", "proxy"]
        for index in range(len(cmdline))
    )
    if not shape:
        return False
    for index, value in enumerate(cmdline):
        if value == "--port" and index + 1 < len(cmdline):
            return cmdline[index + 1] == str(manifest.port)
        if value.startswith("--port="):
            return value.split("=", 1)[1] == str(manifest.port)
    return False


def _proxy_deployment_markers_match(cmdline: list[str], manifest: DeploymentManifest) -> bool:
    """Match the deployment identity embedded in supervisor-launched argv."""
    markers = {
        "--headroom-deployment-profile": manifest.profile,
        "--headroom-deployment-runtime": manifest.runtime_kind,
    }
    for name, expected in markers.items():
        if not any(
            cmdline[index : index + 2] == [name, str(expected)] or value == f"{name}={expected}"
            for index, value in enumerate(cmdline)
        ):
            return False
    return True


def _process_matches_runtime(pid: int, manifest: DeploymentManifest) -> bool:
    """Verify the PID's deployment identity before lifecycle operations."""
    return _process_identity(pid, manifest) is True


def _process_identity(pid: int, manifest: DeploymentManifest) -> bool | None:
    """Return whether a PID matches, or ``None`` when identity is unavailable."""
    if pid == os.getpid():
        return False
    try:
        import psutil  # type: ignore[import-untyped]  # optional dependency

        process = psutil.Process(pid)
        cmdline = process.cmdline()
        environ = process.environ()
    except Exception:
        metadata = _read_proc_metadata(pid)
        if metadata is None:
            return None
        cmdline, environ = metadata
    if not _proxy_command_matches(cmdline, manifest):
        return False
    if _is_windows() and not environ:
        if not _proxy_deployment_markers_match(cmdline, manifest):
            return None
        return True
    return bool(
        environ.get("HEADROOM_DEPLOYMENT_PROFILE") == manifest.profile
        and environ.get("HEADROOM_DEPLOYMENT_RUNTIME") == manifest.runtime_kind
    )


def _clear_pid(profile: str, *, expected_pid: int | None = None) -> None:
    path = pid_path(profile)
    if expected_pid is not None and _read_pid(profile) != expected_pid:
        return
    if path.exists():
        path.unlink()


@contextmanager
def acquire_runtime_start_lock(profile: str) -> Iterator[bool]:
    """Try to hold the profile-local runtime start lock."""

    path = profile_root(profile) / "runner.start.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8", errors="replace") as lock_file:
        acquired = False
        if _is_windows():
            import msvcrt

            lock_file.seek(0)
            msvcrt_any = cast(Any, msvcrt)
            try:
                msvcrt_any.locking(lock_file.fileno(), msvcrt_any.LK_NBLCK, 1)
                acquired = True
            except OSError:
                yield False
                return
        else:
            import fcntl

            try:
                fcntl_any = cast(Any, fcntl)
                fcntl_any.flock(lock_file.fileno(), fcntl_any.LOCK_EX | fcntl_any.LOCK_NB)
                acquired = True
            except BlockingIOError:
                yield False
                return
        try:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(str(os.getpid()))
            lock_file.flush()
            yield True
        finally:
            if acquired:
                if _is_windows():
                    import msvcrt

                    lock_file.seek(0)
                    msvcrt_any = cast(Any, msvcrt)
                    try:
                        msvcrt_any.locking(lock_file.fileno(), msvcrt_any.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl

                    fcntl_any = cast(Any, fcntl)
                    fcntl_any.flock(lock_file.fileno(), fcntl_any.LOCK_UN)


def run_foreground(manifest: DeploymentManifest) -> int:
    """Run the raw runtime command in the foreground."""

    if runtime_ownership(manifest) == "docker-supervisor":
        raise RuntimeError(
            "Docker deployments must be started by the Docker supervisor, not run_foreground"
        )
    command = build_runtime_command(manifest)
    env = _runtime_env(manifest)
    log_file_path = log_path(manifest.profile)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file_path, "a", encoding="utf-8", errors="replace") as log_file:
        if runtime_ownership(manifest) == "launchd-exec":
            _write_pid(manifest.profile, os.getpid())
            try:
                os.dup2(log_file.fileno(), 1)
                os.dup2(log_file.fileno(), 2)
                os.execvpe(command[0], command, env)
            except BaseException:
                _clear_pid(manifest.profile, expected_pid=os.getpid())
                raise

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
            _clear_pid(manifest.profile, expected_pid=proc.pid)


def start_detached_agent(profile: str) -> subprocess.Popen[str]:
    """Start `headroom install agent run` detached for the given profile."""

    command = [*resolve_headroom_command(), "install", "agent", "run", "--profile", profile]
    log_file_path = log_path(profile)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_file_path, "a", encoding="utf-8", errors="replace")  # noqa: SIM115

    kwargs: dict[str, Any] = {"stdout": log_file, "stderr": log_file}
    if _is_windows():
        # DETACHED_PROCESS makes CREATE_NO_WINDOW a no-op (per Win32 docs), so a
        # detached console child pops up a visible window. Use CREATE_NO_WINDOW
        # instead; it still detaches from the parent's console.
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(command, **kwargs)
    finally:
        # The child has inherited the log file descriptor, so the parent's
        # copy is dead weight. Closing it (even when Popen raises) avoids
        # leaking one fd per `headroom install start` and lets the log file
        # be rotated. Wrapped in try/finally so a Popen failure can't leak.
        log_file.close()
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
    remove = run(
        ["docker", "rm", "-f", manifest.container_name],
        capture_output=True,
        text=True,
    )
    _require_docker_success(remove, "docker rm")
    subprocess.run(docker_cmd, check=True)


def stop_runtime(manifest: DeploymentManifest) -> None:
    """Stop the raw runtime for the deployment."""

    if runtime_ownership(manifest) == "docker-supervisor":
        stop = run(
            ["docker", "stop", manifest.container_name],
            capture_output=True,
            text=True,
        )
        _require_docker_success(stop, "docker stop")
        remove = run(
            ["docker", "rm", "-f", manifest.container_name],
            capture_output=True,
            text=True,
        )
        _require_docker_success(remove, "docker rm")
        return

    pid = _read_pid(manifest.profile)
    if pid is None:
        if pid_path(manifest.profile).exists():
            raise RuntimeError(
                f"Cannot stop deployment '{manifest.profile}': runner.pid is invalid"
            )
        return
    identity = _process_identity(pid, manifest)
    if identity is None:
        raise RuntimeError(f"Cannot stop deployment '{manifest.profile}': runtime identity unknown")
    if not identity:
        _clear_pid(manifest.profile, expected_pid=pid)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, SystemError):
        # SystemError covers the Windows WinError 87 surfacing described in #1544.
        pass
    for _ in range(_STOP_POLL_ATTEMPTS):
        if not pid_alive(pid):
            if not pid_alive(pid):
                _clear_pid(manifest.profile, expected_pid=pid)
                return
            raise RuntimeError(
                f"Cannot stop deployment '{manifest.profile}': runtime identity changed"
            )
        if _process_identity(pid, manifest) is not True:
            raise RuntimeError(
                f"Cannot stop deployment '{manifest.profile}': runtime identity changed"
            )
        time.sleep(_STOP_POLL_DELAY)
    raise RuntimeError(
        f"Cannot stop deployment '{manifest.profile}': runtime remained alive after SIGTERM"
    )


def _command_output(result: Any) -> str:
    return str(
        getattr(result, "stderr", "") or getattr(result, "stdout", "") or "unknown error"
    ).strip()


def _require_docker_success(result: Any, operation: str) -> None:
    if result.returncode == 0:
        return
    if result.returncode == 1 and "no such container" in _command_output(result).lower():
        return
    raise RuntimeError(f"{operation} failed: {_command_output(result)}")


def wait_ready(
    manifest: DeploymentManifest, timeout_seconds: int = 30, *, require_identity: bool = False
) -> bool:
    """Wait for the deployment to report ready."""

    for _ in range(timeout_seconds):
        if probe_ready(manifest.health_url) and (
            not require_identity or runtime_status(manifest) == "running"
        ):
            return True
        time.sleep(1)
    return False


def runtime_ready(manifest: DeploymentManifest) -> bool:
    """Return ready only when health and deployment identity both match."""
    from .health import probe_ready as current_probe_ready

    if not hasattr(manifest, "runtime_kind"):
        return bool(getattr(manifest, "health_url", None)) and current_probe_ready(
            manifest.health_url
        )
    try:
        return runtime_status(manifest) == "running" and current_probe_ready(manifest.health_url)
    except AttributeError:
        # Test doubles and old third-party callers may provide only the health
        # contract; persisted DeploymentManifest instances always have identity.
        return False


def runtime_status(manifest: DeploymentManifest) -> str:
    """Return a short status string for the deployment runtime."""

    if runtime_ownership(manifest) == "docker-supervisor":
        result = run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        if manifest.container_name in result.stdout.splitlines():
            return "running"
        return "stopped"
    pid = _read_pid(manifest.profile)
    if pid is None:
        if pid_path(manifest.profile).exists():
            return "unknown"
        return "stopped"
    # Windows-safe liveness probe: a bare os.kill(pid, 0) here raised WinError 87
    # as a SystemError against the detached agent, crashing status and taking the
    # live proxy down with it (#1544).
    if not pid_alive(pid):
        _clear_pid(manifest.profile, expected_pid=pid)
        return "stopped"
    identity = _process_identity(pid, manifest)
    if identity is None:
        return "unknown"
    if not identity:
        _clear_pid(manifest.profile, expected_pid=pid)
        return "stopped"
    return "running"


def detect_current_deployment() -> tuple[DeploymentManifest | None, str]:
    """Detect how THIS running proxy was launched.

    Returns ``(manifest_or_none, mode)`` where ``mode`` is one of:

    * ``"docker"``     — persistent-docker deployment. Cannot self-restart:
      there is no docker socket/CLI inside the container.
    * ``"service"``    — any other persistent (supervised) deployment; can
      self-restart via ``headroom install restart``.
    * ``"foreground"`` — a plain ``headroom proxy`` (or unknown); not
      self-restartable.

    Keys off the ``HEADROOM_DEPLOYMENT_*`` env vars the supervisor injects at
    launch (see :func:`_deployment_env`); a foreground proxy has none set.
    """
    profile = os.environ.get("HEADROOM_DEPLOYMENT_PROFILE")
    preset = os.environ.get("HEADROOM_DEPLOYMENT_PRESET")
    if not profile:
        return None, "foreground"
    manifest = load_manifest(profile)
    runtime = os.environ.get("HEADROOM_DEPLOYMENT_RUNTIME")
    if preset == InstallPreset.PERSISTENT_DOCKER.value or runtime == RuntimeKind.DOCKER.value:
        return manifest, "docker"
    if manifest is None:
        return None, "foreground"
    if manifest.supervisor_kind == SupervisorKind.TASK.value:
        return manifest, "task"
    return manifest, "service"


def _spawn_detached_restart(profile: str) -> None:
    """Spawn a detached ``headroom install restart --profile <p>`` process.

    Detached (``start_new_session`` on POSIX) so it outlives this process being
    torn down by the very restart it triggers.
    """
    command = [*resolve_headroom_command(), "install", "restart", "--profile", profile]
    popen_kwargs: dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if _is_windows():
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    else:
        popen_kwargs["start_new_session"] = True
    subprocess.Popen(command, **popen_kwargs)


def restart_current_deployment() -> dict[str, Any]:
    """Restart the current deployment so new settings take effect.

    * service -> spawn a detached restart, return ``{restarted: True, ...}``.
    * docker  -> not restartable in-container; return the host command to run.
    * task    -> not restartable via the CLI (``headroom install`` rejects
      lifecycle ops for task-scheduled deployments); return an instruction.
    * foreground/unknown -> return a manual-restart instruction.
    """
    manifest, mode = detect_current_deployment()
    profile = os.environ.get("HEADROOM_DEPLOYMENT_PROFILE") or (
        manifest.profile if manifest else "default"
    )
    if mode == "service":
        _spawn_detached_restart(profile)
        return {"restarted": True, "mode": "service", "profile": profile}
    if mode == "docker":
        return {
            "restarted": False,
            "mode": "docker",
            "command": f"headroom install restart --profile {profile}",
        }
    if mode == "task":
        return {
            "restarted": False,
            "mode": "task",
            "instruction": (
                "This deployment is managed by an OS task scheduler, not "
                "`headroom install`; stop the running process so it is "
                "relaunched (with the new settings) on its next scheduled "
                "trigger, or restart it via your OS task scheduler."
            ),
        }
    return {
        "restarted": False,
        "mode": "foreground",
        "instruction": "Restart the proxy to apply the new settings.",
    }
