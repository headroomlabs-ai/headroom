from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

FAKE_DOCKER = r"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path


STATE_PATH = Path(os.environ["FAKE_DOCKER_STATE"])
LOG_PATH = Path(os.environ["FAKE_DOCKER_LOG"])


def load_state() -> dict[str, dict[str, dict[str, int]]]:
    if not STATE_PATH.exists():
        return {"containers": {}}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, dict[str, dict[str, int]]]) -> None:
    STATE_PATH.write_text(json.dumps(state), encoding="utf-8")


def cleanup_dead(state: dict[str, dict[str, dict[str, int]]]) -> dict[str, dict[str, dict[str, int]]]:
    save_state(state)
    return state


def host_port_from_publish(value: str) -> int:
    parts = value.split(":")
    if len(parts) == 2:
        return int(parts[0])
    if len(parts) >= 3:
        return int(parts[-2])
    raise ValueError(f"Unsupported publish value: {value}")


def start_server(port: int) -> int:
    code = '''
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[1])

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt, *args):
        return

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process.pid


def stop_container(state: dict[str, dict[str, dict[str, int]]], name: str) -> None:
    data = state["containers"].pop(name, None)
    if not data:
        return
    try:
        os.kill(int(data["pid"]), signal.SIGTERM)
    except OSError:
        pass
    save_state(state)


def main() -> int:
    args = sys.argv[1:]
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(args) + "\n")

    if not args:
        return 0

    state = cleanup_dead(load_state())
    command = args[0]

    if command == "pull":
        return 0

    if command == "run":
        detached = "-d" in args
        if not detached:
            return 0

        name = None
        publish = None
        for index, arg in enumerate(args):
            if arg == "--name":
                name = args[index + 1]
            elif arg == "-p":
                publish = args[index + 1]

        if name is None or publish is None:
            raise SystemExit("missing --name or -p in fake docker run")

        port = host_port_from_publish(publish)
        state["containers"][name] = {"pid": start_server(port), "port": port}
        save_state(state)
        print(name)
        return 0

    if command == "ps":
        names = sorted(state["containers"])
        if "--format" in args:
            print("\n".join(names))
        return 0

    if command == "stop":
        for name in args[1:]:
            if not name.startswith("-"):
                stop_container(state, name)
        return 0

    if command == "rm":
        for name in args[1:]:
            if not name.startswith("-"):
                stop_container(state, name)
        return 0

    if command == "logs":
        if len(args) > 1:
            print(f"fake logs for {args[1]}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_fake_docker_shims(tmp_path: Path) -> Path:
    shim_dir = tmp_path / "fake-docker"
    shim_dir.mkdir()

    fake_docker = shim_dir / "fake_docker.py"
    fake_docker.write_text(FAKE_DOCKER, encoding="utf-8")

    docker_sh = shim_dir / "docker"
    docker_sh.write_text(
        f'#!/usr/bin/env bash\nexec "{sys.executable}" "{fake_docker}" "$@"\n',
        encoding="utf-8",
    )
    docker_sh.chmod(0o755)

    docker_cmd = shim_dir / "docker.cmd"
    docker_cmd.write_text(
        f'@echo off\r\n"{sys.executable}" "{fake_docker}" %*\r\n',
        encoding="utf-8",
    )

    openclaw_sh = shim_dir / "openclaw"
    openclaw_sh.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    openclaw_sh.chmod(0o755)

    openclaw_cmd = shim_dir / "openclaw.cmd"
    openclaw_cmd.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")

    return shim_dir


def _build_env(home: Path, tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    shim_dir = _write_fake_docker_shims(tmp_path)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["PATH"] = str(shim_dir) + os.pathsep + env.get("PATH", "")
    env["FAKE_DOCKER_STATE"] = str(tmp_path / "fake-docker-state.json")
    env["FAKE_DOCKER_LOG"] = str(tmp_path / "fake-docker.log")
    return env


def _cleanup_fake_docker(env: dict[str, str]) -> None:
    state_path = Path(env["FAKE_DOCKER_STATE"])
    if not state_path.exists():
        return

    state = json.loads(state_path.read_text(encoding="utf-8"))
    for container in state.get("containers", {}).values():
        try:
            os.kill(int(container["pid"]), signal.SIGTERM)
        except OSError:
            pass


def _read_fake_docker_log(env: dict[str, str]) -> list[list[str]]:
    log_path = Path(env["FAKE_DOCKER_LOG"])
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]


def _run(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=check,
    )


def _bash_supports_4_3() -> bool:
    """The Docker-native installer requires bash >= 4.3. macOS ships 3.2."""
    bash = shutil.which("bash")
    if not bash:
        return False
    try:
        out = subprocess.run(
            [bash, "-c", 'echo "${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}"'],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    parts = out.stdout.strip().split(".")
    if len(parts) < 2:
        return False
    try:
        major, minor = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return (major, minor) >= (4, 3)


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None or not _bash_supports_4_3(),
    reason="installer requires bash >= 4.3 (macOS system bash is 3.2)",
)
def test_bash_native_installer_supports_persistent_docker_lifecycle(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".local").mkdir(parents=True)
    env = _build_env(home, tmp_path)
    env["HEADROOM_DOCKER_IMAGE"] = "headroom:test-image"

    try:
        _run(["bash", str(REPO_ROOT / "scripts" / "install.sh")], env=env, cwd=REPO_ROOT)

        wrapper = home / ".local" / "bin" / "headroom"
        assert wrapper.exists()
        assert "HEADROOM_IMAGE_DEFAULT=headroom:test-image" in wrapper.read_text(encoding="utf-8")

        help_result = _run([str(wrapper), "install", "-?"], env=env)
        assert "persistent-docker preset only" in help_result.stdout
        _run([str(wrapper), "--help"], env=env)
        wrap_help = _run([str(wrapper), "wrap", "--help"], env=env)
        assert "Supported commands:" in wrap_help.stdout
        assert "copilot" not in wrap_help.stdout
        unsupported_wrap = _run(
            [str(wrapper), "wrap", "copilot", "--help"],
            env=env,
            check=False,
        )
        assert unsupported_wrap.returncode != 0
        assert "does not support 'wrap copilot'" in unsupported_wrap.stderr

        invalid_profile = _run(
            [str(wrapper), "install", "status", "--profile", ".."],
            env=env,
            check=False,
        )
        assert invalid_profile.returncode != 0
        assert "Invalid profile name '..'" in invalid_profile.stderr
        missing_profile_value = _run(
            [str(wrapper), "install", "apply", "--profile"],
            env=env,
            check=False,
        )
        assert missing_profile_value.returncode != 0
        assert "Option --profile requires a value" in missing_profile_value.stderr
        missing_proxy_port = _run(
            [str(wrapper), "proxy", "--port"],
            env=env,
            check=False,
        )
        assert missing_proxy_port.returncode != 0
        assert "Option --port requires a value" in missing_proxy_port.stderr
        invalid_proxy_port = _run(
            [str(wrapper), "proxy", "--port", "abc"],
            env=env,
            check=False,
        )
        assert invalid_proxy_port.returncode != 0
        assert "Invalid port 'abc'" in invalid_proxy_port.stderr
        missing_wrap_port = _run(
            [str(wrapper), "wrap", "claude", "--port"],
            env=env,
            check=False,
        )
        assert missing_wrap_port.returncode != 0
        assert "Option --port requires a value" in missing_wrap_port.stderr
        invalid_wrap_port = _run(
            [str(wrapper), "wrap", "claude", "--port", "abc"],
            env=env,
            check=False,
        )
        assert invalid_wrap_port.returncode != 0
        assert "Invalid port 'abc'" in invalid_wrap_port.stderr
        missing_openclaw_proxy_port = _run(
            [str(wrapper), "wrap", "openclaw", "--proxy-port"],
            env=env,
            check=False,
        )
        assert missing_openclaw_proxy_port.returncode != 0
        assert "Option --proxy-port requires a value" in missing_openclaw_proxy_port.stderr
        invalid_openclaw_proxy_port = _run(
            [str(wrapper), "wrap", "openclaw", "--proxy-port", "abc"],
            env=env,
            check=False,
        )
        assert invalid_openclaw_proxy_port.returncode != 0
        assert "Invalid port 'abc'" in invalid_openclaw_proxy_port.stderr
        for invalid_port in ("abc", "0", "65536"):
            invalid_port_result = _run(
                [str(wrapper), "install", "apply", "--port", invalid_port],
                env=env,
                check=False,
            )
            assert invalid_port_result.returncode != 0
            assert f"Invalid port '{invalid_port}'" in invalid_port_result.stderr

        port = _free_port()
        _run(
            [
                str(wrapper),
                "install",
                "apply",
                "--profile",
                "smoke",
                "--port",
                str(port),
                "--memory",
                "--no-telemetry",
                "--image",
                "fake/headroom:test",
            ],
            env=env,
        )

        manifest_path = home / ".headroom" / "deploy" / "smoke" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["preset"] == "persistent-docker"
        assert manifest["port"] == port
        assert manifest["memory_enabled"] is True
        assert manifest["memory_db_path"] == "/tmp/headroom-home/.headroom/memory.db"
        assert manifest["telemetry_enabled"] is False

        state_path = home / ".headroom" / "deploy" / "smoke" / "docker-native.env"
        state_text = state_path.read_text(encoding="utf-8")
        assert f"PORT={port!r}" in state_text

        docker_calls = _read_fake_docker_log(env)
        help_call = next(
            call
            for call in docker_calls
            if call[:2] == ["run", "--rm"] and "--entrypoint" in call and "--help" in call
        )
        assert "-it" not in help_call
        install_call = next(
            call for call in docker_calls if call[:2] == ["run", "-d"] and "--name" in call
        )
        assert "/tmp/headroom-home/.headroom/memory.db" in install_call
        # Canonical filesystem contract env vars (issue #175) forwarded into
        # the container so the proxy resolves state/config to the bind mount.
        assert "HEADROOM_WORKSPACE_DIR=/tmp/headroom-home/.headroom" in install_call
        assert "HEADROOM_CONFIG_DIR=/tmp/headroom-home/.headroom/config" in install_call

        status_result = _run(
            [str(wrapper), "install", "status", "--profile", "smoke"],
            env=env,
        )
        assert "Status:     running" in status_result.stdout

        _run([str(wrapper), "install", "stop", "--profile", "smoke"], env=env)
        stopped_result = _run(
            [str(wrapper), "install", "status", "--profile", "smoke"],
            env=env,
        )
        assert "Status:     stopped" in stopped_result.stdout

        _run([str(wrapper), "install", "start", "--profile", "smoke"], env=env)
        restarted_result = _run(
            [str(wrapper), "install", "status", "--profile", "smoke"],
            env=env,
        )
        assert "Status:     running" in restarted_result.stdout

        rejected = _run(
            [str(wrapper), "install", "apply", "--scope", "user"],
            env=env,
            check=False,
        )
        assert rejected.returncode != 0
        assert "does not support provider/user/system mutation flags" in rejected.stderr

        _run([str(wrapper), "install", "restart", "--profile", "smoke"], env=env)
        _run([str(wrapper), "install", "remove", "--profile", "smoke"], env=env)
        assert not manifest_path.parent.exists()
    finally:
        _cleanup_fake_docker(env)


def _powershell_executable() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell") or shutil.which("powershell.exe")


@pytest.mark.skipif(
    os.name != "nt" or _powershell_executable() is None,
    reason="Windows PowerShell coverage runs on Windows hosts only",
)
def test_powershell_native_installer_supports_persistent_docker_lifecycle(tmp_path: Path) -> None:
    powershell = _powershell_executable()
    assert powershell is not None

    home = tmp_path / "home"
    (home / ".local").mkdir(parents=True)
    env = _build_env(home, tmp_path)
    env["HEADROOM_DOCKER_IMAGE"] = "headroom:test-image"

    try:
        _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(REPO_ROOT / "scripts" / "install.ps1"),
            ],
            env=env,
            cwd=REPO_ROOT,
        )

        wrapper = home / ".local" / "bin" / "headroom.ps1"
        assert wrapper.exists()
        assert "__HEADROOM_INSTALL_IMAGE__" not in wrapper.read_text(encoding="utf-8")
        assert "headroom:test-image" in wrapper.read_text(encoding="utf-8")
        cmd_wrapper = home / ".local" / "bin" / "headroom.cmd"
        assert cmd_wrapper.exists()

        help_result = _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "install",
                "-?",
            ],
            env=env,
        )
        _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "proxy",
                "--help",
            ],
            env=env,
        )
        assert "persistent-docker preset only" in help_result.stdout
        cmd_help_result = _run(
            ["cmd.exe", "/c", str(cmd_wrapper), "install", "-?"],
            env=env,
        )
        assert "persistent-docker preset only" in cmd_help_result.stdout
        _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "--help",
            ],
            env=env,
        )
        wrap_help = _run(
            ["cmd.exe", "/c", str(cmd_wrapper), "wrap", "--help"],
            env=env,
        )
        assert "Supported commands:" in wrap_help.stdout
        assert "copilot" not in wrap_help.stdout
        unsupported_wrap = _run(
            ["cmd.exe", "/c", str(cmd_wrapper), "wrap", "copilot", "--help"],
            env=env,
            check=False,
        )
        assert unsupported_wrap.returncode != 0
        assert "does not support 'wrap copilot'" in unsupported_wrap.stderr
        invalid_profile = _run(
            ["cmd.exe", "/c", str(cmd_wrapper), "install", "status", "--profile", ".."],
            env=env,
            check=False,
        )
        assert invalid_profile.returncode != 0
        assert "Invalid profile name '..'" in invalid_profile.stderr
        missing_profile_value = _run(
            ["cmd.exe", "/c", str(cmd_wrapper), "install", "apply", "--profile"],
            env=env,
            check=False,
        )
        assert missing_profile_value.returncode != 0
        assert "Option --profile requires a value" in missing_profile_value.stderr
        missing_proxy_port = _run(
            ["cmd.exe", "/c", str(cmd_wrapper), "proxy", "--port"],
            env=env,
            check=False,
        )
        assert missing_proxy_port.returncode != 0
        assert "Option --port requires a value" in missing_proxy_port.stderr
        invalid_proxy_port = _run(
            ["cmd.exe", "/c", str(cmd_wrapper), "proxy", "--port", "abc"],
            env=env,
            check=False,
        )
        assert invalid_proxy_port.returncode != 0
        assert "Invalid port 'abc'" in invalid_proxy_port.stderr
        missing_wrap_port = _run(
            ["cmd.exe", "/c", str(cmd_wrapper), "wrap", "claude", "--port"],
            env=env,
            check=False,
        )
        assert missing_wrap_port.returncode != 0
        assert "Option --port requires a value" in missing_wrap_port.stderr
        invalid_wrap_port = _run(
            ["cmd.exe", "/c", str(cmd_wrapper), "wrap", "claude", "--port", "abc"],
            env=env,
            check=False,
        )
        assert invalid_wrap_port.returncode != 0
        assert "Invalid port 'abc'" in invalid_wrap_port.stderr
        missing_openclaw_proxy_port = _run(
            ["cmd.exe", "/c", str(cmd_wrapper), "wrap", "openclaw", "--proxy-port"],
            env=env,
            check=False,
        )
        assert missing_openclaw_proxy_port.returncode != 0
        assert "Option --proxy-port requires a value" in missing_openclaw_proxy_port.stderr
        invalid_openclaw_proxy_port = _run(
            ["cmd.exe", "/c", str(cmd_wrapper), "wrap", "openclaw", "--proxy-port", "abc"],
            env=env,
            check=False,
        )
        assert invalid_openclaw_proxy_port.returncode != 0
        assert "Invalid port 'abc'" in invalid_openclaw_proxy_port.stderr
        for invalid_port in ("abc", "0", "65536"):
            invalid_port_result = _run(
                ["cmd.exe", "/c", str(cmd_wrapper), "install", "apply", "--port", invalid_port],
                env=env,
                check=False,
            )
            assert invalid_port_result.returncode != 0
            assert f"Invalid port '{invalid_port}'" in invalid_port_result.stderr

        port = _free_port()
        _run(
            [
                "cmd.exe",
                "/c",
                str(cmd_wrapper),
                "install",
                "apply",
                "--profile",
                "smoke",
                "--port",
                str(port),
                "--memory",
                "--no-telemetry",
                "--image",
                "fake/headroom:test",
            ],
            env=env,
        )

        manifest_path = home / ".headroom" / "deploy" / "smoke" / "manifest.json"
        state_path = home / ".headroom" / "deploy" / "smoke" / "docker-native.json"
        manifest_bytes = manifest_path.read_bytes()
        state_bytes = state_path.read_bytes()
        assert not manifest_bytes.startswith(b"\xef\xbb\xbf")
        assert not state_bytes.startswith(b"\xef\xbb\xbf")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        state = json.loads(state_bytes.decode("utf-8"))
        assert manifest["preset"] == "persistent-docker"
        assert manifest["port"] == port
        assert manifest["memory_enabled"] is True
        assert manifest["memory_db_path"] == "/tmp/headroom-home/.headroom/memory.db"
        assert manifest["telemetry_enabled"] is False
        assert state["container_name"] == "headroom-smoke"

        docker_calls = _read_fake_docker_log(env)
        help_call = next(
            call
            for call in docker_calls
            if call[:2] == ["run", "--rm"] and "--entrypoint" in call and "--help" in call
        )
        assert "-it" not in help_call
        proxy_help_call = next(
            call
            for call in docker_calls
            if call[:2] == ["run", "--rm"] and "-p" in call and "proxy" in call and "--help" in call
        )
        assert "-it" not in proxy_help_call
        install_call = next(
            call for call in docker_calls if call[:2] == ["run", "-d"] and "--name" in call
        )
        assert "/tmp/headroom-home/.headroom/memory.db" in install_call
        # Canonical filesystem contract env vars (issue #175).
        assert "HEADROOM_WORKSPACE_DIR=/tmp/headroom-home/.headroom" in install_call
        assert "HEADROOM_CONFIG_DIR=/tmp/headroom-home/.headroom/config" in install_call

        status_result = _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "install",
                "status",
                "--profile",
                "smoke",
            ],
            env=env,
        )
        assert "Status:     running" in status_result.stdout

        _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "install",
                "stop",
                "--profile",
                "smoke",
            ],
            env=env,
        )
        stopped_result = _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "install",
                "status",
                "--profile",
                "smoke",
            ],
            env=env,
        )
        assert "Status:     stopped" in stopped_result.stdout

        _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "install",
                "start",
                "--profile",
                "smoke",
            ],
            env=env,
        )
        started_result = _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "install",
                "status",
                "--profile",
                "smoke",
            ],
            env=env,
        )
        assert "Status:     running" in started_result.stdout

        rejected = _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "install",
                "apply",
                "--scope",
                "user",
            ],
            env=env,
            check=False,
        )
        assert rejected.returncode != 0
        assert "does not support provider/user/system mutation flags" in rejected.stderr

        _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "install",
                "restart",
                "--profile",
                "smoke",
            ],
            env=env,
        )
        _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "install",
                "remove",
                "--profile",
                "smoke",
            ],
            env=env,
        )
        assert not manifest_path.parent.exists()
    finally:
        _cleanup_fake_docker(env)
