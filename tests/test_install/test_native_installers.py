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

    if command == "network" and len(args) > 1 and args[1] == "inspect":
        # Match the default bridge gateway used by the native installer when
        # it configures the dashboard metadata allowlist.
        gateway = os.environ.get("FAKE_DOCKER_GATEWAY", "172.17.0.1")
        if gateway == "FAIL":
            return 1
        if "--format" in args and gateway:
            print(gateway)
        return 0

    if command == "run":
        detached = "-d" in args
        if not detached:
            return 0

        name = None
        publish = None
        container_env = {}
        for index, arg in enumerate(args):
            if arg == "--name":
                name = args[index + 1]
            elif arg == "-p":
                publish = args[index + 1]
            elif arg == "--env":
                spec = args[index + 1]
                if "=" in spec:
                    env_name, value = spec.split("=", 1)
                    container_env[env_name] = value
                elif spec in os.environ:
                    container_env[spec] = os.environ[spec]

        if name is None or publish is None:
            raise SystemExit("missing --name or -p in fake docker run")

        port = host_port_from_publish(publish)
        state["containers"][name] = {
            "pid": start_server(port),
            "port": port,
            "env": container_env,
        }
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
    # #2970: the PowerShell installer's Ensure-PathEntry persists to the 'User'
    # PATH scope (HKCU\Environment), which a HOME/USERPROFILE override does not
    # redirect. Keep the PATH update ephemeral (Process scope) so running these
    # tests never leaks the throwaway shim dir into the developer's real PATH.
    env["HEADROOM_INSTALL_PATH_SCOPE"] = "Process"
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


def _persistent_run_call(env: dict[str, str], profile: str) -> list[str]:
    container_name = f"headroom-{profile}"
    return next(
        call
        for call in _read_fake_docker_log(env)
        if call[:2] == ["run", "-d"]
        and "--name" in call
        and call[call.index("--name") + 1] == container_name
    )


def _persistent_container_env(env: dict[str, str], profile: str) -> dict[str, str]:
    state = json.loads(Path(env["FAKE_DOCKER_STATE"]).read_text(encoding="utf-8"))
    return state["containers"][f"headroom-{profile}"]["env"]


def _exercise_dashboard_gateway_overrides(wrapper_command: list[str], env: dict[str, str]) -> None:
    trusted_cidrs = "HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS"

    try:
        for profile, configured_value in (("configured", "10.20.0.0/16"), ("empty", "")):
            env[trusted_cidrs] = configured_value
            port = _free_port()
            _run(
                [
                    *wrapper_command,
                    "install",
                    "apply",
                    "--profile",
                    profile,
                    "--port",
                    str(port),
                    "--image",
                    "fake/headroom:test",
                ],
                env=env,
            )

            install_call = _persistent_run_call(env, profile)
            assert install_call[install_call.index("-p") + 1] == f"127.0.0.1:{port}:{port}"
            # Docker's name-only --env form preserves the caller's value,
            # including an explicitly empty value, instead of installing the
            # discovered bridge gateway default.
            assert trusted_cidrs in install_call
            assert not any(arg.startswith(f"{trusted_cidrs}=") for arg in install_call)
            assert _persistent_container_env(env, profile)[trusted_cidrs] == configured_value
            _run([*wrapper_command, "install", "remove", "--profile", profile], env=env)

        env.pop(trusted_cidrs, None)
        env["FAKE_DOCKER_GATEWAY"] = "FAIL"
        port = _free_port()
        result = _run(
            [
                *wrapper_command,
                "install",
                "apply",
                "--profile",
                "no-gateway",
                "--port",
                str(port),
                "--image",
                "fake/headroom:test",
            ],
            env=env,
        )
        install_call = _persistent_run_call(env, "no-gateway")
        assert install_call[install_call.index("-p") + 1] == f"127.0.0.1:{port}:{port}"
        assert not any(
            arg == trusted_cidrs or arg.startswith(f"{trusted_cidrs}=") for arg in install_call
        )
        assert trusted_cidrs not in _persistent_container_env(env, "no-gateway")
        assert "dashboard metadata remains restricted" in (result.stdout + result.stderr)
        _run([*wrapper_command, "install", "remove", "--profile", "no-gateway"], env=env)
    finally:
        env.pop(trusted_cidrs, None)
        env.pop("FAKE_DOCKER_GATEWAY", None)


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
        assert install_call[install_call.index("-p") + 1] == f"127.0.0.1:{port}:{port}"
        assert "/tmp/headroom-home/.headroom/memory.db" in install_call
        # Canonical filesystem contract env vars (issue #175) forwarded into
        # the container so the proxy resolves state/config to the bind mount.
        assert "HEADROOM_WORKSPACE_DIR=/tmp/headroom-home/.headroom" in install_call
        assert "HEADROOM_CONFIG_DIR=/tmp/headroom-home/.headroom/config" in install_call
        assert "HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS=172.17.0.1/32" in install_call
        assert (
            _persistent_container_env(env, "smoke")["HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS"]
            == "172.17.0.1/32"
        )

        _exercise_dashboard_gateway_overrides([str(wrapper)], env)

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


def _read_user_path_entry() -> tuple[str, int] | None:
    """Read the raw ``HKCU\\Environment`` PATH value and its registry kind, if it exists.

    Reading the registry directly rather than through
    ``[Environment]::GetEnvironmentVariable('Path','User')`` keeps two things visible that the
    .NET getter hides: the unexpanded value (the getter expands ``%USERPROFILE%``-style
    references) and the value kind, so a ``REG_EXPAND_SZ`` -> ``REG_SZ`` downgrade cannot pass
    unnoticed.
    """
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        try:
            value, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            return None
    return str(value), int(kind)


def _restore_user_path_entry(previous: tuple[str, int] | None) -> None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as key:
        if previous is None:
            try:
                winreg.DeleteValue(key, "Path")
            except FileNotFoundError:
                pass
            return
        value, kind = previous
        winreg.SetValueEx(key, "Path", 0, kind, value)


@pytest.mark.skipif(
    os.name != "nt" or _powershell_executable() is None,
    reason="Windows PowerShell coverage runs on Windows hosts only",
)
def test_powershell_installer_does_not_leak_into_user_path(tmp_path: Path) -> None:
    """The installer must not mutate the real HKCU User PATH (#2970).

    ``Ensure-PathEntry`` persists to the 'User' scope, which a HOME/USERPROFILE
    override does not redirect, so running the installer against a throwaway home
    used to leak the temp shim dir into the developer's real PATH. ``_build_env``
    now sets ``HEADROOM_INSTALL_PATH_SCOPE=Process`` to keep the update
    ephemeral; the real User PATH must be unchanged across the run.

    The assertion reads ``HKCU\\Environment`` itself instead of counting the entries the .NET
    getter reports, so it verifies the guard rather than trusting the environment variable to
    have taken effect: it catches a count-preserving mutation and a change of the value kind,
    neither of which an entry count can see.
    """
    powershell = _powershell_executable()
    assert powershell is not None

    home = tmp_path / "home"
    (home / ".local").mkdir(parents=True)
    env = _build_env(home, tmp_path)
    env["HEADROOM_DOCKER_IMAGE"] = "headroom:test-image"

    before = _read_user_path_entry()
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

        after = _read_user_path_entry()
        assert str(home) not in (after[0] if after else ""), (
            f"installer leaked the throwaway install dir into the real User PATH: {home}"
        )
        assert after == before, "installer mutated the real User PATH"
    finally:
        _cleanup_fake_docker(env)
        # A passing run never writes to the registry; this only fires if the guard regresses, so
        # that a failing test cannot leave the developer's PATH polluted.
        if _read_user_path_entry() != before:
            _restore_user_path_entry(before)


@pytest.mark.skipif(
    os.name != "nt" or _powershell_executable() is None,
    reason="Windows PowerShell coverage runs on Windows hosts only",
)
def test_powershell_mcp_wrapper_keeps_stdin_attached_for_redirected_stdio(
    tmp_path: Path,
) -> None:
    """Piped MCP stdio must still pass -i to Docker even when input is redirected."""
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
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "mcp",
                "serve",
            ],
            env=env,
            input='{"jsonrpc":"2.0","id":1,"method":"initialize"}\n',
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.returncode == 0
        mcp_call = next(call for call in _read_fake_docker_log(env) if call[:2] == ["run", "--rm"])
        assert "-i" in mcp_call
        assert "-t" not in mcp_call
    finally:
        _cleanup_fake_docker(env)


# AST-extract Ensure-PathEntry from install.ps1 and invoke it in isolation under
# a given HEADROOM_INSTALL_PATH_SCOPE, so the scope allow-list is exercised
# without running the whole installer. Parsing via the PowerShell AST (not a
# regex) keeps this pinned to the real function body. Only 'Process' (ephemeral)
# and the throwing paths are driven — never 'User', which would mutate the real
# HKCU PATH.
_ENSURE_PATH_SCOPE_HARNESS = r"""
param([string]$InstallScript, [string]$ScopeValue)
$ErrorActionPreference = 'Stop'
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $InstallScript, [ref]$null, [ref]$null)
$fn = $ast.FindAll({
    param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq 'Ensure-PathEntry'
}, $true) | Select-Object -First 1
if (-not $fn) { Write-Output 'NOFUNC'; exit 3 }
Invoke-Expression $fn.Extent.Text
$env:HEADROOM_INSTALL_PATH_SCOPE = $ScopeValue
try {
    Ensure-PathEntry -PathEntry 'C:\headroom-scope-test-marker'
    Write-Output 'OK'
} catch {
    Write-Output ('ERR:' + $_.Exception.Message)
}
"""


def _invoke_scope_harness(scope_value: str, tmp_path: Path) -> str:
    powershell = _powershell_executable()
    assert powershell is not None
    harness = tmp_path / "scope_harness.ps1"
    harness.write_text(_ENSURE_PATH_SCOPE_HARNESS, encoding="utf-8")
    result = _run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
            "-InstallScript",
            str(REPO_ROOT / "scripts" / "install.ps1"),
            "-ScopeValue",
            scope_value,
        ],
        env=os.environ.copy(),
        check=False,
    )
    return result.stdout.strip()


@pytest.mark.skipif(
    os.name != "nt" or _powershell_executable() is None,
    reason="Windows PowerShell coverage runs on Windows hosts only",
)
def test_path_scope_accepts_process_case_insensitively(tmp_path: Path) -> None:
    """'Process' (any case) is a supported ephemeral target: Ensure-PathEntry runs."""
    assert _invoke_scope_harness("process", tmp_path).endswith("OK")
    assert _invoke_scope_harness("Process", tmp_path).endswith("OK")


@pytest.mark.skipif(
    os.name != "nt" or _powershell_executable() is None,
    reason="Windows PowerShell coverage runs on Windows hosts only",
)
def test_path_scope_rejects_machine_and_invalid_values(tmp_path: Path) -> None:
    """'Machine' (system-wide) and typos must fail early, before any PATH write."""
    for bad in ("Machine", "machine", "system", "bogus"):
        out = _invoke_scope_harness(bad, tmp_path)
        assert out.startswith("ERR:"), f"scope {bad!r} was not rejected: {out!r}"
        assert "User" in out and "Process" in out, out


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
        assert install_call[install_call.index("-p") + 1] == f"127.0.0.1:{port}:{port}"
        assert "/tmp/headroom-home/.headroom/memory.db" in install_call
        # Canonical filesystem contract env vars (issue #175).
        assert "HEADROOM_WORKSPACE_DIR=/tmp/headroom-home/.headroom" in install_call
        assert "HEADROOM_CONFIG_DIR=/tmp/headroom-home/.headroom/config" in install_call
        assert "HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS=172.17.0.1/32" in install_call
        assert (
            _persistent_container_env(env, "smoke")["HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS"]
            == "172.17.0.1/32"
        )

        _exercise_dashboard_gateway_overrides(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(wrapper)],
            env,
        )

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
