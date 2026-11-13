"""Opt-in real Antigravity recovery proof; requires an authenticated agy on PATH.

Run: HEADROOM_TEST_AGY_LIVE=1 .venv/bin/python -m pytest \
    tests/test_cli/test_init_agy_live.py -q -s

Uses temporary Headroom state and shell configuration, preserving user dotfiles.
This tests a stopped proxy and fresh shells, not an actual OS reboot.
"""

import importlib
import os
import shutil
import socket
import subprocess
import sys
import urllib.request

import pytest
from click.testing import CliRunner


@pytest.mark.skipif(
    os.environ.get("HEADROOM_TEST_AGY_LIVE") != "1", reason="requires live Antigravity OAuth"
)
def test_agy_recovers_stopped_proxy_in_fresh_shell(monkeypatch, tmp_path):
    from headroom.install.runtime import stop_runtime, wait_ready
    from headroom.install.state import load_manifest

    init_cli = importlib.import_module("headroom.cli.init")
    assert shutil.which("agy"), "Install and authenticate agy first"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("HEADROOM_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv(
        "PATH", str(os.path.dirname(sys.executable)) + os.pathsep + os.environ["PATH"]
    )
    shell_profile = tmp_path / ".profile"
    monkeypatch.setattr(init_cli, "unix_user_env_targets", lambda: [shell_profile])
    monkeypatch.setattr("headroom.install.providers.unix_user_env_targets", lambda: [shell_profile])
    monkeypatch.setattr(init_cli, "_agy_hooks_path", lambda: tmp_path / "hooks.json")

    def launch(shell):
        result = subprocess.run(
            [shell, "-c", '. "$1"; agy -p "Reply with exactly: pong"', shell, str(shell_profile)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        print(f"fresh {shell}: exit={result.returncode}, response={result.stdout.strip()!r}")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().lower() == "pong"

    try:
        result = CliRunner().invoke(init_cli.init, ["-g", "--port", str(port), "agy"])
        assert result.exit_code == 0, result.output
        manifest = load_manifest("init-user")
        assert manifest is not None
        assert wait_ready(manifest, timeout_seconds=15)
        print(f"init completed; proxy ready on port {port}")
        launch("bash")
        stop_runtime(manifest)
        assert not wait_ready(manifest, timeout_seconds=1)
        print("proxy stopped; health unavailable")
        launch("zsh" if shutil.which("zsh") else "bash")
        assert wait_ready(manifest, timeout_seconds=1)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=5) as response:
            import json

            stats = json.load(response)
        agents = [
            agent for agent in stats["agent_usage"]["agents"] if agent["agent"] == "antigravity"
        ]
        assert agents and agents[0]["requests"] >= 2, agents
        print(f"recovered proxy attribution: {agents}")
    finally:
        manifest = load_manifest("init-user")
        if manifest is not None:
            stop_runtime(manifest)
