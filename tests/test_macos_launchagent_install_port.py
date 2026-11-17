"""Run the real macOS LaunchAgent install.sh up to its port validation."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "deployment"
    / "macos-launchagent"
    / "install.sh"
)

pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="needs POSIX Python and bash: the fixture passes native paths and env to bash",
)


def _stub(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/bin/sh\n" + body + "\n", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run_install(tmp_path: Path, port: str) -> subprocess.CompletedProcess[str]:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _stub(stubs, "uname", "echo Darwin")
    _stub(stubs, "headroom", "exit 0")
    _stub(stubs, "launchctl", "exit 0")
    _stub(stubs, "lsof", "exit 1")
    home = tmp_path / "home"
    home.mkdir()
    # Only the script is copied, not its plist template: a port that passes
    # validation stops at "Template file not found", before anything is written
    # outside tmp_path.
    script = tmp_path / "install.sh"
    shutil.copy(SCRIPT, script)
    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
    }
    return subprocess.run(
        ["bash", str(script), "--port", port, "--unattended"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    "port",
    ["0100000", "08080", "18446744073709553616", "01024", "1023", "65536", "abc"],
)
def test_invalid_port_is_rejected(tmp_path: Path, port: str) -> None:
    result = _run_install(tmp_path, port)
    assert result.returncode != 0
    assert "Invalid port number" in result.stderr


@pytest.mark.parametrize("port", ["1024", "8787", "65535"])
def test_valid_port_passes_the_port_check(tmp_path: Path, port: str) -> None:
    result = _run_install(tmp_path, port)
    assert "Invalid port number" not in result.stderr
    assert "Template file not found" in result.stderr
