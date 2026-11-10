"""Run the real macOS LaunchAgent install.sh through its "Reinstall?" prompt."""

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


def _run_install(tmp_path: Path, answer: str) -> tuple[subprocess.CompletedProcess[str], str]:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    log = tmp_path / "launchctl.log"
    _stub(stubs, "uname", "echo Darwin")
    _stub(stubs, "headroom", "exit 0")
    _stub(stubs, "lsof", "exit 1")
    # print succeeds, so the service counts as running
    _stub(stubs, "launchctl", f'echo "$@" >> "{log}"\nexit 0')
    home = tmp_path / "home"
    plist_dir = home / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True)
    (plist_dir / "com.headroom.proxy.plist").write_text("<plist/>")
    # Only the script is copied, not its plist template: a confirmed reinstall
    # stops at "Template file not found" without touching anything outside tmp_path.
    script = tmp_path / "install.sh"
    shutil.copy(SCRIPT, script)
    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
    }
    result = subprocess.run(
        ["bash", str(script), "--port", "8787"],
        input=answer,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, log.read_text() if log.exists() else ""


def test_declining_reinstall_keeps_the_service_running(tmp_path: Path) -> None:
    result, calls = _run_install(tmp_path, "N\n")
    assert result.returncode == 0
    assert "Installation cancelled." in result.stdout
    assert "bootout" not in calls


def test_confirming_reinstall_stops_the_service(tmp_path: Path) -> None:
    _, calls = _run_install(tmp_path, "y\n")
    assert "bootout" in calls
