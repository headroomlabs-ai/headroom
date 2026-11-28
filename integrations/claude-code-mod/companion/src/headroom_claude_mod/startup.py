"""Start the local companion without launching another Claude session."""

import importlib.util
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path


def start_proxy(base, check):
    try:
        check(base)
        return
    except (ValueError, OSError):
        pass
    if importlib.util.find_spec("headroom") is None:
        raise ValueError(
            "Install Headroom and its claude-mod companion in the same Python environment first."
        )
    url = urllib.parse.urlsplit(base)
    logs = (
        Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "headroom" / "sidebar"
    )
    logs.mkdir(parents=True, exist_ok=True)
    options = (
        {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    with (logs / "proxy.log").open("ab") as output:
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from headroom.cli import main; main()",
                "proxy",
                "--host",
                url.hostname,
                "--port",
                str(url.port),
                "--proxy-extension",
                "claude_mod",
            ],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            close_fds=True,
            **options,
        )
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise ValueError(f"Headroom could not start. See {logs / 'proxy.log'}.")
        try:
            check(base)
            return
        except (ValueError, OSError):
            time.sleep(0.25)
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)
    raise ValueError(
        f"Headroom is still starting or unavailable. Check {logs / 'proxy.log'} and retry."
    )
