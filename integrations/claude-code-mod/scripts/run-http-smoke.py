"""Start an actual local HTTP server around the shipped extension and fixture records."""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "companion" / "src"), str(root / "companion" / "tests")]
from conftest import OTHER, Log, make_app  # noqa: E402 -- fixture path is added above
from headroom_claude_mod.launcher import preflight  # noqa: E402 -- companion source path above

with pytest.MonkeyPatch.context() as mp:
    app, _, _, _ = make_app(
        mp,
        entries=[
            Log(),
            Log(
                request_id="other",
                tags={"mod-session": OTHER},
                request_messages=[{"content": "OTHER_SESSION_SECRET"}],
            ),
        ],
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, log_level="error", lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        for _ in range(300):
            if server.started:
                break
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("Local test server did not start")
        base = f"http://127.0.0.1:{port}"
        info = preflight(base)
        assert info["schema_version"] == 1
        subprocess.run(["node", str(root / "scripts" / "http-smoke.mjs"), base], check=True)
        installed = os.environ.get("INSTALLED_HEADROOM_MOD")
        if installed:
            subprocess.run([installed, "doctor", "--proxy-url", base], check=True)
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
