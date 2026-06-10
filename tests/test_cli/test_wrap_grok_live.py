"""Optional live Grok + Headroom integration tests.

Run manually when Grok auth is available on the host:

    HEADROOM_LIVE_GROK=1 UV_SKIP_WHEEL_FILENAME_CHECK=1 \
        uv run --extra dev pytest tests/test_cli/test_wrap_grok_live.py -v

Captured on 2026-06-07 (CachyOS, grok 0.2.32, authenticated ~/.grok/auth.json):

**Without Headroom** (direct to cli-chat-proxy.grok.com):

    $ grok -p "Reply with exactly one word: PLAIN_OK" \\
        --output-format plain --no-plan --always-approve
    PLAIN_OK
    # ~12s wall time

**With Headroom** (`headroom wrap grok`, port 8791):

    $ headroom wrap grok --port 8791 -p "Reply with exactly one word: HEADROOM_OK" \\
        --output-format plain --no-plan --always-approve
    HEADROOM_OK
    # ~25s first run (includes proxy cold-start + Kompress model load)

Proxy log confirms Grok traffic is forwarded to the real upstream after the fix:

    GET  https://cli-chat-proxy.grok.com/v1/settings  → HTTP/2 200 OK
    POST https://cli-chat-proxy.grok.com/v1/sessions/register → HTTP/2 200 OK

Before the upstream fix, wrap grok defaulted the proxy to api.openai.com and
/v1/settings returned 404. `wrap grok` now passes
``openai_api_url=https://cli-chat-proxy.grok.com/v1`` to the proxy launcher.

For token benchmarks on ``/v1/chat/completions``, use Hermes + Headroom instead
(``HEADROOM_LIVE_HERMES=1`` — see ``tests/test_cli/test_hermes_grok_live.py``).
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

from headroom.providers.grok.runtime import DEFAULT_API_URL, GROK_PROXY_ENV, proxy_base_url

_LIVE = os.environ.get("HEADROOM_LIVE_GROK") == "1"
pytestmark = pytest.mark.skipif(not _LIVE, reason="set HEADROOM_LIVE_GROK=1 to run live Grok tests")


def _grok_auth_present() -> bool:
    return (Path.home() / ".grok" / "auth.json").exists()


@pytest.mark.skipif(not _grok_auth_present(), reason="~/.grok/auth.json missing")
def test_live_plain_grok_returns_expected_token() -> None:
    result = subprocess.run(
        [
            "grok",
            "-p",
            "Reply with exactly one word: PLAIN_OK",
            "--output-format",
            "plain",
            "--no-plan",
            "--always-approve",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd="/tmp",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PLAIN_OK" in result.stdout


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.skipif(not _grok_auth_present(), reason="~/.grok/auth.json missing")
def test_live_headroom_wrap_grok_routes_through_local_proxy(tmp_path: Path) -> None:
    port = _pick_free_port()
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HEADROOM_TELEMETRY"] = "off"

    result = subprocess.run(
        [
            "uv",
            "run",
            "headroom",
            "wrap",
            "grok",
            "--port",
            str(port),
            "-p",
            "Reply with exactly one word: HEADROOM_OK",
            "--output-format",
            "plain",
            "--no-plan",
            "--always-approve",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=repo_root,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HEADROOM_OK" in result.stdout
    assert f"{GROK_PROXY_ENV}={proxy_base_url(port)}" in result.stdout

    log_path = Path.home() / ".headroom" / "logs" / "proxy.log"
    if log_path.exists():
        # Give the proxy a moment to flush logs after wrap exits.
        time.sleep(0.5)
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
        assert DEFAULT_API_URL.rstrip("/v1") in log_tail or "cli-chat-proxy.grok.com" in log_tail
