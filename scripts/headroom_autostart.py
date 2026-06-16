"""Headroom proxy auto-start for Claude Code and Codex.
Checks if proxy is running, starts it if not.

Proxy chain:
  Claude Code → headroom(8787) → ANTHROPIC_TARGET_API_URL → Anthropic API
  Codex       → headroom(8787) → HTTP_PROXY → OpenAI API

Configuration via environment variables:
  ANTHROPIC_TARGET_API_URL  upstream for Anthropic requests (default: http://127.0.0.1:15721)
  HEADROOM_PROXY_PORT       proxy listen port (default: 8787)
  HEADROOM_PYTHON           path to Python executable with headroom installed
  HTTP_PROXY / HTTPS_PROXY  proxy for OpenAI requests (read from env automatically)
"""
import socket
import subprocess
import sys
import os

PROXY_PYTHON = os.environ.get("HEADROOM_PYTHON", sys.executable)
PROXY_PORT = int(os.environ.get("HEADROOM_PROXY_PORT", "8787"))
ANTHROPIC_UPSTREAM = os.environ.get("ANTHROPIC_TARGET_API_URL", "http://127.0.0.1:15721")


def check_proxy(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            s.connect(("127.0.0.1", port))
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


def main():
    if check_proxy(PROXY_PORT):
        print(f"headroom proxy already running on port {PROXY_PORT}")
        return

    print(f"starting headroom proxy on port {PROXY_PORT}...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # Anthropic upstream: cc-switch
    env["ANTHROPIC_TARGET_API_URL"] = ANTHROPIC_UPSTREAM

    # OpenAI upstream: inherit HTTP_PROXY from environment if set
    # Users configure this in their shell profile or Codex config.toml [env]
    if os.environ.get("HTTP_PROXY"):
        env.setdefault("HTTP_PROXY", os.environ["HTTP_PROXY"])
        env.setdefault("HTTPS_PROXY", os.environ["HTTPS_PROXY"])
    env.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")

    log_path = os.path.join(os.environ.get("TEMP", "/tmp"), "headroom_proxy_err.log")
    log_file = open(log_path, "a")

    proc = subprocess.Popen(
        [PROXY_PYTHON, "-m", "headroom.cli", "proxy",
         "--port", str(PROXY_PORT),
         "--anthropic-api-url", ANTHROPIC_UPSTREAM],
        stdout=log_file,
        stderr=log_file,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    # Wait up to 50s for proxy to start
    import time
    for i in range(50):
        time.sleep(1)
        if check_proxy(PROXY_PORT):
            print(f"headroom proxy started (PID={proc.pid})")
            return
        if proc.poll() is not None:
            print(f"headroom proxy failed to start (exit={proc.returncode})")
            return

    print("headroom proxy startup timeout")


if __name__ == "__main__":
    main()
