"""Shared helpers for Hermes live tests and operator benchmarks."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

# spot-tech-ci agent-tool run floors (2026-06-07).
MIN_UPSTREAM_PROMPT_DELTA = 1000
MIN_HEADROOM_TOKENS_SAVED = 100
MIN_SMART_CRUSHER_SAVED = 1000
EXPECTED_TOOL_ROW_COUNT = 150
EXPECTED_COUNT_TOKEN = f"COUNT_{EXPECTED_TOOL_ROW_COUNT}"


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def hermes_health_url(hermes_base_v1: str) -> str:
    return hermes_base_v1.rstrip("/").removesuffix("/v1") + "/health"


def hermes_reachable(hermes_base_v1: str) -> bool:
    try:
        with urllib.request.urlopen(hermes_health_url(hermes_base_v1), timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def headroom_argv(repo_root: Path | None = None) -> list[str]:
    if shutil.which("headroom"):
        return ["headroom"]
    if repo_root is not None:
        return ["uv", "run", "headroom"]
    return ["headroom"]


def wait_proxy_ready(port: int, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for path in ("/readyz", "/health", "/livez"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                continue
        time.sleep(1.0)
    raise RuntimeError(f"headroom proxy on {port} did not become ready")


def start_headroom_proxy(
    *,
    port: int,
    hermes_base: str,
    repo_root: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    proc_env = {**(env or os.environ), "HEADROOM_TELEMETRY": "off"}
    return subprocess.Popen(
        [
            *headroom_argv(repo_root),
            "proxy",
            "--port",
            str(port),
            "--openai-api-url",
            hermes_base.rstrip("/"),
            "--no-telemetry",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=repo_root,
        env=proc_env,
    )


def stop_process(proc: subprocess.Popen, timeout: float = 15.0) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def assert_compression_delta(
    *,
    plain_prompt_tokens: int,
    wrapped_prompt_tokens: int,
    tokens_saved: int,
    smart_crusher_saved: int,
    content: str,
) -> None:
    upstream_delta = plain_prompt_tokens - wrapped_prompt_tokens
    assert upstream_delta >= MIN_UPSTREAM_PROMPT_DELTA, (
        f"upstream delta too low: {upstream_delta} "
        f"(plain={plain_prompt_tokens}, wrapped={wrapped_prompt_tokens})"
    )
    assert tokens_saved >= MIN_HEADROOM_TOKENS_SAVED, f"tokens.saved too low: {tokens_saved}"
    assert smart_crusher_saved >= MIN_SMART_CRUSHER_SAVED, (
        f"smart_crusher savings too low: {smart_crusher_saved}"
    )
    assert EXPECTED_COUNT_TOKEN in content, (
        f"expected {EXPECTED_COUNT_TOKEN} in response, got: {content!r}"
    )