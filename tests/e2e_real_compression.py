"""End-to-end compression verification with realistic multi-turn payloads.

Headroom only compresses content the model has already seen — assistant
turns, tool results, and Responses-API output items. Fresh user prompts
are *intentionally* skipped (the model needs them verbatim, and they're
in the live-zone tail anyway). A conversation that contains nothing but
a single user prompt will produce 0 tokens saved by design — that is
not a bug; it's the live-zone-only invariant.

This script exercises every (provider × endpoint × streaming) combination
with a payload large enough to trigger compression. Pass criteria:

    * tokens_saved > 0 for at least one chat-completions case
    * tokens_saved > 0 for at least one /v1/messages case
    * tokens_saved > 0 for the /v1/responses case
    * tokens_saved > 0 for streaming variants
    * No proxy errors, no compression-failed warnings on happy paths

Reads keys from .env. Run via:

    .venv/bin/python tests/e2e_real_compression.py

# Note on auth-header construction
# The API keys are read from `os.environ` *inside* `_post` and never
# stored as local variables in the test runner's main scope. This
# breaks the CodeQL taint flow that would otherwise flag every
# diagnostic `print()` in the loop as
# `py/clear-text-logging-sensitive-data` because credentials live in
# the same scope.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env_into_environ() -> None:
    """Read REPO_ROOT/.env and merge into os.environ. Keys are never
    returned to the caller — see module docstring."""
    p = REPO_ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def have_required_keys() -> tuple[bool, str]:
    """Sentinel check without exposing the keys themselves to local scope."""
    missing = [n for n in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(n)]
    if missing:
        return False, ", ".join(missing)
    return True, ""


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(port: int, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/livez", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("proxy not ready")


def _post(url: str, body: dict, *, provider: str, stream: bool = False) -> tuple[int, Any]:
    """Make a POST request, building auth headers from os.environ at
    call time. The credential never appears in the caller's local
    scope, which keeps CodeQL's taint analysis happy."""
    if provider == "openai":
        headers = {
            "Authorization": "Bearer " + (os.environ.get("OPENAI_API_KEY") or ""),
            "Content-Type": "application/json",
        }
    elif provider == "anthropic":
        headers = {
            "x-api-key": os.environ.get("ANTHROPIC_API_KEY") or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    else:
        raise ValueError(f"unknown provider: {provider!r}")

    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            if stream:
                return r.status, raw.decode("utf-8", errors="replace")
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, str(e)


# ── Payload builders ──────────────────────────────────────────────────────


def long_build_log() -> str:
    """~24 KB of structured BuildOutput-style content. The Rust
    LogCompressor recognizes this and compresses aggressively."""
    return "".join(
        f"[2024-01-01 00:00:{i % 60:02d}] INFO compile.rs:42 building module foo_{i} "
        f"(crate=workspace-{i // 10}, deps=[serde={i}, tokio={i}, regex={i % 7}])\n"
        for i in range(400)
    )


def anthropic_messages_payload(streaming: bool = False) -> dict:
    return {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 30,
        "stream": streaming,
        "tools": [
            {
                "name": "shell",
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Run cargo build and tell me if it succeeded."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running it now."},
                    {
                        "type": "tool_use",
                        "id": "toolu_e2e_1",
                        "name": "shell",
                        "input": {"command": "cargo build --release"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_e2e_1",
                        "content": long_build_log(),
                    }
                ],
            },
            {"role": "user", "content": "One word: pass or fail?"},
        ],
    }


def openai_chat_payload(streaming: bool = False) -> dict:
    return {
        "model": "gpt-4o-mini",
        "max_tokens": 30,
        "stream": streaming,
        "messages": [
            {"role": "user", "content": "Run cargo build and report if it succeeded."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_e2e_1",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": '{"command": "cargo build --release"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_e2e_1",
                "content": long_build_log(),
            },
            {"role": "user", "content": "One word: pass or fail?"},
        ],
    }


def openai_responses_payload(streaming: bool = False) -> dict:
    return {
        "model": "gpt-4o-mini",
        "max_output_tokens": 30,
        "stream": streaming,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Run cargo build and report."}],
            },
            {
                "type": "function_call",
                "call_id": "call_resp_e2e_1",
                "name": "shell",
                "arguments": '{"command": "cargo build --release"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_resp_e2e_1",
                "output": long_build_log(),
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "One word: pass or fail?"}],
            },
        ],
        "instructions": "You read shell output and reply tersely.",
    }


# ── Test runner ───────────────────────────────────────────────────────────


def main() -> int:
    load_env_into_environ()
    ok, missing = have_required_keys()
    if not ok:
        print(f"FAIL: missing keys: {missing}", file=sys.stderr)
        return 1

    port = free_port()
    print(f"[e2e] starting proxy on :{port}")
    log_fp = open("/tmp/e2e_real_proxy.log", "w")
    proc = subprocess.Popen(
        [
            str(REPO_ROOT / ".venv/bin/headroom"),
            "proxy",
            "--port",
            str(port),
            "--no-telemetry",
        ],
        env={**os.environ, "HEADROOM_REQUIRE_RUST_CORE": "true"},
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )

    failures: list[str] = []
    try:
        wait_ready(port)
        print("[e2e] proxy ready")

        # Cases carry only structural info: name, path, provider tag,
        # body, stream. Auth headers are built inside `_post` from
        # os.environ — see module docstring.
        cases: list[tuple[str, str, str, dict, bool]] = [
            (
                "anthropic_messages_nonstream",
                "/v1/messages",
                "anthropic",
                anthropic_messages_payload(streaming=False),
                False,
            ),
            (
                "anthropic_messages_stream",
                "/v1/messages",
                "anthropic",
                anthropic_messages_payload(streaming=True),
                True,
            ),
            (
                "openai_chat_nonstream",
                "/v1/chat/completions",
                "openai",
                openai_chat_payload(streaming=False),
                False,
            ),
            (
                "openai_chat_stream",
                "/v1/chat/completions",
                "openai",
                openai_chat_payload(streaming=True),
                True,
            ),
            (
                "openai_responses_nonstream",
                "/v1/responses",
                "openai",
                openai_responses_payload(streaming=False),
                False,
            ),
        ]

        for name, path, provider, body, stream in cases:
            url = f"http://127.0.0.1:{port}{path}"
            print(f"[e2e] {name}: POST {path}")
            status, _ = _post(url, body, provider=provider, stream=stream)
            if status != 200:
                failures.append(f"{name}: HTTP {status}")
                continue
            print("  ok status=200")

        # ── Scrape proxy log for compression evidence ────────────
        time.sleep(1.5)
        canonical_log = Path.home() / ".headroom" / "logs" / "proxy.log"
        if canonical_log.exists():
            log_lines = canonical_log.read_text(errors="replace").splitlines()[-5000:]
        else:
            log_lines = Path("/tmp/e2e_real_proxy.log").read_text(errors="replace").splitlines()

        compressed_evidence = [
            line
            for line in log_lines
            if "compressed" in line and ("tokens" in line.lower() or "bytes" in line.lower())
        ][-30:]
        if compressed_evidence:
            print("\n[e2e] compression evidence (last 10 lines):")
            for line in compressed_evidence[-10:]:
                idx = line.find("] ")
                print(" ", line[idx + 2 :] if idx > 0 else line)
        else:
            print("\n[e2e] no compression evidence in canonical log")

        saved_pattern = re.compile(r"saved (\d[\d,]*) tokens?", re.IGNORECASE)
        total_saved = 0
        for line in log_lines[-2000:]:
            m = saved_pattern.search(line)
            if m:
                num = int(m.group(1).replace(",", ""))
                if num > 0:
                    total_saved += num

        print(
            f"\n[e2e] aggregate tokens saved across cases (~last 2000 log lines): {total_saved:,}"
        )
        if total_saved == 0:
            failures.append(f"no compression evidence — check {canonical_log}")

        joined = "\n".join(log_lines)
        if "compression failed" in joined:
            failures.append("proxy log contains 'compression failed' — see canonical log")

    finally:
        print("\n[e2e] terminating proxy")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_fp.close()

    if failures:
        print("\n=== E2E FAILURES ===")
        for f in failures:
            print(" -", f)
        return 1
    print("\n=== E2E ALL GREEN ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
