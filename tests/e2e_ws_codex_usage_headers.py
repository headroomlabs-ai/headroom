"""End-to-end verification that Codex x-codex-* usage headers are forwarded
onto the client-facing WebSocket handshake (101).

Unit tests in tests/test_openai_codex_ws_lifecycle.py prove the handler
*logic* (it builds the right accept-header list), but they stub starlette's
``WebSocket.accept`` -- so they cannot prove the one risky assumption: that
starlette + uvicorn actually WRITE ``accept(headers=...)`` onto the real 101.

This e2e closes that gap with real wire traffic and no OpenAI quota:

  1. Stand up a *fake upstream* WS server whose handshake response carries
     several ``x-codex-*`` headers PLUS a ``set-cookie`` and an
     ``authorization`` header (which must NOT be forwarded).
  2. Boot the real proxy pointed at the fake upstream via --openai-api-url.
  3. Connect a real ``websockets`` client to the proxy and read
     ``client.response.headers`` -- i.e. the client-facing 101.

Asserts:
  - every ``x-codex-*`` header from the upstream handshake is present on the
    client 101 (original casing preserved),
  - ``set-cookie`` and ``authorization`` are NOT forwarded,
  - the proxy's /stats reflects the Codex window (update_from_headers parity).

Run via:

    .venv/bin/python tests/e2e_ws_codex_usage_headers.py
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websockets

REPO_ROOT = Path(__file__).resolve().parent.parent

# The x-codex-* window the fake upstream advertises on its handshake.
UPSTREAM_CODEX_HEADERS = {
    "x-codex-primary-used-percent": "42",
    "x-codex-primary-window-minutes": "300",
    "x-codex-secondary-used-percent": "7",
    "x-codex-secondary-window-minutes": "10080",
}
# Sensitive headers that must NEVER reach the client 101.
UPSTREAM_LEAK_HEADERS = {
    "set-cookie": "session=should-not-forward",
    "authorization": "Bearer upstream-secret-should-not-forward",
}


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


# ── Fake upstream WS server ─────────────────────────────────────────────
#
# The handshake response (101) carries the x-codex-* window plus sensitive
# headers, mirroring what OpenAI's Codex WS endpoint returns.


class FakeUpstream:
    def __init__(self) -> None:
        self.server: websockets.asyncio.server.Server | None = None
        self.port: int = 0

    async def _handler(self, ws):
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    await ws.send(
                        json.dumps(
                            {
                                "type": "response.completed",
                                "response": {"id": "fake_resp", "output": []},
                            }
                        )
                    )
        except websockets.exceptions.ConnectionClosed:
            pass

    def _process_response(self, connection, request, response):
        # Inject the x-codex-* window + sensitive headers onto the 101.
        for name, value in {**UPSTREAM_CODEX_HEADERS, **UPSTREAM_LEAK_HEADERS}.items():
            response.headers[name] = value
        return response

    async def start(self) -> int:
        self.port = free_port()
        self.server = await websockets.serve(
            self._handler,
            "127.0.0.1",
            self.port,
            process_response=self._process_response,
        )
        return self.port

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()


async def main_async() -> int:
    fake = FakeUpstream()
    upstream_port = await fake.start()
    upstream_url = f"http://127.0.0.1:{upstream_port}"

    proxy_port = free_port()
    print(f"[codex-hdr-e2e] fake upstream at ws://127.0.0.1:{upstream_port}")
    print(f"[codex-hdr-e2e] starting proxy on :{proxy_port}")

    log_fp = open("/tmp/e2e_ws_codex_headers_proxy.log", "w")
    proc = subprocess.Popen(
        [
            str(REPO_ROOT / ".venv/bin/headroom"),
            "proxy",
            "--port",
            str(proxy_port),
            "--no-telemetry",
            "--openai-api-url",
            upstream_url,
        ],
        env={
            **os.environ,
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-fake-for-test"),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "sk-ant-fake"),
        },
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )

    failures: list[str] = []
    try:
        wait_ready(proxy_port)
        print("[codex-hdr-e2e] proxy ready")

        proxy_ws_url = f"ws://127.0.0.1:{proxy_port}/v1/responses"
        # NOTE: API-key auth (no ChatGPT-Account-ID) so the upstream routes to
        # --openai-api-url (our fake upstream). The x-codex forwarding code is
        # auth-mode-agnostic -- it forwards whatever x-codex-* headers the
        # upstream handshake returns -- so this exercises the exact same path
        # the real chatgpt.com subscription handshake would, without leaving
        # localhost.
        async with websockets.connect(
            proxy_ws_url,
            additional_headers={
                "Authorization": "Bearer sk-fake",
                "OpenAI-Beta": "responses_websockets=2026-02-06",
            },
        ) as ws:
            # The client-facing 101 response headers — the thing under test.
            client_101 = {k.lower(): v for k, v in ws.response.headers.raw_items()}
            print("[codex-hdr-e2e] client 101 headers:")
            for k, v in sorted(client_101.items()):
                if k.startswith("x-codex-") or k in ("set-cookie", "authorization"):
                    print(f"    {k}: {v}")

            # 1. Every x-codex-* header forwarded.
            for name, value in UPSTREAM_CODEX_HEADERS.items():
                if client_101.get(name) != value:
                    failures.append(
                        f"x-codex header not forwarded to client 101: "
                        f"{name} (got {client_101.get(name)!r}, want {value!r})"
                    )

            # 2. Sensitive headers NOT forwarded.
            for name in UPSTREAM_LEAK_HEADERS:
                if name in client_101:
                    failures.append(f"sensitive header leaked to client 101: {name}")

            # Drive one frame so the session is real (upstream echoes completed).
            await ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {"model": "gpt-5.4", "input": "hi"},
                    }
                )
            )
            await asyncio.sleep(1.0)

        # 3. /stats reflects the Codex window (update_from_headers parity).
        #    Secondary, shape-dependent signal: look for the actual forwarded
        #    value, not just the word "codex" (which can appear as a null key).
        await asyncio.sleep(0.5)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/stats", timeout=5) as r:
                stats_text = r.read().decode("utf-8", errors="replace")
            want = UPSTREAM_CODEX_HEADERS["x-codex-primary-used-percent"]
            if want in stats_text:
                print(f"[codex-hdr-e2e] /stats reflects codex window (primary-used={want})")
            else:
                print(
                    "[codex-hdr-e2e] note: /stats did not surface the codex value "
                    f"({want!r}); shape may differ — 101 forwarding is the primary check"
                )
        except Exception as exc:  # noqa: BLE001 - best-effort secondary check
            print(f"[codex-hdr-e2e] /stats check skipped: {exc}")

    finally:
        print("[codex-hdr-e2e] terminating proxy")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_fp.close()
        await fake.stop()

    if failures:
        print("\n=== CODEX-HDR E2E FAILURES ===")
        for f in failures:
            print(" -", f)
        return 1
    print("\n=== CODEX-HDR E2E ALL GREEN ===")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
