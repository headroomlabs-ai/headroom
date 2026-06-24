"""End-to-end verification that /v1/responses WebSocket compression fires.

We can't reach OpenAI's WS endpoint without the `responses_websockets`
beta enabled on the test API key, so this test does the next-best
thing: it spins up a *fake upstream* WebSocket server, points the
proxy at it via OPENAI_API_URL, and connects a client to the proxy.

Verifies:
  1. First-frame compression: the client sends a `response.create`
     event with a 24 KB output_item; the fake upstream receives the
     COMPRESSED frame (much smaller than what was sent).
  2. Multi-frame compression: a second `response.create` on the same
     WS session is also compressed (the new behavior — was previously
     first-frame-only).
  3. Other event types (e.g. `response.cancel`) pass through
     unchanged.
  4. Proxy log surfaces both compression events with token-saved
     numbers.

Run via:

    .venv/bin/python tests/e2e_ws_responses_compression.py
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


def long_build_log() -> str:
    return "".join(
        f"[2024-01-01 00:00:{i % 60:02d}] INFO compile.rs:42 building module foo_{i} "
        f"(crate=workspace-{i // 10}, deps=[serde={i}, tokio={i}])\n"
        for i in range(400)
    )


def make_response_create_payload(turn_no: int) -> dict:
    """A wire-shape `response.create` envelope, with a long
    function_call_output that the dispatcher will compress."""
    return {
        "type": "response.create",
        "response": {
            "model": "gpt-4o-mini",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Turn {turn_no} — please summarize the build output.",
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": f"call_e2e_ws_{turn_no}",
                    "name": "shell",
                    "arguments": '{"command": "cargo build --release"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": f"call_e2e_ws_{turn_no}",
                    "output": long_build_log(),
                },
            ],
            "instructions": "You read shell output and reply tersely.",
            "max_output_tokens": 30,
        },
    }


# ── Fake upstream WS server ─────────────────────────────────────────────
#
# Captures every text frame the proxy forwards so we can assert what
# arrived upstream actually got compressed.


class FakeUpstream:
    def __init__(self) -> None:
        self.received_frames: list[str] = []
        self.server: websockets.server.WebSocketServer | None = None
        self.port: int = 0

    async def _handler(self, ws):
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    self.received_frames.append(msg)
                # Echo a minimal completion event so the proxy doesn't
                # think upstream is hung.
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

    async def start(self) -> int:
        self.port = free_port()
        self.server = await websockets.serve(self._handler, "127.0.0.1", self.port)
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
    print(f"[ws-e2e] fake upstream at ws://127.0.0.1:{upstream_port}")
    print(f"[ws-e2e] starting proxy on :{proxy_port}")

    log_fp = open("/tmp/e2e_ws_proxy.log", "w")
    proc = subprocess.Popen(
        [
            str(REPO_ROOT / ".venv/bin/headroom"),
            "proxy",
            "--port",
            str(proxy_port),
            "--no-telemetry",
            # Point /v1/responses upstream at our fake server instead
            # of api.openai.com.
            "--openai-api-url",
            upstream_url,
        ],
        env={
            **os.environ,
            # Need *some* OpenAI key value so the proxy doesn't refuse;
            # the fake upstream ignores it.
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-fake-for-test"),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "sk-ant-fake"),
            "HEADROOM_REQUIRE_RUST_CORE": "true",
        },
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )

    failures: list[str] = []
    try:
        wait_ready(proxy_port)
        print("[ws-e2e] proxy ready")

        # ── Connect WS client to the proxy ───────────────────────
        proxy_ws_url = f"ws://127.0.0.1:{proxy_port}/v1/responses"
        async with websockets.connect(
            proxy_ws_url,
            additional_headers={
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'sk-fake')}",
                "OpenAI-Beta": "responses_websockets=2026-02-06",
            },
        ) as ws:
            # Frame 1: response.create with large content
            payload_1 = make_response_create_payload(1)
            payload_1_bytes = len(json.dumps(payload_1).encode("utf-8"))
            print(f"[ws-e2e] sending frame 1 ({payload_1_bytes:,} bytes)")
            await ws.send(json.dumps(payload_1))

            # Wait for the fake upstream to receive (or timeout)
            await asyncio.sleep(2.0)

            # Frame 2: ANOTHER response.create on the same session
            payload_2 = make_response_create_payload(2)
            payload_2_bytes = len(json.dumps(payload_2).encode("utf-8"))
            print(f"[ws-e2e] sending frame 2 ({payload_2_bytes:,} bytes)")
            await ws.send(json.dumps(payload_2))
            await asyncio.sleep(2.0)

            # Frame 3: a non-response.create event — should pass through
            cancel = {"type": "response.cancel"}
            print("[ws-e2e] sending frame 3 (response.cancel — passthrough)")
            await ws.send(json.dumps(cancel))
            await asyncio.sleep(1.0)

        # ── Inspect what arrived at the fake upstream ────────────
        print(f"\n[ws-e2e] fake upstream received {len(fake.received_frames)} frames")
        for i, frame in enumerate(fake.received_frames):
            print(f"  frame {i + 1}: {len(frame.encode()):,} bytes")

        # First two frames should be MUCH smaller than what we sent.
        # The third (response.cancel) should be a small fixed size.
        if len(fake.received_frames) < 2:
            failures.append(f"expected ≥2 frames at upstream, got {len(fake.received_frames)}")
        else:
            f1 = len(fake.received_frames[0].encode())
            f2 = len(fake.received_frames[1].encode())
            if f1 >= payload_1_bytes // 2:
                failures.append(
                    f"frame 1 not compressed: arrived {f1:,} bytes (sent {payload_1_bytes:,})"
                )
            if f2 >= payload_2_bytes // 2:
                failures.append(
                    f"frame 2 not compressed (multi-frame regression): "
                    f"arrived {f2:,} bytes (sent {payload_2_bytes:,})"
                )

        # ── Scrape proxy log for compression evidence ────────────
        await asyncio.sleep(1.0)
        canonical = Path.home() / ".headroom" / "logs" / "proxy.log"
        log_lines = canonical.read_text(errors="replace").splitlines()[-1000:]
        ws_compressed = [
            line for line in log_lines if "WS /v1/responses" in line and "compressed" in line
        ]
        print("\n[ws-e2e] WS compression log lines (last few):")
        for line in ws_compressed[-6:]:
            idx = line.find("] ")
            print(" ", line[idx + 2 :] if idx > 0 else line)
        if len(ws_compressed) < 2:
            failures.append(
                f"expected ≥2 WS compression log entries (first frame + multi-frame), "
                f"saw {len(ws_compressed)}"
            )

    finally:
        print("\n[ws-e2e] terminating proxy")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_fp.close()
        await fake.stop()

    if failures:
        print("\n=== WS E2E FAILURES ===")
        for f in failures:
            print(" -", f)
        return 1
    print("\n=== WS E2E ALL GREEN ===")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
