"""Smoke test for scripts/repro_codex_replay.py.

Spins up a minimal FastAPI + websockets mock proxy that answers ``/livez``,
``/v1/messages``, and ``/v1/responses`` (WS), then invokes the harness'
``main()`` in-process and verifies it exits 0 with the expected summary shape.

The mock does *not* implement Codex semantics — it just accepts the WS
handshake, consumes the first frame, and closes. That's enough to exercise
the harness end-to-end in < 10s.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
import socket
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect


@pytest.fixture(autouse=True)
def _restore_real_websockets_module() -> Iterator[None]:
    """Some earlier tests replace ``sys.modules['websockets']`` with a stub.

    uvicorn's websockets-sansio backend imports ``websockets.server`` at
    connect time; if a stub is installed the import fails and the mock
    proxy never starts. Force-reload the real package before this test.
    """
    originals = {
        name: sys.modules.pop(name, None)
        for name in list(sys.modules)
        if name == "websockets" or name.startswith("websockets.")
    }
    importlib.import_module("websockets")
    importlib.import_module("websockets.asyncio.server")
    yield
    for name in list(sys.modules):
        if name == "websockets" or name.startswith("websockets."):
            del sys.modules[name]
    for name, mod in originals.items():
        if mod is not None:
            sys.modules[name] = mod


# Make sure `scripts/` is importable when running via pytest from repo root.
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import repro_codex_replay  # type: ignore[import-not-found]  # noqa: E402

# ---------------------------------------------------------------------------
# Mock proxy server
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/messages")
    async def messages(request: Request) -> dict[str, object]:
        # Drain body so the client gets a clean completion.
        _ = await request.body()
        return {
            "id": "msg_mock_001",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-mock",
            "stop_reason": "end_turn",
        }

    @app.websocket("/v1/responses")
    async def responses_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            # Accept one frame and then close. Harness treats this as
            # "handshake worked" — no need to emulate Codex events.
            with contextlib.suppress(WebSocketDisconnect):
                await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        with contextlib.suppress(Exception):
            await websocket.close()

    return app


class _ServerThread:
    """Run uvicorn in a background thread bound to a dedicated port."""

    def __init__(self, app: FastAPI, port: int) -> None:
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            loop="asyncio",
            lifespan="on",
            ws="websockets-sansio",
        )
        self.server = uvicorn.Server(config)
        self.port = port
        self.thread = threading.Thread(target=self.server.run, name="mock-proxy", daemon=True)

    def start(self) -> None:
        self.thread.start()
        # Wait for the socket to accept connections.
        deadline = 5.0
        import time

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < deadline:
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("mock proxy failed to start within 5s")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5.0)


@pytest.fixture
def mock_proxy() -> Iterator[str]:
    port = _free_port()
    srv = _ServerThread(_build_app(), port)
    srv.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run_harness(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout, stderr
    try:
        rc = repro_codex_replay.main(argv)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return rc, stdout.getvalue(), stderr.getvalue()


def test_harness_runs_against_mock_proxy_and_exits_zero(mock_proxy: str) -> None:
    rc, out, err = _run_harness(
        [
            "--url",
            mock_proxy,
            "--ws-clients",
            "2",
            "--anthropic-clients",
            "2",
            "--duration",
            "2",
            "--warmup-timeout",
            "3",
            "--livez-threshold-ms",
            "2000",
            "--livez-interval-ms",
            "100",
            "--json",
        ]
    )
    combined = out + err
    assert rc == 0, f"expected exit 0, got {rc}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    # Human summary shape.
    for needle in (
        "Codex proxy reconnect-storm repro harness",
        "Warmup:",
        "Storm:",
        "/livez:",
        "Codex WS:",
        "Anthropic HTTP:",
        "RESULT: OK",
    ):
        assert needle in combined, f"missing '{needle}' in output:\n{combined}"
    # JSON payload shape — --json appends one indent=2 JSON object at the end.
    import json as _json

    # Find the last top-level JSON object in stdout. With indent=2 the closing
    # brace is on a line by itself (no leading whitespace), so we search for
    # the final "\n}" boundary, then walk back to the matching "{".
    end_brace = out.rfind("\n}")
    assert end_brace != -1, f"no JSON payload found in stdout:\n{out}"
    # The matching opening brace must be preceded by a newline and start a line.
    # Scan backwards for a line that is exactly "{".
    lines = out[: end_brace + 2].splitlines()
    # Find the last line equal to "{" — that's the start of the JSON object.
    start_line_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i] == "{":
            start_line_idx = i
            break
    assert start_line_idx is not None, f"could not locate JSON start in stdout:\n{out}"
    payload_text = "\n".join(lines[start_line_idx:])
    payload = _json.loads(payload_text)
    for key in ("ok", "warmup", "storm", "livez", "codex_ws", "anthropic_http"):
        assert key in payload, f"summary missing key {key!r}: {payload}"
    assert payload["ok"] is True
    assert payload["storm"]["ws_clients"] == 2
    assert payload["storm"]["anthropic_clients"] == 2
    assert payload["livez"]["count"] > 0
    # The mock accepts /v1/messages with 200 — at least one should succeed.
    assert payload["anthropic_http"]["ok_2xx"] >= 1
    # Every WS client should have opened (handshake works against the mock).
    assert payload["codex_ws"]["opened"] == 2


def test_harness_reports_unreachable_proxy_fast() -> None:
    import time

    t0 = time.perf_counter()
    rc, out, err = _run_harness(
        [
            "--url",
            "http://127.0.0.1:1",  # deliberately closed port
            "--ws-clients",
            "1",
            "--anthropic-clients",
            "1",
            "--duration",
            "1",
        ]
    )
    elapsed = time.perf_counter() - t0
    # P3 Fix 25: EXIT_PROXY_UNREACHABLE=2 distinguishes "proxy didn't
    # answer" from a generic crash (exit 1) or a threshold miss (exit 3).
    assert rc == repro_codex_replay.EXIT_PROXY_UNREACHABLE, (
        f"expected exit {repro_codex_replay.EXIT_PROXY_UNREACHABLE}, "
        f"got {rc}. stdout={out} stderr={err}"
    )
    assert "unreachable" in (out + err).lower()
    assert elapsed < 10.0, f"unreachable detection too slow: {elapsed:.2f}s"
