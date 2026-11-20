"""Owned Uvicorn process and TLS upstream with explicit event barriers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from tests.unified_gateway.process.harness import _unused_loopback_port, gateway_config_digest


@contextmanager
def http_process(
    local_pki,
    tmp_path,
    handler,
    *,
    configure=None,
    fail_connect=False,
    public_provider=False,
    reserve_barrier=0,
    websocket_handler=None,
    acquire_barrier=False,
):
    context, raw = local_pki
    calls = []

    class Upstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.body = self.rfile.read(int(self.headers.get("content-length", "0")))
            calls.append((self.path, self.body, dict(self.headers)))
            handler(self)

        do_GET = do_POST
        do_DELETE = do_POST

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    ws_server = ws_thread = None
    if websocket_handler is not None:
        from websockets.sync.server import serve

        ws_server = serve(websocket_handler, "127.0.0.1", 0, ssl=context, close_timeout=1)
        ws_thread = threading.Thread(target=ws_server.serve_forever, daemon=True)
        ws_thread.start()
    raw["runtime"]["port"] = _unused_loopback_port()
    raw["transport"]["ca_bundle"] = str(tmp_path / "ca.pem")
    hostname = "api.anthropic.com" if public_provider else "llm.internal.example"
    origin = (
        f"https://{hostname}" if public_provider else f"https://{hostname}:{server.server_port}"
    )
    raw["credentials"][0]["allowed_origins"] = [origin]
    route = raw["routes"][0]
    route.update(
        upstream_origin=origin, public_model="fixture-model", upstream_model="fixture-model"
    )
    route.setdefault("retry", {"ambiguous_commit": "never", "after_output": "never"}).update(
        max_attempts=2, base_backoff_seconds=0.001
    )
    route["ingress_protocols"] = route["native_protocols"] = [
        "openai-chat",
        "openai-responses",
        "anthropic-messages",
        "gemini-generate",
    ]
    route["capabilities"] = {
        p: {t: {"features": ["text", "tools"]} for t in ["http-json", "http-stream"]}
        for p in route["ingress_protocols"]
    }
    route["pricing"] = {
        "input_usd_per_million": "1",
        "output_usd_per_million": "2",
        "cache_read_usd_per_million": "1",
        "cache_create_usd_per_million": "1",
        "revision": "fixture",
    }
    route["model_bounds"] = {
        "max_input_tokens": 10,
        "max_output_tokens": 10,
        "default_max_output_tokens": 10,
        "provider_contract": "synthetic-http-v1",
    }
    raw["limits"] = {
        "request_deadline_seconds": 5,
        "stream_content_idle_seconds": 1,
        "partial_frame_seconds": 0.5,
    }
    if public_provider:
        raw["credentials"][0]["provider"] = "anthropic"
        route.update(
            provider="anthropic",
            private_network=False,
            ingress_protocols=["anthropic-messages"],
            native_protocols=["anthropic-messages"],
        )
        route["capabilities"] = {"anthropic-messages": route["capabilities"]["anthropic-messages"]}
    if configure:
        configure(raw)
    if ws_server is not None:
        ws_origin = f"https://llm.internal.example:{ws_server.socket.getsockname()[1]}"
        raw["credentials"][0]["allowed_origins"].append(ws_origin)
        route["upstream_origin"] = ws_origin
        route["capabilities"]["openai-responses"]["websocket"] = {"features": ["text", "tools"]}
    path = tmp_path / "http-runtime.json"
    path.write_text(json.dumps(raw))
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper()
        in {"PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP"}
    }
    env.update(
        HEADROOM_GATEWAY_CLIENT_TOKEN="client-secret",
        INTERNAL_LLM_API_KEY="fixture-key",
        HEADROOM_REQUIRE_RUST_CORE="false",
        HEADROOM_SKIP_UPDATE_CHECK="1",
        HEADROOM_SKIP_UPSTREAM_CHECK="1",
        PYTHONIOENCODING="utf-8",
    )
    private_home = tmp_path / "home"
    private_home.mkdir()
    env.update(HOME=str(private_home), USERPROFILE=str(private_home))
    env["HEADROOM_GATEWAY_CLIENT_TOKEN_B"] = "client-b"
    env["GATEWAY_TEST_RUNTIME_HTTP"] = "1"
    env["GATEWAY_TEST_RESERVE_BARRIER"] = str(reserve_barrier)
    env["GATEWAY_TEST_ACQUIRE_BARRIER"] = "1" if acquire_barrier else "0"
    env["OPERATOR_TOKEN"] = "operator-secret"
    if public_provider:
        env["GATEWAY_TEST_UPSTREAM_PORT"] = str(server.server_port)
    if fail_connect:
        env["GATEWAY_TEST_CONNECT_FAIL_PORT"] = str(_unused_loopback_port())
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.unified_gateway.process.runtime_server", str(path)],
        cwd=Path(__file__).parents[3],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    ready = threading.Event()
    output = []

    def logs():
        for line in process.stdout:
            output.append(line)
            if "Uvicorn running on" in line:
                ready.set()
        ready.set()

    reader = threading.Thread(target=logs, daemon=True)
    reader.start()
    client = httpx.Client(
        base_url=f"http://127.0.0.1:{raw['runtime']['port']}",
        headers={"authorization": "Bearer client-secret"},
        timeout=12,
    )
    client.gateway_test_process = process
    try:
        assert ready.wait(30), "".join(output)
        assert any("Uvicorn running on" in line for line in output), "".join(output)
        assert process.poll() is None, "".join(output)
        readiness = client.get("/readyz").json()
        assert readiness["service"] == "headroom" and readiness["profile"] == "gateway"
        assert readiness["ready"] is True and readiness["config_digest"] == gateway_config_digest(
            raw
        )
        yield client, calls, output, raw
    finally:
        client.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        reader.join(timeout=3)
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        if ws_server is not None:
            ws_server.shutdown()
            ws_thread.join(timeout=3)
        (tmp_path / "http-process.log").write_text("".join(output), encoding="utf-8")


def json_response(handler, body, status=200):
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.send_header("connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


def stream_headers(handler):
    handler.send_response(200)
    handler.send_header("content-type", "text/event-stream")
    handler.send_header("connection", "close")
    handler.end_headers()


def frame(handler, data):
    handler.wfile.write(data)
    handler.wfile.flush()
