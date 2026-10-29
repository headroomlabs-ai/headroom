"""Whole-history admission and compression passthrough behavior."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest


@pytest.mark.parametrize("raw", ["0", "-1", "not-int", "1.5"])
def test_invalid_request_body_ceiling_is_rejected(raw: str) -> None:
    from headroom.proxy.request_limit_policy import resolve_request_body_max_bytes

    with pytest.raises(ValueError):
        resolve_request_body_max_bytes(raw)


def test_startup_override_changes_http_admission_after_restart() -> None:
    program = textwrap.dedent(
        """
        import json
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient
        from headroom.proxy.request_body_limit import RequestBodyLimitMiddleware

        async def preprocess(request):
            await request.json()
            return JSONResponse({"preprocessed": True})

        app = Starlette(routes=[Route("/v1/compress", preprocess, methods=["POST"])])
        app.add_middleware(RequestBodyLimitMiddleware)
        with TestClient(app) as client:
            response = client.post(
                "/v1/compress",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "a" * 1500}]},
            )
            print(json.dumps({"status": response.status_code, "body": response.json()}))
        """
    )
    results = []
    for ceiling in (1024, 2048):
        process = subprocess.run(
            [sys.executable, "-c", program],
            env={
                **os.environ,
                "HEADROOM_REQUEST_BODY_MAX_BYTES": str(ceiling),
                "DO_NOT_TRACK": "1",
                "HEADROOM_TELEMETRY": "off",
            },
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        results.append(json.loads(process.stdout))

    assert results[0]["status"] == 413
    assert results[0]["body"]["error"]["code"] == "request_too_large"
    assert results[1] == {"status": 200, "body": {"preprocessed": True}}


@pytest.fixture
def admitted_proxy(monkeypatch, request):
    from fastapi.testclient import TestClient

    from headroom.proxy import request_limit_policy
    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setattr(request_limit_policy, "COMPRESSION_BODY_MAX_BYTES", 1024)
    backend = getattr(request, "param", "anthropic")
    sdk = pytest.importorskip("litellm") if backend != "anthropic" else None

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            url = urlsplit(self.path)
            path = url.path
            if path.endswith("/responses"):
                if "input" not in body or body.get("store") is not False:
                    self.send_error(400, "Unsupported Responses wire request")
                    return
                response = json.dumps(
                    {
                        "id": "resp_admission",
                        "object": "response",
                        "output": [],
                        "usage": {"input_tokens": 101, "output_tokens": 1},
                    }
                ).encode()
                content_type = "application/json"
            elif path.endswith("/chat/completions"):
                payload = {
                    "id": "chat_admission",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 101, "completion_tokens": 1, "total_tokens": 102},
                }
                if body.get("stream", False):
                    payload["object"] = "chat.completion.chunk"
                    payload["choices"] = [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                    usage = payload.pop("usage")
                    usage_chunk = {**payload, "choices": [], "usage": usage}
                    response = (
                        "".join(
                            "data: " + json.dumps(chunk) + "\n\n"
                            for chunk in (payload, usage_chunk)
                        )
                        + "data: [DONE]\n\n"
                    ).encode()
                    content_type = "text/event-stream"
                else:
                    response = json.dumps(payload).encode()
                    content_type = "application/json"
            elif path.endswith((":generateContent", ":streamGenerateContent")):
                query = parse_qs(url.query)
                if query.get("key") != ["synthetic-only"]:
                    self.send_error(401, "Gemini API key required")
                    return
                streaming = path.endswith(":streamGenerateContent")
                if streaming and query.get("alt") != ["sse"]:
                    self.send_error(400, "Streaming endpoint requires one alt=sse")
                    return
                payload = {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "ok"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 101,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 102,
                    },
                }
                if streaming:
                    response = ("data: " + json.dumps(payload) + "\n\n").encode()
                    content_type = "text/event-stream"
                else:
                    response = json.dumps(payload).encode()
                    content_type = "application/json"
            else:
                response = json.dumps(
                    {
                        "id": "msg_admission",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-6",
                        "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 101, "output_tokens": 1},
                    }
                ).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    if sdk is not None:
        monkeypatch.setattr(sdk, "api_base", f"http://127.0.0.1:{provider.server_port}/v1")
        monkeypatch.setattr(sdk, "api_key", "synthetic-only")
        monkeypatch.setenv("OPENAI_API_KEY", "synthetic-only")
    config = ProxyConfig(
        anthropic_api_url=f"http://127.0.0.1:{provider.server_port}",
        openai_api_url=f"http://127.0.0.1:{provider.server_port}",
        gemini_api_url=f"http://127.0.0.1:{provider.server_port}",
        bedrock_api_url=f"http://127.0.0.1:{provider.server_port}",
        vertex_api_url=f"http://127.0.0.1:{provider.server_port}",
        backend=backend,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        stateless=True,
        disable_kompress=True,
    )
    try:
        with TestClient(
            create_app(config),
            base_url="http://127.0.0.1",
            client=("127.0.0.1", 50000),
        ) as client:
            yield client
    finally:
        provider.shutdown()
        provider.server_close()
        thread.join()


def test_gzip_history_above_soft_threshold_stays_live_without_claiming_savings(admitted_proxy):
    raw = json.dumps(
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "history " * 3000}],
        }
    ).encode()
    encoded = gzip.compress(raw)
    assert len(encoded) < 1024 < len(raw)
    response = admitted_proxy.post(
        "/v1/messages",
        content=encoded,
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )
    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "ok"}]
    stats = admitted_proxy.get("/stats").json()
    assert stats["tokens"]["input_provider_reported"] == 101
    assert stats["tokens"]["output"] == 1
    assert stats["tokens"]["proxy_attempted_tokens"] == 0
    assert stats["tokens"]["saved"] == 0
    assert admitted_proxy.get("/livez").status_code == 200
    following = admitted_proxy.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "next turn"}],
        },
    )
    assert following.status_code == 200
    assert admitted_proxy.get("/stats").json()["requests"]["failed"] == 0


@pytest.mark.parametrize(
    "path",
    [
        "/v1/messages",
        "/v1/chat/completions",
        "/v1/responses",
        "/v1beta/models/gemini-2.5-flash:generateContent",
        "/model/anthropic.claude-v2/invoke",
    ],
)
def test_soft_passthrough_still_obeys_decompressed_hard_ceiling(admitted_proxy, monkeypatch, path):
    from headroom.proxy import helpers

    monkeypatch.setattr(helpers, "MAX_DECOMPRESSED_BODY_SIZE", 8192)
    monkeypatch.setattr(helpers, "MAX_REQUEST_BODY_SIZE", 8192)
    content = "history " * 3000
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": content}],
    }
    if path == "/v1/responses":
        body = {"model": "gpt-4o", "input": content}
    elif "/models/" in path:
        body = {"contents": [{"role": "user", "parts": [{"text": content}]}]}
    elif "/model/" in path:
        body["anthropic_version"] = "bedrock-2023-05-31"
    raw = json.dumps(body).encode()
    response = admitted_proxy.post(
        path,
        content=gzip.compress(raw),
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )
    assert response.status_code == 413
    assert admitted_proxy.get("/livez").status_code == 200


def test_large_copilot_chat_remains_a_chat_completions_request(admitted_proxy, monkeypatch):
    base = admitted_proxy.app.state.proxy.config.openai_api_url
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", base)
    response = admitted_proxy.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "history " * 3000}],
        },
        headers={"authorization": "Bearer tid_synthetic_client"},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"


def test_large_codex_request_receives_the_required_store_setting(admitted_proxy, monkeypatch):
    from headroom.providers.codex import endpoints

    monkeypatch.setattr(
        endpoints,
        "CHATGPT_BACKEND_API_URL",
        admitted_proxy.app.state.proxy.config.openai_api_url,
    )
    response = admitted_proxy.post(
        "/v1/responses",
        json={"model": "gpt-5-codex", "input": "history " * 3000},
        headers={
            "authorization": "Bearer synthetic_subscription",
            "chatgpt-account-id": "synthetic_account",
        },
    )
    assert response.status_code == 200
    assert response.json()["object"] == "response"


@pytest.mark.parametrize("admitted_proxy", ["litellm-openai"], indirect=True)
def test_large_vertex_backend_retains_route_forced_streaming(admitted_proxy):
    response = admitted_proxy.post(
        "/v1/projects/synthetic/locations/us-central1/publishers/anthropic/models/gpt-4o:streamRawPredict",
        json={
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "history " * 3000}],
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert events[-1]["type"] == "message_stop"


@pytest.mark.parametrize("streaming", [False, True])
def test_large_gemini_history_preserves_auth_and_streaming_contract(admitted_proxy, streaming):
    endpoint = "streamGenerateContent" if streaming else "generateContent"
    response = admitted_proxy.post(
        f"/v1beta/models/gemini-2.5-flash:{endpoint}?key=synthetic-only&alt=json",
        json={"contents": [{"role": "user", "parts": [{"text": "history " * 3000}]}]},
    )
    assert response.status_code == 200
    if streaming:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert (
            "".join(
                part["text"]
                for event in events
                for candidate in event.get("candidates", [])
                for part in candidate["content"]["parts"]
            )
            == "ok"
        )
        assert events[-1]["candidates"][0]["finishReason"] == "STOP"
    else:
        assert response.json()["candidates"][0]["content"]["parts"][0]["text"] == "ok"
    stats = admitted_proxy.get("/stats").json()
    assert stats["tokens"]["input_provider_reported"] == 101
    assert stats["tokens"]["output"] == 1
    assert stats["tokens"]["proxy_attempted_tokens"] == 0
    assert stats["tokens"]["saved"] == 0
    assert stats["requests"]["failed"] == 0


@pytest.mark.parametrize("admitted_proxy", ["litellm-openai"], indirect=True)
@pytest.mark.parametrize("streaming", [False, True])
def test_large_openai_backend_reports_usage_without_compression(admitted_proxy, streaming):
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "history " * 3000}],
        "stream": streaming,
    }
    if streaming:
        body["stream_options"] = {"include_usage": True}
    response = admitted_proxy.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    if streaming:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert (
            "".join(
                choice["delta"].get("content", "")
                for event in events
                for choice in event.get("choices", [])
            )
            == "ok"
        )
        assert response.text.endswith("data: [DONE]\n\n")
    else:
        assert response.json()["choices"][0]["message"]["content"] == "ok"
    stats = admitted_proxy.get("/stats").json()
    assert stats["tokens"]["input_provider_reported"] == 101
    assert stats["tokens"]["output"] == 1
    assert stats["tokens"]["proxy_attempted_tokens"] == 0
    assert stats["tokens"]["saved"] == 0
    assert stats["requests"]["failed"] == 0
