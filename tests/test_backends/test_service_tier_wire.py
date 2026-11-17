"""Exercise real SDK serialization against a local OpenAI-compatible HTTP endpoint."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tests._dotenv import importorskip_no_env_leak


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk", ["litellm", "anyllm"])
@pytest.mark.parametrize("wire_format", ["anthropic", "openai"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_sdk_puts_service_tier_on_the_provider_request(
    sdk, wire_format, streaming, monkeypatch
) -> None:
    sdk_module = importorskip_no_env_leak("litellm" if sdk == "litellm" else "any_llm")
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    received = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            response = {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            if body.get("stream"):
                response["object"] = "chat.completion.chunk"
                response["choices"][0]["delta"] = response["choices"][0].pop("message")
                payload = ("data: " + json.dumps(response) + "\n\ndata: [DONE]\n\n").encode()
                media = "text/event-stream"
            else:
                payload = json.dumps(response).encode()
                media = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Provider) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}/v1"
        if sdk == "litellm":
            from headroom.backends.litellm import LiteLLMBackend

            monkeypatch.setattr(sdk_module, "api_base", url)
            monkeypatch.setattr(sdk_module, "api_key", "local-fake-key")
            backend = LiteLLMBackend(provider="openai")
        else:
            from headroom.backends.anyllm import AnyLLMBackend

            backend = AnyLLMBackend(provider="openai", api_base=url, api_key="local-fake-key")
        body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "service_tier": "flex",
        }
        try:
            if streaming:
                method = (
                    backend.stream_message
                    if wire_format == "anthropic"
                    else backend.stream_openai_message
                )
                events = [event async for event in method(body, {})]
                assert events
                if wire_format == "anthropic":
                    assert all(event.event_type != "error" for event in events)
                    assert events[-1].event_type == "message_stop"
                    assert any(event.data.get("delta", {}).get("text") == "ok" for event in events)
                else:
                    chunks = [
                        json.loads(event.removeprefix("data: ").strip())
                        for event in events
                        if event.strip() != "data: [DONE]"
                    ]
                    assert all("error" not in chunk for chunk in chunks)
                    assert events[-1].strip() == "data: [DONE]"
                    assert any(
                        choice.get("delta", {}).get("content") == "ok"
                        for chunk in chunks
                        for choice in chunk.get("choices", [])
                    )
            else:
                method = (
                    backend.send_message
                    if wire_format == "anthropic"
                    else backend.send_openai_message
                )
                assert (await method(body, {})).status_code == 200
        finally:
            await backend.close()
            server.shutdown()
            worker.join(timeout=5)
    assert len(received) == 1
    assert received[0]["service_tier"] == "flex"
    assert received[0].get("stream", False) is streaming
    assert received[0]["messages"] == body["messages"]
