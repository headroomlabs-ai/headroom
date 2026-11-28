"""Where ``thinking`` from an OpenAI-format request goes on the litellm call."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, patch

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.litellm import LiteLLMBackend  # noqa: E402  (must follow importorskip)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("provider", "model", "native"),
    [
        ("anthropic", "claude-sonnet-5-5", True),
        ("openai", "gpt-4o", False),
        # OpenAI-compatible gateway serving Claude: the adapter has no `thinking`.
        ("openai", "claude-sonnet-5-5", False),
    ],
)
def test_thinking_is_a_kwarg_only_where_the_adapter_supports_it(stream, provider, model, native):
    thinking = {"type": "between_tools"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": thinking,
        "vendor_field": 1,
    }

    async def call(backend: LiteLLMBackend) -> None:
        if not stream:
            await backend.send_openai_message(body, {})
            return
        async for _ in backend.stream_openai_message(body, {}):
            pass

    with patch(
        "headroom.backends.litellm.acompletion",
        new_callable=AsyncMock,
        side_effect=RuntimeError("stop before the network"),
    ) as mock_acomp:
        asyncio.run(call(LiteLLMBackend(provider=provider)))

    sent = mock_acomp.call_args.kwargs
    if native:
        assert sent["thinking"] == thinking
        assert sent["extra_body"] == {"vendor_field": 1}
    else:
        assert "thinking" not in sent
        assert sent["extra_body"] == {"thinking": thinking, "vendor_field": 1}


@pytest.mark.parametrize("stream", [False, True])
def test_openai_gateway_serving_claude_receives_thinking(monkeypatch, stream):
    """Upstream JSON of a ``litellm-openai`` backend still carries ``thinking``."""
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(400)  # the request body is all that matters here
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    monkeypatch.setenv("OPENAI_API_BASE", url)
    monkeypatch.setenv("OPENAI_BASE_URL", url)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    thinking = {"type": "between_tools"}
    body = {
        "model": "claude-sonnet-5-5",
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": thinking,
    }

    async def call() -> None:
        backend = LiteLLMBackend(provider="openai")
        if not stream:
            await backend.send_openai_message(body, {})
            return
        async for _ in backend.stream_openai_message(body, {}):
            pass

    try:
        asyncio.run(call())
    finally:
        server.shutdown()
        server.server_close()

    assert requests, "nothing reached the gateway"
    assert requests[0]["model"] == "claude-sonnet-5-5"
    assert requests[0]["thinking"] == thinking
