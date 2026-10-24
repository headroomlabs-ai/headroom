"""Cover Headroom's binding-kwargs merge with real LangChain and loopback HTTP."""

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

MODEL = "headroom-binding-loopback"


def _tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Look up the weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def _choice(name):
    return {"type": "function", "function": {"name": name}}


@pytest.fixture(autouse=True)
def offline_headroom(monkeypatch, tmp_path):
    """Disable automatic egress and isolate normal product state/config."""
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    # Track these too: apply_offline_env() sets them lazily during product use.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("HEADROOM_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("HEADROOM_MODEL_LIMITS", json.dumps({"context_limits": {MODEL: 128000}}))
    for name in ("LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
        monkeypatch.setenv(name, "false")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OPENAI_PROXY", "")


@pytest.fixture
def openai_loopback():
    """Run only a local HTTP fixture, without hosted calls or real credentials."""
    pytest.importorskip("langchain_openai")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            message = {"role": "assistant", "content": "Loopback reply"}
            finish_reason = "stop"
            response = json.dumps(
                {
                    "id": "chatcmpl-loopback",
                    "object": "chat.completion",
                    "created": 1,
                    "model": MODEL,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 41, "completion_tokens": 9, "total_tokens": 50},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "Loopback HTTP server did not stop"


@pytest.mark.parametrize("binding", ["bind_tools", "nested"])
def test_binding_reaches_openai_http(openai_loopback, binding):
    import httpx
    from langchain_core.runnables import RunnableBinding
    from langchain_openai import ChatOpenAI
    from langsmith import tracing_context

    from headroom.integrations import HeadroomChatModel

    base_url, requests = openai_loopback
    with tracing_context(enabled=False), httpx.Client(trust_env=False) as http_client:
        original = ChatOpenAI(
            model=MODEL,
            api_key="loopback-dummy-key",
            base_url=base_url,
            max_retries=0,
            timeout=5,
            use_responses_api=False,
            http_client=http_client,
        )
        kwargs = {}
        expected_name = "inner_weather"
        if binding == "bind_tools":
            wrapped = HeadroomChatModel(original, auto_detect_provider=False).bind_tools(
                [_tool(expected_name)], tool_choice=expected_name
            )
        else:
            inner = original.bind_tools(
                [_tool("inner_weather")],
                tool_choice="inner_weather",
                temperature=0.1,
                top_p=0.8,
            )
            outer = RunnableBinding(
                bound=inner,
                kwargs={
                    "tools": [_tool("outer_weather")],
                    "tool_choice": _choice("outer_weather"),
                    "temperature": 0.2,
                },
            )
            wrapped = HeadroomChatModel(outer, auto_detect_provider=False)
            expected_name = "caller_weather"
            kwargs = {
                "tools": [_tool(expected_name)],
                "tool_choice": _choice(expected_name),
                "temperature": 0.3,
            }
        wrapped.cache = False
        wrapped.invoke("What is the weather in Paris?", **kwargs)

    _, body = requests[0]
    assert body["tools"] == [_tool(expected_name)]
    assert body["tool_choice"] == _choice(expected_name)
    if binding == "nested":
        assert body["temperature"] == 0.3
        assert body["top_p"] == 0.8


def test_unwrap_binding_matches_real_runnable_precedence():
    pytest.importorskip("langchain_core")
    from langchain_core.runnables import RunnableBinding, RunnableLambda
    from langsmith import tracing_context

    from headroom.integrations import HeadroomChatModel

    def affine(value, scale=1, offset=0):
        return value * scale + offset

    target = RunnableLambda(affine)
    inner = RunnableBinding(bound=target, kwargs={"scale": 2, "offset": 3})
    outer = RunnableBinding(bound=inner, kwargs={"scale": 5})
    unwrapped, kwargs = HeadroomChatModel._unwrap_binding(outer)
    plain, plain_kwargs = HeadroomChatModel._unwrap_binding(target)
    assert unwrapped is target

    with tracing_context(enabled=False):
        assert outer.invoke(4) == unwrapped.invoke(4, **kwargs) == 23
        caller_kwargs = {**kwargs, "scale": 7}
        assert outer.invoke(4, scale=7) == unwrapped.invoke(4, **caller_kwargs) == 31
        assert asyncio.run(outer.ainvoke(4)) == asyncio.run(unwrapped.ainvoke(4, **kwargs)) == 23
        assert asyncio.run(outer.ainvoke(4, scale=7)) == 31
        assert asyncio.run(unwrapped.ainvoke(4, **caller_kwargs)) == 31
        assert plain.invoke(4, **plain_kwargs) == 4
        # Caller overrides must not mutate either reusable binding.
        assert inner.invoke(4) == 11
        assert outer.invoke(4) == 23
