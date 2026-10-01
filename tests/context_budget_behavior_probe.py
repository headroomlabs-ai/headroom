"""Synthetic loopback proof for finalized context admission, runnable on pinned main."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import os
import platform
import socket
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@contextlib.contextmanager
def listener(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("loopback listener did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


def config(**kwargs):
    from headroom.proxy.models import ProxyConfig

    return ProxyConfig(
        optimize=False,
        image_optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        memory_enabled=False,
        subscription_tracking_enabled=False,
        offline=True,
        **kwargs,
    )


def response(content=None):
    return {
        "id": "msg_synthetic",
        "type": "message",
        "role": "assistant",
        "model": "step-router-v1",
        "content": content or [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def step_body():
    return {
        "model": "step-router-v1",
        "max_tokens": 4096,
        "system": "Synthetic context budget reproduction.",
        "tools": [
            {
                "name": "read_sample",
                "description": "Read synthetic text",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        "messages": [
            {"role": "user", "content": "Read the sample."},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "sample", "name": "read_sample", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "sample",
                        "content": "ordinary synthetic tool output " * 60000,
                    }
                ],
            },
        ],
    }


def counted(body):
    from headroom.tokenizers import get_tokenizer

    tk = get_tokenizer(body["model"])
    messages = body["messages"]
    if body.get("system") is not None:
        messages = [{"role": "system", "content": body["system"]}, *messages]
    return tk.count_messages(messages) + (
        tk.count_text(json.dumps(body["tools"], default=str)) if body.get("tools") else 0
    )


def run_initial(
    mode="reject", *, small=False, forwarding="byte_faithful", bypass=False, stream=False
):
    from headroom.proxy.body_forwarding import select_outbound_body
    from headroom.proxy.server import create_app

    body = step_body()
    if small:
        body["messages"][-1]["content"][0]["content"] = "small synthetic output"
    if stream:
        body["stream"] = True
    payload = json.dumps(body).encode()
    calls = []
    upstream = FastAPI()

    @upstream.post("/v1/messages")
    async def capture(request: Request):
        calls.append(await request.body())
        return (
            JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "input tokens exceed context limit",
                    },
                },
            )
            if not small
            else JSONResponse(response())
        )

    env = {
        "HEADROOM_CONTEXT_LIMIT_MODE": mode,
        "HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN": "12000",
        "HEADROOM_MODEL_LIMITS": json.dumps({"context_limits": {"step-router-v1": 262144}}),
        "HEADROOM_PROXY_PYTHON_FORWARDER_MODE": forwarding,
    }
    with patch.dict(os.environ, env), listener(upstream) as upstream_url:
        app = create_app(config(anthropic_api_url=upstream_url))
        selected = []
        real_select = select_outbound_body

        def inspect_select(**kwargs):
            item = real_select(**kwargs)
            selected.append(item.content)
            return item

        with (
            patch("headroom.proxy.body_forwarding.select_outbound_body", inspect_select),
            listener(app) as proxy_url,
        ):
            headers = {"x-api-key": "synthetic", "anthropic-version": "2023-06-01"}
            if bypass:
                headers["x-headroom-mode"] = "passthrough"
            result = httpx.post(
                proxy_url + "/v1/messages", content=payload, headers=headers, timeout=30
            )
        app.state.proxy._compression_executor.shutdown(wait=True)
    count = counted(body)
    data = result.json() if "json" in result.headers.get("content-type", "") else {}
    message = data.get("error", {}).get("message", "")
    if calls:
        assert selected[-1] == calls[-1]
    evidence = {
        "mode": mode,
        "status": result.status_code,
        "declared_limit": 262144,
        "reserve": 12000,
        "threshold": 250144,
        "count": count,
        "overage": count - 250144,
        "upstream_calls": len(calls),
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "selected_sha256": hashlib.sha256(selected[-1]).hexdigest() if selected else "none",
        "sent_sha256": hashlib.sha256(calls[-1]).hexdigest() if calls else "none",
        "message": message,
    }
    print("initial " + json.dumps(evidence))
    print(
        f"initial HTTP {result.status_code} declared_limit=262144 reserve=12000 threshold=250144 counted_tokens={count} overage={count - 250144} upstream_calls={len(calls)} selected_sha256={evidence['selected_sha256']} sent_sha256={evidence['sent_sha256']} message={message}"
    )
    return evidence


def run_continuation(
    owner="ccr",
    *,
    late=False,
    unavailable=False,
    grace_disabled=False,
    mode="reject",
    ordinary_failure=False,
):
    from headroom.cache.backends import InMemoryBackend
    from headroom.cache.compression_store import get_compression_store, reset_compression_store
    from headroom.ccr.tool_injection import create_ccr_tool_definition
    from headroom.proxy.server import create_app
    from headroom.proxy.turn_hooks import clear_turn_hooks, register_turn_hook
    from headroom.tokenizers import get_tokenizer

    reset_compression_store()
    store = get_compression_store(backend=InMemoryBackend())
    large = "synthetic continuation output " * 12000
    marker = store.store(
        original=json.dumps({"sample": large}), compressed="{}", original_item_count=1
    )
    body = {
        "model": "step-router-v1",
        "max_tokens": 128,
        "stream": True,
        "tools": [create_ccr_tool_definition("anthropic")],
        "messages": [{"role": "user", "content": f"Retrieve <<ccr:{marker}>>"}],
    }
    upstream_content = (
        [
            {
                "type": "tool_use",
                "id": "retrieve",
                "name": "headroom_retrieve",
                "input": {"hash": marker},
            }
        ]
        if owner == "ccr"
        else [{"type": "text", "text": "ordinary answer"}]
    )
    calls = []
    release = threading.Event()
    upstream = FastAPI()

    @upstream.post("/v1/messages")
    async def capture(request: Request):
        calls.append(await request.body())
        if late and len(calls) == 1:
            assert await asyncio.to_thread(release.wait, 10)
        return JSONResponse(response(upstream_content if len(calls) == 1 else None))

    env = {
        "HEADROOM_CONTEXT_LIMIT_MODE": mode,
        "HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN": "0",
        "HEADROOM_MODEL_LIMITS": json.dumps({"context_limits": {"step-router-v1": 6000}}),
    }
    hooks_attempts = []

    class Hook:
        name = "synthetic-continuation"
        stream_safe = True

        async def on_response(self, ctx, current, call_model):
            hooks_attempts.append(True)
            if ordinary_failure:
                raise RuntimeError("unrelated hook failure")
            return await call_model([*ctx.messages, {"role": "user", "content": large}])

    async def memory_noop(*args, **kwargs):
        return None

    async def memory_results(*args, **kwargs):
        if ordinary_failure:
            raise RuntimeError("unrelated memory failure")
        return [{"type": "tool_result", "tool_use_id": "memory", "content": large}]

    tokenizer = get_tokenizer("step-router-v1")
    real_count = tokenizer.count_messages

    def evaluation(messages):
        if unavailable and calls:
            raise LookupError("synthetic continuation estimator unavailable")
        return real_count(messages)

    try:
        with patch.dict(os.environ, env), listener(upstream) as upstream_url:
            app = create_app(
                config(
                    anthropic_api_url=upstream_url,
                    ccr_inject_tool=True,
                    ccr_handle_responses=True,
                    ccr_context_tracking=False,
                    buffered_ccr_grace_seconds=0 if grace_disabled else (0.02 if late else 5),
                )
            )
            proxy = app.state.proxy
            if owner == "hook":
                register_turn_hook(Hook())
                register_turn_hook(Hook())
            if owner == "memory":
                proxy.memory_handler = SimpleNamespace(
                    config=SimpleNamespace(
                        inject_context=False, inject_tools=False, project_root_override=""
                    ),
                    initialized=False,
                    backend=None,
                    has_memory_tool_calls=lambda *args: True,
                    handle_memory_tool_calls=memory_results,
                    ensure_initialized=memory_noop,
                    health_status=lambda: {"initialized": False, "backend": "synthetic"},
                )
            with (
                patch.object(tokenizer, "count_messages", evaluation),
                listener(app) as proxy_url,
            ):
                with httpx.stream(
                    "POST",
                    proxy_url + "/v1/messages",
                    json=body,
                    headers={"x-api-key": "synthetic", "anthropic-version": "2023-06-01"},
                    timeout=20,
                ) as result:
                    chunks = []
                    ping_seen = False
                    for chunk in result.iter_bytes():
                        chunks.append(chunk)
                        if b"ping" in b"".join(chunks):
                            ping_seen = True
                            release.set()
                    content = b"".join(chunks).decode()
                    status = result.status_code
                    content_type = result.headers.get("content-type", "")
            proxy._compression_executor.shutdown(wait=True)
        if status == 200 and "event: error" in content:
            error = next(
                json.loads(line[6:])["error"]
                for line in content.splitlines()
                if line.startswith("data: ") and '"error"' in line
            )
        elif status != 200:
            error = json.loads(content)["error"]
        else:
            error = {}
        evidence = {
            "owner": owner,
            "late": late,
            "unavailable": unavailable,
            "status": status,
            "content_type": content_type,
            "ping": ping_seen,
            "error_type": error.get("type", "none"),
            "message": error.get("message", ""),
            "EOF": True,
            "initial_calls": min(1, len(calls)),
            "refused_continuation_calls": max(0, len(calls) - 1),
            "hook_attempts": len(hooks_attempts),
            "ordinary_answer": "ordinary answer" in content,
        }
        print("continuation " + json.dumps(evidence))
        print(
            f"continuation owner={owner} HTTP {status} {content_type} ping={ping_seen} {error.get('type', 'none')} EOF initial_calls={min(1, len(calls))} refused_continuation_calls={max(0, len(calls) - 1)} message={error.get('message', '')}"
        )
        return evidence
    finally:
        release.set()
        clear_turn_hooks()
        reset_compression_store()


def main():
    print(f"environment Windows={platform.system()} Python={platform.python_version()}")
    reject = run_initial()
    assert reject["status"] == 400 and reject["upstream_calls"] == 0 and reject["overage"] > 0
    assert "tokens counted" in reject["message"]
    assert run_initial("observe")["upstream_calls"] == 1
    assert run_initial(small=True)["status"] == 200
    for owner in ("ccr", "memory", "hook"):
        for late in (False, True):
            for unavailable in (False, True):
                row = run_continuation(owner, late=late, unavailable=unavailable)
                assert row["status"] == (200 if late else (500 if unavailable else 400)), row
                assert row["error_type"] == (
                    "api_error" if unavailable else "invalid_request_error"
                ), row
                assert row["initial_calls"] == 1 and row["refused_continuation_calls"] == 0, row
                assert row["EOF"] and row["ping"] == late
    row = run_continuation("hook", unavailable=True, grace_disabled=True)
    assert row["status"] == 500 and not row["ping"]
    for stream in (False, True):
        for form in ("text", "tool", "redacted"):
            row = run_backend(stream=stream, form=form)
            assert row["prepared_count"] < row["threshold"] < row["view_count"]
            assert row["status"] == 400 and row["sdk_calls"] == 0
        row = run_backend(stream=stream, reject=False)
        assert row["status"] == 200 and row["sdk_calls"] == 1 and row["kwargs_unchanged"]
        row = run_backend(stream=stream, cross_vendor=True)
        assert row["prepared_count"] == row["view_count"] and not row["retained"]
    print("loopback assertions complete")


def run_backend(*, stream=False, cross_vendor=False, reject=True, form="text", beta=False):
    from fastapi.testclient import TestClient

    from headroom.backends import litellm as owner
    from headroom.proxy.handlers.anthropic import _context_budget_counting_messages
    from headroom.proxy.server import create_app
    from headroom.tokenizers import get_tokenizer

    target = "openai/gpt-4o" if cross_vendor else "bedrock/anthropic.claude-sonnet-4-6-v1:0"
    with patch.object(owner, "_fetch_bedrock_inference_profiles", return_value={}):
        backend = owner.LiteLLMBackend(
            provider="bedrock", region="us-east-1", profile_name="synthetic"
        )
    backend._model_overrides = {"source-model": target}
    blocks = [{"type": "thinking", "thinking": "budget sample " * 10000, "signature": "s" * 16000}]
    if form == "redacted":
        blocks.append({"type": "redacted_thinking", "data": "synthetic" * 1000})
    blocks.append(
        {"type": "tool_use", "id": "t", "name": "sample", "input": {}}
        if form == "tool"
        else {"type": "text", "text": "ok"}
    )
    body = {
        "model": "source-model",
        "max_tokens": 32,
        "stream": stream,
        "messages": [{"role": "assistant", "content": blocks}],
        "thinking": {"type": "enabled", "budget_tokens": 1000},
    }
    headers = {"x-api-key": "synthetic", "anthropic-version": "2023-06-01"}
    if beta:
        headers["anthropic-beta"] = "context-1m-2025-08-07"
    prepared = backend.prepare_message(body, headers, stream=stream)
    snapshot = copy.deepcopy(prepared)
    tk = get_tokenizer(target)
    prepared_count = tk.count_messages(prepared["messages"])
    view_count = tk.count_messages(_context_budget_counting_messages(prepared["messages"]))
    retained = [
        block for message in prepared["messages"] for block in message.get("thinking_blocks", [])
    ]
    assert view_count - prepared_count == tk._count_content_parts(retained)
    threshold = (
        (prepared_count + view_count) // 2 if reject and not cross_vendor else view_count + 100
    )
    calls = []
    preparations = []
    real_prepare = backend.prepare_message

    def capture_prepare(*args, **kwargs):
        value = real_prepare(*args, **kwargs)
        preparations.append(copy.deepcopy(value))
        return value

    async def sdk(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if stream:

            async def empty():
                if False:
                    yield None

            return empty()
        from litellm import ModelResponse

        return ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    env = {
        "HEADROOM_CONTEXT_LIMIT_MODE": "reject",
        "HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN": "0",
        "HEADROOM_MODEL_LIMITS": json.dumps({"context_limits": {target: threshold + 32}}),
    }
    with patch.dict(os.environ, env), patch.object(owner, "acompletion", sdk):
        app = create_app(config())
        app.state.proxy.anthropic_backend = backend
        backend.prepare_message = capture_prepare
        with TestClient(app) as client:
            result = client.post("/v1/messages", json=body, headers=headers)
    assert prepared == snapshot
    if calls:
        assert calls[0] == preparations[0] == snapshot
    evidence = {
        "mode": "streaming" if stream else "buffered",
        "form": form,
        "converter": "LiteLLMBackend._convert_messages_for_litellm",
        "registry": type(tk).__name__,
        "prepared_count": prepared_count,
        "view_count": view_count,
        "threshold": threshold,
        "status": result.status_code,
        "sdk_calls": len(calls),
        "prepare_calls": len(preparations),
        "kwargs_unchanged": prepared == snapshot,
        "retained_blocks_counted_once": True,
        "cross_vendor": cross_vendor,
        "retained": any(m.get("thinking_blocks") for m in prepared["messages"]),
    }
    print("backend " + json.dumps(evidence))
    print(
        f"backend mode={evidence['mode']} real_converter=LiteLLMBackend._convert_messages_for_litellm registry={type(tk).__name__} prepared_count={prepared_count} view_count={view_count} threshold={threshold} status={result.status_code} sdk_calls={len(calls)} prepare_calls={len(preparations)} kwargs_unchanged={prepared == snapshot} retained_blocks_counted_once=True cross_vendor={cross_vendor}"
    )
    return evidence


if __name__ == "__main__":
    main()
