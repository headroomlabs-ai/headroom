"""Real loopback HTTP + seeded CCR. Run with headroom-ai[proxy] installed.

Exit 1 is a reproduced accounting regression; exit 2 is invalid setup/execution.
No paid provider, credentials, Lia imports or saved sessions are needed.
"""

import argparse
import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
ORIGINAL = json.dumps({"synthetic_answer": "seeded retrieval succeeded"})


def usage(n):
    return dict(zip(KEYS, (11 * n, 7 * n, 13 * n, 17 * n)))


def sse(message):
    start = {
        **message,
        "content": [],
        "stop_reason": None,
        "usage": {**message["usage"], "output_tokens": 0},
    }
    events = [("message_start", {"type": "message_start", "message": start})]
    for i, block in enumerate(message["content"]):
        initial = {**block, **({"input": {}} if block["type"] == "tool_use" else {"text": ""})}
        delta = (
            {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            if block["type"] == "tool_use"
            else {"type": "text_delta", "text": block["text"]}
        )
        events += [
            (
                "content_block_start",
                {"type": "content_block_start", "index": i, "content_block": initial},
            ),
            ("content_block_delta", {"type": "content_block_delta", "index": i, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": i}),
        ]
    events += [
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": message["usage"]["output_tokens"]},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return b"".join(
        f"event: {name}\ndata: {json.dumps(event)}\n\n".encode() for name, event in events
    )


def read_usage(raw, content_type):
    if "text/event-stream" not in content_type:
        body = json.loads(raw)
        assert body.get("type") == "message", body
        return body["usage"]
    counters = {}
    stopped = False
    for line in raw.decode().splitlines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        if event.get("type") == "message_start":
            counters.update(event["message"]["usage"])
        elif event.get("type") == "message_delta":
            counters.update(event.get("usage", {}))
        elif event.get("type") == "message_stop":
            stopped = True
        assert event.get("type") != "error", event
    assert stopped and all(k in counters for k in KEYS), "Incomplete SSE usage"
    return {k: counters[k] for k in KEYS}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Strip provider/service settings before Headroom is imported.
    for key in list(os.environ):
        if key.startswith(("ANTHROPIC_", "OPENAI_", "HEADROOM_", "LITELLM_")) or key.endswith(
            ("_API_KEY", "_TOKEN")
        ):
            os.environ.pop(key)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        LITELLM_LOCAL_MODEL_COST_MAP="true",
        HEADROOM_BEACON="off",
    )

    # Enforce loopback-only network even if an optional dependency tries to phone home.
    def network_guard(event, values):
        if event == "socket.connect":
            address = values[1]
            if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1"):
                raise RuntimeError("Non-loopback network is forbidden by this reproduction")

    sys.addaudithook(network_guard)
    with tempfile.TemporaryDirectory(prefix="headroom-repro-") as temporary:
        os.environ["HEADROOM_WORKSPACE_DIR"] = temporary
        import uvicorn

        from headroom.cache.backends import InMemoryBackend
        from headroom.cache.compression_store import get_compression_store, reset_compression_store
        from headroom.ccr.tool_injection import create_ccr_tool_definition
        from headroom.proxy.server import ProxyConfig, create_app

        reset_compression_store()
        store = get_compression_store(backend=InMemoryBackend())
        hash_key = store.store(original=ORIGINAL, compressed="{}", original_item_count=1)
        assert store.retrieve(hash_key).original_content == ORIGINAL

        class Provider(BaseHTTPRequestHandler):
            calls = []
            target_calls = 1

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                n = len(self.calls) + 1
                if n > 1:
                    results = [
                        b
                        for m in body["messages"]
                        if isinstance(m.get("content"), list)
                        for b in m["content"]
                        if b.get("type") == "tool_result"
                    ]
                    retrieved = json.loads(results[-1]["content"])
                    assert retrieved.get("original_content") == ORIGINAL, retrieved
                    assert "error" not in retrieved
                self.calls.append(
                    {
                        "usage": usage(n),
                        "stream": body.get("stream", False),
                        "retrieval_verified": n > 1,
                    }
                )
                content = (
                    [
                        {
                            "type": "tool_use",
                            "id": f"retrieve_{n}",
                            "name": "headroom_retrieve",
                            "input": {"hash": hash_key},
                        }
                    ]
                    if n < self.target_calls
                    else [{"type": "text", "text": "synthetic final"}]
                )
                message = {
                    "id": f"msg_{n}",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": content,
                    "stop_reason": "tool_use" if n < self.target_calls else "end_turn",
                    "stop_sequence": None,
                    "usage": usage(n),
                }
                raw = sse(message) if body.get("stream") else json.dumps(message).encode()
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "text/event-stream" if body.get("stream") else "application/json",
                )
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
        provider_thread.start()
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        config = ProxyConfig(
            anthropic_api_url=f"http://127.0.0.1:{provider.server_port}",
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=True,
            ccr_handle_responses=True,
            ccr_context_tracking=False,
            image_optimize=False,
            subscription_tracking_enabled=False,
            memory_enabled=False,
            memory_inject_tools=False,
            memory_inject_context=False,
            traffic_learning_enabled=False,
            read_lifecycle=False,
            disable_kompress=True,
            disable_kompress_fallback=True,
        )
        server = uvicorn.Server(
            uvicorn.Config(create_app(config), log_level="error", access_log=False)
        )
        thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
        thread.start()
        rows = []
        try:
            deadline = time.monotonic() + 30
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline, "Proxy startup failed"
                time.sleep(0.05)
            for streaming in (False, True):
                for target in (1, 2, 3):
                    Provider.calls = []
                    Provider.target_calls = target
                    body = {
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 100,
                        "stream": streaming,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Synthetic task; earlier output at <<ccr:{hash_key}>>",
                            }
                        ],
                        "tools": [create_ccr_tool_definition("anthropic")],
                    }
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=json.dumps(body).encode(),
                        headers={
                            "Content-Type": "application/json",
                            "x-api-key": "synthetic-local-only",
                            "anthropic-version": "2023-06-01",
                            "accept": "text/event-stream" if streaming else "application/json",
                        },
                    )
                    with urllib.request.urlopen(request, timeout=40) as response:
                        raw = response.read()
                        content_type = response.headers.get("Content-Type", "")
                        actual = read_usage(raw, content_type)
                    assert len(Provider.calls) == target, Provider.calls
                    assert all(c["retrieval_verified"] for c in Provider.calls[1:])
                    expected = {k: sum(c["usage"][k] for c in Provider.calls) for k in KEYS}
                    rows.append(
                        {
                            "streaming": streaming,
                            "calls": target,
                            "upstream": Provider.calls,
                            "expected": expected,
                            "actual": actual,
                            "passed": actual == expected,
                            "usage_events": [
                                json.loads(line[6:])
                                for line in raw.decode().splitlines()
                                if line.startswith("data: ")
                                and json.loads(line[6:]).get("type")
                                in ("message_start", "message_delta")
                            ]
                            if "text/event-stream" in content_type
                            else [],
                        }
                    )
            args.output.write_text(json.dumps(rows, indent=2) + "\n")
            for row in rows:
                print(
                    f"stream={row['streaming']} calls={row['calls']}: {'PASS' if row['passed'] else 'FAIL'} expected={row['expected']} actual={row['actual']}"
                )
            return 0 if all(r["passed"] for r in rows) else 1
        finally:
            server.should_exit = True
            thread.join(timeout=15)
            provider.shutdown()
            provider.server_close()
            provider_thread.join(timeout=5)
            listener.close()
            assert not thread.is_alive(), "Proxy did not shut down"


if __name__ == "__main__":
    try:
        code = run()
    except Exception:
        import traceback

        traceback.print_exc()
        code = 2
    raise SystemExit(code)
