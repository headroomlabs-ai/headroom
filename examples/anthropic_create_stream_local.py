"""Actual SDK + loopback HTTP proof; no remote provider or model calls."""

import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    scratch = tempfile.TemporaryDirectory(prefix="headroom-client-proof-")
    for key in ["HEADROOM_WORKSPACE_DIR", "HEADROOM_CONFIG_DIR", "XDG_CACHE_HOME"]:
        os.environ[key] = str(Path(scratch.name) / key.lower())
    os.environ.update(HEADROOM_OFFLINE="1", HEADROOM_BEACON="off", DO_NOT_TRACK="1")

    import httpx
    from anthropic import Anthropic

    from headroom import AnthropicProvider, HeadroomClient
    from headroom.config import HeadroomConfig

    requests = []
    observed_styles = []

    class Observer:
        def on_pipeline_event(self, event):
            if "api_style" in event.metadata:
                observed_styles.append(event.metadata["api_style"])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(data)
            if data.get("stream"):
                events = [
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": "msg_fixture",
                                "type": "message",
                                "role": "assistant",
                                "model": data["model"],
                                "content": [],
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {"input_tokens": 1, "output_tokens": 0},
                            },
                        },
                    ),
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": "local fixture"},
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    (
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"output_tokens": 2},
                        },
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ]
                payload = "".join(
                    f"event: {name}\ndata: {json.dumps(event)}\n\n" for name, event in events
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            payload = json.dumps(
                {
                    "id": "msg_fixture",
                    "type": "message",
                    "role": "assistant",
                    "model": data["model"],
                    "content": [{"type": "text", "text": "local fixture"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    responses = []
    http = httpx.Client(event_hooks={"response": [responses.append]})
    original = Anthropic(
        http_client=http,
        api_key="local-fixture",
        base_url=f"http://127.0.0.1:{server.server_port}",
        max_retries=0,
    )
    wrapped = HeadroomClient(
        original,
        AnthropicProvider(warn=False),
        store_url=f"sqlite:///{scratch.name}/metrics.db",
        enable_cache_optimizer=False,
        config=HeadroomConfig(
            store_url=f"sqlite:///{scratch.name}/metrics.db",
            discover_pipeline_extensions=False,
            pipeline_extensions=[Observer()],
        ),
    )
    kwargs = {
        "model": "claude-fixture",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "stream": False,
    }
    try:
        native = original.messages.create(**kwargs)
        print(
            json.dumps(
                {"case": "native", "text": native.content[0].text, "requests": len(requests)}
            )
        )
        try:
            result = wrapped.messages.create(**kwargs)
            print(
                json.dumps(
                    {"case": "wrapped", "text": result.content[0].text, "requests": len(requests)}
                )
            )
            kwargs["stream"] = True
            native_stream = original.messages.create(**kwargs)
            native_types = [event.type for event in native_stream]
            native_stream.close()
            wrapped_stream = wrapped.messages.create(**kwargs)
            wrapped_types = [event.type for event in wrapped_stream]
            wrapped_stream.close()
            assert native_types == wrapped_types
            print(
                json.dumps(
                    {
                        "case": "create_stream",
                        "native_types": native_types,
                        "wrapped_types": wrapped_types,
                        "native_closed": native_stream.response.is_closed,
                        "requests": len(requests),
                    }
                )
            )
            del kwargs["stream"]
            with wrapped.messages.stream(**kwargs) as manager:
                text = "".join(manager.text_stream)
            assert text == "local fixture"
            assert manager.response.is_closed
            assert all(response.is_closed for response in responses)
            assert observed_styles and set(observed_styles) == {"anthropic"}
            print(
                json.dumps(
                    {
                        "case": "resources",
                        "http_responses": len(responses),
                        "all_closed": all(response.is_closed for response in responses),
                    }
                )
            )
            print(
                json.dumps(
                    {
                        "case": "stream_manager",
                        "text": text,
                        "response_closed": manager.response.is_closed,
                        "requests": len(requests),
                    }
                )
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "case": "wrapped",
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "requests": len(requests),
                    }
                )
            )
            raise
    finally:
        wrapped.close()
        original.close()
        server.shutdown()
        server.server_close()
        scratch.cleanup()


if __name__ == "__main__":
    main()
