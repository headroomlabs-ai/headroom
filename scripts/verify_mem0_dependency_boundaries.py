"""Credential-free checks for the locked graph and telemetry client boundaries."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from urllib.parse import quote

os.environ["MEM0_TELEMETRY"] = "false"

from langgraph_sdk import get_sync_client
from mem0.memory.telemetry import AnonymousTelemetry
from posthog import Posthog

requests = []
events = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        requests.append(self.path)
        payload = (
            {"nodes": [], "edges": []}
            if self.path.endswith("/graph")
            else {"values": {"result": "fixture"}, "next": [], "tasks": []}
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_POST(self):
        data = self.rfile.read(int(self.headers["Content-Length"]))
        events.append(json.loads(data))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":1}')


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base_url = f"http://127.0.0.1:{server.server_port}"
try:
    thread_id = "tenant/../other?query#fragment"
    assistant_id = "assistant/../other?query#fragment"
    with get_sync_client(url=base_url, api_key="local-fixture-only") as client:
        stream = client.threads.stream(thread_id=thread_id, assistant_id=assistant_id)
        tree = stream.agent.get_tree()
        output = stream.output
    assert requests == [
        f"/assistants/{quote(assistant_id, safe='')}/graph",
        f"/threads/{quote(thread_id, safe='')}/state",
    ], requests
    assert tree == {"nodes": [], "edges": []}
    assert output == {"result": "fixture"}
    print(f"langgraph-sdk={version('langgraph-sdk')}: identifiers stay in one encoded path segment")

    telemetry = AnonymousTelemetry()
    assert telemetry.posthog is None
    telemetry.capture_event("headroom.boundary.proof", {"fixture": True})
    assert not events
    print("MEM0_TELEMETRY=false: no client/event created")

    posthog = Posthog(
        project_api_key="phc_local_fixture_only",
        host=base_url,
        sync_mode=True,
        feature_flags_request_timeout_seconds=0.5,
        enable_exception_autocapture=False,
    )
    try:
        posthog.capture(
            "headroom.boundary.proof", distinct_id="local-fixture", properties={"fixture": True}
        )
    finally:
        posthog.shutdown()
    assert len(events) == 1, events
    assert events[0]["batch"][0]["event"] == "headroom.boundary.proof", events
    print(f"posthog={version('posthog')}: capture/shutdown sent one event to loopback only")
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
