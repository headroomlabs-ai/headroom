"""Real proxy/compression/companion/sidebar E2E; local provider stub, no Claude engine.

Run in an environment with Headroom[proxy], this companion, pytest and Node.
Unlike run-http-smoke.py, no RequestLog records or Headroom modules are mocked.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
SID = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
opener = build_opener(ProxyHandler({}))
forwarded = []


class Provider(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        forwarded.append({"body": body, "headers": dict(self.headers)})
        data = json.dumps(
            {
                "id": "msg_local",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "LOCAL_PROVIDER_OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 10},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


def get(base, path):
    with opener.open(base + path, timeout=3) as response:
        return json.load(response)


def send(base, sid, text):
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "stream": False,
        "messages": [
            {"role": "user", "content": "Read the tool output and summarize."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_e2e",
                        "name": "Bash",
                        "input": {"command": "list-service-health"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool_e2e", "content": text}],
            },
        ],
        "tools": [
            {
                "name": "Bash",
                "description": "Run a local command",
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        ],
    }
    request = Request(
        base + "/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "local-e2e-key",
            "X-Headroom-Mod-Session": sid,
        },
    )
    with opener.open(request, timeout=90) as response:
        assert json.load(response)["content"][0]["text"] == "LOCAL_PROVIDER_OK"


def main():
    capture = "--no-capture" not in sys.argv[1:]
    response_cache = "--response-cache" in sys.argv[1:]
    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HEADROOM_DISABLE_KOMPRESS": "1"}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT.parents[1]), str(ROOT / "companion" / "src"), env.get("PYTHONPATH", "")]
    )
    with tempfile.TemporaryDirectory(prefix="headroom-sidebar-e2e-") as temp:
        log_path = Path(temp) / "proxy.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "headroom.cli",
                    "proxy",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--mode",
                    "token",
                    "--compressor",
                    "smart_crusher",
                    "--disable-kompress-anthropic",
                    "--no-subscription-tracking",
                    *([] if response_cache else ["--no-cache"]),
                    *(["--log-messages"] if capture else []),
                    "--proxy-extension",
                    "claude_mod",
                    "--anthropic-api-url",
                    f"http://127.0.0.1:{provider.server_port}",
                ],
                env=env,
                cwd=temp,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Proxy exited before startup")
                    try:
                        health = get(base, "/headroom-mod/v1/health")
                        assert health["log_full_messages"] is capture
                        break
                    except (URLError, TimeoutError):
                        time.sleep(0.1)
                else:
                    raise RuntimeError("Proxy startup timed out")
                rows = [
                    {
                        "id": i,
                        "service": "orders",
                        "status": "healthy",
                        "latency_ms": i % 7,
                        "region": "us-east-1",
                    }
                    for i in range(600)
                ]
                send(base, SID, json.dumps(rows, indent=2))
                send(base, OTHER, "OTHER_SESSION_SECRET")
                summary = get(base, f"/headroom-mod/v1/sessions/{SID}")
                assert summary["totals"]["requests"] == 1, summary
                assert summary["totals"]["before"] > summary["totals"]["after"], summary
                assert summary["totals"]["saved"] > 0, summary
                assert summary["totals"]["failed_requests"] == 0, summary
                request_id = summary["latest"]["request_id"]
                other = get(base, f"/headroom-mod/v1/sessions/{OTHER}")
                assert other["totals"]["requests"] == 1, other
                try:
                    get(base, f"/headroom-mod/v1/sessions/{OTHER}/requests/{request_id}")
                except HTTPError as exc:
                    assert exc.code == 404, exc
                else:
                    raise AssertionError("Cross-session detail leaked")
                assert len(forwarded) == 2, forwarded
                original = json.dumps(rows, indent=2)
                actual = forwarded[0]["body"]["messages"][2]["content"][0]["content"]
                assert len(actual) < len(original), (
                    "Provider did not receive compressed tool output"
                )
                assert not any(
                    k.lower().startswith("x-headroom-") for k in forwarded[0]["headers"]
                ), "Internal tags reached provider"
                subprocess.run(
                    [
                        "node",
                        str(ROOT / "scripts" / "proxy-e2e.mjs"),
                        base,
                        request_id,
                        str(summary["totals"]["saved"]),
                        str(capture).lower(),
                    ],
                    check=True,
                    timeout=30,
                )

                # Exercise the real bypass after the inspector checks above.
                def control(suffix, body):
                    request = Request(
                        base + f"/headroom-mod/v1/sessions/{SID}/" + suffix,
                        data=json.dumps(body).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with opener.open(request, timeout=10) as response:
                        return json.load(response)

                if response_cache:
                    count_before = len(forwarded)
                    send(base, SID, original)
                    assert len(forwarded) == count_before, "Expected a populated response-cache hit"
                assert control("compression", {"enabled": False})["compression_enabled"] is False
                count_before = len(forwarded)
                send(base, SID, original)
                assert len(forwarded) == count_before + 1, (
                    "Paused request replayed a cached response"
                )
                assert forwarded[-1]["body"]["messages"][2]["content"][0]["content"] == original
                assert not any(
                    k.lower().startswith("x-headroom-") for k in forwarded[-1]["headers"]
                )
                send(base, OTHER, original + "\nOTHER")
                assert len(forwarded[-1]["body"]["messages"][2]["content"][0]["content"]) < len(
                    original
                )
                assert control("compression", {"enabled": True})["compression_enabled"] is True
                send(base, SID, original + "\nRESUMED")
                assert len(forwarded[-1]["body"]["messages"][2]["content"][0]["content"]) < len(
                    original
                )
                control("reset", {})
                reset_summary = get(base, f"/headroom-mod/v1/sessions/{SID}?window=1h")
                assert reset_summary["totals"]["requests"] == 0, reset_summary
                assert (
                    get(base, f"/headroom-mod/v1/sessions/{SID}/requests/{request_id}")[
                        "request_id"
                    ]
                    == request_id
                )
                print(
                    json.dumps(
                        {
                            "result": "PASS",
                            "test": "real proxy compression to companion to shipped sidebar hooks",
                            "saved": summary["totals"]["saved"],
                            "requests": len(forwarded),
                            "controls": "pause, resume, cross-session isolation, reset",
                            "response_cache": response_cache,
                            "native_claude": False,
                            "capture": capture,
                        }
                    )
                )
            except Exception:
                log.flush()
                print(log_path.read_text(encoding="utf-8"), file=sys.stderr)
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                provider.shutdown()
                provider.server_close()
                thread.join(timeout=3)


if __name__ == "__main__":
    main()
