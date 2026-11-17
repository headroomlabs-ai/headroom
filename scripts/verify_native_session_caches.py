"""Exercise native session caches against a credentialless loopback upstream.

Build: cargo build -p headroom-proxy --locked
Run: python scripts/verify_native_session_caches.py --proxy target/debug/headroom-proxy
On Windows append .exe. A custom CARGO_TARGET_DIR requires its binary path.
Only Python's standard library is required. Logs are temporary unless --log is set.
"""

import argparse
import concurrent.futures
import http.server
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", required=True, type=Path)
    parser.add_argument("--log", type=Path, help="preserve native proxy logs here")
    args = parser.parse_args()
    binary = args.proxy.resolve(strict=True)
    captured = []
    capture_lock = threading.Lock()

    class Upstream(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            with capture_lock:
                captured.append((json.loads(body), self.headers.get("anthropic-beta", "")))
            reply = json.dumps(
                {
                    "id": "local",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": "local stub"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *unused):
            pass

    with tempfile.TemporaryDirectory(prefix="headroom-cache-proof-") as temporary:
        log_path = args.log.resolve() if args.log else Path(temporary) / "proxy.log"
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        # The native CLI does not publish its port when binding :0. Reserve a
        # candidate briefly; readiness detects exit if another process wins it.
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith(
                (
                    "HEADROOM_",
                    "AWS_",
                    "ANTHROPIC_",
                    "OPENAI_",
                    "GOOGLE_",
                    "AZURE_",
                )
            )
        }
        env.update(
            AWS_EC2_METADATA_DISABLED="true",
            NO_PROXY="127.0.0.1,localhost",
            HEADROOM_OFFLINE="1",
            HF_HUB_OFFLINE="1",
            HEADROOM_LIVE_ZONE_DISABLE_ARMS="source_code,plain_text",
        )
        command = [
            str(binary),
            "--listen",
            f"127.0.0.1:{port}",
            "--upstream",
            f"http://127.0.0.1:{server.server_port}",
            "--compression",
            "--compression-mode",
            "off",
            "--stats",
            "false",
            "--log-level",
            "info",
        ]
        # Explicitly bypass caller HTTP proxy settings for all proof requests.
        client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        process = None
        try:
            with log_path.open("w", encoding="utf-8", newline="\n") as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    cwd=temporary,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"proxy exited: {process.returncode}")
                    try:
                        with client.open(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                            response.read()
                        break
                    except OSError:
                        time.sleep(0.2)
                else:
                    raise RuntimeError("proxy readiness timeout")

                def request(system, token="", session="same"):
                    payload = {
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 16,
                        "system": system,
                        "messages": [{"role": "user", "content": "hello local"}],
                    }
                    headers = {"content-type": "application/json", "x-headroom-session-id": session}
                    if token:
                        headers["anthropic-beta"] = token
                    outgoing = urllib.request.Request(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=json.dumps(payload).encode(),
                        headers=headers,
                    )
                    with client.open(outgoing, timeout=30) as response:
                        if (
                            response.status != 200
                            or json.load(response)["content"][0]["text"] != "local stub"
                        ):
                            raise RuntimeError("unexpected forwarded response")

                request("first system", "beta-a")
                request("changed system", "beta-b")
                request("changed system")
                if captured[1][1] != "beta-a,beta-b" or captured[2][1] != "beta-a,beta-b":
                    raise RuntimeError(f"sequential beta union lost: {captured}")
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    list(
                        pool.map(
                            lambda i: request("parallel system", f"token-{i}", "parallel"), range(8)
                        )
                    )
                request("parallel system", session="parallel")
                if set(captured[-1][1].split(",")) != {f"token-{i}" for i in range(8)}:
                    raise RuntimeError(f"concurrent beta union lost: {captured[-1]}")
                if len(captured) != 12:
                    raise RuntimeError(f"expected 12 upstream requests, got {len(captured)}")
                log.flush()
                events = log_path.read_text(encoding="utf-8")
                if (
                    "cache_drift_first_request" not in events
                    or "cache_drift_observed" not in events
                ):
                    raise RuntimeError("expected first-seen and changed-system drift events")
                print(
                    json.dumps(
                        {
                            "passed": True,
                            "upstream_requests": len(captured),
                            "sequential_beta_union": captured[2][1],
                            "parallel_beta_tokens": sorted(captured[-1][1].split(",")),
                            "drift_first_seen_and_changed": True,
                        }
                    )
                )
        except Exception:
            if log_path.exists():
                print(log_path.read_text(encoding="utf-8"))
            raise
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    main()
